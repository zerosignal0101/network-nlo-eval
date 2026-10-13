"""WP-M1 验收守卫：多槽需求贯通到 RWA 分配层 + 业务模型。

本文件覆盖 brief §5「测试：多槽分配、连续窗口、释放回收、退化等价、
阻断按槽区间」，并对判据 M-J9 / M-J10 的**行为**做独立断言（物理数值类
判据 M-J6/M-J11 由 ``results/*.json`` + tools/ 的 ``--check`` 把关，
这里不重算 SNR，避免造出第二套物理口径）。

场景与三个判据工具共用 ``tools/ms_alloc_scenarios.py``，故测试跑的
拓扑/栅格/业务序列与落盘证据完全同源。

跑测试：``PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python -m pytest -q``
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

RESULTS = REPO / "results"

SLOTS_PER_CHANNEL = 4


# ---------------------------------------------------------------------------
# 业务模型：n_slots / first_slot / wavelength 的不变式
# ---------------------------------------------------------------------------
def test_service_request_n_slots_default_is_one_and_validated():
    """n_slots 默认 1（遗留单信道语义），且拒绝非正整数。"""
    from network_nlo_eval.simulation.traffic import ServiceRequest

    kwargs = dict(
        service_id=0, source_id=0, destination_id=3,
        arrival_time=0.0, departure_time=1.0,
        bit_rate_gbps=100.0, snr_requirement_db=10.0, launch_power_w=1e-3,
    )
    assert ServiceRequest(**kwargs).n_slots == 1

    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ServiceRequest(**kwargs, n_slots=bad)


def test_slots_for_bit_rate_matches_flexgrid_slots_for_rate():
    """业务层的推导外壳必须与 flexgrid 的 slots_for_rate 同口径。"""
    from network_nlo_eval.core.flexgrid import slots_for_rate
    from network_nlo_eval.simulation.traffic import slots_for_bit_rate

    for bit_rate, modulation, expected_hint in (
        (100.0, "QPSK", None), (200.0, "QPSK", None), (400.0, "8QAM", None),
    ):
        slot_width = 12.5e9
        got = slots_for_bit_rate(bit_rate, slot_width_hz=slot_width, modulation=modulation)
        ref = slots_for_rate(bit_rate * 1e9, 2 if modulation == "QPSK" else 3, 0.0, slot_width)
        assert got == ref
        assert expected_hint is None or got == expected_hint

    # 200 Gbps / QPSK / 12.5 GHz 槽 = 200/2/12.5 = 8 槽
    assert slots_for_bit_rate(200.0, slot_width_hz=12.5e9, modulation="QPSK") == 8


def test_allocated_service_rejects_inconsistent_wavelength_and_first_slot():
    """wavelength 必须恒等于 first_slot // slots_per_channel。"""
    from dataclasses import asdict

    from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

    base = dict(
        service_id=0, source_id=0, destination_id=3,
        arrival_time=0.0, departure_time=1.0,
        bit_rate_gbps=100.0, snr_requirement_db=10.0, launch_power_w=1e-3,
    )
    request = ServiceRequest(**base, n_slots=2)
    # ServiceRequest 是 slots=True dataclass，没有 __dict__，必须用 asdict。
    fields = asdict(request)

    # first_slot=5, spc=4 -> 期望 wavelength=1；给 2 应报错
    with pytest.raises(ValueError):
        AllocatedService(**fields, path=[0, 1, 3], wavelength=2, first_slot=5,
                        slots_per_channel=SLOTS_PER_CHANNEL)

    ok = AllocatedService(**fields, path=[0, 1, 3], wavelength=1, first_slot=5,
                          slots_per_channel=SLOTS_PER_CHANNEL)
    assert ok.slot_interval == (5, 7)

    # slots_per_channel=None（遗留语义）时不做一致性校验
    legacy = AllocatedService(**fields, path=[0, 1, 3], wavelength=7, first_slot=28)
    assert legacy.wavelength == 7


# ---------------------------------------------------------------------------
# 窗口卷积：连续 n_slots 首次适配
# ---------------------------------------------------------------------------
def test_first_fit_contiguous_windows_requires_contiguous_free_run():
    """窗口必须**连续**且全空闲；跨空洞的窗口不得入选。"""
    import numpy as np

    from network_nlo_eval.rwa.allocators import first_fit_contiguous_windows

    # 8 槽，空闲 = [T,F,T,T,T,F,T,T]
    free = np.array([True, False, True, True, True, False, True, True])
    got = first_fit_contiguous_windows(free, n_slots=3, slots_per_channel=SLOTS_PER_CHANNEL)
    # 起点 2 -> [2,3,4] 全空闲；起点 5 -> [5,6,7] 中 5 被占。
    # 起点 0 -> [0,1,2] 中 1 被占。故只有 2。
    assert got == [2]


def test_first_fit_contiguous_windows_dedupes_same_anchor_channel():
    """同锚信道的多个起点只留第一个（QoT 只吃锚信道，重复验证是浪费）。"""
    import numpy as np

    from network_nlo_eval.rwa.allocators import first_fit_contiguous_windows

    free = np.ones(16, dtype=bool)
    got = first_fit_contiguous_windows(free, n_slots=1, slots_per_channel=SLOTS_PER_CHANNEL)
    # 16 个空槽但只有 4 个锚信道 -> 4 个候选，且都是信道对齐的起点
    assert got == [0, 4, 8, 12]


def test_first_fit_contiguous_windows_impossible_returns_empty():
    import numpy as np

    from network_nlo_eval.rwa.allocators import first_fit_contiguous_windows

    free = np.array([True, False, True, False])
    assert first_fit_contiguous_windows(free, 3, SLOTS_PER_CHANNEL) == []
    assert first_fit_contiguous_windows(np.zeros(4, dtype=bool), 5, SLOTS_PER_CHANNEL) == []


# ---------------------------------------------------------------------------
# 多槽分配：连续窗口真的被占到
# ---------------------------------------------------------------------------
def test_multislot_allocation_occupies_contiguous_window():
    """分配 n 槽 -> 该窗口逐槽占用，且 window 连续。"""
    import ms_alloc_scenarios as sc
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    topology, grid, gn, validator, path_cache, state = sc.build_stack()
    alloc = KSPFirstFitFlexAllocator(
        path_cache=path_cache, qot_validator=validator, spectrum_grid=grid,
        network_topology=topology, slots_per_channel=SLOTS_PER_CHANNEL,
    )
    plan = sc.ServicePlan(service_id=0, source_id=0, destination_id=3,
                          arrival_time=0.0, departure_time=1.0,
                          bit_rate_gbps=400.0, n_slots=3)
    ok, svc = alloc.allocate(sc.make_request(plan), state)
    assert ok, "3 槽请求应当被分配"
    assert svc.n_slots == 3
    state.allocate_service(svc)

    link = state.get_link_state(svc.path[0], svc.path[1])
    lo, hi = svc.slot_interval
    assert hi - lo == 3
    assert link.occupied_slots[lo:hi].all(), "槽窗口应逐槽占用"


def test_multislot_requests_get_channel_aligned_first_fit():
    """多槽请求按 first_slot 升序首尾排布（first-fit 语义）。"""
    import ms_alloc_scenarios as sc
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    topology, grid, gn, validator, path_cache, state = sc.build_stack()
    alloc = KSPFirstFitFlexAllocator(
        path_cache=path_cache, qot_validator=validator, spectrum_grid=grid,
        network_topology=topology, slots_per_channel=SLOTS_PER_CHANNEL,
    )
    first_slots = []
    for i, n_slots in enumerate((1, 2, 3)):
        plan = sc.ServicePlan(service_id=i, source_id=0, destination_id=3,
                              arrival_time=0.0, departure_time=1.0,
                              bit_rate_gbps=400.0, n_slots=n_slots)
        ok, svc = alloc.allocate(sc.make_request(plan), state)
        assert ok
        first_slots.append((svc.first_slot, svc.n_slots))
        state.allocate_service(svc)

    # 1 槽 -> [0,1)；2 槽 -> [4,6)；3 槽 -> [8,11)（信道对齐避开已占信道）
    assert first_slots == [(0, 1), (4, 2), (8, 3)]


# ---------------------------------------------------------------------------
# 释放回收
# ---------------------------------------------------------------------------
def test_release_reclaims_slot_window_and_allows_reuse():
    """释放后该窗口槽全空，且再分配复用同一 first_slot。"""
    import ms_alloc_scenarios as sc
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    topology, grid, gn, validator, path_cache, state = sc.build_stack()
    alloc = KSPFirstFitFlexAllocator(
        path_cache=path_cache, qot_validator=validator, spectrum_grid=grid,
        network_topology=topology, slots_per_channel=SLOTS_PER_CHANNEL,
    )
    plan = sc.ServicePlan(service_id=0, source_id=0, destination_id=3,
                          arrival_time=0.0, departure_time=1.0,
                          bit_rate_gbps=400.0, n_slots=3)
    ok, svc = alloc.allocate(sc.make_request(plan), state)
    assert ok
    state.allocate_service(svc)
    first_slot_before = svc.first_slot

    alloc.release(svc, state)

    link = state.get_link_state(svc.path[0], svc.path[1])
    lo, hi = svc.slot_interval
    assert not link.occupied_slots[lo:hi].any(), "释放后槽应被回收"

    ok2, svc2 = alloc.allocate(sc.make_request(plan), state)
    assert ok2 and svc2.first_slot == first_slot_before, "释放后应复用同一 first_slot"


def test_release_is_idempotent_and_does_not_clear_neighbours():
    """释放只清自己那段，不误清同信道相邻业务。"""
    import ms_alloc_scenarios as sc
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    topology, grid, gn, validator, path_cache, state = sc.build_stack()
    alloc = KSPFirstFitFlexAllocator(
        path_cache=path_cache, qot_validator=validator, spectrum_grid=grid,
        network_topology=topology, slots_per_channel=SLOTS_PER_CHANNEL,
    )
    # 分配两个不相邻的槽段
    a = sc.make_request(sc.ServicePlan(service_id=0, source_id=0, destination_id=3,
                                       arrival_time=0.0, departure_time=1.0,
                                       bit_rate_gbps=400.0, n_slots=2))
    ok, svc_a = alloc.allocate(a, state)
    assert ok
    state.allocate_service(svc_a)

    b = sc.make_request(sc.ServicePlan(service_id=1, source_id=0, destination_id=3,
                                       arrival_time=0.0, departure_time=1.0,
                                       bit_rate_gbps=400.0, n_slots=2))
    ok, svc_b = alloc.allocate(b, state)
    assert ok
    state.allocate_service(svc_b)
    assert svc_a.first_slot != svc_b.first_slot

    # 只释放 A，B 的窗口必须仍在
    alloc.release(svc_a, state)
    link = state.get_link_state(svc_b.path[0], svc_b.path[1])
    lo_b, hi_b = svc_b.slot_interval
    assert link.occupied_slots[lo_b:hi_b].all(), "释放 A 误清了 B 的窗口"

    # 重复释放应静默，不抛
    alloc.release(svc_a, state)


# ---------------------------------------------------------------------------
# 阻断按槽区间
# ---------------------------------------------------------------------------
def test_blocked_slot_overlap_makes_window_infeasible():
    """ROADM 阻断槽与窗口重叠 -> 该窗口不可行（切片语义）。"""
    import ms_alloc_scenarios as sc
    from network_nlo_eval.network.elements import ROADMConfig
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    # 阻断槽 5（落在第二个信道 [4,8) 里）
    topology, grid, gn, validator, path_cache, state = sc.build_stack(
        roadm_config=ROADMConfig(blocked_slots=[5])
    )
    alloc = KSPFirstFitFlexAllocator(
        path_cache=path_cache, qot_validator=validator, spectrum_grid=grid,
        network_topology=topology, slots_per_channel=SLOTS_PER_CHANNEL,
    )
    assert alloc.blocked_slots_on_path([0, 1, 3]) == {5}

    # 4 槽窗口 [0,4) = 槽 {0,1,2,3}，不含 5 -> 可行，first_slot=0。
    plan = sc.ServicePlan(service_id=0, source_id=0, destination_id=3,
                          arrival_time=0.0, departure_time=1.0,
                          bit_rate_gbps=400.0, n_slots=4)
    ok, svc = alloc.allocate(sc.make_request(plan), state)
    assert ok
    assert svc.first_slot == 0

    # 6 槽窗口：起点 0..5 的窗口都含槽 5 -> 全部不可行；
    # 首个不含 5 的窗口是 [6,12) = 槽 {6..11}，故 first_slot=6。
    plan2 = sc.ServicePlan(service_id=1, source_id=0, destination_id=3,
                           arrival_time=0.0, departure_time=1.0,
                           bit_rate_gbps=400.0, n_slots=6)
    ok2, svc2 = alloc.allocate(sc.make_request(plan2), state)
    assert ok2
    assert svc2.first_slot == 6


def test_blocked_channel_expands_to_slot_interval():
    """blocked_channels 应按 slots_per_channel 展开为槽区间参与阻断。"""
    import ms_alloc_scenarios as sc
    from network_nlo_eval.network.elements import ROADMConfig
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    # 阻断信道 0 -> 槽 [0,4)
    topology, grid, gn, validator, path_cache, state = sc.build_stack(
        roadm_config=ROADMConfig(blocked_channels=[0])
    )
    alloc = KSPFirstFitFlexAllocator(
        path_cache=path_cache, qot_validator=validator, spectrum_grid=grid,
        network_topology=topology, slots_per_channel=SLOTS_PER_CHANNEL,
    )
    assert alloc.blocked_slots_on_path([0, 1, 3]) == {0, 1, 2, 3}

    # 4 槽窗口不能用信道 0（槽 0-3 全被阻断）-> 首个可行起点是 4（信道 1）。
    plan = sc.ServicePlan(service_id=0, source_id=0, destination_id=3,
                          arrival_time=0.0, departure_time=1.0,
                          bit_rate_gbps=400.0, n_slots=4)
    ok, svc = alloc.allocate(sc.make_request(plan), state)
    assert ok
    assert svc.first_slot == 4


# ---------------------------------------------------------------------------
# 退化等价（M-J10 行为侧）
# ---------------------------------------------------------------------------
def test_degenerate_allocation_matches_legacy_bitwise():
    """slots_per_channel=4 + 全 n_slots=1 -> 与 KSPFirstFitAllocator 逐位一致。"""
    import ms_alloc_degenerate
    import ms_alloc_scenarios as sc

    doc = ms_alloc_degenerate.run_degenerate()
    assert doc["conditions"]["slots_per_channel"] == SLOTS_PER_CHANNEL
    assert all(row["n_slots"] == 1 for row in doc["comparison"])
    assert doc["summary"]["n_compared"] > 0
    assert doc["summary"]["n_identical"] == doc["summary"]["n_compared"], (
        "退化条件下存在与旧分配器不一致的请求"
    )
    del sc


# ---------------------------------------------------------------------------
# 端到端证据落盘守卫（M-J9 行为侧）
# ---------------------------------------------------------------------------
def test_e2e_evidence_file_has_multislot_readings():
    """results/ms_alloc_e2e.json 必须含 >=3 个不同 n_slots 与释放回收读数。"""
    path = RESULTS / "ms_alloc_e2e.json"
    if not path.is_file():
        pytest.skip("先跑 tools/ms_alloc_e2e.py --write")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["n_distinct_n_slots_allocated"] >= 3
    for row in doc["per_request"]:
        if row["allocated"]:
            assert row["first_slot"] is not None
            assert row["n_slots"] >= 1
    rp = doc["release_probe"]
    assert rp["window_reclaimed"] is True
    assert rp["reallocate_reused_same_first_slot"] is True


def test_m_j6_evidence_all_delta_zero():
    """results/ms_alloc_legacy_regression.json：6 场景 delta_db 全 0.0。"""
    path = RESULTS / "ms_alloc_legacy_regression.json"
    if not path.is_file():
        pytest.skip("先跑 tools/ms_alloc_legacy_regression.py --write")
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["tolerance"] == 0.0
    assert len(doc["scenarios"]) == 6
    comparable = [r for r in doc["scenarios"] if not r["both_rejected"]]
    for r in comparable:
        assert r["delta_db"] == 0.0, f"{r['scenario']}: delta_db={r['delta_db']!r}"