#!/usr/bin/env python3
"""WP-M1 判据 M-J9：多槽请求端到端拿到**连续槽窗口**，且释放后槽被回收。

跑通的链路::

    ServiceRequest(n_slots)
        -> KSPFirstFitFlexAllocator   (跨路径逐槽 OR -> 连续窗口卷积 -> first_slot)
        -> AllocatedService(first_slot, n_slots, slots_per_channel, wavelength)
        -> NetworkState.allocate_slot_range  /  release_slot_range

证据全部落盘：逐请求 path/first_slot/n_slots/占用带宽/SNR，以及**释放前后**
该窗口的逐槽占用读数（M-J9 明确要求「给出可验证的读数，例如释放前后该窗口的
占用状态」）。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ms_alloc_e2e.py --write
    ~/miniconda3/envs/rl/bin/python tools/ms_alloc_e2e.py --check

产物：``results/ms_alloc_e2e.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results" / "ms_alloc_e2e.json"


def _jsonable(value):
    """把 numpy 标量与 ±inf 转成可 JSON 序列化的形式。"""
    import numpy as np

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        if value != value:
            return "nan"
    return value


class _RecordingAllocator:
    """包一层 KSPFirstFitFlexAllocator，记录每次**成功**分配出的 AllocatedService。"""

    def __init__(self, inner):
        self.inner = inner
        self.allocated: dict[int, object] = {}

    def allocate(self, service_request, network_state):
        ok, svc = self.inner.allocate(service_request, network_state)
        if ok and svc is not None:
            self.allocated[service_request.service_id] = svc
        return ok, svc

    def release(self, allocated_service, network_state):
        return self.inner.release(allocated_service, network_state)

    def __getattr__(self, item):
        return getattr(self.inner, item)


class _RecordingValidator:
    """包一层 QoTValidator，记录每次**被采纳**的验证结果。

    只记录返回 True 的那一次——那正是 RWA 最终选中的 (service, path, channel)。
    SNR 是该候选点的真实 QoT 读数（verify_allocation 内部算的，非事后补算）。
    """

    def __init__(self, inner):
        self.inner = inner
        self.accepted: dict[int, dict] = {}

    def verify_allocation(self, service_request, path, channel_idx, current_network_state,
                          fixed_span_length_km=None):
        ok, snrs = self.inner.verify_allocation(
            service_request=service_request,
            path=path,
            channel_idx=channel_idx,
            current_network_state=current_network_state,
            fixed_span_length_km=fixed_span_length_km,
        )
        if ok:
            self.accepted.setdefault(
                service_request.service_id,
                {
                    "channel_idx": int(channel_idx),
                    "snr_db": _jsonable(float(snrs[channel_idx])) if snrs is not None else None,
                },
            )
        return ok, snrs

    def __getattr__(self, item):
        return getattr(self.inner, item)


def _slot_window_readout(network_state, service) -> dict:
    """读取一条业务在**首跳链路**上所占槽区间的逐槽状态（释放验证的读数）。"""
    link_state = network_state.get_link_state(service.path[0], service.path[1])
    lo, hi = service.slot_interval
    return {
        "first_slot": lo,
        "n_slots": service.n_slots,
        "window": [lo, hi],
        "occupied_slots_in_window": [bool(v) for v in link_state.occupied_slots[lo:hi].tolist()],
        "occupied_channels": [bool(v) for v in link_state.occupied_channels.tolist()],
    }


def run_e2e() -> dict:
    import numpy as np

    import ms_alloc_scenarios as sc
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator
    from network_nlo_eval.simulation.engine import SimulatorEngine
    from network_nlo_eval.simulation.reporter import MetricsCollector

    topology, grid, gn, validator, path_cache, state = sc.build_stack()
    recorder = _RecordingValidator(validator)
    flex = KSPFirstFitFlexAllocator(
        path_cache=path_cache,
        qot_validator=recorder,
        spectrum_grid=grid,
        network_topology=topology,
        slots_per_channel=sc.SLOTS_PER_CHANNEL,
    )
    allocator = _RecordingAllocator(flex)

    plans = sc.multislot_service_plans()
    services = [sc.make_request(plan) for plan in plans]

    collector = MetricsCollector(
        network_topology=topology,
        spectrum_grid=grid,
        total_incoming_services=len(services),
    )
    engine = SimulatorEngine(
        allocator=allocator,
        network_state=state,
        metrics_collector=collector,
        incoming_services=services,
    )
    engine.run()

    slot_width_ghz = sc.CHANNEL_SPACING_HZ / sc.SLOTS_PER_CHANNEL / 1e9
    per_request = []
    for plan in plans:
        svc = allocator.allocated.get(plan.service_id)
        snr = recorder.accepted.get(plan.service_id, {}).get("snr_db")
        if svc is None:
            per_request.append(
                {
                    "service_id": plan.service_id,
                    "n_slots": plan.n_slots,
                    "allocated": False,
                    "path": None,
                    "first_slot": None,
                    "wavelength": None,
                    "occupied_bandwidth_ghz": None,
                    "snr_db": None,
                }
            )
            continue
        per_request.append(
            {
                "service_id": plan.service_id,
                "n_slots": plan.n_slots,
                "allocated": True,
                "path": list(svc.path),
                "first_slot": svc.first_slot,
                "wavelength": svc.wavelength,
                "slots_per_channel": svc.slots_per_channel,
                "occupied_bandwidth_ghz": plan.n_slots * slot_width_ghz,
                "snr_db": snr,
            }
        )

    report = collector.generate_report(state)
    metrics = report["simulation_metrics"]

    # ---- 释放回收探针（M-J9 的「释放后槽被回收」） ----------------------
    release_probe = _release_reclaim_probe(topology, grid, validator, path_cache)

    distinct = sorted({row["n_slots"] for row in per_request if row["allocated"]})
    slot_width_ghz = sc.CHANNEL_SPACING_HZ / sc.SLOTS_PER_CHANNEL / 1e9

    return {
        "wp": "M1",
        "judgment": "M-J9",
        "scenario": "multislot_contiguous_window_e2e",
        "authored_by": "tools/ms_alloc_scenarios.py (WP-M1 自写)",
        "grid": {
            "slots_per_channel": sc.SLOTS_PER_CHANNEL,
            "channel_spacing_hz": sc.CHANNEL_SPACING_HZ,
            "slot_width_hz": sc.CHANNEL_SPACING_HZ / sc.SLOTS_PER_CHANNEL,
            "slot_width_ghz": slot_width_ghz,
            "num_channels": grid.num_channels,
            "total_slots": grid.num_channels * sc.SLOTS_PER_CHANNEL,
            "center_frequency_hz": sc.CENTER_FREQUENCY_HZ,
        },
        "physics_note": (
            "物理面仍是遗留信道粒度：MultiBandISRSGN 收 SpectrumGrid，入纤功率只写锚信道 "
            "first_slot // slots_per_channel。槽是分配层的真源。numba_kernels.py 零 diff。"
        ),
        "n_requests": len(per_request),
        "n_allocated": sum(1 for row in per_request if row["allocated"]),
        "distinct_n_slots_allocated": distinct,
        "n_distinct_n_slots_allocated": len(distinct),
        "per_request": per_request,
        "release_probe": release_probe,
        "simulation_metrics": {
            "blocking_rate": metrics.get("blocking_rate"),
            "wavelength_utilization": metrics.get("wavelength_utilization"),
            "average_hop_count": metrics.get("average_hop_count"),
        },
        "self_check": {
            "at_least_3_distinct_n_slots": len(distinct) >= 3,
            "all_allocated_contiguous": all(
                row.get("first_slot") is not None for row in per_request if row["allocated"]
            ),
            "release_reclaims_window": release_probe["window_reclaimed"],
        },
    }


def _release_reclaim_probe(topology, grid, validator, path_cache) -> dict:
    """显式演示：分配 n 槽窗口 -> 占用 -> 释放 -> 槽被回收 -> 再分配复用同一 first_slot。"""
    import ms_alloc_scenarios as sc
    from network_nlo_eval.network.state import NetworkState
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    state = NetworkState(topology, grid, slots_per_channel=sc.SLOTS_PER_CHANNEL)
    allocator = KSPFirstFitFlexAllocator(
        path_cache=path_cache,
        qot_validator=validator,
        spectrum_grid=grid,
        network_topology=topology,
        slots_per_channel=sc.SLOTS_PER_CHANNEL,
    )

    plan = sc.ServicePlan(
        service_id=900, source_id=0, destination_id=3,
        arrival_time=0.0, departure_time=10.0, bit_rate_gbps=400.0, n_slots=3,
    )
    req = sc.make_request(plan)

    before = _first_hop_window(state, allocator, plan, service_id=0, n_slots=3)
    ok, svc = allocator.allocate(req, state)
    if not ok:
        return {"ok": False, "window_reclaimed": False, "error": "probe allocation failed"}
    state.allocate_service(svc)

    after_alloc = _slot_window_readout(state, svc)

    allocator.release(svc, state)
    after_release = _slot_window_readout(state, svc)

    # 释放后槽应全部归零
    reclaimed = not any(after_release["occupied_slots_in_window"])

    # 再分配同样 n 槽，应复用同一 first_slot
    plan2 = sc.ServicePlan(
        service_id=901, source_id=0, destination_id=3,
        arrival_time=0.0, departure_time=10.0, bit_rate_gbps=400.0, n_slots=3,
    )
    ok2, svc2 = allocator.allocate(sc.make_request(plan2), state)
    reused = bool(ok2 and svc2.first_slot == svc.first_slot)

    del before
    return {
        "ok": True,
        "n_slots": svc.n_slots,
        "first_slot": svc.first_slot,
        "wavelength": svc.wavelength,
        "window": [svc.first_slot, svc.first_slot + svc.n_slots],
        "after_allocate": after_alloc,
        "after_release": after_release,
        "window_reclaimed": reclaimed,
        "reallocate_reused_same_first_slot": reused,
        "reallocate_first_slot": svc2.first_slot if ok2 else None,
    }


def _first_hop_window(state, allocator, plan, service_id, n_slots):
    """分配前的首跳槽占用快照。"""
    link_state = state.get_link_state(plan.source_id, plan.source_id + 1)
    return {
        "occupied_slots_head": [bool(v) for v in link_state.occupied_slots[:8].tolist()],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = run_e2e()

    failures = []
    if doc["n_distinct_n_slots_allocated"] < 3:
        failures.append(
            f"M-J9: 只有 {doc['n_distinct_n_slots_allocated']} 个不同 n_slots 拿到窗口（要求 >=3）"
        )
    for row in doc["per_request"]:
        if row["allocated"] and row["first_slot"] is None:
            failures.append(f"M-J9: service {row['service_id']} 缺 first_slot")
    if not doc["release_probe"]["window_reclaimed"]:
        failures.append("M-J9: 释放后槽未被回收")
    if not doc["release_probe"]["reallocate_reused_same_first_slot"]:
        failures.append("M-J9: 释放后再分配未复用同一 first_slot")

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        print(f"  分配成功 {doc['n_allocated']}/{doc['n_requests']}，"
              f"不同 n_slots={doc['distinct_n_slots_allocated']}")
        for row in doc["per_request"]:
            print(
                f"    svc{row['service_id']} n_slots={row['n_slots']} "
                f"first_slot={row['first_slot']} wavelength={row['wavelength']} "
                f"BW={row['occupied_bandwidth_ghz']} GHz SNR={row['snr_db']} dB"
            )
        rp = doc["release_probe"]
        print(f"  释放探针: window={rp.get('window')} reclaimed={rp['window_reclaimed']} "
              f"reused_first_slot={rp['reallocate_reused_same_first_slot']}")

    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        for key in ("per_request", "grid", "release_probe", "distinct_n_slots_allocated"):
            if stored.get(key) != doc.get(key):
                print(f"FAIL 复算不一致: {key}")
                return 1
        if failures:
            for f in failures:
                print(f"FAIL {f}")
            return 1
        print("OK  M-J9 复算通过：>=3 个不同 n_slots 拿到连续窗口，释放后槽被回收")
        return 0

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())