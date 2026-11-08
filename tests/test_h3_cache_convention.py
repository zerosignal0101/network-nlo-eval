"""链路噪声缓存的口径溯源（第八轮任务 Z）。

背景
----
任务 X 让 ``LinkNoiseCache`` 跟随评估器口径（``total`` 或 ``per_pol``），但缓存
自身没有记录坐标。同一 ``NetworkState`` 上换用不同口径的 ``QoTValidator`` 时，
旧缓存会被静默复用，把两种坐标的噪声混进同一条路径。

本测试覆盖：

* ``LinkNoiseCache.launch_power_convention`` 默认 ``None``，旧构造/旧缓存兼容；
* 默认 ``total`` 路径的缓存与直接调用共用的装配函数逐位相同；
* ``per_pol`` 评估器写出的缓存带 ``per_pol`` 标签；
* 同一状态切换验证器口径时，已知不匹配的缓存被重算并替换，SNR 与基线逐位
  相同；
* 反方向（per_pol → total）同样重算；
* ``None`` 旧缓存保持既有复用行为，不在读取时被改写。

坐标系说明：网络层功率（``NetworkState`` / ``ServiceRequest``）始终按验证器
声明的输入口径解释；本测试统一用 ``launch_power_convention="total"``，只切换
评估器坐标，避免把"输入口径切换"和"缓存坐标切换"混为一谈。
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

SPAN_LENGTH_KM = 80.0


class _PerPolISRSGN(MultiBandISRSGN):
    """物理自洽的 per_pol 评估器：对外每偏振坐标，内部用冻结的 total 核。

    与 ``test_qot_network_perpol`` 的替身同构：每偏振输入乘 2 驱动 total 核，
    信号 / SPM / XPM 减半；EDFA ASE 由共享层的
    ``ase_polarization_scale=0.5`` 减半。
    """

    launch_power_convention = "per_pol"
    ase_polarization_scale = 0.5

    def for_fiber(self, fiber_config: FiberSpanConfig) -> _PerPolISRSGN:
        """返回同类的光纤子评估器，避免基类构造出 total 实例。"""
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = _PerPolISRSGN(
                self.grid,
                fiber_config,
                self.symbol_rate_hz,
                connections=self.connections,
                ode_accel=self.ode_accel,
                slots_per_channel=self.slots_per_channel,
                channel_bandwidth_source=self.channel_bandwidth_source,
                power_width_mode=self.power_width_mode,
            )
        return self._fiber_evaluator_cache[key]  # type: ignore[return-value]

    def _calc_span_noise_and_power(
        self,
        power_in_w: np.ndarray,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """每偏振入参 → 总功率核 → 每偏振输出。"""
        p_out, spm, xpm, ase = MultiBandISRSGN._calc_span_noise_and_power(
            self,
            power_in_w * 2.0,
            span_length_m,
            edfa_config,
            roadm_config,
            n_effective_spans=n_effective_spans,
        )
        return p_out * 0.5, spm * 0.5, xpm * 0.5, ase


def _network(
    *,
    nodes: int = 2,
    equalize: bool = False,
    per_pol: bool = False,
) -> tuple[NetworkTopology, SpectrumGrid, MultiBandISRSGN, NetworkState]:
    """``nodes`` 个节点的链式最小网络栈。"""
    graph = nx.path_graph(nodes)
    for u, v in graph.edges:
        graph[u][v]["weight"] = SPAN_LENGTH_KM
    fiber = FiberSpanConfig(length_km=SPAN_LENGTH_KM, max_span_length_km=SPAN_LENGTH_KM)
    topology = NetworkTopology(
        graph,
        fiber,
        EDFAConfig(noise_figure_db=5.0),
        ROADMConfig(equalize_output_power=equalize),
    )
    grid = SpectrumGrid(4, 50e9, 193.1e12)
    evaluator = _PerPolISRSGN(grid, fiber, 32e9) if per_pol else MultiBandISRSGN(grid, fiber, 32e9)
    state = NetworkState(topology, grid)
    return topology, grid, evaluator, state


def _request(service_id: int = 1, *, path: list[int] | None = None, power: float = 1e-3) -> ServiceRequest:
    """单信道业务请求；默认阈值 0 dB 让基线必然通过。"""
    path = path or [0, 1]
    return ServiceRequest(
        service_id=service_id,
        source_id=path[0],
        destination_id=path[-1],
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=0.0,
        launch_power_w=power,
    )


def _allocated(service_id: int, path: list[int], channel_idx: int, power: float = 1e-3) -> AllocatedService:
    """已分配业务；用于让 ``_collect_affected_services`` 看得到旧业务。"""
    return AllocatedService(
        service_id=service_id,
        source_id=path[0],
        destination_id=path[-1],
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=0.0,
        launch_power_w=power,
        path=path,
        wavelength=channel_idx,
        first_slot=channel_idx,
        slots_per_channel=None,
    )


def _assert_half(per_cache: LinkNoiseCache, total_cache: LinkNoiseCache) -> None:
    """per_pol 缓存每个分量必须恰好是 total 缓存的一半。"""
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(per_cache, name), 0.5 * getattr(total_cache, name)), name
    assert per_cache.span_count == total_cache.span_count


# ---------------------------------------------------------------------------
# 字段语义
# ---------------------------------------------------------------------------


def test_legacy_construction_has_no_convention() -> None:
    """旧构造不传口径时字段为 ``None``，数值不变。"""
    zeros = np.zeros(2)
    cache = LinkNoiseCache(spm=zeros, xpm=zeros.copy(), ase=zeros.copy(), span_count=1)
    assert cache.launch_power_convention is None


def test_known_convention_is_normalised_and_validated() -> None:
    """已知口径被保存，非法口径在构造期拒绝。"""
    zeros = np.zeros(2)
    cache = LinkNoiseCache(
        spm=zeros,
        xpm=zeros.copy(),
        ase=zeros.copy(),
        span_count=1,
        launch_power_convention="per_pol",
    )
    assert cache.launch_power_convention == "per_pol"
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        LinkNoiseCache(
            spm=zeros,
            xpm=zeros.copy(),
            ase=zeros.copy(),
            span_count=1,
            launch_power_convention="per-pal",
        )


# ---------------------------------------------------------------------------
# 默认 total：逐位不变
# ---------------------------------------------------------------------------


def test_total_cache_is_bit_identical_to_direct_assembly() -> None:
    """默认 total 验证器写出的缓存与直接装配路径逐位相同。"""
    topology, grid, evaluator, state = _network()
    validator = QoTValidator(evaluator, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM)
    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)
    assert ok and snr is not None

    cache = state.get_link_state(0, 1).noise_cache
    assert cache is not None
    assert cache.launch_power_convention == "total"

    # 复现验证器内部的入参：state 的功率谱副本 + 候选信道功率。
    power_profile = state.get_link_state(0, 1).launch_power_profile_w.copy()
    power_profile[0] = 1e-3
    assembled = assemble_link_noise(
        evaluator=evaluator.for_fiber(topology.get_fiber_config(0, 1)),
        edfa_config=topology.get_link_edfa_config(0, 1),
        power_profile=power_profile,
        span_length_m=SPAN_LENGTH_KM * 1000,
        num_full_spans=1,
        remainder_length_m=0.0,
        num_channels=validator.num_nodes,
    )
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(cache, name), getattr(assembled, name)), name
    assert cache.span_count == assembled.span_count
    assert assembled.launch_power_convention == "total"


def test_default_total_cache_equals_explicit_total_cache() -> None:
    """默认构造与显式 total 的缓存对象数值、口径标签都相同。"""
    caches = []
    for kwargs in ({}, {"launch_power_convention": "total"}):
        topology, grid, evaluator, state = _network()
        validator = QoTValidator(evaluator, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM, **kwargs)
        ok, _ = validator.verify_allocation(_request(), [0, 1], 0, state)
        assert ok
        cache = state.get_link_state(0, 1).noise_cache
        assert cache is not None
        caches.append(cache)

    assert caches[0].launch_power_convention == caches[1].launch_power_convention == "total"
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(caches[0], name), getattr(caches[1], name)), name
    assert caches[0].span_count == caches[1].span_count


# ---------------------------------------------------------------------------
# per_pol 写入标签
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("equalize", [False, True])
def test_per_pol_cache_records_the_evaluator_convention(equalize: bool) -> None:
    """per_pol 评估器写出的缓存带 ``per_pol`` 标签，数值恰为 total 的一半。"""
    total_topo, total_grid, total_eval, total_state = _network(equalize=equalize)
    per_topo, per_grid, per_eval, per_state = _network(equalize=equalize, per_pol=True)

    total = QoTValidator(total_eval, total_topo, total_grid, fixed_span_length_km=SPAN_LENGTH_KM)
    per_pol = QoTValidator(per_eval, per_topo, per_grid, fixed_span_length_km=SPAN_LENGTH_KM)
    ok_total, snr_total = total.verify_allocation(_request(), [0, 1], 0, total_state)
    ok_per, snr_per = per_pol.verify_allocation(_request(), [0, 1], 0, per_state)
    assert ok_total is ok_per is True

    total_cache = total_state.get_link_state(0, 1).noise_cache
    per_cache = per_state.get_link_state(0, 1).noise_cache
    assert total_cache is not None and per_cache is not None
    assert total_cache.launch_power_convention == "total"
    assert per_cache.launch_power_convention == "per_pol"
    _assert_half(per_cache, total_cache)
    assert snr_total is not None and snr_per is not None
    assert np.array_equal(snr_total, snr_per)


# ---------------------------------------------------------------------------
# 同一状态切换口径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("equalize", [False, True])
def test_switching_validator_convention_recomputes_stale_cache(equalize: bool) -> None:
    """同一 ``NetworkState`` 上切换评估器口径：旧缓存重算，SNR 与基线一致。

    三节点链路上先分配业务 A（跨两条链路）。total 验证器先为链路 (0,1) 写好
    total 缓存；随后 per_pol 验证器在链路 (1,2) 上分配业务 C。验证 C 时会重新
    评估旧业务 A，其路径包含 (0,1)——这里读到的就是**跨口径的旧缓存**。
    """
    topology, grid, total_eval, state = _network(nodes=3, equalize=equalize)
    state.allocate_service(_allocated(service_id=1, path=[0, 1, 2], channel_idx=0))

    total = QoTValidator(total_eval, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM)
    # 先按 state 的真实功率谱（业务 A 的 ch0）写出 total 缓存，并记录基线 SNR。
    total_snr_a = total._eval_service_snr([0, 1], 0, 1e-3, {}, state, SPAN_LENGTH_KM)
    total_cache = state.get_link_state(0, 1).noise_cache
    assert total_cache is not None
    assert total_cache.launch_power_convention == "total"

    per_eval = _PerPolISRSGN(grid, topology.get_fiber_config(1, 2), 32e9)
    per_pol = QoTValidator(per_eval, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM)
    ok, snr_c = per_pol.verify_allocation(_request(service_id=2, path=[1, 2]), [1, 2], 1, state)
    assert ok and snr_c is not None

    per_cache = state.get_link_state(0, 1).noise_cache
    assert per_cache is not None
    assert per_cache is not total_cache, "跨口径旧缓存被静默复用"
    assert per_cache.launch_power_convention == "per_pol"
    _assert_half(per_cache, total_cache)

    # 重算后的缓存必须给出与 total 基线逐位相同的 SNR（同一物理功率）。
    per_pol_snr_a = per_pol._eval_service_snr([0, 1], 0, 1e-3, {}, state, SPAN_LENGTH_KM)
    assert np.array_equal(
        np.asarray(per_pol_snr_a, dtype=np.float64),
        np.asarray(total_snr_a, dtype=np.float64),
    )


def test_known_convention_mismatch_recomputes_in_both_directions() -> None:
    """Total ↔ per_pol 两个方向都重算并替换，不共享同一条缓存。"""
    topology, grid, total_eval, state = _network()
    state.allocate_service(_allocated(service_id=1, path=[0, 1], channel_idx=0))

    total = QoTValidator(total_eval, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM)
    total_cache = total._get_or_calc_noise_cache((0, 1), state.get_link_state(0, 1), {}, SPAN_LENGTH_KM)
    assert total_cache.launch_power_convention == "total"

    per_eval = _PerPolISRSGN(grid, topology.get_fiber_config(0, 1), 32e9)
    per_pol = QoTValidator(per_eval, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM)
    per_cache = per_pol._get_or_calc_noise_cache((0, 1), state.get_link_state(0, 1), {}, SPAN_LENGTH_KM)
    assert per_cache is not total_cache
    assert per_cache.launch_power_convention == "per_pol"
    assert state.get_link_state(0, 1).noise_cache is per_cache
    _assert_half(per_cache, total_cache)

    back_cache = total._get_or_calc_noise_cache((0, 1), state.get_link_state(0, 1), {}, SPAN_LENGTH_KM)
    assert back_cache is not per_cache
    assert back_cache.launch_power_convention == "total"
    assert state.get_link_state(0, 1).noise_cache is back_cache
    # total 重算结果与第一次逐位相同（同一评估器、同一功率谱、同一求和顺序）。
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(back_cache, name), getattr(total_cache, name)), name


# ---------------------------------------------------------------------------
# None 旧缓存
# ---------------------------------------------------------------------------


def test_legacy_none_cache_is_reused_without_rewrite() -> None:
    """``None`` 旧缓存按既有行为复用；读取时不改写、不重算。"""
    topology, grid, evaluator, state = _network()
    values = np.full(4, 7.0)
    legacy = LinkNoiseCache(
        spm=values.copy(),
        xpm=values.copy(),
        ase=values.copy(),
        span_count=2,
    )
    state.get_link_state(0, 1).noise_cache = legacy

    total = QoTValidator(evaluator, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM)
    per_eval = _PerPolISRSGN(grid, topology.get_fiber_config(0, 1), 32e9)
    per_pol = QoTValidator(per_eval, topology, grid, fixed_span_length_km=SPAN_LENGTH_KM)

    assert total._get_or_calc_noise_cache((0, 1), state.get_link_state(0, 1), {}, SPAN_LENGTH_KM) is legacy
    assert per_pol._get_or_calc_noise_cache((0, 1), state.get_link_state(0, 1), {}, SPAN_LENGTH_KM) is legacy
    assert state.get_link_state(0, 1).noise_cache is legacy
    assert legacy.launch_power_convention is None
