"""``QoTValidator`` 的入纤功率口径（任务 O2）。

网络/RWA 参考面始终是每信道双偏振总功率。``QoTValidator`` 只声明**输入**
口径（``NetworkState`` / ``ServiceRequest`` / provider 入参），并在内部换算到
评估器口径一次：

* 默认 ``total`` 与旧调用逐位一致；
* ``launch_power_convention="per_pol"`` 的网络功率谱（物理总功率的一半）与
  ``total`` 输入给出相同的判定与逐位相同的 SNR；
* ``LinkNoiseCache`` 使用评估器坐标：默认 ``total`` 时是总功率，``per_pol``
  评估器时是每偏振功率；
* 评估器声明 ``per_pol`` 时被接受（任务 X）；坐标与 ASE 缩放的等价性在
  ``test_qot_network_perpol`` 中覆盖。
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.rwa.validity import H3LinkValidityProvider, LinkValidityPolicy
from network_nlo_eval.simulation.traffic import ServiceRequest

CASE = build_case("scaling_40ch")


def _network(channels: int = 4) -> tuple[NetworkTopology, SpectrumGrid, MultiBandISRSGN, NetworkState]:
    """两节点单链路的最小 RWA 栈，与 ``test_network_validity`` 同构。"""
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, weight=80.0)
    fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)
    topology = NetworkTopology(graph, fiber, EDFAConfig(noise_figure_db=5.0), ROADMConfig())
    grid = SpectrumGrid(channels, 50e9, 193.1e12)
    evaluator = MultiBandISRSGN(grid, fiber, 32e9)
    state = NetworkState(topology, grid)
    return topology, grid, evaluator, state


def _request(launch_power_w: float = 1e-3) -> ServiceRequest:
    """单信道业务请求；默认阈值 0 dB 让基线必然通过。"""
    return ServiceRequest(
        service_id=1,
        source_id=0,
        destination_id=1,
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=0.0,
        launch_power_w=launch_power_w,
    )


# ---------------------------------------------------------------------------
# 默认不变
# ---------------------------------------------------------------------------


def test_default_total_is_bit_identical_to_explicit_total() -> None:
    """默认构造与显式 ``total`` 的判定和 SNR 数组逐位相同。"""
    topology, grid, evaluator, state = _network()
    default = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    explicit = QoTValidator(
        evaluator, topology, grid, fixed_span_length_km=80.0, launch_power_convention="total"
    )
    default_ok, default_snr = default.verify_allocation(_request(), [0, 1], 0, state)
    explicit_ok, explicit_snr = explicit.verify_allocation(_request(), [0, 1], 0, state)
    assert default.launch_power_convention == "total"
    assert default.eval_power_convention == "total"
    assert default_ok is explicit_ok
    assert default_snr is not None and explicit_snr is not None
    assert np.array_equal(default_snr, explicit_snr)


# ---------------------------------------------------------------------------
# per_pol 输入等价
# ---------------------------------------------------------------------------


def test_per_pol_input_matches_total_input_end_to_end() -> None:
    """同一物理链路：per_pol 网络功率（总功率的一半）= total 网络功率。"""
    topology, grid, evaluator, state = _network()
    total = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    per_pol = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        launch_power_convention="per_pol",
    )
    total_ok, total_snr = total.verify_allocation(_request(1e-3), [0, 1], 0, state)
    per_pol_ok, per_pol_snr = per_pol.verify_allocation(_request(0.5e-3), [0, 1], 0, state)
    assert per_pol_ok is total_ok
    assert total_snr is not None and per_pol_snr is not None
    # 0.5 与 2.0 都是二进制精确因子，换算后逐位回到同一功率谱。
    assert np.array_equal(per_pol_snr, total_snr)


def test_link_noise_cache_stays_in_total_coordinates() -> None:
    """per_pol 输入只在内部换算，写回 ``LinkState`` 的缓存仍是总功率坐标。"""
    results = []
    for convention, power in (("total", 1e-3), ("per_pol", 0.5e-3)):
        topology, grid, evaluator, state = _network()
        validator = QoTValidator(
            evaluator,
            topology,
            grid,
            fixed_span_length_km=80.0,
            launch_power_convention=convention,
        )
        ok, snr = validator.verify_allocation(_request(power), [0, 1], 0, state)
        assert ok
        assert snr is not None
        cache = state.get_link_state(0, 1).noise_cache
        assert cache is not None
        results.append(cache)
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(results[1], name), getattr(results[0], name)), name
    assert results[1].span_count == results[0].span_count


def test_per_pol_spectrum_without_conversion_changes_the_snr() -> None:
    """守卫换算确实发生：把 per_pol 功率喂给总功率验证器会掉 SNR。"""
    topology, grid, evaluator, state = _network()
    per_pol = QoTValidator(
        evaluator, topology, grid, fixed_span_length_km=80.0, launch_power_convention="per_pol"
    )
    total = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    _, converted = per_pol.verify_allocation(_request(0.5e-3), [0, 1], 0, state)
    # 同一 per_pol 数组交给 total 验证器：少了 ×2，SNR 必然更低。
    _, unconverted = total.verify_allocation(_request(0.5e-3), [0, 1], 0, state)
    assert converted is not None and unconverted is not None
    active = np.isfinite(converted)
    assert np.all(active[:1])
    # 信号减半、噪声不同步减半，占用信道 SNR 掉 1 dB 以上。
    assert np.all(unconverted[active] < converted[active] - 1.0)


# ---------------------------------------------------------------------------
# 非法/不支持的口径
# ---------------------------------------------------------------------------


class _PerPolEvaluator:
    """只声明口径的占位评估器；构造校验发生在读取其他属性之前。"""

    launch_power_convention = "per_pol"
    ase_polarization_scale = 0.5


def test_evaluator_declaring_per_pol_is_accepted() -> None:
    """任务 X：``assemble_link_noise`` 在评估器坐标内工作，per_pol 不再被拒绝。"""
    topology, grid, _evaluator, _state = _network()
    validator = QoTValidator(_PerPolEvaluator(), topology, grid)
    assert validator.eval_power_convention == "per_pol"
    assert validator.ase_polarization_scale == 0.5


def test_provider_input_convention_must_match_the_evaluator() -> None:
    """Provider 会收到评估器坐标的功率，输入口径不一致会二次换算。"""
    topology, grid, evaluator, _state = _network()
    provider = H3LinkValidityProvider(
        make_model("pcfm_h3_raman", replace(CASE, num_channels=4)),
        input_power_convention="per_pol",
    )
    with pytest.raises(ValueError, match="must equal the evaluator convention"):
        QoTValidator(
            evaluator,
            topology,
            grid,
            validity_policy=LinkValidityPolicy(mode="flag"),
            validity_provider=provider,
        )


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL", "auto"])
def test_invalid_convention_is_rejected(convention: str) -> None:
    """口径只接受 ``total`` / ``per_pol``。"""
    topology, grid, evaluator, _state = _network()
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        QoTValidator(evaluator, topology, grid, launch_power_convention=convention)
