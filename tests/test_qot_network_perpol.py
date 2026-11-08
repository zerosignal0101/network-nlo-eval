"""网络级 QoT 的 per_pol evaluator 端到端测试（第七轮任务 X）。

背景
----
``MultiBandISRSGN`` 的标称入口是每信道双偏振总功率。一个声明
``launch_power_convention="per_pol"`` 的评估器接收每偏振功率，并返回每偏振
信号/SPM/XPM/ASE；同一物理总功率下 SNR 与 total 坐标一致。

本测试用 ``_PerPolISRSGN`` 作为**物理自洽的 per_pol evaluator** 替身：它把
每偏振输入乘 2 交给冻结的 total 核，再把信号/NLI 减半；EDFA ASE 由共享
route 层的 ``ase_polarization_scale=0.5`` 减半。这正好复刻 H3/PCFM 模型在
``per_pol`` 下的坐标边界。

覆盖：

* 默认 total evaluator 的判定、SNR 与噪声缓存逐位不变；
* per_pol evaluator 不再被 ``NotImplementedError`` 拒绝；
* 相同物理总功率下 total 与 per_pol 给出逐位相同的 SNR（含 ROADM booster
  ASE 的 ``equalize_output_power`` 路径）；
* ``ase_polarization_scale`` 只作用于共享层新产生的 ASE；漏配或与口径不匹配
  时构造期直接拒绝，避免混合坐标进入 SNR；
* H3 有效性 provider 在 per_pol 口径下可以接入网络 QoT；
* 非法口径仍然报 ``ValueError``。
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


# ---------------------------------------------------------------------------
# per_pol evaluator 替身
# ---------------------------------------------------------------------------


class _PerPolISRSGN(MultiBandISRSGN):
    """物理自洽的 per_pol 评估器：对外是每偏振坐标，内部用 total 核。

    ``_calc_span_noise_and_power`` 收到每偏振功率 ``P_pol`` 后乘 2 得到物理
    总功率 ``P_total``，交给冻结的 total 核；信号与 NLI 再减半回每偏振坐标。
    EDFA ASE 由 ``ase_polarization_scale=0.5`` 在共享 route 层减半，因此这里
    不再重复缩放。
    """

    launch_power_convention = "per_pol"
    ase_polarization_scale = 0.5

    def for_fiber(self, fiber_config: FiberSpanConfig) -> _PerPolISRSGN:
        """返回同类的光纤子评估器；基类会构造总功率实例，坐标会丢。"""
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


class _PerPolMissingAseScale(_PerPolISRSGN):
    """每偏振数组正确，但漏声明共享层 ASE 缩放的反例。

    ``_calc_span_noise_and_power`` 手动减半 ASE（数组坐标是对的），但
    ``ase_polarization_scale`` 保持 1.0。若放行，``QoTValidator`` 累加 ROADM
    booster ASE 时不会减半，打开 ``equalize_output_power=True`` 后会静默错位。
    构造期必须拒绝这种口径与缩放不一致的评估器。
    """

    ase_polarization_scale = 1.0

    def _calc_span_noise_and_power(
        self,
        power_in_w: np.ndarray,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p_out, spm, xpm, ase = MultiBandISRSGN._calc_span_noise_and_power(
            self,
            power_in_w * 2.0,
            span_length_m,
            edfa_config,
            roadm_config,
            n_effective_spans=n_effective_spans,
        )
        return p_out * 0.5, spm * 0.5, xpm * 0.5, ase * 0.5


# ---------------------------------------------------------------------------
# 最小网络栈
# ---------------------------------------------------------------------------


def _network(
    *,
    equalize: bool = False,
    evaluator_factory=None,
) -> tuple[NetworkTopology, SpectrumGrid, MultiBandISRSGN, NetworkState]:
    """两节点单链路的最小 RWA 栈，与 ``test_qot_launch_convention`` 同构。"""
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, weight=80.0)
    fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)
    topology = NetworkTopology(
        graph,
        fiber,
        EDFAConfig(noise_figure_db=5.0),
        ROADMConfig(equalize_output_power=equalize),
    )
    grid = SpectrumGrid(4, 50e9, 193.1e12)
    factory = evaluator_factory or (lambda g, f: MultiBandISRSGN(g, f, 32e9))
    evaluator = factory(grid, fiber)
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


def _per_pol_validator(topology, grid, evaluator, *, equalize: bool = False) -> QoTValidator:
    """构造 per_pol evaluator 的网络验证器。"""
    return QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        launch_power_convention="per_pol",
    )


# ---------------------------------------------------------------------------
# 默认 total 路径
# ---------------------------------------------------------------------------


def test_default_total_evaluator_declares_no_coordinate_change() -> None:
    """默认评估器的坐标与 ASE 缩放都保持历史取值。"""
    topology, grid, evaluator, _state = _network()
    validator = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    assert validator.launch_power_convention == "total"
    assert validator.eval_power_convention == "total"
    assert validator.ase_polarization_scale == 1.0


def test_default_and_explicit_total_are_bit_identical() -> None:
    """默认构造与显式 ``total`` 的判定、SNR 与噪声缓存逐位相同。"""
    results = []
    for convention in (None, "total"):
        topology, grid, evaluator, state = _network()
        kwargs = {} if convention is None else {"launch_power_convention": convention}
        validator = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0, **kwargs)
        ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)
        assert ok
        assert snr is not None
        cache = state.get_link_state(0, 1).noise_cache
        assert cache is not None
        results.append((snr, cache))

    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(results[0][1], name), getattr(results[1][1], name)), name
    assert np.array_equal(results[0][0], results[1][0])
    assert results[0][1].span_count == results[1][1].span_count


def test_total_evaluator_with_per_pol_input_is_unchanged() -> None:
    """Total evaluator 仍可接受 per_pol 网络输入，内部换算后逐位回到 total。"""
    total_snr = None
    per_pol_snr = None
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
        assert ok and snr is not None
        if convention == "total":
            total_snr = snr
        else:
            per_pol_snr = snr
    assert total_snr is not None and per_pol_snr is not None
    assert np.array_equal(total_snr, per_pol_snr)


# ---------------------------------------------------------------------------
# per_pol evaluator：接受与等价
# ---------------------------------------------------------------------------


def test_per_pol_evaluator_is_accepted() -> None:
    """构造期不再拒绝声明 ``launch_power_convention="per_pol"`` 的评估器。"""
    topology, grid, evaluator, state = _network(evaluator_factory=lambda g, f: _PerPolISRSGN(g, f, 32e9))
    validator = _per_pol_validator(topology, grid, evaluator)
    assert validator.eval_power_convention == "per_pol"
    assert validator.ase_polarization_scale == 0.5
    ok, snr = validator.verify_allocation(_request(0.5e-3), [0, 1], 0, state)
    assert ok and snr is not None


@pytest.mark.parametrize("equalize", [False, True])
def test_per_pol_and_total_give_equivalent_snr(equalize: bool) -> None:
    """相同物理总功率：total+total 与 per_pol+per_pol 的 SNR 逐位相同。"""
    total_topo, total_grid, total_evaluator, total_state = _network(equalize=equalize)
    per_topo, per_grid, per_evaluator, per_state = _network(
        equalize=equalize,
        evaluator_factory=lambda g, f: _PerPolISRSGN(g, f, 32e9),
    )
    total = QoTValidator(total_evaluator, total_topo, total_grid, fixed_span_length_km=80.0)
    per_pol = _per_pol_validator(per_topo, per_grid, per_evaluator)

    ok_total, snr_total = total.verify_allocation(_request(1e-3), [0, 1], 0, total_state)
    ok_per, snr_per = per_pol.verify_allocation(_request(0.5e-3), [0, 1], 0, per_state)
    assert ok_total is ok_per is True
    assert snr_total is not None and snr_per is not None
    assert np.array_equal(snr_total, snr_per)

    # 缓存坐标：per_pol 缓存的每个分量恰好是 total 的一半。
    total_cache = total_state.get_link_state(0, 1).noise_cache
    per_cache = per_state.get_link_state(0, 1).noise_cache
    assert total_cache is not None and per_cache is not None
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(total_cache, name), 2.0 * getattr(per_cache, name)), name
    assert total_cache.span_count == per_cache.span_count


def test_per_pol_evaluator_accepts_a_total_input_spectrum() -> None:
    """per_pol evaluator 也能直接吃默认 total 网络功率，并在入口换算一次。"""
    total_topo, total_grid, total_evaluator, total_state = _network(equalize=True)
    per_topo, per_grid, per_evaluator, per_state = _network(
        equalize=True, evaluator_factory=lambda g, f: _PerPolISRSGN(g, f, 32e9)
    )
    total = QoTValidator(total_evaluator, total_topo, total_grid, fixed_span_length_km=80.0)
    per_pol = QoTValidator(per_evaluator, per_topo, per_grid, fixed_span_length_km=80.0)
    ok_total, snr_total = total.verify_allocation(_request(1e-3), [0, 1], 0, total_state)
    ok_per, snr_per = per_pol.verify_allocation(_request(1e-3), [0, 1], 0, per_state)
    assert ok_total is ok_per is True
    assert snr_total is not None and snr_per is not None
    assert np.array_equal(snr_total, snr_per)


def test_per_pol_evaluator_returns_the_same_path_snr_array() -> None:
    """``verify_allocation`` 返回的逐信道 SNR 数组在两种坐标下逐位相同。"""
    total_topo, total_grid, total_evaluator, total_state = _network()
    per_topo, per_grid, per_evaluator, per_state = _network(
        evaluator_factory=lambda g, f: _PerPolISRSGN(g, f, 32e9)
    )
    total = QoTValidator(total_evaluator, total_topo, total_grid, fixed_span_length_km=80.0)
    per_pol = _per_pol_validator(per_topo, per_grid, per_evaluator)
    _, snr_total = total.verify_allocation(_request(1e-3), [0, 1], 0, total_state)
    _, snr_per = per_pol.verify_allocation(_request(0.5e-3), [0, 1], 0, per_state)
    assert snr_total is not None and snr_per is not None
    assert np.array_equal(snr_total, snr_per)


def test_missing_shared_ase_scale_is_rejected() -> None:
    """per_pol 评估器必须声明 ``ase_polarization_scale=0.5``。

    若保持 1.0，缓存里的 EDFA ASE 是每偏振坐标，而共享层 ROADM booster ASE
    仍按总量累加；无 equalize 时看不出来，一旦打开 equalize 才会静默错位。
    构造期直接拒绝，避免把混合坐标带进 SNR。
    """
    topo, grid, evaluator, _state = _network(
        evaluator_factory=lambda g, f: _PerPolMissingAseScale(g, f, 32e9)
    )
    with pytest.raises(ValueError, match="ase_polarization_scale must be 0.5"):
        _per_pol_validator(topo, grid, evaluator)


# ---------------------------------------------------------------------------
# H3 有效性 bridge
# ---------------------------------------------------------------------------


def test_h3_per_pol_validity_provider_runs_on_a_per_pol_evaluator() -> None:
    """H3 provider 与 per_pol evaluator 同口径时，网络 QoT 端到端可用。"""
    topology, grid, evaluator, state = _network(
        evaluator_factory=lambda g, f: _PerPolISRSGN(g, f, 32e9)
    )
    h3_model = make_model(
        "pcfm_h3_raman",
        replace(CASE, num_channels=4),
        signal_polarizations=2,
        launch_power_convention="per_pol",
    )
    provider = H3LinkValidityProvider(h3_model, input_power_convention="per_pol")
    validator = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        launch_power_convention="per_pol",
        validity_policy=LinkValidityPolicy(mode="flag"),
        validity_provider=provider,
    )
    ok, snr = validator.verify_allocation(_request(0.5e-3), [0, 1], 0, state)
    assert ok and snr is not None

    # 物理 SNR 与关闭有效性、total evaluator 的基线一致（flag 不改有限值）。
    total_topo, total_grid, total_evaluator, total_state = _network()
    baseline = QoTValidator(total_evaluator, total_topo, total_grid, fixed_span_length_km=80.0)
    ok_base, snr_base = baseline.verify_allocation(_request(1e-3), [0, 1], 0, total_state)
    assert ok_base is True and snr_base is not None
    assert np.array_equal(snr, snr_base)
    assert provider.model_power_convention == "per_pol"


# ---------------------------------------------------------------------------
# 非法口径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL", "auto"])
def test_invalid_input_convention_is_still_rejected(convention: str) -> None:
    """输入口径只接受 ``total`` / ``per_pol``。"""
    topology, grid, evaluator, _state = _network()
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        QoTValidator(evaluator, topology, grid, launch_power_convention=convention)


class _BadScaleEvaluator:
    """声明非法 ASE 缩放的评估器；只用于构造期校验。"""

    launch_power_convention = "per_pol"
    ase_polarization_scale = 0.0


def test_invalid_evaluator_ase_scale_is_rejected() -> None:
    """``ase_polarization_scale`` 必须有限且为正。"""
    topology, grid, _evaluator, _state = _network()
    with pytest.raises(ValueError, match="ase_polarization_scale"):
        QoTValidator(_BadScaleEvaluator(), topology, grid)


def test_provider_convention_must_match_the_evaluator() -> None:
    """Provider 口径与 per_pol evaluator 不一致时仍然报错。"""
    topology, grid, evaluator, _state = _network(
        evaluator_factory=lambda g, f: _PerPolISRSGN(g, f, 32e9)
    )
    h3_model = make_model(
        "pcfm_h3_raman",
        replace(CASE, num_channels=4),
        signal_polarizations=2,
        launch_power_convention="per_pol",
    )
    provider = H3LinkValidityProvider(h3_model, input_power_convention="total")
    with pytest.raises(ValueError, match="must equal the evaluator convention"):
        QoTValidator(
            evaluator,
            topology,
            grid,
            launch_power_convention="per_pol",
            validity_policy=LinkValidityPolicy(mode="flag"),
            validity_provider=provider,
        )
