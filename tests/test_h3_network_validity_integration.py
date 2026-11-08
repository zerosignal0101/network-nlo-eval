"""第九轮任务 AC：网络有效性信号的端到端 RWA 复核。

背景
----
第八轮 Y/AA/Z 分别加了泵-信号带重叠硬失效、密集栅格 ``reject`` 和链路噪声
缓存口径标签，但缺少一个端到端用例证明这些信号真的走到
``H3LinkValidityProvider → LinkValidity → QoTValidator``。本文件把三条链路
钉在一起：

* E+S+C+L 六泵（1505 nm 泵落在占用带内）在 ``mode="reject"`` 下不可分配，
  ``mode="flag"`` 下点值仍可用但报告 ``valid=False`` / ``flagged=True``；
  C+L 六泵保持可用；
* C 50 / 37.5 GHz 在 ``dense_grid_action="reject"`` 下不可分配，75 / 100 GHz
  的判定与 SNR 与默认策略逐位相同；
* 同一 ``NetworkState`` 交替使用 total / per_pol 验证器时，链路噪声缓存按
  口径重算，不把两种坐标的噪声混进同一条路径；
* 默认 ``off`` 路径不构造 provider、不附带有效性报告。

算例全部由 ``benchmarks.cases`` 的波段边界与本地构造的 ``BenchCase`` 生成，
不读 ``data/`` 或 ``results/``。泵-信号带重叠只在**占用带**覆盖泵频率时才会
触发，因此六泵用例先把整条栅格装满，只留下候选信道；这与“满载 DRA 链路”
的实际配置一致。
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, grid_for
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, RamanPumpConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.rwa.validity import (
    H3LinkValidityProvider,
    LinkValidity,
    LinkValidityPolicy,
    link_optical_path,
)
from network_nlo_eval.simulation.traffic import ServiceRequest

#: 与第五/六/七/八轮精度矩阵相同的六泵配置（论文 hybrid 的 E 波段泵组）。
#: 1505 nm 的泵频约 199.2 THz，落在 S 带内；E+S+C+L / S+C+L 的占用带包含它。
SIX_PUMPS: tuple[tuple[float, float], ...] = (
    (1340.0, 0.63),
    (1360.0, 0.63),
    (1427.0, 0.165),
    (1445.0, 0.131),
    (1476.0, 0.125),
    (1505.0, 0.075),
)

#: 候选业务阈值。取 -200 dB 让“满载梳 + 单候选信道”的物理 SNR 必然通过，
#: 于是 ``reject`` 用例的失败只能来自有效性策略，而不是链路预算。
PASSING_REQUIREMENT_DB = -200.0


class _PerPolISRSGN(MultiBandISRSGN):
    """物理自洽的 per_pol 评估器替身：内部用冻结的 total 核。

    与 ``test_qot_network_perpol`` / ``test_h3_cache_convention`` 的替身同构：
    每偏振输入乘 2 驱动 total 核，信号 / SPM / XPM 减半；EDFA ASE 由共享层的
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


def _pumps(spec: tuple[tuple[float, float], ...] = SIX_PUMPS) -> list[RamanPumpConfig]:
    """按波长/功率表构造后向泵浦。"""
    return [
        RamanPumpConfig(
            wavelength_nm=wavelength,
            power_w=power,
            direction="backward",  # type: ignore[arg-type]
            attenuation_db_per_km=0.25,
        )
        for wavelength, power in spec
    ]


def _case(bands: str, spacing_hz: float) -> BenchCase:
    """按波段边界构造与精度矩阵同源的单跨算例。"""
    low, high = band_edges_hz(bands)
    return BenchCase(
        name=f"ac_{bands}_{spacing_hz / 1e9:g}",
        bands=bands,
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


def _dense_case(spacing_hz: float) -> BenchCase:
    """C 波段、64 GBaud、给定栅格间隔的四信道单跨算例。"""
    return BenchCase(
        name=f"ac_dense_{spacing_hz / 1e9:g}",
        bands="C",
        num_channels=4,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=193.1e12,
        symbol_rate_hz=64e9,
        span_length_km=80.0,
        span_count=1,
        edfa_noise_figure_db=5.0,
        total_power_dbm=0.0,
    )


def _network(case: BenchCase) -> tuple[SpectrumGrid, FiberSpanConfig, NetworkTopology, MultiBandISRSGN, NetworkState]:
    """两节点单链路的最小 RWA 栈，网格与光纤取自 ``case``。"""
    grid = grid_for(case)
    fiber = build_path(case).fiber_configs[0]
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    graph.add_edge(0, 1, weight=fiber.length_km)
    topology = NetworkTopology(
        graph,
        fiber,
        EDFAConfig(noise_figure_db=case.edfa_noise_figure_db),
        ROADMConfig(),
    )
    evaluator = MultiBandISRSGN(grid, fiber, case.symbol_rate_hz)
    state = NetworkState(topology, grid)
    return grid, fiber, topology, evaluator, state


def _h3_model(case: BenchCase, *, pumps=None, per_pol: bool = False):
    """构造与网络同网格的 H3 模型。"""
    options = {}
    if pumps is not None:
        options["raman_pumps"] = pumps
    if per_pol:
        options.update(signal_polarizations=2, launch_power_convention="per_pol")
    return make_model("pcfm_h3_raman", case, **options)


def _validator(
    case: BenchCase,
    topology: NetworkTopology,
    grid: SpectrumGrid,
    evaluator: MultiBandISRSGN,
    *,
    policy: LinkValidityPolicy | None = None,
    provider: H3LinkValidityProvider | None = None,
    launch_power_convention: str = "total",
) -> QoTValidator:
    """构造验证器；``span`` 固定为一跨，便于与算例逐跨比较。"""
    return QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=case.span_length_km,
        launch_power_convention=launch_power_convention,
        validity_policy=policy,
        validity_provider=provider,
    )


def _request(*, power_w: float = 1e-3, requirement_db: float = PASSING_REQUIREMENT_DB) -> ServiceRequest:
    """单信道业务；默认阈值保证物理链路本身必然通过。"""
    return ServiceRequest(
        service_id=1,
        source_id=0,
        destination_id=1,
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=requirement_db,
        launch_power_w=power_w,
    )


def _load_comb(state: NetworkState, num_channels: int) -> None:
    """把除候选信道 0 以外的所有信道装满，让占用带覆盖整个算例波段。

    直接写 ``LinkState`` 而不注册业务：本测试只关心链路噪声与有效性，
    ``_collect_affected_services`` 因此不会重新评估数百条旧业务。
    """
    link_state = state.get_link_state(0, 1)
    for channel_idx in range(1, num_channels):
        link_state.allocate(channel_idx, 1e-3, service_id=channel_idx)


def _run(
    case: BenchCase,
    mode: str,
    *,
    pumps=None,
    policy_kwargs: dict | None = None,
    load_comb: bool = False,
    occupy_neighbor: bool = False,
) -> tuple[bool, np.ndarray | None, LinkValidity, H3LinkValidityProvider]:
    """在一个全新状态上跑一次候选分配，返回判定、SNR、报告与 provider。"""
    grid, _fiber, topology, evaluator, state = _network(case)
    if load_comb:
        _load_comb(state, grid.num_channels)
    if occupy_neighbor:
        state.get_link_state(0, 1).allocate(1, 1e-3, service_id=0)
    model = _h3_model(case, pumps=pumps)
    provider = H3LinkValidityProvider(model)
    policy = LinkValidityPolicy(mode=mode, **(policy_kwargs or {}))
    validator = _validator(case, topology, grid, evaluator, policy=policy, provider=provider)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)
    return ok, snr, validator.last_link_validity[(0, 1)], provider


# ---------------------------------------------------------------------------
# 六泵：泵在占用带内必须被拒绝，C+L 保持可用
# ---------------------------------------------------------------------------


def test_in_band_six_pump_is_rejected_and_flag_mode_is_audit_only() -> None:
    """E+S+C+L 六泵：reject 拒绝分配，flag 保留点值但报告无效。

    1505 nm 泵（199.2 THz）落在满载 E+S+C+L 的占用带 184.4–220.4 THz 内，
    Y 的硬有效性门把全部活跃信道判为 ``valid=False``。该信号必须经
    ``H3LinkValidityProvider`` 原样进入 ``LinkValidity``，再由策略拒绝候选
    信道；``flag`` 只审计，物理 SNR 与关闭有效性时逐位相同。
    """
    case = _case("E+S+C+L", 100e9)

    # 基线：不接入有效性。
    grid, _fiber, topology, evaluator, baseline_state = _network(case)
    _load_comb(baseline_state, grid.num_channels)
    baseline = _validator(case, topology, grid, evaluator)
    baseline_ok, baseline_snr = baseline.verify_allocation(_request(), [0, 1], 0, baseline_state)
    assert baseline_ok is True
    assert baseline_snr is not None

    # flag：判定不变，报告无效。
    flag_ok, flag_snr, report, _provider = _run(case, "flag", pumps=_pumps(), load_comb=True)
    assert flag_ok is True
    assert flag_snr is not None
    assert np.array_equal(flag_snr, baseline_snr)
    assert report.num_channels == case.num_channels
    assert bool(np.all(report.active))
    assert not bool(np.any(report.valid)), "泵-信号带重叠必须把所有活跃信道标为无效"
    assert bool(np.all(report.flagged))
    assert bool(np.all(np.isinf(report.margin_db)))
    assert report.dense_grid_overlap is False

    # reject：候选信道被压成 -inf，分配被拒绝。
    reject_ok, reject_snr, reject_report, _provider = _run(case, "reject", pumps=_pumps(), load_comb=True)
    assert reject_ok is False
    assert reject_snr is None
    assert not bool(np.any(reject_report.valid))


def test_c_l_six_pump_stays_usable_end_to_end() -> None:
    """C+L 六泵：泵全在带外，flag / reject 都不得拒绝候选信道。"""
    case = _case("C+L", 75e9)

    grid, _fiber, topology, evaluator, baseline_state = _network(case)
    _load_comb(baseline_state, grid.num_channels)
    baseline = _validator(case, topology, grid, evaluator)
    baseline_ok, baseline_snr = baseline.verify_allocation(_request(), [0, 1], 0, baseline_state)
    assert baseline_ok is True
    assert baseline_snr is not None

    for mode in ("flag", "reject"):
        ok, snr, report, provider = _run(case, mode, pumps=_pumps(), load_comb=True)
        assert ok is True, mode
        assert snr is not None
        assert np.array_equal(snr, baseline_snr), mode
        assert bool(np.all(report.valid)), mode
        assert bool(np.all(np.isfinite(report.margin_db))), mode
        assert report.dense_grid_overlap is False, mode
        # 泵与占用带的最小偏移为正：1505 nm 泵在 195.9 THz 之上约 3.3 THz。
        assert provider.model.last_span_validity[0].raman_pump_min_offset_hz > 0.0, mode


# ---------------------------------------------------------------------------
# 密集栅格：C 50 / 37.5 GHz 拒绝，75 / 100 GHz 不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spacing_hz", "dense"),
    [(50e9, True), (37.5e9, True), (75e9, False), (100e9, False)],
)
def test_dense_grid_reject_only_blocks_dense_c_grids(spacing_hz: float, dense: bool) -> None:
    """密集栅格动作只改可用性：50/37.5 GHz 拒绝，75/100 GHz 逐位不变。"""
    case = _dense_case(spacing_hz)

    grid, _fiber, topology, evaluator, baseline_state = _network(case)
    baseline_state.get_link_state(0, 1).allocate(1, 1e-3, service_id=0)
    baseline = _validator(case, topology, grid, evaluator)
    baseline_ok, baseline_snr = baseline.verify_allocation(_request(), [0, 1], 0, baseline_state)
    assert baseline_ok is True
    assert baseline_snr is not None and np.isfinite(baseline_snr[0])

    # 默认动作 + reject：密集栅格仍然可用，证明拒绝来自动作而不是 mode。
    default_ok, default_snr, default_report, _provider = _run(case, "reject", occupy_neighbor=True)
    assert default_ok is True
    assert default_snr is not None and np.array_equal(default_snr, baseline_snr)
    assert default_report.dense_grid_overlap is dense

    policy_kwargs = {"dense_grid_action": "reject"}
    ok, snr, report, _provider = _run(case, "reject", policy_kwargs=policy_kwargs, occupy_neighbor=True)
    assert report.dense_grid_overlap is dense
    assert bool(report.valid[0]) is True
    effective = report.effective_valid(LinkValidityPolicy(mode="reject", dense_grid_action="reject"))
    assert bool(effective[0]) is (not dense)

    if dense:
        assert ok is False
        assert snr is None
    else:
        assert ok is True
        assert snr is not None and np.array_equal(snr, baseline_snr)


# ---------------------------------------------------------------------------
# provider 审计：valid / dense_grid_overlap 必须原样搬运
# ---------------------------------------------------------------------------


def test_provider_transports_the_budget_valid_and_dense_flags() -> None:
    """``H3LinkValidityProvider`` 的报告与同一预算逐字段一致。

    该用例直接比较 provider 输出与 ``estimate_h3_error_budget`` 的派生结果：
    ``valid`` 必须是 ``报告 valid & active & 有限 margin``，``dense_grid_overlap``
    与 ``flagged`` 必须原样搬运。任何搬运缺口都会在这里先暴露，而不是等到
    网络判定才表现为“策略失效”。
    """
    case = _dense_case(50e9)
    grid = grid_for(case)
    fiber = build_path(case).fiber_configs[0]
    model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=case.symbol_rate_hz)
    provider = H3LinkValidityProvider(model)

    power = np.zeros(grid.num_channels, dtype=np.float64)
    power[0] = 1e-3
    power[1] = 1e-3
    report = provider((0, 1), fiber, None, power, case.span_length_km * 1e3)

    path = link_optical_path(fiber, None, case.span_length_km * 1e3)
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)

    expected_valid = budget.validity.valid & budget.validity.active & np.isfinite(budget.margin_db)
    assert np.array_equal(report.valid, expected_valid)
    assert np.array_equal(report.flagged, budget.validity.flagged)
    assert np.array_equal(report.active, budget.validity.active)
    assert report.dense_grid_overlap is budget.validity.dense_grid_overlap is True


# ---------------------------------------------------------------------------
# 缓存口径：同一 NetworkState 交替 total / per_pol
# ---------------------------------------------------------------------------


def test_alternating_validators_do_not_mix_cache_coordinates() -> None:
    """同一状态交替 total / per_pol 验证器：缓存重算、SNR 一致、报告独立。"""
    case = _dense_case(50e9)
    grid, _fiber, topology, total_evaluator, state = _network(case)
    state.get_link_state(0, 1).allocate(1, 1e-3, service_id=0)

    total_model = _h3_model(case)
    per_pol_model = _h3_model(case, per_pol=True)

    total = _validator(
        case,
        topology,
        grid,
        total_evaluator,
        policy=LinkValidityPolicy(mode="flag"),
        provider=H3LinkValidityProvider(total_model),
    )
    per_pol = _validator(
        case,
        topology,
        grid,
        _PerPolISRSGN(grid, build_path(case).fiber_configs[0], case.symbol_rate_hz),
        policy=LinkValidityPolicy(mode="flag"),
        provider=H3LinkValidityProvider(per_pol_model, input_power_convention="per_pol"),
    )

    ok_total, snr_total = total.verify_allocation(_request(), [0, 1], 0, state)
    total_cache = state.get_link_state(0, 1).noise_cache
    assert ok_total and snr_total is not None and total_cache is not None
    assert total_cache.launch_power_convention == "total"
    assert total_cache.validity is not None

    ok_per, snr_per = per_pol.verify_allocation(_request(), [0, 1], 0, state)
    per_cache = state.get_link_state(0, 1).noise_cache
    assert ok_per and snr_per is not None and per_cache is not None
    assert per_cache.launch_power_convention == "per_pol"
    assert per_cache is not total_cache, "跨口径旧缓存被静默复用"
    assert np.array_equal(snr_total, snr_per)

    # per_pol 缓存每个分量恰为 total 的一半，证明重算发生在同一物理功率上。
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(per_cache, name), 0.5 * getattr(total_cache, name)), name

    # 切回 total：再次重算，数值回到第一次的逐位结果。
    ok_back, snr_back = total.verify_allocation(_request(), [0, 1], 0, state)
    back_cache = state.get_link_state(0, 1).noise_cache
    assert ok_back and snr_back is not None and back_cache is not None
    assert back_cache.launch_power_convention == "total"
    assert back_cache is not per_cache
    assert np.array_equal(snr_back, snr_total)
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(back_cache, name), getattr(total_cache, name)), name

    # 两次 total 使用同一 provider 缓存，因此报告对象复用；per_pol 报告独立。
    assert total_cache.validity is back_cache.validity
    assert per_cache.validity is not total_cache.validity
    assert np.array_equal(per_cache.validity.valid, total_cache.validity.valid)
    assert per_cache.validity.dense_grid_overlap is total_cache.validity.dense_grid_overlap is True


# ---------------------------------------------------------------------------
# 默认路径：诊断不参与
# ---------------------------------------------------------------------------


def test_default_validator_never_attaches_a_validity_report() -> None:
    """默认 ``off`` 验证器不构造 provider、不附带报告，诊断不在热路径上。"""
    case = _dense_case(100e9)
    grid, _fiber, topology, evaluator, state = _network(case)
    state.get_link_state(0, 1).allocate(1, 1e-3, service_id=0)
    validator = _validator(case, topology, grid, evaluator)

    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)
    assert ok is True
    assert snr is not None
    assert validator.validity_provider is None
    assert validator.last_link_validity == {}
    cache = state.get_link_state(0, 1).noise_cache
    assert cache is not None
    assert cache.validity is None
