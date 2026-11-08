"""任务 AA：密集栅格重叠的网络级策略。

第五/七轮矩阵（任务 V）已经证明 C 50 / 37.5 GHz 的 1.4–2.4 dB 乐观偏差来自
GN 非重叠假设失效：提高分辨率、二阶项或边缘求积只能挪动 0.04 dB。任务 U
提供了 opt-in 的 ``dense_grid_penalty_db``（降级 margin），但网络层仍把
``dense_grid_overlap`` 当成普通 flag。

本文件钉住任务 AA 的显式动作：

* ``LinkValidityPolicy.dense_grid_action`` 默认 ``"flag"``，默认路径与接入前
  逐位一致，``dense_grid_overlap`` 绝不默认变成 invalid；
* ``"reject"`` 只改可用性（``effective_valid`` 为 ``False``），不改点值；
  再走 ``mode`` 的既有语义：``flag`` 只审计，``reject`` / ``degrade`` /
  ``soft`` 把信道压成 ``-inf``；
* 75 / 100 GHz 不触发密集栅格，``"reject"`` 对它们没有影响；
* 密集栅格动作与 ``dense_grid_penalty_db`` 正交：penalty 只动 ``degrade``
  的 margin，动作只动可用性；
* 链路报告、路径聚合（``ChannelValidity`` / ``ServiceQoT``）、波形叠加与
  ``QoTValidator`` 都走同一份策略语义。

算例全部由 ``benchmarks.cases`` 自包含构造，不读 ``results/`` 或 ``data/``。
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.uncertainty import H3UncertaintyConfig, estimate_h3_error_budget
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics.qot_checker import apply_link_validity
from network_nlo_eval.rwa.qot_checker import (
    QoTValidator,
    accumulate_service_snr_detail,
    fold_roadm_terms,
)
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    DenseGridAction,
    H3LinkValidityProvider,
    LinkValidity,
    LinkValidityPolicy,
)
from network_nlo_eval.simulation.traffic import ServiceRequest

DENSE_SPACINGS = (50e9, 37.5e9)
VALIDATED_SPACINGS = (75e9, 100e9)


def _case(spacing_hz: float) -> BenchCase:
    """C 波段、64 GBaud、给定栅格间隔的 1 跨算例（与任务 V 同源）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"task_aa_dense_{spacing_hz / 1e9:g}",
        bands="C",
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


@pytest.fixture(scope="module")
def reports() -> dict[float, dict[str, object]]:
    """四个栅格间隔各跑一次 H3：报告、点值、预算与算例材料。"""
    out: dict[float, dict[str, object]] = {}
    for spacing_hz in (*DENSE_SPACINGS, *VALIDATED_SPACINGS):
        case = _case(spacing_hz)
        path = build_path(case)
        power = launch_profile(case, 6.0)
        model = make_model("pcfm_h3_raman", case)
        result = model.evaluate_path(path, power)
        budget = estimate_h3_error_budget(model, path, power, result)
        report = LinkValidity.from_budget(budget, link_key=(0, 1))
        out[spacing_hz] = {
            "case": case,
            "path": path,
            "power": power,
            "model": model,
            "result": result,
            "budget": budget,
            "report": report,
        }
    return out


def _legacy_adjusted(report: LinkValidity, snr_db: np.ndarray, mode: str) -> np.ndarray:
    """任务 AA 之前的 ``_effective_snr`` 语义，用于逐位回归。

    这里刻意照抄旧公式，而不是调用生产代码：默认 ``dense_grid_action="flag"``
    必须与它 ``np.array_equal``。
    """
    values = np.asarray(snr_db, dtype=np.float64)
    if mode == "off":
        return values
    usable = ~np.isnan(values) & ~np.isneginf(values)
    if mode == "flag":
        return np.where(usable, values, -np.inf)
    accept = np.asarray(report.valid, dtype=bool) & usable
    if mode == "reject":
        return np.where(accept, values, -np.inf)
    if mode == "soft":
        return np.where(accept, values - report.soft_margin_db, -np.inf)
    penalty = np.where(accept & np.isfinite(report.margin_db), report.margin_db, 0.0)
    return np.where(accept, values - penalty, -np.inf)


def _stub_validity(
    n: int = 4,
    *,
    dense: bool = False,
    invalid: tuple[int, ...] = (),
    margin_db: float = 1.0,
) -> LinkValidity:
    """构造可控的链路报告：``dense`` 控制密集栅格标志，``invalid`` 控制无效信道。"""
    valid = np.ones(n, dtype=bool)
    valid[list(invalid)] = False
    margin = np.full(n, margin_db, dtype=np.float64)
    margin[~valid] = np.inf
    return LinkValidity(
        valid=valid,
        active=np.ones(n, dtype=bool),
        flagged=~valid,
        confidence=np.where(valid, 1.0, 0.5),
        margin_db=margin,
        model="stub",
        span_count=1,
        dense_grid_overlap=dense,
    )


# ---------------------------------------------------------------------------
# 默认 flag 路径逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", (*DENSE_SPACINGS, *VALIDATED_SPACINGS))
@pytest.mark.parametrize("mode", ["off", "flag", "reject", "degrade", "soft"])
def test_default_dense_action_reproduces_the_legacy_snr(reports, spacing_hz: float, mode: str) -> None:
    """默认动作下，五个 mode 的有效 SNR 与任务 AA 之前逐位相同。"""
    item = reports[spacing_hz]
    report: LinkValidity = item["report"]  # type: ignore[assignment]
    snr = np.asarray(item["result"].snr_db, dtype=np.float64)  # type: ignore[union-attr]

    policy = LinkValidityPolicy(mode=mode)
    assert policy.dense_grid_action is DenseGridAction.FLAG
    adjusted = report.adjusted_snr_db(snr, policy)
    assert np.array_equal(adjusted, _legacy_adjusted(report, snr, mode), equal_nan=True)

    # 显式写出默认值也必须一样，避免"默认值"与"显式值"两条路径漂移。
    explicit = report.adjusted_snr_db(snr, LinkValidityPolicy(mode=mode, dense_grid_action="flag"))
    assert np.array_equal(adjusted, explicit, equal_nan=True)


@pytest.mark.parametrize("spacing_hz", (*DENSE_SPACINGS, *VALIDATED_SPACINGS))
def test_default_action_never_changes_validity_or_point_values(reports, spacing_hz: float) -> None:
    """密集栅格默认只 flag：``valid`` / margin / 点值 SNR 都不动。"""
    item = reports[spacing_hz]
    report: LinkValidity = item["report"]  # type: ignore[assignment]
    budget = item["budget"]  # type: ignore[assignment]
    result = item["result"]  # type: ignore[assignment]

    policy = LinkValidityPolicy(mode="reject")
    assert np.array_equal(report.effective_valid(policy), report.valid)
    assert np.array_equal(report.valid, np.asarray(budget.validity.valid))  # type: ignore[union-attr]
    assert np.array_equal(
        report.margin_db,
        np.asarray(budget.margin_db),  # type: ignore[union-attr]
    )
    assert np.all(np.isfinite(np.asarray(result.snr_db)))  # type: ignore[union-attr]

    if spacing_hz in DENSE_SPACINGS:
        assert report.dense_grid_overlap
        assert bool(np.all(report.valid))
    else:
        assert not report.dense_grid_overlap


# ---------------------------------------------------------------------------
# reject：密集 C 栅格不可用
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", DENSE_SPACINGS)
def test_dense_reject_makes_c_dense_grids_unavailable(reports, spacing_hz: float) -> None:
    """``reject`` 把密集栅格的全部发射信道判无效，点值仍保留在报告里。"""
    item = reports[spacing_hz]
    report: LinkValidity = item["report"]  # type: ignore[assignment]
    snr = np.asarray(item["result"].snr_db, dtype=np.float64)  # type: ignore[union-attr]
    assert report.dense_grid_overlap

    reject = LinkValidityPolicy(mode="reject", dense_grid_action="reject")
    assert not bool(np.any(report.effective_valid(reject)))
    assert np.all(np.isneginf(report.adjusted_snr_db(snr, reject)))

    # degrade / soft 读到同一个有效 mask，因此也把信道压成 -inf；它们不再对
    # 一个不可用信道扣 margin 或软 margin。
    for mode in ("degrade", "soft"):
        policy = LinkValidityPolicy(mode=mode, dense_grid_action="reject")
        assert np.all(np.isneginf(report.adjusted_snr_db(snr, policy)))

    # flag 只审计：点值不变，但可用性已经为 False。
    flagged_policy = LinkValidityPolicy(mode="flag", dense_grid_action="reject")
    assert np.array_equal(report.adjusted_snr_db(snr, flagged_policy), snr)
    assert not bool(np.any(report.effective_valid(flagged_policy)))

    # 点值/报告没有被策略改写。
    assert np.array_equal(report.valid, np.asarray(item["budget"].validity.valid))  # type: ignore[union-attr]
    assert np.array_equal(snr, np.asarray(item["result"].snr_db))  # type: ignore[union-attr]


@pytest.mark.parametrize("spacing_hz", VALIDATED_SPACINGS)
def test_dense_reject_does_not_touch_validated_spacings(reports, spacing_hz: float) -> None:
    """75 / 100 GHz 不触发密集栅格，``reject`` 与默认策略逐位相同。"""
    item = reports[spacing_hz]
    report: LinkValidity = item["report"]  # type: ignore[assignment]
    snr = np.asarray(item["result"].snr_db, dtype=np.float64)  # type: ignore[union-attr]
    assert not report.dense_grid_overlap

    policy = LinkValidityPolicy(mode="reject", dense_grid_action="reject")
    assert np.array_equal(report.effective_valid(policy), report.valid)
    assert np.array_equal(
        report.adjusted_snr_db(snr, policy),
        report.adjusted_snr_db(snr, LinkValidityPolicy(mode="reject")),
    )
    assert np.all(np.isfinite(report.adjusted_snr_db(snr, policy)))


@pytest.mark.parametrize("mode", ["off", "flag", "reject", "degrade", "soft"])
def test_mode_and_dense_action_combination_matrix(reports, mode: str) -> None:
    """``mode`` × ``dense_grid_action`` 的组合语义显式而唯一。

    * ``off``：不接入，密集动作被忽略，返回物理 SNR；
    * ``flag``：只审计，动作只反映在 ``effective_valid``；
    * ``reject`` / ``degrade`` / ``soft``：动作把信道变成不可用，SNR 为 ``-inf``。
    """
    item = reports[50e9]
    report: LinkValidity = item["report"]  # type: ignore[assignment]
    snr = np.asarray(item["result"].snr_db, dtype=np.float64)  # type: ignore[union-attr]

    policy = LinkValidityPolicy(mode=mode, dense_grid_action="reject")
    adjusted = report.adjusted_snr_db(snr, policy)
    if mode == "off":
        assert np.array_equal(adjusted, snr)
        assert np.array_equal(report.effective_valid(policy), report.valid)
    elif mode == "flag":
        assert np.array_equal(adjusted, snr)
        assert not bool(np.any(report.effective_valid(policy)))
    else:
        assert np.all(np.isneginf(adjusted))
        assert not bool(np.any(report.effective_valid(policy)))


# ---------------------------------------------------------------------------
# 与 U 的 dense_grid_penalty_db 正交
# ---------------------------------------------------------------------------


def test_dense_reject_is_orthogonal_to_the_penalty(reports) -> None:
    """密集栅格 penalty 只改 degrade 的 margin；动作只改可用性，两者互不替代。"""
    item = reports[50e9]
    report: LinkValidity = item["report"]  # type: ignore[assignment]
    model = item["model"]  # type: ignore[assignment]
    path = item["path"]  # type: ignore[assignment]
    power = item["power"]  # type: ignore[assignment]
    result = item["result"]  # type: ignore[assignment]
    snr = np.asarray(result.snr_db, dtype=np.float64)  # type: ignore[union-attr]

    tuned_budget = estimate_h3_error_budget(
        model,  # type: ignore[arg-type]
        path,  # type: ignore[arg-type]
        power,  # type: ignore[arg-type]
        result,  # type: ignore[arg-type]
        config=H3UncertaintyConfig(dense_grid_penalty_db=1.1),
    )
    tuned = LinkValidity.from_budget(tuned_budget)

    # 1.1 dB 是降级 margin：只加在 margin 上，valid / 密集标志不变。
    assert tuned.dense_grid_overlap is report.dense_grid_overlap is True
    assert np.allclose(tuned.margin_db, report.margin_db + 1.1)
    assert np.array_equal(tuned.valid, report.valid)

    # flag 动作下，degrade 的差值就是 penalty，可用性不受影响。
    base = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="degrade"))
    tuned_degraded = tuned.adjusted_snr_db(snr, LinkValidityPolicy(mode="degrade"))
    assert np.allclose(base - tuned_degraded, 1.1)
    assert np.array_equal(
        report.effective_valid(LinkValidityPolicy(mode="degrade")),
        tuned.effective_valid(LinkValidityPolicy(mode="degrade")),
    )

    # reject 动作下，两者都不可用；penalty 无法把不可用变回可用。
    reject = LinkValidityPolicy(mode="degrade", dense_grid_action="reject")
    assert np.array_equal(report.effective_valid(reject), tuned.effective_valid(reject))
    assert not bool(np.any(report.effective_valid(reject)))
    assert np.all(np.isneginf(report.adjusted_snr_db(snr, reject)))
    assert np.all(np.isneginf(tuned.adjusted_snr_db(snr, reject)))


def test_unknown_dense_action_is_rejected() -> None:
    """密集栅格动作必须显式且合法，不允许静默退回默认。"""
    with pytest.raises(ValueError, match="unknown dense grid action"):
        LinkValidityPolicy(dense_grid_action="maybe")


# ---------------------------------------------------------------------------
# 路径聚合与使用方
# ---------------------------------------------------------------------------


def test_path_aggregation_propagates_the_dense_flag() -> None:
    """任一跳是密集栅格，整条路径就按不可用处理；非密集路径不受影响。"""
    good = _stub_validity(dense=False, margin_db=2.0)
    plain = ChannelValidity.combine([good], 4)
    assert not plain.dense_grid_overlap
    policy = LinkValidityPolicy(mode="reject", dense_grid_action="reject")
    assert np.array_equal(plain.effective_valid(policy), plain.valid)

    mixed = ChannelValidity.combine([good, _stub_validity(dense=True)], 4)
    assert mixed.dense_grid_overlap
    snr = np.full(4, 20.0)
    assert np.all(np.isneginf(mixed.adjusted_snr_db(snr, policy)))
    assert np.all(np.isfinite(mixed.adjusted_snr_db(snr, LinkValidityPolicy(mode="reject"))))


def test_scalar_service_aggregation_carries_the_dense_flag() -> None:
    """``ServiceQoT`` 带上密集标志，标量可用性走同一份策略。"""
    terms = fold_roadm_terms(ROADMConfig(insertion_loss_db=0.0, add_drop_loss_db=0.0), is_last_hop=True)
    zero = LinkNoiseCache(
        spm=np.zeros(2),
        xpm=np.zeros(2),
        ase=np.full(2, 1e-6),
        span_count=1,
    )
    dense = replace(zero, validity=_stub_validity(2, dense=True))
    plain = replace(zero, validity=_stub_validity(2, dense=False))

    def detail(cache: LinkNoiseCache):
        return accumulate_service_snr_detail(
            [(terms, cache)],
            channel_idx=0,
            launch_power_w=1e-3,
            frequency_hz=193.1e12,
            channel_bandwidth_hz=50e9,
            is_blocked=lambda _terms: False,
            validity_of=lambda _terms, item: item.validity,
        )

    dense_qot = detail(dense)
    assert dense_qot.dense_grid_overlap
    policy = LinkValidityPolicy(mode="reject", dense_grid_action="reject")
    assert np.isneginf(dense_qot.effective_snr_db(policy))
    # 默认动作与 flag 模式都不改标量：可用性只看物理有效性。
    assert dense_qot.effective_snr_db(LinkValidityPolicy(mode="reject")) == dense_qot.snr_db

    plain_qot = detail(plain)
    assert not plain_qot.dense_grid_overlap
    assert plain_qot.effective_snr_db(policy) == plain_qot.snr_db


def test_physics_overlay_reports_effective_validity(reports) -> None:
    """波形叠加的 ``valid`` 用策略口径；flag 下 SNR 不变，reject 下变 -inf。"""
    item = reports[50e9]
    report: LinkValidity = item["report"]  # type: ignore[assignment]
    snr = np.asarray(item["result"].snr_db, dtype=np.float64)  # type: ignore[union-attr]
    metrics = {"snr": snr.copy()}

    audited = apply_link_validity(
        metrics,
        report,
        LinkValidityPolicy(mode="flag", dense_grid_action="reject"),
    )
    assert np.array_equal(audited["snr"], snr)
    assert not bool(np.any(audited["valid"]))

    rejected = apply_link_validity(
        metrics,
        report,
        LinkValidityPolicy(mode="reject", dense_grid_action="reject"),
    )
    assert np.all(np.isneginf(rejected["snr"]))
    assert set(metrics) == {"snr"}
    assert np.array_equal(metrics["snr"], snr)


# ---------------------------------------------------------------------------
# QoTValidator 端到端
# ---------------------------------------------------------------------------


def _network(spacing_hz: float) -> tuple[NetworkTopology, SpectrumGrid, MultiBandISRSGN, NetworkState]:
    """四信道单链路 RWA 栈；信道带宽 64 GHz，因此 50 GHz 间距即重叠。"""
    graph = nx.Graph()
    graph.add_nodes_from([0, 1])
    graph.add_edge(0, 1, weight=80.0)
    fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)
    topology = NetworkTopology(graph, fiber, EDFAConfig(noise_figure_db=5.0), ROADMConfig())
    grid = SpectrumGrid(4, spacing_hz, 193.1e12)
    evaluator = MultiBandISRSGN(grid, fiber, 64e9)
    state = NetworkState(topology, grid)
    return topology, grid, evaluator, state


def _request() -> ServiceRequest:
    """单信道业务；阈值取 0 dB 让基线必然通过。"""
    return ServiceRequest(
        service_id=1,
        source_id=0,
        destination_id=1,
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=0.0,
        launch_power_w=1e-3,
    )


def _validator(
    topology: NetworkTopology,
    grid: SpectrumGrid,
    evaluator: MultiBandISRSGN,
    *,
    policy: LinkValidityPolicy | None = None,
) -> QoTValidator:
    """构造验证器；给出策略时同时挂上真实的 H3 provider。"""
    kwargs = {}
    if policy is not None:
        fiber = topology.get_default_fiber_config()
        model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=64e9)
        kwargs = {"validity_policy": policy, "validity_provider": H3LinkValidityProvider(model)}
    return QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0, **kwargs)


def _occupy_neighbor(state: NetworkState, channel_idx: int = 1) -> None:
    """先占用一个相邻信道，让占用谱至少有两个发射信道。

    ``dense_grid_metrics`` 只在至少两个活跃信道时才可能判密集栅格，而
    ``verify_allocation`` 只把候选业务写进临时谱。没有邻居时，50 GHz 的候选
    业务仍是单信道，不属于密集栅格。
    """
    state.get_link_state(0, 1).allocate(channel_idx, 1e-3, service_id=0)


@pytest.mark.parametrize("spacing_hz, dense", [(50e9, True), (75e9, False)])
def test_validator_dense_reject_only_blocks_dense_grids(spacing_hz: float, dense: bool) -> None:
    """网络级：C 50 GHz 在 reject 下不通过；75 GHz 的判定与 SNR 逐位不变。"""
    topology, grid, evaluator, state = _network(spacing_hz)
    _occupy_neighbor(state)
    baseline = _validator(topology, grid, evaluator)
    expected_ok, expected_snr = baseline.verify_allocation(_request(), [0, 1], 0, state)
    assert expected_ok is True
    assert expected_snr is not None and np.isfinite(expected_snr[0])

    validator = _validator(
        topology,
        grid,
        evaluator,
        policy=LinkValidityPolicy(mode="reject", dense_grid_action="reject"),
    )
    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)
    report = validator.last_link_validity[(0, 1)]
    assert report.dense_grid_overlap is dense

    if dense:
        # 不可用即判定失败；``verify_allocation`` 失败时按约定不返回 SNR 数组。
        assert ok is False
        assert snr is None
        # 同一 provider、同一 mode，只把动作换回默认 "flag"：判定必须恢复通过，
        # 证明拒绝来自密集栅格动作而不是邻居噪声或策略本身。
        default_action = _validator(
            topology,
            grid,
            evaluator,
            policy=LinkValidityPolicy(mode="reject"),
        )
        restored_ok, restored_snr = default_action.verify_allocation(_request(), [0, 1], 0, state)
        assert restored_ok is True
        assert restored_snr is not None and np.array_equal(restored_snr, expected_snr)
    else:
        assert ok is expected_ok
        assert snr is not None and np.array_equal(snr, expected_snr)


def test_validator_flag_with_dense_reject_is_audit_only() -> None:
    """网络级 flag：判定与 SNR 不变，但报告的策略口径可用性为 False。"""
    topology, grid, evaluator, state = _network(50e9)
    _occupy_neighbor(state)
    baseline = _validator(topology, grid, evaluator)
    expected_ok, expected_snr = baseline.verify_allocation(_request(), [0, 1], 0, state)

    policy = LinkValidityPolicy(mode="flag", dense_grid_action="reject")
    validator = _validator(topology, grid, evaluator, policy=policy)
    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)

    assert ok is expected_ok
    assert snr is not None and expected_snr is not None
    assert np.array_equal(snr, expected_snr)
    report = validator.last_link_validity[(0, 1)]
    assert report.dense_grid_overlap
    assert not bool(np.any(report.effective_valid(policy)))
