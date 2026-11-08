"""任务 M：informational tilt soft margin 接入 RWA/QoT 的验收测试。

覆盖：

* 显式映射公式：knee（10 dB）以下为 0，15 dB 硬阈值边界非零，30 dB 到满量程，
  且随倾斜单调不减；
* H3 误差预算把 ``SpanValidityReport.tilt_confidence_informational`` 带到
  ``ValidityReport``，``LinkValidity`` 再映射成逐信道软 margin；
* 默认 ``off`` 逐位不变；``flag`` / ``reject`` / ``degrade`` 语义不变；
* 只有 ``mode="soft"`` 扣软 margin，且不改 ``valid`` / ``flagged``；
* 非有限 SNR 在 soft 下同样被压成 ``-inf``；
* 链路级 / 业务级聚合只累计有效链路，不串状态；
* 不同占用谱（平坦 vs 15 dB 倾斜）产生不同的指纹、informational 分数与软
  margin。
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.pcfm import (
    POWER_TILT_VALIDITY_DB,
    TILT_INFORMATIONAL_KNEE_DB,
    TILT_INFORMATIONAL_ZERO_DB,
    informational_tilt_confidence,
)
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
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
    TILT_SOFT_MARGIN_FULL_DB,
    ChannelValidity,
    H3LinkValidityProvider,
    InvalidChannelPolicy,
    LinkValidity,
    LinkValidityPolicy,
    occupancy_fingerprint,
    tilt_soft_margin_db,
)
from network_nlo_eval.simulation.traffic import ServiceRequest

CASE = build_case("scaling_40ch")
LINK_KEY: LinkKey = (0, 1)


# ---------------------------------------------------------------------------
# 可控报告与最小 RWA 栈
# ---------------------------------------------------------------------------


def _validity(
    n: int,
    *,
    invalid: tuple[int, ...] = (),
    margin_db: float = 2.0,
    soft_margin_db: float = 0.0,
    tilt_confidence: float = 1.0,
    link_key: LinkKey | None = None,
) -> LinkValidity:
    """构建一份可控报告；``invalid`` 里的信道判无效且 margin 为 inf。"""
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
        fingerprint="stub",
        link_key=link_key,
        tilt_confidence_informational=tilt_confidence,
        soft_margin_db=np.full(n, soft_margin_db, dtype=np.float64),
    )


def _network(channels: int = 4) -> tuple[NetworkTopology, SpectrumGrid, MultiBandISRSGN, NetworkState]:
    """两节点单链路的最小 RWA 栈。"""
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


def _request(channel_snr_requirement: float = 0.0) -> ServiceRequest:
    """单信道业务请求；默认阈值 0 dB 让基线必然通过。"""
    return ServiceRequest(
        service_id=1,
        source_id=0,
        destination_id=1,
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=channel_snr_requirement,
        launch_power_w=1e-3,
    )


class _StubProvider:
    """按链路返回固定报告的 provider。"""

    def __init__(self, validity: LinkValidity) -> None:
        self.validity = validity

    def __call__(self, link_key, fiber_config, edfa_config, power_profile_w, span_length_m) -> LinkValidity:
        return self.validity


# ---------------------------------------------------------------------------
# 映射公式
# ---------------------------------------------------------------------------


def test_tilt_soft_margin_endpoints_and_monotonicity() -> None:
    """Knee 以下恒 0，15 dB 边界非零，30 dB 到满量程，且单调不减。"""
    tilts = np.array([0.0, 5.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0, 40.0])
    scores = np.array([informational_tilt_confidence(value) for value in tilts])
    margins = tilt_soft_margin_db(scores)

    assert np.all(margins >= 0.0)
    assert np.all(margins <= TILT_SOFT_MARGIN_FULL_DB + 1e-15)
    assert np.all(np.diff(margins) >= -1e-15)
    assert margins[tilts <= TILT_INFORMATIONAL_KNEE_DB].tolist() == [0.0, 0.0, 0.0]
    assert TILT_INFORMATIONAL_KNEE_DB == pytest.approx(10.0)
    # 15 dB 恰好是硬 flag 边界：硬判定不变，但软 margin 已经非零。
    boundary = float(tilt_soft_margin_db(informational_tilt_confidence(POWER_TILT_VALIDITY_DB)))
    assert POWER_TILT_VALIDITY_DB == pytest.approx(15.0)
    assert boundary == pytest.approx(0.15625)
    assert boundary > 0.0
    assert informational_tilt_confidence(POWER_TILT_VALIDITY_DB) < 1.0
    assert float(tilt_soft_margin_db(informational_tilt_confidence(TILT_INFORMATIONAL_ZERO_DB))) == pytest.approx(
        TILT_SOFT_MARGIN_FULL_DB
    )


def test_tilt_soft_margin_handles_scalars_nan_and_bad_scale() -> None:
    """标量输入可用；NaN 置信度按最坏情况给满量程；非法满量程报错。"""
    assert float(tilt_soft_margin_db(1.0)) == 0.0
    assert float(tilt_soft_margin_db(0.0)) == pytest.approx(TILT_SOFT_MARGIN_FULL_DB)
    assert float(tilt_soft_margin_db(float("nan"))) == pytest.approx(TILT_SOFT_MARGIN_FULL_DB)
    assert float(tilt_soft_margin_db(2.0, full_scale_db=2.0)) == 0.0
    with pytest.raises(ValueError, match="full_scale_db"):
        tilt_soft_margin_db(0.5, full_scale_db=-1.0)
    with pytest.raises(ValueError, match="full_scale_db"):
        tilt_soft_margin_db(0.5, full_scale_db=float("inf"))


# ---------------------------------------------------------------------------
# H3 预算 → LinkValidity
# ---------------------------------------------------------------------------


def test_h3_budget_carries_informational_score_without_changing_hard_validity() -> None:
    """平坦谱无软 margin；15 dB 倾斜有软 margin，但 valid/flagged 仍按硬阈值。"""
    path = build_path(CASE)
    model = make_model("pcfm_h3_raman", CASE)

    flat_power = launch_profile(CASE)
    flat_result = model.evaluate_path(path, flat_power)
    flat_budget = estimate_h3_error_budget(model, path, flat_power, flat_result)
    flat_report = LinkValidity.from_budget(flat_budget)
    assert flat_budget.validity.tilt_confidence_informational == pytest.approx(1.0)
    assert np.all(flat_report.soft_margin_db == 0.0)
    assert np.all(flat_report.tilt_confidence_informational == 1.0)

    tilted_power = launch_profile(CASE, tilt_db=POWER_TILT_VALIDITY_DB)
    tilted_result = model.evaluate_path(path, tilted_power)
    tilted_budget = estimate_h3_error_budget(model, path, tilted_power, tilted_result)
    tilted_report = LinkValidity.from_budget(tilted_budget)

    expected_score = informational_tilt_confidence(POWER_TILT_VALIDITY_DB)
    expected_margin = float(tilt_soft_margin_db(expected_score))
    assert tilted_budget.validity.tilt_confidence_informational == pytest.approx(expected_score, abs=1e-12)
    assert np.all(tilted_report.tilt_confidence_informational == pytest.approx(expected_score, abs=1e-12))
    assert np.all(tilted_report.soft_margin_db == pytest.approx(expected_margin, abs=1e-12))

    # 信息分数不是 hard valid：15 dB 边界不 flag、仍然 valid。
    assert not bool(np.any(tilted_budget.validity.flagged))
    assert bool(np.all(tilted_budget.validity.valid))
    assert bool(np.all(tilted_report.valid))
    assert not bool(np.any(tilted_report.flagged))


def test_soft_margin_is_monotone_in_measured_tilt() -> None:
    """实测倾斜越大，链路软 margin 单调不减；10 dB 及以下为 0。"""
    path = build_path(CASE)
    model = make_model("pcfm_h3_raman", CASE)
    tilts = (0.0, 5.0, 10.0, 15.0, 20.0)
    margins = []
    for tilt in tilts:
        power = launch_profile(CASE, tilt_db=tilt)
        result = model.evaluate_path(path, power)
        budget = estimate_h3_error_budget(model, path, power, result)
        margins.append(LinkValidity.from_budget(budget).soft_margin_at(0))

    assert np.all(np.diff(margins) >= -1e-12)
    assert margins[0] == 0.0
    assert margins[3] > 0.0
    assert margins[4] > margins[3]


# ---------------------------------------------------------------------------
# 策略语义
# ---------------------------------------------------------------------------


def test_off_is_bit_identical_and_only_soft_applies_the_tilt_margin() -> None:
    """默认 off 原样返回；flag/reject/degrade 不受软 margin 影响；soft 才扣。"""
    report = _validity(4, invalid=(2,), margin_db=2.0, soft_margin_db=0.5)
    snr = np.array([20.0, 21.0, 22.0, 23.0])

    assert np.array_equal(report.adjusted_snr_db(snr), snr)
    assert np.array_equal(report.adjusted_snr_db(snr, LinkValidityPolicy()), snr)
    assert np.array_equal(report.adjusted_snr_db(snr, LinkValidityPolicy(mode="flag")), snr)

    rejected = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="reject"))
    assert np.isneginf(rejected[2])
    assert np.array_equal(rejected[[0, 1, 3]], snr[[0, 1, 3]])

    degraded = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="degrade"))
    assert degraded[0] == pytest.approx(18.0)
    assert degraded[1] == pytest.approx(19.0)
    assert np.isneginf(degraded[2])
    assert degraded[3] == pytest.approx(21.0)

    soft = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="soft"))
    assert soft[0] == pytest.approx(19.5)
    assert soft[1] == pytest.approx(20.5)
    assert np.isneginf(soft[2])
    assert soft[3] == pytest.approx(22.5)

    # 策略只读报告，不改变 hard 判定。
    assert list(report.valid) == [True, True, False, True]
    assert list(report.flagged) == [False, False, True, False]


def test_soft_mode_never_accepts_a_non_finite_snr() -> None:
    """Soft 下 NaN / -inf 仍为 -inf；+inf 保持 +inf。"""
    report = _validity(4, soft_margin_db=0.5)
    snr = np.array([20.0, np.nan, -np.inf, np.inf])
    adjusted = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="soft"))

    assert adjusted[0] == pytest.approx(19.5)
    assert np.isneginf(adjusted[1])
    assert np.isneginf(adjusted[2])
    assert np.isposinf(adjusted[3])


def test_soft_mode_requires_an_explicit_opt_in() -> None:
    """Soft 与既有模式一样必须显式写字符串；未知策略仍报错。"""
    assert InvalidChannelPolicy("soft") is InvalidChannelPolicy.SOFT
    assert LinkValidityPolicy(mode="soft").enabled
    assert not LinkValidityPolicy().enabled
    with pytest.raises(ValueError, match="unknown validity policy"):
        LinkValidityPolicy(mode="maybe")


# ---------------------------------------------------------------------------
# 聚合与状态隔离
# ---------------------------------------------------------------------------


def test_channel_aggregation_sums_soft_margin_without_leaking_links() -> None:
    """两条链路独立聚合：软 margin 只累计有效链路，输入报告不被改写。"""
    link_a = _validity(4, margin_db=2.0, soft_margin_db=0.2, link_key=(0, 1))
    link_b = _validity(4, invalid=(2,), margin_db=5.0, soft_margin_db=0.4, link_key=(1, 2))
    verdict = ChannelValidity.combine([link_a, link_b], 4)

    assert list(verdict.valid) == [True, True, False, True]
    assert verdict.margin_db == pytest.approx([7.0, 7.0, 2.0, 7.0])
    # 无效链路在信道 2 不贡献软 margin，只剩有效链路的 0.2。
    assert verdict.soft_margin_db == pytest.approx([0.6, 0.6, 0.2, 0.6])

    # 输入报告保持各自原值。
    assert link_a.soft_margin_at(2) == pytest.approx(0.2)
    assert link_b.soft_margin_at(2) == pytest.approx(0.4)
    assert link_b.channel_valid(2) is False

    snr = np.array([20.0, 20.0, 20.0, 20.0])
    adjusted = verdict.adjusted_snr_db(snr, LinkValidityPolicy(mode="soft"))
    assert adjusted[[0, 1, 3]] == pytest.approx([19.4, 19.4, 19.4])
    assert np.isneginf(adjusted[2])

    with pytest.raises(ValueError, match="does not match the verdict"):
        verdict.adjusted_snr_db(np.zeros(2))


def test_channel_aggregation_defaults_soft_margin_to_zero() -> None:
    """没有软 margin 的旧式聚合结果逐位不变。"""
    verdict = ChannelValidity(valid=np.array([True, False]), margin_db=np.array([1.0, 2.0]))
    assert np.array_equal(verdict.soft_margin_db, np.zeros(2))
    with pytest.raises(ValueError):
        verdict.soft_margin_db[0] = 1.0


def test_service_scalar_aggregation_carries_soft_margin_from_valid_links_only() -> None:
    """业务级聚合：软 margin 只来自有效链路；无效链路让整个业务无效。"""
    terms = fold_roadm_terms(ROADMConfig(insertion_loss_db=0.0, add_drop_loss_db=0.0), is_last_hop=True)
    zero = LinkNoiseCache(
        spm=np.full(4, 1e-6),
        xpm=np.full(4, 2e-6),
        ase=np.full(4, 3e-6),
        span_count=1,
    )
    good = replace(zero, validity=_validity(4, margin_db=0.0, soft_margin_db=0.2, link_key=(0, 1)))
    bad = replace(
        zero,
        validity=_validity(4, invalid=(0,), margin_db=0.0, soft_margin_db=0.4, link_key=(1, 2)),
    )

    def detail(caches) -> object:
        return accumulate_service_snr_detail(
            [(terms, cache) for cache in caches],
            channel_idx=0,
            launch_power_w=1e-3,
            frequency_hz=193.1e12,
            channel_bandwidth_hz=50e9,
            is_blocked=lambda _terms: False,
            validity_of=lambda _terms, cache: cache.validity,
        )

    only_good = detail([good])
    assert only_good.valid
    assert only_good.soft_margin_db == pytest.approx(0.2)
    assert only_good.effective_snr_db(LinkValidityPolicy(mode="soft")) == pytest.approx(only_good.snr_db - 0.2)

    both = detail([good, bad])
    assert not both.valid
    assert both.margin_db == pytest.approx(0.0)
    assert both.soft_margin_db == pytest.approx(0.2)
    assert both.invalid_links == ((1, 2),)
    assert np.isneginf(both.effective_snr_db(LinkValidityPolicy(mode="soft")))


def test_physics_overlay_soft_mode_exposes_and_subtracts_soft_margin() -> None:
    """波形叠加在 soft 下扣软 margin 并暴露审计键，输入字典不变。"""
    report = _validity(3, soft_margin_db=0.25)
    metrics = {"snr": np.array([20.0, 21.0, 22.0])}

    overlaid = apply_link_validity(metrics, report, LinkValidityPolicy(mode="soft"))
    assert overlaid["snr"] == pytest.approx([19.75, 20.75, 21.75])
    assert overlaid["soft_margin_db"] == pytest.approx([0.25, 0.25, 0.25])
    assert set(metrics) == {"snr"}

    off = apply_link_validity(metrics, report, LinkValidityPolicy())
    assert set(off) == {"snr"}
    assert np.array_equal(off["snr"], metrics["snr"])


def test_validator_soft_mode_subtracts_only_the_soft_component() -> None:
    """RWA 验证器在 soft 下只把有效信道的 SNR 减去软 margin。"""
    topology, grid, evaluator, state = _network()
    baseline = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    expected_ok, expected_snr = baseline.verify_allocation(_request(), [0, 1], 0, state)
    assert expected_snr is not None

    validator = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        validity_policy=LinkValidityPolicy(mode="soft"),
        validity_provider=_StubProvider(_validity(4, soft_margin_db=0.75)),
    )
    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)
    assert snr is not None
    assert ok is expected_ok

    usable = np.isfinite(expected_snr)
    assert np.allclose(snr[usable], expected_snr[usable] - 0.75)
    assert np.array_equal(snr[~usable], expected_snr[~usable])


# ---------------------------------------------------------------------------
# 不同占用谱
# ---------------------------------------------------------------------------


def test_different_occupied_spectra_produce_different_soft_margins() -> None:
    """平坦谱与 15 dB 倾斜谱：指纹、informational 分数与软 margin 都不同。"""
    model = make_model("pcfm_h3_raman", CASE)
    provider = H3LinkValidityProvider(model)
    fiber = build_path(CASE).fiber_configs[0]
    span_length_m = CASE.span_length_km * 1e3

    flat = launch_profile(CASE)
    tilted = launch_profile(CASE, tilt_db=POWER_TILT_VALIDITY_DB)
    assert occupancy_fingerprint(flat, span_length_m) != occupancy_fingerprint(tilted, span_length_m)

    flat_report = provider(LINK_KEY, fiber, None, flat, span_length_m)
    tilted_report = provider(LINK_KEY, fiber, None, tilted, span_length_m)

    assert flat_report.fingerprint != tilted_report.fingerprint
    assert provider.cache.misses == 2
    assert provider.cache.hits == 0
    assert flat_report.tilt_confidence_informational[0] == pytest.approx(1.0)
    assert np.all(flat_report.soft_margin_db == 0.0)
    assert tilted_report.tilt_confidence_informational[0] < 1.0
    assert np.all(tilted_report.soft_margin_db > 0.0)


# ---------------------------------------------------------------------------
# 字段校验
# ---------------------------------------------------------------------------


def test_optional_soft_margin_fields_validate_and_default_to_no_margin() -> None:
    """可选字段默认无软 margin；越界/非有限值明确报错；数组只读。"""
    report = _validity(3)
    assert np.all(report.tilt_confidence_informational == 1.0)
    assert np.all(report.soft_margin_db == 0.0)

    with pytest.raises(ValueError, match="soft_margin_db"):
        LinkValidity(
            valid=np.ones(3, dtype=bool),
            active=np.ones(3, dtype=bool),
            flagged=np.zeros(3, dtype=bool),
            confidence=np.ones(3),
            margin_db=np.zeros(3),
            model="stub",
            soft_margin_db=np.array([0.0, -1.0, 0.0]),
        )
    with pytest.raises(ValueError, match="tilt_confidence_informational"):
        LinkValidity(
            valid=np.ones(3, dtype=bool),
            active=np.ones(3, dtype=bool),
            flagged=np.zeros(3, dtype=bool),
            confidence=np.ones(3),
            margin_db=np.zeros(3),
            model="stub",
            tilt_confidence_informational=np.array([1.0, 1.5, 1.0]),
        )
    with pytest.raises(ValueError):
        report.soft_margin_db[0] = 1.0
