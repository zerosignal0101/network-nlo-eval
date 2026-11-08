"""网络级 two-tier 有效性接入（任务 H）的验收测试。

覆盖计划文档里任务 H 的验收点：

* 默认路径逐位不变：``LinkNoiseCache`` 不新增数值、``QoTValidator`` 判定不变、
  ``apply_link_validity`` 在 ``off`` 下不新增键；
* 策略显式：未知策略报错、``off`` 之外的策略必须配 provider、非有限 SNR 永不合格；
* opt-in margin 单调：``k_sigma`` 增大不会让链路 margin 变小；
* invalid 信道被正确标记，并按 reject/degrade 被拒绝；
* 链路级聚合不串状态：一条链路的报告不会写进另一条，聚合不改输入；
* 不同占用谱下缓存键与重算行为明确；
* two-tier 默认不调用数值模型，启用后只覆盖 flagged 信道，超预算报错。
"""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey
from network_nlo_eval.models.base import MultiSpanOpticalPath, PathQoTResult, create_model
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.uncertainty import (
    H3UncertaintyConfig,
    estimate_h3_error_budget,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics.qot_checker import apply_link_validity
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.rwa.qot_checker import (
    QoTValidator,
    ServiceQoT,
    accumulate_service_snr,
    accumulate_service_snr_detail,
    fold_roadm_terms,
)
from network_nlo_eval.rwa.two_tier import (
    TwoTierConfig,
    TwoTierEvaluator,
    numerical_dra_factory,
)
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    H3LinkValidityProvider,
    LinkValidity,
    LinkValidityCache,
    LinkValidityPolicy,
    link_optical_path,
    occupancy_fingerprint,
)
from network_nlo_eval.simulation.traffic import ServiceRequest

CASE = build_case("scaling_40ch")
LINK_KEY: LinkKey = (0, 1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _validity(
    n: int,
    *,
    invalid: tuple[int, ...] = (),
    margin_db: float = 2.0,
    link_key: LinkKey | None = None,
) -> LinkValidity:
    """构建一份可控的链路报告：``invalid`` 里的信道判无效。"""
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
    """按信道返回固定报告的 provider，并记录调用参数。"""

    def __init__(self, validity: LinkValidity) -> None:
        self.validity = validity
        self.calls: list[tuple[LinkKey, str]] = []

    def __call__(self, link_key, fiber_config, edfa_config, power_profile_w, span_length_m) -> LinkValidity:
        self.calls.append((link_key, occupancy_fingerprint(power_profile_w, span_length_m)))
        return self.validity


class _StubFallbackModel:
    """返回固定 SNR 的假数值模型，只用于验证两级编排。"""

    def __init__(self, snr_db: np.ndarray) -> None:
        self._snr = np.asarray(snr_db, dtype=np.float64)
        self.calls = 0

    def evaluate_path(self, optical_path, power_in_w) -> PathQoTResult:
        self.calls += 1
        n = self._snr.shape[0]
        zeros = np.zeros(n, dtype=np.float64)
        return PathQoTResult(
            signal_power_w=zeros,
            spm_noise_w=zeros,
            xpm_noise_w=zeros,
            ase_noise_w=zeros,
            snr_db=self._snr,
            span_count=1,
            roadm_count=0,
        )


# ---------------------------------------------------------------------------
# 默认路径逐位不变
# ---------------------------------------------------------------------------


def test_link_noise_cache_keeps_the_four_argument_contract() -> None:
    """新增的 validity 字段默认 None，四参数构造与数值都保持不变。"""
    cache = LinkNoiseCache(spm=np.zeros(2), xpm=np.zeros(2), ase=np.zeros(2), span_count=3)
    assert cache.validity is None
    assert cache.span_count == 3


def test_assemble_link_noise_without_validity_is_bit_identical() -> None:
    """不传 validity 时三个噪声数组与显式 None 完全相同。"""
    evaluator = MultiBandISRSGN(SpectrumGrid(4, 50e9, 193.1e12), FiberSpanConfig(length_km=150.0), 50e9)
    power = np.full(4, 1e-3)
    kwargs = {
        "evaluator": evaluator,
        "edfa_config": None,
        "power_profile": power,
        "span_length_m": 80e3,
        "num_full_spans": 1,
        "remainder_length_m": 70e3,
        "num_channels": 4,
    }
    baseline = assemble_link_noise(**kwargs)
    explicit = assemble_link_noise(**kwargs, validity=None)
    assert baseline.validity is None
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(baseline, name), getattr(explicit, name)), name


def test_assemble_link_noise_attaches_validity_without_touching_noise() -> None:
    """附带报告只做附加：三个噪声数组逐位相同，且不共享内存。"""
    evaluator = MultiBandISRSGN(SpectrumGrid(4, 50e9, 193.1e12), FiberSpanConfig(length_km=150.0), 50e9)
    power = np.full(4, 1e-3)
    report = _validity(4, invalid=(1,))
    baseline = assemble_link_noise(
        evaluator=evaluator,
        edfa_config=None,
        power_profile=power,
        span_length_m=80e3,
        num_full_spans=1,
        remainder_length_m=70e3,
        num_channels=4,
    )
    with_validity = assemble_link_noise(
        evaluator=evaluator,
        edfa_config=None,
        power_profile=power,
        span_length_m=80e3,
        num_full_spans=1,
        remainder_length_m=70e3,
        num_channels=4,
        validity=report,
    )
    assert with_validity.validity is report
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(baseline, name), getattr(with_validity, name)), name


def test_validator_default_and_explicit_off_are_bit_identical() -> None:
    """默认构造与显式 off 策略给出相同的判定与逐位相同的 SNR 数组。"""
    topology, grid, evaluator, state = _network()
    default = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    explicit = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        validity_policy=LinkValidityPolicy(mode="off"),
    )
    default_ok, default_snr = default.verify_allocation(_request(), [0, 1], 0, state)
    explicit_ok, explicit_snr = explicit.verify_allocation(_request(), [0, 1], 0, state)
    assert default_ok is explicit_ok
    assert default_snr is not None and explicit_snr is not None
    assert np.array_equal(default_snr, explicit_snr)
    assert default.last_link_validity == {}


def test_physics_overlay_off_is_a_no_op() -> None:
    """波形侧的 off 叠加不新增键、不改数组。"""
    report = _validity(3, invalid=(2,))
    metrics = {"snr": np.array([20.0, np.nan, 10.0])}
    overlaid = apply_link_validity(metrics, report, LinkValidityPolicy())
    assert set(overlaid) == {"snr"}
    assert np.array_equal(overlaid["snr"], metrics["snr"], equal_nan=True)


def test_physics_overlay_reject_marks_invalid_and_keeps_input() -> None:
    """Reject 把 NaN/无效信道压成 -inf，输入字典不变。"""
    report = _validity(3, invalid=(2,))
    metrics = {"snr": np.array([20.0, np.nan, 10.0])}
    overlaid = apply_link_validity(metrics, report, LinkValidityPolicy(mode="reject"))
    assert overlaid["snr"][0] == 20.0
    assert np.isneginf(overlaid["snr"][1])
    assert np.isneginf(overlaid["snr"][2])
    assert np.array_equal(overlaid["valid"], np.array([True, True, False]))
    assert set(metrics) == {"snr"}


# ---------------------------------------------------------------------------
# 策略语义
# ---------------------------------------------------------------------------


def test_unknown_policy_and_inconsistent_configuration_are_rejected() -> None:
    """策略必须显式且合法；provider 与策略的组合必须一致。"""
    with pytest.raises(ValueError, match="unknown validity policy"):
        LinkValidityPolicy(mode="maybe")
    topology, grid, evaluator, _state = _network()
    with pytest.raises(ValueError, match="requires a validity_provider"):
        QoTValidator(evaluator, topology, grid, validity_policy=LinkValidityPolicy(mode="reject"))
    with pytest.raises(ValueError, match="mode='off'"):
        QoTValidator(
            evaluator,
            topology,
            grid,
            validity_policy=LinkValidityPolicy(),
            validity_provider=_StubProvider(_validity(4)),
        )


def test_policy_never_accepts_a_non_finite_snr() -> None:
    """NaN 在 flag/reject/degrade 下都是 -inf；off 才保持原值（逐位不变）。"""
    report = _validity(3)
    snr = np.array([20.0, np.nan, -np.inf])
    assert np.array_equal(report.adjusted_snr_db(snr, LinkValidityPolicy()), snr, equal_nan=True)
    for mode in ("flag", "reject", "degrade"):
        adjusted = report.adjusted_snr_db(snr, LinkValidityPolicy(mode=mode))
        assert adjusted[0] == pytest.approx(20.0 if mode != "degrade" else 18.0)
        assert np.isneginf(adjusted[1])
        assert np.isneginf(adjusted[2])


def test_flag_mode_is_audit_only_but_records_validity() -> None:
    """Flag 不改有限 SNR 的判定，但把无效信道标出来。"""
    report = _validity(3, invalid=(1,))
    snr = np.array([25.0, 22.0, 19.0])
    adjusted = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="flag"))
    assert np.array_equal(adjusted, snr)
    assert list(report.valid) == [True, False, True]

    rejected = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="reject"))
    assert np.isneginf(rejected[1])
    assert np.array_equal(rejected[[0, 2]], snr[[0, 2]])

    degraded = report.adjusted_snr_db(snr, LinkValidityPolicy(mode="degrade"))
    assert degraded[0] == pytest.approx(23.0)
    assert np.isneginf(degraded[1])
    assert degraded[2] == pytest.approx(17.0)


def test_zero_power_channel_is_invalid_and_has_zero_margin() -> None:
    """未占用信道在预算里是 inactive：无效、margin 0。"""
    model = make_model("pcfm_h3_raman", CASE)
    path = build_path(CASE)
    power = launch_profile(CASE).copy()
    power[0] = 0.0
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)
    report = LinkValidity.from_budget(budget, link_key=LINK_KEY)

    assert not bool(report.active[0])
    assert not report.channel_valid(0)
    assert report.margin_at(0) == 0.0
    assert report.channel_valid(1)


def test_nan_channel_is_marked_invalid_by_the_budget() -> None:
    """NaN SNR 在预算里必须无效、margin 为 inf。"""
    model = make_model("pcfm_h3_raman", CASE)
    path = build_path(CASE)
    power = launch_profile(CASE)
    result = model.evaluate_path(path, power)
    broken = np.array(result.snr_db, dtype=np.float64, copy=True)
    broken[3] = np.nan
    budget = estimate_h3_error_budget(model, path, power, replace(result, snr_db=broken))
    report = LinkValidity.from_budget(budget)

    assert not report.channel_valid(3)
    assert np.isinf(report.margin_at(3))
    assert np.isneginf(report.adjusted_snr_db(broken, LinkValidityPolicy(mode="reject"))[3])


# ---------------------------------------------------------------------------
# margin 单调
# ---------------------------------------------------------------------------


def test_link_margin_is_monotone_in_k_sigma() -> None:
    """k_sigma 增大时逐信道 margin 不减，且总体严格增大。"""
    model = make_model("pcfm_h3_raman", CASE)
    path = build_path(CASE)
    power = launch_profile(CASE)
    result = model.evaluate_path(path, power)

    margins = []
    for k in (0.0, 2.0, 4.0):
        budget = estimate_h3_error_budget(model, path, power, result, config=H3UncertaintyConfig(k_sigma=k))
        margins.append(LinkValidity.from_budget(budget).margin_db)

    assert np.all(margins[1] >= margins[0])
    assert np.all(margins[2] >= margins[1])
    assert float(np.mean(margins[2])) > float(np.mean(margins[0]))


# ---------------------------------------------------------------------------
# 状态隔离
# ---------------------------------------------------------------------------


def test_link_reports_are_read_only_snapshots() -> None:
    """构造时拷贝并置只读：外部改写输入数组不会影响报告。"""
    valid = np.array([True, True, False])
    margin = np.array([1.0, 2.0, np.inf])
    report = LinkValidity(
        valid=valid,
        active=np.ones(3, dtype=bool),
        flagged=~valid,
        confidence=np.ones(3),
        margin_db=margin,
        model="stub",
    )
    valid[:] = False
    margin[0] = 99.0
    assert list(report.valid) == [True, True, False]
    assert report.margin_at(0) == 1.0
    with pytest.raises(ValueError):
        report.valid[0] = False


def test_path_aggregation_does_not_leak_between_links() -> None:
    """两条链路的报告独立聚合；聚合不改输入，无效链路不污染有效链路。"""
    link_a = _validity(4, margin_db=2.0, link_key=(0, 1))
    link_b = _validity(4, invalid=(2,), margin_db=5.0, link_key=(1, 2))
    verdict = ChannelValidity.combine([link_a, link_b], 4)

    assert list(verdict.valid) == [True, True, False, True]
    assert verdict.margin_db[0] == pytest.approx(7.0)
    assert np.isinf(link_b.margin_db[2])
    # 输入报告仍是各自的原值。
    assert link_a.channel_valid(2)
    assert list(link_b.valid) == [True, True, False, True]

    with pytest.raises(ValueError, match="does not match the verdict"):
        verdict.adjusted_snr_db(np.zeros(2))


def test_scalar_aggregation_only_sums_valid_links() -> None:
    """标量路径聚合：无效链路不贡献 margin，但会让整条路径无效。"""
    terms = fold_roadm_terms(ROADMConfig(insertion_loss_db=0.0, add_drop_loss_db=0.0), is_last_hop=True)
    zero = LinkNoiseCache(spm=np.zeros(4), xpm=np.zeros(4), ase=np.zeros(4), span_count=1)
    good = replace(zero, validity=_validity(4, margin_db=2.0, link_key=(0, 1)))
    bad = replace(zero, validity=_validity(4, invalid=(0,), margin_db=7.0, link_key=(1, 2)))

    def detail(caches) -> ServiceQoT:
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
    assert only_good.margin_db == pytest.approx(2.0)

    both = detail([good, bad])
    assert not both.valid
    assert both.margin_db == pytest.approx(2.0)
    assert both.invalid_links == ((1, 2),)
    assert np.isneginf(both.effective_snr_db(LinkValidityPolicy(mode="reject")))


def test_accumulate_without_validity_matches_the_legacy_scalar() -> None:
    """无 validity 时 detail 与公开标量接口逐位一致。"""
    terms = fold_roadm_terms(ROADMConfig(insertion_loss_db=3.0), is_last_hop=True)
    cache = LinkNoiseCache(
        spm=np.full(2, 1e-6), xpm=np.full(2, 2e-6), ase=np.full(2, 3e-6), span_count=2
    )

    def hops():
        yield terms, cache

    legacy = accumulate_service_snr(
        hops(),
        channel_idx=0,
        launch_power_w=1e-3,
        frequency_hz=193.1e12,
        channel_bandwidth_hz=50e9,
        is_blocked=lambda _terms: False,
    )
    detail = accumulate_service_snr_detail(
        hops(),
        channel_idx=0,
        launch_power_w=1e-3,
        frequency_hz=193.1e12,
        channel_bandwidth_hz=50e9,
        is_blocked=lambda _terms: False,
    )
    assert legacy == detail.snr_db
    assert detail.valid and detail.margin_db == 0.0


# ---------------------------------------------------------------------------
# 缓存与占用谱
# ---------------------------------------------------------------------------


def test_occupancy_fingerprint_tracks_every_channel() -> None:
    """指纹对任一信道变化敏感，对相同谱稳定。"""
    power = np.full(6, 1e-3)
    base = occupancy_fingerprint(power, 80e3)
    assert base == occupancy_fingerprint(power.copy(), 80e3)
    assert base != occupancy_fingerprint(power * 1.0001, 80e3)
    assert base != occupancy_fingerprint(power, 100e3)
    changed = power.copy()
    changed[3] = 0.0
    assert occupancy_fingerprint(changed, 80e3) != base


def test_link_validity_cache_recomputes_when_the_spectrum_changes() -> None:
    """同一占用谱命中缓存；换一个信道就重算；invalidate 清掉整条链路。"""
    model = make_model("pcfm_h3_raman", CASE)
    provider = H3LinkValidityProvider(model)
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE)

    first = provider(LINK_KEY, fiber, None, power, 100e3)
    second = provider(LINK_KEY, fiber, None, power.copy(), 100e3)
    assert provider.cache.misses == 1
    assert provider.cache.hits == 1
    assert np.array_equal(first.margin_db, second.margin_db)
    assert first.fingerprint == second.fingerprint

    changed = power.copy()
    changed[0] = 0.0
    third = provider(LINK_KEY, fiber, None, changed, 100e3)
    assert provider.cache.misses == 2
    assert third.fingerprint != first.fingerprint
    assert not third.channel_valid(0)
    assert third.channel_valid(1)

    assert provider.cache.invalidate(LINK_KEY) == 2
    assert len(provider.cache) == 0


def test_link_validity_cache_isolates_links_with_the_same_spectrum() -> None:
    """链路键是缓存键的一部分：同一份谱在两条链路上互不覆盖。"""
    provider = H3LinkValidityProvider(make_model("pcfm_h3_raman", CASE))
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE)
    provider((0, 1), fiber, None, power, 100e3)
    provider((1, 2), fiber, None, power, 100e3)
    assert provider.cache.misses == 2
    assert len(provider.cache) == 2
    assert provider.cache.invalidate((0, 1)) == 1


def test_link_validity_cache_is_bounded() -> None:
    """超过上限时淘汰最久未用条目，且不会有第三个链路被误删。"""
    cache = LinkValidityCache(max_entries=2)
    power = np.full(2, 1e-3)
    for link in ((0, 1), (1, 2), (2, 3)):
        cache.put(link, power, 80e3, _validity(2, link_key=link))
    assert len(cache) == 2
    assert cache.get((0, 1), power, 80e3) is None
    assert cache.get((2, 3), power, 80e3) is not None


def test_link_optical_path_uses_the_requested_span_length() -> None:
    """链接光路按链路长度与跨长切分，短跨/零跨长会被拒绝。"""
    fiber = FiberSpanConfig(length_km=170.0, max_span_length_km=100.0)
    path = link_optical_path(fiber, None, 80e3)
    assert path.fiber_configs[0].max_span_length_km == 80.0
    assert isinstance(path.edfa_configs[0], EDFAConfig)
    with pytest.raises(ValueError, match="span_length_m must be positive"):
        link_optical_path(fiber, None, 0.0)


# ---------------------------------------------------------------------------
# 验证器接入
# ---------------------------------------------------------------------------


def test_validator_flag_mode_keeps_the_decision_and_records_links() -> None:
    """Flag 只审计：判定与 SNR 与基线相同，但报告挂在每条受影响链路上。"""
    topology, grid, evaluator, state = _network()
    provider = _StubProvider(_validity(4))
    validator = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        validity_policy=LinkValidityPolicy(mode="flag"),
        validity_provider=provider,
    )
    baseline = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)

    expected_ok, expected_snr = baseline.verify_allocation(_request(), [0, 1], 0, state)
    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)

    assert ok is expected_ok
    assert snr is not None and expected_snr is not None
    assert np.array_equal(snr, expected_snr)
    assert set(validator.last_link_validity) == {LINK_KEY}
    assert len(provider.calls) == 1


def test_validator_reject_mode_drops_invalid_channels_only() -> None:
    """Reject 只拒绝无效信道；有效信道的判定与 SNR 逐位不变。"""
    topology, grid, evaluator, state = _network()
    validator = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        validity_policy=LinkValidityPolicy(mode="reject"),
        validity_provider=_StubProvider(_validity(4, invalid=(2,))),
    )
    baseline = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)

    expected_ok, expected_snr = baseline.verify_allocation(_request(), [0, 1], 0, state)
    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)
    assert ok is expected_ok
    assert snr is not None and expected_snr is not None
    assert np.array_equal(snr[[0, 1, 3]], expected_snr[[0, 1, 3]])
    assert np.isneginf(snr[2])

    rejected, _ = validator.verify_allocation(_request(), [0, 1], 2, state)
    assert rejected is False


def test_validator_degrade_mode_subtracts_the_link_margin() -> None:
    """Degrade 让有效信道扣 margin：同一业务从通过变为不通过。"""
    topology, grid, evaluator, state = _network()
    baseline = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    assert baseline.verify_allocation(_request(), [0, 1], 0, state)[0] is True

    validator = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        validity_policy=LinkValidityPolicy(mode="degrade"),
        validity_provider=_StubProvider(_validity(4, margin_db=100.0)),
    )
    ok, _ = validator.verify_allocation(_request(), [0, 1], 0, state)
    assert ok is False


def test_validator_clears_link_state_between_calls() -> None:
    """报告容器按调用清空，不会把上一次的链路状态带进来。"""
    topology, grid, evaluator, state = _network()
    validator = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        validity_policy=LinkValidityPolicy(mode="flag"),
        validity_provider=_StubProvider(_validity(4)),
    )
    validator.verify_allocation(_request(), [0, 1], 0, state)
    assert len(validator.last_link_validity) == 1
    validator.verify_allocation(_request(), [0, 1], 1, state)
    assert len(validator.last_link_validity) == 1


def test_validator_with_real_h3_provider_records_link_report() -> None:
    """真实 H3 provider 能嵌进验证器：形状对齐、报告落到链路上、判定不崩。"""
    topology, grid, evaluator, state = _network()
    fiber = topology.get_default_fiber_config()
    model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=32e9)
    validator = QoTValidator(
        evaluator,
        topology,
        grid,
        fixed_span_length_km=80.0,
        validity_policy=LinkValidityPolicy(mode="degrade"),
        validity_provider=H3LinkValidityProvider(model),
    )
    ok, snr = validator.verify_allocation(_request(), [0, 1], 0, state)

    assert ok is True
    assert snr is not None
    report = validator.last_link_validity[LINK_KEY]
    assert report.num_channels == 4
    assert report.channel_valid(0)
    assert np.isfinite(report.margin_at(0))
    assert report.channel_valid(1) is False
    assert report.margin_at(1) == 0.0
    assert report.span_count == 1


# ---------------------------------------------------------------------------
# two-tier 原型
# ---------------------------------------------------------------------------


def _edge_case_model(dispersion_d: float = 1.0):
    """低色散算例：整条梳都会被 H3 flag，用于触发回退计划。"""
    case = CASE
    fiber = (
        build_path(case)
        .fiber_configs[0]
        .model_copy(update={"dispersion_parameter_d": dispersion_d, "use_attenuation_polynomial": False})
    )
    model = create_model("pcfm_h3_raman", grid_for(case), fiber, symbol_rate_hz=case.symbol_rate_hz)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)])
    return model, path


def test_two_tier_disabled_never_calls_the_fallback() -> None:
    """默认只出计划：工厂一次都不被调用，有效 SNR 仍等于 H3+预算。"""
    model, path = _edge_case_model()
    calls: list[int] = []

    def factory():
        calls.append(1)
        raise AssertionError("fallback must not be built when disabled")

    evaluator = TwoTierEvaluator(model, factory)
    power = launch_profile(CASE)
    outcome = evaluator.evaluate(path, power)

    assert calls == []
    assert outcome.fallback_ran is False
    assert bool(np.all(outcome.fallback_channels))
    assert bool(np.all(np.isnan(outcome.fallback_snr_db)))
    assert np.array_equal(outcome.effective_snr_db, outcome.budget.adjusted_snr_db(outcome.field.snr_db))
    assert "would be re-evaluated" in outcome.notes[0]


def test_two_tier_enabled_overrides_only_flagged_channels() -> None:
    """启用后只有 flagged 信道取数值值，其余仍是 H3+预算。"""
    model, path = _edge_case_model()
    power = launch_profile(CASE)
    fallback = _StubFallbackModel(np.full(power.shape[0], 42.0))
    evaluator = TwoTierEvaluator(
        model,
        lambda: fallback,
        config=TwoTierConfig(enabled=True, max_fallback_channels=power.shape[0]),
    )
    outcome = evaluator.evaluate(path, power)

    assert fallback.calls == 1
    assert outcome.fallback_ran is True
    assert np.all(outcome.fallback_snr_db[outcome.fallback_channels] == 42.0)
    assert np.all(outcome.effective_snr_db[outcome.fallback_channels] == 42.0)
    assert "numerical fallback ran" in outcome.notes[0]


def test_two_tier_refuses_more_flagged_channels_than_the_budget() -> None:
    """超过 max_fallback_channels 直接报错，不静默截断。"""
    model, path = _edge_case_model()
    evaluator = TwoTierEvaluator(
        model,
        lambda: _StubFallbackModel(np.zeros(CASE.num_channels)),
        config=TwoTierConfig(enabled=True, max_fallback_channels=1),
    )
    with pytest.raises(ValueError, match="exceed max_fallback_channels"):
        evaluator.evaluate(path, launch_profile(CASE))


def test_two_tier_requires_a_factory_when_enabled() -> None:
    """启用回退却没有工厂是配置错误。"""
    model, path = _edge_case_model()
    evaluator = TwoTierEvaluator(
        model,
        None,
        config=TwoTierConfig(enabled=True, max_fallback_channels=CASE.num_channels),
    )
    with pytest.raises(ValueError, match="no fallback_factory"):
        evaluator.evaluate(path, launch_profile(CASE))


def test_numerical_factory_is_lazy_and_constructible() -> None:
    """工厂只在被调用时导入并构造数值模型。"""
    fiber = build_path(CASE).fiber_configs[0]
    factory = numerical_dra_factory(grid_for(CASE), fiber, symbol_rate_hz=CASE.symbol_rate_hz)
    model = factory()
    assert model.info.name == "numerical_dra"
    assert model.n_nodes == CASE.num_channels
