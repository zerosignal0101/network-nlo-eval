"""``pcfm_h3_raman`` error-budget and validity layer tests (task D).

Covered contract:

* ``evaluate_path_with_uncertainty`` returns the bit-for-bit default result;
* the no-pump budget is the H3 PCFM result and carries no pump flags;
* pumped links record the pump direction and still produce a finite margin;
* a low-dispersion (edge) comb is flagged, loses confidence and gains margin;
* the margin is monotone in ``k_sigma``, the statistical term and the residual
  scale;
* the strong-tilt row is selected when the measured gain tilt crosses the
  configured threshold;
* invalid channels get ``margin = inf`` and ``-inf`` adjusted SNR.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_case, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.uncertainty import (
    H3_STRONG_TILT_RESIDUAL,
    H3UncertaintyConfig,
    classify_bandwidth,
    classify_pump_mode,
    estimate_h3_error_budget,
    select_residual_entry,
    span_validity,
)
from network_nlo_eval.network.elements import EDFAConfig, RamanPumpConfig
from network_nlo_eval.rwa.validity import LinkValidity

CASE_1 = build_case("scaling_40ch")
CASE_10 = build_case("c_57ch")


def _case_with_spacing(spacing_hz: float, power_dbm: float = 6.0) -> BenchCase:
    """C 波段、64 GBaud、给定栅格间隔的 1 跨算例。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"dense_{spacing_hz / 1e9:g}",
        bands="C",
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=power_dbm,
    )


_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _pump(power_w: float = 0.2, direction: str = "backward") -> RamanPumpConfig:
    """Construct one 1450 nm pump."""
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _assert_bit_identical(left, right) -> None:
    """All point-result arrays must be identical, not merely close."""
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


def _model_with_fiber(case, dispersion_d: float):
    """Build an H3 model on the case grid with an overridden dispersion."""
    fiber = (
        build_path(case)
        .fiber_configs[0]
        .model_copy(update={"dispersion_parameter_d": dispersion_d, "use_attenuation_polynomial": False})
    )
    model = create_model(
        "pcfm_h3_raman",
        grid_for(case),
        fiber,
        symbol_rate_hz=case.symbol_rate_hz,
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)])
    return model, path, fiber


# ---------------------------------------------------------------------------
# Default path is untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("case", "options"),
    [(CASE_1, {}), (CASE_10, {"raman_pumps": [_pump()]})],
    ids=["nopump", "pumped"],
)
def test_evaluate_path_with_uncertainty_is_bit_identical_to_evaluate_path(case, options) -> None:
    """The opt-in interface must not move the point result by one bit."""
    path = build_path(case)
    power = launch_profile(case)
    baseline = make_model("pcfm_h3_raman", case, **options).evaluate_path(path, power)
    result, budget = make_model("pcfm_h3_raman", case, **options).evaluate_path_with_uncertainty(path, power)
    _assert_bit_identical(baseline, result)
    assert budget.margin_db.shape == (case.num_channels,)
    assert budget.validity.valid.shape == (case.num_channels,)


def test_no_pump_budget_keeps_the_pcfm_result_and_has_no_pump_flags() -> None:
    """No-pump H3 is PCFM; its budget must not invent pump warnings."""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    pcfm = make_model("pcfm", CASE_1, poly_degree=3, egn=True, spp_steps="auto").evaluate_path(path, power)
    result, budget = make_model("pcfm_h3_raman", CASE_1).evaluate_path_with_uncertainty(path, power)
    _assert_bit_identical(pcfm, result)
    assert budget.validity.pump_mode == "none"
    assert not budget.validity.pump_not_converged.any()
    assert not budget.validity.flagged.any()
    assert bool(np.all(budget.validity.valid))
    assert bool(np.all(budget.margin_db > 0.0))


def test_error_budget_supports_plain_pcfm_kernels() -> None:
    """任务 Q 回归：纯 PCFM 核没有泵状态与跨增益，预算必须仍可用。

    ``H3LinkValidityProvider`` 的文档声明支持 ``pcfm``/``pcfm_h3_raman``；
    修复前 ``estimate_h3_error_budget`` 在 ``kernel.active`` 上抛
    ``AttributeError``，链路级有效性层因此完全不可用于纯 PCFM。
    """
    from network_nlo_eval.rwa.validity import H3LinkValidityProvider

    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    model = make_model("pcfm", CASE_1, egn=True)
    result = model.evaluate_path(path, power)

    budget = estimate_h3_error_budget(model, path, power, result)
    assert budget.validity.pump_mode == "none"
    assert bool(np.all(budget.validity.valid))
    assert bool(np.all(budget.margin_db > 0.0))

    # 用户可见入口同样不能再崩，且软 margin 默认为 0。
    report = H3LinkValidityProvider(model)((0, 1), path.fiber_configs[0], None, power, 100e3)
    assert np.isfinite(report.margin_at(0))
    assert report.soft_margin_at(0) == 0.0


def test_error_budget_reads_pump_directions_from_plain_pcfm_config() -> None:
    """纯 PCFM 的集总拉曼核没有泵状态，预算须回落到模型配置并标记不可信。"""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    with pytest.warns(UserWarning, match="lumped"):
        model = make_model(
            "pcfm",
            CASE_1,
            egn=True,
            raman_model="lumped",
            raman_pumps=[_pump(0.2, "backward")],
        )
    result = model.evaluate_path(path, power)

    budget = estimate_h3_error_budget(model, path, power, result)
    assert budget.validity.pump_mode == "backward"
    # 集总拉曼有泵：点值保留（兼容），但必须逐信道 flag 且扣 confidence。
    assert bool(np.all(budget.validity.lumped_raman))
    assert bool(np.all(budget.validity.flagged))
    assert bool(np.all(budget.validity.confidence <= 0.6))
    assert bool(np.all(np.isfinite(budget.margin_db)))
    assert any("lumped Raman approximation" in note for note in budget.notes)


def test_plain_pcfm_without_pumps_does_not_raise_the_lumped_warning() -> None:
    """无泵纯 PCFM 不是集总拉曼路径，不得触发 warning 或 flag。"""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    model = make_model("pcfm", CASE_1, egn=True)
    result = model.evaluate_path(path, power)

    budget = estimate_h3_error_budget(model, path, power, result)
    assert not model.lumped_raman_active
    assert not budget.validity.lumped_raman.any()
    assert not budget.validity.flagged.any()


@pytest.mark.parametrize(
    ("pumps", "expected"),
    [
        ([_pump(0.2, "backward")], "backward"),
        ([_pump(0.2, "forward")], "forward"),
        ([_pump(0.2, "forward"), _pump(0.15, "backward")], "mixed"),
    ],
)
def test_pump_mode_is_recorded(pumps, expected: str) -> None:
    """The budget must report the pump set that drove the residual row."""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    _result, budget = make_model("pcfm_h3_raman", CASE_1, raman_pumps=pumps).evaluate_path_with_uncertainty(path, power)
    assert budget.validity.pump_mode == expected
    assert bool(np.all(np.isfinite(budget.margin_db)))
    assert bool(np.all(budget.validity.valid))


# ---------------------------------------------------------------------------
# Normal vs edge channel
# ---------------------------------------------------------------------------


def test_low_dispersion_edge_is_flagged_and_gets_a_larger_margin() -> None:
    """A comb near zero dispersion must lose confidence and gain margin."""
    power = launch_profile(CASE_1)
    normal_model, normal_path, _ = _model_with_fiber(CASE_1, dispersion_d=17.0)
    edge_model, edge_path, _ = _model_with_fiber(CASE_1, dispersion_d=1.0)

    normal_result, normal_budget = normal_model.evaluate_path_with_uncertainty(normal_path, power)
    edge_result, edge_budget = edge_model.evaluate_path_with_uncertainty(edge_path, power)

    assert bool(np.all(np.isfinite(normal_result.snr_db)))
    assert bool(np.all(np.isfinite(edge_result.snr_db)))
    assert not normal_budget.validity.flagged.any()
    assert bool(np.all(edge_budget.validity.flagged))
    assert bool(np.all(edge_budget.validity.xci_edge_gated))
    assert float(np.min(edge_budget.validity.confidence)) < float(np.min(normal_budget.validity.confidence))
    assert float(np.mean(edge_budget.margin_db)) > float(np.mean(normal_budget.margin_db))
    # 有效性下降不等于结果不可用：硬判定仍可做，只是 margin 更大。
    assert bool(np.all(edge_budget.validity.valid))


def test_span_validity_exposes_kernel_level_metrics() -> None:
    """The shared span hook returns model-order shapes and the gate mask."""
    edge_model, edge_path, _ = _model_with_fiber(CASE_1, dispersion_d=1.0)
    kernel = edge_model.kernel_for(edge_path.fiber_configs[0])
    flagged, confidence, min_abs_x, min_s = span_validity(kernel, edge_path.fiber_configs[0].length_km * 1e3)
    assert flagged.shape == confidence.shape == min_abs_x.shape == min_s.shape == (CASE_1.num_channels,)
    assert bool(np.all(flagged))
    assert float(np.max(confidence)) < 1.0
    assert bool(np.all(min_abs_x >= 0.0))


def test_span_validity_min_s_matches_the_kernel_report() -> None:
    """The budget layer must print the same ``s`` as the kernel validity report.

    Both are the exact island integral's ``s`` (not the pre-fix half value);
    ``XCI_DEEP_BREAKDOWN_S`` is expressed in those units as well.
    """
    from network_nlo_eval.models.uncertainty import XCI_DEEP_BREAKDOWN_S

    case = build_case("escl_479ch_1span")
    model = make_model("pcfm", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    length_m = case.span_length_km * 1e3
    power = launch_profile(case, None)

    report = kernel.validity_report(length_m, power)
    flagged, confidence, min_abs_x, min_s = span_validity(kernel, length_m)

    assert np.array_equal(min_s, np.asarray(report.xci_s_min))
    assert np.array_equal(min_abs_x, np.abs(np.asarray(report.sci_x)))
    # 深破坏判据与 gate 使用同一口径：这里只确认常量不是旧的半值单位。
    assert XCI_DEEP_BREAKDOWN_S == 2.0



# ---------------------------------------------------------------------------
# Margin monotonicity
# ---------------------------------------------------------------------------


def test_margin_is_monotone_in_every_spread_knob() -> None:
    """More sigma or more penalty must never reduce the recommended margin."""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    model = make_model("pcfm_h3_raman", CASE_1)
    result = model.evaluate_path(path, power)

    def margin(**kwargs) -> float:
        budget = estimate_h3_error_budget(
            model,
            path,
            power,
            result,
            config=H3UncertaintyConfig(**kwargs),
        )
        return float(np.mean(budget.margin_db))

    k_values = [0.0, 1.0, 2.0, 4.0]
    margins = [margin(k_sigma=k) for k in k_values]
    assert np.all(np.diff(margins) > 0.0), margins

    assert margin(include_statistical=False) < margin(include_statistical=True)
    assert margin(statistical_scale=2.0) > margin(statistical_scale=1.0)
    assert margin(residual_scale=2.0) > margin(residual_scale=1.0)


def test_validity_penalty_only_moves_flagged_channels() -> None:
    """The penalty is proportional to ``1 - confidence``; clean channels ignore it."""
    power = launch_profile(CASE_1)
    edge_model, edge_path, _ = _model_with_fiber(CASE_1, dispersion_d=1.0)
    edge_result = edge_model.evaluate_path(edge_path, power)

    def margin(penalty_db: float) -> float:
        budget = estimate_h3_error_budget(
            edge_model,
            edge_path,
            power,
            edge_result,
            config=H3UncertaintyConfig(validity_penalty_db=penalty_db),
        )
        return float(np.mean(budget.margin_db))

    assert margin(6.0) > margin(1.0)


def test_strong_tilt_threshold_inflates_margin() -> None:
    """Crossing the tilt threshold switches to the optimistic strong-tilt row."""
    six_pumps = [
        RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
        RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
        RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
        RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
        RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
        RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
    ]
    case = replace(
        build_case("c_57ch"),
        name="cl_40ch_1span",
        bands="C+L",
        num_channels=40,
        center_frequency_hz=190.0e12,
        span_count=1,
    )
    path = build_path(case)
    power = launch_profile(case)
    model = make_model("pcfm_h3_raman", case, raman_pumps=six_pumps)
    result = model.evaluate_path(path, power)

    normal = estimate_h3_error_budget(model, path, power, result)
    strong = estimate_h3_error_budget(
        model,
        path,
        power,
        result,
        config=H3UncertaintyConfig(strong_tilt_db=1.0),
    )
    assert normal.validity.tilt_db > 1.0
    assert not normal.validity.strong_tilt.any()
    assert bool(np.all(strong.validity.strong_tilt))
    assert float(np.mean(strong.margin_db)) > float(np.mean(normal.margin_db))
    assert strong.bias_db[0] == pytest.approx(H3_STRONG_TILT_RESIDUAL.bias_db)
    assert any("strong tilt" in note for note in strong.notes)


# ---------------------------------------------------------------------------
# Calibration helpers and invalid channels
# ---------------------------------------------------------------------------


def test_residual_row_selection_and_bandwidth_classification() -> None:
    """The selector is explicit and the wideband boundary is documented."""
    assert classify_bandwidth(4.3e12) == "narrow"
    assert classify_bandwidth(11.4e12) == "wide"
    assert classify_pump_mode([]) == "none"
    assert classify_pump_mode(["backward"]) == "backward"
    assert classify_pump_mode(["forward", "backward"]) == "mixed"
    with pytest.raises(ValueError, match="unknown pump directions"):
        classify_pump_mode(["sideways"])

    entry, notes = select_residual_entry("backward", "wide", "strong")
    assert entry is H3_STRONG_TILT_RESIDUAL
    assert entry.bias_db > 0.0
    assert not entry.calibrated
    assert notes

    with pytest.raises(KeyError, match="no H3 residual row"):
        select_residual_entry("mixed", "ultra", "normal")


def test_non_converged_pump_invalidates_the_channels() -> None:
    """A fixed-point miss is a hard failure: infinite margin, -inf adjusted SNR."""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    model = make_model("pcfm_h3_raman", CASE_1, raman_pumps=[_pump()])
    result = model.evaluate_path(path, power)
    for kernel in model._kernel_cache.values():  # deliberately forces the failure branch
        kernel.last_raman_converged = False

    budget = estimate_h3_error_budget(model, path, power, result)
    assert bool(np.all(budget.validity.pump_not_converged))
    assert bool(np.all(budget.validity.raman_not_converged))
    assert not budget.validity.valid.any()
    assert bool(np.all(np.isinf(budget.margin_db)))
    assert bool(np.all(np.isneginf(budget.adjusted_snr_db(result.snr_db))))


# ---------------------------------------------------------------------------
# 收敛保护与集总近似 flag
# ---------------------------------------------------------------------------


def _wideband_case() -> BenchCase:
    """E+S+C+L 100 GHz 单跨，与第五轮精度矩阵同源。"""
    low, high = band_edges_hz("E+S+C+L")
    return BenchCase(
        name="escl_100GHz_1span",
        bands="E+S+C+L",
        num_channels=int(round((high - low) / 100e9)) + 1,
        channel_spacing_hz=100e9,
        center_frequency_hz=(low + high) / 2.0,
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


def _six_pumps(direction: str) -> list[RamanPumpConfig]:
    """六泵组的方向翻转。"""
    spec = (
        (1340.0, 0.63),
        (1360.0, 0.63),
        (1427.0, 0.165),
        (1445.0, 0.131),
        (1476.0, 0.125),
        (1505.0, 0.075),
    )
    return [
        RamanPumpConfig(
            wavelength_nm=wavelength,
            power_w=power,
            direction=direction,  # type: ignore[arg-type]
            attenuation_db_per_km=0.25,
        )
        for wavelength, power in spec
    ]


def test_wideband_multipump_budget_exposes_the_regime_flag() -> None:
    """E+S+C+L 六泵默认收敛，但预算必须标出宽波段多泵区域。"""
    case = _wideband_case()
    path = build_path(case)
    power = np.full(case.num_channels, 10.0 ** (6.0 / 10.0) * 1e-3)
    result, budget = make_model(
        "pcfm_h3_raman", case, raman_pumps=_six_pumps("backward")
    ).evaluate_path_with_uncertainty(path, power)

    assert bool(np.all(np.isfinite(result.snr_db)))
    assert bool(np.all(budget.validity.raman_wideband_multipump))
    assert bool(np.all(budget.validity.flagged))
    assert not bool(np.any(budget.validity.raman_not_converged))
    assert any("96-step branch" in note for note in budget.notes)


def test_under_resolved_e_s_multipump_is_not_silently_valid() -> None:
    """显式 24 步的 E+S+C+L 后向解 NaN：预算必须判无效、margin 无穷。"""
    import warnings

    case = _wideband_case()
    path = build_path(case)
    power = np.full(case.num_channels, 10.0 ** (6.0 / 10.0) * 1e-3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = make_model(
            "pcfm_h3_raman",
            case,
            raman_pumps=_six_pumps("backward"),
            raman_steps=24,
        )
        result = model.evaluate_path(path, power)

    budget = estimate_h3_error_budget(model, path, power, result)
    assert bool(np.all(budget.validity.raman_not_converged))
    assert not budget.validity.valid.any()
    assert bool(np.all(np.isinf(budget.margin_db)))
    assert bool(np.all(budget.validity.raman_wideband_multipump))
    assert any("marked invalid" in note for note in budget.notes)


def test_zero_power_channels_are_inactive_and_do_not_raise_margin() -> None:
    """An unlaunched channel has no budget; only launched channels do."""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1).copy()
    inactive = CASE_1.num_channels // 2
    power[inactive] = 0.0
    model = make_model("pcfm_h3_raman", CASE_1)
    result, budget = model.evaluate_path_with_uncertainty(path, power)

    assert not bool(budget.validity.active[inactive])
    assert not bool(budget.validity.valid[inactive])
    assert budget.margin_db[inactive] == 0.0
    assert bool(np.isneginf(budget.adjusted_snr_db(result.snr_db)[inactive]))


def test_adjusted_snr_and_json_export() -> None:
    """The budget is consumable by RWA code and serialisable."""
    path = build_path(CASE_1)
    power = launch_profile(CASE_1)
    _result, budget = make_model("pcfm_h3_raman", CASE_1).evaluate_path_with_uncertainty(path, power)

    adjusted = budget.adjusted_snr_db(np.full(CASE_1.num_channels, 30.0))
    assert np.allclose(adjusted, 30.0 - budget.margin_db)
    with pytest.raises(ValueError, match="does not match the budget"):
        budget.adjusted_snr_db(np.zeros(2))

    payload = budget.to_dict()
    encoded = json.dumps(payload)
    assert '"model": "pcfm_h3_raman"' in encoded
    assert len(payload["validity"]["valid"]) == CASE_1.num_channels
    assert len(payload["terms"]) == 4
    assert payload["validity"]["dense_grid_overlap"] is False
    assert payload["validity"]["min_channel_spacing_hz"] == pytest.approx(75e9)
    assert payload["validity"]["spacing_to_bandwidth"] == pytest.approx(75e9 / 64e9)


@pytest.mark.parametrize("spacing_hz", [37.5e9, 50e9])
def test_dense_grid_flags_without_moving_the_default_margin(spacing_hz: float) -> None:
    """密集栅格只 flag：默认 margin 仍是窄带 1.414 dB，valid 不变。"""
    case = _case_with_spacing(spacing_hz)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case)
    result, budget = model.evaluate_path_with_uncertainty(path, power)

    assert budget.validity.dense_grid_overlap
    assert budget.validity.spacing_to_bandwidth == pytest.approx(spacing_hz / 64e9)
    assert budget.validity.min_channel_spacing_hz == pytest.approx(spacing_hz)
    assert bool(np.all(budget.validity.flagged))
    assert bool(np.all(budget.validity.valid))
    assert any(note.startswith("dense grid") for note in budget.notes)

    link = LinkValidity.from_budget(budget)
    assert link.dense_grid_overlap
    assert link.to_dict()["dense_grid_overlap"] is True
    assert bool(np.all(link.valid))


def test_dense_grid_penalty_is_an_explicit_opt_in() -> None:
    """1.1 dB 的密集栅格余量只在显式配置时加入，且逐信道相同。"""
    case = _case_with_spacing(50e9)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case)
    result = model.evaluate_path(path, power)

    baseline = estimate_h3_error_budget(model, path, power, result)
    tuned = estimate_h3_error_budget(
        model,
        path,
        power,
        result,
        config=H3UncertaintyConfig(dense_grid_penalty_db=1.1),
    )

    assert len(baseline.terms) == 4
    assert len(tuned.terms) == 5
    assert tuned.terms[-1].name == "dense_grid_penalty"
    assert tuned.margin_db == pytest.approx(baseline.margin_db + 1.1)
    assert any("dense-grid penalty 1.10 dB applied" in note for note in tuned.notes)


@pytest.mark.parametrize("spacing_hz", [75e9, 100e9])
def test_dense_grid_does_not_fire_or_penalise_validated_spacings(spacing_hz: float) -> None:
    """75 / 100 GHz 不触发 flag，opt-in 余量也不生效。"""
    case = _case_with_spacing(spacing_hz)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case)
    result = model.evaluate_path(path, power)

    baseline = estimate_h3_error_budget(model, path, power, result)
    tuned = estimate_h3_error_budget(
        model,
        path,
        power,
        result,
        config=H3UncertaintyConfig(dense_grid_penalty_db=1.1),
    )

    assert not baseline.validity.dense_grid_overlap
    assert baseline.validity.spacing_to_bandwidth > 1.0
    assert not bool(np.any(baseline.validity.flagged))
    assert np.array_equal(tuned.margin_db, baseline.margin_db)
    assert len(tuned.terms) == 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k_sigma": -1.0},
        {"residual_scale": -0.1},
        {"statistical_scale": -0.1},
        {"validity_penalty_db": -1.0},
        {"strong_tilt_db": -1.0},
        {"dense_grid_penalty_db": -1.0},
    ],
)
def test_negative_config_values_are_rejected(kwargs) -> None:
    """A negative scale would silently turn the margin into an optimism bonus."""
    with pytest.raises(ValueError):
        H3UncertaintyConfig(**kwargs)
