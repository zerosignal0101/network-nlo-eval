"""O 波段低色散有效性门与 FWM 误差审计的验收测试（任务 AP）。

覆盖契约：

* ``fwm_validity_mode="flag"``（默认）不改 NLI/ASE/SNR，也不改 ``valid``；
* 零色散落在占用带内（O 波段）必须 ``fwm_required``，C/L/S 波段不得命中；
* ``min|D|`` / ``zero_dispersion_in_band`` / ``dispersion_reference_in_band``
  与合成光纤的手算一致；
* ``fwm_validity_mode="reject"`` 只清 ``valid``，点值逐位不变；
* H3 的 ``validity_report`` 合并父类字段（不丢新字段）；
* ``uncertainty.estimate_h3_error_budget`` 的聚合、note 与 reject 语义；
* 非法 mode 在 kernel 与 model 两层都被拒绝。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, build_case, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.pcfm import (
    FWM_REQUIRED_MIN_DISPERSION_PS_NM_KM,
    PCFMSpanKernel,
    dispersion_validity,
)
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
from network_nlo_eval.network.elements import EDFAConfig

C_LIGHT = 299_792_458.0
O_SYMBOL_RATE_HZ = 96e9
_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _o_fiber(anchor: str = "in_band"):
    """O 波段合成光纤：in_band 锚点 1310/2.6，out_of_band 锚点 1550/17。"""
    if anchor == "in_band":
        reference_nm, dispersion = 1310.0, 2.6
    else:
        reference_nm, dispersion = 1550.0, 17.0
    return SMF_100KM.model_copy(
        update={
            "reference_wavelength_nm": reference_nm,
            "dispersion_parameter_d": dispersion,
            "dispersion_slope_s": 0.06,
            "use_attenuation_polynomial": False,
            "attenuation_db_km_ref": 0.35,
            "effective_area_um2_ref": 80.0,
            "aeff_slope_um2_nm": 0.05,
        }
    )


def _o_grid(num_channels: int = 41) -> SpectrumGrid:
    """16.1 THz 占用带、中心 229.15 THz 的 O 波段栅格。"""
    return SpectrumGrid(num_channels, 16.1e12 / (num_channels - 1), 229.15e12)


def _o_path(fiber, span_count: int = 1) -> MultiSpanOpticalPath:
    return MultiSpanOpticalPath(
        [fiber] * span_count,
        [EDFAConfig(noise_figure_db=5.5)] * span_count,
    )


def _assert_arrays_equal(left, right) -> None:
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


# ---------------------------------------------------------------------------
# min|D| 与合成光纤手算一致
# ---------------------------------------------------------------------------


def _kernel_for_frequencies(wavelengths_nm: np.ndarray, fiber) -> PCFMSpanKernel:
    frequencies = C_LIGHT / (np.asarray(wavelengths_nm, dtype=np.float64) * 1e-9)
    bandwidths = np.full(frequencies.shape, O_SYMBOL_RATE_HZ, dtype=np.float64)
    return PCFMSpanKernel(frequencies, bandwidths, fiber)


def test_min_abs_dispersion_matches_hand_calculation() -> None:
    """D(λ) = D_ref + S(λ-λ_ref)，min|D| 取占用信道上手算的最小值。"""
    fiber = _o_fiber("in_band")
    wavelengths = np.array([1280.0, 1300.0, 1320.0, 1340.0])
    kernel = _kernel_for_frequencies(wavelengths, fiber)
    report = kernel.validity_report(80e3, np.full(4, 1e-3))

    expected = 2.6 + 0.06 * (wavelengths - 1310.0)
    audit = dispersion_validity(
        C_LIGHT / (wavelengths * 1e-9),
        np.full(4, O_SYMBOL_RATE_HZ),
        np.ones(4, dtype=bool),
        dispersion_parameter_d=2.6,
        dispersion_slope_s=0.06,
        reference_wavelength_nm=1310.0,
    )
    assert np.allclose(audit.per_channel_dispersion_ps_nm_km, expected, rtol=0.0, atol=1e-12)
    assert report.min_abs_dispersion_ps_nm_km == pytest.approx(float(np.min(np.abs(expected))), rel=0.0)
    # 1.4 ps/nm/km 低于标定阈值：低色散命中，但零色散点 1266.7 nm 不在带内。
    assert report.min_abs_dispersion_ps_nm_km < FWM_REQUIRED_MIN_DISPERSION_PS_NM_KM
    assert report.zero_dispersion_in_band is False
    assert report.dispersion_reference_in_band is True
    assert report.fwm_required is True
    assert report.fwm_reason == "low_dispersion"


def test_zero_dispersion_crossing_is_detected_from_band_edges() -> None:
    """跨零色散点的合成带必须报 zero，且 reason 里出现两个原因。"""
    fiber = _o_fiber("in_band")
    wavelengths = np.array([1250.0, 1280.0])  # 1266.67 nm 落在中间
    kernel = _kernel_for_frequencies(wavelengths, fiber)
    report = kernel.validity_report(80e3, np.full(2, 1e-3))

    assert report.zero_dispersion_in_band is True
    assert "zero_dispersion_in_band" in report.fwm_reason
    assert "low_dispersion" in report.fwm_reason


def test_out_of_band_reference_is_reported_separately() -> None:
    """1550 nm 锚点在 O 波段外：独立于 FWM 的第二个误差源。"""
    fiber = _o_fiber("out_of_band")
    kernel = _kernel_for_frequencies(np.array([1300.0, 1320.0]), fiber)
    report = kernel.validity_report(80e3, np.full(2, 1e-3))

    assert report.dispersion_reference_in_band is False
    assert "dispersion_reference_out_of_band" in report.fwm_reason
    # 同一个 D(λ) 直线：min|D| 与 in-band 锚点算出来一致。
    assert report.min_abs_dispersion_ps_nm_km == pytest.approx(
        2.6 + 0.06 * (1300.0 - 1310.0), rel=0.0, abs=1e-9
    )


# ---------------------------------------------------------------------------
# 波段判定：C/L/S 不命中，O 波段命中
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", ["c_57ch", "cl_152ch", "scl_277ch"])
def test_validated_bands_are_not_fwm_flagged(case_name: str) -> None:
    """已验证的 C / C+L / S+C+L 波段不得被低色散门误 flag。"""
    case = build_case(case_name)
    model = make_model("pcfm_h3_raman", case)
    model.evaluate_path(build_path(case), launch_profile(case))
    report = model.last_span_validity[0]

    assert report.fwm_required is False
    assert report.fwm_reason == ""
    assert report.min_abs_dispersion_ps_nm_km >= FWM_REQUIRED_MIN_DISPERSION_PS_NM_KM
    assert report.fwm_validity_mode == "flag"


def test_o_band_zero_dispersion_is_flagged_but_valid_by_default() -> None:
    """O 波段 161 信道：必须命中，默认 flag 模式不改 valid。"""
    fiber = _o_fiber("in_band")
    grid = _o_grid(161)
    model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ, poly_degree=3, egn=True)
    path = _o_path(fiber)
    result = model.evaluate_path(path, np.full(161, 1e-3))
    report = model.last_span_validity[0]

    assert report.fwm_required is True
    assert report.zero_dispersion_in_band is True
    assert report.fwm_reason == "zero_dispersion_in_band+low_dispersion"
    assert bool(np.all(report.flagged))
    assert bool(np.all(report.valid))
    assert np.all(np.isfinite(result.snr_db))


# ---------------------------------------------------------------------------
# 默认逐位不变 / reject 只改 valid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", ["c_57ch", "escl_479ch_1span"])
def test_flag_mode_is_bit_identical_and_reject_only_changes_valid(case_name: str) -> None:
    """Flag 与 reject 的 NLI/ASE/SNR 逐位相同；reject 只清 O 波段 valid。"""
    case = build_case(case_name)
    path = build_path(case)
    power = launch_profile(case)
    flag_model = make_model("pcfm_h3_raman", case, fwm_validity_mode="flag")
    reject_model = make_model("pcfm_h3_raman", case, fwm_validity_mode="reject")

    flag_result = flag_model.evaluate_path(path, power)
    reject_result = reject_model.evaluate_path(path, power)
    _assert_arrays_equal(flag_result, reject_result)

    flag_report = flag_model.last_span_validity[0]
    reject_report = reject_model.last_span_validity[0]
    assert np.array_equal(np.asarray(flag_report.valid), np.asarray(reject_report.valid))
    assert np.array_equal(np.asarray(flag_report.flagged), np.asarray(reject_report.flagged))
    # C / E+S+C+L 不在低色散区，reject 不清任何信道。
    assert bool(np.all(reject_report.valid))


def test_reject_clears_valid_only_on_o_band() -> None:
    """O 波段 reject：valid 全 False，点值仍与 flag 逐位相同。"""
    fiber = _o_fiber("in_band")
    grid = _o_grid(41)
    power = np.full(41, 1e-3)
    path = _o_path(fiber)
    flag_model = create_model(
        "pcfm_h3_raman", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ, poly_degree=3, egn=True
    )
    reject_model = create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        symbol_rate_hz=O_SYMBOL_RATE_HZ,
        poly_degree=3,
        egn=True,
        fwm_validity_mode="reject",
    )
    flag_result = flag_model.evaluate_path(path, power)
    reject_result = reject_model.evaluate_path(path, power)
    _assert_arrays_equal(flag_result, reject_result)

    assert bool(np.all(flag_model.last_span_validity[0].valid))
    assert not bool(np.any(reject_model.last_span_validity[0].valid))
    assert bool(np.all(reject_model.last_span_validity[0].flagged))
    assert reject_model.last_span_validity[0].fwm_validity_mode == "reject"


# ---------------------------------------------------------------------------
# 非法 mode
# ---------------------------------------------------------------------------


def test_invalid_fwm_validity_mode_is_rejected() -> None:
    """Kernel 与 model 两层都必须拒绝未知 mode。"""
    fiber = _o_fiber("in_band")
    frequencies = C_LIGHT / (np.array([1300.0, 1320.0]) * 1e-9)
    with pytest.raises(ValueError, match="fwm_validity_mode"):
        PCFMSpanKernel(frequencies, np.full(2, O_SYMBOL_RATE_HZ), fiber, fwm_validity_mode="strict")
    with pytest.raises(ValueError, match="fwm_validity_mode"):
        create_model(
            "pcfm_h3_raman",
            _o_grid(41),
            fiber,
            symbol_rate_hz=O_SYMBOL_RATE_HZ,
            fwm_validity_mode="on",
        )
    with pytest.raises(ValueError, match="fwm_validity_mode"):
        create_model(
            "pcfm",
            grid_for(build_case("c_57ch")),
            _o_fiber("in_band"),
            fwm_validity_mode="warn",
        )


# ---------------------------------------------------------------------------
# H3 report 合并 / error budget 聚合
# ---------------------------------------------------------------------------


def test_h3_report_keeps_parent_dispersion_fields() -> None:
    """H3 的 replace 覆盖不能丢掉父类新增的色散字段。"""
    fiber = _o_fiber("out_of_band")
    grid = _o_grid(41)
    model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ, poly_degree=3, egn=True)
    model.evaluate_path(_o_path(fiber), np.full(41, 1e-3))
    report = model.last_span_validity[0]

    assert report.fwm_required is True
    assert report.zero_dispersion_in_band is True
    assert report.dispersion_reference_in_band is False
    assert report.fwm_reason.endswith("dispersion_reference_out_of_band")
    assert np.isfinite(report.min_abs_dispersion_ps_nm_km)


def test_error_budget_aggregates_fwm_flag_and_note() -> None:
    """任一跨 fwm_required → 整体为真；flagged OR；note 明确声明未含 FWM。"""
    fiber = _o_fiber("in_band")
    grid = _o_grid(41)
    path = _o_path(fiber, span_count=3)
    power = np.full(41, 1e-3)
    model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ, poly_degree=3, egn=True)
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)

    assert budget.validity.fwm_required is True
    assert budget.validity.zero_dispersion_in_band is True
    assert budget.validity.dispersion_reference_in_band is True
    assert budget.validity.fwm_reason == "zero_dispersion_in_band+low_dispersion"
    assert budget.validity.fwm_validity_mode == "flag"
    assert bool(np.all(budget.validity.flagged))
    assert bool(np.all(budget.validity.valid))
    note = next(note for note in budget.notes if "未含 FWM" in note)
    assert "reference-only" in note


def test_error_budget_reject_clears_valid_and_margin() -> None:
    """Reject 模式：逐信道 valid=False，margin=inf，点值不变。"""
    fiber = _o_fiber("in_band")
    grid = _o_grid(41)
    path = _o_path(fiber)
    power = np.full(41, 1e-3)
    model = create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        symbol_rate_hz=O_SYMBOL_RATE_HZ,
        poly_degree=3,
        egn=True,
        fwm_validity_mode="reject",
    )
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)

    assert budget.validity.fwm_validity_mode == "reject"
    assert not bool(np.any(budget.validity.valid))
    assert bool(np.all(np.isinf(budget.margin_db)))
    assert bool(np.all(np.isfinite(result.snr_db)))
    assert any("reject" in note for note in budget.notes)
