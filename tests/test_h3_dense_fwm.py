"""任务 BB：密集栅格 FWM/MCI 修正（``dense_grid_fwm_fix``）的验收测试。

覆盖：

* 默认 ``dense_grid_fwm_fix=False`` 与不传该选项逐位一致，诊断字段为假/零；
* C 50 GHz 密集点触发修正，NLI/SNR max|Δ| 显著下降；
* C 25 GHz 极密点触发修正，误差仍有限且大幅下降；
* 100 GHz 非密集点不触发，数值逐位不变；
* 所有输出无 NaN/Inf；
* 固定算例的误差方向：加入项为正，且把 NLI 往参考方向推；
* 修正项与 numerical 的 fwm 子集同量级（同一物理项，不是拟合倍率）；
* 与 ``dense_grid_fix`` / ``nli_fallback="numerical"`` 的组合在构造期被拒绝；
* quadrature 档位（12 vs 24 节点）收敛。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _case(spacing_hz: float, span_count: int = 1) -> BenchCase:
    """构造与 AH/AY 同源的 C 波段单跨密集算例（64 GBaud, 100 km）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"bb_dense_{spacing_hz / 1e9:g}_{span_count}span",
        bands="C",
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=span_count,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


def _evaluate(spacing_hz: float, *, fwm_fix: bool, power_dbm: float = 6.0, span_count: int = 1, **options):
    """返回 ``(model, result, reference)``。"""
    case = _case(spacing_hz, span_count)
    path = build_path(case)
    power = launch_profile(case, power_dbm)
    model = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=fwm_fix, **options)
    result = model.evaluate_path(path, power)
    reference = make_model("numerical", case).evaluate_path(path, power)
    return model, result, reference


def _nli(result) -> np.ndarray:
    """``spm + xpm`` 噪声功率 (W)。"""
    return np.asarray(result.spm_noise_w, dtype=np.float64) + np.asarray(
        result.xpm_noise_w, dtype=np.float64
    )


def _max_nli_error(result, reference) -> float:
    """逐信道最大 ``H3 - numerical`` NLI 差 (dB)。"""
    total = _nli(reference)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta = 10.0 * np.log10(np.maximum(_nli(result), 1e-300) / np.maximum(total, 1e-300))
    return float(np.max(np.abs(delta)))


def _max_snr_error(result, reference) -> float:
    """逐信道最大 ``H3 - numerical`` SNR 差 (dB)。"""
    return float(np.max(np.abs(np.asarray(result.snr_db) - np.asarray(reference.snr_db))))


def test_default_is_bit_identical_and_reports_no_fix() -> None:
    """默认关闭：与不传选项逐位一致，诊断全零/False。"""
    case = _case(50e9)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    implicit = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    model = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=False)
    explicit = model.evaluate_path(path, power)

    for name in _ARRAY_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(implicit, name)), np.asarray(getattr(explicit, name))
        ), name

    assert model.last_dense_fwm_fix_applied is False
    diagnostics = model.dense_fwm_fix_diagnostics()
    assert diagnostics["triples"] == 0.0
    assert diagnostics["weight_sum"] == 0.0
    assert diagnostics["ell_eq_i_share"] == 0.0


def test_c50_dense_plan_triggers_and_cuts_the_error_by_more_than_the_gate() -> None:
    """C 50 GHz：触发修正，NLI/SNR max|Δ| 至少下降 0.3 dB。"""
    model, fixed, reference = _evaluate(50e9, fwm_fix=True)
    _base_model, base, _ = _evaluate(50e9, fwm_fix=False)

    assert model.last_dense_fwm_fix_applied is True
    base_nli = _max_nli_error(base, reference)
    fixed_nli = _max_nli_error(fixed, reference)
    base_snr = _max_snr_error(base, reference)
    fixed_snr = _max_snr_error(fixed, reference)

    assert 1.3 < base_nli < 1.7, base_nli
    assert fixed_nli < base_nli - 0.3, (base_nli, fixed_nli)
    assert fixed_nli < 0.5, fixed_nli
    assert fixed_snr < base_snr - 0.3, (base_snr, fixed_snr)

    diagnostics = model.dense_fwm_fix_diagnostics()
    assert diagnostics["triples"] > 0.0
    assert 0.0 <= diagnostics["ell_eq_i_share"] < 1.0


def test_c25_dense_plan_lifts_the_worst_point_out_of_the_6db_error() -> None:
    """C 25 GHz：默认 6.3 dB 的 NLI 误差降到 0.5 dB 以内。"""
    model, fixed, reference = _evaluate(25e9, fwm_fix=True)
    _base_model, base, _ = _evaluate(25e9, fwm_fix=False)

    assert model.last_dense_fwm_fix_applied is True
    assert _max_nli_error(base, reference) > 5.0
    assert _max_nli_error(fixed, reference) < 0.5
    assert np.all(np.isfinite(fixed.snr_db))


def test_non_dense_100ghz_grid_is_untouched() -> None:
    """100 GHz 不满足 dense 判定：开关不改变任何数值，也不触发。"""
    model, fixed, _reference = _evaluate(100e9, fwm_fix=True)
    _base_model, base, _ = _evaluate(100e9, fwm_fix=False)

    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(base, name)), np.asarray(getattr(fixed, name))), name

    assert model.last_dense_fwm_fix_applied is False
    assert model.dense_fwm_fix_diagnostics()["triples"] == 0.0


def test_fix_is_finite_on_the_densest_grid_and_multi_span() -> None:
    """25 GHz × 10 跨：所有输出与诊断都是有限值。"""
    model, fixed, _reference = _evaluate(25e9, fwm_fix=True, span_count=10)
    for name in _ARRAY_FIELDS:
        values = np.asarray(getattr(fixed, name), dtype=np.float64)
        assert np.all(np.isfinite(values)), name
    assert model.last_dense_fwm_fix_applied is True


def test_error_direction_is_towards_the_reference_and_the_added_term_is_positive() -> None:
    """固定算例：加入项逐信道为正，NLI 误差方向由负转近零。"""
    _model, base, reference = _evaluate(50e9, fwm_fix=False)
    _model, fixed, _reference = _evaluate(50e9, fwm_fix=True)

    added = np.asarray(fixed.xpm_noise_w) - np.asarray(base.xpm_noise_w)
    assert np.all(added > 0.0)
    assert np.array_equal(np.asarray(fixed.spm_noise_w), np.asarray(base.spm_noise_w))

    total = _nli(reference)
    base_db = 10.0 * np.log10(_nli(base) / total)
    fixed_db = 10.0 * np.log10(_nli(fixed) / total)
    # 默认偏低（缺项），修正后不再是系统性的负偏差。
    assert float(np.mean(base_db)) < -1.0
    assert abs(float(np.mean(fixed_db))) < 0.25


def test_added_term_tracks_the_numerical_fwm_subset() -> None:
    """修正项与 numerical 的 fwm 子集同量级：比值中位数在 0.7–1.3。"""
    case = _case(50e9)
    path = build_path(case)
    power = launch_profile(case, 0.0)
    base = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    fixed = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True).evaluate_path(path, power)
    added = np.asarray(fixed.xpm_noise_w) - np.asarray(base.xpm_noise_w)

    reference_model = make_model("numerical", case)
    reference_model.evaluate_path(path, power)
    span = reference_model.kernel_for(path.fiber_configs[0]).evaluate(power, 1.0e5)
    fwm = np.asarray(span.fwm_noise_w, dtype=np.float64)

    ratio = added / fwm
    assert 0.7 < float(np.median(ratio)) < 1.3, float(np.median(ratio))
    assert float(np.min(ratio)) > 0.5
    assert float(np.max(ratio)) < 1.6


def test_conflicting_options_are_rejected_at_construction() -> None:
    """与 dense_grid_fix / numerical fallback 组合会双重计数，构造期拒绝。"""
    case = _case(50e9)
    with pytest.raises(ValueError, match="dense_grid_fix"):
        make_model("pcfm_h3_raman", case, dense_grid_fix=True, dense_grid_fwm_fix=True)
    with pytest.raises(ValueError, match="nli_fallback"):
        make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True, nli_fallback="numerical")


def test_quadrature_profile_is_converged() -> None:
    """默认 24 节点与 fast 档 12 节点的 NLI 差 < 0.15 dB。

    ``resolution="fast"`` 是显式速度档，12 节点在 64 GBaud 密集栅格上留下
    ~0.13 dB 的求积残差；默认 24 节点才是精度路径。这里把实测边界钉住，
    不把 fast 档包装成与默认等价。
    """
    _model, default, _reference = _evaluate(50e9, fwm_fix=True)
    _model, fast, _reference = _evaluate(50e9, fwm_fix=True, resolution="fast")
    difference = np.max(np.abs(10.0 * np.log10(_nli(default) / np.maximum(_nli(fast), 1e-300))))
    assert float(difference) < 0.15, float(difference)
