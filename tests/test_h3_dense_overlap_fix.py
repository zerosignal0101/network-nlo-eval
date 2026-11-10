"""任务 AH：密集栅格重叠感知 XCI 修正的验收测试。

覆盖：

* 默认 ``dense_grid_fix=False`` 与未传该选项逐位一致，诊断字段全零；
* C 50 GHz 密集点：最大乐观误差从约 1.46 dB 降到 0.5 dB 以内；
* C 37.5 GHz 密集点：误差显著下降但仍有残差（记录边界，不假装已修复）；
* 非重叠 100 GHz 栅格：不触发新分支，数值逐位不变；
* 重叠对诊断字段与 quadrature 收敛；
* 无 NaN/Inf。
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _case(spacing_hz: float) -> BenchCase:
    """构造与精度矩阵同源的 C 波段单跨密集算例（64 GBaud, 100 km, 6 dBm）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"ah_dense_{spacing_hz / 1e9:g}",
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


def _evaluate(spacing_hz: float, *, dense_grid_fix: bool, **options):
    """返回 ``(model, path, result, reference)``。"""
    case = _case(spacing_hz)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case, dense_grid_fix=dense_grid_fix, **options)
    result = model.evaluate_path(path, power)
    reference = make_model("numerical", case).evaluate_path(path, power)
    return model, path, result, reference


def _max_error(result, reference) -> float:
    """逐信道最大 ``H3 - numerical`` SNR 差 (dB)。"""
    return float(np.max(np.asarray(result.snr_db) - np.asarray(reference.snr_db)))


def _report(model, path):
    """取第一跨的 validity report。"""
    return model.last_span_validity[0]


def test_default_is_bit_identical_and_has_zero_dense_fields() -> None:
    """默认关闭：与不传选项逐位一致，三个新诊断字段全零/False。"""
    case = _case(50e9)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    implicit = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    model = make_model("pcfm_h3_raman", case, dense_grid_fix=False)
    explicit = model.evaluate_path(path, power)

    for name in _ARRAY_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(implicit, name)), np.asarray(getattr(explicit, name))
        ), name

    report = _report(model, path)
    assert report.dense_grid_fix_applied is False
    assert np.all(np.asarray(report.dense_grid_overlap_correction) == 0.0)
    assert np.all(np.asarray(report.dense_grid_overlap_pairs) == 0.0)


def test_c50_overlap_fix_reduces_the_max_error_below_half_db() -> None:
    """C 50 GHz：修复前约 1.46 dB，修复后 <0.5 dB。"""
    _model, _path, base, reference = _evaluate(50e9, dense_grid_fix=False)
    model, path, fixed, _reference = _evaluate(50e9, dense_grid_fix=True)
    base_error = _max_error(base, reference)
    fixed_error = _max_error(fixed, reference)

    assert 1.3 < base_error < 1.6, base_error
    assert fixed_error < 0.5, fixed_error
    assert fixed_error < base_error - 0.7, (base_error, fixed_error)

    report = _report(model, path)
    assert report.dense_grid_fix_applied is True
    assert float(np.max(report.dense_grid_overlap_pairs)) >= 1.0
    assert float(np.max(report.dense_grid_overlap_correction)) > 0.0
    assert np.all(np.isfinite(report.dense_grid_overlap_correction))
    assert np.all(np.isfinite(report.dense_grid_overlap_pairs))
    assert np.all(np.isfinite(fixed.snr_db))


def test_c37_5_overlap_fix_improves_but_leaves_a_documented_residual() -> None:
    """C 37.5 GHz：2.42 dB -> 约 1.91 dB；残差保留，不宣称已修复。"""
    _model, _path, base, reference = _evaluate(37.5e9, dense_grid_fix=False)
    model, path, fixed, _reference = _evaluate(37.5e9, dense_grid_fix=True)
    base_error = _max_error(base, reference)
    fixed_error = _max_error(fixed, reference)

    assert 2.2 < base_error < 2.6, base_error
    assert fixed_error < base_error - 0.3, (base_error, fixed_error)
    assert fixed_error > 0.5, fixed_error  # 仍超出模型误差预算，必须保持可见

    report = _report(model, path)
    assert report.dense_grid_fix_applied is True
    assert float(np.max(report.dense_grid_overlap_pairs)) >= 1.0


def test_non_overlapping_100ghz_grid_is_untouched() -> None:
    """100 GHz 栅格不满足重叠条件：修复开关不改变任何数值。"""
    _model, _path, base, _reference = _evaluate(100e9, dense_grid_fix=False)
    model, path, fixed, _reference = _evaluate(100e9, dense_grid_fix=True)

    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(base, name)), np.asarray(getattr(fixed, name))), name

    report = _report(model, path)
    assert report.dense_grid_fix_applied is False
    assert np.all(np.asarray(report.dense_grid_overlap_pairs) == 0.0)


def test_quadrature_nodes_are_converged_at_the_default_24() -> None:
    """默认 24 节点与 fast 档 12 节点的 SNR 差 <0.01 dB。"""
    _model, _path, default, _reference = _evaluate(50e9, dense_grid_fix=True)
    _model, _path, fast, _reference = _evaluate(50e9, dense_grid_fix=True, resolution="fast")
    difference = np.max(np.abs(np.asarray(default.snr_db) - np.asarray(fast.snr_db)))
    assert float(difference) < 0.01, float(difference)


def test_overlap_fix_rejects_no_nan_on_an_extreme_grid() -> None:
    """25 GHz 极密栅格：修复后仍有限，且误差比默认小。"""
    _model, _path, base, reference = _evaluate(25e9, dense_grid_fix=False)
    _model, _path, fixed, _reference = _evaluate(25e9, dense_grid_fix=True)
    assert np.all(np.isfinite(fixed.snr_db))
    assert _max_error(fixed, reference) < _max_error(base, reference)
