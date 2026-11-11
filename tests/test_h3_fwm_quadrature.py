"""任务 BK：FWM 求积精度档（``fwm_quadrature_points``）的验收测试。

覆盖：

* 默认 24 与显式 ``fwm_quadrature_points=24`` 逐位相同（H3 与 plain pcfm）；
* 选项只作用于位移第三波项：``xci_edge_quadrature_points`` 不变，默认路径
  （``dense_grid_fwm_fix=False``）在 192 节点下也逐位不变；
* 96/192 节点确实改变 FWM 子集（求积生效），且非 dense 100 GHz 上逐位不变；
* 非法值（< 2）在 kernel / pcfm / pcfm_h3_raman 三层构造期 ``ValueError``；
* 所有输出无 NaN/Inf。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.pcfm import FWM_QUADRATURE_POINTS, PCFMSpanKernel

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _case(spacing_hz: float, span_count: int = 1) -> BenchCase:
    """与 BB/BE/BK 审计同源的 C 波段密集算例（64 GBaud, 100 km, NF 5.5 dB）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"bk_quad_{spacing_hz / 1e9:g}_{span_count}span",
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


def _assert_bitwise_equal(left, right) -> None:
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


def _fwm_subset(model, fiber, power_w: np.ndarray, length_m: float, base_xpm: np.ndarray) -> np.ndarray:
    """单跨 FWM 子集：修正 XPM 减去同一算例未修正 XPM (W)。"""
    span = model.kernel_for(fiber).evaluate(power_w, length_m)
    return np.asarray(span.xpm_noise_w, dtype=np.float64) - base_xpm


def test_default_matches_explicit_24_bitwise() -> None:
    """H3 与 plain pcfm：默认与显式 24 节点逐位相同。"""
    case = _case(50e9)
    path = build_path(case)
    power = launch_profile(case, 6.0)

    for name in ("pcfm", "pcfm_h3_raman"):
        implicit_model = make_model(name, case, dense_grid_fwm_fix=True)
        explicit_model = make_model(name, case, dense_grid_fwm_fix=True, fwm_quadrature_points=24)
        assert implicit_model.fwm_quadrature_points == FWM_QUADRATURE_POINTS == 24
        assert explicit_model.fwm_quadrature_points == 24
        _assert_bitwise_equal(
            implicit_model.evaluate_path(path, power),
            explicit_model.evaluate_path(path, power),
        )


def test_option_does_not_move_the_xci_edge_quadrature() -> None:
    """FWM 节点只改 FWM：XCI edge 档位与默认路径 NLI 均不变。"""
    case = _case(50e9)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        fwm_quadrature_points=192,
    )
    assert model.fwm_quadrature_points == 192
    # edge 档位保持模块默认，没有被 FWM 档位带动。
    assert model.xci_edge_quadrature_points == 24

    base = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    fixed = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=False,
        fwm_quadrature_points=192,
    ).evaluate_path(path, power)
    # FWM 关闭时该选项没有任何作用：默认路径逐位不变。
    _assert_bitwise_equal(base, fixed)


def test_96_and_192_change_the_fwm_subset_on_a_dense_plan() -> None:
    """50 GHz 密集点：96/192 节点确实改变 FWM 子集，方向为正。"""
    case = _case(50e9)
    path = build_path(case)
    fiber = path.fiber_configs[0]
    power = launch_profile(case, 6.0)
    base_model = make_model("pcfm_h3_raman", case)
    base_span = base_model.kernel_for(fiber).evaluate(power, 1.0e5)
    base_xpm = np.asarray(base_span.xpm_noise_w, dtype=np.float64)

    subsets = {}
    for nodes in (24, 96, 192):
        model = make_model(
            "pcfm_h3_raman",
            case,
            dense_grid_fwm_fix=True,
            fwm_quadrature_points=nodes,
        )
        assert model.last_dense_fwm_fix_applied is False  # 尚未求值
        result = model.evaluate_path(path, power)
        assert model.last_dense_fwm_fix_applied is True
        assert np.all(np.isfinite(np.asarray(result.snr_db, dtype=np.float64)))
        subsets[nodes] = _fwm_subset(model, fiber, power, 1.0e5, base_xpm)

    # 求积生效：192 与 24 的 FWM 子集不同（50 GHz 上更细的求积整体略小，
    # 方向由积分本身决定，这里只钉住"确实变了"和正值）。
    assert np.all(subsets[24] > 0.0)
    relative = np.max(np.abs(subsets[192] / np.maximum(subsets[24], 1e-300) - 1.0))
    assert relative > 1.0e-3, relative
    assert np.all(subsets[96] > 0.0) and np.all(subsets[192] > 0.0)


def test_non_dense_plan_is_bit_identical_with_finer_quadrature() -> None:
    """100 GHz 非 dense：96/192 节点不产生三元组，任何输出逐位不变。"""
    case = _case(100e9)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    base = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True).evaluate_path(path, power)
    for nodes in (96, 192):
        model = make_model(
            "pcfm_h3_raman",
            case,
            dense_grid_fwm_fix=True,
            fwm_quadrature_points=nodes,
        )
        fixed = model.evaluate_path(path, power)
        _assert_bitwise_equal(base, fixed)
        assert model.last_dense_fwm_fix_applied is False
        assert model.dense_fwm_fix_diagnostics()["triples"] == 0.0


@pytest.mark.parametrize("factory", ["kernel", "pcfm", "h3"])
def test_invalid_quadrature_points_are_rejected(factory: str) -> None:
    """非法节点数（<2）在 kernel / pcfm / pcfm_h3_raman 三层构造期 ValueError。"""
    if factory == "kernel":
        with pytest.raises(ValueError, match="fwm_quadrature_points"):
            PCFMSpanKernel(
                np.linspace(195.9e12, 191.9e12, 8),
                np.full(8, 50e9),
                SMF_100KM,
                fwm_quadrature_points=1,
            )
    else:
        name = "pcfm" if factory == "pcfm" else "pcfm_h3_raman"
        with pytest.raises(ValueError, match="fwm_quadrature_points"):
            create_model(
                name,
                SpectrumGrid(8, 50e9, 193.1e12),
                SMF_100KM,
                fwm_quadrature_points=0,
            )


def test_finer_quadrature_has_no_nan_or_inf_on_the_densest_grid() -> None:
    """25 GHz × 10 跨 × 192 节点：输出与 FWM 子集都是有限值。"""
    case = _case(25e9, 10)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        fwm_quadrature_points=192,
    )
    result = model.evaluate_path(path, power)
    for name in _ARRAY_FIELDS:
        values = np.asarray(getattr(result, name), dtype=np.float64)
        assert np.all(np.isfinite(values)), name
    diagnostics = model.dense_fwm_fix_diagnostics()
    assert diagnostics["triples"] > 0.0
    assert np.isfinite(diagnostics["weight_sum"])
    span = model.kernel_for(path.fiber_configs[0]).evaluate(power, 1.0e5)
    assert np.all(np.isfinite(np.asarray(span.xpm_noise_w, dtype=np.float64)))
