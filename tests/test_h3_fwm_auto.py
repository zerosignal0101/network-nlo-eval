"""任务 BF：密集 FWM 相干自动选择的验收测试。

覆盖：

* 默认 ``"xci"`` 与显式 ``"xci"`` 逐位相同；
* 50 GHz（``max(B)/min(spacing) = 1.28``）10 跨：``"auto"`` 与 ``"xci"``
  逐位相同，且 ``fwm_noise_is_additive`` 为假；
* 25 / 37.5 GHz（2.56 / 1.71）10 跨：``"auto"`` 与 ``"incoherent"`` 逐位相同；
* 非 dense 100 GHz：``"auto"`` 与 ``"xci"`` 逐位相同；
* 关闭 ``dense_grid_fwm_fix`` 时 ``"auto"`` 不触发；
* 单跨时 ``"auto"`` 的路径结果与 ``"xci"`` 相同；
* ``fwm_noise_is_additive`` 与实际拆出的 FWM 字段一致；
* 阈值恰好 1.5 时进入 ``"incoherent"``，1.5 以下走 ``"xci"``；
* 非法值在 kernel / pcfm / h3 三层构造期 ``ValueError``；
* 所有输出无 NaN/Inf。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.pcfm import (
    FWM_AUTO_RATIO_THRESHOLD,
    PCFMSpanKernel,
    fwm_coherence_ratio,
    resolve_fwm_coherence_mode,
)

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _case(spacing_hz: float, span_count: int = 1) -> BenchCase:
    """与 BE 审计同源的 C 波段算例（64 GBaud, 100 km, NF 5.5 dB）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"bf_auto_{spacing_hz / 1e9:g}_{span_count}span",
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


def _run(case: BenchCase, power_dbm: float = 6.0, **options):
    path = build_path(case)
    power = launch_profile(case, power_dbm)
    model = make_model("pcfm_h3_raman", case, **options)
    return model.evaluate_path(path, power), model


def test_pure_resolver_threshold_and_ratio() -> None:
    """栅格量本身：比值的定义与 1.5 阈值的两侧。"""
    spacing = 25e9
    frequencies = 193.1e12 + spacing * np.arange(8)
    bandwidths = np.full(8, 64e9)
    assert fwm_coherence_ratio(frequencies, bandwidths) == pytest.approx(2.56)
    # 稀疏计划（100 GHz）report 0.64。
    sparse = 193.1e12 + 100e9 * np.arange(8)
    assert fwm_coherence_ratio(sparse, bandwidths) == pytest.approx(0.64)
    # 单信道没有可比的间隔：永远不会触发 auto。
    assert np.isinf(fwm_coherence_ratio(frequencies[:1], bandwidths[:1]))

    assert FWM_AUTO_RATIO_THRESHOLD == 1.5
    assert resolve_fwm_coherence_mode("auto", 1.5, fwm_fix_applies=True) == "incoherent"
    assert resolve_fwm_coherence_mode("auto", 1.4999, fwm_fix_applies=True) == "xci"
    assert resolve_fwm_coherence_mode("auto", 2.56, fwm_fix_applies=False) == "xci"
    assert resolve_fwm_coherence_mode("xci", 2.56, fwm_fix_applies=True) == "xci"
    assert resolve_fwm_coherence_mode("incoherent", 0.64, fwm_fix_applies=False) == "incoherent"


def test_default_xci_is_bit_identical_to_explicit() -> None:
    """默认值仍走 ``"xci"``，与显式值逐位相同。"""
    case = _case(25e9, 10)
    implicit, implicit_model = _run(case, dense_grid_fwm_fix=True)
    explicit, explicit_model = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="xci")

    _assert_bitwise_equal(implicit, explicit)
    assert implicit_model.fwm_coherence_mode == "xci"
    assert explicit_model.fwm_coherence_mode == "xci"
    assert implicit_model.fwm_noise_is_additive is False
    assert explicit_model.fwm_noise_is_additive is False


def test_50ghz_auto_stays_on_xci() -> None:
    """50 GHz（ratio 1.28）10 跨：auto 与 xci 逐位相同，路由层不期待 additive FWM。"""
    case = _case(50e9, 10)
    xci, _ = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="xci")
    auto, auto_model = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="auto")
    incoherent, _ = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="incoherent")

    _assert_bitwise_equal(xci, auto)
    assert auto_model.fwm_coherence_ratio == pytest.approx(1.28)
    assert auto_model.fwm_coherence_mode == "xci"
    assert auto_model.fwm_noise_is_additive is False
    # 与显式拆分确实不同，说明 50 GHz 的分支被真正区分开，而不是恒等。
    assert not np.array_equal(np.asarray(xci.xpm_noise_w), np.asarray(incoherent.xpm_noise_w))
    # 拆分只影响跨间累加：单跨 FWM 仍留在 xpm 里。
    assert auto_model.last_dense_fwm_fix_applied is True


@pytest.mark.parametrize(
    ("spacing_hz", "expected_ratio"),
    [(37.5e9, 64.0 / 37.5), (25e9, 64.0 / 25.0)],
)
def test_dense_auto_matches_incoherent(spacing_hz: float, expected_ratio: float) -> None:
    """37.5 / 25 GHz 10 跨：auto 取到 incoherent，结果逐位相同。"""
    case = _case(spacing_hz, 10)
    auto, auto_model = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="auto")
    incoherent, _ = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="incoherent")
    xci, _ = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="xci")

    _assert_bitwise_equal(auto, incoherent)
    assert auto_model.fwm_coherence_ratio == pytest.approx(expected_ratio)
    assert auto_model.fwm_coherence_mode == "incoherent"
    assert auto_model.fwm_noise_is_additive is True
    assert not np.array_equal(np.asarray(auto.xpm_noise_w), np.asarray(xci.xpm_noise_w))


def test_non_dense_100ghz_auto_is_xci() -> None:
    """非 dense 100 GHz：auto 解析成 xci，逐位不变。"""
    case = _case(100e9, 10)
    xci, _ = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="xci")
    auto, auto_model = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="auto")

    _assert_bitwise_equal(xci, auto)
    assert auto_model.dense_grid_plan is False
    assert auto_model.fwm_coherence_mode == "xci"
    assert auto_model.fwm_noise_is_additive is False
    assert auto_model.last_dense_fwm_fix_applied is False


def test_auto_without_fwm_fix_is_xci() -> None:
    """``dense_grid_fwm_fix=False`` 时 auto 不触发，与 xci 逐位相同。"""
    case = _case(25e9, 10)
    xci, _ = _run(case, dense_grid_fwm_coherence="xci")
    auto, auto_model = _run(case, dense_grid_fwm_coherence="auto")

    _assert_bitwise_equal(xci, auto)
    assert auto_model.fwm_coherence_mode == "xci"
    assert auto_model.fwm_noise_is_additive is False
    assert auto_model.last_dense_fwm_fix_applied is False


def test_single_span_auto_matches_xci() -> None:
    """单跨没有跨间相位：auto 即使解析成 incoherent，路径结果也与 xci 相同。"""
    case = _case(25e9, 1)
    xci, _ = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="xci")
    auto, auto_model = _run(case, dense_grid_fwm_fix=True, dense_grid_fwm_coherence="auto")

    _assert_bitwise_equal(xci, auto)
    assert auto_model.fwm_coherence_mode == "incoherent"


@pytest.mark.parametrize("spacing_hz", [50e9, 37.5e9, 25e9])
def test_additive_flag_matches_the_kernel_split(spacing_hz: float) -> None:
    """``fwm_noise_is_additive`` 与实际 ``fwm_noise_w`` 字段一一对应。"""
    case = _case(spacing_hz, 3)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    for mode in ("xci", "incoherent", "auto"):
        model = make_model(
            "pcfm_h3_raman",
            case,
            dense_grid_fwm_fix=True,
            dense_grid_fwm_coherence=mode,
        )
        result = model.evaluate_path(path, power)
        kernel = model.kernel_for(path.fiber_configs[0])
        span = kernel.evaluate(power, 1.0e5, 1)
        additive = model.fwm_noise_is_additive
        assert additive is (span.fwm_noise_w is not None), mode
        assert additive is (kernel.fwm_coherence_mode == "incoherent")
        assert additive is (model.fwm_coherence_mode == "incoherent")
        assert np.all(np.isfinite(result.snr_db))
        assert np.all(np.isfinite(result.xpm_noise_w))


@pytest.mark.parametrize("spacing_hz", [50e9, 37.5e9, 25e9])
@pytest.mark.parametrize("span_count", [1, 5])
def test_auto_outputs_are_finite(spacing_hz: float, span_count: int) -> None:
    """auto（以及两个显式模式）在任何点上都不产生 NaN/Inf。"""
    case = _case(spacing_hz, span_count)
    for mode in ("xci", "incoherent", "auto"):
        result, _ = _run(
            case,
            6.0,
            dense_grid_fwm_fix=True,
            dense_grid_fwm_coherence=mode,
        )
        for name in _ARRAY_FIELDS:
            assert np.all(np.isfinite(np.asarray(getattr(result, name)))), (mode, name)


@pytest.mark.parametrize("factory", ["kernel", "pcfm", "h3"])
def test_invalid_coherence_mode_is_rejected(factory: str) -> None:
    """非法口径在 kernel / pcfm / h3 三层构造期都 ValueError。"""
    if factory == "kernel":
        with pytest.raises(ValueError, match="dense_grid_fwm_coherence"):
            PCFMSpanKernel(
                np.linspace(195.9e12, 191.9e12, 8),
                np.full(8, 50e9),
                SMF_100KM,
                dense_grid_fwm_coherence="automatic",
            )
    else:
        name = "pcfm" if factory == "pcfm" else "pcfm_h3_raman"
        with pytest.raises(ValueError, match="dense_grid_fwm_coherence"):
            create_model(
                name,
                SpectrumGrid(8, 50e9, 193.1e12),
                SMF_100KM,
                dense_grid_fwm_coherence="automatic",
            )
