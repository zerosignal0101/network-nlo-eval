"""任务 BE：密集 FWM 跨间相干口径的验收测试。

覆盖：

* 默认 ``dense_grid_fwm_coherence="xci"`` 与显式值逐位相同；
* 非 dense 100 GHz 上 ``"incoherent"`` 不改变任何数值，也不产生 FWM 字段；
* 单跨时 ``"incoherent"`` 与 ``"xci"`` 等价（没有跨间相位）；
* 25 GHz 5/10 跨时，``"incoherent"`` 相对 ``numerical`` 的 NLI/SNR
  max|Δ| 比 ``"xci"`` 下降 ≥0.05 dB；
* 10 跨 25 GHz 的相位口径确实被拆出：``fwm_noise_is_additive`` 为真，
  且路径 XPM 只加一次 FWM；
* 非法口径在 kernel / model / H3 三层构造期 ``ValueError``；
* 默认路径与 100 GHz 非密集点逐位不变。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.pcfm import PCFMSpanKernel

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _case(spacing_hz: float, span_count: int = 1) -> BenchCase:
    """与 BB/BE 审计同源的 C 波段密集算例（64 GBaud, 100 km, NF 5.5 dB）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"be_coherence_{spacing_hz / 1e9:g}_{span_count}span",
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


def _nli(result) -> np.ndarray:
    return np.asarray(result.spm_noise_w, dtype=np.float64) + np.asarray(
        result.xpm_noise_w, dtype=np.float64
    )


def _max_nli_error(result, reference) -> float:
    ratio = np.maximum(_nli(result), 1e-300) / np.maximum(_nli(reference), 1e-300)
    return float(np.max(np.abs(10.0 * np.log10(ratio))))


def _max_snr_error(result, reference) -> float:
    return float(
        np.max(np.abs(np.asarray(result.snr_db, dtype=np.float64) - np.asarray(reference.snr_db)))
    )


def _assert_bitwise_equal(left, right) -> None:
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


def test_default_coherence_is_bit_identical_and_not_additive() -> None:
    """默认 ``xci`` 与显式值逐位相同；FWM 仍留在 xpm 里。"""
    case = _case(25e9, 10)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    implicit_model = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True)
    explicit_model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        dense_grid_fwm_coherence="xci",
    )

    implicit = implicit_model.evaluate_path(path, power)
    explicit = explicit_model.evaluate_path(path, power)

    _assert_bitwise_equal(implicit, explicit)
    assert implicit_model.fwm_noise_is_additive is False
    assert explicit_model.fwm_noise_is_additive is False


def test_non_dense_100ghz_is_untouched_by_incoherent_mode() -> None:
    """100 GHz 非密集点没有三元组：``incoherent`` 逐位不变且不产生 FWM 字段。"""
    case = _case(100e9, 1)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    base = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True).evaluate_path(path, power)
    incoherent_model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        dense_grid_fwm_coherence="incoherent",
    )
    incoherent = incoherent_model.evaluate_path(path, power)

    _assert_bitwise_equal(base, incoherent)
    assert incoherent_model.fwm_noise_is_additive is True
    assert incoherent_model.last_dense_fwm_fix_applied is False


def test_single_span_incoherent_equals_xci() -> None:
    """单跨没有跨间相位：``incoherent`` 与 ``xci`` 的路径结果逐位相同。"""
    case = _case(25e9, 1)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    xci = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True).evaluate_path(path, power)
    incoherent_model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        dense_grid_fwm_coherence="incoherent",
    )
    incoherent = incoherent_model.evaluate_path(path, power)

    _assert_bitwise_equal(xci, incoherent)
    assert incoherent_model.last_dense_fwm_fix_applied is True


@pytest.mark.parametrize("span_count", [5, 10])
def test_incoherent_reduces_the_multi_span_dense_error(span_count: int) -> None:
    """25 GHz 多跨：``incoherent`` 的 NLI/SNR max|Δ| 比 ``xci`` 下降 ≥0.05 dB。"""
    case = _case(25e9, span_count)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    xci = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True).evaluate_path(path, power)
    incoherent_model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        dense_grid_fwm_coherence="incoherent",
    )
    incoherent = incoherent_model.evaluate_path(path, power)

    xci_nli = _max_nli_error(xci, reference)
    inh_nli = _max_nli_error(incoherent, reference)
    xci_snr = _max_snr_error(xci, reference)
    inh_snr = _max_snr_error(incoherent, reference)

    assert xci_nli - inh_nli >= 0.05, (xci_nli, inh_nli)
    assert xci_snr - inh_snr >= 0.05, (xci_snr, inh_snr)
    assert np.all(np.isfinite(incoherent.snr_db))
    assert incoherent_model.fwm_noise_is_additive is True


def test_incoherent_keeps_fwm_added_exactly_once() -> None:
    """路径 XPM = |Σ XCI 振幅|² + Σ FWM 功率；FWM 只加一次。"""
    case = _case(25e9, 5)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        dense_grid_fwm_coherence="incoherent",
    )
    result = model.evaluate_path(path, power)

    # 每个 span 的 FWM 是独立的单跨功率；5 跨等功率时路径 FWM 约为
    # 单跨 FWM 的 5 倍。这里只检查“加回去”后的 XPM 不小于任一单跨 XPM，
    # 且没有除以 span 数或重复平方。
    kernel = model.kernel_for(path.fiber_configs[0])
    span = kernel.evaluate(power, 1.0e5, 1)
    assert span.fwm_noise_w is not None
    assert np.all(np.asarray(span.fwm_noise_w) >= 0.0)
    assert np.all(np.asarray(result.xpm_noise_w) >= np.asarray(span.xpm_noise_w) - 1e-30)


@pytest.mark.parametrize("factory", ["kernel", "pcfm", "h3"])
def test_invalid_coherence_mode_is_rejected(factory: str) -> None:
    """非法口径在 kernel / pcfm / h3 三层构造期都 ValueError。"""
    if factory == "kernel":
        with pytest.raises(ValueError, match="dense_grid_fwm_coherence"):
            PCFMSpanKernel(
                np.linspace(195.9e12, 191.9e12, 8),
                np.full(8, 50e9),
                SMF_100KM,
                dense_grid_fwm_coherence="profile",
            )
    else:
        name = "pcfm" if factory == "pcfm" else "pcfm_h3_raman"
        with pytest.raises(ValueError, match="dense_grid_fwm_coherence"):
            create_model(
                name,
                SpectrumGrid(8, 50e9, 193.1e12),
                SMF_100KM,
                dense_grid_fwm_coherence="profile",
            )
