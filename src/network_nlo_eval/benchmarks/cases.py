"""Canonical benchmark cases for cross-model QoT comparison.

The band sizes follow the reference system of Lasagni et al. 2023 (the ECZ baseline
paper): 64 GBaud channels on a 75 GHz grid, no guard bands, covering C+L (11.4 THz),
S+C+L (20.8 THz) or E+S+C+L (35.9 THz) over a 10x100 km SMF link. Keeping the case
definition in one module is what makes "same conditions" checkable: every model is
driven through :func:`build_path` and :func:`launch_profile`, never by its own
hand-rolled path.

Absolute band edges used here (frequency, THz): L 184.4-191.6, C 191.6-195.9,
S 195.9-205.4, E 205.4-220.4. The engine models a contiguous uniform grid, so a
multi-band case is the contiguous span covering the same total width; the per-band
gaps of a real comb are not represented. That limitation is deliberate and is stated
in the comparison report rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "BENCH_CASES",
    "POWER_SWEEP_DBM",
    "SMF_100KM",
    "BenchCase",
    "band_edges_hz",
    "build_case",
    "build_path",
    "grid_for",
    "launch_profile",
]

#: 每信道入纤功率扫描区间 (dBm)。覆盖线性区到明显非线性区。
POWER_SWEEP_DBM: tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

#: 参考光纤：SMF，衰减用二次多项式（与 ECZ 基线一致的系数）。
SMF_100KM = FiberSpanConfig(
    length_km=100.0,
    max_span_length_km=100.0,
    attenuation_db_km_ref=0.2,
    use_attenuation_polynomial=True,
    attenuation_alpha2=3.7685e-6,
    attenuation_alpha1=-7.3764e-5,
    attenuation_alpha0=0.162,
    reference_wavelength_nm=1550.0,
    dispersion_parameter_d=17.0,
    dispersion_slope_s=0.06,
    nonlinear_index_n2=2.6e-20,
    effective_area_um2_ref=80.0,
    aeff_slope_um2_nm=0.05,
)

#: 名义波段边界 (Hz)，用于报告与算例构造。
_BANDS_HZ = {
    "L": (184.4e12, 191.6e12),
    "C": (191.6e12, 195.9e12),
    "S": (195.9e12, 205.4e12),
    "E": (205.4e12, 220.4e12),
}


def band_edges_hz(bands: str) -> tuple[float, float]:
    """返回给定波段集合的连续频率范围 ``(low, high)``。

    Args:
        bands: 形如 ``"C+L"`` 或 ``"E+S+C+L"`` 的组合串，字母取自 L/C/S/E。

    Returns
    -------
        最低与最高频率 (Hz)。

    Raises
    ------
        ValueError: 波段字母非法或为空。
    """
    letters = [part for part in bands.replace("+", " ").split() if part]
    if not letters:
        raise ValueError("bands must name at least one of L, C, S, E.")
    unknown = [letter for letter in letters if letter not in _BANDS_HZ]
    if unknown:
        raise ValueError(f"unknown bands {unknown}; expected letters from {sorted(_BANDS_HZ)}.")
    lows = [_BANDS_HZ[letter][0] for letter in letters]
    highs = [_BANDS_HZ[letter][1] for letter in letters]
    return min(lows), max(highs)


@dataclass(frozen=True)
class BenchCase:
    """一个可复现的基准算例。

    Attributes
    ----------
        name: 算例键。
        bands: 波段组合串，仅用于报告。
        num_channels: 信道数。
        channel_spacing_hz: 信道间隔 (Hz)。
        center_frequency_hz: 网格中心频率 (Hz)。
        symbol_rate_hz: 符号率 (Hz)。
        span_length_km: 单跨长度 (km)。
        span_count: 跨段数。
        edfa_noise_figure_db: 每跨 EDFA 噪声系数 (dB)。
        total_power_dbm: 标称总入纤功率 (dBm)，用于把"每信道功率"解释成论文口径。
    """

    name: str
    bands: str
    num_channels: int
    channel_spacing_hz: float
    center_frequency_hz: float
    symbol_rate_hz: float
    span_length_km: float
    span_count: int
    edfa_noise_figure_db: float
    total_power_dbm: float

    @property
    def per_channel_power_dbm(self) -> float:
        """标称总功率摊到每信道的 dBm 值。"""
        return self.total_power_dbm - 10.0 * np.log10(self.num_channels)


def _case(
    name: str,
    bands: str,
    num_channels: int,
    spacing_hz: float,
    span_count: int,
    total_power_dbm: float,
) -> BenchCase:
    """由波段范围与信道数反推网格中心，避免手填中心频率写错半个带宽。"""
    low, high = band_edges_hz(bands)
    center = (low + high) / 2.0
    return BenchCase(
        name=name,
        bands=bands,
        num_channels=num_channels,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=center,
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=span_count,
        edfa_noise_figure_db=5.5,
        total_power_dbm=total_power_dbm,
    )


#: 算例表。C+L/S+C+L/E+S+C+L 的宽度与 Lasagni 2023 的 11.4/20.8/35.9 THz 一致。
BENCH_CASES: dict[str, BenchCase] = {
    "cl_152ch": _case("cl_152ch", "C+L", 152, 75e9, 10, 21.0),
    "scl_277ch": _case("scl_277ch", "S+C+L", 277, 75e9, 10, 23.16),
    "escl_479ch": _case("escl_479ch", "E+S+C+L", 479, 75e9, 10, 25.0),
    "escl_479ch_1span": _case("escl_479ch_1span", "E+S+C+L", 479, 75e9, 1, 25.0),
    "c_80ch": _case("c_80ch", "C", 80, 50e9, 10, 20.0),
    "scaling_40ch": _case("scaling_40ch", "C", 40, 75e9, 1, 16.0),
    "scaling_80ch": _case("scaling_80ch", "C", 80, 75e9, 1, 19.0),
    "scaling_160ch": _case("scaling_160ch", "C+L", 160, 75e9, 1, 22.0),
    "scaling_320ch": _case("scaling_320ch", "S+C+L", 320, 75e9, 1, 25.0),
    "scaling_479ch": _case("scaling_479ch", "E+S+C+L", 479, 75e9, 1, 27.0),
}


def build_case(name: str) -> BenchCase:
    """按名称取算例。

    Raises
    ------
        KeyError: 名称不存在。
    """
    try:
        return BENCH_CASES[name]
    except KeyError:
        raise KeyError(f"unknown benchmark case {name!r}; available: {sorted(BENCH_CASES)}.") from None


def grid_for(case: BenchCase) -> SpectrumGrid:
    """该算例的频谱网格。"""
    return SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz)


def build_path(case: BenchCase) -> MultiSpanOpticalPath:
    """该算例的光路：均匀跨段 + 等 NF 的 EDFA，无 ROADM。

    无 ROADM 是有意的：基准要隔离"模型本身的单跨物理"，不掺入逐跳损耗/
    均衡/滤波惩罚的差异。路径级语义另由 ``tests/test_route_model.py`` 覆盖。
    """
    fiber = SMF_100KM.model_copy(update={"length_km": case.span_length_km, "max_span_length_km": case.span_length_km})
    edfa = EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)
    return MultiSpanOpticalPath([fiber] * case.span_count, [edfa] * case.span_count)


def launch_profile(case: BenchCase, power_dbm: float | None = None) -> NDArrayFloat:
    """均匀入纤功率谱 ``(N,)`` (W)。

    Args:
        case: 算例。
        power_dbm: 每信道功率 (dBm)；``None`` 时取算例的标称每信道功率。

    Returns
    -------
        ``(N,)`` 功率 (W)。
    """
    level = case.per_channel_power_dbm if power_dbm is None else power_dbm
    return np.full(case.num_channels, 1e-3 * 10 ** (level / 10.0), dtype=np.float64)
