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
from typing import TYPE_CHECKING

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.power_conventions import resolve_launch_power_convention
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayFloat

__all__ = [
    "BENCH_CASES",
    "POWER_SWEEP_DBM",
    "SMF_100KM",
    "TILT_REFERENCES",
    "BenchCase",
    "band_edges_hz",
    "build_case",
    "build_path",
    "grid_for",
    "launch_profile",
]

#: Tilt anchors: preserve total comb power, or preserve center-channel power.
TILT_REFERENCES: tuple[str, ...] = ("total", "center")

#: Per-channel physical total power sweep in dBm.
POWER_SWEEP_DBM: tuple[float, ...] = (
    -4.0,
    -3.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
)

#: Reference SMF with the ECZ baseline attenuation polynomial.
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

#: Nominal band edges in Hz for reports and case construction.
_BANDS_HZ = {
    "L": (184.4e12, 191.6e12),
    "C": (191.6e12, 195.9e12),
    "S": (195.9e12, 205.4e12),
    "E": (205.4e12, 220.4e12),
}


def band_edges_hz(bands: str) -> tuple[float, float]:
    """Return the continuous frequency range spanning the requested bands.

    Parameters
    ----------
    bands : str
        Band names L, C, S or E joined by plus signs.

    Returns
    -------
    tuple of float
        Lowest and highest frequencies in Hz.

    Raises
    ------
    ValueError
        If the band list is empty or contains an unknown name.
    """
    letters = [part for part in bands.replace("+", " ").split() if part]
    if not letters:
        msg = "bands must name at least one of L, C, S, E."
        raise ValueError(msg)
    unknown = [letter for letter in letters if letter not in _BANDS_HZ]
    if unknown:
        msg = f"unknown bands {unknown}; expected letters from {sorted(_BANDS_HZ)}."
        raise ValueError(msg)
    lows = [_BANDS_HZ[letter][0] for letter in letters]
    highs = [_BANDS_HZ[letter][1] for letter in letters]
    return min(lows), max(highs)


@dataclass(frozen=True)
class BenchCase:
    """Reproducible benchmark case with grid, path and power settings.

    Parameters
    ----------
    name, bands : str
        Case name and band labels.
    num_channels : int
        Number of channels.
    channel_spacing_hz, center_frequency_hz, symbol_rate_hz : float
        Grid spacing, center frequency and symbol rate in Hz.
    span_length_km : float
        Span length in kilometers.
    span_count : int
        Number of spans.
    edfa_noise_figure_db : float
        Per-span EDFA noise figure in dB.
    total_power_dbm : float
        Nominal physical total launch power in dBm, independent of output convention.
    launch_power_convention : {"total", "per_pol"}, optional
        Output coordinate for the launch profile; dBm inputs remain physical total.
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
    launch_power_convention: str = "total"

    @property
    def per_channel_power_dbm(self) -> float:
        """Return nominal physical total launch power per channel in dBm."""
        return float(self.total_power_dbm - 10.0 * np.log10(self.num_channels))

    @property
    def per_channel_power_dbm_per_pol(self) -> float:
        """Return the corresponding per-polarization channel power in dBm."""
        return float(self.per_channel_power_dbm - 10.0 * np.log10(2.0))


def _case(
    name: str,
    bands: str,
    num_channels: int,
    spacing_hz: float,
    span_count: int,
    total_power_dbm: float,
    launch_power_convention: str = "total",
) -> BenchCase:
    """Construct a case with grid center determined from the band limits."""
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
        launch_power_convention=resolve_launch_power_convention(
            launch_power_convention
        ),
    )


#: Case widths follow the Lasagni 2023 reference system.
BENCH_CASES: dict[str, BenchCase] = {
    "cl_152ch": _case("cl_152ch", "C+L", 152, 75e9, 10, 21.0),
    "scl_277ch": _case("scl_277ch", "S+C+L", 277, 75e9, 10, 23.16),
    "escl_479ch": _case("escl_479ch", "E+S+C+L", 479, 75e9, 10, 25.0),
    "escl_479ch_1span": _case("escl_479ch_1span", "E+S+C+L", 479, 75e9, 1, 25.0),
    # The 50 GHz grid overlaps 64 GBaud channels and is an applicability
    # boundary case, outside the closed-form models' non-overlap assumption.
    "c_80ch_subnyquist": _case("c_80ch_subnyquist", "C", 80, 50e9, 10, 20.0),
    "c_57ch": _case("c_57ch", "C", 57, 75e9, 10, 20.0),
    "scaling_40ch": _case("scaling_40ch", "C", 40, 75e9, 1, 16.0),
    "scaling_80ch": _case("scaling_80ch", "C", 80, 75e9, 1, 19.0),
    "scaling_160ch": _case("scaling_160ch", "C+L", 160, 75e9, 1, 22.0),
    "scaling_320ch": _case("scaling_320ch", "S+C+L", 320, 75e9, 1, 25.0),
    "scaling_479ch": _case("scaling_479ch", "E+S+C+L", 479, 75e9, 1, 27.0),
}


def build_case(name: str) -> BenchCase:
    """Return a named benchmark case.

    Raises
    ------
    KeyError
        If the case name is unknown.
    """
    try:
        return BENCH_CASES[name]
    except KeyError:
        msg = f"unknown benchmark case {name!r}; available: {sorted(BENCH_CASES)}."
        raise KeyError(msg) from None


def grid_for(case: BenchCase) -> SpectrumGrid:
    """Return the case spectrum grid."""
    return SpectrumGrid(
        case.num_channels, case.channel_spacing_hz, case.center_frequency_hz
    )


def build_path(case: BenchCase) -> MultiSpanOpticalPath:
    """Build the case path with uniform spans and EDFAs, without ROADMs.

    The benchmark omits ROADMs to isolate model physical-layer differences.
    """
    fiber = SMF_100KM.model_copy(
        update={
            "length_km": case.span_length_km,
            "max_span_length_km": case.span_length_km,
        }
    )
    edfa = EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)
    return MultiSpanOpticalPath([fiber] * case.span_count, [edfa] * case.span_count)


def launch_profile(
    case: BenchCase,
    power_dbm: float | None = None,
    *,
    tilt_db: float = 0.0,
    tilt_reference: str = "total",
    launch_power_convention: str = "auto",
) -> NDArrayFloat:
    """Construct a launch power spectrum with optional linear dB tilt.

    Power increases with frequency. Input power in dBm always denotes physical
    per-channel total power; the returned array follows the requested convention.

    Parameters
    ----------
    case : BenchCase
        Benchmark case.
    power_dbm : float or None, optional
        Nominal per-channel total power in dBm; None uses the case nominal power.
    tilt_db : float, optional
        Nonnegative, finite maximum-to-minimum power ratio in dB.
    tilt_reference : {"total", "center"}, optional
        Preserve the flat spectrum's total power, or anchor the nominal power at
        the grid center without total-power normalization.
    launch_power_convention : {"auto", "total", "per_pol"}, optional
        Output coordinate; auto uses the case convention.

    Returns
    -------
    numpy.ndarray
        Per-channel launch powers in watts.

    Raises
    ------
    ValueError
        If the tilt, reference or convention is invalid, or a single channel has
        nonzero tilt.
    """
    convention = (
        case.launch_power_convention
        if launch_power_convention == "auto"
        else resolve_launch_power_convention(launch_power_convention)
    )
    convention = resolve_launch_power_convention(convention)
    level = case.per_channel_power_dbm if power_dbm is None else power_dbm
    base_w = 1e-3 * 10 ** (level / 10.0)
    if tilt_reference not in TILT_REFERENCES:
        msg = (
            f"tilt_reference must be one of {TILT_REFERENCES}, got {tilt_reference!r}."
        )
        raise ValueError(msg)
    if not np.isfinite(tilt_db) or tilt_db < 0.0:
        msg = f"tilt_db must be a finite non-negative value, got {tilt_db!r}."
        raise ValueError(msg)
    if tilt_db == 0.0:
        profile = np.full(case.num_channels, base_w, dtype=np.float64)
        return profile if convention == "total" else profile * 0.5
    if case.num_channels < 2:
        msg = "a non-zero tilt needs at least two channels."
        raise ValueError(msg)

    # Uniform dB steps run from -tilt/2 to +tilt/2.
    offsets_db = np.linspace(-0.5, 0.5, case.num_channels, dtype=np.float64) * float(
        tilt_db
    )
    power = base_w * 10.0 ** (offsets_db / 10.0)
    if tilt_reference == "total":
        # Preserve the total power of the flat spectrum.
        power *= (case.num_channels * base_w) / float(np.sum(power))
    if convention == "per_pol":
        # Convert coordinates only after total-power normalization.
        power = power * 0.5
    return np.ascontiguousarray(power)
