"""Numerical FWM diagnostic split (round 14, task AT).

The numerical ISRS-GN kernel splits Eq. (4) into the original ``spm+xpm``
fields (bit-for-bit unchanged) and a non-negative *subset* ``fwm`` whose
generated third wave leaves the CUT band.  The subset is diagnostic only: it is
already contained in ``spm+xpm``, so adding it to a closed-form model would
double-count the interaction.  These tests pin the split and the default
backward compatibility.
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.benchmarks.cases import BenchCase, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import SpanFiberOutput
from network_nlo_eval.models.numerical import NumericalISRSGNKernel
from network_nlo_eval.network.elements import FiberSpanConfig

_O_BAND_LOW_HZ = 221.1e12
_O_BAND_HIGH_HZ = 237.2e12
_SYMBOL_RATE_HZ = 96e9


def _fiber() -> FiberSpanConfig:
    """O-band SMF with the dispersion anchor at 1310 nm."""
    return FiberSpanConfig(
        length_km=80.0,
        max_span_length_km=80.0,
        attenuation_db_km_ref=0.35,
        reference_wavelength_nm=1310.0,
        dispersion_parameter_d=2.6,
        dispersion_slope_s=0.06,
        nonlinear_index_n2=2.6e-20,
        effective_area_um2_ref=80.0,
        aeff_slope_um2_nm=0.05,
    )


def _case(channels: int) -> BenchCase:
    """O-band case with a fixed 16.1 THz total width and 96 GBaud channels."""
    spacing = (_O_BAND_HIGH_HZ - _O_BAND_LOW_HZ) / (channels - 1)
    return BenchCase(
        name=f"fwm_split_{channels}",
        bands="O",
        num_channels=channels,
        channel_spacing_hz=spacing,
        center_frequency_hz=0.5 * (_O_BAND_LOW_HZ + _O_BAND_HIGH_HZ),
        symbol_rate_hz=_SYMBOL_RATE_HZ,
        span_length_km=80.0,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=6.0,
    )


def _kernel(channels: int) -> NumericalISRSGNKernel:
    case = _case(channels)
    return NumericalISRSGNKernel(
        grid_for(case).frequencies,
        np.full(channels, _SYMBOL_RATE_HZ),
        _fiber(),
    )


def _power(channels: int) -> np.ndarray:
    return np.full(channels, 10.0 ** (6.0 / 10.0) * 1e-3, dtype=np.float64)


def test_dense_oband_fwm_is_a_nonnegative_subset_of_total_nli() -> None:
    """Dense O-band: ``0 <= fwm <= spm+xpm`` on every channel."""
    output = _kernel(161).evaluate(_power(161), 80e3)
    spm = np.asarray(output.spm_noise_w)
    xpm = np.asarray(output.xpm_noise_w)
    fwm = np.asarray(output.fwm_noise_w)
    assert fwm.shape == spm.shape == xpm.shape
    assert np.all(np.isfinite(fwm))
    assert np.all(fwm >= 0.0)
    assert np.all(fwm <= spm + xpm + 1e-30)
    assert float(np.max(fwm / (spm + xpm))) > 0.05


def test_sparse_grid_has_no_fwm_in_the_quadrature_domain() -> None:
    """A 402.5 GHz grid has no third wave inside the two-band quadrature window."""
    output = _kernel(41).evaluate(_power(41), 80e3)
    assert output.fwm_noise_w is not None
    assert float(np.max(output.fwm_noise_w)) == 0.0


def test_span_fiber_output_fwm_defaults_to_none() -> None:
    """The new field is optional and does not alter legacy constructors."""
    output = SpanFiberOutput(
        power_out_w=np.zeros(1),
        spm_noise_w=np.zeros(1),
        xpm_noise_w=np.zeros(1),
    )
    assert output.fwm_noise_w is None


def test_pcfm_default_path_is_unchanged_and_has_no_fwm_field() -> None:
    """The closed-form route result keeps its historical output shape."""
    case = _case(41)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    result = make_model("pcfm", case).evaluate_path(path, power)
    assert result.snr_db.shape == (case.num_channels,)
    assert not hasattr(result, "fwm_noise_w")
