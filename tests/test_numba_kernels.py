"""Regression tests for physical-layer numerical kernels."""

import numpy as np
import pytest

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_fiber_attenuation_npm,
    _calc_span_noise_and_power_jit,
    _db_to_lin,
    _dbkm_to_power_npm,
    _dbm_to_w,
    _lin_to_db,
    _w_to_dbm,
)


def span_inputs() -> tuple:
    frequencies = np.array([193.05e12, 193.1e12, 193.15e12])
    return (
        np.array([0.001, 0.002, 0.0005]),
        80e3,
        frequencies - 193.1e12,
        frequencies,
        C_LIGHT / frequencies,
        np.array([0.2, 0.21, 0.19]) / 4.342944819032518 / 1000,
        np.array([80e-12, 81e-12, 82e-12]),
        -21.7e-27,
        0.12e-39,
        2.6e-20,
        np.array([3.2, 3.2, 3.2]),
        32e9,
        np.array([32e9, 45e9, 64e9]),
        1,
        0.028 / 1e3 / 1e12,
        0.0,
        15e12,
    )


def test_span_kernel_matches_reference_with_unequal_channel_widths() -> None:
    expected = (
        [2.5121097701423184e-05, 4.1785330262571184e-05, 1.5097884243747233e-05],
        [2.4842303750654026e-07, 1.294991890617807e-06, 1.4775064512885226e-08],
        [2.668400460685061e-07, 2.2043683612427492e-07, 1.5500380556197085e-07],
        [5.083207651666185e-07, 8.634501234399053e-07, 8.418189192953273e-07],
    )
    for actual, reference in zip(
        _calc_span_noise_and_power_jit(*span_inputs()), expected, strict=True
    ):
        np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=0)


def test_span_kernel_rejects_mismatched_bandwidth_count() -> None:
    inputs = list(span_inputs())
    inputs[12] = np.array([32e9])
    with pytest.raises(ValueError, match="one bandwidth per channel"):
        _calc_span_noise_and_power_jit(*inputs)


def test_physical_unit_conversions_and_wavelength_dependence() -> None:
    assert _dbm_to_w(0) == pytest.approx(0.001)
    assert _w_to_dbm(0.001) == pytest.approx(0)
    assert _w_to_dbm(0) == -np.inf
    assert _lin_to_db(_db_to_lin(13)) == pytest.approx(13)
    wavelengths = np.array([1550e-9, 1560e-9])
    np.testing.assert_allclose(
        _calc_aeff_dynamic(wavelengths, 80, 0.05, 1550), [80e-12, 80.5e-12]
    )
    attenuation = _calc_fiber_attenuation_npm(wavelengths, 1550, 0, 0, 0.2)
    np.testing.assert_allclose(attenuation, _dbkm_to_power_npm(0.2))
