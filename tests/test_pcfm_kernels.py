"""Physical limits of the PCFM profile and SCI kernels."""

import numpy as np

from network_nlo_eval.models.numerical_kernels import _isrs_rhs_jit
from network_nlo_eval.models.pcfm_kernels import (
    _isrs_coefficient_matrix_jit,
    _profile_energy_jit,
    _sci_kernel_batch_jit,
    _sci_small_x_coefficients,
    _solve_spp_jit,
    r_coefficient_matrix,
)


def test_raman_coupling_and_segmented_loss() -> None:
    frequency = np.array([195e12, 190e12])
    power = np.array([1e-3, 2e-3])
    alpha = np.array([4e-5, 5e-5])
    matrix = _isrs_coefficient_matrix_jit(frequency)
    rhs = np.empty(2)
    _isrs_rhs_jit(frequency, power, alpha, rhs)
    np.testing.assert_allclose(rhs, power * (matrix @ power - alpha))

    profile = np.empty((400, 1))
    output = np.empty(1)
    _solve_spp_jit(
        np.zeros((1, 1)),
        np.array([4e-5]),
        np.array([1e-3]),
        np.array([40e3, 40e3]),
        np.array([1.0, 0.5]),
        np.array([100, 100]),
        profile,
        output,
    )
    np.testing.assert_allclose(output, [0.5e-3 * np.exp(-4e-5 * 80e3)], rtol=1e-8)
    np.testing.assert_allclose(
        profile[200], 0.5 * profile[199] * np.exp(-0.008), rtol=1e-6
    )


def test_zero_dispersion_sci_limit_and_profile_energy() -> None:
    # A constant spatial profile has unit normalised energy and SCI integral B²L².
    np.testing.assert_allclose(_profile_energy_jit(np.array([[1.0]])), [1.0])
    r = r_coefficient_matrix(1)[0][None, :]
    x = np.zeros(1)
    output = np.empty(1)
    _sci_kernel_batch_jit(
        r,
        x,
        x,
        x,
        np.ones(1),
        x,
        x,
        np.array([4.0]),
        3.0,
        _sci_small_x_coefficients(r.shape[1] - 1),
        2.0,
        output,
    )
    np.testing.assert_allclose(output, [36.0], rtol=1e-12)
