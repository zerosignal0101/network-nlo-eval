"""Analytical attenuation limits of the accelerated Raman solver."""

import numpy as np

from network_nlo_eval.models.raman_numba import rk4_backward, rk4_forward


def test_forward_and_backward_passive_attenuation() -> None:
    z = np.linspace(0, 80e3, 101)
    alpha = np.array([4e-5])
    initial = np.array([1e-3])
    zero = np.zeros(1)
    forward = rk4_forward(
        np.zeros((1, 1)),
        np.array([0]),
        np.array([], dtype=int),
        alpha,
        np.zeros((1, 1)),
        zero,
        np.ones(1),
        z,
        initial,
        zero,
        np.empty((len(z), 0)),
        True,
        2.0,
        False,
    )
    expected = initial[0] * np.exp(-alpha[0] * z)
    np.testing.assert_allclose(forward[2][:, 0], expected, rtol=5e-8)
    np.testing.assert_array_equal(forward[3], 0)

    backward = rk4_backward(
        np.zeros((1, 1)),
        np.array([0]),
        alpha,
        np.ones(1),
        z,
        np.zeros((100, 3), dtype=int),
        np.zeros((100, 3)),
        initial,
        zero,
        np.zeros((len(z), 1)),
        True,
        False,
    )
    np.testing.assert_allclose(backward[2][:, 0], expected, rtol=5e-8)
    np.testing.assert_array_equal(backward[3], 0)
