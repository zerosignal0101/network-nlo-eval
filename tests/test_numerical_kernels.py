"""Key physical limits of the numerical ISRS-GN kernels."""

import numpy as np

from network_nlo_eval.models.numerical_kernels import (
    _gn_nli_jit,
    _raman_gain_jit,
    _solve_isrs_profile_jit,
)


def test_single_channel_profile_matches_attenuation_and_dark_channel() -> None:
    frequencies = np.array([193.1e12, 193.05e12])
    power = np.array([1e-3, 0.0])
    alpha = np.full(2, 4.6e-5)
    length = 80e3
    steps = 200
    profile, output = _solve_isrs_profile_jit(frequencies, power, alpha, length, steps)
    midpoint = (np.arange(steps) + 0.5) * length / steps
    np.testing.assert_allclose(profile[:, 0], np.exp(-alpha[0] * midpoint), rtol=1e-6)
    np.testing.assert_allclose(
        output, [power[0] * np.exp(-alpha[0] * length), 0], rtol=1e-8
    )
    np.testing.assert_array_equal(profile[:, 1], 1)
    assert _raman_gain_jit(0.0) == 0
    assert _raman_gain_jit(5e12) > 0


def test_single_channel_gn_has_only_self_interference() -> None:
    frequency = np.array([193.1e12])
    spm, xpm, fwm = (np.zeros(1) for _ in range(3))
    _gn_nli_jit(
        frequency,
        np.array([32e9]),
        np.array([1e-3]),
        np.array([1.3e-3]),
        np.array([4.6e-5]),
        np.ones((20, 1)),
        20,
        4e3,
        1,
        80e3,
        -2.1e-26,
        0.0,
        frequency[0],
        8,
        spm,
        xpm,
        fwm,
    )
    assert np.isfinite(spm[0])
    assert spm[0] > 0
    np.testing.assert_array_equal(xpm, 0)
    np.testing.assert_array_equal(fwm, 0)
