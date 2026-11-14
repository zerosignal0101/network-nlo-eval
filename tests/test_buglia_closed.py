"""Closed-form Raman GN kernel against its independent integral."""

import numpy as np
import pytest

from network_nlo_eval.models.buglia2024_closed import (
    BugliaClosedProfile,
    _link_exact,
    eta_xpm_closed,
    fit_buglia_profile_5param,
    link_mu_closed,
    nli_closed_form,
    nli_integral,
)


def profile():
    return BugliaClosedProfile(
        alpha_npm=np.full(2, 4.6e-5),
        alpha_f_npm=np.full(2, 5e-5),
        alpha_b_npm=np.full(2, 5e-5),
        c_f=np.full(2, 1e-20),
        c_b=np.full(2, 2e-20),
        f_hz=np.array([194e12, 193.95e12]),
        bandwidths_hz=np.full(2, 32e9),
        f_hat_hz=206e12,
        length_m=80e3,
        rmse=np.zeros(2),
    )


def test_link_closed_form_matches_complex_integral() -> None:
    fitted = profile()
    phases = np.array([0, 1e-5, 5e-5, 1e-4])
    np.testing.assert_allclose(
        link_mu_closed(fitted, 0, phases),
        np.abs(_link_exact(fitted, 0, phases)) ** 2,
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="phase_slope"):
        eta_xpm_closed(fitted, 1, 0, 32e9, 0.0013, 1)


def test_fitted_profile_and_closed_form_against_frequency_integral() -> None:
    expected = profile()
    z = np.arange(1, 17) * expected.length_m / 16
    samples = np.stack([expected.value(z, i) for i in range(2)], axis=1)
    fitted = fit_buglia_profile_5param(
        samples,
        expected.length_m,
        expected.f_hz,
        expected.bandwidths_hz,
        expected.alpha_npm,
        expected.f_hat_hz,
        8,
    )
    for i in range(2):
        np.testing.assert_allclose(fitted.value(z, i), samples[:, i], rtol=3e-3)
    args = (
        expected,
        expected.bandwidths_hz,
        np.full(2, 1e-3),
        np.full(2, 0.0013),
        -2e-26,
        0.0,
        193.1e12,
    )
    spm, xpm = nli_closed_form(*args)
    spm_integral, xpm_integral = nli_integral(*args, n_frequency=64)
    np.testing.assert_allclose(xpm, xpm_integral, rtol=0.03)
    np.testing.assert_allclose(spm, spm_integral, rtol=0.15)
