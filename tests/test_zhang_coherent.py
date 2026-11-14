"""Coherent Raman link, phase accumulation and calibration."""

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.models.buglia2024_closed import link_mu_closed
from network_nlo_eval.models.zhang2024_coherent import (
    CalibrationSet,
    CalibrationTerm,
    ZhangCoherentSpan,
    ZhangSpanProfile,
    _coherent_field,
    _to_buglia_profile,
    cfm_and_egn_terms,
    coherent_eta,
    fit_zhang_coherent_profile,
    link_function,
)


def make_profile() -> ZhangSpanProfile:
    return ZhangSpanProfile(
        alpha_npm=np.full(2, 4.6e-5),
        t_coeff=np.full(2, 0.1),
        a_npm=np.full(2, 8e-5),
        d_coeff=np.full(2, 0.03),
        d_npm=np.full(2, 5e-5),
        f_hz=np.array([194e12, 193.95e12]),
        bandwidths_hz=np.full(2, 32e9),
        length_m=80e3,
        rmse=np.zeros(2),
    )


def test_profile_fit_and_closed_link_poles() -> None:
    expected = make_profile()
    z = np.arange(1, 17) * expected.length_m / 16
    samples = np.stack([expected.value(z, i) for i in range(2)], axis=1)
    fitted = fit_zhang_coherent_profile(
        samples,
        expected.length_m,
        expected.f_hz,
        expected.bandwidths_hz,
        expected.alpha_npm,
        8,
    )
    for i in range(2):
        np.testing.assert_allclose(fitted.value(z, i), samples[:, i], rtol=3e-2)
    phases = np.array([0.0, 1e-5, 1e-4])
    np.testing.assert_allclose(
        link_mu_closed(_to_buglia_profile(expected), 0, phases),
        np.abs(link_function(expected, 0, phases)) ** 2,
        rtol=1e-12,
    )


def test_coherent_spans_and_calibration() -> None:
    profile = make_profile()
    power = np.ones(2) * 1e-3
    first = ZhangCoherentSpan(profile, power, 0)
    second = ZhangCoherentSpan(profile, power, profile.length_m)
    phases = np.array([0.0, 1e-5])
    field = _coherent_field([first, second], power, 0, phases, 1.0)
    expected = link_function(profile, 0, phases) * (
        1 + np.exp(1j * phases * profile.length_m)
    )
    np.testing.assert_allclose(field, expected)
    args = (power, profile.bandwidths_hz, np.full(2, 0.0013), -2e-26, 0.0, 193.1e12)
    one = coherent_eta([first], *args, n_frequency=16, n_spm_frequency=8)
    two = coherent_eta([first, second], *args, n_frequency=16, n_spm_frequency=8)
    assert all(np.all(x > 0) for x in (*one, *two))
    assert not np.allclose(two[0], 2 * one[0], rtol=1e-3, atol=0)
    term = CalibrationTerm("power", (2.0, 0.1, 1.0, 0.3))
    assert CalibrationSet((term,)).apply({"power": 2}) == pytest.approx(
        1 / (2 * np.exp(0.5))
    )
    assert CalibrationSet((term,)).apply({}) == 1


@pytest.mark.parametrize("zero_dispersion", [False, True])
def test_singular_closed_form_uses_finite_single_span_integral(
    zero_dispersion: bool,
) -> None:
    profile = make_profile()
    if not zero_dispersion:
        profile = replace(profile, d_npm=profile.a_npm.copy())
    power = np.full(2, 1e-3)
    span = ZhangCoherentSpan(profile, power, 0.0)
    cfm_spm, cfm_xpm, integral_spm, integral_xpm = cfm_and_egn_terms(
        [span],
        power,
        profile.bandwidths_hz,
        np.full(2, 0.0013),
        0.0 if zero_dispersion else -2e-26,
        0.0,
        193.1e12,
        n_frequency=16,
        n_spm_frequency=8,
    )
    np.testing.assert_allclose(cfm_spm, integral_spm, rtol=1e-12)
    np.testing.assert_allclose(cfm_xpm, integral_xpm, rtol=1e-12)
    assert np.all(cfm_spm > 0)
    assert np.all(cfm_xpm > 0)
