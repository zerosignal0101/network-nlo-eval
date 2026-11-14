"""Focused physical and integration tests for the Semrau 2019 model."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    SpanContext,
    available_models,
    create_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.semrau2019 import (
    Semrau2019Model,
    _coherent_factor,
    _equivalent_phase_ramp,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


def test_multispan_egn_matches_reference_vector() -> None:
    fiber = FiberSpanConfig(length_km=240, max_span_length_km=80)
    grid = SpectrumGrid(5, 50e9, 193.1e12)
    model = Semrau2019Model(grid, fiber, 32e9, egn=True, excess_kurtosis=-0.68)
    result = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=5.5)]),
        np.array([1e-3, 2e-3, 0, 0.5e-3, 1.5e-3]),
    )
    expected = {
        "signal_power_w": [0.001, 0.002, 0, 0.0005, 0.0015000000000000002],
        "spm_noise_w": [
            9.849244405250054e-7,
            7.892652918509704e-6,
            0,
            1.23738151431184e-7,
            3.346554305189256e-6,
        ],
        "xpm_noise_w": [
            6.792353016145259e-7,
            5.525351966620966e-7,
            0,
            3.388443239531401e-7,
            4.1003387045405803e-7,
        ],
        "ase_noise_w": [
            1.6900938566711432e-6,
            1.6907886917319217e-6,
            0,
            1.6921788755295891e-6,
            1.6928742243369637e-6,
        ],
        "snr_db": [
            24.74404105593814,
            22.951643875381286,
            -np.inf,
            23.655708271573637,
            24.397375986575938,
        ],
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(getattr(result, name), reference, rtol=1e-12, atol=0)
    assert result.span_count == 3
    assert "semrau2019" in available_models()
    assert isinstance(create_model("semrau2019", grid, fiber), Semrau2019Model)


def test_no_raman_limit_matches_ecz_nonlinear_noise() -> None:
    fiber = FiberSpanConfig(length_km=80)
    grid = SpectrumGrid(8, 50e9, 193.1e12)
    power = np.full(8, 1e-3)
    semrau = Semrau2019Model(grid, fiber, 32e9, srs_cr_w_m_hz=0)
    ecz = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off")
    ecz.cr_w_m_hz = 0
    actual = semrau.kernel_for(fiber).evaluate(power, 80e3)
    reference = ecz.span_fiber_output(power, 80e3)
    np.testing.assert_allclose(actual.spm_noise_w, reference.spm_noise_w, rtol=1e-6)
    np.testing.assert_allclose(actual.xpm_noise_w, reference.xpm_noise_w, rtol=1e-6)
    np.testing.assert_allclose(
        actual.power_out_w, power * np.exp(-0.2 * np.log(10) / 10 * 80)
    )


def test_span_amplitudes_match_power_and_phase_changes_with_context() -> None:
    fiber = FiberSpanConfig(length_km=80)
    model = Semrau2019Model(SpectrumGrid(4, 50e9, 193.1e12), fiber, 32e9)
    kernel = model.kernel_for(fiber)
    power = np.full(4, 1e-3)
    first = kernel.evaluate(power, 80e3, span_context=SpanContext(0, 3, 0, 80e3))
    second = kernel.evaluate(power, 80e3, span_context=SpanContext(1, 3, 80e3, 80e3))
    np.testing.assert_allclose(np.abs(first.nli_amplitude_spm) ** 2, first.spm_noise_w)
    np.testing.assert_allclose(
        np.abs(second.nli_amplitude_xpm) ** 2, second.xpm_noise_w
    )
    assert np.any(np.angle(second.nli_amplitude_spm) != 0)
    assert kernel.evaluate(power, 80e3).nli_amplitude_spm is None
    with pytest.raises(ValueError, match="one entry per channel"):
        kernel.evaluate(np.ones(3), 80e3)


def test_coherence_factor_and_phase_ramp_limits() -> None:
    spans = 5
    incoherent = _coherent_factor(spans, np.zeros((spans - 1, 2)))
    coherent = _coherent_factor(spans, np.ones((spans - 1, 2)))
    np.testing.assert_allclose(incoherent, spans)
    np.testing.assert_allclose(coherent, spans**2)
    phase = _equivalent_phase_ramp(spans, coherent)
    np.testing.assert_allclose(phase, 0, atol=1e-10)
