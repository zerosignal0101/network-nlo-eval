"""Key numerical ISRS-GN regression and physical-limit tests."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    SpanContext,
    available_models,
    create_model,
)
from network_nlo_eval.models.numerical import NumericalISRSGN, NumericalISRSGNKernel
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


def test_span_matches_reference_with_coherent_amplitudes() -> None:
    fiber = FiberSpanConfig(length_km=80)
    kernel = NumericalISRSGNKernel(
        np.array([193.05e12, 193.1e12, 193.15e12]),
        np.full(3, 32e9),
        fiber,
        quadrature_points=8,
        ode_steps=64,
    )
    result = kernel.evaluate(
        np.array([0.001, 0.002, 0.0005]),
        80e3,
        span_context=SpanContext(1, 2, 80e3, 80e3),
    )
    reference = {
        "power_out_w": [
            2.5121106585461745e-05,
            5.0237002232318273e-05,
            1.2557948343879239e-05,
        ],
        "spm_noise_w": [
            2.3403173168475265e-07,
            1.8754351907233058e-06,
            2.9353479663268728e-08,
        ],
        "xpm_noise_w": [
            3.70244771533665e-07,
            2.246433002641916e-07,
            2.0284272698537854e-07,
        ],
        "nli_amplitude_spm": [
            0.0001119923975326673 + 0.00047062664032079365j,
            0.00031703134229644326 + 0.0013322636070706958j,
            3.9662580335140275e-05 + 0.0001666744113066768j,
        ],
        "nli_amplitude_xpm": [
            -1.0898207667656366e-05 + 0.0006083798160715866j,
            -8.489015908714815e-06 + 0.00047388947748720187j,
            -8.066596476397654e-06 + 0.00045030840210534106j,
        ],
    }
    for name, expected in reference.items():
        np.testing.assert_allclose(getattr(result, name), expected, rtol=1e-11, atol=0)
    np.testing.assert_array_equal(result.fwm_noise_w, 0)


def test_zero_dispersion_single_channel_closed_form_and_registry() -> None:
    fiber = FiberSpanConfig(
        length_km=100,
        max_span_length_km=100,
        use_attenuation_polynomial=False,
        attenuation_db_km_ref=0.162,
        dispersion_parameter_d=0,
        dispersion_slope_s=0,
        reference_wavelength_nm=1550,
        nonlinear_index_n2=2.6e-20,
        effective_area_um2_ref=80,
        aeff_slope_um2_nm=0,
    )
    grid = SpectrumGrid(1, 75e9, 299792458.0 / 1550e-9)
    model = create_model(
        "numerical",
        grid,
        fiber,
        symbol_rate_hz=64e9,
        quadrature_points=16,
        ode_steps=256,
    )
    result = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [EDFAConfig()]), np.array([1e-3])
    )
    gamma = 2 * np.pi * 2.6e-20 / (1550e-9 * 80e-12)
    alpha = 0.162 * np.log(10) / 10 / 1e3
    effective_length = -np.expm1(-alpha * 100e3) / alpha
    expected = (4 / 9) * gamma**2 * (1e-3) ** 3 * effective_length**2
    np.testing.assert_allclose(result.spm_noise_w, [expected], rtol=1e-6)
    assert "numerical" in available_models()
    assert isinstance(model, NumericalISRSGN)
    with pytest.raises(ValueError, match="egn=True"):
        NumericalISRSGN(grid, fiber, egn=True)
