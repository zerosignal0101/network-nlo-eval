"""Correlation GN physical integrals, sparse grids and coherent accumulation."""

from typing import cast

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.cgn import (
    CGNModel,
    CGNSpanKernel,
    _effective_beta2,
    _island_core,
    _isrs_nsp_energy_spectrum,
    _mci_core,
    _mci_slot_table,
    _nsp_energy_spectrum,
    _third_wave_slots,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


def test_single_span_against_direct_gn_integration() -> None:
    frequencies = 193e12 + np.arange(-1, 2) * 75e9
    bandwidth = np.full(3, 64e9)
    power = np.full(3, 1e-3)
    fiber = FiberSpanConfig(length_km=100, dispersion_slope_s=0)
    kernel = CGNSpanKernel(
        fiber_config=fiber,
        f_abs_hz=frequencies,
        b_hz=bandwidth,
        correlation_points=48,
        include_isrs=False,
    )
    result = kernel.evaluate(power, 100e3)
    sci, xci = np.zeros(3), np.zeros(3)
    for cut in range(3):
        for second in range(3):
            x = np.linspace(-bandwidth[cut] / 2, bandwidth[cut] / 2, 401)
            offset = frequencies[second] - frequencies[cut]
            y = np.linspace(
                offset - bandwidth[second] / 2, offset + bandwidth[second] / 2, 401
            )
            xx, yy = x[:, None], y[None, :]
            omega = 4 * np.pi**2 * kernel.beta2_s2_m * xx * yy
            for third in range(3):
                low = frequencies[third] - frequencies[cut] - bandwidth[third] / 2
                high = low + bandwidth[third]
                mask = (xx + yy >= low) & (xx + yy < high)
                alpha = (kernel.alpha_npm[second] + kernel.alpha_npm[third]) / 2
                phase = alpha - 1j * omega
                response = kernel.gamma_w_m[cut] * (1 - np.exp(-phase * 100e3)) / phase
                integral = (
                    np.sum(abs(response) ** 2 * mask) * (x[1] - x[0]) * (y[1] - y[0])
                )
                contribution = (
                    (16 / 27) * np.prod(power / bandwidth) * integral * bandwidth[cut]
                )
                if second == third == cut:
                    sci[cut] += contribution
                else:
                    xci[cut] += (1 if second == cut else 2) * contribution
    np.testing.assert_allclose(result.spm_noise_w, sci, rtol=0.02)
    np.testing.assert_allclose(result.xpm_noise_w, xci, rtol=0.08)


def test_isrs_segmented_spectrum_matches_exponential() -> None:
    alpha = np.array([[[[3.7e-5]]]])
    gamma = np.array([[[[1.3e-3]]]])
    omega = np.linspace(0, 5, 11)[None, None, :]
    residual = np.ones((1, 1, 8))
    plain = _nsp_energy_spectrum(gamma, alpha, 100e3, omega)
    segmented = _isrs_nsp_energy_spectrum(gamma, alpha, residual, 100e3 / 8, omega)
    np.testing.assert_allclose(segmented, plain, rtol=1e-12)


def test_sparse_isrs_cache_and_multispan_coherence(monkeypatch) -> None:
    grid = SpectrumGrid(5, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    options = {"correlation_points": 12, "band_points": 3, "isrs_segments": 4}
    model = cast("CGNModel", create_model("cgn", grid, fiber, 32e9, **options))
    power = np.array([1e-3, 0, 0.8e-3, 0, 1.2e-3], dtype=np.float64)
    kernel = cast("CGNSpanKernel", model.kernel_for(fiber))
    sparse = kernel.evaluate(power, 20e3)
    kernel.evaluate(2 * power, 20e3)
    repeated = kernel.evaluate(power, 20e3)
    for field in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        np.testing.assert_array_equal(getattr(sparse, field), getattr(repeated, field))
    with monkeypatch.context() as patch:
        patch.setattr(CGNSpanKernel, "_subset_view", lambda self, _act: self._full_view)
        dense_model = cast(
            "CGNModel", create_model("cgn", grid, fiber, 32e9, **options)
        )
        dense = dense_model.kernel_for(fiber).evaluate(power, 20e3)
    for field in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        np.testing.assert_allclose(
            getattr(sparse, field), getattr(dense, field), rtol=1e-12, atol=0
        )
    # Uniform amplification without ISRS gives identical launch powers per span.
    coherent = cast(
        "CGNModel",
        create_model(
            "cgn", grid, fiber, 32e9, correlation_points=12, include_isrs=False
        ),
    )
    coherent_power = np.full(5, 1e-3)
    path = MultiSpanOpticalPath([fiber] * 2, [EDFAConfig()] * 2)
    result = coherent.evaluate_path(path, coherent_power)
    coherent_kernel = cast("CGNSpanKernel", coherent.kernel_for(fiber))
    link_sci, link_xci = coherent_kernel.link_nli(coherent_power, 20e3, 2)
    np.testing.assert_allclose(result.spm_noise_w, link_sci, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.xpm_noise_w, link_xci, rtol=1e-9, atol=0)
    with pytest.raises(ValueError, match="egn"):
        create_model("cgn", grid, fiber, egn=True)


@pytest.mark.parametrize("integrator", [_island_core, _mci_core])
def test_isrs_integrators_require_segment_positions(integrator) -> None:
    """ISRS profiles need segment positions to form their spatial integral."""
    common = {
        "f_abs_hz": np.array([193.1e12]),
        "b_hz": np.array([32e9]),
        "alpha_npm": np.array([4e-5]),
        "gamma_w_m": np.array([1e-3]),
        "beta2_s2_m": -2e-26,
        "beta3_s3_m": 0.0,
        "dispersion_convention": "repo",
        "nsp_convention": "gn",
        "length_m": 80e3,
        "correlation_points": 4,
        "island_cells": 1,
        "n_coherent_spans": 1,
        "isrs_gain": np.ones((1, 1)),
        "isrs_rate": np.ones(1),
    }
    if integrator is _island_core:
        common["r_index"] = np.zeros((1, 1, 1), dtype=np.int64)
    else:
        common["mci_index"] = np.zeros((1, 1, 3), dtype=np.int64)
    with pytest.raises(ValueError, match="segment positions"):
        integrator(**common)


def test_dispersion_matches_propagation_constant_polynomial() -> None:
    reference = 193.4e12
    observation = 205e12
    x, y = np.array([-34e9, 27e9]), np.array([400e9, -800e9])
    beta2, beta3 = -2.1e-26, 1.2e-40

    def beta(frequency: NDArrayFloat | float) -> NDArrayFloat:
        omega = 2 * np.pi * (np.asarray(frequency) - reference)
        return beta2 * omega**2 / 2 + beta3 * omega**3 / 6

    mismatch = (
        beta(observation + x + y)
        + beta(observation)
        - beta(observation + x)
        - beta(observation + y)
    )
    calculated = (
        4
        * np.pi**2
        * x
        * y
        * _effective_beta2(
            beta2, beta3, x, y, np.array(observation), "paper", reference
        )
    )
    np.testing.assert_allclose(calculated, mismatch, rtol=1e-10)


@pytest.mark.parametrize("integrator", [_island_core, _mci_core])
def test_segmented_isrs_matches_exact_exponential_stimulus(integrator) -> None:
    frequencies = 193.1e12 + np.arange(3) * 40e9
    bandwidth = np.full(3, 32e9)
    alpha = np.full(3, 4e-5)
    rate = np.array([-1e-5, 1.4e-5, 2e-5])
    positions = (np.arange(8) + 0.5) * 10e3
    settings = {
        "f_abs_hz": frequencies,
        "b_hz": bandwidth,
        "gamma_w_m": np.full(3, 1.3e-3),
        "beta2_s2_m": -1e-27,
        "beta3_s3_m": 0.0,
        "dispersion_convention": "repo",
        "nsp_convention": "gn",
        "length_m": 80e3,
        "correlation_points": 32,
        "island_cells": 8,
        "n_coherent_spans": 1,
    }
    if integrator is _island_core:
        settings["r_index"] = _third_wave_slots(frequencies, bandwidth)[0]
    else:
        settings["mci_index"] = _mci_slot_table(frequencies, bandwidth, alpha, 0)[0]
    exact, _ = integrator(alpha_npm=alpha + rate, **settings)
    segmented, _ = integrator(
        alpha_npm=alpha,
        isrs_gain=np.exp(-positions[:, None] * rate),
        isrs_rate=rate,
        isrs_z_mid=positions,
        isrs_seg_length_m=10e3,
        **settings,
    )
    np.testing.assert_allclose(segmented, exact, rtol=1e-11, atol=0)


def test_zero_dispersion_includes_all_mci_islands() -> None:
    frequencies = 193.1e12 + np.arange(3) * 10e9
    bandwidth = np.full(3, 10e9)
    power = np.array([1e-3, 1.5e-3, 0.8e-3])
    fiber = FiberSpanConfig(
        length_km=10,
        dispersion_parameter_d=0,
        dispersion_slope_s=0,
        use_attenuation_polynomial=False,
    )
    kernel = CGNSpanKernel(
        fiber_config=fiber,
        f_abs_hz=frequencies,
        b_hz=bandwidth,
        include_isrs=False,
        correlation_points=64,
    )
    result = kernel.evaluate(power, 10e3)
    expected = np.zeros(3)
    psd = power / bandwidth
    effective_length = -np.expm1(-kernel.alpha_npm[0] * 10e3) / kernel.alpha_npm[0]
    for cut in range(3):
        for first in range(3):
            for second in range(3):
                lower = (
                    frequencies[first] + frequencies[second] - frequencies[cut] - 10e9
                )

                def area_below(edge: float, lower_bound: float = lower) -> float:
                    shift = edge - lower_bound
                    return 0.5 * (
                        max(shift, 0) ** 2
                        - 2 * max(shift - 10e9, 0) ** 2
                        + max(shift - 20e9, 0) ** 2
                    )

                for third in range(3):
                    area = area_below(frequencies[third] + 5e9) - area_below(
                        frequencies[third] - 5e9
                    )
                    expected[cut] += psd[first] * psd[second] * psd[third] * area
    expected *= (16 / 27) * kernel.gamma_w_m**2 * effective_length**2 * bandwidth
    np.testing.assert_allclose(
        result.spm_noise_w + result.xpm_noise_w, expected, rtol=0.005
    )
    expected_sci = (
        (16 / 27)
        * kernel.gamma_w_m**2
        * effective_length**2
        * psd**3
        * 0.75
        * bandwidth**3
    )
    np.testing.assert_allclose(result.spm_noise_w, expected_sci, rtol=0.005)
    assert np.any(kernel._mci_table()[1] > 0)


def test_heterogeneous_spans_match_lossless_coherent_field_sum() -> None:
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    first = FiberSpanConfig(
        length_km=2,
        dispersion_parameter_d=0,
        dispersion_slope_s=0,
        use_attenuation_polynomial=True,
        attenuation_alpha0=0,
        attenuation_alpha1=0,
        attenuation_alpha2=0,
    )
    second = first.model_copy(update={"length_km": 3, "nonlinear_index_n2": 3.1e-20})
    model = cast("CGNModel", create_model("cgn", grid, first, 32e9, include_isrs=False))
    result = model.evaluate_path(
        MultiSpanOpticalPath([first, second], [EDFAConfig()] * 2), np.array([1e-3])
    )
    k1 = cast("CGNSpanKernel", model.kernel_for(first))
    k2 = cast("CGNSpanKernel", model.kernel_for(second))
    amplitude = k1.gamma_w_m[0] * 2e3 + k2.gamma_w_m[0] * 3e3
    expected = (
        (16 / 27) * (1e-3 / 32e9) ** 3 * (0.75 * (32e9) ** 2) * 32e9 * amplitude**2
    )
    np.testing.assert_allclose(result.spm_noise_w, [expected], rtol=1e-5)
    np.testing.assert_array_equal(result.xpm_noise_w, [0.0])


def test_heterogeneous_opposite_dispersion_preserves_cross_span_phase() -> None:
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    first = FiberSpanConfig(
        length_km=20,
        use_attenuation_polynomial=True,
        attenuation_alpha0=0,
        attenuation_alpha1=0,
        attenuation_alpha2=0,
    )
    second = first.model_copy(
        update={
            "dispersion_parameter_d": -first.dispersion_parameter_d,
            "dispersion_slope_s": -first.dispersion_slope_s,
        }
    )
    model = cast(
        "CGNModel",
        create_model(
            "cgn", grid, first, 32e9, include_isrs=False, correlation_points=96
        ),
    )
    power = np.array([1e-3])
    reference = model.evaluate_path(
        MultiSpanOpticalPath([first], [EDFAConfig()]), power
    )
    result = model.evaluate_path(
        MultiSpanOpticalPath([first, second], [EDFAConfig()] * 2), power
    )
    # Opposite dispersion reverses the phase evolution in the second span:
    # exp(i*omega*L) * integral(exp(-i*omega*z)) equals the first span's field.
    np.testing.assert_allclose(
        result.spm_noise_w, 4 * reference.spm_noise_w, rtol=0.005
    )


def test_unamplified_segmented_fiber_matches_one_continuous_fiber() -> None:
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    first = FiberSpanConfig(length_km=5)
    second = first.model_copy(update={"length_km": 7})
    continuous = first.model_copy(update={"length_km": 12})
    model = cast(
        "CGNModel",
        create_model(
            "cgn", grid, first, 32e9, include_isrs=False, correlation_points=96
        ),
    )
    amplifier = EDFAConfig(target_gain_db=0)
    power = np.array([1e-3])
    result = model.evaluate_path(
        MultiSpanOpticalPath([first, second], [amplifier] * 2), power
    )
    reference = model.evaluate_path(
        MultiSpanOpticalPath([continuous], [amplifier]), power
    )
    np.testing.assert_allclose(
        result.signal_power_w, reference.signal_power_w, rtol=1e-14
    )
    np.testing.assert_allclose(result.spm_noise_w, reference.spm_noise_w, rtol=0.005)
