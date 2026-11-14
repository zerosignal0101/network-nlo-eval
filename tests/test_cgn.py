"""Correlation GN physical integrals, sparse grids and coherent accumulation."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.cgn import (
    CGNSpanKernel,
    _isrs_nsp_energy_spectrum,
    _nsp_energy_spectrum,
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
                if second == cut:
                    sci[cut] += contribution
                else:
                    xci[cut] += 2 * contribution
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
    model = create_model("cgn", grid, fiber, 32e9, **options)
    power = np.array([1e-3, 0, 0.8e-3, 0, 1.2e-3])
    kernel = model.kernel_for(fiber)
    sparse = kernel.evaluate(power, 20e3)
    kernel.evaluate(2 * power, 20e3)
    repeated = kernel.evaluate(power, 20e3)
    for field in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        np.testing.assert_array_equal(getattr(sparse, field), getattr(repeated, field))
    with monkeypatch.context() as patch:
        patch.setattr(CGNSpanKernel, "_subset_view", lambda self, _act: self._full_view)
        dense = (
            create_model("cgn", grid, fiber, 32e9, **options)
            .kernel_for(fiber)
            .evaluate(power, 20e3)
        )
    for field in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        np.testing.assert_allclose(
            getattr(sparse, field), getattr(dense, field), rtol=1e-12, atol=0
        )
    # Uniform amplification without ISRS gives identical launch powers per span.
    coherent = create_model(
        "cgn", grid, fiber, 32e9, correlation_points=12, include_isrs=False
    )
    coherent_power = np.full(5, 1e-3)
    path = MultiSpanOpticalPath([fiber] * 2, [EDFAConfig()] * 2)
    result = coherent.evaluate_path(path, coherent_power)
    link_sci, link_xci = coherent.kernel_for(fiber).link_nli(coherent_power, 20e3, 2)
    np.testing.assert_allclose(result.spm_noise_w, link_sci, rtol=1e-9, atol=0)
    np.testing.assert_allclose(result.xpm_noise_w, link_xci, rtol=1e-9, atol=0)
    with pytest.raises(ValueError, match="egn"):
        create_model("cgn", grid, fiber, egn=True)
