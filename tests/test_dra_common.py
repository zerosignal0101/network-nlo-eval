"""Distributed Raman profile fitting and shared span calculations."""

import numpy as np
import pytest
from scipy.integrate import quad

from network_nlo_eval.models.dra_common import (
    BugliaProfile,
    DRASpanKernelBase,
    ZhangProfile,
    average_link_squared,
    fit_buglia_profile,
    fit_zhang_profile,
)
from network_nlo_eval.models.raman import RamanGainModel
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig


def test_profile_fits_and_link_integral() -> None:
    length = 80e3
    n_steps = 8
    frequencies = np.array([194e12, 193e12])
    alpha = np.full(2, 4.6e-5)
    z = np.arange(1, 2 * n_steps + 1) * length / (2 * n_steps)
    profile = np.exp(-z[:, None] * alpha)
    fits: list[BugliaProfile | ZhangProfile] = [
        fit_buglia_profile(
            profile, length, frequencies, alpha, 200e12, 5e-5, 5e-5, n_steps
        ),
        fit_zhang_profile(profile, length, frequencies, alpha, n_steps),
    ]
    nodes, weights = np.polynomial.legendre.leggauss(16)  # type: ignore[no-untyped-call]
    for fit in fits:
        np.testing.assert_allclose(
            fit.value(z, frequencies[0]), profile[:, 0], rtol=3e-3
        )
        phase = 2e-5
        real = quad(
            lambda x, fit=fit, phase=phase: float(
                fit.value(np.array([x]), frequencies[0])[0] * np.cos(phase * x)
            ),
            0,
            length,
        )[0]
        imaginary = quad(
            lambda x, fit=fit, phase=phase: float(
                fit.value(np.array([x]), frequencies[0])[0] * np.sin(phase * x)
            ),
            0,
            length,
        )[0]
        assert fit.link(frequencies[0], np.array([phase]))[0] == pytest.approx(
            real + 1j * imaginary, rel=1e-10
        )
        zero = fit.link(frequencies[0], np.zeros(1))[0]
        assert average_link_squared(
            fit, frequencies[0], 0, 32e9, alpha[0], nodes, weights
        ) == pytest.approx(abs(zero) ** 2)


class BugliaKernel(DRASpanKernelBase[BugliaProfile]):
    def _fit_profile(self, profile_norm, span_length_m):
        return fit_buglia_profile(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.alpha_npm,
            200e12,
            5e-5,
            5e-5,
            self._n_steps,
        )


def test_shared_kernel_runs_raman_ode_and_noise() -> None:
    frequencies = np.array([193e12, 194e12])
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.02, direction="backward")
    kernel = BugliaKernel(
        frequencies,
        np.full(2, 32e9),
        FiberSpanConfig(length_km=20),
        pumps=[pump],
        gain_model=RamanGainModel(),
        n_steps=8,
        n_quadrature=8,
    )
    output = kernel.evaluate(np.array([1e-3, 2e-3]), 20e3)
    assert np.all(output.power_out_w > 0)
    assert np.all(output.spm_noise_w > 0)
    assert np.all(output.xpm_noise_w > 0)
    assert np.all(kernel.last_raman_ase_w > 0)
    assert kernel.last_pump_power_out_w is not None
    assert kernel.last_pump_power_out_w[0] > 0
