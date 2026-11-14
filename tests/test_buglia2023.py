"""Buglia finite-length approximation and coherent route behavior."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.buglia2023 import (
    _coherent_factor,
    _equivalent_phase_ramp,
    _gn_weighted_spm_visibility,
    alpha_tilde_kappa,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


def test_finite_length_limits() -> None:
    length = 80e3
    alpha, kappa = alpha_tilde_kappa(1e-12, length)
    assert alpha == pytest.approx(2 / length, rel=1e-6)
    assert kappa == pytest.approx(2, rel=1e-6)
    alpha, kappa = alpha_tilde_kappa(1e-3, length)
    assert alpha == pytest.approx(1e-3)
    assert kappa == pytest.approx(1)


def test_coherence_factor_and_zero_dispersion() -> None:
    vis = _gn_weighted_spm_visibility(
        beta2_abs_s2_m=0,
        alpha_npm_m=4e-5,
        bandwidth_hz=32e9,
        span_length_m=80e3,
        span_separations_m=np.array([80e3, 160e3]),
    )
    np.testing.assert_array_equal(vis, [1, 1])
    factors = _coherent_factor(3, np.array([[0.4, 1.0], [0.1, 1.0]]))
    ramp = _equivalent_phase_ramp(3, factors)
    amplitudes = np.exp(1j * np.arange(3)[:, None] * ramp)
    np.testing.assert_allclose(np.abs(amplitudes.sum(axis=0)) ** 2, factors)


def test_registered_model_on_multispan_path() -> None:
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    path = MultiSpanOpticalPath([fiber] * 2, [EDFAConfig()] * 2)
    model = create_model("buglia2023", grid, fiber, 32e9)
    result = model.evaluate_path(path, np.full(3, 1e-3))
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(result.spm_noise_w > 0)
    assert np.all(result.xpm_noise_w > 0)
    with pytest.raises(ValueError, match="GN model only"):
        create_model("buglia2023", grid, fiber, egn=True)
