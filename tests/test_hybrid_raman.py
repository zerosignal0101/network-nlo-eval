"""Hybrid Raman NLI modes and power convention integration."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
)


@pytest.mark.parametrize("form", ["paper", "integral", "pcfm"])
def test_hybrid_model_modes_and_power_conventions(form: str) -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.02, direction="backward")
    options = {"raman_pumps": [pump], "spp_steps": 8, "nli_form": form}
    total = create_model("hybrid_raman_paper", grid, fiber, 32e9, **options)
    per_pol = create_model(
        "hybrid_raman_paper",
        grid,
        fiber,
        32e9,
        launch_power_convention="per_pol",
        signal_polarizations=2,
        **options,
    )
    power = np.array([1e-3, 1e-3])
    a = total.evaluate_path(path, power)
    b = per_pol.evaluate_path(path, power / 2)
    assert np.all(np.isfinite(a.snr_db))
    assert np.all(a.spm_noise_w > 0)
    assert np.all(a.xpm_noise_w > 0)
    for name in ("signal_power_w", "ase_noise_w", "spm_noise_w", "xpm_noise_w"):
        np.testing.assert_allclose(getattr(a, name), 2 * getattr(b, name), rtol=1e-10)
    np.testing.assert_allclose(a.snr_db, b.snr_db, rtol=1e-10)


def test_legacy_no_pump_matches_pcfm_and_rejects_bad_mode() -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    power = np.full(2, 1e-3)
    legacy = create_model(
        "hybrid_raman_paper", grid, fiber, 32e9, nli_form="pcfm", spp_steps=8
    )
    baseline = create_model("pcfm", grid, fiber, 32e9, spp_steps=8)
    a, b = legacy.evaluate_path(path, power), baseline.evaluate_path(path, power)
    for name in (
        "signal_power_w",
        "ase_noise_w",
        "spm_noise_w",
        "xpm_noise_w",
        "snr_db",
    ):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    with pytest.raises(ValueError, match="nli_form"):
        create_model("hybrid_raman_paper", grid, fiber, nli_form="unknown")
