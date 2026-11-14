"""Registered Buglia Raman model across span and route layers."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
)


def test_registered_model_routes_pumped_span_in_both_nli_modes() -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.02, direction="backward")
    power = np.array([1e-3, 1e-3])
    models = [
        create_model(
            "buglia2024_ra_gn",
            grid,
            fiber,
            32e9,
            raman_pumps=[pump],
            n_steps=8,
            n_quadrature=8,
            nli_form=form,
        )
        for form in ("closed", "integral")
    ]
    results = [model.evaluate_path(path, power) for model in models]
    np.testing.assert_array_equal(results[0].signal_power_w, results[1].signal_power_w)
    np.testing.assert_array_equal(results[0].ase_noise_w, results[1].ase_noise_w)
    for result in results:
        assert np.all(np.isfinite(result.snr_db))
        assert np.all(result.spm_noise_w > 0)
        assert np.all(result.xpm_noise_w > 0)
    assert np.all(models[0].kernel_for(fiber).last_raman_ase_w > 0)
    with pytest.raises(ValueError, match="nli_form"):
        create_model("buglia2024_ra_gn", grid, fiber, nli_form="unknown")
