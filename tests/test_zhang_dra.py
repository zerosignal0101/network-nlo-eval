"""Zhang Raman route and bundled calibration integration."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.zhang2024_dra import default_calibration, load_calibration
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
)


def test_zhang_model_uses_coherent_route_and_calibration(tmp_path) -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    path = MultiSpanOpticalPath([fiber, fiber], [EDFAConfig(), EDFAConfig()])
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.02, direction="backward")
    power = np.full(2, 1e-3)
    options = {
        "raman_pumps": [pump],
        "n_steps": 8,
        "n_quadrature": 8,
        "n_spm_quadrature": 8,
    }
    calibrated = create_model("zhang2024_dra_egn", grid, fiber, 32e9, **options)
    uncalibrated = create_model(
        "zhang2024_dra_egn", grid, fiber, 32e9, use_calibration=False, **options
    )
    a = calibrated.evaluate_path(path, power)
    b = uncalibrated.evaluate_path(path, power)
    assert a.span_count == 2
    assert np.all(np.isfinite(a.snr_db))
    assert np.all(a.spm_noise_w > 0)
    assert np.all(a.xpm_noise_w > 0)
    calibration = default_calibration()
    params = {"launch_power_dbm": 0.0, "spans": 2.0}
    np.testing.assert_allclose(
        a.spm_noise_w,
        b.spm_noise_w * calibration["spm"].apply(params),
        rtol=1e-12,
        atol=0,
    )
    np.testing.assert_allclose(
        a.xpm_noise_w,
        b.xpm_noise_w * calibration["xpm"].apply(params),
        rtol=1e-12,
        atol=0,
    )
    assert load_calibration(tmp_path / "missing.json") == {}


def test_zhang_rejects_forward_pumps_and_unpublished_correction() -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.02, direction="forward")
    with pytest.raises(ValueError, match="backward"):
        create_model("zhang2024_dra_egn", grid, fiber, raman_pumps=[pump])
    with pytest.raises(ValueError, match="eta_XPM_corr"):
        create_model(
            "zhang2024_dra_egn", grid, fiber, modulation_format_correction=True
        )
