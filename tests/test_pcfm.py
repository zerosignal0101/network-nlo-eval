"""PCFM model regression, network adapter, and applicability checks."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    available_models,
    create_model,
)
from network_nlo_eval.models.numerical import NumericalISRSGN
from network_nlo_eval.models.pcfm import PCFMModel
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
)
from network_nlo_eval.rwa.link_noise import assemble_link_noise


def test_multispan_reference_and_network_adapter() -> None:
    fiber = FiberSpanConfig(length_km=160, max_span_length_km=80)
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    power = np.array([1e-3, 2e-3, 0.5e-3])
    edfa = EDFAConfig(noise_figure_db=5.5)
    model = create_model("pcfm", grid, fiber, 32e9)
    result = model.evaluate_path(MultiSpanOpticalPath([fiber], [edfa]), power)
    expected = {
        "signal_power_w": [0.001, 0.002, 0.0005],
        "spm_noise_w": [
            6.518587366940741e-07,
            5.224527568348149e-06,
            8.178448943247477e-08,
        ],
        "xpm_noise_w": [
            8.042925336404108e-07,
            4.879689622958184e-07,
            4.4002700443481964e-07,
        ],
        "ase_noise_w": [
            1.1272369695565567e-06,
            1.1276488923500325e-06,
            1.1280608542122585e-06,
        ],
        "snr_db": [25.878103217507867, 24.659646606496437, 24.81519660527725],
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(getattr(result, name), reference, rtol=1e-11)
    assert isinstance(model, PCFMModel)
    assert "pcfm" in available_models()
    view = model.for_fiber(fiber)
    assert view is model.for_fiber(fiber)
    cache = assemble_link_noise(
        evaluator=view,
        edfa_config=edfa,
        power_profile=power,
        span_length_m=80e3,
        num_full_spans=2,
        remainder_length_m=0,
        num_channels=3,
        coherent=True,
    )
    for name in ("spm", "xpm", "ase"):
        np.testing.assert_allclose(
            getattr(cache, name), getattr(result, f"{name}_noise_w"), rtol=1e-11
        )


def test_lumped_raman_requires_distributed_raman_model_for_active_pumps() -> None:
    fiber = FiberSpanConfig(length_km=160, max_span_length_km=80)
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    pumps = [RamanPumpConfig(wavelength_nm=1450, power_w=0.2)]
    with pytest.raises(ValueError, match="Active Raman pumps require"):
        PCFMModel(grid, fiber, 32e9, raman_model="lumped", raman_pumps=pumps)


def test_dense_plan_falls_back_to_numerical_model() -> None:
    fiber = FiberSpanConfig(length_km=80)
    model = PCFMModel(
        SpectrumGrid(3, 20e9, 193.1e12), fiber, 32e9, nli_fallback="numerical"
    )
    result = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [EDFAConfig()]), np.full(3, 1e-3)
    )
    assert np.all(np.isfinite(result.snr_db))
    assert model.last_nli_fallback_applied
    reference = NumericalISRSGN(model.grid, fiber, 32e9).evaluate_path(
        MultiSpanOpticalPath([fiber], [EDFAConfig()]), np.full(3, 1e-3)
    )
    np.testing.assert_allclose(result.spm_noise_w, reference.spm_noise_w, rtol=1e-12)
    np.testing.assert_allclose(result.xpm_noise_w, reference.xpm_noise_w, rtol=1e-12)
