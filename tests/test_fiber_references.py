"""Physical reference-frequency and per-route fiber consistency checks."""

import numpy as np
import pytest

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.numba_kernels import _dispersion_at_frequency
from network_nlo_eval.network.elements import FiberSpanConfig


@pytest.mark.parametrize("model_name", ["ecz", "semrau2019", "buglia2023"])
def test_equivalent_dispersion_references_preserve_nli(model_name: str) -> None:
    fiber = FiberSpanConfig(length_km=80, aeff_slope_um2_nm=0)
    new_wavelength_nm = 1540.0
    new_wavelength_m = new_wavelength_nm * 1e-9
    beta2, beta3 = _dispersion_at_frequency(
        fiber.dispersion_parameter_d,
        fiber.dispersion_slope_s,
        fiber.reference_wavelength_nm,
        C_LIGHT / new_wavelength_m,
    )
    new_d = -beta2 * 2 * np.pi * C_LIGHT / new_wavelength_m**2 * 1e6
    new_s = (
        beta3 * (2 * np.pi * C_LIGHT) ** 2 / new_wavelength_m**2
        - 2 * new_wavelength_m * new_d * 1e-6
    ) / (new_wavelength_m**2 * 1e3)
    equivalent = fiber.model_copy(
        update={
            "reference_wavelength_nm": new_wavelength_nm,
            "dispersion_parameter_d": new_d,
            "dispersion_slope_s": new_s,
        }
    )
    grid = SpectrumGrid(3, 50e9, 205e12)
    power = np.array([1e-3, 0.8e-3, 1.5e-3])
    first = create_model(model_name, grid, fiber, 32e9).span_step(
        fiber, None, power, 80e3
    )
    second = create_model(model_name, grid, equivalent, 32e9).span_step(
        equivalent, None, power, 80e3
    )
    np.testing.assert_allclose(first.spm_noise_w, second.spm_noise_w, rtol=1e-12)
    np.testing.assert_allclose(first.xpm_noise_w, second.xpm_noise_w, rtol=1e-12)


def test_buglia_uses_actual_route_fiber_properties() -> None:
    reference = FiberSpanConfig(length_km=80)
    actual = reference.model_copy(
        update={
            "dispersion_parameter_d": 4.0,
            "dispersion_slope_s": 0.02,
            "effective_area_um2_ref": 110.0,
            "nonlinear_index_n2": 3.1e-20,
        }
    )
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    power = np.array([1e-3, 0.8e-3, 1.5e-3])
    routed = create_model("buglia2023", grid, reference, 32e9).span_step(
        actual, None, power, 80e3
    )
    direct = create_model("buglia2023", grid, actual, 32e9).span_step(
        actual, None, power, 80e3
    )
    np.testing.assert_array_equal(routed.spm_noise_w, direct.spm_noise_w)
    np.testing.assert_array_equal(routed.xpm_noise_w, direct.xpm_noise_w)


@pytest.mark.parametrize(
    "model_name",
    [
        "ecz",
        "ecz_coherent",
        "semrau2019",
        "buglia2023",
        "damico2022",
        "carena2014",
        "numerical",
        "cgn",
        "hybrid_raman_paper",
        "buglia2024_ra_gn",
        "zhang2024_dra_egn",
        "numerical_dra",
    ],
)
@pytest.mark.parametrize("in_reference", [True, False])
def test_unmodeled_in_span_losses_are_rejected(
    model_name: str, in_reference: bool
) -> None:
    fiber = FiberSpanConfig(length_km=80)
    lossy = fiber.model_copy(update={"lumped_loss": [(40e3, 1.0)]})
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    if in_reference:
        with pytest.raises(ValueError, match="in-span lumped losses"):
            create_model(model_name, grid, lossy, 32e9)
    else:
        model = create_model(model_name, grid, fiber, 32e9)
        with pytest.raises(ValueError, match="in-span lumped losses"):
            model.span_step(lossy, None, np.full(2, 1e-3), 80e3)
