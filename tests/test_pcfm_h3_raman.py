"""Profile-aware Raman model and network validity integration."""

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.pcfm_h3_raman import PCFH3RamanModel
from network_nlo_eval.models.raman import C_LIGHT
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
    ROADMConfig,
)
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.rwa.validity import H3LinkValidityProvider, LinkValidityPolicy
from network_nlo_eval.simulation.traffic import ServiceRequest


def test_unpumped_h3_is_bit_identical_to_pcfm() -> None:
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    power = np.array([1e-3, 2e-3, 0.5e-3])
    options = {
        "poly_degree": 3,
        "egn": True,
        "edge_fix": True,
        "cache_spans": True,
        "spp_steps": "auto",
    }
    baseline = create_model("pcfm", grid, fiber, 32e9, **options).evaluate_path(
        path, power
    )
    h3 = create_model("pcfm_h3_raman", grid, fiber, 32e9)
    assert isinstance(h3, PCFH3RamanModel)
    result, budget = h3.evaluate_path_with_uncertainty(path, power)
    for name in (
        "signal_power_w",
        "spm_noise_w",
        "xpm_noise_w",
        "ase_noise_w",
        "snr_db",
    ):
        np.testing.assert_array_equal(getattr(result, name), getattr(baseline, name))
    assert np.all(budget.validity.valid)


def test_pumped_h3_power_coordinates_and_loss() -> None:
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.02, direction="backward")
    options = {"raman_pumps": [pump], "raman_steps": 16, "lumped_loss": [(10e3, 1.0)]}
    total = create_model("pcfm_h3_raman", grid, fiber, 32e9, **options)
    assert isinstance(total, PCFH3RamanModel)
    per_pol = create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        32e9,
        launch_power_convention="per_pol",
        signal_polarizations=2,
        **options,
    )
    power = np.array([1e-3, 2e-3, 0.5e-3])
    a = total.evaluate_path(path, power)
    b = per_pol.evaluate_path(path, power / 2)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"):
        np.testing.assert_allclose(
            getattr(a, name), 2 * getattr(b, name), rtol=1e-10, atol=0
        )
    np.testing.assert_allclose(a.snr_db, b.snr_db, rtol=1e-10)
    assert total.last_span_validity[0].lumped_loss_in_ode
    assert total.perf_ode_solves > 0


def test_network_validity_policy_rejects_pump_signal_overlap() -> None:
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    topology = NetworkTopology(nx.Graph([(0, 1)]), fiber, EDFAConfig(), ROADMConfig())
    state = NetworkState(topology, grid)
    pump = RamanPumpConfig(
        wavelength_nm=C_LIGHT / 193.1e12 * 1e9, power_w=0.001, direction="backward"
    )
    model = create_model(
        "pcfm_h3_raman", grid, fiber, 32e9, raman_pumps=[pump], raman_steps=8
    )
    assert isinstance(model, PCFH3RamanModel)
    request = ServiceRequest(0, 0, 1, 0, 1, 100, -200, 0.001)
    validator = QoTValidator(
        model,
        topology,
        grid,
        validity_policy=LinkValidityPolicy(mode="reject"),
        validity_provider=H3LinkValidityProvider(model),
    )
    with pytest.warns(RuntimeWarning, match="falls inside the occupied signal"):
        accepted, _ = validator.verify_allocation(request, [0, 1], 0, state)
    assert not accepted
    assert not validator.last_link_validity[(0, 1)].valid[0]
    assert state.get_link_state(0, 1).noise_cache is None
