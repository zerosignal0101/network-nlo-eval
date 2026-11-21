"""Transceiver mode geometry and receiver feasibility through admission."""

import math
from dataclasses import replace

import networkx as nx
import numpy as np
import pytest
from pydantic import ValidationError

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.transceiver import (
    ReceiverPenaltyCurve,
    TransceiverMode,
    evaluate_receiver,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.route import H_PLANCK
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.allocators import (
    KSPFirstFitAllocator,
    KSPFirstFitFlexAllocator,
)
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


def test_receiver_bandwidth_and_penalty_budget() -> None:
    mode = TransceiverMode(
        name="40 GBd",
        symbol_rate_hz=40e9,
        min_gsnr_db=16,
        min_osnr_db=26,
        tx_osnr_db=30,
        osnr_reference_bandwidth_hz=10e9,
        rx_power_min_dbm=-2,
        rx_power_max_dbm=3,
        cd_penalty=ReceiverPenaltyCurve(points=((0, 0), (2000, 2))),
    )
    report = evaluate_receiver(
        mode,
        signal_power_w=1e-3,
        ase_noise_w=4e-6,
        noise_power_w=10e-6,
        noise_bandwidth_hz=40e9,
        line_snr_db=19.5,
        snr_requirement_db=18,
        cd_ps_nm=-1000,
        pmd_ps=0,
        pdl_db=0,
        validity_adjustment_db=-0.5,
    )
    assert report.osnr_db == pytest.approx(10 * math.log10(1e-3 / 2e-6))
    assert report.gsnr_db == pytest.approx(10 * math.log10(1e-3 / 14e-6))
    assert report.effective_gsnr_db == pytest.approx(report.gsnr_db - 1.5)
    assert report.rx_power_dbm == pytest.approx(0)
    assert report.rx_power_margin_db == pytest.approx(2)
    assert report.margin_db == pytest.approx(report.osnr_db - 26)
    assert report.feasible

    with pytest.raises(ValidationError, match="strictly increasing"):
        ReceiverPenaltyCurve(points=((10, 0), (10, 1)))
    assert mode.cd_penalty is not None
    assert mode.cd_penalty.penalty_db(2000) == 2
    assert mode.cd_penalty.penalty_db(2001) == math.inf


@pytest.mark.parametrize("convention", ["total", "per_pol"])
def test_uniform_receiver_budget_includes_add_ase_drop_loss_and_impairments(
    convention: str,
) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        spans=[
            {
                "fiber": {
                    "length_km": 40,
                    "dispersion_parameter_d": 17,
                    "pmd_coefficient_ps_sqrt_km": 0.1,
                },
                "edfa": {"pmd_ps": 0.2, "pdl_db": 0.3},
            },
            {
                "fiber": {
                    "length_km": 15,
                    "dispersion_parameter_d": -25,
                    "pmd_coefficient_ps_sqrt_km": 0.2,
                },
                "edfa": {},
            },
        ],
    )
    graph.nodes[0]["roadm"] = {
        "insertion_loss_db": 3,
        "filtering_penalty_db": 1,
        "pmd_ps": 0.25,
    }
    graph.nodes[1]["roadm"] = {
        "insertion_loss_db": 3,
        "filtering_penalty_db": 2,
        "pdl_db": 0.4,
        "equalize_output_power": False,
    }
    fiber = FiberSpanConfig(length_km=40)
    topology = NetworkTopology(
        graph, fiber, EDFAConfig(), ROADMConfig(add_drop_loss_db=0)
    )
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    model = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention=convention)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid)
    mode = TransceiverMode(
        name="32 GBd",
        symbol_rate_hz=32e9,
        min_gsnr_db=0,
        rx_power_min_dbm=-4,
        pmd_penalty=ReceiverPenaltyCurve(points=((0, 0), (2, 2))),
        pdl_penalty=ReceiverPenaltyCurve(points=((0, 0), (1, 1))),
    )
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 1e-3, transceiver_mode=mode)
    accepted, snrs = validator.verify_allocation(request, [0, 1], 0, state)
    assert accepted
    assert snrs is not None
    report = validator.last_receiver_feasibility[1]

    scale = 0.5 if convention == "per_pol" else 1.0
    direct = model.evaluate_path(
        topology.build_optical_path([0, 1]), np.array([scale * 1e-3])
    )
    signal = direct.signal_power_w[0]
    drop_transfer = signal / (scale * request.launch_power_w)
    add_ase = scale * H_PLANCK * grid.frequencies[0] * 32e9 * 10**0.5 * (10**0.3 - 1)
    ase = direct.ase_noise_w[0] + add_ase * drop_transfer * 10**0.3
    total_noise = ase + direct.spm_noise_w[0] + direct.xpm_noise_w[0]
    assert report.osnr_db == pytest.approx(10 * math.log10(signal / (ase * 12.5 / 32)))
    assert report.gsnr_db == pytest.approx(10 * math.log10(signal / total_noise))
    assert snrs[0] == pytest.approx(report.gsnr_db)
    assert report.rx_power_dbm == pytest.approx(-3)
    assert report.cd_ps_nm == pytest.approx(
        40 * 17 - 15 * 25 + 55 * 0.06 * (1e9 * C_LIGHT / grid.frequencies[0] - 1550)
    )
    assert report.pmd_ps == pytest.approx(
        math.sqrt(40 * 0.1**2 + 15 * 0.2**2 + 0.2**2 + 0.25**2)
    )
    assert report.pdl_db == pytest.approx(0.5)
    assert report.penalty_db == pytest.approx(report.pmd_ps + report.pdl_db)


@pytest.mark.parametrize("mixed", [False, True])
def test_incumbent_receiver_rejects_loaded_route_atomically(mixed: bool) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, edfa={"max_output_signal_power_w": 1.2e-3})
    graph.add_edge(1, 2)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    model = MultiBandISRSGN(grid, fiber, 32e9)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid)
    mode = TransceiverMode(
        name="32 GBd", symbol_rate_hz=32e9, min_gsnr_db=0, rx_power_min_dbm=-1
    )
    incumbent = AllocatedService(
        1, 0, 2, 0, 1, 100, 0, 1e-3, path=[0, 1, 2], transceiver_mode=mode
    )
    accepted, _ = validator.verify_allocation(incumbent, incumbent.path, 0, state)
    assert accepted
    assert validator.last_receiver_feasibility[1].rx_power_dbm == pytest.approx(0)
    state.allocate_service(incumbent)
    assert state.has_per_carrier_services
    old_occupancy = state.occupied_slots_on_path(incumbent.path)

    destination = 1 if mixed else 2
    candidate = ServiceRequest(2, 0, destination, 0, 1, 100, 0, 1e-3)
    path = [0, 1] if mixed else [0, 1, 2]
    assert validator.verify_allocation(candidate, path, 1, state) == (False, None)
    report = validator.last_receiver_feasibility[1]
    assert report.rx_power_dbm < -1
    assert report.rx_power_margin_db < 0
    assert report.gsnr_margin_db > 0
    assert not report.feasible
    np.testing.assert_array_equal(
        old_occupancy, state.occupied_slots_on_path(incumbent.path)
    )
    assert set(state.get_all_allocated_services()) == {1}
    assert all(
        link.noise_cache is None for link in state.get_all_link_states().values()
    )


def test_first_fit_reserves_mode_bandwidth_and_retains_mode() -> None:
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(nx.Graph([(0, 1)]), fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    state = NetworkState(topology, grid, slots_per_channel=4)
    model = MultiBandISRSGN(grid, fiber, 32e9)
    validator = QoTValidator(model, topology, grid)
    cache = PathCache(topology)
    fixed = KSPFirstFitAllocator(cache, validator, grid)
    flex = KSPFirstFitFlexAllocator(cache, validator, grid, topology, 4)
    mode = TransceiverMode(
        name="48 GBd", symbol_rate_hz=48e9, roll_off=0.15, min_gsnr_db=0
    )
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 1e-3, transceiver_mode=mode)
    assert fixed.allocate(request, state) == (False, None)
    assert validator.verify_allocation(
        replace(request, n_slots=4), [0, 1], 0, state, first_slot=0
    ) == (False, None)
    assert not validator.last_receiver_feasibility
    success, allocation = flex.allocate(request, state)
    assert success
    assert allocation is not None
    assert allocation.transceiver_mode is mode
    assert allocation.slot_interval == (0, 5)
    assert request.n_slots == 1
    assert validator.last_receiver_feasibility[1].noise_bandwidth_hz == 48e9
    state.allocate_service(allocation)
    assert state.get_allocated_service(1) is allocation
    assert state.occupied_slots_on_path([0, 1]).sum() == 5

    narrow = mode.model_copy(update={"symbol_rate_hz": 32e9})
    flex.release(allocation, state)
    success, allocation = fixed.allocate(
        replace(request, transceiver_mode=narrow), state
    )
    assert success
    assert allocation is not None
    assert allocation.transceiver_mode is narrow
    state.allocate_service(allocation)
    fixed.release(allocation, state)
    assert not state.has_per_carrier_services

    rejected = replace(
        request, transceiver_mode=mode.model_copy(update={"min_gsnr_db": 100})
    )
    assert flex.allocate(rejected, state) == (False, None)
    assert not validator.last_receiver_feasibility[1].feasible
    assert not state.occupied_slots_on_path([0, 1]).any()
