"""First-fit routing and physical QoT integration."""

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import CarrierGeometry, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    KernelQoTModel,
    MultiSpanOpticalPath,
    QoTModel,
    create_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.allocators import (
    KSPFirstFitAllocator,
    KSPFirstFitFlexAllocator,
    first_fit_contiguous_windows,
)
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.rwa.validity import H3LinkValidityProvider, LinkValidityPolicy
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


def test_window_candidates_are_free_and_deduplicated_by_anchor() -> None:
    assert first_fit_contiguous_windows(
        np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool), 2, 2
    ) == [0, 2, 5, 6]
    assert first_fit_contiguous_windows(np.ones(2, dtype=bool), 3, 2) == []
    with pytest.raises(ValueError, match="n_slots"):
        first_fit_contiguous_windows(np.ones(2, dtype=bool), 0, 2)


@pytest.mark.parametrize("flex", [False, True])
def test_allocator_uses_real_qot_and_releases_resources(flex: bool) -> None:
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(nx.Graph([(0, 1)]), fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    state = NetworkState(topology, grid, slots_per_channel=4 if flex else 1)
    validator = QoTValidator(
        MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off"), topology, grid
    )
    paths = PathCache(topology)
    allocator = (
        KSPFirstFitFlexAllocator(paths, validator, grid, topology, 4)
        if flex
        else KSPFirstFitAllocator(paths, validator, grid)
    )
    rejected = ServiceRequest(0, 0, 1, 0, 1, 100, 1000, 0.001)
    assert allocator.allocate(rejected, state) == (False, None)
    assert state.get_link_state(0, 1).noise_cache is None
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 0.001)
    success, service = allocator.allocate(request, state)
    assert success
    assert service is not None
    assert service.wavelength == 0
    assert (state.get_link_state(0, 1).noise_cache is None) == flex
    state.allocate_service(service)
    assert state.get_link_state(0, 1).occupied_channels[0]
    allocator.release(service, state)
    assert not state.get_link_state(0, 1).occupied_channels.any()


def test_directed_routes_keep_equipment_and_occupancy_independent() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(10, 20, length_km=70, edfa={"noise_figure_db": 4})
    graph.add_edge(20, 10, length_km=90, edfa={"noise_figure_db": 6})
    graph.add_edge(20, 30, length_km=80)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    state = NetworkState(topology, grid)
    validator = QoTValidator(MultiBandISRSGN(grid, fiber, 32e9), topology, grid)
    paths = PathCache(topology)
    allocator = KSPFirstFitAllocator(paths, validator, grid)

    assert paths.get_paths(0, 2) == [[0, 1, 2]]
    assert paths.get_paths(2, 0) == []
    assert topology.get_adjacency_list()[2] == []
    assert topology.build_optical_path([1, 0]).link_keys == [(1, 0)]
    assert topology.get_fiber_config(0, 1).length_km == 70
    assert topology.get_fiber_config(1, 0).length_km == 90
    assert topology.get_link_edfa_config(0, 1).noise_figure_db == 4
    assert topology.get_link_edfa_config(1, 0).noise_figure_db == 6
    with pytest.raises(ValueError, match="non-adjacent"):
        topology.build_optical_path([2, 1])

    forward = ServiceRequest(0, 0, 1, 0, 1, 100, 0, 0.001)
    reverse = ServiceRequest(1, 1, 0, 0, 1, 100, 0, 0.001)
    _, forward_service = allocator.allocate(forward, state)
    assert forward_service is not None
    state.allocate_service(forward_service)
    assert not state.get_link_state(1, 0).occupied_channels[0]
    _, reverse_service = allocator.allocate(reverse, state)
    assert reverse_service is not None
    assert reverse_service.wavelength == forward_service.wavelength == 0
    assert state.get_link_state(1, 0).noise_cache is not None
    state.allocate_service(reverse_service)
    state.release_service(forward_service)
    assert not state.get_link_state(0, 1).occupied_channels[0]
    assert state.get_link_state(1, 0).occupied_channels[0]


def test_flex_first_fit_checks_distinct_centers_within_an_anchor() -> None:
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    flex = FlexibleGrid.from_legacy_grid(2, 50e9, 193.1e12, 4)
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        length_km=80,
        edfa={
            "passbands_hz": (
                flex.connection_edges_hz(1, 1),
                flex.connection_edges_hz(3, 1),
            )
        },
    )
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    state = NetworkState(topology, grid, slots_per_channel=4)
    validator = QoTValidator(MultiBandISRSGN(grid, fiber, 32e9), topology, grid)
    allocator = KSPFirstFitFlexAllocator(
        PathCache(topology), validator, grid, topology, 4
    )
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 0.001, symbol_rate_hz=10e9)

    _, first = allocator.allocate(request, state)
    assert first is not None
    assert first.first_slot == 1
    state.allocate_service(first)
    _, second = allocator.allocate(replace(request, service_id=2), state)
    assert second is not None
    assert second.first_slot == 3
    assert first.wavelength == second.wavelength == 0
    state.allocate_service(second)
    assert allocator.allocate(replace(request, service_id=3), state) == (False, None)
    allocator.release(first, state)
    assert state.get_link_state(0, 1).slot_service_ids[:4].tolist() == [-1, -1, -1, 2]
    assert state.get_link_state(0, 1).allocated_service_ids[0] == 2


@pytest.mark.parametrize("convention", ["total", "per_pol"])
def test_mixed_loading_matches_link_budgets_with_same_slot_replacement(
    convention: str,
) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=80, edfa={"max_output_signal_power_w": 0.0012})
    graph.add_edge(1, 2, length_km=80)
    graph.nodes[1]["roadm"] = {
        "degree_targets": {2: {"power_w": 0.0005}},
        "path_losses": [
            {"from_degree": None, "to_degree": 2, "loss_db": 3},
            {"from_degree": 0, "to_degree": 2, "loss_db": 0},
        ],
    }
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(
        graph,
        fiber,
        EDFAConfig(),
        ROADMConfig(insertion_loss_db=0, add_drop_loss_db=0, filtering_penalty_db=0),
    )
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    flex = FlexibleGrid.from_legacy_grid(2, 50e9, 193.1e12, 4)
    model = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention=convention)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid, slots_per_channel=4)
    departing = AllocatedService(
        1, 0, 1, 0, 1, 100, 0, 0.0006, 1, [0, 1], 0, 0, 4, symbol_rate_hz=10e9
    )
    joining = AllocatedService(
        2, 1, 2, 0, 1, 100, 0, 0.0006, 1, [1, 2], 0, 0, 4, symbol_rate_hz=8e9
    )
    state.allocate_service(departing)
    state.allocate_service(joining)
    candidate = ServiceRequest(3, 0, 2, 0, 1, 100, 0, 0.001, 2, symbol_rate_hz=20e9)

    candidate_geometry = CarrierGeometry(flex, 2, 2, 20e9)
    first = model.for_connections(
        (CarrierGeometry(flex, 0, 1, 10e9), candidate_geometry), 4
    )
    second = model.for_connections(
        (CarrierGeometry(flex, 0, 1, 8e9), candidate_geometry), 4
    )
    scale = 0.5 if convention == "per_pol" else 1.0
    first_result = first.evaluate_path(
        MultiSpanOpticalPath([fiber], [topology.get_link_edfa_config(0, 1)]),
        scale * np.array([departing.launch_power_w, candidate.launch_power_w]),
    )
    after_express = min(0.0005 * scale, first_result.signal_power_w[1])
    after_add = min(0.0005, joining.launch_power_w / 10**0.3) * scale
    second_result = second.evaluate_path(
        MultiSpanOpticalPath([fiber], [topology.get_link_edfa_config(1, 2)]),
        np.array([after_add, after_express]),
    )
    expected = 10 * np.log10(
        1
        / (10 ** (-first_result.snr_db[1] / 10) + 10 ** (-second_result.snr_db[1] / 10))
    )

    accepted, snrs = validator.verify_allocation(
        candidate, [0, 1, 2], 0, state, first_slot=2
    )
    assert accepted
    assert snrs is not None
    assert snrs[0] == pytest.approx(expected, abs=1e-10)

    original_slots = state.occupied_slots_on_path([0, 1, 2])
    departing.snr_requirement_db = 1000
    assert validator.verify_allocation(
        candidate, [0, 1, 2], 0, state, first_slot=2
    ) == (False, None)
    np.testing.assert_array_equal(
        original_slots, state.occupied_slots_on_path([0, 1, 2])
    )
    assert set(state.get_all_allocated_services()) == {1, 2}
    assert all(
        link.noise_cache is None for link in state.get_all_link_states().values()
    )


def test_load_changes_reach_incumbents_beyond_the_candidate_route() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, edfa={"max_output_signal_power_w": 0.001})
    graph.add_edge(1, 2)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    state = NetworkState(topology, grid, slots_per_channel=4)
    validator = QoTValidator(MultiBandISRSGN(grid, fiber, 32e9), topology, grid)
    state.allocate_service(
        AllocatedService(1, 0, 2, 0, 1, 100, 0, 0.001, 2, [0, 1, 2], 0, 0, 4)
    )
    downstream = AllocatedService(2, 1, 2, 0, 1, 100, 0, 0.001, 2, [1, 2], 1, 4, 4)
    state.allocate_service(downstream)
    candidate = ServiceRequest(3, 0, 1, 0, 1, 100, 0, 0.001)
    assert validator.verify_allocation(candidate, [0, 1], 2, state, first_slot=8)[0]
    downstream.snr_requirement_db = 1000
    assert validator.verify_allocation(candidate, [0, 1], 2, state, first_slot=8) == (
        False,
        None,
    )


def test_carrier_loading_settles_coupled_link_cycle() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)], length_km=80)
    fiber = FiberSpanConfig(length_km=80)
    edfa = EDFAConfig(max_output_signal_power_w=0.001)
    topology = NetworkTopology(graph, fiber, edfa, ROADMConfig())
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    state = NetworkState(topology, grid, slots_per_channel=4)
    validator = QoTValidator(MultiBandISRSGN(grid, fiber, 32e9), topology, grid)
    state.allocate_service(
        AllocatedService(1, 0, 2, 0, 1, 100, 0, 0.001, 2, [0, 1, 2], 0, 0, 4)
    )
    state.allocate_service(
        AllocatedService(2, 1, 0, 0, 1, 100, 0, 0.001, 2, [1, 2, 0], 1, 4, 4)
    )
    candidate = ServiceRequest(3, 2, 1, 0, 1, 100, 0, 0.001, 2)
    accepted, snrs = validator.verify_allocation(
        candidate, [2, 0, 1], 2, state, first_slot=8
    )
    assert accepted
    assert snrs is not None
    assert np.isfinite(snrs[2])


def test_validity_provider_receives_the_compact_carrier_spectrum() -> None:
    fiber = FiberSpanConfig(length_km=40)
    topology = NetworkTopology(
        nx.Graph([(0, 1, {"length_km": 40})]), fiber, EDFAConfig(), ROADMConfig()
    )
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    state = NetworkState(topology, grid, slots_per_channel=4)

    def bind(geometry: tuple[CarrierGeometry, ...], slot_count: int) -> QoTModel:
        return create_model(
            "pcfm_h3_raman",
            geometry[0].grid,
            fiber,
            32e9,
            connections=geometry,
            slots_per_channel=slot_count,
            raman_steps=8,
        )

    def provider(model: QoTModel) -> H3LinkValidityProvider:
        assert isinstance(model, KernelQoTModel)
        return H3LinkValidityProvider(model)

    validator = QoTValidator(
        MultiBandISRSGN(grid, fiber, 32e9),
        topology,
        grid,
        validity_policy=LinkValidityPolicy(mode="flag"),
        carrier_model_factory=bind,
        carrier_validity_factory=provider,
    )
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 0.001, 2, symbol_rate_hz=20e9)
    assert validator.verify_allocation(request, [0, 1], 1, state, first_slot=5)[0]
    report = validator.last_link_validity[0, 1]
    assert report.num_channels == 1
    assert report.active.tolist() == [True]
