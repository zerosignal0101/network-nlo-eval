"""Service-level QoT accumulation and blocking."""

from collections.abc import Iterator

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey, NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import (
    QoTValidator,
    RoadmTerms,
    accumulate_service_snr,
    fold_roadm_terms,
)
from network_nlo_eval.rwa.validity import LinkValidity, LinkValidityPolicy
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


def test_accumulates_link_noise_and_roadm_filtering() -> None:
    roadm = fold_roadm_terms(
        ROADMConfig(insertion_loss_db=0, add_drop_loss_db=0, filtering_penalty_db=3),
        is_last_hop=True,
    )
    cache = LinkNoiseCache(np.array([1e-6]), np.array([2e-6]), np.array([3e-6]), 1)
    snr = accumulate_service_snr(
        [(roadm, cache)],
        channel_idx=0,
        launch_power_w=1e-3,
        frequency_hz=193.1e12,
        channel_bandwidth_hz=32e9,
        is_blocked=lambda _: False,
    )
    assert snr == pytest.approx(10 * np.log10(1e-3 / (6e-6 * 10**0.3)))


def test_blocked_hop_stops_lazy_iteration() -> None:
    roadm = fold_roadm_terms(ROADMConfig(), is_last_hop=False)
    cache = LinkNoiseCache(np.zeros(1), np.zeros(1), np.zeros(1), 1)

    def hops() -> Iterator[tuple[RoadmTerms, LinkNoiseCache]]:
        yield roadm, cache
        msg = "blocked paths must not evaluate later hops"
        raise AssertionError(msg)

    assert (
        accumulate_service_snr(
            hops(),
            channel_idx=0,
            launch_power_w=1e-3,
            frequency_hz=193.1e12,
            channel_bandwidth_hz=32e9,
            is_blocked=lambda _: True,
        )
        == -np.inf
    )


def test_fixed_gain_acceptance_matches_direct_path_snr() -> None:
    fiber = FiberSpanConfig(length_km=80)
    edfa = EDFAConfig(target_gain_db=6)
    roadm = ROADMConfig(
        insertion_loss_db=0,
        add_drop_loss_db=0,
        filtering_penalty_db=0,
        equalize_output_power=False,
    )
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=80)
    topology = NetworkTopology(graph, fiber, edfa, roadm)
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    model = create_model("ecz", grid, fiber, 32e9)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid)
    path = [0, 1]
    expected = model.evaluate_path(
        topology.build_optical_path(path), np.array([0, 1e-3, 0])
    )

    rejected = ServiceRequest(0, 0, 1, 0, 1, 100, 37, 1e-3)
    assert validator.verify_allocation(rejected, path, 1, state) == (False, None)
    assert state.get_link_state(0, 1).noise_cache is None

    admitted = ServiceRequest(1, 0, 1, 0, 1, 100, 31, 1e-3)
    success, reported = validator.verify_allocation(admitted, path, 1, state)
    assert success
    assert reported is not None
    np.testing.assert_allclose(reported, expected.snr_db, rtol=1e-12)


def test_admission_recalculates_loaded_edfa_when_a_channel_is_added() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        length_km=80,
        edfa={
            "max_output_signal_power_w": 1.5e-3,
            "compression_curve": [
                {"backoff_db": 0},
                {"backoff_db": 2, "noise_figure_penalty_db": 1, "tilt_db_per_thz": 0.5},
            ],
        },
    )
    fiber = FiberSpanConfig(length_km=80, max_span_length_km=40)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    model = create_model("ecz", grid, fiber, 32e9)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid)
    path = [0, 1]
    state.allocate_service(
        AllocatedService(0, 0, 1, 0, 1, 100, 0, 1e-3, path=path, wavelength=0)
    )
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 1e-3)
    accepted, reported = validator.verify_allocation(request, path, 1, state)
    expected = model.evaluate_path(topology.build_optical_path(path), np.full(2, 1e-3))
    assert accepted
    assert reported is not None
    assert expected.signal_power_w.sum() == pytest.approx(1.5e-3)
    assert expected.signal_power_w[1] > expected.signal_power_w[0]
    np.testing.assert_allclose(reported, expected.snr_db, rtol=1e-12)


@pytest.mark.parametrize("path", [[0, 1], [0, 1, 2]])
def test_admission_applies_passbands_and_voas(path: list[int]) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        spans=[
            {
                "fiber": {"length_km": 80},
                "edfa": {"passbands_hz": [[193.025e12, 193.075e12]], "input_voa_db": 3},
            }
        ],
    )
    graph.add_edge(1, 2, length_km=80)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    model = create_model("ecz", grid, fiber, 32e9)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid)
    optical_path = topology.build_optical_path(path)

    rejected = ServiceRequest(0, 0, path[-1], 0, 1, 100, -np.inf, 1e-3)
    assert validator.verify_allocation(rejected, path, 1, state) == (False, None)
    assert state.get_link_state(0, 1).noise_cache is None
    blocked = model.evaluate_path(optical_path, np.array([0, 1e-3, 0]))
    np.testing.assert_array_equal(blocked.signal_power_w, np.zeros(3))
    np.testing.assert_array_equal(blocked.ase_noise_w, np.zeros(3))

    request = ServiceRequest(1, 0, path[-1], 0, 1, 100, 0, 1e-3)
    accepted, reported = validator.verify_allocation(request, path, 0, state)
    assert accepted
    assert reported is not None
    expected = model.evaluate_path(optical_path, np.array([1e-3, 0, 0]))
    np.testing.assert_allclose(reported, expected.snr_db, rtol=1e-12)
    assert expected.signal_power_w[0] == pytest.approx(1e-3 * 10**-0.3)


@pytest.mark.parametrize(
    ("path", "signal_power_w"), [([0, 1], 1e-3 * 10**-0.8), ([0, 1, 2], 1e-4)]
)
def test_admission_matches_direct_roadm_switching_budget(
    path: list[int], signal_power_w: float
) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=80)
    graph.add_edge(1, 2, length_km=80)
    graph.nodes[1]["roadm"] = {
        "output_target": {"power_w": 5e-4},
        "degree_targets": {"2": {"power_w": 1e-4}},
        "path_losses": [
            {"from_degree": 0, "to_degree": 2, "loss_db": 3},
            {"from_degree": 0, "loss_db": 8},
        ],
    }
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    model = create_model("ecz", grid, fiber, 32e9)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid)

    direct = model.evaluate_path(topology.build_optical_path(path), np.array([1e-3]))
    assert direct.signal_power_w[0] == pytest.approx(signal_power_w)
    snr = float(direct.snr_db[0])
    too_high = ServiceRequest(0, 0, path[-1], 0, 1, 100, snr + 0.01, 1e-3)
    assert validator.verify_allocation(too_high, path, 0, state) == (False, None)
    request = ServiceRequest(1, 0, path[-1], 0, 1, 100, snr - 0.01, 1e-3)
    accepted, reported = validator.verify_allocation(request, path, 0, state)
    assert accepted
    assert reported is not None
    np.testing.assert_allclose(reported, direct.snr_db, rtol=1e-12)


def test_installed_spans_and_uniform_loading_use_ordered_propagation() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(
        0,
        1,
        spans=[
            {"fiber": {"length_km": 35, "attenuation_db_km_ref": 0.2}},
            {"fiber": {"length_km": 110}, "edfa": {"target_gain_db": 19}},
        ],
    )
    graph.add_edge(1, 2, length_km=80)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    model = create_model("ecz", grid, fiber, 32e9)
    state = NetworkState(topology, grid)
    validator = QoTValidator(model, topology, grid)
    assert state.get_link_state(0, 1).length_m == 145_000
    assert len(topology.export_to_dict()["connections"][0]["spans"]) == 2

    for path in ([0, 1], [0, 1, 2]):
        physical_path = topology.build_optical_path(path)
        assert physical_path.edfa_configs[0] is None
        assert physical_path.roadm_configs is not None
        assert physical_path.roadm_configs[0] is None
        assert physical_path.link_keys[:2] == [(0, 1), (0, 1)]
        expected = model.evaluate_path(physical_path, np.array([0, 1e-3, 0]))
        assert expected.span_count == len(path)
        service = ServiceRequest(0, 0, path[-1], 0, 1, 100, 0, 1e-3)
        success, reported = validator.verify_allocation(service, path, 1, state)
        assert success
        assert reported is not None
        np.testing.assert_allclose(reported, expected.snr_db, rtol=1e-12)


def test_coherent_admission_uses_full_route_for_uniform_loading() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=60)
    graph.add_edge(1, 2, length_km=80)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    model = create_model("ecz_coherent", grid, fiber, 32e9)
    state = NetworkState(topology, grid)
    validator = QoTValidator(model, topology, grid, coherent_assembly=True)
    path = [0, 1, 2]
    service = ServiceRequest(0, 0, 2, 0, 1, 100, 0, 1e-3)
    accepted, reported = validator.verify_allocation(service, path, 0, state)
    expected = model.evaluate_path(topology.build_optical_path(path), np.array([1e-3]))
    assert accepted
    assert reported is not None
    np.testing.assert_allclose(reported, expected.snr_db, rtol=1e-12)
    assert state.get_link_state(0, 1).noise_cache is None
    assert state.get_link_state(1, 2).noise_cache is None


def test_equal_powers_from_distinct_carriers_use_link_budgets() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=60)
    graph.add_edge(1, 2, length_km=80)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    model = create_model("ecz_coherent", grid, fiber, 32e9)
    state = NetworkState(topology, grid)
    state.allocate_service(
        AllocatedService(0, 0, 1, 0, 1, 100, 0, 1e-3, path=[0, 1], wavelength=0)
    )
    state.allocate_service(
        AllocatedService(1, 1, 2, 0, 1, 100, 0, 1e-3, path=[1, 2], wavelength=0)
    )
    validator = QoTValidator(model, topology, grid, coherent_assembly=True)
    service = ServiceRequest(2, 0, 2, 0, 1, 100, 0, 1e-3)
    accepted, _ = validator.verify_allocation(service, [0, 1, 2], 1, state)
    assert accepted
    assert state.get_link_state(0, 1).noise_cache is not None
    assert state.get_link_state(1, 2).noise_cache is not None


@pytest.mark.parametrize("convention", ["total", "per_pol"])
def test_fixed_grid_uses_propagated_ingress_with_same_channel_replacement(
    convention: str,
) -> None:
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=80, edfa={"max_output_signal_power_w": 1.2e-3})
    graph.add_edge(1, 2, length_km=80)
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(
        graph,
        fiber,
        EDFAConfig(),
        ROADMConfig(insertion_loss_db=0, add_drop_loss_db=0, filtering_penalty_db=0),
    )
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    model = create_model("ecz", grid, fiber, 32e9, launch_power_convention=convention)
    validator = QoTValidator(model, topology, grid)
    state = NetworkState(topology, grid, slots_per_channel=4)
    for source, destination in [(0, 1), (1, 2)]:
        state.allocate_service(
            AllocatedService(
                source,
                source,
                destination,
                0,
                1,
                100,
                0,
                0.6e-3,
                path=[source, destination],
                wavelength=0,
            )
        )
    request = ServiceRequest(2, 0, 2, 0, 1, 100, 27.6, 1e-3, n_slots=4)
    path = [0, 1, 2]

    scale = 0.5 if convention == "per_pol" else 1.0
    first_input = scale * np.array([0.6e-3, 1e-3])
    first = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [topology.get_link_edfa_config(0, 1)]),
        first_input,
    )
    second_input = np.array([first_input[0], first.signal_power_w[1]])
    second_path = MultiSpanOpticalPath([fiber], [topology.get_link_edfa_config(1, 2)])
    second = model.evaluate_path(second_path, second_input)
    expected = -10 * np.log10(
        10 ** (-first.snr_db[1] / 10) + 10 ** (-second.snr_db[1] / 10)
    )
    nominal = model.evaluate_path(second_path, first_input)
    nominal_snr = -10 * np.log10(
        10 ** (-first.snr_db[1] / 10)
        + 10 ** (-nominal.snr_db[1] / 10) * first_input[1] / second_input[1]
    )
    assert nominal_snr < request.snr_requirement_db < expected
    fixed_accepted, fixed_snrs = validator.verify_allocation(request, path, 1, state)
    flex_accepted, flex_snrs = validator.verify_allocation(
        request, path, 1, state, first_slot=4
    )
    assert fixed_accepted
    assert flex_accepted
    assert fixed_snrs is not None
    assert flex_snrs is not None
    assert fixed_snrs[1] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(fixed_snrs, flex_snrs, rtol=0, atol=1e-12)

    inputs: dict[LinkKey, NDArrayFloat] = {}

    def validity_provider(
        link_key: LinkKey,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_profile_w: NDArrayFloat,
        span_length_m: float,
    ) -> LinkValidity:
        assert fiber_config.length_km == 80
        assert edfa_config is not None
        assert span_length_m == 100_000
        inputs[link_key] = power_profile_w.copy()
        active = power_profile_w > 0
        return LinkValidity(
            active, active, ~active, np.ones(2), np.full(2, 0.5), "test"
        )

    with_policy = QoTValidator(
        model,
        topology,
        grid,
        validity_policy=LinkValidityPolicy(mode="degrade"),
        validity_provider=validity_provider,
    )
    assert with_policy.verify_allocation(request, path, 1, state) == (False, None)
    np.testing.assert_allclose(inputs[0, 1], first_input)
    np.testing.assert_allclose(inputs[1, 2], second_input)
    assert set(with_policy.last_link_validity) == {(0, 1), (1, 2)}
    assert state.get_allocated_service(2) is None
    assert not state.occupied_slots_on_path(path)[4:].any()
    assert all(
        link.noise_cache is None for link in state.get_all_link_states().values()
    )
