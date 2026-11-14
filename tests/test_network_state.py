"""Tests for channel and slot resource allocation across network links."""

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, LinkState, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.traffic import AllocatedService


def make_state() -> NetworkState:
    graph = nx.Graph([(10, 20), (20, 30)])
    topology = NetworkTopology(
        graph, FiberSpanConfig(length_km=80), EDFAConfig(), ROADMConfig()
    )
    return NetworkState(topology, SpectrumGrid(3, 50e9, 193.1e12), slots_per_channel=4)


def make_service(*, first_slot: int = 1, n_slots: int = 2) -> AllocatedService:
    return AllocatedService(
        service_id=7,
        source_id=0,
        destination_id=2,
        arrival_time=0,
        departure_time=10,
        bit_rate_gbps=100,
        snr_requirement_db=10,
        launch_power_w=0.001,
        path=[0, 1, 2],
        wavelength=first_slot // 4,
        first_slot=first_slot,
        n_slots=n_slots,
        slots_per_channel=4,
    )


def test_allocation_and_release_update_each_link_and_invalidate_cache() -> None:
    state = make_state()
    link = state.get_link_state(0, 1)
    link.noise_cache = LinkNoiseCache(
        np.ones(3), np.ones(3), np.ones(3), 1, launch_power_convention="total"
    )
    service = make_service()
    state.allocate_service(service)
    assert link.noise_cache is None
    assert state.get_allocated_service(7) is service
    assert link.launch_power_profile_w[0] == 0.001
    np.testing.assert_array_equal(
        state.occupied_slots_on_path([0, 1, 2]), [True] * 4 + [False] * 8
    )
    assert all(
        state.get_link_state(u, v).slot_service_ids[0] == 7 for u, v in [(0, 1), (1, 2)]
    )
    state.release_service(service)
    assert state.get_allocated_service(7) is None
    assert not state.occupied_slots_on_path([0, 1, 2]).any()
    assert not link.launch_power_profile_w.any()
    state.release_service(service)


def test_slot_collision_and_boundaries() -> None:
    state = make_state()
    link = state.get_link_state(0, 1)
    link.allocate_slot_range(1, 1, 0.001, 1)
    with pytest.raises(ValueError, match="already occupied"):
        link.allocate_slot_range(3, 1, 0.001, 2)
    with pytest.raises(ValueError, match="out of bounds"):
        link.allocate_slot_range(11, 2, 0.001, 2)
    with pytest.raises(ValueError, match="at least two nodes"):
        state.occupied_slots_on_path([0])


def test_channel_allocation_copies_input_arrays() -> None:
    occupied = np.zeros(2, dtype=bool)
    link = LinkState(
        (0, 1), 80_000, occupied, np.zeros(2), np.full(2, -1, dtype=np.int_)
    )
    link.allocate(0, 0.001, 3)
    assert not occupied[0]
    assert link.occupied_channels[0]
    with pytest.raises(ValueError, match="already occupied"):
        link.allocate(0, 0.001, 4)
    link.release(0)
    assert link.allocated_service_ids[0] == -1
