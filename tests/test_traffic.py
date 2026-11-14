"""Tests for service requests and reproducible traffic generation."""

from typing import Any

import networkx as nx
import pytest

from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.simulation.traffic import (
    AllocatedService,
    generate_services,
    slots_for_bit_rate,
)


def test_slot_count_and_allocated_service_validation() -> None:
    assert slots_for_bit_rate(100, slot_width_hz=12.5e9) == 4
    fields: dict[str, Any] = {
        "service_id": 1,
        "source_id": 0,
        "destination_id": 1,
        "arrival_time": 0.0,
        "departure_time": 1.0,
        "bit_rate_gbps": 100.0,
        "snr_requirement_db": 10.0,
        "launch_power_w": 0.001,
        "n_slots": 4,
        "first_slot": 4,
        "wavelength": 1,
        "slots_per_channel": 4,
    }
    assert AllocatedService(**fields).slot_interval == (4, 8)
    fields["wavelength"] = 0
    with pytest.raises(ValueError, match="inconsistent"):
        AllocatedService(**fields)
    fields["n_slots"] = 0
    with pytest.raises(ValueError, match="n_slots"):
        AllocatedService(**fields)


def test_seeded_traffic_is_reproducible_and_slot_width_does_not_change_draws() -> None:
    graph = nx.Graph([(1, 2), (2, 3)])
    topology = NetworkTopology(
        graph, FiberSpanConfig(length_km=80), EDFAConfig(), ROADMConfig()
    )
    cache = PathCache(topology)
    legacy = generate_services(cache, 20, 5, 10, seed=17)
    slotted = generate_services(cache, 20, 5, 10, seed=17, slot_width_hz=12.5e9)
    assert [s.arrival_time for s in legacy] == sorted(s.arrival_time for s in legacy)
    assert [
        (s.source_id, s.destination_id, s.arrival_time, s.bit_rate_gbps) for s in legacy
    ] == [
        (s.source_id, s.destination_id, s.arrival_time, s.bit_rate_gbps)
        for s in slotted
    ]
    assert all(s.n_slots == 1 for s in legacy)
    assert all(
        s.n_slots == slots_for_bit_rate(s.bit_rate_gbps, slot_width_hz=12.5e9)
        for s in slotted
    )
