"""Integration tests for discrete-event simulation."""

from dataclasses import asdict

import networkx as nx

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import BaseRWAAllocator
from network_nlo_eval.simulation.engine import SimulatorEngine
from network_nlo_eval.simulation.reporter import MetricsCollector
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


class SingleChannelAllocator(BaseRWAAllocator):
    """Allocate the only channel when it is free."""

    def allocate(
        self, service_request: ServiceRequest, network_state: NetworkState
    ) -> tuple[bool, AllocatedService | None]:
        if network_state.get_link_state(0, 1).occupied_channels[0]:
            return False, None
        return True, AllocatedService(
            **asdict(service_request), path=[0, 1], wavelength=0
        )

    def release(
        self, allocated_service: AllocatedService, network_state: NetworkState
    ) -> None:
        network_state.release_service(allocated_service)


def test_arrival_blocking_departure_and_reallocation() -> None:
    topology = NetworkTopology(
        nx.Graph([(10, 20)]), FiberSpanConfig(length_km=80), EDFAConfig(), ROADMConfig()
    )
    grid = SpectrumGrid(1, 50e9, 193.1e12)
    state = NetworkState(topology, grid)
    requests = [
        ServiceRequest(0, 0, 1, 0, 5, 100, 10, 0.001),
        ServiceRequest(1, 0, 1, 1, 3, 100, 10, 0.001),
        ServiceRequest(2, 0, 1, 6, 9, 100, 10, 0.001),
    ]
    collector = MetricsCollector(topology, grid, len(requests))
    engine = SimulatorEngine(SingleChannelAllocator(), state, collector, requests)
    engine.run()
    assert engine.current_time == 9
    assert state.get_all_allocated_services() == {}
    assert not state.get_link_state(0, 1).occupied_channels.any()
    events = collector.generate_report(state)["service_events"]
    assert [(event["time"], event["event_type"]) for event in events] == [
        (0, "AllocationSuccess"),
        (1, "AllocationFailure"),
        (5, "Release"),
        (6, "AllocationSuccess"),
        (9, "Release"),
    ]
