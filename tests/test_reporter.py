"""Tests for simulation events and summary metrics."""

import networkx as nx
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.reporter import MetricsCollector
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


def test_report_events_and_metrics() -> None:
    topology = NetworkTopology(
        nx.Graph([(10, 20)]), FiberSpanConfig(length_km=80), EDFAConfig(), ROADMConfig()
    )
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    state = NetworkState(topology, grid)
    service = AllocatedService(1, 0, 1, 1, 5, 100, 10, 0.001, path=[0, 1], wavelength=1)
    rejected = ServiceRequest(2, 0, 1, 2, 6, 200, 10, 0.001)
    collector = MetricsCollector(topology, grid, total_incoming_services=2)
    state.allocate_service(service)
    collector.record_service_release(service)
    collector.record_allocation_failure(rejected, "Insufficient spectrum")
    collector.record_allocation_success(service)
    report = collector.generate_report(state)
    metrics = report["simulation_metrics"]
    assert metrics["blocking_rate"] == 0.5
    assert metrics["wavelength_utilization"] == pytest.approx(1 / 3)
    assert metrics["average_hop_count"] == 1
    assert metrics["average_throughput_gbps"] == 50
    assert metrics["average_wavelength_fragmentation_index"] == 1
    events = report["service_events"]
    assert [event["event_type"] for event in events] == [
        "AllocationSuccess",
        "AllocationFailure",
        "Release",
    ]
    assert events[0]["path_original_ids"] == [10, 20]
    assert events[1]["reason"] == "Insufficient spectrum"


def test_empty_report_has_zero_metrics() -> None:
    topology = NetworkTopology(
        nx.Graph(), FiberSpanConfig(length_km=80), EDFAConfig(), ROADMConfig()
    )
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    report = MetricsCollector(topology, grid, 0).generate_report(
        NetworkState(topology, grid)
    )
    assert report["simulation_metrics"]["blocking_rate"] == 0
    assert report["simulation_metrics"]["wavelength_utilization"] == 0
    assert report["service_events"] == []
