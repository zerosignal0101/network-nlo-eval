"""Collect simulation metrics and events and generate reports."""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import OriginalNodeID
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


class ServiceEvent(BaseModel):
    """Record a service event during simulation."""

    time: float = Field(..., description="Simulation time at which the event occurred")
    event_type: Literal["AllocationSuccess", "AllocationFailure", "Release"] = Field(
        ..., description="Event type"
    )
    service_id: int = Field(..., description="Service ID")
    source_node_id: OriginalNodeID = Field(..., description="Original source node ID")
    dest_node_id: OriginalNodeID = Field(
        ..., description="Original destination node ID"
    )
    bit_rate_gbps: float = Field(..., description="Service bit rate (Gbit/s)")
    wavelength: int | None = Field(
        None, description="Allocated wavelength index, if successful"
    )
    path_original_ids: list[OriginalNodeID] | None = Field(
        None, description="Allocated path using original node IDs, if successful"
    )
    reason: str | None = Field(None, description="Reason for allocation failure")
    departure_time: float | None = Field(
        None, description="Scheduled service departure time"
    )


class MetricsCollector:
    """Collect and calculate simulation performance metrics.

    Parameters
    ----------
    network_topology : NetworkTopology
        Topology used to translate internal node IDs.
    spectrum_grid : SpectrumGrid
        Spectrum grid used to calculate channel utilization.
    total_incoming_services : int
        Number of service requests in the simulation."""

    def __init__(
        self,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        total_incoming_services: int,
    ):
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        self.total_incoming_services = total_incoming_services

        self.service_events: list[ServiceEvent] = []
        self.block_num: int = 0
        self.successfully_allocated_service_count: int = 0
        self.total_hops_for_allocated_services: int = 0
        self.total_allocated_traffic_gbps: float = (
            0.0  # Total successfully allocated traffic for throughput.
        )
        self.total_occupied_slot_hours: float = (
            0.0  # Reserved for future time-averaged utilization.
        )

    def record_allocation_success(self, service: AllocatedService) -> None:
        """Record a successful service allocation."""
        self.successfully_allocated_service_count += 1
        self.total_hops_for_allocated_services += len(service.path) - 1
        self.total_allocated_traffic_gbps += service.bit_rate_gbps

        original_path_ids = [
            self.network_topology.get_original_node_id(node_idx)
            for node_idx in service.path
        ]
        self.service_events.append(
            ServiceEvent(
                time=service.arrival_time,
                event_type="AllocationSuccess",
                service_id=service.service_id,
                source_node_id=self.network_topology.get_original_node_id(
                    service.source_id
                ),
                dest_node_id=self.network_topology.get_original_node_id(
                    service.destination_id
                ),
                bit_rate_gbps=service.bit_rate_gbps,
                wavelength=service.wavelength,
                path_original_ids=original_path_ids,
                departure_time=service.departure_time,
            )
        )

    def record_allocation_failure(
        self,
        service_request: ServiceRequest,
        reason: str = "No suitable path or wavelength found",
    ) -> None:
        """Record a failed service allocation."""
        self.block_num += 1
        self.service_events.append(
            ServiceEvent(
                time=service_request.arrival_time,
                event_type="AllocationFailure",
                service_id=service_request.service_id,
                source_node_id=self.network_topology.get_original_node_id(
                    service_request.source_id
                ),
                dest_node_id=self.network_topology.get_original_node_id(
                    service_request.destination_id
                ),
                bit_rate_gbps=service_request.bit_rate_gbps,
                reason=reason,
            )
        )

    def record_service_release(self, service: AllocatedService) -> None:
        """Record a service release."""
        original_path_ids = [
            self.network_topology.get_original_node_id(node_idx)
            for node_idx in service.path
        ]
        self.service_events.append(
            ServiceEvent(
                time=service.departure_time,
                event_type="Release",
                service_id=service.service_id,
                source_node_id=self.network_topology.get_original_node_id(
                    service.source_id
                ),
                dest_node_id=self.network_topology.get_original_node_id(
                    service.destination_id
                ),
                bit_rate_gbps=service.bit_rate_gbps,
                wavelength=service.wavelength,
                path_original_ids=original_path_ids,
                departure_time=service.departure_time,
            )
        )

    def generate_report(self, final_network_state: NetworkState) -> dict[str, Any]:
        """Calculate final performance metrics and generate a report.

        Parameters
        ----------
        final_network_state : NetworkState
            Network state at the end of the simulation.

        Returns
        -------
        dict
            Simulation metrics and chronologically ordered service events.
        """
        metrics = self._calculate_network_metrics(final_network_state)
        self.service_events.sort(key=lambda x: x.time)  # Sort events chronologically.

        return {
            "simulation_metrics": metrics,
            "service_events": [event.model_dump() for event in self.service_events],
        }

    def _calculate_network_metrics(
        self, final_network_state: NetworkState
    ) -> dict[str, Any]:
        """Calculate key network performance metrics."""
        metrics = {}

        # 1. Blocking rate.
        metrics["total_incoming_services"] = self.total_incoming_services
        metrics["total_block_num"] = self.block_num
        metrics["successfully_allocated_service_count"] = (
            self.successfully_allocated_service_count
        )
        metrics["blocking_rate"] = (
            self.block_num / self.total_incoming_services
            if self.total_incoming_services > 0
            else 0.0
        )

        # 2. Wavelength utilization.
        total_occupied_slots = 0
        total_possible_slots = 0
        for link_state in final_network_state.get_all_link_states().values():
            total_occupied_slots += np.sum(link_state.occupied_channels)
            total_possible_slots += self.spectrum_grid.num_channels
        metrics["wavelength_utilization"] = (
            total_occupied_slots / total_possible_slots
            if total_possible_slots > 0
            else 0.0
        )

        # 3. Average hop count.
        metrics["average_hop_count"] = (
            self.total_hops_for_allocated_services
            / self.successfully_allocated_service_count
            if self.successfully_allocated_service_count > 0
            else 0.0
        )

        # 4. Average throughput.
        metrics["average_throughput_gbps"] = (
            self.total_allocated_traffic_gbps / self.total_incoming_services
            if self.total_incoming_services > 0
            else 0.0
        )

        # 5. Wavelength fragmentation index.
        total_fragmentation_index = 0.0
        num_links_considered_for_fragmentation = 0

        for link_state in final_network_state.get_all_link_states().values():
            free_bands = ~link_state.occupied_channels  # True indicates a free channel.
            num_free_channels = np.sum(free_bands)

            current_link_fragmentation = 0.0

            # Fragmentation is meaningful when multiple, but not all, channels are free.
            if 1 < num_free_channels < self.spectrum_grid.num_channels:
                # Count contiguous free-channel blocks.
                # Pad with False to count blocks touching the array boundaries.
                padded_free_bands = np.concatenate(([False], free_bands, [False]))
                # Count transitions from occupied to free.
                num_free_blocks = np.sum(
                    padded_free_bands[1:] & ~padded_free_bands[:-1]
                )

                # Fragmentation is (number of free blocks - 1) / (number of free channels - 1).
                # One free block has zero fragmentation.
                # Separate free channels approach a fragmentation index of one.
                # Avoid a zero denominator.
                if num_free_channels - 1 > 0:
                    current_link_fragmentation = (num_free_blocks - 1) / (
                        num_free_channels - 1
                    )

            total_fragmentation_index += current_link_fragmentation
            num_links_considered_for_fragmentation += 1

        metrics["average_wavelength_fragmentation_index"] = (
            total_fragmentation_index / num_links_considered_for_fragmentation
            if num_links_considered_for_fragmentation > 0
            else 0.0
        )

        return metrics
