"""Core discrete-event simulation engine."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from network_nlo_eval.rwa.allocators import BaseRWAAllocator

    from network_nlo_eval.network.state import NetworkState
    from network_nlo_eval.simulation.reporter import MetricsCollector
    from network_nlo_eval.simulation.traffic import ServiceRequest


@dataclass(order=True)
class Event:
    """An event ordered by simulation time for use in a priority queue.

    Parameters
    ----------
    time : float
        Event time in simulation units.
    event_type : str
        Event kind, either ``"ARRIVAL"`` or ``"DEPARTURE"``.
    service_id : int
        Identifier of the associated service.
    service_data : ServiceRequest
        Request or allocated service associated with the event.
    """

    time: float = field(compare=True)  # Event time controls priority.
    event_type: str = field(compare=False)  # "ARRIVAL" or "DEPARTURE"
    service_id: int = field(compare=False)  # Associated service ID.

    # Service data is excluded from event ordering.
    service_data: ServiceRequest = field(compare=False)


class SimulatorEngine:
    """Process service arrivals and departures in chronological order.

    Parameters
    ----------
    allocator : BaseRWAAllocator
        Routing and wavelength allocation implementation.
    network_state : NetworkState
        Mutable network resource state.
    metrics_collector : MetricsCollector
        Collector for service events and simulation metrics.
    incoming_services : list of ServiceRequest
        Requests whose arrival events will be scheduled.
    """

    def __init__(
        self,
        allocator: BaseRWAAllocator,
        network_state: NetworkState,
        metrics_collector: MetricsCollector,
        incoming_services: list[ServiceRequest],
    ):
        """Initialize the simulation and schedule incoming services."""
        self.allocator = allocator
        self.network_state = network_state
        self.metrics_collector = metrics_collector
        self.current_time = 0.0
        self.event_queue: list[Event] = []

        # Schedule each incoming service arrival.
        for service in incoming_services:
            self.schedule(
                Event(service.arrival_time, "ARRIVAL", service.service_id, service)
            )

    def schedule(self, event: Event) -> None:
        """Add an event to the priority queue.

        Parameters
        ----------
        event : Event
            Event to schedule.
        """
        heapq.heappush(self.event_queue, event)

    def run(self) -> None:
        """Run the simulation until the event queue is empty."""
        # Display progress over arrival events.
        with tqdm(total=len(self.event_queue), desc="Running Simulation") as pbar:
            while self.event_queue:
                event = heapq.heappop(self.event_queue)
                self.current_time = event.time

                if event.event_type == "ARRIVAL":
                    self._handle_arrival(event)
                    # Advance progress once per arrival.
                    pbar.update(1)
                elif event.event_type == "DEPARTURE":
                    self._handle_departure(event)
                # Additional event types may be handled here.

    def _handle_arrival(self, event: Event) -> None:
        """Allocate an arriving service or record it as blocked."""
        service_request = event.service_data

        # Attempt resource allocation.
        is_success, allocated_service = self.allocator.allocate(
            service_request, self.network_state
        )

        if is_success:
            # Update state, record success, and schedule departure.
            if (
                allocated_service is None
            ):  # An allocator must return a service on success.
                msg = "Allocator returned success but allocated_service is None."
                raise ValueError(msg)
            self.network_state.allocate_service(allocated_service)
            self.metrics_collector.record_allocation_success(allocated_service)

            # Schedule the departure event.
            self.schedule(
                Event(
                    allocated_service.departure_time,
                    "DEPARTURE",
                    allocated_service.service_id,
                    allocated_service,  # Carry the full allocated service to departure.
                )
            )
        else:
            # Record allocation failure.
            self.metrics_collector.record_allocation_failure(service_request)

    def _handle_departure(self, event: Event) -> None:
        """Release an allocated service and record its departure."""
        # A departure carries an AllocatedService.
        allocated_service = event.service_data

        # The service may already have been released or preempted.
        if (
            self.network_state.get_allocated_service(allocated_service.service_id)
            is not None
        ):
            # Release resources.
            self.allocator.release(allocated_service, self.network_state)
            self.metrics_collector.record_service_release(allocated_service)
