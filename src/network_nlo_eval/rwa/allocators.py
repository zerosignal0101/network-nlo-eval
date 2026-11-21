"""Routing and spectrum allocation strategies."""

from dataclasses import fields, replace
from math import ceil
from typing import Any, cast

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayBool, NodeID
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import BaseRWAAllocator, QoTValidator
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

_REQUEST_FIELD_NAMES = tuple(field.name for field in fields(ServiceRequest))


def _request_fields(service: ServiceRequest) -> dict[str, Any]:
    return {name: getattr(service, name) for name in _REQUEST_FIELD_NAMES}


class KSPFirstFitAllocator(BaseRWAAllocator):
    """Try cached shortest paths and available channels in first-fit order.

    Each candidate must pass QoT validation before allocation."""

    def __init__(
        self,
        path_cache: PathCache,
        qot_validator: QoTValidator,
        spectrum_grid: SpectrumGrid,
    ):
        """Initialize the allocator.

        Parameters
        ----------
        path_cache : PathCache
            Precomputed candidate paths.
        qot_validator : QoTValidator
            Candidate and existing-service QoT validator.
        spectrum_grid : SpectrumGrid
            Physical spectrum grid."""
        self.path_cache = path_cache
        self.qot_validator = qot_validator
        self.spectrum_grid = spectrum_grid

    def _get_available_channels_on_path(
        self,
        path: list[NodeID],
        network_state: NetworkState,
    ) -> NDArrayBool:
        """Return channels free on every link of the path."""
        occupied_arrays = [
            network_state.get_link_state(path[i], path[i + 1]).occupied_channels
            for i in range(len(path) - 1)
        ]
        return cast("NDArrayBool", ~np.logical_or.reduce(occupied_arrays))

    def allocate(
        self, service_request: ServiceRequest, network_state: NetworkState
    ) -> tuple[bool, AllocatedService | None]:
        """Try paths and channels in order until QoT validation succeeds.

        Returns
        -------
        tuple of bool and AllocatedService or None
            Success flag and allocated service, if successful."""
        mode = service_request.transceiver_mode
        if (
            mode is not None
            and mode.required_slot_width_hz > self.spectrum_grid.channel_spacing_hz
        ):
            return False, None
        ksp_paths = self.path_cache.get_paths(
            service_request.source_id, service_request.destination_id
        )

        if not ksp_paths:
            return False, None

        for path in ksp_paths:
            available_channels_on_path = self._get_available_channels_on_path(
                path, network_state
            )
            free_channel_indices = np.where(available_channels_on_path)[0].tolist()

            if not free_channel_indices:
                continue

            for channel_idx in free_channel_indices:
                is_qot_satisfied, _ = self.qot_validator.verify_allocation(
                    service_request=service_request,
                    path=path,
                    channel_idx=channel_idx,
                    current_network_state=network_state,
                )

                if is_qot_satisfied:
                    allocated_service = AllocatedService(
                        **_request_fields(service_request),
                        path=path,
                        wavelength=channel_idx,
                    )
                    return True, allocated_service

        return False, None

    def release(
        self, allocated_service: AllocatedService, network_state: NetworkState
    ) -> None:
        """Release the allocated service from the network state."""
        network_state.release_service(allocated_service)


def first_fit_contiguous_windows(
    free_mask: NDArrayBool,
    n_slots: int,
    slots_per_channel: int,
    *,
    distinct_anchors: bool = True,
) -> list[int]:
    """Find feasible contiguous free-slot windows in ascending order.

    A prefix sum checks all windows in O(N). Carrier-aware admission considers
    every viable start because shifting a window changes its center frequency.

    Parameters
    ----------
    free_mask : numpy.ndarray
        Boolean slot availability mask.
    n_slots : int
        Required window width, at least one.
    slots_per_channel : int
        Number of slots per physical channel, at least one.
    distinct_anchors : bool, optional
        Keep only the first start in each anchor channel.

    Returns
    -------
    list of int
        Candidate starting slots in ascending order."""
    total = int(free_mask.shape[0])
    if n_slots < 1:
        msg = f"n_slots must be >= 1, got {n_slots}."
        raise ValueError(msg)
    if slots_per_channel < 1:
        msg = f"slots_per_channel must be >= 1, got {slots_per_channel}."
        raise ValueError(msg)
    if n_slots > total:
        return []

    prefix = np.concatenate(([0], np.cumsum(free_mask.astype(np.int64))))
    window_free = prefix[n_slots:] - prefix[:-n_slots]
    starts = np.flatnonzero(window_free == n_slots)
    if not distinct_anchors:
        return cast("list[int]", starts.tolist())

    candidates: list[int] = []
    last_channel = -1
    for start in starts.tolist():
        channel_idx = start // slots_per_channel
        if channel_idx != last_channel:
            candidates.append(int(start))
            last_channel = channel_idx
    return candidates


class KSPFirstFitFlexAllocator(BaseRWAAllocator):
    """Allocate contiguous slot windows on cached shortest paths.

    Slot occupancy and ROADM blocks are checked across each path. QoT is
    evaluated at the center of the reserved interval."""

    def __init__(
        self,
        path_cache: PathCache,
        qot_validator: QoTValidator,
        spectrum_grid: SpectrumGrid,
        network_topology: NetworkTopology,
        slots_per_channel: int,
    ):
        """Initialize the flexible-grid allocator.

        Parameters
        ----------
        path_cache : PathCache
            Precomputed candidate paths.
        qot_validator : QoTValidator
            Candidate and existing-service QoT validator.
        spectrum_grid : SpectrumGrid
            Physical spectrum grid.
        network_topology : NetworkTopology
            Topology supplying ROADM blocks.
        slots_per_channel : int
            Number of slots per physical channel."""
        if isinstance(slots_per_channel, bool) or slots_per_channel < 1:
            msg = f"slots_per_channel must be a positive integer, got {slots_per_channel!r}."
            raise ValueError(msg)
        self.path_cache = path_cache
        self.qot_validator = qot_validator
        self.spectrum_grid = spectrum_grid
        self.network_topology = network_topology
        self.slots_per_channel = slots_per_channel

    def blocked_slots_on_path(self, path: list[NodeID]) -> set[int]:
        """Return the union of blocked slots at all nodes on the path.

        Blocked channels are expanded into their constituent slots."""
        blocked: set[int] = set()
        for node_id in path:
            roadm = self.network_topology.get_roadm_config(node_id)
            blocked.update(roadm.blocked_slots)
            for channel_index in roadm.blocked_channels:
                blocked.update(
                    range(
                        channel_index * self.slots_per_channel,
                        (channel_index + 1) * self.slots_per_channel,
                    )
                )
        return blocked

    def allocate(
        self, service_request: ServiceRequest, network_state: NetworkState
    ) -> tuple[bool, AllocatedService | None]:
        """Find a contiguous slot window that passes QoT validation.

        Returns
        -------
        tuple of bool and AllocatedService or None
            Success flag and allocated service, if successful."""
        if network_state.slots_per_channel != self.slots_per_channel:
            msg = "Allocator slots_per_channel must match network state."
            raise ValueError(msg)
        n_slots = service_request.n_slots
        mode = service_request.transceiver_mode
        if mode is not None:
            slot_width_hz = (
                self.spectrum_grid.channel_spacing_hz / self.slots_per_channel
            )
            n_slots = max(n_slots, ceil(mode.required_slot_width_hz / slot_width_hz))
            if n_slots != service_request.n_slots:
                service_request = replace(service_request, n_slots=n_slots)
        ksp_paths = self.path_cache.get_paths(
            service_request.source_id, service_request.destination_id
        )
        if not ksp_paths:
            return False, None

        for path in ksp_paths:
            occupied = network_state.occupied_slots_on_path(path)
            free = ~occupied

            blocked = self.blocked_slots_on_path(path)
            if blocked:
                in_range = [slot for slot in blocked if 0 <= slot < free.shape[0]]
                if in_range:
                    free = free.copy()
                    free[in_range] = False

            for first_slot in first_fit_contiguous_windows(
                free, n_slots, self.slots_per_channel, distinct_anchors=False
            ):
                channel_idx = first_slot // self.slots_per_channel

                is_qot_satisfied, _ = self.qot_validator.verify_allocation(
                    service_request=service_request,
                    path=path,
                    channel_idx=channel_idx,
                    current_network_state=network_state,
                    first_slot=first_slot,
                )
                if not is_qot_satisfied:
                    continue

                allocated_service = AllocatedService(
                    **_request_fields(service_request),
                    path=path,
                    wavelength=channel_idx,
                    first_slot=first_slot,
                    slots_per_channel=self.slots_per_channel,
                )
                return True, allocated_service

        return False, None

    def release(
        self, allocated_service: AllocatedService, network_state: NetworkState
    ) -> None:
        """Release the allocated slot interval from the network state."""
        network_state.release_service(allocated_service)
