"""Dynamic network resource state, including spectrum occupancy and link power."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import (
    LinkKey,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    NodeID,
)
from network_nlo_eval.models.power_conventions import resolve_launch_power_convention
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.traffic import AllocatedService

if TYPE_CHECKING:
    from network_nlo_eval.rwa.validity import LinkValidity


@dataclass
class LinkNoiseCache:
    """Cache per-channel link noise and physical span count.

    Parameters
    ----------
    spm, xpm, ase : numpy.ndarray
        Per-channel SPM, XPM, and ASE noise powers in watts.
    span_count : int
        Number of physical spans on the link.
    validity : LinkValidity or None, optional
        Optional network-level validity report.
    launch_power_convention : str or None, optional
        Coordinate convention of the noise arrays: ``"total"`` or ``"per_pol"``.
        None denotes a legacy cache without a recorded convention.

    Notes
    -----
    A cache with a known convention must not be reused by an evaluator using
    a different convention. Legacy caches preserve their previous behavior.
    """

    spm: NDArrayFloat  # Per-channel SPM noise power (W).
    xpm: NDArrayFloat  # Per-channel XPM noise power (W).
    ase: NDArrayFloat  # Per-channel ASE noise power (W).
    span_count: int  # Number of physical spans.
    validity: "LinkValidity | None" = None  # Optional validity report.
    #: Power convention of the noise arrays; metadata only.
    #: ``"total"``, ``"per_pol"``, or None for an unlabelled legacy cache.
    launch_power_convention: str | None = None

    def __post_init__(self) -> None:
        """Validate any recorded launch-power convention."""
        if self.launch_power_convention is not None:
            self.launch_power_convention = resolve_launch_power_convention(
                self.launch_power_convention
            )


@dataclass
class LinkState:
    """Track dynamic resource state for a physical link.

    Parameters
    ----------
    link_key : tuple of int
        Link key normalized as ``(min(u, v), max(u, v))``.
    length_m : float
        Physical link length in meters.
    occupied_channels : numpy.ndarray
        Boolean channel occupancy.
    launch_power_profile_w : numpy.ndarray
        Per-channel launch powers in watts, zero for unoccupied channels.
    allocated_service_ids : numpy.ndarray
        Service ID per channel, with -1 indicating no allocation.
    slots_per_channel : int, optional
        Number of slots in each channel.
    occupied_slots : numpy.ndarray or None, optional
        Slot-level occupancy; initialized empty when omitted.
    allocated_service_ids_slots : numpy.ndarray or None, optional
        Per-slot service IDs; initialized to -1 when omitted.

    Notes
    -----
    Legacy channel allocation does not update the slot-level arrays.
    """

    link_key: LinkKey  # Normalized link key.
    length_m: float  # Link length in meters.

    # Per-channel occupancy and launch power.
    # True indicates an occupied channel.
    occupied_channels: NDArrayBool
    # Launch power in watts; zero for an unoccupied channel.
    # These are transmit powers at the link input, not received powers.
    launch_power_profile_w: NDArrayFloat
    # A service ID of -1 indicates an unallocated channel.
    allocated_service_ids: NDArrayInt

    # Noise is computed lazily and invalidated when power changes.
    _noise_cache: LinkNoiseCache | None = field(default=None, init=False, repr=False)

    # Slot-level state is only updated by slot-range allocation.
    # Array length is num_channels * slots_per_channel.
    slots_per_channel: int = 1
    occupied_slots: NDArrayBool | None = None
    allocated_service_ids_slots: NDArrayInt | None = None

    def __post_init__(self) -> None:
        """Copy input arrays and initialize optional slot arrays."""
        self.occupied_channels = self.occupied_channels.copy()
        self.launch_power_profile_w = self.launch_power_profile_w.copy()
        self.allocated_service_ids = self.allocated_service_ids.copy()

        if isinstance(self.slots_per_channel, bool) or self.slots_per_channel < 1:
            msg = f"slots_per_channel must be a positive integer, got {self.slots_per_channel!r}."
            raise ValueError(msg)
        if self.occupied_slots is None:
            self.occupied_slots = np.zeros(
                len(self.occupied_channels) * self.slots_per_channel, dtype=bool
            )
        else:
            self.occupied_slots = self.occupied_slots.copy()
        if self.allocated_service_ids_slots is None:
            self.allocated_service_ids_slots = np.full(
                len(self.occupied_slots), -1, dtype=np.int_
            )
        else:
            self.allocated_service_ids_slots = self.allocated_service_ids_slots.copy()

    @property
    def slot_occupancy(self) -> NDArrayBool:
        """Return the initialized slot-occupancy array."""
        if (
            self.occupied_slots is None
        ):  # pragma: no cover - defensive against incomplete initialization
            msg = "occupied_slots is not initialised."
            raise RuntimeError(msg)
        return self.occupied_slots

    @property
    def slot_service_ids(self) -> NDArrayInt:
        """Return the initialized array of service IDs per slot."""
        if (
            self.allocated_service_ids_slots is None
        ):  # pragma: no cover - defensive against incomplete initialization
            msg = "allocated_service_ids_slots is not initialised."
            raise RuntimeError(msg)
        return self.allocated_service_ids_slots

    @property
    def noise_cache(self) -> LinkNoiseCache | None:
        """Return the cached noise, or None if it must be computed."""
        return self._noise_cache

    @noise_cache.setter
    def noise_cache(self, value: LinkNoiseCache) -> None:
        self._noise_cache = value

    def invalidate_noise_cache(self) -> None:
        """Invalidate the noise cache after the power profile changes."""
        self._noise_cache = None

    def allocate(
        self, channel_idx: int, launch_power_w: float, service_id: int
    ) -> None:
        """Allocate a channel and invalidate cached noise.

        Parameters
        ----------
        channel_idx : int
            Channel index.
        launch_power_w : float
            Channel launch power in watts.
        service_id : int
            Identifier of the service occupying the channel.

        Raises
        ------
        ValueError
            If the channel is already occupied.
        """
        if self.occupied_channels[channel_idx]:
            msg = f"Channel {channel_idx} on link {self.link_key} is already occupied."
            raise ValueError(msg)
        self.occupied_channels[channel_idx] = True
        self.launch_power_profile_w[channel_idx] = launch_power_w
        self.allocated_service_ids[channel_idx] = service_id
        self.invalidate_noise_cache()

    def release(self, channel_idx: int) -> None:
        """Release a channel and invalidate cached noise.

        Parameters
        ----------
        channel_idx : int
            Channel index. Releasing an unoccupied channel has no effect.
        """
        if not self.occupied_channels[channel_idx]:
            return  # Ignore repeated releases.
        self.occupied_channels[channel_idx] = False
        self.launch_power_profile_w[channel_idx] = 0.0
        self.allocated_service_ids[channel_idx] = -1  # Mark unallocated.
        self.invalidate_noise_cache()

    # ------------------------------------------------------------------
    # Slot interval allocation and release.
    # ------------------------------------------------------------------
    def allocate_slot_range(
        self, first_slot: int, n_slots: int, launch_power_w: float, service_id: int
    ) -> None:
        """Allocate a contiguous half-open slot interval.

        Parameters
        ----------
        first_slot : int
            Zero-based index of the first slot.
        n_slots : int
            Number of requested slots.
        launch_power_w : float
            Launch power in watts, written to the anchor channel.
        service_id : int
            Identifier of the service occupying the interval.

        Raises
        ------
        ValueError
            If the slot count is invalid, the interval is out of bounds, or any
            requested slot is already occupied.

        Notes
        -----
        Occupancy expands to all subslots of every intersected channel, preventing
        channel and slot occupancy from conflicting. Physical launch power is
        written only to the anchor channel used by QoT validation.
        """
        if n_slots < 1:
            msg = f"n_slots must be >= 1, got {n_slots}."
            raise ValueError(msg)
        per_channel = self.slots_per_channel
        occupied_slots = self.slot_occupancy
        service_ids = self.slot_service_ids
        num_slots = len(occupied_slots)
        if first_slot < 0 or first_slot + n_slots > num_slots:
            msg = (
                f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for "
                f"{num_slots} slots on link {self.link_key}."
            )
            raise ValueError(msg)
        window = slice(first_slot, first_slot + n_slots)
        if occupied_slots[window].any():
            msg = f"Slot range [{first_slot}, {first_slot + n_slots}) on link {self.link_key} is already occupied."
            raise ValueError(msg)

        occupied_slots[window] = True
        service_ids[window] = service_id

        # Occupy all subslots in the first and last intersected channels.
        first_channel = first_slot // per_channel
        last_channel = (first_slot + n_slots - 1) // per_channel
        aligned = slice(first_channel * per_channel, (last_channel + 1) * per_channel)
        occupied_slots[aligned] = True
        service_ids[aligned] = service_id

        # Write physical launch power only to the QoT anchor channel.
        self.occupied_channels[first_channel] = True
        self.launch_power_profile_w[first_channel] = launch_power_w
        self.allocated_service_ids[first_channel] = service_id
        self.invalidate_noise_cache()

    def release_slot_range(self, first_slot: int, n_slots: int) -> None:
        """Release a contiguous slot interval and its channel-aligned occupancy.

        Parameters
        ----------
        first_slot : int
            Zero-based index of the first slot.
        n_slots : int
            Number of slots. Invalid or out-of-bounds requests are ignored.
        """
        if n_slots < 1:
            return  # Ignore invalid or repeated releases.
        per_channel = self.slots_per_channel
        occupied_slots = self.slot_occupancy
        service_ids = self.slot_service_ids
        num_slots = len(occupied_slots)
        if first_slot < 0 or first_slot + n_slots > num_slots:
            return  # Ignore out-of-bounds releases.

        # Clear both the requested and channel-aligned intervals.
        first_channel = first_slot // per_channel
        last_channel = (first_slot + n_slots - 1) // per_channel
        lo = min(first_slot, first_channel * per_channel)
        hi = max(first_slot + n_slots, (last_channel + 1) * per_channel)
        window = slice(lo, hi)

        occupied_slots[window] = False
        service_ids[window] = -1

        # Clear channel power when all its subslots are free.
        for channel_idx in range(first_channel, last_channel + 1):
            span = slice(channel_idx * per_channel, (channel_idx + 1) * per_channel)
            if not occupied_slots[span].any():
                self.occupied_channels[channel_idx] = False
                self.launch_power_profile_w[channel_idx] = 0.0
                self.allocated_service_ids[channel_idx] = -1
        self.invalidate_noise_cache()


class NetworkState:
    """Manage link occupancy, power profiles, and allocated network services.

    Parameters
    ----------
    network_topology : NetworkTopology
        Physical topology used to initialize link states.
    spectrum_grid : SpectrumGrid
        Grid defining the number of channels.
    slots_per_channel : int, optional
        Number of slots per channel; the default is one.
    """

    def __init__(
        self,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        slots_per_channel: int = 1,
    ):
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        # One slot per channel preserves legacy channel semantics.
        if isinstance(slots_per_channel, bool) or slots_per_channel < 1:
            msg = f"slots_per_channel must be a positive integer, got {slots_per_channel!r}."
            raise ValueError(msg)
        self.slots_per_channel = slots_per_channel
        self._link_states: dict[LinkKey, LinkState] = {}
        self._allocated_services: dict[
            int, AllocatedService
        ] = {}  # service_id -> AllocatedService

        # Initialize every link state.
        for link_key in network_topology.get_all_internal_links():
            u, v = link_key
            fiber_config = network_topology.get_fiber_config(u, v)
            self._link_states[link_key] = LinkState(
                link_key=link_key,
                length_m=fiber_config.length_km * 1000,
                occupied_channels=np.zeros(spectrum_grid.num_channels, dtype=bool),
                launch_power_profile_w=np.zeros(
                    spectrum_grid.num_channels, dtype=np.float64
                ),
                allocated_service_ids=np.full(
                    spectrum_grid.num_channels, -1, dtype=np.int_
                ),
                slots_per_channel=slots_per_channel,
            )

    def get_link_state(self, u_idx: NodeID, v_idx: NodeID) -> LinkState:
        """Return the current state of a link."""
        link_key = (u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx)
        return self._link_states[link_key]

    def get_all_link_states(self) -> dict[LinkKey, LinkState]:
        """Return the current states of all links."""
        return self._link_states

    def get_allocated_service(self, service_id: int) -> AllocatedService | None:
        """Return a service by ID, or None if it is unallocated."""
        return self._allocated_services.get(service_id)

    def get_all_allocated_services(self) -> dict[int, AllocatedService]:
        """Return all allocated services indexed by service ID."""
        return self._allocated_services

    def allocate_service(self, service: AllocatedService) -> None:
        """Allocate a service after route selection and QoT validation.

        Services with ``slots_per_channel=None`` use legacy channel allocation;
        other services use the requested slot interval.
        """
        # Ensure service IDs are unique.
        if service.service_id in self._allocated_services:
            # Release an existing allocation before replacing it.
            self.release_service(self._allocated_services[service.service_id])

        for i in range(len(service.path) - 1):
            u, v = service.path[i], service.path[i + 1]
            link_state = self.get_link_state(u, v)
            if service.slots_per_channel is None:
                link_state.allocate(
                    service.wavelength, service.launch_power_w, service.service_id
                )
            else:
                link_state.allocate_slot_range(
                    service.first_slot,
                    service.n_slots,
                    service.launch_power_w,
                    service.service_id,
                )
        self._allocated_services[service.service_id] = service

    def release_service(self, service: AllocatedService) -> None:
        """Release an allocated service from each link of its route."""
        if service.service_id not in self._allocated_services:
            # warnings.warn(f"Service {service.service_id} not found for release.")
            return  # Ignore repeated or unknown releases.

        for i in range(len(service.path) - 1):
            u, v = service.path[i], service.path[i + 1]
            link_state = self.get_link_state(u, v)
            if service.slots_per_channel is None:
                link_state.release(service.wavelength)
            else:
                link_state.release_slot_range(service.first_slot, service.n_slots)
        del self._allocated_services[service.service_id]

    def occupied_slots_on_path(self, path: list[NodeID]) -> NDArrayBool:
        """Return the union of occupied slots along a path.

        Parameters
        ----------
        path : list of int
            Internal node IDs, with at least two nodes.

        Returns
        -------
        numpy.ndarray
            Boolean array whose entries are true if a slot is occupied on any hop.

        Raises
        ------
        ValueError
            If the path has fewer than two nodes.
        """
        if len(path) < 2:
            msg = f"Path must have at least two nodes, got {path!r}."
            raise ValueError(msg)
        arrays = [
            self.get_link_state(path[i], path[i + 1]).slot_occupancy
            for i in range(len(path) - 1)
        ]
        return cast("NDArrayBool", np.logical_or.reduce(arrays))
