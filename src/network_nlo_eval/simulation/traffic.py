"""Traffic request models and simulation traffic generation."""

# ruff: noqa: NPY002  # Preserve the reference implementation's seeded sequences.

import math
import random
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from network_nlo_eval.core.flexgrid import bits_per_symbol, slots_for_rate
from network_nlo_eval.core.types import NodeID
from network_nlo_eval.rwa.path_computation import PathCache

#: Default modulation for deriving slot requirements from bit rate.
DEFAULT_MODULATION = "QPSK"


def slots_for_bit_rate(
    bit_rate_gbps: float,
    *,
    slot_width_hz: float,
    modulation: str = DEFAULT_MODULATION,
    roll_off: float = 0.0,
) -> int:
    """Calculate required slots for a service bit rate.

    Parameters
    ----------
    bit_rate_gbps : float
        Service bit rate in Gbit/s.
    slot_width_hz : float
        Slot width in Hz.
    modulation : str, optional
        Supported modulation format.
    roll_off : float, optional
        RRC roll-off factor in the interval [0, 1).

    Returns
    -------
    int
        Number of slots, rounded up.
    """
    return slots_for_rate(
        bit_rate_gbps * 1e9,
        bits_per_symbol(modulation),
        roll_off,
        slot_width_hz,
    )


@dataclass(slots=True)
class ServiceRequest:
    """Describe a network service request.

    Parameters
    ----------
    service_id : int
        Unique service identifier.
    source_id, destination_id : int
        Internal source and destination node IDs.
    arrival_time, departure_time : float
        Arrival and departure times in simulation units.
    bit_rate_gbps : float
        Requested bit rate in Gbit/s.
    snr_requirement_db : float
        Minimum required SNR in dB.
    launch_power_w : float
        Total dual-polarization launch power in watts.
    n_slots : int, optional
        Number of contiguous slots requested; defaults to one.
    symbol_rate_hz : float or None, optional
        Carrier baud rate. Flexible allocations without a rate use the model's
        default, limited to the reserved slot width.
    """

    service_id: int  # Unique service ID.
    source_id: NodeID  # Internal source ID.
    destination_id: NodeID  # Internal destination ID.
    arrival_time: float  # Arrival time in simulation units.
    departure_time: float  # Departure time in simulation units.
    bit_rate_gbps: float  # Bit rate in Gbit/s.
    snr_requirement_db: float  # Minimum SNR in dB.

    # Launch power is a starting value for RWA.
    launch_power_w: float  # Total launch power in watts.

    # A request occupies a contiguous slot interval. The default is one slot.
    # The caller derives the slot count from the rate and grid when needed.

    n_slots: int = 1
    symbol_rate_hz: float | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        """Require a positive slot count and optional baud rate."""
        if (
            isinstance(cast("object", self.n_slots), bool)
            or not isinstance(cast("object", self.n_slots), int)
            or self.n_slots < 1
        ):
            msg = f"n_slots must be a positive integer, got {self.n_slots!r}."
            raise ValueError(msg)
        if self.symbol_rate_hz is not None and (
            not math.isfinite(self.symbol_rate_hz) or self.symbol_rate_hz <= 0
        ):
            msg = "symbol_rate_hz must be finite and positive."
            raise ValueError(msg)


@dataclass(slots=True)
class AllocatedService(ServiceRequest):
    """Describe a service request with its assigned route and spectrum.

    Parameters
    ----------
    path : list of int, optional
        Assigned route as internal node IDs.
    wavelength : int, optional
        Anchor channel index retained for channel-based allocation.
    first_slot : int, optional
        First slot of the allocated half-open interval.
    slots_per_channel : int or None, optional
        Slots per channel for flexible-grid allocation. None selects the
        legacy channel-based allocation path.

    See Also
    --------
    ServiceRequest : Inherited request parameters.
    """

    # Inherited default fields require defaults for the allocation fields.

    path: list[NodeID] = field(default_factory=list)
    wavelength: int = 0  # Anchor channel index.

    first_slot: int = 0  # First allocated slot.

    # None selects legacy channel allocation; a positive integer selects slots.

    slots_per_channel: int | None = None

    class Config:
        frozen = True  # Allocations are normally fixed after assignment.

    @property
    def slot_interval(self) -> tuple[int, int]:
        """Return the occupied half-open slot interval."""
        return self.first_slot, self.first_slot + self.n_slots

    def __post_init__(self) -> None:
        """Validate the slot interval and its anchor channel."""
        # Explicit base call is needed for this slotted dataclass.
        ServiceRequest.__post_init__(self)
        if (
            isinstance(cast("object", self.first_slot), bool)
            or not isinstance(cast("object", self.first_slot), int)
            or self.first_slot < 0
        ):
            msg = f"first_slot must be a non-negative integer, got {self.first_slot!r}."
            raise ValueError(msg)
        if self.slots_per_channel is not None:
            if (
                isinstance(cast("object", self.slots_per_channel), bool)
                or not isinstance(cast("object", self.slots_per_channel), int)
                or self.slots_per_channel < 1
            ):
                msg = f"slots_per_channel must be a positive integer or None, got {self.slots_per_channel!r}."
                raise ValueError(msg)
            expected = self.first_slot // self.slots_per_channel
            if self.wavelength != expected:
                msg = (
                    f"AllocatedService wavelength={self.wavelength} is inconsistent with "
                    f"first_slot={self.first_slot} // slots_per_channel={self.slots_per_channel} = {expected}."
                )
                raise ValueError(msg)


def generate_services(
    path_cache: PathCache,
    service_num: int,
    avg_arrival_interval: float,
    avg_holding_time: float,
    seed: int | None = None,
    slot_width_hz: float | None = None,
    modulation: str = DEFAULT_MODULATION,
    roll_off: float = 0.0,
) -> list[ServiceRequest]:
    """Generate network service requests for a simulation.

    Parameters
    ----------
    path_cache : PathCache
        Cached paths identifying available source-destination pairs.
    service_num : int
        Number of requests to generate.
    avg_arrival_interval : float
        Mean interval between Poisson arrivals.
    avg_holding_time : float
        Mean exponentially distributed holding time.
    seed : int or None, optional
        Random seed for reproducible traffic.
    slot_width_hz : float or None, optional
        Slot width in Hz. If omitted, every request uses one slot. Setting it
        derives the number of slots without changing the random draws.
    modulation : str, optional
        Modulation format used to determine the slot requirement.
    roll_off : float, optional
        RRC roll-off factor used to determine the slot requirement.

    Returns
    -------
    list of ServiceRequest
        Generated requests sorted by arrival time.

    Raises
    ------
    ValueError
        If the path cache contains no connected node pairs.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    services: list[ServiceRequest] = []

    # Select among node pairs with cached paths.
    node_pairs = list(path_cache.get_all_ksp_pairs())
    if not node_pairs:
        msg = "No valid node pairs with KSP paths found in PathCache."
        raise ValueError(msg)

    lambda_rate = 1.0 / avg_arrival_interval  # Arrival rate.
    mu_rate = 1.0 / avg_holding_time  # Inverse mean holding time.

    current_time = 0.0

    # Draw the first arrival time.
    current_time += np.random.exponential(1.0 / lambda_rate)
    modulation_bits = bits_per_symbol(modulation) if slot_width_hz is not None else 0

    for service_id in range(service_num):
        # Choose a source-destination pair.
        source_id, destination_id = random.choice(node_pairs)

        arrival_time = current_time
        holding_time = np.random.exponential(1.0 / mu_rate)
        departure_time = arrival_time + holding_time

        # Draw the next arrival interval.
        next_arrival_interval = np.random.exponential(1.0 / lambda_rate)
        current_time += next_arrival_interval

        # Sample a bit rate with inverse-rate weighting.

        bit_rate_candidates_gbps = np.arange(
            100, 501, 10
        )  # 100 Gbps to 500 Gbps, step 10 Gbps
        weights = 1.0 / bit_rate_candidates_gbps
        weights = weights / np.sum(weights)  # Normalize weights.

        bit_rate_gbps = float(np.random.choice(bit_rate_candidates_gbps, p=weights))

        # Assign an illustrative SNR requirement from the bit rate.
        if bit_rate_gbps > 400:
            snr_requirement_db = 22.0  # Higher rates require higher SNR.
        elif bit_rate_gbps > 300:
            snr_requirement_db = 18.5
        elif bit_rate_gbps > 200:
            snr_requirement_db = 14.5
        else:
            snr_requirement_db = 10.5  # Lower rates require lower SNR.

        # Use the same initial power for all services.
        launch_power_w = 1e-3  # 0 dBm

        # Keep the legacy one-slot default without consuming random numbers.
        n_slots = (
            1
            if slot_width_hz is None
            else slots_for_bit_rate(
                bit_rate_gbps,
                slot_width_hz=slot_width_hz,
                modulation=modulation,
                roll_off=roll_off,
            )
        )

        service = ServiceRequest(
            service_id=service_id,
            source_id=source_id,
            destination_id=destination_id,
            arrival_time=arrival_time,
            departure_time=departure_time,
            bit_rate_gbps=bit_rate_gbps,
            snr_requirement_db=snr_requirement_db,
            launch_power_w=launch_power_w,
            n_slots=n_slots,
            symbol_rate_hz=bit_rate_gbps * 1e9 / modulation_bits
            if modulation_bits
            else None,
        )
        services.append(service)

    # Ensure simulation events are processed in arrival order.
    services.sort(key=lambda s: s.arrival_time)
    return services
