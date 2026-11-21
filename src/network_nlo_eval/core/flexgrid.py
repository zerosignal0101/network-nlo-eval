"""Flexible frequency grids and contiguous connection allocations.

Slots occupy half-open intervals. Connection centers use n_slots / 2,
the center of the occupied interval, avoiding a half-slot offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, cast

__all__ = [
    "BITS_PER_SYMBOL",
    "MODULATION_FORMATS",
    "AllocatedLike",
    "ConnectionSpec",
    "FlexibleGrid",
    "bits_per_symbol",
    "connection_spec_from_allocation",
    "slots_for_rate",
]


MODULATION_FORMATS: tuple[str, ...] = ("BPSK", "QPSK", "8QAM", "16QAM", "64QAM")

BITS_PER_SYMBOL: dict[str, int] = {
    "BPSK": 1,
    "QPSK": 2,
    "8QAM": 3,
    "16QAM": 4,
    "64QAM": 6,
}


def bits_per_symbol(modulation: str) -> int:
    """Return the number of bits per symbol for a supported modulation.

    Parameters
    ----------
    modulation : str
        Modulation format name.

    Returns
    -------
    int
        Bits per symbol.

    Raises
    ------
    ValueError
        If the modulation format is unknown.
    """
    try:
        return BITS_PER_SYMBOL[modulation]
    except KeyError:
        msg = f"Unknown modulation format {modulation!r}; expected one of {list(MODULATION_FORMATS)}."
        raise ValueError(msg) from None


def slots_for_rate(
    bit_rate_bps: float,
    bits_per_symbol: int,
    roll_off: float,
    slot_width_hz: float,
) -> int:
    """Calculate the number of slots required for a bit rate.

    Parameters
    ----------
    bit_rate_bps : float
        Bit rate in bits per second.
    bits_per_symbol : int
        Number of bits per symbol.
    roll_off : float
        RRC roll-off factor, in the interval [0, 1).
    slot_width_hz : float
        Slot width in Hz.

    Returns
    -------
    int
        Required number of slots, rounded up.

    Raises
    ------
    ValueError
        If an argument is outside its valid range.
    """
    if bit_rate_bps <= 0:
        msg = f"bit_rate_bps must be positive, got {bit_rate_bps}."
        raise ValueError(msg)
    if bits_per_symbol < 1:
        msg = f"bits_per_symbol must be >= 1, got {bits_per_symbol}."
        raise ValueError(msg)
    if not (0 <= roll_off < 1):
        msg = f"roll_off must satisfy 0 <= roll_off < 1, got {roll_off}."
        raise ValueError(msg)
    if slot_width_hz <= 0:
        msg = f"slot_width_hz must be positive, got {slot_width_hz}."
        raise ValueError(msg)

    required_width_hz = (bit_rate_bps / bits_per_symbol) * (1.0 + roll_off)
    return max(1, math.ceil(required_width_hz / slot_width_hz))


@dataclass(frozen=True)
class FlexibleGrid:
    """Describe a grid of equally spaced, half-open frequency slots.

    Parameters
    ----------
    slot_width_hz : float
        Positive slot width in Hz.
    num_slots : int
        Positive number of slots.
    band_low_edge_hz : float
        Positive frequency of the lower edge in Hz.

    Notes
    -----
    Connection centers are calculated from the midpoint of their occupied
    interval. Nominal band coverage is the responsibility of the caller.
    """

    slot_width_hz: float
    num_slots: int
    band_low_edge_hz: float

    def __post_init__(self) -> None:
        """Validate the grid dimensions."""
        if self.slot_width_hz <= 0:
            msg = f"slot_width_hz must be positive, got {self.slot_width_hz}."
            raise ValueError(msg)
        if (
            isinstance(cast("object", self.num_slots), bool)
            or not isinstance(cast("object", self.num_slots), int)
            or self.num_slots <= 0
        ):
            msg = f"num_slots must be a positive integer, got {self.num_slots!r}."
            raise ValueError(msg)
        if self.band_low_edge_hz <= 0:
            msg = f"band_low_edge_hz must be positive (optical frequencies are positive), got {self.band_low_edge_hz}."
            raise ValueError(msg)

    @property
    def band_high_edge_hz(self) -> float:
        """Return the exclusive upper edge of the grid in Hz."""
        return self.band_low_edge_hz + self.num_slots * self.slot_width_hz

    @property
    def band_center_hz(self) -> float:
        """Return the grid center frequency in Hz."""
        return self.band_low_edge_hz + self.num_slots * self.slot_width_hz / 2

    @property
    def total_bandwidth_hz(self) -> float:
        """Return the total grid bandwidth in Hz."""
        return self.num_slots * self.slot_width_hz

    def _check_slot(self, slot: int) -> int:
        if isinstance(slot, bool) or not isinstance(slot, (int,)):
            msg = f"slot must be an integer, got {type(slot).__name__}."
            raise TypeError(msg)
        if not (0 <= slot < self.num_slots):
            msg = f"Slot index {slot} out of bounds for {self.num_slots} slots."
            raise IndexError(msg)
        return slot

    def slot_low_edge_hz(self, slot: int) -> float:
        """Return the lower edge of a slot in Hz."""
        return self.band_low_edge_hz + self._check_slot(slot) * self.slot_width_hz

    def slot_high_edge_hz(self, slot: int) -> float:
        """Return the exclusive upper edge of a slot in Hz."""
        return self.band_low_edge_hz + (self._check_slot(slot) + 1) * self.slot_width_hz

    def slot_center_hz(self, slot: int) -> float:
        """Return the center frequency of a slot in Hz."""
        return (
            self.band_low_edge_hz + (self._check_slot(slot) + 0.5) * self.slot_width_hz
        )

    def centre_hz(self, first_slot: int, n_slots: int) -> float:
        """Return the midpoint frequency of a contiguous slot interval in Hz.

        Raises
        ------
        ValueError
            If the number of slots is not positive.
        IndexError
            If the interval falls outside the grid.
        """
        if n_slots <= 0:
            msg = f"n_slots must be positive, got {n_slots}."
            raise ValueError(msg)
        if first_slot < 0 or first_slot + n_slots > self.num_slots:
            msg = f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for {self.num_slots} slots."
            raise IndexError(msg)
        return self.band_low_edge_hz + (first_slot + n_slots / 2) * self.slot_width_hz

    def connection_edges_hz(self, first_slot: int, n_slots: int) -> tuple[float, float]:
        """Return the lower and exclusive upper connection edges in Hz."""
        if n_slots <= 0:
            msg = f"n_slots must be positive, got {n_slots}."
            raise ValueError(msg)
        if first_slot < 0 or first_slot + n_slots > self.num_slots:
            msg = f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for {self.num_slots} slots."
            raise IndexError(msg)
        half = n_slots * self.slot_width_hz / 2
        centre = self.centre_hz(first_slot, n_slots)
        return centre - half, centre + half

    def edges_are_inside(self, low_hz: float, high_hz: float) -> bool:
        """Return whether the connection edges lie within the grid."""
        return low_hz >= self.band_low_edge_hz and high_hz <= self.band_high_edge_hz

    @classmethod
    def from_legacy_grid(
        cls,
        num_channels: int,
        channel_spacing_hz: float,
        center_frequency_hz: float,
        slots_per_channel: int = 4,
    ) -> FlexibleGrid:
        """Construct a flexible grid equivalent to a fixed channel grid.

        Each channel is divided into ``slots_per_channel`` equal slots. The
        first channel center is offset half a channel from the lower edge.
        """
        if slots_per_channel < 1:
            msg = f"slots_per_channel must be >= 1, got {slots_per_channel}."
            raise ValueError(msg)
        slot_width = channel_spacing_hz / slots_per_channel

        first_channel_centre = (
            center_frequency_hz - (num_channels - 1) * channel_spacing_hz / 2
        )
        return cls(
            slot_width_hz=slot_width,
            num_slots=num_channels * slots_per_channel,
            band_low_edge_hz=first_channel_centre - channel_spacing_hz / 2,
        )


@dataclass(frozen=True, slots=True)
class CarrierGeometry:
    """Physical geometry of a carrier with a reserved slot interval."""

    grid: FlexibleGrid
    first_slot: int
    n_slots: int
    symbol_rate_hz: float

    @property
    def centre_hz(self) -> float:
        """Return the midpoint of the reserved interval in Hz."""
        return self.grid.centre_hz(self.first_slot, self.n_slots)

    @property
    def occupied_bandwidth_hz(self) -> float:
        """Return the width of the reserved interval in Hz."""
        return self.n_slots * self.grid.slot_width_hz


@dataclass(frozen=True)
class ConnectionSpec:
    """Describe one WDM connection and its physical allocation.

    The slot interval must lie in the grid and match the number of slots
    required by the bit rate, modulation, and roll-off. Launch power is
    the combined power of both polarizations in watts.
    """

    grid: FlexibleGrid
    connection_id: int
    first_slot: int
    n_slots: int
    bit_rate_bps: float
    modulation: str
    symbol_rate_hz: float
    roll_off: float
    launch_power_w: float

    def __post_init__(self) -> None:
        """Validate the connection and its slot allocation."""
        if (
            isinstance(cast("object", self.connection_id), bool)
            or not isinstance(cast("object", self.connection_id), int)
            or self.connection_id < 0
        ):
            msg = f"connection_id must be a non-negative integer, got {self.connection_id!r}."
            raise ValueError(msg)
        if (
            isinstance(cast("object", self.n_slots), bool)
            or not isinstance(cast("object", self.n_slots), int)
            or self.n_slots < 1
        ):
            msg = f"n_slots must be a positive integer, got {self.n_slots!r}."
            raise ValueError(msg)
        if isinstance(self.first_slot, bool) or not isinstance(self.first_slot, int):
            msg = f"first_slot must be an integer, got {self.first_slot!r}."
            raise ValueError(msg)  # noqa: TRY004 - preserve the allocation API
        if self.first_slot < 0 or self.first_slot + self.n_slots > self.grid.num_slots:
            msg = (
                f"Connection slot range [{self.first_slot}, {self.first_slot + self.n_slots}) "
                f"out of bounds for grid with {self.grid.num_slots} slots."
            )
            raise ValueError(msg)
        if self.bit_rate_bps <= 0:
            msg = f"bit_rate_bps must be positive, got {self.bit_rate_bps}."
            raise ValueError(msg)
        if self.symbol_rate_hz <= 0:
            msg = f"symbol_rate_hz must be positive, got {self.symbol_rate_hz}."
            raise ValueError(msg)
        if not (0 <= self.roll_off < 1):
            msg = f"roll_off must satisfy 0 <= roll_off < 1, got {self.roll_off}."
            raise ValueError(msg)
        if self.launch_power_w < 0:
            msg = f"launch_power_w must be non-negative, got {self.launch_power_w}."
            raise ValueError(msg)

        required_slots = slots_for_rate(
            self.bit_rate_bps,
            bits_per_symbol(self.modulation),
            self.roll_off,
            self.grid.slot_width_hz,
        )
        if self.n_slots != required_slots:
            msg = (
                f"n_slots={self.n_slots} is under-/over-provisioned for "
                f"{self.bit_rate_bps / 1e9:g} Gbps {self.modulation} (roll_off={self.roll_off}): "
                f"slots_for_rate requires {required_slots} slots of "
                f"{self.grid.slot_width_hz / 1e9:g} GHz."
            )
            raise ValueError(msg)

    @property
    def centre_hz(self) -> float:
        """Return the connection center frequency in Hz."""
        return self.grid.centre_hz(self.first_slot, self.n_slots)

    @property
    def occupied_bandwidth_hz(self) -> float:
        """Return the occupied bandwidth in Hz."""
        return self.n_slots * self.grid.slot_width_hz

    @property
    def is_occupied(self) -> bool:
        """Return whether the connection has nonzero launch power."""
        return self.launch_power_w > 0


ConnectionGeometry = ConnectionSpec | CarrierGeometry


class AllocatedLike(Protocol):
    """Describe the service allocation fields needed by the physical layer."""

    service_id: int
    bit_rate_gbps: float
    n_slots: int
    first_slot: int
    launch_power_w: float


def connection_spec_from_allocation(
    service: AllocatedLike,
    grid: FlexibleGrid,
    *,
    modulation: str = "QPSK",
    roll_off: float = 0.0,
    symbol_rate_hz: float | None = None,
    connection_id: int | None = None,
) -> ConnectionSpec:
    """Convert a service allocation to a physical connection specification.

    Parameters
    ----------
    service : AllocatedLike
        Service allocation supplying the slot interval, rate, and launch power.
    grid : FlexibleGrid
        Physical frequency grid.
    modulation : str, optional
        Modulation format.
    roll_off : float, optional
        RRC roll-off factor.
    symbol_rate_hz : float or None, optional
        Symbol rate in Hz; derived from the bit rate when omitted.
    connection_id : int or None, optional
        Physical connection identifier; defaults to the service identifier.

    Returns
    -------
    ConnectionSpec
        Validated physical connection.

    Raises
    ------
    ValueError
        If the allocation or physical parameters are invalid.
    """
    bits = bits_per_symbol(modulation)
    if not (0 <= roll_off < 1):
        msg = f"roll_off must satisfy 0 <= roll_off < 1, got {roll_off!r}."
        raise ValueError(msg)

    bit_rate_gbps = float(service.bit_rate_gbps)
    if bit_rate_gbps <= 0:
        msg = f"bit_rate_gbps must be positive, got {bit_rate_gbps!r}."
        raise ValueError(msg)

    n_slots = int(service.n_slots)
    first_slot = int(service.first_slot)
    if first_slot < 0 or first_slot + n_slots > grid.num_slots:
        msg = (
            f"Slot window [{first_slot}, {first_slot + n_slots}) is out of bounds for a grid of "
            f"{grid.num_slots} slots (service_id={service.service_id})."
        )
        raise ValueError(msg)

    if symbol_rate_hz is None:
        symbol_rate_hz = (bit_rate_gbps * 1e9 / bits) / (1.0 + roll_off)
    if symbol_rate_hz <= 0:
        msg = f"symbol_rate_hz must be positive, got {symbol_rate_hz!r}."
        raise ValueError(msg)

    if connection_id is None:
        connection_id = int(service.service_id)
    if connection_id < 0:
        msg = f"connection_id must be a non-negative integer, got {connection_id!r}."
        raise ValueError(msg)

    return ConnectionSpec(
        grid=grid,
        connection_id=int(connection_id),
        first_slot=first_slot,
        n_slots=n_slots,
        bit_rate_bps=bit_rate_gbps * 1e9,
        modulation=modulation,
        symbol_rate_hz=float(symbol_rate_hz),
        roll_off=float(roll_off),
        launch_power_w=float(service.launch_power_w),
    )
