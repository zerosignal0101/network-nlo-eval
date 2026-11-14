"""Flexible frequency grids and contiguous connection allocations.

Slots occupy half-open intervals. Connection centers use n_slots / 2,
the center of the occupied interval, avoiding a half-slot offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

__all__ = [
    "MODULATION_FORMATS",
    "BITS_PER_SYMBOL",
    "bits_per_symbol",
    "FlexibleGrid",
    "ConnectionSpec",
    "AllocatedLike",
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
        raise ValueError(
            f"Unknown modulation format {modulation!r}; expected one of {list(MODULATION_FORMATS)}."
        ) from None


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
        raise ValueError(f"bit_rate_bps must be positive, got {bit_rate_bps}.")
    if bits_per_symbol < 1:
        raise ValueError(f"bits_per_symbol must be >= 1, got {bits_per_symbol}.")
    if not (0 <= roll_off < 1):
        raise ValueError(f"roll_off must satisfy 0 <= roll_off < 1, got {roll_off}.")
    if slot_width_hz <= 0:
        raise ValueError(f"slot_width_hz must be positive, got {slot_width_hz}.")

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
            raise ValueError(f"slot_width_hz must be positive, got {self.slot_width_hz}.")
        if isinstance(self.num_slots, bool) or not isinstance(self.num_slots, int) or self.num_slots <= 0:
            raise ValueError(f"num_slots must be a positive integer, got {self.num_slots!r}.")
        if self.band_low_edge_hz <= 0:
            raise ValueError(
                f"band_low_edge_hz must be positive (optical frequencies are positive), got {self.band_low_edge_hz}."
            )

    
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
            raise TypeError(f"slot must be an integer, got {type(slot).__name__}.")
        if not (0 <= slot < self.num_slots):
            raise IndexError(f"Slot index {slot} out of bounds for {self.num_slots} slots.")
        return slot

    def slot_low_edge_hz(self, slot: int) -> float:
        """Return the lower edge of a slot in Hz."""
        return self.band_low_edge_hz + self._check_slot(slot) * self.slot_width_hz

    def slot_high_edge_hz(self, slot: int) -> float:
        """Return the exclusive upper edge of a slot in Hz."""
        return self.band_low_edge_hz + (self._check_slot(slot) + 1) * self.slot_width_hz

    def slot_center_hz(self, slot: int) -> float:
        """Return the center frequency of a slot in Hz."""
        return self.band_low_edge_hz + (self._check_slot(slot) + 0.5) * self.slot_width_hz

    
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
            raise ValueError(f"n_slots must be positive, got {n_slots}.")
        if first_slot < 0 or first_slot + n_slots > self.num_slots:
            raise IndexError(
                f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for {self.num_slots} slots."
            )
        return self.band_low_edge_hz + (first_slot + n_slots / 2) * self.slot_width_hz

    def connection_edges_hz(self, first_slot: int, n_slots: int) -> tuple[float, float]:
        """Return the lower and exclusive upper connection edges in Hz."""
        if n_slots <= 0:
            raise ValueError(f"n_slots must be positive, got {n_slots}.")
        if first_slot < 0 or first_slot + n_slots > self.num_slots:
            raise IndexError(
                f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for {self.num_slots} slots."
            )
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
            raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")
        slot_width = channel_spacing_hz / slots_per_channel
        
        first_channel_centre = center_frequency_hz - (num_channels - 1) * channel_spacing_hz / 2
        return cls(
            slot_width_hz=slot_width,
            num_slots=num_channels * slots_per_channel,
            band_low_edge_hz=first_channel_centre - channel_spacing_hz / 2,
        )


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
        if isinstance(self.connection_id, bool) or not isinstance(self.connection_id, int) or self.connection_id < 0:
            raise ValueError(f"connection_id must be a non-negative integer, got {self.connection_id!r}.")
        if isinstance(self.n_slots, bool) or not isinstance(self.n_slots, int) or self.n_slots < 1:
            raise ValueError(f"n_slots must be a positive integer, got {self.n_slots!r}.")
        if isinstance(self.first_slot, bool) or not isinstance(self.first_slot, int):
            raise ValueError(f"first_slot must be an integer, got {self.first_slot!r}.")
        if self.first_slot < 0 or self.first_slot + self.n_slots > self.grid.num_slots:
            raise ValueError(
                f"Connection slot range [{self.first_slot}, {self.first_slot + self.n_slots}) "
                f"out of bounds for grid with {self.grid.num_slots} slots."
            )
        if self.bit_rate_bps <= 0:
            raise ValueError(f"bit_rate_bps must be positive, got {self.bit_rate_bps}.")
        if self.symbol_rate_hz <= 0:
            raise ValueError(f"symbol_rate_hz must be positive, got {self.symbol_rate_hz}.")
        if not (0 <= self.roll_off < 1):
            raise ValueError(f"roll_off must satisfy 0 <= roll_off < 1, got {self.roll_off}.")
        if self.launch_power_w < 0:
            raise ValueError(f"launch_power_w must be non-negative, got {self.launch_power_w}.")

        required_slots = slots_for_rate(
            self.bit_rate_bps,
            bits_per_symbol(self.modulation),
            self.roll_off,
            self.grid.slot_width_hz,
        )
        if self.n_slots != required_slots:
            raise ValueError(
                f"n_slots={self.n_slots} is under-/over-provisioned for "
                f"{self.bit_rate_bps / 1e9:g} Gbps {self.modulation} (roll_off={self.roll_off}): "
                f"slots_for_rate requires {required_slots} slots of "
                f"{self.grid.slot_width_hz / 1e9:g} GHz."
            )

    
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
        raise ValueError(f"roll_off must satisfy 0 <= roll_off < 1, got {roll_off!r}.")

    bit_rate_gbps = float(service.bit_rate_gbps)
    if bit_rate_gbps <= 0:
        raise ValueError(f"bit_rate_gbps must be positive, got {bit_rate_gbps!r}.")

    n_slots = int(service.n_slots)
    first_slot = int(service.first_slot)
    if first_slot < 0 or first_slot + n_slots > grid.num_slots:
        raise ValueError(
            f"Slot window [{first_slot}, {first_slot + n_slots}) is out of bounds for a grid of "
            f"{grid.num_slots} slots (service_id={service.service_id})."
        )

    if symbol_rate_hz is None:
        symbol_rate_hz = (bit_rate_gbps * 1e9 / bits) / (1.0 + roll_off)
    if symbol_rate_hz <= 0:
        raise ValueError(f"symbol_rate_hz must be positive, got {symbol_rate_hz!r}.")

    if connection_id is None:
        connection_id = int(service.service_id)
    if connection_id < 0:
        raise ValueError(f"connection_id must be a non-negative integer, got {connection_id!r}.")

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
