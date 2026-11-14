"""Slot-level occupancy helpers for flexible frequency grids."""

from __future__ import annotations

__all__ = ["connection_overlaps_slots"]


def connection_overlaps_slots(first_slot: int, n_slots: int, blocked: set[int]) -> bool:
    """Check whether a connection overlaps any blocked slot.

    Parameters
    ----------
    first_slot : int
        Zero-based index of the first occupied slot.
    n_slots : int
        Number of occupied slots.
    blocked : set of int
        Indices of blocked slots.

    Returns
    -------
    bool
        Whether a blocked slot lies in the half-open connection interval.
    """
    if not blocked:
        return False
    window = range(first_slot, first_slot + n_slots)
    return any(slot in blocked for slot in window)
