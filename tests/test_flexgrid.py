"""Key checks for flexible grid allocations."""

from types import SimpleNamespace

import pytest

from network_nlo_eval.core.flexgrid import (
    FlexibleGrid,
    bits_per_symbol,
    connection_spec_from_allocation,
    slots_for_rate,
)


def test_legacy_grid_centers_and_edges() -> None:
    grid = FlexibleGrid.from_legacy_grid(3, 50e9, 193.1e12)
    assert grid.num_slots == 12
    assert grid.centre_hz(4, 4) == 193.1e12
    assert grid.connection_edges_hz(4, 4) == (193.075e12, 193.125e12)
    assert grid.slot_center_hz(0) == grid.band_low_edge_hz + 6.25e9
    with pytest.raises(IndexError):
        grid.centre_hz(11, 2)


def test_rate_and_service_allocation() -> None:
    grid = FlexibleGrid(12.5e9, 20, 190e12)
    service = SimpleNamespace(service_id=1, bit_rate_gbps=100, n_slots=4, first_slot=2, launch_power_w=0.001)
    assert bits_per_symbol("QPSK") == 2
    assert slots_for_rate(100e9, 2, 0.0, grid.slot_width_hz) == 4
    connection = connection_spec_from_allocation(service, grid)
    assert connection.centre_hz == grid.centre_hz(2, 4)
    assert connection.occupied_bandwidth_hz == 50e9
    assert connection.is_occupied
    service.n_slots = 3
    with pytest.raises(ValueError, match="under-/over-provisioned"):
        connection_spec_from_allocation(service, grid)


def test_invalid_modulation_and_grid() -> None:
    with pytest.raises(ValueError, match="Unknown modulation"):
        bits_per_symbol("unknown")
    with pytest.raises(ValueError):
        FlexibleGrid(12.5e9, 0, 190e12)
