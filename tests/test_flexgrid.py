"""灵活栅格数据模型与槽数推导的行为测试。"""

import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import (
    BITS_PER_SYMBOL,
    MODULATION_FORMATS,
    ConnectionSpec,
    FlexibleGrid,
    bits_per_symbol,
    slots_for_rate,
)

SLOT_WIDTH_HZ = 12.5e9
CHANNEL_SPACING_HZ = 50e9


def _grid(num_channels: int = 8) -> FlexibleGrid:
    """按遗留 50 GHz 栅格造一个等价的灵活栅格。"""
    return FlexibleGrid.from_legacy_grid(num_channels, CHANNEL_SPACING_HZ, 193.1e12)


def _connection(grid: FlexibleGrid, **overrides: object) -> ConnectionSpec:
    """造一条 100 Gbps QPSK 连接（占 4 槽 = 50 GHz）。"""
    fields: dict[str, object] = {
        "grid": grid,
        "connection_id": 0,
        "first_slot": 0,
        "n_slots": 4,
        "bit_rate_bps": 100e9,
        "modulation": "QPSK",
        "symbol_rate_hz": 32e9,
        "roll_off": 0.0,
        "launch_power_w": 1e-3,
    }
    fields.update(overrides)
    return ConnectionSpec(**fields)  # type: ignore[arg-type]


def test_bits_per_symbol_covers_every_format() -> None:
    """每符号比特数与格式表一致，未知格式显式报错。"""
    for name, bits in BITS_PER_SYMBOL.items():
        assert bits_per_symbol(name) == bits
    assert set(BITS_PER_SYMBOL) == set(MODULATION_FORMATS)
    with pytest.raises(ValueError, match="Unknown modulation format"):
        bits_per_symbol("32QAM")


def test_slots_for_rate_rounds_up_and_never_returns_zero() -> None:
    """槽数向上取整，且至少为 1。"""
    # 100 Gbps QPSK = 50 GBd，12.5 GHz 槽 -> 恰好 4 槽
    assert slots_for_rate(100e9, 2, 0.0, SLOT_WIDTH_HZ) == 4
    # 101 Gbps -> 4.04 槽，向上取整为 5
    assert slots_for_rate(101e9, 2, 0.0, SLOT_WIDTH_HZ) == 5
    # 滚降系数把所需带宽放大 (1 + roll_off)
    assert slots_for_rate(100e9, 2, 0.1, SLOT_WIDTH_HZ) == 5
    # 极低比特率仍占 1 槽
    assert slots_for_rate(1e6, 6, 0.0, SLOT_WIDTH_HZ) == 1


@pytest.mark.parametrize(
    ("bit_rate_bps", "bits_per_symbol_", "roll_off", "slot_width_hz"),
    [
        (0.0, 2, 0.0, SLOT_WIDTH_HZ),
        (-1.0, 2, 0.0, SLOT_WIDTH_HZ),
        (100e9, 0, 0.0, SLOT_WIDTH_HZ),
        (100e9, 2, -0.1, SLOT_WIDTH_HZ),
        (100e9, 2, 1.0, SLOT_WIDTH_HZ),
        (100e9, 2, 0.0, 0.0),
    ],
)
def test_slots_for_rate_rejects_invalid_arguments(
    bit_rate_bps: float, bits_per_symbol_: int, roll_off: float, slot_width_hz: float
) -> None:
    """越界参数一律报错，不做静默截断。"""
    with pytest.raises(ValueError):
        slots_for_rate(bit_rate_bps, bits_per_symbol_, roll_off, slot_width_hz)


def test_from_legacy_grid_is_geometry_equivalent() -> None:
    """固定 50 GHz 栅格等价于 12.5 GHz × 4 槽的灵活栅格。"""
    grid = _grid(num_channels=8)
    assert grid.slot_width_hz == SLOT_WIDTH_HZ
    assert grid.num_slots == 32
    assert grid.total_bandwidth_hz == 8 * CHANNEL_SPACING_HZ
    # 带中心即遗留 grid 的中心频率
    assert grid.band_center_hz == 193.1e12
    assert grid.band_low_edge_hz == 193.1e12 - 4 * CHANNEL_SPACING_HZ


def test_from_legacy_grid_rejects_non_positive_slots_per_channel() -> None:
    """每信道槽数必须 >= 1。"""
    with pytest.raises(ValueError):
        FlexibleGrid.from_legacy_grid(4, CHANNEL_SPACING_HZ, 193.1e12, slots_per_channel=0)


def test_connection_centre_is_interval_centre_not_lattice_centre() -> None:
    """4 槽连接的中心是区间中心（2 槽处），不是点阵中心（1.5 槽处）。

    用点阵中心会让整体偏移半槽；这里断言它精确落在首个信道中心上。
    """
    grid = _grid(num_channels=1)
    assert grid.num_slots == 4
    # 区间中心：0 + 4/2 = 2 槽 -> 正好是首信道中心
    assert grid.centre_hz(0, 4) == 193.1e12
    # 点阵中心会是 0 + (4-1)/2 = 1.5 槽，偏半槽 (6.25 GHz)
    assert grid.centre_hz(0, 4) - grid.slot_center_hz(1) == SLOT_WIDTH_HZ / 2


def test_connection_edges_match_slot_geometry() -> None:
    """连接带边缘与槽边缘一致，且带宽 = n_slots × 槽宽。"""
    grid = _grid(num_channels=4)
    low_hz, high_hz = grid.connection_edges_hz(4, 4)
    assert low_hz == grid.slot_low_edge_hz(4)
    assert high_hz == grid.slot_high_edge_hz(7)
    assert high_hz - low_hz == pytest.approx(4 * SLOT_WIDTH_HZ)
    assert grid.edges_are_inside(low_hz, high_hz)
    assert not grid.edges_are_inside(low_hz, grid.band_high_edge_hz + 1.0)


def test_slot_accessors_reject_out_of_bounds_and_non_integers() -> None:
    """槽索引越界报 IndexError，非整数报 TypeError。"""
    grid = _grid(num_channels=1)
    with pytest.raises(IndexError):
        grid.slot_low_edge_hz(4)
    with pytest.raises(IndexError):
        grid.slot_low_edge_hz(-1)
    with pytest.raises(TypeError):
        grid.slot_low_edge_hz(1.0)  # type: ignore[arg-type]
    with pytest.raises(IndexError):
        grid.centre_hz(1, 4)
    with pytest.raises(ValueError):
        grid.centre_hz(0, 0)


@pytest.mark.parametrize(
    ("slot_width_hz", "num_slots", "band_low_edge_hz"),
    [(0.0, 4, 193e12), (12.5e9, 0, 193e12), (12.5e9, True, 193e12), (12.5e9, 4, 0.0)],
)
def test_flexible_grid_enforces_invariants(slot_width_hz: float, num_slots: int, band_low_edge_hz: float) -> None:
    """槽宽、槽数、下边缘的不变式由构造期强制。"""
    with pytest.raises(ValueError):
        FlexibleGrid(slot_width_hz=slot_width_hz, num_slots=num_slots, band_low_edge_hz=band_low_edge_hz)


def test_connection_spec_derived_quantities() -> None:
    """连接的派生量与边界一致。"""
    grid = _grid(num_channels=8)
    conn = _connection(grid, first_slot=4)
    assert conn.centre_hz == grid.centre_hz(4, 4)
    assert conn.occupied_bandwidth_hz == pytest.approx(50e9)
    assert conn.is_occupied
    assert not _connection(grid, launch_power_w=0.0).is_occupied


def test_connection_spec_rejects_under_provisioned_slots() -> None:
    """槽数必须等于 slots_for_rate 的结果，不允许静默欠配或超配。"""
    grid = _grid()
    with pytest.raises(ValueError, match="under-/over-provisioned"):
        _connection(grid, n_slots=8)
    with pytest.raises(ValueError, match="under-/over-provisioned"):
        _connection(grid, n_slots=2)
    # 100 Gbps 64QAM 只需 2 槽；给 4 槽同样是超配
    with pytest.raises(ValueError, match="under-/over-provisioned"):
        _connection(grid, n_slots=4, modulation="64QAM", symbol_rate_hz=20e9)


def test_connection_spec_rejects_out_of_bounds_window() -> None:
    """槽窗口必须落在栅格内。"""
    grid = _grid(num_channels=1)  # 4 槽
    with pytest.raises(ValueError, match="out of bounds"):
        _connection(grid, first_slot=1)
    with pytest.raises(ValueError, match="out of bounds"):
        _connection(grid, first_slot=-1)


@pytest.mark.parametrize("bad_power", [-1e-3])
def test_connection_spec_rejects_negative_launch_power(bad_power: float) -> None:
    """发射功率不得为负。"""
    with pytest.raises(ValueError, match="launch_power_w"):
        _connection(_grid(), launch_power_w=bad_power)


def test_connection_spec_rejects_invalid_scalars() -> None:
    """连接 id、符号率、滚降、比特率的不变式。"""
    grid = _grid()
    with pytest.raises(ValueError, match="connection_id"):
        _connection(grid, connection_id=-1)
    with pytest.raises(ValueError, match="symbol_rate_hz must be positive"):
        _connection(grid, symbol_rate_hz=0.0)
    with pytest.raises(ValueError, match="bit_rate_bps must be positive"):
        _connection(grid, bit_rate_bps=0.0)
    with pytest.raises(ValueError, match="roll_off"):
        _connection(grid, roll_off=1.0)
    with pytest.raises(ValueError, match="n_slots must be a positive integer"):
        _connection(grid, n_slots=0)


def test_connection_bandwidth_follows_slot_count() -> None:
    """占用带宽由槽数决定，与比特率无关。"""
    grid = FlexibleGrid.from_legacy_grid(4, CHANNEL_SPACING_HZ, 193.1e12, slots_per_channel=4)
    widths = [
        ConnectionSpec(
            grid=grid,
            connection_id=i,
            first_slot=i,
            n_slots=n,
            bit_rate_bps=2.0 * n * SLOT_WIDTH_HZ,
            modulation="QPSK",
            symbol_rate_hz=n * SLOT_WIDTH_HZ,
            roll_off=0.0,
            launch_power_w=1e-3,
        ).occupied_bandwidth_hz
        for i, n in enumerate((1, 2, 3, 4))
    ]
    assert widths == [pytest.approx(n * SLOT_WIDTH_HZ) for n in (1, 2, 3, 4)]
    assert np.all(np.diff(widths) > 0)
