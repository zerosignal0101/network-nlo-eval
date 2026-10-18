"""ISRS-GN 评估器的逐信道带宽行为测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

SLOT_WIDTH_HZ = 12.5e9
CHANNEL_SPACING_HZ = 50e9


def _fiber() -> FiberSpanConfig:
    return FiberSpanConfig(length_km=80.0)


def _path() -> MultiSpanOpticalPath:
    return MultiSpanOpticalPath([_fiber()], [EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)])


def _connections(slots_per_conn: tuple[int, ...]) -> tuple[FlexibleGrid, list[ConnectionSpec]]:
    """按给定槽数依次排布一组连接，栅格宽度恰好容纳它们。"""
    total_slots = sum(slots_per_conn)
    num_channels = -(-total_slots // 4)  # ceil，保证栅格够宽
    grid = FlexibleGrid.from_legacy_grid(num_channels, CHANNEL_SPACING_HZ, 193.1e12, slots_per_channel=4)
    conns: list[ConnectionSpec] = []
    cursor = 0
    for index, n_slots in enumerate(slots_per_conn):
        conns.append(
            ConnectionSpec(
                grid=grid,
                connection_id=index,
                first_slot=cursor,
                n_slots=n_slots,
                bit_rate_bps=2.0 * n_slots * SLOT_WIDTH_HZ,
                modulation="QPSK",
                symbol_rate_hz=float(n_slots * SLOT_WIDTH_HZ),
                roll_off=0.0,
                launch_power_w=1e-3,
            )
        )
        cursor += n_slots
    return grid, conns


def test_legacy_bandwidth_is_the_symbol_rate_scalar() -> None:
    """遗留模式下 B_i 恒等于名义符号率，逐位相同。"""
    grid = SpectrumGrid(4, CHANNEL_SPACING_HZ, 193.1e12)
    evaluator = MultiBandISRSGN(grid, _fiber(), 32e9)
    bandwidth = evaluator.channel_bandwidth_hz_channels
    assert np.array_equal(bandwidth, np.full(4, 32e9))
    assert np.array_equal(bandwidth.view(np.int64), np.full(4, 32e9, dtype=np.float64).view(np.int64))


def test_per_connection_bandwidth_follows_slot_count() -> None:
    """``occupied`` 口径下 B_i 逐连接不同，等于占用带宽。"""
    grid, conns = _connections((1, 2, 3, 4))
    evaluator = MultiBandISRSGN(grid, _fiber(), 50e9, connections=conns, channel_bandwidth_source="occupied")
    assert evaluator.n_nodes == 4
    assert list(evaluator.channel_bandwidth_hz_channels) == [12.5e9, 25e9, 37.5e9, 50e9]


def test_bandwidth_sources_differ_when_symbol_rate_is_not_occupied_width() -> None:
    """两条口径取的是 ConnectionSpec 的不同字段。"""
    grid, conns = _connections((4,))
    narrow_symbol_rate = [replace(conns[0], symbol_rate_hz=25e9)]
    occupied = MultiBandISRSGN(
        grid, _fiber(), 50e9, connections=narrow_symbol_rate, channel_bandwidth_source="occupied"
    )
    symbol_rate = MultiBandISRSGN(
        grid, _fiber(), 50e9, connections=narrow_symbol_rate, channel_bandwidth_source="symbol_rate"
    )
    assert float(occupied.channel_bandwidth_hz_channels[0]) == 50e9
    assert float(symbol_rate.channel_bandwidth_hz_channels[0]) == 25e9


def test_invalid_mode_is_rejected() -> None:
    """非法口径显式报错，不静默取默认值。"""
    grid = SpectrumGrid(4, CHANNEL_SPACING_HZ, 193.1e12)
    with pytest.raises(ValueError, match="channel_bandwidth_source"):
        MultiBandISRSGN(grid, _fiber(), 50e9, channel_bandwidth_source="occupied_bw")
    with pytest.raises(ValueError, match="power_width_mode"):
        MultiBandISRSGN(grid, _fiber(), 50e9, power_width_mode="constant_psd")


def test_fixed_psd_scale_is_exactly_one_for_uniform_bandwidth() -> None:
    """带宽均匀时 fixed_psd 的缩放因子恰为 IEEE 1.0，不引入任何数值扰动。"""
    grid = SpectrumGrid(4, CHANNEL_SPACING_HZ, 193.1e12)
    evaluator = MultiBandISRSGN(grid, _fiber(), 50e9, power_width_mode="fixed_psd")
    scale = evaluator._power_width_scale()
    ones = np.ones(4, dtype=np.float64)
    assert np.array_equal(scale.view(np.int64), ones.view(np.int64))
    # fixed_power 口径恒为全 1
    fixed_power = MultiBandISRSGN(grid, _fiber(), 50e9, power_width_mode="fixed_power")
    assert np.array_equal(fixed_power._power_width_scale(), ones)


def test_flexible_grid_degrades_to_legacy_bitwise() -> None:
    """同宽连接（槽宽 = 符号率 = 50 GHz）与遗留栅格逐位一致。"""
    legacy_grid = SpectrumGrid(4, CHANNEL_SPACING_HZ, 193.1e12)
    legacy = MultiBandISRSGN(legacy_grid, _fiber(), 50e9)
    legacy_result = legacy.evaluate_path(_path(), np.full(4, 1e-3))

    flex_grid, conns = _connections((4, 4, 4, 4))
    flex = MultiBandISRSGN(flex_grid, _fiber(), 50e9, connections=conns, channel_bandwidth_source="occupied")
    flex_result = flex.evaluate_path(_path(), np.full(4, 1e-3))

    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        a = getattr(legacy_result, name)
        b = getattr(flex_result, name)
        assert np.array_equal(a.view(np.int64), b.view(np.int64)), f"{name} 不逐位相同"


def test_wide_connection_has_lower_snr_than_narrow_one() -> None:
    """固定总功率下宽连接 PSD 更低，SNR 必须更低。"""
    snrs = {}
    for slots in (1, 8):
        grid, conns = _connections((slots,))
        evaluator = MultiBandISRSGN(grid, _fiber(), 50e9, connections=conns, channel_bandwidth_source="occupied")
        result = evaluator.evaluate_path(_path(), np.array([1e-3]))
        snrs[slots] = float(result.snr_db[0])
    assert snrs[1] > snrs[8]
    assert abs(snrs[1] - snrs[8]) > 1e-3


def test_power_width_modes_differ_for_heterogeneous_bandwidth() -> None:
    """两种功率口径在异构带宽下给出不同结果。"""
    grid, conns = _connections((1, 8))
    out = {}
    for mode in ("fixed_power", "fixed_psd"):
        evaluator = MultiBandISRSGN(
            grid, _fiber(), 50e9, connections=conns, channel_bandwidth_source="occupied", power_width_mode=mode
        )
        out[mode] = evaluator.evaluate_path(_path(), np.full(2, 1e-3)).snr_db.copy()
    assert not np.array_equal(out["fixed_power"], out["fixed_psd"])
    # fixed_psd 下 PSD 恒定，宽窄连接的 SNR 差距应小于 fixed_power
    assert abs(out["fixed_psd"][0] - out["fixed_psd"][1]) < abs(out["fixed_power"][0] - out["fixed_power"][1])


def test_roadm_blocks_node_by_slot_overlap() -> None:
    """逐连接模式下按槽区间重叠判定阻断。"""
    grid, conns = _connections((4, 4))
    evaluator = MultiBandISRSGN(grid, _fiber(), 50e9, connections=conns, slots_per_channel=4)
    # 连接 0 占槽 [0,4)，连接 1 占槽 [4,8)
    assert evaluator.roadm_blocks_node(ROADMConfig(blocked_slots=[0]), 0)
    assert not evaluator.roadm_blocks_node(ROADMConfig(blocked_slots=[0]), 1)
    # 阻断信道 1 展开为槽 [4,8)，只影响连接 1
    assert evaluator.roadm_blocks_node(ROADMConfig(blocked_channels=[1]), 1)
    assert not evaluator.roadm_blocks_node(ROADMConfig(blocked_channels=[1]), 0)


def test_roadm_blocked_channels_require_slots_per_channel() -> None:
    """缺少槽宽折算率时直接报错，不做取整猜测。"""
    grid, conns = _connections((4,))
    evaluator = MultiBandISRSGN(grid, _fiber(), 50e9, connections=conns)
    with pytest.raises(ValueError, match="slots_per_channel"):
        evaluator.roadm_blocks_node(ROADMConfig(blocked_channels=[0]), 0)


def test_blocked_connection_is_zeroed_by_roadm() -> None:
    """被阻断的连接功率清零，SNR 为 -inf。"""
    grid, conns = _connections((4,))
    evaluator = MultiBandISRSGN(grid, _fiber(), 50e9, connections=conns, slots_per_channel=4)
    path = MultiSpanOpticalPath(
        [_fiber()],
        [EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)],
        roadm_configs=[ROADMConfig(blocked_slots=[0])],
    )
    result = evaluator.evaluate_path(path, np.array([1e-3]))
    assert result.signal_power_w[0] == 0.0
    assert result.snr_db[0] == -np.inf
