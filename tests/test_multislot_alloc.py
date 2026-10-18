"""多槽分配的槽位状态、窗口搜索与分配器行为测试。"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import LinkState, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.allocators import (
    KSPFirstFitAllocator,
    KSPFirstFitFlexAllocator,
    first_fit_contiguous_windows,
)
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import ServiceRequest

SLOT_WIDTH_HZ = 12.5e9
SLOTS_PER_CHANNEL = 4
NUM_CHANNELS = 8


def _stack(roadm_config: ROADMConfig | None = None):
    """0-1-2-3 四节点线形拓扑上的完整分配栈。"""
    graph = nx.Graph()
    for node in range(4):
        graph.add_node(node)
    for u in range(3):
        graph.add_edge(u, u + 1, weight=80.0)

    fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), roadm_config or ROADMConfig())
    grid = SpectrumGrid(NUM_CHANNELS, 50e9, 193.1e12)
    evaluator = MultiBandISRSGN(grid, fiber, 32e9)
    validator = QoTValidator(evaluator, topology, grid, fixed_span_length_km=80.0)
    path_cache = PathCache(topology)
    state = NetworkState(topology, grid, slots_per_channel=SLOTS_PER_CHANNEL)
    return topology, grid, validator, path_cache, state


def _flex_allocator(path_cache, validator, grid, topology) -> KSPFirstFitFlexAllocator:
    return KSPFirstFitFlexAllocator(
        path_cache=path_cache,
        qot_validator=validator,
        spectrum_grid=grid,
        network_topology=topology,
        slots_per_channel=SLOTS_PER_CHANNEL,
    )


def _request(service_id: int = 0, n_slots: int = 1) -> ServiceRequest:
    return ServiceRequest(
        service_id=service_id,
        source_id=0,
        destination_id=3,
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=10.0,
        launch_power_w=1e-3,
        n_slots=n_slots,
    )


def _link(channels: int = 8, slots_per_channel: int = SLOTS_PER_CHANNEL) -> LinkState:
    return LinkState(
        link_key=LinkKey((0, 1)),
        length_m=80e3,
        occupied_channels=np.zeros(channels, dtype=bool),
        launch_power_profile_w=np.zeros(channels, dtype=np.float64),
        allocated_service_ids=np.full(channels, -1, dtype=np.int_),
        slots_per_channel=slots_per_channel,
    )


# ---------------------------------------------------------------------------
# 槽级状态
# ---------------------------------------------------------------------------
def test_slot_state_is_sized_from_channels_and_conversion() -> None:
    """槽数组长度 = 信道数 × 每信道槽数。"""
    link = _link(channels=8)
    assert link.occupied_slots is not None
    assert link.occupied_slots.shape == (8 * SLOTS_PER_CHANNEL,)
    assert not link.occupied_slots.any()
    assert (link.allocated_service_ids_slots == -1).all()


def test_allocate_slot_range_marks_window_and_channel_alignment() -> None:
    """槽窗口逐槽占用，并按信道对齐补齐；功率只写锚信道。"""
    link = _link()
    link.allocate_slot_range(first_slot=5, n_slots=2, launch_power_w=2e-3, service_id=7)

    assert link.occupied_slots[5:7].all()
    # 槽 5、6 同属信道 1，该信道其余子槽一并占用；信道 2 不受影响
    assert link.occupied_slots[4:8].all()
    assert not link.occupied_slots[8:12].any()
    # 功率只注入锚信道 5 // 4 = 1
    assert link.occupied_channels[1]
    assert link.launch_power_profile_w[1] == 2e-3
    assert link.allocated_service_ids[1] == 7
    assert not link.occupied_channels[0]


def test_allocate_slot_range_rejects_invalid_or_occupied_window() -> None:
    """非法槽数、越界与重复占用都报错。"""
    link = _link()
    with pytest.raises(ValueError, match="n_slots"):
        link.allocate_slot_range(0, 0, 1e-3, 1)
    with pytest.raises(ValueError, match="out of bounds"):
        link.allocate_slot_range(31, 2, 1e-3, 1)
    with pytest.raises(ValueError, match="out of bounds"):
        link.allocate_slot_range(-1, 1, 1e-3, 1)

    link.allocate_slot_range(4, 4, 1e-3, 1)
    with pytest.raises(ValueError, match="already occupied"):
        link.allocate_slot_range(4, 4, 1e-3, 2)


def test_release_slot_range_clears_window_and_is_idempotent() -> None:
    """释放回收槽窗口；重复释放或越界释放静默返回。"""
    link = _link()
    link.allocate_slot_range(4, 4, 1e-3, 1)
    link.release_slot_range(4, 4)

    assert not link.occupied_slots.any()
    assert not link.occupied_channels[1]
    assert link.launch_power_profile_w[1] == 0.0

    link.release_slot_range(4, 4)  # 重复释放
    link.release_slot_range(0, 0)  # 非法槽数
    link.release_slot_range(31, 4)  # 越界
    assert not link.occupied_slots.any()


def test_release_does_not_clear_neighbour_channel() -> None:
    """释放只清自己覆盖的信道，不动相邻信道。"""
    link = _link()
    link.allocate_slot_range(0, 4, 1e-3, 1)
    link.allocate_slot_range(4, 4, 1e-3, 2)

    link.release_slot_range(0, 4)
    assert not link.occupied_slots[0:4].any()
    assert link.occupied_slots[4:8].all()
    assert not link.occupied_channels[0]
    assert link.occupied_channels[1]


def test_occupied_slots_on_path_ors_across_links() -> None:
    """路径占用是各链路逐槽 OR。"""
    _, _, _, _, state = _stack()
    state.get_link_state(0, 1).allocate_slot_range(0, 4, 1e-3, 1)
    state.get_link_state(1, 2).allocate_slot_range(8, 4, 1e-3, 2)

    occupied = state.occupied_slots_on_path([0, 1, 2])
    assert occupied[0:4].all()
    assert occupied[8:12].all()
    assert not occupied[4:8].any()

    with pytest.raises(ValueError, match="at least two nodes"):
        state.occupied_slots_on_path([0])


# ---------------------------------------------------------------------------
# 连续窗口搜索
# ---------------------------------------------------------------------------
def test_first_fit_windows_require_contiguous_free_run() -> None:
    """跨空洞的窗口不得入选。"""
    free = np.array([True, False, True, True, True, False, True, True])
    assert first_fit_contiguous_windows(free, 3, SLOTS_PER_CHANNEL) == [2]


def test_first_fit_windows_dedupe_by_anchor_channel() -> None:
    """同一锚信道的多个起点只保留第一个。"""
    free = np.ones(16, dtype=bool)
    assert first_fit_contiguous_windows(free, 1, SLOTS_PER_CHANNEL) == [0, 4, 8, 12]


def test_first_fit_windows_return_empty_when_impossible() -> None:
    """无可行窗口时返回空列表。"""
    assert first_fit_contiguous_windows(np.array([True, False, True, False]), 3, SLOTS_PER_CHANNEL) == []
    assert first_fit_contiguous_windows(np.zeros(4, dtype=bool), 5, SLOTS_PER_CHANNEL) == []


@pytest.mark.parametrize(("n_slots", "slots_per_channel"), [(0, 4), (1, 0)])
def test_first_fit_windows_reject_invalid_arguments(n_slots: int, slots_per_channel: int) -> None:
    """非法窗口宽度或折算率报错。"""
    with pytest.raises(ValueError):
        first_fit_contiguous_windows(np.ones(8, dtype=bool), n_slots, slots_per_channel)


# ---------------------------------------------------------------------------
# 端到端分配
# ---------------------------------------------------------------------------
def test_multislot_allocation_occupies_contiguous_window() -> None:
    """N 槽请求分配后，该窗口逐槽占用。"""
    topology, grid, validator, path_cache, state = _stack()
    allocator = _flex_allocator(path_cache, validator, grid, topology)

    success, service = allocator.allocate(_request(n_slots=3), state)
    assert success and service is not None
    assert service.n_slots == 3
    state.allocate_service(service)

    link = state.get_link_state(service.path[0], service.path[1])
    low, high = service.slot_interval
    assert high - low == 3
    assert link.occupied_slots[low:high].all()


def test_multislot_requests_get_channel_aligned_first_fit() -> None:
    """连续请求按 first_slot 升序排布，且起点信道对齐。"""
    topology, grid, validator, path_cache, state = _stack()
    allocator = _flex_allocator(path_cache, validator, grid, topology)

    placements = []
    for service_id, n_slots in enumerate((1, 2, 3)):
        success, service = allocator.allocate(_request(service_id, n_slots), state)
        assert success and service is not None
        placements.append((service.first_slot, service.n_slots))
        state.allocate_service(service)

    assert placements == [(0, 1), (4, 2), (8, 3)]
    assert all(first_slot % SLOTS_PER_CHANNEL == 0 for first_slot, _ in placements)


def test_release_reclaims_window_and_allows_reuse() -> None:
    """释放后同一窗口可被再次分配。"""
    topology, grid, validator, path_cache, state = _stack()
    allocator = _flex_allocator(path_cache, validator, grid, topology)

    _, service = allocator.allocate(_request(n_slots=3), state)
    assert service is not None
    state.allocate_service(service)
    first_slot_before = service.first_slot

    allocator.release(service, state)
    link = state.get_link_state(service.path[0], service.path[1])
    low, high = service.slot_interval
    assert not link.occupied_slots[low:high].any()

    success, again = allocator.allocate(_request(n_slots=3), state)
    assert success and again is not None
    assert again.first_slot == first_slot_before


def test_degenerate_single_slot_matches_legacy_allocator() -> None:
    """全 n_slots=1 且槽宽折算率 4 时，与遗留分配器选出同一信道。"""
    topology, grid, validator, path_cache, state = _stack()
    flex = _flex_allocator(path_cache, validator, grid, topology)
    legacy = KSPFirstFitAllocator(path_cache=path_cache, qot_validator=validator, spectrum_grid=grid)

    success_flex, flex_service = flex.allocate(_request(service_id=0, n_slots=1), state)
    assert success_flex and flex_service is not None

    legacy_state = NetworkState(topology, grid)
    success_legacy, legacy_service = legacy.allocate(_request(service_id=0), legacy_state)
    assert success_legacy and legacy_service is not None
    assert flex_service.wavelength == legacy_service.wavelength


def test_blocked_slot_makes_overlapping_window_infeasible() -> None:
    """ROADM 阻断槽与窗口重叠则该窗口不可行。"""
    topology, grid, validator, path_cache, state = _stack(ROADMConfig(blocked_slots=[5]))
    allocator = _flex_allocator(path_cache, validator, grid, topology)
    assert allocator.blocked_slots_on_path([0, 1, 3]) == {5}

    # 4 槽窗口 [0,4) 不含槽 5
    success, service = allocator.allocate(_request(service_id=0, n_slots=4), state)
    assert success and service is not None
    assert service.first_slot == 0

    # 6 槽窗口起点 0..5 都含槽 5，首个可行窗口从 6 开始
    success, service = allocator.allocate(_request(service_id=1, n_slots=6), state)
    assert success and service is not None
    assert service.first_slot == 6


def test_blocked_channel_expands_to_slot_interval() -> None:
    """blocked_channels 按槽宽折算率展开为槽区间。"""
    topology, grid, validator, path_cache, state = _stack(ROADMConfig(blocked_channels=[0]))
    allocator = _flex_allocator(path_cache, validator, grid, topology)
    assert allocator.blocked_slots_on_path([0, 1, 3]) == {0, 1, 2, 3}

    success, service = allocator.allocate(_request(n_slots=4), state)
    assert success and service is not None
    assert service.first_slot == 4


def test_qot_validator_indexes_connections_in_per_connection_mode() -> None:
    """逐连接模式下节点数是连接数，频率与带宽按连接取，而非按固定栅格信道。"""
    grid = FlexibleGrid.from_legacy_grid(4, 50e9, 193.1e12, slots_per_channel=SLOTS_PER_CHANNEL)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=index,
            first_slot=index * SLOTS_PER_CHANNEL,
            n_slots=SLOTS_PER_CHANNEL,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=32e9,
            roll_off=0.0,
            launch_power_w=1e-3,
        )
        for index in range(4)
    ]
    fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)
    evaluator = MultiBandISRSGN(grid, fiber, 32e9, connections=connections)
    topology, _, _, _, _ = _stack()
    validator = QoTValidator(evaluator, topology, SpectrumGrid(4, 50e9, 193.1e12))

    assert validator.num_nodes == len(connections)
    for index, connection in enumerate(connections):
        frequency_hz, bandwidth_hz = validator._node_frequency_and_bandwidth(index)
        assert frequency_hz == connection.centre_hz
        assert bandwidth_hz == connection.symbol_rate_hz


def test_qot_validator_blocks_by_slot_overlap_in_per_connection_mode() -> None:
    """逐连接模式按槽区间重叠判定阻断。"""
    grid = FlexibleGrid.from_legacy_grid(2, 50e9, 193.1e12, slots_per_channel=SLOTS_PER_CHANNEL)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=index,
            first_slot=index * SLOTS_PER_CHANNEL,
            n_slots=SLOTS_PER_CHANNEL,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=32e9,
            roll_off=0.0,
            launch_power_w=1e-3,
        )
        for index in range(2)
    ]
    fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)
    evaluator = MultiBandISRSGN(grid, fiber, 32e9, connections=connections, slots_per_channel=SLOTS_PER_CHANNEL)
    topology, _, _, _, _ = _stack()
    validator = QoTValidator(evaluator, topology, SpectrumGrid(2, 50e9, 193.1e12))

    assert validator._is_blocked(ROADMConfig(blocked_slots=[0]), 0)
    assert not validator._is_blocked(ROADMConfig(blocked_slots=[0]), 1)
