#!/usr/bin/env python3
"""WP-M1 多槽 RWA 的**共享场景构造器**。

判据工具（``ms_alloc_e2e`` / ``ms_alloc_degenerate`` / ``ms_alloc_cost``）与
``tests/test_wp_m1_multislot.py`` 全部从这里取材，避免出现"测试里一套拓扑、
证据里另一套拓扑"这种自说自话。

索引空间约定（与 `ode3_flexgrid_scenarios.build_state_grid` 同源）
--------------------------------------------------------------------
物理面仍是**遗留信道粒度**：``MultiBandISRSGN`` 收 ``SpectrumGrid``，
``NetworkState`` 的 ``launch_power_profile_w`` 长度仍是 ``num_channels``，
入纤功率只写在锚信道 ``first_slot // slots_per_channel`` 上。
**槽只是分配层的真源**——决定窗口连续性、冲突与释放区间。
这样 ``models/numba_kernels.py`` 保持零 diff（M-J5）。
"""
from __future__ import annotations

from dataclasses import dataclass

#: 50 GHz 信道 / 12.5 GHz 槽 = 4（WP-M1 §2 步骤 1 要求记录的关系）
SLOTS_PER_CHANNEL = 4

NUM_CHANNELS = 40
CHANNEL_SPACING_HZ = 50e9
CENTER_FREQUENCY_HZ = 193.1e12

LAUNCH_POWER_W = 1e-3  # 0 dBm
#: 统一取一个宽门限，让分配成败由**频谱连续性**决定，而不是被 QoT 噪声淹没。
#: QoT 仍然逐次真算（verify_allocation 不跳过），只是门限不成为瓶颈。
SNR_REQUIREMENT_DB = 8.0

HOP_LENGTH_KM = 80.0
MAX_KSP_PATHS = 5


@dataclass(frozen=True)
class ServicePlan:
    """一个待生成的业务（全部显式给出，不依赖随机流，保证可复算）。"""

    service_id: int
    source_id: int
    destination_id: int
    arrival_time: float
    departure_time: float
    bit_rate_gbps: float
    n_slots: int


def build_topology(
    *,
    hop_length_km: float = HOP_LENGTH_KM,
    roadm_config=None,
    num_hops: int = 3,
):
    """构造线形拓扑 ``0 -- 1 -- ... -- num_hops``，每跳 ``hop_length_km``。

    Args:
        hop_length_km: 每跳长度 (km)。
        roadm_config: 可选的 ``ROADMConfig``（用于阻断按槽区间生效的测试）。
            为 None 时用全通 ROADM。
        num_hops: 跳数，节点数 = num_hops + 1。
    """
    import networkx as nx

    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
    from network_nlo_eval.network.topology import NetworkTopology

    graph = nx.Graph()
    for hop in range(num_hops):
        graph.add_edge(hop, hop + 1, length_km=hop_length_km)

    return NetworkTopology(
        graph,
        FiberSpanConfig(length_km=hop_length_km),
        EDFAConfig(target_gain_db=0.0, noise_figure_db=5.0),
        roadm_config
        if roadm_config is not None
        else ROADMConfig(
            insertion_loss_db=5.0,
            filtering_penalty_db=0.2,
            express_attenuation_db=0.0,
            add_drop_loss_db=1.0,
            booster_noise_figure_db=5.0,
            equalize_output_power=True,
        ),
    )


def build_spectrum_grid(num_channels: int = NUM_CHANNELS):
    """物理面用的遗留固定栅格（物理仍逐信道路入功率）。"""
    from network_nlo_eval.core.spectrum import SpectrumGrid

    return SpectrumGrid(
        num_channels=num_channels,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )


def build_flex_grid(num_channels: int = NUM_CHANNELS):
    """与 `build_spectrum_grid` 等价的灵活栅格视图（仅用于报告槽坐标）。"""
    from network_nlo_eval.core.flexgrid import FlexibleGrid

    return FlexibleGrid.from_legacy_grid(
        num_channels=num_channels,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
        slots_per_channel=SLOTS_PER_CHANNEL,
    )


def build_stack(
    *,
    roadm_config=None,
    num_channels: int = NUM_CHANNELS,
    slots_per_channel: int = SLOTS_PER_CHANNEL,
    num_hops: int = 3,
):
    """组装一次分配所需的全部组件。

    Returns:
        ``(topology, spectrum_grid, gn_evaluator, qot_validator, path_cache,
          network_state)``。``network_state`` 的 ``slots_per_channel`` 与
        ``gn_evaluator`` 的**遗留**信道模式一致：物理面仍按信道定长。
    """
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.state import NetworkState
    from network_nlo_eval.rwa.path_computation import PathCache
    from network_nlo_eval.rwa.qot_checker import QoTValidator

    topology = build_topology(roadm_config=roadm_config, num_hops=num_hops)
    spectrum_grid = build_spectrum_grid(num_channels)

    gn_evaluator = MultiBandISRSGN(
        spectrum_grid,
        topology.get_default_fiber_config(),
        symbol_rate_hz=CHANNEL_SPACING_HZ,
    )
    qot_validator = QoTValidator(gn_evaluator, topology, spectrum_grid)
    path_cache = PathCache(topology, MAX_KSP_PATHS)
    network_state = NetworkState(
        topology,
        spectrum_grid,
        slots_per_channel=slots_per_channel,
    )
    return topology, spectrum_grid, gn_evaluator, qot_validator, path_cache, network_state


def make_request(plan: ServicePlan, *, snr_requirement_db: float = SNR_REQUIREMENT_DB):
    """由 :class:`ServicePlan` 构造 ``ServiceRequest``。"""
    from network_nlo_eval.simulation.traffic import ServiceRequest

    return ServiceRequest(
        service_id=plan.service_id,
        source_id=plan.source_id,
        destination_id=plan.destination_id,
        arrival_time=plan.arrival_time,
        departure_time=plan.departure_time,
        bit_rate_gbps=plan.bit_rate_gbps,
        snr_requirement_db=snr_requirement_db,
        launch_power_w=LAUNCH_POWER_W,
        n_slots=plan.n_slots,
    )


def multislot_service_plans(num_hops: int = 3) -> list[ServicePlan]:
    """端到端场景（M-J9）：6 个不同 ``n_slots`` 的业务，全部走全长路径。

    ``n_slots`` 覆盖 {1, 2, 3, 4, 6, 8}，其中 >=3 个互不相同，满足 M-J9 的
    ">=3 个不同 n_slots 拿到连续槽窗口"。到达/离开时间交错，使窗口在谱上
    首尾相接、部分重叠，用来观察连续窗口的真正排布。
    """
    far = num_hops
    return [
        ServicePlan(0, 0, far, 0.0, 100.0, 100.0, 1),
        ServicePlan(1, 0, far, 1.0, 100.0, 100.0, 2),
        ServicePlan(2, 0, far, 2.0, 100.0, 200.0, 3),
        ServicePlan(3, 0, far, 3.0, 100.0, 400.0, 4),
        ServicePlan(4, 0, far, 4.0, 100.0, 400.0, 6),
        ServicePlan(5, 0, far, 5.0, 100.0, 400.0, 8),
    ]


def degenerate_service_plans(num_services: int = 12, num_hops: int = 3) -> list[ServicePlan]:
    """退化等价场景（M-J10）：全部 ``n_slots = 1`` 的业务。

    到达/离开时间交错以制造真实的释放-再分配，使两套分配器在**相同**的
    占用演化下逐请求比较 ``(path, wavelength)``。
    """
    far = num_hops
    plans: list[ServicePlan] = []
    for i in range(num_services):
        arrival = float(i)
        departure = arrival + (3.0 if i % 3 else 6.0)
        plans.append(
            ServicePlan(
                service_id=i,
                source_id=0,
                destination_id=far,
                arrival_time=arrival,
                departure_time=departure,
                bit_rate_gbps=100.0,
                n_slots=1,
            )
        )
    return plans