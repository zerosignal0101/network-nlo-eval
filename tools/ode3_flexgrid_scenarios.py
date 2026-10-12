"""WP-O3 的灵活栅格场景（**本 WP 自写**，不 import ``wp/fg-uwb-fid`` 的 tests）。

为什么自己写
------------
WP-O3 §2 步骤 2 明确要求："可参照 ``wp/fg-uwb-fid`` 的 ``tests/fu4_scenarios.py``
里的 ``flex_*`` 场景，但**不要直接 import fu4 的 tests**——自己写，避免把它当权威"。
本文件因此只按本仓的**冻结契约**（``core/flexgrid.py`` 与 ``core/bands.py``）构造场景。

场景 ``flex_two_hop_mixed_width``
---------------------------------
一条 3 节点 2 跳链路（每跳 80 km），4 条**宽度互不相同**的 WDM 连接，
横跨 L|C 波段边界（191.55 THz）：

===========  ==========  ========  ========  ===============  ========  ======
conn_id      first_slot  n_slots  调制      bit_rate (Gbps)  频段     中心频率
===========  ==========  ========  ========  ===============  ========  ======
0            0           2        QPSK      50               L        191.0125 THz
1            10          6        QPSK      150              L        191.1625 THz
2            80          3        8QAM      100              C        192.01875 THz
3            120         8        16QAM     400              C        192.5500 THz
===========  ==========  ========  ========  ===============  ========  ======

⇒ ``n_slots`` 取值 ``{2, 6, 3, 8}`` 共 **4 个不同值**（J8 要求 ≥2）。

⚠️ 符号率为什么是**统一**的 50 GHz
----------------------------------
每条连接的 ``n_slots`` 由 ``slots_for_rate(bit_rate, bits_per_symbol, roll_off,
slot_width)`` 强制（``ConnectionSpec.__post_init__``），故不同比特率/调制自然对应
不同宽度。灵活栅格的"每连接 Rs = 该连接占用带宽"因此会推出**异构** Rs。

但冻结的 njit 核 ``_calc_span_noise_and_power_jit`` 的 ``symbol_rate_hz`` 是
**标量**类型（它要拿去算标量 ``f_m_hz`` / ``f_M_hz`` 并喂给逐标量的 SPM/XPM）。
改成数组参数 = 改冻结内核 = 违反 J5。WP-O2 已因此显式拒绝异构 Rs，
WP-O3 §5 明确要求"别重复踩同一个坑"，故本场景：

* **统一** ``symbol_rate_hz = 50 GHz``（= 2 槽宽），使不同宽度的连接共存；
* 连接占用的**谱宽仍互不相同**（2/6/3/8 槽），这正是灵活栅格要表达的东西；
* "连接声明宽度 ≠ 声明 Rs" 这一不一致，按 J5 作为**限制**上报（见
  ``results/ode_dev_flexgrid_e2e.json`` 的 ``known_limitations``），
  并用 ``heterogeneous_rs_probe`` 证明它确实被拒绝。

逐波段发射功率
--------------
``BandPlan.apply_launch_power`` 把逐波段功率（L=1.0 mW，C=0.5 mW）写回连接，
再由场景把 ``conn.launch_power_w`` 铺进 ``LinkState.launch_power_profile_w``
——这就是"逐波段功率 → 逐连接发射谱"的完整通路。
"""
from __future__ import annotations

from dataclasses import dataclass

from network_nlo_eval.core.bands import BandSpec
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid

#: 槽宽：12.5 GHz（= 50 GHz 信道间隔 / 4 槽，见 ``FlexibleGrid.from_legacy_grid``）。
SLOT_WIDTH_HZ = 12.5e9

#: 栅格：191.0 – 194.0 THz，共 240 槽（3 THz）。跨 L|C 边界 191.55 THz。
FLEX_GRID = FlexibleGrid(
    slot_width_hz=SLOT_WIDTH_HZ,
    num_slots=240,
    band_low_edge_hz=191.0e12,
)

#: 频段集合，按**频率从低到高**：L -> C。
#:
#: ⚠️ 波长约定：``low_nm`` 是**短**波长（数值小、频率高）。故
#: ``BandSpec("L", 1565.0, 1625.0)`` = 184.49 – 191.55 THz。
#: 本仓**不预置**波段表（FU-2 的 S/C/L/U 表未移植），故在此显式给出。
BAND_SPECS: tuple[BandSpec, ...] = (
    BandSpec("L", 1565.0, 1625.0, "EDFA", 5.0, launch_power_w=1.0e-3),
    BandSpec("C", 1530.0, 1565.0, "EDFA", 4.5, launch_power_w=0.5e-3),
)

#: 统一符号率 (Hz)。见模块 docstring「符号率为什么是统一的 50 GHz」。
SYMBOL_RATE_HZ = 50e9

#: 新业务承载的连接下标（用于 ``verify_allocation`` 的 ``channel_idx``）。
SERVICE_CONNECTION_IDX = 2

#: 新业务路径（3 节点 2 跳）。
SERVICE_PATH: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True)
class ConnectionPlan:
    """一条待建连接的声明参数（尚无发射功率，交给 ``apply_launch_power`` 填）。"""

    connection_id: int
    first_slot: int
    n_slots: int
    bit_rate_bps: float
    modulation: str


#: 4 条宽度互不相同的连接。``n_slots`` 已按 ``slots_for_rate`` 核对：
#: QPSK 50 Gbps -> 25 GHz/12.5 = 2 槽；QPSK 150 Gbps -> 75 GHz -> 6 槽；
#: 8QAM 100 Gbps -> 33.33 GHz -> ceil 3 槽；16QAM 400 Gbps -> 100 GHz -> 8 槽。
CONNECTION_PLANS: tuple[ConnectionPlan, ...] = (
    ConnectionPlan(0, first_slot=0, n_slots=2, bit_rate_bps=50e9, modulation="QPSK"),
    ConnectionPlan(1, first_slot=10, n_slots=6, bit_rate_bps=150e9, modulation="QPSK"),
    ConnectionPlan(2, first_slot=80, n_slots=3, bit_rate_bps=100e9, modulation="8QAM"),
    ConnectionPlan(3, first_slot=120, n_slots=8, bit_rate_bps=400e9, modulation="16QAM"),
)


def build_connections() -> list[ConnectionSpec]:
    """按声明参数构造 4 条 ``ConnectionSpec``（发射功率暂置 0 = 未占用）。

    ``launch_power_w=0.0`` 是必要的初值：逐波段功率由
    ``BandPlan.apply_launch_power`` 随后覆盖（``ConnectionSpec`` 是 frozen，
    故那里返回的是新对象）。
    """
    return [
        ConnectionSpec(
            grid=FLEX_GRID,
            connection_id=plan.connection_id,
            first_slot=plan.first_slot,
            n_slots=plan.n_slots,
            bit_rate_bps=plan.bit_rate_bps,
            modulation=plan.modulation,
            symbol_rate_hz=SYMBOL_RATE_HZ,
            roll_off=0.0,
            launch_power_w=0.0,
        )
        for plan in CONNECTION_PLANS
    ]


def build_topology(hop_length_km: float = 80.0):
    """构造 3 节点 2 跳线形拓扑（0 -- 1 -- 2），每跳 ``hop_length_km``。"""
    import networkx as nx

    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
    from network_nlo_eval.network.topology import NetworkTopology

    graph = nx.Graph()
    graph.add_edge(0, 1, length_km=hop_length_km)
    graph.add_edge(1, 2, length_km=hop_length_km)

    return NetworkTopology(
        graph,
        FiberSpanConfig(length_km=hop_length_km),
        EDFAConfig(target_gain_db=0.0, noise_figure_db=5.0),
        ROADMConfig(
            insertion_loss_db=5.0,
            filtering_penalty_db=0.2,
            express_attenuation_db=0.0,
            add_drop_loss_db=1.0,
            booster_noise_figure_db=5.0,
            equalize_output_power=True,
        ),
    )


def build_state_grid(num_nodes: int):
    """构造仅用于**数组定长**的遗留 ``SpectrumGrid``。

    ``NetworkState`` 用 ``spectrum_grid.num_channels`` 给
    ``occupied_channels`` / ``launch_power_profile_w`` 定长。逐连接模式下真正的
    索引空间是**连接数**，故这里的网格只承担"定长"职责，其频率值不被读取
    （逐连接模式下频率真源是 ``ConnectionSpec.centre_hz``）。
    """
    from network_nlo_eval.core.spectrum import SpectrumGrid

    return SpectrumGrid(
        num_channels=num_nodes,
        channel_spacing_hz=50e9,
        center_frequency_hz=193.1e12,
    )
