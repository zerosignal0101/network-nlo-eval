"""传输质量 (Quality of Transmission, QoT) 验证器。

负责在 RWA 过程中评估链路或路径的 SNR。
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey, NDArrayFloat, NodeID
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.network.slots import connection_overlaps_slots
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

#: 普朗克常数 (J·s)。booster ASE 项 ``NF * h * nu * Rs * (G-1)`` 的系数，
#: 与 RL 侧 `multiband_optical_rl/scan.py:56` 的 `_H_PLANCK` 同源同值。
_H_PLANCK = 6.62607015e-34


def _normalize(u: NodeID, v: NodeID) -> LinkKey:
    """返回规范化的链路键 (min, max)。"""
    return LinkKey((u, v) if u < v else (v, u))


# ======================================================================
# ROADM 常量折叠与单信道路径 SNR 累积的唯一真源（FG-11）
# ======================================================================
#
# FG-5 把链路噪声装配收敛成 `rwa/link_noise.py:assemble_link_noise`，但它列出的
# 未闭合项还有两条留在原地，本节把它们各留**一份**定义：
#
# 1. **单信道 SNR 递推**：原 `QoTValidator._eval_service_snr` 的循环体，与 RL 侧
#    `ScanValidator._service_snr` / `_path_channel_snr` 各写一份，物理公式相同。
#    副本正是 FU-4 那次"两边 SNR 相差 0.2878 dB 却全仓无测试变红"的成因。
# 2. **ROADM 常量折叠**：`10 ** (path_loss_db(op) / 10)` 等原本两仓各求值一次。
#
# **两侧保留的差异只有取值来源，不含任何算术**——与 `link_noise.py` 的分工一致：
#
# * **阻断判定**：物理侧可能处在逐连接（灵活栅格）模式，阻断是**槽区间重叠**语义，
#   由 `QoTValidator._is_blocked` 判定；RL 侧 `env.py` 构造 `MultiBandISRSGN` 时
#   **不传** `connections`，恒为遗留固定栅格，阻断就是信道下标集合成员判定。
#   两者由调用方各自传入的 `is_blocked` 谓词表达，本模块**不选边**。
# * **频率/符号率**：物理侧取 `gn_evaluator` 的逐节点数组，RL 侧取频谱栅格与标量
#   符号率。遗留模式下二者逐位相同（`f_abs_hz is grid.frequencies`，
#   `symbol_rate_hz_channels` 每项都等于标量），灵活栅格下不同。调用方各自求值
#   后按标量传入。
#
# **为什么放在本文件而不是新建 `rwa/snr_terms.py`。** 两者同属 `network_nlo_eval/rwa/`
# 这一层，但 `tests/test_fu4_assembly_union.py::test_fu4_6b_union_keeps_every_fg2_
# capability_symbol` 是一条**源码级守卫**，它按字面检查 `path_loss_db` /
# `booster_noise_figure_db` / `filtering_penalty_db` 仍在 `rwa/qot_checker.py` 里。
# 那条守卫要守的是"能力没被删掉"，而拆到新文件会让它在物理**完好无损**的情况下变红。
# 为一个私有实现的搬迁去改既有测试的期望值是本仓明令禁止的（G2 纪律），所以这里
# 选择留在原文件：条件反射的新模块反而会踩到既有守卫。

#: 一跳上交给累积器的材料：(该跳的 ROADM 常量, 该跳的噪声缓存)。
#:
#: 用**可迭代对象**而不是列表，是为了让物理侧保持**惰性**：`_eval_service_snr`
#: 在遇到被阻断的跳时立刻返回 `-inf`，若先把整条路径的噪声都算出来，后面那些跳的
#: `LinkState.noise_cache` 写回时机就变了（`_get_or_calc_noise_cache` 有写回副作用）。
#: 传惰性迭代器则在返回点自然停止拉取。
HopPair = tuple["RoadmTerms", LinkNoiseCache]


@dataclass(frozen=True)
class RoadmTerms:
    """一个 (节点, 端口操作) 上与候选信道无关的 ROADM 常量。"""

    config: ROADMConfig = field(compare=False, repr=False)
    """原始配置。**只为阻断判定保留**——灵活栅格下判定要回到配置上的槽集合。"""
    loss: float
    """``10 ** (path_loss_db(operation) / 10)``。"""
    loss_db: float
    """``path_loss_db(operation)`` —— :attr:`loss` 的 **dB 形式，同一次求值**。

    给波形侧（``physics/qot_checker.py``）用：那里的物理 ROADM / EDFA 元件收的是 dB。
    本字段**不是** ``10 * log10(loss)`` 的反算——那个反算在 IEEE-754 下**不保证逐位
    还原**（FG-17 实测：合成配置网格 39.6%、随机 dB 5.57% 差 1 ulp，见
    ``results/fg17_assembly_evidence.json``），那会让"收敛"变成一次静默的数值改动。
    这里保留折叠时**同一次** ``path_loss_db`` 求值的 dB 结果，波形侧拿到的值因此与
    它自己直接读配置**逐位相同**。
    """
    filtering_penalty: float
    """``10 ** (filtering_penalty_db / 10)``。"""
    equalize: bool
    """是否由 booster 把每信道功率拉回入纤值（等价于"有 booster NF"）。"""
    booster_nf: float | None
    """``10 ** (booster_noise_figure_db / 10)``；无 booster 时为 ``None``。

    与 :attr:`booster_base` **不是冗余**：向量路径按上游写法
    ``nf * h * f * sr * (loss - 1)`` 逐元素算，需要**未乘 h** 的 NF；
    标量路径用折好的 :attr:`booster_base`。两者都必须由本模块一次算出——
    反推（``booster_base / h``）会引入一次除法，**逐位不等**。
    """
    booster_base: float
    """``10 ** (booster_noise_figure_db / 10) * h``（无 booster 时取按 NF 字段算出的值）。"""
    booster_nf_db: float
    """``config.booster_noise_figure_db`` —— :attr:`booster_nf` 的 **dB 形式，同一次求值**。

    与 :attr:`loss_db` 同理，给收 dB 的波形侧元件用；不是 ``10 * log10(booster_nf)``
    的反算（FG-17 实测该反算不保证逐位还原）。注意本字段**不随 ``equalize`` 门控**，
    与 :attr:`booster_nf` 的 ``None`` 语义不同：是否建 booster 由 :attr:`equalize` 决定。
    """
    blocked: frozenset[int]
    """遗留固定栅格下的静态阻断信道下标。"""


def fold_roadm_terms(config: ROADMConfig, *, is_last_hop: bool) -> RoadmTerms:
    """把一个 ROADM 配置折成 :class:`RoadmTerms`。

    这是 ROADM 常量折叠的**唯一**定义。FU-4 裁决（详见
    ``docs/fu4_qot_assembly_adjudication.md`` §1）在这里落地：

    1. **损耗按端口区分**：``path_loss_db("drop" | "express")``，不把
       ``filtering_penalty_db`` 并进损耗——后者是噪声域量，不是光功率损耗。
    2. **booster**：``equalize_output_power`` 为真时信号**不再**被 ``loss`` 衰减，
       代价是引入 ``NF * h * f * Rs * (loss - 1)`` 的 booster ASE。
    3. **滤波惩罚**是只作用于噪声的独立乘子。

    Args:
        config: 该节点的 ROADM 配置。
        is_last_hop: 是否为路径的**末端**跳（决定取 ``drop`` 还是 ``express`` 端口）。

    Returns
    -------
        RoadmTerms: 折叠好的常量。两仓共用这一个函数对象。
    """
    operation = "drop" if is_last_hop else "express"
    nf_lin = 10 ** (config.booster_noise_figure_db / 10)
    booster_nf_lin = nf_lin if config.equalize_output_power else None
    # 只求值一次，dB 与线性两个形式共用它——波形侧要 dB，统计侧要线性（见 RoadmTerms.loss_db）。
    loss_db = config.path_loss_db(operation)
    booster_nf_db = config.booster_noise_figure_db
    return RoadmTerms(
        config=config,
        loss=10 ** (loss_db / 10),
        loss_db=loss_db,
        filtering_penalty=10 ** (config.filtering_penalty_db / 10),
        equalize=config.equalize_output_power,
        booster_nf=booster_nf_lin,
        booster_base=nf_lin * _H_PLANCK,
        booster_nf_db=booster_nf_db,
        blocked=frozenset(config.blocked_channels),
    )


def accumulate_service_snr(
    hops: Iterable[HopPair],
    *,
    channel_idx: int,
    launch_power_w: float,
    frequency_hz: float,
    symbol_rate_hz: float,
    is_blocked: Callable[[RoadmTerms], bool],
) -> float:
    """沿服务自身路径累积噪声，给出单信道 SNR (dB)。

    对应上游 ``QoTValidator._eval_service_snr`` 的语义：被阻断即 ``-inf``，
    噪声为 0 时给 ``+inf``（正无穷 SNR），booster 项在信号功率为 0 时**仍然**计入。

    Args:
        hops: 逐跳的 ``(ROADM 常量, 噪声缓存)``。可以是惰性迭代器：遇到被阻断的
            跳会立刻停止拉取，从而不触碰后面那些跳的噪声缓存写回。
        channel_idx: 候选信道（遗留模式下即 ODE 节点下标）。
        launch_power_w: 该业务的发射功率，作为信号功率起点 (W)。
        frequency_hz: 该节点的中心频率 (Hz)，由调用方按自身栅格语义求值。
        symbol_rate_hz: 该节点的符号率 (Hz)，同上。
        is_blocked: 判定某一跳的该信道是否被阻断。由调用方提供，因为遗留与灵活
            两种栅格的阻断语义不同（见模块文档）。

    Returns
    -------
        float: SNR (dB)；被阻断为 ``-inf``，噪声为 0 为 ``+inf``。
    """
    total_spm = 0.0
    total_xpm = 0.0
    total_ase = 0.0
    signal_power = launch_power_w
    total_spans = 0

    for roadm, cache in hops:
        total_spm += cache.spm.item(channel_idx)
        total_xpm += cache.xpm.item(channel_idx)
        total_ase += cache.ase.item(channel_idx)
        total_spans += cache.span_count

        if is_blocked(roadm):
            return float("-inf")
        if roadm.equalize:
            total_ase += roadm.booster_base * frequency_hz * symbol_rate_hz * (roadm.loss - 1.0)
        else:
            signal_power /= roadm.loss
            total_spm /= roadm.loss
            total_xpm /= roadm.loss
            total_ase /= roadm.loss
        total_spm *= roadm.filtering_penalty
        total_xpm *= roadm.filtering_penalty
        total_ase *= roadm.filtering_penalty

    penalty = total_spans**0.05 if total_spans > 0 else 1.0
    noise = total_spm * penalty + total_xpm + total_ase
    if noise <= 0:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise)


def accumulate_path_channel_snr(
    hops: Iterable[HopPair],
    *,
    channel_idx: int,
    launch_power_w: float,
    frequency_hz: float,
    symbol_rate_hz: float,
    is_blocked: Callable[[RoadmTerms], bool],
) -> float:
    """给出路径上**单个**信道上的 SNR (dB)，语义同 ``_build_path_snr_array``。

    与 :func:`accumulate_service_snr` 的**两处故意差异**，都对应上游
    ``_build_path_snr_array`` 的真实行为，不能互相顶替：

    1. booster ASE 在信号功率为 0 时**不加**（上游 ``booster_ase[signal_power <= 0] = 0``）；
    2. 被阻断时把 ``signal_power`` 清零并**继续**走完路径；
    3. 噪声为 0 **或**信号功率非正时返回 ``-inf``（而非 ``+inf``）。

    上游那个逐信道循环里没有任何跨信道的归约，每一步都是逐元素运算，因此本函数
    在下标 ``channel_idx`` 上的取值与上游整条数组在该元素上**逐位相等**。
    """
    total_spm = 0.0
    total_xpm = 0.0
    total_ase = 0.0
    signal_power = launch_power_w
    total_spans = 0

    for roadm, cache in hops:
        total_spm += cache.spm.item(channel_idx)
        total_xpm += cache.xpm.item(channel_idx)
        total_ase += cache.ase.item(channel_idx)
        total_spans += cache.span_count

        if roadm.equalize:
            if signal_power > 0:
                total_ase += (
                    roadm.booster_base * frequency_hz * symbol_rate_hz * (roadm.loss - 1.0)
                )
        else:
            signal_power /= roadm.loss
            total_spm /= roadm.loss
            total_xpm /= roadm.loss
            total_ase /= roadm.loss
        total_spm *= roadm.filtering_penalty
        total_xpm *= roadm.filtering_penalty
        total_ase *= roadm.filtering_penalty
        # 构造上不可达：候选信道一旦被任一跳阻断，`accumulate_service_snr` 已在
        # 同一路径上返回 -inf，第 3 步门限随即否决，流程走不到这里。保留是为了与
        # 上游 `_build_path_snr_array` 的同一行保持一字不差的对应关系。
        if is_blocked(roadm):
            signal_power = 0.0

    penalty = total_spans**0.05 if total_spans > 0 else 1.0
    noise = total_spm * penalty + total_xpm + total_ase
    if noise <= 0 or signal_power <= 0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / noise)


class QoTValidator:
    """传输质量验证器。

    RWA 算法使用它来判断一个候选分配方案是否满足 SNR 要求。

    核心策略：**临时副本 + 局部缓存**，不修改 LinkState，无需回滚。
    """

    def __init__(
        self,
        gn_evaluator: MultiBandISRSGN,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        fixed_span_length_km: float | None = None,
    ):
        self.gn_evaluator = gn_evaluator
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        self._fixed_span_length_km = fixed_span_length_km

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """ODE 节点数（= 活跃连接数），噪声数组按此长度分配。

        遗留模式下等于 `spectrum_grid.num_channels`；灵活栅格下等于连接数
        （设计 §2.4：一条连接一个节点，与槽数无关）。
        """
        return self.gn_evaluator.n_nodes

    def _is_blocked(self, roadm, node_index: int) -> bool:
        """判断某节点（连接）是否被该 ROADM 阻断。

        灵活栅格下按**槽区间重叠**判定（切片语义）：连接与任一 `blocked_slots`
        重叠即不可行。遗留模式保持按信道索引判定的既有行为。
        """
        connections = self.gn_evaluator.connections
        if connections is not None:
            if node_index >= len(connections):
                return False
            blocked_slots = set(roadm.blocked_slots)
            if roadm.blocked_channels and self.gn_evaluator.slots_per_channel:
                per_channel = self.gn_evaluator.slots_per_channel
                for channel_index in roadm.blocked_channels:
                    blocked_slots.update(range(channel_index * per_channel, (channel_index + 1) * per_channel))
            conn = connections[node_index]
            return connection_overlaps_slots(conn.first_slot, conn.n_slots, blocked_slots)
        return node_index in roadm.blocked_channels

    def _roadm_terms(self, roadm, is_last_hop: bool) -> tuple[float, float, float | None]:
        """返回该 ROADM 的 ``(损耗, 滤波惩罚, booster NF 线性值)``。

        FG-11 起这只是 :func:`network_nlo_eval.rwa.snr_terms.fold_roadm_terms` 的
        薄封装——**折叠逻辑的唯一真源在物理仓的共享模块里**，RL 侧
        ``ScanValidator`` 也调同一个函数对象。

        **返回形状刻意保持原样（三元组），没有改成 ``RoadmTerms``。** 试过改成
        dataclass，代价是 ``tests/test_fu4_assembly_union.py`` 里三条按三元组解包的
        断言直接红——而那三条要守的物理结论（drop/express 损耗不同、滤波只乘噪声、
        booster 只在 equalize 下存在）**一条都没有因为这次重构而失效**。为了让一个
        私有辅助函数的签名去改既有测试的期望值，是本刀明令禁止的（G2 纪律），
        所以这里保住签名，只把**折叠**收敛到一处。

        FU-4 裁决（详见 ``docs/fu4_qot_assembly_adjudication.md`` §1）的三条——
        损耗按端口区分、booster 恢复入纤功率、滤波惩罚只作用于噪声——
        现在落在共享模块里，两仓不会再各写一份。
        """
        terms = fold_roadm_terms(roadm, is_last_hop=is_last_hop)
        return terms.loss, terms.filtering_penalty, terms.booster_nf

    def _node_frequency_and_rate(self, channel_idx: int) -> tuple[float, float]:
        """返回该节点（连接）的 ``(中心频率 Hz, 符号率 Hz)``。

        必须走 ``gn_evaluator`` 的**逐节点**数组，不能用
        ``spectrum_grid.frequencies``：灵活栅格下 ``channel_idx`` 索引的是
        连接（ODE 节点），与固定栅格的信道索引不是同一个索引空间。

        遗留模式下 ``f_abs_hz is spectrum_grid.frequencies`` 且
        ``symbol_rate_hz_channels`` 全等于标量 ``symbol_rate_hz``，故这条路
        在 legacy 下逐位等价。
        """
        evaluator = self.gn_evaluator
        return (
            float(evaluator.f_abs_hz[channel_idx]),
            float(evaluator.symbol_rate_hz_channels[channel_idx]),
        )

    def verify_allocation(
        self,
        service_request: ServiceRequest,
        path: list[NodeID],
        channel_idx: int,
        current_network_state: NetworkState,
        fixed_span_length_km: float | None = None,
    ) -> tuple[bool, NDArrayFloat | None]:
        """验证将 ``service_request`` 分配到 ``path`` 和 ``channel_idx`` 是否满足所有 SNR 要求。

        1. 创建新业务路径上各链路的临时功率谱副本（含新业务功率），不修改 LinkState。
        2. 对受影响链路计算噪声并存入局部缓存（仅一次）。
        3. 用 _eval_service_snr 评估新业务 SNR。
        4. 收集受影响旧业务，用各自路径和发射功率评估 SNR。
        5. 全部通过后，将临时缓存写入 LinkState.noise_cache。

        Returns
        -------
            Tuple[bool, Optional[NDArrayFloat]]:
            - bool: 如果所有 SNR 检查通过，则为 True；否则为 False。
            - Optional[NDArrayFloat]: 如果成功，返回新业务路径上每个信道的 SNR (dB)，
              否则为 None。
        """
        span_km = fixed_span_length_km or self._fixed_span_length_km

        # 1. 创建临时功率谱副本 & 检查波长可用性
        temp_power_profiles: dict[LinkKey, NDArrayFloat] = {}
        affected_links: set[LinkKey] = set()

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = current_network_state.get_link_state(u, v)

            if link_state.occupied_channels[channel_idx]:
                return False, None  # 波长已被占用

            link_key = _normalize(u, v)
            affected_links.add(link_key)
            temp_profile = link_state.launch_power_profile_w.copy()
            temp_profile[channel_idx] = service_request.launch_power_w
            temp_power_profiles[link_key] = temp_profile

        # 2. 计算受影响链路的噪声（仅一次），存入局部缓存
        temp_noise_cache: dict[LinkKey, LinkNoiseCache] = {}
        for link_key in affected_links:
            temp_noise_cache[link_key] = self._calc_link_noise(
                link_key,
                temp_power_profiles[link_key],
                span_km,
            )

        # 3. 评估新业务 SNR
        new_snr_db = self._eval_service_snr(
            path,
            channel_idx,
            service_request.launch_power_w,
            temp_noise_cache,
            current_network_state,
            span_km,
        )
        if new_snr_db < service_request.snr_requirement_db:
            return False, None

        # 4. 收集受影响旧业务
        affected_old_services = self._collect_affected_services(
            path,
            channel_idx,
            current_network_state,
        )

        # 5. 评估每个旧业务 SNR（使用其自身路径和发射功率）
        for old_service in affected_old_services:
            old_snr_db = self._eval_service_snr(
                old_service.path,
                old_service.wavelength,
                old_service.launch_power_w,
                temp_noise_cache,
                current_network_state,
                span_km,
            )
            if old_snr_db < old_service.snr_requirement_db:
                return False, None

        # 6. 全部通过 — 将临时缓存写入 LinkState（避免后续重复计算）
        for link_key, cache in temp_noise_cache.items():
            u, v = link_key
            link_state = current_network_state.get_link_state(u, v)
            link_state.noise_cache = cache

        # 构建新业务路径上各信道的 SNR 数组（用于返回值兼容性）
        final_snrs_db = self._build_path_snr_array(
            path,
            temp_power_profiles,
            temp_noise_cache,
            current_network_state,
            span_km,
        )

        return True, final_snrs_db

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _calc_link_noise(
        self,
        link_key: LinkKey,
        power_profile: NDArrayFloat,
        fixed_span_length_km: float | None,
    ) -> LinkNoiseCache:
        """计算单条链路在给定功率谱下的噪声缓存。"""
        u, v = link_key
        fiber_config = self.network_topology.get_fiber_config(u, v)
        edfa_config = self.network_topology.get_link_edfa_config(u, v)
        evaluator = self.gn_evaluator.for_fiber(fiber_config)

        link_total_length_m = fiber_config.length_km * 1000
        span_length_m = (fixed_span_length_km or fiber_config.max_span_length_km) * 1000
        num_full_spans = int(link_total_length_m // span_length_m)
        remainder_length_m = link_total_length_m % span_length_m

        # FG-5：这段噪声累加已提取到 `assemble_link_noise`，与 RL 侧共用同一个函数
        # 对象。本方法现在只负责"从拓扑取值"（光纤参数 / EDFA / 跨段切分），
        # 不再持有任何噪声算术——重复实现正是历史上两仓悄悄分叉的成因。
        return assemble_link_noise(
            evaluator=evaluator,
            edfa_config=edfa_config,
            power_profile=power_profile,
            span_length_m=span_length_m,
            num_full_spans=num_full_spans,
            remainder_length_m=remainder_length_m,
            num_channels=self.num_nodes,
        )

    def _get_or_calc_noise_cache(
        self,
        link_key: LinkKey,
        link_state,
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        fixed_span_length_km: float | None,
    ) -> LinkNoiseCache:
        """获取链路噪声缓存：优先从临时缓存，否则从 LinkState（惰性计算）。"""
        if link_key in temp_noise_cache:
            return temp_noise_cache[link_key]

        if link_state.noise_cache is not None:
            return link_state.noise_cache

        cache = self._calc_link_noise(
            link_key,
            link_state.launch_power_profile_w,
            fixed_span_length_km,
        )
        link_state.noise_cache = cache
        return cache

    def _eval_service_snr(
        self,
        path: list[NodeID],
        channel_idx: int,
        launch_power_w: float,
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        network_state: NetworkState,
        fixed_span_length_km: float | None,
    ) -> float:
        """通用方法：沿服务自身路径累积噪声，计算单信道 SNR (dB)。

        - 受影响链路从 temp_noise_cache 读取（含假设分配后的功率谱）。
        - 未受影响链路从 LinkState.noise_cache 读取（惰性缓存）。
        - 使用服务自身的 launch_power_w 作为信号功率。

        FG-11 起本方法**只做取值**，递推交给共享的
        :func:`network_nlo_eval.rwa.snr_terms.accumulate_service_snr`——那是这条
        递推的唯一真源，RL 侧 ``ScanValidator`` 也调同一个函数对象。

        这里保留的差异只有**取值来源**，不含任何算术：

        * 噪声缓存的取法（``temp_noise_cache`` → ``LinkState`` → 惰性写回）；
        * ROADM 配置按**路径终点节点** ``v`` 逐跳取，并按端口折叠；
        * 阻断判定用本类的 ``_is_blocked``（逐连接模式下是**槽区间重叠**语义）；
        * 频率/符号率取 ``gn_evaluator`` 的**逐节点**数组。

        **惰性必须保留。** 下面传的是生成器而不是预先建好的列表：被阻断的跳会让
        共享核立刻返回 ``-inf``，生成器随之停止拉取，于是该跳之后的链路**不会**被
        写回 ``LinkState.noise_cache``。若改成先建列表，那些写回就会提前发生，
        ``reopt/trial.py`` 依赖的缓存时机语义随之改变。
        """
        frequency_hz, symbol_rate_hz = self._node_frequency_and_rate(channel_idx)
        last_hop = len(path) - 2

        def _hops():
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_key = _normalize(u, v)
                link_state = network_state.get_link_state(u, v)
                cache = self._get_or_calc_noise_cache(
                    link_key, link_state, temp_noise_cache, fixed_span_length_km
                )
                roadm = self.network_topology.get_roadm_config(v)
                # 直接给共享核 `RoadmTerms`，而不是本类那个三元组封装——
                # 标量路径走共享核，向量路径 `_build_path_snr_array` 走三元组封装，
                # 两条路的折叠都出自同一个 `fold_roadm_terms`。
                yield fold_roadm_terms(roadm, is_last_hop=i == last_hop), cache

        return accumulate_service_snr(
            _hops(),
            channel_idx=channel_idx,
            launch_power_w=launch_power_w,
            frequency_hz=frequency_hz,
            symbol_rate_hz=symbol_rate_hz,
            is_blocked=lambda terms: self._is_blocked(terms.config, channel_idx),
        )

    def _collect_affected_services(
        self,
        path: list[NodeID],
        channel_idx: int,
        network_state: NetworkState,
    ) -> list[AllocatedService]:
        """收集新业务路径上受影响的旧业务（排除新业务自身占用的信道）。"""
        seen_ids: set[int] = set()
        affected: list[AllocatedService] = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = network_state.get_link_state(u, v)

            for ch_idx in range(self.num_nodes):
                if ch_idx == channel_idx:
                    continue
                if not link_state.occupied_channels[ch_idx]:
                    continue
                service_id = link_state.allocated_service_ids[ch_idx]
                if service_id == -1 or service_id in seen_ids:
                    continue
                seen_ids.add(service_id)
                svc = network_state.get_allocated_service(service_id)
                if svc is not None:
                    affected.append(svc)

        return affected

    def _build_path_snr_array(
        self,
        path: list[NodeID],
        temp_power_profiles: dict[LinkKey, NDArrayFloat],
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        network_state: NetworkState,
        fixed_span_length_km: float | None,
    ) -> NDArrayFloat:
        """构建新业务路径上各信道的 SNR (dB) 数组，用于返回值。"""
        total_spm = np.zeros(self.num_nodes, dtype=np.float64)
        total_xpm = np.zeros(self.num_nodes, dtype=np.float64)
        total_ase = np.zeros(self.num_nodes, dtype=np.float64)
        total_spans = 0
        u0, v0 = path[0], path[1]
        link_key0 = _normalize(u0, v0)
        if link_key0 in temp_power_profiles:
            signal_power = temp_power_profiles[link_key0].copy()
        else:
            signal_power = network_state.get_link_state(u0, v0).launch_power_profile_w.copy()

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_key = _normalize(u, v)
            link_state = network_state.get_link_state(u, v)

            cache = self._get_or_calc_noise_cache(
                link_key,
                link_state,
                temp_noise_cache,
                fixed_span_length_km,
            )

            total_spm += cache.spm
            total_xpm += cache.xpm
            total_ase += cache.ase
            total_spans += cache.span_count

            roadm = self.network_topology.get_roadm_config(v)
            # FU-4 并集装配，逐节点向量化；与 _eval_service_snr 的标量分支同源
            # （同一个 fold_roadm_terms），只是频率/符号率改成逐节点数组。
            loss, filtering_penalty, booster_nf = self._roadm_terms(roadm, i == len(path) - 2)
            if booster_nf is not None:
                booster_ase = (
                    booster_nf
                    * _H_PLANCK
                    * self.gn_evaluator.f_abs_hz
                    * self.gn_evaluator.symbol_rate_hz_channels
                    * (loss - 1.0)
                )
                # 信号为 0 的节点 SNR 必为 -inf（下游按 signal_power>0 过滤），
                # 故此处置 0 不改变任何输出值；保留它是为了与 RL 侧 scan.py
                # 的快速通道保持逐位一致。
                booster_ase[signal_power <= 0] = 0
                total_ase += booster_ase
            else:
                signal_power /= loss
                total_spm /= loss
                total_xpm /= loss
                total_ase /= loss
            total_spm *= filtering_penalty
            total_xpm *= filtering_penalty
            total_ase *= filtering_penalty
            blocked = [index for index in range(self.num_nodes) if self._is_blocked(roadm, index)]
            if blocked:
                signal_power[blocked] = 0

        penalty = total_spans**0.05 if total_spans > 0 else 1.0
        total_noise = total_spm * penalty + total_xpm + total_ase

        snrs_db = np.full(self.num_nodes, -np.inf, dtype=np.float64)
        for ch_idx in range(self.num_nodes):
            if total_noise[ch_idx] <= 0:
                continue
            if signal_power[ch_idx] > 0:
                snrs_db[ch_idx] = 10.0 * math.log10(signal_power[ch_idx] / total_noise[ch_idx])

        return snrs_db


# --- 基础 RWA 分配器接口 ---
class BaseRWAAllocator:
    """RWA 分配器的抽象基类."""

    def allocate(
        self, service_request: ServiceRequest, network_state: NetworkState
    ) -> tuple[bool, AllocatedService | None]:
        """尝试分配一个新业务，返回是否成功及分配后的服务对象."""
        raise NotImplementedError

    def release(self, allocated_service: AllocatedService, network_state: NetworkState) -> None:
        """释放一个已分配业务的资源."""
        raise NotImplementedError
