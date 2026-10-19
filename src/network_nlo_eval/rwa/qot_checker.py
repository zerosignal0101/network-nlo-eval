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
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

#: 普朗克常数 (J·s)。booster ASE 项 ``NF * h * nu * B_i * (G-1)`` 的系数。
_H_PLANCK = 6.62607015e-34


def _normalize(u: NodeID, v: NodeID) -> LinkKey:
    """返回规范化的链路键 (min, max)。"""
    return LinkKey((u, v) if u < v else (v, u))


@dataclass(frozen=True)
class RoadmTerms:
    """一个 (节点, 端口操作) 上与候选信道无关的 ROADM 常量。

    Attributes
    ----------
        config: 原始配置，只为阻断判定保留。
        loss: ``10 ** (path_loss_db(operation) / 10)``。
        loss_db: 同一次 ``path_loss_db`` 求值的 dB 形式。收 dB 的波形侧元件用它，
            而不是 ``10 * log10(loss)`` 反算——反算在 IEEE-754 下不保证逐位还原。
        filtering_penalty: ``10 ** (filtering_penalty_db / 10)``，只作用于噪声。
        equalize: 是否由 booster 把每信道功率拉回入纤值。
        booster_nf: ``10 ** (booster_noise_figure_db / 10)``；无 booster 时为 ``None``。
        booster_base: ``booster_nf * h``，标量路径把它乘上频率与带宽。
        booster_nf_db: ``config.booster_noise_figure_db``，同样为 dB 形式。
    """

    config: ROADMConfig = field(compare=False, repr=False)
    loss: float
    loss_db: float
    filtering_penalty: float
    equalize: bool
    booster_nf: float | None
    booster_base: float
    booster_nf_db: float


def fold_roadm_terms(config: ROADMConfig, *, is_last_hop: bool) -> RoadmTerms:
    """把一个 ROADM 配置折成 :class:`RoadmTerms`。

    ROADM 常量折叠的唯一定义，统计侧与波形侧共用：

    1. 损耗按端口区分（``drop`` / ``express``），不把 ``filtering_penalty_db``
       并进损耗——后者是噪声域量，不是光功率损耗。
    2. ``equalize_output_power`` 为真时信号不再被 ``loss`` 衰减，代价是引入
       ``NF * h * f * B_i * (loss - 1)`` 的 booster ASE。
    3. 滤波惩罚是只作用于噪声的独立乘子。

    Args:
        config: 该节点的 ROADM 配置。
        is_last_hop: 是否为路径末端跳，决定取 ``drop`` 还是 ``express`` 端口。

    Returns
    -------
        折叠好的常量。
    """
    operation = "drop" if is_last_hop else "express"
    nf_lin = 10 ** (config.booster_noise_figure_db / 10)
    loss_db = config.path_loss_db(operation)
    return RoadmTerms(
        config=config,
        loss=10 ** (loss_db / 10),
        loss_db=loss_db,
        filtering_penalty=10 ** (config.filtering_penalty_db / 10),
        equalize=config.equalize_output_power,
        booster_nf=nf_lin if config.equalize_output_power else None,
        booster_base=nf_lin * _H_PLANCK,
        booster_nf_db=config.booster_noise_figure_db,
    )


#: 一跳上交给累积器的材料：(该跳的 ROADM 常量, 该跳的噪声缓存)。
HopPair = tuple[RoadmTerms, LinkNoiseCache]


def accumulate_service_snr(
    hops: Iterable[HopPair],
    *,
    channel_idx: int,
    launch_power_w: float,
    frequency_hz: float,
    channel_bandwidth_hz: float,
    is_blocked: Callable[[RoadmTerms], bool],
) -> float:
    """沿服务自身路径累积噪声，给出单信道 SNR (dB)。

    被阻断即 ``-inf``，噪声为 0 时给 ``+inf``。

    Args:
        hops: 逐跳的 ``(ROADM 常量, 噪声缓存)``。可以是惰性迭代器：遇到被阻断的
            跳立刻停止拉取，后面那些跳的噪声缓存写回因此不会发生。
        channel_idx: 候选信道（遗留模式下即 ODE 节点下标）。
        launch_power_w: 该业务的发射功率，作为信号功率起点 (W)。
        frequency_hz: 该节点的中心频率 (Hz)。
        channel_bandwidth_hz: 该节点的信道带宽 ``B_i`` (Hz)，booster ASE 用它。
        is_blocked: 判定某一跳的该信道是否被阻断。

    Returns
    -------
        SNR (dB)；被阻断为 ``-inf``，噪声为 0 为 ``+inf``。
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
            total_ase += roadm.booster_base * frequency_hz * channel_bandwidth_hz * (roadm.loss - 1.0)
        else:
            signal_power /= roadm.loss
            total_spm /= roadm.loss
            total_xpm /= roadm.loss
            total_ase /= roadm.loss
        total_spm *= roadm.filtering_penalty
        total_xpm *= roadm.filtering_penalty
        total_ase *= roadm.filtering_penalty

    # SPM 相干累积惩罚
    penalty = total_spans**0.05 if total_spans > 0 else 1.0
    noise = total_spm * penalty + total_xpm + total_ase
    if noise <= 0:
        return float("inf")
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

        遗留模式下等于信道数；逐连接模式下等于连接数（一条连接一个节点，
        与槽数无关）。
        """
        return self.gn_evaluator.n_nodes

    def _is_blocked(self, roadm: ROADMConfig, node_index: int) -> bool:
        """判断某节点（连接）是否被该 ROADM 阻断。

        语义集中在 :meth:`MultiBandISRSGN.roadm_blocks_node`，与波形模型侧共用。
        """
        return self.gn_evaluator.roadm_blocks_node(roadm, node_index)

    def _node_frequency_and_bandwidth(self, channel_idx: int) -> tuple[float, float]:
        """返回该节点（连接）的 ``(中心频率 Hz, 信道带宽 B_i Hz)``。

        必须走 ``gn_evaluator`` 的逐节点数组：灵活栅格下 ``channel_idx`` 索引的是
        连接（ODE 节点），与固定栅格的信道索引不是同一个索引空间。遗留模式下
        这些数组与 ``spectrum_grid`` 逐位相同。
        """
        evaluator = self.gn_evaluator
        return (
            float(evaluator.f_abs_hz[channel_idx]),
            float(evaluator.channel_bandwidth_hz_channels[channel_idx]),
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
        """沿服务自身路径累积噪声，计算单信道 SNR (dB)。

        - 受影响链路从 ``temp_noise_cache`` 读取（含假设分配后的功率谱）。
        - 未受影响链路从 ``LinkState.noise_cache`` 读取（惰性计算并写回）。
        - 信号功率取服务自身的 ``launch_power_w``。

        递推本身交给 :func:`accumulate_service_snr`，本方法只负责取值。传进去的
        是生成器而非列表：被阻断的跳让累积器立刻返回 ``-inf``，生成器随之停止
        拉取，该跳之后的链路因此不会被写回缓存——缓存时机是调用方依赖的语义。
        """
        frequency_hz, channel_bandwidth_hz = self._node_frequency_and_bandwidth(channel_idx)
        last_hop = len(path) - 2

        def _hops() -> Iterable[HopPair]:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_key = _normalize(u, v)
                link_state = network_state.get_link_state(u, v)
                cache = self._get_or_calc_noise_cache(link_key, link_state, temp_noise_cache, fixed_span_length_km)
                roadm = self.network_topology.get_roadm_config(v)
                yield fold_roadm_terms(roadm, is_last_hop=i == last_hop), cache

        return accumulate_service_snr(
            _hops(),
            channel_idx=channel_idx,
            launch_power_w=launch_power_w,
            frequency_hz=frequency_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
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
            # 与 :meth:`_eval_service_snr` 的标量分支同源（同一个 fold_roadm_terms），
            # 只是频率/带宽这里是逐节点数组。
            terms = fold_roadm_terms(roadm, is_last_hop=i == len(path) - 2)
            if terms.booster_nf is not None:
                booster_ase = (
                    terms.booster_base
                    * self.gn_evaluator.f_abs_hz
                    * self.gn_evaluator.channel_bandwidth_hz_channels
                    * (terms.loss - 1.0)
                )
                # 信号为 0 的节点 SNR 必为 -inf（下游按 signal_power > 0 过滤）。
                booster_ase[signal_power <= 0] = 0
                total_ase += booster_ase
            else:
                signal_power /= terms.loss
                total_spm /= terms.loss
                total_xpm /= terms.loss
                total_ase /= terms.loss
            total_spm *= terms.filtering_penalty
            total_xpm *= terms.filtering_penalty
            total_ase *= terms.filtering_penalty
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
