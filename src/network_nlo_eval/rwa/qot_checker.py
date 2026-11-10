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
from network_nlo_eval.models.power_conventions import (
    ase_polarization_scale as convention_ase_polarization_scale,
)
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)
from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    LinkValidity,
    LinkValidityPolicy,
    LinkValidityProvider,
)
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

#: 取某一跳有效性报告的回调；返回 ``None`` 表示该跳未接入有效性。
ValidityOfHop = Callable[[RoadmTerms, LinkNoiseCache], "LinkValidity | None"]


@dataclass(frozen=True)
class ServiceQoT:
    """一条业务路径的物理 SNR 与有效性聚合。

    物理 SNR 与接入前逐位一致；``margin_db`` / ``valid`` 是新增的网络级口径。
    默认策略（``off``）下 :meth:`effective_snr_db` 原样返回 ``snr_db``，不做任何
    算术，因此默认判定逐位不变。

    Attributes
    ----------
        snr_db: 沿业务自身路径累积的物理 SNR (dB)；被阻断为 ``-inf``，
            噪声为零为 ``+inf``。
        margin_db: 路径上各链路建议 margin 之和 (dB)，只累计有效链路。
        soft_margin_db: 路径上各链路倾斜软 margin 之和 (dB)，只累计有效链路；
            只有 ``mode="soft"`` 会扣它。
        valid: 路径上所有已接入链路是否都判该信道有效。
        invalid_links: 判该信道无效的链路键（按首次出现顺序去重）。
        dense_grid_overlap: 路径上已接入的链路是否任一是密集栅格。默认
            ``False``；只有 ``dense_grid_action="reject"`` 会用它扣可用性。
        raman_auto_under_resolved: 路径上已接入的链路是否任一命中 auto-Raman
            分辨率不足。默认 ``False``；只有 ``raman_resolution_action="reject"``
            会用它扣可用性。
    """

    snr_db: float
    margin_db: float = 0.0
    soft_margin_db: float = 0.0
    valid: bool = True
    invalid_links: tuple[LinkKey, ...] = ()
    dense_grid_overlap: bool = False
    raman_auto_under_resolved: bool = False

    def effective_snr_db(self, policy: LinkValidityPolicy | None = None) -> float:
        """Apply the validity policy to the scalar SNR.

        Args:
            policy: 策略；``None`` 等价于 ``"off"``（返回物理 SNR，不做算术）。

        Returns
        -------
            用于阈值比较的 SNR (dB)。
        """
        return float(
            (policy or LinkValidityPolicy()).apply(
                np.array([self.snr_db], dtype=np.float64),
                np.array([self.valid], dtype=bool),
                np.array([self.margin_db], dtype=np.float64),
                np.array([self.soft_margin_db], dtype=np.float64),
                dense_grid_overlap=self.dense_grid_overlap,
                raman_auto_under_resolved=self.raman_auto_under_resolved,
            )[0]
        )


def accumulate_service_snr(
    hops: Iterable[HopPair],
    *,
    channel_idx: int,
    launch_power_w: float,
    frequency_hz: float,
    channel_bandwidth_hz: float,
    is_blocked: Callable[[RoadmTerms], bool],
    validity_of: ValidityOfHop | None = None,
    validity_policy: LinkValidityPolicy | None = None,
    ase_polarization_scale: float = 1.0,
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
        validity_of: 可选回调，返回该跳的链路有效性。``None`` 时不读任何报告，
            物理 SNR 与历史逐位一致。
        validity_policy: 可选策略，默认 ``"off"``。只有显式给出非 ``off`` 策略时
            返回值才会被 margin / 无效标记修正。
        ase_polarization_scale: 共享层新产生 ROADM booster ASE 的偏振缩放。
            ``"total"`` 评估器为 ``1.0``（默认）；``"per_pol"`` 评估器为
            ``0.5``，使累加器与链路噪声缓存处于同一坐标。默认值不产生任何
            算术，因此 total 路径逐位不变。

    Returns
    -------
        SNR (dB)；被阻断为 ``-inf``，噪声为 0 为 ``+inf``。策略为 ``off`` 时与
        :func:`accumulate_service_snr_detail` 的物理值逐位相同。
    """
    return accumulate_service_snr_detail(
        hops,
        channel_idx=channel_idx,
        launch_power_w=launch_power_w,
        frequency_hz=frequency_hz,
        channel_bandwidth_hz=channel_bandwidth_hz,
        is_blocked=is_blocked,
        validity_of=validity_of,
        ase_polarization_scale=ase_polarization_scale,
    ).effective_snr_db(validity_policy)


def accumulate_service_snr_detail(
    hops: Iterable[HopPair],
    *,
    channel_idx: int,
    launch_power_w: float,
    frequency_hz: float,
    channel_bandwidth_hz: float,
    is_blocked: Callable[[RoadmTerms], bool],
    validity_of: ValidityOfHop | None = None,
    ase_polarization_scale: float = 1.0,
) -> ServiceQoT:
    """沿服务自身路径累积噪声，返回物理 SNR 与有效性聚合。

    与 :func:`accumulate_service_snr` 是同一套算术：浮点求和顺序、阻断时机、
    ``+inf``/``-inf`` 约定都没有改动，只是额外收集各跳的 margin 与有效性。

    Args:
        hops: 逐跳的 ``(ROADM 常量, 噪声缓存)``，可为惰性迭代器。
        channel_idx: 候选信道（ODE 节点下标）。
        launch_power_w: 该业务的发射功率 (W)。
        frequency_hz: 该节点的中心频率 (Hz)。
        channel_bandwidth_hz: 该节点的信道带宽 ``B_i`` (Hz)。
        is_blocked: 判定某一跳的该信道是否被阻断。
        validity_of: 可选回调；被阻断的跳不会触发回调。
        ase_polarization_scale: 共享层新产生 ROADM booster ASE 的偏振缩放。
            ``"total"`` 坐标下为 ``1.0``（默认，不做任何算术）；``"per_pol"``
            坐标下为 ``0.5``。它只缩放 booster 项，不影响从缓存读出的 SPM/XPM/
            ASE——后者已经是评估器坐标。

    Returns
    -------
        物理 SNR 与路径级有效性聚合。``margin_db`` 与 ``soft_margin_db`` 都只
        累计有效链路；前者供 ``degrade``，后者供 ``soft``。
    """
    total_spm = 0.0
    total_xpm = 0.0
    total_ase = 0.0
    signal_power = launch_power_w
    total_spans = 0
    total_margin_db = 0.0
    total_soft_margin_db = 0.0
    valid = True
    invalid_links: list[LinkKey] = []
    dense_grid_overlap = False
    raman_auto_under_resolved = False

    for roadm, cache in hops:
        total_spm += cache.spm.item(channel_idx)
        total_xpm += cache.xpm.item(channel_idx)
        total_ase += cache.ase.item(channel_idx)
        total_spans += cache.span_count

        if is_blocked(roadm):
            return ServiceQoT(snr_db=float("-inf"), valid=False)

        # 有效性只在未阻断的跳上读取：阻断信道本来就没有物理量可判。
        if validity_of is not None:
            link_validity = validity_of(roadm, cache)
            if link_validity is not None:
                # 密集栅格是整条占用谱的属性，任一跳越界就整条路径越界。
                dense_grid_overlap = dense_grid_overlap or bool(link_validity.dense_grid_overlap)
                raman_auto_under_resolved = raman_auto_under_resolved or bool(
                    link_validity.raman_auto_under_resolved
                )
                if link_validity.channel_valid(channel_idx):
                    total_margin_db += link_validity.margin_at(channel_idx)
                    total_soft_margin_db += link_validity.soft_margin_at(channel_idx)
                else:
                    valid = False
                    if link_validity.link_key is not None and link_validity.link_key not in invalid_links:
                        invalid_links.append(link_validity.link_key)

        if roadm.equalize:
            booster_ase = roadm.booster_base * frequency_hz * channel_bandwidth_hz * (roadm.loss - 1.0)
            # 默认 total 路径 scale == 1.0，不做乘法，逐位保持历史结果。
            if ase_polarization_scale != 1.0:
                booster_ase *= ase_polarization_scale
            total_ase += booster_ase
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
        snr_db = float("inf")
    else:
        snr_db = 10.0 * math.log10(signal_power / noise)
    return ServiceQoT(
        snr_db=snr_db,
        margin_db=total_margin_db,
        soft_margin_db=total_soft_margin_db,
        valid=valid,
        invalid_links=tuple(invalid_links),
        dense_grid_overlap=dense_grid_overlap,
        raman_auto_under_resolved=raman_auto_under_resolved,
    )


class QoTValidator:
    """传输质量验证器。

    RWA 算法使用它来判断一个候选分配方案是否满足 SNR 要求。

    核心策略：**临时副本 + 局部缓存**，不修改 LinkState，无需回滚。

    网络级有效性接入（任务 H）是**可选**的：只有同时给出非 ``off`` 的
    ``validity_policy`` 与 ``validity_provider`` 时才会在链路噪声缓存上附带
    :class:`~network_nlo_eval.rwa.validity.LinkValidity`，并按策略修正 SNR。
    默认两者都不给，判定与接入前逐位一致。``mode="soft"`` 只扣倾斜软 margin，
    不改变 ``valid`` / ``flagged``，也不扣 ``margin_db``。
    ``dense_grid_action="reject"`` 额外把密集栅格信道判为不可用；默认
    ``"flag"`` 只记录标志，判定不变。

    入纤功率口径（任务 O2 / X）：``NetworkState.launch_power_profile_w``、
    ``ServiceRequest.launch_power_w`` 与 provider 的入参都按构造时的
    ``launch_power_convention`` 解释（默认 ``"total"``，即网络参考面）。
    内部在送进 ``gn_evaluator`` / ``assemble_link_noise`` 前**唯一一次**换算到
    评估器口径（``eval_power_convention``），因此 ``LinkNoiseCache`` 与
    ``accumulate_service_snr`` 的坐标始终是评估器口径：默认 ``total`` 时就是
    物理总功率；``per_pol`` 评估器时则是每偏振功率。同一物理总功率下两种坐标
    给出等价 SNR（信号与噪声同比缩放）。

    ``per_pol`` 评估器必须同时声明 ``ase_polarization_scale=0.5``：链路噪声
    缓存里的 EDFA ASE 由评估器产生，而 ROADM booster ASE 由本验证器累加，
    两者都用同一个 scale 才会落在同一坐标。``total`` 评估器取默认 ``1.0``，
    不做任何乘法。

    缓存口径溯源：``LinkNoiseCache.launch_power_convention`` 记录该缓存是在哪个
    评估器坐标下算出的。同一 ``NetworkState`` 上换用不同口径的验证器时，读取
    已知口径且不匹配的缓存会触发重算并替换，绝不把 ``total`` 与 ``per_pol``
    的噪声混在一条路径里。``None``（旧缓存）按既有行为直接复用，下次写入时
    补上口径。
    """

    def __init__(
        self,
        gn_evaluator: MultiBandISRSGN,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        fixed_span_length_km: float | None = None,
        *,
        launch_power_convention: str = "total",
        validity_policy: LinkValidityPolicy | None = None,
        validity_provider: LinkValidityProvider | None = None,
    ):
        """初始化验证器。

        Args:
            gn_evaluator: 链路噪声评估器。
            network_topology: 网络拓扑。
            spectrum_grid: 频谱网格。
            fixed_span_length_km: 固定跨长 (km)；``None`` 时用光纤配置。
            launch_power_convention: ``NetworkState`` / ``ServiceRequest`` 与
                provider 入参的功率口径，``"total"``（默认）或 ``"per_pol"``。
                内部会换算到评估器口径；默认 ``total`` 时不产生任何算术。
            validity_policy: 可选的有效性策略。默认 ``"off"``。
            validity_provider: 可选的有效性提供者。策略非 ``off`` 时必填；
                策略为 ``off`` 时给出会报错，避免"配了但没生效"的死配置。

        Raises
        ------
            ValueError: 策略与提供者的组合不一致，或 provider 的
                ``input_power_convention`` 与评估器口径不一致，或评估器的
                ``ase_polarization_scale`` 非法/与评估器口径不匹配。
        """
        policy = validity_policy or LinkValidityPolicy()
        if policy.enabled and validity_provider is None:
            raise ValueError(
                f"validity_policy mode={policy.mode} requires a validity_provider; "
                "pass an H3LinkValidityProvider or use mode='off'."
            )
        if not policy.enabled and validity_provider is not None:
            raise ValueError(
                "validity_provider was given with validity_policy mode='off'; "
                "choose an explicit mode ('flag', 'reject', 'degrade' or 'soft') to enable it."
            )
        self.gn_evaluator = gn_evaluator
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        self._fixed_span_length_km = fixed_span_length_km
        self.launch_power_convention = resolve_launch_power_convention(launch_power_convention)
        #: 评估器（``gn_evaluator`` / ``assemble_link_noise`` / ``LinkNoiseCache``）
        #: 的功率坐标。换算只在内部入口做一次；它可以是 ``"total"``（默认）或
        #: ``"per_pol"``。
        self.eval_power_convention = resolve_launch_power_convention(
            str(getattr(gn_evaluator, "launch_power_convention", "total"))
        )
        #: 共享层新产生 ASE（ROADM booster）的偏振缩放。评估器若声明 per_pol
        #: 坐标，必须同时给出 0.5，才能与缓存里的每偏振噪声同坐标。
        scale = float(getattr(gn_evaluator, "ase_polarization_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                f"gn_evaluator.ase_polarization_scale must be finite and positive, got {scale!r}."
            )
        expected_scale = convention_ase_polarization_scale(self.eval_power_convention)
        if scale != expected_scale:
            raise ValueError(
                "gn_evaluator.ase_polarization_scale must be "
                f"{expected_scale:g} for launch_power_convention="
                f"{self.eval_power_convention!r}, got {scale:g}. The shared ROADM booster ASE "
                "would otherwise be accumulated in a different coordinate than the link-noise "
                "cache."
            )
        self.ase_polarization_scale = scale
        provider_convention = getattr(validity_provider, "input_power_convention", None)
        if provider_convention is not None:
            provider_convention = resolve_launch_power_convention(str(provider_convention))
            if provider_convention != self.eval_power_convention:
                raise ValueError(
                    "validity_provider input_power_convention="
                    f"{provider_convention!r} must equal the evaluator convention "
                    f"{self.eval_power_convention!r}; QoTValidator passes evaluator-coordinate power "
                    "to the provider, so a mismatch would convert the spectrum twice."
                )
        self.validity_policy = policy
        self.validity_provider = validity_provider
        #: 最近一次 ``verify_allocation`` 里每条受影响链路的有效性报告。
        #: 每次调用开头清空，因此不会把上一次的状态带到下一次。
        self.last_link_validity: dict[LinkKey, LinkValidity] = {}

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
        self.last_link_validity.clear()

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
        """计算单条链路在给定功率谱下的噪声缓存。

        接入有效性时（策略非 ``off``），先由 ``validity_provider`` 用同一份功率谱
        产出链路报告，再把它附加到缓存上。报告**不参与**噪声算术。

        ``power_profile`` 按验证器构造时声明的口径解释，进入 provider /
        ``assemble_link_noise`` 前换算到评估器口径（默认 ``total``，此时
        ``to_convention`` 返回原数组，不做算术）。写出的缓存总是带上
        ``eval_power_convention`` 标签。
        """
        u, v = link_key
        fiber_config = self.network_topology.get_fiber_config(u, v)
        edfa_config = self.network_topology.get_link_edfa_config(u, v)
        evaluator = self.gn_evaluator.for_fiber(fiber_config)
        power_eval = to_convention(power_profile, self.launch_power_convention, self.eval_power_convention)

        link_total_length_m = fiber_config.length_km * 1000
        span_length_m = (fixed_span_length_km or fiber_config.max_span_length_km) * 1000
        num_full_spans = int(link_total_length_m // span_length_m)
        remainder_length_m = link_total_length_m % span_length_m

        validity: LinkValidity | None = None
        if self.validity_policy.enabled:
            provider = self.validity_provider
            if provider is None:  # pragma: no cover - __init__ 已拒绝该组合
                raise RuntimeError("validity_policy is enabled but no provider is configured.")
            validity = provider(
                link_key=link_key,
                fiber_config=fiber_config,
                edfa_config=edfa_config,
                power_profile_w=power_eval,
                span_length_m=span_length_m,
            )
            self.last_link_validity[link_key] = validity

        return assemble_link_noise(
            evaluator=evaluator,
            edfa_config=edfa_config,
            power_profile=power_eval,
            span_length_m=span_length_m,
            num_full_spans=num_full_spans,
            remainder_length_m=remainder_length_m,
            num_channels=self.num_nodes,
            validity=validity,
            launch_power_convention=self.eval_power_convention,
        )

    def _get_or_calc_noise_cache(
        self,
        link_key: LinkKey,
        link_state,
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        fixed_span_length_km: float | None,
    ) -> LinkNoiseCache:
        """获取链路噪声缓存：优先从临时缓存，否则从 LinkState（惰性计算）。

        ``LinkState`` 缓存会记录写入时的评估器口径。口径已知且与当前验证器
        不一致时**不得复用**：按当前评估器重算并替换。``None`` 是旧缓存
        （未记录口径），保持既有复用行为，等下一次写入时补上口径。
        """
        if link_key in temp_noise_cache:
            return temp_noise_cache[link_key]

        cached = link_state.noise_cache
        if cached is not None:
            convention = cached.launch_power_convention
            if convention is None or convention == self.eval_power_convention:
                return cached

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
          若缓存记录了已知口径且与当前评估器不一致，会先重算再使用。
        - 信号功率取服务自身的 ``launch_power_w``。

        递推本身交给 :func:`accumulate_service_snr_detail`，本方法只负责取值与策略。
        传进去的是生成器而非列表：被阻断的跳让累积器立刻返回 ``-inf``，生成器随之
        停止拉取，该跳之后的链路因此不会被写回缓存——缓存时机是调用方依赖的语义。

        策略为 ``off`` 时返回值就是物理 SNR，与接入前逐位一致；非 ``off`` 时按
        :class:`~network_nlo_eval.rwa.validity.LinkValidityPolicy` 扣除 margin 或
        把无效信道压成 ``-inf``。

        信号功率按构造时声明的口径解释，交给累积器前换算到评估器口径。
        """
        frequency_hz, channel_bandwidth_hz = self._node_frequency_and_bandwidth(channel_idx)
        launch_power_eval = to_convention(launch_power_w, self.launch_power_convention, self.eval_power_convention)
        last_hop = len(path) - 2

        def _hops() -> Iterable[HopPair]:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_key = _normalize(u, v)
                link_state = network_state.get_link_state(u, v)
                cache = self._get_or_calc_noise_cache(link_key, link_state, temp_noise_cache, fixed_span_length_km)
                roadm = self.network_topology.get_roadm_config(v)
                yield fold_roadm_terms(roadm, is_last_hop=i == last_hop), cache

        qot = accumulate_service_snr_detail(
            _hops(),
            channel_idx=channel_idx,
            launch_power_w=launch_power_eval,
            frequency_hz=frequency_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
            is_blocked=lambda terms: self._is_blocked(terms.config, channel_idx),
            validity_of=self._validity_of if self.validity_policy.enabled else None,
            ase_polarization_scale=self.ase_polarization_scale,
        )
        return qot.effective_snr_db(self.validity_policy)

    @staticmethod
    def _validity_of(_roadm: RoadmTerms, cache: LinkNoiseCache) -> LinkValidity | None:
        """取某一跳噪声缓存上附带的有效性报告。

        报告与缓存同生命周期：``LinkState`` 的缓存在占用谱变化时失效，临时缓存在
        一次 ``verify_allocation`` 内构造，因此这里读到的报告永远对应当前功率谱。
        """
        return cache.validity

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
        """构建新业务路径上各信道的 SNR (dB) 数组，用于返回值。

        策略为 ``off`` 时返回物理 SNR，与接入前逐位一致；非 ``off`` 时用各跳链路
        有效性做"最坏链路 + margin 相加"的聚合，再按策略修正。

        功率谱按构造时声明的口径解释，换算到评估器口径后再参与算术；默认
        ``total`` 时 ``to_convention`` 返回原数组，逐步位不变。
        """
        total_spm = np.zeros(self.num_nodes, dtype=np.float64)
        total_xpm = np.zeros(self.num_nodes, dtype=np.float64)
        total_ase = np.zeros(self.num_nodes, dtype=np.float64)
        total_spans = 0
        link_validities: list[LinkValidity] = []
        u0, v0 = path[0], path[1]
        link_key0 = _normalize(u0, v0)
        if link_key0 in temp_power_profiles:
            signal_power = temp_power_profiles[link_key0].copy()
        else:
            signal_power = network_state.get_link_state(u0, v0).launch_power_profile_w.copy()
        signal_power = to_convention(signal_power, self.launch_power_convention, self.eval_power_convention)

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
            if self.validity_policy.enabled and cache.validity is not None:
                link_validities.append(cache.validity)

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
                # total 坐标 scale == 1.0：不做乘法，保持历史逐位结果。
                if self.ase_polarization_scale != 1.0:
                    booster_ase = booster_ase * self.ase_polarization_scale
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

        if not link_validities:
            return snrs_db
        verdict = ChannelValidity.combine(link_validities, self.num_nodes)
        return np.asarray(verdict.adjusted_snr_db(snrs_db, self.validity_policy), dtype=np.float64)


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
