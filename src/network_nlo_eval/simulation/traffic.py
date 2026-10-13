"""业务请求生成器与数据模型."""

import random
from dataclasses import dataclass, field

import numpy as np

from network_nlo_eval.core.flexgrid import bits_per_symbol, slots_for_rate
from network_nlo_eval.core.types import NodeID
from network_nlo_eval.rwa.path_computation import PathCache

#: WP-M1 退化等价的默认调制（见 docs/wp_m1_multislot_alloc.md M-J10）。
DEFAULT_MODULATION = "QPSK"


def slots_for_bit_rate(
    bit_rate_gbps: float,
    *,
    slot_width_hz: float,
    modulation: str = DEFAULT_MODULATION,
    roll_off: float = 0.0,
) -> int:
    """由比特率推导所需槽数（灵活栅格 §2.2 / §3.3）。

    这是 `core.flexgrid.slots_for_rate` 的单位换算外壳：把 Gbps 折成 bit/s，
    把调制名折成每符号比特数。取整语义、``roll_off`` 约束与 ``max(n, 1)``
    全部由 `slots_for_rate` 单点负责，本函数不另立口径。

    Args:
        bit_rate_gbps: 业务比特率 (Gbps)，> 0。
        slot_width_hz: 槽宽 (Hz)，> 0。50 GHz / 12.5 GHz 的场景传 12.5e9。
        modulation: 调制格式名，须在 `MODULATION_FORMATS` 内。
        roll_off: RRC 滚降系数，0 <= roll_off < 1。

    Returns:
        槽数 (int)，>= 1。
    """
    return slots_for_rate(
        bit_rate_gbps * 1e9,
        bits_per_symbol(modulation),
        roll_off,
        slot_width_hz,
    )


@dataclass(slots=True)
class ServiceRequest:
    """网络业务请求的数据模型。

    这是一个不可变的数据结构，代表了一个待处理或正在处理的业务。
    """

    service_id: int  # Field(..., description="业务的唯一ID")
    source_id: NodeID  # Field(..., description="源节点ID (内部ID)")
    destination_id: NodeID  # Field(..., description="目的节点ID (内部ID)")
    arrival_time: float  # Field(..., description="业务到达时间 (仿真时间单位)")
    departure_time: float  # Field(..., description="业务离开时间 (仿真时间单位)")
    bit_rate_gbps: float  # Field(..., description="业务所需比特率 (Gbps)")
    snr_requirement_db: float  # Field(..., description="业务所需的最小 SNR (dB)")

    # 业务的发射功率 (W)，通常由 RWA 决定，这里作为默认值或起点
    launch_power_w: float  # Field(0.001, description="业务发射功率 (W)，约 0 dBm")  # 默认 0 dBm

    # WP-M1：灵活栅格的**槽需求**。占据连续槽区间 [first_slot, first_slot+n_slots)。
    #
    # 默认 1 = 遗留单信道语义。旧分配器 KSPFirstFitAllocator 完全不读这个字段，
    # 故 legacy 场景的数值与行为逐位不变（M-J6）；给默认值而非必填，是为了让
    # 既有的 7 处构造点（含 scripts_shared/compat_run.py）一个字都不用改。
    #
    # 与 bit_rate_gbps 的关系：n_slots 应等于 `slots_for_bit_rate(bit_rate_gbps,
    # slot_width_hz=...)`。这里**不强制**该等式，因为 ServiceRequest 不知道栅格
    # 槽宽（槽宽属于 SpectrumGrid / FlexibleGrid，属于分配器侧的配置）。
    n_slots: int = 1

    def __post_init__(self) -> None:
        """强制槽需求不变式：正整数、非 bool。"""
        if isinstance(self.n_slots, bool) or not isinstance(self.n_slots, int) or self.n_slots < 1:
            raise ValueError(f"n_slots must be a positive integer, got {self.n_slots!r}.")


@dataclass(slots=True)
class AllocatedService(ServiceRequest):
    """已分配业务的数据模型，继承自 ServiceRequest，

    并增加分配结果 (路径、槽区间) 信息。
    """

    # 继承自 ServiceRequest 的 n_slots 即本业务的槽宽度，故此处不重复声明。

    # `path` / `wavelength` / `first_slot` / `slots_per_channel` 必须给默认值：
    # ServiceRequest 末尾的 n_slots 已是带默认值的字段，dataclass 规则禁止
    # 无默认值字段排在其后。既有构造点全是关键字传参，故这不改变任何调用。
    path: list[NodeID] = field(default_factory=list)  # Field(..., description="业务分配的路由路径 (内部节点ID列表)")
    wavelength: int = 0  # Field(..., description="业务分配的波长索引（遗留兼容字段）")

    # WP-M1：槽区间 [first_slot, first_slot+n_slots)。
    first_slot: int = 0

    # 槽/信道的折算率。`None` = **遗留信道粒度**语义（只占 `wavelength` 这一个
    # 信道），是 KSPFirstFitAllocator 造出来的对象的取值；正整数 = 灵活栅格语义
    # （占 [first_slot, first_slot+n_slots) 这段槽区间）。
    #
    # 这个 None/整数二分是刻意的：NetworkState 靠它区分「新分配器写的槽区间」与
    # 「旧分配器写的单信道」，从而让旧路径的写入/释放保持逐位不变。
    slots_per_channel: int | None = None

    class Config:
        frozen = True  # 分配后业务信息通常不应改变

    @property
    def slot_interval(self) -> tuple[int, int]:
        """占据的半开槽区间 ``(first_slot, first_slot + n_slots)``。"""
        return self.first_slot, self.first_slot + self.n_slots

    def __post_init__(self) -> None:
        """强制槽区间不变式，并校验 wavelength 与 first_slot 的自洽性。"""
        super().__post_init__()
        if isinstance(self.first_slot, bool) or not isinstance(self.first_slot, int) or self.first_slot < 0:
            raise ValueError(f"first_slot must be a non-negative integer, got {self.first_slot!r}.")
        if self.slots_per_channel is not None:
            if (
                isinstance(self.slots_per_channel, bool)
                or not isinstance(self.slots_per_channel, int)
                or self.slots_per_channel < 1
            ):
                raise ValueError(
                    f"slots_per_channel must be a positive integer or None, got {self.slots_per_channel!r}."
                )
            expected = self.first_slot // self.slots_per_channel
            if self.wavelength != expected:
                raise ValueError(
                    f"AllocatedService wavelength={self.wavelength} is inconsistent with "
                    f"first_slot={self.first_slot} // slots_per_channel={self.slots_per_channel} = {expected}."
                )


def generate_services(
    path_cache: PathCache,
    service_num: int,
    avg_arrival_interval: float,
    avg_holding_time: float,
    seed: int | None = None,
    slot_width_hz: float | None = None,
    modulation: str = DEFAULT_MODULATION,
    roll_off: float = 0.0,
) -> list[ServiceRequest]:
    """生成一系列模拟网络业务请求。

    Args:
        path_cache: 路径缓存对象，用于获取可用的源/目的节点对。
        service_num: 要生成的业务数量。
        avg_arrival_interval: 业务到达时间的平均间隔 (泊松过程)。
        avg_holding_time: 业务持续时间的平均值 (指数分布)。
        seed: 随机数种子，用于可重现的仿真。
        slot_width_hz: (WP-M1) 槽宽 (Hz)。为 ``None``（**默认**）时每个业务的
            ``n_slots`` 保持 1，即遗留单信道流量，逐位不变；给定槽宽时按
            `slots_for_bit_rate` 由比特率推导真实槽需求。**只影响 n_slots，
            不影响随机数流**，故两种模式下的到达时间/比特率序列完全相同。
        modulation: 推导槽需求时使用的调制格式名。
        roll_off: 推导槽需求时使用的 RRC 滚降系数。

    Returns
    -------
        List[ServiceRequest]: 生成的业务请求列表。
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    services: list[ServiceRequest] = []

    # 获取所有可能的源-目的节点对 (有 KSP 路径的)
    node_pairs = list(path_cache.get_all_ksp_pairs())
    if not node_pairs:
        raise ValueError("No valid node pairs with KSP paths found in PathCache.")

    lambda_rate = 1.0 / avg_arrival_interval  # 到达率
    mu_rate = 1.0 / avg_holding_time  # 持续时间的倒数

    current_time = 0.0

    # 第一个业务的到达时间
    current_time += np.random.exponential(1.0 / lambda_rate)

    for service_id in range(service_num):
        # 随机选择一个源-目的节点对
        # ruff: noqa: S311
        source_id, destination_id = random.choice(node_pairs)

        arrival_time = current_time
        holding_time = np.random.exponential(1.0 / mu_rate)
        departure_time = arrival_time + holding_time

        # 更新下一个业务的到达时间
        next_arrival_interval = np.random.exponential(1.0 / lambda_rate)
        current_time += next_arrival_interval

        # 根据业务类型加权采样比特率
        # 倾向于生成小比特率业务 (权重反比于比特率)
        bit_rate_candidates_gbps = np.arange(100, 501, 10)  # 100 Gbps to 500 Gbps, step 10 Gbps
        weights = 1.0 / bit_rate_candidates_gbps
        weights = weights / np.sum(weights)  # 归一化权重

        bit_rate_gbps = float(np.random.choice(bit_rate_candidates_gbps, p=weights))

        # 根据比特率定义 SNR 需求 (示例)
        if bit_rate_gbps > 400:
            snr_requirement_db = 22.0  # 高比特率需要高SNR
        elif bit_rate_gbps > 300:
            snr_requirement_db = 18.5
        elif bit_rate_gbps > 200:
            snr_requirement_db = 14.5
        else:
            snr_requirement_db = 10.5  # 低比特率需要较低SNR

        # 转换为 dBm，这里为了简化，所有服务初始功率相同，实际中可根据业务类型调整
        launch_power_w = 1e-3  # 0 dBm

        # WP-M1：只在显式给了 slot_width_hz 时推导真实槽需求；否则保持 n_slots=1。
        # 这个分支不消耗任何随机数，故与 legacy 的到达时间/比特率序列逐位一致。
        n_slots = (
            1
            if slot_width_hz is None
            else slots_for_bit_rate(
                bit_rate_gbps,
                slot_width_hz=slot_width_hz,
                modulation=modulation,
                roll_off=roll_off,
            )
        )

        service = ServiceRequest(
            service_id=service_id,
            source_id=source_id,
            destination_id=destination_id,
            arrival_time=arrival_time,
            departure_time=departure_time,
            bit_rate_gbps=bit_rate_gbps,
            snr_requirement_db=snr_requirement_db,
            launch_power_w=launch_power_w,
            n_slots=n_slots,
        )
        services.append(service)

    # 确保服务按到达时间排序，以便仿真引擎正确处理
    services.sort(key=lambda s: s.arrival_time)
    return services
