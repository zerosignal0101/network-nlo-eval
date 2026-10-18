"""业务请求生成器与数据模型."""

import random
from dataclasses import dataclass, field

import numpy as np

from network_nlo_eval.core.flexgrid import bits_per_symbol, slots_for_rate
from network_nlo_eval.core.types import NodeID
from network_nlo_eval.rwa.path_computation import PathCache

#: 由比特率推导槽数时使用的默认调制格式。
DEFAULT_MODULATION = "QPSK"


def slots_for_bit_rate(
    bit_rate_gbps: float,
    *,
    slot_width_hz: float,
    modulation: str = DEFAULT_MODULATION,
    roll_off: float = 0.0,
) -> int:
    """由比特率推导所需槽数。

    只是 :func:`~network_nlo_eval.core.flexgrid.slots_for_rate` 的单位换算外壳：
    Gbps -> bit/s，调制名 -> 每符号比特数。取整语义与参数校验都在后者。

    Args:
        bit_rate_gbps: 业务比特率 (Gbps)，> 0。
        slot_width_hz: 槽宽 (Hz)，> 0。
        modulation: 调制格式名，须在 ``MODULATION_FORMATS`` 内。
        roll_off: RRC 滚降系数，0 <= roll_off < 1。

    Returns
    -------
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

    # 槽需求：占据连续槽区间 [first_slot, first_slot + n_slots)。默认 1 = 遗留
    # 单信道语义，旧分配器不读该字段。与 bit_rate_gbps 的关系由调用方用
    # slots_for_bit_rate 保证，本类不知道槽宽。
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

    # 以下字段必须带默认值：n_slots 已带默认值，dataclass 不允许无默认值字段排在
    # 其后。既有构造点全部使用关键字传参。
    path: list[NodeID] = field(default_factory=list)
    wavelength: int = 0  # 锚信道索引（遗留兼容字段）

    first_slot: int = 0  # 槽区间起点

    # None = 遗留信道粒度（只占 wavelength 这一个信道）；正整数 = 灵活栅格槽区间
    # 语义。NetworkState 靠这个二分区分新旧两条写入/释放路径。
    slots_per_channel: int | None = None

    class Config:
        frozen = True  # 分配后业务信息通常不应改变

    @property
    def slot_interval(self) -> tuple[int, int]:
        """占据的半开槽区间 ``(first_slot, first_slot + n_slots)``。"""
        return self.first_slot, self.first_slot + self.n_slots

    def __post_init__(self) -> None:
        """强制槽区间不变式，并校验 wavelength 与 first_slot 的自洽性。"""
        # 本类是 slots=True 的 dataclass，零参 super() 不可用，必须显式点名基类。
        ServiceRequest.__post_init__(self)
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
        slot_width_hz: 槽宽 (Hz)。为 ``None`` 时每个业务的 ``n_slots`` 保持 1；
            给定后按 :func:`slots_for_bit_rate` 由比特率推导。只影响 n_slots，
            不消耗随机数，故到达时间与比特率序列不变。
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

        # 未给槽宽时保持 n_slots=1；该分支不消耗随机数。
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
