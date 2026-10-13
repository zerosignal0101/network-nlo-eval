"""管理网络的动态资源状态，如波长占用和链路功率谱。"""

from dataclasses import dataclass, field

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import (
    LinkKey,
    NDArrayBool,
    NDArrayFloat,
    NDArrayInt,
    NodeID,
)
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.traffic import AllocatedService  # 依赖 ServiceRequest


@dataclass
class LinkNoiseCache:
    """缓存单条链路上所有信道的噪声方差 (W) 和物理跨段数。"""

    spm: NDArrayFloat  # 每信道 SPM 噪声方差 (W)
    xpm: NDArrayFloat  # 每信道 XPM 噪声方差 (W)
    ase: NDArrayFloat  # 每信道 ASE 噪声方差 (W)
    span_count: int  # 链路上的物理跨段数


@dataclass
class LinkState:
    """跟踪单条物理链路的动态状态。

    LinkKey 应始终是规范化的 (min(u,v), max(u,v))。
    """

    link_key: LinkKey  # 链路的规范化键 (u_idx, v_idx)
    length_m: float  # 链路长度 (m)

    # 核心数据结构：每个信道的占用标记和发射功率
    # True 表示该信道被占用
    occupied_channels: NDArrayBool
    # 该信道上的发射功率 (W)，0 表示未占用
    # 这不是接收功率，而是该链路的发射端注入的功率
    launch_power_profile_w: NDArrayFloat
    # 记录每个信道上分配的服务ID，-1 表示未分配
    allocated_service_ids: NDArrayInt

    # 噪声缓存：惰性计算，功率谱改变时失效
    _noise_cache: LinkNoiseCache | None = field(default=None, init=False, repr=False)

    # WP-M1：灵活栅格的槽粒度占用。长度 = num_channels * slots_per_channel，
    # 仅由 KSPFirstFitFlexAllocator 使用；遗留单信道分配器（slots_per_channel=1）
    # 完全不读这两个数组，故 legacy 行为逐位不变。
    slots_per_channel: int = 1
    occupied_slots: NDArrayBool | None = None
    allocated_service_ids_slots: NDArrayInt | None = None

    def __post_init__(self):
        """确保数组在初始化时是副本，防止外部修改。"""
        self.occupied_channels = self.occupied_channels.copy()
        self.launch_power_profile_w = self.launch_power_profile_w.copy()
        self.allocated_service_ids = self.allocated_service_ids.copy()

        if isinstance(self.slots_per_channel, bool) or self.slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be a positive integer, got {self.slots_per_channel!r}.")
        if self.occupied_slots is None:
            self.occupied_slots = np.zeros(
                len(self.occupied_channels) * self.slots_per_channel, dtype=bool
            )
        else:
            self.occupied_slots = self.occupied_slots.copy()
        if self.allocated_service_ids_slots is None:
            self.allocated_service_ids_slots = np.full(len(self.occupied_slots), -1, dtype=np.int_)
        else:
            self.allocated_service_ids_slots = self.allocated_service_ids_slots.copy()

    @property
    def noise_cache(self) -> LinkNoiseCache | None:
        """获取链路噪声缓存，若不存在则由外部计算后赋值。

        当 _noise_cache 为 None 时，调用方需自行计算并赋值。
        """
        return self._noise_cache

    @noise_cache.setter
    def noise_cache(self, value: LinkNoiseCache) -> None:
        self._noise_cache = value

    def invalidate_noise_cache(self) -> None:
        """使噪声缓存失效，在功率谱改变后调用。"""
        self._noise_cache = None

    def allocate(self, channel_idx: int, launch_power_w: float, service_id: int) -> None:
        """在指定信道上分配资源。

        Args:
            channel_idx: 待分配的信道索引。
            launch_power_w: 该信道上的发射功率 (W)。
            service_id: 占用该信道的服务ID。
        """
        if self.occupied_channels[channel_idx]:
            raise ValueError(f"Channel {channel_idx} on link {self.link_key} is already occupied.")
        self.occupied_channels[channel_idx] = True
        self.launch_power_profile_w[channel_idx] = launch_power_w
        self.allocated_service_ids[channel_idx] = service_id
        self.invalidate_noise_cache()

    def release(self, channel_idx: int) -> None:
        """释放指定信道上的资源。

        Args:
            channel_idx: 待释放的信道索引。
        """
        if not self.occupied_channels[channel_idx]:
            return  # 静默处理重复释放
        self.occupied_channels[channel_idx] = False
        self.launch_power_profile_w[channel_idx] = 0.0
        self.allocated_service_ids[channel_idx] = -1  # 标记为未分配
        self.invalidate_noise_cache()

    # ------------------------------------------------------------------
    # WP-M1：槽区间分配 / 释放
    # ------------------------------------------------------------------
    def allocate_slot_range(
        self, first_slot: int, n_slots: int, launch_power_w: float, service_id: int
    ) -> None:
        """在连续槽区间 ``[first_slot, first_slot + n_slots)`` 上分配资源。

        **物理仍逐信道路入功率**（冻结内核 `_calc_span_noise_and_power_jit` 吃的是
        每信道标量功率向量），因此入纤功率只写在**锚信道**
        ``first_slot // slots_per_channel`` 上，与 RWA 层 QoT 验证所用的信道索引
        严格一致。槽区间是**分配层的真源**：它决定窗口是否连续、与哪些槽冲突、
        以及释放时该清掉哪一段。

        槽占用按**信道对齐**展开：与窗口相交的每个信道，其全部子槽一并占用。
        理由是退化等价（M-J10）——``slots_per_channel=4`` 且 ``n_slots=1`` 时，
        单槽窗口必须占满它所在的整个信道，否则下一个 1 槽请求会落到同一信道的
        另一个子槽上，而旧分配器已把该信道标为占用，两者会打架。

        Args:
            first_slot: 起始槽 (0-based, 闭)。
            n_slots: 槽数，>= 1。
            launch_power_w: 发射功率 (W)，写入锚信道。
            service_id: 占用该区间的服务ID。

        Raises:
            ValueError: ``n_slots < 1``、槽区间越界，或区间内已有槽被占用。
        """
        if n_slots < 1:
            raise ValueError(f"n_slots must be >= 1, got {n_slots}.")
        per_channel = self.slots_per_channel
        num_slots = len(self.occupied_slots)
        if first_slot < 0 or first_slot + n_slots > num_slots:
            raise ValueError(
                f"Slot range [{first_slot}, {first_slot + n_slots}) out of bounds for "
                f"{num_slots} slots on link {self.link_key}."
            )
        window = slice(first_slot, first_slot + n_slots)
        if self.occupied_slots[window].any():
            raise ValueError(
                f"Slot range [{first_slot}, {first_slot + n_slots}) on link {self.link_key} "
                f"is already occupied."
            )

        self.occupied_slots[window] = True
        self.allocated_service_ids_slots[window] = service_id

        # 信道对齐展开：首尾相交信道的全部子槽一并占用。
        first_channel = first_slot // per_channel
        last_channel = (first_slot + n_slots - 1) // per_channel
        aligned = slice(first_channel * per_channel, (last_channel + 1) * per_channel)
        self.occupied_slots[aligned] = True
        self.allocated_service_ids_slots[aligned] = service_id

        # 物理面：只有锚信道注入功率（与 QoT 验证所用信道一致）。
        self.occupied_channels[first_channel] = True
        self.launch_power_profile_w[first_channel] = launch_power_w
        self.allocated_service_ids[first_channel] = service_id
        self.invalidate_noise_cache()

    def release_slot_range(self, first_slot: int, n_slots: int) -> None:
        """释放连续槽区间 ``[first_slot, first_slot + n_slots)`` 上的资源。

        Args:
            first_slot: 起始槽 (0-based, 闭)。
            n_slots: 槽数，>= 1。
        """
        if n_slots < 1:
            return  # 静默处理非法/重复释放
        per_channel = self.slots_per_channel
        num_slots = len(self.occupied_slots)
        if first_slot < 0 or first_slot + n_slots > num_slots:
            return  # 静默处理越界的重复释放

        # 信道对齐展开的区间与业务自己占用的区间取并集后一起清。
        first_channel = first_slot // per_channel
        last_channel = (first_slot + n_slots - 1) // per_channel
        lo = min(first_slot, first_channel * per_channel)
        hi = max(first_slot + n_slots, (last_channel + 1) * per_channel)
        window = slice(lo, hi)

        self.occupied_slots[window] = False
        self.allocated_service_ids_slots[window] = -1

        # 物理面：窗口覆盖到的信道，若其全部子槽都已空闲则一并归零。
        for channel_idx in range(first_channel, last_channel + 1):
            span = slice(channel_idx * per_channel, (channel_idx + 1) * per_channel)
            if not self.occupied_slots[span].any():
                self.occupied_channels[channel_idx] = False
                self.launch_power_profile_w[channel_idx] = 0.0
                self.allocated_service_ids[channel_idx] = -1
        self.invalidate_noise_cache()


class NetworkState:
    """管理整个网络的动态运行状态。

    包含所有链路的当前占用情况和功率谱，以及所有已分配的业务详情。
    """

    def __init__(
        self,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        slots_per_channel: int = 1,
    ):
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        # WP-M1：槽粒度折算率。默认 1 = 遗留纯信道语义，LinkState 的槽数组长度
        # 恰好等于 num_channels 且无人读取，legacy 逐位不变。
        if isinstance(slots_per_channel, bool) or slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be a positive integer, got {slots_per_channel!r}.")
        self.slots_per_channel = slots_per_channel
        self._link_states: dict[LinkKey, LinkState] = {}
        self._allocated_services: dict[int, AllocatedService] = {}  # service_id -> AllocatedService

        # 初始化所有链路状态
        for link_key in network_topology.get_all_internal_links():
            u, v = link_key
            fiber_config = network_topology.get_fiber_config(u, v)
            self._link_states[link_key] = LinkState(
                link_key=link_key,
                length_m=fiber_config.length_km * 1000,
                occupied_channels=np.zeros(spectrum_grid.num_channels, dtype=bool),
                launch_power_profile_w=np.zeros(spectrum_grid.num_channels, dtype=np.float64),
                allocated_service_ids=np.full(spectrum_grid.num_channels, -1, dtype=np.int_),
                slots_per_channel=slots_per_channel,
            )

    def get_link_state(self, u_idx: NodeID, v_idx: NodeID) -> LinkState:
        """获取指定链路的当前状态."""
        link_key = tuple((u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx))
        return self._link_states[link_key]

    def get_all_link_states(self) -> dict[LinkKey, LinkState]:
        """获取所有链路的当前状态字典."""
        return self._link_states

    def get_allocated_service(self, service_id: int) -> AllocatedService | None:
        """获取指定服务ID的 AllocatedService 对象."""
        return self._allocated_services.get(service_id)

    def get_all_allocated_services(self) -> dict[int, AllocatedService]:
        """获取所有已分配服务."""
        return self._allocated_services

    def allocate_service(self, service: AllocatedService) -> None:
        """在网络状态中正式分配一个服务。

        假定 QoT 验证和波长/路径选择已完成。

        槽语义分流：`service.slots_per_channel is None` = 遗留单信道对象
        （KSPFirstFitAllocator 造的），走**原逐位不变**的信道分支；
        否则按槽区间分配。
        """
        # 确保服务ID是唯一的
        if service.service_id in self._allocated_services:
            # 如果是重新分配 (例如，动态调整路由)，则先释放旧的
            self.release_service(self._allocated_services[service.service_id])

        for i in range(len(service.path) - 1):
            u, v = service.path[i], service.path[i + 1]
            link_state = self.get_link_state(u, v)
            if service.slots_per_channel is None:
                link_state.allocate(service.wavelength, service.launch_power_w, service.service_id)
            else:
                link_state.allocate_slot_range(
                    service.first_slot,
                    service.n_slots,
                    service.launch_power_w,
                    service.service_id,
                )
        self._allocated_services[service.service_id] = service

    def release_service(self, service: AllocatedService) -> None:
        """从网络状态中释放一个服务（槽区间语义，与 allocate_service 对称）。"""
        if service.service_id not in self._allocated_services:
            # warnings.warn(f"Service {service.service_id} not found for release.")
            return  # 静默处理重复释放或未分配服务

        for i in range(len(service.path) - 1):
            u, v = service.path[i], service.path[i + 1]
            link_state = self.get_link_state(u, v)
            if service.slots_per_channel is None:
                link_state.release(service.wavelength)
            else:
                link_state.release_slot_range(service.first_slot, service.n_slots)
        del self._allocated_services[service.service_id]

    def occupied_slots_on_path(self, path: list[NodeID]) -> NDArrayBool:
        """返回路径上**所有链路都被占用**的槽（布尔数组，True = 已占用）。

        跨路径各链路做逐槽 OR。这是多槽分配器做连续窗口卷积的输入。

        Args:
            path: 节点路径，至少两跳。

        Returns:
            长度 = ``num_channels * slots_per_channel`` 的布尔数组。
        """
        if len(path) < 2:
            raise ValueError(f"Path must have at least two nodes, got {path!r}.")
        arrays = [
            self.get_link_state(path[i], path[i + 1]).occupied_slots for i in range(len(path) - 1)
        ]
        return np.logical_or.reduce(arrays)
