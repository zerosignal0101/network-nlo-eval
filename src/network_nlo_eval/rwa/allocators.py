"""路由与波长分配 (RWA) 策略实现。"""

from dataclasses import asdict

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayBool, NodeID
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import BaseRWAAllocator, QoTValidator
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


class KSPFirstFitAllocator(BaseRWAAllocator):
    """K-Shortest Path (KSP) First-Fit 路由与波长分配器。

    按预计算的 KSP 路径顺序，并在每条路径上按波长索引升序尝试分配第一个可用波长。
    在分配前会进行 QoT 验证。
    """

    def __init__(
        self,
        path_cache: PathCache,
        qot_validator: QoTValidator,
        spectrum_grid: SpectrumGrid,
    ):
        """初始化 KSP First-Fit 分配器。

        Args:
            path_cache: 路径缓存实例。
            qot_validator: QoT 验证器实例。
            spectrum_grid: 频谱网格配置。
        """
        self.path_cache = path_cache
        self.qot_validator = qot_validator
        self.spectrum_grid = spectrum_grid

    def _get_available_channels_on_path(
        self,
        path: list[NodeID],
        network_state: NetworkState,
    ) -> NDArrayBool:
        """返回路径上所有空闲波段 (布尔数组，True=空闲)。

        一个波段只有在路径上所有链路都空闲时才被认为是空闲的。
        """
        # 提取路径上所有链路的占用数组
        occupied_arrays = [
            network_state.get_link_state(path[i], path[i + 1]).occupied_channels for i in range(len(path) - 1)
        ]
        # 一次性使用 numpy 进行 bitwise OR 并取反
        # 如果任意链路占用，则该位置为 True。整体取反即为可用
        return ~np.logical_or.reduce(occupied_arrays)

    def allocate(
        self, service_request: ServiceRequest, network_state: NetworkState
    ) -> tuple[bool, AllocatedService | None]:
        """尝试分配一个新业务。

        Args:
            service_request: 待分配的业务请求。
            network_state: 当前的网络状态。

        Returns
        -------
            Tuple[bool, Optional[AllocatedService]]:
            - bool: 如果成功分配，则为 True；否则为 False。
            - Optional[AllocatedService]: 如果成功，返回分配后的服务对象；否则为 None。
        """
        # 1. 获取所有 KSP 路径
        ksp_paths = self.path_cache.get_paths(service_request.source_id, service_request.destination_id)

        if not ksp_paths:
            # 没有找到任何 KSP 路径
            return False, None

        # 2. 遍历 KSP 路径
        for path in ksp_paths:
            # 2.1 找出路径上所有链路都空闲的波长
            available_channels_on_path = self._get_available_channels_on_path(path, network_state)
            free_channel_indices = np.where(available_channels_on_path)[0].tolist()

            if not free_channel_indices:
                # 该路径上没有完全空闲的波长，尝试下一条路径
                continue

            # 2.2 First-Fit: 按波长索引升序尝试分配
            for channel_idx in free_channel_indices:
                # 2.3 进行 QoT 验证
                is_qot_satisfied, _ = self.qot_validator.verify_allocation(
                    service_request=service_request,
                    path=path,
                    channel_idx=channel_idx,
                    current_network_state=network_state,
                )

                if is_qot_satisfied:
                    # 分配成功：创建 AllocatedService 对象并返回
                    allocated_service = AllocatedService(
                        **asdict(service_request),
                        path=path,
                        wavelength=channel_idx,
                    )
                    return True, allocated_service
                # 如果 QoT 验证失败，尝试该路径上的下一个空闲波长

        # 3. 所有路径和波长都尝试完毕，未能成功分配
        return False, None

    def release(self, allocated_service: AllocatedService, network_state: NetworkState) -> None:
        """释放一个已分配业务的资源。

        Args:
            allocated_service: 待释放的 AllocatedService 对象。
            network_state: 当前的网络状态。
        """
        network_state.release_service(allocated_service)


def first_fit_contiguous_windows(
    free_mask: NDArrayBool, n_slots: int, slots_per_channel: int
) -> list[int]:
    """在空闲槽掩码上做连续 ``n_slots`` 窗口的首次适配（升序）。

    这是多槽分配器的窗口卷积：返回所有**起点升序**的窗口，使得
    ``[start, start + n_slots)`` 全部落在 ``free_mask`` 为真的槽上。

    **窗口卷积**用前缀和实现（O(N)），不是逐窗口切片扫描（O(N·n)）：
    ``window_sum[i] == n_slots`` 即窗口 ``[i, i+n_slots)`` 全空闲。

    **同一锚信道只留第一个候选**：QoT 验证只吃锚信道
    ``first_slot // slots_per_channel``，同一信道内不同起点的 QoT 结果完全相同，
    重复验证是纯浪费（M-J11 耗时）。由于起点升序 ⇒ 锚信道单调不减，
    「去重相邻」等价于「全局去重」。

    Args:
        free_mask: 槽级空闲掩码，True = 空闲。
        n_slots: 窗口宽度 (槽数)，>= 1。
        slots_per_channel: 每信道的槽数，>= 1。

    Returns:
        候选窗口起点升序列表；无解时为空列表。
    """
    total = int(free_mask.shape[0])
    if n_slots < 1:
        raise ValueError(f"n_slots must be >= 1, got {n_slots}.")
    if slots_per_channel < 1:
        raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")
    if n_slots > total:
        return []

    prefix = np.concatenate(([0], np.cumsum(free_mask.astype(np.int64))))
    window_free = prefix[n_slots:] - prefix[:-n_slots]
    starts = np.flatnonzero(window_free == n_slots)

    candidates: list[int] = []
    last_channel = -1
    for start in starts.tolist():
        channel_idx = start // slots_per_channel
        if channel_idx != last_channel:
            candidates.append(int(start))
            last_channel = channel_idx
    return candidates


class KSPFirstFitFlexAllocator(BaseRWAAllocator):
    """灵活栅格的 K-Shortest-Path + 连续槽窗口 First-Fit 分配器（WP-M1）。

    与 `KSPFirstFitAllocator` 的差别**只在分配层的槽粒度**：

    * 占用真源是**槽**（`LinkState.occupied_slots`），跨路径各链路逐槽 OR；
    * 候选是**连续 ``n_slots`` 的槽窗口**，返回 `first_slot`；
    * 尊重 ROADM 的 `blocked_slots`（权威）与 `blocked_channels`（按
      `slots_per_channel` 展开）——窗口与任一被阻断槽重叠即不可行（切片语义）；
    * 物理/QoT 仍在**锚信道** `first_slot // slots_per_channel` 上验证，
      与遗留分配器调用 `verify_allocation` 的口径一致，故物理层一字未改。

    ⚠️ 本类**不改动** `KSPFirstFitAllocator` 的任何行为；它是新增的第二条路径。
    """

    def __init__(
        self,
        path_cache: PathCache,
        qot_validator: QoTValidator,
        spectrum_grid: SpectrumGrid,
        network_topology: NetworkTopology,
        slots_per_channel: int,
    ):
        """初始化灵活栅格分配器。

        Args:
            path_cache: 路径缓存实例。
            qot_validator: QoT 验证器实例。
            spectrum_grid: 频谱网格配置（物理面仍按它逐信道路入功率）。
            network_topology: 拓扑，用于读取路径沿途 ROADM 的阻断槽。
            slots_per_channel: 每信道的槽数（50 GHz / 12.5 GHz = 4）。
        """
        if isinstance(slots_per_channel, bool) or slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be a positive integer, got {slots_per_channel!r}.")
        self.path_cache = path_cache
        self.qot_validator = qot_validator
        self.spectrum_grid = spectrum_grid
        self.network_topology = network_topology
        self.slots_per_channel = slots_per_channel

    def blocked_slots_on_path(self, path: list[NodeID]) -> set[int]:
        """路径沿途所有 ROADM 的被阻断槽并集。

        解析顺序沿用全仓冻结策略（`slots.resolve_blocked_slots`）：
        ``blocked_slots`` 是权威字段；``blocked_channels`` 按
        ``first_slot = channel * slots_per_channel`` 展开为区间。
        """
        blocked: set[int] = set()
        for node_id in path:
            roadm = self.network_topology.get_roadm_config(node_id)
            blocked.update(roadm.blocked_slots)
            for channel_index in roadm.blocked_channels:
                blocked.update(
                    range(channel_index * self.slots_per_channel, (channel_index + 1) * self.slots_per_channel)
                )
        return blocked

    def allocate(
        self, service_request: ServiceRequest, network_state: NetworkState
    ) -> tuple[bool, AllocatedService | None]:
        """为请求寻找一段**连续槽窗口**，QoT 通过即分配。

        Returns:
            ``(True, AllocatedService)`` 成功；``(False, None)`` 失败。
        """
        n_slots = service_request.n_slots
        ksp_paths = self.path_cache.get_paths(service_request.source_id, service_request.destination_id)
        if not ksp_paths:
            return False, None

        for path in ksp_paths:
            # 跨路径各链路逐槽 OR：该槽在任一跳被占即不可用。
            occupied = network_state.occupied_slots_on_path(path)
            free = ~occupied

            # 尊重 ROADM 阻断（切片语义：重叠即不可行）。
            blocked = self.blocked_slots_on_path(path)
            if blocked:
                in_range = [slot for slot in blocked if 0 <= slot < free.shape[0]]
                if in_range:
                    free = free.copy()
                    free[in_range] = False

            for first_slot in first_fit_contiguous_windows(free, n_slots, self.slots_per_channel):
                channel_idx = first_slot // self.slots_per_channel

                is_qot_satisfied, _ = self.qot_validator.verify_allocation(
                    service_request=service_request,
                    path=path,
                    channel_idx=channel_idx,
                    current_network_state=network_state,
                )
                if not is_qot_satisfied:
                    continue

                allocated_service = AllocatedService(
                    **asdict(service_request),
                    path=path,
                    wavelength=channel_idx,
                    first_slot=first_slot,
                    slots_per_channel=self.slots_per_channel,
                )
                return True, allocated_service

        return False, None

    def release(self, allocated_service: AllocatedService, network_state: NetworkState) -> None:
        """按槽区间 ``[first_slot, first_slot + n_slots)`` 释放资源。"""
        network_state.release_service(allocated_service)
