"""管理网络的动态资源状态，如波长占用和链路功率谱。"""

from copy import deepcopy
from dataclasses import dataclass

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

    def __post_init__(self):
        """确保数组在初始化时是副本，防止外部修改。"""
        self.occupied_channels = self.occupied_channels.copy()
        self.launch_power_profile_w = self.launch_power_profile_w.copy()
        self.allocated_service_ids = self.allocated_service_ids.copy()

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

    def release(self, channel_idx: int) -> None:
        """释放指定信道上的资源。

        Args:
            channel_idx: 待释放的信道索引。
        """
        if not self.occupied_channels[channel_idx]:
            # warnings.warn(f"Channel {channel_idx} on link {self.link_key} is not occupied, skipping release.")
            return  # 静默处理重复释放
        self.occupied_channels[channel_idx] = False
        self.launch_power_profile_w[channel_idx] = 0.0
        self.allocated_service_ids[channel_idx] = -1  # 标记为未分配


class NetworkState:
    """管理整个网络的动态运行状态。

    包含所有链路的当前占用情况和功率谱，以及所有已分配的业务详情。
    """

    def __init__(self, network_topology: NetworkTopology, spectrum_grid: SpectrumGrid):
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
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
            )

    def get_link_state(self, u_idx: NodeID, v_idx: NodeID) -> LinkState:
        """获取指定链路的当前状态."""
        link_key = tuple(sorted((u_idx, v_idx)))
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
        """
        # 确保服务ID是唯一的
        if service.service_id in self._allocated_services:
            # 如果是重新分配 (例如，动态调整路由)，则先释放旧的
            self.release_service(self._allocated_services[service.service_id])

        for i in range(len(service.path) - 1):
            u, v = service.path[i], service.path[i + 1]
            link_state = self.get_link_state(u, v)
            link_state.allocate(service.wavelength, service.launch_power_w, service.service_id)
        self._allocated_services[service.service_id] = service

    def release_service(self, service: AllocatedService) -> None:
        """从网络状态中释放一个服务。"""
        if service.service_id not in self._allocated_services:
            # warnings.warn(f"Service {service.service_id} not found for release.")
            return  # 静默处理重复释放或未分配服务

        for i in range(len(service.path) - 1):
            u, v = service.path[i], service.path[i + 1]
            link_state = self.get_link_state(u, v)
            link_state.release(service.wavelength)
        del self._allocated_services[service.service_id]

    def deep_copy(self) -> "NetworkState":
        """创建当前网络状态的深拷贝。

        用于 QoT 验证器进行 'what-if' 分析，避免修改实际网络状态。
        """
        new_state = NetworkState(self.network_topology, self.spectrum_grid)
        # Deepcopy _link_states
        new_state._link_states = {k: deepcopy(v) for k, v in self._link_states.items()}
        # Deepcopy _allocated_services
        new_state._allocated_services = {k: deepcopy(v) for k, v in self._allocated_services.items()}
        return new_state
