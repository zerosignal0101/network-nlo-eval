"""路由与波长分配 (RWA) 策略实现。"""

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayBool, NodeID
from network_nlo_eval.network.state import NetworkState
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
        if not path or len(path) < 2:
            return np.ones(self.spectrum_grid.num_channels, dtype=bool)  # 空路径或单节点路径，所有波段空闲 (无效情况)

        # 初始状态为全 True (全空闲)
        combined_available_channels = np.ones(self.spectrum_grid.num_channels, dtype=bool)

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = network_state.get_link_state(u, v)
            # combined_available_channels &= ~link_state.occupied_channels
            # Bug fix: ~link_state.occupied_channels gives 'available' channels on *this* link.
            # We need the intersection of available channels across *all* links in the path.
            combined_available_channels &= ~link_state.occupied_channels

            # 如果在某个链路上已经没有空闲通道，则无需继续检查
            if not np.any(combined_available_channels):
                break

        return combined_available_channels

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
            # random.shuffle(free_channel_indices) # 可选：如果希望随机 First-Fit
            free_channel_indices.sort()  # First-Fit 策略，按索引从小到大

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
                        **service_request.model_dump(),
                        path=path,
                        wavelength=channel_idx,
                        launch_power_w=service_request.launch_power_w,
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
