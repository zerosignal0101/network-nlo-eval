"""离散事件仿真器 (DES) 核心引擎。"""

import heapq
from dataclasses import dataclass, field

from tqdm import tqdm

from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.rwa.allocators import BaseRWAAllocator
from network_nlo_eval.simulation.reporter import MetricsCollector
from network_nlo_eval.simulation.traffic import ServiceRequest


@dataclass(order=True)
class Event:
    """仿真事件的数据模型。

    使用 dataclass 的 order=True 属性，使其可以直接用于 heapq (优先队列)。
    """

    time: float = field(compare=True)  # 事件发生时间，用于排序
    event_type: str = field(compare=False)  # "ARRIVAL" 或 "DEPARTURE"
    service_id: int = field(compare=False)  # 关联的服务ID

    # 包含 ServiceRequest 或 AllocatedService，但不对其进行比较
    service_data: ServiceRequest = field(compare=False)


class SimulatorEngine:
    """离散事件仿真主引擎。

    按时间顺序处理业务到达和离开事件，调用 RWA 算法进行资源分配和释放。
    """

    def __init__(
        self,
        allocator: BaseRWAAllocator,
        network_state: NetworkState,
        metrics_collector: MetricsCollector,
        incoming_services: list[ServiceRequest],
    ):
        """初始化仿真引擎。

        Args:
            allocator: 实现 RWA 逻辑的分配器实例。
            network_state: 当前网络状态的实例。
            metrics_collector: 负责收集和记录仿真指标与事件的实例。
            incoming_services: 预生成的业务请求列表。
        """
        self.allocator = allocator
        self.network_state = network_state
        self.metrics_collector = metrics_collector
        self.current_time = 0.0
        self.event_queue: list[Event] = []

        # 将所有业务的到达事件预先加入事件队列
        for service in incoming_services:
            self.schedule(Event(service.arrival_time, "ARRIVAL", service.service_id, service))

    def schedule(self, event: Event) -> None:
        """将一个事件加入事件队列。

        Args:
            event: 待调度的事件。
        """
        heapq.heappush(self.event_queue, event)

    def run(self) -> None:
        """运行仿真，直到事件队列为空。"""
        # 使用 tqdm 显示进度条
        with tqdm(total=len(self.event_queue), desc="Running Simulation") as pbar:
            while self.event_queue:
                event = heapq.heappop(self.event_queue)
                self.current_time = event.time

                if event.event_type == "ARRIVAL":
                    self._handle_arrival(event)
                    # 更新进度条，到达事件处理一次
                    pbar.update(1)
                elif event.event_type == "DEPARTURE":
                    self._handle_departure(event)
                # 其他事件类型可在此扩展

    def _handle_arrival(self, event: Event) -> None:
        """处理业务到达事件。

        尝试分配资源，并根据结果调度离开事件或记录阻塞。
        """
        service_request = event.service_data

        # 尝试分配资源
        is_success, allocated_service = self.allocator.allocate(service_request, self.network_state)

        if is_success:
            # 分配成功：更新网络状态，记录成功事件，调度离开事件
            if allocated_service is None:  # 理论上不应该发生
                raise ValueError("Allocator returned success but allocated_service is None.")
            self.network_state.allocate_service(allocated_service)
            self.metrics_collector.record_allocation_success(allocated_service)

            # 调度离开事件
            self.schedule(
                Event(
                    allocated_service.departure_time,
                    "DEPARTURE",
                    allocated_service.service_id,
                    allocated_service,  # 离开事件也携带完整的服务数据
                )
            )
        else:
            # 分配失败：记录失败事件
            self.metrics_collector.record_allocation_failure(service_request)

    def _handle_departure(self, event: Event) -> None:
        """处理业务离开事件。

        释放资源，并记录释放事件。
        """
        # 离开事件的 service_data 应该是一个 AllocatedService
        allocated_service = event.service_data

        # 检查业务是否仍在已分配列表中 (可能在动态场景中被提前释放或抢占)
        if self.network_state.get_allocated_service(allocated_service.service_id) is not None:
            # 释放资源
            self.allocator.release(allocated_service, self.network_state)
            self.metrics_collector.record_service_release(allocated_service)
