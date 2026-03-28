"""收集仿真指标和事件，并生成报告。"""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import OriginalNodeID
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


class ServiceEvent(BaseModel):
    """记录仿真中发生的业务事件."""

    time: float = Field(..., description="事件发生时的仿真时间")
    event_type: Literal["AllocationSuccess", "AllocationFailure", "Release"] = Field(..., description="事件类型")
    service_id: int = Field(..., description="业务ID")
    source_node_id: OriginalNodeID = Field(..., description="源节点原始ID")
    dest_node_id: OriginalNodeID = Field(..., description="目的节点原始ID")
    bit_rate_gbps: float = Field(..., description="业务比特率 (Gbps)")
    wavelength: int | None = Field(None, description="分配的波长索引 (如果成功分配)")
    path_original_ids: list[OriginalNodeID] | None = Field(
        None, description="分配的路径 (原始节点ID列表, 如果成功分配)"
    )
    reason: str | None = Field(None, description="分配失败原因")
    departure_time: float | None = Field(None, description="服务预计结束时间")


class MetricsCollector:
    """收集和计算仿真过程中的性能指标。"""

    def __init__(
        self,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        total_incoming_services: int,
    ):
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        self.total_incoming_services = total_incoming_services

        self.service_events: list[ServiceEvent] = []
        self.block_num: int = 0
        self.successfully_allocated_service_count: int = 0
        self.total_hops_for_allocated_services: int = 0
        self.total_allocated_traffic_gbps: float = 0.0  # 成功分配的业务总流量 (用于吞吐量)
        self.total_occupied_slot_hours: float = 0.0  # 用于计算时间平均利用率 (未来扩展)

    def record_allocation_success(self, service: AllocatedService) -> None:
        """记录业务成功分配事件."""
        self.successfully_allocated_service_count += 1
        self.total_hops_for_allocated_services += len(service.path) - 1
        self.total_allocated_traffic_gbps += service.bit_rate_gbps

        original_path_ids = [self.network_topology.get_original_node_id(node_idx) for node_idx in service.path]
        self.service_events.append(
            ServiceEvent(
                time=service.arrival_time,
                event_type="AllocationSuccess",
                service_id=service.service_id,
                source_node_id=self.network_topology.get_original_node_id(service.source_id),
                dest_node_id=self.network_topology.get_original_node_id(service.destination_id),
                bit_rate_gbps=service.bit_rate_gbps,
                wavelength=service.wavelength,
                path_original_ids=original_path_ids,
                departure_time=service.departure_time,
            )
        )

    def record_allocation_failure(
        self,
        service_request: ServiceRequest,
        reason: str = "No suitable path or wavelength found",
    ) -> None:
        """记录业务分配失败事件."""
        self.block_num += 1
        self.service_events.append(
            ServiceEvent(
                time=service_request.arrival_time,
                event_type="AllocationFailure",
                service_id=service_request.service_id,
                source_node_id=self.network_topology.get_original_node_id(service_request.source_id),
                dest_node_id=self.network_topology.get_original_node_id(service_request.destination_id),
                bit_rate_gbps=service_request.bit_rate_gbps,
                reason=reason,
            )
        )

    def record_service_release(self, service: AllocatedService) -> None:
        """记录业务释放事件."""
        original_path_ids = [self.network_topology.get_original_node_id(node_idx) for node_idx in service.path]
        self.service_events.append(
            ServiceEvent(
                time=service.departure_time,
                event_type="Release",
                service_id=service.service_id,
                source_node_id=self.network_topology.get_original_node_id(service.source_id),
                dest_node_id=self.network_topology.get_original_node_id(service.destination_id),
                bit_rate_gbps=service.bit_rate_gbps,
                wavelength=service.wavelength,
                path_original_ids=original_path_ids,
                departure_time=service.departure_time,
            )
        )

    def generate_report(self, final_network_state: NetworkState) -> dict[str, Any]:
        """计算最终的性能指标并生成报告。

        Args:
            final_network_state: 仿真结束时的网络最终状态。
        Returns:
            Dict[str, Any]: 包含所有指标和事件的报告。
        """
        metrics = self._calculate_network_metrics(final_network_state)
        self.service_events.sort(key=lambda x: x.time)  # 确保事件按时间排序

        report = {
            "simulation_metrics": metrics,
            "service_events": [event.model_dump() for event in self.service_events],
        }
        return report

    def _calculate_network_metrics(self, final_network_state: NetworkState) -> dict[str, Any]:
        """计算关键网络性能指标."""
        metrics = {}

        # 1. 阻塞率 (Blocking Rate)
        metrics["total_incoming_services"] = self.total_incoming_services
        metrics["total_block_num"] = self.block_num
        metrics["successfully_allocated_service_count"] = self.successfully_allocated_service_count
        metrics["blocking_rate"] = (
            self.block_num / self.total_incoming_services if self.total_incoming_services > 0 else 0.0
        )

        # 2. 波长利用率 (Wavelength Utilization)
        total_occupied_slots = 0
        total_possible_slots = 0
        for link_state in final_network_state.get_all_link_states().values():
            total_occupied_slots += np.sum(link_state.occupied_channels)
            total_possible_slots += self.spectrum_grid.num_channels
        metrics["wavelength_utilization"] = (
            total_occupied_slots / total_possible_slots if total_possible_slots > 0 else 0.0
        )

        # 3. 平均跳数 (Average Hop Count)
        metrics["average_hop_count"] = (
            self.total_hops_for_allocated_services / self.successfully_allocated_service_count
            if self.successfully_allocated_service_count > 0
            else 0.0
        )

        # 4. 平均吞吐量 (Average Throughput)
        metrics["average_throughput_gbps"] = (
            self.total_allocated_traffic_gbps / self.total_incoming_services
            if self.total_incoming_services > 0
            else 0.0
        )

        # 5. 波长碎片化程度 (Wavelength Fragmentation Index)
        total_fragmentation_index = 0.0
        num_links_considered_for_fragmentation = 0

        for link_state in final_network_state.get_all_link_states().values():
            free_bands = ~link_state.occupied_channels  # True表示空闲
            num_free_channels = np.sum(free_bands)

            current_link_fragmentation = 0.0

            # 碎片化在有超过一个空闲信道，且并非所有信道都空闲或都占用时有意义
            if 1 < num_free_channels < self.spectrum_grid.num_channels:
                # 计算空闲信道的连续块数量。
                # 在数组两端填充 False，以便正确捕获边界处的块。
                padded_free_bands = np.concatenate(([False], free_bands, [False]))
                # 统计从“占用”(False) 到“空闲”(True) 的转换次数，即为连续空闲块的数量
                num_free_blocks = np.sum(padded_free_bands[1:] & ~padded_free_bands[:-1])

                # 碎片化指数公式：(连续空闲块数量 - 1) / (总空闲信道数量 - 1)
                # 如果 num_free_blocks 为1，指数为0（一个大块）。
                # 如果每个空闲信道都是独立一块，则指数接近1。
                # 避免分母为0
                if num_free_channels - 1 > 0:
                    current_link_fragmentation = (num_free_blocks - 1) / (num_free_channels - 1)

            total_fragmentation_index += current_link_fragmentation
            num_links_considered_for_fragmentation += 1

        metrics["average_wavelength_fragmentation_index"] = (
            total_fragmentation_index / num_links_considered_for_fragmentation
            if num_links_considered_for_fragmentation > 0
            else 0.0
        )

        return metrics
