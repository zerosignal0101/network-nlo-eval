"""业务请求生成器与数据模型."""

import random

import numpy as np
from pydantic import BaseModel, Field

from network_nlo_eval.core.types import NodeID
from network_nlo_eval.rwa.path_computation import PathCache


class ServiceRequest(BaseModel):
    """网络业务请求的数据模型。

    这是一个不可变的数据结构，代表了一个待处理或正在处理的业务。
    """

    service_id: int = Field(..., description="业务的唯一ID")
    source_id: NodeID = Field(..., description="源节点ID (内部ID)")
    destination_id: NodeID = Field(..., description="目的节点ID (内部ID)")
    arrival_time: float = Field(..., description="业务到达时间 (仿真时间单位)")
    departure_time: float = Field(..., description="业务离开时间 (仿真时间单位)")
    bit_rate_gbps: float = Field(..., description="业务所需比特率 (Gbps)")
    snr_requirement_db: float = Field(..., description="业务所需的最小 SNR (dB)")

    # 业务的发射功率 (W)，通常由 RWA 决定，这里作为默认值或起点
    launch_power_w: float = Field(0.001, description="业务发射功率 (W)，约 0 dBm")  # 默认 0 dBm


class AllocatedService(ServiceRequest):
    """已分配业务的数据模型，继承自 ServiceRequest，

    并增加分配结果 (路径、波长) 信息。
    """

    path: list[NodeID] = Field(..., description="业务分配的路由路径 (内部节点ID列表)")
    wavelength: int = Field(..., description="业务分配的波长索引")

    class Config:
        frozen = True  # 分配后业务信息通常不应改变


def generate_services(
    path_cache: PathCache,
    service_num: int,
    avg_arrival_interval: float,
    avg_holding_time: float,
    seed: int | None = None,
) -> list[ServiceRequest]:
    """生成一系列模拟网络业务请求。

    Args:
        path_cache: 路径缓存对象，用于获取可用的源/目的节点对。
        service_num: 要生成的业务数量。
        avg_arrival_interval: 业务到达时间的平均间隔 (泊松过程)。
        avg_holding_time: 业务持续时间的平均值 (指数分布)。
        seed: 随机数种子，用于可重现的仿真。

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
        bit_rate_candidates_gbps = np.arange(100, 501, 10)  # 100 Mbps to 500 Mbps, step 10 Mbps
        weights = 1.0 / bit_rate_candidates_gbps
        weights = weights / np.sum(weights)  # 归一化权重

        bit_rate_gbps = float(np.random.choice(bit_rate_candidates_gbps, p=weights))

        # 根据比特率定义 SNR 需求 (示例)
        if bit_rate_gbps > 400:
            snr_requirement_db = 26.5  # 高比特率需要高SNR
        elif bit_rate_gbps > 300:
            snr_requirement_db = 23.5
        elif bit_rate_gbps > 200:
            snr_requirement_db = 20.0
        else:
            snr_requirement_db = 17.0  # 低比特率需要较低SNR

        # 转换为 dBm，这里为了简化，所有服务初始功率相同，实际中可根据业务类型调整
        launch_power_w = 1e-3  # 0 dBm

        service = ServiceRequest(
            service_id=service_id,
            source_id=source_id,
            destination_id=destination_id,
            arrival_time=arrival_time,
            departure_time=departure_time,
            bit_rate_gbps=bit_rate_gbps,
            snr_requirement_db=snr_requirement_db,
            launch_power_w=launch_power_w,
        )
        services.append(service)

    # 确保服务按到达时间排序，以便仿真引擎正确处理
    services.sort(key=lambda s: s.arrival_time)
    return services
