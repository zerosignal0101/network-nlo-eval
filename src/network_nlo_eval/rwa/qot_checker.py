"""传输质量 (Quality of Transmission, QoT) 验证器。

负责在 RWA 过程中评估链路或路径的 SNR。
"""

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey, NDArrayFloat, NodeID
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.numba_kernels import _lin_to_db
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


class QoTValidator:
    """传输质量验证器。

    RWA 算法使用它来判断一个候选分配方案是否满足 SNR 要求。
    """

    def __init__(
        self,
        gn_evaluator: MultiBandISRSGN,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
    ):
        """初始化 QoT 验证器。

        Args:
            gn_evaluator: MultiBandISRSGN 实例，用于物理层噪声计算。
            network_topology: 网络拓扑管理实例，提供链路和节点配置。
            spectrum_grid: 频谱网格配置。
        """
        self.gn_evaluator = gn_evaluator
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid

    def verify_allocation(
        self,
        service_request: ServiceRequest,
        path: list[NodeID],
        channel_idx: int,
        current_network_state: NetworkState,
        fixed_span_length_km: float = 100.0,
    ) -> tuple[bool, NDArrayFloat | None]:
        """验证将 `service_request` 分配到 `path` 和 `channel_idx` 是否满足所有 SNR 要求。

        此方法会执行以下检查：
        1. 检查新业务本身的 SNR 是否满足要求。
        2. 检查此分配是否会使路径上任何已存在的业务的 SNR 降至其要求以下。

        Args:
            service_request: 待分配的业务请求。
            path: 路由路径 (内部节点ID列表)。
            channel_idx: 待分配的波长索引。
            current_network_state: 当前的网络状态。
            fixed_span_length_km: 链路上固定跨段长度

        Returns
        -------
            Tuple[bool, Optional[NDArrayFloat]]:
            - bool: 如果所有 SNR 检查通过，则为 True；否则为 False。
            - Optional[NDArrayFloat]: 如果成功，返回路径上每个信道的最终 SNR 数组，否则为 None。
        """
        # 1. 记录原始状态以便回滚 (只备份将被修改的这几个整数/浮点数)
        rollback_data = []

        # 2. 在状态中执行假想分配 (这会影响功率谱)
        # 记录路径上所有受影响的现有服务，以便后续检查它们的 QoT
        # 存储格式: (service_id, original_snr_req_db, wavelength_idx)
        affected_existing_services: set[tuple[int, float, int]] = set()

        # 检查每个链路的占用情况并准备 power_profile_w
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = current_network_state.get_link_state(u, v)

            if link_state.occupied_channels[channel_idx]:
                # 目标波长已被占用，分配失败
                self._rollback(rollback_data)
                return False, None

            # 记录将被覆盖的原始值
            rollback_data.append(
                (
                    link_state,
                    channel_idx,
                    link_state.occupied_channels[channel_idx],
                    link_state.launch_power_profile_w[channel_idx],
                )
            )

            # 在状态中标记为占用，并设置功率
            link_state.occupied_channels[channel_idx] = True
            link_state.launch_power_profile_w[channel_idx] = service_request.launch_power_w
            # 注意：这里没有设置 allocated_service_ids，因为这不是实际分配

            # 记录此链路上所有其他已分配的服务
            for other_ch_idx in range(self.spectrum_grid.num_channels):
                if other_ch_idx == channel_idx:
                    continue
                if link_state.occupied_channels[other_ch_idx]:
                    existing_service_id = link_state.allocated_service_ids[other_ch_idx]
                    if existing_service_id != -1:  # 确保是有效服务ID
                        # 获取现有业务的完整数据
                        existing_allocated_service = current_network_state.get_allocated_service(existing_service_id)
                        if existing_allocated_service:  # 确保服务存在
                            affected_existing_services.add(
                                (
                                    existing_service_id,
                                    existing_allocated_service.snr_requirement_db,
                                    other_ch_idx,
                                )
                            )

        # 3. 评估新业务和受影响业务的端到端 SNR
        # 初始化总噪声 (W)
        # 每个信道的总噪声
        total_spm_power_w = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_xpm_power_w = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_ase_power_w = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_physical_spans_count = 0  # 统计整条路径总物理跨段数

        final_snrs_linear = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        final_snrs_db = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)

        # 假设每个 EDFA 完美补偿了跨段损耗，所以信号功率保持在 service_request.launch_power_w
        # 用于计算 SNR 的信号功率
        signal_power_for_snr = service_request.launch_power_w

        # 为路径上所有链路的功率谱创建一份拷贝，以模拟一个完整的路径
        # Dict[LinkKey, NDArrayFloat]
        path_link_power_profiles: dict[LinkKey, NDArrayFloat] = {}
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_key = LinkKey((u, v) if u < v else (v, u))
            link_state = current_network_state.get_link_state(u, v)
            path_link_power_profiles[link_key] = link_state.launch_power_profile_w.copy()  # 确保是拷贝

        for i in range(len(path) - 1):
            u_node_idx = path[i]
            v_node_idx = path[i + 1]
            link_key = LinkKey((u_node_idx, v_node_idx) if u_node_idx < v_node_idx else (v_node_idx, u_node_idx))

            fiber_config = self.network_topology.get_fiber_config(u_node_idx, v_node_idx)
            edfa_config = self.network_topology.get_edfa_config(u_node_idx)
            roadm_config = self.network_topology.get_roadm_config(u_node_idx)

            # 获取该链路上所有信道的当前功率分布，这用于计算 XPM 贡献
            current_power_profile_on_link = path_link_power_profiles[link_key]

            link_total_length_m = fiber_config.length_km * 1000

            # --- 拆分 Span 逻辑 ---
            span_length_m = fixed_span_length_km * 1000
            # 计算全长度 Span 的数量
            num_full_spans = int(link_total_length_m // span_length_m)
            # 计算剩余长度
            remainder_length_m = link_total_length_m % span_length_m

            # 计算单跨段的 NLI 和 ASE 噪声方差
            # _ 代表该跨段的出纤功率，我们不直接使用它来累积 SNR
            if num_full_spans > 0:
                _, s_spm, s_xpm, s_ase = self.gn_evaluator._calc_span_noise_and_power(
                    current_power_profile_on_link,
                    span_length_m,
                    edfa_config,
                    None,  # 中继跨段通常没有 ROADM
                    n_effective_spans=1,
                )
                total_spm_power_w += s_spm * num_full_spans
                total_xpm_power_w += s_xpm * num_full_spans
                total_ase_power_w += s_ase * num_full_spans
                total_physical_spans_count += num_full_spans

            # 3.2 计算剩余长度跨段的噪声
            if remainder_length_m > 1.0:  # 长度大于1米才计算，避免浮点误差
                _, r_spm, r_xpm, r_ase = self.gn_evaluator._calc_span_noise_and_power(
                    current_power_profile_on_link,
                    remainder_length_m,
                    edfa_config,
                    roadm_config,  # 将 ROADM 损耗挂载在链路最后一个 Span
                    n_effective_spans=1,
                )

                total_spm_power_w += r_spm
                total_xpm_power_w += r_xpm
                total_ase_power_w += r_ase
                total_physical_spans_count += 1

            # 施加 SPM 相干累积惩罚 (Coherent Accumulation Penalty)
            epsilon = 0.05  # GN 模型针对标准 SMF 的经验相干因子
            if total_physical_spans_count > 0:
                spm_coherent_penalty = float(total_physical_spans_count) ** epsilon
            else:
                spm_coherent_penalty = 1.0

            # 计算总噪声功率
            total_noise_power_w = total_spm_power_w * spm_coherent_penalty + total_xpm_power_w + total_ase_power_w

            # 计算所有信道的最终 SNR (dB)
            for ch_idx in range(self.spectrum_grid.num_channels):
                # 只有有信号的信道才有 SNR 概念
                if current_power_profile_on_link[ch_idx] > 0 and total_noise_power_w[ch_idx] > 0:
                    final_snrs_linear[ch_idx] = signal_power_for_snr / total_noise_power_w[ch_idx]
                    final_snrs_db[ch_idx] = _lin_to_db(final_snrs_linear[ch_idx])
                else:
                    final_snrs_db[ch_idx] = -np.inf  # 无信号或无噪声，视为无限 SNR，但为了比较方便设为极小值

            # 4. 检查新业务的 SNR 是否满足要求
            if final_snrs_db[channel_idx] < service_request.snr_requirement_db:
                self._rollback(rollback_data)
                return False, None  # 新业务 SNR 不足

            # 5. 检查所有受影响的现有业务的 SNR 是否仍然满足要求
            for (
                _existing_service_id,
                snr_req_db,
                existing_ch_idx,
            ) in affected_existing_services:
                if final_snrs_db[existing_ch_idx] < snr_req_db:
                    self._rollback(rollback_data)
                    return False, None  # 现有业务 SNR 不足

        self._rollback(rollback_data)

        # 所有检查通过
        return True, final_snrs_db

    def _rollback(self, rollback_data):
        for link_state, ch_idx, orig_occ, orig_pow in rollback_data:
            link_state.occupied_channels[ch_idx] = orig_occ
            link_state.launch_power_profile_w[ch_idx] = orig_pow


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
