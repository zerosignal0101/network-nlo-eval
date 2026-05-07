"""传输质量 (Quality of Transmission, QoT) 验证器。

负责在 RWA 过程中评估链路或路径的 SNR。
"""

import math

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey, NDArrayFloat, NodeID
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.state import LinkNoiseCache, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


def _normalize(u: NodeID, v: NodeID) -> LinkKey:
    """返回规范化的链路键 (min, max)。"""
    return LinkKey((u, v) if u < v else (v, u))


class QoTValidator:
    """传输质量验证器。

    RWA 算法使用它来判断一个候选分配方案是否满足 SNR 要求。

    核心策略：**临时副本 + 局部缓存**，不修改 LinkState，无需回滚。
    """

    def __init__(
        self,
        gn_evaluator: MultiBandISRSGN,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        fixed_span_length_km: float = 100.0,
    ):
        self.gn_evaluator = gn_evaluator
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        self._fixed_span_length_km = fixed_span_length_km

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def verify_allocation(
        self,
        service_request: ServiceRequest,
        path: list[NodeID],
        channel_idx: int,
        current_network_state: NetworkState,
        fixed_span_length_km: float = 100.0,
    ) -> tuple[bool, NDArrayFloat | None]:
        """验证将 ``service_request`` 分配到 ``path`` 和 ``channel_idx`` 是否满足所有 SNR 要求。

        1. 创建新业务路径上各链路的临时功率谱副本（含新业务功率），不修改 LinkState。
        2. 对受影响链路计算噪声并存入局部缓存（仅一次）。
        3. 用 _eval_service_snr 评估新业务 SNR。
        4. 收集受影响旧业务，用各自路径和发射功率评估 SNR。
        5. 全部通过后，将临时缓存写入 LinkState.noise_cache。

        Returns
        -------
            Tuple[bool, Optional[NDArrayFloat]]:
            - bool: 如果所有 SNR 检查通过，则为 True；否则为 False。
            - Optional[NDArrayFloat]: 如果成功，返回新业务路径上每个信道的 SNR (dB)，
              否则为 None。
        """
        span_km = fixed_span_length_km or self._fixed_span_length_km

        # 1. 创建临时功率谱副本 & 检查波长可用性
        temp_power_profiles: dict[LinkKey, NDArrayFloat] = {}
        affected_links: set[LinkKey] = set()

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = current_network_state.get_link_state(u, v)

            if link_state.occupied_channels[channel_idx]:
                return False, None  # 波长已被占用

            link_key = _normalize(u, v)
            affected_links.add(link_key)
            temp_profile = link_state.launch_power_profile_w.copy()
            temp_profile[channel_idx] = service_request.launch_power_w
            temp_power_profiles[link_key] = temp_profile

        # 2. 计算受影响链路的噪声（仅一次），存入局部缓存
        temp_noise_cache: dict[LinkKey, LinkNoiseCache] = {}
        for link_key in affected_links:
            temp_noise_cache[link_key] = self._calc_link_noise(
                link_key,
                temp_power_profiles[link_key],
                span_km,
            )

        # 3. 评估新业务 SNR
        new_snr_db = self._eval_service_snr(
            path,
            channel_idx,
            service_request.launch_power_w,
            temp_noise_cache,
            current_network_state,
            span_km,
        )
        if new_snr_db < service_request.snr_requirement_db:
            return False, None

        # 4. 收集受影响旧业务
        affected_old_services = self._collect_affected_services(
            path,
            channel_idx,
            current_network_state,
        )

        # 5. 评估每个旧业务 SNR（使用其自身路径和发射功率）
        for old_service in affected_old_services:
            old_snr_db = self._eval_service_snr(
                old_service.path,
                old_service.wavelength,
                old_service.launch_power_w,
                temp_noise_cache,
                current_network_state,
                span_km,
            )
            if old_snr_db < old_service.snr_requirement_db:
                return False, None

        # 6. 全部通过 — 将临时缓存写入 LinkState（避免后续重复计算）
        for link_key, cache in temp_noise_cache.items():
            u, v = link_key
            link_state = current_network_state.get_link_state(u, v)
            link_state.noise_cache = cache

        # 构建新业务路径上各信道的 SNR 数组（用于返回值兼容性）
        final_snrs_db = self._build_path_snr_array(
            path,
            temp_power_profiles,
            temp_noise_cache,
            current_network_state,
            span_km,
        )

        return True, final_snrs_db

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _calc_link_noise(
        self,
        link_key: LinkKey,
        power_profile: NDArrayFloat,
        fixed_span_length_km: float,
    ) -> LinkNoiseCache:
        """计算单条链路在给定功率谱下的噪声缓存。"""
        u, v = link_key
        fiber_config = self.network_topology.get_fiber_config(u, v)
        edfa_config = self.network_topology.get_edfa_config(u)
        roadm_config = self.network_topology.get_roadm_config(u)

        link_total_length_m = fiber_config.length_km * 1000
        span_length_m = fixed_span_length_km * 1000
        num_full_spans = int(link_total_length_m // span_length_m)
        remainder_length_m = link_total_length_m % span_length_m

        total_spm = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_xpm = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_ase = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_spans = 0

        if num_full_spans > 0:
            _, s_spm, s_xpm, s_ase = self.gn_evaluator._calc_span_noise_and_power(
                power_profile,
                span_length_m,
                edfa_config,
                None,
                n_effective_spans=1,
            )
            total_spm += s_spm * num_full_spans
            total_xpm += s_xpm * num_full_spans
            total_ase += s_ase * num_full_spans
            total_spans += num_full_spans

        if remainder_length_m > 1.0:
            _, r_spm, r_xpm, r_ase = self.gn_evaluator._calc_span_noise_and_power(
                power_profile,
                remainder_length_m,
                edfa_config,
                roadm_config,
                n_effective_spans=1,
            )
            total_spm += r_spm
            total_xpm += r_xpm
            total_ase += r_ase
            total_spans += 1

        return LinkNoiseCache(spm=total_spm, xpm=total_xpm, ase=total_ase, span_count=total_spans)

    def _get_or_calc_noise_cache(
        self,
        link_key: LinkKey,
        link_state,
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        fixed_span_length_km: float,
    ) -> LinkNoiseCache:
        """获取链路噪声缓存：优先从临时缓存，否则从 LinkState（惰性计算）。"""
        if link_key in temp_noise_cache:
            return temp_noise_cache[link_key]

        if link_state.noise_cache is not None:
            return link_state.noise_cache

        cache = self._calc_link_noise(
            link_key,
            link_state.launch_power_profile_w,
            fixed_span_length_km,
        )
        link_state.noise_cache = cache
        return cache

    def _eval_service_snr(
        self,
        path: list[NodeID],
        channel_idx: int,
        launch_power_w: float,
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        network_state: NetworkState,
        fixed_span_length_km: float,
    ) -> float:
        """通用方法：沿服务自身路径累积噪声，计算单信道 SNR (dB)。

        - 受影响链路从 temp_noise_cache 读取（含假设分配后的功率谱）。
        - 未受影响链路从 LinkState.noise_cache 读取（惰性缓存）。
        - 使用服务自身的 launch_power_w 作为信号功率。
        """
        total_spm = 0.0
        total_xpm = 0.0
        total_ase = 0.0
        total_spans = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_key = _normalize(u, v)
            link_state = network_state.get_link_state(u, v)

            cache = self._get_or_calc_noise_cache(
                link_key,
                link_state,
                temp_noise_cache,
                fixed_span_length_km,
            )

            total_spm += cache.spm[channel_idx]
            total_xpm += cache.xpm[channel_idx]
            total_ase += cache.ase[channel_idx]
            total_spans += cache.span_count

        # SPM 相干累积惩罚
        penalty = total_spans**0.05 if total_spans > 0 else 1.0
        noise = total_spm * penalty + total_xpm + total_ase

        if noise <= 0:
            return float("inf")
        return 10.0 * math.log10(launch_power_w / noise)

    def _collect_affected_services(
        self,
        path: list[NodeID],
        channel_idx: int,
        network_state: NetworkState,
    ) -> list[AllocatedService]:
        """收集新业务路径上受影响的旧业务（排除新业务自身占用的信道）。"""
        seen_ids: set[int] = set()
        affected: list[AllocatedService] = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = network_state.get_link_state(u, v)

            for ch_idx in range(self.spectrum_grid.num_channels):
                if ch_idx == channel_idx:
                    continue
                if not link_state.occupied_channels[ch_idx]:
                    continue
                service_id = link_state.allocated_service_ids[ch_idx]
                if service_id == -1 or service_id in seen_ids:
                    continue
                seen_ids.add(service_id)
                svc = network_state.get_allocated_service(service_id)
                if svc is not None:
                    affected.append(svc)

        return affected

    def _build_path_snr_array(
        self,
        path: list[NodeID],
        temp_power_profiles: dict[LinkKey, NDArrayFloat],
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        network_state: NetworkState,
        fixed_span_length_km: float,
    ) -> NDArrayFloat:
        """构建新业务路径上各信道的 SNR (dB) 数组，用于返回值。"""
        total_spm = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_xpm = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_ase = np.zeros(self.spectrum_grid.num_channels, dtype=np.float64)
        total_spans = 0

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_key = _normalize(u, v)
            link_state = network_state.get_link_state(u, v)

            cache = self._get_or_calc_noise_cache(
                link_key,
                link_state,
                temp_noise_cache,
                fixed_span_length_km,
            )

            total_spm += cache.spm
            total_xpm += cache.xpm
            total_ase += cache.ase
            total_spans += cache.span_count

        penalty = total_spans**0.05 if total_spans > 0 else 1.0
        total_noise = total_spm * penalty + total_xpm + total_ase

        snrs_db = np.full(self.spectrum_grid.num_channels, -np.inf, dtype=np.float64)
        for ch_idx in range(self.spectrum_grid.num_channels):
            if total_noise[ch_idx] <= 0:
                continue
            # 取该信道在路径上第一条链路的功率作为信号功率
            u0, v0 = path[0], path[1]
            link_key0 = _normalize(u0, v0)
            if link_key0 in temp_power_profiles:
                signal_power = temp_power_profiles[link_key0][ch_idx]
            else:
                signal_power = network_state.get_link_state(u0, v0).launch_power_profile_w[ch_idx]
            if signal_power > 0:
                snrs_db[ch_idx] = 10.0 * math.log10(signal_power / total_noise[ch_idx])

        return snrs_db


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
