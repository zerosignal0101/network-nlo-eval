"""多波段广义高斯噪声 (Generalized Gaussian Noise, GGN) 模型评估器。

ISRS-GN (Inter-channel Stimulated Raman Scattering - Generalized Noise)
"""

from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey, NDArrayFloat, NodeID
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_nf_lin_jit,
    _calc_span_noise_and_power_jit,
    _db_to_lin,
    _lin_to_db,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)]


@dataclass
class MultiSpanOpticalPath:
    """定义一条光路，包含多个跨段及其对应的配置。"""

    fiber_configs: list[FiberSpanConfig]
    edfa_configs: list[EDFAConfig]
    roadm_configs: list[ROADMConfig] | None = None
    # 实际路径中的每个 link_key, 方便关联到 network_state
    link_keys: list[tuple[int, int]] = None


class MultiBandISRSGN:
    """多波段 ISRS-GN 模型评估器。

    用于快速计算多波段光网络中单跨段或多跨段的传输质量 (QoT)。
    """

    def __init__(
        self,
        grid: SpectrumGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
    ):
        """初始化 ISRS-GN 评估器。

        Args:
            grid: 频谱网格配置。
            ref_fiber_config: 参考光纤配置，用于计算各种波长相关参数。
            symbol_rate_hz: 系统的符号速率 (Hz)。
        """
        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.cr_w_m_hz = _SRS_CR  # 拉曼增益系数
        self.k_bar = 0.0  # 拉曼泵浦因子，简化为 0 (无泵浦)
        self.delta_f_co_hz = 15e12  # 石英光纤受激拉曼散射（SRS）增益谱的截断频率
        self.n2_m2_w = ref_fiber_config.nonlinear_index_n2

        self._precompute_channel_dependent_properties()

    def _precompute_channel_dependent_properties(self) -> None:
        """预计算所有信道通用的波长依赖物理属性，如波长、有效模面积等。

        这些属性在整个仿真中通常不变。
        """
        self.lambdas_m = self.grid.wavelengths  # (m)
        self.f_abs_hz = self.grid.frequencies  # (Hz)

        # 衰减系数 (Np/m)
        self.alpha_power_npm_channels = _calc_fiber_attenuation_npm(
            self.lambdas_m,
            self.ref_fiber_config.reference_wavelength_nm,
            self.ref_fiber_config.attenuation_alpha2,
            self.ref_fiber_config.attenuation_alpha1,
            self.ref_fiber_config.attenuation_alpha0,
        )

        # 有效模面积 (m^2)
        self.a_eff_m2_channels = _calc_aeff_dynamic(
            self.lambdas_m,
            self.ref_fiber_config.effective_area_um2_ref,
            self.ref_fiber_config.aeff_slope_um2_nm,
            self.ref_fiber_config.reference_wavelength_nm,
        )

        # 噪声系数 (线性)
        self.nf_lin_channels = _calc_nf_lin_jit(self.lambdas_m)

        # beta2 和 beta3 (s^2/m, s^3/m)
        # 简化：在每个跨段的中心波长处计算 beta2 和 beta3，或使用参考波长
        # 这里使用参考波长处的 D 和 S 来计算
        ref_lambda_m = C_LIGHT / self.grid.center_frequency_hz
        self.beta2_s2_m = _calc_beta2_from_d(self.ref_fiber_config.dispersion_parameter_d, ref_lambda_m)
        self.beta3_s3_m = _calc_beta3_from_s_d(
            self.ref_fiber_config.dispersion_slope_s,
            self.ref_fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

        # 相对频率 (Hz)，用于 NLI/SRS 计算
        self.f_rel_hz = self.f_abs_hz - self.grid.center_frequency_hz
        self.f_m_hz = self.f_rel_hz[0] - self.symbol_rate_hz / 2
        self.f_M_hz = self.f_rel_hz[-1] + self.symbol_rate_hz / 2

    def _calc_span_noise_and_power(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """计算单跨段的功率演化、NLI 噪声和 ASE 噪声。

        Args:
            power_in_w: 各信道入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            edfa_config: 跨段后的 EDFA 配置。
            roadm_config: (可选) 跨段后的 ROADM 配置。
            n_effective_spans: 有效跨段数，用于 NLI 累积效应，对于单跨段通常为 1。

        Returns
        -------
            Tuple 包含:
            - power_out_w: 经历损耗、SRS 和放大后的出纤功率。
            - sigma2_spm_w: 产生的 SPM 噪声方差 (W)。
            - sigma2_xpm_w: 产生的 XPM 噪声方差 (W)。
            - sigma2_ase_w: 产生的 ASE 噪声方差 (W)。
        """
        # 1. 处理放大器噪声系数 (NF)
        # 如果 edfa_config 提供了全频段的 NF 数组则使用之，否则使用预计算的默认值
        if edfa_config.gain_ripple_db is not None:
            # 这里可以根据实际需求动态计算 NF
            current_nf_lin = self.nf_lin_channels
        else:
            current_nf_lin = self.nf_lin_channels

        # 2. 调用 JIT 内核 (热路径)
        p_out_fiber, s2_spm, s2_xpm, s2_ase = _calc_span_noise_and_power_jit(
            power_in_w=power_in_w,
            span_length_m=span_length_m,
            f_rel_hz=self.f_rel_hz,
            f_abs_hz=self.f_abs_hz,
            lambdas_m=self.lambdas_m,
            alpha_power_npm=self.alpha_power_npm_channels,
            a_eff_m2=self.a_eff_m2_channels,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            n2_m2_w=self.n2_m2_w,
            nf_lin_channels=current_nf_lin,
            symbol_rate_hz=self.symbol_rate_hz,
            n_effective_spans=n_effective_spans,
            cr_w_m_hz=self.cr_w_m_hz,
            k_bar=self.k_bar,
            delta_f_co_hz=self.delta_f_co_hz,
        )

        # 3. 处理 ROADM 插入损耗 (可选，在 Python 层处理非线性外围逻辑)
        if roadm_config:
            # 若有 ROADM，出纤功率需要扣除插入损耗
            # 损耗通常不增加线性噪声方差，但会降低后续跨段的入纤功率
            loss_lin = _db_to_lin(roadm_config.insertion_loss_db + roadm_config.filtering_penalty_db)
            p_out_fiber = p_out_fiber / loss_lin

        return p_out_fiber, s2_spm, s2_xpm, s2_ase

    def evaluate_path_snr(
        self,
        path: list[NodeID],  # 路由路径 (内部节点ID列表)
        channel_idx: int,  # 待评估的信道索引
        launch_power_w: float,  # 每跨段的发射功率 (W)
        current_network_state: dict[tuple[int, int], NDArrayFloat],  # 每个链路的 (所有信道的) 功率分布
        snr_requirement_db: float,  # 业务 SNR 门限 (dB)
    ) -> tuple[float, NDArrayFloat]:
        """评估指定路径上特定信道的端到端 SNR。

        此函数将沿着路径累积 NLI 和 ASE 噪声。

        Args:
            path: 路由路径 (内部节点ID列表)。
            channel_idx: 待评估的信道索引。
            launch_power_w: 该信道在每个跨段的发射功率 (W)。
            current_network_state: 字典，键为规范化链路键，值为该链路上所有信道的 (当前) 发射功率分布。
                                   用于计算 NLI 中的 XPM 贡献。
            snr_requirement_db: 该信道所需 SNR 门限 (dB)。

        Returns
        -------
            Tuple[float, NDArrayFloat]:
            - accumulated_snr_db: 累积的端到端 SNR (dB)。
            - total_noise_power: 每个跨段的累积噪声方差 (W)。
        """
        num_spans_in_path = len(path) - 1

        total_spm_power_w = 0.0
        total_xpm_power_w = 0.0
        total_ase_power_w = 0.0

        # 理论上，ISRS-GN 模型计算的是每个 span 的 NLI 和 ASE，然后在线性域累加。
        # 为了简化，我们假设每个跨段后的 EDFA 完美补偿了该跨段的损耗 (包括 ROADM 损耗)，
        # 使得每个跨段的输入功率可以视为恒定为 launch_power_w。
        # 这样 NLI 和 ASE 噪声方差可以直接累加。

        for i in range(num_spans_in_path):
            u_node_idx = path[i]
            v_node_idx = path[i + 1]
            link_key = LinkKey((u_node_idx, v_node_idx) if u_node_idx < v_node_idx else (v_node_idx, u_node_idx))

            # 从 NetworkTopology 获取静态配置
            fiber_config = self.network_topology.get_fiber_config(u_node_idx, v_node_idx)
            edfa_config = self.network_topology.get_edfa_config(u_node_idx)  # 假设 EDFA 在每个节点后
            roadm_config = self.network_topology.get_roadm_config(u_node_idx)

            # 获取该链路上所有信道的当前功率分布，这用于计算 XPM 贡献
            # current_network_state 已经是深拷贝后的临时状态
            power_profile_on_link = current_network_state[link_key]

            # 计算单跨段的 NLI 和 ASE 噪声方差
            # 这里的 power_in_w 应该是该链路的发射功率分布
            _, span_spm_power_w, span_xpm_power_w, span_ase_power_w = self._calc_span_noise_and_power(
                power_profile_on_link,
                fiber_config.length_km * 1000,
                edfa_config,
                roadm_config,
                n_effective_spans=1,  # 单个跨段计算
            )

            total_spm_power_w += span_spm_power_w[channel_idx]
            total_xpm_power_w += span_xpm_power_w[channel_idx]
            total_ase_power_w += span_ase_power_w[channel_idx]

        # 施加 SPM 相干累积惩罚 (Coherent Accumulation Penalty)
        epsilon = 0.05  # GN 模型针对标准 SMF 的经验相干因子
        if num_spans_in_path > 0:
            spm_coherent_penalty = float(num_spans_in_path) ** epsilon
        else:
            spm_coherent_penalty = 1.0

        # 计算总噪声功率
        total_noise_power_w = total_spm_power_w * spm_coherent_penalty + total_xpm_power_w + total_ase_power_w

        # 信号功率：由于我们假设 EDFA 完美补偿，每个跨段的信号功率保持为 launch_power_w
        # 但噪声是累积的。所以信号功率应是 launch_power_w
        # SNR = P_signal / (N_NLI + N_ASE)
        if total_noise_power_w <= 0:
            accumulated_snr_db = np.inf
        else:
            accumulated_snr_linear = launch_power_w / total_noise_power_w
            accumulated_snr_db = _lin_to_db(accumulated_snr_linear)

        return accumulated_snr_db, total_noise_power_w
