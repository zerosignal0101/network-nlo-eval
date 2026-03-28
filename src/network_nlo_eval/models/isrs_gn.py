"""多波段广义高斯噪声 (Generalized Gaussian Noise, GGN) 模型评估器。

ISRS-GN (Inter-channel Stimulated Raman Scattering - Generalized Noise)
"""

from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat, NodeID
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_nf_lin_jit,
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
    _calculate_ase_noise_variance,
    _compute_nli_variances,
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
        power_in_w: NDArrayFloat,  # 各信道入纤功率 (W)
        span_length_m: float,  # 跨段长度 (m)
        edfa_config: EDFAConfig,  # EDFA 配置
        roadm_config: ROADMConfig | None = None,  # ROADM 配置 (可选)
        n_effective_spans: int = 1,  # 对于单个跨段，这通常是 1
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
        # 1. 计算有效长度 L_eff (考虑损耗)
        l_eff_m = (1 - np.exp(-self.alpha_power_npm_channels * span_length_m)) / self.alpha_power_npm_channels

        # 2. 计算拉曼功率转移系数 r_f
        total_p_in_w = np.sum(power_in_w)
        r_f_values = _calc_raman_profile_jit(self.f_rel_hz, self.f_m_hz, self.f_M_hz, self.delta_f_co_hz, total_p_in_w)

        # 3. 计算有效衰减因子 T_i (用于 NLI 模型)
        # T_i = (2 * alpha_i - C_r * r_f_i)^2
        t_factors = (2 * self.alpha_power_npm_channels - self.cr_w_m_hz * r_f_values) ** 2

        # 4. 计算经历光纤传播 (衰减 + SRS 倾斜) 后的功率
        # p_out_after_fiber 包含衰减和 SRS 功率倾斜
        p_out_after_fiber = _calc_span_out_power_jit(
            power_in_w,
            self.f_rel_hz,
            self.alpha_power_npm_channels,
            span_length_m,
            r_f_values,
        )

        # 5. 计算 NLI 噪声 (SPM 和 XPM)
        sigma2_spm_w, sigma2_xpm_w = _compute_nli_variances(
            self.grid.num_channels,
            n_effective_spans,
            self.symbol_rate_hz,
            power_in_w,
            self.f_rel_hz,
            self.lambdas_m,
            self.a_eff_m2_channels,
            self.alpha_power_npm_channels,
            t_factors,
            r_f_values,
            l_eff_m,
            self.beta2_s2_m,
            self.beta3_s3_m,
            self.n2_m2_w,
            self.cr_w_m_hz,
            self.k_bar,
        )

        # 6. 计算 ASE 噪声
        # EDFA 增益应补偿跨段损耗 + ROADM 损耗
        # 假设每个 EDFA 补偿了光纤损耗，以及可选的 ROADM 损耗
        # 严格来讲，EDFA 增益和 NF 会根据其入纤功率和目标功率进行动态调整
        # 这里简化：EDFA 噪声与 NF 和目标增益有关
        # span_loss_db = _lin_to_db(np.sum(power_in_w[power_in_w > 0]))
        # - _lin_to_db(np.sum(p_out_after_fiber[p_out_after_fiber > 0]))
        # edfa_gain_db = edfa_config.target_gain_db if edfa_config else 0.0 # 假设 EDFA 有固定的目标增益

        # 简化 ASE：每个放大器引入的 ASE = NF * h * f * G * B_ch
        # 这里返回的是功率 (W)
        sigma2_ase_w = _calculate_ase_noise_variance(
            power_in_w,
            p_out_after_fiber,
            self.f_abs_hz,
            self.nf_lin_channels,
            self.symbol_rate_hz,
            n_effective_spans,
        )

        # 7. 考虑 EDFA 增益和 ROADM 损耗，计算最终出纤功率
        edfa_gain_linear = _db_to_lin(edfa_config.target_gain_db)
        # p_out_w = p_out_after_fiber * edfa_gain_linear
        # 如果 EDFA 增益补偿了光纤损耗 (并考虑 ROADM 损耗)，那么 P_out 应该回到 P_in 的水平
        # 严格来说，EDFA 应该补偿光纤损耗。

        # 为了简化，假设 EDFA 完美补偿了该跨段的光纤损耗和 ROADM 损耗，使功率回到额定发射功率水平。
        # 这样在 QoT 验证器中，可以简单地使用传入的 `launch_power_w` 作为每段的输入功率。
        # 但这里仍应计算实际出纤功率以供参考。

        # 考虑光纤损耗、SRS倾斜、EDFA增益、ROADM损耗后的实际出纤功率
        # 注意：此处 EDFA 增益如果设定为补偿光纤损耗，则 p_out 应该接近 p_in。
        # p_out_after_edfa = p_out_after_fiber * edfa_gain_linear
        # roadm_loss_linear = _db_to_lin(roadm_config.insertion_loss_db
        # + roadm_config.filtering_penalty_db) if roadm_config else 1.0
        # p_out_w = p_out_after_edfa / roadm_loss_linear

        # 为了 RWA 链路评估的实用性，p_out_w 应该是每个 EDFA 后的 *有效* 输出功率
        # 我们可以假设 EDFA 的增益等于光纤损耗 + ROADM 损耗，从而使得每个跨段的有效功率保持恒定。
        # 这样，所有 NLI 和 ASE 噪声都会被累加。

        # 为了简单，这里直接返回衰减和 SRS 后的功率，EDFA 增益在网络层面统一处理。
        # 返回的功率是经过光纤链路但 *未* 经过放大器/ROADM 的功率
        return p_out_after_fiber, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w

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
        # 1/SNR_total = SUM(1/SNR_span_i)

        # 为了简化，我们假设每个跨段后的 EDFA 完美补偿了该跨段的损耗 (包括 ROADM 损耗)，
        # 使得每个跨段的输入功率可以视为恒定为 launch_power_w。
        # 这样 NLI 和 ASE 噪声方差可以直接累加。

        for i in range(num_spans_in_path):
            u_node_idx = path[i]
            v_node_idx = path[i + 1]
            link_key = tuple(sorted((u_node_idx, v_node_idx)))

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
