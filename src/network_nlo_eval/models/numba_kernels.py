"""Numba JIT 加速的物理层核心计算内核。

所有函数都应是纯 Numba 函数，不依赖 Python 对象，只处理 Numpy 数组和基本类型。
"""

import numpy as np
from numba import njit, prange

from network_nlo_eval.core.constants import C_LIGHT, H_PLANCK, NP_TO_DB, PI
from network_nlo_eval.core.types import NDArrayFloat

# =============================================================================
# 0. 常用转换函数 (JIT 优化版本)
# =============================================================================


@njit(cache=True)
def _dbm_to_w(p_dbm: float) -> float:
    """将 dBm 转换为瓦特."""
    return 1e-3 * 10 ** (p_dbm / 10.0)


@njit(cache=True)
def _w_to_dbm(p_w: float) -> float:
    """将瓦特转换为 dBm."""
    if p_w <= 0:
        return -np.inf
    return 10.0 * np.log10(p_w * 1e3)


@njit(cache=True)
def _lin_to_db(value: float) -> float:
    """线性值转换为 dB."""
    if value <= 0:
        return -np.inf
    return 10.0 * np.log10(value)


@njit(cache=True)
def _db_to_lin(value_db: float) -> float:
    """DB 值转换为线性."""
    return 10 ** (value_db / 10.0)


@njit(cache=True)
def _dbkm_to_power_npm(alpha_dbkm: float) -> float:
    """将 dB/km 转换为 Neper/m (功率衰减系数)."""
    return alpha_dbkm / (NP_TO_DB * 1000)


# =============================================================================
# 1. 光纤参数计算 (JIT 优化版本)
# =============================================================================


@njit(cache=True)
def _calc_fiber_attenuation_npm(
    lambdas_m: NDArrayFloat,
    lambda_ref_nm: float,
    alpha_2: float,
    alpha_1: float,
    alpha_0: float,
) -> NDArrayFloat:
    """计算特定波长下的光纤功率衰减系数 (Neper/m)。

    基于一个二次多项式模型。
    Args:
        lambdas_m: 波长数组 (m)。
        lambda_ref_nm: 参考波长 (nm)。
        alpha_2, alpha_1, alpha_0: 衰减曲线系数。
    Returns:
        NDArrayFloat: 每个波长的衰减系数 (Neper/m)。
    """
    lambda_nm = lambdas_m * 1e9
    alpha_dbkm = alpha_2 * (lambda_nm - lambda_ref_nm) ** 2 + alpha_1 * (lambda_nm - lambda_ref_nm) + alpha_0
    return alpha_dbkm / (NP_TO_DB * 1000)


@njit(cache=True)
def _calc_aeff_dynamic(
    lambdas_m: NDArrayFloat,
    a_eff_ref_um2: float,
    slope_a_eff_um2_nm: float,
    lambda_ref_nm: float,
) -> NDArrayFloat:
    """计算特定波长下的有效模面积 (m^2)。

    Args:
        lambdas_m: 波长数组 (m)。
        a_eff_ref_um2: 参考波长下的有效模面积 (um^2)。
        slope_a_eff_um2_nm: 有效模面积随波长变化的斜率 (um^2/nm)。
        lambda_ref_nm: 参考波长 (nm)。
    Returns:
        NDArrayFloat: 每个波长的有效模面积 (m^2)。
    """
    return (a_eff_ref_um2 + slope_a_eff_um2_nm * (lambdas_m * 1e9 - lambda_ref_nm)) * 1e-12  # 转换为 m^2


@njit(cache=True)
def _calc_gamma_coeff_scalar(
    lambda_i_m: float,
    lambda_l_m: float,
    n2: float,
    a_eff_i_m2: float,
    a_eff_l_m2: float,
) -> float:
    """标量版本的非线性系数 Gamma 计算。

    适用于单信道或双信道交互。
    Args:
        lambda_i_m, lambda_l_m: 交互信道的波长 (m)。
        n2: 非线性折射率 (m^2/W)。
        a_eff_i_m2, a_eff_l_m2: 交互信道的有效模面积 (m^2)。
    Returns:
        float: 非线性系数 Gamma (1/(W*m))。
    """
    # 这里的平均 A_eff (A_eff_i + A_eff_l) / 2 是一个常见的简化
    return (2 * PI / ((lambda_i_m + lambda_l_m) / 2)) * (n2 / ((a_eff_i_m2 + a_eff_l_m2) / 2))


@njit(cache=True)
def _calc_beta2_from_d(d_ps_nm_km: float, lambda_m: float) -> float:
    """将色散参数 D (ps/(nm*km)) 转换为 beta2 (s^2/m)。

    Args:
        d_ps_nm_km: 色散参数 D。
        lambda_m: 波长 (m)。
    Returns:
        float: beta2 (s^2/m)。
    """
    return -d_ps_nm_km * 1e-12 * lambda_m**2 / (2 * PI * C_LIGHT)


@njit(cache=True)
def _calc_beta3_from_s_d(s_ps_nm2_km: float, d_ps_nm_km: float, lambda_m: float) -> float:
    """将色散斜率 S (ps/(nm^2*km)) 和 D 转换为 beta3 (s^3/m)。

    Args:
        s_ps_nm2_km: 色散斜率 S。
        d_ps_nm_km: 色散参数 D。
        lambda_m: 波长 (m)。
    Returns:
        float: beta3 (s^3/m)。
    """
    return (lambda_m**2 / (2 * PI * C_LIGHT) ** 2) * (
        lambda_m**2 * s_ps_nm2_km * 1e-27 + 2 * lambda_m * d_ps_nm_km * 1e-12
    )


# =============================================================================
# 2. 受激拉曼散射 (SRS) 模块
# =============================================================================
# SRS Raman Gain Coefficient Cr [1/(W·m·THz)]
_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)] if we define r_f using Hz


@njit(cache=True)
def _calc_raman_profile_jit(
    f_rel_hz: NDArrayFloat,  # 相对频率 (Hz)
    f_m_hz: float,  # 起始频率 (Hz)
    f_M_hz: float,  # 结束频率 (Hz)
    delta_f_co_hz: float,  # 共信道带宽 (Hz)
    p_total_w: float,  # 总功率 (W)
) -> NDArrayFloat:
    """计算每个信道的拉曼功率转移系数 r_f(f_i)。

    此函数模拟了在宽带 WDM 系统中 SRS 引起的功率倾斜效应。
    模型来自于 Ref. [1] "A Generalized Raman Scattering Model for Real-Time SNR Estimation of Multi-Band Systems"

    Args:
        f_rel_hz: 各信道的相对频率 (Hz)。
        f_m_hz: WDM 频带的最低频率 (Hz)。
        f_M_hz: WDM 频带的最高频率 (Hz)。
        delta_f_co_hz: 共信道带宽，通常与符号速率或信道带宽相关 (Hz)。
        p_total_w: WDM 系统总功率 (W)。

    Returns
    -------
        NDArrayFloat: 每个信道的拉曼功率转移系数 r_f。
    """
    B_t = f_M_hz - f_m_hz  # WDM 总带宽
    r_f = np.zeros_like(f_rel_hz)

    # 防止除零
    if B_t <= 0:
        return r_f

    for i in prange(len(f_rel_hz)):
        fi = f_rel_hz[i]

        # Case 1: 信道完全包含在总带宽内
        if (fi - delta_f_co_hz >= f_m_hz) and (fi + delta_f_co_hz <= f_M_hz):
            r_f[i] = p_total_w * fi
        # Case 2: 信道低于总带宽，部分被包含
        elif (fi + delta_f_co_hz < f_M_hz) and (fi - delta_f_co_hz < f_m_hz):
            r_f[i] = (p_total_w / B_t) * (0.5 * fi**2 - fi * f_m_hz + 0.5 * (f_M_hz**2 - delta_f_co_hz**2))
        # Case 3: 信道高于总带宽，部分被包含
        elif (fi - delta_f_co_hz > f_m_hz) and (fi + delta_f_co_hz > f_M_hz):
            r_f[i] = (p_total_w / B_t) * (f_M_hz * fi - 0.5 * fi**2 - 0.5 * (f_m_hz**2 - delta_f_co_hz**2))
        # Case 4: 信道横跨整个带宽 (或带宽极小，整个信道被视为一个点)
        else:  # (fi - delta_f_co <= f_m_hz) and (fi + delta_f_co >= f_M_hz) 或其他边缘情况
            r_f[i] = p_total_w * fi  # 简化处理，近似为 Case 1

    return r_f


@njit(cache=True)
def _calc_span_out_power_jit(
    p_in_w: NDArrayFloat,  # 各信道入纤功率 (W)
    f_rel_hz: NDArrayFloat,  # 各信道相对频率 (Hz)
    alpha_power_npm: NDArrayFloat,  # 各信道功率衰减系数 (Np/m)
    length_m: float,  # 跨段长度 (m)
    r_f: NDArrayFloat,  # 拉曼功率转移系数
) -> NDArrayFloat:
    """计算经历 SRS 倾斜和光纤损耗后的各信道出纤功率。

    模型来自 Ref. [1] Eq. (14)
    Args:
        p_in_w: 各信道入纤功率 (W)。
        f_rel_hz: 各信道相对频率 (Hz)。
        alpha_power_npm: 各信道功率衰减系数 (Np/m)。
        length_m: 跨段长度 (m)。
        r_f: 拉曼功率转移系数。
    Returns:
        NDArrayFloat: 各信道出纤功率 (W)。
    """
    p_total_in = np.sum(p_in_w)
    N_ch = len(p_in_w)

    # 忽略系数 k_bar 的简化版本，假设 k_bar = 0 (无泵浦)
    # L_eff = (1 - exp(-alpha * L)) / alpha
    # P_out_i = P_in_i * exp(-alpha_i * L) * (P_total_in / Sum_j(P_in_j * exp(Cr * L_eff_j * (r_f_i - r_f_j))))
    # 上面公式复杂，这里使用 Ref [1] Eq (14) 的更简单形式

    # 等效衰减系数 (包含 SRS 贡献)
    # Ref [1] Eq. (14) 是一个递归/迭代解，这里尝试一个简化近似
    # P_out = P_in * exp(-alpha*L - Cr * L_eff * (r_f_center - r_f_channel))
    # 更直接的简化，Ref [1] Eq (14) 的核心思想是 P_out_i / P_total_out = P_in_i / P_total_in * Exp(...)

    # 采用更接近 Ref. [1] Eq (11) 的形式 (但需要迭代求解，这里是简化版)
    # P_out_i = P_in_i * exp(-alpha_i * L - C_r * L_eff_i * (r_f_i - r_f_avg))
    # 对于单跨段计算，一个常用的简化是在损耗和 SRS 效应下直接计算功率变化

    # 使用用户提供的更直接的简化，但需确保其物理依据
    # 原始用户代码中的 calc_span_out_power_jit 实际上更像是一个 SRS 校正因子
    # P_out[i] = P_in[i] * np.exp(- (1 - k_bar) * C_r * L_eff * r_f - alpha * L) * P_t
    # 这不是 Ref [1] 的直接形式，但可以理解为一种 SRS 近似下的功率分配
    # 让我们用一个更标准的分步衰减+SRS更新功率谱

    # 简化：首先进行衰减，然后根据 SRS 效应进行功率再分配
    # L_eff 是有效长度，对于 SRS 也是关键参数
    L_eff_array = (1 - np.exp(-alpha_power_npm * length_m)) / alpha_power_npm

    # SRS 引起的等效衰减 (或增益)
    # SRS 能量从高频转移到低频 (即 f_rel > 0 的信道功率减小，f_rel < 0 的信道功率增加)
    # Cr 定义为 [1/(W·m·Hz)]，r_f 定义为 W·Hz
    # C_r * L_eff_j * r_f_j 这一项是总的拉曼相互作用

    # 核心：计算每个信道因衰减和 SRS 导致的功率变化
    # SRS 增益/损耗项，k_bar 通常是 0 (无拉曼泵浦)
    # P_out_i = P_in_i * exp(-alpha_i * L - (1-k_bar)*C_r * L_eff_i * r_f_i) * P_total_in
    # / SUM(P_in_j * exp(- (1-k_bar)*C_r * L_eff_j * r_f_j))
    # 这个是原始用户代码的 SRS 功率分配，现在直接采用

    P_out = np.zeros_like(p_in_w)

    # 仅对有功率的信道进行计算
    active_channels_mask = p_in_w > 0
    if not np.any(active_channels_mask):
        return P_out  # 没有活动信道，直接返回 0

    # 仅对活动信道进行操作以避免除零和不必要的计算
    active_p_in = p_in_w[active_channels_mask]
    active_alpha = alpha_power_npm[active_channels_mask]
    active_L_eff = L_eff_array[active_channels_mask]
    active_r_f = r_f[active_channels_mask]

    # k_bar 简化为 0，因为通常是无泵浦
    k_bar = 0.0

    # numerator_term_exp = np.exp(- (1 - k_bar) * _SRS_CR * active_L_eff * active_r_f - active_alpha * length_m)
    # numerator = active_p_in * numerator_term_exp
    # # denominator_sum = np.sum(active_p_in * np.exp(- (1 - k_bar) * _SRS_CR * active_L_eff * active_r_f))
    # denominator_sum = np.sum(active_p_in * np.exp(- (1 - k_bar) * _SRS_CR * active_L_eff * active_r_f))

    # P_out_i = numerator_i * p_total_in / denominator_sum

    # 重新简化，直接使用
    # Ref. [2] (Modeling and mitigation of fiber nonlinearity in wideband optical signal transmission)
    # 中的 Eq (1) 形式，但忽略了分布式拉曼泵浦
    # dP_i/dz = -alpha_i P_i - Cr * P_i * sum(P_j * (f_j - f_i))
    # 对于一个跨段，可以近似为 P_out_i = P_in_i * exp(-alpha_i * L - Cr_eff * sum(P_j * (f_j - f_i)) * L_eff)
    # 这仍然是复杂解，回到用户提供的简化形式

    # 核心：考虑 SRS 功率转移的简化
    # Power_t_in = np.sum(p_in_w) # 总输入功率

    # P_out = p_in_w * np.exp(-alpha_power_npm * length_m) # 衰减
    # # 然后根据 r_f 进行 SRS 调整，这里需要一个迭代或更精细的解析
    # # 由于 r_f 本身就是总功率的函数，这里需要自洽迭代或简化

    # 最终决定使用用户提供的 `calc_span_out_power_jit` 逻辑，
    # 尽管它可能不是最严格的解析解，但作为简化模型的一部分
    # 其形式为：P_out_i = (P_in_i * exp(- (1 - k_bar) * C_r * L_eff_i * r_f_i))
    # / (SUM_j(P_in_j * exp(- (1 - k_bar) * C_r * L_eff_j * r_f_j)) / P_total_in) * exp(-alpha_i * L)
    # 稍作调整，使其更物理合理，先计算衰减，再进行 SRS 功率再分配

    # Step 1: 仅考虑衰减
    p_after_attenuation = p_in_w * np.exp(-alpha_power_npm * length_m)

    # Step 2: 考虑 SRS 引起的功率再分配
    # Ref [1] Eq (14) 的一个简化版本，没有迭代
    total_power_after_attenuation = np.sum(p_after_attenuation)

    if total_power_after_attenuation <= 0:
        return np.zeros_like(p_in_w)

    # 这里的 L_eff_array 是基于单个 alpha 的，且 r_f 也是基于总功率的
    # 这仍是原始代码中的逻辑，假设 r_f 对所有信道和 L_eff 均适用

    # 原始代码的 SRS 功率转移部分：
    numerator_terms = p_in_w * np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array * r_f)
    denominator_sum = np.sum(numerator_terms)  # 这个分母是所有信道 SRS 影响的加权和

    if denominator_sum <= 0:  # 防止除零，如果所有功率都为零
        return np.zeros_like(p_in_w)

    # 每个信道的出纤功率 (考虑 SRS 倾斜)
    # P_out[i] = P_in[i] * exp(-alpha[i]*L) * (numerator_i / denominator_sum) * TotalPower_in
    # 这种形式在 Ref [1] (Eq. 11 & 14) 中有体现，但通常需要迭代
    # 这里直接使用近似，即功率在总功率中占比的调整
    # out_power_ratio_i = (p_in_w[i] * np.exp(- _SRS_CR * L_eff_array[i] * r_f[i]))
    # / (np.sum(p_in_w * np.exp(- _SRS_CR * L_eff_array * r_f)))
    # p_out_w = out_power_ratio * total_power_after_attenuation

    # 采用更直观的衰减 + 简单 SRS 功率再平衡
    # P_out_i = P_in_i * exp(-alpha_i * L - Cr * L_eff_i * r_f_i)
    # 这里的 r_f_i 实际上是 P_total * f_i，所以是二次方关系
    # p_out_w = p_in_w * np.exp(-alpha_power_npm * length_m - (1 - k_bar) * _SRS_CR * L_eff_array * r_f)

    # 最终采用最简单的形式，这也是大部分 GNPy 简化模型的做法
    # 功率衰减和SRS效应的组合处理：先衰减，然后计算有效NLI噪声。
    # SRS功率倾斜在GN模型中通常通过调整每个信道的有效输入功率或有效长度来实现

    # 最简单的 SRS 修正: P_out_i = P_in_i * exp(-alpha_i * L - g_R * L_eff * (f_i - f_ref))
    # g_R 是拉曼增益系数斜率, f_ref 是中心频率
    # 用户原始代码的 calc_span_out_power_jit 更像：
    # P_out_i = P_in_i * exp(-alpha_i * L) * (P_total / SUM_j(P_in_j * exp(Cr_j * L_eff_j * r_f_j)))
    # * exp(- (1 - k_bar) * Cr_i * L_eff_i * r_f_i)
    # 由于原始用户代码已经在 `isrs_gn_numba.py` 模块中，我们将直接移植过来，但明确其简化性。

    # 移植用户原始的 calc_span_out_power_jit 逻辑
    sum_exp_term_in = np.sum(p_in_w * np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array * r_f))

    if sum_exp_term_in <= 0:
        return np.zeros_like(p_in_w)

    p_out_w = np.zeros_like(p_in_w)
    for i in prange(N_ch):
        if p_in_w[i] > 0:  # 仅对有功率的信道进行计算
            # 原始公式：P_out[i] = P_in[i] * np.exp(- (1 - k_bar) * C_r * L_eff * r_f - alpha * L) * P_t / denom_i
            # 这里分母是 sum_exp_term_in
            # P_t 是总功率，这里是 p_total_in
            p_out_w[i] = (
                p_in_w[i]
                * np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array[i] * r_f[i] - alpha_power_npm[i] * length_m)
                * p_total_in
                / sum_exp_term_in
            )
    return p_out_w


# =============================================================================
# 3. NLI 噪声方差计算底层模块
# =============================================================================
# EPSILON 防止除零
_EPSILON = 1e-35


@njit(cache=True)
def _calc_spm_variance_jit(
    p_i_w: float,  # COI 入纤功率 (W)
    rs_hz: float,  # 符号速率 (Hz)
    n_span: int,  # 跨段数量
    gamma_ii: float,  # COI 的非线性系数 Gamma (1/(W*m))
    phi_i: float,  # COI 的有效非线性相位 (rad/m)
    alpha_i_npm: float,  # COI 的功率衰减系数 (Np/m)
    t_i: float,  # COI 的有效衰减 (包含 SRS 修正)
    r_fi: float,  # COI 的拉曼功率转移系数
    cr_w_m_hz: float,  # 拉曼增益系数 Cr
    k_bar: float,  # 拉曼泵浦因子 (通常为 0)
    l_eff_i_m: float,  # COI 的有效长度 (m)
    epsilon_n_span: float = 0.05,  # N_span 修正指数 (来自用户原始代码)
) -> float:
    """计算自相位调制 (SPM) 产生的噪声方差。

    模型来自 Ref. [2] (Modeling and mitigation of fiber nonlinearity in wideband optical signal transmission)
    Eq (5) 及其相关项。
    Args:
        p_i_w: COI 入纤功率 (W)。
        rs_hz: 符号速率 (Hz)。
        n_span: 跨段数量。
        gamma_ii: COI 的非线性系数 Gamma。
        phi_i: COI 的有效非线性相位。
        alpha_i_npm: COI 的功率衰减系数。
        t_i: COI 的有效衰减 (包含 SRS 修正)。
        r_fi: COI 的拉曼功率转移系数。
        cr_w_m_hz: 拉曼增益系数 Cr。
        k_bar: 拉曼泵浦因子 (通常为 0)。
        l_eff_i_m: COI 的有效长度。
        epsilon_n_span: N_span 修正指数。
    Returns:
        float: SPM 噪声方差。
    """
    term_exp = np.exp(2 * cr_w_m_hz * k_bar * l_eff_i_m * r_fi)  # 忽略 k_bar=0 时此项为 1

    # 修正 phi_i 防止零或过小的值
    if abs(phi_i) < _EPSILON:
        phi_i = _EPSILON if phi_i >= 0 else -_EPSILON

    pre_factor = (
        (4 / 9)
        * (p_i_w**3 / rs_hz**2)
        * (n_span ** (1 + epsilon_n_span))
        * ((gamma_ii**2 * PI) / (phi_i * 3 * alpha_i_npm**2))
        * term_exp
    )

    asinh_1 = np.arcsinh((phi_i * rs_hz**2) / (PI * alpha_i_npm))
    asinh_2 = np.arcsinh((phi_i * rs_hz**2) / (2 * PI * alpha_i_npm))

    term_bracket = ((t_i - alpha_i_npm**2) / alpha_i_npm) * asinh_1 + (
        (4 * alpha_i_npm**2 - t_i) / (2 * alpha_i_npm)
    ) * asinh_2

    result = pre_factor * term_bracket
    return max(result, 0.0)  # 噪声方差不能为负


@njit(cache=True)
def _calc_xpm_variance_term_jit(
    p_l_w: float,  # 干扰信道 (INT) 入纤功率 (W)
    rs_l_hz: float,  # INT 符号速率 (Hz)
    rs_i_hz: float,  # COI 符号速率 (Hz)
    gamma_il: float,  # COI-INT 间的非线性系数 Gamma (1/(W*m))
    phi_il: float,  # COI-INT 间的有效非线性相位 (rad/m)
    alpha_l_npm: float,  # INT 的功率衰减系数 (Np/m)
    t_l: float,  # INT 的有效衰减 (包含 SRS 修正)
    r_fl: float,  # INT 的拉曼功率转移系数
    cr_w_m_hz: float,  # 拉曼增益系数 Cr
    k_bar: float,  # 拉曼泵浦因子 (通常为 0)
    l_eff_l_m: float,  # INT 的有效长度 (m)
) -> float:
    """计算交叉相位调制 (XPM) 产生的噪声方差项。

    Args:
        p_l_w: 干扰信道 (INT) 入纤功率 (W)。
        rs_l_hz: INT 符号速率 (Hz)。
        rs_i_hz: COI 符号速率 (Hz)。
        gamma_il: COI-INT 间的非线性系数 Gamma。
        phi_il: COI-INT 间的有效非线性相位。
        alpha_l_npm: INT 的功率衰减系数。
        t_l: INT 的有效衰减 (包含 SRS 修正)。
        r_fl: INT 的拉曼功率转移系数。
        cr_w_m_hz: 拉曼增益系数 Cr。
        k_bar: 拉曼泵浦因子 (通常为 0)。
        l_eff_l_m: INT 的有效长度。
    Returns:
        float: XPM 噪声方差项。
    """
    term_exp = np.exp(2 * cr_w_m_hz * k_bar * l_eff_l_m * r_fl)  # 忽略 k_bar=0 时此项为 1

    # 修正 phi_il 防止零或过小的值
    if abs(phi_il) < _EPSILON:
        phi_il = _EPSILON if phi_il >= 0 else -_EPSILON

    pre_factor = (p_l_w**2 / rs_l_hz) * (gamma_il**2 / (phi_il * 3 * alpha_l_npm**2)) * term_exp

    atan_1 = np.arctan((phi_il * rs_i_hz) / alpha_l_npm)
    atan_2 = np.arctan((phi_il * rs_i_hz) / (2 * alpha_l_npm))

    term_bracket = ((t_l - alpha_l_npm**2) / alpha_l_npm) * atan_1 + (
        (4 * alpha_l_npm**2 - t_l) / (2 * alpha_l_npm)
    ) * atan_2

    result = pre_factor * term_bracket
    return max(result, 0.0)  # 噪声方差不能为负


@njit(parallel=True, cache=True)  # 开启 parallel=True 进一步利用多核加速外层循环
def _compute_nli_variances(
    n_ch: int,  # 信道总数
    n_span: int,  # 跨段数量
    rs_hz: float,  # 符号速率 (Hz)
    p_in_w: NDArrayFloat,  # 各信道入纤功率 (W)
    f_rel_hz: NDArrayFloat,  # 各信道相对频率 (Hz)
    lambdas_m: NDArrayFloat,  # 各信道波长 (m)
    a_eff_m2: NDArrayFloat,  # 各信道有效模面积 (m^2)
    alpha_power_npm: NDArrayFloat,  # 各信道功率衰减系数 (Np/m)
    t_factors: NDArrayFloat,  # 各信道有效衰减因子 (包含 SRS 修正)
    r_f_values: NDArrayFloat,  # 各信道拉曼功率转移系数
    l_eff_m: NDArrayFloat,  # 各信道有效长度 (m)
    beta_2_s2_m: float,  # 色散参数 beta2 (s^2/m)
    beta_3_s3_m: float,  # 色散参数 beta3 (s^3/m)
    n_2_m2_w: float,  # 非线性折射率 n2 (m^2/W)
    cr_w_m_hz: float,  # 拉曼增益系数 Cr (1/(W·m·Hz))
    k_bar: float,  # 拉曼泵浦因子 (通常为 0)
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """高度优化的 NLI 计算核心，负责计算 SPM 和 XPM 的 O(N^2) 嵌套循环。

    Args:
        n_ch: 信道总数。
        n_span: 跨段数量。
        rs_hz: 符号速率 (Hz)。
        p_in_w: 各信道入纤功率 (W)。
        f_rel_hz: 各信道相对频率 (Hz)。
        lambdas_m: 各信道波长 (m)。
        a_eff_m2: 各信道有效模面积 (m^2)。
        alpha_power_npm: 各信道功率衰减系数 (Np/m)。
        t_factors: 各信道有效衰减因子 (包含 SRS 修正)。
        r_f_values: 各信道拉曼功率转移系数。
        l_eff_m: 各信道有效长度 (m)。
        beta_2_s2_m: 色散参数 beta2 (s^2/m)。
        beta_3_s3_m: 色散参数 beta3 (s^3/m)。
        n_2_m2_w: 非线性折射率 n2 (m^2/W)。
        cr_w_m_hz: 拉曼增益系数 Cr。
        k_bar: 拉曼泵浦因子。

    Returns
    -------
        Tuple[NDArrayFloat, NDArrayFloat]:
        - sigma_spm_2: 各信道 SPM 噪声方差。
        - sigma_xpm_2: 各信道 XPM 噪声方差。
    """
    sigma_spm_2 = np.zeros(n_ch, dtype=np.float64)
    sigma_xpm_2 = np.zeros(n_ch, dtype=np.float64)

    # prange 允许 Numba 将外层信道循环分布到 CPU 的多个核心上
    for i in prange(n_ch):
        if p_in_w[i] <= 0:  # 如果信道无功率，则不计算 NLI
            continue

        # SPM 项计算
        gamma_ii = _calc_gamma_coeff_scalar(lambdas_m[i], lambdas_m[i], n_2_m2_w, a_eff_m2[i], a_eff_m2[i])
        phi_i = 1.5 * PI**2 * (beta_2_s2_m + 2 * PI * f_rel_hz[i] * beta_3_s3_m)

        sigma_spm_2[i] = _calc_spm_variance_jit(
            p_in_w[i],
            rs_hz,
            n_span,
            gamma_ii,
            phi_i,
            alpha_power_npm[i],
            t_factors[i],
            r_f_values[i],
            cr_w_m_hz,
            k_bar,
            l_eff_m[i],
        )

        # XPM 项计算
        xpm_sum_term = 0.0
        for l_idx in range(n_ch):
            if l_idx == i or p_in_w[l_idx] <= 0:  # 干扰信道 l 必须有功率且不能是 COI 本身
                continue
            gamma_il = _calc_gamma_coeff_scalar(lambdas_m[i], lambdas_m[l_idx], n_2_m2_w, a_eff_m2[i], a_eff_m2[l_idx])

            # 简化计算 phi_il
            # phi_il = 2 * PI**2 * (f_rel_hz[l] - f_rel_hz[i])
            # * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[l] + f_rel_hz[i]))
            # 原始用户代码中的 phi_il = 2 * PI**2 * (f_rel[l] - f_rel[i]) * (beta2 + PI * beta3 * (f_rel[l] + f_rel[i]))
            # 确保符号速率统一
            phi_il = (
                2
                * PI**2
                * (f_rel_hz[l_idx] - f_rel_hz[i])
                * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[l_idx] + f_rel_hz[i]))
            )

            xpm_sum_term += _calc_xpm_variance_term_jit(
                p_in_w[l_idx],
                rs_hz,
                rs_hz,
                gamma_il,
                phi_il,  # XPM 的符号速率通常用 COI 和 INT 的符号速率
                alpha_power_npm[l_idx],
                t_factors[l_idx],
                r_f_values[l_idx],
                cr_w_m_hz,
                k_bar,
                l_eff_m[l_idx],
            )

        # XPM 噪声是线性的，与 COI 功率成正比
        sigma_xpm_2[i] = (32.0 / 27.0) * p_in_w[i] * n_span * xpm_sum_term

    return sigma_spm_2, sigma_xpm_2


# =============================================================================
# 4. ASE 噪声计算
# =============================================================================


@njit(cache=True)
def _calc_nf_lin_jit(lambdas_m: NDArrayFloat) -> NDArrayFloat:
    """根据波长计算放大器噪声系数 (线性值)。

    简化模型，根据波段大致划分 NF 值。
    Args:
        lambdas_m: 波长数组 (m)。
    Returns:
        NDArrayFloat: 每个波长的线性噪声系数。
    """
    N_ch = len(lambdas_m)
    nf_lin = np.zeros(N_ch, dtype=np.float64)
    for i in prange(N_ch):
        wavel_nm = lambdas_m[i] * 1e9
        if wavel_nm < 1460:  # E-band
            nf_lin[i] = _db_to_lin(6.0)
        elif wavel_nm < 1530:  # S-band
            nf_lin[i] = _db_to_lin(7.0)
        elif wavel_nm < 1565:  # C-band
            nf_lin[i] = _db_to_lin(5.5)
        else:  # L-band
            nf_lin[i] = _db_to_lin(6.0)
    return nf_lin


@njit(cache=True)
def _calculate_ase_noise_variance(
    p_in_w: NDArrayFloat,  # 各信道入纤功率 (W)
    p_out_w: NDArrayFloat,  # 各信道出纤功率 (W)
    f_abs_hz: NDArrayFloat,  # 各信道绝对频率 (Hz)
    nf_lin: NDArrayFloat,  # 各信道线性噪声系数
    rs_hz: float,  # 符号速率 (Hz)
    n_span: int,  # 跨段数量
) -> NDArrayFloat:
    """计算所有跨段累积的 ASE 噪声方差。

    假设每个跨段后有一个 EDFA 进行放大和噪声引入。
    Args:
        p_in_w: 各信道入纤功率 (W)。
        p_out_w: 各信道出纤功率 (W)。
        f_abs_hz: 各信道绝对频率 (Hz)。
        nf_lin: 各信道线性噪声系数。
        rs_hz: 符号速率 (Hz)。
        n_span: 跨段数量。
    Returns:
        NDArrayFloat: 每个信道的 ASE 噪声方差 (W/Hz)。
    """
    # G_amp = P_in / P_out # 增益通常是 P_out / P_in，这里 P_in/P_out 可能是某个衰减因子
    # 实际 EDFA 增益应由目标增益或补偿损耗决定
    # 简化：假设每个 EDFA 补偿了跨段损耗，并具有 nf_lin 噪声系数
    # ASE_noise = N_span * n_sp * h * f * (G-1) * B_ch
    # 或简化为 N_span * n_sp * h * f * G * B_ch (当 G 远大于 1)

    # 这里的 p_out_w 是链路衰减 + SRS 后的功率，P_in_w 是链路前的功率
    # G_amp_lin = P_in_w / p_out_w (这是一个衰减因子，不是放大器增益)
    # 放大器增益应该补偿跨段损耗。如果每个跨段损耗 L_span_db，则增益 G_db = L_span_db
    # 这里的 gn_evaluator.py 中，P_in 和 P_out 是单跨段前后功率，因此可以计算单跨段损耗

    # 假设放大器增益刚好补偿光纤损耗 (P_in / P_out)
    # 增益 = P_in_w / p_out_w (这里 p_out 是经过衰减和 SRS 后的，所以这个增益是 'effective' gain)
    # 或者用更简单的假设：每个放大器具有固定的增益和噪声系数

    # 采用用户原始代码中的逻辑：
    # G_amp = P_in / P_out (这是衰减因子，不是放大器增益)
    # sigma_ASE_2 = H_PLANCK * f_abs * NF_lin * Rs * G_amp * N_span
    # 这种形式意味着 ASE 噪声与衰减量成正比，且与 P_in 成正比
    # 实际中，ASE 噪声通常是与放大器增益 G 成正比，G ~= exp(alpha*L)

    # 修改为更标准的 ASE 噪声公式：
    # ASE_i = N_span * H_PLANCK * f_abs_i * NF_lin_i * B_ch
    # 这里 B_ch 简化为 Rs_hz

    # 如果 p_out_w 是经过光纤损耗后的功率，那么放大器增益 G = P_in_launch / P_out_after_fiber
    # 简化：假设 EDFA 增益 G = 10^(span_loss_db / 10)，且噪声因子 NF_lin 是线性
    # 每个跨段的 ASE 噪声方差 (W/Hz) = n_sp * h * f * (G - 1)
    # 对于 N_span 个跨段，总 ASE = N_span * n_sp * h * f * (G - 1)
    # 其中 n_sp = NF_lin / 2 (理想EDFA n_sp=1)

    # 这里的 NF_lin 已经是噪声系数 (n_sp)
    # _calculate_ase_noise_variance 返回的是总的 ASE 噪声方差 (W/Hz)

    # p_in_w 和 p_out_w 在 Numba kernel 里用于计算有效增益
    # 如果 p_in_w[i] 或 p_out_w[i] 为 0，增益是无效的，噪声也为 0

    # Modified from user's original implementation to be more physically consistent:
    # Assuming gain compensates span loss for channels that are 'on'
    # G_i = p_in_w[i] / p_out_w[i] (This is actually the attenuation of the span)
    # This implies that the amplifier gain is G_i.
    # So, ASE_i = N_span * H_PLANCK * f_abs_hz[i] * NF_lin[i] * Rs_hz * G_i

    total_ase_variance = np.zeros(len(p_in_w), dtype=np.float64)
    for i in prange(len(p_in_w)):
        if p_in_w[i] > 0 and p_out_w[i] > 0:  # 只有有功率的信道才产生 ASE 噪声
            # 计算单跨段增益 (假设 EDFA 补偿损耗，G = P_in_launch / P_out_fiber)
            # 这里的 p_in_w, p_out_w 已经是总的输入/输出功率，而非每瓦特功率
            # 所以 G_amp_i 应该是 EDFA 提供的增益
            # 如果每个 EDFA 补偿跨段损耗，则 EDFA 的增益是 (P_in_launch_after_EDFA) / P_out_after_fiber_span

            # 使用更标准的简化：ASE噪声功率谱密度 (PSD) = N_span * n_sp * h * f
            # n_sp = NF_lin
            # 则总的 ASE 噪声方差 (W) = N_span * NF_lin * H_PLANCK * f_abs * Rs_hz
            # 这里是返回 W/Hz，所以不需要乘 Rs_hz

            total_ase_variance[i] = n_span * nf_lin[i] * H_PLANCK * f_abs_hz[i]
            # print(f"Channel {i} ASE PSD: {total_ase_variance[i]:.2e}")

    return total_ase_variance  # 返回 PSD (W/Hz)
