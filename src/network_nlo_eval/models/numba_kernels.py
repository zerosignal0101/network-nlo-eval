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
    return (2 * PI / lambda_i_m) * (2 * n2 / (a_eff_i_m2 + a_eff_l_m2))


@njit(cache=True)
def _calc_beta2_from_d(d_ps_nm_km: float, lambda_m: float) -> float:
    """将色散参数 D (ps/(nm*km)) 转换为 beta2 (s^2/m)。

    Args:
        d_ps_nm_km: 色散参数 D。
        lambda_m: 波长 (m)。
    Returns:
        float: beta2 (s^2/m)。
    """
    return -d_ps_nm_km * 1e-6 * lambda_m**2 / (2 * PI * C_LIGHT)


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
        lambda_m**2 * s_ps_nm2_km * 1e3 + 2 * lambda_m * d_ps_nm_km * 1e-6
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
    B_t = f_M_hz - f_m_hz
    r_f = np.zeros_like(f_rel_hz)

    # 提前计算系数
    p_per_b = p_total_w / B_t if B_t > 0 else 0.0

    for i in range(len(f_rel_hz)):
        fi = f_rel_hz[i]

        # 定义判断条件
        case1 = (fi - delta_f_co_hz <= f_m_hz) and (fi + delta_f_co_hz >= f_M_hz)
        case2 = (fi - delta_f_co_hz >= f_m_hz) and (fi + delta_f_co_hz <= f_M_hz)
        case3 = (fi - delta_f_co_hz < f_m_hz) and (fi + delta_f_co_hz < f_M_hz)
        case4 = (fi - delta_f_co_hz > f_m_hz) and (fi + delta_f_co_hz > f_M_hz)

        # 优先级顺序至关重要
        if case1:
            r_f[i] = p_total_w * fi
        elif case2:
            r_f[i] = 0.0
        elif case3:
            # Paper Case ③: (Pt/Bt) * (f^2/2 - f*fm + (fM^2 - delta_f_co^2)/2)
            r_f[i] = p_per_b * (0.5 * fi**2 - fi * f_m_hz + 0.5 * (f_M_hz**2 - delta_f_co_hz**2))
        elif case4:
            # Paper Case ④: (Pt/Bt) * (fM*f - f^2/2 - (fm^2 - delta_f_co^2)/2)
            r_f[i] = p_per_b * (f_M_hz * fi - 0.5 * fi**2 - 0.5 * (f_m_hz**2 - delta_f_co_hz**2))
        else:
            # 默认回退到线性
            r_f[i] = p_total_w * fi

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
    L_eff_array = (1 - np.exp(-alpha_power_npm * length_m)) / alpha_power_npm
    p_out_w = np.zeros_like(p_in_w)

    for i in range(N_ch):
        if p_in_w[i] > 0:
            # 使用的是 L_eff_array[i] 标量与 r_f 数组相乘
            denom_i = np.sum(p_in_w * np.exp(-_SRS_CR * L_eff_array[i] * r_f))
            if denom_i > 0:
                p_out_w[i] = (
                    p_in_w[i]
                    * np.exp(-_SRS_CR * L_eff_array[i] * r_f[i] - alpha_power_npm[i] * length_m)
                    * p_total_in
                    / denom_i
                )

    return p_out_w


# =============================================================================
# 3. NLI 噪声方差计算底层模块
# =============================================================================
# EPSILON 防止除零
_EPSILON = 1e-35


@njit(cache=True)
def _calc_alpha_tilde_and_kappa_jit(alpha: float, L_m: float) -> tuple[float, float]:
    """计算论文中提出的有效衰减系数 a_tilde 和 k (Eq. 15, 16)

    用于解决极低损耗或短跨段下的精度问题。
    """
    alpha_L = alpha * L_m
    exp_term = np.exp(-alpha_L)
    num = 1.0 - exp_term
    den = num - alpha_L * exp_term

    # 防止泰勒展开分母极小导致除 0 (对应极短跨段)
    if den < 1e-12:
        alpha_tilde = 2.0 / L_m  # 取泰勒展开极限
    else:
        alpha_tilde = alpha * num / den

    kappa = (alpha_tilde * num) / alpha
    return alpha_tilde, kappa


@njit(cache=True)
def _calc_spm_variance_jit(
    p_i_w: float,
    rs_hz: float,
    n_span: int,
    gamma_ii: float,
    phi_i: float,
    alpha_i_npm: float,
    t_i_tilde: float,
    L_m: float,
    epsilon_n_span: float = 0.05,
) -> float:
    """计算自相位调制 (SPM) 产生的噪声方差 (基于 Eq. 21 第一部分)"""
    T_i = 1.0 + t_i_tilde

    a_t = np.zeros(2)
    k = np.zeros(2)
    # l=0 -> alpha_0 = alpha; l=1 -> alpha_1 = 2*alpha
    a_t[0], k[0] = _calc_alpha_tilde_and_kappa_jit(alpha_i_npm, L_m)
    a_t[1], k[1] = _calc_alpha_tilde_and_kappa_jit(2 * alpha_i_npm, L_m)

    if abs(phi_i) < _EPSILON:
        phi_i = _EPSILON if phi_i >= 0 else -_EPSILON

    pre_factor = (16.0 / 27.0) * (gamma_ii**2 * p_i_w**3 / rs_hz**2) * (T_i**2)
    sum_val = 0.0

    for l_ch in range(2):
        for lp in range(2):
            coeff = (-t_i_tilde / T_i) ** (l_ch + lp)
            num = 2.0 * PI * k[l_ch] * k[lp] * (float(n_span) ** (1.0 + epsilon_n_span))
            den = (a_t[l_ch] + a_t[lp]) * phi_i

            term_asinh = np.arcsinh((3.0 * phi_i * rs_hz**2) / (8.0 * PI * a_t[l_ch])) + np.arcsinh(
                (3.0 * phi_i * rs_hz**2) / (8.0 * PI * a_t[lp])
            )

            sum_val += coeff * (num / den) * term_asinh

    return max(pre_factor * sum_val, 0.0)


@njit(cache=True)
def _calc_xpm_variance_term_jit(
    p_i_w: float,
    p_l_w: float,
    rs_l_hz: float,
    rs_i_hz: float,
    gamma_il: float,
    phi_il: float,
    phi_base: float,
    alpha_l_npm: float,
    t_l_tilde: float,
    L_m: float,
    n_span: int,
    n_tilde: int,
    phi_kurtosis: float,
    delta_f_hz: float,
) -> float:
    """计算交叉相位调制 (XPM) 及峰度修正项产生的噪声方差 (基于 Eq. 21 第二部分)"""
    T_l = 1.0 + t_l_tilde

    a_t = np.zeros(2)
    k = np.zeros(2)
    a_t[0], k[0] = _calc_alpha_tilde_and_kappa_jit(alpha_l_npm, L_m)
    a_t[1], k[1] = _calc_alpha_tilde_and_kappa_jit(2 * alpha_l_npm, L_m)

    if abs(phi_il) < _EPSILON:
        phi_il = _EPSILON if phi_il >= 0 else -_EPSILON
    abs_phi_base = max(abs(phi_base), _EPSILON)

    pre_factor = (32.0 / 27.0) * (gamma_il**2 * p_l_w**2 * p_i_w / rs_l_hz) * (T_l**2)
    sum_val = 0.0

    for l_ch in range(2):
        for lp in range(2):
            coeff = (-t_l_tilde / T_l) ** (l_ch + lp)

            # Term 1: 基础 XPM + 第一跨段格式修正
            num1 = float(n_span) + (5.0 / 6.0) * phi_kurtosis
            den1 = (a_t[l_ch] + a_t[lp]) * phi_il
            atan_term = np.arctan((phi_il * rs_i_hz) / (2.0 * a_t[l_ch])) + np.arctan(
                (phi_il * rs_i_hz) / (2.0 * a_t[lp])
            )
            term1 = (num1 / den1) * atan_term

            # Term 2: 渐进多跨段格式修正
            den2 = abs_phi_base * (rs_l_hz**2) * a_t[l_ch] * a_t[lp]
            num2 = (5.0 / 6.0) * phi_kurtosis * PI * float(n_tilde)

            abs_df = abs(delta_f_hz)
            log_arg = (2.0 * abs_df - rs_l_hz) / (2.0 * abs_df + rs_l_hz)
            log_term = (2.0 * abs_df - rs_l_hz) * np.log(log_arg) if log_arg > 0 else 0.0
            term2 = (num2 / den2) * (log_term + 2.0 * rs_l_hz)

            sum_val += coeff * 2.0 * k[l_ch] * k[lp] * (term1 + term2)

    return max(pre_factor * sum_val, 0.0)


@njit(cache=True, parallel=True)
def _compute_nli_variances(
    n_ch: int,
    n_span: int,
    rs_hz: float,
    p_in_w: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    r_f_values: NDArrayFloat,
    L_m: float,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    cr_w_m_hz: float,
    phi_kurtosis: float,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    sigma_spm_2 = np.zeros(n_ch, dtype=np.float64)
    sigma_xpm_2 = np.zeros(n_ch, dtype=np.float64)
    n_tilde = n_span if n_span > 1 else 0  # 单跨段 n_tilde 为 0

    for i in prange(n_ch):
        if p_in_w[i] <= 0:
            continue

        gamma_ii = (2 * PI / lambdas_m[i]) * (2 * n_2_m2_w / (2 * a_eff_m2[i]))
        phi_i = -4.0 * PI**2 * (beta_2_s2_m + 2.0 * PI * beta_3_s3_m * f_rel_hz[i])

        # 将拉曼转移系数映射为论文的 t_i_tilde = - (Cr * P_tot * f_i) / alpha_i
        # r_f_values 内部已计算 P_tot * f_i 的等效值
        t_i_tilde = -(cr_w_m_hz * r_f_values[i]) / alpha_power_npm[i]

        sigma_spm_2[i] = _calc_spm_variance_jit(
            p_in_w[i], rs_hz, n_span, gamma_ii, phi_i, alpha_power_npm[i], t_i_tilde, L_m, epsilon_n_span=0.05
        )

        xpm_sum_term = 0.0
        for l_idx in range(n_ch):
            if l_idx == i or p_in_w[l_idx] <= 0:
                continue

            gamma_il = (2 * PI / lambdas_m[i]) * (2 * n_2_m2_w / (a_eff_m2[i] + a_eff_m2[l_idx]))

            delta_f_hz = f_rel_hz[l_idx] - f_rel_hz[i]
            phi_base = -4.0 * PI**2 * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[i] + f_rel_hz[l_idx]))
            phi_il = delta_f_hz * phi_base

            t_l_tilde = -(cr_w_m_hz * r_f_values[l_idx]) / alpha_power_npm[l_idx]

            xpm_sum_term += _calc_xpm_variance_term_jit(
                p_i_w=p_in_w[i],
                p_l_w=p_in_w[l_idx],
                rs_l_hz=rs_hz,
                rs_i_hz=rs_hz,
                gamma_il=gamma_il,
                phi_il=phi_il,
                phi_base=phi_base,
                alpha_l_npm=alpha_power_npm[l_idx],
                t_l_tilde=t_l_tilde,
                L_m=L_m,
                n_span=n_span,
                n_tilde=n_tilde,
                phi_kurtosis=phi_kurtosis,
                delta_f_hz=delta_f_hz,
            )

        sigma_xpm_2[i] = xpm_sum_term

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
    for i in range(N_ch):
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
        NDArrayFloat: 每个信道的 ASE 噪声方差 (W)。
    """
    total_ase_variance = np.zeros(len(p_in_w), dtype=np.float64)
    for i in range(len(p_in_w)):
        if p_in_w[i] > 0 and p_out_w[i] > 0:  # 只有有功率的信道才产生 ASE 噪声
            # 计算单跨段增益 (假设 EDFA 补偿损耗)
            G_i = p_in_w[i] / p_out_w[i]

            total_ase_variance[i] = n_span * nf_lin[i] * H_PLANCK * f_abs_hz[i] * G_i * rs_hz
            # print(f"{G_i}, {p_in_w[i]}/{p_out_w[i]}", end=", ")
            # print(f"Channel {i} ASE variance: {total_ase_variance[i]:.2e}")

    return total_ase_variance  # 返回方差 (W)


@njit(cache=True)
def _calc_span_noise_and_power_jit(
    # --- 输入状态 ---
    power_in_w: NDArrayFloat,  # 各信道入纤功率 (W)
    span_length_m: float,  # 当前跨段长度 (m)
    # --- 频谱属性 (预计算好的数组) ---
    f_rel_hz: NDArrayFloat,  # 相对频率 (Hz)
    f_abs_hz: NDArrayFloat,  # 绝对频率 (Hz)
    lambdas_m: NDArrayFloat,  # 波长 (m)
    # --- 光纤物理参数 (由 FiberSpanConfig 转换) ---
    alpha_power_npm: NDArrayFloat,  # 功率衰减系数 (Np/m)
    a_eff_m2: NDArrayFloat,  # 有效模面积 (m^2)
    beta2_s2_m: float,  # 色散 beta2
    beta3_s3_m: float,  # 色散 beta3
    n2_m2_w: float,  # 非线性折射率
    # --- 系统与放大器参数 ---
    nf_lin_channels: NDArrayFloat,  # 放大器线性噪声系数 (数组)
    symbol_rate_hz: float,  # 符号速率 (Hz)
    n_effective_spans: int,  # 有效累积跨段数 (NLI计算用)
    # --- 物理常数 ---
    cr_w_m_hz: float,  # SRS 增益系数 Cr
    delta_f_co_hz: float,  # SRS 截断频率
    phi_kurtosis: float = 0.0,  # 0.0为高斯(默认), -0.6190为64QAM
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """【高性能内核】单跨段物理层评估的 JIT 版本。

    封装了损耗、SRS、SPM、XPM 和 ASE 的全套计算。
    """
    num_channels = len(power_in_w)

    # 1. 预计算中间量
    total_p_in_w = np.sum(power_in_w)
    # L_eff: 有效长度
    l_eff_m = (1.0 - np.exp(-alpha_power_npm * span_length_m)) / alpha_power_npm

    # 2. 计算 SRS 功率转移分布 (r_f)
    # f_m_hz 和 f_M_hz 在内核中根据 f_rel 动态确定
    f_m_hz = f_rel_hz[0] - symbol_rate_hz / 2.0
    f_M_hz = f_rel_hz[-1] + symbol_rate_hz / 2.0

    r_f_values = _calc_raman_profile_jit(f_rel_hz, f_m_hz, f_M_hz, delta_f_co_hz, total_p_in_w)

    # 3. 计算出纤功率 (考虑衰减与 SRS 倾斜)
    p_out_after_fiber = _calc_span_out_power_jit(power_in_w, f_rel_hz, alpha_power_npm, span_length_m, r_f_values)

    # 4. 计算 NLI 噪声方差 (SPM & XPM)
    sigma2_spm_w, sigma2_xpm_w = _compute_nli_variances(
        num_channels,
        n_effective_spans,
        symbol_rate_hz,
        power_in_w,
        f_rel_hz,
        lambdas_m,
        a_eff_m2,
        alpha_power_npm,
        r_f_values,
        span_length_m,
        beta2_s2_m,
        beta3_s3_m,
        n2_m2_w,
        cr_w_m_hz,
        phi_kurtosis,
    )

    # 5. 计算 ASE 噪声方差
    sigma2_ase_w = _calculate_ase_noise_variance(
        power_in_w, p_out_after_fiber, f_abs_hz, nf_lin_channels, symbol_rate_hz, n_effective_spans
    )

    return p_out_after_fiber, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w
