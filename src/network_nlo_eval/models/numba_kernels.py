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
    k_bar: float = 0.0,  # 泵浦系数
) -> NDArrayFloat:
    """计算经历 SRS 倾斜和光纤损耗后的各信道出纤功率。

    模型来自 Ref. [1] Eq. (14)
    Args:
        p_in_w: 各信道入纤功率 (W)。
        f_rel_hz: 各信道相对频率 (Hz)。
        alpha_power_npm: 各信道功率衰减系数 (Np/m)。
        length_m: 跨段长度 (m)。
        r_f: 拉曼功率转移系数。
        k_bar: 泵浦系数
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
            denom_i = np.sum(p_in_w * np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array[i] * r_f))
            if denom_i > 0:
                p_out_w[i] = (
                    p_in_w[i]
                    * np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array[i] * r_f[i] - alpha_power_npm[i] * length_m)
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
