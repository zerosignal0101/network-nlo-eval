"""Numba JIT 加速的物理层核心计算内核。

所有函数都应是纯 Numba 函数，不依赖 Python 对象，只处理 Numpy 数组和基本类型。
"""

from typing import Any

import numpy as np
from numba import njit

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
@njit(cache=True)
def _calc_dp_dz(p: NDArrayFloat, G: NDArrayFloat, f_hz: NDArrayFloat, alpha_npm: NDArrayFloat) -> NDArrayFloat:
    """计算功率随距离的导数 dP/dz (基于 Paper 2 Eq. 3)"""
    N = len(p)
    dp = np.zeros(N, dtype=np.float64)
    for i in range(N):
        gain_term = 0.0
        loss_term = 0.0
        for k in range(N):
            if f_hz[k] > f_hz[i]:
                # k 频率更高 (波长更短)，k 泵浦 i
                gain_term += G[i, k] * p[k]
            elif f_hz[k] < f_hz[i]:
                # i 频率更高 (波长更短)，i 泵浦 k
                # 能量守恒转移: 损失与增益呈频率比
                loss_term += (f_hz[k] / f_hz[i]) * G[i, k] * p[k]

        dp[i] = (gain_term - loss_term - alpha_npm[i]) * p[i]
    return dp


@njit(cache=True)
def _calc_exact_power_profile_ode(
    p_in_w: NDArrayFloat, f_hz: NDArrayFloat, alpha_npm: NDArrayFloat, L_m: float, cr_w_m_hz: float, n_steps: int = 20
) -> NDArrayFloat:
    """使用 RK4 求解器计算经历精确拉曼转移和损耗的出纤功率。"""
    N = len(p_in_w)

    # 1. 预计算增益矩阵 (引入实际的拉曼谱形：13.2 THz 达峰，15.6 THz 归零)
    G = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for k in range(N):
            df_thz = abs(f_hz[i] - f_hz[k]) / 1e12
            if df_thz <= 13.2:
                # 线性增长区
                g = cr_w_m_hz * (df_thz * 1e12)
            elif df_thz < 15.6:
                # 峰值跌落区 (匹配 Fig. 4)
                peak_g = cr_w_m_hz * 13.2e12
                g = peak_g * (15.6 - df_thz) / (15.6 - 13.2)
            else:
                # 频差过大，不再产生受激拉曼
                g = 0.0
            G[i, k] = g

    # 2. Runge-Kutta 4 (RK4) 积分
    dz = L_m / n_steps
    p = p_in_w.copy()

    for _ in range(n_steps):
        k1 = _calc_dp_dz(p, G, f_hz, alpha_npm)
        k2 = _calc_dp_dz(p + 0.5 * dz * k1, G, f_hz, alpha_npm)
        k3 = _calc_dp_dz(p + 0.5 * dz * k2, G, f_hz, alpha_npm)
        k4 = _calc_dp_dz(p + dz * k3, G, f_hz, alpha_npm)

        p += (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # 强制不出现负功率
        for i in range(N):
            if p[i] < 0:
                p[i] = 0.0

    return p


# =============================================================================
# 2b. FG-4a: G 矩阵的提升（缓存）与去分支（两次 matvec）
# =============================================================================
# 背景：`_calc_exact_power_profile_ode` 里的 G 只依赖 `f_hz`、`cr_w_m_hz` 与
# 谱形常数（13.2 / 15.6 THz），**与功率无关**，却在每跨段用双层循环重建。
# FG-2 的 F2-7 把 `_calc_exact_power_profile_ode` 的函数体逐字节冻结了，因此
# 这里**不修改**那个函数，而是另建两个等价变体：调用侧预计算 G 并缓存，
# 变体直接接收它。
#
# 两个变体都刻意复制冻结函数的 RK4 循环（逐步、逐步长、逐步累加顺序一致），
# 唯一的差别是不再重建 G。`_calc_exact_power_profile_ode_with_g` 内部仍调用
# 未改动的 `_calc_dp_dz`，因此输出与基线**逐位相等**；
# `_calc_exact_power_profile_ode_matvec` 换用去分支的两次 matvec，求和次序
# 改变，偏差量级见 docs/fg4a_accel_design.md。

#: 拉曼谱形常数：线性增长区上界与峰值跌落区上界 (THz)。
_RAMAN_LINEAR_LIMIT_THZ = 13.2
_RAMAN_FLOOR_THZ = 15.6


@njit(cache=True)
def _build_g_matrix(f_hz: NDArrayFloat, cr_w_m_hz: float) -> NDArrayFloat:
    """构建完整的拉曼增益矩阵 G (N×N)。

    算术与 `_calc_exact_power_profile_ode` 内的双层循环**逐条一致**（同样的
    阈值、同样的运算次序），因此在调用侧预计算的 G 与核内重建的 G 逐位相同。

    Args:
        f_hz: 各节点绝对频率 (Hz)。
        cr_w_m_hz: SRS 增益系数 Cr (1/(W·m·Hz))。
    Returns:
        NDArrayFloat: (N, N) 拉曼增益矩阵。
    """
    n = len(f_hz)
    g = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            df_thz = abs(f_hz[i] - f_hz[k]) / 1e12
            if df_thz <= _RAMAN_LINEAR_LIMIT_THZ:
                # 线性增长区
                value = cr_w_m_hz * (df_thz * 1e12)
            elif df_thz < _RAMAN_FLOOR_THZ:
                # 峰值跌落区 (匹配 Fig. 4)
                peak_g = cr_w_m_hz * 13.2e12
                value = peak_g * (_RAMAN_FLOOR_THZ - df_thz) / (_RAMAN_FLOOR_THZ - _RAMAN_LINEAR_LIMIT_THZ)
            else:
                # 频差过大，不再产生受激拉曼
                value = 0.0
            g[i, k] = value
    return g


@njit(cache=True)
def _build_g_pair_matrices(f_hz: NDArrayFloat, cr_w_m_hz: float) -> tuple[NDArrayFloat, NDArrayFloat]:
    """把 G 预分为严格上三角的增益部分与严格下三角的加权损耗部分。

    基线 `_calc_dp_dz` 逐对判断 `f_hz[k] > f_hz[i]`，把贡献累加进两个标量。
    这里把判断**搬到构建期**：频率更高的列只进 `g_gain`，频率更低的列只进
    `g_loss`（并预先乘上能量守恒的频率比 f_k/f_i）。于是运行期退化为两次 matvec。

    与基线的浮点对应关系：`g_loss[i, k] = (f_hz[k] / f_hz[i]) * G[i, k]`，
    而基线累加的是 `(f_hz[k] / f_hz[i]) * G[i, k] * p[k]`。逐项的乘法次序
    完全相同，差别只在于求和由一次 k 序号的标量累加变成 BLAS 的 dot。

    对角线与同频元素在基线中两个分支都不命中（贡献为 0），此处同样留 0。

    Args:
        f_hz: 各节点绝对频率 (Hz)。
        cr_w_m_hz: SRS 增益系数 Cr (1/(W·m·Hz))。
    Returns:
        tuple: (g_gain, g_loss)，均为 (N, N) float64。
    """
    n = len(f_hz)
    g_gain = np.zeros((n, n), dtype=np.float64)
    g_loss = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        f_i = f_hz[i]
        for k in range(n):
            f_k = f_hz[k]
            if f_k > f_i:
                df_thz = (f_k - f_i) / 1e12
                if df_thz <= _RAMAN_LINEAR_LIMIT_THZ:
                    g_gain[i, k] = cr_w_m_hz * (df_thz * 1e12)
                elif df_thz < _RAMAN_FLOOR_THZ:
                    peak_g = cr_w_m_hz * 13.2e12
                    g_gain[i, k] = peak_g * (_RAMAN_FLOOR_THZ - df_thz) / (
                        _RAMAN_FLOOR_THZ - _RAMAN_LINEAR_LIMIT_THZ
                    )
                # else: 频差过大，保持 0
            elif f_k < f_i:
                df_thz = (f_i - f_k) / 1e12
                if df_thz <= _RAMAN_LINEAR_LIMIT_THZ:
                    gain = cr_w_m_hz * (df_thz * 1e12)
                elif df_thz < _RAMAN_FLOOR_THZ:
                    peak_g = cr_w_m_hz * 13.2e12
                    gain = peak_g * (_RAMAN_FLOOR_THZ - df_thz) / (
                        _RAMAN_FLOOR_THZ - _RAMAN_LINEAR_LIMIT_THZ
                    )
                else:
                    gain = 0.0
                # 能量守恒转移：损耗与增益呈频率比
                g_loss[i, k] = (f_k / f_i) * gain
    return g_gain, g_loss


@njit(cache=True)
def _calc_dp_dz_matvec(
    p: NDArrayFloat, g_gain: NDArrayFloat, g_loss: NDArrayFloat, alpha_npm: NDArrayFloat
) -> NDArrayFloat:
    """去分支的 dP/dz：两次 matvec 取代逐对 if。

    `dP/dz = p * (g_gain @ p - g_loss @ p - alpha)`。

    与 `_calc_dp_dz` 相比没有改变数学式，只把逐对分支换成 BLAS dot；因此这是
    FG-4c GPU/batch 化的形状（整块矩阵乘向量，无数据相关的控制流）。
    """
    gain_term = g_gain @ p
    loss_term = g_loss @ p
    n = len(p)
    dp = np.empty(n, dtype=np.float64)
    for i in range(n):
        dp[i] = (gain_term[i] - loss_term[i] - alpha_npm[i]) * p[i]
    return dp


@njit(cache=True)
def _calc_exact_power_profile_ode_with_g(
    p_in_w: NDArrayFloat,
    G: NDArrayFloat,
    f_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    L_m: float,
    n_steps: int = 20,
) -> NDArrayFloat:
    """FG-4a 变体一：接收**预计算并缓存**的 G，内部仍用原始 `_calc_dp_dz`。

    RK4 循环与冻结的 `_calc_exact_power_profile_ode` 逐步一致（同样的 4 次
    `_calc_dp_dz` 调用、同样的组合系数、同样的负功率截断），唯一差别是跳过
    G 的双层重建。因此输出与基线**逐位相等**。
    """
    N = len(p_in_w)
    dz = L_m / n_steps
    p = p_in_w.copy()

    for _ in range(n_steps):
        k1 = _calc_dp_dz(p, G, f_hz, alpha_npm)
        k2 = _calc_dp_dz(p + 0.5 * dz * k1, G, f_hz, alpha_npm)
        k3 = _calc_dp_dz(p + 0.5 * dz * k2, G, f_hz, alpha_npm)
        k4 = _calc_dp_dz(p + dz * k3, G, f_hz, alpha_npm)

        p += (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        for i in range(N):
            if p[i] < 0:
                p[i] = 0.0

    return p


@njit(cache=True)
def _calc_exact_power_profile_ode_matvec(
    p_in_w: NDArrayFloat,
    g_gain: NDArrayFloat,
    g_loss: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    L_m: float,
    n_steps: int = 20,
) -> NDArrayFloat:
    """FG-4a 变体二：去分支 matvec 积分器（`_calc_dp_dz_matvec` 版）。

    RK4 循环与冻结函数逐步一致，唯一差别是 4 次子调用换成 matvec 版本。
    求和次序因此改变，偏差量级为浮点舍入 (见 docs/fg4a_accel_design.md)。
    """
    N = len(p_in_w)
    dz = L_m / n_steps
    p = p_in_w.copy()

    for _ in range(n_steps):
        k1 = _calc_dp_dz_matvec(p, g_gain, g_loss, alpha_npm)
        k2 = _calc_dp_dz_matvec(p + 0.5 * dz * k1, g_gain, g_loss, alpha_npm)
        k3 = _calc_dp_dz_matvec(p + 0.5 * dz * k2, g_gain, g_loss, alpha_npm)
        k4 = _calc_dp_dz_matvec(p + dz * k3, g_gain, g_loss, alpha_npm)

        p += (dz / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        for i in range(N):
            if p[i] < 0:
                p[i] = 0.0

    return p


# -----------------------------------------------------------------------------
# G 缓存：按 (频率栅格, cr_w_m_hz, 谱形常数) 记忆化
# -----------------------------------------------------------------------------
# 只读数组的缓存键：`f_hz.tobytes()` 是**值**键，因此两个数值相同但对象不同的
# 频率数组会命中同一条缓存，而不会误用别人的矩阵。
_G_CACHE: dict[tuple[bytes, float, float, float, int], tuple[Any, ...]] = {}
_G_CACHE_STATS = {"hits": 0, "misses": 0, "builds": 0}
_G_CACHE_MAX_ENTRIES = 32


def g_cache_stats() -> dict[str, int]:
    """返回 G 缓存的命中/未命中计数——用于证明跨调用复用确实发生。"""
    return dict(_G_CACHE_STATS)


def _g_cache_key(f_hz: NDArrayFloat, cr_w_m_hz: float) -> tuple[bytes, float, float, float, int]:
    """G 缓存键：频率栅格的值、Cr、两条谱形常数与形状。"""
    return (
        np.ascontiguousarray(f_hz, dtype=np.float64).tobytes(),
        float(cr_w_m_hz),
        _RAMAN_LINEAR_LIMIT_THZ,
        _RAMAN_FLOOR_THZ,
        int(len(f_hz)),
    )


def _cache_put(key: tuple[Any, ...], value: tuple[Any, ...]) -> None:
    """写入缓存；条目过多时整体清空（键含完整频率栅格，清空只是重算）。"""
    if len(_G_CACHE) >= _G_CACHE_MAX_ENTRIES:
        _G_CACHE.clear()
    _G_CACHE[key] = value


def raman_g_matrix(f_hz: NDArrayFloat, cr_w_m_hz: float) -> NDArrayFloat:
    """返回**缓存的**完整拉曼增益矩阵 G (N×N)。

    同一 (频率栅格, Cr, 谱形常数) 的后续调用直接命中缓存，不再逐对重建。
    返回值被标记为只读，以免调用方就地改坏共享矩阵。

    Args:
        f_hz: 各节点绝对频率 (Hz)。
        cr_w_m_hz: SRS 增益系数 Cr (1/(W·m·Hz))。
    Returns:
        NDArrayFloat: (N, N) 只读的拉曼增益矩阵。
    """
    key = _g_cache_key(f_hz, cr_w_m_hz)
    cached = _G_CACHE.get(key)
    if cached is not None:
        _G_CACHE_STATS["hits"] += 1
        return cached[0]
    _G_CACHE_STATS["misses"] += 1
    g = _build_g_matrix(np.ascontiguousarray(f_hz, dtype=np.float64), cr_w_m_hz)
    g.flags.writeable = False
    _G_CACHE_STATS["builds"] += 1
    _cache_put(key, (g,))
    return g


def raman_g_pair_matrices(f_hz: NDArrayFloat, cr_w_m_hz: float) -> tuple[NDArrayFloat, NDArrayFloat]:
    """返回**缓存的**去分支矩阵对 (g_gain, g_loss)，均为 (N, N) 只读。

    与 :func:`raman_g_matrix` 共用同一套缓存键，两者可同时常驻。

    Args:
        f_hz: 各节点绝对频率 (Hz)。
        cr_w_m_hz: SRS 增益系数 Cr (1/(W·m·Hz))。
    Returns:
        tuple: (g_gain, g_loss)。
    """
    key = _g_cache_key(f_hz, cr_w_m_hz)
    cached = _G_CACHE.get(key)
    if cached is not None and len(cached) == 3:
        _G_CACHE_STATS["hits"] += 1
        return cached[1], cached[2]
    _G_CACHE_STATS["misses"] += 1
    f = np.ascontiguousarray(f_hz, dtype=np.float64)
    g_gain, g_loss = _build_g_pair_matrices(f, cr_w_m_hz)
    g_gain.flags.writeable = False
    g_loss.flags.writeable = False
    _G_CACHE_STATS["builds"] += 1
    if cached is None:
        _cache_put(key, (None, g_gain, g_loss))
    else:  # pragma: no cover - 仅当已有 full-G 条目时走到
        _G_CACHE[key] = (cached[0], g_gain, g_loss)
    return g_gain, g_loss


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


@njit(cache=True)
def _compute_nli_variances(
    n_ch: int,
    n_span: int,
    symbol_rate_hz: NDArrayFloat,  # 逐连接符号率 (Hz)，形状 (N,)
    p_in_w: NDArrayFloat,
    p_out_exact_w: NDArrayFloat,  # 传入精确的数值解
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    L_m: float,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    phi_kurtosis: float,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    sigma_spm_2 = np.zeros(n_ch, dtype=np.float64)
    sigma_xpm_2 = np.zeros(n_ch, dtype=np.float64)
    n_tilde = n_span if n_span > 1 else 0

    # 反向计算等效拉曼倾斜 t_i_tilde
    # 使得 GN 积分解析模型的前后功率与 ODE 高精度积分的前后功率匹配
    t_tilde_array = np.zeros(n_ch, dtype=np.float64)
    for i in range(n_ch):
        p_lin_out = p_in_w[i] * np.exp(-alpha_power_npm[i] * L_m)
        if p_lin_out > 0:
            g_raman = p_out_exact_w[i] / p_lin_out
            # 拟合 rho(L) = exp(-alpha*L) * [1 + t_tilde * (1 - exp(-alpha*L))]
            t_tilde_array[i] = (g_raman - 1.0) / (1.0 - np.exp(-alpha_power_npm[i] * L_m))

    for i in range(n_ch):
        if p_in_w[i] <= 0:
            continue

        gamma_ii = (2 * np.pi / lambdas_m[i]) * (2 * n_2_m2_w / (2 * a_eff_m2[i]))
        phi_i = -4.0 * np.pi**2 * (beta_2_s2_m + 2.0 * np.pi * beta_3_s3_m * f_rel_hz[i])

        t_i_tilde = t_tilde_array[i]

        sigma_spm_2[i] = _calc_spm_variance_jit(
            p_in_w[i], symbol_rate_hz[i], n_span, gamma_ii, phi_i, alpha_power_npm[i], t_i_tilde, L_m, epsilon_n_span=0.05
        )

        xpm_sum_term = 0.0
        for l_idx in range(n_ch):
            if l_idx == i or p_in_w[l_idx] <= 0:
                continue

            gamma_il = (2 * np.pi / lambdas_m[i]) * (2 * n_2_m2_w / (a_eff_m2[i] + a_eff_m2[l_idx]))

            delta_f_hz = f_rel_hz[l_idx] - f_rel_hz[i]
            phi_base = -4.0 * np.pi**2 * (beta_2_s2_m + np.pi * beta_3_s3_m * (f_rel_hz[i] + f_rel_hz[l_idx]))
            phi_il = delta_f_hz * phi_base

            t_l_tilde = t_tilde_array[l_idx]

            xpm_sum_term += _calc_xpm_variance_term_jit(
                p_i_w=p_in_w[i],
                p_l_w=p_in_w[l_idx],
                rs_l_hz=symbol_rate_hz[l_idx],
                rs_i_hz=symbol_rate_hz[i],
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
# 3b. FG-4b: NLI 张量化 —— 把构建期常量折进三张 (N,N) 张量
# =============================================================================
# 上面那个 `_compute_nli_variances` 是 FG-2 冻结的参考实现，**一个字节都不改**。
# 本节另建等价变体，与 FG-4a 对 `dp_dz` 的处理同构。
#
# 关键观察（这是本刀的立足点）：XPM 的 O(N²) 双重循环里，**没有任何一项依赖
# 逐跨段的功率**。进入 `_compute_nli_variances` 的每个量要么是逐连接常量
# （`symbol_rate_hz` / `f_rel_hz` / `lambdas_m` / `a_eff_m2` / `alpha_power_npm`），
# 要么是跨段常量（`L_m` / `beta_2` / `beta_3` / `n_2` / `phi_kurtosis` / `n_span`），
# 唯一的数据相关量是 **t_tilde**——而它只通过
#
#     T_l = 1 + t_tilde[l]        （出现在 pre_factor 的 T_l^2）
#     r_l  = -t_tilde[l] / T_l    （出现在 (l_ch, lp) 求和的系数 r_l^(l_ch+lp)）
#
# 两处进入。注意基线里 `a_t` / `k` 是按 **`alpha_l_npm`（即 l 通道的衰减）** 建的，
# 与配对无关——它们本来就只是 (N,2) 的逐通道量，不是 (N,N)。基线在双层循环里
# 反复重算它们，这本身就是 O(N²) 里可去掉的常数因子。
#
# 于是把 (l_ch, lp) ∈ {(0,0),(0,1),(1,0),(1,1)} 的四项按 r 的幂次归并：
#
#     sum_val[i,l] = W0 + 2*r_l*W1 + r_l^2*W2
#
# 其中 W0/W1/W2 是三张 (N,N) 张量，**全部在构建期一次算好**（含 arcsinh / arctan /
# log 这些超越函数，以及 `log_arg > 0` 的分支选择、`|phi_il| < eps` 的钳位、
# `den < 1e-12` 的泰勒兜底）。热路径只剩乘加与广播。
#
# 逐对相位项 `phi_il` 按 brief 要求写成外积形式：
# `phi_il = (f_l - f_i) * phi_base(f_l + f_i)`，两个因子都是 (N,N) 外积。
#
# 逐连接带宽的语义**原样保留**：`rs_l_hz`（COI 自己的带宽）只进 term2 的对数项与
# 分母，`rs_i_hz`（扰动信道带宽）只进 term1 的 arctan。两者都取自
# `symbol_rate_hz`，没有退回成同一个标量。
#
# 与基线的浮点对应关系：逐项的乘法次序完全一致，差别只在 (1) l 方向求和由顺序
# 累加变成成对求和，(2) (0,1) 与 (1,0) 两项按浮点交换律合并为一项 2*r*W1。
# 偏差量级与证据见 docs/fg4b_nli_vectorization.md。

#: SPM 的跨段指数因子 (n_span ** (1 + epsilon_n_span))。
_NLI_SPAN_EXPONENT = 1.05


@njit(cache=True)
def _build_nli_tensors(
    n_ch: int,
    n_span: int,
    symbol_rate_hz: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    L_m: float,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    phi_kurtosis: float,
) -> tuple[Any, ...]:
    """构建 NLI 热路径所需的全部常量张量（构建期调用一次并缓存）。

    所有数据相关的分支/选择都在这里做完——热路径不再有 ``if``：
    ``log_arg > 0`` 的选择、``|phi_il| < eps`` 的钳位、以及 `_calc_alpha_tilde_and_kappa_jit`
    内部的 ``den < 1e-12`` 泰勒兜底。

    Returns:
        tuple: ``(a0, a1, k0, k1, spm0, spm1, spm2, gamma_ii_sq, gamma_sq, w0, w1, w2)``

        * ``a0, a1, k0, k1`` — (N,) 逐通道的 a_tilde / kappa（l = 0, 1 两档）；
        * ``spm0, spm1, spm2`` — (N,) SPM 三项（已折入 coeff 幂次以外的常量，
          不含跨段指数因子与 pre_factor）；
        * ``gamma_ii_sq`` — (N,) 自相位 Gamma^2；
        * ``gamma_sq`` — (N, N) 逐对 Gamma_il^2；
        * ``w0, w1, w2`` — (N, N) XPM 的 (0,0) / (0,1)=(1,0) / (1,1) 合并项。
    """
    n_tilde = n_span if n_span > 1 else 0

    # --- 逐通道：a_tilde 与 kappa（基线在双层循环内反复重算的同一批常量）---
    a0 = np.empty(n_ch, dtype=np.float64)
    a1 = np.empty(n_ch, dtype=np.float64)
    k0 = np.empty(n_ch, dtype=np.float64)
    k1 = np.empty(n_ch, dtype=np.float64)
    for idx in range(n_ch):
        a0[idx], k0[idx] = _calc_alpha_tilde_and_kappa_jit(alpha_power_npm[idx], L_m)
        a1[idx], k1[idx] = _calc_alpha_tilde_and_kappa_jit(2 * alpha_power_npm[idx], L_m)

    # --- SPM 系数（逐通道，无跨段数据依赖）---
    gamma_ii_sq = np.empty(n_ch, dtype=np.float64)
    spm0 = np.empty(n_ch, dtype=np.float64)
    spm1 = np.empty(n_ch, dtype=np.float64)
    spm2 = np.empty(n_ch, dtype=np.float64)
    for i in range(n_ch):
        gamma_ii = (2 * np.pi / lambdas_m[i]) * (2 * n_2_m2_w / (2 * a_eff_m2[i]))
        gamma_ii_sq[i] = gamma_ii**2

        phi_i = -4.0 * np.pi**2 * (beta_2_s2_m + 2.0 * np.pi * beta_3_s3_m * f_rel_hz[i])
        if abs(phi_i) < _EPSILON:
            phi_i = _EPSILON if phi_i >= 0 else -_EPSILON

        # asinh 的自变量只依赖 phi_i 与 a_t，故逐通道一次算好
        arg0 = (3.0 * phi_i * symbol_rate_hz[i] ** 2) / (8.0 * np.pi * a0[i])
        arg1 = (3.0 * phi_i * symbol_rate_hz[i] ** 2) / (8.0 * np.pi * a1[i])
        asinh0 = np.arcsinh(arg0)
        asinh1 = np.arcsinh(arg1)
        # num = 2*pi*k[l]*k[lp]*span_pow，span_pow 留到运行期乘（对它线性）
        spm0[i] = 2.0 * np.pi * k0[i] * k0[i] / ((a0[i] + a0[i]) * phi_i) * (asinh0 + asinh0)
        spm1[i] = 2.0 * np.pi * k0[i] * k1[i] / ((a0[i] + a1[i]) * phi_i) * (asinh0 + asinh1)
        spm2[i] = 2.0 * np.pi * k1[i] * k1[i] / ((a1[i] + a1[i]) * phi_i) * (asinh1 + asinh1)

    # --- XPM：逐对外积 ---
    gamma_sq = np.empty((n_ch, n_ch), dtype=np.float64)
    w0 = np.empty((n_ch, n_ch), dtype=np.float64)
    w1 = np.empty((n_ch, n_ch), dtype=np.float64)
    w2 = np.empty((n_ch, n_ch), dtype=np.float64)

    num1 = n_span + (5.0 / 6.0) * phi_kurtosis
    num2 = (5.0 / 6.0) * phi_kurtosis * np.pi * n_tilde

    for i in range(n_ch):
        # 逐对带宽语义：rs_i 是扰动信道 i 的符号率，只进 term1 的 arctan。
        rs_i = symbol_rate_hz[i]
        f_i = f_rel_hz[i]
        lam_i = lambdas_m[i]
        aeff_i = a_eff_m2[i]
        for j in range(n_ch):
            gamma_il = (2 * np.pi / lam_i) * (2 * n_2_m2_w / (aeff_i + a_eff_m2[j]))
            gamma_sq[i, j] = gamma_il**2

            # 逐对相位项写成外积：(f_l - f_i) * [-4*pi^2*(b2 + pi*b3*(f_i+f_l))]
            delta_f_hz = f_rel_hz[j] - f_i
            phi_base = -4.0 * np.pi**2 * (beta_2_s2_m + np.pi * beta_3_s3_m * (f_i + f_rel_hz[j]))
            phi_il = delta_f_hz * phi_base
            if abs(phi_il) < _EPSILON:
                phi_il = _EPSILON if phi_il >= 0 else -_EPSILON
            abs_phi_base = max(abs(phi_base), _EPSILON)

            # rs_l 是 j 通道（COI）自己的带宽：只进 term2 的对数与分母。
            rs_l = symbol_rate_hz[j]

            # term1: atan 在构建期求值
            t0 = np.arctan((phi_il * rs_i) / (2.0 * a0[j]))
            t1 = np.arctan((phi_il * rs_i) / (2.0 * a1[j]))

            # term2 的 log 选择在构建期完成（log_arg > 0 与基线的分支同构）
            abs_df = abs(delta_f_hz)
            log_arg = (2.0 * abs_df - rs_l) / (2.0 * abs_df + rs_l)
            log_term = (2.0 * abs_df - rs_l) * np.log(log_arg) if log_arg > 0 else 0.0
            log_plus = log_term + 2.0 * rs_l

            # 逐项的乘法次序与基线一致
            den1_00 = (a0[j] + a0[j]) * phi_il
            den1_01 = (a0[j] + a1[j]) * phi_il
            den1_11 = (a1[j] + a1[j]) * phi_il
            term1_00 = (num1 / den1_00) * (t0 + t0)
            term1_01 = (num1 / den1_01) * (t0 + t1)
            term1_11 = (num1 / den1_11) * (t1 + t1)

            den2_head = abs_phi_base * rs_l**2
            den2_00 = den2_head * a0[j] * a0[j]
            den2_01 = den2_head * a0[j] * a1[j]
            den2_11 = den2_head * a1[j] * a1[j]
            term2_00 = (num2 / den2_00) * log_plus
            term2_01 = (num2 / den2_01) * log_plus
            term2_11 = (num2 / den2_11) * log_plus

            # 基线的 sum_val += coeff * 2 * k[l_ch] * k[lp] * (term1 + term2)
            # 这里把 coeff 的幂次提出来：r^0 / r^1 / r^2。
            w0[i, j] = 2.0 * k0[j] * k0[j] * (term1_00 + term2_00)
            w1[i, j] = 2.0 * k0[j] * k1[j] * (term1_01 + term2_01)
            w2[i, j] = 2.0 * k1[j] * k1[j] * (term1_11 + term2_11)

    return a0, a1, k0, k1, spm0, spm1, spm2, gamma_ii_sq, gamma_sq, w0, w1, w2


@njit(cache=True)
def _compute_nli_variances_t(
    n_ch: int,
    n_span: int,
    symbol_rate_hz: NDArrayFloat,
    p_in_w: NDArrayFloat,
    p_out_exact_w: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    L_m: float,
    spm0: NDArrayFloat,
    spm1: NDArrayFloat,
    spm2: NDArrayFloat,
    gamma_ii_sq: NDArrayFloat,
    gamma_sq: NDArrayFloat,
    w0: NDArrayFloat,
    w1: NDArrayFloat,
    w2: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """FG-4b 变体：张量化的 NLI 方差（热路径无数据相关控制流）。

    ``t_tilde`` 仍是唯一的数据相关量，它只经由 ``T = 1 + t_tilde`` 与
    ``r = -t_tilde / T`` 进入；其余全部是构建期常量张量上的乘加与广播。
    热路径因此是纯张量运算、无 ``if``，这正是 FG-4c 搬到 GPU / 加 batch 维所需要
    的形状。

    ``p_in <= 0`` 的连接与对角线在**求和之后统一掩码**：掩码作用在已算出的
    (N, N) 张量上，因此即使某些惰性连接的 ``log`` / ``1/x`` 产生 ``nan`` / ``inf``，
    也不会污染求和——这与基线逐对 ``continue`` 跳过它们等价。
    """
    # --- 1. 反向拟合等效拉曼倾斜 t_tilde（唯一的数据相关步骤）---
    t_tilde = np.zeros(n_ch, dtype=np.float64)
    for i in range(n_ch):
        p_lin_out = p_in_w[i] * np.exp(-alpha_power_npm[i] * L_m)
        if p_lin_out > 0:
            g_raman = p_out_exact_w[i] / p_lin_out
            t_tilde[i] = (g_raman - 1.0) / (1.0 - np.exp(-alpha_power_npm[i] * L_m))

    big_t = 1.0 + t_tilde
    ratio = -t_tilde / big_t  # r_l
    ratio_sq = ratio * ratio

    # --- 2. SPM（逐通道，纯张量）---
    span_pow = float(n_span) ** _NLI_SPAN_EXPONENT
    spm_sum = span_pow * (spm0 + 2.0 * ratio * spm1 + ratio_sq * spm2)
    p_cube = p_in_w**3
    spm_pre = ((16.0 / 27.0) * (gamma_ii_sq * p_cube / (symbol_rate_hz**2))) * (big_t * big_t)
    sigma_spm_2 = np.where(p_in_w > 0, np.maximum(spm_pre * spm_sum, 0.0), 0.0)

    # --- 3. XPM（(N, N) 张量 + l 方向广播）---
    # sum_val[i, l] = W0 + 2*r_l*W1 + r_l^2*W2
    xpm_sum = w0 + 2.0 * (ratio * w1) + ratio_sq * w2
    # pre_factor[i, l]，乘法次序与基线一致：(gamma^2 * p_l^2) * p_i / rs_l
    #
    # 轴向必须写死：p_l 走最后一轴（axis=1），p_i 走第一轴（axis=0）。一维数组
    # 默认对齐**最后**一轴，所以 `p_l` 可直接广播，而 `p_i` 必须显式 p_in_w[:, None]，
    # 否则会静默变成 p_l——在满载（p_i == p_l）下这个错误完全不可见，只有像
    # path_cplusl_roadm 那样经过拉曼倾斜、逐通道功率不再相同时才会暴露。
    p_sq = p_in_w**2
    pre = ((gamma_sq * p_sq) * p_in_w[:, None]) / symbol_rate_hz
    pre = ((32.0 / 27.0) * pre) * (big_t * big_t)
    terms = np.maximum(pre * xpm_sum, 0.0)

    # 掩码：两端都通电、且不是自配对。基线在这两种情况下直接 continue。
    idx = np.arange(n_ch)
    active = (p_in_w[:, None] > 0) & (p_in_w[None, :] > 0) & (idx[:, None] != idx[None, :])
    sigma_xpm_2 = np.sum(np.where(active, terms, 0.0), axis=1)

    return sigma_spm_2, sigma_xpm_2


# -----------------------------------------------------------------------------
# NLI 张量缓存：按 (逐连接栅格, 跨段参数, 峰度) 记忆化
# -----------------------------------------------------------------------------
# 与 G 缓存同构：键是**值**键（``tobytes()``），因此两个数值相同但对象不同的
# 逐连接数组会命中同一条缓存，而不会误用别人的张量。
#
# ``n_span`` 必须进键：XPM 的 ``num1 = n_span + (5/6)*phi_kurtosis`` 与
# ``n_tilde`` 都被折进了 W0/W1/W2，因此换跨段数必须重建。
_NLI_CACHE: dict[tuple[Any, ...], tuple[Any, ...]] = {}
_NLI_CACHE_STATS = {"hits": 0, "misses": 0, "builds": 0}
_NLI_CACHE_MAX_ENTRIES = 8


def nli_cache_stats() -> dict[str, int]:
    """返回 NLI 张量缓存的命中/未命中计数——用于证明跨调用复用确实发生。"""
    return dict(_NLI_CACHE_STATS)


def _nli_cache_key(
    n_ch: int,
    n_span: int,
    symbol_rate_hz: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    L_m: float,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    phi_kurtosis: float,
) -> tuple[Any, ...]:
    """NLI 张量缓存键：五组逐连接栅格的值 + 全部跨段标量。"""
    return (
        int(n_ch),
        int(n_span),
        np.ascontiguousarray(symbol_rate_hz, dtype=np.float64).tobytes(),
        np.ascontiguousarray(f_rel_hz, dtype=np.float64).tobytes(),
        np.ascontiguousarray(lambdas_m, dtype=np.float64).tobytes(),
        np.ascontiguousarray(a_eff_m2, dtype=np.float64).tobytes(),
        np.ascontiguousarray(alpha_power_npm, dtype=np.float64).tobytes(),
        float(L_m),
        float(beta_2_s2_m),
        float(beta_3_s3_m),
        float(n_2_m2_w),
        float(phi_kurtosis),
    )


def nli_tensors(
    n_ch: int,
    n_span: int,
    symbol_rate_hz: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    L_m: float,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    phi_kurtosis: float,
) -> tuple[NDArrayFloat, ...]:
    """返回**缓存的** NLI 构建期张量，供 `_calc_span_noise_and_power_jit` 使用。

    顺序为 ``(spm0, spm1, spm2, gamma_ii_sq, gamma_sq, w0, w1, w2)``。返回值
    被标记为只读，以免调用方就地改坏共享张量。

    与 :func:`raman_g_matrix` 同理：条目过多时整体清空（键含完整逐连接栅格，
    清空只是重算）。
    """
    key = _nli_cache_key(
        n_ch,
        n_span,
        symbol_rate_hz,
        f_rel_hz,
        lambdas_m,
        a_eff_m2,
        alpha_power_npm,
        L_m,
        beta_2_s2_m,
        beta_3_s3_m,
        n_2_m2_w,
        phi_kurtosis,
    )
    cached = _NLI_CACHE.get(key)
    if cached is not None:
        _NLI_CACHE_STATS["hits"] += 1
        return cached
    _NLI_CACHE_STATS["misses"] += 1
    built = _build_nli_tensors(
        n_ch,
        n_span,
        np.ascontiguousarray(symbol_rate_hz, dtype=np.float64),
        np.ascontiguousarray(f_rel_hz, dtype=np.float64),
        np.ascontiguousarray(lambdas_m, dtype=np.float64),
        np.ascontiguousarray(a_eff_m2, dtype=np.float64),
        np.ascontiguousarray(alpha_power_npm, dtype=np.float64),
        L_m,
        beta_2_s2_m,
        beta_3_s3_m,
        n_2_m2_w,
        phi_kurtosis,
    )
    # 只取热路径需要的八个；a0/a1/k0/k1 已被折进 w*/spm*，不外传。
    tensors = (
        built[4],  # spm0
        built[5],  # spm1
        built[6],  # spm2
        built[7],  # gamma_ii_sq
        built[8],  # gamma_sq
        built[9],  # w0
        built[10],  # w1
        built[11],  # w2
    )
    for tensor in tensors:
        tensor.flags.writeable = False
    _NLI_CACHE_STATS["builds"] += 1
    if len(_NLI_CACHE) >= _NLI_CACHE_MAX_ENTRIES:
        _NLI_CACHE.clear()
    _NLI_CACHE[key] = tensors
    return tensors


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
    symbol_rate_hz: NDArrayFloat,  # 逐连接符号率 (Hz)，形状 (N,)
    n_span: int,  # 跨段数量
) -> NDArrayFloat:
    """计算所有跨段累积的 ASE 噪声方差。

    假设每个跨段后有一个 EDFA 进行放大和噪声引入。
    Args:
        p_in_w: 各信道入纤功率 (W)。
        p_out_w: 各信道出纤功率 (W)。
        f_abs_hz: 各信道绝对频率 (Hz)。
        nf_lin: 各信道线性噪声系数。
        symbol_rate_hz: 逐连接符号速率 (Hz)，长度须与 p_in_w 相同。
        n_span: 跨段数量。
    Returns:
        NDArrayFloat: 每个信道的 ASE 噪声方差 (W)。
    """
    total_ase_variance = np.zeros(len(p_in_w), dtype=np.float64)
    for i in range(len(p_in_w)):
        if p_in_w[i] > 0 and p_out_w[i] > 0:  # 只有有功率的信道才产生 ASE 噪声
            # 计算单跨段增益 (假设 EDFA 补偿损耗)
            G_i = p_in_w[i] / p_out_w[i]

            total_ase_variance[i] = (
                n_span * nf_lin[i] * H_PLANCK * f_abs_hz[i] * max((G_i - 1.0), 0.0) * symbol_rate_hz[i]
            )
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
    symbol_rate_hz: NDArrayFloat,  # 逐连接符号速率 (Hz)，形状 (N,)
    n_effective_spans: int,  # 有效累积跨段数 (NLI计算用)
    # --- 物理常数 ---
    cr_w_m_hz: float,  # SRS 增益系数 Cr
    delta_f_co_hz: float,  # SRS 截断频率
    phi_kurtosis: float = 0.0,  # 0.0为高斯(默认), -0.6190为64QAM
    # --- FG-4a: 可选的预计算拉曼矩阵（默认 None = 保持冻结的基线路径） ---
    g_full: NDArrayFloat | None = None,  # 缓存的完整 G (N,N)：变体一，逐位相等
    g_gain: NDArrayFloat | None = None,  # 缓存的严格上三角增益部分：变体二
    g_loss: NDArrayFloat | None = None,  # 缓存的严格下三角频率比加权损耗部分
    # --- FG-4b: 可选的预计算 NLI 张量（默认 None = 走冻结的逐对基线）---
    nli_spm0: NDArrayFloat | None = None,  # (N,)  SPM 的 (0,0) 项
    nli_spm1: NDArrayFloat | None = None,  # (N,)  SPM 的 (0,1)=(1,0) 项
    nli_spm2: NDArrayFloat | None = None,  # (N,)  SPM 的 (1,1) 项
    nli_gamma_ii_sq: NDArrayFloat | None = None,  # (N,)  自相位 Gamma_ii^2
    nli_gamma_sq: NDArrayFloat | None = None,  # (N,N) 逐对 Gamma_il^2
    nli_w0: NDArrayFloat | None = None,  # (N,N)  XPM 的 (0,0) 项
    nli_w1: NDArrayFloat | None = None,  # (N,N)  XPM 的 (0,1)=(1,0) 项
    nli_w2: NDArrayFloat | None = None,  # (N,N)  XPM 的 (1,1) 项
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """【高性能内核】单跨段物理层评估的 JIT 版本。

    封装了损耗、SRS、SPM、XPM 和 ASE 的全套计算。

    灵活栅格契约（设计 §2.5 / §3.4）：全部逐连接数组长度必须恰为 N
    (= len(power_in_w)) 且同序对齐。`f_abs_hz[i]` 是第 i 条**连接的中心频率**
    （不是栅格频率），`symbol_rate_hz[i]` 是该连接自己的符号率。
    未占用连接（power_in_w == 0）仍占一个数组槽位以保持对齐。

    FG-4a 加速（三个参数全部为 None 时行为与基线逐位相同）：

    * ``g_full`` given  -> `_calc_exact_power_profile_ode_with_g`，跳过 G 重建，
      内部仍用未改动的 `_calc_dp_dz`，因此**逐位相等**；
    * ``g_gain``/``g_loss`` given（且无 ``g_full``）->
      `_calc_exact_power_profile_ode_matvec`，去分支两次 matvec，求和次序改变；
    * 都不给 -> 原 `_calc_exact_power_profile_ode`（FG-2 冻结的参考实现）。

    FG-4b 加速（八个 ``nli_*`` 参数**全部**为 None 时，NLI 走冻结的逐对基线）：

    * 全部给 -> `_compute_nli_variances_t`，张量化 NLI。构建期的
      arcsinh / arctan / log 与分支选择已折进张量，热路径无数据相关控制流；
    * 都不给 -> 原 `_compute_nli_variances`（FG-2 冻结的参考实现）。

    Raises:
        ValueError: 任一逐连接数组长度 != N。
    """
    num_channels = len(power_in_w)

    # 形状守卫：njit 无法格式化动态消息，完整消息（含 (N,) 与实际形状）由
    # Python 层 MultiBandISRSGN._calc_span_noise_and_power 抛出。此处的静态
    # 消息是第二道防线——若缺失， njit 会对越界索引静默读越界内存。
    if (
        len(f_rel_hz) != num_channels
        or len(f_abs_hz) != num_channels
        or len(lambdas_m) != num_channels
        or len(alpha_power_npm) != num_channels
        or len(a_eff_m2) != num_channels
        or len(nf_lin_channels) != num_channels
        or len(symbol_rate_hz) != num_channels
    ):
        raise ValueError("per-connection arrays must all have length N = len(power_in_w).")

    # 1. 采用高精度 RK4 求解 ODE (计算包含了损耗和真实拉曼倾斜的出纤功率)
    # n_steps=20 在精度和速度间取得了极佳的平衡
    # FG-4a：调用侧可传入缓存的拉曼矩阵以跳过 G 的逐对重建。三个分支的数学
    # 结果相同，优先级为 g_full > (g_gain, g_loss) > 现场重建。
    if g_full is not None:
        # 变体一：G 已缓存，内部仍走原始 _calc_dp_dz -> 逐位相等。
        p_out_after_fiber = _calc_exact_power_profile_ode_with_g(
            power_in_w, g_full, f_abs_hz, alpha_power_npm, span_length_m, n_steps=20
        )
    elif g_gain is not None and g_loss is not None:
        # 变体二：去分支两次 matvec -> 求和次序改变，偏差为浮点舍入。
        p_out_after_fiber = _calc_exact_power_profile_ode_matvec(
            power_in_w, g_gain, g_loss, alpha_power_npm, span_length_m, n_steps=20
        )
    else:
        p_out_after_fiber = _calc_exact_power_profile_ode(
            power_in_w, f_abs_hz, alpha_power_npm, span_length_m, cr_w_m_hz, n_steps=20
        )

    # 2. 调用 NLI 引擎，此时它将利用上述精准结果来拟合有效的 T_tilde 边界
    # 2. 调用 NLI 引擎，此时它将利用上述精准结果来拟合有效的 T_tilde 边界
    # FG-4b：调用侧传入构建期缓存的 NLI 张量时走张量化变体；八个参数全为 None
    # 时逐字节保持 FG-2 冻结的参考路径。
    if nli_w0 is not None and nli_gamma_sq is not None and nli_spm0 is not None:
        sigma2_spm_w, sigma2_xpm_w = _compute_nli_variances_t(
            len(power_in_w),
            n_effective_spans,
            symbol_rate_hz,
            power_in_w,
            p_out_after_fiber,
            alpha_power_npm,
            span_length_m,
            nli_spm0,
            nli_spm1,
            nli_spm2,
            nli_gamma_ii_sq,
            nli_gamma_sq,
            nli_w0,
            nli_w1,
            nli_w2,
        )
    else:
        sigma2_spm_w, sigma2_xpm_w = _compute_nli_variances(
            len(power_in_w),
            n_effective_spans,
            symbol_rate_hz,
            power_in_w,
            p_out_after_fiber,  # 传参增加 p_out_after_fiber
            f_rel_hz,
            lambdas_m,
            a_eff_m2,
            alpha_power_npm,
            span_length_m,
            beta2_s2_m,
            beta3_s3_m,
            n2_m2_w,
            phi_kurtosis,
        )

    # 5. 计算 ASE 噪声方差
    sigma2_ase_w = _calculate_ase_noise_variance(
        power_in_w, p_out_after_fiber, f_abs_hz, nf_lin_channels, symbol_rate_hz, n_effective_spans
    )

    return p_out_after_fiber, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w
