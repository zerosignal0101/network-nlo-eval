r"""FG-4e CPU 批量路径：单跨段内核的 (B, N) 批量形态。

这是什么
--------
把 FG-4c ``scripts/fg4c_batch_knee.py`` 里那个**脚本级原型**推进为**库内批量入口**。
它与 :func:`network_nlo_eval.models.numba_kernels._calc_span_noise_and_power_jit`
算的是同一件事：RK4/SRS ODE + 张量化 NLI + ASE。区别只在数据布局——

    串行：  p (N,)      每候选一次调用，逐对 O(N^2) 循环
    批量：  P (B, N)    一次调用，整块 (B,N) @ (N,N) GEMM

**默认路径不变**：本模块**不替换**任何既有路径。``ode_accel`` 的
``"off"`` / ``"cached_g"`` / ``"matvec"``（默认）三个档位逐位不动，批量是
:func:`~network_nlo_eval.models.isrs_gn.MultiBandISRSGN.calc_span_noise_and_power_batch`
这一个**新入口**（FG-4e brief §1.2.2）。

为什么是三次 GEMM，而不是 (B, N, N) 张量
---------------------------------------
朴素批量化会物化 ``(B,N,N)``：N=230、B=256 时是 129 MB，N=2100、B=1024 时是
36 GB——不可行。但对扰动轴 ``l`` 的求和在把数据相关比值提出来之后是**线性**的::

    sigma_xpm[b,i] = (32/27) * P[b,i] * sum_l ( gamma_sq[i,l] * u0[b,l]
                                                 + 2 * gamma_sq[i,l] * u1[b,l]
                                                 +     gamma_sq[i,l] * u2[b,l] )

    u0[b,l] = P[b,l]^2 * T[b,l]^2 / rs[l]        （T = 1 + t_tilde）
    u1[b,l] = u0[b,l] * r[b,l]                   （r = -t_tilde / T）
    u2[b,l] = u0[b,l] * r[b,l]^2

把 ``l`` 收进矩阵侧即得三次 ``(N,N)`` 矩阵积。RK4 的 80 次 ``dp`` 则是 80 次
``(B,N) @ (N,N)``（每子步两次：增益 + 损耗），热路径**无数据相关控制流**。

``g_gain`` / ``g_loss`` / ``gamma_sq * w_k`` 这些 (N,N) 矩阵**不按 batch 复制**，
一次构建、整批复用：这是批量路径省下来的主要常数因子之一。

广播纪律（本项目已栽过一次的地方）
--------------------------------
FG-4b 的 XPM pre-factor 里 ``big_t * big_t`` 必须留在**扰动轴 l** 上，不能提到
``i`` 轴外面——串行张量化核 ``pre = ((32/27)*pre) * (big_t*big_t)`` 是 (N,N) 乘
(N,)，默认广播到最后一维即 ``l``。提出来是**静默 2x 误差**。本模块因此把
``T**2`` 折在 ``u0[b,l]`` 上（:func:`build_batch_constants` 的注释与断言处），
而不是折在 ``P[:, i]`` 上。

同理，``p_i`` 必须写死第一轴：批量维下 ``P`` 是 (B,N)，``P * P`` 是逐元素的
(per-candidate, per-channel)，两个轴**都**带 b，所以这里不存在 FG-4b 那个
"一维数组隐式广播到最后一轴"的陷阱——但 ``alpha`` / ``rs`` / ``f_abs`` /
``nf_lin`` 这些 (N,) 构建期常量在 (B,N) 上广播到最后一维 l/COI，方向必须与串行
一致，这也是 :func:`batched_span_noise_and_power` 里逐处显式写死轴的原因。

数值等价性
----------
逐候选与串行路径的相对偏差预算 **1e-12**（沿用 FG-4a/FG-4b 的常量，**未放宽**）。
重排来源有且只有浮点求和次序：l 方向由成对求和变成 GEMM 累加。逐候选实测见
``results/fg4e_speedup_ab.json`` 的 ``numeric_check``，回归由
``tests/test_fg4e_cpu_batching.py`` 把关。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.numba_kernels import H_PLANCK, _NLI_SPAN_EXPONENT

#: RK4 子步数。与 ``_calc_span_noise_and_power_jit`` 的 ``n_steps=20`` 是同一个值，
#: 少一位就会让批量与串行的偏差离开 1e-12 预算——它是数值契约的一部分。
BATCH_N_STEPS = 20

#: FG-4a/FG-4b 的相对偏差预算，原样沿用，不放宽。
MAX_REL_DEVIATION = 1e-12


@dataclass(frozen=True)
class BatchSpanConstants:
    """一次构建、整批复用的 (N,)/(N,N) 构建期常量。

    全部字段形状为 ``(N,)`` 或 ``(N,N)``——**没有 batch 轴**。批量路径的价值一半
    就在这里：B 从 1 涨到 256，这些张量的构建成本一次都不付第二次。

    Attributes:
        g_gain_t / g_loss_t: ``g.T`` 形状 (N,N)。热路径要 ``g @ p[b]``，对
            (B,N) 而言等价于 ``P @ g.T``，预先转置让热循环里没有临时转置。
        alpha: (N,) 逐通道衰减，广播到最后一轴（与串行 ``- alpha_npm[i]`` 同轴）。
        spm0/spm1/spm2: (N,) SPM 三项（**不含**跨段指数因子，它在运行期乘）。
        gamma_ii_sq: (N,) 自相位 Gamma^2。
        m0/m1/m2: (N,N) 且**已转置**——``m0[l,i] = gamma_sq[i,l] * w0[i,l]``，
            且对角线置零（串行基线 ``continue`` 跳过自配对 i==l）。热循环因此是
            纯 ``(B,N) @ (N,N)``。
        rs: (N,) 逐连接符号率（COI 自己的带宽）。
        f_abs / nf_lin: (N,) 绝对频率与线性噪声系数。
        lin_decay / one_minus_decay: (N,) ``exp(-alpha*L)`` 与其补，用于拟合
            ``t_tilde``。逐元素，所以 per-connection 的 alpha 语义保持不变。
    """

    g_gain_t: NDArrayFloat
    g_loss_t: NDArrayFloat
    alpha: NDArrayFloat
    spm0: NDArrayFloat
    spm1: NDArrayFloat
    spm2: NDArrayFloat
    gamma_ii_sq: NDArrayFloat
    m0: NDArrayFloat
    m1: NDArrayFloat
    m2: NDArrayFloat
    rs: NDArrayFloat
    f_abs: NDArrayFloat
    nf_lin: NDArrayFloat
    lin_decay: NDArrayFloat
    one_minus_decay: NDArrayFloat


def build_batch_constants(
    g_gain: NDArrayFloat,
    g_loss: NDArrayFloat,
    alpha: NDArrayFloat,
    spm0: NDArrayFloat,
    spm1: NDArrayFloat,
    spm2: NDArrayFloat,
    gamma_ii_sq: NDArrayFloat,
    gamma_sq: NDArrayFloat,
    w0: NDArrayFloat,
    w1: NDArrayFloat,
    w2: NDArrayFloat,
    rs: NDArrayFloat,
    f_abs: NDArrayFloat,
    nf_lin: NDArrayFloat,
    span_length_m: float,
) -> BatchSpanConstants:
    """把 FG-4b 的 NLI 张量与 FG-4a 的拉曼矩阵折成批量形态的常量包。

    与 ``Scenario.__init__``（FG-4c 原型）做的整理逐项同构，但搬进库、并加了
    形状断言。**不复制到 batch 维**：``g_*`` 与 ``m*`` 都是 (N,N)。

    Args:
        g_gain/g_loss: ``raman_g_pair_matrices`` 的两个 (N,N) 分量。
        alpha: (N,) 逐通道衰减 (Np/m)。
        spm0/spm1/spm2: (N,) SPM 三项，来自 ``nli_tensors``。
        gamma_ii_sq: (N,) 自相位 Gamma^2。
        gamma_sq: (N,N) 逐对 Gamma^2。
        w0/w1/w2: (N,N) XPM 的 (0,0)/(0,1)=(1,0)/(1,1) 合并项。
        rs: (N,) 逐连接符号率。
        f_abs: (N,) 绝对频率。
        nf_lin: (N,) 线性噪声系数。
        span_length_m: 跨段长度 (m)，只用于拟合 ``t_tilde`` 的两个衰减因子。

    Returns:
        BatchSpanConstants: 可整批复用的构建期常量。

    Raises:
        ValueError: 任一 (N,) 数组与 N 不符，或 (N,N) 数组形状不是方阵。
    """
    n_ch = int(g_gain.shape[0])
    vecs = {
        "alpha": alpha, "spm0": spm0, "spm1": spm1, "spm2": spm2,
        "gamma_ii_sq": gamma_ii_sq, "rs": rs, "f_abs": f_abs, "nf_lin": nf_lin,
    }
    for name, arr in vecs.items():
        if arr.shape != (n_ch,):
            raise ValueError(f"{name} must have shape ({n_ch},), got {arr.shape}.")
    mats = {
        "g_gain": g_gain, "g_loss": g_loss, "gamma_sq": gamma_sq,
        "w0": w0, "w1": w1, "w2": w2,
    }
    for name, arr in mats.items():
        if arr.shape != (n_ch, n_ch):
            raise ValueError(f"{name} must have shape ({n_ch}, {n_ch}), got {arr.shape}.")

    # ``gamma_sq * w_k`` 先按 (i,l) 相乘（与串行 pre_factor 的逐项乘法次序一致），
    # 再转置成 ``m[l,i]``，最后对角线置零。转置只是为了让热循环写成
    # ``u @ m``（(B,N)@(N,N)）而不必每步取转置视图。
    eye = np.eye(n_ch, dtype=np.float64)
    offdiag = 1.0 - eye  # i == l 置零，等价于串行基线的 ``continue``

    def folded(gamma_sq_k: NDArrayFloat, w_k: NDArrayFloat) -> NDArrayFloat:
        return np.ascontiguousarray((gamma_sq_k * w_k * offdiag).T)

    alpha_f = np.asarray(alpha, dtype=np.float64)
    lin_decay = np.exp(-alpha_f * float(span_length_m))

    return BatchSpanConstants(
        g_gain_t=np.ascontiguousarray(np.asarray(g_gain, dtype=np.float64).T),
        g_loss_t=np.ascontiguousarray(np.asarray(g_loss, dtype=np.float64).T),
        alpha=np.ascontiguousarray(alpha_f),
        spm0=np.ascontiguousarray(np.asarray(spm0, dtype=np.float64)),
        spm1=np.ascontiguousarray(np.asarray(spm1, dtype=np.float64)),
        spm2=np.ascontiguousarray(np.asarray(spm2, dtype=np.float64)),
        gamma_ii_sq=np.ascontiguousarray(np.asarray(gamma_ii_sq, dtype=np.float64)),
        m0=folded(np.asarray(gamma_sq, dtype=np.float64), np.asarray(w0, dtype=np.float64)),
        m1=folded(np.asarray(gamma_sq, dtype=np.float64), np.asarray(w1, dtype=np.float64)),
        m2=folded(np.asarray(gamma_sq, dtype=np.float64), np.asarray(w2, dtype=np.float64)),
        rs=np.ascontiguousarray(np.asarray(rs, dtype=np.float64)),
        f_abs=np.ascontiguousarray(np.asarray(f_abs, dtype=np.float64)),
        nf_lin=np.ascontiguousarray(np.asarray(nf_lin, dtype=np.float64)),
        lin_decay=np.ascontiguousarray(lin_decay),
        # 串行 ``(g_raman - 1) / (1 - exp(-alpha*L))``：分母逐通道，逐连接的
        # alpha 语义原样保留，没有退化成标量。
        one_minus_decay=np.ascontiguousarray(1.0 - lin_decay),
    )


def _rk4_batched(
    P: NDArrayFloat, c: BatchSpanConstants, span_length_m: float, n_steps: int = BATCH_N_STEPS
) -> NDArrayFloat:
    """RK4/SRS 求解，批量形态。逐子步两次 ``(B,N) @ (N,N)``。

    与串行 ``_calc_exact_power_profile_ode_matvec`` 逐步一致：同样的 20 步、同样
    的 4 次子调用、同样的组合系数、同样的**每步之后**负功率截断。
    """
    p = np.array(P, dtype=np.float64, copy=True)
    dz = span_length_m / n_steps
    half = 0.5 * dz
    six = dz / 6.0

    def dp(x: NDArrayFloat) -> NDArrayFloat:
        # alpha / g_* 都是 (N,) 或 (N,N)，广播到最后一轴（通道 i），
        # 与串行 ``dp[i] = (gain[i] - loss[i] - alpha[i]) * p[i]`` 同轴。
        return (x @ c.g_gain_t - x @ c.g_loss_t - c.alpha) * x

    for _ in range(n_steps):
        k1 = dp(p)
        k2 = dp(p + half * k1)
        k3 = dp(p + half * k2)
        k4 = dp(p + dz * k3)
        p += six * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        # 串行是在每步之后逐元素截断负功率；批量同样放在步后，不放在子步之间。
        np.maximum(p, 0.0, out=p)
    return p


def _nli_batched(
    P: NDArrayFloat,
    p_out: NDArrayFloat,
    c: BatchSpanConstants,
    span_length_m: float,
    n_effective_spans: int,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """张量化 NLI 的批量形态：SPM 逐通道、XPM 三次 GEMM。

    与串行 ``_compute_nli_variances_t`` 的对应关系（这是等价性的全部依据）：

    * ``t_tilde`` 是唯一的数据相关步骤，批量下逐元素拟合，掩码 ``p_lin_out > 0``；
    * SPM 的 ``span_pow`` 与串行一样在**运行期**乘，不折进张量；
    * XPM 的 ``T**2`` 折在 ``u0`` 上（l 轴），与串行 (N,N)*(N,) 的默认广播同向。

    ``span_length_m`` 只经由构建期常量参与（``lin_decay`` 已在常量包里），
    这里保留形参是为了让调用点的签名与串行核一一对应、可读。
    """
    del span_length_m  # 已折进 c.lin_decay / c.one_minus_decay

    # --- 1. 反向拟合等效拉曼倾斜 t_tilde ---
    p_lin_out = P * c.lin_decay
    t_tilde = np.zeros_like(P)
    np.divide(p_out, p_lin_out, out=t_tilde, where=p_lin_out > 0)
    # (t - 1) / (1 - decay)：除法与乘法次序与串行一致，惰性通道留 0。
    np.subtract(t_tilde, 1.0, out=t_tilde, where=p_lin_out > 0)
    np.divide(t_tilde, c.one_minus_decay, out=t_tilde, where=p_lin_out > 0)

    big_t = 1.0 + t_tilde
    ratio = -t_tilde / big_t
    ratio_sq = ratio * ratio

    # --- 2. SPM（逐通道，纯 (B,N) 元素运算）---
    span_pow = float(n_effective_spans) ** _NLI_SPAN_EXPONENT
    spm_sum = span_pow * (c.spm0 + 2.0 * ratio * c.spm1 + ratio_sq * c.spm2)
    p_sq = P * P
    p_cube = p_sq * P
    spm_pre = ((16.0 / 27.0) * (c.gamma_ii_sq * p_cube / (c.rs * c.rs))) * (big_t * big_t)
    sigma_spm = np.where(P > 0, np.maximum(spm_pre * spm_sum, 0.0), 0.0)

    # --- 3. XPM（l 方向收进矩阵侧 → 三次 GEMM）---
    #
    # T**2 乘在 u0 上，即留在**扰动轴 l**：串行是 (N,N) * (N,) 默认广播到最后一维，
    # 提到 i 轴外是静默 2x 误差（FG-4b 已踩过同类的 p_i 坑）。
    u0 = p_sq * (big_t * big_t) / c.rs
    u1 = u0 * ratio
    u2 = u0 * ratio_sq
    acc = u0 @ c.m0 + 2.0 * (u1 @ c.m1) + u2 @ c.m2  # (B,N)
    # 串行 pre = ((gamma_sq * p_sq) * p_in[:,None]) / rs * (32/27) * T**2，
    # 即 (32/27) * p_i * sum_l (...)：p_i 在这里以 P (B,N) 的形式带出去。
    #
    # 掩码的等价性：p_l == 0 由 p_sq[b,l] == 0 → u* == 0 覆盖；
    # i == l 由 m* 对角线置零覆盖；p_i == 0 由末行 np.where(P > 0, ...) 覆盖。
    sigma_xpm = np.where(P > 0, np.maximum((32.0 / 27.0) * P * acc, 0.0), 0.0)

    return sigma_spm, sigma_xpm


def batched_span_noise_and_power(
    constants: BatchSpanConstants,
    power_in_w: NDArrayFloat,
    span_length_m: float,
    n_effective_spans: int = 1,
    nf_lin_channels: NDArrayFloat | None = None,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """单跨段内核的批量入口：``(B, N)`` 入、四个 ``(B, N)`` 出。

    与串行 ``_calc_span_noise_and_power_jit`` 的返回值逐项对应，只是多一个 b 轴。

    Args:
        constants: :func:`build_batch_constants` 产出的构建期常量（整批复用）。
        power_in_w: ``(B, N)`` 候选功率 (W)。**零功率是合法的**，掩码语义与串行
            相同（惰性连接仍占一个槽位）。
        span_length_m: 跨段长度 (m)。
        n_effective_spans: 有效跨段数，进 SPM 的跨段指数因子与 ASE 的 n_span。
        nf_lin_channels: ``(N,)`` 逐通道噪声系数覆盖（EDFA 给全频段 NF 时用），
            形状须与常量包一致。为 None 时用常量包里的逐通道默认值。

    Returns:
        ``(p_out, sigma2_spm, sigma2_xpm, sigma2_ase)``，四个都是 ``(B, N)``。

    Raises:
        ValueError: ``power_in_w`` 不是 ``(B, N)``、N 与常量不符、或 NF 覆盖形状不对。
    """
    P = np.asarray(power_in_w, dtype=np.float64)
    if P.ndim != 2:
        raise ValueError(f"power_in_w must have shape (B, N), got {P.shape}.")
    n_ch = int(constants.alpha.shape[0])
    if P.shape[1] != n_ch:
        raise ValueError(f"power_in_w must have shape (B, {n_ch}), got {P.shape}.")
    if P.shape[0] == 0:
        empty = np.empty((0, n_ch), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    if nf_lin_channels is None:
        nf_lin = constants.nf_lin
    else:
        nf_lin = np.asarray(nf_lin_channels, dtype=np.float64)
        if nf_lin.shape != (n_ch,):
            raise ValueError(f"nf_lin_channels must have shape ({n_ch},), got {nf_lin.shape}.")

    p_out = _rk4_batched(P, constants, float(span_length_m))
    sigma2_spm, sigma2_xpm = _nli_batched(P, p_out, constants, float(span_length_m), int(n_effective_spans))

    # --- ASE：与串行 ``_calculate_ase_noise_variance`` 同式同掩码 ---
    # G_i = p_in / p_out，只在两端都通电时有定义；np.where 先算安全分母，
    # 不引入 nan/inf（惰性通道的 P/p_out 在被 where 丢弃之前不会外泄）。
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(p_out > 0, P / np.where(p_out > 0, p_out, 1.0), 0.0)
    n_span = float(n_effective_spans)
    sigma2_ase = np.where(
        (P > 0) & (p_out > 0),
        n_span * nf_lin * H_PLANCK * constants.f_abs * np.maximum(gain - 1.0, 0.0) * constants.rs,
        0.0,
    )

    return p_out, sigma2_spm, sigma2_xpm, sigma2_ase


def worst_relative_deviation(expected: NDArrayFloat, actual: NDArrayFloat) -> float:
    """逐元素相对偏差的最大值（FG-4a/FG-4b 的 1e-12 预算用它把关）。

    分母为 0 的元素（惰性通道的噪声恒为 0）不参与——它们的偏差按定义是 0/0。
    任一侧非有限则该元素不参与，调用方需另行确认没有非有限值漏出来。
    """
    exp_arr = np.asarray(expected, dtype=np.float64).ravel()
    act_arr = np.asarray(actual, dtype=np.float64).ravel()
    finite = np.isfinite(exp_arr) & np.isfinite(act_arr)
    if not finite.any():
        return 0.0
    diff = np.abs(exp_arr[finite] - act_arr[finite])
    denom = np.abs(exp_arr[finite])
    return float(np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0).max())
