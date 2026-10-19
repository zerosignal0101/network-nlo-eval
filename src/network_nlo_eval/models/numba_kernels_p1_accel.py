"""P1 闭式核的加速结构：``matvec`` 档与批量入口。

``numba_kernels`` 里的 P1 核对每一对信道都现算 ``gamma_il`` / ``phi_il`` /
两个 ``arctan``。其中只有 ``phi_il`` 与 ``B_i`` 有关的那部分随候选变，
``G1`` / ``G2`` 两张矩阵**只依赖信道几何**（波长、频率、带宽、衰减、色散、n2），
是 evaluator 级常量。本模块把它们提到构建期算一次，运行期把 ``ell`` 方向的求和
降成矩阵-向量乘。

为什么 XPM 不能整体一次 matvec
-------------------------------
P1 的 XPM 求和项是::

    term(i,ell) = (P_l^2 / B_l) * exp(2*Cr*k_bar*L_eff_l*r_fl)
                * (gamma_il^2 / (phi_il * 3 * alpha_l^2))
                * [ c1_l * atan(phi_il * B_i / alpha_l)
                  + c2_l * atan(phi_il * B_i / (2*alpha_l)) ]

``atan`` 的自变量同时含信道对量 ``phi_il`` 与逐信道量 ``B_i``、``alpha_l``，
而 ``atan`` 非线性 ⇒ 这一项不能写成 ``f(i) * g(ell)``。故走分段路线：

1. 与信道对相关的部分预计算成 (N,N) 矩阵 ``G1`` / ``G2``；
2. ``ell`` 方向求和变成两次 matvec；
3. 逐对循环只保留一处：原实现对每一对都做的 ``max(·, 0)`` 截断。

截断看似挡路，其实绝大多数通道上恒等。把求和项摊开成
``term(i,ell) = G1[i,ell]*w1[ell] + G2[i,ell]*w2[ell]`` 后，``G1`` / ``G2`` 逐元素 >= 0
（``gamma_il^2/(phi_il*3*alpha_l^2)`` 与两个 ``atan`` 的符号都等于 ``sign(phi_il)``），
且 ``G1 >= G2``、``G1 <= 2*G2``（``atan`` 单调且在 ``[0, inf)`` 上凹）。由此得到
逐通道 O(1) 的充分判据，见 :func:`_p1_xpm_weights_jit`。不满足的通道改用冻结核
的逐对函数算回来，与 ``off`` 档逐位相同，不是另写一份实现。

精度口径
--------
只重排 ``ell`` 方向的浮点求和次序，相对偏差预算 1e-12。出纤功率与 SPM 走冻结核的
原函数，逐位相同；只有 XPM 的求和次序变了。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from network_nlo_eval.core.constants import H_PLANCK, PI
from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat
from network_nlo_eval.models.numba_kernels import (
    _EPSILON,
    _SRS_CR,
    _calc_gamma_coeff_scalar,
    _calc_raman_profile_jit,
    _calc_spm_variance_jit,
    _calc_xpm_variance_term_jit,
)

_F = TypeVar("_F", bound=Callable[..., Any])

try:
    from numba import njit
except ImportError:  # 无 JIT 的最小环境仍可运行统计模型。

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:  # type: ignore[misc, no-redef]
        """Return an identity decorator when optional Numba is unavailable."""
        del args, kwargs

        def decorate(function: _F) -> _F:
            return function

        return decorate


#: ``matvec`` 档与 ``off`` 档的相对偏差预算。
MAX_REL_DEVIATION = 1e-12


@dataclass(frozen=True)
class P1XpmMatrices:
    """P1 XPM 的 evaluator 级 (N,N) 常量包，跨跨段、跨候选复用。

    Attributes
    ----------
        g1: ``(gamma_il^2 / (phi_il*3*alpha_l^2)) * atan(phi_il*B_i/alpha_l)``，逐元素 >= 0。
        g2: 同上，第二个 ``atan`` 的分母为 ``2*alpha_l``，逐元素 >= 0。
        g1_t / g2_t: 两张矩阵的连续转置，热路径写成 ``w @ g_t``，免得每步取视图。
        phi_raw: 未夹 ``_EPSILON`` 的 ``phi_il``。矩阵化用夹过的值，逐对回退要交给
            冻结核自己夹，才与 ``off`` 逐位相同。
        gamma: ``gamma_il``，回退路径要用。
    """

    g1: NDArrayFloat
    g2: NDArrayFloat
    g1_t: NDArrayFloat
    g2_t: NDArrayFloat
    phi_raw: NDArrayFloat
    gamma: NDArrayFloat

    @property
    def n_ch(self) -> int:
        """信道数 N。"""
        return int(self.g1.shape[0])


@njit(cache=True)
def build_p1_xpm_pair_matrices(
    n_ch: int,
    b_hz_channels: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """构建 P1 XPM 的 (N,N) 对矩阵。

    算式与冻结核 :func:`network_nlo_eval.models.numba_kernels._compute_nli_variances`
    的逐对分支一致：同样的 ``phi_il``、同样的 ``_EPSILON`` 夹取、同样的
    ``gamma_il``、同样的两个 ``arctan``，只是从"每对现算"提前到构建期。
    对角线（``i == ell``）置零，与串行基线跳过自配对一致。

    Args:
        n_ch: 信道数 N。
        b_hz_channels: 逐信道带宽 ``B_i`` (Hz)。
        f_rel_hz: 逐信道相对频率 (Hz)。
        lambdas_m: 逐信道波长 (m)。
        a_eff_m2: 逐信道有效模面积 (m^2)。
        alpha_power_npm: 逐信道功率衰减系数 (Np/m)。
        beta_2_s2_m: 色散 beta2 (s^2/m)。
        beta_3_s3_m: 色散 beta3 (s^3/m)。
        n_2_m2_w: 非线性折射率 n2 (m^2/W)。

    Returns
    -------
        ``(g1, g2, phi_raw, gamma)``，四张都是 ``(N, N)``。
    """
    g1 = np.zeros((n_ch, n_ch), dtype=np.float64)
    g2 = np.zeros((n_ch, n_ch), dtype=np.float64)
    phi_raw = np.zeros((n_ch, n_ch), dtype=np.float64)
    gamma_mat = np.zeros((n_ch, n_ch), dtype=np.float64)

    for i in range(n_ch):
        for ell in range(n_ch):
            if ell == i:  # 串行基线跳过自配对
                continue

            gamma_il = _calc_gamma_coeff_scalar(lambdas_m[i], lambdas_m[ell], n_2_m2_w, a_eff_m2[i], a_eff_m2[ell])
            phi_il = (
                2
                * PI**2
                * (f_rel_hz[ell] - f_rel_hz[i])
                * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[ell] + f_rel_hz[i]))
            )
            phi_raw[i, ell] = phi_il
            gamma_mat[i, ell] = gamma_il

            phi_c = phi_il
            if abs(phi_c) < _EPSILON:
                phi_c = _EPSILON if phi_c >= 0 else -_EPSILON

            a_il = gamma_il**2 / (phi_c * 3 * alpha_power_npm[ell] ** 2)
            g1[i, ell] = a_il * np.arctan((phi_c * b_hz_channels[i]) / alpha_power_npm[ell])
            g2[i, ell] = a_il * np.arctan((phi_c * b_hz_channels[i]) / (2 * alpha_power_npm[ell]))

    return g1, g2, phi_raw, gamma_mat


@njit(cache=True)
def _p1_span_pre_nli_batch_jit(
    p_in: NDArrayFloat,
    span_length_m: float,
    f_rel_hz: NDArrayFloat,
    symbol_rate_hz: float,
    alpha_power_npm: NDArrayFloat,
    cr_w_m_hz: float,
    k_bar: float,
    delta_f_co_hz: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """NLI 之前的跨段物理：拉曼剖面、出纤功率、t 因子、有效长度。

    与冻结核 :func:`network_nlo_eval.models.numba_kernels._calc_span_noise_and_power_jit`
    的前四步逐字相同（同样的调用、次序与表达式），故出纤功率与 ``t_factors``
    逐位相同。

    出纤功率的公共子表达式消除：冻结核对每个 ``i`` 都算一遍 ``exp(u_i * r_f)``，
    其中 ``u_i = -(1-k_bar)*_SRS_CR*L_eff[i]``，而 ``L_eff[i]`` 只依赖逐信道衰减。
    默认衰减口径（常数）下全部 ``u_i`` 逐位相等，那 N 条指数向量是同一条，指数
    计算量从 O(N^2) 降到 O(N)。这是恒等变换，不是改算式。``u`` 互不相等时
    （波长多项式衰减）退回逐 ``i`` 计算，代价与冻结核相同。

    ``f_m_hz`` / ``f_M_hz`` 用核自己的口径（``f_rel_hz[0] - Rs/2``），不是 evaluator
    按连接带边算的 ``self.f_m_hz``；冻结核用的是前者，这里必须一致。

    Args:
        p_in: ``(B, N)`` 入纤功率 (W)。
        span_length_m: 跨段长度 (m)。
        f_rel_hz: 逐信道相对频率 (Hz)。
        symbol_rate_hz: 标量符号率 (Hz)，只用于 SRS 频带边界。
        alpha_power_npm: ``(N,)`` 逐信道功率衰减系数 (Np/m)。
        cr_w_m_hz: SRS 增益系数 Cr。
        k_bar: 拉曼泵浦因子。
        delta_f_co_hz: SRS 截断频率 (Hz)。

    Returns
    -------
        ``(p_out, r_f, t_factors, l_eff)``：前三个 ``(B, N)``，最后一个 ``(N,)``。
    """
    n_batch = p_in.shape[0]
    n_ch = p_in.shape[1]
    p_out = np.zeros((n_batch, n_ch), dtype=np.float64)
    r_f = np.zeros((n_batch, n_ch), dtype=np.float64)
    t_factors = np.zeros((n_batch, n_ch), dtype=np.float64)

    l_eff_m = (1.0 - np.exp(-alpha_power_npm * span_length_m)) / alpha_power_npm

    u = np.empty(n_ch, dtype=np.float64)
    for i in range(n_ch):
        u[i] = (-(1 - k_bar)) * _SRS_CR * l_eff_m[i]

    u_all_same = True
    for i in range(1, n_ch):
        if u[i] != u[0]:
            u_all_same = False
            break

    f_m_hz = f_rel_hz[0] - symbol_rate_hz / 2.0
    f_M_hz = f_rel_hz[-1] + symbol_rate_hz / 2.0

    for b in range(n_batch):
        pb = p_in[b]
        total_p_in_w = np.sum(pb)
        r_f_values = _calc_raman_profile_jit(f_rel_hz, f_m_hz, f_M_hz, delta_f_co_hz, total_p_in_w)
        r_f[b, :] = r_f_values

        if u_all_same:
            # 默认口径：N 条指数向量其实是同一条。
            exp_row = np.exp(u[0] * r_f_values)
            denom_all = np.sum(pb * exp_row)
            for i in range(n_ch):
                if pb[i] > 0 and denom_all > 0:
                    p_out[b, i] = (
                        pb[i]
                        * np.exp(u[i] * r_f_values[i] - alpha_power_npm[i] * span_length_m)
                        * total_p_in_w
                        / denom_all
                    )
        else:
            for i in range(n_ch):
                if pb[i] > 0:
                    exp_row = np.exp(u[i] * r_f_values)
                    denom_i = np.sum(pb * exp_row)
                    if denom_i > 0:
                        p_out[b, i] = (
                            pb[i]
                            * np.exp(u[i] * r_f_values[i] - alpha_power_npm[i] * span_length_m)
                            * total_p_in_w
                            / denom_i
                        )

        t_factors[b, :] = (2.0 * alpha_power_npm - cr_w_m_hz * r_f_values) ** 2

    return p_out, r_f, t_factors, l_eff_m


@njit(cache=True)
def _p1_xpm_weights_jit(
    n_ch: int,
    n_batch: int,
    p_in: NDArrayFloat,
    b_hz_channels: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    t_factors: NDArrayFloat,
    r_f: NDArrayFloat,
    l_eff_m: NDArrayFloat,
    cr_w_m_hz: float,
    k_bar: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
    """XPM 的逐候选权重 ``(w1, w2)`` 与逐对安全掩码 ``safe``。

    ``w1[ell] = (P_l^2/B_l) * exp(2*Cr*k_bar*L_eff_l*r_fl) * c1_l``，
    ``w2[ell]`` 同式乘 ``c2_l``，其中 ``c1_l = (t_l - alpha_l^2)/alpha_l``、
    ``c2_l = (4*alpha_l^2 - t_l)/(2*alpha_l)``。

    安全判据回答的是"原实现的逐对 ``max(·, 0)`` 会不会真的生效"。摊开成
    ``result(i,ell) = v_l * A_il * Q_il``（``Q_il = c1_l*A1_il + c2_l*A2_il``）后，
    因 ``A1 >= A2``、``A1 <= 2*A2``，逐通道有充分判据：

    * ``c1 >= 0`` 且 ``c2 >= 0`` ⇒ 恒安全；
    * ``c1 >= 0``、``c2 < 0`` ⇒ ``c1 + c2 >= 0`` 时恒安全；
    * ``c1 < 0``、``c2 >= 0`` ⇒ ``2*c1 + c2 >= 0`` 时恒安全；
    * 两者都负 ⇒ 截断一定生效，该通道走逐对回退。

    Args:
        n_ch: 信道数 N。
        n_batch: 批宽 B。
        p_in: ``(B, N)`` 入纤功率 (W)。
        b_hz_channels: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
        alpha_power_npm: ``(N,)`` 逐信道衰减 (Np/m)。
        t_factors: ``(B, N)`` 逐候选的 ``t_l``。
        r_f: ``(B, N)`` 逐候选的拉曼剖面。
        l_eff_m: ``(N,)`` 逐信道有效长度 (m)。
        cr_w_m_hz: SRS 增益系数 Cr。
        k_bar: 拉曼泵浦因子。

    Returns
    -------
        ``(w1, w2, safe)``，前两个 ``(B, N)``，第三个 ``(B, N)`` 布尔。
    """
    w1 = np.zeros((n_batch, n_ch), dtype=np.float64)
    w2 = np.zeros((n_batch, n_ch), dtype=np.float64)
    safe = np.ones((n_batch, n_ch), dtype=np.bool_)

    for b in range(n_batch):
        for ell in range(n_ch):
            term_exp = np.exp(2 * cr_w_m_hz * k_bar * l_eff_m[ell] * r_f[b, ell])
            v = (p_in[b, ell] ** 2 / b_hz_channels[ell]) * term_exp
            c1 = (t_factors[b, ell] - alpha_power_npm[ell] ** 2) / alpha_power_npm[ell]
            c2 = (4 * alpha_power_npm[ell] ** 2 - t_factors[b, ell]) / (2 * alpha_power_npm[ell])
            if c1 >= 0.0 and c2 >= 0.0:
                is_safe = True
            elif c1 >= 0.0:
                is_safe = (c1 + c2) >= 0.0
            elif c2 >= 0.0:
                is_safe = (2.0 * c1 + c2) >= 0.0
            else:
                is_safe = False
            if is_safe:
                w1[b, ell] = v * c1
                w2[b, ell] = v * c2
            else:
                # 截断可能生效：这一对不走矩阵化，权重留 0，由逐对回退补上。
                safe[b, ell] = False

    return w1, w2, safe


@njit(cache=True)
def _p1_spm_jit(
    n_ch: int,
    n_batch: int,
    p_in: NDArrayFloat,
    b_hz_channels: NDArrayFloat,
    n_span: int,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    t_factors: NDArrayFloat,
    r_f: NDArrayFloat,
    l_eff_m: NDArrayFloat,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    cr_w_m_hz: float,
    k_bar: float,
) -> NDArrayFloat:
    """SPM 的批量形态：逐候选、逐信道调用冻结核的同一个函数，故逐位相同。

    Returns
    -------
        ``(B, N)`` 的 SPM 噪声方差。
    """
    out = np.zeros((n_batch, n_ch), dtype=np.float64)
    for b in range(n_batch):
        for i in range(n_ch):
            if p_in[b, i] <= 0:  # 串行基线跳过无功率信道
                continue
            gamma_ii = _calc_gamma_coeff_scalar(lambdas_m[i], lambdas_m[i], n_2_m2_w, a_eff_m2[i], a_eff_m2[i])
            phi_i = 1.5 * PI**2 * (beta_2_s2_m + 2 * PI * f_rel_hz[i] * beta_3_s3_m)
            out[b, i] = _calc_spm_variance_jit(
                p_in[b, i],
                b_hz_channels[i],
                n_span,
                gamma_ii,
                phi_i,
                alpha_power_npm[i],
                t_factors[b, i],
                r_f[b, i],
                cr_w_m_hz,
                k_bar,
                l_eff_m[i],
            )
    return out


@njit(cache=True)
def _p1_xpm_clamp_correction_jit(
    n_ch: int,
    n_batch: int,
    p_in: NDArrayFloat,
    b_hz_channels: NDArrayFloat,
    phi_raw: NDArrayFloat,
    gamma_mat: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    t_factors: NDArrayFloat,
    r_f: NDArrayFloat,
    l_eff_m: NDArrayFloat,
    cr_w_m_hz: float,
    k_bar: float,
    safe: NDArrayBool,
) -> NDArrayFloat:
    """``safe`` 为 False 的那些对改用冻结核的逐对实现算出来加回本通道。

    直接调用 :func:`network_nlo_eval.models.numba_kernels._calc_xpm_variance_term_jit`，
    连同 ``max(·, 0)`` 截断一起逐位相同。全部 ``safe`` 时返回全零，这是常态。
    """
    corr = np.zeros((n_batch, n_ch), dtype=np.float64)
    if np.all(safe):
        return corr

    for b in range(n_batch):
        for i in range(n_ch):
            if p_in[b, i] <= 0:
                continue
            acc = 0.0
            for ell in range(n_ch):
                if ell == i or p_in[b, ell] <= 0 or safe[b, ell]:
                    continue
                acc += _calc_xpm_variance_term_jit(
                    p_in[b, ell],
                    b_hz_channels[ell],
                    b_hz_channels[i],
                    gamma_mat[i, ell],
                    phi_raw[i, ell],
                    alpha_power_npm[ell],
                    t_factors[b, ell],
                    r_f[b, ell],
                    cr_w_m_hz,
                    k_bar,
                    l_eff_m[ell],
                )
            corr[b, i] = acc
    return corr


def _p1_xpm_gemv(w1: NDArrayFloat, w2: NDArrayFloat, mats: P1XpmMatrices) -> NDArrayFloat:
    """XPM 的 ``ell`` 方向求和：两次矩阵-向量乘，形状 ``(B, N)``。

    写成 ``w @ g.T`` 而不是 ``g @ w``：批量轴在前，BLAS 侧一次把整批算完，
    (N,N) 常量在整批里只读一次。
    """
    return w1 @ mats.g1_t + w2 @ mats.g2_t


def calc_span_noise_and_power_p1_matvec(
    power_in_w: NDArrayFloat,
    span_length_m: float,
    f_rel_hz: NDArrayFloat,
    f_abs_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    nf_lin_channels: NDArrayFloat,
    symbol_rate_hz: float,
    b_hz_channels: NDArrayFloat,
    n_effective_spans: int,
    cr_w_m_hz: float,
    k_bar: float,
    delta_f_co_hz: float,
    mats: P1XpmMatrices,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """P1 闭式核的 ``matvec`` 形态：``(B, N)`` 入、四个 ``(B, N)`` 出。

    与 :func:`network_nlo_eval.models.numba_kernels._calc_span_noise_and_power_jit`
    算同一件事，区别只在 XPM 的 ``ell`` 方向求和由成对循环变成矩阵-向量乘。
    出纤功率与 SPM 逐位相同；只有 XPM 的求和次序变了，相对偏差在 1e-16 量级。

    Args:
        power_in_w: ``(B, N)`` 入纤功率 (W)。零功率合法（惰性信道仍占槽位）。
        span_length_m: 跨段长度 (m)。
        f_rel_hz / f_abs_hz: 逐信道相对/绝对频率 (Hz)。
        lambdas_m: 逐信道波长 (m)。
        alpha_power_npm: 逐信道衰减 (Np/m)。
        a_eff_m2: 逐信道有效模面积 (m^2)。
        beta_2_s2_m / beta_3_s3_m: 色散 beta2 / beta3。
        n_2_m2_w: 非线性折射率 n2。
        nf_lin_channels: ``(N,)`` 线性噪声系数。
        symbol_rate_hz: 标量符号率 (Hz)，只用于 SRS 频带边界。
        b_hz_channels: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
        n_effective_spans: 有效跨段数。
        cr_w_m_hz: SRS 增益系数 Cr。
        k_bar: 拉曼泵浦因子。
        delta_f_co_hz: SRS 截断频率 (Hz)。
        mats: :func:`build_p1_xpm_pair_matrices` 产出的 (N,N) 常量包。

    Returns
    -------
        ``(p_out, sigma2_spm, sigma2_xpm, sigma2_ase)``，四个都是 ``(B, N)``。
    """
    p_in = np.ascontiguousarray(power_in_w, dtype=np.float64)
    if p_in.ndim != 2:
        raise ValueError(f"power_in_w must have shape (B, N), got {p_in.shape}.")
    n_ch = int(p_in.shape[1])
    if mats.n_ch != n_ch:
        raise ValueError(f"pair matrices are built for N={mats.n_ch}, got N={n_ch}.")
    if b_hz_channels.shape != (n_ch,):
        raise ValueError(f"b_hz_channels must have shape ({n_ch},), got {b_hz_channels.shape}.")

    if p_in.shape[0] == 0:
        empty = np.empty((0, n_ch), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    p_out, r_f, t_factors, l_eff_m = _p1_span_pre_nli_batch_jit(
        p_in,
        float(span_length_m),
        f_rel_hz,
        float(symbol_rate_hz),
        alpha_power_npm,
        float(cr_w_m_hz),
        float(k_bar),
        float(delta_f_co_hz),
    )

    sigma2_spm = _p1_spm_jit(
        n_ch,
        int(p_in.shape[0]),
        p_in,
        b_hz_channels,
        int(n_effective_spans),
        f_rel_hz,
        lambdas_m,
        a_eff_m2,
        alpha_power_npm,
        t_factors,
        r_f,
        l_eff_m,
        float(beta_2_s2_m),
        float(beta_3_s3_m),
        float(n_2_m2_w),
        float(cr_w_m_hz),
        float(k_bar),
    )

    w1, w2, safe = _p1_xpm_weights_jit(
        n_ch,
        int(p_in.shape[0]),
        p_in,
        b_hz_channels,
        alpha_power_npm,
        t_factors,
        r_f,
        l_eff_m,
        float(cr_w_m_hz),
        float(k_bar),
    )
    xpm_sum = _p1_xpm_gemv(w1, w2, mats)
    # 少数"截断可能生效"的通道：用冻结核逐对算（逐位相同）加回来。
    # 生产参数下这条分支不生效，(N,N) 的 phi_raw / gamma 只为它传进 njit。
    if not bool(safe.all()):
        xpm_sum = xpm_sum + _p1_xpm_clamp_correction_jit(
            n_ch,
            int(p_in.shape[0]),
            p_in,
            b_hz_channels,
            mats.phi_raw,
            mats.gamma,
            alpha_power_npm,
            t_factors,
            r_f,
            l_eff_m,
            float(cr_w_m_hz),
            float(k_bar),
            safe,
        )
    sigma2_xpm = (32.0 / 27.0) * p_in * float(n_effective_spans) * xpm_sum

    # ASE 在闭式路径里是死代码（调用方按显式施加 EDFA 那一步另算），
    # 这里照样返回，只为与冻结内核的返回值逐项对应。
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(p_out > 0, p_in / np.where(p_out > 0, p_out, 1.0), 0.0)
    sigma2_ase = np.where(
        (p_in > 0) & (p_out > 0),
        float(n_effective_spans) * nf_lin_channels * H_PLANCK * f_abs_hz * np.maximum(gain - 1.0, 0.0) * b_hz_channels,
        0.0,
    )

    return p_out, sigma2_spm, sigma2_xpm, sigma2_ase
