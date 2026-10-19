"""WP-MERGE-FG-FLEX **第 4 轮**：P1 闭式核的加速结构（``matvec`` 档 + 批量入口）。

这是什么
--------
把加速**实现结构**建到 **P1 (23)/(29)/(30)/(34) 闭式**上。第 3 轮时 FG 的加速只做在
那套被移除的 NLI 物理上——闭式 ECZ 虽是**默认档**，但「加速加入 flexgrid-dev」当时
**还没真正完成**。本模块补这一块。

⚠️ 第 5 轮起，本模块是闭式 ECZ 的**唯一**加速实现：被移除的那套物理连同它的
张量化内核与批量核一起离开了本树，加速不再有任何分叉。

⚠️ **本模块不碰 P1 的算式**。:mod:`network_nlo_eval.models.numba_kernels` 里
``6611c416…`` 那份冻结内核**一个字未改**，它就是 ``off`` 档的基准。本模块是
**另立的新实现**（新函数），机制是「构建期算一次 (N,N) 常量 → 运行期降成
矩阵-向量乘」。

P1 的 XPM 为什么**不能**整体一次 matvec
--------------------------------------
P1 (30) 的 XPM 求和项是::

    term(i,l) = (P_l^2 / B_l) * exp(2*Cr*k_bar*L_eff_l*r_fl)
                * (gamma_il^2 / (phi_il * 3 * alpha_l^2))
                * [ c1_l * atan(phi_il * B_i / alpha_l)
                  + c2_l * atan(phi_il * B_i / (2*alpha_l)) ]

其中 ``c1_l = (t_l - alpha_l^2)/alpha_l``、``c2_l = (4*alpha_l^2 - t_l)/(2*alpha_l)``。

``atan`` 的自变量 ``phi_il * B_i / alpha_l`` 同时含**信道对**量 ``phi_il`` 与**逐信道**
量 ``B_i``、``alpha_l``，而 ``atan`` 是非线性函数 ⇒ 这一项**无法**写成
``f(i) * g(l)``。这与 FG 侧的情况**不同**：P2 Eq. 21 的 XPM 在张量化后是 ``l`` 方向
**线性**的（``gamma_sq[i,l] * u_k[b,l]``），所以能整块收成 GEMM。

⇒ 本模块走**分段**路线（任务书 §1.1 明文允许）：

1. **矩阵化**：「与信道对相关」的部分预计算成 (N,N) 矩阵 ``G1`` / ``G2``；
2. **降成 matvec**：``l`` 方向的求和变成矩阵-向量乘；
3. **逐对循环只保留一处**：原实现对**每一对**都做的 ``max(·, 0)`` 截断。

关于第 3 点：截断看起来挡路，其实**可以证明它在绝大多数情况下是恒等的**。
把求和项按「矩阵项 × 权重」摊开::

    term(i,l) = G1[i,l] * w1[l] + G2[i,l] * w2[l]

    G1[i,l] = (gamma_il^2 / (phi_il * 3 * alpha_l^2)) * atan(phi_il * B_i / alpha_l)
    G2[i,l] = (gamma_il^2 / (phi_il * 3 * alpha_l^2)) * atan(phi_il * B_i / (2*alpha_l))
    w1[l]   = (P_l^2 / B_l) * exp(2*Cr*k_bar*L_eff_l*r_fl) * c1_l
    w2[l]   = (P_l^2 / B_l) * exp(2*Cr*k_bar*L_eff_l*r_fl) * c2_l

其中 ``G1`` / ``G2`` 只依赖**信道几何**（波长、频率、带宽、衰减、色散）⇒
**evaluator 级**常量，构建一次、跨跨段与跨候选复用；``w1`` / ``w2`` 只依赖
**逐候选**量（功率、拉曼剖面、t_l）⇒ 每次调用重算，是 (N,) 向量。

**符号**：``gamma_il^2 / (phi_il * 3*alpha_l^2)`` 的符号 = ``sign(phi_il)``；
``atan(phi_il * B_i / alpha_l)``（``B_i, alpha_l > 0``）的符号也 = ``sign(phi_il)``。
⇒ ``G1[i,l] >= 0`` 且 ``G2[i,l] >= 0``（``phi_il`` 被 ``_EPSILON`` 夹住时符号不变），
且 ``G1[i,l] >= G2[i,l]``、``G1[i,l] <= 2*G2[i,l]``（``atan`` 单调且在 ``[0, inf)``
上凹 ⇒ ``atan(2y) <= 2*atan(y)``）。

把求和项按符号摊开：``result(i,l) = v_l * A_il * Q_il``，其中
``Q_il = c1_l * A1_il + c2_l * A2_il``（``A1_il``/``A2_il`` 是两个 ``atan`` 的绝对值）。
原实现的逐对 ``max(·, 0)`` **当且仅当** ``Q_il < 0`` 时才真的生效。

⇒ 逐通道的**充分**判据（O(1)，见 :func:`_p1_xpm_weights_jit` 的推导）：
``c1 >= 0 且 c2 >= 0``、或 ``c1 >= 0 且 c1 + c2 >= 0``、或 ``c2 >= 0 且 2*c1 + c2 >= 0``。
满足则该通道**恒安全**，矩阵化对它**逐位等价**（截断是恒等变换）。

不满足的通道（实测里 ``t_l > 4*alpha_l^2`` 那一侧会出现）改用**原实现逐对算**
（直接调用冻结核里的 :func:`_calc_xpm_variance_term_jit`，**逐位相同**）加回来。
**这不是「静默回落」**：它只影响**个别**通道对的求和项，且走的是**同一个冻结核函数**，
不是另写一份。

精度口径
--------
与 FG 侧的 ``matvec`` 档**同口径**：只重排 ``l`` 方向的浮点求和次序，
**相对偏差预算 1e-12（不放宽）**。SPM 走**原函数的原样调用** ⇒ **逐位相同**；
出纤功率走**原函数的原样调用** ⇒ **逐位相同**；只有 XPM 的求和次序变了。
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
except ImportError:  # 保持统计模型在无 JIT 的最小环境中可运行。

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:  # type: ignore[misc]
        """Return an identity decorator when optional Numba is unavailable."""
        del args, kwargs

        def decorate(function: _F) -> _F:
            return function

        return decorate


#: FG-4a/FG-4b 的相对偏差预算，原样沿用，**不放宽**。
MAX_REL_DEVIATION = 1e-12


@dataclass(frozen=True)
class P1XpmMatrices:
    """P1 XPM 的 **evaluator 级** (N,N) 常量包（构建一次、跨跨段/跨候选复用）。

    形状全部是 ``(N, N)``；**没有 batch 轴**——这正是批量路径省下来的主要常数因子。

    Attributes:
        g1: ``G1[i,l] = (gamma_il^2 / (phi_il*3*alpha_l^2)) * atan(phi_il*B_i/alpha_l)``，
            **逐元素 >= 0**（见模块 docstring 的符号论证）。
        g2: ``G2[i,l] = (gamma_il^2 / (phi_il*3*alpha_l^2)) * atan(phi_il*B_i/(2*alpha_l))``，
            同样 **逐元素 >= 0**。
        g1_t / g2_t: 上面两张矩阵的**连续转置**。热路径写成 ``w @ g_t``，
            免得每步取一次转置视图。
        phi_raw: **未夹**的 ``phi_il``。矩阵化内部用的是夹过 ``_EPSILON`` 的值，
            但逐对回退路径要交给冻结核的 :func:`_calc_xpm_variance_term_jit`
            **自己夹**（那才是逐位相同的做法），所以原始值要留着。
        gamma: ``gamma_il``。同上，回退路径要用。
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

    算式**逐字**取自冻结核 :func:`network_nlo_eval.models.numba_kernels._compute_nli_variances`
    与 ``_calc_xpm_variance_term_jit``：同样的 ``phi_il`` 写法、同样的
    ``_EPSILON`` 夹取、同样的 ``gamma_il`` 调用、同样的两个 ``arctan``。
    **只是把它们从「每对现算」提前到「构建期算一次」**。

    对角线（``i == l``）置零：串行基线那里是 ``continue``，跳过自配对。

    Args:
        n_ch: 信道数 N。
        b_hz_channels: 逐信道带宽 ``B_i`` (Hz) —— P1 (29)/(30) 的那个 ``B_i``。
        f_rel_hz: 逐信道相对频率 (Hz)。
        lambdas_m: 逐信道波长 (m)。
        a_eff_m2: 逐信道有效模面积 (m^2)。
        alpha_power_npm: 逐信道功率衰减系数 (Np/m)。
        beta_2_s2_m: 色散 beta2 (s^2/m)。
        beta_3_s3_m: 色散 beta3 (s^3/m)。
        n_2_m2_w: 非线性折射率 n2 (m^2/W)。

    Returns:
        ``(g1, g2, phi_raw, gamma)``，四张都是 ``(N, N)``。
    """
    g1 = np.zeros((n_ch, n_ch), dtype=np.float64)
    g2 = np.zeros((n_ch, n_ch), dtype=np.float64)
    phi_raw = np.zeros((n_ch, n_ch), dtype=np.float64)
    gamma_mat = np.zeros((n_ch, n_ch), dtype=np.float64)

    for i in range(n_ch):
        for l in range(n_ch):
            if l == i:  # 串行基线那里是 continue：跳过自配对
                continue

            gamma_il = _calc_gamma_coeff_scalar(
                lambdas_m[i], lambdas_m[l], n_2_m2_w, a_eff_m2[i], a_eff_m2[l]
            )

            # 简化计算 phi_il
            # phi_il = 2 * PI**2 * (f_rel_hz[l] - f_rel_hz[i])
            # * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[l] + f_rel_hz[i]))
            # 原始公式: 2*pi^2*Δf*(beta2 + pi*beta3*(f_l+f_i)).
            phi_il = (
                2
                * PI**2
                * (f_rel_hz[l] - f_rel_hz[i])
                * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[l] + f_rel_hz[i]))
            )
            phi_raw[i, l] = phi_il
            gamma_mat[i, l] = gamma_il

            # 修正 phi_il 防止零或过小的值（与 _calc_xpm_variance_term_jit 同）
            phi_c = phi_il
            if abs(phi_c) < _EPSILON:
                phi_c = _EPSILON if phi_c >= 0 else -_EPSILON

            a_il = gamma_il**2 / (phi_c * 3 * alpha_power_npm[l] ** 2)
            atan_1 = np.arctan((phi_c * b_hz_channels[i]) / alpha_power_npm[l])
            atan_2 = np.arctan((phi_c * b_hz_channels[i]) / (2 * alpha_power_npm[l]))

            g1[i, l] = a_il * atan_1
            g2[i, l] = a_il * atan_2

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
    的第 1/2/3/4 步**逐字相同**（同样的调用、同样的次序、同样的表达式），
    所以出纤功率与 ``t_factors`` **逐位相同**。

    出纤功率的公共子表达式消除（CSE）
    --------------------------------
    冻结核的出纤功率对每个 ``i`` 都要算一遍 ``exp(u_i * r_f)``，其中::

        u_i = (-(1 - k_bar)) * _SRS_CR * L_eff[i]

    而 ``L_eff[i] = (1 - exp(-alpha_i*L))/alpha_i`` **只依赖逐信道衰减**。
    在 ``p1_closed`` 的**默认**衰减口径（常数衰减）下，全部 ``alpha_i`` 相等
    ⇒ 全部 ``u_i`` **逐位相等** ⇒ 那 N 条指数向量是**同一条**。

    本函数据此把指数向量的条数从 N 降到「不同 u 的个数」（默认口径下 = 1），
    指数计算量从 O(N^2) 降到 O(N)。**这是恒等变换，不是改算式**：
    ``u_i`` 逐位相等 ⇒ ``exp(u_i * r_f[j])`` 的输入逐位相同 ⇒ 输出逐位相同。
    实测出纤功率与冻结核**逐位相等**（见报告第 4 轮章节）。

    ``u`` 互不相等的场合（``use_attenuation_polynomial=True`` 的波长多项式衰减）
    自动退回逐 ``i`` 计算，代价与冻结核相同——**不更差**。

    ``f_m_hz`` / ``f_M_hz`` 用的是**核自己的**口径（``f_rel_hz[0] - Rs/2``），
    **不是** evaluator 的 ``self.f_m_hz``（那个按连接带边算）——冻结核用的是前者，
    这里必须跟它一致，否则拉曼剖面就变了。

    Args:
        p_in: ``(B, N)`` 入纤功率 (W)。
        span_length_m: 跨段长度 (m)。
        f_rel_hz: 逐信道相对频率 (Hz)。
        symbol_rate_hz: 标量符号率 (Hz)，只用于 SRS 频带边界。
        alpha_power_npm: ``(N,)`` 逐信道功率衰减系数 (Np/m)。
        cr_w_m_hz: SRS 增益系数 Cr。
        k_bar: 拉曼泵浦因子。
        delta_f_co_hz: SRS 截断频率 (Hz)。

    Returns:
        ``(p_out, r_f, t_factors, l_eff)``：前三个 ``(B, N)``，最后一个 ``(N,)``。
    """
    n_batch = p_in.shape[0]
    n_ch = p_in.shape[1]
    p_out = np.zeros((n_batch, n_ch), dtype=np.float64)
    r_f = np.zeros((n_batch, n_ch), dtype=np.float64)
    t_factors = np.zeros((n_batch, n_ch), dtype=np.float64)

    l_eff_m = (1.0 - np.exp(-alpha_power_npm * span_length_m)) / alpha_power_npm

    # u_i = (-(1 - k_bar)) * _SRS_CR * L_eff[i]  —— 与冻结核的乘法次序逐字相同
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

        r_f_values = _calc_raman_profile_jit(
            f_rel_hz, f_m_hz, f_M_hz, delta_f_co_hz, total_p_in_w
        )
        r_f[b, :] = r_f_values

        if u_all_same:
            # 默认口径：N 条指数向量其实是同一条 ⇒ 只算一次。
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
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """XPM 的逐候选权重向量 ``(w1, w2)`` 与逐对安全掩码 ``safe``。

    ``w1[l] = (P_l^2/B_l) * exp(2*Cr*k_bar*L_eff_l*r_fl) * c1_l``，
    ``w2[l] = (P_l^2/B_l) * exp(2*Cr*k_bar*L_eff_l*r_fl) * c2_l``，
    其中 ``c1_l = (t_l - alpha_l^2)/alpha_l``、``c2_l = (4*alpha_l^2 - t_l)/(2*alpha_l)``。

    安全判据：原实现的逐对 ``max(·, 0)`` 截断**会不会真的生效**。

    把求和项按符号摊开（``A_il = |gamma_il^2/(phi_il*3*alpha_l^2)| >= 0``、
    ``A1_il = |atan(phi_il*B_i/alpha_l)|``、``A2_il = |atan(phi_il*B_i/(2*alpha_l))|``）::

        result(i,l) = v_l * A_il * Q_il ,   Q_il = c1_l * A1_il + c2_l * A2_il

    ``A1_il >= A2_il``（``atan`` 单调）且 ``A1_il <= 2 * A2_il``
    （``atan`` 在 ``[0, inf)`` 上凹且 ``atan(0)=0`` ⇒ ``atan(2y) <= 2*atan(y)``）。
    由此得到**充分**判据（逐通道 O(1)）：

    * ``c1 >= 0`` 且 ``c2 >= 0`` ⇒ ``Q >= 0``，恒安全；
    * ``c1 >= 0`` 且 ``c2 < 0`` ⇒ ``G1*c1 + G2*c2 >= G1*(c1+c2)``（``G2 <= G1``），
      故 ``c1 + c2 >= 0`` 时恒安全；
    * ``c1 < 0`` 且 ``c2 >= 0`` ⇒ ``G1*c1 + G2*c2 >= G2*(2*c1+c2)``（``G1 <= 2*G2``），
      故 ``2*c1 + c2 >= 0`` 时恒安全；
    * 两者都负 ⇒ ``Q < 0``，截断**一定**生效 ⇒ 该通道必须走逐对回退。

    掩码为 False 的通道，权重留 0，由
    :func:`_p1_xpm_clamp_correction_jit` 用**冻结核的逐对实现**算出来加回
    （那部分与 ``off`` 档**逐位相同**）。

    Args:
        n_ch: 信道数 N。
        n_batch: 批宽 B。
        p_in: ``(B, N)`` 入纤功率 (W)。
        b_hz_channels: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
        alpha_power_npm: ``(N,)`` 逐信道衰减 (Np/m)。
        t_factors: ``(B, N)`` 逐候选的 ``t_l``。
        r_f: ``(B, N)`` 逐候选的拉曼剖面 ``r_f``。
        l_eff_m: ``(N,)`` 逐信道有效长度 (m)。
        cr_w_m_hz: SRS 增益系数 Cr。
        k_bar: 拉曼泵浦因子。

    Returns:
        ``(w1, w2, safe)``，前两个 ``(B, N)``，第三个 ``(B, N)`` 布尔。
    """
    w1 = np.zeros((n_batch, n_ch), dtype=np.float64)
    w2 = np.zeros((n_batch, n_ch), dtype=np.float64)
    safe = np.ones((n_batch, n_ch), dtype=np.bool_)

    for b in range(n_batch):
        for l in range(n_ch):
            term_exp = np.exp(2 * cr_w_m_hz * k_bar * l_eff_m[l] * r_f[b, l])
            v = (p_in[b, l] ** 2 / b_hz_channels[l]) * term_exp
            c1 = (t_factors[b, l] - alpha_power_npm[l] ** 2) / alpha_power_npm[l]
            c2 = (4 * alpha_power_npm[l] ** 2 - t_factors[b, l]) / (2 * alpha_power_npm[l])
            if c1 >= 0.0 and c2 >= 0.0:
                is_safe = True
            elif c1 >= 0.0:  # c2 < 0
                is_safe = (c1 + c2) >= 0.0
            elif c2 >= 0.0:  # c1 < 0
                is_safe = (2.0 * c1 + c2) >= 0.0
            else:
                is_safe = False
            if is_safe:
                w1[b, l] = v * c1
                w2[b, l] = v * c2
            else:
                # 截断可能生效：这一对不走矩阵化（权重留 0，由逐对回退补上）。
                safe[b, l] = False

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
    """SPM 的批量形态：逐候选、逐信道调用**冻结核的同一个函数**。

    ⇒ 与 ``off`` 档的 SPM **逐位相同**（连浮点求和次序都没变，因为压根没改算式）。

    Returns:
        ``(B, N)`` 的 SPM 噪声方差。
    """
    out = np.zeros((n_batch, n_ch), dtype=np.float64)
    for b in range(n_batch):
        for i in range(n_ch):
            if p_in[b, i] <= 0:  # 串行基线那里是 continue：无功率不计算
                continue
            gamma_ii = _calc_gamma_coeff_scalar(
                lambdas_m[i], lambdas_m[i], n_2_m2_w, a_eff_m2[i], a_eff_m2[i]
            )
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
    """``safe`` 为 False 的那些对：改用**冻结核的逐对实现**算出来加回。

    直接调用 :func:`network_nlo_eval.models.numba_kernels._calc_xpm_variance_term_jit`
    ⇒ 这部分与 ``off`` 档**逐位相同**（包括 ``max(·, 0)`` 截断本身）。

    全部 ``safe`` 时直接返回全零（热路径的常态，实测从未触发过回退）。
    """
    corr = np.zeros((n_batch, n_ch), dtype=np.float64)
    if np.all(safe):
        return corr

    for b in range(n_batch):
        for i in range(n_ch):
            if p_in[b, i] <= 0:
                continue
            acc = 0.0
            for l in range(n_ch):
                if l == i or p_in[b, l] <= 0 or safe[b, l]:
                    continue
                acc += _calc_xpm_variance_term_jit(
                    p_in[b, l],
                    b_hz_channels[l],
                    b_hz_channels[i],
                    gamma_mat[i, l],
                    phi_raw[i, l],
                    alpha_power_npm[l],
                    t_factors[b, l],
                    r_f[b, l],
                    cr_w_m_hz,
                    k_bar,
                    l_eff_m[l],
                )
            corr[b, i] = acc
    return corr


def _p1_xpm_gemv(
    w1: NDArrayFloat,
    w2: NDArrayFloat,
    mats: P1XpmMatrices,
) -> NDArrayFloat:
    """XPM 的 ``l`` 方向求和：两次矩阵-向量乘，形状 ``(B, N)``。

    ``acc[b,i] = sum_l G1[i,l]*w1[b,l] + G2[i,l]*w2[b,l]``。

    写法是 ``w @ g.T`` 而不是 ``g @ w``：批量轴在前，BLAS 侧一次就把整批算完，
    (N,N) 常量矩阵在整批里**只读一次**。
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
    """P1 闭式核的 **matvec** 形态：``(B, N)`` 入、四个 ``(B, N)`` 出。

    与 :func:`network_nlo_eval.models.numba_kernels._calc_span_noise_and_power_jit`
    算的是同一件事，区别只在 XPM 的 ``l`` 方向求和由**成对循环**变成
    **矩阵-向量乘**（且 (N,N) 常量由构建期算好、整批复用）。

    **出纤功率与 SPM 逐位相同**（走冻结核的同一个函数）；**只有 XPM 的求和次序**
    变了 ⇒ 与 ``off`` 档的相对偏差在 1e-16 量级（预算 1e-12，不放宽）。

    Args:
        power_in_w: ``(B, N)`` 入纤功率 (W)。零功率合法（惰性通道仍占槽位）。
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

    Returns:
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

    # --- SPM：走冻结核的同一个函数 ⇒ 逐位相同 ---
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

    # --- XPM：l 方向收进矩阵侧 → 两次 matvec ---
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
    # 少数「截断可能生效」的对：改用冻结核逐对算（逐位相同）加回来。
    # ⚠️ 实测（68/142 信道、混合与均匀两档）**全部通道都安全**，所以这条分支
    #    在生产参数下**一次都不走**。它必须存在，但不该在热路径上付代价：
    #    (N,N) 的 phi_raw / gamma 只在**真的**需要回退时才传进 njit。
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
    # 串行基线：sigma_xpm_2[i] = (32/27) * p_in[i] * n_span * xpm_sum_term
    sigma2_xpm = (32.0 / 27.0) * p_in * float(n_effective_spans) * xpm_sum

    # --- ASE：与冻结核同式（⚠️ 在 p1_closed 路径里是**死代码**，调用方丢弃它）---
    # 这里照样算出来，只为让本函数与冻结核的返回值**逐项对应**。
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(p_out > 0, p_in / np.where(p_out > 0, p_out, 1.0), 0.0)
    sigma2_ase = np.where(
        (p_in > 0) & (p_out > 0),
        float(n_effective_spans)
        * nf_lin_channels
        * H_PLANCK
        * f_abs_hz
        * np.maximum(gain - 1.0, 0.0)
        * b_hz_channels,
        0.0,
    )

    return p_out, sigma2_spm, sigma2_xpm, sigma2_ase
