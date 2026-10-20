"""Semrau 2019 闭式 ISRS-GN 的常量矩阵与 Numba 热路径。

公式编号的出处（除非另注，均指 Semrau, Killey, Bayvel, *A Closed-Form
Approximation of the Gaussian Noise Model in the Presence of Inter-Channel
Stimulated Raman Scattering*, JLT 2019, doi:10.1109/jlt.2019.2895237）：

===========================  ==========================================
论文位置                      内容
===========================  ==========================================
Eq. (1)                       SNR_i ≈ P_i / (P_ASE + η_n P_i³)
Eq. (2)                       ISRS-GN 积分形式（NLI 系数定义）
Eq. (3)                       ISRS 功率转移 Δρ(z) = 4.3·P_tot·C_r·L_eff·B_tot
Eq. (5)                       多跨累积 η_n = Σ_j (P_ij/P_i)²[η_SPM,j n^ε + η_XPM,j]
Eq. (10)                      SPM 闭式
Eq. (11)                      XPM 闭式
Appendix A Eq. (16)/(17)      线性拉曼增益近似下的功率剖面
Appendix A Eq. (18)           一阶功率剖面下的 |∫dζ|²
===========================  ==========================================

标注 ``2019b`` 的公式出自同组第二篇 *A Modulation Format Correction Formula
for the GN Model in the Presence of ISRS*, JLT 2019,
doi:10.1109/jlt.2019.2929461：Eq. (4)（φ 的定义与超额峭度 Φ）、Eq. (12)
（渐近修正）、Eq. (15)（单跨调制格式修正）、Eq. (16)（含格式修正的总 NLI
系数）、Table I（各调制格式的超额峭度）。

**为什么常量矩阵可以在构造期一次建好**

Eq. (10)/(11) 里的两个括号都是 ``T`` 的**一次**函数::

    bracket = (T - α²)/ᾱ · asinh(φ B²/(πᾱ)) + (A² - T)/A · asinh(φ B²/(πA))
            = T · u + v

其中 ``u``、``v`` 只依赖 (信道对, 光纤参数)，与入纤功率无关；只有 ``T`` 依赖
总入纤功率 ``P_tot``（Eq. (10)：``T_i = (α + ᾱ - P_tot C_r f_i)²``）。于是

    Σ_k P_k² · bracket_{i,k} = (U @ (P²·T))_i + (V @ (P²))_i

可以把 ``U``、``V``（以及 2019b 的渐近项矩阵 ``W``）在核构造期算好，热路径只
剩两次矩阵-向量乘。这是本模型比 ``ecz`` 快的根本原因：``ecz`` 每次调用都要对
每个信道对重算 γ、φ、asinh、atan。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from network_nlo_eval.core.constants import PI
from network_nlo_eval.core.types import NDArrayFloat

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


__all__ = [
    "SemrauMatrices",
    "build_semrau_matrices",
    "semrau_span_jit",
]

#: 与 ``models/numba_kernels.py`` 同口径的 φ 下限，防止除零。
_EPSILON = 1e-35

#: 2019b Eq. (4) 的 |φ| 中 ``β_2 + πβ_3(f_i+f_k)`` 的下限（1/s²·m 量级）。
_BETA_FLOOR = 1e-30


class SemrauMatrices:
    """核构造期一次性算好的逐信道与逐信道对常量。

    Attributes
    ----------
        spm_u, spm_v:
            ``(N,)``：SPM 括号 ``bracket_i = T_i·spm_u[i] + spm_v[i]`` 的两个系数。
        spm_coef:
            ``(N,)``：Eq. (10) 的前置因子 ``(4/9)·γ_ii²·π/(φ_i·3α_i²)``。
        mat_u, mat_v:
            ``(N,N)``：Eq. (11) 的 XPM 系数矩阵（对角线为 0，XPM 不含自项）。
        mat_w:
            ``(N,N)``：2019b Eq. (15) 的渐近项系数矩阵（不含 ``1/L`` 因子）。
    """

    __slots__ = ("mat_u", "mat_v", "mat_w", "spm_coef", "spm_u", "spm_v")

    def __init__(
        self,
        spm_u: NDArrayFloat,
        spm_v: NDArrayFloat,
        spm_coef: NDArrayFloat,
        mat_u: NDArrayFloat,
        mat_v: NDArrayFloat,
        mat_w: NDArrayFloat,
    ) -> None:
        self.spm_u = spm_u
        self.spm_v = spm_v
        self.spm_coef = spm_coef
        self.mat_u = mat_u
        self.mat_v = mat_v
        self.mat_w = mat_w


def _clamp_phi(phi: NDArrayFloat) -> NDArrayFloat:
    """把接近 0 的 φ 夹到同号的下限（与 ``ecz`` 核的 ``_EPSILON`` 同策略）。"""
    small = np.abs(phi) < _EPSILON
    if np.any(small):
        phi = np.where(small, np.where(phi >= 0.0, _EPSILON, -_EPSILON), phi)
    return phi


def build_semrau_matrices(
    *,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    beta2_s2_m: float,
    beta3_s3_m: float,
    n2_m2_w: float,
    with_egn: bool,
) -> SemrauMatrices:
    """构造 Eq. (10)/(11) 与 2019b Eq. (15) 所需的全部常量矩阵。

    所有量都是"相对频带中心"的：``f_k`` 是信道 k 相对频带中心的频率，``f_k=0``
    的信道没有净拉曼增益（论文 Eq. (3) 的 Δρ 以频带总宽 ``B_tot`` 为尺度）。
    ``α``、``A_eff`` 逐信道取值（论文允许把 ``α``、``ᾱ``、``C_r`` 解释成逐信道
    量，见 Eq. (11) 后的说明）。

    Args:
        f_rel_hz: ``(N,)`` 相对频带中心频率 (Hz)。
        lambdas_m: ``(N,)`` 波长 (m)。
        b_hz: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        a_eff_m2: ``(N,)`` 逐信道有效模面积 (m²)。
        beta2_s2_m: ``β_2`` (s²/m)。
        beta3_s3_m: ``β_3`` (s³/m)。
        n2_m2_w: 非线性折射率 n2 (m²/W)。
        with_egn: 是否构造 2019b 的渐近项矩阵。

    Returns
    -------
        :class:`SemrauMatrices`。
    """
    n = int(f_rel_hz.size)
    fi = f_rel_hz[:, None]  # COI 索引 i
    fk = f_rel_hz[None, :]  # 干扰信道索引 k
    bi = b_hz[:, None]
    bk = b_hz[None, :]
    ak = alpha_npm[None, :]

    # γ_ik = (2π/λ_i)·(2n2/(A_eff,i + A_eff,k)) (1/(W·m))
    gamma = (2.0 * PI / lambdas_m[:, None]) * (2.0 * n2_m2_w / (a_eff_m2[:, None] + a_eff_m2[None, :]))

    # --- SPM: Eq. (10) -----------------------------------------------------
    # φ_i = 1.5π²(β_2 + 2πβ_3 f_i)
    phi_spm = _clamp_phi(1.5 * PI**2 * (beta2_s2_m + 2.0 * PI * f_rel_hz * beta3_s3_m))
    asinh_1 = np.arcsinh(phi_spm * b_hz**2 / (PI * alpha_npm))
    asinh_2 = np.arcsinh(phi_spm * b_hz**2 / (2.0 * PI * alpha_npm))
    # bracket = (T-α²)/α·asinh1 + (4α²-T)/(2α)·asinh2 = T·u + v（ᾱ=α, A=2α）
    spm_u = asinh_1 / alpha_npm - asinh_2 / (2.0 * alpha_npm)
    spm_v = -alpha_npm * asinh_1 + 2.0 * alpha_npm * asinh_2
    gamma_ii = (2.0 * PI / lambdas_m) * (2.0 * n2_m2_w / (2.0 * a_eff_m2))
    spm_coef = (4.0 / 9.0) * gamma_ii**2 * PI / (phi_spm * 3.0 * alpha_npm**2)

    # --- XPM: Eq. (11) -----------------------------------------------------
    # φ_{i,k} = 2π²(f_k - f_i)[β_2 + πβ_3(f_i+f_k)]
    phi_xpm = _clamp_phi(2.0 * PI**2 * (fk - fi) * (beta2_s2_m + PI * beta3_s3_m * (fi + fk)))
    atan_1 = np.arctan(phi_xpm * bi / ak)
    atan_2 = np.arctan(phi_xpm * bi / (2.0 * ak))
    # bracket = (T-α²)/α·atan1 + (4α²-T)/(2α)·atan2 = T·u + v
    xpm_u = atan_1 / ak - atan_2 / (2.0 * ak)
    xpm_v = -ak * atan_1 + 2.0 * ak * atan_2
    denom = bk * phi_xpm * 3.0 * ak**2
    mat_u = gamma**2 * xpm_u / denom
    mat_v = gamma**2 * xpm_v / denom
    # XPM 求和不含自项（论文 Eq. (4) 的 S_i 定义）。
    np.fill_diagonal(mat_u, 0.0)
    np.fill_diagonal(mat_v, 0.0)

    # --- 2019b Eq. (15) 的渐近项 -------------------------------------------
    if with_egn:
        delta_f = np.abs(fk - fi)
        numerator = 2.0 * delta_f - bk
        positive = numerator > 0.0
        safe = np.where(positive, numerator, 1.0)
        log_term = np.log(safe / (2.0 * delta_f + bk))
        bracket_q = np.where(positive, numerator * log_term + 2.0 * bk, 0.0)
        # 2019b Eq. (4)：|φ| = 4π²|β_2 + πβ_3(f_i+f_k)|·L，L 在运行期给出。
        beta_sum = np.abs(beta2_s2_m + PI * beta3_s3_m * (fi + fk))
        beta_sum = np.maximum(beta_sum, _BETA_FLOOR)
        # 1/|φ| = 1/(4π²·|β_2+πβ_3(f_i+f_k)|·L)，故运行期再乘 1/L。
        mat_w = gamma**2 * bracket_q / (4.0 * PI**2 * beta_sum * bk**3 * ak**2 * (2.0 * ak) ** 2)
        np.fill_diagonal(mat_w, 0.0)
    else:
        mat_w = np.zeros((n, n), dtype=np.float64)

    return SemrauMatrices(
        spm_u=spm_u,
        spm_v=spm_v,
        spm_coef=spm_coef,
        mat_u=mat_u,
        mat_v=mat_v,
        mat_w=mat_w,
    )


@njit(cache=True)
def semrau_span_jit(
    power_in_w: NDArrayFloat,
    span_length_m: float,
    n_spans: int,
    cr_w_m_hz: float,
    excess_kurtosis: float,
    coherence_epsilon: float,
    per_span: bool,
    alpha_npm: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    spm_u: NDArrayFloat,
    spm_v: NDArrayFloat,
    spm_coef: NDArrayFloat,
    mat_u: NDArrayFloat,
    mat_v: NDArrayFloat,
    mat_w: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """单跨闭式 ISRS-GN：出纤功率 + SPM 噪声 + XPM 噪声。

    噪声功率口径与论文 Eq. (1) 一致：NLI 功率 = η·P_i³，其中 η 由 Eq. (10)/
    (11) 给出；XPM 部分按 Eq. (11) 的 ``(P_k/P_i)²`` 与 ``P_i³`` 相乘后化为
    ``(32/27)·P_i·Σ_k P_k²·(...)``（与 ``ecz`` 的写法逐项一致）。

    Args:
        power_in_w: ``(N,)`` 入纤功率 (W)，0 表示未占用。
        span_length_m: 跨段长度 (m)。
        n_spans: 有效跨段数。
        cr_w_m_hz: 拉曼增益斜率 ``C_r`` (1/(W·m·Hz))；置 0 即退化为无 ISRS。
        excess_kurtosis: 超额峭度 ``Φ`` (2019b Table I)；0 表示高斯调制
            （不叠加调制格式修正）。
        coherence_epsilon: 相干因子指数 ``ε`` (论文 Eq. (5))。
        per_span: ``False`` 返回论文 Eq. (16) 下 n 跨累积后的**总量**；
            ``True`` 返回同一总量的**逐跨等效值**（总量除以 n），使冻结的
            路由层逐跨相加后正好得到 Eq. (16)。``n_spans=1`` 时两者相同。
        alpha_npm: ``(N,)`` 逐信道衰减 (Np/m)。
        f_rel_hz: ``(N,)`` 相对频带中心频率 (Hz)。
        b_hz: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
        spm_u, spm_v, spm_coef: 见 :class:`SemrauMatrices`。
        mat_u, mat_v, mat_w: 见 :class:`SemrauMatrices`。

    Returns
    -------
        ``(power_out_w, spm_noise_w, xpm_noise_w)``，均为 ``(N,)`` 新数组。
    """
    n = int(power_in_w.size)
    power_out = np.zeros(n, dtype=np.float64)
    spm_noise = np.zeros(n, dtype=np.float64)
    xpm_noise = np.zeros(n, dtype=np.float64)

    p_total = 0.0
    for i in range(n):
        p_total += power_in_w[i]

    # --- ISRS 功率剖面（论文 Appendix A Eq. (16)/(17) 的线性拉曼增益近似） ---
    # 有效衰减按 e^{-αζ - P_tot C_r L_eff(ζ) f_k} 沿光纤积分，闭式后得到
    # e^{-α_k L - σ_k}，σ_k = P_tot·C_r·L_eff,k·f_k；归一化因子保持总功率不变。
    tilt = np.zeros(n, dtype=np.float64)
    denom = 0.0
    for i in range(n):
        l_eff = (1.0 - np.exp(-alpha_npm[i] * span_length_m)) / alpha_npm[i]
        tilt[i] = p_total * cr_w_m_hz * l_eff * f_rel_hz[i]
        denom += power_in_w[i] * np.exp(-tilt[i])
    if denom > 0.0:
        for i in range(n):
            power_out[i] = (
                power_in_w[i]
                * np.exp(-alpha_npm[i] * span_length_m - tilt[i])
                * p_total
                / denom
            )

    # --- T 因子 (论文 Eq. (10)：T_i = (α+ᾱ - P_tot C_r f_i)²，ᾱ=α) ----------
    p_sq = np.zeros(n, dtype=np.float64)
    p_sq_t = np.zeros(n, dtype=np.float64)
    t_factors = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t_i = 2.0 * alpha_npm[i] - p_total * cr_w_m_hz * f_rel_hz[i]
        t_factors[i] = t_i * t_i
        p_sq[i] = power_in_w[i] * power_in_w[i]
        p_sq_t[i] = p_sq[i] * t_factors[i]

    # --- 多跨因子 ----------------------------------------------------------
    # 2019b Eq. (11)：ñ = 0（n=1）否则 n。
    n_tilde = 0.0 if n_spans == 1 else float(n_spans)
    if per_span:
        # 逐跨等效值：论文 Eq. (16) 的总量除以 n。
        spm_span_factor = float(n_spans) ** coherence_epsilon
        xpm_span_factor = 1.0 + (5.0 / 6.0) * excess_kurtosis / float(n_spans)
        egn_scale = (5.0 / 3.0) * excess_kurtosis * PI * (1.0 if n_spans > 1 else 0.0)
    else:
        # 论文 Eq. (16) 的总量：SPM 带 n^{1+ε}（相干部分），XPM 带 n + 5Φ/6。
        spm_span_factor = float(n_spans) ** (1.0 + coherence_epsilon)
        xpm_span_factor = float(n_spans) + (5.0 / 6.0) * excess_kurtosis
        egn_scale = (5.0 / 3.0) * excess_kurtosis * PI * n_tilde

    for i in range(n):
        if power_in_w[i] <= 0.0:
            continue
        acc_u = 0.0
        acc_v = 0.0
        acc_w = 0.0
        for k in range(n):
            acc_u += mat_u[i, k] * p_sq_t[k]
            acc_v += mat_v[i, k] * p_sq[k]
            acc_w += mat_w[i, k] * p_sq_t[k]
        gn_term = acc_u + acc_v
        xpm_noise[i] = (
            (32.0 / 27.0)
            * power_in_w[i]
            * (xpm_span_factor * gn_term + egn_scale * acc_w / span_length_m)
        )
        spm_noise[i] = (
            (power_in_w[i] ** 3 / b_hz[i] ** 2)
            * spm_span_factor
            * spm_coef[i]
            * (spm_u[i] * t_factors[i] + spm_v[i])
        )

    return power_out, spm_noise, xpm_noise
