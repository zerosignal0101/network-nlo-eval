r"""Zhang 2024 后向 DRA 相干 EGN：Eq. (1)-(5) 的实现与逐项标定。

**论文**

Y. Zhang, Q. Guo, L. Gan, L. Yi, Q. Qiu, W. Hu, Z. Wang, Q. Zhuge,
"Accurate Closed-Form Coherent EGN Model Supporting Distributed Raman
Amplification", OECC 2024, doi:10.1109/OECC54135.2024.10975528.

**论文给了什么、没给什么**

给全的：

* Eq. (1)：后向 DRA 的归一化功率剖面
  ``rho_i(z) = (1+T_i) e^{-alpha_i z} - T_i e^{-A_i z} + D_i e^{-d_i (L-z)}``；
* Eq. (2)：纵向积分的闭式
  ``int_0^L rho e^{j phi z} dz``，并定义 ``T_F``、``T_B``；
* Eq. (3)：``eta = eta_SPM^GN + eta_XPM^GN + eta_XPM^corr``；
* Eq. (4)：多跨相干叠加的**结构**——第 m 跨的场乘 ``exp(j sum_{q<m} phi_q L_q)``
  后与其它跨复数相加，再取模方；
* Eq. (5)：把模方展开成同跨项 ``{P_FmP_Fm + P_BmP_Bm + P_FmP_Bm}`` 与
  跨间项 ``{P_FmP_Fn + ... + P_BmP_Bn}``；
* Sec. II-C：标定式的**形式** ``P.P.^EGN/P.P.^CFM = prod_i 1/(a_i exp(b_i Param_i^{c_i}+d_i))``。

**没给**（Sec. II-B 原话："the specific expressions of the derived closed-form
terms are not provided in this paper"）：

* Eq. (5) 里每个 ``P`` 项的解析表达式；
* 标定系数 ``a_i, b_i, c_i, d_i``、参与拟合的 ``Param_i`` 清单与数据；
* ``eta(f_i)`` 的完整定义（只引 Carena [12]）；
* 512 组算例的取值与随机种子。

因此**无法**逐字复现论文的 CFM。本模块的做法是：

1. Eq. (1)/(2) 逐字实现；
2. Eq. (4)/(5) 按论文的**结构**精确实现——跨间相位放在频率积分**内部**，
   而不是路由层那种"每跨一个标量相位"的近似。这一步给出的是论文用来当
   基准的 integral-form coherent EGN；
3. 同跨 SPM/XPM 的闭式用 Buglia 2024 Eq. (11)/(12) 的同类推导（Zhang 的
   Eq. (1) 也是三指数之和，数学结构相同），作为 CFM 的"闭式项"；
4. 跨间项的闭式表达式没有公开，本模块用一维/二维数值积分求值，并明确标注
   它是"本仓求值"，不是论文的印刷式；
5. Sec. II-C 的标定流程照样实现：逐项计算 ``EGN/CFM`` 比值，按
   ``f(Params)`` 的函数形式最小二乘拟合 ``a_i,b_i,c_i,d_i``，系数由本仓
   在 Table I 的参数范围内生成的数据集拟合（README 与
   ``results/raman/full_zhang2024.json`` 标注为"本仓拟合"）。

**跨间相位的定义**

Eq. (4) 的 ``phi_q`` 论文未给。本实现取与 GN/EGN 一致的相位失配：第 m 跨的
场相对第 0 跨的相位是 ``exp(j phi (z_m - z_0))``，``z_m`` 是第 m 跨的起点，
``phi`` 是该 (f1, f2, f_i) 组合的相位失配（SPM 用 ``phi_i f1 f2``，XPM 用
``phi_ik f1``）。这与 Eq. (4) 的 ``sum_{q=0}^{m-1} phi_q L_q`` 在等跨长时等价，
且对异构跨自然成立。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat

__all__ = [
    "ZhangCoherentSpan",
    "ZhangSpanProfile",
    "calibration_factor",
    "coherent_eta",
    "fit_zhang_coherent_profile",
    "link_function",
    "phase_spm",
    "phase_xpm",
    "single_span_eta_closed",
]

#: 拟合衰减率的上下界 (1/m)。
_MIN_RATE_NPM: float = 1.0e-7
_MAX_RATE_NPM: float = 5.0e-3

#: 相位失配的下限；低于它闭式退化，改用积分。
_PHI_FLOOR: float = 1.0e-35

#: ``A_i`` 的下界（相对 ``alpha_i``）。不加下界时最小二乘会把 ``A_i`` 压到
#: 1e-7（ISRS 项退化成常数），同跨 SPM 闭式的 ``asinh`` 分支随之失效
#: （与积分差约 5 倍）。取 0.2 倍后剖面 RMSE 反而从 2.2% 降到 1.8%，
#: 且 ``A_i`` 保持在物理上有意义的范围。
_A_LOWER_FACTOR: float = 0.2


# =============================================================================
# Eq. (1)：后向 DRA 的指数剖面
# =============================================================================


@dataclass(frozen=True)
class ZhangSpanProfile:
    r"""Zhang Eq. (1) 剖面参数（逐信道）。

    .. math::

        \rho_i(\zeta) = (1+\tilde T_i) e^{-\alpha_i \zeta}
            - \tilde T_i e^{-A_i \zeta} + D_i e^{-d_i (L-\zeta)}

    论文把 ``alpha_i`` 当作已知的光纤衰减（"`alpha_i` denotes the fiber
    attenuation of the i-th channel"），只拟合 ``T, A, D, d`` 四个系数。

    Attributes
    ----------
        alpha_npm: ``(N,)`` 信道衰减 (1/m)。
        t_coeff: ``(N,)`` ``T_i``。
        a_npm: ``(N,)`` ``A_i = alpha_i + alpha_bar_i`` (1/m)。
        d_coeff: ``(N,)`` ``D_i``。
        d_npm: ``(N,)`` ``d_i`` (1/m)。
        f_hz: ``(N,)`` 信道频率 (Hz)，降序。
        bandwidths_hz: ``(N,)`` 信道带宽 (Hz)。
        length_m: 跨长 (m)。
        rmse: ``(N,)`` 相对 RMSE。
    """

    alpha_npm: NDArrayFloat
    t_coeff: NDArrayFloat
    a_npm: NDArrayFloat
    d_coeff: NDArrayFloat
    d_npm: NDArrayFloat
    f_hz: NDArrayFloat
    bandwidths_hz: NDArrayFloat
    length_m: float
    rmse: NDArrayFloat

    @property
    def n_channels(self) -> int:
        """信道数。"""
        return int(np.asarray(self.f_hz).size)

    def value(self, z_m: NDArrayFloat, index: int) -> NDArrayFloat:
        """在给定纵向位置求 ``rho_i(z)``。"""
        z = np.asarray(z_m, dtype=np.float64)
        alpha = float(self.alpha_npm[index])
        t_value = float(self.t_coeff[index])
        a_value = float(self.a_npm[index])
        d_value = float(self.d_coeff[index])
        d_rate = float(self.d_npm[index])
        return (
            (1.0 + t_value) * np.exp(-alpha * z)
            - t_value * np.exp(-a_value * z)
            + d_value * np.exp(-d_rate * (self.length_m - z))
        )


def _zhang_model(parameters: NDArrayFloat, z_m: NDArrayFloat, length_m: float, alpha: float) -> NDArrayFloat:
    """``(T, A, D, d)`` 形式的 Eq. (1)。"""
    t_value, a_value, d_value, d_rate = parameters
    return (
        (1.0 + t_value) * np.exp(-alpha * z_m)
        - t_value * np.exp(-a_value * z_m)
        + d_value * np.exp(-d_rate * (length_m - z_m))
    )


def fit_zhang_coherent_profile(
    profile_norm: NDArrayFloat,
    length_m: float,
    frequencies_hz: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    n_steps: int,
) -> ZhangSpanProfile:
    """把 Eq. (1) 拟合到耦合 ODE 的剖面（论文 Sec. II-A 的流程）。

    Eq. (1) 对 ``(T, D)`` 是线性的，因此先对 ``(A, d)`` 做网格 + 线性最小二乘
    选起点，再做 4 参数联合精修。``alpha_i`` 按论文取光纤衰减，不参与拟合。

    Args:
        profile_norm: ``(2*n_steps, N)`` 归一化剖面，降序。
        length_m: 跨长 (m)。
        frequencies_hz: ``(N,)`` 信道频率 (Hz)。
        bandwidths_hz: ``(N,)`` 信道带宽 (Hz)。
        alpha_npm: ``(N,)`` 光纤衰减 (1/m)。
        n_steps: RK4 步数。

    Returns
    -------
        :class:`ZhangSpanProfile`。
    """
    step = length_m / n_steps
    z = np.empty(2 * n_steps, dtype=np.float64)
    for index in range(n_steps):
        z[2 * index] = step * (index + 0.5)
        z[2 * index + 1] = step * (index + 1.0)
    z_fit = np.concatenate(([0.0], z))
    n_channels = int(np.asarray(frequencies_hz).size)
    t_coeff = np.zeros(n_channels, dtype=np.float64)
    a_npm = np.full(n_channels, 2.0e-5, dtype=np.float64)
    d_coeff = np.zeros(n_channels, dtype=np.float64)
    d_npm = np.full(n_channels, 4.0e-5, dtype=np.float64)
    rmse = np.empty(n_channels, dtype=np.float64)
    floor = 0.05

    for index in range(n_channels):
        alpha = float(np.clip(alpha_npm[index], _MIN_RATE_NPM, _MAX_RATE_NPM))
        target = np.concatenate(([1.0], profile_norm[:, index]))
        weight = 1.0 / np.maximum(target, floor)

        def residual(parameters: NDArrayFloat, target=target, weight=weight, alpha=alpha) -> NDArrayFloat:
            return (_zhang_model(parameters, z_fit, length_m, alpha) - target) * weight

        base = np.exp(-alpha * z_fit)
        best: tuple[float, NDArrayFloat] | None = None
        a_floor = _A_LOWER_FACTOR * alpha
        for a_guess in (a_floor, 8.0e-5, 3.0e-4):
            for d_guess in (2.0e-5, 1.0e-4, 3.0e-4):
                basis = (
                    np.stack(
                        [
                            base - np.exp(-a_guess * z_fit),
                            np.exp(-d_guess * (length_m - z_fit)),
                        ],
                        axis=1,
                    )
                    * weight[:, None]
                )
                rhs = (target - base) * weight
                try:
                    linear, *_ = np.linalg.lstsq(basis, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                start = np.array([linear[0], a_guess, linear[1], d_guess])
                cost = float(np.sqrt(np.mean(residual(start) ** 2)))
                if best is None or cost < best[0]:
                    best = (cost, start)

        parameters = np.array([0.0, alpha, 0.0, alpha]) if best is None else best[1]
        try:
            solved = least_squares(
                residual,
                np.clip(parameters, [-5.0, a_floor, -5.0, _MIN_RATE_NPM], [5.0, _MAX_RATE_NPM, 5.0, _MAX_RATE_NPM]),
                bounds=([-5.0, a_floor, -5.0, _MIN_RATE_NPM], [5.0, _MAX_RATE_NPM, 5.0, _MAX_RATE_NPM]),
                x_scale=np.array([1.0, 1.0e-5, 1.0, 1.0e-5]),
                max_nfev=4000,
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
            cost = float(np.sqrt(np.mean(residual(solved.x) ** 2)))
            if best is None or cost <= best[0]:
                parameters = solved.x
        except (ValueError, np.linalg.LinAlgError):
            pass

        t_coeff[index], a_npm[index], d_coeff[index], d_npm[index] = (
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            float(parameters[3]),
        )
        rmse[index] = float(np.sqrt(np.mean(residual(parameters) ** 2)))

    return ZhangSpanProfile(
        alpha_npm=np.asarray(alpha_npm, dtype=np.float64),
        t_coeff=t_coeff,
        a_npm=a_npm,
        d_coeff=d_coeff,
        d_npm=d_npm,
        f_hz=np.asarray(frequencies_hz, dtype=np.float64),
        bandwidths_hz=np.asarray(bandwidths_hz, dtype=np.float64),
        length_m=float(length_m),
        rmse=rmse,
    )


# =============================================================================
# Eq. (2)：纵向积分的闭式
# =============================================================================


def link_function(profile: ZhangSpanProfile, index: int, phi: NDArrayFloat) -> NDArrayComplex:
    r"""Zhang Eq. (2)：``int_0^L rho_i(z) exp(j phi z) dz``。

    论文印刷式是

    .. math::

        -\frac{1+\tilde T}{-\alpha+j\phi}
        + \frac{\tilde T}{-A+j\phi}
        + \frac{D}{d+j\phi} e^{j\phi L}.

    逐项计算可得精确积分是

    .. math::

        \frac{e^{-\alpha L+j\phi L}-1}{-\alpha+j\phi}(1+\tilde T)
        -\frac{e^{-AL+j\phi L}-1}{-A+j\phi}\tilde T
        +\frac{e^{j\phi L}-e^{-dL}}{d+j\phi}D,

    即印刷式对三项都只保留了上限项、丢掉 ``e^{-alpha L}``/``e^{-A L}``/``e^{-dL}``
    （长跨/高损耗近似）。但印刷式**只给第三项留了公共相位** ``e^{j\phi L}``，
    前两项没有。逐项照抄会让前向项与后向项落在不同的相位参考上，与 Eq. (4)
    的跨间相干口径冲突（同一个 ``phi`` 下两跨的相对相位会错）。

    本实现取三者统一的写法

    .. math::

        e^{j\phi L}\left[-\frac{1+\tilde T}{-\alpha+j\phi}
        + \frac{\tilde T}{-A+j\phi} + \frac{D}{d+j\phi}\right],

    即"丢掉下限项、保留公共相位"，与 Buglia Eq. (9) 的 ``kappa_f -> 1`` 极限一致。
    公共相位在 ``|mu|^2`` 里抵消；只有跨间叠加时才有意义。

    Args:
        profile: Eq. (1) 参数。
        index: 信道下标。
        phi: ``(M,)`` 相位失配斜率 (1/m)。

    Returns
    -------
        ``(M,)`` 复数纵向积分。
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=np.complex128))
    alpha = float(profile.alpha_npm[index])
    t_value = float(profile.t_coeff[index])
    a_value = float(profile.a_npm[index])
    d_value = float(profile.d_coeff[index])
    d_rate = float(profile.d_npm[index])
    length = float(profile.length_m)
    common = np.exp(1j * phi * length)
    forward = -(1.0 + t_value) / (-alpha + 1j * phi) + t_value / (-a_value + 1j * phi)
    backward = d_value / (d_rate + 1j * phi)
    return common * (forward + backward)


def _link_squared(profile: ZhangSpanProfile, index: int, phi: NDArrayFloat) -> NDArrayFloat:
    """``|Eq. (2)|^2``。"""
    return np.abs(link_function(profile, index, phi)) ** 2


# =============================================================================
# 相位失配（与 Buglia Eq. 34/42 同一定义）
# =============================================================================


def phase_spm(beta2: float, beta3: float, frequency_hz: float, reference_frequency_hz: float) -> float:
    """``phi_i = -4 pi^2 [beta2 + 2 pi beta3 (f_i - f_ref)]`` (1/m)。"""
    return float(-4.0 * np.pi**2 * (beta2 + 2.0 * np.pi * beta3 * (frequency_hz - reference_frequency_hz)))


def phase_xpm(
    beta2: float,
    beta3: float,
    coi_frequency_hz: float,
    interferer_frequency_hz: float,
    reference_frequency_hz: float,
) -> float:
    """``phi_{i,k} = -4 pi^2 (f_k - f_i)[beta2 + pi beta3 (f_i + f_k - 2 f_ref)]`` (1/m)。"""
    separation = interferer_frequency_hz - coi_frequency_hz
    dispersion = beta2 + np.pi * beta3 * (coi_frequency_hz + interferer_frequency_hz - 2.0 * reference_frequency_hz)
    return float(-4.0 * np.pi**2 * separation * dispersion)


# =============================================================================
# Eq. (3)-(5)：相干 NLI
# =============================================================================


@dataclass(frozen=True)
class ZhangCoherentSpan:
    """链路里的一跨：剖面 + 入纤功率。

    Attributes
    ----------
        profile: Eq. (1) 拟合剖面。
        launch_power_w: ``(N,)`` 该跨每信道入纤功率 (W)。
        start_m: 该跨起点在链路中的纵向位置 (m)。
    """

    profile: ZhangSpanProfile
    launch_power_w: NDArrayFloat
    start_m: float


def _coherent_field(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    index: int,
    phi: NDArrayFloat,
    position_scale: float,
) -> NDArrayComplex:
    r"""Eq. (4)：``sum_m exp(j phi z_m) (T_Fm + T_Bm)``，按跨的入纤功率加权。

    权重取 ``P_{i,m}/P_i``：由 Eq. (3)，单跨 NLI 功率正比于 ``P_i^3``，
    而 Eq. (4) 再乘 ``(P_{i,j}/P_i)^2``，故跨间场的幅度正比于 ``P_{i,m}/P_i``。
    """
    total = np.zeros_like(phi, dtype=np.complex128)
    for span in spans:
        ratio = float(span.launch_power_w[index]) / max(float(reference_powers_w[index]), 1.0e-30)
        total += ratio * np.exp(1j * phi * span.start_m * position_scale) * link_function(span.profile, index, phi)
    return total


def _band_average(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    index: int,
    phase_slope: float,
    half_band_hz: float,
    position_scale: float,
    alpha_reference_npm: float,
    nodes: NDArrayFloat,
    weights: NDArrayFloat,
) -> float:
    r"""``int_{-U}^{U} |Eq. (4)|^2 d f1``，用 ``tan`` 变换分辨相干峰。

    相干峰宽度是 ``alpha/|phi|``（典型百 MHz），均匀节点的 Gauss-Legendre
    会整段漏掉；``v = (alpha/|phi|) tan(s)`` 把节点压到峰上。
    """
    tangent = abs(phase_slope) * half_band_hz / max(alpha_reference_npm, _MIN_RATE_NPM)
    if tangent <= 1.0e-12:
        field = _coherent_field(spans, reference_powers_w, index, np.zeros(1), position_scale)
        return float(2.0 * half_band_hz * abs(field[0]) ** 2)
    angles = 0.5 * np.arctan(tangent) * (nodes + 1.0)
    quadrature_weights = 0.5 * np.arctan(tangent) * weights
    values = (alpha_reference_npm / phase_slope) * np.tan(angles)
    jacobian = alpha_reference_npm / abs(phase_slope) / np.cos(angles) ** 2
    field = _coherent_field(spans, reference_powers_w, index, phase_slope * values, position_scale)
    return float(2.0 * np.sum(quadrature_weights * jacobian * np.abs(field) ** 2))


def coherent_eta(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n_frequency: int = 96,
    n_spm_frequency: int = 48,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Eq. (3)-(5) 的相干 SPM/XPM 系数 ``eta``（1/W²）。

    与 :func:`network_nlo_eval.models.buglia2024_closed.nli_closed_form` 的区别：
    这里的跨间相位在**频率积分内部**，因此同跨项与跨间项（``P_FmP_Fn``、
    ``P_FmP_Bn``、``P_BmP_Bn``）都被完整体现，没有"每跨一个标量相位"的近似，
    也没有丢掉论文认为较小的那几项。

    Args:
        spans: 链路各跨（顺序即传播顺序）。
        reference_powers_w: ``(N,)`` ``eta`` 的参考功率（通常是第一跨入纤功率）。
        bandwidths_hz: ``(N,)`` 信道带宽。
        gamma_per_w_m: ``(N,)`` 非线性系数 (1/(W m))。
        beta2: GVD (s²/m)。
        beta3: 色散斜率 (s³/m)。
        reference_frequency_hz: ``beta2`` 参考频率 (Hz)。
        n_frequency: XPM 的 Gauss-Legendre 节点数。
        n_spm_frequency: SPM 每个频率方向的节点数。

    Returns
    -------
        ``(eta_spm, eta_xpm)``，均为 ``(N,)``，单位 1/W²。
    """
    frequencies = np.asarray(spans[0].profile.f_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    nodes, weights = np.polynomial.legendre.leggauss(int(n_frequency))
    spm_nodes, spm_weights = np.polynomial.legendre.leggauss(int(n_spm_frequency))
    eta_spm = np.zeros(n_channels, dtype=np.float64)
    eta_xpm = np.zeros(n_channels, dtype=np.float64)
    # 跨间相位用"每米"为单位；_band_average 的 phi 已经是 1/m，故 scale = 1。
    position_scale = 1.0

    for index in range(n_channels):
        if float(reference_powers_w[index]) <= 0.0:
            continue
        half_band = 0.5 * float(bandwidths[index])
        alpha_ref = float(spans[0].profile.alpha_npm[index])

        phi_spm = phase_spm(beta2, beta3, float(frequencies[index]), reference_frequency_hz)
        offsets = half_band * spm_nodes
        offset_weights = half_band * spm_weights
        spm_total = 0.0
        for offset_1, weight_1 in zip(offsets, offset_weights, strict=True):
            field = np.zeros(offsets.size, dtype=np.complex128)
            for span in spans:
                ratio = float(span.launch_power_w[index]) / max(float(reference_powers_w[index]), 1.0e-30)
                phase = phi_spm * offset_1 * offsets
                field += ratio * np.exp(1j * phase * span.start_m) * link_function(span.profile, index, phase)
            spm_total += weight_1 * float(np.sum(offset_weights * np.abs(field) ** 2))
        eta_spm[index] = (16.0 / 27.0) * float(gammas[index]) ** 2 / (2.0 * half_band) ** 2 * spm_total

        xpm_total = 0.0
        for interferer in range(n_channels):
            if interferer == index or float(reference_powers_w[interferer]) <= 0.0:
                continue
            phi_ik = phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            inner = _band_average(
                spans,
                reference_powers_w,
                interferer,
                phi_ik,
                half_band,
                position_scale,
                alpha_ref,
                nodes,
                weights,
            )
            xpm_total += (
                (float(reference_powers_w[interferer]) / float(reference_powers_w[index])) ** 2
                * inner
                / float(bandwidths[interferer])
            )
        eta_xpm[index] = (32.0 / 27.0) * float(gammas[index]) ** 2 * xpm_total
    return eta_spm, eta_xpm


# =============================================================================
# 同跨闭式项（CFM 的"闭式"部分）
# =============================================================================


def single_span_eta_closed(
    profile: ZhangSpanProfile,
    bandwidths_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """单跨 SPM/XPM 的闭式 ``eta``（Buglia Eq. (11)/(12) 同型推导）。

    Zhang 的 Eq. (1) 是三个指数之和，与 Buglia Eq. (2) 同构，因此 Eq. (9)-(12)
    的整套闭式可以直接搬过来：把 ``(T_i, T_f,i, T_b,i, alpha_l,i, kappa_f,i,
    kappa_b,i)`` 换成由 ``(T, A, D, d)`` 导出的对应量即可。这正是论文
    Sec. II-B "divide and conquer" 里同跨项的求法。

    这个函数只用于 CFM 的**同跨**项；跨间项没有公开表达式，由
    :func:`coherent_eta` 的积分给出。
    """
    from network_nlo_eval.models.buglia2024_closed import (
        eta_spm_closed,
        eta_xpm_closed,
    )
    from network_nlo_eval.models.buglia2024_closed import (
        phase_spm as buglia_phase_spm,
    )
    from network_nlo_eval.models.buglia2024_closed import (
        phase_xpm as buglia_phase_xpm,
    )

    converted = _to_buglia_profile(profile)
    frequencies = np.asarray(profile.f_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)
    for index in range(n_channels):
        launch = float(powers_w[index])
        if launch <= 0.0:
            continue
        phi_i = buglia_phase_spm(beta2, beta3, float(frequencies[index]), reference_frequency_hz)
        spm[index] = eta_spm_closed(converted, index, phi_i, float(bandwidths[index]), float(gammas[index]))
        total = 0.0
        for interferer in range(n_channels):
            if interferer == index or float(powers_w[interferer]) <= 0.0:
                continue
            phi_ik = buglia_phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            total += eta_xpm_closed(
                converted,
                interferer,
                phi_ik,
                float(bandwidths[index]),
                float(gammas[index]),
                (float(powers_w[interferer]) / launch) ** 2,
            )
        xpm[index] = total
    return spm, xpm


def _to_buglia_profile(profile: ZhangSpanProfile):
    r"""把 Zhang 的 ``(T, A, D, d)`` 映射成"极点形式"的闭合式载体。

    Zhang Eq. (2)（取统一的公共相位 ``e^{j phi L}``，见 :func:`link_function`）是

    .. math::

        A(\\phi) = \frac{1+\tilde T}{\alpha - j\\phi}
            - \frac{\tilde T}{A - j\\phi} + \frac{D}{d + j\\phi},

    也就是三个简单极点。把它写成 ``sum_k c_k/(r_k - j phi)``：
    ``r = (alpha, A, -d)``、``c = (1+T, -T, -D)``（``D/(d+j phi) = -D/(-d-j phi)``）。

    Buglia 的 Eq. (9)-(12) 对任意 ``(r_k, c_k)`` 都成立（把 ``kappa_f`` 取 0、
    ``kappa_b`` 取 1 就退化成纯极点之和），因此同跨闭式可以逐字复用。
    """
    from network_nlo_eval.models.buglia2024_closed import BugliaClosedProfile

    alpha = np.asarray(profile.alpha_npm, dtype=np.float64)
    a_value = np.asarray(profile.a_npm, dtype=np.float64)
    d_rate = np.asarray(profile.d_npm, dtype=np.float64)
    t_value = np.asarray(profile.t_coeff, dtype=np.float64)
    d_value = np.asarray(profile.d_coeff, dtype=np.float64)
    rates = np.stack([alpha, a_value, -d_rate], axis=1)
    coeffs = np.stack([1.0 + t_value, -t_value, -d_value], axis=1)
    return BugliaClosedProfile(
        alpha_npm=alpha,
        alpha_f_npm=np.full_like(alpha, _MIN_RATE_NPM),
        alpha_b_npm=np.full_like(alpha, _MIN_RATE_NPM),
        c_f=np.zeros_like(alpha),
        c_b=np.zeros_like(alpha),
        f_hz=np.asarray(profile.f_hz, dtype=np.float64),
        bandwidths_hz=np.asarray(profile.bandwidths_hz, dtype=np.float64),
        f_hat_hz=0.0,
        length_m=float(profile.length_m),
        rmse=np.asarray(profile.rmse, dtype=np.float64),
        pole_coeffs=coeffs,
        pole_rates=rates,
    )


# =============================================================================
# Sec. II-C：数据集标定
# =============================================================================


@dataclass(frozen=True)
class CalibrationTerm:
    """一个标定因子的 ``(a_i, b_i, c_i, d_i)``。

    ``f(Params) = prod_i 1/(a_i exp(b_i Param_i^{c_i} + d_i))``，对应论文
    Sec. II-C 的标定式。

    Attributes
    ----------
        name: 参数名，例如 ``"launch_power_dbm"``、``"spans"``。
        parameters: ``(a, b, c, d)`。
    """

    name: str
    parameters: tuple[float, float, float, float]

    def factor(self, value: float) -> float:
        """该参数对应的乘子 ``1/(a exp(b value^c + d))``。"""
        a, b, c, d = self.parameters
        safe = max(abs(float(value)), 1.0e-12) ** c
        return float(1.0 / (a * np.exp(b * safe + d)))


@dataclass(frozen=True)
class CalibrationSet:
    """一组标定因子（按项独立拟���）。

    Attributes
    ----------
        terms: 各参数的标定因子。
    """

    terms: tuple[CalibrationTerm, ...] = field(default_factory=tuple)

    def apply(self, values: dict[str, float]) -> float:
        """``f(Params)``：所有列出的参数乘子之积；缺参数时按 1 处理。"""
        factor = 1.0
        for term in self.terms:
            if term.name in values:
                factor *= term.factor(values[term.name])
        return float(factor)


def calibration_factor(terms: tuple[CalibrationTerm, ...], values: dict[str, float]) -> float:
    """便捷函数：``CalibrationSet(terms).apply(values)``。"""
    return CalibrationSet(terms).apply(values)


def cfm_and_egn_terms(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n_frequency: int = 96,
    n_spm_frequency: int = 48,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """同时给出 CFM 项与 integral-form EGN 项，供标定与误差报告使用。

    * ``eta_egn``：Eq. (4)/(5) 的相干积分（同跨 + 跨间，全部保留）；
    * ``eta_cfm``：同跨项用闭式（:func:`single_span_eta_closed`），跨间项仍用积分
      （论文的跨间闭式没有公开）。

    Returns
    -------
        ``(cfm_spm, cfm_xpm, egn_spm, egn_xpm)``，均为 ``(N,)`` 的 ``eta``。
    """
    egn_spm, egn_xpm = coherent_eta(
        spans,
        reference_powers_w,
        bandwidths_hz,
        gamma_per_w_m,
        beta2,
        beta3,
        reference_frequency_hz,
        n_frequency=n_frequency,
        n_spm_frequency=n_spm_frequency,
    )
    same_spm = np.zeros_like(egn_spm)
    same_xpm = np.zeros_like(egn_xpm)
    for span in spans:
        one_spm, one_xpm = coherent_eta(
            [span],
            reference_powers_w,
            bandwidths_hz,
            gamma_per_w_m,
            beta2,
            beta3,
            reference_frequency_hz,
            n_frequency=n_frequency,
            n_spm_frequency=n_spm_frequency,
        )
        same_spm += one_spm
        same_xpm += one_xpm
    closed_spm, closed_xpm = single_span_eta_closed(
        spans[0].profile,
        bandwidths_hz,
        reference_powers_w,
        gamma_per_w_m,
        beta2,
        beta3,
        reference_frequency_hz,
    )
    scale = sum(float(np.mean(span.launch_power_w / np.maximum(reference_powers_w, 1e-30))) ** 3 for span in spans)
    cfm_spm = closed_spm * scale + (egn_spm - same_spm)
    cfm_xpm = closed_xpm * scale + (egn_xpm - same_xpm)
    return cfm_spm, cfm_xpm, egn_spm, egn_xpm
