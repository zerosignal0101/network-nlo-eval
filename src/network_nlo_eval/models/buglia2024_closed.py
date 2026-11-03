r"""Buglia 2024 闭式 RA-GN 内核：Eq. (9)、(10)、(11)、(12) 的逐项实现。

**论文**

H. Buglia, M. Jarmolovičius, L. Galdino, R. I. Killey, P. Bayvel,
"A Closed-Form Expression for the Gaussian Noise Model in the Presence of
Raman Amplification", J. Lightwave Technol. 42(2), 636-648, 2024,
doi:10.1109/jlt.2023.3315127.

**本模块负责什么**

论文的 NLI 计算链是

1. 解耦合拉曼方程 Eq. (1)，得到每信道归一化功率剖面 ``rho(z, f_i)``；
2. 用 5 个系数把剖面重写成 Eq. (2) 的半解析形式（非线性最小二乘拟合）；
3. 把 Eq. (2) 代进链路函数 Eq. (8)，解析积分得到 Eq. (9)；
4. 把 Eq. (9) 代进 Eq. (6)/(7) 的频率积分，解析求积得到 XPM 的 Eq. (11)
   与 SPM 的 Eq. (12)；
5. 按 Eq. (4) 跨间累加、按 Eq. (3) 折算 ``SNR_NLI``。

本模块实现第 2–5 步（第 1 步由
:class:`~network_nlo_eval.models.raman_ode.RamanPaperSpanSolver` 完成）。

**公式对象索引**

======================  ====================================================
论文位置                代码位置
======================  ====================================================
Eq. (2) 5 参数剖面       :func:`fit_buglia_profile_5param`
Eq. (9) 链路函数闭式     :func:`link_mu_closed`
Eq. (10) 展开系数        :func:`_upsilon`
Eq. (11) XPM 闭式        :func:`eta_xpm_closed`
Eq. (12) SPM 闭式        :func:`eta_spm_closed`
Eq. (3)/(4) NLI 与跨累加 :func:`nli_closed_form`
Eq. (6)-(8) 积分基准     :func:`nli_integral`
======================  ====================================================

**与提取文本的差异（逐条）**

* Eq. (12) 最后一行，MinerU 提取为
  ``sign(-phi_i) e^{-|alpha_l L|} sign(phi_i) e^{-|alpha'_l L|}``，中间漏了一个
  加号，且与 Appendix D 的 Eq. (56)
  ``e^{-|a L|} sign(-c) + e^{-|b L|} sign(c)`` 对不上。本实现按 Eq. (56)
  取 ``sign(-phi_i) e^{-|alpha_l L|} + sign(phi_i) e^{-|alpha'_l L|}``，
  并由 :func:`spm_closed_matches_direct_integral` 的数值对照确认。
* Eq. (11)、(12) 中 ``sign(alpha_l/phi)`` 与 Appendix C/D 的
  ``sign(phi/alpha_l)`` 只差符号次序，二者恒等，本实现写 ``sign(phi/alpha)``。
* 论文的相位失配用 baseband 频率（相对 ``beta2`` 的定义频率）。本实现显式
  传入 ``reference_frequency_hz``，即
  ``phi_i = -4 pi^2 [beta2 + 2 pi beta3 (f_i - f_ref)]``、
  ``phi_ik = -4 pi^2 (f_k - f_i) [beta2 + pi beta3 (f_i + f_k - 2 f_ref)]``，
  与 Eq. (34)/(42) 的写法一致。

**已知近似（论文自身也有的）**

* Eq. (6) 的矩形函数 ``Pi((f1+f2)/B_k)`` 在闭式推导里被丢掉（Appendix C 明示，
  等价于把积分域近似成矩形）。
* Eq. (37)/(38)、(45)/(46) 解析解不存在，推导取 ``B_i -> inf``。
* SPM 的 ``(45)/(46)`` 用双曲坐标 + 对数在驻点取值，误差约 1–2%
  （本仓数值对照见 ``tests/test_buglia2024_closed.py``）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat

__all__ = [
    "BugliaClosedProfile",
    "eta_spm_closed",
    "eta_xpm_closed",
    "fit_buglia_profile_5param",
    "link_mu_closed",
    "nli_closed_form",
    "nli_integral",
    "phase_spm",
    "phase_xpm",
]

#: ``(l1, l2)`` 求和指标：Eq. (9)-(12) 的和式是 ``0 <= l1 + l2 <= 1``。
_SUM_TERMS: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1))

#: 拟合系数的数值下界/上界（1/m）。上限 5e-3 1/m ≈ 21.7 dB/km，覆盖到 S 波段。
_MIN_RATE_NPM: float = 1.0e-7
_MAX_RATE_NPM: float = 5.0e-3

#: ``phi`` 小于该值时闭式解退化（``1/phi`` 发散、``sign(phi/alpha)`` 抖动），
#: 改用积分基准。SPM 的 ``phi_i`` 本身只有 1e-24 量级，门槛必须远低于它。
_PHI_FLOOR: float = 1.0e-35

#: ``alpha_i`` 的拟合区间（相对光纤衰减）。论文只说 ``alpha_i`` 建模光纤损耗，
#: 未给区间；不加上界时最小二乘会把衰减全部吸收进 ISRS 项（``alpha_i -> 0``），
#: 而 Eq. (9)/(11)/(12) 的 ``B_i -> inf`` 延拓要求 ``alpha_l L`` 足够大，
#: 该解会让闭式与积分基准偏离 4 dB 以上。0.5–2 倍覆盖了拟合需要的自由度。
_ALPHA_BOUNDS: tuple[float, float] = (0.5, 2.0)


# =============================================================================
# Eq. (2)：5 参数半解析剖面
# =============================================================================


@dataclass(frozen=True)
class BugliaClosedProfile:
    r"""Buglia Eq. (2) 的 5 参数拟合结果（逐信道）。

    .. math::

        \rho(z, f_i) = e^{-\alpha_i z}
            \left[1 - (C_{f,i} P_f L_{\mathrm{eff}}(z)
            + C_{b,i} P_b \tilde L_{\mathrm{eff}}(z))(f_i - \hat f)\right]

    其中 ``c_f = C_{f,i} P_f``、``c_b = C_{b,i} P_b``。论文把 5 个系数一起
    交给非线性最小二乘；本实现的 :class:`BugliaClosedProfile` 同样保存 5 个
    逐信道值，不再把 ``alpha_f``/``alpha_b`` 固定成泵浦平均衰减。

    Attributes
    ----------
        alpha_npm: ``(N,)`` 等效衰减 ``alpha_i`` (1/m)。
        alpha_f_npm: ``(N,)`` 前向等效衰减 ``alpha_f,i`` (1/m)。
        alpha_b_npm: ``(N,)`` 后向等效衰减 ``alpha_b,i`` (1/m)。
        c_f: ``(N,)`` 拟合的 ``C_f,i P_f`` (1/(m*Hz))。
        c_b: ``(N,)`` 拟合的 ``C_b,i P_b`` (1/(m*Hz))。
        f_hz: ``(N,)`` 拟合所用信道频率 (Hz)，降序。
        bandwidths_hz: ``(N,)`` 信道带宽 (Hz)，Eq. (11) 的 ``B_k``。
        f_hat_hz: 泵浦平均频率 ``f_hat`` (Hz)。
        length_m: 跨长 (m)。
        rmse: ``(N,)`` 剖面拟合的相对 RMSE（无量纲）。
        forward_kappa_infinite: ``True`` 时把 Eq. (9) 的 ``kappa_f`` 取为 1，
            即丢弃 ``e^{-alpha L}``。Zhang 2024 Eq. (2) 用的正是这个长跨/高损耗
            近似（``int_0^L e^{-alpha z}`` 写成 ``1/(alpha - j phi)``），
            适配器 :func:`~network_nlo_eval.models.zhang2024_coherent._to_buglia_profile`
            会把该标志置为 ``True``，让同跨闭式与 Zhang 的 Eq. (2) 口径一致。
    """

    alpha_npm: NDArrayFloat
    alpha_f_npm: NDArrayFloat
    alpha_b_npm: NDArrayFloat
    c_f: NDArrayFloat
    c_b: NDArrayFloat
    f_hz: NDArrayFloat
    bandwidths_hz: NDArrayFloat
    f_hat_hz: float
    length_m: float
    rmse: NDArrayFloat
    forward_kappa_infinite: bool = False
    #: 极点形式的替代剖面：``A(phi) = sum_k pole_coeffs[:, k]/(pole_rates[:, k] - j phi)``。
    #: Zhang 2024 Eq. (2) 的链路函数正是这个形状（三项简单极点，没有
    #: ``kappa_f e^{j phi L}`` 因子），用它可以让同跨闭式与 Eq. (2) 完全一致。
    pole_coeffs: NDArrayFloat | None = None
    pole_rates: NDArrayFloat | None = None

    @property
    def n_channels(self) -> int:
        """信道数。"""
        return int(np.asarray(self.f_hz).size)

    def bandwidth_hz(self, index: int) -> float:
        """第 ``index`` 个信道的带宽 ``B_k`` (Hz)。"""
        return float(np.asarray(self.bandwidths_hz)[index])

    def delta_hz(self, index: int) -> float:
        """``f_i - f_hat`` (Hz)。"""
        return float(self.f_hz[index]) - float(self.f_hat_hz)

    def value(self, z_m: NDArrayFloat, index: int) -> NDArrayFloat:
        """在给定纵向位置求 ``rho(z, f_i)``。"""
        z = np.asarray(z_m, dtype=np.float64)
        alpha = float(self.alpha_npm[index])
        alpha_f = float(self.alpha_f_npm[index])
        alpha_b = float(self.alpha_b_npm[index])
        leff = -np.expm1(-alpha_f * z) / alpha_f
        ltilde = (np.exp(-alpha_b * (self.length_m - z)) - np.exp(-alpha_b * self.length_m)) / alpha_b
        delta = self.delta_hz(index)
        return np.exp(-alpha * z) * (1.0 - (float(self.c_f[index]) * leff + float(self.c_b[index]) * ltilde) * delta)


def _profile_from_parameters(
    parameters: NDArrayFloat,
    z_m: NDArrayFloat,
    length_m: float,
    delta_hz: float,
) -> NDArrayFloat:
    """由 ``(alpha, alpha_f, alpha_b, c_f, c_b)`` 求 Eq. (2) 剖面。"""
    alpha, alpha_f, alpha_b, c_f, c_b = parameters
    leff = -np.expm1(-alpha_f * z_m) / alpha_f
    ltilde = (np.exp(-alpha_b * (length_m - z_m)) - np.exp(-alpha_b * length_m)) / alpha_b
    return np.exp(-alpha * z_m) * (1.0 - (c_f * leff + c_b * ltilde) * delta_hz)


def fit_buglia_profile_5param(
    profile_norm: NDArrayFloat,
    length_m: float,
    frequencies_hz: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    f_hat_hz: float,
    n_steps: int,
) -> BugliaClosedProfile:
    """按论文 Sec. II-A 的流程拟合 Eq. (2) 的 5 个系数。

    论文的做法是：数值求解 Eq. (1) 得到剖面，再用非线性最小二乘把
    ``(alpha_i, C_f,i, C_b,i, alpha_f,i, alpha_b,i)`` 一起拟合到该剖面上。
    这里完全照做，逐信道独立拟合。

    实现上先用一组 ``(alpha, alpha_f, alpha_b)`` 起点，对每个起点把
    ``(c_f, c_b)`` 用线性最小二乘解出（Eq. (2) 对这两个系数是线性的），选出
    最好的三个起点再做 5 参数联合精修。这样避免了 5 维直接优化在
    ``c ~ 1e-16``、``alpha_f ~ 1e-4`` 这种跨 12 个数量级的参数上走不动。

    Args:
        profile_norm: ``(2*n_steps, N)`` 归一化剖面 ``P_i(z)/P_i(0)``，降序。
        length_m: 跨长 (m)。
        frequencies_hz: ``(N,)`` 信道频率 (Hz)，与 ``profile_norm`` 同序。
        bandwidths_hz: ``(N,)`` 信道带宽 (Hz)，与 ``profile_norm`` 同序。
        alpha_npm: ``(N,)`` 光纤衰减 (1/m)，作为 ``alpha_i`` 的初值。
        f_hat_hz: 泵浦平均频率 (Hz)；无泵时取信道功率加权平均。
        n_steps: 生成剖面所用的 RK4 步数。

    Returns
    -------
        :class:`BugliaClosedProfile`，含逐信道相对 RMSE。

    Notes
    -----
    残差用相对误差，避免深衰落尾部（功率已降到 -20 dB 以下）主导目标函数；
    论文 Fig. 3 的"最差信道"判据也是看整条曲线而不是绝对误差。
    """
    z = _profile_grid(profile_norm, length_m, n_steps)
    # z=0 处 Eq. (2) 恒等于 1，把它作为额外约束点可以锚定 alpha_f/alpha_b。
    z_fit = np.concatenate(([0.0], z))
    n_channels = int(np.asarray(frequencies_hz).size)
    alpha_fit = np.empty(n_channels, dtype=np.float64)
    alpha_f_fit = np.empty(n_channels, dtype=np.float64)
    alpha_b_fit = np.empty(n_channels, dtype=np.float64)
    c_f = np.empty(n_channels, dtype=np.float64)
    c_b = np.empty(n_channels, dtype=np.float64)
    rmse = np.empty(n_channels, dtype=np.float64)
    floor = 0.05

    scale = np.array([1.0e-5, 1.0e-5, 1.0e-5, 1.0e-18, 1.0e-18])

    for index in range(n_channels):
        alpha_lo = _ALPHA_BOUNDS[0] * float(alpha_npm[index])
        alpha_hi = _ALPHA_BOUNDS[1] * float(alpha_npm[index])
        lower = np.array([alpha_lo, _MIN_RATE_NPM, _MIN_RATE_NPM, -1.0e-9, -1.0e-9])
        upper = np.array([alpha_hi, _MAX_RATE_NPM, _MAX_RATE_NPM, 1.0e-9, 1.0e-9])
        target = np.concatenate(([1.0], profile_norm[:, index]))
        weight = 1.0 / np.maximum(target, floor)
        delta = float(frequencies_hz[index]) - float(f_hat_hz)

        def residual(parameters: NDArrayFloat, target=target, weight=weight, delta=delta) -> NDArrayFloat:
            model = _profile_from_parameters(parameters, z_fit, length_m, delta)
            return (model - target) * weight

        candidates = _linear_starts(z_fit, target, weight, length_m, delta, float(alpha_npm[index]))
        best: tuple[float, NDArrayFloat] | None = None
        for start in candidates[:3]:
            try:
                solved = least_squares(
                    residual,
                    np.clip(start, lower + 1.0e-12, upper - 1.0e-12),
                    bounds=(lower, upper),
                    x_scale=scale,
                    max_nfev=4000,
                    xtol=1.0e-12,
                    ftol=1.0e-12,
                    gtol=1.0e-12,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            cost = float(np.sqrt(np.mean(residual(solved.x) ** 2)))
            if best is None or cost < best[0]:
                best = (cost, solved.x)

        if best is None:
            parameters = np.array([alpha_npm[index], alpha_npm[index], alpha_npm[index], 0.0, 0.0])
            cost = float(np.sqrt(np.mean(residual(parameters) ** 2)))
        else:
            cost, parameters = best

        alpha_fit[index], alpha_f_fit[index], alpha_b_fit[index], c_f[index], c_b[index] = (
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            float(parameters[3]),
            float(parameters[4]),
        )
        rmse[index] = cost

    return BugliaClosedProfile(
        alpha_npm=alpha_fit,
        alpha_f_npm=alpha_f_fit,
        alpha_b_npm=alpha_b_fit,
        c_f=c_f,
        c_b=c_b,
        f_hz=np.asarray(frequencies_hz, dtype=np.float64),
        bandwidths_hz=np.asarray(bandwidths_hz, dtype=np.float64),
        f_hat_hz=float(f_hat_hz),
        length_m=float(length_m),
        rmse=rmse,
    )


def _linear_starts(
    z_fit: NDArrayFloat,
    target: NDArrayFloat,
    weight: NDArrayFloat,
    length_m: float,
    delta_hz: float,
    alpha_npm: float,
) -> list[NDArrayFloat]:
    """对 ``(alpha, alpha_f, alpha_b)`` 的网格逐点线性解 ``(c_f, c_b)``。

    Eq. (2) 对 ``c_f``、``c_b`` 是线性的，固定三个衰减率后可以直接最小二乘
    求出两个斜率系数。返回按代价升序排列的 5 参数起点。
    """
    rates = np.array([alpha_npm, 2.0e-4, 5.0e-4, 1.0e-3])
    decays = np.array([3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3])
    scored: list[tuple[float, NDArrayFloat]] = []
    for alpha in rates:
        base = np.exp(-alpha * z_fit)
        for alpha_f in decays:
            leff = -np.expm1(-alpha_f * z_fit) / alpha_f
            for alpha_b in decays:
                ltilde = (np.exp(-alpha_b * (length_m - z_fit)) - np.exp(-alpha_b * length_m)) / alpha_b
                basis = -base[:, None] * delta_hz * np.stack([leff, ltilde], axis=1) * weight[:, None]
                rhs = (target - base) * weight
                try:
                    coefficients, *_ = np.linalg.lstsq(basis, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(coefficients)):
                    continue
                model = base * (1.0 - (coefficients[0] * leff + coefficients[1] * ltilde) * delta_hz)
                cost = float(np.sqrt(np.mean(((model - target) * weight) ** 2)))
                scored.append((cost, np.array([alpha, alpha_f, alpha_b, coefficients[0], coefficients[1]])))
    scored.sort(key=lambda item: item[0])
    if not scored:
        return [np.array([alpha_npm, alpha_npm, alpha_npm, 0.0, 0.0])]
    return [parameters for _, parameters in scored[:3]]


def _profile_grid(profile_norm: NDArrayFloat, length_m: float, n_steps: int) -> NDArrayFloat:
    """RK4 剖面的纵向采样点：每步的中点与终点。"""
    step = length_m / n_steps
    positions = np.empty(2 * n_steps, dtype=np.float64)
    for index in range(n_steps):
        positions[2 * index] = step * (index + 0.5)
        positions[2 * index + 1] = step * (index + 1.0)
    return positions


def _initial_guesses(
    profile: NDArrayFloat,
    length_m: float,
    alpha_npm: float,
    delta_hz: float,
) -> list[NDArrayFloat]:
    """给 5 参数拟合一组起点：纯指数、快速起落、慢速起落。"""
    alpha_tail = -np.log(max(float(profile[-1]), 1.0e-6)) / length_m
    alpha_tail = float(np.clip(alpha_tail, _MIN_RATE_NPM, _MAX_RATE_NPM))
    sign = -1.0 if delta_hz < 0.0 else 1.0
    guesses = [
        np.array([alpha_npm, alpha_npm, alpha_npm, 0.0, 0.0]),
        np.array([alpha_tail, 2.0 * alpha_npm, 2.0 * alpha_npm, 0.0, 0.0]),
        np.array([alpha_npm, 5.0 * alpha_npm, 2.0 * alpha_npm, sign * 2.0e-19, 0.0]),
        np.array([alpha_npm, 2.0 * alpha_npm, 5.0 * alpha_npm, 0.0, sign * 2.0e-19]),
        np.array([alpha_tail, alpha_npm, alpha_npm, sign * 1.0e-18, -sign * 1.0e-19]),
    ]
    return guesses


# =============================================================================
# Eq. (9)-(12)：闭式链路函数与 NLI 贡献
# =============================================================================


def _term_parameters(profile: BugliaClosedProfile, index: int, l1: int, l2: int) -> tuple[float, ...]:
    """返回某个 ``(l1, l2)`` 项的 ``(alpha_l, kappa_f, kappa_b, Upsilon)``。

    极点模式下（``pole_coeffs`` 非空）把三个 ``(l1, l2)`` 依次映射到三个极点，
    并令 ``kappa_f = 0``、``kappa_b = 1``：这样 Eq. (11)/(12) 的 cos/sin 项系数
    ``(kappa_f kappa_b' +- kappa_b kappa_f')`` 全为零，只剩主项，恰好等于
    ``|sum_k c_k/(r_k - j phi)|^2`` 的解析积分。
    """
    if profile.pole_coeffs is not None and profile.pole_rates is not None:
        slot = _SUM_TERMS.index((l1, l2))
        rates = np.asarray(profile.pole_rates, dtype=np.float64)
        coeffs = np.asarray(profile.pole_coeffs, dtype=np.float64)
        return float(rates[index, slot]), 0.0, 1.0, float(coeffs[index, slot])
    alpha = float(profile.alpha_npm[index])
    alpha_f = float(profile.alpha_f_npm[index])
    alpha_b = float(profile.alpha_b_npm[index])
    length = float(profile.length_m)
    delta = profile.delta_hz(index)
    t_f = -float(profile.c_f[index]) * delta / alpha_f
    t_b = -float(profile.c_b[index]) * delta / alpha_b
    t_0 = 1.0 + t_f - t_b * np.exp(-alpha_b * length)
    alpha_l = alpha + l1 * alpha_f - l2 * alpha_b
    if profile.forward_kappa_infinite:
        kappa_f = 1.0
    else:
        kappa_f = np.exp(-(alpha + l1 * alpha_f) * length)
    kappa_b = np.exp(-l2 * alpha_b * length)
    upsilon = t_0 * (-t_f / t_0) ** l1 * (t_b / t_0) ** l2 if abs(t_0) > 0.0 else 0.0
    return alpha_l, kappa_f, kappa_b, upsilon


def link_mu_closed(profile: BugliaClosedProfile, index: int, phi: NDArrayFloat) -> NDArrayFloat:
    r"""Buglia Eq. (9)：链路函数 ``mu(phi)`` 的闭式表达。

    对 ``(l1, l2)`` 与 ``(l1', l2')`` 双求和：

    .. math::

        \mu \approx \sum_{l,l'}\Upsilon\Upsilon'\left[
            \frac{(\kappa_f\kappa_f' + \kappa_b\kappa_b')(\alpha_l\alpha_l' + \phi^2)}
                 {(\alpha_l^2+\phi^2)(\alpha_l'^2+\phi^2)}
          - \frac{(\kappa_f\kappa_b' + \kappa_b\kappa_f')(\alpha_l\alpha_l' + \phi^2)}
                 {(\alpha_l^2+\phi^2)(\alpha_l'^2+\phi^2)}\cos(\phi L)
          + \frac{(\kappa_f\kappa_b' - \kappa_b\kappa_f')(\alpha_l-\alpha_l')\phi}
                 {(\alpha_l^2+\phi^2)(\alpha_l'^2+\phi^2)}\sin(\phi L)\right].

    Args:
        profile: 5 参数剖面。
        index: 信道下标。
        phi: ``(M,)`` 相位失配斜率 (1/m)。

    Returns
    -------
        ``(M,)`` 无量纲的 ``mu``。
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    length = float(profile.length_m)
    total = np.zeros_like(phi)
    for l1, l2 in _SUM_TERMS:
        a1, kf1, kb1, u1 = _term_parameters(profile, index, l1, l2)
        for l1p, l2p in _SUM_TERMS:
            a2, kf2, kb2, u2 = _term_parameters(profile, index, l1p, l2p)
            denominator = (a1**2 + phi**2) * (a2**2 + phi**2)
            shared = (a1 * a2 + phi**2) / denominator
            main = (kf1 * kf2 + kb1 * kb2) * shared
            cross = (kf1 * kb2 + kb1 * kf2) * shared * np.cos(phi * length)
            quadrature = (kf1 * kb2 - kb1 * kf2) * (a1 - a2) * phi / denominator * np.sin(phi * length)
            total += u1 * u2 * (main - cross - quadrature)
    return total


def eta_xpm_closed(
    profile: BugliaClosedProfile,
    interferer: int,
    phase_slope: float,
    coi_bandwidth_hz: float,
    gamma_per_w_m: float,
    power_ratio_squared: float,
) -> float:
    r"""Buglia Eq. (11)：单个干扰信道的 XPM 贡献 ``eta_XPM^(k)(f_i)``。

    .. math::

        \eta_{\mathrm{XPM}}^{(k)} = \frac{32}{27}\frac{\gamma^2}{B_k}
            \left(\frac{P_k}{P_i}\right)^2
            \sum_{l,l'}\Upsilon_k\Upsilon_k'
            \frac{1}{\phi_{i,k}(\alpha_{l,k}+\alpha'_{l,k})}\{\cdots\}

    大括号内的三项分别是主项（``atan``）、``cos`` 项与 ``sin`` 项，逐项对应
    Appendix C 的 Eq. (39)、(40)、(41)。

    Args:
        profile: 5 参数剖面。
        interferer: 干扰信道下标 ``k``。
        phase_slope: ``phi_{i,k}`` (1/m)，见 :func:`phase_xpm`。
        coi_bandwidth_hz: ``B_i`` (Hz)，出现在 ``atan(phi B_i / 2 alpha)``。
        gamma_per_w_m: 非线性系数 (1/(W m))。
        power_ratio_squared: ``(P_k / P_i)^2``。

    Returns
    -------
        ``eta_XPM^(k)`` (1/W²)。

    Notes
    -----
    ``B_k`` 只出现在 ``1/B_k`` 的前因子中（Eq. (6) 的 ``1/B_k^2`` 与 Eq. (39)
    的 ``2 B_k`` 相消），因此这里用干扰信道自身的带宽。
    """
    phi = float(phase_slope)
    if abs(phi) < _PHI_FLOOR:
        raise ValueError("phase_slope is too small for the closed form; use the integral baseline.")
    bandwidth_k = float(profile.bandwidth_hz(interferer)) if hasattr(profile, "bandwidth_hz") else None
    if bandwidth_k is None:
        raise AttributeError("BugliaClosedProfile must carry the interferer bandwidth.")
    length = float(profile.length_m)
    total = 0.0
    for l1, l2 in _SUM_TERMS:
        a1, kf1, kb1, u1 = _term_parameters(profile, interferer, l1, l2)
        for l1p, l2p in _SUM_TERMS:
            a2, kf2, kb2, u2 = _term_parameters(profile, interferer, l1p, l2p)
            prefactor = u1 * u2 / (phi * (a1 + a2))
            main = (
                2.0
                * (kf1 * kf2 + kb1 * kb2)
                * (np.arctan(phi * coi_bandwidth_hz / (2.0 * a1)) + np.arctan(phi * coi_bandwidth_hz / (2.0 * a2)))
            )
            cross_cos = np.pi * (
                -(kf1 * kb2 + kb1 * kf2)
                * (np.sign(phi / a1) * np.exp(-abs(a1 * length)) + np.sign(phi / a2) * np.exp(-abs(a2 * length)))
                - (kf1 * kb2 - kb1 * kf2)
                * (np.sign(-phi) * np.exp(-abs(a1 * length)) + np.sign(phi) * np.exp(-abs(a2 * length)))
            )
            total += prefactor * (main + cross_cos)
    return float((32.0 / 27.0) * gamma_per_w_m**2 / bandwidth_k * power_ratio_squared * total)


def eta_spm_closed(
    profile: BugliaClosedProfile,
    index: int,
    phase_factor: float,
    bandwidth_hz: float,
    gamma_per_w_m: float,
) -> float:
    r"""Buglia Eq. (12)：SPM 贡献 ``eta_SPM(f_i)``。

    .. math::

        \eta_{\mathrm{SPM}} = \frac{16}{27}\frac{\gamma^2}{B_i^2}
            \sum_{l,l'}\Upsilon_i\Upsilon_i'
            \frac{\pi}{\phi_i(\alpha_{l,i}+\alpha'_{l,i})}\{\cdots\}

    大括号内是 ``asinh`` 主项与两个 ``ln`` 修正项，对应 Appendix D 的
    Eq. (50)、(55)、(56)。Eq. (12) 印刷版最后一项漏掉的加号按 Eq. (56) 补齐，
    见模块文档。

    Args:
        profile: 5 参数剖面。
        index: 信道下标。
        phase_factor: ``phi_i`` (1/m)，见 :func:`phase_spm`。
        bandwidth_hz: ``B_i`` (Hz)。
        gamma_per_w_m: 非线性系数 (1/(W m))。

    Returns
    -------
        ``eta_SPM`` (1/W²)。
    """
    phi = float(phase_factor)
    if abs(phi) < _PHI_FLOOR:
        raise ValueError("phase_factor is too small for the closed form; use the integral baseline.")
    length = float(profile.length_m)
    log_argument = float(np.log(np.sqrt(phi * length / (2.0 * np.pi)) * bandwidth_hz))
    total = 0.0
    for l1, l2 in _SUM_TERMS:
        a1, kf1, kb1, u1 = _term_parameters(profile, index, l1, l2)
        for l1p, l2p in _SUM_TERMS:
            a2, kf2, kb2, u2 = _term_parameters(profile, index, l1p, l2p)
            prefactor = u1 * u2 * np.pi / (phi * (a1 + a2))
            main = (
                2.0
                * (kf1 * kf2 + kb1 * kb2)
                * (
                    np.arcsinh(3.0 * phi * bandwidth_hz**2 / (8.0 * np.pi * a1))
                    + np.arcsinh(3.0 * phi * bandwidth_hz**2 / (8.0 * np.pi * a2))
                )
            )
            correction = (
                4.0
                * log_argument
                * (
                    -(kf1 * kb2 + kb1 * kf2)
                    * (np.sign(phi / a1) * np.exp(-abs(a1 * length)) + np.sign(phi / a2) * np.exp(-abs(a2 * length)))
                    - (kf1 * kb2 - kb1 * kf2)
                    * (np.sign(-phi) * np.exp(-abs(a1 * length)) + np.sign(phi) * np.exp(-abs(a2 * length)))
                )
            )
            total += prefactor * (main + correction)
    return float((16.0 / 27.0) * gamma_per_w_m**2 / bandwidth_hz**2 * total)


# =============================================================================
# 相位失配（Eq. 34 / 42）
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
# Eq. (3)/(4)：逐跨 NLI 与跨累加
# =============================================================================


def nli_closed_form(
    profile: BugliaClosedProfile,
    bandwidths_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    spm_coherence_exponent: float = 0.0,
    n_spans: int = 1,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    r"""用 Eq. (11)/(12) 逐跨求 SPM 与 XPM 噪声功率。

    Eq. (4) 的累加口径在这里显式给出：

    .. math::

        \\eta_n(f_i) \approx \\sum_{j=1}^{n}\\left[\frac{P_{i,j}}{P_i}\right]^2
            \\left[\\eta_{\\mathrm{SPM},j}\\,n^{\\epsilon} + \\eta_{\\mathrm{XPM},j}\right].

    路由层按跨逐次调用本函数，因此 ``P_{i,j}/P_i`` 由各跨的入纤功率承担；
    ``n^epsilon`` 由参数 ``spm_coherence_exponent`` 与 ``n_spans`` 给出。
    论文没有印出 ``epsilon`` 的取值（只引用 [16, Eq. 22]），默认 0 即与仓库
    其余模型一致的功率相加；要复现 Fig. 4 的跨数缩放可显式传
    ``spm_coherence_exponent=1``。

    Args:
        profile: 5 参数剖面。
        bandwidths_hz: ``(N,)`` 信道带宽 (Hz)。
        powers_w: ``(N,)`` 本跨入纤功率 (W)。
        gamma_per_w_m: ``(N,)`` 逐信道非线性系数 (1/(W m))。
        beta2: 参考频率处的 GVD (s²/m)。
        beta3: 色散斜率 (s³/m)。
        reference_frequency_hz: ``beta2`` 的参考频率 (Hz)。
        spm_coherence_exponent: Eq. (4) 的 ``epsilon``。
        n_spans: 累计跨数 ``n``。

    Returns
    -------
        ``(spm_w, xpm_w)``，均为 ``(N,)`` 噪声功率 (W)。
    """
    frequencies = np.asarray(profile.f_hz, dtype=np.float64)
    powers = np.asarray(powers_w, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    span_factor = float(n_spans) ** float(spm_coherence_exponent)
    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)

    for index in range(n_channels):
        launch = float(powers[index])
        if launch <= 0.0:
            continue
        phase_i = phase_spm(beta2, beta3, float(frequencies[index]), reference_frequency_hz)
        try:
            eta_spm = eta_spm_closed(profile, index, phase_i, float(bandwidths[index]), float(gammas[index]))
        except ValueError:
            eta_spm = 0.0
        spm[index] = eta_spm * span_factor * launch**3

        eta_xpm = 0.0
        for interferer in range(n_channels):
            if interferer == index or powers[interferer] <= 0.0:
                continue
            phase_ik = phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            try:
                contribution = eta_xpm_closed(
                    profile,
                    interferer,
                    phase_ik,
                    float(bandwidths[index]),
                    float(gammas[index]),
                    (float(powers[interferer]) / launch) ** 2,
                )
            except ValueError:
                contribution = 0.0
            eta_xpm += contribution
        xpm[index] = eta_xpm * launch**3
    return spm, xpm


# =============================================================================
# 积分基准（Eq. 6-8）：用来校验闭式，不作为模型默认路径
# =============================================================================


def _band_average_link_squared(
    profile: BugliaClosedProfile,
    index: int,
    phase_slope: float,
    half_band_hz: float,
    alpha_reference_npm: float,
    nodes: NDArrayFloat,
    weights: NDArrayFloat,
) -> float:
    """``int_{-U}^{U} |link(phase_slope * v)|^2 dv``，用 ``tan`` 变换分辨相干峰。

    XPM 的相干峰宽度是 ``alpha / |phi|``，在 50 GBd 信道上只有百 MHz 量级，
    直接 Gauss-Legendre 求积会整段漏掉。这里令
    ``v = (alpha / |phi|) tan(s)``，把区间 ``[-U, U]`` 映射到
    ``s in [-atan(|phi| U / alpha), +...]``，峰始终落在节点密集处。
    """
    if abs(phase_slope) * half_band_hz <= 1.0e-12 * max(alpha_reference_npm, _MIN_RATE_NPM):
        link_zero = _link_exact(profile, index, np.zeros(1))
        return float(2.0 * half_band_hz * abs(link_zero[0]) ** 2)
    tangent = abs(phase_slope) * half_band_hz / max(alpha_reference_npm, _MIN_RATE_NPM)
    # |link|^2 关于 v 偶对称，只在 v >= 0 上积分再乘 2。
    angles = 0.5 * np.arctan(tangent) * (nodes + 1.0)
    quadrature_weights = 0.5 * np.arctan(tangent) * weights
    values = (alpha_reference_npm / phase_slope) * np.tan(angles)
    jacobian = alpha_reference_npm / abs(phase_slope) / np.cos(angles) ** 2
    link = _link_exact(profile, index, phase_slope * values)
    return float(2.0 * np.sum(quadrature_weights * jacobian * np.abs(link) ** 2))


def _link_exact(profile: BugliaClosedProfile, index: int, phi: NDArrayFloat) -> NDArrayComplex:
    """``int_0^L rho(z, f_index) exp(1j phi z) dz`` 的解析值。

    Eq. (2) 的剖面对固定信道是三个指数之和，纵向积分因此是闭式的；
    这里用它来保证积分基准与闭式只在频率方向的近似上不同。
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=np.complex128))
    if profile.pole_coeffs is not None and profile.pole_rates is not None:
        rates = np.asarray(profile.pole_rates, dtype=np.float64)[index]
        coeffs = np.asarray(profile.pole_coeffs, dtype=np.float64)[index]
        return np.sum(coeffs[:, None] / (rates[:, None] - 1j * phi[None, :]), axis=0)
    alpha = float(profile.alpha_npm[index])
    alpha_f = float(profile.alpha_f_npm[index])
    alpha_b = float(profile.alpha_b_npm[index])
    delta = profile.delta_hz(index)
    t_f = -float(profile.c_f[index]) * delta / alpha_f
    t_b = -float(profile.c_b[index]) * delta / alpha_b
    t_0 = 1.0 + t_f - t_b * np.exp(-alpha_b * float(profile.length_m))
    rates = np.array([alpha, alpha + alpha_f, alpha - alpha_b], dtype=np.float64)
    coefficients = np.array([t_0, -t_f, t_b * np.exp(-alpha_b * float(profile.length_m))], dtype=np.float64)
    denominator = rates[:, None] - 1j * phi[None, :]
    numerator = -np.expm1(-denominator * float(profile.length_m))
    return np.sum(coefficients[:, None] * numerator / denominator, axis=0)


def nli_integral(
    profile: BugliaClosedProfile,
    bandwidths_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n_frequency: int = 96,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """直接对 Eq. (6)-(8) 做数值求积，作为闭式的独立尺子。

    与 :func:`nli_closed_form` 用同一个 Eq. (2) 剖面、同一套相位定义，因此
    两者的差异只来自闭式推导本身的近似（矩形积分域、``B_i -> inf``、
    SPM 的驻点对数）。

    积分口径与 Appendix B/C/D 的推导前提一致：信号剖面在信道带宽内视为常数，
    因此 Eq. (8) 里的四个 ``rho`` 都取所在信道中心的值。XPM 用式 (34) 的
    ``f2 + Delta f ~ Delta f`` 化简，于是 Eq. (6) 的二重积分退化为
    ``B_k * int_{-B_i/2}^{B_i/2} |mu|^2 df1``；SPM 保持二重积分。

    Args:
        profile: 5 参数剖面。
        bandwidths_hz: ``(N,)`` 信道带宽 (Hz)。
        powers_w: ``(N,)`` 入纤功率 (W)。
        gamma_per_w_m: ``(N,)`` 逐信道非线性系数 (1/(W m))。
        beta2: GVD (s²/m)。
        beta3: 色散斜率 (s³/m)。
        reference_frequency_hz: ``beta2`` 参考频率 (Hz)。
        n_frequency: 每个频率方向的 Gauss-Legendre 节点数。

    Returns
    -------
        ``(spm_w, xpm_w)``，均为 ``(N,)``。
    """
    frequencies = np.asarray(profile.f_hz, dtype=np.float64)
    powers = np.asarray(powers_w, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    nodes, weights = np.polynomial.legendre.leggauss(int(n_frequency))
    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)

    for index in range(n_channels):
        launch = float(powers[index])
        if launch <= 0.0:
            continue
        half_band = 0.5 * float(bandwidths[index])
        phase_i = phase_spm(beta2, beta3, float(frequencies[index]), reference_frequency_hz)
        offsets = half_band * nodes
        weights_i = half_band * weights
        spm_mu = 0.0
        for offset_1, weight_1 in zip(offsets, weights_i, strict=True):
            link = _link_exact(profile, index, phase_i * offset_1 * offsets)
            spm_mu += weight_1 * float(np.sum(weights_i * np.abs(link) ** 2))
        spm[index] = (16.0 / 27.0) * float(gammas[index]) ** 2 * spm_mu / (2.0 * half_band) ** 2 * launch**3

        total = 0.0
        for interferer in range(n_channels):
            if interferer == index or powers[interferer] <= 0.0:
                continue
            phase_ik = phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            inner = _band_average_link_squared(
                profile,
                interferer,
                phase_ik,
                half_band,
                float(profile.alpha_npm[interferer]),
                nodes,
                weights,
            )
            total += (float(powers[interferer]) / launch) ** 2 * inner / float(bandwidths[interferer])
        xpm[index] = (32.0 / 27.0) * float(gammas[index]) ** 2 * total * launch**3
    return spm, xpm
