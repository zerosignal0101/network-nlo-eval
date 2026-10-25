r"""Buglia 2023 闭式 ISRS-GN 模型：任意损耗、任意纤长。

文献
----
H. Buglia, M. Jarmolovičius, A. Vasylchenkova, E. Sillekens, L. Galdino, R. I. Killey,
P. Bayvel, "A Closed-Form Expression for the Gaussian Noise Model in the Presence of
Inter-Channel Stimulated Raman Scattering Extended for Arbitrary Loss and Fibre Length",
J. Lightwave Technol. vol. 41, no. 11, pp. 3577-3588, Jun. 2023,
doi:10.1109/JLT.2023.3256185.

模型要点
--------
相对 Semrau 2019 的唯一结构性改动在论文 Eq. (30)：链路函数里

.. math::

    \frac{1 - e^{-(\alpha_{l,i} - j\phi)L}}{-\alpha_{l,i} + j\phi}
    \;\approx\; \frac{\kappa_{l,i}}{-\tilde{\alpha}_{l,i} + j\phi}

取代了 Semrau 的 Eq. (42)（要求 :math:`\alpha_{l,i} L \to \infty`）。新近似用
:math:`\tilde{\alpha}_{l,i}` (Eq. (15)) 与 :math:`\kappa_{l,i}` (Eq. (16)) 在
:math:`\phi = 0` 处做一阶 Taylor 匹配，因此**短跨与极低损耗也成立**（论文
Appendix D 给出适用条件 :math:`\alpha_{l,i} L \gg 0`）。

本模块实现的闭式链条：

* Eq. (13)：功率剖面 :math:`\rho(z,f_i) = e^{-\alpha_i z}[1 - P_{tot} C_{r,i} f_i L_{eff}(z)]`；
* Eq. (14)：链路函数闭式；
* Eq. (15)/(16)：:math:`\tilde{\alpha}_{l,i}` 与 :math:`\kappa_{l,i}`；
* Eq. (17)：单对干扰信道的 XPM 贡献；
* Eq. (18)：SPM 贡献。

相位失配按论文取值：:math:`\phi_i = -4\pi^2(\beta_2 + 2\pi\beta_3 f_i)`、
:math:`\phi_{i,k} = -4\pi^2(f_k-f_i)[\beta_2 + \pi\beta_3(f_i+f_k)]`。

本模型只做 GN。论文 Eq. (19)/(20) 的调制格式修正项未实现（见下）。

实现取舍（与论文的偏差）
------------------------
1. **剖面参数不做逐链路拟合。** 论文 Eq. (13) 里的 :math:`\alpha_i`、
   :math:`\bar\alpha_i`、:math:`C_{r,i}` 是为复现数值拉曼剖面而**拟合**出来的
   逐信道参数，论文自己也指出"每换一条链路就要重新拟合"。本实现改用可解析、
   逐信道确定的取值：:math:`\alpha_i` 取该波长的光纤功率衰减（支持衰减多项式），
   :math:`\bar\alpha_i = \alpha_i`（Semrau 的默认），:math:`C_{r,i}` 取该光纤的
   拉曼增益谱斜率常数。这样"任意损耗 / 任意纤长"的推广完全来自 Eq. (30) 的链路
   函数近似本身，不依赖任何迭代优化。
2. **出纤功率用冻结核的拉曼闭式剖面**（``models.numba_kernels`` 的 ECZ 三角增益
   近似），而不是论文的半解析式 (13)。理由：(13) 是一阶展开，强拉曼下括号可能
   变负；ECZ 闭式数值稳健，且与本仓库的 ``ecz`` 基线口径一致，便于横向比较。
   NLI 一律走论文闭式。
3. **未实现 Eq. (19)/(20) 的调制格式修正项**（论文 Eq. (21) 里的 :math:`\Phi`
   修正）。本模型是 GN 模型，修正项需要超额峰度这一外部参数；论文 Eq. (20) 与
   Eq. (21) 的前因子记账本身也不自洽（``2\pi`` 与 ``5/6``），需要先厘清。
4. **相干因子 :math:`n^\epsilon`** 按论文 Eq. (3) 施加在 SPM 上，但冻结的
   :mod:`network_nlo_eval.models.route` 逐跨累加噪声时固定传 ``n=1``，所以多跨
   场景下该因子实际不生效（``ecz`` 基线同样如此）。

跨间相干
--------
多跨链路上，各跨产生的 NLI 噪声**不是**独立相加的：第 ``k`` 跨生成的噪声比
第 0 跨生成的噪声多走了 ``k·L`` 米，两份拷贝之间的相位差由色散在这段距离上
积累的失配决定。相干度必须按 **NLI 噪声功率**加权，而不是按信号谱均匀平均：

.. math::

    V(d) = \left\langle \cos(\rho d)\right\rangle_W, \qquad
    W(\rho) = \frac{\left|1 - e^{-(\alpha + j\rho)L}\right|^2}{\alpha^2 + \rho^2},

其中 :math:`\rho = 4\pi^2|\beta_2|\,(f_1 - f)(f_2 - f)` 是信号对的相位失配率
(rad/m)。加权平均的权重就是闭式 GN 核本身：失配越大的频率对产生的 NLI 越少，
因此它对跨间相干度的贡献也越小。

可见度是链路级量：给定跨数 :math:`N` 与各跨间距 :math:`kL`，相干因子

.. math::

    R(N) = N + 2\sum_{k=1}^{N-1}(N-k)\,V(kL), \qquad \mathrm{buildup} = R(N)/N

给出"完全相干"（:math:`R = N^2`）与"完全非相干"（:math:`R = N`）之间的插值。
零色散时 :math:`V \equiv 1`，buildup 退化为 :math:`N`；色散很强时
:math:`V \approx 0`，buildup 退化为 1。

冻结的路由层每个信道每个跨只携带**一个**复振幅，无法表达 :math:`V(j-k)` 的
两两结构，因此这里用**等效相位斜坡**：取每跨相位步长 :math:`\psi` 使

.. math::

    \left[\frac{\sin(N\psi/2)}{\sin(\psi/2)}\right]^2 = R(N),

并让第 ``k`` 跨的振幅相位随已走过的长度线性增长。对路由层真正需要的"总 NLI
功率"这个量是精确的；两两结构是等效近似（与本仓库其它相干档模型同口径）。

单跨链路（``total_spans == 1``）**不走**这条路径：核返回纯功率结果，路由层
按 :math:`\sum|a_k|^2` 累加，与改造前逐位一致。

实现说明
--------
* 可见度的一维归约与洛伦兹反导数表都是**本模块自己的实现**（不 import
  :mod:`network_nlo_eval.models.numerical`，避免两个模型耦死），口径与它一致：
  观测频率用余弦栅格，第二个输入频率的积分用解析反导数
  :math:`C_m(t) = \int_0^t \cos(m\alpha L s)/(1+s^2)\,\mathrm{d}s`。
* XPM 用**同一套** NLI 加权：跨拍频的失配变量换成"波 1 的带内偏移 × 波 2 相对
  被观测信道中心的偏移"。见 :func:`_gn_weighted_xpm_visibility`。实测
  （``c_57ch``、10 跨）跨拍频的可见度只有 1.9e-3 量级，XPM 的相干增益
  （buildup）是 1.0036；与均匀带平均口径相比链路级 SNR 差 0.004 dB，远在
  0.05 dB 的噪声地板之下，因此保留 GN 加权口径（与 SPM 同一套权重）。
* 逐跨相位步长按**等跨**假设折算：跨长不等时，步长仍按当前跨长折算。

适用边界（实测，论文未声明）
--------------------------
论文只给损耗侧的适用条件（Appendix D 的 :math:`\alpha_{l,i}L \gg 0`），没有给
色散限制。实测 57 信道 / 20 跨 / 0 dBm per channel，与本仓库 ``numerical`` 参考比，
误差随色散趋零单调恶化（ΔSNR 均值）：D = 17 / 8 / 3.8 / 0.5 / −1.8 ps/(nm·km)
依次为 −0.02 / +0.53 / +1.43 / **+5.01** / +2.80 dB；同样结构的 :class:`Semrau 2019`
在同一条梯度上 ≤0.04 dB。所以本模型**不适合低色散光纤**（|D| 小于约 5 ps/(nm·km)），
这一点论文没有写。机理未查，读数见 ``REPORT.md`` §14.1。
"""

from __future__ import annotations

import math

import numpy as np

from network_nlo_eval.core.constants import PI
from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
    register_model,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_gamma_coeff_scalar,
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
)
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "BUGLIA2023_MODEL_INFO",
    "Buglia2023Model",
    "Buglia2023SpanKernel",
    "alpha_tilde_kappa",
]

#: 拉曼增益谱斜率 ``C_r`` [1/(W·m·Hz)]。
#:
#: 与冻结核 :mod:`network_nlo_eval.models.numba_kernels` 的 ``_SRS_CR`` 同一口径：
#: 该系数乘以 ``P_tot * f`` (W·Hz) 与 ``L_eff`` (m) 得到无量纲的拉曼耗尽指数，
#: 与论文 Eq. (13) 的 ``P_tot C_{r,i} f_i L_eff(z)`` 一致。
_SRS_CR = 0.028 / 1e3 / 1e12

#: GN 相干因子指数 :math:`\epsilon`（Poggiolini 2012, Eq. (22)）。
_COHERENCE_EPSILON = 0.05

#: 相位失配的夹取下界，避免 :math:`1/\phi` 在自配对处发散。
_EPS_PHI = 1e-35

#: SRS 增益谱的截断频率 (Hz)。
_DELTA_F_CO_HZ = 15e12

BUGLIA2023_MODEL_INFO = ModelInfo(
    name="buglia2023",
    summary="Buglia 2023 closed-form ISRS-GN for arbitrary loss and fibre length.",
    family="gn",
    reference="Buglia et al., 'A Closed-Form Expression for the Gaussian Noise Model in the "
    "Presence of Inter-Channel Stimulated Raman Scattering Extended for Arbitrary Loss and "
    "Fibre Length', JLT 2023, doi:10.1109/JLT.2023.3256185",
    options=(
        ("egn", "Not supported: the paper derives the GN model only."),
    ),
)


def alpha_tilde_kappa(alpha_l: float, length_m: float) -> tuple[float, float]:
    r"""论文 Eq. (15)/(16)：一阶 Taylor 匹配出来的有效衰减与幅度系数。

    .. math::

        \tilde{\alpha}_{l,i} = \frac{\alpha_{l,i}(1 - e^{-\alpha_{l,i} L})}
        {1 - e^{-\alpha_{l,i} L} - \alpha_{l,i} L e^{-\alpha_{l,i} L}},\qquad
        \kappa_{l,i} = \frac{\tilde{\alpha}_{l,i}(1 - e^{-\alpha_{l,i} L})}{\alpha_{l,i}}

    极限行为：:math:`\alpha_{l,i} L \to \infty` 时 :math:`\tilde{\alpha} \to \alpha`、
    :math:`\kappa \to 1`（即退回 Semrau 的 Eq. (42)）；:math:`\alpha_{l,i} L \to 0` 时
    :math:`\tilde{\alpha} \to 2/L`、:math:`\kappa \to 2`，链路函数退化为零损耗的
    :math:`L^2`。两种极限都用级数展开求值，避免 ``1 - (1+x) e^{-x}`` 的相消误差。

    Args:
        alpha_l: :math:`\alpha_{l,i} = \alpha_i + l\tilde{\alpha}_i` (Np/m)。
        length_m: 跨段长度 (m)。
    Returns:
        ``(tilde_alpha, kappa)``，分别是有效衰减 (Np/m) 与无量纲幅度系数。
    """
    x = alpha_l * length_m
    if x < 1e-3:
        # 级数：1 - e^{-x} = x - x^2/2 + x^3/6 - x^4/24 ...
        em1 = x * (1.0 - x / 2.0 + x * x / 6.0 - x * x * x / 24.0)
        # 级数：1 - (1+x) e^{-x} = x^2/2 - x^3/3 + x^4/8 - x^5/30 ...
        denom = x * x * (0.5 - x / 3.0 + x * x / 8.0 - x * x * x / 30.0)
    else:
        exp_neg = np.exp(-x)
        em1 = 1.0 - exp_neg
        denom = 1.0 - (1.0 + x) * exp_neg
    tilde_alpha = alpha_l * em1 / denom
    kappa = em1 * em1 / denom
    return float(tilde_alpha), float(kappa)


# =============================================================================
# 跨间相干：GN 噪声功率加权的可见度
#
# 这一节是本模块自洽的实现：口径与 ``models.numerical`` 一致，但代码不共享，
# 免得两个模型耦死（数值模型是"可追溯性优先"的慢参考，闭式模型是"速度优先"
# 的生产线实现，耦合会让两边都难改）。
# =============================================================================

#: 可见度求积里观测频率的节点数。
#:
#: 观测频率跨信道带宽分布：被积函数在带宽边缘（可容许的三波区域退化）很陡，
#: 中间很平滑，因此用余弦栅格而不是均匀栅格。
_VIS_FREQUENCY_POINTS = 41

#: 每个积分子区间上的 Gauss-Legendre 节点数。
_VIS_QUADRATURE_POINTS = 200

#: 反导数表 ``C_m(t)`` 的步长。
_VIS_TABLE_STEP = 1.0e-3

#: 反导数表覆盖的最大 ``t``；超出部分用洛伦兹尾的渐近展开。
_VIS_TABLE_LIMIT = 30.0

#: 反导数表至少覆盖的 ``t`` 范围（保证表格良态，且不会用到只在大 ``t`` 有效的尾）。
_VIS_TABLE_MINIMUM = 1.0

#: 小于该值的失配率视作零：相位差可忽略，两份拷贝完全相干。
_VIS_NEGLIGIBLE_T = 1.0e-6

#: 等效相位斜坡的二分步数。
_RAMP_BISECTION_STEPS = 60


def _lorentzian_tail(beta: float, t: NDArrayFloat) -> NDArrayFloat:
    r"""洛伦兹尾 :math:`\int_t^\infty \cos(\beta s)/(1+s^2)\,\mathrm{d}s`。

    反复分部积分得到的渐近展开。``beta`` 是无量纲的
    :math:`m\alpha L`；展开在 ``t = 30`` 处的相对误差优于 1e-8。

    Args:
        beta: 无量纲频率 :math:`m\alpha L`。
        t: 尾积分的下限（数组）。
    Returns:
        尾积分的值（数组）。
    """
    denominator = 1.0 + t * t
    f = 1.0 / denominator
    df = -2.0 * t / denominator**2
    d2f = (6.0 * t * t - 2.0) / denominator**3
    d3f = -24.0 * t * (t * t - 1.0) / denominator**4
    d4f = 24.0 * (5.0 * t**4 - 10.0 * t * t + 1.0) / denominator**5
    sin_beta = np.sin(beta * t)
    cos_beta = np.cos(beta * t)
    return (
        -f * sin_beta / beta
        - df * cos_beta / beta**2
        + d2f * sin_beta / beta**3
        + d3f * cos_beta / beta**4
        - d4f * sin_beta / beta**5
    )


class _VisibilityAntiderivative:
    r"""反导数 :math:`C_m(t) = \int_0^t \cos(m\alpha L s)/(1+s^2)\,\mathrm{d}s`。

    表用**修正梯形法**（复合梯形 + Euler-Maclaurin 的一阶修正）建立：修正项
    恢复了四阶精度，代价却只有 Simpson 网格的一半。``m = 0`` 时
    :math:`C_0(t) = \arctan t`，直接闭式求值。

    表之外用洛伦兹尾的渐近展开，极限值取洛伦兹的 Fourier 变换
    :math:`C_m(\infty) = (\pi/2)e^{-m\alpha L}`。

    Args:
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        span_length_m: 跨段长度 (m)。
        m_max: 需要建表的最大 ``m``。
        t_end: 需要覆盖的最大 ``t``。
    """

    def __init__(
        self,
        alpha_npm_m: float,
        span_length_m: float,
        m_max: int,
        t_end: float,
        step: float = _VIS_TABLE_STEP,
    ) -> None:
        self.alpha = float(alpha_npm_m)
        self.span_length = float(span_length_m)
        self.beta = self.alpha * self.span_length
        limit = min(max(float(t_end), _VIS_TABLE_MINIMUM), _VIS_TABLE_LIMIT)
        n_points = max(3, int(math.ceil(limit / step)) + 1)
        if n_points % 2 == 0:
            n_points += 1
        self.t = np.linspace(0.0, limit, n_points)
        h = float(self.t[1] - self.t[0])
        self._tables: dict[int, NDArrayFloat] = {}
        for m in range(1, int(m_max) + 1):
            phase = m * self.beta * self.t
            cos_phase = np.cos(phase)
            values = cos_phase / (1.0 + self.t * self.t)
            derivative = (
                -m * self.beta * np.sin(phase) / (1.0 + self.t * self.t)
                - 2.0 * self.t * cos_phase / (1.0 + self.t * self.t) ** 2
            )
            self._tables[m] = self._cumulative(values, derivative, h)

    @staticmethod
    def _cumulative(
        values: NDArrayFloat, derivative: NDArrayFloat, h: float
    ) -> NDArrayFloat:
        """修正梯形法的累积积分。"""
        integral = np.zeros_like(values)
        integral[1:] = np.cumsum(0.5 * h * (values[:-1] + values[1:]))
        # Euler-Maclaurin：int = T - h^2/12 (f'(b) - f'(a)) + O(h^4)。
        integral[1:] -= h * h / 12.0 * (derivative[1:] - derivative[0])
        return integral

    def C(self, m: int, t: NDArrayFloat) -> NDArrayFloat:
        """``C_m(t)``，``t`` 可以是负数（``C_m`` 是奇函数）。"""
        if m == 0:
            return np.arctan(np.asarray(t, dtype=np.float64))
        t = np.asarray(t, dtype=np.float64)
        out = np.empty_like(t)
        sign = np.sign(t)
        magnitude = np.abs(t)
        inside = magnitude <= self.t[-1]
        if np.any(inside):
            out[inside] = sign[inside] * np.interp(magnitude[inside], self.t, self._tables[m])
        outside = ~inside
        if np.any(outside):
            beta = m * self.beta
            tail = _lorentzian_tail(beta, magnitude[outside])
            out[outside] = sign[outside] * (0.5 * math.pi * math.exp(-beta) - tail)
        return out


def _weighted_visibility(
    t_low: NDArrayFloat,
    t_high: NDArrayFloat,
    measure: NDArrayFloat,
    table: _VisibilityAntiderivative,
    exp_alpha_l: float,
    n_separations: int,
) -> NDArrayFloat:
    r"""由归约后的求积节点求各跨距的可见度。

    权重与被积函数都只通过 :math:`t = |\rho|/\alpha` 依赖失配率 :math:`\rho`，
    于是

    .. math::

        \int \left|1 - e^{-(\alpha + j\rho)L}\right|^2 \cos(m\rho L)
        \frac{\mathrm{d}t}{1+t^2}
        = (1 + e^{-2\alpha L})\,C_m(t)
          - e^{-\alpha L}\left(C_{|m-1|}(t) + C_{m+1}(t)\right),

    其中用到 :math:`2\cos(\alpha L s)\cos(m\alpha L s) =
    \cos((m-1)\alpha L s) + \cos((m+1)\alpha L s)`。跨距 :math:`d = mL`` 时
    相位差是 :math:`\rho d = m\alpha L t`，所以 :math:`m` 必须是跨距的整数倍
    ——这正是"等跨"假设在数学上的落点。

    Args:
        t_low: 每个求积节点上 ``t`` 区间的下限。
        t_high: 每个求积节点上 ``t`` 区间的上限。
        measure: 每个求积节点的测度。
        table: 反导数表。
        exp_alpha_l: :math:`e^{-\alpha L}`。
        n_separations: 跨距数（可见度数组的长度）。
    Returns:
        ``(n_separations,)`` 可见度，取值在 ``[-1, 1]``。
    """
    weight_high = (1.0 + exp_alpha_l**2) * table.C(0, t_high) - 2.0 * exp_alpha_l * table.C(1, t_high)
    weight_low = (1.0 + exp_alpha_l**2) * table.C(0, t_low) - 2.0 * exp_alpha_l * table.C(1, t_low)
    total = float(np.sum(measure * (weight_high - weight_low)))
    if not total > 0.0:
        # 权重退化（例如零带宽）：两份拷贝没有任何可分辨的相位差。
        return np.ones(n_separations, dtype=np.float64)
    out = np.empty(n_separations, dtype=np.float64)
    for index in range(n_separations):
        m = index + 1
        cosine_high = (
            (1.0 + exp_alpha_l**2) * table.C(m, t_high)
            - exp_alpha_l * (table.C(abs(m - 1), t_high) + table.C(m + 1, t_high))
        )
        cosine_low = (
            (1.0 + exp_alpha_l**2) * table.C(m, t_low)
            - exp_alpha_l * (table.C(abs(m - 1), t_low) + table.C(m + 1, t_low))
        )
        out[index] = float(np.sum(measure * (cosine_high - cosine_low))) / total
    return out


def _spm_region_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    n_frequency: int,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    r"""自拍频（SPM）三波区域的求积节点。

    被观测频率 :math:`f` 跨信道带宽分布；两个输入波是
    :math:`f_1 = f + u`、:math:`f_2 = f + v`，闲频 :math:`f_3 = f + u + v`，
    三者都必须落在同一信道带内。于是 :math:`v` 的取值区间是

    .. math::

        v \in [f - B/2, f + B/2] \cap [f - B/2 - u, f + B/2 - u].

    对固定的 :math:`(f, u)`，失配率 :math:`\rho = 4\pi^2|\beta_2| u v` 随
    :math:`v` 单调，:math:`t = |\rho|/\alpha` 因此也单调，内层的 :math:`v`
    积分用反导数解析求出；外层的 :math:`(f, u)` 用数值求积。

    Args:
        half_band_hz: 信道带宽的一半 (Hz)。
        rate: :math:`4\pi^2|\beta_2|` (rad/(m·Hz²))。
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        n_frequency: 观测频率的节点数。
        n_quadrature: 每个子区间上的 Gauss-Legendre 节点数。
    Returns:
        ``(t_low, t_high, measure)``：``t`` 区间的上下限与每个节点的测度。
    """
    frequencies, frequency_weights = _cosine_frequency_grid(half_band_hz, n_frequency)
    nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)
    low_parts: list[NDArrayFloat] = []
    high_parts: list[NDArrayFloat] = []
    measure_parts: list[NDArrayFloat] = []
    for frequency, frequency_weight in zip(frequencies, frequency_weights, strict=True):
        low = -half_band_hz - float(frequency)
        high = half_band_hz - float(frequency)
        for start, stop in ((low, 0.0), (0.0, high)):
            if stop - start <= 0.0:
                continue
            u = 0.5 * (stop - start) * (nodes + 1.0) + start
            weight = 0.5 * (stop - start) * weights
            v_low = np.maximum(low, low - u)
            v_high = np.minimum(high, high - u)
            valid = v_high > v_low
            if not np.any(valid):
                continue
            u = u[valid]
            weight = weight[valid]
            v_low = v_low[valid]
            v_high = v_high[valid]
            # dv = dt / (rate |u| alpha)：Jacobian 取绝对值。
            measure = weight * frequency_weight / (rate * np.abs(u) * alpha_npm_m)
            t_low = rate * u * v_low / alpha_npm_m
            t_high = rate * u * v_high / alpha_npm_m
            low_parts.append(np.minimum(t_low, t_high))
            high_parts.append(np.maximum(t_low, t_high))
            measure_parts.append(measure)
    if not low_parts:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    return (
        np.concatenate(low_parts),
        np.concatenate(high_parts),
        np.concatenate(measure_parts),
    )


def _xpm_region_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    channel_offset_hz: float,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    r"""跨拍频（XPM）三波区域的求积节点。

    跨拍的失配变量与自拍频不同：参与拍频的两列波分处两个信道，失配率是

    .. math::

        \rho = 4\pi^2|\beta_2|\,x\,(\Delta + y),

    其中 :math:`x` 是波 1（邻居信道）相对该信道中心的带内偏移，
    :math:`\Delta` 是两个信道中心的间隔，:math:`y` 是波 2 相对**被观测信道**
    中心的偏移。闲频 :math:`f_3 = f_1 + y` 必须落在邻居信道内，于是

    .. math::

        y \in [-B/2, B/2] \cap [-B/2 - x, B/2 - x].

    与自拍频一样，固定 :math:`x` 后 :math:`y` 的积分用反导数解析求出。

    Args:
        half_band_hz: 信道带宽的一半 (Hz)。
        rate: :math:`4\pi^2|\beta_2|` (rad/(m·Hz²))。
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        channel_offset_hz: 两个信道中心的间隔 (Hz)，取正值。
        n_quadrature: 积分子区间上的 Gauss-Legendre 节点数。
    Returns:
        ``(t_low, t_high, measure)``。
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)
    low_parts: list[NDArrayFloat] = []
    high_parts: list[NDArrayFloat] = []
    measure_parts: list[NDArrayFloat] = []
    for start, stop in ((-half_band_hz, 0.0), (0.0, half_band_hz)):
        if stop - start <= 0.0:
            continue
        x = 0.5 * (stop - start) * (nodes + 1.0) + start
        weight = 0.5 * (stop - start) * weights
        y_low = np.maximum(-half_band_hz, -half_band_hz - x)
        y_high = np.minimum(half_band_hz, half_band_hz - x)
        valid = y_high > y_low
        if not np.any(valid):
            continue
        x = x[valid]
        weight = weight[valid]
        y_low = y_low[valid]
        y_high = y_high[valid]
        measure = weight / (rate * np.abs(x) * alpha_npm_m)
        t_low = rate * x * (channel_offset_hz + y_low) / alpha_npm_m
        t_high = rate * x * (channel_offset_hz + y_high) / alpha_npm_m
        low_parts.append(np.minimum(t_low, t_high))
        high_parts.append(np.maximum(t_low, t_high))
        measure_parts.append(measure)
    if not low_parts:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    return (
        np.concatenate(low_parts),
        np.concatenate(high_parts),
        np.concatenate(measure_parts),
    )


def _cosine_frequency_grid(
    half_band_hz: float, n_frequency: int
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """观测频率的余弦栅格及其复合梯形权重。"""
    if n_frequency > 1:
        index = np.arange(n_frequency, dtype=np.float64)
        frequencies = -half_band_hz * np.cos(np.pi * index / (n_frequency - 1))
        weights = np.empty(n_frequency, dtype=np.float64)
        weights[0] = 0.5 * (frequencies[1] - frequencies[0])
        weights[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
        weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
    else:
        frequencies = np.zeros(1, dtype=np.float64)
        weights = np.ones(1, dtype=np.float64)
    return frequencies, weights


def _gn_weighted_spm_visibility(
    *,
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    span_separations_m: NDArrayFloat,
) -> NDArrayFloat:
    r"""GN 噪声功率加权的自拍频（SPM）跨间可见度。

    可见度是"两份噪声拷贝"的相干度，按**产生噪声的频率对**加权：

    .. math::

        V(d) = \frac{\int\int \cos(\rho d)\,W(\rho)\,\mathrm{d}f_1\mathrm{d}f_2}
        {\int\int W(\rho)\,\mathrm{d}f_1\mathrm{d}f_2}, \qquad
        W(\rho) = \frac{\left|1 - e^{-(\alpha + j\rho)L}\right|^2}{\alpha^2 + \rho^2}.

    权重 :math:`W` 就是闭式 GN 核：失配越大的频率对产生的 NLI 越少，因此它
    对跨间相干度的贡献也越小。均匀权重（信号谱口径）给的是**两份信号副本**
    之间的相干度，不是噪声的相干度。

    Args:
        beta2_abs_s2_m: :math:`|\beta_2|` (s²/m)。
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        bandwidth_hz: 信道带宽 (Hz)。
        span_length_m: 跨段长度 (m)，失配与洛伦兹宽度都以**单跨**为尺度。
        span_separations_m: ``(K,)`` 两份拷贝之间的距离 (m)，必须是跨长的整数倍。
    Returns:
        ``(K,)`` 可见度，取值在 ``[-1, 1]``；无色散时为 1。
    """
    separations = np.asarray(span_separations_m, dtype=np.float64)
    if separations.size == 0:
        return np.zeros(0, dtype=np.float64)
    if beta2_abs_s2_m == 0.0 or alpha_npm_m <= 0.0 or bandwidth_hz <= 0.0:
        return np.ones(separations.size, dtype=np.float64)
    rate = 4.0 * PI**2 * abs(float(beta2_abs_s2_m))
    t_low, t_high, measure = _spm_region_nodes(
        0.5 * float(bandwidth_hz),
        rate,
        float(alpha_npm_m),
        _VIS_FREQUENCY_POINTS,
        _VIS_QUADRATURE_POINTS,
    )
    if t_high.size == 0 or float(np.max(np.abs(t_high))) < _VIS_NEGLIGIBLE_T:
        # 失配率小到相位差可以忽略：两份拷贝完全相干。
        return np.ones(separations.size, dtype=np.float64)
    table = _VisibilityAntiderivative(
        float(alpha_npm_m),
        float(span_length_m),
        int(separations.size) + 1,
        float(np.max(np.abs(t_high))),
    )
    exp_alpha_l = float(np.exp(-float(alpha_npm_m) * float(span_length_m)))
    return _weighted_visibility(t_low, t_high, measure, table, exp_alpha_l, int(separations.size))


def _gn_weighted_xpm_visibility(
    *,
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    span_separations_m: NDArrayFloat,
    channel_offset_hz: float,
) -> NDArrayFloat:
    r"""GN 噪声功率加权的跨拍频（XPM）跨间可见度。

    与自拍频用**同一套**权重 :math:`W(\rho)` 与同一个反导数表，只是失配变量
    换成跨拍频的形式 :math:`\rho = 4\pi^2|\beta_2| x (\Delta + y)`（见
    :func:`_xpm_region_nodes`）。干扰信道取**最近邻**：GN 核随信道间隔迅速
    衰减，XPM 噪声由最近的邻居主导。

    Args:
        beta2_abs_s2_m: :math:`|\beta_2|` (s²/m)。
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        bandwidth_hz: 信道带宽 (Hz)。
        span_length_m: 跨段长度 (m)。
        span_separations_m: ``(K,)`` 两份拷贝之间的距离 (m)。
        channel_offset_hz: 干扰信道与被观测信道的中心间隔 (Hz)。
    Returns:
        ``(K,)`` 可见度，取值在 ``[-1, 1]``。
    """
    separations = np.asarray(span_separations_m, dtype=np.float64)
    if separations.size == 0:
        return np.zeros(0, dtype=np.float64)
    offset = abs(float(channel_offset_hz))
    if beta2_abs_s2_m == 0.0 or alpha_npm_m <= 0.0 or bandwidth_hz <= 0.0 or offset == 0.0:
        return np.ones(separations.size, dtype=np.float64)
    rate = 4.0 * PI**2 * abs(float(beta2_abs_s2_m))
    t_low, t_high, measure = _xpm_region_nodes(
        0.5 * float(bandwidth_hz),
        rate,
        float(alpha_npm_m),
        offset,
        _VIS_QUADRATURE_POINTS,
    )
    if t_high.size == 0 or float(np.max(np.abs(t_high))) < _VIS_NEGLIGIBLE_T:
        return np.ones(separations.size, dtype=np.float64)
    table = _VisibilityAntiderivative(
        float(alpha_npm_m),
        float(span_length_m),
        int(separations.size) + 1,
        float(np.max(np.abs(t_high))),
    )
    exp_alpha_l = float(np.exp(-float(alpha_npm_m) * float(span_length_m)))
    return _weighted_visibility(t_low, t_high, measure, table, exp_alpha_l, int(separations.size))


def _coherent_factor(n_spans: int, visibility: NDArrayFloat) -> NDArrayFloat:
    r"""相干因子 :math:`R(N) = N + 2\sum_{k=1}^{N-1}(N-k)V(kL)`。

    Args:
        n_spans: 链路跨数。
        visibility: ``(N - 1, N_channels)`` 相隔 1、2、... 跨的两份拷贝的可见度。
    Returns:
        ``(N_channels,)`` 相干因子：完全相干时是 :math:`N^2`，完全非相干时是 :math:`N`。
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return n + 2.0 * (weights * visibility).sum(axis=0)


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    r"""等效每跨相位步长 :math:`\psi`，使相干因子等于 ``target``。

    路由层按复振幅相加，:math:`N` 跨的线性相位斜坡给出
    :math:`[\sin(N\psi/2)/\sin(\psi/2)]^2` 倍的单跨噪声。该因子从
    :math:`\psi = 0` 处的 :math:`N^2` 单调降到 :math:`\psi = 2\pi/N` 处的 0，
    因此二分即可。

    Args:
        n_spans: 链路跨数。
        target: ``(N_channels,)`` 想要的相干因子。
    Returns:
        ``(N_channels,)`` 每跨相位步长 (rad)。
    """
    n = int(n_spans)
    if n <= 1:
        return np.zeros(np.shape(target), dtype=np.float64)
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n)
    for _ in range(_RAMP_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        factor = (np.sin(0.5 * n * middle) / np.sin(0.5 * middle)) ** 2
        too_large = factor > goal
        low = np.where(too_large, middle, low)
        high = np.where(too_large, high, middle)
    return 0.5 * (low + high)


class Buglia2023SpanKernel:
    """Buglia 2023 单跨物理核。

    只做单跨：出纤功率用冻结核的拉曼闭式功率剖面，NLI 用论文 Eq. (17)/(18) 的闭式。
    放大器、ROADM 与噪声累积在 :mod:`network_nlo_eval.models.route` 里统一处理。

    Attributes
    ----------
        n_channels: 信道数 N。
    """

    def __init__(
        self,
        *,
        f_rel_hz: NDArrayFloat,
        lambdas_m: NDArrayFloat,
        a_eff_m2: NDArrayFloat,
        alpha_npm: NDArrayFloat,
        beta2_s2_m: float,
        beta3_s3_m: float,
        n2_m2_w: float,
        b_hz_channels: NDArrayFloat,
        c_r_w_m_hz: float = _SRS_CR,
        symbol_rate_hz: float = 50e9,
    ) -> None:
        r"""构造单跨核。

        Args:
            f_rel_hz: ``(N,)`` 相对频率 (Hz)，以波段中心为原点。
            lambdas_m: ``(N,)`` 绝对波长 (m)。
            a_eff_m2: ``(N,)`` 有效模面积 (m^2)。
            alpha_npm: ``(N,)`` 逐信道功率衰减系数 (Np/m)。
            beta2_s2_m: :math:`\beta_2` (s^2/m)。
            beta3_s3_m: :math:`\beta_3` (s^3/m)。
            n2_m2_w: 非线性折射率 (m^2/W)。
            b_hz_channels: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
            c_r_w_m_hz: 拉曼增益谱斜率 ``C_r`` [1/(W·m·Hz)]。
            symbol_rate_hz: 符号率 (Hz)，只用于确定 SRS 频带边界。
        """
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.lambdas_m = np.asarray(lambdas_m, dtype=np.float64)
        self.a_eff_m2 = np.asarray(a_eff_m2, dtype=np.float64)
        self.alpha_npm = np.asarray(alpha_npm, dtype=np.float64)
        self.beta2_s2_m = float(beta2_s2_m)
        self.beta3_s3_m = float(beta3_s3_m)
        self.n2_m2_w = float(n2_m2_w)
        self.b_hz_channels = np.asarray(b_hz_channels, dtype=np.float64)
        self.c_r_w_m_hz = float(c_r_w_m_hz)
        self.symbol_rate_hz = float(symbol_rate_hz)

        self.n_channels = int(self.f_rel_hz.shape[0])

        # Eq. (17)/(18) 的 SPM 相位失配：phi_i = -4 pi^2 (beta_2 + 2 pi beta_3 f_i)。
        self.phi_i = -4.0 * PI**2 * (self.beta2_s2_m + 2.0 * PI * self.beta3_s3_m * self.f_rel_hz)
        self.gamma_ii = np.array(
            [
                _calc_gamma_coeff_scalar(
                    self.lambdas_m[i], self.lambdas_m[i], self.n2_m2_w, self.a_eff_m2[i], self.a_eff_m2[i]
                )
                for i in range(self.n_channels)
            ],
            dtype=np.float64,
        )

        # (N, N) 常量按跨长缓存：只依赖信道几何、频带与跨长，不随功率变。
        self._pair_cache: dict[float, tuple[NDArrayFloat, ...]] = {}

        # --- 跨间相干所需的光纤级常量 -------------------------------------
        # 可见度只依赖失配率的绝对值 |beta_2|：相位失配是 beta_2 与 beta_3 的
        # 组合在带内的平均，而带内平均由 beta_2 主导（beta_3 只带来带内的缓变，
        # 在跨间尺度的相位差上贡献不到 1%）。
        self.beta2_abs_s2_m = abs(self.beta2_s2_m)
        # 可见度的权重用整条光纤的平均衰减：C 波段内衰减的变化远小于 1%。
        self.alpha_visibility_npm = float(np.mean(self.alpha_npm))
        # XPM 的干扰信道取最近邻：GN 核随信道间隔迅速衰减。
        if self.n_channels > 1:
            offsets = np.abs(self.f_rel_hz[:, None] - self.f_rel_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            self.partner_offset_hz = offsets.min(axis=1)
        else:
            self.partner_offset_hz = np.zeros(self.n_channels, dtype=np.float64)
        # 等效相位斜坡按 (跨数, 跨长) 记忆化；单位相位因子按
        # (跨数, 跨长, 累计长度/跨长) 记忆化。两者的内容都是光纤常量的纯函数，
        # 缓存命中后热路径上完全免费。
        self._ramp_cache: dict[tuple[int, float], tuple[NDArrayFloat, NDArrayFloat]] = {}
        self._factor_cache: dict[tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]] = {}

    # --- 跨间相干 ---------------------------------------------------------
    def _spm_visibility(self, distances_m: NDArrayFloat, span_length_m: float) -> NDArrayFloat:
        """自拍频（SPM）的跨间可见度 ``(K, N)``。

        逐信道带宽相同的信道共享同一列可见度，因此按唯一带宽分组计算。

        Args:
            distances_m: ``(K,)`` 两份拷贝之间的距离 (m)。
            span_length_m: 跨段长度 (m)。
        Returns:
            ``(K, N)`` 可见度。
        """
        out = np.empty((distances_m.size, self.n_channels), dtype=np.float64)
        unique, inverse = np.unique(self.b_hz_channels, return_inverse=True)
        for index, bandwidth in enumerate(unique):
            column = _gn_weighted_spm_visibility(
                beta2_abs_s2_m=self.beta2_abs_s2_m,
                alpha_npm_m=self.alpha_visibility_npm,
                bandwidth_hz=float(bandwidth),
                span_length_m=float(span_length_m),
                span_separations_m=distances_m,
            )
            out[:, inverse == index] = column[:, None]
        return out

    def _xpm_visibility(self, distances_m: NDArrayFloat, span_length_m: float) -> NDArrayFloat:
        """跨拍频（XPM）的跨间可见度 ``(K, N)``。

        干扰信道取最近邻，可见度按 (带宽, 最近邻间隔) 分组计算。

        Args:
            distances_m: ``(K,)`` 两份拷贝之间的距离 (m)。
            span_length_m: 跨段长度 (m)。
        Returns:
            ``(K, N)`` 可见度。
        """
        out = np.empty((distances_m.size, self.n_channels), dtype=np.float64)
        groups: dict[tuple[float, float], list[int]] = {}
        for channel in range(self.n_channels):
            key = (float(self.b_hz_channels[channel]), float(self.partner_offset_hz[channel]))
            groups.setdefault(key, []).append(channel)
        for (bandwidth, offset), channels in groups.items():
            column = _gn_weighted_xpm_visibility(
                beta2_abs_s2_m=self.beta2_abs_s2_m,
                alpha_npm_m=self.alpha_visibility_npm,
                bandwidth_hz=bandwidth,
                span_length_m=float(span_length_m),
                span_separations_m=distances_m,
                channel_offset_hz=offset,
            )
            out[:, channels] = column[:, None]
        return out

    def _inter_span_ramps(
        self, n_spans: int, span_length_m: float
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """等效每跨相位步长（SPM 与 XPM 各一条）。

        Args:
            n_spans: 链路跨数。
            span_length_m: 跨段长度 (m)。
        Returns:
            ``(ramp_spm, ramp_xpm)``，均为 ``(N,)`` (rad/跨)。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._ramp_cache.get(key)
        if cached is not None:
            return cached
        n = max(1, int(n_spans))
        distances = np.arange(1, n, dtype=np.float64) * float(span_length_m)
        if n > 1:
            visibility_spm = self._spm_visibility(distances, float(span_length_m))
            visibility_xpm = self._xpm_visibility(distances, float(span_length_m))
        else:
            visibility_spm = np.zeros((0, self.n_channels), dtype=np.float64)
            visibility_xpm = np.zeros((0, self.n_channels), dtype=np.float64)
        ramp_spm = _equivalent_phase_ramp(n, _coherent_factor(n, visibility_spm))
        ramp_xpm = _equivalent_phase_ramp(n, _coherent_factor(n, visibility_xpm))
        self._ramp_cache[key] = (ramp_spm, ramp_xpm)
        return ramp_spm, ramp_xpm

    def _span_phase_factors(
        self, span_context: SpanContext, span_length_m: float
    ) -> tuple[NDArrayComplex, NDArrayComplex]:
        """本跨 NLI 振幅的单位相位因子 ``exp(1j·phi)``。

        相位随链路上已走过的长度线性增长（等跨时即随跨号线性增长）：在第
        ``k`` 跨生成的噪声比第 0 跨生成的噪声多走了 ``k·L`` 米，两份拷贝之间
        的相位差就是色散在这段距离上积累的失配。跨号只从
        :class:`~network_nlo_eval.models.base.SpanContext` 取，核自己不记跨号。

        Args:
            span_context: 本跨在链路里的位置。
            span_length_m: 跨段长度 (m)。
        Returns:
            ``(factor_spm, factor_xpm)``，均为 ``(N,)`` 复数，模长为 1。
        """
        scale = span_context.cumulative_length_m / float(span_length_m)
        key = (int(span_context.total_spans), float(span_length_m), float(scale))
        cached = self._factor_cache.get(key)
        if cached is None:
            ramp_spm, ramp_xpm = self._inter_span_ramps(span_context.total_spans, span_length_m)
            cached = (np.exp(1j * ramp_spm * scale), np.exp(1j * ramp_xpm * scale))
            self._factor_cache[key] = cached
        return cached

    # --- 内部常量 ---------------------------------------------------------
    def _profile_coefficients(self, length_m: float) -> tuple[NDArrayFloat, ...]:
        r"""给定跨长，算出 Eq. (15)/(16) 的逐信道量与 Eq. (17) 的 ``(N, N)`` 配对矩阵。

        返回 ``(m0, m1, a0, k0, a1, k1)``：

        * ``m0[i, k]``、``m1[i, k]``：Eq. (17) 的两项配对权重，含 :math:`\gamma_{ik}^2`；
        * ``a0/k0``、``a1/k1``：逐信道的 :math:`\tilde{\alpha}_{l,k}` 与 :math:`\kappa_{l,k}`。

        Args:
            length_m: 跨段长度 (m)。
        Returns:
            上述六个数组。
        """
        cached = self._pair_cache.get(length_m)
        if cached is not None:
            return cached

        n = self.n_channels
        alpha = self.alpha_npm
        a0 = np.empty(n, dtype=np.float64)
        k0 = np.empty(n, dtype=np.float64)
        a1 = np.empty(n, dtype=np.float64)
        k1 = np.empty(n, dtype=np.float64)
        for i in range(n):
            a0[i], k0[i] = alpha_tilde_kappa(alpha[i], length_m)
            # alpha_{1,i} = alpha_i + 1 * bar_alpha_i，bar_alpha_i = alpha_i。
            a1[i], k1[i] = alpha_tilde_kappa(2.0 * alpha[i], length_m)

        # phi_{i,k} = -4 pi^2 (f_k - f_i) [beta_2 + pi beta_3 (f_i + f_k)]
        freq = self.f_rel_hz
        delta = freq[None, :] - freq[:, None]
        total = freq[:, None] + freq[None, :]
        phi_ik = -4.0 * PI**2 * delta * (self.beta2_s2_m + PI * self.beta3_s3_m * total)
        phi_safe = np.where(np.abs(phi_ik) < _EPS_PHI, np.where(phi_ik >= 0.0, _EPS_PHI, -_EPS_PHI), phi_ik)

        # gamma_{ik}（同信道时为 gamma_ii）
        gamma_ik = (2.0 * PI / self.lambdas_m)[:, None] * (
            2.0 * self.n2_m2_w / (self.a_eff_m2[:, None] + self.a_eff_m2[None, :])
        )
        gamma_sq = gamma_ik**2

        band = self.b_hz_channels[:, None]
        atan0 = np.arctan(phi_safe * band / (2.0 * a0[None, :]))
        atan1 = np.arctan(phi_safe * band / (2.0 * a1[None, :]))
        # 因子 4 来自 (l, l') 求和按 (l <-> l') 对称折叠：原始求和是
        # sum_{l,l'} 2*kappa_l*kappa_l'*(A_l + A_l')/(phi*(a_l + a_l'))，
        # 对称化后等于 4/phi * sum_{l,l'} kappa_l*kappa_l'*A_l/(a_l + a_l')。
        m0 = 4.0 * gamma_sq * atan0 / phi_safe
        m1 = 4.0 * gamma_sq * atan1 / phi_safe
        # 自配对不进 XPM 求和（论文 Eq. (4) 的 k != i）。
        np.fill_diagonal(m0, 0.0)
        np.fill_diagonal(m1, 0.0)

        result = (m0, m1, a0, k0, a1, k1)
        self._pair_cache[length_m] = result
        return result

    # --- 单跨核 -----------------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """计算一个跨段的出纤功率与 NLI 噪声。

        Args:
            power_in_w: ``(N,)`` 跨段入口每信道功率 (W)。0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 有效跨段数，SPM 的相干因子用（论文 Eq. (3)）。
            span_context: 本跨在链路中的位置。给出且链路跨数大于 1 时，返回值
                带上**逐跨复振幅**，路由层据此做相干累加；单跨链路（或没给
                ``span_context``）走纯功率分支，结果与改造前逐位一致。
        Returns:
            出纤功率与 SPM/XPM 噪声，均为 ``(N,)`` 且不与输入共享内存。
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        length_m = float(span_length_m)
        n_spans = max(int(n_effective_spans), 1)

        p_total = float(np.sum(power))

        # --- 出纤功率：拉曼闭式剖面（与冻结核同一套 SRS 三角增益近似）---
        f_m_hz = self.f_rel_hz[0] - self.symbol_rate_hz / 2.0
        f_M_hz = self.f_rel_hz[-1] + self.symbol_rate_hz / 2.0
        r_f = _calc_raman_profile_jit(self.f_rel_hz, f_m_hz, f_M_hz, _DELTA_F_CO_HZ, p_total)
        power_out = _calc_span_out_power_jit(power, self.f_rel_hz, self.alpha_npm, length_m, r_f, 0.0)

        # --- Eq. (13) 的剖面参数：T_i = 1 + T_tilde_i, T_tilde_i = -P_tot C_r f_i / bar_alpha_i ---
        m0, m1, a0, k0, a1, k1 = self._profile_coefficients(length_m)
        t_tilde = -p_total * self.c_r_w_m_hz * self.f_rel_hz / self.alpha_npm
        t_factor = 1.0 + t_tilde
        # T_i <= 0 意味着一阶拉曼展开已经失效（强拉曼）；夹到极小正值避免除零。
        t_factor = np.where(np.abs(t_factor) < 1e-12, 1e-12, t_factor)
        weight = -t_tilde / t_factor

        # Eq. (17)/(18) 的 (l, l') 求和按对称性折叠成两项，见模块文档。
        sum_a0_a1 = a0 + a1
        c0 = k0 * k0 / (2.0 * a0) + weight * k0 * k1 / sum_a0_a1
        c1 = weight * k0 * k1 / sum_a0_a1 + weight * weight * k1 * k1 / (2.0 * a1)

        # --- Eq. (18)：SPM ---
        phi_safe = np.where(np.abs(self.phi_i) < _EPS_PHI, np.where(self.phi_i >= 0.0, _EPS_PHI, -_EPS_PHI), self.phi_i)
        band_sq = self.b_hz_channels**2
        asinh0 = np.arcsinh(3.0 * phi_safe * band_sq / (8.0 * PI * a0))
        asinh1 = np.arcsinh(3.0 * phi_safe * band_sq / (8.0 * PI * a1))
        eta_spm = (
            (16.0 / 27.0)
            * (self.gamma_ii**2 / band_sq)
            * t_factor**2
            * (4.0 * PI / phi_safe)
            * (asinh0 * c0 + asinh1 * c1)
        )
        spm_noise = eta_spm * power**3 * n_spans**_COHERENCE_EPSILON

        # --- Eq. (17)：XPM ---
        u0 = (power**2 / self.b_hz_channels) * t_factor**2 * c0
        u1 = (power**2 / self.b_hz_channels) * t_factor**2 * c1
        xpm_noise = (32.0 / 27.0) * power * (m0 @ u0 + m1 @ u1)

        spm_noise = np.where(np.isfinite(spm_noise), spm_noise, 0.0)
        xpm_noise = np.where(np.isfinite(xpm_noise), xpm_noise, 0.0)
        spm_noise = np.maximum(spm_noise, 0.0)
        xpm_noise = np.maximum(xpm_noise, 0.0)

        if span_context is None or int(span_context.total_spans) <= 1:
            # 单跨：相干退化为单跨，路由层保持纯功率口径（逐位不变）。
            return SpanFiberOutput(power_out, spm_noise, xpm_noise)
        factor_spm, factor_xpm = self._span_phase_factors(span_context, length_m)
        # 振幅的参考面是本跨出纤处：|amp|**2 必须精确等于同名噪声功率。
        amplitude_spm = np.sqrt(spm_noise) * factor_spm
        amplitude_xpm = np.sqrt(xpm_noise) * factor_xpm
        return SpanFiberOutput(power_out, spm_noise, xpm_noise, amplitude_spm, amplitude_xpm)


class Buglia2023Model(RouteModel):
    """Buglia 2023 闭式 ISRS-GN 模型评估器（GN only）。

    在冻结契约 :class:`network_nlo_eval.models.route.RouteModel` 之上只实现单跨核：
    逐信道常数在模型层准备好，逐跨核按 :class:`FiberSpanConfig` 记忆化。
    """

    info = BUGLIA2023_MODEL_INFO

    #: 本模型给出跨间相干用的复振幅，路由层据此做复数累加。
    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        **route_options,
    ) -> None:
        """初始化模型。

        Args:
            grid: 频谱网格（遗留 ``SpectrumGrid`` 或 ``FlexibleGrid``）。
            ref_fiber_config: 参考光纤配置，提供色散、非线性与有效模面积。
            symbol_rate_hz: 符号率 (Hz)。
            egn: 论文只给出 GN 模型，``True`` 直接报错，不静默降级。
            **route_options: 透传给 :class:`RouteModel` 的选项。
        Raises:
            ValueError: ``egn=True``。
        """
        if egn:
            raise ValueError(
                "buglia2023 implements the GN model only: Buglia et al. (JLT 2023, "
                "doi:10.1109/JLT.2023.3256185) derive the closed-form expression for the "
                "Gaussian-noise model with ISRS; no EGN variant is published. "
                "Use egn=False."
            )
        super().__init__(grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options)

        self.ref_fiber_config = ref_fiber_config
        self.n2_m2_w = float(ref_fiber_config.nonlinear_index_n2)
        self.a_eff_m2 = _calc_aeff_dynamic(
            self.lambdas_m,
            ref_fiber_config.effective_area_um2_ref,
            ref_fiber_config.aeff_slope_um2_nm,
            ref_fiber_config.reference_wavelength_nm,
        )
        ref_lambda_m = 2.99792458e8 / self.band_center_hz
        self.beta2_s2_m = _calc_beta2_from_d(ref_fiber_config.dispersion_parameter_d, ref_lambda_m)
        self.beta3_s3_m = _calc_beta3_from_s_d(
            ref_fiber_config.dispersion_slope_s,
            ref_fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> Buglia2023SpanKernel:
        """按光纤参数构造单跨核。

        Args:
            fiber_config: 该跨段的光纤配置。
        Returns:
            与该光纤配置匹配的单跨核。
        """
        if fiber_config.use_attenuation_polynomial:
            alpha_npm = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            alpha_db_km = fiber_config.attenuation_db_km_ref
            alpha_npm = np.full(self.n_nodes, alpha_db_km * (np.log(10) / 10.0) / 1e3, dtype=np.float64)
        return Buglia2023SpanKernel(
            f_rel_hz=self.f_rel_hz,
            lambdas_m=self.lambdas_m,
            a_eff_m2=self.a_eff_m2,
            alpha_npm=alpha_npm,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            n2_m2_w=self.n2_m2_w,
            b_hz_channels=self.channel_bandwidth_hz_channels,
            c_r_w_m_hz=_SRS_CR,
            symbol_rate_hz=self.symbol_rate_hz,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """单跨：调用核（带跨间相干振幅），再统一走放大器/ASE。

        振幅的参考面按契约是**本跨出纤处**（放大器之前）。冻结层
        :func:`~network_nlo_eval.models.route.apply_amplifier_and_roadm` 在相干档
        只把振幅按 ROADM 损耗缩放，不乘行内放大器增益——这与纯功率档"NLI 不被
        行内放大器放大"的口径一致。

        Args:
            fiber_config: 该跨的光纤参数。
            edfa_config: 跨段放大器配置。
            power_in_w: 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            span_context: 本跨在链路里的位置；给出时核据此给出相对相位。
        Returns:
            :class:`~network_nlo_eval.models.base.ElementOutput`。
        """
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate(power_in_w, span_length_m, 1, span_context=span_context)
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        return apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )


register_model(BUGLIA2023_MODEL_INFO, Buglia2023Model)
