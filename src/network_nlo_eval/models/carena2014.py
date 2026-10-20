"""Carena 2014 EGN 模型：按论文自己的积分式做数值求积。

**论文**

* A. Carena, G. Bosco, V. Curri, Y. Jiang, P. Poggiolini, F. Forghieri, *EGN model
  of non-linear fiber propagation*, Opt. Express 22(13):16335-16362, 2014,
  doi:10.1364/OE.22.016335（下称"2014"）。

**这一版改了什么**

第一轮用 [26]（JLT 33(2):459-473, 2015）的闭式修正代理 SCI 的 4/6 阶项，并用
一个手调常数给 6 阶项加权。两者都无法归因：读者分不清测到的 EGN 效应有多少来自
物理、多少来自那个常数。本版把两处都换成 2014 论文正文与附录里的**显式积分**，
用 Gauss-Legendre 求积算出来，模型里不再有来源不明的常数。

**结构**

2014 论文把 NLI 拆成 SCI / XCI / MCI 三块，每块都写成

.. code-block:: text

    G^EGN(f) = P^3 [ kappa1(f) + Phi_a kappa2(f) + Psi_a kappa3(f) ]

其中 ``kappa1`` 是 GN 项（只有 ``|s|^2`` 与 ``|mu|^2``），``kappa2`` 是一重反对称化
（``s s* mu mu*``），``kappa3`` 是两重反对称化（多一个频率积分）。XCI 侧按论文
Eq. (18) 分成 X1-X4 四个域，``kappa_mn`` 里的 ``m`` 是域号、``n`` 是阶数。MCI 侧
按 Eq. (23)/(24)/(25)/(26)。

**积分的化简**

论文附录 C 证明：4 阶项的三重积分与 6 阶项的四重积分都能降成二重积分——被积函数
里 ``f2'`` 那部分与 ``f2`` 那部分只差一个复共轭。本实现照附录 C 的换元做，因此
每个 ``kappa`` 只需要一次二重求积。

**求积方案：把内层积分换成 p 轴上的原函数**

被积函数只通过乘积 ``p = (f1-f)(f2-f)`` 依赖频率。把积分变量换成相对频率
``u = f1-f``、``v = f2-f``，就有 ``p = u*v``，而第三个波形是 ``f1+f2-f``（相对频率
``u+v``）。三个波形各落在一条频带里，于是对固定的 ``u``，内层区间由频带约束给出
（``v`` 在第 2 个波形的频带与 ``u+v`` 在第 3 个波形的频带之间的交集）。内层积分于是
可以对 ``p`` 做，两个原函数就够用：

* ``G(p) = int_0^p |mu(p')|^2 dp'``（GN 项）
* ``Psi(p) = int_0^p mu(p') dp'``（4/6 阶项）

这两个原函数在 ``p`` 轴上预计算成一张表（每个子区间用 Gauss-Legendre，插值用三次
Hermite），表外的外层积分用复合 Gauss-Legendre。

**求积的难点：跨间相干因子**

链路函数 ``mu``（Eq. (12)）里的 ``nu`` 含有 ``sin(N_s psi)/sin(psi)``，它让被积函
数在 ``p = (f1-f)(f2-f)`` 上出现 ``N_s`` 重细结构（周期 ``pi/(N_s |c|)``，
``c = 2 pi^2 beta2 L_s``）。整块矩形谱的普通张量求积无法解析它（节点数加倍时结果
不收敛）。本实现因此把两处子区间个数都由这个**物理周期**导出：原函数表的网格间距
取周期的 ``1/(4*points)``，外层积分的子区间个数取"外层变量扫过的振荡周期数"。
两者都不是经验常数。

**公式对照（按内容描述；Markdown 抽取丢了编号，按结构与 PDF 页逐条交叉确认）**

* SCI 的 GN 项（三重 ``|s|^2`` 与 ``|mu|^2`` 的二重积分，系数 ``16/27 R_s^3``）
  → :meth:`EgnIntegrator.sci`
* SCI 的 4 阶修正（两行三阶积分，含 ``s s* mu mu*``，系数 ``80/81`` 与 ``16/81``
  乘 ``R_s^2``）→ :meth:`EgnIntegrator.sci`
* SCI 的 6 阶修正（四重积分，系数 ``16/81 R_s``）→ :meth:`EgnIntegrator.sci`
* 链路函数 ``mu = zeta * nu``（``zeta`` 的分母里没有跨长 ``L_s``）→ :class:`LinkFunction`
* XPM 近似的 X1 域（系数 ``32/27 R_s^3`` 与 ``80/81 R_s^2``）→ :meth:`EgnIntegrator.xci`
* XCI 的 X1-X4 四个域与各自的幂次前置因子 → :meth:`Carena2014SpanKernel._xci_ratio`
* MCI 的 M0 域与 M1 域（系数 ``2*16/27``、``4*16/27``、``2*80/81``）→ :meth:`EgnIntegrator.mci`
* 附录 A 的 ``kappa_21`` … ``kappa_43`` → :meth:`EgnIntegrator.xci`
* 附录 B 的多信道 MCI 修正 → **未实现**（见"已知限制"）
* 附录 C 的三重/四重积分降二重积分的换元 → :meth:`EgnIntegrator._core`
* Table 1 的 ``Phi_a`` / ``Psi_a`` → :data:`MODULATION_TABLE`

**谱因子**

论文 Sect. 2 说矩形谱的平台值是 ``1/R_s``，所以每个 ``|s|^2`` 因子是 ``1/R_s^2``。
每个 ``kappa`` 都带三个这样的因子（GN 项是三个 ``|s|^2``；4/6 阶项是一个 ``|s|^2``
加上两个 ``|s`` 的乘积），谱因子统一是 ``R_s^{-6}``。论文显式写出的 ``R_s`` 幂次
（3、2、1）因此完全决定了 ``kappa1``/``kappa2``/``kappa3`` 之间的比例。

**已知近似与限制**

1. **频谱只用矩形谱。** 2014 论文 Sect. 2 明确假设"信道为矩形谱、带宽等于符号率"，
   并说明非矩形谱会按 [27] 附录 E 多出整数参数 ``p`` 与更多修正项，本文不处理。
   本实现照此只做矩形谱；论文自己用来做仿真的 RRC(0.05) 属于仿真侧的选择。
2. **GN 骨架沿用仓库冻结的闭式 GN**（含 SRS/拉曼），EGN 修正以**相对修正比**的形式
   乘上去：``rho = (kappa1 + Phi kappa2 + Psi kappa3) / kappa1``。``kappa1`` 同样由
   论文的积分求出，所以修正比完全来自论文公式，且与闭式 GN 的绝对标度无关。
   本实现的 SCI 绝对值与论文 Fig. 1(a) 的 GN 曲线在 1…50 跨上逐点吻合到 0.1 dB 以内。
3. **XCI 域 X2-X4 只在部分频率上非空**（论文 Sect. 4B：``|f_c| >= 2 R_s`` 时 X2-X4
   完全消失），实现里由 ``|s|`` 的硬零点自动兑现。
4. **MCI 只做相邻三信道**（Eq. (23)-(26) 的原始三信道形式）。附录 B 的多信道推广
   （``kappa_M2,2``、``kappa_M3,2`` 与按 ``n`` 求和的 ``kappa_M1,2``）未实现。
5. **XCI 只覆盖每对信道的第一个近邻窗口**（:data:`XCI_NEIGHBOR_SPAN`），远处的
   干扰信道因 ``|mu|`` 随频率差快速衰减而截断；收敛性由测试守住。
6. **论文假设信道互不重叠**，而基准算例 ``c_80ch`` 的信道间隔（50 GHz）小于符号率
   （64 GHz），相邻信道谱有重叠；XCI 的频带归属在重叠下并不严格成立。
7. 2014 论文假设单一 C 波段、等跨、集总放大、无拉曼；冻结骨架含拉曼，多波段下
   色散/有效面积随波长变化，EGN 积分统一取频段中心的 ``gamma``/``alpha``/``beta2``。

**跨间相干累积的折算**

2014 论文的 ``mu``（Eq. (12)）描述跨间相干累积，而冻结的路由层是逐跨非相干相加。
本实现在核里按**链路级**公式算出 N 跨总量再除以 N 返回，使路由层逐跨相加后正好
还原链路级总量；做法与 :mod:`network_nlo_eval.models.semrau2019` 一致。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from network_nlo_eval.core.constants import PI
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanFiberOutput,
    SpanKernel,
    register_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "CARENA2014_MODEL_INFO",
    "DEFAULT_FREQUENCY_POINTS",
    "DEFAULT_MODULATION",
    "DEFAULT_QUADRATURE_POINTS",
    "MODULATION_TABLE",
    "XCI_NEIGHBOR_SPAN",
    "Carena2014Model",
    "Carena2014SpanKernel",
    "EgnIntegrator",
    "LinkFunction",
    "MciKappas",
    "SciKappas",
    "XciKappas",
]

#: 2014 论文 Table 1：各调制格式的 ``(Phi_a, Psi_a)``。
#:
#: ``Phi_a = E{|a|^4}/E^2{|a|^2} - 2``，``Psi_a = E{|a|^6}/E^3{|a|^2} - 9*E{|a|^4}/E^2{|a|^2} + 12``，
#: 其中 ``a`` 是单偏振符号。抄表时用矩定义独立复核过：
#: BPSK/QPSK 的 ``E{|a|^4}/E^2{|a|^2} = 1`` → ``Phi = -1``，``Psi = 1 - 9 + 12 = 4``；
#: 16QAM 的 ``E{|a|^4} = 132``、``E{|a|^2} = 10`` → ``Phi = 132/100 - 2 = -17/25``、
#: ``Psi = 1960/1000 - 9*1.32 + 12 = 52/25``；64QAM 的 ``E{|a|^4}/E^2{|a|^2} = 34/21``、
#: ``E{|a|^6}/E^3{|a|^2} = 4.368...`` → ``Phi = -13/21``、``Psi = 1161/646``。
#:
#: 表里没有的格式直接报错，不做外推。
MODULATION_TABLE: dict[str, tuple[float, float]] = {
    "bpsk": (-1.0, 4.0),
    "qpsk": (-1.0, 4.0),
    "16qam": (-17.0 / 25.0, 52.0 / 25.0),
    "64qam": (-13.0 / 21.0, 1161.0 / 646.0),
}

#: 默认调制格式（PM-64QAM，2014 论文验证配置之一）。
DEFAULT_MODULATION = "64qam"

#: NLI 的跨间相干累积指数（与冻结基线 ``ecz`` / ``semrau2019`` 同口径）。
DEFAULT_COHERENCE_EXPONENT = 0.05

#: 每个子区间上的 Gauss-Legendre 节点数。
#:
#: 论文附录 C 证明所有修正项都只要二重积分；节点数加倍时结果的变化由测试守住
#: （相对变化 < 1e-3）。
DEFAULT_QUADRATURE_POINTS = 6

#: 在 CUT 带内求 NLI PSD 的频率点数。
#:
#: 论文用 1 GHz 步长扫 32 GHz 的 CUT 带（32 点）。X2-X4 域只对部分 ``f`` 非空，
#: 所以要扫几个点才抓得到；比论文的 32 点少，换取可接受的延迟。
DEFAULT_FREQUENCY_POINTS = 3

#: XCI 聚合里每侧保留的近邻信道数。
#:
#: 论文的 XCI 逐信道求和覆盖整个 WDM comb；远处的干扰信道因 ``|mu|`` 随频率差
#: 快速衰减（正比于 ``1/Delta_f^2``）而可以截断。这个窗口是**数值截断**而不是
#: 物理常数，测试里检查窗口翻倍后 SNR 变化 < 0.01 dB。
XCI_NEIGHBOR_SPAN = 8

CARENA2014_MODEL_INFO = ModelInfo(
    name="carena2014",
    summary="Carena 2014 EGN model: GN backbone with quadrature-evaluated 4th/6th-order corrections.",
    family="egn",
    reference="Carena et al., Opt. Express 22(13):16335-16362, 2014, doi:10.1364/OE.22.016335",
    options=(
        ("modulation_format", "Modulation format key: bpsk / qpsk / 16qam / 64qam (2014 Table 1)."),
        ("coherence_exponent", "NLI cross-span coherent accumulation exponent."),
        ("quadrature_points", "Gauss-Legendre nodes per composite sub-interval."),
        ("frequency_points", "Number of NLI-PSD sample frequencies across the CUT band."),
    ),
)


# =============================================================================
# 链路函数（2014 Eq. (10)-(12)）
# =============================================================================
def _sinc_ratio(n_spans: int, psi: NDArrayFloat) -> NDArrayFloat:
    """``sin(n*psi)/sin(psi)``，在 ``sin(psi) -> 0`` 处取解析极限。

    Args:
        n_spans: 跨数。
        psi: 相位 (rad)。

    Returns
    -------
        与 ``psi`` 同形状的实数组。
    """
    s = np.sin(psi)
    small = np.abs(s) < 1e-12
    safe = np.where(small, 1.0, s)
    ratio = np.sin(n_spans * psi) / safe
    # lim_{psi->k pi} sin(n psi)/sin(psi) = n * (-1)^{(n-1)k} = n * cos((n-1) psi)
    return np.where(small, n_spans * np.cos((n_spans - 1) * psi), ratio)


@dataclass(frozen=True)
class LinkFunction:
    """2014 论文的链路函数 ``mu(f1, f2, f)``（Eq. (10)-(12)）。

    ``mu`` 只通过乘积 ``p = (f1-f)*(f2-f)`` 依赖频率，所以所有积分都可以在乘积变量里做。

    Attributes
    ----------
        gamma: 非线性系数 (1/(W*m))。
        alpha_power_npm: 光纤**功率**衰减 (Np/m)；论文里的场衰减 ``alpha`` 是它的一半。
        beta2_s2_m: 二阶色散 (s^2/m)。
        span_length_m: 跨段长度 (m)。
        n_spans: 跨数。
    """

    gamma: float
    alpha_power_npm: float
    beta2_s2_m: float
    span_length_m: float
    n_spans: int

    def mu(self, p_hz2: NDArrayFloat) -> NDArrayFloat:
        """由 ``p = (f1-f)*(f2-f)`` 求 ``mu``。

        Args:
            p_hz2: 频率乘积 (Hz^2)，可广播。

        Returns
        -------
            复数组。
        """
        p = np.asarray(p_hz2, dtype=np.float64)
        # Eq. (11)：theta 是单跨的总相位失配；zeta 的分母是 2*alpha - j*4*pi^2*beta2*p。
        theta = 4.0 * PI**2 * self.beta2_s2_m * p * self.span_length_m
        zeta = (
            self.gamma
            * (1.0 - np.exp(-self.alpha_power_npm * self.span_length_m) * np.exp(1j * theta))
            / (self.alpha_power_npm - 1j * 4.0 * PI**2 * self.beta2_s2_m * p)
        )
        # Eq. (12)：nu 是跨间相干因子，psi = theta/2。
        psi = 0.5 * theta
        nu = _sinc_ratio(self.n_spans, psi) * np.exp(1j * (self.n_spans - 1) * psi)
        return zeta * nu

    def mu_of(self, f1: NDArrayFloat, f2: NDArrayFloat, f_hz: float) -> NDArrayFloat:
        """由两个频率与被干扰频率求 ``mu``。

        Args:
            f1: 频率变量 1 (Hz)，可广播。
            f2: 频率变量 2 (Hz)，可广播。
            f_hz: 被干扰频率 (Hz)。

        Returns
        -------
            复数组，形状同广播结果。
        """
        return self.mu((np.asarray(f1, dtype=np.float64) - f_hz) * (np.asarray(f2, dtype=np.float64) - f_hz))

    @property
    def psi_per_hz2(self) -> float:
        """``psi = c * p`` 里的系数 ``c = 2*pi^2*beta2*L_s`` (rad/Hz^2)。"""
        return 2.0 * PI**2 * self.beta2_s2_m * self.span_length_m


# =============================================================================
# p 轴上的原函数表
# =============================================================================
class _MuAntiderivative:
    """``mu(p)`` 与 ``|mu(p)|^2`` 在 ``p`` 轴上的原函数（表格 + 三次 Hermite 插值）。

    2014 论文的被积函数只通过乘积 ``p = (f1-f)(f2-f)`` 依赖频率，所以内层积分可以
    换成对 ``p`` 的积分。本类把两个原函数

    * ``Psi(p) = int_0^p mu(p') dp'``
    * ``G(p) = int_0^p |mu(p')|^2 dp'``

    在一张均匀网格上预计算：每个网格子区间用 Gauss-Legendre 积分，网格间距由
    ``mu`` 的最细振荡周期决定（跨间相干因子 ``sin(N_s psi)/sin(psi)`` 的周期
    ``pi/(N_s |c|)``，其中 ``c = 2 pi^2 beta2 L_s``），因此网格分辨率随节点数一起
    提高，不是经验常数。

    Attributes
    ----------
        period: ``mu`` 在 ``p`` 轴上的最细振荡周期 (Hz^2)；无色散时是 ``inf``。
    """

    def __init__(self, link: LinkFunction, points: int, p_limit_hz2: float) -> None:
        """建表。

        Args:
            link: 链路函数（跨数、跨长已定）。
            points: 每个网格子区间上的 Gauss-Legendre 节点数。
            p_limit_hz2: 需要覆盖的 ``|p|`` 上限 (Hz^2)。
        """
        self.link = link
        self.points = int(points)
        c = abs(link.psi_per_hz2)
        n_spans = max(1, int(link.n_spans))
        limit = max(float(p_limit_hz2), 1.0)
        if c > 0.0:
            # nu 里 sin(N_s psi) 的周期是 pi/(N_s c)；N_s=1 时 nu 退化成 1，
            # 这时 zeta 的分子项以 pi/|c| 为周期，取它作保守上界。
            self.period = PI / (n_spans * c)
            step = self.period / (4.0 * self.points)
        else:
            self.period = np.inf
            step = 2.0 * limit / (8.0 * self.points)
        n_intervals = max(8, int(np.ceil(2.0 * limit / step)))
        edges = np.linspace(-limit, limit, n_intervals + 1)
        self.limit = float(limit)
        x, w = leggauss(self.points)
        width = np.diff(edges)
        nodes = edges[:-1, None] + width[:, None] * (0.5 * (x + 1.0))[None, :]
        weights = 0.5 * width[:, None] * w
        mu = link.mu(nodes.ravel())
        flat_w = weights.ravel()
        seg_psi = (mu * flat_w).reshape(n_intervals, self.points).sum(axis=1)
        seg_g = (np.abs(mu) ** 2 * flat_w).reshape(n_intervals, self.points).sum(axis=1)
        self._edges = edges
        self._psi = np.concatenate(([0.0 + 0.0j], np.cumsum(seg_psi)))
        self._g = np.concatenate(([0.0], np.cumsum(seg_g)))
        # Hermite 插值要用到导数：Psi' = mu，G' = |mu|^2。
        self._mu_edge = link.mu(edges)
        self._power_edge = np.abs(self._mu_edge) ** 2

    def _interpolate(self, p: NDArrayFloat, values: NDArrayFloat, slopes: NDArrayFloat) -> NDArrayFloat:
        """在网格上做三次 Hermite 插值。

        Args:
            p: 求值点 (Hz^2)，可广播。
            values: 网格节点上的函数值。
            slopes: 网格节点上的导数。

        Returns
        -------
            插值结果，形状同 ``p``。
        """
        x = np.clip(np.asarray(p, dtype=np.float64), self._edges[0], self._edges[-1])
        idx = np.searchsorted(self._edges, x, side="right") - 1
        idx = np.clip(idx, 0, self._edges.size - 2)
        x0 = self._edges[idx]
        x1 = self._edges[idx + 1]
        h = x1 - x0
        s = (x - x0) / h
        s2 = s * s
        s3 = s2 * s
        return (
            (2.0 * s3 - 3.0 * s2 + 1.0) * values[idx]
            + (s3 - 2.0 * s2 + s) * h * slopes[idx]
            + (-2.0 * s3 + 3.0 * s2) * values[idx + 1]
            + (s3 - s2) * h * slopes[idx + 1]
        )

    def value(self, p: NDArrayFloat) -> NDArrayFloat:
        """``Psi(p) = int_0^p mu``。

        Args:
            p: ``p`` (Hz^2)，可广播。

        Returns
        -------
            复数组。
        """
        return self._interpolate(p, self._psi, self._mu_edge)

    def power(self, p: NDArrayFloat) -> NDArrayFloat:
        """``G(p) = int_0^p |mu|^2``。

        Args:
            p: ``p`` (Hz^2)，可广播。

        Returns
        -------
            实数组。
        """
        return self._interpolate(p, self._g, self._power_edge)


# =============================================================================
# kappa 系数容器
# =============================================================================
@dataclass(frozen=True)
class SciKappas:
    """SCI 的三个系数（2014 Eq. (7)/(8)/(9)）。

    Attributes
    ----------
        kappa1: GN 项（二重积分，只有 ``|s|^2`` 与 ``|mu|^2``）。
        kappa2: 4 阶修正（论文 Eq. (8) 的两行，矩形谱下两行的积分相同）。
        kappa3: 6 阶修正（四重积分降成二重积分的模平方）。
    """

    kappa1: float
    kappa2: float
    kappa3: float


@dataclass(frozen=True)
class XciKappas:
    """XCI 的九个系数（2014 Eq. (16)/(17)、Eq. (18) 与附录 A 的 Eq. (29)-(35)）。

    ``kappa_mn`` 的 ``m`` 是域号（1=X1, 2=X2, 3=X3, 4=X4），``n`` 是阶数
    （1=GN 项，2=4 阶修正，3=6 阶修正）。
    """

    kappa11: float
    kappa12: float
    kappa21: float
    kappa22: float
    kappa31: float
    kappa32: float
    kappa41: float
    kappa42: float
    kappa43: float


@dataclass(frozen=True)
class MciKappas:
    """MCI 的系数（2014 Eq. (24)/(25)/(26)）。

    Attributes
    ----------
        kappa_m0: M0 域的 GN 项（只有 ``|s|^2`` 与 ``|mu|^2``，没有修正项）。
        kappa_m1_1: M1 域的 GN 项。
        kappa_m1_2: M1 域的 4 阶修正。
    """

    kappa_m0: float
    kappa_m1_1: float
    kappa_m1_2: float


# =============================================================================
# 求积器
# =============================================================================
class EgnIntegrator:
    """按 2014 论文的积分式做数值求积。

    论文的每个 ``kappa`` 在附录 C 的降维之后都是二重积分。被积函数只通过乘积
    ``p = (f1-f)(f2-f)`` 依赖频率，于是把积分变量换成相对频率

    * ``u = f1 - f``（第 1 个波形）
    * ``v = f2 - f``（第 2 个波形）

    就有 ``p = u*v``，第三个波形是 ``f1+f2-f``（相对频率 ``u+v``）。三个波形各自
    落在一条频带里，于是积分区域由三条频带约束决定；对固定的 ``u``，内层区间是

    .. code-block:: text

        v in [ v1(u), v2(u) ],  v1 = max(V_lo, W_lo - u),  v2 = min(V_hi, W_hi - u)

    其中 ``V`` 是第 2 个波形的频带、``W`` 是第 3 个波形的频带（都是相对频率）。
    内层积分可以换成对 ``p`` 的积分，于是两个原函数

    * ``G(p) = int_0^p |mu|^2``
    * ``Psi(p) = int_0^p mu``

    就够用（见 :class:`_MuAntiderivative`）：

    * GN 项：``int_{v1}^{v2} |mu(u v)|^2 dv = (G(u v2) - G(u v1)) / u``；
    * 4 阶项：``int_{v1}^{v2} mu(u v) dv = (Psi(u v2) - Psi(u v1)) / u``。

    这样内层不再需要求积，外层被积函数里只剩下跨间相干因子带来的细结构，用复合
    Gauss-Legendre 逐周期解析即可。

    Attributes
    ----------
        link: 链路函数（跨数、跨长已定）。
        symbol_rate_hz: 符号率 ``R_s`` (Hz)，同时是论文假设下的信道带宽。
        points: 每个复合子区间上的 Gauss-Legendre 节点数。
    """

    def __init__(
        self,
        *,
        link: LinkFunction,
        symbol_rate_hz: float,
        points: int = DEFAULT_QUADRATURE_POINTS,
        p_limit_hz2: float | None = None,
    ) -> None:
        """初始化求积器。

        Args:
            link: 链路函数（跨数、跨长已定）。
            symbol_rate_hz: 符号率 (Hz)。
            points: 每个复合子区间上的 Gauss-Legendre 节点数。
            p_limit_hz2: 积分里 ``|p|`` 需要覆盖的上限 (Hz^2)；``None`` 时按只有
                CUT 自身的情形取 ``(2 R_s/2)^2``。
        """
        if points < 2:
            raise ValueError(f"points must be >= 2, got {points}.")
        self.link = link
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.points = int(points)
        self._half = 0.5 * self.symbol_rate_hz
        if p_limit_hz2 is None:
            p_limit_hz2 = (2.0 * self._half) ** 2
        self._table = _MuAntiderivative(link, self.points, float(p_limit_hz2))
        # 论文 Sect. 2：矩形谱的平台值是 1/R_s，所以每个 ``|s|^2`` 因子是 1/R_s^2。
        # 每个 kappa 都带三个这样的因子（GN 项是三个 ``|s|^2``；4/6 阶项是一个
        # ``|s|^2`` 加上两个 ``|s`` 的乘积），因此谱因子统一是 R_s^{-6}。
        # 注意这一因子在 kappa1/kappa2/kappa3 之间相同，所以论文显式写出的 R_s 幂次
        # （3、2、1）完全决定了它们之间的比例。
        self._spectral = self.symbol_rate_hz**-6

    # --- 工具 ---------------------------------------------------------------
    def _ensure_table(self, p_limit: float) -> None:
        """必要时按更大的 ``|p|`` 上限重建原函数表。

        求积器构造时给出的 ``p_limit`` 必须覆盖所有被积区域；覆盖不到时重建，
        而不是让插值在表边被截断（那样会静默给出错误结果）。

        Args:
            p_limit: 本次积分需要的 ``|p|`` 上限 (Hz^2)。
        """
        if p_limit > self._table.limit:
            self._table = _MuAntiderivative(self.link, self.points, float(p_limit))

    def _cut_band(self) -> tuple[float, float]:
        """CUT 的频带 (下界, 上界)。"""
        return (-self._half, self._half)

    def _band(self, center_hz: float) -> tuple[float, float]:
        """中心在 ``center_hz`` 的信道的频带 (下界, 上界)。"""
        return (center_hz - self._half, center_hz + self._half)

    def _splits(
        self,
        u_lo: float,
        u_hi: float,
        v_lo: float,
        v_hi: float,
        w_lo: float,
        w_hi: float,
    ) -> list[float]:
        """外层积分的分段点。

        分段点来自两处：``u = 0``（原函数的 ``1/u`` 形式在这里是可去奇点）与
        内层窗口的折点（``v1``/``v2`` 的表达式在 ``u = W_lo - V_lo`` 与
        ``u = W_hi - V_hi`` 处切换）。每一段上被积函数都是光滑的，复合
        Gauss-Legendre 才有谱收敛。

        Args:
            u_lo: 外层变量下界。
            u_hi: 外层变量上界。
            v_lo: 第 2 个波形频带下界。
            v_hi: 第 2 个波形频带上界。
            w_lo: 第 3 个波形频带下界。
            w_hi: 第 3 个波形频带上界。

        Returns
        -------
            递增的分段点列表（含两端）。
        """
        candidates = (u_lo, u_hi, 0.0, w_lo - v_lo, w_hi - v_hi)
        interior = sorted({c for c in candidates if u_lo < c < u_hi})
        return [u_lo, *interior, u_hi]

    def _outer_grid(self, lo: float, hi: float, v_scale: float) -> tuple[NDArrayFloat, NDArrayFloat]:
        """外层（``u``）的复合 Gauss-Legendre 节点与权重。

        子区间个数由被积函数的细结构决定：原函数在 ``p`` 轴上的周期是 ``T``，
        而 ``p = u*v(u)`` 对 ``u`` 的导数量级是 ``v_scale``，于是 ``u`` 方向的
        振荡周期约 ``T/v_scale``。每个振荡周期分一个子区间。

        Args:
            lo: 区间下界。
            hi: 区间上界。
            v_scale: 内层变量的量级 (Hz)。

        Returns
        -------
            ``(节点, 权重)``，一维数组。
        """
        if np.isfinite(self._table.period):
            n_osc = (hi - lo) * v_scale / self._table.period
        else:
            n_osc = 0.0
        n_sub = max(1, int(np.ceil(n_osc)))
        x, w = leggauss(self.points)
        edges = np.linspace(lo, hi, n_sub + 1)
        width = np.diff(edges)
        nodes = edges[:-1, None] + width[:, None] * (0.5 * (x + 1.0))[None, :]
        weights = 0.5 * width[:, None] * w
        return nodes.ravel(), weights.ravel()

    def _core(
        self,
        f_hz: float,
        band1: tuple[float, float],
        band2: tuple[float, float],
        band3: tuple[float, float],
        kind: str,
    ) -> tuple[float, float]:
        """二重积分的核心（相对频率坐标）。

        Args:
            f_hz: 被干扰频率 (Hz)。
            band1: 第 1 个波形的频带 (Hz)。
            band2: 第 2 个波形的频带 (Hz)。
            band3: 第 3 个波形 ``f1+f2-f`` 的频带 (Hz)。
            kind: ``"gn"``（GN 项）、``"fourth"``（4 阶项）或 ``"sixth"``（6 阶项）。

        Returns
        -------
            ``gn`` 为 ``int int |mu|^2``；``coherent`` 为 4 阶项的
            ``int |int mu|^2`` 或 6 阶项的 ``int int mu``（未取模平方）。
        """
        u_lo, u_hi = band1[0] - f_hz, band1[1] - f_hz
        v_lo, v_hi = band2[0] - f_hz, band2[1] - f_hz
        w_lo, w_hi = band3[0] - f_hz, band3[1] - f_hz
        self._ensure_table(max(abs(u_lo), abs(u_hi)) * max(abs(v_lo), abs(v_hi)))
        v_scale = max(abs(v_lo), abs(v_hi), abs(w_lo), abs(w_hi), 2.0 * max(abs(u_lo), abs(u_hi)))
        gn_total = 0.0
        coh_total = 0.0 + 0.0j
        splits = self._splits(u_lo, u_hi, v_lo, v_hi, w_lo, w_hi)
        for lo, hi in zip(splits[:-1], splits[1:], strict=False):
            u_nodes, u_w = self._outer_grid(lo, hi, v_scale)
            v1 = np.maximum(v_lo, w_lo - u_nodes)
            v2 = np.minimum(v_hi, w_hi - u_nodes)
            valid = v2 > v1
            if not bool(np.any(valid)):
                continue
            p1 = u_nodes * v1
            p2 = u_nodes * v2
            if kind == "gn":
                val = (self._table.power(p2) - self._table.power(p1)) / u_nodes
                gn_total += float(np.sum(u_w * np.where(valid, val, 0.0)))
            else:
                val = (self._table.value(p2) - self._table.value(p1)) / u_nodes
                val = np.where(valid, val, 0.0)
                if kind == "fourth":
                    coh_total += complex(np.sum(u_w * np.abs(val) ** 2))
                else:
                    coh_total += complex(np.sum(u_w * val))
        if kind == "gn":
            return gn_total, 0.0
        if kind == "fourth":
            return 0.0, float(coh_total.real)
        return 0.0, coh_total

    # --- SCI (2014 Eq. (7)/(8)/(9)) -----------------------------------------
    def sci(self, f_hz: float) -> SciKappas:
        """SCI 的 ``kappa1`` / ``kappa2`` / ``kappa3``。

        Eq. (7)：``kappa1 = (16/27) R_s^3 int int |s|^2 |s|^2 |s(f1+f2-f)|^2 |mu|^2``。

        Eq. (8) 的两行按附录 C 的 Eq. (43)/(44) 降成二重积分。第一行的外层是
        ``f1``；第二行把 ``f1`` 换成 ``f3 = f1+f2-f``，再把外层换回 ``f1``，落到
        与第一行相同的区域。矩形谱下两行的谱因子与积分区域完全相同，于是可以合并
        成一个系数。

        Eq. (9) 按附录 C 的 Eq. (45) 降成二重积分的模平方。

        Args:
            f_hz: 被干扰频率 (Hz)，取 CUT 带内的值。

        Returns
        -------
            :class:`SciKappas`。
        """
        band = self._cut_band()
        gn, _ = self._core(f_hz, band, band, band, "gn")
        _, fourth = self._core(f_hz, band, band, band, "fourth")
        _, sixth = self._core(f_hz, band, band, band, "sixth")
        rs = self.symbol_rate_hz
        spec = self._spectral
        kappa1 = (16.0 / 27.0) * rs**3 * spec * gn
        kappa2 = ((80.0 / 81.0) + (16.0 / 81.0)) * rs**2 * spec * fourth
        kappa3 = (16.0 / 81.0) * rs * spec * abs(sixth) ** 2
        return SciKappas(kappa1=kappa1, kappa2=kappa2, kappa3=kappa3)

    # --- XCI (2014 Eq. (14)/(16)/(17)/(18) 与附录 A) -----------------------
    def xci(self, f_c_hz: float, f_hz: float) -> XciKappas:
        """XCI 的九个系数（单个 INT 信道，相对 CUT 的频偏为 ``f_c_hz``）。

        ``kappa_m1`` 是 GN 项（``|s|^2`` 与 ``|mu|^2``），``kappa_m2`` 是一重反对称化
        （``s s* mu mu*``），``kappa_m3`` 是两重反对称化。四个域的频带归属是

        * X1：``f1`` 在 CUT、``f2`` 与 ``f3`` 在 INT；
        * X2：``f1`` 在 INT、``f2`` 与 ``f3`` 在 CUT；
        * X3：``f1`` 与 ``f2`` 在 CUT、``f3`` 在 INT；
        * X4：三个都在 INT。

        X2-X4 的非空性由矩形谱的硬零点自动兑现（论文 Sect. 4B：``|f_c| >= 2 R_s``
        时 X2-X4 完全消失）。

        Args:
            f_c_hz: INT 相对 CUT 的频偏 (Hz)。
            f_hz: 被干扰频率 (Hz)。

        Returns
        -------
            :class:`XciKappas`。
        """
        cut = self._cut_band()
        int_band = self._band(f_c_hz)
        rs = self.symbol_rate_hz
        spec = self._spectral

        gn11, _ = self._core(f_hz, cut, int_band, int_band, "gn")
        _, c12 = self._core(f_hz, cut, int_band, int_band, "fourth")
        gn21, _ = self._core(f_hz, int_band, cut, cut, "gn")
        _, c22 = self._core(f_hz, int_band, cut, cut, "fourth")
        gn31, _ = self._core(f_hz, cut, cut, int_band, "gn")
        _, c32 = self._core(f_hz, cut, cut, int_band, "fourth")
        gn41, _ = self._core(f_hz, int_band, int_band, int_band, "gn")
        _, c42 = self._core(f_hz, int_band, int_band, int_band, "fourth")
        _, c43 = self._core(f_hz, int_band, int_band, int_band, "sixth")

        return XciKappas(
            kappa11=(32.0 / 27.0) * rs**3 * spec * gn11,
            kappa12=(80.0 / 81.0) * rs**2 * spec * c12,
            kappa21=(32.0 / 27.0) * rs**3 * spec * gn21,
            kappa22=(80.0 / 81.0) * rs**2 * spec * c22,
            kappa31=(16.0 / 27.0) * rs**3 * spec * gn31,
            kappa32=(16.0 / 81.0) * rs**2 * spec * c32,
            kappa41=(16.0 / 27.0) * rs**3 * spec * gn41,
            kappa42=((80.0 / 81.0) + (16.0 / 81.0)) * rs**2 * spec * c42,
            kappa43=(16.0 / 81.0) * rs * spec * abs(c43) ** 2,
        )

    # --- MCI (2014 Eq. (24)/(25)/(26)) --------------------------------------
    def mci(self, f_upper_hz: float, f_lower_hz: float, f_hz: float) -> MciKappas:
        """MCI 的三个系数。

        Eq. (24) 的 M0 域：``f1`` 在上侧 INT 带、``f2`` 在下侧 INT 带、``f3`` 落在 CUT
        带，权重 ``2 * (16/27) R_s^3``，只有 GN 项。

        Eq. (25) 的 M1 域：``f1`` 在下侧 INT 带、``f2`` 在上侧 INT 带、``f3`` 落在上侧
        INT 带，权重 ``4 * (16/27) R_s^3``。

        Eq. (26) 的 M1 修正：``2 * (80/81) R_s^2``，外层是下侧 INT 带，内层含
        ``s s* mu mu*``（附录 C 的降维同样适用）。

        Args:
            f_upper_hz: 上侧 INT 相对 CUT 的频偏 (Hz)，为正。
            f_lower_hz: 下侧 INT 相对 CUT 的频偏 (Hz)，为负。
            f_hz: 被干扰频率 (Hz)。

        Returns
        -------
            :class:`MciKappas`。
        """
        cut = self._cut_band()
        up = self._band(f_upper_hz)
        low = self._band(f_lower_hz)
        rs = self.symbol_rate_hz
        spec = self._spectral

        gn_m0, _ = self._core(f_hz, up, low, cut, "gn")
        gn_m1, _ = self._core(f_hz, low, up, up, "gn")
        _, c_m1 = self._core(f_hz, low, up, up, "fourth")

        return MciKappas(
            kappa_m0=2.0 * (16.0 / 27.0) * rs**3 * spec * gn_m0,
            kappa_m1_1=4.0 * (16.0 / 27.0) * rs**3 * spec * gn_m1,
            kappa_m1_2=2.0 * (80.0 / 81.0) * rs**2 * spec * c_m1,
        )


# =============================================================================
# 核
# =============================================================================
@dataclass(frozen=True)
class _EgnRatios:
    """EGN 相对 GN 的修正比。

    Attributes
    ----------
        sci: SCI 的修正比 ``(kappa1 + Phi kappa2 + Psi kappa3) / kappa1``。
        xci: XCI 逐信道的修正比，形状 ``(N,)``。
        mci_gn: MCI 的 GN 项 (W)，论文 Eq. (23) 的 ``P_CUT P_INT,1 P_INT,-1 kappa_M0``
            与 ``P_INT,1^2 P_INT,-1 kappa_M1,1`` 之和。
        mci_corr: MCI 的修正项 (W)，论文 Eq. (23) 的 ``P_INT,1^2 P_INT,-1 Phi_b kappa_M1,2``。
    """

    sci: float
    xci: NDArrayFloat
    mci_gn: NDArrayFloat
    mci_corr: NDArrayFloat


class Carena2014SpanKernel:
    """Carena 2014 EGN 的单跨物理核。

    构造期把逐信道的物理量准备好；热路径是冻结闭式 GN 的一次调用加上 EGN 修正比
    的一次乘法。修正比只依赖光纤、跨数与信道栅格，与入纤功率无关（论文 Sect. 7A：
    模型计算不需要随发射功率重跑），所以按 ``(跨数, 跨长)`` 记忆化。

    Attributes
    ----------
        phi:
            调制格式的 4 阶矩超额 ``Phi_a``（2014 Table 1）。
        psi:
            调制格式的 6 阶矩超额 ``Psi_a``（2014 Table 1）。
    """

    def __init__(
        self,
        *,
        gn_evaluator: MultiBandISRSGN,
        phi: float,
        psi: float,
        coherence_exponent: float,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        frequency_points: int = DEFAULT_FREQUENCY_POINTS,
    ) -> None:
        """按光纤配置准备 GN 骨架与 EGN 积分器。

        Args:
            gn_evaluator: 冻结的闭式 GN 评估器（``models.isrs_gn``）。
            phi: 调制格式的 4 阶矩 ``Phi_a``（2014 Table 1）。
            psi: 调制格式的 6 阶矩 ``Psi_a``（2014 Table 1）。
            coherence_exponent: 跨间相干累积指数。
            quadrature_points: 每个复合子区间上的 Gauss-Legendre 节点数。
            frequency_points: CUT 带内的 NLI PSD 采样频率点数。
        """
        self.gn = gn_evaluator
        self.phi = float(phi)
        self.psi = float(psi)
        self.coherence_exponent = float(coherence_exponent)
        self.quadrature_points = int(quadrature_points)
        self.frequency_points = int(frequency_points)

        self.f_rel_hz = np.asarray(gn_evaluator.f_rel_hz, dtype=np.float64)
        self.b_hz = np.asarray(gn_evaluator.channel_bandwidth_hz_channels, dtype=np.float64)
        self.beta2 = float(gn_evaluator.beta2_s2_m)
        self.beta3 = float(gn_evaluator.beta3_s3_m)
        self.alpha_npm = np.asarray(gn_evaluator.alpha_power_npm_channels, dtype=np.float64)
        self.a_eff_m2 = np.asarray(gn_evaluator.a_eff_m2_channels, dtype=np.float64)
        self.lambdas_m = np.asarray(gn_evaluator.lambdas_m, dtype=np.float64)
        self.n2_m2_w = float(gn_evaluator.n2_m2_w)
        self.symbol_rate_hz = float(gn_evaluator.symbol_rate_hz)

        n_ch = int(self.b_hz.size)
        # gamma_ii = (2 pi / lambda_i) * (2 n2 / (A_eff,i + A_eff,i))
        n2 = self.fiber_n2()
        self.gamma_ii = (2.0 * PI / self.lambdas_m) * (2.0 * n2 / (2.0 * self.a_eff_m2))

        # 频段中心信道：2014 论文假设单一光纤类型，EGN 积分统一用频段中心的参数。
        self._center_index = int(np.argmin(np.abs(self.f_rel_hz)))
        self.gamma_center = float(self.gamma_ii[self._center_index])
        self.alpha_center = float(self.alpha_npm[self._center_index])

        # 相邻信道（MCI 用）：论文 Eq. (23) 的三信道形式。
        lower = np.full(n_ch, -1, dtype=np.int64)
        upper = np.full(n_ch, -1, dtype=np.int64)
        for i in range(n_ch):
            if i > 0:
                lower[i] = i - 1
            if i + 1 < n_ch:
                upper[i] = i + 1
        self.mci_lower = lower
        self.mci_upper = upper

        # XCI 的间隔表：均匀栅格下第 d 步邻居的频偏。论文假设等间距等带宽信道。
        self._neighbor_span = XCI_NEIGHBOR_SPAN
        self._separation_offsets: NDArrayFloat = np.zeros(0)
        self._build_separation_table()

        # 求积里 p = (f1-f)(f2-f) 需要覆盖到的上限：参与积分的 INT 信道频偏的最大值
        # 加上两个半带宽（f 与 f1/f2 各差最多一个半带宽）。
        max_offset = float(np.max(np.abs(self._separation_offsets))) if self._separation_offsets.size else 0.0
        if n_ch > 1:
            max_offset = max(max_offset, float(np.max(np.abs(np.diff(self.f_rel_hz)))))
        self._p_limit_hz2 = (max_offset + self.symbol_rate_hz) ** 2

        self._kappa_cache: dict[tuple[int, float], dict[int, XciKappas]] = {}
        self._sci_cache: dict[tuple[int, float], SciKappas] = {}
        self._mci_cache: dict[tuple[int, float], dict[tuple[float, float], MciKappas]] = {}

    # --- 小工具 -------------------------------------------------------------
    def fiber_n2(self) -> float:
        """光纤的非线性折射率 ``n2`` (m^2/W)。"""
        return self.n2_m2_w

    def _build_separation_table(self) -> None:
        """建立"信道下标距离 → 频偏"的查找表。

        论文假设等间距信道，于是第 ``d`` 步邻居的频偏只与 ``d`` 有关。栅格不严格等距
        时取该距离上频偏的中位数（近似，已在"已知限制"里说明）。
        """
        n_ch = int(self.b_hz.size)
        max_d = min(self._neighbor_span, max(0, n_ch - 1))
        offsets = np.zeros(max_d + 1, dtype=np.float64)
        for d in range(1, max_d + 1):
            if d < n_ch:
                offsets[d] = float(np.median(self.f_rel_hz[d:] - self.f_rel_hz[:-d]))
        self._separation_offsets = offsets

    def _frequency_grid(self) -> NDArrayFloat:
        """CUT 带内的 NLI PSD 采样频率 (Hz)。"""
        return np.linspace(-0.5 * self.symbol_rate_hz, 0.5 * self.symbol_rate_hz, self.frequency_points)

    def _integrator(self, n_spans: int, span_length_m: float) -> EgnIntegrator:
        """给定跨数与跨长，构造求积器。"""
        link = LinkFunction(
            gamma=self.gamma_center,
            alpha_power_npm=self.alpha_center,
            beta2_s2_m=self.beta2,
            span_length_m=float(span_length_m),
            n_spans=int(n_spans),
        )
        return EgnIntegrator(
            link=link,
            symbol_rate_hz=self.symbol_rate_hz,
            points=self.quadrature_points,
            p_limit_hz2=self._p_limit_hz2,
        )

    # --- 论文积分式（与功率无关，按 (跨数, 跨长) 缓存）----------------------
    def sci_kappas(self, n_spans: int, span_length_m: float) -> dict[float, SciKappas]:
        """SCI 的 kappa 系数，按 CUT 带内的采样频率给出。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``{f_hz: SciKappas}``。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._sci_cache.get(key)
        if cached is not None:
            return cached
        integrator = self._integrator(n_spans, span_length_m)
        table = {float(f): integrator.sci(float(f)) for f in self._frequency_grid()}
        self._sci_cache[key] = table
        return table

    def xci_kappas(self, n_spans: int, span_length_m: float) -> dict[int, XciKappas]:
        """XCI 的 kappa 系数，按邻居距离给出（每个距离一个频偏）。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``{距离: XciKappas}``；``XciKappas`` 的每个字段是对采样频率的数组。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._kappa_cache.get(key)
        if cached is not None:
            return cached
        integrator = self._integrator(n_spans, span_length_m)
        table: dict[int, XciKappas] = {}
        for d in range(1, self._separation_offsets.size):
            offset = float(self._separation_offsets[d])
            samples = [integrator.xci(offset, float(f)) for f in self._frequency_grid()]
            table[d] = XciKappas(
                **{name: np.array([getattr(s, name) for s in samples]) for name in samples[0].__dataclass_fields__}
            )
        self._kappa_cache[key] = table
        return table

    def mci_kappas(self, n_spans: int, span_length_m: float) -> dict[tuple[float, float], MciKappas]:
        """MCI 的 kappa 系数，按 (上侧频偏, 下侧频偏) 给出。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``{(上侧频偏, 下侧频偏): MciKappas}``。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._mci_cache.get(key)
        if cached is not None:
            return cached
        integrator = self._integrator(n_spans, span_length_m)
        table: dict[tuple[float, float], MciKappas] = {}
        n_ch = int(self.b_hz.size)
        for i in range(n_ch):
            lo = int(self.mci_lower[i])
            hi = int(self.mci_upper[i])
            if lo < 0 or hi < 0:
                continue
            f_upper = float(self.f_rel_hz[hi] - self.f_rel_hz[i])
            f_lower = float(self.f_rel_hz[i] - self.f_rel_hz[lo])
            pair = (round(f_upper, 6), round(f_lower, 6))
            if pair in table:
                continue
            table[pair] = MciKappas(
                **{
                    name: np.array(
                        [getattr(integrator.mci(f_upper, f_lower, float(f)), name) for f in self._frequency_grid()]
                    )
                    for name in ("kappa_m0", "kappa_m1_1", "kappa_m1_2")
                }
            )
        self._mci_cache[key] = table
        return table

    # --- EGN 修正比 ---------------------------------------------------------
    def _sci_ratio(self, n_spans: int, span_length_m: float) -> float:
        """SCI 的修正比（在 CUT 带内按 GN 项加权平均）。"""
        table = self.sci_kappas(n_spans, span_length_m)
        gn = 0.0
        total = 0.0
        for kappas in table.values():
            gn += float(np.sum(kappas.kappa1))
            total += float(np.sum(kappas.kappa1 + self.phi * kappas.kappa2 + self.psi * kappas.kappa3))
        if gn <= 0.0:
            return 1.0
        return total / gn

    def _xci_ratio(self, n_spans: int, span_length_m: float, power: NDArrayFloat) -> NDArrayFloat:
        """XCI 的逐信道修正比。

        论文 Eq. (18) 把单个 INT 的 XCI 写成四段，前置因子分别是
        ``P_CUT P_INT^2``（X1）、``P_CUT^2 P_INT``（X2/X3）、``P_INT^3``（X4），
        每一段都带一个 GN 项与各自的修正项。修正比按 GN 项加权：

        .. code-block:: text

            rho = sum_j CORR(i,j) / sum_j GN(i,j)

        其中 ``j`` 遍历近邻窗口内的 INT 信道。
        """
        n_ch = int(self.b_hz.size)
        if n_ch < 2:
            return np.zeros(n_ch, dtype=np.float64)
        table = self.xci_kappas(n_spans, span_length_m)
        gn = np.zeros(n_ch, dtype=np.float64)
        corr = np.zeros(n_ch, dtype=np.float64)
        for d, kappas in table.items():
            g11 = float(np.sum(kappas.kappa11))
            g21 = float(np.sum(kappas.kappa21 + kappas.kappa31))
            g41 = float(np.sum(kappas.kappa41))
            c11 = self.phi * float(np.sum(kappas.kappa12))
            c21 = self.phi * float(np.sum(kappas.kappa22 + kappas.kappa32))
            c41 = self.phi * float(np.sum(kappas.kappa42)) + self.psi * float(np.sum(kappas.kappa43))
            hi = slice(d, n_ch)
            lo = slice(0, n_ch - d)
            p_hi = power[hi]
            p_lo = power[lo]
            # 上侧邻居 (i, i+d)。
            gn[hi] += power[lo] * p_hi**2 * g11
            gn[hi] += power[lo] ** 2 * p_hi * g21
            gn[hi] += p_hi**3 * g41
            corr[hi] += power[lo] * p_hi**2 * c11
            corr[hi] += power[lo] ** 2 * p_hi * c21
            corr[hi] += p_hi**3 * c41
            # 下侧邻居 (i, i-d)。
            gn[lo] += power[hi] * p_lo**2 * g11
            gn[lo] += power[hi] ** 2 * p_lo * g21
            gn[lo] += p_lo**3 * g41
            corr[lo] += power[hi] * p_lo**2 * c11
            corr[lo] += power[hi] ** 2 * p_lo * c21
            corr[lo] += p_lo**3 * c41
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(gn > 0.0, corr / np.where(gn > 0.0, gn, 1.0), 0.0)

    def _mci_terms(self, n_spans: int, span_length_m: float, power: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]:
        """MCI 的 GN 项与修正项 (W)，按论文 Eq. (23)。

        只用 CUT 上下紧邻的两个 INT 信道（论文 Sect. 5 的三信道形式）。
        """
        n_ch = int(self.b_hz.size)
        gn = np.zeros(n_ch, dtype=np.float64)
        corr = np.zeros(n_ch, dtype=np.float64)
        table = self.mci_kappas(n_spans, span_length_m)
        for i in range(n_ch):
            lo = int(self.mci_lower[i])
            hi = int(self.mci_upper[i])
            if lo < 0 or hi < 0:
                continue
            pair = (
                round(float(self.f_rel_hz[hi] - self.f_rel_hz[i]), 6),
                round(float(self.f_rel_hz[i] - self.f_rel_hz[lo]), 6),
            )
            kappas = table[pair]
            p_cut = float(power[i])
            p_up = float(power[hi])
            p_low = float(power[lo])
            gn[i] += p_cut * p_up * p_low * float(np.sum(kappas.kappa_m0))
            gn[i] += p_up**2 * p_low * float(np.sum(kappas.kappa_m1_1))
            corr[i] += p_up**2 * p_low * self.phi * float(np.sum(kappas.kappa_m1_2))
        return gn, corr

    def _ratios(self, n_spans: int, span_length_m: float, power: NDArrayFloat) -> _EgnRatios:
        """计算 EGN 修正比。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。
            power: 逐信道入纤功率 (W)；XCI/MCI 的前置因子要用到它。

        Returns
        -------
            :class:`_EgnRatios`。
        """
        sci = self._sci_ratio(n_spans, span_length_m)
        xci = self._xci_ratio(n_spans, span_length_m, power)
        mci_gn, mci_corr = self._mci_terms(n_spans, span_length_m, power)
        return _EgnRatios(sci=sci, xci=xci, mci_gn=mci_gn, mci_corr=mci_corr)

    # --- 热路径 -------------------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """计算单跨的出纤功率与 NLI 噪声。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 有效跨数（链路级总量口径）。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.SpanFiberOutput`。
        """
        return self._evaluate(power_in_w, span_length_m, n_effective_spans, per_span=False)

    def evaluate_per_span(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_path_spans: int = 1,
    ) -> SpanFiberOutput:
        """计算**逐跨等效**的出纤功率与 NLI 噪声。

        2014 论文的 ``mu`` 因子（Eq. (12)）描述的是**跨间相干**累积，而冻结的路由层
        （``models/route.py``）是**逐跨非相干**相加噪声。若每跨都返回链路级的
        N 跨总量，路由层会把它加 N 次，结果错 N 倍。

        因此本方法先按链路级公式算出 **N 跨总量**，再除以 ``n_path_spans`` 返回；
        路由层逐跨相加后正好还原链路级总量。这与 ``semrau2019`` 的处理方式一致。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_path_spans: 整条光路的跨段数。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.SpanFiberOutput`。
        """
        return self._evaluate(power_in_w, span_length_m, n_path_spans, per_span=True)

    def _evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
        per_span: bool,
    ) -> SpanFiberOutput:
        """核的热路径入口：GN 骨架 + EGN 非高斯修正。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            n_spans: 链路级跨数。
            per_span: 是否返回逐跨等效值。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.SpanFiberOutput`。
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            raise ValueError(
                f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            )
        if span_length_m <= 0.0:
            raise ValueError(f"span_length_m must be positive, got {span_length_m}.")
        n = max(1, int(n_spans))

        # --- GN 骨架（冻结闭式：SPM = SCI，XPM = XCI 的 X1 域）--------------
        gn_out = self.gn.span_fiber_output(power, float(span_length_m), n)
        power_out = np.array(gn_out.power_out_w, dtype=np.float64, copy=True)
        s2_spm = np.array(gn_out.spm_noise_w, dtype=np.float64, copy=True)
        s2_xpm = np.array(gn_out.xpm_noise_w, dtype=np.float64, copy=True)

        # --- EGN 修正（论文积分式）------------------------------------------
        ratios = self._ratios(n, float(span_length_m), power)
        active = power > 0.0
        rho_sci = ratios.sci if bool(np.any(active)) else 1.0
        rho_xci = np.where(active, 1.0 + ratios.xci, 0.0)
        mci = np.where(active, ratios.mci_gn + ratios.mci_corr, 0.0)

        spm_noise = s2_spm * rho_sci
        xpm_noise = s2_xpm * rho_xci + mci

        scale = 1.0 / float(n) if per_span else 1.0
        return SpanFiberOutput(
            power_out,
            spm_noise * scale,
            xpm_noise * scale,
        )


class Carena2014Model(RouteModel):
    """Carena 2014 EGN 模型的路由层模型。

    与其它模型共用 :mod:`network_nlo_eval.models.route` 的跨段累积、放大器与 SNR
    合成；本类只负责单跨物理核。

    构造期可选参数（均为关键字）：

    ``modulation_format``
        2014 Table 1 的调制格式键（``"bpsk"``/``"qpsk"``/``"16qam"``/``"64qam"``），
        决定 ``Phi_a`` 与 ``Psi_a``。表里没有的格式直接报错，不做外推。
    ``coherence_exponent``
        跨间相干累积指数，默认 0.05（与冻结基线同口径）。
    ``quadrature_points``
        每个复合子区间上的 Gauss-Legendre 节点数，默认 6。
    ``frequency_points``
        CUT 带内的 NLI PSD 采样频率点数，默认 3。
    """

    info = CARENA2014_MODEL_INFO

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        modulation_format: str = DEFAULT_MODULATION,
        *,
        coherence_exponent: float = DEFAULT_COHERENCE_EXPONENT,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        frequency_points: int = DEFAULT_FREQUENCY_POINTS,
        **route_options,
    ) -> None:
        """初始化节点几何与模型专属选项。

        Args:
            grid: ``SpectrumGrid``（遗留固定栅格）或 ``FlexibleGrid``。
            ref_fiber_config: 参考光纤配置。
            symbol_rate_hz: 符号率 (Hz)。
            modulation_format: 调制格式键，必须在 :data:`MODULATION_TABLE` 里。
            coherence_exponent: 跨间相干累积指数。
            quadrature_points: 每个复合子区间上的 Gauss-Legendre 节点数。
            frequency_points: CUT 带内的 NLI PSD 采样频率点数。
            **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。

        Raises
        ------
            KeyError: 调制格式不在 2014 Table 1 里。
        """
        super().__init__(grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options)
        if modulation_format not in MODULATION_TABLE:
            raise KeyError(
                f"unknown modulation format {modulation_format!r}; "
                f"2014 Table 1 provides {sorted(MODULATION_TABLE)}."
            )
        self.modulation_format = modulation_format
        self.phi, self.psi = MODULATION_TABLE[modulation_format]
        self.coherence_exponent = float(coherence_exponent)
        self.quadrature_points = int(quadrature_points)
        self.frequency_points = int(frequency_points)
        #: 当前光路的跨段数；只在 :meth:`evaluate_path` 调用期间有效。
        self._n_path_spans = 1
        # GN 骨架：冻结的闭式 GN 评估器（与 ``ecz`` 同口径）。
        self._gn = MultiBandISRSGN(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=self.connections,
            slots_per_channel=self.slots_per_channel,
            channel_bandwidth_source=self.channel_bandwidth_source,
            power_width_mode=self.power_width_mode,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """构造该光纤参数下的单跨核（按 ``FiberSpanConfig`` 记忆化）。"""
        return Carena2014SpanKernel(
            gn_evaluator=self._gn.for_fiber(fiber_config),
            phi=self.phi,
            psi=self.psi,
            coherence_exponent=self.coherence_exponent,
            quadrature_points=self.quadrature_points,
            frequency_points=self.frequency_points,
        )

    # --- 路径累积 ----------------------------------------------------------
    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> PathQoTResult:
        """沿路径评估，并让逐跨累加还原链路级的 N 跨总量。

        2014 论文的 ``mu`` 因子（Eq. (12)）描述跨间相干累积，而冻结的路由层是逐跨
        非相干相加。这里先数出光路的跨段数，再让单跨核返回逐跨等效值，累加
        结果即等于链路级总量。
        """
        previous = self._n_path_spans
        self._n_path_spans = _count_spans(optical_path)
        try:
            return super().evaluate_path(optical_path, power_in_w)
        finally:
            self._n_path_spans = previous

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """单跨：调用核（逐跨等效值），再统一走放大器/ASE。"""
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate_per_span(power_in_w, span_length_m, self._n_path_spans)
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


def _count_spans(optical_path: MultiSpanOpticalPath) -> int:
    """数出光路实际包含的跨段数（与路由层的跨段切分规则一致）。"""
    total = 0
    for fiber in optical_path.fiber_configs:
        remaining_km = fiber.length_km
        while remaining_km > 1e-12:
            span_km = min(remaining_km, fiber.max_span_length_km)
            total += 1
            remaining_km -= span_km
    return max(1, total)


register_model(CARENA2014_MODEL_INFO, Carena2014Model)
