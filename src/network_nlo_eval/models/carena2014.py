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
8. **跨间相干用"有效相位步长"压缩。** 论文的 ``nu(p)`` 随频偏 ``p`` 变化，链路级总量
   是对 ``p`` 的积分；契约层每个信道每跨只给一个复数，于是整条 ``p`` 轴的相干行为
   被压成一个标量步长 ``Theta``（等效频偏 ``p_eff = Theta / (4 pi^2 beta2 L_s)``）。
   这让**总量**逐位正确（相对偏差 < 1e-12），但把 ``nu(p)`` 的谱形换成了 Dirichlet
   核的谱形：NLI 的谱形不再逐点等于论文的 ``|nu(p)|^2`` 加权结果。
9. **等跨、等功率假设。** 逐跨功率按单跨口径算，跨间相位步长按等跨算。跨长不等或
   逐跨入纤功率不等时，路由层的相干求和会带上功率传输因子的权重，链路级总量不再
   等于论文积分式在同一 (跨数, 跨长) 下的结果。
10. **求积表的规模随色散线性增长。** ``p`` 轴上的原函数表按 ``nu`` 的最细周期
    （``pi/(N_s |c|)``，``c = 2 pi^2 beta2 L_s``）细分，色散越大表越大：D = 17
    ps/(nm·km) 时建表约 2.1 s，D = 100 时约 11.8 s，内存也同比例上升。极强色散
    的配置在本模型上不经济。

**跨间相干累积：逐跨复振幅**

2014 论文的 ``mu``（Eq. (12)）里的跨间相干因子就是

.. code-block:: text

    nu(p) = sum_{k=0}^{N_s-1} e^{j (2 N_s - 2 - 2k) psi(p)} ,
    psi(p) = 2 pi^2 beta2 L_s p ,

即 ``N_s`` 个等幅、相位随跨号线性变化的复指数之和。于是"跨间相干"可以写成
``|sum_k a_k|^2``：每一跨给出自己的复振幅 ``a_k = sqrt(Gamma_k) e^{j phi_k}``，
跨与跨之间按复数相加。

本实现按这条路径走：

* **逐跨功率** ``Gamma_k`` 是本跨自己的 NLI 功率：单跨口径的 GN 骨架 × EGN 修正比
  （SCI 用论文 Eq. (7)-(9) 的单跨积分，XPM 用 Eq. (16)-(18)/(23)-(26) 的单跨积分）。
  它与链路跨数无关——一跨单独跑时产生的 NLI 就是这么多。
* **跨间相干累加因子** ``B(N) = EGN(N)/EGN(1)``：论文积分式在 ``N`` 跨与 ``1`` 跨上的
  比值，已经把 Eq. (12) 的 ``nu(p)`` 积进去。它是路由层的 ``|sum_k a_k|^2`` 需要
  乘上的因子，也是"相干 vs 非相干"的比值来源。
* **相位** ``phi_k = Theta * (cumulative_length_m + span_length_m) / L_s``，随
  ``span_index`` 线性增长。``Theta``（rad/跨）不是猜的：它由 ``B(N)`` 反解出来
  （``|sum_k e^{j k Theta}|^2 = B(N)``，Dirichlet 核单调反解），因此
  ``|sum_k a_k|^2 = Gamma * B(N)`` 逐位还原论文积分式给出的链路级总量。
* ``B(N)`` 在色散趋零时趋于 ``N^2``（完全相干），色散很强时趋于 ``N``（非相干），
  这正是论文 ``nu`` 的解析行为。

折算逻辑（"算链路级总量再除以 N"）已删除：跨间累加现在由路由层真的做复数相加。

**已知近似**：``nu(p)`` 随频偏 ``p`` 变化，链路级总量是对 ``p`` 的积分；契约层每个
信道每跨只给一个复数，于是"有效相位步长" ``Theta`` 是把整条 ``p`` 轴的相干行为
压成一个标量（等效频偏 ``p_eff = Theta / (4 pi^2 beta2 L_s)``）。这让**总量**逐位
正确，但把 ``nu(p)`` 的谱形换成了 Dirichlet 核的谱形。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

from network_nlo_eval.core.constants import PI
from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
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

#: 反解等效跨间相位步长时的二分迭代次数。
#:
#: 目标函数 ``|sum_k e^{j k Theta}|^2`` 在 ``(0, 2 pi/N)`` 上单调下降，二分 44 次
#: 后区间宽度小于 ``6e-14``（``N = 10`` 时），相干累加因子的相对误差在 ``1e-12``
#: 量级，远低于门要求的 ``1e-9``。
DIRICHLET_BISECTION_STEPS = 44

CARENA2014_MODEL_INFO = ModelInfo(
    name="carena2014",
    summary="Carena 2014 EGN model: GN backbone with quadrature-evaluated 4th/6th-order corrections.",
    family="egn",
    reference="Carena et al., Opt. Express 22(13):16335-16362, 2014, doi:10.1364/OE.22.016335",
    options=(
        ("modulation_format", "Modulation format key: bpsk / qpsk / 16qam / 64qam (2014 Table 1)."),
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
# 跨间相干累加因子
# =============================================================================
def dirichlet_buildup(theta_rad: NDArrayFloat, n_spans: int) -> NDArrayFloat:
    r"""跨间相干累加因子 :math:`|\sum_{k=0}^{N-1} e^{j k\theta}|^2`。

    2014 论文 Eq. (12) 的跨间因子 ``nu`` 就是这个和（每跨幅度相同，相位差 ``2 psi``），
    等跨时它等于 :math:`(\sin(N\theta/2)/\sin(\theta/2))^2`。

    Args:
        theta_rad: 相邻跨之间的相位差 (rad)，可广播。
        n_spans: 跨数。

    Returns
    -------
        与 ``theta_rad`` 同形状的无量纲数组。
    """
    theta = np.asarray(theta_rad, dtype=np.float64)
    if n_spans <= 1:
        return np.ones_like(theta)
    half = 0.5 * theta
    # 直接算 sin(N x)/sin(x)（不用 np.sinc：热路径上这个函数每跨要调用几十次）。
    numerator = np.sin(float(n_spans) * half)
    denominator = np.sin(half)
    # theta -> 0 是可去奇点：lim sin(N x)/sin(x) = N。
    tiny = np.abs(denominator) < 1e-12
    safe = np.where(tiny, 1.0, denominator)
    ratio = np.where(
        tiny,
        float(n_spans) * np.cos((float(n_spans) - 1.0) * half),
        numerator / safe,
    )
    return ratio * ratio


def solve_dirichlet_step(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    r"""反解等效跨间相位步长 :math:`\Theta`（rad/跨）。

    ``dirichlet_buildup`` 在 ``(0, 2 pi/N)`` 上从 ``N^2`` 单调降到 0，故二分有唯一解。

    Args:
        n_spans: 跨数。
        target: 目标相干累加因子（无量纲），形状任意。

    Returns
    -------
        等效跨间相位步长 (rad)，形状同 ``target``。
    """
    if n_spans <= 1:
        return np.zeros_like(np.asarray(target, dtype=np.float64))
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n_spans) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n_spans)
    for _ in range(DIRICHLET_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        too_large = dirichlet_buildup(middle, n_spans) > goal
        low = np.where(too_large, middle, low)
        high = np.where(too_large, high, middle)
    return 0.5 * (low + high)


def amplitude_from_power(power_w: NDArrayFloat) -> NDArrayComplex:
    """把功率变成实振幅，使 ``|a|**2`` 与功率逐位相同。

    契约要求 ``|amp|**2 == *_noise_w``，而 ``sqrt(p)**2`` 只在部分情形下与 ``p``
    逐位相同（最后一位的舍入）。这里在 ``sqrt(p)`` 及其两个相邻浮点数里挑一个
    平方恰好等于 ``p`` 的。

    Args:
        power_w: 功率 (W)，非负。

    Returns
    -------
        复振幅（虚部为 0），形状同 ``power_w``。
    """
    root = np.sqrt(np.asarray(power_w, dtype=np.float64))
    target = np.asarray(power_w, dtype=np.float64)
    # 逐位比较：root*root 可能与 power 差 1 ulp。
    bad = root * root != target
    if bool(np.any(bad)):
        for candidate in (np.nextafter(root, np.inf), np.nextafter(root, -np.inf)):
            square = candidate * candidate
            take = bad & (square == target)
            root = np.where(take, candidate, root)
            bad = bad & ~take
    return root.astype(np.complex128)


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


@dataclass(frozen=True)
class _XpmBlocks:
    """XPM 的分块：XCI 与 MCI 的 GN 项与 EGN 修正项（逐信道）。

    Attributes
    ----------
        xci_gn: XCI 的 GN 项（论文 Eq. (18) 的四个域）。
        xci_corr: XCI 的 EGN 修正项。
        mci_gn: MCI 的 GN 项（论文 Eq. (23) 的 M0/M1 域）。
        mci_corr: MCI 的 EGN 修正项（论文 Eq. (26)）。
    """

    xci_gn: NDArrayFloat
    xci_corr: NDArrayFloat
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
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        frequency_points: int = DEFAULT_FREQUENCY_POINTS,
    ) -> None:
        """按光纤配置准备 GN 骨架与 EGN 积分器。

        Args:
            gn_evaluator: 冻结的闭式 GN 评估器（``models.isrs_gn``）。
            phi: 调制格式的 4 阶矩 ``Phi_a``（2014 Table 1）。
            psi: 调制格式的 6 阶矩 ``Psi_a``（2014 Table 1）。
            quadrature_points: 每个复合子区间上的 Gauss-Legendre 节点数。
            frequency_points: CUT 带内的 NLI PSD 采样频率点数。
        """
        self.gn = gn_evaluator
        self.phi = float(phi)
        self.psi = float(psi)
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
        # 与功率无关的、已按信道展开的 kappa 标量：按 (跨数, 跨长) 记忆化。
        # 热路径上只剩幂次乘法，不再逐信道查表。
        self._mci_channel_cache: dict[tuple[int, float], tuple[NDArrayFloat, ...]] = {}
        self._xci_distance_cache: dict[tuple[int, float], dict[int, tuple[float, ...]]] = {}
        # 跨间相干累加因子与等效相位步长：只依赖 (跨数, 跨长)，可缓存。
        self._buildup_cache: dict[tuple[int, float], float] = {}
        self._step_cache: dict[tuple[int, float], float] = {}

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
    def _sci_block(self, n_spans: int, span_length_m: float) -> tuple[float, float]:
        """SCI 的 GN 项总和与 EGN 块总和（论文 Eq. (7)-(9) 的积分结果）。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``(gn, egn)``：``gn = sum_f kappa1``（纯 GN 项），
            ``egn = sum_f (kappa1 + Phi_a kappa2 + Psi_a kappa3)``（含修正项）。
        """
        table = self.sci_kappas(n_spans, span_length_m)
        gn = 0.0
        total = 0.0
        for kappas in table.values():
            gn += float(np.sum(kappas.kappa1))
            total += float(np.sum(kappas.kappa1 + self.phi * kappas.kappa2 + self.psi * kappas.kappa3))
        return gn, total

    def _sci_ratio(self, n_spans: int, span_length_m: float) -> float:
        """SCI 的修正比（在 CUT 带内按 GN 项加权平均）。"""
        gn, total = self._sci_block(n_spans, span_length_m)
        if gn <= 0.0:
            return 1.0
        return total / gn

    def _xci_distance_sums(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> dict[int, tuple[float, float, float, float, float, float]]:
        """XCI 各邻居距离上的 kappa 标量（GN 项三项 + 修正项三项）。

        论文 Eq. (18) 把单个 INT 的 XCI 写成四段，前置因子分别是
        ``P_CUT P_INT^2``（X1）、``P_CUT^2 P_INT``（X2/X3）、``P_INT^3``（X4），
        每一段都带一个 GN 项与各自的修正项。这里按近邻窗口内的每个距离 ``d``
        分别累加，返回 GN 项与修正项两块。

        这些标量只依赖 (跨数, 跨长)，与入纤功率无关，因此按 (跨数, 跨长) 记忆化：
        热路径上只剩幂次乘法。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``{距离: (g11, g21, g41, c11, c21, c41)}``。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._xci_distance_cache.get(key)
        if cached is not None:
            return cached
        table = self.xci_kappas(n_spans, span_length_m)
        sums: dict[int, tuple[float, float, float, float, float, float]] = {}
        for distance, kappas in table.items():
            g11 = float(np.sum(kappas.kappa11))
            g21 = float(np.sum(kappas.kappa21 + kappas.kappa31))
            g41 = float(np.sum(kappas.kappa41))
            c11 = self.phi * float(np.sum(kappas.kappa12))
            c21 = self.phi * float(np.sum(kappas.kappa22 + kappas.kappa32))
            c41 = self.phi * float(np.sum(kappas.kappa42)) + self.psi * float(np.sum(kappas.kappa43))
            sums[distance] = (g11, g21, g41, c11, c21, c41)
        self._xci_distance_cache[key] = sums
        return sums

    def _xci_terms(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """XCI 的 GN 项与修正项（逐信道），由给定的 κ 表算出。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。
            power: 逐信道入纤功率 (W)。

        Returns
        -------
            ``(gn, corr)``，均为 ``(N,)``。
        """
        n_ch = int(self.b_hz.size)
        gn = np.zeros(n_ch, dtype=np.float64)
        corr = np.zeros(n_ch, dtype=np.float64)
        if n_ch < 2:
            return gn, corr
        for distance, (g11, g21, g41, c11, c21, c41) in self._xci_distance_sums(
            n_spans, span_length_m
        ).items():
            hi = slice(distance, n_ch)
            lo = slice(0, n_ch - distance)
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
            gn[lo] += p_hi * p_lo**2 * g11
            gn[lo] += p_hi ** 2 * p_lo * g21
            gn[lo] += p_lo**3 * g41
            corr[lo] += p_hi * p_lo**2 * c11
            corr[lo] += p_hi ** 2 * p_lo * c21
            corr[lo] += p_lo**3 * c41
        return gn, corr

    def _xci_ratio(self, n_spans: int, span_length_m: float, power: NDArrayFloat) -> NDArrayFloat:
        """XCI 的逐信道修正比 ``sum_j CORR(i,j) / sum_j GN(i,j)``。

        ``j`` 遍历近邻窗口内的 INT 信道。
        """
        gn, corr = self._xci_terms(n_spans, span_length_m, power)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(gn > 0.0, corr / np.where(gn > 0.0, gn, 1.0), 0.0)

    def _mci_channel_kappas(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """MCI 的逐信道 kappa 标量（``M0``、``M1`` 的 GN 项与 4 阶修正项）。

        只用 CUT 上下紧邻的两个 INT 信道（论文 Sect. 5 的三信道形式）。这些标量与
        入纤功率无关，按 (跨数, 跨长) 记忆化，热路径上只剩幂次乘法。

        Args:
            n_spans: 链路级跨数。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``(kappa_m0, kappa_m1_1, kappa_m1_2)``，均为 ``(N,)``。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._mci_channel_cache.get(key)
        if cached is not None:
            return cached
        n_ch = int(self.b_hz.size)
        km0 = np.zeros(n_ch, dtype=np.float64)
        km11 = np.zeros(n_ch, dtype=np.float64)
        km12 = np.zeros(n_ch, dtype=np.float64)
        table = self.mci_kappas(n_spans, span_length_m)
        for index in range(n_ch):
            lo = int(self.mci_lower[index])
            hi = int(self.mci_upper[index])
            if lo < 0 or hi < 0:
                continue
            pair = (
                round(float(self.f_rel_hz[hi] - self.f_rel_hz[index]), 6),
                round(float(self.f_rel_hz[index] - self.f_rel_hz[lo]), 6),
            )
            kappas = table[pair]
            km0[index] = float(np.sum(kappas.kappa_m0))
            km11[index] = float(np.sum(kappas.kappa_m1_1))
            km12[index] = float(np.sum(kappas.kappa_m1_2))
        cached = (km0, km11, km12)
        self._mci_channel_cache[key] = cached
        return cached

    def _mci_terms(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """MCI 的 GN 项与修正项 (W)，按论文 Eq. (23)。

        只用 CUT 上下紧邻的两个 INT 信道（论文 Sect. 5 的三信道形式）。
        """
        n_ch = int(self.b_hz.size)
        gn = np.zeros(n_ch, dtype=np.float64)
        corr = np.zeros(n_ch, dtype=np.float64)
        if n_ch < 2:
            return gn, corr
        km0, km11, km12 = self._mci_channel_kappas(n_spans, span_length_m)
        lower = np.where(self.mci_lower >= 0, self.mci_lower, 0)
        upper = np.where(self.mci_upper >= 0, self.mci_upper, 0)
        valid = (self.mci_lower >= 0) & (self.mci_upper >= 0)
        p_cut = power
        p_up = power[upper]
        p_low = power[lower]
        # Eq. (23) 的 M0 域：P_CUT P_INT,1 P_INT,-1 kappa_M0。
        # Eq. (25)/(26) 的 M1 域：P_INT,1^2 P_INT,-1 (kappa_M1,1 + Phi_a kappa_M1,2)。
        m1 = p_up**2 * p_low
        gn = np.where(valid, p_cut * p_up * p_low * km0 + m1 * km11, 0.0)
        corr = np.where(valid, m1 * self.phi * km12, 0.0)
        return gn, corr

    def _xpm_blocks(self, n_spans: int, span_length_m: float, power: NDArrayFloat) -> _XpmBlocks:
        """XPM 的分块：XCI 与 MCI 的 GN 项与 EGN 修正项（逐信道）。"""
        xci_gn, xci_corr = self._xci_terms(n_spans, span_length_m, power)
        mci_gn, mci_corr = self._mci_terms(n_spans, span_length_m, power)
        return _XpmBlocks(xci_gn=xci_gn, xci_corr=xci_corr, mci_gn=mci_gn, mci_corr=mci_corr)

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

    # --- 跨间相干累加 --------------------------------------------------------
    def _sci_buildup(self, n_spans: int, span_length_m: float) -> float:
        """SCI 的跨间相干累加因子 ``B(N) = EGN(N)/EGN(1)``。

        论文 Eq. (7)-(9) 的积分式在 ``N`` 跨与 ``1`` 跨上的比值：Eq. (12) 的跨间因子
        ``nu(p)`` 已经积进积分里。色散趋零时 ``nu`` 退化成常数 ``N``（``|nu|^2 =
        N^2``），``B(N) -> N^2``（完全相干）；色散很强时 ``|nu|^2`` 在 ``p`` 轴上的
        平均趋于 ``N``，``B(N) -> N``（非相干）。

        这个因子是**逐跨复振幅相干求和**要乘上的东西：``|sum_k a_k|^2`` 必须等于
        ``Gamma * B(N)``。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._buildup_cache.get(key)
        if cached is not None:
            return cached
        _, egn_now = self._sci_block(n_spans, span_length_m)
        _, egn_ref = self._sci_block(1, span_length_m)
        buildup = egn_now / egn_ref if egn_ref > 0.0 else 1.0
        self._buildup_cache[key] = buildup
        return buildup

    def _sci_step(self, n_spans: int, span_length_m: float) -> float:
        """SCI 的等效跨间相位步长 (rad/跨)。

        由相干累加因子反解：``|sum_k e^{j k Theta}|^2 = B(N)``。它随色散变化，
        色散趋零时趋于 0（完全相干），色散很强时趋于非相干对应的步长。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._step_cache.get(key)
        if cached is not None:
            return cached
        step = float(solve_dirichlet_step(n_spans, np.asarray(self._sci_buildup(n_spans, span_length_m))))
        self._step_cache[key] = step
        return step

    def _coherent_levels(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
        s2_spm: NDArrayFloat,
        s2_xpm: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """相干累加需要的两级口径。

        Returns
        -------
            ``(逐跨 SCI, 逐跨 XPM, XPM 累加因子)``。逐跨值是"本跨自己的 NLI
            功率"，累加因子是链路级与单跨的比值（也就是路由层的 ``|sum_k a_k|^2``
            要乘上的东西）。SCI 的累加因子与信道无关，由
            :meth:`_sci_buildup` 给出。
        """
        spm_power = self._sci_span_level(span_length_m, s2_spm)
        xpm_power, link_xpm = self._xpm_levels(n_spans, span_length_m, power, s2_xpm)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe = np.where(xpm_power > 0.0, xpm_power, 1.0)
            buildup_xpm = np.where(xpm_power > 0.0, link_xpm / safe, 0.0)
        return spm_power, xpm_power, buildup_xpm

    def _xpm_levels(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
        s2_xpm: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """XPM 的两级口径：``(单跨值, N 跨链路级值)``。

        单跨值就是"本跨自己的 NLI 功率"；链路级值是论文 Eq. (16)-(18)/(23)-(26)
        的积分式在 ``N`` 跨上的结果。两者之比就是跨间相干累加因子。
        """
        blocks = self._xpm_blocks(n_spans, span_length_m, power)
        reference = self._xpm_blocks(1, span_length_m, power)
        xci_ref = reference.xci_gn
        safe = np.where(xci_ref > 0.0, xci_ref, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            block_now = np.where(
                xci_ref > 0.0,
                (blocks.xci_gn + blocks.xci_corr) / safe,
                0.0,
            )
            block_ref = np.where(
                xci_ref > 0.0,
                (reference.xci_gn + reference.xci_corr) / safe,
                0.0,
            )
        active = power > 0.0
        per_span = np.where(active, s2_xpm * block_ref + reference.mci_gn + reference.mci_corr, 0.0)
        link = np.where(active, s2_xpm * block_now + blocks.mci_gn + blocks.mci_corr, 0.0)
        return per_span, link

    def link_level_nli(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """链路级 NLI 总量（SCI、XPM），由论文积分式直接给出。

        这是"逐跨复振幅相干求和"要还原的目标：路由层逐跨累加得到的
        ``|sum_k a_k|^2`` 必须与它一致。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            n_spans: 链路级跨数。

        Returns
        -------
            ``(sci_w, xpm_w)``，均为 ``(N,)``。
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        n = max(1, int(n_spans))
        gn_out = self.gn.span_fiber_output(power, float(span_length_m), 1)
        s2_spm = np.asarray(gn_out.spm_noise_w, dtype=np.float64)
        s2_xpm = np.asarray(gn_out.xpm_noise_w, dtype=np.float64)
        sci = self._sci_link_level(n, span_length_m, s2_spm)
        _, xpm = self._xpm_levels(n, span_length_m, power, s2_xpm)
        return sci, xpm

    def _sci_link_level(self, n_spans: int, span_length_m: float, s2_spm: NDArrayFloat) -> NDArrayFloat:
        """SCI 的链路级总量：单跨 GN 骨架 × N 跨上的 EGN 积分块。"""
        gn_ref, _ = self._sci_block(1, span_length_m)
        _, egn_now = self._sci_block(n_spans, span_length_m)
        return s2_spm * (egn_now / gn_ref if gn_ref > 0.0 else 1.0)

    def _sci_span_level(self, span_length_m: float, s2_spm: NDArrayFloat) -> NDArrayFloat:
        """SCI 的单跨值（本跨自己的 NLI 功率）：单跨 GN 骨架 × 单跨 EGN 积分块。"""
        gn_ref, egn_ref = self._sci_block(1, span_length_m)
        return s2_spm * (egn_ref / gn_ref if gn_ref > 0.0 else 1.0)

    # --- 热路径 -------------------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """计算单跨的出纤功率、NLI 噪声与跨间相干用的复振幅。

        跨间相干由**复振幅**承担：每一跨给出 ``a_k = sqrt(Gamma_k) e^{j phi_k}``，
        路由层按 ``|sum_k a_k|^2`` 累加。相位的步长是论文 Eq. (12) 的等效跨间
        相位 ``Theta``（由积分式给出的相干累加因子反解），随跨号线性增长。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 链路级跨数；只在没有给 ``span_context`` 时用作
                链路总跨数（此时按"第 0 跨"处理）。
            span_context: 本跨在链路里的位置（跨号、累计长度、跨长）。

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
        context = span_context if span_context is not None else SpanContext(
            span_index=0,
            total_spans=max(1, int(n_effective_spans)),
            cumulative_length_m=0.0,
            span_length_m=float(span_length_m),
        )
        n_spans = max(1, int(context.total_spans))
        length_m = float(span_length_m)

        # --- GN 骨架（冻结闭式：SPM = SCI，XPM = XCI 的 X1 域）--------------
        # 跨间的累加现在由复振幅负责，闭式核里的跨数幂次不再进入。
        gn_out = self.gn.span_fiber_output(power, length_m, 1)
        power_out = np.array(gn_out.power_out_w, dtype=np.float64, copy=True)
        s2_spm = np.array(gn_out.spm_noise_w, dtype=np.float64, copy=True)
        s2_xpm = np.array(gn_out.xpm_noise_w, dtype=np.float64, copy=True)
        active = power > 0.0

        if n_spans == 1:
            # 单跨：相干退化为单跨，噪声与改造前逐位相同，振幅相位取 0。
            ratios = self._ratios(1, length_m, power)
            rho_sci = ratios.sci if bool(np.any(active)) else 1.0
            rho_xci = np.where(active, 1.0 + ratios.xci, 0.0)
            mci = np.where(active, ratios.mci_gn + ratios.mci_corr, 0.0)
            spm_noise = s2_spm * rho_sci
            xpm_noise = s2_xpm * rho_xci + mci
            return SpanFiberOutput(
                power_out,
                spm_noise,
                xpm_noise,
                amplitude_from_power(spm_noise),
                amplitude_from_power(xpm_noise),
            )

        # --- 多跨：逐跨功率 + 跨间相干相位 ---------------------------------------
        # 逐跨功率 = 本跨自己的 NLI 功率（单跨口径的 GN 骨架 × EGN 修正比）；
        # 累加因子 = 链路级与单跨的比值。
        spm_power, xpm_power, buildup_xpm = self._coherent_levels(
            n_spans, length_m, power, s2_spm, s2_xpm
        )

        # 等效跨间相位步长：由相干累加因子反解（论文 Eq. (12) 的等效形式）。
        step_sci = self._sci_step(n_spans, length_m)
        step_xpm = solve_dirichlet_step(n_spans, buildup_xpm)

        # 相位随跨号线性增长：用本跨出纤处到链路末端的累计长度，
        # 等跨时相邻跨的相位差恰是等效步长。
        travelled = (float(context.cumulative_length_m) + float(context.span_length_m)) / length_m
        phase_sci = step_sci * travelled
        phase_xpm = step_xpm * travelled

        spm_power = np.where(active, spm_power, 0.0)
        amplitude_spm = amplitude_from_power(spm_power) * np.exp(1j * phase_sci)
        amplitude_xpm = amplitude_from_power(xpm_power) * np.exp(1j * phase_xpm)
        return SpanFiberOutput(
            power_out,
            spm_power,
            xpm_power,
            amplitude_spm,
            amplitude_xpm,
        )


class Carena2014Model(RouteModel):
    """Carena 2014 EGN 模型的路由层模型。

    与其它模型共用 :mod:`network_nlo_eval.models.route` 的跨段累积、放大器与 SNR
    合成；本类只负责单跨物理核。

    构造期可选参数（均为关键字）：

    ``modulation_format``
        2014 Table 1 的调制格式键（``"bpsk"``/``"qpsk"``/``"16qam"``/``"64qam"``），
        决定 ``Phi_a`` 与 ``Psi_a``。表里没有的格式直接报错，不做外推。
    ``quadrature_points``
        每个复合子区间上的 Gauss-Legendre 节点数，默认 6。
    ``frequency_points``
        CUT 带内的 NLI PSD 采样频率点数，默认 3。
    """

    info = CARENA2014_MODEL_INFO

    #: 本模型给出跨间相干用的复振幅，路由层据此做复数累加。
    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        modulation_format: str = DEFAULT_MODULATION,
        *,
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
        self.quadrature_points = int(quadrature_points)
        self.frequency_points = int(frequency_points)
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
            quadrature_points=self.quadrature_points,
            frequency_points=self.frequency_points,
        )

    # --- 路径累积 ----------------------------------------------------------
    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """单跨：调用核（带跨间相干振幅），再统一走放大器/ASE。

        振幅的参考面按契约是**本跨出纤处**（即放大器之前）。冻结层
        :func:`~network_nlo_eval.models.route.apply_amplifier_and_roadm` 在相干档会
        把振幅再乘 ``sqrt(gain)``，而纯功率档不给噪声乘 ``gain``——两种口径因此
        差一个放大器增益（见报告里的最小复现）。本模型要与冻结基线同口径（噪声
        不被行内放大器放大），所以先把振幅摘下来走放大器，再按原样贴回。

        Args:
            fiber_config: 该跨的光纤参数。
            edfa_config: 跨段放大器配置。
            power_in_w: 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            span_context: 本跨在链路里的位置；给出时核据此给出相对相位。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.ElementOutput`。
        """
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate(power_in_w, span_length_m, 1, span_context=span_context)
        # 放大器/ROADM 只处理功率与 ASE；振幅保持在出纤处的参考面上。
        stripped = SpanFiberOutput(
            fiber_output.power_out_w,
            fiber_output.spm_noise_w,
            fiber_output.xpm_noise_w,
        )
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=stripped,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )
        return ElementOutput(
            element.power_out_w,
            element.spm_noise_w,
            element.xpm_noise_w,
            element.ase_noise_w,
            fiber_output.nli_amplitude_spm,
            fiber_output.nli_amplitude_xpm,
        )


register_model(CARENA2014_MODEL_INFO, Carena2014Model)
