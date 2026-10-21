"""Carena 2014 EGN 积分的**独立求积**参照实现（回归守卫用）。

这个模块是 ``evidence/egn-coherence/egn_independent_quadrature.py`` 的仓库内副本，
算法照抄：它**不 import** ``network_nlo_eval`` 的任何模块，也不 import
``models/carena2014.py``，算例参数与光纤参数在 :data:`CASE` 里显式写死，公式逐条取自
Carena et al., Opt. Express 22(13):16335-16362 (2014)。它存在的意义是给
``models/carena2014.py`` 的积分式提供一把**外部尺子**：模型的 4 阶修正项如果又把
"两行合并成一个积分"，这里能立刻看出来。

**求积方法（与被审对象不同）**

* 被审对象把内层积分换成 ``p = (f1-f)(f2-f)`` 轴上的原函数表（三次 Hermite 插值）。
* 本模块直接对 ``(f1, f2)`` 做**带振荡自适应细分的 Gauss-Legendre** 二维求积，
  内层积分不借助任何原函数表。

**论文公式索引（本模块用到）**

========  =====================================================================
编号      内容
========  =====================================================================
(5)       ``G_SCI^EGN = P^3 [kappa1 + Phi_a kappa2 + Psi_a kappa3]``
(6)/(19)  ``Phi_a`` / ``Psi_a``（Table 1 的调制格式矩）
(7)       ``kappa1``：SCI 的 GN 项（二重积分）
(8)       ``kappa2``：SCI 的 4 阶修正（**两行，两个不同的积分域**）
(9)       ``kappa3``：SCI 的 6 阶修正（四重积分）
(10)      ``mu = zeta * nu``
(11)      ``zeta``：单跨 FWM 效率
(12)      ``nu``：跨间相干因子（Dirichlet 核）
(14)/(16)/(17)  XPM 近似：``kappa11``（GN）、``kappa12``（4 阶）
(18)      完整 XCI：X1..X4 四个域
(23)/(24)/(25)/(26)  MCI：``kappa_M0``、``kappa_M1,1``、``kappa_M1,2``
(29)-(35)  XCI 各域的 ``kappa_mn``
(43)/(44)/(45)  附录 C 的降维（4 阶、4 阶、6 阶）
========  =====================================================================

**积分域的约定（照抄被审脚本）**

``_double_integral`` 的四个频带参数在两种模式下含义不同，这是被审脚本的原始约定，
本模块原样保留：

* ``"gn"`` / ``"l1"`` / ``"sixth"``：``band1`` 是第 1 个波形的频带，``band2`` 是第 2
  个波形的频带，``band3`` 是第 3 个波形 ``f3 = f1+f2-f`` 的频带（外层变量是 ``f1``）。
* ``"l2"``（论文附录 C 的 Eq. (44)）：外层变量是第 3 个波形，``band1`` 是它的频带，
  ``band2`` 是第 2 个波形的频带，``band3`` 是第 1 个波形 ``f1 = f3-f2+f`` 的频带。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss

# =============================================================================
# 算例与光纤参数（c_57ch；与 network_nlo_eval/benchmarks/cases.py 一致）
# =============================================================================


@dataclass(frozen=True)
class Fiber:
    """单跨光纤参数（SMF，100 km）。

    Attributes
    ----------
        beta2_s2_m: 二阶色散 (s^2/m)。
        gamma_per_w_m: 非线性系数 (1/(W m))。
        alpha_power_npm: **功率**衰减 (Np/m)；论文里的场衰减 ``alpha`` 是它的一半。
        length_m: 跨长 (m)。
    """

    beta2_s2_m: float
    gamma_per_w_m: float
    alpha_power_npm: float
    length_m: float


@dataclass(frozen=True)
class Case:
    """算例参数。

    Attributes
    ----------
        name: 算例名。
        num_channels: 信道数。
        spacing_hz: 信道间隔 (Hz)。
        symbol_rate_hz: 符号率 (Hz)，也是论文假设下的信道带宽。
        power_w: 每信道入纤功率 (W)。
        modulation: 调制格式键（2014 Table 1）。
        neighbor_span: XCI 聚合里每侧保留的近邻信道数。
    """

    name: str
    num_channels: int
    spacing_hz: float
    symbol_rate_hz: float
    power_w: float
    modulation: str
    neighbor_span: int
    fiber: Fiber


#: 2014 Table 1：``(Phi, Psi)``。
MODULATION_TABLE: dict[str, tuple[float, float]] = {
    "bpsk": (-1.0, 4.0),
    "qpsk": (-1.0, 4.0),
    "16qam": (-17.0 / 25.0, 52.0 / 25.0),
    "64qam": (-13.0 / 21.0, 1161.0 / 646.0),
}

#: ``c_57ch``：C 波段 57 信道 x 75 GHz，64 GBaud，10 x 100 km SMF，0 dBm/信道。
#: 光纤参数取频段中心信道（C 波段中心 193.75 THz）的值。
CASE = Case(
    name="c_57ch",
    num_channels=57,
    spacing_hz=75e9,
    symbol_rate_hz=64e9,
    power_w=1e-3,
    modulation="64qam",
    neighbor_span=8,
    fiber=Fiber(
        beta2_s2_m=-2.1607590148802473e-26,
        gamma_per_w_m=1.3219450475712609e-3,
        alpha_power_npm=3.735371857445309e-05,
        length_m=100e3,
    ),
)

#: 把独立求积的累加因子还原成"仓库口径"的总量 buildup 时用的单跨权重 (W)。
#:
#: 这两个数是独立求积审计当时在仓库里实测的单跨 SPM / XPM 功率。它们只用于把
#: :func:`sci_buildup` / :func:`xpm_buildup` 两个指标合成成总量指标，好让独立求积
#: 与模型输出直接可比；**这里冻结不改**：判定基准（总量 1.6173）就是用它们算出来的。
REPO_SPM_SINGLE_SPAN_W = 4.871879e-06
REPO_XPM_SINGLE_SPAN_W = 8.792996e-06

#: 独立求积的判定基准（``evidence/egn-coherence`` 的实测表，nodes/period = 16/32）。
#:
#: * ``EGN SPM buildup``（N=10）= 1.9624；
#: * ``GN SPM buildup``（N=10）= 1.4034；
#: * ``总量 buildup``（N=10）= 1.6173。
TARGET_GN_SPM_BUILDUP = 1.4034
TARGET_EGN_SPM_BUILDUP = 1.9624
TARGET_TOTAL_BUILDUP = 1.6173


# =============================================================================
# 论文 Eq. (10)-(12)：链路函数
# =============================================================================
def mu(p_hz2: np.ndarray, n_spans: int, fiber: Fiber = CASE.fiber) -> np.ndarray:
    """2014 论文的链路函数 ``mu(f1, f2, f)``（Eq. (10)-(12)）。

    ``mu`` 只通过乘积 ``p = (f1 - f) * (f2 - f)`` 依赖频率。

    Args:
        p_hz2: 频率乘积 (Hz^2)，可广播。
        n_spans: 跨数。
        fiber: 光纤参数。

    Returns
    -------
        复数组，形状同输入。
    """
    p = np.asarray(p_hz2, dtype=np.float64)
    beta2 = fiber.beta2_s2_m
    length = fiber.length_m
    # Eq. (11)：单跨总相位失配。
    theta = 4.0 * math.pi**2 * beta2 * p * length
    zeta = (
        fiber.gamma_per_w_m
        * (1.0 - np.exp(-fiber.alpha_power_npm * length) * np.exp(1j * theta))
        / (fiber.alpha_power_npm - 1j * 4.0 * math.pi**2 * beta2 * p)
    )
    # Eq. (12)：跨间相干因子 nu = sin(N psi)/sin(psi) * exp(j (N-1) psi)。
    psi = 0.5 * theta
    n = float(n_spans)
    s = np.sin(psi)
    safe = np.where(np.abs(s) < 1e-9, 1.0, s)
    ratio = np.where(
        np.abs(s) < 1e-9,
        n * np.cos((n - 1.0) * psi),
        np.sin(n * psi) / safe,
    )
    nu = ratio * np.exp(1j * (n - 1.0) * psi)
    return zeta * nu


# =============================================================================
# 求积工具
# =============================================================================
_GL_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _gl(n: int) -> tuple[np.ndarray, np.ndarray]:
    """``n`` 点 Gauss-Legendre 节点与权重（``[-1, 1]`` 上）。"""
    if n not in _GL_CACHE:
        _GL_CACHE[n] = leggauss(n)
    return _GL_CACHE[n]


def _split(lo: float, hi: float, breaks: list[float]) -> list[float]:
    """把 ``[lo, hi]`` 按内层窗口的折点切成若干段。"""
    interior = sorted({b for b in breaks if lo < b < hi})
    return [lo, *interior, hi]


def _osc_periods(rate: np.ndarray | float, span: float) -> np.ndarray:
    """给定相位变化率 (rad/Hz) 与区间宽度，给出需要细分的段数。

    论文的被积函数在 ``p`` 轴上以 ``2 pi^2 beta2 L`` 为系数振荡。本模块按
    "**半个振荡周期一段**、每段 ``npx`` 个 Gauss-Legendre 节点" 细分。
    """
    periods = np.abs(rate) * span / (math.pi)
    return np.maximum(1.0, np.ceil(periods))


def _double_integral(
    f_hz: float,
    band1: tuple[float, float],
    band2: tuple[float, float],
    band3: tuple[float, float],
    mode: str,
    n_spans: int,
    fiber: Fiber = CASE.fiber,
    npx: int = 8,
) -> complex:
    """论文积分式的二维求积（四种模式）。

    所有坐标都是**相对 CUT 中心**的：``f1`` 落在 ``band1``、``f2`` 落在 ``band2``、
    第三个波 ``f3 = f1 + f2 - f`` 落在 ``band3``；被干扰频率 ``f`` 也是相对量。

    模式定义（频带参数的含义随模式变化，见模块文档）：

    * ``"gn"``：``int int |mu|^2``（论文的 GN 项，Eq. (7)/(16)/(21)/(24)/(25)/(29)-(33)）。
    * ``"l1"``：``int df1 |int df2 mu|^2``（4 阶修正的 Eq. (43)/(17)/(30) 降维，
      外层变量是 ``f1``）。
    * ``"l2"``：``int df3 |int df2 mu(f3-f2+f, f2, f)|^2``（4 阶修正的 Eq. (44)/(32)/(34)
      降维，外层变量换成第三个波 ``f3``）。
    * ``"sixth"``：``|int int mu|^2``（6 阶修正的 Eq. (45)/(35) 降维）。

    Args:
        f_hz: 被干扰频率 (Hz)，相对 CUT 中心。
        band1: 模式相关的第 1 条频带。
        band2: 第 2 个波的频带。
        band3: 模式相关的第三条频带。
        mode: 求积模式。
        n_spans: 跨数。
        fiber: 光纤参数。
        npx: 每个振荡子段的 Gauss-Legendre 节点数。

    Returns
    -------
        积分结果。
    """
    if mode == "l2":
        breaks = [band2[0] + band3[1] - f_hz, band2[1] + band3[0] - f_hz]
    else:
        breaks = [band3[0] + f_hz - band2[0], band3[1] + f_hz - band2[1]]
    xg, xw = _gl(npx)
    total = 0.0 + 0.0j
    # 相位系数：psi = 2 pi^2 beta2 L p。
    phase_rate = 2.0 * math.pi**2 * fiber.beta2_s2_m * fiber.length_m
    segments = _split(band1[0], band1[1], breaks)
    for lo, hi in zip(segments[:-1], segments[1:], strict=False):
        # 外层变量的粗网格 + 振荡自适应细分。
        y_far = max(abs(band2[0] - f_hz), abs(band2[1] - f_hz))
        rate_x = phase_rate * y_far
        n_sub_x = int(_osc_periods(rate_x, hi - lo))
        edges = np.linspace(lo, hi, n_sub_x + 1)
        for a, b in zip(edges[:-1], edges[1:], strict=False):
            half = 0.5 * (b - a)
            mid = 0.5 * (a + b)
            x = mid + half * xg
            wx = half * xw
            if mode == "l2":
                y1 = np.maximum(band2[0], x + f_hz - band3[1])
                y2 = np.minimum(band2[1], x + f_hz - band3[0])
            else:
                y1 = np.maximum(band2[0], band3[0] + f_hz - x)
                y2 = np.minimum(band2[1], band3[1] + f_hz - x)
            valid = y2 > y1
            width = np.where(valid, y2 - y1, 0.0)
            # 内层振荡细分：d psi / dy。
            if mode == "l2":
                # d psi / dy = phase_rate * (x - 2 y + f)：在窗口内单调，取端点最大值。
                rate_y = phase_rate * np.maximum(
                    np.abs(x + f_hz - 2.0 * y1),
                    np.abs(x + f_hz - 2.0 * y2),
                )
            else:
                rate_y = phase_rate * np.abs(x - f_hz)
            n_sub_y = int(np.max(_osc_periods(rate_y, np.max(width) if width.size else 0.0)))
            n_sub_y = max(1, min(n_sub_y, 256))
            sub = np.linspace(0.0, 1.0, n_sub_y + 1)
            # (n_x, n_sub_y, npx)
            lo_y = y1[:, None] + width[:, None] * sub[None, :-1]
            hi_y = y1[:, None] + width[:, None] * sub[None, 1:]
            half_y = 0.5 * (hi_y - lo_y)
            mid_y = 0.5 * (hi_y + lo_y)
            y = mid_y[..., None] + half_y[..., None] * xg
            wy = half_y[..., None] * xw
            if mode == "l2":
                p = (x[:, None, None] - y) * (y - f_hz)
            else:
                p = (x[:, None, None] - f_hz) * (y - f_hz)
            m = mu(p, n_spans, fiber)
            mask = valid[:, None, None]
            if mode == "gn":
                total += complex(np.sum(wx[:, None, None] * wy * np.where(mask, np.abs(m) ** 2, 0.0)))
            else:
                # 内层积分（对振荡子段求和）-> (n_x,)
                inner = np.sum(np.sum(wy * np.where(mask, m, 0.0), axis=-1), axis=-1)
                if mode == "sixth":
                    # Eq. (45)/(35)：模平方罩在整个二重积分外面。
                    total += complex(np.sum(wx * inner))
                else:
                    # Eq. (43)/(44)：模平方在内层积分外面。
                    total += complex(np.sum(wx * np.abs(inner) ** 2))
    return total


# =============================================================================
# 论文的各个 kappa
# =============================================================================
@dataclass(frozen=True)
class SciKappas:
    """SCI 的三个系数（Eq. (7)/(8)/(9)）。

    Attributes
    ----------
        kappa1: GN 项。
        kappa2: 4 阶修正（**两行相加**：``(80/81) L1 + (16/81) L2``）。
        kappa3: 6 阶修正。
        l1: 4 阶修正第一行（外层变量 ``f1``）的积分值。
        l2: 4 阶修正第二行（外层变量 ``f3``）的积分值。
    """

    kappa1: float
    kappa2: float
    kappa3: float
    l1: float
    l2: float


@dataclass(frozen=True)
class XciKappas:
    """XCI 的九个系数（Eq. (16)/(17)/(18) 与附录 A）。

    Attributes
    ----------
        kappa11: X1 的 GN 项。
        kappa12: X1 的 4 阶修正。
        kappa21: X2 的 GN 项。
        kappa22: X2 的 4 阶修正。
        kappa31: X3 的 GN 项。
        kappa32: X3 的 4 阶修正。
        kappa41: X4 的 GN 项。
        kappa42: X4 的 4 阶修正（两行相加）。
        kappa43: X4 的 6 阶修正。
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
    """MCI 的三个系数（Eq. (24)/(25)/(26)）。

    Attributes
    ----------
        kappa_m0: M0 域的 GN 项。
        kappa_m1_1: M1 域的 GN 项。
        kappa_m1_2: M1 域的 4 阶修正。
    """

    kappa_m0: float
    kappa_m1_1: float
    kappa_m1_2: float


def sci_kappas(
    f_hz: float,
    n_spans: int,
    fiber: Fiber = CASE.fiber,
    npx: int = 8,
) -> SciKappas:
    """SCI 的 ``kappa1`` / ``kappa2`` / ``kappa3``（Eq. (7)/(8)/(9)）。

    Args:
        f_hz: 被干扰频率 (Hz)，相对 CUT 中心。
        n_spans: 跨数。
        fiber: 光纤参数。
        npx: 每个振荡子段的 Gauss-Legendre 节点数。

    Returns
    -------
        :class:`SciKappas`。
    """
    rs = CASE.symbol_rate_hz
    spec = rs**-6
    half = 0.5 * rs
    band = (-half, half)
    gn = _double_integral(f_hz, band, band, band, "gn", n_spans, fiber, npx)
    l1 = _double_integral(f_hz, band, band, band, "l1", n_spans, fiber, npx)
    l2 = _double_integral(f_hz, band, band, band, "l2", n_spans, fiber, npx)
    sixth = _double_integral(f_hz, band, band, band, "sixth", n_spans, fiber, npx)
    kappa1 = (16.0 / 27.0) * rs**3 * spec * gn.real
    # Eq. (8)：第一行 (80/81) 用附录 C 的 Eq. (43) 降维（外层是 f1），
    # 第二行 (16/81) 用 Eq. (44) 降维（外层换成第三个波 f3）——两者不是同一个积分。
    kappa2 = ((80.0 / 81.0) * l1.real + (16.0 / 81.0) * l2.real) * rs**2 * spec
    kappa3 = (16.0 / 81.0) * rs * spec * abs(sixth) ** 2
    return SciKappas(kappa1=kappa1, kappa2=kappa2, kappa3=kappa3, l1=l1.real, l2=l2.real)


def xci_kappas(
    f_c_hz: float,
    f_hz: float,
    n_spans: int,
    fiber: Fiber = CASE.fiber,
    npx: int = 8,
) -> XciKappas:
    """单个 INT 信道对 CUT 的 XCI 系数（Eq. (16)/(17)/(18) 与附录 A）。

    Args:
        f_c_hz: INT 相对 CUT 的频偏 (Hz)，取正值（负值由对称性给出）。
        f_hz: 被干扰频率 (Hz)，相对 CUT 中心。
        n_spans: 跨数。
        fiber: 光纤参数。
        npx: 每个振荡子段的 Gauss-Legendre 节点数。

    Returns
    -------
        :class:`XciKappas`。
    """
    rs = CASE.symbol_rate_hz
    spec = rs**-6
    half = 0.5 * rs
    cut = (-half, half)
    int_band = (f_c_hz - half, f_c_hz + half)
    gn11 = _double_integral(f_hz, cut, int_band, int_band, "gn", n_spans, fiber, npx)
    c12 = _double_integral(f_hz, cut, int_band, int_band, "l1", n_spans, fiber, npx)
    gn21 = _double_integral(f_hz, int_band, cut, cut, "gn", n_spans, fiber, npx)
    c22 = _double_integral(f_hz, int_band, cut, cut, "l1", n_spans, fiber, npx)
    gn31 = _double_integral(f_hz, cut, cut, int_band, "gn", n_spans, fiber, npx)
    c32 = _double_integral(f_hz, cut, cut, int_band, "l2", n_spans, fiber, npx)
    gn41 = _double_integral(f_hz, int_band, int_band, int_band, "gn", n_spans, fiber, npx)
    c42_l1 = _double_integral(f_hz, int_band, int_band, int_band, "l1", n_spans, fiber, npx)
    c42_l2 = _double_integral(f_hz, int_band, int_band, int_band, "l2", n_spans, fiber, npx)
    c43 = _double_integral(f_hz, int_band, int_band, int_band, "sixth", n_spans, fiber, npx)
    return XciKappas(
        kappa11=(32.0 / 27.0) * rs**3 * spec * gn11.real,
        kappa12=(80.0 / 81.0) * rs**2 * spec * c12.real,
        kappa21=(32.0 / 27.0) * rs**3 * spec * gn21.real,
        kappa22=(80.0 / 81.0) * rs**2 * spec * c22.real,
        kappa31=(16.0 / 27.0) * rs**3 * spec * gn31.real,
        kappa32=(16.0 / 81.0) * rs**2 * spec * c32.real,
        kappa41=(16.0 / 27.0) * rs**3 * spec * gn41.real,
        # Eq. (34)：两行同样是 (80/81) 与 (16/81)，分别走 Eq. (43) 与 Eq. (44)。
        kappa42=((80.0 / 81.0) * c42_l1.real + (16.0 / 81.0) * c42_l2.real) * rs**2 * spec,
        kappa43=(16.0 / 81.0) * rs * spec * abs(c43) ** 2,
    )


def mci_kappas(
    f_upper_hz: float,
    f_lower_hz: float,
    f_hz: float,
    n_spans: int,
    fiber: Fiber = CASE.fiber,
    npx: int = 8,
) -> MciKappas:
    """MCI 的三个系数（Eq. (24)/(25)/(26)）。

    Args:
        f_upper_hz: 上侧 INT 相对 CUT 的频偏 (Hz)，为正。
        f_lower_hz: 下侧 INT 相对 CUT 的频偏 (Hz)，为负。
        f_hz: 被干扰频率 (Hz)，相对 CUT 中心。
        n_spans: 跨数。
        fiber: 光纤参数。
        npx: 每个振荡子段的 Gauss-Legendre 节点数。

    Returns
    -------
        :class:`MciKappas`。
    """
    rs = CASE.symbol_rate_hz
    spec = rs**-6
    half = 0.5 * rs
    cut = (-half, half)
    up = (f_upper_hz - half, f_upper_hz + half)
    low = (f_lower_hz - half, f_lower_hz + half)
    gn_m0 = _double_integral(f_hz, up, low, cut, "gn", n_spans, fiber, npx)
    gn_m1 = _double_integral(f_hz, low, up, up, "gn", n_spans, fiber, npx)
    c_m1 = _double_integral(f_hz, low, up, up, "l1", n_spans, fiber, npx)
    return MciKappas(
        kappa_m0=2.0 * (16.0 / 27.0) * rs**3 * spec * gn_m0.real,
        kappa_m1_1=4.0 * (16.0 / 27.0) * rs**3 * spec * gn_m1.real,
        kappa_m1_2=2.0 * (80.0 / 81.0) * rs**2 * spec * c_m1.real,
    )


# =============================================================================
# 频带平均与逐信道装配
# =============================================================================
@dataclass
class LinkNli:
    """一个跨数下的逐信道 NLI 分解。

    Attributes
    ----------
        n_spans: 跨数。
        spm_w: 逐信道 SCI 功率 (W)。
        xpm_w: 逐信道 XCI+MCI 功率 (W)。
        gn_spm_w: 只含 GN 项的逐信道 SCI 功率 (W)。
        gn_xpm_w: 只含 GN 项的逐信道 XCI+MCI 功率 (W)。
    """

    n_spans: int
    spm_w: np.ndarray
    xpm_w: np.ndarray
    gn_spm_w: np.ndarray
    gn_xpm_w: np.ndarray


def _frequency_grid(n_points: int) -> np.ndarray:
    """CUT 带内的采样频率（相对 CUT 中心），与被审对象同口径的均匀网格。"""
    return np.linspace(-0.5 * CASE.symbol_rate_hz, 0.5 * CASE.symbol_rate_hz, n_points)


def link_nli(n_spans: int, npx: int = 8, n_freq: int = 41, spm_only: bool = False) -> LinkNli:
    """按论文原式算出 ``n_spans`` 跨下的逐信道 NLI。

    Args:
        n_spans: 跨数。
        npx: 每个振荡子段的 Gauss-Legendre 节点数。
        n_freq: CUT 带内的采样频率点数。
        spm_only: 只算 SPM（跳过 XCI/MCI）。

    Returns
    -------
        :class:`LinkNli`。
    """
    case = CASE
    phi_a, psi_a = MODULATION_TABLE[case.modulation]
    phi_b, psi_b = phi_a, psi_a
    freqs = _frequency_grid(n_freq)
    n_ch = case.num_channels
    p3 = case.power_w**3

    # --- SCI ---------------------------------------------------------------
    sci_egn = np.zeros(n_freq)
    sci_gn = np.zeros(n_freq)
    for index, f in enumerate(freqs):
        k = sci_kappas(float(f), n_spans, case.fiber, npx)
        sci_egn[index] = k.kappa1 + phi_a * k.kappa2 + psi_a * k.kappa3
        sci_gn[index] = k.kappa1
    sci_egn = p3 * sci_egn
    sci_gn = p3 * sci_gn

    # --- XCI / MCI ---------------------------------------------------------
    sci_egn_band = float(np.sum(sci_egn))
    sci_gn_band = float(np.sum(sci_gn))
    spm_ch = np.full(n_ch, sci_egn_band)
    gn_spm_ch = np.full(n_ch, sci_gn_band)
    if spm_only:
        zeros = np.zeros(n_ch)
        return LinkNli(n_spans, spm_ch, zeros, gn_spm_ch, zeros)

    xci_egn = np.zeros(n_freq)
    xci_gn = np.zeros(n_freq)
    for d in range(1, case.neighbor_span + 1):
        f_c = d * case.spacing_hz
        for index, f in enumerate(freqs):
            k = xci_kappas(f_c, float(f), n_spans, case.fiber, npx)
            xci_egn[index] += (
                k.kappa11
                + phi_b * k.kappa12
                + k.kappa21
                + phi_a * k.kappa22
                + k.kappa31
                + phi_a * k.kappa32
                + k.kappa41
                + phi_b * k.kappa42
                + psi_b * k.kappa43
            )
            xci_gn[index] += k.kappa11 + k.kappa21 + k.kappa31 + k.kappa41
    xci_egn *= p3
    xci_gn *= p3
    xci_egn_sum = float(np.sum(xci_egn))
    xci_gn_sum = float(np.sum(xci_gn))

    mci_egn = np.zeros(n_freq)
    mci_gn = np.zeros(n_freq)
    for index, f in enumerate(freqs):
        k = mci_kappas(case.spacing_hz, -case.spacing_hz, float(f), n_spans, case.fiber, npx)
        mci_egn[index] = k.kappa_m0 + k.kappa_m1_1 + phi_b * k.kappa_m1_2
        mci_gn[index] = k.kappa_m0 + k.kappa_m1_1
    mci_egn *= p3
    mci_gn *= p3
    mci_egn_sum = float(np.sum(mci_egn))
    mci_gn_sum = float(np.sum(mci_gn))

    # 逐信道装配：每个 CUT 的邻居数（边缘信道更少）。
    xpm_ch = np.zeros(n_ch)
    gn_xpm_ch = np.zeros(n_ch)
    for i in range(n_ch):
        n_up = min(case.neighbor_span, n_ch - 1 - i)
        n_low = min(case.neighbor_span, i)
        has_mci = 1 if (i > 0 and i + 1 < n_ch) else 0
        xpm_ch[i] = (n_up + n_low) * xci_egn_sum + has_mci * mci_egn_sum
        gn_xpm_ch[i] = (n_up + n_low) * xci_gn_sum + has_mci * mci_gn_sum
    return LinkNli(n_spans, spm_ch, xpm_ch, gn_spm_ch, gn_xpm_ch)


# =============================================================================
# buildup
# =============================================================================
def _buildup(link: dict[int, LinkNli], n_spans: int, kind: str) -> float:
    """按 ``NLI(N) / (N * NLI(1))`` 算 buildup。"""
    one = link[1]
    many = link[n_spans]
    if kind == "spm":
        num = float(np.sum(many.spm_w))
        den = float(np.sum(one.spm_w))
    elif kind == "xpm":
        num = float(np.sum(many.xpm_w))
        den = float(np.sum(one.xpm_w))
    elif kind == "gn_spm":
        num = float(np.sum(many.gn_spm_w))
        den = float(np.sum(one.gn_spm_w))
    elif kind == "gn_xpm":
        num = float(np.sum(many.gn_xpm_w))
        den = float(np.sum(one.gn_xpm_w))
    else:
        raise ValueError(kind)
    if den <= 0.0:
        return float("nan")
    return num / (n_spans * den)


def repo_total_buildup(spm_b: float, xpm_b: float) -> float:
    """用冻结的单跨权重把两个 buildup 指标合成成"仓库口径"的总量 buildup。

    ``NLI(N) = NLI(1) * N * buildup``，于是总量指标为

    ``(SPM(1) * b_spm + XPM(1) * b_xpm) / (SPM(1) + XPM(1))``。

    Args:
        spm_b: SPM 的 buildup 指标。
        xpm_b: XPM 的 buildup 指标。

    Returns
    -------
        总量 buildup 指标。
    """
    s = REPO_SPM_SINGLE_SPAN_W * spm_b
    x = REPO_XPM_SINGLE_SPAN_W * xpm_b
    return (s + x) / (REPO_SPM_SINGLE_SPAN_W + REPO_XPM_SINGLE_SPAN_W)


def buildups(npx: int = 16, n_freq: int = 41, spans: tuple[int, ...] = (1, 10)) -> dict[str, float]:
    """跑一遍独立求积，返回各 buildup 指标。

    Args:
        npx: 每个振荡子段的 Gauss-Legendre 节点数。
        n_freq: CUT 带内的采样频率点数。
        spans: 要计算的跨数（至少含 1）。

    Returns
    -------
        字典，键为 ``"gn_spm"``/``"egn_spm"``/``"gn_xpm"``/``"egn_xpm"``/``"total_egn"``。
    """
    started = time.perf_counter()
    link = {n: link_nli(n, npx=npx, n_freq=n_freq) for n in spans}
    out = {
        "gn_spm": _buildup(link, spans[-1], "gn_spm"),
        "egn_spm": _buildup(link, spans[-1], "spm"),
        "gn_xpm": _buildup(link, spans[-1], "gn_xpm"),
        "egn_xpm": _buildup(link, spans[-1], "xpm"),
    }
    out["total_egn"] = repo_total_buildup(out["egn_spm"], out["egn_xpm"])
    out["elapsed_s"] = time.perf_counter() - started
    return out
