"""ECZ 闭式 ISRS-GN 模型 + 跨间相干（NLI 加权可见度）。

**为什么是另一个文件而不是改 ``isrs_gn.py``**

``models/isrs_gn.py`` 的 :class:`~network_nlo_eval.models.isrs_gn.MultiBandISRSGN`
是仓库的**冻结基线**：14 个黄金向量（``tests/fixtures/ecz_golden_bbb9ed4.json``）
与外部守卫脚本 ``scripts/check_golden.py`` 对它逐位比对。它必须保持逐位不变，
因此本文件不改动它，而是给同一个物理（ECZ 三角近似 SRS 剖面 + Poggiolini 闭式
GN）**另建一个进入相干档的模型类**：单跨核完全复用冻结实现（逐位一致），只在
路由层需要跨间相位时补上复振幅。

**跨间相干的口径**

跨间相干度按 **NLI 噪声功率**加权，不是频带均匀平均。GN 模型里，一对信号频率
``(f1, f2)`` 通过四波混频在信道内产生 NLI，闭式核是

.. code-block:: text

    W(rho) = |1 - exp(-(alpha + j rho) L)|^2 / (alpha^2 + rho^2),
    rho    = 4 pi^2 |beta2| (f1 - f)(f2 - f)

失配越大的频率对产生的 NLI 功率越少。第 ``k`` 跨生成的噪声比第 0 跨多走了
``k L`` 米，两份拷贝之间的相位差就是

.. code-block:: text

    dphi(d) = rho * d = theta * (d / L),   theta = rho * L （单跨相位失配）

因此可见度必须是**按 NLI 功率加权**的带内平均：

.. code-block:: text

    V(d) = <cos(theta * d / L)>_W

均匀权重（信号谱口径）给出的是两份**信号副本**之间的相干度，不是 NLI 的相干度，
在 C 波段 10 跨链路上低估约 1.0 dB。本文件用与数值参考
（``models/numerical.py``）**同一套口径**，但算法在此**独立实现**（不 import
数值模型，避免两个模型耦死）。

**相干因子与等效相位斜坡**

``N`` 跨上的相干因子

.. code-block:: text

    R(N) = N + 2 sum_{k=1}^{N-1} (N - k) V(k L),   buildup = R / N

路由层按复数相加累加各跨振幅，因此本模型给每跨 NLI 振幅一个**线性相位斜坡**
``phi_k = Theta * (cumulative_length / L)``，斜坡 ``Theta`` 由 ``R`` 反解：

.. code-block:: text

    |sum_k exp(j k Theta)|^2 = [sin(N Theta/2) / sin(Theta/2)]^2 = R

左边在 ``Theta ∈ (0, 2 pi/N)`` 上从 ``N^2`` 单调降到 0，二分即得唯一解。

**已知边界**

* 契约层每跨每信道只允许**一个**复振幅（模方钉死为该跨噪声功率），因此只能表达
  相位，装不下逐频率对的联合权重；GN 加权可见度被压缩成等效斜坡。
* 斜坡反解假定各跨噪声**等幅**（等跨、等入纤功率）。EDFA 逐跨补偿跨损时成立；
  固定增益 EDFA 或含 ROADM 的链路里逐跨幅度不等，路由层的 ``sqrt(transfer)``
  权重会进入相干和，与 ``R(N)`` 偏离（数值参考模型有同样的限制）。
* 互拍频（XPM）的失配变量是**信道间**偏移，量级远大于带内偏移，两份拷贝几乎
  不相干；本模型沿用数值参考的带内均匀平均 ``sinc`` 口径（见交付报告的 XPM 审计）。
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence

import numpy as np

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
    register_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.route import apply_amplifier_and_roadm
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "ECZ_COHERENT_MODEL_INFO",
    "ECZCoherentModel",
]

#: 可见度积分里观测频率的采样点数。
#:
#: 被积函数在频带边缘（可容许的三波区退化）陡、在带内平滑，所以观测频率落在
#: 余弦栅格上。41 点把 C 波段的相干因子锁在 161 点值的 3e-5 以内。
VISIBILITY_FREQUENCY_POINTS = 41

#: 可见度积分每个子区间上的 Gauss-Legendre 节点数。
#:
#: 外层（观测频率）积分的被积函数光滑，200 节点/子区间已经收敛到 1e-5。节点数
#: 固定，使缓存的斜坡代价与算例无关。
VISIBILITY_QUADRATURE_POINTS = 200

#: 洛伦兹原函数表 ``C_m(t)`` 的步长。
#:
#: 表用修正梯形法建立（误差是步长的四阶），1e-3 让可见度精度到 1e-6。
VISIBILITY_TABLE_STEP = 1.0e-3

#: 洛伦兹原函数表覆盖的最大 ``t``。
#:
#: 超出后改用洛伦兹尾部的渐近展开，其在 ``t = 30`` 处优于 1e-8。
VISIBILITY_TABLE_LIMIT = 30.0

#: 洛伦兹原函数表覆盖的最小 ``t`` 范围。
#:
#: 色散极小会让表只剩几个点，最小值让表保持良态（此时也永远走不到渐近尾部）。
VISIBILITY_TABLE_MINIMUM = 1.0

#: 等效相位斜坡的二分步数。
#:
#: 斜坡是单调函数在 ``[0, 2 pi/N]`` 上的根，60 步足以把斜坡解到远低于双精度。
RAMP_BISECTION_STEPS = 60


#: 本模型在注册表里的静态描述。
#:
#: 与冻结基线 ``ecz`` 同物理（ECZ-SRS 三角近似 + 闭式 GN），差别只在于**进入
#: 跨间相干档**：多跨时给出逐跨复振幅，由路由层做 ``|sum a_k|^2``。
ECZ_COHERENT_MODEL_INFO = ModelInfo(
    name="ecz_coherent",
    summary="ECZ closed-form GN (SPM/XPM) with NLI-weighted inter-span coherence.",
    family="gn",
    reference="Lasagni et al., 'A Generalized Raman Scattering Model for Real-Time SNR "
    "Estimation of Multi-Band Systems', JLT 2023, doi:10.1109/jlt.2023.3250751; "
    "inter-span coherence: Semrau, Killey, Bayvel, JLT 2018, doi:10.1109/jlt.2018.2830973",
    options=(
        ("ode_accel", "'off' | 'cached' | 'matvec'; single-span acceleration tier."),
        ("channel_bandwidth_source", "'symbol_rate' | 'occupied'; per-node B_i definition."),
        ("power_width_mode", "'fixed_power' | 'fixed_psd'; launch-power normalisation."),
    ),
)


def _lorentzian_tail(beta: float, t: NDArrayFloat) -> NDArrayFloat:
    """``int_t^oo cos(beta s)/(1+s^2) ds``，反复分部积分得到。

    Args:
        beta: 被积函数的角频率（无量纲）。
        t: 下限 (m)，非负数组。

    Returns
    -------
        ``(N,)`` 尾部积分。
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


class _LorentzianAntiderivative:
    r"""洛伦兹原函数 ``C_m(t) = int_0^t cos(m alpha L s)/(1+s^2) ds``（``m >= 1``）。

    表用修正梯形法建立：欧拉-麦克拉urin 修正把梯形法的二阶误差升级到四阶，
    代价只有辛普森网格的一小部分。``C_0(t) = arctan(t)`` 走解析式。表外用洛伦兹
    尾部的渐近展开，并以 ``C_m(oo) = (pi/2) exp(-m alpha L)``（洛伦兹的傅里叶
    变换）锚定。

    Args:
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        span_length_m: 跨段长度 (m)。
        m_max: 需要制表的最大索引 ``m``。
        t_end: 表需要覆盖的最大 ``t``。
        step: 表格步长。
    """

    def __init__(
        self,
        alpha_npm_m: float,
        span_length_m: float,
        m_max: int,
        t_end: float,
        step: float = VISIBILITY_TABLE_STEP,
    ) -> None:
        """建立 ``C_m`` 的查表。"""
        self.alpha = float(alpha_npm_m)
        self.span_length = float(span_length_m)
        self.beta = self.alpha * self.span_length
        limit = min(max(float(t_end), VISIBILITY_TABLE_MINIMUM), VISIBILITY_TABLE_LIMIT)
        n_points = max(3, int(math.ceil(limit / step)) + 1)
        if n_points % 2 == 0:
            n_points += 1
        self.t = np.linspace(0.0, limit, n_points)
        h = float(self.t[1] - self.t[0])
        # 所有 m 的表一次性算出来：cos/sin 的 ufunc 调用只付一次代价，数值与
        # 逐 m 计算逐位一致（同样的表达式、同样的累加顺序）。
        index = np.arange(1, int(m_max) + 1, dtype=np.float64)[:, None]
        phase = index * self.beta * self.t[None, :]
        denominator = 1.0 + self.t * self.t
        values = np.cos(phase) / denominator[None, :]
        derivative = (
            -index * self.beta * np.sin(phase) / denominator[None, :]
            - 2.0 * self.t[None, :] * np.cos(phase) / denominator[None, :] ** 2
        )
        table = self._cumulative(values, derivative, h)
        self.tables: dict[int, NDArrayFloat] = {m: table[m - 1] for m in range(1, int(m_max) + 1)}

    @staticmethod
    def _cumulative(values: NDArrayFloat, derivative: NDArrayFloat, h: float) -> NDArrayFloat:
        """均匀网格上的累积积分（修正梯形法）。

        Args:
            values: 被积函数在网格上的取值（逐行独立积分）。
            derivative: 被积函数的导数在网格上的取值。
            h: 网格步长。

        Returns
        -------
            与 ``values`` 同形的累积积分。
        """
        integral = np.zeros_like(values)
        integral[:, 1:] = np.cumsum(0.5 * h * (values[:, :-1] + values[:, 1:]), axis=1)
        # Euler-Maclaurin: int = T - h^2/12 (f'(b) - f'(a)) + O(h^4).
        integral[:, 1:] -= h * h / 12.0 * (derivative[:, 1:] - derivative[:, 0:1])
        return integral

    def C(self, m: int, t: NDArrayFloat) -> NDArrayFloat:
        """``C_m(t)``，``t`` 可以是负数（``C_m`` 是奇函数）。

        Args:
            m: 索引 ``m >= 0``。
            t: 求值点 (m)。

        Returns
        -------
            ``C_m(t)`` 的取值数组。
        """
        if m == 0:
            return np.arctan(np.asarray(t, dtype=np.float64))
        t = np.asarray(t, dtype=np.float64)
        out = np.empty_like(t)
        sign = np.sign(t)
        magnitude = np.abs(t)
        inside = magnitude <= self.t[-1]
        if np.any(inside):
            out[inside] = sign[inside] * np.interp(magnitude[inside], self.t, self.tables[m])
        outside = ~inside
        if np.any(outside):
            beta = m * self.beta
            tail = _lorentzian_tail(beta, magnitude[outside])
            out[outside] = sign[outside] * (0.5 * math.pi * math.exp(-beta) - tail)
        return out


@functools.lru_cache(maxsize=8)
def _gauss_legendre_nodes(n_points: int) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Gauss-Legendre 节点与权重（带缓存）。

    ``numpy.polynomial.legendre.leggauss`` 的 Newton 迭代要十几毫秒，而可见度
    只在冷启动时算一次，这里缓存住，冷启动省掉大头。

    Args:
        n_points: 节点数。

    Returns
    -------
        ``(nodes, weights)``，节点在 ``[-1, 1]`` 上。
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return nodes, weights


def _visibility_quadrature_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    n_frequency: int,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """自拍频三波区的求积节点。

    观测频率 ``f`` 在信道带内扫描；对每个 ``f``，两个输入偏移 ``u = f1 - f``、
    ``v = f2 - f`` 与闲频偏移 ``u + v`` 必须都留在带内。内层（对 ``v``）积分是
    解析的，所以外层只需要 ``(f, u)`` 节点。

    观测频率落在余弦栅格上：被积函数在频带边缘（可容许区退化）陡、带内平滑，
    余弦栅格在这些地方收敛。

    Args:
        half_band_hz: 信道带宽的一半 (Hz)。
        rate: ``4 pi^2 |beta2|``，单位频率乘积的失配率。
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        n_frequency: 观测频率点数。
        n_quadrature: 每个子区间的 Gauss-Legendre 节点数。

    Returns
    -------
        归约变量 ``t = |rho| / alpha`` 的上下限，以及每个节点的求积测度。
    """
    if n_frequency > 1:
        index = np.arange(n_frequency, dtype=np.float64)
        frequencies = -half_band_hz * np.cos(np.pi * index / (n_frequency - 1))
        # 非均匀频率栅格的复合梯形权重。
        frequency_weights = np.empty(n_frequency, dtype=np.float64)
        frequency_weights[0] = 0.5 * (frequencies[1] - frequencies[0])
        frequency_weights[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
        frequency_weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
    else:
        frequencies = np.zeros(1, dtype=np.float64)
        frequency_weights = np.ones(1, dtype=np.float64)
    nodes, weights = _gauss_legendre_nodes(n_quadrature)
    lo_parts: list[NDArrayFloat] = []
    hi_parts: list[NDArrayFloat] = []
    measure_parts: list[NDArrayFloat] = []
    for frequency, frequency_weight in zip(frequencies, frequency_weights, strict=True):
        low = -half_band_hz - float(frequency)
        high = half_band_hz - float(frequency)
        for start, stop in ((low, 0.0), (0.0, high)):
            if stop - start <= 0.0:
                continue
            u = 0.5 * (stop - start) * (nodes + 1.0) + start
            weight = 0.5 * (stop - start) * weights
            # 第二个偏移同时受自己的频带与闲频约束。
            v_low = np.maximum(low, low - u)
            v_high = np.minimum(high, high - u)
            valid = v_high > v_low
            if not np.any(valid):
                continue
            u = u[valid]
            weight = weight[valid]
            v_low = v_low[valid]
            v_high = v_high[valid]
            # dv = dt / (rate |u| alpha)，其中 t = |rho| / alpha。
            measure = weight * frequency_weight / (rate * np.abs(u) * alpha_npm_m)
            t_low = rate * u * v_low / alpha_npm_m
            t_high = rate * u * v_high / alpha_npm_m
            lo_parts.append(np.minimum(t_low, t_high))
            hi_parts.append(np.maximum(t_low, t_high))
            measure_parts.append(measure)
    return (
        np.concatenate(lo_parts),
        np.concatenate(hi_parts),
        np.concatenate(measure_parts),
    )


def _gn_weighted_spm_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int = VISIBILITY_FREQUENCY_POINTS,
    n_quadrature: int = VISIBILITY_QUADRATURE_POINTS,
) -> NDArrayFloat:
    """GN 核加权的自拍频（SPM）跨间可见度（单一带宽）。

    权重是闭式 GN 核 ``|1 - exp(-(alpha + j rho) L)|^2/(alpha^2 + rho^2)``；两份
    拷贝之间的相位差是 ``rho * d``（``d`` 为两份拷贝的距离）。把 ``t = rho/alpha``
    代换后，内层积分有解析原函数 ``C_m``，只剩一维求积。

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m)。
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        bandwidth_hz: 信道带宽 (Hz)。
        span_length_m: 单跨长度 (m)：失配与洛伦兹宽度都以**单跨**为尺度。
        distances_m: ``(K,)`` 两份拷贝之间的距离 (m)。
        n_frequency: 观测频率点数。
        n_quadrature: 每个子区间的 Gauss-Legendre 节点数。

    Returns
    -------
        ``(K,)`` 可见度，取值在 ``[-1, 1]``；无色散时为 1。
    """
    distances = np.asarray(distances_m, dtype=np.float64)
    half_band = 0.5 * bandwidth_hz
    # 单位频率乘积的失配率：rho = rate * u * v。
    rate = 4.0 * np.pi**2 * beta2_abs_s2_m
    exp_alpha_l = float(np.exp(-alpha_npm_m * span_length_m))
    t_lo, t_hi, measure = _visibility_quadrature_nodes(
        half_band,
        rate,
        alpha_npm_m,
        n_frequency,
        n_quadrature,
    )
    tables = _LorentzianAntiderivative(
        alpha_npm_m,
        span_length_m,
        int(distances.size) + 1,
        float(np.max(np.abs(t_hi))),
    )
    # 每个 m 的 C_m 只求一次（逐距离复用）：数值与逐次调用完全一致，但插值次数
    # 从 6K 降到 2(K+2)。
    m_max = int(distances.size) + 1
    c_hi = [tables.C(m, t_hi) for m in range(m_max + 1)]
    c_lo = [tables.C(m, t_lo) for m in range(m_max + 1)]
    # 分母：NLI 功率本身，即去掉余弦的权重。
    weight_hi = (1.0 + exp_alpha_l**2) * c_hi[0] - 2.0 * exp_alpha_l * c_hi[1]
    weight_lo = (1.0 + exp_alpha_l**2) * c_lo[0] - 2.0 * exp_alpha_l * c_lo[1]
    total = float(np.sum(measure * (weight_hi - weight_lo)))
    visibility = np.zeros_like(distances)
    for index in range(distances.size):
        k = index + 1
        cosine_hi = (1.0 + exp_alpha_l**2) * c_hi[k] - exp_alpha_l * (c_hi[abs(k - 1)] + c_hi[k + 1])
        cosine_lo = (1.0 + exp_alpha_l**2) * c_lo[k] - exp_alpha_l * (c_lo[abs(k - 1)] + c_lo[k + 1])
        visibility[index] = float(np.sum(measure * (cosine_hi - cosine_lo))) / total
    return visibility


def _self_beating_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: NDArrayFloat,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int = VISIBILITY_FREQUENCY_POINTS,
    n_quadrature: int = VISIBILITY_QUADRATURE_POINTS,
) -> NDArrayFloat:
    """自拍频（SPM）两份拷贝的 NLI 加权可见度。

    带宽相同的信道共享一条可见度曲线，因此按唯一带宽分组计算。

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m)。
        alpha_npm_m: 光纤功率衰减 (Np/m)。
        bandwidth_hz: ``(N,)`` 逐信道带宽 (Hz)。
        span_length_m: 单跨长度 (m)。
        distances_m: ``(K,)`` 两份拷贝之间的距离 (m)。
        n_frequency: 观测频率点数。
        n_quadrature: 每个子区间的 Gauss-Legendre 节点数。

    Returns
    -------
        ``(K, N)`` 可见度，取值在 ``[-1, 1]``；无色散时为 1。
    """
    bandwidths = np.asarray(bandwidth_hz, dtype=np.float64)
    distances = np.asarray(distances_m, dtype=np.float64)
    visibility = np.ones((distances.size, bandwidths.size), dtype=np.float64)
    if distances.size == 0 or bandwidths.size == 0:
        return visibility
    if beta2_abs_s2_m == 0.0 or alpha_npm_m <= 0.0:
        # 无色散时两份拷贝完全相干。
        return visibility
    unique, inverse = np.unique(bandwidths, return_inverse=True)
    for index, bandwidth in enumerate(unique):
        column = _gn_weighted_spm_visibility(
            beta2_abs_s2_m,
            alpha_npm_m,
            float(bandwidth),
            span_length_m,
            distances,
            n_frequency,
            n_quadrature,
        )
        visibility[:, inverse == index] = column[:, None]
    return visibility


def _cross_beating_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
    partner_offset_hz: NDArrayFloat,
) -> NDArrayFloat:
    """互拍频（XPM）两份拷贝的带内平均可见度。

    互拍频对的相位失配由**信道间**偏移主导（干扰信道在 ``Delta`` 处），量级远
    大于带内偏移，因此可见度是 ``cos(c x)`` 在带内的平均：

    .. code-block:: text

        V(d) = sinc(c B / 2),   c = 4 pi^2 |beta2| d Delta

    除非失配落在一个弧度以内，否则它迅速归零。GN 核筛出来的频率对本来就已经
    不相干，两种权重口径给出几乎相同的（非相干）答案（见交付报告的 XPM 审计），
    所以这里保留带内均匀平均。

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m)。
        bandwidth_hz: ``(N,)`` 逐信道带宽 (Hz)。
        distance_m: 两份拷贝之间的距离 (m)。
        partner_offset_hz: ``(N,)`` 到干扰信道的频率偏移 (Hz)。

    Returns
    -------
        ``(N,)`` 可见度，取值在 ``[-1, 1]``；无色散时为 1。
    """
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    offset = np.asarray(partner_offset_hz, dtype=np.float64)
    argument = 2.0 * np.pi**2 * beta2_abs_s2_m * distance_m * offset * bandwidth
    visibility = np.ones_like(bandwidth)
    nonzero = argument != 0.0
    if np.any(nonzero):
        visibility[nonzero] = np.sin(argument[nonzero]) / argument[nonzero]
    return visibility


def _coherent_factor(n_spans: int, visibility: NDArrayFloat) -> NDArrayFloat:
    """逐信道的跨间相干因子 ``R``。

    Args:
        n_spans: 链路跨数。
        visibility: ``(N_spans - 1, N_channels)`` 可见度。

    Returns
    -------
        ``(N_channels,)`` 相干因子：完全相干时是 ``N**2``，完全非相干时是 ``N``。
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return n + 2.0 * (weights * visibility).sum(axis=0)


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """等效跨间相位步长 (rad/跨)：把相干因子反解成相位斜坡。

    路由层按复数相加逐跨振幅，因此 ``N`` 个等幅、相位线性增长的振幅给出的相干
    因子是 Dirichlet 核 ``[sin(N Theta/2)/sin(Theta/2)]^2``。该函数在
    ``(0, 2 pi/N)`` 上从 ``N^2`` 单调降到 0，二分即得唯一解。

    Args:
        n_spans: 链路跨数。
        target: ``(N_channels,)`` 目标相干因子。

    Returns
    -------
        ``(N_channels,)`` 相位步长 (rad)。
    """
    n = int(n_spans)
    if n <= 1:
        return np.zeros(np.shape(target), dtype=np.float64)
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n)
    for _ in range(RAMP_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        factor = (np.sin(0.5 * n * middle) / np.sin(0.5 * middle)) ** 2
        too_large = factor > goal
        low = np.where(too_large, middle, low)
        high = np.where(too_large, high, middle)
    return 0.5 * (low + high)


class ECZCoherentModel(MultiBandISRSGN):
    """带跨间相干的 ECZ 闭式 ISRS-GN 模型。

    单跨物理完全复用冻结基线 :class:`~network_nlo_eval.models.isrs_gn.MultiBandISRSGN`
    （逐位一致）；差别只在于多跨时给出**逐跨复振幅**，由路由层按 ``|sum a_k|^2``
    相干累加。

    Attributes
    ----------
    supports_span_context:
        置 ``True``，路由层才会把 :class:`~network_nlo_eval.models.base.SpanContext`
        传进来。
    """

    #: 模型需要知道自己在链路里的位置，才能给出各跨 NLI 振幅之间的相对相位。
    supports_span_context = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        ode_accel: str = "matvec",
        slots_per_channel: int | None = None,
        channel_bandwidth_source: str = "symbol_rate",
        power_width_mode: str = "fixed_power",
    ) -> None:
        """初始化模型；物理量由单跨核在构造期自行准备。

        Args:
            grid: 频谱网格配置（遗留 ``SpectrumGrid`` 或 ``FlexibleGrid``）。
            ref_fiber_config: 参考光纤配置。
            symbol_rate_hz: 符号速率 (Hz)。
            connections: 灵活栅格连接列表。
            ode_accel: 单跨加速档位，见 :data:`network_nlo_eval.models.isrs_gn.ODE_ACCEL_MODES`。
            slots_per_channel: 每遗留信道占用的槽数。
            channel_bandwidth_source: 逐信道 ``B_i`` 口径。
            power_width_mode: 功率归一化口径。
        """
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz,
            connections=connections,
            ode_accel=ode_accel,
            slots_per_channel=slots_per_channel,
            channel_bandwidth_source=channel_bandwidth_source,
            power_width_mode=power_width_mode,
        )
        # 跨间相干量按 (跨数, 跨长) 记忆化：只依赖这两个数与光纤常量，与入纤
        # 功率无关，因此缓存是幂等的。
        self._coherence_cache: dict[tuple[int, float], tuple[NDArrayFloat, ...]] = {}
        # 单位相位因子按 (跨数, 跨长, 累计长度/跨长) 记忆化。
        self._phase_cache: dict[tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]] = {}

    # --- 光纤实例 ---------------------------------------------------------
    def for_fiber(self, fiber_config: FiberSpanConfig) -> ECZCoherentModel:
        """返回匹配链路光纤参数的缓存评估器。

        与冻结基线的差别只在于类型：子评估器必须是本类，否则它没有跨间相干的
        状态（可见度表、相位斜坡缓存）。

        Args:
            fiber_config: 光纤配置。

        Returns
        -------
            该光纤参数下的评估器。
        """
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = ECZCoherentModel(
                self.grid,
                fiber_config,
                self.symbol_rate_hz,
                connections=self.connections,
                ode_accel=self.ode_accel,
                slots_per_channel=self.slots_per_channel,
                channel_bandwidth_source=self.channel_bandwidth_source,
                power_width_mode=self.power_width_mode,
            )
        return self._fiber_evaluator_cache[key]  # type: ignore[return-value]

    # --- 跨间相干 ----------------------------------------------------------
    def _coherence(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """跨间相干量：相位斜坡与相干因子。

        Args:
            n_spans: 链路跨数。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``(ramp_spm, ramp_xpm, factor_spm, factor_xpm)``：前两个是等效跨间
            相位步长 (rad)，后两个是相干因子（无量纲）。四者都是 ``(N,)``。
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._coherence_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        n = max(1, int(n_spans))
        distances = np.arange(1, n, dtype=np.float64) * float(span_length_m)
        beta2_abs = abs(float(self.beta2_s2_m))
        # 可见度用的衰减：C 波段内光纤衰减的变化远小于一个百分点，用带内平均
        # 即可（与数值参考模型同口径）。
        alpha_visibility = float(np.mean(self.alpha_power_npm_channels))
        bandwidth = self.channel_bandwidth_hz_channels
        if n > 1:
            visibility_spm = _self_beating_visibility(
                beta2_abs,
                alpha_visibility,
                bandwidth,
                float(span_length_m),
                distances,
            )
            visibility_xpm = np.stack(
                [
                    _cross_beating_visibility(
                        beta2_abs,
                        bandwidth,
                        float(distance),
                        self._partner_offset_hz(),
                    )
                    for distance in distances
                ]
            )
        else:
            visibility_spm = np.zeros((0, bandwidth.size), dtype=np.float64)
            visibility_xpm = np.zeros((0, bandwidth.size), dtype=np.float64)
        factor_spm = _coherent_factor(n, visibility_spm)
        factor_xpm = _coherent_factor(n, visibility_xpm)
        ramp_spm = _equivalent_phase_ramp(n, factor_spm)
        ramp_xpm = _equivalent_phase_ramp(n, factor_xpm)
        result = (ramp_spm, ramp_xpm, factor_spm, factor_xpm)
        self._coherence_cache[key] = result
        return result

    def _partner_offset_hz(self) -> NDArrayFloat:
        """到最近邻信道的频率偏移 (Hz)：互拍频可见度由它决定。"""
        if self.f_abs_hz.size > 1:
            offsets = np.abs(self.f_abs_hz[:, None] - self.f_abs_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            return offsets.min(axis=1)
        return np.zeros(self.f_abs_hz.size, dtype=np.float64)

    def _span_phase_factors(
        self,
        span_context: SpanContext,
        span_length_m: float,
    ) -> tuple[NDArrayComplex, NDArrayComplex]:
        """本跨 NLI 振幅的单位相位因子 ``exp(1j phi)``。

        相位随链路上已走过的长度线性增长（等跨时即随跨号线性增长）：在第 ``k``
        跨生成的噪声比第 0 跨多走了 ``k L`` 米，两者之间的相位差就是色散在这段
        距离上积累的失配。

        Args:
            span_context: 本跨在链路里的位置。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``(factor_spm, factor_xpm)``，均为 ``(N,)`` 复数，模长为 1。
        """
        scale = span_context.cumulative_length_m / float(span_length_m)
        key = (int(span_context.total_spans), float(span_length_m), float(scale))
        cached = self._phase_cache.get(key)
        if cached is None:
            ramp_spm, ramp_xpm, _, _ = self._coherence(span_context.total_spans, span_length_m)
            cached = (np.exp(1j * ramp_spm * scale), np.exp(1j * ramp_xpm * scale))
            self._phase_cache[key] = cached
        return cached

    # --- 单跨物理 ----------------------------------------------------------
    def span_fiber_output(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """单跨光纤物理核：出纤功率与 SPM/XPM（多跨时附带跨间相干振幅）。

        单跨（``total_spans <= 1`` 或没有 context）时返回纯功率口径，与冻结基线
        逐位一致。

        Args:
            power_in_w: 各信道入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 有效跨段数；路由层自己做相干累加，这里保持 1。
            span_context: 本跨在链路里的位置。

        Returns
        -------
            出纤功率与 SPM/XPM 噪声；多跨时附带复振幅。
        """
        output = super().span_fiber_output(power_in_w, span_length_m, n_effective_spans)
        if span_context is None or span_context.total_spans <= 1:
            return output
        factor_spm, factor_xpm = self._span_phase_factors(span_context, float(span_length_m))
        amplitude_spm = np.sqrt(output.spm_noise_w) * factor_spm
        amplitude_xpm = np.sqrt(output.xpm_noise_w) * factor_xpm
        return SpanFiberOutput(
            output.power_out_w,
            output.spm_noise_w,
            output.xpm_noise_w,
            amplitude_spm,
            amplitude_xpm,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """路由层单跨步进：光纤 + 放大器（ROADM 由路由层统一处理）。

        与冻结基线的差别只有一处：多跨时把本跨的 NLI 复振幅带上，供路由层做
        ``|sum a_k|^2`` 的相干累加。单跨时逐位不变。

        Args:
            fiber_config: 光纤配置。
            edfa_config: 跨段放大器配置。
            power_in_w: 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            span_context: 本跨在链路里的位置。

        Returns
        -------
            出跨功率与三项噪声；多跨时附带 NLI 复振幅。
        """
        evaluator = self.for_fiber(fiber_config)
        fiber_output = evaluator.span_fiber_output(power_in_w, span_length_m, 1, span_context)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=evaluator._resolve_nf(edfa_config),
            f_abs_hz=evaluator.f_abs_hz,
            bandwidth_hz=evaluator.channel_bandwidth_hz_channels,
        )
        return ElementOutput(
            element.power_out_w,
            element.spm_noise_w,
            element.xpm_noise_w,
            element.ase_noise_w,
            element.nli_amplitude_spm,
            element.nli_amplitude_xpm,
        )


# 注册为可选的 QoT 模型实现；构造函数签名即 ``create_model`` 的透传参数。
register_model(ECZ_COHERENT_MODEL_INFO, ECZCoherentModel)
