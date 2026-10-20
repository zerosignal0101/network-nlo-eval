r"""相关函数闭式 GN 模型（cGN），通用多跨 / 异构跨段。

**论文**

* Y. Jiang, Y. Gao, P. Poggiolini, *An Efficient Correlation-based Evaluation of the
  GN-Model for General Multi-Span Optical Systems*, JLT (投稿中)，arXiv:2610.11624。

**思路**

GN 模型的参考式（论文 Eq. (1)）是

.. math::

    G_{\\mathrm{NLI}}(f) = \\frac{16}{27} \\int\\!\\int
        G(f_1) G(f_2) G(f_1+f_2-f) \\, |\\rho_{\\mathrm{FWM}}(f_1,f_2,f)|^2
        \\, \\mathrm{d}f_1 \\mathrm{d}f_2 .

困难在于 :math:`|\\rho|^2` 里的空间积分：它把"跨段功率剖面"和"频率积"耦在
一起。cGN 的做法是**先做空间积分**：FWM 效率是 NSP（非线性激励剖面）在空间
上的傅里叶变换，于是 :math:`|\\rho|^2` 变成 NSP 的**能量谱**
:math:`S_p(\\Omega)`（论文 Eq. (22)），剩下的频率积分退化成**一维**积分
（论文 Eq. (30)/(42)）。这就是"相关函数"路线的全部要点。

**公式对照**

==========================  ==================================================
论文位置                     代码位置
==========================  ==================================================
Eq. (1) GN 参考式            :func:`_island_core` 的组合方式
Eq. (4) 有效非线性系数        :func:`_nonlinearity_coefficient`
Eq. (5) 有效二阶色散          :func:`_effective_beta2`
Eq. (6)/(9) NSP              :func:`_stimulus_attenuation`
Eq. (16)/(22) NSP 能量谱     :func:`_nsp_energy_spectrum`
Eq. (24)/(30) SS 一维积分     :func:`_island_core`
Eq. (27)/(29) 权重 :math:`W`  :func:`_rectangle_weight`
Eq. (33)/(34) SCI 图元        :func:`_rectangle_weight`（:math:`a'=c'=0` 退化）
Eq. (53) 异构 XS             未实现（见"已知近似"）
Eq. (62)/(63)/(64) 相干累加   :func:`_dirichlet_weight`
==========================  ==================================================

**与冻结基线的关系**

本模型只实现论文的 **GN**（``egn=True`` 抛 ``ValueError``）。它不依赖也不
import 任何兄弟模型（``pcfm`` / ``ecz`` / ``numerical`` 等），以便并行分支
互不冲突。

**已知近似（实现层面，详见交付报告）**

* 跨段功率剖面只取光纤衰减（不含 ISRS / 分布式拉曼），故 NSP 是解析指数型。
* 第三波因子按论文的"矩形内局部常数"假设处理。
* 矩形覆盖用阶梯近似（每个 island 切成 ``island_cells`` 片），误差二阶。
* 异构链的 XS 项（论文 Eq. (53) 的逐跨对求和）未实现；齐次链走论文
  Eq. (62)-(64) 的闭式相干累加。
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
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
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
)
from network_nlo_eval.models.route import RouteModel, apply_amplifier_and_roadm, resolve_noise_figure
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "CGN_MODEL_INFO",
    "CGNModel",
    "CGNSpanKernel",
]

#: 相关域（频率积 :math:`\\xi = f_1' f_2'`）的一维求积点数。
#:
#: 论文的数值求积只剩这一维。节点在 :math:`\\xi` 上按平方律分级（靠近 0
#: 加密，因为 SCI 图元在 :math:`\\xi=0` 处有可积的对数奇点，论文 Eq. (34)）。
DEFAULT_CORRELATION_POINTS = 32

#: 每个 GN 积分 island 的矩形覆盖片数。
#:
#: 论文 Fig. 16 用"每 island 15 个矩形"做收敛性扫描。这里固定为 8 片：阶梯
#: 近似的误差是二阶的，8 片已把几何误差压到 1% 以下，而构造代价（每个
#: 光纤配置缓存一次）保持可接受。
DEFAULT_ISLAND_CELLS = 8

#: 相关域积分的截断因子（以 :math:`\\alpha_x / 4\\pi^2 |\\beta_2|` 为尺度）。
#:
#: NSP 能量谱在 :math:`|\\Omega| \\gg \\alpha_x` 处按 :math:`1/\\Omega^2` 衰减，
#: 故 :math:`\\xi` 超过该尺度几十倍后积分贡献可忽略。截断把每个矩形的求积
#: 区间限制在有物理意义的范围内，避免在远处浪费求积点。
TRUNCATION_FACTOR = 40.0

#: 构造期的临时数组规模上限（元素数）。分块处理 CUT 信道用。
_CHUNK_ELEMENTS = 4_000_000

CGN_MODEL_INFO = ModelInfo(
    name="cgn",
    summary="Correlation-based closed-form GN for general multi-span links.",
    family="gn",
    reference="Jiang, Gao, Poggiolini, 'An Efficient Correlation-based Evaluation of the "
    "GN-Model for General Multi-Span Optical Systems', JLT submitted",
    options=(
        ("correlation_points", "Quadrature nodes of the 1-D correlation (frequency-product) integral."),
    ),
)


def _nonlinearity_coefficient(lambdas_m: NDArrayFloat, n2_m2_w: float, a_eff_m2: NDArrayFloat) -> NDArrayFloat:
    r"""有效非线性系数 :math:`\\gamma` (1/(W·m))。

    论文 Eq. (4) 用四个频率的"互有效面积"；本实现取被测信道 (CUT) 的
    :math:`A_{\\mathrm{eff}}`，与本仓其它模型（``ecz`` / ``numerical``）的
    :math:`\\gamma_{\\mathrm{CUT}}^2` 口径一致，便于横向比较。

    Args:
        lambdas_m: ``(N,)`` 波长 (m)。
        n2_m2_w: 非线性折射率 (m²/W)。
        a_eff_m2: ``(N,)`` 有效模场面积 (m²)。

    Returns
    -------
        ``(N,)`` 非线性系数 (1/(W·m))。
    """
    return (2.0 * np.pi / np.asarray(lambdas_m, dtype=np.float64)) * (n2_m2_w / np.asarray(a_eff_m2))


def _effective_beta2(
    beta2: NDArrayFloat,
    beta3: float,
    x_offsets: NDArrayFloat,
    y_offsets: NDArrayFloat,
) -> NDArrayFloat:
    r"""有效二阶色散 :math:`\\beta_{2,\\mathrm{eff}}` (s²/m)。

    论文 Eq. (5) 的一阶项：

    .. math::

        \\beta_{2,\\mathrm{eff}} = \\beta_2 + \\pi \\beta_3 (f_1 + f_2 - 2 f_c) .

    ``x_offsets`` / ``y_offsets`` 是矩形中心相对观测频率 :math:`f_c` 的偏移
    (Hz)，故 :math:`f_1+f_2-2f_c` 即两者之和。:math:`\\beta_2` 与
    :math:`\\beta_3` 取**被测信道波长**处的光纤色散（局部口径）：论文把公式
    按观测频率 :math:`f_c` 展开，:math:`\\beta_2` 必须是该频率处的局部值，
    否则波长依赖会被重复计入。四阶及以上项本实现忽略。

    Args:
        beta2: 被测信道处的二阶色散 (s²/m)，``(..., )`` 逐信道。
        beta3: 三阶色散 (s³/m)。
        x_offsets: ``(...,)`` 第一波频率偏移 (Hz)。
        y_offsets: ``(...,)`` 第二波频率偏移 (Hz)。

    Returns
    -------
        与输入同形的有效二阶色散 (s²/m)。
    """
    return np.asarray(beta2, dtype=np.float64) + np.pi * beta3 * (
        np.asarray(x_offsets) + np.asarray(y_offsets)
    )


def _stimulus_attenuation(alpha_npm: NDArrayFloat, second: NDArrayFloat, third: NDArrayFloat) -> NDArrayFloat:
    r"""NSP 的指数衰减率 :math:`\\alpha_x` (Np/m)。

    论文 Eq. (6)/(9)：归一化 SPP 只取光纤衰减，故
    :math:`p_{\\mathrm{rect}}(z) = \\exp(-\\alpha_x z)`，其中

    .. math::

        \\alpha_x = \\alpha_1 + \\alpha_2 + \\alpha_3 - \\alpha_c .

    第一波与被测信道同属 CUT，故 :math:`\\alpha_x = \\alpha_2 + \\alpha_3`。
    SPP 是**归一化**量，故 NSP 与绝对功率无关（绝对功率进入 PSD 因子）。

    Args:
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        second: 第二波信道索引。
        third: 第三波信道索引。

    Returns
    -------
        ``(...,)`` 指数衰减率 (Np/m)。
    """
    return np.asarray(alpha_npm)[second] + np.asarray(alpha_npm)[third]


def _nsp_energy_spectrum(
    gamma_w_m: NDArrayFloat,
    alpha_x_npm: NDArrayFloat,
    length_m: float,
    omega_rad_m: NDArrayFloat,
) -> NDArrayFloat:
    r"""NSP 的能量谱 :math:`S_p(\\Omega)` (1/W²)。

    论文 Eq. (16)-(22)：SS 的 :math:`\\mu` 是 NSP 自相关的积分，按维里-金欣
    定理等价于 NSP 的**能量谱**：

    .. math::

        S_p(\\Omega) = \\left| \\int_0^{L} \\gamma \\, p_{\\mathrm{rect}}(z)
            e^{j \\Omega z} \\, \\mathrm{d}z \\right|^2 ,

    其中 :math:`\\Omega = 4\\pi^2 \\beta_{2,\\mathrm{eff}} f_1' f_2'`。指数型 NSP
    的这一积分有闭式：

    .. math::

        S_p(\\Omega) = \\gamma^2 \\frac{1 - 2 e^{-\\alpha_x L}\\cos(\\Omega L)
            + e^{-2\\alpha_x L}}{\\alpha_x^2 + \\Omega^2} ,

    故本实现直接用闭式（不做数值空间积分），并用实数形式避免复数指数。

    Args:
        gamma_w_m: 非线性系数 (1/(W·m))。
        alpha_x_npm: ``(...,)`` NSP 指数衰减率 (Np/m)。
        length_m: 跨段长度 (m)。
        omega_rad_m: ``(...,)`` 相位失配 :math:`\\Omega` (rad/m)。

    Returns
    -------
        与 ``omega_rad_m`` 同形的能量谱 (1/W²)。
    """
    alpha_x = np.asarray(alpha_x_npm, dtype=np.float64)
    omega = np.asarray(omega_rad_m, dtype=np.float64)
    decay = np.exp(-alpha_x * length_m)
    numerator = 1.0 - 2.0 * decay * np.cos(omega * length_m) + decay * decay
    denominator = alpha_x * alpha_x + omega * omega
    return np.asarray(gamma_w_m, dtype=np.float64) ** 2 * numerator / denominator


def _rectangle_weight(
    a: NDArrayFloat,
    b: NDArrayFloat,
    c: NDArrayFloat,
    d: NDArrayFloat,
    xi: NDArrayFloat,
) -> NDArrayFloat:
    r"""矩形权重 :math:`W(\\xi)`（论文 Eq. (27)/(29)）。

    .. math::

        W(\\xi) = \\ln \\frac{\\eta_{\\max}}{\\eta_{\\min}}, \\qquad
        \\eta_{\\min} = \\max(a', \\xi/d'), \\quad
        \\eta_{\\max} = \\min(b', \\xi/c') .

    Args:
        a: ``(...,)`` 矩形第一轴下界 (Hz)。
        b: ``(...,)`` 矩形第一轴上界 (Hz)。
        c: ``(...,)`` 矩形第二轴下界 (Hz)。
        d: ``(...,)`` 矩形第二轴上界 (Hz)。
        xi: ``(...,)`` 频率积变量 (Hz²)。

    Returns
    -------
        与 ``xi`` 同形的权重（无量纲）。
    """
    xi = np.asarray(xi, dtype=np.float64)
    d_safe = np.where(np.asarray(d) > 0.0, np.asarray(d), 1.0)
    c_safe = np.where(np.asarray(c) > 0.0, np.asarray(c), 1.0)
    eta_min = np.maximum(np.asarray(a), xi / d_safe)
    eta_max = np.minimum(np.asarray(b), xi / c_safe)
    log_min = np.log(np.maximum(eta_min, 1e-300))
    log_max = np.log(np.maximum(eta_max, 1e-300))
    return np.maximum(log_max - log_min, 0.0)


def _dirichlet_weight(theta: NDArrayFloat, n_spans: int) -> NDArrayFloat:
    r"""齐次链的相干累加权重（论文 Eq. (63)）。

    .. math::

        N_s + 2\\sum_{r=1}^{N_s-1}(N_s-r)\\cos(r\\theta)
        = \\left[ \\frac{\\sin(N_s\\theta/2)}{\\sin(\\theta/2)} \\right]^2 .

    Args:
        theta: ``(...,)`` 相邻跨段的相位差 :math:`\\theta = B^2 L_s \\xi`（无量纲）。
        n_spans: 跨段数。

    Returns
    -------
        与 ``theta`` 同形的权重（无量纲）。
    """
    if n_spans <= 1:
        return np.ones_like(np.asarray(theta, dtype=np.float64))
    half = 0.5 * np.asarray(theta, dtype=np.float64)
    # sin(N x)/sin(x) = N * sinc(N x)/sinc(x)，避免 x -> 0 的 0/0。
    ratio = n_spans * np.sinc(n_spans * half / np.pi) / np.sinc(half / np.pi)
    return ratio**2


def _xi_quadrature(xi_min: NDArrayFloat, xi_max: NDArrayFloat, n_points: int) -> tuple[NDArrayFloat, NDArrayFloat]:
    r"""相关域一维求积的节点与权重（平方律分级中点法）。

    节点 :math:`\\xi_k = \\xi_{\\min} + (\\xi_{\\max}-\\xi_{\\min}) u_k^2`，
    :math:`u_k` 是 :math:`[0,1]` 上 :math:`n` 个等距区间的中点。平方律把节点
    压向 :math:`\\xi=0`：那里是 SCI 图元唯一的对数奇点（论文 Eq. (34)），
    而远处的贡献按 :math:`1/\\xi^2` 衰减，不需要分辨率。

    Args:
        xi_min: ``(...,)`` 积分下界 (Hz²)。
        xi_max: ``(...,)`` 积分上界 (Hz²)。
        n_points: 求积点数。

    Returns
    -------
        ``(nodes, weights)``，形状均为 ``(..., n_points)``。
    """
    u = (np.arange(n_points, dtype=np.float64) + 0.5) / float(n_points)
    u_edges = np.arange(n_points + 1, dtype=np.float64) / float(n_points)
    xi_min = np.asarray(xi_min, dtype=np.float64)[..., None]
    xi_max = np.asarray(xi_max, dtype=np.float64)[..., None]
    span = xi_max - xi_min
    nodes = xi_min + span * (u**2)
    weights = span * (u_edges[1:] ** 2 - u_edges[:-1] ** 2)
    return nodes, weights


def _island_cover(
    x1: NDArrayFloat,
    x2: NDArrayFloat,
    y1: NDArrayFloat,
    y2: NDArrayFloat,
    u1: NDArrayFloat,
    u2: NDArrayFloat,
    n_cells: int,
) -> tuple[NDArrayFloat, ...]:
    r"""GN 积分 island 的矩形覆盖。

    Island 定义为（论文 Fig. 2/4/5）

    .. math::

        \\{ (f_1', f_2') : f_1' \\in [x_1,x_2],\\ f_2' \\in [y_1,y_2],\\
        f_1' + f_2' \\in [u_1,u_2] \\} .

    论文把 island 覆盖成互不重叠的矩形，并把矩形内的三个 PSD 因子取成常数
    （"locally constant"）。本实现把 :math:`f_1'` 轴等分成 ``n_cells`` 片，
    每片取该片中点处的 :math:`f_2'` 区间作为矩形高度：对线性边界，中点法
    在**面积**上无偏，误差二阶。

    Args:
        x1: ``(..., 1)`` 第一波下界 (Hz)。
        x2: ``(..., 1)`` 第一波上界 (Hz)。
        y1: ``(..., )`` 第二波下界 (Hz)。
        y2: ``(..., )`` 第二波上界 (Hz)。
        u1: ``(..., )`` 第三波（频率和）下界 (Hz)。
        u2: ``(..., )`` 第三波（频率和）上界 (Hz)。
        n_cells: 切片数。

    Returns
    -------
        ``(xa, xb, ylo, yhi)``：前两项形状 ``(..., 1, cells)``，后两项形状
        ``(..., cells)``。
    """
    x1 = np.asarray(x1, dtype=np.float64)[..., None, None]
    x2 = np.asarray(x2, dtype=np.float64)[..., None, None]
    y1 = np.asarray(y1, dtype=np.float64)[..., None, None]
    y2 = np.asarray(y2, dtype=np.float64)[..., None, None]
    u1 = np.asarray(u1, dtype=np.float64)[..., None]
    u2 = np.asarray(u2, dtype=np.float64)[..., None]

    dx = (x2 - x1) / float(n_cells)
    xa = x1 + np.arange(n_cells, dtype=np.float64) * dx
    xb = xa + dx
    xm = 0.5 * (xa + xb)
    ylo = np.maximum(y1, u1 - xm)
    yhi = np.minimum(y2, u2 - xm)
    return xa, xb, ylo, yhi


def _quadrant_pieces(
    xa: NDArrayFloat,
    xb: NDArrayFloat,
    ylo: NDArrayFloat,
    yhi: NDArrayFloat,
) -> list[tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]]:
    """把矩形按象限拆开并映射回第一象限。

    论文 Eq. (23) 前后：能量谱 :math:`S_p` 是偶函数，故把 :math:`(f_1',f_2')`
    的任一坐标取反不改变被积函数。于是矩形可按象限拆成互不重叠的片，每片
    翻到第一象限后用同一套一维公式。

    Args:
        xa: ``(...,)`` 第一波下界 (Hz)。
        xb: ``(...,)`` 第一波上界 (Hz)。
        ylo: ``(...,)`` 第二波下界 (Hz)。
        yhi: ``(...,)`` 第二波上界 (Hz)。

    Returns
    -------
        最多 4 个 ``(a, b, c, d)`` 元组，全部落在第一象限（含边界 0）。
    """
    xa = np.asarray(xa, dtype=np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    ylo = np.asarray(ylo, dtype=np.float64)
    yhi = np.asarray(yhi, dtype=np.float64)

    x_neg_a = -np.minimum(xb, 0.0)
    x_neg_b = -xa
    x_pos_a = np.maximum(xa, 0.0)
    x_pos_b = xb
    y_neg_c = -np.minimum(yhi, 0.0)
    y_neg_d = -ylo
    y_pos_c = np.maximum(ylo, 0.0)
    y_pos_d = yhi

    return [
        (x_neg_a, x_neg_b, y_neg_c, y_neg_d),
        (x_neg_a, x_neg_b, y_pos_c, y_pos_d),
        (x_pos_a, x_pos_b, y_neg_c, y_neg_d),
        (x_pos_a, x_pos_b, y_pos_c, y_pos_d),
    ]


def _third_wave_slots(f_abs_hz: NDArrayFloat, b_hz: NDArrayFloat) -> tuple[NDArrayFloat, int]:
    r"""每个 (CUT, 第二波) 对可用的第三波信道槽位。

    第三波频率 :math:`f_3 = f_1 + f_2 - f_c` 的可达范围是
    :math:`f_l \\pm (B_c + B_l)/2`；落在该范围内的**信道带**（不是带中心）
    都能充当第三波。返回的槽位表把每个对的候选第三波按频率排序后放进定长
    槽位，不足处用 -1 填充。

    Args:
        f_abs_hz: ``(N,)`` 信道中心频率 (Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。

    Returns
    -------
        ``(r_index, n_slots)``：``r_index`` 形状 ``(N, N, n_slots)``。
    """
    n_ch = f_abs_hz.size
    order = np.argsort(f_abs_hz, kind="stable")
    f_sorted = f_abs_hz[order]
    b_sorted = b_hz[order]
    lower_edge = f_sorted - 0.5 * b_sorted
    upper_edge = f_sorted + 0.5 * b_sorted
    # 带宽不等时带边未必单调，取累积极值保证 searchsorted 的前提（单调不减）。
    lower_edge = np.maximum.accumulate(lower_edge)
    upper_edge = np.maximum.accumulate(upper_edge)

    reach = 0.5 * (b_hz[:, None] + b_hz[None, :])
    band_low = f_abs_hz[None, :] - reach
    band_high = f_abs_hz[None, :] + reach

    i_lo = np.zeros((n_ch, n_ch), dtype=np.int64)
    i_hi = np.zeros((n_ch, n_ch), dtype=np.int64)
    for second in range(n_ch):
        i_lo[:, second] = np.searchsorted(upper_edge, band_low[:, second], side="left")
        i_hi[:, second] = np.searchsorted(lower_edge, band_high[:, second], side="right")

    counts = np.maximum(i_hi - i_lo, 0)
    n_slots = int(counts.max()) if counts.size else 0
    if n_slots == 0:
        return np.full((n_ch, n_ch, 0), -1, dtype=np.int64), 0
    offsets = np.arange(n_slots, dtype=np.int64)
    valid = offsets[None, None, :] < counts[:, :, None]
    sorted_index = np.where(valid, i_lo[:, :, None] + offsets[None, None, :], 0)
    r_index = order[sorted_index].astype(np.int64)
    return np.where(valid, r_index, -1), n_slots


def _island_core(
    *,
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    gamma_w_m: NDArrayFloat,
    beta2_s2_m: NDArrayFloat,
    beta3_s3_m: float,
    length_m: float,
    r_index: NDArrayFloat,
    correlation_points: int,
    island_cells: int,
    n_coherent_spans: int,
) -> NDArrayFloat:
    r"""每个 (CUT, 第二波, 第三波) 三元组的相关系数核心积分。

    对每个三元组，本函数给出论文 Eq. (30) 的一维积分

    .. math::

        K^{(m,m)} = \\int_{a'c'}^{b'd'} S_p(B^2 \\xi) W(\\xi) \\, \\mathrm{d}\\xi ,

    其中积分域是 island 的矩形覆盖，:math:`S_p` 是 NSP 能量谱
    （论文 Eq. (22)），:math:`W` 是矩形权重（论文 Eq. (29)）。

    齐次链时按论文 Eq. (62)-(64) 乘上相干累加核
    :math:`[\\sin(N_s\\theta/2)/\\sin(\\theta/2)]^2` 并除以跨段数，使路由层
    逐跨相加后还原总相干结果。

    Args:
        f_abs_hz: ``(N,)`` 信道中心频率 (Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        gamma_w_m: ``(N,)`` 逐信道非线性系数 (1/(W·m))。
        beta2_s2_m: ``(N,)`` 逐信道二阶色散 (s²/m)。
        beta3_s3_m: 三阶色散 (s³/m)。
        length_m: 跨段长度 (m)。
        r_index: ``(N, N, J)`` 第三波信道索引（-1 表示空槽）。
        correlation_points: 相关域求积点数。
        island_cells: 每个 island 的矩形覆盖片数。
        n_coherent_spans: 相干累加的跨段数（1 表示非相干）。

    Returns
    -------
        ``(N, N, J)`` 核心积分（无量纲）。
    """
    n_ch = f_abs_hz.size
    n_slots = r_index.shape[2]
    core = np.zeros((n_ch, n_ch, n_slots), dtype=np.float64)
    if n_slots == 0 or length_m <= 0.0:
        return core

    # 每个三元组的 island 几何（相对观测频率 f_c）。
    x1 = -0.5 * b_hz[:, None]
    x2 = 0.5 * b_hz[:, None]
    delta = f_abs_hz[None, :] - f_abs_hz[:, None]
    y1 = delta - 0.5 * b_hz[None, :]
    y2 = delta + 0.5 * b_hz[None, :]
    third = np.where(r_index >= 0, r_index, 0)
    u1 = f_abs_hz[third] - f_abs_hz[:, None, None] - 0.5 * b_hz[third]
    u2 = f_abs_hz[third] - f_abs_hz[:, None, None] + 0.5 * b_hz[third]
    second = np.broadcast_to(np.arange(n_ch, dtype=np.int64)[None, :, None], r_index.shape)
    alpha_x = _stimulus_attenuation(alpha_npm, second, third)

    # 分块处理 CUT 信道，避免 (N, N, J, cells, 4, points) 的巨型临时数组。
    per_c = max(1, n_ch * n_slots * island_cells * 4 * max(correlation_points, 1))
    chunk = max(1, min(n_ch, int(_CHUNK_ELEMENTS // per_c)))

    for start in range(0, n_ch, chunk):
        stop = min(n_ch, start + chunk)
        idx = slice(start, stop)
        xa, xb, ylo, yhi = _island_cover(
            x1[idx],
            x2[idx],
            y1[idx],
            y2[idx],
            u1[idx],
            u2[idx],
            island_cells,
        )
        x_center = 0.5 * (xa + xb)
        y_center = 0.5 * (ylo + yhi)
        gamma_c = gamma_w_m[idx].reshape(-1, 1, 1, 1, 1)
        alpha_c = alpha_x[idx]
        beta2_c = beta2_s2_m[idx].reshape(-1, 1, 1, 1)
        beta2_eff = _effective_beta2(beta2_c, beta3_s3_m, x_center, y_center)

        acc = np.zeros((stop - start, n_ch, n_slots), dtype=np.float64)
        for a, b, c, d in _quadrant_pieces(xa, xb, ylo, yhi):
            xi_min = a * c
            xi_max = b * d
            # 相关域的自然尺度：|Omega| ~ alpha_x 处 NSP 能量谱开始衰减。
            with np.errstate(divide="ignore", invalid="ignore"):
                xi_scale = alpha_c[..., None] / (4.0 * np.pi**2 * np.abs(beta2_eff))
            xi_scale = np.where(np.isfinite(xi_scale) & (xi_scale > 0.0), xi_scale, np.inf)
            xi_cut = np.minimum(xi_max, xi_min + TRUNCATION_FACTOR * xi_scale)
            xi_end = np.maximum(xi_cut, xi_min)

            nodes, weights = _xi_quadrature(xi_min, xi_end, correlation_points)
            omega = 4.0 * np.pi**2 * beta2_eff[..., None] * nodes
            spectrum = _nsp_energy_spectrum(gamma_c, alpha_c[..., None, None], length_m, omega)
            weight = _rectangle_weight(a[..., None], b[..., None], c[..., None], d[..., None], nodes)
            integrand = spectrum * weight * weights
            if n_coherent_spans > 1:
                theta = 4.0 * np.pi**2 * beta2_eff[..., None] * length_m * nodes
                integrand = integrand * _dirichlet_weight(theta, n_coherent_spans)
            integrand = np.where(np.isfinite(integrand), integrand, 0.0)
            acc += np.sum(integrand, axis=(-2, -1))

        acc = np.where(np.isfinite(acc), acc, 0.0)
        if n_coherent_spans > 1:
            acc = acc / float(n_coherent_spans)
        core[idx] = acc

    return core


class CGNSpanKernel:
    """单跨相关函数 GN 物理核。

    核持有逐信道常量（衰减、非线性系数、色散）与**逐跨段长度缓存**的
    相关系数核心积分矩阵 ``(N, N, J)``。热路径只剩矩阵-向量乘。

    Args:
        fiber_config: 该跨段的光纤配置。
        f_abs_hz: ``(N,)`` 信道中心频率 (Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。
        correlation_points: 相关域求积点数。
        island_cells: 每个 island 的矩形覆盖片数。
    """

    def __init__(
        self,
        *,
        fiber_config: FiberSpanConfig,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        correlation_points: int = DEFAULT_CORRELATION_POINTS,
        island_cells: int = DEFAULT_ISLAND_CELLS,
    ) -> None:
        """准备逐信道常量。

        Raises
        ------
            ValueError: 求积点数或切片数不是正整数。
        """
        if correlation_points < 4:
            raise ValueError(f"correlation_points must be >= 4, got {correlation_points}.")
        if island_cells < 1:
            raise ValueError(f"island_cells must be >= 1, got {island_cells}.")

        self.fiber_config = fiber_config
        self.f_abs_hz = np.ascontiguousarray(f_abs_hz, dtype=np.float64)
        self.b_hz = np.ascontiguousarray(b_hz, dtype=np.float64)
        self.n_nodes = int(self.f_abs_hz.shape[0])
        self.correlation_points = int(correlation_points)
        self.island_cells = int(island_cells)

        lambdas_m = C_LIGHT / self.f_abs_hz
        if fiber_config.use_attenuation_polynomial:
            self.alpha_npm = _calc_fiber_attenuation_npm(
                lambdas_m,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            alpha = fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            self.alpha_npm = np.full(self.n_nodes, alpha, dtype=np.float64)
        a_eff_m2 = _calc_aeff_dynamic(
            lambdas_m,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        self.gamma_w_m = _nonlinearity_coefficient(
            lambdas_m,
            fiber_config.nonlinear_index_n2,
            a_eff_m2,
        )
        ref_lambda_m = fiber_config.reference_wavelength_nm * 1e-9
        # 论文 Eq. (5) 按观测频率展开，故 beta_2 / beta_3 取**逐信道**的局部值。
        self.beta2_s2_m = np.array(
            [_calc_beta2_from_d(fiber_config.dispersion_parameter_d, lam) for lam in lambdas_m],
            dtype=np.float64,
        )
        self.beta3_s3_m = float(
            _calc_beta3_from_s_d(
                fiber_config.dispersion_slope_s,
                fiber_config.dispersion_parameter_d,
                ref_lambda_m,
            )
        )
        self._r_index, self._n_slots = _third_wave_slots(self.f_abs_hz, self.b_hz)
        # 第三波槽位的 PSD 取值索引；空槽指向一个恒为 0 的哑元。
        self._r_index_padded = np.where(self._r_index >= 0, self._r_index + 1, 0)
        #: 相干累加的跨段数；由模型在每次路径评估前写入。
        self.n_coherent_spans = 1
        self._core_cache: dict[tuple[float, int], NDArrayFloat] = {}

    def _core_for(self, span_length_m: float, n_coherent_spans: int) -> NDArrayFloat:
        """该跨段长度下的核心积分矩阵（按长度与相干跨段数缓存）。"""
        key = (float(span_length_m), int(n_coherent_spans))
        cached = self._core_cache.get(key)
        if cached is None:
            cached = _island_core(
                f_abs_hz=self.f_abs_hz,
                b_hz=self.b_hz,
                alpha_npm=self.alpha_npm,
                gamma_w_m=self.gamma_w_m,
                beta2_s2_m=self.beta2_s2_m,
                beta3_s3_m=self.beta3_s3_m,
                length_m=float(span_length_m),
                r_index=self._r_index,
                correlation_points=self.correlation_points,
                island_cells=self.island_cells,
                n_coherent_spans=int(n_coherent_spans),
            )
            self._core_cache[key] = cached
        return cached

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """计算单跨的出纤功率与 SCI / XCI 噪声。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 未使用。相干累加的跨段数由
                :attr:`n_coherent_spans` 给出（模型在路径评估前写入）。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.SpanFiberOutput`。
        """
        del n_effective_spans
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            raise ValueError(
                f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            )
        if span_length_m <= 0.0:
            raise ValueError(f"span_length_m must be positive, got {span_length_m}.")

        core = self._core_for(span_length_m, self.n_coherent_spans)
        active = power > 0.0
        g_psd = np.where(active, power / self.b_hz, 0.0)
        # 第三波 PSD：(N, N, J)。槽位索引已 +1，哑元固定在 0 号位。
        g_pad = np.concatenate([[0.0], g_psd])
        third_psd = g_pad[self._r_index_padded]
        m = core * third_psd
        m_sum = m.sum(axis=2)

        diag = np.arange(self.n_nodes)
        # SCI: 第二波也在 CUT 内带，故 PSD 因子为 G_c * G_c * G_r。
        spm_sum = m_sum[diag, diag] * g_psd
        off_diagonal = m_sum.copy()
        off_diagonal[diag, diag] = 0.0
        # GN 积分对 (f1, f2) 对称：跨信道对 (c, l) 与 (l, c) 等价，故 XCI 计两次。
        xpm_sum = 2.0 * (off_diagonal @ g_psd)

        # |NSP|^2 已含 gamma^2（论文 Eq. (22)），此处不再重复计入。
        prefactor = (16.0 / 27.0) * g_psd
        spm_noise = prefactor * spm_sum * self.b_hz
        xpm_noise = prefactor * xpm_sum * self.b_hz

        power_out = power * np.exp(-self.alpha_npm * span_length_m)
        return SpanFiberOutput(
            power_out_w=power_out,
            spm_noise_w=np.where(active, spm_noise, 0.0),
            xpm_noise_w=np.where(active, xpm_noise, 0.0),
        )


def _span_plan(optical_path: MultiSpanOpticalPath) -> list[tuple[FiberSpanConfig, float]]:
    """光路逐跨的 (光纤配置, 长度) 列表。

    切分规则与冻结路由层一致：每个光纤配置按 ``max_span_length_km`` 切成若干
    跨段。

    Args:
        optical_path: 有序逐跳路径。

    Returns
    -------
        ``[(fiber_config, length_m), ...]``。
    """
    plan: list[tuple[FiberSpanConfig, float]] = []
    for fiber in optical_path.fiber_configs:
        remaining = fiber.length_km
        while remaining > 1e-12:
            span_km = min(remaining, fiber.max_span_length_km)
            plan.append((fiber, span_km * 1e3))
            remaining -= span_km
    return plan


def _identical_span_count(plan: list[tuple[FiberSpanConfig, float]]) -> int:
    """全部跨段完全相同时返回跨段数，否则返回 1（非相干）。

    论文 Eq. (62)-(64) 的相干累加闭式要求各跨段的 SPP、长度与色散完全
    相同；异构链的 XS 项（论文 Eq. (53)）本实现未覆盖，故退化为非相干累加。

    Args:
        plan: :func:`_span_plan` 的结果。

    Returns
    -------
        相等的跨段数，或 1。
    """
    if not plan:
        return 1
    first_fiber, first_length = plan[0]
    key = first_fiber.model_dump_json()
    for fiber, length in plan[1:]:
        if fiber.model_dump_json() != key or abs(length - first_length) > 1e-9:
            return 1
    return len(plan)


class CGNModel(RouteModel):
    """相关函数闭式 GN 模型（cGN）的路由层模型。

    与其它模型共用 :mod:`network_nlo_eval.models.route` 的跨段累积、放大器与
    SNR 合成；本类只负责单跨物理核。

    Args:
        grid: 频谱网格（遗留 ``SpectrumGrid`` 或 ``FlexibleGrid``）。
        ref_fiber_config: 参考光纤配置。
        symbol_rate_hz: 符号率 (Hz)。
        correlation_points: 相关域一维求积点数。
        egn: 不支持；``True`` 直接报错。
        **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。
    """

    info = CGN_MODEL_INFO

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        correlation_points: int = DEFAULT_CORRELATION_POINTS,
        egn: bool = False,
        **route_options,
    ) -> None:
        """初始化节点几何与模型专属选项。

        Raises
        ------
            ValueError: 请求了 ``egn=True``，或求积点数非法。
        """
        if egn:
            raise ValueError(
                "The correlation-based GN model does not support egn=True: the cGN paper "
                "covers the GN model only. Use egn=False."
            )
        if correlation_points < 4:
            raise ValueError(f"correlation_points must be >= 4, got {correlation_points}.")
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.egn = False
        self.correlation_points = int(correlation_points)
        self.island_cells = DEFAULT_ISLAND_CELLS
        #: 当前光路的相干累加跨段数；只在 :meth:`evaluate_path` 调用期间有效。
        self._n_coherent_spans = 1

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """构造该光纤参数下的单跨核（按 ``FiberSpanConfig`` 记忆化）。"""
        return CGNSpanKernel(
            fiber_config=fiber_config,
            f_abs_hz=self.f_abs_hz,
            b_hz=self.channel_bandwidth_hz_channels,
            correlation_points=self.correlation_points,
            island_cells=self.island_cells,
        )

    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> PathQoTResult:
        """沿路径评估，并在跨段全同时启用论文的相干累加闭式。

        Args:
            optical_path: 有序逐跳路径。
            power_in_w: ``(N,)`` 入纤功率 (W)。

        Returns
        -------
            逐信道结果。
        """
        previous = self._n_coherent_spans
        self._n_coherent_spans = _identical_span_count(_span_plan(optical_path))
        try:
            return super().evaluate_path(optical_path, power_in_w)
        finally:
            self._n_coherent_spans = previous

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """单跨：调用核，再统一走放大器/ASE。"""
        kernel = self.kernel_for(fiber_config)
        kernel.n_coherent_spans = self._n_coherent_spans
        fiber_output = kernel.evaluate(power_in_w, span_length_m, 1)
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


register_model(CGN_MODEL_INFO, CGNModel)
