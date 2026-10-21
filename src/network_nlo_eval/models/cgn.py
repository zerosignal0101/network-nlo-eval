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
Eq. (16)/(22) NSP 能量谱     :func:`_nsp_energy_spectrum`（无 ISRS）
                             :func:`_isrs_nsp_energy_spectrum`（含 ISRS）
ISRS 剖面（论文未给）        :func:`_isrs_span_profile`（复用仓库拉曼核）
Eq. (24)/(30) SS 一维积分     :func:`_island_core`
Eq. (27)/(29) 权重 :math:`W`  :func:`_rectangle_weight`
Eq. (33)/(34) SCI 图元        :func:`_rectangle_weight`（:math:`a'=c'=0` 退化）
Eq. (53) 异构 XS             未实现（见"已知近似"）
Eq. (62)/(63)/(64) 相干累加   :func:`_dirichlet_weight`（核）
                             :func:`_island_core`（逐点加权的链路级矩阵）
                             :func:`_equivalent_phase_ramp`（等效相位斜坡）
Eq. (64) 的标量压缩           :func:`_coherent_ratio`（目标相干因子）
============================  ==================================================

**跨间相干累加（复振幅口径）**

论文 Eq. (62)-(64) 的相干累加核 :math:`[\sin(N_s\theta/2)/\sin(\theta/2)]^2`
是**成对**的：它同时出现在跨 :math:`j` 与跨 :math:`k` 的联合权重里
（:math:`\cos(r\theta)`，:math:`r=|j-k|`）。而路由层的契约每跨每信道只
允许**一个**复振幅 :math:`a_k`，成对结构无法逐项表达。本模型的处理是：

1. 逐跨给出**单跨**噪声的复振幅：:math:`|a_k|^2` 等于该跨的 ``*_noise_w``
   （契约硬性要求），相位是逐跨线性增长的**等效斜坡** :math:`\psi`；
2. 斜坡由"链路级相干总量 ÷ 单跨总量"反解（:func:`_equivalent_phase_ramp`），
   于是路由层的 :math:`|\sum_k a_k|^2` **精确复现** Dirichlet 核算出的链路
   总量（相对误差 < 1e-9）；
3. 斜坡是**每信道**的（各信道的 :math:`\theta` 结构不同），且只依赖
   ``(n_spans, span_length)`` 与光纤常量（含 ISRS 时还依赖入纤功率，因为
   ISRS 会改变 NSP 的形状）——它随核心矩阵一起缓存，不在热路径重算。

压缩的代价：等效斜坡只能复现链路**总量**，不能复现逐对（逐 :math:`r`）的
结构。总量里"自"项（:math:`r=0`，共 :math:`N P_1`）是**精确**的（就是各跨
振幅的模方之和），"互"项（:math:`r\ge 1`）被压缩成一条斜坡。交付报告给出
逐 :math:`r` 的压缩残差。

单跨链（``total_spans == 1``）走**纯功率**分支：不给出复振幅，路由层沿用
原来的逐跨求和，浮点运算顺序不变，故单跨结果与改造前**逐位一致**。
异构链同样走纯功率分支（论文 Eq. (53) 的 XS 项未实现，见"已知近似"）。

**ISRS（信道间受激拉曼散射）**

第一轮只取光纤衰减，故 NSP 是解析指数型，跨信道功率倾斜被完全忽略。本轮
把 ISRS 纳入：

* 单跨出纤功率改用仓库既有的拉曼剖面函数
  (:func:`~network_nlo_eval.models.numba_kernels._calc_raman_profile_jit` 与
  :func:`~network_nlo_eval.models.numba_kernels._calc_span_out_power_jit`，
  即 ``models/isrs_gn.py`` 自己 import 的那两个) 计算，故功率倾斜与 ``ecz``
  /``isrs_gn`` 同源；
* NSP 不再是解析指数型：跨段功率剖面按 ``isrs_segments`` 段推进求解，NSP
  的能量谱 :math:`S_p(\\Omega)` 在 z 域**从头数值积分**（快速衰减与快速相位
  仍在解析的指数核里，逐段用解析指数变换，避开 z 网格的分辨率瓶颈）。

开关 ``include_isrs`` 保留两种模式，报告给出开/关的对比数字。

**确定性契约**

模型输出只由 (光路, 入纤功率) 决定，与调用历史无关：

* 核心积分矩阵的缓存键带**规范化后的绝对功率**（见 :func:`_canonical_power`），
  且矩阵也只用规范化后的功率构造 → 同一个键永远对应同一个矩阵（缓存幂等）；
* 出纤功率**不进缓存**，每次按当前入纤功率重算。

第二轮曾把缓存键做成"相对功率指纹"（除以最大值后再量化），那会让 -4 dBm 与
-1 dBm 落到同一个键上，缓存里的值取决于"哪个功率先到"，于是同一个
(path, power) 在不同调用历史下给出不同结果。第三轮修掉了这一点。

**色散展开口径**

``dispersion_convention`` 有两档：

``"repo"``（默认）
    :math:`\\beta_2` / :math:`\\beta_3` 取**光纤参考波长**处的值，相位里的
    三阶项按**相对被测信道中心的失谐** :math:`\\Delta f_1 + \\Delta f_2` 展开。
    这是本仓 ``isrs_gn`` / ``numerical`` 使用的口径。

``"paper_absolute"``
    三阶项按**绝对频率** :math:`f_1 + f_2` 展开（论文 Eq. (5) 的字面写法）。
    把 ~193 THz 直接代进 :math:`\\pi\\beta_3(f_1+f_2)` 会引入比
    :math:`|\\beta_2|` 大约 7 倍的伪项，把 NLI 压到不合理，故**不是默认档**。
    两档的 :math:`\\beta_2` 都取参考波长处的同一标量，以便隔离"三阶项口径"
    这一个变量。

**NSP 指数口径**

``nsp_convention`` 有两档：

``"gn"``（默认）
    GN 模型的空间积分因子是 :math:`\\sqrt{\\rho_1\\rho_2\\rho_3/\\rho_c}`，故
    NSP 指数要**除以 2**：:math:`\\alpha_x = (\\alpha_2+\\alpha_3)/2`（第一波
    与被测信道相消）。这与本仓 ``numerical`` 参考模型一致，零色散解析极限下
    两者吻合到 0.2%。

``"paper_product"``
    指数不除以 2，即 :math:`\\alpha_x = \\alpha_2+\\alpha_3`（第一轮对论文
    Eq. (9) 的读法）。该档下相干极限的 :math:`|S_p|` 偏小约
    :math:`(1+e^{-\\alpha L})^2/4`，零色散极限与 ``numerical`` 差 3.8 倍。

**与冻结基线的关系**

本模型只实现论文的 **GN**（``egn=True`` 抛 ``ValueError``）。它不依赖也不
import 任何兄弟模型（``pcfm`` / ``ecz`` / ``numerical`` 等），以便并行分支
互不冲突；拉曼物理则直接复用仓库的 numba 内核（见上）。

**已知近似（实现层面，详见交付报告）**

* 第三波因子按论文的"矩形内局部常数"假设处理。
* 矩形覆盖用阶梯近似（每个 island 切成 ``island_cells`` 片），误差二阶。
* ISRS 剖面按分段推进求解，段内用仓库的闭式拉曼近似；残余因子按段取常数
  （中点法，二阶）。
* 异构链的 XS 项（论文 Eq. (53) 的逐跨对求和）未实现；异构链退回**纯功率**
  口径（与改造前一致），齐次链走论文 Eq. (62)-(64) 的闭式相干累加。
* 成对的跨间相干结构被压缩成一条**等效相位斜坡**（见"跨间相干累加"）：
  链路总量精确复现（<1e-9），但逐对（逐 :math:`r`）的贡献分布与精确
  Dirichlet 结构不同，残差见交付报告。
* 相干口径的精确性要求逐跨功率传输因子为 1（EDFA 补偿跨损的常规配置）。
  跨间若存在额外损耗（如 ROADM 或未补偿的跨损），路由层按 :math:`\\sqrt{t}`
  加权各跨振幅，等效斜坡不再与链路总量严格相等。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanContext,
    SpanFiberOutput,
    SpanKernel,
    register_model,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
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

#: ISRS 剖面的 z 方向分段数。
#:
#: 跨段功率剖面按这个段数推进求解，NSP 的能量谱再在这些段上做阶梯展开。
#: 阶梯展开是中点法，误差二阶：ISRS 残余因子在整跨上只变化一个量级，
#: 8 段的几何误差在 1% 以下。段数直接进 NSP 的每点求和代价（热路径已
#: 预计算，代价只出现在冷启动）。
DEFAULT_ISRS_SEGMENTS = 8

#: NSP（空间激励剖面）指数口径。
#:
#: ``"gn"``（默认）
#:     GN 模型的空间积分因子是 :math:`\sqrt{\rho_1\rho_2\rho_3/\rho_c}`，
#:     即指数 :math:`(\alpha_1+\alpha_2+\alpha_3-\alpha_c)/2`；第一波在 CUT
#:     带内时 :math:`\alpha_x = (\alpha_2+\alpha_3)/2`。这与本仓 ``numerical``
#:     参考模型一致，零色散解析极限下两者逐位吻合。
#: ``"paper_product"``
#:     指数取 :math:`\alpha_1+\alpha_2+\alpha_3-\alpha_c`（不做平方根），
#:     即第一轮对论文 Eq. (6)/(9) 的读法。它让 :math:`|S_p|` 在相干极限下
#:     偏小约 :math:`(1+e^{-\alpha L})^2/4`（见交付报告的零色散对照）。
NSP_CONVENTIONS = ("gn", "paper_product")
NSP_CONVENTION_DEFAULT = "gn"

#: 色散/三阶色散展开口径。
#:
#: ``"repo"``
#:     :math:`\\beta_2`/\\ :math:`\\beta_3` 取光纤参考波长处的值，三阶项按相对
#:     被测信道中心的失谐展开。本仓 ``isrs_gn`` / ``numerical`` 的口径，默认档。
#: ``"paper_absolute"``
#:     三阶项按绝对频率展开（论文 Eq. (5) 的字面写法）。见模块 docstring 的
#:     说明：它会引入比 :math:`|\\beta_2|` 大约 7 倍的伪项，故不是默认档。
DISPERSION_CONVENTIONS = ("repo", "paper_absolute")
DISPERSION_CONVENTION_DEFAULT = "repo"

#: 拉曼功率转移系数 ``r_f`` 计算所用的"共信道带宽" (Hz)。
#:
#: 与 ``models/isrs_gn.py`` 的 ``delta_f_co_hz`` 同值（石英光纤 SRS 增益谱的
#: 截断频率）。该值决定 ``r_f`` 在频带内/边沿的分支，故必须与兄弟模型一致，
#: 否则功率倾斜对不上。
DELTA_F_CO_HZ = 15e12

#: 相关域积分的截断因子（以 :math:`\\alpha_x / 4\\pi^2 |\\beta_2|` 为尺度）。
#:
#: NSP 能量谱在 :math:`|\\Omega| \\gg \\alpha_x` 处按 :math:`1/\\Omega^2` 衰减，
#: 故 :math:`\\xi` 超过该尺度几十倍后积分贡献可忽略。截断把每个矩形的求积
#: 区间限制在有物理意义的范围内，避免在远处浪费求积点。
TRUNCATION_FACTOR = 40.0

#: 构造期的临时数组规模上限（元素数）。分块处理 CUT 信道用。
_CHUNK_ELEMENTS = 4_000_000

#: 功率规范化时保留的**有效数字**位数（绝对量级）。
#:
#: 核心矩阵（含 ISRS 时）依赖入纤功率的**绝对**值：拉曼功率倾斜的速率正比于
#: 总功率，所以 -4 dBm 与 -1 dBm 的 NSP 形状不同。规范化把功率向量吸附到一个
#: 由 12 位有效数字决定的等距栅格上：
#:
#: * 栅格步长只由 ``max|power|`` 的 12 位有效数字决定，因此同一物理状态下
#:   跨段之间的浮点往返（1 ulp 量级）落在同一个格点 → 整条光路共用一个缓存键；
#: * 栅格是**绝对**的（不是相对最大值的），所以不同绝对功率一定落在不同格点
#:   → 缓存键能区分 -4 dBm 与 -1 dBm。
#:
#: 量化误差为 1e-12 相对量级，对 NLI 的影响在 1e-11 dB 以下，远低于任何物理
#: 意义；换来的是"同一输入逐位可复现"。
CANONICAL_POWER_DIGITS = 12

#: 规范化功率时，逐信道**形状**（相对最大值的分布）保留的有效数字位数。
#:
#: 形状决定 ISRS 功率倾斜的空间结构，必须比功率标度保留得更细，但也要把
#: 浮点往返噪声吸收掉。
CANONICAL_SHAPE_DIGITS = 9

#: 核心矩阵缓存的内存预算（字节）。核心矩阵是 ``(N, N, J)`` 的 double 数组，
#: 479 信道时单个约 5.5 MiB；预算是为了在功率扫描时既不反复重建，也不无界增长。
CORE_CACHE_BUDGET_BYTES = 256 * 1024 * 1024

#: 等效相位斜坡的二分步数。
#:
#: 相干因子 :math:`[\sin(N\psi/2)/\sin(\psi/2)]^2` 在 :math:`\psi\in[0,2\pi/N]`
#: 上单调，二分反解。60 步把区间压到 :math:`2\pi/N/2^{60}`，远低于双精度
#: 分辨率；相干因子的相对误差因此优于 :math:`10^{-17}`。
RAMP_BISECTION_STEPS = 60

CGN_MODEL_INFO = ModelInfo(
    name="cgn",
    summary="Correlation-based closed-form GN with ISRS for general multi-span links.",
    family="gn",
    reference="Jiang, Gao, Poggiolini, 'An Efficient Correlation-based Evaluation of the "
    "GN-Model for General Multi-Span Optical Systems', JLT submitted",
    options=(
        ("correlation_points", "Quadrature nodes of the 1-D correlation (frequency-product) integral."),
        ("include_isrs", "Include the inter-channel Raman (ISRS) power tilt in the NSP."),
        ("dispersion_convention", "'repo' | 'paper_absolute'; beta3 expansion convention."),
        ("nsp_convention", "'gn' | 'paper_product'; NSP exponent convention."),
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
    beta2: float,
    beta3: float,
    x_offsets: NDArrayFloat,
    y_offsets: NDArrayFloat,
    f_center_hz: NDArrayFloat | None = None,
    convention: str = DISPERSION_CONVENTION_DEFAULT,
) -> NDArrayFloat:
    r"""有效二阶色散 :math:`\\beta_{2,\\mathrm{eff}}` (s²/m)。

    论文 Eq. (5) 的一阶项：

    .. math::

        \\beta_{2,\\mathrm{eff}} = \\beta_2 + \\pi \\beta_3 (f_1 + f_2 - 2 f_c) .

    ``x_offsets`` / ``y_offsets`` 是矩形中心相对观测频率 :math:`f_c` 的偏移
    (Hz)，:math:`f_1 = f_c + x`、:math:`f_2 = f_c + y`。四阶及以上项本实现忽略。

    两档口径只差三阶项的展开原点：

    ``"repo"``（默认）
        :math:`\\beta_2` / :math:`\\beta_3` 取**光纤参考波长**处的标量，三阶项
        用相对被测信道中心的失谐 :math:`x + y`。这是本仓 ``isrs_gn`` /
        ``numerical`` / ``route`` 共同建立的口径。

    ``"paper_absolute"``
        三阶项用**绝对频率** :math:`f_1 + f_2 = 2 f_c + x + y`（论文 Eq. (5) 的
        字面写法）。把 ~193 THz 直接代进 :math:`\\pi\\beta_3(f_1+f_2)` 会引入
        比 :math:`|\\beta_2|` 大约 7 倍的伪项，把 NLI 压到不合理，故不默认。

    Args:
        beta2: 参考波长处的二阶色散 (s²/m)，两档口径共用同一个标量。
        beta3: 参考波长处的三阶色散 (s³/m)。
        x_offsets: ``(...,)`` 第一波频率偏移 (Hz)。
        y_offsets: ``(...,)`` 第二波频率偏移 (Hz)。
        f_center_hz: ``(...,)`` 被测信道中心频率 (Hz)；``"paper_absolute"``
            档需要它来还原绝对频率。
        convention: 见 :data:`DISPERSION_CONVENTIONS`。

    Returns
    -------
        与输入同形的有效二阶色散 (s²/m)。

    Raises
    ------
        ValueError: 口径取值非法，或 ``"paper_absolute"`` 档缺
            ``f_center_hz``。
    """
    if convention not in DISPERSION_CONVENTIONS:
        raise ValueError(
            f"dispersion convention must be one of {DISPERSION_CONVENTIONS}, got {convention!r}."
        )
    x_offsets = np.asarray(x_offsets, dtype=np.float64)
    y_offsets = np.asarray(y_offsets, dtype=np.float64)
    if convention == "repo":
        # 相对被测信道中心的失谐：f1 + f2 - 2 f_c = x + y。
        return float(beta2) + np.pi * float(beta3) * (x_offsets + y_offsets)
    if f_center_hz is None:
        raise ValueError("the 'paper_absolute' dispersion convention needs f_center_hz.")
    # 绝对频率：f1 + f2 = 2 f_c + x + y。
    return float(beta2) + np.pi * float(beta3) * (
        2.0 * np.asarray(f_center_hz, dtype=np.float64) + x_offsets + y_offsets
    )


def _stimulus_attenuation(
    alpha_npm: NDArrayFloat,
    second: NDArrayFloat,
    third: NDArrayFloat,
    convention: str = NSP_CONVENTION_DEFAULT,
) -> NDArrayFloat:
    r"""NSP 的指数衰减率 :math:`\\alpha_x` (Np/m)。

    论文 Eq. (6)/(9)：归一化 SPP 只取光纤衰减，故
    :math:`p_{\\mathrm{rect}}(z) = \\exp(-\\alpha_x z)`，其中

    .. math::

        \\alpha_x = \\alpha_1 + \\alpha_2 + \\alpha_3 - \\alpha_c .

    第一波与被测信道同属 CUT，故第一波项与被测信道项相消。

    两档口径（见 :data:`NSP_CONVENTIONS`）：

    ``"gn"``（默认）
        GN 模型的空间积分因子是 :math:`\\sqrt{\\rho_1\\rho_2\\rho_3/\\rho_c}`，
        指数要**除以 2**：:math:`\\alpha_x = (\\alpha_2+\\alpha_3)/2`。这与本仓
        ``numerical`` 参考模型一致。

    ``"paper_product"``
        指数不除以 2（第一轮对论文 Eq. (9) 的读法）。

    SPP 是**归一化**量，故 NSP 与绝对功率无关（绝对功率进入 PSD 因子）。

    Args:
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        second: 第二波信道索引。
        third: 第三波信道索引。
        convention: 见 :data:`NSP_CONVENTIONS`。

    Returns
    -------
        ``(...,)`` 指数衰减率 (Np/m)。

    Raises
    ------
        ValueError: 口径取值非法。
    """
    if convention not in NSP_CONVENTIONS:
        raise ValueError(f"nsp convention must be one of {NSP_CONVENTIONS}, got {convention!r}.")
    total = np.asarray(alpha_npm)[second] + np.asarray(alpha_npm)[third]
    if convention == "gn":
        return 0.5 * total
    return total


def _canonical_power(power: NDArrayFloat) -> tuple[NDArrayFloat, bytes]:
    r"""把入纤功率吸附到规范栅格上，返回规范值与缓存键。

    **为什么需要规范化**

    含 ISRS 时核心积分矩阵依赖入纤功率的**绝对**值（拉曼速率正比于总功率）。
    若缓存键直接用原始功率的字节，则同一物理状态在不同跨段之间的浮点往返
    （1 ulp 量级）会给出不同的键，整条光路要重建 10 次核心矩阵；若缓存键用
    *相对*功率（除以最大值），则不同绝对功率会挤到同一个键上，缓存里存的
    值取决于"哪个功率先到"——这正是"输出依赖调用历史"的根源。

    规范化把两者都解决掉：

    * 标度 ``max|power|`` 先按 :data:`CANONICAL_POWER_DIGITS` 位有效数字取整，
      1 ulp 的往返不会改变它；
    * 逐信道数值按 ``标度 × 10^-CANONICAL_SHAPE_DIGITS`` 的**绝对**栅格取整，
      形状差异（ISRS 倾斜）被保留，绝对功率差异被区分。

    于是：同一物理状态 → 同一规范值 → 同一缓存键 → 同一个核心矩阵；不同
    功率 → 不同键。缓存因此是幂等的。

    Args:
        power: ``(N,)`` 入纤功率 (W)。

    Returns
    -------
        ``(canonical, key)``：``canonical`` 是规范化的功率向量 (W)，``key`` 是
        它的字节串，可直接用作缓存键的一部分。
    """
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return values.copy(), b"empty"
    scale = float(np.max(np.abs(values)))
    if scale <= 0.0:
        return np.zeros_like(values), b"zero"
    exponent = int(np.floor(np.log10(scale)))
    # 标度按有效数字取整：12 位有效数字对 1e-16 的相对噪声是稳定的。
    # 用"尾数取整 + 指数还原"而不是 np.round(scale, decimals)：后者在极端标度
    # （decimals 达到几百）下会溢出成 inf，进而把规范值变成 NaN。
    decade = 10.0**exponent
    if not np.isfinite(decade) or decade <= 0.0:
        # 次正规数：10^exponent 下溢为 0，尾数无法还原，退回原值。
        return values.copy(), values.tobytes()
    mantissa = scale / decade
    mantissa_canon = float(np.round(mantissa, CANONICAL_POWER_DIGITS - 1))
    if mantissa_canon <= 0.0:
        mantissa_canon = float(mantissa)
    scale_canon = mantissa_canon * decade
    step = scale_canon * (10.0 ** (-CANONICAL_SHAPE_DIGITS))
    if not np.isfinite(step) or step <= 0.0:
        # 极端标度（次正规数）：栅格不可用，退回原值，键仍然唯一。
        return values.copy(), values.tobytes()
    canonical = np.round(values / step) * step
    if not np.all(np.isfinite(canonical)):
        return values.copy(), values.tobytes()
    return canonical, canonical.tobytes()


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


def _isrs_span_profile(
    f_rel_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    power_in_w: NDArrayFloat,
    f_m_hz: float,
    f_M_hz: float,
    k_bar: float,
    length_m: float,
    n_segments: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    r"""ISRS 扰动下的跨段功率剖面（分段推进）。

    逐段调用仓库既有的拉曼剖面函数
    :func:`~network_nlo_eval.models.numba_kernels._calc_raman_profile_jit` 与
    :func:`~network_nlo_eval.models.numba_kernels._calc_span_out_power_jit`
    （即 ``models/isrs_gn.py`` 自己 import 的那两个），把跨段切成
    ``n_segments`` 段推进，得到每个信道在段中点处的**归一化 ISRS 增益因子**

    .. math::

        g_i(z) = \frac{P_i(z)}{P_i(0)} e^{\alpha_i z} ,

    以及整跨的有效 ISRS 衰减率

    .. math::

        s_i = -\frac{1}{L}\ln g_i(L) .

    Args:
        f_rel_hz: ``(N,)`` 相对频率 (Hz)。
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        power_in_w: ``(N,)`` 入纤功率 (W)。
        f_m_hz: 频带下边界 (Hz，相对频率)。
        f_M_hz: 频带上边界 (Hz，相对频率)。
        k_bar: 泵浦系数（分布式拉曼泵浦功率占比，上限 1）。
        length_m: 跨段长度 (m)。
        n_segments: 分段数。

    Returns
    -------
        ``(z_mid, gain, rate)``：``z_mid`` 是 ``(n_segments,)`` 段中点位置 (m)，
        ``gain`` 是 ``(n_segments, N)`` 归一化 ISRS 增益因子，``rate`` 是
        ``(N,)`` 有效 ISRS 衰减率 (Np/m)。
    """
    n_seg = max(1, int(n_segments))
    dz = float(length_m) / n_seg
    z_mid = (np.arange(n_seg, dtype=np.float64) + 0.5) * dz

    power = np.array(power_in_w, dtype=np.float64, copy=True)
    launch = np.array(power_in_w, dtype=np.float64, copy=True)
    active = launch > 0.0
    safe_launch = np.where(active, launch, 1.0)

    # 推进步长取粗段长的一半：这样每个粗段**中点**处正好有一个采样点，
    # 归一化增益因子 g(z) = rho(z) e^{alpha z} 才是在中点位置上取到的
    # （用段末功率配段中点的 e^{alpha z} 会把纯衰减算重，g 直接塌成 0）。
    n_sub = 2 * n_seg
    dz_sub = float(length_m) / n_sub
    gain = np.ones((n_seg, launch.size), dtype=np.float64)
    for index in range(n_sub):
        total_w = float(power.sum())
        r_f = _calc_raman_profile_jit(
            np.ascontiguousarray(f_rel_hz, dtype=np.float64),
            float(f_m_hz),
            float(f_M_hz),
            DELTA_F_CO_HZ,
            total_w,
        )
        power = _calc_span_out_power_jit(
            np.ascontiguousarray(power, dtype=np.float64),
            np.ascontiguousarray(f_rel_hz, dtype=np.float64),
            np.ascontiguousarray(alpha_npm, dtype=np.float64),
            dz_sub,
            np.ascontiguousarray(r_f, dtype=np.float64),
            float(k_bar),
        )
        if index % 2 == 0:
            # 第 index 个子步（偶数）之后的位置是 (index+1) dz_sub = 粗段中点。
            coarse = index // 2
            gain[coarse] = (power / safe_launch) * np.exp(alpha_npm * z_mid[coarse])
    gain[:, ~active] = 1.0

    end_gain = (power / safe_launch) * np.exp(alpha_npm * float(length_m))
    end_gain = np.where(active & (end_gain > 0.0), end_gain, 1.0)
    rate = -np.log(end_gain) / float(length_m)
    rate = np.where(active, rate, 0.0)
    return z_mid, gain, rate


def _isrs_nsp_energy_spectrum(
    gamma_w_m: NDArrayFloat,
    alpha_x_npm: NDArrayFloat,
    residual: NDArrayFloat,
    seg_length_m: float,
    omega_rad_m: NDArrayFloat,
) -> NDArrayFloat:
    r"""含 ISRS 的 NSP 能量谱 :math:`S_p(\Omega)` (1/W²)。

    NSP 拆成"解析指数部分 × 慢变残余"：

    .. math::

        \mathrm{NSP}(z) = e^{-\alpha_x' z} \, r(z), \qquad
        \alpha_x' = \alpha_x + s_2 + s_3 ,

    其中 :math:`s_i` 是参与波的 ISRS 有效衰减率，:math:`r(z)` 是残余因子
    （ISRS 速率沿跨段不是常数，故 NSP 不再是解析指数型）。

    :math:`r(z)` 按段取常数（中点值），而指数部分在段内用**解析**指数变换：

    .. math::

        \int_{z_m}^{z_m + \Delta} e^{-(\alpha_x' - j\Omega) z}\, \mathrm{d}z
        = \frac{e^{-s z_m}\left(1 - e^{-s\Delta}\right)}{s},
        \qquad s = \alpha_x' - j\Omega .

    这样既拿到"从头数值积分"的 NSP，又不必用 z 网格去解析
    :math:`\Omega L \sim 10^4` 的快速相位（那需要 10^5 级别的采样点）。

    Args:
        gamma_w_m: 非线性系数 (1/(W·m))。
        alpha_x_npm: ``(...,)`` 含 ISRS 的 NSP 衰减率 (Np/m)。
        residual: ``(..., n_segments)`` 残余因子（段中点取值）。
        seg_length_m: 段长 (m)。
        omega_rad_m: ``(...,)`` 相位失配 :math:`\Omega` (rad/m)。

    Returns
    -------
        与 ``omega_rad_m`` 同形的能量谱 (1/W²)。
    """
    alpha_x = np.asarray(alpha_x_npm, dtype=np.float64)
    omega = np.asarray(omega_rad_m, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)

    # s = alpha_x - j omega；每段的比值 w = e^{-s Δ}。
    # 复指数在这里是热点（每点都要一次），改用实指数 + 实三角函数：
    #   e^{-s Δ} = e^{-alpha_x Δ} (cos(ωΔ) + j sin(ωΔ))
    # 分子 1 - w 用 expm1 / 1-cos 的稳定形式，避免小 |s Δ| 时的相消。
    decay = float(seg_length_m)
    exponent_real = alpha_x * decay
    phase = omega * decay
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    magnitude = np.exp(-exponent_real)
    w = magnitude * (cos_phase + 1j * sin_phase)
    one_minus_w = (-np.expm1(-exponent_real) + magnitude * (1.0 - cos_phase)) - 1j * (
        magnitude * sin_phase
    )

    accumulator = np.zeros_like(w)
    for index in range(residual.shape[-1] - 1, -1, -1):
        accumulator *= w
        accumulator += residual[..., index][..., None, None]
    segment = one_minus_w / (alpha_x - 1j * omega)
    amplitude = segment * accumulator
    gamma = np.asarray(gamma_w_m, dtype=np.float64)
    return gamma**2 * (amplitude.real**2 + amplitude.imag**2)


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


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    r"""把目标相干因子反解成逐跨相位增量（论文 Eq. (63) 的标量压缩）。

    路由层按 :math:`|\sum_k a_k|^2` 相干累加。:math:`N` 个等幅、逐跨相位
    线性增长 :math:`\psi` 的复振幅给出的相干因子是

    .. math::

        \left[\frac{\sin(N\psi/2)}{\sin(\psi/2)}\right]^2 ,

    它与 Dirichlet 核 :func:`_dirichlet_weight` 是同一个函数。本函数反解这个
    关系：返回使相干因子等于 ``target`` 的逐跨相位增量。

    相干因子在 :math:`\psi \in [0, 2\pi/N]` 上从 :math:`N^2` **单调**降到
    :math:`0`，故二分即可。目标值被截断到 :math:`[0, N^2]`（数值上界）。

    Args:
        n_spans: 跨段数。
        target: ``(N,)`` 目标相干因子。

    Returns
    -------
        ``(N,)`` 逐跨相位增量 (rad)。
    """
    n = int(n_spans)
    if n <= 1:
        return np.zeros_like(np.asarray(target, dtype=np.float64))
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n)
    for _ in range(RAMP_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        too_large = _dirichlet_weight(mid, n) > goal
        low = np.where(too_large, mid, low)
        high = np.where(too_large, high, mid)
    return 0.5 * (low + high)


def _nli_from_core(
    core: NDArrayFloat,
    g_psd: NDArrayFloat,
    third_psd: NDArrayFloat,
    b_hz: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    r"""把核心积分矩阵聚合成逐信道的 SCI / XCI 噪声功率 (W)。

    聚合是**线性**的：核心矩阵的每个元素乘上第三波 PSD 后对槽位求和，再对
    第二波求和（SCI 只留对角，XCI 取非对角并计两次，因为 GN 积分对
    :math:`(f_1, f_2)` 对称）。

    Args:
        core: ``(N, N, J)`` 核心积分矩阵。
        g_psd: ``(N,)`` 第二波 PSD (W/Hz)。
        third_psd: ``(N, N, J)`` 第三波 PSD (W/Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。

    Returns
    -------
        ``(spm, xpm)``：逐信道 SCI / XCI 噪声功率 (W)。
    """
    m = core * third_psd
    m_sum = m.sum(axis=2)
    diag = np.arange(core.shape[0])
    # SCI: 第二波也在 CUT 内带，故 PSD 因子为 G_c * G_c * G_r。
    spm_sum = m_sum[diag, diag] * g_psd
    off_diagonal = m_sum.copy()
    off_diagonal[diag, diag] = 0.0
    # GN 积分对 (f1, f2) 对称：跨信道对 (c, l) 与 (l, c) 等价，故 XCI 计两次。
    xpm_sum = 2.0 * (off_diagonal @ g_psd)
    # |NSP|^2 已含 gamma^2（论文 Eq. (22)），此处不再重复计入。
    prefactor = (16.0 / 27.0) * g_psd
    return prefactor * spm_sum * b_hz, prefactor * xpm_sum * b_hz


def _coherent_ratio(link: NDArrayFloat, single: NDArrayFloat) -> NDArrayFloat:
    r"""链路级相干总量与单跨总量之比，即目标相干因子。

    该比值就是 Dirichlet 核 :math:`[\sin(N\psi/2)/\sin(\psi/2)]^2` 应当
    取到的值：路由层按 :math:`|\sum_k a_k|^2` 相干累加，逐跨振幅的幅度是
    单跨噪声，故相干因子乘以单跨总量就是链路级相干总量。

    Args:
        link: ``(N,)`` 链路级相干总量 (W)。
        single: ``(N,)`` 单跨总量 (W)。

    Returns
    -------
        ``(N,)`` 相干因子；单跨总量为 0 的信道取中性值 1。
    """
    link = np.asarray(link, dtype=np.float64)
    single = np.asarray(single, dtype=np.float64)
    positive = single > 0.0
    safe = np.where(positive, single, 1.0)
    return np.where(positive, link / safe, 1.0)


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
    beta2_s2_m: float,
    beta3_s3_m: float,
    dispersion_convention: str,
    nsp_convention: str,
    length_m: float,
    r_index: NDArrayFloat,
    correlation_points: int,
    island_cells: int,
    n_coherent_spans: int,
    isrs_gain: NDArrayFloat | None = None,
    isrs_rate: NDArrayFloat | None = None,
    isrs_z_mid: NDArrayFloat | None = None,
    isrs_seg_length_m: float = 0.0,
) -> NDArrayFloat:
    r"""每个 (CUT, 第二波, 第三波) 三元组的相关系数核心积分。

    对每个三元组，本函数给出论文 Eq. (30) 的一维积分

    .. math::

        K^{(m,m)} = \int_{a'c'}^{b'd'} S_p(B^2 \xi) W(\xi) \, \mathrm{d}\xi ,

    其中积分域是 island 的矩形覆盖，:math:`S_p` 是 NSP 能量谱
    （论文 Eq. (22)），:math:`W` 是矩形权重（论文 Eq. (29)）。

    ``isrs_gain`` 非空时走含 ISRS 的 NSP：NSP 的衰减率加上参与波的 ISRS
    有效衰减率，残余因子按段展开后逐段做解析指数变换（见
    :func:`_isrs_nsp_energy_spectrum`）。

    齐次链时按论文 Eq. (62)-(64) 额外累加一份"链路级相干总量"矩阵：核心积分
    逐点乘上相干累加核 :math:`[\sin(N_s\theta/2)/\sin(\theta/2)]^2`。该矩阵
    **不**除以跨段数——它就是链路级相干总量本身。模型用它反解等效相位斜坡
    （:func:`_equivalent_phase_ramp`），再由路由层按 :math:`|\sum_k a_k|^2`
    真的相干累加。

    Args:
        f_abs_hz: ``(N,)`` 信道中心频率 (Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        gamma_w_m: ``(N,)`` 逐信道非线性系数 (1/(W·m))。
        beta2_s2_m: 参考波长处的二阶色散 (s²/m)。
        beta3_s3_m: 参考波长处的三阶色散 (s³/m)。
        dispersion_convention: 见 :data:`DISPERSION_CONVENTIONS`。
        nsp_convention: 见 :data:`NSP_CONVENTIONS`。
        length_m: 跨段长度 (m)。
        r_index: ``(N, N, J)`` 第三波信道索引（-1 表示空槽）。
        correlation_points: 相关域求积点数。
        island_cells: 每个 island 的矩形覆盖片数。
        n_coherent_spans: 相干累加的跨段数（1 表示非相干）。
        isrs_gain: ``(n_segments, N)`` 逐段中点的归一化 ISRS 增益因子；
            ``None`` 表示不含 ISRS。
        isrs_rate: ``(N,)`` 逐信道有效 ISRS 衰减率 (Np/m)。
        isrs_z_mid: ``(n_segments,)`` 段中点位置 (m)。
        isrs_seg_length_m: 段长 (m)。

    Returns
    -------
        ``(plain, dirichlet)``：``plain`` 是单跨核心积分 ``(N, N, J)``；
        ``dirichlet`` 是逐点乘上相干累加核的链路级核心积分 ``(N, N, J)``
        （``n_coherent_spans == 1`` 时为 ``None``）。
    """
    n_ch = f_abs_hz.size
    n_slots = r_index.shape[2]
    core = np.zeros((n_ch, n_ch, n_slots), dtype=np.float64)
    if n_slots == 0 or length_m <= 0.0:
        return core, None
    dirichlet = None
    if n_coherent_spans > 1:
        dirichlet = np.zeros((n_ch, n_ch, n_slots), dtype=np.float64)

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
    alpha_x = _stimulus_attenuation(alpha_npm, second, third, convention=nsp_convention)

    use_isrs = isrs_gain is not None and isrs_rate is not None
    if use_isrs:
        # NSP 的衰减率加上两束参与波的 ISRS 有效衰减率；口径与纯衰减部分一致
        # （"gn" 档下 ISRS 项同样要除以 2）。
        scale = 0.5 if nsp_convention == "gn" else 1.0
        alpha_x = alpha_x + scale * (isrs_rate[second] + isrs_rate[third])

    # 分块处理 CUT 信道，避免 (N, N, J, cells, 4, points) 的巨型临时数组。
    per_c = max(1, n_ch * n_slots * island_cells * 4 * max(correlation_points, 1))
    chunk = max(1, min(n_ch, int(_CHUNK_ELEMENTS // per_c)))

    for start in range(0, n_ch, chunk):
        stop = min(start + chunk, n_ch)
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
        beta2_eff = _effective_beta2(
            beta2_s2_m,
            beta3_s3_m,
            x_center,
            y_center,
            f_center_hz=f_abs_hz[idx].reshape(-1, 1, 1, 1),
            convention=dispersion_convention,
        )

        residual_view = None
        if use_isrs:
            # 残余因子 r(z)：逐段中点的 ISRS 增益之积，乘回解析指数部分。
            residual = np.empty((stop - start, n_ch, n_slots, isrs_gain.shape[0]), dtype=np.float64)
            third_chunk = third[idx]
            rate_second = isrs_rate[second[idx]]
            for seg in range(isrs_gain.shape[0]):
                gain_seg = isrs_gain[seg]
                residual[..., seg] = (gain_seg[None, :, None] * gain_seg[third_chunk]) * np.exp(
                    (rate_second + isrs_rate[third_chunk]) * isrs_z_mid[seg]
                )
            residual_view = residual

        acc = np.zeros((stop - start, n_ch, n_slots), dtype=np.float64)
        acc_dirichlet = (
            None if dirichlet is None else np.zeros((stop - start, n_ch, n_slots), dtype=np.float64)
        )
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
            if residual_view is None:
                spectrum = _nsp_energy_spectrum(
                    gamma_c, alpha_c[..., None, None], length_m, omega
                )
            else:
                spectrum = _isrs_nsp_energy_spectrum(
                    gamma_c,
                    alpha_c[..., None, None],
                    residual_view,
                    isrs_seg_length_m,
                    omega,
                )
            weight = _rectangle_weight(a[..., None], b[..., None], c[..., None], d[..., None], nodes)
            integrand = spectrum * weight * weights
            integrand = np.where(np.isfinite(integrand), integrand, 0.0)
            # 单跨核心积分：与 n_coherent_spans == 1 的旧路径逐位一致。
            acc += np.sum(integrand, axis=(-2, -1))
            if acc_dirichlet is not None:
                # 链路级相干总量：逐点乘上论文 Eq. (63) 的相干累加核。
                theta = 4.0 * np.pi**2 * beta2_eff[..., None] * length_m * nodes
                weighted = integrand * _dirichlet_weight(theta, n_coherent_spans)
                weighted = np.where(np.isfinite(weighted), weighted, 0.0)
                acc_dirichlet += np.sum(weighted, axis=(-2, -1))

        acc = np.where(np.isfinite(acc), acc, 0.0)
        core[idx] = acc
        if acc_dirichlet is not None:
            acc_dirichlet = np.where(np.isfinite(acc_dirichlet), acc_dirichlet, 0.0)
            dirichlet[idx] = acc_dirichlet

    return core, dirichlet


@dataclass(frozen=True)
class _CoreEntry:
    """核心矩阵缓存的条目。

    Attributes
    ----------
        plain: ``(N, N, J)`` 单跨核心积分矩阵。
        ramp_spm: ``(N,)`` SPM 的等效逐跨相位增量 (rad)；``None`` 表示非相干档。
        ramp_xpm: ``(N,)`` XPM 的等效逐跨相位增量 (rad)；``None`` 表示非相干档。
    """

    plain: NDArrayFloat
    ramp_spm: NDArrayFloat | None = None
    ramp_xpm: NDArrayFloat | None = None


class CGNSpanKernel:
    """单跨相关函数 GN 物理核。

    核持有逐信道常量（衰减、非线性系数、色散）与**逐跨段长度缓存**的
    相关系数核心积分矩阵 ``(N, N, J)``。热路径只剩矩阵-向量乘。

    含 ISRS 时，核心积分矩阵还依赖**入纤功率**（拉曼功率倾斜的速率正比于
    总功率），故缓存键带上**规范化后的绝对功率**；不含 ISRS 时核心与功率
    无关，缓存键只有跨段长度与相干跨段数。

    **确定性契约**：核的输出只由 (跨段长度, 相干跨段数, 入纤功率) 决定，与
    调用历史无关。缓存是幂等的：键里的功率是规范化后的值，矩阵也只用规范化
    后的功率构造，因此"同一个键"永远对应"同一个矩阵"。出纤功率不进缓存，
    每次按当前入纤功率重算。

    Args:
        fiber_config: 该跨段的光纤配置。
        f_abs_hz: ``(N,)`` 信道中心频率 (Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。
        correlation_points: 相关域一维求积点数。
        island_cells: 每个 island 的矩形覆盖片数。
        include_isrs: 是否把 ISRS 功率倾斜纳入 SPP / NSP。
        dispersion_convention: 见 :data:`DISPERSION_CONVENTIONS`。
        isrs_segments: ISRS 剖面的 z 方向分段数。
        band_center_hz: 频带中心 (Hz)；``None`` 时取信道中心频率的均值。
    """

    def __init__(
        self,
        *,
        fiber_config: FiberSpanConfig,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        correlation_points: int = DEFAULT_CORRELATION_POINTS,
        island_cells: int = DEFAULT_ISLAND_CELLS,
        include_isrs: bool = True,
        dispersion_convention: str = DISPERSION_CONVENTION_DEFAULT,
        nsp_convention: str = NSP_CONVENTION_DEFAULT,
        isrs_segments: int = DEFAULT_ISRS_SEGMENTS,
        band_center_hz: float | None = None,
    ) -> None:
        """准备逐信道常量。

        Raises
        ------
            ValueError: 求积点数或切片数不是正整数，或色散口径非法。
        """
        if correlation_points < 4:
            raise ValueError(f"correlation_points must be >= 4, got {correlation_points}.")
        if island_cells < 1:
            raise ValueError(f"island_cells must be >= 1, got {island_cells}.")
        if isrs_segments < 1:
            raise ValueError(f"isrs_segments must be >= 1, got {isrs_segments}.")
        if dispersion_convention not in DISPERSION_CONVENTIONS:
            raise ValueError(
                f"dispersion_convention must be one of {DISPERSION_CONVENTIONS}, "
                f"got {dispersion_convention!r}."
            )
        if nsp_convention not in NSP_CONVENTIONS:
            raise ValueError(
                f"nsp_convention must be one of {NSP_CONVENTIONS}, got {nsp_convention!r}."
            )

        self.fiber_config = fiber_config
        self.f_abs_hz = np.ascontiguousarray(f_abs_hz, dtype=np.float64)
        self.b_hz = np.ascontiguousarray(b_hz, dtype=np.float64)
        self.n_nodes = int(self.f_abs_hz.shape[0])
        self.correlation_points = int(correlation_points)
        self.island_cells = int(island_cells)
        self.include_isrs = bool(include_isrs)
        self.dispersion_convention = str(dispersion_convention)
        self.nsp_convention = str(nsp_convention)
        self.isrs_segments = int(isrs_segments)

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
        # 仓库口径：beta_2 / beta_3 取**光纤参考波长**处的标量，两档口径共用。
        self.beta2_s2_m = float(
            _calc_beta2_from_d(fiber_config.dispersion_parameter_d, ref_lambda_m)
        )
        self.beta3_s3_m = float(
            _calc_beta3_from_s_d(
                fiber_config.dispersion_slope_s,
                fiber_config.dispersion_parameter_d,
                ref_lambda_m,
            )
        )

        # ISRS 前端的几何：相对频率与频带边界（与 models/isrs_gn.py 同口径）。
        center = float(np.mean(self.f_abs_hz)) if band_center_hz is None else float(band_center_hz)
        self.band_center_hz = center
        self.f_rel_hz = np.ascontiguousarray(self.f_abs_hz - center, dtype=np.float64)
        self.f_m_hz = float(np.min(self.f_abs_hz - 0.5 * self.b_hz) - center)
        self.f_M_hz = float(np.max(self.f_abs_hz + 0.5 * self.b_hz) - center)
        self.k_bar = min(sum(pump.power_w for pump in fiber_config.raman_pumps), 1.0)

        self._r_index, self._n_slots = _third_wave_slots(self.f_abs_hz, self.b_hz)
        # 第三波槽位的 PSD 取值索引；空槽指向一个恒为 0 的哑元。
        self._r_index_padded = np.where(self._r_index >= 0, self._r_index + 1, 0)
        #: 相干累加的跨段数；由模型在每次路径评估前写入。
        self.n_coherent_spans = 1
        #: 核心积分矩阵缓存：键 → 矩阵。矩阵是键的**纯函数**（幂等）。
        self._core_cache: dict[tuple, NDArrayFloat] = {}
        #: 缓存占用的字节数（用于按预算淘汰）。
        self._core_cache_bytes = 0

    def _span_output_power(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> NDArrayFloat:
        """该跨段的出纤功率 (W)。

        出纤功率**不进缓存**：它是入纤功率的逐点函数，每次重算（代价是
        :math:`O(N^2)`，相对核心矩阵的构造可以忽略）。把它和核心矩阵放在
        同一个缓存条目里会让"缓存条目"依赖"哪个功率先到"，那正是调用历史
        依赖的根源。
        """
        length = float(span_length_m)
        if self.include_isrs:
            total_w = float(power.sum())
            r_f = _calc_raman_profile_jit(
                self.f_rel_hz,
                self.f_m_hz,
                self.f_M_hz,
                DELTA_F_CO_HZ,
                total_w,
            )
            return _calc_span_out_power_jit(
                np.ascontiguousarray(power, dtype=np.float64),
                self.f_rel_hz,
                np.ascontiguousarray(self.alpha_npm, dtype=np.float64),
                length,
                np.ascontiguousarray(r_f, dtype=np.float64),
                float(self.k_bar),
            )
        return power * np.exp(-self.alpha_npm * length)

    def _core_matrix(
        self,
        canonical_power: NDArrayFloat,
        span_length_m: float,
        n_coherent_spans: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat | None]:
        """构造该跨段的核心积分矩阵（单跨 + 链路级相干总量）。

        Args:
            canonical_power: 规范化后的入纤功率 (W)；矩阵只依赖它，故矩阵是
                缓存键的纯函数。
            span_length_m: 跨段长度 (m)。
            n_coherent_spans: 相干累加的跨段数。

        Returns
        -------
            ``(plain, dirichlet)``：单跨核心积分与链路级相干总量核心积分
            （非相干档后者为 ``None``）。
        """
        if self.include_isrs:
            z_mid, gain, rate = _isrs_span_profile(
                self.f_rel_hz,
                self.alpha_npm,
                canonical_power,
                self.f_m_hz,
                self.f_M_hz,
                self.k_bar,
                float(span_length_m),
                self.isrs_segments,
            )
            return _island_core(
                f_abs_hz=self.f_abs_hz,
                b_hz=self.b_hz,
                alpha_npm=self.alpha_npm,
                gamma_w_m=self.gamma_w_m,
                beta2_s2_m=self.beta2_s2_m,
                beta3_s3_m=self.beta3_s3_m,
                dispersion_convention=self.dispersion_convention,
                nsp_convention=self.nsp_convention,
                length_m=float(span_length_m),
                r_index=self._r_index,
                correlation_points=self.correlation_points,
                island_cells=self.island_cells,
                n_coherent_spans=int(n_coherent_spans),
                isrs_gain=gain,
                isrs_rate=rate,
                isrs_z_mid=z_mid,
                isrs_seg_length_m=float(span_length_m) / float(self.isrs_segments),
            )
        return _island_core(
            f_abs_hz=self.f_abs_hz,
            b_hz=self.b_hz,
            alpha_npm=self.alpha_npm,
            gamma_w_m=self.gamma_w_m,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            dispersion_convention=self.dispersion_convention,
            nsp_convention=self.nsp_convention,
            length_m=float(span_length_m),
            r_index=self._r_index,
            correlation_points=self.correlation_points,
            island_cells=self.island_cells,
            n_coherent_spans=int(n_coherent_spans),
        )

    def _link_ramps(
        self,
        plain: NDArrayFloat,
        dirichlet: NDArrayFloat,
        canonical_power: NDArrayFloat,
        n_spans: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        r"""由 Dirichlet 链路总量反解逐跨相位斜坡。

        逐跨的 NLI 振幅幅度必须等于**单跨**噪声（契约要求 ``|a|**2`` 等于
        ``*_noise_w``），相干因子
        :math:`[\sin(N\psi/2)/\sin(\psi/2)]^2` 只能通过相位斜坡表达。因此
        斜坡按"链路级相干总量 ÷ 单跨总量"反解：路由层算出的
        :math:`|\sum_k a_k|^2` 恰好等于 Dirichlet 核算出的链路总量。

        Args:
            plain: ``(N, N, J)`` 单跨核心积分矩阵。
            dirichlet: ``(N, N, J)`` 链路级相干总量核心积分矩阵。
            canonical_power: 规范化后的入纤功率 (W)。
            n_spans: 跨段数。

        Returns
        -------
            ``(ramp_spm, ramp_xpm)``：逐跨相位增量 (rad)，各 ``(N,)``。
        """
        g_psd = np.where(canonical_power > 0.0, canonical_power / self.b_hz, 0.0)
        g_pad = np.concatenate([[0.0], g_psd])
        third_psd = g_pad[self._r_index_padded]
        single_spm, single_xpm = _nli_from_core(plain, g_psd, third_psd, self.b_hz)
        link_spm, link_xpm = _nli_from_core(dirichlet, g_psd, third_psd, self.b_hz)
        ramp_spm = _equivalent_phase_ramp(n_spans, _coherent_ratio(link_spm, single_spm))
        ramp_xpm = _equivalent_phase_ramp(n_spans, _coherent_ratio(link_xpm, single_xpm))
        return ramp_spm, ramp_xpm

    def link_nli(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        r"""Dirichlet 核核算出的链路级 NLI (W)（论文 Eq. (62)-(64)）。

        这是逐跨相干累加必须复现的**参考值**：把单跨核心积分逐点乘上相干
        累加核 :math:`[\sin(N_s\theta/2)/\sin(\theta/2)]^2` 后按
        :func:`_nli_from_core` 聚合。相干口径下路由层算出的
        :math:`|\sum_k a_k|^2` 与它逐位一致（相对误差 < 1e-9）。

        功率口径（不含 ISRS 时与功率无关）用**规范化后的功率**构造核心矩阵，
        与 :meth:`_link_ramps` 保持同一口径，故比较的是同一件事。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            n_spans: 跨段数。

        Returns
        -------
            ``(spm, xpm)``：链路级 SCI / XCI 噪声功率 (W)，各 ``(N,)``。
        """
        canonical, _ = _canonical_power(power_in_w)
        plain, dirichlet = self._core_matrix(canonical, span_length_m, int(n_spans))
        g_psd = np.where(canonical > 0.0, canonical / self.b_hz, 0.0)
        g_pad = np.concatenate([[0.0], g_psd])
        third_psd = g_pad[self._r_index_padded]
        if dirichlet is None:
            return _nli_from_core(plain, g_psd, third_psd, self.b_hz)
        return _nli_from_core(dirichlet, g_psd, third_psd, self.b_hz)

    def _core_for(
        self,
        power: NDArrayFloat,
        span_length_m: float,
        n_coherent_spans: int,
    ) -> _CoreEntry:
        """该跨段的核心积分矩阵与相位斜坡（按长度、相干跨段数与**绝对**功率缓存）。

        缓存是**幂等**的：键里带的是规范化后的绝对功率，而矩阵与斜坡也只用
        规范化后的功率构造，所以"同一个键"永远对应"同一个条目"，与调用历史
        无关。
        """
        canonical, power_key = _canonical_power(power)
        key = (
            float(span_length_m),
            int(n_coherent_spans),
            power_key if self.include_isrs else None,
        )
        cached = self._core_cache.get(key)
        if cached is None:
            plain, dirichlet = self._core_matrix(canonical, span_length_m, n_coherent_spans)
            if dirichlet is None:
                cached = _CoreEntry(plain=plain)
            else:
                ramp_spm, ramp_xpm = self._link_ramps(
                    plain, dirichlet, canonical, int(n_coherent_spans)
                )
                cached = _CoreEntry(plain=plain, ramp_spm=ramp_spm, ramp_xpm=ramp_xpm)
            self._store_core(key, cached)
        return cached

    def _store_core(self, key: tuple, entry: _CoreEntry) -> None:
        """把核心条目放进缓存，并按内存预算淘汰旧条目。"""
        size = int(entry.plain.nbytes)
        if size > CORE_CACHE_BUDGET_BYTES:
            # 单个矩阵就超预算：不缓存，避免把内存吃光。
            return
        while self._core_cache and self._core_cache_bytes + size > CORE_CACHE_BUDGET_BYTES:
            oldest = next(iter(self._core_cache))
            self._core_cache_bytes -= int(self._core_cache[oldest].plain.nbytes)
            del self._core_cache[oldest]
        self._core_cache[key] = entry
        self._core_cache_bytes += size

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        r"""计算单跨的出纤功率与 SCI / XCI 噪声。

        单跨链（``span_context`` 为空，或链路只有一跨）走**纯功率**分支：
        返回的噪声就是路由层原来逐跨相加的单跨噪声，浮点运算顺序不变，
        因此单跨结果与改造前逐位一致。

        多跨齐次链走**相干**分支：返回的复振幅幅度是单跨噪声的平方根，
        相位是按 Dirichlet 链路总量反解出来的等效斜坡
        （见 :func:`_equivalent_phase_ramp`）。路由层按
        :math:`|\sum_k a_k|^2` 相干累加，得到的链路级 NLI 恰好等于
        Dirichlet 核算出的链路总量。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 未使用。相干累加的跨段数由
                :attr:`n_coherent_spans` 给出（模型在路径评估前写入）。
            span_context: 该跨在链路中的位置；给出且链路为齐次多跨时，
                输出携带复振幅。

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

        entry = self._core_for(power, span_length_m, self.n_coherent_spans)
        core = entry.plain
        power_out = self._span_output_power(power, span_length_m)
        active = power > 0.0
        g_psd = np.where(active, power / self.b_hz, 0.0)
        # 第三波 PSD：(N, N, J)。槽位索引已 +1，哑元固定在 0 号位。
        g_pad = np.concatenate([[0.0], g_psd])
        third_psd = g_pad[self._r_index_padded]
        spm_noise, xpm_noise = _nli_from_core(core, g_psd, third_psd, self.b_hz)
        spm_noise = np.where(active, spm_noise, 0.0)
        xpm_noise = np.where(active, xpm_noise, 0.0)

        if (
            span_context is None
            or span_context.total_spans <= 1
            or self.n_coherent_spans <= 1
            or entry.ramp_spm is None
            or entry.ramp_xpm is None
        ):
            # 单跨（或非齐次链）：纯功率口径，路由层沿用原来的求和顺序。
            return SpanFiberOutput(
                power_out_w=power_out,
                spm_noise_w=spm_noise,
                xpm_noise_w=xpm_noise,
            )

        # 相干口径：相位随已传播的链路长度线性增长。齐次链上
        # cumulative_length_m / span_length_m 就是跨序号。
        scale = span_context.cumulative_length_m / float(span_length_m)
        phase_spm = entry.ramp_spm * scale
        phase_xpm = entry.ramp_xpm * scale
        amplitude_spm = np.sqrt(spm_noise) * np.exp(1j * phase_spm)
        amplitude_xpm = np.sqrt(xpm_noise) * np.exp(1j * phase_xpm)
        return SpanFiberOutput(
            power_out_w=power_out,
            spm_noise_w=spm_noise,
            xpm_noise_w=xpm_noise,
            nli_amplitude_spm=amplitude_spm.astype(np.complex128),
            nli_amplitude_xpm=amplitude_xpm.astype(np.complex128),
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
        include_isrs: 是否把 ISRS 功率倾斜纳入 SPP / NSP（默认开）。
        dispersion_convention: ``"repo"``（默认）或 ``"paper_absolute"``；
            见 :data:`DISPERSION_CONVENTIONS`。
        isrs_segments: ISRS 剖面的 z 方向分段数。
        egn: 不支持；``True`` 直接报错。
        **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。
    """

    info = CGN_MODEL_INFO

    #: 模型需要知道自己在链路里的位置，才能给出各跨 NLI 振幅之间的相对相位。
    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        correlation_points: int = DEFAULT_CORRELATION_POINTS,
        include_isrs: bool = True,
        dispersion_convention: str = DISPERSION_CONVENTION_DEFAULT,
        nsp_convention: str = NSP_CONVENTION_DEFAULT,
        isrs_segments: int = DEFAULT_ISRS_SEGMENTS,
        egn: bool = False,
        **route_options,
    ) -> None:
        """初始化节点几何与模型专属选项。

        Raises
        ------
            ValueError: 请求了 ``egn=True``，或求积点数/口径非法。
        """
        if egn:
            raise ValueError(
                "The correlation-based GN model does not support egn=True: the cGN paper "
                "covers the GN model only. Use egn=False."
            )
        if correlation_points < 4:
            raise ValueError(f"correlation_points must be >= 4, got {correlation_points}.")
        if dispersion_convention not in DISPERSION_CONVENTIONS:
            raise ValueError(
                f"dispersion_convention must be one of {DISPERSION_CONVENTIONS}, "
                f"got {dispersion_convention!r}."
            )
        if nsp_convention not in NSP_CONVENTIONS:
            raise ValueError(
                f"nsp_convention must be one of {NSP_CONVENTIONS}, got {nsp_convention!r}."
            )
        if isrs_segments < 1:
            raise ValueError(f"isrs_segments must be >= 1, got {isrs_segments}.")
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.egn = False
        self.correlation_points = int(correlation_points)
        self.island_cells = DEFAULT_ISLAND_CELLS
        self.include_isrs = bool(include_isrs)
        self.dispersion_convention = str(dispersion_convention)
        self.nsp_convention = str(nsp_convention)
        self.isrs_segments = int(isrs_segments)
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
            include_isrs=self.include_isrs,
            dispersion_convention=self.dispersion_convention,
            nsp_convention=self.nsp_convention,
            isrs_segments=self.isrs_segments,
            band_center_hz=self.band_center_hz,
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
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        r"""单跨：调用核，再统一走放大器/ASE。

        ``span_context`` 透传给核：核据此给出该跨 NLI 复振幅的相对相位，
        路由层再按 :math:`|\sum_k a_k|^2` 相干累加。跨的序号只从
        :class:`~network_nlo_eval.models.base.SpanContext` 取，模型自己
        不维护任何跨计数器。
        """
        kernel = self.kernel_for(fiber_config)
        kernel.n_coherent_spans = self._n_coherent_spans
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


register_model(CGN_MODEL_INFO, CGNModel)
