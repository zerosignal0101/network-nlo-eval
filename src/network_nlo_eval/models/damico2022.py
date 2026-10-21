"""D'Amico 2022 可扩展且解耦的 GGN 近似（C+L+S 多波段）。

**论文**

* A. D'Amico, B. Correia, E. London, E. Virgillito, G. Borraccini, A. Napoli,
  V. Curri, *Scalable and Disaggregated GGN Approximation Applied to a C+L+S
  Optical Network*, JLT 2022, Vol. 40, No. 11, pp. 3499-3511,
  doi:10.1109/JLT.2022.3162134。

**模型要点**

论文把全梳的 NLI 拆成两个部分（Eq. 24/25）：

* ``SC``（单信道，SPM）：``P_SC,λ = γ_λ² · W_SC · I_λ^λ(L_s) · P_λ³ / Rs_λ³``；
* ``XC``（互信道，XPM）：``P_XC,λ,κ = γ_λ² · W_XC · I_κ^λ(L_s) · (P_λ/Rs_λ)(P_κ/Rs_κ)²``；

其中权重 ``W_SC = (1 + C_∞)·16/27``、``W_XC = 2·16/27``（论文 Eq. 24 后紧跟的
权重式）。``C_∞`` 是**跨间相干 SC 累积**的渐近系数：论文 Eq. (10) 的非相干
累加在真实系统里不保守，因为 SC 分量在跨与跨之间相干累积，论文 Eq. (11) 给出

``SNR_NL;λ,n > (1/N_s) · P_λ / [(1 + C_∞) P_SC;λ + P_XC;λ]``，

即 N_s 跨累积后的 NLI 总量为 ``N_s · [(1 + C_∞) P_SC + P_XC]``。FWM 贡献按论文
假设忽略。

**跨间相干累加（复振幅口径）**

冻结的路由层现在自己按 ``|Σ_k a_k|²`` 累加跨间 NLI：模型给出每一跨的复振幅
``a_k = sqrt(P_k)·e^{iφ_k}``，路由层只做实数幅度传输，**相位由模型负责**。
因此旧的"链路级总量除以跨数"的折算被删掉：逐跨核返回**单跨物理值**
``P_SC`` / ``P_XC``，跨间相干由 ``φ_k`` 表达。

论文 Eq. (11) 的 ``C_∞`` 本身是**重构形式的包络**（论文正文没有给出闭式，引用
了自己的 [38]），它只给出一个标量的总量修正，不含逐频率对的相位。本实现把它
转成一条**等效相位斜坡** ``ψ``：路由层对 N 个等幅、逐跨相位差 ``ψ`` 的振幅求和
给出

``|Σ_k e^{i k ψ}|² = [sin(N ψ/2)/sin(ψ/2)]²``  （Dirichlet 核的平方），

该因子在 ``ψ ∈ (0, 2π/N]`` 上从 ``N²`` 单调降到 0，于是可以二分反演出让
``|Σ_k a_k|²`` **精确复现** ``C_∞`` 给出的总量：

* SC 分量：目标因子 ``N·(1 + C_∞)``（由 ``C_∞`` 的定义，它恰好等于成对求和
  ``N + 2Σ(N-Δs)·κ(Δs)``）；
* XC 分量：论文 Eq. (11) 里 XC 是逐跨**非相干**累加，目标因子就是 ``N``。

**诚实标注**：总量精确，成对结构被压缩。斜坡是**带内标量**的一条直线相位，
它复现总量，但不携带逐频率对的相位结构：部分跨的前缀因子不再等于成对求和，
不同信道的可见度差异也被压成一条斜坡。压缩带来的残差在交付报告里逐项给出。

论文把 GGN 的三重频率积分（Eq. 26/27）换成可按步长求和的半解析式（Eq. 28/29），
并显式使用**逐频率**的 ``β_2(f)``（而非带中心常数），这是它与 ISRS-GN 闭式模型
的关键差别。

**公式对照**

===========================  ==========================================================
论文位置                     代码位置
===========================  ==========================================================
Eq. (1) GSNR 合成            不在本模型内：路由层按 SNR 语义给出（``models/route.py``）
Eq. (2) 频率相关损耗 α(λ)    ``_calc_fiber_attenuation_npm``（复用，仓库既有实现）
Eq. (3)/(4) D(λ) 与 β_2      ``_calc_beta2_from_d``（复用）+ **逐信道**取值（用配置的色散斜率）
Eq. (5) γ(λ)                 ``_gamma_per_channel``
Eq. (7)/(8) 拉曼增益 g_R     复用 ECZ 三角拉曼近似（``isrs_gn`` 前端，见下）
Eq. (9) 逐跨 OSNR            路由层 ASE 累加（``models/route.py``）
Eq. (10) 非相干 NLI 累加     路由层逐跨累加（复振幅口径下走 ``|Σ a_k|²``）
Eq. (11) C_∞ 相干 SC 包络    ``_coherence_envelope`` → :func:`_equivalent_phase_ramp` → 逐跨相位
Eq. (12) 单跨 ASE            路由层/放大器层（``models/route.py``）
Eq. (13) Manakov 方程        解析前提，不直接实现
Eq. (14)-(18) 扰动展开       解析前提，不直接实现
Eq. (19)/(20) 频域幅度       归一化口径（PSD 在带内高度为 1）
Eq. (21) SRS 耦合功率方程    复用 ECZ 三角拉曼前端（``_calc_raman_profile_jit``）
Eq. (24) SC 分解             ``Damico2022SpanKernel._sc_noise``
Eq. (25) XC 分解             ``Damico2022SpanKernel._xc_noise``
Eq. (26) GGN 三重积分        用闭式 GN SPM（SC）与 Eq. (29)（XC）替代，见"已知近似"
Eq. (27) Ψ_κ 与 Δβ           ``_psi_approx`` 的逐频 β_2 用法
Eq. (28) 单步闭式解          ``_psi_approx``（分段 ρ² 指数化）
Eq. (29) 分段求和闭式        ``_psi_approx``（``z_int`` 与 ``Mx`` 矩阵）
Eq. (31) 归一化 NLI 效率     诊断用途：``NliEfficiency``
跨间相位斜坡（本实现构造）   ``_equivalent_phase_ramp`` + ``_dirichlet_factor``
===========================  ==========================================================

**与冻结基线 ``ecz`` 的差别**

* SRS 前端共用 ``isrs_gn`` 的**三角拉曼近似**（同一份 ``r_f`` 与出纤功率公式），
  因此两者的差异集中在 NLI 层，不掺入前端差异。论文本身用的是 Eq. (21) 的
  耦合 SRS 数值解（GNPy 的 Raman solver），这是本实现**未复现**的部分。
* NLI 层：XC 用论文 Eq. (29) 的半解析分段求和（含逐频率 β_2）；SC 用 GN 闭式
  （Poggiolini 的 closed-form GN，Eq. 120 of arXiv:1209.0394），与 GNPy 在
  ``ggn_approx`` 里的做法一致（该实现在对角线上同样回落到 GGN 的 SC 估计器）。
* ``β_2`` 用**逐频率**值，且 ``D(λ)`` 取配置的线性剖面 ``D_ref + S(λ−λ_ref)``
  （论文 Sec. III 的要求）。冻结基线 ``ecz`` / ``buglia2023`` 用的是**带中心**
  常数 ``β_2``（``isrs_gn`` 明确写了这是简化），因此本模型与基线的一部分差异
  来自这个频率相关化，而不是来自 GGN 的 NLI 结构。开关：
  ``use_dispersion_slope=False`` 可退回常数 ``D`` 做同口径对比。
* ``C_∞``：论文未给出闭式（它引用 [38]），本实现用**相位失配相干性模型**给出，
  详见 :func:`_coherence_envelope` 的文档与交付报告。

**已知近似（实现层面）**

* SC 项用 GN 闭式而非 GGN 的三重数值积分：论文自己也说"SC 项用等价于 GGN
  实现的标准估计器"（Sec. IV，Eq. 29 后一段）。
* SRS 前端用 ECZ 三角近似而非 Eq. (21) 的耦合方程数值解。
* 不做拉曼放大与 DRA 泵浦：``raman_pumps`` 不参与计算（``k_bar = 0``）。
* ``C_∞`` 的相干性模型是本实现的构造，论文没有给闭式。
* 跨间相干用**一条带内标量的等效相位斜坡**表达：总量精确（与 ``C_∞`` 给出的
  ``N·(1+C_∞)·P_SC + N·P_XC`` 逐位一致到 1e-12），但成对结构被压缩——
  斜坡的前缀因子不等于成对求和，不同信道的可见度差异也被抹平。XC 分量的
  斜坡只用来复现"逐跨非相干"的总量（目标因子 ``N``），不携带任何成对信息。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
    SpanKernel,
    register_model,
)
from network_nlo_eval.models.isrs_gn import _SRS_CR
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_fiber_attenuation_npm,
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
    "DAMICO2022_MODEL_INFO",
    "DEFAULT_COHERENCE_POINTS",
    "Damico2022Model",
    "Damico2022SpanKernel",
    "NOMINAL_BAND_EDGES_HZ",
    "SPM_WEIGHT",
    "XPM_WEIGHT",
]

#: 论文 Eq. (24) 后的权重式：``W_SC = (1 + C_∞)·(8/9)²·6/2³ = (1 + C_∞)·16/27``。
#: 这里存的是**不含** ``C_∞`` 的部分；``C_∞`` 在 :func:`_coherence_envelope` 里算。
SPM_WEIGHT = 16.0 / 27.0

#: 论文 Eq. (24) 后的权重式：``W_XC = 2·16/27``。
XPM_WEIGHT = 2.0 * 16.0 / 27.0

#: 石英光纤受激拉曼散射增益谱的截断频率 (Hz)，与冻结基线 ``ecz`` 同值。
DELTA_F_CO_HZ = 15e12

#: 名义波段边界 (Hz)，与仓库基准算例 ``benchmarks/cases.py`` 的 L/C/S/E 边界一致。
#: 用于"按带解耦"时把 XC 求和分组；每组内部仍按逐对精确求和。
NOMINAL_BAND_EDGES_HZ: tuple[tuple[float, float], ...] = (
    (184.4e12, 191.6e12),  # L
    (191.6e12, 195.9e12),  # C
    (195.9e12, 205.4e12),  # S
    (205.4e12, 220.4e12),  # E
)

#: XC 聚合口径。
XC_AGGREGATIONS = ("exact", "centroid")

#: XC 聚合的默认口径。``"exact"`` = 组内逐对精确求和（分组只是组织方式，结果与
#: 分组数无关）；``"centroid"`` = 每组塌缩成一个"谱质心"代表干扰信道，是真正的
#: 解耦近似，结果随分组数变化。
XC_AGGREGATION_DEFAULT = "exact"


def _band_group_masks(
    f_abs_hz: NDArrayFloat,
    band_edges_hz: tuple[tuple[float, float], ...],
    n_bands: int | None,
) -> list[NDArrayFloat]:
    """把信道按频带分组，返回每组的布尔掩码。

    Args:
        f_abs_hz: ``(N,)`` 信道中心频率 (Hz)。
        band_edges_hz: 名义波段边界；``n_bands`` 为 ``None`` 时使用。
        n_bands: 若给定，则把整个频谱等分为 ``n_bands`` 段（忽略名义边界）。

    Returns
    -------
        每组一个 ``(N,)`` 布尔数组。空组被丢弃。
    """
    masks: list[NDArrayFloat] = []
    if n_bands is None:
        for low, high in band_edges_hz:
            mask = (f_abs_hz >= low) & (f_abs_hz < high)
            if np.any(mask):
                masks.append(mask)
        # 落在名义边界之外的信道（超出 L..E）单独成组，避免丢信道。
        covered = np.zeros_like(f_abs_hz, dtype=bool)
        for mask in masks:
            covered |= mask
        if np.any(~covered):
            masks.append(~covered)
        return masks
    if n_bands < 1:
        raise ValueError(f"n_bands must be >= 1, got {n_bands}.")
    order = np.argsort(f_abs_hz)
    chunks = np.array_split(order, n_bands)
    for chunk in chunks:
        if chunk.size == 0:
            continue
        mask = np.zeros_like(f_abs_hz, dtype=bool)
        mask[chunk] = True
        masks.append(mask)
    return masks


def _coherence_envelope(
    beta2_abs_mean: float,
    rs_hz: float,
    span_length_m: float,
    n_spans: int,
    n_points: int,
) -> float:
    """跨间相干 SC 累积的渐近系数 ``C_∞``。

    论文 Eq. (11) 用 ``C_∞`` 描述 SC 分量在跨与跨之间的相干累积，但正文没有给出
    它的闭式（它引用了自己的 [38]）。本实现用**相位失配相干性**给出可计算的
    构造，并在交付报告里明确标注这是本实现的构造而非论文公式。

    构造：SC 噪声在频率偏移 ``ν`` 处的边带，其在两个跨之间相隔 ``Δs`` 跨的副本
    之间的相对相位为

    ``Φ(ν) = 2π² · |β_2| · ν² · Δs · L_span``

    （即多走 ``Δs`` 跨所累积的二次相位，论文 Eq. (27) 的相位约定）。把带内 SC
    噪声谱取为平坦（GN 模型的带内白噪声近似），两个副本的**可见度**为

    ``κ(Δs) = |(1/Rs) ∫ cos Φ(ν) dν|``，``ν ∈ [-Rs/2, Rs/2]``。

    ``N_s`` 跨的总 SC 噪声是所有副本的两两叠加，于是相对非相干基线
    ``N_s·P_SC`` 的增强因子为 ``1 + C_∞``，其中

    ``C_∞ = (2/N_s) Σ_{Δs=1}^{N_s-1} (N_s - Δs) · κ(Δs)``

    单跨时 ``C_∞ = 0``（没有跨间相干可言），与论文 Eq. (10) 的极限一致；``κ → 1``
    时退化为完全相干（``C_∞ = N_s - 1``），``κ → 0`` 时退化为非相干（``C_∞ = 0``）。

    Args:
        beta2_abs_mean: 参与信道的平均 ``|β_2|`` (s²/m)。
        rs_hz: 信道符号率 (Hz)。
        span_length_m: 跨长 (m)。
        n_spans: 光路跨段数。
        n_points: 相干性积分的求积点数。

    Returns
    -------
        ``C_∞``（无量纲，非负）。
    """
    if n_spans <= 1:
        return 0.0
    if rs_hz <= 0.0 or span_length_m <= 0.0:
        return 0.0
    # 积分区间的中点求积（点数固定，结果与跨长/跨数无关，只与 β_2、Rs 有关）。
    step = rs_hz / n_points
    nu = (np.arange(n_points) + 0.5) * step - rs_hz / 2.0
    nu2 = nu * nu
    total = 0.0
    for delta in range(1, n_spans):
        phase = 2.0 * np.pi**2 * beta2_abs_mean * nu2 * delta * span_length_m
        kappa = float(np.abs(np.mean(np.cos(phase))))
        total += (n_spans - delta) * kappa
    return 2.0 * total / n_spans


#: ``C_∞`` 可见度积分的默认求积点数。64 点对 SMF 足够（与改造前的模型逐位
#: 一致）；强色散光纤上积分的被积函数在带边振荡很快，需要更多点才能收敛
#: （见交付报告的残差估计）。
DEFAULT_COHERENCE_POINTS = 64

#: 等效相位斜坡的二分步数。斜坡的根在 ``[0, 2π/N]`` 上，60 步已经把分辨率推到
#: 双精度之下（``(2π/N)·2^-60``）。
_RAMP_BISECTION_STEPS = 60


def _dirichlet_factor(n_spans: int, psi: float) -> float:
    """N 个等幅、逐跨相位差 ``psi`` 的振幅相干相加后的功率因子。

    路由层按 ``|Σ_k a_k|²`` 累加，N 个等幅、逐跨相位差为 ``psi`` 的振幅给出

    ``|Σ_{k=0}^{N-1} e^{i k psi}|² = [sin(N psi/2)/sin(psi/2)]²``，

    即 Dirichlet 核的平方。

    Args:
        n_spans: 跨段数 ``N``。
        psi: 逐跨相位增量 (rad)。

    Returns
    -------
        功率因子：完全相干 ``psi = 0`` 时为 ``N²``，``psi = 2π/N`` 时为 0。
    """
    n = int(n_spans)
    if n <= 1:
        return 1.0
    denominator = np.sin(0.5 * psi)
    if abs(denominator) < 1e-12:
        # psi -> 0 的极限：完全相干。
        return float(n) ** 2
    return float(np.sin(0.5 * n * psi) / denominator) ** 2


def _equivalent_phase_ramp(n_spans: int, target: float) -> float:
    """把目标跨间相干因子反演成逐跨相位增量 ``psi``。

    论文 Eq. (11) 的 ``C_∞`` 只给出一个**总量**修正（标量），不含逐频率对的
    相位。路由层的接口每跨只承载**一条**复振幅，因此跨间相干只能被压缩成
    一条等效相位斜坡：把 ``C_∞`` 换算成目标相干因子，再反演出能复现该因子的
    斜坡。

    目标因子落在 ``[N, N²]``（完全非相干到完全相干）时，斜坡存在且唯一：
    Dirichlet 因子在 ``psi ∈ (0, 2π/N]`` 上从 ``N²`` 单调降到 0，因此二分即可。

    Args:
        n_spans: 跨段数 ``N``。
        target: 想要的相干因子（相对单跨噪声功率的倍数）。

    Returns
    -------
        逐跨相位增量 (rad)；``N = 1`` 时返回 0。
    """
    n = int(n_spans)
    if n <= 1:
        return 0.0
    goal = float(np.clip(target, 0.0, float(n) ** 2))
    low = 0.0
    high = 2.0 * np.pi / n
    for _ in range(_RAMP_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        if _dirichlet_factor(n, mid) > goal:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


class Damico2022SpanKernel:
    """单跨 GGN 物理核：SC/XC 分解 + 逐频 β_2 的半解析 NLI。

    构造期一次性算好逐信道量与逐信道对的常量矩阵；热路径只剩 ``O(N)`` 的量、
    一次 ``O(N²)`` 的矩阵-向量乘和 SRS 剖面的重建。

    Attributes
    ----------
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        beta2_s2_m: ``(N,)`` **逐频率**的 ``β_2`` (s²/m)（论文的关键差别）。
        gamma_w_m: ``(N,)`` 逐信道非线性系数 (1/(W·m))，论文 Eq. (5)。
        mx: ``(N,N)`` XC 核 ``1/(4π|((β_2,i+β_2,j)/2)(f_i-f_j)|)``，对角线为 0。
    """

    def __init__(
        self,
        *,
        fiber_config: FiberSpanConfig,
        lambdas_m: NDArrayFloat,
        f_abs_hz: NDArrayFloat,
        f_rel_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        symbol_rate_hz: float,
        srs_cr_w_m_hz: float,
        n_z_steps: int,
        xc_aggregation: str,
        band_edges_hz: tuple[tuple[float, float], ...],
        n_bands: int | None,
        coherence_points: int,
        c_inf_enabled: bool,
        use_dispersion_slope: bool,
    ) -> None:
        """按光纤配置准备逐信道常量与 ``(N,N)`` 常量矩阵。

        Args:
            fiber_config: 光纤配置（衰减、色散、模场面积）。
            lambdas_m: ``(N,)`` 波长 (m)。
            f_abs_hz: ``(N,)`` 绝对频率 (Hz)。
            f_rel_hz: ``(N,)`` 相对频带中心频率 (Hz)。
            b_hz: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
            symbol_rate_hz: 符号率 (Hz)，仅用于 SRS 频带边界。
            srs_cr_w_m_hz: 拉曼增益斜率 ``C_r`` (1/(W·m·Hz))。
            n_z_steps: 论文 Eq. (28) 的分段步数。
            xc_aggregation: ``"exact"`` 或 ``"centroid"``。
            band_edges_hz: 名义波段边界 (Hz)。
            n_bands: 等分频段数；``None`` 时用名义边界。
            coherence_points: ``C_∞`` 相干性积分的求积点数。
            c_inf_enabled: 是否启用跨间相干 SC 累积（``C_∞``）。
            use_dispersion_slope: 是否用配置的色散斜率给出逐波长 ``D(λ)``
                （论文 Sec. III 的频率相关参数）。关掉则用参考波长处的常数 ``D``。
        """
        if xc_aggregation not in XC_AGGREGATIONS:
            raise ValueError(f"xc_aggregation must be one of {XC_AGGREGATIONS}, got {xc_aggregation!r}.")
        if n_z_steps < 2:
            raise ValueError(f"n_z_steps must be >= 2, got {n_z_steps}.")
        if coherence_points < 8:
            raise ValueError(f"coherence_points must be >= 8, got {coherence_points}.")

        self.fiber_config = fiber_config
        self.f_abs_hz = np.asarray(f_abs_hz, dtype=np.float64)
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.b_hz = np.asarray(b_hz, dtype=np.float64)
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)
        self.n_z_steps = int(n_z_steps)
        self.xc_aggregation = xc_aggregation
        self.coherence_points = int(coherence_points)
        self.c_inf_enabled = bool(c_inf_enabled)
        self.use_dispersion_slope = bool(use_dispersion_slope)

        lambdas = np.asarray(lambdas_m, dtype=np.float64)
        if self.fiber_config.use_attenuation_polynomial:
            alpha_npm = _calc_fiber_attenuation_npm(
                lambdas,
                self.fiber_config.reference_wavelength_nm,
                self.fiber_config.attenuation_alpha2,
                self.fiber_config.attenuation_alpha1,
                self.fiber_config.attenuation_alpha0,
            )
        else:
            alpha = self.fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            alpha_npm = np.full(lambdas.shape, alpha, dtype=np.float64)
        a_eff_m2 = _calc_aeff_dynamic(
            lambdas,
            self.fiber_config.effective_area_um2_ref,
            self.fiber_config.aeff_slope_um2_nm,
            self.fiber_config.reference_wavelength_nm,
        )

        # 逐频率的 β_2（论文 Eq. (3)/(4)/(27)/(29) 的关键点：不用带中心常数）。
        # D(λ) 取配置里的线性剖面 D(λ) = D_ref + S·(λ − λ_ref)：论文 Sec. III 明确
        # 要求用实测的 D(λ) 剖面，配置里的 (D_ref, S) 正是这条剖面的参数化。
        if use_dispersion_slope:
            d_ps_nm_km = self.fiber_config.dispersion_parameter_d + self.fiber_config.dispersion_slope_s * (
                lambdas * 1e9 - self.fiber_config.reference_wavelength_nm
            )
        else:
            d_ps_nm_km = np.full(lambdas.shape, self.fiber_config.dispersion_parameter_d, dtype=np.float64)
        beta2 = np.array(
            [_calc_beta2_from_d(float(d), lam) for d, lam in zip(d_ps_nm_km, lambdas, strict=True)],
            dtype=np.float64,
        )
        n2 = self.fiber_config.nonlinear_index_n2
        gamma = 2.0 * np.pi * n2 / (lambdas * a_eff_m2)

        self.alpha_npm = alpha_npm
        self.a_eff_m2 = a_eff_m2
        self.beta2_s2_m = beta2
        self.gamma_w_m = gamma

        # XC 核：ψ(cut,pump) = z_int[pump]·Rs_pump·Mx[cut,pump]，见 Eq. (29)。
        beta2_avg = 0.5 * (beta2[:, None] + beta2[None, :])
        df = self.f_abs_hz[:, None] - self.f_abs_hz[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx = 1.0 / (4.0 * np.pi * np.abs(beta2_avg * df))
        np.fill_diagonal(mx, 0.0)
        self.mx = mx
        self.beta2_avg = beta2_avg

        # 按带分组：XC 求和的组织方式。
        self.group_masks = _band_group_masks(self.f_abs_hz, band_edges_hz, n_bands)
        self.group_ids = np.zeros(self.f_abs_hz.shape[0], dtype=np.int64)
        for group_index, mask in enumerate(self.group_masks):
            self.group_ids[mask] = group_index

        # 跨间相干性（C_∞、两个相干因子、两条等效斜坡）的缓存。键是
        # (跨数, 跨长, 带内平均 |β_2|, 带内平均带宽)：这些都是纯函数，缓存是
        # 幂等的，不引入任何跨调用状态。
        self._coherence_cache: dict[tuple[int, float, float, float], tuple[float, float, float, float, float]] = {}

    # --- 物理子步骤 -------------------------------------------------------
    def _gamma_per_channel(self) -> NDArrayFloat:
        """论文 Eq. (5) 的逐信道非线性系数 (1/(W·m))。"""
        return self.gamma_w_m

    def _srs_profile(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """SRS 前端：拉曼倾斜系数、出纤功率与沿跨的功率剖面。

        复用 ECZ（``isrs_gn``）的三角拉曼近似：``r_f`` 与出纤功率的公式与冻结基线
        完全一致，因此本模型与 ``ecz`` 的差异不掺入前端差异。沿跨的剖面
        ``ρ²(z)`` 由同一套参数解析延拓得到，端点与出纤功率严格自洽。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            span_length_m: 跨长 (m)。

        Returns
        -------
            ``(p_out_w, r_f, rho2)``，其中 ``rho2`` 是 ``(N, M+1)`` 的功率剖面。
        """
        total_power = float(np.sum(power_in_w))
        f_m_hz = float(self.f_rel_hz[0] - self.symbol_rate_hz / 2.0)
        f_M_hz = float(self.f_rel_hz[-1] + self.symbol_rate_hz / 2.0)
        r_f = _calc_raman_profile_jit(self.f_rel_hz, f_m_hz, f_M_hz, DELTA_F_CO_HZ, total_power)
        p_out = _calc_span_out_power_jit(
            power_in_w, self.f_rel_hz, self.alpha_npm, span_length_m, r_f, 0.0
        )

        steps = self.n_z_steps
        z = np.linspace(0.0, span_length_m, steps + 1)
        # L_eff(z)：从 0 走到 z 的有效长度，ECZ 前端的 z 方向推广。
        leff_z = (1.0 - np.exp(-self.alpha_npm[:, None] * z[None, :])) / self.alpha_npm[:, None]
        leff_end = leff_z[:, -1]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio2 = np.where(power_in_w > 0.0, (p_out / power_in_w) ** 2, 0.0)
        ratio2 = np.nan_to_num(ratio2, nan=0.0, posinf=0.0, neginf=0.0)
        exponent = self.alpha_npm[:, None] * (span_length_m - z[None, :]) + self.srs_cr_w_m_hz * r_f[
            :, None
        ] * (leff_end[:, None] - leff_z)
        # 极端拉曼系数下指数可能溢出；截断到 exp(700) 保持数值有限。
        exponent = np.clip(exponent, -700.0, 700.0)
        rho2 = ratio2[:, None] * np.exp(exponent)
        # 论文 Eq. (27)/(28) 里的 ρ_κ²(z) 是**归一化**的功率剖面（z=0 处为 1），
        # 绝对功率单独通过 P_κ 进入 Eq. (24)/(25)。归一化因子只影响剖面形状，
        # 但 Eq. (29) 的 L_eff 正比于 ρ² 的落差，漏掉归一化会整体缩小 ~1e3 倍。
        head = rho2[:, :1]
        rho2 = np.where(head > 0.0, rho2 / np.where(head > 0.0, head, 1.0), rho2)
        return p_out, r_f, rho2

    def _psi_approx(self, rho2: NDArrayFloat, z: NDArrayFloat) -> NDArrayFloat:
        """论文 Eq. (28)/(29) 的分段求和量 ``z_int``。

        对每个干扰信道 ``κ``，把跨分成若干步，步内 ``ρ_κ²(z)`` 指数化：

        ``Ψ_κ(z_m) ≈ L_eff(z_m) / (1 - i Δβ/α̃_m)``（Eq. 28），

        代入 Eq. (26) 后得到 Eq. (29) 的 ``Σ_{m,n}`` 求和。GNPy 的 ``ggn_approx``
        实现（该论文的参考实现）把它整理成

        ``ψ(cut,pump) = z_int[pump] · Rs_pump / (4π|((β_2,cut+β_2,pump)/2)(f_cut-f_pump)|)``

        其中 ``z_int[pump] = (Σ_m L_eff,m)²``，``L_eff,m`` 是 Eq. (29) 里的
        ``[ρ²(z_m) - ρ²(z_{m+1})]/ᾱ*_m``（用 ``1/√|α̃_m|`` 归一，与
        ``√|α̃_m α̃_n|`` 的因子等价）。

        Args:
            rho2: ``(N, M+1)`` 沿跨的功率剖面。
            z: ``(M+1,)`` 位置数组 (m)。

        Returns
        -------
            ``(N,)`` 的 ``z_int``。
        """
        dz = float(z[1] - z[0])
        safe = np.where(rho2 > 0.0, rho2, 1.0)
        log_rho2 = np.log(safe)
        pump_alpha = (log_rho2[:, 1:] - log_rho2[:, :-1]) / dz  # 负值：功率在衰减
        step_drop = rho2[:, 1:] - rho2[:, :-1]
        # 无功率的信道：整个剖面为 0，直接给 0 权重，避免除零。
        alive = (rho2[:, 1:] > 0.0) & (rho2[:, :-1] > 0.0)
        pump_alpha = np.where(alive, pump_alpha, -1.0)
        leff = np.abs(step_drop / np.sqrt(np.abs(pump_alpha))) * np.sign(pump_alpha)
        leff = np.where(alive, leff, 0.0)
        total = np.sum(leff, axis=1)
        total = np.where(np.isfinite(total), total, 0.0)
        return np.square(total)

    def _sc_noise(
        self,
        power_in_w: NDArrayFloat,
        rho2: NDArrayFloat,
        z: NDArrayFloat,
        r_f: NDArrayFloat,
    ) -> NDArrayFloat:
        """SC（单信道，SPM）噪声功率 (W)，论文 Eq. (24)。

        论文对 ``κ = λ`` 不套用 Eq. (29)（分母 ``f_κ - f_λ`` 为零），而是"用等价于
        GGN 实现的 SC 估计器"。本实现用 GN 的闭式（arXiv:1209.0394 Eq. 120，
        即 GNPy ``_gn_analytic`` 的 ``_psi``）作为该估计器：它在
        ``f_κ = f_λ`` 处有限，且与冻结基线 ``ecz`` 的 SPM 项在数值上一致
        （两者的 16/27 与 4/9 权重差被各自 ``ψ`` 的归一化抵消）。

        有效长度取功率剖面的积分 ``∫ρ²(z)dz``（无 SRS 时退化为 GN 的
        ``L_eff = (1-e^{-αL})/α``），渐近长度取局部功率衰减的倒数
        ``1/α̃``，``α̃ = α + C_r·r_f`` 是含拉曼倾斜的 z=0 处功率衰减。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            rho2: ``(N, M+1)`` 功率剖面。
            z: ``(M+1,)`` 位置数组 (m)。
            r_f: ``(N,)`` 拉曼倾斜系数。

        Returns
        -------
            ``(N,)`` 单跨 SC 噪声功率 (W)。
        """
        dz = float(z[1] - z[0])
        # ∫ρ²(z)dz 的梯形积分。
        l_eff = np.sum(0.5 * (rho2[:, :-1] + rho2[:, 1:]) * dz, axis=1)
        alpha_tilde = self.alpha_npm + self.srs_cr_w_m_hz * r_f
        alpha_tilde = np.where(alpha_tilde > 0.0, alpha_tilde, np.inf)
        l_asym = 1.0 / alpha_tilde
        beta2_abs = np.abs(self.beta2_s2_m)
        rs = self.b_hz
        with np.errstate(divide="ignore", invalid="ignore"):
            asinh_arg = np.pi**2 * l_asym * beta2_abs * rs * rs / 2.0
            psi = np.arcsinh(asinh_arg) * l_eff * l_eff / (2.0 * np.pi * beta2_abs * l_asym)
        psi = np.where(np.isfinite(psi), psi, 0.0)
        gamma = self._gamma_per_channel()
        noise = SPM_WEIGHT * gamma * gamma * psi * power_in_w**3 / (rs * rs)
        return np.where(power_in_w > 0.0, np.maximum(noise, 0.0), 0.0)

    def _xc_noise(
        self,
        power_in_w: NDArrayFloat,
        z_int: NDArrayFloat,
    ) -> NDArrayFloat:
        """XC（互信道，XPM）噪声功率 (W)，论文 Eq. (25)+(29)。

        逐对精确口径（``"exact"``，默认）：XC 的求和按频带分组组织，组内逐对精确求和。
        分组只是**组织方式**，结果与分组数无关——这是"按带解耦"在精确口径下的
        自洽性：分带边界怎么定都不改变物理结果。

        质心口径（``"centroid"``）：每组塌缩成一个"谱质心"代表干扰信道——这是
        **真正的解耦近似**，用来量化"把一个频带当成一个干扰信道"这个假设的
        误差（交付报告里有分组数扫描）。它不是生产口径：XC 核 ``1/|Δf|`` 在组
        边缘近乎奇异，塌缩后的误差可达几十个百分点甚至数倍。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            z_int: ``(N,)`` Eq. (29) 的分段求和量。

        Returns
        -------
            ``(N,)`` 单跨 XC 噪声功率 (W)。
        """
        gamma = self._gamma_per_channel()
        rs = self.b_hz
        # a_j = P_j²/Rs_j，w_j = z_int_j · a_j。
        with np.errstate(divide="ignore", invalid="ignore"):
            a = np.where(rs > 0.0, power_in_w * power_in_w / rs, 0.0)
        w = z_int * a

        if self.xc_aggregation == "exact":
            weighted = self.mx @ w
        else:
            # 质心口径：每组塌缩成一个"谱质心"代表干扰信道。
            # 组内信道对 CUT 的贡献用质心处的核代替逐对核；CUT 自己所在的那一组
            # 要把自己从质心与强度里剔除（自相互作用由 SC 项负责）。
            weighted = np.zeros_like(w)
            for mask in self.group_masks:
                sel = np.zeros_like(w)
                sel[mask] = w[mask]
                strength = float(np.sum(sel))
                if strength <= 0.0:
                    continue
                moment = float(np.sum(sel * self.f_abs_hz))
                own = mask & (power_in_w > 0.0)
                own_w = np.where(own, w, 0.0)
                own_strength = float(np.sum(own_w))
                own_moment = float(np.sum(own_w * self.f_abs_hz))
                # 逐信道的"剔除自己"修正：只有 CUT 自己所在的那一组需要。
                group_strength = np.full(self.f_abs_hz.shape, strength, dtype=np.float64)
                centroid_f = np.full(self.f_abs_hz.shape, moment / strength, dtype=np.float64)
                if own_strength > 0.0 and own_strength < strength:
                    usable = own & (own_strength < strength)
                    group_strength = np.where(usable, strength - own_strength, strength)
                    centroid_f = np.where(
                        usable, (moment - own_moment) / np.where(usable, group_strength, 1.0), centroid_f
                    )
                beta2_centroid = np.interp(centroid_f, self.f_abs_hz, self.beta2_s2_m)
                beta2_avg = 0.5 * (self.beta2_s2_m + beta2_centroid)
                df = self.f_abs_hz - centroid_f
                with np.errstate(divide="ignore", invalid="ignore"):
                    kernel = 1.0 / (4.0 * np.pi * np.abs(beta2_avg * df))
                kernel = np.where(np.isfinite(kernel), kernel, 0.0)
                weighted += kernel * group_strength

        noise = XPM_WEIGHT * gamma * gamma * power_in_w * weighted
        noise = np.where(np.isfinite(noise), noise, 0.0)
        return np.where(power_in_w > 0.0, np.maximum(noise, 0.0), 0.0)

    def _band_coherence(
        self,
        power: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> tuple[float, float, float, float, float]:
        """带内（标量）的跨间相干性。

        ``C_∞`` 是**带内标量**的包络：它用参与信道的平均 ``|β_2|`` 与平均带宽
        积分出可见度 ``κ(Δs)``，再给出成对相干因子

        ``N·(1 + C_∞) = N + 2 Σ_{Δs=1}^{N-1} (N-Δs)·κ(Δs)``。

        本方法把它转成两条等效相位斜坡（SC 与 XC 各一条），供路由层的
        ``|Σ_k a_k|²`` 复现同样的总量。

        Args:
            power: ``(N,)`` 入纤功率 (W)，用来取"占用信道"的带内平均量。
            span_length_m: 跨长 (m)。
            n_spans: 跨段数。

        Returns
        -------
            ``(c_inf, factor_spm, factor_xpm, psi_spm, psi_xpm)``：

            * ``c_inf``：论文 Eq. (11) 的渐近系数（关掉开关时为 0）；
            * ``factor_spm``：SC 分量的跨间相干因子 ``N·(1 + C_∞)``；
            * ``factor_xpm``：XC 分量的跨间相干因子（论文口径的非相干累加 -> ``N``）；
            * ``psi_spm`` / ``psi_xpm``：对应的等效相位斜坡 (rad)。
        """
        n = max(1, int(n_spans))
        if n <= 1:
            return 0.0, 1.0, 1.0, 0.0, 0.0
        active = power > 0.0
        if np.any(active):
            beta2_mean = float(np.mean(np.abs(self.beta2_s2_m[active])))
            rs_mean = float(np.mean(self.b_hz[active]))
        else:
            beta2_mean = float(np.mean(np.abs(self.beta2_s2_m)))
            rs_mean = float(np.mean(self.b_hz))
        key = (n, float(span_length_m), beta2_mean, rs_mean)
        cached = self._coherence_cache.get(key)
        if cached is not None:
            return cached
        if self.c_inf_enabled:
            c_inf = _coherence_envelope(beta2_mean, rs_mean, span_length_m, n, self.coherence_points)
        else:
            c_inf = 0.0
        factor_spm = float(n) * (1.0 + c_inf)
        # 论文 Eq. (11) 里 XC 分量是逐跨非相干累加的：目标因子就是 N。
        factor_xpm = float(n)
        psi_spm = _equivalent_phase_ramp(n, factor_spm)
        psi_xpm = _equivalent_phase_ramp(n, factor_xpm)
        cached = (float(c_inf), factor_spm, factor_xpm, psi_spm, psi_xpm)
        self._coherence_cache[key] = cached
        return cached

    def evaluate_components(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> dict[str, NDArrayFloat | float]:
        """计算单跨结果，并额外暴露 SC/XC 分解、``C_∞`` 与跨间相干因子。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨长 (m)。
            n_effective_spans: 光路跨段数（用于 ``C_∞`` 的相干累积）。

        Returns
        -------
            字典，键为 ``"power_out_w"`` / ``"sc_w"`` / ``"xc_w"`` /
            ``"c_inf"`` / ``"spm_w"`` / ``"xpm_w"`` / ``"factor_spm"`` /
            ``"factor_xpm"`` / ``"psi_spm"`` / ``"psi_xpm"`` /
            ``"link_spm_w"`` / ``"link_xpm_w"``。

            ``"spm_w"`` / ``"xpm_w"`` 是**单跨物理值**（``P_SC`` / ``P_XC``）：
            跨间相干现在由复振幅的相位表达，不再折算进逐跨的功率里。
            ``"link_spm_w"`` / ``"link_xpm_w"`` 是链路级总量
            ``N·(1+C_∞)·P_SC`` 与 ``N·P_XC``（论文 Eq. 11）。
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            raise ValueError(
                f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            )
        if span_length_m <= 0.0:
            raise ValueError(f"span_length_m must be positive, got {span_length_m}.")
        n_spans = max(1, int(n_effective_spans))

        p_out, r_f, rho2 = self._srs_profile(power, span_length_m)
        z = np.linspace(0.0, span_length_m, self.n_z_steps + 1)
        z_int = self._psi_approx(rho2, z)
        sc = self._sc_noise(power, rho2, z, r_f)
        xc = self._xc_noise(power, z_int)

        c_inf, factor_spm, factor_xpm, psi_spm, psi_xpm = self._band_coherence(
            power, span_length_m, n_spans
        )

        return {
            "power_out_w": p_out,
            "sc_w": sc,
            "xc_w": xc,
            "c_inf": float(c_inf),
            # 单跨物理值：跨间相干由复振幅的相位承载，不折算进逐跨功率。
            "spm_w": sc,
            "xpm_w": xc,
            "factor_spm": factor_spm,
            "factor_xpm": factor_xpm,
            "psi_spm": psi_spm,
            "psi_xpm": psi_xpm,
            # 论文 Eq. (11) 的链路级总量。
            "link_spm_w": factor_spm * sc,
            "link_xpm_w": factor_xpm * xc,
        }

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """计算**单跨**的出纤功率与 NLI 噪声，多跨时附带跨间相干复振幅。

        论文 Eq. (11) 给出 N 跨后的 NLI 总量

        ``N_s · [(1 + C_∞)·P_SC + P_XC]``

        旧实现把该总量除以跨数、逐跨返回等效值，让路由层做**非相干**累加。新
        口径下路由层自己按 ``|Σ_k a_k|²`` 累加，所以本方法返回的是**单跨物理
        值** ``P_SC`` / ``P_XC``，并额外给出复振幅

        ``a_k = sqrt(P_k)·exp(1j·φ_k)``，``φ_k = ψ · k``，

        其中 ``ψ`` 是由 ``C_∞`` 反演出来的等效相位斜坡（见
        :func:`_equivalent_phase_ramp`）。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨长 (m)。
            n_effective_spans: 光路跨段数。只在**没有** ``span_context`` 时用作
                链路级总量的口径（见下）；路由层总是给出 ``span_context``。
            span_context: 该跨在链路中的位置。给出且链路多于一跳时，返回值
                携带复振幅，其参考面是**本跨出纤处**。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.SpanFiberOutput`。

            * 无 ``span_context``：返回链路级总量（纯功率口径，供链路级诊断）；
            * 单跨（``total_spans <= 1``）：返回单跨物理值且**不带**振幅，
              路由层走原来的纯功率累加，结果与改造前逐位一致；
            * 多跨：返回单跨物理值 + 复振幅，路由层按 ``|Σ a_k|²`` 相干累加。
        """
        if span_context is None:
            # 没有位置信息：只能给链路级总量（纯功率口径）。
            parts = self.evaluate_components(power_in_w, span_length_m, max(1, int(n_effective_spans)))
            return SpanFiberOutput(
                power_out_w=np.asarray(parts["power_out_w"], dtype=np.float64),
                spm_noise_w=np.asarray(parts["link_spm_w"], dtype=np.float64),
                xpm_noise_w=np.asarray(parts["link_xpm_w"], dtype=np.float64),
            )

        # 跨间相干是**链路**相关的量：跨数与相位斜坡都取自 SpanContext。
        parts = self.evaluate_components(power_in_w, span_length_m, int(span_context.total_spans))
        sc = np.asarray(parts["sc_w"], dtype=np.float64)
        xc = np.asarray(parts["xc_w"], dtype=np.float64)
        power_out = np.asarray(parts["power_out_w"], dtype=np.float64)

        if int(span_context.total_spans) <= 1:
            # 单跨：没有跨间相干可言，走纯功率分支（与改造前逐位一致）。
            return SpanFiberOutput(power_out_w=power_out, spm_noise_w=sc, xpm_noise_w=xc)

        # 相位随**累计链路长度**线性增长：等长链路下
        # cumulative_length_m / span_length_m 恰好就是 span_index。
        phase_scale = span_context.cumulative_length_m / float(span_context.span_length_m)
        psi_spm = float(parts["psi_spm"])
        psi_xpm = float(parts["psi_xpm"])
        amplitude_spm = np.sqrt(sc) * np.exp(1j * (psi_spm * phase_scale))
        amplitude_xpm = np.sqrt(xc) * np.exp(1j * (psi_xpm * phase_scale))
        return SpanFiberOutput(
            power_out_w=power_out,
            spm_noise_w=sc,
            xpm_noise_w=xc,
            nli_amplitude_spm=amplitude_spm,
            nli_amplitude_xpm=amplitude_xpm,
        )

DAMICO2022_MODEL_INFO = ModelInfo(
    name="damico2022",
    summary="D'Amico 2022 disaggregated GGN approximation: SC/XC split with C_inf coherent SC envelope.",
    family="ggn",
    reference="A. D'Amico et al., 'Scalable and Disaggregated GGN Approximation Applied to a "
    "C+L+S Optical Network', JLT 2022, doi:10.1109/JLT.2022.3162134",
    options=(
        ("srs_cr_w_m_hz", "Raman gain slope C_r (1/(W*m*Hz)); 0 disables ISRS."),
        ("n_z_steps", "Number of fiber steps for the Eq. (28)/(29) segmented sum."),
        ("xc_aggregation", "'exact' | 'centroid'; per-band XC aggregation mode."),
        ("n_bands", "Equal-width band groups for the XC sum; None uses nominal band edges."),
        ("c_inf_enabled", "Enable the cross-span coherent SC envelope C_inf."),
        (
            "coherence_points",
            "Quadrature points of the C_inf visibility integral; larger values converge "
            "the envelope for strongly dispersive spans (the default 64 is enough for SMF).",
        ),
    ),
)


class Damico2022Model(RouteModel):
    """D'Amico 2022 GGN 近似模型的路由层。

    与其它模型共用 :mod:`network_nlo_eval.models.route` 的跨段累积、放大器与
    SNR 合成；本类只负责单跨物理核，并额外负责**跨间相干相位**：论文 Eq. (11)
    的 ``C_∞`` 是跨数相关的量，跨数与本跨位置由路由层通过
    :class:`~network_nlo_eval.models.base.SpanContext` 给出。

    构造期可选参数（均为关键字）：

    ``srs_cr_w_m_hz``
        拉曼增益斜率 ``C_r`` (1/(W·m·Hz))，默认与冻结基线同值；置 0 即关闭 ISRS。
    ``n_z_steps``
        论文 Eq. (28) 的分段步数，默认 64。
    ``xc_aggregation``
        ``"exact"``（组内逐对精确求和）或 ``"centroid"``（每组塌缩成谱质心）。
    ``n_bands``
        XC 分组数；``None`` 时使用名义 L/C/S/E 边界，给定整数时把频谱等分。
    ``c_inf_enabled``
        是否启用跨间相干 SC 累积（``C_∞``）。
    ``coherence_points``
        ``C_∞`` 可见度积分的求积点数，默认 64。
    ``use_dispersion_slope``
        是否用配置的色散斜率给出逐波长 ``D(λ)``（论文 Sec. III）。关掉则退回
        参考波长处的常数 ``D``，便于与带中心 ``β_2`` 的基线做同口径对比。
    """

    info = DAMICO2022_MODEL_INFO

    #: 跨间相干累加需要知道本跨在链路中的位置，才能给出各跨 NLI 振幅之间的
    #: 相对相位。路由层据此把 :class:`SpanContext` 传进来。
    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        *,
        srs_cr_w_m_hz: float = _SRS_CR,
        n_z_steps: int = 64,
        xc_aggregation: str = XC_AGGREGATION_DEFAULT,
        n_bands: int | None = None,
        c_inf_enabled: bool = True,
        use_dispersion_slope: bool = True,
        coherence_points: int = DEFAULT_COHERENCE_POINTS,
        **route_options,
    ) -> None:
        """初始化节点几何与模型专属选项。

        Args:
            grid: ``SpectrumGrid``（遗留固定栅格）或 ``FlexibleGrid``。
            ref_fiber_config: 参考光纤配置。
            symbol_rate_hz: 符号率 (Hz)。
            srs_cr_w_m_hz: 拉曼增益斜率 ``C_r``。
            n_z_steps: 分段步数。
            xc_aggregation: XC 聚合口径。
            n_bands: XC 分组数。
            c_inf_enabled: 是否启用 ``C_∞``。
            coherence_points: ``C_∞`` 可见度积分的求积点数。
            **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。
        """
        super().__init__(grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)
        self.n_z_steps = int(n_z_steps)
        self.xc_aggregation = xc_aggregation
        self.n_bands = n_bands
        self.c_inf_enabled = bool(c_inf_enabled)
        self.use_dispersion_slope = bool(use_dispersion_slope)
        if coherence_points < 8:
            raise ValueError(f"coherence_points must be >= 8, got {coherence_points}.")
        self.coherence_points = int(coherence_points)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """构造该光纤参数下的单跨核（按 ``FiberSpanConfig`` 记忆化）。"""
        return Damico2022SpanKernel(
            fiber_config=fiber_config,
            lambdas_m=self.lambdas_m,
            f_abs_hz=self.f_abs_hz,
            f_rel_hz=self.f_rel_hz,
            b_hz=self.channel_bandwidth_hz_channels,
            symbol_rate_hz=self.symbol_rate_hz,
            srs_cr_w_m_hz=self.srs_cr_w_m_hz,
            n_z_steps=self.n_z_steps,
            xc_aggregation=self.xc_aggregation,
            band_edges_hz=NOMINAL_BAND_EDGES_HZ,
            n_bands=self.n_bands,
            coherence_points=self.coherence_points,
            c_inf_enabled=self.c_inf_enabled,
            use_dispersion_slope=self.use_dispersion_slope,
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
        """单跨：调用核（单跨物理值 + 跨间相干振幅），再统一走放大器/ASE。

        论文 Eq. (11) 的 N 跨 NLI 总量 ``N_s·[(1 + C_∞)·P_SC + P_XC]`` 现在
        **不再**折算进逐跨的功率里：核返回单跨物理值 ``P_SC`` / ``P_XC`` 以及
        复振幅 ``a_k = sqrt(P_k)·e^{iφ_k}``，相位 ``φ_k`` 由 ``C_∞`` 反演得到。
        路由层按 ``|Σ_k a_k|²`` 相干累加，得到的总量与论文 Eq. (11) 一致。

        放大器/ROADM 的辅助函数在"纯功率"口径下**不**给 NLI 乘放大器增益
        （黄金向量锁定该口径），所以这里先按纯功率口径走一遍辅助函数，再把振幅
        挂到结果上：振幅的参考面是**本跨出纤处**，跨间幅度传输由路由层负责。
        """
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate(power_in_w, span_length_m, 1, span_context=span_context)
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        plain = SpanFiberOutput(
            power_out_w=fiber_output.power_out_w,
            spm_noise_w=fiber_output.spm_noise_w,
            xpm_noise_w=fiber_output.xpm_noise_w,
        )
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=plain,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )
        if fiber_output.nli_amplitude_spm is None:
            return element
        return replace(
            element,
            nli_amplitude_spm=fiber_output.nli_amplitude_spm,
            nli_amplitude_xpm=fiber_output.nli_amplitude_xpm,
        )


register_model(DAMICO2022_MODEL_INFO, Damico2022Model)
