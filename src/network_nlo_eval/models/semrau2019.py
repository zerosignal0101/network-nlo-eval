"""Semrau 2019 闭式 ISRS-GN 模型（含 2019b 调制格式修正档）。

**论文**

* Semrau, Killey, Bayvel, *A Closed-Form Approximation of the Gaussian Noise
  Model in the Presence of Inter-Channel Stimulated Raman Scattering*,
  JLT 2019, doi:10.1109/jlt.2019.2895237（下称"2019a"）。
* Semrau, Killey, Bayvel, *A Modulation Format Correction Formula for the GN
  Model in the Presence of ISRS*, JLT 2019, doi:10.1109/jlt.2019.2929461
  （下称"2019b"）。

**公式对照**

===========  =================================================================
论文位置     代码位置
===========  =================================================================
2019a Eq. (1)  SNR 口径：NLI 功率 = η_n·P_i³
2019a Eq. (3)  ISRS 功率转移 Δρ = 4.3·P_tot·C_r·L_eff·B_tot
2019a Eq. (5)  多跨累积 η_n = Σ_j (P_ij/P_i)²[η_SPM,j·n^ε + η_XPM,j]
              → 跨数因子 ``n^{1+ε}`` 已删除，改为逐跨复振幅相干累加
              （见 :func:`_self_beating_visibility` /
              :func:`_cross_beating_visibility` /
              :func:`_coherent_factor` / :func:`_equivalent_phase_ramp`）
2019a Eq. (10) SPM 闭式 → :func:`semrau_span_jit` 的 ``spm_*`` 分量
2019a Eq. (11) XPM 闭式 → :func:`semrau_span_jit` 的 ``mat_u``/``mat_v`` 分量
2019a 附录 A Eq. (16)/(17) 线性拉曼增益下的功率剖面 → ``semrau_span_jit`` 的
              ISRS 倾斜段
2019b Eq. (4) φ 的定义与超额峭度 Φ → ``build_semrau_matrices`` 的 ``mat_w``
2019b Eq. (15) 单跨调制格式修正 → ``semrau_span_jit`` 的 ``egn_scale``
2019b Eq. (16) 含格式修正的总 NLI 系数 → ``xpm_span_factor`` / ``egn_scale``
2019b Table I  各调制格式的超额峭度 → :data:`EXCESS_KURTOSIS`
===========  =================================================================

**与冻结基线 ``ecz`` 的差别**

``ecz`` 用 Lasagni 2023 的**三角**拉曼近似（``r_f`` 在频带内部为 0、边缘三角
上升），本模型用 2019a 的**线性**拉曼增益近似（``P_tot·C_r·f_k`` 随频率线性，
在频带中心为零）。两者的 GN 主项在 ``C_r=0`` 时逐项相同。

**跨间相干累积：逐跨复振幅**

2019a Eq. (5) 的跨间因子 ``n^{1+ε}`` 是在**模拟**"跨间相干随跨数增长介于 ``N``
与 ``N²`` 之间"：``ε=0`` 给出完全非相干的 ``N``，``ε=1`` 给出完全相干的
``N²``。契约层现在能真的相干累加，因此该因子被删除，改由模型自己给出**逐跨
复振幅**

.. code-block:: text

    a_k = sqrt(P_k) · exp(1j · phi_k) ,
    P_NLI(link) = |sum_k a_k|^2 ,

路由层只做实数幅度传输（``sqrt`` 功率比），相位由本模型负责。

相位的来历（与 :mod:`network_nlo_eval.models.numerical` 同一条物理）：

* 同一份 NLI 噪声在第 ``j`` 跨与第 ``k`` 跨各生成一次，两份拷贝之间的相位差是
  色散在两跨间距 ``d = |j-k|·L`` 上积累的失配。噪声不是单频，而是带内连续谱，
  因此真正相关的是**频带平均可见度**：

  - SPM（自拍频）: ``V_spm(d) = Si(z)/z``，``z = pi^2·|beta2|·d·B^2``；
  - XPM（互拍频）: ``V_xpm(d) = sinc(2·pi^2·|beta2|·d·Delta_B·B)``，
    ``Delta_B`` 是到最近邻信道的频率间隔。

* 频带平均的可见度给出跨间相干因子
  ``R(N) = N + 2·sum_{Delta=1}^{N-1} (N-Delta)·V(Delta·L)``
  （:func:`_coherent_factor`）：色散趋零时 ``V→1``，``R→N²``（完全相干）；
  色散很强时 ``V→0``，``R→N``（非相干）。

* 契约层每信道每跨只给**一个**复数，装不下 ``V(j-k)`` 的跨对结构，因此
  :func:`_equivalent_phase_ramp` 把 ``R`` 反解成**等效相位步长** ``Theta``
  （Dirichlet 核 ``[sin(N·Theta/2)/sin(Theta/2)]^2`` 在 ``(0, 2pi/N)`` 上单调
  下降，二分有唯一解），第 ``k`` 跨的相位取 ``Theta·(累计长度/跨长)``。等跨时
  相邻跨的相位差恰是 ``Theta``，路由层的 ``|sum_k a_k|^2`` 于是逐位还原
  ``R(N)``。

**论文的 ``n^{1+ε}`` 与可见度的关系**

在"所有跨对可见度相同"的模型下，相干因子是
``R = N·[1 + v·(N-1)]``（``v`` 为等效可见度），令其等于论文的 ``N^{1+ε}``
得到

.. code-block:: text

    v = (N^ε - 1) / (N-1) ,   ε = log_N(R/N) .

所以论文的跨数因子**等价于某个可见度下的相干求和**，但它是链路级的唯象量，
本身不含色散信息，无法给出"色散趋零→``N²``、色散很强→``N``"的解析行为。本
实现因此用**由色散导出的**频带平均可见度逐信道算 ``R``，把论文的 ``ε``
换成它的物理解 ``ε_eq = log_N(R/N)``（逐信道给出，报告里按频带平均）。

**已知近似（实现层面，详见交付报告）**

* 2019b Eq. (16) 的跨间结构只保留**幅度**部分：调制格式修正比 ``1+5Φ/6``
  按论文"单跨项只计一次"摊到各跨（每跨 ``1+5Φ/(6N)``），渐近项每跨一次。修正
  本身随相干因子一起放大，这是它在"跨对联合相位"口径下的退化形式。
* 不做拉曼放大与 DRA 泵浦：光纤配置里的 ``raman_pumps`` 不参与计算
  （``k_bar = 0``）。
* 跨间相位按**等跨**假设折算：跨长不等时，相位步长仍按当前跨长折算，链路级
  总量不再等于论文积分式在同一 (跨数, 跨长) 下的结果。
"""

from __future__ import annotations

import numpy as np
from scipy.special import sici

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
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
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
)
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)
from network_nlo_eval.models.semrau2019_kernels import build_semrau_matrices, semrau_span_jit
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "EXCESS_KURTOSIS",
    "SEMRAU2019_MODEL_INFO",
    "Semrau2019Model",
    "Semrau2019SpanKernel",
]

#: 拉曼增益斜率 ``C_r`` (1/(W·m·Hz))。与冻结基线 ``ecz`` 同源，避免两条模型
#: 的拉曼强度悄悄漂移。
SRS_CR_W_M_HZ = _SRS_CR

#: 2019b Table I 的超额峭度 ``Φ = E[|X|⁴]/E[|X|²]² - 2``。
EXCESS_KURTOSIS: dict[str, float] = {
    "gaussian": 0.0,
    "qpsk": -1.0,
    "16qam": -0.6800,
    "64qam": -0.6190,
    "256qam": -0.6050,
}

#: ``egn=True`` 时默认采用 uniform 64-QAM 的超额峭度（2019b 的验证配置之一）。
DEFAULT_EXCESS_KURTOSIS = EXCESS_KURTOSIS["64qam"]

#: 反解等效跨间相位步长时的二分迭代次数。
#:
#: Dirichlet 核 ``[sin(N·Theta/2)/sin(Theta/2)]^2`` 在 ``(0, 2·pi/N)`` 上从
#: ``N²`` 单调降到 0，二分有唯一解。60 次二分后区间宽度小于 ``1e-18``
#: （``N = 10`` 时），相干累加因子的相对误差在 ``1e-13`` 量级，远低于门要求
#: 的 ``1e-12``。
RAMP_BISECTION_STEPS = 60

#: 本模型在注册表里的静态描述。``family`` 标 ``"gn+egn"``：2019a 闭式 GN，
#: 可选叠加 2019b 的调制格式修正。
SEMRAU2019_MODEL_INFO = ModelInfo(
    name="semrau2019",
    summary="Semrau 2019 closed-form ISRS-GN; optional modulation-format correction.",
    family="gn+egn",
    reference="Semrau et al., JLT 2019, doi:10.1109/jlt.2019.2895237; "
    "format correction: doi:10.1109/jlt.2019.2929461",
    options=(("egn", "Apply the modulation-format correction of the 2019b paper."),),
)


# =============================================================================
# 跨间相干：可见度 → 相干因子 → 等效相位步长
# =============================================================================
def _self_beating_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
) -> NDArrayFloat:
    """自拍频（SPM）两份拷贝的频带平均可见度。

    自拍频的相位失配是 ``4·pi^2·|beta2|·d·nu·nu'``（``nu``、``nu'`` 是带内
    偏移），两份拷贝间隔 ``d`` 米时相差一个 ``a·nu·nu'``（``a = 4·pi^2·|beta2|·d``），
    仍能相干干涉的比例是带内余弦的平均：

    ``V(d) = <cos(a·nu·nu')> = Si(a·B^2/4)/(a·B^2/4)``，

    其中 ``Si`` 是正弦积分。闭式用的原因与
    :mod:`network_nlo_eval.models.numerical` 相同：被积函数越往带边振荡越快，
    固定点求积解不动。

    Args
    ----
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m)。
        bandwidth_hz: ``(N,)`` 信道带宽 (Hz)。
        distance_m: 两份拷贝的距离 (m)。

    Returns
    -------
        ``(N,)`` 可见度，取值在 ``[-1, 1]``；无色散时为 1。
    """
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    argument = np.pi**2 * beta2_abs_s2_m * distance_m * bandwidth**2
    visibility = np.ones_like(bandwidth)
    nonzero = argument > 0.0
    if np.any(nonzero):
        visibility[nonzero] = sici(argument[nonzero])[0] / argument[nonzero]
    return visibility


def _cross_beating_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
    partner_offset_hz: NDArrayFloat,
) -> NDArrayFloat:
    """互拍频（XPM）两份拷贝的频带平均可见度。

    互拍频的相位失配是 ``4·pi^2·|beta2|·d·nu·(Delta+nu')``：第一束波在信道
    带内，第二束在 ``Delta`` 之外。信道间偏移 ``Delta`` 压过带内偏移，于是
    可见度退化成带内余弦的平均：

    ``V(d) = sinc(c·B/2)``，``c = 4·pi^2·|beta2|·d·Delta``，

    失配超过一个弧度就归零。

    Args
    ----
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m)。
        bandwidth_hz: ``(N,)`` 信道带宽 (Hz)。
        distance_m: 两份拷贝的距离 (m)。
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
    """频带分辨的跨间相干因子 ``R``。

    ``visibility`` 是相隔 ``1``、``2``、… 跨的两份拷贝的可见度。相干因子是
    可见度加权的跨对求和：

    ``R = N + 2·sum_{Delta=1}^{N-1} (N-Delta)·V(Delta·L)``。

    色散趋零时 ``V→1``，``R→N²``（完全相干）；色散很强时 ``V→0``，``R→N``
    （非相干）。

    Args
    ----
        n_spans: 链路跨数。
        visibility: ``(N_spans-1, N_channels)`` 可见度。

    Returns
    -------
        ``(N_channels,)`` 相干因子。
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return n + 2.0 * (weights * visibility).sum(axis=0)


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """等效跨间相位步长 (rad/跨)：把相干因子反解成相位斜坡。

    路由层按复数相加逐跨振幅，因此 ``N`` 个等幅、相位线性增长的振幅给出的
    相干因子是 Dirichlet 核 ``[sin(N·Theta/2)/sin(Theta/2)]^2``。该函数在
    ``(0, 2·pi/N)`` 上从 ``N²`` 单调降到 0，二分即得唯一解。

    Args
    ----
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


def _paper_visibility_equivalent(n_spans: int, coherence_exponent: float) -> float:
    """论文跨数因子 ``N^{1+ε}`` 对应的等效（常数）可见度。

    在"所有跨对可见度相同"的模型下相干因子是 ``R = N·[1 + v·(N-1)]``，令其
    等于 ``N^{1+ε}`` 得 ``v = (N^ε - 1)/(N-1)``。论文的 ``ε`` 因此可以读成
    "等效可见度"的另一种写法；本模型不再用它算跨数因子（改用色散导出的可见
    度），但这个映射是报告里"论文因子 ↔ 物理可见度"的换算依据。

    Args
    ----
        n_spans: 链路跨数。
        coherence_epsilon: 论文 Eq. (5) 的相干因子指数 ``ε``。

    Returns
    -------
        等效可见度（无量纲）。
    """
    n = int(n_spans)
    if n <= 1:
        return 1.0
    return (float(n) ** coherence_exponent - 1.0) / (n - 1)


class Semrau2019SpanKernel:
    """单跨闭式 ISRS-GN 物理核。

    构造期一次性算好逐信道与逐信道对的常量（``(N,N)`` 矩阵），热路径只剩
    矩阵-向量乘与 ``O(N)`` 的量。

    Attributes
    ----------
        matrices:
            :class:`~network_nlo_eval.models.semrau2019_kernels.SemrauMatrices`，
            Eq. (10)/(11) 与 2019b Eq. (15) 的系数矩阵。
        beta2_s2_m: 频带中心处的 ``beta_2`` (s^2/m)，跨间可见度只依赖它。
        partner_offset_hz: ``(N,)`` 到最近邻信道的频率偏移 (Hz)，互拍频可见度用。
    """

    def __init__(
        self,
        *,
        fiber_config: FiberSpanConfig,
        lambdas_m: NDArrayFloat,
        f_rel_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        band_center_hz: float,
        egn: bool,
        excess_kurtosis: float,
        srs_cr_w_m_hz: float,
    ) -> None:
        """按光纤配置准备逐信道常量与 ``(N,N)`` 系数矩阵。

        Args:
            fiber_config: 光纤配置（衰减、色散、模场面积）。
            lambdas_m: ``(N,)`` 波长 (m)。
            f_rel_hz: ``(N,)`` 相对频带中心频率 (Hz)。
            b_hz: ``(N,)`` 逐信道带宽 ``B_i`` (Hz)。
            band_center_hz: 频带中心频率 (Hz)，``β_2``/``β_3`` 在此波长处取值。
            egn: 是否叠加 2019b 的调制格式修正。
            excess_kurtosis: 超额峭度 ``Φ``；仅 ``egn=True`` 时生效。
            srs_cr_w_m_hz: 拉曼增益斜率 ``C_r`` (1/(W·m·Hz))。
        """
        self.fiber_config = fiber_config
        self.b_hz = np.asarray(b_hz, dtype=np.float64)
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.egn = bool(egn)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)
        # 非 EGN 档把 Φ 置 0：2019b Eq. (16) 的 (n + 5Φ/6) 与渐近项都随之退化
        # 回 2019a 的 Eq. (11)，公式结构不需要分叉。
        self.excess_kurtosis = float(excess_kurtosis) if self.egn else 0.0

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

        # β_2/β_3 在频带中心波长处取值（与冻结基线 ``ecz`` 同口径）。
        ref_lambda_m = C_LIGHT / float(band_center_hz)
        beta2 = _calc_beta2_from_d(self.fiber_config.dispersion_parameter_d, ref_lambda_m)
        beta3 = _calc_beta3_from_s_d(
            self.fiber_config.dispersion_slope_s,
            self.fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

        self.alpha_npm = alpha_npm
        #: 跨间可见度只依赖二阶色散的幅度。
        self.beta2_s2_m = float(beta2)
        self.beta2_abs_s2_m = abs(float(beta2))
        #: 到最近邻信道的频率偏移 (Hz)：互拍频（XPM）可见度由它决定。
        if self.f_rel_hz.size > 1:
            offsets = np.abs(self.f_rel_hz[:, None] - self.f_rel_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            self.partner_offset_hz = offsets.min(axis=1)
        else:
            self.partner_offset_hz = np.zeros(self.f_rel_hz.size, dtype=np.float64)
        # 跨间相干量按 (跨数, 跨长) 记忆化：它们只依赖这两个数与光纤常量，
        # 与入纤功率无关，因此缓存是幂等的。
        self._coherence_cache: dict[tuple[int, float], tuple[NDArrayFloat, ...]] = {}
        # 单位相位因子按 (跨数, 跨长, 累计长度/跨长) 记忆化。
        self._factor_cache: dict[tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]] = {}
        self.matrices = build_semrau_matrices(
            f_rel_hz=self.f_rel_hz,
            lambdas_m=lambdas,
            b_hz=self.b_hz,
            alpha_npm=alpha_npm,
            a_eff_m2=a_eff_m2,
            beta2_s2_m=beta2,
            beta3_s3_m=beta3,
            n2_m2_w=self.fiber_config.nonlinear_index_n2,
            with_egn=self.egn,
        )

    # --- 跨间相干 ----------------------------------------------------------
    def _coherence(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """跨间相干量：相位步长与相干因子。

        Args
        ----
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
        visibility_spm = np.stack(
            [
                _self_beating_visibility(self.beta2_abs_s2_m, self.b_hz, distance)
                for distance in distances
            ]
        ) if n > 1 else np.zeros((0, self.b_hz.size), dtype=np.float64)
        visibility_xpm = np.stack(
            [
                _cross_beating_visibility(
                    self.beta2_abs_s2_m,
                    self.b_hz,
                    distance,
                    self.partner_offset_hz,
                )
                for distance in distances
            ]
        ) if n > 1 else np.zeros((0, self.b_hz.size), dtype=np.float64)
        factor_spm = _coherent_factor(n, visibility_spm)
        factor_xpm = _coherent_factor(n, visibility_xpm)
        ramp_spm = _equivalent_phase_ramp(n, factor_spm)
        ramp_xpm = _equivalent_phase_ramp(n, factor_xpm)
        result = (ramp_spm, ramp_xpm, factor_spm, factor_xpm)
        self._coherence_cache[key] = result
        return result

    def _span_phases(
        self,
        span_context: SpanContext,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """本跨 NLI 振幅的相位 (rad)。

        相位随链路上已走过的长度线性增长（等跨时即随跨号线性增长）：在第
        ``k`` 跨生成的噪声比第 0 跨生成的噪声多走了 ``k·L``，两者之间的相位
        差就是色散在这段距离上积累的失配。

        Args
        ----
            span_context: 本跨在链路里的位置。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``(phase_spm, phase_xpm)``，均为 ``(N,)`` (rad)。
        """
        ramp_spm, ramp_xpm, _, _ = self._coherence(span_context.total_spans, span_length_m)
        scale = span_context.cumulative_length_m / float(span_length_m)
        return ramp_spm * scale, ramp_xpm * scale

    def _span_phase_factors(
        self,
        span_context: SpanContext,
        span_length_m: float,
    ) -> tuple[NDArrayComplex, NDArrayComplex]:
        """本跨 NLI 振幅的单位相位因子 ``exp(1j·phi)``。

        与 :meth:`_span_phases` 同义，只是把相位折成复数因子并按
        (跨数, 跨长, 累计长度/跨长) 记忆化：热路径上每跨省掉两次复指数
        （152 信道 × 10 跨时是可观的一块）。

        Args
        ----
            span_context: 本跨在链路里的位置。
            span_length_m: 跨段长度 (m)。

        Returns
        -------
            ``(factor_spm, factor_xpm)``，均为 ``(N,)`` 复数，模长为 1。
        """
        scale = span_context.cumulative_length_m / float(span_length_m)
        key = (int(span_context.total_spans), float(span_length_m), float(scale))
        cached = self._factor_cache.get(key)
        if cached is None:
            phase_spm, phase_xpm = self._span_phases(span_context, span_length_m)
            cached = (np.exp(1j * phase_spm), np.exp(1j * phase_xpm))
            self._factor_cache[key] = cached
        return cached

    # --- 热路径 -------------------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """计算单跨的出纤功率与 NLI 噪声（多跨时附带跨间相干振幅）。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 链路跨数。只在没给 ``span_context`` 时用作链路
                跨数（此时按"第 0 跨"处理，振幅相位为 0）。
            span_context: 本跨在链路里的位置。给出且链路跨数大于 1 时，返回值
                带上**逐跨复振幅**，路由层据此做相干累加。

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
        n_spans = max(
            1,
            int(span_context.total_spans) if span_context is not None else int(n_effective_spans),
        )
        # 逐跨等效口径（``per_span=True``）：GN 主项是本跨自己的噪声，2019b 的
        # 跨间结构（"单跨项只计一次"的 ``1+5Φ/(6N)`` 与每跨一次的渐近项）留在
        # 核内。相干因子不再进核——跨间相干由复振幅负责。
        power_out, spm, xpm = semrau_span_jit(
            power,
            float(span_length_m),
            n_spans,
            self.srs_cr_w_m_hz,
            self.excess_kurtosis,
            0.0,
            True,
            self.alpha_npm,
            self.f_rel_hz,
            self.b_hz,
            self.matrices.spm_u,
            self.matrices.spm_v,
            self.matrices.spm_coef,
            self.matrices.mat_u,
            self.matrices.mat_v,
            self.matrices.mat_w,
        )
        if span_context is None or n_spans <= 1:
            # 单跨：相干退化为单跨，振幅相位取 0，路由层保持纯功率口径。
            return SpanFiberOutput(power_out, spm, xpm)
        factor_spm, factor_xpm = self._span_phase_factors(span_context, float(span_length_m))
        amplitude_spm = np.sqrt(spm) * factor_spm
        amplitude_xpm = np.sqrt(xpm) * factor_xpm
        return SpanFiberOutput(power_out, spm, xpm, amplitude_spm, amplitude_xpm)

    def link_level_total(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> SpanFiberOutput:
        """链路级 NLI 总量：逐跨值 × 跨间相干因子。

        等跨链路上的相干累加结果就是"逐跨值 × ``R(N)``"；本方法给出的是这个
        解析值（不经路由层的逐跨递推），用于与路由层结果对账。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            n_spans: 链路跨数。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.SpanFiberOutput`，其中噪声是
            链路级总量。
        """
        per_span = self.evaluate(power_in_w, span_length_m, n_spans)
        _, _, factor_spm, factor_xpm = self._coherence(n_spans, float(span_length_m))
        return SpanFiberOutput(
            per_span.power_out_w,
            per_span.spm_noise_w * factor_spm,
            per_span.xpm_noise_w * factor_xpm,
        )

    def paper_eq16_total(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> SpanFiberOutput:
        """论文 2019b Eq. (16) 的 ``N`` 跨总量（非相干口径）。

        这是**改造前**模型在链路级给出的量：跨数因子 ``n^{1+ε}``（此处 ``ε=0``）
        与 ``N + 5Φ/6``、``ñ`` 渐近项都在里面。新的相干口径给出的是
        ``(R(N)/N)`` 倍的它（见交付报告），本方法用于这一对照。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            n_spans: 链路跨数。

        Returns
        -------
            :class:`~network_nlo_eval.models.base.SpanFiberOutput`，其中噪声是
            论文 Eq. (16) 的链路级总量。
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            raise ValueError(
                f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            )
        if span_length_m <= 0.0:
            raise ValueError(f"span_length_m must be positive, got {span_length_m}.")
        power_out, spm, xpm = semrau_span_jit(
            power,
            float(span_length_m),
            max(1, int(n_spans)),
            self.srs_cr_w_m_hz,
            self.excess_kurtosis,
            0.0,
            False,
            self.alpha_npm,
            self.f_rel_hz,
            self.b_hz,
            self.matrices.spm_u,
            self.matrices.spm_v,
            self.matrices.spm_coef,
            self.matrices.mat_u,
            self.matrices.mat_v,
            self.matrices.mat_w,
        )
        return SpanFiberOutput(power_out, spm, xpm)


class Semrau2019Model(RouteModel):
    """Semrau 2019 闭式 ISRS-GN 的路由层模型。

    与其它模型共用 :mod:`network_nlo_eval.models.route` 的跨段累积、放大器与
    SNR 合成；本类只负责单跨物理核。

    构造期可选参数（均为关键字）：

    ``excess_kurtosis``
        2019b Table I 的超额峭度 ``Φ``，默认 uniform 64-QAM 的 ``-0.619``。
        仅在 ``egn=True`` 时生效。
    ``srs_cr_w_m_hz``
        拉曼增益斜率 ``C_r`` (1/(W·m·Hz))，默认与冻结基线同值；置 0 即关闭
        ISRS（用于校验无 ISRS 极限）。
    """

    info = SEMRAU2019_MODEL_INFO

    #: 本模型给出跨间相干用的复振幅，路由层据此做复数累加。
    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        *,
        excess_kurtosis: float = DEFAULT_EXCESS_KURTOSIS,
        srs_cr_w_m_hz: float = SRS_CR_W_M_HZ,
        **route_options,
    ) -> None:
        """初始化节点几何与模型专属选项。

        Args:
            grid: ``SpectrumGrid``（遗留固定栅格）或 ``FlexibleGrid``。
            ref_fiber_config: 参考光纤配置。
            symbol_rate_hz: 符号率 (Hz)。
            egn: 是否叠加 2019b 的调制格式修正。
            excess_kurtosis: 2019b Table I 的超额峭度 ``Φ``。
            srs_cr_w_m_hz: 拉曼增益斜率 ``C_r``。
            **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。
        """
        super().__init__(grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options)
        self.egn = bool(egn)
        self.excess_kurtosis = float(excess_kurtosis)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """构造该光纤参数下的单跨核（按 ``FiberSpanConfig`` 记忆化）。"""
        return Semrau2019SpanKernel(
            fiber_config=fiber_config,
            lambdas_m=self.lambdas_m,
            f_rel_hz=self.f_rel_hz,
            b_hz=self.channel_bandwidth_hz_channels,
            band_center_hz=self.band_center_hz,
            egn=self.egn,
            excess_kurtosis=self.excess_kurtosis,
            srs_cr_w_m_hz=self.srs_cr_w_m_hz,
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

        Returns
        -------
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


register_model(SEMRAU2019_MODEL_INFO, Semrau2019Model)
