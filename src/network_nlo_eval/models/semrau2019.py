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
2019a Eq. (10) SPM 闭式 → :func:`semrau_span_jit` 的 ``spm_*`` 分量
2019a Eq. (11) XPM 闭式 → :func:`semrau_span_jit` 的 ``mat_u``/``mat_v`` 分量
2019a 附录 A Eq. (16)/(17) 线性拉曼增益下的功率剖面 → ``semrau_span_jit`` 的
              ISRS 倾斜段
2019b Eq. (4)  φ 的定义与超额峭度 Φ → ``build_semrau_matrices`` 的 ``mat_w``
2019b Eq. (15) 单跨调制格式修正 → ``semrau_span_jit`` 的 ``egn_scale``
2019b Eq. (16) 含格式修正的总 NLI 系数 → ``xpm_span_factor`` / ``egn_scale``
2019b Table I  各调制格式的超额峭度 → :data:`EXCESS_KURTOSIS`
===========  =================================================================

**与冻结基线 ``ecz`` 的差别**

``ecz`` 用 Lasagni 2023 的**三角**拉曼近似（``r_f`` 在频带内部为 0、边缘三角
上升），本模型用 2019a 的**线性**拉曼增益近似（``P_tot·C_r·f_k`` 随频率线性，
在频带中心为零）。两者的 GN 主项在 ``C_r=0`` 时逐项相同。

**已知近似（实现层面，详见交付报告）**

* 相干因子 ``ε`` 取 0（论文 2019a 第三节明确允许的"非相干累加"简化，论文
  报告该简化对被研究系统的精度代价为 0.2 dB）。论文闭式的相干因子来自
  Poggiolini 2012 Eq. (22)，该文献不在文献包内，故不自行复现。
* 不做拉曼放大与 DRA 泵浦：光纤配置里的 ``raman_pumps`` 不参与计算
  （``k_bar = 0``）。
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


class Semrau2019SpanKernel:
    """单跨闭式 ISRS-GN 物理核。

    构造期一次性算好逐信道与逐信道对的常量（``(N,N)`` 矩阵），热路径只剩
    矩阵-向量乘与 ``O(N)`` 的量。

    Attributes
    ----------
        matrices:
            :class:`~network_nlo_eval.models.semrau2019_kernels.SemrauMatrices`，
            Eq. (10)/(11) 与 2019b Eq. (15) 的系数矩阵。
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
        coherence_epsilon: float,
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
            coherence_epsilon: 相干因子指数 ``ε``。
            srs_cr_w_m_hz: 拉曼增益斜率 ``C_r`` (1/(W·m·Hz))。
        """
        self.fiber_config = fiber_config
        self.b_hz = np.asarray(b_hz, dtype=np.float64)
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.egn = bool(egn)
        self.coherence_epsilon = float(coherence_epsilon)
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

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """计算单跨的出纤功率与 SPM/XPM 噪声。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)；0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 有效跨段数。1 表示单跨；>1 时返回按 2019b
                Eq. (16) 的口径累积 n 跨后的**总量**（SPM 带 ``n^{1+ε}``，XPM
                带 ``n + 5Φ/6`` 与渐近项），与 ``ecz`` 的 ``n_effective_spans``
                语义一致。

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
        """计算**逐跨等效**的出纤功率与噪声。

        冻结的路由层是逐跨累加噪声的，而 2019b Eq. (16) 给的是 n 跨的**总量**
        （其中调制格式修正的"单跨项"只计一次）。本方法返回 Eq. (16) 总量除以
        ``n_path_spans``，路由层逐跨相加后正好还原 Eq. (16)。

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
        """核的热路径入口：参数校验 + Numba 调用。"""
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            raise ValueError(
                f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            )
        if span_length_m <= 0.0:
            raise ValueError(f"span_length_m must be positive, got {span_length_m}.")
        n = max(1, int(n_spans))
        power_out, spm, xpm = semrau_span_jit(
            power,
            float(span_length_m),
            n,
            self.srs_cr_w_m_hz,
            self.excess_kurtosis,
            self.coherence_epsilon,
            per_span,
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
    ``coherence_epsilon``
        论文 Eq. (5) 的相干因子指数 ``ε``，默认 0（论文允许的非相干简化）。
    ``srs_cr_w_m_hz``
        拉曼增益斜率 ``C_r`` (1/(W·m·Hz))，默认与冻结基线同值；置 0 即关闭
        ISRS（用于校验无 ISRS 极限）。
    """

    info = SEMRAU2019_MODEL_INFO

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        *,
        excess_kurtosis: float = DEFAULT_EXCESS_KURTOSIS,
        coherence_epsilon: float = 0.0,
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
            coherence_epsilon: 相干因子指数 ``ε``。
            srs_cr_w_m_hz: 拉曼增益斜率 ``C_r``。
            **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。
        """
        super().__init__(grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options)
        self.egn = bool(egn)
        self.excess_kurtosis = float(excess_kurtosis)
        self.coherence_epsilon = float(coherence_epsilon)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)
        #: 当前光路的跨段数；只在 :meth:`evaluate_path` 调用期间有效。
        self._n_path_spans = 1

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
            coherence_epsilon=self.coherence_epsilon,
            srs_cr_w_m_hz=self.srs_cr_w_m_hz,
        )

    # --- 路径累积 ----------------------------------------------------------
    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> PathQoTResult:
        """沿路径评估，并让逐跨累加还原 2019b Eq. (16) 的 n 跨总量。

        冻结的路由层把噪声逐跨相加，而 2019b Eq. (16) 的调制格式修正里
        "单跨项"只应计一次。这里先数出光路的跨段数，再让单跨核返回逐跨等效
        值，累加结果即等于论文的 n 跨总量。
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


register_model(SEMRAU2019_MODEL_INFO, Semrau2019Model)
