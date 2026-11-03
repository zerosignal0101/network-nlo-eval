"""论文 hybrid Raman 模型：Eq. (2)/(4) 耦合 ODE + Eq. (6)-(9) 闭式 NLI。

**论文**

H. Buglia, R. I. Killey, P. Bayvel, "Ultra-Wideband Modeling of Optical Fibre
Nonlinearity in Hybrid-Amplified Links", J. Lightwave Technol., 2024.
DOI:10.1109/JLT.2024.3439999.

**公式映射（逐项）**

* Eq. (2)：信号/泵浦耦合功率方程 —— :class:`RamanPaperSpanSolver`。
* Eq. (3)：LA 极限（无泵）—— 同一求解器，``pumps=[]``。
* Eq. (4)：信号 ASE 耦合方程 —— 同一求解器。
* Eq. (6)：``eta_n = sum_j (P_ij/P_i)^2 [SPM_j n^eps + XPM_j]`` ——
  :meth:`HybridRamanPaperSpanKernel._nli_span`。
* Eq. (7)：``eta_XPM = sum_k eta_XPM^(k)`` ——
  :func:`buglia2024_closed.nli_closed_form` 内的干扰信道求和。
* Eq. (8)：单干扰信道 XPM 闭式 —— :func:`buglia2024_closed.eta_xpm_closed`。
* Eq. (9)：SPM 闭式与 ``Upsilon/alpha_l/kappa`` 系数 ——
  :func:`buglia2024_closed.eta_spm_closed`。
* Eq. (10)：5 参数半解析剖面 ``rho(z, f_i)`` ——
  :func:`buglia2024_closed.fit_buglia_profile_5param`。
* Eq. (6)-(8) 数值积分基准 —— :func:`buglia2024_closed.nli_integral`。

论文 Eq. (8)/(9) 就是文献 [46]（Buglia 2024 JLT, RA-GN）的 Eq. (11)/(12)：
本文 Sec. III-D 明说 SPM/XPM 半解析式"previously derived in [46]"。因此本模块
直接复用 :mod:`network_nlo_eval.models.buglia2024_closed` 的逐项实现，不再写
第二份闭式。

**跨间口径（明确选择）**

按论文 Eq. (6) 的 ``n^epsilon``：每跨的 SPM 乘 ``n**epsilon``，XPM 不乘；
路由层逐跨把 ``[P_ij/P_i]^2`` 折算后按功率相加。``epsilon`` 在论文印刷版里
是缺字符（"is the coherent factor [28, Eq. (22)]"），因此默认 ``0``（与仓库
其余模型一致的功率相加），可显式传 ``spm_coherence_exponent``。
**不使用** PCFM/Zhang 的复振幅相干契约（``emits_nli_amplitudes = False``）。

**保留的旧路径**

``nli_form="pcfm"`` 保留升级前的近似：Eq. (2)/(4) 的 ODE 剖面 + PCFM 的
``K_SCI/K_XCI`` 闭式核（:class:`HybridRamanPcfmSpanKernel`），只用于回归与
"旧近似偏多少"的对照，不再是默认。

**仍是近似/缺口**

* 拉曼增益谱用 :class:`~network_nlo_eval.models.raman.RamanGainModel` 的
  Gaussian/CZ 近似，不是论文 Fig. 4 的实测二氧化硅谱。
* 泵浦 ASE 未单独求解（信号 ASE 耦合完整）。
* 泵浦-信号 XPM 按论文 Eq. (7) 的说明略去。
* ``epsilon`` 论文未印出，默认 0。
* 论文 Eq. (9) 印刷版最后一项漏加号，本仓按 Appendix D Eq. (56) 补齐；
  Eq. (8)/(9) 的第三项符号按 Eq. (31)-(33) 推导取负号，见
  :mod:`network_nlo_eval.models.buglia2024_closed` 的说明与数值对照。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, SpanContext, register_model
from network_nlo_eval.models.buglia2024_closed import (
    BugliaClosedProfile,
    fit_buglia_profile_5param,
    nli_closed_form,
    nli_integral,
)
from network_nlo_eval.models.dra_common import DRASpanKernelBase
from network_nlo_eval.models.numba_kernels import _calc_aeff_dynamic
from network_nlo_eval.models.pcfm import PCFMModel, PCFMSpanKernel
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_ode import (
    DEFAULT_FIXED_POINT_ITERATIONS,
    DEFAULT_PAPER_SPP_STEPS,
    PaperRamanSpanResult,
    RamanPaperSpanSolver,
)
from network_nlo_eval.models.route import apply_amplifier_and_roadm, resolve_noise_figure
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "DEFAULT_FIXED_POINT_ITERATIONS",
    "DEFAULT_PAPER_SPP_STEPS",
    "HYBRID_RAMAN_PAPER_MODEL_INFO",
    "HybridRamanPaperModel",
    "HybridRamanPaperSpanKernel",
    "HybridRamanPcfmSpanKernel",
    "NLI_FORMS",
    "PaperRamanSpanResult",
    "RamanPaperSpanSolver",
]

#: 可选的 NLI 口径。``paper`` 是论文 Eq. (6)-(9) 闭式；``integral`` 是
#: Eq. (6)-(8) 的数值积分基准；``pcfm`` 是升级前的 PCFM 核近似。
NLI_FORMS: tuple[str, ...] = ("paper", "integral", "pcfm")


class HybridRamanPaperSpanKernel(DRASpanKernelBase):
    r"""论文 Eq. (6)-(9) 的单跨核（``nli_form`` 默认 ``"paper"``）。

    单跨物理链：

    1. :class:`RamanPaperSpanSolver` 解 Eq. (2)/(4)，得到归一化剖面
       ``rho(z, f_i)``；
    2. :func:`~network_nlo_eval.models.buglia2024_closed.fit_buglia_profile_5param`
       把剖面拟合成 Eq. (10) 的 5 参数形式
       ``(alpha_i, C_f,i, C_b,i, alpha_f,i, alpha_b,i)``；
    3. Eq. (8) 求逐干扰信道 ``eta_XPM^(k)``，Eq. (7) 求和得 ``eta_XPM``；
       Eq. (9) 求 ``eta_SPM``；
    4. Eq. (6) 的 ``n^epsilon`` 与 ``P_ij/P_i`` 折算由
       :func:`~network_nlo_eval.models.buglia2024_closed.nli_closed_form` 完成。

    ``Upsilon`` 与 Buglia 2024 Eq. (10) 同名同义：本论文 Eq. (9) 把
    ``Upsilon_i = T_i (-T_f,i/T_i)^{l1} (T_b,i/T_i)^{l2}`` 与
    ``alpha_l,i = alpha_i + l1 alpha_f,i - l2 alpha_b,i``、
    ``kappa_f,i = exp(-(alpha_i + l1 alpha_f,i)L)``、
    ``kappa_b,i = exp(-l2 alpha_b,i L)`` 一起定义；其中
    ``T_f,i = -P_f C_f,i (f_i - f_hat)/alpha_f,i``，
    ``T_i = 1 + T_f,i - T_b,i exp(-alpha_b,i L)``，与
    :class:`~network_nlo_eval.models.buglia2024_closed.BugliaClosedProfile`
    的 ``c_f = C_f,i P_f``、``c_b = C_b,i P_b`` 折算一致。

    Args:
        f_abs_hz: ``(N,)`` 信道中心频率 (Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。
        fiber_config: 跨段光纤参数。
        pumps: 本跨使用的泵浦；空列表表示无拉曼（Eq. (3) 的 LA 极限）。
        gain_model: 拉曼增益谱模型。
        raman_ase: 是否把新产生的拉曼 ASE 暴露给模型层。
        n_steps: Eq. (2) 的 RK4 步数，也是 Eq. (10) 拟合的采样数。
        n_iterations: BW 泵不动点迭代上限。
        include_isrs: 是否计入信道-信道 ISRS。
        nli_form: ``"paper"``（Eq. (6)-(9) 闭式，默认）或 ``"integral"``
            （Eq. (6)-(8) 数值积分，仅作对照）。
        spm_coherence_exponent: Eq. (6) 的 ``epsilon``。
        n_quadrature: ``nli_form="integral"`` 时每个频率方向的 Gauss 节点数。

    Raises
    ------
        ValueError: ``nli_form`` 非法或 ODE/积分参数不合法。
    """

    #: 剖面类型，用于诊断。
    profile_kind = "buglia_hybrid_eq2_5param"

    #: Eq. (6) 的跨间口径是功率相加（``n^epsilon`` 已在核内乘好），
    #: 不返回复振幅。
    emits_nli_amplitudes = False

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        *,
        pumps: Sequence[RamanPumpConfig],
        gain_model: RamanGainModel,
        raman_ase: bool = True,
        n_steps: int = DEFAULT_PAPER_SPP_STEPS,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        nli_form: str = "paper",
        spm_coherence_exponent: float = 0.0,
        n_quadrature: int = 24,
    ) -> None:
        """校验口径并缓存 ``gamma`` 与泵浦平均频率 ``f_hat``。"""
        if nli_form not in ("paper", "integral"):
            raise ValueError("nli_form must be 'paper' or 'integral' for the paper kernel.")
        super().__init__(
            f_abs_hz,
            b_hz,
            fiber_config,
            pumps=pumps,
            gain_model=gain_model,
            raman_ase=raman_ase,
            n_steps=n_steps,
            n_iterations=n_iterations,
            include_isrs=include_isrs,
            n_quadrature=n_quadrature,
        )
        self._nli_form = str(nli_form)
        self._spm_exponent = float(spm_coherence_exponent)
        # Eq. (8)/(9) 的 gamma_i 是逐信道的：gamma = 2 pi f n2 / A_eff。
        self.gamma_per_w_m = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (self.n2_m2_w / self.a_eff_m2)
        self._f_hat_hz = self._pump_mean_frequency()

    def _pump_mean_frequency(self) -> float:
        """论文 Eq. (9) 的 ``f_hat``：泵浦平均频率。

        论文只说 ``f_hat`` 是"泵浦的平均频率"；这里取功率加权平均（未耗尽时
        与"每台泵等权"只在泵浦功率差异大时不同）。无泵时退化为信道平均，
        与原点的选择等价（只移动 ``f_i - f_hat`` 的零点）。
        """
        pumps = list(self._pumps)
        if not pumps:
            return float(np.mean(self.f_abs_hz))
        powers = np.asarray([float(pump.power_w) for pump in pumps], dtype=np.float64)
        frequencies = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in pumps],
            dtype=np.float64,
        )
        if float(np.sum(powers)) <= 0.0:
            return float(np.mean(self.f_abs_hz))
        return float(np.average(frequencies, weights=powers))

    def _fit_profile(self, profile_norm: NDArrayFloat, span_length_m: float) -> BugliaClosedProfile:
        """Eq. (10)：把 ODE 剖面拟合成 5 参数半解析形式。"""
        return fit_buglia_profile_5param(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_npm,
            self._f_hat_hz,
            self._n_steps,
        )

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: BugliaClosedProfile,  # type: ignore[override]
        span_context: SpanContext | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Eq. (6)-(9)：本跨 SPM/XPM 噪声功率 (W)。"""
        if self._nli_form == "integral":
            return nli_integral(
                profile,
                self.b_hz,
                power_internal,
                self.gamma_per_w_m,
                self.beta2,
                self.beta3,
                self.reference_frequency_hz,
                n_frequency=max(16, min(64, int(self._n_quadrature) * 2)),
            )
        n_spans = 1 if span_context is None else max(1, int(span_context.total_spans))
        return nli_closed_form(
            profile,
            self.b_hz,
            power_internal,
            self.gamma_per_w_m,
            self.beta2,
            self.beta3,
            self.reference_frequency_hz,
            spm_coherence_exponent=self._spm_exponent,
            n_spans=n_spans,
        )


class HybridRamanPcfmSpanKernel(PCFMSpanKernel):
    """升级前的近似核（``nli_form="pcfm"``）：ODE 剖面 + PCFM SCI/XCI 核。

    继承 PCFM 的 SCI/XCI 闭式核与多项式拟合，只覆写
    :meth:`_integrate_spp`，把核里的归一化剖面换成 Eq. (2)/(4) 的 ODE 解。
    无有效泵浦时回落到父类实现，因此与 PCFM 基线逐位一致。

    这一路径**不是**论文 Eq. (6)-(9)，只保留作旧近似/回归对照。

    Args:
        f_abs_hz: ``(N,)`` 信号中心频率 (Hz)。
        b_hz: ``(N,)`` 信号带宽 (Hz)。
        fiber_config: 跨段光纤参数。
        pumps: 本跨使用的泵浦；空列表表示无拉曼。
        gain_model: 拉曼增益谱模型。
        raman_ase: 是否把新产生的拉曼 ASE 暴露给模型层。
        n_iterations: BW 泵不动点迭代上限。
        include_isrs: 是否计入信道-信道 ISRS。
        **kwargs: 透传给 :class:`PCFMSpanKernel`。
    """

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        *,
        pumps: Sequence[RamanPumpConfig],
        gain_model: RamanGainModel,
        raman_ase: bool = True,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        **kwargs: object,
    ) -> None:
        """构造核并缓存泵浦参数。"""
        kwargs.setdefault("cache_spans", False)
        super().__init__(f_abs_hz, b_hz, fiber_config, **kwargs)  # type: ignore[arg-type]
        self._pumps = list(pumps)
        self._gain_model = gain_model
        self._raman_ase_enabled = bool(raman_ase)
        self._n_iterations = int(n_iterations)
        self._include_isrs = bool(include_isrs)
        self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._solver_cache: dict[int, RamanPaperSpanSolver] = {}
        self._pump_alpha_npm = np.asarray(
            [
                float(pump.attenuation_db_per_km) * np.log(10.0) / 10.0 / 1e3
                for pump in self._pumps
            ],
            dtype=np.float64,
        )
        pump_f = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in self._pumps],
            dtype=np.float64,
        )
        self._pump_a_eff_m2 = _calc_aeff_dynamic(
            C_LIGHT / pump_f,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        self._active = any(float(pump.power_w) > 0.0 for pump in self._pumps)

    @property
    def active(self) -> bool:
        """是否至少有一个非零泵浦。"""
        return self._active

    def set_input_ase_w(self, ase_w: NDArrayFloat | None) -> None:
        """设置进入本跨光纤的信号 ASE（模型顺序）。

        Args:
            ase_w: ``(N,)`` ASE 功率 (W)；``None`` 表示零。

        Raises
        ------
            ValueError: 形状不匹配。
        """
        if ase_w is None:
            self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
            return
        values = np.asarray(ase_w, dtype=np.float64)
        if values.shape != (self.n_nodes,):
            raise ValueError("ase_w must contain one value per channel.")
        self._input_ase_internal = np.ascontiguousarray(values[self.order])

    @property
    def last_raman_ase_w(self) -> NDArrayFloat:
        """上一跨新产生的拉曼 ASE (W)，模型顺序。"""
        return np.ascontiguousarray(self._last_raman_ase_internal[self.inverse_order])

    @property
    def last_pump_power_out_w(self) -> NDArrayFloat | None:
        """上一跨的泵浦输出功率 (W)；无泵或尚未求解时返回 ``None``。"""
        return getattr(self, "_last_pump_power_out_w", None)

    def _solver_for(self, span_length_m: float) -> RamanPaperSpanSolver:
        """按跨长缓存 ODE 求解器。"""
        key = int(round(span_length_m * 1e6))
        solver = self._solver_cache.get(key)
        if solver is None:
            solver = RamanPaperSpanSolver(
                self.f_abs_hz,
                self.b_hz,
                self._pumps,
                span_length_m,
                self.alpha_npm,
                self.a_eff_m2,
                self._pump_alpha_npm,
                self._pump_a_eff_m2,
                self._gain_model,
                n_steps=self.spp_steps,
                n_iterations=self._n_iterations,
                include_isrs=self._include_isrs,
                include_ase=self._raman_ase_enabled,
            )
            self._solver_cache[key] = solver
        return solver

    def _integrate_spp(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """用耦合 ODE 的剖面替代 ISRS-only 剖面。

        Args:
            power: ``(N,)`` 跨首信号功率 (W)，内部降序。
            span_length_m: 跨长 (m)。

        Returns
        -------
            ``(profile_norm, power_out)``，含义同父类。
        """
        if not self._active:
            self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
            self._last_pump_power_out_w = None
            return super()._integrate_spp(power, span_length_m)

        solver = self._solver_for(span_length_m)
        result = solver.solve(power, self._input_ase_internal)
        if self._raman_ase_enabled:
            self._last_raman_ase_internal = np.ascontiguousarray(result.raman_ase_w)
        else:
            self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_pump_power_out_w = np.ascontiguousarray(result.pump_power_out_w)
        return result.profile_norm, result.power_out_w


HYBRID_RAMAN_PAPER_MODEL_INFO = ModelInfo(
    name="hybrid_raman_paper",
    summary=(
        "Buglia 2024 hybrid Raman: Eq. (2)/(4) coupled signal/pump/ASE ODE, Eq. (10) "
        "five-coefficient profile fit, Eq. (8)/(9) SPM/XPM closed forms and Eq. (6) "
        "n^epsilon span accumulation (default nli_form='paper'). nli_form='pcfm' keeps "
        "the legacy PCFM-kernel approximation for regression."
    ),
    family="gn+raman",
    reference=(
        "Buglia, Killey, Bayvel, 'Ultra-Wideband Modeling of Optical Fibre Nonlinearity "
        "in Hybrid-Amplified Links', JLT 2024, doi:10.1109/JLT.2024.3439999"
    ),
    options=(
        ("raman_pumps", "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear'."),
        ("raman_ase", "Expose the newly generated Raman ASE before the EDFA. Default True."),
        ("raman_temperature_k", "Fibre temperature used by the spontaneous Raman factor. Default 300 K."),
        ("raman_peak_gain_coefficient_m_per_w", "Peak local Raman gain rate before A_eff division. Default 6e-14."),
        ("include_isrs", "Include inter-channel ISRS in the coupled ODE. Default True."),
        ("fixed_point_iterations", "Maximum BW-pump fixed-point iterations. Default 24."),
        ("nli_form", "'paper' (Eq. 6-9 closed form, default), 'integral' (Eq. 6-8 baseline) or 'pcfm' (legacy)."),
        ("spm_coherence_exponent", "Eq. (6) epsilon; 0 (default) sums spans incoherently."),
        ("poly_degree", "PCFM SPP polynomial degree; only used by nli_form='pcfm'. Default 5."),
        ("egn", "EGN correction; only used by nli_form='pcfm'. Default False."),
        ("edge_fix", "PCFM zero-dispersion edge fix; only used by nli_form='pcfm'. Default True."),
        ("spp_steps", "RK4 steps of the coupled ODE and of the Eq. (10) fit. Default 32."),
    ),
)


class HybridRamanPaperModel(PCFMModel):
    """论文 hybrid Raman 模型：Eq. (2)/(4) ODE + Eq. (6)-(9) 闭式 NLI。

    Signals and pumps are solved with the coupled ODE of Eq. (2)/(4).  The
    default ``nli_form="paper"`` evaluates the paper's Eq. (6)-(9): the ODE
    profile is fitted to the five-coefficient form of Eq. (10), then Eq. (8)
    gives the per-interferer XPM and Eq. (9) the SPM; the spans accumulate as
    in Eq. (6) with ``n**epsilon``.  ``nli_form="pcfm"`` keeps the previous
    approximation (PCFM ``K_SCI``/``K_XCI`` on the ODE profile) for regression
    only, and ``nli_form="integral"`` evaluates Eq. (6)-(8) numerically.

    The model keeps its own cross-span ASE state and hands only the newly
    generated Raman ASE to the span EDFA, while the shared route layer
    transports the accumulated ASE with the signal transfer factor.

    Args:
        grid: Spectrum grid.
        ref_fiber_config: Reference fibre.
        symbol_rate_hz: Symbol rate (Hz).
        connections: Optional per-connection grid.
        raman_pumps: Pumps applied to every span.  When omitted, the pumps of
            each :class:`FiberSpanConfig` are used.
        raman_gain_model: ``"gaussian"`` or ``"linear"``.
        raman_ase: Include the Raman ASE before the EDFA.
        raman_temperature_k: Fibre temperature (K).
        raman_peak_gain_coefficient_m_per_w: Peak local Raman gain rate.
        include_isrs: Include inter-channel ISRS in the ODE.
        fixed_point_iterations: Maximum BW-pump fixed-point iterations.
        nli_form: ``"paper"`` (default), ``"integral"`` or ``"pcfm"``.
        spm_coherence_exponent: Eq. (6) ``epsilon``.
        poly_degree: PCFM polynomial degree (legacy path only).
        egn: EGN correction (legacy path only).
        edge_fix: PCFM edge fix (legacy path only).
        spp_steps: RK4 steps of the ODE and of the Eq. (10) fit.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.

    Raises
    ------
        ValueError: Invalid NLI form, ODE or PCFM options.
    """

    info = HYBRID_RAMAN_PAPER_MODEL_INFO

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections=None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str = "gaussian",
        raman_ase: bool = True,
        raman_temperature_k: float = 300.0,
        raman_peak_gain_coefficient_m_per_w: float = 6e-14,
        include_isrs: bool = True,
        fixed_point_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        nli_form: str = "paper",
        spm_coherence_exponent: float = 0.0,
        poly_degree: int = 5,
        egn: bool = False,
        edge_fix: bool = True,
        spp_steps: int = DEFAULT_PAPER_SPP_STEPS,
        **route_options: object,
    ) -> None:
        """Build the paper model."""
        if fixed_point_iterations < 1:
            raise ValueError("fixed_point_iterations must be >= 1.")
        if nli_form not in NLI_FORMS:
            raise ValueError(f"nli_form must be one of {NLI_FORMS}, got {nli_form!r}.")
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            poly_degree=poly_degree,
            egn=egn,
            edge_fix=edge_fix,
            cache_spans=False,
            spp_steps=spp_steps,
            raman_model=None,
            raman_pumps=None,
            raman_gain_model=raman_gain_model,
            raman_ase=raman_ase,
            raman_temperature_k=raman_temperature_k,
            raman_peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            **route_options,
        )
        self._paper_pumps = None if raman_pumps is None else list(raman_pumps)
        self._include_isrs = bool(include_isrs)
        self._fixed_point_iterations = int(fixed_point_iterations)
        self._nli_form = str(nli_form)
        self._spm_exponent = float(spm_coherence_exponent)
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)

    def _build_kernel(self, fiber_config: FiberSpanConfig):
        """Build the span kernel of this fibre for the selected NLI form."""
        pumps = self._paper_pumps if self._paper_pumps is not None else fiber_config.raman_pumps
        if self._nli_form == "pcfm":
            return HybridRamanPcfmSpanKernel(
                self.f_abs_hz,
                self.channel_bandwidth_hz_channels,
                fiber_config,
                pumps=pumps,
                gain_model=self._raman_gain_model,
                raman_ase=self._raman_ase,
                n_iterations=self._fixed_point_iterations,
                include_isrs=self._include_isrs,
                poly_degree=self.poly_degree,
                symbol_rate_hz=self.symbol_rate_hz,
                egn=self.egn,
                edge_fix=self.edge_fix,
                cache_spans=False,
                spp_steps=self._active_spp_steps,
            )
        return HybridRamanPaperSpanKernel(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            pumps=pumps,
            gain_model=self._raman_gain_model,
            raman_ase=self._raman_ase,
            n_steps=self._active_spp_steps,
            n_iterations=self._fixed_point_iterations,
            include_isrs=self._include_isrs,
            nli_form=self._nli_form,
            spm_coherence_exponent=self._spm_exponent,
        )

    def evaluate_path(self, optical_path, power_in_w):
        """Evaluate a path, resetting the cross-span ASE and kernel state."""
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)
        for kernel in self._kernel_cache.values():
            reset = getattr(kernel, "reset_state", None)
            if reset is not None:
                reset()
        return super().evaluate_path(optical_path, power_in_w)

    def span_step(
        self,
        fiber_config,
        edfa_config,
        power_in_w,
        span_length_m,
        span_context: SpanContext | None = None,
    ):
        """Single span: ODE kernel, then the shared amplifier handling.

        Args:
            fiber_config: Fibre parameters of the span.
            edfa_config: Amplifier configuration of the span.
            power_in_w: ``(N,)`` launch power per channel (W).
            span_length_m: Span length (m).
            span_context: Position of the span in the link.

        Returns
        -------
            The :class:`~network_nlo_eval.models.base.ElementOutput` of the span.
        """
        kernel = self.kernel_for(fiber_config)
        kernel.set_input_ase_w(self._ase_in_w)
        if self._nli_form == "pcfm":
            fiber_output = kernel.evaluate(
                power_in_w,
                span_length_m,
                1,
                span_index=(self._span_index if span_context is None else int(span_context.span_index)),
                beta2_accum_ps2=self._beta2_accum_ps2,
                span_context=span_context,
            )
        else:
            fiber_output = kernel.evaluate(
                power_in_w,
                span_length_m,
                1,
                span_context=span_context,
            )
        raman_ase_w = kernel.last_raman_ase_w if self._raman_ase else None
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            input_ase_w=raman_ase_w,
        )
        self._ase_in_w = np.array(element.ase_noise_w, dtype=np.float64, copy=True)
        self._span_index += 1
        if self._nli_form == "pcfm":
            self._beta2_accum_ps2 = self._beta2_accum_ps2 + kernel.effective_dispersion_ps2(span_length_m)
        return element


register_model(HYBRID_RAMAN_PAPER_MODEL_INFO, HybridRamanPaperModel)
