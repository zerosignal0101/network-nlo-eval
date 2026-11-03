"""Buglia 2024 RA-GN 论文模型（``buglia2024_ra_gn``）。

Reference
---------
H. Buglia, M. Jarmolovičius, L. Galdino, R. I. Killey, P. Bayvel,
"A Closed-Form Expression for the Gaussian Noise Model in the Presence of
Raman Amplification", J. Lightwave Technol. 42(2), 636-648, 2024,
doi:10.1109/jlt.2023.3315127.

实现范围（逐条对应论文公式）
----------------------------
* **Eq. (1)**：耦合拉曼方程。复用
  :class:`~network_nlo_eval.models.hybrid_raman.RamanPaperSpanSolver`，
  支持任意数量的前向/后向泵、泵浦耗尽、泵浦-泵浦与信道-信道 ISRS、
  波长相关的衰减/有效面积/色散。
* **Eq. (2)**：五参数半解析剖面。五个系数
  ``(alpha_i, C_f,i, C_b,i, alpha_f,i, alpha_b,i)`` 全部由非线性最小二乘
  拟合到 Eq. (1) 的数值解，见
  :func:`~network_nlo_eval.models.buglia2024_closed.fit_buglia_profile_5param`。
* **Eq. (3)/(4)**：``SNR_NLI`` 定义与逐跨累加。``P_{i,j}/P_i`` 由各跨入纤
  功率承担；``n^epsilon`` 由 ``spm_coherence_exponent`` 控制。
* **Eq. (5)-(8)**：多信道 XPM 求和与 GN 链路函数；``Phi((f1+f2)/B_k)``
  按 Appendix C 的推导在闭式里丢掉。
* **Eq. (9)**：链路函数闭式（:func:`link_mu_closed`）。
* **Eq. (10)**：展开系数 ``Upsilon``。
* **Eq. (11)**：XPM 闭式（:func:`eta_xpm_closed`）。
* **Eq. (12)**：SPM 闭式（:func:`eta_spm_closed`）。

与论文印刷版的两处差异
----------------------
1. **Eq. (12) 最后一项漏了加号**：MinerU 提取为
   ``sign(-phi_i) e^{-|alpha_l L|} sign(phi_i) e^{-|alpha'_l L|}``，
   与 Appendix D 的 Eq. (56) 不符。本实现按 Eq. (56) 取加号。
2. **Eq. (9)/(11)/(12) 的 sin 项符号**：论文印刷版写
   ``+ (kappa_f kappa_b' - kappa_b kappa_f')(alpha_l - alpha_l') phi sin(phi L)``，
   但由 Eq. (31)-(33) 直接展开（``1/(-alpha_l + j phi)`` 的共轭）得到的是**负号**。
   用 Eq. (8) 的数值积分核对：按印刷版符号，单信道 XPM 偏低 10–15%；
   按推导符号，闭式与积分在 ±3% 内一致（见
   `tests/test_buglia2024_closed.py::test_sin_term_sign_matches_integral`）。
   本实现取推导符号，并在 ``results/raman/full_buglia2024.json`` 里报告该差异。

已知近似
--------
* 泵浦 XPM 不建模（论文 Sec. III-C 同样忽略，并指出可作为额外干扰信道加入）。
* 拉曼增益谱用仓库的 Gaussian/CZ 近似，不是论文 Fig. 1 的实测谱
  （原文只在图上给出，数值需从 FigShare 仓库获取）。
* ``epsilon`` 论文未印出具体取值；默认 0（与仓库其余模型一致的功率相加）。
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
from network_nlo_eval.models.dra_common import (
    DEFAULT_DRA_NLI_QUADRATURE,
    DEFAULT_DRA_STEPS,
    DRARamanModelBase,
    DRASpanKernelBase,
)
from network_nlo_eval.models.raman import C_LIGHT
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "BUGLIA2024_RA_MODEL_INFO",
    "Buglia2024RAModel",
    "Buglia2024RASpanKernel",
]


class Buglia2024RASpanKernel(DRASpanKernelBase):
    """Buglia 2024 RA-GN 的单跨核（闭式 Eq. (11)/(12)）。

    Args:
        nli_form: ``"closed"`` 用 Eq. (11)/(12) 的闭式；``"integral"`` 用
            Eq. (6)-(8) 的数值积分（只用于对照与回归，不是论文的闭式）。
        spm_coherence_exponent: Eq. (4) 的 ``epsilon``。
    """

    profile_kind = "buglia_eq2_5param"
    #: Eq. (4) 的跨累加是功率口径（``n^epsilon`` 已在核内乘好），不返回复振幅。
    emits_nli_amplitudes = False

    def __init__(
        self,
        *args: object,
        nli_form: str = "closed",
        spm_coherence_exponent: float = 0.0,
        **kwargs: object,
    ) -> None:
        if nli_form not in ("closed", "integral"):
            raise ValueError("nli_form must be 'closed' or 'integral'.")
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._nli_form = str(nli_form)
        self._spm_exponent = float(spm_coherence_exponent)
        self.gamma_per_w_m = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (self.n2_m2_w / self.a_eff_m2)

    def _fit_profile(self, profile_norm: NDArrayFloat, span_length_m: float) -> BugliaClosedProfile:
        """拟合 Eq. (2) 的 5 个系数。"""
        pumps = list(self._pumps)
        if pumps:
            powers = np.asarray([float(pump.power_w) for pump in pumps], dtype=np.float64)
            frequencies = np.asarray(
                [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in pumps],
                dtype=np.float64,
            )
            f_hat = (
                float(np.average(frequencies, weights=powers))
                if np.sum(powers) > 0.0
                else float(np.mean(self.f_abs_hz))
            )
        else:
            # 无泵时 Eq. (2) 退化为 ISRS GN 的剖面；论文取 f_hat = 0（lumped 极限），
            # 本实现取信道功率加权平均，两者只差 (f_i - f_hat) 的原点，等价。
            f_hat = float(np.mean(self.f_abs_hz))
        return fit_buglia_profile_5param(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_npm,
            f_hat,
            self._n_steps,
        )

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: BugliaClosedProfile,  # type: ignore[override]
        span_context: SpanContext | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Eq. (11)/(12) 逐跨求 SPM/XPM，按 Eq. (4) 施加 ``n^epsilon``。"""
        n_spans = 1 if span_context is None else max(1, int(span_context.total_spans))
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


BUGLIA2024_RA_MODEL_INFO = ModelInfo(
    name="buglia2024_ra_gn",
    summary=(
        "Buglia 2024 RA-GN: Eq. (2) five-coefficient profile fitted to the coupled "
        "Raman ODE, Eq. (9)/(10) link function and Eq. (11)/(12) printed SPM/XPM "
        "closed forms, Eq. (4) span accumulation. Forward and backward pumps."
    ),
    family="gn+raman",
    reference="Buglia et al., JLT 42(2):636-648, 2024, doi:10.1109/jlt.2023.3315127",
    options=(
        ("raman_pumps", "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear' (CZ)."),
        ("raman_ase", "Expose the newly generated Raman ASE before the EDFA. Default True."),
        ("raman_temperature_k", "Fibre temperature of the spontaneous Raman factor. Default 300 K."),
        ("raman_peak_gain_coefficient_m_per_w", "Peak local Raman gain rate. Default 6e-14."),
        ("include_isrs", "Include channel-channel ISRS in the coupled ODE. Default True."),
        ("fixed_point_iterations", "Maximum backward-pump fixed-point iterations. Default 24."),
        ("n_steps", "RK4 steps of the ODE and of the Eq. (2) fit. Default 32."),
        ("n_quadrature", "Gauss-Legendre nodes of the integral baseline. Default 24."),
        ("nli_form", "'closed' (Eq. 11/12, default) or 'integral' (Eq. 6-8 baseline)."),
        ("spm_coherence_exponent", "Eq. (4) epsilon; 0 (default) sums spans incoherently."),
    ),
)


class Buglia2024RAModel(DRARamanModelBase):
    """Buglia 2024 RA-GN 论文模型。

    前向与后向泵都支持。单跨 NLI 由 Eq. (11)/(12) 的闭式给出，跨间按
    Eq. (4) 累加；``spm_coherence_exponent`` 取论文的 ``epsilon``。

    Args:
        grid: 频谱栅格。
        ref_fiber_config: 参考光纤。
        symbol_rate_hz: 符号率 (Hz)。
        connections: 可选逐连接栅格。
        raman_pumps: 每跨的泵浦；``None`` 用光纤配置里的泵浦。
        raman_gain_model: ``"gaussian"`` 或 ``"linear"``。
        raman_ase: 是否在 EDFA 前计入拉曼 ASE。
        raman_temperature_k: 光纤温度 (K)。
        raman_peak_gain_coefficient_m_per_w: 峰值拉曼增益率。
        include_isrs: 是否计入信道-信道 ISRS。
        fixed_point_iterations: 后向泵不定点迭代上限。
        n_steps: ODE 的 RK4 步数。
        n_quadrature: 积分基准的节点数。
        nli_form: ``"closed"``（默认）或 ``"integral"``。
        spm_coherence_exponent: Eq. (4) 的 ``epsilon``。
        **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。
    """

    info = BUGLIA2024_RA_MODEL_INFO
    _kernel_class = Buglia2024RASpanKernel
    #: Eq. (4) 的 ``n^epsilon`` 需要知道本跨是第几跨，但结果仍是功率口径。
    supports_span_context = True

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
        fixed_point_iterations: int = 24,
        n_steps: int = DEFAULT_DRA_STEPS,
        n_quadrature: int = DEFAULT_DRA_NLI_QUADRATURE,
        nli_form: str = "closed",
        spm_coherence_exponent: float = 0.0,
        **route_options: object,
    ) -> None:
        if nli_form not in ("closed", "integral"):
            raise ValueError("nli_form must be 'closed' or 'integral'.")
        self._nli_form = str(nli_form)
        self._spm_exponent = float(spm_coherence_exponent)
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            raman_pumps=raman_pumps,
            raman_gain_model=raman_gain_model,
            raman_ase=raman_ase,
            raman_temperature_k=raman_temperature_k,
            raman_peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            include_isrs=include_isrs,
            fixed_point_iterations=fixed_point_iterations,
            n_steps=n_steps,
            n_quadrature=n_quadrature,
            **route_options,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> DRASpanKernelBase:
        """构造单跨核，透传闭式/积分口径。"""
        return self._kernel_class(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            pumps=self._resolved_pumps(fiber_config),
            gain_model=self._gain_model,
            raman_ase=self._raman_ase,
            n_steps=self._n_steps,
            n_iterations=self._fixed_point_iterations,
            include_isrs=self._include_isrs,
            n_quadrature=self._n_quadrature,
            nli_form=self._nli_form,
            spm_coherence_exponent=self._spm_exponent,
        )


register_model(BUGLIA2024_RA_MODEL_INFO, Buglia2024RAModel)
