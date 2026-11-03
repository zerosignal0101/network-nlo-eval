"""Zhang 2024 backward-DRA coherent-EGN paper control (``zhang2024_dra_egn``).

Reference
---------
Y. Zhang, Q. Guo, L. Gan, L. Yi, Q. Qiu, W. Hu, Z. Wang, Q. Zhuge,
"Accurate Closed-Form Coherent EGN Model Supporting Distributed Raman
Amplification", OECC 2024, doi:10.1109/OECC54135.2024.10975528.

What is implemented
-------------------
* **Eq. (1)**: the exponential normalised profile
  ``rho(z) = (1+T) exp(-alpha z) - T exp(-A z) + D exp(-d (L-z))``.  ``alpha``
  is the fibre attenuation; ``T, A, D, d`` are fitted per channel to the
  coupled Raman ODE, which is the calibration the paper performs with its
  numerical Raman solver (Sec. II-A).  Backward pumps only, as in the paper.
* **Eq. (2)**: the closed-form longitudinal integral
  ``int rho exp(1j phi z) dz``.  It is used as the per-span link function.
* **Eq. (3)-(4)**: the per-span NLI is computed from the Eq. (2) link
  function, and **the spans are summed coherently**: the kernel returns a
  complex NLI amplitude per channel and span, and the shared route layer
  performs ``|sum_m a_m|**2``.  This is the incremental layer the paper claims
  over DRA closed forms that ignore inter-span coherence.
* **Eq. (5)**: the same-span ``P_Fm P_Fm + P_Bm P_Bm + P_Fm P_Bm`` and
  cross-span ``P_Fm P_Fn + ...`` structure is what the complex sum realises
  when the fields are added.  ``P_Fm P_Fn``/``P_Bm P_Bn`` are **not**
  separately dropped: the coherent sum keeps all pair products.  That is a
  deliberate, documented difference from the paper's truncation (the paper
  drops them because they are small).

What is not reproduced
----------------------
* The paper does **not** publish the closed-form ``P`` terms of Eq. (5)
  (explicitly omitted in Sec. II-B), nor the calibration coefficients
  ``a_i, b_i, c_i, d_i`` of Sec. II-C.  This module therefore implements the
  GN part (SPM + XPM) of Eq. (3); the ``eta_XPM_corr`` modulation-format
  correction and the dataset calibration are **not** reproduced.  The model
  is a coherent-GN subset, not the paper's calibrated EGN.
* The inter-span phase ``phi_q`` is not published either.  The complex
  amplitude uses the repository's channel SPM phase-mismatch ramp
  ``4 pi^2 |beta2_eff| B_i^2 / 12`` per metre.  The coherent-sum structure is
  the paper's; the numeric phase is a labelled proxy.
* Only backward pumps are accepted.  A forward pump raises ``ValueError``.
* The signal power profile is the Eq. (1) fit, not a waveform simulation;
  no waveform model is called online.
* The Raman gain spectrum remains the repository Gaussian/CZ approximation.

The model summary and this docstring must be read before quoting any number
as a reproduction of Zhang 2024.
"""

from __future__ import annotations

from collections.abc import Sequence

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, register_model
from network_nlo_eval.models.dra_common import (
    DEFAULT_DRA_NLI_QUADRATURE,
    DEFAULT_DRA_STEPS,
    DRARamanModelBase,
    DRASpanKernelBase,
    ZhangProfile,
    fit_zhang_profile,
)
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "ZHANG2024_DRA_MODEL_INFO",
    "Zhang2024DRAModel",
    "Zhang2024DRASpanKernel",
]


class Zhang2024DRASpanKernel(DRASpanKernelBase):
    """Single-span kernel of the Zhang 2024 backward-DRA coherent-EGN control."""

    profile_kind = "zhang_eq1"

    def _fit_profile(self, profile_norm: NDArrayFloat, span_length_m: float) -> ZhangProfile:
        """Fit Eq. (1) to the coupled-ODE profile."""
        return fit_zhang_profile(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.alpha_npm,
            self._n_steps,
        )


ZHANG2024_DRA_MODEL_INFO = ModelInfo(
    name="zhang2024_dra_egn",
    summary=(
        "Zhang 2024 backward-DRA coherent-EGN subset: Eq. (1) fitted profile and "
        "Eq. (4) per-span coherent NLI sum; XPM_corr and the dataset calibration "
        "are not reproduced (GN subset, not the calibrated EGN)."
    ),
    family="gn+raman+coherent",
    reference="Zhang et al., OECC 2024, doi:10.1109/OECC54135.2024.10975528",
    options=(
        ("raman_pumps", "Backward RamanPumpConfig sequence; omitted uses FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear' (CZ)."),
        ("raman_ase", "Expose the newly generated Raman ASE before the EDFA. Default True."),
        ("raman_temperature_k", "Fibre temperature of the spontaneous Raman factor. Default 300 K."),
        ("raman_peak_gain_coefficient_m_per_w", "Peak local Raman gain rate. Default 6e-14."),
        ("include_isrs", "Include channel-channel ISRS in the coupled ODE. Default True."),
        ("fixed_point_iterations", "Maximum backward-pump fixed-point iterations. Default 24."),
        ("n_steps", "RK4 steps of the ODE and of the Eq. (1) fit. Default 32."),
        ("n_quadrature", "Gauss-Legendre nodes of the in-band NLI integral. Default 24."),
    ),
)


class Zhang2024DRAModel(DRARamanModelBase):
    """Zhang 2024 backward-DRA coherent-EGN control.

    Args:
        grid: Spectrum grid.
        ref_fiber_config: Reference fibre.
        symbol_rate_hz: Symbol rate (Hz).
        connections: Optional per-connection grid.
        raman_pumps: Backward pumps applied to every span; ``None`` uses the fibre config.
        raman_gain_model: ``"gaussian"`` or ``"linear"``.
        raman_ase: Include the Raman ASE before the EDFA.
        raman_temperature_k: Fibre temperature (K).
        raman_peak_gain_coefficient_m_per_w: Peak local Raman gain rate.
        include_isrs: Include channel-channel ISRS in the ODE.
        fixed_point_iterations: Maximum backward-pump iterations.
        n_steps: RK4 steps of the ODE and profile fit.
        n_quadrature: Gauss-Legendre nodes of the NLI in-band integral.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.

    Raises
    ------
        ValueError: A forward pump is supplied.
    """

    info = ZHANG2024_DRA_MODEL_INFO
    _kernel_class = Zhang2024DRASpanKernel
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
        **route_options: object,
    ) -> None:
        if raman_pumps is not None:
            self._validate_backward(raman_pumps)
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

    @staticmethod
    def _validate_backward(pumps: Sequence[RamanPumpConfig]) -> None:
        if any(str(pump.direction) != "backward" for pump in pumps):
            raise ValueError("zhang2024_dra_egn supports backward Raman pumps only.")

    def _resolved_pumps(self, fiber_config: FiberSpanConfig) -> list[RamanPumpConfig]:
        pumps = super()._resolved_pumps(fiber_config)
        self._validate_backward(pumps)
        return pumps


register_model(ZHANG2024_DRA_MODEL_INFO, Zhang2024DRAModel)
