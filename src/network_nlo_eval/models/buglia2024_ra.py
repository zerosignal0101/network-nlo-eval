"""Buglia 2024 Raman-amplified Gaussian-noise model.

The model solves the coupled signal and pump ODE, fits the five-parameter
profile, evaluates the paper's closed-form SPM and XPM terms, and accumulates
span noise powers. The numerical integral is available as a comparison path.
Both forward and backward pumps and pump depletion are supported.

The implementation follows Appendix D for a missing plus sign in printed
Eq. (12), and uses the derived sine-term sign in Eqs. (9), (11) and (12).
Pump XPM is omitted; the gain spectrum uses the repository's Gaussian or
linear approximation rather than the paper's measured spectrum. The default
SPM coherence exponent is zero because the paper does not specify it.

References
----------
H. Buglia et al., "A Closed-Form Expression for the Gaussian Noise Model in
the Presence of Raman Amplification", J. Lightwave Technol. 42(2), 636-648,
2024, doi:10.1109/jlt.2023.3315127.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "BUGLIA2024_RA_MODEL_INFO",
    "Buglia2024RAModel",
    "Buglia2024RASpanKernel",
]


class Buglia2024RASpanKernel(DRASpanKernelBase):
    """Single-span Buglia 2024 Raman-amplified GN kernel.

    Parameters
    ----------
    nli_form : {"closed", "integral"}, optional
        Use the paper's closed form or the numerical integration baseline.
    spm_coherence_exponent : float, optional
        Eq. (4) SPM coherence exponent.
    """

    profile_kind = "buglia_eq2_5param"
    #: Eq. (4) accumulates powers; the kernel applies n^epsilon without amplitudes.
    emits_nli_amplitudes = False

    def __init__(
        self,
        *args: object,
        nli_form: str = "closed",
        spm_coherence_exponent: float = 0.0,
        **kwargs: object,
    ) -> None:
        if nli_form not in ("closed", "integral"):
            msg = "nli_form must be 'closed' or 'integral'."
            raise ValueError(msg)
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._nli_form = str(nli_form)
        self._spm_exponent = float(spm_coherence_exponent)
        self.gamma_per_w_m = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (
            self.n2_m2_w / self.a_eff_m2
        )

    def _fit_profile(
        self, profile_norm: NDArrayFloat, span_length_m: float
    ) -> BugliaClosedProfile:
        """Fit the five Eq. (2) profile coefficients."""
        pumps = list(self._pumps)
        if pumps:
            powers = np.asarray(
                [float(pump.power_w) for pump in pumps], dtype=np.float64
            )
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
            # Without pumps, use the channel mean as the frequency origin.
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
        """Evaluate per-span SPM and XPM and apply the Eq. (4) coherence factor."""
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
        (
            "raman_pumps",
            "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps.",
        ),
        ("raman_gain_model", "'gaussian' (default) or 'linear' (CZ)."),
        (
            "raman_ase",
            "Expose the newly generated Raman ASE before the EDFA. Default True.",
        ),
        (
            "raman_temperature_k",
            "Fibre temperature of the spontaneous Raman factor. Default 300 K.",
        ),
        (
            "raman_peak_gain_coefficient_m_per_w",
            "Peak local Raman gain rate. Default 6e-14.",
        ),
        (
            "include_isrs",
            "Include channel-channel ISRS in the coupled ODE. Default True.",
        ),
        (
            "fixed_point_iterations",
            "Maximum backward-pump fixed-point iterations. Default 24.",
        ),
        ("n_steps", "RK4 steps of the ODE and of the Eq. (2) fit. Default 32."),
        ("n_quadrature", "Gauss-Legendre nodes of the integral baseline. Default 24."),
        ("nli_form", "'closed' (Eq. 11/12, default) or 'integral' (Eq. 6-8 baseline)."),
        (
            "spm_coherence_exponent",
            "Eq. (4) epsilon; 0 (default) sums spans incoherently.",
        ),
    ),
)


class Buglia2024RAModel(DRARamanModelBase):
    """Buglia 2024 Raman-amplified GN model with forward and backward pumps.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Spectrum grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber parameters.
    symbol_rate_hz : float, optional
        Symbol rate in Hz.
    connections : sequence of ConnectionSpec or None, optional
        Optional flexible-grid connections.
    raman_pumps : sequence of RamanPumpConfig or None, optional
        Per-span pumps; None uses the fiber configuration.
    raman_gain_model : {"gaussian", "linear"}, optional
        Raman gain spectrum model.
    raman_ase : bool, optional
        Include newly generated Raman ASE before the EDFA.
    raman_temperature_k : float, optional
        Fiber temperature in kelvin.
    raman_peak_gain_coefficient_m_per_w : float, optional
        Peak Raman gain coefficient in m/W.
    include_isrs : bool, optional
        Include signal-signal ISRS.
    fixed_point_iterations : int, optional
        Maximum backward-pump iterations.
    n_steps : int, optional
        ODE RK4 step count.
    n_quadrature : int, optional
        Frequency quadrature nodes for the integral baseline.
    nli_form : {"closed", "integral"}, optional
        Choose the closed form or numerical baseline.
    spm_coherence_exponent : float, optional
        Eq. (4) SPM coherence exponent.
    **route_options
        Options passed to the shared route model.
    """

    info = BUGLIA2024_RA_MODEL_INFO
    _kernel_class = Buglia2024RASpanKernel
    #: The power-domain coherence factor needs the span context.
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
            msg = "nli_form must be 'closed' or 'integral'."
            raise ValueError(msg)
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
        """Build a span kernel with the selected NLI calculation."""
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
