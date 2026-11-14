"""Zhang 2024 backward-Raman coherent GN model and fitted calibration.

The coupled ODE profile is fitted to the three-exponential form. Coherent
inter-span phase is evaluated inside the frequency integral. Same-span terms
use analogous closed forms; unpublished cross-span terms are integrated
numerically. Calibration coefficients were fitted in this repository, as the
paper did not publish its coefficients or data. The paper's modulation-format
correction is also unpublished, so this is not the complete EGN model.

References
----------
Y. Zhang et al., "Accurate Closed-Form Coherent EGN Model Supporting
Distributed Raman Amplification", OECC 2024, doi:10.1109/OECC54135.2024.10975528.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from network_nlo_eval.models.base import (
    ModelInfo,
    PathQoTResult,
    SpanContext,
    register_model,
)
from network_nlo_eval.models.dra_common import (
    DEFAULT_DRA_NLI_QUADRATURE,
    DEFAULT_DRA_STEPS,
    DRARamanModelBase,
    DRASpanKernelBase,
)
from network_nlo_eval.models.raman import C_LIGHT
from network_nlo_eval.models.zhang2024_coherent import (
    CalibrationSet,
    CalibrationTerm,
    ZhangCoherentSpan,
    ZhangSpanProfile,
    cfm_and_egn_terms,
    fit_zhang_coherent_profile,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "ZHANG2024_DRA_MODEL_INFO",
    "Zhang2024DRAModel",
    "Zhang2024DRASpanKernel",
    "default_calibration",
    "load_calibration",
]

#: Calibration coefficients fitted in this repository.
_CALIBRATION_FILE = Path(__file__).with_name("data") / "zhang2024_calibration.json"


def load_calibration(path: Path | None = None) -> dict[str, CalibrationSet]:
    """Load per-term calibration coefficients from JSON.

    Parameters
    ----------
    path : pathlib.Path or None, optional
        File path; None selects the bundled repository-fitted coefficients.

    Returns
    -------
    dict of str to CalibrationSet
        Calibration sets by term, or an empty dictionary if the file is absent.
    """
    target = _CALIBRATION_FILE if path is None else Path(path)
    if not target.exists():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    result: dict[str, CalibrationSet] = {}
    for term, entries in payload.get("terms", {}).items():
        result[term] = CalibrationSet(
            terms=tuple(
                CalibrationTerm(name=item["name"], parameters=tuple(item["parameters"]))  # type: ignore[arg-type]
                for item in entries
            )
        )
    return result


def default_calibration() -> dict[str, CalibrationSet]:
    """Load the bundled repository-fitted calibration coefficients."""
    return load_calibration()


class Zhang2024DRASpanKernel(DRASpanKernelBase[ZhangSpanProfile]):
    """Evaluate ODE power and profile for a coherent Zhang route.

    The kernel returns zero NLI; the route evaluates coherent NLI across all spans
    because phase accumulation must occur inside the frequency integral.
    """

    profile_kind = "zhang_eq1"
    emits_nli_amplitudes = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.last_profile: ZhangSpanProfile | None = None
        self.gamma_per_w_m = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (
            self.n2_m2_w / self.a_eff_m2
        )

    def _fit_profile(
        self, profile_norm: NDArrayFloat, span_length_m: float
    ) -> ZhangSpanProfile:
        """Fit the four Eq. (1) profile parameters."""
        return fit_zhang_coherent_profile(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_npm,
            self._n_steps,
        )

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: ZhangSpanProfile,
        span_context: SpanContext | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Record the profile and defer NLI to the route-level calculation."""
        del power_internal, span_context
        self.last_profile = profile
        return np.zeros(self.n_nodes, dtype=np.float64), np.zeros(
            self.n_nodes, dtype=np.float64
        )


ZHANG2024_DRA_MODEL_INFO = ModelInfo(
    name="zhang2024_dra_egn",
    summary=(
        "Zhang 2024 backward-DRA coherent EGN: Eq. (1) profile, Eq. (2) closed-form "
        "longitudinal integral, Eq. (4) span-coherent NLI with the inter-span phase "
        "inside the frequency integral, plus the Sec. II-C per-term calibration "
        "fitted in this repository. eta_XPM_corr is not printed in the paper and is "
        "not implemented."
    ),
    family="egn+raman+coherent",
    reference="Zhang et al., OECC 2024, doi:10.1109/OECC54135.2024.10975528",
    options=(
        (
            "raman_pumps",
            "Backward RamanPumpConfig sequence; omitted uses FiberSpanConfig.raman_pumps.",
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
        ("n_steps", "RK4 steps of the ODE and of the Eq. (1) fit. Default 32."),
        (
            "n_quadrature",
            "Gauss-Legendre nodes of the XPM frequency integral. Default 24.",
        ),
        (
            "n_spm_quadrature",
            "Gauss-Legendre nodes per SPM frequency direction. Default 24.",
        ),
        (
            "calibration",
            "Override the per-term calibration dict; None uses the built-in JSON.",
        ),
        ("use_calibration", "Apply the Sec. II-C calibration factors. Default True."),
        (
            "modulation_format_correction",
            "Reserved for eta_XPM_corr; the paper does not print it.",
        ),
    ),
)


class Zhang2024DRAModel(DRARamanModelBase):
    """Backward-Raman model with coherent inter-span GN and fitted calibration.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Spectrum grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber parameters.
    symbol_rate_hz : float, optional
        Symbol rate in Hz.
    connections : sequence or None, optional
        Flexible-grid connections.
    raman_pumps : sequence of RamanPumpConfig or None, optional
        Backward pumps; None uses the fiber configuration.
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
        Maximum backward-pump fixed-point iterations.
    n_steps : int, optional
        ODE RK4 step count.
    n_quadrature, n_spm_quadrature : int, optional
        XPM and per-dimension SPM frequency quadrature node counts.
    calibration : dict or None, optional
        Per-term calibration overrides; None loads the bundled coefficients.
    use_calibration : bool, optional
        Apply calibration factors.
    modulation_format_correction : bool, optional
        Reserved for the unpublished correction; True is rejected.
    **route_options
        Options forwarded to the shared route model.

    Raises
    ------
    ValueError
        If a forward pump or the unpublished modulation correction is requested.
    """

    info = ZHANG2024_DRA_MODEL_INFO
    _kernel_class = Zhang2024DRASpanKernel
    supports_span_context = False

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
        n_spm_quadrature: int = 24,
        calibration: dict[str, CalibrationSet] | None = None,
        use_calibration: bool = True,
        modulation_format_correction: bool = False,
        **route_options: object,
    ) -> None:
        if modulation_format_correction:
            msg = (
                "eta_XPM_corr is not printed in Zhang 2024 (Eq. (3) only cites Carena [12]); "
                "this model cannot implement it faithfully."
            )
            raise ValueError(msg)
        if raman_pumps is not None:
            self._validate_backward(raman_pumps)
        if n_spm_quadrature < 8:
            msg = "n_spm_quadrature must be >= 8."
            raise ValueError(msg)
        self._n_spm_quadrature = int(n_spm_quadrature)
        self._calibration = dict(
            default_calibration() if calibration is None else calibration
        )
        self._use_calibration = bool(use_calibration)
        self._span_records: list[
            tuple[ZhangSpanProfile, NDArrayFloat, float, float]
        ] = []
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
            msg = "zhang2024_dra_egn supports backward Raman pumps only."
            raise ValueError(msg)

    def _resolved_pumps(self, fiber_config: FiberSpanConfig) -> list[RamanPumpConfig]:
        pumps = super()._resolved_pumps(fiber_config)
        self._validate_backward(pumps)
        return pumps

    def span_step(
        self,
        fiber_config,
        edfa_config,
        power_in_w,
        span_length_m,
        span_context: SpanContext | None = None,
    ):
        """Record the span profile and input powers for coherent route accumulation."""
        element = super().span_step(
            fiber_config, edfa_config, power_in_w, span_length_m, span_context
        )
        kernel = self.kernel_for(fiber_config)
        profile = getattr(kernel, "last_profile", None)
        if profile is not None:
            transfer = float(
                np.mean(
                    np.divide(
                        element.power_out_w,
                        power_in_w,
                        out=np.ones_like(power_in_w),
                        where=power_in_w > 0,
                    )
                )
            )
            self._span_records.append(
                (
                    profile,
                    np.array(power_in_w, dtype=np.float64, copy=True),
                    float(span_length_m),
                    transfer,
                )
            )
        return element

    def evaluate_path(self, optical_path, power_in_w: NDArrayFloat) -> PathQoTResult:
        """Evaluate the route and replace NLI with coherent link-level calculations.

        The shared route layer supplies signal, ASE and ROADM behavior. Recorded span
        profiles and spectra are used for same-span closed terms and numerically
        integrated cross-span terms, with optional per-term calibration.
        """
        self._span_records = []
        result = super().evaluate_path(optical_path, power_in_w)
        if not self._span_records:
            return result

        kernel = self.kernel_for(optical_path.fiber_configs[0])
        order = kernel.order
        inverse = kernel.inverse_order
        bandwidths = self.channel_bandwidth_hz_channels[order]
        gamma = kernel.gamma_per_w_m
        beta2 = kernel.beta2
        beta3 = kernel.beta3
        reference_hz = kernel.reference_frequency_hz

        spans = []
        start = 0.0
        for profile, launch, length, _transfer in self._span_records:
            spans.append(
                ZhangCoherentSpan(
                    profile, np.asarray(launch, dtype=np.float64)[order], start
                )
            )
            start += length
        reference = spans[0].launch_power_w

        eta_spm_cfm, eta_xpm_cfm, eta_spm_all, eta_xpm_all = cfm_and_egn_terms(
            spans,
            reference,
            bandwidths,
            gamma,
            beta2,
            beta3,
            reference_hz,
            n_frequency=max(16, int(self._n_quadrature) * 2),
            n_spm_frequency=max(8, self._n_spm_quadrature),
        )
        del eta_spm_all, eta_xpm_all

        if self._use_calibration and self._calibration:
            params = {
                "launch_power_dbm": 10.0
                * np.log10(max(float(np.mean(reference[reference > 0])), 1e-30) * 1e3),
                "spans": float(len(spans)),
            }
            term_spm = self._calibration.get("spm")
            term_xpm = self._calibration.get("xpm")
            if term_spm is not None:
                eta_spm_cfm = eta_spm_cfm * term_spm.apply(params)
            if term_xpm is not None:
                eta_xpm_cfm = eta_xpm_cfm * term_xpm.apply(params)

        spm = eta_spm_cfm * reference**3
        xpm = eta_xpm_cfm * reference**3
        spm_model = np.ascontiguousarray(spm[inverse])
        xpm_model = np.ascontiguousarray(xpm[inverse])

        signal = np.asarray(result.signal_power_w, dtype=np.float64)
        total_noise = (
            spm_model + xpm_model + np.asarray(result.ase_noise_w, dtype=np.float64)
        )
        snr = np.full_like(signal, -np.inf)
        active = (signal > 0) & (total_noise > 0)
        snr[active] = 10.0 * np.log10(signal[active] / total_noise[active])
        snr[(signal > 0) & (total_noise == 0)] = np.inf
        return PathQoTResult(
            signal_power_w=result.signal_power_w,
            spm_noise_w=spm_model,
            xpm_noise_w=xpm_model,
            ase_noise_w=result.ase_noise_w,
            snr_db=snr,
            span_count=result.span_count,
            roadm_count=result.roadm_count,
        )


register_model(ZHANG2024_DRA_MODEL_INFO, Zhang2024DRAModel)
