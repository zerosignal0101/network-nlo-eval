"""Semrau 2019 closed-form ISRS-GN model with optional format correction.

The main GN and ISRS approximation follows Semrau, Killey, and Bayvel,
JLT 2019, doi:10.1109/JLT.2019.2895237. Optional modulation correction
follows their second 2019 paper, doi:10.1109/JLT.2019.2929461.

The single-span SPM, XPM, and Raman power profile use the paper's closed
forms. Inter-span coherence is an implementation extension: SPM visibility
is weighted by the GN noise kernel, XPM visibility uses a band-averaged
approximation, and complex span amplitudes are accumulated by the shared
route layer. A phase ramp is chosen to reproduce the calculated coherence
factor, rather than directly applying the paper's empirical span exponent.

Notes
-----
The format correction retains the amplitude component of the multispan
structure and is amplified by the same coherence factor. The model does
not model Raman amplification or distributed Raman pumps. The phase-ramp
mapping assumes equal spans; unequal spans are an approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.special import sici

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
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
from network_nlo_eval.models.semrau2019_kernels import (
    build_semrau_matrices,
    gn_weighted_spm_visibility,
    semrau_span_jit,
)

if TYPE_CHECKING:
    from network_nlo_eval.core.flexgrid import FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "EXCESS_KURTOSIS",
    "SEMRAU2019_MODEL_INFO",
    "Semrau2019Model",
    "Semrau2019SpanKernel",
]

#: Raman gain slope in 1/(W m Hz), shared with the ECZ baseline
#: so both models use the same Raman strength.
SRS_CR_W_M_HZ = _SRS_CR

#: Modulation excess kurtosis from 2019b Table I.
EXCESS_KURTOSIS: dict[str, float] = {
    "gaussian": 0.0,
    "qpsk": -1.0,
    "16qam": -0.6800,
    "64qam": -0.6190,
    "256qam": -0.6050,
}

#: Default EGN excess kurtosis is for uniform 64-QAM.
DEFAULT_EXCESS_KURTOSIS = EXCESS_KURTOSIS["64qam"]

#: Bisection steps for the equivalent inter-span phase ramp.
#:
#: The Dirichlet kernel decreases from N² to zero on its first lobe;
#: 60 bisections provide a narrow interval and relative coherence error
#: of about 1e-13 for N=10, below the 1e-12 target.
#:
RAMP_BISECTION_STEPS = 60

#: Registry metadata: 2019a closed-form GN with optional 2019b
#: modulation-format correction.
SEMRAU2019_MODEL_INFO = ModelInfo(
    name="semrau2019",
    summary="Semrau 2019 closed-form ISRS-GN; optional modulation-format correction.",
    family="gn+egn",
    reference="Semrau et al., JLT 2019, doi:10.1109/jlt.2019.2895237; "
    "format correction: doi:10.1109/jlt.2019.2929461",
    options=(("egn", "Apply the modulation-format correction of the 2019b paper."),),
)


# =============================================================================
# Inter-span coherence: visibility, coherence factor, and phase ramp.
# =============================================================================
def _self_beating_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
    span_length_m: float,
    alpha_npm: NDArrayFloat,
) -> NDArrayFloat:
    """Return GN-weighted SPM visibility for one span separation.

    The visibility is the band average of the cosine of inter-span phase,
    weighted by the NLI-generating GN kernel. This differs from a uniform
    signal-spectrum average; see ``gn_weighted_spm_visibility`` for the
    one-dimensional quadrature.

    Parameters
    ----------
    beta2_abs_s2_m : float
        Magnitude of second-order dispersion in s²/m.
    bandwidth_hz : numpy.ndarray
        Per-channel bandwidths in Hz.
    distance_m : float
        Distance between span copies in meters.
    span_length_m : float
        Length of one span in meters.
    alpha_npm : numpy.ndarray
        Per-channel power attenuation in nepers per meter.

    Returns
    -------
    numpy.ndarray
        Per-channel visibility between -1 and 1.
    """
    return cast(
        "NDArrayFloat",
        gn_weighted_spm_visibility(
            beta2_abs_s2_m=beta2_abs_s2_m,
            bandwidth_hz=bandwidth_hz,
            alpha_npm=alpha_npm,
            span_length_m=span_length_m,
            span_separations_m=np.array([float(distance_m)], dtype=np.float64),
        )[0],
    )


def _signal_spectrum_spm_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
) -> NDArrayFloat:
    """Return the legacy uniform-spectrum SPM visibility for comparison.

    Unlike the GN-weighted visibility, this signal-spectrum average does not
    weight frequency pairs by generated NLI power.

    Parameters
    ----------
    beta2_abs_s2_m : float
        Magnitude of second-order dispersion in s²/m.
    bandwidth_hz : numpy.ndarray
        Per-channel bandwidths in Hz.
    distance_m : float
        Distance between span copies in meters.

    Returns
    -------
    numpy.ndarray
        Per-channel visibility.
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
    """Calculate band-averaged XPM visibility between span copies.

    Parameters
    ----------
    beta2_abs_s2_m : float
        Magnitude of second-order dispersion in s²/m.
    bandwidth_hz : numpy.ndarray
        Per-channel bandwidths in Hz.
    distance_m : float
        Distance between span copies in meters.
    partner_offset_hz : numpy.ndarray
        Per-channel frequency separation from the interfering channel in Hz.

    Returns
    -------
    numpy.ndarray
        Per-channel visibility between -1 and 1.
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
    """Calculate the inter-span coherence factor from pairwise visibility.

    Parameters
    ----------
    n_spans : int
        Number of spans.
    visibility : numpy.ndarray
        Visibility by span separation and channel, shape ``(n_spans - 1, N)``.

    Returns
    -------
    numpy.ndarray
        Coherence factor per channel. Perfect coherence gives ``n_spans**2``;
        zero visibility gives ``n_spans``.
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return cast("NDArrayFloat", n + 2.0 * (weights * visibility).sum(axis=0))


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Invert the coherence factor into an equivalent phase step per span.

    Parameters
    ----------
    n_spans : int
        Number of spans.
    target : numpy.ndarray
        Target coherence factor for each channel.

    Returns
    -------
    numpy.ndarray
        Equivalent per-channel phase step in radians.

    Notes
    -----
    Bisection in the first lobe of the Dirichlet kernel finds the phase step
    that produces the target coherent sum for equal-amplitude spans.
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
    """Convert the paper's span exponent to an equivalent constant visibility.

    For N > 1, equating ``N * (1 + v * (N - 1))`` with ``N**(1 + epsilon)``
    gives ``v = (N**epsilon - 1) / (N - 1)``.
    """
    n = int(n_spans)
    if n <= 1:
        return 1.0
    return float((float(n) ** coherence_exponent - 1.0) / (n - 1))


class Semrau2019SpanKernel:
    """Single-span closed-form ISRS-GN kernel with coherent NLI amplitudes.

    Channel and channel-pair coefficients are precomputed at construction.
    The kernel caches coherence factors and phase factors for repeated spans.
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
        """Prepare fiber-dependent channel and pair coefficients.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber attenuation, dispersion, and effective-area configuration.
        lambdas_m, f_rel_hz, b_hz : numpy.ndarray
            Per-channel wavelengths, relative frequencies, and bandwidths.
        band_center_hz : float
            Frequency at which reference dispersion is evaluated, in Hz.
        egn : bool
            Whether to include the modulation-format correction.
        excess_kurtosis : float
            Modulation excess kurtosis; used only if ``egn`` is enabled.
        srs_cr_w_m_hz : float
            Raman gain slope in inverse watt-meter-hertz.
        """
        self.fiber_config = fiber_config
        self.b_hz = np.asarray(b_hz, dtype=np.float64)
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.egn = bool(egn)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)
        # With EGN disabled, zero kurtosis reduces the 2019b correction
        # to the 2019a GN expression without a separate formula.
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

        # Evaluate dispersion at the band-center wavelength, as in ECZ.
        ref_lambda_m = C_LIGHT / float(band_center_hz)
        beta2 = _calc_beta2_from_d(
            self.fiber_config.dispersion_parameter_d, ref_lambda_m
        )
        beta3 = _calc_beta3_from_s_d(
            self.fiber_config.dispersion_slope_s,
            self.fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

        self.alpha_npm = alpha_npm
        #: Inter-span visibility uses the magnitude of second-order dispersion.
        self.beta2_s2_m = float(beta2)
        self.beta2_abs_s2_m = abs(float(beta2))
        #: Nearest-neighbor frequency offset in Hz for XPM visibility.
        if self.f_rel_hz.size > 1:
            offsets = np.abs(self.f_rel_hz[:, None] - self.f_rel_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            self.partner_offset_hz = offsets.min(axis=1)
        else:
            self.partner_offset_hz = np.zeros(self.f_rel_hz.size, dtype=np.float64)
        # Coherence depends on span count, length, and fixed fiber properties,
        # not on launch power, so it can be cached.
        self._coherence_cache: dict[tuple[int, float], tuple[NDArrayFloat, ...]] = {}
        # Cache phase factors by span count, length, and normalized distance.
        self._factor_cache: dict[
            tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]
        ] = {}
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

    # --- Inter-span coherence ----------------------------------------------
    def _coherence(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Return cached SPM and XPM phase ramps and coherence factors.

        Parameters
        ----------
        n_spans : int
            Number of spans.
        span_length_m : float
            Span length in meters.

        Returns
        -------
        tuple of numpy.ndarray
            SPM and XPM phase steps in radians, followed by their coherence
            factors. Each array has one value per channel.
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._coherence_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        n = max(1, int(n_spans))
        distances = np.arange(1, n, dtype=np.float64) * float(span_length_m)
        if n > 1:
            # SPM visibility is weighted by the GN kernel.
            visibility_spm = gn_weighted_spm_visibility(
                beta2_abs_s2_m=self.beta2_abs_s2_m,
                bandwidth_hz=self.b_hz,
                alpha_npm=self.alpha_npm,
                span_length_m=float(span_length_m),
                span_separations_m=distances,
            )
            # XPM uses the nearest interfering-channel offset; both visibility
            # conventions are nearly incoherent for the audited cases.
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
            )
        else:
            visibility_spm = np.zeros((0, self.b_hz.size), dtype=np.float64)
            visibility_xpm = np.zeros((0, self.b_hz.size), dtype=np.float64)
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
        """Return per-channel SPM and XPM amplitude phases for a span.

        Phase scales with cumulative distance divided by the current span length.
        """
        ramp_spm, ramp_xpm, _, _ = self._coherence(
            span_context.total_spans, span_length_m
        )
        scale = span_context.cumulative_length_m / float(span_length_m)
        return ramp_spm * scale, ramp_xpm * scale

    def _span_phase_factors(
        self,
        span_context: SpanContext,
        span_length_m: float,
    ) -> tuple[NDArrayComplex, NDArrayComplex]:
        """Return cached unit complex phase factors for a span's SPM and XPM."""
        scale = span_context.cumulative_length_m / float(span_length_m)
        key = (int(span_context.total_spans), float(span_length_m), float(scale))
        cached = self._factor_cache.get(key)
        if cached is None:
            phase_spm, phase_xpm = self._span_phases(span_context, span_length_m)
            cached = (np.exp(1j * phase_spm), np.exp(1j * phase_xpm))
            self._factor_cache[key] = cached
        return cached

    # --- Span evaluation ---------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Evaluate fiber output power and nonlinear noise for one span.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Per-channel input power in watts.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Link span count used when span context is absent.
        span_context : SpanContext or None, optional
            Position in the link; when supplied for a multispan path, the result
            includes complex amplitudes for coherent accumulation.

        Returns
        -------
        SpanFiberOutput
            Fiber output and per-span nonlinear noise, with optional amplitudes.
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            msg = f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            raise ValueError(msg)
        if span_length_m <= 0.0:
            msg = f"span_length_m must be positive, got {span_length_m}."
            raise ValueError(msg)
        n_spans = max(
            1,
            int(span_context.total_spans)
            if span_context is not None
            else int(n_effective_spans),
        )
        # Per-span output retains the 2019b single-span and asymptotic terms;
        # the GN term is this span's noise, and inter-span coherence is
        # represented by complex amplitudes rather than a kernel factor.
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
            # A single span uses the power-only route convention.
            return SpanFiberOutput(power_out, spm, xpm)
        factor_spm, factor_xpm = self._span_phase_factors(
            span_context, float(span_length_m)
        )
        amplitude_spm = np.sqrt(spm) * factor_spm
        amplitude_xpm = np.sqrt(xpm) * factor_xpm
        return SpanFiberOutput(power_out, spm, xpm, amplitude_spm, amplitude_xpm)

    def link_level_total(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> SpanFiberOutput:
        """Calculate equal-span link NLI using analytic coherence factors.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Per-channel launch power in watts.
        span_length_m : float
            Span length in meters.
        n_spans : int
            Number of equal spans.

        Returns
        -------
        SpanFiberOutput
            Fiber output and link-level SPM and XPM noise totals.
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
        """Calculate the 2019b Eq. (16) multispan power-only reference total.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Per-channel launch power in watts.
        span_length_m : float
            Span length in meters.
        n_spans : int
            Number of spans.

        Returns
        -------
        SpanFiberOutput
            Fiber output and paper-formula link-level SPM and XPM totals.
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            msg = f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            raise ValueError(msg)
        if span_length_m <= 0.0:
            msg = f"span_length_m must be positive, got {span_length_m}."
            raise ValueError(msg)
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
    """Route-level Semrau 2019 ISRS-GN model.

    The shared route layer handles amplification, noise accumulation, and
    SNR. Optional EGN correction uses the specified modulation kurtosis;
    setting Raman gain slope to zero disables ISRS.
    """

    info = SEMRAU2019_MODEL_INFO

    #: Supply complex amplitudes for coherent route accumulation.
    supports_span_context = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        *,
        excess_kurtosis: float = DEFAULT_EXCESS_KURTOSIS,
        srs_cr_w_m_hz: float = SRS_CR_W_M_HZ,
        **route_options: Any,
    ) -> None:
        """Initialize channel geometry and Semrau model options.

        Parameters
        ----------
        grid : SpectrumGrid or FlexibleGrid
            Fixed-channel or flexible-connection grid.
        ref_fiber_config : FiberSpanConfig
            Reference fiber configuration.
        symbol_rate_hz : float, optional
            Symbol rate in Hz.
        egn : bool, optional
            Enable the 2019b modulation-format correction.
        excess_kurtosis : float, optional
            Modulation excess kurtosis, used when EGN correction is enabled.
        srs_cr_w_m_hz : float, optional
            Raman gain slope; zero disables ISRS.
        **route_options
            Additional options forwarded to ``RouteModel``.
        """
        super().__init__(
            grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options
        )
        self.egn = bool(egn)
        self.excess_kurtosis = float(excess_kurtosis)
        self.srs_cr_w_m_hz = float(srs_cr_w_m_hz)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> Semrau2019SpanKernel:
        """Build a fiber-specific single-span kernel for the route cache."""
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
        """Evaluate a fiber span and amplifier, including coherent amplitudes.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber properties for the span.
        edfa_config : EDFAConfig or None
            Span amplifier configuration.
        power_in_w : numpy.ndarray
            Per-channel input power in watts.
        span_length_m : float
            Span length in meters.
        span_context : SpanContext or None, optional
            Span position used to calculate relative NLI phase.

        Returns
        -------
        ElementOutput
            Signal and noise after the span and amplifier.

        Notes
        -----
        Complex amplitudes reference fiber output before amplification, following
        the same NLI gain convention as the shared power-only route.
        """
        kernel = cast("Semrau2019SpanKernel", self.kernel_for(fiber_config))
        fiber_output = kernel.evaluate(
            power_in_w, span_length_m, 1, span_context=span_context
        )
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
