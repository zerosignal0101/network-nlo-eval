r"""Numerical ISRS-GN reference model.

This is the ground-truth model of the multi-model comparison: the numerical
counterpart of the closed-form GN models, implemented directly from

    C. Semrau, R. Killey, P. Bayvel,
    "The Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman
    Scattering", JLT 2018, doi:10.1109/jlt.2018.2830973

Two things are done numerically, where the closed-form models use algebraic
approximations:

* the inter-channel SRS power profile is obtained by integrating the coupled
  ordinary differential equations of Eq. (6) of the paper over the span, with a
  Raman gain spectrum that rolls off beyond the Stokes shift instead of the
  triangular approximation;
* the GN non-linear interference is obtained by evaluating the double integral
  of Eq. (4) of the paper with a composite quadrature over the frequency
  variables.

The frequency integral of Eq. (4) runs over the offsets of the two input waves
inside their own bands, ``f1 = f_i + x`` and ``f2 = f_l + y``; the generated
third wave then sits at ``f3 = f_l + x + y``, so the piecewise constant
third-wave PSD is split exactly at the band edges.  The remaining integral over
the intra-band offset uses a graded (cubic) grid that clusters the nodes
towards the band centre, where the phase mismatch of the self-interference
vanishes.  The squared transfer function of the span is the Cauchy-like kernel
of the phase mismatch; because the intra-band mismatch oscillates with a period
of ~156 MHz, far below the channel bandwidth, the kernel is evaluated with its
band average, which reproduces a brute-force evaluation of Eq. (4) to better
than 0.5 % and reproduces the closed-form GN limit exactly in the coherent
(zero dispersion) limit.

Cross-span coherent accumulation
--------------------------------

The paper's multi-span term, Eq. (5), is the phased-array factor

.. math::

    \chi(N, \psi) = \left[\frac{\sin(N\psi/2)}{N\sin(\psi/2)}\right]^2 ,

i.e. the *normalised* coherent gain of ``N`` identical spans whose pairwise
phase mismatch is ``psi`` per span.  In this repository the route layer owns
the cross-span summation, so Eq. (5) is no longer applied inside the kernel:
the kernel returns the **single-span** physical noise (the factor is ``1`` for
``N = 1``) and the route layer adds the spans **coherently**,

.. math::

    P_{\mathrm{NLI}}^{\mathrm{link}} = \left| \sum_k a_k \right|^2 ,

where ``a_k`` is the complex amplitude of the noise generated in span ``k``.
The route layer only applies real power transfer factors; the *phase* between
the spans is the model's business, and this module supplies it.

Why a visibility, and not a single phase mismatch
-------------------------------------------------

The phase mismatch of Eq. (4) is ``-4 pi^2 (f1-f)(f2-f) [beta2 + pi beta3
(f1+f2)] z``.  Two copies of the same noise component generated in two
different spans therefore differ in phase by that mismatch times the distance
between the spans.  The noise is not a single tone, though: it is a continuum
over the channel band, and the mismatch is a product of two frequency offsets
(two intra-band ones for the self-beating, an intra-band and an inter-channel
one for the cross-beating).  The physically relevant quantity is therefore the
*band-averaged visibility*

.. math::

    V(d) = \left\langle \cos\!\big(4\pi^2 |\beta_2| \, \nu \nu' d\big)
    \right\rangle_{\nu, \nu'} = \frac{\mathrm{Si}(z)}{z},\qquad
    z = \pi^2 |\beta_2| d B^2 ,

for the self-beating (SPM) part, and

.. math::

    V_{\mathrm{cross}}(d) = \mathrm{sinc}\!\big(2\pi^2 |\beta_2| d
    \Delta B\big) ,

for the cross-beating (XPM) part, whose phase mismatch carries the
inter-channel offset ``Delta`` on top of the intra-band one.

``V`` is the visibility of the two copies of
the noise: ``V = 1`` means the copies add coherently, ``V = 0`` means they add
incoherently.  The exact frequency-resolved coherent factor of the link is the
visibility-weighted pair sum

.. math::

    R = N + 2 \sum_{\Delta s = 1}^{N-1} (N - \Delta s)\, V(\Delta s L) ,

which is ``N^2`` when the dispersion vanishes (fully coherent) and ``N`` when
the dispersion is large (fully incoherent).

The interface of the route layer carries **one** complex amplitude per channel
and span, so the pairwise structure ``V(j-k)`` cannot be represented
explicitly.  This module therefore uses the *equivalent phase ramp*: the
per-span phase increment ``psi`` is chosen such that the coherent sum of the
route layer reproduces the exact factor ``R``,

.. math::

    \left[\frac{\sin(N\psi/2)}{\sin(\psi/2)}\right]^2 = R ,

with the phase of span ``k`` growing linearly with the accumulated link length
(``cumulative_length_m``), exactly as the accumulated phase of a wave born in
that span.  The equivalence is exact for the quantity the route layer needs
(the total NLI power) and is an approximation for the pairwise structure,
which is documented in the work-package report.

The model is deliberately slow rather than clever: its value is traceability,
not speed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.special import sici

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
    register_model,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
)
from network_nlo_eval.models.numerical_kernels import _gn_nli_jit, _solve_isrs_profile_jit
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "NUMERICAL_MODEL_INFO",
    "NumericalISRSGN",
    "NumericalISRSGNKernel",
]

#: Default number of distance steps of the power-profile integration.
#:
#: The coupled ODEs of Eq. (6) are integrated with a fixed-step RK4 scheme.
#: The Raman interaction rate over a 100 km span is of the order of
#: ``C_r * P_tot * B``, i.e. a few tens of e-foldings, so the step has to stay
#: well below the stability limit of RK4; 64 steps is already converged to
#: three digits (see the report).
DEFAULT_ODE_STEPS = 64

#: Default number of quadrature points per integration sub-interval of the
#: frequency integral of Eq. (4).
#:
#: The frequency integral runs over the offset inside the band of the channel of
#: interest.  Its integrand has a narrow coherent peak at the band centre, so
#: the quadrature uses a graded (cubic) grid whose cells cluster towards the
#: centre; 64 cells per half band put the converged value within 0.02 dB of the
#: result obtained with 128 cells (see the report).
DEFAULT_QUADRATURE_POINTS = 64

#: Bisection steps of the equivalent phase ramp.
#:
#: The ramp is the root of a monotone function on ``[0, 2 pi / N]``; 60 steps
#: resolve the ramp to far below double precision.
_RAMP_BISECTION_STEPS = 60

NUMERICAL_MODEL_INFO = ModelInfo(
    name="numerical",
    summary="Numerical ISRS-GN reference: ODE power profile plus double-integral NLI.",
    family="gn",
    reference="Semrau, Killey, Bayvel, JLT 2018, doi:10.1109/jlt.2018.2830973",
    options=(
        ("quadrature_points", "Frequency quadrature points per integration sub-interval."),
        ("ode_steps", "RK4 steps of the coupled ISRS power-profile integration."),
    ),
)


def _self_beating_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
) -> NDArrayFloat:
    """Band-averaged visibility of two self-beating (SPM) NLI copies.

    Eq. (4) of the paper gives the self-interference the phase mismatch
    ``4 pi^2 |beta2| x y L``, where ``x`` and ``y`` are the offsets of the two
    input waves inside the channel band.  Two copies of the same noise
    component generated ``d`` metres apart therefore differ in phase by
    ``a x y`` with ``a = 4 pi^2 |beta2| d``, and the fraction that still
    interferes coherently is the band average of the cosine,

    ``V(d) = <cos(a x y)> = Si(a B^2 / 4) / (a B^2 / 4)``,

    where ``Si`` is the sine integral.  The closed form is used because the
    integrand oscillates faster and faster towards the band edge (the local
    period shrinks like ``1/nu``), which no fixed-point quadrature resolves
    at acceptable cost.

    Args
    ----
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        bandwidth_hz: ``(N,)`` channel bandwidths (Hz).
        distance_m: Distance between the two copies (m).

    Returns
    -------
        ``(N,)`` visibility in ``[-1, 1]``; ``1`` without dispersion.
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
    """Band-averaged visibility of two cross-beating (XPM) NLI copies.

    For a cross pair the phase mismatch of Eq. (4) is
    ``4 pi^2 |beta2| x (Delta + y) L``: the first wave sits inside the band of
    the channel of interest and the second one ``Delta`` away.  The
    inter-channel offset dominates the intra-band one, so the visibility is
    the band average of ``cos(c x)`` with ``c = 4 pi^2 |beta2| d Delta``,

    ``V(d) = sinc(c B / 2)``,

    which vanishes unless the mismatch stays below a radian.

    Args
    ----
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        bandwidth_hz: ``(N,)`` channel bandwidths (Hz).
        distance_m: Distance between the two copies (m).
        partner_offset_hz: ``(N,)`` offset to the interfering channel (Hz).

    Returns
    -------
        ``(N,)`` visibility in ``[-1, 1]``; ``1`` without dispersion.
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
    """Frequency-resolved cross-span coherent factor ``R``.

    Args
    ----
        n_spans: Number of spans of the link.
        visibility: ``(N_spans - 1, N_channels)`` visibility of the copies
            separated by one, two, ... spans.

    Returns
    -------
        ``(N_channels,)`` factor: ``N**2`` for fully coherent accumulation,
        ``N`` for fully incoherent accumulation.
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return n + 2.0 * (weights * visibility).sum(axis=0)


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Per-span phase increment that reproduces a coherent factor.

    The route layer sums the span amplitudes coherently, so ``N`` spans with a
    linear phase ramp ``psi`` give ``[sin(N psi/2)/sin(psi/2)]**2`` times the
    single-span noise.  This function inverts that relation: it returns the
    ramp whose coherent factor equals ``target``.

    The factor decreases monotonically from ``N**2`` at ``psi = 0`` to ``0`` at
    ``psi = 2 pi / N``, so a bisection is enough.

    Args
    ----
        n_spans: Number of spans of the link.
        target: ``(N_channels,)`` wanted coherent factor.

    Returns
    -------
        ``(N_channels,)`` phase increment per span (rad).
    """
    n = int(n_spans)
    if n <= 1:
        return np.zeros(np.shape(target), dtype=np.float64)
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n)
    for _ in range(_RAMP_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        factor = (np.sin(0.5 * n * mid) / np.sin(0.5 * mid)) ** 2
        too_large = factor > goal
        low = np.where(too_large, mid, low)
        high = np.where(too_large, high, mid)
    return 0.5 * (low + high)


class NumericalISRSGNKernel:
    """Single-span kernel of the numerical ISRS-GN model.

    The kernel owns everything that only depends on the fibre and on the
    channel plan: the per-channel attenuation, effective area and dispersion,
    the ordering of the channels (the paper indexes the channels from the
    highest centre frequency) and the quadrature grids.

    Args
    ----
        f_abs_hz: ``(N,)`` channel centre frequencies (Hz).
        b_hz: ``(N,)`` channel bandwidths (Hz).
        fiber_config: Fibre parameters of the span.
        quadrature_points: Frequency quadrature points per sub-interval.
        ode_steps: RK4 steps of the power-profile integration.
    """

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        ode_steps: int = DEFAULT_ODE_STEPS,
    ) -> None:
        """Prepare the per-channel constants of the span.

        Raises
        ------
            ValueError: The quadrature or integration resolution is not
                positive.
        """
        if quadrature_points < 1:
            raise ValueError(f"quadrature_points must be >= 1, got {quadrature_points}.")
        if ode_steps < 1:
            raise ValueError(f"ode_steps must be >= 1, got {ode_steps}.")

        self.quadrature_points = int(quadrature_points)
        self.ode_steps = int(ode_steps)
        self.n_nodes = int(f_abs_hz.shape[0])

        f_abs = np.asarray(f_abs_hz, dtype=np.float64)
        bandwidths = np.asarray(b_hz, dtype=np.float64)
        # The paper numbers the channels from the highest centre frequency, so
        # the internal order is descending; the model order is restored on the
        # way out.
        self.order = np.argsort(-f_abs, kind="stable")
        self.f_abs_hz = np.ascontiguousarray(f_abs[self.order])
        self.b_hz = np.ascontiguousarray(bandwidths[self.order])
        self.lambdas_m = 2.99792458e8 / self.f_abs_hz

        if fiber_config.use_attenuation_polynomial:
            self.alpha_npm = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            alpha = fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            self.alpha_npm = np.full(self.n_nodes, alpha, dtype=np.float64)

        self.a_eff_m2 = _calc_aeff_dynamic(
            self.lambdas_m,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        # Nonlinearity coefficient of every channel, 1/(W m).  The GN model of
        # Eq. (4) carries gamma^2 outside the frequency integral, so a single
        # coefficient per channel is enough.
        self.gamma_w_m = (2.0 * np.pi / self.lambdas_m) * (
            fiber_config.nonlinear_index_n2 / self.a_eff_m2
        )
        # The GN phase of Eq. (4) carries the frequency dependence through
        # beta3*(f1+f2); beta2 and beta3 are taken at the reference wavelength.
        lambda_ref_m = fiber_config.reference_wavelength_nm * 1e-9
        self.beta2 = float(_calc_beta2_from_d(fiber_config.dispersion_parameter_d, lambda_ref_m))
        self.beta3 = float(
            _calc_beta3_from_s_d(
                fiber_config.dispersion_slope_s,
                fiber_config.dispersion_parameter_d,
                lambda_ref_m,
            )
        )
        # The Taylor expansion of the propagation constant is centred on the
        # reference frequency; the third-order dispersion term of the GN phase
        # is expressed relative to it.
        self.f_ref_hz = 2.99792458e8 / lambda_ref_m
        self.n2_m2_w = float(fiber_config.nonlinear_index_n2)
        # Magnitude of the second-order dispersion, the only quantity the
        # cross-span visibility depends on.
        self.beta2_abs = abs(self.beta2)
        # Offset to the closest other channel (Hz).  The cross-beating (XPM)
        # part of the NLI is dominated by the nearest neighbours, because the
        # FWM kernel of Eq. (4) falls off with the inter-channel offset.
        if self.n_nodes > 1:
            offsets = np.abs(self.f_abs_hz[:, None] - self.f_abs_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            self.partner_offset_hz = offsets.min(axis=1)
        else:
            self.partner_offset_hz = np.zeros(self.n_nodes, dtype=np.float64)
        # Cache of the equivalent phase ramps, keyed by (span count, span
        # length).  The ramps are pure functions of those two numbers and of
        # the fibre constants, so the cache is idempotent.
        self._ramp_cache: dict[tuple[int, float], tuple[NDArrayFloat, NDArrayFloat]] = {}

    def _inter_span_ramps(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Equivalent per-span phase ramps of the SPM and XPM parts.

        Args
        ----
            n_spans: Number of spans of the link.
            span_length_m: Span length (m).

        Returns
        -------
            ``(ramp_spm, ramp_xpm)``, each ``(N,)`` in rad per span.
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._ramp_cache.get(key)
        if cached is not None:
            return cached
        distances = np.arange(1, max(int(n_spans), 1), dtype=np.float64) * float(span_length_m)
        visibility_spm = np.stack(
            [
                _self_beating_visibility(self.beta2_abs, self.b_hz, distance)
                for distance in distances
            ]
        )
        visibility_xpm = np.stack(
            [
                _cross_beating_visibility(
                    self.beta2_abs,
                    self.b_hz,
                    distance,
                    self.partner_offset_hz,
                )
                for distance in distances
            ]
        )
        ramp_spm = _equivalent_phase_ramp(n_spans, _coherent_factor(n_spans, visibility_spm))
        ramp_xpm = _equivalent_phase_ramp(n_spans, _coherent_factor(n_spans, visibility_xpm))
        self._ramp_cache[key] = (ramp_spm, ramp_xpm)
        return ramp_spm, ramp_xpm

    def _span_phases(
        self,
        span_context: SpanContext,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Phase of the NLI amplitudes of one span, at the receiver.

        The phase grows linearly with the accumulated link length, i.e. with the
        span index for a uniform link.  Referencing it to the accumulated length
        instead of the remaining length only shifts every amplitude by the same
        global phase, which leaves the coherent sum unchanged.

        Args
        ----
            span_context: Position of the span in the link.
            span_length_m: Span length (m).

        Returns
        -------
            ``(phase_spm, phase_xpm)``, each ``(N,)`` in rad.
        """
        ramp_spm, ramp_xpm = self._inter_span_ramps(span_context.total_spans, span_length_m)
        scale = span_context.cumulative_length_m / float(span_length_m)
        return ramp_spm * scale, ramp_xpm * scale

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Evaluate one span.

        Args
        ----
            power_in_w: ``(N,)`` launch power per channel (W), in model order.
            span_length_m: Span length (m).
            n_effective_spans: Number of identical spans entering the
                phased-array term of Eq. (5) of the paper.  The route layer
                performs the coherent summation itself, so this stays at 1.
            span_context: Position of the span in the link.  When it is given
                and the link has more than one span, the output carries the
                complex NLI amplitudes whose relative phase the route layer
                needs for the coherent accumulation.

        Returns
        -------
            Output powers and the self- (SCI) and cross- (XCI) interference
            noise powers, all in model order.
        """
        power = np.ascontiguousarray(np.asarray(power_in_w, dtype=np.float64)[self.order])
        dz = span_length_m / self.ode_steps
        z_mid = (np.arange(self.ode_steps, dtype=np.float64) + 0.5) * dz

        profile, power_out = _solve_isrs_profile_jit(
            self.f_abs_hz,
            power,
            self.alpha_npm,
            span_length_m,
            self.ode_steps,
        )
        # Split the profile into the fiber attenuation and the SRS-only gain.
        srs_gain = profile * np.exp(self.alpha_npm[None, :] * z_mid[:, None])

        spm = np.zeros(self.n_nodes, dtype=np.float64)
        xpm = np.zeros(self.n_nodes, dtype=np.float64)
        _gn_nli_jit(
            self.f_abs_hz,
            self.b_hz,
            power,
            self.gamma_w_m,
            self.alpha_npm,
            srs_gain,
            self.ode_steps,
            dz,
            int(n_effective_spans),
            float(span_length_m),
            self.beta2,
            self.beta3,
            self.f_ref_hz,
            self.quadrature_points,
            spm,
            xpm,
        )

        inverse = np.empty(self.n_nodes, dtype=np.int64)
        inverse[self.order] = np.arange(self.n_nodes)
        if span_context is None or span_context.total_spans <= 1:
            # Single span: the coherent amplitude degenerates to the noise
            # itself, and the route layer keeps its power-only accumulation.
            return SpanFiberOutput(
                power_out_w=np.ascontiguousarray(power_out[inverse]),
                spm_noise_w=np.ascontiguousarray(spm[inverse]),
                xpm_noise_w=np.ascontiguousarray(xpm[inverse]),
            )

        phase_spm, phase_xpm = self._span_phases(span_context, span_length_m)
        amplitude_spm = np.sqrt(spm) * np.exp(1j * phase_spm)
        amplitude_xpm = np.sqrt(xpm) * np.exp(1j * phase_xpm)
        return SpanFiberOutput(
            power_out_w=np.ascontiguousarray(power_out[inverse]),
            spm_noise_w=np.ascontiguousarray(spm[inverse]),
            xpm_noise_w=np.ascontiguousarray(xpm[inverse]),
            nli_amplitude_spm=np.ascontiguousarray(amplitude_spm[inverse]),
            nli_amplitude_xpm=np.ascontiguousarray(amplitude_xpm[inverse]),
        )


class NumericalISRSGN(RouteModel):
    """Numerical ISRS-GN model: ODE power profile plus double-integral NLI.

    Args
    ----
        grid: Spectrum grid (legacy ``SpectrumGrid`` or ``FlexibleGrid``).
        ref_fiber_config: Reference fibre configuration.
        symbol_rate_hz: Symbol rate (Hz).
        egn: Not supported; the numerical EGN correction is out of scope.
        quadrature_points: Frequency quadrature points per sub-interval.
        ode_steps: RK4 steps of the power-profile integration.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.
    """

    info = NUMERICAL_MODEL_INFO

    #: The model needs to know its position in the link to give the relative
    #: phase of the cross-span NLI amplitudes.
    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        ode_steps: int = DEFAULT_ODE_STEPS,
        **route_options,
    ) -> None:
        """Initialize the model.

        Raises
        ------
            ValueError: ``egn=True`` was requested, or a resolution setting
                is not positive.
        """
        if egn:
            raise ValueError(
                "The numerical ISRS-GN model does not support egn=True: the numerical EGN "
                "correction is out of scope for this work package. Use egn=False."
            )
        if quadrature_points < 1:
            raise ValueError(f"quadrature_points must be >= 1, got {quadrature_points}.")
        if ode_steps < 1:
            raise ValueError(f"ode_steps must be >= 1, got {ode_steps}.")
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.egn = False
        self.quadrature_points = int(quadrature_points)
        self.ode_steps = int(ode_steps)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> NumericalISRSGNKernel:
        """Build the single-span kernel of this fibre."""
        return NumericalISRSGNKernel(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            quadrature_points=self.quadrature_points,
            ode_steps=self.ode_steps,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ):
        """One span: fibre physics plus amplifier, with the NLI amplitudes.

        The frozen route helper scales a coherent amplitude by ``sqrt(gain)``
        when an amplifier is present, while its power-only branch does not
        scale the noise power at all.  The two conventions therefore differ by
        the amplifier gain, and the golden vectors lock the power-only one.  To
        keep the coherent accumulation bit-identical to the frozen behaviour
        for a single span, the helper is called with the power-only fibre
        output and the amplitudes are attached afterwards, at the fibre output
        plane the contract asks for.
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


register_model(NUMERICAL_MODEL_INFO, NumericalISRSGN)
