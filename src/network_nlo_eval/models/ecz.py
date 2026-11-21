"""Coherent ECZ closed-form ISRS-GN model.

The single-span kernel reuses the ECZ triangular Raman profile and closed-form
GN baseline. For multiple spans this model provides complex amplitudes so the
route layer can accumulate the generated nonlinear interference coherently.

SPM visibility is averaged over nonlinear noise power, with the GN weight
``|1 - exp(-(alpha + j*rho)*L)|**2 / (alpha**2 + rho**2)`` and phase
mismatch ``rho*d``. Uniform spectral weighting describes signal rather than
noise coherence and underestimates the C-band ten-span buildup. The visibility
is calculated independently of the numerical reference model.

For N spans, the coherent factor is
``R = N + 2 * sum((N-k) * V(k*L), k=1..N-1)``. The equivalent phase
increment is found by bisection of
``[sin(N*theta/2)/sin(theta/2)]**2 = R`` on ``(0, 2*pi/N)``.

Each span and channel has one complex amplitude, so this phase ramp compresses
the frequency-dependent visibility into a single phase. The inversion assumes
equal span noise amplitudes: unequal spans, fixed-gain amplifiers, or ROADMs
can cause deviations because the route applies transfer weights to the sum.
For XPM, the much larger inter-channel offsets give nearly incoherent copies;
the model uses the numerical reference's uniform-band sinc approximation.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, cast

import numpy as np

from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
    register_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.route import apply_amplifier_and_roadm

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.flexgrid import ConnectionGeometry, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "ECZ_COHERENT_MODEL_INFO",
    "ECZCoherentModel",
]

#: Observed-frequency samples for visibility integration.
#:
#: A cosine grid resolves steep band edges; 41 points agree with 161 to 3e-5 in C band.
VISIBILITY_FREQUENCY_POINTS = 41

#: Gauss-Legendre nodes per visibility subinterval.
#:
#: 200 nodes per subinterval converge to 1e-5 for the smooth outer integral.
VISIBILITY_QUADRATURE_POINTS = 200

#: Lorentzian antiderivative table step.
#:
#: The fourth-order corrected trapezoid rule achieves 1e-6 visibility accuracy.
VISIBILITY_TABLE_STEP = 1.0e-3

#: Maximum table argument; use the asymptotic tail beyond this limit.
#:
#: The tail approximation is accurate to better than 1e-8 at t = 30.
VISIBILITY_TABLE_LIMIT = 30.0

#: Minimum table range, ensuring a well-conditioned table at low dispersion.
#:

VISIBILITY_TABLE_MINIMUM = 1.0

#: Bisection steps for the equivalent phase increment.
#:
#: Sixty steps resolve the monotonic root below double-precision accuracy.
RAMP_BISECTION_STEPS = 60


#: Registry description for coherent ECZ.
#:
#: Uses the ECZ baseline physics and supplies complex amplitudes for coherent routes.
ECZ_COHERENT_MODEL_INFO = ModelInfo(
    name="ecz_coherent",
    summary="ECZ closed-form GN (SPM/XPM) with NLI-weighted inter-span coherence.",
    family="gn",
    reference="Lasagni et al., 'A Generalized Raman Scattering Model for Real-Time SNR "
    "Estimation of Multi-Band Systems', JLT 2023, doi:10.1109/jlt.2023.3250751; "
    "inter-span coherence: Semrau, Killey, Bayvel, JLT 2018, doi:10.1109/jlt.2018.2830973",
    options=(
        ("ode_accel", "'off' | 'cached' | 'matvec'; single-span acceleration tier."),
        (
            "channel_bandwidth_source",
            "'symbol_rate' | 'occupied'; per-node B_i definition.",
        ),
        (
            "power_width_mode",
            "'fixed_power' | 'fixed_psd'; launch-power normalisation.",
        ),
        (
            "launch_power_convention",
            "'total' | 'per_pol'; coordinate of power_in_w and outputs.",
        ),
    ),
)


def _lorentzian_tail(beta: float, t: NDArrayFloat) -> NDArrayFloat:
    """Compute the asymptotic Lorentzian tail by repeated integration by parts.

    Parameters
    ----------
    beta : float
        Dimensionless angular frequency.
    t : ndarray
        Nonnegative integration limits or evaluation points.

    Returns
    -------
    ndarray
        Tail integral per input value.
    """
    denominator = 1.0 + t * t
    f = 1.0 / denominator
    df = -2.0 * t / denominator**2
    d2f = (6.0 * t * t - 2.0) / denominator**3
    d3f = -24.0 * t * (t * t - 1.0) / denominator**4
    d4f = 24.0 * (5.0 * t**4 - 10.0 * t * t + 1.0) / denominator**5
    sin_beta = np.sin(beta * t)
    cos_beta = np.cos(beta * t)
    return cast(
        "NDArrayFloat",
        -f * sin_beta / beta
        - df * cos_beta / beta**2
        + d2f * sin_beta / beta**3
        + d3f * cos_beta / beta**4
        - d4f * sin_beta / beta**5,
    )


class _LorentzianAntiderivative:
    """Tabulate ``C_m(t) = integral_0^t cos(m*alpha*L*s)/(1+s**2) ds``.

    The composite trapezoid rule with an Euler-Maclaurin correction gives fourth
    order accuracy. ``C_0(t) = arctan(t)`` is analytical; beyond the table the
    asymptotic tail is anchored at ``C_m(infinity) = pi/2 * exp(-m*alpha*L)``.

    Parameters
    ----------
    alpha_npm_m : float
        Power attenuation (Np/m).
    span_length_m : float
        Span length (m).
    m_max : int
        Highest index to tabulate.
    t_end : float
        Maximum table argument.
    step : float, optional
        Table spacing.
    """

    def __init__(
        self,
        alpha_npm_m: float,
        span_length_m: float,
        m_max: int,
        t_end: float,
        step: float = VISIBILITY_TABLE_STEP,
    ) -> None:
        """Build the antiderivative tables."""
        self.alpha = float(alpha_npm_m)
        self.span_length = float(span_length_m)
        self.beta = self.alpha * self.span_length
        limit = min(max(float(t_end), VISIBILITY_TABLE_MINIMUM), VISIBILITY_TABLE_LIMIT)
        n_points = max(3, math.ceil(limit / step) + 1)
        if n_points % 2 == 0:
            n_points += 1
        self.t = np.linspace(0.0, limit, n_points)
        h = float(self.t[1] - self.t[0])
        # Calculate all tables together while retaining the per-row arithmetic order.
        index = np.arange(1, int(m_max) + 1, dtype=np.float64)[:, None]
        phase = index * self.beta * self.t[None, :]
        denominator = 1.0 + self.t * self.t
        values = np.cos(phase) / denominator[None, :]
        derivative = (
            -index * self.beta * np.sin(phase) / denominator[None, :]
            - 2.0 * self.t[None, :] * np.cos(phase) / denominator[None, :] ** 2
        )
        table = self._cumulative(values, derivative, h)
        self.tables: dict[int, NDArrayFloat] = {
            m: table[m - 1] for m in range(1, int(m_max) + 1)
        }

    @staticmethod
    def _cumulative(
        values: NDArrayFloat, derivative: NDArrayFloat, h: float
    ) -> NDArrayFloat:
        """Integrate each row on a uniform grid using a corrected trapezoid rule.

        Parameters
        ----------
        values : ndarray
            Integrand values; rows are integrated independently.
        derivative : ndarray
            Derivative of the integrand at the grid points.
        h : float
            Uniform grid spacing.

        Returns
        -------
        ndarray
            Cumulative integrals with the same shape as ``values``.
        """
        integral = np.zeros_like(values)
        integral[:, 1:] = np.cumsum(0.5 * h * (values[:, :-1] + values[:, 1:]), axis=1)
        # Euler-Maclaurin: int = T - h^2/12 (f'(b) - f'(a)) + O(h^4).
        integral[:, 1:] -= h * h / 12.0 * (derivative[:, 1:] - derivative[:, 0:1])
        return integral

    def C(self, m: int, t: NDArrayFloat) -> NDArrayFloat:
        """Evaluate the odd antiderivative for nonnegative or negative arguments.

        Parameters
        ----------
        m : int
            Antiderivative index.
        t : ndarray
            Nonnegative integration limits or evaluation points.

        Returns
        -------
        ndarray
            Antiderivative values.
        """
        if m == 0:
            return np.arctan(np.asarray(t, dtype=np.float64))
        t = np.asarray(t, dtype=np.float64)
        out = np.empty_like(t)
        sign = np.sign(t)
        magnitude = np.abs(t)
        inside = magnitude <= self.t[-1]
        if np.any(inside):
            out[inside] = sign[inside] * np.interp(
                magnitude[inside], self.t, self.tables[m]
            )
        outside = ~inside
        if np.any(outside):
            beta = m * self.beta
            tail = _lorentzian_tail(beta, magnitude[outside])
            out[outside] = sign[outside] * (0.5 * math.pi * math.exp(-beta) - tail)
        return out


@functools.lru_cache(maxsize=8)
def _gauss_legendre_nodes(n_points: int) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Return cached Gauss-Legendre nodes and weights on ``[-1, 1]``.

    Parameters
    ----------
    n_points : int
        Number of nodes.

    Returns
    -------
    tuple of ndarray
        Nodes and weights on ``[-1, 1]``.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_points)  # type: ignore[no-untyped-call]
    return nodes, weights


def _visibility_quadrature_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    n_frequency: int,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Build quadrature nodes for the SPM three-wave region.

    The input offsets and generated idler remain within the channel band. The
    inner integral is analytical; a cosine frequency grid improves convergence at
    the band edges.

    Parameters
    ----------
    half_band_hz : float
        Half the channel bandwidth (Hz).
    rate : float
        Mismatch rate per frequency product, ``4*pi**2*abs(beta2)``.
    alpha_npm_m : float
        Fiber power attenuation (Np/m).
    n_frequency : int
        Number of observed frequencies.
    n_quadrature : int
        Gauss-Legendre nodes per subinterval.

    Returns
    -------
    tuple of ndarray
        Lower and upper reduced-variable limits and quadrature measures.
    """
    if n_frequency > 1:
        index = np.arange(n_frequency, dtype=np.float64)
        frequencies = -half_band_hz * np.cos(np.pi * index / (n_frequency - 1))
        # Composite trapezoid weights on the nonuniform frequency grid.
        frequency_weights = np.empty(n_frequency, dtype=np.float64)
        frequency_weights[0] = 0.5 * (frequencies[1] - frequencies[0])
        frequency_weights[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
        frequency_weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
    else:
        frequencies = np.zeros(1, dtype=np.float64)
        frequency_weights = np.ones(1, dtype=np.float64)
    nodes, weights = _gauss_legendre_nodes(n_quadrature)
    lo_parts: list[NDArrayFloat] = []
    hi_parts: list[NDArrayFloat] = []
    measure_parts: list[NDArrayFloat] = []
    for frequency, frequency_weight in zip(frequencies, frequency_weights, strict=True):
        low = -half_band_hz - float(frequency)
        high = half_band_hz - float(frequency)
        for start, stop in ((low, 0.0), (0.0, high)):
            if stop - start <= 0.0:
                continue
            u = 0.5 * (stop - start) * (nodes + 1.0) + start
            weight = 0.5 * (stop - start) * weights
            # The second offset is constrained by its band and the idler.
            v_low = np.maximum(low, low - u)
            v_high = np.minimum(high, high - u)
            valid = v_high > v_low
            if not np.any(valid):
                continue
            u = u[valid]
            weight = weight[valid]
            v_low = v_low[valid]
            v_high = v_high[valid]
            # dv = dt / (rate |u| alpha), where t = |rho| / alpha.
            measure = weight * frequency_weight / (rate * np.abs(u) * alpha_npm_m)
            t_low = rate * u * v_low / alpha_npm_m
            t_high = rate * u * v_high / alpha_npm_m
            lo_parts.append(np.minimum(t_low, t_high))
            hi_parts.append(np.maximum(t_low, t_high))
            measure_parts.append(measure)
    return (
        np.concatenate(lo_parts),
        np.concatenate(hi_parts),
        np.concatenate(measure_parts),
    )


def _gn_weighted_spm_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int = VISIBILITY_FREQUENCY_POINTS,
    n_quadrature: int = VISIBILITY_QUADRATURE_POINTS,
) -> NDArrayFloat:
    """Compute GN-weighted SPM visibility for one bandwidth.

    The weight is the closed-form GN kernel and the phase difference is
    ``rho*d``. Substituting ``t = rho/alpha`` gives an analytical inner integral
    and leaves a one-dimensional quadrature.

    Parameters
    ----------
    beta2_abs_s2_m : float
        Magnitude of second-order dispersion (s²/m).
    alpha_npm_m : float
        Fiber power attenuation (Np/m).
    bandwidth_hz : float
        Channel bandwidth or per-channel bandwidths (Hz).
    span_length_m : float
        Single-span length (m).
    distances_m : ndarray
        Separation of the noise copies (m).
    n_frequency : int
        Number of observed frequencies.
    n_quadrature : int
        Gauss-Legendre nodes per subinterval.

    Returns
    -------
    ndarray
        Visibility at each separation, in ``[-1, 1]``.
    """
    distances = np.asarray(distances_m, dtype=np.float64)
    half_band = 0.5 * bandwidth_hz
    # Mismatch rate per frequency product: rho = rate * u * v.
    rate = 4.0 * np.pi**2 * beta2_abs_s2_m
    exp_alpha_l = float(np.exp(-alpha_npm_m * span_length_m))
    t_lo, t_hi, measure = _visibility_quadrature_nodes(
        half_band,
        rate,
        alpha_npm_m,
        n_frequency,
        n_quadrature,
    )
    tables = _LorentzianAntiderivative(
        alpha_npm_m,
        span_length_m,
        int(distances.size) + 1,
        float(np.max(np.abs(t_hi))),
    )
    # Reuse C_m across distances, reducing interpolations from 6K to 2(K+2).
    m_max = int(distances.size) + 1
    c_hi = [tables.C(m, t_hi) for m in range(m_max + 1)]
    c_lo = [tables.C(m, t_lo) for m in range(m_max + 1)]
    # The denominator is the NLI power: the weight without the cosine.
    weight_hi = (1.0 + exp_alpha_l**2) * c_hi[0] - 2.0 * exp_alpha_l * c_hi[1]
    weight_lo = (1.0 + exp_alpha_l**2) * c_lo[0] - 2.0 * exp_alpha_l * c_lo[1]
    total = float(np.sum(measure * (weight_hi - weight_lo)))
    visibility = np.zeros_like(distances)
    for index in range(distances.size):
        k = index + 1
        cosine_hi = (1.0 + exp_alpha_l**2) * c_hi[k] - exp_alpha_l * (
            c_hi[abs(k - 1)] + c_hi[k + 1]
        )
        cosine_lo = (1.0 + exp_alpha_l**2) * c_lo[k] - exp_alpha_l * (
            c_lo[abs(k - 1)] + c_lo[k + 1]
        )
        visibility[index] = float(np.sum(measure * (cosine_hi - cosine_lo))) / total
    return visibility


def _self_beating_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: NDArrayFloat,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int = VISIBILITY_FREQUENCY_POINTS,
    n_quadrature: int = VISIBILITY_QUADRATURE_POINTS,
) -> NDArrayFloat:
    """Compute NLI-weighted SPM visibility, sharing results for equal bandwidths.

    Parameters
    ----------
    beta2_abs_s2_m : float
        Magnitude of second-order dispersion (s²/m).
    alpha_npm_m : float
        Fiber power attenuation (Np/m).
    bandwidth_hz : ndarray
        Channel bandwidth or per-channel bandwidths (Hz).
    span_length_m : float
        Single-span length (m).
    distances_m : ndarray
        Separation of the noise copies (m).
    n_frequency : int
        Number of observed frequencies.
    n_quadrature : int
        Gauss-Legendre nodes per subinterval.

    Returns
    -------
    ndarray
        Visibility by separation and channel, in ``[-1, 1]``.
    """
    bandwidths = np.asarray(bandwidth_hz, dtype=np.float64)
    distances = np.asarray(distances_m, dtype=np.float64)
    visibility = np.ones((distances.size, bandwidths.size), dtype=np.float64)
    if distances.size == 0 or bandwidths.size == 0:
        return visibility
    if beta2_abs_s2_m == 0.0 or alpha_npm_m <= 0.0:
        # Without dispersion, the copies are fully coherent.
        return visibility
    unique, inverse = np.unique(bandwidths, return_inverse=True)
    for index, bandwidth in enumerate(unique):
        column = _gn_weighted_spm_visibility(
            beta2_abs_s2_m,
            alpha_npm_m,
            float(bandwidth),
            span_length_m,
            distances,
            n_frequency,
            n_quadrature,
        )
        visibility[:, inverse == index] = column[:, None]
    return visibility


def _cross_beating_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
    partner_offset_hz: NDArrayFloat,
) -> NDArrayFloat:
    """Compute uniform-band XPM visibility using the inter-channel offset.

    The offset dominates the phase mismatch, making most copies nearly
    incoherent; the band average is a sinc function.

    Parameters
    ----------
    beta2_abs_s2_m : float
        Magnitude of second-order dispersion (s²/m).
    bandwidth_hz : ndarray
        Channel bandwidth or per-channel bandwidths (Hz).
    distance_m : float
        Separation of the noise copies (m).
    partner_offset_hz : ndarray
        Frequency offset to the interfering channel (Hz).

    Returns
    -------
    ndarray
        Visibility by channel, in ``[-1, 1]``.
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
    """Calculate the channel-wise cross-span coherent factor.

    Parameters
    ----------
    n_spans : int
        Number of spans.
    visibility : ndarray
        Visibility at each span separation and channel.

    Returns
    -------
    ndarray
        Per-channel factor: ``N**2`` for full coherence and ``N`` for incoherence.
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return cast("NDArrayFloat", n + 2.0 * (weights * visibility).sum(axis=0))


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Invert the coherent factor to an equivalent phase increment.

    For equal-amplitude spans the Dirichlet factor decreases monotonically from
    ``N**2`` to zero on ``(0, 2*pi/N)``, allowing bisection.

    Parameters
    ----------
    n_spans : int
        Number of spans.
    target : ndarray
        Target coherent factor per channel.

    Returns
    -------
    ndarray
        Per-channel phase increment (rad).
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


class ECZCoherentModel(MultiBandISRSGN):
    """ECZ ISRS-GN model with inter-span coherent accumulation.

    The single-span output is identical to :class:`MultiBandISRSGN`. In a
    multi-span path, complex NLI amplitudes carry relative phase into the route
    layer. The selected launch-power convention applies to both power and
    amplitude: per-polarization inputs are converted at the baseline kernel
    boundary, and ``abs(amplitude)**2`` equals the noise in the same convention.
    Visibility and phase caches depend on fiber, span geometry and bandwidth, not
    launch power. Fiber-specific child evaluators retain the chosen convention.

    Attributes
    ----------
    supports_span_context : bool
        Whether the route supplies span position for phase calculation.
    """

    #: Span position determines the relative phase of NLI amplitudes.
    supports_span_context = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionGeometry] | None = None,
        ode_accel: str = "matvec",
        slots_per_channel: int | None = None,
        channel_bandwidth_source: str = "symbol_rate",
        power_width_mode: str = "fixed_power",
        launch_power_convention: str | None = None,
    ) -> None:
        """Initialize the coherent evaluator and its phase caches.

        Parameters
        ----------
        grid : SpectrumGrid | FlexibleGrid
            Legacy or flexible spectrum grid.
        ref_fiber_config : FiberSpanConfig
            Reference fiber configuration.
        symbol_rate_hz : float
            Symbol rate (Hz).
        connections : Sequence[ConnectionSpec | CarrierGeometry] | None
            Flexible-grid connections.
        ode_accel : str
            Single-span acceleration mode.
        slots_per_channel : int | None
            Slots per legacy channel.
        channel_bandwidth_source : str
            Source of the per-channel bandwidth.
        power_width_mode : str
            Power normalization convention.
        launch_power_convention : str | None
            Power convention for inputs, noise, and amplitudes; ``None`` inherits the default.

        Raises
        ------
        ValueError
            If a configuration convention is invalid.
        """
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz,
            connections=connections,
            ode_accel=ode_accel,
            slots_per_channel=slots_per_channel,
            channel_bandwidth_source=channel_bandwidth_source,
            power_width_mode=power_width_mode,
            launch_power_convention=launch_power_convention,
        )
        # Preserve the original option so child evaluators inherit None semantics.
        self._launch_power_convention_option = launch_power_convention
        # Coherence depends on span geometry and fiber, not launch power.
        self._coherence_cache: dict[tuple[int, float], tuple[NDArrayFloat, ...]] = {}
        # Cache phase factors by span count, span length, and normalized distance.
        self._phase_cache: dict[
            tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]
        ] = {}

    # --- Fiber-specific evaluators -----------------------------------------
    def for_fiber(self, fiber_config: FiberSpanConfig) -> ECZCoherentModel:
        """Return the cached evaluator for the fiber, preserving its power convention.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber configuration.

        Returns
        -------
        ECZCoherentModel
            Cached evaluator for the fiber.
        """
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = ECZCoherentModel(
                self.grid,
                fiber_config,
                self.symbol_rate_hz,
                connections=self.connections,
                ode_accel=self.ode_accel,
                slots_per_channel=self.slots_per_channel,
                channel_bandwidth_source=self.channel_bandwidth_source,
                power_width_mode=self.power_width_mode,
                launch_power_convention=self._launch_power_convention_option,
            )
        return self._fiber_evaluator_cache[key]  # type: ignore[return-value]

    # --- Inter-span coherence ----------------------------------------------
    def _coherence(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Calculate cached phase ramps and coherent factors for a span geometry.

        Parameters
        ----------
        n_spans : int
            Number of spans.
        span_length_m : float
            Single-span length (m).

        Returns
        -------
        tuple of ndarray
            SPM and XPM phase increments (rad), then their dimensionless coherent factors.
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._coherence_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        n = max(1, int(n_spans))
        distances = np.arange(1, n, dtype=np.float64) * float(span_length_m)
        beta2_abs = abs(float(self.beta2_s2_m))
        # Use mean attenuation for visibility, as in the numerical reference.
        alpha_visibility = float(np.mean(self.alpha_power_npm_channels))
        bandwidth = self.channel_bandwidth_hz_channels
        if n > 1:
            visibility_spm = _self_beating_visibility(
                beta2_abs,
                alpha_visibility,
                bandwidth,
                float(span_length_m),
                distances,
            )
            visibility_xpm = np.stack(
                [
                    _cross_beating_visibility(
                        beta2_abs,
                        bandwidth,
                        float(distance),
                        self._partner_offset_hz(),
                    )
                    for distance in distances
                ]
            )
        else:
            visibility_spm = np.zeros((0, bandwidth.size), dtype=np.float64)
            visibility_xpm = np.zeros((0, bandwidth.size), dtype=np.float64)
        factor_spm = _coherent_factor(n, visibility_spm)
        factor_xpm = _coherent_factor(n, visibility_xpm)
        ramp_spm = _equivalent_phase_ramp(n, factor_spm)
        ramp_xpm = _equivalent_phase_ramp(n, factor_xpm)
        result = (ramp_spm, ramp_xpm, factor_spm, factor_xpm)
        self._coherence_cache[key] = result
        return result

    def _partner_offset_hz(self) -> NDArrayFloat:
        """Return the frequency offset to the nearest interfering channel (Hz).

        Returns
        -------
        ndarray
            Nearest-neighbor frequency offsets (Hz).
        """
        if self.f_abs_hz.size > 1:
            offsets = np.abs(self.f_abs_hz[:, None] - self.f_abs_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            return cast("NDArrayFloat", offsets.min(axis=1))
        return np.zeros(self.f_abs_hz.size, dtype=np.float64)

    def _span_phase_factors(
        self,
        span_context: SpanContext,
        span_length_m: float,
    ) -> tuple[NDArrayComplex, NDArrayComplex]:
        """Return the unit complex phase factors for this span.

        Phase grows with cumulative propagation distance; for equal spans this is
        proportional to the span index.

        Parameters
        ----------
        span_context : SpanContext
            Position of the span in the path.
        span_length_m : float
            Single-span length (m).

        Returns
        -------
        tuple of ndarray
            Unit-magnitude complex phase factors for SPM and XPM.
        """
        scale = span_context.cumulative_length_m / float(span_length_m)
        key = (int(span_context.total_spans), float(span_length_m), float(scale))
        cached = self._phase_cache.get(key)
        if cached is None:
            ramp_spm, ramp_xpm, _, _ = self._coherence(
                span_context.total_spans, span_length_m
            )
            cached = (np.exp(1j * ramp_spm * scale), np.exp(1j * ramp_xpm * scale))
            self._phase_cache[key] = cached
        return cached

    # --- Single-span physics -----------------------------------------------
    def span_fiber_output(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Evaluate the fiber and attach complex NLI amplitudes for multiple spans.

        Without a multi-span context the output matches the baseline single-span
        power-only result.

        Parameters
        ----------
        power_in_w : ndarray
            Channel launch powers (W).
        span_length_m : float
            Single-span length (m).
        n_effective_spans : int
            Effective span count; the route performs coherent accumulation.
        span_context : SpanContext | None
            Position of the span in the path.

        Returns
        -------
        SpanFiberOutput
            Fiber output powers and, for multiple spans, complex NLI amplitudes.
        """
        output = super().span_fiber_output(power_in_w, span_length_m, n_effective_spans)
        if span_context is None or span_context.total_spans <= 1:
            return output
        factor_spm, factor_xpm = self._span_phase_factors(
            span_context, float(span_length_m)
        )
        amplitude_spm = np.sqrt(output.spm_noise_w) * factor_spm
        amplitude_xpm = np.sqrt(output.xpm_noise_w) * factor_xpm
        return SpanFiberOutput(
            output.power_out_w,
            output.spm_noise_w,
            output.xpm_noise_w,
            amplitude_spm,
            amplitude_xpm,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """Evaluate fiber and amplifier for one span; ROADMs are handled by the route.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber configuration.
        edfa_config : EDFAConfig | None
            Span amplifier configuration.
        power_in_w : ndarray
            Channel launch powers (W).
        span_length_m : float
            Single-span length (m).
        span_context : SpanContext | None
            Position of the span in the path.

        Returns
        -------
        ElementOutput
            Output powers and, for multiple spans, complex NLI amplitudes.
        """
        evaluator = self.for_fiber(fiber_config)
        return evaluator._calc_span_output_with_context(
            power_in_w,
            span_length_m,
            edfa_config,
            span_context,
        )

    def _calc_span_output_with_context(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        span_context: SpanContext | None,
    ) -> ElementOutput:
        """Evaluate one span using this fiber-specific evaluator.

        This is also used by link-noise assembly to avoid rebuilding the fiber key.

        Parameters
        ----------
        power_in_w : ndarray
            Channel launch powers (W).
        span_length_m : float
            Single-span length (m).
        edfa_config : EDFAConfig | None
            Span amplifier configuration.
        span_context : SpanContext | None
            Position of the span in the path.

        Returns
        -------
        ElementOutput
            Output powers and, for multiple spans, complex NLI amplitudes.
        """
        fiber_output = self.span_fiber_output(
            power_in_w, span_length_m, 1, span_context
        )
        return apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=self._resolve_nf(edfa_config),
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            # ASE must use the fiber output convention; the gain ratio is invariant.
            ase_polarization_scale=self.ase_polarization_scale,
        )


# Register the model for construction through create_model.
register_model(ECZ_COHERENT_MODEL_INFO, ECZCoherentModel)
