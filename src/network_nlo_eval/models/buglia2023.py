"""Buglia 2023 closed-form ISRS-GN model for arbitrary loss and fiber length.

The first-order matched link function in Eq. (30) extends the Semrau 2019
approximation to short spans and low attenuation. Equations (13)-(18) define
the profile and single-span XPM and SPM terms.

The implementation uses fiber attenuation for the profile parameters rather
than fitting them per link, and uses the ECZ triangular Raman approximation
for output power. It implements GN only: the paper's modulation-format
corrections are omitted. The route layer evaluates individual spans with
``n_effective_spans=1``, so the kernel's n**epsilon factor does not build up
across the route.

Cross-span coherence uses GN noise-weighted visibility for SPM and XPM,
with the nearest neighbor representing XPM. An equivalent per-channel phase
ramp reproduces the calculated total for identical spans but cannot preserve
all pairwise phase structure; unequal spans use the current span length to
scale the phase. Zero-dispersion visibility tends to one.

The paper specifies a loss-side applicability condition but no dispersion
limit. In a 57-channel, 20-span comparison against this repository's
numerical reference, mean SNR error grew to +5.01 dB at D=0.5 ps/(nm km).
These measurements suggest avoiding low-dispersion fibers (approximately
|D| < 5 ps/(nm km)); the cause has not been established.

References
----------
H. Buglia et al., "A Closed-Form Expression for the Gaussian Noise Model in
the Presence of Inter-Channel Stimulated Raman Scattering Extended for
Arbitrary Loss and Fibre Length", J. Lightwave Technol. 41, 3577-3588 (2023),
doi:10.1109/JLT.2023.3256185."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from network_nlo_eval.core.constants import PI
from network_nlo_eval.models.base import (
    ElementOutput,
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
    _calc_gamma_coeff_scalar,
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
)
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)

if TYPE_CHECKING:
    from network_nlo_eval.core.flexgrid import FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "BUGLIA2023_MODEL_INFO",
    "Buglia2023Model",
    "Buglia2023SpanKernel",
    "alpha_tilde_kappa",
]

#:
_SRS_CR = 0.028 / 1e3 / 1e12

_COHERENCE_EPSILON = 0.05

_EPS_PHI = 1e-35

_DELTA_F_CO_HZ = 15e12

BUGLIA2023_MODEL_INFO = ModelInfo(
    name="buglia2023",
    summary="Buglia 2023 closed-form ISRS-GN for arbitrary loss and fibre length.",
    family="gn",
    reference="Buglia et al., 'A Closed-Form Expression for the Gaussian Noise Model in the "
    "Presence of Inter-Channel Stimulated Raman Scattering Extended for Arbitrary Loss and "
    "Fibre Length', JLT 2023, doi:10.1109/JLT.2023.3256185",
    options=(("egn", "Not supported: the paper derives the GN model only."),),
)


def alpha_tilde_kappa(alpha_l: float, length_m: float) -> tuple[float, float]:
    """Return effective attenuation and amplitude from Eq. (15)-(16).

    First-order Taylor matching gives the finite-length link approximation. At
    large alpha*L the coefficients approach alpha and 1; at small alpha*L they
    approach 2/L and 2. A series avoids cancellation for small arguments.

    Parameters
    ----------
    alpha_l : float
        Attenuation parameter in Np/m.
    length_m : float
        Span length in meters.

    Returns
    -------
    tuple of float
        Effective attenuation in Np/m and dimensionless amplitude coefficient."""
    x = alpha_l * length_m
    if x < 1e-3:
        em1 = x * (1.0 - x / 2.0 + x * x / 6.0 - x * x * x / 24.0)
        denom = x * x * (0.5 - x / 3.0 + x * x / 8.0 - x * x * x / 30.0)
    else:
        exp_neg = np.exp(-x)
        em1 = 1.0 - exp_neg
        denom = 1.0 - (1.0 + x) * exp_neg
    tilde_alpha = alpha_l * em1 / denom
    kappa = em1 * em1 / denom
    return float(tilde_alpha), float(kappa)


# =============================================================================
#
# =============================================================================

#:
_VIS_FREQUENCY_POINTS = 41

_VIS_QUADRATURE_POINTS = 200

_VIS_TABLE_STEP = 1.0e-3

_VIS_TABLE_LIMIT = 30.0

_VIS_TABLE_MINIMUM = 1.0

_VIS_NEGLIGIBLE_T = 1.0e-6

_RAMP_BISECTION_STEPS = 60


def _lorentzian_tail(beta: float, t: NDArrayFloat) -> NDArrayFloat:
    """Compute the asymptotic Lorentzian cosine integral tail.

    The expansion uses repeated integration by parts and is evaluated beyond the
    lookup table. ``beta`` is the dimensionless product m*alpha*L."""
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
        (
            -f * sin_beta / beta
            - df * cos_beta / beta**2
            + d2f * sin_beta / beta**3
            + d3f * cos_beta / beta**4
            - d4f * sin_beta / beta**5
        ),
    )


class _VisibilityAntiderivative:
    """Tabulate the Lorentzian cosine antiderivative for visibility calculations.

    A corrected trapezoidal rule provides fourth-order accuracy. The m=0 case
    uses arctan; outside the table an asymptotic tail and the Fourier-transform
    limit are used.

    Parameters
    ----------
    alpha_npm_m : float
        Fiber power attenuation in Np/m.
    span_length_m : float
        Span length in meters.
    m_max : int
        Largest tabulated separation index.
    t_end : float
        Required maximum integration coordinate.
    step : float, optional
        Table spacing."""

    def __init__(
        self,
        alpha_npm_m: float,
        span_length_m: float,
        m_max: int,
        t_end: float,
        step: float = _VIS_TABLE_STEP,
    ) -> None:
        self.alpha = float(alpha_npm_m)
        self.span_length = float(span_length_m)
        self.beta = self.alpha * self.span_length
        limit = min(max(float(t_end), _VIS_TABLE_MINIMUM), _VIS_TABLE_LIMIT)
        n_points = max(3, math.ceil(limit / step) + 1)
        if n_points % 2 == 0:
            n_points += 1
        self.t = np.linspace(0.0, limit, n_points)
        h = float(self.t[1] - self.t[0])
        self._tables: dict[int, NDArrayFloat] = {}
        for m in range(1, int(m_max) + 1):
            phase = m * self.beta * self.t
            cos_phase = np.cos(phase)
            values = cos_phase / (1.0 + self.t * self.t)
            derivative = (
                -m * self.beta * np.sin(phase) / (1.0 + self.t * self.t)
                - 2.0 * self.t * cos_phase / (1.0 + self.t * self.t) ** 2
            )
            self._tables[m] = self._cumulative(values, derivative, h)

    @staticmethod
    def _cumulative(
        values: NDArrayFloat, derivative: NDArrayFloat, h: float
    ) -> NDArrayFloat:
        """Return cumulative integrals using the corrected trapezoidal rule."""
        integral = np.zeros_like(values)
        integral[1:] = np.cumsum(0.5 * h * (values[:-1] + values[1:]))
        integral[1:] -= h * h / 12.0 * (derivative[1:] - derivative[0])
        return integral

    def C(self, m: int, t: NDArrayFloat) -> NDArrayFloat:
        """Evaluate the odd antiderivative C_m at positive or negative coordinates."""
        if m == 0:
            return np.arctan(np.asarray(t, dtype=np.float64))
        t = np.asarray(t, dtype=np.float64)
        out = np.empty_like(t)
        sign = np.sign(t)
        magnitude = np.abs(t)
        inside = magnitude <= self.t[-1]
        if np.any(inside):
            out[inside] = sign[inside] * np.interp(
                magnitude[inside], self.t, self._tables[m]
            )
        outside = ~inside
        if np.any(outside):
            beta = m * self.beta
            tail = _lorentzian_tail(beta, magnitude[outside])
            out[outside] = sign[outside] * (0.5 * math.pi * math.exp(-beta) - tail)
        return out


def _weighted_visibility(
    t_low: NDArrayFloat,
    t_high: NDArrayFloat,
    measure: NDArrayFloat,
    table: _VisibilityAntiderivative,
    exp_alpha_l: float,
    n_separations: int,
) -> NDArrayFloat:
    """Calculate GN-weighted visibility for each integer span separation.

    The Lorentzian cosine antiderivatives reduce the inner integral analytically.
    The integer separation index embodies the equal-span assumption.

    Returns
    -------
    numpy.ndarray
        Visibility for each span separation."""
    weight_high = (1.0 + exp_alpha_l**2) * table.C(
        0, t_high
    ) - 2.0 * exp_alpha_l * table.C(1, t_high)
    weight_low = (1.0 + exp_alpha_l**2) * table.C(
        0, t_low
    ) - 2.0 * exp_alpha_l * table.C(1, t_low)
    total = float(np.sum(measure * (weight_high - weight_low)))
    if not total > 0.0:
        return np.ones(n_separations, dtype=np.float64)
    out = np.empty(n_separations, dtype=np.float64)
    for index in range(n_separations):
        m = index + 1
        cosine_high = (1.0 + exp_alpha_l**2) * table.C(m, t_high) - exp_alpha_l * (
            table.C(abs(m - 1), t_high) + table.C(m + 1, t_high)
        )
        cosine_low = (1.0 + exp_alpha_l**2) * table.C(m, t_low) - exp_alpha_l * (
            table.C(abs(m - 1), t_low) + table.C(m + 1, t_low)
        )
        out[index] = float(np.sum(measure * (cosine_high - cosine_low))) / total
    return out


def _spm_region_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    n_frequency: int,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Build quadrature nodes for the SPM three-wave integration region.

    For fixed observed frequency and first offset, the second offset is
    integrated analytically via the Lorentzian antiderivative.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper transformed bounds and quadrature measures."""
    frequencies, frequency_weights = _cosine_frequency_grid(half_band_hz, n_frequency)
    nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)  # type: ignore[no-untyped-call]
    low_parts: list[NDArrayFloat] = []
    high_parts: list[NDArrayFloat] = []
    measure_parts: list[NDArrayFloat] = []
    for frequency, frequency_weight in zip(frequencies, frequency_weights, strict=True):
        low = -half_band_hz - float(frequency)
        high = half_band_hz - float(frequency)
        for start, stop in ((low, 0.0), (0.0, high)):
            if stop - start <= 0.0:
                continue
            u = 0.5 * (stop - start) * (nodes + 1.0) + start
            weight = 0.5 * (stop - start) * weights
            v_low = np.maximum(low, low - u)
            v_high = np.minimum(high, high - u)
            valid = v_high > v_low
            if not np.any(valid):
                continue
            u = u[valid]
            weight = weight[valid]
            v_low = v_low[valid]
            v_high = v_high[valid]
            measure = weight * frequency_weight / (rate * np.abs(u) * alpha_npm_m)
            t_low = rate * u * v_low / alpha_npm_m
            t_high = rate * u * v_high / alpha_npm_m
            low_parts.append(np.minimum(t_low, t_high))
            high_parts.append(np.maximum(t_low, t_high))
            measure_parts.append(measure)
    if not low_parts:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    return (
        np.concatenate(low_parts),
        np.concatenate(high_parts),
        np.concatenate(measure_parts),
    )


def _xpm_region_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    channel_offset_hz: float,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Build quadrature nodes for the XPM three-wave integration region.

    The mismatch is 4*pi²*|beta2|*x*(offset+y); the inner y integral uses the
    Lorentzian antiderivative.

    Returns
    -------
    tuple of numpy.ndarray
        Lower and upper transformed bounds and quadrature measures."""
    nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)  # type: ignore[no-untyped-call]
    low_parts: list[NDArrayFloat] = []
    high_parts: list[NDArrayFloat] = []
    measure_parts: list[NDArrayFloat] = []
    for start, stop in ((-half_band_hz, 0.0), (0.0, half_band_hz)):
        if stop - start <= 0.0:
            continue
        x = 0.5 * (stop - start) * (nodes + 1.0) + start
        weight = 0.5 * (stop - start) * weights
        y_low = np.maximum(-half_band_hz, -half_band_hz - x)
        y_high = np.minimum(half_band_hz, half_band_hz - x)
        valid = y_high > y_low
        if not np.any(valid):
            continue
        x = x[valid]
        weight = weight[valid]
        y_low = y_low[valid]
        y_high = y_high[valid]
        measure = weight / (rate * np.abs(x) * alpha_npm_m)
        t_low = rate * x * (channel_offset_hz + y_low) / alpha_npm_m
        t_high = rate * x * (channel_offset_hz + y_high) / alpha_npm_m
        low_parts.append(np.minimum(t_low, t_high))
        high_parts.append(np.maximum(t_low, t_high))
        measure_parts.append(measure)
    if not low_parts:
        empty = np.zeros(0, dtype=np.float64)
        return empty, empty, empty
    return (
        np.concatenate(low_parts),
        np.concatenate(high_parts),
        np.concatenate(measure_parts),
    )


def _cosine_frequency_grid(
    half_band_hz: float, n_frequency: int
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Return cosine-spaced observation frequencies and trapezoidal weights."""
    if n_frequency > 1:
        index = np.arange(n_frequency, dtype=np.float64)
        frequencies = -half_band_hz * np.cos(np.pi * index / (n_frequency - 1))
        weights = np.empty(n_frequency, dtype=np.float64)
        weights[0] = 0.5 * (frequencies[1] - frequencies[0])
        weights[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
        weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
    else:
        frequencies = np.zeros(1, dtype=np.float64)
        weights = np.ones(1, dtype=np.float64)
    return frequencies, weights


def _gn_weighted_spm_visibility(
    *,
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    span_separations_m: NDArrayFloat,
) -> NDArrayFloat:
    """Compute cross-span SPM visibility weighted by GN noise power.

    The weight is ``|1-exp(-(alpha+1j*rho)*L)|²/(alpha²+rho²)``; separations
    must be integer multiples of the span length. Zero dispersion gives unity.

    Returns
    -------
    numpy.ndarray
        Visibility for each separation."""
    separations = np.asarray(span_separations_m, dtype=np.float64)
    if separations.size == 0:
        return np.zeros(0, dtype=np.float64)
    if beta2_abs_s2_m == 0.0 or alpha_npm_m <= 0.0 or bandwidth_hz <= 0.0:
        return np.ones(separations.size, dtype=np.float64)
    rate = 4.0 * PI**2 * abs(float(beta2_abs_s2_m))
    t_low, t_high, measure = _spm_region_nodes(
        0.5 * float(bandwidth_hz),
        rate,
        float(alpha_npm_m),
        _VIS_FREQUENCY_POINTS,
        _VIS_QUADRATURE_POINTS,
    )
    if t_high.size == 0 or float(np.max(np.abs(t_high))) < _VIS_NEGLIGIBLE_T:
        return np.ones(separations.size, dtype=np.float64)
    table = _VisibilityAntiderivative(
        float(alpha_npm_m),
        float(span_length_m),
        int(separations.size) + 1,
        float(np.max(np.abs(t_high))),
    )
    exp_alpha_l = float(np.exp(-float(alpha_npm_m) * float(span_length_m)))
    return _weighted_visibility(
        t_low, t_high, measure, table, exp_alpha_l, int(separations.size)
    )


def _gn_weighted_xpm_visibility(
    *,
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    span_separations_m: NDArrayFloat,
    channel_offset_hz: float,
) -> NDArrayFloat:
    """Compute GN-weighted XPM visibility using a representative channel offset.

    The same noise weighting as SPM is used with the XPM mismatch variable.
    Separations are assumed to be integer multiples of the span length.

    Returns
    -------
    numpy.ndarray
        Visibility for each separation."""
    separations = np.asarray(span_separations_m, dtype=np.float64)
    if separations.size == 0:
        return np.zeros(0, dtype=np.float64)
    offset = abs(float(channel_offset_hz))
    if (
        beta2_abs_s2_m == 0.0
        or alpha_npm_m <= 0.0
        or bandwidth_hz <= 0.0
        or offset == 0.0
    ):
        return np.ones(separations.size, dtype=np.float64)
    rate = 4.0 * PI**2 * abs(float(beta2_abs_s2_m))
    t_low, t_high, measure = _xpm_region_nodes(
        0.5 * float(bandwidth_hz),
        rate,
        float(alpha_npm_m),
        offset,
        _VIS_QUADRATURE_POINTS,
    )
    if t_high.size == 0 or float(np.max(np.abs(t_high))) < _VIS_NEGLIGIBLE_T:
        return np.ones(separations.size, dtype=np.float64)
    table = _VisibilityAntiderivative(
        float(alpha_npm_m),
        float(span_length_m),
        int(separations.size) + 1,
        float(np.max(np.abs(t_high))),
    )
    exp_alpha_l = float(np.exp(-float(alpha_npm_m) * float(span_length_m)))
    return _weighted_visibility(
        t_low, t_high, measure, table, exp_alpha_l, int(separations.size)
    )


def _coherent_factor(n_spans: int, visibility: NDArrayFloat) -> NDArrayFloat:
    """Return N + 2*sum((N-k)*visibility[k]) for each channel."""
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return cast("NDArrayFloat", n + 2.0 * (weights * visibility).sum(axis=0))


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Invert the Dirichlet factor to obtain a per-channel phase step.

    The scalar phase ramp matches the requested total factor and compresses
    pairwise phase structure."""
    n = int(n_spans)
    if n <= 1:
        return np.zeros(np.shape(target), dtype=np.float64)
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n)
    for _ in range(_RAMP_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        factor = (np.sin(0.5 * n * middle) / np.sin(0.5 * middle)) ** 2
        too_large = factor > goal
        low = np.where(too_large, middle, low)
        high = np.where(too_large, high, middle)
    return 0.5 * (low + high)


class Buglia2023SpanKernel:
    """Single-span Buglia 2023 kernel with GN-weighted coherence phases.

    Output power uses the ECZ Raman profile; NLI uses Eq. (17)-(18). Amplifiers
    and accumulation are handled by the route model."""

    def __init__(
        self,
        *,
        f_rel_hz: NDArrayFloat,
        lambdas_m: NDArrayFloat,
        a_eff_m2: NDArrayFloat,
        alpha_npm: NDArrayFloat,
        beta2_s2_m: float,
        beta3_s3_m: float,
        n2_m2_w: float,
        b_hz_channels: NDArrayFloat,
        c_r_w_m_hz: float = _SRS_CR,
        symbol_rate_hz: float = 50e9,
    ) -> None:
        """Initialize frequency, fiber, and channel properties.

        Parameters
        ----------
        f_rel_hz, lambdas_m : numpy.ndarray
            Relative channel frequencies in Hz and wavelengths in meters.
        a_eff_m2, alpha_npm : numpy.ndarray
            Effective areas in m² and power attenuation in Np/m.
        beta2_s2_m, beta3_s3_m : float
            Second- and third-order dispersion coefficients.
        n2_m2_w : float
            Nonlinear refractive index in m²/W.
        b_hz_channels : numpy.ndarray
            Channel bandwidths in Hz.
        c_r_w_m_hz : float, optional
            Raman gain slope in 1/(W m Hz).
        symbol_rate_hz : float, optional
            Symbol rate used for SRS band edges."""
        self.f_rel_hz = np.asarray(f_rel_hz, dtype=np.float64)
        self.lambdas_m = np.asarray(lambdas_m, dtype=np.float64)
        self.a_eff_m2 = np.asarray(a_eff_m2, dtype=np.float64)
        self.alpha_npm = np.asarray(alpha_npm, dtype=np.float64)
        self.beta2_s2_m = float(beta2_s2_m)
        self.beta3_s3_m = float(beta3_s3_m)
        self.n2_m2_w = float(n2_m2_w)
        self.b_hz_channels = np.asarray(b_hz_channels, dtype=np.float64)
        self.c_r_w_m_hz = float(c_r_w_m_hz)
        self.symbol_rate_hz = float(symbol_rate_hz)

        self.n_channels = int(self.f_rel_hz.shape[0])

        self.phi_i = (
            -4.0
            * PI**2
            * (self.beta2_s2_m + 2.0 * PI * self.beta3_s3_m * self.f_rel_hz)
        )
        self.gamma_ii = np.array(
            [
                _calc_gamma_coeff_scalar(
                    self.lambdas_m[i],
                    self.lambdas_m[i],
                    self.n2_m2_w,
                    self.a_eff_m2[i],
                    self.a_eff_m2[i],
                )
                for i in range(self.n_channels)
            ],
            dtype=np.float64,
        )

        self._pair_cache: dict[float, tuple[NDArrayFloat, ...]] = {}

        self.beta2_abs_s2_m = abs(self.beta2_s2_m)
        self.alpha_visibility_npm = float(np.mean(self.alpha_npm))
        if self.n_channels > 1:
            offsets = np.abs(self.f_rel_hz[:, None] - self.f_rel_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            self.partner_offset_hz = offsets.min(axis=1)
        else:
            self.partner_offset_hz = np.zeros(self.n_channels, dtype=np.float64)
        self._ramp_cache: dict[
            tuple[int, float], tuple[NDArrayFloat, NDArrayFloat]
        ] = {}
        self._factor_cache: dict[
            tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]
        ] = {}

    def _spm_visibility(
        self, distances_m: NDArrayFloat, span_length_m: float
    ) -> NDArrayFloat:
        """Calculate SPM visibility per separation and channel, grouped by bandwidth."""
        out = np.empty((distances_m.size, self.n_channels), dtype=np.float64)
        unique, inverse = np.unique(self.b_hz_channels, return_inverse=True)
        for index, bandwidth in enumerate(unique):
            column = _gn_weighted_spm_visibility(
                beta2_abs_s2_m=self.beta2_abs_s2_m,
                alpha_npm_m=self.alpha_visibility_npm,
                bandwidth_hz=float(bandwidth),
                span_length_m=float(span_length_m),
                span_separations_m=distances_m,
            )
            out[:, inverse == index] = column[:, None]
        return out

    def _xpm_visibility(
        self, distances_m: NDArrayFloat, span_length_m: float
    ) -> NDArrayFloat:
        """Calculate XPM visibility grouped by bandwidth and nearest-neighbor offset."""
        out = np.empty((distances_m.size, self.n_channels), dtype=np.float64)
        groups: dict[tuple[float, float], list[int]] = {}
        for channel in range(self.n_channels):
            key = (
                float(self.b_hz_channels[channel]),
                float(self.partner_offset_hz[channel]),
            )
            groups.setdefault(key, []).append(channel)
        for (bandwidth, offset), channels in groups.items():
            column = _gn_weighted_xpm_visibility(
                beta2_abs_s2_m=self.beta2_abs_s2_m,
                alpha_npm_m=self.alpha_visibility_npm,
                bandwidth_hz=bandwidth,
                span_length_m=float(span_length_m),
                span_separations_m=distances_m,
                channel_offset_hz=offset,
            )
            out[:, channels] = column[:, None]
        return out

    def _inter_span_ramps(
        self, n_spans: int, span_length_m: float
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return cached SPM and XPM equivalent phase steps in radians per span."""
        key = (int(n_spans), float(span_length_m))
        cached = self._ramp_cache.get(key)
        if cached is not None:
            return cached
        n = max(1, int(n_spans))
        distances = np.arange(1, n, dtype=np.float64) * float(span_length_m)
        if n > 1:
            visibility_spm = self._spm_visibility(distances, float(span_length_m))
            visibility_xpm = self._xpm_visibility(distances, float(span_length_m))
        else:
            visibility_spm = np.zeros((0, self.n_channels), dtype=np.float64)
            visibility_xpm = np.zeros((0, self.n_channels), dtype=np.float64)
        ramp_spm = _equivalent_phase_ramp(n, _coherent_factor(n, visibility_spm))
        ramp_xpm = _equivalent_phase_ramp(n, _coherent_factor(n, visibility_xpm))
        self._ramp_cache[key] = (ramp_spm, ramp_xpm)
        return ramp_spm, ramp_xpm

    def _span_phase_factors(
        self, span_context: SpanContext, span_length_m: float
    ) -> tuple[NDArrayComplex, NDArrayComplex]:
        """Return complex unit phase factors for the current span.

        Phase scales with cumulative length divided by the current span length."""
        scale = span_context.cumulative_length_m / float(span_length_m)
        key = (int(span_context.total_spans), float(span_length_m), float(scale))
        cached = self._factor_cache.get(key)
        if cached is None:
            ramp_spm, ramp_xpm = self._inter_span_ramps(
                span_context.total_spans, span_length_m
            )
            cached = (np.exp(1j * ramp_spm * scale), np.exp(1j * ramp_xpm * scale))
            self._factor_cache[key] = cached
        return cached

    def _profile_coefficients(self, length_m: float) -> tuple[NDArrayFloat, ...]:
        """Return and cache Eq. (15)-(17) channel and pair coefficients.

        The tuple contains pair weights ``m0, m1`` and effective attenuation and
        amplitude arrays ``a0, k0, a1, k1``."""
        cached = self._pair_cache.get(length_m)
        if cached is not None:
            return cached

        n = self.n_channels
        alpha = self.alpha_npm
        a0 = np.empty(n, dtype=np.float64)
        k0 = np.empty(n, dtype=np.float64)
        a1 = np.empty(n, dtype=np.float64)
        k1 = np.empty(n, dtype=np.float64)
        for i in range(n):
            a0[i], k0[i] = alpha_tilde_kappa(alpha[i], length_m)
            a1[i], k1[i] = alpha_tilde_kappa(2.0 * alpha[i], length_m)

        # phi_{i,k} = -4 pi^2 (f_k - f_i) [beta_2 + pi beta_3 (f_i + f_k)]
        freq = self.f_rel_hz
        delta = freq[None, :] - freq[:, None]
        total = freq[:, None] + freq[None, :]
        phi_ik = -4.0 * PI**2 * delta * (self.beta2_s2_m + PI * self.beta3_s3_m * total)
        phi_safe = np.where(
            np.abs(phi_ik) < _EPS_PHI,
            np.where(phi_ik >= 0.0, _EPS_PHI, -_EPS_PHI),
            phi_ik,
        )

        gamma_ik = (2.0 * PI / self.lambdas_m)[:, None] * (
            2.0 * self.n2_m2_w / (self.a_eff_m2[:, None] + self.a_eff_m2[None, :])
        )
        gamma_sq = gamma_ik**2

        band = self.b_hz_channels[:, None]
        atan0 = np.arctan(phi_safe * band / (2.0 * a0[None, :]))
        atan1 = np.arctan(phi_safe * band / (2.0 * a1[None, :]))
        m0 = 4.0 * gamma_sq * atan0 / phi_safe
        m1 = 4.0 * gamma_sq * atan1 / phi_safe
        np.fill_diagonal(m0, 0.0)
        np.fill_diagonal(m1, 0.0)

        result = (m0, m1, a0, k0, a1, k1)
        self._pair_cache[length_m] = result
        return result

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Calculate span output power and SPM/XPM noise.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Channel input powers in watts; zero denotes an inactive channel.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Span count for the Eq. (3) SPM coherence factor.
        span_context : SpanContext or None, optional
            Link position; multispan contexts produce complex NLI amplitudes.

        Returns
        -------
        SpanFiberOutput
            Output powers, noise, and optional fiber-output-referenced amplitudes."""
        power = np.asarray(power_in_w, dtype=np.float64)
        length_m = float(span_length_m)
        n_spans = max(int(n_effective_spans), 1)

        p_total = float(np.sum(power))

        f_m_hz = self.f_rel_hz[0] - self.symbol_rate_hz / 2.0
        f_M_hz = self.f_rel_hz[-1] + self.symbol_rate_hz / 2.0
        r_f = _calc_raman_profile_jit(
            self.f_rel_hz, f_m_hz, f_M_hz, _DELTA_F_CO_HZ, p_total
        )
        power_out = _calc_span_out_power_jit(
            power, self.f_rel_hz, self.alpha_npm, length_m, r_f, 0.0
        )

        m0, m1, a0, k0, a1, k1 = self._profile_coefficients(length_m)
        t_tilde = -p_total * self.c_r_w_m_hz * self.f_rel_hz / self.alpha_npm
        t_factor = 1.0 + t_tilde
        t_factor = np.where(np.abs(t_factor) < 1e-12, 1e-12, t_factor)
        weight = -t_tilde / t_factor

        sum_a0_a1 = a0 + a1
        c0 = k0 * k0 / (2.0 * a0) + weight * k0 * k1 / sum_a0_a1
        c1 = weight * k0 * k1 / sum_a0_a1 + weight * weight * k1 * k1 / (2.0 * a1)

        # --- Eq. (18):SPM ---
        phi_safe = np.where(
            np.abs(self.phi_i) < _EPS_PHI,
            np.where(self.phi_i >= 0.0, _EPS_PHI, -_EPS_PHI),
            self.phi_i,
        )
        band_sq = self.b_hz_channels**2
        asinh0 = np.arcsinh(3.0 * phi_safe * band_sq / (8.0 * PI * a0))
        asinh1 = np.arcsinh(3.0 * phi_safe * band_sq / (8.0 * PI * a1))
        eta_spm = (
            (16.0 / 27.0)
            * (self.gamma_ii**2 / band_sq)
            * t_factor**2
            * (4.0 * PI / phi_safe)
            * (asinh0 * c0 + asinh1 * c1)
        )
        spm_noise = eta_spm * power**3 * n_spans**_COHERENCE_EPSILON

        # --- Eq. (17):XPM ---
        u0 = (power**2 / self.b_hz_channels) * t_factor**2 * c0
        u1 = (power**2 / self.b_hz_channels) * t_factor**2 * c1
        xpm_noise = (32.0 / 27.0) * power * (m0 @ u0 + m1 @ u1)

        spm_noise = np.where(np.isfinite(spm_noise), spm_noise, 0.0)
        xpm_noise = np.where(np.isfinite(xpm_noise), xpm_noise, 0.0)
        spm_noise = np.maximum(spm_noise, 0.0)
        xpm_noise = np.maximum(xpm_noise, 0.0)

        if span_context is None or int(span_context.total_spans) <= 1:
            return SpanFiberOutput(power_out, spm_noise, xpm_noise)
        factor_spm, factor_xpm = self._span_phase_factors(span_context, length_m)
        amplitude_spm = np.sqrt(spm_noise) * factor_spm
        amplitude_xpm = np.sqrt(xpm_noise) * factor_xpm
        return SpanFiberOutput(
            power_out, spm_noise, xpm_noise, amplitude_spm, amplitude_xpm
        )


class Buglia2023Model(RouteModel):
    """Route model for the Buglia 2023 closed-form ISRS-GN approximation.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Channel grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber configuration.
    symbol_rate_hz : float, optional
        Symbol rate in Hz.
    egn : bool, optional
        Must be False; this model implements GN only.
    **route_options
        Options passed to the route model.

    Raises
    ------
    ValueError
        If EGN is requested."""

    info = BUGLIA2023_MODEL_INFO

    supports_span_context = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        **route_options: Any,
    ) -> None:
        """Initialize the route model and reference fiber properties."""
        if egn:
            msg = (
                "buglia2023 implements the GN model only: Buglia et al. (JLT 2023, "
                "doi:10.1109/JLT.2023.3256185) derive the closed-form expression for the "
                "Gaussian-noise model with ISRS; no EGN variant is published. "
                "Use egn=False."
            )
            raise ValueError(msg)
        super().__init__(
            grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options
        )

        self.ref_fiber_config = ref_fiber_config
        self.n2_m2_w = float(ref_fiber_config.nonlinear_index_n2)
        self.a_eff_m2 = _calc_aeff_dynamic(
            self.lambdas_m,
            ref_fiber_config.effective_area_um2_ref,
            ref_fiber_config.aeff_slope_um2_nm,
            ref_fiber_config.reference_wavelength_nm,
        )
        ref_lambda_m = 2.99792458e8 / self.band_center_hz
        self.beta2_s2_m = _calc_beta2_from_d(
            ref_fiber_config.dispersion_parameter_d, ref_lambda_m
        )
        self.beta3_s3_m = _calc_beta3_from_s_d(
            ref_fiber_config.dispersion_slope_s,
            ref_fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> Buglia2023SpanKernel:
        """Build a single-span kernel for the given fiber."""
        if fiber_config.use_attenuation_polynomial:
            alpha_npm = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            alpha_db_km = fiber_config.attenuation_db_km_ref
            alpha_npm = np.full(
                self.n_nodes, alpha_db_km * (np.log(10) / 10.0) / 1e3, dtype=np.float64
            )
        return Buglia2023SpanKernel(
            f_rel_hz=self.f_rel_hz,
            lambdas_m=self.lambdas_m,
            a_eff_m2=self.a_eff_m2,
            alpha_npm=alpha_npm,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            n2_m2_w=self.n2_m2_w,
            b_hz_channels=self.channel_bandwidth_hz_channels,
            c_r_w_m_hz=_SRS_CR,
            symbol_rate_hz=self.symbol_rate_hz,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """Evaluate the span and apply amplifier and ASE effects.

        NLI amplitudes are referenced to the fiber output before amplification."""
        kernel = cast("Buglia2023SpanKernel", self.kernel_for(fiber_config))
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


register_model(BUGLIA2023_MODEL_INFO, Buglia2023Model)
