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
*visibility*

.. math::

    V(d) = \left\langle \cos\big(\rho\, d\big)\right\rangle_W ,

where the average is taken over the **NLI noise density** ``W`` and ``rho`` is
the phase-mismatch *rate* of a signal pair.  The averaging weight is the
crucial point: the quantity that adds coherently across spans is the *noise*,
not the signal spectrum, so the pair ``(f1, f2)`` has to be weighted with the
NLI power it generates,

.. math::

    W(\rho) = \frac{\left|1 - e^{-(\alpha + j\rho)L}\right|^2}{\alpha^2 + \rho^2},
    \qquad \rho = 4\pi^2 |k_{\mathrm{mix}}| (f_1-f)(f_2-f) ,

i.e. the closed-form GN weight of the pair (the Lorentzian ``1/(alpha^2 +
rho^2)`` times the coherent numerator of a fibre of length ``L``).  A *uniform*
band average of the cosine -- ``Si(z)/z`` with ``z = pi^2 |beta2| d B^2`` -- is
the visibility of two copies of the **signal** spectrum, not of the noise: it
gives the same weight to the strongly mismatched pairs that the GN kernel
barely sees, and it underestimates the coherence by a factor of about six on
the C band (see the module's own docstring section "口径" and the delivery
report).

The reduction to a *one-dimensional* quadrature is exact.  With
``t = |rho|/alpha`` both the weight and the cosine depend on ``t`` alone,

.. math::

    W(\rho)\,\mathrm{d}\rho =
    \frac{\left|1 - e^{-(\alpha + j\rho)L}\right|^2}{1+t^2}\,\mathrm{d}t ,
    \qquad
    \int_0^t \frac{\cos(m\alpha L s)}{1+s^2}\,\mathrm{d}s =: C_m(t) ,

and the inner integral over the second signal frequency therefore has the
closed-form antiderivative

.. math::

    \int \left|1 - e^{-(\alpha+j\rho)L}\right|^2 \cos(k\rho L)
    \frac{\mathrm{d}t}{1+t^2}
    = (1+e^{-2\alpha L})\,C_k(t) - e^{-\alpha L}
      \left(C_{|k-1|}(t) + C_{k+1}(t)\right) ,

because ``2\cos(\alpha L s)\cos(k\alpha L s) = \cos((k-1)\alpha L s) +
\cos((k+1)\alpha L s)``.  The remaining ``C_m(t)`` is tabulated once per
fibre; its infinite limit is the Fourier transform of a Lorentzian,
``C_m(\infty) = (\pi/2) e^{-m\alpha L}``, which also anchors the asymptotic
tail used beyond the table.

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

import math
from dataclasses import replace

import numpy as np

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

#: Number of observed frequencies across the channel band in the NLI-weighted
#: visibility quadrature.
#:
#: The visibility integrand is steep at the band edges (the admissible
#: three-wave region degenerates there) and smooth in the middle, so the
#: observed frequencies sit on a cosine grid.  41 points put the C-band
#: coherent factor within 3e-5 of the value obtained with 161 points.
VISIBILITY_FREQUENCY_POINTS = 41

#: Gauss-Legendre nodes per integration sub-interval of the visibility
#: quadrature.
#:
#: The integrand of the outer (observed frequency) integral is smooth, so
#: 200 nodes per sub-interval are already converged to 1e-5; the node count is
#: fixed so that the cost of the cached ramp does not depend on the case.
VISIBILITY_QUADRATURE_POINTS = 200

#: Step of the tabulated antiderivative ``C_m(t)``.
#:
#: The table is built with the corrected trapezoid rule, whose error is
#: fourth order in the step; 1e-3 keeps the visibility accurate to 1e-6.
VISIBILITY_TABLE_STEP = 1.0e-3

#: Largest ``t`` covered by the tabulated antiderivative.
#:
#: Beyond it the asymptotic expansion of the Lorentzian tail is used, which is
#: accurate to better than 1e-8 at ``t = 30``.
VISIBILITY_TABLE_LIMIT = 30.0

#: Smallest ``t`` range the tabulated antiderivative covers.
#:
#: A very small dispersion would give a table of a few points; the minimum
#: keeps the table well conditioned (and the asymptotic tail, which is only
#: valid for large ``t``, is never reached in that case).
VISIBILITY_TABLE_MINIMUM = 1.0

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
    alpha_npm_m: float,
    bandwidth_hz: NDArrayFloat,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int = VISIBILITY_FREQUENCY_POINTS,
    n_quadrature: int = VISIBILITY_QUADRATURE_POINTS,
) -> NDArrayFloat:
    """NLI-weighted visibility of two self-beating (SPM) NLI copies.

    The visibility of two noise components generated ``d`` metres apart is the
    average of ``cos(rho d)`` over the **NLI density** of the signal pairs
    that generate the noise, not over the signal spectrum.  For a pair
    ``(f1, f2)`` observed at the frequency ``f`` the closed-form GN weight is

    ``W(rho) = |1 - exp(-(alpha + j rho) L)|^2 / (alpha^2 + rho^2)``,
    ``rho = 4 pi^2 |k_mix| (f1 - f)(f2 - f)``,

    and the signal pairs are those whose three waves (the two inputs and the
    generated idler ``f1 + f2 - f``) all fall inside the channel band.  The
    observed frequency ``f`` is integrated over the band, so the result is the
    frequency-resolved coherence of the NLI.

    The two-dimensional pair integral is reduced to a one-dimensional
    quadrature: with ``t = |rho|/alpha`` the inner integral over the second
    input frequency has the closed-form antiderivative built from
    ``C_m(t) = int_0^t cos(m alpha L s)/(1+s^2) ds`` (see the module
    docstring).  No two-dimensional integration is done anywhere.

    Args
    ----
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        bandwidth_hz: ``(N,)`` channel bandwidths (Hz).
        span_length_m: Length of one span (m); the noise is generated in it.
        distances_m: ``(K,)`` distances between the two copies (m).
        n_frequency: Number of observed frequencies across the channel band.
        n_quadrature: Gauss-Legendre nodes per integration sub-interval.

    Returns
    -------
        ``(K, N)`` visibility in ``[-1, 1]``; ``1`` without dispersion.
    """
    bandwidths = np.asarray(bandwidth_hz, dtype=np.float64)
    distances = np.asarray(distances_m, dtype=np.float64)
    visibility = np.ones((distances.size, bandwidths.size), dtype=np.float64)
    if distances.size == 0 or bandwidths.size == 0:
        return visibility
    if beta2_abs_s2_m == 0.0 or alpha_npm_m <= 0.0:
        # Without dispersion the two copies are perfectly coherent.
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


def _gn_weighted_spm_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int,
    n_quadrature: int,
) -> NDArrayFloat:
    """NLI-weighted SPM visibility of one bandwidth (see :func:`_self_beating_visibility`).

    Args
    ----
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        bandwidth_hz: Channel bandwidth (Hz).
        span_length_m: Length of one span (m).
        distances_m: ``(K,)`` distances between the two copies (m).
        n_frequency: Number of observed frequencies across the channel band.
        n_quadrature: Gauss-Legendre nodes per integration sub-interval.

    Returns
    -------
        ``(K,)`` visibility in ``[-1, 1]``.
    """
    distances = np.asarray(distances_m, dtype=np.float64)
    half_band = 0.5 * bandwidth_hz
    # Mismatch rate per unit frequency product: rho = rate * u * v.
    rate = 4.0 * np.pi**2 * beta2_abs_s2_m
    exp_alpha_l = float(np.exp(-alpha_npm_m * span_length_m))
    t_lo, t_hi, measure = _spm_visibility_nodes(
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
    # Denominator: the NLI power itself, i.e. the weight without the cosine.
    weight_hi = (1.0 + exp_alpha_l**2) * tables.C(0, t_hi) - 2.0 * exp_alpha_l * tables.C(1, t_hi)
    weight_lo = (1.0 + exp_alpha_l**2) * tables.C(0, t_lo) - 2.0 * exp_alpha_l * tables.C(1, t_lo)
    total = float(np.sum(measure * (weight_hi - weight_lo)))
    visibility = np.zeros_like(distances)
    for index in range(distances.size):
        k = index + 1
        cosine_hi = (
            (1.0 + exp_alpha_l**2) * tables.C(k, t_hi)
            - exp_alpha_l * (tables.C(abs(k - 1), t_hi) + tables.C(k + 1, t_hi))
        )
        cosine_lo = (
            (1.0 + exp_alpha_l**2) * tables.C(k, t_lo)
            - exp_alpha_l * (tables.C(abs(k - 1), t_lo) + tables.C(k + 1, t_lo))
        )
        visibility[index] = float(np.sum(measure * (cosine_hi - cosine_lo))) / total
    return visibility


def _spm_visibility_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    n_frequency: int,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Geometry of the SPM three-wave region for the visibility quadrature.

    The observed frequency ``f`` runs over the channel band; for every ``f``
    the two input offsets ``u = f1 - f`` and ``v = f2 - f`` and the idler
    offset ``u + v`` must stay inside the band.  The inner integral over
    ``v`` is analytic, so the quadrature only needs the ``(f, u)`` nodes.

    The observed frequencies are distributed on a cosine grid: the integrand
    is steep at the band edges (where the admissible region degenerates) and
    smooth in the middle, and the cosine grid converges where a uniform grid
    still drifts.

    Args
    ----
        half_band_hz: Half of the channel bandwidth (Hz).
        rate: ``4 pi^2 |beta2|``, the mismatch rate per unit frequency product.
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        n_frequency: Number of observed frequencies.
        n_quadrature: Gauss-Legendre nodes per integration sub-interval.

    Returns
    -------
        ``(t_lo, t_hi, measure)``: the limits of the reduced variable
        ``t = |rho| / alpha`` and the quadrature measure of every node.
    """
    if n_frequency > 1:
        index = np.arange(n_frequency, dtype=np.float64)
        frequencies = -half_band_hz * np.cos(np.pi * index / (n_frequency - 1))
        # Composite trapezoid weights of the (non-uniform) frequency grid.
        frequency_weights = np.empty(n_frequency, dtype=np.float64)
        frequency_weights[0] = 0.5 * (frequencies[1] - frequencies[0])
        frequency_weights[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
        frequency_weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
    else:
        frequencies = np.zeros(1, dtype=np.float64)
        frequency_weights = np.ones(1, dtype=np.float64)
    nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)
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
            # The second offset is limited by its own band and by the idler.
            v_low = np.maximum(low, low - u)
            v_high = np.minimum(high, high - u)
            valid = v_high > v_low
            if not np.any(valid):
                continue
            u = u[valid]
            weight = weight[valid]
            v_low = v_low[valid]
            v_high = v_high[valid]
            # dv = dt / (rate |u| alpha) with t = |rho| / alpha.
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


class _LorentzianAntiderivative:
    r"""``C_m(t) = int_0^t cos(m alpha L s)/(1+s^2) ds`` for ``m >= 1``.

    The table is built with the composite trapezoid rule, corrected by the
    leading Euler-Maclaurin term so that the fourth-order accuracy is
    recovered at a fraction of the cost of a Simpson grid.  ``C_0(t) =
    arctan(t)``.  Beyond the table the asymptotic expansion of the tail
    ``int_t^oo cos(beta s)/(1+s^2) ds`` is used; it is anchored by
    ``C_m(oo) = (pi/2) exp(-m alpha L)``, the Fourier transform of a
    Lorentzian.

    Args
    ----
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        span_length_m: Length of one span (m).
        m_max: Largest index ``m`` to tabulate.
        t_end: Largest ``t`` the table has to cover.
    """

    def __init__(
        self,
        alpha_npm_m: float,
        span_length_m: float,
        m_max: int,
        t_end: float,
        step: float = VISIBILITY_TABLE_STEP,
    ) -> None:
        self.alpha = float(alpha_npm_m)
        self.span_length = float(span_length_m)
        self.beta = self.alpha * self.span_length
        limit = min(max(float(t_end), VISIBILITY_TABLE_MINIMUM), VISIBILITY_TABLE_LIMIT)
        n_points = max(3, int(math.ceil(limit / step)) + 1)
        if n_points % 2 == 0:
            n_points += 1
        self.t = np.linspace(0.0, limit, n_points)
        h = float(self.t[1] - self.t[0])
        self.tables: dict[int, NDArrayFloat] = {}
        for m in range(1, int(m_max) + 1):
            f = np.cos(m * self.beta * self.t) / (1.0 + self.t * self.t)
            derivative = (
                -m * self.beta * np.sin(m * self.beta * self.t) / (1.0 + self.t * self.t)
                - 2.0 * self.t * np.cos(m * self.beta * self.t) / (1.0 + self.t * self.t) ** 2
            )
            self.tables[m] = self._cumulative(f, derivative, h)

    @staticmethod
    def _cumulative(
        values: NDArrayFloat, derivative: NDArrayFloat, h: float
    ) -> NDArrayFloat:
        """Cumulative integral of ``values`` on a uniform grid (corrected trapezoid)."""
        integral = np.zeros_like(values)
        integral[1:] = np.cumsum(0.5 * h * (values[:-1] + values[1:]))
        # Euler-Maclaurin: int = T - h^2/12 (f'(b) - f'(a)) + O(h^4).
        integral[1:] -= h * h / 12.0 * (derivative[1:] - derivative[0])
        return integral

    def C(self, m: int, t: NDArrayFloat) -> NDArrayFloat:
        """``C_m(t)`` for an array of (possibly negative) ``t``."""
        if m == 0:
            return np.arctan(np.asarray(t, dtype=np.float64))
        t = np.asarray(t, dtype=np.float64)
        out = np.empty_like(t)
        sign = np.sign(t)
        magnitude = np.abs(t)
        inside = magnitude <= self.t[-1]
        if np.any(inside):
            out[inside] = sign[inside] * np.interp(magnitude[inside], self.t, self.tables[m])
        outside = ~inside
        if np.any(outside):
            beta = m * self.beta
            tail = _lorentzian_tail(beta, magnitude[outside])
            out[outside] = sign[outside] * (0.5 * math.pi * math.exp(-beta) - tail)
        return out


def _lorentzian_tail(beta: float, t: NDArrayFloat) -> NDArrayFloat:
    """``int_t^oo cos(beta s)/(1+s^2) ds`` by repeated integration by parts."""
    denominator = 1.0 + t * t
    f = 1.0 / denominator
    df = -2.0 * t / denominator**2
    d2f = (6.0 * t * t - 2.0) / denominator**3
    d3f = -24.0 * t * (t * t - 1.0) / denominator**4
    d4f = 24.0 * (5.0 * t**4 - 10.0 * t * t + 1.0) / denominator**5
    sin_beta = np.sin(beta * t)
    cos_beta = np.cos(beta * t)
    return (
        -f * sin_beta / beta
        - df * cos_beta / beta**2
        + d2f * sin_beta / beta**3
        + d3f * cos_beta / beta**4
        - d4f * sin_beta / beta**5
    )


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

    This part keeps the uniform band average on purpose: the inter-channel
    offset is so large that the pairs which survive the GN kernel are already
    incoherent, and both conventions give the same (nearly non-coherent)
    answer -- the independent audit measures an XPM buildup of 1.008 (GN
    weighted) against 1.000 (uniform), i.e. below the noise floor of the
    cross-span phase.  Switching the weight would therefore move nothing and
    would only add cost; the delivery report gives the numbers.

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
        # Power attenuation used by the NLI-weighted visibility.  The weight
        # of a signal pair is |1 - exp(-(alpha + j rho) L)|^2 / (alpha^2 +
        # rho^2), and the attenuation of the fibre varies by well under a per
        # cent over the C band, so a single fibre-level value is used for the
        # whole channel plan.
        self.alpha_visibility_npm = float(np.mean(self.alpha_npm))
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
        visibility_spm = _self_beating_visibility(
            self.beta2_abs,
            self.alpha_visibility_npm,
            self.b_hz,
            float(span_length_m),
            distances,
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
