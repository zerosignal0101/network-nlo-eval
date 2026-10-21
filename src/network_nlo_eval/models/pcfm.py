"""Polynomial closed-form model (PCFM) of Poggiolini et al., JLT 2026.

The model is the closed-form counterpart of the numerical ISRS-GN reference
model of this repository.  It implements the "polynomial closed form model for
ultra-wideband transmission systems" of

    L. Poggiolini, M. Jiang, G. Gao, F. Forghieri,
    "Polynomial Closed Form Model for Ultra-Wideband Transmission Systems",
    J. Lightwave Technol., 2026, doi:10.1109/jlt.2026.3678322

The three ingredients of the model are:

1. the *normalised* spatial power profile (SPP) ``p_i(z) = P_i(z)/P_i(0)`` of
   every channel, obtained by integrating the coupled ISRS power equations
   over the span (Section IV-B of the paper);
2. the polynomial representation of the SPP, Eq. (23), whose coefficients are
   fitted by least squares on the *normalised* profile;
3. the closed-form island kernels, Eq. (24) for XCI and Eq. (30) for SCI.

Assumptions of the paper that are reproduced here: rectangular channel
spectra, locally white NLI noise, rectangular (inscribed) integration
islands, negligible MCI contribution and constant fibre characteristics over
each channel.  Raman pumping is *not* modelled: only the inter-channel
Stokes effect of the WDM comb is integrated.

The EGN correction (``egn=True``) applies the machine-learning coefficient
``rho`` of Zefreh et al., JLT 2020, Eq. (14), with the coefficients of Table IV
of that paper.  Those coefficients are *fitted* values of the reference paper,
not a derivation of this work package; they are reproduced verbatim and are
marked as such in the module documentation.

Two deliberate deviations from the printed paper are documented here, because
both change the numbers and both need to be visible to a reader of the code:

``The outer p_CUT(L) factor of Eqs. (21) and (26) is not applied``
    Eqs. (21) and (26) multiply the closed-form kernels by the end-of-span
    power ratio ``p_CUT(L) = P_CUT(L)/P_CUT(0)``.  The kernels themselves
    already integrate the *normalised* spatial power profile (SPP) over the
    span, Eq. (A.20), so the span loss is carried by the profile.  Applying
    the outer factor on top of it therefore counts the span loss twice: on the
    100 km SMF benchmark the resulting NLI sits about 16 dB below the
    numerical GN reference, which contradicts the accuracy claimed in
    Section IV of the paper.  The kernels are therefore used as printed and the
    outer factor is set to one.

``Fourth-order dispersion is dropped``
    The framework carries no fourth-order dispersion coefficient, so the
    ``beta4`` term of Eq. (A.15) is set to zero.  Over a C+L band this is a
    sub-per-mille effect on the effective dispersion.

**Inter-span coherence (not a formula of the paper)**

The paper accumulates the NLI linearly over transparent spans, Eq. (8): the
NLI power of a link with ``N`` spans is ``N`` times the single-span value.
That is the *incoherent* limit and it is systematically optimistic by
0.2-0.3 dB on a 10-span link.  The rest of this repository
(``numerical``, ``semrau2019``, ``carena2014``, ``damico2022``, ``cgn``) uses
one common convention instead: the inter-span visibility is weighted by the
**NLI noise power** (not by the signal spectrum), and the model hands the
route layer a *complex amplitude* per span so that the route layer can sum
``|sum_k a_k|**2``.

This module follows that convention.  The PCFM paper contains no inter-span
coherence formula of its own, so the physics used here is the one of the
numerical GN reference of the repository, implemented from scratch in this
file (no import from :mod:`network_nlo_eval.models.numerical`, so that the
two models stay independent):

* the NLI weight of a signal pair is the closed-form GN weight

  .. code-block:: text

      W(rho) = |1 - exp(-(alpha + j rho) L)|**2 / (alpha**2 + rho**2) ,
      rho = 4 pi**2 |beta2| (f1 - f)(f2 - f) ,

* the two copies of the same noise, generated ``d`` metres apart, differ in
  phase by ``rho * d``, hence the visibility is the NLI-weighted average

  .. code-block:: text

      V(d) = <cos(rho d)>_{W} ,

* the coherent factor of a link with ``N`` spans is the visibility-weighted
  sum over the span pairs

  .. code-block:: text

      R(N) = N + 2 sum_{k=1}^{N-1} (N - k) V(k L) ,

* the contract layer carries one complex number per channel and span, so the
  factor ``R(N)`` is inverted into an *equivalent phase ramp* ``Theta``
  (the Dirichlet kernel ``[sin(N Theta/2)/sin(Theta/2)]**2`` decreases
  monotonically on ``(0, 2 pi/N)``, so a bisection is enough) and the phase
  of span ``k`` is ``Theta * (accumulated length / span length)``.

The single-span case (``total_spans == 1``) keeps the pure power convention
and is bit-for-bit identical to the implementation without coherence.

**Known approximation: equal spans**

The coherent factor ``R(N)`` is the visibility-weighted sum over the span
pairs, and the visibility only depends on the distance between the two
copies of the noise.  The contract layer carries one complex number per
channel and span, so the factor is inverted into an equivalent phase ramp
that is computed for the *current* span length.  On a link whose spans all
have the same length the route layer then reproduces ``R(N)`` exactly; on a
link with unequal spans the phase step is folded with the accumulated
length, and the link-level factor is only exact in the equal-span case.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import sici

from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
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
from network_nlo_eval.models.pcfm_kernels import (
    C_LIGHT,
    _isrs_coefficient_matrix_jit,
    _j_batch_jit,
    _profile_energy_jit,
    _sci_kernel_batch_jit,
    _solve_spp_jit,
    r_coefficient_matrix,
)
from network_nlo_eval.models.route import RouteModel, apply_amplifier_and_roadm, resolve_noise_figure
from network_nlo_eval.network.elements import FiberSpanConfig

__all__ = [
    "EGN_A_COEFFICIENTS",
    "EGN_PHI_VALUES",
    "PCFM_MODEL_INFO",
    "PCFMModel",
    "PCFMSpanKernel",
]

#: Number of RK4 steps of the ISRS power-profile integration.
#:
#: The coupled power equations are integrated with a fixed-step RK4 scheme.
#: The ISRS rate over a 100 km SMF span is a few e-foldings, so the step
#: stays far inside the RK4 stability and accuracy region; 32 steps resolve
#: the SPP of every channel of the benchmark combs to better than 0.1 % (see
#: the report).
DEFAULT_SPP_STEPS: int = 32

#: Machine-learning coefficients ``a1 ... a24`` of the EGN correction.
#:
#: Source: M. Ranjbar-Zefreh, F. Forghieri, S. Piciaccia, P. Poggiolini,
#: "Accurate closed-form real-time EGN model formula leveraging machine-learning
#: over 8500 thoroughly randomized full C-band systems", J. Lightwave Technol.,
#: vol. 38, no. 18, pp. 4987-4999, 2020, **Table IV** (optimized values of the
#: parameters ``a1-a24`` for the machine-learning factors of CFM4), used
#: through **Eq. (14)** of that paper.
#:
#: These are *fitted* coefficients of the reference paper.  They are not
#: derived here and are reproduced verbatim; the PCFM paper applies them
#: unchanged to its own closed-form kernels.
EGN_A_COEFFICIENTS: dict[str, float] = {
    "a1": +1.0436e0,
    "a2": -1.1878e0,
    "a3": +1.0573e0,
    "a4": -1.8309e+1,
    "a5": +1.6665e0,
    "a6": -1.0020e0,
    "a7": +9.0933e0,
    "a8": +6.6420e-3,
    "a9": +8.4481e-1,
    "a10": -1.8530e0,
    "a11": +9.4539e-1,
    "a12": -1.5421e+1,
    "a13": +1.0229e0,
    "a14": -1.1440e0,
    "a15": +1.1393e-2,
    "a16": +3.8070e+5,
    "a17": +1.4785e+3,
    "a18": -2.2593e0,
    "a19": -6.7997e-1,
    "a20": +2.0215e0,
    "a21": -2.9781e-1,
    "a22": +5.5130e-1,
    "a23": -3.6718e-1,
    "a24": +1.1486e0,
}

#: Exact values of the EGN format constant ``Phi``, from **Table I** of the
#: same reference paper ("Exact values of the Phi parameter").
EGN_PHI_VALUES: dict[str, float] = {
    "PM-BPSK": 1.0,
    "PM-QPSK": 1.0,
    "PM-8QAM": 2.0 / 3.0,
    "PM-16QAM": 17.0 / 25.0,
    "PM-32QPSK": 69.0 / 100.0,
    "PM-64QAM": 13.0 / 21.0,
    "PM-128": 1105.0 / 1681.0,
    "PM-256": 257.0 / 425.0,
    "PM-Gaussian": 0.0,
}

#: Format constant used by this model.
#:
#: The evaluation framework of this repository is format agnostic: it models
#: the Gaussian noise of the *Gaussian* channel, for which the EGN constant of
#: Table I is ``Phi = 0``.  The PCFM paper uses the same Gaussian-shaped
#: format in its own test system (its Table I), so the correction applied
#: here is the one the paper applies.
EGN_PHI: float = EGN_PHI_VALUES["PM-Gaussian"]

#: Channel roll-off assumed by the model.
#:
#: The PCFM assumes rectangular channel spectra (Section II-B), i.e. a
#: roll-off of zero.
EGN_ROLLOFF: float = 0.0

#: Constant of the asymptotic expansion of ``J(X) = int_0^X Si(t)/t dt``.
#:
#: ``C_r = int_0^1 Si(t)/t dt + int_1^inf [Si(t)-pi/2]/t dt``.  It is filled in
#: on first use because computing it needs a quadrature.
_J_CONSTANT: float | None = None


def _j_constant() -> float:
    """Return the constant ``C_r`` of the asymptotic expansion of ``J``.

    The value is computed once with a converged quadrature and then cached.
    The integral over ``[1, inf)`` carries the oscillatory tail of the sine
    integral, so it is split into sub-intervals shorter than half a period;
    a single call over the whole range does not reach the requested
    tolerance.  The quadrature is asked for 1e-12, which is well inside the
    round-off limit of the integrand and far below the accuracy the constant
    has to deliver.
    """
    global _J_CONSTANT
    if _J_CONSTANT is None:
        from scipy.integrate import quad

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            low = quad(lambda t: sici(t)[0] / t, 0.0, 1.0, limit=400, epsabs=1e-12, epsrel=1e-12)[0]
            cut = 200.0
            n_interval = 128
            step = (cut - 1.0) / n_interval
            mid = 0.0
            for index in range(n_interval):
                lo = 1.0 + index * step
                mid += quad(
                    lambda t: (sici(t)[0] - 0.5 * np.pi) / t,
                    lo,
                    lo + step,
                    limit=100,
                    epsabs=1e-12,
                    epsrel=1e-12,
                )[0]
        mid += _j_tail(cut)
        _J_CONSTANT = float(low + mid)
    return _J_CONSTANT


def _j_tail(x: float) -> float:
    """``int_x^inf [Si(t) - pi/2]/t dt`` (asymptotic expansion)."""
    from network_nlo_eval.models.pcfm_kernels import _j_tail_jit

    return float(_j_tail_jit(x))


def egn_rho_nch(
    beta2_accum_ps2: NDArrayFloat,
    roll_off_cut: float = EGN_ROLLOFF,
    roll_off_nch: float = EGN_ROLLOFF,
    phi: float = EGN_PHI,
) -> NDArrayFloat:
    """Machine-learning factor ``rho_nch`` of Zefreh et al. 2020, Eq. (14).

    ``rho_nch = (1 + a19 r_CUT^a20 + a21 r_nch^a22)
    { a1 + a2 Phi^a3 + a4 Phi^a5 (1 + a6 [|beta2_acc| + a7]^a8) }``

    Args:
        beta2_accum_ps2: ``(N,)`` accumulated effective dispersion of every
            channel at the input of the current span (ps^2).
        roll_off_cut: Roll-off of the CUT.
        roll_off_nch: Roll-off of the interfering channel.
        phi: EGN format constant.

    Returns
    -------
        ``(N,)`` correction factors.
    """
    a = EGN_A_COEFFICIENTS
    acc = np.abs(np.asarray(beta2_accum_ps2, dtype=np.float64))
    inner = a["a1"] + a["a2"] * phi**a["a3"] + a["a4"] * phi**a["a5"] * (
        1.0 + a["a6"] * (acc + a["a7"]) ** a["a8"]
    )
    outer = 1.0 + a["a19"] * roll_off_cut**a["a20"] + a["a21"] * roll_off_nch**a["a22"]
    return outer * inner


def egn_rho_cut(
    beta2_accum_ps2: NDArrayFloat,
    symbol_rate_hz: float,
    roll_off_cut: float = EGN_ROLLOFF,
    phi: float = EGN_PHI,
) -> NDArrayFloat:
    """Machine-learning factor ``rho_CUT`` of Zefreh et al. 2020, Eq. (14).

    ``rho_CUT = (1 + a23 r_CUT^a24) { a9 + a10 Phi^a11 + a12 Phi^a13
    (1 + a14 R_CUT^a15 + a16 [|beta2_acc| + a17]^a18) }``

    Args:
        beta2_accum_ps2: ``(N,)`` accumulated effective dispersion (ps^2).
        symbol_rate_hz: Symbol rate of the channels (Hz).  The reference paper
            uses ``R_CUT`` in TBaud.
        roll_off_cut: Roll-off of the CUT.
        phi: EGN format constant.

    Returns
    -------
        ``(N,)`` correction factors.
    """
    a = EGN_A_COEFFICIENTS
    acc = np.abs(np.asarray(beta2_accum_ps2, dtype=np.float64))
    rate_tbaud = float(symbol_rate_hz) / 1.0e12
    inner = a["a9"] + a["a10"] * phi**a["a11"] + a["a12"] * phi**a["a13"] * (
        1.0 + a["a14"] * rate_tbaud**a["a15"] + a["a16"] * (acc + a["a17"]) ** a["a18"]
    )
    outer = 1.0 + a["a23"] * roll_off_cut**a["a24"]
    return outer * inner


# =============================================================================
# Inter-span coherence: NLI-weighted visibility -> coherent factor -> phase ramp
# =============================================================================
#
# The whole block is a self-contained implementation of the convention
# documented in the module docstring.  It deliberately does not import the
# helpers of the numerical model: two models that share the *physics* must not
# share the *code*, otherwise a change in one of them silently changes the
# other.

#: Number of observed frequencies across the channel band in the visibility
#: quadrature.  The integrand is steep at the band edges (the admissible
#: region degenerates there) and smooth in the middle, so the observed
#: frequencies sit on a cosine grid.
VISIBILITY_FREQUENCY_POINTS = 41

#: Number of Gauss-Legendre nodes per integration sub-interval of the
#: visibility quadrature.
VISIBILITY_QUADRATURE_POINTS = 200

#: Step of the tabulated Lorentzian antiderivative (dimensionless variable
#: ``t = |rho| / alpha``).
VISIBILITY_TABLE_STEP = 1.0e-3

#: Largest ``t`` covered by the table.  Beyond it the asymptotic expansion of
#: the tail is used.
VISIBILITY_TABLE_LIMIT = 30.0

#: Smallest ``t`` interval the table covers.  A very small dispersion would
#: give a table of a few points only, and the asymptotic tail (valid for large
#: ``t``) would then be used where it is not valid.
VISIBILITY_TABLE_MINIMUM = 1.0

#: Number of bisection steps of the equivalent phase ramp.
RAMP_BISECTION_STEPS = 60


def _lorentzian_tail(beta: float, t: NDArrayFloat) -> NDArrayFloat:
    r"""Compute ``int_t^oo cos(beta s)/(1+s**2) ds`` by integration by parts.

    The integrand is ``f(s) = 1/(1+s**2)``.  Four integrations by parts give

    .. code-block:: text

        -f sin(beta t)/beta - f' cos(beta t)/beta**2
        + f'' sin(beta t)/beta**3 + f''' cos(beta t)/beta**4
        - f'''' sin(beta t)/beta**5 ,

    and the remainder is bounded by ``|f'''''| / beta**6``.

    Args:
        beta: Angular frequency of the cosine (rad per unit of ``t``).
        t: Lower limit of the integral (positive).

    Returns
    -------
        The value of the tail integral, same shape as ``t``.
    """
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


class _LorentzianAntiderivative:
    r"""Tabulate ``C_m(t) = int_0^t cos(m alpha L s)/(1+s**2) ds``.

    The table is built with the composite trapezoid rule corrected by the
    leading Euler-Maclaurin term, so that the quadrature error is fourth
    order in the step.  ``C_0(t) = arctan(t)`` is exact and is not tabulated.
    Beyond the table the asymptotic expansion of the tail is used, anchored
    by the Fourier transform of a Lorentzian, ``C_m(oo) = (pi/2) exp(-m
    alpha L)``.

    Args:
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        span_length_m: Length of one span (m).
        m_max: Largest index ``m`` to tabulate.
        t_end: Largest ``t`` the table has to cover.
        step: Step of the table.
    """

    def __init__(
        self,
        alpha_npm_m: float,
        span_length_m: float,
        m_max: int,
        t_end: float,
        step: float = VISIBILITY_TABLE_STEP,
    ) -> None:
        """Build the tables of ``C_m`` (see the class docstring)."""
        self.alpha = float(alpha_npm_m)
        self.span_length = float(span_length_m)
        self.beta = self.alpha * self.span_length
        limit = min(max(float(t_end), VISIBILITY_TABLE_MINIMUM), VISIBILITY_TABLE_LIMIT)
        n_points = max(3, int(np.ceil(limit / step)) + 1)
        if n_points % 2 == 0:
            n_points += 1
        self.t = np.linspace(0.0, limit, n_points)
        h = float(self.t[1] - self.t[0])
        self.tables: dict[int, NDArrayFloat] = {}
        for m in range(1, int(m_max) + 1):
            argument = m * self.beta * self.t
            values = np.cos(argument) / (1.0 + self.t * self.t)
            derivative = (
                -m * self.beta * np.sin(argument) / (1.0 + self.t * self.t)
                - 2.0 * self.t * np.cos(argument) / (1.0 + self.t * self.t) ** 2
            )
            self.tables[m] = self._cumulative(values, derivative, h)

    @staticmethod
    def _cumulative(
        values: NDArrayFloat, derivative: NDArrayFloat, h: float
    ) -> NDArrayFloat:
        """Integrate ``values`` on the uniform grid (Euler-Maclaurin corrected)."""
        integral = np.zeros_like(values)
        integral[1:] = np.cumsum(0.5 * h * (values[:-1] + values[1:]))
        # Euler-Maclaurin: int = T - h^2/12 (f'(b) - f'(a)) + O(h^4).
        integral[1:] -= h * h / 12.0 * (derivative[1:] - derivative[0])
        return integral

    def C(self, m: int, t: NDArrayFloat) -> NDArrayFloat:
        """Evaluate ``C_m(t)`` for an array of (possibly negative) ``t``."""
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
            out[outside] = sign[outside] * (0.5 * np.pi * np.exp(-beta) - tail)
        return out


def _spm_visibility_nodes(
    half_band_hz: float,
    rate: float,
    alpha_npm_m: float,
    n_frequency: int,
    n_quadrature: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Build the geometry of the SPM three-wave region for the quadrature.

    The observed frequency ``f`` runs over the channel band.  For every ``f``
    the two input offsets ``u = f1 - f`` and ``v = f2 - f`` and the idler
    offset ``u + v`` must stay inside the band, i.e.

    .. code-block:: text

        u in [-b - f, b - f],  v in [-b - f, b - f],
        v in [-b - f - u, b - f - u] .

    The inner integral over ``v`` is analytic (see
    :func:`_gn_weighted_spm_visibility`), so the quadrature only needs the
    ``(f, u)`` nodes.  The returned ``t`` limits are the dimensionless
    mismatch ``t = |rho| / alpha`` of the two ends of the ``v`` interval.

    Args:
        half_band_hz: Half of the channel bandwidth (Hz).
        rate: ``4 pi**2 |beta2|``, the mismatch rate per unit frequency
            product (s^2/m).
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        n_frequency: Number of observed frequencies.
        n_quadrature: Number of Gauss-Legendre nodes per sub-interval.

    Returns
    -------
        The tuple ``(t_lo, t_hi, measure)``: the limits of the reduced
        variable and the quadrature measure of every node.
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
            # The mismatch is rho = rate * u * v and t = |rho| / alpha, so
            # dv = alpha dt/(rate |u|).  The factor alpha**2 that separates
            # this from the measure used below is constant over the whole
            # band and cancels in the visibility ratio.
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
    n_frequency: int,
    n_quadrature: int,
) -> NDArrayFloat:
    """Compute the NLI-weighted SPM visibility of one bandwidth.

    The GN weight of a signal pair is

    .. code-block:: text

        W(rho) = |1 - exp(-(alpha + j rho) L)|**2 / (alpha**2 + rho**2)
               = L**2 (1 + e**(-2 abar) - 2 e**(-abar) cos(theta)) / (abar**2 + theta**2)

    with ``abar = alpha L`` and ``theta = rho L``.  The factor ``L**2`` is
    constant and drops out of the visibility ratio, and so does the constant
    ``abar**2`` between the two integrals; what is left is a one-dimensional
    quadrature over ``t = |rho| / alpha`` with the antiderivatives
    ``C_m(t) = int_0^t cos(m abar s)/(1+s**2) ds`` of
    :class:`_LorentzianAntiderivative`:

    .. code-block:: text

        denominator:  (1 + e**(-2 abar)) C_0 - 2 e**(-abar) C_1
        numerator:    (1 + e**(-2 abar)) C_k - e**(-abar) (C_(k-1) + C_(k+1))

    where ``k`` is the number of spans between the two copies of the noise
    (the extra phase of a copy generated ``k`` spans later is ``k theta``).

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        bandwidth_hz: Channel bandwidth (Hz).
        span_length_m: Length of one span (m).
        distances_m: ``(K,)`` distances between the two copies (m).
        n_frequency: Number of observed frequencies across the band.
        n_quadrature: Number of Gauss-Legendre nodes per sub-interval.

    Returns
    -------
        The ``(K,)`` visibility, in ``[-1, 1]``; ``1`` without dispersion.
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


def _self_beating_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: NDArrayFloat,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int = VISIBILITY_FREQUENCY_POINTS,
    n_quadrature: int = VISIBILITY_QUADRATURE_POINTS,
) -> NDArrayFloat:
    """Compute the NLI-weighted visibility of two self-beating (SPM) copies.

    The visibility is the average of ``cos(rho d)`` over the **NLI density**
    of the signal pairs that generate the noise, not over the signal
    spectrum.  The GN weight of a pair is ``1/(abar**2 + theta**2)``, so the
    NLI power concentrates on the phase-matched pairs, and those pairs are
    the ones that stay coherent across spans.

    Channels that share a bandwidth share a visibility, so the quadrature is
    run once per distinct bandwidth.

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        bandwidth_hz: ``(N,)`` channel bandwidths (Hz).
        span_length_m: Length of one span (m).
        distances_m: ``(K,)`` distances between the two copies (m).
        n_frequency: Number of observed frequencies across the band.
        n_quadrature: Number of Gauss-Legendre nodes per sub-interval.

    Returns
    -------
        The ``(K, N)`` visibility, in ``[-1, 1]``; ``1`` without dispersion.
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


def _cross_beating_visibility(
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    distance_m: float,
    partner_offset_hz: NDArrayFloat,
) -> NDArrayFloat:
    """Compute the band-averaged visibility of two cross-beating (XPM) copies.

    The phase mismatch of a cross pair is ``4 pi**2 |beta2| (Delta + y) L``
    for the first wave and ``x`` for the second one: the inter-channel offset
    ``Delta`` dominates the intra-band offset, so the visibility degenerates
    into the band average of ``cos(c x)`` with ``c = 4 pi**2 |beta2| d
    Delta``,

    .. code-block:: text

        V(d) = sinc(c B / 2) ,

    which vanishes as soon as the mismatch exceeds a radian.

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        bandwidth_hz: ``(N,)`` channel bandwidths (Hz).
        distance_m: Distance between the two copies (m).
        partner_offset_hz: ``(N,)`` offset to the interfering channel (Hz).

    Returns
    -------
        The ``(N,)`` visibility, in ``[-1, 1]``; ``1`` without dispersion.
    """
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    offset = np.asarray(partner_offset_hz, dtype=np.float64)
    argument = 2.0 * np.pi**2 * beta2_abs_s2_m * distance_m * offset * bandwidth
    visibility = np.ones_like(bandwidth)
    nonzero = argument != 0.0
    if np.any(nonzero):
        visibility[nonzero] = np.sin(argument[nonzero]) / argument[nonzero]
    return visibility


def _gn_weighted_xpm_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    distance_m: float,
    partner_offset_hz: float,
    n_offset: int = 32,
    n_mismatch: int = 400,
) -> float:
    """Audit the NLI-weighted visibility of a cross-beating (XPM) pair.

    This is the XPM counterpart of :func:`_gn_weighted_spm_visibility`: the
    same GN weight is applied to the pairs of one channel and of its
    interfering channel.  The idler of a cross pair falls outside the band,
    so it is not constrained, and the mismatch of such a pair is
    ``theta = 4 pi**2 |beta2| L x (Delta + y)``.

    Because ``Delta`` is two orders of magnitude above the channel bandwidth,
    the weight ``1/(abar**2 + theta**2)`` only survives in a narrow strip
    around ``x = 0``; a quadrature over the raw offset would miss it.  The
    integral is therefore reduced to the dimensionless mismatch
    ``t = theta/abar`` (the variable in which the weight is smooth) and the
    offset of the interfering channel is integrated separately.

    The function is a *diagnostic*: it quantifies how much the GN weighting
    would move the XPM inter-span structure, it is not used in the hot path.

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        bandwidth_hz: Channel bandwidth (Hz).
        span_length_m: Length of one span (m).
        distance_m: Distance between the two copies (m).
        partner_offset_hz: Offset to the interfering channel (Hz).
        n_offset: Number of Gauss-Legendre nodes of the interfering channel.
        n_mismatch: Number of Gauss-Legendre nodes of the mismatch variable.

    Returns
    -------
        The visibility of the pair, in ``[-1, 1]``.
    """
    half_band = 0.5 * bandwidth_hz
    rate = 4.0 * np.pi**2 * beta2_abs_s2_m * span_length_m
    abar = alpha_npm_m * span_length_m
    exp_abar = float(np.exp(-abar))
    offset_nodes, offset_weights = np.polynomial.legendre.leggauss(n_offset)
    offsets = partner_offset_hz + half_band * offset_nodes
    offset_weights = half_band * offset_weights
    mismatch_nodes, mismatch_weights = np.polynomial.legendre.leggauss(n_mismatch)
    numerator = 0.0
    denominator = 0.0
    for offset, offset_weight in zip(offsets, offset_weights, strict=True):
        # theta = rate * x * offset with |x| <= half_band, so the mismatch
        # reaches rate * half_band * offset at the band edge.
        t_max = rate * half_band * offset / abar
        mismatch = 0.5 * t_max * (mismatch_nodes + 1.0)
        weights = 0.5 * t_max * mismatch_weights
        theta = abar * mismatch
        numerator_factor = 1.0 - 2.0 * exp_abar * np.cos(theta) + exp_abar**2
        # du = abar dt/(rate offset) and the two signs of x give a factor two.
        integrand = (
            numerator_factor
            / (abar * (1.0 + mismatch * mismatch))
            / (rate * offset)
        )
        numerator += offset_weight * float(
            np.sum(weights * integrand * np.cos(theta * distance_m / span_length_m))
        )
        denominator += offset_weight * float(np.sum(weights * integrand))
    return numerator / denominator


def _coherent_factor(n_spans: int, visibility: NDArrayFloat) -> NDArrayFloat:
    """Compute the frequency-resolved inter-span coherent factor ``R``.

    ``visibility`` holds the visibility of the copies separated by one, two,
    ... spans.  The coherent factor is the visibility-weighted sum over the
    span pairs,

    .. code-block:: text

        R = N + 2 sum_{k=1}^{N-1} (N - k) V(k L) .

    A vanishing visibility gives ``R = N`` (incoherent accumulation), a
    visibility of one gives ``R = N**2`` (coherent accumulation).

    Args:
        n_spans: Number of spans of the link.
        visibility: ``(N_spans - 1, N_channels)`` visibility.

    Returns
    -------
        The ``(N_channels,)`` coherent factor.
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return n + 2.0 * (weights * visibility).sum(axis=0)


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Invert the coherent factor into an equivalent per-span phase ramp.

    The route layer sums the span amplitudes coherently, so ``N`` spans with
    a linear phase ramp ``psi`` give the Dirichlet kernel
    ``[sin(N psi/2)/sin(psi/2)]**2`` times the single-span noise.  That
    kernel decreases monotonically from ``N**2`` at ``psi = 0`` to ``0`` at
    ``psi = 2 pi/N``, so the wanted factor has a unique inverse and a
    bisection finds it.

    Args:
        n_spans: Number of spans of the link.
        target: ``(N_channels,)`` wanted coherent factor.

    Returns
    -------
        The ``(N_channels,)`` phase increment per span (rad).
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


PCFM_MODEL_INFO = ModelInfo(
    name="pcfm",
    summary="Polynomial closed-form GN model (SCI+XCI) with polynomial SPP fit.",
    family="gn",
    reference="Poggiolini et al., JLT 2026, doi:10.1109/jlt.2026.3678322",
    options=(
        ("poly_degree", "SPP polynomial degree N_p (paper: 5 adequate, 9 reference)."),
        ("egn", "Adds the Zefreh 2020 EGN correction on top of the GN result."),
    ),
)


class PCFMSpanKernel:
    """Single-span kernel of the polynomial closed-form model.

    The kernel owns everything that only depends on the fibre and on the
    channel plan: the per-channel attenuation, effective area and dispersion,
    the effective dispersion of every island, the pairwise XCI matrices and
    the polynomial fit operator.

    Args:
        f_abs_hz: ``(N,)`` channel centre frequencies (Hz).
        b_hz: ``(N,)`` channel bandwidths (Hz).
        fiber_config: Fibre parameters of the span.
        poly_degree: Degree ``N_p`` of the SPP polynomial, Eq. (23).
        symbol_rate_hz: Symbol rate (Hz), used by the EGN correction.
        egn: Apply the machine-learning correction of Zefreh et al. 2020.
        lumped_loss: Optional list of ``(position_m, loss_db)`` of lumped loss
            elements inside the span.  The paper treats lumped loss in
            Section IV-D; the framework has no field for it, so it is
            available at kernel level only.
        spp_steps: Number of RK4 steps of the power-profile integration.
    """

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        poly_degree: int = 5,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        lumped_loss: Sequence[tuple[float, float]] | None = None,
        spp_steps: int = DEFAULT_SPP_STEPS,
    ) -> None:
        """Prepare the per-channel constants of the span.

        Raises
        ------
            ValueError: The polynomial degree or the integration resolution
                is not positive, or a lumped loss element is out of the span.
        """
        if poly_degree < 0:
            raise ValueError(f"poly_degree must be >= 0, got {poly_degree}.")
        if spp_steps < 1:
            raise ValueError(f"spp_steps must be >= 1, got {spp_steps}.")

        self.poly_degree = int(poly_degree)
        self.n_poly = self.poly_degree + 1
        self.spp_steps = int(spp_steps)
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.egn = bool(egn)

        f_abs = np.asarray(f_abs_hz, dtype=np.float64)
        bandwidths = np.asarray(b_hz, dtype=np.float64)
        # The paper numbers the channels from the highest centre frequency, so
        # the internal order is descending; the model order is restored on the
        # way out.
        self.order = np.argsort(-f_abs, kind="stable")
        self.f_abs_hz = np.ascontiguousarray(f_abs[self.order])
        self.b_hz = np.ascontiguousarray(bandwidths[self.order])
        self.n_nodes = int(self.f_abs_hz.shape[0])
        self.lambdas_m = C_LIGHT / self.f_abs_hz

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
        self.n2_m2_w = float(fiber_config.nonlinear_index_n2)

        lambda_ref_m = fiber_config.reference_wavelength_nm * 1e-9
        self.beta2 = float(_calc_beta2_from_d(fiber_config.dispersion_parameter_d, lambda_ref_m))
        self.beta3 = float(
            _calc_beta3_from_s_d(
                fiber_config.dispersion_slope_s,
                fiber_config.dispersion_parameter_d,
                lambda_ref_m,
            )
        )
        self.f_ref_hz = C_LIGHT / lambda_ref_m
        # The framework carries no fourth-order dispersion, so the beta4 term
        # of Eq. (A.15) of the paper is dropped (it is a sub-per-mille effect
        # on the effective dispersion over a C+L band).
        self.beta4 = 0.0

        # --- inter-span coherence -------------------------------------------
        # Magnitude of the second-order dispersion (s^2/m): the only quantity
        # the inter-span visibility depends on.
        self.beta2_abs_s2_m = abs(self.beta2)
        # Fibre-level attenuation used by the NLI-weighted visibility.  The
        # weight of a signal pair is |1 - exp(-(alpha + j rho) L)|^2/(alpha^2
        # + rho^2) and the attenuation of the fibre varies by well under a
        # per cent over the C band, so a single fibre-level value is used for
        # the whole channel plan (the convention of the numerical model).
        self.alpha_visibility_npm = float(np.mean(self.alpha_npm))
        # Offset to the closest other channel (Hz): the cross-beating (XPM)
        # visibility is dominated by the nearest neighbours.
        if self.n_nodes > 1:
            offsets = np.abs(self.f_abs_hz[:, None] - self.f_abs_hz[None, :])
            np.fill_diagonal(offsets, np.inf)
            self.partner_offset_hz = offsets.min(axis=1)
        else:
            self.partner_offset_hz = np.zeros(self.n_nodes, dtype=np.float64)
        # Equivalent phase ramps, cached by (span count, span length), and the
        # unit phase factors, cached by (span count, span length, accumulated
        # length over span length).  All of them are pure functions of the
        # fibre constants and of those two or three numbers, so the caches
        # are idempotent.
        self._ramp_cache: dict[tuple[int, float], tuple[NDArrayFloat, NDArrayFloat]] = {}
        self._factor_cache: dict[tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]] = {}
        self._visibility_cache: dict[tuple[int, float], tuple[NDArrayFloat, NDArrayFloat]] = {}
        # Permutation from the internal order (descending frequency) back to
        # the model order.  It is the inverse of ``self.order`` and is built
        # once, because the hot path needs it for every span.
        self.inverse_order = np.empty(self.n_nodes, dtype=np.int64)
        self.inverse_order[self.order] = np.arange(self.n_nodes)

        # --- island constants ------------------------------------------------
        # Effective dispersion of the SCI island, Eq. (A.15) with
        # m = k = n = CUT.
        df_ref = self.f_abs_hz - self.f_ref_hz
        self.beta2_eff_sci = (
            self.beta2
            + np.pi * self.beta3 * 2.0 * df_ref
            + (2.0 / 3.0) * np.pi**2 * self.beta4 * 3.0 * df_ref**2
        )
        # Effective dispersion of the XCI island, Eq. (A.15) with
        # m = interferer and k = CUT; the last term of Eq. (A.15) vanishes
        # because one of the two island frequencies is the CUT frequency.
        df_i = self.f_abs_hz[:, None] - self.f_ref_hz
        df_j = self.f_abs_hz[None, :] - self.f_ref_hz
        self.beta2_eff_xci = (
            self.beta2
            + np.pi * self.beta3 * (self.f_abs_hz[:, None] + self.f_abs_hz[None, :] - 2.0 * self.f_ref_hz)
            + (2.0 / 3.0)
            * np.pi**2
            * self.beta4
            * (df_i**2 + df_i * df_j + df_j**2)
        )
        # Nonlinear coefficient of the XCI islands, Eq. (16): the cross
        # effective area averages the areas of the CUT and of the interferer.
        a_sum = self.a_eff_m2[:, None] + self.a_eff_m2[None, :]
        self.gamma_xci = (2.0 * np.pi * self.f_abs_hz[:, None] / C_LIGHT) * (
            2.0 * self.n2_m2_w / a_sum
        )
        # Nonlinear coefficient of the SCI islands, Eq. (A.8) with the four
        # effective areas equal to the area of the CUT.
        self.gamma_sci = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (
            self.n2_m2_w / self.a_eff_m2
        )
        # Log term of the XCI kernel, Eq. (24).  The bandwidth of the
        # interfering channel enters the log.
        delta = self.f_abs_hz[None, :] - self.f_abs_hz[:, None]
        half_b = 0.5 * self.b_hz[None, :]
        ratio = (delta + half_b) / (delta - half_b)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.abs(np.log(np.abs(ratio)))
        # Overlapping channels (|f_nch - f_CUT| < B_nch/2) are outside the
        # validity of the stretched-island closed form: the log term of
        # Eq. (24) does not exist there because the stretched island integral
        # diverges.  Such a comb is not physical, and the term is masked out.
        overlap = np.abs(delta) <= half_b
        log_term = np.where(overlap, 0.0, log_term)
        self.log_xci = log_term
        # Closed-form XCI kernel factor of Eq. (24): the logarithmic term over
        # the effective dispersion of the island.
        self.k_xci_matrix = self.log_xci / (2.0 * np.pi * np.abs(self.beta2_eff_xci))
        np.fill_diagonal(self.k_xci_matrix, 0.0)
        # Assembled XCI weight matrix: gamma^2 of Eq. (21) times the kernel
        # factor of Eq. (24).
        self.w_xci = self.gamma_xci**2 * self.k_xci_matrix

        # Scaled polynomial coefficient operator of the SCI kernel.
        self.r_matrix = r_coefficient_matrix(self.n_poly)

        # --- ISRS coupling ---------------------------------------------------
        self.coef_mat = _isrs_coefficient_matrix_jit(self.f_abs_hz)

        # --- lumped loss -----------------------------------------------------
        self.lumped_loss = self._check_lumped_loss(lumped_loss)
        self._pinv_cache: dict[int, NDArrayFloat] = {}
        self._sample_pos_cache: dict[int, NDArrayFloat] = {}
        self._special_cache: dict[int, tuple[NDArrayFloat, ...]] = {}

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _check_lumped_loss(
        lumped_loss: Sequence[tuple[float, float]] | None,
    ) -> list[tuple[float, float]]:
        """Validate the lumped loss elements of the span."""
        if not lumped_loss:
            return []
        checked: list[tuple[float, float]] = []
        for position_m, loss_db in lumped_loss:
            position = float(position_m)
            loss = float(loss_db)
            if position <= 0.0:
                raise ValueError(f"lumped loss position must be positive, got {position}.")
            if loss < 0.0:
                raise ValueError(f"lumped loss must not be a gain, got {loss} dB.")
            checked.append((position, loss))
        checked.sort(key=lambda item: item[0])
        return checked

    def _span_segments(self, span_length_m: float) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Segment lengths, loss factors and step counts of the span.

        A span without lumped loss is a single segment; every lumped loss
        element splits the span, so that the power profile has a genuine
        discontinuity at its position (paper, Section IV-D).
        """
        edges = [0.0]
        losses = [1.0]
        for position, loss_db in self.lumped_loss:
            if position >= span_length_m:
                raise ValueError(
                    f"lumped loss at {position} m lies outside the span of {span_length_m} m."
                )
            edges.append(position)
            losses.append(10.0 ** (-loss_db / 10.0))
        edges.append(span_length_m)
        lengths = np.array(
            [edges[i + 1] - edges[i] for i in range(len(edges) - 1)],
            dtype=np.float64,
        )
        loss_arr = np.array(losses, dtype=np.float64)
        steps = np.maximum(
            1,
            np.round(self.spp_steps * lengths / float(span_length_m)).astype(np.int64),
        )
        return lengths, loss_arr, steps.astype(np.float64)

    def _sample_positions(self, span_length_m: float) -> NDArrayFloat:
        """Normalised positions ``z/L`` of the SPP samples."""
        key = int(round(span_length_m * 1e6))
        cached = self._sample_pos_cache.get(key)
        if cached is not None:
            return cached
        lengths, _, steps = self._span_segments(span_length_m)
        positions: list[float] = []
        edge = 0.0
        for length, n_step in zip(lengths, steps, strict=True):
            dz = length / n_step
            for step in range(int(n_step)):
                positions.append((edge + (step + 0.5) * dz) / span_length_m)
                positions.append((edge + (step + 1.0) * dz) / span_length_m)
            edge += length
        arr = np.asarray(positions, dtype=np.float64)
        self._sample_pos_cache[key] = arr
        return arr

    def _fit_operator(self, span_length_m: float) -> NDArrayFloat:
        """Pseudo-inverse of the polynomial design matrix of the SPP samples."""
        key = int(round(span_length_m * 1e6))
        cached = self._pinv_cache.get(key)
        if cached is not None:
            return cached
        u = self._sample_positions(span_length_m)
        design = np.vander(u, self.n_poly, increasing=True)
        pinv = np.linalg.pinv(design)
        self._pinv_cache[key] = pinv
        return pinv

    def _special_functions(self, span_length_m: float) -> tuple[NDArrayFloat, ...]:
        """Return the cached special functions of the SCI kernel.

        The argument ``X = lambda L`` of every channel depends only on the
        fibre, on the channel plan and on the span length, so the values are
        cached per span length and shared by every span of the same length.
        """
        key = int(round(span_length_m * 1e6))
        cached = self._special_cache.get(key)
        if cached is not None:
            return cached
        x_all = np.ascontiguousarray(
            self.beta2_eff_sci * (np.pi**2) * self.b_hz**2 * float(span_length_m)
        )
        si_all, _ = sici(x_all)
        sin_all = np.ascontiguousarray(np.sin(x_all))
        cos_all = np.ascontiguousarray(np.cos(x_all))
        j_all = _j_batch_jit(x_all, _j_constant())
        cached = (x_all, np.ascontiguousarray(si_all), sin_all, cos_all, j_all)
        self._special_cache[key] = cached
        return cached

    def effective_dispersion_ps2(self, span_length_m: float) -> NDArrayFloat:
        """Accumulated effective dispersion of every channel over one span.

        The value is the one of Eq. (11) of Zefreh et al. 2020: the effective
        dispersion at the channel frequency, times the span length, in ps^2.

        Args:
            span_length_m: Span length (m).

        Returns
        -------
            ``(N,)`` dispersion contribution of the span (ps^2).
        """
        beta2_ps2_per_m = self.beta2_eff_sci * 1.0e24
        return beta2_ps2_per_m * float(span_length_m)

    # -- closed-form kernels (exposed for cross-checking) -----------------
    def spp_samples(self, span_length_m: float, power_in_w: NDArrayFloat) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return the normalised SPP samples and their positions.

        Args:
            span_length_m: Span length (m).
            power_in_w: ``(N,)`` launch power per channel (W), model order.

        Returns
        -------
            ``(positions, profile)``: the normalised positions ``z/L`` and the
            normalised SPP ``p_i(z) = P_i(z)/P_i(0)`` sampled twice per
            integration step.
        """
        power = np.ascontiguousarray(np.asarray(power_in_w, dtype=np.float64)[self.order])
        profile_norm, _ = self._integrate_spp(power, span_length_m)
        return self._sample_positions(span_length_m), profile_norm

    def profile_polynomial(self, span_length_m: float, power_in_w: NDArrayFloat) -> NDArrayFloat:
        """Polynomial representation of the normalised SPP, Eq. (23).

        Args:
            span_length_m: Span length (m).
            power_in_w: ``(N,)`` launch power per channel (W), model order.

        Returns
        -------
            ``(N_p+1, N)`` coefficients in the scaled basis
            ``p_i(z) = sum_n q_n (z/L)^n``.  The coefficients of the z-basis
            of the paper follow as ``p_n = q_n / L^n``.
        """
        power = np.ascontiguousarray(np.asarray(power_in_w, dtype=np.float64)[self.order])
        profile_norm, _ = self._integrate_spp(power, span_length_m)
        pinv = self._fit_operator(span_length_m)
        return pinv @ profile_norm

    def sci_kernel(self, span_length_m: float, q_scaled: NDArrayFloat) -> NDArrayFloat:
        """Closed-form SCI kernel ``K_SCI`` of Eq. (30).

        Args:
            span_length_m: Span length (m).
            q_scaled: ``(N_p+1, N)`` scaled SPP coefficients.

        Returns
        -------
            ``(N,)`` kernel values.
        """
        x_all, si_all, sin_all, cos_all, j_all = self._special_functions(span_length_m)
        q_outer = (q_scaled.T[:, :, None] * q_scaled.T[:, None, :]).reshape(self.n_nodes, -1)
        r_coef = q_outer @ self.r_matrix
        k_sci = np.zeros(self.n_nodes, dtype=np.float64)
        _sci_kernel_batch_jit(
            np.ascontiguousarray(r_coef),
            np.ascontiguousarray(x_all),
            np.ascontiguousarray(si_all),
            np.ascontiguousarray(sin_all),
            np.ascontiguousarray(cos_all),
            np.ascontiguousarray(j_all),
            np.ascontiguousarray(self.beta2_eff_sci),
            float(span_length_m),
            k_sci,
        )
        return k_sci

    def xci_kernel(self, span_length_m: float, q_scaled: NDArrayFloat) -> NDArrayFloat:
        """Closed-form XCI kernel ``K_XCI`` of Eq. (24).

        The kernel does not carry the nonlinearity coefficient ``gamma``: that
        factor belongs to Eq. (21), which assembles the NLI PSD.

        Args:
            span_length_m: Span length (m).
            q_scaled: ``(N_p+1, N)`` scaled SPP coefficients.

        Returns
        -------
            ``(N, N)`` kernel values; row ``i`` is the CUT, column ``j`` the
            interfering channel.
        """
        energy = _profile_energy_jit(np.ascontiguousarray(q_scaled)) * float(span_length_m)
        return self.k_xci_matrix * energy[None, :]

    def _integrate_spp(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Integrate the ISRS power equations and normalise the profile.

        Args:
            power: ``(N,)`` launch power per channel (W), internal order.
            span_length_m: Span length (m).

        Returns
        -------
            ``(profile_norm, power_out)``: the normalised SPP
            ``p_i(z) = P_i(z)/P_i(0)`` sampled twice per integration step and
            the power at the span end.
        """
        lengths, losses, steps = self._span_segments(span_length_m)
        n_samples = 2 * int(steps.sum())
        profile = np.zeros((n_samples, self.n_nodes), dtype=np.float64)
        power_out = np.zeros(self.n_nodes, dtype=np.float64)
        _solve_spp_jit(
            self.coef_mat,
            self.alpha_npm,
            power,
            lengths,
            losses,
            steps,
            profile,
            power_out,
        )
        # A dark channel keeps a passive unit transfer function.
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_norm = np.where(
                power[None, :] > 0.0,
                profile / np.where(power[None, :] > 0.0, power[None, :], 1.0),
                1.0,
            )
        return profile_norm, power_out

    # -- inter-span coherence ---------------------------------------------
    def _visibilities(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute the inter-span visibility of the SPM and XPM parts.

        Args:
            n_spans: Number of spans of the link.
            span_length_m: Span length (m).

        Returns
        -------
            The tuple ``(visibility_spm, visibility_xpm)``, both
            ``(N_spans - 1, N_channels)``, in the internal channel order.
        """
        n = max(1, int(n_spans))
        distances = np.arange(1, n, dtype=np.float64) * float(span_length_m)
        if n <= 1:
            empty = np.zeros((0, self.b_hz.size), dtype=np.float64)
            return empty, empty
        key = (n, float(span_length_m))
        cached = self._visibility_cache.get(key)
        if cached is not None:
            return cached
        # SPM: weighted by the NLI power, see _self_beating_visibility.
        visibility_spm = _self_beating_visibility(
            self.beta2_abs_s2_m,
            self.alpha_visibility_npm,
            self.b_hz,
            float(span_length_m),
            distances,
        )
        # XPM: the mismatch of a cross pair is one to two orders of magnitude
        # above alpha*L, so the pairs are already incoherent across spans;
        # the band average of the numerical model is kept (the delivery
        # report gives the GN-weighted audit of that choice).
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
        self._visibility_cache[key] = (visibility_spm, visibility_xpm)
        return visibility_spm, visibility_xpm

    def _inter_span_ramps(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute the equivalent per-span phase ramps of SPM and XPM.

        The coherent factor ``R`` of the link is inverted into a phase
        increment per span, because the contract layer carries a single
        complex number per channel and span.

        Args:
            n_spans: Number of spans of the link.
            span_length_m: Span length (m).

        Returns
        -------
            The tuple ``(ramp_spm, ramp_xpm)``: the phase increment per span
            (rad) that the route layer needs to reproduce the coherent
            factor of the link, in the internal channel order.
        """
        key = (int(n_spans), float(span_length_m))
        cached = self._ramp_cache.get(key)
        if cached is not None:
            return cached
        n = max(1, int(n_spans))
        visibility_spm, visibility_xpm = self._visibilities(n, span_length_m)
        result = (
            _equivalent_phase_ramp(n, _coherent_factor(n, visibility_spm)),
            _equivalent_phase_ramp(n, _coherent_factor(n, visibility_xpm)),
        )
        self._ramp_cache[key] = result
        return result

    def _span_phase_factors(
        self,
        span_context: SpanContext,
        span_length_m: float,
    ) -> tuple[NDArrayComplex, NDArrayComplex]:
        """Compute the unit phase factors of the NLI amplitudes of one span.

        The phase grows linearly with the length accumulated in the link, so
        the noise generated in span ``k`` differs from the noise generated in
        the first span by the phase mismatch accumulated over ``k L``.  The
        phase therefore comes from the span context only; the model keeps no
        cross-span counter of its own.

        Args:
            span_context: Position of the span in the link.
            span_length_m: Span length (m).

        Returns
        -------
            The tuple ``(factor_spm, factor_xpm)``: the complex numbers
            ``exp(1j phi)`` of the SPM and XPM amplitudes, in the internal
            channel order.
        """
        scale = span_context.cumulative_length_m / float(span_length_m)
        key = (int(span_context.total_spans), float(span_length_m), float(scale))
        cached = self._factor_cache.get(key)
        if cached is None:
            ramp_spm, ramp_xpm = self._inter_span_ramps(
                span_context.total_spans, span_length_m
            )
            cached = (
                np.exp(1j * ramp_spm * scale),
                np.exp(1j * ramp_xpm * scale),
            )
            self._factor_cache[key] = cached
        return cached

    def inter_span_coherent_factor(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return the coherent factors ``R`` of the SPM and XPM parts.

        The factor is the visibility-weighted sum over the span pairs,
        ``R = N + 2 sum_k (N-k) V(k L)``.  ``R = N`` is the incoherent
        accumulation, ``R = N**2`` the coherent one.

        Args:
            n_spans: Number of spans of the link.
            span_length_m: Span length (m).

        Returns
        -------
            The tuple ``(factor_spm, factor_xpm)``, in the model order (the
            order of the noise powers of :class:`SpanFiberOutput`).
        """
        n = max(1, int(n_spans))
        visibility_spm, visibility_xpm = self._visibilities(n, span_length_m)
        factor_spm = _coherent_factor(n, visibility_spm)
        factor_xpm = _coherent_factor(n, visibility_xpm)
        return (
            np.ascontiguousarray(factor_spm[self.inverse_order]),
            np.ascontiguousarray(factor_xpm[self.inverse_order]),
        )

    # -- kernel protocol -------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        *,
        span_index: int = 0,
        beta2_accum_ps2: NDArrayFloat | None = None,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Evaluate one span of the polynomial closed-form model.

        Args:
            power_in_w: ``(N,)`` launch power per channel (W), in model order.
            span_length_m: Span length (m).
            n_effective_spans: Number of identical spans (kept for protocol
                compatibility; the PCFM accumulates the NLI span by span
                through the route layer).
            span_index: Index of the span inside the link, counted from zero.
                Used by the EGN correction.
            beta2_accum_ps2: ``(N,)`` accumulated effective dispersion at the
                input of the span (ps^2).  Used by the EGN correction.
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
        profile_norm, power_out = self._integrate_spp(power, span_length_m)

        # Polynomial representation of the SPP, Eq. (23).  The fit is done in
        # the scaled variable u = z/L, which is the numerically sound basis;
        # the coefficients of the z-basis follow as p_n = q_n / L^n.
        pinv = self._fit_operator(span_length_m)
        q_scaled = pinv @ profile_norm  # (N_p+1, N)

        # --- SCI kernel, Eq. (30) ------------------------------------------
        k_sci = self.sci_kernel(span_length_m, q_scaled)

        # --- NLI PSDs, Eqs. (21) and (26) ----------------------------------
        active = power > 0.0
        psd = np.where(active, power / self.b_hz, 0.0)  # G_WDM,i

        rho_cut = np.ones(self.n_nodes, dtype=np.float64)
        rho_nch = np.ones(self.n_nodes, dtype=np.float64)
        if self.egn:
            if beta2_accum_ps2 is None:
                beta2_accum_ps2 = np.zeros(self.n_nodes, dtype=np.float64)
            rho_cut = egn_rho_cut(beta2_accum_ps2, self.symbol_rate_hz)
            rho_nch = egn_rho_nch(beta2_accum_ps2)

        # SCI: G_SCI = (16/27) gamma^2 Gamma p_CUT(L) G_CUT^3 K_SCI.
        # The outer p_CUT(L) of Eq. (26) is set to one, see the module
        # docstring: the span loss is already carried by the profile that the
        # kernel integrates.
        g_sci = (16.0 / 27.0) * self.gamma_sci**2 * psd**3 * k_sci * rho_cut
        # XCI: G_XCI = (32/27) p_CUT(L) G_CUT sum_nch gamma^2 G_nch^2 K_XCI.
        q_int = _profile_energy_jit(np.ascontiguousarray(q_scaled)) * float(span_length_m)
        xci_weight = q_int * psd**2 * rho_nch
        xci_acc = self.w_xci @ xci_weight
        g_xci = (32.0 / 27.0) * psd * xci_acc

        spm = np.where(active, g_sci * self.b_hz, 0.0)
        xpm = np.where(active, g_xci * self.b_hz, 0.0)

        inverse = self.inverse_order
        n_spans = 1 if span_context is None else max(1, int(span_context.total_spans))
        if span_context is None or n_spans <= 1:
            # Single span (or no context at all): the route layer keeps the
            # pure power convention.  This branch is bit-for-bit identical to
            # the implementation without inter-span coherence.
            return SpanFiberOutput(
                power_out_w=np.ascontiguousarray(power_out[inverse]),
                spm_noise_w=np.ascontiguousarray(spm[inverse]),
                xpm_noise_w=np.ascontiguousarray(xpm[inverse]),
            )
        # Multi-span link: hand the route layer the complex NLI amplitudes.
        # Their modulus is the square root of the noise power of this span
        # and their phase is the inter-span phase, so that the route layer
        # can sum |sum_k a_k|**2.
        factor_spm, factor_xpm = self._span_phase_factors(span_context, float(span_length_m))
        amplitude_spm = np.sqrt(spm) * factor_spm
        amplitude_xpm = np.sqrt(xpm) * factor_xpm
        return SpanFiberOutput(
            power_out_w=np.ascontiguousarray(power_out[inverse]),
            spm_noise_w=np.ascontiguousarray(spm[inverse]),
            xpm_noise_w=np.ascontiguousarray(xpm[inverse]),
            nli_amplitude_spm=np.ascontiguousarray(amplitude_spm[inverse]),
            nli_amplitude_xpm=np.ascontiguousarray(amplitude_xpm[inverse]),
        )


class PCFMModel(RouteModel):
    """Polynomial closed-form model (SCI + XCI) with a polynomial SPP fit.

    Args:
        grid: Spectrum grid (legacy ``SpectrumGrid`` or ``FlexibleGrid``).
        ref_fiber_config: Reference fibre configuration.
        symbol_rate_hz: Symbol rate (Hz).
        poly_degree: Degree ``N_p`` of the SPP polynomial, Eq. (23).  The
            paper finds ``N_p = 5`` adequate and ``N_p = 9`` the reference.
        egn: Apply the machine-learning correction of Zefreh et al. 2020 on
            top of the GN result.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.
    """

    info = PCFM_MODEL_INFO

    #: This model provides the complex NLI amplitudes of the inter-span
    #: coherence, so the route layer sums ``|sum_k a_k|**2`` instead of
    #: ``sum_k |a_k|**2``.
    supports_span_context = True

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        poly_degree: int = 5,
        egn: bool = False,
        **route_options,
    ) -> None:
        """Initialize the model.

        Raises
        ------
            ValueError: ``poly_degree`` is negative.
        """
        if poly_degree < 0:
            raise ValueError(f"poly_degree must be >= 0, got {poly_degree}.")
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.poly_degree = int(poly_degree)
        self.egn = bool(egn)
        self._span_index = 0
        self._beta2_accum_ps2 = np.zeros(self.n_nodes, dtype=np.float64)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> PCFMSpanKernel:
        """Build the single-span kernel of this fibre."""
        return PCFMSpanKernel(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            poly_degree=self.poly_degree,
            symbol_rate_hz=self.symbol_rate_hz,
            egn=self.egn,
        )

    def evaluate_path(self, optical_path, power_in_w):
        """Evaluate a path, resetting the span context of the EGN correction."""
        self._span_index = 0
        self._beta2_accum_ps2 = np.zeros(self.n_nodes, dtype=np.float64)
        return super().evaluate_path(optical_path, power_in_w)

    def span_step(
        self,
        fiber_config,
        edfa_config,
        power_in_w,
        span_length_m,
        span_context: SpanContext | None = None,
    ):
        """Single span: the kernel, then the shared amplifier/ROADM handling.

        The span index and the accumulated effective dispersion are tracked
        here because the machine-learning factor of Zefreh et al. 2020 depends
        on them; both are reset at the beginning of every path.

        Args:
            fiber_config: Fibre parameters of the span.
            edfa_config: Amplifier configuration of the span.
            power_in_w: ``(N,)`` launch power per channel (W).
            span_length_m: Span length (m).
            span_context: Position of the span in the link.  It is handed to
                the kernel, which uses it for the inter-span phase of the NLI
                amplitudes.  ``None`` means the pure power convention.

        Returns
        -------
            The :class:`~network_nlo_eval.models.base.ElementOutput` of the
            span.
        """
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate(
            power_in_w,
            span_length_m,
            1,
            span_index=(
                self._span_index if span_context is None else int(span_context.span_index)
            ),
            beta2_accum_ps2=self._beta2_accum_ps2,
            span_context=span_context,
        )
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )
        self._span_index += 1
        self._beta2_accum_ps2 = self._beta2_accum_ps2 + kernel.effective_dispersion_ps2(
            span_length_m
        )
        return element


register_model(PCFM_MODEL_INFO, PCFMModel)
