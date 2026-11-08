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

**Polarization convention of the NLI kernels**

The ``16/27`` (SCI) and ``32/27`` (XCI = ``2 * 16/27``) prefactors are the
*dual-polarization* weights of the GN model, not single-polarization weights.
The derivation used elsewhere in this repository (``damico2022.SPM_WEIGHT``)
writes them as

.. code-block:: text

    16/27 = (8/9)**2 * 6 / 2**3 ,

where ``(8/9)**2`` is the Manakov nonlinear coefficient and the ``2**3`` is the
fact that each of the three interacting waves carries **half** of the total
channel power.  The kernel therefore expects ``G(f) = P_total / B``: the sum of
both polarizations.  The same convention is used by the numerical GN reference
(``numerical_kernels.py``, ``numerical_dra.py``) and by the waveform reference
``physical-nlo-eval``, whose ``launch_power_w`` is explicitly documented as the
total dual-polarization channel power and whose EDFA ASE is generated per
polarization with a factor ``1/2``.

``nli_power_convention`` makes that assumption explicit.  With ``"total"``
(default) the kernel is bit-for-bit the historical implementation.  With
``"per_pol"`` the input is per-polarization power, so the PSD drive is
``2 * P / B`` and the per-channel NLI is halved on the way out; both steps are
needed to keep the NLI in the same coordinates as the per-polarization signal
and ASE outputs.

The same doubling applies to the **ISRS profile**: the Raman coupling of
:meth:`PCFMSpanKernel._integrate_spp` is a physical quantity driven by the
dual-polarization total power, so ``"per_pol"`` integrates the SPP with
``2 * P`` and halves the span-end power.  Without it the profile is computed
from half the power, and although the signal/NLI ratio barely moves, the
ISRS-induced gain tilt is wrong and the EDFA gain compensation built on it
leaks into the ASE (0.19 dB mean / 1.30 dB worst on C+L unpumped).

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
from dataclasses import dataclass, replace

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import sici

from network_nlo_eval.core.types import NDArrayBool, NDArrayComplex, NDArrayFloat
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
    SCI_X_SERIES_SWITCH,
    _isrs_coefficient_matrix_jit,
    _j_batch_jit,
    _profile_energy_jit,
    _sci_kernel_batch_jit,
    _sci_small_x_coefficients,
    _solve_spp_jit,
    r_coefficient_matrix,
)
from network_nlo_eval.models.raman import RamanAmplifier, RamanGainModel, resolve_raman_gain_model
from network_nlo_eval.models.route import RouteModel, apply_amplifier_and_roadm, resolve_noise_figure
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "DENSE_GRID_SPACING_RATIO",
    "EGN_A_COEFFICIENTS",
    "EGN_PHI_VALUES",
    "NLI_POWER_CONVENTIONS",
    "PCFM_MODEL_INFO",
    "PCFMModel",
    "PCFMSpanKernel",
    "RESOLUTION_PROFILES",
    "SpanValidityReport",
    "dense_grid_metrics",
    "resolve_nli_power_convention",
    "resolve_resolution_profile",
    "TILT_INFORMATIONAL_KNEE_DB",
    "TILT_INFORMATIONAL_ZERO_DB",
    "informational_tilt_confidence",
]

#: Number of RK4 steps of the ISRS power-profile integration.
#:
#: The coupled power equations are integrated with a fixed-step RK4 scheme.
#: The ISRS rate over a 100 km SMF span is a few e-foldings, so the step
#: stays far inside the RK4 stability and accuracy region; 32 steps resolve
#: the SPP of every channel of the benchmark combs to better than 0.1 % (see
#: the report).
DEFAULT_SPP_STEPS: int = 32

#: Explicit, opt-in resolution profiles.  ``None`` (the default) reproduces the
#: historical per-option defaults bit for bit.
#:
#: The values come from the accuracy/runtime sweep in
#: ``docs/h3-pareto-report.md`` (script ``scripts/h3_pareto_sweep.py``), whose
#: reference is ``spp_steps=96``, ``raman_steps=96`` and 64 Gauss-Legendre
#: nodes.  Two facts drive the choice:
#:
#: * the XCI edge quadrature is *free* accuracy: 12 nodes reproduce the
#:   64-node NLI to 1.4e-11 relative on the E-band comb (the only case with
#:   gated rows), while 8 nodes already reach 5.2e-9.  Its cost is a one-time
#:   per-span-length integral, 6-65 % of the first ``evaluate_path``
#:   depending on the band, so lowering it is the only no-regret speed win;
#: * ``spp_steps`` and ``raman_steps`` are *not* free.  On pumped links the
#:   coarse ODE dominates both error and runtime: below 24 steps the backward
#:   fixed point does not converge at all (NaN, and slower because it burns
#:   the iteration cap), 24 steps is 0.3-2.7 dB off, and only 96 steps is
#:   within 0.03 dB of 192.  Lowering them buys nothing that is defensible,
#:   so ``"fast"`` leaves them alone.
RESOLUTION_PROFILES: dict[str, dict[str, int]] = {
    "fast": {"xci_quadrature_points": 12},
    "accurate": {"spp_steps": 96, "raman_steps": 96},
}


def resolve_resolution_profile(resolution: str | None) -> dict[str, int] | None:
    """Resolve an opt-in resolution profile to the knobs it overrides.

    Args:
        resolution: ``None`` or ``"default"`` (historical behaviour),
            ``"fast"`` (drop the XCI edge quadrature to its measured-flat
            value) or ``"accurate"`` (the converged sweep reference).

    Returns
    -------
        A mapping of option name to value, or ``None`` when no profile is
        requested.

    Raises
    ------
        ValueError: ``resolution`` is not one of the accepted values.
    """
    if resolution is None or resolution == "default":
        return None
    if not isinstance(resolution, str) or resolution not in RESOLUTION_PROFILES:
        raise ValueError(
            f"resolution must be None, 'default' or one of {sorted(RESOLUTION_PROFILES)}, got {resolution!r}."
        )
    return dict(RESOLUTION_PROFILES[resolution])


#: Supported NLI power conventions of the closed-form kernels.
#:
#: ``"total"`` (default): ``power_in_w`` is the per-channel *dual-polarization
#: total* power.  The printed ``(16/27)`` / ``(32/27)`` prefactors are used
#: as-is, which is the convention of the paper and of the numerical GN
#: reference of this repository.
#: ``"per_pol"``: ``power_in_w`` is *per-polarization* power.  The power that
#: drives the kernels and the ISRS profile is the physical total
#: ``2 * power_in_w``; the per-channel NLI and the span-end signal power are
#: halved on the way out, so that they stay in the per-polarization
#: coordinates of the signal and ASE outputs.
NLI_POWER_CONVENTIONS: tuple[str, ...] = ("total", "per_pol")


def resolve_nli_power_convention(nli_power_convention: str) -> str:
    """Validate the NLI power convention.

    The two GN prefactors are not symmetric in the input power: SCI goes as
    ``psd**3`` and XCI as ``psd**3`` as well, so feeding a per-polarization
    power to a kernel that was derived for the total PSD scales the NLI by
    ``2**3 = 8``.  The convention therefore has to be declared, exactly like
    the launch-power convention of the coupled Raman ODE.

    Args:
        nli_power_convention: ``"total"`` or ``"per_pol"``.

    Returns
    -------
        The normalized convention name.

    Raises
    ------
        ValueError: The value is not one of :data:`NLI_POWER_CONVENTIONS`.
    """
    convention = str(nli_power_convention)
    if convention not in NLI_POWER_CONVENTIONS:
        raise ValueError(
            f"nli_power_convention must be one of {NLI_POWER_CONVENTIONS}, got {nli_power_convention!r}."
        )
    return convention


#: Dimensionless validity parameter of the XCI island closed form below which
#: the exact island integral is evaluated for the pair.
#:
#: The printed XCI kernel, Eq. (24), is the asymptotic limit
#: ``s = 4 pi**2 |beta2_eff| y B / (2 alpha) -> infinity`` of the frequency
#: integral of the GN weight.  At ``s >> 1`` the weight is phase-mismatch
#: limited and the closed form is accurate; at ``s ~ 1`` it overestimates the
#: kernel like ``1/|beta2_eff|`` instead of saturating at the coherent limit.
#: The E-band edge of the benchmark reaches ``s = 0.019`` at the zero
#: dispersion crossing inside the comb (about 1367 nm on SMF), where the closed
#: form is 12 dB high.  Pairs below this switch are recomputed from the exact
#: island integral (see :func:`_xci_edge_weight_matrix`).
#:
#: The value is in the same units as the integral's ``s`` (returned by
#: :func:`_xci_validity_s`).  The previous code compared a half-``s``
#: diagnostic against 100; the corrected expression returns the full ``s``, so
#: 200 is the equivalent gate and leaves the gated rows bit-for-bit unchanged
#: on C / C+L / S+C+L at 40/60/80/100 km and on the E-band comb at 40/100 km
#: (see ``tests/test_pcfm_xci_norm.py``).
XCI_EDGE_VALIDITY_SWITCH: float = 200.0

#: Materiality threshold of the XCI edge correction, per CUT row.
#:
#: The exact correction is applied to a CUT only when it changes the total XCI
#: weight of that row by more than this fraction.  C/L/S+C+L rows change by
#: less than 1.1 % and keep the old, bit-identical path; the E-band rows next
#: to the zero-dispersion crossing change by up to 91 %.  The threshold is
#: therefore a physical applicability gate, not a tuning knob: below it the
#: asymptotic breakdown is below the model's error budget (3 % of the XCI
#: power is 0.13 dB, inside the C/L XCI error of the closed form against the
#: numerical reference).
XCI_EDGE_MATERIALITY: float = 0.03

#: Gauss-Legendre nodes of the exact XCI island integral (per flagged pair).
XCI_EDGE_QUADRATURE_POINTS: int = 24

#: Channel-power tilt above which the span is reported as outside the
#: validated operating range.
#:
#: The base PCFM flags the launch-power tilt ``10 log10(max/min)`` over the
#: active channels.  The H3 Raman kernel additionally folds in the span-gain
#: tilt, because there the pump profile is what the polynomial representation
#: cannot bound.  The threshold follows the strong-tilt audit
#: (``dra-open-issues.md`` Section 9): at a 15 dB gain tilt the NLI residual is
#: dominated by the *profile representation* rather than by the GN kernel.
#: The ordinary C+L six-pump benchmark stays at 14.3 dB and is not flagged;
#: the pump-depleted stress case is.  The unpumped E+S+C+L comb has 19.2 dB of
#: ISRS tilt, which is reported in ``gain_tilt_db`` but does not flag the
#: unpumped kernel: the ISRS profile is solved by the model itself.
POWER_TILT_VALIDITY_DB: float = 15.0

#: Low knee of the *informational* continuous tilt confidence (dB).
#:
#: The hard flag above stays at :data:`POWER_TILT_VALIDITY_DB`; this knee only
#: shapes an additive, non-gating score.  The value comes from the tilt audit
#: (``scripts/h3_tilt_isrs_audit.py``): between 0 and 10 dB the measured
#: representation residual of the default H3 fit against its converged form is
#: below 0.1 dB for both an unpumped launch tilt and a pumped gain tilt, while
#: at the ~14 dB gain tilt of the C+L six-pump benchmark it reaches 0.15 dB.
#: Starting the decay at 10 dB therefore gives downstream margin *before* the
#: hard flag fires, without moving any default value.
TILT_INFORMATIONAL_KNEE_DB: float = 10.0

#: Tilt at which the informational confidence reaches zero (dB).
#:
#: Kept equal to the zero of the hard map (``2 * POWER_TILT_VALIDITY_DB``) so
#: the two curves share their end point: 30 dB is twice the flag threshold and
#: no audited configuration comes close to it.
TILT_INFORMATIONAL_ZERO_DB: float = 2.0 * POWER_TILT_VALIDITY_DB


def informational_tilt_confidence(tilt_db: float) -> float:
    r"""Continuous, non-gating confidence as a function of power/gain tilt.

    The hard validity flag is a step at :data:`POWER_TILT_VALIDITY_DB`; the
    ``confidence`` field already decays linearly *above* it.  This auxiliary
    curve additionally decays *below* the threshold, so a network layer can
    apply a soft margin without changing ``valid``/``flagged``.

    The curve is a cubic smoothstep from :data:`TILT_INFORMATIONAL_KNEE_DB`
    (score 1) to :data:`TILT_INFORMATIONAL_ZERO_DB` (score 0):

    .. math::

        x = \\mathrm{clip}\\left(\\frac{t - t_{knee}}{t_{zero} - t_{knee}}, 0, 1\\right),
        \\qquad c = 1 - x^2 (3 - 2x).

    It is monotonically non-increasing and C1 at both ends, so a caller can
    interpolate it without kinks.  The knee is placed where the audit measures
    the default H3 fit to be within 0.1 dB of its converged form; it is
    deliberately *not* a second validity threshold.

    Args:
        tilt_db: Tilt ``10 log10(max/min)`` in dB.  Values below the knee
            return 1; values at or above the zero return 0.

    Returns
    -------
        Score in ``[0, 1]``.
    """
    tilt = float(tilt_db)
    if not np.isfinite(tilt):
        return 0.0
    span = TILT_INFORMATIONAL_ZERO_DB - TILT_INFORMATIONAL_KNEE_DB
    x = float(np.clip((tilt - TILT_INFORMATIONAL_KNEE_DB) / span, 0.0, 1.0))
    return float(1.0 - x * x * (3.0 - 2.0 * x))


#: Confidence assigned to a channel whose XCI row is corrected by the edge
#: gate.  The exact island integral makes the value usable, but the channel
#: still sits in the regime where the printed closed form is not asymptotic.
XCI_GATED_CONFIDENCE: float = 0.6

#: Confidence assigned to a channel with a material XCI correction but no
#: gate active (``edge_fix=False``).  The reported value is known to be wrong
#: by more than the model error budget.
XCI_UNCORRECTED_CONFIDENCE: float = 0.2

#: Confidence assigned to an XCI row that is below the raw ``s`` switch but
#: whose exact correction is immaterial (below the 3 % row gate).
XCI_IMMATERIAL_CONFIDENCE: float = 0.9

#: Confidence assigned to a channel using the small-``X`` SCI Taylor branch.
SCI_TAYLOR_CONFIDENCE: float = 0.6

#: Confidence assigned to a channel below the SCI switch without the branch.
SCI_UNCORRECTED_CONFIDENCE: float = 0.2

#: Channel spacing below this multiple of the occupied channel bandwidth means
#: neighbouring rectangular spectra overlap.
#:
#: Every closed-form kernel in this module derives the XCI row from the
#: stretched-island integral of two *disjoint* rectangular spectra.  When the
#: spacing falls below the bandwidth the guard band disappears and the
#: derivation no longer describes the evaluated comb: the measured H3 optimism
#: is 1.5 dB at 64 GBaud / 50 GHz (ratio 0.78) and 2.5 dB at 37.5 GHz
#: (ratio 0.59), while 75 GHz (1.17) and 100 GHz (1.56) stay inside the
#: narrow-band budget.  The ratio is therefore a *domain* boundary, not a
#: calibrated margin, and the report only flags it.
DENSE_GRID_SPACING_RATIO: float = 1.0


def dense_grid_metrics(
    f_abs_hz: NDArrayFloat,
    bandwidth_hz: NDArrayFloat,
    active: NDArrayBool | None = None,
) -> tuple[float, float]:
    """Return the closest channel spacing and its ratio to the widest channel.

    Only active channels are considered: an unlaunched channel does not
    overlap anything.  The ratio is ``min spacing / max bandwidth``; values
    below :data:`DENSE_GRID_SPACING_RATIO` mean adjacent spectra overlap.  A
    plan with fewer than two active channels (or a zero bandwidth) reports
    ``inf`` so it can never be mistaken for a dense grid.

    Args:
        f_abs_hz: ``(N,)`` channel centre frequencies (Hz), any order.
        bandwidth_hz: ``(N,)`` occupied channel bandwidths (Hz).
        active: Optional ``(N,)`` boolean mask; ``None`` means every channel.

    Returns
    -------
        ``(min_spacing_hz, spacing_to_bandwidth)``; both ``inf`` when there is
        no pair to compare.

    Raises
    ------
        ValueError: The arrays are not one-dimensional or differ in shape.
    """
    frequencies = np.asarray(f_abs_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidth_hz, dtype=np.float64)
    if frequencies.ndim != 1 or bandwidths.shape != frequencies.shape:
        raise ValueError(
            f"f_abs_hz and bandwidth_hz must be one-dimensional and equal, got {frequencies.shape} "
            f"and {bandwidths.shape}."
        )
    mask = np.ones(frequencies.shape, dtype=bool) if active is None else np.asarray(active, dtype=bool)
    if mask.shape != frequencies.shape:
        raise ValueError(f"active must have shape {frequencies.shape}, got {mask.shape}.")
    selected = np.sort(frequencies[mask])
    if selected.size < 2:
        return float("inf"), float("inf")
    min_spacing = float(np.min(np.diff(selected)))
    max_bandwidth = float(np.max(bandwidths[mask]))
    if max_bandwidth <= 0.0:
        return min_spacing, float("inf")
    return min_spacing, min_spacing / max_bandwidth


@dataclass(frozen=True)
class SpanValidityReport:
    """Per-channel applicability report of one PCFM span evaluation.

    The report is the diagnostic counterpart of the two-tier ``edge_fix``:
    it states, channel by channel, whether the closed-form kernels are inside
    the domain where they were validated, and how much the edge machinery had
    to intervene.  It never changes the NLI, ASE or power values; the model
    layer decides what to do with it (``validity_mode``).

    All per-channel arrays are in **model order** (the order of the launch
    power and of :class:`~network_nlo_eval.models.base.PathQoTResult`).

    Attributes
    ----------
        xci_gated:
            ``(N,)`` boolean, ``True`` where the exact XCI island integral was
            actually applied (``edge_fix=True`` and the row correction is
            above :data:`XCI_EDGE_MATERIALITY`).
        xci_row_correction:
            ``(N,)`` relative change of the row's total XCI weight produced by
            the exact island integral, before the materiality gate.  It is
            computed even when ``edge_fix=False`` so that the diagnostic can
            say how wrong the uncorrected row is.
        xci_s_min:
            ``(N,)`` minimum of the dimensionless phase-mismatch parameter
            ``s = 2 pi**2 |beta2_eff| y B_CUT / alpha`` over the interfering
            channels of the row.  This is the ``s`` of the exact island
            integral itself (``4 pi**2 |beta2| y (B_CUT/2) / alpha``), not the
            half-value that older diagnostics printed.  ``inf`` for a row with
            no valid partner.  Values below
            :data:`XCI_EDGE_VALIDITY_SWITCH` mean the printed asymptotic
            kernel is outside its derivation domain; the row is only
            *materially* wrong when ``xci_row_correction`` also exceeds
            :data:`XCI_EDGE_MATERIALITY`.  The two conditions differ on the
            validated C/L bands, where many pairs are below the raw switch
            but the exact correction moves the row by <2.3 %.
        xci_s_below_switch:
            ``(N,)`` boolean, ``xci_s_min < XCI_EDGE_VALIDITY_SWITCH``.
        sci_x:
            ``(N,)`` signed ``X = pi**2 beta2_eff_sci B**2 L`` of the SCI
            kernel.  ``|X|`` is the distance from the zero-dispersion
            crossing of the channel.
        sci_taylor:
            ``(N,)`` boolean, the small-``X`` Taylor branch was used (only
            possible with ``edge_fix=True``).
        sci_x_below_switch:
            ``(N,)`` boolean, ``|sci_x| < SCI_X_SERIES_SWITCH``.  ``True``
            with ``sci_taylor=False`` means the unstable recurrence was used.
        sci_x_abs_min:
            Minimum ``|X|`` over the active channels (dimensionless).  ``0``
            when no channel is active.
        power_tilt_db:
            Launch-power tilt ``10 log10(max P / min P)`` over the active
            channels (dB).
        gain_tilt_db:
            Span-gain tilt ``10 log10(max g / min g)`` with
            ``g = P_out/P_in`` over the active channels (dB).  This is the
            pump/ISRS gain tilt seen by the NLI profile.
        pump_active:
            Whether the span carries at least one positive-power pump.
        pump_directions:
            Directions of the active pumps, in the order of the pump list.
            Empty when no pump is active.
        pump_depletion:
            ``(N_p,)`` depletion fraction ``1 - P_out/P_passive`` of every
            active pump, where ``P_passive = P_in exp(-alpha_p L)`` is the
            undepleted (own-attenuation) output.  Measuring against the
            passive output keeps the 25 dB intrinsic pump attenuation out of
            the number; a value near one means the signals consumed most of
            the pump.  Forward pumps are measured at ``z=L`` and backward
            pumps at ``z=0``, following
            :class:`~network_nlo_eval.models.raman_ode.PaperRamanSpanResult`.
        pump_depletion_max:
            Largest entry of ``pump_depletion`` (``0`` without pumps).
        flagged:
            ``(N,)`` boolean, the channel is in a regime the two-tier fix
            had to correct (``xci_row_correction > XCI_EDGE_MATERIALITY`` or
            ``sci_x_below_switch``) or the span exceeds the tilt threshold.
        valid:
            ``(N,)`` boolean, the reported NLI is expected inside the model
            error budget.  A corrected edge row is still valid (the exact
            integral is applied); an uncorrected one is not.
        confidence:
            ``(N,)`` score in ``[0, 1]``.  It is a coarse ordering, not a
            calibrated probability: 1 means inside the validated domain,
            ~0.6 means "usable after the fix", ~0.2 means "known wrong".
        effective_tilt_db:
            The tilt actually compared against
            :data:`POWER_TILT_VALIDITY_DB` (dB).  For the base kernel this is
            ``power_tilt_db``; the H3 Raman kernel uses
            ``max(power_tilt_db, gain_tilt_db)`` when pumps are active.  It
            makes the flag decision readable without re-deriving the rule.
        tilt_confidence_informational:
            Continuous, **non-gating** tilt score in ``[0, 1]`` from
            :func:`informational_tilt_confidence`.  It is 1 below
            :data:`TILT_INFORMATIONAL_KNEE_DB` and decays smoothly to 0 at
            :data:`TILT_INFORMATIONAL_ZERO_DB`, so a caller can apply a soft
            margin before the hard flag fires.  It never changes ``valid``,
            ``flagged`` or ``confidence``.
        min_channel_spacing_hz:
            Closest spacing between two active channels (Hz); ``inf`` when the
            span carries fewer than two active channels.
        spacing_to_bandwidth:
            ``min_channel_spacing_hz / max bandwidth`` over the active
            channels.  A value below :data:`DENSE_GRID_SPACING_RATIO` means
            adjacent rectangular spectra overlap and the XCI closed form is
            outside its derivation domain.
        dense_grid:
            ``True`` when ``spacing_to_bandwidth`` is below
            :data:`DENSE_GRID_SPACING_RATIO`.  It is a domain warning only:
            it ORs into ``flagged`` for every active channel but never changes
            ``valid``, ``confidence`` or the NLI itself.
        raman_converged:
            Whether the coupled Raman solve behind the report converged.
            ``False`` also invalidates every active channel: the NLI profile
            is not a solution.  Always ``True`` for the no-pump/base kernel.
        raman_steps:
            RK4 steps actually used by the coupled solve (``0`` without a
            Raman solve).  It makes an explicitly under-resolved kernel
            visible even when the solve happens to converge.
        raman_wideband_multipump:
            ``True`` when the comb reaches S/E spectrum *and* the pump set is
            forward or multi-pump.  ``raman_steps="auto"`` uses the extended
            96-step branch there; the flag is reported so a caller can see
            that the default needed it.
        lumped_raman_approximation:
            ``True`` for the pure PCFM kernel with ``raman_model="lumped"``
            and active pumps.  The lumped model multiplies only the span-end
            power and does not model pump depletion or the intra-span profile,
            so it is not a trustworthy QoT for pumped links.  It never changes
            the point result; the error budget refuses it.
    """

    xci_gated: NDArrayFloat
    xci_row_correction: NDArrayFloat
    xci_s_min: NDArrayFloat
    xci_s_below_switch: NDArrayFloat
    sci_x: NDArrayFloat
    sci_taylor: NDArrayFloat
    sci_x_below_switch: NDArrayFloat
    sci_x_abs_min: float
    power_tilt_db: float
    gain_tilt_db: float
    pump_active: bool
    pump_directions: tuple[str, ...]
    pump_depletion: NDArrayFloat
    pump_depletion_max: float
    flagged: NDArrayFloat
    valid: NDArrayFloat
    confidence: NDArrayFloat
    effective_tilt_db: float = 0.0
    tilt_confidence_informational: float = 1.0
    min_channel_spacing_hz: float = float("inf")
    spacing_to_bandwidth: float = float("inf")
    dense_grid: bool = False
    raman_converged: bool = True
    raman_steps: int = 0
    raman_wideband_multipump: bool = False
    lumped_raman_approximation: bool = False


#: Relative power tolerance under which two span inputs are treated as the
#: same state by the span memoisation.
#:
#: Transparent spans differ only by the last bits of the EDFA gain division
#: (``2.2e-16`` relative on the benchmark), far below the ``0.01`` dB accuracy
#: the cache must preserve.  A tighter tolerance would make every span a miss
#: and remove the speedup; ``1e-12`` keeps the physics indistinguishable while
#: still recognising the identical state.
SPAN_CACHE_RTOL: float = 1.0e-12

#: Maximum number of span states kept by one kernel's memoisation.
SPAN_CACHE_MAX_ENTRIES: int = 8

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
    "a4": -1.8309e1,
    "a5": +1.6665e0,
    "a6": -1.0020e0,
    "a7": +9.0933e0,
    "a8": +6.6420e-3,
    "a9": +8.4481e-1,
    "a10": -1.8530e0,
    "a11": +9.4539e-1,
    "a12": -1.5421e1,
    "a13": +1.0229e0,
    "a14": -1.1440e0,
    "a15": +1.1393e-2,
    "a16": +3.8070e5,
    "a17": +1.4785e3,
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
    inner = a["a1"] + a["a2"] * phi ** a["a3"] + a["a4"] * phi ** a["a5"] * (1.0 + a["a6"] * (acc + a["a7"]) ** a["a8"])
    outer = 1.0 + a["a19"] * roll_off_cut ** a["a20"] + a["a21"] * roll_off_nch ** a["a22"]
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
    inner = (
        a["a9"]
        + a["a10"] * phi ** a["a11"]
        + a["a12"]
        * phi ** a["a13"]
        * (1.0 + a["a14"] * rate_tbaud ** a["a15"] + a["a16"] * (acc + a["a17"]) ** a["a18"])
    )
    outer = 1.0 + a["a23"] * roll_off_cut ** a["a24"]
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
    def _cumulative(values: NDArrayFloat, derivative: NDArrayFloat, h: float) -> NDArrayFloat:
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
        cosine_hi = (1.0 + exp_alpha_l**2) * tables.C(k, t_hi) - exp_alpha_l * (
            tables.C(abs(k - 1), t_hi) + tables.C(k + 1, t_hi)
        )
        cosine_lo = (1.0 + exp_alpha_l**2) * tables.C(k, t_lo) - exp_alpha_l * (
            tables.C(abs(k - 1), t_lo) + tables.C(k + 1, t_lo)
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
        integrand = numerator_factor / (abar * (1.0 + mismatch * mismatch)) / (rate * offset)
        numerator += offset_weight * float(np.sum(weights * integrand * np.cos(theta * distance_m / span_length_m)))
        denominator += offset_weight * float(np.sum(weights * integrand))
    return numerator / denominator


def _xci_validity_s(
    beta2_eff_xci: NDArrayFloat,
    y2_hz: NDArrayFloat,
    half_b_cut_hz: NDArrayFloat,
    alpha_npm_m: float,
) -> NDArrayFloat:
    r"""Dimensionless phase-mismatch parameter ``s`` of the XCI closed form.

    The printed kernel, Eq. (24), is the ``s >> 1`` limit of the GN frequency
    integral with

    .. math::

        s = 4 \pi^2 |\beta_{2,eff}| y (B_{CUT} / 2) / \alpha
          = 2 \pi^2 |\beta_{2,eff}| y B_{CUT} / \alpha ,

    evaluated at the upper edge ``y = y2`` of the interfering band.  The
    argument is the half bandwidth, so the coefficient is ``4 pi**2``; this is
    the same ``s`` that the exact island integral of
    :func:`_xci_edge_weight_matrix` uses internally (there
    ``c = 4 pi**2 |beta2|`` multiplies ``y (B_CUT/2)``).  The expression is
    shared by the edge gate and the validity report, so a change of the gate
    definition cannot silently desynchronise the diagnostic.

    Args:
        beta2_eff_xci: ``(N, N)`` effective dispersion of the pair (s^2/m).
        y2_hz: ``(N, N)`` upper edge of the interfering band relative to the
            CUT centre (Hz).
        half_b_cut_hz: ``(N,)`` half bandwidth of the CUT channel (Hz).
        alpha_npm_m: Fibre power attenuation (Np/m).

    Returns
    -------
        ``(N, N)`` parameter ``s`` of every pair.
    """
    # 4 pi**2 而不是 2 pi**2：入参是半带宽 B/2，积分内部同样是
    # 4 pi**2 |beta2| y (B/2)/alpha。旧实现乘 2 pi**2 只返回真值的一半。
    return 4.0 * np.pi**2 * np.abs(beta2_eff_xci) * y2_hz * half_b_cut_hz[:, None] / alpha_npm_m


def _xci_edge_weight_matrix(
    w_xci: NDArrayFloat,
    k_xci_matrix: NDArrayFloat,
    log_xci: NDArrayFloat,
    gamma_xci: NDArrayFloat,
    beta2_eff_xci: NDArrayFloat,
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm_m: float,
    length_m: float,
    validity_switch: float = XCI_EDGE_VALIDITY_SWITCH,
    materiality: float = XCI_EDGE_MATERIALITY,
    n_quad: int = XCI_EDGE_QUADRATURE_POINTS,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    r"""Replace the XCI kernel by the exact island integral where it matters.

    Eq. (24) of the paper is the asymptotic form of the XCI frequency integral
    for ``s = 4 pi**2 |beta2_eff| y B_CUT / (2 alpha) >> 1``.  Its kernel
    carries ``|log(...)| / (2 pi |beta2_eff|)``: as the effective dispersion of
    a pair falls below the phase-matching scale, the closed form grows like
    ``1/|beta2_eff|`` instead of saturating at the coherent limit.  On an
    ultra-wideband comb the E-band edge reaches ``s = 0.019`` near the
    zero-dispersion wavelength, and the channel there is 12 dB high.

    This function restores the exact frequency integral for the pairs below
    the validity switch.  With the GN weight
    ``W(rho) = |1 - exp(-(alpha + j rho) L)|**2 / (alpha**2 + rho**2)`` and
    the offsets ``x`` (CUT band) and ``y`` (interfering band) the integral is

    .. code-block:: text

        I = int_{y1}^{y2} dy (2 / (4 pi**2 |beta2| y))
            int_0^{4 pi**2 |beta2| y B_CUT/2} W(t) dt ,

    whose asymptotic limit (``t -> infinity``) is exactly the printed closed
    form.  The antiderivative of ``W`` is the same Lorentzian table that the
    inter-span visibility uses:

    .. code-block:: text

        F(s) = (1/alpha) [ (1 + e**(-2 ab)) arctan(s)
                           - 2 e**(-ab) C_1(s) ] ,   s = t/alpha ,

    and the corrected kernel is ``K = (pi / F_inf) I`` with
    ``F_inf = int_0^inf W dt = (pi / (2 alpha)) (1 - e**(-2 ab))``.  The ratio
    to the closed form contains no fitted constant: it is one at
    ``s -> infinity`` and finite (not divergent) at ``beta2 = 0``.

    The correction is applied per CUT row and only when it moves the row's
    total XCI weight by more than ``materiality``.  Rows that pass the gate
    keep the old matrix bit for bit, so the validated C/L/S+C+L cases do not
    move; the gate is a statement of where the asymptotic breakdown is above
    the model error budget.

    Args:
        w_xci: ``(N, N)`` closed-form XCI weight matrix, Eq. (24).
        k_xci_matrix: ``(N, N)`` kernel factor ``log / (2 pi |beta2_eff|)``.
        log_xci: ``(N, N)`` logarithmic frequency factor of Eq. (24).
        gamma_xci: ``(N, N)`` nonlinear coefficient of the pair.
        beta2_eff_xci: ``(N, N)`` effective dispersion of the pair (s^2/m).
        f_abs_hz: ``(N,)`` channel centre frequencies (Hz).
        b_hz: ``(N,)`` channel bandwidths (Hz).
        alpha_npm_m: Fibre power attenuation (Np/m).
        length_m: Span length (m).
        validity_switch: Pairs with ``s`` below this value are recomputed.
        materiality: Row-level relative change above which the correction is
            applied.
        n_quad: Gauss-Legendre nodes of the ``y`` integral.

    Returns
    -------
        The tuple ``(w_edge, gated, fraction)``: the corrected weight matrix,
        the boolean mask of the corrected CUT rows, and the relative change of
        every row before gating.
    """
    n_ch = f_abs_hz.size
    delta = np.abs(f_abs_hz[:, None] - f_abs_hz[None, :])
    half_b = 0.5 * b_hz
    y1 = delta - half_b[None, :]
    y2 = delta + half_b[None, :]
    # 适用域判据：相位失配相对线宽的量级 s（积分内部口径，非半值）。用干扰
    # 信道带上边缘 y2 与 CUT 半带宽计算，与 validity 报告共用同一表达式。
    s_edge = _xci_validity_s(beta2_eff_xci, y2, half_b, alpha_npm_m)
    valid = (log_xci != 0.0) & (y1 > 0.0) & (s_edge < validity_switch)

    w_edge = np.array(w_xci, dtype=np.float64, copy=True)
    gated = np.zeros(n_ch, dtype=bool)
    fraction = np.zeros(n_ch, dtype=np.float64)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return w_edge, gated, fraction

    b_abs = np.abs(beta2_eff_xci[rows, cols])
    b_cut = b_hz[rows]
    b_nch = b_hz[cols]
    y_lo = y1[rows, cols]
    y_hi = y2[rows, cols]
    c_mix = 4.0 * np.pi**2 * b_abs
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))  # type: ignore[no-untyped-call]
    y = 0.5 * (y_hi - y_lo)[:, None] * nodes[None, :] + 0.5 * (y_hi + y_lo)[:, None]

    ab = alpha_npm_m * float(length_m)
    exp_ab = float(np.exp(-ab))
    exp_2ab = float(np.exp(-2.0 * ab))
    # 覆盖所有有效 s 的查表范围；超出的尾部由类的渐近展开处理。
    max_s = float(np.max(np.abs(c_mix[:, None] * y * (0.5 * b_cut)[:, None] / alpha_npm_m)))
    table = _LorentzianAntiderivative(alpha_npm_m, float(length_m), 1, max_s)
    s = c_mix[:, None] * y * (0.5 * b_cut)[:, None] / alpha_npm_m
    # s 很小时 arctan(s) 与 C_1(s) 几乎相等，组合式会损失有效数字；用极限。
    small = np.abs(s) < 1.0e-4
    c1 = table.C(1, s)
    bracket = (1.0 + exp_2ab) * np.arctan(s) - 2.0 * exp_ab * c1
    bracket = np.where(small, (1.0 - exp_ab) ** 2 * s, bracket)
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = 2.0 * bracket / (alpha_npm_m * c_mix[:, None] * y)
    # Gauss-Legendre 权重作用在 [-1, 1] 上，乘以 y 区间的雅可比。
    integral = np.sum(0.5 * (y_hi - y_lo)[:, None] * weights[None, :] * integrand, axis=1)

    # 精确积分的总权重：∫_0^∞ W dt = (π/(2α)) (1 - e^{-2αL})。用
    # (1 - e^{-αL})² 会在短跨上把修正核放大 (1+e^{-αL})/(1-e^{-αL}) 倍。
    f_inf = (1.0 / alpha_npm_m) * 0.5 * np.pi * (1.0 - exp_2ab)
    k_corrected = np.pi * integral / f_inf
    # beta2_eff 恰为零时上面的 1/(c y) 退化；此时积分就是 W(0) B_CUT B_NCH，
    # 除以 f_inf 后为 (2 B_CUT B_NCH/α) tanh(αL/2)。
    zero = b_abs == 0.0
    if np.any(zero):
        k_corrected[zero] = (
            2.0 * b_cut[zero] * b_nch[zero] / alpha_npm_m * (1.0 - exp_ab) / (1.0 + exp_ab)
        )

    k_edge = np.array(k_xci_matrix, dtype=np.float64, copy=True)
    k_edge[rows, cols] = k_corrected
    w_candidate = gamma_xci**2 * k_edge
    candidate_delta = np.abs(w_candidate - w_xci).sum(axis=1)
    scale = w_xci.sum(axis=1)
    fraction = np.divide(candidate_delta, scale, out=np.zeros_like(scale), where=scale > 0.0)
    gated = fraction > materiality
    w_edge[gated, :] = w_candidate[gated, :]
    return w_edge, gated, fraction


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
        (
            "edge_fix",
            "Evaluates the exact XCI island integral where the closed form leaves its validity domain "
            "(zero-dispersion edge); stabilises the SCI kernel at small X. Default True.",
        ),
        (
            "cache_spans",
            "Memoises the SPP/kernels of identical spans inside one path. Default True.",
        ),
        (
            "spp_steps",
            "RK4 steps of the SPP power-profile integration, or 'auto' (12 at <=0 dBm/ch, 32 above).",
        ),
        (
            "resolution",
            "Explicit opt-in resolution profile. None (default) keeps every per-option default "
            "bit-for-bit. 'fast': XCI edge quadrature 12 nodes (NLI identical to 1e-11, 6-65% less "
            "first-call work). 'accurate': spp_steps=96 (and raman_steps=96 on pcfm_h3_raman), the "
            "converged reference of docs/h3-pareto-report.md. A profile never overrides an explicitly "
            "different spp_steps/raman_steps: that raises ValueError.",
        ),
        (
            "raman_model",
            "Set to 'lumped' to add a lumped-equivalent distributed Raman amplifier at the span end. "
            "Default off, so links without pumps are bit-for-bit unchanged.",
        ),
        (
            "raman_pumps",
            "RamanPumpConfig sequence applied to every span; providing it enables the lumped Raman model.",
        ),
        (
            "raman_gain_model",
            "'gaussian' (default), 'linear' CZ approximation, 'table' (normalized spectrum given by "
            "raman_gain_table), a .json/.csv Raman gain table file path, or a RamanGainModel instance "
            "(use RamanGainModel.from_file(path)). The file/instance entry is loaded once at construction "
            "time, never in the hot path.",
        ),
        (
            "raman_gain_table",
            "(offsets_hz, gains) normalized Raman gain table used when raman_gain_model='table'. "
            "Linear interpolation, zero outside the table. Mutually exclusive with a RamanGainModel "
            "instance or a table file path.",
        ),
        (
            "raman_ase",
            "Add the distributed Raman ASE before the span EDFA. Default True when Raman is enabled.",
        ),
        (
            "validity_mode",
            "'warn' (default) records the per-channel applicability report without touching any "
            "number; 'strict' additionally replaces the SNR of invalid channels with NaN.",
        ),
        (
            "nli_power_convention",
            "'total' (default): power_in_w is the per-channel dual-polarization total power and the "
            "printed 16/27 and 32/27 GN weights apply as-is. 'per_pol': power_in_w is "
            "per-polarization power, so the NLI kernels and the ISRS profile are driven with 2*P and "
            "the NLI and span-end signal power are halved. Declaring the wrong side scales the NLI by 8. "
            "plain pcfm has no launch_power_convention and does not scale EDFA/booster ASE, so 'per_pol' "
            "here is an NLI-only comparison (mixed coordinate) and warns at construction; use "
            "pcfm_h3_raman(launch_power_convention='per_pol') for an end-to-end per-pol link.",
        ),
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
        edge_fix: Evaluate the exact XCI island integral on the CUT rows whose
            closed-form weight is materially outside its asymptotic validity
            domain, and use the small-``X`` Taylor branch of the SCI kernel.
        cache_spans: Memoise the SPP integration and the closed-form kernels of
            a span state, so that the identical spans of a transparent link
            are evaluated once.
        nli_power_convention: Power convention of ``power_in_w`` for the NLI
            kernels and the ISRS profile.  ``"total"`` (default) is the
            per-channel dual-polarization total power with the printed
            ``16/27``/``32/27`` weights; ``"per_pol"`` treats the input as
            per-polarization power, drives the kernels and the SPP with
            ``2 * P``, and halves the per-channel NLI and the span-end power.
            See the module docstring.
        xci_edge_quadrature_points: Gauss-Legendre nodes of the exact XCI
            island integral.  Only used for the CUT rows flagged by the
            ``edge_fix`` gate.  Defaults to
            :data:`XCI_EDGE_QUADRATURE_POINTS`; the sweep in
            ``docs/h3-pareto-report.md`` shows 12 nodes already reproduce the
            64-node result to 1e-11, so :data:`RESOLUTION_PROFILES` may lower
            it without moving any number.
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
        edge_fix: bool = True,
        cache_spans: bool = True,
        nli_power_convention: str = "total",
        xci_edge_quadrature_points: int = XCI_EDGE_QUADRATURE_POINTS,
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
        if int(xci_edge_quadrature_points) < 2:
            raise ValueError(f"xci_edge_quadrature_points must be >= 2, got {xci_edge_quadrature_points}.")

        self.poly_degree = int(poly_degree)
        self.n_poly = self.poly_degree + 1
        self.spp_steps = int(spp_steps)
        self.xci_edge_quadrature_points = int(xci_edge_quadrature_points)
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.egn = bool(egn)
        self.edge_fix = bool(edge_fix)
        self.cache_spans = bool(cache_spans)
        self.nli_power_convention = resolve_nli_power_convention(nli_power_convention)
        # 小 |X| 的 SCI 泰勒系数只依赖多项式阶数；edge_fix=False 时关闭该分支。
        self._sci_small_x_switch = SCI_X_SERIES_SWITCH if self.edge_fix else 0.0
        self._sci_small_x_coeff = (
            _sci_small_x_coefficients(2 * self.n_poly + 1) if self.edge_fix else np.zeros((1, 1), dtype=np.float64)
        )
        # XCI 适用域修正按跨长惰性计算并缓存；键为跨长（微米取整）。
        self._xci_edge_cache: dict[int, tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = {}
        # 诊断层的静态量（s、行修正、X）只依赖光纤与跨长，按跨长缓存。
        self._xci_validity_cache: dict[
            int, tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]
        ] = {}
        # 跨段状态记忆化：同一路径内的透明跨输入只差末位，命中即复用 SPP 与核。
        self._span_cache: list[tuple[NDArrayFloat, int, NDArrayFloat, NDArrayFloat, NDArrayFloat]] = []
        self.span_cache_hits = 0
        self.span_cache_misses = 0

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
            self.beta2 + np.pi * self.beta3 * 2.0 * df_ref + (2.0 / 3.0) * np.pi**2 * self.beta4 * 3.0 * df_ref**2
        )
        # Effective dispersion of the XCI island, Eq. (A.15) with
        # m = interferer and k = CUT; the last term of Eq. (A.15) vanishes
        # because one of the two island frequencies is the CUT frequency.
        df_i = self.f_abs_hz[:, None] - self.f_ref_hz
        df_j = self.f_abs_hz[None, :] - self.f_ref_hz
        self.beta2_eff_xci = (
            self.beta2
            + np.pi * self.beta3 * (self.f_abs_hz[:, None] + self.f_abs_hz[None, :] - 2.0 * self.f_ref_hz)
            + (2.0 / 3.0) * np.pi**2 * self.beta4 * (df_i**2 + df_i * df_j + df_j**2)
        )
        # Nonlinear coefficient of the XCI islands, Eq. (16): the cross
        # effective area averages the areas of the CUT and of the interferer.
        a_sum = self.a_eff_m2[:, None] + self.a_eff_m2[None, :]
        self.gamma_xci = (2.0 * np.pi * self.f_abs_hz[:, None] / C_LIGHT) * (2.0 * self.n2_m2_w / a_sum)
        # Nonlinear coefficient of the SCI islands, Eq. (A.8) with the four
        # effective areas equal to the area of the CUT.
        self.gamma_sci = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (self.n2_m2_w / self.a_eff_m2)
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
        self.w_xci: NDArrayFloat = self.gamma_xci**2 * self.k_xci_matrix

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
                raise ValueError(f"lumped loss at {position} m lies outside the span of {span_length_m} m.")
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
        x_all = np.ascontiguousarray(self.beta2_eff_sci * (np.pi**2) * self.b_hz**2 * float(span_length_m))
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

    def _xci_weight_matrix(self, span_length_m: float) -> NDArrayFloat:
        """Return the XCI weight matrix of a span length.

        Without ``edge_fix`` this is the printed Eq. (24) matrix.  With the
        fix, the CUT rows whose closed-form XCI leaves its asymptotic validity
        domain are replaced by the exact island integral; see
        :func:`_xci_edge_weight_matrix`.  The corrected matrix depends on the
        span length (through ``alpha L`` and the island phase), so it is
        cached per span length.

        Args:
            span_length_m: Span length (m).

        Returns
        -------
            ``(N, N)`` XCI weight matrix, internal channel order.
        """
        if not self.edge_fix:
            return self.w_xci
        key = int(round(span_length_m * 1e6))
        cached = self._xci_edge_cache.get(key)
        if cached is not None:
            return cached[0]
        w_edge, gated, fraction = _xci_edge_weight_matrix(
            self.w_xci,
            self.k_xci_matrix,
            self.log_xci,
            self.gamma_xci,
            self.beta2_eff_xci,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_visibility_npm,
            float(span_length_m),
            n_quad=self.xci_edge_quadrature_points,
        )
        self._xci_edge_cache[key] = (w_edge, gated, fraction)
        return w_edge

    def xci_edge_audit(self, span_length_m: float) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return the CUT-row gate and the relative correction of the XCI edge fix.

        Args:
            span_length_m: Span length (m).

        Returns
        -------
            ``(gated, fraction)``: the boolean mask of the corrected CUT rows
            (internal order) and the relative change of every row before
            gating.  Without ``edge_fix`` both are zero.
        """
        if not self.edge_fix:
            empty = np.zeros(self.n_nodes, dtype=np.float64)
            return np.zeros(self.n_nodes, dtype=bool), empty
        self._xci_weight_matrix(span_length_m)
        key = int(round(span_length_m * 1e6))
        _, gated, fraction = self._xci_edge_cache[key]
        return gated, fraction

    def _xci_validity_metrics(
        self,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Return the span-length-dependent part of the validity report.

        The raw ``s`` parameter, the exact row correction and the SCI ``X``
        depend only on the fibre, the channel plan and the span length, so
        they are computed once per span length.  With ``edge_fix=True`` the
        correction is already available from :meth:`xci_edge_audit`; with
        ``edge_fix=False`` it is computed here on demand, because the
        diagnostic must still say how large the omitted correction is.

        Args:
            span_length_m: Span length (m).

        Returns
        -------
            The tuple ``(s_min, s_below, fraction, gated, material)`` in the
            internal channel order.
        """
        key = int(round(span_length_m * 1e6))
        cached = self._xci_validity_cache.get(key)
        if cached is not None:
            return cached

        delta = np.abs(self.f_abs_hz[:, None] - self.f_abs_hz[None, :])
        half_b = 0.5 * self.b_hz
        y1 = delta - half_b[None, :]
        y2 = delta + half_b[None, :]
        s_pair = _xci_validity_s(self.beta2_eff_xci, y2, half_b, self.alpha_visibility_npm)
        # 只有带内不重叠、有定义的干扰对参与判据；其余对本来就不进核。
        pairs = (self.log_xci != 0.0) & (y1 > 0.0)
        s_min = np.where(pairs, s_pair, np.inf).min(axis=1)
        s_below = s_min < XCI_EDGE_VALIDITY_SWITCH

        if self.edge_fix:
            gated, fraction = self.xci_edge_audit(span_length_m)
        else:
            _, _, fraction = _xci_edge_weight_matrix(
                self.w_xci,
                self.k_xci_matrix,
                self.log_xci,
                self.gamma_xci,
                self.beta2_eff_xci,
                self.f_abs_hz,
                self.b_hz,
                self.alpha_visibility_npm,
                float(span_length_m),
                n_quad=self.xci_edge_quadrature_points,
            )
            gated = np.zeros(self.n_nodes, dtype=bool)
        material = fraction > XCI_EDGE_MATERIALITY

        cached = (
            np.ascontiguousarray(s_min),
            np.ascontiguousarray(s_below),
            np.ascontiguousarray(fraction),
            np.ascontiguousarray(gated),
            np.ascontiguousarray(material),
        )
        self._xci_validity_cache[key] = cached
        return cached

    def _validity_tilt_db(self, power_tilt_db: float, gain_tilt_db: float) -> float:
        """Return the tilt metric that drives the validity flag.

        The base kernel uses the launch-power tilt only: the span gain of an
        unpumped fibre is dominated by ISRS, whose profile the model solves
        explicitly, so a large ISRS tilt is not by itself a validity problem.
        :class:`~network_nlo_eval.models.pcfm_h3_raman.PCFH3RamanSpanKernel`
        overrides this to include the gain tilt when pumps are active, where
        the pump-induced profile is the known weak point.

        Args:
            power_tilt_db: Launch-power tilt ``10 log10(max/min)`` (dB).
            gain_tilt_db: Span-gain tilt ``10 log10(max/min)`` (dB).

        Returns
        -------
            The tilt used for ``flagged`` and ``valid`` (dB).
        """
        return power_tilt_db

    def validity_report(
        self,
        span_length_m: float,
        power_in_w: NDArrayFloat,
        *,
        power_out_w: NDArrayFloat | None = None,
    ) -> SpanValidityReport:
        """Report the per-channel applicability of the closed-form kernels.

        The report is a pure diagnostic: it never modifies the kernel state
        beyond its caches and never feeds back into the NLI.  It is meant for
        the network-level QoT layer, which can use ``valid``/``confidence`` to
        warn, drop a channel, or trigger an exact fallback (the fallback
        itself is intentionally not called here).

        Args:
            span_length_m: Span length (m).
            power_in_w: ``(N,)`` launch power per channel (W), model order.
            power_out_w: Optional ``(N,)`` span-end signal power in model
                order.  The model passes its already-computed output here so
                the diagnostic does not integrate the SPP a second time.
                When omitted, the kernel integrates it itself.

        Returns
        -------
            A :class:`SpanValidityReport` in model order.

        Raises
        ------
            ValueError: ``power_in_w`` or ``power_out_w`` does not have one
                value per channel.
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != (self.n_nodes,):
            raise ValueError(f"power_in_w must have shape ({self.n_nodes},), got {power.shape}.")
        power_internal = np.ascontiguousarray(power[self.order])
        active = power_internal > 0.0

        if power_out_w is None:
            _, power_out = self._integrate_spp(power_internal, float(span_length_m))
        else:
            out = np.asarray(power_out_w, dtype=np.float64)
            if out.shape != (self.n_nodes,):
                raise ValueError(f"power_out_w must have shape ({self.n_nodes},), got {out.shape}.")
            power_out = np.ascontiguousarray(out[self.order])

        s_min, s_below, fraction, gated, material = self._xci_validity_metrics(span_length_m)

        x_all = self._special_functions(span_length_m)[0]
        abs_x = np.abs(x_all)
        sci_taylor = abs_x < float(self._sci_small_x_switch)
        sci_below = abs_x < SCI_X_SERIES_SWITCH
        sci_x_abs_min = float(np.min(abs_x[active])) if np.any(active) else 0.0

        def _tilt(values: NDArrayFloat) -> float:
            """``10 log10(max/min)`` over the active channels, 0 below two."""
            selected = values[active]
            if selected.size < 2:
                return 0.0
            return float(10.0 * np.log10(float(selected.max()) / float(selected.min())))

        power_tilt_db = _tilt(power_internal)
        with np.errstate(divide="ignore", invalid="ignore"):
            gain = np.divide(
                power_out,
                power_internal,
                out=np.ones_like(power_internal),
                where=active,
            )
        gain_tilt_db = _tilt(gain)

        tilt_db = self._validity_tilt_db(power_tilt_db, gain_tilt_db)
        tilt_exceeded = tilt_db > POWER_TILT_VALIDITY_DB
        tilt_confidence = (
            1.0
            if not tilt_exceeded
            else float(np.clip(1.0 - (tilt_db - POWER_TILT_VALIDITY_DB) / POWER_TILT_VALIDITY_DB, 0.0, 1.0))
        )

        # 置信度是逐信道取最弱项：XCI 修正、SCI 分支、功率倾斜。
        xci_confidence = np.ones(self.n_nodes, dtype=np.float64)
        xci_confidence = np.where(s_below & ~material, XCI_IMMATERIAL_CONFIDENCE, xci_confidence)
        xci_confidence = np.where(
            material,
            np.where(gated, XCI_GATED_CONFIDENCE, XCI_UNCORRECTED_CONFIDENCE),
            xci_confidence,
        )
        sci_confidence = np.where(
            sci_below,
            np.where(sci_taylor, SCI_TAYLOR_CONFIDENCE, SCI_UNCORRECTED_CONFIDENCE),
            1.0,
        )
        confidence = np.minimum(np.minimum(xci_confidence, sci_confidence), tilt_confidence)

        # 只有"本应修正却没修正"才算无效；已由精确积分兜底的信道仍然可用。
        valid = ~((material & ~gated) | (sci_below & ~sci_taylor) | tilt_exceeded)
        flagged = material | sci_below | tilt_exceeded

        # 密集栅格只是适用域警告：它把相邻矩形谱重叠的事实标出来，但不改
        # ``valid``（点值仍是模型给出的有限数）和置信度。
        min_spacing_hz, spacing_ratio = dense_grid_metrics(self.f_abs_hz, self.b_hz, active)
        dense_grid = bool(spacing_ratio < DENSE_GRID_SPACING_RATIO)
        flagged = flagged | (dense_grid & active)

        # 未占用信道没有可判定的物理量，也不应被 strict 变成 NaN。
        valid = np.where(active, valid, True)
        flagged = np.where(active, flagged, False)
        confidence = np.where(active, confidence, 1.0)

        return SpanValidityReport(
            xci_gated=np.ascontiguousarray(gated[self.inverse_order]),
            xci_row_correction=np.ascontiguousarray(fraction[self.inverse_order]),
            xci_s_min=np.ascontiguousarray(s_min[self.inverse_order]),
            xci_s_below_switch=np.ascontiguousarray(s_below[self.inverse_order]),
            sci_x=np.ascontiguousarray(x_all[self.inverse_order]),
            sci_taylor=np.ascontiguousarray(sci_taylor[self.inverse_order]),
            sci_x_below_switch=np.ascontiguousarray(sci_below[self.inverse_order]),
            sci_x_abs_min=sci_x_abs_min,
            power_tilt_db=power_tilt_db,
            gain_tilt_db=gain_tilt_db,
            pump_active=False,
            pump_directions=(),
            pump_depletion=np.zeros(0, dtype=np.float64),
            pump_depletion_max=0.0,
            flagged=np.ascontiguousarray(flagged[self.inverse_order]),
            valid=np.ascontiguousarray(valid[self.inverse_order]),
            confidence=np.ascontiguousarray(confidence[self.inverse_order]),
            effective_tilt_db=float(tilt_db),
            tilt_confidence_informational=informational_tilt_confidence(tilt_db),
            min_channel_spacing_hz=float(min_spacing_hz),
            spacing_to_bandwidth=float(spacing_ratio),
            dense_grid=dense_grid,
        )

    def clear_span_cache(self) -> None:
        """Forget the memoised span states and reset the hit/miss counters.

        The model calls this at the beginning of every path evaluation, so the
        memoisation only shares work between the identical (transparent) spans
        of one link.  It never carries a state from one path to the next.
        """
        self._span_cache.clear()
        self.span_cache_hits = 0
        self.span_cache_misses = 0

    def _span_cache_lookup(
        self,
        power_internal: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat] | None:
        """Look up the memoised kernels of a span state.

        Args:
            power_internal: ``(N,)`` launch power in the internal channel order.
            span_length_m: Span length (m).

        Returns
        -------
            ``(power_out, k_sci, profile_energy)`` or ``None`` on a miss.
        """
        if not self.cache_spans:
            return None
        key = int(round(span_length_m * 1e6))
        for cached_power, cached_key, power_out, k_sci, energy in self._span_cache:
            if cached_key != key:
                continue
            if np.array_equal(cached_power, power_internal):
                self.span_cache_hits += 1
                return power_out, k_sci, energy
            # 透明跨的输入只差 EDFA 增益除法的末位；在这个容差内物理不可分辨。
            if np.allclose(cached_power, power_internal, rtol=SPAN_CACHE_RTOL, atol=0.0):
                self.span_cache_hits += 1
                return power_out, k_sci, energy
        self.span_cache_misses += 1
        return None

    def _span_cache_store(
        self,
        power_internal: NDArrayFloat,
        span_length_m: float,
        power_out: NDArrayFloat,
        k_sci: NDArrayFloat,
        energy: NDArrayFloat,
    ) -> None:
        """Store the memoised kernels of a span state (FIFO, bounded size)."""
        if not self.cache_spans:
            return
        if len(self._span_cache) >= SPAN_CACHE_MAX_ENTRIES:
            self._span_cache.pop(0)
        key = int(round(span_length_m * 1e6))
        self._span_cache.append(
            (
                np.array(power_internal, dtype=np.float64, copy=True),
                key,
                power_out,
                k_sci,
                energy,
            )
        )

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
            np.ascontiguousarray(self.b_hz**2),
            float(span_length_m),
            np.ascontiguousarray(self._sci_small_x_coeff),
            float(self._sci_small_x_switch),
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

    def _spp_drive_scale(self) -> float:
        """Power scale between ``power_in_w`` and the physical ISRS drive.

        The ISRS power equations are a physical statement about the fibre: the
        Raman coupling between two channels is driven by their
        **dual-polarization total** power.  The coordinate of ``power_in_w`` is
        therefore irrelevant to the profile itself.  Under
        ``nli_power_convention="per_pol"`` the input is per-polarization power,
        so the profile must be driven with ``2 * P``, exactly like the NLI
        kernels; the span-end power is halved back on the way out.

        ``"total"`` returns ``1.0`` and :meth:`_integrate_spp` then multiplies
        nothing, which keeps the historical path bit-for-bit identical.

        Returns
        -------
            ``2.0`` for a per-polarization input, ``1.0`` otherwise.

        Notes
        -----
            Subclasses whose launch and NLI conventions may disagree override
            this, see
            :meth:`~network_nlo_eval.models.pcfm_h3_raman.PCFH3RamanSpanKernel._spp_drive_scale`.
        """
        return 2.0 if self.nli_power_convention == "per_pol" else 1.0

    def _integrate_spp(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Integrate the ISRS power equations and normalise the profile.

        Args:
            power: ``(N,)`` launch power per channel (W), internal order, in the
                kernel's ``nli_power_convention`` coordinate.
            span_length_m: Span length (m).

        Returns
        -------
            ``(profile_norm, power_out)``: the normalised SPP
            ``p_i(z) = P_i(z)/P_i(0)`` sampled twice per integration step and
            the power at the span end, both in the input coordinate.
        """
        drive_scale = self._spp_drive_scale()
        # 物理驱动永远是双偏振总功率；total 口径下 drive 就是输入本身（不复制、
        # 不乘 1.0），保证历史路径逐位不变。
        drive = power * drive_scale if drive_scale != 1.0 else power
        lengths, losses, steps = self._span_segments(span_length_m)
        n_samples = 2 * int(steps.sum())
        profile = np.zeros((n_samples, self.n_nodes), dtype=np.float64)
        power_out = np.zeros(self.n_nodes, dtype=np.float64)
        _solve_spp_jit(
            self.coef_mat,
            self.alpha_npm,
            drive,
            lengths,
            losses,
            steps,
            profile,
            power_out,
        )
        # A dark channel keeps a passive unit transfer function.
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_norm = np.where(
                drive[None, :] > 0.0,
                profile / np.where(drive[None, :] > 0.0, drive[None, :], 1.0),
                1.0,
            )
        if drive_scale != 1.0:
            # 出射功率回到输入坐标；除以 2 在二进制浮点下精确，per_pol 与
            # total 因此逐位对齐，下游不需要再做换算。
            power_out = power_out / drive_scale
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
            ramp_spm, ramp_xpm = self._inter_span_ramps(span_context.total_spans, span_length_m)
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

        # 跨段记忆化：透明链路的每一跨输入只差末位，SPP/SCI/能量只算一次。
        cached = self._span_cache_lookup(power, span_length_m)
        if cached is None:
            profile_norm, power_out = self._integrate_spp(power, span_length_m)
            # Polynomial representation of the SPP, Eq. (23).  The fit is done
            # in the scaled variable u = z/L, which is the numerically sound
            # basis; the coefficients of the z-basis follow as p_n = q_n/L^n.
            pinv = self._fit_operator(span_length_m)
            q_scaled = pinv @ profile_norm  # (N_p+1, N)
            # --- SCI kernel, Eq. (30) --------------------------------------
            k_sci = self.sci_kernel(span_length_m, q_scaled)
            q_int = _profile_energy_jit(np.ascontiguousarray(q_scaled)) * float(span_length_m)
            self._span_cache_store(power, span_length_m, power_out, k_sci, q_int)
        else:
            power_out, k_sci, q_int = cached

        # --- NLI PSDs, Eqs. (21) and (26) ----------------------------------
        active = power > 0.0
        if self.nli_power_convention == "per_pol":
            # (16/27) = (8/9)^2 * 6/2^3 是按“总功率 PSD 由两个偏振均分”推出
            # 的双偏振权重，核的输入必须是物理总功率；per_pol 输入是每偏振
            # 功率，所以驱动 PSD 乘 2。
            psd = np.where(active, 2.0 * power / self.b_hz, 0.0)  # G_WDM,i
        else:
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
        xci_weight = q_int * psd**2 * rho_nch
        xci_acc = self._xci_weight_matrix(span_length_m) @ xci_weight
        g_xci = (32.0 / 27.0) * psd * xci_acc

        spm = np.where(active, g_sci * self.b_hz, 0.0)
        xpm = np.where(active, g_xci * self.b_hz, 0.0)
        if self.nli_power_convention == "per_pol":
            # 输出回到每偏振坐标：信号与 ASE 都是每偏振量，NLI 也必须除 2。
            spm = 0.5 * spm
            xpm = 0.5 * xpm

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
        edge_fix: Evaluate the exact XCI island integral where the closed form
            leaves its validity domain, and stabilise the SCI kernel at small
            ``X``.  See :class:`PCFMSpanKernel`.
        cache_spans: Memoise the identical spans of one path.
        spp_steps: RK4 steps of the SPP integration, or ``"auto"`` (12 at
            ``<=0 dBm`` per channel, 32 above it).
        resolution: Explicit opt-in resolution profile; ``None`` (default)
            leaves every per-option default untouched.  See
            :data:`RESOLUTION_PROFILES`.
        raman_model: ``"lumped"`` enables the lumped-equivalent distributed
            Raman amplifier of :mod:`network_nlo_eval.models.raman`.  ``None``
            (default) and ``"off"`` leave the model bit-for-bit unchanged.
        raman_pumps: Pumps applied to every span.  Providing this argument
            also enables the lumped model when ``raman_model`` is ``None``.
            When omitted, the pumps of each :class:`FiberSpanConfig` are used.
        raman_gain_model: ``"gaussian"``、``"linear"``、``"table"``、
            :class:`~network_nlo_eval.models.raman.RamanGainModel` 实例，或
            ``.json``/``.csv`` 增益表文件路径。文件/实例只在构造期加载一次。
        raman_gain_table: ``gain_model="table"`` 时的归一化增益表
            ``(offsets_hz, gains)``；与实例/文件路径互斥。
        raman_ase: Include the distributed Raman ASE before the span EDFA.
        raman_temperature_k: Fibre temperature used by the Raman spontaneous
            emission factor.  ``None`` (default) means the module default of
            300 K; passing ``300.0`` explicitly counts as a calibration
            override and warns when the gain table comes from a file.
        raman_peak_gain_coefficient_m_per_w: Peak local Raman gain rate per
            watt before division by ``A_eff``.  ``None`` (default) means the
            module default of ``6e-14``; the default matches the
            waveform-level ``physics_nlo_eval.RamanPump`` for cross-checking.
            Explicitly passing ``6e-14`` counts as a calibration override for
            file-backed gain tables and warns.
        nli_power_convention: ``"total"`` (default) means ``power_in_w`` is the
            per-channel dual-polarization total power and the printed
            ``16/27``/``32/27`` weights apply as-is; ``"per_pol"`` means it is
            per-polarization power, so the kernels are driven with ``2 * P`` and
            the NLI is halved.  See :class:`PCFMSpanKernel`.  ``PCFMModel``
            itself declares no ``launch_power_convention`` and does not scale
            the EDFA/booster ASE, so ``"per_pol"`` here is an **NLI-only
            comparison** (mixed coordinate) and warns at construction.  Use
            :class:`~network_nlo_eval.models.pcfm_h3_raman.PCFH3RamanModel`
            with ``launch_power_convention="per_pol"`` for an end-to-end
            per-polarization link.
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
        edge_fix: bool = True,
        cache_spans: bool = True,
        spp_steps: int | str = DEFAULT_SPP_STEPS,
        raman_model: str | None = None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str | RamanGainModel = "gaussian",
        raman_gain_table: tuple[Sequence[float], Sequence[float]] | None = None,
        raman_ase: bool = True,
        raman_temperature_k: float | None = None,
        raman_peak_gain_coefficient_m_per_w: float | None = None,
        validity_mode: str = "warn",
        nli_power_convention: str = "total",
        resolution: str | None = None,
        **route_options,
    ) -> None:
        """Initialize the model.

        Raises
        ------
            ValueError: ``poly_degree`` is negative, ``spp_steps`` is invalid,
                the Raman options are inconsistent, ``validity_mode`` is not
                ``"warn"``/``"strict"``, ``nli_power_convention`` is unknown,
                or ``resolution`` is unknown or conflicts with an explicit
                ``spp_steps``.
        """
        if poly_degree < 0:
            raise ValueError(f"poly_degree must be >= 0, got {poly_degree}.")
        if spp_steps != "auto" and int(spp_steps) < 1:
            raise ValueError(f"spp_steps must be >= 1 or 'auto', got {spp_steps!r}.")
        if raman_model not in (None, "off", "lumped"):
            raise ValueError(f"raman_model must be None, 'off' or 'lumped', got {raman_model!r}.")
        if validity_mode not in ("warn", "strict"):
            raise ValueError(f"validity_mode must be 'warn' or 'strict', got {validity_mode!r}.")
        profile = resolve_resolution_profile(resolution)
        quadrature_points = XCI_EDGE_QUADRATURE_POINTS
        if profile is not None:
            if "spp_steps" in profile:
                # 显式给了非默认步数就是调用方在自行定标，档位不应悄悄覆盖它。
                if spp_steps != DEFAULT_SPP_STEPS:
                    raise ValueError(
                        f"resolution={resolution!r} sets spp_steps={profile['spp_steps']}, but "
                        f"spp_steps={spp_steps!r} was also given explicitly."
                    )
                spp_steps = profile["spp_steps"]
            quadrature_points = profile.get("xci_quadrature_points", XCI_EDGE_QUADRATURE_POINTS)
        nli_power_convention = resolve_nli_power_convention(nli_power_convention)
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.poly_degree = int(poly_degree)
        self.egn = bool(egn)
        self.edge_fix = bool(edge_fix)
        self.cache_spans = bool(cache_spans)
        self.nli_power_convention = resolve_nli_power_convention(nli_power_convention)
        # plain PCFM 没有 launch_power_convention，也不覆盖 ase_polarization_scale：
        # 声明 per_pol 只把信号/NLI 换到每偏振，EDFA/booster ASE 仍是双偏振总量。
        # 子类（如 pcfm_h3_raman）用类级属性声明了 launch 口径，不受此 warning 影响。
        if self.nli_power_convention == "per_pol" and not hasattr(type(self), "launch_power_convention"):
            warnings.warn(
                "pcfm: nli_power_convention='per_pol' without a launch_power_convention is an "
                "NLI-only comparison. The signal and the NLI are per-polarization, but this model "
                "owns no launch coordinate and does not halve the route-layer ASE "
                "(ase_polarization_scale stays 1.0), so the SNR is a mixed coordinate, not an "
                "end-to-end per-pol result. Use pcfm_h3_raman with launch_power_convention='per_pol' "
                "for an end-to-end per-pol link, or keep nli_power_convention='total'.",
                UserWarning,
                stacklevel=2,
            )
        self.spp_steps = spp_steps
        self.resolution = None if profile is None else str(resolution)
        self.xci_edge_quadrature_points = int(quadrature_points)
        self.validity_mode = str(validity_mode)
        self._active_spp_steps = DEFAULT_SPP_STEPS if spp_steps == "auto" else int(spp_steps)
        self._span_index = 0
        self._beta2_accum_ps2 = np.zeros(self.n_nodes, dtype=np.float64)
        # 诊断层状态：逐跨报告与跨间最坏值，入口在 evaluate_path 里重置。
        self.last_span_validity: list[SpanValidityReport] = []
        self.last_validity_valid: NDArrayFloat | None = None
        self.last_validity_flagged: NDArrayFloat | None = None
        self.last_validity_confidence: NDArrayFloat | None = None
        # 拉曼默认关闭：只有显式 raman_model="lumped" 或显式给出 raman_pumps
        # 才启用；仅 fiber_config 自带泵浦不会改变既有 PCFM 结果。
        self._raman_pumps = None if raman_pumps is None else list(raman_pumps)
        self._raman_enabled = raman_model == "lumped" or (raman_model is None and self._raman_pumps is not None)
        _pump_source = self._raman_pumps if self._raman_pumps is not None else ref_fiber_config.raman_pumps
        self._lumped_raman_active = bool(
            self._raman_enabled and any(float(pump.power_w) > 0.0 for pump in (_pump_source or ()))
        )
        if self._lumped_raman_active:
            warnings.warn(
                "pcfm: raman_model='lumped' with active pumps is not a trustworthy QoT model. "
                "The lumped approximation multiplies only the span-end signal power by the on-off "
                "gain; it does not model pump depletion, the intra-span Raman profile or the "
                "wideband Raman ASE. The accuracy matrix measures 5-28 dB of optimistic error "
                "against the DRA reference, while the H3 budget only covers 2.24-8.79 dB. The "
                "point values are kept for backward compatibility, but validity_report and the "
                "error budget flag every such span/channel as unusable. Use pcfm_h3_raman or "
                "hybrid_raman_paper for pumped links.",
                UserWarning,
                stacklevel=2,
            )
        self._raman_ase = bool(raman_ase)
        self._raman_gain_model = resolve_raman_gain_model(
            raman_gain_model,
            table=raman_gain_table,
            peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            temperature_k=raman_temperature_k,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> PCFMSpanKernel:
        """Build the single-span kernel of this fibre."""
        return PCFMSpanKernel(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            poly_degree=self.poly_degree,
            symbol_rate_hz=self.symbol_rate_hz,
            egn=self.egn,
            edge_fix=self.edge_fix,
            cache_spans=self.cache_spans,
            spp_steps=self._active_spp_steps,
            nli_power_convention=self.nli_power_convention,
            xci_edge_quadrature_points=self.xci_edge_quadrature_points,
        )

    @property
    def raman_pumps(self) -> list[RamanPumpConfig]:
        """Configured pumps of the model (empty when Raman is off).

        ``PCFMSpanKernel`` carries no pump state, so the H3 error budget has to
        read the pump directions from the model instead.  The H3 Raman subclass
        keeps its pumps on the kernel and therefore leaves this empty.
        """
        return list(self._raman_pumps or [])

    @property
    def lumped_raman_active(self) -> bool:
        """Whether this model applies the lumped Raman approximation with pumps.

        Such a link is numerically supported for backward compatibility but is
        not a trustworthy QoT: the intra-span profile and pump depletion are
        missing.  The span reports and the error budget use this flag to refuse
        it, without changing the point result.
        """
        return bool(self._lumped_raman_active)

    @property
    def physical_power_scale(self) -> float:
        """Ratio between the physical total power and one ``power_in_w`` entry.

        The auto SPP-step heuristic compares the launch power with a fixed dBm
        threshold, which only means the same thing in every coordinate if the
        threshold is expressed in *physical total* power.  Under
        ``nli_power_convention="per_pol"`` each entry of ``power_in_w`` is one
        polarization, i.e. half of the physical total, so the peak has to be
        doubled before it is turned into dBm.  Subclasses whose launch and NLI
        conventions may disagree override this with the factor that matches
        their ISRS drive.

        Returns
        -------
            ``2.0`` for a per-polarization coordinate, ``1.0`` otherwise.
        """
        return 2.0 if self.nli_power_convention == "per_pol" else 1.0

    def _reset_validity(self) -> None:
        """Forget the validity state of the previous path."""
        self.last_span_validity = []
        self.last_validity_valid = None
        self.last_validity_flagged = None
        self.last_validity_confidence = None

    def _record_span_validity(
        self,
        kernel: PCFMSpanKernel,
        power_in_w: NDArrayFloat,
        power_out_w: NDArrayFloat,
        span_length_m: float,
    ) -> None:
        """Attach the validity report of one span to the path state.

        The link-level verdict is the worst span: a channel is valid only if
        every span reports it valid, its confidence is the minimum, and any
        span-level flag survives.  Keeping the per-span reports makes the
        offending span identifiable without recomputing anything.

        Args:
            kernel: Kernel that evaluated the span.
            power_in_w: ``(N,)`` span-input power (W), model order.
            power_out_w: ``(N,)`` span-output power before the EDFA (W).
            span_length_m: Span length (m).
        """
        report = kernel.validity_report(span_length_m, power_in_w, power_out_w=power_out_w)
        if self._lumped_raman_active:
            # 集总拉曼只改跨末功率，跨内剖面/耗尽缺失；点值保留（兼容），
            # 但诊断层必须让下游看见“不可信”。未占用信道保持原样。
            active = np.asarray(power_in_w, dtype=np.float64) > 0.0
            report = replace(
                report,
                flagged=np.where(active, True, report.flagged),
                confidence=np.where(active, 0.2, report.confidence),
                lumped_raman_approximation=True,
            )
        self.last_span_validity.append(report)
        if self.last_validity_valid is None:
            self.last_validity_valid = np.array(report.valid, dtype=bool, copy=True)
            self.last_validity_flagged = np.array(report.flagged, dtype=bool, copy=True)
            self.last_validity_confidence = np.array(report.confidence, dtype=np.float64, copy=True)
            return
        self.last_validity_valid &= report.valid
        self.last_validity_flagged |= report.flagged
        np.minimum(self.last_validity_confidence, report.confidence, out=self.last_validity_confidence)

    def evaluate_path(self, optical_path, power_in_w):
        """Evaluate a path, resetting the EGN context and the span memoisation.

        The span memoisation is deliberately path-scoped: clearing it here
        means that the identical (transparent) spans of one link share their
        SPP integration, while a new path or a new launch power never reuses a
        stale state.

        In ``validity_mode="strict"`` the SNR of every channel that the span
        reports as invalid is replaced by ``NaN``.  The NLI, ASE and signal
        arrays are left untouched, so a caller can still inspect them and
        decide on an exact fallback.  ``"warn"`` (the default) only records
        the report in :attr:`last_validity_valid`,
        :attr:`last_validity_flagged` and :attr:`last_validity_confidence`.
        """
        if self.spp_steps == "auto":
            # 阈值是物理总功率的 dBm：per_pol 口径下输入只是单偏振功率，先乘
            # 倍数再取峰值，否则同一物理链路会被判成低功率而选 12 步。
            peak_power_w = max(float(np.max(power_in_w)), 1e-30) * self.physical_power_scale
            peak_dbm = 10.0 * np.log10(peak_power_w * 1e3)
            wanted = 12 if peak_dbm <= 0.0 else DEFAULT_SPP_STEPS
            if wanted != self._active_spp_steps:
                self._active_spp_steps = wanted
                self._kernel_cache.clear()
        self._span_index = 0
        self._beta2_accum_ps2 = np.zeros(self.n_nodes, dtype=np.float64)
        self._reset_validity()
        for kernel in self._kernel_cache.values():
            clear = getattr(kernel, "clear_span_cache", None)
            if clear is not None:
                clear()
        result = super().evaluate_path(optical_path, power_in_w)
        if (
            self.validity_mode == "strict"
            and self.last_validity_valid is not None
            and not bool(np.all(self.last_validity_valid))
        ):
            snr_db = np.array(result.snr_db, dtype=np.float64, copy=True)
            snr_db[~self.last_validity_valid] = np.nan
            result = replace(result, snr_db=snr_db)
        return result

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
            span_index=(self._span_index if span_context is None else int(span_context.span_index)),
            beta2_accum_ps2=self._beta2_accum_ps2,
            span_context=span_context,
        )
        input_ase_w: NDArrayFloat | None = None
        if self._raman_enabled:
            pumps = self._raman_pumps if self._raman_pumps is not None else fiber_config.raman_pumps
            if pumps and any(float(pump.power_w) > 0 for pump in pumps):
                a_eff_m2 = _calc_aeff_dynamic(
                    self.lambdas_m,
                    fiber_config.effective_area_um2_ref,
                    fiber_config.aeff_slope_um2_nm,
                    fiber_config.reference_wavelength_nm,
                )
                if fiber_config.use_attenuation_polynomial:
                    alpha_npm = _calc_fiber_attenuation_npm(
                        self.lambdas_m,
                        fiber_config.reference_wavelength_nm,
                        fiber_config.attenuation_alpha2,
                        fiber_config.attenuation_alpha1,
                        fiber_config.attenuation_alpha0,
                    )
                else:
                    alpha_npm = np.full(
                        self.n_nodes,
                        fiber_config.attenuation_db_km_ref * np.log(10.0) / 10.0 / 1e3,
                        dtype=np.float64,
                    )
                amplifier = RamanAmplifier(
                    pumps,
                    span_length_m,
                    model=self._raman_gain_model,
                    a_eff_m2=a_eff_m2,
                    signal_attenuation_npm=alpha_npm,
                )
                # 集总等效：只把末端功率乘以 on-off 增益，跨内 NLI 不变。
                fiber_output = SpanFiberOutput(
                    power_out_w=fiber_output.power_out_w * amplifier.gain(self.f_abs_hz),
                    spm_noise_w=fiber_output.spm_noise_w,
                    xpm_noise_w=fiber_output.xpm_noise_w,
                    nli_amplitude_spm=fiber_output.nli_amplitude_spm,
                    nli_amplitude_xpm=fiber_output.nli_amplitude_xpm,
                )
                if self._raman_ase:
                    input_ase_w = amplifier.ase_w(self.f_abs_hz, self.channel_bandwidth_hz_channels)
        self._record_span_validity(kernel, power_in_w, fiber_output.power_out_w, span_length_m)
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            input_ase_w=input_ase_w,
        )
        self._span_index += 1
        self._beta2_accum_ps2 = self._beta2_accum_ps2 + kernel.effective_dispersion_ps2(span_length_m)
        return element


register_model(PCFM_MODEL_INFO, PCFMModel)
