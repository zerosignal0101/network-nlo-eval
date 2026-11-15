"""Polynomial closed-form model (PCFM) of Poggiolini et al., JLT 2026.

The model evaluates analytical GN kernels on polynomial spatial power
profiles. It implements the "polynomial closed form model for
ultra-wideband transmission systems" of

    P. Poggiolini, Y. Jiang, Y. Gao, F. Forghieri,
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
each channel. The spatial profile integrates inter-channel Stokes transfer
without Raman pumps. Pumped links use ``pcfm_h3_raman`` or a DRA paper model.

The EGN correction (``egn=True``) applies the machine-learning coefficient
``rho`` of Zefreh et al., JLT 2020, Eq. (14), with the coefficients of Table IV
of that paper.  Those coefficients are *fitted* values of the reference paper,
not a derivation of this work package; they are reproduced verbatim and are
marked as such in the module documentation.

The output reference plane follows Eqs. (21) and (26):

``Output transfer Gamma * p_CUT(L)``
    The kernel returns compensated NLI. The shared route layer applies
    ``Gamma * P_CUT(L)/P_CUT(0)`` to noise powers and its square root to
    amplitudes. Exact per-channel loss compensation gives a unit factor;
    fixed gain, ripple, and unamplified spans use their actual transfer.

``Fourth-order dispersion is dropped``
    The framework carries no fourth-order dispersion coefficient, so the
    ``beta4`` term of Eq. (A.15) is set to zero. Its error depends on the
    fiber and frequency range; this implementation supplies no general bound.

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

The paper's Eq. (8) sums individual-span NLI powers. For identical transparent
spans this gives ``N`` times the single-span result. Section IV-E discusses
additional fitted corrections for formats and inter-span coherence, but does
not derive the visibility construction used here. This implementation weights
SPM inter-span visibility by **NLI noise power** and uses a uniform-band
approximation for XPM. It hands the route layer a *complex amplitude* per span
so that the route layer can sum
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
length. Matching ``R(N)`` additionally requires identical fiber parameters,
equal span noise amplitudes, and balanced downstream transfer weights; equal
span lengths alone do not establish those conditions.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.special import sici

from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanContext,
    SpanFiberOutput,
    SpanKernel,
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
    sci_small_x_branch_mask,
)
from network_nlo_eval.models.raman import (
    RamanAmplifier,
    RamanGainModel,
    resolve_raman_gain_model,
)
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.flexgrid import FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import (
        NDArrayBool,
        NDArrayComplex,
        NDArrayFloat,
        NDArrayInt,
    )
    from network_nlo_eval.models.numerical import NumericalISRSGNKernel
    from network_nlo_eval.network.elements import (
        EDFAConfig,
        FiberSpanConfig,
        RamanPumpConfig,
        ROADMConfig,
    )

__all__ = [
    "DENSE_GRID_SPACING_RATIO",
    "EGN_A_COEFFICIENTS",
    "EGN_PHI_VALUES",
    "NLI_POWER_CONVENTIONS",
    "PCFM_MODEL_INFO",
    "RESOLUTION_PROFILES",
    "TILT_INFORMATIONAL_KNEE_DB",
    "TILT_INFORMATIONAL_ZERO_DB",
    "DenseGridDiagnostics",
    "PCFMModel",
    "PCFMSpanKernel",
    "SpanValidityReport",
    "dense_grid_metrics",
    "informational_tilt_confidence",
    "resolve_nli_power_convention",
    "resolve_resolution_profile",
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

    Parameters
    ----------
    resolution : str | None
        ``None`` or ``"default"`` (historical behaviour),
        ``"fast"`` (drop the XCI edge quadrature to its measured-flat
        value) or ``"accurate"`` (the converged sweep reference).

    Returns
    -------
    dict[str, int] | None
        A mapping of option name to value, or ``None`` when no profile is
        requested.

    Raises
    ------
    ValueError
        ``resolution`` is not one of the accepted values.
    """
    if resolution is None or resolution == "default":
        return None
    if (
        not isinstance(cast("object", resolution), str)
        or resolution not in RESOLUTION_PROFILES
    ):
        msg = f"resolution must be None, 'default' or one of {sorted(RESOLUTION_PROFILES)}, got {resolution!r}."
        raise ValueError(msg)
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

    Parameters
    ----------
    nli_power_convention : str
        ``"total"`` or ``"per_pol"``.

    Returns
    -------
    str
        The normalized convention name.

    Raises
    ------
    ValueError
        The value is not one of :data:`NLI_POWER_CONVENTIONS`.
    """
    convention = str(nli_power_convention)
    if convention not in NLI_POWER_CONVENTIONS:
        msg = f"nli_power_convention must be one of {NLI_POWER_CONVENTIONS}, got {nli_power_convention!r}."
        raise ValueError(msg)
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

#: Gauss-Legendre nodes per ``y`` interval of the displaced-third-wave (FWM/MCI)
#: triple integral of :func:`_fwm_displaced_weight_terms`.
#:
#: Deliberately a separate knob from :data:`XCI_EDGE_QUADRATURE_POINTS`: the
#: XCI edge/overlap repairs are already converged (12 nodes, 1e-11), while the
#: FWM triple integrand still moves by 1-2 % between 24 and 192 nodes on the
#: densest C grids.  Coupling the two would silently change the XCI rows, so
#: this one only acts on the opt-in ``dense_grid_fwm_fix`` term.  Default 24
#: keeps every existing number bit-for-bit unchanged.
FWM_QUADRATURE_POINTS: int = 24

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

    Parameters
    ----------
    tilt_db : float
        Tilt ``10 log10(max/min)`` in dB.  Values below the knee
        return 1; values at or above the zero return 0.

    Returns
    -------
    float
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

#: Accepted values of ``fwm_validity_mode``.
#:
#: ``"flag"`` (default) only ORs ``fwm_required`` into ``flagged`` and writes
#: an error-budget note; ``"reject"`` additionally clears ``valid`` on the
#: affected channels.  Neither value ever changes the point NLI/ASE/SNR.
FWM_VALIDITY_MODES: tuple[str, ...] = ("flag", "reject")

#: Accepted values of ``dense_grid_fwm_coherence``.
#:
#: ``"xci"`` (default) keeps the displaced-third-wave term inside the XCI
#: complex amplitude, so it inherits the inter-span XCI visibility and the
#: route layer accumulates ``|sum_k a_k|**2``.  ``"incoherent"`` splits the
#: term out of ``xpm`` and hands it to the route layer as an additive power
#: that accumulates as ``sum_k P_k`` across spans.  The split is only
#: meaningful together with ``dense_grid_fwm_fix=True``; on a non-dense plan
#: no triple exists and both values are bit-for-bit identical.
#:
#: ``"auto"`` picks between the two per span plan: it resolves to
#: ``"incoherent"`` only when ``dense_grid_fwm_fix=True``, the plan is dense
#: and :data:`FWM_AUTO_RATIO_THRESHOLD` is reached; everywhere else it is
#: bit-for-bit ``"xci"``.  See :func:`resolve_fwm_coherence_mode`.
FWM_COHERENCE_MODES: tuple[str, ...] = ("xci", "incoherent", "auto")

#: Grid metric above which ``dense_grid_fwm_coherence="auto"`` splits the
#: displaced-third-wave term out of the XCI amplitude.
#:
#: The metric is ``max(B_hz) / min_spacing``: how much wider the occupied
#: channel is than the tightest gap between two neighbouring centres.  The
#: inter-span XCI visibility is built from the *disjoint-rectangle* phase
#: mismatch, which degenerates when neighbouring spectra overlap; the deeper
#: the overlap, the less the FWM term the XCI amplitude carries looks like the
#: multi-span coherent sum, so the additive (power) accumulation wins.
#:
#: Calibration (C band, 64 GBaud, 100 km, NF 5.5 dB, NLI max|Δ| vs the
#: repository ``numerical`` reference, task BE data):
#:
#: ``B/s`` 1.28 (50 GHz): xci 0.103 dB, incoherent 0.120 dB -> keep xci;
#: ``B/s`` 1.71 (37.5 GHz): xci 0.178 dB, incoherent 0.161 dB -> split;
#: ``B/s`` 2.56 (25 GHz): xci 0.504 dB, incoherent 0.397 dB -> split.
#:
#: The boundary therefore lies between 1.28 and 1.71.  A finer sweep
#: (``scripts/h3_fwm_coherence_auto_audit.py --spacing-ghz 56 45 40 32``,
#: 5/10 spans, 6 dBm) places the crossover inside ``(1.42, 1.60)``: at 1.42
#: ``"xci"`` wins by 0.009-0.010 dB, at 1.60 ``"incoherent"`` wins by
#: 0.006 dB.  1.5 is the round value inside that measured transition band, and
#: it is a calibration of the overlap regime, not a per-case switch: the same
#: value applies to every comb because only the plan geometry enters.
FWM_AUTO_RATIO_THRESHOLD: float = 1.5

#: Accepted values of ``nli_fallback``.
#:
#: ``None`` (default) never imports, constructs or evaluates the numerical
#: kernel, so every number stays bit-for-bit the closed-form result.
#: ``"numerical"`` replaces the SCI/XCI noise powers of the *flagged* spans
#: (dense plan or low-dispersion / FWM regime) by the ``spm + xpm`` output of
#: :class:`~network_nlo_eval.models.numerical.NumericalISRSGNKernel`.  That
#: kernel already carries the FWM contribution as part of its double integral,
#: which is exactly the term the closed form cannot add without double
#: counting (task AT).  Accuracy is the numerical reference's; the price is
#: runtime, and it is paid only on the spans that need it.
NLI_FALLBACK_MODES: tuple[str | None, ...] = (None, "numerical")

#: Accepted values of ``nli_fallback_trigger``.
#:
#: ``"audit"`` (default) keeps the historical trigger set: the fallback fires
#: on a dense plan (``dense_grid_plan``) *or* where
#: :func:`dispersion_validity` reports ``fwm_required``.  ``"fwm_only"``
#: removes the dense-plan trigger, so the fallback is driven only by the
#: low-dispersion / zero-dispersion / out-of-band-anchor audit.  On a dense
#: plan that leaves the closed form (optionally with the displaced-third-wave
#: correction of ``dense_grid_fwm_fix``) in charge, which is what makes the
#: two options composable: their trigger sets no longer overlap.
NLI_FALLBACK_TRIGGERS: tuple[str, ...] = ("audit", "fwm_only")

#: Local dispersion magnitude (ps/nm/km) below which the PCFM/H3 closed form
#: is reported as needing the FWM term it does not carry.
#:
#: The kernels contain SPM and XCI only.  The GN double integral behind the
#: numerical reference also contains the FWM (Bragg-scattering) contribution,
#: which is phase-matched where ``beta2`` vanishes.  The value is *calibrated*
#: against that reference by ``scripts/h3_o_band_audit.py``: on the O band the
#: NLI error grows from +0.4 dB at 4-8 ps/nm/km to -0.8 dB at the zero-
#: dispersion crossing, so 4.0 ps/nm/km is the magnitude where the missing FWM
#: term leaves the C-band error budget.  It is a domain boundary, not a tuning
#: knob: the C/C+L/S+C+L bands stay above it (>= 5.6 ps/nm/km), the O band
#: does not.  See ``docs``/``H3_TASK_REPORT.md`` for the full table.
FWM_REQUIRED_MIN_DISPERSION_PS_NM_KM: float = 4.0

#: ``|D|`` below which a channel is treated as sitting on the zero-dispersion
#: crossing even when the crossing itself falls just outside the occupied
#: channel centres.  It is a discretisation guard: with 96 GBaud channels at
#: 100 GHz spacing the nearest centre can be ~0.1 nm from ``lambda0``, which
#: is already ~0.006 ps/nm/km; a half-ps/nm/km window keeps the sign-change
#: test robust without widening the physical claim.
ZERO_DISPERSION_NEAR_PS_NM_KM: float = 0.5


@dataclass(frozen=True)
class DispersionValidity:
    """Read-only low-dispersion / FWM applicability audit of one span plan.

    The audit depends only on the channel plan, the occupied mask and the
    fibre's ``D``/``S`` anchor, so it is pure geometry: it never looks at the
    NLI, the power profile or the Kerr kernel.  ``PCFM``/``H3`` carry no FWM
    term, so a span where ``beta2`` is small is outside the model's physics.

    Attributes
    ----------
    per_channel_dispersion_ps_nm_km : ndarray
        ``(N,)`` ``D(lambda_i) = D_ref + S (lambda_i - lambda_ref)`` at the
        channel centre, in model order (ps/nm/km).
    min_abs_dispersion_ps_nm_km : float
        Smallest ``|D|`` over the occupied channels; ``inf`` without an
        occupied channel.
    zero_dispersion_in_band : bool
        ``True`` when the zero-dispersion wavelength
        ``lambda0 = lambda_ref - D_ref / S`` falls inside the occupied band
        edges (or ``|D|`` of an occupied channel is below
        :data:`ZERO_DISPERSION_NEAR_PS_NM_KM`).  ``False`` for ``S = 0``.
    dispersion_reference_in_band : bool
        ``True`` when the fibre's ``reference_wavelength_nm`` falls inside
        the occupied band.  The kernel takes a single ``beta2``/``beta3``
        at that wavelength, so an out-of-band anchor means the dispersion
        anchor is extrapolated, not measured, on this comb.
    fwm_required : bool
        ``True`` when at least one of the three reasons below applies.
    fwm_reason : str
        ``""`` or a ``+``-joined combination of
        ``"zero_dispersion_in_band"``, ``"low_dispersion"`` and
        ``"dispersion_reference_out_of_band"``.
    """

    per_channel_dispersion_ps_nm_km: NDArrayFloat
    min_abs_dispersion_ps_nm_km: float
    zero_dispersion_in_band: bool
    dispersion_reference_in_band: bool
    fwm_required: bool
    fwm_reason: str


def dispersion_validity(
    f_abs_hz: NDArrayFloat,
    bandwidth_hz: NDArrayFloat,
    active: NDArrayBool | None,
    *,
    dispersion_parameter_d: float,
    dispersion_slope_s: float,
    reference_wavelength_nm: float,
    min_dispersion_ps_nm_km: float = FWM_REQUIRED_MIN_DISPERSION_PS_NM_KM,
) -> DispersionValidity:
    """Audit whether the span sits in the low-dispersion / FWM regime.

    The model is geometry only.  ``D(lambda) = D_ref + S (lambda - lambda_ref)``
    with ``lambda = c / f``; the reference anchor of the *kernel* is a single
    ``beta2``/``beta3`` pair taken at ``reference_wavelength_nm``, so the audit
    reports separately whether that anchor is inside the occupied band.

    Parameters
    ----------
    f_abs_hz : ndarray
        ``(N,)`` channel centre frequencies (Hz), any order.
    bandwidth_hz : ndarray
        ``(N,)`` occupied channel bandwidths (Hz).
    active : ndarray | None
        Optional ``(N,)`` mask of launched channels; ``None`` means all.
    dispersion_parameter_d : float
        ``D`` at the reference wavelength (ps/nm/km).
    dispersion_slope_s : float
        ``S`` (ps/nm^2/km); ``0`` disables the crossing test.
    reference_wavelength_nm : float
        Anchor wavelength of the kernel (nm).
    min_dispersion_ps_nm_km : float
        Magnitude threshold of the ``low_dispersion``
        reason (ps/nm/km).

    Returns
    -------
    DispersionValidity
        A :class:`DispersionValidity`; the per-channel array is in the input
        order and ``nan`` for an inactive channel.

    Raises
    ------
    ValueError
        The arrays are not one-dimensional or differ in shape.
    """
    frequencies = np.asarray(f_abs_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidth_hz, dtype=np.float64)
    if frequencies.ndim != 1 or bandwidths.shape != frequencies.shape:
        msg = (
            f"f_abs_hz and bandwidth_hz must be one-dimensional and equal, got {frequencies.shape} "
            f"and {bandwidths.shape}."
        )
        raise ValueError(msg)
    mask = (
        np.ones(frequencies.shape, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool)
    )
    if mask.shape != frequencies.shape:
        msg = f"active must have shape {frequencies.shape}, got {mask.shape}."
        raise ValueError(msg)

    d_ref = float(dispersion_parameter_d)
    slope = float(dispersion_slope_s)
    lambda_ref_nm = float(reference_wavelength_nm)
    lambdas_nm = C_LIGHT / frequencies * 1e9
    dispersion = d_ref + slope * (lambdas_nm - lambda_ref_nm)
    per_channel = np.where(mask, dispersion, np.nan)

    if np.any(mask):
        abs_d = np.abs(dispersion[mask])
        min_abs = float(np.min(abs_d))
        # The occupied band is center +/- half bandwidth; a zero-dispersion wavelength inside it
        # changes the sign of beta2.
        band_low = float(np.min(frequencies[mask] - 0.5 * bandwidths[mask]))
        band_high = float(np.max(frequencies[mask] + 0.5 * bandwidths[mask]))
        sign_change = bool(
            np.any(dispersion[mask] < 0.0) and np.any(dispersion[mask] > 0.0)
        )
    else:
        min_abs = float("inf")
        band_low = band_high = float("nan")
        sign_change = False

    zero_crossing = False
    if slope != 0.0:
        lambda0_nm = lambda_ref_nm - d_ref / slope
        if lambda0_nm > 0.0:
            f0_hz = C_LIGHT / (lambda0_nm * 1e-9)
            zero_crossing = bool(band_low <= f0_hz <= band_high)
    zero_in_band = bool(
        zero_crossing
        or sign_change
        or (np.isfinite(min_abs) and min_abs <= ZERO_DISPERSION_NEAR_PS_NM_KM)
    )
    reference_in_band = bool(
        np.isfinite(band_low)
        and band_low <= C_LIGHT / (lambda_ref_nm * 1e-9) <= band_high
    )
    low_dispersion = bool(
        np.isfinite(min_abs) and min_abs < float(min_dispersion_ps_nm_km)
    )

    reasons = []
    if zero_in_band:
        reasons.append("zero_dispersion_in_band")
    if low_dispersion:
        reasons.append("low_dispersion")
    if np.isfinite(band_low) and not reference_in_band:
        reasons.append("dispersion_reference_out_of_band")

    return DispersionValidity(
        per_channel_dispersion_ps_nm_km=np.ascontiguousarray(per_channel),
        min_abs_dispersion_ps_nm_km=min_abs,
        zero_dispersion_in_band=zero_in_band,
        dispersion_reference_in_band=reference_in_band,
        fwm_required=bool(reasons),
        fwm_reason="+".join(reasons),
    )


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

    Parameters
    ----------
    f_abs_hz : ndarray
        ``(N,)`` channel centre frequencies (Hz), any order.
    bandwidth_hz : ndarray
        ``(N,)`` occupied channel bandwidths (Hz).
    active : ndarray | None
        Optional ``(N,)`` boolean mask; ``None`` means every channel.

    Returns
    -------
    tuple[float, float]
        ``(min_spacing_hz, spacing_to_bandwidth)``; both ``inf`` when there is
        no pair to compare.

    Raises
    ------
    ValueError
        The arrays are not one-dimensional or differ in shape.
    """
    frequencies = np.asarray(f_abs_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidth_hz, dtype=np.float64)
    if frequencies.ndim != 1 or bandwidths.shape != frequencies.shape:
        msg = (
            f"f_abs_hz and bandwidth_hz must be one-dimensional and equal, got {frequencies.shape} "
            f"and {bandwidths.shape}."
        )
        raise ValueError(msg)
    mask = (
        np.ones(frequencies.shape, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool)
    )
    if mask.shape != frequencies.shape:
        msg = f"active must have shape {frequencies.shape}, got {mask.shape}."
        raise ValueError(msg)
    selected = np.sort(frequencies[mask])
    if selected.size < 2:
        return float("inf"), float("inf")
    min_spacing = float(np.min(np.diff(selected)))
    max_bandwidth = float(np.max(bandwidths[mask]))
    if max_bandwidth <= 0.0:
        return min_spacing, float("inf")
    return min_spacing, min_spacing / max_bandwidth


def fwm_coherence_ratio(
    f_abs_hz: NDArrayFloat,
    bandwidth_hz: NDArrayFloat,
    active: NDArrayBool | None = None,
) -> float:
    """Return ``max(B) / min_spacing`` of a channel plan.

    It is the inverse of :func:`dense_grid_metrics`' ratio and is the metric
    :data:`FWM_AUTO_RATIO_THRESHOLD` is calibrated against: values above one
    mean neighbouring spectra overlap, and the larger the value the stronger
    the overlap.  ``dense_grid_fwm_coherence="auto"`` uses it to decide
    whether the displaced-third-wave term may leave the coherent XCI
    amplitude (see :func:`resolve_fwm_coherence_mode`).

    A plan with fewer than two active channels, or with two centres at the
    same frequency, reports ``inf``: it can never reach the threshold, so the
    auto mode stays on ``"xci"``.

    Parameters
    ----------
    f_abs_hz : ndarray
        ``(N,)`` channel centre frequencies (Hz), any order.
    bandwidth_hz : ndarray
        ``(N,)`` occupied channel bandwidths (Hz).
    active : ndarray | None
        Optional ``(N,)`` boolean mask; ``None`` means every channel.

    Returns
    -------
    float
        The dimensionless ratio; ``inf`` when there is no comparable pair.

    Raises
    ------
    ValueError
        The arrays are not one-dimensional or differ in shape.
    """
    frequencies = np.asarray(f_abs_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidth_hz, dtype=np.float64)
    if frequencies.ndim != 1 or bandwidths.shape != frequencies.shape:
        msg = (
            f"f_abs_hz and bandwidth_hz must be one-dimensional and equal, got {frequencies.shape} "
            f"and {bandwidths.shape}."
        )
        raise ValueError(msg)
    mask = (
        np.ones(frequencies.shape, dtype=bool)
        if active is None
        else np.asarray(active, dtype=bool)
    )
    if mask.shape != frequencies.shape:
        msg = f"active must have shape {frequencies.shape}, got {mask.shape}."
        raise ValueError(msg)
    selected = np.sort(frequencies[mask])
    if selected.size < 2:
        return float("inf")
    spacing = np.diff(selected)
    positive = spacing[spacing > 0.0]
    if positive.size == 0:
        return float("inf")
    return float(np.max(bandwidths[mask])) / float(np.min(positive))


def resolve_fwm_coherence_mode(
    mode: str,
    ratio: float,
    *,
    fwm_fix_applies: bool,
) -> str:
    """Resolve ``dense_grid_fwm_coherence`` to the mode actually applied.

    ``"xci"`` and ``"incoherent"`` are returned unchanged.  ``"auto"`` returns
    ``"incoherent"`` only when the displaced-third-wave term is actually built
    (``dense_grid_fwm_fix=True`` on a dense plan, i.e. ``fwm_fix_applies``)
    *and* the plan ratio reaches :data:`FWM_AUTO_RATIO_THRESHOLD`; otherwise
    it returns ``"xci"``, so non-dense plans, plans without the opt-in and
    single-channel plans stay bit-for-bit on the historical path.

    Parameters
    ----------
    mode : str
        One of :data:`FWM_COHERENCE_MODES`.
    ratio : float
        :func:`fwm_coherence_ratio` of the channel plan.
    fwm_fix_applies : bool
        ``dense_grid_fwm_fix`` ANDed with the plan-level
        ``dense_grid`` flag.

    Returns
    -------
    str
        ``"xci"`` or ``"incoherent"``.

    Raises
    ------
    ValueError
        ``mode`` is not one of :data:`FWM_COHERENCE_MODES`.
    """
    if mode not in FWM_COHERENCE_MODES:
        msg = f"dense_grid_fwm_coherence must be one of {FWM_COHERENCE_MODES}, got {mode!r}."
        raise ValueError(msg)
    if mode != "auto":
        return str(mode)
    if fwm_fix_applies and float(ratio) >= FWM_AUTO_RATIO_THRESHOLD:
        return "incoherent"
    return "xci"


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
    xci_gated : ndarray
        ``(N,)`` boolean, ``True`` where the exact XCI island integral was
        actually applied (``edge_fix=True`` and the row correction is
        above :data:`XCI_EDGE_MATERIALITY`).
    xci_row_correction : ndarray
        ``(N,)`` relative change of the row's total XCI weight produced by
        the exact island integral, before the materiality gate.  It is
        computed even when ``edge_fix=False`` so that the diagnostic can
        say how wrong the uncorrected row is.
    xci_s_min : ndarray
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
    xci_s_below_switch : ndarray
        ``(N,)`` boolean, ``xci_s_min < XCI_EDGE_VALIDITY_SWITCH``.
    sci_x : ndarray
        ``(N,)`` signed ``X = pi**2 beta2_eff_sci B**2 L`` of the SCI
        kernel.  ``|X|`` is the distance from the zero-dispersion
        crossing of the channel.
    sci_taylor : ndarray
        ``(N,)`` boolean, the small-``X`` Taylor branch was used (only
        possible with ``edge_fix=True``).
    sci_x_below_switch : ndarray
        ``(N,)`` boolean, ``|sci_x| < SCI_X_SERIES_SWITCH``.  ``True``
        with ``sci_taylor=False`` means the unstable recurrence was used.
    sci_x_abs_min : float
        Minimum ``|X|`` over the active channels (dimensionless).  ``0``
        when no channel is active.
    power_tilt_db : float
        Launch-power tilt ``10 log10(max P / min P)`` over the active
        channels (dB).
    gain_tilt_db : float
        Span-gain tilt ``10 log10(max g / min g)`` with
        ``g = P_out/P_in`` over the active channels (dB).  This is the
        pump/ISRS gain tilt seen by the NLI profile.
    pump_active : bool
        Whether the span carries at least one positive-power pump.
    pump_directions : tuple[str, ...]
        Directions of the active pumps, in the order of the pump list.
        Empty when no pump is active.
    pump_depletion : ndarray
        ``(N_p,)`` depletion fraction ``1 - P_out/P_passive`` of every
        active pump, where ``P_passive = P_in exp(-alpha_p L)`` is the
        undepleted (own-attenuation) output.  Measuring against the
        passive output keeps the 25 dB intrinsic pump attenuation out of
        the number; a value near one means the signals consumed most of
        the pump.  Forward pumps are measured at ``z=L`` and backward
        pumps at ``z=0``, following
        :class:`~network_nlo_eval.models.raman_ode.PaperRamanSpanResult`.
    pump_depletion_max : float
        Largest entry of ``pump_depletion`` (``0`` without pumps).
    flagged : ndarray
        ``(N,)`` boolean, the channel is in a regime the two-tier fix
        had to correct (``xci_row_correction > XCI_EDGE_MATERIALITY`` or
        ``sci_x_below_switch``) or the span exceeds the tilt threshold.
    valid : ndarray
        ``(N,)`` boolean, the reported NLI is expected inside the model
        error budget.  A corrected edge row is still valid (the exact
        integral is applied); an uncorrected one is not.
    confidence : ndarray
        ``(N,)`` score in ``[0, 1]``.  It is a coarse ordering, not a
        calibrated probability: 1 means inside the validated domain,
        ~0.6 means "usable after the fix", ~0.2 means "known wrong".
    effective_tilt_db : float
        The tilt actually compared against
        :data:`POWER_TILT_VALIDITY_DB` (dB).  For the base kernel this is
        ``power_tilt_db``; the H3 Raman kernel uses
        ``max(power_tilt_db, gain_tilt_db)`` when pumps are active.  It
        makes the flag decision readable without re-deriving the rule.
    tilt_confidence_informational : float
        Continuous, **non-gating** tilt score in ``[0, 1]`` from
        :func:`informational_tilt_confidence`.  It is 1 below
        :data:`TILT_INFORMATIONAL_KNEE_DB` and decays smoothly to 0 at
        :data:`TILT_INFORMATIONAL_ZERO_DB`, so a caller can apply a soft
        margin before the hard flag fires.  It never changes ``valid``,
        ``flagged`` or ``confidence``.
    min_channel_spacing_hz : float
        Closest spacing between two active channels (Hz); ``inf`` when the
        span carries fewer than two active channels.
    spacing_to_bandwidth : float
        ``min_channel_spacing_hz / max bandwidth`` over the active
        channels.  A value below :data:`DENSE_GRID_SPACING_RATIO` means
        adjacent rectangular spectra overlap and the XCI closed form is
        outside its derivation domain.
    dense_grid : bool
        ``True`` when ``spacing_to_bandwidth`` is below
        :data:`DENSE_GRID_SPACING_RATIO`.  It is a domain warning only:
        it ORs into ``flagged`` for every active channel but never changes
        ``valid``, ``confidence`` or the NLI itself.
    raman_converged : bool
        Whether the coupled Raman solve behind the report converged.
        ``False`` also invalidates every active channel: the NLI profile
        is not a solution.  Always ``True`` for the no-pump/base kernel.
    raman_steps : int
        RK4 steps actually used by the coupled solve (``0`` without a
        Raman solve).  It makes an explicitly under-resolved kernel
        visible even when the solve happens to converge.
    raman_wideband_multipump : bool
        ``True`` when the comb reaches S/E spectrum *and* the pump set is
        forward or multi-pump.  ``raman_steps="auto"`` uses the extended
        96-step branch there; the flag is reported so a caller can see
        that the default needed it.  It is a *numerical convergence risk*
        for pumps outside the signal band, not a physical-configuration
        error; see ``raman_pump_signal_overlap`` for the latter.
    raman_pump_signal_overlap : bool
        ``True`` when at least one active pump frequency falls inside the
        occupied signal band of the span (channel centres widened by half
        the channel bandwidth on each side, plus the kernel's
        ``pump_signal_overlap_guard_hz``).  That configuration is not a
        physical DRA: a pump inside the data band cannot be separated from
        the signals, so the whole span is outside the model's domain.  It
        invalidates every active channel (``valid=False``,
        ``confidence<=0.2``, ``flagged=True``) *without* changing the
        point NLI/ASE/SNR.  It is independent of
        ``raman_wideband_multipump``: the latter can be ``True`` with all
        pumps outside the band, where it only signals the extended-step
        regime.  Always ``False`` for the no-pump/base kernel.
    raman_pump_overlap_channels : ndarray
        ``(N,)`` boolean in model order, the active channels invalidated
        by ``raman_pump_signal_overlap``.  It is the all-active mask when
        the flag is set (the pump contaminates every signal channel) and
        all-``False`` otherwise.
    raman_pump_min_offset_hz : float
        Smallest distance from an active pump frequency to the occupied
        signal band *edge*
        ``[min(f_i - B_i/2) - guard, max(f_i + B_i/2) + guard]`` (Hz),
        ``0`` when a pump lies inside, positive when every pump is
        outside, and ``inf`` without pumps or active channels.  It makes
        the overlap decision readable without re-deriving the band.
    lumped_raman_approximation : bool
        ``True`` for the pure PCFM kernel with ``raman_model="lumped"``
        and active pumps.  The lumped model multiplies only the span-end
        power and does not model pump depletion or the intra-span profile,
        so it is not a trustworthy QoT for pumped links.  It never changes
        the point result; the error budget refuses it.
    xci_overlap_pairs : ndarray
        ``(N,)`` number of active interfering channels whose rectangular
        spectrum overlaps the CUT band (``|f_j - f_i| < (B_i + B_j)/2``).
        It is the pair-level counterpart of ``dense_grid``: the plan flag
        says the grid is outside the non-overlapping derivation domain,
        this array says which rows actually see an overlapping pair.
        It never changes ``valid``, ``confidence`` or the NLI.
    xci_overlap_weight_fraction : ndarray
        ``(N,)`` share of the CUT row's closed-form XCI weight
        ``sum_j |w_xci[i, j]|`` that comes from overlapping pairs.  A
        large value means the printed Eq. (24) island integral is being
        applied where its disjoint-spectrum premise does not hold.  Read
        only.
    min_guard_hz : float
        Minimum over active pairs of ``|f_j - f_i| - (B_i + B_j)/2`` (Hz).
        Negative means at least one pair overlaps; ``inf`` when no pair is
        active.
    sci_taylor_fraction : float
        Fraction of active channels that take the small-``X`` SCI Taylor
        branch.  ``0`` means the branch is irrelevant on this plan.
    dense_grid_fix_applied : bool
        ``True`` when the opt-in ``dense_grid_fix`` actually replaced at
        least one CUT row of this span by the exact overlapping-band XCI
        integral.  Always ``False`` with the default ``dense_grid_fix=False``.
    dense_grid_overlap_correction : ndarray
        ``(N,)`` relative change of the CUT row's total XCI weight produced
        by the overlapping-band integral, before the materiality gate.
        Zero where no pair overlaps (``|df| < (B_i+B_j)/2``) and everywhere with
        ``dense_grid_fix=False``.
    dense_grid_overlap_pairs : ndarray
        ``(N,)`` number of interfering channels that entered the
        overlapping-band integral (``|df| < (B_i+B_j)/2``).  Zero with
        ``dense_grid_fix=False`` and on every non-overlapping grid.
    lumped_loss_positions_m : ndarray
        ``(K,)`` positions (m) of the lumped loss elements configured on
        the span, sorted ascending with duplicate positions merged.
        Empty when the span has none.
    lumped_loss_db : ndarray
        ``(K,)`` loss of the same elements (dB), non-negative.
    lumped_loss_in_ode : bool
        ``True`` when the coupled Raman ODE applied the lumped loss
        internally (active pump *and* at least one element).  ``False``
        without pumps, where the base PCFM SPP carries the same list.
    lumped_loss_segments : int
        Number of integration segments the lumped loss splits the span
        into (``K + 1``); ``1`` without elements.  The coupled ODE
        allocates its RK4 steps proportionally to the segment lengths.
    raman_auto_under_resolved : bool
        ``True`` when the opt-in ``raman_resolution_check`` compared the
        solved step count with a finer one and the normalised signal
        profile, the span-end gain or the newly generated Raman ASE moved
        by more than ``raman_resolution_tol``.  It is a numerical
        resolution warning only: ``valid``, ``confidence`` and every point
        value (NLI/ASE/SNR) are left untouched.
    raman_resolution_check_steps : int
        Finer RK4 step count that produced the largest change during the
        check ladder (``0`` when no check ran).  Never above
        :data:`~network_nlo_eval.models.pcfm_h3_raman.RAMAN_RESOLUTION_MAX_STEPS`.
    raman_resolution_change : float
        Largest relative change seen during the check (fraction, not dB).
        ``1.0`` is the neutral default for "not checked".
    raman_resolution_checked : bool
        ``True`` only when the check actually ran and the finer solves
        were usable.  It is ``False`` with the default option, at or above
        the maximum step count, and when the finer solve did not converge.
    min_abs_dispersion_ps_nm_km : float
        Smallest ``|D(lambda)|`` over the occupied channels (ps/nm/km),
        with ``D(lambda) = D_ref + S (lambda - lambda_ref)`` at the channel
        centre.  The closed-form kernels carry no FWM term, so this is the
        distance of the comb from the phase-matched regime.
    zero_dispersion_in_band : bool
        ``True`` when the zero-dispersion wavelength of the fibre falls
        inside the occupied band edges (or an occupied channel has
        ``|D| <= ZERO_DISPERSION_NEAR_PS_NM_KM``).
    dispersion_reference_in_band : bool
        ``True`` when ``reference_wavelength_nm`` falls inside the occupied
        band.  The kernel evaluates a single ``beta2``/``beta3`` pair at
        that wavelength; an out-of-band anchor means the dispersion used by
        the closed form is an extrapolation, not a measurement, on this
        comb.  It is a separate error source from the missing FWM term.
    fwm_required : bool
        ``True`` when the span sits in the low-dispersion / zero-dispersion
        / out-of-band-anchor regime where FWM cannot be neglected.  With
        ``fwm_validity_mode="flag"`` (default) it only ORs into
        ``flagged``; ``"reject"`` also clears ``valid``.  It never changes
        the point NLI/ASE/SNR.
    fwm_reason : str
        ``""`` or a ``+``-joined subset of ``"zero_dispersion_in_band"``,
        ``"low_dispersion"`` and ``"dispersion_reference_out_of_band"``.
    fwm_validity_mode : str
        The mode this span was evaluated with (``"flag"`` or ``"reject"``),
        echoed so the report is self-describing.
    """

    xci_gated: NDArrayBool
    xci_row_correction: NDArrayFloat
    xci_s_min: NDArrayFloat
    xci_s_below_switch: NDArrayBool
    sci_x: NDArrayFloat
    sci_taylor: NDArrayBool
    sci_x_below_switch: NDArrayBool
    sci_x_abs_min: float
    power_tilt_db: float
    gain_tilt_db: float
    pump_active: bool
    pump_directions: tuple[str, ...]
    pump_depletion: NDArrayFloat
    pump_depletion_max: float
    flagged: NDArrayBool
    valid: NDArrayBool
    confidence: NDArrayFloat
    effective_tilt_db: float = 0.0
    tilt_confidence_informational: float = 1.0
    min_channel_spacing_hz: float = float("inf")
    spacing_to_bandwidth: float = float("inf")
    dense_grid: bool = False
    raman_converged: bool = True
    raman_steps: int = 0
    raman_wideband_multipump: bool = False
    raman_pump_signal_overlap: bool = False
    raman_pump_overlap_channels: NDArrayBool = field(
        default_factory=lambda: np.zeros(0, dtype=bool)
    )
    raman_pump_min_offset_hz: float = float("inf")
    lumped_raman_approximation: bool = False
    xci_overlap_pairs: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    xci_overlap_weight_fraction: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    min_guard_hz: float = float("inf")
    sci_taylor_fraction: float = 0.0
    dense_grid_fix_applied: bool = False
    dense_grid_overlap_correction: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    dense_grid_overlap_pairs: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    lumped_loss_positions_m: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    lumped_loss_db: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    lumped_loss_in_ode: bool = False
    lumped_loss_segments: int = 1
    raman_auto_under_resolved: bool = False
    raman_resolution_check_steps: int = 0
    raman_resolution_change: float = 1.0
    raman_resolution_checked: bool = False
    min_abs_dispersion_ps_nm_km: float = float("inf")
    zero_dispersion_in_band: bool = False
    dispersion_reference_in_band: bool = True
    fwm_required: bool = False
    fwm_reason: str = ""
    fwm_validity_mode: str = "flag"


@dataclass(frozen=True)
class DenseGridDiagnostics:
    """Read-only diagnostics of dense-grid error sources.

    The fields describe overlap, XCI weights and corrections, SCI Taylor-branch
    use, and spatial-profile fit residuals. They do not change NLI, ASE or SNR.
    Arrays use model channel order. Power-dependent fit residuals are None when
    no input power is supplied.

    Attributes
    ----------
    span_length_m : float
        Span length used for the diagnostics (m).
    dense_grid : bool
        Whether spacing to bandwidth is below the dense-grid threshold.
    min_channel_spacing_hz : float
        Minimum active-channel spacing (Hz), or infinity with fewer than two.
    spacing_to_bandwidth : float
        Minimum spacing divided by maximum bandwidth; below one means overlap.
    min_guard_hz : float
        Minimum ``abs(delta_f) - (B_i+B_j)/2`` across active pairs (Hz).
    xci_overlap_pairs : ndarray
        Number of overlapping interferers per channel of interest.
    xci_overlap_weight_fraction : ndarray
        Fraction of the closed-form XCI row weight contributed by overlap pairs.
    xci_edge_gated : ndarray
        Channels where exact XCI edge integration was applied.
    xci_row_correction : ndarray
        Relative XCI row change from edge integration before the materiality gate.
    xci_edge_quadrature_points : int
        Number of Gauss-Legendre nodes used for edge integration.
    sci_x_abs_min : float
        Minimum absolute SCI X over active channels.
    sci_taylor : ndarray
        SCI small-X Taylor-branch indicator for each channel.
    sci_taylor_fraction : float
        Fraction of active channels using the Taylor branch.
    spp_fit_residual : ndarray or None
        Maximum absolute profile-fit residual per channel.
    spp_fit_residual_max, spp_fit_residual_rms, spp_fit_residual_rel_rms : float or None
        Maximum, RMS, and profile-normalized RMS fit residuals.
    dense_grid_fix_applied : bool
        Whether the optional dense-grid fix replaced at least one row.
    dense_grid_overlap_correction : ndarray
        Relative row change from overlap integration before the materiality gate.
    dense_grid_overlap_pairs : ndarray
        Number of interfering channels integrated as overlapping pairs per row.
    """

    span_length_m: float
    dense_grid: bool
    min_channel_spacing_hz: float
    spacing_to_bandwidth: float
    min_guard_hz: float
    xci_overlap_pairs: NDArrayFloat
    xci_overlap_weight_fraction: NDArrayFloat
    xci_edge_gated: NDArrayBool
    xci_row_correction: NDArrayFloat
    xci_edge_quadrature_points: int
    sci_x_abs_min: float
    sci_taylor: NDArrayFloat
    sci_taylor_fraction: float
    spp_fit_residual: NDArrayFloat | None = None
    spp_fit_residual_max: float | None = None
    spp_fit_residual_rms: float | None = None
    spp_fit_residual_rel_rms: float | None = None
    dense_grid_fix_applied: bool = False
    dense_grid_overlap_correction: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    dense_grid_overlap_pairs: NDArrayFloat = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )


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
#: Table I is ``Phi = 0``. The PCFM paper's Gaussian-shaped test format does
#: not establish validation of every QAM format or of this repository's
#: zero-roll-off setting. The model classes fix Phi to zero; their egn option
#: applies fitted Gaussian-limit factors rather than selecting a QAM format.
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
    global _J_CONSTANT  # noqa: PLW0603 - cache the expensive integral
    if _J_CONSTANT is None:
        from scipy.integrate import quad  # noqa: PLC0415 - evaluated lazily

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IntegrationWarning)
            low = quad(
                lambda t: sici(t)[0] / t,
                0.0,
                1.0,
                limit=400,
                epsabs=1e-12,
                epsrel=1e-12,
            )[0]
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
    from network_nlo_eval.models.pcfm_kernels import _j_tail_jit  # noqa: PLC0415

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

    Parameters
    ----------
    beta2_accum_ps2 : ndarray
        ``(N,)`` accumulated effective dispersion of every
        channel at the input of the current span (ps^2).
    roll_off_cut : float
        Roll-off of the CUT.
    roll_off_nch : float
        Roll-off of the interfering channel.
    phi : float
        EGN format constant.

    Returns
    -------
    ndarray
        ``(N,)`` correction factors.
    """
    a = EGN_A_COEFFICIENTS
    acc = np.abs(np.asarray(beta2_accum_ps2, dtype=np.float64))
    inner = (
        a["a1"]
        + a["a2"] * phi ** a["a3"]
        + a["a4"] * phi ** a["a5"] * (1.0 + a["a6"] * (acc + a["a7"]) ** a["a8"])
    )
    outer = (
        1.0 + a["a19"] * roll_off_cut ** a["a20"] + a["a21"] * roll_off_nch ** a["a22"]
    )
    return cast("NDArrayFloat", outer * inner)


def egn_rho_cut(
    beta2_accum_ps2: NDArrayFloat,
    symbol_rate_hz: float,
    roll_off_cut: float = EGN_ROLLOFF,
    phi: float = EGN_PHI,
) -> NDArrayFloat:
    """Machine-learning factor ``rho_CUT`` of Zefreh et al. 2020, Eq. (14).

    ``rho_CUT = (1 + a23 r_CUT^a24) { a9 + a10 Phi^a11 + a12 Phi^a13
    (1 + a14 R_CUT^a15 + a16 [|beta2_acc| + a17]^a18) }``

    Parameters
    ----------
    beta2_accum_ps2 : ndarray
        ``(N,)`` accumulated effective dispersion (ps^2).
    symbol_rate_hz : float
        Symbol rate of the channels (Hz).  The reference paper
        uses ``R_CUT`` in TBaud.
    roll_off_cut : float
        Roll-off of the CUT.
    phi : float
        EGN format constant.

    Returns
    -------
    ndarray
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
        * (
            1.0
            + a["a14"] * rate_tbaud ** a["a15"]
            + a["a16"] * (acc + a["a17"]) ** a["a18"]
        )
    )
    outer = 1.0 + a["a23"] * roll_off_cut ** a["a24"]
    return cast("NDArrayFloat", outer * inner)


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

    Parameters
    ----------
    beta : float
        Angular frequency of the cosine (rad per unit of ``t``).
    t : ndarray
        Lower limit of the integral (positive).

    Returns
    -------
    ndarray
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
    return cast(
        "NDArrayFloat",
        -f * sin_beta / beta
        - df * cos_beta / beta**2
        + d2f * sin_beta / beta**3
        + d3f * cos_beta / beta**4
        - d4f * sin_beta / beta**5,
    )


class _LorentzianAntiderivative:
    r"""Tabulate ``C_m(t) = int_0^t cos(m alpha L s)/(1+s**2) ds``.

    The table is built with the composite trapezoid rule corrected by the
    leading Euler-Maclaurin term, so that the quadrature error is fourth
    order in the step.  ``C_0(t) = arctan(t)`` is exact and is not tabulated.
    Beyond the table the asymptotic expansion of the tail is used, anchored
    by the Fourier transform of a Lorentzian, ``C_m(oo) = (pi/2) exp(-m
    alpha L)``.

    Parameters
    ----------
    alpha_npm_m : float
        Power attenuation of the fibre (Np/m).
    span_length_m : float
        Length of one span (m).
    m_max : int
        Largest index ``m`` to tabulate.
    t_end : float
        Largest ``t`` the table has to cover.
    step : float
        Step of the table.
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
            out[inside] = sign[inside] * np.interp(
                magnitude[inside], self.t, self.tables[m]
            )
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

    Parameters
    ----------
    half_band_hz : float
        Half of the channel bandwidth (Hz).
    rate : float
        ``4 pi**2 |beta2|``, the mismatch rate per unit frequency
        product (s^2/m).
    alpha_npm_m : float
        Power attenuation of the fibre (Np/m).
    n_frequency : int
        Number of observed frequencies.
    n_quadrature : int
        Number of Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
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
    nodes, weights = np.polynomial.legendre.leggauss(n_quadrature)  # type: ignore[no-untyped-call]
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

    Parameters
    ----------
    beta2_abs_s2_m : float
        ``|beta_2|`` (s^2/m).
    alpha_npm_m : float
        Power attenuation of the fibre (Np/m).
    bandwidth_hz : float
        Channel bandwidth (Hz).
    span_length_m : float
        Length of one span (m).
    distances_m : ndarray
        ``(K,)`` distances between the two copies (m).
    n_frequency : int
        Number of observed frequencies across the band.
    n_quadrature : int
        Number of Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    ndarray
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
    weight_hi = (1.0 + exp_alpha_l**2) * tables.C(
        0, t_hi
    ) - 2.0 * exp_alpha_l * tables.C(1, t_hi)
    weight_lo = (1.0 + exp_alpha_l**2) * tables.C(
        0, t_lo
    ) - 2.0 * exp_alpha_l * tables.C(1, t_lo)
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

    Parameters
    ----------
    beta2_abs_s2_m : float
        ``|beta_2|`` (s^2/m).
    alpha_npm_m : float
        Power attenuation of the fibre (Np/m).
    bandwidth_hz : ndarray
        ``(N,)`` channel bandwidths (Hz).
    span_length_m : float
        Length of one span (m).
    distances_m : ndarray
        ``(K,)`` distances between the two copies (m).
    n_frequency : int
        Number of observed frequencies across the band.
    n_quadrature : int
        Number of Gauss-Legendre nodes per sub-interval.

    Returns
    -------
    ndarray
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

    Parameters
    ----------
    beta2_abs_s2_m : float
        ``|beta_2|`` (s^2/m).
    bandwidth_hz : ndarray
        ``(N,)`` channel bandwidths (Hz).
    distance_m : float
        Distance between the two copies (m).
    partner_offset_hz : ndarray
        ``(N,)`` offset to the interfering channel (Hz).

    Returns
    -------
    ndarray
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

    Parameters
    ----------
    beta2_abs_s2_m : float
        ``|beta_2|`` (s^2/m).
    alpha_npm_m : float
        Power attenuation of the fibre (Np/m).
    bandwidth_hz : float
        Channel bandwidth (Hz).
    span_length_m : float
        Length of one span (m).
    distance_m : float
        Distance between the two copies (m).
    partner_offset_hz : float
        Offset to the interfering channel (Hz).
    n_offset : int
        Number of Gauss-Legendre nodes of the interfering channel.
    n_mismatch : int
        Number of Gauss-Legendre nodes of the mismatch variable.

    Returns
    -------
    float
        The visibility of the pair, in ``[-1, 1]``.
    """
    half_band = 0.5 * bandwidth_hz
    rate = 4.0 * np.pi**2 * beta2_abs_s2_m * span_length_m
    abar = alpha_npm_m * span_length_m
    exp_abar = float(np.exp(-abar))
    offset_nodes, offset_weights = np.polynomial.legendre.leggauss(n_offset)  # type: ignore[no-untyped-call]
    offsets = partner_offset_hz + half_band * offset_nodes
    offset_weights = half_band * offset_weights
    mismatch_nodes, mismatch_weights = np.polynomial.legendre.leggauss(n_mismatch)  # type: ignore[no-untyped-call]
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
            numerator_factor / (abar * (1.0 + mismatch * mismatch)) / (rate * offset)
        )
        numerator += offset_weight * float(
            np.sum(weights * integrand * np.cos(theta * distance_m / span_length_m))
        )
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

    Parameters
    ----------
    beta2_eff_xci : ndarray
        ``(N, N)`` effective dispersion of the pair (s^2/m).
    y2_hz : ndarray
        ``(N, N)`` upper edge of the interfering band relative to the
        CUT centre (Hz).
    half_b_cut_hz : ndarray
        ``(N,)`` half bandwidth of the CUT channel (Hz).
    alpha_npm_m : float
        Fibre power attenuation (Np/m).

    Returns
    -------
    ndarray
        ``(N, N)`` parameter ``s`` of every pair.
    """
    # Use 4*pi**2 because the input is half bandwidth; 2*pi**2 returned half the physical value.
    return (
        4.0
        * np.pi**2
        * np.abs(beta2_eff_xci)
        * y2_hz
        * half_b_cut_hz[:, None]
        / alpha_npm_m
    )


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
) -> tuple[NDArrayFloat, NDArrayBool, NDArrayFloat]:
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

    Parameters
    ----------
    w_xci : ndarray
        ``(N, N)`` closed-form XCI weight matrix, Eq. (24).
    k_xci_matrix : ndarray
        ``(N, N)`` kernel factor ``log / (2 pi |beta2_eff|)``.
    log_xci : ndarray
        ``(N, N)`` logarithmic frequency factor of Eq. (24).
    gamma_xci : ndarray
        ``(N, N)`` nonlinear coefficient of the pair.
    beta2_eff_xci : ndarray
        ``(N, N)`` effective dispersion of the pair (s^2/m).
    f_abs_hz : ndarray
        ``(N,)`` channel centre frequencies (Hz).
    b_hz : ndarray
        ``(N,)`` channel bandwidths (Hz).
    alpha_npm_m : float
        Fibre power attenuation (Np/m).
    length_m : float
        Span length (m).
    validity_switch : float
        Pairs with ``s`` below this value are recomputed.
    materiality : float
        Row-level relative change above which the correction is
        applied.
    n_quad : int
        Gauss-Legendre nodes of the ``y`` integral.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        The tuple ``(w_edge, gated, fraction)``: the corrected weight matrix,
        the boolean mask of the corrected CUT rows, and the relative change of
        every row before gating.
    """
    n_ch = f_abs_hz.size
    delta = np.abs(f_abs_hz[:, None] - f_abs_hz[None, :])
    half_b = 0.5 * b_hz
    y1 = delta - half_b[None, :]
    y2 = delta + half_b[None, :]
    # Use the interferer upper edge and CUT half bandwidth to compute the validity parameter s.
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
    # The table covers valid s; its asymptotic tail handles larger arguments.
    max_s = float(
        np.max(np.abs(c_mix[:, None] * y * (0.5 * b_cut)[:, None] / alpha_npm_m))
    )
    table = _LorentzianAntiderivative(alpha_npm_m, float(length_m), 1, max_s)
    s = c_mix[:, None] * y * (0.5 * b_cut)[:, None] / alpha_npm_m
    # Use the limit at small s, where arctan(s) and C_1(s) nearly cancel.
    small = np.abs(s) < 1.0e-4
    c1 = table.C(1, s)
    bracket = (1.0 + exp_2ab) * np.arctan(s) - 2.0 * exp_ab * c1
    bracket = np.where(small, (1.0 - exp_ab) ** 2 * s, bracket)
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = 2.0 * bracket / (alpha_npm_m * c_mix[:, None] * y)
    # Scale Gauss-Legendre weights on [-1, 1] by the interval Jacobian.
    integral = np.sum(
        0.5 * (y_hi - y_lo)[:, None] * weights[None, :] * integrand, axis=1
    )

    # The integral weight is pi/(2*alpha)*(1-exp(-2*alpha*L)); using the squared loss term
    # overstates short-span correction.
    f_inf = (1.0 / alpha_npm_m) * 0.5 * np.pi * (1.0 - exp_2ab)
    k_corrected = np.pi * integral / f_inf
    # At zero beta2, the integral becomes W(0)*B_CUT*B_NCH and its normalized limit is
    # 2*B_CUT*B_NCH/alpha*tanh(alpha*L/2).
    zero = b_abs == 0.0
    if np.any(zero):
        k_corrected[zero] = (
            2.0
            * b_cut[zero]
            * b_nch[zero]
            / alpha_npm_m
            * (1.0 - exp_ab)
            / (1.0 + exp_ab)
        )

    k_edge = np.array(k_xci_matrix, dtype=np.float64, copy=True)
    k_edge[rows, cols] = k_corrected
    w_candidate = gamma_xci**2 * k_edge
    candidate_delta = np.abs(w_candidate - w_xci).sum(axis=1)
    scale = w_xci.sum(axis=1)
    fraction = np.divide(
        candidate_delta, scale, out=np.zeros_like(scale), where=scale > 0.0
    )
    gated = fraction > materiality
    w_edge[gated, :] = w_candidate[gated, :]
    return w_edge, gated, fraction


def _xci_overlap_weight_matrix(
    w_base: NDArrayFloat,
    gamma_xci: NDArrayFloat,
    beta2_eff_xci: NDArrayFloat,
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm_m: float,
    length_m: float,
    materiality: float = XCI_EDGE_MATERIALITY,
    n_quad: int = XCI_EDGE_QUADRATURE_POINTS,
    force_rows: NDArrayBool | None = None,
) -> tuple[NDArrayFloat, NDArrayBool, NDArrayFloat, NDArrayFloat]:
    r"""Replace the XCI weight of overlapping channel pairs by the exact band integral.

    The printed Eq. (24) and the exact island integral of
    :func:`_xci_edge_weight_matrix` are derived for two *disjoint*
    rectangular spectra, ``|f_i - f_j| >= (B_i + B_j) / 2``.  A dense comb
    violates that condition even when the interfering band has not yet
    crossed the CUT centre: at 50 GHz spacing with 64 GHz occupied bandwidth
    the two rectangles already overlap by 14 GHz, while
    ``y1 = |f_i - f_j| - B_j/2`` is still positive.  The closed form then
    evaluates its logarithmic factor outside the derivation domain and the
    XCI island no longer starts at ``y1``; the pair is silently kept with a
    weight that is too optimistic.

    This function integrates the overlapping-band contribution on the folded
    absolute-frequency axis: the integral starts at zero frequency difference
    (plus the negative half when ``y1 < 0``) and ends at ``y2``.  The GN
    weight ``W`` is even and its antiderivative ``F`` is odd, so the
    ``1/y`` singularity at ``y = 0`` is removable and the limit is

    .. code-block:: text

        lim_{y -> 0} 2 F(c y B_CUT / 2) / (c y) = W(0) B_CUT ,
        c = 4 pi**2 |beta2_eff| .

    The integral is therefore split at ``y = 0`` and evaluated with the same
    Gauss-Legendre rule as the edge integral; the nodes never sample
    ``y = 0`` and the small-``s`` expansion of the bracket guards the
    removable singularity.  The corrected kernel is ``K = pi I / F_inf`` with
    ``F_inf = int_0^inf W dt``, exactly as in the edge integral, so the two
    corrections share their normalisation.

    Only pairs that fail the non-overlap condition are touched.  Rows whose
    relative change stays below ``materiality`` keep the old weights bit for
    bit, unless they are in ``force_rows`` (the dense-grid opt-in).  Every
    other entry of ``w_base`` is copied unchanged.

    Parameters
    ----------
    w_base : ndarray
        ``(N, N)`` XCI weight matrix currently in use (closed form or
        edge-corrected).  Entries of the corrected pairs are replaced.
    gamma_xci : ndarray
        ``(N, N)`` nonlinear coefficient of the pair.
    beta2_eff_xci : ndarray
        ``(N, N)`` effective dispersion of the pair (s^2/m).
    f_abs_hz : ndarray
        ``(N,)`` channel centre frequencies (Hz).
    b_hz : ndarray
        ``(N,)`` channel bandwidths (Hz).
    alpha_npm_m : float
        Fibre power attenuation (Np/m).
    length_m : float
        Span length (m).
    materiality : float
        Row-level relative change above which the correction is
        applied.
    n_quad : int
        Gauss-Legendre nodes per half interval.
    force_rows : ndarray | None
        Optional ``(N,)`` boolean mask of CUT rows that take the
        correction even below ``materiality``.  Set by the kernel when the
        plan is a dense grid and ``dense_grid_fix=True``.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        ``(w_out, gated, fraction, pairs)``: the corrected matrix, the mask of
        the CUT rows actually replaced, the relative row change before gating,
        and the number of overlapping pairs per row that entered the exact
        integral.
    """
    n_ch = f_abs_hz.size
    delta = np.abs(f_abs_hz[:, None] - f_abs_hz[None, :])
    half_b = 0.5 * b_hz
    y1 = delta - half_b[None, :]
    y2 = delta + half_b[None, :]
    off_diagonal = ~np.eye(n_ch, dtype=bool)
    # Rectangular spectra overlap when abs(df) < (B_i+B_j)/2, including edge overlap before either
    # center crosses the other band.
    overlap = (delta < (half_b[:, None] + half_b[None, :])) & (y2 > 0.0) & off_diagonal

    w_out = np.array(w_base, dtype=np.float64, copy=True)
    gated = np.zeros(n_ch, dtype=bool)
    fraction = np.zeros(n_ch, dtype=np.float64)
    pairs = overlap.sum(axis=1).astype(np.float64)
    rows, cols = np.nonzero(overlap)
    if rows.size == 0:
        return w_out, gated, fraction, pairs

    b_abs = np.abs(beta2_eff_xci[rows, cols])
    b_cut = b_hz[rows]
    b_nch = b_hz[cols]
    y_lo = y1[rows, cols]
    y_hi = y2[rows, cols]
    c_mix = 4.0 * np.pi**2 * b_abs
    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))  # type: ignore[no-untyped-call]

    ab = alpha_npm_m * float(length_m)
    exp_ab = float(np.exp(-ab))
    exp_2ab = float(np.exp(-2.0 * ab))
    # Tabulate the largest absolute s on both half-intervals; use the asymptotic tail outside.
    extreme_y = np.maximum(np.abs(y_lo), np.abs(y_hi))
    max_s = float(np.max(np.abs(c_mix * extreme_y * (0.5 * b_cut) / alpha_npm_m)))
    table = _LorentzianAntiderivative(alpha_npm_m, float(length_m), 1, max_s)

    # The removable y=0 limit is W(0)*B_CUT.
    limit = (1.0 - exp_ab) ** 2 / alpha_npm_m**2 * b_cut

    def _integrand(
        y: NDArrayFloat,
        c: NDArrayFloat,
        b_cut_band: NDArrayFloat,
        limit_band: NDArrayFloat,
    ) -> NDArrayFloat:
        """Integrand ``2 F(c y b/2)/(alpha c y)`` with its ``y -> 0`` limit."""
        s = c * y * (0.5 * b_cut_band) / alpha_npm_m
        small = np.abs(s) < 1.0e-4
        bracket = (1.0 + exp_2ab) * np.arctan(s) - 2.0 * exp_ab * table.C(1, s)
        bracket = np.where(small, (1.0 - exp_ab) ** 2 * s, bracket)
        denominator = alpha_npm_m * c * y
        zero = denominator == 0.0
        safe = np.where(zero, 1.0, denominator)
        return np.where(
            zero, np.broadcast_to(limit_band[:, None], y.shape), 2.0 * bracket / safe
        )

    integral = np.zeros(rows.size, dtype=np.float64)
    # On [y1, 0], s is negative and odd F keeps the integrand positive.
    negative = y_lo < 0.0
    if np.any(negative):
        low = y_lo[negative]
        jacobian = 0.5 * (0.0 - low)
        y = jacobian[:, None] * nodes[None, :] + 0.5 * low[:, None]
        integral[negative] += np.sum(
            jacobian[:, None]
            * weights[None, :]
            * _integrand(
                y,
                c_mix[negative, None],
                b_cut[negative, None],
                limit[negative],
            ),
            axis=1,
        )
    # Positive half-interval [0, y2].
    jacobian = 0.5 * y_hi
    y = jacobian[:, None] * nodes[None, :] + jacobian[:, None]
    integral += np.sum(
        jacobian[:, None]
        * weights[None, :]
        * _integrand(y, c_mix[:, None], b_cut[:, None], limit),
        axis=1,
    )

    f_inf = (1.0 / alpha_npm_m) * 0.5 * np.pi * (1.0 - exp_2ab)
    k_corrected = np.pi * integral / f_inf
    # At zero beta2 the integral reduces to W(0)*B_CUT*B_NCH.
    zero_beta = b_abs == 0.0
    if np.any(zero_beta):
        k_corrected[zero_beta] = (
            2.0
            * b_cut[zero_beta]
            * b_nch[zero_beta]
            / alpha_npm_m
            * (1.0 - exp_ab)
            / (1.0 + exp_ab)
        )

    w_candidate = np.array(w_base, dtype=np.float64, copy=True)
    w_candidate[rows, cols] = gamma_xci[rows, cols] ** 2 * k_corrected
    delta_row = np.abs(w_candidate - w_base).sum(axis=1)
    scale = w_base.sum(axis=1)
    fraction = np.divide(delta_row, scale, out=np.zeros_like(scale), where=scale > 0.0)
    # If the closed-form row is zero, any added weight is a 100% correction.
    fraction = np.where((scale <= 0.0) & (delta_row > 0.0), np.inf, fraction)
    gated = fraction > materiality
    rows_with_pair = np.zeros(n_ch, dtype=bool)
    rows_with_pair[rows] = True
    if force_rows is not None:
        gated = gated | (np.asarray(force_rows, dtype=bool) & rows_with_pair)
    w_out[gated, :] = w_candidate[gated, :]
    return w_out, gated, fraction, pairs


def _fwm_displaced_weight_terms(
    gamma_xci: NDArrayFloat,
    beta2_eff_xci: NDArrayFloat,
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm_m: float,
    length_m: float,
    n_quad: int = FWM_QUADRATURE_POINTS,
    min_relative_weight: float = 1.0e-9,
) -> tuple[NDArrayInt, NDArrayInt, NDArrayInt, NDArrayFloat]:
    r"""Displaced-third-wave (FWM/MCI) weight of every channel triple.

    The closed-form XCI of Eq. (24) — and the two exact repairs above — integrate
    the GN kernel over the *second* input band only, implicitly assuming that the
    generated third wave is co-located with that pump (``r == ell``).  The GN
    double integral of Eq. (4) also contains the triples whose product lands in a
    *different* band (``r != ell``): the inter-channel four-wave mixing / MCI
    transfer that no PCFM term represents.  This helper evaluates those triples
    with the same Lorentzian antiderivative and the same Gauss-Legendre rule as
    :func:`_xci_overlap_weight_matrix`.

    For a CUT ``i`` the integral is

    .. code-block:: text

        K(i, ell, r) = (pi / F_inf) *
            int_{y_lo}^{y_hi} dy [F(c (d+y) x_hi(y)) - F(c (d+y) x_lo(y))]
                                 / (c (d + y)) ,

        c = 4 pi**2 |beta2_eff(i, ell)| ,   d = f_ell - f_i ,

        y_lo = max(-B_ell/2, f_r - f_ell - B_r/2 - B_i/2)
        y_hi = min( B_ell/2, f_r - f_ell + B_r/2 + B_i/2)
        x_lo(y) = max(-B_i/2, f_r - f_ell - B_r/2 - y)
        x_hi(y) = min( B_i/2, f_r - f_ell + B_r/2 - y)

    i.e. ``x`` runs over the CUT band, ``y`` over the pump band, and the
    ``(x, y)`` pair is restricted to the products that land inside band ``r``.
    The ``y`` integral is split where ``d + y = 0`` because the integrand has a
    removable singularity there; its limit is ``W(0) (x_hi - x_lo)`` with
    ``W(0) = (1 - e^{-alpha L})**2 / alpha**2``, exactly as in
    :func:`_xci_overlap_weight_matrix`.

    Summing ``r`` over *all* bands (including ``r == ell``) reproduces the exact
    full-band XCI integral; the ``r == ell`` part is the one the closed form
    already models (to 0.3 % on the pairwise kernel), so only the ``r != ell``
    triples are returned here.  When ``ell == i`` the two input waves are both in
    the CUT band, which the numerical reference counts once instead of twice, so
    the weight is halved.

    Parameters
    ----------
    gamma_xci : ndarray
        ``(N, N)`` nonlinear coefficient of the pump pair.
    beta2_eff_xci : ndarray
        ``(N, N)`` effective dispersion of the pump pair
        (s^2/m).
    f_abs_hz : ndarray
        ``(N,)`` channel centre frequencies (Hz), internal order.
    b_hz : ndarray
        ``(N,)`` channel bandwidths (Hz).
    alpha_npm_m : float
        Fibre power attenuation (Np/m).
    length_m : float
        Span length (m).
    n_quad : int
        Gauss-Legendre nodes per ``y`` interval.  This is the FWM knob
        (:data:`FWM_QUADRATURE_POINTS`), independent of the XCI edge /
        overlap quadrature: the two integrals have different smoothness.
    min_relative_weight : float
        Triples whose ``gamma**2 K`` is below this
        fraction of the largest weight are dropped.  They are physically
        negligible and only cost memory.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        ``(i_idx, ell_idx, r_idx, weight)``: the triple indices and the
        ``(n_triples,)`` weight ``gamma(i, ell)**2 K(i, ell, r)``.  All arrays
        are empty when no triple can reach another band (sparse comb).
    """
    n_ch = int(f_abs_hz.size)
    empty_int = np.zeros(0, dtype=np.int_)
    empty_float = np.zeros(0, dtype=np.float64)
    if n_ch < 2:
        return empty_int, empty_int, empty_int, empty_float

    bi = 0.5 * np.asarray(b_hz, dtype=np.float64)
    c_pair = 4.0 * np.pi**2 * np.abs(np.asarray(beta2_eff_xci, dtype=np.float64))
    gamma2 = np.asarray(gamma_xci, dtype=np.float64) ** 2

    ab = alpha_npm_m * float(length_m)
    exp_ab = float(np.exp(-ab))
    exp_2ab = float(np.exp(-2.0 * ab))
    f_inf = (1.0 / alpha_npm_m) * 0.5 * np.pi * (1.0 - exp_2ab)
    # Use the removable y=0 limit W(0), shared with the overlap integral.
    w_zero = (1.0 - exp_ab) ** 2 / alpha_npm_m**2
    exp_ab_sq = (1.0 - exp_ab) ** 2

    # Build candidate (ell, r) pairs within three maximum half-bandwidths once; filter precisely for
    # each CUT.
    max_half_b = float(np.max(bi)) if bi.size else 0.0
    reach_all = 3.0 * max_half_b
    ell_parts: list[NDArrayInt] = []
    r_parts: list[NDArrayInt] = []
    for ell in range(n_ch):
        offsets = f_abs_hz - f_abs_hz[ell]
        window = np.nonzero(np.abs(offsets) <= reach_all)[0]
        window = window[window != ell]
        if window.size:
            ell_parts.append(np.full(window.size, ell, dtype=np.int_))
            r_parts.append(window.astype(np.int_))
    if not ell_parts:
        return empty_int, empty_int, empty_int, empty_float
    pair_ell = np.concatenate(ell_parts)
    pair_r = np.concatenate(r_parts)

    nodes, weights = np.polynomial.legendre.leggauss(int(n_quad))  # type: ignore[no-untyped-call]
    # Tabulate the largest absolute s across the triplets; the tail handles larger values.
    max_offset = (
        float(np.max(np.abs(f_abs_hz[:, None] - f_abs_hz[None, :]))) + max_half_b
    )
    max_s = float(np.max(c_pair) * max_offset * max_half_b / alpha_npm_m) * 1.5
    table = _LorentzianAntiderivative(alpha_npm_m, float(length_m), 1, max_s)

    def _antiderivative(t: NDArrayFloat) -> NDArrayFloat:
        """``F(t) = int_0^t W d(rho)`` with the small-``s`` Taylor branch."""
        s = t / alpha_npm_m
        small = np.abs(s) < 1.0e-4
        bracket = (1.0 + exp_2ab) * np.arctan(s) - 2.0 * exp_ab * table.C(1, s)
        bracket = np.where(small, exp_ab_sq * s, bracket)
        return bracket / alpha_npm_m

    i_parts: list[NDArrayInt] = []
    ell_out: list[NDArrayFloat] = []
    r_out: list[NDArrayFloat] = []
    w_parts: list[NDArrayFloat] = []

    for i in range(n_ch):
        if bi[i] <= 0.0:
            continue
        d_ell = f_abs_hz[pair_ell] - f_abs_hz[i]
        drl = f_abs_hz[pair_r] - f_abs_hz[pair_ell]
        y_lo = np.maximum(-bi[pair_ell], drl - bi[pair_r] - bi[i])
        y_hi = np.minimum(bi[pair_ell], drl + bi[pair_r] + bi[i])
        # Apply the exact in-band condition; retain ell == i and halve its reference weight.
        keep = y_hi > y_lo
        if not np.any(keep):
            continue
        ell = pair_ell[keep]
        r = pair_r[keep]
        lo = y_lo[keep]
        hi = y_hi[keep]
        d = d_ell[keep]
        dr = drl[keep]
        c = c_pair[i, ell]

        # Split the y integral at the three changes of constraint and removable singularity; this
        # makes each subinterval smooth and convergent.
        kink_lo = dr - bi[r] + bi[i]
        kink_hi = dr + bi[r] - bi[i]
        kink_zero = -d
        cuts = np.stack(
            [
                lo,
                np.clip(kink_lo, lo, hi),
                np.clip(kink_hi, lo, hi),
                np.clip(kink_zero, lo, hi),
                hi,
            ],
            axis=1,
        )
        cuts.sort(axis=1)
        integral = np.zeros(ell.size, dtype=np.float64)
        for segment in range(cuts.shape[1] - 1):
            start = cuts[:, segment]
            stop = cuts[:, segment + 1]
            active = stop > start
            if not np.any(active):
                continue
            jacobian = 0.5 * (stop[active] - start[active])
            y = start[active, None] + jacobian[:, None] * (nodes[None, :] + 1.0)
            offset = y + d[active, None]
            x_lo = np.maximum(-bi[i], dr[active, None] - bi[r[active], None] - y)
            x_hi = np.minimum(bi[i], dr[active, None] + bi[r[active], None] - y)
            c_y = c[active, None] * offset
            zero = c_y == 0.0
            safe = np.where(zero, 1.0, c_y)
            values = (_antiderivative(c_y * x_hi) - _antiderivative(c_y * x_lo)) / safe
            values = np.where(zero, w_zero * (x_hi - x_lo), values)
            integral[active] += jacobian * np.sum(weights[None, :] * values, axis=1)

        k_pair = np.pi * integral / f_inf
        w = gamma2[i, ell] * k_pair
        # When ell == i both input waves share a band and the reference counts them once, versus
        # twice for XPM.
        same_band = ell == i
        if np.any(same_band):
            w[same_band] *= 0.5
        valid = np.isfinite(w) & (w > 0.0)
        if not np.any(valid):
            continue
        i_parts.append(np.full(np.count_nonzero(valid), i, dtype=np.int_))
        ell_out.append(ell[valid].astype(np.float64))
        r_out.append(r[valid].astype(np.float64))
        w_parts.append(w[valid])

    if not i_parts:
        return empty_int, empty_int, empty_int, empty_float
    i_idx = np.concatenate(i_parts)
    ell_idx = np.concatenate(ell_out).astype(np.int_)
    r_idx = np.concatenate(r_out).astype(np.int_)
    weight = np.concatenate(w_parts)
    floor = min_relative_weight * float(np.max(weight))
    keep = weight >= floor
    return i_idx[keep], ell_idx[keep], r_idx[keep], weight[keep]


def _coherent_factor(n_spans: int, visibility: NDArrayFloat) -> NDArrayFloat:
    """Compute the frequency-resolved inter-span coherent factor ``R``.

    ``visibility`` holds the visibility of the copies separated by one, two,
    ... spans.  The coherent factor is the visibility-weighted sum over the
    span pairs,

    .. code-block:: text

        R = N + 2 sum_{k=1}^{N-1} (N - k) V(k L) .

    A vanishing visibility gives ``R = N`` (incoherent accumulation), a
    visibility of one gives ``R = N**2`` (coherent accumulation).

    Parameters
    ----------
    n_spans : int
        Number of spans of the link.
    visibility : ndarray
        ``(N_spans - 1, N_channels)`` visibility.

    Returns
    -------
    ndarray
        The ``(N_channels,)`` coherent factor.
    """
    n = int(n_spans)
    if n <= 1:
        return np.ones(visibility.shape[-1], dtype=np.float64)
    weights = (n - np.arange(1, n, dtype=np.float64))[:, None]
    return cast("NDArrayFloat", n + 2.0 * (weights * visibility).sum(axis=0))


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Invert the coherent factor into an equivalent per-span phase ramp.

    The route layer sums the span amplitudes coherently, so ``N`` spans with
    a linear phase ramp ``psi`` give the Dirichlet kernel
    ``[sin(N psi/2)/sin(psi/2)]**2`` times the single-span noise.  That
    kernel decreases monotonically from ``N**2`` at ``psi = 0`` to ``0`` at
    ``psi = 2 pi/N``, so the wanted factor has a unique inverse and a
    bisection finds it.

    Parameters
    ----------
    n_spans : int
        Number of spans of the link.
    target : ndarray
        ``(N_channels,)`` wanted coherent factor.

    Returns
    -------
    ndarray
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
            (
                "Evaluates the exact XCI island integral where the closed form leaves its validity domain "
                "(zero-dispersion edge); stabilises the SCI kernel at small X. Default True."
            ),
        ),
        (
            "dense_grid_fix",
            (
                "Opt-in: replaces the XCI weight of pairs whose rectangular spectra overlap "
                "(y1 = |df| - B_j/2 <= 0) by the exact integral on the actual band, including y < 0. "
                "Default False keeps every number bit-for-bit identical."
            ),
        ),
        (
            "dense_grid_fwm_fix",
            (
                "Opt-in: adds the displaced-third-wave (FWM/MCI) triples (third wave outside the "
                "second pump band) on a dense plan, using the same quadrature as the XCI repair. "
                "Default False keeps every number bit-for-bit identical. Mutually exclusive with "
                "dense_grid_fix and with nli_fallback='numerical' under nli_fallback_trigger='audit'; "
                "allowed with nli_fallback='numerical' + nli_fallback_trigger='fwm_only'."
            ),
        ),
        (
            "dense_grid_fwm_coherence",
            (
                "How the dense-grid FWM term accumulates across spans: 'xci' (default) keeps it inside "
                "the XCI complex amplitude; 'incoherent' splits it out of xpm and power-accumulates it "
                "in the route layer; 'auto' picks 'incoherent' only when dense_grid_fwm_fix=True on a "
                "dense plan whose max(bandwidth)/min(spacing) reaches 1.5 (C 25/37.5 GHz), and 'xci' "
                "elsewhere (C 50 GHz, non-dense). Only acts with dense_grid_fwm_fix=True on a dense plan."
            ),
        ),
        (
            "fwm_quadrature_points",
            (
                "Gauss-Legendre nodes per y interval of the opt-in displaced-third-wave (FWM/MCI) "
                "triple integral. Default 24 keeps every number bit-for-bit identical. Independent of "
                "the XCI edge/overlap quadrature: raising it must not move the XCI rows. Only read with "
                "dense_grid_fwm_fix=True on a dense plan."
            ),
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
            (
                "Explicit opt-in resolution profile. None (default) keeps every per-option default "
                "bit-for-bit. 'fast': XCI edge quadrature 12 nodes (NLI identical to 1e-11, 6-65% less "
                "first-call work). 'accurate': spp_steps=96 (and raman_steps=96 on pcfm_h3_raman), the "
                "converged reference of docs/h3-pareto-report.md. A profile never overrides an explicitly "
                "different spp_steps/raman_steps: that raises ValueError."
            ),
        ),
        (
            "raman_model",
            (
                "Compatibility setting for unpumped links. Active pumps require a distributed-Raman model."
            ),
        ),
        (
            "raman_pumps",
            "Compatibility pump list; positive pump power requires pcfm_h3_raman or hybrid_raman_paper.",
        ),
        (
            "raman_gain_model",
            (
                "'gaussian' (default), 'linear' CZ approximation, 'table' (normalized spectrum given by "
                "raman_gain_table), a .json/.csv Raman gain table file path, or a RamanGainModel instance "
                "(use RamanGainModel.from_file(path)). The file/instance entry is loaded once at construction "
                "time, never in the hot path."
            ),
        ),
        (
            "raman_gain_table",
            (
                "(offsets_hz, gains) normalized Raman gain table used when raman_gain_model='table'. "
                "Linear interpolation, zero outside the table. Mutually exclusive with a RamanGainModel "
                "instance or a table file path."
            ),
        ),
        (
            "raman_ase",
            "Add the distributed Raman ASE before the span EDFA. Default True when Raman is enabled.",
        ),
        (
            "validity_mode",
            (
                "'warn' (default) records the per-channel applicability report without touching any "
                "number; 'strict' additionally replaces the SNR of invalid channels with NaN."
            ),
        ),
        (
            "fwm_validity_mode",
            (
                "'flag' (default) marks spans in the low-dispersion / zero-dispersion / out-of-band-"
                "anchor regime in the validity report (fwm_required, fwm_reason) without touching any "
                "number; 'reject' additionally clears 'valid' on those channels. The PCFM kernels carry "
                "no FWM term, so that regime is outside the model physics. Neither mode changes the "
                "point NLI/ASE/SNR."
            ),
        ),
        (
            "nli_power_convention",
            (
                "'total' (default): power_in_w is the per-channel dual-polarization total power and the "
                "printed 16/27 and 32/27 GN weights apply as-is. 'per_pol': power_in_w is "
                "per-polarization power, so the NLI kernels and the ISRS profile are driven with 2*P and "
                "the NLI and span-end signal power are halved. Declaring the wrong side scales the NLI by 8. "
                "plain pcfm has no launch_power_convention and does not scale EDFA/booster ASE, so 'per_pol' "
                "here is an NLI-only comparison (mixed coordinate) and warns at construction; use "
                "pcfm_h3_raman(launch_power_convention='per_pol') for an end-to-end per-pol link."
            ),
        ),
        (
            "nli_fallback",
            (
                "Opt-in reference fallback. None (default) never imports or constructs the numerical "
                "kernel and stays bit-for-bit closed form. 'numerical' replaces the SCI/XCI of the spans "
                "the validity audit flags (dense plan, or low-dispersion / zero-dispersion / out-of-band "
                "anchor) by the spm+xpm of the numerical ISRS-GN kernel, which already contains the FWM "
                "average; the closed form is replaced, never corrected, because adding a FWM subset to "
                "the PCFM XCI double counts it. Accuracy is the numerical reference's, the cost is "
                "runtime and is paid only on the flagged spans (C 100 GHz and C+L 75 GHz never trigger). "
                "Rejected together with lumped_loss, which the numerical kernel does not model."
            ),
        ),
        (
            "nli_fallback_trigger",
            (
                "'audit' (default) keeps the historical trigger set: a dense plan or the low-dispersion / "
                "zero-dispersion / out-of-band-anchor audit. 'fwm_only' drops the dense-plan trigger, so "
                "only dispersion_validity(...).fwm_required fires the fallback. The dense plan is then "
                "left to the closed form, optionally with dense_grid_fwm_fix; that is why this trigger "
                "is what allows dense_grid_fwm_fix + nli_fallback='numerical' (the two trigger sets are "
                "disjoint). Under 'audit' the same combination raises ValueError because the dense points "
                "would be replaced wholesale and the FWM term double counted. Ignored when nli_fallback "
                "is None."
            ),
        ),
    ),
)


class PCFMSpanKernel:
    """Single-span kernel of the polynomial closed-form model.

    The kernel owns everything that only depends on the fibre and on the
    channel plan: the per-channel attenuation, effective area and dispersion,
    the effective dispersion of every island, the pairwise XCI matrices and
    the polynomial fit operator.

    Parameters
    ----------
    f_abs_hz : ndarray
        ``(N,)`` channel centre frequencies (Hz).
    b_hz : ndarray
        ``(N,)`` channel bandwidths (Hz).
    fiber_config : FiberSpanConfig
        Fibre parameters of the span.
    poly_degree : int
        Degree ``N_p`` of the SPP polynomial, Eq. (23).
    symbol_rate_hz : float
        Symbol rate (Hz), used by the EGN correction.
    egn : bool
        Apply the machine-learning correction of Zefreh et al. 2020.
    lumped_loss : Sequence[tuple[float, float]] | None
        Optional list of ``(position_m, loss_db)`` of lumped loss
        elements inside the span.  The paper treats lumped loss in
        Section IV-D.  The list is forwarded by the model from
        :attr:`~network_nlo_eval.network.elements.FiberSpanConfig.lumped_loss`;
        duplicate positions are merged by summing their dB.
    spp_steps : int
        Number of RK4 steps of the power-profile integration.
    edge_fix : bool
        Evaluate the exact XCI island integral on the CUT rows whose
        closed-form weight is materially outside its asymptotic validity
        domain, and use the small-``X`` Taylor branch of the SCI kernel.
    cache_spans : bool
        Memoise the SPP integration and the closed-form kernels of
        a span state, so that the identical spans of a transparent link
        are evaluated once.
    nli_power_convention : str
        Power convention of ``power_in_w`` for the NLI
        kernels and the ISRS profile.  ``"total"`` (default) is the
        per-channel dual-polarization total power with the printed
        ``16/27``/``32/27`` weights; ``"per_pol"`` treats the input as
        per-polarization power, drives the kernels and the SPP with
        ``2 * P``, and halves the per-channel NLI and the span-end power.
        See the module docstring.
    xci_edge_quadrature_points : int
        Gauss-Legendre nodes of the exact XCI
        island integral.  Only used for the CUT rows flagged by the
        ``edge_fix`` gate.  Defaults to
        :data:`XCI_EDGE_QUADRATURE_POINTS`; the sweep in
        ``docs/h3-pareto-report.md`` shows 12 nodes already reproduce the
        64-node result to 1e-11, so :data:`RESOLUTION_PROFILES` may lower
        it without moving any number.
    fwm_quadrature_points : int
        Gauss-Legendre nodes per ``y`` interval of the
        opt-in displaced-third-wave (FWM/MCI) triple integral.  Defaults to
        :data:`FWM_QUADRATURE_POINTS` (24).  Independent of
        ``xci_edge_quadrature_points`` on purpose: raising it must not
        move the XCI edge/overlap rows.  Only read when
        ``dense_grid_fwm_fix=True`` on a dense plan, so every other path
        stays bit-for-bit unchanged.
    dense_grid_fix : bool
        Opt-in overlap-aware XCI correction (default ``False``).
        When ``True`` and the plan is dense, the pairs whose interfering
        spectra overlap (``|df| < (B_i+B_j)/2``) are integrated over
        the actual rectangular band, including the ``y < 0`` half where the
        closed form does not exist.  Rows with any such pair are forced
        through the materiality gate on a dense plan; every other row and
        every other pair stays bit-for-bit unchanged.
    dense_grid_fwm_fix : bool
        Opt-in displaced-third-wave (FWM/MCI) correction
        (default ``False``).  When ``True`` and the plan is dense, the
        triples whose generated third wave lands outside the second pump's
        band (``r != ell``) are integrated with the same Lorentzian
        antiderivative and Gauss-Legendre rule as the XCI overlap repair,
        and their power is added to the XCI.  The closed-form XCI of
        Eq. (24) only represents the co-located (``r == ell``) part, so this
        is a disjoint term, not a double count; it is the PCFM counterpart
        of the ``fwm`` subset of the numerical ISRS-GN reference, but it is
        evaluated from the model's own dispersion/band/profile constants
        and never reads the numerical output.  Every non-dense plan and
        every sparse comb produce no triple and stay bit-for-bit unchanged.
        Cannot be combined with ``dense_grid_fix`` (that repair already
        supplies the missing energy on the overlapping rows, so adding the
        triple term would double count it) nor with
        ``nli_fallback="numerical"`` (the fallback replaces the NLI
        wholesale).  The term is added before the inter-span phase factor,
        so it inherits the XCI visibility; the numerical reference instead
        power-accumulates it, which is exact for the near-incoherent FWM
        average but is an approximation here.
    dense_grid_fwm_coherence : str
        ``"xci"`` (default), ``"incoherent"`` or
        ``"auto"``.  ``"xci"`` keeps the displaced-third-wave term inside
        ``xpm``, so it inherits the XCI inter-span phase (the historical
        behaviour).  On a dense plan with ``dense_grid_fwm_fix=True``,
        ``"incoherent"`` removes the term from ``xpm`` and returns it as
        :attr:`SpanFiberOutput.fwm_noise_w`; the route layer accumulates it
        as a power across spans and adds it back to the path XPM.
        ``"auto"`` selects between the two from the plan geometry alone
        (``max(B)/min_spacing >=`` :data:`FWM_AUTO_RATIO_THRESHOLD`, see
        :func:`resolve_fwm_coherence_mode`) and falls back to ``"xci"``
        everywhere the split cannot act.  One span, every non-dense plan
        and the default value stay bit-for-bit unchanged.
    fwm_validity_mode : str
        ``"flag"`` (default) or ``"reject"``.  The model
        carries no FWM term, so a comb near the zero-dispersion point is
        outside its physics.  ``"flag"`` only ORs ``fwm_required`` into
        ``flagged`` (point NLI/ASE/SNR and ``valid`` unchanged);
        ``"reject"`` also clears ``valid`` on the affected channels.
    nli_fallback : str | None
        ``None`` (default) or ``"numerical"``.  ``None`` never
        imports nor constructs the numerical kernel and leaves every
        number bit-for-bit unchanged.  ``"numerical"`` replaces the
        ``spm_noise_w``/``xpm_noise_w`` of the spans that the applicability
        audit flags (dense plan, i.e. ``dense_grid_plan``, or
        ``dispersion_validity(...).fwm_required``) by the ``spm + xpm``
        part of :class:`~network_nlo_eval.models.numerical.NumericalISRSGNKernel`,
        driven with the physical dual-polarization total power.  The
        closed-form SCI/XCI is *replaced*, not corrected, because the
        numerical double integral already contains the FWM average: adding
        a FWM subset on top of the PCFM XCI double counts it (task AT).
        The span output power, the ASE and the EDFA/ROADM handling stay
        the PCFM ones.  A dense C grid (37.5/25 GHz) and a low-dispersion
        O band trigger it; C 100 GHz and C+L 75 GHz do not.  With
        ``nli_power_convention="per_pol"`` the fallback is driven by
        ``2 * P`` and halved back, exactly like the closed form.  The
        numerical result is cached per ``(span length, power)``, so a
        transparent link pays the double integral once.  Lumped loss is
        rejected because the numerical kernel does not model it.
    nli_fallback_trigger : str
        ``"audit"`` (default) or ``"fwm_only"``.  The
        first keeps the historical trigger set (dense plan or
        ``fwm_required``); the second fires the fallback only where
        :func:`dispersion_validity` reports ``fwm_required``, so a dense
        plan is left to the closed form.  This is what makes
        ``dense_grid_fwm_fix=True`` + ``nli_fallback="numerical"`` legal:
        the fallback handles the low-dispersion / O-band spans and the
        fix handles the dense ones, with disjoint trigger sets.  Under
        ``"audit"`` that combination still raises ``ValueError`` because
        the dense points would be replaced and the FWM term double
        counted.  The option has no effect when ``nli_fallback`` is
        ``None``.
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
        fwm_quadrature_points: int = FWM_QUADRATURE_POINTS,
        dense_grid_fix: bool = False,
        dense_grid_fwm_fix: bool = False,
        dense_grid_fwm_coherence: str = "xci",
        fwm_validity_mode: str = "flag",
        nli_fallback: str | None = None,
        nli_fallback_trigger: str = "audit",
    ) -> None:
        """Prepare the per-channel constants of the span.

        Raises
        ------
        ValueError
            The polynomial degree or the integration resolution
            is not positive, a lumped loss element is out of the span,
            ``fwm_validity_mode`` is not ``"flag"``/``"reject"``, or
            ``nli_fallback``/``nli_fallback_trigger`` is invalid (or
            ``nli_fallback`` is combined with a lumped loss, which the
            numerical kernel does not model).
        """
        if poly_degree < 0:
            msg = f"poly_degree must be >= 0, got {poly_degree}."
            raise ValueError(msg)
        if spp_steps < 1:
            msg = f"spp_steps must be >= 1, got {spp_steps}."
            raise ValueError(msg)
        if int(xci_edge_quadrature_points) < 2:
            msg = f"xci_edge_quadrature_points must be >= 2, got {xci_edge_quadrature_points}."
            raise ValueError(msg)
        if int(fwm_quadrature_points) < 2:
            msg = f"fwm_quadrature_points must be >= 2, got {fwm_quadrature_points}."
            raise ValueError(msg)
        if nli_fallback not in NLI_FALLBACK_MODES:
            msg = f"nli_fallback must be one of {NLI_FALLBACK_MODES}, got {nli_fallback!r}."
            raise ValueError(msg)
        if nli_fallback_trigger not in NLI_FALLBACK_TRIGGERS:
            msg = f"nli_fallback_trigger must be one of {NLI_FALLBACK_TRIGGERS}, got {nli_fallback_trigger!r}."
            raise ValueError(msg)
        if nli_fallback is not None and lumped_loss:
            msg = (
                "nli_fallback='numerical' cannot be combined with lumped_loss: the numerical "
                "ISRS-GN kernel has no lumped-loss treatment, so the fallback would silently "
                "evaluate a different fibre than the closed form."
            )
            raise ValueError(msg)
        if dense_grid_fwm_fix and dense_grid_fix:
            msg = (
                "dense_grid_fwm_fix=True cannot be combined with dense_grid_fix=True: the "
                "overlap repair of the XCI rows already supplies the energy that the displaced "
                "triples would add, so enabling both double counts it (task BB audit)."
            )
            raise ValueError(msg)
        if (
            dense_grid_fwm_fix
            and nli_fallback is not None
            and nli_fallback_trigger != "fwm_only"
        ):
            msg = (
                "dense_grid_fwm_fix=True cannot be combined with nli_fallback='numerical' under "
                "nli_fallback_trigger='audit': the audit trigger replaces spm/xpm wholesale on the "
                "dense plan, where the FWM fix is already added, so the two would double count. "
                "Use nli_fallback_trigger='fwm_only' to keep the fallback on the low-dispersion / "
                "FWM spans only, which leaves the dense plan to the FWM fix."
            )
            raise ValueError(msg)
        if dense_grid_fwm_coherence not in FWM_COHERENCE_MODES:
            msg = (
                "dense_grid_fwm_coherence must be one of "
                f"{FWM_COHERENCE_MODES}, got {dense_grid_fwm_coherence!r}."
            )
            raise ValueError(msg)

        self.poly_degree = int(poly_degree)
        self.n_poly = self.poly_degree + 1
        self.spp_steps = int(spp_steps)
        self.xci_edge_quadrature_points = int(xci_edge_quadrature_points)
        self.fwm_quadrature_points = int(fwm_quadrature_points)
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.egn = bool(egn)
        self.edge_fix = bool(edge_fix)
        self.dense_grid_fix = bool(dense_grid_fix)
        self.dense_grid_fwm_fix = bool(dense_grid_fwm_fix)
        self.dense_grid_fwm_coherence = str(dense_grid_fwm_coherence)
        if fwm_validity_mode not in FWM_VALIDITY_MODES:
            msg = f"fwm_validity_mode must be one of {FWM_VALIDITY_MODES}, got {fwm_validity_mode!r}."
            raise ValueError(msg)
        self.fwm_validity_mode = str(fwm_validity_mode)
        self.nli_fallback = nli_fallback
        self.nli_fallback_trigger = str(nli_fallback_trigger)
        # Construct the optional numerical fallback lazily; cache by span length and power and reset
        # diagnostics on each evaluation.
        self._fiber_config = fiber_config
        self._fallback_kernel: NumericalISRSGNKernel | None = None
        self._fallback_cache: dict[
            tuple[int, bytes], tuple[NDArrayFloat, NDArrayFloat]
        ] = {}
        self._fwm_required_cache: dict[bytes, bool] = {}
        self.last_nli_fallback_applied = False
        self.last_fallback_nli_w: NDArrayFloat | None = None
        self.fallback_evaluations = 0
        self.fallback_seconds = 0.0
        # Fiber dispersion anchors are fixed; occupied-band limits and minimum abs(D) are recomputed
        # from active channels.
        self.dispersion_parameter_d = float(fiber_config.dispersion_parameter_d)
        self.dispersion_slope_s = float(fiber_config.dispersion_slope_s)
        self.reference_wavelength_nm = float(fiber_config.reference_wavelength_nm)

        self.cache_spans = bool(cache_spans)
        self.nli_power_convention = resolve_nli_power_convention(nli_power_convention)
        # Small-X SCI Taylor coefficients depend only on polynomial order; edge_fix=False disables
        # this branch.
        self._sci_small_x_switch = SCI_X_SERIES_SWITCH if self.edge_fix else 0.0
        self._sci_small_x_coeff = (
            _sci_small_x_coefficients(2 * self.n_poly + 1)
            if self.edge_fix
            else np.zeros((1, 1), dtype=np.float64)
        )
        # Cache XCI validity corrections lazily by span length rounded to micrometers.
        self._xci_edge_cache: dict[
            int, tuple[NDArrayFloat, NDArrayBool, NDArrayFloat]
        ] = {}
        # Cache overlap corrections only when dense_grid_fix is enabled.
        self._xci_dense_cache: dict[
            int, tuple[NDArrayFloat, NDArrayBool, NDArrayFloat, NDArrayFloat]
        ] = {}
        # Cache displaced-wave FWM/MCI triplets only for an enabled dense plan; None means no
        # reachable triplets.
        self._fwm_weight_cache: dict[
            int, tuple[NDArrayInt, NDArrayInt, NDArrayInt, NDArrayFloat] | None
        ] = {}
        self.last_dense_fwm_fix_applied = False
        # Cache fiber- and span-length-dependent diagnostics by span length.
        self._xci_validity_cache: dict[
            int,
            tuple[NDArrayFloat, NDArrayBool, NDArrayFloat, NDArrayBool, NDArrayBool],
        ] = {}
        # Dense-grid barriers, overlaps and XCI row weights depend only on the channel plan.
        self._overlap_cache: (
            tuple[NDArrayFloat, NDArrayBool, NDArrayFloat, NDArrayFloat] | None
        ) = None
        # Memoize span state: transparent-span inputs differ only in their last bits.
        self._span_cache: list[
            tuple[NDArrayFloat, int, NDArrayFloat, NDArrayFloat, NDArrayFloat]
        ] = []
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
        # The plan-level dense-grid flag gates optional row correction and matches the validity
        # report.
        _, _spacing_ratio = dense_grid_metrics(self.f_abs_hz, self.b_hz)
        self.dense_grid_plan = bool(_spacing_ratio < DENSE_GRID_SPACING_RATIO)
        # Resolve FWM coherence once from channel geometry, independent of power and span count.
        self.fwm_coherence_ratio = fwm_coherence_ratio(self.f_abs_hz, self.b_hz)
        self.fwm_coherence_mode = resolve_fwm_coherence_mode(
            self.dense_grid_fwm_coherence,
            self.fwm_coherence_ratio,
            fwm_fix_applies=self.dense_grid_fwm_fix and self.dense_grid_plan,
        )

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
        self.beta2 = float(
            _calc_beta2_from_d(fiber_config.dispersion_parameter_d, lambda_ref_m)
        )
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
        self._ramp_cache: dict[
            tuple[int, float], tuple[NDArrayFloat, NDArrayFloat]
        ] = {}
        self._factor_cache: dict[
            tuple[int, float, float], tuple[NDArrayComplex, NDArrayComplex]
        ] = {}
        self._visibility_cache: dict[
            tuple[int, float], tuple[NDArrayFloat, NDArrayFloat]
        ] = {}
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
            + np.pi
            * self.beta3
            * (self.f_abs_hz[:, None] + self.f_abs_hz[None, :] - 2.0 * self.f_ref_hz)
            + (2.0 / 3.0) * np.pi**2 * self.beta4 * (df_i**2 + df_i * df_j + df_j**2)
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
        """Validate, sort and merge the lumped loss elements of the span.

        Duplicate positions are merged by summing their dB: two connectors at
        the same physical point multiply the power by the product of their
        transmission factors, which is the sum in dB.  This matches
        :func:`~network_nlo_eval.models.raman_ode._resolve_lumped_loss`, so the
        pumped and unpumped paths agree on the same input.
        """
        if not lumped_loss:
            return []
        checked: list[tuple[float, float]] = []
        for position_m, loss_db in lumped_loss:
            position = float(position_m)
            loss = float(loss_db)
            if not np.isfinite(position) or not np.isfinite(loss):
                msg = f"lumped loss must be finite, got ({position_m!r}, {loss_db!r})."
                raise ValueError(msg)
            if position <= 0.0:
                msg = f"lumped loss position must be positive, got {position}."
                raise ValueError(msg)
            if loss < 0.0:
                msg = f"lumped loss must not be a gain, got {loss} dB."
                raise ValueError(msg)
            checked.append((position, loss))
        checked.sort(key=lambda item: item[0])
        merged: list[tuple[float, float]] = []
        for position, loss in checked:
            if merged and position == merged[-1][0]:
                merged[-1] = (position, merged[-1][1] + loss)
            else:
                merged.append((position, loss))
        return merged

    def _span_segments(
        self, span_length_m: float
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Segment lengths, loss factors and step counts of the span.

        A span without lumped loss is a single segment; every lumped loss
        element splits the span, so that the power profile has a genuine
        discontinuity at its position (paper, Section IV-D).
        """
        edges = [0.0]
        losses = [1.0]
        for position, loss_db in self.lumped_loss:
            if position >= span_length_m:
                msg = f"lumped loss at {position} m lies outside the span of {span_length_m} m."
                raise ValueError(msg)
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
        key = round(span_length_m * 1e6)
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
        key = round(span_length_m * 1e6)
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
        key = round(span_length_m * 1e6)
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

        Parameters
        ----------
        span_length_m : float
            Span length (m).

        Returns
        -------
        ndarray
            ``(N,)`` dispersion contribution of the span (ps^2).
        """
        beta2_ps2_per_m = self.beta2_eff_sci * 1.0e24
        return beta2_ps2_per_m * float(span_length_m)

    def _xci_weight_matrix(self, span_length_m: float) -> NDArrayFloat:
        """Return the XCI weight matrix of a span length.

        Without ``edge_fix`` and ``dense_grid_fix`` this is the printed
        Eq. (24) matrix.  With ``edge_fix``, the CUT rows whose closed-form XCI
        leaves its asymptotic validity domain are replaced by the exact island
        integral; see :func:`_xci_edge_weight_matrix`.  With
        ``dense_grid_fix=True`` (opt-in, default ``False``) the pairs whose
        rectangular spectra overlap are additionally replaced by
        the exact integral on the actual band, including ``y < 0``; see
        :func:`_xci_overlap_weight_matrix`.  Both corrections depend on the
        span length (through ``alpha L`` and the island phase), so the matrix
        is cached per span length.

        Parameters
        ----------
        span_length_m : float
            Span length (m).

        Returns
        -------
        ndarray
            ``(N, N)`` XCI weight matrix, internal channel order.
        """
        if not self.edge_fix and not self.dense_grid_fix:
            return self.w_xci
        key = round(span_length_m * 1e6)
        if not self.dense_grid_fix:
            cached = self._xci_edge_cache.get(key)
            if cached is not None:
                return cached[0]
        else:
            cached_dense = self._xci_dense_cache.get(key)
            if cached_dense is not None:
                return cached_dense[0]

        if self.edge_fix:
            cached = self._xci_edge_cache.get(key)
            if cached is None:
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
            else:
                w_edge = cached[0]
        else:
            w_edge = self.w_xci

        if not self.dense_grid_fix:
            return w_edge

        force_rows = np.full(self.n_nodes, self.dense_grid_plan, dtype=bool)
        w_dense, dense_gated, dense_fraction, dense_pairs = _xci_overlap_weight_matrix(
            w_edge,
            self.gamma_xci,
            self.beta2_eff_xci,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_visibility_npm,
            float(span_length_m),
            n_quad=self.xci_edge_quadrature_points,
            force_rows=force_rows,
        )
        self._xci_dense_cache[key] = (w_dense, dense_gated, dense_fraction, dense_pairs)
        return w_dense

    def xci_edge_audit(self, span_length_m: float) -> tuple[NDArrayBool, NDArrayFloat]:
        """Return the CUT-row gate and the relative correction of the XCI edge fix.

        Parameters
        ----------
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(gated, fraction)``: the boolean mask of the corrected CUT rows
            (internal order) and the relative change of every row before
            gating.  Without ``edge_fix`` both are zero.
        """
        if not self.edge_fix:
            empty = np.zeros(self.n_nodes, dtype=np.float64)
            return np.zeros(self.n_nodes, dtype=bool), empty
        self._xci_weight_matrix(span_length_m)
        key = round(span_length_m * 1e6)
        _, gated, fraction = self._xci_edge_cache[key]
        return gated, fraction

    def xci_dense_overlap_audit(
        self,
        span_length_m: float,
    ) -> tuple[NDArrayBool, NDArrayFloat, NDArrayFloat]:
        """Return the overlap-correction audit of the dense-grid XCI fix.

        Parameters
        ----------
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray]
            ``(gated, fraction, pairs)``: the mask of CUT rows replaced by the
            exact overlapping-band integral, the relative row change before
            gating, and the number of overlapping pairs per row.  All arrays
            are ``(N,)`` in the internal channel order and all-zero without
            ``dense_grid_fix``.
        """
        zeros = np.zeros(self.n_nodes, dtype=np.float64)
        if not self.dense_grid_fix:
            return np.zeros(self.n_nodes, dtype=bool), zeros, zeros
        self._xci_weight_matrix(span_length_m)
        key = round(span_length_m * 1e6)
        _, gated, fraction, pairs = self._xci_dense_cache[key]
        return gated, fraction, pairs

    def _dense_fwm_weight_terms(
        self,
        span_length_m: float,
    ) -> tuple[NDArrayInt, NDArrayInt, NDArrayInt, NDArrayFloat] | None:
        """Weight of the displaced-third-wave triples of a span length.

        Only built when ``dense_grid_fwm_fix=True`` and the plan is dense
        (``min spacing < max bandwidth``).  On any other plan the result is
        ``None`` and the caller leaves the XCI untouched, so the default and
        every sparse comb stay bit-for-bit identical.

        Parameters
        ----------
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray, ndarray] | None
            ``(i, ell, r, weight)`` or ``None`` when no triple can reach a
            different band.  Cached per span length.
        """
        if not self.dense_grid_fwm_fix or not self.dense_grid_plan:
            return None
        key = round(span_length_m * 1e6)
        if key in self._fwm_weight_cache:
            return self._fwm_weight_cache[key]
        terms = _fwm_displaced_weight_terms(
            self.gamma_xci,
            self.beta2_eff_xci,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_visibility_npm,
            float(span_length_m),
            n_quad=self.fwm_quadrature_points,
        )
        result: tuple[NDArrayInt, NDArrayInt, NDArrayInt, NDArrayFloat] | None = (
            None if terms[0].size == 0 else terms
        )
        self._fwm_weight_cache[key] = result
        return result

    def dense_fwm_fix_diagnostics(self, span_length_m: float) -> dict[str, float]:
        """Read-only summary of the opt-in displaced-third-wave term.

        Parameters
        ----------
        span_length_m : float
            Span length (m).

        Returns
        -------
        dict[str, float]
            ``{"triples", "weight_sum", "ell_eq_i_share"}``.  All zeros when
            the term does not apply.
        """
        terms = self._dense_fwm_weight_terms(span_length_m)
        if terms is None:
            return {"triples": 0.0, "weight_sum": 0.0, "ell_eq_i_share": 0.0}
        i_idx, ell_idx, _r_idx, weight = terms
        total = float(np.sum(weight))
        same_band = float(np.sum(weight[ell_idx == i_idx]))
        return {
            "triples": float(weight.size),
            "weight_sum": total,
            "ell_eq_i_share": same_band / total if total > 0.0 else 0.0,
        }

    @property
    def dense_grid_fix_applies(self) -> bool:
        """Whether the plan itself is dense enough for the overlap fix to act.

        It is the plan-level ``dense_grid`` flag ANDed with the opt-in.  The
        per-span diagnostic ``dense_grid_fix_applied`` additionally requires
        that at least one CUT row was actually replaced.
        """
        return bool(self.dense_grid_fix and self.dense_grid_plan)

    def _xci_validity_metrics(
        self,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayBool, NDArrayFloat, NDArrayBool, NDArrayBool]:
        """Return the span-length-dependent part of the validity report.

        The raw ``s`` parameter, the exact row correction and the SCI ``X``
        depend only on the fibre, the channel plan and the span length, so
        they are computed once per span length.  With ``edge_fix=True`` the
        correction is already available from :meth:`xci_edge_audit`; with
        ``edge_fix=False`` it is computed here on demand, because the
        diagnostic must still say how large the omitted correction is.

        Parameters
        ----------
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray, ndarray, ndarray]
            The tuple ``(s_min, s_below, fraction, gated, material)`` in the
            internal channel order.
        """
        key = round(span_length_m * 1e6)
        cached = self._xci_validity_cache.get(key)
        if cached is not None:
            return cached

        delta = np.abs(self.f_abs_hz[:, None] - self.f_abs_hz[None, :])
        half_b = 0.5 * self.b_hz
        y1 = delta - half_b[None, :]
        y2 = delta + half_b[None, :]
        s_pair = _xci_validity_s(
            self.beta2_eff_xci, y2, half_b, self.alpha_visibility_npm
        )
        # Only defined, non-overlapping interfering pairs enter this validity test.
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

    def _xci_overlap_metrics(
        self,
    ) -> tuple[NDArrayFloat, NDArrayBool, NDArrayFloat, NDArrayFloat]:
        """Return the plan-level overlap metrics of the XCI rows (read only).

        The closed-form XCI kernel of Eq. (24) is derived for two *disjoint*
        rectangular spectra. The non-overlap condition is

        .. code-block:: text

            |f_j - f_i| >= (B_i + B_j) / 2 ,

        which is stronger than the ``log_xci`` mask (``|f_j - f_i| >
        B_j / 2``).  A comb with ``spacing < bandwidth`` therefore enters the
        kernel even though the spectra overlap; ``dense_grid`` says so at plan
        level, and this helper quantifies it row by row.

        The result only depends on the channel plan (not on the span length,
        the launch power or the pumps), so it is computed once and cached.

        Returns
        -------
        tuple[ndarray, ndarray, ndarray, ndarray]
            ``(guard_hz, overlap, weight_abs, weight_row_sum)``: the pairwise
            guard ``|f_j - f_i| - (B_i + B_j)/2`` (negative where the
            rectangles overlap), the off-diagonal overlap mask, the absolute
            closed-form XCI weight matrix, and its row sum.  All arrays are in
            the internal (descending-frequency) order.
        """
        cached = self._overlap_cache
        if cached is not None:
            return cached
        delta = np.abs(self.f_abs_hz[:, None] - self.f_abs_hz[None, :])
        half_b = 0.5 * self.b_hz
        guard = delta - (half_b[:, None] + half_b[None, :])
        off_diagonal = ~np.eye(self.n_nodes, dtype=bool)
        overlap = (guard < 0.0) & off_diagonal
        weight_abs = np.abs(self.w_xci)
        weight_row_sum = weight_abs.sum(axis=1)
        self._overlap_cache = (
            np.ascontiguousarray(guard),
            np.ascontiguousarray(overlap),
            np.ascontiguousarray(weight_abs),
            np.ascontiguousarray(weight_row_sum),
        )
        return self._overlap_cache

    def dense_grid_diagnostics(
        self,
        span_length_m: float,
        power_in_w: NDArrayFloat | None = None,
    ) -> DenseGridDiagnostics:
        """Report the error sources of a dense (overlapping) channel plan.

        This is a pure read-only diagnostic.  It never changes the NLI, the
        ASE or the SPP state, so a caller can inspect it before and after an
        :meth:`evaluate` call without affecting the numbers.

        Parameters
        ----------
        span_length_m : float
            Span length (m).
        power_in_w : ndarray | None
            Optional ``(N,)`` launch power per channel in model
            order.  When given, the SPP polynomial fit residual is
            computed (it needs the power-dependent profile).

        Returns
        -------
        DenseGridDiagnostics
            A :class:`DenseGridDiagnostics` in model order.

        Raises
        ------
        ValueError
            ``power_in_w`` does not have one value per channel.
        """
        length_m = float(span_length_m)
        if power_in_w is None:
            power = None
            active = np.ones(self.n_nodes, dtype=bool)
        else:
            power = np.asarray(power_in_w, dtype=np.float64)
            if power.shape != (self.n_nodes,):
                msg = (
                    f"power_in_w must have shape ({self.n_nodes},), got {power.shape}."
                )
                raise ValueError(msg)
            power = np.ascontiguousarray(power[self.order])
            active = power > 0.0

        min_spacing_hz, spacing_ratio = dense_grid_metrics(
            self.f_abs_hz, self.b_hz, active
        )
        dense_grid = bool(spacing_ratio < DENSE_GRID_SPACING_RATIO)

        guard, overlap, weight_abs, weight_row_sum = self._xci_overlap_metrics()
        pair_active = active[:, None] & active[None, :]
        overlap_active = overlap & pair_active
        overlap_counts = overlap_active.sum(axis=1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            overlap_fraction = np.divide(
                (weight_abs * overlap_active).sum(axis=1),
                weight_row_sum,
                out=np.zeros(self.n_nodes, dtype=np.float64),
                where=weight_row_sum > 0.0,
            )
        overlap_fraction = np.where(active, overlap_fraction, 0.0)
        guard_active = guard[pair_active & ~np.eye(self.n_nodes, dtype=bool)]
        min_guard_hz = (
            float(np.min(guard_active)) if guard_active.size else float("inf")
        )

        x_all = self._special_functions(length_m)[0]
        abs_x = np.abs(x_all)
        sci_taylor = sci_small_x_branch_mask(x_all, self._sci_small_x_switch)
        sci_x_abs_min = float(np.min(abs_x[active])) if np.any(active) else 0.0
        n_active = int(np.count_nonzero(active))
        sci_taylor_fraction = (
            float(np.count_nonzero(sci_taylor & active)) / float(n_active)
            if n_active
            else 0.0
        )

        if self.edge_fix:
            # Build the edge cache before reading the gate and row correction.
            self._xci_weight_matrix(length_m)
            gated, correction = self.xci_edge_audit(length_m)
        else:
            _, _, correction = _xci_edge_weight_matrix(
                self.w_xci,
                self.k_xci_matrix,
                self.log_xci,
                self.gamma_xci,
                self.beta2_eff_xci,
                self.f_abs_hz,
                self.b_hz,
                self.alpha_visibility_npm,
                length_m,
                n_quad=self.xci_edge_quadrature_points,
            )
            gated = np.zeros(self.n_nodes, dtype=bool)

        if self.dense_grid_fix:
            # Build the cache before reading overlap diagnostics; the default path skips this
            # branch.
            self._xci_weight_matrix(length_m)
            dense_gated, dense_correction, dense_pairs = self.xci_dense_overlap_audit(
                length_m
            )
            dense_applied = bool(np.any(dense_gated))
        else:
            dense_applied = False
            dense_correction = np.zeros(self.n_nodes, dtype=np.float64)
            dense_pairs = np.zeros(self.n_nodes, dtype=np.float64)
        dense_correction = np.where(active, dense_correction, 0.0)
        dense_pairs = np.where(active, dense_pairs, 0.0)

        fit_residual = None
        fit_max = None
        fit_rms = None
        fit_rel_rms = None
        if power is not None:
            profile_norm, _ = self._integrate_spp(power, length_m)
            q_scaled = self._fit_operator(length_m) @ profile_norm
            design = np.vander(
                self._sample_positions(length_m), self.n_poly, increasing=True
            )
            residual = profile_norm - design @ q_scaled
            fit_residual = np.max(np.abs(residual), axis=0)
            fit_max = float(np.max(np.abs(residual)))
            fit_rms = float(np.sqrt(np.mean(residual**2)))
            profile_rms = float(np.sqrt(np.mean(profile_norm**2)))
            fit_rel_rms = fit_rms / profile_rms if profile_rms > 0.0 else 0.0

        inverse = self.inverse_order
        return DenseGridDiagnostics(
            span_length_m=length_m,
            dense_grid=dense_grid,
            min_channel_spacing_hz=float(min_spacing_hz),
            spacing_to_bandwidth=float(spacing_ratio),
            min_guard_hz=min_guard_hz,
            xci_overlap_pairs=np.ascontiguousarray(overlap_counts[inverse]),
            xci_overlap_weight_fraction=np.ascontiguousarray(overlap_fraction[inverse]),
            xci_edge_gated=np.ascontiguousarray(gated[inverse]),
            xci_row_correction=np.ascontiguousarray(correction[inverse]),
            xci_edge_quadrature_points=int(self.xci_edge_quadrature_points),
            sci_x_abs_min=sci_x_abs_min,
            sci_taylor=np.ascontiguousarray(sci_taylor[inverse]),
            sci_taylor_fraction=sci_taylor_fraction,
            spp_fit_residual=None
            if fit_residual is None
            else np.ascontiguousarray(fit_residual[inverse]),
            spp_fit_residual_max=fit_max,
            spp_fit_residual_rms=fit_rms,
            spp_fit_residual_rel_rms=fit_rel_rms,
            dense_grid_fix_applied=dense_applied,
            dense_grid_overlap_correction=np.ascontiguousarray(
                dense_correction[inverse]
            ),
            dense_grid_overlap_pairs=np.ascontiguousarray(dense_pairs[inverse]),
        )

    def _validity_tilt_db(self, power_tilt_db: float, gain_tilt_db: float) -> float:  # noqa: ARG002 - subclass hook
        """Return the tilt metric that drives the validity flag.

        The base kernel uses the launch-power tilt only: the span gain of an
        unpumped fibre is dominated by ISRS, whose profile the model solves
        explicitly, so a large ISRS tilt is not by itself a validity problem.
        :class:`~network_nlo_eval.models.pcfm_h3_raman.PCFH3RamanSpanKernel`
        overrides this to include the gain tilt when pumps are active, where
        the pump-induced profile is the known weak point.

        Parameters
        ----------
        power_tilt_db : float
            Launch-power tilt ``10 log10(max/min)`` (dB).
        gain_tilt_db : float
            Span-gain tilt ``10 log10(max/min)`` (dB).

        Returns
        -------
        float
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

        Parameters
        ----------
        span_length_m : float
            Span length (m).
        power_in_w : ndarray
            ``(N,)`` launch power per channel (W), model order.
        power_out_w : ndarray | None
            Optional ``(N,)`` span-end signal power in model
            order.  The model passes its already-computed output here so
            the diagnostic does not integrate the SPP a second time.
            When omitted, the kernel integrates it itself.

        Returns
        -------
        SpanValidityReport
            A :class:`SpanValidityReport` in model order.

        Raises
        ------
        ValueError
            ``power_in_w`` or ``power_out_w`` does not have one
            value per channel.
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != (self.n_nodes,):
            msg = f"power_in_w must have shape ({self.n_nodes},), got {power.shape}."
            raise ValueError(msg)
        power_internal = np.ascontiguousarray(power[self.order])
        active = power_internal > 0.0

        if power_out_w is None:
            _, power_out = self._integrate_spp(power_internal, float(span_length_m))
        else:
            out = np.asarray(power_out_w, dtype=np.float64)
            if out.shape != (self.n_nodes,):
                msg = f"power_out_w must have shape ({self.n_nodes},), got {out.shape}."
                raise ValueError(msg)
            power_out = np.ascontiguousarray(out[self.order])

        s_min, s_below, fraction, gated, material = self._xci_validity_metrics(
            span_length_m
        )

        x_all = self._special_functions(span_length_m)[0]
        abs_x = np.abs(x_all)
        sci_taylor = sci_small_x_branch_mask(x_all, self._sci_small_x_switch)
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
            else float(
                np.clip(
                    1.0 - (tilt_db - POWER_TILT_VALIDITY_DB) / POWER_TILT_VALIDITY_DB,
                    0.0,
                    1.0,
                )
            )
        )

        # Per-channel confidence is the weakest of XCI correction, SCI branch, and tilt.
        xci_confidence = np.ones(self.n_nodes, dtype=np.float64)
        xci_confidence = np.where(
            s_below & ~material, XCI_IMMATERIAL_CONFIDENCE, xci_confidence
        )
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
        confidence = np.minimum(
            np.minimum(xci_confidence, sci_confidence), tilt_confidence
        )

        # Only missing required correction invalidates a channel; exact integration keeps corrected
        # channels usable.
        valid = ~((material & ~gated) | (sci_below & ~sci_taylor) | tilt_exceeded)
        flagged = material | sci_below | tilt_exceeded

        # Dense-grid overlap is a validity warning and does not alter point values or confidence.
        min_spacing_hz, spacing_ratio = dense_grid_metrics(
            self.f_abs_hz, self.b_hz, active
        )
        dense_grid = bool(spacing_ratio < DENSE_GRID_SPACING_RATIO)
        flagged = flagged | (dense_grid & active)

        # Report how much XCI row weight lies outside the non-overlap assumption without modifying
        # validity.
        guard, overlap, weight_abs, weight_row_sum = self._xci_overlap_metrics()
        pair_active = active[:, None] & active[None, :]
        overlap_active = overlap & pair_active
        overlap_counts = overlap_active.sum(axis=1).astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            overlap_fraction = np.divide(
                (weight_abs * overlap_active).sum(axis=1),
                weight_row_sum,
                out=np.zeros(self.n_nodes, dtype=np.float64),
                where=weight_row_sum > 0.0,
            )
        overlap_fraction = np.where(active, overlap_fraction, 0.0)
        guard_active = guard[pair_active & ~np.eye(self.n_nodes, dtype=bool)]
        min_guard_hz = (
            float(np.min(guard_active)) if guard_active.size else float("inf")
        )
        n_active = int(np.count_nonzero(active))
        sci_taylor_fraction = (
            float(np.count_nonzero(sci_taylor & active)) / float(n_active)
            if n_active
            else 0.0
        )

        # Cache overlap corrections only when dense_grid_fix is enabled.
        if self.dense_grid_fix:
            dense_gated, dense_fraction, dense_pairs = self.xci_dense_overlap_audit(
                span_length_m
            )
            dense_applied = bool(np.any(dense_gated))
        else:
            dense_gated = np.zeros(self.n_nodes, dtype=bool)
            dense_fraction = np.zeros(self.n_nodes, dtype=np.float64)
            dense_pairs = np.zeros(self.n_nodes, dtype=np.float64)
            dense_applied = False
        dense_pairs = np.where(active, dense_pairs, 0.0)
        dense_fraction = np.where(active, dense_fraction, 0.0)

        # Flag low-dispersion FWM risk geometrically; only reject mode clears validity, and neither
        # mode alters point values.
        dispersion_audit = dispersion_validity(
            self.f_abs_hz,
            self.b_hz,
            active,
            dispersion_parameter_d=self.dispersion_parameter_d,
            dispersion_slope_s=self.dispersion_slope_s,
            reference_wavelength_nm=self.reference_wavelength_nm,
        )
        if dispersion_audit.fwm_required:
            flagged = flagged | active
            if self.fwm_validity_mode == "reject":
                valid = np.where(active, False, valid)

        # Inactive channels have no physical validity value and should not become NaN in strict
        # mode.
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
            xci_overlap_pairs=np.ascontiguousarray(overlap_counts[self.inverse_order]),
            xci_overlap_weight_fraction=np.ascontiguousarray(
                overlap_fraction[self.inverse_order]
            ),
            min_guard_hz=min_guard_hz,
            sci_taylor_fraction=sci_taylor_fraction,
            dense_grid_fix_applied=dense_applied,
            dense_grid_overlap_correction=np.ascontiguousarray(
                dense_fraction[self.inverse_order]
            ),
            dense_grid_overlap_pairs=np.ascontiguousarray(
                dense_pairs[self.inverse_order]
            ),
            raman_pump_overlap_channels=np.zeros(self.n_nodes, dtype=bool),
            min_abs_dispersion_ps_nm_km=float(
                dispersion_audit.min_abs_dispersion_ps_nm_km
            ),
            zero_dispersion_in_band=bool(dispersion_audit.zero_dispersion_in_band),
            dispersion_reference_in_band=bool(
                dispersion_audit.dispersion_reference_in_band
            ),
            fwm_required=bool(dispersion_audit.fwm_required),
            fwm_reason=str(dispersion_audit.fwm_reason),
            fwm_validity_mode=str(self.fwm_validity_mode),
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

        Parameters
        ----------
        power_internal : ndarray
            ``(N,)`` launch power in the internal channel order.
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray, ndarray] | None
            ``(power_out, k_sci, profile_energy)`` or ``None`` on a miss.
        """
        if not self.cache_spans:
            return None
        key = round(span_length_m * 1e6)
        for cached_power, cached_key, power_out, k_sci, energy in self._span_cache:
            if cached_key != key:
                continue
            if np.array_equal(cached_power, power_internal):
                self.span_cache_hits += 1
                return power_out, k_sci, energy
            # Transparent-span inputs differ only by gain-division roundoff within this tolerance.
            if np.allclose(
                cached_power, power_internal, rtol=SPAN_CACHE_RTOL, atol=0.0
            ):
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
        key = round(span_length_m * 1e6)
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
    def spp_samples(
        self, span_length_m: float, power_in_w: NDArrayFloat
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return the normalised SPP samples and their positions.

        Parameters
        ----------
        span_length_m : float
            Span length (m).
        power_in_w : ndarray
            ``(N,)`` launch power per channel (W), model order.

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(positions, profile)``: the normalised positions ``z/L`` and the
            normalised SPP ``p_i(z) = P_i(z)/P_i(0)`` sampled twice per
            integration step.
        """
        power = np.ascontiguousarray(
            np.asarray(power_in_w, dtype=np.float64)[self.order]
        )
        profile_norm, _ = self._integrate_spp(power, span_length_m)
        return self._sample_positions(span_length_m), profile_norm

    def profile_polynomial(
        self, span_length_m: float, power_in_w: NDArrayFloat
    ) -> NDArrayFloat:
        """Polynomial representation of the normalised SPP, Eq. (23).

        Parameters
        ----------
        span_length_m : float
            Span length (m).
        power_in_w : ndarray
            ``(N,)`` launch power per channel (W), model order.

        Returns
        -------
        ndarray
            ``(N_p+1, N)`` coefficients in the scaled basis
            ``p_i(z) = sum_n q_n (z/L)^n``.  The coefficients of the z-basis
            of the paper follow as ``p_n = q_n / L^n``.
        """
        power = np.ascontiguousarray(
            np.asarray(power_in_w, dtype=np.float64)[self.order]
        )
        profile_norm, _ = self._integrate_spp(power, span_length_m)
        pinv = self._fit_operator(span_length_m)
        return pinv @ profile_norm

    def sci_kernel(self, span_length_m: float, q_scaled: NDArrayFloat) -> NDArrayFloat:
        """Closed-form SCI kernel ``K_SCI`` of Eq. (30).

        Parameters
        ----------
        span_length_m : float
            Span length (m).
        q_scaled : ndarray
            ``(N_p+1, N)`` scaled SPP coefficients.

        Returns
        -------
        ndarray
            ``(N,)`` kernel values.
        """
        x_all, si_all, sin_all, cos_all, j_all = self._special_functions(span_length_m)
        q_outer = (q_scaled.T[:, :, None] * q_scaled.T[:, None, :]).reshape(
            self.n_nodes, -1
        )
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

        Parameters
        ----------
        span_length_m : float
            Span length (m).
        q_scaled : ndarray
            ``(N_p+1, N)`` scaled SPP coefficients.

        Returns
        -------
        ndarray
            ``(N, N)`` kernel values; row ``i`` is the CUT, column ``j`` the
            interfering channel.
        """
        energy = _profile_energy_jit(np.ascontiguousarray(q_scaled)) * float(
            span_length_m
        )
        return cast("NDArrayFloat", self.k_xci_matrix * energy[None, :])

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
        float
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

        Parameters
        ----------
        power : ndarray
            ``(N,)`` launch power per channel (W), internal order, in the
            kernel's ``nli_power_convention`` coordinate.
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(profile_norm, power_out)``: the normalised SPP
            ``p_i(z) = P_i(z)/P_i(0)`` sampled twice per integration step and
            the power at the span end, both in the input coordinate.
        """
        drive_scale = self._spp_drive_scale()
        # Physics is driven by dual-polarization total power; total mode reuses the input unchanged.
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
            # Convert output power to the input convention; division by two is exact in binary
            # floating point.
            power_out = power_out / drive_scale
        return profile_norm, power_out

    # -- inter-span coherence ---------------------------------------------
    def _visibilities(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute the inter-span visibility of the SPM and XPM parts.

        Parameters
        ----------
        n_spans : int
            Number of spans of the link.
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray]
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

        Parameters
        ----------
        n_spans : int
            Number of spans of the link.
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray]
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

        Parameters
        ----------
        span_context : SpanContext
            Position of the span in the link.
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray]
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

        Parameters
        ----------
        n_spans : int
            Number of spans of the link.
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray]
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

    # -- numerical NLI fallback ------------------------------------------
    def _dense_fallback_trigger(self) -> bool:
        """Whether the plan itself is dense (the disjoint-rectangle XCI limit).

        This is the plan-level predicate the validity report exposes as
        ``dense_grid``: ``min spacing < max bandwidth``.  It does not depend on
        the occupied mask, so it is a frozen property of the channel plan.
        """
        return bool(self.dense_grid_plan)

    def _fwm_required_fallback_trigger(self, active: NDArrayBool) -> bool:
        """Whether :func:`dispersion_validity` asks for the FWM term.

        The audit covers the low-dispersion, zero-dispersion and out-of-band-
        anchor regimes, where the PCFM kernels are outside their physics.  It
        only depends on the occupied mask, so the answer is cached per mask.

        Parameters
        ----------
        active : ndarray
            ``(N,)`` boolean mask of launched channels, internal order.

        Returns
        -------
        bool
            ``True`` when the low-dispersion / FWM audit flags the span.
        """
        key = np.ascontiguousarray(active, dtype=bool).tobytes()
        cached = self._fwm_required_cache.get(key)
        if cached is None:
            cached = bool(
                dispersion_validity(
                    self.f_abs_hz,
                    self.b_hz,
                    active,
                    dispersion_parameter_d=self.dispersion_parameter_d,
                    dispersion_slope_s=self.dispersion_slope_s,
                    reference_wavelength_nm=self.reference_wavelength_nm,
                ).fwm_required
            )
            self._fwm_required_cache[key] = cached
        return cached

    def _nli_fallback_trigger(self, active: NDArrayBool) -> bool:
        """Whether this span plan is flagged by the applicability audit.

        The trigger set is selected by ``nli_fallback_trigger``:

        * ``"audit"`` (default) — dense plan *or* ``fwm_required``, the
          historical behaviour.  The dense test short-circuits first, so a
          dense plan never computes (nor caches) the dispersion audit.
        * ``"fwm_only"`` — only ``dispersion_validity(...).fwm_required``.
          A dense plan is not flagged, so the fallback and
          ``dense_grid_fwm_fix`` can coexist on disjoint span sets.

        Parameters
        ----------
        active : ndarray
            ``(N,)`` boolean mask of launched channels, internal order.

        Returns
        -------
        bool
            ``True`` when the fallback has to replace the closed-form NLI.
        """
        if self.nli_fallback_trigger == "fwm_only":
            return self._fwm_required_fallback_trigger(active)
        return self._dense_fallback_trigger() or self._fwm_required_fallback_trigger(
            active
        )

    def _numerical_kernel(self) -> NumericalISRSGNKernel:
        """Return the lazily built numerical ISRS-GN kernel of this span.

        The kernel is built on the *internal* (descending-frequency) channel
        order, so its outputs come back in that same order and need no
        permutation.  It is only constructed when the fallback actually fires:
        the default ``nli_fallback=None`` path never imports
        :mod:`network_nlo_eval.models.numerical`.
        """
        kernel = self._fallback_kernel
        if kernel is None:
            from network_nlo_eval.models.numerical import NumericalISRSGNKernel  # noqa: I001, PLC0415

            kernel = NumericalISRSGNKernel(self.f_abs_hz, self.b_hz, self._fiber_config)
            self._fallback_kernel = kernel
        return kernel

    def _fallback_nli(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Numerical ``(spm, xpm)`` noise powers of one span, internal order.

        The result is cached per ``(span length, power)``: a transparent link
        repeats the same span input, so the double integral is paid once per
        distinct span state.  The returned arrays are the model's physical
        total-power values; the caller applies the coordinate conversion.

        Parameters
        ----------
        power : ndarray
            ``(N,)`` launch power per channel (W), internal order.
        span_length_m : float
            Span length (m).

        Returns
        -------
        tuple[ndarray, ndarray]
            ``(spm, xpm)`` in W, internal order, total-power coordinate.
        """
        key = (round(float(span_length_m) * 1e6), np.ascontiguousarray(power).tobytes())
        cached = self._fallback_cache.get(key)
        if cached is not None:
            return cached
        kernel = self._numerical_kernel()
        started = time.perf_counter()
        output = kernel.evaluate(power, float(span_length_m))
        elapsed = time.perf_counter() - started
        result = (
            np.ascontiguousarray(output.spm_noise_w),
            np.ascontiguousarray(output.xpm_noise_w),
        )
        self._fallback_cache[key] = result
        self.fallback_evaluations += 1
        self.fallback_seconds += elapsed
        return result

    def _maybe_nli_fallback(
        self,
        power: NDArrayFloat,
        span_length_m: float,
        active: NDArrayBool,
    ) -> tuple[NDArrayFloat, NDArrayFloat] | None:
        """Closed-form NLI replaced by the numerical one on flagged spans.

        Parameters
        ----------
        power : ndarray
            ``(N,)`` launch power per channel (W), internal order.
        span_length_m : float
            Span length (m).
        active : ndarray
            ``(N,)`` boolean mask of launched channels.

        Returns
        -------
        tuple[ndarray, ndarray] | None
            ``None`` when the fallback is off or the plan is not flagged (the
            closed-form values are kept untouched), else the ``(spm, xpm)``
            replacement in the NLI power coordinate of this kernel
            (``"per_pol"`` halves it, exactly like the closed form).
        """
        self.last_nli_fallback_applied = False
        self.last_fallback_nli_w = None
        if self.nli_fallback is None:
            return None
        if not self._nli_fallback_trigger(active):
            return None

        # Drive the numerical kernel with physical total power and convert output back to
        # per-polarization.
        drive = 2.0 * power if self.nli_power_convention == "per_pol" else power
        spm, xpm = self._fallback_nli(drive, span_length_m)
        if self.nli_power_convention == "per_pol":
            spm = 0.5 * spm
            xpm = 0.5 * xpm
        spm = np.where(active, spm, 0.0)
        xpm = np.where(active, xpm, 0.0)
        self.last_nli_fallback_applied = True
        # Diagnostics use model order and total power regardless of the NLI convention.
        total = spm + xpm
        if self.nli_power_convention == "per_pol":
            total = 2.0 * total
        self.last_fallback_nli_w = np.ascontiguousarray(total[self.inverse_order])
        return spm, xpm

    # -- kernel protocol -------------------------------------------------
    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,  # noqa: ARG002 - kernel protocol
        *,
        span_index: int = 0,  # noqa: ARG002 - kernel protocol
        beta2_accum_ps2: NDArrayFloat | None = None,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Evaluate one span of the polynomial closed-form model.

        Parameters
        ----------
        power_in_w : ndarray
            ``(N,)`` launch power per channel (W), in model order.
        span_length_m : float
            Span length (m).
        n_effective_spans : int
            Number of identical spans (kept for protocol
            compatibility; the PCFM accumulates the NLI span by span
            through the route layer).
        span_index : int
            Index of the span inside the link, counted from zero.
            Used by the EGN correction.
        beta2_accum_ps2 : ndarray | None
            ``(N,)`` accumulated effective dispersion at the
            input of the span (ps^2).  Used by the EGN correction.
        span_context : SpanContext | None
            Position of the span in the link.  When it is given
            and the link has more than one span, the output carries the
            complex NLI amplitudes whose relative phase the route layer
            needs for the coherent accumulation.

        Returns
        -------
        SpanFiberOutput
            Output powers and the self- (SCI) and cross- (XCI) interference
            noise powers, all in model order.
        """
        power = np.ascontiguousarray(
            np.asarray(power_in_w, dtype=np.float64)[self.order]
        )

        # Cache the SPP, SCI and profile energy for equivalent transparent spans.
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
            q_int = _profile_energy_jit(np.ascontiguousarray(q_scaled)) * float(
                span_length_m
            )
            self._span_cache_store(power, span_length_m, power_out, k_sci, q_int)
        else:
            power_out, k_sci, q_int = cached

        # --- NLI PSDs, Eqs. (21) and (26) ----------------------------------
        active = power > 0.0
        if self.nli_power_convention == "per_pol":
            # The 16/27 dual-polarization weight assumes total-power PSD; double per-polarization
            # input before evaluation.
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
        # Return compensated NLI: Gamma * p_CUT(L) = 1 under exact per-channel
        # loss compensation. The shared route applies the actual output transfer.
        g_sci = (16.0 / 27.0) * self.gamma_sci**2 * psd**3 * k_sci * rho_cut
        # XCI: G_XCI = (32/27) p_CUT(L) G_CUT sum_nch gamma^2 G_nch^2 K_XCI.
        xci_weight = q_int * psd**2 * rho_nch
        xci_acc = self._xci_weight_matrix(span_length_m) @ xci_weight
        g_xci = (32.0 / 27.0) * psd * xci_acc

        spm = np.where(active, g_sci * self.b_hz, 0.0)
        xpm = np.where(active, g_xci * self.b_hz, 0.0)
        # Optional dense-grid displaced-wave FWM/MCI adds triplets with r != ell, omitted by
        # closed-form XCI.
        self.last_dense_fwm_fix_applied = False
        fwm_split: NDArrayFloat | None = None
        if self.dense_grid_fwm_fix:
            fwm_terms = self._dense_fwm_weight_terms(span_length_m)
            if fwm_terms is not None:
                i_idx, ell_idx, r_idx, weight = fwm_terms
                values = (
                    psd[ell_idx]
                    * psd[r_idx]
                    * q_int[ell_idx]
                    * rho_nch[ell_idx]
                    * rho_nch[r_idx]
                    * weight
                )
                accumulated = np.bincount(i_idx, weights=values, minlength=self.n_nodes)
                fwm_noise = np.where(
                    active, (32.0 / 27.0) * psd * self.b_hz * accumulated, 0.0
                )
                if self.fwm_coherence_mode == "incoherent":
                    # The separated term is excluded from XPM and accumulated as power by the route.
                    fwm_split = fwm_noise
                else:
                    xpm = xpm + fwm_noise
                self.last_dense_fwm_fix_applied = True
        if self.nli_power_convention == "per_pol":
            # Return NLI to the per-polarization convention used by signal and ASE.
            spm = 0.5 * spm
            xpm = 0.5 * xpm
            if fwm_split is not None:
                fwm_split = 0.5 * fwm_split
        # The optional numerical fallback replaces flagged span NLI; when disabled it updates
        # diagnostics only.
        fallback = self._maybe_nli_fallback(power, span_length_m, active)
        if fallback is not None:
            spm, xpm = fallback

        inverse = self.inverse_order
        fwm_out = (
            None if fwm_split is None else np.ascontiguousarray(fwm_split[inverse])
        )
        n_spans = 1 if span_context is None else max(1, int(span_context.total_spans))
        if span_context is None or n_spans <= 1:
            # Single span (or no context at all): the route layer keeps the
            # pure power convention.  This branch is bit-for-bit identical to
            # the implementation without inter-span coherence.
            return SpanFiberOutput(
                power_out_w=np.ascontiguousarray(power_out[inverse]),
                spm_noise_w=np.ascontiguousarray(spm[inverse]),
                xpm_noise_w=np.ascontiguousarray(xpm[inverse]),
                fwm_noise_w=fwm_out,
            )
        # Multi-span link: hand the route layer the complex NLI amplitudes.
        # Their modulus is the square root of the noise power of this span
        # and their phase is the inter-span phase, so that the route layer
        # can sum |sum_k a_k|**2.
        factor_spm, factor_xpm = self._span_phase_factors(
            span_context, float(span_length_m)
        )
        amplitude_spm = np.sqrt(spm) * factor_spm
        amplitude_xpm = np.sqrt(xpm) * factor_xpm
        return SpanFiberOutput(
            power_out_w=np.ascontiguousarray(power_out[inverse]),
            spm_noise_w=np.ascontiguousarray(spm[inverse]),
            xpm_noise_w=np.ascontiguousarray(xpm[inverse]),
            nli_amplitude_spm=np.ascontiguousarray(amplitude_spm[inverse]),
            nli_amplitude_xpm=np.ascontiguousarray(amplitude_xpm[inverse]),
            fwm_noise_w=fwm_out,
        )


class _PCFMNetworkView:
    """Fiber-specific PCFM adapter for the network QoT evaluator.

    The adapter delegates physics to its parent without copying or rebuilding the
    kernel. Independent single-span evaluations reset the parent's accumulated
    span and dispersion state; coherent assembly instead advances that state across
    the link. ROADM configurations are rejected because the parent span step
    handles fiber and EDFA only and the route must apply ROADM loss and booster ASE.

    Parameters
    ----------
    parent : PCFMModel
        Parent providing the physical kernel, channel arrays, and block checks.
    fiber_config : FiberSpanConfig
        Fiber configuration bound to this view.
    """

    def __init__(self, parent: PCFMModel, fiber_config: FiberSpanConfig) -> None:
        """Bind a parent and fiber without constructing a kernel or changing state."""
        self._parent = parent
        self.fiber_config = fiber_config

    @property
    def parent(self) -> PCFMModel:
        """Return the parent model."""
        return self._parent

    @property
    def n_nodes(self) -> int:
        """Return the parent node count."""
        return self._parent.n_nodes

    @property
    def f_abs_hz(self) -> NDArrayFloat:
        """Return parent node center frequencies (Hz)."""
        return self._parent.f_abs_hz

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """Return parent per-node bandwidths (Hz)."""
        return self._parent.channel_bandwidth_hz_channels

    @property
    def ase_polarization_scale(self) -> float:
        """Return the parent scaling for newly generated ASE."""
        return float(getattr(self._parent, "ase_polarization_scale", 1.0))

    @property
    def launch_power_convention(self) -> str:
        """Return the parent power convention, defaulting to total."""
        return str(getattr(self._parent, "launch_power_convention", "total"))

    @property
    def supports_span_context(self) -> bool:
        """Return whether the parent supplies coherent span amplitudes.

        Pumped H3 may use incoherent accumulation; unpumped models can supply
        complex amplitudes for coherent network assembly.
        """
        return bool(getattr(self._parent, "supports_span_context", False))

    @property
    def fwm_noise_is_additive(self) -> bool:
        """Return whether the parent declares its FWM term additive."""
        return bool(getattr(self._parent, "fwm_noise_is_additive", False))

    @property
    def connections(self) -> Sequence[object] | None:
        """Return the parent flexible-grid connections, or None."""
        return getattr(self._parent, "connections", None)

    @property
    def slots_per_channel(self) -> int | None:
        """Return the parent slots-per-channel value."""
        return getattr(self._parent, "slots_per_channel", None)

    @property
    def channel_bandwidth_source(self) -> str:
        """Return the parent per-channel bandwidth source."""
        return str(getattr(self._parent, "channel_bandwidth_source", "symbol_rate"))

    def power_width_scale(self) -> NDArrayFloat:
        """Return the parent fixed-PSD power-width scaling."""
        return self._parent.power_width_scale()

    def roadm_blocks_node(self, roadm_config: ROADMConfig, node_index: int) -> bool:
        """Return the parent ROADM block decision for one node."""
        return self._parent.roadm_blocks_node(roadm_config, node_index)

    def _reset_span_state(self) -> None:
        """Reset span index, accumulated dispersion, and the previous validity report.

        Independent network span evaluations start from zero so EGN and phase state
        from one link cannot contaminate another.
        """
        self._parent._span_index = 0
        self._parent._beta2_accum_ps2 = np.zeros(self._parent.n_nodes, dtype=np.float64)
        self._parent._reset_validity()

    def _calc_span_noise_and_power(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Evaluate one independent span using the parent power convention.

        ROADM configurations and effective span counts other than one are rejected;
        link assembly applies repeated spans and ROADMs. An additive FWM term is
        included in XPM for this power-only network interface.

        Parameters
        ----------
        power_in_w : ndarray
            Channel input powers (W).
        span_length_m : float
            Span length (m).
        edfa_config : EDFAConfig or None
            Span amplifier.
        roadm_config : ROADMConfig or None
            Unsupported; ROADMs must be applied by the route.
        n_effective_spans : int, optional
            Must be one; the caller assembles repeated spans.

        Returns
        -------
        tuple of ndarray
            Output power, SPM, XPM and ASE per channel.

        Raises
        ------
        ValueError
            If a ROADM or an effective span count other than one is supplied.
        """
        if roadm_config is not None:
            msg = (
                "pcfm network view: _calc_span_noise_and_power does not support roadm_config; "
                "PCFMModel.span_step handles the fibre and the EDFA only. Apply the ROADM in the "
                "route layer (evaluate_path) or call span_step directly."
            )
            raise ValueError(msg)
        if int(n_effective_spans) != 1:
            msg = (
                "pcfm network view: only n_effective_spans=1 is supported; the full-span repeat is "
                f"assembled by the caller, got n_effective_spans={n_effective_spans!r}."
            )
            raise ValueError(msg)
        self._reset_span_state()
        self._parent._sync_auto_spp_steps(np.asarray(power_in_w, dtype=np.float64))
        element = self._parent.span_step(
            self.fiber_config,
            edfa_config,
            np.asarray(power_in_w, dtype=np.float64),
            float(span_length_m),
        )
        xpm = element.xpm_noise_w
        if element.fwm_noise_w is not None:
            # Independent network spans accumulate powers; add separated FWM back to XPM to retain
            # the single-span total.
            xpm = xpm + element.fwm_noise_w
        return element.power_out_w, element.spm_noise_w, xpm, element.ase_noise_w

    def _calc_span_output_with_context(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        span_context: SpanContext | None,
    ) -> ElementOutput:
        """Evaluate a span and provide coherent amplitudes when context is supplied.

        Unlike independent evaluations, this resets accumulated state only for the
        first span or when no context is supplied. Additive FWM is left for the caller
        to accumulate as power.

        Parameters
        ----------
        power_in_w : ndarray
            Channel input powers (W).
        span_length_m : float
            Span length (m).
        edfa_config : EDFAConfig or None
            Span amplifier.
        span_context : SpanContext or None
            Span position in the link.

        Returns
        -------
        ElementOutput
            Span output, including coherent amplitudes for multi-span paths.
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if span_context is None or int(span_context.span_index) == 0:
            # Reset state at the first span or when position information is absent.
            self._reset_span_state()
            self._parent._sync_auto_spp_steps(power)
        return self._parent.span_step(
            self.fiber_config,
            edfa_config,
            power,
            float(span_length_m),
            span_context,
        )


class PCFMModel(RouteModel):
    """Polynomial closed-form model (SCI + XCI) with a polynomial SPP fit.

    Network use: :meth:`for_fiber` returns a lightweight view that satisfies the
    ``QoTValidator`` single-span contract (``_calc_span_noise_and_power``) without
    copying the model or rebuilding the kernel.  The path-level coherent
    accumulation stays in :meth:`evaluate_path`; the network view evaluates every
    span as the first span of a link, which is what
    :func:`~network_nlo_eval.rwa.link_noise.assemble_link_noise` assumes.

    Parameters
    ----------
    grid : object
        Spectrum grid (legacy ``SpectrumGrid`` or ``FlexibleGrid``).
    ref_fiber_config : FiberSpanConfig
        Reference fibre configuration.
    symbol_rate_hz : float
        Symbol rate (Hz).
    poly_degree : int
        Degree ``N_p`` of the SPP polynomial, Eq. (23).  The
        paper finds ``N_p = 5`` adequate and ``N_p = 9`` the reference.
    egn : bool
        Apply the machine-learning correction of Zefreh et al. 2020 on
        top of the GN result.
    edge_fix : bool
        Evaluate the exact XCI island integral where the closed form
        leaves its validity domain, and stabilise the SCI kernel at small
        ``X``.  See :class:`PCFMSpanKernel`.
    dense_grid_fix : bool
        Opt-in overlap-aware XCI correction.  ``False``
        (default) keeps every number bit-for-bit unchanged; ``True``
        replaces the XCI weight of the pairs whose interfering band
        rectangular spectra overlap by the exact integral on the folded band.
        Only the pairs with ``|f_i - f_j| <= B_j/2`` are touched, and only
        on a dense plan (``min spacing < max bandwidth``); see
        :class:`PCFMSpanKernel`.
    dense_grid_fwm_fix : bool
        Opt-in displaced-third-wave (FWM/MCI) term
        (default ``False``).  On a dense plan the triples whose generated
        third wave lands outside the second pump band are integrated with
        the same quadrature as the XCI repair and added to the XCI; see
        :class:`PCFMSpanKernel`.  Non-dense plans and the default path stay
        bit-for-bit unchanged.  Mutually exclusive with ``dense_grid_fix``
        and with ``nli_fallback="numerical"`` under
        ``nli_fallback_trigger="audit"``; the combination is allowed with
        ``nli_fallback_trigger="fwm_only"`` because the trigger sets are
        disjoint.
    dense_grid_fwm_coherence : str
        ``"xci"`` (default), ``"incoherent"`` or
        ``"auto"``.  ``"xci"`` keeps the displaced-third-wave term inside
        the XCI complex amplitude, so it inherits the inter-span XCI
        visibility.  On a dense plan with ``dense_grid_fwm_fix=True``,
        ``"incoherent"`` splits the term out of ``xpm`` and lets the route
        layer accumulate it as a power across spans (task BE audit: the FWM
        average is near-incoherent, and the split removes the 10-span
        residual growth of the densest C grids).  ``"auto"`` chooses per
        plan from ``max(B)/min_spacing`` (:func:`resolve_fwm_coherence_mode`)
        and is bit-for-bit ``"xci"`` wherever the split cannot act.  One
        span, non-dense plans and the default value stay bit-for-bit
        unchanged.
    fwm_quadrature_points : int
        Gauss-Legendre nodes per ``y`` interval of the
        opt-in displaced-third-wave (FWM/MCI) triple integral.  Default
        :data:`FWM_QUADRATURE_POINTS` (24) keeps every number bit-for-bit
        identical.  It is deliberately independent of the XCI edge/overlap
        quadrature (``xci_edge_quadrature_points``), which is already
        converged at 12 nodes: raising this knob only refines the FWM term
        and leaves the XCI rows untouched.  Read only with
        ``dense_grid_fwm_fix=True`` on a dense plan.
    fwm_validity_mode : str
        ``"flag"`` (default) or ``"reject"``.  The kernels
        carry SPM/XCI only, so a comb whose local dispersion reaches the
        zero-dispersion point (or whose dispersion anchor is outside the
        band) is outside the model physics.  ``"flag"`` only marks those
        spans in the validity report; ``"reject"`` also clears ``valid``.
        Neither mode changes the point NLI/ASE/SNR.
    nli_fallback : str | None
        ``None`` (default) or ``"numerical"``.  Opt-in reference
        fallback: on the spans the applicability audit flags (dense plan
        or low-dispersion / FWM regime) the closed-form SCI/XCI is
        *replaced* by the ``spm + xpm`` output of the numerical ISRS-GN
        kernel.  The numerical double integral already contains the FWM
        average, so this is the honest alternative to adding a FWM subset
        on top of the PCFM XCI (which double counts it, task AT).  Accuracy
        is the reference's, the price is runtime, and both are paid only on
        the flagged spans: C 100 GHz and C+L 75 GHz never trigger it, and
        ``None`` keeps every number bit-for-bit unchanged without importing
        the numerical module.  See :class:`PCFMSpanKernel`.
    nli_fallback_trigger : str
        ``"audit"`` (default) or ``"fwm_only"``.
        ``"audit"`` keeps the historical trigger set (dense plan or
        ``dispersion_validity(...).fwm_required``).  ``"fwm_only"`` drops
        the dense-plan trigger, so only the low-dispersion / FWM regime
        fires the fallback; the dense plan is then handled by the closed
        form, optionally with ``dense_grid_fwm_fix``.  Because the two
        trigger sets are disjoint, ``dense_grid_fwm_fix=True`` +
        ``nli_fallback="numerical"`` is accepted with ``"fwm_only"`` and
        rejected with ``"audit"`` (which would replace the dense points
        wholesale and double count the FWM term).  No effect when
        ``nli_fallback`` is ``None``.  See :class:`PCFMSpanKernel`.
    cache_spans : bool
        Memoise the identical spans of one path.
    spp_steps : int | str
        RK4 steps of the SPP integration, or ``"auto"`` (12 at
        ``<=0 dBm`` per channel, 32 above it).
    resolution : str | None
        Explicit opt-in resolution profile; ``None`` (default)
        leaves every per-option default untouched.  See
        :data:`RESOLUTION_PROFILES`.
    raman_model : str | None
        Compatibility setting for unpumped links. Positive pump power requires
        ``pcfm_h3_raman`` or ``hybrid_raman_paper``. Default ``None``.
    raman_pumps : Sequence[RamanPumpConfig] | None
        Compatibility pump list; every pump must have zero power.
        Pumped links require a distributed-Raman model.
    raman_gain_model : str | RamanGainModel
        Analytic spectrum name, RamanGainModel instance,
        or JSON/CSV table path; the model is loaded at construction.
    raman_gain_table : tuple[Sequence[float], Sequence[float]] | None
        Normalized ``(offsets_hz, gains)`` table for table
        gain; mutually exclusive with a model instance or file path.
    raman_ase : bool
        Include the distributed Raman ASE before the span EDFA.
    raman_temperature_k : float | None
        Fibre temperature used by the Raman spontaneous
        emission factor.  ``None`` (default) means the module default of
        300 K; passing ``300.0`` explicitly counts as a calibration
        override and warns when the gain table comes from a file.
    raman_peak_gain_coefficient_m_per_w : float | None
        Peak local Raman gain rate per
        watt before division by ``A_eff``.  ``None`` (default) means the
        module default of ``6e-14``; the default matches the
        waveform-level ``physics_nlo_eval.RamanPump`` for cross-checking.
        Explicitly passing ``6e-14`` counts as a calibration override for
        file-backed gain tables and warns.
    nli_power_convention : str
        ``"total"`` (default) means ``power_in_w`` is the
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
    **route_options : dict
        Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.
    """

    info = PCFM_MODEL_INFO

    #: This model provides the complex NLI amplitudes of the inter-span
    #: coherence, so the route layer sums ``|sum_k a_k|**2`` instead of
    #: ``sum_k |a_k|**2``.
    supports_span_context = True

    supports_lumped_loss = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
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
        dense_grid_fix: bool = False,
        dense_grid_fwm_fix: bool = False,
        dense_grid_fwm_coherence: str = "xci",
        fwm_quadrature_points: int = FWM_QUADRATURE_POINTS,
        fwm_validity_mode: str = "flag",
        nli_fallback: str | None = None,
        nli_fallback_trigger: str = "audit",
        **route_options: Any,
    ) -> None:
        """Initialize the model.

        Raises
        ------
        ValueError
            ``poly_degree`` is negative, ``spp_steps`` is invalid,
            the Raman options are inconsistent, ``validity_mode`` is not
            ``"warn"``/``"strict"``, ``nli_power_convention`` is unknown,
            ``resolution`` is unknown or conflicts with an explicit
            ``spp_steps``, or ``nli_fallback``/``nli_fallback_trigger``
            is invalid.
        """
        if poly_degree < 0:
            msg = f"poly_degree must be >= 0, got {poly_degree}."
            raise ValueError(msg)
        if spp_steps != "auto" and int(spp_steps) < 1:
            msg = f"spp_steps must be >= 1 or 'auto', got {spp_steps!r}."
            raise ValueError(msg)
        if raman_model not in (None, "off", "lumped"):
            msg = f"raman_model must be None, 'off' or 'lumped', got {raman_model!r}."
            raise ValueError(msg)
        if validity_mode not in ("warn", "strict"):
            msg = f"validity_mode must be 'warn' or 'strict', got {validity_mode!r}."
            raise ValueError(msg)
        if fwm_validity_mode not in FWM_VALIDITY_MODES:
            msg = f"fwm_validity_mode must be one of {FWM_VALIDITY_MODES}, got {fwm_validity_mode!r}."
            raise ValueError(msg)
        if nli_fallback not in NLI_FALLBACK_MODES:
            msg = f"nli_fallback must be one of {NLI_FALLBACK_MODES}, got {nli_fallback!r}."
            raise ValueError(msg)
        if nli_fallback_trigger not in NLI_FALLBACK_TRIGGERS:
            msg = f"nli_fallback_trigger must be one of {NLI_FALLBACK_TRIGGERS}, got {nli_fallback_trigger!r}."
            raise ValueError(msg)
        if dense_grid_fwm_fix and dense_grid_fix:
            msg = (
                "dense_grid_fwm_fix=True cannot be combined with dense_grid_fix=True: the "
                "overlap repair of the XCI rows already supplies the energy that the displaced "
                "triples would add, so enabling both double counts it (task BB audit)."
            )
            raise ValueError(msg)
        if (
            dense_grid_fwm_fix
            and nli_fallback is not None
            and nli_fallback_trigger != "fwm_only"
        ):
            msg = (
                "dense_grid_fwm_fix=True cannot be combined with nli_fallback='numerical' under "
                "nli_fallback_trigger='audit': the audit trigger replaces spm/xpm wholesale on the "
                "dense plan, where the FWM fix is already added, so the two would double count. "
                "Use nli_fallback_trigger='fwm_only' to keep the fallback on the low-dispersion / "
                "FWM spans only, which leaves the dense plan to the FWM fix."
            )
            raise ValueError(msg)
        if dense_grid_fwm_coherence not in FWM_COHERENCE_MODES:
            msg = (
                "dense_grid_fwm_coherence must be one of "
                f"{FWM_COHERENCE_MODES}, got {dense_grid_fwm_coherence!r}."
            )
            raise ValueError(msg)
        if int(fwm_quadrature_points) < 2:
            msg = f"fwm_quadrature_points must be >= 2, got {fwm_quadrature_points}."
            raise ValueError(msg)
        profile = resolve_resolution_profile(resolution)
        quadrature_points = XCI_EDGE_QUADRATURE_POINTS
        if profile is not None:
            if "spp_steps" in profile:
                # Respect explicitly supplied nondefault resolution rather than overriding it with a
                # profile.
                if spp_steps != DEFAULT_SPP_STEPS:
                    msg = (
                        f"resolution={resolution!r} sets spp_steps={profile['spp_steps']}, but "
                        f"spp_steps={spp_steps!r} was also given explicitly."
                    )
                    raise ValueError(msg)
                spp_steps = profile["spp_steps"]
            quadrature_points = profile.get(
                "xci_quadrature_points", XCI_EDGE_QUADRATURE_POINTS
            )
        nli_power_convention = resolve_nli_power_convention(nli_power_convention)
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.poly_degree = int(poly_degree)
        self.egn = bool(egn)
        self.edge_fix = bool(edge_fix)
        self.dense_grid_fix = bool(dense_grid_fix)
        self.dense_grid_fwm_fix = bool(dense_grid_fwm_fix)
        self.dense_grid_fwm_coherence = str(dense_grid_fwm_coherence)
        self.cache_spans = bool(cache_spans)
        # Use the same plan-level FWM coherence decision as the kernel for its additive declaration.
        self.dense_grid_plan = bool(
            dense_grid_metrics(self.f_abs_hz, self.channel_bandwidth_hz_channels)[1]
            < DENSE_GRID_SPACING_RATIO
        )
        self.fwm_coherence_ratio = fwm_coherence_ratio(
            self.f_abs_hz, self.channel_bandwidth_hz_channels
        )
        self.fwm_coherence_mode = resolve_fwm_coherence_mode(
            self.dense_grid_fwm_coherence,
            self.fwm_coherence_ratio,
            fwm_fix_applies=self.dense_grid_fwm_fix and self.dense_grid_plan,
        )
        self.nli_power_convention = resolve_nli_power_convention(nli_power_convention)
        # Plain PCFM per-pol changes signal and NLI but leaves EDFA ASE in total power; subclasses
        # may declare a full launch convention.
        if self.nli_power_convention == "per_pol" and not hasattr(
            type(self), "launch_power_convention"
        ):
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
        self.fwm_quadrature_points = int(fwm_quadrature_points)
        self.validity_mode = str(validity_mode)
        self.fwm_validity_mode = str(fwm_validity_mode)
        self.nli_fallback = nli_fallback
        self.nli_fallback_trigger = str(nli_fallback_trigger)
        self._active_spp_steps = (
            DEFAULT_SPP_STEPS if spp_steps == "auto" else int(spp_steps)
        )
        self._span_index = 0
        self._beta2_accum_ps2 = np.zeros(self.n_nodes, dtype=np.float64)
        # Reset per-span and worst-link validity reports on path evaluation.
        self.last_span_validity: list[SpanValidityReport] = []
        self.last_validity_valid: NDArrayBool | None = None
        self.last_validity_flagged: NDArrayBool | None = None
        self.last_validity_confidence: NDArrayFloat | None = None
        # Enable lumped Raman only when requested or pumps are explicitly supplied; fiber pumps
        # alone do not change defaults.
        self._raman_pumps = None if raman_pumps is None else list(raman_pumps)
        self._raman_enabled = raman_model == "lumped" or (
            raman_model is None and self._raman_pumps is not None
        )
        _pump_source = (
            self._raman_pumps
            if self._raman_pumps is not None
            else ref_fiber_config.raman_pumps
        )
        self._lumped_raman_active = bool(
            self._raman_enabled
            and any(float(pump.power_w) > 0.0 for pump in (_pump_source or ()))
        )
        if self._lumped_raman_active:
            msg = "Active Raman pumps require pcfm_h3_raman or hybrid_raman_paper."
            raise ValueError(msg)
        self._raman_ase = bool(raman_ase)
        self._raman_gain_model = resolve_raman_gain_model(
            raman_gain_model,
            table=raman_gain_table,
            peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            temperature_k=raman_temperature_k,
        )
        # Cache stateless network adapters by fiber JSON; physics remains in the parent.
        self._fiber_view_cache: dict[str, _PCFMNetworkView] = {}

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Build the single-span kernel of this fibre.

        The fibre-level ``lumped_loss`` (connectors, splices, discrete
        attenuators) is forwarded to the kernel.  The default empty list maps to
        ``None``, so a plain :class:`FiberSpanConfig` keeps every number
        bit-for-bit unchanged.
        """
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
            dense_grid_fix=self.dense_grid_fix,
            dense_grid_fwm_fix=self.dense_grid_fwm_fix,
            dense_grid_fwm_coherence=self.dense_grid_fwm_coherence,
            fwm_quadrature_points=self.fwm_quadrature_points,
            fwm_validity_mode=self.fwm_validity_mode,
            nli_fallback=self.nli_fallback,
            nli_fallback_trigger=self.nli_fallback_trigger,
            lumped_loss=fiber_config.lumped_loss or None,
        )

    def for_fiber(self, fiber_config: FiberSpanConfig) -> _PCFMNetworkView:
        """Return the cached network view bound to a fiber configuration.

        The view shares the parent physical state and uses the fiber JSON as its
        cache key. Independent span calls reset inter-span state before evaluation.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber configuration for the link.

        Returns
        -------
        _PCFMNetworkView
            Cached view for the fiber.
        """
        key = fiber_config.model_dump_json()
        view = self._fiber_view_cache.get(key)
        if view is None:
            view = _PCFMNetworkView(self, fiber_config)
            self._fiber_view_cache[key] = view
        return view

    @property
    def raman_pumps(self) -> list[RamanPumpConfig]:
        """Configured pumps of the model (empty when Raman is off).

        ``PCFMSpanKernel`` carries no pump state, so the H3 error budget has to
        read the pump directions from the model instead.  The H3 Raman subclass
        keeps its pumps on the kernel and therefore leaves this empty.
        """
        return list(self._raman_pumps or [])

    def dense_grid_diagnostics(
        self,
        power_in_w: NDArrayFloat | None = None,
        span_length_m: float | None = None,
    ) -> DenseGridDiagnostics:
        """Return the kernel's dense-grid diagnostic through the model layer.

        Parameters
        ----------
        power_in_w : ndarray | None
            Optional ``(N,)`` launch power per channel in model
            order.  When given, the SPP polynomial fit residual is
            included.
        span_length_m : float | None
            Span length (m); ``None`` uses the reference
            fibre's length.

        Returns
        -------
        DenseGridDiagnostics
            A :class:`DenseGridDiagnostics` in model order.
        """
        length_m = float(
            self.ref_fiber_config.length_km * 1.0e3
            if span_length_m is None
            else span_length_m
        )
        kernel = cast("PCFMSpanKernel", self.kernel_for(self.ref_fiber_config))
        return kernel.dense_grid_diagnostics(length_m, power_in_w)

    @property
    def lumped_raman_active(self) -> bool:
        """Compatibility flag; active pumps require a distributed-Raman model."""
        return bool(self._lumped_raman_active)

    @property
    def fwm_noise_is_additive(self) -> bool:
        """Whether ``span_step`` hands the route layer a *split* FWM power.

        ``True`` when the kernel removes the displaced-third-wave term from
        ``xpm_noise_w`` and returns it as ``fwm_noise_w``, so the route layer
        must accumulate it as a power and add it back to the path XPM exactly
        once.  That is the *resolved* mode (``fwm_coherence_mode``), not the
        requested one: ``dense_grid_fwm_coherence="auto"`` on a plan that stays
        with ``"xci"`` (50 GHz C, non-dense, no ``dense_grid_fwm_fix``) reports
        ``False``, otherwise the route layer would look for a split term that
        the kernel never produced.  The default ``"xci"`` also returns
        ``False`` and keeps the historical single-array convention.
        """
        return bool(self.dense_grid_fwm_fix and self.fwm_coherence_mode == "incoherent")

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
        float
            ``2.0`` for a per-polarization coordinate, ``1.0`` otherwise.
        """
        return 2.0 if self.nli_power_convention == "per_pol" else 1.0

    @property
    def last_nli_fallback_applied(self) -> bool:
        """Whether any span of the last path replaced its NLI numerically.

        Aggregated over the kernels of the path; ``False`` with the default
        ``nli_fallback=None``.
        """
        return any(
            bool(getattr(kernel, "last_nli_fallback_applied", False))
            for kernel in self._kernel_cache.values()
        )

    @property
    def last_dense_fwm_fix_applied(self) -> bool:
        """Whether any span of the last path added the displaced-third-wave term.

        Aggregated over the kernels of the path; always ``False`` with the
        default ``dense_grid_fwm_fix=False`` and on a non-dense plan.
        """
        return any(
            bool(getattr(kernel, "last_dense_fwm_fix_applied", False))
            for kernel in self._kernel_cache.values()
        )

    def dense_fwm_fix_diagnostics(
        self, span_length_m: float | None = None
    ) -> dict[str, float]:
        """Read-only summary of the opt-in FWM/MCI term of the reference fibre.

        Parameters
        ----------
        span_length_m : float | None
            Span length (m); ``None`` uses the reference fibre.

        Returns
        -------
        dict[str, float]
            ``{"triples", "weight_sum", "ell_eq_i_share"}``; all zeros when the
            term does not apply.
        """
        length_m = float(
            self.ref_fiber_config.length_km * 1.0e3
            if span_length_m is None
            else span_length_m
        )
        kernel = cast("PCFMSpanKernel", self.kernel_for(self.ref_fiber_config))
        return kernel.dense_fwm_fix_diagnostics(length_m)

    @property
    def last_fallback_nli_w(self) -> NDArrayFloat | None:
        """``(N,)`` numerical NLI of the last flagged span, model order, total power.

        ``None`` when the fallback did not fire.  The value is the replacement
        itself (``spm + xpm`` in the dual-polarization coordinate), so the
        cost/benefit of the fallback can be read without re-evaluating.
        """
        for kernel in self._kernel_cache.values():
            values = getattr(kernel, "last_fallback_nli_w", None)
            if values is not None:
                return np.asarray(values, dtype=np.float64)
        return None

    @property
    def fallback_evaluations(self) -> int:
        """Number of numerical double integrals actually paid by this model."""
        return sum(
            int(getattr(kernel, "fallback_evaluations", 0))
            for kernel in self._kernel_cache.values()
        )

    @property
    def fallback_seconds(self) -> float:
        """Wall time (s) spent inside the numerical fallback kernel."""
        return sum(
            float(getattr(kernel, "fallback_seconds", 0.0))
            for kernel in self._kernel_cache.values()
        )

    def _reset_nli_fallback(self) -> None:
        """Forget the per-span fallback verdict of the previous path.

        ``fallback_evaluations``/``fallback_seconds`` are cumulative counters
        and are deliberately kept: they measure the cost of the whole session,
        while the verdict and the NLI snapshot describe the last path.
        """
        for cached_kernel in self._kernel_cache.values():
            kernel = cast("PCFMSpanKernel", cached_kernel)
            kernel.last_nli_fallback_applied = False
            kernel.last_fallback_nli_w = None

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

        Parameters
        ----------
        kernel : PCFMSpanKernel
            Kernel that evaluated the span.
        power_in_w : ndarray
            ``(N,)`` span-input power (W), model order.
        power_out_w : ndarray
            ``(N,)`` span-output power before the EDFA (W).
        span_length_m : float
            Span length (m).
        """
        report = kernel.validity_report(
            span_length_m, power_in_w, power_out_w=power_out_w
        )
        if self._lumped_raman_active:
            # Lumped Raman changes span-end power but omits in-span profile and depletion effects,
            # so diagnostics flag it.
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
            self.last_validity_confidence = np.array(
                report.confidence, dtype=np.float64, copy=True
            )
            return
        assert self.last_validity_flagged is not None
        assert self.last_validity_confidence is not None
        self.last_validity_valid &= report.valid
        self.last_validity_flagged |= report.flagged
        np.minimum(
            self.last_validity_confidence,
            report.confidence,
            out=self.last_validity_confidence,
        )

    def _sync_auto_spp_steps(self, power_in_w: NDArrayFloat) -> None:
        """Resolve the ``"auto"`` SPP step count from the peak physical power.

        ``spp_steps="auto"`` is a resolution decision, not a per-span one, so it
        must be taken by every entry point that evaluates a span: the path entry
        :meth:`evaluate_path` and the network view of :meth:`for_fiber`.  Both
        call this helper, so a network result and the equivalent one-span path
        are evaluated with the same step count.  What ``power_in_w`` is in
        physical total power is decided by :attr:`physical_power_scale`, never by
        the NLI convention.

        Changing the step count invalidates the kernels: ``spp_steps`` is baked
        into them at construction.

        Parameters
        ----------
        power_in_w : ndarray
            ``(N,)`` launch power of the span, in the model's
            declared coordinate.
        """
        if self.spp_steps != "auto":
            return
        # Auto-resolution threshold uses physical total dBm; convert per-polarization power before
        # testing.
        peak_power_w = max(float(np.max(power_in_w)), 1e-30) * self.physical_power_scale
        peak_dbm = 10.0 * np.log10(peak_power_w * 1e3)
        wanted = 12 if peak_dbm <= 0.0 else DEFAULT_SPP_STEPS
        if wanted != self._active_spp_steps:
            self._active_spp_steps = wanted
            self._kernel_cache.clear()

    def evaluate_path(
        self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat
    ) -> PathQoTResult:
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
        self._sync_auto_spp_steps(power_in_w)
        self._span_index = 0
        self._beta2_accum_ps2 = np.zeros(self.n_nodes, dtype=np.float64)
        self._reset_validity()
        self._reset_nli_fallback()
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
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """Single span: the kernel, then the shared amplifier/ROADM handling.

        The span index and the accumulated effective dispersion are tracked
        here because the machine-learning factor of Zefreh et al. 2020 depends
        on them; both are reset at the beginning of every path.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber parameters of the span.
        edfa_config : EDFAConfig or None
            Amplifier configuration of the span.
        power_in_w : numpy.ndarray
            ``(N,)`` launch power per channel (W).
        span_length_m : float
            Span length (m).
        span_context : SpanContext | None
            Position of the span in the link.  It is handed to
            the kernel, which uses it for the inter-span phase of the NLI
            amplitudes.  ``None`` means the pure power convention.

        Returns
        -------
        ElementOutput
            Output power and noise for the span.
        """
        kernel = cast("PCFMSpanKernel", self.kernel_for(fiber_config))
        fiber_output = kernel.evaluate(
            power_in_w,
            span_length_m,
            1,
            span_index=(
                self._span_index
                if span_context is None
                else int(span_context.span_index)
            ),
            beta2_accum_ps2=self._beta2_accum_ps2,
            span_context=span_context,
        )
        input_ase_w: NDArrayFloat | None = None
        if self._raman_enabled:
            pumps = (
                self._raman_pumps
                if self._raman_pumps is not None
                else fiber_config.raman_pumps
            )
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
                # Lumped Raman scales span-end power only; in-span NLI is unchanged.
                fiber_output = SpanFiberOutput(
                    power_out_w=fiber_output.power_out_w
                    * amplifier.gain(self.f_abs_hz),
                    spm_noise_w=fiber_output.spm_noise_w,
                    xpm_noise_w=fiber_output.xpm_noise_w,
                    nli_amplitude_spm=fiber_output.nli_amplitude_spm,
                    nli_amplitude_xpm=fiber_output.nli_amplitude_xpm,
                    fwm_noise_w=fiber_output.fwm_noise_w,
                )
                if self._raman_ase:
                    input_ase_w = amplifier.ase_w(
                        self.f_abs_hz, self.channel_bandwidth_hz_channels
                    )
        self._record_span_validity(
            kernel, power_in_w, fiber_output.power_out_w, span_length_m
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
            input_ase_w=input_ase_w,
            fwm_is_additive=self.fwm_noise_is_additive,
        )
        self._span_index += 1
        self._beta2_accum_ps2 = self._beta2_accum_ps2 + kernel.effective_dispersion_ps2(
            span_length_m
        )
        return element


register_model(PCFM_MODEL_INFO, PCFMModel)
