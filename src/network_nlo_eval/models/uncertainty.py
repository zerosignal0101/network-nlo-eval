"""H3 error budget and validity propagation (task D prototype).

The fast models answer "what is the SNR"; they do not answer "how much should I
trust it".  This module adds that second answer as a *separate* object so the
existing :class:`~network_nlo_eval.models.base.PathQoTResult` contract stays
untouched:

* :class:`H3ErrorBudget` carries a signed model bias, the residual and
  waveform/statistical 1-sigma spreads, a per-channel recommended QoT margin and
  the validity report of the same evaluation;
* :func:`estimate_h3_error_budget` derives the budget from an already computed
  path result (it never re-evaluates the physics in the default path);
* :class:`H3UncertaintyConfig` keeps every coefficient explicit and lets a
  caller scale a term for sensitivity studies.

**Why the coefficients are constants and not a fitted model**

The repository has waveform references for a handful of operating points, not
a calibration set over launch power, span count, pump set and comb width.
Fitting a smooth residual surface to those points would hide the sample size.
The tables below therefore quote the published residual of the matching
reference explicitly, mark entries that are *provisional* (the reference itself
is not step/statistics converged), and inflate the spread instead of pretending
to know the bias.

**What is deliberately not done here**

* No field is added to :class:`PathQoTResult`; callers opt in through
  :meth:`PCFH3RamanModel.evaluate_path_with_uncertainty`.
* :mod:`network_nlo_eval.rwa.qot_checker` is not modified.  The design for
  propagating per-link validity into ``rwa.link_noise`` and the two-tier
  "H3 field + numerical fallback" orchestrator is in ``docs/h3-uq-design.md``.
* The numerical model is never imported or called at runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath, PathQoTResult, QoTModel
from network_nlo_eval.models.pcfm import (
    DENSE_GRID_SPACING_RATIO,
    SCI_X_SERIES_SWITCH,
    PCFMSpanKernel,
    _xci_validity_s,
    dense_grid_metrics,
)

__all__ = [
    "DENSE_GRID_PENALTY_SCHEDULE_V4",
    "DENSE_GRID_SPACING_RATIO",
    "H3_RESIDUAL_TABLE",
    "H3_STATISTICAL_TABLE",
    "H3_STRONG_TILT_RESIDUAL",
    "H3ErrorBudget",
    "H3UncertaintyConfig",
    "ResidualEntry",
    "UncertaintyTerm",
    "ValidityReport",
    "classify_bandwidth",
    "classify_pump_mode",
    "dense_grid_metrics",
    "dense_grid_schedule_penalty",
    "estimate_h3_error_budget",
    "select_residual_entry",
    "span_validity",
]

#: 1-sigma multiplier of the recommended margin.  Two sigma is the repository
#: convention for "worst channel of a link" screening; it is not a confidence
#: statement about a distribution we have not measured.
DEFAULT_K_SIGMA: float = 2.0

#: Additive margin at zero confidence.  The default makes a fully flagged
#: channel cost 3 dB, i.e. one order of margin step in RWA screening.
DEFAULT_VALIDITY_PENALTY_DB: float = 3.0

#: Power tilt above which the strong-tilt residual row applies.  Below it the
#: C-band residual rows are used; above it the §9 reference (16.5 dB tilt) is
#: the closest measured point and the budget says so.
DEFAULT_STRONG_TILT_DB: float = 6.0

#: Occupied bandwidth above which the wideband statistical spread is used.
#: 6 THz separates the C-band cases (4.3 THz) from C+L / S+C+L (11.4 THz+).
WIDEBAND_THRESHOLD_HZ: float = 6.0e12

#: Deep-breakdown threshold of the XCI validity parameter.  The value is in
#: the units of the exact island integral's ``s`` (the same as the kernel's
#: ``xci_s_min``).  The materiality gate of the kernel
#: (``XCI_EDGE_MATERIALITY``) is the authoritative flag; ``s`` below two is
#: reported as a second, independent warning because there the asymptotic
#: closed form is qualitatively wrong even if a row happens to stay below the
#: materiality gate.
#:
#: The old diagnostics compared a half-``s`` value against one; doubling both
#: the returned ``s`` and this constant keeps every flag and confidence value
#: bit-for-bit identical while making the printed ``s`` physical.
XCI_DEEP_BREAKDOWN_S: float = 2.0

#: Calibrated dense-grid penalty schedule of the v4 audit (opt-in).
#:
#: Each entry is ``(spacing_to_bandwidth_upper, penalty_db)`` and the upper
#: bounds increase strictly.  The penalty is the 0.1 dB ceiling of the largest
#: ``|H3 - numerical|`` SNR residual measured with ``dense_grid_fix=True`` on
#: the full C-band matrix (64 GBaud, 100 km/span, 1/5/10 spans, 0/6 dBm):
#:
#: * ``0.390625`` (C 25 GHz): 3.571 dB -> 3.6 dB;
#: * ``0.5859375`` (C 37.5 GHz): 1.988 dB -> 2.0 dB;
#: * ``0.78125`` (C 50 GHz): 0.546 dB -> 0.6 dB.
#:
#: The residual above 4 dB (C 25 GHz with the default, uncorrected kernel:
#: 6.572 dB) is deliberately *not* covered by any tier: that operating point is
#: still not trustworthy and must stay flagged rather than be masked by a
#: penalty.  This table is a safety-margin calibration, **not** a model
#: correction, and it is never active unless a caller passes it to
#: :attr:`H3UncertaintyConfig.dense_grid_penalty_schedule`.
#:
#: Per the tier lookup (``first ratio <= upper``), ratios below the first upper
#: bound (grids denser than 25 GHz) also take the 3.6 dB tier.  That reuse is an
#: extrapolation beyond the measured points, not a validation of such grids.
DENSE_GRID_PENALTY_SCHEDULE_V4: tuple[tuple[float, float], ...] = (
    (0.390625, 3.6),
    (0.5859375, 2.0),
    (0.78125, 0.6),
)


@dataclass(frozen=True)
class ResidualEntry:
    """A calibration row of the error budget.

    Attributes
    ----------
        bias_db: Signed expected residual (model minus reference) in dB.
            Positive means the model is optimistic.
        sigma_db: 1-sigma spread of the residual in dB.
        calibrated: Whether the reference behind the row is converged enough to
            call the row a calibration rather than a placeholder.
        source: Where the number comes from.  Kept as text so a reader can
            re-derive it without leaving the repository.
    """

    bias_db: float
    sigma_db: float
    calibrated: bool
    source: str


#: Residual rows keyed by ``(pump_mode, band_class)``.
#:
#: Sources (``docs/dra-open-issues.md`` of the development line, photon-factor
#: corrected values):
#:   * c57 nopump H3 1x: -0.497 dB (§1.1 / §2.3);
#:   * c57 backward 0.2 W H3 1x: -0.542 dB (§1.1);
#:   * c57 forward 0.2 W H3 1x: -2.849 dB (§2.3, still step/statistics limited);
#:   * cl152 backward six pumps H3 1x: -1.041 dB (§2.3, mixed with the physical
#:     pump-pump convention, hence provisional).
#: Rows without a direct measurement are marked ``calibrated=False`` and use a
#: widened spread; they are placeholders that must be replaced when the
#: corresponding reference is regenerated.
H3_RESIDUAL_TABLE: dict[tuple[str, str], ResidualEntry] = {
    ("none", "narrow"): ResidualEntry(
        bias_db=-0.50,
        sigma_db=0.50,
        calibrated=True,
        source="dra-open-issues §1.1 c57 nopump H3 1x: -0.497 dB",
    ),
    ("none", "wide"): ResidualEntry(
        bias_db=-0.50,
        sigma_db=1.00,
        calibrated=False,
        source="no published wideband nopump H3 residual above 4-seed spread; spread widened to 1.0 dB",
    ),
    ("backward", "narrow"): ResidualEntry(
        bias_db=-0.54,
        sigma_db=0.50,
        calibrated=True,
        source="dra-open-issues §1.1 c57 backward H3 1x: -0.542 dB",
    ),
    ("backward", "wide"): ResidualEntry(
        bias_db=-1.04,
        sigma_db=1.50,
        calibrated=False,
        source="dra-open-issues §2.3 cl152 six-pump backward H3 1x: -1.041 dB (physical pp convention unresolved)",
    ),
    ("forward", "narrow"): ResidualEntry(
        bias_db=-2.85,
        sigma_db=1.00,
        calibrated=False,
        source="dra-open-issues §2.3 c57 forward H3 1x: -2.849 dB (step/statistics limited)",
    ),
    ("forward", "wide"): ResidualEntry(
        bias_db=-2.00,
        sigma_db=2.00,
        calibrated=False,
        source="no converged wideband forward reference; placeholder between c57 and cl152 behaviour",
    ),
    ("mixed", "narrow"): ResidualEntry(
        bias_db=-1.50,
        sigma_db=1.00,
        calibrated=False,
        source="no direct mixed-pump H3 residual; placeholder spanning forward and backward rows",
    ),
    ("mixed", "wide"): ResidualEntry(
        bias_db=-1.50,
        sigma_db=2.00,
        calibrated=False,
        source="no direct wideband mixed-pump H3 residual; placeholder",
    ),
}

#: Row used when the measured power tilt exceeds
#: :data:`DEFAULT_STRONG_TILT_DB`.  The values are the H3 f=1 mean and rms of
#: §9.2 (16.5 dB on-off tilt, 152 channels, 0.488 m SSFM).  The same section
#: states that the 0.488 m waveform is not step-converged, so the row is
#: explicitly provisional and the wideband statistical term dominates anyway.
H3_STRONG_TILT_RESIDUAL: ResidualEntry = ResidualEntry(
    bias_db=2.40,
    sigma_db=2.70,
    calibrated=False,
    source="dra-open-issues §9.2 H3 f1 mean +2.399 dB, rms 2.696 dB (reference not step-converged)",
)

#: Waveform/statistical 1-sigma spread keyed by band class.
#:
#: The narrow row is an engineering floor: the c57 reference has two seeds and
#: no published per-seed spread, so 0.5 dB is stated as an assumption rather
#: than derived.  The wide row is the published cl152 2x, 128-symbol, 4-seed
#: nopump result (30.587 +/- 5.082 dB, §2.4).  A caller that only wants model
#: error can set ``include_statistical=False``; the reference uncertainty is
#: then reported separately by the terms list.
H3_STATISTICAL_TABLE: dict[str, ResidualEntry] = {
    "narrow": ResidualEntry(
        bias_db=0.0,
        sigma_db=0.50,
        calibrated=False,
        source="c57 256-symbol two-seed reference; 0.5 dB engineering floor (per-seed spread not published)",
    ),
    "wide": ResidualEntry(
        bias_db=0.0,
        sigma_db=5.082,
        calibrated=True,
        source="dra-open-issues §2.4 cl152 2x 128-symbol 4-seed nopump: 30.587 +/- 5.082 dB",
    ),
}


@dataclass(frozen=True)
class H3UncertaintyConfig:
    """Knobs of the error-budget estimate.

    The scales are sensitivity-study knobs, not calibration parameters: a value
    other than one means "what if the spread were larger", and the ``terms``
    list of the budget records the scaled value.

    Args:
        k_sigma: Margin multiplier of the combined 1-sigma spread.
        include_statistical: Add the waveform/statistical spread to the margin.
        residual_scale: Multiplier of the residual bias and spread.
        statistical_scale: Multiplier of the statistical spread.
        validity_penalty_db: Additive margin at zero confidence, scaled by
            ``1 - confidence``.
        strong_tilt_db: Tilt above which the strong-tilt residual row applies.
        dense_grid_penalty_db: Additive margin added to every active channel
            when the comb is a dense grid (channel spacing below the
            bandwidth).  Defaults to ``0``: the flag is reported but the
            default margin is not moved.  The fifth-round matrix measured H3
            optimism of 1.52 dB at 50 GHz and 2.48 dB at 37.5 GHz against a
            1.41 dB narrow margin, so 1.1 dB covers the observed gap and is the
            recommended opt-in value.
        dense_grid_penalty_schedule: Optional per-spacing dense-grid penalty
            schedule, a non-empty sequence of
            ``(spacing_to_bandwidth_upper, penalty_db)`` pairs sorted by
            strictly increasing upper bound.  ``None`` (default) keeps the
            scalar :attr:`dense_grid_penalty_db` path bit-for-bit unchanged.
            When it is set, every active channel of a dense grid takes the
            penalty of the first tier whose upper bound is ``>=``
            :attr:`ValidityReport.spacing_to_bandwidth`; a ratio above the
            last upper bound reuses the last tier (clamped) and the budget
            notes that the point is outside the calibrated range.  The
            schedule **overrides** the scalar.  A schedule is a calibrated
            safety margin, not a model correction: see
            :data:`DENSE_GRID_PENALTY_SCHEDULE_V4` for the v4 calibration.

    Raises
    ------
        ValueError: A scale or penalty is negative, the schedule is empty, an
            upper bound is non-finite/non-positive or not strictly increasing,
            or a schedule penalty is non-finite or negative.
    """

    k_sigma: float = DEFAULT_K_SIGMA
    include_statistical: bool = True
    residual_scale: float = 1.0
    statistical_scale: float = 1.0
    validity_penalty_db: float = DEFAULT_VALIDITY_PENALTY_DB
    strong_tilt_db: float = DEFAULT_STRONG_TILT_DB
    dense_grid_penalty_db: float = 0.0
    dense_grid_penalty_schedule: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        """Reject negative scales; a negative margin has no safety meaning."""
        if self.k_sigma < 0.0:
            raise ValueError(f"k_sigma must be >= 0, got {self.k_sigma}.")
        if self.residual_scale < 0.0:
            raise ValueError(f"residual_scale must be >= 0, got {self.residual_scale}.")
        if self.statistical_scale < 0.0:
            raise ValueError(f"statistical_scale must be >= 0, got {self.statistical_scale}.")
        if self.validity_penalty_db < 0.0:
            raise ValueError(f"validity_penalty_db must be >= 0, got {self.validity_penalty_db}.")
        if self.strong_tilt_db < 0.0:
            raise ValueError(f"strong_tilt_db must be >= 0, got {self.strong_tilt_db}.")
        if self.dense_grid_penalty_db < 0.0:
            raise ValueError(f"dense_grid_penalty_db must be >= 0, got {self.dense_grid_penalty_db}.")
        schedule = self.dense_grid_penalty_schedule
        if schedule is not None:
            if len(schedule) == 0:
                raise ValueError("dense_grid_penalty_schedule must be None or a non-empty sequence.")
            previous_upper: float | None = None
            for index, item in enumerate(schedule):
                try:
                    upper, penalty = item
                except (TypeError, ValueError):
                    raise ValueError(
                        "each dense_grid_penalty_schedule entry must be a "
                        f"(spacing_to_bandwidth_upper, penalty_db) pair, got {item!r} at index {index}."
                    ) from None
                if not np.isfinite(upper) or upper <= 0.0:
                    raise ValueError(
                        "dense_grid_penalty_schedule upper bounds must be finite and > 0, "
                        f"got {upper!r} at index {index}."
                    )
                if not np.isfinite(penalty) or penalty < 0.0:
                    raise ValueError(
                        "dense_grid_penalty_schedule penalties must be finite and >= 0, "
                        f"got {penalty!r} at index {index}."
                    )
                if previous_upper is not None and not upper > previous_upper:
                    raise ValueError(
                        "dense_grid_penalty_schedule upper bounds must be strictly increasing; "
                        f"{upper!r} at index {index} does not exceed {previous_upper!r}."
                    )
                previous_upper = float(upper)


@dataclass(frozen=True)
class UncertaintyTerm:
    """One auditable term of the budget.

    Attributes
    ----------
        name: Term name.
        kind: ``"bias"`` (signed, enters as ``max(bias, 0)``), ``"spread"``
            (1-sigma, enters through the RSS) or ``"penalty"`` (additive).
        value_db: ``(N,)`` per-channel value in dB.
        source: Provenance of the coefficient.
        calibrated: Whether the term comes from a converged reference.
    """

    name: str
    kind: str
    value_db: NDArrayFloat
    source: str
    calibrated: bool


@dataclass(frozen=True)
class ValidityReport:
    """Per-channel validity of the closed-form kernel on one path.

    All arrays are in model order.  ``active`` marks launched channels;
    ``valid`` additionally requires finite noise/SNR and a converged pump solve.
    A channel can be valid while ``flagged`` is true: the flag says the
    coefficient budget is being stretched, not that the answer is unusable.

    Attributes
    ----------
        valid: Channel can be used for a hard QoT decision.
        active: Channel carries positive launch power and positive received
            power.
        confidence: ``[0, 1]`` score reduced by every flag; feeds the margin.
        flagged: OR of all warning flags below.
        xci_edge_gated: Kernel replaced the closed-form XCI row with the exact
            island integral (relative change above the materiality threshold).
        sci_small_x: SCI kernel used the small-``X`` Taylor branch.
        xci_low_s: XCI validity parameter fell below the deep-breakdown limit.
        pump_not_converged: Backward-pump fixed point stopped at its iteration
            cap.
        raman_not_converged: The coupled Raman solve reported a non-converged
            or non-finite result.  It is a hard failure: the channel is
            invalid and its margin is infinite.  Kept separate from
            ``pump_not_converged`` because a forward-pump solve has no fixed
            point and can still be under-resolved.
        raman_wideband_multipump: The comb reaches S/E spectrum and the pump
            set is forward or multi-pump.  ``raman_steps="auto"`` uses the
            extended 96-step branch there; the flag is informational when the
            solve converged, and invalidating when it did not.  It describes a
            *numerical convergence risk* for pumps outside the signal band.
        raman_pump_signal_overlap: ``(N,)`` boolean, a pump frequency falls
            inside the occupied signal band (channel band edges, i.e. centre
            plus half bandwidth, widened by the model's overlap guard).  This
            is a physical-configuration error, not a convergence risk: the
            channel is invalid and its margin is infinite.  It is independent
            of ``raman_wideband_multipump``.
        raman_pump_min_offset_hz: Smallest distance from an active pump to the
            occupied signal band *edge* over the spans (Hz); ``0`` when a pump
            is inside, ``inf`` without pumps.  Read-only.
        lumped_raman: The path used the pure PCFM lumped Raman approximation
            with active pumps.  The intra-span profile and pump depletion are
            not modelled, so the point value is not a trustworthy QoT even
            though it is returned for backward compatibility.  The channel is
            flagged and loses confidence (a scope warning, not a numerical
            failure), so the margin grows but stays finite.
        strong_tilt: Measured power tilt exceeded the configured threshold.
        min_abs_x_sci: Smallest ``|X|`` over the spans of the path.
        min_s_xci: Smallest XCI validity parameter over the spans and pairs.
        xci_correction_fraction: Largest relative XCI row change before gating.
        tilt_db: Power tilt of the path result (max/min over active channels).
        pump_mode: ``"none"``, ``"backward"``, ``"forward"`` or ``"mixed"``.
        band_class: ``"narrow"`` or ``"wide"``.
        span_count: Number of physical spans inspected.
        tilt_confidence_informational: Worst-span informational tilt score in
            ``[0, 1]``, carried over from
            :attr:`~network_nlo_eval.models.pcfm.SpanValidityReport.tilt_confidence_informational`.
            It is a scalar because the tilt describes the whole occupied
            spectrum, not one channel.  It never contributes to ``valid``,
            ``flagged`` or ``confidence``; a network layer may map it to an
            additive soft margin.  The default 1 means "no information" and
            produces a zero soft margin.
        min_channel_spacing_hz: Closest spacing between active channels (Hz);
            ``inf`` when fewer than two channels are launched.
        spacing_to_bandwidth: ``min_channel_spacing_hz / max bandwidth``.
            Below :data:`~network_nlo_eval.models.pcfm.DENSE_GRID_SPACING_RATIO`
            the closed-form XCI row is outside its non-overlapping derivation
            domain.
        dense_grid_overlap: ``True`` when the occupied spectrum is a dense
            grid.  It ORs into ``flagged`` for every active channel but never
            changes ``valid`` or the point SNR; the opt-in
            :attr:`H3UncertaintyConfig.dense_grid_penalty_db` or
            :attr:`H3UncertaintyConfig.dense_grid_penalty_schedule` adds
            margin.
        raman_auto_under_resolved: ``True`` when at least one span's opt-in
            Raman resolution check found the solved step count under-resolved
            (profile, span gain or Raman ASE moved by more than the tolerance).
            It ORs into ``flagged`` but never changes ``valid``, ``confidence``
            or the margin: it is an error-budget warning, not a failure.
        raman_resolution_check_steps: Finest step count actually compared
            during the check ladder (``0`` when no span was checked).
        raman_resolution_change: Largest relative change observed across the
            checked spans (``1.0`` when nothing was checked).
        raman_resolution_checked: ``True`` when at least one span ran the
            check against a usable finer solve.
        min_abs_dispersion_ps_nm_km: Smallest ``|D(lambda)|`` over the occupied
            channels and the spans of the path (ps/nm/km).  The closed-form
            kernels carry no FWM term, so this is the distance of the comb from
            the phase-matched regime.
        zero_dispersion_in_band: ``True`` when any span has the fibre's
            zero-dispersion wavelength inside the occupied band.
        dispersion_reference_in_band: ``True`` only when **every** span's
            ``reference_wavelength_nm`` lies inside the occupied band.  The
            kernel takes one ``beta2``/``beta3`` pair at that anchor, so a
            single out-of-band span makes the dispersion extrapolated on this
            comb.  It is a separate error source from the missing FWM term.
        fwm_required: ``True`` when any span reports the low-dispersion /
            zero-dispersion / out-of-band-anchor regime.  It ORs into
            ``flagged``; with ``fwm_validity_mode="reject"`` it also clears
            ``valid``.  It never changes the point NLI/ASE/SNR.
        fwm_reason: ``""`` or the ``+``-joined union of the span reasons
            (``zero_dispersion_in_band``, ``low_dispersion``,
            ``dispersion_reference_out_of_band``).
        fwm_validity_mode: The mode the spans were evaluated with.
    """

    valid: NDArrayBool
    active: NDArrayBool
    confidence: NDArrayFloat
    flagged: NDArrayBool
    xci_edge_gated: NDArrayBool
    sci_small_x: NDArrayBool
    xci_low_s: NDArrayBool
    pump_not_converged: NDArrayBool
    raman_not_converged: NDArrayBool
    raman_wideband_multipump: NDArrayBool
    raman_pump_signal_overlap: NDArrayBool
    lumped_raman: NDArrayBool
    strong_tilt: NDArrayBool
    min_abs_x_sci: NDArrayFloat
    min_s_xci: NDArrayFloat
    xci_correction_fraction: NDArrayFloat
    tilt_db: float
    pump_mode: str
    band_class: str
    span_count: int
    tilt_confidence_informational: float = 1.0
    min_channel_spacing_hz: float = float("inf")
    spacing_to_bandwidth: float = float("inf")
    dense_grid_overlap: bool = False
    raman_pump_min_offset_hz: float = float("inf")
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
class H3ErrorBudget:
    """Error budget of one H3 path evaluation.

    The recommended margin is::

        margin = max(bias, 0) + k_sigma * sqrt(sigma_residual**2
                                                + sigma_statistical**2)
                 + validity_penalty_db * (1 - confidence)
                 + dense_grid_penalty(channel) * [dense grid]

    ``dense_grid_penalty`` is the scalar
    :attr:`H3UncertaintyConfig.dense_grid_penalty_db`, or the per-channel value
    of :attr:`H3UncertaintyConfig.dense_grid_penalty_schedule` when that
    schedule is set (it overrides the scalar).  ``max(bias, 0)`` makes an
    optimistic model pay for its bias and keeps a pessimistic model
    conservative instead of letting it buy a smaller margin.  Invalid channels
    get ``margin = inf`` so a caller can never accept them.  The dense-grid
    term is zero by default; the flag alone never moves the margin.

    Attributes
    ----------
        model: Model name.
        bias_db: ``(N,)`` signed residual (model minus reference).
        residual_sigma_db: ``(N,)`` 1-sigma residual spread.
        statistical_sigma_db: ``(N,)`` waveform/statistical 1-sigma spread.
        sigma_db: ``(N,)`` combined 1-sigma spread.
        margin_db: ``(N,)`` recommended additive QoT margin; ``inf`` when the
            channel is invalid, zero when it is not launched.
        k_sigma: Margin multiplier used.
        validity: Per-channel validity report.
        terms: Auditable terms.
        notes: Human-readable provenance and provisional warnings.
    """

    model: str
    bias_db: NDArrayFloat
    residual_sigma_db: NDArrayFloat
    statistical_sigma_db: NDArrayFloat
    sigma_db: NDArrayFloat
    margin_db: NDArrayFloat
    k_sigma: float
    validity: ValidityReport
    terms: tuple[UncertaintyTerm, ...]
    notes: tuple[str, ...] = ()

    def adjusted_snr_db(self, snr_db: NDArrayFloat) -> NDArrayFloat:
        """Return ``snr - margin`` with invalid channels forced to ``-inf``.

        Args:
            snr_db: ``(N,)`` point-value SNR in dB.

        Returns
        -------
            ``(N,)`` SNR that can be compared directly with a requirement.

        Raises
        ------
            ValueError: ``snr_db`` shape differs from the budget.
        """
        values = np.asarray(snr_db, dtype=np.float64)
        if values.shape != self.margin_db.shape:
            raise ValueError(f"snr_db shape {values.shape} does not match the budget {self.margin_db.shape}.")
        adjusted = values - self.margin_db
        return np.where(self.validity.valid, adjusted, -np.inf)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (non-finite values as ``None``)."""
        return {
            "model": self.model,
            "k_sigma": float(self.k_sigma),
            "bias_db": _json_values(self.bias_db),
            "residual_sigma_db": _json_values(self.residual_sigma_db),
            "statistical_sigma_db": _json_values(self.statistical_sigma_db),
            "sigma_db": _json_values(self.sigma_db),
            "margin_db": _json_values(self.margin_db),
            "validity": {
                "valid": [bool(value) for value in self.validity.valid],
                "active": [bool(value) for value in self.validity.active],
                "confidence": _json_values(self.validity.confidence),
                "flagged": [bool(value) for value in self.validity.flagged],
                "xci_edge_gated": [bool(value) for value in self.validity.xci_edge_gated],
                "sci_small_x": [bool(value) for value in self.validity.sci_small_x],
                "xci_low_s": [bool(value) for value in self.validity.xci_low_s],
                "pump_not_converged": [bool(value) for value in self.validity.pump_not_converged],
                "raman_not_converged": [bool(value) for value in self.validity.raman_not_converged],
                "raman_wideband_multipump": [
                    bool(value) for value in self.validity.raman_wideband_multipump
                ],
                "raman_pump_signal_overlap": [
                    bool(value) for value in self.validity.raman_pump_signal_overlap
                ],
                "lumped_raman": [bool(value) for value in self.validity.lumped_raman],
                "strong_tilt": [bool(value) for value in self.validity.strong_tilt],
                "min_abs_x_sci": _json_values(self.validity.min_abs_x_sci),
                "min_s_xci": _json_values(self.validity.min_s_xci),
                "xci_correction_fraction": _json_values(self.validity.xci_correction_fraction),
                "tilt_db": float(self.validity.tilt_db),
                "pump_mode": self.validity.pump_mode,
                "band_class": self.validity.band_class,
                "span_count": int(self.validity.span_count),
                "tilt_confidence_informational": float(self.validity.tilt_confidence_informational),
                "min_channel_spacing_hz": (
                    float(self.validity.min_channel_spacing_hz)
                    if np.isfinite(self.validity.min_channel_spacing_hz)
                    else None
                ),
                "spacing_to_bandwidth": (
                    float(self.validity.spacing_to_bandwidth)
                    if np.isfinite(self.validity.spacing_to_bandwidth)
                    else None
                ),
                "dense_grid_overlap": bool(self.validity.dense_grid_overlap),
                "raman_pump_min_offset_hz": (
                    float(self.validity.raman_pump_min_offset_hz)
                    if np.isfinite(self.validity.raman_pump_min_offset_hz)
                    else None
                ),
                "raman_auto_under_resolved": bool(self.validity.raman_auto_under_resolved),
                "raman_resolution_check_steps": int(self.validity.raman_resolution_check_steps),
                "raman_resolution_change": float(self.validity.raman_resolution_change),
                "raman_resolution_checked": bool(self.validity.raman_resolution_checked),
            },
            "terms": [
                {
                    "name": term.name,
                    "kind": term.kind,
                    "value_db": _json_values(term.value_db),
                    "source": term.source,
                    "calibrated": bool(term.calibrated),
                }
                for term in self.terms
            ],
            "notes": list(self.notes),
        }


def _json_values(values: NDArrayFloat) -> list[float | None]:
    """Convert an array to a JSON-safe list, mapping non-finite entries to ``None``."""
    return [float(value) if np.isfinite(value) else None for value in np.asarray(values, dtype=np.float64)]


def classify_pump_mode(directions: Sequence[str]) -> str:
    """Classify the active pump directions of a link.

    Args:
        directions: Directions of the active pumps (``"forward"``/``"backward"``).

    Returns
    -------
        ``"none"``, ``"backward"``, ``"forward"`` or ``"mixed"``.

    Raises
    ------
        ValueError: A direction is neither forward nor backward.
    """
    normalized = {str(direction).lower() for direction in directions}
    if not normalized:
        return "none"
    unknown = normalized - {"forward", "backward"}
    if unknown:
        raise ValueError(f"unknown pump directions {sorted(unknown)}; expected 'forward' or 'backward'.")
    if normalized == {"forward", "backward"}:
        return "mixed"
    return "forward" if "forward" in normalized else "backward"


def classify_bandwidth(occupied_hz: float) -> str:
    """Classify the comb width.

    Args:
        occupied_hz: Occupied bandwidth of the comb in Hz.

    Returns
    -------
        ``"wide"`` above :data:`WIDEBAND_THRESHOLD_HZ`, else ``"narrow"``.
    """
    return "wide" if float(occupied_hz) > WIDEBAND_THRESHOLD_HZ else "narrow"


def select_residual_entry(
    pump_mode: str,
    band_class: str,
    tilt_class: str,
) -> tuple[ResidualEntry, tuple[str, ...]]:
    """Pick the residual calibration row for an operating point.

    Args:
        pump_mode: Output of :func:`classify_pump_mode`.
        band_class: Output of :func:`classify_bandwidth`.
        tilt_class: ``"strong"`` above the configured tilt threshold, else
            ``"normal"``.

    Returns
    -------
        The row and any provenance notes that the caller should surface.

    Raises
    ------
        KeyError: The operating point has no row.  The table is deliberately
            explicit: silently falling back to another pump mode would hide a
            missing calibration.
    """
    if tilt_class == "strong":
        return H3_STRONG_TILT_RESIDUAL, (
            "strong tilt: using the §9.2 strong-tilt residual row; the reference waveform is not step-converged",
        )
    key = (pump_mode, band_class)
    try:
        return H3_RESIDUAL_TABLE[key], ()
    except KeyError:
        raise KeyError(
            f"no H3 residual row for pump_mode={pump_mode!r}, band_class={band_class!r}; "
            f"available keys: {sorted(H3_RESIDUAL_TABLE)}."
        ) from None


def dense_grid_schedule_penalty(
    schedule: Sequence[tuple[float, float]],
    spacing_to_bandwidth: float,
) -> tuple[float, bool]:
    """Look up the dense-grid penalty of one spacing ratio.

    The first tier whose ``upper`` is at least ``spacing_to_bandwidth`` wins.
    A ratio above the last upper bound reuses the last tier and reports
    ``clamped=True`` so the caller can say that the point is outside the
    calibrated range instead of silently pretending it is covered.  A ratio
    below the first upper bound also takes the first tier; that is an
    extrapolation to a denser grid than any calibrated point.

    Args:
        schedule: Non-empty ``(upper, penalty_db)`` sequence with strictly
            increasing upper bounds, as validated by
            :class:`H3UncertaintyConfig`.
        spacing_to_bandwidth: ``min channel spacing / max bandwidth`` of the
            active comb (dimensionless).

    Returns
    -------
        ``(penalty_db, clamped)``.

    Raises
    ------
        ValueError: The schedule is empty.
    """
    if not schedule:
        raise ValueError("dense_grid_schedule_penalty requires a non-empty schedule.")
    ratio = float(spacing_to_bandwidth)
    for upper, penalty in schedule:
        if ratio <= float(upper):
            return float(penalty), False
    return float(schedule[-1][1]), True


def _span_metrics(
    kernel: PCFMSpanKernel,
    span_length_m: float,
) -> tuple[NDArrayBool, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Compute the closed-form validity metrics of one span state.

    Args:
        kernel: Span kernel of the fibre.
        span_length_m: Span length (m).

    Returns
    -------
        ``(gated, correction_fraction, min_abs_x, min_s, confidence)`` in model
        order.
    """
    gated_internal, fraction_internal = kernel.xci_edge_audit(span_length_m)

    bandwidth = kernel.b_hz
    # SCI argument X = pi**2 beta2_eff B**2 L of the closed form.
    x_internal = np.abs(kernel.beta2_eff_sci) * np.pi**2 * bandwidth**2 * float(span_length_m)

    # XCI validity parameter s = 4 pi**2 |beta2_eff| y (B_CUT/2) / alpha of the
    # stretched-island closed form, i.e. the same value the exact integral uses
    # internally.  The smallest value over the interfering channels decides
    # whether the row is trustworthy.
    delta = np.abs(kernel.f_abs_hz[:, None] - kernel.f_abs_hz[None, :])
    y_upper = delta + 0.5 * bandwidth[None, :]
    s_matrix = _xci_validity_s(
        kernel.beta2_eff_xci,
        y_upper,
        0.5 * bandwidth,
        kernel.alpha_visibility_npm,
    )
    # 只统计闭式核真正处理的非重叠信道对；重叠对的 log 项已被核置零。
    eligible = (kernel.log_xci != 0.0) & ((delta - 0.5 * bandwidth[None, :]) > 0.0)
    s_masked = np.where(eligible, s_matrix, np.inf)
    min_s_internal = np.min(s_masked, axis=1) if kernel.n_nodes > 1 else np.full(kernel.n_nodes, np.inf)

    inverse = kernel.inverse_order
    gated = np.ascontiguousarray(gated_internal[inverse])
    fraction = np.ascontiguousarray(fraction_internal[inverse])
    min_abs_x = np.ascontiguousarray(x_internal[inverse])
    min_s = np.ascontiguousarray(min_s_internal[inverse])

    # 置信度只做"证据强度"的粗粒度扣分：每一项都对应核里一个已知的近似分支。
    confidence = np.ones(kernel.n_nodes, dtype=np.float64)
    confidence -= 0.25 * gated
    confidence -= 0.25 * (min_abs_x < SCI_X_SERIES_SWITCH)
    confidence -= 0.25 * (min_s < XCI_DEEP_BREAKDOWN_S)
    confidence = np.clip(confidence, 0.05, 1.0)
    return gated, fraction, min_abs_x, min_s, confidence


def span_validity(
    kernel: PCFMSpanKernel,
    span_length_m: float,
) -> tuple[NDArrayBool, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Return the per-channel validity metrics of one span.

    This is the small public hook that a future ``validity_report`` on the
    kernel (task B) can share instead of duplicating the thresholds.

    Args:
        kernel: Span kernel of the fibre.
        span_length_m: Span length (m).

    Returns
    -------
        ``(flagged, confidence, min_abs_x, min_s)`` in model order.  The flags
        here are the two kernel-level warnings (material XCI edge gate and SCI
        small-``X`` branch); path-level flags are added by
        :func:`estimate_h3_error_budget`.
    """
    gated, fraction, min_abs_x, min_s, confidence = _span_metrics(kernel, span_length_m)
    flagged = gated | (min_abs_x < SCI_X_SERIES_SWITCH) | (min_s < XCI_DEEP_BREAKDOWN_S)
    return flagged, confidence, min_abs_x, min_s


def _iter_span_states(model: QoTModel, optical_path: MultiSpanOpticalPath) -> list[tuple[PCFMSpanKernel, float]]:
    """List the distinct ``(kernel, span length)`` states of a path.

    The split mirrors :func:`network_nlo_eval.models.route.accumulate_route`
    exactly, so the validity report describes the spans that were evaluated.
    Identical spans share a kernel and are collapsed to one entry: the metrics
    are pure functions of the kernel and the span length.

    Raises
    ------
        ValueError: A fibre has a non-positive maximum span length.
        TypeError: The model returned a kernel without the H3 validity metrics.
    """
    states: dict[tuple[int, int], tuple[PCFMSpanKernel, float]] = {}
    for fiber in optical_path.fiber_configs:
        kernel = model.kernel_for(fiber)
        if not isinstance(kernel, PCFMSpanKernel):
            raise TypeError(
                "the H3 error budget requires PCFMSpanKernel; "
                f"model {type(model).__name__} returned {type(kernel).__name__}."
            )
        remaining_km = float(fiber.length_km)
        while remaining_km > 1e-12:
            span_km = min(remaining_km, float(fiber.max_span_length_km))
            if span_km <= 0.0:
                raise ValueError("max_span_length_km must be positive for a non-empty fibre.")
            key = (id(kernel), int(round(span_km * 1e3)))
            states.setdefault(key, (kernel, span_km * 1e3))
            remaining_km -= span_km
    return list(states.values())


def _path_tilt_db(states: Sequence[tuple[PCFMSpanKernel, float]]) -> float:
    """Gain tilt of the evaluated span states.

    The route EDFA compensates each channel independently, so the path result
    is flat by construction; the tilt has to be read from the span kernel's net
    gain side output.  The worst span wins.
    """
    tilt = 0.0
    for kernel, _span_length_m in states:
        # 只有 H3 Raman 核记录逐跨增益；纯 PCFM 核没有拉曼状态，跳过即可，
        # 否则 H3LinkValidityProvider 在 ``pcfm`` 上会因缺属性直接崩溃。
        last_gain = getattr(kernel, "last_span_gain", None)
        if last_gain is None:
            continue
        gain = np.asarray(last_gain, dtype=np.float64)
        gain = gain[gain > 0.0]
        if gain.size < 2:
            continue
        tilt = max(tilt, float(10.0 * np.log10(float(np.max(gain)) / float(np.min(gain)))))
    return tilt


def estimate_h3_error_budget(
    model: QoTModel,
    optical_path: MultiSpanOpticalPath,
    power_in_w: NDArrayFloat,
    result: PathQoTResult,
    *,
    config: H3UncertaintyConfig | None = None,
) -> H3ErrorBudget:
    """Build the error budget of an already evaluated H3 path.

    The function is a post-processing step: it reads the cached kernels of the
    model and the path result, and never calls the Raman ODE again (the kernel
    caches are populated by ``evaluate_path``).

    Args:
        model: H3 model that produced ``result`` (must expose ``kernel_for``).
        optical_path: Path that was evaluated.
        power_in_w: ``(N,)`` launch power used for the evaluation.
        result: Result returned by ``model.evaluate_path``.
        config: Optional budget configuration.

    Returns
    -------
        The budget in model channel order.

    Raises
    ------
        ValueError: The power vector has the wrong shape or a negative entry.
    """
    cfg = config or H3UncertaintyConfig()
    power = np.asarray(power_in_w, dtype=np.float64)
    if power.shape != (model.n_nodes,) or np.any(power < 0.0):
        raise ValueError("power_in_w must be a non-negative vector with one value per channel.")

    states = _iter_span_states(model, optical_path)
    n_nodes = int(model.n_nodes)

    gated = np.zeros(n_nodes, dtype=bool)
    small_x = np.zeros(n_nodes, dtype=bool)
    low_s = np.zeros(n_nodes, dtype=bool)
    not_converged = np.zeros(n_nodes, dtype=bool)
    fraction = np.zeros(n_nodes, dtype=np.float64)
    min_abs_x = np.full(n_nodes, np.inf, dtype=np.float64)
    min_s = np.full(n_nodes, np.inf, dtype=np.float64)
    confidence = np.ones(n_nodes, dtype=np.float64)
    directions: list[str] = []

    for kernel, span_length_m in states:
        span_gated, span_fraction, span_min_x, span_min_s, span_confidence = _span_metrics(kernel, span_length_m)
        gated |= span_gated
        small_x |= span_min_x < SCI_X_SERIES_SWITCH
        low_s |= span_min_s < XCI_DEEP_BREAKDOWN_S
        fraction = np.maximum(fraction, span_fraction)
        min_abs_x = np.minimum(min_abs_x, span_min_x)
        min_s = np.minimum(min_s, span_min_s)
        # 最差跨决定整个链路的可信度：链路预算里任何一跨的近似失效都不能被别的跨平均掉。
        confidence = np.minimum(confidence, span_confidence)
        # 纯 PCFM 核没有泵状态（其拉曼增益是路由层的集总近似），用模型配置兜底；
        # H3 Raman 核有 ``active``/``active_pumps``，仍以核记录为准。
        if bool(getattr(kernel, "active", False)):
            directions.extend(str(pump.direction) for pump in kernel.active_pumps)
            if not bool(getattr(kernel, "last_raman_converged", True)):
                not_converged[:] = True

    if not directions:
        configured = getattr(model, "raman_pumps", None) or ()
        directions = [str(pump.direction) for pump in configured if float(pump.power_w) > 0.0]

    pump_mode = classify_pump_mode(directions)
    bandwidth = np.asarray(model.channel_bandwidth_hz_channels, dtype=np.float64)
    occupied_hz = float(np.ptp(np.asarray(model.f_abs_hz, dtype=np.float64)) + np.max(bandwidth))
    band_class = classify_bandwidth(occupied_hz)
    launched = power > 0.0
    signal_values = np.asarray(result.signal_power_w, dtype=np.float64)
    # 已发射但结果非有限（例如 RK4 溢出）不能按“未占用”处理：它是硬失效，
    # 必须保持 active=True，否则后面的 ``margin = np.where(active, ...)`` 会把
    # 无效信道的 inf margin 压回 0。
    unusable_signal = launched & ~np.isfinite(signal_values)
    active = launched & ((signal_values > 0.0) | unusable_signal)
    tilt_db = _path_tilt_db(states)
    # 倾斜是整条占用谱的属性，不是某一信道的属性；与 ``confidence`` 一样由最差跨
    # 决定。模型未记录逐跨报告时退化为 1（不产生软 margin），因此旧调用方逐位不变。
    span_reports = tuple(getattr(model, "last_span_validity", ()) or ())
    tilt_confidence_informational = min(
        (float(report.tilt_confidence_informational) for report in span_reports),
        default=1.0,
    )
    tilt_confidence_informational = float(np.clip(tilt_confidence_informational, 0.0, 1.0))

    # 收敛/适用域 flag 来自逐跨报告。纯 PCFM 核（含集总拉曼）没有拉曼剖面，
    # 只能由模型层声明；H3 核把每次耦合求解的收敛与宽波段多泵区域写在报告里。
    raman_not_converged = np.zeros(n_nodes, dtype=bool)
    raman_wideband = np.zeros(n_nodes, dtype=bool)
    pump_overlap = np.zeros(n_nodes, dtype=bool)
    pump_min_offset = float("inf")
    lumped_raman = np.zeros(n_nodes, dtype=bool)
    auto_under_resolved = False
    resolution_checked = False
    resolution_steps = 0
    resolution_changes: list[float] = []
    fwm_required_any = False
    fwm_reasons: list[str] = []
    fwm_min_dispersion = float("inf")
    fwm_zero_in_band = False
    fwm_reference_in_band = True
    fwm_mode = str(getattr(model, "fwm_validity_mode", "flag"))
    for report in span_reports:
        if not bool(getattr(report, "raman_converged", True)):
            raman_not_converged[:] = True
        if bool(getattr(report, "raman_wideband_multipump", False)):
            raman_wideband[:] = True
        pump_min_offset = min(
            pump_min_offset,
            float(getattr(report, "raman_pump_min_offset_hz", float("inf"))),
        )
        overlap_channels = np.asarray(
            getattr(report, "raman_pump_overlap_channels", ()),
            dtype=bool,
        )
        if overlap_channels.shape == (n_nodes,):
            pump_overlap |= overlap_channels
        elif bool(getattr(report, "raman_pump_signal_overlap", False)):
            # 旧报告没有逐信道掩码：保守地按全部已发射信道处理。
            pump_overlap |= launched
        if bool(getattr(report, "lumped_raman_approximation", False)):
            lumped_raman[:] = True
        if bool(getattr(report, "raman_resolution_checked", False)):
            resolution_checked = True
            resolution_steps = max(
                resolution_steps,
                int(getattr(report, "raman_resolution_check_steps", 0)),
            )
            resolution_changes.append(float(getattr(report, "raman_resolution_change", 1.0)))
        if bool(getattr(report, "raman_auto_under_resolved", False)):
            auto_under_resolved = True
        # 低色散 / FWM 适用域是整条链路的属性：任一跨落在该区域，链路预算就
        # 不能声称模型包含 FWM。字段来自逐跨报告，旧报告没有这些字段时保持默认
        # （不 flag、不 reject），因此旧调用方逐位不变。
        reason = str(getattr(report, "fwm_reason", "") or "")
        for token in reason.split("+") if reason else ():
            if token and token not in fwm_reasons:
                fwm_reasons.append(token)
        fwm_min_dispersion = min(
            fwm_min_dispersion,
            float(getattr(report, "min_abs_dispersion_ps_nm_km", float("inf"))),
        )
        fwm_zero_in_band = fwm_zero_in_band or bool(getattr(report, "zero_dispersion_in_band", False))
        fwm_reference_in_band = fwm_reference_in_band and bool(
            getattr(report, "dispersion_reference_in_band", True)
        )
        if bool(getattr(report, "fwm_required", False)):
            fwm_required_any = True
    resolution_change = max(resolution_changes, default=1.0)
    if not resolution_checked:
        # Nothing was compared: keep the neutral "not checked" values so the
        # default report is bit-for-bit the pre-audit report.
        auto_under_resolved = False
        resolution_steps = 0
        resolution_change = 1.0
    raman_not_converged |= not_converged
    if bool(getattr(model, "lumped_raman_active", False)):
        lumped_raman |= launched
    raman_not_converged &= launched
    raman_wideband &= launched
    pump_overlap &= launched
    lumped_raman &= launched
    tilt_class = "strong" if tilt_db >= cfg.strong_tilt_db else "normal"
    strong_tilt = active & (tilt_class == "strong")

    # 密集栅格：相邻矩形谱重叠，闭式 XCI 的非重叠推导不再成立。只标记，
    # 不改 valid / 点值 SNR；需要额外余量的调用方显式设置
    # ``dense_grid_penalty_db`` 或按间距标定的 ``dense_grid_penalty_schedule``。
    bandwidth_channels = np.asarray(model.channel_bandwidth_hz_channels, dtype=np.float64)
    min_spacing_hz, spacing_ratio = dense_grid_metrics(
        np.asarray(model.f_abs_hz, dtype=np.float64),
        bandwidth_channels,
        active,
    )
    dense_grid = bool(spacing_ratio < DENSE_GRID_SPACING_RATIO)
    active_bandwidth = bandwidth_channels[active]

    # 泵不收敛是硬失效：点的 NLI 不是解出来的，给任何有限 margin 都会误导。
    # ``+inf`` SNR（零噪声）仍是有限模型能给出的理想结果，只排除 NaN 与 -inf。
    # 集总拉曼有泵不是数值失效，而是模型适用域问题：保持点值/有限 margin，
    # 只通过 ``lumped_raman`` flag、confidence 扣分与 notes 让下游看见。
    snr_values = np.asarray(result.snr_db, dtype=np.float64)
    snr_ok = ~np.isnan(snr_values) & ~np.isneginf(snr_values)
    # 已发射但点值非有限（例如 RK4 溢出）不能按“未占用”处理：那是硬失效，
    # margin 必须是无穷，否则只检查 margin 的调用方会把它当成可用结果。
    unusable_result = launched & (~snr_ok | ~np.isfinite(signal_values))
    # 泵在信号带内是物理配置错误（不是收敛问题）：与未收敛一样是硬失效，
    # 对应信道 margin=inf；点值本身仍保留，便于调用方检查。
    valid = active & snr_ok & ~raman_not_converged & ~pump_overlap
    # 低色散/FWM：默认只 OR 进 flagged。reject 模式按信道把 valid 扣掉，但
    # NLI/ASE/SNR 与 margin 数值路径都不变（invalid 会拿到 inf margin）。
    fwm_channels = launched & fwm_required_any
    if fwm_mode == "reject":
        valid = valid & ~fwm_channels
    confidence = np.where(strong_tilt, confidence - 0.30, confidence)
    confidence = np.where(not_converged, confidence - 0.40, confidence)
    confidence = np.where(raman_wideband, confidence - 0.15, confidence)
    confidence = np.where(lumped_raman, confidence - 0.40, confidence)
    confidence = np.clip(confidence, 0.05, 1.0)
    confidence = np.where(active, confidence, 0.0)
    confidence = np.where(unusable_result, 0.0, confidence)
    confidence = np.where(pump_overlap, np.minimum(confidence, 0.2), confidence)

    entry, entry_notes = select_residual_entry(pump_mode, band_class, tilt_class)
    statistical = H3_STATISTICAL_TABLE[band_class]
    residual_bias = np.full(n_nodes, entry.bias_db * cfg.residual_scale, dtype=np.float64)
    residual_sigma = np.full(n_nodes, entry.sigma_db * cfg.residual_scale, dtype=np.float64)
    statistical_sigma = np.full(
        n_nodes,
        statistical.sigma_db * cfg.statistical_scale if cfg.include_statistical else 0.0,
        dtype=np.float64,
    )
    sigma = np.hypot(residual_sigma, statistical_sigma)
    penalty = cfg.validity_penalty_db * (1.0 - confidence)
    # 密集栅格余量：默认走标量路径，数值与引入 schedule 之前逐位相同。
    # schedule 非空时按有效性报告的 spacing_to_bandwidth 查表（逐信道数组，
    # 便于将来按信道比查表），并覆盖标量；超出最后一级时取最后一级（clamped）。
    schedule = cfg.dense_grid_penalty_schedule
    schedule_active = bool(dense_grid and schedule is not None)
    dense_penalty_clamped = False
    if schedule_active:
        penalty_value, dense_penalty_clamped = dense_grid_schedule_penalty(schedule, float(spacing_ratio))
        dense_penalty: float | NDArrayFloat = np.where(active, penalty_value, 0.0)
        dense_penalty_applied = bool(penalty_value > 0.0)
        dense_penalty_source = (
            f"calibrated dense-grid penalty schedule: {penalty_value:.2f} dB at "
            f"spacing/bandwidth={float(spacing_ratio):.4f}; calibrated safety margin from the v4 "
            "audit, not a model correction"
        )
        dense_penalty_note = (
            f"calibrated dense-grid penalty schedule {penalty_value:.2f} dB applied at "
            f"spacing/bandwidth {float(spacing_ratio):.4f} (calibrated safety margin, not a model correction)"
        )
        if dense_penalty_clamped:
            last_upper = float(schedule[-1][0])
            dense_penalty_note += (
                f"; spacing/bandwidth {float(spacing_ratio):.4f} is above the last calibrated "
                f"upper bound {last_upper:.4f}, penalty clamped to the last tier (outside the "
                "calibrated range, so the residual is not covered)"
            )
    else:
        dense_penalty = cfg.dense_grid_penalty_db if dense_grid else 0.0
        dense_penalty_applied = bool(dense_penalty > 0.0)
        dense_penalty_source = f"opt-in dense-grid penalty at {cfg.dense_grid_penalty_db:.2f} dB full scale"
        dense_penalty_note = f"dense-grid penalty {cfg.dense_grid_penalty_db:.2f} dB applied (opt-in)"
    margin = np.maximum(residual_bias, 0.0) + cfg.k_sigma * sigma + penalty + dense_penalty
    margin = np.where(valid, margin, np.inf)
    margin = np.where(active, margin, 0.0)

    flagged = (
        gated
        | small_x
        | low_s
        | strong_tilt
        | not_converged
        | raman_not_converged
        | raman_wideband
        | pump_overlap
        | lumped_raman
        | unusable_result
        | (dense_grid & active)
        | (auto_under_resolved & active)
        | fwm_channels
    )
    validity = ValidityReport(
        valid=valid,
        active=active,
        confidence=confidence,
        flagged=flagged,
        xci_edge_gated=gated,
        sci_small_x=small_x,
        xci_low_s=low_s,
        pump_not_converged=not_converged,
        raman_not_converged=raman_not_converged,
        raman_wideband_multipump=raman_wideband,
        raman_pump_signal_overlap=pump_overlap,
        lumped_raman=lumped_raman,
        strong_tilt=strong_tilt,
        min_abs_x_sci=min_abs_x,
        min_s_xci=min_s,
        xci_correction_fraction=fraction,
        tilt_db=tilt_db,
        pump_mode=pump_mode,
        band_class=band_class,
        span_count=len(states),
        tilt_confidence_informational=tilt_confidence_informational,
        min_channel_spacing_hz=float(min_spacing_hz),
        spacing_to_bandwidth=float(spacing_ratio),
        dense_grid_overlap=dense_grid,
        raman_pump_min_offset_hz=float(pump_min_offset),
        raman_auto_under_resolved=bool(auto_under_resolved),
        raman_resolution_check_steps=int(resolution_steps),
        raman_resolution_change=float(resolution_change),
        raman_resolution_checked=bool(resolution_checked),
        min_abs_dispersion_ps_nm_km=float(fwm_min_dispersion),
        zero_dispersion_in_band=bool(fwm_zero_in_band),
        dispersion_reference_in_band=bool(fwm_reference_in_band),
        fwm_required=bool(fwm_required_any),
        fwm_reason="+".join(fwm_reasons),
        fwm_validity_mode=fwm_mode,
    )

    terms = [
        UncertaintyTerm("model_residual_bias", "bias", residual_bias, entry.source, entry.calibrated),
        UncertaintyTerm("model_residual_spread", "spread", residual_sigma, entry.source, entry.calibrated),
        UncertaintyTerm(
            "waveform_statistical_spread",
            "spread",
            statistical_sigma,
            statistical.source,
            statistical.calibrated,
        ),
        UncertaintyTerm(
            "validity_penalty",
            "penalty",
            penalty,
            f"confidence-based penalty at {cfg.validity_penalty_db:.2f} dB full scale",
            False,
        ),
    ]
    if dense_penalty_applied:
        dense_penalty_values = (
            dense_penalty if isinstance(dense_penalty, np.ndarray) else np.where(active, dense_penalty, 0.0)
        )
        terms.append(
            UncertaintyTerm(
                "dense_grid_penalty",
                "penalty",
                dense_penalty_values,
                dense_penalty_source,
                False,
            )
        )
    notes = [
        f"operating point: pump_mode={pump_mode}, band_class={band_class}, tilt={tilt_db:.2f} dB",
        f"residual row: {entry.source}",
        f"statistical row: {statistical.source}",
        *entry_notes,
    ]
    if not entry.calibrated:
        notes.append("residual row is provisional: the reference behind it is not a converged calibration")
    if bool(np.any(not_converged)):
        notes.append("pump fixed point did not converge; all channels are marked invalid")
    if bool(np.any(unusable_result)):
        notes.append(
            "non-finite signal or SNR on launched channels; they stay active and get an infinite "
            "margin instead of being silently treated as unoccupied"
        )
    if bool(np.any(raman_wideband)):
        notes.append(
            "S/E wideband with a forward or multi-pump set: raman_steps='auto' uses the extended "
            "96-step branch; the flag is informational when the solve converged"
        )
    if bool(np.any(pump_overlap)):
        notes.append(
            "a Raman pump lies inside the occupied signal band; that is not a physical DRA "
            "configuration (the pump cannot be separated from the data channels), so the "
            f"channel is invalid and gets an infinite margin (min pump/band offset "
            f"{pump_min_offset / 1e12:.3f} THz). The point value is unchanged."
        )
    if bool(np.any(lumped_raman)):
        notes.append(
            "lumped Raman approximation with active pumps is not a trustworthy QoT: it misses the "
            "intra-span profile and pump depletion (5-28 dB optimistic in the accuracy matrix)"
        )
    if bool(np.any(gated)):
        notes.append("XCI edge gate corrected at least one channel; confidence reduced")
    if auto_under_resolved:
        notes.append(
            "auto resolution not certified: the solved Raman step count is under-resolved "
            f"(max relative change {resolution_change:.3g} against {resolution_steps} steps, "
            "opt-in raman_resolution_tol); margin is unchanged"
        )
    if bool(np.any(strong_tilt)):
        notes.append(f"power tilt {tilt_db:.2f} dB exceeds the strong-tilt threshold {cfg.strong_tilt_db:.2f} dB")
    if dense_grid:
        notes.append(
            f"dense grid: channel spacing {min_spacing_hz / 1e9:.2f} GHz is below the "
            f"{float(np.max(active_bandwidth)) / 1e9:.2f} GHz bandwidth; the closed-form XCI row "
            "is outside its non-overlapping derivation domain"
        )
    if dense_penalty_applied or schedule_active:
        notes.append(dense_penalty_note)
    if fwm_required_any:
        note = (
            f"PCFM/H3 do not contain FWM: the occupied band has min |D| = "
            f"{fwm_min_dispersion:.2f} ps/nm/km (reason: {'+'.join(fwm_reasons)}), so FWM is not "
            "negligible and low-dispersion band results are reference-only "
            "(PCFM/H3 未含 FWM，低色散波段结果仅作参考)"
        )
        if fwm_mode == "reject":
            note += "; fwm_validity_mode='reject' clears valid on those channels"
        notes.append(note)

    return H3ErrorBudget(
        model=str(getattr(model.info, "name", type(model).__name__)),
        bias_db=residual_bias,
        residual_sigma_db=residual_sigma,
        statistical_sigma_db=statistical_sigma,
        sigma_db=sigma,
        margin_db=margin,
        k_sigma=cfg.k_sigma,
        validity=validity,
        terms=tuple(terms),
        notes=tuple(notes),
    )
