"""Assemble full-span and remainder noise into a network link cache.

The historical power-only path starts with zero arrays and adds full-span
noise multiplied by the span count before adding remainder noise. Preserve
this operation order to retain established numerical results.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from network_nlo_eval.models.base import SpanContext
from network_nlo_eval.network.state import LinkNoiseCache

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import EDFAConfig
    from network_nlo_eval.rwa.validity import LinkValidity

#: Remainders at or below this length in meters do not form a span.
_REMAINDER_EPSILON_M = 1.0


def _emits_nli_amplitudes(evaluator: object) -> bool:
    """Return whether the evaluator declares inter-span NLI amplitude support.

    Evaluators with ``supports_span_context`` may provide complex amplitudes;
    other evaluators continue to use power-only accumulation.
    """
    return bool(getattr(evaluator, "supports_span_context", False))


def _assemble_coherent_link_noise(
    *,
    evaluator: MultiBandISRSGN,
    edfa_config: EDFAConfig | None,
    power_profile: NDArrayFloat,
    span_length_m: float,
    num_full_spans: int,
    remainder_length_m: float,
    num_channels: int,
    convention: str,
    validity: LinkValidity | None,
) -> LinkNoiseCache:
    """Assemble a link span by span, coherently when amplitudes are provided.

    Existing amplitudes are transferred by the square root of power transfer
    before adding each new amplitude; ASE and separated FWM accumulate as
    powers. A power-only first span selects power accumulation for the link.

    Raises
    ------
    TypeError
        If later spans change between coherent and power-only NLI conventions.
    """
    total_ase = np.zeros(num_channels, dtype=np.float64)
    total_spm_power = np.zeros(num_channels, dtype=np.float64)
    total_xpm_power = np.zeros(num_channels, dtype=np.float64)
    total_fwm = np.zeros(num_channels, dtype=np.float64)
    amp_spm = np.zeros(num_channels, dtype=np.complex128)
    amp_xpm = np.zeros(num_channels, dtype=np.complex128)
    #: None means no span yet; subsequent spans fix the accumulation convention.
    coherent: bool | None = None
    fwm_present = False
    fwm_is_additive = bool(getattr(evaluator, "fwm_noise_is_additive", False))

    span_lengths = [span_length_m] * num_full_spans
    if remainder_length_m > _REMAINDER_EPSILON_M:
        span_lengths.append(remainder_length_m)
    total_spans = len(span_lengths)

    power = np.array(power_profile, dtype=np.float64, copy=True)
    cumulative_m = 0.0
    for span_index, length_m in enumerate(span_lengths):
        previous_power = power
        step = evaluator._calc_span_output_with_context(
            power,
            length_m,
            edfa_config,
            SpanContext(
                span_index=span_index,
                total_spans=total_spans,
                cumulative_length_m=cumulative_m,
                span_length_m=length_m,
            ),
        )
        power = step.power_out_w
        transfer = np.divide(
            power, previous_power, out=np.zeros_like(power), where=previous_power > 0
        )

        step_amp = step.nli_amplitude_spm
        if step_amp is None:
            if coherent is True:
                msg = (
                    "evaluator returned a coherent NLI amplitude for an earlier span and a "
                    "power-only result for a later one; a model must pick one convention."
                )
                raise TypeError(msg)
            coherent = False
            total_spm_power *= transfer
            total_xpm_power *= transfer
            total_spm_power += step.spm_noise_w
            total_xpm_power += step.xpm_noise_w
            total_ase *= transfer
            total_ase += step.ase_noise_w
            if fwm_is_additive and fwm_present:
                total_fwm *= transfer
            step_fwm = getattr(step, "fwm_noise_w", None) if fwm_is_additive else None
            if step_fwm is not None:
                total_fwm += step_fwm
                fwm_present = True
        else:
            if coherent is False:
                msg = (
                    "evaluator returned a power-only NLI result for an earlier span and a "
                    "coherent amplitude for a later one; a model must pick one convention."
                )
                raise TypeError(msg)
            coherent = True
            amp_root = np.sqrt(transfer)
            amp_spm = amp_spm * amp_root + step_amp
            amp_xpm = amp_xpm * amp_root + step.nli_amplitude_xpm
            total_ase *= transfer
            total_ase += step.ase_noise_w
            if fwm_is_additive:
                if fwm_present:
                    total_fwm *= transfer
                step_fwm = getattr(step, "fwm_noise_w", None)
                if step_fwm is not None:
                    total_fwm += step_fwm
                    fwm_present = True

        cumulative_m += length_m

    if coherent:
        total_spm = cast("NDArrayFloat", np.abs(amp_spm) ** 2)
        total_xpm = cast("NDArrayFloat", np.abs(amp_xpm) ** 2)
    else:
        total_spm = total_spm_power
        total_xpm = total_xpm_power
    if fwm_present:
        total_xpm = total_xpm + total_fwm

    return LinkNoiseCache(
        spm=total_spm,
        xpm=total_xpm,
        ase=total_ase,
        span_count=total_spans,
        validity=validity,
        launch_power_convention=convention,
    )


def assemble_link_noise(
    *,
    evaluator: MultiBandISRSGN,
    edfa_config: EDFAConfig | None,
    power_profile: NDArrayFloat,
    span_length_m: float,
    num_full_spans: int,
    remainder_length_m: float,
    num_channels: int,
    validity: LinkValidity | None = None,
    launch_power_convention: str | None = None,
    coherent: bool = False,
) -> LinkNoiseCache:
    """Accumulate full-span and remainder noise into a link cache.

    Parameters
    ----------
    evaluator : MultiBandISRSGN
        Evaluator configured for the link fiber.
    edfa_config : EDFAConfig or None
        Span amplifier configuration.
    power_profile : numpy.ndarray
        Per-channel input power in the evaluator's convention, in watts.
    span_length_m : float
        Length of a full span in meters.
    num_full_spans : int
        Number of full spans.
    remainder_length_m : float
        Remaining span length in meters; lengths of one meter or less are ignored.
    num_channels : int
        Number of channels, used to initialize output arrays.
    validity : LinkValidity or None, optional
        Optional validity metadata; it does not affect noise calculations.
    launch_power_convention : str or None, optional
        Convention recorded in the cache, defaulting to the evaluator's.
    coherent : bool, optional
        When enabled and supported by the evaluator, assemble individual
        complex span amplitudes coherently. Other evaluators retain the
        historical full-span multiplication and remainder accumulation.

    Returns
    -------
    LinkNoiseCache
        SPM, XPM, and ASE powers, span count, convention, and optional validity.
        Zero-span outputs use separate arrays and do not share memory.
    """
    convention = launch_power_convention
    if convention is None:
        convention = str(getattr(evaluator, "launch_power_convention", "total"))

    if coherent and _emits_nli_amplitudes(evaluator):
        return _assemble_coherent_link_noise(
            evaluator=evaluator,
            edfa_config=edfa_config,
            power_profile=power_profile,
            span_length_m=span_length_m,
            num_full_spans=num_full_spans,
            remainder_length_m=remainder_length_m,
            num_channels=num_channels,
            convention=convention,
            validity=validity,
        )

    total_spm = np.zeros(num_channels, dtype=np.float64)
    total_xpm = np.zeros(num_channels, dtype=np.float64)
    total_ase = np.zeros(num_channels, dtype=np.float64)
    total_spans = 0

    if num_full_spans > 0:
        _, s_spm, s_xpm, s_ase = evaluator._calc_span_noise_and_power(
            power_profile,
            span_length_m,
            edfa_config,
            None,
            n_effective_spans=1,
        )
        total_spm += s_spm * num_full_spans
        total_xpm += s_xpm * num_full_spans
        total_ase += s_ase * num_full_spans
        total_spans += num_full_spans

    if remainder_length_m > _REMAINDER_EPSILON_M:
        _, r_spm, r_xpm, r_ase = evaluator._calc_span_noise_and_power(
            power_profile,
            remainder_length_m,
            edfa_config,
            n_effective_spans=1,
        )
        total_spm += r_spm
        total_xpm += r_xpm
        total_ase += r_ase
        total_spans += 1

    return LinkNoiseCache(
        spm=total_spm,
        xpm=total_xpm,
        ase=total_ase,
        span_count=total_spans,
        validity=validity,
        launch_power_convention=convention,
    )
