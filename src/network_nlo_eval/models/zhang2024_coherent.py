"""Coherent distributed-Raman GN calculations based on Zhang et al. (2024).

The paper specifies a three-exponential profile and the structure of coherent
inter-span accumulation, but does not publish the individual closed-form
cross-span terms, calibration coefficients, data, or modulation correction.
This implementation evaluates inter-span phases inside the frequency integral,
uses analogous Buglia closed forms for same-span terms, and numerically
integrates the unpublished cross-span terms. Calibration follows the paper's
functional form using coefficients fitted in this repository, not the paper's
unpublished coefficients. It is therefore not a reproduction of the complete
published EGN model.

A span's phase is computed from its starting position and GN phase mismatch.
The link function uses a consistent common phase for all three profile terms,
as described in :func:`link_function`.

References
----------
Y. Zhang et al., "Accurate Closed-Form Coherent EGN Model Supporting
Distributed Raman Amplification", OECC 2024, doi:10.1109/OECC54135.2024.10975528.
"""

# ruff: noqa: PLC0415  # Keep closed-form imports local to their calculation.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
    from network_nlo_eval.models.buglia2024_closed import BugliaClosedProfile

__all__ = [
    "ZhangCoherentSpan",
    "ZhangSpanProfile",
    "calibration_factor",
    "coherent_eta",
    "fit_zhang_coherent_profile",
    "link_function",
    "phase_spm",
    "phase_xpm",
    "single_span_eta_closed",
]

#: Bounds for fitted decay rates in 1/m.
_MIN_RATE_NPM: float = 1.0e-7
_MAX_RATE_NPM: float = 5.0e-3

#: Below this phase mismatch use integration instead of the unstable closed form.
_PHI_FLOOR: float = 1.0e-35

#: Bound fitted A relative to attenuation; lower values can invalidate the
#: same-span SPM closed-form branch and degrade agreement with integration.
_A_LOWER_FACTOR: float = 0.2


# =============================================================================
# Eq. (1): backward-Raman exponential profile.
# =============================================================================


@dataclass(frozen=True)
class ZhangSpanProfile:
    """Per-channel three-exponential profile parameters from Zhang Eq. (1).

    The channel fiber attenuation is fixed; T, A, D and d are fitted.

    Attributes
    ----------
    alpha_npm, a_npm, d_npm : numpy.ndarray
        Channel attenuation and fitted forward and backward rates in 1/m.
    t_coeff, d_coeff : numpy.ndarray
        Fitted forward and backward coefficients.
    f_hz, bandwidths_hz : numpy.ndarray
        Channel frequencies and bandwidths in Hz.
    length_m : float
        Span length in meters.
    rmse : numpy.ndarray
        Relative profile fit RMSE for each channel.
    """

    alpha_npm: NDArrayFloat
    t_coeff: NDArrayFloat
    a_npm: NDArrayFloat
    d_coeff: NDArrayFloat
    d_npm: NDArrayFloat
    f_hz: NDArrayFloat
    bandwidths_hz: NDArrayFloat
    length_m: float
    rmse: NDArrayFloat

    @property
    def n_channels(self) -> int:
        """Return the number of channels."""
        return int(np.asarray(self.f_hz).size)

    def value(self, z_m: NDArrayFloat, index: int) -> NDArrayFloat:
        """Evaluate the channel profile at the given longitudinal positions."""
        z = np.asarray(z_m, dtype=np.float64)
        alpha = float(self.alpha_npm[index])
        t_value = float(self.t_coeff[index])
        a_value = float(self.a_npm[index])
        d_value = float(self.d_coeff[index])
        d_rate = float(self.d_npm[index])
        return cast(
            "NDArrayFloat",
            (1.0 + t_value) * np.exp(-alpha * z)
            - t_value * np.exp(-a_value * z)
            + d_value * np.exp(-d_rate * (self.length_m - z)),
        )


def _zhang_model(
    parameters: NDArrayFloat, z_m: NDArrayFloat, length_m: float, alpha: float
) -> NDArrayFloat:
    """Evaluate Eq. (1) from parameters T, A, D and d."""
    t_value, a_value, d_value, d_rate = parameters
    return cast(
        "NDArrayFloat",
        (1.0 + t_value) * np.exp(-alpha * z_m)
        - t_value * np.exp(-a_value * z_m)
        + d_value * np.exp(-d_rate * (length_m - z_m)),
    )


def fit_zhang_coherent_profile(
    profile_norm: NDArrayFloat,
    length_m: float,
    frequencies_hz: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    n_steps: int,
) -> ZhangSpanProfile:
    """Fit the Eq. (1) profile to a coupled Raman ODE result.

    A grid over decay rates and linear least squares provides starting values;
    all four coefficients are then jointly refined. Fiber attenuation stays fixed.

    Parameters
    ----------
    profile_norm : numpy.ndarray
        Normalized ODE profile of shape ``(2 * n_steps, N)``.
    length_m : float
        Span length in meters.
    frequencies_hz, bandwidths_hz : numpy.ndarray
        Per-channel center frequencies and bandwidths in Hz.
    alpha_npm : numpy.ndarray
        Channel fiber attenuation in 1/m.
    n_steps : int
        Number of RK4 steps used for the profile.

    Returns
    -------
    ZhangSpanProfile
        Fitted profile and per-channel relative RMSE.
    """
    step = length_m / n_steps
    z = np.empty(2 * n_steps, dtype=np.float64)
    for index in range(n_steps):
        z[2 * index] = step * (index + 0.5)
        z[2 * index + 1] = step * (index + 1.0)
    z_fit = np.concatenate(([0.0], z))
    n_channels = int(np.asarray(frequencies_hz).size)
    t_coeff = np.zeros(n_channels, dtype=np.float64)
    a_npm = np.full(n_channels, 2.0e-5, dtype=np.float64)
    d_coeff = np.zeros(n_channels, dtype=np.float64)
    d_npm = np.full(n_channels, 4.0e-5, dtype=np.float64)
    rmse = np.empty(n_channels, dtype=np.float64)
    floor = 0.05

    for index in range(n_channels):
        alpha = float(np.clip(alpha_npm[index], _MIN_RATE_NPM, _MAX_RATE_NPM))
        target = np.concatenate(([1.0], profile_norm[:, index]))
        weight = 1.0 / np.maximum(target, floor)

        def residual(
            parameters: NDArrayFloat,
            target: NDArrayFloat = target,
            weight: NDArrayFloat = weight,
            alpha: float = alpha,
        ) -> NDArrayFloat:
            return cast(
                "NDArrayFloat",
                (_zhang_model(parameters, z_fit, length_m, alpha) - target) * weight,
            )

        base = np.exp(-alpha * z_fit)
        best: tuple[float, NDArrayFloat] | None = None
        a_floor = _A_LOWER_FACTOR * alpha
        for a_guess in (a_floor, 8.0e-5, 3.0e-4):
            for d_guess in (2.0e-5, 1.0e-4, 3.0e-4):
                basis = (
                    np.stack(
                        [
                            base - np.exp(-a_guess * z_fit),
                            np.exp(-d_guess * (length_m - z_fit)),
                        ],
                        axis=1,
                    )
                    * weight[:, None]
                )
                rhs = (target - base) * weight
                try:
                    linear, *_ = np.linalg.lstsq(basis, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                start = np.array([linear[0], a_guess, linear[1], d_guess])
                cost = float(np.sqrt(np.mean(residual(start) ** 2)))
                if best is None or cost < best[0]:
                    best = (cost, start)

        parameters = np.array([0.0, alpha, 0.0, alpha]) if best is None else best[1]
        try:
            solved = least_squares(
                residual,
                np.clip(
                    parameters,
                    [-5.0, a_floor, -5.0, _MIN_RATE_NPM],
                    [5.0, _MAX_RATE_NPM, 5.0, _MAX_RATE_NPM],
                ),
                bounds=(
                    [-5.0, a_floor, -5.0, _MIN_RATE_NPM],
                    [5.0, _MAX_RATE_NPM, 5.0, _MAX_RATE_NPM],
                ),
                x_scale=np.array([1.0, 1.0e-5, 1.0, 1.0e-5]),
                max_nfev=4000,
                xtol=1.0e-12,
                ftol=1.0e-12,
                gtol=1.0e-12,
            )
            cost = float(np.sqrt(np.mean(residual(solved.x) ** 2)))
            if best is None or cost <= best[0]:
                parameters = solved.x
        except (ValueError, np.linalg.LinAlgError):
            pass

        t_coeff[index], a_npm[index], d_coeff[index], d_npm[index] = (
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            float(parameters[3]),
        )
        rmse[index] = float(np.sqrt(np.mean(residual(parameters) ** 2)))

    return ZhangSpanProfile(
        alpha_npm=np.asarray(alpha_npm, dtype=np.float64),
        t_coeff=t_coeff,
        a_npm=a_npm,
        d_coeff=d_coeff,
        d_npm=d_npm,
        f_hz=np.asarray(frequencies_hz, dtype=np.float64),
        bandwidths_hz=np.asarray(bandwidths_hz, dtype=np.float64),
        length_m=float(length_m),
        rmse=rmse,
    )


# =============================================================================
# Eq. (2): closed-form longitudinal link.
# =============================================================================


def link_function(
    profile: ZhangSpanProfile, index: int, phi: NDArrayFloat
) -> NDArrayComplex:
    """Evaluate the Zhang Eq. (2) approximate complex link function.

    The printed formula retains long-span terms but assigns a common phase only to
    the backward contribution. Here all three terms share ``exp(1j * phi * L)``
    so their relative phases are consistent for inter-span coherent accumulation.
    This retains the long-span approximation and is not the exact finite-span
    integral; the common phase cancels for a single span's squared magnitude.

    Parameters
    ----------
    profile : ZhangSpanProfile
        Fitted profile.
    index : int
        Channel index.
    phi : numpy.ndarray
        Phase mismatch in 1/m.

    Returns
    -------
    numpy.ndarray
        Complex link function for each phase mismatch.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=np.complex128))
    alpha = float(profile.alpha_npm[index])
    t_value = float(profile.t_coeff[index])
    a_value = float(profile.a_npm[index])
    d_value = float(profile.d_coeff[index])
    d_rate = float(profile.d_npm[index])
    length = float(profile.length_m)
    common = np.exp(1j * phi * length)
    forward = -(1.0 + t_value) / (-alpha + 1j * phi) + t_value / (-a_value + 1j * phi)
    backward = d_value / (d_rate + 1j * phi)
    return cast("NDArrayComplex", common * (forward + backward))


def _link_squared(
    profile: ZhangSpanProfile, index: int, phi: NDArrayFloat
) -> NDArrayFloat:
    """Return the squared magnitude of the approximate link function."""
    return cast("NDArrayFloat", np.abs(link_function(profile, index, phi)) ** 2)


# =============================================================================
# Phase mismatch, using the Buglia Eq. (34)/(42) convention.
# =============================================================================


def phase_spm(
    beta2: float, beta3: float, frequency_hz: float, reference_frequency_hz: float
) -> float:
    """Return the SPM phase coefficient relative to the dispersion reference."""
    return float(
        -4.0
        * np.pi**2
        * (beta2 + 2.0 * np.pi * beta3 * (frequency_hz - reference_frequency_hz))
    )


def phase_xpm(
    beta2: float,
    beta3: float,
    coi_frequency_hz: float,
    interferer_frequency_hz: float,
    reference_frequency_hz: float,
) -> float:
    """Return the XPM phase coefficient relative to the dispersion reference."""
    separation = interferer_frequency_hz - coi_frequency_hz
    dispersion = beta2 + np.pi * beta3 * (
        coi_frequency_hz + interferer_frequency_hz - 2.0 * reference_frequency_hz
    )
    return float(-4.0 * np.pi**2 * separation * dispersion)


# =============================================================================
# Eqs. (3)-(5): coherent NLI.
# =============================================================================


@dataclass(frozen=True)
class ZhangCoherentSpan:
    """One link span and its input spectrum.

    Attributes
    ----------
    profile : ZhangSpanProfile
        Fitted span profile.
    launch_power_w : numpy.ndarray
        Per-channel input powers in watts.
    start_m : float
        Span start position in meters along the link.
    """

    profile: ZhangSpanProfile
    launch_power_w: NDArrayFloat
    start_m: float


def _coherent_field(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    index: int,
    phi: NDArrayFloat,
    position_scale: float,
) -> NDArrayComplex:
    """Sum complex span fields with propagation phase and launch-power weights.

    Each field is weighted by the ratio of its input power to the reference power.
    """
    total = np.zeros_like(phi, dtype=np.complex128)
    for span in spans:
        ratio = float(span.launch_power_w[index]) / max(
            float(reference_powers_w[index]), 1.0e-30
        )
        total += (
            ratio
            * np.exp(1j * phi * span.start_m * position_scale)
            * link_function(span.profile, index, phi)
        )
    return total


def _band_average(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    index: int,
    phase_slope: float,
    half_band_hz: float,
    position_scale: float,
    alpha_reference_npm: float,
    nodes: NDArrayFloat,
    weights: NDArrayFloat,
) -> float:
    """Integrate squared coherent field over a band with a tangent substitution.

    The transform resolves the narrow coherence peak that uniform nodes can miss.
    """
    tangent = abs(phase_slope) * half_band_hz / max(alpha_reference_npm, _MIN_RATE_NPM)
    if tangent <= 1.0e-12:
        field = _coherent_field(
            spans, reference_powers_w, index, np.zeros(1), position_scale
        )
        return float(2.0 * half_band_hz * abs(field[0]) ** 2)
    angles = 0.5 * np.arctan(tangent) * (nodes + 1.0)
    quadrature_weights = 0.5 * np.arctan(tangent) * weights
    values = (alpha_reference_npm / phase_slope) * np.tan(angles)
    jacobian = alpha_reference_npm / abs(phase_slope) / np.cos(angles) ** 2
    field = _coherent_field(
        spans, reference_powers_w, index, phase_slope * values, position_scale
    )
    return float(2.0 * np.sum(quadrature_weights * jacobian * np.abs(field) ** 2))


def coherent_eta(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n_frequency: int = 96,
    n_spm_frequency: int = 48,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Numerically integrate coherent SPM and XPM coefficients across spans.

    Inter-span phase is inside the frequency integral, retaining same-span and
    cross-span terms without a scalar phase approximation.

    Parameters
    ----------
    spans : list of ZhangCoherentSpan
        Spans in propagation order.
    reference_powers_w : numpy.ndarray
        Per-channel reference launch powers in watts.
    bandwidths_hz, gamma_per_w_m : numpy.ndarray
        Per-channel bandwidths in Hz and nonlinear coefficients in 1/(W m).
    beta2, beta3 : float
        Dispersion and dispersion slope at the reference frequency.
    reference_frequency_hz : float
        Dispersion reference frequency in Hz.
    n_frequency, n_spm_frequency : int, optional
        Frequency quadrature node counts for XPM and each SPM dimension.

    Returns
    -------
    tuple of numpy.ndarray
        Per-channel SPM and XPM coefficients in 1/W².
    """
    frequencies = np.asarray(spans[0].profile.f_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    nodes, weights = np.polynomial.legendre.leggauss(int(n_frequency))  # type: ignore[no-untyped-call]
    spm_nodes, spm_weights = np.polynomial.legendre.leggauss(int(n_spm_frequency))  # type: ignore[no-untyped-call]
    eta_spm = np.zeros(n_channels, dtype=np.float64)
    eta_xpm = np.zeros(n_channels, dtype=np.float64)
    # Phase is already in inverse meters, so position scale is one.
    position_scale = 1.0

    for index in range(n_channels):
        if float(reference_powers_w[index]) <= 0.0:
            continue
        half_band = 0.5 * float(bandwidths[index])
        alpha_ref = float(spans[0].profile.alpha_npm[index])

        phi_spm = phase_spm(
            beta2, beta3, float(frequencies[index]), reference_frequency_hz
        )
        offsets = half_band * spm_nodes
        offset_weights = half_band * spm_weights
        spm_total = 0.0
        for offset_1, weight_1 in zip(offsets, offset_weights, strict=True):
            field = np.zeros(offsets.size, dtype=np.complex128)
            for span in spans:
                ratio = float(span.launch_power_w[index]) / max(
                    float(reference_powers_w[index]), 1.0e-30
                )
                phase = phi_spm * offset_1 * offsets
                field += (
                    ratio
                    * np.exp(1j * phase * span.start_m)
                    * link_function(span.profile, index, phase)
                )
            spm_total += weight_1 * float(np.sum(offset_weights * np.abs(field) ** 2))
        eta_spm[index] = (
            (16.0 / 27.0)
            * float(gammas[index]) ** 2
            / (2.0 * half_band) ** 2
            * spm_total
        )

        xpm_total = 0.0
        for interferer in range(n_channels):
            if interferer == index or float(reference_powers_w[interferer]) <= 0.0:
                continue
            phi_ik = phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            inner = _band_average(
                spans,
                reference_powers_w,
                interferer,
                phi_ik,
                half_band,
                position_scale,
                alpha_ref,
                nodes,
                weights,
            )
            xpm_total += (
                (
                    float(reference_powers_w[interferer])
                    / float(reference_powers_w[index])
                )
                ** 2
                * inner
                / float(bandwidths[interferer])
            )
        eta_xpm[index] = (32.0 / 27.0) * float(gammas[index]) ** 2 * xpm_total
    return eta_spm, eta_xpm


# =============================================================================
# Same-span closed-form terms.
# =============================================================================


def single_span_eta_closed(
    profile: ZhangSpanProfile,
    bandwidths_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Calculate same-span SPM and XPM with analogous Buglia closed forms.

    The Zhang profile is mapped to three poles. Cross-span terms are evaluated by
    :func:`coherent_eta` because their closed forms were not published.
    """
    from network_nlo_eval.models.buglia2024_closed import (
        eta_spm_closed,
        eta_xpm_closed,
    )
    from network_nlo_eval.models.buglia2024_closed import (
        phase_spm as buglia_phase_spm,
    )
    from network_nlo_eval.models.buglia2024_closed import (
        phase_xpm as buglia_phase_xpm,
    )

    converted = _to_buglia_profile(profile)
    frequencies = np.asarray(profile.f_hz, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)
    for index in range(n_channels):
        launch = float(powers_w[index])
        if launch <= 0.0:
            continue
        phi_i = buglia_phase_spm(
            beta2, beta3, float(frequencies[index]), reference_frequency_hz
        )
        spm[index] = eta_spm_closed(
            converted, index, phi_i, float(bandwidths[index]), float(gammas[index])
        )
        total = 0.0
        for interferer in range(n_channels):
            if interferer == index or float(powers_w[interferer]) <= 0.0:
                continue
            phi_ik = buglia_phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            total += eta_xpm_closed(
                converted,
                interferer,
                phi_ik,
                float(bandwidths[index]),
                float(gammas[index]),
                (float(powers_w[interferer]) / launch) ** 2,
            )
        xpm[index] = total
    return spm, xpm


def _to_buglia_profile(profile: ZhangSpanProfile) -> BugliaClosedProfile:
    """Map the Zhang profile to a three-pole Buglia closed-form carrier.

    The pole rates are ``(alpha, A, -d)`` and coefficients are
    ``(1 + T, -T, -D)``. This preserves the common-phase link magnitude.
    """
    from network_nlo_eval.models.buglia2024_closed import BugliaClosedProfile

    alpha = np.asarray(profile.alpha_npm, dtype=np.float64)
    a_value = np.asarray(profile.a_npm, dtype=np.float64)
    d_rate = np.asarray(profile.d_npm, dtype=np.float64)
    t_value = np.asarray(profile.t_coeff, dtype=np.float64)
    d_value = np.asarray(profile.d_coeff, dtype=np.float64)
    rates = np.stack([alpha, a_value, -d_rate], axis=1)
    coeffs = np.stack([1.0 + t_value, -t_value, -d_value], axis=1)
    return BugliaClosedProfile(
        alpha_npm=alpha,
        alpha_f_npm=np.full_like(alpha, _MIN_RATE_NPM),
        alpha_b_npm=np.full_like(alpha, _MIN_RATE_NPM),
        c_f=np.zeros_like(alpha),
        c_b=np.zeros_like(alpha),
        f_hz=np.asarray(profile.f_hz, dtype=np.float64),
        bandwidths_hz=np.asarray(profile.bandwidths_hz, dtype=np.float64),
        f_hat_hz=0.0,
        length_m=float(profile.length_m),
        rmse=np.asarray(profile.rmse, dtype=np.float64),
        pole_coeffs=coeffs,
        pole_rates=rates,
    )


# =============================================================================
# Section II-C: dataset calibration.
# =============================================================================


@dataclass(frozen=True)
class CalibrationTerm:
    """One calibration term from the paper's functional form.

    Attributes
    ----------
    name : str
        Name of the input parameter.
    parameters : tuple of float
        Coefficients ``(a, b, c, d)`` of ``1 / (a * exp(b * value**c + d))``.
    """

    name: str
    parameters: tuple[float, float, float, float]

    def factor(self, value: float) -> float:
        """Evaluate the calibration factor for one parameter value."""
        a, b, c, d = self.parameters
        safe = max(abs(float(value)), 1.0e-12) ** c
        return float(1.0 / (a * np.exp(b * safe + d)))


@dataclass(frozen=True)
class CalibrationSet:
    """Collection of independently fitted calibration terms.

    Attributes
    ----------
    terms : tuple of CalibrationTerm
        Per-parameter factors.
    """

    terms: tuple[CalibrationTerm, ...] = field(default_factory=tuple)

    def apply(self, values: dict[str, float]) -> float:
        """Multiply factors for supplied parameters; missing terms contribute one."""
        factor = 1.0
        for term in self.terms:
            if term.name in values:
                factor *= term.factor(values[term.name])
        return float(factor)


def calibration_factor(
    terms: tuple[CalibrationTerm, ...], values: dict[str, float]
) -> float:
    """Return the product of applicable calibration factors."""
    return CalibrationSet(terms).apply(values)


def cfm_and_egn_terms(
    spans: list[ZhangCoherentSpan],
    reference_powers_w: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n_frequency: int = 96,
    n_spm_frequency: int = 48,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Return closed-form and coherent-integral coefficients for calibration.

    The integral contains same-span and cross-span terms; the closed-form estimate
    combines same-span closed forms with numerically integrated cross-span terms.

    Returns
    -------
    tuple of numpy.ndarray
        CFM SPM, CFM XPM, integral SPM and integral XPM coefficients per channel.
    """
    egn_spm, egn_xpm = coherent_eta(
        spans,
        reference_powers_w,
        bandwidths_hz,
        gamma_per_w_m,
        beta2,
        beta3,
        reference_frequency_hz,
        n_frequency=n_frequency,
        n_spm_frequency=n_spm_frequency,
    )
    same_spm = np.zeros_like(egn_spm)
    same_xpm = np.zeros_like(egn_xpm)
    for span in spans:
        one_spm, one_xpm = coherent_eta(
            [span],
            reference_powers_w,
            bandwidths_hz,
            gamma_per_w_m,
            beta2,
            beta3,
            reference_frequency_hz,
            n_frequency=n_frequency,
            n_spm_frequency=n_spm_frequency,
        )
        same_spm += one_spm
        same_xpm += one_xpm
    closed_spm, closed_xpm = single_span_eta_closed(
        spans[0].profile,
        bandwidths_hz,
        reference_powers_w,
        gamma_per_w_m,
        beta2,
        beta3,
        reference_frequency_hz,
    )
    scale = sum(
        float(np.mean(span.launch_power_w / np.maximum(reference_powers_w, 1e-30))) ** 3
        for span in spans
    )
    cfm_spm = closed_spm * scale + (egn_spm - same_spm)
    cfm_xpm = closed_xpm * scale + (egn_xpm - same_xpm)
    return cfm_spm, cfm_xpm, egn_spm, egn_xpm
