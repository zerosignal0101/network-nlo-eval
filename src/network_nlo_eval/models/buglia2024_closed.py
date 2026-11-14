"""Closed-form Raman-amplified GN kernels from Buglia et al. (2024).

The coupled Raman ODE is solved separately. This module fits the five-parameter
profile, evaluates the link function and the closed-form SPM and XPM terms,
and provides numerical quadrature as an independent reference.

The Eq. (12) final term uses the missing plus sign from Appendix D Eq. (56).
The sine-term sign follows the derivation and numerical integration, rather
than the printed Eq. (9), (11) and (12). Baseband phases are calculated
relative to an explicit dispersion reference frequency. As in the paper,
the closed forms use rectangular-band and infinite-band approximations; the
SPM approximation also evaluates a logarithm at a stationary point.

References
----------
H. Buglia et al., "A Closed-Form Expression for the Gaussian Noise Model in
the Presence of Raman Amplification", J. Lightwave Technol. 42(2), 636-648,
2024, doi:10.1109/jlt.2023.3315127.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat

__all__ = [
    "BugliaClosedProfile",
    "eta_spm_closed",
    "eta_xpm_closed",
    "fit_buglia_profile_5param",
    "link_mu_closed",
    "nli_closed_form",
    "nli_integral",
    "phase_spm",
    "phase_xpm",
]

#: Sum indices in Eqs. (9)-(12): 0 <= l1 + l2 <= 1.
_SUM_TERMS: tuple[tuple[int, int], ...] = ((0, 0), (1, 0), (0, 1))

#: Bounds for fitted decay rates in 1/m; the upper bound covers the S band.
_MIN_RATE_NPM: float = 1.0e-7
_MAX_RATE_NPM: float = 5.0e-3

#: Below this phase threshold the closed form is ill-conditioned; use the
#: integral baseline. The SPM threshold must stay well below its 1e-24 scale.
_PHI_FLOOR: float = 1.0e-35

#: Bound fitted signal attenuation relative to fiber loss. Without the upper
#: bound, the ISRS terms can absorb attenuation and cause more than 4 dB
#: disagreement with the integral baseline under the infinite-band approximation.
_ALPHA_BOUNDS: tuple[float, float] = (0.5, 2.0)


# =============================================================================
# Eq. (2): five-parameter semi-analytic profile.
# =============================================================================


@dataclass(frozen=True)
class BugliaClosedProfile:
    """Five-parameter Buglia Eq. (2) profile fitted independently per channel.

    Attributes
    ----------
    alpha_npm, alpha_f_npm, alpha_b_npm : numpy.ndarray
        Fitted signal, forward-pump and backward-pump effective attenuations (1/m).
    c_f, c_b : numpy.ndarray
        Fitted forward and backward coefficients (1/(m Hz)).
    f_hz, bandwidths_hz : numpy.ndarray
        Channel center frequencies and bandwidths in Hz.
    f_hat_hz : float
        Mean pump frequency in Hz.
    length_m : float
        Span length in meters.
    rmse : numpy.ndarray
        Per-channel relative profile fit RMSE.
    forward_kappa_infinite : bool
        Use the long-span forward approximation for Zhang-compatible link terms.
    pole_coeffs, pole_rates : numpy.ndarray or None
        Optional coefficients and rates of a three-pole alternative profile.
    """

    alpha_npm: NDArrayFloat
    alpha_f_npm: NDArrayFloat
    alpha_b_npm: NDArrayFloat
    c_f: NDArrayFloat
    c_b: NDArrayFloat
    f_hz: NDArrayFloat
    bandwidths_hz: NDArrayFloat
    f_hat_hz: float
    length_m: float
    rmse: NDArrayFloat
    forward_kappa_infinite: bool = False
    #: Optional three-pole profile, A(phi) = sum_k c_k / (r_k - j phi),
    #: allows same-span closed forms to match the Zhang link convention.
    pole_coeffs: NDArrayFloat | None = None
    pole_rates: NDArrayFloat | None = None

    @property
    def n_channels(self) -> int:
        """Return the number of channels."""
        return int(np.asarray(self.f_hz).size)

    def bandwidth_hz(self, index: int) -> float:
        """Return the channel bandwidth in Hz."""
        return float(np.asarray(self.bandwidths_hz)[index])

    def delta_hz(self, index: int) -> float:
        """Return the offset ``f_i - f_hat`` in Hz."""
        return float(self.f_hz[index]) - float(self.f_hat_hz)

    def value(self, z_m: NDArrayFloat, index: int) -> NDArrayFloat:
        """Evaluate the channel profile at longitudinal positions."""
        z = np.asarray(z_m, dtype=np.float64)
        alpha = float(self.alpha_npm[index])
        alpha_f = float(self.alpha_f_npm[index])
        alpha_b = float(self.alpha_b_npm[index])
        leff = -np.expm1(-alpha_f * z) / alpha_f
        ltilde = (
            np.exp(-alpha_b * (self.length_m - z)) - np.exp(-alpha_b * self.length_m)
        ) / alpha_b
        delta = self.delta_hz(index)
        return np.exp(-alpha * z) * (
            1.0
            - (float(self.c_f[index]) * leff + float(self.c_b[index]) * ltilde) * delta
        )


def _profile_from_parameters(
    parameters: NDArrayFloat,
    z_m: NDArrayFloat,
    length_m: float,
    delta_hz: float,
) -> NDArrayFloat:
    """Evaluate the Eq. (2) profile from its five parameters."""
    alpha, alpha_f, alpha_b, c_f, c_b = parameters
    leff = -np.expm1(-alpha_f * z_m) / alpha_f
    ltilde = (
        np.exp(-alpha_b * (length_m - z_m)) - np.exp(-alpha_b * length_m)
    ) / alpha_b
    return np.exp(-alpha * z_m) * (1.0 - (c_f * leff + c_b * ltilde) * delta_hz)


def fit_buglia_profile_5param(
    profile_norm: NDArrayFloat,
    length_m: float,
    frequencies_hz: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    f_hat_hz: float,
    n_steps: int,
) -> BugliaClosedProfile:
    """Fit the five Eq. (2) coefficients independently for each channel.

    Linear least squares first finds coefficient starts on a grid of decay rates;
    the best starts are refined jointly with nonlinear least squares. Relative
    residuals prevent low-power profile tails from dominating the fit.

    Parameters
    ----------
    profile_norm : numpy.ndarray
        Normalized RK4 profile of shape ``(2 * n_steps, N)``.
    length_m : float
        Span length in meters.
    frequencies_hz, bandwidths_hz : numpy.ndarray
        Per-channel frequencies and bandwidths in Hz, in profile order.
    alpha_npm : numpy.ndarray
        Fiber attenuation in 1/m, used to initialize the fit.
    f_hat_hz : float
        Mean pump frequency, or the channel-weighted mean without pumps.
    n_steps : int
        RK4 steps used to generate the profile.

    Returns
    -------
    BugliaClosedProfile
        Fitted coefficients and per-channel relative RMSE.
    """
    z = _profile_grid(profile_norm, length_m, n_steps)
    # Anchor the fit with rho(0) = 1 to constrain forward and backward rates.
    z_fit = np.concatenate(([0.0], z))
    n_channels = int(np.asarray(frequencies_hz).size)
    alpha_fit = np.empty(n_channels, dtype=np.float64)
    alpha_f_fit = np.empty(n_channels, dtype=np.float64)
    alpha_b_fit = np.empty(n_channels, dtype=np.float64)
    c_f = np.empty(n_channels, dtype=np.float64)
    c_b = np.empty(n_channels, dtype=np.float64)
    rmse = np.empty(n_channels, dtype=np.float64)
    floor = 0.05

    scale = np.array([1.0e-5, 1.0e-5, 1.0e-5, 1.0e-18, 1.0e-18])

    for index in range(n_channels):
        alpha_lo = _ALPHA_BOUNDS[0] * float(alpha_npm[index])
        alpha_hi = _ALPHA_BOUNDS[1] * float(alpha_npm[index])
        lower = np.array([alpha_lo, _MIN_RATE_NPM, _MIN_RATE_NPM, -1.0e-9, -1.0e-9])
        upper = np.array([alpha_hi, _MAX_RATE_NPM, _MAX_RATE_NPM, 1.0e-9, 1.0e-9])
        target = np.concatenate(([1.0], profile_norm[:, index]))
        weight = 1.0 / np.maximum(target, floor)
        delta = float(frequencies_hz[index]) - float(f_hat_hz)

        def residual(
            parameters: NDArrayFloat, target=target, weight=weight, delta=delta
        ) -> NDArrayFloat:
            model = _profile_from_parameters(parameters, z_fit, length_m, delta)
            return (model - target) * weight

        candidates = _linear_starts(
            z_fit, target, weight, length_m, delta, float(alpha_npm[index])
        )
        best: tuple[float, NDArrayFloat] | None = None
        for start in candidates[:3]:
            try:
                solved = least_squares(
                    residual,
                    np.clip(start, lower + 1.0e-12, upper - 1.0e-12),
                    bounds=(lower, upper),
                    x_scale=scale,
                    max_nfev=4000,
                    xtol=1.0e-12,
                    ftol=1.0e-12,
                    gtol=1.0e-12,
                )
            except (ValueError, np.linalg.LinAlgError):
                continue
            cost = float(np.sqrt(np.mean(residual(solved.x) ** 2)))
            if best is None or cost < best[0]:
                best = (cost, solved.x)

        if best is None:
            parameters = np.array(
                [alpha_npm[index], alpha_npm[index], alpha_npm[index], 0.0, 0.0]
            )
            cost = float(np.sqrt(np.mean(residual(parameters) ** 2)))
        else:
            cost, parameters = best

        (
            alpha_fit[index],
            alpha_f_fit[index],
            alpha_b_fit[index],
            c_f[index],
            c_b[index],
        ) = (
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            float(parameters[3]),
            float(parameters[4]),
        )
        rmse[index] = cost

    return BugliaClosedProfile(
        alpha_npm=alpha_fit,
        alpha_f_npm=alpha_f_fit,
        alpha_b_npm=alpha_b_fit,
        c_f=c_f,
        c_b=c_b,
        f_hz=np.asarray(frequencies_hz, dtype=np.float64),
        bandwidths_hz=np.asarray(bandwidths_hz, dtype=np.float64),
        f_hat_hz=float(f_hat_hz),
        length_m=float(length_m),
        rmse=rmse,
    )


def _linear_starts(
    z_fit: NDArrayFloat,
    target: NDArrayFloat,
    weight: NDArrayFloat,
    length_m: float,
    delta_hz: float,
    alpha_npm: float,
) -> list[NDArrayFloat]:
    """Solve the linear forward and backward coefficients for each decay-rate start."""
    rates = np.array([alpha_npm, 2.0e-4, 5.0e-4, 1.0e-3])
    decays = np.array([3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3])
    scored: list[tuple[float, NDArrayFloat]] = []
    for alpha in rates:
        base = np.exp(-alpha * z_fit)
        for alpha_f in decays:
            leff = -np.expm1(-alpha_f * z_fit) / alpha_f
            for alpha_b in decays:
                ltilde = (
                    np.exp(-alpha_b * (length_m - z_fit)) - np.exp(-alpha_b * length_m)
                ) / alpha_b
                basis = (
                    -base[:, None]
                    * delta_hz
                    * np.stack([leff, ltilde], axis=1)
                    * weight[:, None]
                )
                rhs = (target - base) * weight
                try:
                    coefficients, *_ = np.linalg.lstsq(basis, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                if not np.all(np.isfinite(coefficients)):
                    continue
                model = base * (
                    1.0 - (coefficients[0] * leff + coefficients[1] * ltilde) * delta_hz
                )
                cost = float(np.sqrt(np.mean(((model - target) * weight) ** 2)))
                scored.append(
                    (
                        cost,
                        np.array(
                            [alpha, alpha_f, alpha_b, coefficients[0], coefficients[1]]
                        ),
                    )
                )
    scored.sort(key=lambda item: item[0])
    if not scored:
        return [np.array([alpha_npm, alpha_npm, alpha_npm, 0.0, 0.0])]
    return [parameters for _, parameters in scored[:3]]


def _profile_grid(
    profile_norm: NDArrayFloat,  # noqa: ARG001 - Retain the reference interface.
    length_m: float,
    n_steps: int,
) -> NDArrayFloat:
    """Return midpoint and endpoint positions of the RK4 profile."""
    step = length_m / n_steps
    positions = np.empty(2 * n_steps, dtype=np.float64)
    for index in range(n_steps):
        positions[2 * index] = step * (index + 0.5)
        positions[2 * index + 1] = step * (index + 1.0)
    return positions


def _initial_guesses(
    profile: NDArrayFloat,
    length_m: float,
    alpha_npm: float,
    delta_hz: float,
) -> list[NDArrayFloat]:
    """Return exponential, fast and slow initial guesses for the five-parameter fit."""
    alpha_tail = -np.log(max(float(profile[-1]), 1.0e-6)) / length_m
    alpha_tail = float(np.clip(alpha_tail, _MIN_RATE_NPM, _MAX_RATE_NPM))
    sign = -1.0 if delta_hz < 0.0 else 1.0
    return [
        np.array([alpha_npm, alpha_npm, alpha_npm, 0.0, 0.0]),
        np.array([alpha_tail, 2.0 * alpha_npm, 2.0 * alpha_npm, 0.0, 0.0]),
        np.array([alpha_npm, 5.0 * alpha_npm, 2.0 * alpha_npm, sign * 2.0e-19, 0.0]),
        np.array([alpha_npm, 2.0 * alpha_npm, 5.0 * alpha_npm, 0.0, sign * 2.0e-19]),
        np.array([alpha_tail, alpha_npm, alpha_npm, sign * 1.0e-18, -sign * 1.0e-19]),
    ]


# =============================================================================
# Eqs. (9)-(12): closed-form link and NLI terms.
# =============================================================================


def _term_parameters(
    profile: BugliaClosedProfile, index: int, l1: int, l2: int
) -> tuple[float, ...]:
    """Return the decay, forward, backward and coefficient terms for one sum index.

    In pole mode the three indices map to the three poles, and the cross terms
    vanish, yielding the analytic squared magnitude of the pole sum.
    """
    if profile.pole_coeffs is not None and profile.pole_rates is not None:
        slot = _SUM_TERMS.index((l1, l2))
        rates = np.asarray(profile.pole_rates, dtype=np.float64)
        coeffs = np.asarray(profile.pole_coeffs, dtype=np.float64)
        return float(rates[index, slot]), 0.0, 1.0, float(coeffs[index, slot])
    alpha = float(profile.alpha_npm[index])
    alpha_f = float(profile.alpha_f_npm[index])
    alpha_b = float(profile.alpha_b_npm[index])
    length = float(profile.length_m)
    delta = profile.delta_hz(index)
    t_f = -float(profile.c_f[index]) * delta / alpha_f
    t_b = -float(profile.c_b[index]) * delta / alpha_b
    t_0 = 1.0 + t_f - t_b * np.exp(-alpha_b * length)
    alpha_l = alpha + l1 * alpha_f - l2 * alpha_b
    if profile.forward_kappa_infinite:
        kappa_f = 1.0
    else:
        kappa_f = np.exp(-(alpha + l1 * alpha_f) * length)
    kappa_b = np.exp(-l2 * alpha_b * length)
    upsilon = t_0 * (-t_f / t_0) ** l1 * (t_b / t_0) ** l2 if abs(t_0) > 0.0 else 0.0
    return alpha_l, kappa_f, kappa_b, upsilon


def link_mu_closed(
    profile: BugliaClosedProfile, index: int, phi: NDArrayFloat
) -> NDArrayFloat:
    """Evaluate the Buglia Eq. (9) closed-form link function.

    Parameters
    ----------
    profile : BugliaClosedProfile
        Five-parameter fitted profile.
    index : int
        Channel index.
    phi : numpy.ndarray
        Phase-mismatch slopes in 1/m.

    Returns
    -------
    numpy.ndarray
        Link function values for each phase mismatch.
    """
    phi = np.atleast_1d(np.asarray(phi, dtype=np.float64))
    length = float(profile.length_m)
    total = np.zeros_like(phi)
    for l1, l2 in _SUM_TERMS:
        a1, kf1, kb1, u1 = _term_parameters(profile, index, l1, l2)
        for l1p, l2p in _SUM_TERMS:
            a2, kf2, kb2, u2 = _term_parameters(profile, index, l1p, l2p)
            denominator = (a1**2 + phi**2) * (a2**2 + phi**2)
            shared = (a1 * a2 + phi**2) / denominator
            main = (kf1 * kf2 + kb1 * kb2) * shared
            cross = (kf1 * kb2 + kb1 * kf2) * shared * np.cos(phi * length)
            quadrature = (
                (kf1 * kb2 - kb1 * kf2)
                * (a1 - a2)
                * phi
                / denominator
                * np.sin(phi * length)
            )
            total += u1 * u2 * (main - cross - quadrature)
    return total


def eta_xpm_closed(
    profile: BugliaClosedProfile,
    interferer: int,
    phase_slope: float,
    coi_bandwidth_hz: float,
    gamma_per_w_m: float,
    power_ratio_squared: float,
) -> float:
    """Evaluate the Eq. (11) XPM coefficient for one interfering channel.

    Parameters
    ----------
    profile : BugliaClosedProfile
        Five-parameter fitted profile.
    interferer : int
        Interfering channel index.
    phase_slope : float
        Phase mismatch slope from :func:`phase_xpm`.
    coi_bandwidth_hz : float
        Channel-of-interest bandwidth in Hz.
    gamma_per_w_m : float
        Nonlinear coefficient in 1/(W m).
    power_ratio_squared : float
        Squared ratio of interferer to channel-of-interest launch powers.

    Returns
    -------
    float
        XPM coefficient in 1/W².

    Notes
    -----
    The prefactor uses the interfering channel's own bandwidth.
    """
    phi = float(phase_slope)
    if abs(phi) < _PHI_FLOOR:
        msg = "phase_slope is too small for the closed form; use the integral baseline."
        raise ValueError(msg)
    bandwidth_k = (
        float(profile.bandwidth_hz(interferer))
        if hasattr(profile, "bandwidth_hz")
        else None
    )
    if bandwidth_k is None:
        msg = "BugliaClosedProfile must carry the interferer bandwidth."
        raise AttributeError(msg)
    length = float(profile.length_m)
    total = 0.0
    for l1, l2 in _SUM_TERMS:
        a1, kf1, kb1, u1 = _term_parameters(profile, interferer, l1, l2)
        for l1p, l2p in _SUM_TERMS:
            a2, kf2, kb2, u2 = _term_parameters(profile, interferer, l1p, l2p)
            prefactor = u1 * u2 / (phi * (a1 + a2))
            main = (
                2.0
                * (kf1 * kf2 + kb1 * kb2)
                * (
                    np.arctan(phi * coi_bandwidth_hz / (2.0 * a1))
                    + np.arctan(phi * coi_bandwidth_hz / (2.0 * a2))
                )
            )
            cross_cos = np.pi * (
                -(kf1 * kb2 + kb1 * kf2)
                * (
                    np.sign(phi / a1) * np.exp(-abs(a1 * length))
                    + np.sign(phi / a2) * np.exp(-abs(a2 * length))
                )
                - (kf1 * kb2 - kb1 * kf2)
                * (
                    np.sign(-phi) * np.exp(-abs(a1 * length))
                    + np.sign(phi) * np.exp(-abs(a2 * length))
                )
            )
            total += prefactor * (main + cross_cos)
    return float(
        (32.0 / 27.0) * gamma_per_w_m**2 / bandwidth_k * power_ratio_squared * total
    )


def eta_spm_closed(
    profile: BugliaClosedProfile,
    index: int,
    phase_factor: float,
    bandwidth_hz: float,
    gamma_per_w_m: float,
) -> float:
    """Evaluate the Eq. (12) SPM coefficient.

    The final term includes the plus sign supplied by Appendix D Eq. (56).

    Parameters
    ----------
    profile : BugliaClosedProfile
        Five-parameter fitted profile.
    index : int
        Channel index.
    phase_factor : float
        Phase coefficient from :func:`phase_spm`.
    bandwidth_hz : float
        Channel bandwidth in Hz.
    gamma_per_w_m : float
        Nonlinear coefficient in 1/(W m).

    Returns
    -------
    float
        SPM coefficient in 1/W².
    """
    phi = float(phase_factor)
    if abs(phi) < _PHI_FLOOR:
        msg = (
            "phase_factor is too small for the closed form; use the integral baseline."
        )
        raise ValueError(msg)
    length = float(profile.length_m)
    log_argument = float(np.log(np.sqrt(phi * length / (2.0 * np.pi)) * bandwidth_hz))
    total = 0.0
    for l1, l2 in _SUM_TERMS:
        a1, kf1, kb1, u1 = _term_parameters(profile, index, l1, l2)
        for l1p, l2p in _SUM_TERMS:
            a2, kf2, kb2, u2 = _term_parameters(profile, index, l1p, l2p)
            prefactor = u1 * u2 * np.pi / (phi * (a1 + a2))
            main = (
                2.0
                * (kf1 * kf2 + kb1 * kb2)
                * (
                    np.arcsinh(3.0 * phi * bandwidth_hz**2 / (8.0 * np.pi * a1))
                    + np.arcsinh(3.0 * phi * bandwidth_hz**2 / (8.0 * np.pi * a2))
                )
            )
            correction = (
                4.0
                * log_argument
                * (
                    -(kf1 * kb2 + kb1 * kf2)
                    * (
                        np.sign(phi / a1) * np.exp(-abs(a1 * length))
                        + np.sign(phi / a2) * np.exp(-abs(a2 * length))
                    )
                    - (kf1 * kb2 - kb1 * kf2)
                    * (
                        np.sign(-phi) * np.exp(-abs(a1 * length))
                        + np.sign(phi) * np.exp(-abs(a2 * length))
                    )
                )
            )
            total += prefactor * (main + correction)
    return float((16.0 / 27.0) * gamma_per_w_m**2 / bandwidth_hz**2 * total)


# =============================================================================
# Phase mismatch (Eqs. 34 and 42).
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
# Eqs. (3) and (4): per-span NLI and span accumulation.
# =============================================================================


def nli_closed_form(
    profile: BugliaClosedProfile,
    bandwidths_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    spm_coherence_exponent: float = 0.0,
    n_spans: int = 1,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Calculate per-span SPM and XPM noise with the closed forms.

    The route supplies each span's launch power. SPM is multiplied by
    ``n_spans ** spm_coherence_exponent``; the default exponent zero corresponds
    to power addition across spans.

    Parameters
    ----------
    profile : BugliaClosedProfile
        Fitted profile.
    bandwidths_hz, powers_w, gamma_per_w_m : numpy.ndarray
        Per-channel bandwidths (Hz), launch powers (W), and nonlinear coefficients
        (1/(W m)), in the profile's channel order.
    beta2, beta3 : float
        Dispersion and dispersion slope at the reference frequency, in s²/m and s³/m.
    reference_frequency_hz : float
        Reference frequency for dispersion in Hz.
    spm_coherence_exponent : float, optional
        Eq. (4) SPM coherence exponent.
    n_spans : int, optional
        Number of spans for the coherence factor.

    Returns
    -------
    tuple of numpy.ndarray
        Per-channel SPM and XPM noise powers in watts.
    """
    frequencies = np.asarray(profile.f_hz, dtype=np.float64)
    powers = np.asarray(powers_w, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    span_factor = float(n_spans) ** float(spm_coherence_exponent)
    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)

    for index in range(n_channels):
        launch = float(powers[index])
        if launch <= 0.0:
            continue
        phase_i = phase_spm(
            beta2, beta3, float(frequencies[index]), reference_frequency_hz
        )
        try:
            eta_spm = eta_spm_closed(
                profile, index, phase_i, float(bandwidths[index]), float(gammas[index])
            )
        except ValueError:
            eta_spm = 0.0
        spm[index] = eta_spm * span_factor * launch**3

        eta_xpm = 0.0
        for interferer in range(n_channels):
            if interferer == index or powers[interferer] <= 0.0:
                continue
            phase_ik = phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            try:
                contribution = eta_xpm_closed(
                    profile,
                    interferer,
                    phase_ik,
                    float(bandwidths[index]),
                    float(gammas[index]),
                    (float(powers[interferer]) / launch) ** 2,
                )
            except ValueError:
                contribution = 0.0
            eta_xpm += contribution
        xpm[index] = eta_xpm * launch**3
    return spm, xpm


# =============================================================================
# Numerical integral baseline (Eqs. 6-8) for closed-form validation.
# =============================================================================


def _band_average_link_squared(
    profile: BugliaClosedProfile,
    index: int,
    phase_slope: float,
    half_band_hz: float,
    alpha_reference_npm: float,
    nodes: NDArrayFloat,
    weights: NDArrayFloat,
) -> float:
    """Integrate the squared link function over a channel band using a tangent transform."""
    if abs(phase_slope) * half_band_hz <= 1.0e-12 * max(
        alpha_reference_npm, _MIN_RATE_NPM
    ):
        link_zero = _link_exact(profile, index, np.zeros(1))
        return float(2.0 * half_band_hz * abs(link_zero[0]) ** 2)
    tangent = abs(phase_slope) * half_band_hz / max(alpha_reference_npm, _MIN_RATE_NPM)
    # The squared link is even in v; integrate the positive half and double it.
    angles = 0.5 * np.arctan(tangent) * (nodes + 1.0)
    quadrature_weights = 0.5 * np.arctan(tangent) * weights
    values = (alpha_reference_npm / phase_slope) * np.tan(angles)
    jacobian = alpha_reference_npm / abs(phase_slope) / np.cos(angles) ** 2
    link = _link_exact(profile, index, phase_slope * values)
    return float(2.0 * np.sum(quadrature_weights * jacobian * np.abs(link) ** 2))


def _link_exact(
    profile: BugliaClosedProfile, index: int, phi: NDArrayFloat
) -> NDArrayComplex:
    """Evaluate the analytic longitudinal integral of the fitted profile."""
    phi = np.atleast_1d(np.asarray(phi, dtype=np.complex128))
    if profile.pole_coeffs is not None and profile.pole_rates is not None:
        rates = np.asarray(profile.pole_rates, dtype=np.float64)[index]
        coeffs = np.asarray(profile.pole_coeffs, dtype=np.float64)[index]
        return np.sum(coeffs[:, None] / (rates[:, None] - 1j * phi[None, :]), axis=0)
    alpha = float(profile.alpha_npm[index])
    alpha_f = float(profile.alpha_f_npm[index])
    alpha_b = float(profile.alpha_b_npm[index])
    delta = profile.delta_hz(index)
    t_f = -float(profile.c_f[index]) * delta / alpha_f
    t_b = -float(profile.c_b[index]) * delta / alpha_b
    t_0 = 1.0 + t_f - t_b * np.exp(-alpha_b * float(profile.length_m))
    rates = np.array([alpha, alpha + alpha_f, alpha - alpha_b], dtype=np.float64)
    coefficients = np.array(
        [t_0, -t_f, t_b * np.exp(-alpha_b * float(profile.length_m))], dtype=np.float64
    )
    denominator = rates[:, None] - 1j * phi[None, :]
    numerator = -np.expm1(-denominator * float(profile.length_m))
    return np.sum(coefficients[:, None] * numerator / denominator, axis=0)


def nli_integral(
    profile: BugliaClosedProfile,
    bandwidths_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    gamma_per_w_m: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n_frequency: int = 96,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Calculate the Eq. (6)-(8) numerical integral as a closed-form reference.

    The same fitted profile and phase conventions are used as in the closed form.
    Profiles are constant within each channel bandwidth; XPM reduces to a single
    frequency integral while SPM retains a double integral.

    Parameters
    ----------
    profile : BugliaClosedProfile
        Fitted profile.
    bandwidths_hz, powers_w, gamma_per_w_m : numpy.ndarray
        Per-channel bandwidths (Hz), launch powers (W), and nonlinear coefficients
        (1/(W m)), in the profile's channel order.
    beta2, beta3 : float
        Dispersion and dispersion slope at the reference frequency, in s²/m and s³/m.
    reference_frequency_hz : float
        Reference frequency for dispersion in Hz.
    n_frequency : int, optional
        Gauss-Legendre nodes per frequency direction.

    Returns
    -------
    tuple of numpy.ndarray
        Per-channel SPM and XPM noise powers in watts.
    """
    frequencies = np.asarray(profile.f_hz, dtype=np.float64)
    powers = np.asarray(powers_w, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    gammas = np.asarray(gamma_per_w_m, dtype=np.float64)
    n_channels = frequencies.size
    nodes, weights = np.polynomial.legendre.leggauss(int(n_frequency))
    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)

    for index in range(n_channels):
        launch = float(powers[index])
        if launch <= 0.0:
            continue
        half_band = 0.5 * float(bandwidths[index])
        phase_i = phase_spm(
            beta2, beta3, float(frequencies[index]), reference_frequency_hz
        )
        offsets = half_band * nodes
        weights_i = half_band * weights
        spm_mu = 0.0
        for offset_1, weight_1 in zip(offsets, weights_i, strict=True):
            link = _link_exact(profile, index, phase_i * offset_1 * offsets)
            spm_mu += weight_1 * float(np.sum(weights_i * np.abs(link) ** 2))
        spm[index] = (
            (16.0 / 27.0)
            * float(gammas[index]) ** 2
            * spm_mu
            / (2.0 * half_band) ** 2
            * launch**3
        )

        total = 0.0
        for interferer in range(n_channels):
            if interferer == index or powers[interferer] <= 0.0:
                continue
            phase_ik = phase_xpm(
                beta2,
                beta3,
                float(frequencies[index]),
                float(frequencies[interferer]),
                reference_frequency_hz,
            )
            inner = _band_average_link_squared(
                profile,
                interferer,
                phase_ik,
                half_band,
                float(profile.alpha_npm[interferer]),
                nodes,
                weights,
            )
            total += (
                (float(powers[interferer]) / launch) ** 2
                * inner
                / float(bandwidths[interferer])
            )
        xpm[index] = (32.0 / 27.0) * float(gammas[index]) ** 2 * total * launch**3
    return spm, xpm
