"""Shared machinery for the two distributed-Raman paper-control models.

The module contains no model registration.  It provides the pieces that both
``buglia2024_ra_gn`` and ``zhang2024_dra_egn`` need and that must behave
identically in the two models:

* the coupled signal/pump/ASE ODE is reused from
  :class:`~network_nlo_eval.models.raman_ode.RamanPaperSpanSolver`, so the
  two DRA controls do not duplicate the routing or the Raman solver;
* the *fitted exponential profile* of the papers.  Buglia 2024 Eq. (2) and
  Zhang 2024 Eq. (1) are both sums of three exponentials in ``z``.  Fitting
  their coefficients to the ODE profile is the step the papers perform with
  an external numerical Raman solver (Buglia Sec. II-A, Zhang Sec. II-A);
* the GN SPM/XPM integral of Eq. (6)-(8) of Buglia, which is also the
  per-span kernel that Zhang Eq. (2)-(4) needs.  The longitudinal integral is
  evaluated analytically from the three-exponential profile; only the
  in-band frequency integral is quadrature, with a ``tan`` substitution that
  resolves the narrow XPM coherence peak.  No waveform model is called.

Everything here is a **documented subset**: the papers' printed closed forms
for the in-band integral (Buglia Eq. (9), (11), (12); Zhang's unpublished
``P`` terms) are not copied term by term because the extracted text is
incomplete.  The exact GN double integral over the channel bands is evaluated
instead, which is the quantity those closed forms approximate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol

import numpy as np
from scipy.optimize import least_squares

from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
from network_nlo_eval.models.base import SpanContext, SpanFiberOutput
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
)
from network_nlo_eval.models.pcfm import (
    _coherent_factor,
    _equivalent_phase_ramp,
    _self_beating_visibility,
)
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.models.route import RouteModel, apply_amplifier_and_roadm, resolve_noise_figure
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "DEFAULT_DRA_NLI_QUADRATURE",
    "DEFAULT_DRA_STEPS",
    "BugliaProfile",
    "DRARamanModelBase",
    "DRASpanKernelBase",
    "ZhangProfile",
    "average_link_squared",
    "fit_buglia_profile",
    "fit_zhang_profile",
    "nli_power_from_profile",
]

#: RK4 steps of the profile ODE.  Two profile samples per step are fitted.
DEFAULT_DRA_STEPS: int = 32

#: Gauss-Legendre nodes of the in-band NLI quadrature (per direction).
DEFAULT_DRA_NLI_QUADRATURE: int = 24

#: Lower bound of the fitted profile decay rates (1/m); keeps the fit stable.
_MIN_RATE_NPM: float = 1.0e-7

#: Upper bound of the fitted profile decay rates (1/m).
_MAX_RATE_NPM: float = 5.0e-3

#: Soft floor of the relative Zhang-profile residual, so a dark channel cannot
#: dominate the least-squares objective.
_ZHANG_RELATIVE_FLOOR: float = 0.05


class _Profile(Protocol):
    """Minimal interface of a fitted normalised profile ``rho(z, f)``."""

    def link(self, frequency_hz: NDArrayFloat, phi: NDArrayFloat) -> NDArrayComplex:
        """Return ``int_0^L rho(z, frequency_hz) exp(1j phi z) dz``."""


@dataclass(frozen=True)
class BugliaProfile:
    """Buglia 2024 Eq. (2) profile fitted to the coupled-ODE solution.

    The paper writes

    .. code-block:: text

        rho(z, f) = exp(-alpha z) [1 - (C_f P_f L_eff(z)
                    + C_b P_b L_tilde(z)) (f - f_hat)] ,

    which is re-grouped here as three exponentials in ``z`` so that the
    longitudinal NLI integral is analytic.  ``C_f P_f`` and ``C_b P_b`` are
    fitted per channel; ``alpha_f`` and ``alpha_b`` are fixed to the
    power-weighted pump attenuation, because the three-exponential form is
    otherwise degenerate on a single-pump, narrow-band profile.

    Attributes
    ----------
        alpha_npm: ``(N,)`` fitted signal attenuation (1/m).
        c_f: ``(N,)`` fitted product ``C_f P_f`` (1/(m*Hz)).
        c_b: ``(N,)`` fitted product ``C_b P_b`` (1/(m*Hz)).
        f_hz: ``(N,)`` channel frequencies of the fit, descending.
        f_hat_hz: pump mean frequency (Hz); WDM mean if there is no pump.
        alpha_f_npm: forward-pump effective attenuation (1/m).
        alpha_b_npm: backward-pump effective attenuation (1/m).
        length_m: span length (m).
        rmse: ``(N,)`` profile fit RMSE (dimensionless).
    """

    alpha_npm: NDArrayFloat
    c_f: NDArrayFloat
    c_b: NDArrayFloat
    f_hz: NDArrayFloat
    f_hat_hz: float
    alpha_f_npm: float
    alpha_b_npm: float
    length_m: float
    rmse: NDArrayFloat

    def _nearest(self, frequency_hz: float) -> int:
        return int(np.argmin(np.abs(self.f_hz - frequency_hz)))

    def _coefficients(self, frequency_hz: float) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return ``(rates, coefficients)`` of the three-exponential form."""
        index = self._nearest(float(frequency_hz))
        delta = float(frequency_hz) - self.f_hat_hz
        t_f = -self.c_f[index] * delta / self.alpha_f_npm
        t_b = -self.c_b[index] * delta / self.alpha_b_npm
        t_0 = 1.0 + t_f - t_b * np.exp(-self.alpha_b_npm * self.length_m)
        rates = np.array(
            [
                self.alpha_npm[index],
                self.alpha_npm[index] + self.alpha_f_npm,
                self.alpha_npm[index] - self.alpha_b_npm,
            ],
            dtype=np.float64,
        )
        coefficients = np.array(
            [t_0, -t_f, t_b * np.exp(-self.alpha_b_npm * self.length_m)],
            dtype=np.float64,
        )
        return rates, coefficients

    def value(self, z_m: NDArrayFloat, frequency_hz: float) -> NDArrayFloat:
        """Evaluate ``rho(z, f)`` at the given longitudinal positions."""
        rates, coefficients = self._coefficients(frequency_hz)
        return np.sum(coefficients[:, None] * np.exp(-rates[:, None] * np.asarray(z_m)[None, :]), axis=0)

    def link(self, frequency_hz: NDArrayFloat, phi: NDArrayFloat) -> NDArrayComplex:
        """Analytic ``int_0^L rho(z, f) exp(1j phi z) dz``."""
        rates, coefficients = self._coefficients(float(frequency_hz))
        return _exponential_link(rates, coefficients, np.asarray(phi, dtype=np.complex128), self.length_m)


@dataclass(frozen=True)
class ZhangProfile:
    """Zhang 2024 Eq. (1) profile fitted to the coupled-ODE solution.

    .. code-block:: text

        rho(z, f) = (1 + T) exp(-alpha z) - T exp(-A z)
                    + D exp(-d (L - z)) .

    ``alpha`` is the fibre attenuation of the channel (not fitted); the four
    remaining coefficients ``T, A, D, d`` are fitted per channel.  The fitted
    curve is renormalised to ``rho(0) = 1``, because the printed Eq. (1) does
    not enforce it (a point the extraction also flags).

    Attributes
    ----------
        alpha_npm: ``(N,)`` channel attenuation (1/m), descending frequencies.
        t_coeff: ``(N,)`` fitted ``T``.
        a_npm: ``(N,)`` fitted ``A`` (1/m).
        d_coeff: ``(N,)`` fitted ``D``.
        d_npm: ``(N,)`` fitted ``d`` (1/m).
        f_hz: ``(N,)`` channel frequencies of the fit, descending.
        length_m: span length (m).
        rmse: ``(N,)`` profile fit RMSE (dimensionless).
    """

    alpha_npm: NDArrayFloat
    t_coeff: NDArrayFloat
    a_npm: NDArrayFloat
    d_coeff: NDArrayFloat
    d_npm: NDArrayFloat
    f_hz: NDArrayFloat
    length_m: float
    rmse: NDArrayFloat

    def _interpolated(self, name: str, frequency_hz: float) -> float:
        values = np.asarray(getattr(self, name), dtype=np.float64)
        order = np.argsort(self.f_hz)
        return float(np.interp(float(frequency_hz), self.f_hz[order], values[order]))

    def _coefficients(self, frequency_hz: float) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return ``(rates, coefficients)`` of the three-exponential form."""
        t_coeff = self._interpolated("t_coeff", frequency_hz)
        a_npm = self._interpolated("a_npm", frequency_hz)
        d_coeff = self._interpolated("d_coeff", frequency_hz)
        d_npm = self._interpolated("d_npm", frequency_hz)
        alpha = self._interpolated("alpha_npm", frequency_hz)
        rates = np.array([alpha, a_npm, -d_npm], dtype=np.float64)
        coefficients = np.array(
            [1.0 + t_coeff, -t_coeff, d_coeff * np.exp(-d_npm * self.length_m)],
            dtype=np.float64,
        )
        rho_0 = float(np.sum(coefficients))
        if abs(rho_0) < 1.0e-12:
            rho_0 = 1.0
        return rates, coefficients / rho_0

    def value(self, z_m: NDArrayFloat, frequency_hz: float) -> NDArrayFloat:
        """Evaluate ``rho(z, f)`` at the given longitudinal positions."""
        rates, coefficients = self._coefficients(frequency_hz)
        return np.sum(coefficients[:, None] * np.exp(-rates[:, None] * np.asarray(z_m)[None, :]), axis=0)

    def link(self, frequency_hz: NDArrayFloat, phi: NDArrayFloat) -> NDArrayComplex:
        """Analytic ``int_0^L rho(z, f) exp(1j phi z) dz``."""
        rates, coefficients = self._coefficients(float(frequency_hz))
        return _exponential_link(rates, coefficients, np.asarray(phi, dtype=np.complex128), self.length_m)


def _exponential_link(
    rates: NDArrayFloat,
    coefficients: NDArrayFloat,
    phi: NDArrayComplex,
    length_m: float,
) -> NDArrayComplex:
    r"""Compute ``sum_m c_m (1 - exp(-(a_m - j phi) L)) / (a_m - j phi)``.

    ``a_m`` may be negative (the growing backward-pump term of Zhang Eq. (1));
    the rewritten ``expm1`` form keeps that case numerically stable.
    """
    denominator = rates[:, None].astype(np.complex128) - 1j * phi[None, :]
    numerator = -np.expm1(-denominator * length_m)
    return np.sum(coefficients[:, None] * numerator / denominator, axis=0)


def _profile_grid(profile: NDArrayFloat, length_m: float, n_steps: int) -> NDArrayFloat:
    """Longitudinal samples of the RK4 profile: step midpoint then endpoint."""
    step = length_m / n_steps
    positions = np.empty(2 * n_steps, dtype=np.float64)
    for index in range(n_steps):
        positions[2 * index] = step * (index + 0.5)
        positions[2 * index + 1] = step * (index + 1.0)
    return positions


def fit_buglia_profile(
    profile_norm: NDArrayFloat,
    length_m: float,
    frequencies_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    f_hat_hz: float,
    alpha_f_npm: float,
    alpha_b_npm: float,
    n_steps: int,
) -> BugliaProfile:
    """Fit Buglia Eq. (2) to a normalised ODE profile, channel by channel.

    Args:
        profile_norm: ``(2*n_steps, N)`` normalised profile, descending frequencies.
        length_m: Span length (m).
        frequencies_hz: ``(N,)`` channel frequencies (Hz), same order.
        alpha_npm: ``(N,)`` channel attenuation (1/m).
        f_hat_hz: Pump mean frequency (Hz).
        alpha_f_npm: Forward-pump effective attenuation (1/m).
        alpha_b_npm: Backward-pump effective attenuation (1/m).
        n_steps: RK4 steps used to produce ``profile_norm``.

    Returns
    -------
        Fitted profile with per-channel RMSE.
    """
    positions = _profile_grid(profile_norm, length_m, n_steps)
    n_channels = frequencies_hz.size
    alpha_fit = np.empty(n_channels, dtype=np.float64)
    c_f = np.zeros(n_channels, dtype=np.float64)
    c_b = np.zeros(n_channels, dtype=np.float64)
    rmse = np.empty(n_channels, dtype=np.float64)
    alpha_f = max(float(alpha_f_npm), _MIN_RATE_NPM)
    alpha_b = max(float(alpha_b_npm), _MIN_RATE_NPM)

    for index in range(n_channels):
        target = profile_norm[:, index]
        delta = float(frequencies_hz[index]) - float(f_hat_hz)
        leff = -np.expm1(-alpha_f * positions) / alpha_f
        ltilde = (np.exp(-alpha_b * (length_m - positions)) - np.exp(-alpha_b * length_m)) / alpha_b

        def residual(
            parameters: NDArrayFloat,
            positions: NDArrayFloat = positions,
            leff: NDArrayFloat = leff,
            ltilde: NDArrayFloat = ltilde,
            delta: float = delta,
            target: NDArrayFloat = target,
        ) -> NDArrayFloat:
            alpha, cf, cb = parameters
            model = np.exp(-alpha * positions) * (1.0 - (cf * leff + cb * ltilde) * delta)
            return model - target

        start = np.array([alpha_npm[index], 0.0, 0.0], dtype=np.float64)
        bounds = (
            np.array([_MIN_RATE_NPM, -1.0e-16, -1.0e-16]),
            np.array([_MAX_RATE_NPM, 1.0e-16, 1.0e-16]),
        )
        try:
            solution = least_squares(
                residual,
                start,
                bounds=bounds,
                x_scale=np.array([1.0e-5, 1.0e-19, 1.0e-19]),
                max_nfev=600,
            )
            alpha, cf, cb = solution.x
        except (ValueError, np.linalg.LinAlgError):
            alpha, cf, cb = alpha_npm[index], 0.0, 0.0

        # Linear fallback for alpha: the remaining two coefficients are linear.
        model_without_alpha = np.exp(-alpha * positions)
        basis = -model_without_alpha[:, None] * delta * np.stack([leff, ltilde], axis=1)
        try:
            linear, *_ = np.linalg.lstsq(basis, target - model_without_alpha, rcond=None)
        except np.linalg.LinAlgError:
            linear = np.zeros(2, dtype=np.float64)
        fallback = np.array([model_without_alpha + basis @ linear])
        fallback_rmse = float(np.sqrt(np.mean((fallback[0] - target) ** 2)))
        fitted_rmse = float(np.sqrt(np.mean(residual(np.array([alpha, cf, cb])) ** 2)))
        if fallback_rmse < fitted_rmse:
            cf, cb = float(linear[0]), float(linear[1])
            fitted_rmse = fallback_rmse

        alpha_fit[index] = float(alpha)
        c_f[index] = float(cf)
        c_b[index] = float(cb)
        rmse[index] = fitted_rmse

    return BugliaProfile(
        alpha_npm=alpha_fit,
        c_f=c_f,
        c_b=c_b,
        f_hz=np.asarray(frequencies_hz, dtype=np.float64),
        f_hat_hz=float(f_hat_hz),
        alpha_f_npm=alpha_f,
        alpha_b_npm=alpha_b,
        length_m=float(length_m),
        rmse=rmse,
    )


def fit_zhang_profile(
    profile_norm: NDArrayFloat,
    length_m: float,
    frequencies_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    n_steps: int,
) -> ZhangProfile:
    """Fit Zhang Eq. (1) to a normalised ODE profile, channel by channel.

    Args:
        profile_norm: ``(2*n_steps, N)`` normalised profile, descending frequencies.
        length_m: Span length (m).
        frequencies_hz: ``(N,)`` channel frequencies (Hz), same order.
        alpha_npm: ``(N,)`` channel attenuation (1/m).
        n_steps: RK4 steps used to produce ``profile_norm``.

    Returns
    -------
        Fitted profile with per-channel RMSE.
    """
    positions = _profile_grid(profile_norm, length_m, n_steps)
    n_channels = frequencies_hz.size
    t_coeff = np.zeros(n_channels, dtype=np.float64)
    a_npm = np.full(n_channels, 2.0e-5, dtype=np.float64)
    d_coeff = np.zeros(n_channels, dtype=np.float64)
    d_npm = np.full(n_channels, 4.0e-5, dtype=np.float64)
    rmse = np.empty(n_channels, dtype=np.float64)

    for index in range(n_channels):
        target = profile_norm[:, index]
        alpha = max(float(alpha_npm[index]), _MIN_RATE_NPM)

        def model(
            parameters: NDArrayFloat,
            positions: NDArrayFloat = positions,
            alpha: float = alpha,
        ) -> NDArrayFloat:
            t_value, a_value, d_value, d_rate = parameters
            curve = (1.0 + t_value) * np.exp(-alpha * positions)
            curve -= t_value * np.exp(-a_value * positions)
            curve += d_value * np.exp(-d_rate * (length_m - positions))
            # The first ODE sample is at z = h/2, not at z = 0.  Normalise by
            # the analytic rho(0) so a fast-decaying fit cannot exploit the
            # half-step offset.
            rho_zero = 1.0 + d_value * np.exp(-d_rate * length_m)
            return curve / rho_zero

        def residual(
            parameters: NDArrayFloat,
            target: NDArrayFloat = target,
        ) -> NDArrayFloat:
            return (model(parameters) - target) / np.maximum(target, _ZHANG_RELATIVE_FLOOR)

        start = np.array([0.2, 2.0 * alpha, 0.0, 4.0e-5], dtype=np.float64)
        bounds = (
            np.array([-5.0, _MIN_RATE_NPM, -5.0, _MIN_RATE_NPM]),
            np.array([5.0, _MAX_RATE_NPM, 5.0, _MAX_RATE_NPM]),
        )
        try:
            solution = least_squares(
                residual,
                start,
                bounds=bounds,
                x_scale=np.array([1.0, 1.0e-5, 1.0, 1.0e-5]),
                max_nfev=600,
            )
            parameters = solution.x
            fitted_rmse = float(np.sqrt(np.mean((model(parameters) - target) ** 2)))
        except (ValueError, np.linalg.LinAlgError):
            parameters = np.array([0.0, alpha, 0.0, alpha], dtype=np.float64)
            fitted_rmse = float(np.sqrt(np.mean((model(parameters) - target) ** 2)))

        t_coeff[index], a_npm[index], d_coeff[index], d_npm[index] = (
            float(parameters[0]),
            float(parameters[1]),
            float(parameters[2]),
            float(parameters[3]),
        )
        rmse[index] = fitted_rmse

    return ZhangProfile(
        alpha_npm=np.asarray(alpha_npm, dtype=np.float64),
        t_coeff=t_coeff,
        a_npm=a_npm,
        d_coeff=d_coeff,
        d_npm=d_npm,
        f_hz=np.asarray(frequencies_hz, dtype=np.float64),
        length_m=float(length_m),
        rmse=rmse,
    )


def average_link_squared(
    profile: _Profile,
    frequency_hz: float,
    coefficient: float,
    half_band_hz: float,
    alpha_ref_npm: float,
    nodes: NDArrayFloat,
    weights: NDArrayFloat,
) -> float:
    """Average ``|int rho exp(1j coefficient v z) dz|**2`` over ``v in [-U, U]``.

    The XPM coherence peak is narrow (width ``alpha / coefficient``), so the
    average is computed with ``v = (alpha / coefficient) tan(s)``; a plain
    Gauss-Legendre rule over the full channel band would miss the peak for
    typical 75 GHz-spaced channels.

    Args:
        profile: Fitted profile.
        frequency_hz: Frequency at which ``rho`` is evaluated.
        coefficient: Phase slope ``d phi / d v`` (1/m per Hz).
        half_band_hz: ``U``, half the COI bandwidth (Hz).
        alpha_ref_npm: Attenuation used by the substitution (1/m).
        nodes: Gauss-Legendre nodes on ``[-1, 1]``.
        weights: Gauss-Legendre weights.

    Returns
    -------
        Band-averaged squared link function (m²).
    """
    alpha = max(float(alpha_ref_npm), _MIN_RATE_NPM)
    link_zero = profile.link(frequency_hz, np.zeros(1, dtype=np.complex128))[0]
    threshold = 1.0e-12 * alpha
    if abs(coefficient) * half_band_hz <= threshold:
        return float(abs(link_zero) ** 2)
    tangent = abs(coefficient) * half_band_hz / alpha
    angles = 0.5 * np.arctan(tangent) * (nodes + 1.0)
    quadrature_weights = 0.5 * np.arctan(tangent) * weights
    values = (alpha / coefficient) * np.tan(angles)
    jacobian = (alpha / abs(coefficient)) / np.cos(angles) ** 2
    link = profile.link(frequency_hz, coefficient * values)
    return float(np.sum(quadrature_weights * jacobian * np.abs(link) ** 2) / (2.0 * half_band_hz))


def nli_power_from_profile(
    profile: _Profile,
    frequencies_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n2_m2_w: float,
    n_quadrature: int = DEFAULT_DRA_NLI_QUADRATURE,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Compute per-span SPM and XPM noise powers from a fitted profile.

    The kernel is the GN double integral of Buglia Eq. (6)-(8): for XPM the
    inner longitudinal integral is Eq. (8), for SPM Eq. (7) halves the
    self-channel term.  The outer frequency integrals are evaluated with the
    analytic link function and the ``tan`` substitution above.

    Args:
        profile: Fitted profile.
        frequencies_hz: ``(N,)`` channel frequencies (Hz).
        powers_w: ``(N,)`` launch powers (W).
        bandwidths_hz: ``(N,)`` channel bandwidths (Hz).
        alpha_npm: ``(N,)`` attenuation (1/m).
        a_eff_m2: ``(N,)`` effective areas (m²).
        beta2: GVD parameter (s²/m).
        beta3: dispersion slope (s³/m).
        reference_frequency_hz: Reference frequency of ``beta2``/``beta3`` (Hz).
        n2_m2_w: Nonlinear index (m²/W).
        n_quadrature: Gauss-Legendre nodes per direction.

    Returns
    -------
        ``(spm_w, xpm_w)``, both ``(N,)`` in the same order as the inputs.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    powers = np.asarray(powers_w, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    alpha = np.asarray(alpha_npm, dtype=np.float64)
    areas = np.asarray(a_eff_m2, dtype=np.float64)
    n_channels = frequencies.size
    if powers.shape != (n_channels,) or bandwidths.shape != (n_channels,):
        raise ValueError("powers_w and bandwidths_hz must contain one value per channel.")

    nodes, weights = np.polynomial.legendre.leggauss(int(n_quadrature))
    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)

    for index in range(n_channels):
        launch = float(powers[index])
        if launch <= 0.0:
            continue
        half_band = 0.5 * float(bandwidths[index])
        offsets = half_band * nodes
        offset_weights = half_band * weights

        # SPM: integrate both in-band offsets; the phase slope is 4 pi^2 beta2.
        spm_total = 0.0
        for offset, weight in zip(offsets, offset_weights, strict=True):
            beta2_eff = beta2 + np.pi * beta3 * (2.0 * frequencies[index] + offset - 2.0 * reference_frequency_hz)
            coefficient = 2.0 * np.pi**2 * beta2_eff * offset
            spm_total += weight * average_link_squared(
                profile,
                float(frequencies[index] + offset),
                coefficient,
                half_band,
                float(alpha[index]),
                nodes,
                weights,
            )
        gamma_sci = (2.0 * np.pi * frequencies[index] / C_LIGHT) * (n2_m2_w / areas[index])
        eta_spm = (16.0 / 27.0) * gamma_sci**2 * (spm_total / (2.0 * half_band))
        spm[index] = eta_spm * launch**3

        # XPM: the in-band offset is on the COI, the interferer band is
        # integrated in closed form (its profile varies slowly).
        eta_xpm = 0.0
        for interferer in range(n_channels):
            if interferer == index or powers[interferer] <= 0.0:
                continue
            beta2_eff = beta2 + np.pi * beta3 * (
                frequencies[index] + frequencies[interferer] - 2.0 * reference_frequency_hz
            )
            coefficient = 2.0 * np.pi**2 * (frequencies[interferer] - frequencies[index]) * beta2_eff
            link_squared = average_link_squared(
                profile,
                float(frequencies[interferer]),
                coefficient,
                half_band,
                float(alpha[index]),
                nodes,
                weights,
            )
            gamma_xci = (2.0 * np.pi * frequencies[index] / C_LIGHT) * (
                2.0 * n2_m2_w / (areas[index] + areas[interferer])
            )
            eta_xpm += (
                (32.0 / 27.0)
                * gamma_xci**2
                * (powers[interferer] / launch) ** 2
                * (2.0 * half_band / float(bandwidths[interferer]))
                * link_squared
            )
        xpm[index] = eta_xpm * launch**3

    return spm, xpm


class DRASpanKernelBase:
    """Common single-span kernel of the two DRA paper-control models.

    It owns the coupled-ODE solver, the profile fit cache, the Raman ASE, and
    the NLI quadrature.  Subclasses choose the profile fit and the inter-span
    convention.

    Args:
        f_abs_hz: ``(N,)`` channel centre frequencies (Hz).
        b_hz: ``(N,)`` channel bandwidths (Hz).
        fiber_config: Span fibre parameters.
        pumps: Raman pumps of the span.
        gain_model: Raman gain spectrum model.
        raman_ase: Expose the newly generated Raman ASE.
        n_steps: RK4 steps of the coupled ODE.
        n_iterations: Fixed-point iterations for backward pumps.
        include_isrs: Include channel-channel ISRS in the ODE.
        n_quadrature: Gauss-Legendre nodes of the NLI in-band integral.
    """

    #: Human-readable profile kind, used in diagnostics.
    profile_kind: ClassVar[str] = "generic"

    #: 是否返回跨间复振幅。``True`` 时路由层按 ``|sum a_k|**2`` 相干累加；
    #: ``False`` 时只返回功率，跨间按功率相加（但模型仍可读 ``SpanContext``，
    #: 例如 Buglia Eq. (4) 的 ``n^epsilon`` 需要知道总跨数）。
    emits_nli_amplitudes: ClassVar[bool] = True

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        *,
        pumps: Sequence[RamanPumpConfig],
        gain_model: RamanGainModel,
        raman_ase: bool = True,
        n_steps: int = DEFAULT_DRA_STEPS,
        n_iterations: int = 24,
        include_isrs: bool = True,
        n_quadrature: int = DEFAULT_DRA_NLI_QUADRATURE,
    ) -> None:
        if n_steps < 4:
            raise ValueError("n_steps must be >= 4.")
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")
        if n_quadrature < 8:
            raise ValueError("n_quadrature must be >= 8.")

        f_abs = np.asarray(f_abs_hz, dtype=np.float64)
        bandwidths = np.asarray(b_hz, dtype=np.float64)
        if f_abs.shape != bandwidths.shape:
            raise ValueError("b_hz must contain one value per channel.")
        self.order = np.argsort(-f_abs, kind="stable")
        self.inverse_order = np.empty(self.order.size, dtype=np.int64)
        self.inverse_order[self.order] = np.arange(self.order.size)
        self.f_abs_hz = np.ascontiguousarray(f_abs[self.order])
        self.b_hz = np.ascontiguousarray(bandwidths[self.order])
        self.n_nodes = int(self.f_abs_hz.size)
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
            alpha = fiber_config.attenuation_db_km_ref * np.log(10.0) / 10.0 / 1e3
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
        self.reference_frequency_hz = C_LIGHT / lambda_ref_m

        self._pumps = [pump for pump in pumps if float(pump.power_w) > 0.0]
        self._gain_model = gain_model
        self._raman_ase_enabled = bool(raman_ase)
        self._n_steps = int(n_steps)
        self._n_iterations = int(n_iterations)
        self._include_isrs = bool(include_isrs)
        self._n_quadrature = int(n_quadrature)
        self._pump_alpha_npm = np.asarray(
            [float(pump.attenuation_db_per_km) * np.log(10.0) / 10.0 / 1e3 for pump in self._pumps],
            dtype=np.float64,
        )
        pump_frequencies = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in self._pumps],
            dtype=np.float64,
        )
        self._pump_a_eff_m2 = (
            _calc_aeff_dynamic(
                C_LIGHT / pump_frequencies if pump_frequencies.size else pump_frequencies,
                fiber_config.effective_area_um2_ref,
                fiber_config.aeff_slope_um2_nm,
                fiber_config.reference_wavelength_nm,
            )
            if pump_frequencies.size
            else np.zeros(0, dtype=np.float64)
        )
        self._active = any(float(pump.power_w) > 0.0 for pump in self._pumps)

        self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_pump_power_out_w: NDArrayFloat | None = None
        self._solver_cache: dict[int, RamanPaperSpanSolver] = {}
        self._profile_cache: dict[tuple[int, bytes], object] = {}
        self._ramp_cache: dict[tuple[int, float], NDArrayFloat] = {}
        self.last_profile_rmse: NDArrayFloat | None = None

        # Inter-span coherence constants.  The phase ramp is the NLI-weighted
        # visibility ramp of the repository's GN reference, not a raw
        # beta2*B^2 phase: the raw band-edge phase is far too large and would
        # make every span incoherent.
        self.beta2_abs_s2_m = abs(self.beta2)
        self.alpha_visibility_npm = float(np.mean(self.alpha_npm))

    @property
    def active(self) -> bool:
        """Whether at least one pump carries non-zero power."""
        return self._active

    def set_input_ase_w(self, ase_w: NDArrayFloat | None) -> None:
        """Set the signal ASE entering the span (model order)."""
        if ase_w is None:
            self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
            return
        values = np.asarray(ase_w, dtype=np.float64)
        if values.shape != (self.n_nodes,):
            raise ValueError("ase_w must contain one value per channel.")
        self._input_ase_internal = np.ascontiguousarray(values[self.order])

    @property
    def last_raman_ase_w(self) -> NDArrayFloat:
        """Raman ASE generated by the last span (model order)."""
        return np.ascontiguousarray(self._last_raman_ase_internal[self.inverse_order])

    @property
    def last_pump_power_out_w(self) -> NDArrayFloat | None:
        """Pump power at its output end, or ``None`` before the first solve."""
        return self._last_pump_power_out_w

    def reset_state(self) -> None:
        """Drop the per-path caches so a call does not depend on call history."""
        self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_pump_power_out_w = None
        self._profile_cache.clear()
        self._ramp_cache.clear()
        self.last_profile_rmse = None

    def _solver_for(self, span_length_m: float) -> RamanPaperSpanSolver:
        key = int(round(span_length_m * 1e6))
        solver = self._solver_cache.get(key)
        if solver is None:
            solver = RamanPaperSpanSolver(
                self.f_abs_hz,
                self.b_hz,
                self._pumps,
                span_length_m,
                self.alpha_npm,
                self.a_eff_m2,
                self._pump_alpha_npm,
                self._pump_a_eff_m2,
                self._gain_model,
                n_steps=self._n_steps,
                n_iterations=self._n_iterations,
                include_isrs=self._include_isrs,
                include_ase=self._raman_ase_enabled,
            )
            self._solver_cache[key] = solver
        return solver

    def _fit_profile(self, profile_norm: NDArrayFloat, span_length_m: float) -> _Profile:
        raise NotImplementedError

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: _Profile,
        span_context: SpanContext | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """本跨 SPM/XPM 噪声功率 (W)，模型可替换的口径。

        默认实现是 Buglia Eq. (6)-(8) 的 GN 双重积分（``nli_power_from_profile``），
        即两个 DRA 对照模型最初共用的"积分基准"路径。子类可以换成论文的闭式
        求积，或换成带跨间相位的相干累加。

        Args:
            power_internal: ``(N,)`` 本跨入纤功率 (W)，降序。
            profile: 本跨拟合剖面。
            span_context: 本跨在链路中的位置；只有声明了
                ``supports_span_context`` 的模型才会收到非 ``None``。

        Returns
        -------
            ``(spm_w, xpm_w)``，均为 ``(N,)``。
        """
        del span_context
        return nli_power_from_profile(
            profile,
            self.f_abs_hz,
            power_internal,
            self.b_hz,
            self.alpha_npm,
            self.a_eff_m2,
            self.beta2,
            self.beta3,
            self.reference_frequency_hz,
            self.n2_m2_w,
            n_quadrature=self._n_quadrature,
        )

    def _profile_for(
        self,
        power_internal: NDArrayFloat,
        span_length_m: float,
        profile_norm: NDArrayFloat,
    ) -> _Profile:
        scale = float(np.max(power_internal)) if power_internal.size else 1.0
        scale = scale if scale > 0.0 else 1.0
        quantized = np.round(np.asarray(power_internal) / scale * 1.0e8).astype(np.int64)
        key = (int(round(span_length_m * 1e6)), quantized.tobytes())
        cached = self._profile_cache.get(key)
        if cached is not None:
            return cached  # type: ignore[return-value]
        profile = self._fit_profile(profile_norm, span_length_m)
        if len(self._profile_cache) >= 8:
            self._profile_cache.clear()
        self._profile_cache[key] = profile
        self.last_profile_rmse = getattr(profile, "rmse", None)
        return profile

    def _span_phase_factor(self, span_context: SpanContext) -> NDArrayComplex:
        """Inter-span phase of the NLI field.

        The paper writes ``exp(1j sum_q phi_q L_q)`` but does not publish
        ``phi_q``.  The phase used here is the NLI-weighted equivalent ramp of
        the repository's GN reference, which reproduces the coherent factor
        ``R(N) = N + 2 sum_k (N-k) V(kL)`` on equal spans.  It is a labelled
        proxy for the paper's phase, not the paper's own number.
        """
        n_spans = max(1, int(span_context.total_spans))
        span_length = float(span_context.span_length_m)
        key = (n_spans, span_length)
        ramp = self._ramp_cache.get(key)
        if ramp is None:
            distances = np.arange(1, n_spans, dtype=np.float64) * span_length
            if n_spans <= 1:
                visibility = np.zeros((0, self.n_nodes), dtype=np.float64)
            else:
                visibility = _self_beating_visibility(
                    self.beta2_abs_s2_m,
                    self.alpha_visibility_npm,
                    self.b_hz,
                    span_length,
                    distances,
                )
            factor = _coherent_factor(n_spans, visibility)
            ramp = _equivalent_phase_ramp(n_spans, factor)
            self._ramp_cache[key] = ramp
        scale = float(span_context.cumulative_length_m) / span_length if span_length > 0 else 0.0
        return np.exp(1j * ramp * scale)

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        *,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Evaluate one span, including the coupled Raman ODE and the NLI.

        Args:
            power_in_w: ``(N,)`` launch power per channel (W), model order.
            span_length_m: Span length (m).
            n_effective_spans: Kept for protocol compatibility; the route layer
                accumulates the spans one by one.
            span_context: Position of the span in the link.  When given and the
                link has more than one span, complex NLI amplitudes are returned.

        Returns
        -------
            Span output power and SPM/XPM noise, in model order.

        Raises
        ------
            ValueError: Input shape mismatch.
        """
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != (self.n_nodes,):
            raise ValueError("power_in_w must contain one value per channel.")
        power_internal = np.ascontiguousarray(power[self.order])

        solver = self._solver_for(span_length_m)
        result = solver.solve(power_internal, self._input_ase_internal)
        profile = self._profile_for(power_internal, span_length_m, result.profile_norm)
        spm, xpm = self._nli_span(power_internal, profile, span_context)
        self._last_raman_ase_internal = (
            np.ascontiguousarray(result.raman_ase_w)
            if self._raman_ase_enabled
            else np.zeros(self.n_nodes, dtype=np.float64)
        )
        self._last_pump_power_out_w = np.ascontiguousarray(result.pump_power_out_w)

        output = SpanFiberOutput(
            power_out_w=np.ascontiguousarray(result.power_out_w[self.inverse_order]),
            spm_noise_w=np.ascontiguousarray(spm[self.inverse_order]),
            xpm_noise_w=np.ascontiguousarray(xpm[self.inverse_order]),
        )
        total_spans = 1 if span_context is None else max(1, int(span_context.total_spans))
        if not self.emits_nli_amplitudes or span_context is None or total_spans <= 1:
            return output
        factor = self._span_phase_factor(span_context)
        amplitude_spm = np.sqrt(spm) * factor
        amplitude_xpm = np.sqrt(xpm) * factor
        return SpanFiberOutput(
            output.power_out_w,
            output.spm_noise_w,
            output.xpm_noise_w,
            nli_amplitude_spm=np.ascontiguousarray(amplitude_spm[self.inverse_order]),
            nli_amplitude_xpm=np.ascontiguousarray(amplitude_xpm[self.inverse_order]),
        )


class DRARamanModelBase(RouteModel):
    """Shared route-level model for the two DRA paper controls.

    The class implements the same cross-span ASE bookkeeping as
    :class:`~network_nlo_eval.models.hybrid_raman.HybridRamanPaperModel`: only
    the Raman ASE *newly generated* by the span is handed to the EDFA, while
    the route layer transports the accumulated ASE with the signal transfer
    factor.  Amps, ROADMs and SNR are the shared route implementation.

    Args:
        grid: Spectrum grid (legacy or flexible).
        ref_fiber_config: Reference fibre.
        symbol_rate_hz: Symbol rate (Hz).
        connections: Optional per-connection grid.
        raman_pumps: Pumps applied to every span; ``None`` uses the fibre config.
        raman_gain_model: ``"gaussian"`` or ``"linear"``.
        raman_ase: Include the Raman ASE before the EDFA.
        raman_temperature_k: Fibre temperature (K).
        raman_peak_gain_coefficient_m_per_w: Peak local Raman gain rate.
        include_isrs: Include channel-channel ISRS in the ODE.
        fixed_point_iterations: Maximum backward-pump fixed-point iterations.
        n_steps: RK4 steps of the coupled ODE.
        n_quadrature: Gauss-Legendre nodes of the NLI in-band integral.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.
    """

    _kernel_class: ClassVar[type[DRASpanKernelBase]] = DRASpanKernelBase
    supports_span_context: ClassVar[bool] = False

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections=None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str = "gaussian",
        raman_ase: bool = True,
        raman_temperature_k: float = 300.0,
        raman_peak_gain_coefficient_m_per_w: float = 6e-14,
        include_isrs: bool = True,
        fixed_point_iterations: int = 24,
        n_steps: int = DEFAULT_DRA_STEPS,
        n_quadrature: int = DEFAULT_DRA_NLI_QUADRATURE,
        **route_options: object,
    ) -> None:
        if fixed_point_iterations < 1:
            raise ValueError("fixed_point_iterations must be >= 1.")
        if n_steps < 4:
            raise ValueError("n_steps must be >= 4.")
        if n_quadrature < 8:
            raise ValueError("n_quadrature must be >= 8.")
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            **route_options,
        )
        self._paper_pumps = None if raman_pumps is None else list(raman_pumps)
        self._gain_model = RamanGainModel(
            gain_model=raman_gain_model,  # type: ignore[arg-type]
            temperature_k=raman_temperature_k,
            peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
        )
        self._raman_ase = bool(raman_ase)
        self._include_isrs = bool(include_isrs)
        self._fixed_point_iterations = int(fixed_point_iterations)
        self._n_steps = int(n_steps)
        self._n_quadrature = int(n_quadrature)
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)

    def _resolved_pumps(self, fiber_config: FiberSpanConfig) -> list[RamanPumpConfig]:
        return list(self._paper_pumps) if self._paper_pumps is not None else list(fiber_config.raman_pumps)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> DRASpanKernelBase:
        """Build the span kernel of this fibre."""
        return self._kernel_class(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            pumps=self._resolved_pumps(fiber_config),
            gain_model=self._gain_model,
            raman_ase=self._raman_ase,
            n_steps=self._n_steps,
            n_iterations=self._fixed_point_iterations,
            include_isrs=self._include_isrs,
            n_quadrature=self._n_quadrature,
        )

    def evaluate_path(self, optical_path, power_in_w):
        """Evaluate a path with fresh per-path kernel state."""
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)
        for kernel in self._kernel_cache.values():
            reset = getattr(kernel, "reset_state", None)
            if reset is not None:
                reset()
        return super().evaluate_path(optical_path, power_in_w)

    def span_step(
        self,
        fiber_config,
        edfa_config,
        power_in_w,
        span_length_m,
        span_context: SpanContext | None = None,
    ):
        """Single span: DRA kernel, then the shared amplifier handling."""
        kernel = self.kernel_for(fiber_config)
        kernel.set_input_ase_w(self._ase_in_w)
        fiber_output = kernel.evaluate(
            power_in_w,
            span_length_m,
            1,
            span_context=span_context,
        )
        raman_ase_w = kernel.last_raman_ase_w if self._raman_ase else None
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            input_ase_w=raman_ase_w,
        )
        self._ase_in_w = np.array(element.ase_noise_w, dtype=np.float64, copy=True)
        return element
