"""JIT-accelerated physical-layer computation kernels.

The kernels operate on NumPy arrays and primitive values so Numba can
compile them without depending on Python objects.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT, H_PLANCK, NP_TO_DB, PI
from network_nlo_eval.core.types import NDArrayFloat

_F = TypeVar("_F", bound=Callable[..., Any])

try:
    from numba import njit
except ImportError:  # Keep statistical models usable without JIT.

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:
        """Return an identity decorator when optional Numba is unavailable."""
        del args, kwargs

        def decorate(function: _F) -> _F:
            return function

        return decorate
# =============================================================================
# 0. Common conversions (JIT optimized)
# =============================================================================


@njit(cache=True)  # type: ignore[untyped-decorator]
def _dbm_to_w(p_dbm: float) -> float:
    """Convert dBm to watts."""
    return 1e-3 * 10 ** (p_dbm / 10.0)


@njit(cache=True)  # type: ignore[untyped-decorator]
def _w_to_dbm(p_w: float) -> float:
    """Convert watts to dBm, returning negative infinity for non-positive power."""
    if p_w <= 0:
        return -np.inf
    return 10.0 * np.log10(p_w * 1e3)  # type: ignore[no-any-return]


@njit(cache=True)  # type: ignore[untyped-decorator]
def _lin_to_db(value: float) -> float:
    """Convert a linear ratio to dB, returning negative infinity if non-positive."""
    if value <= 0:
        return -np.inf
    return 10.0 * np.log10(value)  # type: ignore[no-any-return]


@njit(cache=True)  # type: ignore[untyped-decorator]
def _db_to_lin(value_db: float) -> float:
    """Convert a dB ratio to a linear ratio."""
    return 10 ** (value_db / 10.0)


@njit(cache=True)  # type: ignore[untyped-decorator]
def _dbkm_to_power_npm(alpha_dbkm: float) -> float:
    """Convert power attenuation in dB/km to nepers per meter."""
    return alpha_dbkm / (NP_TO_DB * 1000)


# =============================================================================
# 1. Fiber parameter calculations (JIT optimized)
# =============================================================================


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_fiber_attenuation_npm(
    lambdas_m: NDArrayFloat,
    lambda_ref_nm: float,
    alpha_2: float,
    alpha_1: float,
    alpha_0: float,
) -> NDArrayFloat:
    """Calculate wavelength-dependent fiber power attenuation.

    Parameters
    ----------
    lambdas_m : numpy.ndarray
        Wavelengths in meters.
    lambda_ref_nm : float
        Reference wavelength in nanometers.
    alpha_2, alpha_1, alpha_0 : float
        Quadratic, linear, and constant attenuation coefficients.

    Returns
    -------
    numpy.ndarray
        Power attenuation at each wavelength in nepers per meter.
    """
    lambda_nm = lambdas_m * 1e9
    alpha_dbkm = (
        alpha_2 * (lambda_nm - lambda_ref_nm) ** 2
        + alpha_1 * (lambda_nm - lambda_ref_nm)
        + alpha_0
    )
    return alpha_dbkm / (NP_TO_DB * 1000)


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_aeff_dynamic(
    lambdas_m: NDArrayFloat,
    a_eff_ref_um2: float,
    slope_a_eff_um2_nm: float,
    lambda_ref_nm: float,
) -> NDArrayFloat:
    """Calculate wavelength-dependent effective mode area.

    Parameters
    ----------
    lambdas_m : numpy.ndarray
        Wavelengths in meters.
    a_eff_ref_um2 : float
        Effective area at the reference wavelength in square micrometers.
    slope_a_eff_um2_nm : float
        Effective-area slope in square micrometers per nanometer.
    lambda_ref_nm : float
        Reference wavelength in nanometers.

    Returns
    -------
    numpy.ndarray
        Effective area at each wavelength in square meters.
    """
    return (
        a_eff_ref_um2 + slope_a_eff_um2_nm * (lambdas_m * 1e9 - lambda_ref_nm)
    ) * 1e-12  # Convert to square meters.


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_gamma_coeff_scalar(
    lambda_i_m: float,
    lambda_l_m: float,  # noqa: ARG001 - retained for kernel API compatibility
    n2: float,
    a_eff_i_m2: float,
    a_eff_l_m2: float,
) -> float:
    """Calculate the nonlinear coefficient for two interacting channels.

    Parameters
    ----------
    lambda_i_m, lambda_l_m : float
        Channel wavelengths in meters.
    n2 : float
        Nonlinear refractive index in square meters per watt.
    a_eff_i_m2, a_eff_l_m2 : float
        Effective mode areas in square meters.

    Returns
    -------
    float
        Nonlinear coefficient in inverse watt-meters.
    """
    return (2 * PI / lambda_i_m) * (2 * n2 / (a_eff_i_m2 + a_eff_l_m2))


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_beta2_from_d(d_ps_nm_km: float, lambda_m: float) -> float:
    """Convert dispersion parameter D to beta2.

    Parameters
    ----------
    d_ps_nm_km : float
        Dispersion parameter in ps/(nm km).
    lambda_m : float
        Wavelength in meters.

    Returns
    -------
    float
        Beta2 in seconds squared per meter.
    """
    return -d_ps_nm_km * 1e-6 * lambda_m**2 / (2 * PI * C_LIGHT)


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_beta3_from_s_d(
    s_ps_nm2_km: float, d_ps_nm_km: float, lambda_m: float
) -> float:
    """Convert dispersion slope S and parameter D to beta3.

    Parameters
    ----------
    s_ps_nm2_km : float
        Dispersion slope in ps/(nm² km).
    d_ps_nm_km : float
        Dispersion parameter in ps/(nm km).
    lambda_m : float
        Wavelength in meters.

    Returns
    -------
    float
        Beta3 in seconds cubed per meter.
    """
    return (lambda_m**2 / (2 * PI * C_LIGHT) ** 2) * (
        lambda_m**2 * s_ps_nm2_km * 1e3 + 2 * lambda_m * d_ps_nm_km * 1e-6
    )


# =============================================================================
# 2. Stimulated Raman scattering (SRS)
# =============================================================================
# SRS Raman Gain Coefficient Cr [1/(W·m·THz)]
_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)] if we define r_f using Hz


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_raman_profile_jit(
    f_rel_hz: NDArrayFloat,  # Relative frequency (Hz)
    f_m_hz: float,  # Lower frequency (Hz)
    f_M_hz: float,  # Upper frequency (Hz)
    delta_f_co_hz: float,  # Co-channel bandwidth (Hz)
    p_total_w: float,  # Total power (W)
) -> NDArrayFloat:
    """Calculate the Raman power-transfer coefficient for each channel.

    The profile models stimulated Raman scattering power tilt using the
    four cases of the generalized Raman scattering model for multiband SNR
    estimation.

    Parameters
    ----------
    f_rel_hz : numpy.ndarray
        Relative channel frequencies in Hz.
    f_m_hz, f_M_hz : float
        Lower and upper WDM band frequencies in Hz.
    delta_f_co_hz : float
        Co-channel bandwidth in Hz.
    p_total_w : float
        Total WDM launch power in watts.

    Returns
    -------
    numpy.ndarray
        Raman power-transfer coefficient per channel.
    """
    B_t = f_M_hz - f_m_hz
    r_f = np.zeros_like(f_rel_hz)

    # Precompute the coefficient.
    p_per_b = p_total_w / B_t if B_t > 0 else 0.0

    for i in range(len(f_rel_hz)):
        fi = f_rel_hz[i]

        # Define the case conditions.
        case1 = (fi - delta_f_co_hz <= f_m_hz) and (fi + delta_f_co_hz >= f_M_hz)
        case2 = (fi - delta_f_co_hz >= f_m_hz) and (fi + delta_f_co_hz <= f_M_hz)
        case3 = (fi - delta_f_co_hz < f_m_hz) and (fi + delta_f_co_hz < f_M_hz)
        case4 = (fi - delta_f_co_hz > f_m_hz) and (fi + delta_f_co_hz > f_M_hz)

        # Case precedence is significant.
        if case1:
            r_f[i] = p_total_w * fi
        elif case2:
            r_f[i] = 0.0
        elif case3:
            # Paper Case ③: (Pt/Bt) * (f^2/2 - f*fm + (fM^2 - delta_f_co^2)/2)
            r_f[i] = p_per_b * (
                0.5 * fi**2 - fi * f_m_hz + 0.5 * (f_M_hz**2 - delta_f_co_hz**2)
            )
        elif case4:
            # Paper Case ④: (Pt/Bt) * (fM*f - f^2/2 - (fm^2 - delta_f_co^2)/2)
            r_f[i] = p_per_b * (
                f_M_hz * fi - 0.5 * fi**2 - 0.5 * (f_m_hz**2 - delta_f_co_hz**2)
            )
        else:
            # Fall back to the linear case.
            r_f[i] = p_total_w * fi

    return r_f


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_span_out_power_jit(
    p_in_w: NDArrayFloat,  # Per-channel launch power (W)
    f_rel_hz: NDArrayFloat,  # noqa: ARG001 - retained for kernel API compatibility
    alpha_power_npm: NDArrayFloat,  # Per-channel power attenuation (Np/m)
    length_m: float,  # Span length (m)
    r_f: NDArrayFloat,  # Raman power-transfer coefficient
    k_bar: float = 0.0,  # Pump coefficient
) -> NDArrayFloat:
    """Calculate channel output powers after fiber loss and SRS tilt.

    Parameters
    ----------
    p_in_w : numpy.ndarray
        Per-channel launch power in watts.
    f_rel_hz : numpy.ndarray
        Relative channel frequencies in Hz.
    alpha_power_npm : numpy.ndarray
        Per-channel power attenuation coefficients in nepers per meter.
    length_m : float
        Span length in meters.
    r_f : numpy.ndarray
        Raman power-transfer coefficients.
    k_bar : float, optional
        Raman pumping factor.

    Returns
    -------
    numpy.ndarray
        Per-channel fiber output power in watts.
    """
    p_total_in = np.sum(p_in_w)
    N_ch = len(p_in_w)
    L_eff_array = (1 - np.exp(-alpha_power_npm * length_m)) / alpha_power_npm
    p_out_w = np.zeros_like(p_in_w)

    for i in range(N_ch):
        if p_in_w[i] > 0:
            # Multiply the scalar effective length by the Raman profile array.
            denom_i = np.sum(
                p_in_w * np.exp(-(1 - k_bar) * _SRS_CR * L_eff_array[i] * r_f)
            )
            if denom_i > 0:
                p_out_w[i] = (
                    p_in_w[i]
                    * np.exp(
                        -(1 - k_bar) * _SRS_CR * L_eff_array[i] * r_f[i]
                        - alpha_power_npm[i] * length_m
                    )
                    * p_total_in
                    / denom_i
                )

    return p_out_w


# =============================================================================
# 3. NLI noise variance kernels
# =============================================================================
# EPSILON prevents division by zero.
_EPSILON = 1e-35


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_spm_variance_jit(
    p_i_w: float,  # Channel-of-interest launch power (W)
    b_i_hz: float,  # Channel-of-interest bandwidth B_i (Hz)
    n_span: int,  # Number of spans
    gamma_ii: float,  # Channel-of-interest nonlinear coefficient (1/(W m))
    phi_i: float,  # Channel-of-interest effective phase (rad/m)
    alpha_i_npm: float,  # Channel-of-interest power attenuation (Np/m)
    t_i: float,  # Channel-of-interest effective attenuation including SRS
    r_fi: float,  # Channel-of-interest Raman transfer coefficient
    cr_w_m_hz: float,  # Raman gain coefficient Cr
    k_bar: float,  # Raman pumping factor (normally zero)
    l_eff_i_m: float,  # Channel-of-interest effective length (m)
    epsilon_n_span: float = 0.05,  # Span-count correction exponent from the original implementation
) -> float:
    """Calculate single-channel SPM noise variance.

    Parameters
    ----------
    p_i_w : float
        Channel-of-interest launch power in watts.
    b_i_hz : float
        Channel-of-interest bandwidth in Hz.
    n_span : int
        Effective number of spans.
    gamma_ii : float
        Self-interaction nonlinear coefficient.
    phi_i : float
        Effective nonlinear phase coefficient.
    alpha_i_npm : float
        Power attenuation in nepers per meter.
    t_i : float
        Effective attenuation factor including the SRS correction.
    r_fi : float
        Raman power-transfer coefficient.
    cr_w_m_hz : float
        Raman gain coefficient.
    k_bar : float
        Raman pump factor.
    l_eff_i_m : float
        Effective length in meters.
    epsilon_n_span : float, optional
        Span accumulation correction exponent.

    Returns
    -------
    float
        SPM noise variance in watts.

    Notes
    -----
    The channel's own bandwidth enters the formula, so heterogeneous
    channel widths are supported.
    """
    term_exp = np.exp(
        2 * cr_w_m_hz * k_bar * l_eff_i_m * r_fi
    )  # This factor equals one when k_bar is zero.

    # Guard against zero or near-zero phase.
    if abs(phi_i) < _EPSILON:
        phi_i = _EPSILON if phi_i >= 0 else -_EPSILON

    pre_factor = (
        (4 / 9)
        * (p_i_w**3 / b_i_hz**2)
        * (n_span ** (1 + epsilon_n_span))
        * ((gamma_ii**2 * PI) / (phi_i * 3 * alpha_i_npm**2))
        * term_exp
    )

    asinh_1 = np.arcsinh((phi_i * b_i_hz**2) / (PI * alpha_i_npm))
    asinh_2 = np.arcsinh((phi_i * b_i_hz**2) / (2 * PI * alpha_i_npm))

    term_bracket = ((t_i - alpha_i_npm**2) / alpha_i_npm) * asinh_1 + (
        (4 * alpha_i_npm**2 - t_i) / (2 * alpha_i_npm)
    ) * asinh_2

    result = pre_factor * term_bracket
    return max(result, 0.0)  # type: ignore[no-any-return]  # Noise variance cannot be negative.


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_xpm_variance_term_jit(
    p_l_w: float,  # Interfering-channel launch power (W)
    b_l_hz: float,  # Interfering-channel bandwidth B_l (Hz)
    b_i_hz: float,  # Channel-of-interest bandwidth used in arctangent terms (Hz)
    gamma_il: float,  # Inter-channel nonlinear coefficient (1/(W m))
    phi_il: float,  # Inter-channel effective phase (rad/m)
    alpha_l_npm: float,  # Interfering-channel power attenuation (Np/m)
    t_l: float,  # Interfering-channel effective attenuation including SRS
    r_fl: float,  # Interfering-channel Raman transfer coefficient
    cr_w_m_hz: float,  # Raman gain coefficient Cr
    k_bar: float,  # Raman pumping factor (normally zero)
    l_eff_l_m: float,  # Interfering-channel effective length (m)
) -> float:
    """Calculate one interfering channel's XPM noise contribution.

    Parameters
    ----------
    p_l_w : float
        Interfering channel launch power in watts.
    b_l_hz, b_i_hz : float
        Interfering-channel and channel-of-interest bandwidths in Hz.
    gamma_il, phi_il : float
        Nonlinear interaction and effective phase coefficients.
    alpha_l_npm : float
        Interfering-channel attenuation in nepers per meter.
    t_l, r_fl : float
        Effective attenuation and Raman power-transfer factors.
    cr_w_m_hz, k_bar : float
        Raman gain and pumping coefficients.
    l_eff_l_m : float
        Interfering-channel effective length in meters.

    Returns
    -------
    float
        XPM noise variance contribution.

    Notes
    -----
    The power-density factor uses the interfering-channel bandwidth, while
    the arctangent terms use the channel-of-interest bandwidth.
    """
    term_exp = np.exp(
        2 * cr_w_m_hz * k_bar * l_eff_l_m * r_fl
    )  # This factor equals one when k_bar is zero.

    # Guard against zero or near-zero phase.
    if abs(phi_il) < _EPSILON:
        phi_il = _EPSILON if phi_il >= 0 else -_EPSILON

    pre_factor = (
        (p_l_w**2 / b_l_hz) * (gamma_il**2 / (phi_il * 3 * alpha_l_npm**2)) * term_exp
    )

    atan_1 = np.arctan((phi_il * b_i_hz) / alpha_l_npm)
    atan_2 = np.arctan((phi_il * b_i_hz) / (2 * alpha_l_npm))

    term_bracket = ((t_l - alpha_l_npm**2) / alpha_l_npm) * atan_1 + (
        (4 * alpha_l_npm**2 - t_l) / (2 * alpha_l_npm)
    ) * atan_2

    result = pre_factor * term_bracket
    return max(result, 0.0)  # type: ignore[no-any-return]  # Noise variance cannot be negative.


@njit(cache=True)  # type: ignore[untyped-decorator]  # No fine-grained multithreading.
def _compute_nli_variances(
    n_ch: int,  # Number of channels
    n_span: int,  # Number of spans
    b_hz_channels: NDArrayFloat,  # Per-channel bandwidths B_i (Hz)
    p_in_w: NDArrayFloat,  # Per-channel launch power (W)
    f_rel_hz: NDArrayFloat,  # Per-channel relative frequency (Hz)
    lambdas_m: NDArrayFloat,  # Per-channel wavelengths (m)
    a_eff_m2: NDArrayFloat,  # Per-channel effective areas (m²)
    alpha_power_npm: NDArrayFloat,  # Per-channel power attenuation (Np/m)
    t_factors: NDArrayFloat,  # Per-channel effective attenuation including SRS
    r_f_values: NDArrayFloat,  # Per-channel Raman transfer coefficients
    l_eff_m: NDArrayFloat,  # Per-channel effective lengths (m)
    beta_2_s2_m: float,  # Dispersion parameter beta2 (s²/m)
    beta_3_s3_m: float,  # Dispersion parameter beta3 (s³/m)
    n_2_m2_w: float,  # Nonlinear refractive index n2 (m²/W)
    cr_w_m_hz: float,  # Raman gain coefficient Cr (1/(W m Hz))
    k_bar: float,  # Raman pumping factor (normally zero)
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Calculate SPM and XPM variances for all channels.

    Parameters
    ----------
    n_ch, n_span : int
        Number of channels and effective number of spans.
    b_hz_channels, p_in_w, f_rel_hz : numpy.ndarray
        Per-channel bandwidths in Hz, input powers in watts, and relative
        frequencies in Hz.
    lambdas_m, a_eff_m2 : numpy.ndarray
        Per-channel wavelengths in meters and effective areas in square meters.
    alpha_power_npm, t_factors, r_f_values, l_eff_m : numpy.ndarray
        Per-channel power attenuation, effective attenuation, Raman transfer,
        and effective length, respectively.
    beta_2_s2_m, beta_3_s3_m : float
        Second- and third-order dispersion coefficients.
    n_2_m2_w : float
        Nonlinear refractive index in square meters per watt.
    cr_w_m_hz, k_bar : float
        Raman gain and pumping coefficients.

    Returns
    -------
    tuple of numpy.ndarray
        Per-channel SPM and XPM noise variances in watts.
    """
    sigma_spm_2 = np.zeros(n_ch, dtype=np.float64)
    sigma_xpm_2 = np.zeros(n_ch, dtype=np.float64)

    # Iterate over channels of interest.
    for i in range(n_ch):
        if p_in_w[i] <= 0:  # Skip unpowered channels.
            continue

        # Calculate the SPM contribution.
        gamma_ii = _calc_gamma_coeff_scalar(
            lambdas_m[i], lambdas_m[i], n_2_m2_w, a_eff_m2[i], a_eff_m2[i]
        )
        phi_i = 1.5 * PI**2 * (beta_2_s2_m + 2 * PI * f_rel_hz[i] * beta_3_s3_m)

        sigma_spm_2[i] = _calc_spm_variance_jit(
            p_in_w[i],
            b_hz_channels[i],  # Channel-of-interest bandwidth B_i.
            n_span,
            gamma_ii,
            phi_i,
            alpha_power_npm[i],
            t_factors[i],
            r_f_values[i],
            cr_w_m_hz,
            k_bar,
            l_eff_m[i],
        )

        # Calculate the XPM contribution.
        xpm_sum_term = 0.0
        for l_idx in range(n_ch):
            if (
                l_idx == i or p_in_w[l_idx] <= 0
            ):  # The interfering channel must be powered and distinct from the channel of interest.
                continue
            gamma_il = _calc_gamma_coeff_scalar(
                lambdas_m[i], lambdas_m[l_idx], n_2_m2_w, a_eff_m2[i], a_eff_m2[l_idx]
            )

            # Calculate phi_il.
            # phi_il = 2 * PI**2 * (f_rel_hz[l] - f_rel_hz[i])
            # * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[l] + f_rel_hz[i]))
            # Original formula: 2*pi^2*Δf*(beta2 + pi*beta3*(f_l+f_i)).
            phi_il = (
                2
                * PI**2
                * (f_rel_hz[l_idx] - f_rel_hz[i])
                * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[l_idx] + f_rel_hz[i]))
            )

            xpm_sum_term += _calc_xpm_variance_term_jit(
                p_in_w[l_idx],
                b_hz_channels[l_idx],  # Interfering-channel bandwidth B_l.
                b_hz_channels[i],  # Channel-of-interest bandwidth for arctangent terms.
                gamma_il,
                phi_il,
                alpha_power_npm[l_idx],
                t_factors[l_idx],
                r_f_values[l_idx],
                cr_w_m_hz,
                k_bar,
                l_eff_m[l_idx],
            )

        # XPM noise is proportional to channel-of-interest power.
        sigma_xpm_2[i] = (32.0 / 27.0) * p_in_w[i] * n_span * xpm_sum_term

    return sigma_spm_2, sigma_xpm_2


# =============================================================================
# 4. ASE noise
# =============================================================================


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_nf_lin_jit(lambdas_m: NDArrayFloat) -> NDArrayFloat:
    """Estimate linear amplifier noise figure from wavelength band.

    Parameters
    ----------
    lambdas_m : numpy.ndarray
        Wavelengths in meters.

    Returns
    -------
    numpy.ndarray
        Approximate linear noise figure for each wavelength.
    """
    N_ch = len(lambdas_m)
    nf_lin = np.zeros(N_ch, dtype=np.float64)
    for i in range(N_ch):
        wavel_nm = lambdas_m[i] * 1e9
        if wavel_nm < 1460:  # E-band
            nf_lin[i] = _db_to_lin(6.0)
        elif wavel_nm < 1530:  # S-band
            nf_lin[i] = _db_to_lin(7.0)
        elif wavel_nm < 1565:  # C-band
            nf_lin[i] = _db_to_lin(5.5)
        else:  # L-band
            nf_lin[i] = _db_to_lin(6.0)
    return nf_lin


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calculate_ase_noise_variance(
    p_in_w: NDArrayFloat,  # Per-channel launch power (W)
    p_out_w: NDArrayFloat,  # Per-channel output power (W)
    f_abs_hz: NDArrayFloat,  # Per-channel absolute frequency (Hz)
    nf_lin: NDArrayFloat,  # Per-channel linear noise figure
    b_hz_channels: NDArrayFloat,  # Per-channel bandwidths B_i (Hz)
    n_span: int,  # Number of spans
) -> NDArrayFloat:
    """Calculate accumulated ASE noise for loss-compensating amplifiers.

    Parameters
    ----------
    p_in_w, p_out_w : numpy.ndarray
        Per-channel fiber input and output powers in watts.
    f_abs_hz : numpy.ndarray
        Absolute channel frequencies in Hz.
    nf_lin : numpy.ndarray
        Linear noise figures for each channel.
    b_hz_channels : numpy.ndarray
        Per-channel bandwidths in Hz.
    n_span : int
        Number of spans.

    Returns
    -------
    numpy.ndarray
        Per-channel ASE noise variance in watts.

    Notes
    -----
    This historical kernel uses input/output power as gain. The current
    ISRS-GN route computes the effective ASE separately and discards this
    kernel's ASE output.
    """
    total_ase_variance = np.zeros(len(p_in_w), dtype=np.float64)
    for i in range(len(p_in_w)):
        if p_in_w[i] > 0 and p_out_w[i] > 0:  # Only powered channels produce ASE noise.
            # Calculate span gain assuming loss-compensating EDFA.
            G_i = p_in_w[i] / p_out_w[i]

            total_ase_variance[i] = (
                n_span
                * nf_lin[i]
                * H_PLANCK
                * f_abs_hz[i]
                * max((G_i - 1.0), 0.0)
                * b_hz_channels[i]
            )

    return total_ase_variance  # Return variance in watts.


@njit(cache=True)  # type: ignore[untyped-decorator]
def _calc_span_noise_and_power_jit(
    # --- Input state ---
    power_in_w: NDArrayFloat,  # Per-channel launch power (W)
    span_length_m: float,  # Current span length (m)
    # --- Precomputed spectrum arrays ---
    f_rel_hz: NDArrayFloat,  # Relative frequency (Hz)
    f_abs_hz: NDArrayFloat,  # Absolute frequency (Hz)
    lambdas_m: NDArrayFloat,  # Wavelength (m)
    # --- Fiber parameters derived from FiberSpanConfig ---
    alpha_power_npm: NDArrayFloat,  # Power attenuation (Np/m)
    a_eff_m2: NDArrayFloat,  # Effective mode area (m²)
    beta2_s2_m: float,  # Dispersion beta2
    beta3_s3_m: float,  # Dispersion beta3
    n2_m2_w: float,  # Nonlinear refractive index
    # --- System and amplifier parameters ---
    nf_lin_channels: NDArrayFloat,  # Per-channel linear amplifier noise figure
    symbol_rate_hz: float,  # Symbol rate (Hz), used for SRS band edges, not B_i
    b_hz_channels: NDArrayFloat,  # Per-channel bandwidth B_i (Hz)
    n_effective_spans: int,  # Effective span count for NLI accumulation
    # --- Physical constants ---
    cr_w_m_hz: float,  # SRS gain coefficient Cr
    k_bar: float,  # Pumping factor
    delta_f_co_hz: float,  # SRS cutoff frequency
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Evaluate fiber output and SPM, XPM, and ASE for one span.

    Parameters
    ----------
    power_in_w : numpy.ndarray
        Per-channel launch power in watts.
    span_length_m : float
        Span length in meters.
    f_rel_hz, f_abs_hz, lambdas_m : numpy.ndarray
        Relative frequencies, absolute frequencies, and wavelengths.
    alpha_power_npm, a_eff_m2 : numpy.ndarray
        Per-channel attenuation and effective mode area.
    beta2_s2_m, beta3_s3_m, n2_m2_w : float
        Fiber dispersion coefficients and nonlinear refractive index.
    nf_lin_channels : numpy.ndarray
        Per-channel linear amplifier noise figures.
    symbol_rate_hz : float
        Symbol rate used to determine the SRS band edges, in Hz.
    b_hz_channels : numpy.ndarray
        Per-channel bandwidths in Hz, one for each channel.
    n_effective_spans : int
        Effective number of accumulated spans for NLI.
    cr_w_m_hz, k_bar, delta_f_co_hz : float
        Raman gain, pumping factor, and SRS cutoff frequency.

    Returns
    -------
    tuple of numpy.ndarray
        Output powers and SPM, XPM, and ASE noise variances, in watts.

    Raises
    ------
    ValueError
        If the channel bandwidth array has the wrong length.
    """
    num_channels = len(power_in_w)

    # Require one bandwidth per channel to avoid silently misaligned SPM/XPM.
    if len(b_hz_channels) != num_channels:
        raise ValueError("b_hz_channels must have one bandwidth per channel.")  # noqa: EM101, TRY003

    # 1. Precompute intermediate quantities.
    total_p_in_w = np.sum(power_in_w)
    # L_eff: effective length.
    l_eff_m = (1.0 - np.exp(-alpha_power_npm * span_length_m)) / alpha_power_npm

    # 2. Calculate the SRS power-transfer profile r_f.
    # Determine SRS band edges from the relative frequencies.
    f_m_hz = f_rel_hz[0] - symbol_rate_hz / 2.0
    f_M_hz = f_rel_hz[-1] + symbol_rate_hz / 2.0

    r_f_values = _calc_raman_profile_jit(
        f_rel_hz, f_m_hz, f_M_hz, delta_f_co_hz, total_p_in_w
    )

    # 3. Calculate fiber output power with attenuation and SRS tilt.
    p_out_after_fiber = _calc_span_out_power_jit(
        power_in_w, f_rel_hz, alpha_power_npm, span_length_m, r_f_values, k_bar
    )

    # 4. Calculate SPM and XPM noise variances.
    # Compute the T factor: (2*alpha - Cr*r_f)^2.
    t_factors = (2.0 * alpha_power_npm - cr_w_m_hz * r_f_values) ** 2

    sigma2_spm_w, sigma2_xpm_w = _compute_nli_variances(
        num_channels,
        n_effective_spans,
        b_hz_channels,
        power_in_w,
        f_rel_hz,
        lambdas_m,
        a_eff_m2,
        alpha_power_npm,
        t_factors,
        r_f_values,
        l_eff_m,
        beta2_s2_m,
        beta3_s3_m,
        n2_m2_w,
        cr_w_m_hz,
        k_bar,
    )

    # 5. Calculate ASE noise variance.
    sigma2_ase_w = _calculate_ase_noise_variance(
        power_in_w,
        p_out_after_fiber,
        f_abs_hz,
        nf_lin_channels,
        b_hz_channels,
        n_effective_spans,
    )

    return p_out_after_fiber, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w
