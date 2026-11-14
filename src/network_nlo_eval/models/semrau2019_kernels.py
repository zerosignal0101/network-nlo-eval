"""Constant matrices and Numba kernels for the Semrau 2019 ISRS-GN model.

The closed-form SPM and XPM brackets are linear in the power-dependent T
factor. Their channel-pair coefficients are precomputed, so runtime XPM
sums use matrix products. The optional EGN term follows the 2019 modulation
format correction. See Semrau, Killey, and Bayvel, JLT 2019,
doi:10.1109/JLT.2019.2895237 and doi:10.1109/JLT.2019.2929461.
"""

# ruff: noqa: RUF003  # Retain the mathematical notation in equations.

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from network_nlo_eval.core.constants import PI
from network_nlo_eval.core.types import NDArrayFloat  # noqa: TC001

_F = TypeVar("_F", bound=Callable[..., Any])

try:
    from numba import njit
except ImportError:  # Keep statistical models usable without JIT.

    def njit(*args: Any, **kwargs: Any) -> Callable[[_F], _F]:  # type: ignore[misc]
        """Return an identity decorator when optional Numba is unavailable."""
        del args, kwargs

        def decorate(function: _F) -> _F:
            return function

        return decorate


__all__ = [
    "GN_VISIBILITY_QUADRATURE_NODES",
    "SemrauMatrices",
    "build_semrau_matrices",
    "gn_weighted_spm_visibility",
    "semrau_span_jit",
]

#: Phase floor matching the shared Numba kernel, preventing division by zero.
_EPSILON = 1e-35

#: Floor for the dispersion term in the 2019b phase expression.
_BETA_FLOOR = 1e-30

#: Fixed quadrature node count for GN-weighted inter-span visibility.
#:
#: The fixed node count gives predictable per-evaluation cost.
#: For the ``c_57ch`` reference, 1024 nodes yield approximately
#: 16 nodes per oscillation period at the largest tested phase;
#: relative visibility error is about 5e-5 versus a 1201x1201 reference.
GN_VISIBILITY_QUADRATURE_NODES = 1024


class SemrauMatrices:
    """Per-channel and channel-pair constants for the Semrau kernel.

    Attributes
    ----------
    spm_u, spm_v : numpy.ndarray
        Coefficients of the linear SPM bracket, each with shape ``(N,)``.
    spm_coef : numpy.ndarray
        Per-channel SPM prefactors, shape ``(N,)``.
    mat_u, mat_v : numpy.ndarray
        XPM coefficient matrices, shape ``(N, N)`` with zero diagonal.
    mat_w : numpy.ndarray
        EGN asymptotic correction matrix, excluding the runtime inverse length.
    """

    __slots__ = ("mat_u", "mat_v", "mat_w", "spm_coef", "spm_u", "spm_v")

    def __init__(
        self,
        spm_u: NDArrayFloat,
        spm_v: NDArrayFloat,
        spm_coef: NDArrayFloat,
        mat_u: NDArrayFloat,
        mat_v: NDArrayFloat,
        mat_w: NDArrayFloat,
    ) -> None:
        self.spm_u = spm_u
        self.spm_v = spm_v
        self.spm_coef = spm_coef
        self.mat_u = mat_u
        self.mat_v = mat_v
        self.mat_w = mat_w


def _clamp_phi(phi: NDArrayFloat) -> NDArrayFloat:
    """Clamp near-zero phase values to the same-sign epsilon floor."""
    small = np.abs(phi) < _EPSILON
    if np.any(small):
        phi = np.where(small, np.where(phi >= 0.0, _EPSILON, -_EPSILON), phi)
    return phi


def build_semrau_matrices(
    *,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    beta2_s2_m: float,
    beta3_s3_m: float,
    n2_m2_w: float,
    with_egn: bool,
) -> SemrauMatrices:
    """Construct constant SPM, XPM, and optional EGN coefficient matrices.

    Parameters
    ----------
    f_rel_hz : numpy.ndarray
        Per-channel frequencies relative to the band center, in Hz.
    lambdas_m : numpy.ndarray
        Per-channel wavelengths in meters.
    b_hz : numpy.ndarray
        Per-channel bandwidths in Hz.
    alpha_npm : numpy.ndarray
        Per-channel power attenuation in nepers per meter.
    a_eff_m2 : numpy.ndarray
        Per-channel effective areas in square meters.
    beta2_s2_m, beta3_s3_m : float
        Second- and third-order dispersion coefficients.
    n2_m2_w : float
        Nonlinear refractive index in square meters per watt.
    with_egn : bool
        Whether to build the modulation-format correction matrix.

    Returns
    -------
    SemrauMatrices
        Precomputed per-channel and channel-pair coefficients.

    Notes
    -----
    The reference frequency is the band center; the channel at zero relative
    frequency has no net Raman gain under this approximation.
    """
    n = int(f_rel_hz.size)
    fi = f_rel_hz[:, None]  # Channel-of-interest index i.
    fk = f_rel_hz[None, :]  # Interfering-channel index k.
    bi = b_hz[:, None]
    bk = b_hz[None, :]
    ak = alpha_npm[None, :]

    # γ_ik = (2π/λ_i)·(2n2/(A_eff,i + A_eff,k)) (1/(W·m))
    gamma = (2.0 * PI / lambdas_m[:, None]) * (
        2.0 * n2_m2_w / (a_eff_m2[:, None] + a_eff_m2[None, :])
    )

    # --- SPM: Eq. (10) -----------------------------------------------------
    # φ_i = 1.5π²(β_2 + 2πβ_3 f_i)
    phi_spm = _clamp_phi(1.5 * PI**2 * (beta2_s2_m + 2.0 * PI * f_rel_hz * beta3_s3_m))
    asinh_1 = np.arcsinh(phi_spm * b_hz**2 / (PI * alpha_npm))
    asinh_2 = np.arcsinh(phi_spm * b_hz**2 / (2.0 * PI * alpha_npm))
    # bracket = (T-α²)/α·asinh1 + (4α²-T)/(2α)·asinh2 = T·u + v（ᾱ=α, A=2α）
    spm_u = asinh_1 / alpha_npm - asinh_2 / (2.0 * alpha_npm)
    spm_v = -alpha_npm * asinh_1 + 2.0 * alpha_npm * asinh_2
    gamma_ii = (2.0 * PI / lambdas_m) * (2.0 * n2_m2_w / (2.0 * a_eff_m2))
    spm_coef = (4.0 / 9.0) * gamma_ii**2 * PI / (phi_spm * 3.0 * alpha_npm**2)

    # --- XPM: Eq. (11) -----------------------------------------------------
    # φ_{i,k} = 2π²(f_k - f_i)[β_2 + πβ_3(f_i+f_k)]
    phi_xpm = _clamp_phi(
        2.0 * PI**2 * (fk - fi) * (beta2_s2_m + PI * beta3_s3_m * (fi + fk))
    )
    atan_1 = np.arctan(phi_xpm * bi / ak)
    atan_2 = np.arctan(phi_xpm * bi / (2.0 * ak))
    # bracket = (T-α²)/α·atan1 + (4α²-T)/(2α)·atan2 = T·u + v
    xpm_u = atan_1 / ak - atan_2 / (2.0 * ak)
    xpm_v = -ak * atan_1 + 2.0 * ak * atan_2
    denom = bk * phi_xpm * 3.0 * ak**2
    mat_u = gamma**2 * xpm_u / denom
    mat_v = gamma**2 * xpm_v / denom
    # XPM excludes self-interaction terms.
    np.fill_diagonal(mat_u, 0.0)
    np.fill_diagonal(mat_v, 0.0)

    # --- Asymptotic correction from 2019b Eq. (15) ------------------------
    if with_egn:
        delta_f = np.abs(fk - fi)
        numerator = 2.0 * delta_f - bk
        positive = numerator > 0.0
        safe = np.where(positive, numerator, 1.0)
        log_term = np.log(safe / (2.0 * delta_f + bk))
        bracket_q = np.where(positive, numerator * log_term + 2.0 * bk, 0.0)
        # The 2019b phase is 4π²|beta2 + π beta3 (fi + fk)| times runtime L.
        beta_sum = np.abs(beta2_s2_m + PI * beta3_s3_m * (fi + fk))
        beta_sum = np.maximum(beta_sum, _BETA_FLOOR)
        # Apply the remaining inverse length at runtime.
        mat_w = (
            gamma**2
            * bracket_q
            / (4.0 * PI**2 * beta_sum * bk**3 * ak**2 * (2.0 * ak) ** 2)
        )
        np.fill_diagonal(mat_w, 0.0)
    else:
        mat_w = np.zeros((n, n), dtype=np.float64)

    return SemrauMatrices(
        spm_u=spm_u,
        spm_v=spm_v,
        spm_coef=spm_coef,
        mat_u=mat_u,
        mat_v=mat_v,
        mat_w=mat_w,
    )


def gn_weighted_spm_visibility(
    *,
    beta2_abs_s2_m: float,
    bandwidth_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    span_length_m: float,
    span_separations_m: NDArrayFloat,
) -> NDArrayFloat:
    """Calculate GN-weighted inter-span self-beating visibility.

    Parameters
    ----------
    beta2_abs_s2_m : float
        Absolute second-order dispersion in seconds squared per meter.
    bandwidth_hz, alpha_npm : numpy.ndarray
        Per-channel bandwidth in Hz and power attenuation in nepers per meter.
    span_length_m : float
        Length of a single span in meters.
    span_separations_m : numpy.ndarray
        Distances between span copies in meters, shape ``(K,)``.

    Returns
    -------
    numpy.ndarray
        Visibility in the interval [-1, 1], with shape ``(K, N)``. Zero
        dispersion or bandwidth yields unit visibility.

    Raises
    ------
    ValueError
        If the span length is non-positive.

    Notes
    -----
    The average of ``cos(theta * separation / length)`` is weighted by the
    GN kernel ``1 / ((alpha * length)**2 + theta**2)`` and the logarithmic
    density induced by the product of two uniformly distributed in-band
    frequencies. The substitution ``theta = theta_max * s**2`` reduces the
    finite-bandwidth integral to a fixed-node one-dimensional quadrature.
    """
    bandwidth = np.asarray(bandwidth_hz, dtype=np.float64)
    alpha = np.asarray(alpha_npm, dtype=np.float64)
    separations = np.asarray(span_separations_m, dtype=np.float64)
    length = float(span_length_m)
    if length <= 0.0:
        raise ValueError(f"span_length_m must be positive, got {length}.")  # noqa: EM102, TRY003
    n_ch = int(bandwidth.size)
    n_sep = int(separations.size)
    # Separation-to-length ratio equals the span separation for equal spans.
    ratios = separations / length

    # Phase mismatch coefficient and its single-span upper bound.
    a_mix = 4.0 * PI**2 * abs(float(beta2_abs_s2_m)) * length
    theta_max = a_mix * (0.5 * bandwidth) ** 2
    # Dimensionless Lorentzian width: alpha * length.
    abar = alpha * length

    # Fixed quadrature nodes s = j/N, j = 1..N.
    nodes = GN_VISIBILITY_QUADRATURE_NODES
    s = np.arange(1, nodes + 1, dtype=np.float64) / float(nodes)
    s_squared = s * s
    log_term = -4.0 * np.log(s)  # Logarithmic density after the change of variables.

    out = np.empty((n_sep, n_ch), dtype=np.float64)
    for i in range(n_ch):
        if theta_max[i] <= 0.0:
            # Zero dispersion or bandwidth gives perfect coherence.
            out[:, i] = 1.0
            continue
        theta = theta_max[i] * s_squared
        weight = log_term * (2.0 * theta_max[i] * s) / (abar[i] ** 2 + theta**2)
        cosine = np.cos(np.outer(ratios, theta))
        out[:, i] = (cosine @ weight) / weight.sum()
    return out


@njit(cache=True)
def semrau_span_jit(
    power_in_w: NDArrayFloat,
    span_length_m: float,
    n_spans: int,
    cr_w_m_hz: float,
    excess_kurtosis: float,
    coherence_epsilon: float,
    per_span: bool,
    alpha_npm: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    spm_u: NDArrayFloat,
    spm_v: NDArrayFloat,
    spm_coef: NDArrayFloat,
    mat_u: NDArrayFloat,
    mat_v: NDArrayFloat,
    mat_w: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Evaluate output power and closed-form SPM and XPM noise for a span.

    Parameters
    ----------
    power_in_w : numpy.ndarray
        Per-channel launch power in watts; zero indicates an idle channel.
    span_length_m : float
        Span length in meters.
    n_spans : int
        Effective span count.
    cr_w_m_hz : float
        Raman gain slope; zero disables ISRS.
    excess_kurtosis : float
        Modulation excess kurtosis; zero disables format correction.
    coherence_epsilon : float
        SPM coherence exponent.
    per_span : bool
        Return the equivalent per-span noise when true, or accumulated
        multispan noise when false.
    alpha_npm, f_rel_hz, b_hz : numpy.ndarray
        Per-channel attenuation, relative frequency, and bandwidth.
    spm_u, spm_v, spm_coef, mat_u, mat_v, mat_w : numpy.ndarray
        Precomputed coefficients from ``SemrauMatrices``.

    Returns
    -------
    tuple of numpy.ndarray
        New per-channel output power, SPM noise, and XPM noise arrays in watts.

    Notes
    -----
    XPM is expressed as channel-of-interest power times the sum over squared
    interfering powers. Per-span output is the paper's multispan result
    normalized by the span count.
    """
    n = int(power_in_w.size)
    power_out = np.zeros(n, dtype=np.float64)
    spm_noise = np.zeros(n, dtype=np.float64)
    xpm_noise = np.zeros(n, dtype=np.float64)

    p_total = 0.0
    for i in range(n):
        p_total += power_in_w[i]

    # --- ISRS power profile with linear Raman gain approximation ---
    # Integrating the effective attenuation gives the exponential profile;
    # normalize it so the total power transfer is conserved.
    tilt = np.zeros(n, dtype=np.float64)
    denom = 0.0
    for i in range(n):
        l_eff = (1.0 - np.exp(-alpha_npm[i] * span_length_m)) / alpha_npm[i]
        tilt[i] = p_total * cr_w_m_hz * l_eff * f_rel_hz[i]
        denom += power_in_w[i] * np.exp(-tilt[i])
    if denom > 0.0:
        for i in range(n):
            power_out[i] = (
                power_in_w[i]
                * np.exp(-alpha_npm[i] * span_length_m - tilt[i])
                * p_total
                / denom
            )

    # --- T factor: (2*alpha - P_tot*C_r*f_i)^2 ---------------------------
    p_sq = np.zeros(n, dtype=np.float64)
    p_sq_t = np.zeros(n, dtype=np.float64)
    t_factors = np.zeros(n, dtype=np.float64)
    for i in range(n):
        t_i = 2.0 * alpha_npm[i] - p_total * cr_w_m_hz * f_rel_hz[i]
        t_factors[i] = t_i * t_i
        p_sq[i] = power_in_w[i] * power_in_w[i]
        p_sq_t[i] = p_sq[i] * t_factors[i]

    # --- Multispan factors ------------------------------------------------
    # In 2019b Eq. (11), n_tilde is zero for one span, otherwise n.
    n_tilde = 0.0 if n_spans == 1 else float(n_spans)
    if per_span:
        # Equivalent per-span value: Eq. (16) total divided by n.
        spm_span_factor = float(n_spans) ** coherence_epsilon
        xpm_span_factor = 1.0 + (5.0 / 6.0) * excess_kurtosis / float(n_spans)
        egn_scale = (5.0 / 3.0) * excess_kurtosis * PI * (1.0 if n_spans > 1 else 0.0)
    else:
        # Eq. (16) total uses n^(1+epsilon) for SPM and n+5*Phi/6 for XPM.
        spm_span_factor = float(n_spans) ** (1.0 + coherence_epsilon)
        xpm_span_factor = float(n_spans) + (5.0 / 6.0) * excess_kurtosis
        egn_scale = (5.0 / 3.0) * excess_kurtosis * PI * n_tilde

    for i in range(n):
        if power_in_w[i] <= 0.0:
            continue
        acc_u = 0.0
        acc_v = 0.0
        acc_w = 0.0
        for k in range(n):
            acc_u += mat_u[i, k] * p_sq_t[k]
            acc_v += mat_v[i, k] * p_sq[k]
            acc_w += mat_w[i, k] * p_sq_t[k]
        gn_term = acc_u + acc_v
        xpm_noise[i] = (
            (32.0 / 27.0)
            * power_in_w[i]
            * (xpm_span_factor * gn_term + egn_scale * acc_w / span_length_m)
        )
        spm_noise[i] = (
            (power_in_w[i] ** 3 / b_hz[i] ** 2)
            * spm_span_factor
            * spm_coef[i]
            * (spm_u[i] * t_factors[i] + spm_v[i])
        )

    return power_out, spm_noise, xpm_noise
