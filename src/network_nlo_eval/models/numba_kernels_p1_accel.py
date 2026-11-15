"""Matrix-vector and batched acceleration for the P1 closed-form kernel.

Channel-pair geometry determines two constant XPM matrices. Precomputing
those matrices turns most pairwise summation into matrix multiplication.
A conservative safety test identifies cases where the baseline's per-pair
nonnegative clamp could apply; those contributions use the original kernel.
Only the XPM summation order changes, with a relative deviation budget of
1e-12. Fiber output power and SPM retain the baseline calculations.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np

from network_nlo_eval.core.constants import H_PLANCK, PI
from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat  # noqa: TC001
from network_nlo_eval.models.numba_kernels import (
    _EPSILON,
    _SRS_CR,
    _calc_gamma_coeff_scalar,
    _calc_raman_profile_jit,
    _calc_spm_variance_jit,
    _calc_xpm_variance_term_jit,
)

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


#: Relative deviation budget for matrix-vector versus baseline evaluation.
MAX_REL_DEVIATION = 1e-12


@dataclass(frozen=True)
class P1XpmMatrices:
    """Precomputed channel-pair matrices shared across spans and candidates.

    Attributes
    ----------
    g1, g2 : numpy.ndarray
        Nonnegative pair matrices for the first and second arctangent terms.
    g1_t, g2_t : numpy.ndarray
        Contiguous transposes used for batched matrix multiplication.
    phi_raw, gamma : numpy.ndarray
        Unclamped phase and nonlinear coefficients used by the pairwise fallback.
    """

    g1: NDArrayFloat
    g2: NDArrayFloat
    g1_t: NDArrayFloat
    g2_t: NDArrayFloat
    phi_raw: NDArrayFloat
    gamma: NDArrayFloat

    @property
    def n_ch(self) -> int:
        """Return the number of channels represented by the pair matrices."""
        return int(self.g1.shape[0])


@njit(cache=True)  # type: ignore[untyped-decorator]
def build_p1_xpm_pair_matrices(
    n_ch: int,
    b_hz_channels: NDArrayFloat,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Precompute the P1 XPM matrices for all channel pairs.

    Parameters
    ----------
    n_ch : int
        Number of channels.
    b_hz_channels, f_rel_hz, lambdas_m : numpy.ndarray
        Per-channel bandwidths and relative frequencies in Hz, and wavelengths
        in meters.
    a_eff_m2, alpha_power_npm : numpy.ndarray
        Per-channel effective areas and power attenuation coefficients.
    beta_2_s2_m, beta_3_s3_m, n_2_m2_w : float
        Dispersion coefficients and nonlinear refractive index.

    Returns
    -------
    tuple of numpy.ndarray
        ``(g1, g2, phi_raw, gamma)``, each with shape ``(N, N)``.

    Notes
    -----
    The equations, phase clamp, and zero diagonal match the baseline pairwise
    kernel; only their evaluation is moved to matrix construction time.
    """
    g1 = np.zeros((n_ch, n_ch), dtype=np.float64)
    g2 = np.zeros((n_ch, n_ch), dtype=np.float64)
    phi_raw = np.zeros((n_ch, n_ch), dtype=np.float64)
    gamma_mat = np.zeros((n_ch, n_ch), dtype=np.float64)

    for i in range(n_ch):
        for ell in range(n_ch):
            if ell == i:  # The serial baseline skips self-pairs.
                continue

            gamma_il = _calc_gamma_coeff_scalar(
                lambdas_m[i], lambdas_m[ell], n_2_m2_w, a_eff_m2[i], a_eff_m2[ell]
            )
            phi_il = (
                2
                * PI**2
                * (f_rel_hz[ell] - f_rel_hz[i])
                * (beta_2_s2_m + PI * beta_3_s3_m * (f_rel_hz[ell] + f_rel_hz[i]))
            )
            phi_raw[i, ell] = phi_il
            gamma_mat[i, ell] = gamma_il

            phi_c = phi_il
            if abs(phi_c) < _EPSILON:
                phi_c = _EPSILON if phi_c >= 0 else -_EPSILON

            a_il = gamma_il**2 / (phi_c * 3 * alpha_power_npm[ell] ** 2)
            g1[i, ell] = a_il * np.arctan(
                (phi_c * b_hz_channels[i]) / alpha_power_npm[ell]
            )
            g2[i, ell] = a_il * np.arctan(
                (phi_c * b_hz_channels[i]) / (2 * alpha_power_npm[ell])
            )

    return g1, g2, phi_raw, gamma_mat


@njit(cache=True)  # type: ignore[untyped-decorator]
def _p1_span_pre_nli_batch_jit(
    p_in: NDArrayFloat,
    span_length_m: float,
    f_rel_hz: NDArrayFloat,
    symbol_rate_hz: float,
    alpha_power_npm: NDArrayFloat,
    cr_w_m_hz: float,
    k_bar: float,
    delta_f_co_hz: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Calculate Raman profiles, output power, attenuation factors, and length.

    Parameters
    ----------
    p_in : numpy.ndarray
        Candidate launch powers in watts with shape ``(B, N)``.
    span_length_m : float
        Span length in meters.
    f_rel_hz : numpy.ndarray
        Relative channel frequencies in Hz.
    symbol_rate_hz : float
        Symbol rate used for SRS band edges in Hz.
    alpha_power_npm : numpy.ndarray
        Per-channel attenuation in nepers per meter.
    cr_w_m_hz, k_bar, delta_f_co_hz : float
        SRS gain, launch pre-emphasis factor, and cutoff frequency in Hz.

    Returns
    -------
    tuple of numpy.ndarray
        Output power, Raman profile, and attenuation factors with shape
        ``(B, N)``, followed by effective lengths with shape ``(N,)``.

    Notes
    -----
    When all effective attenuations are equal, reuse the common exponential
    vector. Unequal attenuations retain the baseline per-channel calculation.
    SRS band edges follow the same frequency convention as the baseline.
    """
    n_batch = p_in.shape[0]
    n_ch = p_in.shape[1]
    p_out = np.zeros((n_batch, n_ch), dtype=np.float64)
    r_f = np.zeros((n_batch, n_ch), dtype=np.float64)
    t_factors = np.zeros((n_batch, n_ch), dtype=np.float64)

    l_eff_m = (1.0 - np.exp(-alpha_power_npm * span_length_m)) / alpha_power_npm

    u = np.empty(n_ch, dtype=np.float64)
    for i in range(n_ch):
        u[i] = (-(1 - k_bar)) * _SRS_CR * l_eff_m[i]

    u_all_same = True
    for i in range(1, n_ch):
        if u[i] != u[0]:
            u_all_same = False
            break

    f_m_hz = f_rel_hz[0] - symbol_rate_hz / 2.0
    f_M_hz = f_rel_hz[-1] + symbol_rate_hz / 2.0

    for b in range(n_batch):
        pb = p_in[b]
        total_p_in_w = np.sum(pb)
        r_f_values = _calc_raman_profile_jit(
            f_rel_hz, f_m_hz, f_M_hz, delta_f_co_hz, total_p_in_w
        )
        r_f[b, :] = r_f_values

        if u_all_same:
            # Equal attenuation gives one common exponential vector.
            exp_row = np.exp(u[0] * r_f_values)
            denom_all = np.sum(pb * exp_row)
            for i in range(n_ch):
                if pb[i] > 0 and denom_all > 0:
                    p_out[b, i] = (
                        pb[i]
                        * np.exp(
                            u[i] * r_f_values[i] - alpha_power_npm[i] * span_length_m
                        )
                        * total_p_in_w
                        / denom_all
                    )
        else:
            for i in range(n_ch):
                if pb[i] > 0:
                    exp_row = np.exp(u[i] * r_f_values)
                    denom_i = np.sum(pb * exp_row)
                    if denom_i > 0:
                        p_out[b, i] = (
                            pb[i]
                            * np.exp(
                                u[i] * r_f_values[i]
                                - alpha_power_npm[i] * span_length_m
                            )
                            * total_p_in_w
                            / denom_i
                        )

        t_factors[b, :] = (2.0 * alpha_power_npm - cr_w_m_hz * r_f_values) ** 2

    return p_out, r_f, t_factors, l_eff_m


@njit(cache=True)  # type: ignore[untyped-decorator]
def _p1_xpm_weights_jit(
    n_ch: int,
    n_batch: int,
    p_in: NDArrayFloat,
    b_hz_channels: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    t_factors: NDArrayFloat,
    r_f: NDArrayFloat,
    l_eff_m: NDArrayFloat,
    cr_w_m_hz: float,
    k_bar: float,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayBool]:
    """Compute batched XPM weights and a mask for safe matrix evaluation.

    Parameters
    ----------
    n_ch, n_batch : int
        Number of channels and candidates.
    p_in, t_factors, r_f : numpy.ndarray
        Candidate power, attenuation factor, and Raman profile arrays of
        shape ``(B, N)``.
    b_hz_channels, alpha_power_npm, l_eff_m : numpy.ndarray
        Per-channel bandwidth, attenuation, and effective length arrays.
    cr_w_m_hz, k_bar : float
        SRS gain coefficient and dimensionless launch pre-emphasis.

    Returns
    -------
    tuple of numpy.ndarray
        Weights ``w1`` and ``w2`` and boolean ``safe`` mask, all ``(B, N)``.

    Notes
    -----
    The safe mask identifies channels for which the baseline per-pair
    nonnegative clamp cannot affect the result. Unsafe weights remain zero;
    their contributions are evaluated by the pairwise fallback.
    """
    w1 = np.zeros((n_batch, n_ch), dtype=np.float64)
    w2 = np.zeros((n_batch, n_ch), dtype=np.float64)
    safe = np.ones((n_batch, n_ch), dtype=np.bool_)

    for b in range(n_batch):
        for ell in range(n_ch):
            term_exp = np.exp(2 * cr_w_m_hz * k_bar * l_eff_m[ell] * r_f[b, ell])
            v = (p_in[b, ell] ** 2 / b_hz_channels[ell]) * term_exp
            c1 = (t_factors[b, ell] - alpha_power_npm[ell] ** 2) / alpha_power_npm[ell]
            c2 = (4 * alpha_power_npm[ell] ** 2 - t_factors[b, ell]) / (
                2 * alpha_power_npm[ell]
            )
            if c1 >= 0.0 and c2 >= 0.0:
                is_safe = True
            elif c1 >= 0.0:
                is_safe = (c1 + c2) >= 0.0
            elif c2 >= 0.0:
                is_safe = (2.0 * c1 + c2) >= 0.0
            else:
                is_safe = False
            if is_safe:
                w1[b, ell] = v * c1
                w2[b, ell] = v * c2
            else:
                # A clamp may apply: leave zero weight and use the pairwise fallback.
                safe[b, ell] = False

    return w1, w2, safe


@njit(cache=True)  # type: ignore[untyped-decorator]
def _p1_spm_jit(
    n_ch: int,
    n_batch: int,
    p_in: NDArrayFloat,
    b_hz_channels: NDArrayFloat,
    n_span: int,
    f_rel_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    t_factors: NDArrayFloat,
    r_f: NDArrayFloat,
    l_eff_m: NDArrayFloat,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    cr_w_m_hz: float,
    k_bar: float,
) -> NDArrayFloat:
    """Calculate batched SPM using the same per-channel baseline kernel.

    Returns
    -------
    numpy.ndarray
        SPM variance for each candidate and channel, with shape ``(B, N)``.
    """
    out = np.zeros((n_batch, n_ch), dtype=np.float64)
    for b in range(n_batch):
        for i in range(n_ch):
            if p_in[b, i] <= 0:  # The serial baseline skips unpowered channels.
                continue
            gamma_ii = _calc_gamma_coeff_scalar(
                lambdas_m[i], lambdas_m[i], n_2_m2_w, a_eff_m2[i], a_eff_m2[i]
            )
            phi_i = 1.5 * PI**2 * (beta_2_s2_m + 2 * PI * f_rel_hz[i] * beta_3_s3_m)
            out[b, i] = _calc_spm_variance_jit(
                p_in[b, i],
                b_hz_channels[i],
                n_span,
                gamma_ii,
                phi_i,
                alpha_power_npm[i],
                t_factors[b, i],
                r_f[b, i],
                cr_w_m_hz,
                k_bar,
                l_eff_m[i],
            )
    return out


@njit(cache=True)  # type: ignore[untyped-decorator]
def _p1_xpm_clamp_correction_jit(
    n_ch: int,
    n_batch: int,
    p_in: NDArrayFloat,
    b_hz_channels: NDArrayFloat,
    phi_raw: NDArrayFloat,
    gamma_mat: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    t_factors: NDArrayFloat,
    r_f: NDArrayFloat,
    l_eff_m: NDArrayFloat,
    cr_w_m_hz: float,
    k_bar: float,
    safe: NDArrayBool,
) -> NDArrayFloat:
    """Calculate unsafe XPM contributions using the baseline pairwise clamp.

    Returns
    -------
    numpy.ndarray
        Per-candidate, per-channel correction, zero when every weight is safe.
    """
    corr = np.zeros((n_batch, n_ch), dtype=np.float64)
    if np.all(safe):
        return corr

    for b in range(n_batch):
        for i in range(n_ch):
            if p_in[b, i] <= 0:
                continue
            acc = 0.0
            for ell in range(n_ch):
                if ell == i or p_in[b, ell] <= 0 or safe[b, ell]:
                    continue
                acc += _calc_xpm_variance_term_jit(
                    p_in[b, ell],
                    b_hz_channels[ell],
                    b_hz_channels[i],
                    gamma_mat[i, ell],
                    phi_raw[i, ell],
                    alpha_power_npm[ell],
                    t_factors[b, ell],
                    r_f[b, ell],
                    cr_w_m_hz,
                    k_bar,
                    l_eff_m[ell],
                )
            corr[b, i] = acc
    return corr


def _p1_xpm_gemv(
    w1: NDArrayFloat, w2: NDArrayFloat, mats: P1XpmMatrices
) -> NDArrayFloat:
    """Sum XPM terms across interfering channels using batched matrix products.

    Returns
    -------
    numpy.ndarray
        XPM sums with shape ``(B, N)``.
    """
    return w1 @ mats.g1_t + w2 @ mats.g2_t


def calc_span_noise_and_power_p1_matvec(
    power_in_w: NDArrayFloat,
    span_length_m: float,
    f_rel_hz: NDArrayFloat,
    f_abs_hz: NDArrayFloat,
    lambdas_m: NDArrayFloat,
    alpha_power_npm: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    beta_2_s2_m: float,
    beta_3_s3_m: float,
    n_2_m2_w: float,
    nf_lin_channels: NDArrayFloat,
    symbol_rate_hz: float,
    b_hz_channels: NDArrayFloat,
    n_effective_spans: int,
    cr_w_m_hz: float,
    k_bar: float,
    delta_f_co_hz: float,
    mats: P1XpmMatrices,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Evaluate batched P1 span noise and output power using matrix products.

    Parameters
    ----------
    power_in_w : numpy.ndarray
        Candidate launch powers in watts, shape ``(B, N)``; zero is allowed.
    span_length_m : float
        Span length in meters.
    f_rel_hz, f_abs_hz, lambdas_m : numpy.ndarray
        Relative frequencies, absolute frequencies, and wavelengths per channel.
    alpha_power_npm, a_eff_m2 : numpy.ndarray
        Per-channel power attenuation and effective mode area.
    beta_2_s2_m, beta_3_s3_m, n_2_m2_w : float
        Fiber dispersion coefficients and nonlinear refractive index.
    nf_lin_channels : numpy.ndarray
        Linear noise figure for each channel.
    symbol_rate_hz : float
        Symbol rate in Hz, used only for SRS band edges.
    b_hz_channels : numpy.ndarray
        Per-channel bandwidth in Hz, shape ``(N,)``.
    n_effective_spans : int
        Effective span count for NLI accumulation.
    cr_w_m_hz, k_bar, delta_f_co_hz : float
        SRS gain coefficient, launch pre-emphasis factor, and cutoff frequency.
    mats : P1XpmMatrices
        Precomputed pair matrices matching the channel geometry.

    Returns
    -------
    tuple of numpy.ndarray
        Output power and SPM, XPM, and ASE variance arrays, all ``(B, N)``.

    Raises
    ------
    ValueError
        If power, matrix, or channel-bandwidth dimensions do not match.

    Notes
    -----
    The XPM summation order differs from the pairwise baseline; fiber output
    and SPM use the same calculations as the baseline. The returned historical
    ASE estimate is kept for compatibility; callers compute ASE separately.
    """
    p_in = np.ascontiguousarray(power_in_w, dtype=np.float64)
    if p_in.ndim != 2:
        raise ValueError(f"power_in_w must have shape (B, N), got {p_in.shape}.")  # noqa: EM102, TRY003
    n_ch = int(p_in.shape[1])
    if mats.n_ch != n_ch:
        raise ValueError(f"pair matrices are built for N={mats.n_ch}, got N={n_ch}.")  # noqa: EM102, TRY003
    if b_hz_channels.shape != (n_ch,):
        msg = f"b_hz_channels must have shape ({n_ch},), got {b_hz_channels.shape}."
        raise ValueError(msg)

    if p_in.shape[0] == 0:
        empty = np.empty((0, n_ch), dtype=np.float64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    p_out, r_f, t_factors, l_eff_m = _p1_span_pre_nli_batch_jit(
        p_in,
        float(span_length_m),
        f_rel_hz,
        float(symbol_rate_hz),
        alpha_power_npm,
        float(cr_w_m_hz),
        float(k_bar),
        float(delta_f_co_hz),
    )

    sigma2_spm = _p1_spm_jit(
        n_ch,
        int(p_in.shape[0]),
        p_in,
        b_hz_channels,
        int(n_effective_spans),
        f_rel_hz,
        lambdas_m,
        a_eff_m2,
        alpha_power_npm,
        t_factors,
        r_f,
        l_eff_m,
        float(beta_2_s2_m),
        float(beta_3_s3_m),
        float(n_2_m2_w),
        float(cr_w_m_hz),
        float(k_bar),
    )

    w1, w2, safe = _p1_xpm_weights_jit(
        n_ch,
        int(p_in.shape[0]),
        p_in,
        b_hz_channels,
        alpha_power_npm,
        t_factors,
        r_f,
        l_eff_m,
        float(cr_w_m_hz),
        float(k_bar),
    )
    xpm_sum = _p1_xpm_gemv(w1, w2, mats)
    # Add potentially clamped terms using the baseline pairwise kernel.
    # Raw phase and gamma matrices are needed only for this fallback.
    if not bool(safe.all()):
        xpm_sum = xpm_sum + _p1_xpm_clamp_correction_jit(
            n_ch,
            int(p_in.shape[0]),
            p_in,
            b_hz_channels,
            mats.phi_raw,
            mats.gamma,
            alpha_power_npm,
            t_factors,
            r_f,
            l_eff_m,
            float(cr_w_m_hz),
            float(k_bar),
            safe,
        )
    sigma2_xpm = (32.0 / 27.0) * p_in * float(n_effective_spans) * xpm_sum

    # The caller computes ASE separately when applying the EDFA;
    # retain this output for compatibility with the baseline kernel.
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(p_out > 0, p_in / np.where(p_out > 0, p_out, 1.0), 0.0)
    sigma2_ase = np.where(
        (p_in > 0) & (p_out > 0),
        float(n_effective_spans)
        * nf_lin_channels
        * H_PLANCK
        * f_abs_hz
        * np.maximum(gain - 1.0, 0.0)
        * b_hz_channels,
        0.0,
    )

    return p_out, sigma2_spm, sigma2_xpm, sigma2_ase
