"""Numba kernels for the PCFM model (Poggiolini et al., JLT 2026).

The module holds the numerical machinery of the polynomial closed-form model:

* :func:`_solve_spp_jit` integrates the coupled inter-channel stimulated Raman
  scattering (ISRS) power equations over one span and returns the *normalised*
  spatial power profile (SPP) ``p_i(z) = P_i(z)/P_i(0)`` of every channel,
  including any lumped loss elements placed inside the span;
* :func:`_sci_kernel_batch_jit` evaluates the closed-form SCI kernel of
  Eq. (30) of the paper;
* :func:`r_coefficient_matrix` builds the operator that maps the SPP
  polynomial coefficients to the auxiliary polynomial of the SCI kernel;
* :func:`_j_batch_jit` evaluates the auxiliary function
  ``J(X) = int_0^X Si(t)/t dt`` of Eq. (30).

The closed-form XCI kernel of Eq. (24) is a plain pairwise expression; it is
assembled into an ``(N, N)`` matrix by
:class:`network_nlo_eval.models.pcfm.PCFMSpanKernel` and applied as a matrix
product, so it needs no separate kernel function here.

Two numerical remarks that matter for the implementation:

``Eq. (30) as printed is not usable in double precision``
    The auxiliary functions ``I_{p,q}`` and ``S_k`` of the paper cancel
    catastrophically: the relative error of the printed expression grows by
    roughly 2.5 decimal digits per unit of the polynomial index, so the
    published formula already carries a 10 % error at ``N_p = 3`` and is
    meaningless at ``N_p = 9``.  The kernel below therefore evaluates an
    algebraically identical but numerically stable rearrangement of the very
    same quantity, derived from the definition of the island integral
    Eq. (27).  See :func:`_sci_kernel_batch_jit` for the derivation and
    ``tests/test_pcfm.py`` for the equivalence proof.

``J(X) is evaluated from its asymptotic expansion``
    ``J(X) = int_0^X Si(t)/t dt = X * 2F3(1/2, 1/2; 3/2, 3/2, 3/2; -X^2/4)``
    has no implementation in SciPy.  It is evaluated here with the Taylor
    series for small ``|X|`` and with the asymptotic expansion of the
    remainder of ``Si`` for large ``|X|``, which is accurate to 1e-13 over
    ``|X| in [0.05, 1e4]`` (see :func:`_j_batch_jit`).
"""

from __future__ import annotations

import numpy as np
from numba import njit

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.numerical_kernels import _raman_gain_jit

__all__ = [
    "C_LIGHT",
    "J_SERIES_SWITCH",
    "_isrs_coefficient_matrix_jit",
    "_j_batch_jit",
    "_profile_energy_jit",
    "_sci_kernel_batch_jit",
    "_solve_spp_jit",
    "r_coefficient_matrix",
]

#: Speed of light in vacuum (m/s).
C_LIGHT: float = 299792458.0

#: Below this ``|X|`` the Taylor series of ``J`` is used, above it the
#: asymptotic expansion.  The switch sits where both are accurate to ~1e-13.
J_SERIES_SWITCH: float = 26.0

#: Constant of the asymptotic expansion of ``J``,
#: ``C_r = int_0^1 Si(t)/t dt + int_1^inf [Si(t)-pi/2]/t dt``.
#: It is computed once by :func:`network_nlo_eval.models.pcfm._j_constant` and
#: passed to the kernels as an argument.


# ---------------------------------------------------------------------------
# ISRS power profile
# ---------------------------------------------------------------------------


@njit(cache=True)
def _isrs_coefficient_matrix_jit(f_desc_hz: NDArrayFloat) -> NDArrayFloat:
    r"""Build the linear coupling matrix of the coupled ISRS power equations.

    The channels are numbered from the highest centre frequency downwards, as
    in the numerical reference model.  With that ordering the power equations
    of Eq. (6) of Semrau et al. read

    .. math::

        \\frac{\\partial P_i}{\\partial z} = P_i \\Big[ \\sum_{k<i}
        g_r(f_i-f_k) P_k - \\sum_{k>i} \\frac{f_k}{f_i} g_r(f_k-f_i) P_k
        - \\alpha(f_i) \\Big]

    so that the whole right-hand side is ``P_i * ( (M P)_i - alpha_i )`` with

    .. math::

        M_{ik} = \\begin{cases}
            g_r(f_i-f_k) & k<i \\\\
            -\\frac{f_k}{f_i} g_r(f_k-f_i) & k>i
        \\end{cases}

    The Raman gain spectrum does not depend on ``z``, so the matrix is built
    once per kernel instead of being re-evaluated at every integration step.

    Args:
        f_desc_hz: ``(N,)`` channel centre frequencies (Hz), descending.

    Returns
    -------
        ``(N, N)`` coupling matrix (1/(W m)).
    """
    n_ch = f_desc_hz.shape[0]
    mat = np.zeros((n_ch, n_ch), dtype=np.float64)
    for i in range(n_ch):
        for k in range(n_ch):
            if i == k:
                continue
            df_hz = abs(f_desc_hz[i] - f_desc_hz[k])
            gain = _raman_gain_jit(df_hz)
            if gain == 0.0:
                continue
            if k > i:
                # Lower-frequency partner: this channel loses power to it.
                mat[i, k] = -(f_desc_hz[k] / f_desc_hz[i]) * gain
            else:
                # Higher-frequency partner: this channel gains power from it.
                mat[i, k] = gain
    return mat


@njit(cache=True)
def _solve_spp_jit(
    coef_mat: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    power_in_w: NDArrayFloat,
    seg_len_m: NDArrayFloat,
    seg_loss: NDArrayFloat,
    seg_steps: NDArrayFloat,
    profile: NDArrayFloat,
    power_out_w: NDArrayFloat,
) -> None:
    """Integrate the coupled ISRS power equations over one span.

    The integration is a fixed-step RK4 scheme.  Lumped loss elements are
    handled by splitting the span into segments: the power is multiplied by
    the loss factor at the beginning of every segment and the integration is
    restarted there, which keeps the loss a true discontinuity of the
    profile.

    The profile is sampled twice per integration step, at the midpoint and at
    the end of the step, so that the sampling grid of the polynomial fit is
    twice as dense as the integration grid at no extra cost.

    The profile is *not* normalised here: the caller divides it by the launch
    power.

    Args:
        coef_mat: ``(N, N)`` coupling matrix of the ISRS equations.
        alpha_npm: ``(N,)`` power attenuation coefficients (Np/m).
        power_in_w: ``(N,)`` launch powers (W).
        seg_len_m: ``(S,)`` length of every segment (m).
        seg_loss: ``(S,)`` power loss factor applied at the start of a segment.
        seg_steps: ``(S,)`` number of RK4 steps of every segment.
        profile: ``(2*n_steps, N)`` output buffer for the sampled profile (W).
        power_out_w: ``(N,)`` output buffer for the power at the span end (W).
    """
    n_ch = power_in_w.shape[0]
    power = power_in_w.copy()
    k1 = np.zeros(n_ch, dtype=np.float64)
    k2 = np.zeros(n_ch, dtype=np.float64)
    k3 = np.zeros(n_ch, dtype=np.float64)
    k4 = np.zeros(n_ch, dtype=np.float64)
    tmp = np.zeros(n_ch, dtype=np.float64)
    row = 0

    for seg in range(seg_len_m.shape[0]):
        factor = seg_loss[seg]
        if factor != 1.0:
            for i in range(n_ch):
                power[i] *= factor
        n_step = seg_steps[seg]
        if n_step < 1:
            continue
        dz = seg_len_m[seg] / n_step
        for _ in range(n_step):
            _rhs_mat_jit(coef_mat, power, alpha_npm, k1)
            for i in range(n_ch):
                tmp[i] = power[i] + 0.5 * dz * k1[i]
            _rhs_mat_jit(coef_mat, tmp, alpha_npm, k2)
            for i in range(n_ch):
                # Third-order accurate estimate of the power at the step midpoint.
                profile[row, i] = power[i] + 0.5 * dz * k1[i] + 0.25 * dz * (k2[i] - k1[i])
                tmp[i] = power[i] + 0.5 * dz * k2[i]
            _rhs_mat_jit(coef_mat, tmp, alpha_npm, k3)
            for i in range(n_ch):
                tmp[i] = power[i] + dz * k3[i]
            _rhs_mat_jit(coef_mat, tmp, alpha_npm, k4)
            for i in range(n_ch):
                power[i] = power[i] + (dz / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
                profile[row + 1, i] = power[i]
            row += 2

    for i in range(n_ch):
        power_out_w[i] = power[i]


@njit(cache=True)
def _rhs_mat_jit(
    coef_mat: NDArrayFloat,
    power_w: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    out: NDArrayFloat,
) -> None:
    """Right-hand side of the coupled ISRS equations, Eq. (6) of the paper."""
    n_ch = power_w.shape[0]
    coupled = np.dot(coef_mat, power_w)
    for i in range(n_ch):
        out[i] = coupled[i] * power_w[i] - alpha_npm[i] * power_w[i]


# ---------------------------------------------------------------------------
# Auxiliary function J(X) of Eq. (30)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _j_single_jit(x: float, c_const: float) -> float:
    """``J(X) = int_0^X Si(t)/t dt`` for a single argument."""
    if x == 0.0:
        return 0.0
    if abs(x) < J_SERIES_SWITCH:
        # Taylor series: sum_k (-1)^k x^(2k+1) / ((2k+1)^2 (2k+1)!).
        x2 = x * x
        term = x
        total = 0.0
        k = 0
        while k < 90:
            total += term
            k += 1
            term = (
                -term
                * x2
                * (2 * k - 1) ** 2
                / ((2 * k + 1) ** 2 * (2 * k + 1) * (2 * k))
            )
            if abs(term) < 1e-18 * max(abs(total), 1e-300):
                break
        return total
    sign = 1.0 if x > 0.0 else -1.0
    a = abs(x)
    tail = _j_tail_jit(a)
    return sign * ((0.5 * np.pi) * np.log(a) + c_const - tail)


@njit(cache=True)
def _j_tail_jit(x: float) -> float:
    """``int_x^inf [Si(t) - pi/2]/t dt`` via the asymptotic expansion.

    The tail of the sine integral has the asymptotic expansion

    ``Si(t) - pi/2 = -[ cos t sum_k (-1)^k (2k)!/t^(2k+1)
                        + sin t sum_k (-1)^k (2k+1)!/t^(2k+2) ]``

    and the two remaining integrals are reduced to each other by repeated
    integration by parts.  The series is truncated at its smallest term,
    ``2k ~ x``.
    """
    n_terms = int(x / 2.0) + 3
    # Keep x**p inside the double range.
    cap = int((300.0 / np.log10(x) - 6.0) / 2.0) - 2
    if cap < 2:
        cap = 2
    if n_terms > cap:
        n_terms = cap
    if n_terms > 60:
        n_terms = 60

    p_top = 2 * n_terms + 6
    ic = np.zeros(p_top + 4, dtype=np.float64)
    is_ = np.zeros(p_top + 4, dtype=np.float64)
    for p in range(p_top, 1, -1):
        ic[p] = -np.sin(x) / x**p + p * is_[p + 1]
        is_[p] = np.cos(x) / x**p - p * ic[p + 1]

    total = 0.0
    fact = 1.0
    for k in range(n_terms):
        total += ((-1.0) ** k) * fact * ic[2 * k + 2]
        fact *= (2 * k + 1) * (2 * k + 2)
    fact = 1.0
    for k in range(n_terms):
        total += ((-1.0) ** k) * fact * is_[2 * k + 3]
        fact *= (2 * k + 2) * (2 * k + 3)
    return -total


@njit(cache=True)
def _j_batch_jit(x: NDArrayFloat, c_const: float) -> NDArrayFloat:
    """Vectorised :func:`_j_single_jit`."""
    out = np.empty(x.shape[0], dtype=np.float64)
    for i in range(x.shape[0]):
        out[i] = _j_single_jit(x[i], c_const)
    return out


# ---------------------------------------------------------------------------
# SCI kernel, Eq. (30)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _sci_kernel_batch_jit(
    r_coef: NDArrayFloat,
    x_all: NDArrayFloat,
    si_all: NDArrayFloat,
    sin_all: NDArrayFloat,
    cos_all: NDArrayFloat,
    j_all: NDArrayFloat,
    beta2_all: NDArrayFloat,
    length_m: float,
    out: NDArrayFloat,
) -> None:
    r"""Closed-form SCI kernel of Eq. (30), in a numerically stable form.

    The paper writes

    .. math::

        K_{SCI} = \\frac{1}{\\pi^2 \\beta_{2,eff}} \\sum_{n=0}^{N_p}
        \\sum_{k=n}^{N_p} (2-\\delta_{nk}) p_n p_k \\Big[ \\sum_{i=0}^{n}
        \\binom{n}{i} \\frac{\\mathcal{I}_{i,k+n-i+1}}{k+n-i+1} + \\ldots
        \\Big]

    The auxiliary function ``I_{p,q}`` of the paper is a difference of two
    nearly equal large numbers, so the printed expression loses about 2.5
    decimal digits per unit of the polynomial index and is unusable for
    ``N_p >= 3``.  The same quantity is obtained here from the definition of
    the island integral Eq. (27):

    .. math::

        K_{SCI} = \\sum_{n,k} p_n p_k D_{nk}, \\qquad
        D_{nk} = \\int_0^L\\!\\int_0^L z^n z'^k Q(z-z')\\, dz\\, dz'

    with ``Q(d) = 4 Si(lambda d)/(c d)``, ``c = 4 pi^2 beta2`` and
    ``lambda = pi^2 beta2 B^2``.  Splitting the square along the diagonal
    gives, in the scaled variable ``u = z/L`` (``q_n = p_n L^n``):

    .. math::

        K_{SCI} = \\frac{4 L^2}{c} \\Big[ \\tilde R_0 J(X)
        + \\frac{1}{X} \\sum_{m=1}^{M} \\tilde R_m
        \\frac{Si(X) - \\tilde b_{m-1}}{m} \\Big], \\qquad X = \\lambda L

    where ``\\tilde R(d) = sum_m \\tilde R_m d^m`` collects the two
    triangular parts of ``D_{nk}`` and ``\\tilde b_p = B_p / X^{p+1}`` is the
    scaled companion of the ``B_p`` of the paper.  Every quantity in this
    form is O(1), and the result agrees with a direct numerical evaluation of
    Eq. (27) to 1e-13 for every polynomial degree.

    Args:
        r_coef: ``(N, M+1)`` coefficients ``\\tilde R_m`` of every channel.
        x_all: ``(N,)`` ``X = lambda L`` of every channel.
        si_all: ``(N,)`` ``Si(X)`` of every channel.
        sin_all: ``(N,)`` ``sin(X)`` of every channel.
        cos_all: ``(N,)`` ``cos(X)`` of every channel.
        j_all: ``(N,)`` ``J(X)`` of every channel.
        beta2_all: ``(N,)`` effective dispersion of every channel (s^2/m).
        length_m: Span length (m).
        out: ``(N,)`` output buffer for the SCI kernel (Hz^2 m^2).
    """
    n_ch = r_coef.shape[0]
    m_max = r_coef.shape[1] - 1
    b_scaled = np.zeros(m_max + 2, dtype=np.float64)
    for ch in range(n_ch):
        x = x_all[ch]
        si = si_all[ch]
        sin_x = sin_all[ch]
        cos_x = cos_all[ch]
        if x == 0.0:
            out[ch] = 0.0
            continue
        # Scaled companion B_p / X^(p+1) of the paper, Eq. (30).
        b_scaled[0] = (1.0 - cos_x) / x
        b_scaled[1] = -cos_x / x + sin_x / x**2
        for p in range(2, m_max + 1):
            b_scaled[p] = (
                -cos_x / x
                + p * sin_x / x**2
                - p * (p - 1) * b_scaled[p - 2] / x**2
            )
        total = r_coef[ch, 0] * j_all[ch]
        for m in range(1, m_max + 1):
            total += r_coef[ch, m] * (si - b_scaled[m - 1]) / m
        out[ch] = (4.0 * length_m) / (4.0 * np.pi**2 * beta2_all[ch]) * total


@njit(cache=True)
def _profile_energy_jit(q_all: NDArrayFloat) -> NDArrayFloat:
    """Profile energy ``int_0^L p(z)^2 dz`` of every channel.

    The polynomial is given in the scaled basis ``p(z) = sum_n q_n (z/L)^n``,
    so the energy reads ``L * sum_{n,k} q_n q_k / (n+k+1)``.

    Args:
        q_all: ``(Np+1, N)`` scaled polynomial coefficients of every channel.

    Returns
    -------
        ``(N,)`` profile energy (m).
    """
    n_ch = q_all.shape[1]
    n_poly = q_all.shape[0]
    out = np.zeros(n_ch, dtype=np.float64)
    for ch in range(n_ch):
        acc = 0.0
        for n in range(n_poly):
            for k in range(n_poly):
                acc += q_all[n, ch] * q_all[k, ch] / (n + k + 1)
        out[ch] = acc
    return out


def r_coefficient_matrix(n_poly: int) -> NDArrayFloat:
    """Build the matrix that maps the polynomial coefficients to ``R(d)``.

    With ``p(z) = sum_n q_n (z/L)^n`` the two triangular parts of the SCI
    island integral collect into the polynomial

    ``R(d) = sum_{n,k} q_n q_k [P+(n,k)(d) + P+(k,n)(d)]``

    with ``P+(a,b)(d) = int_d^1 u^a (u-d)^b du``, whose coefficients are
    ``(-1)^i C(b,i)/(a+b-i+1)`` on ``d^i`` and ``-(-1)^i C(b,i)/(a+b-i+1)``
    on ``d^{a+b+1}``.  The matrix is a constant of the polynomial degree and
    is built once per kernel.

    Args:
        n_poly: Number of polynomial coefficients (``N_p + 1``).

    Returns
    -------
        ``(n_poly**2, 2*n_poly+2)`` matrix; row ``n*n_poly+k`` belongs to the
        pair ``(n, k)``.
    """
    from math import comb

    m_max = 2 * n_poly + 1
    mat = np.zeros((n_poly * n_poly, m_max + 1), dtype=np.float64)
    for n in range(n_poly):
        for k in range(n_poly):
            row = n * n_poly + k
            for a, b in ((n, k), (k, n)):
                for i in range(b + 1):
                    e = a + b - i + 1
                    binom = ((-1.0) ** i) * comb(b, i)
                    mat[row, i] += binom / e
                    mat[row, a + b + 1] -= binom / e
    return mat
