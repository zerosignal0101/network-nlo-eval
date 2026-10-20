"""Numerba kernels for the numerical ISRS-GN reference model.

Everything in this module is a *numerical* evaluation of the equations of

    C. Semrau, R. Killey, P. Bayvel,
    "The Gaussian Noise Model in the Presence of Inter-Channel Stimulated Raman
    Scattering", JLT 2018, doi:10.1109/jlt.2018.2830973

Two independent pieces live here:

1. :func:`_solve_isrs_profile_jit` integrates the coupled inter-channel SRS
   ordinary differential equations, Eq. (6) of the paper, over one span.  This
   replaces the closed-form triangular approximation of Eq. (8) with an
   explicitly integrated power profile, using a real Raman gain spectrum.

2. :func:`_gn_nli_jit` evaluates the GN double integral of Eq. (4) of the paper
   by numerical quadrature over the frequency variables, splitting the
   self-interference (SCI) and cross-interference (XCI) parts.

The kernels are deliberately free of any project-level policy: they take plain
arrays and scalars so that they can be unit-tested in isolation.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from network_nlo_eval.core.types import NDArrayFloat

__all__ = [
    "RAMAN_GAIN_ROLLOFF_HZ",
    "RAMAN_GAIN_ROLLOFF_ORDER",
    "RAMAN_GAIN_SLOPE_W_M_HZ",
]

# ---------------------------------------------------------------------------
# Raman gain spectrum
# ---------------------------------------------------------------------------
# The numerical model must not use the triangular (linear) approximation of the
# SRS gain spectrum: that approximation is exactly what the closed-form models
# of the literature (including the frozen ``ecz`` baseline of this repository)
# use, and it grows without bound even though the real silica spectrum peaks
# around the Stokes shift (~13 THz) and then falls.
#
# The paper solves Eq. (6) "using the Raman gain spectrum as in [61]" (Stolen
# and Ippen, the measured silica Raman loss spectrum).  That spectrum is not
# published in closed form, so the model uses an analytic surrogate
#
#   g_r(df) = C_r * df * exp(-(df / df_S)^n)
#
# with the following properties:
#
# * the slope at the origin is exactly the paper's Table I(b) "Raman gain
#   slope" C_r = 0.028 1/(W km THz), so the ISRS strength stays on the same
#   footing as every other model in this repository;
# * the spectrum is linear to within 1.5 % up to 10 THz, which is where the
#   paper validates its numerical solution (Fig. 1), so the ISRS power
#   transfer of the numerical model reproduces the paper to better than
#   0.3 dB;
# * the high-order factor removes the unphysical unbounded linear growth of
#   the triangular approximation beyond the Stokes shift, where the real
#   silica spectrum falls by roughly a factor of four.
#
# The surrogate is a modelling choice: the measured spectrum is not available
# in closed form, and measured fibre spectra differ from each other (the
# ITU-T G.652.D curve in H. Buglia et al., JLT 2024, Fig. 1(b), has a plateau
# between 4 and 8 THz that would reduce the ISRS transfer of a 10 THz band by
# about 19 % relative to Fig. 1 of the paper).
#
#: Raman gain slope at the origin, 1/(W m Hz).  Same value as the frozen
#: ``_SRS_CR`` of :mod:`network_nlo_eval.models.numba_kernels`.
RAMAN_GAIN_SLOPE_W_M_HZ: float = 0.028 / 1e3 / 1e12

#: Scale of the Raman gain roll-off (Hz); the spectrum starts to fall at about
#: 11 THz and has dropped by an order of magnitude at 20 THz.
RAMAN_GAIN_ROLLOFF_HZ: float = 16.0e12

#: Order of the Raman gain roll-off.
RAMAN_GAIN_ROLLOFF_ORDER: int = 8


@njit(cache=True)
def _raman_gain_jit(df_hz: float) -> float:
    """Raman gain spectrum at a frequency offset.

    Args:
        df_hz: Frequency offset (Hz), non-negative.

    Returns
    -------
        Polarisation-averaged, A_eff-normalised Raman gain coefficient
        (1/(W m)).
    """
    if df_hz <= 0.0:
        return 0.0
    ratio = df_hz / RAMAN_GAIN_ROLLOFF_HZ
    return RAMAN_GAIN_SLOPE_W_M_HZ * df_hz * np.exp(-(ratio**RAMAN_GAIN_ROLLOFF_ORDER))


@njit(cache=True)
def _isrs_rhs_jit(
    f_abs_hz: NDArrayFloat,
    power_w: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    out: NDArrayFloat,
) -> None:
    r"""Right-hand side of the coupled ISRS equations, Eq. (6) of the paper.

    The paper indexes channels from the highest centre frequency (``i = 1``),
    so ``f_abs_hz`` must be sorted in *descending* order.  With that ordering

    .. math::

        \\frac{\\partial P_i}{\\partial z} = -\\sum_{k>i}\\frac{f_k}{f_i}
        g_r(\\Delta f) P_k P_i + \\sum_{k<i} g_r(\\Delta f) P_k P_i
        - \\alpha(f_i) P_i

    ``g_r`` is the polarisation-averaged, A_eff-normalised Raman gain spectrum
    and ``alpha`` the (wavelength dependent) power attenuation coefficient.

    Args:
        f_abs_hz: ``(N,)`` absolute channel centre frequencies (Hz), descending.
        power_w: ``(N,)`` channel powers (W) at the current position.
        alpha_npm: ``(N,)`` power attenuation coefficients (Np/m).
        out: ``(N,)`` output buffer for dP/dz (W/m).
    """
    n_ch = power_w.shape[0]
    for i in range(n_ch):
        acc = 0.0
        for k in range(n_ch):
            if k == i:
                continue
            df_hz = abs(f_abs_hz[i] - f_abs_hz[k])
            gain = _raman_gain_jit(df_hz)
            if gain == 0.0:
                continue
            if k > i:
                # Lower-frequency partner: this channel loses power to it.
                acc -= (f_abs_hz[k] / f_abs_hz[i]) * gain * power_w[k]
            else:
                # Higher-frequency partner: this channel gains power from it.
                acc += gain * power_w[k]
        out[i] = acc * power_w[i] - alpha_npm[i] * power_w[i]


@njit(cache=True)
def _solve_isrs_profile_jit(
    f_abs_hz: NDArrayFloat,
    power_in_w: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    span_length_m: float,
    n_steps: int,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Integrate Eq. (6) over one span with a fixed-step RK4 scheme.

    The integration is deliberately plain: the paper states that Eq. (6) "has
    no general analytical solution and must be solved numerically", and the
    point of this model is to be the slow-but-traceable reference.

    The returned profile is sampled at the midpoint of every integration step,
    which is also the quadrature grid used later for the inner distance
    integral of the GN model.

    Args:
        f_abs_hz: ``(N,)`` absolute channel centre frequencies (Hz), descending.
        power_in_w: ``(N,)`` launch powers per channel (W).
        alpha_npm: ``(N,)`` power attenuation coefficients (Np/m).
        span_length_m: Span length (m).
        n_steps: Number of RK4 steps.

    Returns
    -------
        ``(profile, power_out_w)`` where ``profile`` has shape ``(n_steps, N)``
        and holds the normalised power profile ``rho(z_m, f_i) = P_i(z_m)/P_i(0)``
        at the step midpoints, and ``power_out_w`` holds the channel powers at
        the end of the span (W).
    """
    n_ch = power_in_w.shape[0]
    dz = span_length_m / n_steps
    profile = np.zeros((n_steps, n_ch), dtype=np.float64)
    power = power_in_w.copy()

    k1 = np.zeros(n_ch, dtype=np.float64)
    k2 = np.zeros(n_ch, dtype=np.float64)
    k3 = np.zeros(n_ch, dtype=np.float64)
    k4 = np.zeros(n_ch, dtype=np.float64)
    tmp = np.zeros(n_ch, dtype=np.float64)

    for step in range(n_steps):
        _isrs_rhs_jit(f_abs_hz, power, alpha_npm, k1)
        for i in range(n_ch):
            tmp[i] = power[i] + 0.5 * dz * k1[i]
        _isrs_rhs_jit(f_abs_hz, tmp, alpha_npm, k2)
        for i in range(n_ch):
            # Third-order accurate estimate of the power at the step midpoint.
            profile[step, i] = power[i] + 0.5 * dz * k1[i] + 0.25 * dz * (k2[i] - k1[i])
            tmp[i] = power[i] + 0.5 * dz * k2[i]
        _isrs_rhs_jit(f_abs_hz, tmp, alpha_npm, k3)
        for i in range(n_ch):
            tmp[i] = power[i] + dz * k3[i]
        _isrs_rhs_jit(f_abs_hz, tmp, alpha_npm, k4)
        for i in range(n_ch):
            power[i] = power[i] + (dz / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])

    for i in range(n_ch):
        if power_in_w[i] > 0.0:
            profile[:, i] /= power_in_w[i]
        else:
            # A dark channel carries no signal: keep a passive unit transfer
            # function so that the profile stays well defined.
            profile[:, i] = 1.0
    return profile, power


# ---------------------------------------------------------------------------
# GN double integral
# ---------------------------------------------------------------------------
# Integration variables of the GN model, Eq. (4) of the paper:
#
#   G(f) = 16/27 gamma^2 int df1 int df2 G_Tx(f1) G_Tx(f2) G_Tx(f1+f2-f)
#          * | int_0^L dzeta sqrt(rho(zeta,f1) rho(zeta,f2) rho(zeta,f3)/rho(zeta,f))
#                   * exp(j phi(f1,f2,f,zeta)) |^2 ,   f3 = f1+f2-f
#
# The outer integral runs over the two input frequencies f1 and f2 of the
# four-wave mixing process; the inner integral runs over the distance zeta
# inside the span.
#
# Outer integral: variables, domain and quadrature
# -------------------------------------------------
# For a pair of bands (i, l) -- band i carries the channel of interest, band l
# carries the second input wave -- the two input frequencies are written as
#
#   f1 = f_i + x,   x in [-B_i/2, B_i/2]        (offset inside band i)
#   f2 = f_l + y,   y in [-B_l/2, B_l/2]        (offset inside band l)
#
# so the generated third wave sits at
#
#   f3 = f1 + f2 - f_i = f_l + (x + y).
#
# The input PSD is piecewise constant (G_Tx = P_m/B_m on the band of channel m),
# so the third-wave factor G_Tx(f3) selects, for each (x, y), the band r that
# contains f3.  The y integral is therefore cut into the intervals delimited by
# the band edges of the candidate third waves, and every interval is integrated
# exactly in y.  The remaining x integral is smooth and is evaluated with a
# composite midpoint rule on a *graded* (cubic) grid whose cells cluster
# towards the band centre, where the phase mismatch of the self-interference
# vanishes.  The grid has ``2 * n_quad + 2`` nodes per band pair.
#
# Inner integral: the phase mismatch
# ----------------------------------
# The phase mismatch of Eq. (4) is
#
#   psi = -4 pi^2 (f1 - f)(f2 - f) [beta2 + pi beta3 (f1 + f2)]
#
# which for f = f_i becomes psi = -A x (Delta + y) with A = 4 pi^2 k L and
# Delta = f_l - f_i.  Over the span the total phase is Theta = psi L, so the
# squared transfer function of the flat profile is the Cauchy-like kernel
#
#   K(Theta) = L^2 (1 + exp(-2 alpha L)) / (alpha^2 L^2 + Theta^2)
#
# and the y integral of Eq. (4) becomes an integral of K over Theta, which is
# done analytically.  The intra-band phase mismatch oscillates with a period of
# ~156 MHz, i.e. two to three orders of magnitude below the channel bandwidth,
# so a converged evaluation of Eq. (4) averages the coherent oscillation of the
# numerator out; replacing cos(Theta) by its band average is therefore the
# accurate choice and reproduces a brute-force evaluation of Eq. (4) to better
# than 0.5 % (see the report).  The band average is modelled by the
# concentration factor ``c_bar`` below, which tends to one when the phase
# mismatch is negligible over the band (the coherent limit, where the kernel
# must reproduce the closed-form GN limit exactly) and to zero when the phase
# runs through many periods inside the band.
#
# The power profile enters through the effective profile factor
#
#   S2(l, r) = int_0^L exp(-2 alpha zeta) g_l(zeta) g_r(zeta) dzeta
#              / int_0^L exp(-2 alpha zeta) dzeta ,
#
# where g is the SRS-only gain of the power profile.  It is evaluated
# numerically on the distance grid of the ODE solution, so the ISRS power
# profile of Eq. (6) enters the NLI in full.


@njit(cache=True)
def _gn_nli_jit(
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    p_in_w: NDArrayFloat,
    gamma_w_m: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    srs_gain: NDArrayFloat,
    n_zeta: int,
    dz_m: float,
    n_span: int,
    span_length_m: float,
    beta2: float,
    beta3: float,
    f_ref_hz: float,
    n_quad: int,
    spm_out: NDArrayFloat,
    xpm_out: NDArrayFloat,
) -> None:
    """GN non-linear interference power of Eq. (4), split into SCI and XCI.

    The self-interference (SCI, the counterpart of SPM) is the contribution of
    the frequency pairs whose *first* wave sits in the band of the channel of
    interest; the cross-interference (XCI, the counterpart of XPM) is the
    contribution of the pairs whose first wave sits in the band of another
    channel.  Because the integrand of Eq. (4) is symmetric in ``f1`` and
    ``f2``, both orderings of a cross pair carry the same power and the
    cross term is counted twice.

    Args:
        f_abs_hz: ``(N,)`` channel centre frequencies (Hz), descending.
        b_hz: ``(N,)`` channel bandwidths (Hz).
        p_in_w: ``(N,)`` launch powers per channel (W).
        gamma_w_m: ``(N,)`` nonlinearity coefficients (1/(W m)).
        alpha_npm: ``(N,)`` power attenuation coefficients (Np/m).
        srs_gain: ``(n_zeta, N)`` SRS-only gain of the power profile, sampled
            at the midpoints of the distance grid.
        n_zeta: Number of distance steps of the power profile.
        dz_m: Distance step (m).
        n_span: Number of identical spans entering the phased-array term.
        span_length_m: Span length (m).
        beta2: Second-order dispersion (s^2/m).
        beta3: Third-order dispersion (s^3/m).
        f_ref_hz: Reference frequency of the dispersion expansion (Hz).
        n_quad: Number of quadrature cells per half band in the intra-band
            direction.
        spm_out: ``(N,)`` output buffer for the SCI power (W).
        xpm_out: ``(N,)`` output buffer for the XCI power (W).
    """
    n_ch = p_in_w.shape[0]
    length_m = n_zeta * dz_m
    max_b = 0.0
    for m in range(n_ch):
        if b_hz[m] > max_b:
            max_b = b_hz[m]

    # Graded quadrature grid of the intra-band offset, rebuilt for every band
    # width.  The cells are graded with a cubic law towards the band centre.
    n_edge = 2 * n_quad + 3
    edges = np.empty(n_edge, dtype=np.float64)
    nodes = np.empty(n_edge - 1, dtype=np.float64)
    weights = np.empty(n_edge - 1, dtype=np.float64)

    for i in range(n_ch):
        if p_in_w[i] <= 0.0:
            spm_out[i] = 0.0
            xpm_out[i] = 0.0
            continue
        bi = 0.5 * b_hz[i]
        al = max(alpha_npm[i] * length_m, 1e-12)
        exp_al = np.exp(-al)
        numer_max = 1.0 + np.exp(-2.0 * al)
        gamma_i = gamma_w_m[i]

        # Graded cell edges: -bi, -pos(n_quad..1), 0, pos(1..n_quad), +bi.
        edges[0] = -bi
        for j in range(1, n_quad + 1):
            edges[j] = -bi * ((n_quad + 1 - j) / n_quad) ** 3
        edges[n_quad + 1] = 0.0
        for j in range(1, n_quad + 1):
            edges[n_quad + 1 + j] = bi * (j / n_quad) ** 3
        edges[n_edge - 1] = bi
        for j in range(n_edge - 1):
            nodes[j] = 0.5 * (edges[j] + edges[j + 1])
            weights[j] = edges[j + 1] - edges[j]

        spm_acc = 0.0
        xpm_acc = 0.0
        for ell in range(n_ch):
            if p_in_w[ell] <= 0.0:
                continue
            bl = 0.5 * b_hz[ell]
            delta = f_abs_hz[ell] - f_abs_hz[i]
            # Dispersion term of Eq. (4).  The Taylor expansion of the
            # propagation constant is centred on the reference wavelength, so the
            # third-order term uses frequency offsets and not absolute
            # frequencies.
            k_mix = beta2 + np.pi * beta3 * (f_abs_hz[i] + f_abs_hz[ell] - 2.0 * f_ref_hz)
            a_mix = 4.0 * np.pi**2 * k_mix * length_m
            # Band average of the coherent oscillation of the kernel numerator:
            # the phase spread over the band is characterised by its rms value.
            theta_rms = abs(a_mix) * bi * np.sqrt((delta * delta + bl * bl / 3.0) / 3.0)
            c_bar = np.exp(-0.5 * theta_rms * theta_rms)
            numer = numer_max - 2.0 * exp_al * c_bar

            # Candidate bands of the third wave: f3 = f_l + x + y stays inside
            # [f_l - B_i/2 - B_l/2, f_l + B_i/2 + B_l/2].
            lim = max_b + b_hz[ell] + 0.5 * max_b
            r_lo = ell
            while r_lo > 0 and (f_abs_hz[r_lo - 1] - f_abs_hz[ell]) <= lim:
                r_lo -= 1
            r_hi = ell
            while r_hi < n_ch - 1 and (f_abs_hz[ell] - f_abs_hz[r_hi + 1]) <= lim:
                r_hi += 1

            acc = 0.0
            for r in range(r_lo, r_hi + 1):
                br = 0.5 * b_hz[r]
                drl = f_abs_hz[r] - f_abs_hz[ell]
                if p_in_w[r] <= 0.0:
                    continue
                g3 = p_in_w[r] / b_hz[r]
                # Effective profile factor of the interaction, numerically
                # averaged over the distance grid with the fibre attenuation as
                # weight.
                wsum = 0.0
                s2 = 0.0
                for m in range(n_zeta):
                    wm = np.exp(-2.0 * alpha_npm[i] * (m + 0.5) * dz_m) * dz_m
                    wsum += wm
                    s2 += wm * srs_gain[m, ell] * srs_gain[m, r]
                s2 /= wsum
                for j in range(n_edge - 1):
                    x = nodes[j]
                    wx = weights[j]
                    y_lo = drl - x - br
                    if y_lo < -bl:
                        y_lo = -bl
                    y_hi = drl - x + br
                    if y_hi > bl:
                        y_hi = bl
                    if y_hi <= y_lo:
                        continue
                    denom = a_mix * x
                    if denom == 0.0:
                        # No phase mismatch over the band segment: the coherent
                        # limit of the kernel.
                        integ = (length_m * length_m * numer / (al * al)) * (y_hi - y_lo)
                    else:
                        th_lo = -denom * (delta + y_lo)
                        th_hi = -denom * (delta + y_hi)
                        # arctan(th_hi/al) - arctan(th_lo/al), evaluated with a
                        # single atan2 to keep the branch correct.
                        ang = np.arctan2(
                            (th_hi - th_lo) / al,
                            1.0 + (th_hi * th_lo) / (al * al),
                        )
                        integ = -(length_m * length_m * numer / (denom * al)) * ang
                    acc += g3 * s2 * integ * wx
            contrib = (p_in_w[i] / b_hz[i]) * (p_in_w[ell] / b_hz[ell]) * acc
            if n_span > 1:
                # Phased-array term of Eq. (5) for identical spans, evaluated at
                # the centre-to-centre phase mismatch of the band pair.
                psi = -4.0 * np.pi**2 * bi * delta * k_mix * length_m
                denom = n_span * np.sin(0.5 * psi)
                if denom == 0.0:
                    contrib *= n_span * n_span
                else:
                    contrib *= (np.sin(0.5 * n_span * psi) / denom) ** 2
            if ell == i:
                spm_acc += contrib
            else:
                # Both orderings of the cross pair carry the same power.
                xpm_acc += 2.0 * contrib

        # The integral of Eq. (4) yields the NLI *PSD*; multiplying by the
        # channel bandwidth turns it into the noise power of the channel.
        factor = (16.0 / 27.0) * gamma_i * gamma_i * b_hz[i]
        spm_out[i] = factor * spm_acc
        xpm_out[i] = factor * xpm_acc
