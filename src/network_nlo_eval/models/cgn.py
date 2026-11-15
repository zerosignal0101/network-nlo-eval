"""Correlation-based Gaussian-noise model for multi-span optical systems.

The nonlinear stimulus profile (NSP) energy spectrum reduces the GN frequency
integrals to one-dimensional correlation integrals over rectangular covers of
integration islands. The model supports an attenuation-only NSP and a
segmented ISRS profile using the repository's Raman power approximation.
Fused kernels and independent channel chunks accelerate core construction.

``band_points=1`` evaluates center-frequency NLI PSD and multiplies by
bandwidth. Larger values integrate the PSD across each channel. Both modes
include SCI, XCI and MCI, including at zero second-order dispersion. The
island cover is a midpoint staircase; ``island_cells`` controls its refinement.
Fiber parameters and the third-wave PSD are locally constant on each rectangle.
The correlation window uses ``TRUNCATION_FACTOR`` times its attenuation scale.

Identical, power-balanced spans use pointwise Dirichlet coherence. General
routes integrate the coherent field sum across every span at each frequency
product, retaining the SS and XS terms, span-dependent dispersion, launch
powers, and receiver transfer. The general route calculation uses adaptive
one-dimensional quadrature and is more expensive than the identical-span path.

The default ``paper`` dispersion convention expands beta2 and beta3 about the
fiber reference frequency, following Eq. (5). Fourth-order dispersion is zero.
The default ``gn`` NSP convention uses the square root of the three signal
profiles divided by the CUT profile, as in Eq. (6). The other convention
settings are diagnostic comparisons rather than physical model defaults.
Only GN is implemented. ISRS core cache keys use canonical absolute powers;
output powers are recalculated for each call. Launch pre-emphasis is zero.
Active Raman amplification requires a distributed-Raman model.

References
----------
Y. Jiang, Y. Gao and P. Poggiolini, "An Efficient Correlation-based Evaluation
of the GN-Model for General Multi-Span Optical Systems", arXiv:2610.11624."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import llvmlite.ir as _ll_ir
import numpy as np
from numba import njit
from numba.core import types as _nb_types
from numba.extending import intrinsic as _nb_intrinsic

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanContext,
    SpanFiberOutput,
    SpanKernel,
    register_model,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
)
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)

if TYPE_CHECKING:
    from network_nlo_eval.core.flexgrid import FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayFloat, NDArrayInt
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "CGN_MODEL_INFO",
    "CGNModel",
    "CGNSpanKernel",
]

#:
DEFAULT_CORRELATION_POINTS = 32

#:
#:
DEFAULT_BAND_POINTS = 1

#:
DEFAULT_ISLAND_CELLS = 8

#:
DEFAULT_ISRS_SEGMENTS = 8

#:
#: ``"paper_product"``
NSP_CONVENTIONS = ("gn", "paper_product")
NSP_CONVENTION_DEFAULT = "gn"

#:
#: ``"repo"``
#: ``"paper_absolute"``
DISPERSION_CONVENTIONS = ("paper", "repo", "paper_absolute")
DISPERSION_CONVENTION_DEFAULT = "paper"

#:
DELTA_F_CO_HZ = 15e12

#:
TRUNCATION_FACTOR = 40.0

_CHUNK_ELEMENTS = 4_000_000

_CHUNK_BYTES = 4 * 1024 * 1024

#:
#:
#:
CANONICAL_POWER_DIGITS = 12

#:
CANONICAL_SHAPE_DIGITS = 9

CORE_CACHE_BUDGET_BYTES = 256 * 1024 * 1024

_VIEW_CACHE_ENTRIES = 64

#:
RAMP_BISECTION_STEPS = 60

CGN_MODEL_INFO = ModelInfo(
    name="cgn",
    summary=(
        "Semi-analytical correlation GN with ISRS and coherent span-pair accumulation."
    ),
    family="gn",
    reference="Jiang, Gao, Poggiolini, 'An Efficient Correlation-based Evaluation of the "
    "GN-Model for General Multi-Span Optical Systems', arXiv:2610.11624v1 (2026)",
    options=(
        (
            "correlation_points",
            "Quadrature nodes of the 1-D correlation (frequency-product) integral.",
        ),
        (
            "include_isrs",
            "Include the inter-channel Raman (ISRS) power tilt in the NSP.",
        ),
        (
            "dispersion_convention",
            "'paper' (default): beta3 expansion about the fiber reference frequency; comparison conventions also available.",
        ),
        (
            "nsp_convention",
            "'gn' (paper Eq. (6) square root) | 'paper_product' (historical doubled-exponent comparison).",
        ),
        (
            "band_points",
            (
                "In-band NLI-PSD quadrature nodes; 1 = flat-PSD convention (default), "
                ">1 = numerical in-band integration with the repository's island approximation."
            ),
        ),
    ),
)


# Fused kernels preserve the NumPy 1.26 complex arithmetic operation order.


@_nb_intrinsic  # type: ignore[untyped-decorator]
def _fma(typingctx: Any, a: Any, b: Any, c: Any) -> Any:  # noqa: ARG001
    """Emit LLVM's correctly rounded fused multiply-add intrinsic."""
    sig = _nb_types.float64(_nb_types.float64, _nb_types.float64, _nb_types.float64)

    def codegen(context: Any, builder: Any, sig: Any, args: Any) -> Any:  # noqa: ARG001
        module = builder.module
        function = module.declare_intrinsic(
            "llvm.fma", [_ll_ir.DoubleType(), _ll_ir.DoubleType(), _ll_ir.DoubleType()]
        )
        return builder.call(function, args)

    return sig, codegen


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _complex_multiply(
    ar: float, ai: float, br: float, bi: float
) -> tuple[float, float]:
    """Multiply complex scalars using the NumPy 1.26 SIMD FMA order.

    The explicit operation order preserves the fused kernel's rounding behavior."""
    cross = ai * bi
    cross2 = ai * br
    return _fma(ar, br, -cross), _fma(ar, bi, cross2)


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _np_maximum(a: float, b: float) -> float:
    """Return the scalar maximum with NumPy's NaN propagation semantics."""
    if a != a:  # noqa: PLR0124
        return a
    if b != b:  # noqa: PLR0124
        return b
    # Keep the second operand on equality, including signed zero.
    return a if a > b else b  # noqa: FURB136


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _np_minimum(a: float, b: float) -> float:
    """Return the scalar minimum with NumPy's NaN propagation semantics."""
    if a != a:  # noqa: PLR0124
        return a
    if b != b:  # noqa: PLR0124
        return b
    # Keep the second operand on equality, including signed zero.
    return a if a < b else b  # noqa: FURB136


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _square(value: float) -> float:
    """Square by multiplication, matching NumPy array exponentiation by two."""
    return value * value


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _fused_integrand_isrs(
    out: NDArrayFloat,
    alpha_x: NDArrayFloat,
    residual: NDArrayFloat,
    magnitude: NDArrayFloat,
    expm1_neg: NDArrayFloat,
    beta2_eff: NDArrayFloat,
    xi_min: NDArrayFloat,
    xi_span: NDArrayFloat,
    uu: NDArrayFloat,
    wq: NDArrayFloat,
    a_b: NDArrayFloat,
    b_b: NDArrayFloat,
    c_d: NDArrayFloat,
    d_d: NDArrayFloat,
    gamma2: NDArrayFloat,
    decay: float,
    four_pi2: float,
) -> NDArrayFloat:
    """Evaluate the ISRS spectrum, rectangle and quadrature weights in one pass.

    The output has shape ``(CUT, second, third, cells, points)``. The NSP
    residual is sampled by segment; attenuation and phase are integrated
    analytically within each segment. Zero-weight elements are set to zero."""
    n_c = alpha_x.shape[0]
    n_s = alpha_x.shape[1]
    n_j = alpha_x.shape[2]
    n_seg = residual.shape[3]
    n_cell = beta2_eff.shape[3]
    n_p = uu.shape[0]
    for c in range(n_c):
        g2 = gamma2[c]
        for s in range(n_s):
            for j in range(n_j):
                ax = alpha_x[c, s, j]
                mag = magnitude[c, s, j]
                exm = expm1_neg[c, s, j]
                for cell in range(n_cell):
                    om_coef = four_pi2 * beta2_eff[c, s, j, cell]
                    xi0 = xi_min[c, s, j, cell]
                    xspan = xi_span[c, s, j, cell]
                    av = a_b[c, cell]
                    bv = b_b[c, cell]
                    d_safe = d_d[c, s, j, cell]
                    if not (d_safe > 0.0):
                        d_safe = 1.0
                    c_safe = c_d[c, s, j, cell]
                    if not (c_safe > 0.0):
                        c_safe = 1.0
                    for p in range(n_p):
                        node = xi0 + xspan * uu[p]
                        eta_min = _np_maximum(av, node / d_safe)
                        eta_max = _np_minimum(bv, node / c_safe)
                        log_min = np.log(_np_maximum(eta_min, 1e-300))
                        log_max = np.log(_np_maximum(eta_max, 1e-300))
                        weight = _np_maximum(log_max - log_min, 0.0)
                        if weight == 0.0:
                            out[c, s, j, cell, p] = 0.0
                            continue
                        omega = om_coef * node
                        phase = omega * decay
                        cos_p = np.cos(phase)
                        sin_p = np.sin(phase)
                        w_re = mag * cos_p
                        w_im = mag * sin_p
                        acc_re = 0.0
                        acc_im = 0.0
                        for k in range(n_seg - 1, -1, -1):
                            acc_re, acc_im = _complex_multiply(
                                acc_re, acc_im, w_re, w_im
                            )
                            acc_re += residual[c, s, j, k]
                        omw_re = -exm + mag * (1.0 - cos_p)
                        omw_im = -(mag * sin_p)
                        div_re = ax
                        div_im = -omega
                        if div_re == 0.0 and div_im == 0.0:
                            seg_re = decay
                            seg_im = 0.0
                        elif abs(div_re) >= abs(div_im):
                            rat = div_im / div_re
                            scl = 1.0 / (div_re + div_im * rat)
                            seg_re = (omw_re + omw_im * rat) * scl
                            seg_im = (omw_im - omw_re * rat) * scl
                        else:
                            rat = div_re / div_im
                            scl = 1.0 / (div_im + div_re * rat)
                            seg_re = (omw_re * rat + omw_im) * scl
                            seg_im = (omw_im * rat - omw_re) * scl
                        amp_re, amp_im = _complex_multiply(
                            seg_re, seg_im, acc_re, acc_im
                        )
                        spectrum = g2 * (_square(amp_re) + _square(amp_im))
                        value = (spectrum * weight) * (xspan * wq[p])
                        if value != value or value == np.inf or value == -np.inf:  # noqa: PLR0124, PLR1714
                            value = 0.0
                        out[c, s, j, cell, p] = value
    return out


@njit(cache=True, nogil=True)  # type: ignore[untyped-decorator]
def _fused_integrand_plain(
    out: NDArrayFloat,
    alpha_x: NDArrayFloat,
    length_m: float,
    beta2_eff: NDArrayFloat,
    xi_min: NDArrayFloat,
    xi_span: NDArrayFloat,
    uu: NDArrayFloat,
    wq: NDArrayFloat,
    a_b: NDArrayFloat,
    b_b: NDArrayFloat,
    c_d: NDArrayFloat,
    d_d: NDArrayFloat,
    gamma2: NDArrayFloat,
    four_pi2: float,
) -> NDArrayFloat:
    """Evaluate the attenuation-only spectrum and quadrature weights in one pass.

    The output has shape ``(CUT, second, third, cells, points)``."""
    n_c = alpha_x.shape[0]
    n_s = alpha_x.shape[1]
    n_j = alpha_x.shape[2]
    n_cell = beta2_eff.shape[3]
    n_p = uu.shape[0]
    for c in range(n_c):
        g2 = gamma2[c]
        for s in range(n_s):
            for j in range(n_j):
                ax = alpha_x[c, s, j]
                decay = np.exp(-ax * length_m)
                for cell in range(n_cell):
                    om_coef = four_pi2 * beta2_eff[c, s, j, cell]
                    xi0 = xi_min[c, s, j, cell]
                    xspan = xi_span[c, s, j, cell]
                    av = a_b[c, cell]
                    bv = b_b[c, cell]
                    d_safe = d_d[c, s, j, cell]
                    if not (d_safe > 0.0):
                        d_safe = 1.0
                    c_safe = c_d[c, s, j, cell]
                    if not (c_safe > 0.0):
                        c_safe = 1.0
                    for p in range(n_p):
                        node = xi0 + xspan * uu[p]
                        eta_min = _np_maximum(av, node / d_safe)
                        eta_max = _np_minimum(bv, node / c_safe)
                        log_min = np.log(_np_maximum(eta_min, 1e-300))
                        log_max = np.log(_np_maximum(eta_max, 1e-300))
                        weight = _np_maximum(log_max - log_min, 0.0)
                        if weight == 0.0:
                            out[c, s, j, cell, p] = 0.0
                            continue
                        omega = om_coef * node
                        numerator = np.expm1(-ax * length_m) ** 2 + (
                            4.0 * decay * np.sin(omega * length_m / 2.0) ** 2
                        )
                        denominator = ax * ax + omega * omega
                        spectrum = g2 * (
                            length_m**2
                            if denominator == 0.0
                            else numerator / denominator
                        )
                        value = (spectrum * weight) * (xspan * wq[p])
                        if value != value or value == np.inf or value == -np.inf:  # noqa: PLR0124, PLR1714
                            value = 0.0
                        out[c, s, j, cell, p] = value
    return out


def _core_workers(n_tasks: int) -> int:
    """Return the core-construction thread count, bounded by the task count.

    ``CGN_CORE_WORKERS`` overrides the default; one selects serial construction."""
    if n_tasks <= 1:
        return 1
    override = os.environ.get("CGN_CORE_WORKERS")
    if override:
        try:
            requested = int(override)
        except ValueError:
            requested = 1
        return max(1, min(requested, n_tasks))
    return max(1, min(8, os.cpu_count() or 1, n_tasks))


def _nonlinearity_coefficient(
    lambdas_m: NDArrayFloat, n2_m2_w: float, a_eff_m2: NDArrayFloat
) -> NDArrayFloat:
    """Calculate the effective channel nonlinearity in 1/(W m).

    The CUT effective area is used in place of a four-frequency mutual area."""
    return (2.0 * np.pi / np.asarray(lambdas_m, dtype=np.float64)) * (
        n2_m2_w / np.asarray(a_eff_m2)
    )


def _effective_beta2(
    beta2: float,
    beta3: float,
    x_offsets: NDArrayFloat,
    y_offsets: NDArrayFloat,
    f_center_hz: NDArrayFloat | None = None,
    convention: str = DISPERSION_CONVENTION_DEFAULT,
    f_reference_hz: float | None = None,
) -> NDArrayFloat:
    """Evaluate Eq. (5) through beta3 about the fiber reference frequency.

    ``paper`` uses beta2 + pi*beta3*(2*(fCUT-fref)+x+y), with x and y
    measured relative to the observation frequency. ``repo`` and
    ``paper_absolute`` retain the two comparison expansion conventions.
    """
    if convention not in DISPERSION_CONVENTIONS:
        msg = f"dispersion convention must be one of {DISPERSION_CONVENTIONS}, got {convention!r}."
        raise ValueError(msg)
    x_offsets = np.asarray(x_offsets, dtype=np.float64)
    y_offsets = np.asarray(y_offsets, dtype=np.float64)
    if convention == "repo":
        return float(beta2) + np.pi * float(beta3) * (x_offsets + y_offsets)
    if f_center_hz is None:
        msg = "the dispersion convention needs f_center_hz."
        raise ValueError(msg)
    reference = 0.0
    if convention == "paper":
        if f_reference_hz is None:
            msg = "the 'paper' dispersion convention needs f_reference_hz."
            raise ValueError(msg)
        reference = float(f_reference_hz)
    return float(beta2) + np.pi * float(beta3) * (
        2.0 * (np.asarray(f_center_hz, dtype=np.float64) - reference)
        + x_offsets
        + y_offsets
    )


def _stimulus_attenuation(
    alpha_npm: NDArrayFloat,
    second: NDArrayInt,
    third: NDArrayInt,
    convention: str = NSP_CONVENTION_DEFAULT,
) -> NDArrayFloat:
    """Calculate the NSP exponential attenuation for CUT-first islands.

    The CUT and first-wave attenuations cancel. ``gn`` uses half the sum of the
    other two attenuations; ``paper_product`` uses the full sum."""
    if convention not in NSP_CONVENTIONS:
        msg = f"nsp convention must be one of {NSP_CONVENTIONS}, got {convention!r}."
        raise ValueError(msg)
    total = np.asarray(alpha_npm)[second] + np.asarray(alpha_npm)[third]
    if convention == "gn":
        return 0.5 * total
    return total


def _canonical_power(power: NDArrayFloat) -> tuple[NDArrayFloat, bytes]:
    """Canonicalize launch powers and return the values and cache-key bytes.

    The grid preserves absolute power and channel shape while absorbing small
    roundoff changes. Core construction uses these same canonical values, making
    cache entries independent of call order; nearby values can share a key."""
    values = np.asarray(power, dtype=np.float64)
    if values.size == 0:
        return values.copy(), b"empty"
    scale = float(np.max(np.abs(values)))
    if scale <= 0.0:
        return np.zeros_like(values), b"zero"
    exponent = int(np.floor(np.log10(scale)))
    decade = 10.0**exponent
    if not np.isfinite(decade) or decade <= 0.0:
        return values.copy(), values.tobytes()
    mantissa = scale / decade
    mantissa_canon = float(np.round(mantissa, CANONICAL_POWER_DIGITS - 1))
    if mantissa_canon <= 0.0:
        mantissa_canon = float(mantissa)
    scale_canon = mantissa_canon * decade
    step = scale_canon * (10.0 ** (-CANONICAL_SHAPE_DIGITS))
    if not np.isfinite(step) or step <= 0.0:
        return values.copy(), values.tobytes()
    canonical = np.round(values / step) * step
    if not np.all(np.isfinite(canonical)):
        return values.copy(), values.tobytes()
    return canonical, canonical.tobytes()


def _nsp_energy_spectrum(
    gamma_w_m: NDArrayFloat,
    alpha_x_npm: NDArrayFloat,
    length_m: float,
    omega_rad_m: NDArrayFloat,
) -> NDArrayFloat:
    """Calculate the analytical attenuation-only NSP energy spectrum.

    This is the squared magnitude of the spatial Fourier transform of the
    exponential stimulus profile, in inverse watts squared."""
    alpha_x = np.asarray(alpha_x_npm, dtype=np.float64)
    omega = np.asarray(omega_rad_m, dtype=np.float64)
    decay = np.exp(-alpha_x * length_m)
    numerator = (
        np.expm1(-alpha_x * length_m) ** 2
        + 4 * decay * np.sin(omega * length_m / 2) ** 2
    )
    denominator = alpha_x * alpha_x + omega * omega
    integral_squared = np.full(
        np.broadcast_shapes(numerator.shape, denominator.shape), length_m**2
    )
    np.divide(numerator, denominator, out=integral_squared, where=denominator != 0)
    return cast(
        "NDArrayFloat",
        np.asarray(gamma_w_m, dtype=np.float64) ** 2 * integral_squared,
    )


def _isrs_span_profile(
    f_rel_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    power_in_w: NDArrayFloat,
    f_m_hz: float,
    f_M_hz: float,
    k_bar: float,
    length_m: float,
    n_segments: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Propagate a segmented ISRS profile using the shared Raman approximation.

    Returns
    -------
    tuple of numpy.ndarray
        Segment midpoint positions in meters, normalized channel gain at each
        midpoint, and channel effective ISRS attenuation rates in Np/m."""
    n_seg = max(1, int(n_segments))
    dz = float(length_m) / n_seg
    z_mid = (np.arange(n_seg, dtype=np.float64) + 0.5) * dz

    power = np.array(power_in_w, dtype=np.float64, copy=True)
    launch = np.array(power_in_w, dtype=np.float64, copy=True)
    active = launch > 0.0
    safe_launch = np.where(active, launch, 1.0)

    n_sub = 2 * n_seg
    dz_sub = float(length_m) / n_sub
    gain = np.ones((n_seg, launch.size), dtype=np.float64)
    for index in range(n_sub):
        total_w = float(power.sum())
        r_f = _calc_raman_profile_jit(
            np.ascontiguousarray(f_rel_hz, dtype=np.float64),
            float(f_m_hz),
            float(f_M_hz),
            DELTA_F_CO_HZ,
            total_w,
        )
        power = _calc_span_out_power_jit(
            np.ascontiguousarray(power, dtype=np.float64),
            np.ascontiguousarray(f_rel_hz, dtype=np.float64),
            np.ascontiguousarray(alpha_npm, dtype=np.float64),
            dz_sub,
            np.ascontiguousarray(r_f, dtype=np.float64),
            float(k_bar),
        )
        if index % 2 == 0:
            coarse = index // 2
            gain[coarse] = (power / safe_launch) * np.exp(alpha_npm * z_mid[coarse])
    gain[:, ~active] = 1.0

    end_gain = (power / safe_launch) * np.exp(alpha_npm * float(length_m))
    end_gain = np.where(active & (end_gain > 0.0), end_gain, 1.0)
    rate = -np.log(end_gain) / float(length_m)
    rate = np.where(active, rate, 0.0)
    return z_mid, gain, rate


def _isrs_nsp_energy_spectrum(
    gamma_w_m: NDArrayFloat,
    alpha_x_npm: NDArrayFloat,
    residual: NDArrayFloat,
    seg_length_m: float,
    omega_rad_m: NDArrayFloat,
) -> NDArrayFloat:
    """Calculate the segmented ISRS NSP energy spectrum in inverse watts squared.

    The slowly varying residual is constant at each segment midpoint; attenuation
    and phase are integrated analytically within the segment."""
    alpha_x = np.asarray(alpha_x_npm, dtype=np.float64)
    omega = np.asarray(omega_rad_m, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)

    #   e^{-s Δ} = e^{-alpha_x Δ} (cos(ωΔ) + j sin(ωΔ))
    decay = float(seg_length_m)
    exponent_real = alpha_x * decay
    phase = omega * decay
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    magnitude = np.exp(-exponent_real)
    w = magnitude * (cos_phase + 1j * sin_phase)
    one_minus_w = (-np.expm1(-exponent_real) + magnitude * (1.0 - cos_phase)) - 1j * (
        magnitude * sin_phase
    )

    accumulator = np.zeros_like(w)
    for index in range(residual.shape[-1] - 1, -1, -1):
        accumulator *= w
        accumulator += residual[..., index][..., None, None]
    denominator = alpha_x - 1j * omega
    segment = np.full(
        np.broadcast_shapes(one_minus_w.shape, denominator.shape), decay, dtype=complex
    )
    np.divide(one_minus_w, denominator, out=segment, where=denominator != 0)
    amplitude = segment * accumulator
    gamma = np.asarray(gamma_w_m, dtype=np.float64)
    return cast("NDArrayFloat", gamma**2 * (amplitude.real**2 + amplitude.imag**2))


def _rectangle_weight(
    a: NDArrayFloat,
    b: NDArrayFloat,
    c: NDArrayFloat,
    d: NDArrayFloat,
    xi: NDArrayFloat,
) -> NDArrayFloat:
    """Return the dimensionless Eq. (27)/(29) rectangle weight at frequency product xi."""
    xi = np.asarray(xi, dtype=np.float64)
    d_safe = np.where(np.asarray(d) > 0.0, np.asarray(d), 1.0)
    c_safe = np.where(np.asarray(c) > 0.0, np.asarray(c), 1.0)
    eta_min = np.maximum(np.asarray(a), xi / d_safe)
    eta_max = np.minimum(np.asarray(b), xi / c_safe)
    log_min = np.log(np.maximum(eta_min, 1e-300))
    log_max = np.log(np.maximum(eta_max, 1e-300))
    return cast("NDArrayFloat", np.maximum(log_max - log_min, 0.0))


def _dirichlet_weight(theta: NDArrayFloat, n_spans: int) -> NDArrayFloat:
    """Return the equal-span Dirichlet coherent power factor for each phase."""
    if n_spans <= 1:
        return np.ones_like(np.asarray(theta, dtype=np.float64))
    half = 0.5 * np.asarray(theta, dtype=np.float64)
    ratio = n_spans * np.sinc(n_spans * half / np.pi) / np.sinc(half / np.pi)
    return ratio**2


def _equivalent_phase_ramp(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Invert the Dirichlet factor to an equivalent phase step per span.

    Targets are clipped to [0, N²]. The resulting ramp matches total coherence,
    not the original pairwise phase distribution."""
    n = int(n_spans)
    if n <= 1:
        return np.zeros_like(np.asarray(target, dtype=np.float64))
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n)
    for _ in range(RAMP_BISECTION_STEPS):
        mid = 0.5 * (low + high)
        too_large = _dirichlet_weight(mid, n) > goal
        low = np.where(too_large, mid, low)
        high = np.where(too_large, high, mid)
    return 0.5 * (low + high)


def _nli_from_core(
    core: NDArrayFloat,
    g_psd: NDArrayFloat,
    third_psd: NDArrayFloat,
    b_hz: NDArrayFloat,
    r_index: NDArrayInt,
    apply_bandwidth: bool = True,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Aggregate the core integrals and channel PSDs into SCI/XCI powers.

    SCI requires all three interacting waves in the CUT. Other CUT-first
    islands contribute XCI, with a symmetry factor for distinct first and
    second waves. ``apply_bandwidth`` distinguishes center PSD from in-band integrals."""
    m = core * third_psd
    m_sum = m.sum(axis=2)
    diag = np.arange(core.shape[0])
    sci_integral = np.where(
        r_index[diag, diag] == diag[:, None], m[diag, diag], 0.0
    ).sum(axis=1)
    spm_sum = sci_integral * g_psd
    off_diagonal = m_sum.copy()
    off_diagonal[diag, diag] = 0.0
    xpm_sum = 2.0 * (off_diagonal @ g_psd) + (m_sum[diag, diag] - sci_integral) * g_psd
    prefactor = (16.0 / 27.0) * g_psd
    if apply_bandwidth:
        return prefactor * spm_sum * b_hz, prefactor * xpm_sum * b_hz
    return prefactor * spm_sum, prefactor * xpm_sum


def _coherent_ratio(link: NDArrayFloat, single: NDArrayFloat) -> NDArrayFloat:
    """Return link-to-single-span noise ratios, using one when single-span power is zero."""
    link = np.asarray(link, dtype=np.float64)
    single = np.asarray(single, dtype=np.float64)
    positive = single > 0.0
    safe = np.where(positive, single, 1.0)
    return np.where(positive, link / safe, 1.0)


def _xi_quadrature(
    xi_min: NDArrayFloat, xi_max: NDArrayFloat, n_points: int
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Return quadratically graded midpoint nodes and weights for frequency products.

    Nodes concentrate near zero, where the SCI rectangle has a logarithmic
    singularity."""
    u = (np.arange(n_points, dtype=np.float64) + 0.5) / float(n_points)
    u_edges = np.arange(n_points + 1, dtype=np.float64) / float(n_points)
    xi_min = np.asarray(xi_min, dtype=np.float64)[..., None]
    xi_max = np.asarray(xi_max, dtype=np.float64)[..., None]
    span = xi_max - xi_min
    nodes = xi_min + span * (u**2)
    weights = span * (u_edges[1:] ** 2 - u_edges[:-1] ** 2)
    return nodes, weights


def _band_quadrature(n_points: int) -> tuple[tuple[float, float], ...]:
    """Return Gauss-Legendre nodes and weights for in-band observation frequencies.

    Parameters
    ----------
    n_points : int
        Number of nodes; at least two.

    Returns
    -------
    tuple
        Normalized node and weight pairs on [-1, 1]."""
    if n_points < 2:
        msg = f"band_points must be >= 2, got {n_points}."
        raise ValueError(msg)
    nodes, weights = np.polynomial.legendre.leggauss(int(n_points))  # type: ignore[no-untyped-call]
    return tuple(
        (float(node), float(weight))
        for node, weight in zip(nodes, weights, strict=True)
    )


def _island_cover(
    x1: NDArrayFloat,
    x2: NDArrayFloat,
    y1: NDArrayFloat,
    y2: NDArrayFloat,
    u1: NDArrayFloat,
    u2: NDArrayFloat,
    n_cells: int,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Cover a three-band integration island with midpoint rectangles.

    Each first-axis slice uses the midpoint height of its allowed second-axis
    interval. The staircase area approximation has second-order error."""
    x1 = np.asarray(x1, dtype=np.float64)[..., None, None]
    x2 = np.asarray(x2, dtype=np.float64)[..., None, None]
    y1 = np.asarray(y1, dtype=np.float64)[..., None, None]
    y2 = np.asarray(y2, dtype=np.float64)[..., None, None]
    u1 = np.asarray(u1, dtype=np.float64)[..., None]
    u2 = np.asarray(u2, dtype=np.float64)[..., None]

    dx = (x2 - x1) / float(n_cells)
    xa = x1 + np.arange(n_cells, dtype=np.float64) * dx
    xb = xa + dx
    xm = 0.5 * (xa + xb)
    ylo = np.maximum(y1, u1 - xm)
    yhi = np.minimum(y2, u2 - xm)
    return xa, xb, ylo, yhi


def _quadrant_pieces(
    xa: NDArrayFloat,
    xb: NDArrayFloat,
    ylo: NDArrayFloat,
    yhi: NDArrayFloat,
) -> list[tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]]:
    """Split a rectangle by quadrant and map each piece into the first quadrant."""
    xa = np.asarray(xa, dtype=np.float64)
    xb = np.asarray(xb, dtype=np.float64)
    ylo = np.asarray(ylo, dtype=np.float64)
    yhi = np.asarray(yhi, dtype=np.float64)

    x_neg_a = -np.minimum(xb, 0.0)
    x_neg_b = -xa
    x_pos_a = np.maximum(xa, 0.0)
    x_pos_b = xb
    y_neg_c = -np.minimum(yhi, 0.0)
    y_neg_d = -ylo
    y_pos_c = np.maximum(ylo, 0.0)
    y_pos_d = yhi

    return [
        (x_neg_a, x_neg_b, y_neg_c, y_neg_d),
        (x_neg_a, x_neg_b, y_pos_c, y_pos_d),
        (x_pos_a, x_pos_b, y_neg_c, y_neg_d),
        (x_pos_a, x_pos_b, y_pos_c, y_pos_d),
    ]


def _third_wave_slots(
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    widen_by_cut_bandwidth: bool = False,
) -> tuple[NDArrayInt, int]:
    """Enumerate reachable third-wave channels for each CUT and second wave.

    The optional CUT-bandwidth widening covers observation frequencies throughout
    the band. Missing slots are represented by -1."""
    n_ch = f_abs_hz.size
    order = np.argsort(f_abs_hz, kind="stable")
    f_sorted = f_abs_hz[order]
    b_sorted = b_hz[order]
    lower_edge = f_sorted - 0.5 * b_sorted
    upper_edge = f_sorted + 0.5 * b_sorted
    lower_edge = np.maximum.accumulate(lower_edge)
    upper_edge = np.maximum.accumulate(upper_edge)

    reach = 0.5 * (b_hz[:, None] + b_hz[None, :])
    if widen_by_cut_bandwidth:
        reach = reach + 0.5 * b_hz[:, None]
    band_low = f_abs_hz[None, :] - reach
    band_high = f_abs_hz[None, :] + reach

    i_lo = np.zeros((n_ch, n_ch), dtype=np.int64)
    i_hi = np.zeros((n_ch, n_ch), dtype=np.int64)
    for second in range(n_ch):
        i_lo[:, second] = np.searchsorted(upper_edge, band_low[:, second], side="left")
        i_hi[:, second] = np.searchsorted(
            lower_edge, band_high[:, second], side="right"
        )

    counts = np.maximum(i_hi - i_lo, 0)
    n_slots = int(counts.max()) if counts.size else 0
    if n_slots == 0:
        return np.full((n_ch, n_ch, 0), -1, dtype=np.int64), 0
    offsets = np.arange(n_slots, dtype=np.int64)
    valid = offsets[None, None, :] < counts[:, :, None]
    sorted_index = np.where(valid, i_lo[:, :, None] + offsets[None, None, :], 0)
    r_index = order[sorted_index].astype(np.int64)
    return np.where(valid, r_index, -1), n_slots


def _island_core(
    *,
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    gamma_w_m: NDArrayFloat,
    beta2_s2_m: float,
    beta3_s3_m: float,
    dispersion_convention: str,
    nsp_convention: str,
    length_m: float,
    r_index: NDArrayInt,
    correlation_points: int,
    island_cells: int,
    n_coherent_spans: int,
    isrs_gain: NDArrayFloat | None = None,
    isrs_rate: NDArrayFloat | None = None,
    isrs_z_mid: NDArrayFloat | None = None,
    isrs_seg_length_m: float = 0.0,
    band_points: int = DEFAULT_BAND_POINTS,
    compact_live: bool = False,
    compact_scatter: bool = True,
    f_reference_hz: float | None = None,
) -> tuple[NDArrayFloat, NDArrayFloat | None]:
    """Calculate single-span and coherent link core matrices for CUT-first islands.

    Each core integral combines the NSP spectrum and rectangular correlation
    weight. ISRS uses segmented profiles. When ``n_coherent_spans > 1``, a second
    matrix includes the pointwise Dirichlet factor and represents the link total.
    With in-band quadrature the matrix already includes bandwidth. Live-rectangle
    compression removes zero-width pieces; the scatter reduction retains the
    full-grid summation order while compact reduction can differ by rounding.

    Returns
    -------
    tuple
        Single-span core matrix and coherent link matrix, or None for the latter
        when the coherent span count is one."""
    n_ch = f_abs_hz.size
    n_slots = r_index.shape[2]
    core = np.zeros((n_ch, n_ch, n_slots), dtype=np.float64)
    if n_slots == 0 or length_m <= 0.0:
        return core, None
    dirichlet = None
    if n_coherent_spans > 1:
        dirichlet = np.zeros((n_ch, n_ch, n_slots), dtype=np.float64)

    x1 = -0.5 * b_hz[:, None]
    x2 = 0.5 * b_hz[:, None]
    delta = f_abs_hz[None, :] - f_abs_hz[:, None]
    y1 = delta - 0.5 * b_hz[None, :]
    y2 = delta + 0.5 * b_hz[None, :]
    third = np.where(r_index >= 0, r_index, 0)
    u1 = f_abs_hz[third] - f_abs_hz[:, None, None] - 0.5 * b_hz[third]
    u2 = f_abs_hz[third] - f_abs_hz[:, None, None] + 0.5 * b_hz[third]
    second = np.broadcast_to(
        np.arange(n_ch, dtype=np.int64)[None, :, None], r_index.shape
    )
    alpha_x = _stimulus_attenuation(alpha_npm, second, third, convention=nsp_convention)

    use_isrs = isrs_gain is not None and isrs_rate is not None
    if use_isrs and isrs_z_mid is None:
        msg = "ISRS segment positions are required with gain and rate profiles."
        raise ValueError(msg)
    # The three arrays are only accessed when use_isrs is true.
    gain_profile = cast("NDArrayFloat", isrs_gain)
    rate_profile = cast("NDArrayFloat", isrs_rate)
    segment_positions = cast("NDArrayFloat", isrs_z_mid)
    if use_isrs:
        scale = 0.5 if nsp_convention == "gn" else 1.0
        alpha_x = alpha_x + scale * (rate_profile[second] + rate_profile[third])

    per_c = max(1, n_ch * n_slots * island_cells * max(correlation_points, 1))
    chunk = max(1, min(n_ch, int(_CHUNK_BYTES // (per_c * 8))))

    integrate_band = int(band_points) > 1
    band_nodes = _band_quadrature(int(band_points)) if integrate_band else ((0.0, 1.0),)

    u_nodes = (np.arange(correlation_points, dtype=np.float64) + 0.5) / float(
        correlation_points
    )
    u_edges = np.arange(correlation_points + 1, dtype=np.float64) / float(
        correlation_points
    )
    quad_u = u_nodes**2
    quad_w = u_edges[1:] ** 2 - u_edges[:-1] ** 2

    four_pi2 = 4.0 * np.pi**2

    def _chunk_body(start: int, stop: int) -> tuple[NDArrayFloat, NDArrayFloat | None]:
        """Compute independent CUT rows of the island core in fixed summation order."""
        idx = slice(start, stop)
        xa, xb, ylo, yhi = _island_cover(
            x1[idx],
            x2[idx],
            y1[idx],
            y2[idx],
            u1[idx],
            u2[idx],
            island_cells,
        )
        alpha_c = alpha_x[idx]

        residual_view = None
        magnitude = None
        expm1_neg = None
        if use_isrs:
            residual = np.empty(
                (stop - start, n_ch, n_slots, gain_profile.shape[0]), dtype=np.float64
            )
            third_chunk = third[idx]
            rate_second = rate_profile[second[idx]]
            for seg in range(gain_profile.shape[0]):
                gain_seg = gain_profile[seg]
                residual[..., seg] = (
                    gain_seg[None, :, None] * gain_seg[third_chunk]
                ) ** scale * np.exp(
                    scale
                    * (rate_second + rate_profile[third_chunk])
                    * segment_positions[seg]
                )
            residual_view = residual
            decay = float(isrs_seg_length_m)
            exponent = alpha_c * decay
            magnitude = np.exp(-exponent)
            expm1_neg = np.expm1(-exponent)

        gamma2 = gamma_w_m[idx] ** 2
        shape = (stop - start, n_ch, n_slots, island_cells, correlation_points)
        integrand = np.empty(shape, dtype=np.float64)
        weighted_buf = None
        compact_buf = None
        if compact_live:
            weighted_buf = np.empty(shape, dtype=np.float64)
            n_rows = (stop - start) * n_ch * n_slots
            compact_buf = np.empty(
                (n_rows * island_cells, 1, 1, 1, correlation_points), dtype=np.float64
            )
        row_second = None
        row_third = None
        row_gamma = None
        if compact_live:
            row_second = np.ascontiguousarray(second[idx]).reshape(-1)
            row_third = np.ascontiguousarray(third[idx]).reshape(-1)
            row_gamma = np.ascontiguousarray(
                np.broadcast_to(gamma2[:, None, None], (stop - start, n_ch, n_slots))
            ).reshape(-1)

        acc = np.zeros((stop - start, n_ch, n_slots), dtype=np.float64)
        acc_dirichlet = (
            None
            if dirichlet is None
            else np.zeros((stop - start, n_ch, n_slots), dtype=np.float64)
        )
        for node, node_weight in band_nodes:
            if node == 0.0:
                cover = (xa, xb, ylo, yhi)
            else:
                shift = (0.5 * b_hz[idx] * node)[:, None]
                cover = _island_cover(
                    x1[idx] - shift,
                    x2[idx] - shift,
                    y1[idx] - shift,
                    y2[idx] - shift,
                    u1[idx] - shift[..., None],
                    u2[idx] - shift[..., None],
                    island_cells,
                )
            beta2_eff = _effective_beta2(
                beta2_s2_m,
                beta3_s3_m,
                0.5 * (cover[0] + cover[1]),
                0.5 * (cover[2] + cover[3]),
                f_center_hz=(f_abs_hz[idx] + 0.5 * b_hz[idx] * node).reshape(
                    -1, 1, 1, 1
                ),
                convention=dispersion_convention,
                f_reference_hz=f_reference_hz,
            )
            band_scale = (
                1.0
                if not integrate_band
                else (node_weight * 0.5 * b_hz[idx])[:, None, None]
            )
            for a, b, c, d in _quadrant_pieces(*cover):
                xi_min = a * c
                xi_max = b * d
                with np.errstate(divide="ignore", invalid="ignore"):
                    xi_scale = alpha_c[..., None] / (4.0 * np.pi**2 * np.abs(beta2_eff))
                xi_scale = np.where(
                    np.isfinite(xi_scale) & (xi_scale > 0.0), xi_scale, np.inf
                )
                xi_cut = np.minimum(xi_max, xi_min + TRUNCATION_FACTOR * xi_scale)
                xi_end = np.maximum(xi_cut, xi_min)

                if compact_live:
                    assert row_gamma is not None
                    assert row_second is not None
                    assert row_third is not None
                    assert compact_buf is not None
                    assert weighted_buf is not None
                    xi_span = xi_end - xi_min
                    span_rows = xi_span.reshape(n_rows, island_cells)
                    live_rows, live_cells = np.nonzero(span_rows > 0.0)
                    n_live = int(live_rows.size)
                    if compact_scatter:
                        integrand.fill(0.0)
                    if n_live:
                        shape_rows = (stop - start, n_ch, n_slots, island_cells)
                        a_rows = np.ascontiguousarray(
                            np.broadcast_to(a, shape_rows)
                        ).reshape(n_rows, island_cells)
                        b_rows = np.ascontiguousarray(
                            np.broadcast_to(b, shape_rows)
                        ).reshape(n_rows, island_cells)
                        c_rows = np.ascontiguousarray(c).reshape(n_rows, island_cells)
                        d_rows = np.ascontiguousarray(d).reshape(n_rows, island_cells)
                        beta_rows = beta2_eff.reshape(n_rows, island_cells)
                        xi_min_rows = xi_min.reshape(n_rows, island_cells)
                        xi_span_rows = span_rows
                        alpha_rows = alpha_c.reshape(n_rows)
                        a_live = np.ascontiguousarray(a_rows[live_rows, live_cells])[
                            :, None
                        ]
                        b_live = np.ascontiguousarray(b_rows[live_rows, live_cells])[
                            :, None
                        ]
                        c_live = np.ascontiguousarray(c_rows[live_rows, live_cells])[
                            :, None, None, None
                        ]
                        d_live = np.ascontiguousarray(d_rows[live_rows, live_cells])[
                            :, None, None, None
                        ]
                        beta_live = np.ascontiguousarray(
                            beta_rows[live_rows, live_cells]
                        )[:, None, None, None]
                        xi_min_live = np.ascontiguousarray(
                            xi_min_rows[live_rows, live_cells]
                        )[:, None, None, None]
                        xi_span_live = np.ascontiguousarray(
                            xi_span_rows[live_rows, live_cells]
                        )[:, None, None, None]
                        alpha_live = np.ascontiguousarray(alpha_rows[live_rows])[
                            :, None, None
                        ]
                        gamma_live = np.ascontiguousarray(row_gamma[live_rows])
                        if integrate_band:
                            band_flat = np.ascontiguousarray(
                                np.broadcast_to(
                                    band_scale, (stop - start, n_ch, n_slots)
                                )
                            ).reshape(-1)
                        else:
                            band_flat = np.ones(n_rows, dtype=np.float64)
                        band_live = band_flat[live_rows]
                        residual_live = None
                        magnitude_live = None
                        expm1_live = None
                        if use_isrs:
                            second_live = row_second[live_rows]
                            third_live = row_third[live_rows]
                            residual_live = np.empty(
                                (n_live, 1, 1, gain_profile.shape[0]), dtype=np.float64
                            )
                            for seg in range(gain_profile.shape[0]):
                                gain_seg = gain_profile[seg]
                                residual_live[:, 0, 0, seg] = (
                                    gain_seg[second_live] * gain_seg[third_live]
                                ) * np.exp(
                                    (
                                        rate_profile[second_live]
                                        + rate_profile[third_live]
                                    )
                                    * segment_positions[seg]
                                )
                            magnitude_live = np.exp(-alpha_live * decay)
                            expm1_live = np.expm1(-alpha_live * decay)
                        out = compact_buf[:n_live]
                        if use_isrs:
                            _fused_integrand_isrs(
                                out,
                                alpha_live,
                                residual_live,
                                magnitude_live,
                                expm1_live,
                                beta_live,
                                xi_min_live,
                                xi_span_live,
                                quad_u,
                                quad_w,
                                a_live,
                                b_live,
                                c_live,
                                d_live,
                                gamma_live,
                                decay,
                                four_pi2,
                            )
                        else:
                            _fused_integrand_plain(
                                out,
                                alpha_live,
                                float(length_m),
                                beta_live,
                                xi_min_live,
                                xi_span_live,
                                quad_u,
                                quad_w,
                                a_live,
                                b_live,
                                c_live,
                                d_live,
                                gamma_live,
                                four_pi2,
                            )
                        if compact_scatter:
                            integrand_rows = integrand.reshape(
                                n_rows, island_cells, correlation_points
                            )
                            integrand_rows[live_rows, live_cells, :] = out[
                                :, 0, 0, 0, :
                            ]
                    if compact_scatter:
                        acc += band_scale * np.sum(integrand, axis=(-2, -1))
                    elif n_live:
                        values = band_live * np.sum(out, axis=(-2, -1)).reshape(n_live)
                        acc += np.bincount(
                            live_rows, weights=values, minlength=n_rows
                        ).reshape(stop - start, n_ch, n_slots)
                    if acc_dirichlet is not None:
                        dirichlet_values = None
                        if n_live:
                            nodes = (
                                xi_min_live.reshape(n_live)[:, None]
                                + xi_span_live.reshape(n_live)[:, None]
                                * quad_u[None, :]
                            )
                            theta = (
                                four_pi2
                                * beta_live.reshape(n_live)[:, None]
                                * length_m
                                * nodes
                            )
                            dirichlet_values = out[:, 0, 0, 0, :] * _dirichlet_weight(
                                theta, n_coherent_spans
                            )
                        if compact_scatter:
                            weighted_rows = weighted_buf.reshape(
                                n_rows, island_cells, correlation_points
                            )
                            weighted_rows.fill(0.0)
                            if n_live:
                                weighted_rows[live_rows, live_cells, :] = (
                                    dirichlet_values
                                )
                            weighted = np.where(
                                np.isfinite(weighted_buf), weighted_buf, 0.0
                            )
                            acc_dirichlet += band_scale * np.sum(
                                weighted, axis=(-2, -1)
                            )
                        elif n_live:
                            assert dirichlet_values is not None
                            weighted = np.where(
                                np.isfinite(dirichlet_values), dirichlet_values, 0.0
                            )
                            dirichlet_sum = band_live * np.sum(weighted, axis=-1)
                            acc_dirichlet += np.bincount(
                                live_rows, weights=dirichlet_sum, minlength=n_rows
                            ).reshape(stop - start, n_ch, n_slots)
                    continue

                a_b = np.ascontiguousarray(a[:, 0, 0, :])
                b_b = np.ascontiguousarray(b[:, 0, 0, :])
                if use_isrs:
                    _fused_integrand_isrs(
                        integrand,
                        np.ascontiguousarray(alpha_c),
                        residual_view,
                        magnitude,
                        expm1_neg,
                        beta2_eff,
                        xi_min,
                        xi_end - xi_min,
                        quad_u,
                        quad_w,
                        a_b,
                        b_b,
                        np.ascontiguousarray(c),
                        np.ascontiguousarray(d),
                        gamma2,
                        decay,
                        four_pi2,
                    )
                else:
                    _fused_integrand_plain(
                        integrand,
                        np.ascontiguousarray(alpha_c),
                        float(length_m),
                        beta2_eff,
                        xi_min,
                        xi_end - xi_min,
                        quad_u,
                        quad_w,
                        a_b,
                        b_b,
                        np.ascontiguousarray(c),
                        np.ascontiguousarray(d),
                        gamma2,
                        four_pi2,
                    )
                acc += band_scale * np.sum(integrand, axis=(-2, -1))
                if acc_dirichlet is not None:
                    nodes, _ = _xi_quadrature(xi_min, xi_end, correlation_points)
                    theta = 4.0 * np.pi**2 * beta2_eff[..., None] * length_m * nodes
                    weighted = integrand * _dirichlet_weight(theta, n_coherent_spans)
                    weighted = np.where(np.isfinite(weighted), weighted, 0.0)
                    acc_dirichlet += band_scale * np.sum(weighted, axis=(-2, -1))

        acc = np.where(np.isfinite(acc), acc, 0.0)
        if acc_dirichlet is not None:
            acc_dirichlet = np.where(np.isfinite(acc_dirichlet), acc_dirichlet, 0.0)
        return acc, acc_dirichlet

    ranges = [(start, min(start + chunk, n_ch)) for start in range(0, n_ch, chunk)]
    workers = _core_workers(len(ranges))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda item: _chunk_body(*item), ranges))
    else:
        results = [_chunk_body(*item) for item in ranges]
    for (start, stop), (acc, acc_dirichlet) in zip(ranges, results, strict=True):
        core[start:stop] = acc
        if dirichlet is not None:
            dirichlet[start:stop] = acc_dirichlet

    return core, dirichlet


def _mci_slot_table(
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    beta2_s2_m: float,
) -> tuple[NDArrayInt, NDArrayFloat]:
    """Enumerate MCI wave triples for each CUT and their symmetry weights.

    The first two waves lie outside the CUT. Candidates are screened against the
    existing correlation-window truncation using a conservative detuning bound.
    The third wave must be reachable within the observation and input bands.
    Zero dispersion retains all geometrically valid candidates.

    Returns
    -------
    tuple of numpy.ndarray
        Wave indices of shape (channels, slots, 3), padded with -1, and symmetry
        weights of shape (channels, slots)."""
    n_ch = int(f_abs_hz.size)
    half = 0.5 * np.asarray(b_hz, dtype=np.float64)
    order = np.argsort(f_abs_hz, kind="stable")
    f_sorted = f_abs_hz[order]
    half_sorted = half[order]
    lower_edge = np.maximum.accumulate(f_sorted - half_sorted)
    upper_edge = np.maximum.accumulate(f_sorted + half_sorted)
    rank = np.empty(n_ch, dtype=np.int64)
    rank[order] = np.arange(n_ch, dtype=np.int64)

    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_ref = float(np.max(np.asarray(alpha_npm, dtype=np.float64)))
        xi_scale = float(np.divide(alpha_ref, 4.0 * np.pi**2 * abs(float(beta2_s2_m))))
    xi_scale = xi_scale if np.isfinite(xi_scale) and xi_scale > 0.0 else np.inf
    xi_max = TRUNCATION_FACTOR * xi_scale

    rows: list[list[tuple[int, int, int, float]]] = []
    for cut in range(n_ch):
        f_cut = float(f_abs_hz[cut])
        offset = np.abs(f_abs_hz - f_cut)
        detune_min = np.maximum(0.0, offset - half - half[cut])
        by_distance = np.argsort(offset, kind="stable")
        cand = by_distance[by_distance != cut]
        if cand.size == 0:
            rows.append([])
            continue
        reach = detune_min[cand]
        thr = np.full(reach.shape, np.inf)
        positive = reach > 0.0
        np.divide(float(xi_max), reach, out=thr, where=positive)
        by_detune = np.argsort(reach, kind="stable")
        detune_sorted = reach[by_detune]
        count = np.searchsorted(detune_sorted, thr, side="right")
        max_count = int(count.max()) if count.size else 0
        if max_count == 0:
            rows.append([])
            continue
        pool = by_detune[:max_count]
        allowed = detune_sorted[:max_count][None, :] <= thr[:, None]
        rank_cand = rank[cand]
        allowed &= rank_cand[pool][None, :] <= rank_cand[:, None]
        first_pos, pool_pos = np.nonzero(allowed)
        if first_pos.size == 0:
            rows.append([])
            continue
        first_idx = cand[first_pos]
        second_idx = cand[pool[pool_pos]]
        sum_low = (
            f_abs_hz[first_idx]
            - half[first_idx]
            + f_abs_hz[second_idx]
            - half[second_idx]
            - f_cut
            - half[cut]
        )
        sum_high = (
            f_abs_hz[first_idx]
            + half[first_idx]
            + f_abs_hz[second_idx]
            + half[second_idx]
            - f_cut
            + half[cut]
        )
        k_lo = np.searchsorted(upper_edge, sum_low, side="left")
        k_hi = np.searchsorted(lower_edge, sum_high, side="right")
        entries: list[tuple[int, int, int, float]] = []
        for pos in range(first_pos.size):
            first_int = int(first_idx[pos])
            second_int = int(second_idx[pos])
            symmetry = 2.0 if first_int != second_int else 1.0
            for k in range(int(k_lo[pos]), int(k_hi[pos])):
                entries.append((first_int, second_int, int(order[k]), symmetry))  # noqa: PERF401
        rows.append(entries)

    n_slots = max((len(row) for row in rows), default=0)
    index = np.full((n_ch, n_slots, 3), -1, dtype=np.int64)
    weight = np.zeros((n_ch, n_slots), dtype=np.float64)
    for cut, entries in enumerate(rows):
        for slot, (first, second, third, sym) in enumerate(entries):
            index[cut, slot] = (first, second, third)
            weight[cut, slot] = sym
    return index, weight


def _mci_stimulus_attenuation(
    alpha_npm: NDArrayFloat,
    first: NDArrayInt,
    second: NDArrayInt,
    third: NDArrayInt,
    cut_alpha: NDArrayFloat,
    convention: str = NSP_CONVENTION_DEFAULT,
) -> NDArrayFloat:
    """Calculate the MCI NSP attenuation from all three waves and the CUT.

    Unlike CUT-first islands, the first-wave attenuation does not cancel. The
    ``gn`` convention halves the exponent; ``paper_product`` retains it."""
    if convention not in NSP_CONVENTIONS:
        msg = f"nsp convention must be one of {NSP_CONVENTIONS}, got {convention!r}."
        raise ValueError(msg)
    total = (
        np.asarray(alpha_npm)[first]
        + np.asarray(alpha_npm)[second]
        + np.asarray(alpha_npm)[third]
        - np.asarray(cut_alpha)
    )
    if convention == "gn":
        return cast("NDArrayFloat", 0.5 * total)
    return cast("NDArrayFloat", total)


def _mci_core(
    *,
    f_abs_hz: NDArrayFloat,
    b_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    gamma_w_m: NDArrayFloat,
    beta2_s2_m: float,
    beta3_s3_m: float,
    dispersion_convention: str,
    nsp_convention: str,
    length_m: float,
    mci_index: NDArrayInt,
    correlation_points: int,
    island_cells: int,
    n_coherent_spans: int,
    isrs_gain: NDArrayFloat | None = None,
    isrs_rate: NDArrayFloat | None = None,
    isrs_z_mid: NDArrayFloat | None = None,
    isrs_seg_length_m: float = 0.0,
    band_points: int = DEFAULT_BAND_POINTS,
    f_reference_hz: float | None = None,
) -> tuple[NDArrayFloat, NDArrayFloat | None]:
    """Compute single-span and coherent link core integrals for MCI slots.

    MCI uses the same fused NSP kernels as CUT-first islands, with off-band wave
    geometry and the full NSP attenuation. Zero-width rectangles are compressed.

    Returns
    -------
    tuple
        Single-span and optional coherent link MCI matrices."""
    n_ch = int(f_abs_hz.size)
    n_slots = int(mci_index.shape[1])
    core = np.zeros((n_ch, n_slots), dtype=np.float64)
    dirichlet = None
    if n_coherent_spans > 1:
        dirichlet = np.zeros((n_ch, n_slots), dtype=np.float64)
    if n_slots == 0 or length_m <= 0.0:
        return core, dirichlet

    half = 0.5 * np.asarray(b_hz, dtype=np.float64)
    use_isrs = isrs_gain is not None and isrs_rate is not None
    if use_isrs and isrs_z_mid is None:
        msg = "ISRS segment positions are required with gain and rate profiles."
        raise ValueError(msg)
    # The three arrays are only accessed when use_isrs is true.
    gain_profile = cast("NDArrayFloat", isrs_gain)
    rate_profile = cast("NDArrayFloat", isrs_rate)
    segment_positions = cast("NDArrayFloat", isrs_z_mid)

    integrate_band = int(band_points) > 1
    band_nodes = _band_quadrature(int(band_points)) if integrate_band else ((0.0, 1.0),)

    u_nodes = (np.arange(correlation_points, dtype=np.float64) + 0.5) / float(
        correlation_points
    )
    u_edges = np.arange(correlation_points + 1, dtype=np.float64) / float(
        correlation_points
    )
    quad_u = u_nodes**2
    quad_w = u_edges[1:] ** 2 - u_edges[:-1] ** 2

    four_pi2 = 4.0 * np.pi**2

    per_c = max(1, n_slots * island_cells * 4 * max(correlation_points, 1))
    chunk = max(1, min(n_ch, int(_CHUNK_ELEMENTS // per_c)))

    def _chunk_body(start: int, stop: int) -> tuple[NDArrayFloat, NDArrayFloat | None]:
        """Compute the MCI core for independent CUT rows and their wave slots."""
        idx = slice(start, stop)
        n_c = stop - start
        flat = n_c * n_slots
        slots = mci_index[idx]
        valid = slots[..., 0] >= 0
        first = np.where(valid, slots[..., 0], 0).reshape(flat)
        second = np.where(valid, slots[..., 1], 0).reshape(flat)
        third = np.where(valid, slots[..., 2], 0).reshape(flat)
        f_cut = np.repeat(f_abs_hz[idx], n_slots)
        d1 = f_abs_hz[first] - f_cut
        d2 = f_abs_hz[second] - f_cut
        d3 = f_abs_hz[third] - f_cut

        alpha_x = _mci_stimulus_attenuation(
            alpha_npm,
            first,
            second,
            third,
            np.repeat(alpha_npm[idx], n_slots),
            convention=nsp_convention,
        )
        residual_view = None
        magnitude = None
        expm1_neg = None
        if use_isrs:
            scale = 0.5 if nsp_convention == "gn" else 1.0
            alpha_x = alpha_x + scale * (
                rate_profile[first]
                + rate_profile[second]
                + rate_profile[third]
                - np.repeat(rate_profile[idx], n_slots)
            )
            residual = np.empty((flat, 1, 1, gain_profile.shape[0]), dtype=np.float64)
            for seg in range(gain_profile.shape[0]):
                gain_seg = gain_profile[seg]
                residual[..., seg] = (
                    (
                        gain_seg[first]
                        * gain_seg[second]
                        * gain_seg[third]
                        / gain_seg[np.repeat(np.arange(start, stop), n_slots)]
                    )
                    ** scale
                    * np.exp(
                        scale
                        * (
                            rate_profile[first]
                            + rate_profile[second]
                            + rate_profile[third]
                            - np.repeat(rate_profile[idx], n_slots)
                        )
                        * segment_positions[seg]
                    )
                ).reshape(flat, 1, 1)
            residual_view = residual
            decay = float(isrs_seg_length_m)
            exponent = alpha_x * decay
            magnitude = np.exp(-exponent)
            expm1_neg = np.expm1(-exponent)

        gamma2 = np.repeat(gamma_w_m[idx] ** 2, n_slots)
        integrand = np.empty(
            (flat * island_cells, 1, 1, 1, correlation_points), dtype=np.float64
        )

        acc = np.zeros(flat, dtype=np.float64)
        acc_dirichlet = None if dirichlet is None else np.zeros(flat, dtype=np.float64)
        for node, node_weight in band_nodes:
            shift = np.repeat(0.5 * b_hz[idx] * node, n_slots)[:, None]
            x1 = (d1 - half[first])[:, None] - shift
            x2 = (d1 + half[first])[:, None] - shift
            y1 = (d2 - half[second])[:, None] - shift
            y2 = (d2 + half[second])[:, None] - shift
            u1 = ((d3 - half[third])[:, None] - shift)[:, :, None]
            u2 = ((d3 + half[third])[:, None] - shift)[:, :, None]
            xa, xb, ylo, yhi = _island_cover(x1, x2, y1, y2, u1, u2, island_cells)
            x_center = 0.5 * (xa + xb)
            y_center = 0.5 * (ylo + yhi)
            beta2_eff = _effective_beta2(
                beta2_s2_m,
                beta3_s3_m,
                x_center,
                y_center,
                f_center_hz=np.repeat(f_abs_hz[idx] + 0.5 * b_hz[idx] * node, n_slots)[
                    :, None, None, None
                ],
                convention=dispersion_convention,
                f_reference_hz=f_reference_hz,
            )
            empty = ylo > yhi
            band_scale = (
                np.ones(flat, dtype=np.float64)
                if not integrate_band
                else np.repeat(node_weight * 0.5 * b_hz[idx], n_slots)
            )
            for a, b, c, d in _quadrant_pieces(xa, xb, ylo, yhi):
                xi_min = a * c
                xi_max = b * d
                with np.errstate(divide="ignore", invalid="ignore"):
                    xi_scale = alpha_x.reshape(flat, 1, 1)[..., None] / (
                        four_pi2 * np.abs(beta2_eff)
                    )
                xi_scale = np.where(
                    np.isfinite(xi_scale) & (xi_scale > 0.0), xi_scale, np.inf
                )
                xi_cut = np.minimum(xi_max, xi_min + TRUNCATION_FACTOR * xi_scale)
                xi_end = np.maximum(xi_cut, xi_min)
                xi_min = np.where(empty, 0.0, xi_min)
                xi_end = np.where(empty, 0.0, xi_end)
                xi_span = xi_end - xi_min

                span2 = xi_span.reshape(flat, island_cells)
                live_rows, live_cells = np.nonzero(span2 > 0.0)
                if live_rows.size == 0:
                    continue
                n_live = int(live_rows.size)
                a_flat = np.ascontiguousarray(
                    a.reshape(flat, island_cells)[live_rows, live_cells]
                )[:, None]
                b_flat = np.ascontiguousarray(
                    b.reshape(flat, island_cells)[live_rows, live_cells]
                )[:, None]
                c_flat = np.ascontiguousarray(
                    c.reshape(flat, island_cells)[live_rows, live_cells]
                )[:, None, None, None]
                d_flat = np.ascontiguousarray(
                    d.reshape(flat, island_cells)[live_rows, live_cells]
                )[:, None, None, None]
                alpha_flat = np.ascontiguousarray(alpha_x.reshape(flat)[live_rows])[
                    :, None, None
                ]
                beta_flat = np.ascontiguousarray(
                    beta2_eff.reshape(flat, island_cells)[live_rows, live_cells]
                )[:, None, None, None]
                xi_min_flat = np.ascontiguousarray(
                    xi_min.reshape(flat, island_cells)[live_rows, live_cells]
                )[:, None, None, None]
                xi_span_flat = np.ascontiguousarray(span2[live_rows, live_cells])[
                    :, None, None, None
                ]
                xi_end_flat = np.ascontiguousarray(
                    xi_end.reshape(flat, island_cells)[live_rows, live_cells]
                )[:, None, None, None]
                gamma_live = np.ascontiguousarray(gamma2.reshape(flat)[live_rows])
                band_live = band_scale.reshape(flat)[live_rows]
                out = integrand[:n_live]
                if use_isrs:
                    assert residual_view is not None
                    assert magnitude is not None
                    assert expm1_neg is not None
                    _fused_integrand_isrs(
                        out,
                        alpha_flat,
                        np.ascontiguousarray(
                            residual_view.reshape(flat, residual_view.shape[3])[
                                live_rows
                            ]
                        )[:, None, None, :],
                        np.ascontiguousarray(magnitude.reshape(flat)[live_rows])[
                            :, None, None
                        ],
                        np.ascontiguousarray(expm1_neg.reshape(flat)[live_rows])[
                            :, None, None
                        ],
                        beta_flat,
                        xi_min_flat,
                        xi_span_flat,
                        quad_u,
                        quad_w,
                        a_flat,
                        b_flat,
                        c_flat,
                        d_flat,
                        gamma_live,
                        float(isrs_seg_length_m),
                        four_pi2,
                    )
                else:
                    _fused_integrand_plain(
                        out,
                        alpha_flat,
                        float(length_m),
                        beta_flat,
                        xi_min_flat,
                        xi_span_flat,
                        quad_u,
                        quad_w,
                        a_flat,
                        b_flat,
                        c_flat,
                        d_flat,
                        gamma_live,
                        four_pi2,
                    )
                values = band_live * np.sum(out, axis=(-2, -1)).reshape(n_live)
                acc += np.bincount(live_rows, weights=values, minlength=flat)
                if acc_dirichlet is not None:
                    nodes, _ = _xi_quadrature(
                        xi_min_flat,
                        xi_end_flat,
                        correlation_points,
                    )
                    theta = four_pi2 * beta_flat[..., None] * float(length_m) * nodes
                    weighted = out * _dirichlet_weight(theta, n_coherent_spans)
                    weighted = np.where(np.isfinite(weighted), weighted, 0.0)
                    dirichlet_values = band_live * np.sum(
                        weighted, axis=(-2, -1)
                    ).reshape(n_live)
                    acc_dirichlet += np.bincount(
                        live_rows, weights=dirichlet_values, minlength=flat
                    )
        acc = np.where(np.isfinite(acc), acc, 0.0)
        if acc_dirichlet is not None:
            acc_dirichlet = np.where(np.isfinite(acc_dirichlet), acc_dirichlet, 0.0)
        return acc.reshape(n_c, n_slots), (
            None if acc_dirichlet is None else acc_dirichlet.reshape(n_c, n_slots)
        )

    ranges = [(start, min(start + chunk, n_ch)) for start in range(0, n_ch, chunk)]
    workers = _core_workers(len(ranges))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(lambda item: _chunk_body(*item), ranges))
    else:
        results = [_chunk_body(*item) for item in ranges]
    for (start, stop), (acc, acc_dirichlet) in zip(ranges, results, strict=True):
        core[start:stop] = acc
        if dirichlet is not None:
            dirichlet[start:stop] = acc_dirichlet
    return core, dirichlet


def _mci_from_core(
    mci_core: NDArrayFloat,
    mci_index: NDArrayInt,
    mci_weight: NDArrayFloat,
    g_psd: NDArrayFloat,
    b_hz: NDArrayFloat,
    apply_bandwidth: bool = True,
) -> NDArrayFloat:
    """Aggregate MCI core integrals, wave PSDs, and symmetry weights into powers."""
    g_pad = np.concatenate([[0.0], np.asarray(g_psd, dtype=np.float64)])
    padded = np.where(mci_index >= 0, mci_index + 1, 0)
    values = (
        mci_core
        * mci_weight
        * g_pad[padded[..., 0]]
        * g_pad[padded[..., 1]]
        * g_pad[padded[..., 2]]
    )
    total = (16.0 / 27.0) * values.sum(axis=1)
    if apply_bandwidth:
        return cast("NDArrayFloat", total * b_hz)
    return cast("NDArrayFloat", total)


@dataclass(frozen=True)
class _GridView:
    """Channel geometry for the full grid or an active-channel subset.

    Indices and slot tables use view-local coordinates; ``act`` and ``inv`` map
    between the view and full grid. Sparse views omit zero-power contributions."""

    act: NDArrayInt
    inv: NDArrayInt
    f_abs_hz: NDArrayFloat
    b_hz: NDArrayFloat
    f_rel_hz: NDArrayFloat
    alpha_npm: NDArrayFloat
    gamma_w_m: NDArrayFloat
    r_index: NDArrayInt
    r_index_padded: NDArrayInt
    mci_index: NDArrayInt
    mci_weight: NDArrayFloat
    sparse: bool = False


@dataclass(frozen=True)
class _CoreEntry:
    """Cached single-span core, MCI geometry, and optional coherent phase ramps."""

    plain: NDArrayFloat
    mci_plain: NDArrayFloat
    mci_index: NDArrayInt
    mci_weight: NDArrayFloat
    view: _GridView
    ramp_spm: NDArrayFloat | None = None
    ramp_xpm: NDArrayFloat | None = None


class CGNSpanKernel:
    """Single-span correlation-based GN kernel with optional ISRS.

    Core integrals are cached by span length and coherent span count, and by
    canonical absolute launch power when ISRS is enabled. Output power is always
    recomputed. Active-channel views discard zero-power contributions; different
    summation order can cause small floating-point differences.

    Parameters
    ----------
    fiber_config : FiberSpanConfig
        Fiber parameters.
    f_abs_hz, b_hz : numpy.ndarray
        Channel frequencies and bandwidths in Hz.
    correlation_points : int, optional
        One-dimensional correlation quadrature nodes.
    island_cells : int, optional
        Rectangular slices per integration island.
    include_isrs : bool, optional
        Include ISRS in the power and stimulus profiles.
    dispersion_convention : {"paper", "repo", "paper_absolute"}, optional
        Third-order dispersion expansion convention.
    nsp_convention : {"gn", "paper_product"}, optional
        NSP attenuation exponent convention.
    isrs_segments : int, optional
        Number of segments in the ISRS profile.
    band_points : int, optional
        Observation-frequency quadrature count; one uses center PSD.
    band_center_hz : float or None, optional
        Band center frequency in Hz, or None to use the channel mean."""

    def __init__(
        self,
        *,
        fiber_config: FiberSpanConfig,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        correlation_points: int = DEFAULT_CORRELATION_POINTS,
        island_cells: int = DEFAULT_ISLAND_CELLS,
        include_isrs: bool = True,
        dispersion_convention: str = DISPERSION_CONVENTION_DEFAULT,
        nsp_convention: str = NSP_CONVENTION_DEFAULT,
        isrs_segments: int = DEFAULT_ISRS_SEGMENTS,
        band_points: int = DEFAULT_BAND_POINTS,
        band_center_hz: float | None = None,
    ) -> None:
        """Initialize per-channel fiber properties and validate quadrature settings."""
        if correlation_points < 4:
            msg = f"correlation_points must be >= 4, got {correlation_points}."
            raise ValueError(msg)
        if island_cells < 1:
            msg = f"island_cells must be >= 1, got {island_cells}."
            raise ValueError(msg)
        if isrs_segments < 1:
            msg = f"isrs_segments must be >= 1, got {isrs_segments}."
            raise ValueError(msg)
        if band_points < 1:
            msg = f"band_points must be >= 1, got {band_points}."
            raise ValueError(msg)
        if dispersion_convention not in DISPERSION_CONVENTIONS:
            msg = (
                f"dispersion_convention must be one of {DISPERSION_CONVENTIONS}, "
                f"got {dispersion_convention!r}."
            )
            raise ValueError(msg)
        if nsp_convention not in NSP_CONVENTIONS:
            msg = f"nsp_convention must be one of {NSP_CONVENTIONS}, got {nsp_convention!r}."
            raise ValueError(msg)

        self.fiber_config = fiber_config
        self.f_abs_hz = np.ascontiguousarray(f_abs_hz, dtype=np.float64)
        self.b_hz = np.ascontiguousarray(b_hz, dtype=np.float64)
        self.n_nodes = int(self.f_abs_hz.shape[0])
        self.correlation_points = int(correlation_points)
        self.island_cells = int(island_cells)
        self.include_isrs = bool(include_isrs)
        self.dispersion_convention = str(dispersion_convention)
        self.nsp_convention = str(nsp_convention)
        self.isrs_segments = int(isrs_segments)
        self.band_points = int(band_points)

        lambdas_m = C_LIGHT / self.f_abs_hz
        if fiber_config.use_attenuation_polynomial:
            self.alpha_npm = _calc_fiber_attenuation_npm(
                lambdas_m,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            alpha = fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            self.alpha_npm = np.full(self.n_nodes, alpha, dtype=np.float64)
        a_eff_m2 = _calc_aeff_dynamic(
            lambdas_m,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        self.gamma_w_m = _nonlinearity_coefficient(
            lambdas_m,
            fiber_config.nonlinear_index_n2,
            a_eff_m2,
        )
        ref_lambda_m = fiber_config.reference_wavelength_nm * 1e-9
        self.f_reference_hz = C_LIGHT / ref_lambda_m
        self.beta2_s2_m = float(
            _calc_beta2_from_d(fiber_config.dispersion_parameter_d, ref_lambda_m)
        )
        self.beta3_s3_m = float(
            _calc_beta3_from_s_d(
                fiber_config.dispersion_slope_s,
                fiber_config.dispersion_parameter_d,
                ref_lambda_m,
            )
        )

        center = (
            float(np.mean(self.f_abs_hz))
            if band_center_hz is None
            else float(band_center_hz)
        )
        self.band_center_hz = center
        self.f_rel_hz = np.ascontiguousarray(self.f_abs_hz - center, dtype=np.float64)
        self.f_m_hz = float(np.min(self.f_abs_hz - 0.5 * self.b_hz) - center)
        self.f_M_hz = float(np.max(self.f_abs_hz + 0.5 * self.b_hz) - center)
        if any(pump.power_w > 0.0 for pump in fiber_config.raman_pumps):
            msg = (
                "The cGN implementation does not support active Raman pumps. "
                "The ECZ k_bar parameter is dimensionless launch pre-emphasis, "
                "not pump power."
            )
            raise ValueError(msg)
        self.k_bar = 0.0

        self._r_index, self._n_slots = _third_wave_slots(
            self.f_abs_hz,
            self.b_hz,
            widen_by_cut_bandwidth=self.band_points > 1,
        )
        self._r_index_padded = np.where(self._r_index >= 0, self._r_index + 1, 0)
        self.n_coherent_spans = 1
        self._core_cache: dict[tuple[float, int, bytes | None, bytes], _CoreEntry] = {}
        self._core_cache_bytes = 0
        self._mci_table_cache: tuple[NDArrayInt, NDArrayFloat] | None = None
        self._view_cache: dict[bytes, _GridView] = {}
        self._full_view = _GridView(
            act=np.arange(self.n_nodes, dtype=np.int64),
            inv=np.arange(self.n_nodes, dtype=np.int64),
            f_abs_hz=self.f_abs_hz,
            b_hz=self.b_hz,
            f_rel_hz=self.f_rel_hz,
            alpha_npm=self.alpha_npm,
            gamma_w_m=self.gamma_w_m,
            r_index=self._r_index,
            r_index_padded=self._r_index_padded,
            mci_index=np.zeros((self.n_nodes, 0, 3), dtype=np.int64),
            mci_weight=np.zeros((self.n_nodes, 0), dtype=np.float64),
        )

    def _minimum_abs_beta2(self) -> float:
        """Bound dispersion over the occupied band for conservative MCI screening."""
        if self.dispersion_convention != "paper":
            return 0.0
        edges = np.array(
            [
                np.min(self.f_abs_hz - self.b_hz / 2),
                np.max(self.f_abs_hz + self.b_hz / 2),
            ]
        )
        beta = self.beta2_s2_m + 2 * np.pi * self.beta3_s3_m * (
            edges - self.f_reference_hz
        )
        if beta[0] * beta[1] <= 0:
            return 0.0
        return float(np.min(np.abs(beta)))

    def _mci_table(self) -> tuple[NDArrayInt, NDArrayFloat]:
        """Build and cache the power-independent MCI geometry and symmetry weights."""
        if self._mci_table_cache is None:
            index, weight = _mci_slot_table(
                self.f_abs_hz,
                self.b_hz,
                self.alpha_npm,
                self._minimum_abs_beta2(),
            )
            self._mci_table_cache = (index, weight)
        return self._mci_table_cache

    def _subset_view(self, act: NDArrayInt) -> _GridView:
        """Return a cached view of active channels using filtered full-grid geometry.

        Zero-power slots contribute nothing, although their removal changes summation
        order and can introduce small rounding differences."""
        key = np.ascontiguousarray(act, dtype=np.int64).tobytes()
        cached = self._view_cache.get(key)
        if cached is not None:
            return cached
        n_full = self.n_nodes
        n = int(act.size)
        inv = np.full(n_full, -1, dtype=np.int64)
        inv[act] = np.arange(n, dtype=np.int64)
        is_active = np.zeros(n_full, dtype=bool)
        is_active[act] = True

        sub = self._r_index[np.ix_(act, act)]
        safe = np.where(sub >= 0, sub, 0)
        keep = (sub >= 0) & is_active[safe]
        counts = keep.sum(axis=2)
        n_slots = int(counts.max()) if counts.size else 0
        if n_slots:
            order = np.argsort(~keep, axis=2, kind="stable")
            ordered = np.take_along_axis(sub, order, axis=2)[..., :n_slots]
            valid = np.arange(n_slots) < counts[..., None]
            mapped = inv[np.where(ordered >= 0, ordered, 0)]
            r_index = np.where(valid, mapped, -1).astype(np.int64)
        else:
            r_index = np.full((n, n, 0), -1, dtype=np.int64)
        r_index_padded = np.where(r_index >= 0, r_index + 1, 0)

        mci_index_full, mci_weight_full = self._mci_table()
        mci_index = np.full((n, 0, 3), -1, dtype=np.int64)
        mci_weight = np.zeros((n, 0), dtype=np.float64)
        if mci_index_full.shape[1]:
            tri = mci_index_full[act]
            weight = mci_weight_full[act]
            valid_tri = (tri >= 0).all(axis=2)
            safe_tri = np.where(valid_tri[:, :, None], tri, 0)
            keep_tri = (
                valid_tri
                & is_active[safe_tri[..., 0]]
                & is_active[safe_tri[..., 1]]
                & is_active[safe_tri[..., 2]]
            )
            counts_tri = keep_tri.sum(axis=1)
            m_slots = int(counts_tri.max()) if counts_tri.size else 0
            if m_slots:
                order = np.argsort(~keep_tri, axis=1, kind="stable")
                ordered = np.take_along_axis(tri, order[:, :, None], axis=1)[
                    :, :m_slots
                ]
                ordered_w = np.take_along_axis(weight, order, axis=1)[:, :m_slots]
                valid_m = np.arange(m_slots) < counts_tri[:, None]
                mapped = inv[np.where(ordered >= 0, ordered, 0)]
                mci_index = np.where(valid_m[:, :, None], mapped, -1).astype(np.int64)
                mci_weight = np.where(valid_m, ordered_w, 0.0)

        view = _GridView(
            act=np.ascontiguousarray(act, dtype=np.int64),
            inv=inv,
            f_abs_hz=np.ascontiguousarray(self.f_abs_hz[act]),
            b_hz=np.ascontiguousarray(self.b_hz[act]),
            f_rel_hz=np.ascontiguousarray(self.f_rel_hz[act]),
            alpha_npm=np.ascontiguousarray(self.alpha_npm[act]),
            gamma_w_m=np.ascontiguousarray(self.gamma_w_m[act]),
            r_index=r_index,
            r_index_padded=r_index_padded,
            mci_index=mci_index,
            mci_weight=mci_weight,
            sparse=True,
        )
        self._view_cache[key] = view
        while len(self._view_cache) > _VIEW_CACHE_ENTRIES:
            self._view_cache.pop(next(iter(self._view_cache)))
        return view

    def _span_output_power(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> NDArrayFloat:
        """Calculate output power for the current launch power without caching it."""
        length = float(span_length_m)
        if self.include_isrs:
            total_w = float(power.sum())
            r_f = _calc_raman_profile_jit(
                self.f_rel_hz,
                self.f_m_hz,
                self.f_M_hz,
                DELTA_F_CO_HZ,
                total_w,
            )
            return cast(
                "NDArrayFloat",
                _calc_span_out_power_jit(
                    np.ascontiguousarray(power, dtype=np.float64),
                    self.f_rel_hz,
                    np.ascontiguousarray(self.alpha_npm, dtype=np.float64),
                    length,
                    np.ascontiguousarray(r_f, dtype=np.float64),
                    float(self.k_bar),
                ),
            )
        return cast("NDArrayFloat", power * np.exp(-self.alpha_npm * length))

    def _span_output_power_view(
        self,
        view: _GridView,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> NDArrayFloat:
        """Calculate output power for channels in the supplied active view."""
        length = float(span_length_m)
        if self.include_isrs:
            total_w = float(power.sum())
            r_f = _calc_raman_profile_jit(
                view.f_rel_hz,
                self.f_m_hz,
                self.f_M_hz,
                DELTA_F_CO_HZ,
                total_w,
            )
            return cast(
                "NDArrayFloat",
                _calc_span_out_power_jit(
                    np.ascontiguousarray(power, dtype=np.float64),
                    view.f_rel_hz,
                    np.ascontiguousarray(view.alpha_npm, dtype=np.float64),
                    length,
                    np.ascontiguousarray(r_f, dtype=np.float64),
                    float(self.k_bar),
                ),
            )
        return cast("NDArrayFloat", power * np.exp(-view.alpha_npm * length))

    def _core_matrix(
        self,
        view: _GridView,
        canonical_power: NDArrayFloat,
        span_length_m: float,
        n_coherent_spans: int,
    ) -> tuple[
        NDArrayFloat,
        NDArrayFloat | None,
        NDArrayFloat,
        NDArrayFloat | None,
        NDArrayInt,
        NDArrayFloat,
    ]:
        """Build single-span and coherent island and MCI core integrals for a view.

        ISRS uses the canonical launch power that also determines the cache key."""
        if view.sparse:
            mci_index, mci_weight = view.mci_index, view.mci_weight
        else:
            mci_index, mci_weight = self._mci_table()
        if self.include_isrs:
            z_mid, gain, rate = _isrs_span_profile(
                view.f_rel_hz,
                view.alpha_npm,
                canonical_power,
                self.f_m_hz,
                self.f_M_hz,
                self.k_bar,
                float(span_length_m),
                self.isrs_segments,
            )
            plain, dirichlet = _island_core(
                f_abs_hz=view.f_abs_hz,
                b_hz=view.b_hz,
                alpha_npm=view.alpha_npm,
                gamma_w_m=view.gamma_w_m,
                beta2_s2_m=self.beta2_s2_m,
                beta3_s3_m=self.beta3_s3_m,
                dispersion_convention=self.dispersion_convention,
                f_reference_hz=self.f_reference_hz,
                nsp_convention=self.nsp_convention,
                length_m=float(span_length_m),
                r_index=view.r_index,
                correlation_points=self.correlation_points,
                island_cells=self.island_cells,
                n_coherent_spans=int(n_coherent_spans),
                isrs_gain=gain,
                isrs_rate=rate,
                isrs_z_mid=z_mid,
                isrs_seg_length_m=float(span_length_m) / float(self.isrs_segments),
                band_points=self.band_points,
                compact_live=True,
                compact_scatter=not view.sparse,
            )
            mci_plain, mci_dirichlet = _mci_core(
                f_abs_hz=view.f_abs_hz,
                b_hz=view.b_hz,
                alpha_npm=view.alpha_npm,
                gamma_w_m=view.gamma_w_m,
                beta2_s2_m=self.beta2_s2_m,
                beta3_s3_m=self.beta3_s3_m,
                dispersion_convention=self.dispersion_convention,
                f_reference_hz=self.f_reference_hz,
                nsp_convention=self.nsp_convention,
                length_m=float(span_length_m),
                mci_index=mci_index,
                correlation_points=self.correlation_points,
                island_cells=self.island_cells,
                n_coherent_spans=int(n_coherent_spans),
                isrs_gain=gain,
                isrs_rate=rate,
                isrs_z_mid=z_mid,
                isrs_seg_length_m=float(span_length_m) / float(self.isrs_segments),
                band_points=self.band_points,
            )
            return plain, dirichlet, mci_plain, mci_dirichlet, mci_index, mci_weight
        plain, dirichlet = _island_core(
            f_abs_hz=view.f_abs_hz,
            b_hz=view.b_hz,
            alpha_npm=view.alpha_npm,
            gamma_w_m=view.gamma_w_m,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            dispersion_convention=self.dispersion_convention,
            f_reference_hz=self.f_reference_hz,
            nsp_convention=self.nsp_convention,
            length_m=float(span_length_m),
            r_index=view.r_index,
            correlation_points=self.correlation_points,
            island_cells=self.island_cells,
            n_coherent_spans=int(n_coherent_spans),
            band_points=self.band_points,
            compact_live=True,
            compact_scatter=not view.sparse,
        )
        mci_plain, mci_dirichlet = _mci_core(
            f_abs_hz=view.f_abs_hz,
            b_hz=view.b_hz,
            alpha_npm=view.alpha_npm,
            gamma_w_m=view.gamma_w_m,
            beta2_s2_m=self.beta2_s2_m,
            beta3_s3_m=self.beta3_s3_m,
            dispersion_convention=self.dispersion_convention,
            f_reference_hz=self.f_reference_hz,
            nsp_convention=self.nsp_convention,
            length_m=float(span_length_m),
            mci_index=mci_index,
            correlation_points=self.correlation_points,
            island_cells=self.island_cells,
            n_coherent_spans=int(n_coherent_spans),
            band_points=self.band_points,
        )
        return plain, dirichlet, mci_plain, mci_dirichlet, mci_index, mci_weight

    def _link_ramps(
        self,
        view: _GridView,
        plain: NDArrayFloat,
        dirichlet: NDArrayFloat,
        mci_plain: NDArrayFloat,
        mci_dirichlet: NDArrayFloat,
        mci_index: NDArrayInt,
        mci_weight: NDArrayFloat,
        canonical_power: NDArrayFloat,
        n_spans: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Invert link-to-single-span NLI ratios into per-channel phase steps.

        The steps reproduce integrated Dirichlet totals for equal-amplitude spans."""
        g_psd = np.where(canonical_power > 0.0, canonical_power / view.b_hz, 0.0)
        g_pad = np.concatenate([[0.0], g_psd])
        third_psd = g_pad[view.r_index_padded]
        apply_bandwidth = self.band_points == 1
        single_spm, single_xpm = _nli_from_core(
            plain, g_psd, third_psd, view.b_hz, view.r_index, apply_bandwidth
        )
        link_spm, link_xpm = _nli_from_core(
            dirichlet, g_psd, third_psd, view.b_hz, view.r_index, apply_bandwidth
        )
        single_xpm = single_xpm + _mci_from_core(
            mci_plain, mci_index, mci_weight, g_psd, view.b_hz, apply_bandwidth
        )
        link_xpm = link_xpm + _mci_from_core(
            mci_dirichlet, mci_index, mci_weight, g_psd, view.b_hz, apply_bandwidth
        )
        ramp_spm = _equivalent_phase_ramp(
            n_spans, _coherent_ratio(link_spm, single_spm)
        )
        ramp_xpm = _equivalent_phase_ramp(
            n_spans, _coherent_ratio(link_xpm, single_xpm)
        )
        return ramp_spm, ramp_xpm

    def link_nli(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Calculate the Dirichlet-weighted link SCI and XCI powers in watts.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Channel launch powers in watts.
        span_length_m : float
            Span length in meters.
        n_spans : int
            Number of identical coherent spans.

        Returns
        -------
        tuple of numpy.ndarray
            Link-level SCI and XCI powers in watts."""
        canonical, _ = _canonical_power(power_in_w)
        view = self._full_view
        plain, dirichlet, mci_plain, mci_dirichlet, mci_index, mci_weight = (
            self._core_matrix(view, canonical, span_length_m, int(n_spans))
        )
        g_psd = np.where(canonical > 0.0, canonical / view.b_hz, 0.0)
        g_pad = np.concatenate([[0.0], g_psd])
        third_psd = g_pad[view.r_index_padded]
        apply_bandwidth = self.band_points == 1
        if dirichlet is None:
            spm, xpm = _nli_from_core(
                plain, g_psd, third_psd, view.b_hz, view.r_index, apply_bandwidth
            )
            mci_source = mci_plain
        else:
            spm, xpm = _nli_from_core(
                dirichlet, g_psd, third_psd, view.b_hz, view.r_index, apply_bandwidth
            )
            mci_source = cast("NDArrayFloat", mci_dirichlet)
        xpm = xpm + _mci_from_core(
            mci_source, mci_index, mci_weight, g_psd, view.b_hz, apply_bandwidth
        )
        return spm, xpm

    def _core_for(
        self,
        power: NDArrayFloat,
        span_length_m: float,
        n_coherent_spans: int,
    ) -> _CoreEntry:
        """Return the cached core and ramps for the span and canonical absolute power."""
        canonical_full, power_key = _canonical_power(power)
        active = power > 0.0
        if bool(active.all()):
            view = self._full_view
            canonical = canonical_full
            view_key = b""
        else:
            view = self._subset_view(np.flatnonzero(active))
            canonical = canonical_full[view.act]
            view_key = view.act.tobytes()
        key = (
            float(span_length_m),
            int(n_coherent_spans),
            power_key if self.include_isrs else None,
            view_key,
        )
        cached = self._core_cache.get(key)
        if cached is None:
            (
                plain,
                dirichlet,
                mci_plain,
                mci_dirichlet,
                mci_index,
                mci_weight,
            ) = self._core_matrix(view, canonical, span_length_m, n_coherent_spans)
            if dirichlet is None:
                cached = _CoreEntry(
                    plain=plain,
                    mci_plain=mci_plain,
                    mci_index=mci_index,
                    mci_weight=mci_weight,
                    view=view,
                )
            else:
                ramp_spm, ramp_xpm = self._link_ramps(
                    view,
                    plain,
                    dirichlet,
                    mci_plain,
                    cast("NDArrayFloat", mci_dirichlet),
                    mci_index,
                    mci_weight,
                    canonical,
                    int(n_coherent_spans),
                )
                cached = _CoreEntry(
                    plain=plain,
                    mci_plain=mci_plain,
                    mci_index=mci_index,
                    mci_weight=mci_weight,
                    ramp_spm=ramp_spm,
                    ramp_xpm=ramp_xpm,
                    view=view,
                )
            self._store_core(key, cached)
        return cached

    def _store_core(
        self, key: tuple[float, int, bytes | None, bytes], entry: _CoreEntry
    ) -> None:
        """Cache a core entry and evict old entries to respect the memory budget."""
        size = int(entry.plain.nbytes)
        if size > CORE_CACHE_BUDGET_BYTES:
            return
        while (
            self._core_cache and self._core_cache_bytes + size > CORE_CACHE_BUDGET_BYTES
        ):
            oldest = next(iter(self._core_cache))
            self._core_cache_bytes -= int(self._core_cache[oldest].plain.nbytes)
            del self._core_cache[oldest]
        self._core_cache[key] = entry
        self._core_cache_bytes += size

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Evaluate fiber output and SCI/XCI noise for one span.

        A homogeneous multispan context produces equivalent coherent amplitudes.
        General route coherence is integrated by the route-level span-pair evaluator.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Channel launch powers in watts; zero means inactive.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Unused; the kernel's coherent span count is set by the model.
        span_context : SpanContext or None, optional
            Position within the link for the amplitude phase.

        Returns
        -------
        SpanFiberOutput
            Output powers, noise powers, and optional coherent amplitudes."""
        del n_effective_spans
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            msg = f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            raise ValueError(msg)
        if span_length_m <= 0.0:
            msg = f"span_length_m must be positive, got {span_length_m}."
            raise ValueError(msg)

        entry = self._core_for(power, span_length_m, self.n_coherent_spans)
        view = entry.view
        active = power > 0.0
        if view.sparse:
            act = view.act
            power_out = np.zeros(self.n_nodes, dtype=np.float64)
            power_out[act] = self._span_output_power_view(
                view, power[act], span_length_m
            )
            g_psd = np.where(power[act] > 0.0, power[act] / view.b_hz, 0.0)
        else:
            power_out = self._span_output_power(power, span_length_m)
            g_psd = np.where(active, power / self.b_hz, 0.0)
        g_pad = np.concatenate([[0.0], g_psd])
        third_psd = g_pad[view.r_index_padded]
        spm_view, xpm_view = _nli_from_core(
            entry.plain,
            g_psd,
            third_psd,
            view.b_hz,
            view.r_index,
            self.band_points == 1,
        )
        xpm_view = xpm_view + _mci_from_core(
            entry.mci_plain,
            entry.mci_index,
            entry.mci_weight,
            g_psd,
            view.b_hz,
            self.band_points == 1,
        )
        if view.sparse:
            spm_noise = np.zeros(self.n_nodes, dtype=np.float64)
            xpm_noise = np.zeros(self.n_nodes, dtype=np.float64)
            spm_noise[view.act] = spm_view
            xpm_noise[view.act] = xpm_view
        else:
            spm_noise = np.where(active, spm_view, 0.0)
            xpm_noise = np.where(active, xpm_view, 0.0)

        if (
            span_context is None
            or span_context.total_spans <= 1
            or self.n_coherent_spans <= 1
            or entry.ramp_spm is None
            or entry.ramp_xpm is None
        ):
            return SpanFiberOutput(
                power_out_w=power_out,
                spm_noise_w=spm_noise,
                xpm_noise_w=xpm_noise,
            )

        scale = span_context.cumulative_length_m / float(span_length_m)
        if view.sparse:
            phase_spm = np.zeros(self.n_nodes, dtype=np.float64)
            phase_xpm = np.zeros(self.n_nodes, dtype=np.float64)
            phase_spm[view.act] = entry.ramp_spm * scale
            phase_xpm[view.act] = entry.ramp_xpm * scale
        else:
            phase_spm = entry.ramp_spm * scale
            phase_xpm = entry.ramp_xpm * scale
        amplitude_spm = np.sqrt(spm_noise) * np.exp(1j * phase_spm)
        amplitude_xpm = np.sqrt(xpm_noise) * np.exp(1j * phase_xpm)
        return SpanFiberOutput(
            power_out_w=power_out,
            spm_noise_w=spm_noise,
            xpm_noise_w=xpm_noise,
            nli_amplitude_spm=amplitude_spm.astype(np.complex128),
            nli_amplitude_xpm=amplitude_xpm.astype(np.complex128),
        )


def _span_plan(
    optical_path: MultiSpanOpticalPath,
) -> list[tuple[FiberSpanConfig, float]]:
    """Expand each path fiber into span configurations and lengths in meters."""
    plan: list[tuple[FiberSpanConfig, float]] = []
    for fiber in optical_path.fiber_configs:
        remaining = fiber.length_km
        while remaining > 1e-12:
            span_km = min(remaining, fiber.max_span_length_km)
            plan.append((fiber, span_km * 1e3))
            remaining -= span_km
    return plan


def _identical_span_count(plan: list[tuple[FiberSpanConfig, float]]) -> int:
    """Return the span count for the identical-span Dirichlet calculation."""
    if not plan:
        return 1
    first_fiber, first_length = plan[0]
    key = first_fiber.model_dump_json()
    for fiber, length in plan[1:]:
        if fiber.model_dump_json() != key or abs(length - first_length) > 1e-9:
            return 1
    return len(plan)


class CGNModel(RouteModel):
    """Route model for correlation-based GN with optional ISRS.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Channel grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber configuration.
    symbol_rate_hz : float, optional
        Symbol rate in Hz.
    correlation_points : int, optional
        Correlation-domain quadrature nodes.
    include_isrs : bool, optional
        Include ISRS power tilt.
    dispersion_convention : {"paper", "repo", "paper_absolute"}, optional
        Third-order dispersion expansion convention.
    nsp_convention : {"gn", "paper_product"}, optional
        Stimulus profile exponent convention.
    isrs_segments : int, optional
        Number of ISRS profile segments.
    band_points : int, optional
        Observation-frequency quadrature count; one uses center PSD including MCI.
    egn : bool, optional
        Must be False; this model implements GN only.
    **route_options
        Options passed to the route model.

    Raises
    ------
    ValueError
        If EGN or invalid quadrature or convention settings are requested."""

    info = CGN_MODEL_INFO

    supports_span_context = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        correlation_points: int = DEFAULT_CORRELATION_POINTS,
        include_isrs: bool = True,
        dispersion_convention: str = DISPERSION_CONVENTION_DEFAULT,
        nsp_convention: str = NSP_CONVENTION_DEFAULT,
        isrs_segments: int = DEFAULT_ISRS_SEGMENTS,
        band_points: int = DEFAULT_BAND_POINTS,
        egn: bool = False,
        **route_options: Any,
    ) -> None:
        """Initialize route and correlation model options."""
        if egn:
            msg = (
                "The correlation-based GN model does not support egn=True: the cGN paper "
                "covers the GN model only. Use egn=False."
            )
            raise ValueError(msg)
        if correlation_points < 4:
            msg = f"correlation_points must be >= 4, got {correlation_points}."
            raise ValueError(msg)
        if dispersion_convention not in DISPERSION_CONVENTIONS:
            msg = (
                f"dispersion_convention must be one of {DISPERSION_CONVENTIONS}, "
                f"got {dispersion_convention!r}."
            )
            raise ValueError(msg)
        if nsp_convention not in NSP_CONVENTIONS:
            msg = f"nsp_convention must be one of {NSP_CONVENTIONS}, got {nsp_convention!r}."
            raise ValueError(msg)
        if isrs_segments < 1:
            msg = f"isrs_segments must be >= 1, got {isrs_segments}."
            raise ValueError(msg)
        if band_points < 1:
            msg = f"band_points must be >= 1, got {band_points}."
            raise ValueError(msg)
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.egn = False
        self.correlation_points = int(correlation_points)
        self.island_cells = DEFAULT_ISLAND_CELLS
        self.include_isrs = bool(include_isrs)
        self.dispersion_convention = str(dispersion_convention)
        self.nsp_convention = str(nsp_convention)
        self.isrs_segments = int(isrs_segments)
        self.band_points = int(band_points)
        self._n_coherent_spans: int | None = None
        self._route_spans: list[tuple[CGNSpanKernel, NDArrayFloat, float]] | None = None

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Build the fiber-specific single-span kernel."""
        return CGNSpanKernel(
            fiber_config=fiber_config,
            f_abs_hz=self.f_abs_hz,
            b_hz=self.channel_bandwidth_hz_channels,
            correlation_points=self.correlation_points,
            island_cells=self.island_cells,
            include_isrs=self.include_isrs,
            dispersion_convention=self.dispersion_convention,
            nsp_convention=self.nsp_convention,
            isrs_segments=self.isrs_segments,
            band_points=self.band_points,
            band_center_hz=self.band_center_hz,
        )

    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> PathQoTResult:
        """Evaluate coherent NLI with each span's dispersion and receiver transfer."""
        from network_nlo_eval.models.cgn_link import coherent_route_nli  # noqa: PLC0415

        previous = self._n_coherent_spans
        previous_spans = self._route_spans
        self._n_coherent_spans = _identical_span_count(_span_plan(optical_path))
        self._route_spans = []
        try:
            result = super().evaluate_path(optical_path, power_in_w)
            spans = self._route_spans
            if len(spans) <= 1:
                return result
            balanced = self._n_coherent_spans > 1 and all(
                np.allclose(power, spans[0][1], rtol=2e-14, atol=0)
                for _, power, _ in spans[1:]
            )
            balanced = balanced and not any(
                roadm.filtering_penalty_db != 0
                for roadm in (optical_path.roadm_configs or [])
            )
            if balanced:
                return result
            spm, xpm = coherent_route_nli(spans, result.signal_power_w, optical_path)
            noise = spm + xpm + result.ase_noise_w
            snr = np.full_like(noise, -np.inf)
            active = result.signal_power_w > 0
            noisy = active & (noise > 0)
            snr[noisy] = 10 * np.log10(result.signal_power_w[noisy] / noise[noisy])
            snr[active & (noise == 0)] = np.inf
            return replace(result, spm_noise_w=spm, xpm_noise_w=xpm, snr_db=snr)
        finally:
            self._n_coherent_spans = previous
            self._route_spans = previous_spans

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """Evaluate a span and apply amplifier and ASE effects.

        Direct calls outside path evaluation assume equal spans when context is given."""
        kernel = cast("CGNSpanKernel", self.kernel_for(fiber_config))
        if self._route_spans is not None:
            self._route_spans.append(
                (kernel, np.array(power_in_w, copy=True), span_length_m)
            )
        if self._n_coherent_spans is None:
            kernel.n_coherent_spans = (
                span_context.total_spans if span_context is not None else 1
            )
        else:
            kernel.n_coherent_spans = self._n_coherent_spans
        fiber_output = kernel.evaluate(
            power_in_w, span_length_m, 1, span_context=span_context
        )
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        return apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )


register_model(CGN_MODEL_INFO, CGNModel)
