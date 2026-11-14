"""Carena 2014 enhanced Gaussian noise (EGN) model.

The SCI, XCI, and MCI coefficients follow the paper's explicit integrals.
Fourth- and sixth-order terms are reduced to two-dimensional quadrature using
Appendix C. In particular, the two fourth-order SCI terms have different
integration domains: the first uses f1 as the outer variable and the second
uses f3. Their coefficients are 80/81 and 16/81 respectively.

The frequency dependence of the link function is through
``p = (f1-f)*(f2-f)``. Tables of the antiderivatives of ``mu`` and ``|mu|²``
allow the inner integral to be evaluated by interpolation. The table spacing
and outer quadrature subdivisions resolve the shortest inter-span oscillation
period. The second SCI fourth-order term uses product integration with
cell-averaged inverse-square-root weights; its resolution is controlled by
``L2_CELLS`` and ``L2_OUTER_DIVISOR``.

This implementation uses rectangular spectra and applies EGN-to-GN ratios to
the repository's closed-form GN baseline, which includes SRS. The paper
assumes equally spaced, non-overlapping channels, equal spans, lumped
amplification, and no Raman effect. EGN integrals use band-center fiber
parameters; irregular grids use median frequency offsets for each index
separation. XCI is truncated to a neighbor window and MCI uses only the
adjacent three-channel form, omitting the paper's Appendix B multi-channel
extension. Overlapping channels violate the frequency-region assumptions.

Inter-span coherence is compressed to one equivalent phase step per channel.
For equal spans and powers this reproduces the integrated link total but not
its spectral or pairwise phase structure. Unequal spans or powers can change
the accumulated total. Table size and computation grow with dispersion.

References
----------
A. Carena et al., "EGN model of non-linear fiber propagation", Optics
Express 22, 16335-16362 (2014), doi:10.1364/OE.22.016335."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, overload

import numpy as np
from numpy.polynomial.legendre import leggauss

from network_nlo_eval.core.constants import PI
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    SpanContext,
    SpanFiberOutput,
    SpanKernel,
    register_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.route import (
    RouteModel,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)

if TYPE_CHECKING:
    from network_nlo_eval.core.flexgrid import FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

_KappaValue = TypeVar("_KappaValue", float, "NDArrayFloat")

__all__ = [
    "CARENA2014_MODEL_INFO",
    "DEFAULT_FREQUENCY_POINTS",
    "DEFAULT_MODULATION",
    "DEFAULT_QUADRATURE_POINTS",
    "MODULATION_TABLE",
    "XCI_NEIGHBOR_SPAN",
    "Carena2014Model",
    "Carena2014SpanKernel",
    "EgnIntegrator",
    "LinkFunction",
    "MciKappas",
    "SciKappas",
    "XciKappas",
]

#:
#:
MODULATION_TABLE: dict[str, tuple[float, float]] = {
    "bpsk": (-1.0, 4.0),
    "qpsk": (-1.0, 4.0),
    "16qam": (-17.0 / 25.0, 52.0 / 25.0),
    "64qam": (-13.0 / 21.0, 1161.0 / 646.0),
}

DEFAULT_MODULATION = "64qam"

#:
DEFAULT_QUADRATURE_POINTS = 6

#:
#:
#:
#:
DEFAULT_FREQUENCY_POINTS = 41

#:
XCI_NEIGHBOR_SPAN = 8

#:
#:
#:
L2_CELLS = 256

#:
#:
L2_OUTER_DIVISOR = 8

#:
DIRICHLET_BISECTION_STEPS = 44

CARENA2014_MODEL_INFO = ModelInfo(
    name="carena2014",
    summary="Carena 2014 EGN model: GN backbone with quadrature-evaluated 4th/6th-order corrections.",
    family="egn",
    reference="Carena et al., Opt. Express 22(13):16335-16362, 2014, doi:10.1364/OE.22.016335",
    options=(
        (
            "modulation_format",
            "Modulation format key: bpsk / qpsk / 16qam / 64qam (2014 Table 1).",
        ),
        ("quadrature_points", "Gauss-Legendre nodes per composite sub-interval."),
        (
            "frequency_points",
            "Number of NLI-PSD sample frequencies across the CUT band.",
        ),
    ),
)


# =============================================================================
# =============================================================================
def _sinc_ratio(n_spans: int, psi: NDArrayFloat) -> NDArrayFloat:
    """Evaluate sin(n*psi)/sin(psi), using its limit at singular points.

    Parameters
    ----------
    n_spans : int
        Number of spans.
    psi : numpy.ndarray
        Phase in radians.

    Returns
    -------
    numpy.ndarray
        Real ratios with the shape of ``psi``."""
    s = np.sin(psi)
    small = np.abs(s) < 1e-12
    safe = np.where(small, 1.0, s)
    ratio = np.sin(n_spans * psi) / safe
    # lim_{psi->k pi} sin(n psi)/sin(psi) = n * (-1)^{(n-1)k} = n * cos((n-1) psi)
    return np.where(small, n_spans * np.cos((n_spans - 1) * psi), ratio)


@dataclass(frozen=True)
class LinkFunction:
    """Paper Eq. (10)-(12) link function, depending on p = (f1-f)*(f2-f).

    Attributes
    ----------
    gamma : float
        Nonlinear coefficient in 1/(W m).
    alpha_power_npm : float
        Fiber power attenuation in Np/m; field attenuation is half this value.
    beta2_s2_m : float
        Second-order dispersion in s²/m.
    span_length_m : float
        Span length in meters.
    n_spans : int
        Number of spans."""

    gamma: float
    alpha_power_npm: float
    beta2_s2_m: float
    span_length_m: float
    n_spans: int

    def mu(self, p_hz2: NDArrayFloat) -> NDArrayComplex:
        """Evaluate the complex link function at frequency products in Hz²."""
        p = np.asarray(p_hz2, dtype=np.float64)
        theta = 4.0 * PI**2 * self.beta2_s2_m * p * self.span_length_m
        zeta = (
            self.gamma
            * (
                1.0
                - np.exp(-self.alpha_power_npm * self.span_length_m)
                * np.exp(1j * theta)
            )
            / (self.alpha_power_npm - 1j * 4.0 * PI**2 * self.beta2_s2_m * p)
        )
        psi = 0.5 * theta
        nu = _sinc_ratio(self.n_spans, psi) * np.exp(1j * (self.n_spans - 1) * psi)
        return cast("NDArrayComplex", zeta * nu)

    def mu_of(self, f1: NDArrayFloat, f2: NDArrayFloat, f_hz: float) -> NDArrayComplex:
        """Evaluate the complex link function at broadcastable frequencies in Hz."""
        return self.mu(
            (np.asarray(f1, dtype=np.float64) - f_hz)
            * (np.asarray(f2, dtype=np.float64) - f_hz)
        )

    @property
    def psi_per_hz2(self) -> float:
        """Return 2*pi²*beta2*span_length, the phase coefficient in rad/Hz²."""
        return 2.0 * PI**2 * self.beta2_s2_m * self.span_length_m


# =============================================================================
# =============================================================================
def dirichlet_buildup(theta_rad: NDArrayFloat, n_spans: int) -> NDArrayFloat:
    """Calculate the equal-span coherent power factor for a phase step.

    Parameters
    ----------
    theta_rad : array_like
        Phase difference between adjacent spans in radians.
    n_spans : int
        Number of spans.

    Returns
    -------
    numpy.ndarray
        Dimensionless factor with the broadcast input shape."""
    theta = np.asarray(theta_rad, dtype=np.float64)
    if n_spans <= 1:
        return np.ones_like(theta)
    half = 0.5 * theta
    numerator = np.sin(float(n_spans) * half)
    denominator = np.sin(half)
    tiny = np.abs(denominator) < 1e-12
    safe = np.where(tiny, 1.0, denominator)
    ratio = np.where(
        tiny,
        float(n_spans) * np.cos((float(n_spans) - 1.0) * half),
        numerator / safe,
    )
    return cast("NDArrayFloat", ratio * ratio)


def solve_dirichlet_step(n_spans: int, target: NDArrayFloat) -> NDArrayFloat:
    """Invert the equal-span coherent factor to obtain phase steps in radians.

    The Dirichlet factor decreases from N² to zero on [0, 2*pi/N], so bisection
    finds an equivalent step. This compresses frequency-dependent coherence."""
    if n_spans <= 1:
        return np.zeros_like(np.asarray(target, dtype=np.float64))
    goal = np.clip(np.asarray(target, dtype=np.float64), 0.0, float(n_spans) ** 2)
    low = np.zeros_like(goal)
    high = np.full_like(goal, 2.0 * np.pi / n_spans)
    for _ in range(DIRICHLET_BISECTION_STEPS):
        middle = 0.5 * (low + high)
        too_large = dirichlet_buildup(middle, n_spans) > goal
        low = np.where(too_large, middle, low)
        high = np.where(too_large, high, middle)
    return 0.5 * (low + high)


def amplitude_from_power(power_w: NDArrayFloat) -> NDArrayComplex:
    """Convert nonnegative powers in watts to real-valued complex amplitudes.

    The square root and its adjacent floating-point values are considered to
    minimize rounding differences between squared magnitude and input power."""
    root = np.sqrt(np.asarray(power_w, dtype=np.float64))
    target = np.asarray(power_w, dtype=np.float64)
    bad = root * root != target
    if bool(np.any(bad)):
        for candidate in (np.nextafter(root, np.inf), np.nextafter(root, -np.inf)):
            square = candidate * candidate
            take = bad & (square == target)
            root = np.where(take, candidate, root)
            bad = bad & ~take
    return root.astype(np.complex128)


# =============================================================================
# =============================================================================
class _MuAntiderivative:
    """Tabulate antiderivatives of mu(p) and |mu(p)|² with Hermite interpolation.

    Each table cell is integrated with Gauss-Legendre quadrature. Table spacing
    resolves the shortest phase period pi/(N*|2*pi²*beta2*L|).

    Attributes
    ----------
    period : float
        Shortest oscillation period in Hz², or infinity without dispersion."""

    def __init__(self, link: LinkFunction, points: int, p_limit_hz2: float) -> None:
        """Build the antiderivative table for a link and frequency-product limit.

        Parameters
        ----------
        link : LinkFunction
            Link geometry and physical parameters.
        points : int
            Gauss-Legendre nodes per table cell.
        p_limit_hz2 : float
            Required absolute frequency-product limit in Hz²."""
        self.link = link
        self.points = int(points)
        c = abs(link.psi_per_hz2)
        n_spans = max(1, int(link.n_spans))
        limit = max(float(p_limit_hz2), 1.0)
        if c > 0.0:
            self.period = PI / (n_spans * c)
            step = self.period / (4.0 * self.points)
        else:
            self.period = np.inf
            step = 2.0 * limit / (8.0 * self.points)
        n_intervals = max(8, int(np.ceil(2.0 * limit / step)))
        edges = np.linspace(-limit, limit, n_intervals + 1)
        self.limit = float(limit)
        x, w = leggauss(self.points)  # type: ignore[no-untyped-call]
        width = np.diff(edges)
        nodes = edges[:-1, None] + width[:, None] * (0.5 * (x + 1.0))[None, :]
        weights = 0.5 * width[:, None] * w
        mu = link.mu(nodes.ravel())
        flat_w = weights.ravel()
        seg_psi = (mu * flat_w).reshape(n_intervals, self.points).sum(axis=1)
        seg_g = (np.abs(mu) ** 2 * flat_w).reshape(n_intervals, self.points).sum(axis=1)
        self._edges = edges
        self._psi = np.concatenate(([0.0 + 0.0j], np.cumsum(seg_psi)))
        self._g = np.concatenate(([0.0], np.cumsum(seg_g)))
        self._mu_edge = link.mu(edges)
        self._power_edge = cast("NDArrayFloat", np.abs(self._mu_edge) ** 2)

    @overload
    def _interpolate(
        self, p: NDArrayFloat, values: NDArrayFloat, slopes: NDArrayFloat
    ) -> NDArrayFloat: ...

    @overload
    def _interpolate(
        self, p: NDArrayFloat, values: NDArrayComplex, slopes: NDArrayComplex
    ) -> NDArrayComplex: ...

    def _interpolate(self, p: NDArrayFloat, values: Any, slopes: Any) -> Any:
        """Interpolate table values using cubic Hermite interpolation."""
        x = np.clip(np.asarray(p, dtype=np.float64), self._edges[0], self._edges[-1])
        idx = np.searchsorted(self._edges, x, side="right") - 1
        idx = np.clip(idx, 0, self._edges.size - 2)
        x0 = self._edges[idx]
        x1 = self._edges[idx + 1]
        h = x1 - x0
        s = (x - x0) / h
        s2 = s * s
        s3 = s2 * s
        return (
            (2.0 * s3 - 3.0 * s2 + 1.0) * values[idx]
            + (s3 - 2.0 * s2 + s) * h * slopes[idx]
            + (-2.0 * s3 + 3.0 * s2) * values[idx + 1]
            + (s3 - s2) * h * slopes[idx + 1]
        )

    def value(self, p: NDArrayFloat) -> NDArrayComplex:
        """Return the complex antiderivative of mu from zero to p in Hz²."""
        return self._interpolate(p, self._psi, self._mu_edge)

    def power(self, p: NDArrayFloat) -> NDArrayFloat:
        """Return the real antiderivative of |mu|² from zero to p in Hz²."""
        return self._interpolate(p, self._g, self._power_edge)


# =============================================================================
# =============================================================================
@dataclass(frozen=True)
class SciKappas(Generic[_KappaValue]):
    """SCI GN, fourth-order, and sixth-order coefficients from Eq. (7)-(9).

    The fourth-order coefficient contains distinct f1-outer and f3-outer terms."""

    kappa1: _KappaValue
    kappa2: _KappaValue
    kappa3: _KappaValue


@dataclass(frozen=True)
class XciKappas(Generic[_KappaValue]):
    """XCI coefficients for regions X1-X4 from Eq. (16)-(18) and Appendix A.

    In ``kappa_mn``, m identifies the region and n identifies the GN,
    fourth-order, or sixth-order contribution."""

    kappa11: _KappaValue
    kappa12: _KappaValue
    kappa21: _KappaValue
    kappa22: _KappaValue
    kappa31: _KappaValue
    kappa32: _KappaValue
    kappa41: _KappaValue
    kappa42: _KappaValue
    kappa43: _KappaValue


@dataclass(frozen=True)
class MciKappas(Generic[_KappaValue]):
    """MCI coefficients for the M0 and M1 regions in Eq. (24)-(26)."""

    kappa_m0: _KappaValue
    kappa_m1_1: _KappaValue
    kappa_m1_2: _KappaValue


# =============================================================================
# =============================================================================
class EgnIntegrator:
    """Evaluate the paper's reduced SCI, XCI, and MCI integrals.

    With u=f1-f and v=f2-f, p=u*v and the third wave has offset u+v.
    The intersection of three rectangular bands determines the integration
    region. Antiderivatives integrate the inner variable, and composite
    Gauss-Legendre quadrature resolves the remaining oscillatory outer integral.

    Parameters
    ----------
    link : LinkFunction
        Link geometry and fiber properties.
    symbol_rate_hz : float
        Symbol rate and rectangular channel bandwidth in Hz.
    points : int
        Gauss-Legendre nodes per composite interval.
    p_limit_hz2 : float or None, optional
        Initial absolute frequency-product limit in Hz²."""

    def __init__(
        self,
        *,
        link: LinkFunction,
        symbol_rate_hz: float,
        points: int = DEFAULT_QUADRATURE_POINTS,
        p_limit_hz2: float | None = None,
    ) -> None:
        """Initialize the quadrature and antiderivative table."""
        if points < 2:
            msg = f"points must be >= 2, got {points}."
            raise ValueError(msg)
        self.link = link
        self.symbol_rate_hz = float(symbol_rate_hz)
        self.points = int(points)
        self._half = 0.5 * self.symbol_rate_hz
        if p_limit_hz2 is None:
            p_limit_hz2 = (2.0 * self._half) ** 2
        self._table = _MuAntiderivative(link, self.points, float(p_limit_hz2))
        self._spectral = self.symbol_rate_hz**-6

    def _ensure_table(self, p_limit: float) -> None:
        """Expand the antiderivative table if the requested product exceeds its range."""
        if p_limit > self._table.limit:
            self._table = _MuAntiderivative(self.link, self.points, float(p_limit))

    def _cut_band(self) -> tuple[float, float]:
        """Return the lower and upper edges of the channel under test."""
        return (-self._half, self._half)

    def _band(self, center_hz: float) -> tuple[float, float]:
        """Return the band edges of the channel centered at ``center_hz``."""
        return (center_hz - self._half, center_hz + self._half)

    def _splits(
        self,
        u_lo: float,
        u_hi: float,
        v_lo: float,
        v_hi: float,
        w_lo: float,
        w_hi: float,
    ) -> list[float]:
        """Return sorted outer-integral breakpoints, including band intersections and zero."""
        candidates = (u_lo, u_hi, 0.0, w_lo - v_lo, w_hi - v_hi)
        interior = sorted({c for c in candidates if u_lo < c < u_hi})
        return [u_lo, *interior, u_hi]

    def _outer_grid(
        self,
        lo: float,
        hi: float,
        v_scale: float,
        divisor: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return composite Gauss-Legendre nodes and weights for the outer integral.

        The subdivision count follows the phase period and the scale of the inner
        frequency. ``divisor`` coarsens this count for the L2 integral."""
        if np.isfinite(self._table.period):
            n_osc = (hi - lo) * v_scale / self._table.period
        else:
            n_osc = 0.0
        n_sub = max(1, int(np.ceil(n_osc / divisor)))
        x, w = leggauss(self.points)  # type: ignore[no-untyped-call]
        edges = np.linspace(lo, hi, n_sub + 1)
        width = np.diff(edges)
        nodes = edges[:-1, None] + width[:, None] * (0.5 * (x + 1.0))[None, :]
        weights = 0.5 * width[:, None] * w
        return nodes.ravel(), weights.ravel()

    @overload
    def _core(
        self,
        f_hz: float,
        band1: tuple[float, float],
        band2: tuple[float, float],
        band3: tuple[float, float],
        kind: Literal["gn", "fourth", "l2"],
    ) -> tuple[float, float]: ...

    @overload
    def _core(
        self,
        f_hz: float,
        band1: tuple[float, float],
        band2: tuple[float, float],
        band3: tuple[float, float],
        kind: Literal["sixth"],
    ) -> tuple[float, complex]: ...

    def _core(
        self,
        f_hz: float,
        band1: tuple[float, float],
        band2: tuple[float, float],
        band3: tuple[float, float],
        kind: str,
    ) -> tuple[float, float | complex]:
        """Evaluate a reduced two-dimensional integral over three rectangular bands.

        The ``kind`` selects the GN, fourth-order, sixth-order, or L2 contribution.
        L2 uses the third wave as the outer variable."""
        if kind == "l2":
            return 0.0, self._core_l2(f_hz, band1, band2, band3)
        u_lo, u_hi = band1[0] - f_hz, band1[1] - f_hz
        v_lo, v_hi = band2[0] - f_hz, band2[1] - f_hz
        w_lo, w_hi = band3[0] - f_hz, band3[1] - f_hz
        self._ensure_table(max(abs(u_lo), abs(u_hi)) * max(abs(v_lo), abs(v_hi)))
        v_scale = max(
            abs(v_lo), abs(v_hi), abs(w_lo), abs(w_hi), 2.0 * max(abs(u_lo), abs(u_hi))
        )
        gn_total = 0.0
        coh_total = 0.0 + 0.0j
        splits = self._splits(u_lo, u_hi, v_lo, v_hi, w_lo, w_hi)
        for lo, hi in itertools.pairwise(splits):
            u_nodes, u_w = self._outer_grid(lo, hi, v_scale)
            v1 = np.maximum(v_lo, w_lo - u_nodes)
            v2 = np.minimum(v_hi, w_hi - u_nodes)
            valid = v2 > v1
            if not bool(np.any(valid)):
                continue
            p1 = u_nodes * v1
            p2 = u_nodes * v2
            if kind == "gn":
                val = (self._table.power(p2) - self._table.power(p1)) / u_nodes
                gn_total += float(np.sum(u_w * np.where(valid, val, 0.0)))
            else:
                complex_val = (self._table.value(p2) - self._table.value(p1)) / u_nodes
                complex_val = np.where(valid, complex_val, 0.0)
                if kind == "fourth":
                    coh_total += complex(np.sum(u_w * np.abs(complex_val) ** 2))
                else:
                    coh_total += complex(np.sum(u_w * complex_val))
        if kind == "gn":
            return gn_total, 0.0
        if kind == "fourth":
            return 0.0, float(coh_total.real)
        return 0.0, coh_total

    def _l2_half_integral(
        self,
        w_nodes: NDArrayFloat,
        a2: NDArrayFloat,
        b2: NDArrayFloat,
    ) -> NDArrayComplex:
        """Evaluate one even half of the L2 inner integral by product integration.

        After s=w-2v and u=s², the mu contribution uses its p-axis antiderivative;
        the inverse-square-root weight is averaged analytically in each cell. Cells
        are concentrated near zero where the weight varies most rapidly."""
        frac = (np.arange(L2_CELLS + 1, dtype=np.float64) / float(L2_CELLS)) ** 2
        u = a2[:, None] + (b2 - a2)[:, None] * frac[None, :]
        p = (w_nodes[:, None] ** 2 - u) / 4.0
        du = u[:, 1:] - u[:, :-1]
        weight = 2.0 * (np.sqrt(u[:, 1:]) - np.sqrt(u[:, :-1]))
        average = np.where(du > 0.0, weight / np.where(du > 0.0, du, 1.0), 0.0)
        psi = self._table.value(p)
        segment = 4.0 * (psi[:, :-1] - psi[:, 1:])
        return cast("NDArrayComplex", np.sum(average * segment, axis=1))

    def _core_l2(
        self,
        f_hz: float,
        band1: tuple[float, float],
        band2: tuple[float, float],
        band3: tuple[float, float],
    ) -> float:
        """Evaluate the distinct Eq. (44) fourth-order integral with f3 outermost.

        Its integration region differs from the f1-outer fourth-order term and the
        two must not be combined into a single integral."""
        u_lo, u_hi = band1[0] - f_hz, band1[1] - f_hz
        v_lo, v_hi = band2[0] - f_hz, band2[1] - f_hz
        w_lo, w_hi = band3[0] - f_hz, band3[1] - f_hz
        self._ensure_table(
            max(
                max(abs(u_lo), abs(u_hi)) * max(abs(v_lo), abs(v_hi)),
                0.25 * max(abs(w_lo), abs(w_hi)) ** 2,
            )
        )
        v_scale = max(
            abs(v_lo), abs(v_hi), abs(w_lo), abs(w_hi), 2.0 * max(abs(u_lo), abs(u_hi))
        )
        coh_total = 0.0
        kinks = sorted({v_lo + u_hi, v_hi + u_lo})
        edges = [w_lo, *[k for k in kinks if w_lo < k < w_hi], w_hi]
        for lo, hi in itertools.pairwise(edges):
            w_nodes, w_w = self._outer_grid(lo, hi, v_scale, divisor=L2_OUTER_DIVISOR)
            v1 = np.maximum(v_lo, w_nodes - u_hi)
            v2 = np.minimum(v_hi, w_nodes - u_lo)
            valid = v2 > v1
            if not bool(np.any(valid)):
                continue
            s1 = w_nodes - 2.0 * v1
            s2 = w_nodes - 2.0 * v2
            inner = np.zeros(w_nodes.shape, dtype=np.complex128)
            positive = valid & (s1 > 0.0)
            if bool(np.any(positive)):
                a2 = np.where(s2 > 0.0, s2 * s2, 0.0)
                b2 = cast("NDArrayFloat", s1 * s1)
                part = 0.25 * self._l2_half_integral(w_nodes, a2, b2)
                inner = inner + np.where(positive, part, 0.0)
            negative = valid & (s2 < 0.0)
            if bool(np.any(negative)):
                a2 = np.where(s1 < 0.0, s1 * s1, 0.0)
                b2 = cast("NDArrayFloat", s2 * s2)
                part = 0.25 * self._l2_half_integral(w_nodes, a2, b2)
                inner = inner + np.where(negative, part, 0.0)
            coh_total += float(np.sum(w_w * np.abs(inner) ** 2))
        return coh_total

    # --- SCI (2014 Eq. (7)/(8)/(9)) -----------------------------------------
    def sci(self, f_hz: float) -> SciKappas[float]:
        """Calculate SCI kappa coefficients at an observation frequency in Hz.

        The fourth-order result combines the distinct 80/81 f1-outer and 16/81
        f3-outer terms; the sixth-order term uses the Appendix C reduction.

        Returns
        -------
        SciKappas
            GN, fourth-order, and sixth-order SCI coefficients."""
        band = self._cut_band()
        gn, _ = self._core(f_hz, band, band, band, "gn")
        _, fourth = self._core(f_hz, band, band, band, "fourth")
        _, fourth_third = self._core(f_hz, band, band, band, "l2")
        _, sixth = self._core(f_hz, band, band, band, "sixth")
        rs = self.symbol_rate_hz
        spec = self._spectral
        kappa1 = (16.0 / 27.0) * rs**3 * spec * gn
        kappa2 = ((80.0 / 81.0) * fourth + (16.0 / 81.0) * fourth_third) * rs**2 * spec
        kappa3 = (16.0 / 81.0) * rs * spec * abs(sixth) ** 2
        return SciKappas(kappa1=kappa1, kappa2=kappa2, kappa3=kappa3)

    def xci(self, f_c_hz: float, f_hz: float) -> XciKappas[float]:
        """Calculate the nine XCI coefficients for one interfering channel.

        The four domains assign the three waves to CUT/INT as follows: X1 CUT/INT/INT,
        X2 INT/CUT/CUT, X3 CUT/CUT/INT, and X4 INT/INT/INT. Rectangular spectral
        support makes X2-X4 vanish at channel offsets of at least twice the symbol
        rate.

        Parameters
        ----------
        f_c_hz : float
            Interferer offset relative to the channel under test in Hz.
        f_hz : float
            Observation frequency in Hz.

        Returns
        -------
        XciKappas
            GN and modulation-dependent XCI coefficients."""
        cut = self._cut_band()
        int_band = self._band(f_c_hz)
        rs = self.symbol_rate_hz
        spec = self._spectral

        gn11, _ = self._core(f_hz, cut, int_band, int_band, "gn")
        _, c12 = self._core(f_hz, cut, int_band, int_band, "fourth")
        gn21, _ = self._core(f_hz, int_band, cut, cut, "gn")
        _, c22 = self._core(f_hz, int_band, cut, cut, "fourth")
        gn31, _ = self._core(f_hz, cut, cut, int_band, "gn")
        _, c32 = self._core(f_hz, cut, cut, int_band, "fourth")
        gn41, _ = self._core(f_hz, int_band, int_band, int_band, "gn")
        _, c42 = self._core(f_hz, int_band, int_band, int_band, "fourth")
        _, c42_third = self._core(f_hz, int_band, int_band, int_band, "l2")
        _, c43 = self._core(f_hz, int_band, int_band, int_band, "sixth")

        return XciKappas(
            kappa11=(32.0 / 27.0) * rs**3 * spec * gn11,
            kappa12=(80.0 / 81.0) * rs**2 * spec * c12,
            kappa21=(32.0 / 27.0) * rs**3 * spec * gn21,
            kappa22=(80.0 / 81.0) * rs**2 * spec * c22,
            kappa31=(16.0 / 27.0) * rs**3 * spec * gn31,
            kappa32=(16.0 / 81.0) * rs**2 * spec * c32,
            kappa41=(16.0 / 27.0) * rs**3 * spec * gn41,
            kappa42=((80.0 / 81.0) * c42 + (16.0 / 81.0) * c42_third) * rs**2 * spec,
            kappa43=(16.0 / 81.0) * rs * spec * abs(c43) ** 2,
        )

    # --- MCI (2014 Eq. (24)/(25)/(26)) --------------------------------------
    def mci(
        self, f_upper_hz: float, f_lower_hz: float, f_hz: float
    ) -> MciKappas[float]:
        """Calculate the adjacent three-channel M0 and M1 coefficients.

        Parameters
        ----------
        f_upper_hz, f_lower_hz : float
            Upper and lower interferer offsets relative to CUT in Hz.
        f_hz : float
            Observation frequency in Hz.

        Returns
        -------
        MciKappas
            M0 and M1 GN coefficients and M1 fourth-order correction."""
        cut = self._cut_band()
        up = self._band(f_upper_hz)
        low = self._band(f_lower_hz)
        rs = self.symbol_rate_hz
        spec = self._spectral

        gn_m0, _ = self._core(f_hz, up, low, cut, "gn")
        gn_m1, _ = self._core(f_hz, low, up, up, "gn")
        _, c_m1 = self._core(f_hz, low, up, up, "fourth")

        return MciKappas(
            kappa_m0=2.0 * (16.0 / 27.0) * rs**3 * spec * gn_m0,
            kappa_m1_1=4.0 * (16.0 / 27.0) * rs**3 * spec * gn_m1,
            kappa_m1_2=2.0 * (80.0 / 81.0) * rs**2 * spec * c_m1,
        )


# =============================================================================
# =============================================================================
@dataclass(frozen=True)
class _EgnRatios:
    """SCI and XCI EGN/GN ratios and MCI GN and correction powers."""

    sci: float
    xci: NDArrayFloat
    mci_gn: NDArrayFloat
    mci_corr: NDArrayFloat


@dataclass(frozen=True)
class _XpmBlocks:
    """Per-channel XCI and MCI GN and EGN correction blocks."""

    xci_gn: NDArrayFloat
    xci_corr: NDArrayFloat
    mci_gn: NDArrayFloat
    mci_corr: NDArrayFloat


class Carena2014SpanKernel:
    """Single-span EGN kernel based on the closed-form GN evaluator.

    The integrals and fiber-dependent coefficients are cached for each span
    count and length. SCI ratios are power-independent; XCI and MCI combine
    cached coefficients with channel launch powers.

    Parameters
    ----------
    gn_evaluator : MultiBandISRSGN
        Closed-form GN baseline.
    phi, psi : float
        Fourth- and sixth-order modulation moments from Table 1.
    quadrature_points : int
        Gauss-Legendre nodes per composite interval.
    frequency_points : int
        Number of observation frequencies across the CUT bandwidth."""

    def __init__(
        self,
        *,
        gn_evaluator: MultiBandISRSGN,
        phi: float,
        psi: float,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        frequency_points: int = DEFAULT_FREQUENCY_POINTS,
    ) -> None:
        """Initialize the GN baseline and EGN quadrature settings."""
        self.gn = gn_evaluator
        self.phi = float(phi)
        self.psi = float(psi)
        self.quadrature_points = int(quadrature_points)
        self.frequency_points = int(frequency_points)

        self.f_rel_hz = np.asarray(gn_evaluator.f_rel_hz, dtype=np.float64)
        self.b_hz = np.asarray(
            gn_evaluator.channel_bandwidth_hz_channels, dtype=np.float64
        )
        self.beta2 = float(gn_evaluator.beta2_s2_m)
        self.beta3 = float(gn_evaluator.beta3_s3_m)
        self.alpha_npm = np.asarray(
            gn_evaluator.alpha_power_npm_channels, dtype=np.float64
        )
        self.a_eff_m2 = np.asarray(gn_evaluator.a_eff_m2_channels, dtype=np.float64)
        self.lambdas_m = np.asarray(gn_evaluator.lambdas_m, dtype=np.float64)
        self.n2_m2_w = float(gn_evaluator.n2_m2_w)
        self.symbol_rate_hz = float(gn_evaluator.symbol_rate_hz)

        n_ch = int(self.b_hz.size)
        # gamma_ii = (2 pi / lambda_i) * (2 n2 / (A_eff,i + A_eff,i))
        n2 = self.fiber_n2()
        self.gamma_ii = (2.0 * PI / self.lambdas_m) * (2.0 * n2 / (2.0 * self.a_eff_m2))

        self._center_index = int(np.argmin(np.abs(self.f_rel_hz)))
        self.gamma_center = float(self.gamma_ii[self._center_index])
        self.alpha_center = float(self.alpha_npm[self._center_index])

        lower = np.full(n_ch, -1, dtype=np.int64)
        upper = np.full(n_ch, -1, dtype=np.int64)
        for i in range(n_ch):
            if i > 0:
                lower[i] = i - 1
            if i + 1 < n_ch:
                upper[i] = i + 1
        self.mci_lower = lower
        self.mci_upper = upper

        self._neighbor_span = XCI_NEIGHBOR_SPAN
        self._separation_offsets: NDArrayFloat = np.zeros(0)
        self._build_separation_table()

        max_offset = (
            float(np.max(np.abs(self._separation_offsets)))
            if self._separation_offsets.size
            else 0.0
        )
        if n_ch > 1:
            max_offset = max(max_offset, float(np.max(np.abs(np.diff(self.f_rel_hz)))))
        self._p_limit_hz2 = (max_offset + self.symbol_rate_hz) ** 2

        self._kappa_cache: dict[
            tuple[int, float], dict[int, XciKappas[NDArrayFloat]]
        ] = {}
        self._sci_cache: dict[tuple[int, float], dict[float, SciKappas[float]]] = {}
        self._mci_cache: dict[
            tuple[int, float], dict[tuple[float, float], MciKappas[NDArrayFloat]]
        ] = {}
        self._mci_channel_cache: dict[
            tuple[int, float], tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
        ] = {}
        self._xci_distance_cache: dict[
            tuple[int, float],
            dict[int, tuple[float, float, float, float, float, float]],
        ] = {}
        self._buildup_cache: dict[tuple[int, float], float] = {}
        self._step_cache: dict[tuple[int, float], float] = {}

    def fiber_n2(self) -> float:
        """Return the fiber nonlinear refractive index in m²/W."""
        return self.n2_m2_w

    def _build_separation_table(self) -> None:
        """Map channel index separation to median frequency offset.

        The paper assumes uniformly spaced channels; the median approximates
        nonuniform grids."""
        n_ch = int(self.b_hz.size)
        max_d = min(self._neighbor_span, max(0, n_ch - 1))
        offsets = np.zeros(max_d + 1, dtype=np.float64)
        for d in range(1, max_d + 1):
            if d < n_ch:
                offsets[d] = float(np.median(self.f_rel_hz[d:] - self.f_rel_hz[:-d]))
        self._separation_offsets = offsets

    def _frequency_grid(self) -> NDArrayFloat:
        """Return observation frequencies across the channel under test in Hz."""
        return np.linspace(
            -0.5 * self.symbol_rate_hz, 0.5 * self.symbol_rate_hz, self.frequency_points
        )

    def _integrator(self, n_spans: int, span_length_m: float) -> EgnIntegrator:
        """Build an EGN integrator for the span count and length."""
        link = LinkFunction(
            gamma=self.gamma_center,
            alpha_power_npm=self.alpha_center,
            beta2_s2_m=self.beta2,
            span_length_m=float(span_length_m),
            n_spans=int(n_spans),
        )
        return EgnIntegrator(
            link=link,
            symbol_rate_hz=self.symbol_rate_hz,
            points=self.quadrature_points,
            p_limit_hz2=self._p_limit_hz2,
        )

    def sci_kappas(
        self, n_spans: int, span_length_m: float
    ) -> dict[float, SciKappas[float]]:
        """Return SCI coefficients keyed by observation frequency for the link."""
        key = (int(n_spans), float(span_length_m))
        cached = self._sci_cache.get(key)
        if cached is not None:
            return cached
        integrator = self._integrator(n_spans, span_length_m)
        table = {float(f): integrator.sci(float(f)) for f in self._frequency_grid()}
        self._sci_cache[key] = table
        return table

    def xci_kappas(
        self, n_spans: int, span_length_m: float
    ) -> dict[int, XciKappas[NDArrayFloat]]:
        """Return XCI coefficients keyed by channel index separation for the link."""
        key = (int(n_spans), float(span_length_m))
        cached = self._kappa_cache.get(key)
        if cached is not None:
            return cached
        integrator = self._integrator(n_spans, span_length_m)
        table: dict[int, XciKappas[NDArrayFloat]] = {}
        for d in range(1, self._separation_offsets.size):
            offset = float(self._separation_offsets[d])
            samples = [integrator.xci(offset, float(f)) for f in self._frequency_grid()]
            table[d] = XciKappas(
                **{
                    name: np.array([getattr(s, name) for s in samples])
                    for name in samples[0].__dataclass_fields__
                }
            )
        self._kappa_cache[key] = table
        return table

    def mci_kappas(
        self, n_spans: int, span_length_m: float
    ) -> dict[tuple[float, float], MciKappas[NDArrayFloat]]:
        """Return MCI coefficients keyed by upper and lower frequency offsets."""
        key = (int(n_spans), float(span_length_m))
        cached = self._mci_cache.get(key)
        if cached is not None:
            return cached
        integrator = self._integrator(n_spans, span_length_m)
        table: dict[tuple[float, float], MciKappas[NDArrayFloat]] = {}
        n_ch = int(self.b_hz.size)
        for i in range(n_ch):
            lo = int(self.mci_lower[i])
            hi = int(self.mci_upper[i])
            if lo < 0 or hi < 0:
                continue
            f_upper = float(self.f_rel_hz[hi] - self.f_rel_hz[i])
            f_lower = float(self.f_rel_hz[lo] - self.f_rel_hz[i])
            pair = (round(f_upper, 6), round(f_lower, 6))
            if pair in table:
                continue
            table[pair] = MciKappas(
                **{
                    name: np.array(
                        [
                            getattr(integrator.mci(f_upper, f_lower, float(f)), name)
                            for f in self._frequency_grid()
                        ]
                    )
                    for name in ("kappa_m0", "kappa_m1_1", "kappa_m1_2")
                }
            )
        self._mci_cache[key] = table
        return table

    def _sci_block(self, n_spans: int, span_length_m: float) -> tuple[float, float]:
        """Return the sums of GN and corrected EGN SCI coefficients over frequency."""
        table = self.sci_kappas(n_spans, span_length_m)
        gn = 0.0
        total = 0.0
        for kappas in table.values():
            gn += float(np.sum(kappas.kappa1))
            total += float(
                np.sum(
                    kappas.kappa1 + self.phi * kappas.kappa2 + self.psi * kappas.kappa3
                )
            )
        return gn, total

    def _sci_ratio(self, n_spans: int, span_length_m: float) -> float:
        """Return the SCI EGN/GN ratio weighted by GN contribution across the band."""
        gn, total = self._sci_block(n_spans, span_length_m)
        if gn <= 0.0:
            return 1.0
        return total / gn

    def _xci_distance_sums(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> dict[int, tuple[float, float, float, float, float, float]]:
        """Return cached XCI GN and correction sums for each neighbor separation.

        The X1, X2/X3, and X4 blocks have different channel-power prefactors."""
        key = (int(n_spans), float(span_length_m))
        cached = self._xci_distance_cache.get(key)
        if cached is not None:
            return cached
        table = self.xci_kappas(n_spans, span_length_m)
        sums: dict[int, tuple[float, float, float, float, float, float]] = {}
        for distance, kappas in table.items():
            g11 = float(np.sum(kappas.kappa11))
            g21 = float(np.sum(kappas.kappa21 + kappas.kappa31))
            g41 = float(np.sum(kappas.kappa41))
            c11 = self.phi * float(np.sum(kappas.kappa12))
            c21 = self.phi * float(np.sum(kappas.kappa22 + kappas.kappa32))
            c41 = self.phi * float(np.sum(kappas.kappa42)) + self.psi * float(
                np.sum(kappas.kappa43)
            )
            sums[distance] = (g11, g21, g41, c11, c21, c41)
        self._xci_distance_cache[key] = sums
        return sums

    def _xci_terms(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute per-channel XCI GN and correction terms from cached coefficients."""
        n_ch = int(self.b_hz.size)
        gn = np.zeros(n_ch, dtype=np.float64)
        corr = np.zeros(n_ch, dtype=np.float64)
        if n_ch < 2:
            return gn, corr
        for distance, (g11, g21, g41, c11, c21, c41) in self._xci_distance_sums(
            n_spans, span_length_m
        ).items():
            hi = slice(distance, n_ch)
            lo = slice(0, n_ch - distance)
            p_hi = power[hi]
            p_lo = power[lo]
            gn[hi] += power[lo] * p_hi**2 * g11
            gn[hi] += power[lo] ** 2 * p_hi * g21
            gn[hi] += p_hi**3 * g41
            corr[hi] += power[lo] * p_hi**2 * c11
            corr[hi] += power[lo] ** 2 * p_hi * c21
            corr[hi] += p_hi**3 * c41
            gn[lo] += p_hi * p_lo**2 * g11
            gn[lo] += p_hi**2 * p_lo * g21
            gn[lo] += p_lo**3 * g41
            corr[lo] += p_hi * p_lo**2 * c11
            corr[lo] += p_hi**2 * p_lo * c21
            corr[lo] += p_lo**3 * c41
        return gn, corr

    def _xci_ratio(
        self, n_spans: int, span_length_m: float, power: NDArrayFloat
    ) -> NDArrayFloat:
        """Return per-channel XCI EGN/GN ratios over the neighbor window."""
        gn, corr = self._xci_terms(n_spans, span_length_m, power)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(gn > 0.0, corr / np.where(gn > 0.0, gn, 1.0), 0.0)

    def _mci_channel_kappas(
        self,
        n_spans: int,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Return cached M0 and M1 coefficients for the nearest adjacent channels."""
        key = (int(n_spans), float(span_length_m))
        cached = self._mci_channel_cache.get(key)
        if cached is not None:
            return cached
        n_ch = int(self.b_hz.size)
        km0 = np.zeros(n_ch, dtype=np.float64)
        km11 = np.zeros(n_ch, dtype=np.float64)
        km12 = np.zeros(n_ch, dtype=np.float64)
        table = self.mci_kappas(n_spans, span_length_m)
        for index in range(n_ch):
            lo = int(self.mci_lower[index])
            hi = int(self.mci_upper[index])
            if lo < 0 or hi < 0:
                continue
            pair = (
                round(float(self.f_rel_hz[hi] - self.f_rel_hz[index]), 6),
                round(float(self.f_rel_hz[lo] - self.f_rel_hz[index]), 6),
            )
            kappas = table[pair]
            grid = self._frequency_grid()
            km0[index] = float(np.trapz(kappas.kappa_m0, grid))  # noqa: NPY201
            km11[index] = float(np.trapz(kappas.kappa_m1_1, grid))  # noqa: NPY201
            km12[index] = float(np.trapz(kappas.kappa_m1_2, grid))  # noqa: NPY201
        cached = (km0, km11, km12)
        self._mci_channel_cache[key] = cached
        return cached

    def _mci_terms(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute adjacent three-channel MCI GN and correction powers in watts."""
        n_ch = int(self.b_hz.size)
        gn = np.zeros(n_ch, dtype=np.float64)
        corr = np.zeros(n_ch, dtype=np.float64)
        if n_ch < 2:
            return gn, corr
        km0, km11, km12 = self._mci_channel_kappas(n_spans, span_length_m)
        lower = np.where(self.mci_lower >= 0, self.mci_lower, 0)
        upper = np.where(self.mci_upper >= 0, self.mci_upper, 0)
        valid = (self.mci_lower >= 0) & (self.mci_upper >= 0)
        p_cut = power
        p_up = power[upper]
        p_low = power[lower]
        m1 = p_up**2 * p_low
        gn = np.where(valid, p_cut * p_up * p_low * km0 + m1 * km11, 0.0)
        corr = np.where(valid, m1 * self.phi * km12, 0.0)
        return gn, corr

    def _xpm_blocks(
        self, n_spans: int, span_length_m: float, power: NDArrayFloat
    ) -> _XpmBlocks:
        """Return per-channel XCI and MCI GN and correction blocks."""
        xci_gn, xci_corr = self._xci_terms(n_spans, span_length_m, power)
        mci_gn, mci_corr = self._mci_terms(n_spans, span_length_m, power)
        return _XpmBlocks(
            xci_gn=xci_gn, xci_corr=xci_corr, mci_gn=mci_gn, mci_corr=mci_corr
        )

    def _ratios(
        self, n_spans: int, span_length_m: float, power: NDArrayFloat
    ) -> _EgnRatios:
        """Calculate EGN correction ratios and MCI powers for the supplied launch powers."""
        sci = self._sci_ratio(n_spans, span_length_m)
        xci = self._xci_ratio(n_spans, span_length_m, power)
        mci_gn, mci_corr = self._mci_terms(n_spans, span_length_m, power)
        return _EgnRatios(sci=sci, xci=xci, mci_gn=mci_gn, mci_corr=mci_corr)

    def _sci_buildup(self, n_spans: int, span_length_m: float) -> float:
        """Return the SCI coherent factor EGN(N)/EGN(1).

        The factor tends to N² at zero dispersion and toward N for strongly
        dispersive incoherent accumulation."""
        key = (int(n_spans), float(span_length_m))
        cached = self._buildup_cache.get(key)
        if cached is not None:
            return cached
        _, egn_now = self._sci_block(n_spans, span_length_m)
        _, egn_ref = self._sci_block(1, span_length_m)
        buildup = egn_now / egn_ref if egn_ref > 0.0 else 1.0
        self._buildup_cache[key] = buildup
        return buildup

    def _sci_step(self, n_spans: int, span_length_m: float) -> float:
        """Invert the SCI buildup factor to an equivalent phase step in radians."""
        key = (int(n_spans), float(span_length_m))
        cached = self._step_cache.get(key)
        if cached is not None:
            return cached
        step = float(
            solve_dirichlet_step(
                n_spans, np.asarray(self._sci_buildup(n_spans, span_length_m))
            )
        )
        self._step_cache[key] = step
        return step

    def _coherent_levels(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
        s2_spm: NDArrayFloat,
        s2_xpm: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Return single-span SCI and XPM powers and XPM link buildup factors."""
        spm_power = self._sci_span_level(span_length_m, s2_spm)
        xpm_power, link_xpm = self._xpm_levels(n_spans, span_length_m, power, s2_xpm)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe = np.where(xpm_power > 0.0, xpm_power, 1.0)
            buildup_xpm = np.where(xpm_power > 0.0, link_xpm / safe, 0.0)
        return spm_power, xpm_power, buildup_xpm

    def _xpm_levels(
        self,
        n_spans: int,
        span_length_m: float,
        power: NDArrayFloat,
        s2_xpm: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return single-span and link-level XPM powers for coherent buildup."""
        blocks = self._xpm_blocks(n_spans, span_length_m, power)
        reference = self._xpm_blocks(1, span_length_m, power)
        xci_ref = reference.xci_gn
        safe = np.where(xci_ref > 0.0, xci_ref, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            block_now = np.where(
                xci_ref > 0.0,
                (blocks.xci_gn + blocks.xci_corr) / safe,
                0.0,
            )
            block_ref = np.where(
                xci_ref > 0.0,
                (reference.xci_gn + reference.xci_corr) / safe,
                0.0,
            )
        active = power > 0.0
        per_span = np.where(
            active, s2_xpm * block_ref + reference.mci_gn + reference.mci_corr, 0.0
        )
        link = np.where(
            active, s2_xpm * block_now + blocks.mci_gn + blocks.mci_corr, 0.0
        )
        return per_span, link

    def link_level_nli(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_spans: int,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Calculate link-level SCI and XPM powers in watts from the EGN integrals.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Channel launch powers in watts.
        span_length_m : float
            Span length in meters.
        n_spans : int
            Number of spans.

        Returns
        -------
        tuple of numpy.ndarray
            Link-level SCI and XPM noise powers in watts."""
        power = np.asarray(power_in_w, dtype=np.float64)
        n = max(1, int(n_spans))
        gn_out = self.gn.span_fiber_output(power, float(span_length_m), 1)
        s2_spm = np.asarray(gn_out.spm_noise_w, dtype=np.float64)
        s2_xpm = np.asarray(gn_out.xpm_noise_w, dtype=np.float64)
        sci = self._sci_link_level(n, span_length_m, s2_spm)
        _, xpm = self._xpm_levels(n, span_length_m, power, s2_xpm)
        return sci, xpm

    def _sci_link_level(
        self, n_spans: int, span_length_m: float, s2_spm: NDArrayFloat
    ) -> NDArrayFloat:
        """Return link-level SCI from the single-span GN baseline and N-span EGN block."""
        gn_ref, _ = self._sci_block(1, span_length_m)
        _, egn_now = self._sci_block(n_spans, span_length_m)
        return s2_spm * (egn_now / gn_ref if gn_ref > 0.0 else 1.0)

    def _sci_span_level(
        self, span_length_m: float, s2_spm: NDArrayFloat
    ) -> NDArrayFloat:
        """Return single-span SCI from the GN baseline and single-span EGN block."""
        gn_ref, egn_ref = self._sci_block(1, span_length_m)
        return s2_spm * (egn_ref / gn_ref if gn_ref > 0.0 else 1.0)

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        span_context: SpanContext | None = None,
    ) -> SpanFiberOutput:
        """Compute fiber output and single-span NLI with coherent amplitudes.

        The equivalent phase steps reproduce EGN link totals under equal-span,
        equal-power assumptions, but compress their frequency dependence.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Channel input powers in watts; zero indicates an inactive channel.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Link span count when no span context is provided.
        span_context : SpanContext or None, optional
            Span position and cumulative link length.

        Returns
        -------
        SpanFiberOutput
            Output powers, NLI powers, and coherent amplitudes."""
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != self.b_hz.shape:
            msg = f"power_in_w must have one entry per channel: expected {self.b_hz.shape}, got {power.shape}."
            raise ValueError(msg)
        if span_length_m <= 0.0:
            msg = f"span_length_m must be positive, got {span_length_m}."
            raise ValueError(msg)
        context = (
            span_context
            if span_context is not None
            else SpanContext(
                span_index=0,
                total_spans=max(1, int(n_effective_spans)),
                cumulative_length_m=0.0,
                span_length_m=float(span_length_m),
            )
        )
        n_spans = max(1, int(context.total_spans))
        length_m = float(span_length_m)

        gn_out = self.gn.span_fiber_output(power, length_m, 1)
        power_out = np.array(gn_out.power_out_w, dtype=np.float64, copy=True)
        s2_spm = np.array(gn_out.spm_noise_w, dtype=np.float64, copy=True)
        s2_xpm = np.array(gn_out.xpm_noise_w, dtype=np.float64, copy=True)
        active = power > 0.0

        if n_spans == 1:
            ratios = self._ratios(1, length_m, power)
            rho_sci = ratios.sci if bool(np.any(active)) else 1.0
            rho_xci = np.where(active, 1.0 + ratios.xci, 0.0)
            mci = np.where(active, ratios.mci_gn + ratios.mci_corr, 0.0)
            spm_noise = s2_spm * rho_sci
            xpm_noise = s2_xpm * rho_xci + mci
            return SpanFiberOutput(
                power_out,
                spm_noise,
                xpm_noise,
                amplitude_from_power(spm_noise),
                amplitude_from_power(xpm_noise),
            )

        spm_power, xpm_power, buildup_xpm = self._coherent_levels(
            n_spans, length_m, power, s2_spm, s2_xpm
        )

        step_sci = self._sci_step(n_spans, length_m)
        step_xpm = solve_dirichlet_step(n_spans, buildup_xpm)

        travelled = (
            float(context.cumulative_length_m) + float(context.span_length_m)
        ) / length_m
        phase_sci = step_sci * travelled
        phase_xpm = step_xpm * travelled

        spm_power = np.where(active, spm_power, 0.0)
        amplitude_spm = amplitude_from_power(spm_power) * np.exp(1j * phase_sci)
        amplitude_xpm = amplitude_from_power(xpm_power) * np.exp(1j * phase_xpm)
        return SpanFiberOutput(
            power_out,
            spm_power,
            xpm_power,
            amplitude_spm,
            amplitude_xpm,
        )


class Carena2014Model(RouteModel):
    """Route model for the Carena 2014 EGN approximation.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Channel grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber properties.
    symbol_rate_hz : float, optional
        Symbol rate and rectangular bandwidth in Hz.
    modulation_format : {"bpsk", "qpsk", "16qam", "64qam"}, optional
        Modulation format from the paper's Table 1.
    quadrature_points : int, optional
        Gauss-Legendre nodes per composite interval.
    frequency_points : int, optional
        Observation frequency samples across the channel under test.
    **route_options
        Options passed to the route model.

    Raises
    ------
    KeyError
        If the modulation format is not supported."""

    info = CARENA2014_MODEL_INFO

    supports_span_context = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        modulation_format: str = DEFAULT_MODULATION,
        *,
        quadrature_points: int = DEFAULT_QUADRATURE_POINTS,
        frequency_points: int = DEFAULT_FREQUENCY_POINTS,
        **route_options: Any,
    ) -> None:
        """Initialize the route and model-specific quadrature options."""
        super().__init__(
            grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **route_options
        )
        if modulation_format not in MODULATION_TABLE:
            msg = (
                f"unknown modulation format {modulation_format!r}; "
                f"2014 Table 1 provides {sorted(MODULATION_TABLE)}."
            )
            raise KeyError(msg)
        self.modulation_format = modulation_format
        self.phi, self.psi = MODULATION_TABLE[modulation_format]
        self.quadrature_points = int(quadrature_points)
        self.frequency_points = int(frequency_points)
        self._gn = MultiBandISRSGN(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=self.connections,
            slots_per_channel=self.slots_per_channel,
            channel_bandwidth_source=self.channel_bandwidth_source,
            power_width_mode=self.power_width_mode,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Build the fiber-specific single-span EGN kernel."""
        return Carena2014SpanKernel(
            gn_evaluator=self._gn.for_fiber(fiber_config),
            phi=self.phi,
            psi=self.psi,
            quadrature_points=self.quadrature_points,
            frequency_points=self.frequency_points,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """Evaluate a span and apply amplifier and ASE effects.

        NLI amplitudes are referenced to the fiber output before amplification."""
        kernel = cast("Carena2014SpanKernel", self.kernel_for(fiber_config))
        fiber_output = kernel.evaluate(
            power_in_w, span_length_m, 1, span_context=span_context
        )
        stripped = SpanFiberOutput(
            fiber_output.power_out_w,
            fiber_output.spm_noise_w,
            fiber_output.xpm_noise_w,
        )
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=stripped,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )
        return ElementOutput(
            element.power_out_w,
            element.spm_noise_w,
            element.xpm_noise_w,
            element.ase_noise_w,
            fiber_output.nli_amplitude_spm,
            fiber_output.nli_amplitude_xpm,
        )


register_model(CARENA2014_MODEL_INFO, Carena2014Model)
