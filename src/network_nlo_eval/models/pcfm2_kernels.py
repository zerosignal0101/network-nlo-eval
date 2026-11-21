"""Analytical rectangle and polynomial span-pair kernels for coherent PCFM2.

The overlap polynomial implements Gao, Jiang and Poggiolini,
arXiv:2602.03860v2, Sec. III and Appendix A. Coordinates along each span
are normalized to [0, 1]. Signed dispersion uses ordered transformed
endpoints and absolute Jacobians. Adaptive extended precision protects the
differences of primitive moments; returned physical quantities are float64.
No longitudinal or frequency quadrature is used inside a rectangle.
"""

from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
from itertools import pairwise
from math import ceil, comb, log10
from typing import Any

from mpmath import mp

__all__ = ["polynomial_pair", "rectangle_response"]

# A private context avoids changing application-wide mpmath precision.
_DEFAULT_CONTEXT: Any = mp.clone()
_DEFAULT_CONTEXT.dps = 50
_ARITHMETIC: ContextVar[Any] = ContextVar("pcfm2_arithmetic", default=_DEFAULT_CONTEXT)
Rectangle = tuple[float, float, float, float]


def _j(value: Any) -> Any:
    """Evaluate the odd primitive int_0^x Si(t)/t dt."""
    _mp = _ARITHMETIC.get()
    if abs(value) <= max(120, 3 * _mp.dps):
        return value * _mp.hyper(
            [_mp.mpf("0.5")] * 2, [_mp.mpf("1.5")] * 3, -(value**2) / 4
        )
    x = abs(value)
    # Retain enough terms for the current precision, with the switch
    # above placing the omitted remainder below that precision.
    count = _mp.dps + 10
    top = 2 * count + 6
    cosine, sine = [_mp.mpf(0)] * (top + 2), [_mp.mpf(0)] * (top + 2)
    sx, cx = _mp.sin(x), _mp.cos(x)
    for order in range(top, 1, -1):
        cosine[order] = -sx / x**order + order * sine[order + 1]
        sine[order] = cx / x**order - order * cosine[order + 1]
    tail = -sum(
        (-1) ** k
        * (
            _mp.factorial(2 * k) * cosine[2 * k + 2]
            + _mp.factorial(2 * k + 1) * sine[2 * k + 3]
        )
        for k in range(count)
    )
    result = _mp.pi / 2 * (_mp.log(x) + _mp.euler) - tail
    return result if value > 0 else -result


@lru_cache(maxsize=8192)
def _trigonometric(value: Any, precision: int) -> tuple[Any, Any, Any]:
    """Share special-function evaluations between all primitive moment orders."""
    _mp = _ARITHMETIC.get()
    assert precision == _mp.dps
    return _mp.sin(value), _mp.cos(value), _mp.si(value)


def _h(order: int, lam: Any, point: Any) -> Any:
    """Primitive of s**(order-1) Si(lam*s), with stable small arguments."""
    _mp = _ARITHMETIC.get()
    x = lam * point
    if x == 0:
        return _mp.mpf(0)
    if abs(x) < 1:
        total = _mp.mpf(0)
        for k in range(max(100, _mp.dps)):
            term = (
                (-1) ** k
                * lam ** (2 * k + 1)
                * point ** (order + 2 * k + 1)
                / ((2 * k + 1) * _mp.factorial(2 * k + 1) * (order + 2 * k + 1))
            )
            total += term
            if abs(term) < _mp.mpf(10) ** (-_mp.dps - 2) * abs(total):
                break
        return total
    if order == 0:
        return _j(x)
    sx, cx, si = _trigonometric(x, _mp.dps)
    integrals = [1 - cx, sx - x * cx]
    for n in range(2, order):
        integrals.append(
            -(x**n) * cx + n * x ** (n - 1) * sx - n * (n - 1) * integrals[n - 2]
        )
    return (point**order * si - integrals[order - 1] / lam**order) / order


@lru_cache(maxsize=8192)
def _moments(
    rectangle: Rectangle, scale: float, point: Any, degree: int, precision: int
) -> tuple[Any, ...]:
    """Return primitive moments of F(scale*s) at one normalized endpoint."""
    _mp = _ARITHMETIC.get()
    assert precision == _mp.dps
    a, b, c, d = map(_mp.mpf, rectangle)
    factor = 4 * _mp.pi**2 * scale
    corners = (a * d, a * c, b * c, b * d)
    return tuple(
        sum(
            sign * _h(order, factor * corner, point)
            for sign, corner in zip((-1, 1, -1, 1), corners, strict=True)
        )
        / factor
        for order in range(degree + 1)
    )


def rectangle_response(rectangle: Rectangle, dispersion: float) -> float:
    """Integrate cos(4*pi**2*x*y*dispersion) over a frequency rectangle."""
    _mp = _ARITHMETIC.get()
    a, b, c, d = map(_mp.mpf, rectangle)
    if dispersion == 0:
        return float((b - a) * (d - c))
    factor = 4 * _mp.pi**2 * dispersion
    return float(
        sum(
            sign * _mp.si(factor * corner)
            for sign, corner in zip(
                (-1, 1, -1, 1), (a * d, a * c, b * c, b * d), strict=True
            )
        )
        / factor
    )


def _add(target: list[Any], source: list[Any], scale: Any = 1) -> None:
    """Add a scaled polynomial in ascending power order."""
    _mp = _ARITHMETIC.get()
    target.extend([_mp.mpf(0)] * max(0, len(source) - len(target)))
    for index, coefficient in enumerate(source):
        target[index] += scale * coefficient


def _multiply(first: list[Any], second: list[Any]) -> list[Any]:
    """Multiply ascending-order polynomials."""
    _mp = _ARITHMETIC.get()
    result = [_mp.mpf(0)] * (len(first) + len(second) - 1)
    for n, left in enumerate(first):
        for m, right in enumerate(second):
            result[n + m] += left * right
    return result


def _affine_power(offset: Any, slope: Any, degree: int) -> list[Any]:
    """Expand (offset + slope*s)**degree."""
    return [
        comb(degree, k) * offset ** (degree - k) * slope**k for k in range(degree + 1)
    ]


def _integrate_moments(
    polynomial: list[Any], rectangle: Rectangle, scale: float, low: Any, high: Any
) -> Any:
    precision = _ARITHMETIC.get().dps
    degree = len(polynomial) - 1
    left = _moments(rectangle, scale, low, degree, precision)
    right = _moments(rectangle, scale, high, degree, precision)
    return sum(
        coefficient * (b - a)
        for coefficient, a, b in zip(polynomial, left, right, strict=True)
    )


@lru_cache(maxsize=1024)
def _overlap(
    first: tuple[float, ...],
    second: tuple[float, ...],
    dv: Any,
    dw: Any,
    center: Any,
    precision: int,
) -> tuple[tuple[Any, Any, tuple[Any, ...]], ...]:
    """Build the piecewise correlation polynomial independently of the rectangle."""
    _mp = _ARITHMETIC.get()
    assert precision == _mp.dps
    p, q = list(map(_mp.mpf, first)), list(map(_mp.mpf, second))
    x0, x1 = sorted((_mp.mpf(0), dv))
    y0, y1 = sorted((_mp.mpf(0), dw))
    breaks = sorted({center - x + y for x in (x0, x1) for y in (y0, y1)})
    pieces = []
    for low, high in pairwise(breaks):
        midpoint = (low + high) / 2
        lower = (x0, 0) if x0 >= center + y0 - midpoint else (center + y0, -1)
        upper = (x1, 0) if x1 <= center + y1 - midpoint else (center + y1, -1)
        overlap: list[Any] = []
        for n, pn in enumerate(p):
            for m, qm in enumerate(q):
                for k in range(m + 1):
                    exponent = n + k + 1
                    bound = _affine_power(*upper, exponent)
                    _add(bound, _affine_power(*lower, exponent), -1)
                    term = _multiply(_affine_power(-center, 1, m - k), bound)
                    _add(
                        overlap,
                        term,
                        pn * qm * comb(m, k) / (dv**n * dw**m * exponent),
                    )
        pieces.append((low, high, tuple(overlap)))
    return tuple(pieces)


@lru_cache(maxsize=8192)
def polynomial_pair(
    rectangle: Rectangle,
    first: tuple[float, ...],
    second: tuple[float, ...],
    delta_first: float,
    delta_second: float,
    separation: float = 0.0,
) -> float:
    """Integrate p(u)q(v)F(C-delta_first*u+delta_second*v) on [0,1]^2.

    Polynomial coefficients are in ascending powers of normalized position.
    Deltas are beta_eff times span length in s^2; separation is the accumulated
    dispersion at the second span's entrance minus that at the first entrance.
    Lengths, gamma, PSDs and the self/cross multiplicities are applied by callers.
    """
    nonzero = [abs(v) for v in (delta_first, delta_second) if v != 0]
    digits = 50
    if nonzero:
        scale = max(abs(delta_first), abs(delta_second), abs(separation))
        lost = max(0, ceil(log10(scale) - log10(min(nonzero))))
        digits = max(digits, 30 + (len(first) + len(second) + 3) * lost)
    context = _DEFAULT_CONTEXT
    if digits > 50:
        context = mp.clone()
        context.dps = digits
    token = _ARITHMETIC.set(context)
    try:
        return _pair(rectangle, first, second, delta_first, delta_second, separation)
    finally:
        _ARITHMETIC.reset(token)


def _pair(
    rectangle: Rectangle,
    first: tuple[float, ...],
    second: tuple[float, ...],
    delta_first: float,
    delta_second: float,
    separation: float,
) -> float:
    """Evaluate one pair within the caller's context-local precision."""
    _mp = _ARITHMETIC.get()
    p, q = list(map(_mp.mpf, first)), list(map(_mp.mpf, second))
    ip = sum(value / (k + 1) for k, value in enumerate(p))
    iq = sum(value / (k + 1) for k, value in enumerate(q))
    if delta_first == delta_second == 0:
        return float(ip * iq * rectangle_response(rectangle, separation))
    scale = max(abs(delta_first), abs(delta_second), abs(separation))
    dv, dw, center = (
        _mp.mpf(value) / scale for value in (delta_first, delta_second, separation)
    )
    if dv == 0 or dw == 0:
        # One-dimensional limiting kernel; no division by a vanishing beta.
        profile, delta, sign, integrated = (
            (p, dv, -1, iq) if dw == 0 else (q, dw, 1, ip)
        )
        slope = sign * delta
        transformed: list[Any] = []
        for degree, coefficient in enumerate(profile):
            _add(
                transformed,
                _affine_power(-center / slope, 1 / slope, degree),
                coefficient,
            )
        low, high = sorted((center, center + slope))
        return float(
            integrated
            / abs(slope)
            * _integrate_moments(transformed, rectangle, scale, low, high)
        )
    total = _mp.mpf(0)
    for low, high, overlap in _overlap(first, second, dv, dw, center, _mp.dps):
        total += _integrate_moments(list(overlap), rectangle, scale, low, high)
    return float(total / abs(dv * dw))
