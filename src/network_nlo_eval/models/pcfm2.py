"""Coherent PCFM2 for arbitrary rectangular-spectrum SCI, XCI and MCI islands.

Gao, Jiang and Poggiolini, arXiv:2602.03860v2, supplies the polynomial
span-pair closed form. Island tiling, observation-frequency integration,
and polynomial profile fitting remain controlled approximations. Coupled
Raman profiles and ASE are repository extensions, using the shared solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import roots_legendre

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    PathQoTResult,
    SpanFiberOutput,
    SpanKernel,
    register_model,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
)
from network_nlo_eval.models.pcfm2_kernels import polynomial_pair
from network_nlo_eval.models.raman_ode import (
    PaperRamanSpanResult,
    RamanPaperSpanSolver,
    validate_raman_backend,
)
from network_nlo_eval.models.route import (
    RouteModel,
    accumulate_route,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from network_nlo_eval.core.flexgrid import CarrierGeometry, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.models.base import MultiSpanOpticalPath
    from network_nlo_eval.models.pcfm2_kernels import Rectangle
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    _WaveIdentity = tuple[int | None, float, float]
    _IslandIdentity = tuple[_WaveIdentity, _WaveIdentity, _WaveIdentity]

__all__ = ["PCFM2_MODEL_INFO", "PCFM2Model", "PCFM2SpanKernel"]

PCFM2_MODEL_INFO = ModelInfo(
    name="pcfm2",
    summary="Coherent polynomial GN over SCI/XCI/MCI islands and receiver bandwidth.",
    family="gn",
    reference="Gao, Jiang, Poggiolini, 'The Coherent Polynomials Closed-Form Model "
    "for Evaluating Nonlinear Interference in Any Island', arXiv:2602.03860v2 (2026)",
    options=(
        ("poly_degree", "Effective triple-profile polynomial degree, 0 through 9 (5)."),
        (
            "island_cells",
            "First-axis island slices before support-breakpoint splits (8).",
        ),
        (
            "band_points",
            "Gauss nodes across the receiver bandwidth; 1 uses center PSD (5).",
        ),
        (
            "span_coherence",
            "'coherent' (default) or 'incoherent' span-pair accumulation.",
        ),
        ("spp_steps", "Coupled Raman RK4 steps per span (64)."),
        ("include_isrs", "Include signal-to-signal Raman transfer (True)."),
        ("raman_ase", "Include newly generated Raman ASE (True)."),
        ("raman_gain_model", "Shared Raman gain spectrum name or file ('gaussian')."),
        ("raman_iterations", "Maximum backward-pump fixed-point iterations (64)."),
        (
            "raman_backend",
            "Shared Raman solver backend: python, numba, auto ('python').",
        ),
        ("egn", "Only False is supported: PCFM2 implements the GN model."),
    ),
)


def _rectangles(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    strip0: float,
    strip1: float,
    cells: int,
) -> Iterator[Rectangle]:
    """Tile a three-wave island with a midpoint staircase of exact total area."""
    edges = sorted(
        set(np.linspace(x0, x1, cells + 1))
        | {
            point
            for point in (strip0 - y0, strip0 - y1, strip1 - y0, strip1 - y1)
            if x0 < point < x1
        }
    )
    for a, b in pairwise(edges):
        middle = (a + b) / 2
        c, d = max(y0, strip0 - middle), min(y1, strip1 - middle)
        if c < d:
            yield float(a), float(b), float(c), float(d)


@dataclass(frozen=True, slots=True)
class _Span:
    kernel: PCFM2SpanKernel
    power: NDArrayFloat
    length: float
    profile: PaperRamanSpanResult


@dataclass(frozen=True, slots=True)
class _IslandSpan:
    span: _Span
    waves: tuple[int, int, int]
    cut: int
    penalty: float


@dataclass(frozen=True, slots=True)
class PCFM2LinkSpectrum:
    """Actual launches and carrier identities shared by receivers on one link."""

    spans: list[_Span]
    active_rows: list[int]
    identities: dict[int, _WaveIdentity]

    @classmethod
    def from_spans(
        cls, spans: list[_Span], service_rows: dict[int, int]
    ) -> PCFM2LinkSpectrum:
        """Build the identity map once for all receivers using the link."""
        kernel = spans[0].kernel
        services = {row: service_id for service_id, row in service_rows.items()}
        live = _active_rows(spans)
        return cls(
            spans,
            live,
            {
                row: (
                    services.get(row),
                    float(kernel.f_abs_hz[row]),
                    float(kernel.b_hz[row]),
                )
                for row in live
            },
        )


@dataclass(frozen=True, slots=True)
class PCFM2LinkCut:
    """Span launches, carrier identities, and receiver row on one traversed link."""

    spectrum: PCFM2LinkSpectrum
    row: int
    downstream_penalty: float


class PCFM2SpanKernel:
    """Fiber-specific polynomial profiles and analytical PCFM2 island integration.

    Parameters
    ----------
    fiber_config : FiberSpanConfig
        Physical fiber and distributed pumps. In-span lumped losses are unsupported.
    f_abs_hz, b_hz : numpy.ndarray
        Non-overlapping channel centers and rectangular bandwidths in Hz.
    poly_degree, island_cells, band_points, spp_steps : int, optional
        Polynomial, island, receiver, and longitudinal profile resolutions.
    span_coherence : str, optional
        Coherent or incoherent accumulation.
    include_isrs, raman_ase : bool, optional
        Enable shared Raman signal transfer and spontaneous emission.
    raman_gain_model, raman_backend : str, optional
        Shared Raman gain spectrum and solver backend.
    raman_iterations : int, optional
        Backward-pump iteration limit.
    """

    def __init__(
        self,
        fiber_config: FiberSpanConfig,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        *,
        poly_degree: int = 5,
        island_cells: int = 8,
        band_points: int = 5,
        span_coherence: str = "coherent",
        spp_steps: int = 64,
        include_isrs: bool = True,
        raman_ase: bool = True,
        raman_gain_model: str = "gaussian",
        raman_backend: str = "python",
        raman_iterations: int = 64,
    ) -> None:
        """Validate resolutions and construct wavelength-dependent fiber parameters."""
        for name, value, low, high in (
            ("poly_degree", poly_degree, 0, 9),
            ("island_cells", island_cells, 1, 4096),
            ("band_points", band_points, 1, 256),
            ("spp_steps", spp_steps, 1, 100000),
            ("raman_iterations", raman_iterations, 1, 10000),
        ):
            if (
                isinstance(value, bool)
                or int(value) != value
                or not low <= value <= high
            ):
                msg = f"{name} must be an integer in [{low}, {high}], got {value!r}."
                raise ValueError(msg)
        if 2 * spp_steps < poly_degree:
            msg = "spp_steps must supply at least poly_degree + 1 profile samples."
            raise ValueError(msg)
        if span_coherence not in ("coherent", "incoherent"):
            msg = "span_coherence must be 'coherent' or 'incoherent'."
            raise ValueError(msg)
        if fiber_config.lumped_loss:
            msg = "PCFM2 requires continuous span profiles; in-span lumped losses are unsupported."
            raise ValueError(msg)
        self.fiber = fiber_config
        self.f_abs_hz = np.array(f_abs_hz, dtype=float, copy=True)
        self.b_hz = np.array(b_hz, dtype=float, copy=True)
        if (
            self.f_abs_hz.ndim != 1
            or self.f_abs_hz.size == 0
            or self.f_abs_hz.shape != self.b_hz.shape
            or not np.all(np.isfinite(self.f_abs_hz))
            or not np.all(np.isfinite(self.b_hz))
            or np.any(self.f_abs_hz <= 0)
            or np.any(self.b_hz <= 0)
        ):
            msg = "Channel centers and bandwidths must be finite positive vectors of equal size."
            raise ValueError(msg)
        order = np.argsort(self.f_abs_hz)
        lower, upper = self.f_abs_hz - self.b_hz / 2, self.f_abs_hz + self.b_hz / 2
        if np.any(lower[order][1:] < upper[order][:-1]):
            msg = "PCFM2 requires non-overlapping rectangular channel spectra."
            raise ValueError(msg)
        for pump in fiber_config.raman_pumps:
            frequency = C_LIGHT / (pump.wavelength_nm * 1e-9)
            if pump.power_w > 0 and np.any((lower <= frequency) & (frequency <= upper)):
                msg = "A Raman pump overlaps a signal channel."
                raise ValueError(msg)
        self.poly_degree, self.island_cells, self.band_points = (
            int(poly_degree),
            int(island_cells),
            int(band_points),
        )
        self.span_coherence = span_coherence
        self.spp_steps, self.raman_iterations = int(spp_steps), int(raman_iterations)
        self.include_isrs, self.raman_ase = bool(include_isrs), bool(raman_ase)
        self.raman_gain_model = raman_gain_model
        self.raman_backend = validate_raman_backend(raman_backend)
        wavelengths = C_LIGHT / self.f_abs_hz
        self.a_eff_m2 = _calc_aeff_dynamic(
            wavelengths,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        if fiber_config.use_attenuation_polynomial:
            self.alpha_npm = _calc_fiber_attenuation_npm(
                wavelengths,
                fiber_config.reference_wavelength_nm,
                fiber_config.attenuation_alpha2,
                fiber_config.attenuation_alpha1,
                fiber_config.attenuation_alpha0,
            )
        else:
            self.alpha_npm = np.full(
                self.f_abs_hz.size,
                fiber_config.attenuation_db_km_ref * np.log(10) / 1e4,
            )
        if (
            not np.all(np.isfinite(self.a_eff_m2))
            or np.any(self.a_eff_m2 <= 0)
            or not np.all(np.isfinite(self.alpha_npm))
            or np.any(self.alpha_npm < 0)
        ):
            msg = "PCFM2 requires positive effective areas and nonnegative finite attenuation."
            raise ValueError(msg)
        reference = fiber_config.reference_wavelength_nm * 1e-9
        self.f_reference_hz = C_LIGHT / reference
        self.beta2 = float(
            _calc_beta2_from_d(fiber_config.dispersion_parameter_d, reference)
        )
        self.beta3 = float(
            _calc_beta3_from_s_d(
                fiber_config.dispersion_slope_s,
                fiber_config.dispersion_parameter_d,
                reference,
            )
        )
        self._solvers: dict[float, RamanPaperSpanSolver] = {}
        self._profile_cache: dict[tuple[float, bytes], PaperRamanSpanResult] = {}
        self.last_profile_error = 0.0

    def _profile(self, power: NDArrayFloat, length: float) -> PaperRamanSpanResult:
        """Solve and cache a continuous normalized signal profile."""
        if (
            power.shape != self.f_abs_hz.shape
            or not np.all(np.isfinite(power))
            or np.any(power < 0)
            or not np.isfinite(length)
            or length <= 0
        ):
            msg = "Power must be a finite nonnegative channel vector and length must be positive."
            raise ValueError(msg)
        key = (length, power.tobytes())
        cached = self._profile_cache.get(key)
        if cached is not None:
            return cached
        if not self.include_isrs and not any(
            p.power_w > 0 for p in self.fiber.raman_pumps
        ):
            positions = np.arange(1, 2 * self.spp_steps + 1) / (2 * self.spp_steps)
            return PaperRamanSpanResult(
                profile_norm=np.exp(-length * positions[:, None] * self.alpha_npm),
                power_out_w=power * np.exp(-length * self.alpha_npm),
                raman_ase_w=np.zeros_like(power),
                pump_power_out_w=np.zeros(len(self.fiber.raman_pumps)),
                iterations=0,
                converged=True,
            )
        solver = self._solvers.get(length)
        if solver is None:
            pumps = self.fiber.raman_pumps
            pump_wavelengths = np.array([p.wavelength_nm * 1e-9 for p in pumps])
            solver = RamanPaperSpanSolver(
                self.f_abs_hz,
                self.b_hz,
                pumps,
                length,
                self.alpha_npm,
                self.a_eff_m2,
                np.array([p.attenuation_db_per_km * np.log(10) / 1e4 for p in pumps]),
                _calc_aeff_dynamic(
                    pump_wavelengths,
                    self.fiber.effective_area_um2_ref,
                    self.fiber.aeff_slope_um2_nm,
                    self.fiber.reference_wavelength_nm,
                ),
                self.raman_gain_model,
                n_steps=self.spp_steps,
                n_iterations=self.raman_iterations,
                include_isrs=self.include_isrs,
                include_ase=self.raman_ase,
                backend=self.raman_backend,
            )
            self._solvers[length] = solver
        result = solver.solve(power)
        if not result.converged or not np.all(np.isfinite(result.profile_norm)):
            msg = "PCFM2 Raman profiles did not converge to finite values."
            raise RuntimeError(msg)
        if len(self._profile_cache) >= 8:
            self._profile_cache.pop(next(iter(self._profile_cache)))
        self._profile_cache[key] = result
        return result

    def _polynomial(
        self,
        span: _Span,
        waves: tuple[int, int, int],
        cut: int,
    ) -> tuple[float, ...]:
        """Fit the effective triple profile, rather than individual power profiles."""
        profile = span.profile.profile_norm
        denominator = profile[:, cut]
        effective = np.sqrt(
            np.divide(
                np.prod(profile[:, waves], axis=1),
                denominator,
                out=np.zeros_like(denominator),
                where=denominator > 0,
            )
        )
        # The shared solver samples the midpoint and endpoint of each step;
        # its first row is at 1/(2*steps), rather than at the launch plane.
        positions = np.arange(1, profile.shape[0] + 1) / profile.shape[0]
        positions = np.concatenate(([0.0], positions))
        effective = np.concatenate(([1.0], effective))
        fitted = Polynomial.fit(  # type: ignore[no-untyped-call]
            positions, effective, self.poly_degree, domain=[0, 1]
        )
        error = float(np.max(np.abs(fitted(positions) - effective)))
        self.last_profile_error = max(
            self.last_profile_error, error / max(float(np.max(effective)), 1e-300)
        )
        return tuple(float(v) for v in fitted.convert().coef)

    def nli_psd(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        cut: int,
        observation_hz: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Return single-span SCI and non-SCI PSD in W/Hz at arbitrary CUT frequencies.

        Results use exact span-loss compensation. Non-SCI includes XCI and MCI.
        Observation frequencies must lie within the selected channel.
        """
        observations = np.asarray(observation_hz, dtype=float)
        if not 0 <= cut < self.f_abs_hz.size:
            msg = "CUT index is outside the channel grid."
            raise ValueError(msg)
        if (
            observations.ndim != 1
            or not np.all(np.isfinite(observations))
            or np.any(abs(observations - self.f_abs_hz[cut]) > self.b_hz[cut] / 2)
        ):
            msg = "Observation frequencies must be a finite vector within the CUT."
            raise ValueError(msg)
        power = np.asarray(power_in_w, dtype=float)
        span = _Span(self, power, span_length_m, self._profile(power, span_length_m))
        return _psd([span], power, np.ones(1), cut, observations)

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """Evaluate one physical span; full-route coherence is handled by PCFM2Model."""
        if n_effective_spans != 1:
            msg = "Use PCFM2Model.evaluate_path for multiple physical spans."
            raise ValueError(msg)
        power = np.asarray(power_in_w, dtype=float)
        profile = self._profile(power, span_length_m)
        sci, other = _integrate(
            [_Span(self, power, span_length_m, profile)], power, np.ones(1)
        )
        return SpanFiberOutput(profile.power_out_w.copy(), sci, other)


def _supported_triples(
    kernel: PCFM2SpanKernel,
    live: list[int],
    observations: NDArrayFloat,
    center: float,
) -> Iterator[tuple[int, int, int]]:
    """Find third-wave bands that overlap the mixing support of each input pair."""
    if not live or not observations.size:
        return
    offsets = kernel.f_abs_hz - center
    lower, upper = offsets - kernel.b_hz / 2, offsets + kernel.b_hz / 2
    ordered = sorted(live, key=offsets.__getitem__)
    minimum, maximum = observations.min() - center, observations.max() - center
    ordered_lower, ordered_upper = lower[ordered], upper[ordered]
    starts = np.searchsorted(
        ordered_upper, lower[live, None] + lower[live] - maximum, side="right"
    )
    stops = np.searchsorted(
        ordered_lower, upper[live, None] + upper[live] - minimum, side="left"
    )
    for first_index, first in enumerate(live):
        for second_index in range(first_index, len(live)):
            second = live[second_index]
            for third in sorted(
                ordered[
                    starts[first_index, second_index] : stops[first_index, second_index]
                ]
            ):
                yield first, second, third


def _island_psd(
    spans: list[_IslandSpan],
    receiver_power: float,
    observations: NDArrayFloat,
    destination: NDArrayFloat,
) -> None:
    """Integrate a single mixing island across spans with matching carriers."""
    initial = spans[0]
    kernel = initial.span.kernel
    first, second, _ = initial.waves
    frequencies, bandwidths = kernel.f_abs_hz, kernel.b_hz
    center_first, center_second, center_third = frequencies[list(initial.waves)]
    half_first, half_second, half_third = bandwidths[list(initial.waves)] / 2
    scale = (16 / 27) * (1 if first == second else 2)
    parameters = []
    for item in spans:
        span, cut = item.span, item.cut
        local = span.kernel
        wave_first, wave_second, wave_third = item.waves
        stimulus = (
            (span.power[wave_first] / local.b_hz[wave_first])
            * (span.power[wave_second] / local.b_hz[wave_second])
            * (span.power[wave_third] / local.b_hz[wave_third])
        )
        amplitude = 0.0
        profile: tuple[float, ...] = (0.0,)
        if stimulus > 0 and span.power[cut] > 0:
            area = (
                local.a_eff_m2[cut]
                + local.a_eff_m2[wave_first]
                + local.a_eff_m2[wave_second]
                + local.a_eff_m2[wave_third]
            ) / 4
            amplitude = (
                (2 * np.pi / C_LIGHT)
                * local.fiber.nonlinear_index_n2
                * span.length
                / area
                * np.sqrt(stimulus * receiver_power / span.power[cut] * item.penalty)
            )
            profile = local._polynomial(span, item.waves, cut)
        parameters.append(
            (
                local.beta2 * span.length,
                np.pi * local.beta3 * span.length,
                local.f_reference_hz,
                amplitude,
                profile,
            )
        )
    coherent = kernel.span_coherence == "coherent"
    for index, observation in enumerate(observations):
        offset_first = center_first - observation
        offset_second = center_second - observation
        offset_third = center_third - observation
        x0, x1 = offset_first - half_first, offset_first + half_first
        y0, y1 = offset_second - half_second, offset_second + half_second
        strip0, strip1 = offset_third - half_third, offset_third + half_third
        if x1 + y1 <= strip0 or x0 + y0 >= strip1:
            continue
        for rectangle in _rectangles(
            x0, x1, y0, y1, strip0, strip1, kernel.island_cells
        ):
            middle_sum = sum(rectangle) / 2
            phase = 0.0
            contribution = 0.0
            previous: list[tuple[float, float, float, tuple[float, ...]]] = []
            for beta2_length, beta3_length, reference, amplitude, profile in parameters:
                dispersion = beta2_length + beta3_length * (
                    2 * (observation - reference) + middle_sum
                )
                if amplitude:
                    field = amplitude * observation
                    contribution += field**2 * polynomial_pair(
                        rectangle, profile, profile, dispersion, dispersion
                    )
                    if coherent:
                        for (
                            old_dispersion,
                            old_phase,
                            old_field,
                            old_profile,
                        ) in previous:
                            contribution += (
                                2
                                * field
                                * old_field
                                * polynomial_pair(
                                    rectangle,
                                    profile,
                                    old_profile,
                                    dispersion,
                                    old_dispersion,
                                    old_phase - phase,
                                )
                            )
                        previous.append((dispersion, phase, field, profile))
                phase += dispersion
            destination[index] += scale * contribution


def _check_psd(*arrays: NDArrayFloat) -> None:
    for values in arrays:
        maximum = float(np.max(np.abs(values), initial=0.0))
        if np.any(values < -1e-12 * max(maximum, 1e-300)):
            msg = "PCFM2 produced negative NLI; increase profile or island resolution."
            raise ArithmeticError(msg)
        np.maximum(values, 0, out=values)


def _psd(
    spans: list[_Span],
    receiver: NDArrayFloat,
    penalties: NDArrayFloat,
    cut: int,
    observations: NDArrayFloat,
    live: list[int] | None = None,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Integrate all channel triples and explicit self and cross span pairs."""
    kernel = spans[0].kernel
    sci, other = np.zeros(observations.size), np.zeros(observations.size)
    if receiver[cut] <= 0:
        return sci, other
    if live is None:
        live = _active_rows(spans)
    for waves in _supported_triples(kernel, live, observations, kernel.f_abs_hz[cut]):
        _island_psd(
            [
                _IslandSpan(span, waves, cut, penalty)
                for span, penalty in zip(spans, penalties, strict=True)
            ],
            float(receiver[cut]),
            observations,
            sci if waves == (cut, cut, cut) else other,
        )
    _check_psd(sci, other)
    return sci, other


def _active_rows(spans: list[_Span]) -> list[int]:
    live = np.any(np.array([span.power for span in spans]) > 0, axis=0)
    return cast("list[int]", np.flatnonzero(live).tolist())


@lru_cache(maxsize=16)
def _receiver_quadrature(points: int) -> tuple[NDArrayFloat, NDArrayFloat]:
    nodes, weights = roots_legendre(points)
    return nodes, weights


def mixed_path_nli(links: list[PCFM2LinkCut], receiver_power: float) -> float:
    """Integrate PCFM2 NLI for one receiver using its link-local span spectra."""
    initial = links[0]
    kernel = initial.spectrum.spans[0].kernel
    center, half = kernel.f_abs_hz[initial.row], kernel.b_hz[initial.row] / 2
    quadrature, weights = _receiver_quadrature(kernel.band_points)
    observations = center + half * quadrature
    psd = np.zeros_like(observations)
    groups: list[list[_IslandSpan]] = []
    previous: dict[_IslandIdentity, list[_IslandSpan]] = {}
    for link in links:
        spectrum = link.spectrum
        local = spectrum.spans[0].kernel
        row = link.row
        if (
            local.f_abs_hz[row] != center
            or local.b_hz[row] != 2 * half
            or local.band_points != kernel.band_points
            or local.island_cells != kernel.island_cells
        ):
            msg = "Mixed-load PCFM2 requires consistent receiver geometry and integration settings."
            raise ValueError(msg)
        current: dict[_IslandIdentity, list[_IslandSpan]] = {}
        for waves in _supported_triples(
            local, spectrum.active_rows, observations, center
        ):
            first, second, third = (spectrum.identities[channel] for channel in waves)
            key = first, second, third
            group = previous.get(key)
            if group is None:
                group = []
                groups.append(group)
            group.extend(
                _IslandSpan(span, waves, row, link.downstream_penalty)
                for span in spectrum.spans
            )
            if all(identity[0] is not None for identity in key):
                current[key] = group
        previous = current
    for group in groups:
        _island_psd(group, receiver_power, observations, psd)
    _check_psd(psd)
    return float(half * (weights @ psd))


def _integrate(
    spans: list[_Span],
    receiver: NDArrayFloat,
    penalties: NDArrayFloat,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Integrate the PSD through rectangular matched receiver filters."""
    kernel = spans[0].kernel
    nodes, weights = _receiver_quadrature(kernel.band_points)
    sci, other = np.zeros_like(receiver), np.zeros_like(receiver)
    live = _active_rows(spans)
    for cut, center in enumerate(kernel.f_abs_hz):
        half = kernel.b_hz[cut] / 2
        psd_sci, psd_other = _psd(
            spans, receiver, penalties, cut, center + half * nodes, live
        )
        sci[cut], other[cut] = half * (weights @ psd_sci), half * (weights @ psd_other)
    return sci, other


class PCFM2Model(RouteModel):
    """PCFM2 with actual launches and receiver-referenced span pairs.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Channel grid; flexible grids require the shared connections option.
    ref_fiber_config : FiberSpanConfig
        Reference physical fiber.
    symbol_rate_hz : float, optional
        Fixed-grid rectangular channel bandwidth.
    egn : bool, optional
        Must be False.
    **options
        PCFM2SpanKernel resolution/profile options and shared RouteModel options.
    """

    info = PCFM2_MODEL_INFO
    supports_distributed_raman = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        egn: bool = False,
        **options: Any,
    ) -> None:
        """Initialize shared route conventions and validate kernel settings."""
        if egn:
            msg = "PCFM2 implements GN only; use egn=False."
            raise ValueError(msg)
        kernel_names = {name for name, _ in self.info.options} - {"egn"}
        self._options = {
            key: options.pop(key) for key in list(options) if key in kernel_names
        }
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **options)
        self.kernel_for(ref_fiber_config)

    @property
    def supports_mixed_load_coherence(self) -> bool:
        """Return whether this model is configured for coherent span pairs."""
        return bool(self._options.get("span_coherence", "coherent") == "coherent")

    def for_connections(
        self, geometry: tuple[CarrierGeometry, ...], slots_per_channel: int
    ) -> PCFM2Model:
        """Bind the same PCFM2 settings to a compact carrier spectrum."""
        return type(self)(
            geometry[0].grid,
            self.ref_fiber_config,
            self.symbol_rate_hz,
            connections=geometry,
            slots_per_channel=slots_per_channel,
            channel_bandwidth_source=self.channel_bandwidth_source,
            power_width_mode=self.power_width_mode,
            **self._options,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        return PCFM2SpanKernel(
            fiber_config,
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            **self._options,
        )

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """Evaluate standalone span signal, ASE and NLI."""
        return self._span_step(fiber_config, edfa_config, power_in_w, span_length_m)

    def _span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        collected: list[_Span] | None = None,
    ) -> ElementOutput:
        kernel = cast("PCFM2SpanKernel", self.kernel_for(fiber_config))
        profile = kernel._profile(power_in_w, span_length_m)
        span = _Span(kernel, power_in_w, span_length_m, profile)
        if collected is None:
            sci, other = _integrate([span], power_in_w, np.ones(1))
        else:
            collected.append(span)
            sci, other = np.zeros_like(power_in_w), np.zeros_like(power_in_w)
        fiber_output = SpanFiberOutput(profile.power_out_w.copy(), sci, other)
        return apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes),
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            input_ase_w=profile.raman_ase_w.copy(),
        )

    def propagate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> tuple[PathQoTResult, list[_Span]]:
        """Propagate linear impairments and record span launches for later NLI."""
        spans: list[_Span] = []

        def record(
            fiber: FiberSpanConfig,
            edfa: EDFAConfig | None,
            power: NDArrayFloat,
            length: float,
        ) -> ElementOutput:
            return self._span_step(fiber, edfa, power, length, spans)

        result = accumulate_route(
            self,
            optical_path,
            power_in_w,
            ase_polarization_scale=self.ase_polarization_scale,
            span_step=record,
        )
        return result, spans

    @staticmethod
    def complete_path(
        optical_path: MultiSpanOpticalPath,
        result: PathQoTResult,
        spans: list[_Span],
    ) -> PathQoTResult:
        """Integrate a propagated path with a common spectrum on every span."""
        penalties: NDArrayFloat = np.ones(len(spans))
        if optical_path.roadm_configs:
            cursor = 0
            for fiber, roadm in zip(
                optical_path.fiber_configs, optical_path.roadm_configs, strict=True
            ):
                remaining = fiber.length_km
                while remaining > 1e-12:
                    remaining -= min(remaining, fiber.max_span_length_km)
                    cursor += 1
                if roadm is not None:
                    penalties[cursor - 1] *= 10 ** (roadm.filtering_penalty_db / 10)
            penalties = np.cumprod(penalties[::-1])[::-1]
        sci, other = _integrate(spans, result.signal_power_w, penalties)
        noise = sci + other + result.ase_noise_w
        snr = np.full_like(noise, -np.inf)
        active = result.signal_power_w > 0
        noisy = active & (noise > 0)
        snr[noisy] = 10 * np.log10(result.signal_power_w[noisy] / noise[noisy])
        snr[active & (noise == 0)] = np.inf
        return replace(result, spm_noise_w=sci, xpm_noise_w=other, snr_db=snr)

    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> PathQoTResult:
        """Replace route NLI by the full receiver-referenced span-pair integral."""
        result, spans = self.propagate_path(optical_path, power_in_w)
        return self.complete_path(optical_path, result, spans)


register_model(PCFM2_MODEL_INFO, PCFM2Model)
