"""Receiver-referenced cGN span-pair integration on general optical routes.

The squared coherent sum includes both the SS and XS terms of Jiang et al.,
arXiv:2610.11624v1, Eqs. (11)-(14). Each integration island is covered by the
same midpoint rectangles used by the single-span correlation kernel. The
frequency-product integral retains each span's own dispersion, launch PSD,
NSP, nonlinearity, and transfer to the receiver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.integrate import quad

from network_nlo_eval.models.cgn import (
    _band_quadrature,
    _effective_beta2,
    _island_cover,
    _isrs_span_profile,
    _quadrant_pieces,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.models.base import MultiSpanOpticalPath
    from network_nlo_eval.models.cgn import CGNSpanKernel
    from network_nlo_eval.network.elements import ROADMConfig


def _exponential_integral(rate: complex, length_m: float) -> complex:
    """Integrate exp(-rate*z) over a segment, including its zero-rate limit."""
    value = rate * length_m
    if abs(value) < 1e-8:
        return length_m * (1 - value / 2 + value**2 / 6)
    return complex(-np.expm1(-value) / rate)


def _filtering_to_receiver(path: MultiSpanOpticalPath, span_count: int) -> NDArrayFloat:
    """Return the downstream noise-only ROADM penalty for each source span."""
    penalties = np.ones(span_count)
    cursor = 0
    roadms: Sequence[ROADMConfig | None] = path.roadm_configs or [None] * len(
        path.fiber_configs
    )
    for fiber, roadm in zip(path.fiber_configs, roadms, strict=True):
        remaining = fiber.length_km
        while remaining > 1e-12:
            remaining -= min(remaining, fiber.max_span_length_km)
            cursor += 1
        if roadm is not None:
            penalties[cursor - 1] *= 10 ** (roadm.filtering_penalty_db / 10)
    return np.asarray(np.cumprod(penalties[::-1])[::-1], dtype=np.float64)


def coherent_route_nli(
    spans: list[tuple[CGNSpanKernel, NDArrayFloat, float]],
    receiver_power_w: NDArrayFloat,
    optical_path: MultiSpanOpticalPath,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Integrate the coherent SS+XS contribution for heterogeneous spans.

    Rectangular spectra and channel-wise fiber parameters follow the span
    kernel. Adaptive one-dimensional integration resolves the span-pair
    oscillations; ``correlation_points`` controls its subdivision budget.
    """
    first_kernel = spans[0][0]
    frequencies = first_kernel.f_abs_hz
    bandwidth = first_kernel.b_hz
    half = bandwidth / 2
    n_channels = frequencies.size
    sci, other = np.zeros(n_channels), np.zeros(n_channels)
    penalties = _filtering_to_receiver(optical_path, len(spans))
    profiles = []
    for kernel, power, length in spans:
        if kernel.include_isrs:
            positions, gain, rate = _isrs_span_profile(
                kernel.f_rel_hz,
                kernel.alpha_npm,
                power,
                kernel.f_m_hz,
                kernel.f_M_hz,
                kernel.k_bar,
                length,
                kernel.isrs_segments,
            )
        else:
            positions = np.array([length / 2])
            gain = np.ones((1, n_channels))
            rate = np.zeros(n_channels)
        profiles.append((positions, gain, rate))
    band_nodes = (
        _band_quadrature(first_kernel.band_points)
        if first_kernel.band_points > 1
        else ((0.0, 2.0),)
    )
    for cut in range(n_channels):
        if receiver_power_w[cut] <= 0:
            continue
        for node, weight in band_nodes:
            observation = frequencies[cut] + half[cut] * node
            band_weight = half[cut] * weight
            offsets = frequencies - observation
            for first in range(n_channels):
                for second in range(first, n_channels):
                    symmetry = 1 if first == second else 2
                    sum_low = (
                        offsets[first] - half[first] + offsets[second] - half[second]
                    )
                    sum_high = (
                        offsets[first] + half[first] + offsets[second] + half[second]
                    )
                    candidates = np.flatnonzero(
                        (offsets + half > sum_low) & (offsets - half < sum_high)
                    )
                    for third in candidates:
                        cover = _island_cover(
                            np.array(offsets[first] - half[first]),
                            np.array(offsets[first] + half[first]),
                            np.array(offsets[second] - half[second]),
                            np.array(offsets[second] + half[second]),
                            np.array([offsets[third] - half[third]]),
                            np.array([offsets[third] + half[third]]),
                            first_kernel.island_cells,
                        )
                        for cell in range(first_kernel.island_cells):
                            rectangle = [
                                float(values.ravel()[cell]) for values in cover
                            ]
                            if (
                                rectangle[0] >= rectangle[1]
                                or rectangle[2] >= rectangle[3]
                            ):
                                continue
                            x_center = (rectangle[0] + rectangle[1]) / 2
                            y_center = (rectangle[2] + rectangle[3]) / 2
                            parameters: list[
                                tuple[float, float, float, float, NDArrayFloat, float]
                            ] = []
                            accumulated = 0.0
                            for index, (kernel, power, length) in enumerate(spans):
                                dispersion = float(
                                    _effective_beta2(
                                        kernel.beta2_s2_m,
                                        kernel.beta3_s3_m,
                                        np.array(x_center),
                                        np.array(y_center),
                                        np.array(observation),
                                        kernel.dispersion_convention,
                                        kernel.f_reference_hz,
                                    )
                                )
                                coefficient = 4 * np.pi**2 * dispersion
                                waves = np.array([first, second, third])
                                psd_product = float(
                                    np.prod(power[waves] / bandwidth[waves])
                                )
                                launch_cut = float(power[cut])
                                scale = 0.5 if kernel.nsp_convention == "gn" else 1.0
                                positions, gain, rate = profiles[index]
                                attenuation = scale * (
                                    np.sum(kernel.alpha_npm[waves] + rate[waves])
                                    - kernel.alpha_npm[cut]
                                    - rate[cut]
                                )
                                residual = (
                                    np.prod(gain[:, waves], axis=1) / gain[:, cut]
                                ) ** scale * np.exp(
                                    scale
                                    * (np.sum(rate[waves]) - rate[cut])
                                    * positions
                                )
                                amplitude = (
                                    kernel.gamma_w_m[cut]
                                    * np.sqrt(
                                        psd_product
                                        * receiver_power_w[cut]
                                        * penalties[index]
                                        / launch_cut
                                    )
                                    if launch_cut > 0
                                    else 0.0
                                )
                                parameters.append(
                                    (
                                        coefficient,
                                        accumulated,
                                        float(attenuation),
                                        length / positions.size,
                                        residual,
                                        float(amplitude),
                                    )
                                )
                                accumulated += coefficient * length
                            for piece in _quadrant_pieces(
                                *[np.array(v) for v in rectangle]
                            ):
                                a, b, c, d = (float(v) for v in piece)
                                if b <= a or d <= c:
                                    continue
                                low, high = a * c, b * d
                                width = high - low
                                if width <= 0:
                                    continue

                                def integrand(
                                    unit: float,
                                    low: float = low,
                                    width: float = width,
                                    a: float = a,
                                    b: float = b,
                                    c: float = c,
                                    d: float = d,
                                    parameters: list[
                                        tuple[
                                            float,
                                            float,
                                            float,
                                            float,
                                            NDArrayFloat,
                                            float,
                                        ]
                                    ] = parameters,
                                ) -> float:
                                    xi = low + width * unit
                                    eta_low = max(a, xi / d)
                                    eta_high = min(b, xi / c) if c > 0 else b
                                    if eta_high <= eta_low or eta_low <= 0:
                                        return 0.0
                                    response = 0j
                                    for (
                                        coefficient,
                                        phase_start,
                                        attenuation,
                                        dz,
                                        residual,
                                        amplitude,
                                    ) in parameters:
                                        rate_complex = complex(
                                            attenuation, -coefficient * xi
                                        )
                                        segment = _exponential_integral(
                                            rate_complex, dz
                                        )
                                        step = np.exp(-rate_complex * dz)
                                        stimulus = 0j
                                        for value in residual[::-1]:
                                            stimulus = stimulus * step + value
                                        response += (
                                            amplitude
                                            * segment
                                            * stimulus
                                            * np.exp(1j * phase_start * xi)
                                        )
                                    return float(
                                        abs(response) ** 2
                                        * np.log(eta_high / eta_low)
                                        * width
                                    )

                                points = [
                                    (corner - low) / width
                                    for corner in (a * d, b * c)
                                    if low < corner < high
                                ]
                                integral, _ = quad(
                                    integrand,
                                    0.0,
                                    1.0,
                                    points=sorted(set(points)),
                                    epsabs=1e-30,
                                    epsrel=1e-5,
                                    limit=max(
                                        128, 16 * first_kernel.correlation_points
                                    ),
                                )
                                contribution = (
                                    (16 / 27) * symmetry * band_weight * integral
                                )
                                if first == second == third == cut:
                                    sci[cut] += contribution
                                else:
                                    other[cut] += contribution
    return sci, other
