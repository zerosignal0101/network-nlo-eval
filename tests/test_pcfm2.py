"""Independent GN integrals and route invariants for coherent PCFM2."""

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from scipy.special import roots_legendre, sici

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.pcfm2 import PCFM2Model, PCFM2SpanKernel
from network_nlo_eval.models.pcfm2_kernels import polynomial_pair, rectangle_response
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
    ROADMConfig,
)

if TYPE_CHECKING:
    from network_nlo_eval.models.pcfm_auto import PCFMAutoModel


@pytest.mark.parametrize(
    ("dv", "dw", "separation"),
    [
        (0, 0, 0),
        (-2e-21, -2e-21, 0),
        (2e-21, -3e-21, 7e-21),
        (0, 2e-21, -1e-21),
        (2e-21, 0, 3e-21),
        (1e-23, 2e-21, 1e-20),
        (1e-50, -1e-50, 0),
        (1e-30, 2e-21, 0),
        (1e-50, 2e-21, 0),
    ],
)
def test_polynomial_pair_against_independent_quadrature(dv, dw, separation) -> None:
    """Check the signed overlap polynomial and all zero-dispersion limits."""
    nodes, weights = roots_legendre(180)
    nodes, weights = (nodes + 1) / 2, weights / 2
    rectangle = (-20e9, 30e9, -10e9, 45e9)
    a, b, c, d = rectangle
    phase = 4 * np.pi**2 * (separation - dv * nodes[:, None] + dw * nodes[None, :])
    integral = sum(
        sign * sici(corner * phase)[0]
        for sign, corner in zip(
            [-1, 1, -1, 1], [a * d, a * c, b * c, b * d], strict=True
        )
    )
    response = np.divide(
        integral, phase, out=np.full_like(phase, (b - a) * (d - c)), where=phase != 0
    )
    first, second = (1.0, -0.8, 0.1, -0.03, 0.02, -0.001), (0.7, -0.2, 0.05)
    expected = np.sum(
        response
        * np.polynomial.polynomial.polyval(nodes, first)[:, None]  # type: ignore[no-untyped-call]
        * np.polynomial.polynomial.polyval(nodes, second)[None, :]  # type: ignore[no-untyped-call]
        * weights[:, None]
        * weights[None, :]
    )
    actual = polynomial_pair(rectangle, first, second, dv, dw, separation)
    assert actual == pytest.approx(expected, rel=2e-11)


@pytest.mark.parametrize("phase", [0.0, -2e-21, 2e-21, 1e-50])
def test_rectangle_against_frequency_quadrature(phase) -> None:
    nodes, weights = roots_legendre(150)
    a, b, c, d = (-10e9, 32e9, -25e9, 19e9)
    x, y = (a + b) / 2 + (b - a) * nodes / 2, (c + d) / 2 + (d - c) * nodes / 2
    expected = (
        (b - a)
        * (d - c)
        / 4
        * np.sum(
            np.cos(4 * np.pi**2 * phase * x[:, None] * y[None, :])
            * weights[:, None]
            * weights[None, :]
        )
    )
    assert rectangle_response((a, b, c, d), phase) == pytest.approx(expected, rel=1e-12)


def _fiber(dispersion: float = 0, **options) -> FiberSpanConfig:
    # S=-2D/lambda removes beta3, allowing an independent constant-beta reference.
    return FiberSpanConfig(
        length_km=100,
        dispersion_parameter_d=dispersion,
        dispersion_slope_s=-2 * dispersion / 1550,
        aeff_slope_um2_nm=0,
        **options,
    )


def _model(fiber: FiberSpanConfig, channels: int = 1, **options) -> PCFM2Model:
    return cast(
        "PCFM2Model",
        create_model(
            "pcfm2",
            SpectrumGrid(channels, 32e9, C_LIGHT / 1550e-9),
            fiber,
            32e9,
            include_isrs=False,
            **options,
        ),
    )


def test_zero_dispersion_sci_psd_and_band_integration() -> None:
    """The actual hexagonal SCI island gives a quadratic PSD, not flat PSD."""
    fiber = _fiber()
    model = _model(fiber, band_points=3)
    kernel = cast("PCFM2SpanKernel", model.kernel_for(fiber))
    bandwidth, power, length = 32e9, np.array([1e-3]), 100e3
    frequencies = model.f_abs_hz[0] + bandwidth * np.array([-0.5, 0, 0.5])
    sci, other = kernel.nli_psd(power, length, 0, frequencies)
    alpha = kernel.alpha_npm[0]
    gamma = 2 * np.pi * model.f_abs_hz[0] / C_LIGHT * fiber.nonlinear_index_n2 / 80e-12
    leff = -np.expm1(-alpha * length) / alpha
    scale = 16 / 27 * (power[0] / bandwidth) ** 3 * gamma**2 * leff**2
    np.testing.assert_allclose(
        sci, scale * bandwidth**2 * np.array([0.5, 0.75, 0.5]), rtol=3e-4
    )
    np.testing.assert_array_equal(other, 0)
    result = kernel.evaluate(power, length)
    assert result.spm_noise_w[0] == pytest.approx(
        scale * bandwidth**3 * 2 / 3, rel=3e-4
    )


def test_subcarrier_mci_and_sparse_channels() -> None:
    """All three-wave combinations contribute in an adjacent eight-channel comb."""
    fiber = _fiber(attenuation_db_km_ref=1e-15)
    model = _model(fiber, channels=8, band_points=1, island_cells=1)
    kernel = cast("PCFM2SpanKernel", model.kernel_for(fiber))
    power = np.full(8, 1e-3)
    center = model.f_abs_hz[0]
    _, other = kernel.nli_psd(power, 100e3, 0, np.array([center]))
    # Independent dense frequency-plane integral, including every active wave.
    bandwidth = 32e9
    edges = np.arange(-0.5, 8.5) * bandwidth
    nodes, weights = roots_legendre(180)
    reference = 0.0
    xci_only = 0.0
    for first in range(8):
        x = (first + nodes / 2) * bandwidth
        for second in range(8):
            y = (second + nodes / 2) * bandwidth
            third = np.searchsorted(edges, x[:, None] + y[None, :], side="right") - 1
            valid = (third >= 0) & (third < 8)
            sci = (first == 0) & (second == 0) & (third == 0)
            area = (
                np.sum(weights[:, None] * weights[None, :] * valid * ~sci)
                * bandwidth**2
                / 4
            )
            reference += area
            if first == 0 or second == 0:
                xci_only += area
    gamma = 2 * np.pi * center / C_LIGHT * fiber.nonlinear_index_n2 / 80e-12
    factor = 16 / 27 * (power[0] / bandwidth) ** 3 * gamma**2 * (100e3) ** 2
    assert other[0] == pytest.approx(reference * factor, rel=0.003)
    assert other[0] > 1.2 * xci_only * factor
    power[1::2] = 0
    result = kernel.evaluate(power, 100e3)
    np.testing.assert_array_equal(result.spm_noise_w[1::2], 0)
    np.testing.assert_array_equal(result.xpm_noise_w[1::2], 0)


def test_coherent_and_incoherent_zero_dispersion_scaling() -> None:
    fiber = _fiber()
    power = np.array([1e-3])
    model = _model(fiber, band_points=1)
    one = model.evaluate_path(MultiSpanOpticalPath([fiber], [EDFAConfig()]), power)
    path = MultiSpanOpticalPath([fiber] * 3, [EDFAConfig()] * 3)
    coherent = model.evaluate_path(path, power)
    incoherent = _model(
        fiber, band_points=1, span_coherence="incoherent"
    ).evaluate_path(path, power)
    np.testing.assert_allclose(coherent.spm_noise_w, 9 * one.spm_noise_w, rtol=1e-12)
    np.testing.assert_allclose(incoherent.spm_noise_w, 3 * one.spm_noise_w, rtol=1e-12)


def test_opposite_dispersion_and_unamplified_segmentation() -> None:
    """Field reversal and launch/receiver transfer survive heterogeneous routes."""
    forward = _fiber(3, attenuation_db_km_ref=1e-15)
    reverse = _fiber(-3, attenuation_db_km_ref=1e-15)
    power = np.array([1e-3])
    model = _model(forward, island_cells=2, band_points=1)
    passive = cast("EDFAConfig", None)  # Shared route supports passive propagation.
    one = model.evaluate_path(MultiSpanOpticalPath([forward], [passive]), power)
    two = model.evaluate_path(
        MultiSpanOpticalPath([forward, reverse], [passive] * 2), power
    )
    np.testing.assert_allclose(two.spm_noise_w, 4 * one.spm_noise_w, rtol=1e-12)
    # Splitting a passive lossy fiber must preserve both signal and the field.
    fiber = _fiber(3)
    half = fiber.model_copy(update={"length_km": 50.0})
    model = _model(fiber, island_cells=2, band_points=1, poly_degree=9, spp_steps=128)
    whole = model.evaluate_path(MultiSpanOpticalPath([fiber], [passive]), power)
    split = model.evaluate_path(MultiSpanOpticalPath([half] * 2, [passive] * 2), power)
    np.testing.assert_allclose(split.signal_power_w, whole.signal_power_w, rtol=1e-7)
    np.testing.assert_allclose(split.spm_noise_w, whole.spm_noise_w, rtol=1e-6)


def test_roadm_filtering_and_blocking() -> None:
    fiber = _fiber()
    model = _model(fiber, band_points=1)
    power = np.array([1e-3])
    one = model.evaluate_path(MultiSpanOpticalPath([fiber], [EDFAConfig()]), power)
    roadm = ROADMConfig(filtering_penalty_db=3, equalize_output_power=True)
    filtered = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [EDFAConfig()], [roadm]), power
    )
    np.testing.assert_allclose(filtered.spm_noise_w, one.spm_noise_w * 10**0.3)
    blocked = roadm.model_copy(update={"blocked_channels": [0]})
    result = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [EDFAConfig()], [blocked]), power
    )
    np.testing.assert_array_equal(result.spm_noise_w, 0)
    assert result.snr_db[0] == -np.inf
    # Span bookkeeping follows the shared route's remainder tolerance.
    almost_two = fiber.model_copy(update={"length_km": np.nextafter(200.0, np.inf)})
    two = model.evaluate_path(
        MultiSpanOpticalPath([almost_two], [EDFAConfig()], [roadm]), power
    )
    assert two.span_count == 2


def test_unequal_launches_and_heterogeneous_receiver_transfer() -> None:
    """Compare against a direct zero-dispersion sum of receiver-referenced fields."""
    first = _fiber().model_copy(update={"length_km": 40.0})
    second = _fiber(attenuation_db_km_ref=0.15).model_copy(
        update={"length_km": 60.0, "effective_area_um2_ref": 100.0}
    )
    model = _model(first, poly_degree=7, band_points=3)
    input_power = np.array([1e-3])
    path = MultiSpanOpticalPath(
        [first, second],
        [EDFAConfig(target_gain_db=3.0), EDFAConfig(target_gain_db=8.0)],
    )
    result = model.evaluate_path(path, input_power)
    alpha_first, alpha_second = 0.2 * np.log(10) / 1e4, 0.15 * np.log(10) / 1e4
    launches = [
        input_power[0],
        input_power[0] * np.exp(-alpha_first * 40e3) * 10**0.3,
    ]
    receiver = launches[1] * np.exp(-alpha_second * 60e3) * 10**0.8
    np.testing.assert_allclose(result.signal_power_w, receiver)
    field = 0.0
    for fiber, alpha, launch in zip(
        [first, second], [alpha_first, alpha_second], launches, strict=True
    ):
        effective_length = -np.expm1(-alpha * fiber.length_km * 1000) / alpha
        gamma = (
            2
            * np.pi
            * model.f_abs_hz[0]
            / C_LIGHT
            * fiber.nonlinear_index_n2
            / (fiber.effective_area_um2_ref * 1e-12)
        )
        field += gamma * effective_length * launch * np.sqrt(receiver)
    expected = 16 / 27 * 2 / 3 * field**2
    assert result.spm_noise_w[0] == pytest.approx(expected, rel=1e-6)


def test_flexible_grid_uses_per_connection_bandwidth_and_power_scaling() -> None:
    grid = FlexibleGrid(12.5e9, 16, 193e12)
    connections = [
        ConnectionSpec(grid, index, slot, slots, rate * 2, "QPSK", rate, 0.0, 1e-3)
        for index, slot, slots, rate in [(0, 0, 2, 25e9), (1, 8, 4, 50e9)]
    ]
    fiber = _fiber()
    options = {
        "connections": connections,
        "channel_bandwidth_source": "occupied",
        "include_isrs": False,
        "band_points": 3,
        "island_cells": 1,
    }
    model = create_model(
        "pcfm2", grid, fiber, 25e9, power_width_mode="fixed_psd", **options
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    scaled = model.evaluate_path(path, np.full(2, 1e-3))
    plain = create_model(
        "pcfm2", grid, fiber, 25e9, power_width_mode="fixed_power", **options
    )
    actual = plain.evaluate_path(path, np.array([1e-3, 2e-3]))
    np.testing.assert_allclose(scaled.signal_power_w, actual.signal_power_w)
    np.testing.assert_allclose(scaled.spm_noise_w, actual.spm_noise_w)
    np.testing.assert_allclose(scaled.xpm_noise_w, actual.xpm_noise_w)


def test_validation_and_registry_contracts() -> None:
    fiber = _fiber()
    cases: list[tuple[dict[str, Any], str]] = [
        ({"egn": True}, "GN only"),
        ({"poly_degree": -1}, "poly_degree"),
        ({"poly_degree": 2.5}, "poly_degree"),
        ({"band_points": 0}, "band_points"),
        ({"span_coherence": "auto"}, "span_coherence"),
        ({"spp_steps": 1}, "profile samples"),
        ({"raman_backend": "bad"}, "raman_backend"),
    ]
    for options, message in cases:
        with pytest.raises(ValueError, match=message):
            _model(fiber, **options)
    with pytest.raises(ValueError, match="lumped"):
        _model(fiber.model_copy(update={"lumped_loss": [(50000.0, 1.0)]}))
    kernel = cast("PCFM2SpanKernel", _model(fiber).kernel_for(fiber))
    with pytest.raises(ValueError, match="within the CUT"):
        kernel.nli_psd(np.array([1e-3]), 100e3, 0, np.array([190e12]))
    with pytest.raises(ValueError, match="finite nonnegative"):
        kernel.evaluate(np.array([np.nan]), 100e3)


def test_single_span_against_independent_full_island_gn() -> None:
    """Resolve the true SCI strip and exponential link without polynomial fitting."""
    fiber = _fiber(4)
    model = _model(fiber, island_cells=16, band_points=8, poly_degree=7)
    kernel = cast("PCFM2SpanKernel", model.kernel_for(fiber))
    bandwidth, length, power = 32e9, 100e3, np.array([1e-3])
    nodes, weights = roots_legendre(128)
    band_nodes, band_weights = roots_legendre(16)
    reference = 0.0
    for node, weight in zip(band_nodes, band_weights, strict=True):
        observation = model.f_abs_hz[0] + bandwidth / 2 * node
        low = model.f_abs_hz[0] - observation - bandwidth / 2
        high = low + bandwidth
        for left, right in [(low, 0), (0, high)]:
            x = (left + right) / 2 + (right - left) / 2 * nodes
            ylow, yhigh = np.maximum(low, low - x), np.minimum(high, high - x)
            y = (ylow + yhigh)[:, None] / 2 + (yhigh - ylow)[:, None] / 2 * nodes
            rate = kernel.alpha_npm[0] - 4j * np.pi**2 * kernel.beta2 * x[:, None] * y
            response = -np.expm1(-rate * length) / rate
            gamma = (
                2 * np.pi * observation / C_LIGHT * fiber.nonlinear_index_n2 / 80e-12
            )
            reference += (
                (16 / 27)
                * (power[0] / bandwidth) ** 3
                * gamma**2
                * (bandwidth / 2 * weight)
                * (right - left)
                / 4
                * np.sum(
                    abs(response) ** 2
                    * (yhigh - ylow)[:, None]
                    * weights[:, None]
                    * weights[None, :]
                )
            )
    actual = kernel.evaluate(power, length)
    assert actual.spm_noise_w[0] == pytest.approx(reference, rel=0.005)


def test_raman_signal_and_ase_share_existing_route_conventions() -> None:
    fiber = _fiber(17, nonlinear_index_n2=0).model_copy(
        update={
            "length_km": 20.0,
            "raman_pumps": [
                RamanPumpConfig(wavelength_nm=1450, power_w=0.05, direction="backward")
            ],
        }
    )
    grid = SpectrumGrid(1, 32e9, C_LIGHT / 1550e-9)
    options = {
        "include_isrs": False,
        "raman_iterations": 64,
        "raman_backend": "python",
        "egn": False,
    }
    pcfm2 = create_model("pcfm2", grid, fiber, 32e9, spp_steps=64, **options)
    h3 = create_model(
        "pcfm_h3_raman", grid, fiber, 32e9, spp_steps=32, raman_steps=64, **options
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    actual = pcfm2.evaluate_path(path, np.array([1e-3]))
    existing = h3.evaluate_path(path, np.array([1e-3]))
    np.testing.assert_allclose(
        actual.signal_power_w, existing.signal_power_w, rtol=1e-10
    )
    np.testing.assert_allclose(actual.ase_noise_w, existing.ase_noise_w, rtol=1e-10)
    np.testing.assert_array_equal(actual.spm_noise_w, 0)


def test_opt_in_selector_records_both_branches_and_refuses_low_dispersion_fallback() -> (
    None
):
    fiber = _fiber(17)
    grid = SpectrumGrid(1, 32e9, C_LIGHT / 1550e-9)
    model = cast(
        "PCFMAutoModel",
        create_model(
            "pcfm_auto",
            grid,
            fiber,
            32e9,
            max_pcfm2_work=1,
            pcfm2_options={"island_cells": 1, "band_points": 1, "include_isrs": False},
            h3_options={"include_isrs": False},
        ),
    )
    power = np.array([1e-3])
    model.evaluate_path(MultiSpanOpticalPath([fiber], [EDFAConfig()]), power)
    assert model.last_selection is not None
    assert model.last_selection.model == "pcfm2"
    assert model.last_selection.estimated_work == 1
    model.evaluate_path(MultiSpanOpticalPath([fiber] * 2, [EDFAConfig()] * 2), power)
    assert model.last_selection.model == "pcfm_h3_raman"
    zero = _fiber()
    with pytest.raises(ValueError, match="exceeds budget"):
        model.evaluate_path(MultiSpanOpticalPath([zero] * 2, [EDFAConfig()] * 2), power)
