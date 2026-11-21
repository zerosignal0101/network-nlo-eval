"""Regression tests for optical route accumulation and amplifier effects."""

from typing import cast

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    MultiSpanOpticalPath,
    SpanContext,
    SpanFiberOutput,
    create_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.route import (
    H_PLANCK,
    _ContextualSpanModel,
    accumulate_route,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)
from network_nlo_eval.network.elements import (
    EDFACompressionPoint,
    EDFAConfig,
    FiberSpanConfig,
    ROADMConfig,
    ROADMOutputTarget,
)


class TwoSpanModel:
    """A simple span model with optional coherent amplitudes and additive FWM."""

    n_nodes = 2
    f_abs_hz = np.array([193.1e12, 193.15e12])
    channel_bandwidth_hz_channels = np.array([50e9, 50e9])
    fwm_noise_is_additive = True

    def __init__(self, coherent: bool = False) -> None:
        self.coherent = coherent

    def power_width_scale(self) -> NDArrayFloat:
        return np.ones(2)

    def blocked_nodes(self, roadm: ROADMConfig) -> list[int]:
        return roadm.blocked_channels

    def span_step(
        self,
        _fiber: FiberSpanConfig,
        _edfa: EDFAConfig | None,
        power: NDArrayFloat,
        _length: float,
    ) -> ElementOutput:
        amplitude = np.full(2, 1e-3, dtype=complex) if self.coherent else None
        return ElementOutput(
            power_out_w=power * 0.5,
            spm_noise_w=np.full(2, 1e-6),
            xpm_noise_w=np.full(2, 1e-6),
            ase_noise_w=np.full(2, 2e-6),
            nli_amplitude_spm=amplitude,
            nli_amplitude_xpm=amplitude,
            fwm_noise_w=np.full(2, 2e-7),
        )


def test_incoherent_span_accumulation_and_additive_fwm() -> None:
    path = MultiSpanOpticalPath(
        [FiberSpanConfig(length_km=100, max_span_length_km=50)], [EDFAConfig()]
    )
    result = accumulate_route(TwoSpanModel(), path, np.array([0.001, 0.002]))
    np.testing.assert_allclose(result.signal_power_w, [0.00025, 0.0005])
    np.testing.assert_allclose(result.spm_noise_w, [1.5e-6] * 2)
    np.testing.assert_allclose(result.xpm_noise_w, [1.8e-6] * 2)
    np.testing.assert_allclose(result.ase_noise_w, [3e-6] * 2)
    np.testing.assert_allclose(
        result.snr_db, 10 * np.log10(result.signal_power_w / 6.3e-6)
    )
    assert result.span_count == 2
    assert result.roadm_count == 0


def test_coherent_accumulation_and_roadm_blocking() -> None:
    roadm = ROADMConfig(
        insertion_loss_db=0,
        add_drop_loss_db=0,
        filtering_penalty_db=0,
        blocked_channels=[1],
    )
    path = MultiSpanOpticalPath(
        [FiberSpanConfig(length_km=100, max_span_length_km=50)], [EDFAConfig()], [roadm]
    )
    result = accumulate_route(
        TwoSpanModel(coherent=True), path, np.array([0.001, 0.002])
    )
    expected_coherent = (1 + np.sqrt(0.5)) ** 2 * 1e-6
    assert result.spm_noise_w[0] == pytest.approx(expected_coherent)
    assert result.xpm_noise_w[0] == pytest.approx(expected_coherent + 3e-7)
    assert result.signal_power_w[1] == 0
    assert result.snr_db[1] == -np.inf
    assert result.roadm_count == 1


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_passive_roadm_psd_target_scales_signal_ase_and_amplitudes(
    scale: float,
) -> None:
    model = TwoSpanModel(coherent=True)
    model.channel_bandwidth_hz_channels = np.array([25e9, 75e9])
    fiber = FiberSpanConfig(length_km=50)
    roadm = ROADMConfig(
        insertion_loss_db=10 * np.log10(2),
        filtering_penalty_db=0,
        output_target=ROADMOutputTarget(psd_w_per_hz=1e-14),
    )
    result = accumulate_route(
        model,
        MultiSpanOpticalPath([fiber, fiber], [None, None], [roadm, None]),
        np.array([8e-3, 2e-3]) * scale,
        ase_polarization_scale=scale,
    )
    transfer_to_last_span = np.array([0.0625, 0.5])
    np.testing.assert_allclose(
        result.signal_power_w, np.array([0.125e-3, 0.25e-3]) * scale
    )
    np.testing.assert_allclose(
        result.ase_noise_w, 2e-6 * (1 + 0.5 * transfer_to_last_span)
    )
    np.testing.assert_allclose(
        result.spm_noise_w, 1e-6 * (1 + np.sqrt(0.5 * transfer_to_last_span)) ** 2
    )


def test_amplifier_ase_scaling_does_not_rescale_input_ase() -> None:
    power = np.array([1e-3])
    fiber = SpanFiberOutput(np.array([1e-4]), np.array([1e-6]), np.array([2e-6]))
    edfa = EDFAConfig(noise_figure_db=5)
    nf = resolve_noise_figure(edfa, np.ones(1), 1)
    freq, bandwidth = np.array([193.1e12]), np.array([50e9])
    result = apply_amplifier_and_roadm(
        power_in_w=power,
        fiber_output=fiber,
        edfa_config=edfa,
        roadm_config=None,
        nf_lin=nf,
        f_abs_hz=freq,
        bandwidth_hz=bandwidth,
        input_ase_w=np.array([1e-8]),
        ase_polarization_scale=0.5,
    )
    np.testing.assert_allclose(result.power_out_w, power)
    expected_ase = nf * H_PLANCK * freq * bandwidth * 9 * 0.5 + 1e-7
    np.testing.assert_allclose(result.ase_noise_w, expected_ase)
    np.testing.assert_allclose(result.spm_noise_w, [1e-6])
    with pytest.raises(ValueError, match="finite and positive"):
        accumulate_route(
            TwoSpanModel(),
            MultiSpanOpticalPath([FiberSpanConfig(length_km=1)], [edfa]),
            power,
            ase_polarization_scale=0,
        )


@pytest.mark.parametrize("polarization_scale", [1.0, 0.5])
def test_edfa_output_loading_reduces_gain_and_ase(polarization_scale: float) -> None:
    input_power = np.array([1e-3, 2e-3, 0.0]) * polarization_scale
    freq = np.array([193.1e12, 193.15e12, 193.2e12])
    bandwidth = np.full(3, 32e9)
    nf = np.full(3, 10**0.5)
    edfa = EDFAConfig(max_output_signal_power_w=1.5e-3)
    result = apply_amplifier_and_roadm(
        power_in_w=input_power,
        fiber_output=SpanFiberOutput(
            input_power / 100, input_power / 1000, np.zeros(3)
        ),
        edfa_config=edfa,
        roadm_config=None,
        nf_lin=nf,
        f_abs_hz=freq,
        bandwidth_hz=bandwidth,
        ase_polarization_scale=polarization_scale,
    )
    np.testing.assert_allclose(result.power_out_w, input_power / 2)
    np.testing.assert_allclose(result.spm_noise_w, input_power / 2000)
    expected_ase = nf * H_PLANCK * freq * bandwidth * np.array([49.0, 49.0, 0.0])
    np.testing.assert_allclose(result.ase_noise_w, expected_ase * polarization_scale)


@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_edfa_passbands_and_voas_set_internal_load_and_output_noise(
    scale: float,
) -> None:
    launch = np.array([2e-3, 4e-3, 8e-3, 4e-3]) * scale
    frequency = np.array([193e12, 193.02e12, 194e12, 196e12])
    bandwidth = np.array([50e9, 25e9, 32e9, 50e9])
    input_ase = np.full(4, 1e-9 * scale)
    nf = np.full(4, 10**0.5)
    spm = launch / 1000
    fiber = SpanFiberOutput(
        launch / 100,
        spm,
        spm,
        np.sqrt(spm).astype(complex),
        np.sqrt(spm).astype(complex),
    )
    edfa = EDFAConfig(
        passbands_hz=((192.975e12, 193.025e12), (195.975e12, 196.025e12)),
        input_voa_db=10 * np.log10(2),
        output_voa_db=10 * np.log10(4),
        max_output_signal_power_w=1.5e-3,
    )
    result = apply_amplifier_and_roadm(
        power_in_w=launch,
        fiber_output=fiber,
        edfa_config=edfa,
        roadm_config=None,
        nf_lin=nf,
        f_abs_hz=frequency,
        bandwidth_hz=bandwidth,
        input_ase_w=input_ase,
        ase_polarization_scale=scale,
    )
    transfer = np.array([0.0625, 0, 0, 0.0625])
    np.testing.assert_allclose(result.power_out_w, launch * transfer, atol=0)
    assert result.power_out_w.sum() == pytest.approx(1.5e-3 * scale / 4)
    np.testing.assert_allclose(result.spm_noise_w, spm * transfer, atol=0)
    assert result.nli_amplitude_spm is not None
    np.testing.assert_allclose(
        abs(result.nli_amplitude_spm) ** 2, spm * transfer, atol=0
    )
    expected_ase = (
        nf * H_PLANCK * frequency * bandwidth * 49 * scale + 25 * input_ase
    ) / 4
    expected_ase[[1, 2]] = 0
    np.testing.assert_allclose(result.ase_noise_w, expected_ase, atol=0)


@pytest.mark.parametrize(
    ("bands", "reason"),
    [
        (((193e12, np.inf),), "finite, positive, increasing"),
        (((194e12, 193e12),), "finite, positive, increasing"),
        (((195e12, 197e12), (193e12, 196e12)), "must not overlap"),
    ],
)
def test_edfa_rejects_invalid_passbands(
    bands: tuple[tuple[float, float], ...], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        EDFAConfig(passbands_hz=bands)


@pytest.mark.parametrize("convention", ["total", "per_pol"])
def test_edfa_compression_curve_in_ordered_cascade(convention: str) -> None:
    fiber = FiberSpanConfig(length_km=80)
    model = MultiBandISRSGN(
        SpectrumGrid(2, 1e12, 193.1e12), fiber, 32e9, launch_power_convention=convention
    )
    backoff_db = 10 * np.log10(2.0)
    curve = (
        EDFACompressionPoint(backoff_db=0),
        EDFACompressionPoint(
            backoff_db=2 * backoff_db,
            noise_figure_penalty_db=4,
            tilt_db_per_thz=2 * backoff_db,
        ),
    )
    scale = model.ase_polarization_scale
    launch = np.array([1e-3, 2e-3]) * scale
    nf_db = np.array([5.0, 6.0])
    loaded = EDFAConfig(
        max_output_signal_power_w=1.5e-3, compression_curve=curve, noise_figure_db=nf_db
    )
    second = EDFAConfig(
        max_output_signal_power_w=0.75e-3,
        compression_curve=curve,
        noise_figure_db=nf_db,
    )
    first_output = np.array([1, 4]) * (1.5e-3 * scale / 5)
    final_output = np.array([1, 8]) * (0.75e-3 * scale / 9)
    gain_first = first_output / model.span_fiber_output(launch, 80e3).power_out_w
    gain_second = final_output / model.span_fiber_output(first_output, 80e3).power_out_w
    ase_factor = (
        10 ** ((nf_db + 2) / 10)
        * H_PLANCK
        * model.f_abs_hz
        * model.channel_bandwidth_hz_channels
        * scale
    )
    expected_ase = ase_factor * (
        (gain_first - 1) * final_output / first_output + gain_second - 1
    )
    result = model.evaluate_path(
        MultiSpanOpticalPath([fiber, fiber], [loaded, second]), launch
    )
    np.testing.assert_allclose(result.signal_power_w, final_output, rtol=1e-12)
    np.testing.assert_allclose(result.ase_noise_w, expected_ase, rtol=1e-12)


def test_edfa_compression_curve_requires_cap_and_ordered_backoff() -> None:
    origin = EDFACompressionPoint(backoff_db=0)
    endpoint = EDFACompressionPoint(backoff_db=1)
    with pytest.raises(ValueError, match="requires max_output_signal_power_w"):
        EDFAConfig(compression_curve=(origin, endpoint))
    with pytest.raises(ValueError, match="start at zero backoff"):
        EDFAConfig(max_output_signal_power_w=1e-3, compression_curve=(endpoint, origin))
    with pytest.raises(ValueError, match="values must increase strictly"):
        EDFAConfig(max_output_signal_power_w=1e-3, compression_curve=(origin, origin))


@pytest.mark.parametrize(
    ("edfa", "expected_transfer"),
    [
        (None, np.array([0.1, 0.2])),
        (EDFAConfig(), np.ones(2)),
        (EDFAConfig(target_gain_db=10), np.array([1.0, 2.0])),
        (
            EDFAConfig(gain_ripple_db=np.array([3.0, -3.0])),
            10 ** (np.array([3.0, -3.0]) / 10),
        ),
    ],
)
def test_nli_output_transfer_preserves_noise_to_signal_ratio_and_amplitudes(
    edfa: EDFAConfig | None, expected_transfer: NDArrayFloat
) -> None:
    power = np.array([1e-3, 2e-3])
    spm, xpm, fwm = power * 0.001, power * 0.002, power * 0.0001
    fiber = SpanFiberOutput(
        power * [0.1, 0.2],
        spm.copy(),
        xpm.copy(),
        np.sqrt(spm).astype(complex),
        1j * np.sqrt(xpm),
        fwm.copy(),
    )
    result = apply_amplifier_and_roadm(
        power_in_w=power,
        fiber_output=fiber,
        edfa_config=edfa,
        roadm_config=None,
        nf_lin=np.ones(2),
        f_abs_hz=np.full(2, 193.1e12),
        bandwidth_hz=np.full(2, 32e9),
        fwm_is_additive=True,
    )
    np.testing.assert_allclose(result.spm_noise_w, spm * expected_transfer)
    np.testing.assert_allclose(result.xpm_noise_w, xpm * expected_transfer)
    assert result.fwm_noise_w is not None
    np.testing.assert_allclose(result.fwm_noise_w, fwm * expected_transfer)
    assert result.nli_amplitude_spm is not None
    assert result.nli_amplitude_xpm is not None
    np.testing.assert_allclose(abs(result.nli_amplitude_spm) ** 2, result.spm_noise_w)
    np.testing.assert_allclose(abs(result.nli_amplitude_xpm) ** 2, result.xpm_noise_w)
    np.testing.assert_allclose(result.spm_noise_w / result.power_out_w, spm / power)
    np.testing.assert_array_equal(fiber.spm_noise_w, spm)


@pytest.mark.parametrize("model_name", ["carena2014", "damico2022", "numerical"])
def test_model_amplitudes_share_the_output_noise_reference(model_name: str) -> None:
    fiber = FiberSpanConfig(length_km=80)
    model = cast(
        "_ContextualSpanModel",
        create_model(model_name, SpectrumGrid(2, 50e9, 193.1e12), fiber, 32e9),
    )
    power = np.full(2, 1e-3)
    output = model.span_step(
        fiber, EDFAConfig(target_gain_db=10), power, 80e3, SpanContext(0, 2, 0, 80e3)
    )
    assert output.nli_amplitude_spm is not None
    assert output.nli_amplitude_xpm is not None
    np.testing.assert_allclose(abs(output.nli_amplitude_spm) ** 2, output.spm_noise_w)
    np.testing.assert_allclose(abs(output.nli_amplitude_xpm) ** 2, output.xpm_noise_w)
