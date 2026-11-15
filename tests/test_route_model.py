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
from network_nlo_eval.models.route import (
    H_PLANCK,
    _ContextualSpanModel,
    accumulate_route,
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig


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
