"""Tests for network-level link-noise assembly."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import ElementOutput
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig
from network_nlo_eval.rwa.link_noise import assemble_link_noise


def test_incoherent_link_uses_full_spans_and_remainder() -> None:
    fiber = FiberSpanConfig(length_km=180)
    model = MultiBandISRSGN(
        SpectrumGrid(3, 50e9, 193.1e12), fiber, 32e9, ode_accel="off"
    )
    power = np.full(3, 1e-3)
    edfa = EDFAConfig()
    cache = assemble_link_noise(
        evaluator=model,
        edfa_config=edfa,
        power_profile=power,
        span_length_m=80e3,
        num_full_spans=2,
        remainder_length_m=20e3,
        num_channels=3,
        coherent=True,
    )
    full = model._calc_span_noise_and_power(power, 80e3, edfa)
    remainder = model._calc_span_noise_and_power(power, 20e3, edfa)
    for actual, full_value, remaining in zip(
        (cache.spm, cache.xpm, cache.ase), full[1:], remainder[1:], strict=True
    ):
        np.testing.assert_array_equal(actual, full_value * 2 + remaining)
    assert cache.span_count == 3
    assert cache.launch_power_convention == "total"


class CoherentSpanEvaluator:
    """Emit fixed coherent amplitudes with a per-span power transfer of one half."""

    supports_span_context = True
    fwm_noise_is_additive = True
    launch_power_convention = "per_pol"

    def _calc_span_output_with_context(self, power, _length, _edfa, context):
        assert context.total_spans == 2
        amplitude = np.full(2, 1e-3, dtype=complex)
        return ElementOutput(
            power * 0.5,
            np.full(2, 1e-6),
            np.full(2, 1e-6),
            np.full(2, 2e-6),
            amplitude,
            amplitude,
            np.full(2, 2e-7),
        )


def test_coherent_link_accumulates_amplitudes_and_additive_fwm() -> None:
    cache = assemble_link_noise(
        evaluator=CoherentSpanEvaluator(),
        edfa_config=None,
        power_profile=np.array([1e-3, 2e-3]),
        span_length_m=80e3,
        num_full_spans=1,
        remainder_length_m=40e3,
        num_channels=2,
        coherent=True,
    )
    expected = (1 + np.sqrt(0.5)) ** 2 * 1e-6
    np.testing.assert_allclose(cache.spm, [expected, expected])
    np.testing.assert_allclose(cache.xpm, [expected + 3e-7, expected + 3e-7])
    np.testing.assert_allclose(cache.ase, [3e-6, 3e-6])
    assert cache.span_count == 2
    assert cache.launch_power_convention == "per_pol"


def test_remainder_threshold_and_empty_arrays() -> None:
    fiber = FiberSpanConfig(length_km=80)
    model = MultiBandISRSGN(SpectrumGrid(2, 50e9, 193.1e12), fiber)
    cache = assemble_link_noise(
        evaluator=model,
        edfa_config=None,
        power_profile=np.ones(2),
        span_length_m=80e3,
        num_full_spans=0,
        remainder_length_m=1,
        num_channels=2,
    )
    assert cache.span_count == 0
    assert not np.shares_memory(cache.spm, cache.xpm)
    assert not np.shares_memory(cache.xpm, cache.ase)
    np.testing.assert_array_equal(cache.spm, [0, 0])
    with pytest.raises(ValueError, match="launch_power_convention"):
        assemble_link_noise(
            evaluator=model,
            edfa_config=None,
            power_profile=np.ones(2),
            span_length_m=80e3,
            num_full_spans=0,
            remainder_length_m=0,
            num_channels=2,
            launch_power_convention="invalid",
        )
