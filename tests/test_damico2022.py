"""D'Amico coherence and cross-channel aggregation."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, SpanContext, create_model
from network_nlo_eval.models.damico2022 import (
    _band_group_masks,
    _dirichlet_factor,
    _equivalent_phase_ramp,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


@pytest.mark.parametrize("n", [2, 4, 10])
@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
def test_phase_ramp_matches_target(n: int, fraction: float) -> None:
    target = n + fraction * (n * n - n)
    phase = _equivalent_phase_ramp(n, target)
    assert _dirichlet_factor(n, phase) == pytest.approx(target, rel=1e-12)


def test_band_masks_cover_outside_channels_once() -> None:
    frequencies = np.array([180e12, 185e12, 193e12, 221e12])
    masks = _band_group_masks(
        frequencies, ((184.4e12, 191.6e12), (191.6e12, 195.9e12)), None
    )
    assert all(mask.dtype == np.bool_ for mask in masks)
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(4))
    np.testing.assert_array_equal(masks[-1], [True, False, False, True])


def test_exact_aggregation_and_multispan_amplitudes() -> None:
    grid = SpectrumGrid(4, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    power = np.full(4, 1e-3)
    models = [create_model("damico2022", grid, fiber, 32e9, n_bands=n) for n in (1, 3)]
    kernels = [model.kernel_for(fiber) for model in models]
    parts = [kernel.evaluate_components(power, 20e3, 3) for kernel in kernels]
    np.testing.assert_array_equal(parts[0]["xc_w"], parts[1]["xc_w"])
    path = MultiSpanOpticalPath([fiber] * 3, [EDFAConfig()] * 3)
    result = models[0].evaluate_path(path, power)
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(result.spm_noise_w > 0)
    assert np.all(result.xpm_noise_w > 0)
    context = SpanContext(
        span_index=1, total_spans=3, cumulative_length_m=20e3, span_length_m=20e3
    )
    output = kernels[0].evaluate(power, 20e3, span_context=context)
    np.testing.assert_allclose(np.abs(output.nli_amplitude_spm) ** 2, parts[0]["sc_w"])
    np.testing.assert_allclose(np.abs(output.nli_amplitude_xpm) ** 2, parts[0]["xc_w"])
