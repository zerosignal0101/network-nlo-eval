"""Coherent ECZ regression and power-convention checks."""

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    SpanContext,
    available_models,
    create_model,
)
from network_nlo_eval.models.ecz import (
    ECZCoherentModel,
    _coherent_factor,
    _equivalent_phase_ramp,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


def test_multispan_coherent_path_matches_reference() -> None:
    fiber = FiberSpanConfig(length_km=240, max_span_length_km=80)
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    model = create_model("ecz_coherent", grid, fiber, 32e9, ode_accel="off")
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=5.5)])
    result = model.evaluate_path(path, np.array([1e-3, 2e-3, 0.5e-3]))
    expected = {
        "signal_power_w": [0.001, 0.002, 0.0005],
        "spm_noise_w": [
            1.016577851441297e-06,
            8.146587255565953e-06,
            1.275090341231375e-07,
        ],
        "xpm_noise_w": [
            1.1596962536243704e-06,
            7.030760486261723e-07,
            6.357204697171221e-07,
        ],
        "ase_noise_w": [
            1.6908657516241165e-06,
            1.6914836543830825e-06,
            1.6921016690152674e-06,
        ],
        "snr_db": [24.126101212058867, 22.781421276072198, 23.088599266779774],
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(getattr(result, name), reference, rtol=1e-12)
    assert isinstance(model, ECZCoherentModel)
    assert "ecz_coherent" in available_models()


def test_single_span_matches_baseline_and_amplitudes_follow_power_convention() -> None:
    fiber = FiberSpanConfig(length_km=80)
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    power = np.array([1e-3, 2e-3, 0.5e-3])
    baseline = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off")
    total = ECZCoherentModel(grid, fiber, 32e9, ode_accel="off")
    per_pol = ECZCoherentModel(
        grid, fiber, 32e9, ode_accel="off", launch_power_convention="per_pol"
    )
    single = total.span_fiber_output(power, 80e3)
    ref = baseline.span_fiber_output(power, 80e3)
    for name in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        np.testing.assert_array_equal(getattr(single, name), getattr(ref, name))
    context = SpanContext(1, 3, 80e3, 80e3)
    a = total.span_fiber_output(power, 80e3, span_context=context)
    b = per_pol.span_fiber_output(power / 2, 80e3, span_context=context)
    for kind in ("spm", "xpm"):
        amp_a = getattr(a, f"nli_amplitude_{kind}")
        amp_b = getattr(b, f"nli_amplitude_{kind}")
        np.testing.assert_allclose(
            abs(amp_a) ** 2, getattr(a, f"{kind}_noise_w"), rtol=1e-14
        )
        np.testing.assert_allclose(amp_a, np.sqrt(2) * amp_b, rtol=1e-14)
    child = per_pol.for_fiber(fiber)
    assert child.launch_power_convention == "per_pol"
    assert child is per_pol.for_fiber(fiber)


def test_phase_ramp_reproduces_coherence_limits() -> None:
    n = 4
    factors = _coherent_factor(n, np.array([[1.0, 0.0]] * (n - 1)))
    np.testing.assert_array_equal(factors, [16, 4])
    ramp = _equivalent_phase_ramp(n, factors)
    reconstructed = np.abs(np.exp(1j * np.arange(n)[:, None] * ramp).sum(axis=0)) ** 2
    np.testing.assert_allclose(reconstructed, factors, rtol=1e-14)
