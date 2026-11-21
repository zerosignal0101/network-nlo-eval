"""Waveform validity policy integration without an optional solver."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator, apply_link_validity
from network_nlo_eval.rwa.validity import LinkValidity, LinkValidityPolicy


def test_validity_overlay_preserves_input_and_applies_policy() -> None:
    report = LinkValidity(
        valid=np.array([True, False]),
        active=np.array([True, True]),
        flagged=np.array([False, True]),
        confidence=np.array([1.0, 0.5]),
        margin_db=np.array([2.0, np.inf]),
        soft_margin_db=np.array([0.2, 0.4]),
        model="test",
    )
    snr = np.array([20.0, 15.0])
    metrics = {"snr": snr}
    off = apply_link_validity(metrics, report, LinkValidityPolicy())
    assert off["snr"] is snr
    adjusted = apply_link_validity(metrics, report, LinkValidityPolicy(mode="degrade"))
    np.testing.assert_array_equal(adjusted["snr"], [18.0, -np.inf])
    np.testing.assert_array_equal(snr, [20.0, 15.0])
    np.testing.assert_array_equal(adjusted["valid"], [True, False])
    with pytest.raises(ValueError, match="shape"):
        apply_link_validity({"snr": np.ones(3)}, report, LinkValidityPolicy())


@pytest.mark.parametrize(
    "edfa",
    [
        EDFAConfig(passbands_hz=((192e12, 194e12),)),
        EDFAConfig(input_voa_db=1),
        EDFAConfig(output_voa_db=1),
    ],
)
def test_waveform_rejects_edfa_passbands_and_voas(edfa: EDFAConfig) -> None:
    with pytest.raises(ValueError, match="without passbands or VOA losses"):
        PyNLOQoTEvaluator._edfa_values(edfa)


def test_gpu_waveform_path_and_roadm() -> None:
    """Propagate a waveform through a fiber and ROADM when CUDA is available."""
    pytest.importorskip("physics_nlo_eval")
    cupy = pytest.importorskip("cupy")
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except cupy.cuda.runtime.CUDARuntimeError:
        pytest.skip("CUDA is unavailable")
    if device_count == 0:
        pytest.skip("No CUDA device is available")

    grid = SpectrumGrid(3, 2e9, 193.1e12)
    fiber = FiberSpanConfig(
        length_km=0.01,
        max_span_length_km=0.01,
        dispersion_parameter_d=0,
        dispersion_slope_s=0,
        nonlinear_index_n2=0,
    )
    path = MultiSpanOpticalPath(
        [fiber],
        [EDFAConfig()],
        [ROADMConfig(insertion_loss_db=1, add_drop_loss_db=0, passband_hz=2e9)],
    )
    evaluator = PyNLOQoTEvaluator(grid, symbol_rate_hz=1e9, sps=2, seed=3)
    signal, _ = evaluator.create_wdm_signal(np.array([0, 1e-3, 0]), num_symbols=64)
    output = evaluator.propagate_path(signal, path)
    assert output.data.shape == signal.data.shape
    assert output.ctx.meta["roadm_pass_channels"] == [1]
