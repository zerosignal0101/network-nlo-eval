"""Focused integration tests for the ISRS-GN physical model."""

import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    available_models,
    create_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig


def test_multispan_path_matches_reference_vector() -> None:
    fiber = FiberSpanConfig(length_km=120, max_span_length_km=80)
    model = MultiBandISRSGN(
        SpectrumGrid(4, 50e9, 193.1e12), fiber, 32e9, ode_accel="off"
    )
    path = MultiSpanOpticalPath(
        [fiber], [EDFAConfig(target_gain_db=17, noise_figure_db=5.5)], [ROADMConfig()]
    )
    result = model.evaluate_path(path, np.array([0.001, 0.002, 0.0, 0.0005]))
    expected = {
        "signal_power_w": [
            0.010002164551407077,
            0.01999999959844569,
            0.0,
            0.004997835850147171,
        ],
        "spm_noise_w": [
            2.571260335472043e-6,
            2.0602308488771237e-5,
            0.0,
            3.229208040086056e-7,
        ],
        "xpm_noise_w": [
            4.118901608114996e-6,
            2.275749274131666e-6,
            0.0,
            1.2090188427351127e-6,
        ],
        "ase_noise_w": [
            6.720871581488242e-6,
            6.721943533669781e-6,
            0.0,
            6.7240866248934956e-6,
        ],
        "snr_db": [28.7263174693283, 28.297382568619437, -np.inf, 27.820109217147454],
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(getattr(result, name), reference, rtol=1e-12, atol=0)
    assert result.span_count == 2
    assert result.roadm_count == 1
    assert "ecz" in available_models()
    assert isinstance(create_model("ecz", model.grid, fiber), MultiBandISRSGN)


def test_total_and_per_polarization_are_physically_equivalent() -> None:
    fiber = FiberSpanConfig(length_km=80)
    grid = SpectrumGrid(4, 50e9, 193.1e12)
    path = MultiSpanOpticalPath(
        [fiber], [EDFAConfig()], [ROADMConfig(equalize_output_power=True)]
    )
    total = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention="total")
    per_pol = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention="per_pol")
    a = total.evaluate_path(path, np.full(4, 1e-3))
    b = per_pol.evaluate_path(path, np.full(4, 0.5e-3))
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"):
        np.testing.assert_array_equal(getattr(a, name), 2 * getattr(b, name))
    np.testing.assert_array_equal(a.snr_db, b.snr_db)


def test_flexible_bandwidth_blocking_and_fiber_cache() -> None:
    grid = FlexibleGrid.from_legacy_grid(2, 50e9, 193.1e12)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=i,
            first_slot=i * 4,
            n_slots=4,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=25e9,
            roll_off=0,
            launch_power_w=1e-3,
        )
        for i in range(2)
    ]
    fiber = FiberSpanConfig(length_km=80)
    model = MultiBandISRSGN(
        grid,
        fiber,
        50e9,
        connections=connections,
        slots_per_channel=4,
        channel_bandwidth_source="occupied",
    )
    np.testing.assert_array_equal(model.channel_bandwidth_hz_channels, [50e9, 50e9])
    assert model.blocked_nodes(ROADMConfig(blocked_slots=[1])) == [0]
    assert model.blocked_nodes(ROADMConfig(blocked_channels=[1])) == [1]
    child = model.for_fiber(fiber)
    assert child is model.for_fiber(fiber)
    assert child.connections == connections


def test_cached_and_batched_models_match_pairwise() -> None:
    fiber = FiberSpanConfig(length_km=80)
    grid = SpectrumGrid(48, 50e9, 193.1e12)
    power = np.full(48, 1e-3)
    edfa = EDFAConfig(target_gain_db=17, noise_figure_db=5.5)
    off = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off")
    cached = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="cached")
    matvec = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="matvec")
    reference = off._calc_span_noise_and_power(power, 80e3, edfa)
    cached._calc_span_noise_and_power(power, 80e3, edfa)
    hit = cached._calc_span_noise_and_power(power, 80e3, edfa)
    for array in hit:
        array[:] = -1
    again = cached._calc_span_noise_and_power(power, 80e3, edfa)
    batch = matvec.calc_span_noise_and_power_batch(
        np.stack([power, np.zeros(48)]), 80e3, edfa
    )
    for expected, actual, accelerated in zip(reference, again, batch, strict=True):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_allclose(accelerated[0], expected, rtol=1e-12, atol=0)
        np.testing.assert_array_equal(accelerated[1], np.zeros(48))
    assert matvec._p1_xpm_matrices is not None
    with pytest.raises(ValueError, match="requires ode_accel"):
        off.calc_span_noise_and_power_batch(power[None, :], 80e3)
