"""Bit-exactness guard for the ECZ model and the shared route layer.

The expected arrays in ``tests/fixtures/ecz_golden_bbb9ed4.json`` were captured from
``ecz-dev`` at commit ``bbb9ed4`` *before* the single-span kernel / route split. They
are compared on raw IEEE-754 bits, not with a tolerance: ``accumulate_route`` feeds
SNR thresholds in the RWA engine, so a last-place change is a behaviour change that
must be a deliberate decision rather than a silent refactor side effect.

Refreshing the fixture is only legitimate when the physics itself is meant to change;
doing it to make a red test green defeats the purpose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid, bits_per_symbol
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

FIXTURE = Path(__file__).parent / "fixtures" / "ecz_golden_bbb9ed4.json"


def _bits(array: Any) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).view(np.int64)


def _assert_bit_equal(expected: Any, actual: Any, label: str) -> None:
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    assert expected_array.shape == actual_array.shape, f"{label}: shape {expected_array.shape} != {actual_array.shape}"
    mismatch = _bits(expected_array) != _bits(actual_array)
    assert not mismatch.any(), (
        f"{label}: {int(mismatch.sum())}/{expected_array.size} elements differ from the golden vector; "
        f"first difference at flat index {int(np.flatnonzero(mismatch)[0])}"
    )


def _assert_path_result(case: dict, result: Any) -> None:
    _assert_bit_equal(case["signal_power_w"], result.signal_power_w, "signal_power_w")
    _assert_bit_equal(case["spm_noise_w"], result.spm_noise_w, "spm_noise_w")
    _assert_bit_equal(case["xpm_noise_w"], result.xpm_noise_w, "xpm_noise_w")
    _assert_bit_equal(case["ase_noise_w"], result.ase_noise_w, "ase_noise_w")
    _assert_bit_equal(case["snr_db"], result.snr_db, "snr_db")
    assert result.span_count == case["span_count"]
    assert result.roadm_count == case["roadm_count"]


@pytest.fixture(scope="module")
def golden() -> dict[str, dict]:
    """Load the captured cases once per module."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


def _fiber() -> FiberSpanConfig:
    return FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)


def test_golden_legacy_off_80ch(golden: dict[str, dict]) -> None:
    """Fixed grid, pairwise tier, one span."""
    case = golden["legacy_off_80ch_1span"]
    evaluator = MultiBandISRSGN(SpectrumGrid(80, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9, ode_accel="off")
    path = MultiSpanOpticalPath([_fiber()], [EDFAConfig(noise_figure_db=5.5)])
    _assert_path_result(case, evaluator.evaluate_path(path, np.asarray(case["power_in_w"])))


def test_golden_legacy_matvec_320ch(golden: dict[str, dict]) -> None:
    """Fixed grid, matvec tier: the widest captured case."""
    case = golden["legacy_matvec_320ch_1span"]
    evaluator = MultiBandISRSGN(SpectrumGrid(320, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9, ode_accel="matvec")
    path = MultiSpanOpticalPath([_fiber()], [EDFAConfig(noise_figure_db=5.5)])
    _assert_path_result(case, evaluator.evaluate_path(path, np.asarray(case["power_in_w"])))


def test_golden_preemphasis_four_spans(golden: dict[str, dict]) -> None:
    """Sloped launch profile with dark channels over four spans."""
    case = golden["legacy_preemphasis_150ch_4span"]
    evaluator = MultiBandISRSGN(SpectrumGrid(150, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9)
    path = MultiSpanOpticalPath([_fiber()] * 4, [EDFAConfig(noise_figure_db=5.5)] * 4)
    _assert_path_result(case, evaluator.evaluate_path(path, np.asarray(case["power_in_w"])))


def test_golden_elements_roadm_and_array_nf(golden: dict[str, dict]) -> None:
    """Polynomial attenuation, array NF, ROADM equalisation and blocked channels."""
    case = golden["elements_poly_nf_array_roadm_64ch"]
    fiber = FiberSpanConfig(
        length_km=100.0,
        max_span_length_km=100.0,
        use_attenuation_polynomial=True,
        effective_area_um2_ref=83.0,
        aeff_slope_um2_nm=0.06,
    )
    edfa = EDFAConfig(noise_figure_db=np.linspace(5.0, 6.5, 64))
    roadm = ROADMConfig(
        insertion_loss_db=6.0,
        express_attenuation_db=0.4,
        add_drop_loss_db=1.2,
        booster_noise_figure_db=5.2,
        filtering_penalty_db=0.3,
        equalize_output_power=True,
        blocked_channels=[0, 7, 63],
    )
    evaluator = MultiBandISRSGN(SpectrumGrid(64, 50e9, 193.1e12), fiber, symbol_rate_hz=32e9)
    path = MultiSpanOpticalPath([fiber, fiber], [edfa, edfa], [roadm, roadm])
    _assert_path_result(case, evaluator.evaluate_path(path, np.asarray(case["power_in_w"])))


def test_golden_flexible_grid(golden: dict[str, dict]) -> None:
    """Per-connection mode with heterogeneous occupied bandwidths."""
    case = golden["flex_hetero_4conn_2span"]
    grid = FlexibleGrid.from_legacy_grid(4, 50e9, 193.1e12, slots_per_channel=4)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=index,
            first_slot=first_slot,
            n_slots=n_slots,
            bit_rate_bps=bit_rate,
            modulation=modulation,
            symbol_rate_hz=bit_rate / bits_per_symbol(modulation),
            roll_off=0.0,
            launch_power_w=1e-3,
        )
        for index, (first_slot, n_slots, bit_rate, modulation) in enumerate(
            ((0, 4, 100e9, "QPSK"), (4, 4, 100e9, "QPSK"), (8, 8, 400e9, "16QAM"))
        )
    ]
    fiber = FiberSpanConfig(length_km=60.0, max_span_length_km=60.0)
    evaluator = MultiBandISRSGN(
        grid,
        fiber,
        symbol_rate_hz=50e9,
        connections=connections,
        slots_per_channel=4,
        channel_bandwidth_source="occupied",
        ode_accel="off",
    )
    path = MultiSpanOpticalPath([fiber] * 2, [EDFAConfig(noise_figure_db=5.5)] * 2)
    _assert_path_result(case, evaluator.evaluate_path(path, np.asarray(case["power_in_w"])))


def test_golden_batch_matches_captured(golden: dict[str, dict]) -> None:
    """The batched span entry keeps its captured (B, N) results."""
    case = golden["kernel_batch_3x120ch"]
    evaluator = MultiBandISRSGN(SpectrumGrid(120, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9, ode_accel="matvec")
    matrix = np.asarray(case["power_in_w"], dtype=np.float64).reshape(3, 120)
    out = evaluator.calc_span_noise_and_power_batch(matrix, 80e3, EDFAConfig(noise_figure_db=5.5))
    for name, array in zip(("power_out_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"), out, strict=True):
        _assert_bit_equal(case[name], np.asarray(array).ravel(), name)
