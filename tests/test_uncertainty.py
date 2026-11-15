"""Error-budget regression and applicability decisions."""

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.pcfm import PCFMModel
from network_nlo_eval.models.uncertainty import (
    DENSE_GRID_PENALTY_SCHEDULE_V4,
    H3UncertaintyConfig,
    dense_grid_schedule_penalty,
    estimate_h3_error_budget,
)
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
)


def _evaluate(spacing: float = 50e9):
    fiber = FiberSpanConfig(length_km=80)
    model = PCFMModel(SpectrumGrid(3, spacing, 193.1e12), fiber, 32e9)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    power = np.array([1e-3, 2e-3, 0.0])
    return model, path, power, model.evaluate_path(path, power)


def test_plain_pcfm_budget_matches_reference_and_marks_invalid_result() -> None:
    model, path, power, result = _evaluate()
    budget = estimate_h3_error_budget(model, path, power, result)
    np.testing.assert_allclose(budget.bias_db, [-0.5, -0.5, -0.5])
    np.testing.assert_allclose(budget.sigma_db, [np.sqrt(0.5)] * 3)
    np.testing.assert_allclose(
        budget.margin_db, [2.164213562373095, 2.164213562373095, 0]
    )
    np.testing.assert_array_equal(budget.validity.valid, [True, True, False])
    assert budget.validity.pump_mode == "none"
    snr = result.snr_db.copy()
    snr[0] = np.nan
    broken = estimate_h3_error_budget(model, path, power, replace(result, snr_db=snr))
    assert broken.validity.active[0]
    assert not broken.validity.valid[0]
    assert np.isinf(broken.margin_db[0])
    assert np.isneginf(broken.adjusted_snr_db(snr)[0])


def test_dense_grid_schedule_adds_margin_and_checks_order() -> None:
    model, path, power, result = _evaluate(20e9)
    plain = estimate_h3_error_budget(model, path, power, result)
    cfg = H3UncertaintyConfig(
        dense_grid_penalty_schedule=DENSE_GRID_PENALTY_SCHEDULE_V4
    )
    adjusted = estimate_h3_error_budget(model, path, power, result, config=cfg)
    assert plain.validity.dense_grid_overlap
    penalty, clamped = dense_grid_schedule_penalty(
        DENSE_GRID_PENALTY_SCHEDULE_V4, 20 / 32
    )
    assert not clamped
    np.testing.assert_allclose(adjusted.margin_db[:2] - plain.margin_db[:2], penalty)
    assert adjusted.margin_db[2] == 0
    with pytest.raises(ValueError, match="strictly increasing"):
        H3UncertaintyConfig(dense_grid_penalty_schedule=((0.5, 1), (0.5, 2)))


def test_active_lumped_raman_requires_a_distributed_raman_model() -> None:
    fiber = FiberSpanConfig(length_km=80)
    with pytest.raises(ValueError, match="Active Raman pumps require"):
        PCFMModel(
            SpectrumGrid(3, 50e9, 193.1e12),
            fiber,
            32e9,
            raman_model="lumped",
            raman_pumps=[RamanPumpConfig(wavelength_nm=1450, power_w=0.2)],
        )
