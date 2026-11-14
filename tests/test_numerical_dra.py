"""Independent DRA reference and numerical fallback integration."""

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.numerical_dra import NumericalDRASpanSolver
from network_nlo_eval.models.raman import RamanGainModel
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
)
from network_nlo_eval.rwa.two_tier import (
    H3UncertaintyModel,
    TwoTierConfig,
    TwoTierEvaluator,
    numerical_dra_factory,
)


def test_independent_ode_matches_fixed_step_solver() -> None:
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.02, direction="backward")
    args = (
        np.array([193.75e12]),
        np.array([32e9]),
        [pump],
        20e3,
        np.array([4.6e-5]),
        np.array([80e-12]),
        np.array([5.7e-5]),
        np.array([80e-12]),
        RamanGainModel(),
    )
    numerical = NumericalDRASpanSolver(*args, n_profile=65, n_iterations=60).solve(
        np.array([1e-3])
    )
    fixed = RamanPaperSpanSolver(*args, n_steps=128, n_iterations=60).solve(
        np.array([1e-3])
    )
    assert numerical.converged
    assert fixed.converged
    np.testing.assert_allclose(numerical.power_out_w, fixed.power_out_w, rtol=2e-4)
    np.testing.assert_allclose(
        numerical.pump_power_out_w, fixed.pump_power_out_w, rtol=2e-4
    )
    np.testing.assert_allclose(numerical.raman_ase_w, fixed.raman_ase_w, rtol=2e-3)


def test_two_tier_runs_registered_numerical_fallback() -> None:
    fiber = FiberSpanConfig(length_km=20)
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    power = np.array([1e-3, 1e-3])

    class Budget:
        validity = SimpleNamespace(
            flagged=np.array([True, False]), active=np.ones(2, dtype=bool)
        )

        def adjusted_snr_db(self, snr):
            return snr - 1

    class H3:
        def evaluate_path_with_uncertainty(self, _path, _power):
            return SimpleNamespace(snr_db=np.array([20.0, 21.0])), Budget()

    factory = numerical_dra_factory(
        grid,
        fiber,
        symbol_rate_hz=32e9,
        n_profile=33,
        n_xpm_x=8,
        n_xpm_y=4,
        n_spm_x=8,
        n_spm_y=4,
    )
    result = TwoTierEvaluator(
        cast("H3UncertaintyModel", H3()), factory, config=TwoTierConfig(enabled=True)
    ).evaluate(path, power)
    assert result.fallback_ran
    assert np.isfinite(result.effective_snr_db[0])
    assert result.effective_snr_db[0] == result.fallback_snr_db[0]
    assert result.effective_snr_db[1] == pytest.approx(20)
    assert np.isnan(result.fallback_snr_db[1])
