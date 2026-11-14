"""Coupled Raman span solver: conservation, loss, conventions and backends."""

from typing import Any, Literal

import numpy as np
import pytest

from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_ode import (
    RamanPaperSpanSolver,
    clear_raman_matrix_cache,
    raman_matrix_cache_info,
    resolve_raman_backend,
)
from network_nlo_eval.network.elements import RamanPumpConfig


def make_solver(
    *,
    direction: Literal["forward", "backward"] = "forward",
    convention: str = "total",
    **kwargs: Any,
) -> RamanPaperSpanSolver:
    pump = RamanPumpConfig(
        wavelength_nm=1450, power_w=0.02, direction=direction, attenuation_db_per_km=0
    )
    return RamanPaperSpanSolver(
        np.array([193.75e12]),
        np.array([75e9]),
        [pump],
        20e3,
        np.zeros(1),
        np.array([80e-12]),
        np.zeros(1),
        np.array([80e-12]),
        RamanGainModel(),
        n_steps=64,
        n_iterations=60,
        signal_polarizations=2 if convention == "per_pol" else 1,
        launch_power_convention=convention,
        **kwargs,
    )


def test_forward_pump_conserves_photons_and_power_conventions() -> None:
    total = make_solver().solve(np.array([0.03]))
    per_pol = make_solver(convention="per_pol").solve(np.array([0.015]))
    photons_received = (total.power_out_w[0] - 0.03) / 193.75e12
    photons_lost = (0.02 - total.pump_power_out_w[0]) / (C_LIGHT / 1450e-9)
    assert photons_lost / photons_received == pytest.approx(1, rel=1e-6)
    np.testing.assert_allclose(per_pol.power_out_w * 2, total.power_out_w, rtol=1e-12)
    np.testing.assert_allclose(per_pol.raman_ase_w * 2, total.raman_ase_w, rtol=1e-12)
    np.testing.assert_allclose(
        per_pol.pump_power_out_w, total.pump_power_out_w, rtol=1e-12
    )


def test_backward_pump_loss_segments_and_matrix_cache() -> None:
    clear_raman_matrix_cache()
    solver = make_solver(direction="backward", lumped_loss=[(10e3, 3)])
    again = make_solver(direction="backward", lumped_loss=[(10e3, 3)])
    assert raman_matrix_cache_info()["hits"] == 1
    result = solver.solve(np.array([0.003]))
    other = again.solve(np.array([0.003]))
    assert result.converged
    assert result.sample_positions_u is not None
    assert result.power_out_w[0] > 0
    np.testing.assert_array_equal(result.power_out_w, other.power_out_w)
    undepleted = make_solver(
        direction="backward", pump_depletion=False, lumped_loss=[(10e3, 3)]
    ).solve(np.array([0.003]))
    assert undepleted.pump_power_out_w[0] == pytest.approx(
        0.02 * 10 ** (-0.3), rel=1e-12
    )


def test_numba_backend_matches_python_and_rejects_loss() -> None:
    python = make_solver(backend="python").solve(np.array([0.003]))
    numba = make_solver(backend="numba").solve(np.array([0.003]))
    np.testing.assert_allclose(numba.power_out_w, python.power_out_w, rtol=1e-12)
    np.testing.assert_allclose(numba.raman_ase_w, python.raman_ase_w, rtol=1e-12)
    assert (
        resolve_raman_backend("auto", n_steps=100, n_lumped_loss=1, available=True)
        == "python"
    )
    with pytest.raises(ValueError, match="lumped"):
        make_solver(backend="numba", lumped_loss=[(10e3, 3)])
