"""Boundary iteration accuracy, safeguards and independent BVP comparison."""

from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_bvp

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.pcfm_h3_raman import (
    PCFH3RamanModel,
    PCFH3RamanSpanKernel,
)
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_iteration import BackwardPumpIteration
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
)


def test_anderson_solves_affine_map_without_loosening_residual() -> None:
    updater = BackwardPumpIteration("anderson")
    current = np.array([0.0])
    for _ in range(3):
        current = updater.update(current, 0.5 * current + 0.1)
    np.testing.assert_allclose(current, [0.2], atol=1e-14)
    assert np.max(np.abs(0.5 * current + 0.1 - current)) < 1e-14


def test_anderson_rejects_negative_extrapolation_and_residual_growth() -> None:
    updater = BackwardPumpIteration("anderson")
    first = updater.update(np.array([1.0]), np.array([0.5]))
    np.testing.assert_array_equal(first, [0.75])
    # The scalar secant extrapolation would be negative.
    second = updater.update(first, np.array([0.3]))
    np.testing.assert_array_equal(second, [0.525])

    for bad_map in (np.array([-0.5]), np.array([np.nan])):
        updater = BackwardPumpIteration("anderson")
        first = updater.update(np.array([1.0]), np.array([0.5]))
        # Roll back to the preceding half-damped step after actual growth.
        np.testing.assert_array_equal(updater.update(first, bad_map), first)


def test_rank_deficient_history_is_finite() -> None:
    updater = BackwardPumpIteration("anderson")
    current = np.ones(4)
    for _ in range(8):
        current = updater.update(current, current + 0.01)
        assert np.all(np.isfinite(current))
        assert np.all(current >= 0)


@pytest.mark.parametrize("method", ["relaxed", "picard", "anderson"])
def test_backward_two_wave_solution_matches_independent_bvp(method: str) -> None:
    """Solve the counterpropagating two-wave boundary problem independently."""
    f_signal = 193.75e12
    f_pump = C_LIGHT / 1450e-9
    gain_model = RamanGainModel()
    gain = float(gain_model.gain_shape(f_pump - f_signal)) * 6e-14 / 80e-12
    length = 20e3
    signal_input = 0.03
    pump_input = 0.02

    def rhs(_z, power):
        transfer = length * gain * power[0] * power[1]
        return np.vstack([transfer, (f_pump / f_signal) * transfer])

    def boundary(left, right):
        return np.array([left[0] - signal_input, right[1] - pump_input])

    z = np.linspace(0, 1, 65)
    reference = solve_bvp(
        rhs,
        boundary,
        z,
        np.vstack([np.full(z.size, signal_input), np.full(z.size, pump_input)]),
        tol=1e-10,
    )
    assert reference.success
    pump = RamanPumpConfig(
        wavelength_nm=1450,
        power_w=pump_input,
        direction="backward",
        attenuation_db_per_km=0,
    )
    solver = RamanPaperSpanSolver(
        np.array([f_signal]),
        np.array([32e9]),
        [pump],
        length,
        np.zeros(1),
        np.array([80e-12]),
        np.zeros(1),
        np.array([80e-12]),
        gain_model,
        n_steps=256,
        n_iterations=80,
        include_ase=False,
        iteration_method=method,
    )
    result = solver.solve(np.array([signal_input]))
    assert result.converged
    # Includes the grid interpolation error and the peak-pump-normalized
    # 1e-7 residual, which is larger relative to the depleted pump output.
    np.testing.assert_allclose(result.power_out_w, reference.sol(1)[0], rtol=5e-7)
    np.testing.assert_allclose(result.pump_power_out_w, reference.sol(0)[1], rtol=5e-7)


@pytest.mark.parametrize("method", ["picard", "anderson"])
@pytest.mark.parametrize(
    ("backend", "loss"), [("python", False), ("numba", False), ("python", True)]
)
def test_iteration_preserves_profiles_ase_and_coordinate_scaling(
    method: str, backend: str, loss: bool
) -> None:
    grid = SpectrumGrid(7, 75e9, 193.75e12)
    fiber = FiberSpanConfig(length_km=60)
    pumps = [
        RamanPumpConfig(wavelength_nm=1450, power_w=0.2, direction="backward"),
        RamanPumpConfig(wavelength_nm=1435, power_w=0.1, direction="forward"),
    ]
    path = MultiSpanOpticalPath([fiber] * 2, [EDFAConfig()] * 2)
    options: dict[str, Any] = {
        "raman_pumps": pumps,
        "raman_steps": 32,
        "raman_iterations": 80,
        "raman_backend": backend,
        "lumped_loss": [(30e3, 1.0)] if loss else None,
    }
    power = np.linspace(0.0005, 0.002, 7)
    reference = create_model("pcfm_h3_raman", grid, fiber, 64e9, **options)
    accelerated = create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        64e9,
        raman_iteration_method=method,
        **options,
    )
    per_pol = create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        64e9,
        raman_iteration_method=method,
        signal_polarizations=2,
        launch_power_convention="per_pol",
        **options,
    )
    assert isinstance(accelerated, PCFH3RamanModel)
    expected = reference.evaluate_path(path, power)
    actual = accelerated.evaluate_path(path, power)
    scaled = per_pol.evaluate_path(path, power / 2)
    assert all(report.raman_converged for report in accelerated.last_span_validity)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"):
        np.testing.assert_allclose(
            getattr(actual, name), getattr(expected, name), rtol=3e-6, atol=0
        )
        np.testing.assert_allclose(
            getattr(actual, name), 2 * getattr(scaled, name), rtol=1e-11, atol=0
        )
    np.testing.assert_allclose(actual.snr_db, expected.snr_db, rtol=0, atol=1e-5)


@pytest.mark.parametrize("method", ["relaxed", "picard", "anderson"])
def test_iteration_cap_does_not_report_false_convergence(method: str) -> None:
    grid = SpectrumGrid(3, 75e9, 193.75e12)
    fiber = FiberSpanConfig(length_km=80)
    model = create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        64e9,
        raman_pumps=[
            RamanPumpConfig(wavelength_nm=1450, power_w=0.3, direction="backward")
        ],
        raman_iteration_method=method,
        raman_iterations=1,
    )
    assert isinstance(model, PCFH3RamanModel)
    with pytest.warns(RuntimeWarning, match="converged=False"):
        model.evaluate_path(
            MultiSpanOpticalPath([fiber], [EDFAConfig()]), np.full(3, 1e-3)
        )
    assert not model.last_span_validity[0].raman_converged


@pytest.mark.parametrize("method", ["picard", "anderson"])
@pytest.mark.parametrize("direction", ["none", "forward"])
def test_no_backward_pump_keeps_results_identical(method: str, direction: str) -> None:
    grid = SpectrumGrid(3, 75e9, 193.75e12)
    fiber = FiberSpanConfig(length_km=40)
    pumps = (
        []
        if direction == "none"
        else [RamanPumpConfig(wavelength_nm=1450, power_w=0.1, direction="forward")]
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig()])
    power = np.full(3, 1e-3)
    reference = create_model(
        "pcfm_h3_raman", grid, fiber, 64e9, raman_pumps=pumps
    ).evaluate_path(path, power)
    actual = create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        64e9,
        raman_pumps=pumps,
        raman_iteration_method=method,
    ).evaluate_path(path, power)
    for name in (
        "signal_power_w",
        "spm_noise_w",
        "xpm_noise_w",
        "ase_noise_w",
        "snr_db",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))


def test_invalid_method_is_rejected_before_evaluation() -> None:
    with pytest.raises(ValueError, match="raman_iteration_method"):
        create_model(
            "pcfm_h3_raman",
            SpectrumGrid(3, 75e9, 193.75e12),
            FiberSpanConfig(length_km=40),
            raman_iteration_method="unknown",
        )


@pytest.mark.parametrize("method", ["picard", "anderson"])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_accepted_state_satisfies_actual_boundary_residual(
    method: str, backend: str, monkeypatch
) -> None:
    """Reevaluate the physical map at the returned state, independently of mixing."""
    fiber = FiberSpanConfig(length_km=80)
    model = create_model(
        "pcfm_h3_raman",
        SpectrumGrid(3, 75e9, 193.75e12),
        fiber,
        64e9,
        raman_pumps=[
            RamanPumpConfig(wavelength_nm=1450, power_w=0.3, direction="backward")
        ],
        raman_backend=backend,
        raman_iteration_method=method,
    )
    assert isinstance(model, PCFH3RamanModel)
    kernel = model.kernel_for(fiber)
    assert isinstance(kernel, PCFH3RamanSpanKernel)
    solver = kernel._solver_for(80e3)
    name = "_integrate_forward_numba" if backend == "numba" else "_integrate_forward"
    integrate = getattr(solver, name)
    final_backward = None

    def recording_forward(power_fw, ase_fw, backward):
        nonlocal final_backward
        final_backward = backward.copy()
        return integrate(power_fw, ase_fw, backward)

    monkeypatch.setattr(solver, name, recording_forward)
    power = np.full(3, 1e-3)
    result = solver.solve(power)
    assert result.converged
    assert final_backward is not None
    np.testing.assert_allclose(final_backward[-1], [0.3], rtol=0, atol=1e-15)
    power_fw, ase_fw = solver._initial_forward_state(power, np.zeros(3))
    _, _, power_grid, ase_grid, _ = integrate(power_fw, ase_fw, final_backward)
    forward_total = np.empty((solver.n_steps + 1, solver._n_waves))
    forward_total[:, solver._fwd_slots] = power_grid + ase_grid
    forward_total[:, solver._bw_slots] = final_backward
    backward_integrate = (
        solver._integrate_backward_numba
        if backend == "numba"
        else solver._integrate_backward
    )
    _, _, power_bw, ase_bw = backward_integrate(
        solver._p0_w[solver._bw_slots],
        np.zeros(solver._bw_slots.size),
        forward_total,
    )
    actual_residual = np.max(np.abs((power_bw + ase_bw)[::-1] - final_backward)) / 0.3
    assert actual_residual < 1e-7
