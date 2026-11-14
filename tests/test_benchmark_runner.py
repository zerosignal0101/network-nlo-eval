"""Benchmark runner power conventions and comparison statistics."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import (
    ModelRun,
    accuracy_vs_reference,
    sweep_snr,
    time_calls,
)


class Model:
    launch_power_convention = "total"

    def evaluate_path(self, _path, power):
        return SimpleNamespace(snr_db=10 * np.log10(power / 1e-3))


def test_sweep_timing_and_convention_checks() -> None:
    case = replace(build_case("scaling_40ch"), num_channels=3)
    model = Model()
    np.testing.assert_allclose(
        sweep_snr(model, case, (-3.0, 0.0)), [[-3.0] * 3, [0.0] * 3], atol=1e-12
    )
    cold, median, p95, peak, reps = time_calls(
        model, build_path(case), launch_profile(case), reps=3
    )
    assert cold >= 0
    assert median > 0
    assert p95 >= median
    assert peak > 0
    assert reps == 3
    with pytest.raises(ValueError, match="launch_power_convention"):
        sweep_snr(model, replace(case, launch_power_convention="per_pol"), (0.0,))


def test_accuracy_and_serialization_drop_nonfinite_snr() -> None:
    def run(snr: list[float], powers: tuple[float, ...] = (0.0,)) -> ModelRun:
        return ModelRun("model", "case", np.array([snr]), powers, 1, 1, 1, 1000, 1, 1)

    sample = run([10.0, 22.0, np.nan])
    reference = run([8.0, 20.0, 30.0])
    comparison = accuracy_vs_reference(sample, reference)
    assert comparison["samples"] == 2
    assert comparison["dropped"] == 1
    assert comparison["mean_db"] == pytest.approx(2)
    assert comparison["rms_db"] == pytest.approx(2)
    assert sample.to_dict()["snr_db"][0][2] is None
    with pytest.raises(ValueError, match="same power sweep"):
        accuracy_vs_reference(sample, run([8.0, 20.0, 30.0], powers=(1.0,)))
