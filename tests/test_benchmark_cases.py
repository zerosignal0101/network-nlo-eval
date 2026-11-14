"""Benchmark cases preserve physical power and path conventions."""

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    band_edges_hz,
    build_case,
    build_path,
    grid_for,
    launch_profile,
)


def test_benchmark_grid_path_and_tilt_conventions() -> None:
    case = build_case("scaling_40ch")
    assert band_edges_hz("C+L") == (184.4e12, 195.9e12)
    assert grid_for(case).num_channels == case.num_channels
    assert len(build_path(case).fiber_configs) == case.span_count
    flat = launch_profile(case, power_dbm=0)
    tilted = launch_profile(case, power_dbm=0, tilt_db=6)
    assert tilted.sum() == pytest.approx(flat.sum())
    assert 10 * np.log10(tilted[-1] / tilted[0]) == pytest.approx(6)
    np.testing.assert_array_equal(
        launch_profile(case, power_dbm=0, tilt_db=6, launch_power_convention="per_pol"),
        tilted / 2,
    )
    centered = launch_profile(case, power_dbm=0, tilt_db=6, tilt_reference="center")
    assert np.sqrt(centered[19] * centered[20]) == pytest.approx(1e-3)
    with pytest.raises(ValueError, match="tilt_db"):
        launch_profile(case, tilt_db=-1)
