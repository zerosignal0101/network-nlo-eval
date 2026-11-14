"""Tests for launch-power and ASE polarization conventions."""

import numpy as np
import pytest

from network_nlo_eval.models.power_conventions import (
    ase_polarization_scale,
    launch_power_scale,
    resolve_launch_power_convention,
    to_convention,
)


def test_power_conversion_preserves_snr_and_round_trips() -> None:
    signal = np.array([0.001, 0.002])
    noise = np.array([1e-6, 4e-6])
    per_pol_signal = to_convention(signal, "total", "per_pol")
    per_pol_noise = to_convention(noise, "total", "per_pol")
    np.testing.assert_allclose(per_pol_signal / per_pol_noise, signal / noise)
    np.testing.assert_array_equal(
        to_convention(per_pol_signal, "per_pol", "total"), signal
    )
    assert to_convention(signal, "total", "total") is signal
    np.testing.assert_array_equal(to_convention([2.0, 4.0], "total", "per_pol"), [1, 2])


def test_scales_and_polarization_validation() -> None:
    assert launch_power_scale("total") == 1
    assert ase_polarization_scale("per_pol") == 0.5
    assert resolve_launch_power_convention("per_pol", 2) == "per_pol"
    assert resolve_launch_power_convention("total", 2) == "total"
    with pytest.raises(ValueError, match="signal_polarizations must be 2"):
        resolve_launch_power_convention("per_pol", 1)
    with pytest.raises(ValueError, match="must be one of"):
        to_convention(1.0, "unknown", "total")
