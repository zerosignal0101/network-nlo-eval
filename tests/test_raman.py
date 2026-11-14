"""Raman gain, ASE and calibration-table behavior."""

import json
from pathlib import Path

import numpy as np
import pytest

from network_nlo_eval.models.raman import (
    C_LIGHT,
    RamanAmplifier,
    RamanGainModel,
    resolve_raman_gain_model,
)
from network_nlo_eval.models.raman_table import (
    RamanGainTable,
    load_raman_gain_table,
    save_raman_gain_table,
)
from network_nlo_eval.network.elements import RamanPumpConfig


def test_pump_gain_ase_and_direction() -> None:
    frequencies = np.array([C_LIGHT / (1450e-9) - 13.2e12, 193e12])
    bandwidths = np.full(2, 32e9)
    model = RamanGainModel()
    forward = RamanPumpConfig(wavelength_nm=1450, power_w=0.2, direction="forward")
    backward = RamanPumpConfig(wavelength_nm=1450, power_w=0.2, direction="backward")
    weak = RamanPumpConfig(wavelength_nm=1450, power_w=0.1)
    np.testing.assert_array_equal(model.on_off_gain(frequencies, [], 80e3), 1)
    np.testing.assert_array_equal(model.ase_w(frequencies, bandwidths, [], 80e3), 0)
    forward_gain = model.on_off_gain(frequencies, [forward], 80e3)
    np.testing.assert_array_equal(
        forward_gain, model.on_off_gain(frequencies, [backward], 80e3)
    )
    assert np.all(forward_gain > model.on_off_gain(frequencies, [weak], 80e3))
    amp = RamanAmplifier([backward], 80e3, model=model)
    signal, ase = amp.apply(np.full(2, 1e-3), frequencies, bandwidths)
    np.testing.assert_allclose(signal, forward_gain * 1e-3)
    assert np.all(ase > 0)
    assert amp.active


@pytest.mark.parametrize("suffix", ["json", "csv"])
def test_table_round_trip_and_override_policy(tmp_path: Path, suffix: str) -> None:
    table = RamanGainTable(
        offsets_hz=np.array([0.0, 13.2e12, 26.4e12]),
        normalized_gain=np.array([0.0, 1.0, 0.0]),
        reference_a_eff_m2=80e-12,
        fiber="sample",
        source="measurement",
    )
    path = tmp_path / f"gain.{suffix}"
    save_raman_gain_table(table, path)
    loaded = load_raman_gain_table(path)
    np.testing.assert_array_equal(loaded.offsets_hz, table.offsets_hz)
    np.testing.assert_array_equal(loaded.normalized_gain, table.normalized_gain)
    assert not loaded.offsets_hz.flags.writeable
    model = RamanGainModel.from_file(path)
    assert model.fiber == "sample"
    np.testing.assert_allclose(
        model.gain_shape(np.array([-1, 6.6e12, 13.2e12, 30e12])), [0, 0.5, 1, 0]
    )
    with pytest.warns(UserWarning, match="overriding calibration"):
        changed = resolve_raman_gain_model(path, temperature_k=310)
    assert changed.temperature_k == 310
    assert changed.calibration_overrides == {"temperature_k": 310}
    with pytest.raises(ValueError, match="cannot be overridden"):
        loaded.to_model(strict_calibration_override=True, temperature_k=310)


def test_invalid_table_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        RamanGainTable(np.array([1.0, 1.0]), np.array([0.0, 1.0]))
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"format": "incorrect", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required"):
        load_raman_gain_table(path)


def test_two_pump_gain_and_ase_match_reference() -> None:
    frequencies = np.array([190e12, 193e12, 196e12])
    bandwidths = np.full(3, 32e9)
    pumps = [
        RamanPumpConfig(wavelength_nm=1450, power_w=0.2, direction="backward"),
        RamanPumpConfig(wavelength_nm=1480, power_w=0.1, direction="forward"),
    ]
    model = RamanGainModel()
    np.testing.assert_allclose(
        model.on_off_gain(frequencies, pumps, 80e3),
        [7.413643066778057, 19.719466828936834, 4.088181720064254],
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        model.ase_w(frequencies, bandwidths, pumps, 80e3),
        [7.278514720345502e-09, 4.947195345925197e-08, 1.4064271058411139e-08],
        rtol=1e-13,
    )
