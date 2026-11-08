"""拉曼增益谱文件入口的验收测试。

覆盖 H3 第四轮任务 N：

* JSON/CSV 往返与元数据保留；
* 与内存表 ``RamanGainModel.from_table`` 逐位一致；
* 缺字段/未知字段/单位错误/单调性/有限性/长度校验；
* 表内插值与表外置零、峰值归一化 warning；
* ``fiber``/``source``/``note``/``reference_wavelength_nm`` 溯源不丢失；
* 标定字段 override 必须 warning 或显式确认，不能静默；
* 默认 Gaussian/CZ 逐位不变；
* PCFM/H3/Hybrid 用 ``RamanGainModel.from_file`` 实例或路径跑短链路；
* 不支持文件入口的模型给出清晰错误；
* C+L 六泵可用。
"""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.hybrid_raman import RamanPaperSpanSolver
from network_nlo_eval.models.raman import (
    C_LIGHT,
    RamanGainModel,
    resolve_raman_gain_model,
)
from network_nlo_eval.models.raman_table import (
    RAMAN_GAIN_TABLE_FORMAT,
    RAMAN_GAIN_TABLE_VERSION,
    RamanGainTable,
    load_raman_gain_table,
    save_raman_gain_table,
)
from network_nlo_eval.network.elements import RamanPumpConfig

#: C 波段单跨短链路（40 信道），用于下游集成测试。
CASE_C = build_case("scaling_40ch")

#: C+L 40 信道单跨，用于六泵文件入口测试。
CASE_C_L = replace(
    build_case("c_57ch"),
    name="cl_40ch_1span",
    bands="C+L",
    num_channels=40,
    center_frequency_hz=190.0e12,
    span_count=1,
)

#: 论文 hybrid 的六泵 E 波段配置。
SIX_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]


def _payload(**overrides: object) -> dict[str, object]:
    """最小合法 JSON payload；``overrides`` 可替换任意字段。"""
    payload: dict[str, object] = {
        "format": RAMAN_GAIN_TABLE_FORMAT,
        "version": RAMAN_GAIN_TABLE_VERSION,
        "units": {
            "offsets_thz": "THz",
            "normalized_gain": "1",
            "temperature_k": "K",
            "peak_gain_coefficient_m_per_w": "m/W",
            "reference_wavelength_nm": "nm",
            "reference_a_eff_um2": "um^2",
        },
        "fiber": "synthetic-minimal",
        "source": "synthetic fixture, not measured",
        "note": "3-point triangular shape; shift = |f_p - f_s|",
        "temperature_k": 300.0,
        "peak_gain_coefficient_m_per_w": 6e-14,
        "reference_wavelength_nm": 1550.0,
        "reference_a_eff_um2": 80.0,
        "offsets_thz": [0.0, 13.2, 26.4],
        "normalized_gain": [0.0, 1.0, 0.0],
    }
    payload.update(overrides)
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    """把 payload 写到 ``path``。"""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _write_csv(path: Path, lines: list[str]) -> Path:
    """按行写 CSV 文本。"""
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _csv_lines(
    *,
    offsets_thz: list[float] | None = None,
    gains: list[float] | None = None,
    temperature_k: float = 300.0,
    peak: float = 6e-14,
    reference_nm: float = 1550.0,
    reference_a_eff_um2: float | None = 80.0,
    note: str = "shift = |f_p - f_s|",
    units: str | None = None,
) -> list[str]:
    """构造一份合法 CSV；默认 3 点三角谱。"""
    offsets = [0.0, 13.2, 26.4] if offsets_thz is None else offsets_thz
    values = [0.0, 1.0, 0.0] if gains is None else gains
    lines = [
        f"# format = {RAMAN_GAIN_TABLE_FORMAT}",
        f"# version = {RAMAN_GAIN_TABLE_VERSION}",
        "# units = "
        + (
            "offsets_thz=THz, normalized_gain=1, temperature_k=K, "
            "peak_gain_coefficient_m_per_w=m/W, reference_wavelength_nm=nm"
            + (", reference_a_eff_um2=um^2" if reference_a_eff_um2 is not None else "")
            if units is None
            else units
        ),
        f"# temperature_k = {temperature_k!r}",
        f"# peak_gain_coefficient_m_per_w = {peak!r}",
        f"# reference_wavelength_nm = {reference_nm!r}",
    ]
    if reference_a_eff_um2 is not None:
        lines.append(f"# reference_a_eff_um2 = {reference_a_eff_um2!r}")
    lines.extend(["# fiber = synthetic-csv", "# source = synthetic fixture, not measured", f"# note = {note}"])
    lines.append("offsets_thz,normalized_gain")
    lines.extend(f"{offset!r},{gain!r}" for offset, gain in zip(offsets, values, strict=True))
    return lines


def _measured_style_table(*, points: int = 601, span_hz: float = 30e12) -> tuple[np.ndarray, np.ndarray]:
    """合成“实测风格”谱：Gaussian + 纹波 + 有限频移范围，峰值归一到 1。"""
    analytic = RamanGainModel()
    offsets = np.linspace(0.0, span_hz, points)
    gains = analytic.gain_shape(offsets) * (1.0 + 0.08 * np.sin(2.0 * np.pi * offsets / 1.5e12))
    gains = np.maximum(gains, 0.0)
    return offsets, gains / float(np.max(gains))


def _write_measured_style(path: Path) -> tuple[RamanGainTable, Path]:
    """写一份合成实测风格 JSON 并返回表对象与路径。"""
    offsets_hz, gains = _measured_style_table()
    table = RamanGainTable(
        offsets_hz=offsets_hz,
        normalized_gain=gains,
        temperature_k=300.0,
        peak_gain_coefficient_m_per_w=6e-14,
        reference_wavelength_nm=1550.0,
        reference_a_eff_m2=80e-12,
        fiber="synthetic measured-style",
        source="synthetic fixture, not measured",
        note="Gaussian + 8% ripple; finite 0-30 THz range",
    )
    save_raman_gain_table(table, path)
    return table, path


def _result_arrays(result) -> dict[str, np.ndarray]:
    """路径结果里的数值数组。"""
    return {
        name: np.asarray(getattr(result, name))
        for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")
    }


# ---------------------------------------------------------------------------
# 往返与内存表一致性
# ---------------------------------------------------------------------------


def test_json_round_trip_preserves_arrays_and_provenance(tmp_path: Path) -> None:
    """JSON 加载/保存必须保留数组、标定字段与全部溯源文本。"""
    path = _write_json(tmp_path / "table.json", _payload())
    table = load_raman_gain_table(path)

    np.testing.assert_array_equal(table.offsets_hz, np.array([0.0, 13.2e12, 26.4e12]))
    np.testing.assert_array_equal(table.normalized_gain, np.array([0.0, 1.0, 0.0]))
    assert table.temperature_k == 300.0
    assert table.peak_gain_coefficient_m_per_w == 6e-14
    assert table.reference_wavelength_nm == 1550.0
    assert table.reference_a_eff_m2 == 80e-12
    assert table.fiber == "synthetic-minimal"
    assert table.source == "synthetic fixture, not measured"
    assert "shift" in (table.note or "")
    assert table.metadata["reference_wavelength_nm"] == 1550.0

    output = tmp_path / "roundtrip.json"
    save_raman_gain_table(table, output)
    reloaded = load_raman_gain_table(output)
    np.testing.assert_array_equal(reloaded.offsets_hz, table.offsets_hz)
    np.testing.assert_array_equal(reloaded.normalized_gain, table.normalized_gain)
    assert reloaded.metadata == table.metadata


def test_csv_round_trip_preserves_arrays_and_provenance(tmp_path: Path) -> None:
    """CSV 的 ``# key = value`` 元数据必须与 JSON 等价。"""
    path = _write_csv(tmp_path / "table.csv", _csv_lines())
    table = load_raman_gain_table(path)

    np.testing.assert_array_equal(table.offsets_hz, np.array([0.0, 13.2e12, 26.4e12]))
    np.testing.assert_array_equal(table.normalized_gain, np.array([0.0, 1.0, 0.0]))
    assert table.reference_a_eff_m2 == 80e-12
    assert table.note == "shift = |f_p - f_s|"

    output = tmp_path / "roundtrip.csv"
    save_raman_gain_table(table, output)
    reloaded = load_raman_gain_table(output)
    np.testing.assert_array_equal(reloaded.offsets_hz, table.offsets_hz)
    np.testing.assert_array_equal(reloaded.normalized_gain, table.normalized_gain)
    assert reloaded.metadata == table.metadata


def test_loaded_model_matches_in_memory_table_bit_for_bit(tmp_path: Path) -> None:
    """``from_file`` 与 ``from_table`` 使用同一频移/增益时结果必须逐位一致。"""
    path = _write_json(tmp_path / "table.json", _payload())
    table = load_raman_gain_table(path)
    loaded = RamanGainModel.from_file(path)
    memory = RamanGainModel.from_table(
        table.offsets_hz,
        table.normalized_gain,
        peak_gain_coefficient_m_per_w=table.peak_gain_coefficient_m_per_w,
        temperature_k=table.temperature_k,
        reference_a_eff_m2=table.reference_a_eff_m2,
    )

    np.testing.assert_array_equal(loaded.table[0], table.offsets_hz)
    np.testing.assert_array_equal(loaded.table[1], table.normalized_gain)
    probe = np.linspace(0.0, 32e12, 2001)
    np.testing.assert_array_equal(loaded.gain_shape(probe), memory.gain_shape(probe))

    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )
    frequencies = C_LIGHT / np.linspace(1530e-9, 1565e-9, 57)
    bandwidths = np.full(frequencies.size, 64e9)
    np.testing.assert_array_equal(
        loaded.on_off_gain(frequencies, [pump], 100e3),
        memory.on_off_gain(frequencies, [pump], 100e3),
    )
    np.testing.assert_array_equal(
        loaded.ase_w(frequencies, bandwidths, [pump], 100e3),
        memory.ase_w(frequencies, bandwidths, [pump], 100e3),
    )


# ---------------------------------------------------------------------------
# 字段、单位与数值校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "missing",
    ["format", "version", "offsets_thz", "normalized_gain", "temperature_k",
     "peak_gain_coefficient_m_per_w", "reference_wavelength_nm"],
)
def test_missing_required_fields_are_rejected(tmp_path: Path, missing: str) -> None:
    """任一必填字段缺失都必须报错并指出字段名。"""
    payload = _payload()
    payload.pop(missing)
    path = _write_json(tmp_path / "bad.json", payload)
    with pytest.raises(ValueError, match=missing):
        load_raman_gain_table(path)


def test_unknown_fields_are_rejected(tmp_path: Path) -> None:
    """未知字段不能静默忽略。"""
    path = _write_json(tmp_path / "bad.json", _payload(unexpected=1))
    with pytest.raises(ValueError, match="unknown"):
        load_raman_gain_table(path)


def test_units_are_validated(tmp_path: Path) -> None:
    """错误单位、未知单位键与非法 units 结构都必须报错。"""
    wrong = _payload(units={"offsets_thz": "GHz"})
    with pytest.raises(ValueError, match="unit of offsets_thz"):
        load_raman_gain_table(_write_json(tmp_path / "wrong.json", wrong))

    unknown_key = _payload(units={"offsets_thz": "THz", "mystery": "1"})
    with pytest.raises(ValueError, match="unknown units"):
        load_raman_gain_table(_write_json(tmp_path / "unknown.json", unknown_key))

    not_mapping = _payload(units="offsets_thz=THz")
    with pytest.raises(ValueError, match="units must be"):
        load_raman_gain_table(_write_json(tmp_path / "not_mapping.json", not_mapping))


def test_unit_aliases_are_accepted(tmp_path: Path) -> None:
    """常见单位别名（小写、um2）应被接受。"""
    payload = _payload(units={
        "offsets_thz": "thz",
        "normalized_gain": "dimensionless",
        "temperature_k": "kelvin",
        "peak_gain_coefficient_m_per_w": "m/W",
        "reference_wavelength_nm": "nm",
        "reference_a_eff_um2": "um2",
    })
    table = load_raman_gain_table(_write_json(tmp_path / "aliases.json", payload))
    assert table.reference_a_eff_m2 == 80e-12


def test_non_monotonic_offsets_are_rejected(tmp_path: Path) -> None:
    """频移必须非负且严格递增。"""
    decreasing = _payload(offsets_thz=[0.0, 20.0, 10.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        load_raman_gain_table(_write_json(tmp_path / "decreasing.json", decreasing))

    duplicate = _payload(offsets_thz=[0.0, 10.0, 10.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        load_raman_gain_table(_write_json(tmp_path / "duplicate.json", duplicate))

    negative = _payload(offsets_thz=[-1.0, 0.0, 10.0])
    with pytest.raises(ValueError, match="non-negative"):
        load_raman_gain_table(_write_json(tmp_path / "negative.json", negative))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_values_are_rejected(tmp_path: Path, bad: float) -> None:
    """NaN/Inf 不得进入插值表。"""
    path = _write_json(tmp_path / "bad.json", _payload(offsets_thz=[0.0, 13.2, bad], normalized_gain=[0.0, 1.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        load_raman_gain_table(path)


def test_length_and_shape_errors_are_rejected(tmp_path: Path) -> None:
    """长度不一致、点数不足、非一维与全零谱都必须报错。"""
    mismatch = _payload(normalized_gain=[0.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        load_raman_gain_table(_write_json(tmp_path / "mismatch.json", mismatch))

    too_short = _payload(offsets_thz=[13.2], normalized_gain=[1.0])
    with pytest.raises(ValueError, match="at least two"):
        load_raman_gain_table(_write_json(tmp_path / "short.json", too_short))

    nested = _payload(offsets_thz=[[0.0], [13.2]], normalized_gain=[0.0, 1.0])
    with pytest.raises(ValueError, match="must be a number"):
        load_raman_gain_table(_write_json(tmp_path / "nested.json", nested))

    all_zero = _payload(normalized_gain=[0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="positive gain"):
        load_raman_gain_table(_write_json(tmp_path / "zero.json", all_zero))

    negative_gain = _payload(normalized_gain=[0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match="non-negative"):
        load_raman_gain_table(_write_json(tmp_path / "negative_gain.json", negative_gain))


def test_invalid_json_and_extension_are_rejected(tmp_path: Path) -> None:
    """坏 JSON、非对象顶层与不支持的扩展名必须报错。"""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_raman_gain_table(bad_json)

    array_json = tmp_path / "array.json"
    array_json.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        load_raman_gain_table(array_json)

    txt = tmp_path / "table.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_raman_gain_table(txt)

    with pytest.raises(FileNotFoundError):
        load_raman_gain_table(tmp_path / "missing.json")


def test_csv_structure_errors_are_rejected(tmp_path: Path) -> None:
    """CSV 头、列数、非数值与元数据行错误必须带行号报错。"""
    bad_header = _write_csv(tmp_path / "header.csv", ["offset,gain", "0,0", "1,1"])
    with pytest.raises(ValueError, match="CSV header"):
        load_raman_gain_table(bad_header)

    three_columns = _write_csv(
        tmp_path / "columns.csv",
        [f"# format = {RAMAN_GAIN_TABLE_FORMAT}", "# version = 1", "offsets_thz,normalized_gain", "0,0,1"],
    )
    with pytest.raises(ValueError, match="two columns"):
        load_raman_gain_table(three_columns)

    nonnumeric = _write_csv(
        tmp_path / "nonnumeric.csv",
        [f"# format = {RAMAN_GAIN_TABLE_FORMAT}", "# version = 1", "offsets_thz,normalized_gain", "0,oops"],
    )
    with pytest.raises(ValueError, match="non-numeric"):
        load_raman_gain_table(nonnumeric)

    nonfinite = _write_csv(
        tmp_path / "nonfinite.csv",
        [f"# format = {RAMAN_GAIN_TABLE_FORMAT}", "# version = 1", "offsets_thz,normalized_gain", "0,nan"],
    )
    with pytest.raises(ValueError, match="finite"):
        load_raman_gain_table(nonfinite)

    bad_metadata = _write_csv(
        tmp_path / "metadata.csv",
        [f"# format = {RAMAN_GAIN_TABLE_FORMAT}", "# version = 1", "# broken", "offsets_thz,normalized_gain", "0,0"],
    )
    with pytest.raises(ValueError, match="metadata line"):
        load_raman_gain_table(bad_metadata)

    duplicate = _write_csv(
        tmp_path / "duplicate.csv",
        [f"# format = {RAMAN_GAIN_TABLE_FORMAT}", "# version = 1", "# version = 1", "offsets_thz,normalized_gain"],
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_raman_gain_table(duplicate)


def test_peak_normalisation_warning(tmp_path: Path) -> None:
    """峰值明显偏离 1 时必须 warning，提示可能是绝对增益误填。"""
    path = _write_json(tmp_path / "unnormalized.json", _payload(normalized_gain=[0.0, 0.5, 0.0]))
    with pytest.warns(UserWarning, match="normalized_gain peak"):
        load_raman_gain_table(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        load_raman_gain_table(_write_json(tmp_path / "normalized.json", _payload()))
    assert not [item for item in caught if "normalized_gain peak" in str(item.message)]


# ---------------------------------------------------------------------------
# 插值/外插、温度与参考波长语义
# ---------------------------------------------------------------------------


def test_interpolation_inside_and_zero_outside(tmp_path: Path) -> None:
    """表内线性插值，表外与负频移置零，节点值精确复现。"""
    model = RamanGainModel.from_file(_write_json(tmp_path / "table.json", _payload()))
    shifts = np.array([-1e12, 0.0, 6.6e12, 13.2e12, 26.4e12, 30e12])
    np.testing.assert_allclose(model.gain_shape(shifts), np.array([0.0, 0.0, 0.5, 1.0, 0.0, 0.0]))


def test_temperature_changes_ase_but_not_on_off_gain(tmp_path: Path) -> None:
    """温度只影响 ASE 的声子因子，不改变 on-off 增益。"""
    cold = load_raman_gain_table(_write_json(tmp_path / "cold.json", _payload(temperature_k=300.0)))
    hot = load_raman_gain_table(_write_json(tmp_path / "hot.json", _payload(temperature_k=323.15)))
    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )
    pump_frequency = C_LIGHT / (pump.wavelength_nm * 1e-9)
    frequencies = np.array([pump_frequency - 13.2e12])
    bandwidths = np.array([64e9])

    cold_model = cold.to_model()
    hot_model = hot.to_model()
    np.testing.assert_array_equal(
        cold_model.on_off_gain(frequencies, [pump], 100e3),
        hot_model.on_off_gain(frequencies, [pump], 100e3),
    )
    cold_ase = cold_model.ase_w(frequencies, bandwidths, [pump], 100e3)
    hot_ase = hot_model.ase_w(frequencies, bandwidths, [pump], 100e3)
    assert hot_ase[0] > cold_ase[0]
    assert hot_ase[0] / cold_ase[0] == pytest.approx(1.023, rel=0.01)


def test_reference_wavelength_is_provenance_only(tmp_path: Path) -> None:
    """``reference_wavelength_nm`` 只溯源，换一个值不得改变增益或 ASE。"""
    first = RamanGainModel.from_file(
        _write_json(tmp_path / "1550.json", _payload(reference_wavelength_nm=1550.0))
    )
    second = RamanGainModel.from_file(
        _write_json(tmp_path / "1310.json", _payload(reference_wavelength_nm=1310.0))
    )
    frequencies = C_LIGHT / np.linspace(1530e-9, 1565e-9, 9)
    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )
    np.testing.assert_array_equal(
        first.on_off_gain(frequencies, [pump], 100e3),
        second.on_off_gain(frequencies, [pump], 100e3),
    )
    np.testing.assert_array_equal(
        first.gain_shape(frequencies),
        second.gain_shape(frequencies),
    )


# ---------------------------------------------------------------------------
# 溯源与 override
# ---------------------------------------------------------------------------


def test_provenance_survives_from_file(tmp_path: Path) -> None:
    """fiber/source/note/reference_wavelength_nm 不能静默丢失。"""
    model = RamanGainModel.from_file(_write_json(tmp_path / "table.json", _payload()))
    assert model.source_table is not None
    assert model.fiber == "synthetic-minimal"
    assert model.source == "synthetic fixture, not measured"
    assert "shift" in (model.note or "")
    assert model.reference_wavelength_nm == 1550.0
    assert model.table_metadata["reference_wavelength_nm"] == 1550.0
    assert model.table_metadata["fiber"] == "synthetic-minimal"


def test_calibration_override_warns_and_is_recorded(tmp_path: Path) -> None:
    """覆盖文件标定字段必须 warning，并记录在模型上。"""
    path = _write_json(tmp_path / "table.json", _payload())
    with pytest.warns(UserWarning, match="overriding calibration"):
        model = RamanGainModel.from_file(path, temperature_k=310.0)
    assert model.temperature_k == 310.0
    assert model.calibration_overrides == {"temperature_k": 310.0}

    with pytest.raises(ValueError, match="strict_calibration_override"):
        RamanGainModel.from_file(path, strict_calibration_override=True, temperature_k=310.0)


def test_explicit_default_calibration_values_override_the_file_and_warn(tmp_path: Path) -> None:
    """显式传入等于默认值的标定也要覆盖文件并告警，不能静默输给文件。"""
    path = _write_json(
        tmp_path / "table.json",
        _payload(temperature_k=310.0, peak_gain_coefficient_m_per_w=5e-14),
    )
    with pytest.warns(UserWarning, match="overriding calibration"):
        model = resolve_raman_gain_model(
            path,
            temperature_k=300.0,
            peak_gain_coefficient_m_per_w=6e-14,
        )
    assert model.temperature_k == 300.0
    assert model.peak_gain_coefficient_m_per_w == 6e-14
    assert model.calibration_overrides == {
        "temperature_k": 300.0,
        "peak_gain_coefficient_m_per_w": 6e-14,
    }


def test_omitted_calibration_values_keep_the_file_values_without_warning(tmp_path: Path) -> None:
    """不传（``None``）或显式传 ``None`` 都表示沿用文件标定，不告警。"""
    path = _write_json(
        tmp_path / "table.json",
        _payload(temperature_k=310.0, peak_gain_coefficient_m_per_w=5e-14),
    )
    for kwargs in ({}, {"temperature_k": None, "peak_gain_coefficient_m_per_w": None}):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = resolve_raman_gain_model(path, **kwargs)
        assert model.temperature_k == 310.0
        assert model.peak_gain_coefficient_m_per_w == 5e-14
        assert model.calibration_overrides == {}
        assert not [item for item in caught if issubclass(item.category, UserWarning)]


def test_model_forwards_an_explicit_default_calibration_override(tmp_path: Path) -> None:
    """模型层必须把显式默认值继续当作 override，而不是在自己的默认参数处吞掉。"""
    path = _write_json(
        tmp_path / "table.json",
        _payload(temperature_k=310.0, peak_gain_coefficient_m_per_w=5e-14),
    )
    with pytest.warns(UserWarning, match="overriding calibration"):
        model = make_model(
            "pcfm_h3_raman",
            CASE_C,
            raman_gain_model=path,
            raman_temperature_k=300.0,
            raman_peak_gain_coefficient_m_per_w=6e-14,
        )
    assert model._raman_gain_model.temperature_k == 300.0
    assert model._raman_gain_model.peak_gain_coefficient_m_per_w == 6e-14
    assert model._raman_gain_model.calibration_overrides == {
        "temperature_k": 300.0,
        "peak_gain_coefficient_m_per_w": 6e-14,
    }


def test_invalid_model_overrides_are_rejected(tmp_path: Path) -> None:
    """未知 override 与 gain_model/table override 必须报错。"""
    path = _write_json(tmp_path / "table.json", _payload())
    with pytest.raises(ValueError, match="unsupported"):
        RamanGainModel.from_file(path, bogus=1)
    with pytest.raises(ValueError, match="gain_model and table"):
        RamanGainModel.from_file(path, gain_model="gaussian")
    with pytest.raises(ValueError, match="gain_model and table"):
        RamanGainModel.from_file(path, table=(np.array([0.0]), np.array([1.0])))


def test_runtime_overrides_are_accepted(tmp_path: Path) -> None:
    """运行时参数（ASE 步数、光子因子开关）可以覆盖。"""
    path = _write_json(tmp_path / "table.json", _payload())
    model = RamanGainModel.from_file(path, ase_steps=32, include_photon_energy_factor=False)
    assert model.ase_steps == 32
    assert model.include_photon_energy_factor is False


# ---------------------------------------------------------------------------
# 下游集成：实例/路径透传
# ---------------------------------------------------------------------------


def test_model_file_instance_and_memory_table_agree_bit_for_bit(tmp_path: Path) -> None:
    """PCFM/H3/Hybrid 用 from_file 实例时与内存表路径逐位一致。"""
    _, path = _write_measured_style(tmp_path / "measured.json")
    instance = RamanGainModel.from_file(path)
    # 用加载后的 Hz 数组构造内存表：THz↔Hz 的十进制往返可能有个别 1 ulp
    # 差异，这里要比较的是“文件入口 vs 内存入口”的模型分支，而不是浮点
    # 转写本身。逐位一致性由 test_loaded_model_matches_in_memory_table_bit_for_bit
    # 在同一数组上单独验证。
    loaded = load_raman_gain_table(path)
    memory_kwargs = {
        "raman_gain_model": "table",
        "raman_gain_table": (loaded.offsets_hz, loaded.normalized_gain),
        "raman_temperature_k": loaded.temperature_k,
        "raman_peak_gain_coefficient_m_per_w": loaded.peak_gain_coefficient_m_per_w,
    }
    shared = {
        "pcfm": {"raman_model": "lumped", "raman_pumps": SIX_PUMPS},
        "pcfm_h3_raman": {"raman_pumps": SIX_PUMPS},
        "hybrid_raman_paper": {"raman_pumps": SIX_PUMPS},
    }
    power = launch_profile(CASE_C_L)
    optical_path = build_path(CASE_C_L)

    for name, options in shared.items():
        file_model = make_model(name, CASE_C_L, raman_gain_model=instance, **options)
        table_model = make_model(name, CASE_C_L, **options, **memory_kwargs)
        assert file_model._raman_gain_model is instance
        assert file_model._raman_gain_model.gain_model == "table"

        file_result = file_model.evaluate_path(optical_path, power)
        table_result = table_model.evaluate_path(optical_path, power)
        assert np.all(np.isfinite(file_result.snr_db))
        for key, values in _result_arrays(file_result).items():
            np.testing.assert_array_equal(values, _result_arrays(table_result)[key], err_msg=f"{name}:{key}")


def test_table_file_path_reaches_the_models(tmp_path: Path) -> None:
    """``raman_gain_model`` 直接给 .json 路径时应与 from_file 实例一致。"""
    _, path = _write_measured_style(tmp_path / "measured.json")
    power = launch_profile(CASE_C_L)
    optical_path = build_path(CASE_C_L)
    options = {"raman_pumps": SIX_PUMPS}
    from_path = make_model("pcfm_h3_raman", CASE_C_L, raman_gain_model=path, **options)
    from_instance = make_model(
        "pcfm_h3_raman", CASE_C_L, raman_gain_model=RamanGainModel.from_file(path), **options
    )
    result_path = from_path.evaluate_path(optical_path, power)
    result_instance = from_instance.evaluate_path(optical_path, power)
    for key, values in _result_arrays(result_path).items():
        np.testing.assert_array_equal(values, _result_arrays(result_instance)[key])


def test_paper_span_solver_accepts_file_and_path(tmp_path: Path) -> None:
    """``RamanPaperSpanSolver`` 直接接受 from_file 实例与路径。"""
    _, path = _write_measured_style(tmp_path / "measured.json")
    instance = RamanGainModel.from_file(path)
    frequencies = C_LIGHT / np.linspace(1530e-9, 1565e-9, 9)
    bandwidths = np.full(frequencies.size, 64e9)
    a_eff = np.full(frequencies.size, 80e-12)
    alpha = np.full(frequencies.size, 0.2 * np.log(10.0) / 10.0 / 1e3)
    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )
    pump_alpha = np.array([0.25 * np.log(10.0) / 10.0 / 1e3])
    pump_a_eff = np.array([80e-12])

    def solve(gain_model):
        solver = RamanPaperSpanSolver(
            frequencies,
            bandwidths,
            [pump],
            100e3,
            alpha,
            a_eff,
            pump_alpha,
            pump_a_eff,
            gain_model,
            n_steps=16,
        )
        return solver.solve(np.full(frequencies.size, 1e-3))

    by_instance = solve(instance)
    by_path = solve(path)
    np.testing.assert_array_equal(by_instance.power_out_w, by_path.power_out_w)
    np.testing.assert_array_equal(by_instance.pump_power_out_w, by_path.pump_power_out_w)


def test_resolve_returns_instance_and_rejects_table_combination(tmp_path: Path) -> None:
    """``resolve_raman_gain_model`` 对实例原样返回，实例/路径与表互斥。"""
    _, path = _write_measured_style(tmp_path / "measured.json")
    instance = RamanGainModel.from_file(path)
    assert resolve_raman_gain_model(instance) is instance
    with pytest.raises(ValueError, match="table must not be given"):
        resolve_raman_gain_model(instance, table=(np.array([0.0, 1.0]), np.array([1.0, 1.0])))
    with pytest.raises(ValueError, match="must not be given together"):
        resolve_raman_gain_model(path, table=(np.array([0.0, 1.0]), np.array([1.0, 1.0])))


def test_unsupported_models_give_clear_errors(tmp_path: Path) -> None:
    """文件入口不支持的模型必须给出清晰错误，而不是静默退回 Gaussian。"""
    _, path = _write_measured_style(tmp_path / "measured.json")
    instance = RamanGainModel.from_file(path)
    with pytest.raises(TypeError, match="File-backed tables are only supported"):
        make_model("buglia2024_ra_gn", CASE_C, raman_gain_model=instance)
    with pytest.raises(ValueError, match="does not accept the table file"):
        make_model("buglia2024_ra_gn", CASE_C, raman_gain_model=str(path))
    with pytest.raises(ValueError, match="does not accept the table file"):
        RamanGainModel(gain_model=str(path))


# ---------------------------------------------------------------------------
# 默认 Gaussian/CZ 逐位不变
# ---------------------------------------------------------------------------


def test_default_gaussian_and_cz_shapes_are_bit_for_bit_unchanged() -> None:
    """新增文件分支不得改变默认 Gaussian/CZ 的解析形状。"""
    shifts = np.linspace(0.0, 30e12, 2001)
    gaussian = RamanGainModel()
    explicit = RamanGainModel(gain_model="gaussian")
    expected_gaussian = np.where(
        shifts > 0,
        np.exp(-0.5 * ((shifts - 13.2e12) / 2.12e12) ** 2),
        0.0,
    )
    np.testing.assert_array_equal(gaussian.gain_shape(shifts), explicit.gain_shape(shifts))
    np.testing.assert_array_equal(gaussian.gain_shape(shifts), expected_gaussian)

    linear = RamanGainModel(gain_model="linear")
    expected_linear = np.where((shifts > 0) & (shifts <= 15.0e12), shifts / 13.2e12, 0.0)
    np.testing.assert_array_equal(linear.gain_shape(shifts), expected_linear)


def test_default_model_paths_are_unaffected_by_file_entry() -> None:
    """默认（无表）PCFM/H3 的输出与显式 Gaussian 逐位一致。"""
    power = launch_profile(CASE_C)
    optical_path = build_path(CASE_C)
    for name in ("pcfm", "pcfm_h3_raman", "hybrid_raman_paper"):
        implicit = make_model(name, CASE_C)
        explicit = make_model(name, CASE_C, raman_gain_model="gaussian")
        implicit_result = implicit.evaluate_path(optical_path, power)
        explicit_result = explicit.evaluate_path(optical_path, power)
        for key, values in _result_arrays(implicit_result).items():
            np.testing.assert_array_equal(values, _result_arrays(explicit_result)[key], err_msg=f"{name}:{key}")
