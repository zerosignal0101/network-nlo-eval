"""实测拉曼谱标定流程（任务 AG）的验收测试。

覆盖：

* 只有频移/增益两列的 CSV/JSON 输入经脚本校验、标定、落盘后可由
  ``RamanGainTable``/``RamanGainModel.from_file`` 逐位读回；
* 完整增益表文件可以原样作为输入，provenance 不丢失；
* 非单调、负值、非有限、长度不匹配、全零、未归一化峰值等非法表被拒绝；
* 峰值缩放必须显式 opt-in，并记录原始峰值；
* 默认 ``gaussian``/``linear`` 预设逐位不变；
* 形状/峰值/斜率对照的度量与单跨灵敏度路径可运行。

所有“谱”都是测试内生成的合成数据，来源字段明确写
``synthetic fixture for tests, not measured``；本文件不读取 ``data/``，也不
访问网络。真实标定数据的接入方式见 ``docs/h3-raman-calibration.md``。
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_table import (
    RamanGainTable,
    load_raman_gain_table,
    save_raman_gain_table,
)
from network_nlo_eval.network.elements import RamanPumpConfig

ROOT = Path(__file__).resolve().parents[1]


def _load_workflow() -> ModuleType:
    """按路径加载 ``scripts/raman_spectrum_calibration.py``（scripts 不是包）。"""
    path = ROOT / "scripts" / "raman_spectrum_calibration.py"
    spec = importlib.util.spec_from_file_location("raman_spectrum_calibration", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


WORKFLOW = _load_workflow()

SOURCE = "synthetic fixture for tests, not measured"
FIBER = "synthetic-test-fiber"


def _synthetic_spectrum(
    *,
    points: int = 301,
    peak_scale: float = 1.0,
    ripple: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """合成“实测风格”谱：Gaussian + 可选纹波，峰值归一到 ``peak_scale``。"""
    offsets_thz = np.linspace(0.0, 30.0, points)
    gains = np.exp(-0.5 * ((offsets_thz - 13.2) / 2.12) ** 2)
    if ripple:
        gains = gains * (1.0 + ripple * np.sin(2.0 * np.pi * offsets_thz / 1.5))
    gains = np.maximum(gains, 0.0)
    gains = gains / float(np.max(gains)) * peak_scale
    return offsets_thz, gains


def _write_csv(
    path: Path,
    offsets_thz: np.ndarray,
    gains: np.ndarray,
    *,
    source: str | None = SOURCE,
    fiber: str | None = FIBER,
) -> Path:
    """写带 ``# key = value`` 元数据的两列 CSV。"""
    lines = []
    if fiber is not None:
        lines.append(f"# fiber = {fiber}")
    if source is not None:
        lines.append(f"# source = {source}")
    lines.extend(
        [
            "# note = synthetic fixture for pipeline tests",
            "# temperature_k = 300.0",
            "# peak_gain_coefficient_m_per_w = 6e-14",
            "# reference_wavelength_nm = 1550.0",
            "# reference_a_eff_um2 = 80.0",
            "shift_thz,normalized_gain",
        ]
    )
    lines.extend(f"{float(offset)!r},{float(gain)!r}" for offset, gain in zip(offsets_thz, gains, strict=True))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_json(
    path: Path,
    offsets_thz: np.ndarray,
    gains: np.ndarray,
    *,
    source: str | None = SOURCE,
) -> Path:
    """写最小 JSON 输入。"""
    payload: dict[str, object] = {
        "offsets_thz": [float(value) for value in offsets_thz],
        "normalized_gain": [float(value) for value in gains],
        "fiber": FIBER,
        "note": "synthetic fixture for pipeline tests",
        "temperature_k": 300.0,
        "peak_gain_coefficient_m_per_w": 6e-14,
        "reference_wavelength_nm": 1550.0,
        "reference_a_eff_um2": 80.0,
    }
    if source is not None:
        payload["source"] = source
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _calibrate(path: Path, **config_kwargs: object) -> object:
    """加载并标定一个输入文件，返回 ``CalibrationResult``。"""
    spectrum = WORKFLOW.load_measured_spectrum(path)
    config = WORKFLOW.CalibrationConfig(source=SOURCE, **config_kwargs)
    return WORKFLOW.calibrate_spectrum(spectrum, config)


# ---------------------------------------------------------------------------
# 往返与脚本入口
# ---------------------------------------------------------------------------


def test_csv_input_round_trips_through_the_script(tmp_path: Path) -> None:
    """CSV 输入 → 标定表 JSON → ``load_raman_gain_table`` 保留数组与 provenance。"""
    offsets_thz, gains = _synthetic_spectrum(points=121)
    input_path = _write_csv(tmp_path / "measured.csv", offsets_thz, gains)
    result = _calibrate(input_path)
    assert isinstance(result, WORKFLOW.CalibrationResult)

    output = tmp_path / "table.json"
    save_raman_gain_table(result.table, output)
    loaded = load_raman_gain_table(output)

    np.testing.assert_array_equal(loaded.offsets_hz, result.table.offsets_hz)
    np.testing.assert_array_equal(loaded.normalized_gain, result.table.normalized_gain)
    assert loaded.fiber == FIBER
    assert loaded.source == SOURCE
    assert loaded.temperature_k == 300.0
    assert loaded.peak_gain_coefficient_m_per_w == 6e-14
    assert loaded.reference_wavelength_nm == 1550.0
    assert loaded.reference_a_eff_m2 == 80e-12
    assert result.input_peak_gain == pytest.approx(1.0)
    assert result.peak_scale == 1.0
    assert result.normalized_peak is False


def test_csv_and_json_inputs_produce_the_same_table(tmp_path: Path) -> None:
    """同一份数字的 CSV 与 JSON 输入应得到逐位相同的表。"""
    offsets_thz, gains = _synthetic_spectrum(points=61)
    csv_result = _calibrate(_write_csv(tmp_path / "measured.csv", offsets_thz, gains))
    json_result = _calibrate(_write_json(tmp_path / "measured.json", offsets_thz, gains))

    np.testing.assert_array_equal(csv_result.table.offsets_hz, json_result.table.offsets_hz)
    np.testing.assert_array_equal(csv_result.table.normalized_gain, json_result.table.normalized_gain)
    assert csv_result.table.metadata == json_result.table.metadata


def test_full_gain_table_file_is_accepted_as_input(tmp_path: Path) -> None:
    """完整 ``RamanGainTable`` 文件可以直接作为输入，标定字段不丢失。"""
    offsets_thz, gains = _synthetic_spectrum(points=61)
    table = RamanGainTable(
        offsets_hz=offsets_thz * 1e12,
        normalized_gain=gains,
        temperature_k=310.0,
        peak_gain_coefficient_m_per_w=5e-14,
        reference_wavelength_nm=1310.0,
        reference_a_eff_m2=70e-12,
        fiber=FIBER,
        source=SOURCE,
        note="table input",
    )
    path = tmp_path / "full_table.json"
    save_raman_gain_table(table, path)

    spectrum = WORKFLOW.load_measured_spectrum(path)
    assert spectrum.metadata["temperature_k"] == 310.0
    assert spectrum.metadata["peak_gain_coefficient_m_per_w"] == 5e-14
    assert spectrum.metadata["reference_wavelength_nm"] == 1310.0
    assert spectrum.metadata["reference_a_eff_um2"] == 70.0

    result = WORKFLOW.calibrate_spectrum(spectrum, WORKFLOW.CalibrationConfig())
    assert result.table.temperature_k == 310.0
    assert result.table.peak_gain_coefficient_m_per_w == 5e-14
    assert result.table.reference_wavelength_nm == 1310.0
    assert result.table.reference_a_eff_m2 == 70e-12


def test_script_output_is_loadable_by_raman_gain_model(tmp_path: Path) -> None:
    """``main`` 写出的表必须能被 ``RamanGainModel.from_file`` 直接加载。"""
    offsets_thz, gains = _synthetic_spectrum(points=121, ripple=0.03)
    input_path = _write_csv(tmp_path / "measured.csv", offsets_thz, gains)
    output = tmp_path / "out" / "table.json"
    report = tmp_path / "out" / "report.json"

    assert WORKFLOW.main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output),
            "--report",
            str(report),
            "--source",
            SOURCE,
            "--fiber",
            FIBER,
        ]
    ) == 0

    model = RamanGainModel.from_file(output)
    assert model.gain_model == "table"
    assert model.fiber == FIBER
    assert model.source == SOURCE
    assert model.reference_wavelength_nm == 1550.0
    assert model.table_metadata["reference_wavelength_nm"] == 1550.0
    assert model.source_table is not None

    report_payload = json.loads(report.read_text(encoding="utf-8"))
    assert report_payload["generated_by"] == "scripts/raman_spectrum_calibration.py"
    assert report_payload["table"]["path"] == str(output)
    assert report_payload["table"]["provenance"]["source"] == SOURCE
    assert report_payload["span_sensitivity"] is None
    for name in ("gaussian", "linear"):
        assert name in report_payload["comparison"]["deviation_vs_measured"]
    assert len(report_payload["input"]["sha256"]) == 64


# ---------------------------------------------------------------------------
# 非法输入
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("offsets", "gains", "match"),
    [
        ([0.0, 10.0, 9.0], [0.0, 1.0, 0.5], "strictly increasing"),
        ([0.0, 10.0, 10.0], [0.0, 1.0, 0.5], "strictly increasing"),
        ([-1.0, 0.0, 10.0], [0.0, 1.0, 0.5], "non-negative"),
        ([0.0, 10.0, 20.0], [0.0, -1.0, 0.5], "non-negative"),
        ([0.0, 10.0, 20.0], [0.0, 0.0, 0.0], "positive gain"),
        ([0.0, float("nan"), 20.0], [0.0, 1.0, 0.5], "finite"),
        ([0.0, 10.0, 20.0], [0.0, 1.0], "same length"),
        ([5.0], [1.0], "at least two"),
    ],
)
def test_invalid_spectrum_arrays_are_rejected(
    tmp_path: Path,
    offsets: list[float],
    gains: list[float],
    match: str,
) -> None:
    """形状、单调性、符号、有限性与长度错误都必须拒绝。"""
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"offsets_thz": offsets, "normalized_gain": gains, "source": SOURCE}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=match):
        WORKFLOW.load_measured_spectrum(path)


def test_unknown_fields_and_bad_header_are_rejected(tmp_path: Path) -> None:
    """未知字段和错误 CSV 表头不能静默忽略。"""
    bad_json = tmp_path / "unknown.json"
    bad_json.write_text(
        json.dumps(
            {
                "offsets_thz": [0.0, 1.0],
                "normalized_gain": [0.0, 1.0],
                "source": SOURCE,
                "peak_shift": 13.2,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown spectrum field"):
        WORKFLOW.load_measured_spectrum(bad_json)

    bad_csv = tmp_path / "bad_header.csv"
    bad_csv.write_text("offset,gain\n0,0\n1,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV header"):
        WORKFLOW.load_measured_spectrum(bad_csv)


def test_unnormalised_peak_requires_explicit_opt_in(tmp_path: Path) -> None:
    """峰值明显偏离 1 时默认拒绝；``normalize_peak=True`` 才缩放并记录原值。"""
    offsets_thz, gains = _synthetic_spectrum(points=121, peak_scale=0.8)
    path = _write_csv(tmp_path / "measured.csv", offsets_thz, gains)

    spectrum = WORKFLOW.load_measured_spectrum(path)
    with pytest.raises(ValueError, match="peak"):
        WORKFLOW.calibrate_spectrum(spectrum, WORKFLOW.CalibrationConfig(source=SOURCE))

    result = WORKFLOW.calibrate_spectrum(
        spectrum,
        WORKFLOW.CalibrationConfig(source=SOURCE, normalize_peak=True),
    )
    assert result.input_peak_gain == pytest.approx(0.8)
    assert result.peak_scale == pytest.approx(1.25)
    assert result.normalized_peak is True
    assert result.table.peak_normalized_gain == pytest.approx(1.0)
    assert "peak-normalized" in (result.table.note or "")


def test_missing_source_is_rejected(tmp_path: Path) -> None:
    """没有来源的谱不得被当成可追溯的标定表。"""
    offsets_thz, gains = _synthetic_spectrum(points=31)
    path = _write_csv(tmp_path / "measured.csv", offsets_thz, gains, source=None)
    spectrum = WORKFLOW.load_measured_spectrum(path)
    with pytest.raises(ValueError, match="provenance source"):
        WORKFLOW.calibrate_spectrum(spectrum, WORKFLOW.CalibrationConfig())


# ---------------------------------------------------------------------------
# 与预设对照
# ---------------------------------------------------------------------------


def _calibrated_table(**kwargs: object) -> RamanGainTable:
    """构造一份 Gaussian 形状的已标定表（合成测试夹具，不是实测）。"""
    offsets_thz, gains = _synthetic_spectrum(points=301)
    table = RamanGainTable(
        offsets_hz=offsets_thz * 1e12,
        normalized_gain=gains,
        temperature_k=300.0,
        peak_gain_coefficient_m_per_w=6e-14,
        reference_wavelength_nm=1550.0,
        reference_a_eff_m2=80e-12,
        fiber=FIBER,
        source=SOURCE,
        note="synthetic fixture, not measured",
        **kwargs,
    )
    return table


def test_comparison_reports_shape_peak_and_slope() -> None:
    """Gaussian 形状的表对 gaussian 预设偏差小、对 linear 偏差大；峰值/斜率可读。"""
    table = _calibrated_table()
    comparison = WORKFLOW.compare_with_presets(table)

    shapes = comparison["shapes"]
    assert shapes["gaussian"]["peak_shift_thz"] == pytest.approx(13.2, abs=0.2)
    assert shapes["linear"]["peak_shift_thz"] == pytest.approx(15.0, abs=0.2)
    assert shapes["measured"]["peak_shift_thz"] == pytest.approx(13.2, abs=0.2)
    # 峰值 6e-14 / 80e-12 = 0.75 1/(W km)，0 到 10 THz 的斜率应落在 Semrau 量级。
    assert shapes["gaussian"]["secant_slope_1_per_w_km_thz"] == pytest.approx(0.024, rel=0.05)
    assert shapes["linear"]["secant_slope_1_per_w_km_thz"] == pytest.approx(0.057, rel=0.05)

    deviation = comparison["deviation_vs_measured"]
    assert deviation["gaussian"]["rmse"] < 0.02
    assert deviation["linear"]["rmse"] > 10.0 * deviation["gaussian"]["rmse"]
    assert deviation["gaussian"]["correlation"] > 0.99
    assert comparison["slope_ratio_vs_measured"]["gaussian"] == pytest.approx(1.0, abs=0.1)
    assert comparison["slope_ratio_vs_measured"]["linear"] > 2.0
    assert comparison["reference"]["mean_shape_band_thz"] == pytest.approx(21.0)


def test_default_presets_are_bit_for_bit_unchanged() -> None:
    """脚本/对照不得改变默认 Gaussian/CZ 的解析形状、增益与 ASE。"""
    shifts = np.linspace(0.0, 30e12, 2001)
    expected_gaussian = np.where(
        shifts > 0,
        np.exp(-0.5 * ((shifts - 13.2e12) / 2.12e12) ** 2),
        0.0,
    )
    np.testing.assert_array_equal(RamanGainModel().gain_shape(shifts), expected_gaussian)
    np.testing.assert_array_equal(
        RamanGainModel().gain_shape(shifts),
        RamanGainModel(gain_model="gaussian").gain_shape(shifts),
    )
    expected_linear = np.where((shifts > 0) & (shifts <= 15.0e12), shifts / 13.2e12, 0.0)
    np.testing.assert_array_equal(
        RamanGainModel(gain_model="linear").gain_shape(shifts),
        expected_linear,
    )

    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )
    frequencies = C_LIGHT / np.linspace(1530e-9, 1565e-9, 9)
    bandwidths = np.full(frequencies.size, 64e9)
    default = RamanGainModel()
    explicit = RamanGainModel(gain_model="gaussian")
    np.testing.assert_array_equal(
        default.on_off_gain(frequencies, [pump], 100e3),
        explicit.on_off_gain(frequencies, [pump], 100e3),
    )
    np.testing.assert_array_equal(
        default.ase_w(frequencies, bandwidths, [pump], 100e3),
        explicit.ase_w(frequencies, bandwidths, [pump], 100e3),
    )

    # 跑一次对照后默认模型仍逐位不变。
    WORKFLOW.compare_with_presets(_calibrated_table())
    np.testing.assert_array_equal(RamanGainModel().gain_shape(shifts), expected_gaussian)


# ---------------------------------------------------------------------------
# 可选单跨灵敏度
# ---------------------------------------------------------------------------


def test_span_sensitivity_runs_on_a_small_single_span() -> None:
    """小算例上 H3 与 numerical_dra 的跨损/ASE 灵敏度路径可运行且数值有限。"""
    table = _calibrated_table()
    config = WORKFLOW.SpanSensitivityConfig(
        num_channels=8,
        span_length_km=20.0,
        models=("h3", "numerical_dra"),
    )
    result = WORKFLOW.run_span_sensitivity(table, config)

    assert set(result["models"]) == {"h3", "numerical_dra"}
    for model_name in ("h3", "numerical_dra"):
        by_spectrum = result["models"][model_name]["by_spectrum"]
        assert set(by_spectrum) == {"measured", "gaussian"}
        for summary in by_spectrum.values():
            assert np.isfinite(summary["loss_db_mean"])
            assert summary["loss_db_width"] >= 0.0
            assert np.isfinite(summary["raman_ase_w_total"])
            assert summary["raman_ase_w_total"] >= 0.0
            assert len(summary["loss_db_per_channel"]) == 8
        delta = result["models"][model_name]["delta_measured_minus_gaussian"]
        assert np.isfinite(delta["loss_db_mean"])
        assert np.isfinite(delta["loss_db_max_abs"])
