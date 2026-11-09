"""实测拉曼谱 → 标定表 → 模型对照的可复现流程。

本脚本补上仓库此前缺失的一环：把一份**实测归一化拉曼增益谱**（频移 +
归一化增益）校验、落盘为 :class:`~network_nlo_eval.models.raman_table.RamanGainTable`
JSON，并与 :class:`~network_nlo_eval.models.raman.RamanGainModel` 的
``gaussian``/``linear`` 预设做形状、峰值位置与斜率对照。可选地在固定单跨
算例上比较 ``pcfm_h3_raman`` 与 ``numerical_dra`` 的跨损/拉曼 ASE 对谱标定的
灵敏度。

输入格式
--------
CSV（两列，可选 ``# key = value`` 元数据头）::

    # fiber = SMF-28-like
    # source = vendor report 2026-11, not bundled
    # temperature_k = 300.0
    # peak_gain_coefficient_m_per_w = 6e-14
    # reference_wavelength_nm = 1550.0
    # reference_a_eff_um2 = 80.0
    shift_thz,normalized_gain
    0.0,0.0
    1.0,0.08
    13.2,1.0

JSON::

    {
      "fiber": "SMF-28-like",
      "source": "vendor report 2026-11, not bundled",
      "temperature_k": 300.0,
      "peak_gain_coefficient_m_per_w": 6e-14,
      "reference_wavelength_nm": 1550.0,
      "reference_a_eff_um2": 80.0,
      "offsets_thz": [0.0, 1.0, 13.2],
      "normalized_gain": [0.0, 0.08, 1.0]
    }

``offsets_thz`` 与 ``shift_thz`` 等价；也可以直接给本仓库的完整增益表
（``format = network_nlo_eval.raman_gain_table``），此时标定字段从表里读取。

输出
----
``--output`` 写纯增益表 JSON，可直接交给
``RamanGainModel.from_file(path)`` 或模型层的 ``raman_gain_model=path``。
``--report`` 写对照报告：输入指纹、provenance 来源、gaussian/linear 形状
偏差、峰值频移、斜率，以及可选的单跨灵敏度。

约束
----
* 不伪造实测数据：脚本不内置任何谱，``--source`` 缺失且输入也没有来源时
  直接拒绝。仓库内测试只使用明确标注 ``synthetic fixture, not measured``
  的合成数据。
* 峰值偏离 1 超过 ``--peak-tolerance`` 时默认拒绝；只有显式
  ``--normalize-peak`` 才会缩放，并把原始峰值写进 provenance。
* 不依赖网络、``data/`` 或 ``results/``；所有输入都必须由调用方显式给出。

运行::

    PYTHONPATH=src python scripts/raman_spectrum_calibration.py \
        --input measured.csv \
        --output results/raman/measured_gain_table.json \
        --report results/raman/measured_calibration.json \
        --source "vendor report 2026-11" --fiber SMF-28-like

单跨灵敏度（可选，较慢）::

    PYTHONPATH=src python scripts/raman_spectrum_calibration.py \
        --input measured.csv --output /tmp/table.json \
        --source "vendor report 2026-11" --span-sensitivity both
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import trapezoid

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.numba_kernels import _calc_aeff_dynamic, _calc_fiber_attenuation_npm
from network_nlo_eval.models.numerical_dra import NumericalDRASpanSolver
from network_nlo_eval.models.raman import (
    C_LIGHT,
    DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
    RamanGainModel,
)
from network_nlo_eval.models.raman_table import (
    RAMAN_GAIN_TABLE_FORMAT,
    RAMAN_GAIN_TABLE_UNITS,
    RamanGainTable,
    load_raman_gain_table,
    save_raman_gain_table,
)
from network_nlo_eval.network.elements import RamanPumpConfig

#: 与 ``RamanGainTable`` 的峰值归一化范围一致。
DEFAULT_PEAK_TOLERANCE = 0.02

#: 归一化谱允许的输入列名（CSV 表头 / JSON 键）。
_OFFSET_KEYS = ("offsets_thz", "shift_thz")
_GAIN_KEYS = ("normalized_gain",)

#: 输入可携带的溯源字段。``units`` 只做显式校验，不改变解释。
_METADATA_KEYS = (
    "fiber",
    "source",
    "note",
    "temperature_k",
    "peak_gain_coefficient_m_per_w",
    "reference_wavelength_nm",
    "reference_a_eff_um2",
)
_NUMERIC_METADATA_KEYS = (
    "temperature_k",
    "peak_gain_coefficient_m_per_w",
    "reference_wavelength_nm",
    "reference_a_eff_um2",
)
_TEXT_METADATA_KEYS = ("fiber", "source", "note")

#: THz → Hz（与 raman_table 保持一致）。
_THZ_TO_HZ = 1e12

#: 斜率的固定参照带宽：取 0 到该频移的割线，便于跨谱比较。
SLOPE_REFERENCE_THZ = 10.0

#: 平均形状的参照带宽，覆盖 C+L 的典型 ISRS 倾斜范围。
MEAN_SHAPE_REFERENCE_THZ = 21.0


@dataclass(frozen=True)
class MeasuredSpectrum:
    """一份已校验的输入谱（不假定它已经是 RamanGainTable 文件）。

    Attributes
    ----------
        offsets_hz: ``(K,)`` 泵浦-信号频移 (Hz)，非负、严格递增。
        normalized_gain: ``(K,)`` 无量纲归一化增益，非负且至少一个正值。
        metadata: 输入文件携带的溯源/标定字段，键名与增益表一致。
    """

    offsets_hz: np.ndarray
    normalized_gain: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """校验形状、单调性、非负与有限性。"""
        offsets = np.array(self.offsets_hz, dtype=np.float64, copy=True)
        gains = np.array(self.normalized_gain, dtype=np.float64, copy=True)
        if offsets.ndim != 1 or gains.ndim != 1:
            raise ValueError("measured spectrum offsets and gains must be one-dimensional.")
        if offsets.size != gains.size:
            raise ValueError(
                "measured spectrum offsets and gains must have the same length, "
                f"got {offsets.size} and {gains.size}."
            )
        if offsets.size < 2:
            raise ValueError("measured spectrum must contain at least two points.")
        if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(gains)):
            raise ValueError("measured spectrum values must be finite.")
        if np.any(offsets < 0.0):
            raise ValueError("measured spectrum offsets must be non-negative.")
        if np.any(np.diff(offsets) <= 0.0):
            raise ValueError("measured spectrum offsets must be strictly increasing.")
        if np.any(gains < 0.0):
            raise ValueError("measured spectrum gains must be non-negative.")
        if not np.any(gains > 0.0):
            raise ValueError("measured spectrum must contain at least one positive gain.")
        offsets.setflags(write=False)
        gains.setflags(write=False)
        object.__setattr__(self, "offsets_hz", offsets)
        object.__setattr__(self, "normalized_gain", gains)
        unknown = sorted(set(self.metadata) - set(_METADATA_KEYS) - {"units", "format", "version"})
        if unknown:
            raise ValueError(f"unknown measured spectrum metadata field(s): {', '.join(unknown)}.")


@dataclass(frozen=True)
class CalibrationConfig:
    """标定表的 provenance 与峰值处理选项；``None`` 表示沿用输入/默认值。

    Attributes
    ----------
        fiber: 光纤型号/批次。
        source: 数据来源；输入与这里都没有时拒绝。
        note: 附加说明。
        temperature_k: 光纤温度 (K)。
        peak_gain_coefficient_m_per_w: 峰值局部增益率 (m/W)。
        reference_wavelength_nm: 峰值系数的标定波长 (nm)。
        reference_a_eff_um2: 标定有效模面积 (µm²)。
        normalize_peak: 是否允许把峰值缩放到 1。
        peak_tolerance: 峰值偏离 1 的容差；超出且未允许缩放时拒绝。
    """

    fiber: str | None = None
    source: str | None = None
    note: str | None = None
    temperature_k: float | None = None
    peak_gain_coefficient_m_per_w: float | None = None
    reference_wavelength_nm: float | None = None
    reference_a_eff_um2: float | None = None
    normalize_peak: bool = False
    peak_tolerance: float = DEFAULT_PEAK_TOLERANCE


@dataclass(frozen=True)
class CalibrationResult:
    """一次标定的结果与审计信息。

    Attributes
    ----------
        table: 已校验、可直接落盘/加载的增益表。
        input_peak_gain: 缩放前的最大归一化增益。
        peak_scale: 实际乘到增益上的比例（未缩放时为 1）。
        normalized_peak: 是否执行了显式峰值缩放。
        provenance_sources: 每个标定字段来自 ``cli``/``input``/``default``。
    """

    table: RamanGainTable
    input_peak_gain: float
    peak_scale: float
    normalized_peak: bool
    provenance_sources: Mapping[str, str]


# ---------------------------------------------------------------------------
# 输入解析
# ---------------------------------------------------------------------------


def load_measured_spectrum(path: str | Path) -> MeasuredSpectrum:
    """读取 CSV/JSON 实测谱并完成基本校验。

    支持两种输入：只有频移/增益两列的原始谱，或本仓库的完整
    ``RamanGainTable`` 文件（``format`` 匹配时按表加载，保留全部标定字段）。

    Args:
        path: ``.csv`` 或 ``.json`` 文件。

    Returns
    -------
        冻结的 :class:`MeasuredSpectrum`。

    Raises
    ------
        FileNotFoundError: 文件不存在。
        ValueError: 扩展名不支持、结构错误、非有限、非单调、负增益或
            全零谱。
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        payload = _load_json_input(resolved)
    elif suffix == ".csv":
        payload = _load_csv_input(resolved)
    else:
        raise ValueError(f"{resolved}: unsupported spectrum extension {suffix!r}; expected '.json' or '.csv'.")
    return _spectrum_from_payload(payload, resolved)


def _load_json_input(path: Path) -> dict[str, Any]:
    """读取 JSON 对象；完整增益表直接走 ``load_raman_gain_table``。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: spectrum input must be UTF-8 text: {error}") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: spectrum JSON must be an object, got {type(payload).__name__}.")
    if payload.get("format") == RAMAN_GAIN_TABLE_FORMAT:
        table = load_raman_gain_table(path)
        return _payload_from_table(table)
    if "format" in payload:
        raise ValueError(
            f"{path}: format must be {RAMAN_GAIN_TABLE_FORMAT!r}, got {payload['format']!r}."
        )
    return payload


def _payload_from_table(table: RamanGainTable) -> dict[str, Any]:
    """把完整增益表还原成输入 payload，便于统一重建。"""
    payload: dict[str, Any] = {
        "offsets_thz": [float(value) for value in table.offsets_thz],
        "normalized_gain": [float(value) for value in table.normalized_gain],
        "temperature_k": table.temperature_k,
        "peak_gain_coefficient_m_per_w": table.peak_gain_coefficient_m_per_w,
        "reference_wavelength_nm": table.reference_wavelength_nm,
        "units": dict(RAMAN_GAIN_TABLE_UNITS),
        "format": table.format,
        "version": table.version,
    }
    if table.reference_a_eff_m2 is not None:
        payload["reference_a_eff_um2"] = table.reference_a_eff_m2 / 1e-12
    for key in _TEXT_METADATA_KEYS:
        value = getattr(table, key)
        if value is not None:
            payload[key] = value
    return payload


def _load_csv_input(path: Path) -> dict[str, Any]:
    """读取 ``# key = value`` 元数据 + 两列数据区。"""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: spectrum input must be UTF-8 text: {error}") from None

    metadata: dict[str, Any] = {}
    rows: list[tuple[float, float]] = []
    header_seen = False
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            _parse_metadata_line(metadata, line, path, line_number)
            continue
        if not header_seen:
            header_seen = True
            columns = tuple(part.strip().lower() for part in line.split(","))
            if len(columns) != 2 or columns[0] not in _OFFSET_KEYS or columns[1] not in _GAIN_KEYS:
                raise ValueError(
                    f"{path}:{line_number}: CSV header must be 'shift_thz,normalized_gain' "
                    f"(or 'offsets_thz,normalized_gain'), got {line!r}."
                )
            continue
        rows.append(_parse_data_line(line, path, line_number))
    if not header_seen:
        raise ValueError(f"{path}: CSV is missing the 'shift_thz,normalized_gain' header.")
    if len(rows) < 2:
        raise ValueError(f"{path}: measured spectrum must contain at least two data rows.")
    metadata["offsets_thz"] = [row[0] for row in rows]
    metadata["normalized_gain"] = [row[1] for row in rows]
    return _finalize_metadata(metadata, path)


def _parse_metadata_line(metadata: dict[str, Any], line: str, path: Path, line_number: int) -> None:
    """解析一行 ``# key = value``，重复键报错。"""
    body = line[1:].strip()
    key, separator, value = body.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ValueError(f"{path}:{line_number}: metadata line must be '# key = value', got {line!r}.")
    if key in metadata:
        raise ValueError(f"{path}:{line_number}: duplicate metadata key {key!r}.")
    metadata[key] = value.strip()


def _parse_data_line(line: str, path: Path, line_number: int) -> tuple[float, float]:
    """解析一行两个有限数值，错误信息带行号。"""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{path}:{line_number}: data row must have two columns, got {line!r}.")
    try:
        values = (float(parts[0]), float(parts[1]))
    except ValueError:
        raise ValueError(f"{path}:{line_number}: data row contains a non-numeric value: {line!r}.") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}:{line_number}: data row values must be finite: {line!r}.")
    return values


def _finalize_metadata(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
    """把 CSV 字符串元数据转成与 JSON 相同的类型。"""
    for key in ("version",):
        if key in metadata:
            try:
                metadata[key] = int(metadata[key])
            except (TypeError, ValueError):
                raise ValueError(f"{path}: {key} must be an integer.") from None
    for key in _NUMERIC_METADATA_KEYS:
        if key in metadata and isinstance(metadata[key], str):
            try:
                metadata[key] = float(metadata[key])
            except ValueError:
                raise ValueError(f"{path}: {key} must be a number, got {metadata[key]!r}.") from None
    if "units" in metadata and isinstance(metadata["units"], str):
        metadata["units"] = _parse_units_text(metadata["units"], path)
    return metadata


def _parse_units_text(text: str, path: Path) -> dict[str, str]:
    """解析 CSV 的 ``units = field=unit, ...``。"""
    units: dict[str, str] = {}
    for item in text.split(","):
        key, separator, unit = item.partition("=")
        key = key.strip()
        if not separator or not key:
            raise ValueError(f"{path}: units must be a comma-separated list of field=unit.")
        units[key] = unit.strip()
    return units


def _spectrum_from_payload(payload: dict[str, Any], path: Path) -> MeasuredSpectrum:
    """把 JSON/CSV payload 校验成 :class:`MeasuredSpectrum`。"""
    offset_key = next((key for key in _OFFSET_KEYS if key in payload), None)
    if offset_key is None:
        raise ValueError(f"{path}: missing offsets; expected one of {', '.join(_OFFSET_KEYS)}.")
    gain_key = next((key for key in _GAIN_KEYS if key in payload), None)
    if gain_key is None:
        raise ValueError(f"{path}: missing normalized_gain.")
    known = {*_OFFSET_KEYS, *_GAIN_KEYS, *_METADATA_KEYS, "units", "format", "version"}
    unknown = sorted(set(payload) - known)
    if unknown:
        raise ValueError(f"{path}: unknown spectrum field(s): {', '.join(unknown)}.")
    if "format" in payload and payload["format"] != RAMAN_GAIN_TABLE_FORMAT:
        raise ValueError(f"{path}: format must be {RAMAN_GAIN_TABLE_FORMAT!r}, got {payload['format']!r}.")
    if "version" in payload and payload["version"] != 1:
        raise ValueError(f"{path}: version must be the integer 1, got {payload['version']!r}.")
    _validate_units(payload.get("units"), path)

    offsets_thz = _as_float_array(payload[offset_key], offset_key, path)
    gains = _as_float_array(payload[gain_key], gain_key, path)
    metadata: dict[str, Any] = {}
    for key in _TEXT_METADATA_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{path}: {key} must be a string, got {type(value).__name__}.")
        metadata[key] = value
    for key in _NUMERIC_METADATA_KEYS:
        value = payload.get(key)
        if value is None:
            continue
        metadata[key] = _as_float(value, key, path)
    try:
        return MeasuredSpectrum(offsets_hz=offsets_thz * _THZ_TO_HZ, normalized_gain=gains, metadata=metadata)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def _validate_units(value: Any, path: Path) -> None:
    """输入若显式声明单位，只接受本流程使用的单位。"""
    if value is None:
        return
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: units must be an object of field -> unit string.")
    expected = {
        "offsets_thz": {"thz", "terahertz"},
        "shift_thz": {"thz", "terahertz"},
        "normalized_gain": {"1", "dimensionless", "unitless", "a.u.", "a.u", "none"},
    }
    for key, declared in value.items():
        if key not in expected:
            continue
        if not isinstance(declared, str) or declared.strip().lower() not in expected[key]:
            raise ValueError(f"{path}: unit of {key} must be THz/dimensionless, got {declared!r}.")


def _as_float(value: Any, key: str, path: Path) -> float:
    """把 JSON/CSV 标量转成有限 float，拒绝布尔。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{path}: {key} must be a number, got {value!r}.")
    try:
        result = float(value)
    except ValueError:
        raise ValueError(f"{path}: {key} must be a number, got {value!r}.") from None
    if not math.isfinite(result):
        raise ValueError(f"{path}: {key} must be finite, got {value!r}.")
    return result


def _as_float_array(value: Any, key: str, path: Path) -> np.ndarray:
    """把列表转成一维 float64 数组。"""
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{path}: {key} must be a list of numbers, got {type(value).__name__}.")
    items: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{path}: {key}[{index}] must be a number, got {item!r}.")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{path}: {key}[{index}] must be finite, got {item!r}.")
        items.append(number)
    return np.asarray(items, dtype=np.float64)


# ---------------------------------------------------------------------------
# 标定表构造
# ---------------------------------------------------------------------------


def calibrate_spectrum(spectrum: MeasuredSpectrum, config: CalibrationConfig) -> CalibrationResult:
    """把输入谱与 provenance 校验成增益表。

    峰值处理是显式 opt-in：默认要求峰值落在 ``1 ± peak_tolerance`` 内；
    超出时拒绝，只有 ``normalize_peak=True`` 才缩放并记录原始峰值。

    Args:
        spectrum: 已校验的输入谱。
        config: provenance 与峰值选项。

    Returns
    -------
        :class:`CalibrationResult`，其中 ``table`` 可直接保存为 JSON/CSV。

    Raises
    ------
        ValueError: 峰值未归一化且未允许缩放、来源缺失、标量非法，或
            ``RamanGainTable`` 自身校验失败。
    """
    if not math.isfinite(config.peak_tolerance) or config.peak_tolerance < 0.0:
        raise ValueError("peak_tolerance must be finite and non-negative.")
    peak = float(np.max(spectrum.normalized_gain))
    gains = np.array(spectrum.normalized_gain, dtype=np.float64, copy=True)
    normalized_peak = False
    peak_scale = 1.0
    auto_note: list[str] = []
    if abs(peak - 1.0) > config.peak_tolerance:
        if not config.normalize_peak:
            raise ValueError(
                f"normalized_gain peak is {peak:.6g}, outside 1 +/- {config.peak_tolerance:g}. "
                "The table is interpreted as a shape multiplied by peak_gain_coefficient_m_per_w; "
                "pass --normalize-peak only after confirming the input really is a normalized shape "
                "whose scale should be removed."
            )
        peak_scale = 1.0 / peak
        gains = gains * peak_scale
        normalized_peak = True
        auto_note.append(
            f"peak-normalized from {peak:.6g} to 1 by scripts/raman_spectrum_calibration.py (--normalize-peak)"
        )

    values: dict[str, tuple[float | None, str]] = {}
    for key in _NUMERIC_METADATA_KEYS:
        cli_value = getattr(config, key)
        input_value = spectrum.metadata.get(key)
        if cli_value is not None:
            values[key] = (float(cli_value), "cli")
        elif input_value is not None:
            values[key] = (float(input_value), "input")
        else:
            values[key] = (None, "default")
    text_values: dict[str, tuple[str | None, str]] = {}
    for key in _TEXT_METADATA_KEYS:
        cli_value = getattr(config, key)
        input_value = spectrum.metadata.get(key)
        if cli_value is not None:
            text_values[key] = (cli_value, "cli")
        elif input_value is not None:
            text_values[key] = (input_value, "input")
        else:
            text_values[key] = (None, "default")

    if text_values["source"][0] is None:
        raise ValueError(
            "provenance source is required: pass --source or put 'source' in the input file. "
            "A gain table without a source cannot be traced and must not be reported as measured."
        )

    note = text_values["note"][0]
    auto_note.append(
        "shift = |f_p - f_s|; normalized shape from an external spectrum, not bundled with the repository"
    )
    if values["peak_gain_coefficient_m_per_w"][0] is None:
        auto_note.append("peak_gain_coefficient_m_per_w left at the RamanGainModel default")
    if values["reference_wavelength_nm"][0] is None:
        auto_note.append("reference_wavelength_nm left at the default 1550 nm (provenance only)")
    if text_values["fiber"][0] is None:
        auto_note.append("fiber not specified")
    combined_note = "; ".join([item for item in [note, *auto_note] if item])

    table = RamanGainTable(
        offsets_hz=spectrum.offsets_hz,
        normalized_gain=gains,
        temperature_k=300.0 if values["temperature_k"][0] is None else values["temperature_k"][0],
        peak_gain_coefficient_m_per_w=(
            DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W
            if values["peak_gain_coefficient_m_per_w"][0] is None
            else values["peak_gain_coefficient_m_per_w"][0]
        ),
        reference_wavelength_nm=(
            1550.0
            if values["reference_wavelength_nm"][0] is None
            else values["reference_wavelength_nm"][0]
        ),
        reference_a_eff_m2=(
            None
            if values["reference_a_eff_um2"][0] is None
            else values["reference_a_eff_um2"][0] * 1e-12
        ),
        fiber=text_values["fiber"][0],
        source=text_values["source"][0],
        note=combined_note,
    )
    sources = {key: source for key, (_, source) in values.items()}
    sources.update({key: source for key, (_, source) in text_values.items()})
    return CalibrationResult(
        table=table,
        input_peak_gain=peak,
        peak_scale=peak_scale,
        normalized_peak=normalized_peak,
        provenance_sources=sources,
    )


# ---------------------------------------------------------------------------
# 与 gaussian/linear 预设对照
# ---------------------------------------------------------------------------


def compare_with_presets(table: RamanGainTable) -> dict[str, Any]:
    """比较实测表与 ``RamanGainModel`` 的 gaussian/linear 预设。

    比较在输入谱的频移网格上进行。度量包括逐点形状偏差、峰值频移、0 到
    10 THz 的割线斜率，以及 0 到 21 THz 的平均形状。斜率同时给出按
    ``peak_gain_coefficient_m_per_w / A_eff`` 折算后的绝对单位
    ``1/(W km THz)``，便于与 Semrau 的线性代理谱对照。

    Args:
        table: 已标定的增益表。

    Returns
    -------
        可 JSON 序列化的对照字典；不包含大数组。
    """
    measured = table.to_model()
    gaussian = RamanGainModel()
    linear = RamanGainModel(gain_model="linear")
    grid = table.offsets_hz
    grid_thz = grid / _THZ_TO_HZ
    a_eff_m2 = table.reference_a_eff_m2 if table.reference_a_eff_m2 is not None else 80e-12
    peak_coefficient = table.peak_gain_coefficient_m_per_w

    shapes = {
        "measured": measured.gain_shape(grid),
        "gaussian": gaussian.gain_shape(grid),
        "linear": linear.gain_shape(grid),
    }
    results: dict[str, Any] = {
        "grid": {
            "points": int(grid.size),
            "min_thz": float(grid_thz[0]),
            "max_thz": float(grid_thz[-1]),
        },
        "reference": {
            "peak_gain_coefficient_m_per_w": peak_coefficient,
            "reference_a_eff_um2": a_eff_m2 / 1e-12,
            "slope_band_thz": SLOPE_REFERENCE_THZ,
            "mean_shape_band_thz": MEAN_SHAPE_REFERENCE_THZ,
        },
        "shapes": {},
        "deviation_vs_measured": {},
    }
    for name, shape in shapes.items():
        results["shapes"][name] = _shape_metrics(shape, grid_thz, peak_coefficient, a_eff_m2)
    for name in ("gaussian", "linear"):
        results["deviation_vs_measured"][name] = _deviation_metrics(shapes[name], shapes["measured"])
    results["peak_shift_difference_thz"] = {
        "gaussian_minus_measured": results["shapes"]["gaussian"]["peak_shift_thz"]
        - results["shapes"]["measured"]["peak_shift_thz"],
        "linear_minus_measured": results["shapes"]["linear"]["peak_shift_thz"]
        - results["shapes"]["measured"]["peak_shift_thz"],
    }
    results["slope_ratio_vs_measured"] = {
        "gaussian": _safe_ratio(
            results["shapes"]["gaussian"]["secant_slope_per_thz"],
            results["shapes"]["measured"]["secant_slope_per_thz"],
        ),
        "linear": _safe_ratio(
            results["shapes"]["linear"]["secant_slope_per_thz"],
            results["shapes"]["measured"]["secant_slope_per_thz"],
        ),
    }
    return results


def _shape_metrics(
    shape: np.ndarray,
    grid_thz: np.ndarray,
    peak_coefficient: float,
    a_eff_m2: float,
) -> dict[str, float]:
    """单个形状的峰值、斜率与带宽平均。"""
    peak_index = int(np.argmax(shape))
    slope_hi_thz = min(SLOPE_REFERENCE_THZ, float(grid_thz[-1]))
    slope_per_thz = _secant_slope(shape, grid_thz, slope_hi_thz)
    mean_shape = _mean_shape(shape, grid_thz, MEAN_SHAPE_REFERENCE_THZ)
    return {
        "peak_normalized_gain": float(shape[peak_index]),
        "peak_shift_thz": float(grid_thz[peak_index]),
        "secant_slope_per_thz": slope_per_thz,
        "secant_slope_1_per_w_km_thz": slope_per_thz * peak_coefficient / a_eff_m2 * 1e3,
        "mean_shape_0_21thz": mean_shape,
        "coverage_thz": float(grid_thz[-1]),
    }


def _secant_slope(shape: np.ndarray, grid_thz: np.ndarray, hi_thz: float) -> float:
    """在 ``[0, hi]`` 上取割线斜率（对降序网格也成立）。"""
    if hi_thz <= 0.0:
        return 0.0
    lo_value = float(np.interp(0.0, grid_thz, shape, left=shape[0], right=shape[-1]))
    hi_value = float(np.interp(hi_thz, grid_thz, shape, left=0.0, right=0.0))
    return (hi_value - lo_value) / hi_thz


def _mean_shape(shape: np.ndarray, grid_thz: np.ndarray, band_thz: float) -> float:
    """``[0, band]`` 上的带宽平均形状；表外按 0 处理。"""
    sample = np.linspace(0.0, band_thz, 2101)
    values = np.interp(sample, grid_thz, shape, left=0.0, right=0.0)
    if grid_thz[0] > 0.0:
        values[sample < grid_thz[0]] = 0.0
    return float(trapezoid(values, sample) / band_thz)


def _deviation_metrics(model: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """逐点绝对/均方根偏差与相关系数。"""
    difference = np.asarray(model, dtype=np.float64) - np.asarray(measured, dtype=np.float64)
    scale = max(float(np.max(np.abs(measured))), np.finfo(float).tiny)
    if np.std(model) == 0.0 or np.std(measured) == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(model, measured)[0, 1])
    return {
        "max_abs_deviation": float(np.max(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_abs_deviation": float(np.mean(np.abs(difference))),
        "relative_rmse": float(np.sqrt(np.mean(difference**2)) / scale),
        "correlation": correlation,
    }


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """除法；分母为 0 时返回 ``None`` 而不是 Inf。"""
    if denominator == 0.0 or not math.isfinite(denominator):
        return None
    return float(numerator / denominator)


# ---------------------------------------------------------------------------
# 可选：单跨跨损/ASE 灵敏度
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanSensitivityConfig:
    """固定单跨灵敏度算例的参数。

    Attributes
    ----------
        bands: 波段集合串，默认 C。
        num_channels: 信道数。
        channel_spacing_hz: 信道间隔 (Hz)。
        symbol_rate_hz: 符号率 (Hz)。
        span_length_km: 跨长 (km)。
        power_dbm: 每信道物理总功率 (dBm)。
        pump_wavelength_nm: 后向泵浦波长 (nm)。
        pump_power_w: 泵浦功率 (W)。
        pump_attenuation_db_per_km: 泵浦衰减 (dB/km)。
        models: 要比较的模型；只允许 ``h3``/``numerical_dra``。
    """

    bands: str = "C"
    num_channels: int = 40
    channel_spacing_hz: float = 100e9
    symbol_rate_hz: float = 64e9
    span_length_km: float = 100.0
    power_dbm: float = 0.0
    pump_wavelength_nm: float = 1450.0
    pump_power_w: float = 0.2
    pump_attenuation_db_per_km: float = 0.25
    models: tuple[str, ...] = ("h3", "numerical_dra")


def run_span_sensitivity(table: RamanGainTable, config: SpanSensitivityConfig) -> dict[str, Any]:
    """在单跨算例上比较标定表与默认 Gaussian 的跨损/拉曼 ASE。

    对比对象是同一 ``RamanGainModel`` 的 ``table`` 与默认 ``gaussian`` 两条
    谱；模型侧只调用公开入口：H3 走 ``pcfm_h3_raman`` 的 span kernel，
    ``numerical_dra`` 直接构造其独立 ODE 求解器（该模型层不接受表文件，
    所以这里用求解器实例，口径与 ``tests/test_h3_reference_anomaly.py`` 一致）。

    Args:
        table: 已标定的增益表。
        config: 单跨算例与要比较的模型。

    Returns
    -------
        逐模型、逐谱的跨损/ASE 摘要与 measured-gaussian 差值。

    Raises
    ------
        ValueError: 模型名非法、算例参数非法，或频谱范围覆盖不到泵浦-信号频移。
    """
    unknown = sorted(set(config.models) - {"h3", "numerical_dra"})
    if unknown:
        raise ValueError(f"unsupported span sensitivity model(s): {', '.join(unknown)}.")
    case, frequencies_hz, power_w, fiber, pump = _build_span_case(config)
    pump_frequency_hz = C_LIGHT / (float(pump.wavelength_nm) * 1e-9)
    shifts_hz = np.abs(pump_frequency_hz - frequencies_hz)
    if float(np.min(shifts_hz)) < float(table.offsets_hz[0]) or float(np.max(shifts_hz)) > float(table.offsets_hz[-1]):
        raise ValueError(
            "the Raman table does not cover the pump-signal shifts of the span case: "
            f"shifts {np.min(shifts_hz) / 1e12:.3f}..{np.max(shifts_hz) / 1e12:.3f} THz vs table "
            f"{table.offsets_hz[0] / 1e12:.3f}..{table.offsets_hz[-1] / 1e12:.3f} THz. Extend the "
            "measured spectrum or move the pump so the table is not silently truncated to zero."
        )

    variants: dict[str, RamanGainModel] = {
        "measured": table.to_model(),
        "gaussian": RamanGainModel(),
    }
    span_length_m = config.span_length_km * 1e3
    bandwidths_hz = np.full(frequencies_hz.size, config.symbol_rate_hz)
    alpha_desc, a_eff_desc, fiber_desc = _fiber_arrays_descending(frequencies_hz, fiber)
    pump_a_eff_m2 = float(
        _calc_aeff_dynamic(
            np.array([C_LIGHT / (float(pump.wavelength_nm) * 1e-9)]),
            fiber.effective_area_um2_ref,
            fiber.aeff_slope_um2_nm,
            fiber.reference_wavelength_nm,
        )[0]
    )

    results: dict[str, Any] = {
        "case": {
            "bands": config.bands,
            "num_channels": config.num_channels,
            "channel_spacing_hz": config.channel_spacing_hz,
            "symbol_rate_hz": config.symbol_rate_hz,
            "span_length_km": config.span_length_km,
            "power_dbm": config.power_dbm,
            "frequency_range_thz": [
                float(np.min(frequencies_hz) / 1e12),
                float(np.max(frequencies_hz) / 1e12),
            ],
            "pump": pump.model_dump(),
        },
        "models": {},
        "notes": [
            "Both models receive the same launch power, fibre and pump; only the Raman gain spectrum changes.",
            "Loss is pre-EDFA span loss 10log10(P_in/P_out); ASE is newly generated Raman ASE at span end.",
            "numerical_dra is used through its span solver because the model layer accepts only gaussian/linear names.",
        ],
    }
    for model_name in config.models:
        per_variant: dict[str, Any] = {}
        for variant_name, gain_model in variants.items():
            if model_name == "h3":
                loss_db, ase_w = _h3_span_metrics(
                    case, fiber, power_w, frequencies_hz, span_length_m, pump, gain_model
                )
            else:
                loss_db, ase_w = _numerical_dra_span_metrics(
                    power_w,
                    frequencies_hz,
                    bandwidths_hz,
                    span_length_m,
                    alpha_desc,
                    a_eff_desc,
                    fiber_desc,
                    pump,
                    pump_a_eff_m2,
                    gain_model,
                )
            per_variant[variant_name] = _loss_ase_summary(loss_db, ase_w, frequencies_hz)
        measured_summary = per_variant["measured"]
        gaussian_summary = per_variant["gaussian"]
        results["models"][model_name] = {
            "by_spectrum": per_variant,
            "delta_measured_minus_gaussian": {
                "loss_db_mean": measured_summary["loss_db_mean"] - gaussian_summary["loss_db_mean"],
                "loss_db_max_abs": float(
                    np.max(np.abs(np.asarray(measured_summary["loss_db_per_channel"])
                                  - np.asarray(gaussian_summary["loss_db_per_channel"])))
                ),
                "raman_ase_total_ratio_db": _ratio_db(
                    measured_summary["raman_ase_w_total"], gaussian_summary["raman_ase_w_total"]
                ),
            },
        }
    return results


def _build_span_case(
    config: SpanSensitivityConfig,
) -> tuple[BenchCase, np.ndarray, np.ndarray, Any, RamanPumpConfig]:
    """构造单跨算例、升序频率、入纤功率、参考光纤与泵浦。"""
    if config.num_channels < 2:
        raise ValueError("span sensitivity needs at least two channels.")
    if config.span_length_km <= 0.0 or config.channel_spacing_hz <= 0.0 or config.symbol_rate_hz <= 0.0:
        raise ValueError("span length, channel spacing and symbol rate must be positive.")
    low, high = band_edges_hz(config.bands)
    case = BenchCase(
        name="raman_spectrum_calibration_span",
        bands=config.bands,
        num_channels=config.num_channels,
        channel_spacing_hz=config.channel_spacing_hz,
        center_frequency_hz=(low + high) / 2.0,
        symbol_rate_hz=config.symbol_rate_hz,
        span_length_km=config.span_length_km,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )
    path = build_path(case)
    fiber = path.fiber_configs[0]
    power = launch_profile(case, power_dbm=config.power_dbm)
    frequencies = case.center_frequency_hz + (
        np.arange(case.num_channels) - (case.num_channels - 1) / 2.0
    ) * config.channel_spacing_hz
    pump = RamanPumpConfig(
        wavelength_nm=config.pump_wavelength_nm,
        power_w=config.pump_power_w,
        direction="backward",
        attenuation_db_per_km=config.pump_attenuation_db_per_km,
    )
    return case, frequencies, power, fiber, pump


def _fiber_arrays_descending(
    frequencies_hz: np.ndarray,
    fiber: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """核使用的降序衰减、有效面积与降序频率。"""
    frequencies_desc = np.asarray(frequencies_hz[::-1], dtype=np.float64)
    lambdas_m = C_LIGHT / frequencies_desc
    alpha = _calc_fiber_attenuation_npm(
        lambdas_m,
        fiber.reference_wavelength_nm,
        fiber.attenuation_alpha2,
        fiber.attenuation_alpha1,
        fiber.attenuation_alpha0,
    )
    a_eff = _calc_aeff_dynamic(
        lambdas_m,
        fiber.effective_area_um2_ref,
        fiber.aeff_slope_um2_nm,
        fiber.reference_wavelength_nm,
    )
    return np.asarray(alpha), np.asarray(a_eff), frequencies_desc


def _h3_span_metrics(
    case: BenchCase,
    fiber: Any,
    power_w: np.ndarray,
    frequencies_hz: np.ndarray,
    span_length_m: float,
    pump: RamanPumpConfig,
    gain_model: RamanGainModel,
) -> tuple[np.ndarray, np.ndarray]:
    """``pcfm_h3_raman`` 单跨：跨损 (dB) 与新产生拉曼 ASE (W)，模型顺序。"""
    model = make_model("pcfm_h3_raman", case, raman_pumps=[pump], raman_gain_model=gain_model)
    kernel = model.kernel_for(fiber)
    output = kernel.evaluate(power_w, span_length_m)
    power_out = np.asarray(output.power_out_w, dtype=np.float64)
    loss_db = 10.0 * np.log10(power_w / power_out, where=power_w > 0.0, out=np.zeros_like(power_w))
    return np.asarray(loss_db), np.asarray(kernel.last_raman_ase_w, dtype=np.float64)


def _numerical_dra_span_metrics(
    power_w: np.ndarray,
    frequencies_hz: np.ndarray,
    bandwidths_hz: np.ndarray,
    span_length_m: float,
    alpha_desc: np.ndarray,
    a_eff_desc: np.ndarray,
    frequencies_desc: np.ndarray,
    pump: RamanPumpConfig,
    pump_a_eff_m2: float,
    gain_model: RamanGainModel,
) -> tuple[np.ndarray, np.ndarray]:
    """``numerical_dra`` 单跨 ODE：跨损 (dB) 与新产生拉曼 ASE (W)，模型顺序。"""
    pump_alpha = np.array([float(pump.attenuation_db_per_km) * np.log(10.0) / 10.0 / 1e3])
    pump_a_eff = np.array([pump_a_eff_m2])
    solver = NumericalDRASpanSolver(
        np.ascontiguousarray(frequencies_desc),
        np.ascontiguousarray(bandwidths_hz[::-1]),
        [pump],
        span_length_m,
        np.ascontiguousarray(alpha_desc),
        np.ascontiguousarray(a_eff_desc),
        pump_alpha,
        pump_a_eff,
        gain_model,
    )
    result = solver.solve(np.ascontiguousarray(power_w[::-1]))
    power_out_desc = np.asarray(result.power_out_w, dtype=np.float64)
    power_in_desc = np.asarray(power_w[::-1], dtype=np.float64)
    loss_desc = 10.0 * np.log10(
        power_in_desc / power_out_desc,
        where=power_in_desc > 0.0,
        out=np.zeros_like(power_in_desc),
    )
    ase_desc = np.asarray(result.raman_ase_w, dtype=np.float64)
    # 核内部降序，返回值也按降序；转回模型顺序，便于两个模型逐信道对照。
    return np.asarray(loss_desc[::-1]), np.asarray(ase_desc[::-1])


def _loss_ase_summary(
    loss_db: np.ndarray,
    ase_w: np.ndarray,
    frequencies_hz: np.ndarray,
) -> dict[str, Any]:
    """跨损/ASE 的标量摘要与逐信道数组。"""
    order = np.argsort(frequencies_hz)
    loss_sorted = np.asarray(loss_db, dtype=np.float64)[order]
    ase_sorted = np.asarray(ase_w, dtype=np.float64)[order]
    return {
        "loss_db_width": float(loss_sorted[-1] - loss_sorted[0]),
        "loss_db_mean": float(np.mean(loss_sorted)),
        "loss_db_min": float(np.min(loss_sorted)),
        "loss_db_max": float(np.max(loss_sorted)),
        "raman_ase_w_total": float(np.sum(ase_sorted)),
        "raman_ase_w_max": float(np.max(ase_sorted)),
        "raman_ase_w_edge_low": float(ase_sorted[0]),
        "raman_ase_w_edge_high": float(ase_sorted[-1]),
        "loss_db_per_channel": [float(value) for value in loss_sorted],
        "raman_ase_w_per_channel": [float(value) for value in ase_sorted],
    }


def _ratio_db(numerator: float, denominator: float) -> float | None:
    """功率比的 dB；任一值为 0/非有限时返回 ``None``。"""
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return float(10.0 * math.log10(numerator / denominator))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="实测拉曼谱 → RamanGainTable JSON → gaussian/linear 模型对照。",
    )
    parser.add_argument("--input", required=True, type=Path, help="输入 CSV/JSON 实测谱。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raman/raman_spectrum_calibration_table.json"),
        help="输出增益表 JSON（默认 results/raman/raman_spectrum_calibration_table.json）。",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("results/raman/raman_spectrum_calibration.json"),
        help="输出对照报告 JSON（默认 results/raman/raman_spectrum_calibration.json）。",
    )
    parser.add_argument("--fiber", default=None, help="光纤型号/批次（provenance）。")
    parser.add_argument("--source", default=None, help="数据来源（provenance；输入也可携带）。")
    parser.add_argument("--note", default=None, help="附加说明（provenance）。")
    parser.add_argument("--temperature-k", type=float, default=None, help="光纤温度 K。")
    parser.add_argument(
        "--peak-coefficient-m-per-w",
        type=float,
        default=None,
        help="峰值局部增益率系数 m/W；未给时用模型默认 6e-14。",
    )
    parser.add_argument("--reference-wavelength-nm", type=float, default=None, help="峰值系数标定波长 nm。")
    parser.add_argument("--reference-a-eff-um2", type=float, default=None, help="标定有效模面积 µm²。")
    parser.add_argument(
        "--normalize-peak",
        action="store_true",
        help="允许把峰值缩放到 1，并把原始峰值写进 provenance。",
    )
    parser.add_argument(
        "--peak-tolerance",
        type=float,
        default=DEFAULT_PEAK_TOLERANCE,
        help=f"峰值偏离 1 的容差（默认 {DEFAULT_PEAK_TOLERANCE}）。",
    )
    parser.add_argument(
        "--span-sensitivity",
        choices=("none", "h3", "numerical_dra", "both"),
        default="none",
        help="可选单跨跨损/ASE 灵敏度对照（较慢）。",
    )
    parser.add_argument("--span-bands", default="C", help="灵敏度算例波段（默认 C）。")
    parser.add_argument("--span-channels", type=int, default=40, help="灵敏度算例信道数（默认 40）。")
    parser.add_argument("--span-length-km", type=float, default=100.0, help="灵敏度算例跨长 km（默认 100）。")
    parser.add_argument("--span-power-dbm", type=float, default=0.0, help="每信道功率 dBm（默认 0）。")
    parser.add_argument("--pump-wavelength-nm", type=float, default=1450.0, help="后向泵浦波长 nm。")
    parser.add_argument("--pump-power-w", type=float, default=0.2, help="泵浦功率 W。")
    parser.add_argument(
        "--pump-attenuation-db-per-km",
        type=float,
        default=0.25,
        help="泵浦衰减 dB/km。",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：读取、标定、对照并写出表与报告。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    spectrum = load_measured_spectrum(args.input)
    config = CalibrationConfig(
        fiber=args.fiber,
        source=args.source,
        note=args.note,
        temperature_k=args.temperature_k,
        peak_gain_coefficient_m_per_w=args.peak_coefficient_m_per_w,
        reference_wavelength_nm=args.reference_wavelength_nm,
        reference_a_eff_um2=args.reference_a_eff_um2,
        normalize_peak=args.normalize_peak,
        peak_tolerance=args.peak_tolerance,
    )
    calibration = calibrate_spectrum(spectrum, config)
    table = calibration.table
    save_raman_gain_table(table, args.output)

    report: dict[str, Any] = {
        "generated_by": "scripts/raman_spectrum_calibration.py",
        "input": {
            "path": str(args.input),
            "sha256": _sha256(args.input),
            "points": int(spectrum.offsets_hz.size),
            "shift_range_thz": [
                float(spectrum.offsets_hz[0] / _THZ_TO_HZ),
                float(spectrum.offsets_hz[-1] / _THZ_TO_HZ),
            ],
        },
        "table": {
            "path": str(args.output),
            "provenance": table.metadata,
            "temperature_k": table.temperature_k,
            "peak_gain_coefficient_m_per_w": table.peak_gain_coefficient_m_per_w,
            "reference_wavelength_nm": table.reference_wavelength_nm,
            "reference_a_eff_um2": table.reference_a_eff_um2,
            "peak_normalized_gain": table.peak_normalized_gain,
            "input_peak_gain": calibration.input_peak_gain,
            "peak_scale": calibration.peak_scale,
            "normalized_peak": calibration.normalized_peak,
            "provenance_sources": dict(calibration.provenance_sources),
        },
        "comparison": compare_with_presets(table),
        "span_sensitivity": None,
        "notes": [
            "The script never embeds or downloads measured spectra; the input is the caller's data.",
            "gaussian/linear are the RamanGainModel defaults, not calibrated to any specific fibre.",
            "reference_wavelength_nm is provenance only and does not scale the model.",
        ],
    }
    if args.span_sensitivity != "none":
        models = ("h3", "numerical_dra") if args.span_sensitivity == "both" else (args.span_sensitivity,)
        span_config = SpanSensitivityConfig(
            bands=args.span_bands,
            num_channels=args.span_channels,
            span_length_km=args.span_length_km,
            power_dbm=args.span_power_dbm,
            pump_wavelength_nm=args.pump_wavelength_nm,
            pump_power_w=args.pump_power_w,
            pump_attenuation_db_per_km=args.pump_attenuation_db_per_km,
            models=models,
        )
        report["span_sensitivity"] = run_span_sensitivity(table, span_config)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    _print_summary(table, report, args.output, args.report)
    return 0


def _sha256(path: Path) -> str:
    """输入文件的 SHA-256，用于报告溯源。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _print_summary(table: RamanGainTable, report: dict[str, Any], output: Path, report_path: Path) -> None:
    """打印一段简短的人读摘要。"""
    comparison = report["comparison"]
    grid = comparison["grid"]
    print(f"input points: {grid['points']} ({grid['min_thz']:.3f}..{grid['max_thz']:.3f} THz)")
    print(f"peak: {table.peak_normalized_gain:.6g} at {comparison['shapes']['measured']['peak_shift_thz']:.3f} THz")
    for name in ("gaussian", "linear"):
        deviation = comparison["deviation_vs_measured"][name]
        shape = comparison["shapes"][name]
        print(
            f"{name}: peak {shape['peak_shift_thz']:.3f} THz, "
            f"slope {shape['secant_slope_1_per_w_km_thz']:.4g} 1/(W km THz), "
            f"rmse {deviation['rmse']:.4g}"
        )
    span = report.get("span_sensitivity")
    if span:
        for model_name, values in span["models"].items():
            delta = values["delta_measured_minus_gaussian"]
            print(
                f"span {model_name}: d(loss) mean {delta['loss_db_mean']:+.3f} dB, "
                f"max |d(loss)| {delta['loss_db_max_abs']:.3f} dB, "
                f"ase ratio {delta['raman_ase_total_ratio_db']} dB"
            )
    print(f"wrote {output}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    raise SystemExit(main())
