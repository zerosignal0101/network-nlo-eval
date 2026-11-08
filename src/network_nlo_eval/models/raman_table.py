"""实测拉曼增益谱表的文件格式、加载与保存。

K 任务给 :class:`~network_nlo_eval.models.raman.RamanGainModel` 增加了
``gain_model="table"``，但那只接受内存中的 ``(offsets_hz, gains)``。本模块
补上可复现的文件入口：把“光纤、温度、峰值系数、参考波长、A_eff”等标定
上下文和谱点放在同一个文件里，避免表和标定条件分离后无法追溯。

文件格式（JSON 与 CSV 等价）
---------------------------

* ``format`` 固定为 ``"network_nlo_eval.raman_gain_table"``，``version`` 为
  ``1``。两者都必须出现，用于拒绝误读的其它 JSON/CSV。
* ``offsets_thz``：泵浦-信号**频移** ``|f_p - f_s|``，单位 THz，非负且严格
  递增。它是频移而不是光频，因此与泵浦波长无关。
* ``normalized_gain``：无量纲归一化增益形，``1.0`` 对应峰值系数。允许
  实测谱的相对起伏，但至少有一个正值；表外频移按 0 处理。峰值不强制
  等于 1，但明显偏离 ``[0.98, 1.02]`` 时会发 ``UserWarning``，提醒调用方
  不要把绝对增益误填成谱形。
* ``temperature_k``：测量/使用时的光纤温度，单位 K，只进入自发拉曼散射的
  声子占据因子。它**不改变** on-off 增益谱形。
* ``peak_gain_coefficient_m_per_w``：峰值**局部**增益率系数，单位 m/W，
  使用前由模型除以信号有效模面积 ``A_eff``。它不是跨段 on-off 增益，也
  不是 ``g_R/A_eff``。
* ``reference_wavelength_nm``：上述峰值系数的标定波长，单位 nm。**只做
  溯源，不参与任何计算**；模型不会按信号波长对它做缩放。
* 可选 ``reference_a_eff_um2``：标定时的有效模面积，单位 µm²；给出时作为
  模型的 ``reference_a_eff_m2``，只在调用方没有显式传 ``A_eff`` 时生效。
* 可选 ``fiber``/``source``/``note``：自由文本，只用于追溯。加载后保留在
  :class:`RamanGainTable` 和 ``RamanGainModel`` 的只读属性上。
* 可选 ``units``：显式单位声明，出现时逐项校验；键名本身已经带单位，
  因此缺省不会改变解释。

JSON 示例::

    {
      "format": "network_nlo_eval.raman_gain_table",
      "version": 1,
      "units": {"offsets_thz": "THz", "normalized_gain": "1", "temperature_k": "K",
                "peak_gain_coefficient_m_per_w": "m/W",
                "reference_wavelength_nm": "nm", "reference_a_eff_um2": "um^2"},
      "fiber": "SMF-28-like",
      "source": "vendor measurement, 2026-11",
      "note": "normalized to the 13.2 THz peak; shift = |f_p - f_s|",
      "temperature_k": 300.0,
      "peak_gain_coefficient_m_per_w": 6e-14,
      "reference_wavelength_nm": 1550.0,
      "reference_a_eff_um2": 80.0,
      "offsets_thz": [0.0, 0.25, 0.5],
      "normalized_gain": [0.0, 0.12, 0.35]
    }

CSV 用 ``# key = value`` 头行携带同样的元数据，数据区固定两列
``offsets_thz,normalized_gain``::

    # format = network_nlo_eval.raman_gain_table
    # version = 1
    # units = offsets_thz=THz, normalized_gain=1, temperature_k=K,
    #         peak_gain_coefficient_m_per_w=m/W, reference_wavelength_nm=nm
    # fiber = SMF-28-like
    # source = vendor measurement, 2026-11
    # note = shift = |f_p - f_s|
    # temperature_k = 300.0
    # peak_gain_coefficient_m_per_w = 6e-14
    # reference_wavelength_nm = 1550.0
    # reference_a_eff_um2 = 80.0
    offsets_thz,normalized_gain
    0.0,0.0
    0.25,0.12
    0.5,0.35

支持的模型入口
---------------

文件/实例入口只覆盖使用 :class:`RamanGainModel` 的集总/PCFM/H3/Hybrid 路径：

* ``pcfm`` + ``raman_model="lumped"``；
* ``pcfm_h3_raman``；
* ``hybrid_raman_paper``（及其 ``RamanPaperSpanSolver`` 核）。

这些模型既可接收 ``RamanGainModel.from_file(path)`` 实例，也可把 ``.json``/
``.csv`` 路径直接传给 ``raman_gain_model``。``raman_gain_table`` 与实例/路径
互斥：已经给了实例时不要再传内存表。

``buglia2024_ra_gn``、``zhang2024_dra_egn``、``numerical_dra``、``semrau2019``
和 ``ecz`` 的模型层只接受 ``"gaussian"``/``"linear"`` 名称，暂不支持文件或
实例；传错类型时会由 :class:`RamanGainModel` 或 Python 的参数检查给出明确
错误，而不是静默退回 Gaussian。

加载只在模型构造期发生，不进入 PCFM/ODE 热路径。校验失败一律抛
``ValueError`` 并在消息里带文件路径（JSON 另带字段名，CSV 另带行号）。
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.raman import (
    DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
    RamanGainModel,
)

__all__ = [
    "RAMAN_GAIN_TABLE_FORMAT",
    "RAMAN_GAIN_TABLE_UNITS",
    "RAMAN_GAIN_TABLE_VERSION",
    "RamanGainTable",
    "load_raman_gain_table",
    "save_raman_gain_table",
]

#: ``format`` 字段的固定值。
RAMAN_GAIN_TABLE_FORMAT = "network_nlo_eval.raman_gain_table"

#: 当前支持的格式版本。
RAMAN_GAIN_TABLE_VERSION = 1

#: 每个字段的规范单位；``units`` 块出现时按此逐项校验。
RAMAN_GAIN_TABLE_UNITS: dict[str, str] = {
    "offsets_thz": "THz",
    "normalized_gain": "1",
    "temperature_k": "K",
    "peak_gain_coefficient_m_per_w": "m/W",
    "reference_wavelength_nm": "nm",
    "reference_a_eff_um2": "um^2",
}

#: 单位别名。文件里可以写常见变体，但 ``GHz``、``dB`` 等错误单位必须拒绝。
_UNIT_ALIASES: dict[str, frozenset[str]] = {
    "offsets_thz": frozenset({"thz", "terahertz"}),
    "normalized_gain": frozenset({"1", "dimensionless", "unitless", "a.u.", "a.u", "none"}),
    "temperature_k": frozenset({"k", "kelvin"}),
    "peak_gain_coefficient_m_per_w": frozenset({"m/w", "m w^-1", "m·w^-1", "m/watt"}),
    "reference_wavelength_nm": frozenset({"nm", "nanometer", "nanometre"}),
    "reference_a_eff_um2": frozenset(
        {"um^2", "um2", "µm^2", "µm2", "μm^2", "μm2", "micron^2", "micron2"}
    ),
}

_REQUIRED_KEYS = (
    "format",
    "version",
    "offsets_thz",
    "normalized_gain",
    "temperature_k",
    "peak_gain_coefficient_m_per_w",
    "reference_wavelength_nm",
)

_OPTIONAL_KEYS = (
    "units",
    "reference_a_eff_um2",
    "fiber",
    "source",
    "note",
)

_CSV_COLUMNS = ("offsets_thz", "normalized_gain")

#: THz → Hz。文件只存 THz，加载时一次性换算，模型内部仍全部用 Hz。
_THZ_TO_HZ = 1e12

#: µm² → m²。
_UM2_TO_M2 = 1e-12

#: 允许在 ``to_model``/``from_file`` 中覆盖的运行时参数。它们不改变标定。
_RUNTIME_OVERRIDES = frozenset({"include_photon_energy_factor", "ase_steps"})

#: 文件中的标定参数。覆盖它们会改变绝对增益/ASE，必须显式确认并告警。
_CALIBRATION_FIELDS = frozenset(
    {"temperature_k", "peak_gain_coefficient_m_per_w", "reference_a_eff_m2"}
)

#: ``normalized_gain`` 峰值应落在此区间内；偏离说明可能误填了绝对增益。
_PEAK_NORMALIZATION_RANGE = (0.98, 1.02)


@dataclass(frozen=True, slots=True)
class RamanGainTable:
    """一份已校验的拉曼增益谱表及其标定上下文。

    构造时即完成全部校验并把数组冻结为只读，之后可安全地在多个模型/
    跨段之间共享。``offsets_hz`` 与 ``normalized_gain`` 一一对应。

    Attributes
    ----------
        offsets_hz: ``(K,)`` 泵浦-信号频移 (Hz)，非负、严格递增。
        normalized_gain: ``(K,)`` 无量纲增益形，至少一个正值。
        temperature_k: 光纤温度 (K)，必须为正；只影响 ASE 的声子因子。
        peak_gain_coefficient_m_per_w: 峰值局部增益率系数 (m/W)，非负。
        reference_wavelength_nm: 峰值系数的标定波长 (nm)，必须为正。
            **只做溯源，不参与计算。**
        reference_a_eff_m2: 标定有效模面积 (m²)；``None`` 时沿用模型默认值。
        fiber: 光纤型号/批次等自由文本。
        source: 数据来源（测量报告、厂商文件）自由文本。
        note: 其它说明。
        units: 文件声明的单位；构造内存表时默认使用规范单位。
        format: 文件格式标识。
        version: 文件格式版本。

    Raises
    ------
        ValueError: 数组形状、长度、单调性、有限性或标量范围不合法。
    """

    offsets_hz: NDArrayFloat
    normalized_gain: NDArrayFloat
    temperature_k: float = 300.0
    peak_gain_coefficient_m_per_w: float = DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W
    reference_wavelength_nm: float = 1550.0
    reference_a_eff_m2: float | None = None
    fiber: str | None = None
    source: str | None = None
    note: str | None = None
    units: Mapping[str, str] = field(default_factory=lambda: dict(RAMAN_GAIN_TABLE_UNITS))
    format: str = RAMAN_GAIN_TABLE_FORMAT
    version: int = RAMAN_GAIN_TABLE_VERSION

    def __post_init__(self) -> None:
        """校验并冻结数组与标量。"""
        offsets = np.array(self.offsets_hz, dtype=np.float64, copy=True)
        gains = np.array(self.normalized_gain, dtype=np.float64, copy=True)
        if offsets.ndim != 1 or gains.ndim != 1:
            raise ValueError("Raman gain table offsets and gains must be one-dimensional.")
        if offsets.size != gains.size:
            raise ValueError("Raman gain table offsets and gains must have the same length.")
        if offsets.size < 2:
            raise ValueError("Raman gain table must contain at least two points.")
        if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(gains)):
            raise ValueError("Raman gain table values must be finite.")
        if np.any(offsets < 0.0):
            raise ValueError("Raman gain table offsets must be non-negative.")
        if np.any(np.diff(offsets) <= 0.0):
            raise ValueError("Raman gain table offsets must be strictly increasing.")
        if np.any(gains < 0.0):
            raise ValueError("Raman gain table gains must be non-negative.")
        if not np.any(gains > 0.0):
            raise ValueError("Raman gain table must contain at least one positive gain.")
        if not math.isfinite(self.temperature_k) or self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be finite and positive.")
        if not math.isfinite(self.peak_gain_coefficient_m_per_w) or self.peak_gain_coefficient_m_per_w < 0.0:
            raise ValueError("peak_gain_coefficient_m_per_w must be finite and non-negative.")
        if not math.isfinite(self.reference_wavelength_nm) or self.reference_wavelength_nm <= 0.0:
            raise ValueError("reference_wavelength_nm must be finite and positive.")
        if self.reference_a_eff_m2 is not None and (
            not math.isfinite(self.reference_a_eff_m2) or self.reference_a_eff_m2 <= 0.0
        ):
            raise ValueError("reference_a_eff_m2 must be finite and positive.")
        if self.format != RAMAN_GAIN_TABLE_FORMAT:
            raise ValueError(f"format must be {RAMAN_GAIN_TABLE_FORMAT!r}, got {self.format!r}.")
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version != RAMAN_GAIN_TABLE_VERSION
        ):
            raise ValueError(f"version must be the integer {RAMAN_GAIN_TABLE_VERSION}, got {self.version!r}.")
        units = _freeze_units(self.units)

        offsets.setflags(write=False)
        gains.setflags(write=False)
        object.__setattr__(self, "offsets_hz", offsets)
        object.__setattr__(self, "normalized_gain", gains)
        object.__setattr__(self, "temperature_k", float(self.temperature_k))
        object.__setattr__(
            self,
            "peak_gain_coefficient_m_per_w",
            float(self.peak_gain_coefficient_m_per_w),
        )
        object.__setattr__(self, "reference_wavelength_nm", float(self.reference_wavelength_nm))
        if self.reference_a_eff_m2 is not None:
            object.__setattr__(self, "reference_a_eff_m2", float(self.reference_a_eff_m2))
        object.__setattr__(self, "units", units)
        for key in ("fiber", "source", "note"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{key} must be a string or None, got {type(value).__name__}.")

        peak = float(np.max(gains))
        low, high = _PEAK_NORMALIZATION_RANGE
        if not low <= peak <= high:
            warnings.warn(
                "normalized_gain peak is "
                f"{peak:.6g}, outside [{low}, {high}]. The table is interpreted as a normalized "
                "shape multiplied by peak_gain_coefficient_m_per_w; if this file stores absolute "
                "gain (on-off dB, 1/(km W) or m/W), convert it to a normalized shape first.",
                UserWarning,
                stacklevel=2,
            )

    @property
    def offsets_thz(self) -> NDArrayFloat:
        """``(K,)`` 泵浦-信号频移 (THz)，便于按文件单位核对。"""
        return self.offsets_hz / _THZ_TO_HZ

    @property
    def reference_a_eff_um2(self) -> float | None:
        """标定有效模面积 (µm²)；只做溯源与模型 fallback。"""
        if self.reference_a_eff_m2 is None:
            return None
        return self.reference_a_eff_m2 / _UM2_TO_M2

    @property
    def peak_normalized_gain(self) -> float:
        """表内最大归一化增益。

        归一化谱通常应为 ``1.0``；明显偏离说明数据可能保留了未归一化的
        绝对增益，此时模型给出的绝对增益会按同一比例缩放。
        """
        return float(np.max(self.normalized_gain))

    @property
    def metadata(self) -> dict[str, Any]:
        """溯源元数据；全部字段都**不参与**增益/ASE 计算。"""
        return {
            "format": self.format,
            "version": self.version,
            "fiber": self.fiber,
            "source": self.source,
            "note": self.note,
            "reference_wavelength_nm": self.reference_wavelength_nm,
            "reference_a_eff_um2": self.reference_a_eff_um2,
            "units": dict(self.units),
        }

    def to_model(
        self,
        *,
        strict_calibration_override: bool = False,
        **overrides: Any,
    ) -> RamanGainModel:
        """把本表转成 ``gain_model="table"`` 的模型。

        Args:
            strict_calibration_override: 覆盖文件中的标定字段时是否直接拒绝。
                默认 ``False``：覆盖 ``temperature_k``、
                ``peak_gain_coefficient_m_per_w`` 或 ``reference_a_eff_m2`` 会先发
                ``UserWarning`` 再应用（不会静默）。设为 ``True`` 时抛
                ``ValueError``。无论哪种情况，应用的覆盖值都记录在
                ``model.calibration_overrides``。
            **overrides: 模型参数。只允许运行时参数
                （``include_photon_energy_factor``、``ase_steps``）以及上述标定
                字段；``gain_model`` 与 ``table`` 由本方法管理，禁止覆盖。

        Returns
        -------
            以表内容构造的 :class:`RamanGainModel`。模型上还挂了
            ``source_table``、``fiber``、``source``、``note``、
            ``reference_wavelength_nm`` 与 ``table_metadata`` 溯源属性。

        Raises
        ------
            ValueError: 试图覆盖 ``gain_model``/``table``、给出未知参数，或
                ``strict_calibration_override=True`` 时覆盖标定字段。
        """
        if "gain_model" in overrides or "table" in overrides:
            raise ValueError("RamanGainTable.to_model() manages gain_model and table; do not override them.")
        calibration = {key: overrides.pop(key) for key in list(overrides) if key in _CALIBRATION_FIELDS}
        unknown = set(overrides) - _RUNTIME_OVERRIDES
        if unknown:
            raise ValueError(
                "unsupported RamanGainTable.to_model() override(s): "
                f"{', '.join(sorted(unknown))}; allowed runtime overrides are "
                f"{', '.join(sorted(_RUNTIME_OVERRIDES))}."
            )
        if calibration and strict_calibration_override:
            raise ValueError(
                "calibration field(s) "
                f"{', '.join(sorted(calibration))} from the Raman gain table file cannot be overridden "
                "when strict_calibration_override=True."
            )
        if calibration:
            warnings.warn(
                "overriding calibration field(s) "
                f"{', '.join(sorted(calibration))} from the Raman gain table file; the file values "
                "will be replaced and the override recorded in model.calibration_overrides.",
                UserWarning,
                stacklevel=2,
            )

        params: dict[str, Any] = {
            "gain_model": "table",
            "table": (self.offsets_hz, self.normalized_gain),
            "peak_gain_coefficient_m_per_w": self.peak_gain_coefficient_m_per_w,
            "temperature_k": self.temperature_k,
        }
        if self.reference_a_eff_m2 is not None:
            params["reference_a_eff_m2"] = self.reference_a_eff_m2
        params.update(overrides)
        for key, value in calibration.items():
            if value is None:
                if key != "reference_a_eff_m2":
                    raise ValueError(f"{key} override must not be None.")
                # 显式 None 表示丢弃文件里的 A_eff，回落到模型默认值。
                params.pop(key, None)
            else:
                params[key] = float(value)
        model = RamanGainModel(**params)
        model.source_table = self
        model.table_metadata = self.metadata
        model.reference_wavelength_nm = self.reference_wavelength_nm
        model.fiber = self.fiber
        model.source = self.source
        model.note = self.note
        model.calibration_overrides = {
            key: float(value) if value is not None else None for key, value in calibration.items()
        }
        return model


def load_raman_gain_table(path: str | Path) -> RamanGainTable:
    """从 JSON 或 CSV 文件加载拉曼增益表。

    按扩展名选择解析器（``.json`` / ``.csv``，不区分大小写）。加载时完成
    单位、单调性、有限性、长度和标量范围的全部校验；解析结果只依赖文件
    内容，不读取任何全局状态。本函数只在模型构造期调用，不在热路径中。

    Args:
        path: 表格文件路径。

    Returns
    -------
        已校验的 :class:`RamanGainTable`。

    Raises
    ------
        FileNotFoundError: 文件不存在。
        ValueError: 扩展名不支持，或文件内容缺字段、单位不符、非单调、
            非有限、长度不匹配。
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        payload = _load_json_payload(resolved)
    elif suffix == ".csv":
        payload = _load_csv_payload(resolved)
    else:
        raise ValueError(
            f"{resolved}: unsupported Raman gain table extension {suffix!r}; expected '.json' or '.csv'."
        )
    return _table_from_payload(payload, resolved)


def save_raman_gain_table(table: RamanGainTable, path: str | Path) -> None:
    """把增益表写成 JSON 或 CSV，供后续 ``load_raman_gain_table`` 复现。

    写出的是**规范单位**（THz、m/W、nm、µm²），不写内部 Hz 数组；数值用
    Python 的最短往返表示。``normalized_gain`` 与标量可逐位还原；``offsets``
    经过一次 ``Hz/1e12`` 的十进制表示，个别值可能有 1 ulp 的 THz 表示误差
    （远小于任何物理分辨率）。

    Args:
        table: 已校验的增益表。
        path: 输出路径；父目录不存在时自动创建。

    Raises
    ------
        ValueError: 扩展名不是 ``.json`` 或 ``.csv``，或自由文本包含换行
            （CSV 元数据不能安全表示换行）。
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix not in (".json", ".csv"):
        raise ValueError(f"{resolved}: unsupported Raman gain table extension {suffix!r}; expected '.json' or '.csv'.")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        _save_json(table, resolved)
    else:
        _save_csv(table, resolved)


# ---------------------------------------------------------------------------
# 单位与字段校验
# ---------------------------------------------------------------------------


def _freeze_units(units: Mapping[str, str]) -> Mapping[str, str]:
    """校验并冻结单位映射。"""
    if not isinstance(units, Mapping):
        raise ValueError(f"units must be a mapping of field -> unit string, got {type(units).__name__}.")
    unknown = sorted(set(units) - set(RAMAN_GAIN_TABLE_UNITS))
    if unknown:
        raise ValueError(f"unknown units field(s): {', '.join(unknown)}.")
    frozen: dict[str, str] = {}
    for key, value in units.items():
        if not isinstance(value, str):
            raise ValueError(f"unit of {key} must be a string, got {type(value).__name__}.")
        frozen[key] = value
    return MappingProxyType(frozen)


def _validate_units(value: Any, path: Path) -> None:
    """逐项校验显式单位声明，未知键和错误单位都报错。"""
    if not isinstance(value, Mapping):
        raise ValueError(f"{path}: units must be an object of field -> unit string, got {type(value).__name__}.")
    unknown = sorted(set(value) - set(RAMAN_GAIN_TABLE_UNITS))
    if unknown:
        raise ValueError(f"{path}: unknown units field(s): {', '.join(unknown)}.")
    for key, declared in value.items():
        if not isinstance(declared, str):
            raise ValueError(f"{path}: unit of {key} must be a string, got {declared!r}.")
        expected = RAMAN_GAIN_TABLE_UNITS[key]
        normalized = declared.strip().lower()
        if normalized not in _UNIT_ALIASES[key]:
            raise ValueError(f"{path}: unit of {key} must be {expected!r}, got {declared!r}.")


def _units_line() -> str:
    """CSV 的 ``units =`` 值：``field=unit`` 逗号列表。"""
    return ", ".join(f"{key}={unit}" for key, unit in RAMAN_GAIN_TABLE_UNITS.items())


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _load_json_payload(path: Path) -> dict[str, Any]:
    """读取 JSON 并检查顶层结构。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: Raman gain table must be UTF-8 text: {error}") from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: Raman gain table JSON must be an object, got {type(payload).__name__}.")
    return payload


def _save_json(table: RamanGainTable, path: Path) -> None:
    """按规范顺序写 JSON。"""
    payload = _payload_from_table(table)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _load_csv_payload(path: Path) -> dict[str, Any]:
    """读取 ``# key = value`` 元数据 + 两列数据区。"""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: Raman gain table must be UTF-8 text: {error}") from None

    metadata: dict[str, Any] = {}
    rows: list[list[float]] = []
    header_seen = False
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            _parse_csv_metadata_line(metadata, line, path, line_number)
            continue
        if not header_seen:
            columns = tuple(part.strip() for part in line.split(","))
            if columns != _CSV_COLUMNS:
                raise ValueError(
                    f"{path}:{line_number}: CSV header must be 'offsets_thz,normalized_gain', got {line!r}."
                )
            header_seen = True
            continue
        rows.append(_parse_csv_data_line(line, path, line_number))
    if not header_seen:
        raise ValueError(f"{path}: CSV is missing the 'offsets_thz,normalized_gain' header.")
    if len(rows) < 2:
        raise ValueError(f"{path}: Raman gain table must contain at least two data rows.")
    metadata["offsets_thz"] = [row[0] for row in rows]
    metadata["normalized_gain"] = [row[1] for row in rows]
    return _finalize_csv_metadata(metadata, path)


def _finalize_csv_metadata(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
    """把 CSV 的字符串元数据转成与 JSON 相同的类型。"""
    if "version" in metadata:
        try:
            metadata["version"] = int(metadata["version"])
        except (TypeError, ValueError):
            raise ValueError(f"{path}: version must be the integer {RAMAN_GAIN_TABLE_VERSION}.") from None
    if "units" in metadata and isinstance(metadata["units"], str):
        units: dict[str, str] = {}
        for item in metadata["units"].split(","):
            key, separator, unit = item.partition("=")
            key = key.strip()
            if not separator or not key:
                raise ValueError(f"{path}: units must be a comma-separated list of field=unit.")
            units[key] = unit.strip()
        metadata["units"] = units
    return metadata


def _parse_csv_metadata_line(metadata: dict[str, Any], line: str, path: Path, line_number: int) -> None:
    """解析一行 ``# key = value``，重复键直接报错。"""
    body = line[1:].strip()
    key, separator, value = body.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ValueError(f"{path}:{line_number}: metadata line must be '# key = value', got {line!r}.")
    if key in metadata:
        raise ValueError(f"{path}:{line_number}: duplicate metadata key {key!r}.")
    metadata[key] = value.strip()


def _parse_csv_data_line(line: str, path: Path, line_number: int) -> list[float]:
    """解析一行两个有限数值，报错时带行号。"""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
        raise ValueError(f"{path}:{line_number}: data row must have two columns, got {line!r}.")
    try:
        values = [float(parts[0]), float(parts[1])]
    except ValueError:
        raise ValueError(f"{path}:{line_number}: data row contains a non-numeric value: {line!r}.") from None
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{path}:{line_number}: data row values must be finite: {line!r}.")
    return values


def _save_csv(table: RamanGainTable, path: Path) -> None:
    """按规范顺序写元数据头与数据区。"""
    for key, value in (("fiber", table.fiber), ("source", table.source), ("note", table.note)):
        if value is not None and ("\n" in value or "\r" in value):
            raise ValueError(f"{key} must not contain a newline when saving CSV metadata.")
    lines = [
        f"# format = {RAMAN_GAIN_TABLE_FORMAT}",
        f"# version = {RAMAN_GAIN_TABLE_VERSION}",
        f"# units = {_units_line()}",
        f"# temperature_k = {table.temperature_k!r}",
        f"# peak_gain_coefficient_m_per_w = {table.peak_gain_coefficient_m_per_w!r}",
        f"# reference_wavelength_nm = {table.reference_wavelength_nm!r}",
    ]
    if table.reference_a_eff_m2 is not None:
        lines.append(f"# reference_a_eff_um2 = {table.reference_a_eff_m2 / _UM2_TO_M2!r}")
    for key, value in (("fiber", table.fiber), ("source", table.source), ("note", table.note)):
        if value is not None:
            lines.append(f"# {key} = {value}")
    lines.append(",".join(_CSV_COLUMNS))
    for offset_hz, gain in zip(table.offsets_hz, table.normalized_gain, strict=True):
        lines.append(f"{float(offset_hz / _THZ_TO_HZ)!r},{float(gain)!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 公共 payload 校验
# ---------------------------------------------------------------------------


def _table_from_payload(payload: dict[str, Any], path: Path) -> RamanGainTable:
    """把 JSON dict 或 CSV 元数据 dict 统一校验成表。"""
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        raise ValueError(f"{path}: missing required Raman gain table field(s): {', '.join(missing)}.")
    unknown = sorted(set(payload) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unknown:
        raise ValueError(f"{path}: unknown Raman gain table field(s): {', '.join(unknown)}.")

    _validate_format(payload["format"], path)
    _validate_version(payload["version"], path)
    if "units" in payload:
        _validate_units(payload["units"], path)

    offsets_thz = _as_float_array(payload["offsets_thz"], "offsets_thz", path)
    gains = _as_float_array(payload["normalized_gain"], "normalized_gain", path)
    if offsets_thz.size != gains.size:
        raise ValueError(
            f"{path}: offsets_thz and normalized_gain must have the same length, "
            f"got {offsets_thz.size} and {gains.size}."
        )
    temperature_k = _as_float(payload["temperature_k"], "temperature_k", path, positive=True)
    peak_coefficient = _as_float(
        payload["peak_gain_coefficient_m_per_w"],
        "peak_gain_coefficient_m_per_w",
        path,
    )
    reference_wavelength_nm = _as_float(
        payload["reference_wavelength_nm"],
        "reference_wavelength_nm",
        path,
        positive=True,
    )
    reference_a_eff_m2 = None
    if "reference_a_eff_um2" in payload:
        reference_a_eff_um2 = _as_float(payload["reference_a_eff_um2"], "reference_a_eff_um2", path, positive=True)
        reference_a_eff_m2 = reference_a_eff_um2 * _UM2_TO_M2
    strings = {key: _as_optional_text(payload.get(key), key, path) for key in ("fiber", "source", "note")}

    try:
        return RamanGainTable(
            offsets_hz=offsets_thz * _THZ_TO_HZ,
            normalized_gain=gains,
            temperature_k=temperature_k,
            peak_gain_coefficient_m_per_w=peak_coefficient,
            reference_wavelength_nm=reference_wavelength_nm,
            reference_a_eff_m2=reference_a_eff_m2,
            fiber=strings["fiber"],
            source=strings["source"],
            note=strings["note"],
            units=payload.get("units", RAMAN_GAIN_TABLE_UNITS),
        )
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None


def _payload_from_table(table: RamanGainTable) -> dict[str, Any]:
    """把表转成 JSON 可序列化的规范 payload。"""
    payload: dict[str, Any] = {
        "format": RAMAN_GAIN_TABLE_FORMAT,
        "version": RAMAN_GAIN_TABLE_VERSION,
        "units": dict(RAMAN_GAIN_TABLE_UNITS),
        "temperature_k": table.temperature_k,
        "peak_gain_coefficient_m_per_w": table.peak_gain_coefficient_m_per_w,
        "reference_wavelength_nm": table.reference_wavelength_nm,
    }
    if table.reference_a_eff_m2 is not None:
        payload["reference_a_eff_um2"] = table.reference_a_eff_m2 / _UM2_TO_M2
    for key in ("fiber", "source", "note"):
        value = getattr(table, key)
        if value is not None:
            payload[key] = value
    payload["offsets_thz"] = [float(value) for value in table.offsets_thz]
    payload["normalized_gain"] = [float(value) for value in table.normalized_gain]
    return payload


def _validate_format(value: Any, path: Path) -> None:
    """校验 ``format`` 字段。"""
    if value != RAMAN_GAIN_TABLE_FORMAT:
        raise ValueError(f"{path}: format must be {RAMAN_GAIN_TABLE_FORMAT!r}, got {value!r}.")


def _validate_version(value: Any, path: Path) -> None:
    """校验 ``version`` 字段，只接受整数 1。"""
    if isinstance(value, bool) or not isinstance(value, int) or value != RAMAN_GAIN_TABLE_VERSION:
        raise ValueError(
            f"{path}: version must be the integer {RAMAN_GAIN_TABLE_VERSION}, got {value!r}; "
            "this loader only understands version 1."
        )


def _as_float(value: Any, key: str, path: Path, *, positive: bool = False) -> float:
    """把 JSON 标量或 CSV 文本转成 float，拒绝布尔与非有限值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{path}: {key} must be a number, got {value!r}.")
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            raise ValueError(f"{path}: {key} must be a number, got {value!r}.") from None
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path}: {key} must be finite, got {value!r}.")
    if positive and result <= 0.0:
        raise ValueError(f"{path}: {key} must be positive, got {value!r}.")
    if not positive and result < 0.0:
        raise ValueError(f"{path}: {key} must be non-negative, got {value!r}.")
    return result


def _as_float_array(value: Any, key: str, path: Path) -> NDArrayFloat:
    """把 JSON 列表或 CSV 数值列表转成一维 float64 数组。"""
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


def _as_optional_text(value: Any, key: str, path: Path) -> str | None:
    """可选自由文本；``None`` 保留，其它必须是字符串。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{path}: {key} must be a string, got {type(value).__name__}.")
    return value
