"""File format and I/O for measured Raman gain spectra.

JSON and CSV carry the same normalized spectrum and calibration context.
``format`` must be ``network_nlo_eval.raman_gain_table`` and ``version`` must
be 1. Offsets are pump-signal frequency shifts in THz, nonnegative and strictly
increasing. Gains are dimensionless and at least one must be positive. The
spectrum is zero outside the table. Peaks outside [0.98, 1.02] issue a warning;
a peak of one corresponds to the local gain coefficient in m/W, which the
model divides by effective area. Temperature (K) affects spontaneous scattering
but not the on-off gain shape. Reference wavelength (nm) is provenance only,
while optional reference effective area (um²) is the model fallback.

Optional fiber, source and note fields provide provenance. Optional unit
annotations are validated against canonical units. JSON stores offsets and
gains as arrays; CSV uses ``# key = value`` metadata lines and the two-column
header ``offsets_thz,normalized_gain``. Both formats are loaded at model
construction, outside the solver hot path. Invalid files raise errors with the
path and, where applicable, the field or line number.

File or model-instance input is supported by the lumped PCFM, H3 and hybrid
Raman implementations. Other model entry points may accept only analytic gain
names. A supplied instance or path must not be combined with an in-memory table.
"""

from __future__ import annotations

import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.models.raman import (
    DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
    RamanGainModel,
)

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayFloat

__all__ = [
    "RAMAN_GAIN_TABLE_FORMAT",
    "RAMAN_GAIN_TABLE_UNITS",
    "RAMAN_GAIN_TABLE_VERSION",
    "RamanGainTable",
    "load_raman_gain_table",
    "save_raman_gain_table",
]

#: Required format identifier.
RAMAN_GAIN_TABLE_FORMAT = "network_nlo_eval.raman_gain_table"

#: Supported format version.
RAMAN_GAIN_TABLE_VERSION = 1

#: Canonical units for validating optional unit declarations.
RAMAN_GAIN_TABLE_UNITS: dict[str, str] = {
    "offsets_thz": "THz",
    "normalized_gain": "1",
    "temperature_k": "K",
    "peak_gain_coefficient_m_per_w": "m/W",
    "reference_wavelength_nm": "nm",
    "reference_a_eff_um2": "um^2",
}

#: Accepted unit aliases; incompatible units are rejected.
_UNIT_ALIASES: dict[str, frozenset[str]] = {
    "offsets_thz": frozenset({"thz", "terahertz"}),
    "normalized_gain": frozenset(
        {"1", "dimensionless", "unitless", "a.u.", "a.u", "none"}
    ),
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

#: Convert file offsets from THz to internal Hz.
_THZ_TO_HZ = 1e12

#: µm² → m²。
_UM2_TO_M2 = 1e-12

#: Runtime options that do not change calibration.
_RUNTIME_OVERRIDES = frozenset({"include_photon_energy_factor", "ase_steps"})

#: Calibration fields whose explicit overrides change gain or ASE and warn.
_CALIBRATION_FIELDS = frozenset(
    {"temperature_k", "peak_gain_coefficient_m_per_w", "reference_a_eff_m2"}
)

#: Expected normalized peak range; a deviation may indicate absolute gain.
_PEAK_NORMALIZATION_RANGE = (0.98, 1.02)


@dataclass(frozen=True, slots=True)
class RamanGainTable:
    """Validated Raman gain spectrum and calibration context.

    Arrays are copied and made read-only so a table can be shared across spans.

    Attributes
    ----------
    offsets_hz : ndarray
        Nonnegative, strictly increasing pump-signal offsets (Hz).
    normalized_gain : ndarray
        Dimensionless gain shape with at least one positive value.
    temperature_k : float
        Fiber temperature (K), used for spontaneous emission.
    peak_gain_coefficient_m_per_w : float
        Nonnegative peak local gain coefficient (m/W).
    reference_wavelength_nm : float
        Calibration wavelength (nm); provenance only, not used in calculations.
    reference_a_eff_m2 : float or None
        Calibration effective area (m²), used as the model fallback.
    fiber, source, note : str or None
        Fiber, measurement source and other provenance.
    units : Mapping[str, str]
        Declared units, validated and frozen.
    format : str
        File format identifier.
    version : int
        Format version.

    Raises
    ------
    ValueError
        If a field, array, unit, or calibration value is invalid.
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
    units: Mapping[str, str] = field(
        default_factory=lambda: dict(RAMAN_GAIN_TABLE_UNITS)
    )
    format: str = RAMAN_GAIN_TABLE_FORMAT
    version: int = RAMAN_GAIN_TABLE_VERSION

    def __post_init__(self) -> None:
        """Validate scalars and copy arrays into read-only storage."""
        offsets = np.array(self.offsets_hz, dtype=np.float64, copy=True)
        gains = np.array(self.normalized_gain, dtype=np.float64, copy=True)
        if offsets.ndim != 1 or gains.ndim != 1:
            msg = "Raman gain table offsets and gains must be one-dimensional."
            raise ValueError(msg)
        if offsets.size != gains.size:
            msg = "Raman gain table offsets and gains must have the same length."
            raise ValueError(msg)
        if offsets.size < 2:
            msg = "Raman gain table must contain at least two points."
            raise ValueError(msg)
        if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(gains)):
            msg = "Raman gain table values must be finite."
            raise ValueError(msg)
        if np.any(offsets < 0.0):
            msg = "Raman gain table offsets must be non-negative."
            raise ValueError(msg)
        if np.any(np.diff(offsets) <= 0.0):
            msg = "Raman gain table offsets must be strictly increasing."
            raise ValueError(msg)
        if np.any(gains < 0.0):
            msg = "Raman gain table gains must be non-negative."
            raise ValueError(msg)
        if not np.any(gains > 0.0):
            msg = "Raman gain table must contain at least one positive gain."
            raise ValueError(msg)
        if not math.isfinite(self.temperature_k) or self.temperature_k <= 0.0:
            msg = "temperature_k must be finite and positive."
            raise ValueError(msg)
        if (
            not math.isfinite(self.peak_gain_coefficient_m_per_w)
            or self.peak_gain_coefficient_m_per_w < 0.0
        ):
            msg = "peak_gain_coefficient_m_per_w must be finite and non-negative."
            raise ValueError(msg)
        if (
            not math.isfinite(self.reference_wavelength_nm)
            or self.reference_wavelength_nm <= 0.0
        ):
            msg = "reference_wavelength_nm must be finite and positive."
            raise ValueError(msg)
        if self.reference_a_eff_m2 is not None and (
            not math.isfinite(self.reference_a_eff_m2) or self.reference_a_eff_m2 <= 0.0
        ):
            msg = "reference_a_eff_m2 must be finite and positive."
            raise ValueError(msg)
        if self.format != RAMAN_GAIN_TABLE_FORMAT:
            msg = f"format must be {RAMAN_GAIN_TABLE_FORMAT!r}, got {self.format!r}."
            raise ValueError(msg)
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version != RAMAN_GAIN_TABLE_VERSION
        ):
            msg = f"version must be the integer {RAMAN_GAIN_TABLE_VERSION}, got {self.version!r}."
            raise ValueError(msg)
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
        object.__setattr__(
            self, "reference_wavelength_nm", float(self.reference_wavelength_nm)
        )
        if self.reference_a_eff_m2 is not None:
            object.__setattr__(
                self, "reference_a_eff_m2", float(self.reference_a_eff_m2)
            )
        object.__setattr__(self, "units", units)
        for key in ("fiber", "source", "note"):
            value = getattr(self, key)
            if value is not None and not isinstance(value, str):
                msg = f"{key} must be a string or None, got {type(value).__name__}."
                raise ValueError(msg)

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
        """Return pump-signal frequency shifts in THz.

        Returns
        -------
        ndarray
            Pump-signal shifts (THz).
        """
        return self.offsets_hz / _THZ_TO_HZ

    @property
    def reference_a_eff_um2(self) -> float | None:
        """Return the reference effective area in um², if provided.

        Returns
        -------
        float or None
            Effective area (um²).
        """
        if self.reference_a_eff_m2 is None:
            return None
        return self.reference_a_eff_m2 / _UM2_TO_M2

    @property
    def peak_normalized_gain(self) -> float:
        """Return the maximum normalized gain.

        A peak far from one may indicate an unnormalized absolute spectrum, which
        would scale the model gain by the same factor.

        Returns
        -------
        float
            Peak normalized gain.
        """
        return float(np.max(self.normalized_gain))

    @property
    def metadata(self) -> dict[str, Any]:
        """Return provenance metadata, which does not affect gain or ASE.

        Returns
        -------
        dict
            Provenance metadata.
        """
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
        """Create a table-backed Raman gain model.

        Runtime options may be overridden. Calibration overrides warn and are
        recorded in the model; strict mode rejects them. The gain model and table
        itself cannot be overridden. Source and calibration provenance are retained.

        Parameters
        ----------
        strict_calibration_override : bool
            Reject calibration overrides rather than warn.
        **overrides : Any
            Runtime or explicitly overridden calibration fields.

        Returns
        -------
        RamanGainModel
            Model with its source table and provenance.

        Raises
        ------
        ValueError
            If the file, table, or requested overrides are invalid.
        """
        if "gain_model" in overrides or "table" in overrides:
            msg = "RamanGainTable.to_model() manages gain_model and table; do not override them."
            raise ValueError(msg)
        calibration = {
            key: overrides.pop(key)
            for key in list(overrides)
            if key in _CALIBRATION_FIELDS
        }
        unknown = set(overrides) - _RUNTIME_OVERRIDES
        if unknown:
            msg = (
                "unsupported RamanGainTable.to_model() override(s): "
                f"{', '.join(sorted(unknown))}; allowed runtime overrides are "
                f"{', '.join(sorted(_RUNTIME_OVERRIDES))}."
            )
            raise ValueError(msg)
        if calibration and strict_calibration_override:
            msg = (
                "calibration field(s) "
                f"{', '.join(sorted(calibration))} from the Raman gain table file cannot be overridden "
                "when strict_calibration_override=True."
            )
            raise ValueError(msg)
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
                    msg = f"{key} override must not be None."
                    raise ValueError(msg)
                # Explicit None discards the file area and uses the model default.
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
            key: float(value) if value is not None else None
            for key, value in calibration.items()
        }
        return model


def load_raman_gain_table(path: str | Path) -> RamanGainTable:
    """Load and validate a Raman gain table from a JSON or CSV file.

    Parameters
    ----------
    path : str | Path
        JSON or CSV file path.

    Returns
    -------
    RamanGainTable
        Validated table.

    Raises
    ------
    ValueError
        If the file, table, or requested overrides are invalid.
    FileNotFoundError
        If the file does not exist.
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix == ".json":
        payload = _load_json_payload(resolved)
    elif suffix == ".csv":
        payload = _load_csv_payload(resolved)
    else:
        msg = f"{resolved}: unsupported Raman gain table extension {suffix!r}; expected '.json' or '.csv'."
        raise ValueError(msg)
    return _table_from_payload(payload, resolved)


def save_raman_gain_table(table: RamanGainTable, path: str | Path) -> None:
    """Write a Raman gain table to a JSON or CSV file.

    Values are written in canonical units. THz offsets may incur a one-ULP
    conversion difference, while normalized gains and scalar values round-trip.

    Parameters
    ----------
    table : RamanGainTable
        Validated Raman gain table.
    path : str | Path
        JSON or CSV file path.

    Raises
    ------
    ValueError
        If the file, table, or requested overrides are invalid.
    """
    resolved = Path(path)
    suffix = resolved.suffix.lower()
    if suffix not in (".json", ".csv"):
        msg = f"{resolved}: unsupported Raman gain table extension {suffix!r}; expected '.json' or '.csv'."
        raise ValueError(msg)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        _save_json(table, resolved)
    else:
        _save_csv(table, resolved)


# ---------------------------------------------------------------------------
# Unit and field validation
# ---------------------------------------------------------------------------


def _freeze_units(units: Mapping[str, str]) -> Mapping[str, str]:
    """Validate and freeze the unit mapping.

    Parameters
    ----------
    units : Mapping[str, str]
        Declared unit mapping.

    Returns
    -------
    Mapping
        Validated immutable unit mapping.
    """
    if not isinstance(units, Mapping):
        msg = f"units must be a mapping of field -> unit string, got {type(units).__name__}."
        raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
    unknown = sorted(set(units) - set(RAMAN_GAIN_TABLE_UNITS))
    if unknown:
        msg = f"unknown units field(s): {', '.join(unknown)}."
        raise ValueError(msg)
    frozen: dict[str, str] = {}
    for key, value in units.items():
        if not isinstance(value, str):
            msg = f"unit of {key} must be a string, got {type(value).__name__}."
            raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
        frozen[key] = value
    return MappingProxyType(frozen)


def _validate_units(value: Any, path: Path) -> None:
    """Validate explicit units and reject unknown keys or incorrect units.

    Parameters
    ----------
    value : Any
        Value to validate or convert.
    path : Path
        JSON or CSV file path.
    """
    if not isinstance(value, Mapping):
        msg = f"{path}: units must be an object of field -> unit string, got {type(value).__name__}."
        raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
    unknown = sorted(set(value) - set(RAMAN_GAIN_TABLE_UNITS))
    if unknown:
        msg = f"{path}: unknown units field(s): {', '.join(unknown)}."
        raise ValueError(msg)
    for key, declared in value.items():
        if not isinstance(declared, str):
            msg = f"{path}: unit of {key} must be a string, got {declared!r}."
            raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
        expected = RAMAN_GAIN_TABLE_UNITS[key]
        normalized = declared.strip().lower()
        if normalized not in _UNIT_ALIASES[key]:
            msg = f"{path}: unit of {key} must be {expected!r}, got {declared!r}."
            raise ValueError(msg)


def _units_line() -> str:
    """Format the CSV units field as a comma-separated list.

    Returns
    -------
    str
        Formatted units.
    """
    return ", ".join(f"{key}={unit}" for key, unit in RAMAN_GAIN_TABLE_UNITS.items())


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


def _load_json_payload(path: Path) -> dict[str, Any]:
    """Read and validate the JSON top-level object.

    Parameters
    ----------
    path : Path
        JSON or CSV file path.

    Returns
    -------
    dict
        Decoded JSON payload.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        msg = f"{path}: Raman gain table must be UTF-8 text: {error}"
        raise ValueError(msg) from None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        msg = f"{path}: invalid JSON at line {error.lineno}, column {error.colno}: {error.msg}"
        raise ValueError(msg) from None
    if not isinstance(payload, dict):
        msg = f"{path}: Raman gain table JSON must be an object, got {type(payload).__name__}."
        raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
    return payload


def _save_json(table: RamanGainTable, path: Path) -> None:
    """Write JSON in canonical field order.

    Parameters
    ----------
    table : RamanGainTable
        Validated Raman gain table.
    path : Path
        JSON or CSV file path.
    """
    payload = _payload_from_table(table)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def _load_csv_payload(path: Path) -> dict[str, Any]:
    """Read CSV metadata and the two-column data section.

    Parameters
    ----------
    path : Path
        JSON or CSV file path.

    Returns
    -------
    dict
        Decoded CSV payload.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        msg = f"{path}: Raman gain table must be UTF-8 text: {error}"
        raise ValueError(msg) from None

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
                msg = f"{path}:{line_number}: CSV header must be 'offsets_thz,normalized_gain', got {line!r}."
                raise ValueError(msg)
            header_seen = True
            continue
        rows.append(_parse_csv_data_line(line, path, line_number))
    if not header_seen:
        msg = f"{path}: CSV is missing the 'offsets_thz,normalized_gain' header."
        raise ValueError(msg)
    if len(rows) < 2:
        msg = f"{path}: Raman gain table must contain at least two data rows."
        raise ValueError(msg)
    metadata["offsets_thz"] = [row[0] for row in rows]
    metadata["normalized_gain"] = [row[1] for row in rows]
    return _finalize_csv_metadata(metadata, path)


def _finalize_csv_metadata(metadata: dict[str, Any], path: Path) -> dict[str, Any]:
    """Convert CSV metadata strings to JSON-equivalent types.

    Parameters
    ----------
    metadata : dict[str, Any]
        CSV metadata mapping.
    path : Path
        JSON or CSV file path.
    """
    if "version" in metadata:
        try:
            metadata["version"] = int(metadata["version"])
        except (TypeError, ValueError):
            msg = f"{path}: version must be the integer {RAMAN_GAIN_TABLE_VERSION}."
            raise ValueError(msg) from None
    if "units" in metadata and isinstance(metadata["units"], str):
        units: dict[str, str] = {}
        for item in metadata["units"].split(","):
            key, separator, unit = item.partition("=")
            key = key.strip()
            if not separator or not key:
                msg = f"{path}: units must be a comma-separated list of field=unit."
                raise ValueError(msg)
            units[key] = unit.strip()
        metadata["units"] = units
    return metadata


def _parse_csv_metadata_line(
    metadata: dict[str, Any], line: str, path: Path, line_number: int
) -> None:
    """Parse one metadata line and reject duplicate keys.

    Parameters
    ----------
    metadata : dict[str, Any]
        CSV metadata mapping.
    line : str
        CSV line to parse.
    path : Path
        JSON or CSV file path.
    line_number : int
        One-based CSV line number.
    """
    body = line[1:].strip()
    key, separator, value = body.partition("=")
    key = key.strip()
    if not separator or not key:
        msg = f"{path}:{line_number}: metadata line must be '# key = value', got {line!r}."
        raise ValueError(msg)
    if key in metadata:
        msg = f"{path}:{line_number}: duplicate metadata key {key!r}."
        raise ValueError(msg)
    metadata[key] = value.strip()


def _parse_csv_data_line(line: str, path: Path, line_number: int) -> list[float]:
    """Parse two finite values from a CSV row.

    Parameters
    ----------
    line : str
        CSV line to parse.
    path : Path
        JSON or CSV file path.
    line_number : int
        One-based CSV line number.
    """
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 2:
        msg = f"{path}:{line_number}: data row must have two columns, got {line!r}."
        raise ValueError(msg)
    try:
        values = [float(parts[0]), float(parts[1])]
    except ValueError:
        msg = f"{path}:{line_number}: data row contains a non-numeric value: {line!r}."
        raise ValueError(msg) from None
    if not all(math.isfinite(value) for value in values):
        msg = f"{path}:{line_number}: data row values must be finite: {line!r}."
        raise ValueError(msg)
    return values


def _save_csv(table: RamanGainTable, path: Path) -> None:
    """Write CSV metadata and data in canonical order.

    Parameters
    ----------
    table : RamanGainTable
        Validated Raman gain table.
    path : Path
        JSON or CSV file path.
    """
    for key, value in (
        ("fiber", table.fiber),
        ("source", table.source),
        ("note", table.note),
    ):
        if value is not None and ("\n" in value or "\r" in value):
            msg = f"{key} must not contain a newline when saving CSV metadata."
            raise ValueError(msg)
    lines = [
        f"# format = {RAMAN_GAIN_TABLE_FORMAT}",
        f"# version = {RAMAN_GAIN_TABLE_VERSION}",
        f"# units = {_units_line()}",
        f"# temperature_k = {table.temperature_k!r}",
        f"# peak_gain_coefficient_m_per_w = {table.peak_gain_coefficient_m_per_w!r}",
        f"# reference_wavelength_nm = {table.reference_wavelength_nm!r}",
    ]
    if table.reference_a_eff_m2 is not None:
        lines.append(
            f"# reference_a_eff_um2 = {table.reference_a_eff_m2 / _UM2_TO_M2!r}"
        )
    for key, value in (
        ("fiber", table.fiber),
        ("source", table.source),
        ("note", table.note),
    ):
        if value is not None:
            lines.append(f"# {key} = {value}")
    lines.append(",".join(_CSV_COLUMNS))
    for offset_hz, gain in zip(table.offsets_hz, table.normalized_gain, strict=True):
        lines.append(f"{float(offset_hz / _THZ_TO_HZ)!r},{float(gain)!r}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared payload validation
# ---------------------------------------------------------------------------


def _table_from_payload(payload: dict[str, Any], path: Path) -> RamanGainTable:
    """Validate a JSON or CSV payload and construct a table.

    Parameters
    ----------
    payload : dict[str, Any]
        Decoded JSON or CSV payload.
    path : Path
        JSON or CSV file path.

    Returns
    -------
    RamanGainTable
        Validated table.
    """
    missing = [key for key in _REQUIRED_KEYS if key not in payload]
    if missing:
        msg = (
            f"{path}: missing required Raman gain table field(s): {', '.join(missing)}."
        )
        raise ValueError(msg)
    unknown = sorted(set(payload) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unknown:
        msg = f"{path}: unknown Raman gain table field(s): {', '.join(unknown)}."
        raise ValueError(msg)

    _validate_format(payload["format"], path)
    _validate_version(payload["version"], path)
    if "units" in payload:
        _validate_units(payload["units"], path)

    offsets_thz = _as_float_array(payload["offsets_thz"], "offsets_thz", path)
    gains = _as_float_array(payload["normalized_gain"], "normalized_gain", path)
    if offsets_thz.size != gains.size:
        msg = (
            f"{path}: offsets_thz and normalized_gain must have the same length, "
            f"got {offsets_thz.size} and {gains.size}."
        )
        raise ValueError(msg)
    temperature_k = _as_float(
        payload["temperature_k"], "temperature_k", path, positive=True
    )
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
        reference_a_eff_um2 = _as_float(
            payload["reference_a_eff_um2"], "reference_a_eff_um2", path, positive=True
        )
        reference_a_eff_m2 = reference_a_eff_um2 * _UM2_TO_M2
    strings = {
        key: _as_optional_text(payload.get(key), key, path)
        for key in ("fiber", "source", "note")
    }

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
        msg = f"{path}: {error}"
        raise ValueError(msg) from None


def _payload_from_table(table: RamanGainTable) -> dict[str, Any]:
    """Convert a table to a canonical JSON-serializable payload.

    Parameters
    ----------
    table : RamanGainTable
        Validated Raman gain table.

    Returns
    -------
    dict
        Canonical payload.
    """
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
    """Validate the file format identifier.

    Parameters
    ----------
    value : Any
        Value to validate or convert.
    path : Path
        JSON or CSV file path.
    """
    if value != RAMAN_GAIN_TABLE_FORMAT:
        msg = f"{path}: format must be {RAMAN_GAIN_TABLE_FORMAT!r}, got {value!r}."
        raise ValueError(msg)


def _validate_version(value: Any, path: Path) -> None:
    """Validate that the version is integer 1.

    Parameters
    ----------
    value : Any
        Value to validate or convert.
    path : Path
        JSON or CSV file path.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != RAMAN_GAIN_TABLE_VERSION
    ):
        msg = (
            f"{path}: version must be the integer {RAMAN_GAIN_TABLE_VERSION}, got {value!r}; "
            "this loader only understands version 1."
        )
        raise ValueError(msg)


def _as_float(value: Any, key: str, path: Path, *, positive: bool = False) -> float:
    """Convert a scalar to float, rejecting booleans and nonfinite values.

    Parameters
    ----------
    value : Any
        Value to validate or convert.
    key : str
        Field name.
    path : Path
        JSON or CSV file path.
    positive : bool
        Require a positive value.

    Returns
    -------
    float
        Validated numeric value.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        msg = f"{path}: {key} must be a number, got {value!r}."
        raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            msg = f"{path}: {key} must be a number, got {value!r}."
            raise ValueError(msg) from None
    else:
        result = float(value)
    if not math.isfinite(result):
        msg = f"{path}: {key} must be finite, got {value!r}."
        raise ValueError(msg)
    if positive and result <= 0.0:
        msg = f"{path}: {key} must be positive, got {value!r}."
        raise ValueError(msg)
    if not positive and result < 0.0:
        msg = f"{path}: {key} must be non-negative, got {value!r}."
        raise ValueError(msg)
    return result


def _as_float_array(value: Any, key: str, path: Path) -> NDArrayFloat:
    """Convert a numeric list to a one-dimensional float64 array.

    Parameters
    ----------
    value : Any
        Value to validate or convert.
    key : str
        Field name.
    path : Path
        JSON or CSV file path.

    Returns
    -------
    ndarray
        Validated one-dimensional numeric array.
    """
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        msg = f"{path}: {key} must be a list of numbers, got {type(value).__name__}."
        raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
    items: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            msg = f"{path}: {key}[{index}] must be a number, got {item!r}."
            raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
        number = float(item)
        if not math.isfinite(number):
            msg = f"{path}: {key}[{index}] must be finite, got {item!r}."
            raise ValueError(msg)
        items.append(number)
    return np.asarray(items, dtype=np.float64)


def _as_optional_text(value: Any, key: str, path: Path) -> str | None:
    """Return optional text, rejecting non-string values.

    Parameters
    ----------
    value : Any
        Value to validate or convert.
    key : str
        Field name.
    path : Path
        JSON or CSV file path.

    Returns
    -------
    str or None
        Validated optional text.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"{path}: {key} must be a string, got {type(value).__name__}."
        raise ValueError(msg)  # noqa: TRY004 - preserve the file-validation API
    return value
