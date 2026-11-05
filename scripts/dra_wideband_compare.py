"""把 57/152ch 全 WDM 波形参考接入 network-nlo-eval 的解析模型。

参考数据来自 ``physical-nlo-eval`` 的 WP7 产物（``examples/dra_reference_wide/
{c57,cl152}.{json,npz}``），复制到 ``data/dra_waveform_wide/`` 后使用。脚本读取
JSON 的 ``plan`` / ``fiber`` / ``pump`` 与 ``datasets[*].configs``：

* 栅格：``SpectrumGrid(n_channels, channel_spacing_hz, center_frequency_hz)``，
  默认 64 GBd、75 GHz 间隔；``channel_bandwidth_source="symbol_rate"``，
  即解析侧 B = 符号率（64 GHz），与波形参考的 NLI 分析带宽一致；
* 光纤：``FiberSpanConfig`` 用波形同口径的平坦近似（D=17、S=0.06、
  α=0.162 dB/km、A_eff=80 µm²、n2=2.6e-20），若 JSON 给出
  ``beta2_ps2_per_km`` 而没给 ``dispersion_parameter_d``，按
  ``D = -(2πc/λ²)·β2`` 换算；
* 模型：``hybrid_raman_paper``（``nli_form="paper"``）、
  ``pcfm_h3_raman``（默认 ``egn="auto"``）、``numerical_dra``、
  ``pcfm`` + ``raman_model="lumped"``；泵浦逐 config 从 JSON 读取，
  hybrid/H3 用 ``pump_depletion=True``。

逐信道输出解析与波形的 SNR/NLI，并给出中心信道与全带残差。参考还没有
WP7 数据时，可指向 ``data/dra_waveform_converged/dra_reference.json`` 做
3ch 自检；该参考是未耗尽泵浦，脚本会跳过带泵配置（可用
``--include-pump-mismatch`` 强制跑）。

用法::

    PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python \
        scripts/dra_wideband_compare.py \
        --reference data/dra_waveform_wide/c57.json \
        --output results/raman/wideband_compare_c57.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# 与 3/5/7ch 对照脚本共用配置名 -> 泵浦的旧解析，仅作为 JSON 缺字段时的回退。
from dra_waveform_compare import DEPLETION_AWARE_MODELS, _pumps

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, RamanPumpConfig

DEFAULT_DATA_DIR = Path("data/dra_waveform_wide")
DEFAULT_REFERENCE_NAME = "c57.json"
DEFAULT_OUTPUT = Path("results/raman/wideband_compare.json")

#: 基准算例的默认值（``benchmarks/cases.py`` 的 ``c_57ch`` / ``cl_152ch``）。
DEFAULT_SYMBOL_RATE_HZ = 64e9
DEFAULT_CHANNEL_SPACING_HZ = 75e9
DEFAULT_REFERENCE_FREQUENCY_HZ = 193.1e12
DEFAULT_LAUNCH_POWER_DBM_PER_POL = 0.0
DEFAULT_ATTENUATION_DB_KM = 0.162
DEFAULT_DISPERSION_PS_NM_KM = 17.0
DEFAULT_DISPERSION_SLOPE = 0.06
DEFAULT_EFFECTIVE_AREA_UM2 = 80.0
DEFAULT_N2_M2_W = 2.6e-20
DEFAULT_LENGTH_KM = 100.0

#: EDFA 只影响 ASE；SNR_NLI 用 SPM+XPM，噪声系数仅用于口径完整。
EDFA_NOISE_FIGURE_DB = 5.5
C_LIGHT_M_S = 2.99792458e8

MODELS: tuple[str, ...] = (
    "hybrid_raman_paper",
    "pcfm_h3_raman",
    "numerical_dra",
    "pcfm",
)


def _warn(message: str) -> None:
    """Print a warning to stderr."""
    print(f"warning: {message}", file=sys.stderr)


def _first_value(*sources: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """按顺序在多个字典里查找第一个非 None 的键值。"""
    for key in keys:
        for source in sources:
            if source is None:
                continue
            if source.get(key) is not None:
                return source[key]
    return None


def _beta2_to_dispersion(beta2_ps2_per_km: float, wavelength_nm: float = 1550.0) -> float:
    """把 ``β2`` (ps²/km) 换算成 ``D`` (ps/(nm·km))：``D = -(2πc/λ²)β2``。"""
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    beta2_s2_per_m = float(beta2_ps2_per_km) * 1e-24 / 1e3
    wavelength_m = float(wavelength_nm) * 1e-9
    return float(-(2.0 * np.pi * C_LIGHT_M_S / wavelength_m**2) * beta2_s2_per_m * 1e6)


def _beta3_to_slope(
    beta3_ps3_per_km: float, dispersion_ps_nm_km: float, wavelength_nm: float = 1550.0
) -> float:
    """把 ``β3`` (ps³/km) 换算成 ``FiberSpanConfig`` 用的色散斜率 S。

    与 ``models.numba_kernels._calc_beta3_from_s_d`` 互逆::

        β3 = (λ²/(2πc)²) · (λ²·S·1e3 + 2λ·D·1e-6)

    Args:
        beta3_ps3_per_km: 三阶色散 β3 (ps³/km)。
        dispersion_ps_nm_km: 色散参数 D (ps/(nm·km))。
        wavelength_nm: 参考波长 (nm)。

    Returns
    -------
        S (ps/(nm²·km))，使解析侧 β3 与波形参考一致。
    """
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive.")
    wavelength_m = float(wavelength_nm) * 1e-9
    beta3_s3_per_m = float(beta3_ps3_per_km) * 1e-36 / 1e3
    numerator = (
        beta3_s3_per_m * (2.0 * np.pi * C_LIGHT_M_S) ** 2 / wavelength_m**2
        - 2.0 * wavelength_m * float(dispersion_ps_nm_km) * 1e-6
    )
    return float(numerator / (wavelength_m**2 * 1e3))


def _datasets_from_reference(reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """把参考 JSON 归一化成 ``{dataset_tag: dataset}``。

    旧 3/5/7ch 参考用顶层 ``datasets``；WP7 的全 WDM 参考把单计划几何
    放在 ``plan``，配置直接放在顶层 ``configs``。两种 schema 都接受。
    """
    datasets = reference.get("datasets") or {}
    if datasets:
        return datasets
    configs = reference.get("configs") or {}
    if not configs:
        return {}
    plan = dict(reference.get("plan") or {})
    tag = str(plan.get("name") or reference.get("dataset_family") or "wideband")
    dataset = {
        "n_channels": plan.get("n_channels"),
        "channel_spacing_hz": plan.get("channel_spacing_hz"),
        "center_frequency_hz": plan.get("center_frequency_hz"),
        "launch_power_dbm_per_pol": plan.get("launch_power_dbm_per_pol"),
        "center_channel_index": (
            int(plan["n_channels"]) // 2 if plan.get("n_channels") is not None else None
        ),
        "configs": configs,
    }
    return {tag: dataset}


def plan_from_reference(reference: dict[str, Any]) -> dict[str, Any]:
    """从参考 JSON 解析统一的全 WDM 计划。

    优先 ``plan`` 段；缺字段时回落到顶层 ``signal`` 与第一个数据集。
    信道间隔可从 ``per_channel[*].freq_offset_hz`` 的最小间隔反推。

    Args:
        reference: 参考 JSON。

    Returns
    -------
        含 ``n_channels``、``channel_spacing_hz``、``center_frequency_hz``、
        ``symbol_rate_hz``、``launch_power_dbm_per_pol`` 的字典。

    Raises
    ------
        SystemExit: 缺少无法安全推断的关键字段。
    """
    plan = dict(reference.get("plan") or {})
    signal = dict(reference.get("signal") or {})
    datasets = _datasets_from_reference(reference)
    first_dataset = next(iter(datasets.values()), {})

    n_channels = _first_value(plan, first_dataset, keys=("n_channels", "num_channels"))
    spacing = _first_value(plan, signal, keys=("channel_spacing_hz", "channel_spacing"))
    center = _first_value(plan, signal, reference, keys=("center_frequency_hz", "reference_freq_hz"))
    symbol_rate = _first_value(plan, signal, keys=("symbol_rate_hz", "symbol_rate"))
    launch_dbm = _first_value(plan, signal, first_dataset, keys=("launch_power_dbm_per_pol",))

    if n_channels is None:
        raise SystemExit("reference JSON has no plan.n_channels or dataset.n_channels")
    n_channels = int(n_channels)

    if spacing is None:
        spacing = _spacing_from_offsets(first_dataset)
    if spacing is None:
        _warn(
            f"reference has no channel_spacing_hz; assuming {DEFAULT_CHANNEL_SPACING_HZ / 1e9:g} GHz "
            "(benchmark default)"
        )
        spacing = DEFAULT_CHANNEL_SPACING_HZ

    if center is None:
        _warn(
            f"reference has no center_frequency_hz; assuming {DEFAULT_REFERENCE_FREQUENCY_HZ / 1e12:g} THz "
            "(physical reference carrier)"
        )
        center = DEFAULT_REFERENCE_FREQUENCY_HZ

    if symbol_rate is None:
        _warn(f"reference has no symbol_rate_hz; assuming {DEFAULT_SYMBOL_RATE_HZ / 1e9:g} GBd")
        symbol_rate = DEFAULT_SYMBOL_RATE_HZ

    if launch_dbm is None:
        _warn(
            f"reference has no launch_power_dbm_per_pol; assuming {DEFAULT_LAUNCH_POWER_DBM_PER_POL:g} dBm/pol"
        )
        launch_dbm = DEFAULT_LAUNCH_POWER_DBM_PER_POL

    return {
        "n_channels": n_channels,
        "channel_spacing_hz": float(spacing),
        "center_frequency_hz": float(center),
        "symbol_rate_hz": float(symbol_rate),
        "launch_power_dbm_per_pol": float(launch_dbm),
        "name": plan.get("name") or reference.get("name"),
        "roll_off": _first_value(plan, signal, keys=("roll_off",)),
        "sample_rate_hz": _first_value(plan, signal, keys=("sample_rate_hz",)),
        "num_symbols": _first_value(plan, signal, keys=("num_symbols",)),
    }


def _spacing_from_offsets(dataset: dict[str, Any]) -> float | None:
    """从 ``per_channel`` 的频偏反推信道间隔。"""
    configs = dataset.get("configs") or {}
    for config in configs.values():
        offsets = [
            float(channel["freq_offset_hz"])
            for channel in config.get("per_channel") or []
            if channel.get("freq_offset_hz") is not None
        ]
        if len(offsets) >= 2:
            unique = np.unique(np.sort(offsets))
            diffs = np.diff(unique)
            positive = diffs[diffs > 0]
            if positive.size:
                return float(np.median(positive))
    return None


def fiber_from_reference(reference: dict[str, Any]) -> tuple[FiberSpanConfig, dict[str, Any]]:
    """按波形参考的光纤参数构造平坦衰减的 ``FiberSpanConfig``。

    Args:
        reference: 参考 JSON。

    Returns
    -------
        ``(fiber, metadata)``；metadata 记录实际使用的参数与来源。
    """
    raw = dict(reference.get("fiber") or {})
    length_km = float(_first_value(raw, keys=("length_km",)) or DEFAULT_LENGTH_KM)
    attenuation = float(
        _first_value(raw, keys=("alpha_db_per_km", "attenuation_db_km_ref")) or DEFAULT_ATTENUATION_DB_KM
    )
    wavelength_nm = float(_first_value(raw, keys=("reference_wavelength_nm",)) or 1550.0)

    dispersion = _first_value(raw, keys=("dispersion_parameter_d",))
    source = "json"
    if dispersion is None:
        beta2 = _first_value(raw, keys=("beta2_ps2_per_km",))
        if beta2 is not None:
            dispersion = _beta2_to_dispersion(float(beta2), wavelength_nm)
            source = "beta2_ps2_per_km"
        else:
            dispersion = DEFAULT_DISPERSION_PS_NM_KM
            source = "default"
    slope_raw = _first_value(raw, keys=("dispersion_slope_s",))
    slope_source = "json"
    if slope_raw is None:
        beta3 = _first_value(raw, keys=("beta3_ps3_per_km",))
        if beta3 is not None:
            slope = _beta3_to_slope(float(beta3), float(dispersion), wavelength_nm)
            slope_source = "beta3_ps3_per_km"
        else:
            slope = DEFAULT_DISPERSION_SLOPE
            slope_source = "default"
    else:
        slope = float(slope_raw)
    effective_area = float(
        _first_value(raw, keys=("a_eff_um2", "effective_area_um2_ref")) or DEFAULT_EFFECTIVE_AREA_UM2
    )
    n2 = float(_first_value(raw, keys=("n2_si", "nonlinear_index_n2")) or DEFAULT_N2_M2_W)

    fiber = FiberSpanConfig(
        length_km=length_km,
        attenuation_db_km_ref=attenuation,
        use_attenuation_polynomial=False,
        dispersion_parameter_d=float(dispersion),
        dispersion_slope_s=slope,
        effective_area_um2_ref=effective_area,
        nonlinear_index_n2=n2,
    )
    metadata = {
        "length_km": length_km,
        "attenuation_db_km_ref": attenuation,
        "attenuation_model": "flat",
        "dispersion_parameter_d": float(dispersion),
        "dispersion_source": source,
        "dispersion_slope_s": slope,
        "dispersion_slope_source": slope_source,
        "effective_area_um2_ref": effective_area,
        "nonlinear_index_n2": n2,
    }
    return fiber, metadata


def pumps_from_config(config_name: str, config: dict[str, Any], reference: dict[str, Any]) -> list[RamanPumpConfig]:
    """读取一个参考配置的泵浦列表。

    优先 ``config["pumps"]``；其次是顶层 ``pump.configs[config_name]``；
    最后回退到 3/5/7ch 的配置名解析（``fwd_0p20w`` 等）。

    Args:
        config_name: 配置名。
        config: 该配置的字典。
        reference: 参考 JSON。

    Returns
    -------
        泵浦列表；``nopump`` 返回空列表。

    Raises
    ------
        SystemExit: 有泵配置但无法解析。
    """
    raw = config.get("pumps")
    if raw is None:
        pump_section = reference.get("pump") or {}
        named = (pump_section.get("configs") or {}).get(config_name)
        if named is not None:
            raw = named
    if raw is not None:
        if isinstance(raw, dict):
            raw = [raw]
        pumps = []
        for entry in raw:
            pumps.append(
                RamanPumpConfig(
                    wavelength_nm=float(entry["wavelength_nm"]),
                    power_w=float(entry["power_w"]),
                    direction=str(entry.get("direction", "backward")),
                    attenuation_db_per_km=float(entry.get("attenuation_db_per_km", 0.25)),
                )
            )
        return pumps
    if config_name == "nopump":
        return []
    try:
        return _pumps(config_name)
    except (IndexError, ValueError):
        raise SystemExit(
            f"config {config_name!r} has no 'pumps' list and its name is not a legacy fwd/bwd tag"
        ) from None


def evaluate_analytic(
    model_name: str,
    grid: SpectrumGrid,
    fiber: FiberSpanConfig,
    pumps: list[RamanPumpConfig],
    *,
    symbol_rate_hz: float,
    launch_power_w: float,
    signal_polarizations: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """跑一个解析模型，返回逐信道 ``(signal_power_w, nli_power_w)``。

    Args:
        model_name: ``MODELS`` 中的模型名。
        grid: 全 WDM 频谱栅格。
        fiber: 参考光纤。
        pumps: 泵浦列表。
        symbol_rate_hz: 符号率 (Hz)。
        launch_power_w: 每信道入纤功率 (W, 每偏振)。
        signal_polarizations: ODE 信号状态代表的偏振数；仅
            ``hybrid_raman_paper`` / ``pcfm_h3_raman`` 使用。1（默认）是历史
            每偏振驱动，2 用双偏振总功率驱动泵浦耗尽。

    Returns
    -------
        ``(N,)`` 信号功率与 ``(N,)`` NLI 功率（SPM+XPM）。
    """
    options: dict[str, Any] = {"raman_pumps": pumps} if pumps else {}
    if model_name == "hybrid_raman_paper" and pumps:
        options["nli_form"] = "paper"
    if model_name == "pcfm" and pumps:
        options["raman_model"] = "lumped"
    if model_name in DEPLETION_AWARE_MODELS:
        options["pump_depletion"] = True
        options["signal_polarizations"] = int(signal_polarizations)
    model = create_model(
        model_name,
        grid,
        fiber,
        symbol_rate_hz=symbol_rate_hz,
        channel_bandwidth_source="symbol_rate",
        **options,
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=EDFA_NOISE_FIGURE_DB)])
    result = model.evaluate_path(path, np.full(grid.num_channels, launch_power_w, dtype=np.float64))
    signal = np.asarray(result.signal_power_w, dtype=np.float64)
    nli = np.asarray(result.spm_noise_w, dtype=np.float64) + np.asarray(result.xpm_noise_w, dtype=np.float64)
    return signal, nli


def _snr_db(signal: np.ndarray, nli: np.ndarray) -> np.ndarray:
    """逐信道 ``10log10(P/NLI)``；非有限值记为 NaN。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10.0 * np.log10(np.divide(signal, nli, out=np.full_like(signal, np.nan), where=nli > 0.0))
    snr[~np.isfinite(snr)] = np.nan
    return snr


def _json_float(value: Any) -> float | None:
    """把 NaN/Inf 转成 None，保证输出是合法 JSON。"""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _nan_stat(values: np.ndarray, reducer: Any) -> float | None:
    """忽略 NaN 的统计量；全 NaN 时返回 None（不发出 RuntimeWarning）。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return _json_float(reducer(finite))


def _waveform_per_channel(config: dict[str, Any]) -> list[dict[str, Any]]:
    """按频偏排序波形逐信道结果。"""
    channels = [dict(channel) for channel in config.get("per_channel") or []]
    channels.sort(key=lambda channel: float(channel.get("freq_offset_hz", 0.0)))
    return channels


def _row(
    tag: str,
    dataset: dict[str, Any],
    config_name: str,
    config: dict[str, Any],
    plan: dict[str, Any],
    fiber: FiberSpanConfig,
    model_name: str,
    reference: dict[str, Any],
    *,
    pump_model: str,
    launch_power_dbm: float,
    signal_polarizations: int = 1,
) -> dict[str, Any]:
    """跑一个数据集/配置/模型并组装逐信道结果。"""
    n_channels = int(dataset.get("n_channels", plan["n_channels"]))
    spacing = float(dataset.get("channel_spacing_hz", plan["channel_spacing_hz"]))
    center_frequency = float(dataset.get("center_frequency_hz", plan["center_frequency_hz"]))
    launch_dbm = dataset.get("launch_power_dbm_per_pol")
    if launch_dbm is None:
        launch_dbm = launch_power_dbm
    launch_power_w = float(1e-3 * 10.0 ** (float(launch_dbm) / 10.0))
    pumps = pumps_from_config(config_name, config, reference)
    grid = SpectrumGrid(n_channels, spacing, center_frequency)
    signal, nli = evaluate_analytic(
        model_name,
        grid,
        fiber,
        pumps,
        symbol_rate_hz=float(plan["symbol_rate_hz"]),
        launch_power_w=launch_power_w,
        signal_polarizations=signal_polarizations,
    )
    analytic_snr = _snr_db(signal, nli)

    waveform = _waveform_per_channel(config)
    wave_snr = np.asarray([channel.get("snr_nli_db", np.nan) for channel in waveform], dtype=np.float64)
    wave_nli = np.asarray([channel.get("nli_power_w", np.nan) for channel in waveform], dtype=np.float64)
    center = int(dataset.get("center_channel_index", n_channels // 2))
    has_wave = wave_snr.size == n_channels and 0 <= center < n_channels

    deltas = analytic_snr - wave_snr if wave_snr.size == analytic_snr.size else np.asarray([np.nan])
    return {
        "dataset": tag,
        "n_channels": n_channels,
        "config": config_name,
        "model": model_name,
        "pump_model": pump_model,
        "pump_depletion": model_name in DEPLETION_AWARE_MODELS,
        "pump_depletion_applied": model_name in DEPLETION_AWARE_MODELS,
        "n_pumps": len(pumps),
        "launch_power_w_per_pol": launch_power_w,
        "center_channel_index": center,
        "waveform_center_snr_nli_db": _json_float(wave_snr[center]) if has_wave else None,
        "analytic_center_snr_nli_db": _json_float(analytic_snr[center]),
        "center_delta_db": _json_float(analytic_snr[center] - wave_snr[center]) if has_wave else None,
        "waveform_center_nli_w": _json_float(wave_nli[center]) if has_wave else None,
        "analytic_center_nli_w": _json_float(nli[center]),
        "delta_mean_db": _nan_stat(deltas, np.mean),
        "delta_min_db": _nan_stat(deltas, np.min),
        "delta_max_db": _nan_stat(deltas, np.max),
        "analytic_snr_mean_db": _nan_stat(analytic_snr, np.mean),
        "analytic_snr_min_db": _nan_stat(analytic_snr, np.min),
        "analytic_snr_max_db": _nan_stat(analytic_snr, np.max),
        "waveform_per_channel": [
            {
                "channel": int(channel.get("channel_id", index))
                if channel.get("channel_id") is not None
                else index,
                "freq_offset_hz": _json_float(channel.get("freq_offset_hz")),
                "snr_nli_db": _json_float(channel.get("snr_nli_db")),
                "nli_power_w": _json_float(channel.get("nli_power_w")),
            }
            for index, channel in enumerate(waveform)
        ],
        "analytic_per_channel": [
            {
                "channel": index,
                "snr_nli_db": _json_float(analytic_snr[index]),
                "nli_power_w": _json_float(nli[index]),
                "signal_power_w": _json_float(signal[index]),
            }
            for index in range(n_channels)
        ],
    }


def _error_row(
    tag: str,
    dataset: dict[str, Any],
    config_name: str,
    model_name: str,
    *,
    pump_model: str,
    error: Exception,
) -> dict[str, Any]:
    """记录单行失败，不中断整个对照。"""
    return {
        "dataset": tag,
        "n_channels": dataset.get("n_channels"),
        "config": config_name,
        "model": model_name,
        "pump_model": pump_model,
        "pump_depletion": model_name in DEPLETION_AWARE_MODELS,
        "error": f"{type(error).__name__}: {error}",
    }


def _dataset_pumps(dataset: dict[str, Any], reference: dict[str, Any], config_name: str) -> list[RamanPumpConfig]:
    """给跳过判断用：返回该配置的泵浦。"""
    config = (dataset.get("configs") or {}).get(config_name, {})
    return pumps_from_config(config_name, config, reference)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    """打印中心信道残差表。"""
    selected = [row for row in rows if "error" not in row and row.get("model") in DEPLETION_AWARE_MODELS]
    if not selected:
        return
    print(f"\n{'dataset':>10} {'config':>14} {'model':>20} {'wave':>9} {'analytic':>9} {'delta':>8}")
    for row in selected:
        wave = row.get("waveform_center_snr_nli_db")
        analytic = row.get("analytic_center_snr_nli_db")
        delta = row.get("center_delta_db")
        wave_text = "n/a" if wave is None else f"{wave:.3f}"
        analytic_text = "n/a" if analytic is None else f"{analytic:.3f}"
        delta_text = "n/a" if delta is None else f"{delta:+.3f}"
        print(
            f"{row['dataset']:>10} {row['config']:>14} {row['model']:>20} "
            f"{wave_text:>9} {analytic_text:>9} {delta_text:>8}"
        )


def _resolve_reference(data_dir: Path, reference: str) -> Path:
    """解析参考路径；裸文件名按 ``--data-dir`` 拼接。"""
    path = Path(reference)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return data_dir / path
    return path


def main() -> None:
    """跑全 WDM 对照并写 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"参考数据目录（默认 {DEFAULT_DATA_DIR}）",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE_NAME,
        help=f"参考 JSON 文件名或路径（默认 {DEFAULT_REFERENCE_NAME}）",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help=f"输出 JSON（默认 {DEFAULT_OUTPUT}）")
    parser.add_argument("--skip-numerical", action="store_true", help="跳过 numerical_dra（152ch 较慢）")
    parser.add_argument(
        "--include-pump-mismatch",
        action="store_true",
        help="参考 pump_model=undepleted 时也跑带泵配置（默认跳过，避免口径错配）",
    )
    parser.add_argument(
        "--signal-polarizations",
        type=int,
        choices=(1, 2),
        default=1,
        help="ODE 信号状态代表的偏振数：1=历史每偏振驱动（默认），2=双偏振总功率驱动",
    )
    args = parser.parse_args()

    reference_path = _resolve_reference(args.data_dir, args.reference)
    if not reference_path.exists():
        if not args.data_dir.exists():
            raise SystemExit(
                f"reference data directory not found: {args.data_dir}\n"
                "WP7 writes the wideband references to examples/dra_reference_wide/ in physical-nlo-eval; "
                f"copy c57.json / cl152.json into {args.data_dir}/ first."
            )
        raise SystemExit(
            f"reference JSON not found: {reference_path}\n"
            f"expected {args.data_dir}/c57.json or {args.data_dir}/cl152.json from WP7"
        )

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    plan = plan_from_reference(reference)
    fiber, fiber_metadata = fiber_from_reference(reference)
    pump_model = str(reference.get("pump_model") or "depleted")
    if reference.get("pump_model") is None:
        _warn(f"{reference_path} has no pump_model; assuming depleted (model default)")

    models = MODELS if not args.skip_numerical else tuple(name for name in MODELS if name != "numerical_dra")
    datasets = _datasets_from_reference(reference)
    if not datasets:
        raise SystemExit(f"{reference_path} has no datasets or configs")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for tag, dataset in datasets.items():
        configs = dataset.get("configs") or {}
        for config_name, config in configs.items():
            pumps = _dataset_pumps(dataset, reference, config_name)
            if pump_model == "undepleted" and pumps and not args.include_pump_mismatch:
                skipped.append(
                    {
                        "dataset": tag,
                        "config": config_name,
                        "reason": "reference pump_model=undepleted but analytic uses pump_depletion=True",
                    }
                )
                continue
            for model_name in models:
                started = time.time()
                try:
                    row = _row(
                        tag,
                        dataset,
                        config_name,
                        config,
                        plan,
                        fiber,
                        model_name,
                        reference,
                        pump_model=pump_model,
                        launch_power_dbm=plan["launch_power_dbm_per_pol"],
                        signal_polarizations=args.signal_polarizations,
                    )
                except Exception as error:  # 记录失败，不让基准整体停掉
                    row = _error_row(
                        tag, dataset, config_name, model_name, pump_model=pump_model, error=error
                    )
                row["elapsed_s"] = float(time.time() - started)
                rows.append(row)
                print(
                    f"{tag} {config_name} {model_name}: "
                    f"center delta={row.get('center_delta_db')} ({row['elapsed_s']:.1f}s)",
                    flush=True,
                )

    output = {
        "generated_by": "scripts/dra_wideband_compare.py",
        "reference_path": str(reference_path),
        "reference_pump_model": pump_model,
        "reference_note": (
            "Waveform NLI is the residual after one complex common-gain fit inside the channel band; "
            "analytic NLI is the transmitter-plane GN power. Compare SNR trends and residual dB, "
            "not absolute NLI across the two receiver/bandwidth conventions."
        ),
        "plan": plan,
        "fiber": fiber_metadata,
        "grid": {
            "type": "SpectrumGrid",
            "channel_bandwidth_source": "symbol_rate",
            "channel_bandwidth_hz": float(plan["symbol_rate_hz"]),
        },
        "models": list(models),
        "edfa_noise_figure_db": EDFA_NOISE_FIGURE_DB,
        "skipped": skipped,
        "rows": rows,
    }
    if int(args.signal_polarizations) != 1:
        output["signal_polarizations"] = int(args.signal_polarizations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"\nwrote {args.output}")
    if skipped:
        print(f"skipped {len(skipped)} pump-mismatched config(s) (reference pump_model={pump_model})")
    _print_summary(rows)


if __name__ == "__main__":
    main()
