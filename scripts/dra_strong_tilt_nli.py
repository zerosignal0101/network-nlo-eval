#!/usr/bin/env python3
"""复核强增益倾斜下解析 NLI 与物理波形的逐信道差异。

背景：physical-nlo-eval 的 ``cl152`` ``bwd_sixpumps`` 在逐频箱泵浦耗尽
（``depletion_model="spectral"``）并打开泵浦-泵浦拉曼转移
（``pump_pump_raman=True``）后，中心信道波形 SNR_NLI 从 31.45 dB 降到
28.19 dB，而 network-nlo-eval 的 ``hybrid_raman_paper`` 仍是 31.45 dB。
本脚本把两侧都展开到逐信道，用同一份光纤/泵浦/栅格口径比较残差，并回答
三个问题：

1. 残差是全带还是集中在中心/带边？
2. ``hybrid_raman_paper``、``pcfm_h3_raman(egn="auto")``、
   ``numerical_dra`` 三种解析模型是否一致？
3. 双偏振驱动（``signal_polarizations=2``）是否改变残差方向和量级？

物理侧只读调用 physical-nlo-eval：
``DistributedRamanFiber`` 跑一次非线性 SSFM，线性参考复用
``examples/diagnose_dra_profile.py::_linear_forward`` 的频域精确解，
逐信道 NLI 复用 ``examples/generate_dra_reference.py::band_nli_metrics``。
解析侧直接构造 ``SpectrumGrid(152, 75e9, 190.15e12)`` 并逐信道取
``spm_noise_w + xpm_noise_w``。

用法::

    PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python \
        scripts/dra_strong_tilt_nli.py \
        --output results/raman/strong_tilt_nli.json

物理 SSFM 是长任务（cl152 在 0.488 m 步长下约数分钟）；脚本会把
``nl/lin`` 波形缓存到 ``--cache-dir``（默认 ``/tmp/dra_strong_tilt_nli``），
带指纹校验，重复运行会自动复用。缓存文件不提交。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PHYSICAL_REPO = Path("/home/zsig/Documents/gits/physical-nlo-eval")
DEFAULT_PLAN = "cl152"
DEFAULT_CONFIG = "bwd_sixpumps"
DEFAULT_STEP_M = 0.48828125
DEFAULT_SPECTRAL_BINS = 256
DEFAULT_ITERATIONS = 32
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "dra_strong_tilt_nli"
DEFAULT_OUTPUT = Path("results/raman/strong_tilt_nli.json")

#: 解析侧逐信道输出的模型标签（numerical_dra 只支持单偏振驱动）。
MODEL_FACTORS: dict[str, tuple[int, ...]] = {
    "hybrid_raman_paper": (1, 2),
    "pcfm_h3_raman": (1, 2),
    "numerical_dra": (1,),
}
DEPLETION_AWARE_MODELS = frozenset({"hybrid_raman_paper", "pcfm_h3_raman"})
C_LIGHT_M_S = 2.99792458e8


def _json_float(value: Any) -> float | None:
    """把 NaN/Inf 转成 ``None``，保证输出是合法 JSON。"""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _nan_stat(values: np.ndarray, reducer: Any) -> float | None:
    """忽略 NaN 的统计量；全 NaN 时返回 ``None``。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return _json_float(reducer(finite))


def _bootstrap_physical(repo: Path) -> None:
    """把 physical-nlo-eval 的 ``src``/``examples`` 加到 ``sys.path``。

    Args:
        repo: physical-nlo-eval 仓库根目录（只读）。

    Raises
    ------
        SystemExit: 目录结构不符。
    """
    src = repo / "src"
    examples = repo / "examples"
    if not (src / "physics_nlo_eval").is_dir() or not examples.is_dir():
        raise SystemExit(f"{repo} does not look like physical-nlo-eval")
    for path in (str(examples), str(src)):
        if path not in sys.path:
            sys.path.append(path)


def load_reference(repo: Path, plan: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """读取 physical 参考 JSON 与输入波形。

    Args:
        repo: physical-nlo-eval 根目录。
        plan: 计划名（``cl152`` / ``c57``）。

    Returns
    -------
        ``(metadata, waveform, offsets_hz)``。``waveform`` 形状为
        ``(2, n_samples)``。若 npz 里没有 ``offsets_<plan>``，用 JSON 的
        ``plan`` 首末频偏线性重建。

    Raises
    ------
        SystemExit: 缺少 JSON 或输入波形键。
    """
    base = repo / "examples" / "dra_reference_wide"
    json_path = base / f"{plan}.json"
    npz_path = base / f"{plan}.npz"
    if not json_path.is_file() or not npz_path.is_file():
        raise SystemExit(f"missing reference files under {base} for plan {plan!r}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    data = np.load(npz_path)
    input_key = f"input_{plan}"
    if input_key not in data:
        raise SystemExit(f"{npz_path} has no {input_key!r}; keys={sorted(data.files)}")
    waveform = np.asarray(data[input_key])
    offsets_key = f"offsets_{plan}"
    if offsets_key in data:
        offsets = np.asarray(data[offsets_key], dtype=np.float64)
    else:
        plan_meta = metadata["plan"]
        offsets = np.linspace(
            float(plan_meta["first_channel_offset_hz"]),
            float(plan_meta["last_channel_offset_hz"]),
            int(plan_meta["n_channels"]),
        )
    return metadata, waveform, offsets


def build_physical_fiber(
    metadata: dict[str, Any],
    config: str,
    *,
    step_m: float,
    pump_pump: bool,
    spectral_bins: int,
    iterations: int,
):
    """按参考 JSON 构造 physical ``DistributedRamanFiber``。

    Args:
        metadata: 参考 JSON。
        config: 泵浦配置名。
        step_m: SSFM 粗步长 (m)。
        pump_pump: 是否打开泵浦-泵浦拉曼转移（要求 spectral）。
        spectral_bins: 谱域耗尽的频箱数。
        iterations: 后向泵不动点迭代上限。

    Returns
    -------
        配置好的 :class:`DistributedRamanFiber`（未传播）。
    """
    from physics_nlo_eval.channel import DistributedRamanFiber, PumpSpec

    fiber_meta = metadata["fiber"]
    pumps = [
        PumpSpec(
            wavelength_nm=float(entry["wavelength_nm"]),
            power_w=float(entry["power_w"]),
            direction=str(entry["direction"]),
            attenuation_db_per_km=float(entry["attenuation_db_per_km"]),
        )
        for entry in metadata["pump"]["configs"][config]
    ]
    return DistributedRamanFiber(
        length_m=float(fiber_meta["length_km"]) * 1e3,
        alpha_db_per_km=float(fiber_meta["alpha_db_per_km"]),
        beta2_ps2_per_km=float(fiber_meta["beta2_ps2_per_km"]),
        beta3_ps3_per_km=float(fiber_meta["beta3_ps3_per_km"]),
        pumps=pumps,
        coarse_step_size_m=float(step_m),
        n2_si=float(fiber_meta["n2_si"]),
        a_eff_um2=float(fiber_meta["a_eff_um2"]),
        pump_model="depleted",
        depletion_model="spectral",
        spectral_bins=int(spectral_bins),
        depleted_max_iterations=int(iterations),
        pump_pump_raman=bool(pump_pump),
    )


def _fingerprint(metadata: dict[str, Any], config: str, **options: Any) -> str:
    """给缓存波形生成一个可读的指纹。"""
    payload = {
        "plan": metadata["plan"],
        "fiber": metadata["fiber"],
        "config": config,
        "pumps": metadata["pump"]["configs"][config],
        **options,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def run_physical_waveform(
    metadata: dict[str, Any],
    waveform: np.ndarray,
    config: str,
    *,
    step_m: float,
    pump_pump: bool,
    spectral_bins: int,
    iterations: int,
    cache_dir: Path,
    reuse_cache: bool = True,
) -> dict[str, Any]:
    """跑一次物理非线性 SSFM，并构造频域精确线性参考。

    The linear reference reuses
    ``examples/diagnose_dra_profile.py::_linear_forward`` so that both sides
    see the same pump profiles and the same ``n/(n-1)`` frequency grid.

    Args:
        metadata: 参考 JSON。
        waveform: 输入波形 ``(2, n_samples)``。
        config: 泵浦配置名。
        step_m: SSFM 粗步长。
        pump_pump: 是否打开泵浦-泵浦拉曼。
        spectral_bins: 谱域耗尽频箱数。
        iterations: 不动点迭代上限。
        cache_dir: 波形缓存目录。
        reuse_cache: 指纹一致时复用缓存。

    Returns
    -------
        含 ``nonlinear``、``linear``、``fiber``、``elapsed_s`` 与元数据的字典。
    """
    from diagnose_dra_profile import _linear_forward
    from physics_nlo_eval.core.signal import Signal, SignalContext

    plan = metadata["plan"]
    waveform_hash = hashlib.sha256(np.ascontiguousarray(waveform).tobytes()).hexdigest()[:16]
    fingerprint = _fingerprint(
        metadata,
        config,
        step_m=float(step_m),
        pump_pump=bool(pump_pump),
        spectral_bins=int(spectral_bins),
        iterations=int(iterations),
        num_samples=int(waveform.shape[-1]),
        waveform_sha256=waveform_hash,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{plan['name']}_{config}_pp{int(pump_pump)}_step{step_m:.9g}_{fingerprint}"
    cache_npz = cache_dir / f"{stem}.npz"
    cache_meta = cache_dir / f"{stem}.json"

    fiber = build_physical_fiber(
        metadata,
        config,
        step_m=step_m,
        pump_pump=pump_pump,
        spectral_bins=spectral_bins,
        iterations=iterations,
    )

    elapsed = 0.0
    cached = reuse_cache and cache_npz.is_file() and cache_meta.is_file()
    if cached:
        stored = np.load(cache_npz)
        nonlinear = np.asarray(stored["nonlinear"])
        linear = np.asarray(stored["linear"])
        elapsed = float(json.loads(cache_meta.read_text(encoding="utf-8"))["elapsed_s"])
        # 缓存波形是在同一 fiber 参数下算出的；把泵浦剖面补算出来，供
        # on-off 增益与线性口径审计使用。
        from physics_nlo_eval.channel.dra import signal_spectrum_bins

        input_power_w = float(np.sum(np.mean(np.abs(waveform.astype(np.complex128)) ** 2, axis=1)))
        bin_freqs, bin_weights = signal_spectrum_bins(
            waveform.astype(np.complex128),
            float(plan["sample_rate_hz"]),
            float(plan["center_frequency_hz"]),
            int(spectral_bins),
        )
        (
            fiber._depleted_pump_z,  # noqa: SLF001 - 只读诊断需要剖面
            fiber._depleted_pump_power,  # noqa: SLF001
            fiber._depleted_signal_power,  # noqa: SLF001
            fiber._depletion_diagnostics,  # noqa: SLF001
        ) = fiber._solve_depleted_profiles(  # noqa: SLF001
            input_power_w,
            float(plan["center_frequency_hz"]),
            bin_freqs_hz=bin_freqs,
            bin_weights=bin_weights,
        )
    else:
        ctx = SignalContext(
            sample_rate=float(plan["sample_rate_hz"]),
            center_freq=float(plan["center_frequency_hz"]),
            n_pol=2,
            sps=int(plan["sps"]),
            sym_rate=float(plan["symbol_rate_hz"]),
        )
        signal = Signal(waveform.astype(np.complex64), ctx)
        started = time.perf_counter()
        nonlinear = fiber.forward(signal).data
        elapsed = time.perf_counter() - started
        profiles = {
            "z_m": np.asarray(fiber._depleted_pump_z),  # noqa: SLF001
            "pump_w": np.asarray(fiber._depleted_pump_power),  # noqa: SLF001
        }
        linear = _linear_forward(metadata, waveform, fiber, profiles)
        np.savez_compressed(cache_npz, nonlinear=nonlinear, linear=linear)
        cache_meta.write_text(
            json.dumps({"elapsed_s": elapsed, "fingerprint": fingerprint}, indent=1),
            encoding="utf-8",
        )

    return {
        "nonlinear": np.asarray(nonlinear),
        "linear": np.asarray(linear),
        "fiber": fiber,
        "elapsed_s": float(elapsed),
        "cached": bool(cached),
        "fingerprint": fingerprint,
    }


def per_channel_waveform_metrics(
    nonlinear: np.ndarray,
    linear: np.ndarray,
    offsets_hz: np.ndarray,
    *,
    sample_rate_hz: float,
    bandwidth_hz: float,
    center_frequency_hz: float,
    input_signal_power_w: np.ndarray | None = None,
    on_off_gain_linear: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """用 physical ``band_nli_metrics`` 逐信道计算信号/NLI/SNR。

    Args:
        nonlinear: SSFM 输出波形 ``(2, n_samples)``。
        linear: 频域精确线性参考。
        offsets_hz: 逐信道频偏。
        sample_rate_hz: 全带宽采样率。
        bandwidth_hz: NLI 分析带宽（这里取符号率）。
        center_frequency_hz: 光载波中心频率。
        input_signal_power_w: 逐信道输入带内功率；缺省时用 1 mW/偏振 ×2。
        on_off_gain_linear: 逐信道拉曼开关增益（线性）。

    Returns
    -------
        逐信道字典列表，含 ``signal_power_w``、``nli_power_w``、
        ``snr_nli_db``、``common_gain_db`` 与 ``net_gain_db``。
    """
    import generate_dra_reference as generator

    rows: list[dict[str, Any]] = []
    for index, offset in enumerate(np.asarray(offsets_hz, dtype=np.float64)):
        power, nli, raw_nli, gain_db, phase_deg = generator.band_nli_metrics(
            nonlinear,
            linear,
            float(sample_rate_hz),
            float(offset),
            float(bandwidth_hz),
        )
        if input_signal_power_w is None:
            input_power = 2.0e-3
        else:
            input_power = float(input_signal_power_w[index])
        net_gain_db = float(10.0 * np.log10(power / input_power)) if power > 0.0 and input_power > 0.0 else None
        rows.append(
            {
                "channel": int(index),
                "freq_offset_hz": float(offset),
                "freq_hz": float(center_frequency_hz) + float(offset),
                "signal_power_w": _json_float(power),
                "nli_power_w": _json_float(nli),
                "nli_power_raw_w": _json_float(raw_nli),
                "snr_nli_db": _json_float(10.0 * np.log10(power / nli)) if nli > 0.0 else None,
                "common_gain_db": _json_float(gain_db),
                "common_phase_deg": _json_float(phase_deg),
                "net_gain_db": _json_float(net_gain_db),
                "on_off_gain_db": (
                    _json_float(10.0 * np.log10(max(float(on_off_gain_linear[index]), 1e-300)))
                    if on_off_gain_linear is not None
                    else None
                ),
            }
        )
    return rows


def _fiber_config(metadata: dict[str, Any]):
    """把参考 JSON 的 β2/β3 换算成 ``FiberSpanConfig``。

    Args:
        metadata: 参考 JSON。

    Returns
    -------
        ``FiberSpanConfig``（平坦 alpha、1550 nm 参考）。
    """
    from network_nlo_eval.network.elements import FiberSpanConfig

    fiber = metadata["fiber"]
    beta2_si = float(fiber["beta2_ps2_per_km"]) * 1e-24 / 1e3
    dispersion = -(2.0 * np.pi * C_LIGHT_M_S / (1550e-9) ** 2) * beta2_si * 1e6
    beta3_si = float(fiber["beta3_ps3_per_km"]) * 1e-36 / 1e3
    slope = (beta3_si * (2.0 * np.pi * C_LIGHT_M_S) ** 2 / (1550e-9) ** 2 - 2.0 * 1550e-9 * dispersion * 1e-6) / (
        (1550e-9) ** 2 * 1e3
    )
    return FiberSpanConfig(
        length_km=float(fiber["length_km"]),
        attenuation_db_km_ref=float(fiber["alpha_db_per_km"]),
        use_attenuation_polynomial=False,
        dispersion_parameter_d=float(dispersion),
        dispersion_slope_s=float(slope),
        nonlinear_index_n2=float(fiber["n2_si"]),
        effective_area_um2_ref=float(fiber["a_eff_um2"]),
    )


def _raman_pumps(metadata: dict[str, Any], config: str):
    """参考 JSON 泵浦 -> ``RamanPumpConfig`` 列表。"""
    from network_nlo_eval.network.elements import RamanPumpConfig

    return [
        RamanPumpConfig(
            wavelength_nm=float(entry["wavelength_nm"]),
            power_w=float(entry["power_w"]),
            direction=str(entry["direction"]),
            attenuation_db_per_km=float(entry["attenuation_db_per_km"]),
        )
        for entry in metadata["pump"]["configs"][config]
    ]


def build_analytic_inputs(metadata: dict[str, Any], config: str):
    """构造解析侧栅格、光纤与泵浦。

    Returns
    -------
        ``(grid, fiber_config, pumps)``。
    """
    from network_nlo_eval.core.spectrum import SpectrumGrid

    plan = metadata["plan"]
    grid = SpectrumGrid(
        int(plan["n_channels"]),
        float(plan["channel_spacing_hz"]),
        float(plan["center_frequency_hz"]),
    )
    return grid, _fiber_config(metadata), _raman_pumps(metadata, config)


def evaluate_analytic(
    model_name: str,
    grid,
    fiber_config,
    pumps,
    *,
    symbol_rate_hz: float,
    power_w: np.ndarray,
    signal_polarizations: int,
) -> dict[str, Any]:
    """跑一个解析模型，返回逐信道信号/SPM/XPM/SNR。

    Args:
        model_name: ``MODEL_FACTORS`` 中的模型名。
        grid: 全 WDM 栅格。
        fiber_config: 参考光纤。
        pumps: 泵浦列表。
        symbol_rate_hz: 符号率。
        power_w: 逐信道入纤功率（每偏振）。
        signal_polarizations: ODE 信号状态代表的偏振数。

    Returns
    -------
        含 ``signal_power_w``、``spm_noise_w``、``xpm_noise_w``、
        ``snr_nli_db`` 与 ``elapsed_s`` 的字典。
    """
    from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
    from network_nlo_eval.network.elements import EDFAConfig

    options: dict[str, Any] = {
        "symbol_rate_hz": float(symbol_rate_hz),
        "channel_bandwidth_source": "symbol_rate",
        "raman_pumps": pumps,
        "raman_ase": False,
    }
    if model_name in DEPLETION_AWARE_MODELS:
        options["pump_depletion"] = True
        options["signal_polarizations"] = int(signal_polarizations)
    if model_name == "hybrid_raman_paper":
        options["nli_form"] = "paper"
    if model_name == "pcfm_h3_raman":
        options["egn"] = "auto"
    model = create_model(model_name, grid, fiber_config, **options)
    path = MultiSpanOpticalPath([fiber_config], [EDFAConfig(noise_figure_db=5.5)])
    started = time.perf_counter()
    result = model.evaluate_path(path, np.asarray(power_w, dtype=np.float64))
    elapsed = time.perf_counter() - started
    signal = np.asarray(result.signal_power_w, dtype=np.float64)
    spm = np.asarray(result.spm_noise_w, dtype=np.float64)
    xpm = np.asarray(result.xpm_noise_w, dtype=np.float64)
    nli = spm + xpm
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10.0 * np.log10(np.divide(signal, nli, out=np.full_like(signal, np.nan), where=nli > 0.0))
    return {
        "signal_power_w": signal,
        "spm_noise_w": spm,
        "xpm_noise_w": xpm,
        "nli_power_w": nli,
        "snr_nli_db": snr,
        "elapsed_s": float(elapsed),
    }


def residual_stats(
    analytic_snr_db: np.ndarray,
    waveform_snr_db: np.ndarray,
    *,
    center_index: int,
    net_gain_db: np.ndarray | None = None,
) -> dict[str, Any]:
    """逐信道残差 ``解析 - 波形`` 的统计量。

    Args:
        analytic_snr_db: 解析逐信道 SNR。
        waveform_snr_db: 波形逐信道 SNR。
        center_index: 中心信道索引。
        net_gain_db: 逐信道净增益，用于给出残差与增益倾斜的相关系数。

    Returns
    -------
        含 ``mean``/``rms``/``min``/``max``/``center``/两端信道与最大残差
        位置的字典。
    """
    analytic = np.asarray(analytic_snr_db, dtype=np.float64)
    waveform = np.asarray(waveform_snr_db, dtype=np.float64)
    delta = analytic - waveform
    finite = np.isfinite(delta)
    if not np.any(finite):
        return {"count": 0}
    delta_finite = delta[finite]
    index_max = int(np.nanargmax(delta))
    index_min = int(np.nanargmin(delta))
    result: dict[str, Any] = {
        "count": int(np.count_nonzero(finite)),
        "mean_db": _nan_stat(delta, np.mean),
        "rms_db": _json_float(np.sqrt(np.mean(delta_finite**2))),
        "min_db": _nan_stat(delta, np.min),
        "max_db": _nan_stat(delta, np.max),
        "max_channel": index_max,
        "min_channel": index_min,
        "center_db": _json_float(delta[center_index]),
        "edge_low_db": _json_float(delta[0]),
        "edge_high_db": _json_float(delta[-1]),
    }
    if net_gain_db is not None:
        gain = np.asarray(net_gain_db, dtype=np.float64)
        mask = finite & np.isfinite(gain)
        if np.count_nonzero(mask) >= 2:
            result["corr_with_net_gain"] = _json_float(np.corrcoef(delta[mask], gain[mask])[0, 1])
    return result


def representative_indices(n_channels: int, center_index: int) -> list[int]:
    """返回报告表格用的代表信道（两端、中心与等间隔采样）。"""
    base = sorted(set(int(round(value)) for value in np.linspace(0, n_channels - 1, 10)))
    return sorted(set(base + [center_index - 1, center_index, center_index + 1]))


def _per_channel_analytic_rows(
    result: dict[str, Any],
    offsets_hz: np.ndarray,
    center_frequency_hz: float,
) -> list[dict[str, Any]]:
    """把解析数组整理成逐信道字典。"""
    rows = []
    for index in range(len(offsets_hz)):
        signal = float(result["signal_power_w"][index])
        nli = float(result["nli_power_w"][index])
        rows.append(
            {
                "channel": int(index),
                "freq_offset_hz": float(offsets_hz[index]),
                "freq_hz": float(center_frequency_hz) + float(offsets_hz[index]),
                "signal_power_w": _json_float(signal),
                "spm_noise_w": _json_float(result["spm_noise_w"][index]),
                "xpm_noise_w": _json_float(result["xpm_noise_w"][index]),
                "nli_power_w": _json_float(nli),
                "snr_nli_db": _json_float(result["snr_nli_db"][index]),
            }
        )
    return rows


def _fmt(value: Any, digits: int = 3) -> str:
    """表格里安全格式化浮点数。"""
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: Any, digits: int = 3) -> str:
    """带显式正号的格式化。"""
    if value is None:
        return "-"
    return f"{float(value):+.{digits}f}"


def _print_table(rows: list[dict[str, Any]], models: dict[str, dict[str, Any]], center_index: int) -> None:
    """打印中心/带边/最差信道的逐信道表。"""
    print("\n代表信道（SNR_NLI, dB）")
    header = f"{'ch':>4} {'offset[GHz]':>11} {'wave':>8}"
    for label in models:
        header += f" {label:>22}"
    print(header)
    for row in rows:
        line = f"{row['channel']:>4} {row['freq_offset_hz'] / 1e9:>11.1f} {_fmt(row['snr_nli_db']):>8}"
        for _label, data in models.items():
            analytic = data["snr_per_channel"][row["channel"]]
            delta = data["delta_per_channel"][row["channel"]]
            line += f" {_fmt(analytic):>10} ({_fmt_signed(delta):>7})"
        print(line)
    print(f"(center channel index = {center_index})")


def _build_8ch_input(
    offsets_hz: np.ndarray,
    powers_dbm_per_pol: np.ndarray,
    *,
    center_frequency_hz: float,
    symbol_rate_hz: float,
    sample_rate_hz: float,
    num_symbols: int,
    sps: int,
    roll_off: float,
    seed: int,
) -> np.ndarray:
    """用 physical 的 TX 链生成一个 8ch WDM 输入波形。

    这是受控强倾斜算例：8 个 75 GHz 间隔的信道，逐信道发射功率按
    ``powers_dbm_per_pol`` 设定，泵浦与光纤保持 cl152 口径。发射功率倾斜
    模拟全带内的强干扰信道，用于隔离 GN XPM 项在强功率不平衡下的表现。

    Args:
        offsets_hz: 逐信道频偏。
        powers_dbm_per_pol: 逐信道发射功率 (dBm/偏振)。
        center_frequency_hz: 光载波中心频率。
        symbol_rate_hz: 符号率。
        sample_rate_hz: 合波后的采样率。
        num_symbols: 每信道符号数。
        sps: 每符号样点数。
        roll_off: RRC 滚降。
        seed: 比特随机种子。

    Returns
    -------
        ``(2, n_samples)`` 复基带波形。
    """
    from physics_nlo_eval.channel import ChannelSpec, WDMMux, WDMPlan
    from physics_nlo_eval.tx.mapper import Mapper
    from physics_nlo_eval.tx.optics import IQModulator
    from physics_nlo_eval.tx.rrc import RRCFilter

    rng = np.random.default_rng(seed)
    mapper = Mapper(modulation="DP-QPSK", power_norm=True, sym_rate=symbol_rate_hz)
    rrc = RRCFilter(sps=sps, roll_off=roll_off)
    iq_mod = IQModulator(launch_power_dbm_per_pol=0.0)
    bits_per_channel = num_symbols * mapper.bits_per_symbol * mapper.n_pol
    bits = rng.integers(0, 2, size=(len(offsets_hz), bits_per_channel), dtype=np.int8).astype(bool)
    channel_signals = [iq_mod.forward(rrc.forward(mapper.forward(bits[index]))) for index in range(len(offsets_hz))]
    wdm_plan = WDMPlan(
        reference_freq_hz=float(center_frequency_hz),
        channels=[
            ChannelSpec(
                channel_id=index,
                freq_offset_hz=float(offsets_hz[index]),
                power_dbm_per_pol=float(powers_dbm_per_pol[index]),
            )
            for index in range(len(offsets_hz))
        ],
    )
    mux = WDMMux(wdm_plan, target_sample_rate_hz=float(sample_rate_hz))
    return np.asarray(mux.combine(channel_signals).data)


def run_8ch_ablation(
    metadata: dict[str, Any],
    config: str,
    *,
    step_m: float,
    num_symbols: int,
    seed: int,
    cache_dir: Path,
    reuse_cache: bool,
) -> dict[str, Any]:
    """跑 8ch 强功率倾斜 + 带边消融，比较三种解析模型。

    The 8 channels keep the realistic 75 GHz spacing and the cl152 pumps/fibre,
    but the launch powers are tilted by up to 15 dB.  Two cases are compared:
    ``edge_on`` (all 8 channels) and ``edge_off`` (the four strongest channels
    set to -60 dBm), which isolates the contribution of the strong neighbours
    to the weak-channel NLI.  Because the input power changes, the pump
    depletion also changes; the reported net gains document that confound.

    Args:
        metadata: 参考 JSON。
        config: 泵浦配置名。
        step_m: SSFM 步长（建议比全带 0.488 m 更细）。
        num_symbols: 每信道符号数。
        seed: 比特种子。
        cache_dir: 波形缓存目录。
        reuse_cache: 是否复用缓存。

    Returns
    -------
        两个 case 的逐信道物理/解析结果。
    """
    plan = metadata["plan"]
    center_frequency_hz = float(plan["center_frequency_hz"])
    symbol_rate_hz = float(plan["symbol_rate_hz"])
    n_channels = 8
    spacing_hz = 75e9
    sample_rate_hz = 1.536e12
    offsets = np.arange(n_channels) * spacing_hz - 0.5 * (n_channels - 1) * spacing_hz
    tilts_on = np.array([0.0, 0.0, 0.0, 0.0, 6.0, 9.0, 12.0, 15.0])
    cases = {
        "edge_on": tilts_on,
        "edge_off": np.array([0.0, 0.0, 0.0, 0.0, -60.0, -60.0, -60.0, -60.0]),
    }

    plan8 = copy.deepcopy(plan)
    plan8.update(
        {
            "name": f"{plan['name']}_8ch_ablation",
            "n_channels": n_channels,
            "channel_spacing_hz": spacing_hz,
            "center_frequency_hz": center_frequency_hz,
            "symbol_rate_hz": symbol_rate_hz,
            "sample_rate_hz": sample_rate_hz,
            "num_symbols": int(num_symbols),
            "first_channel_offset_hz": float(offsets[0]),
            "last_channel_offset_hz": float(offsets[-1]),
        }
    )
    metadata8 = {**metadata, "plan": plan8}

    grid, fiber_config, pumps = build_analytic_inputs(metadata8, config)
    result: dict[str, Any] = {
        "generated_by": "scripts/dra_strong_tilt_nli.py --ablation-only",
        "plan": "cl152-derived 8ch",
        "config": config,
        "pump_pump_raman": True,
        "step_m": float(step_m),
        "num_symbols": int(num_symbols),
        "spacing_hz": spacing_hz,
        "sample_rate_hz": sample_rate_hz,
        "offsets_hz": offsets.tolist(),
        "launch_power_dbm_per_pol": {name: values.tolist() for name, values in cases.items()},
        "cases": {},
    }

    for case_name, tilts in cases.items():
        print(f"[8ch] {case_name}: launch {tilts.tolist()} dBm/pol")
        waveform = _build_8ch_input(
            offsets,
            tilts,
            center_frequency_hz=center_frequency_hz,
            symbol_rate_hz=symbol_rate_hz,
            sample_rate_hz=sample_rate_hz,
            num_symbols=num_symbols,
            sps=int(plan["sps"]),
            roll_off=float(plan["roll_off"]),
            seed=seed,
        )
        physics = run_physical_waveform(
            metadata8,
            waveform,
            config,
            step_m=step_m,
            pump_pump=True,
            spectral_bins=int(DEFAULT_SPECTRAL_BINS),
            iterations=int(DEFAULT_ITERATIONS),
            cache_dir=cache_dir,
            reuse_cache=reuse_cache,
        )
        import generate_dra_reference as generator

        input_power = np.asarray(
            [
                generator.band_nli_metrics(
                    waveform,
                    waveform,
                    sample_rate_hz,
                    float(offset),
                    symbol_rate_hz,
                )[0]
                for offset in offsets
            ],
            dtype=np.float64,
        )
        on_off = np.asarray(physics["fiber"].on_off_gain(center_frequency_hz + offsets), dtype=np.float64)
        wave_rows = per_channel_waveform_metrics(
            physics["nonlinear"],
            physics["linear"],
            offsets,
            sample_rate_hz=sample_rate_hz,
            bandwidth_hz=symbol_rate_hz,
            center_frequency_hz=center_frequency_hz,
            input_signal_power_w=input_power,
            on_off_gain_linear=on_off,
        )
        wave_snr = np.asarray([row["snr_nli_db"] for row in wave_rows], dtype=np.float64)
        net_gain = np.asarray([row["net_gain_db"] for row in wave_rows], dtype=np.float64)

        power_w = np.where(tilts > -30.0, 1e-3 * 10.0 ** (tilts / 10.0), 0.0)
        models: dict[str, Any] = {}
        for model_name, factors in MODEL_FACTORS.items():
            for factor in factors:
                label = f"{model_name}/f{factor}"
                analytic = evaluate_analytic(
                    model_name,
                    grid,
                    fiber_config,
                    pumps,
                    symbol_rate_hz=symbol_rate_hz,
                    power_w=power_w,
                    signal_polarizations=factor,
                )
                analytic_snr = np.asarray(analytic["snr_nli_db"], dtype=np.float64)
                models[label] = {
                    "center_snr_nli_db": _json_float(analytic_snr[3]),
                    "snr_per_channel": analytic_snr.tolist(),
                    "residual": residual_stats(
                        analytic_snr,
                        wave_snr,
                        center_index=3,
                        net_gain_db=net_gain,
                    ),
                    "per_channel": _per_channel_analytic_rows(analytic, offsets, center_frequency_hz),
                }
                print(
                    f"  {label}: ch3 res={_fmt_signed(models[label]['residual'].get('center_db'))} dB, "
                    f"mean={_fmt_signed(models[label]['residual'].get('mean_db'))} dB"
                )
        result["cases"][case_name] = {
            "physical": {
                "elapsed_s": physics["elapsed_s"],
                "cache_hit": bool(physics["cached"]),
                "totals": {
                    "input_total_w": float(np.sum(np.mean(np.abs(waveform) ** 2, axis=1))),
                    "ssfm_total_output_w": float(np.sum(np.mean(np.abs(physics["nonlinear"]) ** 2, axis=1))),
                    "linear_total_output_w": float(np.sum(np.mean(np.abs(physics["linear"]) ** 2, axis=1))),
                },
                "net_gain_db": net_gain.tolist(),
                "on_off_gain_db": (10.0 * np.log10(np.maximum(on_off, 1e-300))).tolist(),
                "per_channel": wave_rows,
            },
            "analytic": models,
        }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """构造命令行参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-repo", type=Path, default=DEFAULT_PHYSICAL_REPO)
    parser.add_argument("--plan", default=DEFAULT_PLAN)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--step-m", type=float, default=DEFAULT_STEP_M)
    parser.add_argument("--spectral-bins", type=int, default=DEFAULT_SPECTRAL_BINS)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--pump-pump", dest="pump_pump", action="store_true", default=True)
    parser.add_argument("--no-pump-pump", dest="pump_pump", action="store_false")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-reuse-cache", dest="reuse_cache", action="store_false", default=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--probe-symbols",
        type=int,
        default=None,
        help="只取前 N 个符号做步长收敛探针（不改变其它参数）",
    )
    parser.add_argument(
        "--models",
        default=",".join(MODEL_FACTORS),
        help="逗号分隔的解析模型名，默认全部；传空串只跑物理侧",
    )
    parser.add_argument(
        "--ablation-8ch",
        action="store_true",
        help="跑 8ch 强功率倾斜 + 带边消融（单独写 --ablation-output）",
    )
    parser.add_argument(
        "--ablation-only",
        action="store_true",
        help="只跑 8ch 消融，跳过 152ch 主对照",
    )
    parser.add_argument("--ablation-step-m", type=float, default=0.1220703125)
    parser.add_argument("--ablation-symbols", type=int, default=128)
    parser.add_argument("--ablation-seed", type=int, default=20261104)
    parser.add_argument(
        "--ablation-output",
        type=Path,
        default=Path("results/raman/strong_tilt_nli_8ch.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行逐信道复核并写出 JSON。"""
    args = build_arg_parser().parse_args(argv)
    _bootstrap_physical(args.physical_repo)
    metadata, waveform, offsets = load_reference(args.physical_repo, args.plan)

    plan = metadata["plan"]
    center_index = int(plan["n_channels"]) // 2
    symbol_rate_hz = float(plan["symbol_rate_hz"])
    sample_rate_hz = float(plan["sample_rate_hz"])
    center_frequency_hz = float(plan["center_frequency_hz"])
    if args.probe_symbols is not None:
        samples_per_symbol = float(sample_rate_hz) / float(symbol_rate_hz)
        n_samples = int(round(int(args.probe_symbols) * samples_per_symbol))
        if n_samples < 1 or n_samples > waveform.shape[-1]:
            raise SystemExit(
                f"--probe-symbols={args.probe_symbols} gives {n_samples} samples, outside 1..{waveform.shape[-1]}"
            )
        waveform = waveform[..., :n_samples]
        print(f"[physical] probe: {args.probe_symbols} symbols -> {n_samples} samples")

    if args.ablation_only:
        ablation = run_8ch_ablation(
            metadata,
            args.config,
            step_m=args.ablation_step_m,
            num_symbols=args.ablation_symbols,
            seed=args.ablation_seed,
            cache_dir=args.cache_dir,
            reuse_cache=args.reuse_cache,
        )
        args.ablation_output.parent.mkdir(parents=True, exist_ok=True)
        args.ablation_output.write_text(json.dumps(ablation, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.ablation_output}")
        return 0

    print(
        f"[physical] {args.plan}:{args.config} step={args.step_m:g} m, "
        f"pp={args.pump_pump}, bins={args.spectral_bins}, iters={args.iterations}"
    )
    started = time.perf_counter()
    physics = run_physical_waveform(
        metadata,
        waveform,
        args.config,
        step_m=args.step_m,
        pump_pump=args.pump_pump,
        spectral_bins=args.spectral_bins,
        iterations=args.iterations,
        cache_dir=args.cache_dir,
        reuse_cache=args.reuse_cache,
    )
    print(f"[physical] SSFM {'cache hit' if physics['cached'] else 'done'} in {physics['elapsed_s']:.1f} s")

    # 输入逐信道带内功率（与输出同口径），用于净增益。
    import generate_dra_reference as generator

    input_power = np.asarray(
        [
            generator.band_nli_metrics(
                waveform,
                waveform,
                sample_rate_hz,
                float(offset),
                symbol_rate_hz,
            )[0]
            for offset in offsets
        ],
        dtype=np.float64,
    )
    freq_query = center_frequency_hz + offsets
    on_off_linear = np.asarray(physics["fiber"].on_off_gain(freq_query), dtype=np.float64)
    wave_rows = per_channel_waveform_metrics(
        physics["nonlinear"],
        physics["linear"],
        offsets,
        sample_rate_hz=sample_rate_hz,
        bandwidth_hz=symbol_rate_hz,
        center_frequency_hz=center_frequency_hz,
        input_signal_power_w=input_power,
        on_off_gain_linear=on_off_linear,
    )
    wave_snr = np.asarray([row["snr_nli_db"] for row in wave_rows], dtype=np.float64)
    wave_nli = np.asarray([row["nli_power_w"] for row in wave_rows], dtype=np.float64)
    net_gain = np.asarray([row["net_gain_db"] for row in wave_rows], dtype=np.float64)

    grid, fiber_config, pumps = build_analytic_inputs(metadata, args.config)
    launch_w = 1e-3 * 10 ** (float(plan["launch_power_dbm_per_pol"]) / 10.0)
    power_in = np.full(int(plan["n_channels"]), launch_w, dtype=np.float64)

    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    models: dict[str, dict[str, Any]] = {}
    for model_name in requested:
        for factor in MODEL_FACTORS[model_name]:
            label = f"{model_name}/f{factor}"
            print(f"[analytic] {label}")
            result = evaluate_analytic(
                model_name,
                grid,
                fiber_config,
                pumps,
                symbol_rate_hz=symbol_rate_hz,
                power_w=power_in,
                signal_polarizations=factor,
            )
            analytic_snr = np.asarray(result["snr_nli_db"], dtype=np.float64)
            models[label] = {
                "model": model_name,
                "signal_polarizations": factor,
                "elapsed_s": result["elapsed_s"],
                "center_snr_nli_db": _json_float(analytic_snr[center_index]),
                "center_nli_power_w": _json_float(result["nli_power_w"][center_index]),
                "delta_per_channel": analytic_snr - wave_snr,
                "snr_per_channel": analytic_snr,
                "residual": residual_stats(
                    analytic_snr,
                    wave_snr,
                    center_index=center_index,
                    net_gain_db=net_gain,
                ),
                "per_channel": _per_channel_analytic_rows(result, offsets, center_frequency_hz),
                "snr_min_db": _json_float(np.nanmin(analytic_snr)),
                "snr_max_db": _json_float(np.nanmax(analytic_snr)),
            }
            stats = models[label]["residual"]
            print(
                f"  center={_fmt(stats.get('center_db'))} dB, "
                f"mean={_fmt(stats.get('mean_db'))}, rms={_fmt(stats.get('rms_db'))}, "
                f"range=[{_fmt(stats.get('min_db'))}, {_fmt(stats.get('max_db'))}] dB"
            )

    represent = representative_indices(int(plan["n_channels"]), center_index)
    table_rows = [wave_rows[index] for index in represent]
    _print_table(table_rows, models, center_index)

    # 能量账：把 forward() 的诊断元数据完整保留，便于审计 pp 开关。
    energy = getattr(physics["fiber"], "_depletion_diagnostics", None) or {}
    ssfm_total = float(np.sum(np.mean(np.abs(physics["nonlinear"]) ** 2, axis=1)))
    linear_total = float(np.sum(np.mean(np.abs(physics["linear"]) ** 2, axis=1)))
    input_total = float(np.sum(np.mean(np.abs(waveform.astype(np.complex128)) ** 2, axis=1)))

    output: dict[str, Any] = {
        "generated_by": "scripts/dra_strong_tilt_nli.py",
        "physical_repo": str(args.physical_repo),
        "plan": plan,
        "config": args.config,
        "physical": {
            "waveform_model": {
                "pump_model": "depleted",
                "depletion_model": "spectral",
                "spectral_bins": int(args.spectral_bins),
                "pump_pump_raman": bool(args.pump_pump),
                "depleted_max_iterations": int(args.iterations),
                "coarse_step_m": float(args.step_m),
                "n2_si": float(metadata["fiber"]["n2_si"]),
            },
            "probe_symbols": args.probe_symbols,
            "num_samples": int(waveform.shape[-1]),
            "nli_bandwidth_hz": symbol_rate_hz,
            "elapsed_s": physics["elapsed_s"],
            "cache_hit": bool(physics["cached"]),
            "fingerprint": physics["fingerprint"],
            "totals": {
                "input_total_w": input_total,
                "ssfm_total_output_w": ssfm_total,
                "linear_total_output_w": linear_total,
                "ssfm_vs_linear_relative": (
                    float(abs(ssfm_total - linear_total) / linear_total) if linear_total > 0 else None
                ),
            },
            "energy": energy,
            "on_off_gain_db": {
                "center": _json_float(10.0 * np.log10(max(on_off_linear[center_index], 1e-300))),
                "low_offset": _json_float(10.0 * np.log10(max(on_off_linear[0], 1e-300))),
                "high_offset": _json_float(10.0 * np.log10(max(on_off_linear[-1], 1e-300))),
                "max": _json_float(10.0 * np.log10(max(float(np.max(on_off_linear)), 1e-300))),
                "min": _json_float(10.0 * np.log10(max(float(np.min(on_off_linear)), 1e-300))),
                "max_channel": int(np.argmax(on_off_linear)),
                "min_channel": int(np.argmin(on_off_linear)),
            },
            "net_gain_db": {
                "center": _json_float(net_gain[center_index]),
                "low_offset": _json_float(net_gain[0]),
                "high_offset": _json_float(net_gain[-1]),
                "max": _json_float(np.nanmax(net_gain)),
                "min": _json_float(np.nanmin(net_gain)),
                "max_channel": int(np.nanargmax(net_gain)),
                "min_channel": int(np.nanargmin(net_gain)),
            },
            "per_channel": wave_rows,
        },
        "analytic": {
            label: {key: value for key, value in data.items() if key not in {"delta_per_channel", "snr_per_channel"}}
            for label, data in models.items()
        },
        "comparison": {
            "waveform_center_snr_nli_db": _json_float(wave_snr[center_index]),
            "waveform_center_nli_power_w": _json_float(wave_nli[center_index]),
            "center_channel_index": center_index,
            "models": {
                label: {
                    "residual": data["residual"],
                    "analytic_center_snr_nli_db": data["center_snr_nli_db"],
                    "analytic_center_nli_power_w": data["center_nli_power_w"],
                }
                for label, data in models.items()
            },
        },
        "representative_channels": represent,
        "total_elapsed_s": float(time.perf_counter() - started),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.output}")

    if args.ablation_8ch:
        ablation = run_8ch_ablation(
            metadata,
            args.config,
            step_m=args.ablation_step_m,
            num_symbols=args.ablation_symbols,
            seed=args.ablation_seed,
            cache_dir=args.cache_dir,
            reuse_cache=args.reuse_cache,
        )
        args.ablation_output.parent.mkdir(parents=True, exist_ok=True)
        args.ablation_output.write_text(json.dumps(ablation, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.ablation_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
