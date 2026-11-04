"""7 信道无泵 XPM 缺口审计（只读 npz，可选步长探针）。

审计对象是 ``data/dra_waveform/dra_reference.{json,npz}``：3/5/7 信道、
100 km、50 GBd DP-QPSK、1 mW/偏振的无泵与泵浦波形参考。目标是判断
“7ch 无泵中心信道比解析 GN 低 3 dB”是波形提取/伪影，还是 GN 适用域问题。

审计分四层：

1. **逐信道**：用参考 JSON 相同的口径（50 GHz 理想矩形带、单复标量
   ``NL - a*LIN`` 投影）复算每个信道的 NLI/SNR，确认 7ch 缺口是否只在中心。
2. **频谱**：10/20/30/40/50 GHz 累积带宽、1 GHz 细粒度 PSD，判断 NLI 是
   白噪声、边缘集中还是窄带杂散。
3. **有限记录与投影**：把 256 symbols 切成 2/4/8/16 块看 NLI 方差与块间
   一致性；把分析带中心平移 ±5/±10 GHz 看投影口径敏感性。
4. **输入占用**：检查 ``input_*`` 是否存在相邻信道重叠、RRC 阻带泄漏、
   外信道与 Nyquist 的距离，以及 50 GHz/55 GHz 带内功率分数是否随 N 变化。

第五层是对照：``hybrid_raman_paper``（``nli_form="paper"`` 与 ``"integral"``）
和 ``numerical_dra`` 的逐信道 SPM/XPM，比较波形 NLI 与 GN XPM 随 N 的增长。

``--step-probe`` 是可选的决定性证据：用 ``physics-nlo-eval`` 的
``DistributedRamanFiber`` 直接对同一 ``input_*`` 波形重跑 500/250/125/62.5/
31.25 m 步长。它不修改 ``physical-nlo-eval``，只调用其公开接口；默认关闭，
因为默认审计只读 npz。

用法::

    PYTHONPATH=src python scripts/seven_ch_audit.py
    PYTHONPATH=src python scripts/seven_ch_audit.py --step-probe

输出：``results/raman/seven_ch_audit.json``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DATA_DIR = Path("data/dra_waveform")
REFERENCE_JSON = DATA_DIR / "dra_reference.json"
REFERENCE_NPZ = DATA_DIR / "dra_reference.npz"
OUTPUT_JSON = Path("results/raman/seven_ch_audit.json")

TAGS = ("3ch", "5ch", "7ch")
NOPUMP = "nopump"
ANALYSIS_BANDWIDTH_HZ = 50e9
CHANNEL_ROLLOFF_HALF_BANDWIDTH_HZ = 27.5e9
WIDTHS_HZ = (10e9, 20e9, 30e9, 40e9, 50e9)
SHIFTS_HZ = (-10e9, -5e9, 0.0, 5e9, 10e9)
BLOCK_COUNTS = (2, 4, 8, 16)
FINE_BIN_HZ = 1e9
FINE_SPAN_HZ = 40e9
STEP_PROBE_STEPS_M = (500.0, 250.0, 125.0, 62.5, 31.25, 15.625)
PUMPED_PROBE_CONFIGS = ("fwd_0p20w", "bwd_0p20w")
PUMPED_PROBE_STEPS_M = (500.0, 125.0, 31.25, 15.625)

#: 与 ``scripts/dra_waveform_compare.py`` 相同的解析对照口径。
FIBER_KWARGS = {
    "length_km": 100.0,
    "attenuation_db_km_ref": 0.2,
    "dispersion_parameter_d": 15.68076226687911,
    "dispersion_slope_s": 0.04123833468291227,
    "effective_area_um2_ref": 80.0,
    "nonlinear_index_n2": 2.6e-20,
}
SYMBOL_RATE_HZ = 50e9
LAUNCH_POWER_W = 1e-3


# ---------------------------------------------------------------------------
# 基础 NLI 提取
# ---------------------------------------------------------------------------


def _spectra(
    reference: dict[str, Any],
    arrays: Any,
    tag: str,
    config_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 ``(NL_fft, LIN_fft, freqs)``，形状均为 ``(n_pol, n_samples)`` / ``(n_samples,)``。"""
    dataset = reference["datasets"][tag]
    config = dataset["configs"][config_name]
    num_samples = int(dataset["num_samples"])
    sample_rate = float(dataset["sample_rate_hz"])
    nonlinear = arrays[config["waveform_nonlinear_key"]].astype(np.complex128)
    linear = arrays[config["waveform_linear_key"]].astype(np.complex128)
    return np.fft.fft(nonlinear, axis=1), np.fft.fft(linear, axis=1), np.fft.fftfreq(num_samples, 1.0 / sample_rate)


def _band_metrics(
    nl_spectrum: np.ndarray,
    lin_spectrum: np.ndarray,
    freqs: np.ndarray,
    offset_hz: float,
    bandwidth_hz: float = ANALYSIS_BANDWIDTH_HZ,
    shift_hz: float = 0.0,
) -> dict[str, float]:
    """在给定分析带内做单复标量投影，返回信号/NLI 功率与 SNR。"""
    num_samples = freqs.size
    mask = np.abs(freqs - (offset_hz + shift_hz)) <= bandwidth_hz / 2.0
    nl_band = nl_spectrum[:, mask]
    lin_band = lin_spectrum[:, mask]
    signal_power = float(np.sum(np.abs(nl_band) ** 2) / num_samples**2)
    linear_power = float(np.sum(np.abs(lin_band) ** 2) / num_samples**2)
    common_gain = np.sum(nl_band * np.conj(lin_band)) / np.sum(np.abs(lin_band) ** 2)
    residual = nl_band - common_gain * lin_band
    nli_power = float(np.sum(np.abs(residual) ** 2) / num_samples**2)
    snr_db = float(10.0 * np.log10(linear_power / nli_power)) if nli_power > 0.0 else float("inf")
    return {
        "signal_power_w": signal_power,
        "linear_power_w": linear_power,
        "nli_power_w": nli_power,
        "snr_nli_db": snr_db,
        "common_gain_magnitude": float(np.abs(common_gain)),
        "common_phase_deg": float(np.degrees(np.angle(common_gain))),
    }


def _per_channel(
    reference: dict[str, Any],
    arrays: Any,
    tag: str,
    config_name: str,
) -> list[dict[str, float]]:
    """逐信道复算 NLI/SNR。"""
    dataset = reference["datasets"][tag]
    nl_spectrum, lin_spectrum, freqs = _spectra(reference, arrays, tag, config_name)
    offsets = np.asarray(arrays[dataset["offsets_key"]], dtype=np.float64)
    rows = []
    for channel, offset in enumerate(offsets):
        metrics = _band_metrics(nl_spectrum, lin_spectrum, freqs, float(offset))
        rows.append({"channel": channel, "offset_hz": float(offset), **metrics})
    return rows


def _spectral_widths(
    reference: dict[str, Any],
    arrays: Any,
    tag: str,
    config_name: str,
) -> dict[str, Any]:
    """不同分析带宽下的 NLI 与相对 50 GHz 的累积份额。"""
    dataset = reference["datasets"][tag]
    nl_spectrum, lin_spectrum, freqs = _spectra(reference, arrays, tag, config_name)
    offsets = np.asarray(arrays[dataset["offsets_key"]], dtype=np.float64)
    rows = {}
    for channel, offset in enumerate(offsets):
        full = _band_metrics(nl_spectrum, lin_spectrum, freqs, float(offset))["nli_power_w"]
        per_width = {}
        for width in WIDTHS_HZ:
            metrics = _band_metrics(nl_spectrum, lin_spectrum, freqs, float(offset), bandwidth_hz=width)
            per_width[f"{width / 1e9:g}GHz"] = {
                "nli_power_w": metrics["nli_power_w"],
                "snr_nli_db": metrics["snr_nli_db"],
                "fraction_of_50ghz": metrics["nli_power_w"] / full if full > 0.0 else float("nan"),
            }
        rows[f"ch{channel}"] = per_width
    return rows


def _fine_psd(
    reference: dict[str, Any],
    arrays: Any,
    tag: str,
    config_name: str,
) -> dict[str, Any]:
    """1 GHz 细粒度 NLI 功率谱（含信道外 12.5 GHz 保护区）。"""
    dataset = reference["datasets"][tag]
    nl_spectrum, lin_spectrum, freqs = _spectra(reference, arrays, tag, config_name)
    offsets = np.asarray(arrays[dataset["offsets_key"]], dtype=np.float64)
    num_samples = freqs.size
    rows = {}
    for channel, offset in enumerate(offsets):
        # 投影仍用 50 GHz 标准带，保证与其它表一致；PSD 只看残差。
        mask = np.abs(freqs - offset) <= ANALYSIS_BANDWIDTH_HZ / 2.0
        common_gain = np.sum(nl_spectrum[:, mask] * np.conj(lin_spectrum[:, mask])) / np.sum(
            np.abs(lin_spectrum[:, mask]) ** 2
        )
        residual = nl_spectrum - common_gain * lin_spectrum
        bins = []
        edges = np.arange(-FINE_SPAN_HZ, FINE_SPAN_HZ + FINE_BIN_HZ / 2.0, FINE_BIN_HZ)
        for lower, upper in zip(edges[:-1], edges[1:], strict=True):
            center = offset + 0.5 * (lower + upper)
            bin_mask = np.abs(freqs - center) <= FINE_BIN_HZ / 2.0
            power = float(np.sum(np.abs(residual[:, bin_mask]) ** 2) / num_samples**2)
            bins.append(
                {
                    "offset_low_hz": float(lower),
                    "offset_high_hz": float(upper),
                    "nli_power_w": power,
                    "nli_psd_w_per_hz": power / FINE_BIN_HZ,
                }
            )
        rows[f"ch{channel}"] = bins
    return rows


def _block_variance(
    arrays: Any,
    reference: dict[str, Any],
    tag: str,
    config_name: str,
) -> dict[str, Any]:
    """把记录切成 2/4/8/16 块，逐块重做投影，返回均值/相对标准差。"""
    dataset = reference["datasets"][tag]
    config = dataset["configs"][config_name]
    nonlinear = arrays[config["waveform_nonlinear_key"]].astype(np.complex128)
    linear = arrays[config["waveform_linear_key"]].astype(np.complex128)
    num_samples = int(dataset["num_samples"])
    sample_rate = float(dataset["sample_rate_hz"])
    offsets = np.asarray(arrays[dataset["offsets_key"]], dtype=np.float64)
    rows = {}
    for channel, offset in enumerate(offsets):
        per_count = {}
        for n_blocks in BLOCK_COUNTS:
            block = num_samples // n_blocks
            values = []
            for index in range(n_blocks):
                start = index * block
                stop = start + block
                nl_spectrum = np.fft.fft(nonlinear[:, start:stop], axis=1)
                lin_spectrum = np.fft.fft(linear[:, start:stop], axis=1)
                freqs = np.fft.fftfreq(block, 1.0 / sample_rate)
                metrics = _band_metrics(nl_spectrum, lin_spectrum, freqs, float(offset))
                values.append(metrics["nli_power_w"])
            sample = np.asarray(values, dtype=np.float64)
            symbols_per_block = int(round(block * SYMBOL_RATE_HZ / sample_rate))
            per_count[f"{n_blocks}x{symbols_per_block}"] = {
                "samples_per_block": block,
                "nli_mean_w": float(sample.mean()),
                "relative_std": float(sample.std() / sample.mean()) if sample.mean() > 0.0 else float("nan"),
                "relative_values": [float(value / sample.mean()) for value in sample],
            }
        rows[f"ch{channel}"] = per_count
    return rows


def _band_shift(
    reference: dict[str, Any],
    arrays: Any,
    tag: str,
    config_name: str,
) -> dict[str, Any]:
    """分析带中心平移 ±5/±10 GHz 的 NLI 敏感性。"""
    dataset = reference["datasets"][tag]
    nl_spectrum, lin_spectrum, freqs = _spectra(reference, arrays, tag, config_name)
    offsets = np.asarray(arrays[dataset["offsets_key"]], dtype=np.float64)
    rows = {}
    for channel, offset in enumerate(offsets):
        base = _band_metrics(nl_spectrum, lin_spectrum, freqs, float(offset))["nli_power_w"]
        per_shift = {}
        for shift in SHIFTS_HZ:
            metrics = _band_metrics(nl_spectrum, lin_spectrum, freqs, float(offset), shift_hz=float(shift))
            per_shift[f"{shift / 1e9:+g}GHz"] = {
                "nli_power_w": metrics["nli_power_w"],
                "snr_nli_db": metrics["snr_nli_db"],
                "delta_db": float(10.0 * np.log10(metrics["nli_power_w"] / base)) if base > 0.0 else float("nan"),
            }
        rows[f"ch{channel}"] = per_shift
    return rows


def _input_occupancy(
    reference: dict[str, Any],
    arrays: Any,
    tag: str,
) -> dict[str, Any]:
    """输入波形的信道占用、保护带泄漏与外信道/Nyquist 距离。"""
    dataset = reference["datasets"][tag]
    input_field = arrays[dataset["input_key"]].astype(np.complex128)
    num_samples = int(dataset["num_samples"])
    sample_rate = float(dataset["sample_rate_hz"])
    offsets = np.asarray(arrays[dataset["offsets_key"]], dtype=np.float64)
    spectrum = np.fft.fft(input_field, axis=1)
    freqs = np.fft.fftfreq(num_samples, 1.0 / sample_rate)
    power = np.sum(np.abs(spectrum) ** 2, axis=0) / num_samples**2
    total = float(power.sum())

    in_band = np.zeros(num_samples, dtype=bool)
    for offset in offsets:
        in_band |= np.abs(freqs - offset) <= CHANNEL_ROLLOFF_HALF_BANDWIDTH_HZ
    edges = np.abs(offsets).max() + CHANNEL_ROLLOFF_HALF_BANDWIDTH_HZ
    outside = np.abs(freqs) > edges

    per_channel = {}
    for channel, offset in enumerate(offsets):
        wide = np.abs(freqs - offset) <= CHANNEL_ROLLOFF_HALF_BANDWIDTH_HZ
        narrow = np.abs(freqs - offset) <= ANALYSIS_BANDWIDTH_HZ / 2.0
        per_channel[f"ch{channel}"] = {
            "power_in_55ghz_w": float(power[wide].sum()),
            "power_in_50ghz_w": float(power[narrow].sum()),
            "fraction_50_of_55": float(power[narrow].sum() / power[wide].sum()),
        }

    # 相邻信道之间的保护带：取每个信道外 10 GHz 的 50/50 GHz 中心切片。
    guard_slices = {}
    for channel, offset in enumerate(offsets[:-1]):
        next_offset = float(offsets[channel + 1])
        midpoint = 0.5 * (float(offset) + next_offset)
        slice_mask = np.abs(freqs - midpoint) <= 5e9
        guard_slices[f"ch{channel}-ch{channel + 1}"] = {
            "gap_center_hz": float(midpoint),
            "power_fraction_of_total_db": float(10.0 * np.log10(max(power[slice_mask].sum(), 1e-30) / total)),
        }

    return {
        "total_power_w": total,
        "fraction_in_channel_bands": float(power[in_band].sum() / total),
        "fraction_outside_outermost_band_db": float(10.0 * np.log10(max(power[outside].sum(), 1e-30) / total)),
        "nyquist_hz": sample_rate / 2.0,
        "outermost_channel_edge_hz": float(edges),
        "nyquist_margin_hz": float(sample_rate / 2.0 - edges),
        "guard_slices": guard_slices,
        "per_channel": per_channel,
    }


# ---------------------------------------------------------------------------
# 解析 SPM/XPM 对照
# ---------------------------------------------------------------------------


def _grid(n_channels: int) -> tuple[Any, list[Any]]:
    """构造与 ``dra_waveform_compare.py`` 相同的 100 GHz 栅格。"""
    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid

    grid = FlexibleGrid(slot_width_hz=12.5e9, num_slots=8 * n_channels + 8, band_low_edge_hz=192.95e12)
    connections = []
    for index in range(n_channels):
        first_slot = 2 + 8 * index
        connections.append(
            ConnectionSpec(
                grid=grid,
                connection_id=index,
                first_slot=first_slot,
                n_slots=4,
                bit_rate_bps=100e9,
                modulation="QPSK",
                symbol_rate_hz=SYMBOL_RATE_HZ,
                roll_off=0.0,
                launch_power_w=LAUNCH_POWER_W,
            )
        )
    return grid, connections


def _analytic_components(
    model_name: str,
    n_channels: int,
    nli_form: str | None,
) -> dict[str, Any]:
    """用解析模型返回逐信道 SPM/XPM 与 SNR。"""
    from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    fiber = FiberSpanConfig(
        length_km=FIBER_KWARGS["length_km"],
        attenuation_db_km_ref=FIBER_KWARGS["attenuation_db_km_ref"],
        dispersion_parameter_d=FIBER_KWARGS["dispersion_parameter_d"],
        dispersion_slope_s=FIBER_KWARGS["dispersion_slope_s"],
        effective_area_um2_ref=FIBER_KWARGS["effective_area_um2_ref"],
        nonlinear_index_n2=FIBER_KWARGS["nonlinear_index_n2"],
    )
    grid, connections = _grid(n_channels)
    options: dict[str, Any] = {}
    if model_name == "hybrid_raman_paper" and nli_form is not None:
        options["nli_form"] = nli_form
    model = create_model(
        model_name,
        grid,
        fiber,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        connections=connections,
        slots_per_channel=8,
        channel_bandwidth_source="occupied",
        **options,
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=5.5)])
    result = model.evaluate_path(path, np.ones(n_channels) * LAUNCH_POWER_W)
    spm = np.asarray(result.spm_noise_w, dtype=np.float64)
    xpm = np.asarray(result.xpm_noise_w, dtype=np.float64)
    signal = np.asarray(result.signal_power_w, dtype=np.float64)
    rows = []
    for channel in range(n_channels):
        total = spm[channel] + xpm[channel]
        rows.append(
            {
                "channel": channel,
                "spm_noise_w": float(spm[channel]),
                "xpm_noise_w": float(xpm[channel]),
                "total_nli_w": float(total),
                "snr_nli_db": float(10.0 * np.log10(signal[channel] / total)),
                "signal_power_w": float(signal[channel]),
            }
        )
    return {"per_channel": rows, "model": model_name, "nli_form": nli_form}


def _analytic_block(reference: dict[str, Any]) -> dict[str, Any]:
    """跑 3/5/7ch 无泵的 paper/integral/numerical_dra 逐信道分量。"""
    block: dict[str, Any] = {}
    for tag in TAGS:
        n_channels = int(reference["datasets"][tag]["n_channels"])
        per_tag = {}
        per_tag["hybrid_paper"] = _analytic_components("hybrid_raman_paper", n_channels, "paper")
        per_tag["hybrid_integral"] = _analytic_components("hybrid_raman_paper", n_channels, "integral")
        per_tag["numerical_dra"] = _analytic_components("numerical_dra", n_channels, None)
        block[tag] = per_tag
    return block


def _analytic_growth(block: dict[str, Any]) -> dict[str, Any]:
    """提取 SPM/XPM/总 NLI 随 N 的斜率。"""
    growth = {}
    for model in ("hybrid_paper", "hybrid_integral", "numerical_dra"):
        growth[model] = {}
        for quantity in ("spm_noise_w", "xpm_noise_w", "total_nli_w", "snr_nli_db"):
            values = {}
            for n_channels, tag in ((3, "3ch"), (5, "5ch"), (7, "7ch")):
                rows = block[tag][model]["per_channel"]
                center = rows[n_channels // 2]
                values[tag] = float(center[quantity])
            first = values["3ch"]
            last = values["7ch"]
            growth[model][quantity] = {
                "values": values,
                "log_slope_3_to_7": float(np.log(last / first) / np.log(7.0 / 3.0)),
            }
    return growth


# ---------------------------------------------------------------------------
# 可选：SSFM 步长探针
# ---------------------------------------------------------------------------


def _pump_specs(config_name: str) -> list[Any]:
    """把配置名映射成 physics-nlo-eval 的 PumpSpec。"""
    from physics_nlo_eval.channel.network import PumpSpec

    if config_name == NOPUMP:
        return []
    direction = "forward" if config_name.startswith("fwd") else "backward"
    power = float(config_name.split("_")[1].replace("p", ".").removesuffix("w"))
    return [PumpSpec(1450.0, power, direction, 0.25)]


def _step_probe_config(
    reference: dict[str, Any],
    arrays: Any,
    tag: str,
    config_name: str,
    steps_m: tuple[float, ...],
    input_signal: Any,
) -> dict[str, Any]:
    """对同一输入按不同 SSFM 步长重跑，返回逐信道 SNR 序列。"""
    from physics_nlo_eval.channel import DistributedRamanFiber

    dataset = reference["datasets"][tag]
    offsets = np.asarray(arrays[dataset["offsets_key"]], dtype=np.float64)
    pumps = _pump_specs(config_name)
    series = []
    for step_m in steps_m:
        outputs = {}
        for nonlinear in (True, False):
            fiber = DistributedRamanFiber(
                length_m=100e3,
                alpha_db_per_km=0.2,
                beta2_ps2_per_km=-20.0,
                beta3_ps3_per_km=1e-4,
                pumps=pumps,
                coarse_step_size_m=step_m,
                n2_si=2.6e-20 if nonlinear else 0.0,
                a_eff_um2=80.0,
                add_ase=False,
            )
            outputs[nonlinear] = fiber.forward(input_signal).data
        nonlinear_fft = np.fft.fft(outputs[True], axis=1)
        linear_fft = np.fft.fft(outputs[False], axis=1)
        freqs = np.fft.fftfreq(nonlinear_fft.shape[1], 1.0 / float(dataset["sample_rate_hz"]))
        channels = []
        for offset in offsets:
            metrics = _band_metrics(nonlinear_fft, linear_fft, freqs, float(offset))
            channels.append(
                {
                    "channel": len(channels),
                    "snr_nli_db": metrics["snr_nli_db"],
                    "nli_power_w": metrics["nli_power_w"],
                    "signal_power_w": metrics["linear_power_w"],
                }
            )
        snr = np.asarray([row["snr_nli_db"] for row in channels], dtype=np.float64)
        nli = np.asarray([row["nli_power_w"] for row in channels], dtype=np.float64)
        series.append(
            {
                "step_m": float(step_m),
                "n_steps": int(np.ceil(100e3 / step_m)),
                "center_snr_db": float(snr[len(snr) // 2]),
                "mean_snr_db": float(snr.mean()),
                "min_snr_db": float(snr.min()),
                "center_nli_w": float(nli[len(nli) // 2]),
                "per_channel": channels,
            }
        )
    return {
        "config": config_name,
        "steps": series,
        "last_increment_db": float(series[-1]["center_snr_db"] - series[-2]["center_snr_db"]),
        "artifact_500m_db": float(series[0]["center_snr_db"] - series[-1]["center_snr_db"]),
    }


def _step_probe(reference: dict[str, Any], arrays: Any) -> dict[str, Any]:
    """跑 3/5/7ch 无泵与 7ch 0.2W 泵浦的步长收敛。"""
    try:
        from physics_nlo_eval.core.signal import Signal, SignalContext
    except ImportError as error:  # pragma: no cover - 环境缺依赖时跳过
        return {"available": False, "reason": f"{type(error).__name__}: {error}"}

    probes: dict[str, Any] = {}
    for tag in TAGS:
        dataset = reference["datasets"][tag]
        context = SignalContext(
            sample_rate=float(dataset["sample_rate_hz"]),
            center_freq=193.1e12,
            n_pol=2,
            sps=4,
        )
        signal = Signal(data=arrays[dataset["input_key"]].astype(np.complex128), ctx=context)
        probes[tag] = {NOPUMP: _step_probe_config(reference, arrays, tag, NOPUMP, STEP_PROBE_STEPS_M, signal)}
    # 7ch 泵浦配置的连带影响（仅记录，不改变本 WP 的无泵结论）。
    dataset = reference["datasets"]["7ch"]
    context = SignalContext(
        sample_rate=float(dataset["sample_rate_hz"]),
        center_freq=193.1e12,
        n_pol=2,
        sps=4,
    )
    signal = Signal(data=arrays[dataset["input_key"]].astype(np.complex128), ctx=context)
    for config_name in PUMPED_PROBE_CONFIGS:
        probes["7ch"][config_name] = _step_probe_config(
            reference, arrays, "7ch", config_name, PUMPED_PROBE_STEPS_M, signal
        )
    return {"available": True, "probes": probes}


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


def _summary(
    reference: dict[str, Any],
    per_channel: dict[str, Any],
    analytic: dict[str, Any],
    probe: dict[str, Any],
    widths: dict[str, Any],
    blocks: dict[str, Any],
    shifts: dict[str, Any],
    occupancy: dict[str, Any],
) -> dict[str, Any]:
    """把决定性指标汇总成可被下游引用的数字（不做硬编码判断）。"""
    summary: dict[str, Any] = {"nopump": {}, "analytic_gap": {}, "step_probe": {}}
    for tag in TAGS:
        n_channels = int(reference["datasets"][tag]["n_channels"])
        center = per_channel[tag][n_channels // 2]
        snr = [row["snr_nli_db"] for row in per_channel[tag]]
        summary["nopump"][tag] = {
            "center_snr_db": center["snr_nli_db"],
            "min_snr_db": float(min(snr)),
            "max_snr_db": float(max(snr)),
            "spread_db": float(max(snr) - min(snr)),
            "center_nli_w": center["nli_power_w"],
        }
        hybrid = analytic[tag]["hybrid_paper"]["per_channel"][n_channels // 2]
        # 解析 NLI 在发射平面（signal_power_w = 1 mW），波形 NLI 在接收平面；
        # 两边都按同一线性衰减折算后仍可比值：scale = P_rx / P_launch。
        receiver_scale = center["linear_power_w"] / hybrid["signal_power_w"]
        hybrid_spm = hybrid["spm_noise_w"] * receiver_scale
        hybrid_xpm = hybrid["xpm_noise_w"] * receiver_scale
        hybrid_total = hybrid_spm + hybrid_xpm
        implied_wave_xpm = max(center["nli_power_w"] - hybrid_spm, 0.0)
        summary["analytic_gap"][tag] = {
            "receiver_scale": receiver_scale,
            "hybrid_spm_w": hybrid_spm,
            "hybrid_xpm_w": hybrid_xpm,
            "hybrid_total_w": hybrid_total,
            "wave_nli_w": center["nli_power_w"],
            "hybrid_minus_wave_db": hybrid["snr_nli_db"] - center["snr_nli_db"],
            "wave_over_hybrid_total": center["nli_power_w"] / hybrid_total,
            "implied_wave_xpm_w": implied_wave_xpm,
            "implied_wave_xpm_over_gn_xpm": implied_wave_xpm / hybrid_xpm,
        }
        summary["nopump"][tag]["block_rel_std_32sym"] = blocks[tag][f"ch{n_channels // 2}"]["8x32"]["relative_std"]
        summary["nopump"][tag]["max_band_shift_db"] = float(
            max(abs(row["delta_db"]) for row in shifts[tag][f"ch{n_channels // 2}"].values())
        )
        summary["nopump"][tag]["fraction_50_of_55"] = occupancy[tag]["per_channel"][f"ch{n_channels // 2}"][
            "fraction_50_of_55"
        ]
        summary["nopump"][tag]["width_10ghz_fraction"] = widths[tag][f"ch{n_channels // 2}"]["10GHz"][
            "fraction_of_50ghz"
        ]
    if probe.get("available"):
        converged_nli = []
        for tag in TAGS:
            series = probe["probes"][tag][NOPUMP]["steps"]
            hybrid_snr = analytic[tag]["hybrid_paper"]["per_channel"][
                int(reference["datasets"][tag]["n_channels"]) // 2
            ]["snr_nli_db"]
            converged_nli.append(series[-1]["center_nli_w"])
            summary["step_probe"][tag] = {
                "center_snr_500m_db": series[0]["center_snr_db"],
                "center_snr_converged_db": series[-1]["center_snr_db"],
                "converged_step_m": series[-1]["step_m"],
                "artifact_500m_db": series[0]["center_snr_db"] - series[-1]["center_snr_db"],
                "last_increment_db": probe["probes"][tag][NOPUMP]["last_increment_db"],
                "converged_nli_w": series[-1]["center_nli_w"],
                "hybrid_minus_wave_converged_db": hybrid_snr - series[-1]["center_snr_db"],
            }
        summary["step_probe"]["converged_nli_log_slope_3_to_7"] = float(
            np.log(converged_nli[2] / converged_nli[0]) / np.log(7.0 / 3.0)
        )
    return summary


def _step_criterion(reference: dict[str, Any], arrays: Any) -> dict[str, Any]:
    """给出“每步信道走离 ≤ 0.15 symbol”的 SSFM 步长设计规则。

    走离 ``d tau = 2 pi * df * |beta2| * dz``；用最外信道到中心信道的距离
    作为 ``df``（这是 500 m 误差随信道数增长的经验尺度）。该规则偏保守，
    只作为参考数据生成器的建议，不改变任何模型默认值。
    """
    beta2 = 20e-24 / 1e3  # -20 ps^2/km -> s^2/m
    symbol_period = 1.0 / SYMBOL_RATE_HZ
    fraction = 0.15
    rows = {}
    for tag in TAGS:
        offsets = np.asarray(arrays[reference["datasets"][tag]["offsets_key"]], dtype=np.float64)
        separation = float(np.max(np.abs(offsets - np.mean(offsets)))) if offsets.size else 0.0
        walkoff_500m = float(2.0 * np.pi * separation * beta2 * 500.0 / symbol_period)
        suggested = float(fraction * symbol_period / (2.0 * np.pi * separation * beta2)) if separation > 0 else 0.0
        rows[tag] = {
            "max_offset_from_center_hz": separation,
            "walkoff_per_500m_symbols": walkoff_500m,
            "suggested_step_m_for_0p15_symbol": suggested,
        }
    return rows


def main() -> None:
    """运行审计并写出 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-probe",
        action="store_true",
        help="额外用 physics-nlo-eval 重跑 SSFM 步长收敛（需要该包可导入）",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()

    reference = json.loads(REFERENCE_JSON.read_text(encoding="utf-8"))
    arrays = np.load(REFERENCE_NPZ)

    per_channel: dict[str, Any] = {}
    widths: dict[str, Any] = {}
    fine_psd: dict[str, Any] = {}
    blocks: dict[str, Any] = {}
    shifts: dict[str, Any] = {}
    occupancy: dict[str, Any] = {}
    for tag in TAGS:
        per_channel[tag] = _per_channel(reference, arrays, tag, NOPUMP)
        widths[tag] = _spectral_widths(reference, arrays, tag, NOPUMP)
        fine_psd[tag] = _fine_psd(reference, arrays, tag, NOPUMP)
        blocks[tag] = _block_variance(arrays, reference, tag, NOPUMP)
        shifts[tag] = _band_shift(reference, arrays, tag, NOPUMP)
        occupancy[tag] = _input_occupancy(reference, arrays, tag)

    analytic = _analytic_block(reference)
    analytic_growth = _analytic_growth(analytic)
    probe = _step_probe(reference, arrays) if args.step_probe else {"available": False}

    growth = {}
    for tag in TAGS:
        n_channels = int(reference["datasets"][tag]["n_channels"])
        center = per_channel[tag][n_channels // 2]
        growth[tag] = {
            "center_snr_db": center["snr_nli_db"],
            "center_nli_w": center["nli_power_w"],
        }
    wave_slope = float((np.log(growth["7ch"]["center_nli_w"] / growth["3ch"]["center_nli_w"])) / np.log(7.0 / 3.0))

    output = {
        "generated_by": "scripts/seven_ch_audit.py",
        "reference": {
            "json": str(REFERENCE_JSON),
            "npz": str(REFERENCE_NPZ),
            "fiber_length_km": 100.0,
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "npz_step_m": 500.0,
        },
        "definitions": {
            "nli": (
                "50 GHz ideal rectangular band, one complex scalar projection NL - a*LIN, "
                "band power / N^2, both polarizations"
            ),
            "widths_hz": list(WIDTHS_HZ),
            "shifts_hz": list(SHIFTS_HZ),
            "block_counts": list(BLOCK_COUNTS),
            "fine_bin_hz": FINE_BIN_HZ,
        },
        "per_channel": per_channel,
        "spectral_width": widths,
        "fine_psd": fine_psd,
        "block_variance": blocks,
        "band_shift": shifts,
        "input_occupancy": occupancy,
        "waveform_growth": {
            "center_nli_w": {tag: growth[tag]["center_nli_w"] for tag in TAGS},
            "center_snr_db": {tag: growth[tag]["center_snr_db"] for tag in TAGS},
            "nli_log_slope_3_to_7": wave_slope,
        },
        "analytic": analytic,
        "analytic_growth": analytic_growth,
        "step_criterion": _step_criterion(reference, arrays),
        "step_probe": probe,
        "summary": _summary(reference, per_channel, analytic, probe, widths, blocks, shifts, occupancy),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
