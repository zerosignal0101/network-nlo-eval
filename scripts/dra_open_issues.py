"""量化 DRA 校准的三个开放问题：发射功率依赖、1ch 前向、单次实现统计预算。

三个部分各自写一个 JSON（默认在 ``results/raman/``）：

1. **launch sweep**：读 62.5 m 收敛耗尽参考的 ``3ch_0p5mw/1p0mw/2p0mw``，
   对 hybrid/H3/numerical 跑 ``pump_depletion=True``，给出残差、NLI 的
   log-log 指数，以及"把解析泵浦耗尽驱动乘以 2（模拟双偏振总功率）"的
   负结果。负结果用 ``HybridRamanPaperSpanKernel`` 的 ODE 求解器只改信号
   驱动、NLI 仍按标称功率计算，不动模型源码；
2. **1ch forward**：用 ``physics_nlo_eval.DistributedRamanFiber``
   （``pump_model="depleted"``）生成 1ch、100 km、50 GBd、100 GHz 栅格、
   1450 nm 0.2 W 前向的波形，扫 0.25/0.5/1/2 mW/pol，并跑无泵对照；
   解析侧用与 1ch 参考相同的 50 GHz NLI 口径；
3. **seed ensemble**：3ch（500 m）与 5/7ch（62.5 m）各跑多 seed ×
   {无泵, 前向 0.2 W, 后向 0.2 W}，输出每 seed 与 mean/std。

用法::

    PYTHONPATH=src:scripts /home/zsig/miniconda3/envs/nlo/bin/python \
        scripts/dra_open_issues.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# 与波形对照脚本共用同一栅格/光纤/泵浦解析，保证数字可对照。
from dra_waveform_compare import FIBER, SYMBOL_RATE_HZ, _analytic, _grid, _pumps

DEFAULT_REFERENCE = Path("data/dra_waveform_converged/dra_reference_depleted.json")
DEFAULT_OUTPUT_DIR = Path("results/raman")

#: 3ch 发射功率扫描（同一耗尽参考）。
LAUNCH_DATASETS: tuple[str, ...] = ("3ch_0p5mw", "3ch_1p0mw", "3ch_2p0mw")
LAUNCH_CONFIG = "fwd_0p20w"
LAUNCH_MODELS: tuple[str, ...] = ("hybrid_raman_paper", "pcfm_h3_raman", "numerical_dra")

#: 2× 耗尽驱动负结果使用 1 mW/pol 的 3/5ch 前向 0.2 W。
PUMP_DRIVE_DATASETS: tuple[str, ...] = ("3ch", "5ch")
PUMP_DRIVE_FACTORS: tuple[float, ...] = (1.0, 2.0)

#: 1ch 前向探针的发射功率 (W, 每偏振)。
ONE_CH_LAUNCH_POWERS_W: tuple[float, ...] = (0.25e-3, 0.5e-3, 1.0e-3, 2.0e-3)
ONE_CH_CONFIGS: tuple[str, ...] = ("nopump", "fwd_0p20w")
ONE_CH_NUM_SYMBOLS = 256
ONE_CH_SEED = 20261104
ONE_CH_SAMPLE_RATE_HZ = 100e9
ONE_CH_STEP_M = 62.5
ONE_CH_NLI_BANDWIDTH_HZ = 50e9

#: seed 集成：3ch 用 500 m（已收敛），5/7ch 用 62.5 m。
ENSEMBLE_SPECS: tuple[dict[str, Any], ...] = (
    {"tag": "3ch", "count": 3, "offsets_hz": (-100e9, 0.0, 100e9), "sample_rate_hz": 500e9, "step_m": 500.0},
    {
        "tag": "5ch",
        "count": 5,
        "offsets_hz": (-200e9, -100e9, 0.0, 100e9, 200e9),
        "sample_rate_hz": 700e9,
        "step_m": 62.5,
    },
    {
        "tag": "7ch",
        "count": 7,
        "offsets_hz": (-300e9, -200e9, -100e9, 0.0, 100e9, 200e9, 300e9),
        "sample_rate_hz": 900e9,
        "step_m": 62.5,
    },
)
ENSEMBLE_CONFIGS: tuple[str, ...] = ("nopump", "fwd_0p20w", "bwd_0p20w")
ENSEMBLE_SEED_BASE = 20261104
ENSEMBLE_NUM_SYMBOLS = 256
ENSEMBLE_BANDWIDTH_HZ = 50e9

#: physical-nlo-eval 参考的 1ch 光纤（与生成脚本一致）。
ONE_CH_FIBER = {
    "length_m": 100e3,
    "alpha_db_per_km": 0.2,
    "beta2_ps2_per_km": -20.0,
    "beta3_ps3_per_km": 1e-4,
    "n2_si": 2.6e-20,
    "a_eff_um2": 80.0,
}


def fit_loglog_exponent(powers_w: Any, nli_power_w: Any) -> float | None:
    """拟合 ``NLI ∝ P^p`` 的指数 ``p``（log-log 最小二乘斜率）。

    Args:
        powers_w: 发射功率序列 (W)。
        nli_power_w: 对应 NLI 功率序列 (W)。

    Returns
    -------
        指数；有效点不足 2 个或功率非正时返回 ``None``。
    """
    power = np.asarray(powers_w, dtype=np.float64)
    nli = np.asarray(nli_power_w, dtype=np.float64)
    valid = np.isfinite(power) & np.isfinite(nli) & (power > 0.0) & (nli > 0.0)
    if int(np.count_nonzero(valid)) < 2:
        return None
    slope = np.polyfit(np.log(power[valid]), np.log(nli[valid]), 1)[0]
    return float(slope) if np.isfinite(slope) else None


def summarize(values: Any) -> dict[str, float | int | None]:
    """返回 mean/std/sem/min/max；``std`` 用样本标准差（ddof=1）。

    Args:
        values: 一维数值序列。

    Returns
    -------
        统计量字典；全为非有限值时各统计量为 ``None``。
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0, "mean": None, "std": None, "sem": None, "min": None, "max": None}
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "sem": float(std / np.sqrt(array.size)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _json_float(value: Any) -> float | None:
    """把 NaN/Inf 转成 None，保证输出是合法 JSON。"""
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _paper_span_center(
    n_channels: int,
    pumps: list[Any],
    *,
    drive_factor: float,
    launch_power_w: float,
) -> tuple[float, float]:
    """用论文 Eq. (6)-(9) 链路算中心信道，可单独放大 ODE 的信号驱动。

    这是"2× 偏振耗尽驱动"的只读探针：``drive_factor`` 只乘进泵浦-信号
    耦合 ODE 的信号功率，NLI 仍按标称入纤功率计算。它复刻
    ``hybrid_raman_paper(nli_form="paper")`` 的单跨链路，不改模型源码。

    Args:
        n_channels: 信道数。
        pumps: 泵浦列表。
        drive_factor: 耗尽驱动倍数（1 = 正常，2 = 双偏振总功率假设）。
        launch_power_w: 每信道入纤功率 (W, 每偏振)。

    Returns
    -------
        ``(中心信道 SNR_NLI, 中心信道 NLI 功率)``。
    """
    from network_nlo_eval.models.hybrid_raman import HybridRamanPaperSpanKernel
    from network_nlo_eval.models.raman import RamanGainModel

    grid, connections = _grid(n_channels, launch_power_w)
    frequencies = np.asarray([connection.centre_hz for connection in connections], dtype=np.float64)
    bandwidths = np.full(
        n_channels,
        float(connections[0].n_slots) * float(grid.slot_width_hz),
        dtype=np.float64,
    )
    kernel = HybridRamanPaperSpanKernel(
        frequencies,
        bandwidths,
        FIBER,
        pumps=pumps,
        gain_model=RamanGainModel(),
        nli_form="paper",
        pump_depletion=True,
    )
    span_length_m = float(FIBER.length_km) * 1e3
    power = np.full(n_channels, launch_power_w, dtype=np.float64)
    solver = kernel._solver_for(span_length_m)  # 只读探针：复用模型自己的 ODE 求解器
    solved = solver.solve(drive_factor * power, np.zeros(n_channels, dtype=np.float64))
    profile = kernel._fit_profile(solved.profile_norm, span_length_m)
    spm, xpm = kernel._nli_span(power, profile, None)
    nli = np.asarray(spm) + np.asarray(xpm)
    center = n_channels // 2
    snr = float(10.0 * np.log10(power[center] / nli[center]))
    return snr, float(nli[center])


def launch_power_analysis(reference: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    """发射功率扫描 + NLI 指数 + 2× 耗尽驱动负结果。"""
    rows = []
    curves: dict[str, dict[str, list[float]]] = {
        name: {"powers_w": [], "nli_w": [], "snr_db": []}
        for name in ("waveform", *LAUNCH_MODELS)
    }
    for tag in LAUNCH_DATASETS:
        dataset = reference["datasets"][tag]
        n_channels = int(dataset["n_channels"])
        center = int(dataset["center_channel_index"])
        power_w = float(1e-3 * 10.0 ** (float(dataset["launch_power_dbm_per_pol"]) / 10.0))
        channel = dataset["configs"][LAUNCH_CONFIG]["per_channel"][center]
        wave_snr = float(channel["snr_nli_db"])
        wave_nli = float(channel["nli_power_w"])
        curves["waveform"]["powers_w"].append(power_w)
        curves["waveform"]["nli_w"].append(wave_nli)
        curves["waveform"]["snr_db"].append(wave_snr)
        for model_name in LAUNCH_MODELS:
            started = time.time()
            snr, nli = _analytic(
                model_name,
                n_channels,
                LAUNCH_CONFIG,
                pump_depletion=True,
                launch_power_w=power_w,
            )
            curves[model_name]["powers_w"].append(power_w)
            curves[model_name]["nli_w"].append(nli)
            curves[model_name]["snr_db"].append(snr)
            rows.append(
                {
                    "dataset": tag,
                    "n_channels": n_channels,
                    "config": LAUNCH_CONFIG,
                    "model": model_name,
                    "launch_power_w_per_pol": _json_float(power_w),
                    "waveform_snr_nli_db": _json_float(wave_snr),
                    "analytic_snr_nli_db": _json_float(snr),
                    "snr_delta_db": _json_float(snr - wave_snr),
                    "waveform_nli_w": _json_float(wave_nli),
                    "analytic_nli_w": _json_float(nli),
                    "nli_ratio_db": _json_float(10.0 * np.log10(nli / wave_nli)),
                    "elapsed_s": float(time.time() - started),
                }
            )

    exponents = {name: fit_loglog_exponent(curve["powers_w"], curve["nli_w"]) for name, curve in curves.items()}

    drive_rows = []
    for tag in PUMP_DRIVE_DATASETS:
        dataset = reference["datasets"][tag]
        n_channels = int(dataset["n_channels"])
        center = int(dataset["center_channel_index"])
        power_w = 1e-3
        wave_snr = float(dataset["configs"][LAUNCH_CONFIG]["per_channel"][center]["snr_nli_db"])
        pumps = _pumps(LAUNCH_CONFIG)
        base_snr, base_nli = _paper_span_center(
            n_channels, pumps, drive_factor=1.0, launch_power_w=power_w
        )
        for factor in PUMP_DRIVE_FACTORS:
            snr, nli = _paper_span_center(n_channels, pumps, drive_factor=factor, launch_power_w=power_w)
            drive_rows.append(
                {
                    "dataset": tag,
                    "n_channels": n_channels,
                    "config": LAUNCH_CONFIG,
                    "model": "hybrid_raman_paper",
                    "launch_power_w_per_pol": _json_float(power_w),
                    "pump_drive_factor": float(factor),
                    "waveform_snr_nli_db": _json_float(wave_snr),
                    "analytic_snr_nli_db": _json_float(snr),
                    "snr_delta_db": _json_float(snr - wave_snr),
                    "analytic_nli_w": _json_float(nli),
                    "delta_vs_nominal_drive_db": _json_float(snr - base_snr),
                    "nli_vs_nominal_drive_db": _json_float(10.0 * np.log10(nli / base_nli)),
                }
            )

    return {
        "generated_by": "scripts/dra_open_issues.py (launch sweep)",
        "reference_path": str(reference_path),
        "config": {
            "datasets": list(LAUNCH_DATASETS),
            "pump": "fwd 0.2 W, 1450 nm, 0.25 dB/km",
            "pump_depletion": True,
            "models": list(LAUNCH_MODELS),
            "reference": "62.5 m converged depleted waveform, center channel, 100 GHz grid, 50 GBd",
        },
        "rows": rows,
        "exponents": {
            "definition": "log-log slope of center-channel NLI vs launch power (per pol)",
            **exponents,
            "waveform_minus_analytic": (
                None
                if exponents["waveform"] is None or exponents["hybrid_raman_paper"] is None
                else float(exponents["waveform"] - exponents["hybrid_raman_paper"])
            ),
        },
        "pump_drive_2x": {
            "note": (
                "Negative result: doubling only the pump-depletion drive (the dual-polarization "
                "total-power hypothesis) makes hybrid farther from the waveform, not closer. "
                "The residual is therefore not a 2x polarization-convention bug."
            ),
            "rows": drive_rows,
        },
    }


def _build_wdm_signal(
    count: int,
    offsets_hz: tuple[float, ...],
    sample_rate_hz: float,
    launch_power_dbm_per_pol: float,
    *,
    num_symbols: int,
    seed: int,
) -> Any:
    """按 physical-nlo-eval 参考的 TX 口径生成 WDM 波形（DP-QPSK, 4 sps）。"""
    from physics_nlo_eval.channel import ChannelSpec, WDMMux, WDMPlan
    from physics_nlo_eval.tx.mapper import Mapper
    from physics_nlo_eval.tx.optics import IQModulator
    from physics_nlo_eval.tx.rrc import RRCFilter

    rng = np.random.default_rng(seed)
    mapper = Mapper(modulation="DP-QPSK", power_norm=True, sym_rate=SYMBOL_RATE_HZ)
    rrc = RRCFilter(sps=4, roll_off=0.1)
    modulator = IQModulator(launch_power_dbm_per_pol=launch_power_dbm_per_pol)
    bits_per_channel = num_symbols * mapper.bits_per_symbol * mapper.n_pol
    bits = rng.integers(0, 2, size=(count, bits_per_channel), dtype=np.int8).astype(bool)
    channel_signals = [
        modulator.forward(rrc.forward(mapper.forward(bits[index]))) for index in range(count)
    ]
    plan = WDMPlan(
        reference_freq_hz=193.1e12,
        channels=[
            ChannelSpec(
                channel_id=index,
                freq_offset_hz=float(offsets_hz[index]),
                power_dbm_per_pol=launch_power_dbm_per_pol,
            )
            for index in range(count)
        ],
    )
    return WDMMux(plan, target_sample_rate_hz=sample_rate_hz).combine(channel_signals)


def _propagate(signal: Any, pumps: list[Any], step_m: float, *, nonlinear: bool) -> np.ndarray:
    """用 ``DistributedRamanFiber(pump_model="depleted")`` 传播一次。"""
    from physics_nlo_eval.channel import DistributedRamanFiber

    fiber = DistributedRamanFiber(
        length_m=ONE_CH_FIBER["length_m"],
        alpha_db_per_km=ONE_CH_FIBER["alpha_db_per_km"],
        beta2_ps2_per_km=ONE_CH_FIBER["beta2_ps2_per_km"],
        beta3_ps3_per_km=ONE_CH_FIBER["beta3_ps3_per_km"],
        pumps=pumps,
        coarse_step_size_m=step_m,
        n2_si=ONE_CH_FIBER["n2_si"] if nonlinear else 0.0,
        a_eff_um2=ONE_CH_FIBER["a_eff_um2"],
        pump_model="depleted",
        add_ase=False,
    )
    return fiber.forward(signal).data


def band_nli_metrics(
    nonlinear_data: Any,
    linear_data: Any,
    sample_rate_hz: float,
    offset_hz: float,
    bandwidth_hz: float,
) -> tuple[float, float, float]:
    """单复标量投影后的 ``(signal_power, nli_power, snr_db)``。

    与 ``physics_nlo_eval`` 参考生成脚本 ``band_nli_metrics`` 同口径：
    在 ``offset ± bandwidth/2`` 带内拟合 ``NL ≈ a·LIN``，取残差功率为 NLI。
    """
    n_samples = nonlinear_data.shape[-1]
    spectrum_nl = np.fft.fft(nonlinear_data, axis=-1)
    spectrum_lin = np.fft.fft(linear_data, axis=-1)
    frequencies = np.fft.fftfreq(n_samples, d=1.0 / sample_rate_hz)
    mask = np.abs(frequencies - offset_hz) <= bandwidth_hz / 2.0
    nl_band = spectrum_nl[:, mask]
    lin_band = spectrum_lin[:, mask]
    signal_power = float(np.sum(np.abs(nl_band) ** 2) / n_samples**2)
    denominator = np.sum(np.abs(lin_band) ** 2)
    if denominator == 0.0:
        return signal_power, float("nan"), float("nan")
    common_gain = np.sum(np.conj(lin_band) * nl_band) / denominator
    residual = nl_band - common_gain * lin_band
    nli_power = float(np.sum(np.abs(residual) ** 2) / n_samples**2)
    snr_db = float(10.0 * np.log10(signal_power / nli_power)) if nli_power > 0.0 else float("inf")
    return signal_power, nli_power, snr_db


def one_ch_forward_analysis(reference: dict[str, Any], reference_path: Path) -> dict[str, Any]:
    """1ch 前向泵的发射功率扫描。"""
    from physics_nlo_eval.channel import PumpSpec

    reference_dataset = reference["datasets"].get("1ch")
    rows = []
    for config_name in ONE_CH_CONFIGS:
        pumps = [PumpSpec(1450.0, 0.2, "forward", 0.25)] if config_name.startswith("fwd") else []
        for power_w in ONE_CH_LAUNCH_POWERS_W:
            launch_dbm = float(10.0 * np.log10(power_w / 1e-3))
            signal = _build_wdm_signal(
                1,
                (0.0,),
                ONE_CH_SAMPLE_RATE_HZ,
                launch_dbm,
                num_symbols=ONE_CH_NUM_SYMBOLS,
                seed=ONE_CH_SEED,
            )
            started = time.time()
            nonlinear = _propagate(signal, pumps, ONE_CH_STEP_M, nonlinear=True)
            linear = _propagate(signal, pumps, ONE_CH_STEP_M, nonlinear=False)
            signal_power, nli_power, snr = band_nli_metrics(
                nonlinear, linear, ONE_CH_SAMPLE_RATE_HZ, 0.0, ONE_CH_NLI_BANDWIDTH_HZ
            )
            hybrid, hybrid_nli = _analytic(
                "hybrid_raman_paper",
                1,
                config_name,
                pump_depletion=True,
                launch_power_w=power_w,
            )
            h3, h3_nli = _analytic(
                "pcfm_h3_raman",
                1,
                config_name,
                pump_depletion=True,
                launch_power_w=power_w,
            )
            rows.append(
                {
                    "dataset": "1ch",
                    "config": config_name,
                    "n_channels": 1,
                    "launch_power_w_per_pol": _json_float(power_w),
                    "launch_power_dbm_per_pol": _json_float(launch_dbm),
                    "waveform_snr_nli_db": _json_float(snr),
                    "waveform_nli_w": _json_float(nli_power),
                    "waveform_signal_power_w": _json_float(signal_power),
                    "hybrid_snr_nli_db": _json_float(hybrid),
                    "h3_snr_nli_db": _json_float(h3),
                    "hybrid_nli_w": _json_float(hybrid_nli),
                    "h3_nli_w": _json_float(h3_nli),
                    "hybrid_delta_db": _json_float(hybrid - snr),
                    "h3_delta_db": _json_float(h3 - snr),
                    "elapsed_s": float(time.time() - started),
                }
            )

    reproduction = {}
    if reference_dataset is not None:
        for config_name in ONE_CH_CONFIGS:
            if config_name not in reference_dataset.get("configs", {}):
                continue
            expected = float(reference_dataset["configs"][config_name]["per_channel"][0]["snr_nli_db"])
            measured = next(
                row["waveform_snr_nli_db"]
                for row in rows
                if row["config"] == config_name and abs(row["launch_power_w_per_pol"] - 1e-3) < 1e-12
            )
            reproduction[config_name] = {
                "reference_snr_nli_db": expected,
                "measured_snr_nli_db": measured,
                "delta_db": measured - expected,
            }

    exponents = {}
    for config_name in ONE_CH_CONFIGS:
        subset = [row for row in rows if row["config"] == config_name]
        exponents[config_name] = {
            "waveform": fit_loglog_exponent(
                [row["launch_power_w_per_pol"] for row in subset],
                [row["waveform_nli_w"] for row in subset],
            ),
            "hybrid_raman_paper": fit_loglog_exponent(
                [row["launch_power_w_per_pol"] for row in subset],
                [row["hybrid_nli_w"] for row in subset],
            ),
            "pcfm_h3_raman": fit_loglog_exponent(
                [row["launch_power_w_per_pol"] for row in subset],
                [row["h3_nli_w"] for row in subset],
            ),
        }

    return {
        "generated_by": "scripts/dra_open_issues.py (1ch forward)",
        "reference_path": str(reference_path),
        "config": {
            "fiber": ONE_CH_FIBER,
            "length_km": 100.0,
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "grid": "1 channel, 100 GHz grid, 100 GHz sample rate",
            "step_m": ONE_CH_STEP_M,
            "pump": "fwd 0.2 W, 1450 nm, 0.25 dB/km, pump_model=depleted",
            "nli_bandwidth_hz": ONE_CH_NLI_BANDWIDTH_HZ,
            "num_symbols": ONE_CH_NUM_SYMBOLS,
            "seed": ONE_CH_SEED,
            "launch_powers_w_per_pol": list(ONE_CH_LAUNCH_POWERS_W),
        },
        "reproduction_check": reproduction,
        "rows": rows,
        "exponents": exponents,
    }


def seed_ensemble_analysis(
    reference: dict[str, Any], reference_path: Path, seeds: int, seeds_7ch: int
) -> dict[str, Any]:
    """3ch/5ch/7ch 的多 seed 集成，输出每 seed 与 mean/std。"""
    from physics_nlo_eval.channel import PumpSpec

    rows = []
    for spec in ENSEMBLE_SPECS:
        count = int(spec["count"])
        n_seeds = seeds_7ch if spec["tag"] == "7ch" else seeds
        pumps_by_config = {
            "nopump": [],
            "fwd_0p20w": [PumpSpec(1450.0, 0.2, "forward", 0.25)],
            "bwd_0p20w": [PumpSpec(1450.0, 0.2, "backward", 0.25)],
        }
        center = count // 2
        for index in range(n_seeds):
            seed = ENSEMBLE_SEED_BASE + index
            signal = _build_wdm_signal(
                count,
                tuple(float(offset) for offset in spec["offsets_hz"]),
                float(spec["sample_rate_hz"]),
                0.0,
                num_symbols=ENSEMBLE_NUM_SYMBOLS,
                seed=seed,
            )
            for config_name in ENSEMBLE_CONFIGS:
                started = time.time()
                pumps = pumps_by_config[config_name]
                nonlinear = _propagate(signal, pumps, float(spec["step_m"]), nonlinear=True)
                linear = _propagate(signal, pumps, float(spec["step_m"]), nonlinear=False)
                signal_power, nli_power, snr = band_nli_metrics(
                    nonlinear, linear, float(spec["sample_rate_hz"]), 0.0, ENSEMBLE_BANDWIDTH_HZ
                )
                rows.append(
                    {
                        "dataset": spec["tag"],
                        "n_channels": count,
                        "config": config_name,
                        "seed": seed,
                        "step_m": float(spec["step_m"]),
                        "waveform_snr_nli_db": _json_float(snr),
                        "waveform_nli_w": _json_float(nli_power),
                        "waveform_signal_power_w": _json_float(signal_power),
                        "center_channel_index": center,
                        "elapsed_s": float(time.time() - started),
                    }
                )

    summary = []
    for spec in ENSEMBLE_SPECS:
        reference_dataset = reference["datasets"].get(spec["tag"])
        for config_name in ENSEMBLE_CONFIGS:
            subset = [
                row
                for row in rows
                if row["dataset"] == spec["tag"] and row["config"] == config_name
            ]
            entry: dict[str, Any] = {
                "dataset": spec["tag"],
                "n_channels": spec["count"],
                "config": config_name,
                "step_m": spec["step_m"],
                "snr_nli_db": summarize([row["waveform_snr_nli_db"] for row in subset]),
                "nli_w": summarize([row["waveform_nli_w"] for row in subset]),
                "seeds": [row["seed"] for row in subset],
            }
            if reference_dataset is not None and config_name in reference_dataset["configs"]:
                center = int(reference_dataset["center_channel_index"])
                reference_snr = float(
                    reference_dataset["configs"][config_name]["per_channel"][center]["snr_nli_db"]
                )
                entry["reference_snr_nli_db"] = reference_snr
                if entry["snr_nli_db"]["mean"] is not None:
                    entry["reference_minus_ensemble_mean_db"] = reference_snr - entry["snr_nli_db"]["mean"]
            summary.append(entry)

    return {
        "generated_by": "scripts/dra_open_issues.py (seed ensemble)",
        "reference_path": str(reference_path),
        "note": (
            "3ch runs at 500 m (its 62.5 m step artifact is only 0.07 dB); 5/7ch run at 62.5 m. "
            "reference_snr_nli_db is the single 62.5 m realization from the converged reference, "
            "so for 3ch reference_minus_ensemble_mean_db also contains the 500 m step artifact. "
            "std uses the sample standard deviation (ddof=1)."
        ),
        "config": {
            "datasets": [
                {
                    "tag": spec["tag"],
                    "n_channels": spec["count"],
                    "offsets_hz": list(spec["offsets_hz"]),
                    "sample_rate_hz": spec["sample_rate_hz"],
                    "step_m": spec["step_m"],
                }
                for spec in ENSEMBLE_SPECS
            ],
            "configs": {
                "nopump": [],
                "fwd_0p20w": "0.2 W forward, 1450 nm, 0.25 dB/km",
                "bwd_0p20w": "0.2 W backward, 1450 nm, 0.25 dB/km",
            },
            "pump_model": "depleted",
            "launch_power_w_per_pol": 1e-3,
            "num_symbols": ENSEMBLE_NUM_SYMBOLS,
            "nli_bandwidth_hz": ENSEMBLE_BANDWIDTH_HZ,
            "seeds": [ENSEMBLE_SEED_BASE + index for index in range(seeds)],
            "seeds_7ch": [ENSEMBLE_SEED_BASE + index for index in range(seeds_7ch)],
        },
        "rows": rows,
        "summary": summary,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    """写 JSON 并打印路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    """跑三个开放问题分析并写 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="62.5 m 收敛耗尽参考 JSON")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--seeds", type=int, default=4, help="3ch/5ch 的 seed 数（默认 4）")
    parser.add_argument("--seeds-7ch", type=int, default=4, help="7ch 的 seed 数（默认 4；太慢可降到 2）")
    parser.add_argument("--skip-launch", action="store_true", help="跳过 launch sweep")
    parser.add_argument("--skip-one-ch", action="store_true", help="跳过 1ch 前向")
    parser.add_argument("--skip-ensemble", action="store_true", help="跳过 seed 集成")
    args = parser.parse_args()

    if not args.reference.exists():
        raise SystemExit(f"reference JSON not found: {args.reference}")
    reference = json.loads(args.reference.read_text(encoding="utf-8"))

    if not args.skip_launch:
        payload = launch_power_analysis(reference, args.reference)
        _write(args.output_dir / "launch_power_analysis.json", payload)
        exponents = payload["exponents"]

        def _fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:.3f}"

        print(
            "NLI exponent: waveform "
            f"{_fmt(exponents['waveform'])}, hybrid {_fmt(exponents['hybrid_raman_paper'])}, "
            f"H3 {_fmt(exponents['pcfm_h3_raman'])}, numerical {_fmt(exponents['numerical_dra'])}"
        )
        for row in payload["pump_drive_2x"]["rows"]:
            print(
                f"2x drive {row['dataset']} factor={row['pump_drive_factor']:g}: "
                f"delta={row['snr_delta_db']:+.3f} dB"
            )

    if not args.skip_one_ch:
        payload = one_ch_forward_analysis(reference, args.reference)
        _write(args.output_dir / "one_ch_forward.json", payload)
        for row in payload["rows"]:
            print(
                f"1ch {row['config']} {row['launch_power_w_per_pol'] * 1e3:g} mW: "
                f"wave {row['waveform_snr_nli_db']:.3f}, hybrid {row['hybrid_delta_db']:+.3f}, "
                f"H3 {row['h3_delta_db']:+.3f}"
            )

    if not args.skip_ensemble:
        payload = seed_ensemble_analysis(reference, args.reference, args.seeds, args.seeds_7ch)
        _write(args.output_dir / "seed_ensemble.json", payload)
        for entry in payload["summary"]:
            stats = entry["snr_nli_db"]
            if stats["mean"] is None:
                continue
            print(
                f"{entry['dataset']} {entry['config']}: mean {stats['mean']:.3f} "
                f"std {stats['std']:.3f} (n={stats['n']})"
            )


if __name__ == "__main__":
    main()
