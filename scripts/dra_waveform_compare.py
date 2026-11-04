"""把 physical-nlo-eval 的 DRA 波形参考接入 network-nlo-eval。

读取 `data/dra_waveform/dra_reference.json`，在与波形相同的 3/5/7 信道、
100 km、50 GBd、2 mW/信道条件下运行解析模型，比较中心信道的 NLI SNR。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, RamanPumpConfig

DATA_DIR = Path("data/dra_waveform")
WAVEFORM_JSON = DATA_DIR / "dra_reference.json"
SYMBOL_RATE_HZ = 50e9
#: 波形数据的 launch_power_dbm_per_pol=0 dBm，对应 PCFM 的 per-pol 1 mW。
LAUNCH_POWER_W = 1e-3
FIBER = FiberSpanConfig(
    length_km=100.0,
    attenuation_db_km_ref=0.2,
    dispersion_parameter_d=15.68076226687911,
    dispersion_slope_s=0.04123833468291227,
    effective_area_um2_ref=80.0,
    nonlinear_index_n2=2.6e-20,
)
MODELS = (
    "hybrid_raman_paper",
    "buglia2024_ra_gn",
    "pcfm_h3_raman",
    "numerical_dra",
    "pcfm",
)

#: 支持 ``pump_depletion`` 对照口径的模型。
UNDEPLETED_MODELS = ("hybrid_raman_paper", "pcfm_h3_raman")


def _pumps(config_name: str) -> list[RamanPumpConfig]:
    """把参考数据的配置名映射为泵浦列表。"""
    if config_name == "nopump":
        return []
    direction = "forward" if config_name.startswith("fwd") else "backward"
    power = float(config_name.split("_")[1].replace("p", ".").removesuffix("w"))
    return [RamanPumpConfig(wavelength_nm=1450.0, power_w=power, direction=direction, attenuation_db_per_km=0.25)]


def _grid(n_channels: int) -> tuple[FlexibleGrid, list[ConnectionSpec]]:
    """构造 100 GHz 间隔、50 GHz 占用的灵活栅格。"""
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


def _analytic(
    model_name: str, n_channels: int, config_name: str, *, pump_depletion: bool = True
) -> tuple[float, float]:
    """返回中心信道 (SNR_NLI, NLI 功率)。"""
    grid, connections = _grid(n_channels)
    pumps = _pumps(config_name)
    options = {"raman_pumps": pumps} if pumps else {}
    if model_name == "hybrid_raman_paper" and pumps:
        options["nli_form"] = "paper"
    if model_name == "pcfm" and pumps:
        options["raman_model"] = "lumped"
    if model_name in UNDEPLETED_MODELS:
        options["pump_depletion"] = pump_depletion
    model = create_model(
        model_name,
        grid,
        FIBER,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        connections=connections,
        slots_per_channel=8,
        channel_bandwidth_source="occupied",
        **options,
    )
    path = MultiSpanOpticalPath([FIBER], [EDFAConfig(noise_figure_db=5.5)])
    result = model.evaluate_path(path, np.ones(n_channels) * LAUNCH_POWER_W)
    nli = np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)
    center = n_channels // 2
    snr = 10.0 * np.log10(np.asarray(result.signal_power_w)[center] / nli[center])
    return float(snr), float(nli[center])


def _measure(
    dataset: str,
    n_channels: int,
    center: int,
    config_name: str,
    wave_channel: dict,
    model_name: str,
    *,
    pump_depletion: bool,
) -> dict:
    """跑一个模型/配置并组装一行结果。"""
    snr, nli = _analytic(model_name, n_channels, config_name, pump_depletion=pump_depletion)
    wave_snr = float(wave_channel["snr_nli_db"])
    return {
        "dataset": dataset,
        "config": config_name,
        "model": model_name,
        "pump_depletion": pump_depletion,
        "waveform_snr_nli_db": wave_snr,
        "analytic_snr_nli_db": snr,
        "snr_delta_db": snr - wave_snr,
        "waveform_nli_w": float(wave_channel["nli_power_w"]),
        "analytic_nli_w": nli,
        "nli_ratio_db": float(
            10.0 * np.log10(nli / max(float(wave_channel["nli_power_w"]), 1e-30))
        ),
    }


def _print_closure_table(rows: list[dict]) -> None:
    """打印前向泵的 depleted -> undepleted 闭合量。"""
    forward = [row for row in rows if row["config"].startswith("fwd")]
    if not forward:
        return
    print("\n前向泵闭合量（解析 SNR - 波形；正 closure = 未耗尽口径更接近波形）")
    print(f"{'dataset':>8} {'config':>12} {'model':>20} {'depleted':>10} {'undepleted':>11} {'closure':>9}")
    for row in forward:
        depleted = row["depleted_snr_delta_db"]
        closure = row["closure_db"]
        depleted_text = "n/a" if depleted is None else f"{depleted:+.3f}"
        closure_text = "n/a" if closure is None else f"{closure:+.3f}"
        print(
            f"{row['dataset']:>8} {row['config']:>12} {row['model']:>20} "
            f"{depleted_text:>10} {row['snr_delta_db']:>+11.3f} {closure_text:>9}"
        )


def main() -> None:
    """Run all comparisons and write JSON."""
    reference = json.loads(WAVEFORM_JSON.read_text(encoding="utf-8"))
    rows = []
    for tag, dataset in reference["datasets"].items():
        n_channels = int(dataset["n_channels"])
        center = int(dataset["center_channel_index"])
        for config_name, config in dataset["configs"].items():
            wave_channel = config["per_channel"][center]
            for model_name in MODELS:
                try:
                    rows.append(
                        _measure(
                            tag,
                            n_channels,
                            center,
                            config_name,
                            wave_channel,
                            model_name,
                            pump_depletion=True,
                        )
                    )
                except Exception as error:  # 记录失败，不让基准整体停掉
                    rows.append(
                        {
                            "dataset": tag,
                            "config": config_name,
                            "model": model_name,
                            "pump_depletion": True,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    output = {
        "generated_by": "scripts/dra_waveform_compare.py",
        "reference": "physics_nlo_eval DistributedRamanFiber, 3/5ch, 100km, 50GBd DP-QPSK, 2mW/ch",
        "note": (
            "Waveform NLI is the residual after one complex common-gain fit in the channel band; "
            "analytic NLI is the GN power at the receiver. Compare SNR trends, not absolute NLI "
            "without accounting for the different receiver/bandwidth conventions."
        ),
        "rows": rows,
    }
    path = Path("results/raman/waveform_compare.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {path}")

    # 未耗尽泵浦对照 pass：只重跑有泵配置上的 hybrid/H3。
    depleted_delta = {
        (row["dataset"], row["config"], row["model"]): row["snr_delta_db"]
        for row in rows
        if "error" not in row
    }
    undepleted_rows: list[dict] = []
    for tag, dataset in reference["datasets"].items():
        n_channels = int(dataset["n_channels"])
        center = int(dataset["center_channel_index"])
        for config_name, config in dataset["configs"].items():
            if config_name == "nopump":
                continue
            wave_channel = config["per_channel"][center]
            for model_name in UNDEPLETED_MODELS:
                try:
                    row = _measure(
                        tag,
                        n_channels,
                        center,
                        config_name,
                        wave_channel,
                        model_name,
                        pump_depletion=False,
                    )
                    depleted = depleted_delta.get((tag, config_name, model_name))
                    row["depleted_snr_delta_db"] = depleted
                    row["closure_db"] = None if depleted is None else depleted - row["snr_delta_db"]
                    undepleted_rows.append(row)
                except Exception as error:  # 记录失败，不让基准整体停掉
                    undepleted_rows.append(
                        {
                            "dataset": tag,
                            "config": config_name,
                            "model": model_name,
                            "pump_depletion": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
    undepleted_output = {
        "generated_by": "scripts/dra_waveform_compare.py",
        "reference": output["reference"],
        "note": (
            "Control pass with pump_depletion=False: the pumps follow only their own attenuation "
            "(undepleted), matching the waveform reference convention. closure_db = "
            "depleted_snr_delta_db - undepleted snr_delta_db; positive means the undepleted "
            "convention moves the analytic model towards the waveform."
        ),
        "rows": undepleted_rows,
    }
    undepleted_path = Path("results/raman/waveform_compare_undepleted.json")
    undepleted_path.write_text(json.dumps(undepleted_output, indent=2), encoding="utf-8")
    print(f"wrote {undepleted_path}")
    _print_closure_table(undepleted_rows)


if __name__ == "__main__":
    main()
