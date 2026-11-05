"""把 physical-nlo-eval 的 DRA 波形参考接入 network-nlo-eval。

读取参考目录中的 ``dra_reference.json``（默认 ``data/dra_waveform``），在与
波形相同的 3/5/7 信道、100 km、50 GBd、1 mW/偏振条件下运行解析模型，
比较中心信道的 NLI SNR。

参考 JSON 的 ``pump_model`` 决定解析侧的 ``pump_depletion`` 口径：
``undepleted`` -> ``False``，``depleted`` -> ``True``。这样物理波形和解析模型
使用同一泵浦耗尽假设；``numerical_dra`` / ``buglia2024_ra_gn`` / lumped
``pcfm`` 没有该开关，行内会用 ``pump_depletion_applied=False`` 标注。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, RamanPumpConfig

DEFAULT_DATA_DIR = Path("data/dra_waveform")
DEFAULT_REFERENCE_NAME = "dra_reference.json"
DEFAULT_OUTPUT = Path("results/raman/waveform_compare.json")
CONVERGED_DATA_DIR_NAME = "dra_waveform_converged"
#: WP5a 收敛数据的步长上限；超过它的参考不能当作收敛基准。
MAX_CONVERGED_STEP_M = 62.5

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
DEPLETION_AWARE_MODELS = ("hybrid_raman_paper", "pcfm_h3_raman")


def _warn(message: str) -> None:
    """Print a warning to stderr."""
    print(f"warning: {message}", file=sys.stderr)


def _pumps(config_name: str) -> list[RamanPumpConfig]:
    """把参考数据的配置名映射为泵浦列表。"""
    if config_name == "nopump":
        return []
    direction = "forward" if config_name.startswith("fwd") else "backward"
    power = float(config_name.split("_")[1].replace("p", ".").removesuffix("w"))
    return [RamanPumpConfig(wavelength_nm=1450.0, power_w=power, direction=direction, attenuation_db_per_km=0.25)]


def _grid(n_channels: int, launch_power_w: float) -> tuple[FlexibleGrid, list[ConnectionSpec]]:
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
                launch_power_w=launch_power_w,
            )
        )
    return grid, connections


def _dataset_launch_power_w(reference: dict, dataset: dict) -> float:
    """返回数据集的发射功率（每偏振），缺省回落到顶层 signal。"""
    dbm = dataset.get("launch_power_dbm_per_pol")
    if dbm is None:
        dbm = reference.get("signal", {}).get("launch_power_dbm_per_pol", 0.0)
    return float(LAUNCH_POWER_W * 10.0 ** (float(dbm) / 10.0))


def _reference_pump_model(reference: dict, reference_path: Path) -> tuple[str, bool]:
    """读取参考泵浦口径并映射到解析侧的 ``pump_depletion``。"""
    pump_model = reference.get("pump_model")
    if pump_model is None:
        _warn(f"{reference_path} has no top-level pump_model; assuming depleted (model default)")
        return "unknown", True
    pump_model = str(pump_model)
    if pump_model not in ("undepleted", "depleted"):
        _warn(f"{reference_path} has unknown pump_model={pump_model!r}; assuming depleted (model default)")
        return pump_model, True
    return pump_model, pump_model == "depleted"


def _check_step_metadata(reference: dict, reference_path: Path, data_dir: Path) -> None:
    """检查步长/收敛元数据；收敛数据目录必须满足硬性条件。"""
    converged_dir = data_dir.name == CONVERGED_DATA_DIR_NAME
    top_step = reference.get("coarse_step_m")
    if top_step is None:
        _warn(f"{reference_path} has no top-level coarse_step_m")
    if converged_dir and top_step is not None and float(top_step) > MAX_CONVERGED_STEP_M:
        raise SystemExit(
            f"{reference_path} coarse_step_m={top_step} exceeds {MAX_CONVERGED_STEP_M} m for a converged reference"
        )

    for tag, dataset in reference["datasets"].items():
        step = dataset.get("coarse_step_m", top_step)
        flag = dataset.get("step_converged")
        if step is None:
            _warn(f"dataset {tag!r} in {reference_path} has no coarse_step_m")
        if flag is None:
            _warn(f"dataset {tag!r} in {reference_path} has no step_converged")
        if not converged_dir:
            continue
        if flag is not True:
            raise SystemExit(f"dataset {tag!r} in {reference_path} is not marked step_converged")
        if step is None or float(step) > MAX_CONVERGED_STEP_M:
            raise SystemExit(
                f"dataset {tag!r} in {reference_path} has coarse_step_m={step}, "
                f"expected <= {MAX_CONVERGED_STEP_M} m"
            )


def _analytic(
    model_name: str,
    n_channels: int,
    config_name: str,
    *,
    pump_depletion: bool,
    launch_power_w: float,
    signal_polarizations: int = 1,
) -> tuple[float, float]:
    """返回中心信道 (SNR_NLI, NLI 功率)。"""
    grid, connections = _grid(n_channels, launch_power_w)
    pumps = _pumps(config_name)
    options = {"raman_pumps": pumps} if pumps else {}
    if model_name == "hybrid_raman_paper" and pumps:
        options["nli_form"] = "paper"
    if model_name == "pcfm" and pumps:
        options["raman_model"] = "lumped"
    if model_name in DEPLETION_AWARE_MODELS:
        options["pump_depletion"] = pump_depletion
        options["signal_polarizations"] = int(signal_polarizations)
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
    result = model.evaluate_path(path, np.ones(n_channels) * launch_power_w)
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
    pump_model: str,
    pump_depletion: bool,
    launch_power_w: float,
    signal_polarizations: int = 1,
) -> dict:
    """跑一个模型/配置并组装一行结果。"""
    applied = model_name in DEPLETION_AWARE_MODELS
    snr, nli = _analytic(
        model_name,
        n_channels,
        config_name,
        pump_depletion=pump_depletion,
        launch_power_w=launch_power_w,
        signal_polarizations=signal_polarizations,
    )
    wave_snr = float(wave_channel["snr_nli_db"])
    row = {
        "dataset": dataset,
        "n_channels": n_channels,
        "config": config_name,
        "model": model_name,
        "pump_model": pump_model,
        "pump_depletion": pump_depletion,
        "pump_depletion_applied": applied,
        "launch_power_w": launch_power_w,
        "waveform_snr_nli_db": wave_snr,
        "analytic_snr_nli_db": snr,
        "snr_delta_db": snr - wave_snr,
        "waveform_nli_w": float(wave_channel["nli_power_w"]),
        "analytic_nli_w": nli,
        "nli_ratio_db": float(10.0 * np.log10(nli / max(float(wave_channel["nli_power_w"]), 1e-30))),
    }
    if not applied:
        row["pump_depletion_note"] = (
            "model has no pump_depletion option; it uses its own default (coupled/depleted ODE or lumped)"
        )
        if pump_model == "undepleted":
            row["pump_model_mismatch"] = True
    return row


def _print_summary(rows: list[dict]) -> None:
    """打印 hybrid/H3 的中心信道误差，便于当场核对。"""
    selected = [row for row in rows if row.get("model") in DEPLETION_AWARE_MODELS and "error" not in row]
    if not selected:
        return
    print(f"\n{'dataset':>14} {'config':>12} {'model':>20} {'wave':>9} {'analytic':>9} {'delta':>8}")
    for row in selected:
        print(
            f"{row['dataset']:>14} {row['config']:>12} {row['model']:>20} "
            f"{row['waveform_snr_nli_db']:>9.3f} {row['analytic_snr_nli_db']:>9.3f} "
            f"{row['snr_delta_db']:>+8.3f}"
        )


def _resolve_output_path(data_dir: Path, output: Path | None) -> Path:
    """解析输出路径；默认与旧脚本保持兼容，不覆盖旧结果。"""
    if output is not None:
        return output
    if data_dir.name == DEFAULT_DATA_DIR.name:
        return DEFAULT_OUTPUT
    return Path("results/raman") / f"waveform_compare_{data_dir.name}.json"


def _write_control_pass(
    reference: dict,
    main_rows: list[dict],
    *,
    reference_path: Path,
    pump_model: str,
    pump_depletion: bool,
    control_output: Path,
    signal_polarizations: int = 1,
) -> None:
    """用相反的 ``pump_depletion`` 口径重跑 hybrid/H3，并写闭合量。"""
    control_depletion = not pump_depletion
    control_model = "depleted" if control_depletion else "undepleted"
    main_delta = {
        (row["dataset"], row["config"], row["model"]): row["snr_delta_db"]
        for row in main_rows
        if "error" not in row and row["model"] in DEPLETION_AWARE_MODELS
    }
    rows = []
    for tag, dataset in reference["datasets"].items():
        n_channels = int(dataset["n_channels"])
        center = int(dataset["center_channel_index"])
        launch_power_w = _dataset_launch_power_w(reference, dataset)
        for config_name, config in dataset["configs"].items():
            if config_name == "nopump":
                continue
            wave_channel = config["per_channel"][center]
            for model_name in DEPLETION_AWARE_MODELS:
                row = _measure(
                    tag,
                    n_channels,
                    center,
                    config_name,
                    wave_channel,
                    model_name,
                    pump_model=control_model,
                    pump_depletion=control_depletion,
                    launch_power_w=launch_power_w,
                    signal_polarizations=signal_polarizations,
                )
                reference_delta = main_delta.get((tag, config_name, model_name))
                if reference_delta is None:
                    row["closure_db"] = None
                elif control_depletion:
                    row["depleted_snr_delta_db"] = row["snr_delta_db"]
                    row["undepleted_snr_delta_db"] = reference_delta
                    row["closure_db"] = row["snr_delta_db"] - reference_delta
                else:
                    row["undepleted_snr_delta_db"] = row["snr_delta_db"]
                    row["depleted_snr_delta_db"] = reference_delta
                    row["closure_db"] = reference_delta - row["snr_delta_db"]
                rows.append(row)

    control_output.parent.mkdir(parents=True, exist_ok=True)
    control_output.write_text(
        json.dumps(
            {
                "generated_by": "scripts/dra_waveform_compare.py --control-output",
                "reference_path": str(reference_path),
                "reference_pump_model": pump_model,
                "control_pump_model": control_model,
                "control_pump_depletion": control_depletion,
                "note": (
                    "Control pass with the opposite pump_depletion convention for hybrid/H3. "
                    "closure_db = depleted_snr_delta_db - undepleted_snr_delta_db; positive "
                    "means the depleted convention is farther from the waveform."
                ),
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {control_output}")
    forward = [row for row in rows if row["config"].startswith("fwd")]
    if forward:
        print("\n前向泵口径对照（解析 SNR - 波形；正 closure = 耗尽口径更远）")
        print(f"{'dataset':>14} {'config':>12} {'model':>20} {'control':>10} {'closure':>9}")
        for row in forward:
            closure = row["closure_db"]
            closure_text = "n/a" if closure is None else f"{closure:+.3f}"
            print(
                f"{row['dataset']:>14} {row['config']:>12} {row['model']:>20} "
                f"{row['snr_delta_db']:>+10.3f} {closure_text:>9}"
            )



def main() -> None:
    """Run all comparisons and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--output", type=Path, default=None, help="输出 JSON 路径")
    parser.add_argument(
        "--control-output",
        type=Path,
        default=None,
        help="可选：用相反的 pump_depletion 口径重跑 hybrid/H3，另存闭合量 JSON",
    )
    parser.add_argument(
        "--signal-polarizations",
        type=int,
        choices=(1, 2),
        default=1,
        help="ODE 信号状态代表的偏振数：1=历史每偏振驱动（默认），2=双偏振总功率驱动",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    reference_path = Path(args.reference)
    if not reference_path.is_absolute() and reference_path.parent == Path("."):
        reference_path = data_dir / reference_path
    if not reference_path.exists():
        raise SystemExit(f"reference JSON not found: {reference_path}")

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    pump_model, pump_depletion = _reference_pump_model(reference, reference_path)
    _check_step_metadata(reference, reference_path, data_dir)

    rows = []
    for tag, dataset in reference["datasets"].items():
        n_channels = int(dataset["n_channels"])
        center = int(dataset["center_channel_index"])
        launch_power_w = _dataset_launch_power_w(reference, dataset)
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
                            pump_model=pump_model,
                            pump_depletion=pump_depletion,
                            launch_power_w=launch_power_w,
                            signal_polarizations=args.signal_polarizations,
                        )
                    )
                except Exception as error:  # 记录失败，不让基准整体停掉
                    rows.append(
                        {
                            "dataset": tag,
                            "n_channels": n_channels,
                            "config": config_name,
                            "model": model_name,
                            "pump_model": pump_model,
                            "pump_depletion": pump_depletion,
                            "pump_depletion_applied": model_name in DEPLETION_AWARE_MODELS,
                            "launch_power_w": launch_power_w,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )

    n_values = sorted({int(dataset["n_channels"]) for dataset in reference["datasets"].values()})
    step_flags = [dataset.get("step_converged") for dataset in reference["datasets"].values()]
    output = {
        "generated_by": "scripts/dra_waveform_compare.py",
        "reference_path": str(reference_path),
        "reference": (
            "physics_nlo_eval DistributedRamanFiber, "
            + "/".join(str(value) for value in n_values)
            + "ch, 100km, 50GBd DP-QPSK, per-dataset launch power"
        ),
        "pump_model": pump_model,
        "pump_depletion": pump_depletion,
        "coarse_step_m": reference.get("coarse_step_m"),
        "step_converged": all(step_flags) if any(flag is not None for flag in step_flags) else None,
        "note": (
            "Waveform NLI is the residual after one complex common-gain fit in the channel band; "
            "analytic NLI is the GN power at the transmitter plane. Compare SNR trends, not absolute "
            "NLI without accounting for the different receiver/bandwidth conventions. "
            "pump_depletion is taken from the reference pump_model; models without that option are "
            "marked with pump_depletion_applied=false."
        ),
        "rows": rows,
    }
    if int(args.signal_polarizations) != 1:
        output["signal_polarizations"] = int(args.signal_polarizations)
    path = _resolve_output_path(data_dir, args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    _print_summary(rows)

    if args.control_output is not None:
        _write_control_pass(
            reference,
            rows,
            reference_path=reference_path,
            pump_model=pump_model,
            pump_depletion=pump_depletion,
            control_output=args.control_output,
            signal_polarizations=args.signal_polarizations,
        )


if __name__ == "__main__":
    main()
