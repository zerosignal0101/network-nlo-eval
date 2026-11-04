"""用收敛耗尽参考重跑 ``pcfm_h3_raman`` 的前向 EGN 三档对照。

只读脚本，复用 ``scripts/dra_waveform_compare.py`` 的光纤、栅格与泵浦构造
（同一取数口径：单跨 100 km、50 GBd、1 mW/偏振、中心信道），不修改它的
核心逻辑。

对 3/5/7ch（外加 1ch SPM 探针）前向 0.2 W 比较：

* ``egn="auto"``（新默认：前向泵关闭 EGN）
* ``egn=True``（旧默认）
* ``egn=False``

并给出 ``hybrid_raman_paper(nli_form="paper")`` 与 62.5 m 收敛耗尽波形的
中心信道 ``SNR_NLI``。无泵/后向 0.2 W 另外验证 ``"auto"`` 与显式
``egn=True`` 逐位一致。

输出 ``results/raman/h3_egn_auto.json``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

# 与波形对照脚本共用同一栅格/光纤/泵浦解析，保证数字可对照。
from dra_waveform_compare import FIBER, LAUNCH_POWER_W, SYMBOL_RATE_HZ, _grid, _pumps

from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig

#: 参考数据的层叠顺序（1ch 是单信道 SPM 探针，不属于 3/5/7 主表）。
DATASETS = (("1ch", 1), ("3ch", 3), ("5ch", 5), ("7ch", 7))
#: 前向泵对照配置。
FORWARD_CONFIG = "fwd_0p20w"
#: 必须保持逐位一致的配置（无泵/后向）。
INVARIANCE_CONFIGS = ("nopump", "bwd_0p20w")
#: 三档 EGN 口径。
EGN_MODES = ("auto", True, False)
#: 逐位一致比较覆盖的输出数组。
OUTPUT_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")

DEFAULT_REFERENCE = Path("data/dra_waveform_converged/dra_reference_depleted.json")
DEFAULT_OUTPUT = Path("results/raman/h3_egn_auto.json")


def _run(model_name: str, n_channels: int, config: str, **options: Any):
    """按波形对照脚本的栅格跑一次单跨路径。"""
    grid, connections = _grid(n_channels, LAUNCH_POWER_W)
    pumps = _pumps(config)
    resolved: dict[str, Any] = {"raman_pumps": pumps} if pumps else {}
    if model_name == "hybrid_raman_paper" and pumps:
        resolved["nli_form"] = "paper"
    # 参考是 62.5 m 耗尽耦合 ODE。
    resolved["pump_depletion"] = True
    resolved.update(options)
    model = create_model(
        model_name,
        grid,
        FIBER,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        connections=connections,
        slots_per_channel=8,
        channel_bandwidth_source="occupied",
        **resolved,
    )
    path = MultiSpanOpticalPath([FIBER], [EDFAConfig(noise_figure_db=5.5)])
    return model.evaluate_path(path, np.full(n_channels, LAUNCH_POWER_W, dtype=np.float64))


def _center_snr(result, n_channels: int) -> float:
    """中心信道 NLI-limited SNR (dB)，与波形对照脚本同口径。"""
    center = n_channels // 2
    signal = float(np.asarray(result.signal_power_w)[center])
    nli = float(np.asarray(result.spm_noise_w)[center] + np.asarray(result.xpm_noise_w)[center])
    return float(10.0 * np.log10(signal / nli))


def _waveform_center_snr(dataset: str, config: str, reference: dict) -> float:
    """从参考 JSON 读中心信道的波形 SNR_NLI (dB)。"""
    entry = reference["datasets"][dataset]
    center = int(entry["center_channel_index"])
    return float(entry["configs"][config]["per_channel"][center]["snr_nli_db"])


def _max_abs_diff(left, right) -> float:
    """两路径结果的全部输出数组最大绝对差。"""
    return float(
        max(
            np.max(np.abs(np.asarray(getattr(left, name)) - np.asarray(getattr(right, name))))
            for name in OUTPUT_FIELDS
        )
    )


def _forward_row(dataset: str, n_channels: int, reference: dict) -> dict[str, Any]:
    """一个数据集的前向三档 EGN 行。"""
    wave = _waveform_center_snr(dataset, FORWARD_CONFIG, reference)
    hybrid = _center_snr(_run("hybrid_raman_paper", n_channels, FORWARD_CONFIG), n_channels)
    values = {
        str(mode): _center_snr(
            _run("pcfm_h3_raman", n_channels, FORWARD_CONFIG, egn=mode),
            n_channels,
        )
        for mode in EGN_MODES
    }
    return {
        "dataset": dataset,
        "n_channels": n_channels,
        "config": FORWARD_CONFIG,
        "waveform_snr_nli_db": wave,
        "hybrid_snr_nli_db": hybrid,
        "h3_auto_snr_nli_db": values["auto"],
        "h3_egn_true_snr_nli_db": values["True"],
        "h3_egn_false_snr_nli_db": values["False"],
        "h3_auto_delta_db": values["auto"] - wave,
        "h3_egn_true_delta_db": values["True"] - wave,
        "h3_egn_false_delta_db": values["False"] - wave,
        "hybrid_delta_db": hybrid - wave,
        "h3_auto_minus_hybrid_db": values["auto"] - hybrid,
        "egn_contribution_db": values["True"] - values["False"],
        "auto_equals_explicit_false": bool(values["auto"] == values["False"]),
    }


def _invariance_row(dataset: str, n_channels: int, config: str) -> dict[str, Any]:
    """无泵/后向：``"auto"`` 与显式 ``egn=True`` 的输出最大绝对差。"""
    auto = _run("pcfm_h3_raman", n_channels, config)
    legacy = _run("pcfm_h3_raman", n_channels, config, egn=True)
    return {
        "dataset": dataset,
        "n_channels": n_channels,
        "config": config,
        "auto_vs_true_max_abs_diff": _max_abs_diff(auto, legacy),
    }


def main() -> None:
    """跑全部对照并写 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="收敛耗尽参考 JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出 JSON 路径")
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    rows = [_forward_row(dataset, n_channels, reference) for dataset, n_channels in DATASETS]
    invariance = [
        _invariance_row(dataset, n_channels, config)
        for dataset, n_channels in DATASETS[1:]
        for config in INVARIANCE_CONFIGS
    ]

    output = {
        "generated_by": "scripts/h3_egn_auto.py",
        "reference_path": str(args.reference),
        "reference_pump_model": reference.get("pump_model"),
        "coarse_step_m": reference.get("coarse_step_m"),
        "config": {
            "fiber": {
                "length_km": FIBER.length_km,
                "attenuation_db_km_ref": FIBER.attenuation_db_km_ref,
                "dispersion_parameter_d": FIBER.dispersion_parameter_d,
                "effective_area_um2_ref": FIBER.effective_area_um2_ref,
            },
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "launch_power_w_per_pol": LAUNCH_POWER_W,
            "grid": "100 GHz spacing, 50 GHz occupied, FlexibleGrid 12.5 GHz",
            "pump": {"wavelength_nm": 1450.0, "power_w": 0.2, "attenuation_db_per_km": 0.25},
            "edfa_noise_figure_db": 5.5,
            "snr_metric": "center-channel NLI-limited SNR_NLI, same convention as dra_waveform_compare.py",
        },
        "note": (
            'egn="auto" disables the Zefreh correction when any active pump is forward. '
            "Rows are forward-pumped 0.2 W on the 62.5 m converged depleted reference; "
            "invariance rows show that no-pump and backward-pump results are unchanged "
            "(auto vs explicit egn=True)."
        ),
        "forward_rows": rows,
        "invariance": invariance,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    header = (
        f"{'dataset':>8}{'wave':>9}{'hybrid':>9}{'auto':>9}{'egn=True':>10}{'egn=False':>10}"
        f"{'auto-wave':>11}{'true-wave':>11}{'hybrid-wave':>12}"
    )
    print(header)
    for row in rows:
        print(
            f"{row['dataset']:>8}"
            f"{row['waveform_snr_nli_db']:>9.3f}"
            f"{row['hybrid_snr_nli_db']:>9.3f}"
            f"{row['h3_auto_snr_nli_db']:>9.3f}"
            f"{row['h3_egn_true_snr_nli_db']:>10.3f}"
            f"{row['h3_egn_false_snr_nli_db']:>10.3f}"
            f"{row['h3_auto_delta_db']:>+11.3f}"
            f"{row['h3_egn_true_delta_db']:>+11.3f}"
            f"{row['hybrid_delta_db']:>+12.3f}"
        )
    for row in invariance:
        print(
            f"{row['dataset']:>8} {row['config']:>10} auto vs egn=True max|diff| = "
            f"{row['auto_vs_true_max_abs_diff']:.3e}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
