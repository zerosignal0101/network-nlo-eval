"""DRA 基准：独立数值参考 vs 论文闭式 / 快速近似。

参考模型 `numerical_dra` 用 solve_ivp + Eq. (6)-(8) 双重积分，不复用闭式核。
运行：PYTHONPATH=src python scripts/dra_benchmark.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.network.elements import RamanPumpConfig

PUMP_BW = RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="backward", attenuation_db_per_km=0.25)
PUMP_FW = RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="forward", attenuation_db_per_km=0.25)


def _one_span(case):
    """把 10 跨 benchmark 裁成一跨。"""
    path = build_path(case)
    return MultiSpanOpticalPath(path.fiber_configs[:1], path.edfa_configs[:1])


def _run(name, case, path, power, **opts):
    model = make_model(name, case, **opts)
    started = time.perf_counter()
    result = model.evaluate_path(path, power)
    elapsed = (time.perf_counter() - started) * 1e3
    nli = np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)
    return {
        "model": name,
        "options": {key: value for key, value in opts.items() if key != "raman_pumps"},
        "pump": "none" if not opts.get("raman_pumps") else opts["raman_pumps"][0].direction,
        "mean_snr_db": float(np.mean(result.snr_db)),
        "mean_nli_w": float(np.mean(nli)),
        "mean_ase_w": float(np.mean(result.ase_noise_w)),
        "warm_ms": float(elapsed),
        "snr_db": np.asarray(result.snr_db).tolist(),
        "nli_w": nli.tolist(),
    }


def main() -> None:
    """Run all DRA benchmark scenarios and write JSON."""
    scenarios = []
    c57 = build_case("c_57ch")
    c57_path = build_path(c57)
    c57_power = launch_profile(c57)
    scenarios.append(("c_57ch_10span", c57, c57_path, c57_power, None))
    scenarios.append(("c_57ch_10span_bw", c57, c57_path, c57_power, [PUMP_BW]))
    scenarios.append(("c_57ch_1span_bw", c57, _one_span(c57), c57_power, [PUMP_BW]))
    scenarios.append(("c_57ch_1span_fw", c57, _one_span(c57), c57_power, [PUMP_FW]))
    s40 = build_case("scaling_40ch")
    scenarios.append(("scaling_40ch_10span_bw", s40, build_path(s40), launch_profile(s40), [PUMP_BW]))

    rows = []
    comparisons = []
    for label, case, path, power, pumps in scenarios:
        ref = _run("numerical_dra", case, path, power, **({"raman_pumps": pumps} if pumps else {}))
        rows.append({"scenario": label, **ref})
        for model in (
            "hybrid_raman_paper",
            "buglia2024_ra_gn",
            "zhang2024_dra_egn",
            "pcfm",
        ):
            options = {"raman_pumps": pumps} if pumps else {}
            if model == "pcfm" and pumps:
                options["raman_model"] = "lumped"
            if model == "hybrid_raman_paper" and pumps:
                options["nli_form"] = "paper"
            try:
                row = _run(model, case, path, power, **options)
            except Exception as error:  # 记录失败，不让整个基准停掉
                rows.append({"scenario": label, "model": model, "error": f"{type(error).__name__}: {error}"})
                continue
            rows.append({"scenario": label, **row})
            error = np.asarray(row["snr_db"]) - np.asarray(ref["snr_db"])
            comparisons.append(
                {
                    "scenario": label,
                    "model": model,
                    "snr_error_mean_db": float(np.mean(error)),
                    "snr_error_abs_mean_db": float(np.mean(np.abs(error))),
                    "snr_error_max_abs_db": float(np.max(np.abs(error))),
                    "runtime_ms": row["warm_ms"],
                }
            )
        print(label, "reference", round(ref["mean_snr_db"], 3), "done")

    output = {
        "generated_by": "scripts/dra_benchmark.py",
        "reference": "numerical_dra (solve_ivp + numerical Eq. 6-8, no closed-form reuse)",
        "note": (
            "NLI is power-summed across spans (epsilon=0), matching Buglia/hybrid default; "
            "Zhang coherent terms are not in this reference."
        ),
        "rows": rows,
        "comparisons": comparisons,
    }
    path = Path("results/raman/dra_benchmark.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
