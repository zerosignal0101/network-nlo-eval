"""FG-1 physics parity harness.

Two modes:

``--emit-golden PATH``
    Run every fixed scenario and write the raw numbers. Executed against a
    pristine ``origin/main`` checkout to produce ``results/fg1_golden_main.json``.

``--check GOLDEN --out PARITY``
    Recompute the same scenarios on the current tree, compare point by point
    against the golden file, and write ``results/fg1_physics_parity.json``.

The comparison is a *bit-exact* equality check. Any deviation is reported with
its absolute and relative magnitude rather than being tolerated.

Usage (always set PYTHONPATH to the tree under test):

    PYTHONPATH=$PWD/src python scripts/fg1_parity.py --emit-golden results/fg1_golden_main.json
    PYTHONPATH=$PWD/src python scripts/fg1_parity.py \
        --check results/fg1_golden_main.json --out results/fg1_physics_parity.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from fg1_scenarios import SCENARIOS, Scenario, run_reference  # noqa: E402


def _golden_payload() -> dict[str, Any]:
    return {
        "schema": "fg1-golden-v1",
        "source_branch": "origin/main",
        "source_commit": os.environ.get("FG1_SOURCE_COMMIT", "unknown"),
        "source_tree": os.environ.get("FG1_SOURCE_TREE", "unknown"),
        "scenarios": {scenario.name: run_reference(scenario) for scenario in SCENARIOS},
    }


def emit_golden(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_golden_payload(), indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote golden -> {path}")


def _pipeline_payload(scenario: Scenario) -> dict[str, Any] | None:
    """Evaluate the same scenario through the ported pipeline, if it exists."""
    try:
        from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
        from network_nlo_eval.models.isrs_gn import MultiBandISRSGN as _Evaluator
    except ImportError:
        return None
    if not hasattr(_Evaluator, "evaluate_path"):
        return None

    grid = scenario.grid()
    evaluator = _Evaluator(
        grid,
        scenario.fiber(scenario.hops[0]),
        symbol_rate_hz=scenario.symbol_rate_hz,
        phi_kurtosis=scenario.phi_kurtosis,
    )
    # max_span_length_km reproduces the reference span split exactly.
    fibers, edfas, roadms = [], [], []
    for hop in scenario.hops:
        fibers.append(
            scenario.fiber(hop).model_copy(update={"max_span_length_km": hop.span_km})
        )
        edfas.append(scenario.edfa(hop))
        roadm = scenario.roadm(hop)
        if roadm is not None:
            roadm = roadm.model_copy(update={"blocked_channels": list(hop.blocked_channels)})
        roadms.append(roadm)
    path = MultiSpanOpticalPath(
        fiber_configs=fibers, edfa_configs=edfas, roadm_configs=roadms
    )
    result = evaluator.evaluate_path(path, scenario.launch_power())
    return {
        "signal_power_w": np.asarray(result.signal_power_w).tolist(),
        "spm_w": np.asarray(result.spm_noise_w).tolist(),
        "xpm_w": np.asarray(result.xpm_noise_w).tolist(),
        "ase_w": np.asarray(result.ase_noise_w).tolist(),
        "snr_db": np.asarray(result.snr_db).tolist(),
        "span_count": result.span_count,
    }


def _compare(golden: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    """Point-by-point comparison of two arrays, treating non-finite as equal."""
    if golden.shape != candidate.shape:
        return {
            "comparable": False,
            "reason": f"shape mismatch {golden.shape} vs {candidate.shape}",
            "max_abs_dev": None,
            "max_rel_dev": None,
        }
    g_finite, c_finite = np.isfinite(golden), np.isfinite(candidate)
    non_finite_mismatch = int(np.count_nonzero(g_finite != c_finite))
    g, c = golden[g_finite], candidate[c_finite]
    if g.size == 0:
        return {
            "comparable": True,
            "max_abs_dev": 0.0,
            "max_rel_dev": 0.0,
            "n_points": 0,
            "non_finite_mismatch": non_finite_mismatch,
            "bit_exact": non_finite_mismatch == 0,
        }
    abs_dev = np.abs(g - c)
    denom = np.maximum(np.abs(g), 1e-300)
    rel_dev = abs_dev / denom
    return {
        "comparable": True,
        "n_points": int(g.size),
        "max_abs_dev": float(np.max(abs_dev)),
        "max_rel_dev": float(np.max(rel_dev)),
        "non_finite_mismatch": non_finite_mismatch,
        "bit_exact": bool(np.max(abs_dev) == 0.0 and non_finite_mismatch == 0),
    }


def check(golden_path: Path, out_path: Path) -> int:
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    report: dict[str, Any] = {
        "schema": "fg1-parity-v1",
        "golden_file": str(golden_path),
        "golden_source_branch": golden.get("source_branch"),
        "tolerance": 0.0,
        "tolerance_rationale": (
            "numba_kernels.py is byte-identical to origin/main and the pipeline only "
            "orchestrates the same low-level kernel calls, so the expected deviation "
            "is exactly zero. Any non-zero deviation is a regression, not noise."
        ),
        "golden_file_digest_note": "run scripts/fg1_parity.py --emit-golden against origin/main to regenerate",
        "scenarios": {},
    }

    failures: list[str] = []
    for scenario in SCENARIOS:
        expected = golden["scenarios"].get(scenario.name)
        if expected is None:
            failures.append(f"{scenario.name}: missing from golden file")
            continue
        actual = run_reference(scenario)

        entry: dict[str, Any] = {"scenario": scenario.to_dict(), "checks": {}}
        for key in ("signal_power_w", "spm_w", "xpm_w", "ase_w", "snr_db"):
            cmp = _compare(np.array(expected[key], dtype=float), np.array(actual[key], dtype=float))
            entry["checks"][key] = cmp
            if not cmp.get("bit_exact"):
                failures.append(f"{scenario.name}.{key}: {cmp}")

        # Power-evolution curve, point by point along the whole path.
        curve_checks = []
        for exp_span, act_span in zip(expected["span_curve"], actual["span_curve"], strict=True):
            for key in ("power_out_w", "spm_w", "xpm_w", "ase_w"):
                cmp = _compare(np.array(exp_span[key], dtype=float), np.array(act_span[key], dtype=float))
                curve_checks.append(
                    {"hop": exp_span["hop"], "span": exp_span["span"], "field": key, **cmp}
                )
                if not cmp.get("bit_exact"):
                    failures.append(
                        f"{scenario.name}.curve[h{exp_span['hop']}s{exp_span['span']}].{key}: {cmp}"
                    )
        entry["span_curve_checks"] = curve_checks

        # Pipeline vs reference: proves the new pipeline is physics-neutral.
        pipeline = _pipeline_payload(scenario)
        if pipeline is None:
            entry["pipeline_vs_reference"] = {"available": False}
        else:
            pipe_checks = {}
            for key in ("signal_power_w", "spm_w", "xpm_w", "ase_w", "snr_db"):
                cmp = _compare(np.array(expected[key], dtype=float), np.array(pipeline[key], dtype=float))
                pipe_checks[key] = cmp
                if not cmp.get("bit_exact"):
                    failures.append(f"{scenario.name}.pipeline.{key}: {cmp}")
            entry["pipeline_vs_reference"] = {"available": True, "checks": pipe_checks}
        report["scenarios"][scenario.name] = entry

    report["failures"] = failures
    report["parity_ok"] = not failures
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote parity -> {out_path}")
    print(f"parity_ok = {report['parity_ok']}  failures = {len(failures)}")
    for failure in failures[:20]:
        print("  FAIL", failure)
    return 0 if report["parity_ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-golden", type=Path)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.emit_golden:
        emit_golden(args.emit_golden)
        return 0
    if args.check:
        if not args.out:
            parser.error("--check requires --out")
        return check(args.check, args.out)
    parser.error("one of --emit-golden or --check is required")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
