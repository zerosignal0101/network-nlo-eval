"""WP-MERGE-FG-FLEX **第 3 轮**判据 V1' 的 ``p2_gn`` 档探针。

与 :mod:`v1p_p1_probe` 对称：那条量 ``p1_closed`` 档与 ``flexgrid-dev @ cb24704``
是否**逐位**相同；本条量 ``p2_gn`` 档与 ``wp/fg-integrate @ 57c7221`` 是否**逐位**
相同。

为什么必须**重新量**而不是复用第 2 轮的读数：第 3 轮改判时把 P2 路径上残留的
五处「``B_i`` 改名成 ``symbol_rate_hz``」全部改回了 FG 口径（符号率）——那是
**物理**改动，量级远超浮点噪声。编排方说「上轮已拿到这个读数，直接复用」，
但那条读数是**改口径之前**的；本轮如实重测。

口径与第 2 轮完全一致：``fg4a_scenarios.build_scenarios(ode_accel="off",
raman_spectrum="legacy")`` 的 **7 个场景**，逐项物理量逐位比对。

⚠️ **档位必须显式**：合并树的 ``fg4a_scenarios`` 已钉 ``nli_model="p2_gn"``；
``wp/fg-integrate`` 那棵树没有档位概念（它**只有** P2 那套物理），所以
"不给档位"在那边不是"测错档"，而是"那棵树没有档位"。本探针把两边的
``nli_model_requested`` 写进 provenance，避免读者误读。

用法（物理树由 ``PYTHONPATH`` 切换）::

    cd <tree root>
    PYTHONPATH=$PWD/src:$PWD/tests ~/miniconda3/envs/rl/bin/python \
        <MERGE_WT>/evidence/merge/v1p_p2_probe.py --out <json> [--compare <other.json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _provenance() -> dict[str, Any]:
    import network_nlo_eval

    root = Path(network_nlo_eval.__file__).resolve().parents[2]
    commit = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if (root / ".git").exists()
        else f"unknown (no .git at {root})"
    )
    return {
        "phys_commit": commit,
        "phys_tree": str(Path(network_nlo_eval.__file__).resolve()),
        "python": sys.version.split()[0],
    }


def _to_list(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def capture() -> dict[str, Any]:
    from fg4a_scenarios import build_scenarios

    scenarios = build_scenarios(ode_accel="off", raman_spectrum="legacy")
    cases = []
    for scenario in scenarios:
        outputs = scenario.outputs() if callable(getattr(scenario, "outputs", None)) else scenario.outputs
        cases.append(
            {
                "name": scenario.name,
                "outputs": {k: _to_list(v) for k, v in outputs.items()},
            }
        )
    payload = {
        "probe": "v1p_p2_probe",
        "wp": "WP-MERGE-FG-FLEX",
        "round": 3,
        "criterion": "V1' p2_gn vs wp/fg-integrate @ 57c7221 (bit-exact)",
        "ode_accel": "off",
        "raman_spectrum": "legacy",
        "provenance": _provenance(),
        "cases": cases,
    }
    payload["readings_sha256"] = hashlib.sha256(
        json.dumps(
            [(c["name"], c["outputs"]) for c in cases], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return payload


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    a = {c["name"]: c["outputs"] for c in left["cases"]}
    b = {c["name"]: c["outputs"] for c in right["cases"]}
    mismatches: list[dict[str, Any]] = []
    total = 0
    for case in sorted(set(a) | set(b)):
        if case not in a or case not in b:
            mismatches.append({"case": case, "reason": "case missing on one side"})
            continue
        for key in sorted(set(a[case]) | set(b[case])):
            va, vb = a[case].get(key), b[case].get(key)
            if isinstance(va, list) and isinstance(vb, list):
                if len(va) != len(vb):
                    mismatches.append({"case": case, "key": key, "reason": f"length {len(va)} != {len(vb)}"})
                    continue
                for i, (xa, xb) in enumerate(zip(va, vb, strict=True)):
                    total += 1
                    if repr(xa) != repr(xb):
                        mismatches.append({"case": case, "key": key, "index": i, "left": repr(xa), "right": repr(xb)})
            else:
                total += 1
                if repr(va) != repr(vb):
                    mismatches.append({"case": case, "key": key, "left": repr(va), "right": repr(vb)})
    return {
        "cases": len(set(a) | set(b)),
        "values_compared": total,
        "mismatches": len(mismatches),
        "mismatch_detail": mismatches[:20],
        "sha256_equal": left.get("readings_sha256") == right.get("readings_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", default=None)
    args = parser.parse_args()

    payload = capture()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OK: cases={len(payload['cases'])} sha256={payload['readings_sha256']}")
    print(f"    tree={payload['provenance']['phys_tree']} commit={payload['provenance']['phys_commit']}")

    if args.compare:
        other = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        result = compare(payload, other)
        print(
            f"COMPARE: cases={result['cases']} values={result['values_compared']} "
            f"mismatched={result['mismatches']} sha256_equal={result['sha256_equal']}"
        )
        for detail in result["mismatch_detail"]:
            print(f"  MISMATCH {detail}")
        return 0 if result["mismatches"] == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
