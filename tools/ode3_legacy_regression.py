#!/usr/bin/env python3
"""WP-O3 步骤 2 的另一半：J6 legacy 逐位回归守卫。

**协议完全沿用 WP-O2**，直接复用 ``tools/ode_compare.py`` 的
``scripts_shared/compat_run.py`` + ``cross_snr_compat.py`` 对照方法与签名兼容层，
本工具不重新发明跑法——否则"J6 没退步"就成了另一套口径下的自说自话。

判据（J6）：6 个 ``legacy_*`` 场景上 ``ode-dev`` 与 ``ecz-dev`` 的
``delta_db`` 必须**恰为 0.0**（逐位相同），且拒绝场景必须**两边都拒**。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode3_legacy_regression.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode3_legacy_regression.py --check

产物：``results/ode_dev_legacy_regression.json``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ode_dev_legacy_regression.json"
BASELINE = "ecz-dev"

#: 逐位相同的判据：不设容差。0.0 就是 0.0。
REQUIRED_DELTA = 0.0


def _load_ode_compare():
    spec = importlib.util.spec_from_file_location("ode_compare", REPO / "tools" / "ode_compare.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run() -> dict:
    oc = _load_ode_compare()
    raw = oc.run_all()
    diff = oc.build_diff(raw, BASELINE)

    rows = []
    for r in diff["scenarios"]:
        both_reject = r["accepted_ode_dev"] is False and r["accepted_other"] is False
        rows.append(
            {
                "scenario": r["scenario"],
                "ode_dev_snr_db": r["ode_dev_snr_db"],
                "ecz_dev_snr_db": r["other_snr_db"],
                "delta_db": r["delta_db"],
                "bit_identical": r["bit_identical"],
                "accepted_ode_dev": r["accepted_ode_dev"],
                "accepted_ecz_dev": r["accepted_other"],
                "both_rejected": both_reject,
                "ode_dev_error": r["ode_dev_error"],
                "ecz_dev_error": r["other_error"],
                "verdict": _verdict(r, both_reject),
            }
        )

    return {
        "wp": "O3",
        "judgment": "J6",
        "baseline_tree": BASELINE,
        "baseline_path": str(oc.TREES[BASELINE]),
        "candidate_tree": "ode-dev",
        "protocol": "复用 tools/ode_compare.py（scripts_shared/compat_run.py + "
        "cross_snr_compat.py 签名兼容层），与 WP-O2 同口径",
        "tolerance": 0.0,
        "scenarios": rows,
        "summary": {
            "n_scenarios": len(rows),
            "n_bit_identical": sum(1 for r in rows if r["bit_identical"]),
            "n_both_rejected": sum(1 for r in rows if r["both_rejected"]),
            "max_abs_delta_db": max((abs(r["delta_db"]) for r in rows
                                     if isinstance(r["delta_db"], (int, float))), default=None),
            "j6_passed": all(r["verdict"] == "pass" for r in rows) and len(rows) == 6,
        },
    }


def _verdict(r: dict, both_reject: bool) -> str:
    if both_reject:
        return "pass"  # 两端一致拒绝
    if r["delta_db"] is None:
        return "fail"
    return "pass" if r["delta_db"] == REQUIRED_DELTA else "fail"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = run()

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        for r in doc["scenarios"]:
            d = r["delta_db"]
            ds = "n/a" if d is None else f"{d:+.9f}"
            print(f"    {r['scenario']:<34} ode={r['ode_dev_snr_db']!s:<20} "
                  f"ecz={r['ecz_dev_snr_db']!s:<20} delta={ds}  [{r['verdict']}]")
        print(f"  J6: {doc['summary']['n_bit_identical']}/{doc['summary']['n_scenarios']} "
              f"逐位相同，两端一致拒绝 {doc['summary']['n_both_rejected']} 条")

    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        if stored.get("scenarios") != doc["scenarios"]:
            print("FAIL 复算与落盘不一致")
            return 1
        if not doc["summary"]["j6_passed"]:
            for r in doc["scenarios"]:
                if r["verdict"] != "pass":
                    print(f"FAIL {r['scenario']}: delta_db={r['delta_db']} "
                          f"ode={r['ode_dev_snr_db']} ecz={r['ecz_dev_snr_db']}")
            return 1
        if doc["summary"]["n_scenarios"] != 6:
            print(f"FAIL 场景数 {doc['summary']['n_scenarios']} != 6")
            return 1
        print(f"OK  J6 复算通过：6 个 legacy 场景 delta_db 全为 0.0，两端一致拒绝 "
              f"{doc['summary']['n_both_rejected']} 条")
        return 0

    if not doc["summary"]["j6_passed"]:
        print("J6 FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
