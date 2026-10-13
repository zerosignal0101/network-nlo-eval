#!/usr/bin/env python3
"""WP-M1 判据 M-J6：6 个 legacy 场景 vs ``ecz-dev``，逐条 ``delta_db == 0.0``。

**协议完全复用** ``tools/ode3_legacy_regression.py``：两者都调用
``tools/ode_compare.py`` 的 ``run_all()`` + ``build_diff()``，共用
``scripts_shared/compat_run.py`` 与 ``cross_snr_compat.py`` 的签名兼容层。
本工具**不重新发明对照口径**——否则「M-J6 没退步」就成了另一套标准下的自说自话。

为什么 WP-M1 要重跑这条：本 WP 改了 ``traffic.py`` / ``allocators.py`` /
``network/state.py`` 三处装配代码。J6 守的正是「物理与 legacy 装配算式
逐位不变」，所以每个提交都必须重新证明一次，不能靠"我们没碰物理"。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ms_alloc_legacy_regression.py --write
    ~/miniconda3/envs/rl/bin/python tools/ms_alloc_legacy_regression.py --check

产物：``results/ms_alloc_legacy_regression.json``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ms_alloc_legacy_regression.json"
BASELINE = "ecz-dev"

#: 逐位相同：容差就是 0.0，不设任何余量。
REQUIRED_DELTA = 0.0


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run() -> dict:
    oc = _load_module("ode_compare", REPO / "tools" / "ode_compare.py")
    raw = oc.run_all()
    diff = oc.build_diff(raw, BASELINE)

    rows = []
    for r in diff["scenarios"]:
        both_reject = r["accepted_ode_dev"] is False and r["accepted_other"] is False
        if both_reject:
            verdict = "pass"  # 两端一致拒绝：没有 SNR 可比，但必须都拒
        elif r["delta_db"] is None:
            verdict = "fail"
        else:
            verdict = "pass" if r["delta_db"] == REQUIRED_DELTA else "fail"
        rows.append(
            {
                "scenario": r["scenario"],
                "wp_ms_alloc_snr_db": r["ode_dev_snr_db"],
                "ecz_dev_snr_db": r["other_snr_db"],
                "delta_db": r["delta_db"],
                "bit_identical": r["bit_identical"],
                "accepted_wp_ms_alloc": r["accepted_ode_dev"],
                "accepted_ecz_dev": r["accepted_other"],
                "both_rejected": both_reject,
                "wp_ms_alloc_error": r["ode_dev_error"],
                "ecz_dev_error": r["other_error"],
                "verdict": verdict,
            }
        )

    comparable = [r for r in rows if not r["both_rejected"]]
    return {
        "wp": "M1",
        "judgment": "M-J6",
        "baseline_tree": BASELINE,
        "baseline_path": str(oc.TREES[BASELINE]),
        "candidate_tree": "wp/ms-alloc",
        "protocol": (
            "与 tools/ode3_legacy_regression.py 同口径：直接调用 tools/ode_compare.py 的 "
            "run_all()/build_diff()，共用 scripts_shared/compat_run.py + cross_snr_compat.py。"
        ),
        "tolerance": REQUIRED_DELTA,
        "why_rerun": (
            "WP-M1 改了 traffic.py / allocators.py / network/state.py 三处装配代码，"
            "J6 守的正是物理与 legacy 装配算式逐位不变，故本 WP 重新独立证明一次。"
        ),
        "scenarios": rows,
        "summary": {
            "n_scenarios": len(rows),
            "n_comparable": len(comparable),
            "n_bit_identical": sum(1 for r in rows if r["bit_identical"]),
            "n_both_rejected": sum(1 for r in rows if r["both_rejected"]),
            "max_abs_delta_db": max(
                (abs(r["delta_db"]) for r in comparable if isinstance(r["delta_db"], (int, float))),
                default=None,
            ),
            "all_delta_exactly_zero": all(
                r["delta_db"] == REQUIRED_DELTA for r in comparable
            ),
            "j6_passed": len(rows) == 6 and all(r["verdict"] == "pass" for r in rows),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = run()

    failures = []
    if doc["summary"]["n_scenarios"] != 6:
        failures.append(f"M-J6: 场景数 {doc['summary']['n_scenarios']} != 6")
    if not doc["summary"]["all_delta_exactly_zero"]:
        failures.append("M-J6: 存在 delta_db != 0.0 的场景")

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        for r in doc["scenarios"]:
            d = r["delta_db"]
            ds = "n/a(两端同拒)" if d is None else f"{d:+.17g}"
            print(
                f"    {r['scenario']:<34} wp={r['wp_ms_alloc_snr_db']!s:<20} "
                f"ecz={r['ecz_dev_snr_db']!s:<20} delta={ds}  [{r['verdict']}]"
            )
        print(
            f"  M-J6: 逐位相同 {doc['summary']['n_bit_identical']}/{doc['summary']['n_scenarios']}，"
            f"两端一致拒绝 {doc['summary']['n_both_rejected']} 条，"
            f"max|delta|={doc['summary']['max_abs_delta_db']}"
        )

    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        if stored.get("scenarios") != doc["scenarios"]:
            print("FAIL 复算与落盘不一致")
            return 1
        if failures:
            for f in failures:
                print(f"FAIL {f}")
            return 1
        print("OK  M-J6 复算通过：6 个 legacy 场景 delta_db 全为 0.0，两端一致拒绝 "
              f"{doc['summary']['n_both_rejected']} 条")
        return 0

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())