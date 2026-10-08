"""FG-18 门 1 归因：RL shard 2 的唯一失败 ``test_scan_validator_table_does_not_leak``
到底是不是本刀引起的。

裁决口径（跨 WP 执行规则第 9 条：**不靠传参意图，靠读回实际生效的值**）：

* **被测变量只有一个**：物理仓 import 到哪棵树。本刀 ``wp/fg-isrs``（改后）
  vs ``git archive 1004458`` 解出的只读副本（改前，``isrs_gn.py:528/596`` 仍是
  合并式）。
* **RL 源码两侧完全相同**：都取 ``wt/fg-rt-is``（本刀未改 RL 任何文件）。
* **交错 A/B**：同一进程内按 A/B/A/B 交替跑，避免把机器负载的时间漂移
  误当成"某档生效"。
* **重复多轮**：该断言依赖 ``gc.collect()`` 的时机，是典型的 flake 候选；
  单次读数不足以归因。

结论只有三种，如实报：两棵树都失败 => 与本刀无关（既有问题/flaky）；
只有改后失败 => 本刀引入的回归（必须修）；只有改前失败 => 基线自带。

用法::

    PYTHONPATH=<RL>/src:<phys>/src python scripts/fg18_shard2_failure_attribution.py --rounds 5
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
RL_ROOT = "/home/zsig/Documents/gits/wt/fg-rt-is"
NEW_PHYS = str(REPO / "src")
BASE_PHYS = "/tmp/fg17_baseline_1004458/src"
NODEID = "tests/test_scan_fastpath.py::test_scan_validator_table_does_not_leak"


def run_once(phys: str, python: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{RL_ROOT}/src:{phys}"
    t0 = time.time()
    proc = subprocess.run(
        [python, "-m", "pytest", NODEID, "-q", "-p", "no:randomly", "--no-header"],
        cwd=RL_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-2:]
    return {
        "phys": phys,
        "rc": proc.returncode,
        "passed": proc.returncode == 0,
        "seconds": round(time.time() - t0, 1),
        "tail": tail,
    }


def loadavg() -> str:
    p = pathlib.Path("/proc/loadavg")
    return p.read_text().split()[0] if p.exists() else "?"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    # 先读回两侧物理仓确实是哪棵树（纪律：不靠传参意图）。
    probe = {}
    for label, phys in (("new", NEW_PHYS), ("baseline", BASE_PHYS)):
        env = dict(os.environ)
        env["PYTHONPATH"] = f"{RL_ROOT}/src:{phys}"
        out = subprocess.run(
            [args.python, "-c", "import network_nlo_eval as m; print(m.__file__)"],
            capture_output=True, text=True, env=env, cwd=RL_ROOT,
        )
        probe[label] = {"declared": phys, "imported": out.stdout.strip()}

    rows = []
    for i in range(args.rounds):
        order = ("new", "baseline") if i % 2 == 0 else ("baseline", "new")
        for label in order:
            phys = NEW_PHYS if label == "new" else BASE_PHYS
            r = run_once(phys, args.python)
            r["round"] = i + 1
            r["variant"] = label
            rows.append(r)
            print(f"[round {i+1}] {label:8s} passed={r['passed']} {r['seconds']}s :: {r['tail'][-1:]}")

    new_pass = sum(1 for r in rows if r["variant"] == "new" and r["passed"])
    base_pass = sum(1 for r in rows if r["variant"] == "baseline" and r["passed"])
    new_n = sum(1 for r in rows if r["variant"] == "new")
    base_n = sum(1 for r in rows if r["variant"] == "baseline")

    if new_pass == base_pass:
        verdict = "两棵树读数相同 => 与本刀无关（既有问题或负载 flake）"
    elif new_pass < base_pass:
        verdict = "只有改后失败 => 本刀引入的回归，必须修"
    else:
        verdict = "只有改前失败 => 基线自带问题，非本刀引入"

    payload = {
        "question": "RL shard 2 的 test_scan_validator_table_does_not_leak 是否由本刀引起",
        "nodeid": NODEID,
        "import_probe": probe,
        "rounds": args.rounds,
        "loadavg_at_end": loadavg(),
        "new_pass": f"{new_pass}/{new_n}",
        "baseline_pass": f"{base_pass}/{base_n}",
        "verdict": verdict,
        "rows": rows,
        "mechanism_note": (
            "该断言依赖 gc.collect() 后 WeakKeyDictionary 的条目数归零，属 GC 时机敏感；"
            "断言路径不经过 isrs_gn 的两个 roadm_config 施加点（§0 已证其 roadm_config 恒 None）。"
        ),
    }
    out = REPO / "results/fg18_shard2_failure_attribution.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("verdict:", verdict)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())