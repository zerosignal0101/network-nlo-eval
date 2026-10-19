"""FG-18 §0 取证：``roadm_config`` 参数在两个施加点上**是否真被非 None 走到**。

做法不是"看调用点"，而是**运行时挂钩**：用 ``fg18_probe_plugin`` 把两个方法的
``roadm_config`` 形参包一层计数器，逐个测试文件跑，统计 (调用次数, 非 None 次数)。

诚实边界：本探针覆盖本仓 ``tests/`` 里"最可能碰到 roadm_config"的测试面。
它**不能**证明外部调用方（RL 仓 / 用户代码）永不传非 None。因此结论分两档：
* 本仓读到 0 次非 None ==> "**本仓不可达**"，不等于"全宇宙不可达"。
* 探针**不**改变被测行为（只包一层转发）。

用法::

    PYTHONPATH=<repo>/src python scripts/fg18_reachability_probe.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
PLUGIN = "fg18_probe_plugin"

#: 最可能碰到 ``roadm_config`` 的测试面：batch 入口、归因、场景、端到端路径、拓扑。
SUITES = [
    "tests/test_fg4e_cpu_batching.py",
    "tests/test_fg4g_matvec_attribution.py",
    "tests/test_fg4b_nli_vectorization.py",
    "tests/test_fg1_physics_parity.py",
    "tests/test_fu4_assembly_union.py",
    "tests/test_topology_physical_integration.py",
    "tests/test_fg2_flexible_grid.py",
    "tests/test_fg13_evaluate_path_roadm.py",
]


def main() -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{REPO / 'src'}:{REPO / 'scripts'}"
    counter = pathlib.Path("/tmp/fg18_roadm_probe_counts.json")

    per_suite: dict[str, dict] = {}
    for suite in SUITES:
        counter.unlink(missing_ok=True)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", suite, "-q", "-p", PLUGIN, "--no-header"],
            cwd=REPO,
            capture_output=True,
            text=True,
            env=env,
        )
        entry: dict = {
            "returncode": proc.returncode,
            "counts": json.loads(counter.read_text()) if counter.exists() else None,
            "tail": proc.stdout.strip().splitlines()[-3:],
        }
        per_suite[suite] = entry
        print(f"{suite}: rc={proc.returncode} {entry['counts']}")

    totals = {"calls": 0, "roadm_not_none": 0}
    for e in per_suite.values():
        c = e.get("counts")
        if not c:
            continue
        for v in c.values():
            totals["calls"] += v["calls"]
            totals["roadm_not_none"] += v["roadm_not_none"]

    # 自检：探针必须**真的**看到过调用，否则"0 次非 None"是探针失效的假阴性，
    # 不能读成"不可达"。早先版本只在非 None 时落盘，整套件读出 0/0 就是这么来的。
    probe_live = totals["calls"] > 0
    if not probe_live:
        print(
            "!! PROBE INVALID: 两个方法合计调用 0 次 —— 挂钩没生效，"
            "本结果**不可**当作可达性证据。",
            file=sys.stderr,
        )

    payload = {
        "question": "roadm_config 是否在 isrs_gn 的两个施加点被非 None 走到",
        "python": sys.executable,
        "repo_pkg": __import__("network_nlo_eval").__file__,
        "per_suite": per_suite,
        "totals": totals,
        "probe_live": probe_live,
        "verdict": (
            "本仓选定测试面内 roadm_config 恒为 None（不可达）"
            if probe_live and totals["roadm_not_none"] == 0
            else "已观察到非 None 调用" if totals["roadm_not_none"] > 0 else "探针失效"
        ),
        "scope_caveat": "只覆盖本仓 tests/ 选定子集；不能证明外部调用方不可达",
    }
    out = REPO / "results/fg18_reachability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(totals, indent=2))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
