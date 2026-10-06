#!/usr/bin/env python
"""FG-5 整合证据：ΔSNR 表（整合前 / 整合后）+ 两侧树的身份。

**这个脚本回答什么。** 物理仓两条线合并本身**不应该**改任何一个数——两侧的改动都是
增量式的（速度线加批量入口与默认档，保真线加谱档位与并集装配），legacy 档的数值
路径两边都没动。所以"整合后 ΔSNR 全为 0"不是一句保证，而是必须能被复算的结论。

**为什么不自己写场景。** 直接复用**两棵树各自提交在库里的**
``scripts/fu4_capture_golden.py``：合并前用 ``a233002`` 那棵自己的脚本，
合并后用本树自己的脚本。这样 ΔSNR 的两侧都由各自那刀的权威工具产出，而不是由本
刀新写的一段近似代码产出——近似代码会把"整合没改物理"与"我的近似没改物理"混在
一起，分不开。

选 FU-4 场景而不是 FG-4a 场景，是因为 FU-4 走 ``rwa/qot_checker.py``——正是 RL 侧
依赖、也是历史上两仓分叉的那条路径；FG-4a 走 ``evaluate_path``，根本不调这个文件。

用法::

    # 还原"整合前"那棵树（用 git archive，不动共享仓库的 git 元数据）
    git -C /home/zsig/Documents/gits/network-nlo-eval archive a233002 \
        | tar -x -C /tmp/fg5-before-phys

    # 写出产物
    PYTHONPATH=$PWD/src python scripts/fg5_integration_evidence.py \
        --before-repo /tmp/fg5-before-phys --out results/fg5_integration.json

    # 重渲并比对
    PYTHONPATH=$PWD/src python scripts/fg5_integration_evidence.py \
        --before-repo /tmp/fg5-before-phys --check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BEFORE_REPO = Path("/tmp/fg5-before-phys")
DEFAULT_OUT = REPO_ROOT / "results" / "fg5_integration.json"

#: 两侧树的身份。写进产物，读者不必猜"整合前"到底是哪棵树。
BEFORE_COMMIT = "a233002"
AFTER_DESCRIPTION = "wp/fg-integrate = merge(a233002, a06d34c)"

_JSON_MARKER = "@@FG5_SNR@@"


def _capture(repo: Path, python: str) -> dict[str, Any]:
    """用该树**自己**的 FU-4 捕获脚本跑一遍，返回 golden 字典。"""
    script = repo / "scripts" / "fu4_capture_golden.py"
    if not script.exists():
        raise SystemExit(f"{repo} 里没有 scripts/fu4_capture_golden.py，无法作为对照树")
    env = {"PYTHONPATH": str(repo / "src")}
    import os

    child_env = dict(os.environ)
    child_env.update(env)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "golden.json"
        proc = subprocess.run(
            [python, str(script), "--write", "--out", str(out)],
            cwd=str(repo), env=child_env, capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise SystemExit(f"{repo} 的 FU-4 捕获失败：\n{proc.stderr[-3000:]}")
        for line in proc.stdout.splitlines():
            if line.startswith(_JSON_MARKER):
                return json.loads(line[len(_JSON_MARKER):])
        return json.loads(out.read_text(encoding="utf-8"))


def _git_describe(repo: Path) -> str:
    proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() or "(not a git repo)"


def build_table(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """逐场景给出整合前/后的 SNR 与 ΔSNR。"""
    before_rows = {row["name"]: row for row in before["scenarios"]}
    after_rows = {row["name"]: row for row in after["scenarios"]}
    names = [row["name"] for row in after["scenarios"]]

    table: list[dict[str, Any]] = []
    for name in names:
        b = before_rows.get(name)
        a = after_rows[name]
        b_snr = None if b is None or b["snr_db"] is None else float(b["snr_db"])
        a_snr = None if a["snr_db"] is None else float(a["snr_db"])
        if b is None:
            delta = None
            note = "before tree lacks this scenario"
        elif b_snr is None and a_snr is None:
            delta = None
            note = "rejected on both sides (无 SNR 可比；接受性仍单独比对)"
        elif b_snr is None or a_snr is None:
            delta = None
            note = "仅单侧被拒（接受性发生翻转）"
        else:
            delta = a_snr - b_snr
            note = ""
        table.append(
            {
                "scenario": name,
                "snr_db_before": b_snr,
                "snr_db_after": a_snr,
                "delta_db": delta,
                "accepted_before": None if b is None else b["accepted"],
                "accepted_after": a["accepted"],
                "note": note,
            }
        )
    return table


def build_report(before_repo: Path, python: str) -> dict[str, Any]:
    before = _capture(before_repo, python)
    after = _capture(REPO_ROOT, python)
    table = build_table(before, after)
    deltas = [row["delta_db"] for row in table if row["delta_db"] is not None]
    return {
        "schema": "fg5_integration/v1",
        "description": (
            "FG-5 物理仓整合证据：两条线合并前后的 ΔSNR。合并是增量式的，"
            "因此 legacy/并集路径的 SNR 应当逐位不变——本表就是那句话的复算。"
        ),
        "scenario_source": "各树自己的 scripts/fu4_capture_golden.py（FU-4 走 rwa/qot_checker.py）",
        "before_tree": {"repo": str(before_repo), "commit": _git_describe(before_repo),
                        "expected_commit": BEFORE_COMMIT},
        "after_tree": {"repo": str(REPO_ROOT), "commit": _git_describe(REPO_ROOT),
                       "description": AFTER_DESCRIPTION},
        "delta_snr_table": table,
        "summary": {
            "scenarios": len(table),
            "compared": len(deltas),
            "max_abs_delta_db": max((abs(d) for d in deltas), default=None),
            "all_bit_identical": all(d == 0.0 for d in deltas),
            "acceptance_changed": [
                row["scenario"] for row in table
                if row["accepted_before"] != row["accepted_after"]
            ],
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--before-repo", type=Path, default=DEFAULT_BEFORE_REPO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="重渲并与已提交产物比对")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    import network_nlo_eval

    package_dir = str(Path(network_nlo_eval.__file__).resolve().parent)
    if not package_dir.startswith(str(REPO_ROOT)):
        print(f"IMPORT_FAIL: imported {package_dir}, not under {REPO_ROOT}."
              " 设置 PYTHONPATH=$PWD/src。", file=sys.stderr)
        return 3

    report = build_report(args.before_repo, args.python)
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    if args.check:
        if not args.out.exists():
            print(f"CHECK_FAIL: 产物不存在 {args.out}", file=sys.stderr)
            return 1
        if args.out.read_text(encoding="utf-8") != payload:
            print(f"CHECK_FAIL: 重渲结果与 {args.out} 不一致", file=sys.stderr)
            return 1
        print(f"OK: {args.out} 与重渲结果逐字节一致")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    print(f"已写出 {args.out}")
    summary = report["summary"]
    print(f"scenarios={summary['scenarios']} compared={summary['compared']} "
          f"max|ΔSNR|={summary['max_abs_delta_db']} all_bit_identical={summary['all_bit_identical']}")
    for row in report["delta_snr_table"]:
        print(f"  {row['scenario']:34s} before={row['snr_db_before']} "
              f"after={row['snr_db_after']} Δ={row['delta_db']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())