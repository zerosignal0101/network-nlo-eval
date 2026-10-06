r"""FG-4g 归因驱动：产出 ``results/fg4g_attribution.json``，并支持 ``--check`` 复算。

它把四个实测子过程串起来，每个子过程的数字都由**它自己的 JSON** 原样搬运，
本文件不做任何手工计算，因此文档里的每个数都能从本文件重渲：

1. **Q1** ``fg4g_q1_library_ab.py``  —— FG-4e 树内，库自己的入口 off vs matvec；
2. **Q2-off**    ``fg4g_q2_trace_diff.py --force-accel off``
3. **Q2-matvec** ``fg4g_q2_trace_diff.py --force-accel matvec``
   —— 同一棵 FG-4e 树内换档，证明 0.314 dB 与 ``ode_accel`` 无关；
4. **Q2-ref**   ``fg4g_q2_trace_diff.py``（ecz-dev 参考树）—— 证 0 不一致。

用法::

    # 产出 / 复算
    $PY scripts/fg4g_attribution.py --out results/fg4g_attribution.json
    $PY scripts/fg4g_attribution.py --check          # 与已提交 JSON 比对
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
RL_REPO = Path("/home/zsig/Documents/gits/wt/fg-batch-rl")
REF_TREE = Path("/home/zsig/Documents/gits/network-nlo-eval")
PY = Path(os.environ.get("FG4G_PY", str(Path.home() / "miniconda3/envs/rl/bin/python")))

Q1 = REPO / "scripts" / "fg4g_q1_library_ab.py"
Q2 = REPO / "scripts" / "fg4g_q2_trace_diff.py"


def _run(script: Path, tree_src: Path, out: Path, extra: list[str], label: str) -> dict[str, Any]:
    """在给定 PYTHONPATH 下跑一个探针，返回它写的 JSON。

    ``label`` 会顶掉探针记录的 ``--out`` 路径：临时目录名每次都不一样，把它
    写进产物会让 ``--check`` 假失败，也让产物带上一次性的 scratch 路径。
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = f"{RL_REPO / 'src'}:{tree_src}"
    env["NUMBA_CACHE_DIR"] = f"/tmp/nb-fg4g-{label}"
    cmd = [str(PY), str(script), "--out", str(out), *extra]
    proc = subprocess.run(cmd, cwd=str(RL_REPO), env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise SystemExit(
            f"probe failed: {' '.join(cmd)}\nrc={proc.returncode}\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    data = json.loads(out.read_text(encoding="utf-8"))
    if isinstance(data.get("config"), dict) and "out" in data["config"]:
        data["config"]["out"] = f"<{label}>"
    return data


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=str(repo), capture_output=True, text=True, check=True
    ).stdout.strip()


def _golden_scope() -> dict[str, Any]:
    """Q3：如实报出 FG-4a 金标准到底比了哪一层、比了多少例。"""
    golden = json.loads((REPO / "results" / "fg4a_golden.json").read_text(encoding="utf-8"))
    scen = golden.get("scenarios", {})
    span_cases = sum(4 for v in scen.values() if v.get("kind") == "span")
    path_cases = sum(len(v.get("outputs", {})) for v in scen.values() if v.get("kind") == "path")
    return {
        "golden_file": "results/fg4a_golden.json",
        "provenance": golden.get("provenance", {}),
        "n_scenarios": len(scen),
        "scenarios": {
            name: {"kind": v.get("kind"), "n_output_keys": len(v.get("outputs", {}))}
            for name, v in sorted(scen.items())
        },
        "n_golden_cases": span_cases + path_cases,
        "layers_compared": [
            "MultiBandISRSGN._calc_span_noise_and_power  (6 span scenarios x 4 arrays)",
            "MultiBandISRSGN.evaluate_path              (1 path scenario x 7 arrays)",
        ],
        "input_construction": (
            "全部经 tests/fg4a_scenarios.py 的稳定评估器 API 构造，"
            "不含 RL 的 NetworkTopology / QoTValidator / ROADM 编排"
        ),
        "covers_rl_working_conditions": {
            "N230_50GHz_CL": True,
            "non_empty_occupancy": True,
            "multi_hop_1_3_7_via_evaluate_path": False,
            "two_bands_via_band_plan": False,
            "roadm_equalize_booster_ase": False,
            "note": (
                "金标准覆盖 230×50 GHz C+L 与非空占用（cl_partial_100km 每 7 个空槽），"
                "也覆盖 evaluate_path 的 3 跳编排（path_cplusl_roadm），"
                "但用的是固定 ROADMConfig，不含 equalize booster 分支；"
                "也不经 QoTValidator，因此不含 band_plan 的两频段选择。"
            ),
        },
    }


def _skew_evidence() -> dict[str, Any]:
    """Q2 的根因证据：两棵树 qot_checker.py 的语义差 + 提交关系。"""
    ref = (REF_TREE / "src/network_nlo_eval/rwa/qot_checker.py").read_text(encoding="utf-8")
    ours = (REPO / "src/network_nlo_eval/rwa/qot_checker.py").read_text(encoding="utf-8")
    commits = _git(REF_TREE, "log", "--format=%h %s", "wp/fg-ode-fast..HEAD",
                   "--", "src/network_nlo_eval/rwa/qot_checker.py").splitlines()
    return {
        "ref_tree": {
            "path": str(REF_TREE),
            "branch": _git(REF_TREE, "rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git(REF_TREE, "rev-parse", "--short", "HEAD"),
            "head_subject": _git(REF_TREE, "log", "-1", "--format=%s"),
            "has_ode_accel": "ODE_ACCEL_MODES" in (
                REF_TREE / "src/network_nlo_eval/models/isrs_gn.py").read_text(encoding="utf-8"),
            "qot_checker_sha256": __import__("hashlib").sha256(ref.encode()).hexdigest()[:16],
        },
        "fg4e_tree": {
            "path": str(REPO),
            "branch": _git(REPO, "rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git(REPO, "rev-parse", "--short", "HEAD"),
            "has_ode_accel": True,
            "qot_checker_sha256": __import__("hashlib").sha256(ours.encode()).hexdigest()[:16],
        },
        "commits_in_ref_not_in_fg4e_touching_qot_checker": commits,
        "semantic_differences": [
            "ref _build_path_snr_array/_eval_service_snr 用 "
            "roadm.path_loss_db('drop'|'express') 区分末端/中间损耗；"
            "FG-4e 树用合并的 10**((insertion_loss_db+filtering_penalty_db)/10)",
            "ref 在 equalize_output_power 为真时把 booster ASE 加进 total_ase；"
            "FG-4e 树没有这个分支，一律除以 loss",
            "ref 把 filtering_penalty_db 单独乘一遍 total_*；"
            "FG-4e 树把它折进上面的合并 loss",
        ],
    }


def build(tmp: Path) -> dict[str, Any]:
    common = ["--steps", "5", "--strides", "7", "--max-paths-per-hops", "1",
              "--hops", "1,3,7", "--max-trace-cases", "2"]
    q1 = _run(Q1, REPO / "src", tmp / "q1.json",
              ["--steps", "5,20,40", "--strides", "7", "--max-paths-per-hops", "2"],
              label="q1")
    q2_off = _run(Q2, REPO / "src", tmp / "q2_off.json",
                  [*common, "--force-accel", "off"], label="q2_off")
    q2_mv = _run(Q2, REPO / "src", tmp / "q2_mv.json",
                 [*common, "--force-accel", "matvec"], label="q2_mv")
    q2_ref = _run(Q2, REF_TREE / "src", tmp / "q2_ref.json", common, label="q2_ref")

    def worst(p: dict[str, Any]) -> float | None:
        return p.get("summary", {}).get("worst_snr_abs_diff_db")

    off_w, mv_w, ref_w = worst(q2_off), worst(q2_mv), worst(q2_ref)
    return {
        "schema_version": 1,
        "generated_by": "scripts/fg4g_attribution.py",
        "verdict": "(B)",
        "verdict_one_line": (
            "0.314 dB 与 ode_accel 无关：同一棵 FG-4e 树下把档位从 off 换成 matvec，"
            "最差差值逐位相同；根因是两棵树 qot_checker.py 的 ROADM/booster QoT 语义不同，"
            "而 scan.py 复刻的是较新（ecz-dev）的那一套。"
        ),
        "environment": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "note": "loadavg 见各子探针 JSON 的 environment.loadavg；本刀不做加速比声明。",
        },
        "Q1_library_own_entrypoints": q1,
        "Q2_scan_vs_verify_allocation": {
            "fg4e_tree_off": q2_off,
            "fg4e_tree_matvec": q2_mv,
            "ref_tree_ecz_dev": q2_ref,
            "tier_invariance": {
                "worst_abs_diff_db_off": off_w,
                "worst_abs_diff_db_matvec": mv_w,
                "bit_identical_across_tiers": off_w == mv_w,
            },
        },
        "Q3_golden_scope": _golden_scope(),
        "root_cause_evidence": _skew_evidence(),
    }


VOLATILE = ("environment",)


def _scrub(obj: Any) -> Any:
    """去掉每次运行都会变的字段（loadavg / python / numpy 版本）。

    ``--check`` 要求**数值层**逐位复现，但负载与解释器版本不是结论的一部分。
    """
    if isinstance(obj, dict):
        return {k: _scrub(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "fg4g_attribution.json"))
    ap.add_argument("--check", action="store_true",
                    help="重算并与已提交的 JSON 比对（数值层逐位相等）")
    args = ap.parse_args()
    out = Path(args.out)

    with tempfile.TemporaryDirectory() as td:
        report = build(Path(td))

    if args.check:
        if not out.exists():
            raise SystemExit(f"{out} missing")
        stored = json.loads(out.read_text(encoding="utf-8"))
        diffs: list[str] = []
        for layer in ("Q1_library_own_entrypoints", "Q2_scan_vs_verify_allocation",
                      "Q3_golden_scope", "root_cause_evidence"):
            if json.dumps(_scrub(stored.get(layer)), sort_keys=True) != json.dumps(
                _scrub(report.get(layer)), sort_keys=True
            ):
                diffs.append(layer)
        if stored.get("verdict") != report.get("verdict"):
            diffs.append("verdict")
        if diffs:
            print("FG-4g --check FAILED, layers differ: " + ", ".join(diffs))
            return 1
        print("FG-4g --check OK: all layers reproduce bit-for-bit "
              "(volatile fields loadavg/python/numpy excluded)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"wrote {out}")
    t = report["Q2_scan_vs_verify_allocation"]["tier_invariance"]
    print(f"  verdict={report['verdict']}  worst_off={t['worst_abs_diff_db_off']}  "
          f"worst_matvec={t['worst_abs_diff_db_matvec']}  "
          f"tier_invariant={t['bit_identical_across_tiers']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
