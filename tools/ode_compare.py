#!/usr/bin/env python3
"""WP-O2 步骤 3：三树 SNR 对照（ode-dev / ecz-dev / wp/fg-uwb-fid）。

协议**完全沿用**编排方已建立的对照（`scripts_shared/cross_snr_compat.py` +
`compat_run.py` 的签名兼容层），只把它扩到三棵树并把差值落盘。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode_compare.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode_compare.py --check

产物：
* ``results/ode_dev_diff_vs_ecz.json``
* ``results/ode_dev_diff_vs_fu4.json``

两个 JSON 都覆盖全部 6 个 ``legacy_*`` 场景（``flex_*`` 无法在 ez 树上对照——
``ecz-dev`` 无 ``core/flexgrid``，见 PLAN §0）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = "/home/zsig/miniconda3/envs/rl/bin/python"
SCEN_DIR = "/home/zsig/.hermes/cache/scratch/xsnr/tests_legacy"
COMPAT_DIR = REPO / "scripts_shared"

TREES = {
    "ode-dev": REPO,
    "ecz-dev": Path("/home/zsig/Documents/gits/network-nlo-eval"),
    "wp/fg-uwb-fid": Path("/home/zsig/Documents/gits/wt/fg-uwb-fid"),
}
BASELINE = "ode-dev"

CHILD = r'''
import json, sys
tree = sys.argv[1]
sys.path.insert(0, tree + "/src")
sys.path.insert(0, sys.argv[2])

import compat_run as cr

out = {"tree": tree, "results": []}
for sc in cr.SCENARIOS:
    row = {"name": sc.name, "kind": sc.kind}
    try:
        r = cr.run_scenario(sc)
        row["snr_db"] = r.get("snr_db")
        row["accepted"] = bool(r.get("accepted"))
        row["init_params"] = r.get("init_params")
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        row["snr_db"] = None
    out["results"].append(row)
print(json.dumps(out))
'''


def probe(tree: Path) -> dict:
    p = subprocess.run(
        [PY, "-c", CHILD, str(tree), SCEN_DIR],
        capture_output=True,
        text=True,
        timeout=3600,
        cwd=str(COMPAT_DIR),
        env={"PYTHONPATH": str(COMPAT_DIR), "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    if p.returncode != 0:
        return {"tree": str(tree), "fatal": p.stderr[-3000:], "results": []}
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"tree": str(tree), "fatal": (p.stdout + p.stderr)[-3000:], "results": []}


def sha256_of(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run_all() -> dict[str, dict]:
    return {name: probe(path) for name, path in TREES.items()}


def build_diff(raw: dict[str, dict], other: str) -> dict:
    """构造 ode-dev 相对 ``other`` 的逐场景差表。"""
    base = {r["name"]: r for r in raw[BASELINE]["results"]}
    oth = {r["name"]: r for r in raw[other]["results"]}
    rows = []
    for name, b in base.items():
        o = oth.get(name)
        b_snr, o_snr = b.get("snr_db"), (o or {}).get("snr_db")
        delta = None
        if isinstance(b_snr, (int, float)) and isinstance(o_snr, (int, float)):
            delta = float(b_snr) - float(o_snr)
        rows.append(
            {
                "scenario": name,
                "kind": b.get("kind"),
                "ode_dev_snr_db": b_snr,
                "other_snr_db": o_snr,
                "delta_db": delta,
                "bit_identical": b_snr == o_snr,
                "accepted_ode_dev": b.get("accepted"),
                "accepted_other": (o or {}).get("accepted"),
                "ode_dev_error": b.get("error"),
                "other_error": (o or {}).get("error"),
            }
        )
    return {
        "wp": "O2",
        "baseline_tree": BASELINE,
        "baseline_path": str(TREES[BASELINE]),
        "compared_tree": other,
        "compared_path": str(TREES[other]),
        "protocol": "scripts_shared/cross_snr_compat.py + compat_run.py (unchanged)",
        "scenarios": rows,
        "summary": {
            "n_scenarios": len(rows),
            "n_bit_identical": sum(1 for r in rows if r["bit_identical"]),
            "n_rejected_both": sum(
                1 for r in rows if r["accepted_ode_dev"] is False and r["accepted_other"] is False
            ),
            "n_errors": sum(1 for r in rows if r["ode_dev_error"]),
            "max_abs_delta_db": max(
                (abs(r["delta_db"]) for r in rows if r["delta_db"] is not None), default=None
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    raw = run_all()
    for name, data in raw.items():
        if data.get("fatal"):
            print(f"  FATAL {name}: {data['fatal'][-500:]}", file=sys.stderr)

    outputs = {
        REPO / "results" / "ode_dev_diff_vs_ecz.json": build_diff(raw, "ecz-dev"),
        REPO / "results" / "ode_dev_diff_vs_fu4.json": build_diff(raw, "wp/fg-uwb-fid"),
    }

    if args.write:
        for path, doc in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            doc["raw"] = {
                name: {
                    "path": str(TREES[name]),
                    "results": data.get("results", []),
                    "fatal": data.get("fatal"),
                }
                for name, data in raw.items()
            }
            doc["evidence_sha256"] = {
                "compat_run.py": sha256_of(COMPAT_DIR / "compat_run.py"),
                "cross_snr_compat.py": sha256_of(COMPAT_DIR / "cross_snr_compat.py"),
            }
            path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"wrote {path.relative_to(REPO)}")
            for r in doc["scenarios"]:
                d = r["delta_db"]
                print(
                    f"    {r['scenario']:<34} ode={r['ode_dev_snr_db']!s:<20} "
                    f"other={r['other_snr_db']!s:<20} "
                    f"delta={'n/a' if d is None else f'{d:+.9f}'}  "
                    f"{'BIT-IDENTICAL' if r['bit_identical'] else 'DIFFERS'}"
                )

    if args.check:
        bad = 0
        for path, doc in outputs.items():
            if not path.is_file():
                print(f"FAIL missing {path.relative_to(REPO)}")
                bad += 1
                continue
            stored = json.loads(path.read_text(encoding="utf-8"))
            if stored["scenarios"] != doc["scenarios"]:
                print(f"FAIL {path.name}: 复算结果与落盘不一致")
                bad += 1
            elif stored["summary"]["n_scenarios"] != 6:
                print(f"FAIL {path.name}: 场景数 {stored['summary']['n_scenarios']} != 6")
                bad += 1
            else:
                print(f"OK   {path.name}: 6 场景一致，复算通过")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
