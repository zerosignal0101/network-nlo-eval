"""FG-18 门 4：RL 决策序列逐项对比（FG-18 树 vs 基线 ``1004458``）。

固定 seed 的 rollout，逐步 ``(action, channel, allocated)``。两侧的 RL 源码都取
同一个 ``wt/fg-rt-is``（本刀**没有**改 RL 侧任何文件），唯一变量就是物理仓的
ROADM 折叠——与 FG-17 的 A/B 口径一致。

用法::

    python scripts/fg18_decision_compare.py \\
        --new results/fg18_rl_decision_trace.json \\
        --baseline results/fg18_rl_decision_trace_baseline.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--baseline", required=True)
    args = ap.parse_args()

    new = json.loads(pathlib.Path(args.new).read_text())
    base = json.loads(pathlib.Path(args.baseline).read_text())

    same_meta = (
        new["policy"] == base["policy"]
        and new["seed"] == base["seed"]
        and new["num_requests"] == base["num_requests"]
    )
    nt, bt = new["trace"], base["trace"]
    mismatches = [
        {"step": i, "new": a, "baseline": b}
        for i, (a, b) in enumerate(zip(nt, bt, strict=False))
        if a != b
    ]
    payload = {
        "question": "固定 seed 的 RL 决策序列是否逐项相同",
        "new_phys": new["network_nlo_eval"],
        "baseline_phys": base["network_nlo_eval"],
        "rl_source": new["multiband_optical_rl"],
        "meta_equal": same_meta,
        "steps_new": len(nt),
        "steps_baseline": len(bt),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches[:20],
        "identical": same_meta and len(nt) == len(bt) and not mismatches,
    }
    out = REPO / "results/fg18_decision_compare.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("wrote", out)
    return 0 if payload["identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())