"""WP-MERGE-FG-FLEX 判据 V1/V2 的判决序列探针（4 seeds）。

口径与 RL 仓 `tools/fg11_rollout_trace.py` **逐字一致**（同一策略、同一 env、
同一逐步记录的决策三元组），只把 episodes 从 `(seed 0, seed 7)` 扩成
**任务书要求的 seeds 0..3**，并额外导出可用于 V2 的多槽分配读数。

- 场景：`assets/pan_europe_multiband.json`
- 口径：`num_requests=40`、`avg_arrival_interval=10.0`、`avg_holding_time=400.0`
- 策略：`KSPFirstFitPolicy`（确定性规则策略 ⇒ 序列可复现、可逐位比较）

用法（物理树由 ``PYTHONPATH`` 切换，因此同一个脚本可以分别量合并树与 FG 树）::

    cd <fg-rt worktree>
    PYTHONPATH=$RL/src:$TREE/src python <merge worktree>/evidence/merge/v1_rollout_probe.py \
        --out <json>

⚠️ 必须 `cd` 到 RL 仓根目录：`ASSET` 是相对路径。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from multiband_optical_rl.benchmark import build_env
from multiband_optical_rl.policies.ksp_ff import KSPFirstFitPolicy

ASSET = "assets/pan_europe_multiband.json"

#: 任务书 §4 V1 规定的 4 个 seed。
SEEDS = (0, 1, 2, 3)
EPISODE_KW = {"num_requests": 40, "avg_arrival_interval": 10.0, "avg_holding_time": 400.0}


def _provenance() -> dict[str, Any]:
    import multiband_optical_rl
    import network_nlo_eval

    def _commit(pkg_file: str) -> str:
        root = Path(pkg_file).resolve().parents[2]
        if not (root / ".git").exists():
            return f"unknown (no .git at {root})"
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    return {
        "rl_commit": _commit(multiband_optical_rl.__file__),
        "phys_commit": _commit(network_nlo_eval.__file__),
        "rl_tree": str(Path(multiband_optical_rl.__file__).resolve()),
        "phys_tree": str(Path(network_nlo_eval.__file__).resolve()),
    }


def _rollout(seed: int) -> list[dict[str, Any]]:
    """跑完一个 episode，逐步记录决策三元组 (action, channel, allocated)。

    与 `tools/fg11_rollout_trace.py::_rollout` 逐步同构，只把 seeds 换成本判据的
    0..3。多出的 `n_slots` 读数用于 V2 的多槽分配口径（若该 env 未暴露槽位，
    则该键缺省——不影响 V1 的逐位比较）。
    """
    env = build_env(ASSET, seed=seed, **EPISODE_KW)
    policy = KSPFirstFitPolicy()
    env.reset(seed=seed)
    steps: list[dict[str, Any]] = []
    step = 0
    while True:
        request = env.current_request
        assert request is not None
        action = policy.act(request, env._decision_context())
        _obs, reward, _terminated, truncated, info = env.step(-1 if action is None else action)
        row: dict[str, Any] = {
            "step": step,
            "action": -1 if action is None else int(action),
            "channel": None if info.get("channel") is None else int(info["channel"]),
            "allocated": len(env.network_state.get_all_allocated_services()),
            "reward": float(reward),
            "blocked": int(env._blocked),
            "blocked_qot": int(env._blocked_qot),
            "blocked_no_spectrum": int(env._blocked_no_spectrum),
        }
        n_slots = getattr(request, "n_slots", None)
        if n_slots is not None:
            row["request_n_slots"] = int(n_slots)
        first_slot = getattr(request, "first_slot", None)
        if first_slot is not None:
            row["request_first_slot"] = int(first_slot)
        steps.append(row)
        step += 1
        if truncated:
            break
    return steps


def capture() -> dict[str, Any]:
    return {
        "schema": "wp-merge-fg-flex/v1-rollout-probe/v1",
        "policy": "KSPFirstFitPolicy (deterministic rule policy)",
        "episode_kwargs": EPISODE_KW,
        "seeds": list(SEEDS),
        "episodes": [{"seed": s, "steps": _rollout(s)} for s in SEEDS],
        "provenance": _provenance(),
    }


def _summary(trace: dict[str, Any]) -> dict[str, Any]:
    """每 seed 一个可逐位比较的摘要（步骤数 + 决策序列的 sha256 + 累计奖励）。"""
    out: dict[str, Any] = {}
    for ep in trace["episodes"]:
        steps = ep["steps"]
        digest_src = json.dumps(
            [[s["action"], s["channel"], s["allocated"]] for s in steps], sort_keys=True
        )
        out[str(ep["seed"])] = {
            "n_steps": len(steps),
            "decision_sha256": hashlib.sha256(digest_src.encode()).hexdigest(),
            "total_reward": round(sum(s["reward"] for s in steps), 9),
            "final_allocated": steps[-1]["allocated"] if steps else 0,
            "blocked_total": sum(s["blocked"] + s["blocked_qot"] + s["blocked_no_spectrum"] for s in steps),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="写出轨迹 JSON")
    parser.add_argument("--compare", type=Path, help="与该轨迹逐 seed 比对")
    args = parser.parse_args(argv)

    trace = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print("rl   :", trace["provenance"]["rl_commit"], trace["provenance"]["rl_tree"])
    print("phys :", trace["provenance"]["phys_commit"], trace["provenance"]["phys_tree"])
    for seed, row in _summary(trace).items():
        print(f"  seed {seed}: steps={row['n_steps']} sha256={row['decision_sha256'][:16]} "
              f"reward={row['total_reward']} alloc={row['final_allocated']} blocked={row['blocked_total']}")

    if args.compare is not None:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        a, b = _summary(trace), _summary(other)
        mismatches = [s for s in a if a[s] != b.get(s)]
        print(f"\nCOMPARE vs {args.compare}: seeds={len(a)} mismatched={len(mismatches)}")
        for s in mismatches:
            print(f"  seed {s}:\n    new={a[s]}\n    old={b.get(s)}")
        return 1 if mismatches else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
