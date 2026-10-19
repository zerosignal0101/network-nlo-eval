"""WP-MERGE-FG-FLEX 判据 V2 的**多槽专用探针**。

为什么不用 RL 仓的 `fg11_rollout_trace`：那条探针所在的 `wp/fg-rt` 分支上，
`multiband_optical_rl.config` 已经硬依赖 FG 侧的符号
（``from network_nlo_eval.models.isrs_gn import RAMAN_SPECTRUM_DEFAULT``），
而 ``flexgrid-dev`` 上没有这些符号 ⇒ 探针在 flexgrid 树上**根本 import 不了**：

    ImportError: cannot import name 'RAMAN_SPECTRUM_DEFAULT' from
    'network_nlo_eval.models.isrs_gn' (.../wt/flexgrid-dev/src/...)

任务书 §4 V2 允许「同一探针口径**或多槽专用探针**」，本文件即多槽专用探针，
且刻意**只用两棵树都有的符号**，因此同一脚本能分别量合并树与 `flexgrid-dev`：

* ``network.state.LinkState.allocate_slot_range`` / ``release_slot_range``
  （槽区间分配/释放的真源）
* ``rwa.allocators.first_fit_contiguous_windows``（连续槽窗口候选搜索）
* ``network.slots.connection_overlaps_slots`` / ``resolve_blocked_slots``（阻断语义）

记录每一步的决策三元组（``first_slot``、``n_slots``、分配后占用摘要），
序列逐位比较。物理树由 ``PYTHONPATH`` 切换。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

#: 混合槽宽请求序列：同一链路里 1/2/3/4 槽混排（V2 的核心口径）。
SLOT_DEMANDS = (1, 2, 3, 4, 2, 1, 4, 3, 2, 4, 1, 3)
#: 槽位总数与每信道槽数（12.5 GHz 槽 / 50 GHz 信道）。
NUM_SLOTS = 240
SLOTS_PER_CHANNEL = 4


def _provenance() -> dict[str, Any]:
    import network_nlo_eval

    root = Path(network_nlo_eval.__file__).resolve().parents[2]
    commit = (
        subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True).stdout.strip()
        if (root / ".git").exists() else f"unknown (no .git at {root})"
    )
    return {"phys_commit": commit, "phys_tree": str(Path(network_nlo_eval.__file__).resolve())}


def _make_state():
    """构造一条空链路状态（两侧签名一致）。"""
    from network_nlo_eval.network.state import LinkState

    n_ch = NUM_SLOTS // SLOTS_PER_CHANNEL
    return LinkState(
        link_key=(0, 1),
        length_m=80e3,
        occupied_channels=np.zeros(n_ch, dtype=bool),
        launch_power_profile_w=np.zeros(n_ch, dtype=np.float64),
        allocated_service_ids=np.full(n_ch, -1, dtype=np.int64),
        slots_per_channel=SLOTS_PER_CHANNEL,
    )


def capture() -> dict[str, Any]:
    from network_nlo_eval.network.slots import connection_overlaps_slots
    from network_nlo_eval.rwa.allocators import first_fit_contiguous_windows

    steps: list[dict[str, Any]] = []
    state = _make_state()
    service_id = 0
    free_mask = np.ones(NUM_SLOTS, dtype=bool)

    for demand in SLOT_DEMANDS:
        # 候选搜索：连续 n_slots 的空闲槽窗口（业务层的真源动作）
        windows = first_fit_contiguous_windows(free_mask, NUM_SLOTS, SLOTS_PER_CHANNEL)
        picked = None
        for start in windows:
            if start + demand <= NUM_SLOTS and free_mask[start:start + demand].all():
                picked = start
                break
        if picked is None:
            steps.append({"demand": demand, "first_slot": None, "service_id": None,
                          "occupied_slots": int(state.occupied_slots.sum())})
            continue

        state.allocate_slot_range(picked, demand, 1e-3, service_id)
        # 槽区间是真源；free_mask 与之同步（业务层自己的占用视图）
        free_mask[picked:picked + demand] = False

        anchor = picked // SLOTS_PER_CHANNEL
        steps.append({
            "demand": demand,
            "first_slot": int(picked),
            "anchor_channel": int(anchor),
            "service_id": int(service_id),
            "occupied_slots": int(state.occupied_slots.sum()),
            "allocated_slots": int((state.allocated_service_ids_slots >= 0).sum()),
            "occupied_channels": int(state.occupied_channels.sum()),
            "anchor_overlaps_itself": bool(connection_overlaps_slots(picked, demand, set())),
            "launch_power_w": float(state.launch_power_profile_w[anchor]),
        })
        service_id += 1

    # 逐个释放，验证释放路径也逐位一致
    for row in [r for r in steps if r["first_slot"] is not None]:
        state.release_slot_range(row["first_slot"], row["demand"])
        free_mask[row["first_slot"]:row["first_slot"] + row["demand"]] = True

    final = {
        "occupied_slots": int(state.occupied_slots.sum()),
        "allocated_slots": int((state.allocated_service_ids_slots >= 0).sum()),
        "occupied_channels": int(state.occupied_channels.sum()),
        "total_launch_power_w": float(state.launch_power_profile_w.sum()),
        "free_mask_true": int(free_mask.sum()),
    }
    digest_src = json.dumps(steps, sort_keys=True) + json.dumps(final, sort_keys=True)
    return {
        "schema": "wp-merge-fg-flex/v2-multislot-probe/v1",
        "slot_demands": list(SLOT_DEMANDS),
        "num_slots": NUM_SLOTS,
        "slots_per_channel": SLOTS_PER_CHANNEL,
        "steps": steps,
        "final": final,
        "decision_sha256": hashlib.sha256(digest_src.encode()).hexdigest(),
        "provenance": _provenance(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)

    trace = capture()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(trace, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print("phys :", trace["provenance"]["phys_commit"], trace["provenance"]["phys_tree"])
    print(f"decision_sha256 = {trace['decision_sha256']}")
    print(f"steps={len(trace['steps'])} final={trace['final']}")

    if args.compare is not None:
        other = json.loads(args.compare.read_text(encoding="utf-8"))
        same_steps = trace["steps"] == other["steps"]
        same_final = trace["final"] == other["final"]
        same_hash = trace["decision_sha256"] == other["decision_sha256"]
        print(f"\nCOMPARE vs {args.compare}: steps_equal={same_steps} "
              f"final_equal={same_final} sha256_equal={same_hash}")
        if not (same_steps and same_final):
            for a, b in zip(trace["steps"], other["steps"]):
                if a != b:
                    print(f"  new={a}\n  old={b}")
                    break
            return 1
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
