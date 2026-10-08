"""FG-20 门 4：RL 决策序列逐项对比（FG-20 树 vs 基线 ``9115da2``）。

固定 seed 的 rollout，逐步记录 ``(action, channel, allocated)``。两侧的 RL 源码都取
**同一棵** ``/home/zsig/Documents/gits/wt/fg-rt``（``9ca3ec2``，本刀**未改** RL 侧
任何文件），唯一变量是物理仓的 ``src/``。

为什么跑这个就够了（不必跑 RL 全量分片）
----------------------------------------
本刀对物理仓的改动只有两类，两类都在 RL 决策路径**之外**：

1. ``physics/qot_checker.py`` 的 ``_active_channel_mask``（``None`` 成员防御）
   —— 该模块**只在**波形 QoT 路径被 import（``run_waveform=True``）。RL 的
   ``env.py`` 走的是**统计**侧 ``MultiBandISRSGN``，从不 import
   ``physics/qot_checker``。
2. 两个**新增测试文件**与三个**新增取证脚本** —— 仓库根与 ``tests/`` 下的文件，
   RL 运行时不会加载。

⇒ 决策路径与本刀改动**在结构上不相交**。但"结构上不相交"是**推理**不是**读数**，
所以仍要给出实测的逐项对比：若本刀真碰了可达路径，`n_mismatches` 会立刻非零。

用法::

    PYTHONPATH=<phys>/src:<rl>/src python scripts/fg20_decision_trace.py \
        --out results/fg20_rl_decision_trace.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
RL_SRC = "/home/zsig/Documents/gits/wt/fg-rt/src"


def main() -> int:
    """跑固定 seed 的 rollout，把逐步决策写成 JSON。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-requests", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topology", default="assets/pan_europe_multiband.json")
    args = ap.parse_args()

    # 证明 import 落在哪棵树（跨 WP 硬要求）。
    import multiband_optical_rl
    import numpy as np
    from multiband_optical_rl.benchmark import build_env
    from multiband_optical_rl.config import SystemConfig

    import network_nlo_eval

    config = SystemConfig()
    env = build_env(
        args.topology,
        num_requests=args.num_requests,
        seed=args.seed,
        config=config,
    )

    # 确定性"策略"：用固定 seed 的 RNG 在**合法动作**里随机选一个，并点名一条
    # 可行信道。RNG 的调用序固定 ⇒ 轨迹可复现。策略本身只读 env 的公开接口，
    # 不改 env。
    # 确定性"策略"：固定 seed 的 RNG 只在**合法动作**（env 给的 action_masks）里
    # 随机选一个。信道选择交给 env 自己的 first-fit（``channel=None`` 即阶段一
    # 语义），再从 info 里**读回**实际选中的信道 —— 不自己复刻 first-fit，避免
    # 策略口径与 env 悄悄分叉。
    rng = np.random.default_rng(args.seed)
    trace = []
    truncated = False
    n_alloc = 0
    while not truncated:
        masks = env.action_masks()
        legal = np.flatnonzero(masks)
        action = int(legal[rng.integers(0, legal.size)]) if legal.size else 0
        _, reward, terminated, truncated, info = env.step(action)
        allocated = bool(info["allocated"])
        n_alloc += int(allocated)
        trace.append(
            {
                "action": action,
                "channel": info.get("channel"),
                "allocated": allocated,
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )

    payload = {
        "policy": "seeded_random",
        "network_nlo_eval": network_nlo_eval.__file__,
        "multiband_optical_rl": multiband_optical_rl.__file__,
        "seed": args.seed,
        "num_requests": args.num_requests,
        "steps": len(trace),
        "n_allocated": n_alloc,
        "trace": trace,
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("phys:", payload["network_nlo_eval"])
    print("rl  :", payload["multiband_optical_rl"])
    print("steps:", len(trace), "n_allocated:", n_alloc)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
