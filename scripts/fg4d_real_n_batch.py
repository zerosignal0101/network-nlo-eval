"""FG-4d：真实 rollout 下的 ODE 节点数 N 与真实批宽 B。

本脚本只读地驱动 `multiband-optical-rl`（不修改它的任何文件），通过**运行时
monkeypatch 包装**两个入口来采集真实数字：

1. `MultiBandISRSGN._calc_span_noise_and_power` —— 每次物理内核调用记录
   `N = len(power_in_w)`（= ODE 节点数）与本次耗时。
2. `ScanValidator.verify_for_scan` —— 每次候选评估计数。env 侧的 `_try_first_fit`
   一次决策会连续调用它若干次（早退），因此"相邻两次 env.step 之间的
   verify_for_scan 调用数"就是**该次枚举决策的候选评估次数**，即批宽 B。

口径要点（FG-4c 遗留的纠正已落到代码里）：快速通道 `scan.py` 本身就是
`verify_allocation` 的逐位等价快通道，所以 step 成本 ≈ 内核调用次数 × 单次内核成本。
本脚本据此统计内核调用数，而不是再去引用 `verify_allocation` 的 33–38% 剖析。

用法::

    PYTHONPATH="$PWD/src:<RL>/src" python scripts/fg4d_real_n_batch.py \\
        --out results/fg4d_real_n_batch.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

TOPOLOGY = "/home/zsig/Documents/gits/multiband-optical-rl/assets/pan_europe_multiband.json"

#: 阶段四文档记录的真实工况负载（Erl）。见 RL 仓 docs/stage4-meta-notation.md §4。
REALISTIC_LOADS_ERL: tuple[float, ...] = (400.0, 1600.0, 6400.0)
EPISODE_LENGTH = 200


def install_probes() -> dict:
    """包装内核与快速通道入口，返回可变计数器字典。

    计数器刻意放在**模块级**而不是闭包上：被包装的方法是 bound method，闭包
    也能工作，但计数器需要在多个入口之间共享（内核计数与候选计数在 step 维度
    上对齐），模块级单例最不容易出错。
    """
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from multiband_optical_rl.scan import ScanValidator

    counters: dict = {
        "kernel_calls": 0,
        "kernel_nodes": [],  # 每次内核调用的 N
        "kernel_seconds": 0.0,
        "verify_calls": 0,
        "n_at_verify": [],  # 每次候选评估时的 N（取该次内最后一个内核调用）
    }

    original_kernel = MultiBandISRSGN._calc_span_noise_and_power
    original_verify = ScanValidator.verify_for_scan

    def probed_kernel(self, power_in_w, *args, **kwargs):
        counters["kernel_calls"] += 1
        counters["kernel_nodes"].append(int(np.shape(power_in_w)[0]))
        started = time.perf_counter()
        result = original_kernel(self, power_in_w, *args, **kwargs)
        counters["kernel_seconds"] += time.perf_counter() - started
        return result

    def probed_verify(self, *args, **kwargs):
        counters["verify_calls"] += 1
        before = counters["kernel_calls"]
        result = original_verify(self, *args, **kwargs)
        # 该次候选评估消耗的内核调用数与其 N；用于"每候选内核次数"统计。
        if counters["kernel_calls"] > before:
            counters["n_at_verify"].append(counters["kernel_nodes"][-1])
        return result

    MultiBandISRSGN._calc_span_noise_and_power = probed_kernel
    ScanValidator.verify_for_scan = probed_verify
    counters["_restore"] = lambda: (
        setattr(MultiBandISRSGN, "_calc_span_noise_and_power", original_kernel),
        setattr(ScanValidator, "verify_for_scan", original_verify),
    )
    return counters


def run_rollout(
    load_erl: float,
    seed: int,
    episode: int,
    counters: dict,
    max_steps: int | None = None,
    progress_every: int = 20,
) -> dict:
    """跑一次真实 rollout，统计 N / B / 内核调用。

    ``max_steps`` 是硬预算：本机与另一条线并行，单 episode 可能跑很久；预算用尽
    即停并**如实标注被截断**，已采集的分布仍可用（截断只影响尾部计数，不影响
    已完成决策的候选评估分布）。
    """
    from multiband_optical_rl.benchmark import arrival_interval_for_load, build_env
    from multiband_optical_rl.policies.ksp_ff import KSPFirstFitPolicy

    interval = arrival_interval_for_load(load_erl, 400.0)
    env = build_env(TOPOLOGY, num_requests=EPISODE_LENGTH, avg_arrival_interval=interval, seed=seed)
    policy = KSPFirstFitPolicy()
    policy.reset(episode)

    per_decision_b: list[int] = []
    per_step_kernel: list[int] = []
    per_step_wall: list[float] = []
    occupied_trace: list[int] = []
    blocked = 0
    arrivals = 0
    budget = max_steps if max_steps is not None else EPISODE_LENGTH

    for _ in range(budget):
        # 一次枚举决策 = 一次 env.step。决策内的候选评估数就是批宽 B。
        verify_before = counters["verify_calls"]
        kernel_before = counters["kernel_calls"]
        wall_started = time.perf_counter()

        masks = env.action_masks()
        action = policy.act(env.current_request, _simple_context(env))
        if action is not None and not (0 <= action < masks.size and masks[action]):
            action = None
        _, _, _, truncated, info = env.step(0 if action is None else action)

        per_decision_b.append(counters["verify_calls"] - verify_before)
        per_step_kernel.append(counters["kernel_calls"] - kernel_before)
        per_step_wall.append(time.perf_counter() - wall_started)
        arrivals += 1
        if not info["allocated"]:
            blocked += 1
        occupied_trace.append(_max_occupied(env))
        if arrivals % progress_every == 0:
            recent = per_decision_b[-progress_every:]
            print(
                f"    [{load_erl:.0f} Erl s{seed}] step {arrivals}/{budget} "
                f"B_recent_mean={statistics.mean(recent):.1f} "
                f"B_recent_max={max(recent)} busy={occupied_trace[-1]} "
                f"loadavg={os.getloadavg()[0]:.1f}",
                flush=True,
            )
        if truncated:
            break

    truncated_run = arrivals < budget
    return _summarize(
        load_erl=load_erl,
        seed=seed,
        episode=episode,
        env=env,
        counters=counters,
        per_decision_b=per_decision_b,
        per_step_kernel=per_step_kernel,
        per_step_wall=per_step_wall,
        occupied_trace=occupied_trace,
        arrivals=arrivals,
        blocked=blocked,
        truncated=truncated_run,
    )


def _simple_context(env):
    """`benchmark._context_of` 的等价物（该模块是私有的，这里重建只读上下文）。"""
    from multiband_optical_rl.benchmark import _context_of

    return _context_of(env)


def _max_occupied(env) -> int:
    """最忙那条链路上的占用信道数——灵活栅格下"连接数"的下界代理。"""
    best = 0
    for link_state in env.network_state.get_all_link_states().values():
        best = max(best, int(np.count_nonzero(link_state.occupied_channels)))
    return best


def _dist(values: list[int]) -> dict:
    """整数分布的均值/中位/分位/直方图。"""
    if not values:
        return {"count": 0}
    arr = np.asarray(values, dtype=np.int64)
    uniq, counts = np.unique(arr, return_counts=True)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "histogram": {int(u): int(c) for u, c in zip(uniq, counts)},
        "frac_zero": float(np.count_nonzero(arr == 0) / arr.size),
        "frac_ge_4": float(np.count_nonzero(arr >= 4) / arr.size),
        "frac_ge_16": float(np.count_nonzero(arr >= 16) / arr.size),
    }


def _summarize(**kw) -> dict:
    counters = kw["counters"]
    env = kw["env"]
    nodes = counters["kernel_nodes"]
    b = kw["per_decision_b"]
    return {
        "load_erlangs": kw["load_erl"],
        "seed": kw["seed"],
        "episode": kw["episode"],
        "arrivals": kw["arrivals"],
        "blocked": kw["blocked"],
        "blocking_rate": kw["blocked"] / kw["arrivals"] if kw["arrivals"] else 0.0,
        "truncated_by_budget": bool(kw.get("truncated", False)),
        "grid_num_channels": int(env.config.num_channels),
        "band_names": list(env.band_plan.band_names),
        "band_sizes": [
            int(env.band_plan.band_channel_indices(b).size) for b in range(env.band_plan.num_bands)
        ],
        "ode_nodes_N": _dist(nodes),
        "batch_width_B": _dist(b),
        "kernel_calls_per_step": _dist(kw["per_step_kernel"]),
        "occupied_channels_busiest_link": _dist(kw["occupied_trace"]),
        "kernel_seconds_total": counters["kernel_seconds"],
        "verify_calls_total": counters["verify_calls"],
        "kernel_calls_total": counters["kernel_calls"],
        "step_wall_seconds_total": sum(kw["per_step_wall"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/fg4d_real_n_batch.json")
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--loads", type=float, nargs="*", default=list(REALISTIC_LOADS_ERL))
    parser.add_argument(
        "--max-steps",
        type=int,
        default=EPISODE_LENGTH,
        help="单 episode 步数硬预算；用尽即截断并如实标注。",
    )
    args = parser.parse_args()

    counters = install_probes()
    episodes = []
    for load in args.loads:
        for seed in range(args.seeds):
            loadavg_before = os.getloadavg()
            started = time.perf_counter()
            episode = run_rollout(load, seed, seed, counters, max_steps=args.max_steps)
            episode["loadavg_before"] = list(loadavg_before)
            episode["loadavg_after"] = list(os.getloadavg())
            episode["wall_seconds"] = time.perf_counter() - started
            episodes.append(episode)
            print(
                f"load={load:>6.0f} Erl seed={seed} "
                f"N_med={episode['ode_nodes_N'].get('median')} "
                f"B_mean={episode['batch_width_B'].get('mean'):.3f} "
                f"B_med={episode['batch_width_B'].get('median')} "
                f"B_p95={episode['batch_width_B'].get('p95')} "
                f"block={episode['blocking_rate']:.3f} "
                f"trunc={episode['truncated_by_budget']} "
                f"loadavg={loadavg_before[0]:.1f}",
                flush=True,
            )

    payload = {
        "topology": TOPOLOGY,
        "episode_length": EPISODE_LENGTH,
        "max_steps": args.max_steps,
        "loads_erlangs": list(args.loads),
        "policy": "KSPFirstFitPolicy (规则基线，在线 first-fit 形态)",
        "note": (
            "N 与 B 均由运行时 monkeypatch 实测：N 取自物理内核入参长度，"
            "B 取自相邻两次 env.step 之间的 verify_for_scan 调用数。"
        ),
        "episodes": episodes,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
