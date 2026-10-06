r"""FG-4g Q1: 用**库自己的公开入口**比较 ``ode_accel="off"`` 与 ``"matvec"``。

本脚本只读引用 RL 仓（``multiband_optical_rl``）来搭出真实的 230×50 GHz C+L、
多跳 1/3/7、双频段、非空占用的工况，然后对**同一工况、同一输入**分别驱动
两个档位的库入口，逐元素比对。

被测入口（全部是库自己的 API，本脚本不自己实现任何物理）：

1. ``MultiBandISRSGN._calc_span_noise_and_power`` —— 逐跨段入口
2. ``MultiBandISRSGN.evaluate_path``              —— 端到端路径编排
3. ``QoTValidator.verify_allocation``             —— 上游闸门（RL 的参考档比对基准）

两个档位用**两个 evaluator 实例**，但共享同一个 ``NetworkTopology`` /
``NetworkState`` / ``SpectrumGrid``，因此比较的是同一件事。

用法::

    # 参考档（主检出，无 ode_accel）
    env -u PYTHONPATH PYTHONPATH=$RL:/home/zsig/Documents/gits/network-nlo-eval/src \
        NUMBA_CACHE_DIR=/tmp/nb-fg4g-b $PY scripts/fg4g_q1_library_ab.py --out /tmp/q1_ref.json
    # matvec 档（FG-4e 树）
    env -u PYTHONPATH PYTHONPATH=$RL:$PWD/src \
        NUMBA_CACHE_DIR=/tmp/nb-fg4g-a $PY scripts/fg4g_q1_library_ab.py --out /tmp/q1_mv.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ASSET = "assets/pan_europe_multiband.json"


def _pkg_paths() -> dict[str, str]:
    """打印实际 import 到的包路径，证明落在本 worktree 内。"""
    import network_nlo_eval

    out = {"network_nlo_eval": str(Path(network_nlo_eval.__file__).resolve())}
    try:
        import multiband_optical_rl

        out["multiband_optical_rl"] = str(Path(multiband_optical_rl.__file__).resolve())
    except ImportError:  # pragma: no cover
        out["multiband_optical_rl"] = "<not importable>"
    return out


def _dev(expected: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """两数组的 (不一致数, 最大绝对差, 最大相对差)。

    有限性不匹配单独计，因为那是语义变化而不是浮点重排。
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if expected.shape != actual.shape:
        return {
            "shape_mismatch": [list(expected.shape), list(actual.shape)],
            "n_differing": -1,
            "n_nonfinite_mismatch": -1,
            "max_abs": float("nan"),
            "max_rel": float("nan"),
        }
    n_differing = int(np.count_nonzero(expected != actual))
    n_nonfinite = int(np.count_nonzero(np.isfinite(expected) != np.isfinite(actual)))
    both = np.isfinite(expected) & np.isfinite(actual)
    if not both.any():
        return {
            "shape_mismatch": None,
            "n_differing": n_differing,
            "n_nonfinite_mismatch": n_nonfinite,
            "max_abs": 0.0,
            "max_rel": 0.0,
        }
    exp_f = expected[both]
    act_f = actual[both]
    diff = np.abs(exp_f - act_f)
    denom = np.abs(exp_f)
    rel = np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)
    return {
        "shape_mismatch": None,
        "n_differing": n_differing,
        "n_nonfinite_mismatch": n_nonfinite,
        "max_abs": float(diff.max()),
        "max_rel": float(rel.max()),
    }


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐例的 deviation 行汇总成一张表。"""
    ok = [r for r in rows if r.get("shape_mismatch") is None]
    worst_abs = max(ok, key=lambda r: r["max_abs"], default=None)
    worst_rel = max(ok, key=lambda r: r["max_rel"], default=None)
    n_bad = sum(1 for r in rows if r.get("shape_mismatch") is not None or r["n_differing"] > 0)
    return {
        "n_cases": len(rows),
        "n_cases_with_any_difference": n_bad,
        "total_elements": sum(r.get("n_elements", 0) for r in rows),
        "n_nonfinite_mismatch_total": sum(r.get("n_nonfinite_mismatch", 0) for r in rows),
        "worst_abs_db": worst_abs["max_abs"] if worst_abs else None,
        "worst_abs_where": worst_abs["where"] if worst_abs else None,
        "worst_rel": worst_rel["max_rel"] if worst_rel else None,
        "worst_rel_where": worst_rel["where"] if worst_rel else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="5,20", help="env 推进步数（占用档位）")
    ap.add_argument("--strides", default="7", help="信道采样步长")
    ap.add_argument("--max-paths-per-hops", type=int, default=2)
    args = ap.parse_args()

    # --- 搭 RL 工况（只读引用）-----------------------------------------
    from multiband_optical_rl.benchmark import build_env
    from multiband_optical_rl.request import to_service_request

    from network_nlo_eval.core.types import NodeID
    from network_nlo_eval.models.isrs_gn import ODE_ACCEL_DEFAULT, MultiBandISRSGN
    from network_nlo_eval.rwa.qot_checker import QoTValidator

    env = build_env(ASSET, num_requests=60, avg_arrival_interval=10.0, avg_holding_time=400.0, seed=0)
    grid = env.spectrum_grid
    topology = env.topology

    # 两个 evaluator：同 grid、同默认光纤、同 Rs，只有 ode_accel 不同。
    mv_evaluator = MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=topology.get_default_fiber_config(),
        symbol_rate_hz=env.evaluator.symbol_rate_hz,
    )
    off_evaluator = MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=topology.get_default_fiber_config(),
        symbol_rate_hz=env.evaluator.symbol_rate_hz,
        ode_accel="off",
    )
    mv_validator = QoTValidator(mv_evaluator, topology, grid)
    off_validator = QoTValidator(off_evaluator, topology, grid)

    report: dict[str, Any] = {
        "schema_version": 1,
        "what": "FG-4g Q1 — library-own-entry A/B: ode_accel=off vs matvec, identical inputs",
        "package_paths": _pkg_paths(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "loadavg": os.getloadavg(),
        },
        "config": {
            "ode_accel_default": ODE_ACCEL_DEFAULT,
            "num_channels": int(grid.num_channels),
            "steps": args.steps,
            "strides": args.strides,
            "max_paths_per_hops": args.max_paths_per_hops,
        },
    }

    # ==================================================================
    # 入口 1/3：QoTValidator.verify_allocation（上游闸门，最贴近 RL 工况）
    # ==================================================================
    def purge() -> None:
        for ls in env.network_state.get_all_link_states().values():
            ls.invalidate_noise_cache()

    def paths_of_hop_counts(wanted, limit):
        collected = {h: [] for h in wanted}
        for source, destination in sorted(env.path_cache.get_all_ksp_pairs()):
            for path in env.path_cache.get_paths(source, destination):
                hops = len(path) - 1
                if hops in collected and len(collected[hops]) < limit:
                    collected[hops].append([int(n) for n in path])
            if all(len(v) >= limit for v in collected.values()):
                break
        return {h: v for h, v in collected.items() if v}

    def free_channels(path, band):
        idx = env.band_plan.band_channel_indices(band)
        for pos in range(len(path) - 1):
            occ = env.network_state.get_link_state(path[pos], path[pos + 1]).occupied_channels
            idx = idx[~occ[idx]]
        return [int(c) for c in idx]

    alloc_rows: list[dict[str, Any]] = []
    n_verdict_mismatch = 0
    n_infeasible = 0
    for steps in [int(s) for s in args.steps.split(",")]:
        env.reset(seed=0)
        for _ in range(steps):
            env.step(int(env.action_masks().argmax()))
        request = to_service_request(env.current_request)
        hop_counts = paths_of_hop_counts((1, 2, 3, 7), args.max_paths_per_hops)
        for hops, paths in hop_counts.items():
            for path in paths:
                for band in (0, 1):
                    for stride in [int(s) for s in args.strides.split(",")]:
                        for channel in free_channels(path, band)[::stride]:
                            purge()
                            off_ok, off_arr = off_validator.verify_allocation(
                                service_request=request, path=list(path), channel_idx=channel,
                                current_network_state=env.network_state,
                            )
                            purge()
                            mv_ok, mv_arr = mv_validator.verify_allocation(
                                service_request=request, path=list(path), channel_idx=channel,
                                current_network_state=env.network_state,
                            )
                            where = f"steps={steps},hops={hops},band={band},stride={stride},ch={channel}"
                            if bool(off_ok) != bool(mv_ok):
                                n_verdict_mismatch += 1
                                alloc_rows.append({
                                    "where": where, "kind": "verdict",
                                    "off_ok": bool(off_ok), "mv_ok": bool(mv_ok),
                                    "n_differing": 1, "n_elements": 1,
                                    "n_nonfinite_mismatch": 0,
                                    "max_abs": float("inf"), "max_rel": float("inf"),
                                    "shape_mismatch": None,
                                })
                                continue
                            if not off_ok:
                                n_infeasible += 1
                                continue
                            dev = _dev(np.asarray(off_arr), np.asarray(mv_arr))
                            dev.update({"where": where, "n_elements": int(np.size(off_arr))})
                            alloc_rows.append(dev)

    report["verify_allocation_snr_db"] = _agg(alloc_rows)
    report["verify_allocation_snr_db"]["n_verdict_mismatch"] = n_verdict_mismatch
    report["verify_allocation_snr_db"]["n_infeasible_skipped"] = n_infeasible
    report["verify_allocation_snr_db"]["note"] = (
        "SNR 数组本身就是 dB；max_abs 因此是 dB 绝对差。"
        "verdict 不一致单独计数在 n_cases_with_any_difference 里。"
    )

    # ==================================================================
    # 入口 1：_calc_span_noise_and_power（逐跨段），在真实链路功率谱上
    # ==================================================================
    span_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []
    from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath

    link_keys = sorted(env.network_state.get_all_link_states().keys())[:6]
    for (u, v) in link_keys:
        fiber = topology.get_fiber_config(u, v)
        edfa = topology.get_link_edfa_config(u, v)
        power = env.network_state.get_link_state(u, v).launch_power_profile_w
        total_m = fiber.length_km * 1000.0
        span_m = fiber.max_span_length_km * 1000.0
        n_full = int(total_m // span_m)
        rem_m = total_m % span_m
        off_ev = off_evaluator.for_fiber(fiber)
        mv_ev = mv_evaluator.for_fiber(fiber)
        lengths = [(span_m, f"full")] + ([(rem_m, "remainder")] if rem_m > 1.0 else [])
        for length_m, tag in lengths:
            off_out = off_ev._calc_span_noise_and_power(power, length_m, edfa, None, 1)
            mv_out = mv_ev._calc_span_noise_and_power(power, length_m, edfa, None, 1)
            for key, a, b in zip(("p_out_w", "sigma2_spm_w", "sigma2_xpm_w", "sigma2_ase_w"), off_out, mv_out):
                dev = _dev(a, b)
                dev.update({"where": f"link={(u, v)},L={length_m:.0f}m,{tag},{key}",
                            "n_elements": int(np.size(a))})
                span_rows.append(dev)

        # 入口 2：evaluate_path，单跳 1 span
        opt_path = MultiSpanOpticalPath(
            fiber_configs=[fiber], edfa_configs=[edfa], roadm_configs=[None],
            link_keys=[(u, v)], node_path=[u, v],
        )
        off_path = off_evaluator.evaluate_path(opt_path, power)
        mv_path = mv_evaluator.evaluate_path(opt_path, power)
        for key in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
            dev = _dev(getattr(off_path, key), getattr(mv_path, key))
            dev.update({"where": f"link={(u, v)},evaluate_path,{key}",
                        "n_elements": int(np.size(getattr(off_path, key)))})
            eval_rows.append(dev)

    report["_calc_span_noise_and_power"] = _agg(span_rows)
    report["evaluate_path"] = _agg(eval_rows)
    report["evaluate_path"]["note"] = "snr_db 的 max_abs/max_rel 单位为 dB；其余分量单位为 W。"

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}")
    for layer in ("verify_allocation_snr_db", "_calc_span_noise_and_power", "evaluate_path"):
        a = report[layer]
        print(f"  {layer}: differing_cases={a['n_cases_with_any_difference']}/{a['n_cases']} "
              f"worst_abs={a['worst_abs_db']} worst_rel={a['worst_rel']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
