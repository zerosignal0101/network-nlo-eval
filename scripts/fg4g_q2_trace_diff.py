r"""FG-4g Q2: 在同一档位下并列 ``verify_allocation`` 与 ``verify_for_scan`` 的编排。

Q1 已经证明库自己的入口在 off/matvec 两档间差 <= ~1e-15。本脚本负责另一件事：
**在 RL 的真实工况里复现 433/800 的不一致，并定位第一个不同的量。**

做法：把 ``MultiBandISRSGN._calc_span_noise_and_power`` 包一层记录器，记录每次
调用的 (evaluator 身份, 入纤谱, 跨段长度, n_effective_spans, NF, 四个输出)。
对同一个候选先跑上游 ``verify_allocation``（记为 A 相），再跑 ``verify_for_scan``
（记为 B 相），然后逐次调用并列比较：

* **输入就不同**  => 编排层分歧，(B)：scan.py 的复刻假设在���档位下不成立；
* **输入相同、输出不同** => 内核对同一输入不确定，(A) 方向的候选。

用法::

    cd /home/zsig/Documents/gits/wt/fg-batch-rl
    env -u PYTHONPATH PYTHONPATH=$PWD/src:/home/zsig/Documents/gits/wt/fg-ode-fast/src \
        NUMBA_CACHE_DIR=/tmp/nb-fg4g-a $PY \
        /home/zsig/Documents/gits/wt/fg-ode-fast/scripts/fg4g_q2_trace_diff.py --out /tmp/q2.json
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

_TRACE: list[dict[str, Any]] = []
_PHASE: dict[str, str | None] = {"name": None}
_ORIG = None


def _install_recorder() -> None:
    global _ORIG
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN

    _ORIG = MultiBandISRSGN._calc_span_noise_and_power

    def wrapper(self, power_in_w, span_length_m, edfa_config, roadm_config=None, n_effective_spans=1):
        out = _ORIG(self, power_in_w, span_length_m, edfa_config, roadm_config, n_effective_spans)
        if _PHASE["name"] is not None:
            nf = None if edfa_config is None else edfa_config.noise_figure_db
            nf_arr = None if nf is None else np.asarray(nf, dtype=np.float64)
            _TRACE.append(
                {
                    "phase": _PHASE["name"],
                    "evaluator_id": id(self),
                    "ode_accel": getattr(self, "ode_accel", "<absent>"),
                    "span_length_m": float(span_length_m),
                    "n_effective_spans": int(n_effective_spans),
                    "power": np.array(power_in_w, dtype=np.float64, copy=True),
                    "nf": None if nf_arr is None else nf_arr.copy(),
                    "out": [np.array(o, dtype=np.float64, copy=True) for o in out],
                }
            )
        return out

    MultiBandISRSGN._calc_span_noise_and_power = wrapper


def _uninstall_recorder() -> None:
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN

    MultiBandISRSGN._calc_span_noise_and_power = _ORIG


def _describe(trace: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n_kernel_calls": len(trace),
        "calls": [
            {
                "i": i,
                "span_length_m": t["span_length_m"],
                "n_effective_spans": t["n_effective_spans"],
                "nf": None if t["nf"] is None else float(np.asarray(t["nf"]).ravel()[0]),
                "power_sum": float(np.sum(t["power"])),
                "power_nonzero": int(np.count_nonzero(t["power"])),
                "spm_sum": float(np.sum(t["out"][1])),
                "xpm_sum": float(np.sum(t["out"][2])),
                "ase_sum": float(np.sum(t["out"][3])),
            }
            for i, t in enumerate(trace)
        ],
    }


def _first_difference(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> dict[str, Any]:
    """并列两相的逐次调用，找第一个不同的量。"""
    if len(a) != len(b):
        return {
            "kind": "call_count_mismatch",
            "n_calls_A": len(a),
            "n_calls_B": len(b),
        }
    for i, (ta, tb) in enumerate(zip(a, b)):
        # 1) 调用参数
        if ta["span_length_m"] != tb["span_length_m"]:
            return {"kind": "span_length_m", "i": i, "A": ta["span_length_m"], "B": tb["span_length_m"]}
        if ta["n_effective_spans"] != tb["n_effective_spans"]:
            return {"kind": "n_effective_spans", "i": i, "A": ta["n_effective_spans"],
                    "B": tb["n_effective_spans"]}
        nf_a = None if ta["nf"] is None else ta["nf"]
        nf_b = None if tb["nf"] is None else tb["nf"]
        if (nf_a is None) != (nf_b is None) or (
            nf_a is not None and not np.array_equal(nf_a, nf_b)
        ):
            return {"kind": "nf", "i": i}
        if not np.array_equal(ta["power"], tb["power"]):
            d = np.abs(ta["power"] - tb["power"])
            k = int(np.argmax(d))
            return {"kind": "power_profile", "i": i, "max_abs": float(d.max()), "argmax": k,
                    "A": float(ta["power"][k]), "B": float(tb["power"][k])}
        # 2) 输入全同，看输出
        for key, oa, ob in zip(("p_out_w", "sigma2_spm_w", "sigma2_xpm_w", "sigma2_ase_w"),
                               ta["out"], tb["out"]):
            if not np.array_equal(oa, ob):
                d = np.abs(oa - ob)
                fin = np.isfinite(oa) & np.isfinite(ob)
                rel = np.divide(d[fin], np.abs(oa[fin]), out=np.zeros_like(d[fin]),
                                where=np.abs(oa[fin]) > 0) if fin.any() else d
                k = int(np.argmax(rel)) if fin.any() else 0
                return {
                    "kind": "kernel_output", "i": i, "output": key,
                    "n_differing": int(np.count_nonzero(oa != ob)),
                    "max_abs": float(d.max()) if d.size else 0.0,
                    "max_rel": float(rel.max()) if rel.size else 0.0,
                    "argmax_rel": k,
                    "A_at": float(oa[k]) if oa.size else None,
                    "B_at": float(ob[k]) if ob.size else None,
                }
    return {"kind": "identical", "n_calls": len(a)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="5")
    ap.add_argument("--strides", default="7")
    ap.add_argument("--max-paths-per-hops", type=int, default=1)
    ap.add_argument("--hops", default="1,3,7")
    ap.add_argument("--max-trace-cases", type=int, default=3)
    ap.add_argument(
        "--force-accel",
        default=None,
        choices=["off", "cached_g", "matvec"],
        help="强制 evaluator 的 ode_accel 档位。用来证明 0.314 dB 与档位无关。",
    )
    args = ap.parse_args()

    import network_nlo_eval
    from multiband_optical_rl.benchmark import build_env
    from multiband_optical_rl.request import to_service_request
    from multiband_optical_rl.scan import verify_for_scan

    # 参考树（ecz-dev / pre-FG-4a）根本没有 ode_accel 这个概念，因此这里不能
    # 直接 import 常量，否则参考档一侧根本跑不起来。
    import network_nlo_eval.models.isrs_gn as _isrs_gn

    accel_default = getattr(_isrs_gn, "ODE_ACCEL_DEFAULT", "<absent: pre-FG-4a tree>")
    accel_modes = list(getattr(_isrs_gn, "ODE_ACCEL_MODES", ()))

    report: dict[str, Any] = {
        "schema_version": 1,
        "what": "FG-4g Q2 — side-by-side orchestration trace, verify_allocation vs verify_for_scan",
        "package_paths": {
            "network_nlo_eval": str(Path(network_nlo_eval.__file__).resolve()),
            "multiband_optical_rl": str(
                Path(sys.modules["multiband_optical_rl"].__file__).resolve()
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "loadavg": os.getloadavg(),
        },
        "ode_accel_default": accel_default,
        "ode_accel_modes": accel_modes,
        "config": vars(args),
        "cases": [],
    }

    env = build_env(ASSET, num_requests=60, avg_arrival_interval=10.0, avg_holding_time=400.0, seed=0)
    if args.force_accel is not None:
        # env.py 用默认参数构造 evaluator（不传 ode_accel），因此这里改的是
        # 实例属性：下游 for_fiber 会把它透传给每个子 evaluator。
        env.evaluator.ode_accel = args.force_accel
        # ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：被强制的是 FG 的加速档（``"off"`` /
        # ``"matvec"``），它们属于 **P2** 那套物理。构造器里的 ``nli_model`` 默认
        # 是 ``"p1_closed"``，而 P1 档**没有** matvec——只改 ``ode_accel`` 会让
        # ``for_fiber`` 透传出一对不匹配的组合并抛错。故这里**一起**把 NLI 物理
        # 档位钉到 ``"p2_gn"``：被比较的本来就是 FG 那套物理（Q2 分歧是 FG 侧
        # 的编排问题）。
        env.evaluator.nli_model = "p2_gn"
        if args.force_accel == "off":
            # FG-15: ``off`` 是结构上只认 legacy 的冻结参照路径（在核内逐对重建
            # legacy 折线），而 FG-15 起全局默认谱档是 ``tabular``。两者**必须**
            # 一起改：for_fiber 会把 (ode_accel, raman_spectrum) 一起透传给子
            # evaluator，只改前者会让子构造 ``tabular`` + ``off`` 而抛错。
            # 构造期建好的 G 矩阵在这里用不到——``off`` 走的是核内 legacy 折线。
            env.evaluator.raman_spectrum = "legacy"
        env.evaluator._fiber_evaluator_cache.clear()

    def purge() -> None:
        for ls in env.network_state.get_all_link_states().values():
            ls.invalidate_noise_cache()

    def paths_of_hop_counts(wanted, limit):
        collected = {h: [] for h in wanted}
        for s, d in sorted(env.path_cache.get_all_ksp_pairs()):
            for path in env.path_cache.get_paths(s, d):
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

    _install_recorder()
    try:
        for steps in [int(s) for s in args.steps.split(",")]:
            env.reset(seed=0)
            for _ in range(steps):
                env.step(int(env.action_masks().argmax()))
            request = to_service_request(env.current_request)
            hop_counts = paths_of_hop_counts(
                tuple(int(h) for h in args.hops.split(",")), args.max_paths_per_hops
            )
            for hops, paths in hop_counts.items():
                for path in paths:
                    for band in (0, 1):
                        for channel in free_channels(path, band)[::int(args.strides)]:
                            purge()
                            _TRACE.clear()
                            _PHASE["name"] = "A"
                            base_ok, base_arr = env.qot_validator.verify_allocation(
                                service_request=request, path=list(path), channel_idx=channel,
                                current_network_state=env.network_state,
                            )
                            trace_a = list(_TRACE)

                            purge()
                            _TRACE.clear()
                            _PHASE["name"] = "B"
                            fast_ok, fast_snr = verify_for_scan(
                                service_request=request, path=list(path), channel_idx=channel,
                                current_network_state=env.network_state, validator=env.qot_validator,
                            )
                            trace_b = list(_TRACE)
                            _PHASE["name"] = None

                            case = {
                                "steps": steps, "hops": hops, "band": band, "channel": channel,
                                "path": list(path),
                                "base_ok": bool(base_ok), "fast_ok": bool(fast_ok),
                            }
                            if base_ok != fast_ok:
                                case["verdict_mismatch"] = True
                            elif base_ok:
                                a = float(np.asarray(base_arr)[channel])
                                b = float(fast_snr)
                                case["snr_A_db"] = a
                                case["snr_B_db"] = b
                                case["snr_abs_diff_db"] = abs(a - b)
                                case["snr_mismatch"] = (a != b)
                            case["orchestration"] = _first_difference(trace_a, trace_b)
                            if len(report["cases"]) < args.max_trace_cases or case.get("snr_mismatch") \
                                    or case.get("verdict_mismatch"):
                                if case.get("snr_mismatch") or case.get("verdict_mismatch"):
                                    case["trace_A"] = _describe(trace_a)
                                    case["trace_B"] = _describe(trace_b)
                            report["cases"].append(case)
                            if sum(1 for c in report["cases"] if c.get("snr_mismatch")
                                   or c.get("verdict_mismatch")) >= args.max_trace_cases:
                                break
                        else:
                            continue
                        break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
    finally:
        _uninstall_recorder()

    mism = [c for c in report["cases"] if c.get("snr_mismatch") or c.get("verdict_mismatch")]
    diffs = [c.get("snr_abs_diff_db") for c in report["cases"] if c.get("snr_abs_diff_db") is not None]
    report["summary"] = {
        "n_cases": len(report["cases"]),
        "n_mismatch": len(mism),
        "worst_snr_abs_diff_db": max(diffs) if diffs else None,
        "orchestration_kinds": sorted({str(c["orchestration"]["kind"]) for c in report["cases"]}),
    }

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                              encoding="utf-8")
    print(f"wrote {args.out}")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    for c in report["cases"][:6]:
        print("  ", c["hops"], c["band"], c["channel"],
              "snrA=", c.get("snr_A_db"), "snrB=", c.get("snr_B_db"),
              "orch=", c["orchestration"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
