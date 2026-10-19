"""WP-MERGE-FG-FLEX **第 5 轮**判据 V1 / V2 / V3 的实测探针。

本文件由 ``evidence/merge/v4p_probe.py``（第 4 轮版，**原样保留未改**，作为
历史读数出处）派生，**唯一改动**：去掉 NLI 物理档位形参——第 5 轮把两套 NLI
物理的「二选一」机制整体移除，本树只剩**闭式 ECZ 一套**，形参不再存在。

第 5 轮的其余两段判据由**另外两个**来源承担：

* **公式不动 + 与 ``flexgrid-dev @ cb24704`` 的 5752 项逐项对照**（V1）
  → ``evidence/merge/v1p_p1_probe.py``；它显式钉 ``ode_accel="off"``，
  因为要回答的命题是「**冻结参考路径**与 cb24704 逐位相同」。
* **套件**（V4）→ 外部 ``pytest``，结论抄进报告。

本探针负责**可机器复算**的三段：

* **V1 公式不动** —— 冻结内核 ``numba_kernels.py`` 的字节身份（sha256 必须仍是
  ``6611c416…``）、``b_hz_channels[i]`` / ``b_hz_channels[l_idx]`` 字面量是否还在、
  P1 的常数与 ``asinh``/``atan`` 参数形式是否逐字未变。
* **V2 加速正确性** —— ``matvec ≡ off`` 与 ``batch ≡ serial``，在**混合 n_slots**
  （``B_i`` 真非均匀 ``[1.25e10, 2.50e10, 3.75e10, 5.00e10]``）与**均匀**两种构造下，
  逐批宽（8/16/32/68/142）读数。**预算 1e-12，不放宽。**
* **V3 收益** —— 单候选串行 vs ``matvec`` 档、批量 vs 串行，
  **必须带 loadavg 与线程数**（没有这两个数的倍率不算证据）。

用法::

    PYTHONPATH=$PWD/src:$PWD/tests ~/miniconda3/envs/rl/bin/python \
        evidence/merge/v5_probe.py --out <json>
"""


from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

#: FG-4a/FG-4b 的相对偏差预算，原样沿用，**不放宽**。
MAX_REL_DEVIATION = 1e-12

#: 批宽（候选数）——任务书 V2''/V3'' 点名的五个。
BATCH_WIDTHS = (8, 16, 32, 68, 142)

#: 信道数 N。
N_CH = 68

SLOT_WIDTH_HZ = 12.5e9
NUM_SLOTS = 4000
BAND_LOW_EDGE_HZ = 191.0e12
SPAN_LENGTH_M = 80e3
SLOTS_PER_CHANNEL = 4

#: 混合构造的槽数循环 —— 1/2/3/4 槽 ⇒ ``B_i`` 恰为
#: ``[1.25e10, 2.50e10, 3.75e10, 5.00e10]``（= n_slots × 12.5 GHz）。
SLOT_CYCLE = (1, 2, 3, 4)

#: 冻结内核的 sha256（flexgrid-dev @ cb24704 的字节身份）。
FROZEN_KERNEL_SHA256 = "6611c4164a60d0dcc653e47c5f02d81534e9c78b03e71213aa30d8b13bfd4aa4"


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def _provenance() -> dict[str, Any]:
    import network_nlo_eval

    root = Path(network_nlo_eval.__file__).resolve().parents[2]
    commit = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if (root / ".git").exists()
        else f"unknown (no .git at {root})"
    )
    return {
        "phys_commit": commit,
        "phys_tree": str(Path(network_nlo_eval.__file__).resolve()),
        "python": sys.version.split()[0],
    }


def _environment() -> dict[str, Any]:
    """**计时证据的必备上下文**：loadavg、线程数、BLAS 实现。

    这台机上一切计时都受 loadavg 与线程数影响；没有这两个数的倍率不算证据。
    """
    load1, load5, load15 = os.getloadavg()
    env: dict[str, Any] = {
        "loadavg_1m": load1,
        "loadavg_5m": load5,
        "loadavg_15m": load15,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)),
        "thread_env": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    try:
        import mkl  # type: ignore[import-not-found]

        env["mkl_max_threads"] = int(mkl.get_max_threads())
    except Exception:  # pragma: no cover - 没装 mkl 就记 null
        env["mkl_max_threads"] = None
    try:
        import numba

        env["numba_threading_layer"] = str(numba.config.THREADING_LAYER)
    except Exception:  # pragma: no cover
        env["numba_threading_layer"] = None
    try:
        cfg = np.__config__.CONFIG  # numpy >= 2
        blas = cfg.get("Build Dependencies", {}).get("blas", {})
        env["blas_name"] = blas.get("name")
        env["blas_version"] = blas.get("version")
    except Exception:  # pragma: no cover
        env["blas_name"] = None
        env["blas_version"] = None
    return env


def _grid():
    from network_nlo_eval.core.flexgrid import FlexibleGrid

    return FlexibleGrid(
        slot_width_hz=SLOT_WIDTH_HZ, num_slots=NUM_SLOTS, band_low_edge_hz=BAND_LOW_EDGE_HZ
    )


def _connections(grid, n_ch: int, mixed: bool):
    """构造 ``n_ch`` 条连接；``mixed=True`` 时槽数按 1/2/3/4 循环。

    ⇒ 混合构造下 ``B_i`` 恰为 ``[1.25e10, 2.50e10, 3.75e10, 5.00e10]``（真非均匀）。
    """
    from network_nlo_eval.core.flexgrid import ConnectionSpec

    conns = []
    first = 0
    for index in range(n_ch):
        n_slots = SLOT_CYCLE[index % len(SLOT_CYCLE)] if mixed else 1
        conns.append(
            ConnectionSpec(
                grid=grid,
                connection_id=index,
                first_slot=first,
                n_slots=n_slots,
                bit_rate_bps=25e9 * n_slots,
                modulation="QPSK",
                symbol_rate_hz=25e9,
                roll_off=0.0,
                launch_power_w=1e-3,
            )
        )
        first += n_slots + 1
    return conns


def _evaluator(grid, conns, ode_accel: str):
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    return MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
        symbol_rate_hz=50e9,
        connections=conns,
        slots_per_channel=SLOTS_PER_CHANNEL,
        ode_accel=ode_accel,
        # p1_closed 档的拉曼谱只有 legacy（第 4 轮起 p1_closed 也有 matvec 档，
        # 但拉曼谱口径不变）。
        raman_spectrum="legacy",
        channel_bandwidth_source="occupied",
    )


def _candidate_profiles(n_ch: int, n_batch: int) -> np.ndarray:
    """B 个候选的入纤功率剖面 ``(B, N)``（确定性、可复算）。

    逐候选带一个**倾斜 + 全局缩放**，另外**故意**在每个候选里留几个零功率通道
    （惰性连接）——批量路径的掩码语义必须与串行一致，零功率是最容易露馅的那种。
    """
    P = np.empty((n_batch, n_ch), dtype=np.float64)
    for b in range(n_batch):
        tilt = 1.0 + 0.08 * (np.arange(n_ch, dtype=np.float64) / max(n_ch - 1, 1) - 0.5)
        scale = 0.85 + 0.05 * (b % 7)
        P[b] = 1e-3 * scale * tilt
    # 每个候选掐掉 3 个通道（确定性位置）。
    for b in range(n_batch):
        for k in range(3):
            P[b, (b * 13 + k * 29) % n_ch] = 0.0
    return P


# ---------------------------------------------------------------------------
# 数值比较
# ---------------------------------------------------------------------------
def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0:
        return 0.0
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.any():
        return 0.0
    diff = np.abs(a[finite] - b[finite])
    denom = np.abs(b[finite])
    return float(np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0).max())


def _max_ulp(a: np.ndarray, b: np.ndarray) -> int:
    """两个数组之间的最大 ULP 距离（int64 视图之差）。"""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size == 0:
        return 0
    ia = a.view(np.int64)
    ib = b.view(np.int64)
    d = np.abs(ia - ib)
    return int(d.max()) if d.size else 0


def _bitwise(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def _compare(a: tuple[np.ndarray, ...], b: tuple[np.ndarray, ...]) -> dict[str, Any]:
    """四个返回值的整体比较：逐位？max_rel？max ULP？逐项拆开。"""
    per_item = []
    for name, x, y in zip(("p_out", "sigma2_spm", "sigma2_xpm", "sigma2_ase"), a, b, strict=True):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        per_item.append(
            {
                "item": name,
                "bitwise": _bitwise(x, y),
                "max_rel": _max_rel(x, y),
                "max_ulp": _max_ulp(x, y),
            }
        )
    return {
        "bitwise": all(it["bitwise"] for it in per_item),
        "max_rel": max(it["max_rel"] for it in per_item),
        "max_ulp": max(it["max_ulp"] for it in per_item),
        "per_item": per_item,
    }


# ---------------------------------------------------------------------------
# V1''：公式不动
# ---------------------------------------------------------------------------
def _v1_formula_untouched() -> dict[str, Any]:
    import network_nlo_eval.models.numba_kernels as K

    path = Path(K.__file__)
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8")

    # 冻结核里必须**仍在**的 P1 字面量。
    literals = {
        "b_hz_channels[i]": "b_hz_channels[i]",
        "b_hz_channels[l_idx]": "b_hz_channels[l_idx]",
        "asinh_form_spm": "np.arcsinh((phi_i * b_i_hz**2) / (PI * alpha_i_npm))",
        "asinh_form_spm_half": "np.arcsinh((phi_i * b_i_hz**2) / (2 * PI * alpha_i_npm))",
        "atan_form_xpm": "np.arctan((phi_il * b_i_hz) / alpha_l_npm)",
        "atan_form_xpm_half": "np.arctan((phi_il * b_i_hz) / (2 * alpha_l_npm))",
        "xpm_32_27": "sigma_xpm_2[i] = (32.0 / 27.0) * p_in_w[i] * n_span * xpm_sum_term",
        "spm_4_9": "(4 / 9)",
        "xpm_prefactor": "(gamma_il**2 / (phi_il * 3 * alpha_l_npm**2))",
    }
    found = {name: (lit in text) for name, lit in literals.items()}

    return {
        "frozen_kernel_path": str(path),
        "frozen_kernel_sha256": sha,
        "frozen_kernel_sha256_matches_cb24704": sha == FROZEN_KERNEL_SHA256,
        "literal_occurrences": {
            "b_hz_channels[i]": text.count("b_hz_channels[i]"),
            "b_hz_channels[l_idx]": text.count("b_hz_channels[l_idx]"),
        },
        "literals_present": found,
        "all_literals_present": all(found.values()),
        "passed": bool(sha == FROZEN_KERNEL_SHA256 and all(found.values())),
    }


# ---------------------------------------------------------------------------
# V2''：加速正确性
# ---------------------------------------------------------------------------
def _v2_accuracy(grid, conns, mixed: bool) -> dict[str, Any]:
    n_ch = len(conns)
    results: list[dict[str, Any]] = []

    ev_off = _evaluator(grid, conns, "off")
    ev_mv = _evaluator(grid, conns, "matvec")

    b_i = ev_off.channel_bandwidth_hz_channels
    b_i_distinct = sorted({float(x) for x in b_i})

    # --- matvec ≡ off（单候选）---
    power = np.full(n_ch, 1e-3, dtype=np.float64)
    off = ev_off._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
    mv = ev_mv._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
    cmp_off = _compare(off, mv)
    results.append(
        {
            "id": "matvec==off",
            "requirement": f"max_rel <= {MAX_REL_DEVIATION}",
            **cmp_off,
            "passed": bool(cmp_off["max_rel"] <= MAX_REL_DEVIATION),
        }
    )

    # --- batch ≡ serial（逐批宽）---
    for width in BATCH_WIDTHS:
        P = _candidate_profiles(n_ch, width)
        batch = ev_mv.calc_span_noise_and_power_batch(P, SPAN_LENGTH_M, None, None, 1)
        serial = tuple(
            np.stack([ev_mv._calc_span_noise_and_power(P[b], SPAN_LENGTH_M, None, None, 1)[k] for b in range(width)])
            for k in range(4)
        )
        cmp_batch = _compare(serial, batch)
        results.append(
            {
                "id": f"batch==serial:B={width}",
                "requirement": f"max_rel <= {MAX_REL_DEVIATION}",
                "batch_width": width,
                **cmp_batch,
                "passed": bool(cmp_batch["max_rel"] <= MAX_REL_DEVIATION),
            }
        )

    return {
        "construction": "mixed_n_slots" if mixed else "uniform",
        "n_channels": n_ch,
        "b_i_values": [float(x) for x in b_i],
        "b_i_distinct": b_i_distinct,
        "b_i_nonuniform": len(b_i_distinct) > 1,
        "results": results,
        "passed": all(r["passed"] for r in results),
    }


# ---------------------------------------------------------------------------
# V3''：收益
# ---------------------------------------------------------------------------
def _time_serial(ev, P: np.ndarray) -> float:
    """逐候选串行跑一遍 B 个候选，返回秒数。"""
    t0 = time.perf_counter()
    for b in range(P.shape[0]):
        ev._calc_span_noise_and_power(P[b], SPAN_LENGTH_M, None, None, 1)
    return time.perf_counter() - t0


def _time_batch(ev, P: np.ndarray) -> float:
    t0 = time.perf_counter()
    ev.calc_span_noise_and_power_batch(P, SPAN_LENGTH_M, None, None, 1)
    return time.perf_counter() - t0


def _best_of(fn, reps: int) -> float:
    """取 ``reps`` 次里的**最小**耗时（受 loadavg 抖动影响，最小值最接近真值）。"""
    fn()
    return min(fn() for _ in range(reps))


def _v3_speedup(grid, conns) -> dict[str, Any]:
    n_ch = len(conns)
    rows: list[dict[str, Any]] = []

    for width in BATCH_WIDTHS:
        P = _candidate_profiles(n_ch, width)

        ev_off = _evaluator(grid, conns, "off")
        ev_mv = _evaluator(grid, conns, "matvec")

        # 冷启动：第一次调用（含 (N,N) 对矩阵的构建成本）。
        t_cold = _time_serial(ev_mv, P[:1])

        # 热路径：同一个 evaluator 上重复跑（构建成本已摊销）。
        t_off = _best_of(lambda: _time_serial(ev_off, P), reps=7)
        t_mv = _best_of(lambda: _time_serial(ev_mv, P), reps=7)
        t_batch = _best_of(lambda: _time_batch(ev_mv, P), reps=7)

        rows.append(
            {
                "batch_width": width,
                "n_channels": n_ch,
                "serial_off_s": t_off,
                "serial_matvec_s": t_mv,
                "batch_matvec_s": t_batch,
                # 单候选串行 vs matvec 档（同一批宽下）
                "tier_speedup_serial": (t_off / t_mv) if t_mv > 0 else None,
                # 批量 vs 串行（同一批宽下）
                "batch_speedup": (t_mv / t_batch) if t_batch > 0 else None,
                # 端到端：批量 vs off 串行
                "end_to_end_speedup": (t_off / t_batch) if t_batch > 0 else None,
                "cold_first_call_s": t_cold,
            }
        )

    # (N,N) 对矩阵的**一次性**构建成本（evaluator 级，跨跨段/跨候选复用）。
    ev_build = _evaluator(grid, conns, "matvec")
    t0 = time.perf_counter()
    ev_build._p1_xpm_matrices_for()
    t_build = time.perf_counter() - t0

    return {
        "rows": rows,
        "pair_matrix_build_s": t_build,
        "note": (
            "pair_matrix_build_s 是 (N,N) 对矩阵的**一次性**构建成本（evaluator 级，"
            "跨跨段/跨候选复用）。serial_matvec_s / batch_matvec_s 都是**热路径**读数"
            "（构建已摊销）；cold_first_call_s 是**冷启动**首次调用的读数（含构建）。"
        ),
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def measure() -> dict[str, Any]:
    grid = _grid()

    v1 = _v1_formula_untouched()
    mixed_conns = _connections(grid, N_CH, mixed=True)
    uniform_conns = _connections(grid, N_CH, mixed=False)

    v2_mixed = _v2_accuracy(grid, mixed_conns, mixed=True)
    v2_uniform = _v2_accuracy(grid, uniform_conns, mixed=False)

    v3 = _v3_speedup(grid, mixed_conns)

    return {
        "probe": "v5_probe",
        "wp": "WP-MERGE-FG-FLEX",
        "round": 5,
        "max_rel_budget": MAX_REL_DEVIATION,
        "batch_widths": list(BATCH_WIDTHS),
        "n_channels": N_CH,
        "environment": _environment(),
        "V1pp_formula_untouched": v1,
        "V2pp_accuracy": {
            "mixed_n_slots": v2_mixed,
            "uniform": v2_uniform,
            "passed": bool(v2_mixed["passed"] and v2_uniform["passed"]),
        },
        "V3pp_speedup": v3,
        "provenance": _provenance(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = measure()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    env = payload["environment"]
    print(
        f"[env] loadavg={env['loadavg_1m']:.2f}/{env['loadavg_5m']:.2f}/{env['loadavg_15m']:.2f} "
        f"cpu={env['cpu_count']} affinity={env['cpu_affinity']} "
        f"mkl_threads={env['mkl_max_threads']} blas={env['blas_name']}"
    )

    v1 = payload["V1pp_formula_untouched"]
    print(f"[V1''] frozen kernel sha256={v1['frozen_kernel_sha256'][:16]}… "
          f"matches_cb24704={v1['frozen_kernel_sha256_matches_cb24704']} "
          f"literals={v1['literal_occurrences']} passed={v1['passed']}")

    for name, block in payload["V2pp_accuracy"].items():
        if not isinstance(block, dict) or "results" not in block:
            continue
        print(f"[V2''] construction={name} B_i_distinct={block['b_i_distinct']}")
        for r in block["results"]:
            print(f"    {r['id']:24s} bitwise={str(r['bitwise']):5s} "
                  f"max_rel={r['max_rel']:.3e} max_ulp={r['max_ulp']:3d} passed={r['passed']}")

    v3 = payload["V3pp_speedup"]
    print(f"[V3''] pair-matrix build = {v3['pair_matrix_build_s'] * 1e3:.3f} ms (one-time, evaluator-level)")
    for r in v3["rows"]:
        print(f"    B={r['batch_width']:4d}  serial_off={r['serial_off_s'] * 1e3:9.3f} ms  "
              f"serial_matvec={r['serial_matvec_s'] * 1e3:9.3f} ms  "
              f"batch_matvec={r['batch_matvec_s'] * 1e3:9.3f} ms  "
              f"tier={r['tier_speedup_serial']:6.2f}x  batch={r['batch_speedup']:7.2f}x  "
              f"e2e={r['end_to_end_speedup']:7.2f}x  cold_first={r['cold_first_call_s'] * 1e3:7.3f} ms")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
