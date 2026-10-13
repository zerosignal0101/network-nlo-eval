#!/usr/bin/env python3
"""WP-Q1 判据 Q-J1：legacy 逐位不变（**硬闸门**）。

判据（跑前写死，不许放宽、不许换口径）::

    6 个 ``legacy_*`` 场景上，本树与只读基线 ``ecz-dev`` 的 ``delta_db``
    必须**恰为 0.0**；拒绝场景必须两边都拒。

协议**不重新发明**：直接复用 ``tools/ode3_legacy_regression.py`` 的同一条链路
（它自己又复用 ``tools/ode_compare.py`` + ``scripts_shared/compat_run.py``
+ ``cross_snr_compat.py`` 签名兼容层）。本文件只做两件事：

1. 把同一份读数落到 WP-Q1 自己的证据文件 ``results/bwqot_legacy_regression.json``；
2. 追加 **B 退化的等价性检查**——Q-J1 的物理内容是「均匀 50 GHz ⇒ ``B_i``
   退化为标量 ⇒ 逐位不变」，所以额外钉住：在 legacy 构造下
   ``channel_bandwidth_hz_channels`` 必须恒等于那个全局标量
   ``symbol_rate_hz``（逐元素 bit 级相等），这正是 legacy 数字不动的**原因**，
   而不是碰巧。

用法::

    PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_legacy_regression.py --write
    PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_legacy_regression.py --check

产物：``results/bwqot_legacy_regression.json``
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "bwqot_legacy_regression.json"

#: 判据（跑前写死）：逐位相同，不设容差。
REQUIRED_DELTA = 0.0
N_SCENARIOS = 6


def _load_ode3_legacy():
    spec = importlib.util.spec_from_file_location(
        "ode3_legacy_regression", REPO / "tools" / "ode3_legacy_regression.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_b_degenerates_to_scalar() -> dict:
    """Q-J1 的物理内容：legacy 均匀 50 GHz 下 ``B_i`` 必须退化成那个标量。

    这是**构造性**证明（不是靠测试反证）：legacy 走 ``connections is None``
    分支，``channel_bandwidth_hz_channels`` 直接 ``np.full(n_nodes, symbol_rate_hz)``。
    这里对实际构造出来的评估器逐元素比对，要求与标量**按位相等**。
    """
    import numpy as np

    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    symbol_rate_hz = 50e9
    grid = SpectrumGrid(8, 50e9, 193.1e12)
    ev = MultiBandISRSGN(grid, FiberSpanConfig(length_km=80.0), symbol_rate_hz)
    b = ev.channel_bandwidth_hz_channels
    scalar_vec = np.full(ev.n_nodes, float(symbol_rate_hz), dtype=np.float64)

    # bit 级相等：直接比对 int64 位模式，任何一位不同都会被抓出来。
    bits_equal = bool(
        np.array_equal(b.view(np.int64), scalar_vec.view(np.int64))
    )
    return {
        "mode": "legacy (connections is None)",
        "n_nodes": int(ev.n_nodes),
        "symbol_rate_hz": symbol_rate_hz,
        "b_hz_channels": [float(v) for v in b],
        "equals_scalar_bitwise": bits_equal,
        "note": (
            "Q-J1 说『均匀 50 GHz ⇒ B_i 退化标量』。这里证明的是该退化在 IEEE "
            "双精度层面逐位成立 —— 因此内核读到的 B 与 WP-Q1 之前那个全局标量是"
            "同一个数，SPM/XPM/ASE 的浮点运算序列一字未变。"
        ),
    }


def run() -> dict:
    legacy = _load_ode3_legacy()
    base = legacy.run()  # 同一协议、同一场景集、同一基线树

    rows = base["scenarios"]
    degeneration = check_b_degenerates_to_scalar()

    failures = []
    if len(rows) != N_SCENARIOS:
        failures.append(f"Q-J1: 场景数 {len(rows)} != {N_SCENARIOS}")
    for r in rows:
        if r["both_rejected"]:
            continue
        if r["delta_db"] is None:
            failures.append(f"Q-J1: {r['scenario']} 无 delta_db 读数")
        elif r["delta_db"] != REQUIRED_DELTA:
            failures.append(f"Q-J1: {r['scenario']} delta_db={r['delta_db']!r} != 0.0")
    if not degeneration["equals_scalar_bitwise"]:
        failures.append("Q-J1: legacy 下 B_i 未逐位退化为标量")

    return {
        "wp": "Q1",
        "judgment": "Q-J1",
        "baseline_tree": base["baseline_tree"],
        "baseline_path": base["baseline_path"],
        "candidate_tree": "wp/bw-qot (ode-dev)",
        "protocol": base["protocol"] + "；Q-J1 复用该链路，未另立口径",
        "tolerance": REQUIRED_DELTA,
        "scenarios": rows,
        "b_degenerates_to_scalar": degeneration,
        "summary": {
            "n_scenarios": len(rows),
            "n_bit_identical": sum(1 for r in rows if r["bit_identical"]),
            "n_both_rejected": sum(1 for r in rows if r["both_rejected"]),
            "max_abs_delta_db": base["summary"]["max_abs_delta_db"],
            "all_delta_db_zero": all(
                r["both_rejected"] or r["delta_db"] == REQUIRED_DELTA for r in rows
            ),
        },
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = run()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    for r in doc["scenarios"]:
        d = r["delta_db"]
        ds = "n/a（两端都拒）" if d is None else f"{d:+.9f}"
        print(f"    {r['scenario']:<34} ours={r['ode_dev_snr_db']!s:<20} "
              f"ecz={r['ecz_dev_snr_db']!s:<20} delta={ds}")
    print(f"verdict={doc['verdict']}  max_abs_delta_db={doc['summary']['max_abs_delta_db']}")
    for f in doc["failures"]:
        print(f"  FAIL: {f}")
    if args.check:
        return 0 if doc["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())