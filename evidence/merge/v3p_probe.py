"""WP-MERGE-FG-FLEX **第 3 轮**判据 V3' 的实测探针。

V3' 的原文（编排方第 3 轮裁决 §2）：

    在**混合 n_slots**（1/2/3/4 槽同链路、channel_bandwidth_source="occupied"、
    B_i 真非均匀）下：
    * p1_closed 的 cached ≡ off 逐位；
    * p1_closed 的 batch ≡ serial 逐位（若无批量路径，写明「P1 档暂无批量」
      并把该条记为未达成，**不许假装**）；
    * p2_gn 的 matvec ≡ off、cached_g ≡ matvec 按 FG 既有 1e-12 预算。

本探针把三段**分别**测出来，并如实报告每一段的逐位/预算结论。

⚠️ **为什么 P1 档没有批量路径**：批量核（``models/batch_kernels.py``）是
**P2** 的实现结构——它批量求值的是 P2 的**张量化** NLI（``_compute_nli_variances_t``），
而 P1 的 NLI 是**闭式**、没有对应的批量形式。本轮范围明确写着「P1 核的加速
（matvec/批量）不在本轮，另立一轮」。

因此本探针对 P1 档的批量段**实测并如实记录**：
``calc_span_noise_and_power_batch`` 在 ``nli_model="p1_closed"`` 下**显式抛
``ValueError``**（不是静默退回串行、也不是假装跑通）⇒ 该条记为**未达成**。

用法::

    PYTHONPATH=$PWD/src:$PWD/tests ~/miniconda3/envs/rl/bin/python \
        <MERGE_WT>/evidence/merge/v3p_probe.py --out <json>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: FG 自己定的相对偏差预算，**原样沿用，不放宽**。
MAX_REL_DEVIATION = 1e-12

#: 混合 n_slots 计划：(槽数, 比特率, 调制, 符号率) —— 与
#: ``tests/test_merge_fg_into_flex.py`` 的 ``MIXED_PLAN`` 逐字一致。
MIXED_PLAN = (
    (1, 25e9, "QPSK", 25e9),
    (2, 50e9, "QPSK", 50e9),
    (3, 100e9, "8QAM", 100e9 / 3.0),
    (4, 100e9, "QPSK", 100e9),
)
SLOT_WIDTH_HZ = 12.5e9
NUM_SLOTS = 800
BAND_LOW_EDGE_HZ = 191.0e12
SLOTS_PER_CHANNEL = 4
SPAN_LENGTH_M = 80e3


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


def _grid():
    from network_nlo_eval.core.flexgrid import FlexibleGrid

    return FlexibleGrid(
        slot_width_hz=SLOT_WIDTH_HZ, num_slots=NUM_SLOTS, band_low_edge_hz=BAND_LOW_EDGE_HZ
    )


def _connections(grid):
    from network_nlo_eval.core.flexgrid import ConnectionSpec

    conns = []
    first = 0
    for index, (n_slots, bit_rate, modulation, symbol_rate) in enumerate(MIXED_PLAN):
        conns.append(
            ConnectionSpec(
                grid=grid,
                connection_id=index,
                first_slot=first,
                n_slots=n_slots,
                bit_rate_bps=bit_rate,
                modulation=modulation,
                symbol_rate_hz=symbol_rate,
                roll_off=0.0,
                launch_power_w=1e-3,
            )
        )
        first += n_slots + 4
    return conns


def _evaluator(grid, conns, ode_accel, *, nli_model, raman_spectrum=None, bandwidth_source="occupied"):
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    if raman_spectrum is None:
        raman_spectrum = "legacy" if ode_accel in ("off", "cached") else "tabular"
    return MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
        symbol_rate_hz=50e9,
        connections=conns,
        slots_per_channel=SLOTS_PER_CHANNEL,
        ode_accel=ode_accel,
        raman_spectrum=raman_spectrum,
        channel_bandwidth_source=bandwidth_source,
        nli_model=nli_model,
    )


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _bitwise(a, b) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def _pair(a, b) -> dict[str, Any]:
    return {
        "bitwise": _bitwise(a, b),
        "max_rel": max(_max_rel(x, y) for x, y in zip(a, b, strict=True)),
    }


def measure() -> dict[str, Any]:
    grid = _grid()
    conns = _connections(grid)
    power = np.array([c.launch_power_w for c in conns], dtype=np.float64)

    # 前置：确认 B_i **真的**非均匀（否则下面全是自证）
    ev = _evaluator(grid, conns, "off", nli_model="p1_closed")
    b_i = ev.channel_bandwidth_hz_channels
    b_i_nonuniform = len(set(b_i.tolist())) > 1
    by_rate = _evaluator(
        grid, conns, "off", nli_model="p1_closed", bandwidth_source="symbol_rate"
    )
    b_i_differs_from_symbol_rate = not np.array_equal(
        b_i, by_rate.channel_bandwidth_hz_channels
    )

    results: list[dict[str, Any]] = []

    # ---- P1 档：cached ≡ off（要求**逐位**）----
    p1_off = _evaluator(grid, conns, "off", nli_model="p1_closed")._calc_span_noise_and_power(
        power, SPAN_LENGTH_M, None, None, 1
    )
    p1_cached = _evaluator(grid, conns, "cached", nli_model="p1_closed")._calc_span_noise_and_power(
        power, SPAN_LENGTH_M, None, None, 1
    )
    p1_cached_off = _pair(p1_off, p1_cached)
    results.append(
        {
            "id": "p1_closed:cached==off",
            "requirement": "bit-exact",
            "nli_model": "p1_closed",
            **p1_cached_off,
            "passed": bool(p1_cached_off["bitwise"]),
        }
    )

    # ---- P1 档：batch ≡ serial（要求**逐位**；P1 暂无批量 ⇒ 如实记未达成）----
    p1_batch_status: str
    p1_batch_error: str | None = None
    p1_batch = None
    try:
        ev_p1 = _evaluator(grid, conns, "off", nli_model="p1_closed")
        ev_p1.calc_span_noise_and_power_batch(
            np.ascontiguousarray(power[None, :]), SPAN_LENGTH_M, None, None, 1
        )
        p1_batch_status = "available"
    except ValueError as exc:
        p1_batch_status = "unavailable"
        p1_batch_error = str(exc)
    results.append(
        {
            "id": "p1_closed:batch==serial",
            "requirement": "bit-exact",
            "nli_model": "p1_closed",
            "status": p1_batch_status,
            "error": p1_batch_error,
            "bitwise": p1_batch is not None,
            "max_rel": None,
            "passed": False,
            "note": (
                "P1 档暂无批量路径：批量核是 P2 的实现结构（批量求值 P2 的张量化 NLI），"
                "P1 的 NLI 是闭式、没有对应批量形式。批量入口对 p1_closed **显式抛 "
                "ValueError**（不是静默退回串行）。本轮范围明确 P1 核的加速另立一轮。"
                "⇒ 本条**记为未达成**，不假装。"
            ),
        }
    )

    # ---- P2 档：matvec ≡ off、cached_g ≡ matvec（FG 既有 1e-12 预算）----
    p2_off = _evaluator(grid, conns, "off", nli_model="p2_gn", raman_spectrum="legacy")._calc_span_noise_and_power(
        power, SPAN_LENGTH_M, None, None, 1
    )
    p2_matvec = _evaluator(
        grid, conns, "matvec", nli_model="p2_gn", raman_spectrum="legacy"
    )._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
    p2_matvec_off = _pair(p2_off, p2_matvec)
    results.append(
        {
            "id": "p2_gn:matvec==off",
            "requirement": f"max_rel <= {MAX_REL_DEVIATION}",
            "nli_model": "p2_gn",
            **p2_matvec_off,
            "passed": bool(p2_matvec_off["max_rel"] <= MAX_REL_DEVIATION),
        }
    )

    p2_cached_g = _evaluator(
        grid, conns, "cached_g", nli_model="p2_gn", raman_spectrum="tabular"
    )._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
    p2_matvec_tab = _evaluator(
        grid, conns, "matvec", nli_model="p2_gn", raman_spectrum="tabular"
    )._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
    p2_cached_g_matvec = _pair(p2_cached_g, p2_matvec_tab)
    results.append(
        {
            "id": "p2_gn:cached_g==matvec",
            "requirement": f"max_rel <= {MAX_REL_DEVIATION}",
            "nli_model": "p2_gn",
            **p2_cached_g_matvec,
            "passed": bool(p2_cached_g_matvec["max_rel"] <= MAX_REL_DEVIATION),
        }
    )

    payload = {
        "probe": "v3p_probe",
        "wp": "WP-MERGE-FG-FLEX",
        "round": 3,
        "criterion": "V3' per-channel B_i under mixed n_slots",
        "max_rel_budget": MAX_REL_DEVIATION,
        "mixed_plan": [[n, br, mod, sr] for n, br, mod, sr in MIXED_PLAN],
        "b_i": {
            "values": b_i.tolist(),
            "nonuniform": b_i_nonuniform,
            "differs_from_symbol_rate_source": b_i_differs_from_symbol_rate,
        },
        "results": results,
        "provenance": _provenance(),
    }
    payload["readings_sha256"] = hashlib.sha256(
        json.dumps(results, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payload = measure()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"B_i nonuniform={payload['b_i']['nonuniform']} "
          f"differs_from_symbol_rate={payload['b_i']['differs_from_symbol_rate_source']}")
    print(f"B_i = {payload['b_i']['values']}")
    for row in payload["results"]:
        if row.get("status"):
            print(f"  {row['id']:32s} status={row['status']:12s} passed={row['passed']}  (未达成)")
        else:
            print(f"  {row['id']:32s} bitwise={row['bitwise']!s:5s} max_rel={row['max_rel']:.3e} "
                  f"passed={row['passed']}")
    print(f"sha256={payload['readings_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
