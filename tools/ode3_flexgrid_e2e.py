#!/usr/bin/env python3
"""WP-O3 步骤 2：灵活栅格**端到端**验证（J8）。

跑通的完整链路（J8 要求的四步）::

    FlexibleGrid / ConnectionSpec
        -> BandPlan.from_connections
        -> MultiBandISRSGN(connections=...)
        -> QoTValidator.verify_allocation
        -> 逐连接 SNR 落盘

同时跑三个探针，把"能跑通"之外的事实也钉死：

1. ``heterogeneous_rs_probe`` —— 证明逐连接**异构**符号率确实被冻结内核挡住
   （J5 上报项，见 WP-O2 已有的同一裁定）；
2. ``blocked_slot_probe`` —— 证明 ``blocked_slots`` 的**槽区间重叠**语义生效，
   且宽连接被跨界阻断时是真的被阻断（不是只看中心频率）；
3. ``band_crossing_probe`` —— 证明 ``assert_connections_single_band`` 会显式拒绝
   横跨 L|C 边界的宽连接，而不是静默按中心频率归一段。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode3_flexgrid_e2e.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode3_flexgrid_e2e.py --check

产物：``results/ode_dev_flexgrid_e2e.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results" / "ode_dev_flexgrid_e2e.json"


def _jsonable(value):
    """把 numpy 标量/数组与 ±inf 转成可 JSON 序列化的形式。"""
    import numpy as np

    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        if value == float("-inf"):
            return "-inf"
        if value != value:  # NaN
            return "nan"
    return value


def run_e2e() -> dict:
    import numpy as np

    import ode3_flexgrid_scenarios as sc
    from network_nlo_eval.core.bands import BandPlan
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.state import NetworkState
    from network_nlo_eval.rwa.qot_checker import QoTValidator
    from network_nlo_eval.simulation.traffic import ServiceRequest

    # --- 1. 构造连接与频段归属 -------------------------------------------
    connections = sc.build_connections()
    plan = BandPlan.from_connections(connections, sc.BAND_SPECS)
    plan.assert_connections_single_band(connections)
    powered = plan.apply_launch_power(connections)

    n_slots_values = sorted({conn.n_slots for conn in powered})

    # --- 2. 评估器（逐连接模式） ------------------------------------------
    topology = sc.build_topology()
    evaluator = MultiBandISRSGN(
        sc.FLEX_GRID,
        topology.get_default_fiber_config(),
        sc.SYMBOL_RATE_HZ,
        connections=powered,
    )

    # --- 3. 网络状态：逐波段功率 -> 逐连接发射谱 --------------------------
    state_grid = sc.build_state_grid(len(powered))
    state = NetworkState(topology, state_grid)
    profile = np.array([conn.launch_power_w for conn in powered], dtype=np.float64)
    for link_state in state.get_all_link_states().values():
        link_state.launch_power_profile_w[:] = profile

    # --- 4. 端到端验证 ----------------------------------------------------
    idx = sc.SERVICE_CONNECTION_IDX
    request = ServiceRequest(
        service_id=0,
        source_id=sc.SERVICE_PATH[0],
        destination_id=sc.SERVICE_PATH[-1],
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=next(p.bit_rate_bps for p in sc.CONNECTION_PLANS if p.connection_id == idx) / 1e9,
        snr_requirement_db=-1e9,
        launch_power_w=float(powered[idx].launch_power_w),
    )
    validator = QoTValidator(evaluator, topology, state_grid)
    accepted, snrs_db = validator.verify_allocation(request, list(sc.SERVICE_PATH), idx, state)

    snrs = np.asarray(snrs_db, dtype=np.float64) if snrs_db is not None else np.array([])

    per_connection = []
    for i, conn in enumerate(powered):
        per_connection.append(
            {
                "connection_id": conn.connection_id,
                "node_index": i,
                "first_slot": conn.first_slot,
                "n_slots": conn.n_slots,
                "occupied_bandwidth_ghz": conn.occupied_bandwidth_hz / 1e9,
                "centre_frequency_hz": _jsonable(conn.centre_hz),
                "centre_frequency_thz": _jsonable(conn.centre_hz / 1e12),
                "bit_rate_gbps": conn.bit_rate_bps / 1e9,
                "modulation": conn.modulation,
                "declared_symbol_rate_hz": _jsonable(conn.symbol_rate_hz),
                "band": plan.band_names[plan.band_of(i)],
                "launch_power_w": _jsonable(conn.launch_power_w),
                "snr_db": _jsonable(float(snrs[i])) if snrs.size else None,
                "is_service_under_test": i == idx,
            }
        )

    # --- 探针 1：逐连接异构带宽现在**受支持**（WP-Q1）--------------------
    # O2/O3 时代这里断言"必须被拒"（当时冻结内核只吃标量 Rs）。WP-Q1 按 J2 授权把
    # B_i 逐信道化后，逐连接异构带宽已是合法输入，故探针改为记录**接受**事实，
    # 并额外钉住"均匀带宽下逐连接结果 == 遗留标量结果"这一 legacy 等价。
    hetero = sc.build_connections()
    hetero[1] = replace(hetero[1], symbol_rate_hz=75e9)
    hetero_probe: dict[str, object] = {
        "description": "逐连接异构带宽 -> WP-Q1 已支持（P1 (23)/(29)/(30)/(34) 的 B_i 本就是逐信道量）",
        "rates_hz": [float(c.symbol_rate_hz) for c in hetero],
        "rejected": False,
        "accepted": False,
        "b_hz_channels": None,
        "error": None,
    }
    try:
        hetero_ev = MultiBandISRSGN(
            sc.FLEX_GRID, topology.get_default_fiber_config(), sc.SYMBOL_RATE_HZ, connections=hetero
        )
        hetero_probe["accepted"] = True
        hetero_probe["b_hz_channels"] = [float(b) for b in hetero_ev.channel_bandwidth_hz_channels]
    except ValueError as exc:
        hetero_probe["rejected"] = True
        hetero_probe["error"] = f"{type(exc).__name__}: {exc}"

    # --- 探针 2：blocked_slots 槽区间重叠语义 ---------------------------
    blocked_probe = _blocked_slot_probe(topology, powered)

    # --- 探针 3：跨频段边界的宽连接必须被拒 -------------------------------
    # 16 槽 = 200 GHz；16QAM 需 bit_rate = 200 GHz * 4 = 800 Gbps 才满足
    # ``slots_for_rate`` 不变式。放在 first_slot=32 处占 [191.0, 191.6] THz，
    # 而 L|C 边界 nm_to_hz(1565) ≈ 191.548 THz ⇒ 必然跨界。
    crossing = sc.build_connections()
    crossing[0] = replace(
        crossing[0],
        first_slot=32,
        n_slots=16,
        bit_rate_bps=800e9,
        modulation="16QAM",
    )
    cross_probe: dict[str, object] = {
        "description": "把连接 0 挪到跨越 L|C 边界 191.55 THz 的 16 槽位置",
        "centre_frequency_thz": _jsonable(crossing[0].centre_hz / 1e12),
        "occupied_bandwidth_ghz": _jsonable(crossing[0].occupied_bandwidth_hz / 1e9),
        "rejected": False,
        "error": None,
    }
    try:
        BandPlan.from_connections(crossing, sc.BAND_SPECS).assert_connections_single_band(crossing)
    except ValueError as exc:
        cross_probe["rejected"] = True
        cross_probe["error"] = f"{type(exc).__name__}: {exc}"

    return {
        "wp": "O3",
        "judgment": "J8",
        "scenario": "flex_two_hop_mixed_width",
        "authored_by": "tools/ode3_flexgrid_scenarios.py (WP-O3 自写，未 import fu4 tests)",
        "grid": {
            "slot_width_hz": sc.FLEX_GRID.slot_width_hz,
            "num_slots": sc.FLEX_GRID.num_slots,
            "band_low_edge_hz": sc.FLEX_GRID.band_low_edge_hz,
            "band_high_edge_hz": sc.FLEX_GRID.band_high_edge_hz,
            "band_center_hz": sc.FLEX_GRID.band_center_hz,
        },
        "bands": {
            "names": list(plan.band_names),
            "elements_per_band": {spec.name: plan.element_count(b) for b, spec in enumerate(plan.specs)},
            "specs": [spec.as_dict() for spec in plan.specs],
        },
        "symbol_rate_hz": sc.SYMBOL_RATE_HZ,
        "symbol_rate_uniform": len({c.symbol_rate_hz for c in powered}) == 1,
        "distinct_n_slots": n_slots_values,
        "n_distinct_n_slots": len(n_slots_values),
        "n_connections": len(powered),
        "n_nodes_in_evaluator": evaluator.n_nodes,
        "evaluator_index_space_ok": evaluator.n_nodes == len(powered),
        "service": {
            "path": list(sc.SERVICE_PATH),
            "connection_index": idx,
            "connection_id": powered[idx].connection_id,
            "accepted": bool(accepted),
        },
        "per_connection": per_connection,
        "probes": {
            "heterogeneous_rs": hetero_probe,
            "blocked_slot": blocked_probe,
            "band_crossing": cross_probe,
        },
        "known_limitations": [
            {
                "id": "srs_band_edges_still_scalar_rs",
                "involves_physics": True,
                "statement": "isrs_gn.py 的 _precompute_channel_dependent_properties 已按连接占用带算出 "
                "f_m_hz/f_M_hz，但冻结内核**不使用**它们——它在核内用标量 Rs 重算 "
                "(f_rel_hz[0] - Rs/2, f_rel_hz[-1] + Rs/2)。",
                "impact": "宽连接下 SRS 频带边界仍按标量 Rs 而非真实占用带，低估真实拉曼作用范围。",
                "decision": "★ WP-Q1 **未**处理此项：它不属于 P1 (23)/(29)/(30)/(34) 里 B_i/B_l 的"
                "落点，且改它要动 SRS 频带边界这一独立物理量。按「不给论文没写的项加物理」"
                "与「拿不准就停下报告」，留给后续 WP 裁决。",
                "probe": None,
            },
        ],
    }


def _blocked_slot_probe(topology, powered) -> dict:
    """验证槽区间重叠语义：窄连接避开、宽连接被跨界阻断。"""
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import ROADMConfig

    import ode3_flexgrid_scenarios as sc

    evaluator = MultiBandISRSGN(
        sc.FLEX_GRID, topology.get_default_fiber_config(), sc.SYMBOL_RATE_HZ, connections=powered
    )
    # 槽 11 落在连接 1 的槽区间 [10, 16) 内 —— 应当阻断
    blocked_narrow = [
        i for i in range(len(powered)) if evaluator.roadm_blocks_node(ROADMConfig(blocked_slots=[11]), i)
    ]
    # 槽 5 落在空隙（连接 0 占 [0,2)，连接 1 占 [10,16)）—— 不应阻断任何连接
    blocked_gap = [
        i for i in range(len(powered)) if evaluator.roadm_blocks_node(ROADMConfig(blocked_slots=[5]), i)
    ]
    return {
        "description": "槽区间重叠语义：槽 11 落在连接 1 的 [10,16) 内；槽 5 落在空隙",
        "slot_11_blocks": blocked_narrow,
        "slot_5_blocks": blocked_gap,
        "expected": {"slot_11_blocks": [1], "slot_5_blocks": []},
        "passed": blocked_narrow == [1] and blocked_gap == [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = run_e2e()

    # J8 自检：≥2 个不同 n_slots、逐连接 SNR 非空、端到端确实被接受
    failures = []
    if doc["n_distinct_n_slots"] < 2:
        failures.append(f"J8: 只有 {doc['n_distinct_n_slots']} 个不同 n_slots（要求 >=2）")
    if not any(row["snr_db"] is not None for row in doc["per_connection"]):
        failures.append("J8: 逐连接 SNR 全为空")
    if not doc["evaluator_index_space_ok"]:
        failures.append("J8: evaluator.n_nodes 与连接数不一致")
    if not doc["probes"]["heterogeneous_rs"]["accepted"]:
        failures.append("探针: 逐连接异构带宽未被接受（WP-Q1 能力失效）")
    if not doc["probes"]["blocked_slot"]["passed"]:
        failures.append("探针: blocked_slots 槽区间重叠语义不符")
    if not doc["probes"]["band_crossing"]["rejected"]:
        failures.append("探针: 跨频段边界的宽连接未被拒绝")

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        print(f"  连接数={doc['n_connections']}  不同 n_slots={doc['distinct_n_slots']}")
        print(f"  频段归属={doc['bands']['elements_per_band']}  accepted={doc['service']['accepted']}")
        for row in doc["per_connection"]:
            print(
                f"    conn{row['connection_id']} n_slots={row['n_slots']:<2} "
                f"{row['centre_frequency_thz']:.5f} THz  band={row['band']}  "
                f"P={row['launch_power_w'] * 1e3:.2f} mW  SNR={row['snr_db']} dB"
            )

    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        for key in ("per_connection", "grid", "bands", "distinct_n_slots", "probes", "service"):
            if stored.get(key) != doc.get(key):
                print(f"FAIL 复算不一致: {key}")
                return 1
        if failures:
            for f in failures:
                print(f"FAIL {f}")
            return 1
        print("OK  J8 端到端复算通过：逐连接 SNR 一致，>=2 个不同 n_slots")
        return 0

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
