#!/usr/bin/env python3
"""WP-O3 步骤 3：flexgrid 能力矩阵（`wp/fg-uwb-fid` vs `ode-dev`）。

逐条回答三个问题：

* ``fu4`` 有而 ``ode-dev`` 无 —— 为什么？是否涉及物理？
* 两者都有 —— ``ode-dev`` 的是否逐字？
* ``ode-dev`` 有而 ``fu4`` 无 —— 本仓独有的东西（多为 O2 的冻结裁决）。

每条都带 ``involves_physics`` 布尔字段（WP-O3 §4 测试要求 5 强制）。
**不涉及物理的缺口**记为 ``engineering``（可后续补齐），
**涉及物理的缺口**记为 ``physics``（须走 J1/J3 论文裁定，本 WP 不得直接补）。

``--check`` 断言：每条都有 ``involves_physics``；能力矩阵覆盖 WP-O3 §0 定位的
全部 7 个 flexgrid 触点文件。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode3_capability.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode3_capability.py --check

产物：``results/ode_dev_capability_matrix.json``
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ode_dev_capability_matrix.json"
FU4_REF = "a233002"

#: WP-O3 §0 实测的 7 个 flexgrid 触点文件——能力矩阵必须全覆盖。
TOUCHPOINT_FILES = (
    "src/network_nlo_eval/core/flexgrid.py",
    "src/network_nlo_eval/network/slots.py",
    "src/network_nlo_eval/core/bands.py",
    "src/network_nlo_eval/models/isrs_gn.py",
    "src/network_nlo_eval/network/elements.py",
    "src/network_nlo_eval/physics/qot_checker.py",
    "src/network_nlo_eval/rwa/qot_checker.py",
)

CAPABILITIES: list[dict] = [
    # ---- 数据类（O2 已逐字移植）----
    {
        "capability": "FlexibleGrid（槽位化栅格）",
        "file": "src/network_nlo_eval/core/flexgrid.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "逐字移植（O2）。centre_hz 用 n_slots/2 区间中心，非点阵中心。",
    },
    {
        "capability": "ConnectionSpec（冻结连接规格）",
        "file": "src/network_nlo_eval/core/flexgrid.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "逐字移植（O2）。n_slots 与 slots_for_rate 一致性由 __post_init__ 强制。",
    },
    {
        "capability": "slots.py 槽工具（slots_for_rate / connection_overlaps_slots / "
        "resolve_blocked_slots / migrate_roadm_blocked_slots）",
        "file": "src/network_nlo_eval/network/slots.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "逐字移植（O2）。",
    },
    # ---- 本 WP 接线的下游 ----
    {
        "capability": "MultiBandISRSGN(connections=...) 逐连接模式",
        "file": "src/network_nlo_eval/models/isrs_gn.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "O2 已接入；O3 补齐 for_fiber 透传（否则逐连接模式下 AttributeError）。",
    },
    {
        "capability": "ROADMConfig.blocked_slots（12.5 GHz 粒度权威阻断）",
        "file": "src/network_nlo_eval/network/elements.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "O3 应用。默认空，legacy 不变。",
    },
    {
        "capability": "槽区间重叠阻断语义（connection_overlaps_slots 单一实现）",
        "file": "src/network_nlo_eval/models/isrs_gn.py",
        "status": "ode_dev_better",
        "involves_physics": False,
        "evidence": "fu4 在 rwa/physics/isrs_gn 三处各写一遍（physics 侧还靠私有属性 "
        "_slots_per_channel_hint）；O3 收敛为 evaluator.roadm_blocks_node 一处。",
    },
    {
        "capability": "QoTValidator 逐连接 SNR 装配",
        "file": "src/network_nlo_eval/rwa/qot_checker.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "O3 应用 num_nodes / _is_blocked / _node_frequency_and_rate；"
        "booster ASE 公式一字未动。",
    },
    {
        "capability": "PyNLO 波形模型逐连接频率 + 最宽连接 WSS 通带",
        "file": "src/network_nlo_eval/physics/qot_checker.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "O3 应用。legacy 分支的 passband/occupied_width 表达式逐字不变。",
    },
    {
        "capability": "BandPlan.from_connections / assert_connections_single_band / "
        "apply_launch_power",
        "file": "src/network_nlo_eval/core/bands.py",
        "status": "both",
        "involves_physics": False,
        "evidence": "O3 最小闭包移植（fu4 728 行 -> 本仓只取 flexgrid 接线片段）。",
    },
    # ---- fu4 有、ode-dev 无 ----
    {
        "capability": "S/C/L/U 预置波段表（CL/CLS/CLU/SCLU_BAND_SPECS）",
        "file": "src/network_nlo_eval/core/bands.py",
        "status": "fu4_only",
        "involves_physics": False,
        "evidence": "FU-2 自有资产，含 4 条无 ITU 出处的 U 波段假设。与 flexgrid 接线无关，"
        "本 WP 不搬。场景可自带频段规格。",
    },
    {
        "capability": "sclu_span_report / guard_report（跨距与保护间隔口径）",
        "file": "src/network_nlo_eval/core/bands.py",
        "status": "fu4_only",
        "involves_physics": False,
        "evidence": "FU-2 自有报告逻辑，依赖上条预置波段表。",
    },
    {
        "capability": "ROADMConfig.blocked_slot_set()",
        "file": "src/network_nlo_eval/network/elements.py",
        "status": "fu4_only",
        "involves_physics": False,
        "evidence": "实测本仓零调用方（死代码）。语义由 roadm_blocks_node 覆盖。",
    },
    {
        "capability": "QoTValidator._roadm_terms（把 inline 装配抽成函数）",
        "file": "src/network_nlo_eval/rwa/qot_checker.py",
        "status": "fu4_only",
        "involves_physics": False,
        "evidence": "纯重构（PLAN §0 已证语义为零）。故意不搬，使 legacy 装配算式保持逐字不变。",
    },
    {
        "capability": "逐连接异构符号率（symbol_rate_hz 数组参数）",
        "file": "src/network_nlo_eval/models/numba_kernels.py",
        "status": "fu4_only",
        "involves_physics": True,
        "evidence": "★ 物理缺口。fu4 核接受 (N,) Rs 并据此算 ASE；本仓冻结核只吃标量。"
        "改它 = 改 numba_kernels.py = 违反 J5。O2 已显式拒绝，O3 沿用。"
        "证据：e2e.probes.heterogeneous_rs。",
    },
    {
        "capability": "SRS 频带边界按真实连接占用带（f_m_hz/f_M_hz）",
        "file": "src/network_nlo_eval/models/numba_kernels.py",
        "status": "fu4_only",
        "involves_physics": True,
        "evidence": "★ 物理缺口。本仓 Python 侧已按占用带算出 f_m_hz/f_M_hz，但冻结核在核内"
        "用标量 Rs 重算，故该修正是死代码。让它生效需给内核加参数 = J5。"
        "证据：e2e.known_limitations.srs_band_edges_still_scalar_rs。",
    },
    {
        "capability": "ode_accel: matvec / cached_g 加速档",
        "file": "src/network_nlo_eval/models/numba_kernels.py",
        "status": "fu4_only",
        "involves_physics": True,
        "evidence": "★ 物理缺口（求和次序改变 ⇒ 非逐位相等）。需改冻结内核，"
        "WP-O3 §8 明确列为不做。fu4 侧 NLI 张量公式出处是其 docstring 的 'Eq. 21'，"
        "P1/P2 均无此式，O2 已按 J1/J3 拒绝。",
    },
    {
        "capability": "phi_kurtosis（64QAM 峰度进入 XPM）",
        "file": "src/network_nlo_eval/models/numba_kernels.py",
        "status": "fu4_only",
        "involves_physics": True,
        "evidence": "本仓闭式 ECZ 模型无此参数；引入它等于改变 NLI 模型，"
        "按 J1 需 P1/P2 公式号支撑，O2 未取得，故不移植。",
    },
    {
        "capability": "numba 缺失时的 njit 降级垫片",
        "file": "src/network_nlo_eval/models/numba_kernels.py",
        "status": "ode_dev_only",
        "involves_physics": False,
        "evidence": "本仓独有：ImportError 时 njit 退化为恒等装饰器，让无 JIT 的最小环境"
        "也能跑统计模型。O2 已带（ecz-dev 既有）。",
    },
    {
        "capability": "闭式 ECZ GN 模型（按 P1 公式号逐条可 grep）",
        "file": "src/network_nlo_eval/models/numba_kernels.py",
        "status": "ode_dev_only",
        "involves_physics": True,
        "evidence": "本仓物理底：P1 §II.B 扩展 CZ / §III 闭式 GN。fu4 走的是 RK4 真 ODE 传播。"
        "两棵树内核**不是同一个物理模型**，故 numba_kernels.py 的 diff 恒非空——"
        "J5 要求的是本 WP 相对自身起点的 diff 为空。",
    },
]


def _fu4_lines(rel: str) -> int | None:
    proc = subprocess.run(
        ["git", "show", f"{FU4_REF}:{rel}"], capture_output=True, text=True, cwd=str(REPO)
    )
    if proc.returncode != 0:
        return None
    return len(proc.stdout.splitlines())


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    errs: list[str] = []
    rows = []
    for cap in CAPABILITIES:
        rel = cap["file"]
        path = REPO / rel
        row = dict(cap)
        row["ode_dev_lines"] = len(path.read_text(encoding="utf-8").splitlines()) if path.is_file() else None
        row["fu4_lines"] = _fu4_lines(rel)
        row["ode_dev_sha256"] = _sha(path)
        if not isinstance(cap.get("involves_physics"), bool):
            errs.append(f"[能力矩阵] {cap['capability']} 缺 involves_physics 布尔字段")
        if cap.get("status") not in ("both", "fu4_only", "ode_dev_only", "ode_dev_better"):
            errs.append(f"[能力矩阵] {cap['capability']} status 非法: {cap.get('status')!r}")
        rows.append(row)

    covered = {r["file"] for r in rows}
    missing = [f for f in TOUCHPOINT_FILES if f not in covered]
    for f in missing:
        errs.append(f"[能力矩阵] 未覆盖 WP-O3 §0 的触点文件 {f}")

    doc = {
        "wp": "O3",
        "judgment": "J7 / 步骤 3",
        "fu4_ref": FU4_REF,
        "touchpoint_files": list(TOUCHPOINT_FILES),
        "summary": {
            "n_capabilities": len(rows),
            "n_fu4_only": sum(1 for r in rows if r["status"] == "fu4_only"),
            "n_ode_dev_only": sum(1 for r in rows if r["status"] == "ode_dev_only"),
            "n_both": sum(1 for r in rows if r["status"] == "both"),
            "n_ode_dev_better": sum(1 for r in rows if r["status"] == "ode_dev_better"),
            "n_involves_physics": sum(1 for r in rows if r["involves_physics"]),
            "n_fu4_only_involves_physics": sum(
                1 for r in rows if r["status"] == "fu4_only" and r["involves_physics"]
            ),
            "n_files_covered": len(covered),
        },
        "capabilities": rows,
        "errors": errs,
    }

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        print(f"  能力 {len(rows)} 条；fu4_only {doc['summary']['n_fu4_only']}"
              f"（其中涉物理 {doc['summary']['n_fu4_only_involves_physics']}）；"
              f"覆盖触点文件 {doc['summary']['n_files_covered']}/{len(TOUCHPOINT_FILES)}")

    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        if stored.get("capabilities") != rows:
            print("FAIL 能力矩阵与落盘不一致")
            return 1
        if errs:
            for e in errs:
                print(f"FAIL {e}")
            return 1
        print(f"OK  能力矩阵复算通过：{len(rows)} 条全有 involves_physics，"
              f"覆盖 {doc['summary']['n_files_covered']}/{len(TOUCHPOINT_FILES)} 个触点文件")
        return 0

    if errs:
        for e in errs:
            print(f"FAIL {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
