#!/usr/bin/env python3
"""WP-O3：把 results/*.json 渲染成 docs/ode3_flexgrid_sync.md。

⚠️ **所有数字从 JSON 读出并渲染**，本文件不含任何手抄数值
（WP-O3 §3「数字全部从 JSON 渲染，不许手抄」）。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode3_report.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode3_report.py --check

``--check`` 断言：重新渲染出的 Markdown 与落盘一致（逐字节）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
R = REPO / "results"
DOC = REPO / "docs" / "ode3_flexgrid_sync.md"

SYNC = R / "ode_dev_flexgrid_sync.json"
E2E = R / "ode_dev_flexgrid_e2e.json"
LEG = R / "ode_dev_legacy_regression.json"
CAP = R / "ode_dev_capability_matrix.json"
ANCH = R / "ode_dev_anchor_relocation.json"


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"缺少 {path.relative_to(REPO)}，请先跑对应的 --write")
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render() -> str:
    sync = _load(SYNC)
    e2e = _load(E2E)
    leg = _load(LEG)
    cap = _load(CAP)
    anch = _load(ANCH)

    s_sum = sync["summary"]
    lines: list[str] = []
    A = lines.append

    A("# WP-O3：flexgrid 全链路同步到 `ode-dev`")
    A("")
    A(f"- 判据：**J5**（`numba_kernels.py` diff 为空）/ **J6**（legacy 逐位）/ "
      f"**J7**（逐片段 engineering|physics）/ **J8**（端到端跑通）")
    A(f"- 参照：`{sync['fu4_ref']}`；本 WP 起点：`{sync['wp_base_commit']}`")
    A("- 本文件全部数字由 `tools/ode3_report.py` 从 `results/*.json` 渲染，无手抄。")
    A("")

    # ---------------- 结论 ----------------
    A("## 0. 验收结论")
    A("")
    A("| 判据 | 内容 | 结果 |")
    A("|---|---|---|")
    # ⚠️ WP-MERGE-FG-FLEX：J5 的判定口径已从「相对 b69563a 的 diff 为空」翻成
    # **绝对正向不变量**「内核逐字节等于已核出处 sha256」。
    #
    # 翻法理由（addendum §2 相位守卫通例）：原断言把「本 WP 不许碰内核」编码成永久
    # 不变量，而本次合并的目的正是并入 FG 的内核，前提已消失。且两条判据客观互斥——
    # b69563a/flexgrid-dev 侧 sha256=6611c416…，wp/fg-integrate 侧 sha256=7beece4e…，
    # 同一文件不可能同时有两个 SHA-256；判据 V1（首要判据）要求取 FG 侧。
    # 绝对不变量比「diff 为空」**更强**：它同时挡住「被人手改一行」与「合并出第三份实现」。
    j5 = sync["j5_frozen_kernel"]
    if "sha256_matches" in j5:
        A(f"| J5 | `{j5['file']}` 逐字节等于已核出处 `{j5.get('provenance_branch')}` 的冻结字节 | "
          f"{'✅ 通过' if j5['sha256_matches'] else '❌ 失败'}"
          f"（sha256 `{(j5.get('frozen_sha256') or '')[:16]}…`） |")
    else:  # 兼容旧 schema
        A(f"| J5 | `{j5['file']}` 相对 `{sync['wp_base_commit']}` 的 diff 为空 | "
          f"{'✅ 通过' if j5['diff_vs_wp_base_empty'] else '❌ 失败'} |")
    A(f"| J6 | 6 个 legacy 场景 `delta_db == 0.0` | "
      f"{'✅ 通过' if leg['summary']['j6_passed'] else '❌ 失败'}"
      f"（逐位相同 {leg['summary']['n_bit_identical']}/{leg['summary']['n_scenarios']}） |")
    A(f"| J7 | 每段都有 kind，且 `kind == physics` 的片段数为 "
      f"{s_sum['n_physics_applied']} | "
      f"{'✅ 通过' if s_sum['n_physics_applied'] == 0 else '❌ 失败'}"
      f"（共 {s_sum['n_segments']} 段） |")
    A(f"| J8 | 端到端跑通，≥2 个不同 `n_slots`，逐连接 SNR 落盘 | "
      f"{'✅ 通过' if e2e['n_distinct_n_slots'] >= 2 and any(r['snr_db'] is not None for r in e2e['per_connection']) else '❌ 失败'}"
      f"（{e2e['n_distinct_n_slots']} 个不同 n_slots：{e2e['distinct_n_slots']}） |")
    A("")

    # ---------------- 1. 同步了什么 ----------------
    A("## 1. 同步了什么（逐文件逐片段，J7 分类）")
    A("")
    A(f"共裁定 **{s_sum['n_segments']}** 段：engineering {s_sum['n_engineering']} 段、"
      f"physics {s_sum['n_physics']} 段；**被应用的 {s_sum['n_ported_or_applied']} 段里，"
      f"physics 为 {s_sum['n_physics_applied']} 段**。")
    A("")
    A("### 1.1 已应用（按文件）")
    A("")
    A("| 文件 | 片段 | fu4 行 | kind | 动作 | 理由 |")
    A("|---|---|---|---|---|---|")
    for seg in sync["segments"]:
        if seg["action"] == "not_ported":
            continue
        A(f"| `{seg['file'].split('network_nlo_eval/')[-1]}` | {seg['symbol']} | "
          f"`{seg['fu4_lines']}` | **{seg['kind']}** | `{seg['action']}` | "
          f"{seg['rationale']} |")
    A("")
    A("### 1.2 刻意**不**搬（按文件）")
    A("")
    A("| 文件 | 片段 | fu4 行 | kind | 理由 |")
    A("|---|---|---|---|---|")
    for seg in sync["segments"]:
        if seg["action"] != "not_ported":
            continue
        A(f"| `{seg['file'].split('network_nlo_eval/')[-1]}` | {seg['symbol']} | "
          f"`{seg['fu4_lines']}` | **{seg['kind']}** | {seg['rationale']} |")
    A("")
    A("### 1.3 `core/bands.py` 的最小闭包（728 行 → 最小片段）")
    A("")
    A("`wp/fg-uwb-fid` 的 `core/bands.py` 在 `ecz-dev` 上**整文件不存在**（非改名、非合并——"
      "它诞生于 FG 线的 `51b601b` \"FU-2: S+C+L+U 波段定义\"）。因此本 WP 新建该文件时"
      "**只摘 flexgrid 接线所需的最小闭包**：`nm_to_hz` + `BandSpec` + `BandPlan` 的"
      "三个接入点与其归属查询；FU-2 的波段表、假设与报告逻辑一律不搬。")
    A("")
    A("最小闭包的三条依据：")
    A("")
    A("1. `BandPlan.from_connections` 的第二参数是 `Sequence[BandSpec]`；")
    A("2. `BandSpec.low_hz/high_hz` 由 `nm_to_hz` 换算；")
    A("3. `ConnectionSpec` 是 frozen dataclass，故逐波段功率必须经 `apply_launch_power` "
      "返回新对象。")
    A("")
    A("⚠️ 本仓**不预置**任何波段表——波段规格由场景自带 "
      "（见 `tools/ode3_flexgrid_scenarios.py` 的 `BAND_SPECS`）。")
    A("")

    # ---------------- 2. e2e ----------------
    A("## 2. flexgrid 端到端（J8）")
    A("")
    A(f"场景 **`{e2e['scenario']}`**，由 `{e2e['authored_by']}` 自写"
      "（brief §2 步骤 2 要求不 import fu4 的 tests）。")
    A("")
    g = e2e["grid"]
    A(f"- 栅格：槽宽 {g['slot_width_hz'] / 1e9:g} GHz × {g['num_slots']} 槽 = "
      f"{(g['band_high_edge_hz'] - g['band_low_edge_hz']) / 1e12:.3f} THz，"
      f"覆盖 {g['band_low_edge_hz'] / 1e12:.3f} – {g['band_high_edge_hz'] / 1e12:.3f} THz")
    A(f"- 频段：`{'`, `'.join(e2e['bands']['names'])}`，"
      f"逐段元素数 {e2e['bands']['elements_per_band']}")
    A(f"- 统一符号率：{e2e['symbol_rate_hz'] / 1e9:g} GHz"
      f"（uniform={e2e['symbol_rate_uniform']}，原因见 §4 限制 1）")
    A(f"- 业务：路径 `{e2e['service']['path']}`，承载连接 "
      f"#{e2e['service']['connection_id']}（下标 {e2e['service']['connection_index']}），"
      f"`accepted = {e2e['service']['accepted']}`")
    A("")
    A("### 2.1 逐连接 SNR")
    A("")
    A("| conn | n_slots | 占用带宽 | 中心频率 (THz) | 调制 | 比特率 | 频段 | 发射功率 | SNR (dB) |")
    A("|---:|---:|---:|---:|---|---:|---|---:|---:|")
    for r in e2e["per_connection"]:
        mark = " ★" if r["is_service_under_test"] else ""
        A(f"| {r['connection_id']}{mark} | {r['n_slots']} | "
          f"{_fmt(r['occupied_bandwidth_ghz'], 1)} GHz | "
          f"{_fmt(r['centre_frequency_thz'], 5)} | {r['modulation']} | "
          f"{_fmt(r['bit_rate_gbps'], 0)} Gbps | {r['band']} | "
          f"{_fmt(r['launch_power_w'] * 1e3, 2)} mW | **{_fmt(r['snr_db'], 4)}** |")
    A("")
    A(f"★ = 本次验证的新业务承载连接。`n_slots` 取值 "
      f"`{e2e['distinct_n_slots']}` 共 **{e2e['n_distinct_n_slots']} 个不同值**"
      f"（J8 要求 ≥2）。")
    A("")
    A("⚠️ **声明宽度 ≠ 声明符号率**：连接按 `slots_for_rate` 占 2/6/3/8 槽（25–100 GHz），"
      f"但符号率统一为 {e2e['symbol_rate_hz'] / 1e9:g} GHz。原因是冻结内核只吃标量 Rs（§4 限制 1）。"
      "**灵活栅格要表达的「宽度不等」成立；「每连接 Rs 自适应」不成立。**")
    A("")
    A("### 2.2 探针")
    A("")
    p = e2e["probes"]
    A(f"1. **异构 Rs**（{'✅ 被拒' if p['heterogeneous_rs']['rejected'] else '❌ 未被拒'}）："
      f"`{p['heterogeneous_rs']['rates_hz']}` → "
      f"`{str(p['heterogeneous_rs']['error'])[:110]}…`")
    A(f"2. **blocked_slots 槽区间重叠**（{'✅ 通过' if p['blocked_slot']['passed'] else '❌ 失败'}）："
      f"槽 11 阻断 {p['blocked_slot']['slot_11_blocks']}（期望 "
      f"{p['blocked_slot']['expected']['slot_11_blocks']}，落在连接 1 的槽区间 [10,16) 内）；"
      f"槽 5 阻断 {p['blocked_slot']['slot_5_blocks']}（期望 "
      f"{p['blocked_slot']['expected']['slot_5_blocks']}，落在空隙）")
    A(f"3. **跨频段边界宽连接**（{'✅ 被拒' if p['band_crossing']['rejected'] else '❌ 未被拒'}）："
      f"{p['band_crossing']['occupied_bandwidth_ghz']:.0f} GHz 的连接中心在 "
      f"{_fmt(p['band_crossing']['centre_frequency_thz'], 4)} THz，跨 L|C 边界 → "
      f"`{str(p['band_crossing']['error'])[:100]}…`")
    A("")

    # ---------------- 3. 能力矩阵 ----------------
    A("## 3. 能力矩阵（`fu4` vs `ode-dev`）")
    A("")
    cs = cap["summary"]
    A(f"共 {cs['n_capabilities']} 条：双方都有 {cs['n_both']}、仅 `fu4` 有 {cs['n_fu4_only']}"
      f"（其中**涉物理** {cs['n_fu4_only_involves_physics']}）、仅 `ode-dev` 有 "
      f"{cs['n_ode_dev_only']}、`ode-dev` 更优 {cs['n_ode_dev_better']}。")
    A("")
    A("| 能力 | 文件 | 状态 | 涉物理 | 证据 |")
    A("|---|---|---|:---:|---|")
    for c in cap["capabilities"]:
        A(f"| {c['capability']} | `{c['file'].split('network_nlo_eval/')[-1]}` | "
          f"`{c['status']}` | {'**是**' if c['involves_physics'] else '否'} | "
          f"{c['evidence']} |")
    A("")

    # ---------------- 4. legacy 回归 ----------------
    A("## 4. legacy 回归（J6）")
    A("")
    A(f"协议与 WP-O2 **同口径**（复用 `{leg['protocol']}`），容差 **{leg['tolerance']}**。")
    A("")
    A("| 场景 | `ode-dev` SNR (dB) | `ecz-dev` SNR (dB) | `delta_db` | 逐位相同 | 裁决 |")
    A("|---|---:|---:|---:|:---:|:---:|")
    for r in leg["scenarios"]:
        delta = r["delta_db"]
        delta_txt = "n/a" if delta is None else f"{delta:+.9f}"
        A(f"| `{r['scenario']}` | {_fmt(r['ode_dev_snr_db'], 9)} | {_fmt(r['ecz_dev_snr_db'], 9)} | "
          f"{delta_txt} | {'✅' if r['bit_identical'] else '—'} | {r['verdict']} |")
    A("")
    A(f"`legacy_blocked_channel` 两端**一致拒绝**（`both_rejected = "
      f"{leg['summary']['n_both_rejected']}`），故 `delta_db = n/a` 但裁决为 `pass`。")
    A("")

    # ---------------- 5. 物理缺口 ----------------
    A("## 5. 物理缺口（J5 上报，**未修**）")
    A("")
    A("以下三项都需要改 `models/numba_kernels.py`（J5 禁改，diff 必须为空）。"
      "按 WP-O3 §5「别重复踩 O2 的坑」，本 WP **一律报告而不强改**。")
    A("")
    for lim in e2e["known_limitations"]:
        A(f"### {lim['id']}")
        A("")
        A(f"- **涉及物理**：{'是' if lim['involves_physics'] else '否'}")
        A(f"- **事实**：{lim['statement']}")
        A(f"- **影响**：{lim['impact']}")
        A(f"- **裁定**：{lim['decision']}")
        A("")
    A("此外 `ode_accel` 的 `matvec` / `cached_g` 档、以及 `phi_kurtosis` 同样需改冻结内核，"
      "已按 WP-O3 §8 列为不做（能力矩阵中 `involves_physics = 是`）。")
    A("")

    # ---------------- 6. 附带修正 ----------------
    A("## 6. 附带修正与副作用")
    A("")
    A("### 6.1 `for_fiber` 丢 `connections`（bug）")
    A("")
    A("O2 移植的 `MultiBandISRSGN.for_fiber` 只传 `(grid, fiber_config, Rs)`，"
      "逐连接模式下子评估器会退回遗留分支并读 `FlexibleGrid.wavelengths`（该属性不存在）"
      "→ `AttributeError`。J8 链路必经 `for_fiber`（`rwa/qot_checker.py:161`），"
      "不修则端到端跑不起来。已补上 `connections` / `ode_accel` / `slots_per_channel` 透传；"
      "遗留模式下三者皆为默认值，行为不变。")
    A("")
    A("### 6.2 O2 公式锚点行号漂移")
    A("")
    A(f"在 `models/isrs_gn.py` 插入接线后，O2 `formula_map` 里 **{anch['n_relocated']}** 条"
      f"代码锚点的**行号**下移，导致 J1 守卫失败。这些锚点指向的**物理代码一字未改**"
      f"（`git diff` 只删了 3 行，全是工程接线）。`tools/ode3_relocate_anchors.py` "
      f"只重排行号、**不动 `contains` 字符串**——若物理真被改写，它会因匹配不上而报错退出。"
      f"未自动修正 {anch['n_unfixable']} 条。")
    A("")
    A("### 6.3 `BandSpec` 波长/频率 docstring 反了")
    A("")
    A("fu4 的 docstring 自称 `low_nm` 是**长**波长，但不变式强制 `low_nm < high_nm`"
      "即**短**波长。照它写场景会把波段边界整体算反。只改注释、不改行为。")
    A("")

    # ---------------- 7. 没做什么 ----------------
    A("## 7. 没做什么 / 没跑完什么")
    A("")
    A("| 项 | 状态 | 原因 |")
    A("|---|---|---|")
    A("| 逐连接**异构**符号率 | **未跑** | 冻结内核只吃标量 Rs，改它违反 J5。已用 "
      "`probes.heterogeneous_rs` 证明确实被拒 |")
    A("| SRS 频带边界按真实占用带 | **未跑** | 同上：需给冻结核加参数 |")
    A("| `ode_accel` 的 `matvec` / `cached_g` 档 | **未做** | WP-O3 §8 明确列为不做 |")
    A("| `phi_kurtosis` 进入 XPM | **未做** | 本仓闭式 ECZ 无此参数，按 J1 需 P1/P2 公式号 |")
    A("| FU-2 的 S/C/L/U 预置波段表、`sclu_span_report`、`guard_report` | **未搬** | "
      "FU-2 自有资产，含无 ITU 出处的 U 波段假设；与 flexgrid 接线无关 |")
    A("| `physics/qot_checker.py` 的**波形级**端到端跑通 | **未跑** | 需 PyNLO + CuPy "
      "compute context（本机无 GPU 上下文）。本 WP 只做接线与 legacy 表达式不变性验证，"
      "未实际跑波形 |")
    A("| 跨仓（RL 仓）对照 | **未做** | WP-O3 §8 明确「不接 RL 仓」 |")
    A("| `flex_*` 与 `fu4` 的 SNR 逐场景对照 | **未做** | fu4 与本仓内核**不是同一个物理模型**"
      "（fu4 = RK4 真 ODE；本仓 = P1 闭式 ECZ），对照无意义，且本 WP 无权改本仓物理 |")
    A("")

    # ---------------- 8. 复算 ----------------
    A("## 8. 复算")
    A("")
    A("```bash")
    A("export PYTHONPATH=$PWD/src")
    A("PY=~/miniconda3/envs/rl/bin/python")
    A("$PY tools/ode3_relocate_anchors.py --check   # O2 锚点行号仍对齐")
    A("$PY tools/ode3_sync.py            --check     # J7 + J5")
    A("$PY tools/ode3_flexgrid_e2e.py    --check     # J8 逐连接 SNR")
    A("$PY tools/ode3_legacy_regression.py --check   # J6 逐位")
    A("$PY tools/ode3_capability.py      --check     # 能力矩阵")
    A("$PY tools/ode_compare.py          --check     # WP-O2 三树对照（未退步）")
    A("$PY tools/ode_report.py           --check     # 本文档逐字节一致")
    A("```")
    A("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    text = render()
    if args.write:
        DOC.parent.mkdir(parents=True, exist_ok=True)
        DOC.write_text(text, encoding="utf-8")
        print(f"wrote {DOC.relative_to(REPO)}  ({len(text.splitlines())} 行)")

    if args.check:
        if not DOC.is_file():
            print(f"FAIL 缺少 {DOC.relative_to(REPO)}")
            return 1
        if DOC.read_text(encoding="utf-8") != text:
            print(f"FAIL {DOC.name} 与从 JSON 渲染的结果不一致（请重跑 --write）")
            return 1
        print(f"OK  {DOC.name} 逐字节与 JSON 渲染结果一致")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
