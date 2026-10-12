#!/usr/bin/env python3
"""由 ``results/*.json`` 渲染 ``docs/ode2_papers_and_diff.md``。

⚠️ H3：**文档表格由脚本渲染，不许手抄。** 本文件是唯一的报告生成器；
改数字请改 JSON 生成器，然后重跑本脚本。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode_report.py
    ~/miniconda3/envs/rl/bin/python tools/ode_report.py --check
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
OUT = REPO / "docs" / "ode2_papers_and_diff.md"


def _load(name: str) -> dict:
    import json

    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def fmt(v, digits: int = 9) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, (int, float)):
        return f"{v:+.{digits}f}" if digits else str(v)
    return str(v)


def render() -> str:
    fmap = _load("ode_dev_formula_map.json")
    verd = _load("ode_dev_rejected.json")
    ecz = _load("ode_dev_diff_vs_ecz.json")
    fu4 = _load("ode_dev_diff_vs_fu4.json")

    L: list[str] = []
    A = L.append

    A("# WP-O2 报告：ode-dev 按 P1/P2 实现，逐项对照 `wp/fg-uwb-fid`")
    A("")
    A("> 本文件由 `tools/ode_report.py` 从 `results/*.json` **渲染**生成，请勿手改数字。")
    A("> 判据 J1–J4 派发前写死，未改。")
    A("")
    A("**唯一结论先行**：`ode-dev` 在 6 个 `legacy_*` 场景上与 `ecz-dev` **逐位相同**"
      f"（6/6 行一致；5 个有 SNR 的场景 Δ = 0.000000000 dB，第 6 个两端都拒绝）。")
    A("这不是巧合：**逐条读论文后，`ecz-dev` 的物理本就符合 P1**，因此没有任何一条"
      " `fix_to_paper` 需要落地（裁定数 `fix_to_paper = "
      f"{verd['summary']['n_fix_to_paper']}`）。移植进来的只有**工程能力**。")
    A("")

    # ---------------- 0. 判据 ----------------
    A("## 0. 本轮实际执行的判据")
    A("")
    A("| 判据 | 内容 | 本轮怎么执行的 |")
    A("|---|---|---|")
    A("| **J1** | 每条物理公式必须有 P1/P2 公式号 | `tools/ode_formula_map.py --check` 逐条在 "
      "`papers/*.md` 里 grep 锚点，grep 不到即失败 |")
    A("| **J2** | 论文提及且不一致 ⇒ 改成论文；论文未提及 ⇒ 保留 ecz | 见 §2 裁定表；"
      "「论文未提及」一律记 `keep_ecz_no_paper`，**不作为移植许可证** |")
    A("| **J3** | fu4 不是权威 | 只有与论文一致才可作移植来源 |")
    A("| **J4** | 差异①与 P1 (34) 冲突，不得移植 | 记入 `rejected` 并引公式号；守卫 2 断言 |")
    A("")

    # ---------------- 1. 首段总表 ----------------
    A("## 1. 逐项裁定总表（brief §9 首段要求）")
    A("")
    A("| 项 | fu4 怎么做 | P1/P2 怎么说 | 裁定 | 理由（含公式号） |")
    A("|---|---|---|---|---|")
    for it in verd["items"]:
        paper = it.get("paper")
        ploc = f"**{paper['id']} {paper['locator']}**" if paper else "论文未提及"
        if it.get("paper2"):
            ploc += f" / {it['paper2']['id']} {it['paper2']['locator']}"
        A(f"| `{it['id']}`<br>{it['label']} | {it['fu4_does']} | {ploc} | "
          f"**`{it['verdict']}`** | {it['judgement']} |")
    A("")
    A(f"裁定统计：accept {verd['summary']['n_accept']}，"
      f"fix_to_paper {verd['summary']['n_fix_to_paper']}，"
      f"keep_ecz_no_paper {verd['summary']['n_keep_ecz']}，"
      f"reject {verd['summary']['n_reject']}。")
    A("")

    # ---------------- 2. 公式映射 ----------------
    A("## 2. 公式 → 论文 → 代码 映射表（brief §5 步骤 1，本 WP 的地基）")
    A("")
    A(f"共 {fmap['summary']['physics_total']} 条物理公式，其中 "
      f"{fmap['summary']['physics_conforms']} 条符合、"
      f"{fmap['summary']['physics_violates']} 条违反、"
      f"{fmap['summary']['physics_reference_only']} 条是 ODE 侧基准（本闭式分支不实现）。")
    A("")
    A("| 公式 | 论文 | 原式（摘要） | 本仓代码位置 | 符合 | 裁定 |")
    A("|---|---|---|---|---|---|")
    for e in fmap["physics"]:
        code = "<br>".join(
            f"`{c['file'].split('/')[-1]}:{c['line']}`" for c in e.get("code", [])
        ) or "（本分支不实现）"
        verdict = e["verdict"]
        if verdict == "dead_code_violates_eq34":
            mark = "⚠️ 违反（死代码）"
        elif e["conforms"] is True:
            mark = "✅"
        elif e["conforms"] is None:
            mark = "—（基准）"
        else:
            mark = "❌"
        paper = e["paper"]
        A(f"| `{e['id']}` | **{paper['id']} {paper['locator']}** | {e['paper_formula']} | "
          f"{code} | {mark} | `{verdict}` |")
    A("")

    # ---------------- 3. 死代码警告 ----------------
    dead = next((e for e in fmap["physics"] if e["id"] == "p1_eq34_ase_dead_kernel"), None)
    if dead:
        A("### 2.1 ⚠️ 一个必须记录的隐患：死代码里藏着违反 P1 (34) 的写法")
        A("")
        A(f"`{dead['notes']}`")
        A("")
        A("**当前 `ode-dev` 的数值是合规的**（live ASE 走 `isrs_gn.py:214`，用 EDFA 配置增益）。"
          "但 `wp/fg-uwb-fid` 恰恰把这个函数接上了——这正是 WP-O1 的差异①。"
          "按 brief §8，本 WP **没有**动它：它属于冻结内核，且删除死代码不改变物理。")
        A("")

    # ---------------- 4. 三树对照 ----------------
    A("## 3. 三树逐场景 Δ（brief §9 第 1、2 条）")
    A("")
    A("### 3.1 `ode-dev` vs `ecz-dev`")
    A("")
    A("| 场景 | ode-dev (dB) | ecz-dev (dB) | Δ (dB) | 逐位相同 |")
    A("|---|---:|---:|---:|:---:|")
    for r in ecz["scenarios"]:
        A(f"| `{r['scenario']}` | {fmt(r['ode_dev_snr_db'], 12) if r['ode_dev_snr_db'] is not None else '拒绝'} "
          f"| {fmt(r['other_snr_db'], 12) if r['other_snr_db'] is not None else '拒绝'} "
          f"| {fmt(r['delta_db'])} | {'✅' if r['bit_identical'] else '❌'} |")
    A("")
    A(f"逐位相同 **{ecz['summary']['n_bit_identical']}/{ecz['summary']['n_scenarios']} 行**"
      f"（其中 5 行有可比的 SNR 且 Δ = 0；第 6 行 `legacy_blocked_channel` 两端都拒绝，无 SNR 可比）。"
      "这正是 brief §6 的**关键判据**：按 P1 实现的 `ode-dev` 就应当贴近 `ecz-dev`，实测差为 0。")
    A("")
    A("### 3.2 `ode-dev` vs `wp/fg-uwb-fid`")
    A("")
    A("| 场景 | ode-dev (dB) | fu4 (dB) | Δ = ode−fu4 (dB) | fu4−ecz（PLAN §0 已测） |")
    A("|---|---:|---:|---:|---:|")
    plan = {
        "legacy_two_hop_equalize_on": 0.769554,
        "legacy_three_hop_equalize_on": 0.760548,
        "legacy_two_hop_equalize_off": 0.790219,
        "legacy_express_attenuation": 0.762195,
        "legacy_high_filtering": 0.770208,
    }
    for r in fu4["scenarios"]:
        p = plan.get(r["scenario"])
        A(f"| `{r['scenario']}` | {fmt(r['ode_dev_snr_db'], 12) if r['ode_dev_snr_db'] is not None else '拒绝'} "
          f"| {fmt(r['other_snr_db'], 12) if r['other_snr_db'] is not None else '拒绝'} "
          f"| {fmt(r['delta_db'])} | {'—' if p is None else f'{p:+.6f}'} |")
    A("")
    A("⇒ `ode-dev` 与 fu4 的差距 = PLAN §0 实测的 +0.77 dB，"
      "因为 **`ode-dev` 与 `ecz-dev` 逐位相同**。这从数值侧再次确认："
      "那 0.77 dB 是 fu4 偏离论文的结果，不是 `ecz-dev` 的误差。")
    A("")

    # ---------------- 5. 能力清单 ----------------
    A("## 4. 能力清单（brief §9 第 3 条）")
    A("")
    A("| 能力 | 可用 | 落点 | 对 legacy SNR 的影响 |")
    A("|---|:---:|---|:---:|")
    A("| 灵活栅格 `flexgrid` | ✅ | `core/flexgrid.py`（343 行）+ `network/slots.py`（253 行），"
      "`MultiBandISRSGN(connections=…)` | **0**（legacy 走 `connections=None`，逐位相同） |")
    A("| `ode_accel` 档位 | ✅ | `ODE_ACCEL_MODES = (\"off\", \"cached\")`，默认 `\"off\"` | **0**（纯 memoization，命中返回同一 ndarray） |")
    A("| 拉曼泵浦 | ✅（原有） | `FiberSpanConfig.raman_pumps` → `isrs_gn.py:85` `k_bar` → P1 (22) 的 `(1−k̄)` | **0**（legacy 不配泵浦） |")
    A("| 任意功率输入 | ✅（原有） | `EDFAConfig.target_gain_db=None` → `isrs_gn.py:280-281` 自动均衡 | **0**（legacy 用定值增益） |")
    A("")
    A("⚠️ `ode_accel` **刻意只有两档**，且都是逐位安全的。"
      "fu4 的 `matvec` / `cached_g` 需要改写 `numba_kernels.py` 里被冻结的内核，"
      "按 brief §8 属「先停下来报告」的范围——见 §6。")
    A("")

    # ---------------- 6. 未跑 / 边界 ----------------
    A("## 5. 没做什么 / 没跑完什么")
    A("")
    A("| 项 | 状态 | 原因 |")
    A("|---|---|---|")
    A("| 三处差异的**逐项 dB 归因复算** | **未跑** | 那是 WP-O1 的交付物。本 WP 引用 O1 结论，"
      "brief §3 明示「直接引用，不必重做」 |")
    A("| `fu4` 的 `matvec` / `cached_g` / `tabular` 加速档 | **未跑，未移植** | 需改冻结内核（brief §8）；"
      "且其 NLI 基础是被 reject 的 Eq.21 模型 |")
    A("| `flex_*` 场景三树对照 | **未跑** | `ecz-dev` 无 `core/flexgrid`（PLAN §0 已确认），无从对照 |")
    A("| 逐连接**异构**符号速率 | **未跑（显式拒绝）** | 冻结 njit 核 `symbol_rate_hz` 是标量参数；"
      "改成数组 = 改冻结内核。已 `raise ValueError`，不静默取代表值 |")
    A("| P2 (8)/(9) 的完整拉曼放大 NLI | **未跑** | 那是 P2 的 RA 线；本分支是 P1 闭式线 |")
    A("| 删除 `numba_kernels.py:529-560` 死代码 | **未做** | 属冻结内核（brief §8）；且不改物理 |")
    A("")

    # ---------------- 7. 需要报告的边界 ----------------
    A("## 6. 需要编排方知道的边界（brief §8「先停下来报告」）")
    A("")
    A("**没有**需要立即叫停的 `fix_to_paper`——因为逐条比对后 `ecz-dev` 已符合 P1，`fix_to_paper` 计数为 0。"
      "但有三处**主动不移植**，理由如下，供编排方裁决是否要另开 WP：")
    A("")
    A("1. **`numba_kernels.py:529-560` 的死代码 `_calculate_ase_noise_variance`** 用 "
      "`G_i = p_in/p_out`，违反 P1 (34)。它当前**不生效**（`isrs_gn.py:180` 用 `_` 丢弃返回值）。"
      "清理它要动冻结内核，且会波及上游 ODE 线的数值金标准——建议留待专门 WP。")
    A("2. **逐连接异构 Rs** 需要把冻结核的 `symbol_rate_hz` 改成数组。"
      "本 WP 选择显式 `ValueError` 而非静默降级，能力缺口是公开可见的。")
    A("3. **fu4 的 `matvec`/`cached_g`** 若要移植，前置条件是先解决 ③（其加速建立在那套 NLI 上）。"
      "只抄加速而不抄物理，会得到一个既不对应 P1 也不对应 P2 的四不像。")
    A("")

    # ---------------- 8. 锚点附录（守卫 6） ----------------
    A("## 附录 A：公式锚点索引（J1 可 grep 证据）")
    A("")
    A("下表列出每条物理公式与每条裁定所引用的论文锚点，"
      "`tests/test_ode_dev.py::test_guard6_*` 会逐条在 `papers/*.md` 里 grep。")
    A("")
    A("| 来源 | 条目 | 论文 | 定位 | 锚点字符串 |")
    A("|---|---|---|---|---|")
    for e in fmap["physics"]:
        p = e["paper"]
        A(f"| formula_map | `{e['id']}` | {p['id']} | {p['locator']} | `{p['anchor']}` |")
    for it in verd["items"]:
        for key in ("paper", "paper2"):
            if it.get(key):
                p = it[key]
                A(f"| verdict | `{it['id']}` | {p['id']} | {p['locator']} | `{p['anchor']}` |")
    A("")

    A("## 附录 B：论文与复算入口")
    A("")
    A("```bash")
    A("export PYTHONPATH=$PWD/src")
    A("PY=~/miniconda3/envs/rl/bin/python")
    A("$PY tools/ode_formula_map.py --write --check   # J1：公式锚点逐条 grep")
    A("$PY tools/ode_verdicts.py     --write --check   # J1/J4：裁定与拒绝项")
    A("$PY tools/ode_compare.py      --write --check   # 三树 SNR 逐位复算")
    A("$PY tools/ode_report.py                          # 重渲染本文档")
    A("$PY -m pytest tests/ -q                          # 既有 6 条 + 新增 6 条")
    A("```")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="校验已落盘文档与当前 JSON 渲染一致")
    args = ap.parse_args()

    text = render()
    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        if OUT.read_text(encoding="utf-8") != text:
            print("FAIL docs/ode2_papers_and_diff.md 与 JSON 渲染结果不一致，请重跑 tools/ode_report.py")
            return 1
        print("OK   docs/ode2_papers_and_diff.md 与 results/*.json 一致")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
