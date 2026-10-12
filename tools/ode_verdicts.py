#!/usr/bin/env python3
"""WP-O2 步骤 2：逐项裁定 + 被拒绝的移植项（`results/ode_dev_rejected.json`）。

裁定四选一（brief §5 步骤 2）：

============  =========================================================
``accept``    论文提及且实现与论文一致 ⇒ 采纳（ ecz 侧已合规，或纯工程能力）
``fix_to_paper`` 论文提及但当前实现与论文不一致 ⇒ 改成论文的
``keep_ecz_no_paper`` 论文**未提及** ⇒ 保留 ecz-dev 实现（保持现状的许可证）
``reject``    移植会**违反**论文 ⇒ 不移植
============  =========================================================

⚠️ J2 的方向：``keep_ecz_no_paper`` 是「保留现状」的许可证，**不是**移植新东西的
许可证。J3：``wp/fg-uwb-fid`` 任何实现都不是权威。J4：差异①与 P1 (34) 冲突，不移植。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode_verdicts.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode_verdicts.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
P1 = "papers/A_Generalized_Raman_Scattering_Model_for_Real-Time_SNR_Estimation_of_Multi-Band_Systems_2036278533910728704.md"
P2 = "papers/Ultra-Wideband_Modeling_of_Optical_Fibre_Nonlinearity_in_Hybrid-Amplified_Links.md"
OUT = REPO / "results" / "ode_dev_rejected.json"

VALID_VERDICTS = {"accept", "fix_to_paper", "keep_ecz_no_paper", "reject"}

# ---------------------------------------------------------------------------
# 逐项裁定
# ---------------------------------------------------------------------------
ITEMS: list[dict] = [
    # ================= WP-O1 归因出的三处差异 =================
    {
        "id": "ase_gain_source",
        "label": "差异①：ASE 增益因子",
        "fu4_does": "把 numba 内核里 G_i = p_in/p_out_fiber（跨段入/出功率比）的 ASE 接上，"
                    "并删掉 Python 侧的 EDFA 增益块（isrs_gn.py 收下 s2_ase）。",
        "fu4_location": "wp/fg-uwb-fid:src/.../numba_kernels.py:529-560 (G_i at :554) "
                        "+ isrs_gn.py:415 (收下 s2_ase)",
        "ecz_location": "isrs_gn.py:214 (gain 来自 target_gain_db × gain_ripple_db)",
        "paper_says": "σ²_ASE(f_i) = h f_i F(f_i) B_i G(f_i) N，其中 G(f_i)、F(f_i) 是"
                      "**each line amplifier + DGE** 的增益与噪声系数。",
        "paper": {"id": "P1", "locator": "Eq. (34)", "anchor": "\\tag{34}", "file": P1},
        "corroboration": "P2 §III.C (5) 独立地确认同一件事：G_i = P_i(L)/P_i(0) 是**放大器**增益，"
                         "且用 (G_i − 1) 形式；P2 (5) 的 n_sp 定义展开后同样是 NF·(G−1)。",
        "paper2": {"id": "P2", "locator": "Eq. (5)", "anchor": "P_{\\text{ASE},i}^{\\prime } = 2(G_{i}-1)", "file": P2},
        "verdict": "reject",
        "judgement": "J4 明令禁止移植。本 WP 独立复核后**同意**该裁决："
                     "ecz 的 gain 取自 EDFA 配置增益，正是 (34) 的 G；fu4 换成跨段损耗比后，"
                     "即使 EDFA 配的是 17 dB 它也照用 16 dB，等于用「损耗补偿量」冒充「放大器增益」。"
                     "另外它让 ASE 对 target_gain_db / gain_ripple_db 完全不敏感——"
                     "而 (34) 明确要求 G 是**每一级**放大器+DGE 的增益。",
        "quantified_effect_db": None,
        "quantified_note": "本分支不移植 ⇒ 该项对 ode-dev 的 SNR 贡献恒为 0。"
                           "WP-O1 已测：fu4 侧此为主要抬升项，本 WP 未复算（不重做 O1）。",
    },
    {
        "id": "alpha_polynomial",
        "label": "差异②：损耗多项式的开关",
        "fu4_does": "删掉 use_attenuation_polynomial 开关，**永远**走二次多项式 α(λ)。",
        "fu4_location": "wp/fg-uwb-fid:src/.../isrs_gn.py:215-227",
        "ecz_location": "isrs_gn.py:101-111（开关保留，两条路都在）",
        "paper_says": "§II 给出实验拟合的二次多项式 α(λ) 及其系数；§II-A 的假设 (ii) 明确允许"
                      "频率无关的 ᾱ；(20) 又规定窗口内取被测信道处的 α_i = α(f_i)。",
        "paper": {"id": "P1", "locator": "§II（(1) 之后的无编号式）", "anchor": "3.7685\\times10^{-6}", "file": P1},
        "corroboration": None,
        "paper2": None,
        "verdict": "keep_ecz_no_paper",
        "judgement": "两种口径 P1 都写了（§II-A 的 ᾱ 与 (20) 的 α_i=α(f_i)），"
                     "所以**论文没有要求删掉开关**，把「必须无条件用多项式」当作 fix_to_paper 是误判。"
                     "ecz 的多项式系数与 λ_0 与论文逐字相同（elements.py:26-29），开关默认 False 是场景取值。"
                     "J2 的方向：论文未裁定 ⇒ 保留 ecz 实现。",
        "quantified_effect_db": 0.0,
        "quantified_note": "未移植 ⇒ 对 legacy 场景零影响（见 results/ode_dev_diff_vs_ecz.json 逐位相同）。",
    },
    {
        "id": "nli_model",
        "label": "差异③：NLI 模型换成 Eq.21 张量版 + 峰度修正",
        "fu4_does": "把 SPM/XPM 换成 2×2 双重求和的 Upsilon/T_i=1+t_tilde 展开（docstring 自称"
                    "「基于 Eq. 21」），phi_i 取 −4π²(β2+2πβ3 f)（**负号**且前因子 −4π² 而非 (3/2)π²），"
                    "并加入 (5/6)·φ_kurtosis 的格式修正项。",
        "fu4_location": "wp/fg-uwb-fid:src/.../numba_kernels.py:731-830 (_calc_spm/xpm_variance_jit) "
                        "+ :831-910 (_compute_nli_variances)",
        "ecz_location": "numba_kernels.py:278-335 (SPM)、:338-385+493 (XPM) —— 对应 P1 (29)/(30)",
        "paper_says": "P1 (29)/(30) 的前因子是 4/9（P1 (23) 为 16/27）与 32/27，"
                      "phi_i = (3/2)π²(β2 + 2π f_i β3)，T_i = (2α_i − C_r r(f_i))²，"
                      "**不含任何峰度项**。",
        "paper": {"id": "P1", "locator": "Eq. (29) 与 (30)", "anchor": "]_{(29)}", "file": P1},
        "corroboration": "P2 §III.D 的 (8)/(9) 确有 Upsilon / T_i=1+T_f−T_b·e^{−α_b L} / κ_f / κ_b 结构，"
                         "但那是**含泵浦的拉曼放大**情形；P2 同一节末段明说「For LA … 可用 [36] 的 (10)/(11)」。"
                         "fu4 的实现自称「Eq. 21」——**P1 与 P2 都没有 (21) 式**（P1 的 (21) 是 SRS gain 定义，"
                         "P2 的公式编号只到 (10)）。故其具体变体在两篇论文里都找不到公式号。",
        "paper2": {"id": "P2", "locator": "§III.D (8)/(9) 与末段 LA 说明",
                   "anchor": "\\Upsilon _{i}", "file": P2},
        "verdict": "reject",
        "judgement": "两重理由：(a) J3——fu4 不是权威；(b) J1——它自称的 Eq. 21 在 P1/P2 均不存在，"
                     "而 phi 的符号与归一化与 P1 (29)/(30) 冲突。brief §10 已预先把「fu4 的 NLI 数值拟合」"
                     "列为 reject 候选；本 WP 读论文后**确认**该候选成立。"
                     "ecz 侧的 (29)/(30) 实现是逐项对得上的（见 results/ode_dev_formula_map.json 的"
                     "p1_eq29_spm / p1_eq30_xpm），故保留 ecz。",
        "quantified_effect_db": None,
        "quantified_note": "未移植 ⇒ 零影响。WP-O1 已测该项方向与①相反、抵消一部分；本 WP 未复算。",
    },
    # ================= 工程能力 =================
    {
        "id": "flexgrid",
        "label": "灵活栅格（FlexibleGrid + ConnectionSpec）",
        "fu4_does": "新增 core/flexgrid.py + network/slots.py，支持逐连接中心频率与逐槽占用带。",
        "fu4_location": "wp/fg-uwb-fid:src/.../core/flexgrid.py (343 行)、network/slots.py (253 行)",
        "ecz_location": "core/flexgrid.py、network/slots.py（本 WP 新增移植）",
        "paper_says": "P1/P2 **均未提及**灵活栅格/槽位化频谱模型。",
        "paper": None,
        "corroboration": None,
        "paper2": None,
        "verdict": "accept",
        "judgement": "纯工程能力，与物理公式无关，**论文未提及**，不受 J1 约束。移植后 legacy 模式"
                     "（connections=None）数值**逐位不变**（实测 Δ=0，见 diff_vs_ecz.json）。",
        "quantified_effect_db": 0.0,
        "quantified_note": "legacy 场景走 connections=None 分支，逐位相同。",
    },
    {
        "id": "ode_accel",
        "label": "ode_accel 加速档位",
        "fu4_does": "三档 off / cached_g / matvec：缓存拉曼 G 矩阵、NLI 张量化、matvec 去分支。",
        "fu4_location": "wp/fg-uwb-fid:src/.../numba_kernels.py:575-700, 1200-1480",
        "ecz_location": "isrs_gn.py ODE_ACCEL_MODES（本 WP 新增 off / cached 两档）",
        "paper_says": "P1/P2 未提及加速档位。**关键**：fu4 那套加速是建立在它的 Eq.21 NLI 模型之上的，"
                      "本分支不采用该模型（见 nli_model），故其加速也无从移植。",
        "paper": None,
        "corroboration": None,
        "paper2": None,
        "verdict": "accept",
        "judgement": "**论文未提及**加速档位。按 brief §7 测试 4 的口径「只测可达，不测物理」，交付**档位接口 + 逐位相等的 "
                     "memoization 档**。刻意不做浮点求和重排：matvec/cached_g 需改写 "
                     "numba_kernels.py 里被冻结的内核，按 brief §8 属「先停下来报告」的范围。",
        "quantified_effect_db": 0.0,
        "quantified_note": "cached 档是纯 memoization（命中返回同一 ndarray），与 off 逐位相等。",
    },
    {
        "id": "raman_spectrum",
        "label": "拉曼增益谱档位 legacy / tabular",
        "fu4_does": "两档拉曼谱：legacy 合成折线（13.2 THz 爬升、15.6 THz 归零）与"
                    "tabular 熔融石英谱（Lin & Agrawal 闭式）。",
        "fu4_location": "wp/fg-uwb-fid:src/.../physics/raman_spectrum.py (334 行) "
                        "+ numba_kernels.py:575-700",
        "ecz_location": "（不移植）",
        "paper_says": "P1 §II-B 用的是**截断三角近似** g_R(Δf) ≈ C_r Δf，截断 Δf_co，"
                      "并把它写进 r(f) 的四个 case（(18)）。ecz 的 _calc_raman_profile_jit 正是在实现 (18)。",
        "paper": {"id": "P1", "locator": "Eq. (18) + §II.B 三角近似", "anchor": "P_{\\mathrm{t}} f}{}", "file": P1},
        "corroboration": "P2 §II 用的是实测拉曼增益谱（Fig. 4，峰值约 13.4 THz），但那属于 ODE/RA 线，"
                         "不是闭式 ECZ 线。",
        "paper2": None,
        "verdict": "reject",
        "judgement": "两档都**不是** P1 的三角近似：(a) legacy 折线是 fg-ode-fast 那条数值线的合成曲线，"
                     "fu4 自己注明它是「FG-4a 的数值金标准」依赖的曲线；(b) tabular 明确「会改变结果」——"
                     "那是物理改进，不是数值重排。J3 下 fu4 无权威性，故**两档都不移植**。"
                     "本分支按 P1 (18) 的三角近似走，已由 p1_eq18_raman_profile 锁定。",
        "quantified_effect_db": None,
        "quantified_note": "未移植。WP-O1 已测 legacy 下 raman_spectrum 差异 ~6e-5 dB（可忽略）。",
    },
    {
        "id": "phi_kurtosis",
        "label": "峰度修正 phi_kurtosis",
        "fu4_does": "给 MultiBandISRSGN 与 NLI 加 phi_kurtosis（0.0 高斯 / −0.6190 64QAM）。",
        "fu4_location": "wp/fg-uwb-fid:src/.../isrs_gn.py:134",
        "ecz_location": "（不存在该参数）",
        "paper_says": "P1 (29)/(30) **没有**任何峰度项；P2 §III.D 把任意调制的推广明确推给 [47]。"
                      "两篇论文都没有公式支撑这一项。",
        "paper": {"id": "P1", "locator": "Eq. (29)/(30)（无峰度项）", "anchor": "]_{(30)}", "file": P1},
        "corroboration": "P2 §III.D：「An extension of these equations for arbitrary-modulated signals "
                         "can be found in [47].」——即论文自己承认 (8)/(9) 只覆盖高斯调制。",
        "paper2": {"id": "P2", "locator": "§III.D 引言", "anchor": "\\Upsilon _{i}", "file": P2},
        "verdict": "reject",
        "judgement": "J1：找不到公式号 ⇒ 不许保留。本分支保持无该参数（ecz 原状），"
                     "NLI 严格按 P1 (29)/(30) 的高斯模型。与 PLAN §0 的裁决一致："
                     "phi_kurtosis 不是 +0.77 dB 的原因，两边都是默认 0。",
        "quantified_effect_db": 0.0,
        "quantified_note": "未移植；legacy 读数与 ecz 逐位相同。",
    },
    {
        "id": "raman_pumps",
        "label": "拉曼泵浦",
        "fu4_does": "同样通过 raman_pumps → k_bar 进入。",
        "fu4_location": "（与 ecz 同源）",
        "ecz_location": "elements.py:39 (raman_pumps) → isrs_gn.py:85 (k_bar) → numba_kernels.py:259",
        "paper_says": "P1 (22) 定义预加重：SRS 项乘 (1 − k̄)。",
        "paper": {"id": "P1", "locator": "Eq. (22)", "anchor": "\\tag{22}", "file": P1},
        "corroboration": None,
        "paper2": None,
        "verdict": "keep_ecz_no_paper",
        "judgement": "**能力已在 ecz-dev 存在**，无需移植。论文给了 (1−k̄) 缩放，但**没有**规定 k̄ 如何由"
                     "泵浦功率导出；ecz 取 min(ΣP_pump, 1) 是论文未提及的工程取值，保留。",
        "quantified_effect_db": 0.0,
        "quantified_note": "路径可达已测（见 tests/test_ode_dev.py 能力存在性守卫）。",
    },
    {
        "id": "arbitrary_power_input",
        "label": "任意功率输入 / 自动均衡",
        "fu4_does": "始终用 p_in/p_out_fiber 推出增益。",
        "fu4_location": "wp/fg-uwb-fid:src/.../numba_kernels.py:554",
        "ecz_location": "isrs_gn.py:280-281（target_gain_db=None → 自动均衡跨段损耗）",
        "paper_says": "P2 (5) 把放大器增益定义为 G_i = P_i(L)/P_i(0)，即「恢复目标功率所需的增益」。"
                      "P1 §IV 也说均衡放大器「restores the target signal PSD」。",
        "paper": {"id": "P2", "locator": "Eq. (5)", "anchor": "P_{\\text{ASE},i}^{\\prime } = 2(G_{i}-1)", "file": P2},
        "corroboration": "P1 §IV：「The gain of the equalized amplifier restores the target signal PSD by "
                         "equalizing the SRS and the fiber loss.」",
        "paper2": None,
        "verdict": "keep_ecz_no_paper",
        "judgement": "**能力已在 ecz-dev 存在**，无需移植。ecz 的两条分支都合规："
                     "给了 target_gain_db 用定值（(34) 的 G）；给 None 时用自动均衡比，"
                     "此时该比**就是**放大器增益（P2 (5) 的定义）。"
                     "fu4 的问题不在于用了这个比值，而在于**无视这个区分、永远用它**，"
                     "从而在配了定值增益时违反 (34)——即 ase_gain_source。",
        "quantified_effect_db": 0.0,
        "quantified_note": "两条路径可达均已测。",
    },
    {
        "id": "qot_checker_refactor",
        "label": "qot_checker 的函数化重构",
        "fu4_does": "把 ecz 的 inline ROADM/booster 逻辑抽成 _roadm_terms() 等函数。",
        "fu4_location": "wp/fg-uwb-fid:src/.../rwa/qot_checker.py:144",
        "ecz_location": "rwa/qot_checker.py（inline）",
        "paper_says": "P2 (1) 允许把实现代价作为独立可加项（SNR_TRX 项）；论文不规定代码组织。",
        "paper": {"id": "P2", "locator": "Eq. (1)", "anchor": "P_{\\text{ASE,i}}^{\\prime \\prime } + \\eta _{n}(f_{i})P_{i}^{3}}", "file": P2},
        "corroboration": None,
        "paper2": None,
        "verdict": "keep_ecz_no_paper",
        "judgement": "PLAN §0 已实测：两树 SNR 累加主循环逻辑几乎逐字相同，"
                     "「_roadm_terms 与 inline 写法的**语义差异是零**」，且该路径解释不了 0.77 dB。"
                     "纯重构，无物理收益 ⇒ 不移植。",
        "quantified_effect_db": 0.0,
        "quantified_note": "零影响（PLAN §0 已排除）。",
    },
]


def build() -> dict:
    return {
        "wp": "O2",
        "branch": "ode-dev",
        "generated_by": "tools/ode_verdicts.py",
        "judgement_rules": {
            "accept": "论文提及且实现与论文一致，或纯工程能力",
            "fix_to_paper": "论文提及但当前实现与论文不一致 ⇒ 改成论文的",
            "keep_ecz_no_paper": "论文未提及 ⇒ 保留 ecz-dev 实现（J2：这是保留现状的许可证）",
            "reject": "移植会违反论文，或找不到论文公式号（J1）",
        },
        "items": ITEMS,
        "rejected": [i for i in ITEMS if i["verdict"] == "reject"],
        "summary": {
            "n_items": len(ITEMS),
            "n_accept": sum(1 for i in ITEMS if i["verdict"] == "accept"),
            "n_fix_to_paper": sum(1 for i in ITEMS if i["verdict"] == "fix_to_paper"),
            "n_keep_ecz": sum(1 for i in ITEMS if i["verdict"] == "keep_ecz_no_paper"),
            "n_reject": sum(1 for i in ITEMS if i["verdict"] == "reject"),
            "ase_gain_source_rejected": any(
                i["id"] == "ase_gain_source" for i in ITEMS if i["verdict"] == "reject"
            ),
        },
    }


def check(doc: dict) -> list[str]:
    """J1/J4 守卫：每条裁定都要么有论文定位，要么明确写「论文未提及」。"""
    errs: list[str] = []
    cache: dict[str, str] = {}
    for item in ITEMS:
        iid = item["id"]
        if item["verdict"] not in VALID_VERDICTS:
            errs.append(f"{iid}: 非法裁定 {item['verdict']!r}")
        if item["verdict"] == "reject" and not item.get("paper"):
            errs.append(f"[J1] {iid}: reject 项必须有 paper 公式号或明确的「论文未提及」判定")
        if item.get("paper"):
            p = item["paper"]
            if not p.get("locator"):
                errs.append(f"[J1] {iid}: paper 缺少 locator")
            if not p.get("anchor"):
                errs.append(f"[J1] {iid}: paper 缺少 anchor")
                continue
            f = p["file"]
            if f not in cache:
                cache[f] = (REPO / f).read_text(encoding="utf-8")
            if p["anchor"] not in cache[f]:
                errs.append(f"[J1] {iid}: anchor {p['anchor']!r} 在 {f} 里 grep 不到")
        else:
            if "未提及" not in item.get("judgement", ""):
                errs.append(f"[J2] {iid}: 无 paper 时 judgement 必须明确写「论文未提及」")
    if not doc["summary"]["ase_gain_source_rejected"]:
        errs.append("[J4] ase_gain_source 必须落在 rejected 里")
    return errs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = build()
    errs = check(doc)

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        print(f"  accept={doc['summary']['n_accept']}  fix_to_paper={doc['summary']['n_fix_to_paper']}  "
              f"keep_ecz={doc['summary']['n_keep_ecz']}  reject={doc['summary']['n_reject']}")

    if args.check:
        if errs:
            print(f"FAIL: {len(errs)} 条")
            for e in errs:
                print("  " + e)
            return 1
        print(f"OK: {len(ITEMS)} 条裁定合法；rejected {doc['summary']['n_reject']} 项均带论文判定；"
              f"J4 守卫通过")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
