"""FG-20 §2b：**行为**测试的变异读数（删掉 booster 注入 ⇒ 变红）。

为什么源码守卫需要这个对照
--------------------------
FG-18 的守卫是 AST 级：它能抓住"``insertion_loss_db + filtering_penalty_db`` 这个
字面形状回来了"，却**抓不住**"用新写法干错的事"。FG-13 正是后者：源码看着没问题，
漏的是 booster 注入。FG-20 补的行为测试只有在**真的能变红**时才有意义，故本脚本
证明之。

四个变异方向（都在**临时树副本**上注入，不碰工作树）：

* **M1**：删掉串行施加点的 booster 注入（``isrs_gn.py`` 的 ``if terms.equalize`` 分支
  变成空操作）——FG-13 的**同型**语义事故。
* **M2**：删掉 batch 施加点的 booster 注入。
* **M3**：把 equalize 分支的"不衰减信号"退回成"照样除 loss"（信号口径事故）。
* **M4**：把 ``filtering_penalty`` 并进信号光功率（FG 树旧口径的量纲错误）。
* **M5**：退回合并式 ``insertion_loss_db + filtering_penalty_db``（FG-18 盲区同款，
  用来证明**两把尺子都抓得住**：AST 守卫红 + 行为测试红）。

探针自检（FG-18 踩过的坑，本脚本不重踩）
--------------------------------------
1. 每次运行都记录 ``returncode``，并把 rc=4/5（收集/内部错误）**以及任何出现
   ERROR 的输出**单列为 INVALID，不计入判决——否则"探针自己坏了"会被误读成
   "守卫抓到了"。
2. **一律传绝对路径**给 pytest：临时树的 rootdir 与工作树不同，相对
   ``tests/...`` 会被解析成缺前缀的路径而 rc=4。
3. **baseline 必须先跑一遍且必须绿**，否则"变异后红"可能只是"本来就在红"。
4. 临时树必须复制 ``results/``（既有守卫的兄弟测试要读它）与
   ``references/``（拉曼谱表 CSV，按仓库根相对路径加载）。

⚠️ **本脚本初版自己翻过一次车，如实记录**：初版临时树只复制 ``src/ tests/ results/``，
漏了 ``references/``。于是 ``MultiBandISRSGN.__init__`` 在
``load_raman_gain_table()`` 处抛 ``FileNotFoundError``，45 条用例变成
**44 errors + 1 passed**，而 pytest 对 error 也给 rc=1——初版把它读成了"RED = 抓到了"。
**探针失效被误读成判决**，与 FG-18 那两次同源。修法是补 ``references/`` 并把
ERROR 显式判为 INVALID（见 ``_run``）。**教训：rc 相同不代表判决相同。**

诚实边界：本脚本只回答"行为测试能否抓住这五种语义回归"，**不代表**物理仓全量门
（那是 §4 门 1 的事，两者分别记数，不互相顶替）。
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
PY = "/home/zsig/miniconda3/envs/rl/bin/python"

BEHAVIOR_TEST = "tests/test_fg20_roadm_injection_behavior.py"
AST_GUARD_TEST = "tests/test_fg18_lumped_assembly_tree_guard.py"

ISRS_GN = "src/network_nlo_eval/models/isrs_gn.py"

# ---------------------------------------------------------------------------
# 变异片段。每个都是"把一段现有代码换成别的"，故用**成对**替换而不是追加，
# 保证注入点唯一且不会留下半截死代码（追加式注入会同时留下正确实现与错误实现，
# 测试仍绿 —— 那是 FG-18 初版探针的失效模式）。
# ---------------------------------------------------------------------------

#: M1：串行施加点 —— booster 注入变空操作（**不删**分支本身，保持控制流形状）。
M1_SERIAL = (
    """            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
""",
    """            if terms.equalize:
                pass  # FG-20 M1: booster 注入被删掉
""",
)

#: M2：batch 施加点 —— 同上，(B, N) 布局。
M2_BATCH = (
    """            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
""",
    """            if terms.equalize:
                pass  # FG-20 M2: booster 注入被删掉
""",
)

#: M3：equalize 分支 —— 信号仍被 loss 衰减（booster "有损"）。
M3_EQUALIZE_ATTENUATES = (
    """            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
            else:
                p_out_fiber = p_out_fiber / terms.loss
""",
    """            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
                p_out_fiber = p_out_fiber / terms.loss
            else:
                p_out_fiber = p_out_fiber / terms.loss
""",
)

#: M4：把 filtering_penalty 并进信号光功率（FG 树旧口径的量纲错误）。
M4_FILTER_IN_POWER = (
    """            terms = fold_roadm_terms(roadm_config, is_last_hop=False)
            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
""",
    """            terms = fold_roadm_terms(roadm_config, is_last_hop=False)
            p_out_fiber = p_out_fiber * terms.filtering_penalty  # FG-20 M4: 滤波惩罚并进光功率
            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
""",
)

#: M5：退回 FG 树合并式（FG-17 盲区同款）—— 同时应被 AST 守卫与行为测试抓住。
M5_LUMPED = (
    """            terms = fold_roadm_terms(roadm_config, is_last_hop=False)
""",
    """            terms = fold_roadm_terms(roadm_config, is_last_hop=False)
            terms = terms.__class__(
                **{
                    **terms.__dict__,
                    "loss": 10 ** ((roadm_config.insertion_loss_db + roadm_config.filtering_penalty_db) / 10),
                }
            )
""",
)

#: (名字, 目标文件, 旧片段, 新片段, 注入序号)
#: ``注入序号`` 区分同一文件里**重复出现**的旧片段（M1/M2/M5 用同一段 anchor）。
MUTATIONS = [
    ("M1_serial_booster_deleted", ISRS_GN, M1_SERIAL, 0),
    ("M2_batch_booster_deleted", ISRS_GN, M2_BATCH, 1),
    ("M3_equalize_still_attenuates_signal", ISRS_GN, M3_EQUALIZE_ATTENUATES, 0),
    ("M4_filtering_penalty_folded_into_power", ISRS_GN, M4_FILTER_IN_POWER, 0),
    ("M5_lumped_assembly_restored", ISRS_GN, M5_LUMPED, 1),
]


def _apply_mutation(source: str, old: str, new: str, occurrence: int, label: str) -> str:
    """把第 ``occurrence`` 处出现的 ``old`` 换成 ``new``。

    找不到或找到多处（超过指定序号）都直接抛 —— 静默不注入会让"变异后仍绿"被
    误读成"行为测试抓不住"，那是探针失效，不是判决。
    """
    if old not in source:
        raise RuntimeError(f"{label}: anchor 未命中 —— 探针失效（src 已变？）")
    first = source.index(old)
    if occurrence == 0:
        idx, start = 0, first
    else:
        second = source.find(old, first + len(old))
        if second < 0:
            raise RuntimeError(f"{label}: 第 {occurrence + 1} 处 anchor 不存在 —— 探针失效")
        idx, start = 1, second
    return source[:start] + new + source[start + len(old) :]


def _run(tree: pathlib.Path, test_spec: str) -> dict:
    """在临时树上跑一个测试，返回判决。

    ``returncode``: pytest 退出码。0=GREEN，1=RED（有断言失败），4/5=INVALID（收集或
    内部错误 —— **不是**判决，单列出来）。
    """
    abs_spec = str((tree / test_spec).resolve())
    env = {"PYTHONPATH": str(tree / "src")}
    proc = subprocess.run(  # noqa: S603
        [PY, "-m", "pytest", abs_spec, "-q", "-p", "no:randomly"],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    rc = proc.returncode
    tail = [ln for ln in (proc.stdout + proc.stderr).strip().splitlines() if ln.strip()][-4:]
    output = proc.stdout + proc.stderr
    # ERROR（fixture 炸了 / 收集不到模块）时 pytest 也可能给 rc=1，**那是探针失效，
    # 不是判决**。必须显式排除，否则"探针自己坏了"会被读成"守卫抓到了变异"。
    # 本脚本初版正是这样翻车的：临时树缺 `references/`，44 errors 被当成 RED。
    if " error" in output or "ERROR " in output or "INTERNALERROR" in output:
        verdict = "INVALID"
    elif rc == 0:
        verdict = "GREEN"
    elif rc == 1:
        verdict = "RED"
    else:
        verdict = "INVALID"
    return {"returncode": rc, "verdict": verdict, "tail": tail, "valid": verdict != "INVALID"}


def _make_tree() -> pathlib.Path:
    """复制一棵可独立 import 的临时树。

    必须一起复制的目录（少任何一个都会让 pytest 报 ERROR 而不是断言失败，
    于是被误读成"守卫抓到了"——FG-18 探针踩过同源的坑）：

    * ``src/``  被测代码
    * ``tests/`` 测试自身
    * ``results/`` 既有守卫的兄弟测试要读它，缺了 rc=4
    * ``references/`` **拉曼谱表 CSV**（``physics/raman_spectrum.py`` 从
      ``<repo>/references/`` 按**仓库根相对路径**加载）。缺它时
      ``MultiBandISRSGN.__init__`` 直接 ``FileNotFoundError``，全部用例变 ERROR
      —— 本脚本初版就漏了它，44 errors 而非 45 passed，差点把探针失效当成判决。
    """
    tree = pathlib.Path(tempfile.mkdtemp(prefix="fg20_mut_"))
    for sub in ("src", "tests", "results", "references"):
        if (REPO / sub).exists():
            shutil.copytree(REPO / sub, tree / sub)
    for extra in ("pyproject.toml",):
        if (REPO / extra).exists():
            shutil.copy2(REPO / extra, tree / extra)
    return tree


def main() -> int:
    """跑五个变异方向，读行为测试与 AST 守卫各自的判决。"""
    baseline_tree = _make_tree()
    try:
        baseline = {
            "behavior": _run(baseline_tree, BEHAVIOR_TEST),
            "ast_guard": _run(baseline_tree, AST_GUARD_TEST),
        }
    finally:
        shutil.rmtree(baseline_tree, ignore_errors=True)

    scenarios = []
    for label, rel, (old, new), occurrence in MUTATIONS:
        tree = _make_tree()
        try:
            target = tree / rel
            source = target.read_text(encoding="utf-8")
            try:
                mutated_source = _apply_mutation(source, old, new, occurrence, label)
            except RuntimeError as exc:
                scenarios.append(
                    {
                        "scenario": label,
                        "injection_valid": False,
                        "error": str(exc),
                        "behavior": {"verdict": "INVALID", "valid": False},
                        "ast_guard": {"verdict": "INVALID", "valid": False},
                    }
                )
                continue
            target.write_text(mutated_source, encoding="utf-8")
            behavior = _run(tree, BEHAVIOR_TEST)
            ast_guard = _run(tree, AST_GUARD_TEST)
            scenarios.append(
                {
                    "scenario": label,
                    "target": rel,
                    "injection_valid": True,
                    "behavior": behavior,
                    "ast_guard": ast_guard,
                    # "抓住" = 变异后**红**，且探针本身有效。
                    "behavior_catches": behavior["verdict"] == "RED",
                    "ast_guard_catches": ast_guard["verdict"] == "RED",
                }
            )
        finally:
            shutil.rmtree(tree, ignore_errors=True)

    payload = {
        "question": "FG-20 行为测试能否抓住两处施加点的语义回归（源码守卫抓不住的那些）",
        "baseline": baseline,
        "scenarios": scenarios,
        "verdict": {
            "baseline_behavior_green": baseline["behavior"]["verdict"] == "GREEN",
            "baseline_ast_guard_green": baseline["ast_guard"]["verdict"] == "GREEN",
            "all_injections_valid": all(s.get("injection_valid") for s in scenarios),
            "behavior_catches_all_semantic": all(
                s.get("behavior_catches", False) for s in scenarios if s["scenario"] != "M5_lumped_assembly_restored"
            ),
            "ast_guard_catches_only_m5": [
                s["scenario"] for s in scenarios if s.get("ast_guard_catches")
            ],
            "n_scenarios": len(scenarios),
            "n_behavior_red": sum(1 for s in scenarios if s.get("behavior_catches")),
        },
    }
    payload["verdict"]["all_valid"] = (
        payload["verdict"]["baseline_behavior_green"]
        and payload["verdict"]["baseline_ast_guard_green"]
        and payload["verdict"]["all_injections_valid"]
        and all(s["behavior"]["valid"] for s in scenarios)
        and all(s["ast_guard"]["valid"] for s in scenarios)
    )

    out = REPO / "results/fg20_mutation_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== baseline (no mutation, must be GREEN) ===")
    print("  behavior :", baseline["behavior"]["verdict"], baseline["behavior"]["returncode"])
    print("  ast_guard:", baseline["ast_guard"]["verdict"], baseline["ast_guard"]["returncode"])
    for s in scenarios:
        print(f"=== {s['scenario']} ===")
        if not s.get("injection_valid"):
            print("  INJECTION INVALID:", s.get("error"))
            continue
        print("  behavior :", s["behavior"]["verdict"], s["behavior"]["returncode"])
        print("  ast_guard:", s["ast_guard"]["verdict"], s["ast_guard"]["returncode"])
    print("verdict:", json.dumps(payload["verdict"], ensure_ascii=False))
    print("wrote", out)
    return 0 if payload["verdict"]["all_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
