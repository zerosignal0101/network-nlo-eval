"""FG-18 §2：**两向变异测试**读数。

守卫只有在"回归真的来了会变红"时才有意义。本脚本把 FG 树的合并式装配**人为塞回**
两个施加点，再读守卫的判决：

* **方向 A**：塞回 ``src/network_nlo_eval/models/isrs_gn.py``（FG-17 的盲区，
  逐跨段入口 :524 附近）
* **方向 B**：塞回 ``src/network_nlo_eval/rwa/qot_checker.py``（既有守卫的覆盖面）

两个方向都**必须变红**（pytest 非零退出）。任何一方向变绿都是守卫失效。

做法：在临时副本上操作，**不碰工作树**——把整个 ``src/`` 复制到临时目录，
注入变异，跑那个测试文件（它按 ``Path(__file__).parents[1]`` 定位仓库根，
因此测试文件与被测 ``src/`` 必须同属一棵临时树）。跑完删除临时树。

诚实声明：本脚本测的是**新守卫**（``test_fg18_lumped_assembly_tree_guard.py``）
与**既有守卫**（``test_fu4_assembly_union.py::test_fu4_6_*``）能否抓住变异；
它不改工作树，也不代表跑过物理仓全量（那是 §4 门 1 的事）。
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]

#: 注入到 ``models/isrs_gn.py`` 的变异（FG-17 观察到的原始写法）。
ISRS_GN_MUTATION = '''

def _fg18_mutation_probe(cfg):
    """FG-18 变异测试注入（方向 A）。不是生产代码。"""
    loss_lin = _db_to_lin(cfg.insertion_loss_db + cfg.filtering_penalty_db)
    return loss_lin
'''

#: 注入到 ``rwa/qot_checker.py`` 的变异。
QOT_CHECKER_MUTATION = '''

def _fg18_mutation_probe(config):
    """FG-18 变异测试注入（方向 B）。不是生产代码。"""
    loss = 10 ** ((config.insertion_loss_db + config.filtering_penalty_db) / 10)
    return loss
'''

NEW_GUARD = "tests/test_fg18_lumped_assembly_tree_guard.py"
OLD_GUARD = "tests/test_fu4_assembly_union.py"

#: 旧守卫的 node id。**用绝对路径**传给 pytest：临时树的 rootdir 与工作树不同，
#: 相对 ``tests/...`` 会被 pytest 解析成缺前缀的路径而报 "file or directory
#: not found"（rc=4）——那是探针自身失效，不是守卫判决。
OLD_GUARD_TEST_NAME = "test_fu4_6_lumped_assembly_is_gone_from_the_source"


def _inject(tree: pathlib.Path, rel: str, snippet: str) -> None:
    target = tree / rel
    text = target.read_text(encoding="utf-8")
    assert "_fg18_mutation_probe" not in text, f"{rel} already mutated?"
    target.write_text(text + snippet, encoding="utf-8")


def _run(tree: pathlib.Path, guard: str) -> dict:
    """跑一个守卫。``guard`` 是相对 ``tests/`` 的文件路径（可选带 ``::test`` 名）。

    一律用**绝对路径**定位测试文件：临时树 rootdir 与工作树不同，相对路径会被
    pytest 解析成缺前缀的路径，报 rc=4 而让人误读成"守卫判决"。
    """
    if "::" in guard:
        rel_file, test_name = guard.split("::", 1)
        nodeid = f"{tree / rel_file}::{test_name}"
    else:
        nodeid = str(tree / guard)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree / 'src'}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "-q", "-p", "no:randomly", "--no-header"],
        cwd=tree,
        capture_output=True,
        text=True,
        env=env,
    )
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-6:]
    # pytest 退出码 4 = usage error，5 = no tests collected。两者都是**探针失效**，
    # 不是"守卫变红"。早先版本只按 rc!=0 判红，把收集失败误读成了有效判决。
    valid = proc.returncode in (0, 1)
    return {
        "returncode": proc.returncode,
        "valid": valid,
        "turned_red": proc.returncode == 1,
        "verdict": (
            "RED"
            if proc.returncode == 1
            else "GREEN"
            if proc.returncode == 0
            else f"INVALID(rc={proc.returncode}, collection/usage error)"
        ),
        "tail": tail,
    }


def _scenario(name: str, rel: str, snippet: str, guard: str) -> dict:
    """在临时树上注入变异，跑给定守卫，读判决。"""
    with tempfile.TemporaryDirectory(prefix="fg18_mut_") as tmp:
        tree = pathlib.Path(tmp) / "repo"
        # 复制守卫需要的东西：src/ + tests/ + results/（既有守卫的兄弟测试要读
        # results/*.json 断言，缺了会**收集失败**）+ 仓库根配置。
        shutil.copytree(REPO / "src", tree / "src")
        shutil.copytree(REPO / "tests", tree / "tests")
        shutil.copytree(REPO / "results", tree / "results")
        for extra in ("pyproject.toml",):
            src = REPO / extra
            if src.exists():
                shutil.copy2(src, tree / extra)

        inject_rel = f"src/network_nlo_eval/{rel}"
        _inject(tree, inject_rel, snippet)

        result = _run(tree, guard)

        # 对照：同一棵（已变异的）树在**未注入**前应绿。这里用一个恒真节点校验
        # 临时树本身可跑（避免"因为环境坏了所以红"的假红）。
        sanity = _run(tree, NEW_GUARD + "::test_detector_does_not_fire_on_legal_forms")

        return {
            "scenario": name,
            "injected_into": inject_rel,
            "guard": guard,
            "mutated": result,
            "sanity_non_red": {"returncode": sanity["returncode"], "tail": sanity["tail"]},
        }


def main() -> int:
    # baseline：干净树上两个守卫都必须**绿**。否则"变红"说明不了任何问题。
    with tempfile.TemporaryDirectory(prefix="fg18_base_") as tmp:
        tree = pathlib.Path(tmp) / "repo"
        shutil.copytree(REPO / "src", tree / "src")
        shutil.copytree(REPO / "tests", tree / "tests")
        shutil.copytree(REPO / "results", tree / "results")
        for extra in ("pyproject.toml",):
            src = REPO / extra
            if src.exists():
                shutil.copy2(src, tree / extra)
        baseline = {
            "new_guard": _run(tree, NEW_GUARD),
            "old_guard": _run(tree, f"{OLD_GUARD}::{OLD_GUARD_TEST_NAME}"),
        }

    old_guard_spec = f"{OLD_GUARD}::{OLD_GUARD_TEST_NAME}"
    scenarios = [
        _scenario("A: isrs_gn.py (FG-17 blind spot)", "models/isrs_gn.py", ISRS_GN_MUTATION, NEW_GUARD),
        _scenario("A2: isrs_gn.py vs OLD guard", "models/isrs_gn.py", ISRS_GN_MUTATION, old_guard_spec),
        _scenario("B: rwa/qot_checker.py", "rwa/qot_checker.py", QOT_CHECKER_MUTATION, NEW_GUARD),
        _scenario("B2: rwa/qot_checker.py vs OLD guard", "rwa/qot_checker.py", QOT_CHECKER_MUTATION, old_guard_spec),
    ]

    def _v(s: dict) -> str:
        return s["mutated"]["verdict"]

    payload = {
        "question": "把合并式装配塞回 src/ 后，守卫是否变红（两向）",
        "baseline": baseline,
        "scenarios": scenarios,
        "verdict": {
            "A_new_guard": _v(scenarios[0]),
            "B_new_guard": _v(scenarios[2]),
            "A_old_guard": _v(scenarios[1]),
            "B_old_guard": _v(scenarios[3]),
            "baseline_new_guard_green": baseline["new_guard"]["verdict"],
            "baseline_old_guard_green": baseline["old_guard"]["verdict"],
            "all_valid": all(
                s["mutated"]["valid"]
                for s in scenarios
            )
            and baseline["new_guard"]["valid"]
            and baseline["old_guard"]["valid"],
        },
    }
    out = REPO / "results/fg18_mutation_test.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("=== baseline (no mutation, must be GREEN) ===")
    print(" new guard:", baseline["new_guard"]["verdict"], baseline["new_guard"]["tail"][-1:])
    print(" old guard:", baseline["old_guard"]["verdict"], baseline["old_guard"]["tail"][-1:])
    for s in scenarios:
        print(f"=== {s['scenario']} -> {s['guard']} ===")
        print("  rc:", s["mutated"]["returncode"], s["mutated"]["verdict"])
        for ln in s["mutated"]["tail"]:
            print("   ", ln)
    print("verdict:", json.dumps(payload["verdict"]))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
