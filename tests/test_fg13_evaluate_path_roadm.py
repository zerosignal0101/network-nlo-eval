"""FG-13：`evaluate_path` 的 ROADM 施加点守卫。

裁决（详见 ``docs/fg13_evaluate_path_roadm.md``）：``models/isrs_gn.py`` 的
``evaluate_path`` 用合并式 ROADM 装配、**完全不读** ``express_attenuation_db``，
这是**缺陷**，不是"另一套刻意不同的模型"。

本文件钉住三件事：

1. **惰性字段清单**（FG-13 交付时逐位为 0 的那四个字段）——FG-14 §2 已修复，
   四个字段全部转为**活字段**，断言相应翻转成新状态的守卫；
2. **权威口径在哪里**（``ROADMConfig.path_loss_db`` + ``rwa.qot_checker.fold_roadm_terms``），
   并守住"折叠逻辑只有一处"这个 FG-11 已经建立的性质；
3. **按权威口径修复的影响量级**——证明它是物理量级改动，不是浮点噪声。

⚠️ **这些断言钉的是"当前状态"，不是"当前状态正确"。**
FG-14 §2 把它们**翻转成新状态的守卫**（钉住"端口字段现在被消费"），
并按编排方 FG-14 brief §0 把这次翻转记为**被授权的分歧**，不是掩盖：
授权来源见 ``docs/fg14_assembly_provenance.md``，分歧幅度见
``results/fg14_assembly_divergence.json``。
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.rwa.qot_checker import fold_roadm_terms

REPO_ROOT = Path(__file__).resolve().parent.parent
PROBE_JSON = REPO_ROOT / "results" / "fg13_roadm_probe.json"
IMPACT_JSON = REPO_ROOT / "results" / "fg13_impact_probe.json"


def _load_script(name: str) -> ModuleType:
    """按路径加载 ``scripts/`` 下的机理探针（它不是包，不能直接 import）。"""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fg13_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================================================
# FG13-1  惰性字段清单：逐位钉住
# ======================================================================


def test_fg13_probe_report_reproduces() -> None:
    """``scripts/fg13_probe_roadm.py`` 的读数逐条复现。

    这是本刀的基础证据：下面所有断言都建立在它之上，所以它必须能被重跑复算，
    不能只活在文档里。
    """
    probe = _load_script("fg13_probe_roadm")
    assert probe.probe() == json.loads(PROBE_JSON.read_text(encoding="utf-8"))


def test_fg13_port_and_booster_fields_are_inert_today() -> None:
    """**已修复（FG-14）—— 这四个字段现在被消费了。**

    FG-13 记录它们在 ``evaluate_path`` 上**逐位惰性**（max|ΔSNR| 恰为 ``0.0``）：

    * ``express_attenuation_db``（直通端口附加衰减）
    * ``add_drop_loss_db``（本地 add/drop 端口附加损耗）
    * ``booster_noise_figure_db``（补偿 ROADM 损耗的 booster NF）
    * ``equalize_output_power``（是否由 booster 恢复每信道入纤功率）

    FG-14 §2 按编排方授权把装配改为遵循仓内权威定义
    ``ROADMConfig.path_loss_db()`` + ``fold_roadm_terms``。⇒ 四个字段全部转为**活字段**，
    且不再是"影响很小"，而是 dB 量级可观测。

    这条断言现在钉的是**新状态**。再有人把装配退回合并式，它会变红。
    """
    report = json.loads(PROBE_JSON.read_text(encoding="utf-8"))
    by_field = {row["field"]: row for row in report["field_sweep"]}

    for field in (
        "express_attenuation_db",
        "add_drop_loss_db",
        "booster_noise_figure_db",
        "equalize_output_power",
    ):
        row = by_field[field]
        assert row["evaluate_path_snr_bitwise_identical"] is False, (
            f"{field} 又变回惰性字段了 —— evaluate_path 不再消费端口/booster 装配。"
            "若这是有意回退，请同步更新 docs/fg14_assembly_provenance.md 并说明授权来源。"
        )
        assert row["evaluate_path_max_abs_delta_db"] > 0.0, (
            f"{field} 声称是活字段但 max|ΔSNR| 为 0。"
        )

    # 惰性清单必须为空：FG-13 交付时它有四个成员。
    assert report["summary"]["inert_on_evaluate_path"] == [], (
        f"evaluate_path 上仍有惰性字段：{report['summary']['inert_on_evaluate_path']}"
    )

    # 同一把尺子量 RWA 并集路径：这些字段在那里一直是活的。
    assert by_field["express_attenuation_db"]["rwa_fold_express_loss_delta"] > 0.0
    assert by_field["add_drop_loss_db"]["rwa_fold_drop_loss_delta"] > 0.0
    assert by_field["express_attenuation_db"]["rwa_fold_drop_loss_delta"] == 0.0
    assert by_field["add_drop_loss_db"]["rwa_fold_express_loss_delta"] == 0.0


def test_fg13_last_hop_cancellation_is_not_mistaken_for_inertness() -> None:
    """守住 FG-13 最容易踩的假阳性—— 并记录 FG-14 让它**不再成立**。

    FG-13 时期 ``evaluate_path`` 把损耗同时除信号与**全部已累积噪声**，因此**末跳**
    ROADM 的损耗在 ``power/(spm+xpm+ase)`` 里代数抵消。拿一个根本没被读的字段去测末跳，
    同样会得到"逐位不变"，于是把"代数抵消"和"字段是死的"混为一谈。

    FG-14 修复后这个代数恒等**被打破**了：``equalize_output_power`` 默认 ``True``，
    booster 以 ``NF·h·ν·Rs·(loss−1)`` 在损耗**之后注入** ASE，这部分噪声不再被同一
    损耗约掉 ⇒ **末跳 ROADM 的损耗重新变得可观测**。

    ⇒ 本断言现在钉两件事：
    1. 非末跳仍必须可观测（否则是比本刀更严重的问题）；
    2. 末跳也必须可观测（钉住"booster 注入打断了末跳抵消"这个结构事实）。
    """
    demo = json.loads(PROBE_JSON.read_text(encoding="utf-8"))["structural_demo"]

    non_last_hop = demo["on_NON_LAST_hop_only"]
    assert non_last_hop["max_abs_delta_db"] > 0.01, (
        f"非末跳 ROADM 的损耗必须可观测，实测 {non_last_hop['max_abs_delta_db']}；"
        "若此处也变成惰性，说明 evaluate_path 的噪声累积或施加点被改动了。"
    )

    last_hop = demo["on_LAST_hop_only"]
    assert last_hop["max_abs_delta_db"] > 0.01, (
        "末跳 ROADM 的损耗应当重新可观测：FG-14 的 booster ASE 注入打断了 FG-13 §2.1 "
        "记录的代数抵消。若这里退回 ~1e-15，说明 equalize 分支没有生效，"
        "装配又退回合并式了。"
    )


# ======================================================================
# FG13-2  权威口径的位置（FG-11 的"折叠只有一处"不许回退）
# ======================================================================


def test_fg13_authoritative_port_loss_is_the_single_definition() -> None:
    """按端口取损耗的**唯一**定义仍是 ``ROADMConfig.path_loss_db``。

    ``evaluate_path`` 不读它不是"另有定义"，而是"没有定义"——本断言把权威口径钉在
    字段方法上，任何人想改口径都得先改这里，从而在 diff 里看得见。
    """
    roadm = ROADMConfig(insertion_loss_db=5.0, express_attenuation_db=2.0, add_drop_loss_db=1.0)
    assert roadm.path_loss_db("express") == pytest.approx(7.0)
    assert roadm.path_loss_db("drop") == pytest.approx(6.0)
    assert roadm.path_loss_db("express") != pytest.approx(roadm.path_loss_db("drop"))
    # filtering_penalty_db 是噪声域量，不进 path_loss_db（FU-4 §1.3）。
    assert roadm.path_loss_db("drop") == pytest.approx(6.0)


def test_fg13_fold_roadm_terms_is_the_single_folding_source() -> None:
    """RWA 路径的折叠只有一处实现，且按端口区分、滤波不并入损耗。"""
    roadm = ROADMConfig(
        insertion_loss_db=5.0, filtering_penalty_db=0.2, express_attenuation_db=2.0, add_drop_loss_db=1.0
    )
    express = fold_roadm_terms(roadm, is_last_hop=False)
    drop = fold_roadm_terms(roadm, is_last_hop=True)

    assert express.loss == pytest.approx(10 ** (7.0 / 10))
    assert drop.loss == pytest.approx(10 ** (6.0 / 10))
    # 滤波是独立乘子，不得并进损耗（并进去就会变成 7.2/6.2 dB）。
    assert express.filtering_penalty == pytest.approx(10 ** (0.2 / 10))
    assert express.loss != pytest.approx(10 ** (7.2 / 10))


def _code_without_docstring(source: str) -> str:
    """``evaluate_path`` 的源码，摘掉 docstring —— 结构守卫只看代码。

    docstring 必须能写明它遵循哪个权威定义（``path_loss_db`` /
    ``fold_roadm_terms``），若把散文一起扫进守卫，这条断言就没法如实书写。
    """
    tree = ast.parse(source)
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_path"
    )
    doc = ast.get_docstring(func, clean=False) or ""
    body = source.split("def evaluate_path(", 1)[1].split("\n    def ", 1)[0]
    return body.replace(doc, "")


def test_fg13_evaluate_path_calls_the_union_folding() -> None:
    """结构守卫：``evaluate_path`` **现在**消费权威折叠（FG-14 修复后）。

    这条断言在 FG-13 交付时是反的（"不消费"是缺陷标记）。FG-14 §2 按编排方授权
    把装配改为遵循 ``ROADMConfig.path_loss_db()``，于是它翻转成新状态的守卫：
    钉住"端口字段现在被消费"。装配退回合并式时它会变红。
    """
    source = (REPO_ROOT / "src" / "network_nlo_eval" / "models" / "isrs_gn.py").read_text(encoding="utf-8")
    body = source.split("def evaluate_path(", 1)[1].split("\n    def ", 1)[0]
    # 只看**代码**，不看 docstring：docstring 必须能说明它遵循 ``path_loss_db``
    # 这个权威定义，把散文也扫进来会让这条守卫无法书写。用 ast 精确摘掉它。
    code = _code_without_docstring(source)

    assert "fold_roadm_terms" in code, "evaluate_path 不再消费权威折叠，请更新本文件与文档的期望值"
    assert "path_loss_db" not in code, (
        "evaluate_path 应当通过 fold_roadm_terms 取端口损耗，而不是自己调 path_loss_db —— "
        "折叠逻辑只允许有一处（FG-11 的性质）。"
    )
    # 合并式写法的特征：插入损耗与滤波惩罚被加在一起。
    assert "insertion_loss_db + roadm.filtering_penalty_db" not in code, (
        "合并式 ROADM 装配回来了：那正是 FG-13 裁决的缺陷。"
    )


# ======================================================================
# FG13-3  修复的影响是物理量级（不是浮点噪声）
# ======================================================================


def test_fg13_impact_is_physics_scale_not_float_noise() -> None:
    """FG-13 量化的影响是 dB 量级；FG-14 之后由**冻结的修复前基线**继续担保。

    FG-13 交付时这条比较的是"若修复"与"现状"。FG-14 §2 真的修复了之后，
    ``scripts/fg13_impact_probe.py`` 的 before/after 两侧变成同一套装配，差值归零
    —— 那不是"影响变小"，而是这把尺子**已经完成使命**。

    ⇒ 数字改由 FG-14 的产物担保：``results/fg14_pre_fix_baseline.json`` 冻结了
    修复前的读数，当前树与它的差**就是**当初量化的那个影响量。这里断言两份
    独立产物的数字一致，任何一侧被悄悄改写都会红。
    """
    report = json.loads(IMPACT_JSON.read_text(encoding="utf-8"))
    snr = {
        f"{row['scenario_set'].split('/')[-1]}/{row['scenario']}": row["outputs"]["snr_db"]
        for rows in report.values()
        for row in rows
        if "snr_db" in row["outputs"]
    }
    fg13_roadm = snr["fg1_scenarios.py/roadm_hoop"]["max_abs_delta"]
    fg13_cplusl = snr["fg4a_scenarios.py/path_cplusl_roadm"]["max_abs_delta"]
    assert fg13_roadm > 0.9
    assert fg13_cplusl > 5.0

    # FG-14 的独立复算：冻结的修复前基线 vs 当前树。
    fg14 = json.loads(
        (REPO_ROOT / "results" / "fg14_pre_fix_baseline.json").read_text(encoding="utf-8")
    )
    assert fg14["schema"] == "fg14-pre-fix-baseline-v1"
    divergence = json.loads(
        (REPO_ROOT / "results" / "fg14_assembly_divergence.json").read_text(encoding="utf-8")
    )
    committed = divergence["committed"]["roadm_hoop"]["outputs"]["snr_db"]["authorized_max_abs_dev"]
    assert abs(committed - fg13_roadm) < 1e-12, (
        f"FG-13 量化的 {fg13_roadm!r} dB 与 FG-14 实测的 {committed!r} dB 不一致 —— "
        "两份产物必有一侧被改写。"
    )


def test_fg13_both_goldens_are_reproducible_after_the_repin() -> None:
    """FG-14 重钉后，两套既有金标准都必须仍逐位重现。

    * ``results/fg4a_golden.json`` —— **经** ``evaluate_path``
      （``path_cplusl_roadm``）；FG-14 §2 按授权重钉，重钉出处记在 ``repins`` 链里；
    * ``results/fu4_fidelity_golden.json`` —— **不经** ``evaluate_path``
      （走 ``QoTValidator``），对这一刀本就免疫，本断言顺带钉住这个结构性事实。
    """
    import subprocess
    import sys

    for script in ("fg4a_capture_golden", "fu4_capture_golden"):
        proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no user input
            [sys.executable, str(REPO_ROOT / "scripts" / f"{script}.py"), "--check"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            check=False,
        )
        assert proc.returncode == 0, f"{script} --check 失败:\n{proc.stdout}\n{proc.stderr}"


def test_fg13_fu4_golden_is_structurally_immune_to_the_repin() -> None:
    """``fu4_fidelity_golden.json`` 走 ``QoTValidator``，不经 ``evaluate_path``。

    所以 FG-14 的装配修复**不可能**动它。若哪天它也开始动，说明依赖关系变了
    （有人让 ``QoTValidator`` 也走了 ``evaluate_path``），那条"免疫"就不成立了。
    """
    report = json.loads(
        (REPO_ROOT / "results" / "fu4_fidelity_golden.json").read_text(encoding="utf-8")
    )
    serialised = json.dumps(report, ensure_ascii=False)
    assert "evaluate_path" not in serialised, (
        "fu4 金标准里出现了 evaluate_path 的字样，其免疫性论证需要重新检查。"
    )
