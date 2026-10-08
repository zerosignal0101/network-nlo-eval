"""FG-13：`evaluate_path` 的 ROADM 施加点守卫。

裁决（详见 ``docs/fg13_evaluate_path_roadm.md``）：``models/isrs_gn.py`` 的
``evaluate_path`` 用合并式 ROADM 装配、**完全不读** ``express_attenuation_db``，
这是**缺陷**，不是"另一套刻意不同的模型"。

本文件钉住三件事：

1. **惰性字段清单本身**（逐位为 0 的那四个字段）——让下一个人看见它，而不是重新发现它；
2. **权威口径在哪里**（``ROADMConfig.path_loss_db`` + ``rwa.qot_checker.fold_roadm_terms``），
   并守住"折叠逻辑只有一处"这个 FG-11 已经建立的性质；
3. **按权威口径修复的影响量级**——证明它是物理量级改动，不是浮点噪声。

⚠️ **这些断言钉的是"当前状态"，不是"当前状态正确"。** 第 1 组断言一旦因合入修复而变红，
那是**预期的**：修复必须显式改这些断言并在文档里归因，不能悄悄改绿。反过来，
也**不要**因为它们现在是绿的就把惰性当成合理——修复的授权被
``results/fg1_golden_main.json`` 的出处约束挡住，见文档 §5。
"""

from __future__ import annotations

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
    """**已知缺陷，不是合理行为。**

    这四个字段在 ``ROADMConfig`` 里有明确物理语义、在 ``path_loss_db`` /
    ``fold_roadm_terms`` 里被权威消费，但 ``evaluate_path`` 一个都不读：

    * ``express_attenuation_db``（直通端口附加衰减）
    * ``add_drop_loss_db``（本地 add/drop 端口附加损耗）
    * ``booster_noise_figure_db``（补偿 ROADM 损耗的 booster NF）
    * ``equalize_output_power``（是否由 booster 恢复每信道入纤功率）

    实测是**逐位不变**（max|ΔSNR| 恰为 ``0.0``），不是"影响很小"。
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
        assert row["evaluate_path_snr_bitwise_identical"] is True, (
            f"{field} 不再是惰性字段（max|ΔSNR|={row['evaluate_path_max_abs_delta_db']}）。"
            "若这是刚合入的 ROADM 装配修复，请同步更新 "
            "docs/fg13_evaluate_path_roadm.md 并按机制归因本文件的期望值。"
        )
        assert row["evaluate_path_max_abs_delta_db"] == 0.0

    # 同一把尺子量 RWA 并集路径：这些字段在那里**是活的**。
    # 没有这一半，断言就退化成"evaluate_path 没读"，读不出"该读却没读"。
    assert by_field["express_attenuation_db"]["rwa_fold_express_loss_delta"] > 0.0
    assert by_field["add_drop_loss_db"]["rwa_fold_drop_loss_delta"] > 0.0
    assert by_field["express_attenuation_db"]["rwa_fold_drop_loss_delta"] == 0.0
    assert by_field["add_drop_loss_db"]["rwa_fold_express_loss_delta"] == 0.0


def test_fg13_last_hop_cancellation_is_not_mistaken_for_inertness() -> None:
    """守住 FG-13 最容易踩的假阳性。

    ``evaluate_path`` 把损耗同时除信号与**全部已累积噪声**，因此**末跳** ROADM 的损耗在
    ``power/(spm+xpm+ase)`` 里代数抵消。若拿一个**根本没被读**的字段去测末跳，
    同样会得到"逐位不变"，于是把"代数抵消"和"字段是死的"混为一谈。

    所以对照必须用一个 ``evaluate_path`` **确实读取**的字段（``insertion_loss_db``）：

    * 末跳 → 抵消（残差 ``3.55e-15 dB``，浮点残差而非严格逐位）；
    * 非末跳 → 可观测（约 ``2.55 dB``）。

    这条同时说明：**末跳 drop 损耗看起来"无效"是代数恒等，不是字段没被消费**，
    修复时不能据此宣称末跳端口损耗也不用管。
    """
    demo = json.loads(PROBE_JSON.read_text(encoding="utf-8"))["structural_demo"]

    last_hop = demo["on_LAST_hop_only"]
    assert last_hop["max_abs_delta_db"] < 1e-12, "末跳损耗应当代数抵消"
    assert last_hop["bitwise_identical"] is False, (
        "末跳抵消是数学恒等，浮点实现仍会留下 ~1e-15 dB 残差；"
        "若这里变成严格逐位相等，说明探针的比较口径变了，需要重新归因。"
    )

    non_last_hop = demo["on_NON_LAST_hop_only"]
    assert non_last_hop["max_abs_delta_db"] > 1.0, (
        "非末跳 ROADM 的损耗必须可观测；若此处也变成惰性，"
        "说明 evaluate_path 的噪声累积或施加点被改动了，那是比本刀更严重的问题。"
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


def test_fg13_evaluate_path_does_not_call_the_union_folding() -> None:
    """结构守卫：``evaluate_path`` 目前**没有**消费权威折叠。

    这是一条会随修复变红的断言，属于**有意的缺陷标记**：它让"evaluate_path 绕过
    ``fold_roadm_terms``"这件事在测试层面可见，而不是只写在文档里。
    """
    source = (REPO_ROOT / "src" / "network_nlo_eval" / "models" / "isrs_gn.py").read_text(encoding="utf-8")
    body = source.split("def evaluate_path(", 1)[1].split("\n    def ", 1)[0]

    assert "path_loss_db" not in body, "evaluate_path 已改为按端口取损耗，请更新本文件与文档的期望值"
    assert "express_attenuation_db" not in body
    assert "add_drop_loss_db" not in body
    # 合并式写法的特征：插入损耗与滤波惩罚被加在一起。
    assert "insertion_loss_db + roadm.filtering_penalty_db" in body


# ======================================================================
# FG13-3  修复的影响是物理量级（不是浮点噪声）
# ======================================================================


def test_fg13_impact_is_physics_scale_not_float_noise() -> None:
    """按权威口径修复后，既有场景的 ΔSNR 是 dB 量级。

    量化的意义：它证明这一刀动的是**物理**而不是数值。若哪天这个断言的实测值掉到
    1e-13 量级，说明修复的实现退化成浮点噪声，必须重新归因。
    """
    report = json.loads(IMPACT_JSON.read_text(encoding="utf-8"))
    snr = {
        f"{row['scenario_set'].split('/')[-1]}/{row['scenario']}": row["outputs"]["snr_db"]
        for rows in report.values()
        for row in rows
        if "snr_db" in row["outputs"]
    }

    # 无 ROADM 的场景构造性地不受影响。
    assert snr["fg1_scenarios.py/c_band_single"]["bitwise_identical"] is True
    assert snr["fg1_scenarios.py/multiband_wide"]["bitwise_identical"] is True
    assert snr["fg1_scenarios.py/kurtosis_64qam"]["bitwise_identical"] is True

    # 含 ROADM 的两个场景移动 dB 量级。
    for key, floor in (("fg1_scenarios.py/roadm_hoop", 0.9), ("fg4a_scenarios.py/path_cplusl_roadm", 5.0)):
        assert not snr[key]["bitwise_identical"], f"{key} 不应逐位不变"
        assert snr[key]["max_abs_delta"] > floor, (
            f"{key} 的 ΔSNR={snr[key]['max_abs_delta']} dB 掉到了物理量级以下，"
            "请重新归因后再更新本断言。"
        )
        assert snr[key]["max_rel_delta"] > 1e-3


def test_fg13_unpatched_tree_still_reproduces_both_goldens() -> None:
    """FG-13 未改 ``evaluate_path`` 数值 ⇒ 两套既有金标准必须仍逐位重现。

    * ``results/fg4a_golden.json`` —— **经** ``evaluate_path``（``path_cplusl_roadm``）；
    * ``results/fu4_fidelity_golden.json`` —— **不经** ``evaluate_path``（走
      ``QoTValidator``），所以它对这一刀本就免疫，本断言顺带钉住这个结构性事实。
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
