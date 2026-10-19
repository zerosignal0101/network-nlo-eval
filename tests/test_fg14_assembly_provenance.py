"""FG-14 命题 (B)：装配按裁决 —— 分歧记录 + "分歧不许悄悄扩大"的钉子。

FG-13 裁决 ``models/isrs_gn.py`` 的 ``evaluate_path`` 用合并式 ROADM 装配、
不读 ``express_attenuation_db`` 等四个端口/booster 字段，是**缺陷**
（``docs/fg13_evaluate_path_roadm.md``）。但把它改成权威口径会动到
``results/fg1_golden_main.json`` 的数字，而那份文件声明
``source_branch: "origin/main"``。

编排方（FG-14 brief §0）裁定的处置**不是**改数字，而是**把它所声称的命题说准确**：

* **(A) 内核逐位** —— 留在 ``tests/test_fg1_physics_parity.py``，**一格不动**；
* **(B) 装配按裁决** —— 就是本文件。装配遵循仓内权威定义
  ``ROADMConfig.path_loss_db()``，与 ``origin/main`` 的分歧被**显式记录**，
  并由一条**不允许分歧悄悄扩大**的钉子看守。

本文件钉四件事：

1. **main 出处的金标准是逐字节冻结的** —— 我们没有往里写 FG 树的数值；
2. **ODE 内核逐字节冻结** —— 证明这一系列重构没动 legacy 物理；
3. **记录可复算** —— 每个数字都能从 ``scripts/fg14_divergence_probe.py`` 重跑出来；
4. **钉子不是真空断言** —— 把分歧人为放大 ``1e-6``，钉子必须变红（§7 纪律）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "results" / "fg1_golden_main.json"
RECORD_PATH = REPO_ROOT / "results" / "fg14_assembly_divergence.json"
KERNEL_PATH = REPO_ROOT / "src" / "network_nlo_eval" / "models" / "numba_kernels_p2.py"

#: 提交 c621cda（FG-13）时这两份文件的 sha256。阶段 0 **零物理变化**，
#: 所以它们到阶段 0 结束必须一模一样；阶段 1 也不许变。
#:
#: ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：FG 的内核现在住在 ``numba_kernels_p2.py``
#: （**同字节、只换文件名**——sha256 与下表一字不差）。守卫**改指向**那份文件，
#: **不是放宽**：断言的 sha256 一个字符都没动，被守的仍是 FG 的那份字节。
#: 合并树另有一份 P1 内核（``numba_kernels.py``，sha256 6611c416…），它是**另一套
#: 物理**，由 ``tests/test_ode3_flexgrid_sync.py`` 的 J5 守卫单独守住。
FROZEN_SHA256 = {
    "results/fg1_golden_main.json": "4f4d9060ca273f337e71927ac0f116d9e2bf666ed9867102e81c230200a03794",
    "src/network_nlo_eval/models/numba_kernels_p2.py": (
        "7beece4ea353b1bf5119ab9708d4db9845b092328357fff30f9b806afe613205"
    ),
}


def _load_script(name: str) -> ModuleType:
    """按路径加载 ``scripts/`` 下的机理探针（它不是包，不能直接 import）。"""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fg14_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def probe() -> ModuleType:
    """按路径加载分歧探针（``scripts/`` 不是包，不能直接 import）。"""
    return _load_script("fg14_divergence_probe")


@pytest.fixture(scope="module")
def record() -> dict:
    """已提交的 (B) 分歧记录；缺失时给出可复制的重生成命令。"""
    if not RECORD_PATH.exists():
        pytest.fail(
            f"missing {RECORD_PATH}; regenerate it with "
            "PYTHONPATH=$PWD/src python scripts/fg14_divergence_probe.py --emit"
        )
    return json.loads(RECORD_PATH.read_text(encoding="utf-8"))


# ======================================================================
# FG14-B0  出处与内核：两份文件逐字节冻结
# ======================================================================


def test_fg14_main_provenance_golden_is_byte_frozen() -> None:
    """``fg1_golden_main.json`` 仍是 FG-13 交付时的那份字节。

    它声明 ``source_branch: "origin/main"``。把 FG 树的端口口径写进去，
    就是**伪造出处** —— 那正是 FG-13 拒绝做的事，本刀延续同一纪律。
    """
    assert _sha256(GOLDEN_PATH) == FROZEN_SHA256["results/fg1_golden_main.json"], (
        "声明来自 origin/main 的金标准被改动了。本刀授权的是**记录分歧**，"
        "不是**改写分歧**；分歧只允许出现在 results/fg14_assembly_divergence.json。"
    )
    assert json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["source_branch"] == "origin/main"


def test_fg14_ode_kernel_is_byte_frozen() -> None:
    """``numba_kernels_p2.py``（含 ``_calc_exact_power_profile_ode``）逐字节未变。

    这是命题 (A) 的物理侧支点：FG-11 复核过 ``off ≡ matvec ≤ 1.8e-15``，
    FG-14 只动 ROADM 装配那一段，内核必须一个字节都不动。

    ⚠️ WP-MERGE-FG-FLEX 第 3 轮：守卫**改指向** ``numba_kernels_p2.py``（FG 的内核
    同字节搬了家），断言值一字未改。
    """
    assert _sha256(KERNEL_PATH) == FROZEN_SHA256["src/network_nlo_eval/models/numba_kernels_p2.py"], (
        "ODE 内核被改动了 —— FG-14 只授权动装配层，触碰内核超出授权范围。"
    )


# ======================================================================
# FG14-B1  记录可复算
# ======================================================================


def test_fg14_assembly_divergence_record_reproduces(probe: ModuleType, record: dict) -> None:
    """``scripts/fg14_divergence_probe.py`` 重跑逐条复现记录里的读数。

    报告里的每个数字都必须能被重算，不能只活在文档里。
    """
    assert probe.build_record() == record


def test_fg14_record_declares_both_propositions_and_their_authority(record: dict) -> None:
    """记录必须同时写明 (A)/(B) 两条命题、权威定义位置与授权来源。"""
    assert record["propositions"]["A_kernel"]["claim"]
    assert record["propositions"]["B_assembly"]["claim"]
    assert "path_loss_db" in record["propositions"]["B_assembly"]["authoritative_definition"]
    assert record["propositions"]["B_assembly"]["adjudicated_by"]


# ======================================================================
# FG14-B2  钉子：分歧不许悄悄扩大
# ======================================================================


def test_fg14_divergence_stays_within_its_authorized_magnitude(probe: ModuleType, record: dict) -> None:
    """实测分歧不得超过记录里授权的幅度。

    这条**不是**"必须逐位等于 main"—— 阶段 1 之后授权幅度非零，它检查的是**上界**。
    语义：任何超出授权的移动（哪怕 1e-6）都必须先改授权、再改代码、再在文档归因。
    """
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    measured = probe.measure("committed", golden)
    assert probe.violations(record, measured) == []


def test_fg14_the_pin_is_not_a_vacuous_assertion(probe: ModuleType, record: dict) -> None:
    """把分歧人为放大 ``1e-6``，钉子**必须**变红（§7 纪律：不是真空断言）。

    没有这一条，上面那条"不得超出授权"就可能因为比较写错而恒真。
    这里直接把候选读数整体抬高 ``1e-6``，再要求钉子报出违反项。
    """
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    inflated = probe.measure("committed", golden, inflate=1e-6)
    problems = probe.violations(record, inflated)

    assert problems, (
        "把分歧放大 1e-6 之后钉子仍然是绿的 —— (B) 是真空断言，纪律失效。"
        "请检查 violations() 的比较逻辑，或 record 里 authorized_* 的字段名。"
    )
    # 而且必须是**真的**超出，而不是别的原因报错。
    assert any("超过授权幅度" in item for item in problems), problems[:5]


# ======================================================================
# FG14-B3  采纳档与对照档都留在记录里（让选择可复核）
# ======================================================================


def test_fg14_both_prospective_tiers_are_recorded(record: dict) -> None:
    """完整并集与"仅端口化"两档都在记录里，选择因此可复核。

    编排方 §0 采纳**完整并集**（含 booster 注入），但要求"仅端口化"那一档的
    ΔSNR **必须留在记录里**。缺了它，读者就无法判断这个选择值多少 dB。
    """
    for tier in ("committed", "prospective_full_union", "prospective_ports_only"):
        assert tier in record, f"记录缺少 {tier} 档"
        assert record[tier], f"{tier} 档是空的"
    assert record["adopted_tier"] == "prospective_full_union"
    assert record["adoption_rationale"]


def test_fg14_impact_table_is_measured_against_a_frozen_pre_fix_baseline(record: dict) -> None:
    """门 2 的 ΔSNR 表以**冻结的修复前基线**为参照，而不是以记忆为参照。

    基线在阶段 0（零物理变化）采集并提交，所以阶段 1 的每个 ΔSNR 都能回溯到
    "改动之前那个树"的具体读数。基线缺失 ⇒ 这张表无法复算 ⇒ 判失败。
    """
    table = record["impact_vs_pre_fix_baseline"]
    assert table is not None, (
        "缺少 impact_vs_pre_fix_baseline；先在阶段 0 跑 "
        "scripts/fg14_divergence_probe.py --emit-baseline"
    )
    for tier in ("committed", "prospective_ports_only", "prospective_full_union"):
        assert tier in table
    # 无 ROADM 的三个场景必须构造性不受影响，否则这张尺子本身有问题。
    for key in ("fg1/c_band_single", "fg1/multiband_wide", "fg1/kurtosis_64qam"):
        assert table["committed"][key]["snr_db"]["bit_exact"] is True, f"{key} 不应移动"
