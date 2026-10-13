"""WP-O3 验收守卫（brief §4）。

验收标准 = WP-O2 达成的 17 条（ecz-dev 基线 6 + O2 新增 11）+ 本文件新增的守卫，
**既有 17 条一条不许少**。跑测试 ``PYTHONPATH=$PWD/src``。

每条守卫对应 brief §4 的一个编号。物理一律**不重算**：读 ``results/*.json``
并逐条断言，真正的物理正确性由 ``tools/ode_formula_map.py --check``（J1）、
``tools/ode_verdicts.py --check``（J1/J4）与 ``tools/ode_compare.py --check``
（三树对照）把关。

本文件刻意**不**重新实现 SNR 计算——那会造出第四套口径。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PY = "/home/zsig/miniconda3/envs/rl/bin/python"

#: WP-O2 达成的条数（ecz-dev 基线 6 + O2 新增 11）。
BASELINE_TESTS = 17

FROZEN_KERNEL = "src/network_nlo_eval/models/numba_kernels.py"
#: J5 冻结基线。WP-Q1 按 J2 授权改过冻结核（B_i 逐信道化，P1 (23)/(29)/(30)/(34)），
#: 基线随之推进到该提交 ``b69563a``；详见 tools/ode3_sync.py 的 WP_BASE 注释。
WP_BASE = "b69563a"


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        pytest.skip(f"{name} 尚未落盘；先跑对应的 tools/ode3_*.py --write")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_check(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, str(REPO / "tools" / script), "--check"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        timeout=3600,
    )


# ---------------------------------------------------------------------------
# 守卫 1（J6）：6 个 legacy 场景 delta_db == 0.0 —— 不许放宽
# ---------------------------------------------------------------------------
def test_guard1_j6_legacy_bit_exact_regression():
    """J6：``ode-dev`` 与 ``ecz-dev`` 在 6 个 legacy 场景上必须逐位相同。"""
    doc = _load("ode_dev_legacy_regression.json")
    assert doc["judgment"] == "J6"
    assert doc["tolerance"] == 0.0, "J6 容差必须是 0.0，不许放宽"

    scenarios = doc["scenarios"]
    assert len(scenarios) == 6, f"场景数 {len(scenarios)} != 6"

    for row in scenarios:
        if row["both_rejected"]:
            # 两端一致拒绝：没有 SNR 可比，但必须都拒。
            assert row["accepted_ode_dev"] is False and row["accepted_ecz_dev"] is False
            assert row["ode_dev_snr_db"] is None
            continue
        assert row["delta_db"] == 0.0, (
            f"{row['scenario']}: delta_db = {row['delta_db']!r}，J6 要求逐位相同 (0.0)"
        )
        assert row["bit_identical"], f"{row['scenario']}: 非逐位相同"
        assert row["ode_dev_snr_db"] == row["ecz_dev_snr_db"]

    assert doc["summary"]["j6_passed"] is True


# ---------------------------------------------------------------------------
# 守卫 2（J5）：numba_kernels.py 的 diff 必须为空
# ---------------------------------------------------------------------------
def test_guard2_j5_frozen_kernel_diff_is_empty():
    """J5：物理内核相对本 WP 起点一字未改。"""
    proc = subprocess.run(
        ["git", "diff", WP_BASE, "--", FROZEN_KERNEL],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert proc.stdout.strip() == "", (
        f"J5 被破坏：{FROZEN_KERNEL} 相对 {WP_BASE} 的 diff 非空\n{proc.stdout[:2000]}"
    )

    doc = _load("ode_dev_flexgrid_sync.json")
    assert doc["j5_frozen_kernel"]["file"] == FROZEN_KERNEL
    assert doc["j5_frozen_kernel"]["diff_vs_wp_base_empty"] is True


# ---------------------------------------------------------------------------
# 守卫 3（J7）：每段都有 kind，且没有 physics 片段被应用
# ---------------------------------------------------------------------------
def test_guard3_j7_every_segment_classified_no_physics_applied():
    """J7：逐片段 engineering|physics 标签齐全；物理片段**只有**凭 J2 论文授权才可应用。

    WP-Q1 起本闸门不是被拆掉，而是**收紧成例外制**：物理片段仍默认不许应用；
    真要应用，必须自带非空的 ``paper_formulas``（论文公式号清单），
    且该清单里的每一项都要在 ``results/ode_dev_formula_map.json`` 里真实存在。
    """
    doc = _load("ode_dev_flexgrid_sync.json")
    segments = doc["segments"]
    assert segments, "sync 记录为空"

    formula_map = json.loads((RESULTS / "ode_dev_formula_map.json").read_text(encoding="utf-8"))
    known_locators = {
        str(e.get("paper", {}).get("locator", "")).replace("Eq. ", "").replace("  ", " ").strip()
        for e in formula_map["physics"]
    }

    for seg in segments:
        assert seg["kind"] in ("engineering", "physics"), (
            f"{seg['file']}::{seg['symbol']} 缺少合法 kind（实际 {seg['kind']!r}）"
        )
        assert seg.get("rationale"), f"{seg['file']}::{seg['symbol']} 缺少理由"

    # 未授权的物理片段：一条都不许存在。
    unauthorized = [
        s for s in segments
        if s["kind"] == "physics" and s["action"] != "not_ported" and not s.get("paper_formulas")
    ]
    assert unauthorized == [], (
        f"物理片段被应用却无 paper_formulas 授权：{[s['symbol'] for s in unauthorized]}"
    )

    # 已授权的物理片段：授权的公式号必须真实存在于公式映射里。
    for seg in segments:
        if seg["kind"] == "physics" and seg["action"] != "not_ported":
            for ref in seg["paper_formulas"]:
                assert any(ref in loc or loc in ref for loc in known_locators), (
                    f"{seg['symbol']} 引用了公式映射里不存在的公式号 {ref!r}"
                )

    applied_physics = [
        s for s in segments if s["kind"] == "physics" and s["action"] != "not_ported"
    ]
    assert doc["summary"]["n_physics_applied"] == len(applied_physics)
    assert doc["summary"]["n_physics"] >= 1, "应至少记录一条 physics 片段作为 J7 的证明"
    # 不允许出现"整段缺失"：applied 的片段必须能在源码里 grep 到锚点。
    for seg in segments:
        if seg.get("verify") and seg["action"] != "not_ported":
            text = (REPO / seg["file"]).read_text(encoding="utf-8")
            assert seg["verify"] in text, f"{seg['file']} 缺锚点 {seg['verify']!r}"
        if seg.get("verify_absent") and seg["action"] == "not_ported":
            text = (REPO / seg["file"]).read_text(encoding="utf-8")
            assert seg["verify_absent"] not in text, (
                f"{seg['file']} 竟含声称未移植的 {seg['verify_absent']!r}"
            )


# ---------------------------------------------------------------------------
# 守卫 4（J8）：端到端跑通，逐连接 SNR 非空，且覆盖 >=2 个不同 n_slots
# ---------------------------------------------------------------------------
def test_guard4_j8_end_to_end_flexgrid_scenario():
    """J8：flexgrid 全链路真正可达，且逐连接 SNR 已落盘。"""
    doc = _load("ode_dev_flexgrid_e2e.json")
    assert doc["judgment"] == "J8"

    # >= 2 个不同 n_slots（brief §6 的硬验收条件）
    assert doc["n_distinct_n_slots"] >= 2, (
        f"只有 {doc['n_distinct_n_slots']} 个不同 n_slots（{doc['distinct_n_slots']}）"
    )
    assert doc["n_distinct_n_slots"] == len(set(doc["distinct_n_slots"]))

    per = doc["per_connection"]
    assert len(per) == doc["n_connections"] >= 2

    # 逐连接 SNR 非空
    for row in per:
        assert row["snr_db"] is not None, f"连接 {row['connection_id']} 的 SNR 为空"
        assert isinstance(row["n_slots"], int) and row["n_slots"] >= 1
        assert row["centre_frequency_hz"] > 0
        assert row["band"] in doc["bands"]["names"]

    # 索引空间：一条连接 = 一个 ODE 节点
    assert doc["evaluator_index_space_ok"] is True
    assert doc["n_nodes_in_evaluator"] == doc["n_connections"]

    # 端到端确实被接受（而非静默拒绝）
    assert doc["service"]["accepted"] is True

    # 探针：逐连接异构带宽必须被**接受**（WP-Q1 能力，P1 (23)/(29)/(30)/(34) 的 B_i 逐信道）
    assert doc["probes"]["heterogeneous_rs"]["accepted"] is True
    # 探针：槽区间重叠语义
    assert doc["probes"]["blocked_slot"]["passed"] is True
    # 探针：跨频段边界宽连接必须被显式拒绝
    assert doc["probes"]["band_crossing"]["rejected"] is True


# ---------------------------------------------------------------------------
# 守卫 5：能力矩阵每一项都有 involves_physics，且覆盖全部 7 个触点文件
# ---------------------------------------------------------------------------
def test_guard5_capability_matrix_has_involves_physics_and_full_coverage():
    """brief §4 守卫 5 + §6「能力矩阵覆盖 fu4 的全部 flexgrid 触点」。"""
    doc = _load("ode_dev_capability_matrix.json")
    caps = doc["capabilities"]
    assert caps

    for cap in caps:
        assert isinstance(cap.get("involves_physics"), bool), (
            f"{cap['capability']} 的 involves_physics 必须是布尔字段"
        )
        assert cap["status"] in ("both", "fu4_only", "ode_dev_only", "ode_dev_better")
        assert cap.get("evidence"), f"{cap['capability']} 缺少证据"

    covered = {cap["file"] for cap in caps}
    for rel in doc["touchpoint_files"]:
        assert rel in covered, f"能力矩阵未覆盖触点文件 {rel}"
    # 矩阵还额外覆盖了 J5 冻结内核（numba_kernels.py），故 >= 触点数。
    assert doc["summary"]["n_files_covered"] >= len(doc["touchpoint_files"])


# ---------------------------------------------------------------------------
# 守卫 6：既有测试一条不许少（ecz-dev 6 + O2 11 = 17）
# ---------------------------------------------------------------------------
def test_guard6_baseline_tests_not_lost():
    """既有 17 条测试必须仍在收集集合里。"""
    proc = subprocess.run(
        [PY, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        timeout=1800,
    )
    assert proc.returncode == 0, f"pytest 收集失败:\n{proc.stdout[-2000:]}"
    lines = [ln for ln in proc.stdout.splitlines() if "::" in ln]
    assert len(lines) >= BASELINE_TESTS, (
        f"只收集到 {len(lines)} 条测试，少于 WP-O2 达成的 {BASELINE_TESTS} 条"
    )
    # O2 的守卫文件必须仍在
    assert any("test_ode_dev.py" in ln for ln in lines), "WP-O2 的 tests/test_ode_dev.py 消失了"


# ---------------------------------------------------------------------------
# 守卫 7：bands.py 的最小闭包（不搬 FU-2 资产）+ 三个接入点可达
# ---------------------------------------------------------------------------
def test_guard7_bands_minimal_closure_has_three_access_points():
    """brief §0 定位的三个 flexgrid 接入点必须可达，且 FU-2 资产不在。"""
    from network_nlo_eval.core.bands import BandPlan, BandSpec
    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid

    assert hasattr(BandPlan, "from_connections")
    assert hasattr(BandPlan, "assert_connections_single_band")
    assert hasattr(BandPlan, "apply_launch_power")

    # FU-2 资产确实没被搬进来
    text = (REPO / "src/network_nlo_eval/core/bands.py").read_text(encoding="utf-8")
    for absent in ("SCLU_BAND_SPECS: tuple[BandSpec, ...]", "def sclu_span_report(",
                   "def guard_report(", "def hz_to_nm(", "U_BAND_ASSUMPTIONS: tuple[str, ...] = ("):
        assert absent not in text, f"bands.py 竟含未移植的 FU-2 资产 {absent!r}"

    # 三个接入点真能跑（2 槽 + 3 槽，跨两个频段）
    grid = FlexibleGrid(slot_width_hz=12.5e9, num_slots=240, band_low_edge_hz=191.0e12)
    specs = (
        BandSpec("L", 1565.0, 1625.0, "EDFA", 5.0, launch_power_w=1.0e-3),
        BandSpec("C", 1530.0, 1565.0, "EDFA", 4.5, launch_power_w=0.5e-3),
    )
    conns = [
        ConnectionSpec(grid=grid, connection_id=0, first_slot=0, n_slots=2,
                       bit_rate_bps=50e9, modulation="QPSK", symbol_rate_hz=50e9,
                       roll_off=0.0, launch_power_w=0.0),
        ConnectionSpec(grid=grid, connection_id=1, first_slot=80, n_slots=3,
                       bit_rate_bps=100e9, modulation="8QAM", symbol_rate_hz=50e9,
                       roll_off=0.0, launch_power_w=0.0),
    ]
    plan = BandPlan.from_connections(conns, specs)
    assert plan.band_names == ("L", "C")
    assert plan.band_of(0) == 0 and plan.band_of(1) == 1
    plan.assert_connections_single_band(conns)

    powered = plan.apply_launch_power(conns)
    # frozen dataclass => 返回新对象，原列表不变
    assert all(c.launch_power_w == 0.0 for c in conns)
    assert [c.launch_power_w for c in powered] == [1.0e-3, 0.5e-3]


# ---------------------------------------------------------------------------
# 守卫 8：O2 公式锚点行号仍对齐（接线只许移位，不许改物理）
# ---------------------------------------------------------------------------
def test_guard8_formula_map_anchors_still_aligned():
    """J1 交叉守卫：接线导致行号漂移，但物理锚点内容必须仍能 grep 到。"""
    doc = _load("ode_dev_anchor_relocation.json")
    assert doc["n_unfixable"] == 0, (
        f"有 {doc['n_unfixable']} 条锚点无法自动修正——那通常意味着物理真的被改了"
    )
    assert doc["n_relocated"] > 0, "本 WP 插入了接线，应当有锚点被重定位"

    # 每个被重定位的锚点，其 contains 串必须仍存在于目标文件的**新行号**上
    for reloc in doc["relocations"]:
        lines = (REPO / reloc["file"]).read_text(encoding="utf-8").splitlines()
        assert 1 <= reloc["new_line"] <= len(lines)
        assert reloc["contains"] in lines[reloc["new_line"] - 1], (
            f"{reloc['file']}:{reloc['new_line']} 不含 {reloc['contains']!r}"
        )


# ---------------------------------------------------------------------------
# 守卫 9：tools 可 --check 逐位复算
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "script",
    [
        "ode3_relocate_anchors.py",
        "ode3_sync.py",
        "ode3_flexgrid_e2e.py",
        "ode3_capability.py",
        "ode3_report.py",
    ],
)
def test_guard9_ode3_tools_check_reproduces(script):
    proc = _run_check(script)
    assert proc.returncode == 0, f"{script} --check 失败:\n{proc.stdout}\n{proc.stderr}"


def test_guard9b_ode2_tools_still_pass():
    """WP-O2 的工具必须仍通过（O2 的验收不许被本 WP 破坏）。"""
    for script in ("ode_formula_map.py", "ode_verdicts.py"):
        proc = _run_check(script)
        assert proc.returncode == 0, f"{script} --check 失败:\n{proc.stdout}\n{proc.stderr}"
