"""WP-O2 验收守卫（brief §7）。

验收标准 = ``ecz-dev`` 上既有的 6 条 + 本文件新增的 6 条，既有 6 条一条不许消失。
跑测试 ``PYTHONPATH=$PWD/src``。

每条守卫对应 brief §7 的一条编号。物理一律**不重算**：读 ``results/*.json``
并逐位复算断言，物理本身的正确性由 ``tools/ode_formula_map.py --check``
（J1）与 ``tools/ode_verdicts.py --check``（J1/J4）把关。
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PAPERS = REPO / "papers"
PY = "/home/zsig/miniconda3/envs/rl/bin/python"

LEGACY_SCENARIOS = {
    "legacy_two_hop_equalize_on",
    "legacy_three_hop_equalize_on",
    "legacy_two_hop_equalize_off",
    "legacy_express_attenuation",
    "legacy_high_filtering",
    "legacy_blocked_channel",
}


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        pytest.skip(f"{name} 尚未落盘；先跑 tools/ode_compare.py --write")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 守卫 1：公式映射守卫（每条物理公式都有论文公式号）
# ---------------------------------------------------------------------------
def test_guard1_formula_map_every_physics_entry_has_paper():
    doc = _load("ode_dev_formula_map.json")
    physics = doc["physics"]
    assert physics, "formula_map 不得为空"
    for entry in physics:
        assert entry["class"] == "physics"
        paper = entry.get("paper")
        assert paper, f"{entry['id']}: 缺少 paper 字段"
        assert paper.get("locator"), f"{entry['id']}: paper.locator 为空"
        assert paper.get("anchor"), f"{entry['id']}: paper.anchor 为空"
    for entry in doc["engineering"]:
        assert entry.get("paper_note"), f"{entry['id']}: 工程条目必须写明 paper_note"
    assert doc["summary"]["physics_total"] == len(physics)


# ---------------------------------------------------------------------------
# 守卫 2：拒绝项守卫（J4 —— ase_gain_source 必须在 rejected 里）
# ---------------------------------------------------------------------------
def test_guard2_ase_gain_source_is_rejected():
    doc = _load("ode_dev_rejected.json")
    rejected_ids = {i["id"] for i in doc["rejected"]}
    assert "ase_gain_source" in rejected_ids, "J4 失败：ase_gain_source 未被拒绝"
    item = next(i for i in doc["items"] if i["id"] == "ase_gain_source")
    assert item["verdict"] == "reject"
    assert item["paper"]["locator"] == "Eq. (34)", "拒绝理由必须引 P1 公式号"
    assert doc["summary"]["ase_gain_source_rejected"] is True


# ---------------------------------------------------------------------------
# 守卫 3：legacy 逐位守卫
#   差 == 各 fix_to_paper 项贡献之和（容差 1e-9 dB）
# ---------------------------------------------------------------------------
def test_guard3_legacy_bitwise_matches_sum_of_fixes():
    verdicts = _load("ode_dev_rejected.json")
    diff = _load("ode_dev_diff_vs_ecz.json")
    fixes = [i for i in verdicts["items"] if i["verdict"] == "fix_to_paper"]
    expected = sum(float(i.get("quantified_effect_db") or 0.0) for i in fixes)

    rows = diff["scenarios"]
    assert len(rows) == 6
    for row in rows:
        assert row["scenario"] in LEGACY_SCENARIOS
        if row["ode_dev_error"]:
            pytest.fail(f"{row['scenario']}: ode-dev 运行报错 {row['ode_dev_error']}")
        if row["delta_db"] is None:
            # legacy_blocked_channel：两端都拒绝，没有 SNR 可比
            assert row["accepted_ode_dev"] is False and row["accepted_other"] is False
            continue
        assert abs(row["delta_db"] - expected) < 1e-9, (
            f"{row['scenario']}: Δ={row['delta_db']} 与 fix_to_paper 贡献之和 {expected} 不符"
        )
    assert diff["summary"]["n_bit_identical"] == 6, (
        f"期望 6 行全部逐位相同（5 行有可比 SNR + 1 行两端都拒绝），"
        f"实得 {diff['summary']['n_bit_identical']}"
    )


# ---------------------------------------------------------------------------
# 守卫 4：能力存在性守卫（只测可达，不测物理）
# ---------------------------------------------------------------------------
def test_guard4_capabilities_reachable():
    import network_nlo_eval.core.flexgrid as flexgrid  # noqa: F401  可导入
    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import ODE_ACCEL_MODES, MultiBandISRSGN
    from network_nlo_eval.network.elements import (
        EDFAConfig,
        FiberSpanConfig,
        RamanPumpConfig,
    )

    # (a) ode_accel 档位可选
    assert ODE_ACCEL_MODES  # 至少一档
    fiber = FiberSpanConfig(length_km=80.0)
    grid = SpectrumGrid(8, 50e9, 193.1e12)
    power = np.full(8, 1e-3)
    for mode in ODE_ACCEL_MODES:
        ev = MultiBandISRSGN(grid, fiber, 32e9, ode_accel=mode)
        out = ev._calc_span_noise_and_power(power, 80e3, EDFAConfig(target_gain_db=17.0))
        assert np.all(np.isfinite(out[3]))
    with pytest.raises(ValueError):
        MultiBandISRSGN(grid, fiber, 32e9, ode_accel="not_a_mode")

    # (b) flexible grid 可达
    fg = FlexibleGrid.from_legacy_grid(8, 50e9, 193.1e12, slots_per_channel=4)
    conns = [
        ConnectionSpec(
            grid=fg, connection_id=i, first_slot=i * 4, n_slots=4,
            bit_rate_bps=100e9, modulation="QPSK", symbol_rate_hz=32e9,
            roll_off=0.0, launch_power_w=1e-3,
        )
        for i in range(8)
    ]
    ev_flex = MultiBandISRSGN(fg, fiber, 32e9, connections=conns)
    assert ev_flex.n_nodes == 8
    assert ev_flex.f_M_hz > ev_flex.f_m_hz

    # (c) 拉曼泵浦路径可达（只测 k_bar 注入，不测物理）
    pumped = FiberSpanConfig(
        length_km=80.0, raman_pumps=[RamanPumpConfig(wavelength_nm=1450.0, power_w=0.5)]
    )
    assert MultiBandISRSGN(grid, pumped, 32e9).k_bar == pytest.approx(0.5)
    assert MultiBandISRSGN(grid, fiber, 32e9).k_bar == 0.0

    # (d) 任意功率输入路径可达（target_gain_db=None → 自动均衡）
    auto = MultiBandISRSGN(grid, fiber, 32e9)._calc_span_noise_and_power(
        power, 80e3, EDFAConfig(target_gain_db=None, noise_figure_db=5.0)
    )
    assert np.allclose(auto[0], power)


def test_guard4b_cached_tier_is_bit_identical_to_off():
    """cached 档是纯 memoization，必须与 off 逐位相等（不得重排浮点求和）。"""
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    fiber = FiberSpanConfig(length_km=80.0)
    grid = SpectrumGrid(8, 50e9, 193.1e12)
    power = np.full(8, 1e-3)
    edfa = EDFAConfig(target_gain_db=17.0, noise_figure_db=5.0)

    off = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off")
    cached = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="cached")
    a = off._calc_span_noise_and_power(power, 80e3, edfa)
    b = cached._calc_span_noise_and_power(power, 80e3, edfa)
    c = cached._calc_span_noise_and_power(power, 80e3, edfa)  # 命中缓存
    for x, y in ((a, b), (a, c)):
        for arr_x, arr_y in zip(x, y, strict=True):
            assert np.array_equal(arr_x, arr_y)


# ---------------------------------------------------------------------------
# 守卫 5：三树对照守卫（全部 6 个场景都被覆盖）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", ["ode_dev_diff_vs_ecz.json", "ode_dev_diff_vs_fu4.json"])
def test_guard5_diffs_cover_all_six_scenarios(filename):
    doc = _load(filename)
    names = {r["scenario"] for r in doc["scenarios"]}
    assert names == LEGACY_SCENARIOS, f"{filename}: 场景集合不完整 {names ^ LEGACY_SCENARIOS}"
    assert len(doc["scenarios"]) == 6
    assert doc["summary"]["n_scenarios"] == 6


def test_guard5b_formula_map_and_rejected_also_landed():
    """三份 JSON 证据都在（formula_map / diff_vs_ecz / diff_vs_fu4）+ rejected。"""
    for name in (
        "ode_dev_formula_map.json",
        "ode_dev_rejected.json",
        "ode_dev_diff_vs_ecz.json",
        "ode_dev_diff_vs_fu4.json",
    ):
        assert (RESULTS / name).is_file(), f"缺少落盘证据 {name}"


# ---------------------------------------------------------------------------
# 守卫 6：论文引用守卫（docs 里每个公式号都能在 papers/*.md 里 grep 到）
# ---------------------------------------------------------------------------
def test_guard6_every_formula_number_cited_in_docs_is_greppable():
    doc_path = REPO / "docs" / "ode2_papers_and_diff.md"
    assert doc_path.is_file(), "缺少 docs/ode2_papers_and_diff.md"
    text = doc_path.read_text(encoding="utf-8")

    paper_text = "\n".join(p.read_text(encoding="utf-8") for p in sorted(PAPERS.glob("*.md")))

    fmap = _load("ode_dev_formula_map.json")
    cited: set[str] = set()
    for entry in fmap["physics"] + [
        e for e in fmap["engineering"] if e.get("paper_note")
    ]:
        if entry.get("paper"):
            cited.add(entry["paper"]["anchor"])
    for item in _load("ode_dev_rejected.json")["items"]:
        for key in ("paper", "paper2"):
            if item.get(key):
                cited.add(item[key]["anchor"])

    for anchor in cited:
        assert anchor in paper_text, f"docs 引用的锚点 {anchor!r} 在 papers/*.md 里 grep 不到"
        assert anchor in text, f"锚点 {anchor!r} 出现在 formula_map 但未出现在 docs"

    # docs 里出现的 P1/P2 公式号必须在公式表里有对应条目
    for entry in fmap["physics"]:
        locator = entry["paper"]["locator"]
        number = locator[locator.find("(") + 1: locator.find(")")] if "(" in locator else None
        if number and number.isdigit():
            assert locator in text, f"公式号 {locator} 未出现在 docs"


# ---------------------------------------------------------------------------
# 交叉守卫：tools 可 --check 逐位复算
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "script", ["ode_formula_map.py", "ode_verdicts.py", "ode_compare.py"]
)
def test_tools_check_reproduces(script):
    proc = subprocess.run(
        [PY, str(REPO / "tools" / script), "--check"],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        timeout=3600,
    )
    assert proc.returncode == 0, f"{script} --check 失败:\n{proc.stdout}\n{proc.stderr}"
