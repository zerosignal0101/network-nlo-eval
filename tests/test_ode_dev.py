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

#: ``ode_accel`` 里**只能**跑 legacy 拉曼谱的档位：``"off"`` 在核内逐对重建 legacy
#: 折线，``"cached"`` 只是把它记忆化，两者结构上都无法换谱。FG-15 定的口径是
#: "显式拒绝而非静默回退"，因此这些档位必须由调用方显式配 ``raman_spectrum``。
LEGACY_ONLY_ACCEL = ("off", "cached")


def _raman_tier_for(ode_accel: str, nli_model: str = "p2_gn") -> str:
    """返回该 ``(nli_model, ode_accel)`` 组合可用的 Raman 谱档位。

    ⚠️ WP-MERGE-FG-FLEX **第 4 轮**：``p1_closed`` 档的拉曼谱**只有** ``"legacy"``
    （它走的是 P1 核里的 legacy 合成折线，没有 tabular 闭式增益谱那条路）——
    这与 ``ode_accel`` 选哪个**无关**：``p1_closed`` + ``"matvec"`` 也必须配
    ``"legacy"``。第 4 轮给 ``p1_closed`` 加了 ``"matvec"`` 档之后，这条区分
    才变得必要（之前 ``p1_closed`` 只有 off/cached，两者都已在
    ``LEGACY_ONLY_ACCEL`` 里，按档位判就够用）。
    """
    if nli_model == "p1_closed":
        return "legacy"
    return "legacy" if ode_accel in LEGACY_ONLY_ACCEL else "tabular"


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
# 守卫 4b：NLI 物理档位与加速档位**不匹配**时必须显式报错（第 3 轮新增）
# ---------------------------------------------------------------------------
def test_guard4b_incompatible_ode_accel_is_rejected():
    """⚠️ WP-MERGE-FG-FLEX **第 3 轮新增 / 第 4 轮修订**：不匹配的组合必须**显式**报错。

    这条守卫的命题是「**没有的能力不许静默回落**」——给一个档位配它**没有**的
    加速结构，如果静默跑成 ``"off"``，调用方就**以为**自己拿到了加速：那是最坏的
    一种错，数字对、账对不上、没人知道加速其实没生效。

    ⚠️ **第 4 轮修订**：``"matvec"`` **不再是** P2 独有的了——第 4 轮给
    ``p1_closed`` 档补上了**自己的** matvec 实现（P1 闭式核的 XPM 信道对矩阵 +
    矩阵-向量乘，``models/numba_kernels_p1_accel.py``）。所以：

    * ``p1_closed`` + ``"cached_g"`` ⇒ **仍然**必须显式拒绝（``cached_g`` 是 FG 侧
      拉曼 G 矩阵缓存，P1 核走的是**解析**出纤功率，没有拉曼 G 矩阵可缓存）；
    * ``p1_closed`` + ``"matvec"`` ⇒ **现在合法**（配 ``legacy`` 谱）；
    * ``p1_closed`` + ``raman_spectrum="tabular"`` ⇒ **仍然**必须显式拒绝。

    ⇒ 命题**没变**，只是「哪些组合算不匹配」的集合变了。**没有**任何一档被
    改成静默回落。
    """
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    fiber = FiberSpanConfig(length_km=80.0)
    grid = SpectrumGrid(8, 50e9, 193.1e12)
    # ``cached_g`` 仍然只属于 p2_gn：P1 核没有拉曼 G 矩阵（G 缓存无处可缓存）。
    with pytest.raises(ValueError, match="nli_model"):
        MultiBandISRSGN(
            grid, fiber, 32e9, ode_accel="cached_g", raman_spectrum="legacy",
            nli_model="p1_closed",
        )
    # 反向：P1 档只有 legacy 谱，tabular 必须显式拒绝。
    with pytest.raises(ValueError, match="raman_spectrum"):
        MultiBandISRSGN(
            grid, fiber, 32e9, ode_accel="off", raman_spectrum="tabular",
            nli_model="p1_closed",
        )
    with pytest.raises(ValueError, match="raman_spectrum"):
        MultiBandISRSGN(
            grid, fiber, 32e9, ode_accel="matvec", raman_spectrum="tabular",
            nli_model="p1_closed",
        )


def test_guard4b_p1_matvec_tier_is_reachable_and_not_a_silent_off():
    """⚠️ WP-MERGE-FG-FLEX **第 4 轮新增**：``p1_closed`` 的 ``matvec`` 档必须**真的**是加速档。

    第 3 轮把 ``p1_closed`` + ``"matvec"`` 记成「P1 档没有这个能力」。第 4 轮
    补上了它，所以这条守卫要反过来盯住一件事：**别让它退化成静默的 ``"off"``**。

    判据是**可观测的差异**而不是"构造成功"：``p1_closed`` 的 ``matvec`` 档把 XPM
    的 ``l`` 方向求和从成对循环换成矩阵-向量乘 ⇒ 与 ``off`` 档**非逐位相等**
    （但仍在 1e-12 预算内，见 V2''）。如果哪天有人把 ``matvec`` 悄悄改回 ``off``，
    这个差异会**归零** ⇒ 守卫立刻响。

    同时守住「P1 的算式没被动过」：出纤功率与 SPM 必须与 ``off`` 档**逐位相同**。
    """
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    fiber = FiberSpanConfig(length_km=80.0)
    grid = SpectrumGrid(8, 50e9, 193.1e12)
    power = np.full(8, 1e-3)

    off = MultiBandISRSGN(
        grid, fiber, 32e9, ode_accel="off", raman_spectrum="legacy", nli_model="p1_closed"
    )
    mv = MultiBandISRSGN(
        grid, fiber, 32e9, ode_accel="matvec", raman_spectrum="legacy", nli_model="p1_closed"
    )
    a = off._calc_span_noise_and_power(power, 80e3, None)
    b = mv._calc_span_noise_and_power(power, 80e3, None)

    # 出纤功率与 SPM 逐位相同（走冻结核的同一个函数）。
    assert np.array_equal(a[0], b[0]), "P1 matvec 的出纤功率必须与 off 逐位相同"
    assert np.array_equal(a[1], b[1]), "P1 matvec 的 SPM 必须与 off 逐位相同"
    # XPM 非逐位（求和次序变了）——这正是「它真的走了加速路径」的证据。
    assert not np.array_equal(a[2], b[2]), (
        "P1 matvec 的 XPM 必须与 off 非逐位相等；若逐位相等，说明它退化成了 off"
    )
    # 且仍在 1e-12 预算内。
    denom = np.maximum(np.abs(a[2]), 1e-300)
    max_rel = float(np.max(np.abs(a[2] - b[2]) / denom))
    assert max_rel <= 1e-12, f"P1 matvec 的 XPM 相对偏差 {max_rel:.3e} 超出 1e-12 预算"


# ---------------------------------------------------------------------------
# 守卫 4：能力存在性守卫（只测可达，不测物理）
# ---------------------------------------------------------------------------
def test_guard4_capabilities_reachable():
    import network_nlo_eval.core.flexgrid as flexgrid  # noqa: F401  可导入
    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import (
        ODE_ACCEL_MODES,
        ODE_ACCEL_MODES_BY_NLI_MODEL,
        MultiBandISRSGN,
    )
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
    # ⚠️ WP-MERGE-FG-FLEX **第 3 轮**（已声明的测试改动，见 docs/wp_merge_fg_into_flex.md
    # V5'）：``ode_accel`` 档位现在**按 NLI 物理档位分组**
    # （``ODE_ACCEL_MODES_BY_NLI_MODEL``）——``"cached_g"`` / ``"matvec"`` 是 **P2**
    # 的实现结构（P1 核里没有拉曼 G 矩阵、没有去分支 matvec，也没有张量化 NLI）。
    #
    # 这条守卫的命题是「**每个能力档位都可达**」，所以这里走**笛卡尔积**：
    # 每个 NLI 档位 × 该档位支持的每个 ``ode_accel`` 档位，全部构造一次并跑一遍
    # 跨段计算。这比换底前（只按并集档位各跑一次）**多跑**了几个组合
    # （P1 档的 off/cached、P2 档的 off/cached），覆盖是**扩大**的。
    #
    # 另见 :func:`test_guard4b_incompatible_ode_accel_is_rejected`：**不匹配**的
    # 组合必须显式报错，不许静默回退。
    for nli_model, modes in ODE_ACCEL_MODES_BY_NLI_MODEL.items():
        for mode in modes:
            # WP-MERGE-FG-FLEX（已声明的测试改动，见 docs/wp_merge_fg_into_flex.md V5）：
            # 合并后 ``ode_accel`` 是两侧并集，``"off"`` / ``"cached"`` 是 legacy-only
            # 的冻结路径，**必须**显式配 ``raman_spectrum="legacy"``（FG-15 的口径是
            # "显式拒绝而非静默回退"）。此前这里不传该参数，是因为 Raman 档位当时还不存在。
            ev = MultiBandISRSGN(
                grid, fiber, 32e9, ode_accel=mode, raman_spectrum=_raman_tier_for(mode, nli_model),
                nli_model=nli_model,
            )
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
    #
    # ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：这条断言按 NLI 物理档位**拆成两段**。
    #
    # 第 2 轮把旧断言 `assert np.allclose(auto[0], power)`（flexgrid 侧口径：
    # 跨段函数**内部**施加 EDFA，`target_gain_db=None` 时把出纤功率自动均衡回
    # 入纤功率）改成了 FG 侧口径的 `auto[0] < power`（跨段函数返回**光纤输出**，
    # 跨段后放大由调用方负责，否则与调用方的 ASE 重复计入）。
    #
    # 第 3 轮查明：那是**两套不同物理**各自的契约，不能二选一：
    #   * ``p1_closed``（P1 闭式，flexgrid-dev @ cb24704）：跨段函数**内部**施加
    #     EDFA ⇒ 出纤功率被自动均衡回入纤功率（**旧断言原样恢复**）。
    #   * ``p2_gn``（P2 Eq. 21，wp/fg-integrate @ 57c7221）：跨段函数返回**光纤
    #     输出**，放大由调用方负责 ⇒ 出纤功率**小于**入纤功率。
    # 两段都在这里断言，谁都不被抹掉。
    p1_auto = MultiBandISRSGN(grid, fiber, 32e9, nli_model="p1_closed")._calc_span_noise_and_power(
        power, 80e3, EDFAConfig(target_gain_db=None, noise_figure_db=5.0)
    )
    assert np.all(np.isfinite(p1_auto[0]))
    assert np.allclose(p1_auto[0], power), (
        "P1 档下跨段函数内部施加 EDFA：target_gain_db=None 时出纤功率应被自动均衡回入纤功率"
    )

    p2_auto = MultiBandISRSGN(grid, fiber, 32e9, nli_model="p2_gn")._calc_span_noise_and_power(
        power, 80e3, EDFAConfig(target_gain_db=None, noise_figure_db=5.0)
    )
    assert np.all(np.isfinite(p2_auto[0]))
    assert np.all(p2_auto[0] > 0.0) and np.all(p2_auto[0] < power), (
        "P2 档下出纤功率应当是 80 km 损耗后的光纤输出，而不是入纤功率；"
        "若这里等于入纤功率，说明跨段内又加回了 EDFA（会与调用方的 ASE 重复计入）"
    )


def test_guard4b_cached_tier_is_bit_identical_to_off():
    """cached 档是纯 memoization，必须与 off 逐位相等（不得重排浮点求和）。"""
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    fiber = FiberSpanConfig(length_km=80.0)
    grid = SpectrumGrid(8, 50e9, 193.1e12)
    power = np.full(8, 1e-3)
    edfa = EDFAConfig(target_gain_db=17.0, noise_figure_db=5.0)

    off = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off", raman_spectrum="legacy")
    cached = MultiBandISRSGN(grid, fiber, 32e9, ode_accel="cached", raman_spectrum="legacy")
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
