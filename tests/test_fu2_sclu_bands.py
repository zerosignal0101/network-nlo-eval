"""FU-2：S/C/L/U 波段定义 + 逐波段发射功率的自检。

这一组测试盯的是**本刀自己的契约**，不去动任何既有期望值：

* S/C/L 边界与 ITU-T G.694.1 / RL 侧既有定义一致（**不自创**）；
* U 波段是**假设**，且假设逐条落盘、不为空；
* guard 间隔是**算出来的**（C|L、S|C 相接为 0；L|U 有 35 nm 真实空隙）；
* ``BandPlan`` 的覆盖/不重叠/非空段校验会抛错，而不是静默错分；
* **逐波段功率**能同时落到逐信道 profile 与逐连接 ``ConnectionSpec``，
  且未设定时拒绝构造（不静默填 0）；
* 横跨频段边界的连接被**显式拒绝**，不按中心频率静默归段；
* **回归锁**：`for_fiber` 必须透传 ``raman_spectrum``——这是 ``evaluate_path``
  能否真正用上 ``tabular`` 的唯一开关，FG-U1 时是断的。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.core.bands import (
    ASSUMED_U_HIGH_NM,
    ASSUMED_U_LOW_NM,
    CL_BAND_SPECS,
    CLS_BAND_SPECS,
    ITU_C_HIGH_NM,
    ITU_C_LOW_NM,
    ITU_L_HIGH_NM,
    ITU_L_LOW_NM,
    ITU_S_HIGH_NM,
    ITU_S_LOW_NM,
    SCLU_BAND_SPECS,
    U_BAND_ASSUMPTIONS,
    BandPlan,
    BandSpec,
    hz_to_nm,
    nm_to_hz,
    sclu_span_report,
)
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import FiberSpanConfig

THZ = 1e12


def _spec_by_name(specs, name):
    return next(spec for spec in specs if spec.name == name)


# ---------------------------------------------------------------------------
# 波长/频率换算
# ---------------------------------------------------------------------------
def test_nm_hz_roundtrip():
    assert nm_to_hz(hz_to_nm(193.1e12)) == pytest.approx(193.1e12, rel=1e-15)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nm_hz_rejects_non_positive(bad):
    with pytest.raises(ValueError, match="must be positive"):
        nm_to_hz(bad)
    with pytest.raises(ValueError, match="must be positive"):
        hz_to_nm(bad)


# ---------------------------------------------------------------------------
# S/C/L 边界：与 RL 侧既有定义对齐
# ---------------------------------------------------------------------------
def test_scl_edges_match_itu_g6941():
    assert (ITU_C_LOW_NM, ITU_C_HIGH_NM) == (1530.0, 1565.0)
    assert (ITU_L_LOW_NM, ITU_L_HIGH_NM) == (1565.0, 1625.0)
    assert (ITU_S_LOW_NM, ITU_S_HIGH_NM) == (1460.0, 1530.0)


def test_scl_boundaries_are_asymmetric_wavelengths():
    """S/C/L 边界是成对相接的：S 上沿 = C 下沿 = 1565 的一侧，C 上沿 = L 下沿。"""
    assert ITU_C_HIGH_NM == ITU_L_LOW_NM
    assert ITU_S_HIGH_NM == ITU_C_LOW_NM


def test_cl_band_order_is_low_to_high_frequency():
    assert [s.name for s in CL_BAND_SPECS] == ["L", "C"]
    assert [s.name for s in CLS_BAND_SPECS] == ["L", "C", "S"]


def test_sclu_order_is_low_to_high_frequency():
    """U 波长最长 => 频率最低 => 必须排第一。"""
    assert [s.name for s in SCLU_BAND_SPECS] == ["U", "L", "C", "S"]
    lows = [s.low_hz for s in SCLU_BAND_SPECS]
    assert lows == sorted(lows)


def test_band_frequency_edges_invert_wavelength_order():
    for spec in SCLU_BAND_SPECS:
        assert spec.low_hz < spec.high_hz
        assert hz_to_nm(spec.low_hz) == pytest.approx(spec.high_nm, rel=1e-12)
        assert hz_to_nm(spec.high_hz) == pytest.approx(spec.low_nm, rel=1e-12)


def test_band_spec_rejects_inverted_wavelengths():
    with pytest.raises(ValueError, match="low_nm < high_nm"):
        BandSpec("X", 1565.0, 1530.0)


def test_band_spec_rejects_negative_power():
    with pytest.raises(ValueError, match="launch_power_w must be non-negative"):
        BandSpec("X", 1530.0, 1565.0, launch_power_w=-1.0)


# ---------------------------------------------------------------------------
# U 波段：假设必须逐条落盘
# ---------------------------------------------------------------------------
def test_u_band_is_declared_as_assumption():
    u = _spec_by_name(SCLU_BAND_SPECS, "U")
    assert (u.low_nm, u.high_nm) == (ASSUMED_U_LOW_NM, ASSUMED_U_HIGH_NM)
    assert (ASSUMED_U_LOW_NM, ASSUMED_U_HIGH_NM) == (1660.0, 1770.0)
    assert u.source.startswith("assumption")


def test_u_band_assumptions_are_numbered_and_non_empty():
    assert len(U_BAND_ASSUMPTIONS) >= 4
    for index, text in enumerate(U_BAND_ASSUMPTIONS, start=1):
        assert text.startswith(f"U{index}:"), f"assumption {index} must be numbered"


def test_u_band_nf_is_worse_than_s_and_c():
    """U3：假设 U 的 NF 比 S 更差（频率更低 -> 增益效率更低），故应单调更差。"""
    u = _spec_by_name(SCLU_BAND_SPECS, "U")
    s = _spec_by_name(SCLU_BAND_SPECS, "S")
    c = _spec_by_name(SCLU_BAND_SPECS, "C")
    assert u.noise_figure_db > s.noise_figure_db > c.noise_figure_db


# ---------------------------------------------------------------------------
# 保护间隔：算出来的，不是断言的
# ---------------------------------------------------------------------------
def test_guard_report_zero_for_itu_adjacent_bands():
    # 栅格必须真的横跨 S/C/L 三段，否则会先触发"频段内 0 个信道"——
    # 417 道 @ 50 GHz、中心 194.8877 THz 正是 RL 侧错误信息里给出的那一组。
    grid = SpectrumGrid(num_channels=417, channel_spacing_hz=50e9, center_frequency_hz=194.8877e12)
    plan = BandPlan.from_channels(grid, CLS_BAND_SPECS)
    assert plan.num_bands == 3
    assert plan.element_count(0) > 0 and plan.element_count(1) > 0 and plan.element_count(2) > 0
    guards = {g["higher_frequency_band"]: g for g in plan.guard_report()}
    assert guards["C"]["touching"] is True
    assert guards["C"]["gap_thz"] == pytest.approx(0.0, abs=1e-9)
    assert guards["S"]["touching"] is True
    assert guards["S"]["gap_thz"] == pytest.approx(0.0, abs=1e-9)


def test_guard_report_shows_u_l_gap_of_35nm():
    report = sclu_span_report()
    # 1625 -> 1660 nm 的空隙（假设 U4）
    assert report["u_l_gap_thz"] == pytest.approx(3.889800196478, rel=1e-9)
    assert report["inter_band_gaps_thz"][1:] == [pytest.approx(0.0, abs=1e-9)] * 2


def test_span_report_is_self_consistent():
    report = sclu_span_report()
    # 包络 = 各段跨度之和 + 空隙，必须自洽
    assert report["envelope_span_thz"] == pytest.approx(
        report["sum_of_band_spans_thz"] + report["total_gap_thz"], rel=1e-12
    )
    assert report["envelope_span_thz"] == pytest.approx(35.96303, rel=1e-6)


# ---------------------------------------------------------------------------
# 半开区间语义
# ---------------------------------------------------------------------------
def test_band_contains_is_half_open():
    """左闭右开：共享边界归高频段，避免 RL 侧闭区间那种双重归属。"""
    l_band = _spec_by_name(CL_BAND_SPECS, "L")
    c_band = _spec_by_name(CL_BAND_SPECS, "C")
    shared = c_band.low_hz
    assert shared == pytest.approx(l_band.high_hz, rel=1e-12)
    assert l_band.contains_hz(shared) is False
    assert c_band.contains_hz(shared) is True


def test_contains_hz_array_matches_scalar():
    u = _spec_by_name(SCLU_BAND_SPECS, "U")
    probes = np.array([u.low_hz, u.high_hz - 1.0, u.high_hz, u.low_hz - 1.0])
    mask = u.contains_hz_array(probes)
    assert mask.tolist() == [u.contains_hz(float(p)) for p in probes]


# ---------------------------------------------------------------------------
# BandPlan 校验
# ---------------------------------------------------------------------------
def test_bandplan_rejects_overlapping_bands():
    specs = (BandSpec("A", 1530.0, 1565.0), BandSpec("B", 1540.0, 1600.0))
    freqs = np.array([nm_to_hz(1545.0)])
    with pytest.raises(ValueError, match="overlap"):
        BandPlan(freqs, specs)


def test_bandplan_rejects_non_adjacent_overlap():
    """非相邻的一对若重叠，只比相邻对发现不了——必须全对检查。

    排列 A, B, C：A|B 与 B|C 都不相交，只有 A|C 相交。若只查相邻对就会漏掉。
    """
    specs = (
        BandSpec("A", 1530.0, 1565.0),
        BandSpec("B", 1600.0, 1650.0),
        BandSpec("C", 1550.0, 1570.0),
    )
    freqs = np.array([nm_to_hz(1545.0)])
    with pytest.raises(ValueError, match="overlap"):
        BandPlan(freqs, specs)


def test_bandplan_rejects_uncovered_elements():
    specs = (BandSpec("A", 1530.0, 1565.0),)
    freqs = np.array([nm_to_hz(1550.0), nm_to_hz(1600.0)])
    with pytest.raises(ValueError, match="uncovered"):
        BandPlan(freqs, specs)


def test_bandplan_rejects_empty_band():
    specs = (BandSpec("A", 1530.0, 1565.0), BandSpec("B", 1700.0, 1720.0))
    freqs = np.array([nm_to_hz(1550.0)])
    with pytest.raises(ValueError, match="contain 0"):
        BandPlan(freqs, specs)


@pytest.mark.parametrize("bad_freqs", [np.array([]), np.array([1.0, 2.0]).reshape(2, 1)])
def test_bandplan_rejects_bad_frequency_arrays(bad_freqs):
    specs = (BandSpec("A", 1530.0, 1565.0),)
    with pytest.raises(ValueError):
        BandPlan(bad_freqs, specs)


def test_bandplan_rejects_non_positive_frequencies():
    specs = (BandSpec("A", 1530.0, 1565.0),)
    with pytest.raises(ValueError, match="finite and strictly positive"):
        BandPlan(np.array([-1.0]), specs)


def test_bandplan_rejects_empty_specs():
    with pytest.raises(ValueError, match="at least one band"):
        BandPlan(np.array([nm_to_hz(1550.0)]), ())


def test_bandplan_band_of_and_counts():
    specs = tuple(
        replace(s, launch_power_w=1e-3)
        for s in SCLU_BAND_SPECS
        if s.name in ("C", "L")
    )
    grid = SpectrumGrid(num_channels=200, channel_spacing_hz=50e9, center_frequency_hz=190e12)
    plan = BandPlan.from_channels(grid, specs)
    assert plan.band_names == ("L", "C")
    assert plan.element_count(0) + plan.element_count(1) == grid.num_channels
    assert plan.band_of(0) in (0, 1)


def test_bandplan_index_accessors_reject_out_of_range():
    specs = tuple(replace(s, launch_power_w=1e-3) for s in CL_BAND_SPECS)
    grid = SpectrumGrid(num_channels=100, channel_spacing_hz=50e9, center_frequency_hz=190e12)
    plan = BandPlan.from_channels(grid, specs)
    with pytest.raises(IndexError):
        plan.band_element_indices(2)
    with pytest.raises(IndexError):
        plan.band_of(plan.num_elements)
    with pytest.raises(TypeError):
        plan.band_element_indices(True)


# ---------------------------------------------------------------------------
# 逐波段发射功率 ★
# ---------------------------------------------------------------------------
def _flat_specs():
    return tuple(replace(s, launch_power_w=1e-3) for s in CL_BAND_SPECS)


def test_launch_power_profile_is_per_channel():
    specs = tuple(replace(s, launch_power_w=float(1e-3 * (i + 1))) for i, s in enumerate(CL_BAND_SPECS))
    grid = SpectrumGrid(num_channels=200, channel_spacing_hz=50e9, center_frequency_hz=190e12)
    plan = BandPlan.from_channels(grid, specs)
    profile = plan.launch_power_profile_w()
    assert profile.shape == (grid.num_channels,)
    l_idx, c_idx = plan.band_element_indices(0), plan.band_element_indices(1)
    assert set(np.unique(profile[l_idx]).tolist()) == {1e-3}
    assert set(np.unique(profile[c_idx]).tolist()) == {2e-3}
    # 逐连接顺序无关：profile 是按频段常量填充的
    assert plan.per_band_launch_power_w() == {"L": 1e-3, "C": 2e-3}


def test_launch_power_profile_rejects_unset_band():
    """未设定必须是显式异常，而不是静默填 0（0 在既有语义里 = 未占用）。"""
    grid = SpectrumGrid(num_channels=200, channel_spacing_hz=50e9, center_frequency_hz=190e12)
    plan = BandPlan.from_channels(grid, CL_BAND_SPECS)  # launch_power_w 全部为 None
    with pytest.raises(ValueError, match="launch_power_w=None"):
        plan.launch_power_profile_w()


def test_noise_figure_profile_per_element():
    grid = SpectrumGrid(num_channels=200, channel_spacing_hz=50e9, center_frequency_hz=190e12)
    plan = BandPlan.from_channels(grid, CL_BAND_SPECS)
    profile = plan.noise_figure_db_per_element()
    assert set(np.unique(profile[plan.band_element_indices(0)]).tolist()) == {5.0}
    assert set(np.unique(profile[plan.band_element_indices(1)]).tolist()) == {4.5}


# ---------------------------------------------------------------------------
# 逐连接路径
# ---------------------------------------------------------------------------
def _connection(grid, slot, power=1e-3, connection_id=0):
    return ConnectionSpec(
        grid=grid,
        connection_id=connection_id,
        first_slot=slot,
        n_slots=1,
        bit_rate_bps=300e9,
        modulation="64QAM",
        symbol_rate_hz=50e9,
        roll_off=0.0,
        launch_power_w=power,
    )


def _single_band_grid(low_nm, high_nm, power_w=1e-3):
    """横跨单一频段的栅格 + 每信道一条连接。"""
    n_slots = int(np.ceil((nm_to_hz(low_nm) - nm_to_hz(high_nm)) / 50e9)) + 1
    grid = FlexibleGrid(
        slot_width_hz=50e9,
        num_slots=n_slots,
        band_low_edge_hz=nm_to_hz(high_nm) - 25e9,
    )
    conns = []
    cid = 0
    for slot in range(n_slots):
        centre = grid.centre_hz(slot, 1)
        if nm_to_hz(high_nm) <= centre < nm_to_hz(low_nm):
            conns.append(_connection(grid, slot, power=power_w, connection_id=cid))
            cid += 1
    return grid, conns


def test_apply_launch_power_rewrites_connection_power():
    # BandPlan 要求每个声明的频段都至少有一个元素，所以这里每次只声明一段。
    l_spec = replace(_spec_by_name(CL_BAND_SPECS, "L"), launch_power_w=1.1e-3)
    c_spec = replace(_spec_by_name(CL_BAND_SPECS, "C"), launch_power_w=2.2e-3)
    _l_grid, l_conns = _single_band_grid(ITU_L_LOW_NM, ITU_L_HIGH_NM, power_w=1e-3)
    _c_grid, c_conns = _single_band_grid(ITU_C_LOW_NM, ITU_C_HIGH_NM, power_w=1e-3)
    assert l_conns and c_conns

    l_new = BandPlan.from_connections(l_conns, (l_spec,)).apply_launch_power(l_conns)
    c_new = BandPlan.from_connections(c_conns, (c_spec,)).apply_launch_power(c_conns)
    assert {c.launch_power_w for c in l_new} == {1.1e-3}
    assert {c.launch_power_w for c in c_new} == {2.2e-3}
    # 原对象未被修改（ConnectionSpec 是 frozen dataclass）
    assert {c.launch_power_w for c in l_conns} == {1e-3}
    assert {c.launch_power_w for c in c_conns} == {1e-3}


def test_apply_launch_power_rejects_band_crossing_connection():
    """横跨 L|C 边界的宽连接必须被显式拒绝，而不是按中心频率静默归段。"""
    l_spec = replace(_spec_by_name(CL_BAND_SPECS, "L"), launch_power_w=1.1e-3)
    c_spec = replace(_spec_by_name(CL_BAND_SPECS, "C"), launch_power_w=2.2e-3)
    boundary_hz = nm_to_hz(ITU_C_HIGH_NM)  # = 191.5607 THz, L|C 共享边
    # 让槽 20 的中心正好落在边界上；20 槽 (1 THz) 的连接必然横跨该边界。
    grid = FlexibleGrid(slot_width_hz=50e9, num_slots=100, band_low_edge_hz=boundary_hz - 20.5 * 50e9)
    assert grid.centre_hz(20, 1) == pytest.approx(boundary_hz, rel=1e-15)

    # plan 需要 L、C 两段都非空，故用两段各自的单槽连接建 plan。
    covering = [
        _connection(grid, 10, connection_id=0),
        _connection(grid, 30, connection_id=1),
    ]
    plan = BandPlan.from_connections(covering, (l_spec, c_spec))
    assert plan.band_names == ("L", "C")

    wide = ConnectionSpec(
        grid=grid,
        connection_id=2,
        first_slot=20,
        n_slots=20,
        bit_rate_bps=6e12,  # 6 Tbps / 6 bpsym = 1 Tbaud = 20 x 50 GHz 槽
        modulation="64QAM",
        symbol_rate_hz=1e12,
        roll_off=0.0,
        launch_power_w=1e-3,
    )
    with pytest.raises(ValueError, match="straddles"):
        plan.assert_connections_single_band([wide])
    with pytest.raises(ValueError, match="straddles"):
        plan.apply_launch_power([wide])
    # 显式放行时才不报错
    assert plan.apply_launch_power([wide], allow_crossing=True)[0].launch_power_w == 1.1e-3


def test_from_connections_rejects_empty():
    with pytest.raises(ValueError, match="at least one connection"):
        BandPlan.from_connections([], CL_BAND_SPECS)


# ---------------------------------------------------------------------------
# 回归锁：for_fiber 必须透传 raman_spectrum
# ---------------------------------------------------------------------------
def test_for_fiber_propagates_raman_spectrum():
    """FG-U1 遗留缺陷：`for_fiber` 漏传 raman_spectrum，子评估器拿到的是默认值。

    ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本条原有 ``("legacy", "tabular")`` 两个循环
    档位，并另有一条断言 ``for_fiber`` 透传 **NLI 物理档位**。
    ``"tabular"`` 档与物理档位机制都随被移除的那套物理**整体**离场（闭式 ECZ 的
    冻结核内部只走 legacy 那条合成折线，结构上没有第二条路）。**保留**本条
    ``for_fiber`` 透传命题本身——它仍是 ``for_fiber`` 的正确接线形状。
    """
    grid = SpectrumGrid(num_channels=40, channel_spacing_hz=50e9, center_frequency_hz=190e12)
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    for mode in ("legacy",):
        evaluator = MultiBandISRSGN(
            grid=grid, ref_fiber_config=fiber, symbol_rate_hz=50e9, raman_spectrum=mode,
        )
        assert evaluator.for_fiber(fiber).raman_spectrum == mode


def test_legacy_path_still_bit_identical_for_fiber():
    """修复不得影响 legacy：子评估器与父评估器在 legacy 下应当同档且逐位一致。"""
    grid = SpectrumGrid(num_channels=40, channel_spacing_hz=50e9, center_frequency_hz=190e12)
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    evaluator = MultiBandISRSGN(
        grid=grid, ref_fiber_config=fiber, symbol_rate_hz=50e9,
        # FG-15: 显式钉 ``legacy``。这条测试的命题是 "FU-2 补的 ``for_fiber``
        # 透传"——子评估器必须与父评估器**同档**。若跟着全局默认走，命题会被
        # 悄悄换成"默认档透传"，而测试名与文档都在说 legacy。
        raman_spectrum="legacy",
    )
    child = evaluator.for_fiber(fiber)
    assert child.raman_spectrum == "legacy"
    assert np.array_equal(child.f_abs_hz, evaluator.f_abs_hz)
    # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：原断言比的是 ``_g_gain``（拉曼 G 对矩阵）。
    # 那是被移除那套物理的构建期常量，闭式 ECZ 没有它；子评估器与父评估器的
    # **逐位一致**命题改由上面两条仍成立的属性承载。
