"""WP-Q1 验收守卫（brief §4 判据 Q-J1..Q-J6）。

跑测试::

    PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python -m pytest -q

证据文件（落盘读数）::

    results/bwqot_legacy_regression.json   Q-J1  legacy 逐位不变（硬闸门）
    results/bwqot_probe.json               Q-J2  带宽真的进了公式 + Q-J6 两种功率口径
    results/bwqot_cost.json                Q-J3  不引入 ODE

本文件刻意**不**重新实现 SNR 计算——那会造出第四套口径。物理正确性由
``tools/ode_formula_map.py --check``（Q-J5）把关；判据读数一律读 JSON 断言，
真正的物理交叉验证由 ``tools/bwqot_probe.py`` 里那份**照 P1 原文独立重写**的
标量参考实现完成。
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
PY = "/home/zsig/miniconda3/envs/rl/bin/python"


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.is_file():
        pytest.skip(f"{name} 尚未落盘；先跑 tools/bwqot_*.py --write")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_check(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, str(REPO / "tools" / script), "--check"],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin:/usr/local/bin"},
        timeout=3600,
    )


# ---------------------------------------------------------------------------
# Q-J1（硬闸门）：legacy 逐位不变
# ---------------------------------------------------------------------------
def test_qj1_legacy_bit_exact_against_ecz_dev():
    """Q-J1：6 个 legacy 场景 delta_db 必须**恰为 0.0**，容差不许放宽。"""
    doc = _load("bwqot_legacy_regression.json")
    assert doc["judgment"] == "Q-J1"
    assert doc["tolerance"] == 0.0, "Q-J1 容差必须是 0.0，不许放宽"

    scenarios = doc["scenarios"]
    assert len(scenarios) == 6, f"场景数 {len(scenarios)} != 6"
    for row in scenarios:
        if row["both_rejected"]:
            continue
        assert row["delta_db"] == 0.0, f"{row['scenario']} delta_db={row['delta_db']!r} != 0.0"
    assert doc["summary"]["max_abs_delta_db"] == 0.0
    assert doc["verdict"] == "PASS", doc["failures"]


def test_qj1_bandwidth_degenerates_to_scalar_bitwise():
    """Q-J1 的物理内容：均匀 50 GHz 下 B_i 必须**按位**退化为那个全局标量。"""
    doc = _load("bwqot_legacy_regression.json")
    deg = doc["b_degenerates_to_scalar"]
    assert deg["mode"].startswith("legacy")
    assert deg["equals_scalar_bitwise"] is True, (
        "legacy 下 B_i 未逐位退化为标量 —— 这正是 legacy 数字不动的根据"
    )
    assert deg["symbol_rate_hz"] == 50e9
    assert all(b == deg["symbol_rate_hz"] for b in deg["b_hz_channels"])


# ---------------------------------------------------------------------------
# Q-J2：带宽真的进了公式
# ---------------------------------------------------------------------------
def test_qj2_probe_covers_sweep_and_both_power_modes():
    """Q-J2/Q-J6：4 个带宽点 × 2 种功率口径，读数齐全。"""
    doc = _load("bwqot_probe.json")
    assert doc["judgment"] == "Q-J2+Q-J6"
    assert doc["rel_tol"] == 1e-9

    modes = doc["power_width_modes"]
    assert modes == ["fixed_power", "fixed_psd"], "Q-J6 要求两种归一化口径各一组"

    b_values = {round(r["b_i_hz"] / 1e9, 6) for r in doc["rows"]}
    assert b_values == {12.5, 25.0, 50.0, 100.0}, f"扫描点不对：{sorted(b_values)}"
    for mode in modes:
        got = {round(r["b_i_hz"] / 1e9, 6) for r in doc["rows"] if r["power_width_mode"] == mode}
        assert got == {12.5, 25.0, 50.0, 100.0}, f"{mode} 口径下的扫描点不对：{sorted(got)}"


def test_qj2_each_term_matches_independent_p1_reference():
    """Q-J2：SPM/XPM/ASE/SNR 四项都要与 P1 原文的独立参考实现一致。"""
    doc = _load("bwqot_probe.json")
    assert doc["max_rel_err"] <= doc["rel_tol"], doc["max_rel_err"]
    assert doc["max_scaling_rel_err"] <= doc["rel_tol"], doc["max_scaling_rel_err"]
    for row in doc["rows"]:
        for term, err in row["rel_err"].items():
            assert err <= doc["rel_tol"], f"{term} @ B={row['b_i_hz']}: rel_err={err:.3e}"
    assert doc["verdict"] == "PASS", doc["failures"]


def test_qj2_snr_actually_moves_with_bandwidth():
    """Q-J2 的要害：SNR 必须随 B 真变，否则 B 根本没进 QoT。"""
    doc = _load("bwqot_probe.json")
    for mode, rec in doc["snr_vs_bandwidth"].items():
        assert rec["strictly_varies"] is True, f"{mode} 口径下 SNR 不随 B 变化"
        assert rec["spread_db"] > 1e-6
        assert len(rec["snr_db_by_b"]) == 4

    # 三项的 B 依赖必须彼此不同，否则"逐项对照"无从谈起：
    # ASE 随 B 增、SPM 随 B 减、XPM 随 B 增但渐近饱和。
    rows = [r for r in doc["rows"] if r["power_width_mode"] == "fixed_power"]
    rows.sort(key=lambda r: r["b_i_hz"])
    ase = [r["measured"]["ase_noise_w"] for r in rows]
    spm = [r["measured"]["spm_noise_w"] for r in rows]
    xpm = [r["measured"]["xpm_noise_w"] for r in rows]
    assert ase == sorted(ase), "P1 (34)：ASE 正比于 B_i，应随带宽单调增"
    assert spm == sorted(spm, reverse=True), "P1 (29)：B^-2 前因子，SPM 应随带宽单调减"
    assert xpm == sorted(xpm), "P1 (30)：atan(φ B_i/α) 单调增，XPM 应随带宽单调增"
    assert spm[-1] < spm[0], "SPM 必须真的随带宽下降，否则 B 没进 (29)"
    assert ase[-1] > ase[0], "ASE 必须真的随带宽上升，否则 B 没进 (34)"


def test_qj2_ase_linearity_follows_eq34():
    """P1 (34)：ASE ∝ f_i·B_i，比值必须与公式一致。"""
    doc = _load("bwqot_probe.json")
    entries = doc["ase_linearity_eq34"]
    assert entries
    for e in entries:
        assert e["rel_err"] <= doc["rel_tol"], e
    # 随 B 增大，(f_i·B_i) 必须严格增大
    fixed = [e for e in entries if e["power_width_mode"] == "fixed_power"]
    ratios = [e["measured_ase_ratio"] for e in fixed]
    assert ratios == sorted(ratios)


# ---------------------------------------------------------------------------
# Q-J3：不引入 ODE
# ---------------------------------------------------------------------------
def test_qj3_median_latency_under_budget_and_no_ode_symbols():
    """Q-J3：单次 QoT 验证中位耗时 < 10 ms，且内核无 ODE 符号。"""
    doc = _load("bwqot_cost.json")
    assert doc["judgment"] == "Q-J3"
    assert doc["budget_ms"] == 10.0, "Q-J3 预算写死为 10 ms，不许放宽"
    assert doc["median_ms"] < doc["budget_ms"], f"中位耗时 {doc['median_ms']} ms 超预算"
    assert doc["latency_budget_ok"] is True
    assert doc["no_ode_ok"] is True, doc["no_ode_scan"]["hits"]
    assert doc["no_ode_scan"]["hits"] == []
    # 带宽逐连接化之后也不能掉出预算
    bw = doc["bandwidth_resolved_timing"]
    assert bw["heterogeneous"] is True
    assert bw["median_ms"] < doc["budget_ms"]
    assert doc["verdict"] == "PASS", doc["failures"]


def test_qj3_forbidden_symbols_not_silently_narrowed():
    """防"通过判据"式偷懒：禁用符号表必须仍是前言点名的那四项。"""
    doc = _load("bwqot_cost.json")
    assert set(doc["forbidden_symbols"]) == {"ode", "rk4", "ode_accel", "phi_kurtosis"}
    assert doc["no_ode_scan"]["n_identifiers_scanned"] > 0


# ---------------------------------------------------------------------------
# Q-J5：公式号可 grep
# ---------------------------------------------------------------------------
def test_qj5_per_channel_bandwidth_mapped_to_p1_23_29_30_34():
    """Q-J5：本 WP 改动的每一项都能映射到 P1 公式号（落进 formula map）。"""
    doc = _load("ode_dev_formula_map.json")
    by_id = {e["id"]: e for e in doc["physics"]}

    for eid in ("p1_eq23_nli_total", "p1_eq29_spm", "p1_eq30_xpm", "p1_eq34_ase"):
        assert eid in by_id, f"formula map 缺 {eid}"
        assert by_id[eid]["paper"]["id"] == "P1"

    # 逐信道锚点必须真的在源码里（这是 J1 守卫的可执行形式）
    for eid in ("p1_eq23_nli_total", "p1_eq29_spm", "p1_eq30_xpm", "p1_eq34_ase"):
        for anchor in by_id[eid]["code"]:
            text = (REPO / anchor["file"]).read_text(encoding="utf-8")
            assert anchor["contains"] in text, f"{eid} 锚点失效：{anchor['contains']!r}"

    # (30) 必须分别锚定 B_l 与 B_i（这正是本 WP 与标量版的分界）
    xpm_src = " ".join(a["contains"] for a in by_id["p1_eq30_xpm"]["code"])
    assert "b_hz_channels[l_idx]" in xpm_src, "XPM 未锚定 B_l = 干扰信道自己的带宽"
    assert "b_hz_channels[i]" in xpm_src, "XPM 未锚定 B_i = COI 的带宽"
    # SPM 走 B_i，ASE 走逐信道 B
    assert "b_hz_channels[i]" in " ".join(a["contains"] for a in by_id["p1_eq29_spm"]["code"])
    ase_src = (REPO / by_id["p1_eq34_ase"]["code"][2]["file"]).read_text(encoding="utf-8")
    assert "channel_bandwidth_hz_channels" in ase_src, "ASE 未改读逐信道 B_i"


def test_qj5_formula_map_check_reproduces():
    """Q-J5：公式映射 --check 必须可独立复算通过。"""
    proc = _run_check("ode_formula_map.py")
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Q-J6：归一化口径显式
# ---------------------------------------------------------------------------
def test_qj6_normalisation_is_an_explicit_parameter_and_documented():
    """Q-J6：功率-宽度归一化必须是显式参数 + 文档，且两口径读数都在案。"""
    from network_nlo_eval.models.isrs_gn import (
        CHANNEL_BANDWIDTH_SOURCES,
        POWER_WIDTH_DEFAULT,
        POWER_WIDTH_MODES,
    )

    assert POWER_WIDTH_MODES == ("fixed_power", "fixed_psd")
    assert POWER_WIDTH_DEFAULT == "fixed_power"
    assert CHANNEL_BANDWIDTH_SOURCES == ("symbol_rate", "occupied")

    # 源码里必须有显式声明与说明，不是隐含行为
    src = (REPO / "src/network_nlo_eval/models/isrs_gn.py").read_text(encoding="utf-8")
    assert "power_width_mode: str = POWER_WIDTH_DEFAULT" in src
    assert "POWER_WIDTH_MODES" in src and "CHANNEL_BANDWIDTH_SOURCES" in src

    # 两种口径的读数都必须落盘
    doc = _load("bwqot_probe.json")
    assert set(doc["power_width_modes"]) == {"fixed_power", "fixed_psd"}
    assert set(doc["snr_vs_bandwidth"]) == {"fixed_power", "fixed_psd"}
    # 两口径必须给出**不同**的读数，否则"显式"只是摆设
    assert (
        doc["snr_vs_bandwidth"]["fixed_power"]["snr_db_by_b"]
        != doc["snr_vs_bandwidth"]["fixed_psd"]["snr_db_by_b"]
    )


def test_qj6_invalid_normalisation_mode_is_rejected_not_silently_defaulted():
    """非法口径必须显式报错，不许静默取默认。"""
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    grid = SpectrumGrid(4, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=80.0)
    with pytest.raises(ValueError, match="power_width_mode"):
        MultiBandISRSGN(grid, fiber, 50e9, power_width_mode="constant_psd")
    with pytest.raises(ValueError, match="channel_bandwidth_source"):
        MultiBandISRSGN(grid, fiber, 50e9, channel_bandwidth_source="occupied_bw")


# ---------------------------------------------------------------------------
# 能力测试（不读证据，直接验行为）
# ---------------------------------------------------------------------------
def _flex_conns(slots_per_conn):
    """按给定槽数造一组**依次排布**的连接，网格开到足够宽。

    注意：``first_slot`` 必须留出前一条连接的宽度，所以不能用 ``i * 4``；
    且网格槽数要覆盖所有连接占用带之和，否则 ConnectionSpec 的不变式 1 会抛错。
    """
    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid

    total_slots = sum(slots_per_conn)
    # from_legacy_grid 的槽数 = num_channels * slots_per_channel
    n_channels = -(-total_slots // 4)  # ceil，保证够用
    grid = FlexibleGrid.from_legacy_grid(n_channels, 50e9, 193.1e12, slots_per_channel=4)

    conns, cursor = [], 0
    for i, n in enumerate(slots_per_conn):
        conns.append(ConnectionSpec(
            grid=grid, connection_id=i, first_slot=cursor, n_slots=n,
            bit_rate_bps=2.0 * n * 12.5e9, modulation="QPSK",
            symbol_rate_hz=float(n * 12.5e9), roll_off=0.0, launch_power_w=1e-3,
        ))
        cursor += n
    return grid, conns


def test_per_connection_heterogeneous_bandwidth_is_supported():
    """逐连接异构带宽必须被接受，且 B_i 真的逐连接不同（WP-Q1 §3.3）。"""
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    _, conns = _flex_conns((1, 2, 3, 4))
    ev = MultiBandISRSGN(
        conns[0].grid, FiberSpanConfig(length_km=80.0),
        50e9, connections=conns, channel_bandwidth_source="occupied",
    )
    b = ev.channel_bandwidth_hz_channels
    assert list(b) == [12.5e9, 25e9, 37.5e9, 50e9]
    assert np.ptp(b) > 0


def test_occupied_vs_symbol_rate_bandwidth_source_differ():
    """占用带口径与符号率口径在 n_slots != Rs 时必须给出不同的 B。"""
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    # n_slots=4 => 占用带 50 GHz，但把声明 Rs 压到 25 GHz（不被 slots_for_rate 管）
    from dataclasses import replace

    grid, conns = _flex_conns((4,))
    conns = [replace(conns[0], symbol_rate_hz=25e9)]
    fiber = FiberSpanConfig(length_km=80.0)

    occ = MultiBandISRSGN(grid, fiber, 50e9, connections=conns, channel_bandwidth_source="occupied")
    sym = MultiBandISRSGN(grid, fiber, 50e9, connections=conns, channel_bandwidth_source="symbol_rate")
    assert float(occ.channel_bandwidth_hz_channels[0]) == 50e9
    assert float(sym.channel_bandwidth_hz_channels[0]) == 25e9


def test_invalid_bandwidth_is_rejected():
    """B_i 合法域校验：ConnectionSpec 已挡非正 Rs，这里再挡非法 source/构造。"""
    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    # ConnectionSpec.__post_init__ 必须挡住非正 symbol_rate_hz（B_i 的来源之一）
    grid = FlexibleGrid.from_legacy_grid(1, 50e9, 193.1e12, slots_per_channel=4)
    with pytest.raises(ValueError, match="symbol_rate_hz must be positive"):
        ConnectionSpec(
            grid=grid, connection_id=0, first_slot=0, n_slots=4,
            bit_rate_bps=100e9, modulation="QPSK", symbol_rate_hz=0.0,
            roll_off=0.0, launch_power_w=1e-3,
        )
    # 槽数非法 ⇒ 占用带为 0，无法成为合法 B_i
    with pytest.raises(ValueError):
        ConnectionSpec(
            grid=grid, connection_id=0, first_slot=0, n_slots=0,
            bit_rate_bps=100e9, modulation="QPSK", symbol_rate_hz=50e9,
            roll_off=0.0, launch_power_w=1e-3,
        )


def test_fixed_psd_is_bitwise_noop_when_bandwidth_is_uniform():
    """B_i 均匀时 fixed_psd 的缩放因子必须**恰好**是 1.0（legacy 安全性）。"""
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    grid = SpectrumGrid(4, 50e9, 193.1e12)
    fp = MultiBandISRSGN(grid, FiberSpanConfig(length_km=80.0), 50e9, power_width_mode="fixed_psd")
    scale = fp._power_width_scale()
    assert scale.dtype == np.float64
    # 按位比较 float64 的位模式：因子必须**恰好**是 IEEE 的 1.0，不能是 0.9999999…
    ones = np.ones(4, dtype=np.float64)
    assert np.array_equal(scale.view(np.int64), ones.view(np.int64))
    assert np.all(scale == 1.0)


def test_uniform_bandwidth_is_bitwise_identical_to_scalar_legacy_path():
    """均匀 B_i 下逐连接模式必须与"标量退化"路径逐位相同。"""
    from network_nlo_eval.core.flexgrid import FlexibleGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    # 4 条连接，各 4 槽 = 50 GHz，全部同宽 ⇒ B_i 均匀
    grid, conns = _flex_conns((4, 4, 4, 4))
    fiber = FiberSpanConfig(length_km=80.0)
    ev = MultiBandISRSGN(grid, fiber, 50e9, connections=conns, channel_bandwidth_source="occupied")
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)])
    power = np.full(4, 1e-3)

    res = ev.evaluate_path(path, power)
    b = ev.channel_bandwidth_hz_channels
    # B_i 均匀且等于名义 Rs ⇒ 与标量路径等价；用另一条等价构造交叉验证：
    # symbol_rate 口径下同样的均匀连接也必须给出逐位相同的结果。
    ev_sym = MultiBandISRSGN(
        grid, fiber, 50e9, connections=conns, channel_bandwidth_source="symbol_rate"
    )
    res_sym = ev_sym.evaluate_path(path, power)
    for name in ("spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        a = getattr(res, name)
        c = getattr(res_sym, name)
        assert np.array_equal(a.view(np.int64), c.view(np.int64)), f"{name} 不逐位相同"
    assert np.all(b == 50e9)


def test_wide_connection_is_not_physics_identical_to_narrow_one():
    """本 WP 的存在意义：8 槽连接与 1 槽连接的 SNR 必须有物理差别。"""
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    fiber = FiberSpanConfig(length_km=80.0)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)])
    snrs = {}
    for slots in (1, 8):
        grid, conns = _flex_conns((slots,))
        ev = MultiBandISRSGN(grid, fiber, 50e9, connections=conns,
                             channel_bandwidth_source="occupied")
        res = ev.evaluate_path(path, np.array([1e-3]))
        snrs[slots] = float(res.snr_db[0])
    assert abs(snrs[1] - snrs[8]) > 1e-3, (
        f"1 槽与 8 槽 SNR 几乎相同（{snrs}）—— B 仍可能没进 QoT"
    )
    assert snrs[1] > snrs[8], "固定总功率下，宽连接 PSD 更低 ⇒ SNR 应更低"


def test_power_width_modes_give_different_results_when_bandwidth_varies():
    """两种功率口径在异构带宽下必须给出不同结果（否则口径是摆设）。"""
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    fiber = FiberSpanConfig(length_km=80.0)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)])
    grid, conns = _flex_conns((1, 8))
    out = {}
    for mode in ("fixed_power", "fixed_psd"):
        ev = MultiBandISRSGN(grid, fiber, 50e9, connections=conns,
                             channel_bandwidth_source="occupied", power_width_mode=mode)
        out[mode] = ev.evaluate_path(path, np.full(2, 1e-3)).snr_db.copy()
    assert not np.array_equal(out["fixed_power"], out["fixed_psd"])
    # fixed_psd 下 PSD 恒定 ⇒ 两个宽度不同的连接 SNR 应比 fixed_power 更接近
    spread_fp = abs(out["fixed_power"][0] - out["fixed_power"][1])
    spread_psd = abs(out["fixed_psd"][0] - out["fixed_psd"][1])
    assert spread_psd < spread_fp


# ---------------------------------------------------------------------------
# tools 可独立复算
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "script", ["bwqot_probe.py", "bwqot_legacy_regression.py", "bwqot_cost.py"]
)
def test_q_evidence_tools_check_reproduces(script: str):
    """证据工具的 --check 必须能独立复算出同样的 PASS。"""
    proc = _run_check(script)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    assert "verdict=PASS" in proc.stdout