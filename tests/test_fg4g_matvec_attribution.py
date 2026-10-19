r"""FG-4g 归因门：把「0.314 dB 与 ``ode_accel`` 无关」钉成可执行断言。

FG-4f 报告 RL 的 ``scan.py`` 快速通道在 ``matvec`` 档下与 ``verify_allocation``
有最差 0.314 dB 的分歧，而 FG-4a 金标准说同一档位差 <= 1e-15。本文件把本刀的
裁决结果固化成测试：

* **物理层**：库自己的三个公开入口在 off/matvec 之间必须 <= 1e-12 相对偏差
  （FG-4a 只覆盖了自造工况；这里补上多跨段数、多跨段长度与非空占用的组合）。
* **结构层**：``qot_checker.py`` 里**不允许**出现 ``ode_accel``。SNR 闸门对加速
  档位无知，是「0.314 dB 不可能来自档位」的可执行表述。
* **端到端层**：若 RL 仓可导入，则在同一棵树下分别强制 ``off`` 与 ``matvec``
  跑同一组差分用例，断言两档的失配集合与最差差值**逐位相同**。

不改动任何既有 ``tests/*`` 的期望值；本文件只新增断言。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NUM_CHANNELS = 230
CHANNEL_SPACING_HZ = 50e9
CENTER_FREQUENCY_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9

#: FG-4a 的 1e-12 预算，本刀沿用同一口径（不放松）。
MAX_REL = 1e-12

_RL_SRC = Path("/home/zsig/Documents/gits/wt/fg-batch-rl/src")


def _dev(expected: np.ndarray, actual: np.ndarray) -> tuple[int, float, float]:
    """(不一致数, 最大绝对差, 最大相对差)，有限性不匹配单独算硬失败。"""
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    assert expected.shape == actual.shape
    n_diff = int(np.count_nonzero(expected != actual))
    both = np.isfinite(expected) & np.isfinite(actual)
    if not both.any():
        return n_diff, 0.0, 0.0
    e, a = expected[both], actual[both]
    d = np.abs(e - a)
    rel = np.divide(d, np.abs(e), out=np.zeros_like(d), where=np.abs(e) > 0)
    return n_diff, float(d.max()), float(rel.max())


def _evaluators(ode_accel: str, **kw):
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    return MultiBandISRSGN(
        grid=grid, ref_fiber_config=fiber, symbol_rate_hz=SYMBOL_RATE_HZ,
        ode_accel=ode_accel,
        # FG-15: pin the tier to ``legacy``. This test compares ``off`` against
        # ``matvec`` **on the same spectrum** — the whole proposition is the
        # float-reordering budget between two ODE paths. ``off`` is structurally
        # legacy-only (it rebuilds the legacy polyline in-kernel), so letting it
        # inherit the new ``tabular`` default would raise, and pinning only one
        # side would compare two different physics instead of two ODE paths.
        raman_spectrum="legacy",
        **kw,
        nli_model="p2_gn",
    )


# ======================================================================
# 物理层：库自己的入口
# ======================================================================
@pytest.mark.parametrize("span_length_m", [1e3, 40e3, 80e3, 100e3, 120e3])
@pytest.mark.parametrize("n_effective_spans", [1, 2, 7, 10])
@pytest.mark.parametrize("stride", [1, 7], ids=["full", "every7th_empty"])
def test_fg4g_span_level_off_vs_matvec_within_budget(
    span_length_m: float, n_effective_spans: int, stride: int
) -> None:
    """逐跨段入口：非空占用 + 多跨段数 + 多跨段长度下 off/matvec 必须 <= 1e-12。

    这层 FG-4a 金标准已经覆盖过；本刀补的是**参数网格更宽**（它当时只有
    100 km/1 km 两档、n_span 只有 1 与 10）。
    """
    from network_nlo_eval.network.elements import EDFAConfig

    power = np.full(NUM_CHANNELS, 1e-3, dtype=np.float64)
    if stride == 7:
        power[::7] = 0.0
    edfa = EDFAConfig(target_gain_db=None, noise_figure_db=5.0)

    off = _evaluators("off")._calc_span_noise_and_power(power, span_length_m, edfa, None, n_effective_spans)
    mv = _evaluators("matvec")._calc_span_noise_and_power(power, span_length_m, edfa, None, n_effective_spans)

    worst_rel = 0.0
    worst_abs = 0.0
    for name, e, a in zip(("p_out_w", "sigma2_spm_w", "sigma2_xpm_w", "sigma2_ase_w"), off, mv):
        n_nonfinite = int(np.count_nonzero(np.isfinite(e) != np.isfinite(a)))
        assert n_nonfinite == 0, f"{name}: {n_nonfinite} 元素在有限/无穷之间变了"
        _n, d, r = _dev(e, a)
        worst_abs = max(worst_abs, d)
        worst_rel = max(worst_rel, r)
    assert worst_rel <= MAX_REL, (
        f"L={span_length_m}m n_span={n_effective_spans} stride={stride}: "
        f"相对偏差 {worst_rel:.3e} 超过 {MAX_REL:.0e}（最大绝对差 {worst_abs:.3e}）"
    )


@pytest.mark.parametrize("equalize", [True, False])
def test_fg4g_evaluate_path_with_roadm_within_budget(equalize: bool) -> None:
    """``evaluate_path`` 在含 ROADM 的多跨段路径上 off/matvec 必须 <= 1e-12。

    0.314 dB 那条路径的 SNR 由 ROADM 装配决定，所以这一层最贴近分歧现场。

    两个 ``equalize_output_power`` 取值都跑：FG-4e 树的 ``evaluate_path`` 不消费
    该字段（统一按合并损耗处理），这正是它与 ``ecz-dev`` 的 ``_build_path_snr_array``
    分歧的一半。无论取哪个值，**档位之间**都必须逐位一致。
    """
    from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

    power = np.full(NUM_CHANNELS, 1e-3, dtype=np.float64)
    power[::7] = 0.0
    roadm = ROADMConfig(
        insertion_loss_db=5.0, filtering_penalty_db=0.2, blocked_slots=[40, 41],
        equalize_output_power=equalize,
    )
    optical_path = MultiSpanOpticalPath(
        fiber_configs=[
            FiberSpanConfig(length_km=160.0, max_span_length_km=80.0),
            FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        ],
        edfa_configs=[
            EDFAConfig(target_gain_db=None, noise_figure_db=5.0),
            EDFAConfig(target_gain_db=None, noise_figure_db=6.0),
        ],
        roadm_configs=[roadm, None],
        link_keys=[(0, 1), (1, 2)],
        node_path=[0, 1, 2],
    )
    off = _evaluators("off").evaluate_path(optical_path, power)
    mv = _evaluators("matvec").evaluate_path(optical_path, power)
    for key in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        _n, _d, rel = _dev(getattr(off, key), getattr(mv, key))
        assert rel <= MAX_REL, (
            f"evaluate_path.{key} (equalize={equalize}): 相对偏差 {rel:.3e} 超过 {MAX_REL:.0e}"
        )


# ======================================================================
# 结构层：SNR 闸门对档位无知
# ======================================================================
def test_fg4g_qot_gate_is_blind_to_ode_accel() -> None:
    """``rwa/qot_checker.py`` 里不允许出现 ``ode_accel``。

    SNR 由这一层装配出来。若它对加速档位无知，则「换档导致 0.314 dB」在结构上
    不可能发生——这正是本刀实测到的结论的可执行表述。
    """
    src = (REPO_ROOT / "src/network_nlo_eval/rwa/qot_checker.py").read_text(encoding="utf-8")
    assert "ode_accel" not in src, (
        "QoT SNR 闸门开始引用 ode_accel 了：档位已经渗入 SNR 装配，"
        "FG-4a 的 <=1e-12 契约必须重新取证"
    )


# ======================================================================
# 端到端层：同一棵树下换档，差分结果必须逐位不变
# ======================================================================
def _rl_available() -> bool:
    return (_RL_SRC / "multiband_optical_rl").is_dir() and importlib.util.find_spec("network_nlo_eval") is not None


@pytest.mark.skipif(not _rl_available(), reason="RL 仓不在 PYTHONPATH 上，无法复现差分")
@pytest.mark.parametrize("stride", [7])
def test_fg4g_scan_differential_is_invariant_to_tier(stride: int) -> None:
    """把 FG-4f 的 0.314 dB 分歧按档位拆开：换档必须不改变差分结果。

    这条测试**不**断言差分为零——那取决于两棵树的 QoT 语义是否一致，是 RL 侧
    的契约问题。它断言的是本刀的裁决：**失配与档位无关**。
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile

    script = REPO_ROOT / "scripts" / "fg4g_q2_trace_diff.py"
    assert script.exists(), f"{script} 缺失，无法复现 Q2 差分"
    common = ["--steps", "5", "--strides", str(stride), "--max-paths-per-hops", "1",
              "--hops", "1,3,7", "--max-trace-cases", "1"]
    out: dict[str, dict] = {}
    with tempfile.TemporaryDirectory() as td:
        for tier in ("off", "matvec"):
            path = Path(td) / f"q2_{tier}.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = f"{_RL_SRC}:{REPO_ROOT / 'src'}"
            env["NUMBA_CACHE_DIR"] = f"/tmp/nb-fg4g-test-{tier}"
            proc = subprocess.run(
                [sys.executable, str(script), "--out", str(path), *common, "--force-accel", tier],
                cwd=str(Path("/home/zsig/Documents/gits/wt/fg-batch-rl")),
                env=env, capture_output=True, text=True,
            )
            assert proc.returncode == 0, f"tier={tier} 探针失败:\n{proc.stderr[-2000:]}"
            out[tier] = json.loads(path.read_text(encoding="utf-8"))

    sig = {
        tier: (
            [(c["hops"], c["band"], c["channel"], c.get("snr_A_db"), c.get("snr_B_db"))
             for c in rep["cases"]],
            rep["summary"]["worst_snr_abs_diff_db"],
        )
        for tier, rep in out.items()
    }
    assert sig["off"] == sig["matvec"], (
        "同一棵树下换 ode_accel 档位后差分结果变了："
        f"off={sig['off'][1]} vs matvec={sig['matvec'][1]}；"
        "0.314 dB 就真的与档位有关，本刀的 (B) 结论需要推翻"
    )
