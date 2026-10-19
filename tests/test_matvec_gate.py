"""``matvec`` 档的自适应门（第 6 轮）。

门的位置来自实测（见 ``isrs_gn.MATVEC_MIN_ACTIVE_CHANNELS`` 的注释与
``2026-10-05-flexgrid-multiband/PLAN.md`` §6.26）：矩阵-向量乘的成本近似恒定，
``off`` 的成本随活跃信道数增长，交叉点 k* ≈ 34，取 1.2× 余量得 40。

这里守三件事：
1. 门确实**改变派发**（稀疏谱不走 matvec，密谱走 matvec）——只比结果不够，
   两档结果本来就等价，那样测不出门在不在。
2. 门两侧的数值仍然等价（换路不许改数）。
3. 阈值本身被钉住，防止悄悄挪动。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models import isrs_gn
from network_nlo_eval.models.isrs_gn import MATVEC_MIN_ACTIVE_CHANNELS, MultiBandISRSGN
from network_nlo_eval.models.numba_kernels import _dbm_to_w
from network_nlo_eval.network.elements import FiberSpanConfig

N = 230
SPAN_M = 100_000.0
P_DBM = -3.0


def _power(k: int) -> np.ndarray:
    p = np.zeros(N, dtype=np.float64)
    p[np.linspace(0, N - 1, k).astype(int)] = _dbm_to_w(P_DBM)
    return p


def _evaluator(accel: str = "matvec") -> MultiBandISRSGN:
    grid = SpectrumGrid(num_channels=N, channel_spacing_hz=50e9, center_frequency_hz=193.1e12)
    return MultiBandISRSGN(grid=grid, ref_fiber_config=FiberSpanConfig(length_km=100.0), ode_accel=accel)


def test_threshold_is_the_measured_one() -> None:
    """阈值是实测定出来的（k*≈34 取 1.2× 余量 ⇒ 40），不是拍的。"""
    assert MATVEC_MIN_ACTIVE_CHANNELS == 40


def test_sparse_spectrum_falls_back_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """活跃信道数 < 门 ⇒ 不许走 matvec（实测那条路在稀疏谱上慢 3.18×）。"""
    ev = _evaluator("matvec")
    calls = {"matvec": 0, "jit": 0}
    orig_matvec = MultiBandISRSGN._p1_span_matvec
    orig_jit = isrs_gn._calc_span_noise_and_power_jit

    def matvec_counted(self, *a, **kw):
        calls["matvec"] += 1
        return orig_matvec(self, *a, **kw)

    def jit_counted(*a, **kw):
        calls["jit"] += 1
        return orig_jit(*a, **kw)

    monkeypatch.setattr(MultiBandISRSGN, "_p1_span_matvec", matvec_counted)
    monkeypatch.setattr(isrs_gn, "_calc_span_noise_and_power_jit", jit_counted)

    ev._calc_span_noise_and_power(_power(MATVEC_MIN_ACTIVE_CHANNELS - 1), SPAN_M, None, None, 1)
    assert calls == {"matvec": 0, "jit": 1}, calls


def test_dense_spectrum_uses_matvec(monkeypatch: pytest.MonkeyPatch) -> None:
    ev = _evaluator("matvec")
    calls = {"matvec": 0, "jit": 0}
    orig_matvec = MultiBandISRSGN._p1_span_matvec
    orig_jit = isrs_gn._calc_span_noise_and_power_jit

    def matvec_counted(self, *a, **kw):
        calls["matvec"] += 1
        return orig_matvec(self, *a, **kw)

    def jit_counted(*a, **kw):
        calls["jit"] += 1
        return orig_jit(*a, **kw)

    monkeypatch.setattr(MultiBandISRSGN, "_p1_span_matvec", matvec_counted)
    monkeypatch.setattr(isrs_gn, "_calc_span_noise_and_power_jit", jit_counted)

    ev._calc_span_noise_and_power(_power(MATVEC_MIN_ACTIVE_CHANNELS), SPAN_M, None, None, 1)
    assert calls == {"matvec": 1, "jit": 0}, calls


@pytest.mark.parametrize("k", [4, 16, 39, 40, 68, 142, 230])
def test_gate_does_not_change_the_numbers(k: int) -> None:
    """门只影响速度：两侧都按 ``off`` 的答案对齐（预算沿用 1e-12，不放宽）。"""
    p = _power(k)
    out = {}
    for accel in ("off", "matvec"):
        o = _evaluator(accel)._calc_span_noise_and_power(p, SPAN_M, None, None, 1)
        out[accel] = np.concatenate([np.atleast_1d(x) for x in o[:3]])
    ref, got = out["off"], out["matvec"]
    denom = np.maximum(np.abs(ref), 1e-30)
    max_rel = float(np.max(np.abs(got - ref) / denom))
    assert max_rel <= 1e-12, f"k={k} max_rel={max_rel:.3e}"
