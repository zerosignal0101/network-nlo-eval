"""``scripts/dra_open_issues.py`` 的指数拟合、统计与 NLI 口径测试。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _load(name: str) -> ModuleType:
    """按脚本目录加载模块（脚本不是包）。"""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def issues() -> ModuleType:
    """被测模块。"""
    return _load("dra_open_issues")


def test_fit_loglog_exponent_recovers_slope(issues: ModuleType) -> None:
    """NLI ∝ P^2.46 的合成数据应还原 2.46。"""
    powers = np.array([0.5e-3, 1.0e-3, 2.0e-3])
    nli = 3.1e-9 * (powers / 1e-3) ** 2.46
    assert issues.fit_loglog_exponent(powers, nli) == pytest.approx(2.46, abs=1e-9)


def test_fit_loglog_exponent_drops_invalid_points(issues: ModuleType) -> None:
    """零/负/NaN 点被剔除；有效点不足 2 个时返回 None。"""
    assert issues.fit_loglog_exponent([0.0, 1e-3], [1e-9, 2e-9]) is None
    assert issues.fit_loglog_exponent([1e-3, np.nan], [1e-9, 2e-9]) is None
    assert issues.fit_loglog_exponent(
        [1e-3, 2e-3, 0.0], [1e-9, 4e-9, np.nan]
    ) == pytest.approx(2.0, abs=1e-9)


def test_summarize_uses_sample_std(issues: ModuleType) -> None:
    """样本标准差用 ddof=1，sem = std/sqrt(n)。"""
    stats = issues.summarize([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    assert stats["std"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert stats["sem"] == pytest.approx(stats["std"] / 2.0)
    assert stats["min"] == 1.0
    assert stats["max"] == 4.0


def test_summarize_ignores_non_finite(issues: ModuleType) -> None:
    """NaN/Inf 不计入；全空时统计量为 None。"""
    stats = issues.summarize([1.0, np.nan, np.inf, 3.0])
    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(2.0)
    empty = issues.summarize([])
    assert empty == {"n": 0, "mean": None, "std": None, "sem": None, "min": None, "max": None}


def test_band_nli_metrics_projects_common_gain(issues: ModuleType) -> None:
    """纯复增益不产生 NLI；叠加独立噪声后才计入。"""
    rng = np.random.default_rng(7)
    linear = (rng.normal(size=(2, 512)) + 1j * rng.normal(size=(2, 512))).astype(np.complex128)
    scaled = 0.5 * np.exp(1j * 0.3) * linear
    signal, nli, snr = issues.band_nli_metrics(scaled, linear, 100e9, 0.0, 50e9)
    assert signal > 0.0
    assert nli < 1e-12 * signal
    assert np.isinf(snr) or snr > 100.0
    noisy = scaled + 1e-3 * (
        rng.normal(size=linear.shape) + 1j * rng.normal(size=linear.shape)
    ).astype(np.complex128)
    _, nli_noisy, snr_noisy = issues.band_nli_metrics(noisy, linear, 100e9, 0.0, 50e9)
    assert nli_noisy > 0.0
    assert np.isfinite(snr_noisy)


def test_doubled_pump_drive_moves_hybrid_away(issues: ModuleType) -> None:
    """2× 耗尽驱动让 hybrid 更偏离 3ch 前向波形（负结果回归）。"""
    from dra_waveform_compare import _pumps

    pumps = _pumps("fwd_0p20w")
    nominal, _ = issues._paper_span_center(3, pumps, drive_factor=1.0, launch_power_w=1e-3)
    doubled, _ = issues._paper_span_center(3, pumps, drive_factor=2.0, launch_power_w=1e-3)
    assert np.isfinite(nominal) and np.isfinite(doubled)
    assert doubled > nominal
    assert doubled - nominal == pytest.approx(0.47, abs=0.05)
