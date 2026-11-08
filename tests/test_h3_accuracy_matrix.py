"""任务 Q：H3 广域精度矩阵的稳定锚点回归。

完整 661 点矩阵在 ``results/h3_accuracy_matrix``；这里只钉住三个亚秒级、
跨平台稳定的代表点，容差取实测残差的约 4 倍，既能抓住物理回归，又不会被
BLAS/平台的末位差异误伤。

参考选择与矩阵一致：无泵用 ``numerical``（同跨间相干口径的 ISRS-GN 基准），
不采用 ``numerical_dra``——后者在无泵 L/S 边缘的 Raman ASE 会高约 19 dB。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget


def _case(bands: str, spacing_hz: float, spans: int = 1) -> BenchCase:
    """构造与矩阵同源的算例（64 GBaud, 100 km, NF 5.5 dB）。"""
    low, high = band_edges_hz(bands)
    return BenchCase(
        name=f"regress_{bands}_{spacing_hz / 1e9:g}",
        bands=bands,
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=spans,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


def _h3_and_reference(case: BenchCase, power_dbm: float):
    """返回 H3 结果、预算与 ``numerical`` 参考 SNR。"""
    path = build_path(case)
    power = launch_profile(case, power_dbm)
    model = make_model("pcfm_h3_raman", case)
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)
    reference = make_model("numerical", case).evaluate_path(path, power)
    return result, budget, reference


def test_h3_matches_the_numerical_reference_on_a_narrow_c_anchor() -> None:
    """C 波段 75 GHz、1 跨、0 dBm：H3 与数值参考的逐信道偏差 < 0.15 dB。

    实测 max |Δ| = 0.030 dB、mean Δ = +0.012 dB；容差取 5 倍。
    """
    case = _case("C", 75e9)
    result, budget, reference = _h3_and_reference(case, 0.0)
    delta = result.snr_db - reference.snr_db

    assert float(np.max(np.abs(delta))) < 0.15
    assert abs(float(np.mean(delta))) < 0.05
    assert budget.validity.band_class == "narrow"
    assert float(np.median(budget.margin_db)) == pytest.approx(1.414, abs=1e-3)


def test_h3_budget_covers_the_narrow_anchor_error() -> None:
    """同一窄带锚点：1.41 dB 的 2σ margin 必须覆盖全部实测乐观偏差。"""
    case = _case("C", 75e9)
    result, budget, reference = _h3_and_reference(case, 0.0)
    delta = result.snr_db - reference.snr_db

    assert bool(np.all(np.isfinite(budget.margin_db)))
    assert bool(np.all(delta <= budget.margin_db + 1e-12))
    assert float(budget.margin_db.min()) > 1.0


def test_wideband_budget_is_dominated_by_the_statistical_row() -> None:
    """C+L 100 GHz、1 跨、0 dBm：宽带 margin 由 5.08 dB 统计项主导，约 10.36 dB。

    实测 max |Δ| = 0.041 dB，说明这一档 margin 不是模型误差，而是参考波形的
    统计不确定度；回归同时锁住"不会误缩到窄带量级"。
    """
    case = _case("C+L", 100e9)
    result, budget, reference = _h3_and_reference(case, 0.0)
    delta = result.snr_db - reference.snr_db

    assert budget.validity.band_class == "wide"
    assert float(np.median(budget.margin_db)) == pytest.approx(10.359, abs=1e-2)
    assert float(np.max(np.abs(delta))) < 0.15


def test_dense_grid_error_exceeds_the_narrow_margin() -> None:
    """已知边界：64 GBaud 塞进 50 GHz 时 GN 非重叠假设失效，margin 小 0.1 dB。

    这是矩阵里唯一稳定的预算不足点（6 dBm、1 跨、C 50 GHz）：实测最大乐观偏差
    1.46 dB > 1.41 dB。测试把缺口钉成 1.0–2.5 dB 的区间，避免以后误以为该档
    已被覆盖；若未来加入频谱重叠 validity flag，应先更新本测试与残差表。
    """
    case = _case("C", 50e9)
    result, budget, reference = _h3_and_reference(case, 6.0)
    delta = result.snr_db - reference.snr_db

    assert 1.0 < float(np.max(delta)) < 2.5
    assert float(np.median(budget.margin_db)) == pytest.approx(1.414, abs=1e-3)
    assert float(np.mean(delta <= budget.margin_db)) < 0.5
