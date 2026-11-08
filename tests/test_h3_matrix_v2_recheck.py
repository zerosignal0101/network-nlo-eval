"""第七轮任务 W：S/U/T 合入后的 v2 定向复核锚点。

完整复跑与报告见 ``scripts/h3_matrix_v2_recheck.py`` 和
``results/h3_matrix_v2/recheck.json``。这里只钉住三件容易回归的事：

* **U**：密集栅格（64 GBaud / 50 GHz）被 ``dense_grid_overlap`` 标出，
  默认 margin 不足，而 opt-in 的 1.1 dB 惩罚刚好覆盖；
* **S**：E+S+C+L 六泵后向在 ``raman_steps="auto"`` 下升到 96 步，
  点值有限且与 192 步的逐信道差 < 0.15 dB；
* **T**：无泵 S+C+L 边缘用 ``numerical`` 主参考时 H3 误差 < 0.5 dB，
  不再出现第五轮把 ``numerical_dra`` 参考缺陷（约 17.5 dB）记成 H3 误差的假象。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.uncertainty import H3UncertaintyConfig, estimate_h3_error_budget
from network_nlo_eval.network.elements import RamanPumpConfig

#: 与第五轮矩阵、``tests/test_pcfm_h3_raman.py::SIX_PUMPS`` 相同的六泵配置。
SIX_PUMPS = (
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
)


def _case(bands: str, spacing_hz: float, spans: int = 1) -> BenchCase:
    """构造与第五轮矩阵同源的算例（64 GBaud, 100 km, NF 5.5 dB）。"""
    low, high = band_edges_hz(bands)
    return BenchCase(
        name=f"v2_{bands}_{spacing_hz / 1e9:g}",
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


def _h3_result(case: BenchCase, power_dbm: float, **options):
    """返回 H3 结果、模型与预算。"""
    path = build_path(case)
    power = launch_profile(case, power_dbm)
    model = make_model("pcfm_h3_raman", case, **options)
    result = model.evaluate_path(path, power)
    return model, path, power, result


def test_dense_grid_flag_and_optin_penalty_cover_c50_6dbm() -> None:
    """U：C 50 GHz 被标记为密集栅格；默认 1.41 dB 不足，opt-in 1.1 dB 覆盖。"""
    case = _case("C", 50e9)
    model, path, power, result = _h3_result(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    delta = np.asarray(result.snr_db) - np.asarray(reference.snr_db)

    budget = estimate_h3_error_budget(model, path, power, result)
    dense = estimate_h3_error_budget(
        model,
        path,
        power,
        result,
        config=H3UncertaintyConfig(dense_grid_penalty_db=1.1),
    )

    assert budget.validity.dense_grid_overlap is True
    assert budget.validity.spacing_to_bandwidth == pytest.approx(0.78125, abs=1e-6)
    assert float(np.median(budget.margin_db)) == pytest.approx(1.4142, abs=1e-3)
    # 第五轮实测最大乐观偏差约 1.46 dB：默认 margin 留 0.04 dB 缺口。
    assert 1.0 < float(np.max(delta)) < 2.5
    assert float(np.mean(delta <= budget.margin_db)) < 0.5
    # 1.1 dB 是 U 推荐的 opt-in 值，必须覆盖该缺口。
    assert float(np.median(dense.margin_db)) == pytest.approx(2.5142, abs=1e-3)
    assert bool(np.all(delta <= dense.margin_db + 1e-12))


def test_non_dense_c_grids_are_not_flagged() -> None:
    """U：75/100 GHz 的间距不小于带宽，不应被误标为密集栅格。"""
    for spacing_hz, ratio in ((75e9, 1.171875), (100e9, 1.5625)):
        case = _case("C", spacing_hz)
        model, path, power, result = _h3_result(case, 6.0)
        budget = estimate_h3_error_budget(model, path, power, result)

        assert budget.validity.dense_grid_overlap is False
        assert budget.validity.spacing_to_bandwidth == pytest.approx(ratio, abs=1e-6)
        assert float(np.median(budget.margin_db)) == pytest.approx(1.4142, abs=1e-3)


def test_escl_six_pump_backward_uses_96_steps_and_matches_192() -> None:
    """S：E+S+C+L 六泵后向 auto=96 步，有限、收敛，且与 192 步一致。

    修复前 24 步在固定点不收敛时给出全 NaN；96 步与 192 步的逐信道差
    < 0.15 dB（第五轮/任务 S 实测 ≤0.027 dB）。
    """
    case = _case("E+S+C+L", 100e9)
    model_96, path, power, result_96 = _h3_result(case, 6.0, raman_pumps=SIX_PUMPS)
    model_192, _, _, result_192 = _h3_result(case, 6.0, raman_pumps=SIX_PUMPS, raman_steps=192)

    report = model_96.last_span_validity[0]
    assert report.raman_steps == 96
    assert report.raman_converged is True
    assert report.raman_wideband_multipump is True
    assert bool(np.all(np.isfinite(result_96.snr_db)))

    delta = np.asarray(result_96.snr_db) - np.asarray(result_192.snr_db)
    assert float(np.max(np.abs(delta))) < 0.15

    budget = estimate_h3_error_budget(model_96, path, power, result_96)
    assert bool(np.all(budget.validity.valid))
    # S 的宽波段多泵 flag 把 confidence 降 0.15，margin 从 14.81 升到 15.26 dB。
    assert float(np.median(budget.margin_db)) == pytest.approx(15.2594, abs=1e-3)


def test_unpumped_scl_edge_matches_the_numerical_primary_reference() -> None:
    """T：无泵 S+C+L 100 GHz 用 ``numerical`` 主参考时 H3 误差 < 0.5 dB。

    第五轮把 ``numerical_dra`` 当主参考，在 L/S 边缘得到约 17.5 dB 的“伪异常”；
    参考选择规则改为 ``numerical`` 后该假象消失（根因见
    ``tests/test_h3_reference_anomaly.py``）。
    """
    case = _case("S+C+L", 100e9)
    model, path, power, result = _h3_result(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    delta = np.asarray(result.snr_db) - np.asarray(reference.snr_db)

    assert bool(np.all(np.isfinite(result.snr_db)))
    assert float(np.max(np.abs(delta))) < 0.5
    assert float(np.max(delta)) < 0.5
