"""任务 AF：H3 精度矩阵 v3 的亚秒级代表点锚点。

完整 40 点复核见 ``scripts/h3_matrix_v3_recheck.py`` 与
``results/h3_matrix_v3/recheck.json``。这里只钉住五个跨平台稳定的代表点，
全部取本 worktree 当前基线重算的 H3；参考要么用本地快速 ``numerical``，
要么直接检查 validity，不读第五轮矩阵、不跑 661 点。

覆盖的回归面：

* 密集栅格 C 50 GHz / 5 跨 / 6 dBm：AA 标记为 dense，默认 1.41 dB 不够，
  opt-in 1.1 dB 刚好覆盖；
* 密集栅格 C 37.5 GHz / 1 跨 / 6 dBm：误差 >2 dB，但强倾斜行把 margin 抬到
  8.79 dB，说明默认覆盖是“倾斜兜底”而不是“密集栅格已处理”；
* T 规则：无泵 S+C+L 100 GHz 用 ``numerical`` 主参考，H3 误差 <0.5 dB；
* Y/S：S+C+L 六泵的 1505 nm 泵落在 S 带内，全部信道被硬判无效，同时
  96 步宽波段多泵标记为真；C+L 六泵保持有效；
* provider：361 信道第一次 miss、第二次 hit，命中返回同一报告。
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.uncertainty import H3UncertaintyConfig, estimate_h3_error_budget
from network_nlo_eval.network.elements import RamanPumpConfig
from network_nlo_eval.rwa.validity import H3LinkValidityProvider

#: 与第五轮矩阵、``tests/test_pcfm_h3_raman.py::SIX_PUMPS`` 相同的六泵配置。
#: 1505 nm 的泵频约 199.2 THz，落在 S 带内。
SIX_PUMPS = (
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
)


def _case(bands: str, spacing_hz: float, spans: int = 1) -> BenchCase:
    """构造与复核脚本同源的算例（64 GBaud, 100 km, NF 5.5 dB）。"""
    low, high = band_edges_hz(bands)
    return BenchCase(
        name=f"v3_{bands}_{spacing_hz / 1e9:g}_{spans}",
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


def _h3_and_budget(case: BenchCase, power_dbm: float, **options):
    """返回 H3 结果、光路、入纤功率、模型与预算。"""
    path = build_path(case)
    power = launch_profile(case, power_dbm)
    model = make_model("pcfm_h3_raman", case, **options)
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)
    return model, path, power, result, budget


def test_dense_c50_five_span_keeps_the_known_coverage_gap() -> None:
    """AA：C 50 GHz / 5 跨 / 6 dBm 默认 margin 不足，opt-in 1.1 dB 覆盖。

    实测最大乐观偏差约 1.50 dB，默认 1.414 dB margin 只覆盖 11.5% 信道；
    1.1 dB 惩罚把 margin 抬到 2.51 dB，全部覆盖。
    """
    case = _case("C", 50e9, spans=5)
    model, path, power, result, budget = _h3_and_budget(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    delta = np.asarray(result.snr_db) - np.asarray(reference.snr_db)
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
    assert 1.2 < float(np.max(delta)) < 2.0
    assert float(np.mean(delta <= budget.margin_db)) < 0.2

    assert float(np.median(dense.margin_db)) == pytest.approx(2.5142, abs=1e-3)
    assert bool(np.all(delta <= dense.margin_db + 1e-12))


def test_dense_c37_5_one_span_is_covered_by_strong_tilt_not_by_dense_penalty() -> None:
    """AA：C 37.5 GHz / 1 跨 / 6 dBm 误差 >2 dB，默认覆盖来自强倾斜行。

    默认 margin 是 8.79 dB（strong tilt），不是 1.41 dB 的窄带 margin；
    因此不能把“覆盖”误读成密集栅格已经被正确处理。
    """
    case = _case("C", 37.5e9, spans=1)
    _model, path, power, result, budget = _h3_and_budget(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    delta = np.asarray(result.snr_db) - np.asarray(reference.snr_db)

    assert budget.validity.dense_grid_overlap is True
    assert bool(np.any(budget.validity.strong_tilt)) is True
    assert float(np.median(budget.margin_db)) == pytest.approx(8.7918, abs=1e-2)
    assert 2.0 < float(np.max(delta)) < 3.0
    assert float(np.mean(delta <= budget.margin_db)) == pytest.approx(1.0)


def test_unpumped_scl_uses_numerical_primary_and_stays_under_half_db() -> None:
    """T：无泵 S+C+L 100 GHz 用 ``numerical`` 主参考，H3 误差 <0.5 dB。

    实测 max|Δ| = 0.262 dB、mean Δ = +0.090 dB；宽带 margin 约 14.81 dB。
    该点不触发 Y（无泵）或 S（无泵）flag。
    """
    case = _case("S+C+L", 100e9)
    _model, path, power, result, budget = _h3_and_budget(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    delta = np.asarray(result.snr_db) - np.asarray(reference.snr_db)

    assert bool(np.all(np.isfinite(result.snr_db)))
    assert float(np.max(np.abs(delta))) < 0.5
    assert float(np.mean(delta)) == pytest.approx(0.090, abs=0.02)
    assert budget.validity.band_class == "wide"
    assert float(np.median(budget.margin_db)) == pytest.approx(14.8094, abs=1e-2)
    assert not bool(np.any(budget.validity.raman_pump_signal_overlap))
    assert not bool(np.any(budget.validity.raman_wideband_multipump))


def test_scl_six_pump_is_rejected_by_y_and_flagged_by_s() -> None:
    """Y/S：S+C+L 六泵的 1505 nm 泵在 S 带内 -> 全部无效；C+L 保持有效。

    与 W（基线 6705cdb）相比，点值未变，但 Y 之后 S+C+L 后向/前向从
    “valid + 有限 margin” 变成 ``valid=False`` + ``margin=inf``。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*pump falls inside.*")

        scl_case = _case("S+C+L", 100e9)
        scl_model, scl_path, scl_power, scl_result, scl_budget = _h3_and_budget(
            scl_case, 6.0, raman_pumps=SIX_PUMPS
        )
        cl_case = _case("C+L", 100e9)
        cl_model, cl_path, cl_power, cl_result, cl_budget = _h3_and_budget(
            cl_case, 6.0, raman_pumps=SIX_PUMPS
        )

    assert bool(np.all(np.isfinite(scl_result.snr_db)))
    assert scl_model.last_span_validity[0].raman_steps == 96
    assert bool(np.all(scl_budget.validity.raman_pump_signal_overlap))
    assert not bool(np.any(scl_budget.validity.valid))
    assert bool(np.all(np.isinf(scl_budget.margin_db)))
    assert bool(np.all(scl_budget.validity.raman_wideband_multipump))
    assert scl_budget.validity.raman_pump_min_offset_hz == 0.0

    assert not bool(np.any(cl_budget.validity.raman_pump_signal_overlap))
    assert bool(np.all(cl_budget.validity.valid))
    assert cl_budget.validity.raman_pump_min_offset_hz > 0.0


def test_provider_caches_361_channel_report_on_the_second_call() -> None:
    """Provider：361 信道第一次 miss 重算，第二次 hit 返回同一报告。"""
    case = _case("E+S+C+L", 100e9)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case)
    provider = H3LinkValidityProvider(model)
    fiber = path.fiber_configs[0]
    span_length_m = 100e3

    started = time.perf_counter()
    first = provider((0, 1), fiber, None, power, span_length_m)
    miss_s = time.perf_counter() - started
    started = time.perf_counter()
    second = provider((0, 1), fiber, None, power, span_length_m)
    hit_s = time.perf_counter() - started

    assert provider.cache.misses == 1
    assert provider.cache.hits == 1
    assert second is first
    assert first.fingerprint == second.fingerprint
    assert hit_s < miss_s / 10.0
