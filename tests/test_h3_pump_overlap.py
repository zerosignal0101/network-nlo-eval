"""泵-信号带重叠的硬有效性门（第八轮任务 Y）。

覆盖四件事：

* **E+S+C+L / S+C+L 六泵**：1505 nm（约 199.2 THz）落在 S 带内，
  逐跨报告与误差预算都必须判为不可用，但点值 SNR 保持有限且不变；
* **C+L 六泵**：泵全部在信号带外，必须保持 ``valid``，且最小偏移为正；
* **无泵**：新字段全部为默认值，结果与 H3 PCFM 逐位一致；
* **区分**：``raman_pump_signal_overlap``（物理配置错误）与
  ``raman_wideband_multipump``（带外多泵的数值收敛风险）互相独立，
  以及 warning 只发一次。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.raman import C_LIGHT
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
from network_nlo_eval.network.elements import RamanPumpConfig

#: 与第五/六/七轮精度矩阵相同的六泵配置（论文 hybrid 的 E 波段泵组）。
SIX_PUMPS = (
    (1340.0, 0.63),
    (1360.0, 0.63),
    (1427.0, 0.165),
    (1445.0, 0.131),
    (1476.0, 0.125),
    (1505.0, 0.075),
)

#: ``pcfm`` 的显式 H3 配置；对照必须用同一组选项才是逐位比较。
H3_OPTIONS = {
    "poly_degree": 3,
    "egn": True,
    "edge_fix": True,
    "cache_spans": True,
    "spp_steps": "auto",
}


def _pumps(direction: str = "backward", spec=SIX_PUMPS) -> list[RamanPumpConfig]:
    """按波长/功率表构造泵浦，方向统一。"""
    return [
        RamanPumpConfig(
            wavelength_nm=wavelength,
            power_w=power,
            direction=direction,  # type: ignore[arg-type]
            attenuation_db_per_km=0.25,
        )
        for wavelength, power in spec
    ]


def _case(bands: str, spacing_hz: float = 100e9, spans: int = 1) -> BenchCase:
    """按波段边界构造与精度矩阵同源的单跨/多跨算例。"""
    low, high = band_edges_hz(bands)
    return BenchCase(
        name=f"overlap_{bands}_{spacing_hz / 1e9:g}",
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


def _run(case: BenchCase, **options) -> tuple:
    """跑一次路径，返回 ``(model, path, power, result)``。"""
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case, **options)
    return model, path, power, model.evaluate_path(path, power)


@pytest.mark.parametrize("bands,spacing_hz", [("E+S+C+L", 100e9), ("S+C+L", 75e9)])
def test_six_pump_in_band_configuration_is_invalid_but_point_value_survives(
    bands: str, spacing_hz: float
) -> None:
    """1505 nm 泵落在 S 带内：全信道无效，点值不变、warning 一次。

    1505 nm 的泵频约 199.2 THz，位于 S 带的 195.9-205.4 THz 内。该配置不是
    物理 DRA，必须由 validity 报告拒绝；但点 NLI/ASE/SNR 是模型输出，默认
    ``validity_mode="warn"`` 不得改写。
    """
    case = _case(bands, spacing_hz)
    path = build_path(case)
    power = launch_profile(case, 6.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_model = make_model("pcfm_h3_raman", case, raman_pumps=_pumps(), validity_mode="warn")
        warn_result = warn_model.evaluate_path(path, power)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strict_model = make_model("pcfm_h3_raman", case, raman_pumps=_pumps(), validity_mode="strict")
        strict_result = strict_model.evaluate_path(path, power)

    report = warn_model.last_span_validity[0]
    assert report.raman_pump_signal_overlap is True
    assert report.raman_pump_min_offset_hz == 0.0
    assert bool(np.all(report.raman_pump_overlap_channels))
    assert not bool(np.any(report.valid))
    assert bool(np.all(report.flagged))
    assert bool(np.all(report.confidence <= 0.2))

    # 点值仍有限且未被动过：warn 模式与 strict 的点值只在 SNR 上由 strict 置 NaN。
    assert bool(np.all(np.isfinite(warn_result.snr_db)))
    assert np.array_equal(warn_result.signal_power_w, strict_result.signal_power_w)
    assert np.array_equal(warn_result.spm_noise_w, strict_result.spm_noise_w)
    assert np.array_equal(warn_result.xpm_noise_w, strict_result.xpm_noise_w)
    assert bool(np.all(np.isnan(strict_result.snr_db)))

    overlap_warnings = [item for item in caught if "falls inside the occupied signal" in str(item.message)]
    assert len(overlap_warnings) == 1


def test_c_l_six_pump_stays_valid_and_reports_a_positive_offset() -> None:
    """C+L 六泵的泵全在带外：不得误判为重叠，信道必须保持可用。"""
    case = build_case("cl_152ch")
    model, path, power, result = _run(case, raman_pumps=_pumps())
    report = model.last_span_validity[0]
    budget = estimate_h3_error_budget(model, path, power, result)

    expected_offset = C_LIGHT / (1505.0e-9) - float(np.max(model.f_abs_hz))
    assert report.raman_pump_signal_overlap is False
    assert report.raman_pump_min_offset_hz == pytest.approx(expected_offset, rel=1e-6)
    assert report.raman_pump_min_offset_hz > 3e12
    assert not bool(np.any(report.raman_pump_overlap_channels))
    assert bool(np.all(report.valid))
    assert not bool(np.any(budget.validity.raman_pump_signal_overlap))
    assert bool(np.all(budget.validity.valid))
    assert bool(np.all(np.isfinite(budget.margin_db)))


def test_overlap_budget_makes_the_channels_unusable() -> None:
    """重叠信道在预算里与未收敛同义：``valid=False``、``margin=inf``。"""
    case = _case("E+S+C+L", 100e9)
    model, path, power, result = _run(case, raman_pumps=_pumps())
    budget = estimate_h3_error_budget(model, path, power, result)

    assert bool(np.all(budget.validity.raman_pump_signal_overlap))
    assert not bool(np.any(budget.validity.valid))
    assert bool(np.all(np.isinf(budget.margin_db)))
    assert bool(np.all(np.isneginf(budget.adjusted_snr_db(result.snr_db))))
    assert bool(np.all(budget.validity.confidence <= 0.2))
    assert budget.validity.raman_pump_min_offset_hz == 0.0
    assert any("not a physical DRA" in note for note in budget.notes)

    payload = budget.to_dict()
    assert payload["validity"]["raman_pump_signal_overlap"] == [True] * case.num_channels
    assert payload["validity"]["raman_pump_min_offset_hz"] == 0.0


def test_no_pump_report_uses_the_defaults_and_the_h3_result_is_bit_identical() -> None:
    """无泵：新字段全默认，结果与 H3 PCFM 逐位一致。"""
    case = build_case("cl_152ch")
    path = build_path(case)
    power = launch_profile(case, 6.0)

    h3 = make_model("pcfm_h3_raman", case)
    base = make_model("pcfm", case, **H3_OPTIONS)
    h3_result = h3.evaluate_path(path, power)
    base_result = base.evaluate_path(path, power)

    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(
            np.asarray(getattr(h3_result, name), dtype=np.float64),
            np.asarray(getattr(base_result, name), dtype=np.float64),
        ), name

    report = h3.last_span_validity[0]
    assert report.raman_pump_signal_overlap is False
    assert report.raman_pump_min_offset_hz == float("inf")
    assert report.raman_pump_overlap_channels.shape == (case.num_channels,)
    assert not bool(np.any(report.raman_pump_overlap_channels))
    assert bool(np.all(report.valid))


def test_overlap_warning_is_emitted_once_per_kernel() -> None:
    """两跨同一 H3 Raman 核：重叠 warning 只发一次。"""
    case = _case("E+S+C+L", 100e9, spans=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = make_model("pcfm_h3_raman", case, raman_pumps=_pumps())
        model.evaluate_path(build_path(case), launch_profile(case, 6.0))

    overlap_warnings = [item for item in caught if "falls inside the occupied signal" in str(item.message)]
    assert len(overlap_warnings) == 1
    assert all(report.raman_pump_signal_overlap for report in model.last_span_validity)


def test_outside_band_wideband_flag_is_independent_of_overlap() -> None:
    """带外四泵仍是宽波段收敛风险，但不是“泵在信号带内”。

    去掉 1476/1505 nm 后泵全部落在 S+C+L 带外，``raman_wideband_multipump``
    仍为真（96 步分支），而 ``raman_pump_signal_overlap`` 为假。该算例本身
    因泵组不匹配而欠收敛，属于另一个 flag，正好说明两个判据互不替代。
    """
    case = _case("S+C+L", 100e9)
    outside = tuple(item for item in SIX_PUMPS if item[0] < 1460.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, path, power, result = _run(case, raman_pumps=_pumps(spec=outside))

    report = model.last_span_validity[0]
    assert len(outside) >= 4
    assert report.raman_wideband_multipump is True
    assert report.raman_pump_signal_overlap is False
    assert report.raman_pump_min_offset_hz > 0.0
    assert bool(np.all(np.isfinite(result.snr_db)))
