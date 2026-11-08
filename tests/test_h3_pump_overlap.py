"""泵-信号带重叠的硬有效性门（第八轮任务 Y，第九轮任务 AD 精化）。

覆盖五件事：

* **E+S+C+L / S+C+L 六泵**：1505 nm（约 199.2 THz）落在 S 带内，
  逐跨报告与误差预算都必须判为不可用，但点值 SNR 保持有限且不变；
* **C+L 六泵**：泵全部在信号带外，必须保持 ``valid``，且最小偏移为正；
* **无泵**：新字段全部为默认值，结果与 H3 PCFM 逐位一致；
* **区分**：``raman_pump_signal_overlap``（物理配置错误）与
  ``raman_wideband_multipump``（带外多泵的数值收敛风险）互相独立，
  以及 warning 只发一次；
* **带边与 guard（AD）**：占用带按信道带边 ``f_i ± B_i/2`` 构造，
  两侧可加 ``pump_signal_overlap_guard_hz``（默认 0）；正 guard 扩带、
  负 guard 缩带，且 guard 只改有效性报告，不改任何点值。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm_h3_raman import PUMP_SIGNAL_OVERLAP_GUARD_HZ, _pump_signal_overlap
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

    # 偏移量到上侧信道带边，而不是最外侧信道中心（AD 的带边语义）。
    upper_edge = float(np.max(model.f_abs_hz + 0.5 * model.channel_bandwidth_hz_channels))
    expected_offset = C_LIGHT / (1505.0e-9) - upper_edge
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
    assert h3.pump_signal_overlap_guard_hz == 0.0
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


def _assert_point_values_equal(left, right) -> None:
    """断言两次评估的逐信道点值逐位相同（AD：guard 不得进入数值路径）。"""
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(
            np.asarray(getattr(left, name), dtype=np.float64),
            np.asarray(getattr(right, name), dtype=np.float64),
        ), name


def test_pump_band_uses_the_channel_edges_not_the_centres() -> None:
    """占用带是 ``[min(f_i - B_i/2), max(f_i + B_i/2)]``，不是中心区间。

    两个 100 GHz 信道、中心 193.0/193.1 THz 时，上带边是 193.15 THz。
    泵在 193.14 THz 处落在最外侧信道的半带宽内，旧的中心区间规则会漏判；
    AD 必须判为重叠并把最小偏移记为 0。带外 10 GHz 的泵仍为正偏移。
    """
    centres = np.array([193.0e12, 193.1e12])
    bandwidths = np.full(2, 100e9)
    active = np.ones(2, dtype=bool)

    inside, offset = _pump_signal_overlap(np.array([193.14e12]), centres, bandwidths, active)
    assert inside is True
    assert offset == 0.0
    # 中心区间 [193.0, 193.1] 会给出正偏移；带边语义才有下面的 0。
    assert 193.14e12 > float(np.max(centres))

    outside, offset = _pump_signal_overlap(np.array([193.16e12]), centres, bandwidths, active)
    assert outside is False
    assert offset == pytest.approx(10e9)

    left_inside, left_offset = _pump_signal_overlap(np.array([192.96e12]), centres, bandwidths, active)
    assert left_inside is True
    assert left_offset == 0.0

    left_outside, left_offset = _pump_signal_overlap(np.array([192.94e12]), centres, bandwidths, active)
    assert left_outside is False
    assert left_offset == pytest.approx(10e9)


def test_positive_guard_widens_and_negative_guard_shrinks_the_band() -> None:
    """正 guard 扩带、负 guard 缩带；偏移始终量到（加 guard 后的）带边。"""
    centres = np.array([193.0e12, 193.1e12])
    bandwidths = np.full(2, 100e9)
    active = np.ones(2, dtype=bool)
    # 上带边 193.15 THz，泵在上带边外 50 GHz。
    pumps = np.array([193.20e12])

    overlap, offset = _pump_signal_overlap(pumps, centres, bandwidths, active, 0.0)
    assert overlap is False
    assert offset == pytest.approx(50e9)

    overlap, offset = _pump_signal_overlap(pumps, centres, bandwidths, active, 60e9)
    assert overlap is True
    assert offset == 0.0

    overlap, offset = _pump_signal_overlap(pumps, centres, bandwidths, active, -50e9)
    assert overlap is False
    assert offset == pytest.approx(100e9)

    # 无泵/无活跃信道时 guard 不改变默认值。
    empty = np.zeros(0, dtype=np.float64)
    assert _pump_signal_overlap(empty, centres, bandwidths, active, 60e9) == (False, float("inf"))
    assert _pump_signal_overlap(pumps, centres, bandwidths, np.zeros(2, dtype=bool), 60e9) == (
        False,
        float("inf"),
    )


def test_default_guard_is_zero_and_keeps_the_round_y_decision() -> None:
    """默认 guard 为 0：E+S+C+L 的判定与 Y 相同，点值逐位不变。

    默认模型与显式 ``pump_signal_overlap_guard_hz=0.0`` 的逐信道点值逐位
    相同；重叠判定仍由 1505 nm 泵触发，最小偏移仍为 0（带内）。
    """
    assert PUMP_SIGNAL_OVERLAP_GUARD_HZ == 0.0
    case = _case("E+S+C+L", 100e9)
    default_model, path, power, default_result = _run(case, raman_pumps=_pumps())
    explicit_model, _, _, explicit_result = _run(
        case, raman_pumps=_pumps(), pump_signal_overlap_guard_hz=0.0
    )

    assert default_model.pump_signal_overlap_guard_hz == 0.0
    assert explicit_model.pump_signal_overlap_guard_hz == 0.0
    kernel = default_model.kernel_for(build_path(case).fiber_configs[0])
    assert kernel.pump_signal_overlap_guard_hz == 0.0

    report = default_model.last_span_validity[0]
    assert report.raman_pump_signal_overlap is True
    assert report.raman_pump_min_offset_hz == 0.0
    assert bool(np.all(report.raman_pump_overlap_channels))
    _assert_point_values_equal(default_result, explicit_result)


def test_positive_guard_flips_validity_without_touching_point_values() -> None:
    """Guard 只改有效性报告：把 C+L 的带外泵扩进带内，点值逐位不变。"""
    case = build_case("cl_152ch")
    default_model, path, power, default_result = _run(case, raman_pumps=_pumps())
    pump_hz = C_LIGHT / (1505.0e-9)
    upper_edge = float(np.max(default_model.f_abs_hz + 0.5 * default_model.channel_bandwidth_hz_channels))
    guard_hz = pump_hz - upper_edge + 1e9

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        guarded_model, _, _, guarded_result = _run(
            case,
            raman_pumps=_pumps(),
            pump_signal_overlap_guard_hz=guard_hz,
        )

    assert guarded_model.pump_signal_overlap_guard_hz == pytest.approx(guard_hz)
    report = guarded_model.last_span_validity[0]
    assert report.raman_pump_signal_overlap is True
    assert report.raman_pump_min_offset_hz == 0.0
    assert not bool(np.any(report.valid))
    _assert_point_values_equal(default_result, guarded_result)


def test_non_finite_guard_is_rejected() -> None:
    """Guard 必须是有限数；NaN/inf 不能静默进入判定。"""
    case = build_case("cl_152ch")
    with pytest.raises(ValueError, match="pump_signal_overlap_guard_hz"):
        make_model("pcfm_h3_raman", case, pump_signal_overlap_guard_hz=float("nan"))
