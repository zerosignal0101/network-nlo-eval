"""任务 I 的验收测试：倾斜发射谱、倾斜下 ISRS 开关与 validity 倾斜阈值。

覆盖：

* ``launch_profile(tilt_db=0)`` 与旧实现逐位相同；
* ``total`` / ``center`` 两种功率口径的总功率/中心功率关系；
* 非法倾斜、非法口径、单信道非零倾斜直接报错；
* ``informational_tilt_confidence`` 的端点、单调性与中点值；
* ``validity_report`` 的 ``power_tilt_db`` 与硬 flag 边界（严格大于才 flag）；
* 倾斜输入下无泵 H3 仍与 PCFM 逐位一致，新字段不改变任何门控结果；
* 倾斜谱下 ISRS 开关：默认 Gaussian 谱接近零，CZ 线性谱可分辨。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    TILT_REFERENCES,
    build_case,
    build_path,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm import (
    POWER_TILT_VALIDITY_DB,
    TILT_INFORMATIONAL_KNEE_DB,
    TILT_INFORMATIONAL_ZERO_DB,
    informational_tilt_confidence,
)
from network_nlo_eval.network.elements import RamanPumpConfig

CASE = build_case("c_57ch")
CASE_1SPAN = replace(CASE, span_count=1)

#: C+L 40 信道单跨；与审计脚本的泵浦对照同算例。
CASE_CL40 = replace(
    CASE,
    name="cl_40ch_1span_test",
    bands="C+L",
    num_channels=40,
    center_frequency_hz=190.0e12,
    span_count=1,
    total_power_dbm=19.0,
)

#: H3 最优配置；逐位一致对照都用它。
H3_OPTIONS = {"poly_degree": 3, "egn": True, "edge_fix": True, "cache_spans": True, "spp_steps": "auto"}

PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]

ONE_PUMP = [RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="backward", attenuation_db_per_km=0.25)]

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _nominal_w(case) -> float:
    """算例的标称每信道功率 (W)。"""
    return 1e-3 * 10 ** (case.per_channel_power_dbm / 10.0)


def _kernel(case, model_name: str = "pcfm", **options):
    """构造模型并返回（单跨核，跨长 m）。"""
    model = make_model(model_name, case, **options)
    return model.kernel_for(build_path(case).fiber_configs[0]), case.span_length_km * 1e3


def _snr_db(result) -> np.ndarray:
    """逐信道 NLI-limited SNR (dB)。"""
    signal = np.asarray(result.signal_power_w, dtype=np.float64)
    nli = np.asarray(result.spm_noise_w, dtype=np.float64) + np.asarray(result.xpm_noise_w, dtype=np.float64)
    return 10.0 * np.log10(signal / nli)


# ---------------------------------------------------------------------------
# 倾斜谱构造器
# ---------------------------------------------------------------------------


def test_zero_tilt_is_bit_identical_to_the_flat_profile() -> None:
    """``tilt_db=0`` 必须走旧的 ``np.full`` 分支，逐位不变。"""
    expected = np.full(CASE.num_channels, _nominal_w(CASE), dtype=np.float64)
    assert np.array_equal(launch_profile(CASE), expected)
    assert np.array_equal(launch_profile(CASE, tilt_db=0.0), expected)
    assert np.array_equal(launch_profile(CASE, tilt_db=0.0, tilt_reference="center"), expected)
    # 显式功率同样保持旧语义。
    expected_0dbm = np.full(CASE.num_channels, 1e-3, dtype=np.float64)
    assert np.array_equal(launch_profile(CASE, 0.0), expected_0dbm)


@pytest.mark.parametrize("tilt_db", [3.0, 6.0, 10.0, 15.0])
def test_total_reference_keeps_the_total_power(tilt_db: float) -> None:
    """``total`` 口径：最大/最小比恰为 tilt，总功率保持平坦谱的总功率。"""
    power = launch_profile(CASE, tilt_db=tilt_db)
    assert power.shape == (CASE.num_channels,)
    assert np.all(power > 0.0)
    assert float(10.0 * np.log10(power.max() / power.min())) == pytest.approx(tilt_db, abs=1e-12)
    assert float(np.sum(power)) == pytest.approx(CASE.num_channels * _nominal_w(CASE), rel=1e-13)
    # 功率随频率单调升高。
    assert np.all(np.diff(power) > 0.0)


def test_center_reference_anchors_the_centre_channel() -> None:
    """``center`` 口径：中间信道等于标称功率，且与 ``total`` 只差全局倍率。"""
    tilt_db = 15.0
    center = launch_profile(CASE, tilt_db=tilt_db, tilt_reference="center")
    total = launch_profile(CASE, tilt_db=tilt_db, tilt_reference="total")
    # 奇数信道时中间信道严格落在 0 dB。
    assert float(center[CASE.num_channels // 2]) == pytest.approx(_nominal_w(CASE), rel=1e-13)
    # 两种口径共用同一倾斜直线，只差一个全局归一化因子。
    ratio = center / total
    assert np.allclose(ratio, ratio[0], rtol=1e-13, atol=0.0)
    assert float(ratio[0]) > 1.0
    # 15 dB 时中心口径的总功率比 total 高约 2 dB。
    excess_db = 10.0 * np.log10(np.sum(center) / np.sum(total))
    assert 1.9 < excess_db < 2.2, excess_db


@pytest.mark.parametrize("tilt_db", [-1.0, float("nan"), float("inf")])
def test_invalid_tilt_is_rejected(tilt_db: float) -> None:
    """负值/非有限倾斜必须报错，不能静默产生非物理谱。"""
    with pytest.raises(ValueError, match="tilt_db"):
        launch_profile(CASE, tilt_db=tilt_db)


def test_invalid_reference_and_single_channel_are_rejected() -> None:
    """非法口径与单信道非零倾斜直接报错。"""
    with pytest.raises(ValueError, match="tilt_reference"):
        launch_profile(CASE, tilt_db=6.0, tilt_reference="peak")
    single = replace(CASE, num_channels=1)
    with pytest.raises(ValueError, match="at least two channels"):
        launch_profile(single, tilt_db=6.0)
    # 单信道零倾斜仍然合法。
    assert launch_profile(single).shape == (1,)
    assert TILT_REFERENCES == ("total", "center")


# ---------------------------------------------------------------------------
# informational 置信度
# ---------------------------------------------------------------------------


def test_informational_confidence_endpoints_and_monotonicity() -> None:
    """Knee 以下恒为 1，zero 以上恒为 0，中间单调不增且中点为 0.5。"""
    assert informational_tilt_confidence(0.0) == 1.0
    assert informational_tilt_confidence(TILT_INFORMATIONAL_KNEE_DB) == 1.0
    assert informational_tilt_confidence(-5.0) == 1.0
    assert informational_tilt_confidence(TILT_INFORMATIONAL_ZERO_DB) == 0.0
    assert informational_tilt_confidence(100.0) == 0.0
    assert informational_tilt_confidence(float("nan")) == 0.0

    grid = np.linspace(0.0, 40.0, 401)
    scores = np.asarray([informational_tilt_confidence(value) for value in grid])
    assert np.all(np.diff(scores) <= 1e-15)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    midpoint = 0.5 * (TILT_INFORMATIONAL_KNEE_DB + TILT_INFORMATIONAL_ZERO_DB)
    assert informational_tilt_confidence(midpoint) == pytest.approx(0.5, abs=1e-12)
    # knee 由硬阈值和审计共同决定：严格小于默认阈值。
    assert TILT_INFORMATIONAL_KNEE_DB < POWER_TILT_VALIDITY_DB
    assert TILT_INFORMATIONAL_ZERO_DB == pytest.approx(2.0 * POWER_TILT_VALIDITY_DB)


# ---------------------------------------------------------------------------
# validity_report 的倾斜字段与硬 flag 边界
# ---------------------------------------------------------------------------


def test_report_reports_launch_tilt_and_flags_only_above_the_threshold() -> None:
    """``power_tilt_db`` 等于构造值；恰好等于阈值不 flag，超出才 flag。"""
    kernel, length_m = _kernel(CASE_1SPAN)
    power = launch_profile(CASE_1SPAN, tilt_db=10.0)
    report = kernel.validity_report(length_m, power)
    assert report.power_tilt_db == pytest.approx(10.0, abs=1e-12)
    assert report.effective_tilt_db == pytest.approx(10.0, abs=1e-12)
    assert report.tilt_confidence_informational == pytest.approx(informational_tilt_confidence(10.0), abs=1e-15)
    assert not bool(np.any(report.flagged))
    assert bool(np.all(report.valid))

    at_threshold = launch_profile(CASE_1SPAN, tilt_db=POWER_TILT_VALIDITY_DB)
    boundary = kernel.validity_report(length_m, at_threshold)
    assert boundary.power_tilt_db == pytest.approx(POWER_TILT_VALIDITY_DB, abs=1e-12)
    assert not bool(np.any(boundary.flagged))
    assert bool(np.all(boundary.valid))
    assert boundary.tilt_confidence_informational < 1.0

    above = kernel.validity_report(length_m, launch_profile(CASE_1SPAN, tilt_db=POWER_TILT_VALIDITY_DB + 5.0))
    assert bool(np.any(above.flagged))
    assert bool(np.any(~above.valid))
    assert float(np.min(above.confidence)) < float(np.min(boundary.confidence))


def test_flat_report_is_unchanged_by_the_informational_fields() -> None:
    """平坦谱：倾斜字段取默认值，门控数组仍是原来的结论。"""
    kernel, length_m = _kernel(CASE)
    report = kernel.validity_report(length_m, launch_profile(CASE))
    assert report.effective_tilt_db == 0.0
    assert report.tilt_confidence_informational == 1.0
    assert bool(np.all(report.valid))
    assert not bool(np.any(report.flagged))
    # 已验证的 C 波段只可能被 XCI 的“不material”分支打 0.9，倾斜不参与。
    assert float(np.min(report.confidence)) >= 0.9


def test_pumped_report_uses_the_max_of_power_and_gain_tilt() -> None:
    """有泵 H3 的有效倾斜是功率倾斜与增益倾斜的较大者。"""
    kernel, length_m = _kernel(CASE_CL40, "pcfm_h3_raman", raman_pumps=PUMPS)
    power = launch_profile(CASE_CL40, tilt_db=10.0)
    report = kernel.validity_report(length_m, power)
    assert report.power_tilt_db == pytest.approx(10.0, abs=1e-12)
    assert report.gain_tilt_db > 0.0
    assert report.effective_tilt_db == pytest.approx(max(report.power_tilt_db, report.gain_tilt_db), abs=1e-12)
    assert report.pump_active
    assert report.tilt_confidence_informational == pytest.approx(
        informational_tilt_confidence(report.effective_tilt_db), abs=1e-15
    )


# ---------------------------------------------------------------------------
# 默认路径逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tilt_db", [0.0, 10.0, 15.0])
def test_tilted_no_pump_h3_is_bit_identical_to_pcfm(tilt_db: float) -> None:
    """倾斜输入下，无泵 H3 仍逐位等于 PCFM（新字段不触碰数值）。"""
    path = build_path(CASE_1SPAN)
    power = launch_profile(CASE_1SPAN, tilt_db=tilt_db)
    baseline = make_model("pcfm", CASE_1SPAN, **H3_OPTIONS).evaluate_path(path, power)
    new = make_model("pcfm_h3_raman", CASE_1SPAN).evaluate_path(path, power)
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(baseline, name)), np.asarray(getattr(new, name))), name


def test_default_launch_profile_matches_every_registered_case() -> None:
    """所有内置算例的默认谱必须与旧的 ``np.full`` 完全一致。"""
    from network_nlo_eval.benchmarks.cases import BENCH_CASES

    for case in BENCH_CASES.values():
        assert np.array_equal(
            launch_profile(case),
            np.full(case.num_channels, _nominal_w(case), dtype=np.float64),
        ), case.name


# ---------------------------------------------------------------------------
# 倾斜谱下的 ISRS 开关
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gain_model", ["gaussian", "linear"])
def test_isrs_switch_under_tilt(gain_model: str) -> None:
    """Gaussian 谱下 ISRS 开关是数值零；CZ 线性谱下必须可分辨。"""
    path = build_path(CASE_1SPAN)
    power = launch_profile(CASE_1SPAN, tilt_db=15.0)
    options = {"raman_pumps": ONE_PUMP, "raman_gain_model": gain_model}
    on = make_model("pcfm_h3_raman", CASE_1SPAN, include_isrs=True, **options).evaluate_path(path, power)
    off = make_model("pcfm_h3_raman", CASE_1SPAN, include_isrs=False, **options).evaluate_path(path, power)
    delta = _snr_db(on) - _snr_db(off)
    if gain_model == "gaussian":
        # 默认 Gaussian 谱在 75 GHz–4 THz 的尾部近零，开关没有可分辨作用。
        assert float(np.max(np.abs(delta))) < 1.0e-3, float(np.max(np.abs(delta)))
    else:
        assert float(np.max(np.abs(delta))) > 0.2, float(np.max(np.abs(delta)))
        # 重分配不是单调的：两端与中段的符号/幅度都不同，至少要覆盖正负两侧。
        assert float(np.max(delta)) > 0.2 and float(np.min(delta)) < -0.05
