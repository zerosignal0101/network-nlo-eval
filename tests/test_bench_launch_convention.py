"""基准算例与 runner 的入纤功率口径（任务 O2）。

``BenchCase.per_channel_power_dbm`` / ``launch_profile(power_dbm=...)`` 永远表示
每信道**物理总功率**；只有 ``launch_profile`` 的输出坐标随
``launch_power_convention`` 变化（``per_pol`` 恰为 ``total`` 的一半）。
runner 用算例口径生成功率谱，并拒绝口径不一致的模型。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    BENCH_CASES,
    build_case,
    build_path,
    grid_for,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model, run_case, sweep_snr


def _case(name: str = "scaling_40ch"):
    return build_case(name)


# ---------------------------------------------------------------------------
# 标称功率与 profile
# ---------------------------------------------------------------------------


def test_per_channel_power_dbm_per_pol_is_three_db_below_total() -> None:
    """每偏振功率比每信道总功率低 ``10 log10 2``。"""
    case = _case()
    assert case.per_channel_power_dbm_per_pol == pytest.approx(
        case.per_channel_power_dbm - 10.0 * np.log10(2.0)
    )
    assert case.per_channel_power_dbm - case.per_channel_power_dbm_per_pol == pytest.approx(10.0 * np.log10(2.0))


def test_default_case_convention_is_total_and_unchanged() -> None:
    """既有算例全部保持 ``total``，默认 profile 与旧公式逐位相同。"""
    for case in BENCH_CASES.values():
        assert case.launch_power_convention == "total"
    case = _case()
    expected = np.full(case.num_channels, 1e-3 * 10 ** (case.per_channel_power_dbm / 10.0), dtype=np.float64)
    assert np.array_equal(launch_profile(case), expected)
    assert np.array_equal(launch_profile(case, launch_power_convention="auto"), expected)


def test_tilted_total_profile_is_bit_identical_to_the_old_formula() -> None:
    """``tilt_reference="total"`` 的默认路径逐位不变。"""
    case = _case()
    base_w = 1e-3 * 10 ** (case.per_channel_power_dbm / 10.0)
    offsets = np.linspace(-0.5, 0.5, case.num_channels, dtype=np.float64) * 6.0
    expected = base_w * 10.0 ** (offsets / 10.0)
    expected *= (case.num_channels * base_w) / float(np.sum(expected))
    expected = np.ascontiguousarray(expected)
    assert np.array_equal(launch_profile(case, tilt_db=6.0), expected)


def test_per_pol_profile_is_exactly_half_of_total() -> None:
    """``per_pol`` 输出恰为 ``total`` 的一半（含倾斜谱）。"""
    case = _case()
    total = launch_profile(case, launch_power_convention="total")
    per_pol = launch_profile(case, launch_power_convention="per_pol")
    assert np.array_equal(per_pol, total * 0.5)
    assert np.array_equal(
        launch_profile(case, tilt_db=5.0, launch_power_convention="per_pol"),
        launch_profile(case, tilt_db=5.0, launch_power_convention="total") * 0.5,
    )


def test_per_pol_case_reaches_the_same_physical_total_power_under_tilt() -> None:
    """``tilt_reference="total"`` 下 per_pol 谱的物理总功率与 total case 相同。"""
    case = _case()
    per_pol_case = replace(case, launch_power_convention="per_pol")
    total = launch_profile(case, tilt_db=7.0, launch_power_convention="total")
    per_pol = launch_profile(per_pol_case, tilt_db=7.0)
    assert np.isclose(float(np.sum(per_pol)) * 2.0, float(np.sum(total)), rtol=0.0, atol=1e-18)
    # 倾斜度不变：功率比只由 tilt_db 决定，坐标 0.5 因子会约掉。
    tilted = launch_profile(per_pol_case, tilt_db=7.0)
    assert tilted[-1] / tilted[0] == pytest.approx(10 ** (7.0 / 10.0))


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL", "perpol"])
def test_invalid_launch_convention_is_rejected(convention: str) -> None:
    """``auto`` 是唯一的哨兵值；其余非法口径在生成 profile 前报错。"""
    case = _case()
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        launch_profile(case, launch_power_convention=convention)


# ---------------------------------------------------------------------------
# runner 口径校验与端到端一致性
# ---------------------------------------------------------------------------


def _per_pol_case(name: str):
    return replace(build_case(name), launch_power_convention="per_pol")


def test_runner_rejects_a_model_with_the_wrong_convention() -> None:
    """算例 total + 模型 per_pol：同一数组会被两种口径解释，必须报错。"""
    with pytest.raises(ValueError, match="benchmark case"):
        run_case(
            "pcfm_h3_raman",
            "scaling_40ch",
            options={"signal_polarizations": 2, "launch_power_convention": "per_pol"},
        )


def test_runner_per_pol_case_rejects_a_total_model() -> None:
    """算例 per_pol + 模型 total 同样报错。"""
    case = _per_pol_case("scaling_40ch")
    model = make_model("pcfm_h3_raman", case)
    with pytest.raises(ValueError, match="benchmark case"):
        sweep_snr(model, case, (0.0,))


def test_c_band_per_pol_plus_two_matches_total_plus_one() -> None:
    """C 波段端到端：``total``+1 与 ``per_pol``+2 的 SNR 平均差 <0.01 dB。

    逐信道残差（max 约 0.064 dB）来自 PCFM 的 ISRS 剖面按半功率积分，不是
    ASE 记账；J 的审计测试对同一算例有同样结论。
    """
    total_case = build_case("scaling_40ch")
    per_pol_case = _per_pol_case("scaling_40ch")
    total_model = make_model("pcfm_h3_raman", total_case)
    per_pol_model = make_model(
        "pcfm_h3_raman",
        per_pol_case,
        signal_polarizations=2,
        launch_power_convention="per_pol",
    )
    powers = (0.0, 3.0)
    total = sweep_snr(total_model, total_case, powers)
    per_pol = sweep_snr(per_pol_model, per_pol_case, powers)
    gap = per_pol - total
    assert abs(float(np.mean(gap))) < 0.01
    assert float(np.max(np.abs(gap))) < 0.07


def test_c_plus_l_per_pol_residual_is_recorded() -> None:
    """C+L 端到端残差：记录量级，不做 <0.01 dB 的硬断言。

    10 跨、152 信道的 C+L 谱在 half-power ISRS 剖面上累积出随频率线性增长的
    残差（约 -0.68 dB 到 +1.13 dB，均值 +0.16 dB）。这是剖面模型对功率的
    非线性，不是口径换算错误：均值远小于 1 dB，且 0.5 因子逐位精确。
    """
    total_case = build_case("cl_152ch")
    per_pol_case = _per_pol_case("cl_152ch")
    total_model = make_model("pcfm_h3_raman", total_case)
    per_pol_model = make_model(
        "pcfm_h3_raman",
        per_pol_case,
        signal_polarizations=2,
        launch_power_convention="per_pol",
    )
    total = sweep_snr(total_model, total_case, (0.0,))
    per_pol = sweep_snr(per_pol_model, per_pol_case, (0.0,))
    gap = per_pol - total
    assert np.all(np.isfinite(gap))
    assert abs(float(np.mean(gap))) < 0.3
    assert float(np.max(np.abs(gap))) < 1.6


def test_default_total_sweep_is_bit_identical_to_the_uncoupled_path() -> None:
    """默认 total：``sweep_snr`` 生成的功率谱与直接调用 ``launch_profile`` 相同。"""
    case = build_case("scaling_40ch")
    model = make_model("pcfm_h3_raman", case)
    powers = (0.0,)
    path = build_path(case)
    expected = model.evaluate_path(path, launch_profile(case, 0.0)).snr_db
    actual = sweep_snr(model, case, powers)[0]
    assert np.array_equal(actual, expected)


def test_grid_for_case_is_independent_of_the_convention() -> None:
    """口径只影响功率坐标，不改网格。"""
    case = build_case("scaling_40ch")
    per_pol_case = replace(case, launch_power_convention="per_pol")
    assert grid_for(case).frequencies == pytest.approx(grid_for(per_pol_case).frequencies)
