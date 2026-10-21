"""数值 ISRS-GN 参考模型的验收测试。

模型为 Semrau, Killey, Bayvel, JLT 2018 (doi:10.1109/jlt.2018.2830973) 的
数值实现：Eq. (6) 的耦合 SRS 功率剖面用常步长 RK4 数值积分，Eq. (4) 的 GN
双重积分用频率四边形求积。本文件覆盖以下验收点：

* 单信道的 GN 自相位调制闭式解；
* 频率求积的收敛性；
* 零色散下的数值有限性；
* 跨间相干累加：单跨逐位一致、多跨相干增益、解析极限、确定性；
* 与闭式 ``ecz`` 基线的逐信道 SNR 一致性；
* ``egn=True`` 被显式拒绝。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, build_case, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, SpanContext, create_model
from network_nlo_eval.network.elements import EDFAConfig

#: 参考波长，模型的色散展开以它为中心。
REFERENCE_WAVELENGTH_M = 1550e-9
#: 光速 (m/s)。
C_LIGHT_M_S = 299792458.0
#: 单信道测试用的非线性折射率与有效模面积。
N2_M2_W = 2.6e-20
A_EFF_M2 = 80e-12
#: 单信道测试用的信道带宽 (Hz) 与入纤功率 (W)。
SINGLE_BANDWIDTH_HZ = 64e9
SINGLE_POWER_W = 1e-3


def _zero_dispersion_fiber(length_km: float = 100.0):
    """构造一根零色散、光衰减为常数的参考光纤。"""
    return SMF_100KM.model_copy(
        update={
            "length_km": length_km,
            "max_span_length_km": length_km,
            "use_attenuation_polynomial": False,
            "attenuation_db_km_ref": 0.162,
            "dispersion_parameter_d": 0.0,
            "dispersion_slope_s": 0.0,
        }
    )


def _single_channel_path(fiber, span_count: int = 1):
    """构造只有一根信道的光路与入纤功率。"""
    path = MultiSpanOpticalPath(
        [fiber] * span_count,
        [EDFAConfig(noise_figure_db=5.5)] * span_count,
    )
    grid = SpectrumGrid(1, 75e9, C_LIGHT_M_S / REFERENCE_WAVELENGTH_M)
    return grid, path, np.array([SINGLE_POWER_W])


def test_single_channel_matches_closed_form_spm() -> None:
    """单信道零色散时，模型必须复现 GN 自相位调制闭式解 (4/9) gamma^2 P^3 L_eff^2。"""
    fiber = _zero_dispersion_fiber()
    grid, path, power = _single_channel_path(fiber)

    gamma = 2 * np.pi * N2_M2_W / (REFERENCE_WAVELENGTH_M * A_EFF_M2)
    alpha = 0.162 * np.log(10) / 10 / 1e3
    length_m = 100e3
    l_eff = (1 - np.exp(-alpha * length_m)) / alpha
    closed_form = (4.0 / 9.0) * gamma**2 * SINGLE_POWER_W**3 * l_eff**2

    model = create_model(
        "numerical",
        grid,
        fiber,
        symbol_rate_hz=64e9,
        quadrature_points=16,
        ode_steps=256,
    )
    result = model.evaluate_path(path, power)

    modelled = float(result.spm_noise_w[0])
    relative_error = abs(modelled / closed_form - 1.0)
    assert relative_error < 1e-6, f"SPM relative error {relative_error:.3e} exceeds 1e-6"


def test_quadrature_convergence() -> None:
    """求积点加倍后，逐信道 SNR 的变化必须小于 0.05 dB。"""
    case = build_case("scaling_160ch")
    path = build_path(case)
    power = launch_profile(case, None)

    coarse = make_model("numerical", case, quadrature_points=64).evaluate_path(path, power)
    fine = make_model("numerical", case, quadrature_points=128).evaluate_path(path, power)

    deviation = np.abs(np.asarray(fine.snr_db) - np.asarray(coarse.snr_db))
    max_deviation = float(np.max(deviation))
    print(f"max per-channel SNR deviation (64 -> 128 quadrature cells): {max_deviation:.5f} dB")
    assert max_deviation < 0.05, f"SNR deviation {max_deviation:.4f} dB exceeds 0.05 dB"


def test_zero_dispersion_finite() -> None:
    """零色散链路下，SNR 与非线性噪声必须有限且非负。"""
    case = build_case("scaling_40ch")
    fiber = SMF_100KM.model_copy(
        update={
            "length_km": case.span_length_km,
            "max_span_length_km": case.span_length_km,
            "dispersion_parameter_d": 0.0,
            "dispersion_slope_s": 0.0,
        }
    )
    path = MultiSpanOpticalPath(
        [fiber] * case.span_count,
        [EDFAConfig(noise_figure_db=5.5)] * case.span_count,
    )
    power = launch_profile(case, None)

    model = create_model(
        "numerical",
        grid_for(case),
        fiber,
        symbol_rate_hz=case.symbol_rate_hz,
    )
    result = model.evaluate_path(path, power)

    assert np.all(np.isfinite(np.asarray(result.snr_db)))
    assert np.all(np.isfinite(np.asarray(result.spm_noise_w)))
    assert np.all(np.isfinite(np.asarray(result.xpm_noise_w)))
    assert np.all(np.asarray(result.spm_noise_w) >= 0.0)
    assert np.all(np.asarray(result.xpm_noise_w) >= 0.0)
    assert float(np.max(np.asarray(result.snr_db))) < 60.0


def test_multi_span_accumulation() -> None:
    """十跨链路 SNR 必须低于单跨链路，且 NLI 以相干增益大于线性增长。

    这条测试改造前锁的是**非相干**累加口径（十跨 = 单跨 x 10）。跨间相干
    累加上线后，路由层按 ``|Σ a_k|²`` 相加，物理上正确的增长因子是
    ``R = N + 2 Σ (N-Δs) V(Δs)``（``V`` 是跨间相干可见度）：色散不为零时
    部分相干，增长因子严格大于 10，但离完全相干的 100 还差得远。
    """
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)
    fiber = SMF_100KM.model_copy(
        update={
            "length_km": case.span_length_km,
            "max_span_length_km": case.span_length_km,
        }
    )

    def _run(span_count: int):
        path = MultiSpanOpticalPath(
            [fiber] * span_count,
            [EDFAConfig(noise_figure_db=5.5)] * span_count,
        )
        return make_model("numerical", case).evaluate_path(path, power)

    single = _run(1)
    ten = _run(10)

    single_snr = np.asarray(single.snr_db)
    ten_snr = np.asarray(ten.snr_db)
    assert np.all(ten_snr < single_snr), "ten-span SNR must be below the single-span SNR"

    single_nli = np.asarray(single.spm_noise_w) + np.asarray(single.xpm_noise_w)
    ten_nli = np.asarray(ten.spm_noise_w) + np.asarray(ten.xpm_noise_w)
    ratio = ten_nli / single_nli
    print(f"NLI growth over ten spans: min {ratio.min():.3f}, max {ratio.max():.3f}")
    # 相干增益：严格大于非相干基线 10，且远小于完全相干极限 100。
    assert np.all(ratio > 10.0), f"NLI growth {ratio} shows no coherent gain over the incoherent sum"
    assert np.all(ratio < 20.0), f"NLI growth {ratio} is closer to full coherence than the C-band visibility allows"


def test_single_span_is_bit_identical_to_the_frozen_power_path() -> None:
    """单跨链路必须与改造前逐位相同：相干口径退化为功率口径。

    冻结的黄金值取自改造前的 ``numerical`` 模型（同一算例、同一入纤功率）。
    比较的是逐位的 IEEE-754 位型，不是容差。
    """
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)
    fiber = SMF_100KM.model_copy(
        update={
            "length_km": case.span_length_km,
            "max_span_length_km": case.span_length_km,
        }
    )
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)])
    result = make_model("numerical", case).evaluate_path(path, power)

    # 改造前的数值（同一 commit 的旧实现）。
    assert float(np.sum(np.asarray(result.spm_noise_w))) == 6.115456062189079e-06
    assert float(np.sum(np.asarray(result.xpm_noise_w))) == 1.1117805960227494e-05
    assert float(np.sum(np.asarray(result.ase_noise_w))) == 4.796057499933039e-05
    assert float(np.asarray(result.snr_db)[len(result.snr_db) // 2]) == 27.80040675089618

    # 单跨时不给出复振幅：路由层走原来的纯功率累加，浮点求和顺序不变。
    step = make_model("numerical", case).span_step(
        fiber,
        EDFAConfig(noise_figure_db=case.edfa_noise_figure_db),
        power,
        case.span_length_km * 1e3,
    )
    assert getattr(step, "nli_amplitude_spm", None) is None
    assert getattr(step, "nli_amplitude_xpm", None) is None


def test_kernel_amplitudes_match_the_span_noise_power() -> None:
    """核给出的复振幅必须满足 ``|amp|² == 该跨的噪声功率``（契约要求）。"""
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)
    fiber = SMF_100KM.model_copy(
        update={
            "length_km": case.span_length_km,
            "max_span_length_km": case.span_length_km,
        }
    )
    model = make_model("numerical", case)
    kernel = model.kernel_for(fiber)
    context = SpanContext(
        span_index=3,
        total_spans=10,
        cumulative_length_m=3 * case.span_length_km * 1e3,
        span_length_m=case.span_length_km * 1e3,
    )
    out = kernel.evaluate(power, case.span_length_km * 1e3, 1, span_context=context)

    assert out.nli_amplitude_spm is not None and out.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-12, atol=0.0)
    # 跨 3 的相位必须已经转起来（第一跨相位为 0）。
    assert float(np.angle(out.nli_amplitude_spm[0])) != 0.0


def test_coherent_accumulation_is_deterministic() -> None:
    """同一实例、同功率连调三次必须逐位相同：跨序号只能来自 SpanContext。"""
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)
    fiber = SMF_100KM.model_copy(
        update={
            "length_km": case.span_length_km,
            "max_span_length_km": case.span_length_km,
        }
    )
    path = MultiSpanOpticalPath(
        [fiber] * 10,
        [EDFAConfig(noise_figure_db=5.5)] * 10,
    )
    model = make_model("numerical", case)

    first = model.evaluate_path(path, power)
    # 先用别的功率与别的跨数打乱调用历史。
    other = MultiSpanOpticalPath([fiber] * 3, [EDFAConfig(noise_figure_db=5.5)] * 3)
    model.evaluate_path(other, launch_profile(case, 3.0))
    second = model.evaluate_path(path, power)
    third = model.evaluate_path(path, power)

    for name in ("spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        lhs = np.asarray(getattr(first, name)).view(np.int64)
        rhs = np.asarray(getattr(second, name)).view(np.int64)
        again = np.asarray(getattr(third, name)).view(np.int64)
        assert np.array_equal(lhs, rhs), f"{name} changed after an unrelated call"
        assert np.array_equal(lhs, again), f"{name} changed between two identical calls"


def test_coherent_limits_zero_and_large_dispersion() -> None:
    """解析极限：色散趋零时完全相干（NLI ∝ N²），色散很大时非相干（NLI ∝ N）。"""
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)

    def _run(span_count: int, dispersion: float):
        fiber = SMF_100KM.model_copy(
            update={
                "length_km": case.span_length_km,
                "max_span_length_km": case.span_length_km,
                "dispersion_parameter_d": dispersion,
                "dispersion_slope_s": 0.0,
            }
        )
        path = MultiSpanOpticalPath(
            [fiber] * span_count,
            [EDFAConfig(noise_figure_db=5.5)] * span_count,
        )
        result = make_model("numerical", case).evaluate_path(path, power)
        return np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)

    for dispersion, expected, label in (
        (0.0, 100.0, "zero dispersion"),
        (1000.0, 10.0, "very large dispersion"),
    ):
        single = _run(1, dispersion)
        ten = _run(10, dispersion)
        ratio = ten / single
        print(f"{label}: ten-span / one-span NLI ratio {ratio[0]:.3f} (expect {expected:.0f})")
        assert np.all(np.isfinite(ratio))
        assert np.all(ratio > 0.9 * expected) and np.all(ratio < 1.1 * expected), (
            f"{label}: NLI growth {ratio} is not within 10 % of {expected}"
        )


def test_coherent_result_converges_with_resolution() -> None:
    """相干结果对求积点数与积分步数的敏感性。"""
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)
    fiber = SMF_100KM.model_copy(
        update={
            "length_km": case.span_length_km,
            "max_span_length_km": case.span_length_km,
        }
    )
    path = MultiSpanOpticalPath([fiber] * 10, [EDFAConfig(noise_figure_db=5.5)] * 10)

    def _run(**options):
        return make_model("numerical", case, **options).evaluate_path(path, power)

    coarse = _run(quadrature_points=64, ode_steps=64)
    fine = _run(quadrature_points=128, ode_steps=128)
    deviation = np.abs(np.asarray(fine.snr_db) - np.asarray(coarse.snr_db))
    print(f"max per-channel SNR deviation (64/64 -> 128/128): {float(np.max(deviation)):.5f} dB")
    assert float(np.max(deviation)) < 0.05, "the coherent result is not converged in the spatial/frequency resolution"


def test_matches_ecz_within_a_few_db() -> None:
    """cl_152ch 上，中间信道的 SNR 必须与闭式 ecz 基线相差不超过 3 dB。"""
    case = build_case("cl_152ch")
    path = build_path(case)
    power = launch_profile(case, None)
    mid = case.num_channels // 2

    reference = make_model("ecz", case).evaluate_path(path, power)
    numerical = make_model("numerical", case).evaluate_path(path, power)

    difference = float(np.asarray(numerical.snr_db)[mid] - np.asarray(reference.snr_db)[mid])
    print(
        f"cl_152ch mid-channel SNR: numerical {np.asarray(numerical.snr_db)[mid]:.3f} dB, "
        f"ecz {np.asarray(reference.snr_db)[mid]:.3f} dB, difference {difference:+.3f} dB"
    )
    assert abs(difference) < 3.0, f"mid-channel SNR differs from ecz by {difference:.3f} dB"


def test_egn_is_rejected() -> None:
    """egn=True 必须显式报错，而不是静默退回闭式模型。"""
    fiber = SMF_100KM.model_copy(update={"length_km": 100.0, "max_span_length_km": 100.0})
    grid = SpectrumGrid(4, 75e9, 193.4e12)
    with pytest.raises(ValueError, match="egn"):
        create_model("numerical", grid, fiber, symbol_rate_hz=64e9, egn=True)
