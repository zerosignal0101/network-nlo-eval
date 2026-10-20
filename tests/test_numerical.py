"""数值 ISRS-GN 参考模型的验收测试。

模型为 Semrau, Killey, Bayvel, JLT 2018 (doi:10.1109/jlt.2018.2830973) 的
数值实现：Eq. (6) 的耦合 SRS 功率剖面用常步长 RK4 数值积分，Eq. (4) 的 GN
双重积分用频率四边形求积。本文件覆盖以下验收点：

* 单信道的 GN 自相位调制闭式解；
* 频率求积的收敛性；
* 零色散下的数值有限性；
* 多跨���的非线性干涉累加；
* 与闭式 ``ecz`` 基线的逐信道 SNR 一致性；
* ``egn=True`` 被显式拒绝。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, build_case, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
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
    """十跨链路 SNR 必须低于单跨链路，且非线性噪声随跨数近似线性增长。"""
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
    assert np.all(ratio > 9.5) and np.all(ratio < 10.5), f"NLI growth {ratio} is not linear in the span count"


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
