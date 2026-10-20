"""Semrau 2019 闭式 ISRS-GN 模型的契约与物理行为测试。

覆盖 WP 要求的五项：零 ISRS 极限退化为经典 GN、EGN 修正的方向与量级、
多跨线性累积、宽带全信道 SNR 有限、单信道 XPM 归零。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    build_case,
    grid_for,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    available_models,
    create_model,
    get_model_info,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.semrau2019 import (
    EXCESS_KURTOSIS,
    SEMRAU2019_MODEL_INFO,
    Semrau2019Model,
)
from network_nlo_eval.network.elements import EDFAConfig

#: 论文 2019a 第三节 C 段声明的量级：SSFM 的 uniform 64-QAM 与（高斯假设的）
#: 闭式近似之间的 NLI 系数平均差 1.6 dB（6×100 km SMF，含/不含 ISRS 均是）。
PAPER_64QAM_GAP_DB = 1.6

#: 论文 2019b 声明的闭式修正自身的误差量级：与数值仿真的平均绝对误差
#: 0.45 dB，总 NLI 系数（Eq. (16)）与仿真的偏差 0.3 dB（SMF）。
PAPER_CLOSED_FORM_ERROR_DB = 0.45
PAPER_TOTAL_NLI_ERROR_DB = 0.3


def _fiber(length_km: float = 100.0):
    """基准光纤（SMF，100 km 单跨）。"""
    return SMF_100KM.model_copy(
        update={"length_km": length_km, "max_span_length_km": length_km},
    )


def _path(n_spans: int, noise_figure_db: float = 5.5) -> MultiSpanOpticalPath:
    """N 个等长跨段 + 等 NF 的 EDFA（与基准算例同构）。"""
    fiber = _fiber()
    edfa = EDFAConfig(noise_figure_db=noise_figure_db)
    return MultiSpanOpticalPath([fiber] * n_spans, [edfa] * n_spans)


def _mid(case) -> int:
    return case.num_channels // 2


def _alpha_npm(model, case) -> np.ndarray:
    """模型核里的逐信道功率衰减 (Np/m)。"""
    return np.asarray(model.kernel_for(_fiber()).alpha_npm, dtype=np.float64)


# ---------------------------------------------------------------------------
# 注册与契约
# ---------------------------------------------------------------------------


def test_model_is_registered() -> None:
    """模型进了注册表，ModelInfo 字段齐全。"""
    assert "semrau2019" in available_models()
    info = get_model_info("semrau2019")
    assert info is SEMRAU2019_MODEL_INFO
    assert info.family == "gn+egn"
    assert "2895237" in info.reference
    assert "2929461" in info.reference
    assert any(option[0] == "egn" for option in info.options)


def test_model_constructible_from_registry() -> None:
    """能按注册表构造，节点几何与基准算例一致。"""
    case = build_case("scaling_160ch")
    model = create_model("semrau2019", grid_for(case), _fiber(), symbol_rate_hz=case.symbol_rate_hz)
    assert isinstance(model, Semrau2019Model)
    assert model.n_nodes == case.num_channels
    assert model.channel_bandwidth_hz_channels.shape == (case.num_channels,)
    assert np.all(model.channel_bandwidth_hz_channels > 0)
    assert model.power_width_scale().shape == (case.num_channels,)


# ---------------------------------------------------------------------------
# 零 ISRS 极限：退化为经典 GN
# ---------------------------------------------------------------------------


def test_zero_isrs_limit_matches_gaussian_gn() -> None:
    """``C_r=0`` 时 SPM/XPM 与冻结基线 ``ecz`` 逐项一致（相对差 < 1e-6）。

    这是"主项没写错"的证据：Semrau 2019 Eq. (10)/(11) 在 ``ᾱ=α``、``A=2α``、
    ``T=4α²`` 下应当与经典 GN 的闭式逐项相同。

    两侧都把拉曼增益斜率置 0。``ecz`` 的出纤功率用不到本测试（它内部的功率
    剖面用的是模块级常数），故只比较 SPM/XPM 噪声。
    """
    case = build_case("scaling_160ch")
    grid = grid_for(case)
    fiber = _fiber()
    power = launch_profile(case, None)
    span_length_m = 100e3

    semrau = make_model("semrau2019", case, srs_cr_w_m_hz=0.0)
    semrau_out = semrau.kernel_for(fiber).evaluate(power, span_length_m, 1)

    ecz = MultiBandISRSGN(grid, fiber, case.symbol_rate_hz, ode_accel="off")
    ecz.cr_w_m_hz = 0.0
    ecz_out = ecz.span_fiber_output(power, span_length_m, 1)

    for name, mine, theirs in (
        ("SPM", semrau_out.spm_noise_w, ecz_out.spm_noise_w),
        ("XPM", semrau_out.xpm_noise_w, ecz_out.xpm_noise_w),
    ):
        assert np.all(theirs > 0.0)
        rel = np.abs(mine - theirs) / theirs
        assert rel.max() < 1e-6, f"{name} 相对差 {rel.max():.3e} 超过 1e-6"


def test_zero_isrs_turns_off_the_tilt() -> None:
    """``C_r=0`` 时出纤功率只剩光纤损耗，没有拉曼倾斜。

    SMF 的衰减是波长相关的（二次多项式），所以各信道的出纤功率本来就不相等；
    这里比的是"逐信道的 e^{-α_k L}"，而不是一个常数。
    """
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    model = make_model("semrau2019", case, srs_cr_w_m_hz=0.0)
    out = model.kernel_for(fiber).evaluate(power, 100e3, 1)
    alpha = _alpha_npm(model, case)
    expected = power * np.exp(-alpha * 100e3)
    assert np.allclose(out.power_out_w, expected, rtol=1e-12, atol=0.0)


def test_isrs_tilt_follows_the_paper_magnitude() -> None:
    """ISRS 倾斜的量级与论文 Eq. (3) 的 Δρ 一致。

    论文 Eq. (3)：``Δρ(z)[dB] = 4.3·P_tot·C_r·L_eff·B_tot``（频带两端的净功率
    转移）。本模型的功率剖面做总功率守恒归一化，倾斜幅度应略小于该值但同量级。
    """
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)
    out = model.kernel_for(fiber).evaluate(power, 100e3, 1)

    spread_db = 10 * np.log10(out.power_out_w[0] / out.power_out_w[-1])
    assert spread_db > 1.0, "低频端相对高频端应有可见增益"
    assert spread_db < 12.0, f"倾斜 {spread_db:.2f} dB 超出一致性范围"

    alpha = 0.2 * np.log(10) / 10 / 1e3
    l_eff = (1 - np.exp(-alpha * 100e3)) / alpha
    p_total = float(power.sum())
    cr = 0.028 / 1e3 / 1e12
    b_tot = abs(float(model.f_rel_hz[-1] - model.f_rel_hz[0])) + case.channel_spacing_hz
    predicted_db = 4.3 * p_total * cr * l_eff * b_tot
    assert 0.5 * predicted_db < spread_db < 1.5 * predicted_db


# ---------------------------------------------------------------------------
# EGN（2019b 调制格式修正）
# ---------------------------------------------------------------------------


def test_egn_xpm_ratio_follows_the_paper_formula() -> None:
    """单跨时 XPM 噪声必须精确按 ``1 + 5Φ/6`` 缩放（2019b Eq. (16)）。

    Eq. (16) 的 XPM 项带 ``(n + 5Φ/6)``；单跨时 n=1，XPM 系数为
    ``1 + 5Φ/6``。Φ=−0.619（uniform 64-QAM）时该系数为 0.4842。
    """
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    kernel_gn = make_model("semrau2019", case).kernel_for(fiber)
    kernel_egn = make_model("semrau2019", case, egn=True).kernel_for(fiber)

    out_gn = kernel_gn.evaluate(power, 100e3, 1)
    out_egn = kernel_egn.evaluate(power, 100e3, 1)

    expected = 1.0 + (5.0 / 6.0) * EXCESS_KURTOSIS["64qam"]
    ratio = out_egn.xpm_noise_w / out_gn.xpm_noise_w
    assert np.allclose(ratio, expected, rtol=1e-12)
    # SPM 不在 2019b 的修正范围内（论文只修正 XPM/XCI 项）。
    assert np.allclose(out_egn.spm_noise_w, out_gn.spm_noise_w, rtol=1e-15)


def test_egn_changes_result_in_expected_direction() -> None:
    """EGN 档把 NLI 拉低、SNR 抬高，且量级落在论文声明的范围内。

    论文 2019a 第三节 C 段：SSFM 的 uniform 64-QAM 与高斯假设闭式之间的 NLI
    系数平均差 **1.6 dB**（6×100 km SMF）。2019b 的修正只覆盖 XPM 项，故
    修正量应显著但不越过这个总缺口。本测试用论文自己的 6 跨配置，取中频信道：

    * 方向：NLI 下降、SNR 上升（2019b：多数格式 Φ<0 ⇒ 修正为负 ⇒ NLI 减小）。
    * 量级：NLI 下降幅度落在 [0.5, 1.6] dB 内（1.6 dB 是论文声明的总缺口上界，
      0.5 dB 保证修正确实吃到了缺口的大部分）。
    * SNR 改善为正且不超过 NLI 下降幅度（总噪声里还有 ASE，只能稀释收益）。
    """
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    n_spans = 6
    path = _path(n_spans, case.edfa_noise_figure_db)

    model_gn = make_model("semrau2019", case)
    model_egn = make_model("semrau2019", case, egn=True)
    res_gn = model_gn.evaluate_path(path, power)
    res_egn = model_egn.evaluate_path(path, power)

    nli_gn = res_gn.spm_noise_w + res_gn.xpm_noise_w
    nli_egn = res_egn.spm_noise_w + res_egn.xpm_noise_w
    assert np.all(nli_egn < nli_gn), "EGN 档必须降低 NLI"

    idx = _mid(case)
    reduction_db = 10 * np.log10(nli_gn[idx] / nli_egn[idx])
    assert 0.5 <= reduction_db <= PAPER_64QAM_GAP_DB, (
        f"NLI 下降 {reduction_db:.3f} dB 不在论文声明的 {PAPER_64QAM_GAP_DB} dB 尺度内"
    )

    snr_gain_db = res_egn.snr_db[idx] - res_gn.snr_db[idx]
    assert snr_gain_db > 0.0, "EGN 档必须提高 SNR"
    assert snr_gain_db <= reduction_db + 1e-9, "SNR 改善不应超过 NLI 下降幅度"

    # 修正后的 NLI 系数应落在论文声明的闭式模型误差带内（1.6 dB 缺口之外还剩
    # 约 0.45 dB 的闭式误差），故剩余缺口不应超过 1.6 dB。
    residual_db = PAPER_64QAM_GAP_DB - reduction_db
    assert residual_db <= PAPER_CLOSED_FORM_ERROR_DB + PAPER_TOTAL_NLI_ERROR_DB


def test_egn_matches_the_paper_equation_16_total() -> None:
    """路由层逐跨累加的结果必须精确等于论文 Eq. (16) 的 n 跨总量。"""
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    model = make_model("semrau2019", case, egn=True)

    kernel = model.kernel_for(fiber)
    total = kernel.evaluate(power, 100e3, 6)
    result = model.evaluate_path(_path(6, case.edfa_noise_figure_db), power)

    assert np.allclose(result.spm_noise_w, total.spm_noise_w, rtol=1e-12)
    assert np.allclose(result.xpm_noise_w, total.xpm_noise_w, rtol=1e-12)


# ---------------------------------------------------------------------------
# 多跨累积
# ---------------------------------------------------------------------------


def test_multi_span_scaling() -> None:
    """跨数加倍，NLI 近似加倍（5% 容差）。

    论文 Eq. (5) 在非相干近似（ε=0，本模型的默认）下给出 NLI ∝ n：SPM 项
    随 n^{1+ε} 累加、XPM 项随 n 累加。
    """
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)

    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    two = model.evaluate_path(_path(2, case.edfa_noise_figure_db), power)

    nli_one = one.spm_noise_w + one.xpm_noise_w
    nli_two = two.spm_noise_w + two.xpm_noise_w
    ratio = nli_two / nli_one
    assert np.allclose(ratio, 2.0, rtol=0.05), f"NLI 跨数加倍比值 {ratio.min():.3f}..{ratio.max():.3f}"


def test_multi_span_scaling_at_kernel_level() -> None:
    """单跨核的 n 跨口径同样线性（Eq. (16) 的 n 因子）。"""
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    kernel = make_model("semrau2019", case).kernel_for(fiber)

    one = kernel.evaluate(power, 100e3, 1)
    two = kernel.evaluate(power, 100e3, 2)
    assert np.allclose(two.spm_noise_w / one.spm_noise_w, 2.0, rtol=1e-12)
    assert np.allclose(two.xpm_noise_w / one.xpm_noise_w, 2.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 数值健壮性
# ---------------------------------------------------------------------------


def test_finite_on_wide_band() -> None:
    """escl_479ch 单跨：全信道 SNR 有限，噪声有限非负。

    **适用边界（重要）**：E+S+C+L 是 35.9 THz，超出论文 2019a/2019b 明确的
    适用带宽（"applicable ... for optical bandwidths of up to 15 THz, as the
    formula relies on a triangular Raman gain spectrum"）。论文的线性拉曼增益
    近似在这么宽的频带上会给出很大的功率转移——按论文自己的 Eq. (3)，
    ``Δρ = 4.3·P_tot·C_r·L_eff·B_tot`` 在本算例约 29.5 dB——于是 E 波段的
    信道在纤内被压得很低，放大器要补更多增益，ASE 随之变大，SNR 在频带边缘
    落到 0 以下。这是模型的**适用域**问题，不是数值发散。

    因此本测试断言：
    * 全信道 SNR 有限，噪声有限且非负（数值不发散）；
    * 在论文的 15 THz 适用带宽内，SNR 严格为正。
    * 另有一个落在适用域内的宽带用例（cl_152ch，11.3 THz）做全信道正 SNR 断言。
    """
    case = build_case("escl_479ch_1span")
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)
    result = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)

    assert result.snr_db.shape == (case.num_channels,)
    assert np.all(np.isfinite(result.snr_db)), "存在非有限 SNR"
    assert np.all(np.isfinite(result.spm_noise_w))
    assert np.all(np.isfinite(result.xpm_noise_w))
    assert np.all(np.isfinite(result.ase_noise_w))
    assert np.all(result.spm_noise_w >= 0.0)
    assert np.all(result.xpm_noise_w >= 0.0)
    assert np.all(result.signal_power_w > 0.0)

    # 论文适用带宽内（以频带中心为原点 ±7.5 THz）SNR 必须为正。
    in_range = np.abs(model.f_rel_hz) <= 7.5e12
    assert in_range.sum() > 0
    assert np.all(result.snr_db[in_range] > 0.0), "论文适用带宽内出现非正 SNR"


def test_finite_on_wide_band_within_paper_validity() -> None:
    """C+L 宽带（11.3 THz，落在论文 15 THz 适用域内）全信道 SNR 有限且为正。"""
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)
    result = model.evaluate_path(_path(case.span_count, case.edfa_noise_figure_db), power)

    assert np.all(np.isfinite(result.snr_db))
    assert np.all(result.snr_db > 0.0)
    assert np.all(result.spm_noise_w >= 0.0)
    assert np.all(result.xpm_noise_w >= 0.0)


def test_single_channel() -> None:
    """单信道时 XPM 为 0、SPM 有限为正。"""
    case = build_case("scaling_160ch")
    grid = grid_for(case)
    fiber = _fiber()
    model = Semrau2019Model(grid, fiber, symbol_rate_hz=case.symbol_rate_hz)
    power = np.full(case.num_channels, 1e-3, dtype=np.float64)
    power[:] = 0.0
    power[case.num_channels // 2] = 1e-3

    out = model.kernel_for(fiber).evaluate(power, 100e3, 1)
    idx = case.num_channels // 2
    assert out.xpm_noise_w[idx] == 0.0
    assert np.isfinite(out.spm_noise_w[idx]) and out.spm_noise_w[idx] > 0.0
    # 未占用的信道不产生噪声。
    assert np.all(out.spm_noise_w[:idx] == 0.0)
    assert np.all(out.xpm_noise_w[:idx] == 0.0)


def test_unoccupied_channels_are_silent() -> None:
    """功率为 0 的信道不出噪声、不出功率。"""
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = np.zeros(case.num_channels, dtype=np.float64)
    power[10] = 1e-3
    out = make_model("semrau2019", case).kernel_for(fiber).evaluate(power, 100e3, 1)
    assert out.power_out_w[0] == 0.0
    assert out.spm_noise_w[0] == 0.0
    assert out.xpm_noise_w[0] == 0.0
    assert out.power_out_w[10] > 0.0


def test_kernel_rejects_bad_shapes() -> None:
    """功率向量长度必须与通道数一致。"""
    case = build_case("scaling_160ch")
    fiber = _fiber()
    kernel = make_model("semrau2019", case).kernel_for(fiber)
    with pytest.raises(ValueError):
        kernel.evaluate(np.ones(3, dtype=np.float64), 100e3, 1)
    with pytest.raises(ValueError):
        kernel.evaluate(np.ones(case.num_channels, dtype=np.float64), 0.0, 1)


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------


def test_single_span_is_faster_than_ecz() -> None:
    """160 信道单跨的闭式热路径必须快于冻结基线 ``ecz``。

    理由：本模型把 (N,N) 的逐信道对常量在核构造期算好，热路径只剩矩阵-向量乘；
    ``ecz`` 每次调用都要对每个信道对重算 γ、φ、asinh、atan。
    """
    import time

    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    path = _path(1, case.edfa_noise_figure_db)

    timings = {}
    for name in ("semrau2019", "ecz"):
        model = make_model(name, case)
        model.evaluate_path(path, power)  # 预热（JIT 与常量矩阵）
        samples = []
        for _ in range(15):
            started = time.perf_counter()
            model.evaluate_path(path, power)
            samples.append((time.perf_counter() - started) * 1e3)
        timings[name] = float(np.median(samples))

    assert timings["semrau2019"] < 2.0, f"单跨热调用 {timings['semrau2019']:.3f} ms 超过 2 ms"
    assert timings["semrau2019"] < timings["ecz"], (
        f"semrau2019 {timings['semrau2019']:.3f} ms 不快于 ecz {timings['ecz']:.3f} ms"
    )
