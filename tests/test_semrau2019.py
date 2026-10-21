"""Semrau 2019 闭式 ISRS-GN 模型的契约与物理行为测试。

覆盖 WP 要求的五项：零 ISRS 极限退化为经典 GN、EGN 修正的方向与量级、
多跨线性累积、宽带全信道 SNR 有限、单信道 XPM 归零；以及本轮改造带来的
跨间相干累加：单跨逐位一致、``|a|**2`` 与噪声功率一致、多跨相干增益、
色散两端的解析极限、以及确定性。

相干口径的说明见 :mod:`network_nlo_eval.models.semrau2019` 的模块文档。
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
    SpanContext,
    available_models,
    create_model,
    get_model_info,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.semrau2019 import (
    EXCESS_KURTOSIS,
    SEMRAU2019_MODEL_INFO,
    Semrau2019Model,
    _coherent_factor,
    _cross_beating_visibility,
    _equivalent_phase_ramp,
    _paper_visibility_equivalent,
    _self_beating_visibility,
)
from network_nlo_eval.network.elements import EDFAConfig

#: 论文 2019a 第三节 C 段声明的量级：SSFM 的 uniform 64-QAM 与（高斯假设的）
#: 闭式近似之间的 NLI 系数平均差 1.6 dB（6×100 km SMF，含/不含 ISRS 均是）。
PAPER_64QAM_GAP_DB = 1.6

#: 论文 2019b 声明的闭式修正自身的误差量级：与数值仿真的平均绝对误差
#: 0.45 dB，总 NLI 系数（Eq. (16)）与仿真的偏差 0.3 dB（SMF）。
PAPER_CLOSED_FORM_ERROR_DB = 0.45
PAPER_TOTAL_NLI_ERROR_DB = 0.3

#: 改造前（纯功率口径）单跨链路的逐位基准值。
#:
#: 这些数字来自改造前的实现（跨数因子 ``n^{1+ε}`` 在核内折算成"总量除以跨
#: 数"）。单跨时相干退化为单跨，路由层走原来的纯功率分支，结果必须逐位相同。
PRE_CHANGE_SINGLE_SPAN = {
    # (算例, egn): (SNR_dB, SPM_W, XPM_W)
    ("scaling_160ch", False): (27.482292565970468, 1.3561239821022055e-07, 3.623338915324885e-07),
    ("scaling_160ch", True): (27.967333352482399, 1.3561239821022055e-07, 1.754299924836465e-07),
    ("cl_152ch", False): (27.358672631615075, 7.9332559984377515e-08, 2.0965866287524476e-07),
    ("cl_152ch", True): (27.678878513729504, 7.9332559984377515e-08, 1.0150973594209766e-07),
}


def _fiber(length_km: float = 100.0, dispersion_d: float | None = None, slope_s: float | None = None):
    """基准光纤（SMF，100 km 单跨）。"""
    update: dict[str, float] = {"length_km": length_km, "max_span_length_km": length_km}
    if dispersion_d is not None:
        update["dispersion_parameter_d"] = dispersion_d
    if slope_s is not None:
        update["dispersion_slope_s"] = slope_s
    return SMF_100KM.model_copy(update=update)


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


def _nli(result) -> np.ndarray:
    """逐信道的链路级 NLI 功率（SPM + XPM）。"""
    return np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)


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


def test_model_declares_span_context_support() -> None:
    """模型声明了要跨间上下文，路由层才会传 ``span_context=``。"""
    assert Semrau2019Model.supports_span_context is True
    import inspect

    signature = inspect.signature(Semrau2019Model.span_step)
    assert "span_context" in signature.parameters
    assert signature.parameters["span_context"].default is None


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

    nli_gn = _nli(res_gn)
    nli_egn = _nli(res_egn)
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


def test_egn_cross_span_factor_stays_in_the_kernel() -> None:
    """2019b 的跨间结构（"单跨项只计一次"）仍在核内，跨间相干只乘一个因子。

    论文 Eq. (16) 在 ``N`` 跨上的总量是 ``N`` 份"逐跨等效值"之和。新口径下
    路由层真的相干累加，于是链路级总量变成论文总量的 ``R(N)/N`` 倍——
    相干因子只作用一次，EGN 的跨间结构原样保留。
    """
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    n_spans = 6
    model = make_model("semrau2019", case, egn=True)
    kernel = model.kernel_for(fiber)

    per_span = kernel.evaluate(power, 100e3, n_spans)
    paper_total = kernel.paper_eq16_total(power, 100e3, n_spans)
    link_total = kernel.link_level_total(power, 100e3, n_spans)

    # 论文总量 = N 份逐跨等效值（相干因子不在核内）。
    assert np.allclose(n_spans * per_span.spm_noise_w, paper_total.spm_noise_w, rtol=1e-12)
    assert np.allclose(n_spans * per_span.xpm_noise_w, paper_total.xpm_noise_w, rtol=1e-12)
    # 新的链路级总量是论文总量的 R(N)/N 倍。
    _, _, factor_spm, factor_xpm = kernel._coherence(n_spans, 100e3)
    assert np.allclose(link_total.spm_noise_w, paper_total.spm_noise_w * factor_spm / n_spans, rtol=1e-12)
    assert np.allclose(link_total.xpm_noise_w, paper_total.xpm_noise_w * factor_xpm / n_spans, rtol=1e-12)


# ---------------------------------------------------------------------------
# 多跨累积（相干口径）
# ---------------------------------------------------------------------------


def test_multi_span_scaling() -> None:
    """跨数加倍，NLI 接近加倍（5% 容差）。

    SMF（``D = 17 ps/(nm·km)``）、100 km 跨、64 GHz 信道的相干长度远小于跨距，
    跨间几乎完全非相干：相干因子 ``R(2) ≈ 2.04``，与线性累加只差 2%。
    """
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)

    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    two = model.evaluate_path(_path(2, case.edfa_noise_figure_db), power)

    nli_one = _nli(one)
    nli_two = _nli(two)
    ratio = nli_two / nli_one
    assert np.allclose(ratio, 2.0, rtol=0.05), f"NLI 跨数加倍比值 {ratio.min():.3f}..{ratio.max():.3f}"


def test_multi_span_scaling_at_kernel_level() -> None:
    """链路级总量按相干因子 ``R(N)`` 增长，而不是论文的 ``N``。

    ``R(N) = N + 2·sum (N-Delta)·V(Delta·L)``：完全非相干时是 ``N``，完全
    相干时是 ``N²``。SMF 上 ``V ≈ 0.018``，故 ``R(10) ≈ 10.7``。
    """
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    kernel = make_model("semrau2019", case).kernel_for(fiber)

    one = kernel.link_level_total(power, 100e3, 1)
    ten = kernel.link_level_total(power, 100e3, 10)
    ratio = ten.spm_noise_w / one.spm_noise_w
    assert np.all(ratio > 10.0), "强色散下相干因子必须略大于 N（相干增益）"
    assert np.all(ratio < 12.0), f"SMF 上不应接近完全相干，实测 {ratio.max():.3f}"

    # 相干因子与可见度公式一致。
    _, _, factor_spm, _ = kernel._coherence(10, 100e3)
    assert np.allclose(ratio, factor_spm, rtol=1e-12)


# ---------------------------------------------------------------------------
# 跨间相干累加（本轮改造）
# ---------------------------------------------------------------------------


def test_single_span_is_bit_identical_to_the_pre_change_model() -> None:
    """单跨链路必须与改造前逐位一致（硬编码数值断言）。

    相干累加在单跨时退化为单跨：路由层走纯功率分支，噪声与 SNR 的每一个
    比特都必须与改造前相同。基准值取自改造前的实现。
    """
    for (case_name, egn), (snr_db, spm_w, xpm_w) in PRE_CHANGE_SINGLE_SPAN.items():
        case = build_case(case_name)
        power = launch_profile(case, None)
        model = make_model("semrau2019", case, egn=egn)
        result = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
        idx = _mid(case)
        assert result.snr_db[idx] == snr_db, f"{case_name} egn={egn}: SNR 变了"
        assert result.spm_noise_w[idx] == spm_w, f"{case_name} egn={egn}: SPM 变了"
        assert result.xpm_noise_w[idx] == xpm_w, f"{case_name} egn={egn}: XPM 变了"


def test_single_span_uses_the_power_only_branch() -> None:
    """单跨时核不给出复振幅，路由层因此走原来的纯功率分支。"""
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)
    step = model.span_step(
        fiber,
        EDFAConfig(noise_figure_db=5.5),
        power,
        100e3,
        span_context=SpanContext(
            span_index=0, total_spans=1, cumulative_length_m=0.0, span_length_m=100e3
        ),
    )
    assert step.nli_amplitude_spm is None
    assert step.nli_amplitude_xpm is None


def test_amplitude_magnitude_equals_the_noise_power() -> None:
    """``|a|**2`` 必须逐位等于同名噪声功率（``rtol=1e-12``、``atol=0``）。"""
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)
    step = model.span_step(
        fiber,
        EDFAConfig(noise_figure_db=5.5),
        power,
        100e3,
        span_context=SpanContext(
            span_index=2, total_spans=10, cumulative_length_m=200e3, span_length_m=100e3
        ),
    )
    assert step.nli_amplitude_spm is not None
    assert step.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(step.nli_amplitude_spm) ** 2, step.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(step.nli_amplitude_xpm) ** 2, step.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_route_accumulation_matches_the_analytic_link_level_total() -> None:
    """路由层的相干累加必须逐位复现解析的链路级总量 ``R(N)·逐跨值``。"""
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    n_spans = 10
    model = make_model("semrau2019", case)
    result = model.evaluate_path(_path(n_spans, case.edfa_noise_figure_db), power)
    analytic = model.kernel_for(fiber).link_level_total(power, 100e3, n_spans)
    assert np.allclose(result.spm_noise_w, analytic.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(result.xpm_noise_w, analytic.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_multi_span_nli_is_finite_and_monotonic() -> None:
    """多跨 NLI 有限、非负，且随跨数单调不减。"""
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)
    previous = None
    for n_spans in (1, 2, 5, 10):
        result = model.evaluate_path(_path(n_spans, case.edfa_noise_figure_db), power)
        nli = _nli(result)
        assert np.all(np.isfinite(nli))
        assert np.all(nli >= 0.0)
        if previous is not None:
            assert np.all(nli >= previous - 1e-18), f"{n_spans} 跨的 NLI 比 {n_spans - 1} 跨小了"
        previous = nli


def test_ten_span_nli_is_not_ten_times_the_single_span() -> None:
    """10 跨的 NLI 不等于"单跨 × 10"：相干因子给出的是 ``R(10)`` 而不是 10。

    SMF 上相干长度远小于跨距，跨间几乎非相干，故 ``R(10) ≈ 10.69``：
    比值 10.69 而不是 10.00，即 0.29 dB 的相干增益。
    """
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)
    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    ten = model.evaluate_path(_path(10, case.edfa_noise_figure_db), power)

    ratio = _nli(ten) / _nli(one)
    assert np.all(ratio > 10.0), f"10 跨/1 跨比值 {ratio.min():.4f}..{ratio.max():.4f} 未超过 10"
    assert np.all(ratio < 11.0), f"SMF 上不应接近完全相干，实测 {ratio.max():.4f}"
    # 相干增益 = 相对于"非相干累加（比值恰好 N）"的额外 NLI。
    gain_db = 10 * np.log10(ratio / 10.0)
    assert np.all(gain_db > 0.0)
    assert np.all(gain_db < 1.0), f"相干增益 {gain_db.max():.3f} dB 不在 SMF 的量级内"


def test_zero_dispersion_is_fully_coherent() -> None:
    """色散趋零时跨间接近完全相干：NLI ∝ N²。"""
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    fiber = _fiber(dispersion_d=0.0, slope_s=0.0)
    model = make_model("semrau2019", case)
    kernel = model.kernel_for(fiber)
    path = MultiSpanOpticalPath(
        [fiber] * 10, [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)] * 10
    )
    result = model.evaluate_path(path, power)

    one = model.evaluate_path(
        MultiSpanOpticalPath(
            [fiber], [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)]
        ),
        power,
    )
    ratio = _nli(result) / _nli(one)
    assert np.all(ratio > 90.0), f"零色散下 10 跨/1 跨比值 {ratio.min():.2f} 未接近 100"

    _, _, factor_spm, factor_xpm = kernel._coherence(10, 100e3)
    assert np.allclose(factor_spm, 100.0, rtol=1e-6)
    assert np.allclose(factor_xpm, 100.0, rtol=1e-6)


def test_strong_dispersion_is_incoherent() -> None:
    """色散很强时跨间接近非相干：NLI ∝ N。"""
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    fiber = _fiber(dispersion_d=100.0, slope_s=0.0)
    model = make_model("semrau2019", case)
    kernel = model.kernel_for(fiber)
    edfa = EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)
    result = model.evaluate_path(MultiSpanOpticalPath([fiber] * 10, [edfa] * 10), power)
    one = model.evaluate_path(MultiSpanOpticalPath([fiber], [edfa]), power)

    ratio = _nli(result) / _nli(one)
    assert np.all(ratio > 10.0), f"强色散下 10 跨/1 跨比值 {ratio.min():.4f} 未超过 10"
    assert np.all(ratio < 10.2), f"强色散下应接近 N，实测 {ratio.max():.4f}"

    _, _, factor_spm, factor_xpm = kernel._coherence(10, 100e3)
    assert np.all(factor_spm < 10.2)
    assert np.all(factor_xpm < 10.2)


def test_phase_comes_only_from_the_span_context() -> None:
    """振幅的相位只由 ``SpanContext`` 决定，模型内部没有跨号计数器。

    同一个上下文重复调用必须给出逐位相同的振幅；换一个上下文（跨号不同、
    累计长度相同）必须给出同一个振幅——相位只跟累计长度走。
    """
    case = build_case("scaling_160ch")
    fiber = _fiber()
    power = launch_profile(case, None)
    kernel = make_model("semrau2019", case).kernel_for(fiber)

    context = SpanContext(
        span_index=3, total_spans=10, cumulative_length_m=300e3, span_length_m=100e3
    )
    first = kernel.evaluate(power, 100e3, 1, span_context=context)
    second = kernel.evaluate(power, 100e3, 1, span_context=context)
    assert np.array_equal(first.nli_amplitude_spm, second.nli_amplitude_spm)
    assert np.array_equal(first.nli_amplitude_xpm, second.nli_amplitude_xpm)

    # 跨号不同但累计长度相同 → 相位相同（相位只跟累计长度走）。
    other = kernel.evaluate(
        power,
        100e3,
        1,
        span_context=SpanContext(
            span_index=7, total_spans=10, cumulative_length_m=300e3, span_length_m=100e3
        ),
    )
    assert np.allclose(first.nli_amplitude_spm, other.nli_amplitude_spm, rtol=1e-15, atol=0.0)


def test_repeated_path_evaluations_are_bit_identical() -> None:
    """同一实例、同一条链路、同一功率连调三次必须逐位相同。"""
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    path = _path(case.span_count, case.edfa_noise_figure_db)
    model = make_model("semrau2019", case)
    first = model.evaluate_path(path, power)
    second = model.evaluate_path(path, power)
    third = model.evaluate_path(path, power)
    assert np.array_equal(first.snr_db, second.snr_db)
    assert np.array_equal(second.snr_db, third.snr_db)
    assert np.array_equal(first.spm_noise_w, third.spm_noise_w)
    assert np.array_equal(first.xpm_noise_w, third.xpm_noise_w)


def test_paper_factor_maps_to_an_equivalent_visibility() -> None:
    """论文的 ``n^{1+ε}`` 等价于常数可见度 ``v=(N^ε-1)/(N-1)`` 下的相干求和。

    相干因子 ``R = N·[1+v·(N-1)]``；令其等于 ``N^{1+ε}`` 即得映射关系。这是
    报告里"论文跨数因子 ↔ 物理可见度"的换算依据。
    """
    for n_spans in (2, 6, 10):
        for epsilon in (0.0, 0.25, 0.5, 1.0):
            visibility = _paper_visibility_equivalent(n_spans, epsilon)
            factor = n_spans * (1.0 + visibility * (n_spans - 1))
            assert factor == pytest.approx(n_spans ** (1.0 + epsilon), rel=1e-12)


def test_visibility_models_the_two_analytic_limits() -> None:
    """可见度公式的两个解析极限：无色散 → 1，强色散 → 0。"""
    bandwidth = np.array([64e9], dtype=np.float64)
    assert _self_beating_visibility(0.0, bandwidth, 100e3)[0] == pytest.approx(1.0)
    assert _self_beating_visibility(2.168e-26, bandwidth, 100e3)[0] < 0.05
    assert _cross_beating_visibility(0.0, bandwidth, 100e3, np.array([75e9]))[0] == pytest.approx(1.0)
    assert _cross_beating_visibility(2.168e-26, bandwidth, 100e3, np.array([75e9]))[0] < 0.05

    # 完全相干时相干因子是 N²，完全非相干时是 N。
    full = np.ones((4, 1), dtype=np.float64)
    assert _coherent_factor(5, full)[0] == pytest.approx(25.0)
    assert _coherent_factor(5, np.zeros((4, 1), dtype=np.float64))[0] == pytest.approx(5.0)
    # Dirichlet 反解：相位步长 0 → N²。
    assert _equivalent_phase_ramp(5, np.array([25.0]))[0] == pytest.approx(0.0, abs=1e-9)


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


def test_coherent_accumulation_does_not_regress_the_latency() -> None:
    """相干累加不得把热路径拖慢一个量级。

    基准（本机实测，见交付报告）：单跨 0.08 ms、152 信道 10 跨 0.38 ms。
    这里用绝对上限（单跨 1 ms、10 跨 2 ms）加上一条**与机器无关**的相对
    守卫：多跨的每跨开销不得显著高于单跨。
    """
    import time

    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    model = make_model("semrau2019", case)

    def median_ms(path) -> float:
        model.evaluate_path(path, power)
        samples = []
        for _ in range(15):
            started = time.perf_counter()
            model.evaluate_path(path, power)
            samples.append((time.perf_counter() - started) * 1e3)
        return float(np.median(samples))

    single = median_ms(_path(1, case.edfa_noise_figure_db))
    multi = median_ms(_path(10, case.edfa_noise_figure_db))

    assert single < 1.0, f"单跨热调用 {single:.3f} ms 超标"
    assert multi < 2.0, f"10 跨热调用 {multi:.3f} ms 超标"
    assert multi / 10.0 < 2.0 * single, f"每跨开销 {multi / 10.0:.3f} ms 明显高于单跨 {single:.3f} ms"
