"""跨间相干口径的回归测试（buglia2023）。

本文件锁定"给三个纯闭式模型补上跨间相干"这次改造对 ``buglia2023`` 的结论：

* **单跨逐位不变**：``total_spans == 1`` 时核走纯功率分支，返回值与改造前
  逐位一致（硬编码的 IEEE-754 位模式守卫）；
* **多跨按 NLI 噪声功率加权**：可见度 ``V(d)`` 按闭式 GN 核的权重
  ``|1 - e^{-(alpha + j rho) L}|^2 / (alpha^2 + rho^2)`` 加权，而不是按信号谱
  均匀平均；相干因子 ``R(N) = N + 2 sum_k (N-k) V(kL)``、buildup ``= R(N)/N``；
* **量级落在区间内**：``c_57ch``、10 跨、0 dBm 的 SPM buildup 落在
  1.40 ± 0.03（``numerical`` 1.408，独立求积 1.402，``semrau2019`` 1.392）；
* **零色散完全相干**：色散归零时 ``V = 1``，buildup 退化为跨数 N；
* **XPM 的权重效应在噪声地板以下**：GN 加权与均匀带平均两种口径给出的 XPM
  相干增益差 < 0.05 dB，因此保留 GN 加权口径（与 SPM 同一套权重）。

参考量（``numerical`` 的 1.408）在 ``tests/test_numerical_visibility.py`` 里
有它自己的独立实现；本文件不 import 数值模型的私有函数，避免把两个模型耦死。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    build_case,
    build_path,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models import buglia2023 as buglia
from network_nlo_eval.models.base import MultiSpanOpticalPath, SpanContext

#: 审计算例：C 波段 57 信道 ×75 GHz、64 GBd、10×100 km SMF。
CASE_NAME = "c_57ch"
SPAN_M = 100.0e3
N_SPANS = 10

#: 硬编码的**改造前**单跨黄金值（IEEE-754 位模式）。
#:
#: 取自 ``c_57ch``、0 dBm/信道、1 跨 100 km SMF 链路、5.5 dB EDFA 的路径级
#: 结果。改造只加跨间相干，单跨必须逐位不变，因此这些位模式是守卫：任何
#: 浮点求和顺序的改动都会在这里暴露。
SINGLE_SPAN_GOLDEN_BITS = {
    "spm_noise_w": 4504929545509494897,
    "xpm_noise_w": 4506361383818147730,
    "ase_noise_w": 4517642954500968234,
    "snr_db": 4628714710883576385,
}

#: 同一组黄金值的可读形式（供失败信息用）。
SINGLE_SPAN_GOLDEN_VALUES = {
    "spm_noise_w": 1.5441192957838094e-07,
    "xpm_noise_w": 1.9231237403368787e-07,
    "ase_noise_w": 1.0664415593325432e-06,
    "snr_db": 28.498068621187716,
}


def _case():
    """审计算例。"""
    return build_case(CASE_NAME)


def _model(case=None, fiber=None):
    """按算例构造模型（``fiber`` 可覆盖光纤参数）。"""
    case = case or _case()
    return make_model("buglia2023", case)


def _path(case, spans: int, fiber=None):
    """均匀跨段路径。"""
    path = build_path(case)
    fiber = fiber or path.fiber_configs[0]
    edfa = path.edfa_configs[0]
    return MultiSpanOpticalPath([fiber] * spans, [edfa] * spans)


def _buildup(case, spans: int, power_dbm: float = 0.0) -> tuple[float, float, float]:
    """(SPM buildup, XPM buildup, SNR of channel 0)。

    buildup 定义为"链路级 NLI ÷ (跨数 × 单跨 NLI)"：纯功率相加时恒为 1，
    完全相干时等于跨数 N。
    """
    model = _model(case)
    power = launch_profile(case, power_dbm)
    result = model.evaluate_path(_path(case, spans), power)
    single = model.evaluate_path(_path(case, 1), power)
    spm = float(result.spm_noise_w[0]) / (spans * float(single.spm_noise_w[0]))
    xpm = float(result.xpm_noise_w[0]) / (spans * float(single.xpm_noise_w[0]))
    return spm, xpm, float(result.snr_db[0])


# =============================================================================
# 契约
# =============================================================================
def test_model_declares_span_context_support() -> None:
    """模型必须声明支持 ``SpanContext``，否则路由层不会把跨号传下来。"""
    assert buglia.Buglia2023Model.supports_span_context is True


def test_single_span_keeps_the_pre_change_golden_bits() -> None:
    """单跨链路必须与改造前**逐位**一致（硬编码位模式守卫）。"""
    case = _case()
    power = launch_profile(case, 0.0)
    result = _model(case).evaluate_path(_path(case, 1), power)

    for name, expected_bits in SINGLE_SPAN_GOLDEN_BITS.items():
        actual = float(np.asarray(getattr(result, name))[0])
        actual_bits = int(np.float64(actual).view(np.int64))
        assert actual_bits == expected_bits, (
            f"{name} drifted from the pre-change golden: "
            f"{actual!r} != {SINGLE_SPAN_GOLDEN_VALUES[name]!r} "
            f"(bits {actual_bits} != {expected_bits})"
        )


def test_single_span_kernel_takes_the_power_only_branch() -> None:
    """单跨时核不返回复振幅，路由层因此保持纯功率口径。"""
    case = _case()
    model = _model(case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    power = launch_profile(case, 0.0)

    without_context = kernel.evaluate(power, SPAN_M, 1)
    assert without_context.nli_amplitude_spm is None
    assert without_context.nli_amplitude_xpm is None

    context = SpanContext(span_index=0, total_spans=1, cumulative_length_m=0.0, span_length_m=SPAN_M)
    with_context = kernel.evaluate(power, SPAN_M, 1, span_context=context)
    assert with_context.nli_amplitude_spm is None
    assert with_context.nli_amplitude_xpm is None
    # 纯功率分支的数值也必须逐位不变。
    for name in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        assert np.array_equal(
            np.asarray(getattr(without_context, name)).view(np.int64),
            np.asarray(getattr(with_context, name)).view(np.int64),
        )


# =============================================================================
# 相干因子
# =============================================================================
def test_cross_span_spm_buildup_in_the_expected_window() -> None:
    """c_57ch / 10 跨 / 0 dBm：SPM buildup 必须落在 1.40 ± 0.03。"""
    spm, _, _ = _buildup(_case(), N_SPANS)
    assert 1.40 - 0.03 <= spm <= 1.40 + 0.03, f"SPM buildup {spm:.5f} is outside 1.40 +/- 0.03"


def test_buildup_grows_with_the_span_count() -> None:
    """相干 buildup 随跨数单调上升：相干增益是"部分相干"的插值。"""
    case = _case()
    values = [_buildup(case, spans)[0] for spans in (1, 2, 5, 10)]
    assert values[0] == pytest.approx(1.0, abs=1e-12)
    assert all(b > a for a, b in zip(values, values[1:], strict=False)), values


def test_zero_dispersion_gives_full_coherence() -> None:
    """色散归零时两份拷贝完全相干，buildup 退化为跨数 N。"""
    case = _case()
    fiber = build_path(case).fiber_configs[0].model_copy(
        update={"dispersion_parameter_d": 1e-9, "dispersion_slope_s": 1e-9}
    )
    model = make_model("buglia2023", case)
    # 模型的色散取自**参考光纤**，参考光纤也要一起换掉。
    model = type(model)(model.grid, fiber, symbol_rate_hz=case.symbol_rate_hz)
    power = launch_profile(case, 0.0)

    result = model.evaluate_path(_path(case, N_SPANS, fiber=fiber), power)
    single = model.evaluate_path(_path(case, 1, fiber=fiber), power)
    buildup = float(result.spm_noise_w[0]) / (N_SPANS * float(single.spm_noise_w[0]))
    assert buildup == pytest.approx(float(N_SPANS), rel=1e-9), f"zero-dispersion buildup {buildup}"


def test_dispersion_slope_alone_does_not_remove_the_coherence() -> None:
    r"""只把色散**斜率**调到 0 不等于零色散：``beta_2`` 来自 D 参数。

    这条测试把两个容易混淆的口径钉死：``dispersion_slope_s`` 只进入
    :math:`\beta_3`，而跨间相干的失配率由 :math:`|\beta_2|` 决定。
    """
    case = _case()
    fiber = build_path(case).fiber_configs[0].model_copy(update={"dispersion_slope_s": 1e-9})
    model = make_model("buglia2023", case)
    model = type(model)(model.grid, fiber, symbol_rate_hz=case.symbol_rate_hz)
    power = launch_profile(case, 0.0)

    result = model.evaluate_path(_path(case, N_SPANS, fiber=fiber), power)
    single = model.evaluate_path(_path(case, 1, fiber=fiber), power)
    buildup = float(result.spm_noise_w[0]) / (N_SPANS * float(single.spm_noise_w[0]))
    assert buildup == pytest.approx(1.41, abs=0.03), (
        f"slope-only zeroing changed the buildup to {buildup}; beta_2 comes from D, not S"
    )


def test_route_level_total_matches_the_analytic_coherent_factor() -> None:
    r"""路由层给出的总 NLI 必须等于"单跨值 × 相干因子 R(N)"。

    这是对"等效相位斜坡"反解正确性的端到端检查：斜坡的二分反解让
    :math:`[\sin(N\psi/2)/\sin(\psi/2)]^2` 精确等于 :math:`R(N)`。
    """
    case = _case()
    model = _model(case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    power = launch_profile(case, 0.0)

    distances = np.arange(1, N_SPANS, dtype=np.float64) * SPAN_M
    visibility = kernel._spm_visibility(distances, SPAN_M)
    coherent_factor = buglia._coherent_factor(N_SPANS, visibility)

    result = model.evaluate_path(_path(case, N_SPANS), power)
    single = model.evaluate_path(_path(case, 1), power)
    expected = coherent_factor * float(single.spm_noise_w[0])
    assert float(result.spm_noise_w[0]) == pytest.approx(expected, rel=1e-9)


# =============================================================================
# 振幅契约
# =============================================================================
def test_amplitude_modulus_squared_equals_the_noise_power() -> None:
    """``|amp|**2`` 必须精确等于同名噪声功率（``rtol=1e-12``, ``atol=0``）。"""
    case = _case()
    model = _model(case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    power = launch_profile(case, 0.0)

    context = SpanContext(
        span_index=3, total_spans=N_SPANS, cumulative_length_m=3.0 * SPAN_M, span_length_m=SPAN_M
    )
    out = kernel.evaluate(power, SPAN_M, 1, span_context=context)
    assert out.nli_amplitude_spm is not None
    assert out.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_amplitudes_survive_the_amplifier_and_the_roadm() -> None:
    """经放大器与 ROADM 之后，``|amp|**2`` 仍等于同名噪声功率。"""
    from network_nlo_eval.models.route import apply_amplifier_and_roadm
    from network_nlo_eval.network.elements import ROADMConfig

    case = _case()
    model = _model(case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    power = launch_profile(case, 0.0)
    context = SpanContext(
        span_index=1, total_spans=N_SPANS, cumulative_length_m=SPAN_M, span_length_m=SPAN_M
    )
    fiber_out = kernel.evaluate(power, SPAN_M, 1, span_context=context)

    for roadm in (None, ROADMConfig(loss_db=6.0, filtering_penalty_db=0.5)):
        out = apply_amplifier_and_roadm(
            power_in_w=power,
            fiber_output=fiber_out,
            edfa_config=build_path(case).edfa_configs[0],
            roadm_config=roadm,
            nf_lin=np.full(model.n_nodes, 10 ** (5.5 / 10)),
            f_abs_hz=model.f_abs_hz,
            bandwidth_hz=model.channel_bandwidth_hz_channels,
        )
        assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-12, atol=0.0)
        assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_results_are_deterministic() -> None:
    """同一输入必须给出逐位相同的结果（可见度求积没有随机性）。"""
    case = _case()
    power = launch_profile(case, 0.0)
    first = _model(case).evaluate_path(_path(case, N_SPANS), power)
    second = _model(case).evaluate_path(_path(case, N_SPANS), power)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(
            np.asarray(getattr(first, name)).view(np.int64),
            np.asarray(getattr(second, name)).view(np.int64),
        ), name


# =============================================================================
# 跨号来源
# =============================================================================
def test_span_phase_comes_from_the_span_context_only() -> None:
    """核不记跨号：同一个 context 的结果与调用顺序无关。

    跨号与累计长度只从 :class:`SpanContext` 取。若核自己维护了一个跨计数器，
    打乱调用顺序就会给出不同的结果——这条测试就是防这个。
    """
    case = _case()
    model = _model(case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    power = launch_profile(case, 0.0)

    contexts = [
        SpanContext(
            span_index=index,
            total_spans=N_SPANS,
            cumulative_length_m=index * SPAN_M,
            span_length_m=SPAN_M,
        )
        for index in range(N_SPANS)
    ]
    ordered = [kernel.evaluate(power, SPAN_M, 1, span_context=ctx) for ctx in contexts]
    scrambled = [kernel.evaluate(power, SPAN_M, 1, span_context=ctx) for ctx in reversed(contexts)]

    for index in range(len(contexts)):
        forward = ordered[index]
        backward = scrambled[N_SPANS - 1 - index]
        assert np.array_equal(
            np.asarray(forward.nli_amplitude_spm).view(np.int64),
            np.asarray(backward.nli_amplitude_spm).view(np.int64),
        ), f"span {index} depends on the call order, not on the SpanContext"


def test_span_phase_advances_with_the_accumulated_length() -> None:
    """第 k 跨的振幅相位必须随累计长度线性增长（等跨时即随跨号）。"""
    case = _case()
    model = _model(case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    power = launch_profile(case, 0.0)

    phases = []
    for index in range(N_SPANS):
        ctx = SpanContext(
            span_index=index,
            total_spans=N_SPANS,
            cumulative_length_m=index * SPAN_M,
            span_length_m=SPAN_M,
        )
        out = kernel.evaluate(power, SPAN_M, 1, span_context=ctx)
        phases.append(float(np.angle(out.nli_amplitude_spm[0])))
    steps = np.diff(phases)
    # 相位步长是常数（线性斜坡），只允许 2*pi 的缠绕。
    unwrapped = np.unwrap(np.asarray(phases))
    increments = np.diff(unwrapped)
    assert np.allclose(increments, increments[0], rtol=1e-6, atol=1e-9)
    assert abs(increments[0]) > 1e-6, "the phase must actually advance across spans"


# =============================================================================
# XPM 的权重口径
# =============================================================================
def _uniform_xpm_visibility(
    *,
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    span_separations_m: np.ndarray,
    channel_offset_hz: float,
) -> np.ndarray:
    r"""均匀带平均口径的 XPM 可见度（本文件内的独立实现，只用于对照）。

    跨拍的失配由**信道间隔**主导，因此均匀带平均给出
    :math:`V(d) = \mathrm{sinc}(cB/2)`，:math:`c = 4\pi^2|\beta_2|d\Delta`。
    """
    separations = np.asarray(span_separations_m, dtype=np.float64)
    argument = 2.0 * np.pi**2 * beta2_abs_s2_m * separations * channel_offset_hz * bandwidth_hz
    visibility = np.ones_like(separations)
    nonzero = argument != 0.0
    visibility[nonzero] = np.sin(argument[nonzero]) / argument[nonzero]
    return visibility


def test_xpm_weighting_effect_is_below_the_noise_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    """XPM 的权重口径选择对 SNR 的影响 < 0.05 dB。

    对照：把 GN 噪声功率加权的 XPM 可见度换成均匀带平均口径（数值模型早先
    用的就是后者），链路级 SNR 的变化必须小于 0.05 dB。两条口径给出的 XPM
    可见度都在 1e-3 量级——跨拍频的失配太大，两份拷贝基本不相干。
    """
    case = _case()
    power = launch_profile(case, 0.0)
    weighted = _model(case).evaluate_path(_path(case, N_SPANS), power)

    monkeypatch.setattr(buglia, "_gn_weighted_xpm_visibility", _uniform_xpm_visibility)
    uniform = _model(case).evaluate_path(_path(case, N_SPANS), power)

    delta_db = float(weighted.snr_db[0]) - float(uniform.snr_db[0])
    assert abs(delta_db) < 0.05, f"the XPM weighting convention moves the SNR by {delta_db:.4f} dB"


def test_xpm_buildup_is_almost_incoherent() -> None:
    """XPM 的相干增益基本为 1（跨拍频的失配让两份拷贝几乎不相干）。"""
    _, xpm, _ = _buildup(_case(), N_SPANS)
    assert xpm == pytest.approx(1.0, abs=0.01), f"XPM buildup {xpm}"


def test_spm_uses_the_gn_weight_not_a_uniform_band_average() -> None:
    """SPM 的可见度必须按 NLI 噪声功率加权。

    对照：均匀带平均口径给的是**两份信号副本**之间的相干度，不是噪声的
    相干度。两者在 10 跨上的 buildup 差在 0.1 dB 量级。
    """
    case = _case()
    model = _model(case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    distances = np.arange(1, N_SPANS, dtype=np.float64) * SPAN_M
    visibility = kernel._spm_visibility(distances, SPAN_M)
    buildup = float(buglia._coherent_factor(N_SPANS, visibility)[0]) / N_SPANS

    # 均匀带平均：sinc(z)，z = pi^2 |beta_2| d B^2。
    beta2 = abs(float(kernel.beta2_s2_m))
    argument = np.pi**2 * beta2 * distances * float(kernel.b_hz_channels[0]) ** 2
    uniform_visibility = np.sin(argument) / argument
    uniform_buildup = float(buglia._coherent_factor(N_SPANS, uniform_visibility[None, :])[0]) / N_SPANS

    assert buildup == pytest.approx(1.40, abs=0.03)
    # 加权口径明显高于均匀口径：均匀口径低估了噪声的相干度。
    assert buildup > uniform_buildup + 0.05, (
        f"weighted {buildup:.4f} vs uniform {uniform_buildup:.4f}: "
        "the GN weighting is not in effect"
    )


# =============================================================================
# 基准算例的链路级读数
# =============================================================================
@pytest.mark.parametrize("case_name", ["c_57ch", "cl_152ch", "escl_479ch"])
def test_benchmark_cases_stay_finite_and_ordered(case_name: str) -> None:
    """三个基准算例：多跨链路必须给出有限且单调下降的 SNR。"""
    case = build_case(case_name)
    power = launch_profile(case, 0.0)
    model = _model(case)
    single = model.evaluate_path(_path(case, 1), power)
    ten = model.evaluate_path(_path(case, 10), power)
    assert np.all(np.isfinite(ten.snr_db))
    assert float(ten.snr_db[0]) < float(single.snr_db[0])


def test_fibre_reference_is_the_smf_of_the_benchmark() -> None:
    """算例用的就是基准里的 SMF（100 km，0.2 dB/km，D = 17 ps/(nm·km)）。"""
    assert SMF_100KM.dispersion_parameter_d == 17.0
    assert SMF_100KM.length_km == 100.0
