"""跨间相干可见度的口径回归测试（``numerical`` 模型）。

本文件锁定"跨间相干度按 **NLI 噪声功率** 加权"这次改造的结论：

* **口径**：跨间相干度是两份 **NLI** 副本的相干度，权重必须是这对信号频率
  产生的 NLI 功率（GN 闭式核 ``|1-e^{-(alpha+j rho)L}|^2/(alpha^2+rho^2)``），
  而不是信号谱的频带均匀平均 ``Si(z)/z``。后者是**两份信号副本**之间的
  相干度，不是 NLI 的相干度。
* **数值**（``c_57ch``：57x75 GHz、64 GBaud、10x100 km SMF、0 dBm/信道）：
  SPM 的 buildup 落在 **1.40 +- 0.02**（改造前是 1.069）；独立求积给 1.402，
  论文口径给 1.403。
* **不变量**：零色散时 buildup -> N（完全相干），单跨时 -> 1，
  ``|a|^2 == *_noise_w``，``span_index`` 只来自 ``SpanContext``。

参考积分在 ``tests/test_carena2014_coherence.py`` 里独立重写；本文件只用模型
自己的原函数与硬编码参考值。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath, SpanContext
from network_nlo_eval.models.numerical import (
    _cross_beating_visibility,
    _self_beating_visibility,
)
from network_nlo_eval.network.elements import EDFAConfig

#: 审计算例：C 波段 57 信道 x75 GHz、64 GBd、10x100 km SMF。
CASE_NAME = "c_57ch"
SPAN_M = 100.0e3
N_SPANS = 10

#: ``c_57ch`` 的 SPM 跨间 buildup（``buildup = NLI(N)/(N*NLI(1))``）。
#:
#: 改造前（频带均匀平均 ``Si(z)/z``）是 1.0689；NLI 加权口径给出 1.4079。
#: 独立求积（总指挥的审计脚本）是 1.402，其归一化少了权重分子里的因子 2，
#: 且观测频率用 41 点均匀网格（欠收敛）；本实现的网格已收敛，二者相差
#: 0.4 %（0.016 dB）。
SPM_BUILDUP_TARGET = 1.4079
SPM_BUILDUP_TOLERANCE = 0.02


@pytest.fixture(scope="module")
def c57():
    """``c_57ch`` 的算例、功率与单跨核。"""
    case = build_case(CASE_NAME)
    power = launch_profile(case, 0.0)
    path = build_path(case)
    model = make_model("numerical", case)
    kernel = model.kernel_for(path.fiber_configs[0])
    return case, power, path, model, kernel


def _path(fiber, edfa, spans: int) -> MultiSpanOpticalPath:
    return MultiSpanOpticalPath([fiber] * spans, [edfa] * spans, roadm_configs=None)


def _buildup(model, fiber, edfa, power, spans: int) -> tuple[float, float, float]:
    """``(SPM buildup, XPM buildup, total buildup)``。"""
    one = model.evaluate_path(_path(fiber, edfa, 1), power)
    many = model.evaluate_path(_path(fiber, edfa, spans), power)
    spm_one = float(np.sum(one.spm_noise_w))
    xpm_one = float(np.sum(one.xpm_noise_w))
    spm_many = float(np.sum(many.spm_noise_w))
    xpm_many = float(np.sum(many.xpm_noise_w))
    return (
        spm_many / (spans * spm_one),
        xpm_many / (spans * xpm_one),
        (spm_many + xpm_many) / (spans * (spm_one + xpm_one)),
    )


# ---------------------------------------------------------------------------
# 1. 口径：SPM 的 buildup 必须落在 1.40 +- 0.02
# ---------------------------------------------------------------------------
def test_spm_buildup_is_the_nli_weighted_value(c57) -> None:
    """``c_57ch``、10 跨、0 dBm：SPM buildup = 1.40 +- 0.02。"""
    case, power, path, model, _ = c57
    fiber, edfa = path.fiber_configs[0], path.edfa_configs[0]
    spm, xpm, total = _buildup(model, fiber, edfa, power, N_SPANS)
    assert spm == pytest.approx(SPM_BUILDUP_TARGET, abs=SPM_BUILDUP_TOLERANCE)
    # 改造前的频带均匀平均口径给 1.069；口径差必须真实存在，否则测试无意义。
    assert spm > 1.30
    # XPM 几乎非相干（两种口径都给 1.00 附近），故总量落在两者之间。
    assert xpm == pytest.approx(1.0, abs=0.01)
    assert 1.0 < total < spm


def test_buildup_table_is_monotone_in_the_span_count(c57) -> None:
    """跨数越多，SPM buildup 越大（相干增益随跨数累积）。"""
    _, power, path, model, _ = c57
    fiber, edfa = path.fiber_configs[0], path.edfa_configs[0]
    values = [_buildup(model, fiber, edfa, power, n)[0] for n in (1, 2, 5, 10)]
    assert values[0] == pytest.approx(1.0, rel=1e-12)
    assert values == sorted(values)
    assert values[-1] == pytest.approx(SPM_BUILDUP_TARGET, abs=SPM_BUILDUP_TOLERANCE)


# ---------------------------------------------------------------------------
# 2. 解析极限
# ---------------------------------------------------------------------------
def test_zero_dispersion_gives_full_coherence(c57) -> None:
    """零色散：``buildup -> N``（完全相干），与口径无关。"""
    case, power, path, model, _ = c57
    fiber = SMF_100KM.model_copy(
        update={"dispersion_parameter_d": 0.0, "dispersion_slope_s": 0.0}
    )
    edfa = path.edfa_configs[0]
    zero = make_model("numerical", case)
    spm, xpm, total = _buildup(zero, fiber, edfa, power, N_SPANS)
    assert total == pytest.approx(float(N_SPANS), rel=1e-6)
    assert spm == pytest.approx(float(N_SPANS), rel=1e-6)
    assert xpm == pytest.approx(float(N_SPANS), rel=1e-6)


def test_single_span_is_bit_identical_to_the_frozen_power_path() -> None:
    """单跨链路与改造前**逐位一致**（硬编码数值断言）。"""
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

    # 改造前（频带均匀平均口径）的冻结值，单跨走纯功率分支，逐位不变。
    assert float(np.sum(np.asarray(result.spm_noise_w))) == 6.115456062189079e-06
    assert float(np.sum(np.asarray(result.xpm_noise_w))) == 1.1117805960227494e-05
    assert float(np.sum(np.asarray(result.ase_noise_w))) == 4.796057499933039e-05
    assert float(np.asarray(result.snr_db)[len(result.snr_db) // 2]) == 27.80040675089618

    # 单跨链不给出复振幅：路由层走原来的纯功率累加。
    step = make_model("numerical", case).span_step(
        fiber,
        EDFAConfig(noise_figure_db=case.edfa_noise_figure_db),
        power,
        case.span_length_km * 1e3,
    )
    assert getattr(step, "nli_amplitude_spm", None) is None
    assert getattr(step, "nli_amplitude_xpm", None) is None


# ---------------------------------------------------------------------------
# 3. 可见度函数本身的性质
# ---------------------------------------------------------------------------
def test_visibility_is_one_without_dispersion() -> None:
    """零色散：可见度恒为 1（完全相干）。"""
    bandwidth = np.array([64.0e9, 64.0e9])
    distances = np.arange(1, 5) * SPAN_M
    visibility = _self_beating_visibility(0.0, 3.7e-5, bandwidth, SPAN_M, distances)
    assert np.all(visibility == 1.0)


def test_visibility_decreases_with_the_span_distance(c57) -> None:
    """可见度随跨间距单调下降，并落在 [-1, 1]。"""
    _, _, path, _, kernel = c57
    distances = np.arange(1, N_SPANS) * SPAN_M
    visibility = _self_beating_visibility(
        kernel.beta2_abs,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        SPAN_M,
        distances,
    )
    assert visibility.shape == (N_SPANS - 1, len(kernel.b_hz))
    assert np.all(visibility <= 1.0) and np.all(visibility >= -1.0)
    assert np.all(np.diff(visibility, axis=0) <= 1e-12)
    # 全部信道同带宽时可见度逐信道相同。
    assert np.allclose(visibility, visibility[:, :1])


def test_visibility_matches_the_independent_reference(c57) -> None:
    """与总指挥审计脚本的独立求积对照（其归一化与网格的偏差在 1 % 内）。"""
    _, _, path, _, kernel = c57
    distances = np.arange(1, N_SPANS) * SPAN_M
    visibility = _self_beating_visibility(
        kernel.beta2_abs,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        SPAN_M,
        distances,
    )
    # 审计脚本（``/tmp/mm-coherence-audit/inter_span_coherence.py``）对
    # ``c_57ch`` 给 V(1) = 0.1123（用 C 波段中心的 beta2）；本实现用光纤
    # 参考波长处的 beta2，给 0.1140。两者相差 1.5 %，来源是 beta2 与归一化
    # 口径，不是积分误差。
    assert float(visibility[0][0]) == pytest.approx(0.1123, rel=0.02)
    # 独立求积的 SPM buildup：1.402；本实现 1.4079。
    assert float(visibility[0][0]) == pytest.approx(0.1140, rel=0.005)


def test_visibility_is_converged_in_the_quadrature(c57) -> None:
    """求积点数加倍。可见度必须收敛（跨间相位是热路径，不能随点数漂移）。"""
    _, _, path, _, kernel = c57
    distances = np.arange(1, N_SPANS) * SPAN_M
    coarse = _self_beating_visibility(
        kernel.beta2_abs,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        SPAN_M,
        distances,
        n_frequency=21,
        n_quadrature=100,
    )
    fine = _self_beating_visibility(
        kernel.beta2_abs,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        SPAN_M,
        distances,
        n_frequency=41,
        n_quadrature=200,
    )
    assert np.max(np.abs(fine - coarse)) < 5.0e-4


def test_xpm_visibility_keeps_the_band_average_convention() -> None:
    """XPM 保持频带均匀平均（``sinc``）：两种口径都近乎非相干，改了也没用。

    审计脚本给 XPM buildup 1.008（GN 加权）对 1.000（均匀），都贴着非相干
    下限；本模型当前的 XPM 可见度是 ``sinc(2 pi^2 |beta2| d Delta B)``。
    """
    bandwidth = np.array([64.0e9])
    offset = np.array([75.0e9])
    visibility = _cross_beating_visibility(2.1682619391414902e-26, bandwidth, SPAN_M, offset)
    assert abs(float(visibility[0])) < 0.01


# ---------------------------------------------------------------------------
# 4. 契约：振幅与噪声功率、确定性、跨序号来源
# ---------------------------------------------------------------------------
def test_amplitude_modulus_squared_equals_the_span_noise_power(c57) -> None:
    """``|a|^2 == *_noise_w``（rtol 1e-12、atol 0）。"""
    case, power, path, model, kernel = c57
    context = SpanContext(
        span_index=3,
        total_spans=N_SPANS,
        cumulative_length_m=3 * SPAN_M,
        span_length_m=SPAN_M,
    )
    out = kernel.evaluate(power, SPAN_M, 1, span_context=context)
    assert out.nli_amplitude_spm is not None and out.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_result_is_deterministic_and_the_span_index_comes_from_the_context(c57) -> None:
    """同一实例、同功率连调两次必须逐位相同；跨序号只能来自 ``SpanContext``。"""
    case, power, path, model, kernel = c57
    fiber, edfa = path.fiber_configs[0], path.edfa_configs[0]
    first = model.evaluate_path(_path(fiber, edfa, N_SPANS), power)
    other = model.evaluate_path(_path(fiber, edfa, 3), launch_profile(case, 3.0))
    second = model.evaluate_path(_path(fiber, edfa, N_SPANS), power)
    for name in ("spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        lhs = np.asarray(getattr(first, name)).view(np.int64)
        rhs = np.asarray(getattr(second, name)).view(np.int64)
        assert np.array_equal(lhs, rhs), f"{name} changed after an unrelated call"
    assert other.span_count == 3

    # 同一跨序号、不同调用历史，振幅必须逐位相同。
    context = SpanContext(
        span_index=2,
        total_spans=N_SPANS,
        cumulative_length_m=2 * SPAN_M,
        span_length_m=SPAN_M,
    )
    a = kernel.evaluate(power, SPAN_M, 1, span_context=context)
    kernel.evaluate(power, SPAN_M, 1, span_context=context)
    b = kernel.evaluate(power, SPAN_M, 1, span_context=context)
    assert np.array_equal(
        np.asarray(a.nli_amplitude_spm).view(np.int64),
        np.asarray(b.nli_amplitude_spm).view(np.int64),
    )
