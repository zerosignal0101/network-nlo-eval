"""carena2014 的带内采样收敛测试。

**背景**

2014 论文 Eq. (7)-(9) 的积分在 CUT 带内逐点求值，模型把带内所有采样点的 ``kappa``
相加当作带平均。这个和无权重，误差按 ``O(1/fp)`` 收敛：点数不足时，"单跨"与
"N 跨"两条曲线的比值（跨间相干累加因子）一起偏。``c_57ch`` 上 3 点采样把 SCI 的
GN 累加因子压到 13.067，而论文的收敛值是 14.034（buildup 1.4034）。

本文件锁住三件事：

* 默认点数是收敛的（相对论文收敛值的相对误差 < 1e-3，且相邻两点的相对步长 < 1e-3）；
* 收敛的**唯一**旋钮是带内点数——``p`` 轴上的求积（``quadrature_points``）早已收敛，
  把节点数加倍不再改变结果；
* 收敛之后模型的不变量仍然成立：``|a|^2 == *_noise_w``、单跨走纯功率分支、
  同一实例同输入逐位可复现。

收敛表（``c_57ch``，N=10，``_sci_block`` 的比值）：

=========  ==========  ==============  ==============
``fp``     ``R_gn``     ``buildup`` GN  ``buildup`` EGN
=========  ==========  ==============  ==============
3          13.06654     1.306654        1.890356
5          13.48578     1.348578        1.963397
7          13.75846     1.375846        2.012900
9          13.82043     1.382043        2.021614
13         13.92152     1.392152        2.037090
17         13.95711     1.395711        2.041248
25         13.99667     1.399667        2.046003
33         14.01981     1.401981        2.048564
41         14.03370     1.403370        2.049977
49         14.04232     1.404232        2.050752
=========  ==========  ==============  ==============
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import SpanContext
from network_nlo_eval.models.carena2014 import DEFAULT_FREQUENCY_POINTS

CASE_NAME = "c_57ch"
SPAN_M = 100.0e3
N_SPANS = 10

#: 论文的收敛值（独立求积 1.4020，论文实现 1.4034）。
PAPER_BUILDUP = 1.4034

#: WP 的验收带。
BUILDUP_BAND = (1.403 - 0.005, 1.403 + 0.005)

#: 收敛表：点数 -> (R_gn, buildup_gn)。
CONVERGENCE_ROWS: tuple[tuple[int, float, float], ...] = (
    (3, 13.066541, 1.306654),
    (5, 13.485778, 1.348578),
    (7, 13.758464, 1.375846),
    (9, 13.820429, 1.382043),
    (13, 13.921523, 1.392152),
    (17, 13.957113, 1.395711),
    (25, 13.996669, 1.399667),
    (33, 14.019814, 1.401981),
    (41, 14.033703, 1.403370),
    (49, 14.042316, 1.404232),
)


def _kernel(frequency_points: int, quadrature_points: int | None = None):
    """c_57ch 的 SMF 单跨核，按给定的带内点数构造。"""
    case = build_case(CASE_NAME)
    options: dict[str, int] = {"frequency_points": int(frequency_points)}
    if quadrature_points is not None:
        options["quadrature_points"] = int(quadrature_points)
    model = make_model("carena2014", case, **options)
    return model.kernel_for(build_path(case).fiber_configs[0])


def _buildup_rows() -> list[tuple[int, float, float]]:
    """实测的 (fp, R_gn, buildup_gn)。"""
    rows = []
    for fp, _, _ in CONVERGENCE_ROWS:
        kernel = _kernel(fp)
        gn_many, _ = kernel._sci_block(N_SPANS, SPAN_M)
        gn_one, _ = kernel._sci_block(1, SPAN_M)
        rows.append((fp, gn_many / gn_one, gn_many / gn_one / N_SPANS))
    return rows


@pytest.fixture(scope="module")
def measured_rows() -> list[tuple[int, float, float]]:
    """模块内共享一次扫描（每个点数都要建原函数表，很贵）。"""
    return _buildup_rows()


def test_default_frequency_points_meets_the_paper_buildup(measured_rows) -> None:
    """默认点数下，c_57ch 的 SCI GN buildup 落在 1.403 +- 0.005 内。"""
    by_fp = {fp: (ratio, buildup) for fp, ratio, buildup in measured_rows}
    assert DEFAULT_FREQUENCY_POINTS in by_fp
    _, buildup = by_fp[DEFAULT_FREQUENCY_POINTS]
    assert BUILDUP_BAND[0] <= buildup <= BUILDUP_BAND[1]
    # 相对论文收敛值的误差 < 1e-3。
    assert abs(buildup - PAPER_BUILDUP) / PAPER_BUILDUP < 1e-3


def test_measured_buildup_matches_the_recorded_convergence_table(measured_rows) -> None:
    """实测值与本文件记录的收敛表逐点一致（防止表本身腐化）。"""
    for (fp, ratio, buildup), (want_fp, want_ratio, want_buildup) in zip(
        measured_rows, CONVERGENCE_ROWS, strict=True
    ):
        assert fp == want_fp
        assert ratio == pytest.approx(want_ratio, rel=1e-5)
        assert buildup == pytest.approx(want_buildup, rel=1e-5)


def test_buildup_increases_monotonically_with_frequency_points(measured_rows) -> None:
    """点数越多，累加因子越大：3 点把相干度算低了。"""
    buildups = [buildup for _, _, buildup in measured_rows]
    assert all(later > earlier for earlier, later in zip(buildups[:-1], buildups[1:], strict=False))
    # 起点（3 点）比论文收敛值低 6% 以上，末端在 1e-3 以内。
    assert buildups[0] < PAPER_BUILDUP * (1.0 - 0.05)
    assert abs(buildups[-1] - PAPER_BUILDUP) / PAPER_BUILDUP < 1e-3


def test_consecutive_points_differ_by_less_than_1e_3(measured_rows) -> None:
    """收敛表的最后两点相对步长 < 1e-3（收敛判据）。"""
    last_two = [row[2] for row in measured_rows[-2:]]
    assert abs(last_two[1] - last_two[0]) / last_two[1] < 1e-3


def test_error_against_the_paper_value_decreases_monotonically(measured_rows) -> None:
    """相对论文收敛值的误差随点数单调下降（在默认点数之前）。"""
    errors = [abs(buildup - PAPER_BUILDUP) for _, _, buildup in measured_rows]
    cut = [fp for fp, _, _ in measured_rows].index(DEFAULT_FREQUENCY_POINTS)
    head = errors[: cut + 1]
    assert all(later < earlier for earlier, later in zip(head[:-1], head[1:], strict=False))


def test_band_sampling_is_the_only_knob() -> None:
    """``p`` 轴上的求积早已收敛：节点数加倍不再改变结果。

    带内点数是唯一剩下的旋钮，所以这条测试守住"别把误差算到求积节点数上"。
    """
    coarse = _kernel(9, quadrature_points=6)
    fine = _kernel(9, quadrature_points=12)
    for n_spans in (1, N_SPANS):
        gn_coarse, _ = coarse._sci_block(n_spans, SPAN_M)
        gn_fine, _ = fine._sci_block(n_spans, SPAN_M)
        assert gn_coarse == pytest.approx(gn_fine, rel=1e-6)


def test_single_span_amplitudes_match_the_noise_power() -> None:
    """``|a|^2 == *_noise_w``（契约要求，振幅取模平方后逐位一致）。"""
    case = build_case(CASE_NAME)
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    out = kernel.evaluate(power, SPAN_M, 1)
    assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-15, atol=0.0)
    assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-15, atol=0.0)
    # 单跨走纯功率分支：振幅相位取 0。
    assert np.all(out.nli_amplitude_spm.imag == 0.0)
    assert np.all(out.nli_amplitude_xpm.imag == 0.0)


def test_single_span_path_is_deterministic() -> None:
    """同一实例、同一输入连调三次逐位相同。"""
    case = build_case(CASE_NAME)
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    context = SpanContext(
        span_index=0,
        total_spans=N_SPANS,
        cumulative_length_m=0.0,
        span_length_m=SPAN_M,
    )
    first = kernel.evaluate(power, SPAN_M, N_SPANS, span_context=context)
    second = kernel.evaluate(power, SPAN_M, N_SPANS, span_context=context)
    third = kernel.evaluate(power, SPAN_M, N_SPANS, span_context=context)
    assert np.array_equal(np.asarray(first.spm_noise_w), np.asarray(second.spm_noise_w))
    assert np.array_equal(np.asarray(second.spm_noise_w), np.asarray(third.spm_noise_w))
    assert np.array_equal(np.asarray(first.xpm_noise_w), np.asarray(third.xpm_noise_w))
    assert np.array_equal(np.asarray(first.nli_amplitude_spm), np.asarray(third.nli_amplitude_spm))
