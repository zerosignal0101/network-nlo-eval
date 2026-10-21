"""carena2014 的 4 阶修正项与**独立求积**的对照（回归守卫）。

2014 论文 Eq. (8) 的 4 阶修正 ``kappa2`` 是**两行**：

* 第一行（``80/81``）按附录 C 的 Eq. (43) 化简，外层变量是 ``f1``，记作 ``L1``；
* 第二行（``16/81``）把积分变量 ``f1`` 换成 ``f3 = f1+f2-f``（``f1 = f3-f2+f``），
  按 Eq. (44) 化简，**外层变量是 ``f3``**，记作 ``L2``。

论文原文（``full.md`` 第 636 行与第 644 行）把这两项分别叫做 ``kappa21(f)`` 与
``kappa22(f)``。两者的谱因子相同（矩形谱下都是 ``R_s^{-6}``），**积分域不同**：
``L2`` 的外层变量是第三个波形。把两行合并成一个积分（``(96/81) L1``）会把 ``kappa2``
抬高 8.5 %（N=1）到 15 %（N=10），并把 N=10 的 EGN SPM buildup 从 1.9624 抬到 2.0500。

本文件用 ``tests/egn_reference.py``（独立求积，不 import 任何模型代码）把这件事钉住：

* ``L2/L1`` 的比值与独立求积一致（模型的两行确实是**两个不同的积分**）；
* c_57ch 上 N=10 的 EGN SPM buildup 落在 1.9624 +- 0.01，GN SPM buildup 落在
  1.4034 +- 0.01；
* N=10 的总量 buildup 落在 1.6173 +- 0.01。

**运行时间**：独立求积是二维自适应求积，比被审对象慢两个量级；下面的
``reference_buildups`` fixture 要跑几分钟（模块级，只跑一次）。这是本文件唯一的慢点，
它换来的是"模型的积分式有没有跑偏"这把外部尺子。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, launch_profile
from network_nlo_eval.benchmarks.runner import build_path, make_model
from network_nlo_eval.models.carena2014 import EgnIntegrator, LinkFunction

from . import egn_reference as ref

#: 独立求积每个振荡周期的 Gauss-Legendre 节点数（**逐点对照**用）。
#:
#: 32 是独立求积自己的收敛值：同一个 ``sci_kappas`` 调用在 npx=32 与 npx=64 下
#: L1 相对差 1.2e-7、L2 相对差 < 1e-6（N=10, f=0，实测）。npx=16 时 L1 还有
#: 6.8e-4 的偏差（L2 只有 2.6e-5），正好和本模型的求积误差同量级，会让"模型 vs
#: 参照"的比较失去分辨力，所以逐点对照取 32。
REFERENCE_NPX = 32

#: 独立求积每个振荡周期的 Gauss-Legendre 节点数（**buildup** 用）。
#:
#: buildup 的判定带只有 +-0.01，而独立求积在这一口径上的耗时随 npx 线性增长
#: （N=1 与 N=10 各跑一遍 41 点频率网格 + 8 个邻居的 XCI 聚合）。npx=16 与
#: npx=32 给出的 EGN SPM buildup 相同（1.9624，落在带心），所以取 16。
BUILDUP_NPX = 16

#: 独立求积在 CUT 带内的采样点数（与模型一致：41 点无权重 Riemann 和）。
REFERENCE_FREQ_POINTS = 41

#: 判定带（来自 ``evidence/egn-coherence`` 的实测表）。
TOLERANCE = 0.01


@pytest.fixture(scope="module")
def reference_buildups() -> dict[str, float]:
    """独立求积的 buildup 指标（N=1 与 N=10，SPM 与 XPM）。

    Returns
    -------
        字典，键见 :func:`egn_reference.buildups`。
    """
    return ref.buildups(
        npx=BUILDUP_NPX,
        n_freq=REFERENCE_FREQ_POINTS,
        spans=(1, 10),
    )


def _reference_link(n_spans: int) -> LinkFunction:
    """独立求积所用的链路函数（与 ``egn_reference.CASE`` 逐位一致）。"""
    fiber = ref.CASE.fiber
    return LinkFunction(
        gamma=fiber.gamma_per_w_m,
        alpha_power_npm=fiber.alpha_power_npm,
        beta2_s2_m=fiber.beta2_s2_m,
        span_length_m=fiber.length_m,
        n_spans=n_spans,
    )


def _model_integrator(n_spans: int) -> EgnIntegrator:
    """模型侧、参数与独立求积完全一致的求积器。"""
    return EgnIntegrator(
        link=_reference_link(n_spans),
        symbol_rate_hz=ref.CASE.symbol_rate_hz,
        points=6,
        p_limit_hz2=(ref.CASE.spacing_hz * ref.CASE.neighbor_span + ref.CASE.symbol_rate_hz) ** 2,
    )


def _model_rows(integrator: EgnIntegrator, f_hz: float) -> tuple[float, float]:
    """模型侧 Eq. (8) 两行的积分值：``(L1, L2)``。"""
    half = 0.5 * integrator.symbol_rate_hz
    band = (-half, half)
    l1 = integrator._core(f_hz, band, band, band, "fourth")[1]
    l2 = integrator._core(f_hz, band, band, band, "l2")[1]
    return l1, l2


@pytest.mark.parametrize("n_spans", [1, 10])
def test_the_two_rows_of_eq8_are_two_different_integrals(n_spans: int) -> None:
    """Eq. (8) 的两行是**两个不同的积分**（``L2/L1`` 既不等于 1 也不等于常数）。

    这条断言直接盯住被修掉的那个错误：把两行合并成 ``(96/81) L1`` 等价于宣称
    ``L2 = L1``。
    """
    integrator = _model_integrator(n_spans)
    l1, l2 = _model_rows(integrator, 0.0)
    assert l1 > 0.0
    assert l2 > 0.0
    assert l2 != pytest.approx(l1, rel=1e-6)


@pytest.mark.parametrize("n_spans", [1, 10])
def test_kappa21_over_kappa22_ratio_matches_the_independent_quadrature(n_spans: int) -> None:
    """模型的两行之比与独立求积逐点一致（相对偏差 < 1e-3）。

    独立求积与模型用的是**不同的求积方法**（二维自适应 Gauss-Legendre vs
    ``p`` 轴原函数表 + 乘积积分），所以这是一次真正的交叉验证，而不是自证。

    实测偏差（``L2_CELLS = 256``、参照 npx=32，最坏点是 N=10、f=0）：

    * ``L1``：-1.3e-6（N=1 与 N=10 都是这个量级）；
    * ``L2``：N=1 时 -2.4e-5 ~ +2.4e-5，N=10 时 f=0 是 +2.0e-4、f=+16 GHz 是
      +5.6e-5、f=-25.6 GHz 是 +1.0e-5。

    参照自己的收敛误差（npx=32 与 npx=64 之差）是 L1 1.2e-7、L2 < 1e-6，比这里的
    断言紧一个量级，所以这条断言测到的是**被审对象**的误差，不是参照的。
    """
    integrator = _model_integrator(n_spans)
    for f_hz in (0.0, 0.25 * ref.CASE.symbol_rate_hz, -0.4 * ref.CASE.symbol_rate_hz):
        l1, l2 = _model_rows(integrator, f_hz)
        expected = ref.sci_kappas(f_hz, n_spans, npx=REFERENCE_NPX)
        assert l1 == pytest.approx(expected.l1, rel=1e-3), (n_spans, f_hz, "L1")
        assert l2 == pytest.approx(expected.l2, rel=1e-3), (n_spans, f_hz, "L2")
        assert (l2 / l1) == pytest.approx(expected.l2 / expected.l1, rel=1e-3)


@pytest.mark.parametrize("n_spans", [1, 10])
def test_l2_quadrature_is_converged(n_spans: int) -> None:
    """``l2`` 的两个求积参数（分段数、外层网格细分因子）加倍时结果变化 < 1e-3。

    这两个参数只影响截断误差：``L2_CELLS`` 是 ``u`` 轴上的乘积积分分段数，
    ``L2_OUTER_DIVISOR`` 是外层网格的细分因子（>1 = 更粗）。把它们同时加倍，
    ``L1``/``L2`` 的相对变化必须小于 1e-3。
    """
    from network_nlo_eval.models import carena2014 as module

    original_cells = module.L2_CELLS
    original_divisor = module.L2_OUTER_DIVISOR
    try:
        base = _model_rows(_model_integrator(n_spans), 0.0)
        module.L2_CELLS = original_cells * 2
        module.L2_OUTER_DIVISOR = original_divisor * 2
        doubled = _model_rows(_model_integrator(n_spans), 0.0)
    finally:
        module.L2_CELLS = original_cells
        module.L2_OUTER_DIVISOR = original_divisor
    for old, new in zip(base, doubled, strict=True):
        assert new == pytest.approx(old, rel=1e-3)


def test_independent_quadrature_reproduces_the_paper_buildups(reference_buildups: dict[str, float]) -> None:
    """独立求积自身落在论文口径的判定带里（fixture 自身的合理性检查）。"""
    assert reference_buildups["gn_spm"] == pytest.approx(ref.TARGET_GN_SPM_BUILDUP, abs=TOLERANCE)
    assert reference_buildups["egn_spm"] == pytest.approx(ref.TARGET_EGN_SPM_BUILDUP, abs=TOLERANCE)
    assert reference_buildups["total_egn"] == pytest.approx(ref.TARGET_TOTAL_BUILDUP, abs=TOLERANCE)


def test_model_sci_buildup_matches_the_independent_quadrature(
    reference_buildups: dict[str, float],
) -> None:
    """模型自己的 EGN SPM buildup 与独立求积一致（相对偏差 < 1e-3）。

    模型侧用的是 ``Carena2014SpanKernel._sci_block``：``sum_f (kappa1 + Phi kappa2 +
    Psi kappa3)``，与独立求积的带平均同口径。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    one = kernel._sci_block(1, 100e3)[1]
    ten = kernel._sci_block(10, 100e3)[1]
    model_buildup = ten / (10.0 * one)
    assert model_buildup == pytest.approx(reference_buildups["egn_spm"], rel=1e-3)
    assert model_buildup == pytest.approx(ref.TARGET_EGN_SPM_BUILDUP, abs=TOLERANCE)


def test_model_total_buildup_is_in_the_paper_band(reference_buildups: dict[str, float]) -> None:
    """模型的总量 buildup（N=10）落在 1.6173 +- 0.01。

    总量 buildup 用独立求积同一套冻结权重合成：``(SPM(1) b_spm + XPM(1) b_xpm) /
    (SPM(1) + XPM(1))``，其中 ``SPM(1)``/``XPM(1)`` 是仓库单跨的实测值。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    one_sci, one_xpm = kernel.link_level_nli(power, 100e3, 1)
    ten_sci, ten_xpm = kernel.link_level_nli(power, 100e3, 10)
    b_spm = float(np.sum(ten_sci)) / (10.0 * float(np.sum(one_sci)))
    b_xpm = float(np.sum(ten_xpm)) / (10.0 * float(np.sum(one_xpm)))
    model_total = ref.repo_total_buildup(b_spm, b_xpm)
    assert model_total == pytest.approx(ref.TARGET_TOTAL_BUILDUP, abs=TOLERANCE)
    # 模型与独立求积的总量 buildup 也要一致（相对偏差 < 1e-3）。
    assert model_total == pytest.approx(reference_buildups["total_egn"], rel=1e-3)
