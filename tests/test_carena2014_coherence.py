"""跨间相干口径的审计回归测试（carena2014）。

本文件锁定"跨间相干增益为什么比频带平均口径高 2.07 dB"这次审计的结论：

* 论文 Eq. (12) 的 ``nu(p)`` 与逐跨相位步长的对应关系是精确的
  （``|nu_N(p)|^2 = N + 2 sum_Delta (N-Delta) cos(Delta·2 psi(p))``，逐点恒等式）；
* 论文口径下的 SCI GN 累加因子等于**按噪声密度加权**的逐对可见度之和
  ``R = N + 2 sum_Delta (N-Delta) <cos(Delta theta)>``；
* 同一积分换成**均匀**（频带平均）权重就落到 ``semrau2019`` 的口径上；
  ``numerical`` 也一度如此，自 2026-10-21 起改成 NLI 加权（见
  ``tests/test_numerical_visibility.py``），不再属于这一类；
* 两个口径在零色散（buildup = N）与单跨（buildup = 1）处收敛。

参考积分在这里**独立重写**：直接按论文的 ``zeta``/``nu`` 公式做嵌套复合
Gauss-Legendre，不复用模型内部的原函数表、外层网格或 Dirichlet 反解。
参考积分的带内采样点跟模型自己的网格走（``kernel._frequency_grid()``）：模型把
带内各点的 ``kappa`` 相加当带平均，参考积分用同一组 ``f`` 才是在比同一条积分。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss

from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    build_case,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import build_path, make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.carena2014 import LinkFunction

#: 审计算例：C 波段 57 信道 ×75 GHz、64 GBd、10×100 km SMF。
CASE_NAME = "c_57ch"
SPAN_M = 100.0e3
N_SPANS = 10


@pytest.fixture(scope="module")
def kernel():
    """c_57ch 的 SMF 单跨核（模块内复用，避免重复建原函数表）。"""
    case = build_case(CASE_NAME)
    model = make_model("carena2014", case)
    return model.kernel_for(build_path(case).fiber_configs[0])


@pytest.fixture(scope="module")
def reference_visibility(kernel):
    """参考积分（模块内算一次：带内 41 个采样点的嵌套求积很贵）。"""
    return _reference_visibility(kernel)


def _composite_gl(lo: float, hi: float, panels: int, order: int = 8):
    """复合 Gauss-Legendre 的节点与权重。"""
    x, w = leggauss(order)
    edges = np.linspace(lo, hi, panels + 1)
    width = np.diff(edges)
    nodes = edges[:-1, None] + width[:, None] * (0.5 * (x + 1.0))[None, :]
    return nodes.ravel(), (0.5 * width[:, None] * w).ravel()


def _domains(half_band_hz: float, f_values=None):
    """每个 CUT 带采样频率各自的双重积分区域。

    频率 ``f`` 处：``u = f1 - f`` 与 ``v = f2 - f`` 各在 ``[f-B/2, f+B/2]``，
    第三个波形 ``f3 = f1+f2-f`` 也在同一频带内，于是 ``u+v`` 落在
    ``[f-B/2, f+B/2]``。``f = 0`` 是六边形，``f = ±B/2`` 是三角形。

    采样点必须与模型自己的带内网格一致（``kernel._frequency_grid()``）：模型把
    带内各点的 ``kappa`` 相加当带平均，参考积分用同一组 ``f`` 才是在比同一条
    积分。默认的三点（``-B/2, 0, +B/2``）只用于没有模型网格的场合。
    """
    if f_values is None:
        f_values = (-half_band_hz, 0.0, half_band_hz)
    return [
        tuple(
            -half_band_hz - f_hz if index % 2 == 0 else half_band_hz - f_hz
            for index in range(6)
        )
        for f_hz in f_values
    ]


def _domain_integral(func, domain, u_panels: int, v_panels: int) -> float:
    """在一个积分区域上算 ``int int func(uv) du dv``（嵌套复合 Gauss-Legendre）。"""
    u_lo, u_hi, v_lo, v_hi, w_lo, w_hi = domain
    breaks = sorted(
        {u_lo, u_hi} | {c for c in (w_lo - v_lo, w_hi - v_hi) if u_lo < c < u_hi}
    )
    total = 0.0
    for lo, hi in zip(breaks[:-1], breaks[1:], strict=False):
        u_nodes, u_weights = _composite_gl(lo, hi, max(1, u_panels // max(1, len(breaks) - 1)))
        v_low = np.maximum(v_lo, w_lo - u_nodes)
        v_high = np.minimum(v_hi, w_hi - u_nodes)
        for u, weight, start, stop in zip(u_nodes, u_weights, v_low, v_high, strict=True):
            if stop <= start:
                continue
            v_nodes, v_weights = _composite_gl(start, stop, v_panels)
            total += weight * float(np.sum(v_weights * func(u * v_nodes)))
    return total


def _reference_visibility(kernel, u_panels: int = 30, v_panels: int = 60):
    """按论文的 ``zeta``/``nu`` 公式独立算逐对可见度。

    Returns
    -------
        ``(weighted, uniform)``：``weighted`` 是按噪声密度 ``|zeta|^2`` 加权的
        ``<cos(Delta theta)>``，``uniform`` 是同一积分区域上的均匀平均。
    """
    beta2 = float(kernel.beta2)
    alpha = float(kernel.alpha_center)
    gamma = float(kernel.gamma_center)
    band = float(kernel.symbol_rate_hz)

    def zeta(p):
        theta = 4.0 * np.pi**2 * beta2 * p * SPAN_M
        return (
            gamma
            * (1.0 - np.exp(-alpha * SPAN_M) * np.exp(1j * theta))
            / (alpha - 1j * 4.0 * np.pi**2 * beta2 * p)
        )

    def theta(p):
        return 4.0 * np.pi**2 * beta2 * p * SPAN_M

    domains = _domains(0.5 * band, kernel._frequency_grid())
    weight_total = sum(
        _domain_integral(lambda p: np.abs(zeta(p)) ** 2, d, u_panels, v_panels) for d in domains
    )
    area_total = sum(
        _domain_integral(lambda p: np.ones_like(p), d, u_panels, v_panels) for d in domains
    )
    weighted = []
    uniform = []
    for delta in range(1, N_SPANS):
        cos_delta = sum(
            _domain_integral(
                lambda p, k=delta: np.abs(zeta(p)) ** 2 * np.cos(k * theta(p)),
                d,
                u_panels,
                v_panels,
            )
            for d in domains
        )
        cos_uniform = sum(
            _domain_integral(
                lambda p, k=delta: np.cos(k * theta(p)), d, u_panels, v_panels
            )
            for d in domains
        )
        weighted.append(cos_delta / weight_total)
        uniform.append(cos_uniform / area_total)
    return weighted, uniform


def _coherent_factor(visibility, n_spans: int) -> float:
    """``R(N) = N + 2 sum_Delta (N-Delta) <cos(Delta theta)>``。"""
    return float(
        n_spans
        + 2.0 * sum((n_spans - d) * visibility[d - 1] for d in range(1, n_spans))
    )


def _buildup(model_name: str, fiber, spans: int) -> float:
    """与 ``scripts/coherence_buildup.py`` 同一把尺子：``NLI(N)/(N·NLI(1))``。"""
    case = build_case(CASE_NAME)
    power = launch_profile(case, 0.0)
    edfa = build_path(case).edfa_configs[0]
    model = make_model(model_name, case)
    one = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [edfa], roadm_configs=None), power
    )
    many = model.evaluate_path(
        MultiSpanOpticalPath([fiber] * spans, [edfa] * spans, roadm_configs=None), power
    )
    nli_one = float(np.sum(one.spm_noise_w) + np.sum(one.xpm_noise_w))
    nli_many = float(np.sum(many.spm_noise_w) + np.sum(many.xpm_noise_w))
    return (nli_many / spans) / nli_one


# ---------------------------------------------------------------------------
# 1. 论文 nu(p) 与逐跨相位步长的对应关系
# ---------------------------------------------------------------------------
def test_nu_is_exactly_the_pairwise_span_sum() -> None:
    """``|nu_N(p)|^2`` 逐点等于 ``N + 2 sum_Delta (N-Delta) cos(Delta·2 psi)``。

    这条恒等式把论文 Eq. (12) 的 ``nu(p)`` 与"相邻跨间相位差 ``2 psi``"绑在一起：
    相位定义若差一个因子 2，或把逐跨相位差误当成到接收端的累积相位，这里就会破。
    """
    beta2 = -2.1607590148802473e-26
    link = LinkFunction(
        gamma=1.3219450475712609e-3,
        alpha_power_npm=3.735371857445309e-05,
        beta2_s2_m=beta2,
        span_length_m=SPAN_M,
        n_spans=N_SPANS,
    )
    p = np.array([0.0, 1e17, 1e18, 5e18, 1e19, 5e19, 1e20, 1e21])
    psi = 2.0 * np.pi**2 * beta2 * SPAN_M * p
    series = np.full_like(p, float(N_SPANS))
    for delta in range(1, N_SPANS):
        series += 2.0 * (N_SPANS - delta) * np.cos(delta * 2.0 * psi)
    # |mu|^2 = |zeta|^2 |nu|^2；单跨时 |nu| = 1，可以直接分离出 |nu|^2。
    single = LinkFunction(
        gamma=link.gamma,
        alpha_power_npm=link.alpha_power_npm,
        beta2_s2_m=beta2,
        span_length_m=SPAN_M,
        n_spans=1,
    )
    nu_squared = np.abs(link.mu(p)) ** 2 / np.abs(single.mu(p)) ** 2
    assert np.allclose(nu_squared, series, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# 2. 论文口径：GN 加权的逐对可见度之和
# ---------------------------------------------------------------------------
def test_sci_gn_block_ratio_is_the_gn_weighted_pairwise_sum(kernel, reference_visibility) -> None:
    """模型算出的 SCI GN 块比值 = 按噪声密度加权的逐对可见度之和。

    这是"论文口径"的定义式。若实现的相位映射、归一化或积分区间有错，两边就对不上。
    """
    weighted, _ = reference_visibility
    reference = _coherent_factor(weighted, N_SPANS)
    table = kernel.sci_kappas(N_SPANS, SPAN_M)
    single = kernel.sci_kappas(1, SPAN_M)
    kappa_one = sum(float(np.sum(v.kappa1)) for v in single.values())
    kappa_many = sum(float(np.sum(v.kappa1)) for v in table.values())
    assert kappa_many / kappa_one == pytest.approx(reference, rel=0.02)


def test_uniform_band_average_is_materially_smaller(reference_visibility) -> None:
    """同一条积分换成均匀（频带平均）权重，累加因子明显变小。

    这就是 ``carena2014`` 与频带平均口径相差约 2.07 dB（对 ``semrau2019``）的来源：
    论文按噪声密度加权，另两者按频带均匀平均。
    """
    weighted, uniform = reference_visibility
    factor_weighted = _coherent_factor(weighted, N_SPANS)
    factor_uniform = _coherent_factor(uniform, N_SPANS)
    # 均匀口径给 buildup ≈ 1.10，论文口径给 ≈ 1.40。
    assert factor_uniform / N_SPANS == pytest.approx(1.10, abs=0.03)
    assert factor_weighted / N_SPANS == pytest.approx(1.40, abs=0.03)
    assert factor_uniform < 0.85 * factor_weighted


# ---------------------------------------------------------------------------
# 3. 归一化：没有藏起来的 1/N
# ---------------------------------------------------------------------------
def test_both_blocks_are_quadratic_without_dispersion() -> None:
    """零色散时 GN 块与 EGN 块的跨数比值都精确等于 ``N^2``。

    ``nu -> N``（完全相干）时链路级 NLI 按 ``N^2`` 增长；比值里若混进 ``1/N``
    或"单跨只计一次"的归一化，这里就会偏。
    """
    case = build_case(CASE_NAME)
    model = make_model("carena2014", case)
    fiber = SMF_100KM.model_copy(
        update={"dispersion_parameter_d": 0.0, "dispersion_slope_s": 0.0}
    )
    zero = model.kernel_for(fiber)
    gn_now, egn_now = zero._sci_block(N_SPANS, SPAN_M)
    gn_ref, egn_ref = zero._sci_block(1, SPAN_M)
    assert gn_now / gn_ref == pytest.approx(float(N_SPANS) ** 2, rel=1e-9)
    assert egn_now / egn_ref == pytest.approx(float(N_SPANS) ** 2, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. 两个口径在什么条件下收敛
# ---------------------------------------------------------------------------
def test_both_conventions_give_full_coherence_without_dispersion() -> None:
    """色散为零时两个模型的 buildup 都等于 ``N``（完全相干），口径差异消失。"""
    case = build_case(CASE_NAME)
    fiber = SMF_100KM.model_copy(
        update={"dispersion_parameter_d": 0.0, "dispersion_slope_s": 0.0}
    )
    for name in ("carena2014", "numerical", "semrau2019"):
        assert _buildup(name, fiber, N_SPANS) == pytest.approx(
            float(N_SPANS), rel=1e-6
        )


def test_both_conventions_give_unit_buildup_for_one_span() -> None:
    """单跨时两个模型的 buildup 都是 1（无跨间相干可言）。"""
    case = build_case(CASE_NAME)
    fiber = build_path(case).fiber_configs[0]
    for name in ("carena2014", "numerical", "semrau2019"):
        assert _buildup(name, fiber, 1) == pytest.approx(1.0, rel=1e-12)


def test_carena_buildup_stays_above_the_band_average_convention() -> None:
    """名义算例上 ``carena2014`` 的 buildup 明显高于频带平均口径的 ``semrau2019``。

    差距的量级（1.65 对 1.02）就是审计结论本身：口径差异，不是实现错误。
    ``carena2014`` 的带内采样从 3 点收敛到 41 点后 buildup 由 1.595 升到 1.649；
    ``numerical`` 改用 NLI 加权口径后由 1.019 升到 1.132。
    """
    case = build_case(CASE_NAME)
    fiber = build_path(case).fiber_configs[0]
    carena = _buildup("carena2014", fiber, N_SPANS)
    numerical = _buildup("numerical", fiber, N_SPANS)
    semrau = _buildup("semrau2019", fiber, N_SPANS)
    assert carena == pytest.approx(1.649, abs=0.01)
    assert semrau == pytest.approx(1.022, abs=0.005)
    # NLI 加权口径：numerical 的 buildup 从 1.019 抬到 1.132。
    assert numerical == pytest.approx(1.132, abs=0.005)
    # carena 的 41 点收敛把它的 buildup 抬到 1.649，两个差距随之重算。
    assert 10.0 * np.log10(carena / semrau) == pytest.approx(2.07, abs=0.06)
    assert 10.0 * np.log10(carena / numerical) == pytest.approx(1.63, abs=0.06)


# ---------------------------------------------------------------------------
# 5. 两个口径差距的分解
# ---------------------------------------------------------------------------
def test_buildup_decomposition(kernel) -> None:
    """把约 2 dB 的差距拆成三段：口径、EGN 修正项、XPM。

    * SCI 的 **GN 项**（论文 Eq. (7) + Eq. (12)）：累加因子 14.03 → buildup 1.403；
      同一积分换成均匀权重是 11.02 → buildup 1.103（``numerical`` 的口径）。
    * SCI 的 **EGN 块**（含 Eq. (8)/(9) 的 4/6 阶修正）：累加因子 20.50 → 2.050。
    * XPM（Eq. (16)-(18)/(23)-(26)）：累加因子 14.26 → 1.426。
    * 两者按噪声占比混合，链路级总量比值 16.49 → buildup 1.649。
    """
    gn_one, egn_one = kernel._sci_block(1, SPAN_M)
    gn_many, egn_many = kernel._sci_block(N_SPANS, SPAN_M)
    assert gn_many / gn_one == pytest.approx(14.034, rel=1e-3)
    assert egn_many / egn_one == pytest.approx(20.500, rel=1e-3)

    case = build_case(CASE_NAME)
    power = launch_profile(case, 0.0)
    gn_out = kernel.gn.span_fiber_output(power, SPAN_M, 1)
    per_span, link_xpm = kernel._xpm_levels(
        N_SPANS, SPAN_M, power, np.asarray(gn_out.xpm_noise_w)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(per_span > 0.0, per_span, 1.0)
        buildup_xpm = np.where(per_span > 0.0, link_xpm / safe, 0.0)
    assert float(np.median(buildup_xpm)) == pytest.approx(14.262, rel=1e-3)

    sci_link, xpm_link = kernel.link_level_nli(power, SPAN_M, N_SPANS)
    sci_one, xpm_one = kernel.link_level_nli(power, SPAN_M, 1)
    ratio = float((sci_link.sum() + xpm_link.sum()) / (sci_one.sum() + xpm_one.sum()))
    assert ratio == pytest.approx(16.494, rel=1e-3)
    assert ratio / N_SPANS == pytest.approx(1.6494, rel=1e-3)
