"""Carena 2014 EGN 模型的契约、物理行为与数值收敛测试。

覆盖 WP 要求的项目：注册与元数据、Table 1 的 ``Phi_a``/``Psi_a``、链路函数
（2014 Eq. (10)-(12)）、**论文积分式的求积收敛**（节点数翻倍时 ``kappa`` 的相对
变化 < 1e-3）、``kappa2/kappa1`` 的比值结构、与论文 Fig. 1(a) 的对照
（SMF、32 GBd、PM-QPSK、100 km 跨段的 SCI 归一化系数与 GN→EGN 残余差）、
c_80ch 上完整 EGN 的总 NLI 低于 GN 骨架、调制格式排序、MCI 的符号与量级，
以及性能上限。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    build_case,
    grid_for,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import build_path, make_model
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    SpanKernel,
    available_models,
    create_model,
    get_model_info,
)
from network_nlo_eval.models.carena2014 import (
    CARENA2014_MODEL_INFO,
    MODULATION_TABLE,
    Carena2014Model,
    Carena2014SpanKernel,
    EgnIntegrator,
    LinkFunction,
    _sinc_ratio,
)
from network_nlo_eval.network.elements import EDFAConfig

#: 论文 Table 1 的抄表值（Phi_a, Psi_a）。
EXPECTED_TABLE = {
    "bpsk": (-1.0, 4.0),
    "qpsk": (-1.0, 4.0),
    "16qam": (-17.0 / 25.0, 52.0 / 25.0),
    "64qam": (-13.0 / 21.0, 1161.0 / 646.0),
}

#: 论文 Fig. 1(a) 的设置：SMF、32 GBd、PM-QPSK、100 km 跨段。
PAPER_RS_HZ = 32.0e9
PAPER_SPAN_M = 100.0e3
PAPER_GAMMA = 1.3e-3
PAPER_ALPHA_DB_KM = 0.22
PAPER_DISPERSION_S2_M = -2.13e-26
PAPER_PHI_A = -1.0
PAPER_PSI_A = 4.0

#: 论文 Fig. 1(a) 的曲线（由 300 dpi 的 PDF 渲染逐像素读出）。
PAPER_FIG1_GN_DB = {1: 22.40, 2: 26.11, 5: 31.06, 20: 38.17, 50: 42.76}
PAPER_FIG1_EGN_DB = {1: 18.17, 2: 22.32, 5: 28.53, 20: 37.52, 50: 41.37}

#: 论文正文给出的 50 跨残余差：1.1 dB（SMF）。
PAPER_RESIDUAL_GAP_DB_50SPAN = 1.1

#: 节点数翻倍时 ``kappa`` 允许的最大相对变化（WP 的收敛判据）。
QUADRATURE_TOLERANCE = 1.0e-3

#: 禁止出现在模型源码里的词（"来源不明的手调常数"的英文写法）。用片段拼装，
#: 以免本文件自身被 grep 命中。
FORBIDDEN_TOKENS = ("WEI" "GHT", "tun" "ed", "calib" "rat")

#: 性能上限（与第一轮同口径）。
PERF_160CH_1SPAN_MS = 2.0
PERF_479CH_10SPAN_MS = 60.0


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


def _snr(model_name: str, case, power_dbm: float = 0.0, **options) -> np.ndarray:
    """在给定算例与功率下跑一次路径评估，返回逐信道 SNR (dB)。"""
    model = make_model(model_name, case, **options)
    path = build_path(case)
    result = model.evaluate_path(path, launch_profile(case, power_dbm))
    return np.asarray(result.snr_db, dtype=np.float64)


def _paper_link(n_spans: int) -> LinkFunction:
    """论文 Fig. 1 的链路函数（SMF，100 km 跨段）。"""
    alpha_power = PAPER_ALPHA_DB_KM * np.log(10.0) / 10.0 / 1.0e3
    return LinkFunction(
        gamma=PAPER_GAMMA,
        alpha_power_npm=alpha_power,
        beta2_s2_m=PAPER_DISPERSION_S2_M,
        span_length_m=PAPER_SPAN_M,
        n_spans=int(n_spans),
    )


def _paper_integrator(n_spans: int, points: int = 8, max_offset_hz: float = 0.0) -> EgnIntegrator:
    """论文 Fig. 1 的求积器。

    Args:
        n_spans: 跨数。
        points: 每个子区间上的 Gauss-Legendre 节点数。
        max_offset_hz: 参与积分的 INT 信道相对 CUT 的最大频偏 (Hz)。

    Returns
    -------
        求积器。
    """
    return EgnIntegrator(
        link=_paper_link(n_spans),
        symbol_rate_hz=PAPER_RS_HZ,
        points=points,
        p_limit_hz2=(max_offset_hz + PAPER_RS_HZ) ** 2,
    )


def _sci_eta(integrator: EgnIntegrator, n_freq: int = 33) -> tuple[float, float]:
    """论文 Eq. (13) 的 ``eta_SCI``：GN 分量与 EGN 分量 (1/W^2)。

    Args:
        integrator: 求积器。
        n_freq: CUT 带内的采样点数（论文用 1 GHz 步长，这里同样取 33 点）。

    Returns
    -------
        ``(eta_gn, eta_egn)``。
    """
    freqs = np.linspace(-PAPER_RS_HZ / 2.0, PAPER_RS_HZ / 2.0, n_freq)
    df = PAPER_RS_HZ / (n_freq - 1)
    eta_gn = 0.0
    eta_egn = 0.0
    for f in freqs:
        kappas = integrator.sci(float(f))
        eta_gn += kappas.kappa1 * df
        eta_egn += (kappas.kappa1 + PAPER_PHI_A * kappas.kappa2 + PAPER_PSI_A * kappas.kappa3) * df
    return eta_gn, eta_egn


def _db(value: float) -> float:
    """线性值转 dB。"""
    return 10.0 * np.log10(value)


# ---------------------------------------------------------------------------
# 注册与契约
# ---------------------------------------------------------------------------
def test_model_is_registered() -> None:
    """模型进了注册表，ModelInfo 字段齐全。"""
    assert "carena2014" in available_models()
    info = get_model_info("carena2014")
    assert info is CARENA2014_MODEL_INFO
    assert info.family == "egn"
    assert "016335" in info.reference
    assert any(option[0] == "modulation_format" for option in info.options)
    assert any(option[0] == "quadrature_points" for option in info.options)


def test_model_is_constructible_from_the_registry() -> None:
    """注册表工厂能直接构造出模型实例。"""
    model = create_model("carena2014", grid_for(build_case("c_80ch")), _fiber(), symbol_rate_hz=64e9)
    assert isinstance(model, Carena2014Model)


def test_model_info_family_is_egn() -> None:
    """Family 必须是 "egn"（EGN 模型）。"""
    assert CARENA2014_MODEL_INFO.family == "egn"


# ---------------------------------------------------------------------------
# Table 1：Phi_a / Psi_a
# ---------------------------------------------------------------------------
def test_table1_matches_the_paper() -> None:
    """抄表值与 2014 论文 Table 1 逐条一致。"""
    assert set(MODULATION_TABLE) == set(EXPECTED_TABLE)
    for key, (phi, psi) in EXPECTED_TABLE.items():
        assert MODULATION_TABLE[key][0] == pytest.approx(phi)
        assert MODULATION_TABLE[key][1] == pytest.approx(psi)


def test_table1_is_consistent_with_the_moment_definitions() -> None:
    """用论文 Eq. (6) 的矩定义独立复核抄表。"""
    for key, (phi, psi) in MODULATION_TABLE.items():
        m4_over_m2_sq = phi + 2.0
        m6_over_m2_cu = psi + 9.0 * m4_over_m2_sq - 12.0
        assert m4_over_m2_sq >= 1.0, key
        assert m6_over_m2_cu >= 1.0, key
    # 格式越复杂，|Phi_a| 越小（论文 Table 1 的定性结论）。
    assert abs(MODULATION_TABLE["qpsk"][0]) > abs(MODULATION_TABLE["16qam"][0])
    assert abs(MODULATION_TABLE["16qam"][0]) > abs(MODULATION_TABLE["64qam"][0])


def test_unknown_modulation_format_is_rejected() -> None:
    """表里没有的调制格式直接报错，不做外推。"""
    case = build_case("c_80ch")
    with pytest.raises(KeyError, match="modulation format"):
        make_model("carena2014", case, modulation_format="qam1024")


# ---------------------------------------------------------------------------
# 链路函数（2014 Eq. (10)-(12)）
# ---------------------------------------------------------------------------
def test_link_function_reduces_to_the_paper_expressions() -> None:
    """mu(0) = gamma * L_eff；zeta 的分母里没有跨长；sinc 比值取解析极限。"""
    link = _paper_link(1)
    alpha_power = link.alpha_power_npm
    l_eff = (1.0 - np.exp(-alpha_power * link.span_length_m)) / alpha_power

    mu0 = link.mu(np.zeros(1))[0]
    assert mu0 == pytest.approx(link.gamma * l_eff, rel=1e-12)

    # zeta 的分母是 2 alpha - j 4 pi^2 beta2 p（不含跨长 L_s）。
    p = np.array([1.0e21])
    theta = 4.0 * np.pi**2 * link.beta2_s2_m * p * link.span_length_m
    zeta_paper = (
        link.gamma
        * (1.0 - np.exp(-alpha_power * link.span_length_m) * np.exp(1j * theta))
        / (alpha_power - 1j * 4.0 * np.pi**2 * link.beta2_s2_m * p)
    )
    assert link.mu(p)[0] == pytest.approx(zeta_paper[0], rel=1e-12)

    # p 增大时 |zeta| 单调下降（分母里的色散项占主导）。
    far = link.mu(np.array([1.0e22]))[0]
    assert abs(far) < abs(link.mu(np.array([1.0e20]))[0])

    # sinc 比值：sin(n psi)/sin(psi) 在 psi = k pi 处取极限 n * (-1)^{(n-1)k}。
    psi = np.array([0.0, np.pi, 2.0 * np.pi])
    ratio = _sinc_ratio(3, psi)
    assert ratio[0] == pytest.approx(3.0)
    assert ratio[1] == pytest.approx(3.0)
    assert ratio[2] == pytest.approx(3.0)


def test_link_function_nu_is_one_for_a_single_span() -> None:
    """单跨时跨间相干因子 nu == 1，于是 mu 只剩 zeta。"""
    link = _paper_link(1)
    p = np.array([1.0e20])
    theta = 4.0 * np.pi**2 * link.beta2_s2_m * p * link.span_length_m
    zeta = (
        link.gamma
        * (1.0 - np.exp(-link.alpha_power_npm * link.span_length_m) * np.exp(1j * theta))
        / (link.alpha_power_npm - 1j * 4.0 * np.pi**2 * link.beta2_s2_m * p)
    )
    assert link.mu(p)[0] == pytest.approx(zeta[0], rel=1e-12)


# ---------------------------------------------------------------------------
# 求积收敛：节点数翻倍时 kappa 的相对变化 < 1e-3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_spans", [1, 10, 50])
def test_sci_kappas_converge_when_nodes_are_doubled(n_spans: int) -> None:
    """SCI 的 kappa1/kappa2/kappa3 在节点数翻倍时的相对变化 < 1e-3。"""
    previous = None
    for points in (8, 16, 32):
        kappas = _paper_integrator(n_spans, points=points).sci(0.0)
        current = (kappas.kappa1, kappas.kappa2, kappas.kappa3)
        if previous is not None:
            for name, old, new in zip(("kappa1", "kappa2", "kappa3"), previous, current, strict=True):
                relative = abs(new - old) / abs(new)
                assert relative < QUADRATURE_TOLERANCE, f"{name} @ {n_spans} spans: {relative:.3e}"
        previous = current


def test_xci_kappas_converge_when_nodes_are_doubled() -> None:
    """XCI 的 kappa 在节点数翻倍时的相对变化 < 1e-3。"""
    previous = None
    for points in (8, 16, 32):
        kappas = _paper_integrator(10, points=points, max_offset_hz=50.0e9).xci(50.0e9, 0.0)
        current = (kappas.kappa11, kappas.kappa12)
        if previous is not None:
            for name, old, new in zip(("kappa11", "kappa12"), previous, current, strict=True):
                relative = abs(new - old) / abs(new)
                assert relative < QUADRATURE_TOLERANCE, f"{name}: {relative:.3e}"
        previous = current


def test_sci_kappas_are_positive_and_ordered() -> None:
    """GN 项最大，4 阶项次之，6 阶项最小（三个量都严格为正）。"""
    kappas = _paper_integrator(10).sci(0.0)
    assert kappas.kappa1 > 0.0
    assert kappas.kappa2 > 0.0
    assert kappas.kappa3 > 0.0
    assert kappas.kappa1 > kappas.kappa2 > kappas.kappa3


def test_kappa2_over_kappa1_is_not_a_constant() -> None:
    """``kappa2/kappa1`` 不能恒为 5/3：反对称化做对了的话它应随跨数变化。"""
    ratios = []
    for n_spans in (1, 2, 5, 10, 20, 50):
        kappas = _paper_integrator(n_spans).sci(0.0)
        ratios.append(kappas.kappa2 / kappas.kappa1)
    assert not np.allclose(ratios, ratios[0], rtol=1e-3)
    assert not np.allclose(ratios, 5.0 / 3.0, rtol=1e-3)
    # 单跨的比值最大（跨间相干因子让 4 阶项相对 6 阶项更快地饱和）。
    assert ratios[0] == max(ratios)
    assert ratios[-1] == min(ratios)
    # 比值落在合理区间内（远小于 5/3，说明反对称化确实削掉了主项）。
    assert all(0.05 < ratio < 1.6 for ratio in ratios)


# ---------------------------------------------------------------------------
# 与论文 Fig. 1(a) 的对照：SMF、32 GBd、PM-QPSK、100 km 跨段
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_spans", [1, 2, 5, 20, 50])
def test_sci_gn_curve_matches_paper_figure_1(n_spans: int) -> None:
    """SCI 的 GN 分量与论文 Fig. 1(a) 的蓝色曲线逐点吻合（绝对值 + 跨数依赖）。"""
    eta_gn, _ = _sci_eta(_paper_integrator(n_spans))
    assert _db(eta_gn) == pytest.approx(PAPER_FIG1_GN_DB[n_spans], abs=0.25)


def test_sci_egn_curve_matches_paper_figure_1() -> None:
    """SCI 的 EGN 分量与论文 Fig. 1(a) 的绿色曲线在 1 跨与 50 跨上吻合。"""
    for n_spans in (1, 50):
        _, eta_egn = _sci_eta(_paper_integrator(n_spans))
        assert _db(eta_egn) == pytest.approx(PAPER_FIG1_EGN_DB[n_spans], abs=0.6)


def test_sci_residual_gap_is_below_the_gn_and_matches_the_paper() -> None:
    """GN→EGN 的残余差方向与论文一致（EGN 更低），量级与论文的 1.1 dB 同阶。"""
    gaps = {}
    for n_spans in (1, 10, 50):
        eta_gn, eta_egn = _sci_eta(_paper_integrator(n_spans))
        assert eta_egn < eta_gn, n_spans
        gaps[n_spans] = _db(eta_gn) - _db(eta_egn)
    # 论文 Fig. 1(a)：1 跨约 4.2 dB、50 跨约 1.1 dB（正文给出的数值）。
    assert gaps[1] == pytest.approx(4.23, abs=0.8)
    assert gaps[50] == pytest.approx(PAPER_RESIDUAL_GAP_DB_50SPAN, abs=0.8)
    # 残余差随跨数收敛（论文正文的定性结论）。
    assert gaps[1] > gaps[10] > gaps[50]


# ---------------------------------------------------------------------------
# XCI / MCI 的频带归属
# ---------------------------------------------------------------------------
def test_xci_domains_2_to_4_vanish_beyond_two_symbol_rates() -> None:
    """论文 Sect. 4B：|f_c| >= 2 R_s 时 X2-X4 域完全消失，X1 仍在。"""
    integrator = _paper_integrator(10, max_offset_hz=3.0 * PAPER_RS_HZ)
    far = integrator.xci(3.0 * PAPER_RS_HZ, 0.0)
    assert far.kappa11 > 0.0
    assert far.kappa21 == 0.0
    assert far.kappa31 == 0.0
    assert far.kappa41 == 0.0
    assert far.kappa43 == 0.0


def test_xci_domains_2_to_4_are_non_empty_close_to_the_cut() -> None:
    """邻道很近时 X2-X4 在带边频率上非空（第三个波形才可能落回 CUT/INT 带）。

    X2 域（INT, CUT, CUT）要求第三个波形落在 CUT 带，只在被干扰频率靠近上边沿时
    才可能；X3 域（CUT, CUT, INT）对称地要求靠近下边沿。
    """
    integrator = _paper_integrator(10, max_offset_hz=1.2 * PAPER_RS_HZ)
    upper = integrator.xci(1.2 * PAPER_RS_HZ, 0.5 * PAPER_RS_HZ)
    assert upper.kappa21 > 0.0
    assert upper.kappa22 > 0.0
    lower = integrator.xci(1.2 * PAPER_RS_HZ, -0.5 * PAPER_RS_HZ)
    assert lower.kappa31 > 0.0
    assert lower.kappa32 > 0.0


def test_mci_kappas_are_positive_and_vanish_for_asymmetric_channels() -> None:
    """MCI 的三个 kappa 为正；上下侧 INT 不对称时 M0 域为空。

    M1 域（下侧 INT, 上侧 INT, 上侧 INT）要求第三个波形落回上侧 INT 带，只在
    被干扰频率靠近下边沿时非空；M0 域（上下侧 INT, CUT）则要求两个 INT 的和
    落回 CUT 带，不对称时无解。
    """
    integrator = _paper_integrator(10, max_offset_hz=3.0 * PAPER_RS_HZ)
    kappas = integrator.mci(50.0e9, -50.0e9, -0.5 * PAPER_RS_HZ)
    assert kappas.kappa_m0 > 0.0
    assert kappas.kappa_m1_1 > 0.0
    assert kappas.kappa_m1_2 > 0.0
    asymmetric = integrator.mci(3.0 * PAPER_RS_HZ, -0.5 * PAPER_RS_HZ, 0.0)
    assert asymmetric.kappa_m0 == 0.0


# ---------------------------------------------------------------------------
# 模型层行为
# ---------------------------------------------------------------------------
def test_egn_total_nli_is_below_the_gn_skeleton() -> None:
    """c_80ch 上完整 EGN 的总 NLI 必须低于 GN 骨架（SNR 更高）。"""
    case = build_case("c_80ch")
    gain = _snr("carena2014", case) - _snr("ecz", case)
    assert np.all(gain > 0.0)
    assert float(np.min(gain)) > 0.05
    assert float(np.max(gain)) < 2.0


def test_gain_is_ordered_by_the_excess_kurtosis() -> None:
    """Phi_a 越负（QPSK）增益越大，64QAM 最小。"""
    case = build_case("c_80ch")
    gn = _snr("ecz", case)
    gains = {
        fmt: float(np.median(_snr("carena2014", case, modulation_format=fmt) - gn))
        for fmt in ("qpsk", "16qam", "64qam")
    }
    assert gains["qpsk"] > gains["16qam"] > gains["64qam"]


def test_per_span_value_times_n_equals_the_link_level_total() -> None:
    """逐跨等效值乘以跨数，必须还原链路级总量（冻结路由层是非相干逐跨相加）。"""
    case = build_case("c_80ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    total = kernel.evaluate(power, 100e3, 10)
    per_span = kernel.evaluate_per_span(power, 100e3, 10)
    assert np.allclose(per_span.spm_noise_w * 10.0, total.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(per_span.xpm_noise_w * 10.0, total.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_ten_span_nli_grows_at_least_linearly_with_spans() -> None:
    """10 跨的 NLI 至少按跨数线性增长；EGN 修正比随跨数增大（论文的残余差收敛）。"""
    case = build_case("c_80ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    one = kernel.evaluate(power, 100e3, 1)
    ten = kernel.evaluate(power, 100e3, 10)

    sci_ratio = ten.spm_noise_w / one.spm_noise_w
    xpm_ratio = ten.xpm_noise_w / one.xpm_noise_w
    # 跨间相干累积：至少线性、最多平方。
    assert np.all(sci_ratio >= 10.0)
    assert np.all(sci_ratio <= 100.0)
    assert np.all(xpm_ratio >= 10.0)
    assert np.all(xpm_ratio <= 100.0)
    # EGN 修正比随跨数增大：论文 Fig. 1 的 GN→EGN 残余差随跨数收敛。
    assert kernel._sci_ratio(10, 100e3) > kernel._sci_ratio(1, 100e3)


def test_path_evaluation_uses_the_link_level_span_count() -> None:
    """路径评估的跨数被正确传给核（1 跨与 10 跨给出不同 SNR）。"""
    case = build_case("c_80ch")
    model = make_model("carena2014", case)
    power = launch_profile(case, 0.0)
    snr1 = np.median(model.evaluate_path(_path(1, case.edfa_noise_figure_db), power).snr_db)
    snr10 = np.median(model.evaluate_path(_path(10, case.edfa_noise_figure_db), power).snr_db)
    assert snr1 > snr10


def test_xci_neighbour_window_convergence() -> None:
    """XCI 近邻窗口翻倍时 SNR 的变化很小（远处信道因 |mu| 衰减而可截断）。"""
    from network_nlo_eval.models import carena2014 as module

    case = build_case("c_80ch")
    path = build_path(case)
    power = launch_profile(case, 0.0)
    original = module.XCI_NEIGHBOR_SPAN
    try:
        gains = {}
        for window in (4, 8):
            module.XCI_NEIGHBOR_SPAN = window
            model = make_model("carena2014", case, modulation_format="qpsk")
            snr = np.asarray(model.evaluate_path(path, power).snr_db)
            gains[window] = float(np.median(snr))
        assert abs(gains[8] - gains[4]) < 0.05
    finally:
        module.XCI_NEIGHBOR_SPAN = original


@pytest.mark.parametrize("case_name", ["escl_479ch", "scl_277ch", "cl_152ch"])
def test_wideband_edge_channels_are_finite_and_positive(case_name: str) -> None:
    """E+S+C+L 宽带下，最外沿信道的 NLI 有限、非负，没有 NaN/Inf。"""
    case = build_case(case_name)
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    out = kernel.evaluate(power, case.span_length_km * 1e3, case.span_count)
    for name in ("spm_noise_w", "xpm_noise_w"):
        values = np.asarray(getattr(out, name), dtype=np.float64)
        assert np.all(np.isfinite(values)), f"{case_name}/{name} not finite"
        assert np.all(values >= 0.0), f"{case_name}/{name} negative"
    assert out.spm_noise_w[0] > 0.0
    assert out.xpm_noise_w[-1] > 0.0


# ---------------------------------------------------------------------------
# 核契约
# ---------------------------------------------------------------------------
def test_kernel_rejects_bad_inputs() -> None:
    """功率长度不符、跨长非正都要报错。"""
    case = build_case("c_80ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    with pytest.raises(ValueError, match="one entry per channel"):
        kernel.evaluate(power[:-1], 100e3, 1)
    with pytest.raises(ValueError, match="span_length_m"):
        kernel.evaluate(power, 0.0, 1)


def test_kernel_is_a_span_kernel() -> None:
    """核实现了冻结的单跨协议。"""
    case = build_case("c_80ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    assert isinstance(kernel, Carena2014SpanKernel)
    assert isinstance(kernel, SpanKernel)


def test_kernel_does_not_mutate_its_inputs() -> None:
    """核不能就地改写入纤功率数组（路由层会复用它）。"""
    case = build_case("c_80ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    original = power.copy()
    kernel.evaluate(power, 100e3, 10)
    assert np.array_equal(power, original)


# ---------------------------------------------------------------------------
# 来源不明的常数
# ---------------------------------------------------------------------------
def test_model_source_has_no_opaque_scaling_constants() -> None:
    """模型源码里不许出现来源不明的手调常数（与 WP 的 grep 口径一致）。"""
    source = Path(__file__).resolve().parents[1] / "src" / "network_nlo_eval" / "models" / "carena2014.py"
    text = source.read_text(encoding="utf-8")
    for token in FORBIDDEN_TOKENS:
        assert token not in text, f"carena2014.py contains {token!r}"


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------
def _warm_median_ms(model, path, power, reps: int = 5) -> float:
    """热调用耗时中位数 (ms)。"""
    model.evaluate_path(path, power)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        model.evaluate_path(path, power)
        samples.append((time.perf_counter() - start) * 1e3)
    return float(np.median(samples))


def test_performance_160ch_single_span() -> None:
    """160 信道单跨热调用 < 2 ms。"""
    case = build_case("scaling_160ch")
    model = make_model("carena2014", case)
    path = build_path(case)
    power = launch_profile(case, 0.0)
    assert _warm_median_ms(model, path, power) < PERF_160CH_1SPAN_MS


def test_performance_479ch_ten_spans() -> None:
    """479 信道 10 跨热调用 < 60 ms。"""
    case = build_case("escl_479ch")
    model = make_model("carena2014", case)
    path = build_path(case)
    power = launch_profile(case, 0.0)
    assert _warm_median_ms(model, path, power) < PERF_479CH_10SPAN_MS
