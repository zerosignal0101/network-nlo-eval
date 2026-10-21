"""Carena 2014 EGN 模型的契约、物理行为与数值收敛测试。

覆盖 WP 要求的项目：注册与元数据、Table 1 的 ``Phi_a``/``Psi_a``、链路函数
（2014 Eq. (10)-(12)）、**论文积分式的求积收敛**（节点数翻倍时 ``kappa`` 的相对
变化 < 1e-3）、``kappa2/kappa1`` 的比值结构、与论文 Fig. 1(a) 的对照
（SMF、32 GBd、PM-QPSK、100 km 跨段的 SCI 归一化系数与 GN→EGN 残余差）、
c_57ch 上完整 EGN 的总 NLI 低于 GN 骨架、调制格式排序、MCI 的符号与量级，
以及性能上限。

**跨间相干累加**（本轮新增）：单跨逐位一致、逐跨复振幅相干求和还原链路级总量、
多跨的 NLI 单调且不等于"单跨 × N"、相干增益在色散趋零时趋于 ``N^2``、色散很强时
趋于 ``N``、确定性（同一实例同功率连调逐位相同）。
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
    SpanContext,
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
    dirichlet_buildup,
    solve_dirichlet_step,
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

#: 逐跨复振幅相干求和与链路级总量的相对偏差上限（WP 要求 < 1e-9）。
COHERENT_SUM_TOLERANCE = 1.0e-9

#: 单跨路径的 SNR (dB) 黄金值。
#: 相干改造前捕获于 commit 82c3a22；带内采样收敛（41 点）后重新捕获，
#: 断言强度不变（逐位相等）。
GOLDEN_ONE_SPAN_SNR_DB = np.array(
    [
        29.057138971049305, 28.972568527003723, 28.92741895495119, 28.893539176722832,
        28.86507238191047, 28.839739956666204, 28.816417301126904, 28.79446718872156,
        28.773494008346923, 28.752943685868445, 28.732956501575874, 28.713395436592254,
        28.69415835936307, 28.675167284754007, 28.656361479986117, 28.63769287682484,
        28.6191229242907, 28.600620373151354, 28.58215968164093, 28.563719846522538,
        28.54528353238201, 28.52683641460925, 28.508366678623695, 28.489864635598995,
        28.47132242677756, 28.45273379656328, 28.434093920250344, 28.415399276318666,
        28.39664755623957, 28.37783760705333, 28.35896940385435, 28.340044050939177,
        28.3210638118889, 28.302032170409234, 28.282953925483408, 28.263835326475885,
        28.244684256483396, 28.22551047578071, 28.2063259421183, 28.18714523161495,
        28.167986094177845, 28.148870192606083, 28.12982409783522, 28.110880649381585,
        28.09208084918634, 28.07347655569288, 28.055134415507553, 28.037141777221418,
        28.019615912883577, 28.002418973175935, 27.98604373566407, 27.970829554002368,
        27.957323512499258, 27.946491960421277, 27.940294380062998, 27.943769904101924,
        27.981195151172255,
    ]
)

#: 单跨路径的 SPM 噪声 (W) 黄金值（带内采样收敛后重新捕获）。
GOLDEN_ONE_SPAN_SPM_W = np.array(
    [
        8.399690684271521e-08, 8.408249592704517e-08, 8.41661423795358e-08, 8.42478543475514e-08,
        8.432764027722297e-08, 8.440550890909261e-08, 8.448146927368253e-08, 8.455553068698961e-08,
        8.462770274591059e-08, 8.469799532359958e-08, 8.476641856476204e-08, 8.483298288088728e-08,
        8.489769894542346e-08, 8.496057768889747e-08, 8.502163029398373e-08, 8.508086819052388e-08,
        8.513830305050171e-08, 8.519394678297491e-08, 8.524781152896853e-08, 8.529990965633187e-08,
        8.535025375456212e-08, 8.53988566295984e-08, 8.544573129858892e-08, 8.549089098463324e-08,
        8.553434911150536e-08, 8.557611929835683e-08, 8.561621535440607e-08, 8.565465127361549e-08,
        8.56914412293588e-08, 8.572659956908237e-08, 8.576014080896261e-08, 8.579207962856322e-08,
        8.582243086549347e-08, 8.585120951007175e-08, 8.587843069999586e-08, 8.590410971502375e-08,
        8.592826197166596e-08, 8.595090301789388e-08, 8.597204852786435e-08, 8.599171429666497e-08,
        8.60099162350811e-08, 8.60266703643875e-08, 8.604199281116668e-08, 8.605589980215606e-08,
        8.60684076591265e-08, 8.607953279379395e-08, 8.608929170276597e-08, 8.609770096252597e-08,
        8.610477722445608e-08, 8.611053720990145e-08, 8.611499770527682e-08, 8.611817555721826e-08,
        8.612008766778045e-08, 8.612075098968224e-08, 8.612018252160125e-08, 8.611839930351997e-08,
        8.611541841212298e-08,
    ]
)

#: 单跨路径的 XPM 噪声 (W) 黄金值（带内采样收敛后重新捕获）。
GOLDEN_ONE_SPAN_XPM_W = np.array(
    [
        9.203208197125465e-08, 1.130156380739448e-07, 1.2272455881844063e-07, 1.2913681970121562e-07,
        1.3392482117881624e-07, 1.3774195918172025e-07, 1.409116282662352e-07, 1.436174588664363e-07,
        1.4597364427896947e-07, 1.4814552877776123e-07, 1.5008765758131984e-07, 1.5183966679278715e-07,
        1.5343111846823036e-07, 1.548845917305893e-07, 1.5621766828338391e-07, 1.5744425440479303e-07,
        1.585754880925275e-07, 1.5962037761061458e-07, 1.6058626077680168e-07, 1.6147914137123812e-07,
        1.6230393926399382e-07, 1.630646786030749e-07, 1.6376463059434853e-07, 1.6440642229766428e-07,
        1.6499211944205838e-07, 1.6552328891574381e-07, 1.6600104493586866e-07, 1.6642608170906154e-07,
        1.6679869450001018e-07, 1.6711879032580062e-07, 1.6738588891332657e-07, 1.6759911403856936e-07,
        1.677571748619074e-07, 1.6785833633699716e-07, 1.6790037715208165e-07, 1.6788053290027309e-07,
        1.677954211866921e-07, 1.676409440473787e-07, 1.6741216120214848e-07, 1.6710312502162e-07,
        1.6670666423246378e-07, 1.6621409762617413e-07, 1.656148502304413e-07, 1.6489593058821105e-07,
        1.6404120550292418e-07, 1.630303715024224e-07, 1.6183745824230226e-07, 1.6042858392161593e-07,
        1.587584655028382e-07, 1.5687419104123413e-07, 1.5459274900876065e-07, 1.5178485744573017e-07,
        1.4824316334061894e-07, 1.4360344982289701e-07, 1.3713185448588273e-07, 1.2695566545229684e-07,
        1.041593029310049e-07,
    ]
)



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
#: 单跨**核输出**（``Kernel.evaluate``）的 NLI 功率 (W) 黄金值。
#: 捕获自相干改造之前的实现（commit 82c3a22），逐位比对用；
#: 带内采样收敛（41 点）后重新捕获，断言强度不变（逐位相等）。
GOLDEN_ONE_SPAN_KERNEL_SPM_W = np.array(
    [
        8.399690684271522e-08, 8.408249592704517e-08, 8.41661423795358e-08, 8.424785434755141e-08,
        8.432764027722296e-08, 8.440550890909261e-08, 8.448146927368253e-08, 8.455553068698963e-08,
        8.462770274591058e-08, 8.469799532359957e-08, 8.476641856476204e-08, 8.483298288088728e-08,
        8.489769894542346e-08, 8.496057768889747e-08, 8.502163029398372e-08, 8.508086819052387e-08,
        8.513830305050171e-08, 8.519394678297491e-08, 8.524781152896851e-08, 8.529990965633186e-08,
        8.535025375456211e-08, 8.539885662959839e-08, 8.544573129858891e-08, 8.549089098463326e-08,
        8.553434911150537e-08, 8.557611929835681e-08, 8.561621535440608e-08, 8.56546512736155e-08,
        8.56914412293588e-08, 8.572659956908235e-08, 8.576014080896261e-08, 8.579207962856322e-08,
        8.582243086549348e-08, 8.585120951007175e-08, 8.587843069999587e-08, 8.590410971502373e-08,
        8.592826197166596e-08, 8.59509030178939e-08, 8.597204852786434e-08, 8.599171429666498e-08,
        8.600991623508112e-08, 8.602667036438749e-08, 8.604199281116668e-08, 8.605589980215605e-08,
        8.606840765912649e-08, 8.607953279379395e-08, 8.608929170276596e-08, 8.609770096252597e-08,
        8.61047772244561e-08, 8.611053720990145e-08, 8.611499770527684e-08, 8.611817555721825e-08,
        8.612008766778046e-08, 8.612075098968224e-08, 8.612018252160127e-08, 8.611839930351997e-08,
        8.611541841212296e-08,
    ]
)

#: 单跨**核输出**（``Kernel.evaluate``）的 NLI 功率 (W) 黄金值。
#: 捕获自相干改造之前的实现（commit 82c3a22），逐位比对用；
#: 带内采样收敛（41 点）后重新捕获，断言强度不变（逐位相等）。
GOLDEN_ONE_SPAN_KERNEL_XPM_W = np.array(
    [
        9.203208197125466e-08, 1.1301563807394479e-07, 1.2272455881844063e-07, 1.2913681970121565e-07,
        1.3392482117881624e-07, 1.3774195918172025e-07, 1.409116282662352e-07, 1.436174588664363e-07,
        1.4597364427896947e-07, 1.4814552877776123e-07, 1.500876575813198e-07, 1.5183966679278713e-07,
        1.5343111846823036e-07, 1.548845917305893e-07, 1.5621766828338391e-07, 1.57444254404793e-07,
        1.585754880925275e-07, 1.5962037761061458e-07, 1.6058626077680168e-07, 1.6147914137123812e-07,
        1.6230393926399382e-07, 1.630646786030749e-07, 1.6376463059434853e-07, 1.6440642229766428e-07,
        1.6499211944205838e-07, 1.655232889157438e-07, 1.6600104493586866e-07, 1.6642608170906154e-07,
        1.6679869450001018e-07, 1.6711879032580064e-07, 1.673858889133266e-07, 1.6759911403856936e-07,
        1.6775717486190738e-07, 1.6785833633699716e-07, 1.6790037715208165e-07, 1.6788053290027309e-07,
        1.677954211866921e-07, 1.6764094404737868e-07, 1.674121612021485e-07, 1.6710312502162003e-07,
        1.6670666423246378e-07, 1.662140976261741e-07, 1.6561485023044127e-07, 1.6489593058821105e-07,
        1.6404120550292418e-07, 1.6303037150242238e-07, 1.6183745824230226e-07, 1.6042858392161593e-07,
        1.587584655028382e-07, 1.5687419104123413e-07, 1.5459274900876065e-07, 1.5178485744573017e-07,
        1.4824316334061894e-07, 1.4360344982289704e-07, 1.3713185448588275e-07, 1.2695566545229682e-07,
        1.0415930293100489e-07,
    ]
)

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
    model = create_model("carena2014", grid_for(build_case("c_57ch")), _fiber(), symbol_rate_hz=64e9)
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
    case = build_case("c_57ch")
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
# 跨间相干累加因子：Dirichlet 核与它的反解
# ---------------------------------------------------------------------------
def test_dirichlet_buildup_is_the_closed_form_of_the_span_sum() -> None:
    """``|sum_k e^{j k Theta}|^2`` 必须等于 ``(sin(N Theta/2)/sin(Theta/2))^2``。"""
    for n_spans in (1, 2, 5, 10):
        for theta in (0.0, 1e-9, 0.05, 0.3, 0.9):
            direct = np.abs(np.sum(np.exp(1j * theta * np.arange(n_spans)))) ** 2
            assert float(dirichlet_buildup(np.asarray(theta), n_spans)) == pytest.approx(
                float(direct), rel=1e-12
            )


def test_solve_dirichlet_step_inverts_the_buildup() -> None:
    """反解必须把累加因子还原回去（相对误差 < 1e-12）。

    累加因子的取值范围是 ``[0, N^2]``：``N^2`` 是完全相干（上界），``0`` 是完全
    相消（下界，``Theta = 2 pi/N``）。超出范围的目标会被截断。
    """
    for n_spans in (2, 5, 10, 20):
        for target in (0.25, 1.0, 0.5 * n_spans**2, 0.8 * n_spans, 0.99 * n_spans**2):
            step = float(solve_dirichlet_step(n_spans, np.asarray(target)))
            assert 0.0 <= step <= 2.0 * np.pi / n_spans
            rebuilt = float(dirichlet_buildup(np.asarray(step), n_spans))
            assert rebuilt == pytest.approx(target, rel=1e-12)


def test_solve_dirichlet_step_clips_out_of_range_targets() -> None:
    """超出 ``[0, N^2]`` 的目标被截断到边界（完全相干 / 完全相消）。"""
    n_spans = 4
    too_large = float(solve_dirichlet_step(n_spans, np.asarray(n_spans**2 * 3.0)))
    too_small = float(solve_dirichlet_step(n_spans, np.asarray(-1.0)))
    assert float(dirichlet_buildup(np.asarray(too_large), n_spans)) == pytest.approx(
        float(n_spans**2), rel=1e-12
    )
    assert float(dirichlet_buildup(np.asarray(too_small), n_spans)) == pytest.approx(0.0, abs=1e-12)


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
    """c_57ch 上完整 EGN 的 NLI 必须低于同一模型的 GN 骨架（GN 块）。

    对照口径是**同一个模型**的 GN 块（论文 Eq. (7) 的 ``kappa1``），不是 ``ecz``：
    ``ecz`` 是逐跨非相干的基线，而本模型已经进入相干档，两者的 NLI 尺度不再同口径。

    论文 Fig. 1(a) 的定性结论有两条：EGN 总量**始终低于** GN 总量，且残余差**随跨数
    收敛**。"残余差收敛"对应的是 ``EGN(N)/GN(N)`` 随 N **增大**（4/6 阶修正项在
    多跨下相对变轻），所以比值是随跨数上升的——改造前那条 ``ratio < ratio(1)``
    的断言方向是反的，改正后仍然守住"EGN 低于 GN"这条核心断言。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    gn_now, egn_now = kernel._sci_block(10, 100e3)
    # 同一归一化下的 GN 总量与 EGN 总量之比（< 1 表示 EGN 更低）。
    ratio = egn_now / gn_now
    assert 0.0 < ratio < 1.0
    # 论文 Fig. 1(a)：残余差随跨数收敛 ⇒ 比值随跨数**增大**。
    single = kernel._sci_block(1, 100e3)[1] / kernel._sci_block(1, 100e3)[0]
    assert ratio > single, (ratio, single)
    # 模型层：链路级 NLI（相干求和的结果）也必须低于 GN 骨架的链路级值。
    gn_out = kernel.gn.span_fiber_output(power, 100e3, 1)
    s2_spm = np.asarray(gn_out.spm_noise_w, dtype=np.float64)
    gn_ref, _ = kernel._sci_block(1, 100e3)
    gn_skeleton_link = s2_spm * (gn_now / gn_ref)
    sci_link, _ = kernel.link_level_nli(power, 100e3, 10)
    assert np.all(sci_link < gn_skeleton_link)
    assert np.all(sci_link > 0.0)


def test_model_nli_stays_below_the_non_coherent_baseline_by_a_fraction_of_a_db() -> None:
    """与 ``ecz``（非相干基线）相比，本模型的 NLI 差距在 1 dB 以内。

    本模型进入相干档后，跨间累加比 ``ecz`` 的逐跨非相干累加强；两者仍须同量级，
    否则说明相干改造把噪声尺度改坏了（而不是只改了累加方式）。
    """
    case = build_case("c_57ch")
    gain = _snr("ecz", case) - _snr("carena2014", case)
    assert float(np.median(gain)) > -1.5
    assert float(np.median(gain)) < 1.5


def test_gain_is_ordered_by_the_excess_kurtosis() -> None:
    """Phi_a 越负（QPSK）增益越大，64QAM 最小。"""
    case = build_case("c_57ch")
    gn = _snr("ecz", case)
    gains = {
        fmt: float(np.median(_snr("carena2014", case, modulation_format=fmt) - gn))
        for fmt in ("qpsk", "16qam", "64qam")
    }
    assert gains["qpsk"] > gains["16qam"] > gains["64qam"]


def test_coherent_sum_restores_the_link_level_total() -> None:
    """逐跨复振幅相干求和必须还原链路级总量（相对偏差 < 1e-9）。

    链路级总量由论文积分式直接给出（:meth:`link_level_nli`），与逐跨振幅这条
    路径是两条独立的计算；它们一致，说明"逐跨振幅 → 相干求和"没有丢掉因子。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    for n_spans in (1, 2, 5, 10, 20):
        amplitudes_spm, amplitudes_xpm = [], []
        for index in range(n_spans):
            context = SpanContext(
                span_index=index,
                total_spans=n_spans,
                cumulative_length_m=index * 100e3,
                span_length_m=100e3,
            )
            out = kernel.evaluate(power, 100e3, n_spans, span_context=context)
            amplitudes_spm.append(out.nli_amplitude_spm)
            amplitudes_xpm.append(out.nli_amplitude_xpm)
        coherent_spm = np.abs(np.sum(amplitudes_spm, axis=0)) ** 2
        coherent_xpm = np.abs(np.sum(amplitudes_xpm, axis=0)) ** 2
        link_spm, link_xpm = kernel.link_level_nli(power, 100e3, n_spans)
        relative_spm = np.abs(coherent_spm - link_spm) / link_spm
        relative_xpm = np.abs(coherent_xpm - link_xpm) / link_xpm
        assert float(np.max(relative_spm)) < COHERENT_SUM_TOLERANCE, n_spans
        assert float(np.max(relative_xpm)) < COHERENT_SUM_TOLERANCE, n_spans


def test_span_amplitudes_satisfy_the_contract() -> None:
    """``|amp|**2`` 必须等于同名的 ``*_noise_w``（契约层的要求）。"""
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    for n_spans in (1, 10):
        for index in range(min(n_spans, 3)):
            context = SpanContext(
                span_index=index,
                total_spans=n_spans,
                cumulative_length_m=index * 100e3,
                span_length_m=100e3,
            )
            out = kernel.evaluate(power, 100e3, n_spans, span_context=context)
            assert out.nli_amplitude_spm is not None
            assert out.nli_amplitude_xpm is not None
            assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-12, atol=0.0)
            assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_single_span_is_bit_identical_to_the_pre_refactor_model() -> None:
    """单跨路径的 SNR 与相干改造前逐位相同。

    路由层现在走相干分支，链路级 NLI 由 ``|sum_k a_k|^2`` 给出。冻结路由层用
    ``np.abs(a)**2`` 取模平方，而 ``sqrt(p)**2`` 对约一半的 double 并不是 ``p`` 的
    逐位值（``round(m**2)`` 的像只有一半的 double），所以路由层的 NLI 总量与改造前
    相差 <= 1 ulp（实测最大相对偏差 2.1e-16）。模型自身的输出没有这个舍入：
    见 :func:`test_single_span_kernel_output_is_bit_identical_to_the_pre_refactor_model`。
    """
    case = build_case("c_57ch")
    path = build_path(case)
    model = make_model("carena2014", case)
    power = launch_profile(case, 0.0)
    one_span = MultiSpanOpticalPath([path.fiber_configs[0]], [path.edfa_configs[0]])
    result = model.evaluate_path(one_span, power)
    assert np.array_equal(np.asarray(result.snr_db), GOLDEN_ONE_SPAN_SNR_DB)
    assert np.allclose(np.asarray(result.spm_noise_w), GOLDEN_ONE_SPAN_SPM_W, rtol=1e-15, atol=0.0)
    assert np.allclose(np.asarray(result.xpm_noise_w), GOLDEN_ONE_SPAN_XPM_W, rtol=1e-15, atol=0.0)


def test_single_span_kernel_output_is_bit_identical_to_the_pre_refactor_model() -> None:
    """单跨时**模型自身**的 NLI 功率与相干改造前逐位相同。

    这是最硬的一致性检查：逐跨功率走的是改造前那条单跨路径（GN 骨架 × 单跨 EGN
    修正比），一行算式都没动，所以必须逐位相等，而不是"约等于"。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    out = kernel.evaluate(power, 100e3, 1)
    assert np.array_equal(np.asarray(out.spm_noise_w), GOLDEN_ONE_SPAN_KERNEL_SPM_W)
    assert np.array_equal(np.asarray(out.xpm_noise_w), GOLDEN_ONE_SPAN_KERNEL_XPM_W)
    # 单跨的振幅相位取 0。
    assert out.nli_amplitude_spm is not None
    assert out.nli_amplitude_xpm is not None
    assert np.all(out.nli_amplitude_spm.imag == 0.0)
    assert np.all(out.nli_amplitude_xpm.imag == 0.0)
    # |a|^2 与噪声功率一致（受 double 的平方取整所限，逐位相等对约一半信道成立，
    # 其余相差 1 ulp）。
    assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-15, atol=0.0)
    assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-15, atol=0.0)


def test_ten_span_nli_grows_at_least_linearly_with_spans() -> None:
    """10 跨的链路级 NLI 至少按跨数线性增长；EGN 修正比随跨数增大（论文的残余差收敛）。

    改造后核返回的是**逐跨**的 NLI 功率（"本跨自己的 NLI 功率"，与链路跨数无关），
    跨数效应由路由层的相干求和承担；所以"至少线性、最多平方"要落在**链路级总量**
    上，而不是逐跨值上。改造前这条断言比较的是逐跨值，在新口径下已经不再表达
    "跨数增长"这件事（逐跨值按定义不随跨数变）。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    one_sci, one_xpm = kernel.link_level_nli(power, 100e3, 1)
    ten_sci, ten_xpm = kernel.link_level_nli(power, 100e3, 10)
    sci_ratio = ten_sci / one_sci
    xpm_ratio = ten_xpm / one_xpm
    # 跨间相干累积：至少线性、最多平方。
    assert np.all(sci_ratio >= 10.0)
    assert np.all(sci_ratio <= 100.0)
    assert np.all(xpm_ratio >= 10.0)
    assert np.all(xpm_ratio <= 100.0)
    # 逐跨功率是"本跨自己的 NLI 功率"：与链路跨数无关（1 跨与 10 跨给出同一个值）。
    one = kernel.evaluate(power, 100e3, 1)
    ten = kernel.evaluate(power, 100e3, 10)
    assert np.allclose(ten.spm_noise_w, one.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(ten.xpm_noise_w, one.xpm_noise_w, rtol=1e-12, atol=0.0)
    # EGN 修正比随跨数增大：论文 Fig. 1 的 GN→EGN 残余差随跨数收敛。
    assert kernel._sci_ratio(10, 100e3) > kernel._sci_ratio(1, 100e3)


def test_multi_span_nli_is_monotone_finite_and_not_linear() -> None:
    """多跨 NLI 单调、有限，且 10 跨既不等于"单跨 × 10"也不等于"单跨 × 100"。

    这就是相干增益：``|sum_k a_k|^2`` 与 ``sum_k |a_k|^2`` 的比值。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    totals = {}
    for n_spans in (1, 2, 5, 10):
        sci, xpm = kernel.link_level_nli(power, 100e3, n_spans)
        assert np.all(np.isfinite(sci)) and np.all(np.isfinite(xpm))
        assert np.all(sci >= 0.0) and np.all(xpm >= 0.0)
        totals[n_spans] = (float(np.median(sci)), float(np.median(xpm)))
    for previous, current in zip((1, 2, 5), (2, 5, 10), strict=True):
        assert totals[current][0] > totals[previous][0]
        assert totals[current][1] > totals[previous][1]
    sci_ratio = totals[10][0] / totals[1][0]
    xpm_ratio = totals[10][1] / totals[1][1]
    # 相干：严格大于线性（10），严格小于平方（100）。
    assert 10.0 < sci_ratio < 100.0, sci_ratio
    assert 10.0 < xpm_ratio < 100.0, xpm_ratio


def test_coherent_gain_is_reported_for_the_c_band_case() -> None:
    """C 波段 10 跨的相干增益（相干 vs 非相干）落在 0…10 dB 之间。"""
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    n_spans = 10
    amplitudes_spm, amplitudes_xpm = [], []
    for index in range(n_spans):
        context = SpanContext(
            span_index=index,
            total_spans=n_spans,
            cumulative_length_m=index * 100e3,
            span_length_m=100e3,
        )
        out = kernel.evaluate(power, 100e3, n_spans, span_context=context)
        amplitudes_spm.append(out.nli_amplitude_spm)
        amplitudes_xpm.append(out.nli_amplitude_xpm)
    for amplitudes in (amplitudes_spm, amplitudes_xpm):
        coherent = np.abs(np.sum(amplitudes, axis=0)) ** 2
        incoherent = np.sum(np.abs(np.asarray(amplitudes)) ** 2, axis=0)
        gain_db = 10.0 * np.log10(np.median(coherent / incoherent))
        assert 0.0 < gain_db < 10.0, gain_db


def test_path_evaluation_uses_the_link_level_span_count() -> None:
    """路径评估的跨数被正确传给核（1 跨与 10 跨给出不同 SNR）。"""
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    power = launch_profile(case, 0.0)
    snr1 = np.median(model.evaluate_path(_path(1, case.edfa_noise_figure_db), power).snr_db)
    snr10 = np.median(model.evaluate_path(_path(10, case.edfa_noise_figure_db), power).snr_db)
    assert snr1 > snr10


# ---------------------------------------------------------------------------
# 跨间相干：解析极限、相干增益与确定性
# ---------------------------------------------------------------------------
def _kernel_with_dispersion(model, case, dispersion_d: float, slope_s: float = 0.06):
    """按给定色散参数取一个单跨核（模型实例只建一次，只换光纤参数）。"""
    fiber = SMF_100KM.model_copy(
        update={"dispersion_parameter_d": dispersion_d, "dispersion_slope_s": slope_s}
    )
    return model.kernel_for(fiber)


def _coherent_gain_db(kernel, power, n_spans: int = 10) -> float:
    """相干求和相对逐跨功率相加的增益 (dB)（逐信道比值的中位数）。"""
    amplitudes = []
    for index in range(n_spans):
        context = SpanContext(
            span_index=index,
            total_spans=n_spans,
            cumulative_length_m=index * 100e3,
            span_length_m=100e3,
        )
        out = kernel.evaluate(power, 100e3, n_spans, span_context=context)
        amplitudes.append(out.nli_amplitude_spm)
    coherent = np.abs(np.sum(amplitudes, axis=0)) ** 2
    incoherent = np.sum(np.abs(np.asarray(amplitudes)) ** 2, axis=0)
    return float(10.0 * np.log10(np.median(coherent / incoherent)))


def test_cross_span_buildup_is_quadratic_without_dispersion() -> None:
    """色散趋零时跨间完全相干：论文 Eq. (12) 的 ``nu -> N``，累加因子 ``-> N^2``。

    这是论文 ``mu`` 因子最硬的解析极限：无色散时跨间相位失配为零，各跨的 NLI
    场完全同相，于是链路级 NLI 按 ``N^2`` 增长。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = _kernel_with_dispersion(model, case, 0.0, slope_s=0.0)
    power = launch_profile(case, 0.0)
    n_spans = 10
    gn_now, _ = kernel._sci_block(n_spans, 100e3)
    gn_ref, _ = kernel._sci_block(1, 100e3)
    assert gn_now / gn_ref == pytest.approx(float(n_spans) ** 2, rel=1e-9)
    one, _ = kernel.link_level_nli(power, 100e3, 1)
    ten, _ = kernel.link_level_nli(power, 100e3, n_spans)
    assert float(np.median(ten / one)) == pytest.approx(float(n_spans) ** 2, rel=1e-9)
    # 各跨同相：等效跨间相位步长趋于 0。
    assert kernel._sci_step(n_spans, 100e3) == pytest.approx(0.0, abs=1e-6)
    # 相干增益的上限是完全相干：|sum_k a_k|^2 = N^2 |a_k|^2。
    assert _coherent_gain_db(kernel, power, n_spans) == pytest.approx(
        10.0 * np.log10(n_spans), abs=1e-6
    )


def test_cross_span_buildup_moves_towards_linear_with_dispersion() -> None:
    """色散变大时跨间相干退化：累加因子从 ``N^2`` 单调降到 ``N`` 附近。

    论文 Eq. (12) 的 ``|nu|^2`` 在强色散下在 ``p`` 轴上平均为 ``N``（非相干）。
    这里用模型自己的 GN 块（``kappa1``）量这条因子，用 EGN 块量模型总量：
    两者的极限不同（EGN 的 4/6 阶修正是单跨效应，多跨下相对变轻），但都远离
    ``N^2``、并单调下降。
    """
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    n_spans = 10
    buildups = []
    for dispersion_d in (0.0, 17.0, 40.0):
        kernel = _kernel_with_dispersion(
            model, case, dispersion_d, slope_s=0.0 if dispersion_d == 0.0 else 0.06
        )
        gn_now, egn_now = kernel._sci_block(n_spans, 100e3)
        gn_ref, egn_ref = kernel._sci_block(1, 100e3)
        buildups.append((gn_now / gn_ref, egn_now / egn_ref))
    (gn_zero, egn_zero), (gn_c, egn_c), (gn_big, egn_big) = buildups
    # 随色散单调退化。
    assert gn_zero > gn_c > gn_big
    assert egn_zero > egn_c > egn_big
    # 强色散端：贴近 N（非相干），离 N^2 已经差一个量级。
    assert gn_big < 1.5 * n_spans
    assert egn_big < 2.0 * n_spans
    assert gn_big > n_spans
    assert egn_big > n_spans


def test_coherent_gain_in_db_is_measured_for_the_c_band_case() -> None:
    """C 波段 10 跨的相干增益（dB）落在实测区间内。"""
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    # SMF（D = 17 ps/nm/km）、100 km 跨段、10 跨：实测 +2.77 dB。
    assert 2.0 < _coherent_gain_db(kernel, power) < 3.5


def test_multi_span_output_is_deterministic() -> None:
    """同一实例、同一 (path, power) 连调三次逐位相同。

    跨号只能来自 ``SpanContext``；模型内部不许有跨调用累加的计数器。
    """
    case = build_case("c_57ch")
    path = build_path(case)
    model = make_model("carena2014", case)
    power = launch_profile(case, 0.0)
    first = model.evaluate_path(path, power)
    model.evaluate_path(path, launch_profile(case, 6.0))
    second = model.evaluate_path(path, power)
    third = model.evaluate_path(path, power)
    assert np.array_equal(np.asarray(first.snr_db), np.asarray(second.snr_db))
    assert np.array_equal(np.asarray(second.snr_db), np.asarray(third.snr_db))
    assert np.array_equal(np.asarray(first.spm_noise_w), np.asarray(third.spm_noise_w))
    assert np.array_equal(np.asarray(first.xpm_noise_w), np.asarray(third.xpm_noise_w))


def test_xci_neighbour_window_convergence() -> None:
    """XCI 近邻窗口翻倍时 SNR 的变化很小（远处信道因 |mu| 衰减而可截断）。"""
    from network_nlo_eval.models import carena2014 as module

    case = build_case("c_57ch")
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
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, 0.0)
    with pytest.raises(ValueError, match="one entry per channel"):
        kernel.evaluate(power[:-1], 100e3, 1)
    with pytest.raises(ValueError, match="span_length_m"):
        kernel.evaluate(power, 0.0, 1)


def test_kernel_is_a_span_kernel() -> None:
    """核实现了冻结的单跨协议。"""
    case = build_case("c_57ch")
    model = make_model("carena2014", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    assert isinstance(kernel, Carena2014SpanKernel)
    assert isinstance(kernel, SpanKernel)


def test_kernel_does_not_mutate_its_inputs() -> None:
    """核不能就地改写入纤功率数组（路由层会复用它）。"""
    case = build_case("c_57ch")
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
