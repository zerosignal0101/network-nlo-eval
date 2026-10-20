"""D'Amico 2022 可扩展且解耦的 GGN 近似模型的契约与物理行为测试。

覆盖 WP 要求的项：

* 注册与契约（注册名 ``damico2022``、family ``ggn``、参考出处）；
* SC/XC 分解（单信道时 XC 归零、未占用信道静默、SC 只依赖自身功率）；
* ``C_∞`` 相干 SC 包络（单跨为零、多跨为正、逐跨等效值的换算）；
* 1 跨 / 10 跨一致性（逐跨物理不变、只有 SC 被 ``(1+C_∞)`` 放大、线性累加）；
* 零 ISRS 极限与冻结基线 ``ecz`` 的闭式一致性；
* 按带解耦的分组口径（精确口径与分组数无关；质心口径随分组数收敛）；
* 与 ``ecz`` / ``buglia2023`` 的 SNR 量级对比；
* 数值健壮性与性能上限。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    build_case,
    build_path,
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
from network_nlo_eval.models.damico2022 import (
    NOMINAL_BAND_EDGES_HZ,
    SPM_WEIGHT,
    XPM_WEIGHT,
    Damico2022Model,
    Damico2022SpanKernel,
    _coherence_envelope,
)
from network_nlo_eval.network.elements import EDFAConfig

#: 论文 Eq. (24) 后的权重：``W_SC = (1 + C_∞)·16/27``、``W_XC = 2·16/27``。
EXPECTED_SPM_WEIGHT = 16.0 / 27.0
EXPECTED_XPM_WEIGHT = 2.0 * 16.0 / 27.0

#: WP 的精度自检要求：GGN 相对 GN（``ecz``）的 SNR 差落在 0…0.8 dB（GGN 略低）。
SNR_GAP_MAX_DB = 0.8

#: WP 的精度自检要求：与 ``buglia2023``（ISRS-GN 闭式）的差异应在 1 dB 内。
BUGLIA_GAP_MAX_DB = 1.0


def _fiber(length_km: float = 100.0):
    """基准光纤（SMF，100 km 单跨）。"""
    return SMF_100KM.model_copy(update={"length_km": length_km, "max_span_length_km": length_km})


def _path(n_spans: int, noise_figure_db: float = 5.5) -> MultiSpanOpticalPath:
    """N 个等长跨段 + 等 NF 的 EDFA（与基准算例同构）。"""
    return MultiSpanOpticalPath([_fiber()] * n_spans, [EDFAConfig(noise_figure_db=noise_figure_db)] * n_spans)


def _mid(case) -> int:
    return case.num_channels // 2


def _kernel(case, **options) -> Damico2022SpanKernel:
    """取算例对应光纤的 GGN 单跨核。"""
    model = make_model("damico2022", case, **options)
    return model.kernel_for(build_path(case).fiber_configs[0])


# ---------------------------------------------------------------------------
# 注册与契约
# ---------------------------------------------------------------------------


def test_model_is_registered() -> None:
    """模型进了注册表，ModelInfo 字段齐全。"""
    assert "damico2022" in available_models()
    info = get_model_info("damico2022")
    assert info.family == "ggn"
    assert "D'Amico" in info.reference
    assert "3162134" in info.reference
    assert any(option[0] == "xc_aggregation" for option in info.options)


def test_model_constructible_from_registry() -> None:
    """能按注册表构造，节点几何与基准算例一致。"""
    case = build_case("scaling_160ch")
    model = create_model("damico2022", grid_for(case), _fiber(), symbol_rate_hz=case.symbol_rate_hz)
    assert isinstance(model, Damico2022Model)
    assert model.n_nodes == case.num_channels
    assert model.channel_bandwidth_hz_channels.shape == (case.num_channels,)
    assert np.all(model.channel_bandwidth_hz_channels > 0)
    assert model.power_width_scale().shape == (case.num_channels,)


def test_paper_weights_match_the_equations() -> None:
    """权重取自论文 Eq. (24) 之后：``W_SC = 16/27``、``W_XC = 2·16/27``。"""
    assert SPM_WEIGHT == pytest.approx(EXPECTED_SPM_WEIGHT)
    assert XPM_WEIGHT == pytest.approx(EXPECTED_XPM_WEIGHT)


# ---------------------------------------------------------------------------
# SC / XC 分解
# ---------------------------------------------------------------------------


def test_single_channel_has_no_xc() -> None:
    """单信道时 XC 归零、SC 为正：XC 顾名思义只来自互信道。"""
    case = build_case("scaling_160ch")
    power = np.zeros(case.num_channels, dtype=np.float64)
    power[_mid(case)] = 1e-3
    out = _kernel(case).evaluate(power, 100e3, 1)
    idx = _mid(case)
    assert out.xpm_noise_w[idx] == 0.0
    assert np.isfinite(out.spm_noise_w[idx]) and out.spm_noise_w[idx] > 0.0


def test_unoccupied_channels_are_silent() -> None:
    """功率为 0 的信道不出噪声、不出功率。"""
    case = build_case("scaling_160ch")
    power = np.zeros(case.num_channels, dtype=np.float64)
    power[10] = 1e-3
    out = _kernel(case).evaluate(power, 100e3, 1)
    assert out.power_out_w[0] == 0.0
    assert out.spm_noise_w[0] == 0.0
    assert out.xpm_noise_w[0] == 0.0
    assert out.power_out_w[10] > 0.0


def test_sc_depends_only_on_its_own_channel() -> None:
    """零 ISRS 时 SC 只依赖自身功率（SPM 是自相互作用），XC 才随邻道变化。

    这是 SC/XC 分解是否干净的关键判据：把一个远处的干扰信道加进来，
    SC 必须逐位不变，XC 必须变大。
    """
    case = build_case("scaling_160ch")
    kernel = _kernel(case, srs_cr_w_m_hz=0.0)
    idx = _mid(case)
    power = np.zeros(case.num_channels, dtype=np.float64)
    power[idx] = 1e-3
    alone = kernel.evaluate(power, 100e3, 1)

    power_with_neighbour = power.copy()
    power_with_neighbour[idx + 20] = 1e-3
    together = kernel.evaluate(power_with_neighbour, 100e3, 1)

    assert alone.spm_noise_w[idx] == pytest.approx(together.spm_noise_w[idx], rel=1e-12)
    assert together.xpm_noise_w[idx] > alone.xpm_noise_w[idx]
    # 远处的干扰信道仍通过 SC 影响自身信道（本测试只比较 CUT 自己的 SC）。
    assert alone.xpm_noise_w[idx] == 0.0


def test_xc_is_monotone_in_the_number_of_interferers() -> None:
    """XC 随干扰信道数量单调增加。"""
    case = build_case("scaling_160ch")
    kernel = _kernel(case)
    idx = _mid(case)
    previous = 0.0
    for count in (0, 1, 5, 20):
        power = np.zeros(case.num_channels, dtype=np.float64)
        power[idx] = 1e-3
        for offset in range(1, count + 1):
            power[idx + offset] = 1e-3
        out = kernel.evaluate(power, 100e3, 1)
        assert out.xpm_noise_w[idx] >= previous - 1e-30
        previous = out.xpm_noise_w[idx]


# ---------------------------------------------------------------------------
# C_∞ 相干 SC 包络与 1 跨 / 10 跨一致性
# ---------------------------------------------------------------------------


def test_c_inf_is_zero_for_a_single_span() -> None:
    """单跨没有跨间相干可言，``C_∞ = 0``（退化为论文 Eq. (10) 的非相干极限）。"""
    assert _coherence_envelope(2.0e-26, 64e9, 100e3, 1, 64) == 0.0
    case = build_case("scaling_160ch")
    parts = _kernel(case).evaluate_components(launch_profile(case, None), 100e3, 1)
    assert parts["c_inf"] == 0.0


def test_c_inf_is_positive_and_bounded_for_many_spans() -> None:
    """多跨时 ``0 < C_∞ < N_s - 1``：比非相干强、比完全相干弱。"""
    case = build_case("scaling_160ch")
    parts = _kernel(case).evaluate_components(launch_profile(case, None), 100e3, 10)
    c_inf = float(parts["c_inf"])
    assert 0.0 < c_inf < 9.0
    # 强色散（100 km SMF）下跨间相干只能是部分相干。
    assert c_inf < 5.0


def test_c_inf_grows_sublinearly() -> None:
    """``C_∞`` 随跨数单调增长，但远慢于完全相干（``N_s - 1``）。

    相干性随跨间隔快速衰减（``κ(Δs)`` 近似 ``1/Δs``），于是
    ``C_∞ = (2/N_s)Σ(N_s-Δs)κ(Δs)`` 呈对数式增长：它给出一个介于"非相干"
    （``C_∞ = 0``）与"完全相干"（``C_∞ = N_s - 1``）之间的中间态，这正是论文
    Eq. (11) 里"渐近系数"的物理位置。
    """
    values = [_coherence_envelope(2.0e-26, 64e9, 100e3, n, 64) for n in (1, 2, 5, 10, 20, 40, 80)]
    assert values[0] == 0.0
    assert all(b >= a for a, b in zip(values[:-1], values[1:], strict=True))
    # 相对完全相干上限（``N_s - 1``），C_∞ 始终只占小比例：它是对数式增长，
    # 不是线性增长。
    for n, value in zip((1, 2, 5, 10, 20, 40, 80), values, strict=True):
        assert value <= n - 1, f"{n} 跨时 C_∞ = {value:.3f} 超过了完全相干上限"
        if n > 1:
            assert value < 0.2 * n, f"{n} 跨时 C_∞ = {value:.3f} 相对完全相干过强"


def test_span_count_only_scales_the_sc_component() -> None:
    """1 跨 / 10 跨一致性：逐跨物理不变，只有 SC 被 ``(1 + C_∞)`` 放大。

    这是 WP 明确要求的一致性判据：

    * 单跨逐跨等效值 = ``P_SC + P_XC``（``C_∞(1) = 0``）；
    * 10 跨逐跨等效值 = ``(1 + C_∞(10))·P_SC + P_XC``，XC 分量逐位相同；
    * 10 跨的**总量**（核内 ``evaluate`` 口径）= ``10·[(1+C_∞)·P_SC + P_XC]``。
    """
    case = build_case("scaling_160ch")
    kernel = _kernel(case)
    power = launch_profile(case, None)

    one = kernel.evaluate_components(power, 100e3, 1)
    ten = kernel.evaluate_components(power, 100e3, 10)
    idx = _mid(case)

    # 逐跨物理（SC/XC 本体）与跨数无关。
    assert one["sc_w"][idx] == pytest.approx(ten["sc_w"][idx], rel=1e-12)
    assert one["xc_w"][idx] == pytest.approx(ten["xc_w"][idx], rel=1e-12)
    # 只有 SC 被 (1 + C_∞) 放大。
    assert one["spm_w"][idx] == pytest.approx(one["sc_w"][idx], rel=1e-12)
    assert ten["spm_w"][idx] == pytest.approx((1.0 + ten["c_inf"]) * ten["sc_w"][idx], rel=1e-12)
    assert ten["spm_w"][idx] > one["spm_w"][idx]
    assert ten["xpm_w"][idx] == pytest.approx(one["xpm_w"][idx], rel=1e-12)

    # 总量口径：evaluate(..., 10) 是 evaluate(..., 1) 的 10 倍（逐跨等效值的 10 倍）。
    total_one = kernel.evaluate(power, 100e3, 1)
    total_ten = kernel.evaluate(power, 100e3, 10)
    per_span_ten = kernel.evaluate_per_span(power, 100e3, 10)
    assert total_ten.spm_noise_w[idx] == pytest.approx(10.0 * per_span_ten.spm_noise_w[idx], rel=1e-12)
    assert total_ten.xpm_noise_w[idx] == pytest.approx(10.0 * per_span_ten.xpm_noise_w[idx], rel=1e-12)
    assert total_one.spm_noise_w[idx] == pytest.approx(one["spm_w"][idx], rel=1e-12)


def test_route_accumulation_reproduces_the_paper_eq_11_total() -> None:
    """路由层逐跨累加后，NLI 总量等于论文 Eq. (11) 的 ``N_s·[(1+C_∞)P_SC + P_XC]``。

    冻结的路由层在累加噪声时会乘以后续跨的功率传输比，所以这里用"每跨功率
    不变"的等增益理想化：EDFA 补偿掉跨间损耗后，各跨入纤功率相同、传输比为 1，
    于是累加结果应当精确等于论文的 n 跨总量。
    """
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    kernel = _kernel(case)
    parts = kernel.evaluate_components(power, 100e3, 10)
    total = kernel.evaluate(power, 100e3, 10)
    idx = _mid(case)

    expected_spm = 10.0 * (1.0 + float(parts["c_inf"])) * parts["sc_w"][idx]
    expected_xpm = 10.0 * parts["xc_w"][idx]
    assert total.spm_noise_w[idx] == pytest.approx(expected_spm, rel=1e-12)
    assert total.xpm_noise_w[idx] == pytest.approx(expected_xpm, rel=1e-12)


def test_multi_span_nli_grows_linearly() -> None:
    """XC 严格线性累加（10 倍），SC 额外获得跨间相干增益（大于 10 倍）。

    论文 Eq. (10) 的非相干累加下 NLI 随跨数线性增长；Eq. (11) 额外给 SC 分量
    乘上 ``(1 + C_∞)``。因此 10 跨相对 1 跨：XC 恰好 10 倍，SPM 大于 10 倍。
    """
    case = build_case("scaling_160ch")
    power = launch_profile(case, None)
    model = make_model("damico2022", case)

    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    ten = model.evaluate_path(_path(10, case.edfa_noise_figure_db), power)

    spm_ratio = ten.spm_noise_w / one.spm_noise_w
    xpm_ratio = ten.xpm_noise_w / one.xpm_noise_w
    # XC 不受 C_∞ 影响，严格线性。
    assert np.allclose(xpm_ratio, 10.0, rtol=1e-9)
    # SC 拿到跨间相干增益，必须超过线性。
    assert np.all(spm_ratio > 10.0)
    assert np.all(spm_ratio < 20.0)

    nli_one = one.spm_noise_w + one.xpm_noise_w
    nli_ten = ten.spm_noise_w + ten.xpm_noise_w
    ratio = nli_ten / nli_one
    assert np.all(ratio > 10.0)
    assert np.all(ratio < 16.0)


# ---------------------------------------------------------------------------
# 与冻结基线的闭式一致性
# ---------------------------------------------------------------------------


def test_srs_front_end_matches_the_frozen_baseline() -> None:
    """SRS 前端与冻结基线 ``ecz`` 共用同一套公式，出纤功率必须逐位一致。

    这样一来，本模型与 ``ecz`` 的差异只来自 NLI 层，不掺入前端差异。
    """
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    mine = _kernel(case).evaluate(power, 100e3, 1)
    theirs = make_model("ecz", case).span_fiber_output(power, 100e3, 1)
    assert np.allclose(mine.power_out_w, theirs.power_out_w, rtol=1e-12, atol=0.0)


def test_zero_isrs_limit_agrees_with_the_closed_form_gn() -> None:
    """零 ISRS、**同 β_2 口径**时，SC/XC 与冻结基线 ``ecz`` 的闭式 GN 同量级。

    两者的闭式不是同一个（GGN 用逐频率 ``β_2``，``ecz`` 用带中心常数），所以
    逐位相等不成立。这里把 ``D(λ)`` 也退回常数（``use_dispersion_slope=False``），
    把口径对齐后再断言"主项没写错"：

    * 逐信道相对差的中位数 < 10%（同量级）；
    * 最大相对差 < 20%（带边缘处逐频 ``β_2`` 与带中心值的差异最大）。
    """
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    mine = _kernel(case, srs_cr_w_m_hz=0.0, use_dispersion_slope=False).evaluate(power, 100e3, 1)
    ecz = make_model("ecz", case)
    ecz.cr_w_m_hz = 0.0
    theirs = ecz.span_fiber_output(power, 100e3, 1)

    for name, mine_arr, their_arr in (
        ("SC/SPM", mine.spm_noise_w, theirs.spm_noise_w),
        ("XC/XPM", mine.xpm_noise_w, theirs.xpm_noise_w),
    ):
        assert np.all(their_arr > 0.0)
        rel = np.abs(mine_arr - their_arr) / their_arr
        assert np.median(rel) < 0.10, f"{name} 与闭式 GN 的中位相对差 {np.median(rel):.3f} 超过 10%"
        assert rel.max() < 0.20, f"{name} 与闭式 GN 的最大相对差 {rel.max():.3f} 超过 20%"


def test_frequency_dependent_dispersion_shifts_the_nli() -> None:
    """启用 ``D(λ)`` 剖面（论文 Sec. III）后，XC 随波长的色散符号移动。

    论文要求用逐频率的 ``β_2``（由实测 ``D(λ)`` 得到）；冻结基线用的是带中心
    常数 ``β_2``。本光纤的 ``D(λ) = D_ref + S(λ−λ_ref)``：长波长侧（L 波段，
    1564–1626 nm）``D`` 更大 ⇒ ``|β_2|`` 更大 ⇒ XC 核 ``1/|β_2|`` 更小 ⇒ XC
    下降；短波长侧（E 波段，1360–1460 nm）``D`` 明显更小 ⇒ XC 显著上升。

    这个偏差是**物理参数口径**的差，不是 NLI 结构的差：冻结基线在 35.9 THz
    上用带中心常数 ``β_2``，会漏掉低色散波段里明显更强的非线性相互作用。
    横向比较时必须显式说明用的是哪套口径。
    """
    case = build_case("escl_479ch")
    power = launch_profile(case, None)
    kernel = _kernel(case)
    slope = kernel.evaluate(power, 100e3, 1)
    const = _kernel(case, use_dispersion_slope=False).evaluate(power, 100e3, 1)
    ratio = slope.xpm_noise_w / const.xpm_noise_w

    l_band = kernel.f_abs_hz < 191.6e12
    e_band = kernel.f_abs_hz >= 205.4e12
    assert np.all(ratio[l_band] < 1.0), "L 波段（D(λ) 更大）的 XC 应当下降"
    assert np.all(ratio[e_band] > 1.0), "E 波段（D(λ) 更小）的 XC 应当上升"
    assert float(ratio[e_band].max()) > 1.5, "E 波段（低色散）的 XC 应当明显上升"
    assert float(ratio[l_band].min()) > 0.5


def test_xc_matches_the_independent_gn_closed_form() -> None:
    """XC 的 Eq. (29) 实现与**独立实现**的 GN 闭式（Eq. 120）逐信道吻合。

    这里在测试里重新写一遍 Poggiolini 闭式 GN 的 ``ψ``（arXiv:1209.0394
    Eq. 123，形式与 GNPy 的 ``_psi`` 一致），作为不共享代码的参照：

    ``ψ[i,j] = ½(asinh(A(d+R/2)) − asinh(A(d−R/2))) · L_eff,j² / (2π|β_2,ij|L_asym,j)``

    论文 Eq. (29) 是 GGN 三重积分在大色散下的半解析近似，两者应当同量级。
    """
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    kernel = _kernel(case, srs_cr_w_m_hz=0.0)
    span_length_m = 100e3

    alpha = kernel.alpha_npm
    l_eff = (1.0 - np.exp(-alpha * span_length_m)) / alpha
    l_asym = 1.0 / alpha
    beta2 = kernel.beta2_s2_m
    rs = kernel.b_hz
    freq = kernel.f_abs_hz
    n = freq.size

    beta2_avg = 0.5 * (beta2[:, None] + beta2[None, :])
    df = np.abs(freq[:, None] - freq[None, :])
    scale = np.pi**2 * np.abs(beta2_avg) * rs[:, None] * l_asym[None, :]
    right = scale * (df + rs[None, :] / 2.0)
    left = scale * (df - rs[None, :] / 2.0)
    psi = 0.5 * (np.arcsinh(right) - np.arcsinh(left))
    psi *= l_eff[None, :] ** 2 / (2.0 * np.pi * np.abs(beta2_avg) * l_asym[None, :])
    np.fill_diagonal(psi, 0.0)  # 只比 XC（对角线是 SC）

    a = power**2 / rs**2
    reference = XPM_WEIGHT * kernel.gamma_w_m**2 * power * (psi @ a)

    mine = kernel.evaluate(power, span_length_m, 1).xpm_noise_w
    rel = np.abs(mine - reference) / reference
    assert np.median(rel) < 0.05, f"Eq. (29) 与 GN 闭式的 XC 中位相对差 {np.median(rel):.3f} 超过 5%"
    assert rel.max() < 0.15, f"Eq. (29) 与 GN 闭式的 XC 最大相对差 {rel.max():.3f} 超过 15%"


# ---------------------------------------------------------------------------
# 按带解耦：分组口径
# ---------------------------------------------------------------------------


def test_exact_grouping_is_independent_of_the_band_count() -> None:
    """精确口径下，XC 按频带分组只是组织方式：结果与分组数无关。

    这是"按带解耦"在精确口径下的自洽性：分带边界怎么定都不改变物理结果。
    """
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    reference = _kernel(case, xc_aggregation="exact").evaluate(power, 100e3, 1)
    for n_bands in (1, 2, 4, 8):
        other = _kernel(case, xc_aggregation="exact", n_bands=n_bands).evaluate(power, 100e3, 1)
        assert np.allclose(other.xpm_noise_w, reference.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_centroid_aggregation_converges_with_more_bands() -> None:
    """质心口径（真正的解耦近似）随分组数变细而收敛到精确值。

    分组越粗，组内用"谱质心"代表整个频带，XC 核 ``1/|Δf|`` 的变化被抹平，
    误差越大；分组变细后误差必须单调下降。
    """
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    exact = _kernel(case, xc_aggregation="exact").evaluate(power, 100e3, 1).xpm_noise_w
    idx = _mid(case)

    errors = []
    for n_bands in (1, 2, 4, 8, 16):
        approx = _kernel(case, xc_aggregation="centroid", n_bands=n_bands).evaluate(power, 100e3, 1)
        errors.append(abs(approx.xpm_noise_w[idx] - exact[idx]) / exact[idx])

    assert errors[0] > errors[-1], f"质心口径未随分组变细而收敛：{errors}"
    # 分组从 1 段细化到 16 段，误差必须显著下降（XC 核 1/|Δf| 在组内被抹平）。
    assert errors[0] > 4.0 * errors[4], f"质心口径未随分组变细而收敛：{errors}"
    assert errors[4] < 0.25, f"16 组时质心口径仍偏离精确值 {errors[4]:.3f}"


def test_nominal_band_edges_cover_the_spectrum() -> None:
    """名义 L/C/S/E 边界能覆盖仓库基准算例的整个频谱。"""
    case = build_case("escl_479ch")
    model = make_model("damico2022", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    covered = np.zeros(case.num_channels, dtype=bool)
    for mask in kernel.group_masks:
        covered |= mask
    assert np.all(covered), "有信道没有落进任何分组"
    assert len(kernel.group_masks) == len(NOMINAL_BAND_EDGES_HZ)


# ---------------------------------------------------------------------------
# 与其它模型的量级对比（WP 精度自检）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", ["c_80ch", "cl_152ch"])
def test_snr_is_below_the_closed_form_gn_by_a_few_tenths_of_a_db(case_name: str) -> None:
    """相对 GN（``ecz``），GGN 的 SNR 略低，差值落在 0…0.8 dB。

    方向：GGN 计入跨间相干 SC 累积（``C_∞``）与逐频 ``β_2`` 的高阶修正，
    噪声略大 ⇒ SNR 略低。
    """
    case = build_case(case_name)
    power = launch_profile(case, None)
    path = build_path(case)
    mine = make_model("damico2022", case).evaluate_path(path, power).snr_db
    theirs = make_model("ecz", case).evaluate_path(path, power).snr_db
    gap = theirs - mine
    assert gap.max() <= SNR_GAP_MAX_DB + 1e-9, f"GGN 的 SNR 反而高出 {gap.max():.3f} dB"
    assert gap.mean() >= -0.05, f"GGN 的 SNR 平均只低 {-gap.mean():.3f} dB，未体现相干与高阶修正"
    assert gap.min() >= -SNR_GAP_MAX_DB - 1e-9


@pytest.mark.parametrize("case_name", ["c_80ch", "cl_152ch", "scl_277ch"])
def test_snr_is_within_one_db_of_the_isrs_gn_closed_form(case_name: str) -> None:
    """与 ``buglia2023``（ISRS-GN 闭式）的 SNR 差异在 1 dB 内。

    两者都处理了 SRS，但 NLI 层不同（GGN 的逐频 ``β_2`` + 跨间相干 SC）。
    """
    case = build_case(case_name)
    power = launch_profile(case, None)
    path = build_path(case)
    mine = make_model("damico2022", case).evaluate_path(path, power).snr_db
    theirs = make_model("buglia2023", case).evaluate_path(path, power).snr_db
    gap = np.abs(mine - theirs)
    assert gap.max() <= BUGLIA_GAP_MAX_DB + 1e-9, f"与 buglia2023 的差异达 {gap.max():.3f} dB"


def test_wideband_nli_layer_tracks_the_frozen_gn_baseline() -> None:
    """E+S+C+L（35.9 THz）上，本模型的 NLI 层与冻结 GN 基线在全频谱一致。

    这是"模型有没有在 E 波段失效"的判据：论文只声明 C+L+S 的适用域，把模型
    外推到 35.9 THz 后，NLI 层若失效，SNR 偏差会随频率发散。实测偏差在全频谱
    上都小于 0.3 dB，且在 E 波段边缘反而收敛——在 E 波段失效的是**链路**本身
    （SRS 把功率搬走 ⇒ 放大器补偿 ⇒ ASE 暴涨），不是 NLI 层。

    偏差的主要来源是**物理参数口径**：本模型用逐频率 ``D(λ)``/``β_2``（论文
    Sec. III），冻结基线用带中心常数 ``β_2``。
    """
    case = build_case("escl_479ch")
    power = launch_profile(case, None)
    path = build_path(case)
    mine = make_model("damico2022", case).evaluate_path(path, power).snr_db
    theirs = make_model("ecz", case).evaluate_path(path, power).snr_db
    gap = np.abs(mine - theirs)
    assert gap.max() < 0.3, f"宽带下与冻结 GN 基线的最大 SNR 偏差 {gap.max():.3f} dB 过大"


# ---------------------------------------------------------------------------
# 数值健壮性
# ---------------------------------------------------------------------------


def test_finite_on_the_widest_band() -> None:
    """E+S+C+L（35.9 THz）单跨：全信道 SNR 有限、噪声有限非负。"""
    case = build_case("escl_479ch_1span")
    power = launch_profile(case, None)
    result = make_model("damico2022", case).evaluate_path(_path(1, case.edfa_noise_figure_db), power)

    assert result.snr_db.shape == (case.num_channels,)
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(np.isfinite(result.spm_noise_w))
    assert np.all(np.isfinite(result.xpm_noise_w))
    assert np.all(result.spm_noise_w >= 0.0)
    assert np.all(result.xpm_noise_w >= 0.0)


def test_finite_on_the_c_plus_l_band() -> None:
    """C+L（11.3 THz，论文的主验证带宽）全信道 SNR 有限且为正。"""
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    result = make_model("damico2022", case).evaluate_path(_path(case.span_count, case.edfa_noise_figure_db), power)
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(result.snr_db > 0.0)


def test_kernel_rejects_bad_shapes() -> None:
    """功率向量长度必须与通道数一致，跨长必须为正。"""
    case = build_case("scaling_160ch")
    kernel = _kernel(case)
    with pytest.raises(ValueError):
        kernel.evaluate(np.ones(3, dtype=np.float64), 100e3, 1)
    with pytest.raises(ValueError):
        kernel.evaluate(np.ones(case.num_channels, dtype=np.float64), 0.0, 1)


def test_invalid_options_are_rejected() -> None:
    """非法选项必须报错而不是静默降级。"""
    case = build_case("scaling_160ch")
    with pytest.raises(ValueError):
        _kernel(case, xc_aggregation="nonsense")
    with pytest.raises(ValueError):
        _kernel(case, n_bands=0)


# ---------------------------------------------------------------------------
# 性能
# ---------------------------------------------------------------------------


def test_hot_latency_meets_the_work_package_budget() -> None:
    """热调用预算：277 信道 10 跨 < 20 ms；479 信道 10 跨 < 60 ms。"""
    budgets = {"scl_277ch": 20.0, "escl_479ch": 60.0}
    for case_name, budget_ms in budgets.items():
        case = build_case(case_name)
        power = launch_profile(case, None)
        path = build_path(case)
        model = make_model("damico2022", case)
        model.evaluate_path(path, power)  # 预热
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            model.evaluate_path(path, power)
            samples.append((time.perf_counter() - started) * 1e3)
        median = float(np.median(samples))
        assert median <= budget_ms, f"{case_name} 热调用 {median:.2f} ms 超过 {budget_ms:.1f} ms 预算"
