"""cGN 相关函数闭式 GN 模型的测试。

覆盖五类证据：

* **物理正确性**：与直接数值积分的 GN 参考式（论文 Eq. (1)，按 island 逐块积分）
  对比，验证相关域改写与矩形覆盖没有引入系统偏差；
* **数值收敛性**：相关域求积点数加倍后 SNR 收敛；矩形覆盖片数加倍后收敛；
  ISRS 剖面分段数加倍后收敛；
* **ISRS 前端**：含 ISRS 的出纤功率与仓库既有的拉曼核逐位一致；功率倾斜
  随开关变化；ISRS 档的 NSP 能量谱在残余因子退化为常数时与解析闭式一致；
* **色散口径**：两档口径都能跑且给出不同结果；默认档是仓库口径；
* **确定性**：同一 (path, power) 的输出与调用历史无关（重复调用、先调别的功率、
  正序/逆序调用都逐位相同）；核心矩阵缓存幂等；
* **模型契约**：注册名、``egn`` 拒绝、异构跨段生效、与兄弟模型（PCFM）对照。
"""

from __future__ import annotations

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
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanContext,
    available_models,
    create_model,
    get_model_info,
)
from network_nlo_eval.models.cgn import (
    CGN_MODEL_INFO,
    CGNSpanKernel,
    _canonical_power,
    _dirichlet_weight,
    _isrs_nsp_energy_spectrum,
    _nsp_energy_spectrum,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_raman_profile_jit,
    _calc_span_out_power_jit,
)

#: 与兄弟模型 PCFM 的对照需要 PCFM 分支已经合入 multi-model-dev。
PCFM_AVAILABLE = "pcfm" in available_models()


def _model(case_name: str, **options):
    """按算例名构造 cGN 模型。"""
    return make_model("cgn", build_case(case_name), **options)


def _kernel(case_name: str, **options) -> CGNSpanKernel:
    """按算例名构造 cGN 单跨核。"""
    case = build_case(case_name)
    model = _model(case_name, **options)
    return model.kernel_for(build_path(case).fiber_configs[0])


def _direct_island_nli(
    f_abs: np.ndarray,
    b_hz: np.ndarray,
    power_w: np.ndarray,
    alpha_npm: np.ndarray,
    gamma_w_m: np.ndarray,
    beta2_s2_m: float,
    length_m: float,
    n_grid: int,
) -> tuple[np.ndarray, np.ndarray]:
    r"""直接对 GN island 做二维积分，返回逐信道 (SCI, XCI) 噪声功率。

    这是论文 Eq. (1) 的**直接**数值评估：不用矩形覆盖、不做相关域改写，
    只按 (CUT, 第二波, 第三波) 三元组在精确 island 上做二维求积，功率剖面
    取纯光纤衰减（无 ISRS），跨段相干项为零（单跨）。因此它可以作为 cGN 单
    跨核的独立基准。

    空间积分因子取 GN 模型口径 :math:`\\sqrt{\\rho_1\\rho_2\\rho_3/\\rho_c}`，
    即指数 :math:`(\\alpha_2+\\alpha_3)/2`（第一波与被测信道相消）。

    Args:
        f_abs: ``(N,)`` 信道中心频率 (Hz)。
        b_hz: ``(N,)`` 信道带宽 (Hz)。
        power_w: ``(N,)`` 入纤功率 (W)。
        alpha_npm: ``(N,)`` 逐信道功率衰减 (Np/m)。
        gamma_w_m: ``(N,)`` 逐信道非线性系数 (1/(W·m))。
        beta2_s2_m: 二阶色散 (s²/m)。
        length_m: 跨段长度 (m)。
        n_grid: 每个轴的求积点数。

    Returns
    -------
        ``(spm, xpm)``：逐信道 SCI / XCI 噪声功率 (W)。
    """
    n_ch = f_abs.size
    spm = np.zeros(n_ch)
    xpm = np.zeros(n_ch)
    for c in range(n_ch):
        fc = f_abs[c]
        for second in range(n_ch):
            delta = f_abs[second] - fc
            x = np.linspace(-b_hz[c] / 2, b_hz[c] / 2, n_grid)
            y = np.linspace(delta - b_hz[second] / 2, delta + b_hz[second] / 2, n_grid)
            xx = x[:, None]
            yy = y[None, :]
            cell = (x[1] - x[0]) * (y[1] - y[0])
            for r in range(n_ch):
                u_lo = f_abs[r] - fc - b_hz[r] / 2
                u_hi = f_abs[r] - fc + b_hz[r] / 2
                mask = (xx + yy >= u_lo) & (xx + yy < u_hi)
                if not mask.any():
                    continue
                alpha_x = 0.5 * (alpha_npm[second] + alpha_npm[r])
                omega = 4.0 * np.pi**2 * beta2_s2_m * xx * yy
                denom = alpha_x - 1j * omega
                nsp = gamma_w_m[c] * (1.0 - np.exp(-denom * length_m)) / denom
                total = float((np.abs(nsp) ** 2 * mask).sum() * cell)
                g_c = power_w[c] / b_hz[c]
                g_l = power_w[second] / b_hz[second]
                g_r = power_w[r] / b_hz[r]
                contrib = (16.0 / 27.0) * g_c * g_l * g_r * total * b_hz[c]
                if second == c:
                    spm[c] += contrib
                else:
                    xpm[c] += 2.0 * contrib
    return spm, xpm


# ---------------------------------------------------------------------------
# 模型契约
# ---------------------------------------------------------------------------


def test_model_is_registered_with_metadata() -> None:
    """模型以 ``cgn`` 注册，元数据齐全。"""
    info = get_model_info("cgn")
    assert info.name == "cgn"
    assert info.family == "gn"
    assert info.reference
    assert any(option[0] == "correlation_points" for option in info.options)
    assert any(option[0] == "include_isrs" for option in info.options)
    assert any(option[0] == "dispersion_convention" for option in info.options)
    assert "cgn" in available_models()
    assert create_model is not None


def test_egn_is_rejected() -> None:
    """论文只给 GN，``egn=True`` 必须报错。"""
    case = build_case("scaling_40ch")
    with pytest.raises(ValueError, match="egn"):
        create_model("cgn", grid_for(case), case_span := build_path(case).fiber_configs[0], egn=True)
    assert case_span.length_km > 0


def test_invalid_correlation_points_are_rejected() -> None:
    """求积点数非法时报错。"""
    case = build_case("scaling_40ch")
    with pytest.raises(ValueError, match="correlation_points"):
        create_model("cgn", grid_for(case), build_path(case).fiber_configs[0], correlation_points=2)


def test_invalid_dispersion_convention_is_rejected() -> None:
    """色散口径非法时报错。"""
    case = build_case("scaling_40ch")
    with pytest.raises(ValueError, match="dispersion_convention"):
        create_model(
            "cgn",
            grid_for(case),
            build_path(case).fiber_configs[0],
            dispersion_convention="bogus",
        )


def test_result_shape_matches_contract() -> None:
    """路径级结果满足冻结契约的形状约定。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    result = model.evaluate_path(build_path(case), launch_profile(case, None))
    assert isinstance(result, PathQoTResult)
    assert result.signal_power_w.shape == (case.num_channels,)
    assert result.spm_noise_w.shape == (case.num_channels,)
    assert result.xpm_noise_w.shape == (case.num_channels,)
    assert result.snr_db.shape == (case.num_channels,)
    assert result.span_count == case.span_count


def test_model_info_is_shared_with_registry() -> None:
    """注册表里的描述与模块常量一致。"""
    assert get_model_info("cgn") is CGN_MODEL_INFO


# ---------------------------------------------------------------------------
# 物理正确性（不含 ISRS 的相关域改写）
# ---------------------------------------------------------------------------


def test_single_span_matches_direct_gn_integration() -> None:
    """单跨核与 GN 参考式的直接数值积分一致（SCI 2%、XCI 8% 以内）。"""
    n_ch = 3
    spacing = 75e9
    f_abs = 193e12 + (np.arange(n_ch) - (n_ch - 1) / 2) * spacing
    b_hz = np.full(n_ch, 64e9)
    power = np.full(n_ch, 1e-3)
    kernel = CGNSpanKernel(
        fiber_config=SMF_100KM,
        f_abs_hz=f_abs,
        b_hz=b_hz,
        correlation_points=48,
        include_isrs=False,
    )
    out = kernel.evaluate(power, 100e3, 1)
    spm_ref, xpm_ref = _direct_island_nli(
        f_abs,
        b_hz,
        power,
        kernel.alpha_npm,
        kernel.gamma_w_m,
        float(kernel.beta2_s2_m),
        100e3,
        n_grid=401,
    )
    for index in range(n_ch):
        assert out.spm_noise_w[index] == pytest.approx(spm_ref[index], rel=0.02)
        assert out.xpm_noise_w[index] == pytest.approx(xpm_ref[index], rel=0.08)


def test_sci_and_xci_are_separated() -> None:
    """SCI 进 ``spm_noise_w``、XCI 进 ``xpm_noise_w``，且 XCI 明显更大。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    path = build_path(case)
    result = model.evaluate_path(path, launch_profile(case, None))
    assert np.all(result.spm_noise_w > 0.0)
    assert np.all(result.xpm_noise_w > 0.0)
    # 40 信道、1 跨时跨信道干扰总量应明显超过自干扰。
    assert result.xpm_noise_w[20] > result.spm_noise_w[20]


def test_nli_is_split_across_spans() -> None:
    """逐跨 NLI 相加：跨段数加倍时总 NLI 约加倍（齐次链的相干修正除外）。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    one = model.evaluate_path(build_path(case), power)
    fiber = build_path(case).fiber_configs[0]
    edfa = build_path(case).edfa_configs[0]
    two = model.evaluate_path(MultiSpanOpticalPath([fiber] * 2, [edfa] * 2), power)
    ratio = two.spm_noise_w[20] / one.spm_noise_w[20]
    assert 1.5 < ratio < 2.5


def test_identical_spans_accumulate_coherently() -> None:
    """齐次链按论文 Eq. (62)-(64) 相干累加：相干总量高于非相干求和。

    论文 Fig. 13 给出 10 跨齐次链的相干增强量：``|β2| = 10 ps²/km`` 时约
    1.07 dB、``|β2| = 1 ps²/km`` 时约 2.40 dB。本仓库的 SMF 参考光纤
    ``D = 17 ps/(nm·km)``（约 21.7 ps²/km），色散更强，相干增强更弱，故这里
    只断言"相干高于非相干"并给出量级上界。

    相干口径下逐跨给出的是**单跨**噪声（契约要求 ``|a|**2 == spm_noise_w``），
    链路总量由路由层按 ``|Σ a_k|²`` 累加得到。
    """
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    fiber = build_path(case).fiber_configs[0]
    edfa = build_path(case).edfa_configs[0]

    single = model.evaluate_path(MultiSpanOpticalPath([fiber], [edfa]), power)
    ten = model.evaluate_path(MultiSpanOpticalPath([fiber] * 10, [edfa] * 10), power)

    # 相干总量 = 10 跨逐跨相干累加；非相干求和 = 10 × 单跨。
    ratio = ten.spm_noise_w[20] / single.spm_noise_w[20]
    assert ratio > 10.0
    gain_db = 10.0 * np.log10(ratio / 10.0)
    assert 0.05 < gain_db < 3.0


def test_dark_channel_carries_no_noise() -> None:
    """未占用的信道不产生任何 NLI。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    power[3] = 0.0
    path = build_path(case)
    result = model.evaluate_path(path, power)
    assert result.spm_noise_w[3] == 0.0
    assert result.xpm_noise_w[3] == 0.0
    assert np.isinf(result.snr_db[3])


def test_output_power_follows_fiber_attenuation_without_isrs() -> None:
    """关掉 ISRS 时出纤功率 = 入纤功率 × exp(-αL)，且噪声非负。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch", include_isrs=False)
    power = launch_profile(case, None)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    out = kernel.evaluate(power, 100e3, 1)
    expected = power * np.exp(-kernel.alpha_npm * 100e3)
    assert np.allclose(out.power_out_w, expected, rtol=1e-12)
    assert np.all(out.spm_noise_w >= 0.0)
    assert np.all(out.xpm_noise_w >= 0.0)


# ---------------------------------------------------------------------------
# ISRS 前端
# ---------------------------------------------------------------------------


def test_isrs_output_power_matches_repo_raman_kernel() -> None:
    """含 ISRS 的出纤功率与仓库既有的拉曼核逐位一致。

    cGN 复用 :func:`_calc_raman_profile_jit` 与
    :func:`_calc_span_out_power_jit`（``models/isrs_gn.py`` 自己 import 的
    那两个），故功率倾斜必须与兄弟模型同源。
    """
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)
    kernel = _kernel("scaling_40ch", include_isrs=True)
    out = kernel.evaluate(power, 100e3, 1)

    r_f = _calc_raman_profile_jit(
        kernel.f_rel_hz,
        kernel.f_m_hz,
        kernel.f_M_hz,
        15e12,
        float(power.sum()),
    )
    expected = _calc_span_out_power_jit(
        power,
        kernel.f_rel_hz,
        kernel.alpha_npm,
        100e3,
        r_f,
        kernel.k_bar,
    )
    assert np.allclose(out.power_out_w, expected, rtol=1e-12, atol=0.0)


def test_isrs_switch_changes_the_power_tilt() -> None:
    """开关 ISRS 会改变跨频带的功率倾斜（高端信道更亏）。"""
    case = build_case("scaling_40ch")
    power = launch_profile(case, None)
    without = _kernel("scaling_40ch", include_isrs=False).evaluate(power, 100e3, 1)
    with_isrs = _kernel("scaling_40ch", include_isrs=True).evaluate(power, 100e3, 1)

    tilt_without = 10.0 * np.log10(without.power_out_w / power)
    tilt_with = 10.0 * np.log10(with_isrs.power_out_w / power)
    # 纯衰减本身的波长依赖也会让相对衰减有一点起伏，ISRS 让这个起伏明显变大
    # （C 波段上端信道更亏）。
    spread_without = float(np.ptp(tilt_without))
    spread_with = float(np.ptp(tilt_with))
    assert spread_with > spread_without + 0.1
    extra = tilt_with - tilt_without
    assert extra[-1] < extra[0]


def test_isrs_nsp_spectrum_reduces_to_the_analytic_exponential() -> None:
    """残余因子退化为常数时，ISRS 档的 NSP 能量谱与解析闭式逐位一致。"""
    alpha_x = np.array([[[[3.7e-5]]]])
    length_m = 100e3
    n_seg = 8
    seg = length_m / n_seg
    omega = np.linspace(0.0, 5.0, 11)[None, None, :]
    residual = np.ones((1, 1, n_seg))
    gamma = np.array([[[[1.3e-3]]]])

    analytic = _nsp_energy_spectrum(gamma, alpha_x, length_m, omega)
    numeric = _isrs_nsp_energy_spectrum(gamma, alpha_x, residual, seg, omega)
    assert np.allclose(numeric, analytic, rtol=1e-12)


def test_isrs_segment_count_converges() -> None:
    """ISRS 剖面分段数加倍后 NLI 收敛。"""
    n_ch = 3
    spacing = 75e9
    f_abs = 193e12 + (np.arange(n_ch) - (n_ch - 1) / 2) * spacing
    b_hz = np.full(n_ch, 64e9)
    power = np.full(n_ch, 1e-3)
    values = []
    for segments in (4, 16):
        kernel = CGNSpanKernel(
            fiber_config=SMF_100KM,
            f_abs_hz=f_abs,
            b_hz=b_hz,
            correlation_points=48,
            include_isrs=True,
            isrs_segments=segments,
        )
        out = kernel.evaluate(power, 100e3, 1)
        values.append((out.spm_noise_w[1], out.xpm_noise_w[1]))
    assert abs(values[1][0] - values[0][0]) < 0.02 * values[1][0]


# ---------------------------------------------------------------------------
# 色散展开口径
# ---------------------------------------------------------------------------


def test_dispersion_conventions_differ() -> None:
    """两档口径都能跑，且给出可测地不同的 SNR。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    repo = _model("scaling_40ch", dispersion_convention="repo").evaluate_path(path, power)
    absolute = _model(
        "scaling_40ch", dispersion_convention="paper_absolute"
    ).evaluate_path(path, power)
    delta = repo.snr_db - absolute.snr_db
    assert np.all(np.isfinite(delta))
    assert np.max(np.abs(delta)) > 0.01


def test_repo_convention_is_the_default() -> None:
    """默认档是仓库口径（与显式指定 ``repo`` 逐位一致）。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    default = _model("scaling_40ch").evaluate_path(path, power)
    explicit = _model("scaling_40ch", dispersion_convention="repo").evaluate_path(path, power)
    assert np.array_equal(default.snr_db, explicit.snr_db)


def test_paper_absolute_convention_uses_absolute_frequencies() -> None:
    """``paper_absolute`` 档的三阶项按绝对频率展开，与仓库口径显著不同。"""
    kernel_repo = _kernel("scaling_40ch", dispersion_convention="repo")
    kernel_abs = _kernel("scaling_40ch", dispersion_convention="paper_absolute")
    assert kernel_repo.beta2_s2_m == kernel_abs.beta2_s2_m
    # 绝对频率口径引入 ~193 THz 的伪项：有效 beta2 的量级完全不同。
    offset = 193e12
    delta = np.pi * kernel_abs.beta3_s3_m * 2.0 * offset
    assert abs(delta) > 5.0 * abs(kernel_abs.beta2_s2_m)


# ---------------------------------------------------------------------------
# NSP 指数口径
# ---------------------------------------------------------------------------


def test_nsp_convention_is_the_default() -> None:
    """默认档是 GN 模型口径（与显式指定 ``gn`` 逐位一致）。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    default = _model("scaling_40ch").evaluate_path(path, power)
    explicit = _model("scaling_40ch", nsp_convention="gn").evaluate_path(path, power)
    assert np.array_equal(default.snr_db, explicit.snr_db)


def test_nsp_conventions_differ() -> None:
    """两档 NSP 指数口径都能跑，且给出可测地不同的 SNR。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    gn = _model("scaling_40ch", nsp_convention="gn").evaluate_path(path, power)
    product = _model("scaling_40ch", nsp_convention="paper_product").evaluate_path(path, power)
    delta = gn.snr_db - product.snr_db
    assert np.all(np.isfinite(delta))
    assert np.max(np.abs(delta)) > 0.01


def test_invalid_nsp_convention_is_rejected() -> None:
    """NSP 口径非法时报错。"""
    case = build_case("scaling_40ch")
    with pytest.raises(ValueError, match="nsp_convention"):
        create_model(
            "cgn",
            grid_for(case),
            build_path(case).fiber_configs[0],
            nsp_convention="bogus",
        )


def test_zero_dispersion_limit_matches_the_numerical_reference() -> None:
    r"""零色散极限下 cGN 与仓库的 ``numerical`` 参考模型一致。

    零色散时 GN 模型有解析极限，``numerical`` 已按该极限校验过
    （相对误差 1.3e-7）。cGN 的 NSP 指数必须落在同一口径上：
    GN 模型的空间积分因子是 :math:`\\sqrt{\\rho_1\\rho_2\\rho_3/\\rho_c}`，
    指数要除以 2。这一条是 NSP 指数口径的回归守卫。
    """
    if "numerical" not in available_models():
        pytest.skip("the numerical reference model is not registered")
    n_ch = 3
    spacing = 75e9
    f_abs = 193e12 + (np.arange(n_ch) - (n_ch - 1) / 2) * spacing
    b_hz = np.full(n_ch, 64e9)
    power = np.full(n_ch, 1e-3)
    zero_dispersion = SMF_100KM.model_copy(
        update={"dispersion_parameter_d": 0.0, "dispersion_slope_s": 0.0}
    )
    grid = SpectrumGrid(n_ch, spacing, 193e12)

    cgn = create_model(
        "cgn", grid, zero_dispersion, symbol_rate_hz=64e9, include_isrs=False
    ).kernel_for(zero_dispersion).evaluate(power, 100e3, 1)
    reference = create_model(
        "numerical", grid, zero_dispersion, symbol_rate_hz=64e9
    ).kernel_for(zero_dispersion).evaluate(power, 100e3, 1)

    assert cgn.spm_noise_w[1] == pytest.approx(reference.spm_noise_w[1], rel=0.02)
    assert cgn.xpm_noise_w[1] == pytest.approx(reference.xpm_noise_w[1], rel=0.02)


# ---------------------------------------------------------------------------
# 数值收敛性
# ---------------------------------------------------------------------------


def test_correlation_convergence() -> None:
    """相关域求积点数加倍后，逐信道 SNR 变化 < 0.05 dB。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    coarse = _model("scaling_40ch", correlation_points=16).evaluate_path(path, power)
    fine = _model("scaling_40ch", correlation_points=32).evaluate_path(path, power)
    delta = fine.snr_db - coarse.snr_db
    assert np.max(np.abs(delta)) < 0.05


def test_correlation_convergence_is_monotone_in_resolution() -> None:
    """求积点数加倍时 SNR 变化继续减小（收敛而非震荡）。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    snr_16 = _model("scaling_40ch", correlation_points=16).evaluate_path(path, power).snr_db
    snr_32 = _model("scaling_40ch", correlation_points=32).evaluate_path(path, power).snr_db
    snr_64 = _model("scaling_40ch", correlation_points=64).evaluate_path(path, power).snr_db
    step_1 = np.max(np.abs(snr_32 - snr_16))
    step_2 = np.max(np.abs(snr_64 - snr_32))
    assert step_2 <= step_1 + 1e-6


def test_island_cover_convergence() -> None:
    """矩形覆盖片数加倍时，NLI 收敛到直接积分值。"""
    n_ch = 3
    spacing = 75e9
    f_abs = 193e12 + (np.arange(n_ch) - (n_ch - 1) / 2) * spacing
    b_hz = np.full(n_ch, 64e9)
    power = np.full(n_ch, 1e-3)
    values = []
    for cells in (8, 32):
        kernel = CGNSpanKernel(
            fiber_config=SMF_100KM,
            f_abs_hz=f_abs,
            b_hz=b_hz,
            correlation_points=48,
            island_cells=cells,
            include_isrs=False,
        )
        out = kernel.evaluate(power, 100e3, 1)
        values.append((out.spm_noise_w[1], out.xpm_noise_w[1]))
    # 片数越多越接近参考值：出厂档（8 片）与 32 片的偏差应在 2% 以内。
    assert abs(values[1][0] - values[0][0]) < 0.02 * values[1][0]


# ---------------------------------------------------------------------------
# 异构跨段
# ---------------------------------------------------------------------------


def test_heterogeneous_spans_differ_from_uniform() -> None:
    """10 跨中第 3、7 跨改成 80 km，结果必须与全 100 km 可测地不同。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    base_fiber = build_path(case).fiber_configs[0]
    edfa = build_path(case).edfa_configs[0]

    uniform = MultiSpanOpticalPath([base_fiber] * 10, [edfa] * 10)
    short = base_fiber.model_copy(update={"length_km": 80.0, "max_span_length_km": 80.0})
    fibers = [base_fiber] * 10
    fibers[2] = short
    fibers[6] = short
    heterogeneous = MultiSpanOpticalPath(fibers, [edfa] * 10)

    uniform_result = model.evaluate_path(uniform, power)
    heterogeneous_result = model.evaluate_path(heterogeneous, power)
    delta = heterogeneous_result.snr_db - uniform_result.snr_db
    assert np.max(np.abs(delta)) > 0.01


def test_heterogeneous_spans_use_per_span_fiber_parameters() -> None:
    """不同跨段的光纤参数（衰减/色散）必须真正进入计算。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    base_fiber = build_path(case).fiber_configs[0]
    edfa = build_path(case).edfa_configs[0]
    lossy = base_fiber.model_copy(
        update={"attenuation_db_km_ref": 0.4, "use_attenuation_polynomial": False},
    )
    path = MultiSpanOpticalPath([base_fiber, lossy], [edfa] * 2)
    result = model.evaluate_path(path, power)
    assert np.all(np.isfinite(result.snr_db))
    assert result.snr_db[20] < 60.0


# ---------------------------------------------------------------------------
# 与兄弟模型对照
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not PCFM_AVAILABLE, reason="pcfm branch is not merged into multi-model-dev")
def test_matches_pcfm_on_homogeneous_link() -> None:
    """同质跨段单跨下，cGN 与 PCFM 的逐信道 SNR 差 RMS < 0.5 dB。"""
    case = build_case("cl_152ch")
    path = build_path(case)
    power = launch_profile(case, None)
    cgn = _model("cl_152ch").evaluate_path(path, power)
    pcfm = make_model("pcfm", case).evaluate_path(path, power)
    delta = cgn.snr_db - pcfm.snr_db
    assert float(np.sqrt(np.mean(delta**2))) < 0.5


# ---------------------------------------------------------------------------
# 确定性：输出不得依赖调用历史
# ---------------------------------------------------------------------------


def _snr(model: object, case, path, power_dbm: float) -> np.ndarray:
    """在给定功率点上取逐信道 SNR。"""
    return np.asarray(model.evaluate_path(path, launch_profile(case, power_dbm)).snr_db, dtype=float)


def test_repeated_calls_on_one_instance_are_bit_identical() -> None:
    """同一实例对同一 (path, power) 连调三次，结果必须逐位相同。

    这是 QoT 引擎的基本契约：一次路由搜索里同一个模型会被反复评估，同一个
    问题必须永远得到同一个答案。任何跨调用累积的状态（缓存污染、分段推进的
    残余因子、功率指纹）都会在这里暴露。
    """
    case = build_case("scaling_40ch")
    path = build_path(case)
    model = _model("scaling_40ch")

    first = _snr(model, case, path, -4.0)
    second = _snr(model, case, path, -4.0)
    third = _snr(model, case, path, -4.0)

    assert np.array_equal(first, second)
    assert np.array_equal(second, third)


def test_call_history_does_not_change_the_result() -> None:
    """先调别的功率之后，同一功率的结果必须与全新实例逐位相同。

    回归用例：第二轮引入"带功率指纹的核心矩阵缓存"时，指纹用的是**相对**
    功率（除以最大值），于是 -4 dBm 与 -1 dBm 落到同一个键上，缓存里存的
    值取决于"哪个功率先到"——输出因此依赖调用历史。
    """
    case = build_case("scaling_40ch")
    path = build_path(case)

    fresh = _snr(_model("scaling_40ch"), case, path, -4.0)

    reused = _model("scaling_40ch")
    _snr(reused, case, path, 0.0)
    _snr(reused, case, path, 4.0)
    after = _snr(reused, case, path, -4.0)

    assert np.array_equal(fresh, after), "先调 0/+4 dBm 改变了 -4 dBm 的结果"


def test_call_order_does_not_change_the_result() -> None:
    """正序 [-4, 0, +4, -4] 与逆序 [+4, 0, -4] 的同功率结果必须逐位相同。"""
    case = build_case("scaling_40ch")
    path = build_path(case)

    forward_model = _model("scaling_40ch")
    forward_minus4 = _snr(forward_model, case, path, -4.0)
    _snr(forward_model, case, path, 0.0)
    _snr(forward_model, case, path, 4.0)
    forward_minus4_again = _snr(forward_model, case, path, -4.0)

    reverse_model = _model("scaling_40ch")
    _snr(reverse_model, case, path, 4.0)
    _snr(reverse_model, case, path, 0.0)
    reverse_minus4 = _snr(reverse_model, case, path, -4.0)

    assert np.array_equal(forward_minus4, forward_minus4_again)
    assert np.array_equal(forward_minus4, reverse_minus4)


def test_determinism_holds_on_a_multi_span_link() -> None:
    """多跨链（10 跨、逐跨功率不同）上同样不能依赖调用历史。"""
    case = build_case("c_57ch")
    path = build_path(case)

    fresh = _snr(_model("c_57ch"), case, path, -4.0)

    reused = _model("c_57ch")
    _snr(reused, case, path, 0.0)
    after = _snr(reused, case, path, -4.0)
    again = _snr(reused, case, path, -4.0)

    assert np.array_equal(fresh, after)
    assert np.array_equal(after, again)


def test_determinism_holds_without_isrs() -> None:
    """不含 ISRS 的档同样不能依赖调用历史（出纤功率也不能被缓存成旧值）。"""
    case = build_case("scaling_40ch")
    path = build_path(case)

    fresh = _snr(_model("scaling_40ch", include_isrs=False), case, path, -4.0)

    reused = _model("scaling_40ch", include_isrs=False)
    _snr(reused, case, path, 0.0)
    after = _snr(reused, case, path, -4.0)

    assert np.array_equal(fresh, after)


def test_core_cache_key_separates_absolute_power_levels() -> None:
    """核心矩阵缓存必须区分**绝对**功率，而不是只看功率分布的形状。

    -4 dBm 与 -1 dBm 的逐信道功率分布完全相同（只是标度不同），但拉曼倾斜
    的速率正比于总功率，两者必须落到不同的缓存键上。
    """
    case = build_case("scaling_40ch")
    kernel = _kernel("scaling_40ch")
    low, _ = _canonical_power(launch_profile(case, -4.0))
    high, _ = _canonical_power(launch_profile(case, -1.0))
    assert low.shape == high.shape
    assert not np.array_equal(low, high)

    _, key_low = _canonical_power(launch_profile(case, -4.0))
    _, key_high = _canonical_power(launch_profile(case, -1.0))
    assert key_low != key_high


def test_canonical_power_is_stable_under_float_round_trips() -> None:
    """规范化必须吸收 1 ulp 级的浮点往返，否则整条光路要重建 10 次核心矩阵。"""
    case = build_case("scaling_40ch")
    power = launch_profile(case, -4.0)
    canonical, key = _canonical_power(power)
    nudged = power * (1.0 + 2.0 * np.finfo(np.float64).eps)
    canonical_nudged, key_nudged = _canonical_power(nudged)
    assert key == key_nudged
    assert np.array_equal(canonical, canonical_nudged)


def test_canonical_power_survives_extreme_scales() -> None:
    """极端功率标度下规范化不能产生 NaN/Inf（次正规数、下溢、上溢）。"""
    for scale in (5e-324, 1e-320, 1e-305, 1e-30, 1e-4, 1.0, 1e12, 1e300):
        canonical, key = _canonical_power(np.full(3, scale))
        assert np.all(np.isfinite(canonical)), f"scale={scale:g} produced non-finite values"
        assert len(key) > 0


def test_canonical_power_handles_zero_and_mixed_vectors() -> None:
    """全零功率与含零（未占用信道）的功率向量都要给出有限的规范值。"""
    canonical, key = _canonical_power(np.zeros(4))
    assert np.all(canonical == 0.0)
    assert key == b"zero"

    canonical, key = _canonical_power(np.array([0.0, 1e-4, 2e-4]))
    assert np.all(np.isfinite(canonical))
    assert canonical[0] == 0.0


def test_core_cache_is_idempotent() -> None:
    """同一个键必须永远对应同一个矩阵（缓存幂等）。"""
    kernel = _kernel("scaling_40ch")
    case = build_case("scaling_40ch")
    power = launch_profile(case, -4.0)
    core_a = kernel._core_for(power, 100e3, 1)
    kernel._core_cache.clear()
    kernel._core_cache_bytes = 0
    core_b = kernel._core_for(power, 100e3, 1)
    assert np.array_equal(core_a.plain, core_b.plain)


# ---------------------------------------------------------------------------
# 跨间相干累加（复振幅口径）
# ---------------------------------------------------------------------------

#: 改造前（纯功率口径）单跨链的硬编码读数：``scaling_40ch``、标称每信道功率、
#: 通道 20。这三个数由**改造前**的实现产出（``git show HEAD~1`` 可复现），
#: 用来钉死"单跨走纯功率分支时与改造前逐位一致"这条硬性要求。
SINGLE_SPAN_SNR_MID_BEFORE = 27.866528441287723
SINGLE_SPAN_SPM_MID_BEFORE = 1.496862542480594e-07
SINGLE_SPAN_XPM_MID_BEFORE = 2.817325422025434e-07


def _homogeneous_path(case, spans: int, fiber=None) -> MultiSpanOpticalPath:
    """同一根光纤、同一台放大器重复 ``spans`` 次的光路。"""
    if fiber is None:
        fiber = build_path(case).fiber_configs[0]
    edfa = build_path(case).edfa_configs[0]
    return MultiSpanOpticalPath([fiber] * spans, [edfa] * spans)


def test_single_span_result_is_bit_identical_to_the_previous_implementation() -> None:
    """单跨链：改造后的 SNR / NLI 与改造前逐位相同。

    单跨时相干退化为单跨，``|a|**2 == 噪声功率``，路由层必须走**纯功率**
    分支，浮点运算顺序不变。这里用改造前的硬编码数值做断言。
    """
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    result = model.evaluate_path(build_path(case), launch_profile(case, None))
    assert result.snr_db[20] == SINGLE_SPAN_SNR_MID_BEFORE
    assert result.spm_noise_w[20] == SINGLE_SPAN_SPM_MID_BEFORE
    assert result.xpm_noise_w[20] == SINGLE_SPAN_XPM_MID_BEFORE


def test_single_span_keeps_the_power_only_branch() -> None:
    """单跨链不给出复振幅（路由层因此走原来的纯功率累加）。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    step = model.span_step(
        build_path(case).fiber_configs[0],
        build_path(case).edfa_configs[0],
        launch_profile(case, None),
        100e3,
    )
    assert getattr(step, "nli_amplitude_spm", None) is None
    assert getattr(step, "nli_amplitude_xpm", None) is None


def test_amplitude_magnitude_matches_the_noise_power() -> None:
    """``|a|**2`` 必须严格等于同名的噪声功率（契约硬性要求）。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    kernel.n_coherent_spans = 10
    context = SpanContext(
        span_index=3,
        total_spans=10,
        cumulative_length_m=300e3,
        span_length_m=100e3,
    )
    out = kernel.evaluate(power, 100e3, 1, span_context=context)
    assert out.nli_amplitude_spm is not None
    assert out.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_coherent_sum_reproduces_the_dirichlet_link_total() -> None:
    """路由层的 ``|Σ a_k|²`` 必须精确复现 Dirichlet 核算出的链路总量。

    斜坡是按"链路级相干总量 ÷ 单跨总量"反解出来的，所以这条等式是斜坡的定义
    性质；这里用**独立**的 :meth:`CGNSpanKernel.link_nli` 重新算一遍链路总量
    来验证（相对误差 < 1e-9）。
    """
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    result = model.evaluate_path(_homogeneous_path(case, 10), power)
    link_spm, link_xpm = kernel.link_nli(power, 100e3, 10)
    assert np.all(np.isfinite(result.spm_noise_w))
    assert np.allclose(result.spm_noise_w, link_spm, rtol=1e-9, atol=0.0)
    assert np.allclose(result.xpm_noise_w, link_xpm, rtol=1e-9, atol=0.0)


def test_multi_span_nli_is_finite_and_monotonic_in_span_count() -> None:
    """NLI 随跨数单调增长、有限，且 10 跨不等于"单跨 × 10"。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    single = model.evaluate_path(_homogeneous_path(case, 1), power)
    ratios = []
    for spans in (1, 2, 3, 5, 10):
        result = model.evaluate_path(_homogeneous_path(case, spans), power)
        assert np.all(np.isfinite(result.spm_noise_w))
        assert np.all(result.spm_noise_w >= 0.0)
        ratios.append(float(result.spm_noise_w[20] / single.spm_noise_w[20]))
    assert all(b > a for a, b in zip(ratios, ratios[1:], strict=False)), ratios
    # 10 跨不是"单跨 × 10"：相干累加给出的是 Dirichlet 总量。
    assert ratios[-1] != pytest.approx(10.0, rel=1e-6)
    assert ratios[-1] > 10.0


def test_zero_dispersion_limit_is_fully_coherent() -> None:
    """色散趋零：跨间完全相干，NLI ∝ N²（10 跨 → 100 倍单跨）。"""
    case = build_case("scaling_40ch")
    zero_dispersion = SMF_100KM.model_copy(
        update={"dispersion_parameter_d": 0.0, "dispersion_slope_s": 0.0}
    )
    model = create_model(
        "cgn", grid_for(case), zero_dispersion, symbol_rate_hz=case.symbol_rate_hz
    )
    power = launch_profile(case, None)
    single = model.evaluate_path(_homogeneous_path(case, 1, zero_dispersion), power)
    ten = model.evaluate_path(_homogeneous_path(case, 10, zero_dispersion), power)
    ratio = ten.spm_noise_w[20] / single.spm_noise_w[20]
    assert ratio == pytest.approx(100.0, rel=1e-4)
    ratio_xpm = ten.xpm_noise_w[20] / single.xpm_noise_w[20]
    assert ratio_xpm == pytest.approx(100.0, rel=1e-4)


def test_large_dispersion_limit_is_nearly_incoherent(monkeypatch: pytest.MonkeyPatch) -> None:
    r"""色散很大：跨间接近非相干，NLI ∝ N（10 跨 → 约 10 倍单跨）。

    这里必须把相关域的截断窗口放宽（``TRUNCATION_FACTOR``）：默认的 40 倍
    谱衰减尺度只覆盖约 ``40·α_x·L/(2π)`` 个 Dirichlet 周期，与 :math:`\\beta_2`
    无关，所以默认档下再大的色散也到不了非相干极限（见交付报告的"已知限制"）。
    """
    import network_nlo_eval.models.cgn as cgn_module

    monkeypatch.setattr(cgn_module, "TRUNCATION_FACTOR", 400.0)
    case = build_case("scaling_40ch")
    strong = SMF_100KM.model_copy(update={"dispersion_parameter_d": 1e4})
    model = create_model("cgn", grid_for(case), strong, symbol_rate_hz=case.symbol_rate_hz)
    power = launch_profile(case, None)
    single = model.evaluate_path(_homogeneous_path(case, 1, strong), power)
    ten = model.evaluate_path(_homogeneous_path(case, 10, strong), power)
    ratio = ten.spm_noise_w[20] / single.spm_noise_w[20]
    assert ratio == pytest.approx(10.0, rel=0.05)


def test_long_span_is_nearly_incoherent() -> None:
    r"""累计色散足够大（跨长 150 km）时，跨间接近非相干：NLI ∝ N。

    这是不改动模型任何参数就能走到的解析极限端：相邻跨之间的相位失配正比于
    :math:`\\beta_2 L`，跨长加倍即把失配加倍。
    """
    case = build_case("scaling_40ch")
    long_span = SMF_100KM.model_copy(update={"length_km": 150.0, "max_span_length_km": 150.0})
    model = create_model("cgn", grid_for(case), long_span, symbol_rate_hz=case.symbol_rate_hz)
    power = launch_profile(case, None)
    fiber = long_span
    edfa = build_path(case).edfa_configs[0]
    single = model.evaluate_path(MultiSpanOpticalPath([fiber], [edfa]), power)
    ten = model.evaluate_path(MultiSpanOpticalPath([fiber] * 10, [edfa] * 10), power)
    ratio = ten.spm_noise_w[20] / single.spm_noise_w[20]
    assert ratio == pytest.approx(10.0, rel=0.10)


def test_phase_comes_from_the_span_context() -> None:
    """相位只由 ``SpanContext`` 决定：跨序号不同 → 相位不同。

    这条守住"``span_index`` 只能来自 ``SpanContext``"的约定：模型不得自己
    维护跨计数器（那正是上一轮调用历史 bug 的来源）。
    """
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    kernel.n_coherent_spans = 10
    phases = []
    for index in (0, 1, 2):
        context = SpanContext(
            span_index=index,
            total_spans=10,
            cumulative_length_m=index * 100e3,
            span_length_m=100e3,
        )
        out = kernel.evaluate(power, 100e3, 1, span_context=context)
        assert out.nli_amplitude_spm is not None
        phases.append(np.angle(out.nli_amplitude_spm[20]))
    # 逐跨相位线性增长，且步长等于斜坡本身。
    step = phases[1] - phases[0]
    assert step != 0.0
    assert phases[2] - phases[1] == pytest.approx(step, rel=1e-9)


def test_heterogeneous_link_keeps_the_power_convention() -> None:
    """异构链不进入相干分支：结果等于逐跨功率相加。

    论文 Eq. (53) 的 XS 项本实现未覆盖，异构链没有闭式相干解，退回纯功率
    口径（与改造前一致）。
    """
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    base_fiber = build_path(case).fiber_configs[0]
    edfa = build_path(case).edfa_configs[0]
    short = base_fiber.model_copy(update={"length_km": 80.0, "max_span_length_km": 80.0})
    fibers = [base_fiber, short]
    result = model.evaluate_path(MultiSpanOpticalPath(fibers, [edfa] * 2), power)

    # 纯功率口径：跨间功率传输因子为 1（放大器补偿跨损），总量 = 逐跨之和。
    total = np.zeros(case.num_channels)
    for fiber in fibers:
        kernel = model.kernel_for(fiber)
        kernel.n_coherent_spans = 1
        out = kernel.evaluate(power, fiber.length_km * 1e3, 1)
        total += out.spm_noise_w
    assert np.allclose(result.spm_noise_w, total, rtol=1e-9, atol=0.0)


def test_coherent_ramp_does_not_depend_on_call_history() -> None:
    """相位斜坡的缓存是幂等的：先调别的功率再调回来，结果逐位相同。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    path = _homogeneous_path(case, 10)
    fresh = model.evaluate_path(path, launch_profile(case, -4.0))

    reused = _model("scaling_40ch")
    reused.evaluate_path(path, launch_profile(case, 0.0))
    reused.evaluate_path(path, launch_profile(case, 4.0))
    after = reused.evaluate_path(path, launch_profile(case, -4.0))

    assert np.array_equal(fresh.spm_noise_w, after.spm_noise_w)
    assert np.array_equal(fresh.snr_db, after.snr_db)


def test_coherent_factor_stays_within_the_dirichlet_bounds() -> None:
    """相干因子必须落在 ``[0, N²]`` 内（Dirichlet 核的取值范围）。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    entry = kernel._core_for(power, 100e3, 10)
    assert entry.ramp_spm is not None and entry.ramp_xpm is not None
    for ramp in (entry.ramp_spm, entry.ramp_xpm):
        assert np.all(ramp >= 0.0)
        assert np.all(ramp <= 2.0 * np.pi / 10.0 + 1e-12)
    # 斜坡给出的相干因子就是链路总量 ÷ 单跨总量。
    link_spm, link_xpm = kernel.link_nli(power, 100e3, 10)
    single_spm, single_xpm = kernel.link_nli(power, 100e3, 1)
    factor = _dirichlet_weight(entry.ramp_spm, 10)
    assert np.allclose(factor * single_spm, link_spm, rtol=1e-9, atol=0.0)
    factor = _dirichlet_weight(entry.ramp_xpm, 10)
    assert np.allclose(factor * single_xpm, link_xpm, rtol=1e-9, atol=0.0)
