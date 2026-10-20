"""Buglia 2023 闭式 ISRS-GN 模型的测试。

覆盖任务要求的六项测试，外加把论文闭式与数值 GN 积分对照的一致性测试。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import ModelInfo, available_models
from network_nlo_eval.models.buglia2023 import (
    BUGLIA2023_MODEL_INFO,
    Buglia2023Model,
    alpha_tilde_kappa,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_gamma_coeff_scalar,
    _calc_span_noise_and_power_jit,
)
from network_nlo_eval.models.route import RouteModel
from network_nlo_eval.network.elements import FiberSpanConfig

#: 拉曼增益谱斜率，与 ``models.buglia2023`` 同口径。
C_R = 0.028 / 1e3 / 1e12

#: 测试用的低损耗光纤 (dB/km)。比标准 SMF 的 0.2 dB/km 低，但远离 ULL 的 0.04 dB/km。
LOW_LOSS_DB_KM = 0.1


# =============================================================================
# 辅助：Semrau 2019 闭式 GN 的独立转写
# =============================================================================
def semrau2019_nli(
    power_w: np.ndarray,
    length_m: float,
    alpha_npm: np.ndarray,
    f_rel_hz: np.ndarray,
    lambdas_m: np.ndarray,
    a_eff_m2: np.ndarray,
    beta2_s2_m: float,
    beta3_s3_m: float,
    b_hz_channels: np.ndarray,
    c_r: float = C_R,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Semrau 2019 闭式 GN 的 SPM/XPM 噪声功率 (W)。

    逐字转写 Semrau et al., JLT 2019, doi:10.1109/JLT.2019.2927607 的
    Eq. (10) (SPM) 与 Eq. (11) (XPM)，取其默认的 :math:`\bar\alpha = \alpha`：

    * :math:`\phi_i = \tfrac32\pi^2(\beta_2 + 2\pi\beta_3 f_i)`、
      :math:`\phi_{i,k} = 2\pi^2(f_k-f_i)[\beta_2+\pi\beta_3(f_i+f_k)]`；
    * :math:`T_i = (\alpha+\bar\alpha - P_{tot} C_r f_i)^2`、:math:`A = \alpha+\bar\alpha`。

    与 Buglia 2023 的差别正是论文的卖点：Semrau 的链路函数用了
    Eq. (42) 的 :math:`1/(-\alpha_{l,i}+j\phi)`（要求 :math:`\alpha_{l,i}L\to\infty`），
    Buglia 换成了 Eq. (30) 的一阶 Taylor 匹配。

    Args:
        power_w: ``(N,)`` 入纤功率 (W)。
        length_m: 跨段长度 (m)，Semrau 公式本身不含 L（被 :math:`\alpha L\to\infty` 吸收）。
        alpha_npm: ``(N,)`` 逐信道衰减 (Np/m)。
        f_rel_hz: ``(N,)`` 相对频率 (Hz)。
        lambdas_m: ``(N,)`` 波长 (m)。
        a_eff_m2: ``(N,)`` 有效模面积 (m^2)。
        beta2_s2_m: :math:`\beta_2` (s^2/m)。
        beta3_s3_m: :math:`\beta_3` (s^3/m)。
        b_hz_channels: ``(N,)`` 逐信道带宽 (Hz)。
        c_r: 拉曼增益谱斜率 [1/(W·m·Hz)]。
    Returns:
        ``(spm_w, xpm_w)``，均为 ``(N,)``。
    """
    power = np.asarray(power_w, dtype=np.float64)
    p_total = float(np.sum(power))
    n = power.shape[0]
    t_factor = (2.0 * alpha_npm - p_total * c_r * f_rel_hz) ** 2
    a_sum = 2.0 * alpha_npm

    spm = np.zeros(n, dtype=np.float64)
    xpm = np.zeros(n, dtype=np.float64)
    for i in range(n):
        phi_i = 1.5 * np.pi**2 * (beta2_s2_m + 2 * np.pi * beta3_s3_m * f_rel_hz[i])
        gamma_ii = _calc_gamma_coeff_scalar(
            lambdas_m[i], lambdas_m[i], 2.6e-20, a_eff_m2[i], a_eff_m2[i]
        )
        bracket = (t_factor[i] - alpha_npm[i] ** 2) / alpha_npm[i] * np.arcsinh(
            phi_i * b_hz_channels[i] ** 2 / (np.pi * alpha_npm[i])
        ) + (a_sum[i] ** 2 - t_factor[i]) / a_sum[i] * np.arcsinh(
            phi_i * b_hz_channels[i] ** 2 / (np.pi * a_sum[i])
        )
        eta_spm = (
            (4 / 9)
            * (gamma_ii**2 / b_hz_channels[i] ** 2)
            * (np.pi / (phi_i * alpha_npm[i] * (2 * alpha_npm[i] + alpha_npm[i])))
            * bracket
        )
        spm[i] = eta_spm * power[i] ** 3

        acc = 0.0
        for k in range(n):
            if k == i:
                continue
            phi_ik = 2 * np.pi**2 * (f_rel_hz[k] - f_rel_hz[i]) * (
                beta2_s2_m + np.pi * beta3_s3_m * (f_rel_hz[i] + f_rel_hz[k])
            )
            gamma_ik = _calc_gamma_coeff_scalar(
                lambdas_m[i], lambdas_m[k], 2.6e-20, a_eff_m2[i], a_eff_m2[k]
            )
            bracket_k = (t_factor[k] - alpha_npm[k] ** 2) / alpha_npm[k] * np.arctan(
                phi_ik * b_hz_channels[i] / alpha_npm[k]
            ) + (a_sum[k] ** 2 - t_factor[k]) / a_sum[k] * np.arctan(
                phi_ik * b_hz_channels[i] / a_sum[k]
            )
            acc += (
                (power[k] / power[i]) ** 2
                * (gamma_ik**2 / b_hz_channels[k])
                * (1.0 / (phi_ik * alpha_npm[k] * (2 * alpha_npm[k] + alpha_npm[k])))
                * bracket_k
            )
        xpm[i] = (32 / 27) * power[i] ** 3 * acc
    return spm, xpm


# =============================================================================
# 辅助：构造被测模型
# =============================================================================
def make_fiber(
    span_km: float,
    attenuation_db_km: float = LOW_LOSS_DB_KM,
    use_polynomial: bool = False,
) -> FiberSpanConfig:
    """构造单跨光纤配置。

    Args:
        span_km: 跨段长度 (km)，同时作为 ``max_span_length_km``，避免被拆成多个跨段。
        attenuation_db_km: 常值衰减 (dB/km)。
        use_polynomial: 是否启用波长相关的衰减多项式。
    Returns:
        光纤配置。
    """
    return FiberSpanConfig(
        length_km=span_km,
        max_span_length_km=span_km,
        attenuation_db_km_ref=attenuation_db_km,
        use_attenuation_polynomial=use_polynomial,
    )


def make_model(
    num_channels: int = 8,
    spacing_hz: float = 50e9,
    center_hz: float = 193.1e12,
    symbol_rate_hz: float = 50e9,
    fiber: FiberSpanConfig | None = None,
) -> Buglia2023Model:
    """构造被测模型。

    Args:
        num_channels: 信道数。
        spacing_hz: 信道间隔 (Hz)。
        center_hz: 波段中心频率 (Hz)。
        symbol_rate_hz: 符号率 (Hz)。
        fiber: 参考光纤配置。
    Returns:
        模型实例。
    """
    grid = SpectrumGrid(num_channels, spacing_hz, center_hz)
    ref = fiber if fiber is not None else make_fiber(60.0)
    return Buglia2023Model(grid, ref, symbol_rate_hz=symbol_rate_hz)


def run_span(
    model: Buglia2023Model,
    fiber: FiberSpanConfig,
    span_km: float,
    power_w: np.ndarray,
):
    """跑一个跨段并返回内核输出。

    Args:
        model: 模型。
        fiber: 光纤配置。
        span_km: 跨段长度 (km)。
        power_w: ``(N,)`` 入纤功率 (W)。
    Returns:
        ``SpanFiberOutput``。
    """
    kernel = model.kernel_for(fiber)
    return kernel.evaluate(power_w, span_km * 1e3, 1)


# =============================================================================
# 任务要求的六项测试
# =============================================================================
def test_matches_semrau2019_on_short_low_loss_span() -> None:
    """60 km 低损耗窄带：Buglia 2023 应与 Semrau 2019 接近（< 0.5 dB）。

    这正是 Semrau 近似最接近真值的区域：窄带下拉曼项可忽略，两者只差
    链路函数近似（论文 Eq. (30) vs Semrau 的 Eq. (42)）在 60 km 上的残差。
    """
    fiber = make_fiber(60.0)
    model = make_model(num_channels=8, spacing_hz=50e9, fiber=fiber)
    kernel = model.kernel_for(fiber)
    power = np.full(model.n_nodes, 1e-3)  # 0 dBm/信道

    out = run_span(model, fiber, 60.0, power)
    ref_spm, ref_xpm = semrau2019_nli(
        power,
        60e3,
        kernel.alpha_npm,
        kernel.f_rel_hz,
        kernel.lambdas_m,
        kernel.a_eff_m2,
        kernel.beta2_s2_m,
        kernel.beta3_s3_m,
        kernel.b_hz_channels,
    )

    nli_buglia = out.spm_noise_w + out.xpm_noise_w
    nli_semrau = ref_spm + ref_xpm
    assert np.all(nli_buglia > 0)
    assert np.all(nli_semrau > 0)
    delta_db = 10 * np.log10(nli_buglia / nli_semrau)
    assert np.max(np.abs(delta_db)) < 0.5, f"Buglia vs Semrau NLI mismatch: {delta_db} dB"


def test_long_span_differs_meaningfully() -> None:
    """120 km 长跨：与 Semrau 2019 的差必须可测（> 0.05 dB），且不能写死成 100 km。"""
    fiber = make_fiber(120.0)
    model = make_model(num_channels=8, spacing_hz=50e9, fiber=fiber)
    kernel = model.kernel_for(fiber)
    power = np.full(model.n_nodes, 1e-3)

    out_120 = run_span(model, fiber, 120.0, power)
    ref_spm, ref_xpm = semrau2019_nli(
        power,
        120e3,
        kernel.alpha_npm,
        kernel.f_rel_hz,
        kernel.lambdas_m,
        kernel.a_eff_m2,
        kernel.beta2_s2_m,
        kernel.beta3_s3_m,
        kernel.b_hz_channels,
    )
    delta_db = 10 * np.log10(
        (out_120.spm_noise_w + out_120.xpm_noise_w) / (ref_spm + ref_xpm)
    )
    assert np.max(np.abs(delta_db)) > 0.05, f"120 km difference not measurable: {delta_db} dB"

    # 跨长必须真的进公式：120 km 的结果不能等于 100 km 的结果。
    out_100 = run_span(model, fiber, 100.0, power)
    assert not np.allclose(out_120.spm_noise_w, out_100.spm_noise_w, rtol=1e-9, atol=0.0)
    assert not np.allclose(out_120.xpm_noise_w, out_100.xpm_noise_w, rtol=1e-9, atol=0.0)


def test_arbitrary_span_length_continuity() -> None:
    r"""50/100/150 km：NLI 连续、不出现跳变或 NaN，并且总体单调上升。

    物理背景：单跨的 NLI 随跨长**饱和**而不是线性增长——链路函数的幅值
    :math:`|\int_0^L e^{-\alpha z}e^{j\phi z}dz|` 本身就随 L 趋近
    :math:`1/(\alpha^2+\phi^2)`，所以 50 -> 150 km 的增幅只有几个百分点。

    论文闭式 (30) 相对精确 GN 积分有 <=0.6% 的非单调残差（论文自述整体精度
    0.93 dB），因此这里用 2% 的容差断言“单调不减”，并额外要求总趋势严格上升。
    """
    fiber = make_fiber(150.0)
    model = make_model(num_channels=8, spacing_hz=50e9, fiber=fiber)
    power = np.full(model.n_nodes, 1e-3)

    lengths = [50.0, 100.0, 150.0]
    nli = []
    for span_km in lengths:
        out = run_span(model, fiber, span_km, power)
        total = out.spm_noise_w + out.xpm_noise_w
        assert np.all(np.isfinite(total))
        assert np.all(total > 0)
        nli.append(total)
    nli = np.stack(nli)

    # 连续性：相邻跨长的相对变化在 (0.5, 2) 倍之间，无跳变。
    for i in range(len(lengths) - 1):
        ratio = nli[i + 1] / nli[i]
        assert np.all(ratio < 2.0), f"jump between {lengths[i]} and {lengths[i + 1]} km: {ratio}"
        assert np.all(ratio > 0.5), f"drop between {lengths[i]} and {lengths[i + 1]} km: {ratio}"

    # 单调性：容差内单调不减（闭式残差 <=0.6%），且总趋势严格上升。
    for i in range(len(lengths) - 1):
        rel = (nli[i + 1] - nli[i]) / nli[i]
        assert np.all(rel > -0.02), f"non-monotonic NLI between {lengths[i]} and {lengths[i + 1]} km: {rel}"
    assert np.all(nli[-1] > nli[0])


def test_polynomial_attenuation_supported() -> None:
    """``use_attenuation_polynomial=True`` 与常值衰减结果不同但同量级。"""
    power = np.full(160, 1e-3)
    model = make_model(num_channels=160, spacing_hz=75e9, center_hz=190.15e12, fiber=make_fiber(100.0))

    fiber_const = make_fiber(100.0, attenuation_db_km=0.2, use_polynomial=False)
    fiber_poly = make_fiber(100.0, use_polynomial=True)

    out_const = run_span(model, fiber_const, 100.0, power)
    out_poly = run_span(model, fiber_poly, 100.0, power)

    nli_const = out_const.spm_noise_w + out_const.xpm_noise_w
    nli_poly = out_poly.spm_noise_w + out_poly.xpm_noise_w
    assert np.all(nli_poly > 0)
    assert np.all(np.isfinite(nli_poly))
    # 结果必须不同（多项式给的是逐波长衰减）。
    assert not np.allclose(nli_const, nli_poly, rtol=1e-6, atol=0.0)
    # 但仍落在同一量级。
    ratio = nli_poly / nli_const
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0), f"attenuation polynomial changed the order: {ratio}"


def test_egn_is_rejected() -> None:
    """``egn=True`` 必须直接报错：论文只给了 GN。"""
    with pytest.raises(ValueError, match="GN"):
        Buglia2023Model(SpectrumGrid(4, 50e9, 193.1e12), make_fiber(100.0), egn=True)


def test_finite_on_wide_band() -> None:
    """escl_479ch 单跨：SNR 必须有限且为正。"""
    from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
    from network_nlo_eval.benchmarks.runner import make_model as bench_make_model

    case = build_case("escl_479ch_1span")
    model = bench_make_model("buglia2023", case)
    path = build_path(case)
    result = model.evaluate_path(path, launch_profile(case, None))

    assert np.all(np.isfinite(result.snr_db))
    assert np.all(result.snr_db > 0)
    assert np.all(result.snr_db < 60.0)


# =============================================================================
# 论文闭式自身的一致性测试
# =============================================================================
def test_spm_and_xpm_match_paper_equations_17_18() -> None:
    """内核结果必须逐条等于论文 Eq. (17)/(18) 的显式 (l, l') 求和。"""
    fiber = make_fiber(100.0, attenuation_db_km=0.2)
    model = make_model(num_channels=16, spacing_hz=75e9, center_hz=190.15e12, fiber=fiber)
    kernel = model.kernel_for(fiber)
    power = np.full(model.n_nodes, 1e-3)
    length_m = 100e3

    out = run_span(model, fiber, 100.0, power)
    m0, m1, a0, k0, a1, k1 = kernel._profile_coefficients(length_m)
    p_total = float(np.sum(power))
    t_tilde = -p_total * kernel.c_r_w_m_hz * kernel.f_rel_hz / kernel.alpha_npm
    t_factor = 1.0 + t_tilde
    weight = -t_tilde / t_factor

    for i in (0, 5, 15):
        # --- Eq. (18) 的 SPM ---
        phi_i = kernel.phi_i[i]
        s = 0.0
        for ell in range(2):
            for ell_p in range(2):
                a_l, k_l = (a0[i], k0[i]) if ell == 0 else (a1[i], k1[i])
                a_p, k_p = (a0[i], k0[i]) if ell_p == 0 else (a1[i], k1[i])
                s += (
                    weight[i] ** (ell + ell_p)
                    * (2 * np.pi * k_l * k_p / (phi_i * (a_l + a_p)))
                    * (
                        np.arcsinh(3 * phi_i * kernel.b_hz_channels[i] ** 2 / (8 * np.pi * a_l))
                        + np.arcsinh(3 * phi_i * kernel.b_hz_channels[i] ** 2 / (8 * np.pi * a_p))
                    )
                )
        eta_spm = (
            (16 / 27)
            * (kernel.gamma_ii[i] ** 2 / kernel.b_hz_channels[i] ** 2)
            * t_factor[i] ** 2
            * s
        )
        assert out.spm_noise_w[i] == pytest.approx(eta_spm * power[i] ** 3, rel=1e-12)

        # --- Eq. (17) 的 XPM ---
        acc = 0.0
        for k in range(model.n_nodes):
            if k == i:
                continue
            phi_ik = -4 * np.pi**2 * (kernel.f_rel_hz[k] - kernel.f_rel_hz[i]) * (
                kernel.beta2_s2_m + np.pi * kernel.beta3_s3_m * (kernel.f_rel_hz[i] + kernel.f_rel_hz[k])
            )
            s_k = 0.0
            for ell in range(2):
                for ell_p in range(2):
                    a_l, k_l = (a0[k], k0[k]) if ell == 0 else (a1[k], k1[k])
                    a_p, k_p = (a0[k], k0[k]) if ell_p == 0 else (a1[k], k1[k])
                    s_k += (
                        weight[k] ** (ell + ell_p)
                        * (2 * k_l * k_p / (phi_ik * (a_l + a_p)))
                        * (
                            np.arctan(phi_ik * kernel.b_hz_channels[i] / (2 * a_l))
                            + np.arctan(phi_ik * kernel.b_hz_channels[i] / (2 * a_p))
                        )
                    )
            gamma_ik = _calc_gamma_coeff_scalar(
                kernel.lambdas_m[i], kernel.lambdas_m[k], kernel.n2_m2_w, kernel.a_eff_m2[i], kernel.a_eff_m2[k]
            )
            acc += (
                (power[k] / power[i]) ** 2
                * (gamma_ik**2 / kernel.b_hz_channels[k])
                * t_factor[k] ** 2
                * s_k
            )
        assert out.xpm_noise_w[i] == pytest.approx((32 / 27) * power[i] ** 3 * acc, rel=1e-10)


def test_closed_form_spm_matches_gn_integral() -> None:
    r"""论文闭式 SPM 必须复现数值 GN 积分（论文 Fig. 4 的核对方式）。

    这里用无拉曼的单信道情形：数值积分 Eq. (18) 的被积函数
    :math:`|\int_0^L e^{-\alpha z}e^{j\phi f_1 f_2 z}dz|^2`，与闭式对比。
    """
    fiber = make_fiber(100.0, attenuation_db_km=0.2)
    model = make_model(num_channels=4, spacing_hz=75e9, center_hz=193.1e12, fiber=fiber)
    kernel = model.kernel_for(fiber)
    length_m = 100e3
    alpha = float(kernel.alpha_npm[0])
    band = float(kernel.b_hz_channels[0])
    gamma = float(kernel.gamma_ii[0])
    phi = float(kernel.phi_i[0])

    n = 1201
    f1 = np.linspace(-band / 2, band / 2, n)
    f2 = np.linspace(-band / 2, band / 2, n)
    grid1, grid2 = np.meshgrid(f1, f2, indexing="ij")
    p = phi * grid1 * grid2
    num = np.exp(-2 * alpha * length_m) - 2 * np.exp(-alpha * length_m) * np.cos(p * length_m) + 1.0
    integral = np.trapz(np.trapz(num / (alpha**2 + p**2), f2, axis=1), f1)
    exact = (16 / 27) * (gamma**2 / band**2) * integral

    # 无拉曼时闭式退化为 Eq. (18) 的 (l, l') = (0, 0) 项。
    a_tilde, kappa = alpha_tilde_kappa(alpha, length_m)
    closed = (
        (16 / 27)
        * (gamma**2 / band**2)
        * (2 * np.pi * kappa**2 / (phi * (a_tilde + a_tilde)))
        * (2 * np.arcsinh(3 * phi * band**2 / (8 * np.pi * a_tilde)))
    )
    assert closed == pytest.approx(exact, rel=0.05)

    # 内核在无拉曼时（把 C_r 置零）应复现同一闭式。
    kernel.c_r_w_m_hz = 0.0
    try:
        out = run_span(model, fiber, 100.0, np.full(model.n_nodes, 1e-3))
    finally:
        kernel.c_r_w_m_hz = C_R
    eta = out.spm_noise_w[0] / (1e-3**3)
    assert eta == pytest.approx(closed, rel=1e-12)


def test_alpha_tilde_kappa_limits() -> None:
    r"""Eq. (15)/(16) 的两个极限：:math:`\alpha L\to\infty` 退回 Semrau，:math:`\alpha L\to 0` 退化到 :math:`L^2`。"""
    alpha = 0.2 * np.log(10) / 10 / 1e3
    # 大 alpha*L：tilde_alpha -> alpha，kappa -> 1（Semrau 的 Eq. (42)）。
    tilde, kappa = alpha_tilde_kappa(alpha, 1e6)
    assert tilde == pytest.approx(alpha, rel=1e-9)
    assert kappa == pytest.approx(1.0, rel=1e-9)
    # 小 alpha*L：tilde_alpha -> 2/L，kappa -> 2，链路函数在 phi=0 时 -> L^2。
    length = 1.0
    tilde, kappa = alpha_tilde_kappa(alpha * 1e-9, length)
    assert tilde == pytest.approx(2.0 / length, rel=1e-6)
    assert kappa == pytest.approx(2.0, rel=1e-6)
    assert (kappa**2 / tilde**2) == pytest.approx(length**2, rel=1e-6)


def test_semrau_reference_matches_frozen_ecz_kernel() -> None:
    """本文件的 Semrau 2019 转写必须与冻结核 ``ecz`` 的 GN 公式一致（关掉拉曼后逐位比较）。"""
    fiber = make_fiber(100.0, attenuation_db_km=0.2)
    model = make_model(num_channels=4, spacing_hz=75e9, center_hz=193.1e12, fiber=fiber)
    kernel = model.kernel_for(fiber)
    power = np.full(model.n_nodes, 1e-3)
    length_m = 100e3

    # 冻结的 ``ecz`` 核（拉曼系数置零 => 纯 GN）。
    p_out, spm, xpm, ase = _calc_span_noise_and_power_jit(
        power_in_w=power,
        span_length_m=length_m,
        f_rel_hz=kernel.f_rel_hz,
        f_abs_hz=kernel.f_rel_hz + 193.1e12,
        lambdas_m=kernel.lambdas_m,
        alpha_power_npm=kernel.alpha_npm,
        a_eff_m2=kernel.a_eff_m2,
        beta2_s2_m=kernel.beta2_s2_m,
        beta3_s3_m=kernel.beta3_s3_m,
        n2_m2_w=kernel.n2_m2_w,
        nf_lin_channels=np.ones(model.n_nodes),
        symbol_rate_hz=50e9,
        b_hz_channels=kernel.b_hz_channels,
        n_effective_spans=1,
        cr_w_m_hz=0.0,
        k_bar=0.0,
        delta_f_co_hz=15e12,
    )
    ref_spm, ref_xpm = semrau2019_nli(
        power,
        length_m,
        kernel.alpha_npm,
        kernel.f_rel_hz,
        kernel.lambdas_m,
        kernel.a_eff_m2,
        kernel.beta2_s2_m,
        kernel.beta3_s3_m,
        kernel.b_hz_channels,
        c_r=0.0,
    )
    assert ref_spm == pytest.approx(spm, rel=1e-12)
    assert ref_xpm == pytest.approx(xpm, rel=1e-10)


def test_model_is_registered() -> None:
    """模型必须注册在冻结的注册表里，family 为 gn。"""
    assert "buglia2023" in available_models()
    assert BUGLIA2023_MODEL_INFO.family == "gn"
    assert "10.1109/JLT.2023.3256185" in BUGLIA2023_MODEL_INFO.reference


def test_registration_uses_model_info() -> None:
    """注册表里的 ``ModelInfo`` 必须和模块里的一致。"""
    assert isinstance(BUGLIA2023_MODEL_INFO, ModelInfo)
    assert BUGLIA2023_MODEL_INFO.name == "buglia2023"


def test_model_is_route_model() -> None:
    """模型必须实现冻结的 ``RouteModel`` 契约。"""
    model = make_model(num_channels=4, fiber=make_fiber(100.0))
    assert isinstance(model, RouteModel)
    assert model.n_nodes == 4
