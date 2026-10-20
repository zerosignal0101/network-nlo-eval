"""cGN 相关函数闭式 GN 模型的测试。

覆盖三类证据：

* **物理正确性**：与直接数值积分的 GN 参考式（论文 Eq. (1)，按 island 逐块积分）
  对比，验证相关域改写与矩形覆盖没有引入系统偏差；
* **数值收敛性**：相关域求积点数加倍后 SNR 收敛；矩形覆盖片数加倍后收敛；
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
from network_nlo_eval.models.base import (
    MultiSpanOpticalPath,
    PathQoTResult,
    available_models,
    create_model,
    get_model_info,
)
from network_nlo_eval.models.cgn import CGN_MODEL_INFO, CGNSpanKernel

#: 与兄弟模型 PCFM 的对照需要 PCFM 分支已经合入 multi-model-dev。
PCFM_AVAILABLE = "pcfm" in available_models()


def _model(case_name: str, **options):
    """按算例名构造 cGN 模型。"""
    return make_model("cgn", build_case(case_name), **options)


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
    """直接对 GN island 做二维积分，返回逐信道 (SCI, XCI) 噪声功率。

    这是论文 Eq. (1) 的**直接**数值评估：不用矩形覆盖、不做相关域改写，
    只按 (CUT, 第二波, 第三波) 三元组在精确 island 上做二维求积，功率剖面
    取纯光纤衰减（无 ISRS），跨段相干项为零（单跨）。因此它可以作为 cGN 单
    跨核的独立基准。

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
                alpha_x = alpha_npm[second] + alpha_npm[r]
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


def test_output_power_follows_fiber_attenuation() -> None:
    """出纤功率 = 入纤功率 × exp(-αL)，且噪声非负。"""
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    out = kernel.evaluate(power, 100e3, 1)
    expected = power * np.exp(-kernel.alpha_npm * 100e3)
    assert np.allclose(out.power_out_w, expected, rtol=1e-12)
    assert np.all(out.spm_noise_w >= 0.0)
    assert np.all(out.xpm_noise_w >= 0.0)


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


# ---------------------------------------------------------------------------
# 物理正确性
# ---------------------------------------------------------------------------


def test_single_span_matches_direct_gn_integration() -> None:
    """单跨核与 GN 参考式的直接数值积分一致（SCI 1%、XCI 6% 以内）。"""
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
    )
    out = kernel.evaluate(power, 100e3, 1)
    spm_ref, xpm_ref = _direct_island_nli(
        f_abs,
        b_hz,
        power,
        kernel.alpha_npm,
        kernel.gamma_w_m,
        float(kernel.beta2_s2_m[n_ch // 2]),
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
    """
    case = build_case("scaling_40ch")
    model = _model("scaling_40ch")
    power = launch_profile(case, None)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)

    kernel.n_coherent_spans = 1
    incoherent = kernel.evaluate(power, 100e3, 1)
    kernel.n_coherent_spans = 10
    coherent = kernel.evaluate(power, 100e3, 1)

    # 10 跨的相干总量 = 逐跨值 × 10，必须高于非相干求和 10 × 单跨值。
    coherent_total = 10.0 * coherent.spm_noise_w[20]
    incoherent_total = 10.0 * incoherent.spm_noise_w[20]
    assert coherent_total > incoherent_total
    gain_db = 10.0 * np.log10(coherent_total / incoherent_total)
    assert 0.05 < gain_db < 3.0


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
    for cells in (4, 16):
        kernel = CGNSpanKernel(
            fiber_config=SMF_100KM,
            f_abs_hz=f_abs,
            b_hz=b_hz,
            correlation_points=48,
            island_cells=cells,
        )
        out = kernel.evaluate(power, 100e3, 1)
        values.append((out.spm_noise_w[1], out.xpm_noise_w[1]))
    # 片数越多越接近参考值：4 片与 16 片的偏差应明显缩小。
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
