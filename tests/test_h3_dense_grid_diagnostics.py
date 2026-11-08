"""任务 V：密集 C 栅格的 H3 误差来源分解与只读诊断。

本文件自包含复现第五/七轮矩阵里的边界点：64 GBaud 塞进 50 / 37.5 GHz、
6 dBm/信道、1 跨、C 波段，参考仍是 ``numerical``（ISRS-GN）。它不读
``results/`` 或 ``data/``，只经由 ``benchmarks.cases`` 构造算例。

目标不是把默认数字改小，而是把 ``dense_grid_overlap`` 一个布尔 flag 拆成
可核查的分量，并钉住三件事：

* 误差几乎全部来自 NLI（信号与 ASE 逐位一致）；
* SCI 小 ``X`` 分支、SPP 多项式拟合残差、XCI 边缘/求积重采样都不是主因；
* 提高分辨率/多项式阶数只能把最大乐观偏差挪动约 0.02–0.04 dB，剩余的
  1.4–2.4 dB 来自 GN 非重叠假设失效，只能 flag/降级。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm import DENSE_GRID_SPACING_RATIO
from network_nlo_eval.models.pcfm_kernels import sci_small_x_branch_mask


def _case(spacing_hz: float) -> BenchCase:
    """构造与精度矩阵同源的 C 波段单跨算例（64 GBaud, 100 km, NF 5.5 dB）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"task_v_dense_{spacing_hz / 1e9:g}",
        bands="C",
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


@pytest.fixture(scope="module", params=[50e9, 37.5e9], ids=["50ghz", "37.5ghz"])
def dense_run(request):
    """模块级缓存：H3 与 numerical 各跑一次，供分解和只读测试共用。"""
    spacing_hz = float(request.param)
    case = _case(spacing_hz)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case)
    h3 = model.evaluate_path(path, power)
    reference = make_model("numerical", case).evaluate_path(path, power)
    return {
        "spacing_hz": spacing_hz,
        "case": case,
        "path": path,
        "power": power,
        "model": model,
        "h3": h3,
        "reference": reference,
    }


def _db_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """逐信道 ``10 log10(a / b)``，保持完整精度。"""
    return 10.0 * np.log10(np.asarray(numerator) / np.asarray(denominator))


def test_dense_grid_reproduces_the_numerical_optimism(dense_run) -> None:
    """复现边界点：信号/ASE 一致，全部乐观偏差集中在 NLI。

    实测（本基线）最大 ``H3 - numerical`` 为 1.459 dB @ 50 GHz、2.421 dB
    @ 37.5 GHz；信号与 EDFA ASE 的逐信道比值在 1e-3 dB 以内。
    """
    spacing = dense_run["spacing_hz"]
    h3 = dense_run["h3"]
    reference = dense_run["reference"]
    delta = h3.snr_db - reference.snr_db

    # 信号与 ASE 不是误差来源：两者与 numerical 的逐信道比值都在浮点噪声内。
    assert float(np.max(np.abs(_db_ratio(h3.signal_power_w, reference.signal_power_w)))) < 1e-9
    assert float(np.max(np.abs(_db_ratio(h3.ase_noise_w, reference.ase_noise_w)))) < 1e-2

    # 偏差集中在 NLI，且 H3 系统性乐观（NLI 偏低）。
    nli_h3 = h3.spm_noise_w + h3.xpm_noise_w
    nli_ref = reference.spm_noise_w + reference.xpm_noise_w
    nli_delta = _db_ratio(nli_h3, nli_ref)
    assert float(np.median(nli_delta)) < -1.0
    assert bool(np.all(nli_delta < 0.0))

    if spacing == 50e9:
        assert 1.3 < float(np.max(delta)) < 1.6
        assert -1.7 < float(np.median(nli_delta)) < -1.3
    else:
        assert 2.2 < float(np.max(delta)) < 2.6
        assert -2.7 < float(np.median(nli_delta)) < -2.3

    # 最差信道存在且是有限数；分解表以它为锚点。
    worst = int(np.argmax(delta))
    assert np.isfinite(delta[worst])
    assert delta[worst] > 1.0


@pytest.mark.parametrize("spacing_hz", [50e9, 37.5e9])
def test_diagnostics_expose_the_overlapping_xci_weight(spacing_hz: float) -> None:
    """诊断字段必须直接指出：相邻矩形谱重叠，且 XCI 行权重有 20–45 % 在其中。

    ``dense_grid`` 只说明整条栅格越界；``min_guard_hz < 0`` 说明至少一对面
    重叠；``xci_overlap_weight_fraction`` 说明该失效面在闭式 XCI 权重里占的
    份额。C 波段的最近邻对恰好是唯一的重叠对。
    """
    case = _case(spacing_hz)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case)
    diagnostics = model.dense_grid_diagnostics(power)

    assert diagnostics.dense_grid
    assert diagnostics.spacing_to_bandwidth == pytest.approx(spacing_hz / 64e9)
    assert diagnostics.spacing_to_bandwidth < DENSE_GRID_SPACING_RATIO
    # 最近邻重叠量：|Δf| - 2B/2 = spacing - 64 GHz。
    assert diagnostics.min_guard_hz == pytest.approx(spacing_hz - 64e9)

    assert float(np.max(diagnostics.xci_overlap_pairs)) >= 1.0
    assert 0.2 < float(np.max(diagnostics.xci_overlap_weight_fraction)) < 0.5
    assert diagnostics.xci_overlap_pairs.shape == (case.num_channels,)
    assert diagnostics.xci_overlap_weight_fraction.shape == (case.num_channels,)

    # 本算例的三个"数值嫌疑"都被排除：小 X 分支未用、边缘修正未触发、
    # 拟合残差是小量（H3 默认 poly_degree=3，相对 RMS 约 2.7–3.1 %）。
    assert diagnostics.sci_taylor_fraction == 0.0
    assert not bool(np.any(diagnostics.sci_taylor))
    assert not bool(np.any(diagnostics.xci_edge_gated))
    assert 0.0 < diagnostics.spp_fit_residual_rel_rms < 0.05
    assert diagnostics.spp_fit_residual is not None
    assert float(np.max(diagnostics.spp_fit_residual)) == pytest.approx(
        diagnostics.spp_fit_residual_max, rel=1e-12
    )
    assert diagnostics.xci_edge_quadrature_points >= 2

    # 同一组字段也进入逐跨 validity_report，模型层无需再自己展开矩阵。
    report = model.kernel_for(build_path(case).fiber_configs[0]).validity_report(
        100e3, power
    )
    assert report.dense_grid
    assert report.min_guard_hz == pytest.approx(spacing_hz - 64e9)
    assert np.array_equal(report.xci_overlap_pairs, diagnostics.xci_overlap_pairs)
    assert np.allclose(
        report.xci_overlap_weight_fraction,
        diagnostics.xci_overlap_weight_fraction,
        rtol=0.0,
        atol=1e-15,
    )
    assert report.sci_taylor_fraction == 0.0


def test_diagnostics_are_read_only_for_the_default_path(dense_run) -> None:
    """读取诊断前后，``evaluate_path`` 的所有输出数组必须逐位相同。"""
    model = dense_run["model"]
    path = dense_run["path"]
    power = dense_run["power"]

    before = model.evaluate_path(path, power)
    diagnostics = model.dense_grid_diagnostics(power)
    kernel = model.kernel_for(model.ref_fiber_config)
    kernel.validity_report(100e3, power)
    after = model.evaluate_path(path, power)

    assert diagnostics.dense_grid
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(np.asarray(getattr(before, name)), np.asarray(getattr(after, name))), name


@pytest.mark.parametrize("spacing_hz", [50e9, 37.5e9])
def test_xci_edge_quadrature_and_polynomial_degree_do_not_close_the_gap(spacing_hz: float) -> None:
    """边缘求积与更高阶数是"免费但不治病"的：省时/细微，但挪不动 1.4–2.4 dB。

    * ``resolution="fast"`` 只把 XCI 边缘积分降到 12 节点；本算例没有行被
      gate，所以结果必须与默认**逐位相同**。
    * ``edge_fix=False`` 同样逐位相同，说明边缘项不是误差来源。
    * 提高 ``spp_steps`` / ``poly_degree`` 至收敛参考，最大乐观偏差只改善
      约 0.02–0.04 dB，仍远大于窄带 1.41 dB margin。
    """
    case = _case(spacing_hz)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)

    default = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    fast = make_model("pcfm_h3_raman", case, resolution="fast").evaluate_path(path, power)
    no_edge = make_model("pcfm_h3_raman", case, edge_fix=False).evaluate_path(path, power)
    accurate = make_model(
        "pcfm_h3_raman", case, poly_degree=9, spp_steps=96
    ).evaluate_path(path, power)

    fields = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")
    for name in fields:
        assert np.array_equal(np.asarray(getattr(default, name)), np.asarray(getattr(fast, name))), name
        assert np.array_equal(np.asarray(getattr(default, name)), np.asarray(getattr(no_edge, name))), name

    default_max = float(np.max(default.snr_db - reference.snr_db))
    accurate_max = float(np.max(accurate.snr_db - reference.snr_db))
    # 只改善 0.02–0.04 dB；给 0.1 dB 的余量以免平台差异误伤。
    assert default_max - accurate_max < 0.1
    assert accurate_max > (1.3 if spacing_hz == 50e9 else 2.3)


def test_non_dense_plan_has_no_overlap_diagnostics() -> None:
    """75 GHz 在已验证域内：无重叠对、无 guard 负值、无小 X 分支。"""
    case = _case(75e9)
    power = launch_profile(case, 6.0)
    diagnostics = make_model("pcfm_h3_raman", case).dense_grid_diagnostics(power)

    assert not diagnostics.dense_grid
    assert diagnostics.spacing_to_bandwidth > 1.0
    assert diagnostics.min_guard_hz == pytest.approx(75e9 - 64e9)
    assert not bool(np.any(diagnostics.xci_overlap_pairs))
    assert not bool(np.any(diagnostics.xci_overlap_weight_fraction))
    assert diagnostics.sci_taylor_fraction == 0.0
    assert 0.0 < diagnostics.spp_fit_residual_rel_rms < 0.05


def test_sci_small_x_branch_mask_matches_the_kernel_switch() -> None:
    """共享诊断 helper 的条件必须与 JIT 内分支一致（严格小于 switch）。"""
    x = np.array([-2.0, -1.999, 0.0, 1.999, 2.0])
    mask = sci_small_x_branch_mask(x, 2.0)

    assert np.array_equal(mask, np.array([False, True, True, True, False]))
    assert not bool(np.any(sci_small_x_branch_mask(x, 0.0)))
