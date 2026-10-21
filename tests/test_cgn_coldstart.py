"""cGN 构造期融合内核的等价性与并行性测试。

这一轮把 cGN 的核心积分矩阵构造从"纯 numpy 多趟数组运算"改成"融合内核 +
线程并行"。契约是**逐位不变**：融合内核必须与原来的 numpy 链路
(:func:`~network_nlo_eval.models.cgn._isrs_nsp_energy_spectrum` +
:func:`~network_nlo_eval.models.cgn._rectangle_weight`) 给出**逐位相同**的被积
函数，线程并行也必须与单线程给出逐位相同的核心矩阵。

覆盖：

* **逐位等价**：真实输入下，融合内核的被积函数与 numpy 链路逐位相同；
* **并行等价**：核心矩阵在 ``CGN_CORE_WORKERS=1`` 与默认并行度下逐位相同；
* **端到端逐位**：带内积分档（``band_points=9``）的逐信道 SNR / NLI 与并行
  关闭时逐位相同；
* **确定性**：调用历史无关的契约在并行构造下依然成立；
* **冷启动预算**：带内积分档的冷启动不再随节点数线性放大。
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models import cgn as cgn_module
from network_nlo_eval.models.cgn import (
    _canonical_power,
    _fused_integrand_isrs,
    _island_core,
    _isrs_nsp_energy_spectrum,
    _rectangle_weight,
    _xi_quadrature,
)

#: 复现融合内核输入所用的算例（小算例即可覆盖全部几何分支）。
_CASE = "c_57ch"


def _kernel_inputs(case_name: str = _CASE):
    """构造一份真实的融合内核输入（几何、ISRS 剖面、能量谱参数）。

    Returns
    -------
        ``(kernel, canonical, z_mid, gain, rate)``：核对象与规范化入纤功率、
        ISRS 分段中点、逐段增益、逐信道有效衰减率。
    """
    case = build_case(case_name)
    model = make_model("cgn", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    canonical, _ = _canonical_power(launch_profile(case, None))
    z_mid, gain, rate = cgn_module._isrs_span_profile(
        kernel.f_rel_hz,
        kernel.alpha_npm,
        canonical,
        kernel.f_m_hz,
        kernel.f_M_hz,
        kernel.k_bar,
        1.0e5,
        kernel.isrs_segments,
    )
    return kernel, canonical, z_mid, gain, rate


def _quadrant_inputs(kernel, canonical, z_mid, gain, rate):
    """重建 :func:`_island_core` 内部用到的逐象限积分输入。

    Returns
    -------
        字典，含 ``alpha_x`` / ``residual`` / ``beta2_eff`` / 四个象限片的
        ``xi_min``、``xi_end``、``a``、``b``、``c``、``d``。
    """
    n_ch = kernel.n_nodes
    n_slots = kernel._n_slots
    second = np.broadcast_to(
        np.arange(n_ch, dtype=np.int64)[None, :, None], kernel._r_index.shape
    )
    third = np.where(kernel._r_index >= 0, kernel._r_index, 0)
    alpha_x = cgn_module._stimulus_attenuation(kernel.alpha_npm, second, third)
    alpha_x = alpha_x + 0.5 * (rate[second] + rate[third])

    x1 = -0.5 * kernel.b_hz[:, None]
    x2 = 0.5 * kernel.b_hz[:, None]
    delta = kernel.f_abs_hz[None, :] - kernel.f_abs_hz[:, None]
    y1 = delta - 0.5 * kernel.b_hz[None, :]
    y2 = delta + 0.5 * kernel.b_hz[None, :]
    u1 = kernel.f_abs_hz[third] - kernel.f_abs_hz[:, None, None] - 0.5 * kernel.b_hz[third]
    u2 = kernel.f_abs_hz[third] - kernel.f_abs_hz[:, None, None] + 0.5 * kernel.b_hz[third]
    xa, xb, ylo, yhi = cgn_module._island_cover(
        x1, x2, y1, y2, u1, u2, kernel.island_cells
    )
    beta2_eff = cgn_module._effective_beta2(
        kernel.beta2_s2_m,
        kernel.beta3_s3_m,
        0.5 * (xa + xb),
        0.5 * (ylo + yhi),
        f_center_hz=kernel.f_abs_hz.reshape(-1, 1, 1, 1),
        convention=kernel.dispersion_convention,
    )
    residual = np.empty((n_ch, n_ch, n_slots, gain.shape[0]), dtype=np.float64)
    for seg in range(gain.shape[0]):
        gain_seg = gain[seg]
        residual[..., seg] = (gain_seg[None, :, None] * gain_seg[third]) * np.exp(
            (rate[second] + rate[third]) * z_mid[seg]
        )
    decay = 1.0e5 / gain.shape[0]
    pieces = []
    for a, b, c, d in cgn_module._quadrant_pieces(xa, xb, ylo, yhi):
        xi_min = a * c
        xi_max = b * d
        with np.errstate(divide="ignore", invalid="ignore"):
            xi_scale = alpha_x[..., None] / (4.0 * np.pi**2 * np.abs(beta2_eff))
        xi_scale = np.where(np.isfinite(xi_scale) & (xi_scale > 0.0), xi_scale, np.inf)
        xi_cut = np.minimum(xi_max, xi_min + cgn_module.TRUNCATION_FACTOR * xi_scale)
        xi_end = np.maximum(xi_cut, xi_min)
        pieces.append((a, b, c, d, xi_min, xi_end))
    return {
        "alpha_x": alpha_x,
        "residual": residual,
        "beta2_eff": beta2_eff,
        "decay": decay,
        "pieces": pieces,
    }


def _numpy_integrand(kernel, data, piece):
    """按原来的 numpy 链路算被积函数（论文 Eq. (22)/(29) 的直接实现）。"""
    a, b, c, d, xi_min, xi_end = piece
    nodes, weights = _xi_quadrature(xi_min, xi_end, kernel.correlation_points)
    omega = 4.0 * np.pi**2 * data["beta2_eff"][..., None] * nodes
    gamma_c = kernel.gamma_w_m.reshape(-1, 1, 1, 1, 1)
    spectrum = _isrs_nsp_energy_spectrum(
        gamma_c,
        data["alpha_x"][..., None, None],
        data["residual"],
        data["decay"],
        omega,
    )
    weight = _rectangle_weight(a[..., None], b[..., None], c[..., None], d[..., None], nodes)
    integrand = spectrum * weight * weights
    return np.where(np.isfinite(integrand), integrand, 0.0)


def _fused_integrand(kernel, data, piece):
    """按融合内核算同一个被积函数。"""
    a, b, c, d, xi_min, xi_end = piece
    n_ch = kernel.n_nodes
    n_slots = kernel._n_slots
    points = kernel.correlation_points
    cells = kernel.island_cells
    u_nodes = (np.arange(points, dtype=np.float64) + 0.5) / float(points)
    u_edges = np.arange(points + 1, dtype=np.float64) / float(points)
    decay = data["decay"]
    exponent = data["alpha_x"] * decay
    out = np.zeros((n_ch, n_ch, n_slots, cells, points), dtype=np.float64)
    _fused_integrand_isrs(
        out,
        np.ascontiguousarray(data["alpha_x"]),
        np.ascontiguousarray(data["residual"]),
        np.exp(-exponent),
        np.expm1(-exponent),
        np.ascontiguousarray(data["beta2_eff"]),
        np.ascontiguousarray(xi_min),
        np.ascontiguousarray(xi_end - xi_min),
        u_nodes**2,
        u_edges[1:] ** 2 - u_edges[:-1] ** 2,
        np.ascontiguousarray(a[:, 0, 0, :]),
        np.ascontiguousarray(b[:, 0, 0, :]),
        np.ascontiguousarray(c),
        np.ascontiguousarray(d),
        kernel.gamma_w_m**2,
        decay,
        4.0 * np.pi**2,
    )
    return out


def test_fused_kernel_matches_numpy_path_bit_exactly() -> None:
    """融合内核的被积函数与 numpy 链路逐位相同。"""
    kernel, canonical, z_mid, gain, rate = _kernel_inputs()
    data = _quadrant_inputs(kernel, canonical, z_mid, gain, rate)
    for piece in data["pieces"]:
        reference = _numpy_integrand(kernel, data, piece)
        fused = _fused_integrand(kernel, data, piece)
        assert reference.shape == fused.shape
        assert np.array_equal(reference.view(np.uint64), fused.view(np.uint64))


def test_core_matrix_is_independent_of_worker_count(monkeypatch) -> None:
    """核心矩阵与并行度无关（单线程与多线程逐位相同）。"""
    case = build_case("scaling_160ch")
    path = build_path(case)
    power = launch_profile(case, None)

    monkeypatch.setenv("CGN_CORE_WORKERS", "1")
    serial = make_model("cgn", case, band_points=9)
    serial_result = serial.evaluate_path(path, power)

    monkeypatch.delenv("CGN_CORE_WORKERS", raising=False)
    parallel = make_model("cgn", case, band_points=9)
    parallel_result = parallel.evaluate_path(path, power)

    assert np.array_equal(
        serial_result.snr_db.view(np.uint64), parallel_result.snr_db.view(np.uint64)
    )
    assert np.array_equal(
        serial_result.spm_noise_w.view(np.uint64), parallel_result.spm_noise_w.view(np.uint64)
    )
    assert np.array_equal(
        serial_result.xpm_noise_w.view(np.uint64), parallel_result.xpm_noise_w.view(np.uint64)
    )


def test_band_points_nine_is_stable_across_repeated_calls() -> None:
    """带内积分档重复调用逐位相同（构造期并行不引入不确定性）。"""
    case = build_case("scaling_160ch")
    path = build_path(case)
    power = launch_profile(case, None)
    model = make_model("cgn", case, band_points=9)
    first = model.evaluate_path(path, power)
    for _ in range(3):
        again = model.evaluate_path(path, power)
        assert np.array_equal(first.snr_db.view(np.uint64), again.snr_db.view(np.uint64))


def test_call_history_does_not_change_the_result(monkeypatch) -> None:
    """调用历史无关的契约在并行构造下依然成立。"""
    case = build_case("scaling_160ch")
    path = build_path(case)
    model = make_model("cgn", case, band_points=9)

    first = model.evaluate_path(path, launch_profile(case, -2.0))
    model.evaluate_path(path, launch_profile(case, 4.0))
    again = model.evaluate_path(path, launch_profile(case, -2.0))
    assert np.array_equal(first.snr_db.view(np.uint64), again.snr_db.view(np.uint64))


def test_island_core_still_matches_the_fused_kernel() -> None:
    """整条核心矩阵链路（几何 + 融合内核 + 归约）与逐位参考一致。"""
    kernel, canonical, z_mid, gain, rate = _kernel_inputs()
    data = _quadrant_inputs(kernel, canonical, z_mid, gain, rate)
    reference = np.zeros(
        (kernel.n_nodes, kernel.n_nodes, kernel._n_slots), dtype=np.float64
    )
    for piece in data["pieces"]:
        reference += np.sum(_numpy_integrand(kernel, data, piece), axis=(-2, -1))

    core, _ = _island_core(
        f_abs_hz=kernel.f_abs_hz,
        b_hz=kernel.b_hz,
        alpha_npm=kernel.alpha_npm,
        gamma_w_m=kernel.gamma_w_m,
        beta2_s2_m=kernel.beta2_s2_m,
        beta3_s3_m=kernel.beta3_s3_m,
        dispersion_convention=kernel.dispersion_convention,
        nsp_convention=kernel.nsp_convention,
        length_m=1.0e5,
        r_index=kernel._r_index,
        correlation_points=kernel.correlation_points,
        island_cells=kernel.island_cells,
        n_coherent_spans=1,
        isrs_gain=gain,
        isrs_rate=rate,
        isrs_z_mid=z_mid,
        isrs_seg_length_m=data["decay"],
        band_points=1,
    )
    # 归约顺序与核心矩阵一致（逐象限累加），故逐位相同。
    assert np.array_equal(core.view(np.uint64), reference.view(np.uint64))


def test_cold_start_of_the_in_band_mode_is_bounded() -> None:
    """带内积分档的冷启动有界（融合内核 + 并行后不再是分钟级）。"""
    case = build_case("scaling_160ch")
    path = build_path(case)
    power = launch_profile(case, None)
    model = make_model("cgn", case, band_points=9)
    started = time.perf_counter()
    result = model.evaluate_path(path, power)
    elapsed = time.perf_counter() - started
    assert np.all(np.isfinite(result.snr_db))
    # 上界很宽松，只用来挡住"退回到分钟级"的回归。
    assert elapsed < 120.0, f"cold start took {elapsed:.1f} s"


def test_core_worker_count_helper() -> None:
    """并行度解析：环境变量优先，且被任务数夹住。"""
    assert cgn_module._core_workers(1) == 1
    assert cgn_module._core_workers(0) == 1
    previous = os.environ.get("CGN_CORE_WORKERS")
    try:
        os.environ["CGN_CORE_WORKERS"] = "3"
        assert cgn_module._core_workers(10) == 3
        os.environ["CGN_CORE_WORKERS"] = "99"
        assert cgn_module._core_workers(10) == 10
        os.environ["CGN_CORE_WORKERS"] = "not-a-number"
        assert cgn_module._core_workers(10) == 1
    finally:
        if previous is None:
            os.environ.pop("CGN_CORE_WORKERS", None)
        else:
            os.environ["CGN_CORE_WORKERS"] = previous


@pytest.mark.parametrize("band_points", [1, 9])
def test_model_produces_finite_physics(band_points: int) -> None:
    """两档都给出有限物理量。"""
    case = build_case("scaling_160ch")
    model = make_model("cgn", case, band_points=band_points)
    result = model.evaluate_path(build_path(case), launch_profile(case, None))
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(result.spm_noise_w >= 0.0)
    assert np.all(result.xpm_noise_w >= 0.0)
