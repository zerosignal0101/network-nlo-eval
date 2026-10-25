"""cGN 活跃子集（稀疏）路径的等价性、确定性与性能测试。

背景：cGN 的核心积分矩阵是 ``(N, N, J)`` 的整张网格，构造代价随**信道数**
（网格大小）走，而不是随**活跃信道数**走。RWA/RL 评估候选分配时，一条链路上
通常只有几个到几十个信道被占用，其余功率为零，于是绝大多数网格项的功率因子
是零，代价却一分不少。

本轮改造把"功率为零的信道对 NLI 的贡献恒为零"这条事实用起来：核心积分矩阵
只在**活跃信道**上构造，最后散射回全网格。这是**等价变换**（不是截断）：
功率为零的信道在 PSD 因子上贡献恒为零，故被丢掉的项对结果的贡献恒为零。

覆盖：

* **逐信道等价**：稀疏路径与"全网格路径"（改造前的路径，在测试里通过把
  视图选择强制成全网格来复现）逐信道比较，SCI / XCI 噪声的相对误差
  < 1e-12，SNR 的绝对差 < 1e-9 dB；
* **未占用信道**：噪声严格为 0（与全网格路径逐位相同）；
* **全活跃逐位不变**：全部信道都有功率时，输出与改造前的实现逐位相同
  （用改造前跑出来的黄金字节守住）；
* **确定性**：重复调用、调用历史无关的契约在稀疏路径下依然成立；
* **ISRS**：稀疏路径的 ISRS 剖面按**真实（稀疏）功率谱**算，与全网格路径在
  活跃信道上一致；
* **性能**：稀疏路径的冷启动代价随活跃信道数走，而不是随网格大小走。
"""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models import cgn as cgn_module
from network_nlo_eval.models.cgn import CGNSpanKernel

#: 等价性测试用的算例（小算例即可覆盖全部几何分支；dense 路径很快）。
_CASE = "c_57ch"

#: 性能测试用的算例（与交付报告的 headline 场景一致）。
_PERF_CASE = "escl_479ch"

#: 逐位黄金值：全部信道活跃时，改造前的实现（``git`` 里的前一版 ``cgn.py``）
#: 在 ``c_57ch`` / 10 跨 / 标称功率下给出的 SNR（dB）。稀疏改造不得改动它。
_GOLDEN_FULL_ACTIVE_SNR_DB = (
    16.748584486547685,
    16.37637822046809,
    16.268931100746066,
    16.169372502183723,
    16.100531931922376,
    16.049766128728994,
    16.010600287045417,
    15.97936846613457,
    15.953709818252573,
    15.931914565250885,
    15.91304590489697,
    15.897156779213349,
    15.88439092433719,
    15.87457549444686,
    15.867546205131138,
    15.862869831903046,
    15.85932207561748,
    15.855965644508567,
    15.853006824395358,
    15.850838482508237,
    15.849401387017235,
    15.848891298211205,
    15.849308803665567,
    15.850011429241409,
    15.850702947255737,
    15.852148894573471,
    15.855446786788029,
    15.860937405139612,
    15.868019385685114,
    15.875590220333928,
    15.882573734409725,
    15.888628375002689,
    15.894319039632842,
    15.900605987188548,
    15.908995182795309,
    15.920774108513573,
    15.936055849154853,
    15.954016014883798,
    15.973558105411744,
    15.992463702823436,
    16.00896880597291,
    16.023560351529376,
    16.03738690323011,
    16.052164214150107,
    16.07064251692244,
    16.09453011861681,
    16.12272418623785,
    16.153437177583715,
    16.186466220646366,
    16.222210541183312,
    16.26097172292989,
    16.30385148715289,
    16.353309848821805,
    16.413539478296826,
    16.492628644431083,
    16.58969436848067,
    16.89795690138343,
)


def _digest(array: np.ndarray) -> str:
    """数组字节的 SHA-256 摘要（逐位守卫）。"""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


#: 全部信道活跃时（``c_57ch`` / 10 跨 / 标称功率）改造前的实现给出的字节摘要。
_GOLDEN_FULL_ACTIVE_SNR_SHA256 = (
    "c923e758345e25b819109c6ea6b150ad733272c1e996d770144193ce1cd972b1"
)
_GOLDEN_FULL_ACTIVE_SPM_SHA256 = (
    "49d3d71c534eb9dbc25a90fba6da6b7c8e8e07e5fc4d6c35044c1b86b0aa2270"
)
_GOLDEN_FULL_ACTIVE_XPM_SHA256 = (
    "8b0a8e151c95989a22ca35ceeace3692979aeba4a79e6a0d9a073d17a2f2cae3"
)
_GOLDEN_FULL_ACTIVE_SIG_SHA256 = (
    "b9e0b8060f0aa11fd565537f6c1f18dcab3fe4798e9e9373a2f46ed6be04d876"
)


def _dense_model(case, **options):
    """构造一个**强制走全网格路径**的模型，即改造前的路径。

    改造前核心积分矩阵总是按全网格构造。把"活跃子集视图"的选择固定成全网格
    就复现了那条路径，于是稀疏路径可以在同一个进程里逐信道对照。
    """
    patcher = pytest.MonkeyPatch()
    patcher.setattr(CGNSpanKernel, "_subset_view", lambda self, act: self._full_view)
    try:
        return make_model("cgn", case, **options)
    finally:
        patcher.undo()


def _sparse_power(case, n_active, *, contiguous: bool = True, seed: int = 0):
    """构造只有 ``n_active`` 个信道有功率的入纤功率谱。

    Args:
        case: 算例。
        n_active: 活跃信道数。
        contiguous: ``True`` 时活跃信道连成一段，``False`` 时均匀散开。
        seed: 散开时的随机种子。

    Returns
    -------
        ``(power, active_index)``：功率谱 (W) 与活跃信道的索引。
    """
    base = launch_profile(case, None)
    power = np.zeros(case.num_channels, dtype=np.float64)
    if contiguous:
        index = np.unique(np.linspace(0, case.num_channels - 1, n_active).astype(int))
    else:
        rng = np.random.default_rng(seed)
        index = np.sort(rng.choice(case.num_channels, n_active, replace=False))
    power[index] = base[index]
    return power, index


def _max_relative_error(new: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> float:
    """被占用信道上的最大相对误差。"""
    new = np.asarray(new, dtype=np.float64)[mask]
    reference = np.asarray(reference, dtype=np.float64)[mask]
    scale = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(new - reference) / scale))


@pytest.mark.parametrize("n_active", [1, 3, 17, 40])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("band_points", [1, 9])
def test_sparse_path_matches_the_dense_path_channel_by_channel(
    n_active: int, contiguous: bool, band_points: int
) -> None:
    """稀疏路径与全网格路径逐信道一致（相对误差 < 1e-12）。"""
    case = build_case(_CASE)
    path = build_path(case)
    power, index = _sparse_power(case, n_active, contiguous=contiguous)
    mask = power > 0.0

    dense = _dense_model(case, band_points=band_points).evaluate_path(path, power)
    sparse = make_model("cgn", case, band_points=band_points).evaluate_path(path, power)

    assert _max_relative_error(sparse.spm_noise_w, dense.spm_noise_w, mask) < 1e-12
    assert _max_relative_error(sparse.xpm_noise_w, dense.xpm_noise_w, mask) < 1e-12
    finite = np.isfinite(sparse.snr_db) & np.isfinite(dense.snr_db)
    delta = np.abs(sparse.snr_db[finite] - dense.snr_db[finite])
    assert (delta.max() if delta.size else 0.0) < 1e-9
    # 两侧的 "无穷 SNR" 位置必须一致（噪声为零的通道）。
    assert np.array_equal(np.isfinite(sparse.snr_db), np.isfinite(dense.snr_db))


@pytest.mark.parametrize("band_points", [1, 9])
def test_unoccupied_channels_carry_exactly_zero_noise(band_points: int) -> None:
    """未占用信道的噪声严格为 0，且与全网格路径逐位相同。"""
    case = build_case(_CASE)
    path = build_path(case)
    power, _ = _sparse_power(case, 7)
    empty = power <= 0.0

    dense = _dense_model(case, band_points=band_points).evaluate_path(path, power)
    sparse = make_model("cgn", case, band_points=band_points).evaluate_path(path, power)

    assert np.array_equal(sparse.spm_noise_w[empty], dense.spm_noise_w[empty])
    assert np.array_equal(sparse.xpm_noise_w[empty], dense.xpm_noise_w[empty])
    assert np.all(sparse.spm_noise_w[empty] == 0.0)
    assert np.all(sparse.xpm_noise_w[empty] == 0.0)


def test_all_channels_active_is_bit_identical_to_the_previous_implementation() -> None:
    """全部信道活跃时输出逐位不变（改造前跑出来的黄金字节守住）。"""
    case = build_case(_CASE)
    path = build_path(case)
    power = launch_profile(case, None)

    result = make_model("cgn", case).evaluate_path(path, power)

    assert len(result.snr_db) == len(_GOLDEN_FULL_ACTIVE_SNR_DB)
    for got, want in zip(result.snr_db, _GOLDEN_FULL_ACTIVE_SNR_DB, strict=True):
        assert got == want
    # 整张数组的字节摘要：覆盖 SNR 之外的全部输出（逐位守卫）。
    assert _digest(result.snr_db) == _GOLDEN_FULL_ACTIVE_SNR_SHA256
    assert _digest(result.spm_noise_w) == _GOLDEN_FULL_ACTIVE_SPM_SHA256
    assert _digest(result.xpm_noise_w) == _GOLDEN_FULL_ACTIVE_XPM_SHA256
    assert _digest(result.signal_power_w) == _GOLDEN_FULL_ACTIVE_SIG_SHA256


def test_all_channels_active_matches_the_dense_path_bit_for_bit() -> None:
    """全部信道活跃时稀疏机制不介入，输出与全网格路径逐位相同。"""
    case = build_case(_CASE)
    path = build_path(case)
    power = launch_profile(case, None)

    dense = _dense_model(case).evaluate_path(path, power)
    sparse = make_model("cgn", case).evaluate_path(path, power)

    assert np.array_equal(sparse.snr_db.view(np.uint64), dense.snr_db.view(np.uint64))
    assert np.array_equal(sparse.spm_noise_w.view(np.uint64), dense.spm_noise_w.view(np.uint64))
    assert np.array_equal(sparse.xpm_noise_w.view(np.uint64), dense.xpm_noise_w.view(np.uint64))
    assert np.array_equal(
        sparse.signal_power_w.view(np.uint64), dense.signal_power_w.view(np.uint64)
    )


def test_repeated_sparse_calls_are_bit_identical() -> None:
    """稀疏路径重复调用逐位相同（构造期并行不引入不确定性）。"""
    case = build_case(_CASE)
    path = build_path(case)
    power, _ = _sparse_power(case, 9, contiguous=False)
    model = make_model("cgn", case)

    first = model.evaluate_path(path, power)
    for _ in range(3):
        again = model.evaluate_path(path, power)
        assert np.array_equal(first.snr_db.view(np.uint64), again.snr_db.view(np.uint64))
        assert np.array_equal(
            first.spm_noise_w.view(np.uint64), again.spm_noise_w.view(np.uint64)
        )


def test_sparse_call_history_does_not_change_the_result() -> None:
    """调用历史无关的契约在稀疏路径下依然成立。"""
    case = build_case(_CASE)
    path = build_path(case)
    power, _ = _sparse_power(case, 9, contiguous=False)
    other, _ = _sparse_power(case, 4, contiguous=False, seed=3)
    model = make_model("cgn", case)

    first = model.evaluate_path(path, power)
    model.evaluate_path(path, other)
    again = model.evaluate_path(path, power)
    assert np.array_equal(first.snr_db.view(np.uint64), again.snr_db.view(np.uint64))


def test_isrs_profile_of_the_active_subset_matches_the_dense_one() -> None:
    """ISRS 按真实（稀疏）功率谱算：活跃信道上的剖面与全网格路径一致。"""
    case = build_case(_PERF_CASE)
    power, index = _sparse_power(case, 6, contiguous=False, seed=11)
    kernel = make_model("cgn", case).kernel_for(build_path(case).fiber_configs[0])
    canonical, _ = cgn_module._canonical_power(power)

    _, gain_dense, rate_dense = cgn_module._isrs_span_profile(
        kernel.f_rel_hz,
        kernel.alpha_npm,
        canonical,
        kernel.f_m_hz,
        kernel.f_M_hz,
        kernel.k_bar,
        1.0e5,
        kernel.isrs_segments,
    )
    view = kernel._subset_view(index)
    _, gain_sparse, rate_sparse = cgn_module._isrs_span_profile(
        view.f_rel_hz,
        view.alpha_npm,
        canonical[index],
        kernel.f_m_hz,
        kernel.f_M_hz,
        kernel.k_bar,
        1.0e5,
        kernel.isrs_segments,
    )

    assert np.max(np.abs(rate_dense[index] - rate_sparse)) < 1e-18
    scale = np.maximum(np.abs(gain_dense[:, index]), np.finfo(np.float64).tiny)
    assert np.max(np.abs(gain_dense[:, index] - gain_sparse) / scale) < 1e-12


def test_live_piece_compaction_is_bit_identical_when_scattered_back() -> None:
    """活矩形片压缩在"散射回整块数组"模式下逐位不变。

    压缩只把"相关域积分跨度非零"的矩形片送进融合内核；死片的被积函数恒为
    0（内核写 +0.0），故散射回去后整块数组与不压缩时**逐位相同**，归约也
    用同一个 ``np.sum(integrand, axis=(-2,-1))``。
    """
    case = build_case(_CASE)
    path = build_path(case)
    power, _ = _sparse_power(case, 11, contiguous=False, seed=4)
    kernel = make_model("cgn", case).kernel_for(path.fiber_configs[0])
    canonical, _ = cgn_module._canonical_power(power)
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
    common = {
        "f_abs_hz": kernel.f_abs_hz,
        "b_hz": kernel.b_hz,
        "alpha_npm": kernel.alpha_npm,
        "gamma_w_m": kernel.gamma_w_m,
        "beta2_s2_m": kernel.beta2_s2_m,
        "beta3_s3_m": kernel.beta3_s3_m,
        "dispersion_convention": kernel.dispersion_convention,
        "nsp_convention": kernel.nsp_convention,
        "length_m": 1.0e5,
        "r_index": kernel._r_index,
        "correlation_points": kernel.correlation_points,
        "island_cells": kernel.island_cells,
        "n_coherent_spans": 10,
        "isrs_gain": gain,
        "isrs_rate": rate,
        "isrs_z_mid": z_mid,
        "isrs_seg_length_m": 1.0e5 / kernel.isrs_segments,
        "band_points": 1,
    }

    plain, dirichlet = cgn_module._island_core(**common)
    scattered, dirichlet_scattered = cgn_module._island_core(
        **common, compact_live=True, compact_scatter=True
    )
    binned, dirichlet_binned = cgn_module._island_core(
        **common, compact_live=True, compact_scatter=False
    )

    assert np.array_equal(plain.view(np.uint64), scattered.view(np.uint64))
    assert np.array_equal(dirichlet.view(np.uint64), dirichlet_scattered.view(np.uint64))
    # bincount 归约只改求和次序：相对误差仍在 1e-12 量级以内。
    assert _max_relative_error(binned, plain, np.ones_like(plain, dtype=bool)) < 1e-12
    assert _max_relative_error(dirichlet_binned, dirichlet, np.ones_like(dirichlet, dtype=bool)) < 1e-12


def test_all_zero_power_is_handled() -> None:
    """整条链路无功率时不崩溃，噪声为零。"""
    case = build_case(_CASE)
    path = build_path(case)
    power = np.zeros(case.num_channels, dtype=np.float64)

    result = make_model("cgn", case).evaluate_path(path, power)

    assert np.all(result.spm_noise_w == 0.0)
    assert np.all(result.xpm_noise_w == 0.0)
    assert result.snr_db.shape == (case.num_channels,)


def test_sparse_cold_start_scales_with_the_active_channel_count() -> None:
    """稀疏路径的冷启动代价随活跃信道数走，而不是随网格大小走。"""
    case = build_case(_PERF_CASE)
    path = build_path(case)
    # 先把小算例跑一遍，把 numba 的一次性 JIT 排除在计时之外。
    warm_case = build_case("c_57ch")
    make_model("cgn", warm_case).evaluate_path(
        build_path(warm_case), launch_profile(warm_case, None)
    )

    def cold_ms(n_active: int) -> float:
        power, _ = _sparse_power(case, n_active, contiguous=False, seed=5)
        model = make_model("cgn", case)
        started = time.perf_counter()
        model.evaluate_path(path, power)
        return (time.perf_counter() - started) * 1e3

    sparse_cold = cold_ms(3)
    dense_cold = cold_ms(case.num_channels)

    # 3 个活跃信道必须远快于 479 个：稀疏路径的代价随活跃信道数走。
    assert sparse_cold < 0.5 * dense_cold, (
        f"sparse cold start {sparse_cold:.1f} ms is not much cheaper than "
        f"the dense {dense_cold:.1f} ms"
    )
    assert sparse_cold < 500.0, f"sparse cold start took {sparse_cold:.1f} ms"
