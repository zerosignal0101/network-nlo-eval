"""FG-U1 宽波段拉曼谱（`tabular` 档）的自检。

这一组测试盯的是**本刀自己的契约**，不去动 FG-4a 既有的任何期望值：

* 谱表与闭式一致、插值误差在**声明容差**内（容差由本文件钉住，不许默默放宽）；
* njit 插值器与 numpy 参考实现等价（热路径虽不调它，但 G 矩阵全靠它）；
* 两档在 13.2 THz 锚点**逐位相等**——这是"只差谱形、不差幅度"的根据；
* `legacy` 档与被冻结的折线**逐位相等**（本刀没有偷偷动 legacy）；
* `tabular` 档在 `>15.6 THz` 处非零（legacy 在那里恒零）。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models import numba_kernels as nk
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import FiberSpanConfig
from network_nlo_eval.physics import raman_spectrum as rs

CR = 0.028 / 1e3 / 1e12

#: 插值容差：相对峰值（= 1.0）的绝对偏差上限。线性插值在 0.05 THz 步长、
#: 13 THz 附近最陡处的二阶截断误差约 step^2 * |S''| / 8，声明 1e-3 留足余量。
DECLARED_INTERP_TOLERANCE = 1e-3


def _grid(n: int, spacing_hz: float = 50e9, center_hz: float = 190.2127e12) -> np.ndarray:
    return SpectrumGrid(
        num_channels=n, channel_spacing_hz=spacing_hz, center_frequency_hz=center_hz
    ).frequencies


def _frozen_legacy_g(df_thz: np.ndarray, cr: float = CR) -> np.ndarray:
    """FG-2 冻结的合成折线**逐条重写**一遍，作为 legacy 档的独立对照。

    注意跌落区用的是 `cr * 13.2e12 * (15.6-df) / 2.4`——**不是** `cr*df*shape`，
    两者在 13.2 < df < 15.6 上并不相等。
    """
    df = np.asarray(df_thz, dtype=np.float64)
    return np.where(
        df <= 13.2,
        cr * (df * 1e12),
        np.where(df < 15.6, (cr * 13.2e12) * (15.6 - df) / (15.6 - 13.2), 0.0),
    )


# --- 谱表与插值自检 --------------------------------------------------------


def test_table_rows_and_grid_are_uniform() -> None:
    """提交的谱表必须是均匀栅格且覆盖 0 ~ TABLE_MAX_THz（njit 插值依赖这两条）。"""
    df_grid, shape_grid = rs.load_raman_gain_table()
    assert len(df_grid) == len(shape_grid)
    assert df_grid[0] == 0.0
    assert df_grid[-1] == pytest.approx(rs.TABLE_MAX_THZ)
    steps = np.diff(df_grid)
    assert np.allclose(steps, rs.TABLE_STEP_THZ, rtol=0, atol=1e-12)
    # 峰前单调上升、峰后单调下降（真实石英谱是**宽峰**，不是 legacy 的线性爬升）。
    peak = int(np.argmax(shape_grid))
    assert np.all(np.diff(shape_grid[:peak]) > 0)
    assert np.all(np.diff(shape_grid[peak:]) < 0)


def test_committed_table_matches_closed_form() -> None:
    """提交在 references/ 里的 CSV 必须与闭式谱一致——谱表不是手工填的数。"""
    df_grid, shape_grid = rs.load_raman_gain_table()
    exact = np.asarray(rs.raman_gain_shape(df_grid), dtype=np.float64)
    assert np.allclose(shape_grid, exact, rtol=0, atol=1e-15)


def test_interpolation_error_within_declared_tolerance() -> None:
    """FU1-3：插值误差 <= 声明容差（自己定、自己证）。"""
    df_grid, shape_grid = rs.load_raman_gain_table()
    dense = np.linspace(0.0, rs.TABLE_MAX_THZ, 20_001)
    got = np.array([rs.interp_raman_gain_shape(v, df_grid, shape_grid) for v in dense])
    want = np.asarray(rs.raman_gain_shape(dense), dtype=np.float64)
    assert np.max(np.abs(got - want)) <= DECLARED_INTERP_TOLERANCE


def test_kernel_interpolator_matches_numpy() -> None:
    """Njit 插值器必须与 numpy 参考实现等价（G 矩阵全靠它）。"""
    df_grid, shape_grid = rs.load_raman_gain_table()
    probe = np.linspace(-1.0, rs.TABLE_MAX_THZ + 5.0, 5_000)
    for value in probe:
        kernel = nk._interp_uniform_shape(
            float(value), float(df_grid[0]), rs.TABLE_STEP_THZ, len(shape_grid), shape_grid
        )
        reference = float(rs.interp_raman_gain_shape(value, df_grid, shape_grid))
        assert kernel == pytest.approx(reference, rel=0, abs=1e-15)


def test_spectrum_peak_is_near_13_thz() -> None:
    """闭式谱的峰位应贴近文献常引的 13.2 THz。"""
    fine = np.linspace(0.0, 20.0, 200_001)
    peak = float(fine[np.argmax(rs.raman_gain_shape(fine))])
    assert 12.5 <= peak <= 13.6
    assert peak == pytest.approx(13.2, abs=0.3)


def test_out_of_range_is_truncated_not_extrapolated() -> None:
    """超出表范围必须**显式截断为 0**，不得外推。"""
    df_grid, shape_grid = rs.load_raman_gain_table()
    assert float(rs.interp_raman_gain_shape(rs.TABLE_MAX_THZ + 1e-6, df_grid, shape_grid)) == 0.0
    assert float(rs.interp_raman_gain_shape(-1.0, df_grid, shape_grid)) == 0.0


# --- 两档的形状对照 --------------------------------------------------------


def test_anchor_is_bit_identical_between_tiers() -> None:
    """两档在 13.2 THz 处必须**逐位相等**——否则比较会被幅度混淆。"""
    f_hz = np.ascontiguousarray(np.array([190e12, 190e12 + 13.2e12]), dtype=np.float64)
    legacy = nk._build_g_matrix(f_hz, CR)
    tabular = nk._build_g_matrix_tabular(f_hz, CR, *rs.load_raman_gain_table(), rs.TABLE_STEP_THZ)
    assert legacy[0, 1] == tabular[0, 1]


def test_legacy_builder_matches_frozen_polyline() -> None:
    """Legacy 档必须与 FG-2 冻结折线逐位相等（本刀没有动 legacy）。"""
    f_hz = np.ascontiguousarray(_grid(400), dtype=np.float64)
    g = nk._build_g_matrix(f_hz, CR)
    df = np.abs(f_hz[:, None] - f_hz[None, :]) / 1e12
    assert np.array_equal(g, _frozen_legacy_g(df))


def test_tabular_is_nonzero_beyond_legacy_cutoff() -> None:
    """FU1 的核心物理主张：legacy 在 >15.6 THz 恒零，真实谱不是。"""
    f_hz = np.ascontiguousarray(np.array([190e12, 190e12 + 20e12, 190e12 + 27e12]), dtype=np.float64)
    legacy = nk._build_g_matrix(f_hz, CR)
    tabular = nk._build_g_matrix_tabular(f_hz, CR, *rs.load_raman_gain_table(), rs.TABLE_STEP_THZ)
    assert legacy[0, 1] == 0.0 and legacy[0, 2] == 0.0
    assert tabular[0, 1] > 0.0 and tabular[0, 2] > 0.0
    # 相对峰值 G(13.2 THz) = cr*13.2e12 归一：27 THz 处只剩 ~10%。
    peak_g = CR * 13.2e12
    assert tabular[0, 2] / peak_g == pytest.approx(0.10, abs=0.02)
    assert tabular[0, 1] / peak_g == pytest.approx(0.33, abs=0.02)


def test_pair_matrices_agree_with_full_matrix() -> None:
    """Tabular 矩阵对必须与完整 G 自洽（gain = G，loss = 频率比 * G）。"""
    f_hz = np.ascontiguousarray(_grid(120), dtype=np.float64)
    df_grid, shape_grid = rs.load_raman_gain_table()
    g = nk._build_g_matrix_tabular(f_hz, CR, df_grid, shape_grid, rs.TABLE_STEP_THZ)
    g_gain, g_loss = nk._build_g_pair_matrices_tabular(f_hz, CR, df_grid, shape_grid, rs.TABLE_STEP_THZ)
    assert np.array_equal(g_gain, np.triu(g, k=1))
    expected_loss = np.tril(g, k=-1) * (f_hz[None, :] / f_hz[:, None])
    assert np.allclose(g_loss, expected_loss, rtol=1e-15, atol=0)


def test_cache_survives_pair_then_full_request_order() -> None:
    """先取矩阵对、再取完整 G 不得返回 None（FG-U1 修掉的缓存槽位缺陷）。"""
    f_hz = np.ascontiguousarray(_grid(60), dtype=np.float64)
    gain, loss = nk.raman_g_pair_matrices_tabular(f_hz, CR)
    assert gain is not None and loss is not None
    full = nk.raman_g_matrix_tabular(f_hz, CR)
    assert full is not None and np.allclose(full, gain + loss / (f_hz[None, :] / f_hz[:, None]))


def test_tier_cache_entries_do_not_collide() -> None:
    """Legacy 与 tabular 必须各占各的缓存条目，不能互相覆盖。"""
    f_hz = np.ascontiguousarray(_grid(60), dtype=np.float64)
    legacy = nk.raman_g_matrix(f_hz, CR)
    tabular = nk.raman_g_matrix_tabular(f_hz, CR)
    assert not np.allclose(legacy, tabular)


# --- 档位接线 --------------------------------------------------------------


def _evaluator(raman_spectrum: str, ode_accel: str = "matvec") -> MultiBandISRSGN:
    return MultiBandISRSGN(
        grid=SpectrumGrid(num_channels=80, channel_spacing_hz=50e9, center_frequency_hz=190.2127e12),
        ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        symbol_rate_hz=50e9,
        raman_spectrum=raman_spectrum,
        ode_accel=ode_accel,
    )


def test_default_tier_is_tabular() -> None:
    """默认档现在是 tabular（FG-15 换底），且 legacy 仍逐位可达。

    **为什么这条期望值必须移动（逐条归因，不是只改数字）**：这条测试的命题
    本身就是"**默认档是什么**"，所以换底必然让它移动——这是 FG-15 唯一
    授权移动的既有期望值之一。机制是物理的：legacy 的两个截断（13.2 / 15.6 THz）
    使相隔 >15.6 THz 的耦合被迫为零，而 S+C+L+U 跨约 27 THz。

    下面两行仍在同一档上核对 G 矩阵，确保换底没有顺手改坏 legacy 的数值：
    ``_evaluator("legacy")`` 是**显式**取 legacy，不是靠默认。
    """
    assert MultiBandISRSGN.__init__.__defaults__[-1] == "tabular"
    ev = _evaluator("legacy")
    assert ev.raman_spectrum == "legacy"
    assert np.allclose(ev._g_gain, nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)[0])


def test_tabular_changes_the_numbers() -> None:
    """Tabular 档必须真的改变结果（否则这一刀什么都没做）。"""
    legacy = _evaluator("legacy")
    tabular = _evaluator("tabular")
    assert not np.allclose(legacy._g_gain, tabular._g_gain)
    assert not np.allclose(legacy._g_loss, tabular._g_loss)


def test_tabular_rejects_frozen_off_path() -> None:
    """`off` 是冻结的 legacy 路径，与 tabular 的组合必须显式报错而不是静默退化。"""
    with pytest.raises(ValueError, match="cached_g"):
        _evaluator("tabular", ode_accel="off")


def test_unknown_tier_rejected() -> None:
    """未知档位名必须被拒绝（而不是悄悄落到某个默认分支）。"""
    with pytest.raises(ValueError, match="raman_spectrum"):
        _evaluator("no-such-tier")


# --- 不对称近似的量化（FU1-5）----------------------------------------------


def test_bose_approximation_is_quantified() -> None:
    """对称近似（legacy 的 abs(df)）在该谱上的量级必须可测且单调。"""
    freqs = np.array([1.0, 5.0, 11.45, 13.2, 20.0, 27.0])
    stokes = np.array([rs.asymmetry_factor(f, "stokes", 300.0) for f in freqs])
    anti = np.array([rs.asymmetry_factor(f, "anti_stokes", 300.0) for f in freqs])
    assert np.all(stokes > 1.0) and np.all(anti < 1.0)
    # n_B 随频差单调下降，因此对称近似 (n_B + 1/2) 在大频差处：
    #   Stokes 侧被低估（因子趋近 2），anti-Stokes 侧被高估（因子趋近 0）。
    assert np.all(np.diff(stokes) > 0)
    assert np.all(np.diff(anti) < 0)
    assert stokes[-1] > 1.9 and anti[-1] < 0.1
    with pytest.raises(ValueError, match="anti_stokes"):
        rs.asymmetry_factor(1.0, "nonsense", 300.0)
