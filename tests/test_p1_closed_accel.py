"""WP-MERGE-FG-FLEX **第 4 轮**：P1 闭式核加速档（``matvec`` + 批量入口）的验收守卫。

本文件只测**第 4 轮新增**的性质，不复述两棵树既有的守卫：

1. :class:`TestFrozenKernelUntouched`（判据 V1''）——冻结核 ``numba_kernels.py``
   的字节身份**没变**，P1 的字面量（``b_hz_channels[i]`` / ``b_hz_channels[l_idx]``）
   与 ``asinh``/``atan`` 的参数形式**逐字还在**。
2. :class:`TestMatvecTier`（判据 V2''）——``p1_closed`` 的 ``matvec`` 档：
   与 ``off`` 档在 1e-12 预算内等价；出纤功率与 SPM **逐位相同**；
   XPM **非逐位**（证明它真的走了加速路径，不是静默的 ``off``）。
3. :class:`TestBatchEntry`（判据 V2''）——``p1_closed`` 的批量入口可用，
   ``batch ≡ serial`` 在 1e-12 预算内；**档位不匹配时显式抛错**（不许静默回落）。
4. :class:`TestTierWiring`——档位表与默认档的接线。

⚠️ **相对偏差预算 1e-12 原样沿用，不放宽**（FG-4a/FG-4b 的既有常量）。
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.models.isrs_gn import (
    MATVEC_MIN_ACTIVE_CHANNELS,
    ODE_ACCEL_DEFAULT,
    ODE_ACCEL_MODES,
    MultiBandISRSGN,
)
from network_nlo_eval.models.numba_kernels import _calc_span_noise_and_power_jit
from network_nlo_eval.models.numba_kernels_p1_accel import (
    P1XpmMatrices,
    build_p1_xpm_pair_matrices,
)
from network_nlo_eval.network.elements import FiberSpanConfig

#: FG-4a/FG-4b 的相对预算常量（本文件不另设一个更松的数）。
MAX_REL_DEVIATION = 1e-12

#: 冻结核 ``numba_kernels.py`` 的 sha256（= flexgrid-dev @ cb24704 的字节身份）。
FROZEN_KERNEL_SHA256 = "6611c4164a60d0dcc653e47c5f02d81534e9c78b03e71213aa30d8b13bfd4aa4"

SLOT_WIDTH_HZ = 12.5e9
NUM_SLOTS = 800
BAND_LOW_EDGE_HZ = 191.0e12
SPAN_LENGTH_M = 80e3
SLOTS_PER_CHANNEL = 4

#: 混合 n_slots：1/2/3/4 槽 ⇒ ``B_i`` 恰为 [1.25e10, 2.50e10, 3.75e10, 5.00e10]。
SLOT_CYCLE = (1, 2, 3, 4)


def _grid() -> FlexibleGrid:
    return FlexibleGrid(
        slot_width_hz=SLOT_WIDTH_HZ, num_slots=NUM_SLOTS, band_low_edge_hz=BAND_LOW_EDGE_HZ
    )


def _connections(grid: FlexibleGrid, n_ch: int, mixed: bool = True) -> list[ConnectionSpec]:
    conns: list[ConnectionSpec] = []
    first = 0
    for index in range(n_ch):
        n_slots = SLOT_CYCLE[index % len(SLOT_CYCLE)] if mixed else 1
        conns.append(
            ConnectionSpec(
                grid=grid,
                connection_id=index,
                first_slot=first,
                n_slots=n_slots,
                bit_rate_bps=25e9 * n_slots,
                modulation="QPSK",
                symbol_rate_hz=25e9,
                roll_off=0.0,
                launch_power_w=1e-3,
            )
        )
        first += n_slots + 1
    return conns


def _evaluator(
    grid: FlexibleGrid, conns: list[ConnectionSpec], ode_accel: str
) -> MultiBandISRSGN:
    return MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
        symbol_rate_hz=50e9,
        connections=conns,
        slots_per_channel=SLOTS_PER_CHANNEL,
        ode_accel=ode_accel,
        raman_spectrum="legacy",
        channel_bandwidth_source="occupied",
    )


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.maximum(np.abs(b), 1e-300)
    return float(np.max(np.abs(a - b) / denom))


def _profiles(n_ch: int, n_batch: int) -> np.ndarray:
    """B 个候选的确定性功率剖面（含零功率通道 ⇒ 掩码语义也被覆盖）。"""
    P = np.empty((n_batch, n_ch), dtype=np.float64)
    for b in range(n_batch):
        tilt = 1.0 + 0.08 * (np.arange(n_ch, dtype=np.float64) / max(n_ch - 1, 1) - 0.5)
        P[b] = 1e-3 * (0.85 + 0.05 * (b % 7)) * tilt
        for k in range(3):
            P[b, (b * 13 + k * 29) % n_ch] = 0.0
    return P


# ---------------------------------------------------------------------------
# V1''：公式不动
# ---------------------------------------------------------------------------
class TestFrozenKernelUntouched:
    """判据 V1''：P1 的公式与冻结核的字节身份一个字没动。"""

    def test_frozen_kernel_bytes_are_unchanged(self) -> None:
        """``numba_kernels.py`` 必须**逐字节**等于 flexgrid-dev @ cb24704。"""
        import network_nlo_eval.models.numba_kernels as K

        sha = hashlib.sha256(Path(K.__file__).read_bytes()).hexdigest()
        assert sha == FROZEN_KERNEL_SHA256, (
            f"冻结核 numba_kernels.py 的 sha256 变了：{sha} != {FROZEN_KERNEL_SHA256}。"
            "第 4 轮的加速全部在 numba_kernels_p1_accel.py 里，P1 的算式不许动。"
        )

    def test_p1_literals_are_still_there(self) -> None:
        """``b_hz_channels[i]`` / ``b_hz_channels[l_idx]`` 与 asinh/atan 参数形式逐字还在。"""
        import network_nlo_eval.models.numba_kernels as K

        text = Path(K.__file__).read_text(encoding="utf-8")
        for literal in (
            "b_hz_channels[i]",
            "b_hz_channels[l_idx]",
            "np.arcsinh((phi_i * b_i_hz**2) / (PI * alpha_i_npm))",
            "np.arcsinh((phi_i * b_i_hz**2) / (2 * PI * alpha_i_npm))",
            "np.arctan((phi_il * b_i_hz) / alpha_l_npm)",
            "np.arctan((phi_il * b_i_hz) / (2 * alpha_l_npm))",
            "sigma_xpm_2[i] = (32.0 / 27.0) * p_in_w[i] * n_span * xpm_sum_term",
        ):
            assert literal in text, f"冻结核里少了 P1 的字面量 {literal!r}"

    def test_pair_matrices_are_nonnegative(self) -> None:
        """``G1``/``G2`` 必须逐元素 >= 0（这是「逐对 max(·,0) 是恒等」的符号前提）。

        见 ``models/numba_kernels_p1_accel.py`` 的模块 docstring 里的符号论证。
        """
        grid = _grid()
        conns = _connections(grid, 8)
        ev = _evaluator(grid, conns, "matvec")
        g1, g2, _phi, _gamma = build_p1_xpm_pair_matrices(
            ev.n_nodes,
            ev.channel_bandwidth_hz_channels,
            ev.f_rel_hz,
            ev.lambdas_m,
            ev.a_eff_m2_channels,
            ev.alpha_power_npm_channels,
            ev.beta2_s2_m,
            ev.beta3_s3_m,
            ev.n2_m2_w,
        )
        assert float(g1.min()) >= 0.0, f"G1 必须逐元素 >= 0，实测 min={g1.min()!r}"
        assert float(g2.min()) >= 0.0, f"G2 必须逐元素 >= 0，实测 min={g2.min()!r}"
        # 对角线必须为零（串行基线那里是 continue：跳过自配对）。
        assert np.all(np.diag(g1) == 0.0)
        assert np.all(np.diag(g2) == 0.0)


# ---------------------------------------------------------------------------
# V2''：加速正确性
# ---------------------------------------------------------------------------
class TestMatvecTier:
    """判据 V2''：``matvec ≡ off``（1e-12 预算，不放宽）。"""

    @pytest.mark.parametrize("mixed", [True, False], ids=["mixed_n_slots", "uniform"])
    def test_matvec_equals_off_within_budget(self, mixed: bool) -> None:
        grid = _grid()
        conns = _connections(grid, 16, mixed=mixed)
        ev_off = _evaluator(grid, conns, "off")
        ev_mv = _evaluator(grid, conns, "matvec")
        power = np.full(len(conns), 1e-3, dtype=np.float64)

        a = ev_off._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        b = ev_mv._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)

        for name, x, y in zip(("p_out", "spm", "xpm", "ase"), a, b, strict=True):
            rel = _max_rel(np.asarray(y), np.asarray(x))
            assert rel <= MAX_REL_DEVIATION, f"{name} 相对偏差 {rel:.3e} 超出 1e-12 预算"

    def test_out_power_and_spm_are_bit_identical_to_off(self) -> None:
        """出纤功率与 SPM 走**冻结核的同一个函数** ⇒ 必须逐位相同。

        这条同时是「P1 的算式没被动过」的可执行形式：只有 XPM 的求和次序允许变。
        """
        grid = _grid()
        # 第 6 轮：``matvec`` 有自适应门（活跃信道数 ≥ MATVEC_MIN_ACTIVE_CHANNELS）。
        # 这条测的是「matvec 真的走了加速路径」，所以谱必须够密，否则会被门落回 off。
        n_ch = MATVEC_MIN_ACTIVE_CHANNELS + 8
        conns = _connections(grid, n_ch)
        assert len(conns) >= MATVEC_MIN_ACTIVE_CHANNELS
        ev_off = _evaluator(grid, conns, "off")
        ev_mv = _evaluator(grid, conns, "matvec")
        power = np.full(len(conns), 1e-3, dtype=np.float64)

        a = ev_off._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        b = ev_mv._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)

        assert np.array_equal(a[0], b[0]), "出纤功率必须与 off 逐位相同"
        assert np.array_equal(a[1], b[1]), "SPM 必须与 off 逐位相同"
        assert not np.array_equal(a[2], b[2]), (
            "XPM 必须与 off **非**逐位相同；若逐位相同，说明 matvec 退化成了 off"
        )

    def test_off_tier_is_unchanged_by_the_new_module(self) -> None:
        """``off`` 档必须**仍然**逐字节走冻结核（它是 P1 的基准）。

        ⚠️ **只比前三个返回值**：冻结核的第 4 个返回值（ASE）在 P1 路径里是
        **死代码**——``_calc_span_noise_and_power_p1`` 丢弃它，P1 (34) 的 ASE 在
        Python 侧另算（锚点 ``p1_eq34_ase_dead_kernel``）。这是**既有**事实，
        与第 4 轮无关，所以这条断言刻意不比 ASE。
        """
        grid = _grid()
        conns = _connections(grid, 8)
        ev = _evaluator(grid, conns, "off")
        power = np.full(len(conns), 1e-3, dtype=np.float64)

        frozen = _calc_span_noise_and_power_jit(
            power_in_w=power,
            span_length_m=SPAN_LENGTH_M,
            f_rel_hz=ev.f_rel_hz,
            f_abs_hz=ev.f_abs_hz,
            lambdas_m=ev.lambdas_m,
            alpha_power_npm=ev.alpha_power_npm_channels,
            a_eff_m2=ev.a_eff_m2_channels,
            beta2_s2_m=ev.beta2_s2_m,
            beta3_s3_m=ev.beta3_s3_m,
            n2_m2_w=ev.n2_m2_w,
            nf_lin_channels=ev.nf_lin_channels,
            symbol_rate_hz=ev.symbol_rate_hz,
            b_hz_channels=ev.channel_bandwidth_hz_channels,
            n_effective_spans=1,
            cr_w_m_hz=ev.cr_w_m_hz,
            k_bar=ev.k_bar,
            delta_f_co_hz=ev.delta_f_co_hz,
        )
        via_evaluator = ev._calc_span_noise_and_power(power, SPAN_LENGTH_M, None, None, 1)
        for name, x, y in zip(
            ("p_out", "spm", "xpm"), frozen[:3], via_evaluator[:3], strict=True
        ):
            assert np.array_equal(np.asarray(x), np.asarray(y)), f"{name} 与冻结核不逐位相同"
        # 死代码那一位：edfa=None ⇒ P1 路径的 ASE 恒为 0（不是核里那个值）。
        assert np.array_equal(via_evaluator[3], np.zeros(len(conns)))

    def test_pair_matrices_are_cached_per_evaluator(self) -> None:
        """(N,N) 对矩阵是 evaluator 级常量：同一个 evaluator 上必须**复用**同一个对象。"""
        grid = _grid()
        conns = _connections(grid, 8)
        ev = _evaluator(grid, conns, "matvec")
        first = ev._p1_xpm_matrices_for()
        second = ev._p1_xpm_matrices_for()
        assert first is second, "对矩阵必须复用同一个对象（否则每次调用都重算 O(N^2)）"
        assert isinstance(first, P1XpmMatrices)


# ---------------------------------------------------------------------------
# V2''：批量入口
# ---------------------------------------------------------------------------
class TestBatchEntry:
    """判据 V2''：``batch ≡ serial``（1e-12 预算）+ **不许静默回落**。"""

    @pytest.mark.parametrize("mixed", [True, False], ids=["mixed_n_slots", "uniform"])
    @pytest.mark.parametrize("width", [8, 16, 32])
    def test_batch_equals_serial_within_budget(self, width: int, mixed: bool) -> None:
        grid = _grid()
        conns = _connections(grid, 16, mixed=mixed)
        ev = _evaluator(grid, conns, "matvec")
        P = _profiles(len(conns), width)

        batch = ev.calc_span_noise_and_power_batch(P, SPAN_LENGTH_M, None, None, 1)
        serial = tuple(
            np.stack(
                [
                    ev._calc_span_noise_and_power(P[b], SPAN_LENGTH_M, None, None, 1)[k]
                    for b in range(width)
                ]
            )
            for k in range(4)
        )

        for name, x, y in zip(
            ("p_out", "spm", "xpm", "ase"), serial, batch, strict=True
        ):
            rel = _max_rel(np.asarray(y), np.asarray(x))
            assert rel <= MAX_REL_DEVIATION, (
                f"B={width} {name} 的 batch≡serial 相对偏差 {rel:.3e} 超出 1e-12 预算"
            )

    def test_batch_rejects_wrong_tier_instead_of_silently_falling_back(self) -> None:
        """档位不匹配必须**显式抛错**——静默退回串行是最坏的一种错。

        调用方会以为自己拿到了批量加速，实际跑的是逐候选串行：数字对、账对不上、
        没人知道加速没生效。
        """
        grid = _grid()
        conns = _connections(grid, 8)
        P = _profiles(len(conns), 4)
        for accel in ("off", "cached"):
            ev = _evaluator(grid, conns, accel)
            with pytest.raises(ValueError, match="matvec"):
                ev.calc_span_noise_and_power_batch(P, SPAN_LENGTH_M, None, None, 1)

    def test_batch_shape_is_validated(self) -> None:
        grid = _grid()
        conns = _connections(grid, 8)
        ev = _evaluator(grid, conns, "matvec")
        with pytest.raises(ValueError, match=r"shape \(B, 8\)"):
            ev.calc_span_noise_and_power_batch(
                np.ones((4, 7)), SPAN_LENGTH_M, None, None, 1
            )

    def test_batch_handles_zero_power_channels(self) -> None:
        """零功率通道（惰性连接）在批量路径里必须与串行**同样**被掩码掉。"""
        grid = _grid()
        conns = _connections(grid, 8)
        ev = _evaluator(grid, conns, "matvec")
        P = _profiles(len(conns), 4)
        batch = ev.calc_span_noise_and_power_batch(P, SPAN_LENGTH_M, None, None, 1)
        assert np.all(batch[0] >= 0.0)
        assert np.all(np.isfinite(batch[1]))
        assert np.all(np.isfinite(batch[2]))


# ---------------------------------------------------------------------------
# 档位接线
# ---------------------------------------------------------------------------
class TestTierWiring:
    """闭式 ECZ 的档位表与默认档接线。

    ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本类原先覆盖两套物理的档位表与默认档，
    判据走按档分派表（``*_BY_NLI_MODEL``）。那套物理整体移除后**只剩一套**，
    按档查表已无消费者；**保留下来的判据改钉在全局档位元组上，命题不变**，
    只删掉只测被移除那套物理的那几条。
    """

    def test_p1_closed_supports_matvec_but_not_cached_g(self) -> None:
        """ECZ 认的三档，``cached_g`` **不许**回来。"""
        modes = ODE_ACCEL_MODES
        assert "matvec" in modes, "闭式 ECZ 必须支持 matvec（第 4 轮新增）"
        assert "off" in modes and "cached" in modes
        assert "cached_g" not in modes, (
            "cached_g 是拉曼 G 矩阵缓存；ECZ 闭式核走解析出纤功率，没有 G 矩阵可缓存"
        )

    def test_p1_closed_default_is_the_measured_faster_tier(self) -> None:
        """默认加速档是 ``"matvec"``——第 4 轮按实测定档，**本轮不许改**。

        读数依据见 ``docs/wp_merge_fg_into_flex.md`` 的「第 4 轮」章节。
        想要与 ``cb24704`` **逐位相同**的读数，显式配 ``ode_accel="off"``。
        """
        assert ODE_ACCEL_DEFAULT == "matvec"
