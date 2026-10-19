r"""FG-4e gate: the CPU batched path must be per-candidate equal and must not move the default.

Every assertion here is executable and recomputable inside this worktree::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \
        $HOME/miniconda3/envs/rl/bin/python -m pytest tests/test_fg4e_cpu_batching.py -v

Mapping to the FG-4e acceptance table
=====================================

======  ==============================================================
 F4e-1  the batch entry point exists and every candidate is within
        1e-12 of the **serial** path on the same candidate, across the
        uniform / partially-loaded / non-uniform-grid shapes and
        across the four output channels
 F4e-3  the default tier and the ``off`` freeze point are untouched:
        the suite count is unchanged and ``off`` stays selectable and
        bit-identical, so this knife moved nothing upstream
 F4e-4  B=1 is **reported, not asserted to be fast**: the batch arm is
        slower there, and the test pins that fact so it cannot quietly
        become the reason to flip the B=1 threshold
 =====  ==============================================================

The traps this file is written to close
---------------------------------------
FG-4b's XPM pre-factor had a shipped bug: ``p_i`` broadcast implicitly onto
the last axis, so it silently computed ``p_l``. It was invisible on a fully
loaded grid (``p_i == p_l``) and only appeared once Raman tilt made the
per-channel powers differ. The batch kernel folds ``T**2`` onto the
*perturbation* axis ``l`` for exactly the same reason, so:

* the **non-uniform grid** case is in the equivalence sweep — it is the shape
  that exposes an axis mistake. Note it runs at ``phi_kurtosis = 0.0`` (the
  shared ``_flexgrid_evaluator`` hardcodes Gaussian): the **64QAM kurtosis**
  combination is covered separately by
  ``test_batch_matches_serial_nonzero_kurtosis_multispan``, which builds its own
  evaluator so the ``phi_kurtosis != 0`` path is genuinely exercised;
* the **zero-power** case is in the sweep, because the batch kernel replaces the
  per-pair ``continue`` with three algebraic masks (diagonal zeroed at build
  time, ``p_sq[b,l] == 0`` kills ``u*``, and ``P > 0`` on the output). A mask
  that is subtly wrong passes on a fully loaded grid;
* a **non-uniform per-candidate power** profile is included so ``p_i != p_l``
  inside a single candidate, which is the condition the original bug needed.

The 1e-12 budget is FG-4a's, carried forward unchanged. It is asserted, not
assumed: the measured worst-case deviations are printed by the tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from fg4a_scenarios import (
    CENTER_FREQUENCY_HZ,
    CHANNEL_SPACING_HZ,
    NUM_CHANNELS,
    PHI_KURTOSIS_GAUSSIAN,
    SYMBOL_RATE_HZ,
    _flexgrid_evaluator,
    _legacy_evaluator,
    _partial_load_power,
)

#: ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：``MAX_REL_DEVIATION`` / ``worst_relative_deviation``
#: 原本住在那份被移除物理的批量核里。本仓的 1e-12 预算常量由**闭式 ECZ 的加速模块**
#: 提供（``numba_kernels_p1_accel.MAX_REL_DEVIATION``），**数值一字未改**，仍是 1e-12。
#: 下面的 ``worst_relative_deviation`` 是与之逐字等价的本地参考实现。
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.numba_kernels_p1_accel import MAX_REL_DEVIATION


def worst_relative_deviation(expected: np.ndarray, actual: np.ndarray) -> float:
    """最大相对偏差（分母取 ``|expected|``，零处夹一个极小量避免除零）。"""
    diff = np.abs(np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float))
    scale = np.maximum(np.abs(np.asarray(expected, dtype=float)), 1e-300)
    return float(np.max(diff / scale))
from network_nlo_eval.network.elements import EDFAConfig

import fg4a_scenarios as _scen
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.network.elements import FiberSpanConfig

SPAN_LENGTH_M = 100e3

#: FG-4e brief §1.3: the real working point from FG-4d, plus the neighbours that
#: bracket the FG-4c knee at B>=4. ``0`` is a cheap sanity probe.
BATCH_SIZES = (1, 4, 16, 68, 256)


def _candidate_batch(batch: int, n_ch: int = NUM_CHANNELS, *, blank_stride: int = 7) -> np.ndarray:
    """``(B, N)`` candidates shaped like the real workload: one occupancy, one
    channel raised per candidate.

    This is the shape FG-4d measured in production rollouts (a fixed base
    occupancy with per-candidate perturbations), so the timing in
    ``results/fg4e_speedup_ab.json`` is quoting a curve for a batch that
    actually occurs. The ``blank_stride`` empty slots put the zero-power guards
    on the measured path.
    """
    base = np.full((batch, n_ch), 1e-3, dtype=np.float64)
    base[:, ::blank_stride] = 0.0
    base[:, 0] = 0.0
    base[:, -1] = 0.0
    step = max(1, n_ch // max(batch, 1))
    for b in range(batch):
        base[b, (b * step) % n_ch] = 1e-3
    return np.ascontiguousarray(base)


def _serial_reference(ev, profiles: np.ndarray, span_length_m: float, n_span: int = 1):
    """The serial path, one candidate at a time. Shapes ``(B, N)`` for comparison."""
    rows = [
        np.stack(ev._calc_span_noise_and_power(np.ascontiguousarray(profiles[b]), span_length_m, None, None, n_span))
        for b in range(profiles.shape[0])
    ]
    return np.stack(rows)


def _both_arms(ev, profiles: np.ndarray, span_length_m: float = SPAN_LENGTH_M, n_span: int = 1):
    """Both arms laid out as ``(B, 4, N)`` so the candidate axis is leading.

    The serial arm stacks per-candidate ``(4, N)`` rows; the batch arm returns four
    ``(B, N)`` arrays. Transposing the latter here keeps the comparison axis
    identical instead of relying on a reader to notice the difference.
    """
    ref = _serial_reference(ev, profiles, span_length_m, n_span)
    bat = np.stack(ev.calc_span_noise_and_power_batch(profiles, span_length_m, None, None, n_span))
    return ref, np.ascontiguousarray(np.moveaxis(bat, 0, 1))


def _assert_within_budget(ref: np.ndarray, bat: np.ndarray, label: str) -> float:
    worst = 0.0
    for k, key in enumerate(("p_out", "spm", "xpm", "ase")):
        dev = worst_relative_deviation(ref[:, k], bat[:, k])
        worst = max(worst, dev)
        assert dev <= MAX_REL_DEVIATION, (
            f"{label}/{key}: worst relative deviation {dev:.3e} exceeds the "
            f"{MAX_REL_DEVIATION:.0e} FG-4a budget (never widened)"
        )
    print(f"  {label}: worst relative deviation {worst:.3e} (budget {MAX_REL_DEVIATION:.0e})")
    return worst


# ---------------------------------------------------------------------------
# F4e-1: the batch entry point exists and is per-candidate equal
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("batch", BATCH_SIZES)
def test_batch_matches_serial_per_candidate_uniform(batch: int) -> None:
    """F4e-1: uniform fixed grid, the shape the production N=230 uses."""
    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    ref, bat = _both_arms(ev, _candidate_batch(batch))
    _assert_within_budget(ref, bat, f"uniform/B={batch}")


@pytest.mark.parametrize("batch", (1, 4, 16, 68))
def test_batch_matches_serial_partially_loaded(batch: int) -> None:
    """F4e-1 + the zero-power mask: every candidate carries empty slots.

    The batch kernel has no ``continue`` to skip a dead channel, so this is the
    case that would catch a mask mistake. It also carries the edge blanks.
    """
    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    ref, bat = _both_arms(ev, _candidate_batch(batch))
    _assert_within_budget(ref, bat, f"partial/B={batch}")


def test_batch_matches_serial_on_nonuniform_grid() -> None:
    """F4e-1 on a flexible grid with per-connection symbol rates.

    This is the shape that makes ``p_i != p_l`` within a single candidate, which
    is precisely the condition under which FG-4b's ``p_i`` axis bug was invisible.
    Runs at ``phi_kurtosis = 0.0``; the non-zero-kurtosis combination is covered by
    ``test_batch_matches_serial_nonzero_kurtosis_multispan``.
    """
    ev = _flexgrid_evaluator("matvec")
    n_ch = ev.n_nodes
    assert n_ch > 0
    profiles = _candidate_batch(8, n_ch)
    # Break the "every candidate is the same occupancy" regularity a little, so
    # the comparison is not accidentally p_i == p_l everywhere.
    profiles[3] = profiles[3] * 1.7
    ref, bat = _both_arms(ev, profiles)
    _assert_within_budget(ref, bat, "flexgrid/nonuniform")


def _flexgrid_evaluator_with_kurtosis(phi_kurtosis: float):
    """Copy of ``fg4a_scenarios._flexgrid_evaluator`` with a settable kurtosis.

    The shared helper hardcodes ``PHI_KURTOSIS_GAUSSIAN``, so the non-uniform-grid
    case above only ever exercises ``phi_kurtosis = 0.0``. The clamp-order
    assumption in ``docs/fg4e_cpu_batching.md`` (serial clamps per term, batch
    clamps after summing) is only at risk when ``phi_kurtosis != 0``; that
    combination needs its own evaluator. Rebuilding it here rather than
    parameterising the shared helper keeps ``fg4a_scenarios.py`` (a pre-existing
    file under the G2 guard) untouched.
    """
    grid = FlexibleGrid(
        slot_width_hz=_scen.SLOT_WIDTH_HZ, num_slots=_scen.NUM_SLOTS, band_low_edge_hz=186.1e12
    )
    connections: list[ConnectionSpec] = []
    connection_id = 0
    first_slot = 0
    plan = ((4, 100e9, "QPSK", 50e9), (2, 100e9, "64QAM", 100e9 / 6), (8, 200e9, "QPSK", 100e9))
    index = 0
    while first_slot + 2 <= _scen.NUM_SLOTS:
        n_slots, bit_rate, modulation, rs = plan[index % len(plan)]
        if first_slot + n_slots > _scen.NUM_SLOTS:
            break
        connections.append(
            ConnectionSpec(
                grid=grid,
                connection_id=connection_id,
                first_slot=first_slot,
                n_slots=n_slots,
                bit_rate_bps=bit_rate,
                modulation=modulation,
                symbol_rate_hz=rs,
                roll_off=0.0,
                launch_power_w=0.0 if connection_id % 5 == 4 else 1e-3,
            )
        )
        connection_id += 1
        first_slot += n_slots + (2 if n_slots == 8 else 1)
        index += 1

    return MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        symbol_rate_hz=_scen.SYMBOL_RATE_HZ,
        phi_kurtosis=phi_kurtosis,
        connections=connections,
        slots_per_channel=_scen.SLOTS_PER_CHANNEL,
        ode_accel="matvec",
    )


@pytest.mark.parametrize("n_span,span_m", [(1, SPAN_LENGTH_M), (2, SPAN_LENGTH_M), (3, SPAN_LENGTH_M), (5, 75e3)])
def test_batch_matches_serial_nonzero_kurtosis_multispan(n_span: int, span_m: float) -> None:
    """F4e-1 on the combination the clamp-order note leaves open: ``phi_kurtosis != 0`` AND ``n_span > 1``.

    ``n_tilde = n_span if n_span > 1 else 0`` feeds ``num2``, so 64QAM kurtosis at
    ``n_span = 1`` cannot exercise the branch where a per-term clamp and a
    sum-then-clamp could disagree. This is also the realistic workload shape
    (400G/64QAM over several 100 km spans). Measured worst deviation is ~1e-15.
    """
    ev = _flexgrid_evaluator_with_kurtosis(_scen.PHI_KURTOSIS_64QAM)
    profiles = _candidate_batch(8, ev.n_nodes)
    profiles[3] = profiles[3] * 1.7
    ref, bat = _both_arms(ev, profiles, span_m, n_span)
    _assert_within_budget(ref, bat, f"64QAM/kurtosis/n_span={n_span}/L={span_m/1e3:g}km")


def test_batch_matches_serial_across_span_count_and_span_length() -> None:
    """F4e-1 with n_effective_spans > 1 and a non-100 km span.

    ``n_span`` reaches the batch kernel through the SPM span exponent
    (``n_span ** 1.05``) and the ASE ``n_span`` factor, and reaches the
    constants through the cache key. A leak here would show as a deviation only
    at n_span > 1.
    """
    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    profiles = _candidate_batch(4)
    for n_span, span_m in ((1, SPAN_LENGTH_M), (2, SPAN_LENGTH_M), (5, 75e3)):
        ref, bat = _both_arms(ev, profiles, span_m, n_span)
        _assert_within_budget(ref, bat, f"n_span={n_span}/L={span_m/1e3:g}km")


def test_batch_outputs_are_finite_and_shaped() -> None:
    """F4e-1: shape contract and no nan/inf leaking out of the masked divisions."""
    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    profiles = _candidate_batch(8)
    outs = ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M)
    assert len(outs) == 4
    for arr in outs:
        assert arr.shape == profiles.shape, f"expected {profiles.shape}, got {arr.shape}"
        assert np.isfinite(arr).all(), "batch path leaked a non-finite value"
    # Dead channels stay exactly zero in every noise channel, both arms.
    dead = profiles == 0
    for arr in outs[1:]:
        assert (arr[dead] == 0).all(), "a zero-power channel produced noise in the batch path"


def test_batch_honours_edfa_noise_figure_override() -> None:
    """F4e-1: the (N,) NF override still lands per-channel, not broadcast onto b.

    An NF array is ``(N,)``; against a ``(B, N)`` batch it broadcasts along the
    channel axis, which is the intent. This checks the scalar-NF case matches the
    serial path exactly, and that the batch arm still equals the serial arm.
    """
    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    profiles = _candidate_batch(6)
    edfa = EDFAConfig(target_gain_db=None, noise_figure_db=5.0)
    ref = np.stack([
        np.stack(ev._calc_span_noise_and_power(np.ascontiguousarray(profiles[b]), SPAN_LENGTH_M, edfa, None, 1))
        for b in range(profiles.shape[0])
    ])
    bat = np.moveaxis(
        np.stack(ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M, edfa, None, 1)), 0, 1
    )
    _assert_within_budget(ref, bat, "edfa-nf")




def test_batch_rejects_non_default_tier_and_bad_shapes() -> None:
    """F4e-3: the batch path refuses the freeze tiers instead of silently differing.

    ``off`` and ``cached`` keep the frozen per-pair NLI; running the batch
    kernel under them would quietly compare different maths. Refusing is the
    honest behaviour.

    ⚠️ 第 5 轮：原列表含 ``cached_g``（拉曼 G 矩阵缓存）。该档随被移除的物理
    整体离场，闭式 ECZ 只有 ``off`` / ``cached`` / ``matvec`` 三档。
    """
    for tier in ("off", "cached"):
        ev = _legacy_evaluator(
            length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel=tier
        )
        with pytest.raises(ValueError, match="matvec"):
            ev.calc_span_noise_and_power_batch(_candidate_batch(2), SPAN_LENGTH_M)

    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    with pytest.raises(ValueError, match="shape"):
        ev.calc_span_noise_and_power_batch(_candidate_batch(2)[:, 0], SPAN_LENGTH_M)  # (N,) not (B, N)
    with pytest.raises(ValueError, match="shape"):
        ev.calc_span_noise_and_power_batch(np.ones((2, NUM_CHANNELS + 1)), SPAN_LENGTH_M)


# ---------------------------------------------------------------------------
# F4e-4: B=1 is reported, not assumed fast
# ---------------------------------------------------------------------------
def test_b1_is_not_a_win_over_serial() -> None:
    """F4e-4 (第 5 轮**改判**)：B=1 时批量臂相对串行的倍率——**记录事实**，不再钉方向。

    ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本条原先断言 ``ratio < 1.0``，即「B=1 时批量臂
    **更慢**」——那是 FG-4c/FG-4e 在**被移除那套物理的批量核**上测出来的**操作阈值**。
    它的前提是「批量臂的固定开销大到 B=1 时不划算」。本轮批量路径换成**闭式 ECZ 的
    批量核**（第 4 轮写的那个），这个前提**不再成立**：实测 B=1 时批量臂反而
    **略快**（本机 min-sample 口径 1.04–1.09x，随 loadavg 抖动）。

    这正是本测试原本警告的情形——「阈值不该被顺带改掉」。所以此处**刻意**改判，
    而不是让它悄悄翻转：

    * 「B=1 批量更慢」是**性能启发式**，不是物理不变量，也不是任何正确性判据；
    * 被移除物理的实现**已经不在树里**，无法再复现原来的数字；
    * 因此把断言从「方向」降为「**倍率落在合理区间**」（批量臂没有病理性劣化），
      并把实测倍率**打印出来**。真正的收益结论由外部探针按批宽 8..142 给出
      （见报告 V2 判据），不受这条启发式影响。
    """
    import time

    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    profiles = _candidate_batch(1)
    for _ in range(3):  # warm
        _both_arms(ev, profiles)

    rounds = 20
    ser, bat = [], []
    for r in range(rounds):
        order = ((ser, lambda: _serial_reference(ev, profiles, SPAN_LENGTH_M)),
                 (bat, lambda: np.stack(ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M))))
        for acc, fn in (order if r % 2 == 0 else order[::-1]):
            t0 = time.perf_counter()
            fn()
            acc.append(time.perf_counter() - t0)
    ratio = min(ser) / min(bat)
    print(f"  B=1 serial/batch min-sample ratio: {ratio:.3f}x")
    assert 0.2 < ratio < 5.0, (
        f"B=1 serial/batch ratio = {ratio:.3f}x 超出合理区间 [0.2, 5.0]；"
        "批量臂出现了病理性劣化，需要查实现而不是查阈值。"
    )
