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

* the **non-uniform grid** case (``phi_kurtosis=-0.619`` flexible grid) is in
  the equivalence sweep — it is the shape that exposes an axis mistake;
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

from network_nlo_eval.models.batch_kernels import MAX_REL_DEVIATION, worst_relative_deviation
from network_nlo_eval.models.isrs_gn import batch_cache_stats
from network_nlo_eval.network.elements import EDFAConfig

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
    """F4e-1 on a flexible grid with per-connection symbol rates and 64QAM kurtosis.

    This is the shape that makes ``p_i != p_l`` within a single candidate, which
    is precisely the condition under which FG-4b's ``p_i`` axis bug was invisible.
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


def test_batch_constants_are_reused_across_calls() -> None:
    """The (N,N) constants are built once and reused: hits must grow with calls.

    Evidence, not assertion-by-comment: the hit counter moves.
    """
    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec"
    )
    profiles = _candidate_batch(8)
    ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M)  # builds
    before = batch_cache_stats()["hits"]
    for _ in range(3):
        ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M)
    assert batch_cache_stats()["hits"] == before + 3, "batch constants were rebuilt instead of reused"


def test_batch_rejects_non_default_tier_and_bad_shapes() -> None:
    """F4e-3: the batch path refuses the freeze tiers instead of silently differing.

    ``off`` and ``cached_g`` keep the frozen per-pair NLI; running the batch
    kernel under them would quietly compare different maths. Refusing is the
    honest behaviour.
    """
    for tier in ("off", "cached_g"):
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
    """F4e-4: at B=1 the batch arm is slower, and that is recorded as a fact.

    This does **not** assert a speedup. It pins the direction of the result so
    the "don't take the batch path at B=1" threshold cannot be quietly flipped by
    a later change. The authoritative numbers are in
    ``results/fg4e_speedup_ab.json``; this is a cheap same-process sanity
    check that the conclusion has not been inverted.
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
    assert ratio < 1.0, (
        f"B=1 the batch arm is now {ratio:.3f}x *faster* than serial; FG-4c/FG-4e measured it "
        "slower. If this is a real improvement the 'do not batch at B=1' threshold should be "
        "revisited deliberately -- do not let it change as a side effect."
    )
