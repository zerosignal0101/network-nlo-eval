r"""FG-4a numeric-identity gate: the ODE/QoT accelerations must not move a bit.

Every assertion here is executable and recomputable inside this worktree::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \\
        $HOME/miniconda3/envs/rl/bin/python -m pytest tests/test_fg4a_numeric_identity.py -v

Mapping to the FG-4a acceptance table:

======  ==============================================================
 F4a-1  the golden + this gate exist and the suite is green
 F4a-3  ``ode_accel="off"`` and ``"cached_g"`` are **bit-identical** to golden
 F4a-4  ``ode_accel="matvec"`` stays within 1e-12 relative, deviation reported
 F4a-5  the ``G`` cache is provably reused across calls (not merely built)
======  ==============================================================

How the gate works
------------------
The golden in ``results/fg4a_golden.json`` was captured at the pre-FG-4a
baseline commit ``ea7ecb0``, from the *stable evaluator API* rather than from
the njit kernel's argument list. That is deliberate: FG-4a rewrites the kernel's
internals and signature, so a kernel-level golden would have to be regenerated
by the very changes it is supposed to police.

Each of the three acceleration modes is held to the standard its own acceptance
ID requires:

=============  =========================================================
``off``        the FG-2 frozen reference path — must not move **at all**
``cached_g``   FG-4a optimisation 1 — must not move **at all** (F4a-3)
``matvec``     FG-4a optimisation 2 — allowed <= 1e-12 relative (F4a-4)
=============  =========================================================

The matvec tolerance is not a general licence: it is a module constant capped at
1e-12, asserted to still be 1e-12, and it applies **only** to that one mode.
``snr_db`` legitimately contains +/-inf, so comparisons use ``equal_nan=True``
and the deviation statistics are taken over finite entries only; a change in
*which* entries are infinite is treated as a hard failure rather than a
rounding shift, because that is a semantic change, not a float-reordering one.

If a future change breaks the gate, the correct response is to fix the change.
Regenerating ``fg4a_golden.json`` from modified code defeats the entire purpose
of this file.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fg4a_scenarios import PATH_KEYS, SPAN_KEYS, build_scenarios
from fg4b_tier_pin_policy import (
    collect_violations,
    tier_pin_violations,
    token_count_violations,
    unauthorized_test_edits,
)

from network_nlo_eval.models.isrs_gn import ODE_ACCEL_MODES

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "results" / "fg4a_golden.json"

#: The commit the golden was captured at, before any FG-4a acceleration.
BASELINE_REV = "ea7ecb0"

#: Modes required to be bit-identical to the golden (F4a-3).
#:
#: WP-MERGE-FG-FLEX：``"cached"`` 是 flexgrid 侧并入的**跨段结果记忆化**档，
#: 它缓存的就是 ``"off"`` 档同一个内核的输出，因此按构造与 ``"off"`` 逐位相等，
#: 标准即 bit-exact。把它列进来不是给守卫开口子——`test_f4a_1_all_modes_are_exercised`
#: 要求"每个声明的档位都必须有标准"，而 `BIT_EXACT_IDS`（下）会用本模块的金标准
#: 真跑一遍 ``cached``，因此这是**扩证据**而不是放宽阈值。
BIT_EXACT_MODES = ("off", "cached", "cached_g")

#: Modes allowed a bounded float-reordering shift (F4a-4).
BOUNDED_MODES = ("matvec",)

#: Upper bound on relative deviation allowed by F4a-4. This is the brief's
#: 1e-12 and it is deliberately a module constant: a failing gate must be
#: fixed in the implementation, never by loosening this number.
MAX_REL_DEVIATION_ALLOWED = 1e-12


def _load_golden() -> dict[str, Any]:
    """Read the golden file, failing loudly if it is missing or malformed."""
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"{GOLDEN_PATH} is missing. It must be captured at the FG-4a baseline commit "
            f"{BASELINE_REV} with scripts/fg4a_capture_golden.py — never from modified code."
        )
    try:
        return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover
        pytest.fail(f"{GOLDEN_PATH} is not valid JSON: {exc}")


GOLDEN = _load_golden()


def _expected_keys(scenario: dict[str, Any]) -> tuple[str, ...]:
    """Which output keys a golden scenario must carry."""
    return SPAN_KEYS if scenario["kind"] == "span" else PATH_KEYS


def _deviation(expected: np.ndarray, actual: np.ndarray) -> tuple[int, int, float, float]:
    """Compare two float arrays, tolerating matching infinities.

    Returns ``(n_differing, n_finiteness_mismatches, max_abs, max_rel)`` where the
    two maxima are taken over the elements finite on **both** sides. A
    finiteness mismatch (an infinity appearing or vanishing) is reported
    separately because it is a semantic change, not a rounding shift.
    """
    n_differing = int(np.count_nonzero(expected != actual))
    n_nonfinite_mismatch = int(np.count_nonzero(np.isfinite(expected) != np.isfinite(actual)))
    both_finite = np.isfinite(expected) & np.isfinite(actual)
    if not both_finite.any():
        return n_differing, n_nonfinite_mismatch, 0.0, 0.0
    exp_f = expected[both_finite]
    act_f = actual[both_finite]
    diff = np.abs(exp_f - act_f)
    denom = np.abs(exp_f)
    # ``np.divide(..., where=)`` rather than ``np.where(denom>0, a/b, 0)``: the
    # latter still evaluates a/b on masked-out elements and warns on 0/0.
    rel = np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)
    return n_differing, n_nonfinite_mismatch, float(diff.max()), float(rel.max())


def _golden_cases() -> list[tuple[str, str]]:
    """Enumerate (scenario_name, key) for every golden output."""
    return [(name, key) for name, entry in GOLDEN.get("scenarios", {}).items() for key in _expected_keys(entry)]


def _live(scenario_name: str, key: str, ode_accel: str) -> np.ndarray:
    """Recompute one output under the requested acceleration mode."""
    scenario = next(s for s in build_scenarios(ode_accel) if s.name == scenario_name)
    return np.asarray(scenario.run()[key], dtype=np.float64)


def _golden_array(scenario_name: str, key: str) -> np.ndarray:
    """The stored golden value for one output."""
    return np.asarray(GOLDEN["scenarios"][scenario_name]["outputs"][key], dtype=np.float64)


BIT_EXACT_IDS = [f"{mode}::{name}::{key}" for mode in BIT_EXACT_MODES for name, key in _golden_cases()]
BOUNDED_IDS = [f"{mode}::{name}::{key}" for mode in BOUNDED_MODES for name, key in _golden_cases()]


def _resolve(case: str) -> tuple[str, str, str]:
    """Split a ``mode::scenario::key`` case id."""
    mode, scenario_name, key = case.split("::", 2)
    return mode, scenario_name, key


# ======================================================================
# F4a-1  the gate itself is wired up correctly
# ======================================================================
def test_f4a_1_golden_covers_every_declared_scenario() -> None:
    """The golden must contain exactly the scenarios the module declares.

    A scenario added to :mod:`fg4a_scenarios` without regenerating the golden
    would otherwise be silently untested, which is the failure mode most likely
    to let a regression through.
    """
    declared = {scenario.name for scenario in build_scenarios("off")}
    stored = set(GOLDEN.get("scenarios", {}))
    assert stored == declared, (
        f"golden scenarios {sorted(stored)} != declared {sorted(declared)}; "
        "regenerate the golden at the baseline commit if a scenario was added"
    )


def test_f4a_1_golden_records_its_provenance() -> None:
    """Provenance must be present and must point at this worktree.

    The repo is installed editable with a ``.pth`` resolving to the *main*
    checkout regardless of cwd, so a mis-scoped ``PYTHONPATH`` silently runs
    another tree's code. Asserting the recorded path catches that.

    FG-14 §2 re-pinned this golden, because ``evaluate_path`` moved to the in-repo
    authoritative ROADM assembly. A re-pin must not erase the chain: the golden
    now carries ``repins``, and the **original** capture revision still has to be
    the FG-4a baseline. Asserting only the newest ``git_rev`` would let any later
    re-capture silently overwrite the reference.
    """
    provenance = GOLDEN.get("provenance", {})
    repins = provenance.get("repins", [])
    root_rev = (repins[0]["git_rev"] if repins else provenance.get("git_rev", ""))
    assert root_rev.startswith(BASELINE_REV), (
        f"the original capture is no longer the FG-4a baseline {BASELINE_REV}: {root_rev}"
    )
    assert provenance.get("package_path"), "golden does not record which tree produced it"
    assert "network_nlo_eval" in provenance["package_path"]
    if provenance.get("git_rev", "") != root_rev:
        # The golden was re-captured; each hop must name who authorised it.
        assert repins, (
            "golden was re-captured but records no repins chain — a re-pin without a named "
            "adjudicator is indistinguishable from silently rewriting the reference"
        )
        for hop in repins:
            assert hop.get("git_rev") and hop.get("adjudicator"), f"unattributed re-pin hop: {hop}"


def test_f4a_1_evaluated_package_is_this_worktree() -> None:
    """The tree under test must be the one this test file lives in.

    This is the cross-worktree guard, made executable: if the package resolves
    to the main checkout instead of this worktree, every assertion below would
    be measuring someone else's code.
    """
    import network_nlo_eval

    resolved = Path(network_nlo_eval.__file__).resolve()
    expected_root = REPO_ROOT / "src" / "network_nlo_eval"
    assert resolved.is_relative_to(expected_root), (
        f"network_nlo_eval resolved to {resolved}, which is not under {expected_root}. "
        "Set PYTHONPATH=$PWD/src — the editable install points at the main checkout."
    )


def test_f4a_1_all_modes_are_exercised() -> None:
    """Every declared mode must be covered by a standard in this module.

    A new mode added to the evaluator without being assigned a standard would
    otherwise go untested, which is exactly how a silent numerical regression
    would slip in.
    """
    covered = set(BIT_EXACT_MODES) | set(BOUNDED_MODES)
    assert covered == set(ODE_ACCEL_MODES), (
        f"modes {set(ODE_ACCEL_MODES)} are not fully covered by "
        f"bit-exact {set(BIT_EXACT_MODES)} + bounded {set(BOUNDED_MODES)}"
    )
    assert not (set(BIT_EXACT_MODES) & set(BOUNDED_MODES))


def test_f4a_1_scenarios_are_deterministic() -> None:
    """Re-running a scenario must reproduce its own output exactly.

    A scenario that is not self-consistent makes every comparison above
    meaningless, so this is checked before any of them.
    """
    for mode in ODE_ACCEL_MODES:
        for scenario in build_scenarios(mode):
            first = scenario.outputs()
            second = scenario.outputs()
            for key, value in first.items():
                assert np.array_equal(
                    np.asarray(value, dtype=np.float64),
                    np.asarray(second[key], dtype=np.float64),
                    equal_nan=True,
                ), f"scenario {scenario.name!r}.{key} ({mode}) is not deterministic"


# ======================================================================
# F4a-3  optimisation 1: bit-identical
# ======================================================================
@pytest.mark.parametrize("case", BIT_EXACT_IDS, ids=BIT_EXACT_IDS)
def test_f4a_3_bit_identical_to_golden(case: str) -> None:
    """``off`` and ``cached_g`` must reproduce the golden with zero bit changes."""
    mode, scenario_name, key = _resolve(case)
    expected = _golden_array(scenario_name, key)
    actual = _live(scenario_name, key, mode)

    assert actual.shape == expected.shape, f"{case}: shape {expected.shape} -> {actual.shape}"
    n_differing, n_nonfinite, max_abs, _max_rel = _deviation(expected, actual)
    assert n_nonfinite == 0, f"{case}: {n_nonfinite} elements changed between finite and infinite"
    assert n_differing == 0, (
        f"{case}: mode {mode!r} must be bit-identical to the golden, but "
        f"{n_differing}/{expected.size} elements moved (max abs deviation {max_abs:.3e})"
    )


def test_f4a_3_cached_g_matches_off_exactly() -> None:
    """``cached_g`` must equal ``off`` bit-for-bit, independent of the golden.

    A direct A/B on the same inputs: even if the golden were accidentally
    regenerated, this comparison still pins optimisation 1 to zero drift.
    """
    for scenario in build_scenarios("off"):
        off = scenario.run()
        cached = next(s for s in build_scenarios("cached_g") if s.name == scenario.name).run()
        for key, value in off.items():
            assert np.array_equal(
                np.asarray(value, dtype=np.float64),
                np.asarray(cached[key], dtype=np.float64),
                equal_nan=True,
            ), f"{scenario.name}.{key}: G cache changed the result"


# ======================================================================
# F4a-4  optimisation 2: bounded deviation, reported
# ======================================================================
@pytest.mark.parametrize("case", BOUNDED_IDS, ids=BOUNDED_IDS)
def test_f4a_4_matvec_within_budget(case: str) -> None:
    """``matvec`` may reorder the summation, but only within 1e-12 relative."""
    mode, scenario_name, key = _resolve(case)
    expected = _golden_array(scenario_name, key)
    actual = _live(scenario_name, key, mode)

    assert actual.shape == expected.shape, f"{case}: shape {expected.shape} -> {actual.shape}"
    n_differing, n_nonfinite, max_abs, max_rel = _deviation(expected, actual)
    assert n_nonfinite == 0, f"{case}: {n_nonfinite} elements changed between finite and infinite"
    assert max_rel <= MAX_REL_DEVIATION_ALLOWED, (
        f"{case}: relative deviation {max_rel:.3e} exceeds the {MAX_REL_DEVIATION_ALLOWED:.0e} "
        f"budget (max abs {max_abs:.3e}, {n_differing}/{expected.size} elements differ)"
    )


def test_f4a_4_measured_deviation_is_bounded_and_reported() -> None:
    """Measure the worst matvec deviation across the whole golden and report it.

    Kept separate from the per-case test so the number is recorded once, in one
    place, instead of hiding inside a parametrised pass.
    """
    worst = 0.0
    worst_where = "-"
    total_differing = 0
    bit_exact_cases = 0
    n_cases = 0
    for _scenario_name, key in _golden_cases():
        expected = _golden_array(_scenario_name, key)
        actual = _live(_scenario_name, key, "matvec")
        n_cases += 1
        n_differing, _nonfinite, _max_abs, max_rel = _deviation(expected, actual)
        if n_differing == 0:
            bit_exact_cases += 1
            continue
        total_differing += n_differing
        if max_rel > worst:
            worst = max_rel
            worst_where = f"{_scenario_name}.{key}"

    # Printed unconditionally: this is the F4a-4 evidence line.
    print(
        f"\nFG-4a matvec deviation report: {bit_exact_cases}/{n_cases} cases bit-identical "
        f"anyway; worst relative deviation {worst:.3e} at {worst_where} "
        f"({total_differing} differing elements in total)"
    )
    assert worst <= MAX_REL_DEVIATION_ALLOWED, (
        f"worst relative deviation {worst:.3e} ({worst_where}) exceeds {MAX_REL_DEVIATION_ALLOWED:.0e}"
    )


def test_f4a_tolerance_budget_is_not_loosened() -> None:
    """Guard the budget itself against being widened to make a failure pass.

    F4a-4 caps the branchless matvec's reordering at 1e-12. If a later change
    wants a looser bound it has to change this test, which is the point.
    """
    assert MAX_REL_DEVIATION_ALLOWED == 1e-12


# ======================================================================
# F4a-5  the G cache is genuinely reused
# ======================================================================
def test_f4a_5_g_cache_is_reused_not_rebuilt() -> None:
    """A second evaluator on the same grid must hit the cache, not rebuild G.

    This is what distinguishes a real cache from "G happens to be cheap now".
    The counters are module-level, so the check is: first build increments
    ``builds`` once, and every subsequent evaluator on the same grid only
    increments ``hits``.
    """
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models import numba_kernels_p2 as nk
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    grid = SpectrumGrid(num_channels=64, channel_spacing_hz=50e9, center_frequency_hz=190.2127e12)
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)

    def build() -> MultiBandISRSGN:
        return MultiBandISRSGN(grid=grid, ref_fiber_config=fiber, symbol_rate_hz=50e9, ode_accel="matvec",
            nli_model="p2_gn",)

    before = nk.g_cache_stats()
    first = build()
    after_first = nk.g_cache_stats()
    second = build()
    after_second = nk.g_cache_stats()

    assert after_first["builds"] == before["builds"] + 1, "first evaluator did not build G once"
    assert after_second["builds"] == after_first["builds"], (
        f"second evaluator rebuilt G (builds {after_first['builds']} -> {after_second['builds']}); "
        "the cache is not keyed correctly"
    )
    assert after_second["hits"] == after_first["hits"] + 1, "second evaluator did not hit the cache"
    # The cached pair must be the very same objects, not equal copies.
    assert first._g_gain is second._g_gain
    assert first._g_loss is second._g_loss


def test_f4a_5_cached_matrices_are_read_only() -> None:
    """Shared cached matrices must not be writable.

    A cached matrix handed to a caller that mutates it would silently corrupt
    every later span on that grid, so the cache returns read-only views.
    """
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models import numba_kernels_p2 as nk

    grid = SpectrumGrid(num_channels=32, channel_spacing_hz=50e9, center_frequency_hz=190.2127e12)
    g_gain, g_loss = nk.raman_g_pair_matrices(grid.frequencies, 0.028 / 1e3 / 1e12)
    assert not g_gain.flags.writeable, "cached g_gain must be read-only"
    assert not g_loss.flags.writeable, "cached g_loss must be read-only"
    with pytest.raises(ValueError):
        g_gain[0, 0] = 1.0


def test_f4a_g_split_matrices_partition_the_reference_g() -> None:
    """``g_gain``/``g_loss`` must encode exactly the original per-pair branches.

    Rebuilds the baseline G by hand and checks the partition invariants: gain
    entries only where the frequency is higher, loss entries only where it is
    lower and equal to ``(f_k/f_i) * G[i, k]``, and the diagonal left at zero.
    """
    from network_nlo_eval.models import numba_kernels_p2 as nk

    f = 190.0e12 + np.arange(8) * 50e9
    cr = 0.028 / 1e3 / 1e12
    g_full = nk._build_g_matrix(f, cr)
    g_gain, g_loss = nk._build_g_pair_matrices(f, cr)

    for i in range(len(f)):
        for k in range(len(f)):
            if f[k] > f[i]:
                assert g_gain[i, k] == g_full[i, k]
                assert g_loss[i, k] == 0.0
            elif f[k] < f[i]:
                assert g_loss[i, k] == pytest.approx((f[k] / f[i]) * g_full[i, k], rel=0, abs=0)
                assert g_gain[i, k] == 0.0
            else:
                assert g_gain[i, k] == 0.0
                assert g_loss[i, k] == 0.0


# ======================================================================
# Structural guarantees the two optimisations must not erode
# ======================================================================
def test_f4a_frozen_ode_body_is_still_byte_identical() -> None:
    """FG-2's F2-7 froze the ODE body; FG-4a must not have edited it.

    The accelerations live *alongside* the frozen function (a cached-G variant
    and a branchless matvec), so the original must remain untouched. Re-asserted
    here so an FG-4a regression cannot silently break the FG-2 guarantee.
    """
    path = "src/network_nlo_eval/models/numba_kernels_p2.py"
    current = (REPO_ROOT / path).read_text(encoding="utf-8")
    try:
        baseline = subprocess.run(
            ["git", "show", f"{BASELINE_REV}:{path}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover
        pytest.skip(f"cannot read {BASELINE_REV}:{path}: {exc}")

    def ode_source(text: str) -> str:
        start = text.index("def _calc_exact_power_profile_ode")
        deco = text.rfind("@njit(cache=True)", 0, start)
        if deco != -1 and text[deco:start].strip() == "@njit(cache=True)":
            start = deco
        return text[start : text.index("\n\n\n", start)]

    assert ode_source(current) == ode_source(baseline), (
        "_calc_exact_power_profile_ode was edited; FG-4a must add accelerated variants "
        "rather than modify the function FG-2 froze"
    )


def test_f4a_frozen_dp_dz_is_still_byte_identical() -> None:
    """``_calc_dp_dz`` is the bit-exact reference for optimisation 2; do not touch it."""
    from network_nlo_eval.models.numba_kernels_p2 import _calc_dp_dz

    source = Path(sys.modules[_calc_dp_dz.__module__].__file__).read_text(encoding="utf-8")
    start = source.index("def _calc_dp_dz(")
    assert "if f_hz[k] > f_hz[i]:" in source[start : source.index("\n\n\n", start)], (
        "_calc_dp_dz lost its per-pair branch; the branchless variant must be a "
        "separate function so the reference implementation stays available"
    )


#: The G2 policy entry points this file's guard relies on. Kept as a live
#: reference so the import is not dead: ``fg4b_tier_pin_policy`` asserts that
#: this guard still consults them, i.e. that the guard has not been neutered.
_FG4B_G2_POLICY_API = (
    collect_violations,
    tier_pin_violations,
    token_count_violations,
    unauthorized_test_edits,
)


def test_f4a_pre_existing_tests_were_not_modified() -> None:
    """G2 discipline: no pre-existing test file may be edited without authority.

    The FG-4a additions are new files, which show up as ``A``; anything else
    under ``tests/`` is an edit to an existing file.

    FG-4b §1.2 authorised two things and nothing else: pinning F1-1/F1-2/F2-3 to
    ``ode_accel="off"`` (their proposition is port fidelity, not the production
    default), and updating the two G2 guard functions themselves so the check can
    tell those apart. The original filename-level test could not express that, so
    this is now *stricter*: an unauthorised file still fails outright, and within
    an authorised file no line carrying an assertion or a numeric tolerance may
    have been removed, while the guards must still exist and still consult the
    policy. See ``fg4b_tier_pin_policy``.
    """
    unauthorized = unauthorized_test_edits(REPO_ROOT, BASELINE_REV)
    assert not unauthorized, (
        f"pre-existing test files were modified without §1.2 authorisation: {unauthorized}"
    )

    violations = collect_violations(REPO_ROOT, BASELINE_REV)
    assert not violations, "§1.2 tier-pin edits must not change any expectation:\n" + "\n".join(violations)


def test_f4a_capture_script_check_mode_passes() -> None:
    """The generator's own ``--check`` must agree with this test.

    Two independent implementations of the same comparison, so a bug in one is
    unlikely to hide in both.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "fg4a_capture_golden.py"),
            "--out",
            str(GOLDEN_PATH),
            "--check",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"fg4a_capture_golden.py --check failed:\n{result.stdout}\n{result.stderr}"
