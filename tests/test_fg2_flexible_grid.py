"""FG-2 flexible-grid acceptance tests (design §5, F2-1 … F2-8).

Every assertion here is executable and recomputable inside this worktree::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-nb-impl \\
        $HOME/miniconda3/envs/rl/bin/python -m pytest tests/test_fg2_flexible_grid.py -v

Mapping to the frozen acceptance table (design §5):

======  ==============================================================
 F2-1   centre frequencies are bit-identical to the legacy fixed grid
 F2-2   the half-slot trap (``(n-1)/2``) is explicitly guarded against
 F2-3   the four kernel returns are ``np.array_equal`` to the pre-FG-2 golden
 F2-4   ``slots_for_rate`` reproduces the four frozen worked examples
 F2-5   wrong-length arrays raise ``ValueError`` naming ``(N,)`` and the shape
 F2-6   centre sampling == band-overlap integral on C+L for widths <= 1600 GHz
 F2-7   ``_calc_exact_power_profile_ode`` body is byte-identical to origin/main
 F2-8   the pre-existing suite stays green with zero expectation edits
======  ==============================================================

F2-1/F2-3 are the executable form of "the fixed grid is a special case of the
flexible grid"; F2-7 is what makes the per-connection call-site change safe.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fg4b_tier_pin_policy import (
    collect_violations,
    tier_pin_violations,
    token_count_violations,
    unauthorized_test_edits,
)

from network_nlo_eval.core.flexgrid import (
    BITS_PER_SYMBOL,
    MODULATION_FORMATS,
    ConnectionSpec,
    FlexibleGrid,
    bits_per_symbol,
    slots_for_rate,
)
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.numba_kernels_p2 import _calc_span_noise_and_power_jit
from network_nlo_eval.network.elements import FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.slots import (
    expand_blocked_channels,
    migrate_roadm_blocked_slots,
    resolve_blocked_slots,
    slots_per_channel,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "results" / "fg2_golden_cplusl_230x50.npz"

# --- Frozen C+L fixed-grid special case (design §2.3) ---------------------
NUM_CHANNELS = 230
CHANNEL_SPACING_HZ = 50e9
SLOT_WIDTH_HZ = 12.5e9
CENTER_FREQUENCY_HZ = 190.2127e12
SLOTS_PER_CHANNEL = 4
NUM_SLOTS = NUM_CHANNELS * SLOTS_PER_CHANNEL  # 920
SYMBOL_RATE_HZ = 50e9
SPAN_LENGTH_M = 100e3
LAUNCH_POWER_W = 1e-3


@pytest.fixture(scope="module")
def legacy_grid() -> SpectrumGrid:
    """The legacy 230x50 GHz fixed grid that the flexible grid must reproduce."""
    return SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )


@pytest.fixture(scope="module")
def flex_grid() -> FlexibleGrid:
    """The flexible grid that is bit-equivalent to ``legacy_grid``."""
    return FlexibleGrid.from_legacy_grid(
        NUM_CHANNELS, CHANNEL_SPACING_HZ, CENTER_FREQUENCY_HZ, slots_per_channel=SLOTS_PER_CHANNEL
    )


@pytest.fixture(scope="module")
def cplusl_connections(flex_grid: FlexibleGrid) -> list[ConnectionSpec]:
    """230 fully loaded 100 Gbps QPSK connections, 4 slots each (case 1)."""
    return [
        ConnectionSpec(
            grid=flex_grid,
            connection_id=i,
            first_slot=i * SLOTS_PER_CHANNEL,
            n_slots=SLOTS_PER_CHANNEL,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=SYMBOL_RATE_HZ,
            roll_off=0.0,
            launch_power_w=LAUNCH_POWER_W,
        )
        for i in range(NUM_CHANNELS)
    ]


@pytest.fixture(scope="module")
def golden() -> dict[str, np.ndarray]:
    """Load the pre-FG-2 golden arrays recorded by scripts/fg2_capture_golden.py."""
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"missing golden file {GOLDEN_PATH}; regenerate it from the pre-FG-2 baseline with "
            "PYTHONPATH=$PWD/src python scripts/fg2_capture_golden.py "
            f"--out {GOLDEN_PATH}"
        )
    with np.load(GOLDEN_PATH) as handle:
        return {key: handle[key] for key in handle.files}


# ======================================================================
# F2-1  centre frequencies are bit-identical to the legacy fixed grid
# ======================================================================
def test_f2_1_centres_bit_identical_to_legacy(flex_grid: FlexibleGrid, legacy_grid: SpectrumGrid) -> None:
    """Raw float64 ``==``, not ``allclose``: the design measured a 0.0 Hz deviation."""
    centres = np.array([flex_grid.centre_hz(i * SLOTS_PER_CHANNEL, SLOTS_PER_CHANNEL) for i in range(NUM_CHANNELS)])
    legacy = legacy_grid.frequencies

    assert np.all(centres == legacy), "centre sampling regressed away from the legacy grid"
    assert float(np.max(np.abs(centres - legacy))) == 0.0
    # The frozen special case itself: 920 slots of 12.5 GHz, first slot aligned to 0 (mod 4).
    assert flex_grid.num_slots == NUM_SLOTS
    assert flex_grid.slot_width_hz == SLOT_WIDTH_HZ
    # f_rel_hz must be reproducible bit-exactly from band_center_hz (design §3.4).
    assert flex_grid.band_center_hz == CENTER_FREQUENCY_HZ
    assert np.all((centres - flex_grid.band_center_hz) == (legacy - legacy_grid.center_frequency_hz))


# ======================================================================
# F2-2  the half-slot trap is explicitly guarded against
# ======================================================================
def test_f2_2_half_slot_trap_offset_is_exactly_half_a_slot(
    flex_grid: FlexibleGrid, legacy_grid: SpectrumGrid
) -> None:
    """A ``(n-1)/2`` numerator shifts everything by half a slot (6.25 GHz).

    This locks in the *failure mode* as well as the fix: if someone reverts
    ``centre_hz`` to the point-lattice form, this test fails.
    """
    starts = np.arange(NUM_CHANNELS) * SLOTS_PER_CHANNEL
    n_slots = np.full(NUM_CHANNELS, float(SLOTS_PER_CHANNEL))
    legacy = legacy_grid.frequencies

    wrong = flex_grid.band_low_edge_hz + (starts + (n_slots - 1) / 2.0) * SLOT_WIDTH_HZ
    right = flex_grid.band_low_edge_hz + (starts + n_slots / 2.0) * SLOT_WIDTH_HZ

    assert np.all(wrong != legacy), "the (n-1)/2 form must NOT reproduce the legacy grid"
    assert float(np.max(np.abs(wrong - legacy))) / 1e9 == pytest.approx(6.25, abs=1e-9)
    assert np.all(right == legacy)


def test_f2_2_single_slot_connection_centre_is_the_slot_centre(flex_grid: FlexibleGrid) -> None:
    """For n=1 both formulas coincide, so the trap only shows for multi-slot spans."""
    for slot in (0, 1, 7, 500):
        assert flex_grid.centre_hz(slot, 1) == flex_grid.slot_center_hz(slot)
        assert flex_grid.centre_hz(slot, SLOTS_PER_CHANNEL) != flex_grid.slot_center_hz(slot)
    # A 4-slot block whose first slot is a multiple of 4 is centred between slots.
    for block in (0, 4, 916):
        centre = flex_grid.centre_hz(block, SLOTS_PER_CHANNEL)
        assert centre == (flex_grid.slot_center_hz(block) + flex_grid.slot_center_hz(block + 3)) / 2


# ======================================================================
# F2-3  kernel returns are np.array_equal to the pre-FG-2 golden
# ======================================================================
def test_f2_3_span_noise_bit_identical_to_golden(
    flex_grid: FlexibleGrid, cplusl_connections: list[ConnectionSpec], golden: dict[str, np.ndarray]
) -> None:
    """Fully loaded 230x50 GHz C+L through the *flexible* path.

    ``symbol_rate_hz`` is now a per-connection array, yet every return must be
    bit-identical to the values recorded before FG-2 touched the kernel.
    """
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    evaluator = MultiBandISRSGN(
        grid=flex_grid,
        ref_fiber_config=fiber,
        connections=cplusl_connections,
        slots_per_channel=SLOTS_PER_CHANNEL,
        # FG-4b §1.2: this test's proposition is "the flexible path did not move a
        # bit" — a port/contract fidelity statement, pinned to the frozen ``off``
        # tier. Production default is ``matvec`` (gated at <=1e-12 elsewhere); the
        # bit-equality assertions and the golden values are unchanged.
        ode_accel="off",
        # FG-15: the Raman tier is pinned to ``legacy`` explicitly. ``off`` is the
        # frozen legacy-only reference path, so this gate would otherwise raise
        # (``tabular`` + ``off`` is rejected by design) instead of running.
        raman_spectrum="legacy",
        nli_model="p2_gn",
    )
    assert evaluator.n_nodes == NUM_CHANNELS
    # Node frequencies are the connection centres, and they match the legacy grid exactly.
    assert np.array_equal(evaluator.f_abs_hz, golden["in__f_abs_hz"])
    assert np.array_equal(evaluator.f_rel_hz, golden["in__f_rel_hz"])
    # Per-connection symbol rate really is an array now (design §2.5 change 1).
    assert isinstance(evaluator.symbol_rate_hz_channels, np.ndarray)
    assert np.all(evaluator.symbol_rate_hz_channels == SYMBOL_RATE_HZ)

    power = np.full(NUM_CHANNELS, LAUNCH_POWER_W)
    p_out, s2_spm, s2_xpm, s2_ase = evaluator._calc_span_noise_and_power(power, SPAN_LENGTH_M, None)

    for name, got in (
        ("p_out_w", p_out),
        ("sigma2_spm_w", s2_spm),
        ("sigma2_xpm_w", s2_xpm),
        ("sigma2_ase_w", s2_ase),
    ):
        expected = golden[name]
        assert np.array_equal(got, expected), f"{name} changed bit-for-bit under the flexible path"
        assert float(np.max(np.abs(got - expected))) == 0.0


def test_f2_3_legacy_path_still_matches_golden(golden: dict[str, np.ndarray], legacy_grid: SpectrumGrid) -> None:
    """The legacy scalar-``Rs`` call site is unchanged too (used by FG-1 parity tests)."""
    kwargs = {
        "power_in_w": np.full(NUM_CHANNELS, LAUNCH_POWER_W),
        "span_length_m": SPAN_LENGTH_M,
        "f_rel_hz": golden["in__f_rel_hz"],
        "f_abs_hz": golden["in__f_abs_hz"],
        "lambdas_m": golden["in__lambdas_m"],
        "alpha_power_npm": golden["in__alpha_power_npm"],
        "a_eff_m2": golden["in__a_eff_m2"],
        "beta2_s2_m": golden["in__beta2_s2_m"],
        "beta3_s3_m": golden["in__beta3_s3_m"],
        "n2_m2_w": golden["in__n2_m2_w"],
        "nf_lin_channels": golden["in__nf_lin_channels"],
        "symbol_rate_hz": np.full(NUM_CHANNELS, SYMBOL_RATE_HZ),
        "n_effective_spans": 1,
        "cr_w_m_hz": golden["in__cr_w_m_hz"],
        "delta_f_co_hz": golden["in__delta_f_co_hz"],
    }
    p_out, s2_spm, s2_xpm, s2_ase = _calc_span_noise_and_power_jit(**kwargs)
    assert np.array_equal(p_out, golden["p_out_w"])
    assert np.array_equal(s2_spm, golden["sigma2_spm_w"])
    assert np.array_equal(s2_xpm, golden["sigma2_xpm_w"])
    assert np.array_equal(s2_ase, golden["sigma2_ase_w"])
    # Guard the golden's provenance so a stale file cannot silently pass.
    assert int(golden["meta__num_channels"]) == NUM_CHANNELS


# ======================================================================
# F2-4  slots_for_rate reproduces the four frozen worked examples
# ======================================================================
def test_f2_4_slots_for_rate_four_worked_examples() -> None:
    """Design §2.2 cases 1-4 (roll_off 0, 0, 0, 0.1 -> 4, 8, 3, 5)."""
    cases = [
        (100e9, "QPSK", 0.0, 4, 50.0e9),  # aligns exactly with the legacy 50 GHz channel
        (400e9, "16QAM", 0.0, 8, 100.0e9),  # 1 connection instead of 2 legacy channels
        (200e9, "64QAM", 0.0, 3, 37.5e9),  # non-integer width -> ceil engages
        (100e9, "QPSK", 0.1, 5, 62.5e9),  # roll-off protection breaks 50 GHz alignment
    ]
    got = [slots_for_rate(rate, bits_per_symbol(fmt), roll, SLOT_WIDTH_HZ) for rate, fmt, roll, _, _ in cases]
    assert got == [4, 8, 3, 5]
    for (rate, fmt, roll, expected_slots, expected_width) in cases:
        n = slots_for_rate(rate, bits_per_symbol(fmt), roll, SLOT_WIDTH_HZ)
        assert n == expected_slots
        assert n * SLOT_WIDTH_HZ == expected_width


def test_f2_4_rolls_off_default_zero_preserves_50ghz_alignment() -> None:
    """[需定夺-2] resolution: roll_off default 0.0, so 100 Gbps QPSK == 4 slots."""
    assert slots_for_rate(100e9, 2, 0.0, SLOT_WIDTH_HZ) == NUM_CHANNELS * 0 + 4


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 2, 0.0, SLOT_WIDTH_HZ),  # bit_rate_bps <= 0
        (-1.0, 2, 0.0, SLOT_WIDTH_HZ),
        (100e9, 0, 0.0, SLOT_WIDTH_HZ),  # bits_per_symbol < 1
        (100e9, 2, 1.0, SLOT_WIDTH_HZ),  # roll_off must be < 1
        (100e9, 2, -0.1, SLOT_WIDTH_HZ),
        (100e9, 2, 0.0, 0.0),  # slot_width_hz <= 0
    ],
)
def test_f2_4_slots_for_rate_rejects_invalid_arguments(args: tuple) -> None:
    with pytest.raises(ValueError):
        slots_for_rate(*args)


def test_f2_4_tiny_rate_still_occupies_one_slot() -> None:
    """occupancy_slots >= 1: a 1 Gbps BPSK connection must not get 0 slots."""
    assert slots_for_rate(1e9, 1, 0.0, SLOT_WIDTH_HZ) == 1


def test_modulation_ladder_matches_the_frozen_spectrum_efficiency() -> None:
    """Design §2.2: BPSK=1, QPSK=2, 8QAM=3, 16QAM=4, 64QAM=6."""
    assert BITS_PER_SYMBOL == {"BPSK": 1, "QPSK": 2, "8QAM": 3, "16QAM": 4, "64QAM": 6}
    assert MODULATION_FORMATS == ("BPSK", "QPSK", "8QAM", "16QAM", "64QAM")
    with pytest.raises(ValueError):
        bits_per_symbol("128QAM")


# ======================================================================
# F2-5  array shape contract
# ======================================================================
def test_f2_5_wrong_length_power_raises_with_expected_and_actual_shape(
    flex_grid: FlexibleGrid, cplusl_connections: list[ConnectionSpec]
) -> None:
    """The message must name both ``(N,)`` and the shape actually supplied."""
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    evaluator = MultiBandISRSGN(
        grid=flex_grid, ref_fiber_config=fiber, connections=cplusl_connections, slots_per_channel=SLOTS_PER_CHANNEL,
        nli_model="p2_gn",
    )
    bad = np.full(NUM_CHANNELS - 1, LAUNCH_POWER_W)
    with pytest.raises(ValueError) as excinfo:
        evaluator._calc_span_noise_and_power(bad, SPAN_LENGTH_M, None)
    message = str(excinfo.value)
    assert f"({NUM_CHANNELS},)" in message
    assert str((NUM_CHANNELS - 1,)) in message


def test_f2_5_wrong_length_nf_array_raises(
    flex_grid: FlexibleGrid, cplusl_connections: list[ConnectionSpec]
) -> None:
    from network_nlo_eval.network.elements import EDFAConfig

    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    evaluator = MultiBandISRSGN(
        grid=flex_grid, ref_fiber_config=fiber, connections=cplusl_connections, slots_per_channel=SLOTS_PER_CHANNEL,
        nli_model="p2_gn",
    )
    edfa = EDFAConfig(noise_figure_db=np.full(NUM_CHANNELS + 5, 5.0))
    with pytest.raises(ValueError) as excinfo:
        evaluator._calc_span_noise_and_power(np.full(NUM_CHANNELS, LAUNCH_POWER_W), SPAN_LENGTH_M, edfa)
    message = str(excinfo.value)
    assert f"({NUM_CHANNELS},)" in message
    assert str((NUM_CHANNELS + 5,)) in message


def test_f2_5_kernel_guards_against_short_symbol_rate_array(golden: dict[str, np.ndarray]) -> None:
    """The njit layer also refuses a short per-connection symbol-rate array."""
    kwargs = {
        "power_in_w": np.full(NUM_CHANNELS, LAUNCH_POWER_W),
        "span_length_m": SPAN_LENGTH_M,
        "f_rel_hz": golden["in__f_rel_hz"],
        "f_abs_hz": golden["in__f_abs_hz"],
        "lambdas_m": golden["in__lambdas_m"],
        "alpha_power_npm": golden["in__alpha_power_npm"],
        "a_eff_m2": golden["in__a_eff_m2"],
        "beta2_s2_m": golden["in__beta2_s2_m"],
        "beta3_s3_m": golden["in__beta3_s3_m"],
        "n2_m2_w": golden["in__n2_m2_w"],
        "nf_lin_channels": golden["in__nf_lin_channels"],
        "symbol_rate_hz": np.full(NUM_CHANNELS - 3, SYMBOL_RATE_HZ),
        "n_effective_spans": 1,
        "cr_w_m_hz": golden["in__cr_w_m_hz"],
        "delta_f_co_hz": golden["in__delta_f_co_hz"],
    }
    with pytest.raises(ValueError):
        _calc_span_noise_and_power_jit(**kwargs)


# ======================================================================
# F2-6  centre sampling is exact on C+L (design §2.4 / appendix A2)
# ======================================================================
_CRAMMER = 0.028 / 1e3 / 1e12  # 1/(W*m*Hz)


def _raman_gain(delta_f_thz: np.ndarray) -> np.ndarray:
    """Transcription of the SRS gain spectrum in the RK4 kernel (13.2/15.6 THz)."""
    d = np.abs(delta_f_thz)
    return np.where(
        d <= 13.2,
        _CRAMMER * d * 1e12,
        np.where(d < 15.6, (_CRAMMER * 13.2e12) * (15.6 - d) / 2.4, 0.0),
    )


def _band_overlap_gain(delta_f_centre_thz: float, bandwidth_thz: float, n: int = 400001) -> float:
    """Band-overlap weighted Raman coupling (the physically correct g_eff)."""
    total = 2 * bandwidth_thz
    w = np.linspace(-total / 2, total / 2, n)
    weight = np.maximum(1 - 2 * np.abs(w) / total, 0.0)
    weight /= np.trapezoid(weight, w)
    return float(np.trapezoid(_raman_gain(delta_f_centre_thz + w) * weight, w))


@pytest.mark.parametrize("width_ghz", [50, 100, 200, 400, 800, 1600])
def test_f2_6_centre_sampling_matches_band_overlap_integral_on_cplusl(width_ghz: float) -> None:
    """For any C+L connection width <= 1600 GHz, centre sampling is exact.

    Rationale (design §2.4): the convolution support [df_c ± (B_i+B_j)/2] never
    crosses the 13.2 THz knee on C+L, and g is linear there, so the triangular
    weight has zero mean and ``g_eff == g(df_c)`` holds independently of width.
    The residual is trapezoid-grid float noise (design measured <= 2.9e-16).
    """
    bandwidth_thz = width_ghz / 1e3
    cplusl_span_thz = 11.45
    assert 2 * bandwidth_thz < cplusl_span_thz, "connection must fit inside C+L"
    dfcs = np.linspace(bandwidth_thz, cplusl_span_thz - bandwidth_thz, 200)

    worst = 0.0
    for dfc in dfcs:
        centre = float(_raman_gain(np.array([dfc]))[0])
        if centre <= 0:
            continue
        worst = max(worst, abs(_band_overlap_gain(float(dfc), bandwidth_thz) - centre) / centre)
    assert worst < 1e-12, f"width={width_ghz} GHz: centre sampling deviates by {worst:.3e}"


def test_f2_6_isolated_wide_connection_still_exact_at_the_band_edge() -> None:
    """The most extreme case: the widest connection tucked against the band edge."""
    bandwidth_thz = 1.6
    centre = float(_raman_gain(np.array([bandwidth_thz]))[0])
    assert centre > 0
    assert abs(_band_overlap_gain(bandwidth_thz, bandwidth_thz) - centre) / centre < 1e-12


# ======================================================================
# F2-7  the RK4 ODE function body is untouched
# ======================================================================
def _ode_source(text: str) -> str:
    """Extract the ``_calc_exact_power_profile_ode`` block verbatim."""
    start = text.index("def _calc_exact_power_profile_ode")
    # Include the decorator line immediately above, if present.
    deco = text.rfind("@njit(cache=True)", 0, start)
    if deco != -1 and text[deco:start].strip() == "@njit(cache=True)":
        start = deco
    match = re.search(r"\n(?=(?:@|def |class |# =====))", text[start + 1 :])
    end = start + 1 + (match.start() + 1 if match else len(text) - start - 1)
    return text[start:end]


@pytest.mark.parametrize("ref", ["origin/main", "8f55e14"])
def test_f2_7_ode_function_body_unchanged(ref: str) -> None:
    """``git diff`` for this function must be empty against the baselines."""
    path = "src/network_nlo_eval/models/numba_kernels_p2.py"
    current = (REPO_ROOT / path).read_text(encoding="utf-8")
    try:
        baseline = subprocess.run(
            ["git", "show", f"{ref}:{path}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"cannot read {ref}:{path}: {exc}")

    assert _ode_source(current) == _ode_source(baseline), f"_calc_exact_power_profile_ode changed vs {ref}"


def test_f2_7_ode_signature_is_the_frozen_one() -> None:
    """Design §3.5 freezes the signature: arrays in, array out, n_steps=20."""
    import inspect

    from network_nlo_eval.models.numba_kernels_p2 import _calc_exact_power_profile_ode

    params = list(inspect.signature(_calc_exact_power_profile_ode.py_func).parameters)
    assert params == ["p_in_w", "f_hz", "alpha_npm", "L_m", "cr_w_m_hz", "n_steps"]
    assert (
        inspect.signature(_calc_exact_power_profile_ode.py_func).parameters["n_steps"].default == 20
    )


def test_f2_7_kernel_takes_frequencies_from_the_caller() -> None:
    """The only permitted change is *which* frequencies the caller passes.

    The ODE consumes ``f_hz`` verbatim, so feeding it connection centres (rather
    than grid frequencies) is a call-site change with no kernel edit.
    """
    import inspect

    source = inspect.getsource(_calc_span_noise_and_power_jit.py_func)
    assert "power_in_w, f_abs_hz, alpha_power_npm, span_length_m, cr_w_m_hz, n_steps=20" in source


# ======================================================================
# F2-8  pre-existing suite untouched
# ======================================================================
#: The G2 policy entry points this file's guard relies on. Kept as a live
#: reference so the import is not dead: ``fg4b_tier_pin_policy`` asserts that
#: this guard still consults them, i.e. that the guard has not been neutered.
_FG4B_G2_POLICY_API = (
    collect_violations,
    tier_pin_violations,
    token_count_violations,
    unauthorized_test_edits,
)


def test_f2_8_no_pre_existing_test_expectations_were_modified() -> None:
    """G2 discipline: the FG-1 baseline commit's test files must be unmodified.

    FG-4b §1.2 authorised exactly one kind of edit to this file: pinning F2-3's
    evaluator to ``ode_accel="off"``. The original filename-level check cannot
    express that, so it is now a *stricter* two-part rule: no unauthorised file
    is touched at all, and within an authorised file the change is proven to be
    confined to the tier pin (no assertion or tolerance added, removed, or
    relaxed). See ``fg4b_tier_pin_policy``.
    """
    unauthorized = unauthorized_test_edits(REPO_ROOT, "8f55e14")
    assert not unauthorized, (
        f"pre-existing test files were modified without §1.2 authorisation: {unauthorized}"
    )

    violations = collect_violations(REPO_ROOT, "8f55e14")
    assert not violations, "§1.2 tier-pin edits must not change any expectation:\n" + "\n".join(violations)


# ======================================================================
# Frozen interface behaviour: FlexibleGrid / ConnectionSpec invariants
# ======================================================================
def test_flexible_grid_invariants_are_enforced(flex_grid: FlexibleGrid) -> None:
    """Enforce the four frozen FlexibleGrid invariants."""
    assert flex_grid.slot_width_hz == 12.5e9
    assert flex_grid.num_slots == 920
    # band edges bracket the occupied centres with the documented overhang.
    legacy_lo = flex_grid.slot_low_edge_hz(0)
    legacy_hi = flex_grid.slot_high_edge_hz(NUM_SLOTS - 1)
    assert flex_grid.band_low_edge_hz == legacy_lo
    assert flex_grid.band_high_edge_hz == legacy_hi

    with pytest.raises(ValueError):
        FlexibleGrid(slot_width_hz=0.0, num_slots=10, band_low_edge_hz=190e12)
    with pytest.raises(ValueError):
        FlexibleGrid(slot_width_hz=12.5e9, num_slots=0, band_low_edge_hz=190e12)
    with pytest.raises(ValueError):
        FlexibleGrid(slot_width_hz=12.5e9, num_slots=True, band_low_edge_hz=190e12)
    with pytest.raises(ValueError):
        FlexibleGrid(slot_width_hz=12.5e9, num_slots=10, band_low_edge_hz=-1.0)


def test_flexible_grid_slot_indexing_is_bounds_checked(flex_grid: FlexibleGrid) -> None:
    assert flex_grid.slot_low_edge_hz(0) < flex_grid.slot_high_edge_hz(0) == flex_grid.slot_low_edge_hz(1)
    for accessor in (flex_grid.slot_low_edge_hz, flex_grid.slot_high_edge_hz, flex_grid.slot_center_hz):
        with pytest.raises(IndexError):
            accessor(NUM_SLOTS)
        with pytest.raises(IndexError):
            accessor(-1)
    with pytest.raises(IndexError):
        flex_grid.centre_hz(-1, 4)
    with pytest.raises(IndexError):
        flex_grid.centre_hz(NUM_SLOTS - 2, 4)
    with pytest.raises(ValueError):
        flex_grid.centre_hz(0, 0)


def test_edges_are_inside_reports_the_documented_overhang(flex_grid: FlexibleGrid) -> None:
    """[需定夺-1] coverage is checked on centres, so edge overhang is expected.

    The overhang is measured against the **nominal band edges** (L_LOW / C_HIGH,
    i.e. where ``BandPlan`` checks coverage), not against the grid's own edges:
    the grid's low edge is 25.0 GHz below L_LOW and its high edge 19.9 GHz above
    C_HIGH. That is precisely why the coverage invariant cannot be applied to
    slot edges on 230 channels.
    """
    l_low, c_high = 184.4877e12, 195.9428e12
    low, high = flex_grid.connection_edges_hz(0, SLOTS_PER_CHANNEL)
    last_low, last_high = flex_grid.connection_edges_hz(NUM_SLOTS - SLOTS_PER_CHANNEL, SLOTS_PER_CHANNEL)

    assert (l_low - low) / 1e9 == pytest.approx(25.0)
    assert (last_high - c_high) / 1e9 == pytest.approx(19.9)

    # The grid's own edges coincide with the outermost connections' edges.
    assert low == flex_grid.band_low_edge_hz
    assert last_high == flex_grid.band_high_edge_hz
    # Both outermost connections therefore stick out of the nominal band...
    assert low < l_low and last_high > c_high
    # ...while every connection *centre* stays inside the nominal band.
    centres = np.array(
        [flex_grid.centre_hz(i * SLOTS_PER_CHANNEL, SLOTS_PER_CHANNEL) for i in range(NUM_CHANNELS)]
    )
    assert centres.min() >= l_low and centres.max() <= c_high
    # `edges_are_inside` is the grid-relative diagnostic: the outermost edges are
    # exactly the grid edges, and the nominal band is strictly inside the grid.
    assert flex_grid.edges_are_inside(low, high)
    assert flex_grid.edges_are_inside(l_low, c_high)
    assert not flex_grid.edges_are_inside(low - 1.0, high + 1.0)


def test_connection_spec_enforces_rate_slot_consistency(flex_grid: FlexibleGrid) -> None:
    """Invariant 3: no silent under-provisioning (4 slots must carry a real rate)."""
    ok = dict(
        grid=flex_grid,
        connection_id=0,
        first_slot=0,
        bit_rate_bps=100e9,
        modulation="QPSK",
        symbol_rate_hz=50e9,
        roll_off=0.0,
        launch_power_w=1e-3,
    )
    assert ConnectionSpec(n_slots=4, **ok).occupied_bandwidth_hz == 50e9
    # 100 Gbps 64QAM needs 100/6 = 16.67 GHz -> 2 slots, not 4.
    with pytest.raises(ValueError):
        ConnectionSpec(n_slots=4, **{**ok, "modulation": "64QAM"})
    with pytest.raises(ValueError):
        ConnectionSpec(n_slots=4, **{**ok, "first_slot": NUM_SLOTS - 2})
    with pytest.raises(ValueError):
        ConnectionSpec(n_slots=4, **{**ok, "modulation": "QPSK", "roll_off": 0.1})
    with pytest.raises(ValueError):
        ConnectionSpec(n_slots=4, **{**ok, "launch_power_w": -1.0})


def test_connection_spec_zero_power_marks_unoccupied_but_keeps_its_slot(flex_grid: FlexibleGrid) -> None:
    """[需定夺-8] unoccupied connections keep their array slot for alignment."""
    idle = ConnectionSpec(
        grid=flex_grid,
        connection_id=7,
        first_slot=28,
        n_slots=4,
        bit_rate_bps=100e9,
        modulation="QPSK",
        symbol_rate_hz=50e9,
        roll_off=0.0,
        launch_power_w=0.0,
    )
    assert idle.is_occupied is False
    assert idle.centre_hz == flex_grid.centre_hz(28, 4)


# ======================================================================
# §2.6  slot-space occupancy and asset migration
# ======================================================================
def test_slots_per_channel_requires_an_exact_multiple() -> None:
    """Require an exact integer multiple, never a rounding guess."""
    assert slots_per_channel(50e9, 12.5e9) == 4
    assert slots_per_channel(100e9, 12.5e9) == 8
    with pytest.raises(ValueError):
        slots_per_channel(45e9, 12.5e9)  # not an integer multiple -> no guessing
    with pytest.raises(ValueError):
        slots_per_channel(50e9, 0.0)


def test_expand_blocked_channels_maps_channels_to_slot_intervals() -> None:
    assert expand_blocked_channels([3, 17], 4) == [12, 13, 14, 15, 68, 69, 70, 71]
    with pytest.raises(ValueError):
        expand_blocked_channels([-1], 4)


def test_resolve_blocked_slots_prefers_the_new_field_and_detects_conflicts() -> None:
    """Accept blocked_slots as authoritative and refuse silent merges."""
    num_slots = NUM_SLOTS
    # blocked_slots wins.
    assert resolve_blocked_slots({"blocked_slots": [12, 13]}, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ, num_slots) == [12, 13]
    # Only the legacy field -> expanded.
    assert resolve_blocked_slots({"blocked_channels": [3]}, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ, num_slots) == [
        12,
        13,
        14,
        15,
    ]
    # Both present and consistent -> accepted.
    assert resolve_blocked_slots(
        {"blocked_slots": [12, 13, 14, 15], "blocked_channels": [3]}, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ, num_slots
    ) == [12, 13, 14, 15]
    # Both present and inconsistent -> refuse to merge silently.
    with pytest.raises(ValueError):
        resolve_blocked_slots(
            {"blocked_slots": [99], "blocked_channels": [3]}, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ, num_slots
        )
    # Out of bounds -> ValueError, never a silent drop.
    with pytest.raises(ValueError):
        resolve_blocked_slots({"blocked_slots": [num_slots]}, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ, num_slots)


def _legacy_graph() -> dict:
    """A 2-node graph carrying the legacy per-channel blocking field."""
    return {
        "directed": False,
        "nodes": [
            {"id": 0, "roadm": {"blocked_channels": [3, 17]}},
            {"id": 1, "roadm": {"blocked_channels": [0]}},
        ],
        "edges": [],
    }


def test_migration_writes_slots_and_keeps_the_legacy_field() -> None:
    """Write the new field while preserving the legacy one."""
    graph = _legacy_graph()
    migrated, stats = migrate_roadm_blocked_slots(graph, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)

    assert stats == {
        "nodes_migrated": 2,
        "nodes_skipped": 0,
        "slots_written": 12,  # 2 nodes: 8 + 4
        "legacy_channels_read": 3,
    }
    assert migrated["nodes"][0]["roadm"]["blocked_slots"] == [12, 13, 14, 15, 68, 69, 70, 71]
    # The old field is preserved verbatim so the diff stays reviewable.
    assert migrated["nodes"][0]["roadm"]["blocked_channels"] == [3, 17]
    # The input graph is not mutated.
    assert "blocked_slots" not in graph["nodes"][0]["roadm"]


def test_migration_is_idempotent() -> None:
    """Assert a second migration pass changes nothing."""
    once, first_stats = migrate_roadm_blocked_slots(_legacy_graph(), SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)
    twice, second_stats = migrate_roadm_blocked_slots(once, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)

    assert first_stats["nodes_migrated"] == 2
    assert second_stats["nodes_migrated"] == 0
    assert second_stats["nodes_skipped"] == 2
    assert twice == once


def test_migration_ratio_is_four_slots_per_legacy_channel() -> None:
    """F3-3 analogue: len(blocked_slots) == 4 * len(blocked_channels)."""
    migrated, _ = migrate_roadm_blocked_slots(_legacy_graph(), SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)
    for node in migrated["nodes"]:
        roadm = node["roadm"]
        assert len(roadm["blocked_slots"]) == 4 * len(roadm["blocked_channels"])


def test_migration_rejects_out_of_range_channels_and_conflicts() -> None:
    """Raise on out-of-range channels and conflicting fields, never drop silently."""
    with pytest.raises(ValueError):
        migrate_roadm_blocked_slots(_legacy_graph(), SLOT_WIDTH_HZ, 45e9)  # spacing not a multiple

    # Capacity declared in the graph -> out-of-bounds is caught, not dropped.
    graph = {"nodes": [{"id": 0, "roadm": {"blocked_channels": [999]}}], "edges": [], "num_slots": NUM_SLOTS}
    with pytest.raises(ValueError):
        migrate_roadm_blocked_slots(graph, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)

    # New and old fields disagree -> refuse to merge.
    conflicting = {
        "nodes": [{"id": 0, "roadm": {"blocked_channels": [3], "blocked_slots": [99]}}],
        "edges": [],
    }
    with pytest.raises(ValueError):
        migrate_roadm_blocked_slots(conflicting, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)


def test_roadm_config_accepts_blocked_slots_and_rejects_duplicates() -> None:
    """Accept blocked_slots while keeping blocked_channels working."""
    roadm = ROADMConfig(blocked_slots=[12, 13, 14, 15], blocked_channels=[3])
    assert roadm.blocked_slot_set() == {12, 13, 14, 15}
    with pytest.raises(ValueError):
        ROADMConfig(blocked_slots=[1, 1])
    with pytest.raises(ValueError):
        ROADMConfig(blocked_slots=[-1])
    # The legacy field alone still works (read-only compatibility input).
    assert ROADMConfig(blocked_channels=[2]).blocked_slot_set() == {2}


def test_wide_connection_is_blocked_by_overlap_not_equality(
    flex_grid: FlexibleGrid, cplusl_connections: list[ConnectionSpec]
) -> None:
    """[Design §2.6] a wide connection is infeasible if it overlaps *any* blocked slot.

    This is the documented semantic change: blocking gets finer-grained, so a
    solution that was feasible on the fixed grid can become infeasible.
    """
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    evaluator = MultiBandISRSGN(
        grid=flex_grid, ref_fiber_config=fiber, connections=cplusl_connections, slots_per_channel=SLOTS_PER_CHANNEL,
        nli_model="p2_gn",
    )
    # Block a single slot that sits inside connection 10 (slots 40..43).
    roadm = ROADMConfig(blocked_slots=[41])
    assert evaluator._resolve_blocked_node_indices(roadm) == [10]
    # A fully loaded grid has no orphan slots: the last slot belongs to connection 229.
    assert evaluator._resolve_blocked_node_indices(ROADMConfig(blocked_slots=[NUM_SLOTS - 1])) == [NUM_CHANNELS - 1]
    # A gap outside every connection is a no-op.
    sparse = cplusl_connections[:100]
    sparse_eval = MultiBandISRSGN(
        grid=flex_grid, ref_fiber_config=fiber, connections=sparse, slots_per_channel=SLOTS_PER_CHANNEL,
        nli_model="p2_gn",
    )
    assert sparse_eval._resolve_blocked_node_indices(ROADMConfig(blocked_slots=[500])) == []


def test_band_edges_follow_connection_widths(flex_grid: FlexibleGrid) -> None:
    """Design §2.5: f_m/f_M must track real connection band edges, not a single Rs/2."""
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    # A single 1600 GHz connection: 128 slots of 12.5 GHz, BPSK at Rs = W.
    wide = ConnectionSpec(
        grid=flex_grid,
        connection_id=0,
        first_slot=0,
        n_slots=128,
        bit_rate_bps=1600e9,
        modulation="BPSK",
        symbol_rate_hz=1600e9,
        roll_off=0.0,
        launch_power_w=1e-3,
    )
    evaluator = MultiBandISRSGN(
        grid=flex_grid, ref_fiber_config=fiber, connections=[wide], slots_per_channel=SLOTS_PER_CHANNEL,
        nli_model="p2_gn",
    )
    assert (evaluator.f_M_hz - evaluator.f_m_hz) / 1e12 == pytest.approx(1.6)
    # The fully loaded 230x50 GHz case collapses back to one legacy channel width.
    full = [
        ConnectionSpec(
            grid=flex_grid,
            connection_id=i,
            first_slot=i * 4,
            n_slots=4,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=50e9,
            roll_off=0.0,
            launch_power_w=1e-3,
        )
        for i in range(NUM_CHANNELS)
    ]
    full_eval = MultiBandISRSGN(
        grid=flex_grid, ref_fiber_config=fiber, connections=full, slots_per_channel=SLOTS_PER_CHANNEL,
        nli_model="p2_gn",
    )
    assert (full_eval.f_M_hz - full_eval.f_m_hz) / 1e12 == pytest.approx((NUM_CHANNELS - 1) * 50e9 / 1e12 + 50e-3)
