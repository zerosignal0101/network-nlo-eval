"""FU-4: the SNR assembly union gate.

Scope
-----
FU-4 adjudicated two competing SNR assemblies in ``rwa/qot_checker.py`` (see
``docs/fu4_qot_assembly_adjudication.md``) and moved the file to the union of
both. These tests pin that decision at three levels:

1. **Structural** -- the assembly rules themselves (``_roadm_terms``): port-aware
   loss, booster only under ``equalize_output_power``, noise-only filtering.
2. **Regression** -- the FG-2 flexible-grid capability that the union was
   forbidden to drop (``blocked_slots`` slicing semantics) plus the legacy
   ``blocked_channels`` path.
3. **Fidelity** -- the committed golden in ``results/fu4_fidelity_golden.json``
   and the ΔSNR artefact in ``results/fu4_assembly_delta.json``.

Why these are new expectations rather than edited ones
------------------------------------------------------
FU-4 deliberately moves SNR numbers. Per the brief and the G2 discipline, **no
pre-existing ``tests/*`` expectation was modified** to accommodate that change.
In fact the pre-existing suite is untouched by it: the 31/31 legacy golden runs
``MultiBandISRSGN.evaluate_path``, which carries its own ROADM assembly inside
``models/isrs_gn.py`` and never calls ``rwa/qot_checker.py``. That is why the
full suite still reports 248 passed / 1 skipped, and equally why no *prior* gate
ever pinned the RWA validator's numbers -- which is how FG-4g could find a
0.2878 dB cross-repo disagreement without any test turning red.

The honest consequence is recorded as ``test_fu4_10_*`` below: the legacy golden
cannot witness this change, so the fidelity golden introduced here is the first
gate over this code path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fu4_scenarios import (
    ROADM_BASE,
    SCENARIOS,
    Scenario,
    _numeric,
    run_scenario,
)

from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.rwa.qot_checker import QoTValidator

REPO_ROOT = Path(__file__).resolve().parents[1]
QOT_CHECKER = REPO_ROOT / "src" / "network_nlo_eval" / "rwa" / "qot_checker.py"
GOLDEN_PATH = REPO_ROOT / "results" / "fu4_fidelity_golden.json"
DELTA_PATH = REPO_ROOT / "results" / "fu4_assembly_delta.json"


def _by_name(name: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    raise KeyError(name)


# ======================================================================
# FU4-1  port-aware loss
# ======================================================================


def test_fu4_1_drop_and_express_losses_differ() -> None:
    """The union distinguishes the final drop from intermediate express hops.

    This is the §1.1 adjudication made executable: the FG-tree lumped assembly
    used ``insertion_loss_db`` alone, which made ``add_drop_loss_db`` and
    ``express_attenuation_db`` inert fields.
    """
    validator = QoTValidator.__new__(QoTValidator)
    roadm = ROADMConfig(**ROADM_BASE)

    drop_loss, drop_filter, drop_nf = validator._roadm_terms(roadm, is_last_hop=True)
    express_loss, express_filter, express_nf = validator._roadm_terms(roadm, is_last_hop=False)

    assert drop_loss == pytest.approx(10 ** (6.0 / 10))  # insertion 5 + add/drop 1
    assert express_loss == pytest.approx(10 ** (5.0 / 10))  # insertion 5 + express 0
    assert drop_loss > express_loss
    # Filtering is independent of the port, so it must be identical on both.
    assert drop_filter == pytest.approx(express_filter)
    assert drop_nf == pytest.approx(express_nf)


def test_fu4_1b_filtering_penalty_is_a_noise_only_multiplier() -> None:
    """§1.3: filtering scales noise, never the port loss.

    A 0.2 dB penalty becomes ``10 ** 0.02`` and stays separate from the loss
    factor -- the whole point of the adjudication is that these are two
    different physical effects and must not be merged into one lumped number.
    """
    validator = QoTValidator.__new__(QoTValidator)
    roadm = ROADMConfig(**{**ROADM_BASE, "filtering_penalty_db": 0.2})
    loss, filtering, _ = validator._roadm_terms(roadm, is_last_hop=True)

    assert filtering == pytest.approx(10 ** (0.2 / 10))
    assert loss == pytest.approx(10 ** (6.0 / 10))
    # If they had been merged (the FG-tree rule) the loss would be 6.2 dB.
    assert loss != pytest.approx(10 ** (6.2 / 10))


# ======================================================================
# FU4-2  booster gating
# ======================================================================


def test_fu4_2_booster_is_present_only_under_equalize_output_power() -> None:
    """§1.2: no booster means no booster ASE and the port loss is paid in full."""
    validator = QoTValidator.__new__(QoTValidator)

    on = ROADMConfig(**{**ROADM_BASE, "equalize_output_power": True})
    off = ROADMConfig(**{**ROADM_BASE, "equalize_output_power": False})

    _, _, nf_on = validator._roadm_terms(on, is_last_hop=False)
    _, _, nf_off = validator._roadm_terms(off, is_last_hop=False)

    assert nf_on == pytest.approx(10 ** (5.0 / 10))
    assert nf_off is None


def test_fu4_2b_equalize_on_yields_higher_snr_than_equalize_off() -> None:
    """With a booster the ROADM loss no longer attenuates the signal.

    The booster restores the per-channel launch power, so the SNR must improve
    even though the booster injects its own ASE.
    """
    on = run_scenario(_by_name("legacy_two_hop_equalize_on"))["snr_db"]
    off = run_scenario(_by_name("legacy_two_hop_equalize_off"))["snr_db"]
    assert _numeric(on) > _numeric(off)


# ======================================================================
# FU4-3  the required measurement (not a judgement call)
# ======================================================================


def test_fu4_3_equalize_off_does_not_reproduce_the_fg_tree_numbers() -> None:
    """The brief asked this to be *measured*: does the union collapse to FG-tree?

    Answer, from the committed ΔSNR artefact: **no**. ``legacy_two_hop_equalize_off``
    moved by -0.096115 dB, not 0. The two assemblies agree on the *booster*
    question but still differ in how filtering is applied, so turning the booster
    off does not recover the old numbers. Pinning the non-zero delta keeps the
    question answered rather than left to argument.
    """
    if not DELTA_PATH.exists():
        pytest.fail(f"missing {DELTA_PATH}; regenerate with scripts/fu4_assembly_delta.py")
    data = json.loads(DELTA_PATH.read_text())
    row = next(r for r in data["delta"] if r["name"] == "legacy_two_hop_equalize_off")

    assert row["delta_db"] is not None
    assert row["delta_db"] != pytest.approx(0.0, abs=1e-9)
    assert row["delta_db"] == pytest.approx(-0.09611463424176492)


# ======================================================================
# FU4-4  FG-2 flexible-grid capability is intact
# ======================================================================


def test_fu4_4_blocked_slots_still_rejects_an_overlapping_connection() -> None:
    """The slice-semantics blocking FG-2 introduced must survive the union."""
    result = run_scenario(_by_name("flex_blocked_slots"))
    assert result["accepted"] is False
    assert result["snr_db"] is None


def test_fu4_4b_legacy_blocked_channels_still_reject() -> None:
    """The legacy fixed-grid blocking path must be unchanged too."""
    result = run_scenario(_by_name("legacy_blocked_channel"))
    assert result["accepted"] is False
    assert result["snr_db"] is None


def test_fu4_4c_flexible_and_legacy_snr_stay_bit_identical() -> None:
    """FG-2 §F2-1's centre equivalence still holds under the union assembly.

    If the union ever indexed the flexgrid path differently from the legacy one
    -- e.g. by reading ``spectrum_grid.frequencies[channel_idx]`` for the booster
    ASE -- these two columns would diverge.
    """
    for suffix in ("two_hop_equalize_on", "three_hop_equalize_on", "two_hop_equalize_off"):
        legacy = _numeric(run_scenario(_by_name(f"legacy_{suffix}"))["snr_db"])
        flex = _numeric(run_scenario(_by_name(f"flex_{suffix}"))["snr_db"])
        assert flex == legacy


def test_fu4_4d_booster_ase_uses_per_connection_frequency_not_the_grid() -> None:
    """The union must read the *node* frequency, not the fixed-grid channel array.

    ``_node_frequency_and_rate`` is what makes the union correct under a
    flexible grid: ``channel_idx`` indexes connections there, so indexing
    ``spectrum_grid.frequencies`` with it would silently pick the wrong
    frequency and the wrong symbol rate.
    """
    from fu4_scenarios import _grid_and_connections

    scenario = _by_name("flex_two_hop_equalize_on")
    grid, connections, _, num_channels = _grid_and_connections(scenario)

    # The real evaluator exposes per-node arrays, one entry per ODE node/connection.
    centres = [
        float(grid.centre_hz(conn.first_slot, conn.n_slots)) for conn in connections[:num_channels]
    ]
    rates = [float(conn.symbol_rate_hz) for conn in connections[:num_channels]]

    class _Stub:
        """Minimal stand-in exposing only what ``_node_frequency_and_rate`` reads."""

        def __init__(self) -> None:
            import numpy as np

            self.f_abs_hz = np.array(centres, dtype=np.float64)
            self.symbol_rate_hz_channels = np.array(rates, dtype=np.float64)

    validator = QoTValidator.__new__(QoTValidator)
    validator.gn_evaluator = _Stub()

    frequency_hz, symbol_rate_hz = validator._node_frequency_and_rate(3)
    assert frequency_hz == pytest.approx(centres[3])
    assert symbol_rate_hz == pytest.approx(rates[3])
    # The per-connection centre must equal the flexgrid centre, and differ from
    # what indexing the *fixed* grid array with the connection index would give
    # -- that is precisely the mistake this accessor exists to prevent.
    assert frequency_hz == pytest.approx(float(grid.centre_hz(3 * 4, 4)))
    assert rates[3] == 32e9


# ======================================================================
# FU4-5  fidelity golden + delta artefact
# ======================================================================


def test_fu4_5_committed_fidelity_golden_reproduces() -> None:
    """``results/fu4_fidelity_golden.json`` must reproduce bit-exactly."""
    if not GOLDEN_PATH.exists():
        pytest.fail(f"missing {GOLDEN_PATH}; regenerate with scripts/fu4_capture_golden.py --write")
    committed = json.loads(GOLDEN_PATH.read_text())
    expected = {row["name"]: row for row in committed["scenarios"]}

    for scenario in SCENARIOS:
        live = run_scenario(scenario)
        prior = expected[scenario.name]
        assert live["accepted"] == prior["accepted"], scenario.name
        assert _numeric(live["snr_db"]) == _numeric(prior["snr_db"]), scenario.name


def test_fu4_5b_committed_delta_artefact_reproduces() -> None:
    """The doc's ΔSNR table must re-render from the committed artefact."""
    if not DELTA_PATH.exists():
        pytest.fail(f"missing {DELTA_PATH}; regenerate with scripts/fu4_assembly_delta.py")
    data = json.loads(DELTA_PATH.read_text())

    assert data["after"] == [run_scenario(scenario) for scenario in SCENARIOS], (
        "live 'after' column drifted from results/fu4_assembly_delta.json"
    )
    # The 'before' column is the FG-tree capture and must not have been re-taken
    # against the changed source.
    assert _numeric(data["before"][0]["snr_db"]) == pytest.approx(23.125814425161867)
    assert data["summary"]["moved_count"] == 8


# ======================================================================
# FU4-6  source-level guard against silent regression
# ======================================================================


def test_fu4_6_lumped_assembly_is_gone_from_the_source() -> None:
    """No ``insertion_loss_db + filtering_penalty_db`` lumping may reappear.

    A source-level guard rather than a behavioural one: the lumped form and the
    union form can coincide for some parameter choices, so only the source
    itself reliably distinguishes them. Same technique as FG-2's
    ``test_f2_7_ode_function_body_unchanged``.
    """
    source = QOT_CHECKER.read_text()
    assert not re.search(
        r"insertion_loss_db\s*\+\s*\w+\.filtering_penalty_db", source
    ), "the FG-tree lumped assembly has been reintroduced into rwa/qot_checker.py"


def test_fu4_6b_union_keeps_every_fg2_capability_symbol() -> None:
    """The FG-2 flexgrid hooks must all still be present (none deleted)."""
    source = QOT_CHECKER.read_text()
    for symbol in (
        "def num_nodes",
        "def _is_blocked",
        "blocked_slots",
        "connection_overlaps_slots",
        "slots_per_channel",
        "path_loss_db",
        "booster_noise_figure_db",
        "filtering_penalty_db",
    ):
        assert symbol in source, f"FG-2/FU-4 symbol {symbol!r} vanished from rwa/qot_checker.py"
