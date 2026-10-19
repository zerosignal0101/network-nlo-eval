"""Emit the FG-2 acceptance evidence as JSON (design §5, F2-1 … F2-8).

Every number in the FG-2 report must be recomputable from a committed product.
This script writes ``results/fg2_evidence.json`` containing the measured value
of each acceptance criterion, so the report can be re-derived by re-running it::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-nb-impl \\
        $HOME/miniconda3/envs/rl/bin/python scripts/fg2_evidence.py \\
        --out results/fg2_evidence.json

The per-criterion assertions themselves live in ``tests/test_fg2_flexible_grid.py``;
this file records *what the numbers are*, not just whether they hold.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

from network_nlo_eval.core.flexgrid import (
    ConnectionSpec,
    FlexibleGrid,
    bits_per_symbol,
    slots_for_rate,
)
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.numba_kernels import _calc_exact_power_profile_ode
from network_nlo_eval.network.elements import FiberSpanConfig
from network_nlo_eval.network.slots import migrate_roadm_blocked_slots

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "results" / "fg2_golden_cplusl_230x50.npz"

NUM_CHANNELS = 230
CHANNEL_SPACING_HZ = 50e9
SLOT_WIDTH_HZ = 12.5e9
CENTER_FREQUENCY_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9
SPAN_LENGTH_M = 100e3
LAUNCH_POWER_W = 1e-3

_CRAMMER = 0.028 / 1e3 / 1e12


def _raman_gain(delta_f_thz: np.ndarray) -> np.ndarray:
    d = np.abs(delta_f_thz)
    return np.where(
        d <= 13.2,
        _CRAMMER * d * 1e12,
        np.where(d < 15.6, (_CRAMMER * 13.2e12) * (15.6 - d) / 2.4, 0.0),
    )


def _band_overlap_gain(dfc: float, bandwidth_thz: float, n: int = 400001) -> float:
    total = 2 * bandwidth_thz
    w = np.linspace(-total / 2, total / 2, n)
    weight = np.maximum(1 - 2 * np.abs(w) / total, 0.0)
    weight /= np.trapezoid(weight, w)
    return float(np.trapezoid(_raman_gain(dfc + w) * weight, w))


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> int:
    """Compute every FG-2 acceptance number and write it to disk."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "fg2_evidence.json")
    args = parser.parse_args()

    legacy = SpectrumGrid(NUM_CHANNELS, CHANNEL_SPACING_HZ, CENTER_FREQUENCY_HZ)
    grid = FlexibleGrid.from_legacy_grid(NUM_CHANNELS, CHANNEL_SPACING_HZ, CENTER_FREQUENCY_HZ, 4)
    conns = [
        ConnectionSpec(
            grid=grid,
            connection_id=i,
            first_slot=i * 4,
            n_slots=4,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=SYMBOL_RATE_HZ,
            roll_off=0.0,
            launch_power_w=LAUNCH_POWER_W,
        )
        for i in range(NUM_CHANNELS)
    ]
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    ev = MultiBandISRSGN(grid=grid, ref_fiber_config=fiber, connections=conns, slots_per_channel=4)

    evidence: dict = {
        "meta": {
            "git_rev": _git("rev-parse", "HEAD"),
            "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "loadavg": open("/proc/loadavg").read().strip(),
            "golden_git_rev": None,
        },
        "parameters": {
            "num_channels": NUM_CHANNELS,
            "channel_spacing_hz": CHANNEL_SPACING_HZ,
            "slot_width_hz": SLOT_WIDTH_HZ,
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "num_slots": grid.num_slots,
            "span_length_m": SPAN_LENGTH_M,
            "launch_power_w": LAUNCH_POWER_W,
        },
    }

    # --- F2-1 -----------------------------------------------------------
    centres = np.array([grid.centre_hz(i * 4, 4) for i in range(NUM_CHANNELS)])
    legacy_freq = legacy.frequencies
    evidence["F2-1_frequency_bit_identity"] = {
        "raw_equality_all": bool(np.all(centres == legacy_freq)),
        "n_mismatched_elements": int(np.count_nonzero(centres != legacy_freq)),
        "max_abs_diff_hz": float(np.max(np.abs(centres - legacy_freq))),
        "band_center_equals_legacy_center": bool(grid.band_center_hz == CENTER_FREQUENCY_HZ),
        "note": "raw float64 ==, not allclose",
    }

    # --- F2-2 -----------------------------------------------------------
    starts = np.arange(NUM_CHANNELS) * 4
    n_slots_arr = np.full(NUM_CHANNELS, 4.0)
    wrong = grid.band_low_edge_hz + (starts + (n_slots_arr - 1) / 2.0) * SLOT_WIDTH_HZ
    evidence["F2-2_half_slot_trap"] = {
        "wrong_formula_offset_ghz": float(np.max(np.abs(wrong - legacy_freq))) / 1e9,
        "wrong_formula_reproduces_legacy": bool(np.all(wrong == legacy_freq)),
        "correct_formula_reproduces_legacy": bool(
            np.all(grid.band_low_edge_hz + (starts + n_slots_arr / 2.0) * SLOT_WIDTH_HZ == legacy_freq)
        ),
    }

    # --- F2-3 -----------------------------------------------------------
    if not GOLDEN_PATH.exists():
        raise SystemExit(f"missing golden file {GOLDEN_PATH}")
    with np.load(GOLDEN_PATH) as handle:
        golden = {key: handle[key] for key in handle.files}
    evidence["meta"]["golden_git_rev"] = str(golden["meta__git_rev"])
    p_out, s2_spm, s2_xpm, s2_ase = ev._calc_span_noise_and_power(
        np.full(NUM_CHANNELS, LAUNCH_POWER_W), SPAN_LENGTH_M, None
    )
    evidence["F2-3_qot_bit_identity"] = {
        "n_nodes": ev.n_nodes,
        "f_abs_array_equal_golden": bool(np.array_equal(ev.f_abs_hz, golden["in__f_abs_hz"])),
        "f_rel_array_equal_golden": bool(np.array_equal(ev.f_rel_hz, golden["in__f_rel_hz"])),
        "symbol_rate_is_array": isinstance(ev.symbol_rate_hz_channels, np.ndarray),
    }
    for name, got in (
        ("p_out_w", p_out),
        ("sigma2_spm_w", s2_spm),
        ("sigma2_xpm_w", s2_xpm),
        ("sigma2_ase_w", s2_ase),
    ):
        ref = golden[name]
        evidence["F2-3_qot_bit_identity"][name] = {
            "array_equal": bool(np.array_equal(got, ref)),
            "max_abs_diff_w": float(np.max(np.abs(got - ref))),
        }

    # --- F2-4 -----------------------------------------------------------
    cases = [(100e9, "QPSK", 0.0), (400e9, "16QAM", 0.0), (200e9, "64QAM", 0.0), (100e9, "QPSK", 0.1)]
    got = [slots_for_rate(r, bits_per_symbol(f), ro, SLOT_WIDTH_HZ) for r, f, ro in cases]
    evidence["F2-4_slots_for_rate"] = {
        "cases": [
            {
                "bit_rate_gbps": r / 1e9,
                "modulation": f,
                "roll_off": ro,
                "n_slots": n,
                "occupied_ghz": n * SLOT_WIDTH_HZ / 1e9,
            }
            for (r, f, ro), n in zip(cases, got, strict=True)
        ],
        "result": got,
        "expected": [4, 8, 3, 5],
        "matches": got == [4, 8, 3, 5],
    }

    # --- F2-5 -----------------------------------------------------------
    shape_errors: dict[str, str] = {}
    for label, kwargs in (
        ("power_in_w", {"power": np.full(NUM_CHANNELS - 1, LAUNCH_POWER_W)}),
        ("nf_lin_channels", {"nf": np.full(NUM_CHANNELS + 5, 5.0)}),
    ):
        try:
            if label == "power_in_w":
                ev._calc_span_noise_and_power(kwargs["power"], SPAN_LENGTH_M, None)
            else:
                from network_nlo_eval.network.elements import EDFAConfig

                ev._calc_span_noise_and_power(
                    np.full(NUM_CHANNELS, LAUNCH_POWER_W), SPAN_LENGTH_M, EDFAConfig(noise_figure_db=kwargs["nf"])
                )
        except ValueError as exc:
            message = str(exc)
            shape_errors[label] = {
                "message": message,
                "mentions_expected_shape": f"({NUM_CHANNELS},)" in message,
                "mentions_actual_shape": (
                    str((NUM_CHANNELS - 1,)) in message if label == "power_in_w" else str((NUM_CHANNELS + 5,)) in message
                ),
            }
    evidence["F2-5_shape_contract"] = shape_errors

    # --- F2-6 -----------------------------------------------------------
    span_thz = 11.45
    srs: dict[str, float] = {}
    for width_ghz in (50, 100, 200, 400, 800, 1600):
        bandwidth = width_ghz / 1e3
        dfcs = np.linspace(bandwidth, span_thz - bandwidth, 200)
        worst = 0.0
        for dfc in dfcs:
            centre = float(_raman_gain(np.array([dfc]))[0])
            if centre <= 0:
                continue
            worst = max(worst, abs(_band_overlap_gain(float(dfc), bandwidth) - centre) / centre)
        srs[str(width_ghz)] = worst
    evidence["F2-6_srs_centre_sampling_cplusl"] = {
        "band_span_thz": span_thz,
        "max_relative_error_by_width_ghz": srs,
        "threshold": 1e-12,
        "all_below_threshold": all(value < 1e-12 for value in srs.values()),
        "worst_overall": max(srs.values()),
    }

    # --- F2-7 -----------------------------------------------------------
    kernels_rel = "src/network_nlo_eval/models/numba_kernels.py"
    current = (REPO_ROOT / kernels_rel).read_text(encoding="utf-8")
    try:
        baseline = subprocess.run(
            ["git", "show", f"origin/main:{kernels_rel}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
        hunks = [
            line
            for line in _git("diff", "origin/main", "-U0", "--", kernels_rel).splitlines()
            if line.startswith("@@")
        ]
        added_lines = sorted({int(m) for h in hunks for m in __import__("re").findall(r"\+(\d+)", h)})
        ode_span = (173, 214)
        evidence["F2-7_ode_unchanged"] = {
            "ode_line_span_in_current_file": list(ode_span),
            "first_added_line_vs_origin_main": added_lines[0] if added_lines else None,
            "any_added_line_inside_ode_span": any(ode_span[0] <= n <= ode_span[1] for n in added_lines),
            "kernels_file_sha256": __import__("hashlib").sha256(current.encode()).hexdigest(),
            "note": "call-site f_abs_hz now carries connection centres; ODE body untouched",
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        evidence["F2-7_ode_unchanged"] = {"error": str(exc)}

    # Appendix A6 style cross-check: the RK4 kernel's coupling matches Cr*df.
    ode_checks = []
    for df_thz in (1.0, 6.0, 11.45):
        freq = np.array([190.0e12, (190.0 + df_thz) * 1e12])
        power = np.array([1e-3, 1e-3])
        out = _calc_exact_power_profile_ode(power, freq, np.zeros(2), 1.0, _CRAMMER, n_steps=20)
        g_ode = (out[0] / power[0] - 1.0) / (power[1] * 1.0)
        expected = _CRAMMER * df_thz * 1e12
        ode_checks.append(
            {"df_thz": df_thz, "g_ode": float(g_ode), "cr_times_df": float(expected), "rel": float(abs(g_ode - expected) / expected)}
        )
    evidence["ode_coupling_transcription_check"] = ode_checks

    # --- F2-8 -----------------------------------------------------------
    test_diff = _git("diff", "--name-status", "8f55e14", "--", "tests/")
    modified = [ln for ln in test_diff.splitlines() if ln and not ln.startswith("A")]
    evidence["F2-8_no_test_expectation_changes"] = {
        "baseline_commit": "8f55e14",
        "modified_or_deleted_test_files": modified,
        "zero_expectation_edits": not modified,
    }

    # --- §2.6 migration -------------------------------------------------
    graph = {
        "nodes": [
            {"id": 0, "roadm": {"blocked_channels": [3, 17]}},
            {"id": 1, "roadm": {"blocked_channels": [0]}},
        ],
        "edges": [],
    }
    once, first = migrate_roadm_blocked_slots(graph, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)
    twice, second = migrate_roadm_blocked_slots(once, SLOT_WIDTH_HZ, CHANNEL_SPACING_HZ)
    ratios = [
        len(n["roadm"]["blocked_slots"]) == 4 * len(n["roadm"]["blocked_channels"]) for n in once["nodes"]
    ]
    try:
        migrate_roadm_blocked_slots(
            {"nodes": [{"id": 0, "roadm": {"blocked_channels": [999]}}], "edges": [], "num_slots": grid.num_slots},
            SLOT_WIDTH_HZ,
            CHANNEL_SPACING_HZ,
        )
        out_of_bounds_raises = False
    except ValueError:
        out_of_bounds_raises = True
    evidence["migration_blocked_slots"] = {
        "first_pass": first,
        "second_pass": second,
        "idempotent": second["nodes_migrated"] == 0,
        "four_slots_per_channel_all_nodes": all(ratios),
        "out_of_bounds_raises": out_of_bounds_raises,
        "legacy_field_preserved": all("blocked_channels" in n["roadm"] for n in once["nodes"]),
        "input_graph_not_mutated": "blocked_slots" not in graph["nodes"][0]["roadm"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    print(json.dumps(evidence, indent=2, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
