"""FU-4 SNR-assembly A/B harness: measures how much the union assembly moves SNR.

What this is
------------
FU-4 adjudicates *two* SNR assemblies that both live in this repository's
history and both claim to model the same physics:

* **FG-tree** (``main``-style, what ``rwa/qot_checker.py`` did before FU-4):
  a single lumped ``10 ** ((insertion_loss_db + filtering_penalty_db) / 10)``
  applied to signal **and** accumulated noise, no booster.
* **ecz-dev / RL** style: ``path_loss_db("drop"|"express")``, plus booster ASE
  when ``equalize_output_power`` is true, plus ``filtering_penalty_db`` as an
  independent **noise-only** multiplier.

The adjudication and its evidence live in ``docs/fu4_qot_assembly_adjudication.md``.
This module is the *numeric* half: it re-renders the per-scenario ``ΔSNR`` table
so the doc's numbers are recomputable rather than transcribed.

Two-phase protocol (why it exists)
----------------------------------
Both sides of the A/B must come from **real code runs**, not from a
reimplementation of one side. So:

    # 1. BEFORE the source change, capture the FG-tree numbers:
    PYTHONPATH=$PWD/src python scripts/fu4_assembly_delta.py --phase capture

    # 2. AFTER the source change, recompute and write the delta table:
    PYTHONPATH=$PWD/src python scripts/fu4_assembly_delta.py --phase compare

    # 3. Verify a committed artefact without rewriting it:
    PYTHONPATH=$PWD/src python scripts/fu4_assembly_delta.py --check

``--check`` re-runs every scenario and asserts the live values still equal the
values recorded in the JSON; it never writes. That is what makes the doc's
table re-renderable rather than hand-copied.

Scenario coverage
-----------------
Deliberately spans the axes that could make the union behave differently from
the FG tree, rather than just re-running one happy path:

``legacy_*``   fixed-grid (``SpectrumGrid``, ``connections=None``) paths, varying
               hop count, ``equalize_output_power``, express attenuation, and
               a fully blocked channel.
``flex_*``     flexible-grid (``connections=[...]``) paths, i.e. exactly the
               FG-2 capability that the brief forbids losing, with
               ``blocked_slots`` (the authoritative flexgrid blocking field) and
               with the legacy ``blocked_channels`` field for compatibility.

``legacy_two_hop_equalize_off`` is the one that answers a *required
measurement* rather than a judgement call: the brief asks whether the union
assembly collapses to the FG-tree result when ``equalize_output_power`` is
false. It does not, and this scenario is what demonstrates it -- see §1.2 of
the doc.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Scenario definitions live in tests/, same convention as tests/fg1_scenarios.py.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fu4_scenarios import SCENARIOS, _cell, _numeric, run_scenario  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "fu4_assembly_delta.json"

def build_delta(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-scenario before/after/delta, keyed by scenario name.

    ``None`` means the scenario was *rejected* by ``verify_allocation`` (a
    blocked channel fails the SNR gate, so no array is returned). That is a
    legitimate outcome, not a missing measurement, so it is carried through as
    ``None`` and never coerced into a number.
    """
    before_by_name = {row["scenario"]["name"]: row for row in before}
    rows: list[dict[str, Any]] = []
    for row in after:
        name = row["scenario"]["name"]
        prior = before_by_name.get(name)
        if prior is None:
            raise KeyError(f"no 'before' row captured for scenario {name!r}")
        b = _numeric(prior["snr_db"])
        a = _numeric(row["snr_db"])
        both_finite = b is not None and a is not None and math.isfinite(b) and math.isfinite(a)
        rows.append(
            {
                "name": name,
                "kind": row["scenario"]["kind"],
                "description": row["scenario"]["description"],
                "before_snr_db": prior["snr_db"],
                "after_snr_db": row["snr_db"],
                "delta_db": (a - b) if both_finite else None,
                "moved": bool(a != b),
            }
        )
    return rows


def render_delta_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table for the doc, rendered from the JSON (never hand-copied)."""
    lines = [
        "| 场景 | 类型 | before (dB) | after (dB) | ΔSNR (dB) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        delta = row["delta_db"]
        delta_text = "n/a（两端都被阻断）" if delta is None else f"{delta:+.6f}"
        lines.append(
            f"| `{row['name']}` | {row['kind']} | {_cell(_numeric(row['before_snr_db']))} | "
            f"{_cell(_numeric(row['after_snr_db']))} | {delta_text} |"
        )
    return "\n".join(lines)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the delta table into the headline numbers."""
    finite = [row["delta_db"] for row in rows if row["delta_db"] is not None]
    return {
        "scenario_count": len(rows),
        "moved_count": sum(1 for row in rows if row["moved"]),
        "rejected_both_ends": sum(
            1 for row in rows if row["before_snr_db"] is None and row["after_snr_db"] is None
        ),
        "max_abs_delta_db": max((abs(d) for d in finite), default=0.0),
        "min_delta_db": min(finite, default=0.0),
        "max_delta_db": max(finite, default=0.0),
    }


# ======================================================================
# CLI
# ======================================================================


def _load_before(path: Path) -> list[dict[str, Any]] | None:
    """Read the immutable 'before' column, tolerating the older artefact shape."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if "before" in data:
        return data["before"]
    # Migration path: the first `--phase compare` of this WP wrote the captured
    # FG-tree column under `before_scenarios`. Read it from there so the
    # original capture is never re-taken against already-changed source.
    return data.get("before_scenarios")


def main(argv: list[str] | None = None) -> int:
    """Run the requested phase and return a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--phase",
        choices=["capture", "compare"],
        default="compare",
        help="'capture' records the pre-change 'before' column; 'compare' recomputes live.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="artefact path")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run and assert the committed artefact still matches; never writes",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="allow --phase capture to overwrite an existing 'before' column",
    )
    args = parser.parse_args(argv)

    if args.check and args.phase != "compare":
        print("error: --check only makes sense with --phase compare", file=sys.stderr)
        return 2

    # Prove we are measuring THIS worktree, per the cross-WP rules.
    import network_nlo_eval

    package_dir = str(Path(network_nlo_eval.__file__).resolve().parent)
    if not package_dir.startswith(str(REPO_ROOT)):
        print(
            f"IMPORT_FAIL: imported {package_dir}, which is not under {REPO_ROOT}. "
            "Set PYTHONPATH=$PWD/src explicitly.",
            file=sys.stderr,
        )
        return 3

    after = [run_scenario(scenario) for scenario in SCENARIOS]

    if args.phase == "capture":
        if args.out.exists() and not args.force:
            print(
                f"refusing to overwrite {args.out}: a 'before' column already exists.\n"
                "It must be captured from the FG-tree source exactly once; re-capturing it "
                "after the change would silently relabel the union numbers as the baseline. "
                "Use --force only if you really mean to discard the original capture.",
                file=sys.stderr,
            )
            return 2
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "schema": "fu4_assembly_delta/v2",
                    "assembly": {
                        "before": "FG-tree lumped insertion_loss_db + filtering_penalty_db on signal and noise",
                        "after": "union: port loss + booster ASE + noise-only filtering_penalty_db",
                    },
                    "package_dir": package_dir,
                    "before": after,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        print(f"captured {len(after)} scenarios as 'before' -> {args.out}")
        return 0

    before = _load_before(args.out)
    if before is None:
        print(
            f"error: no 'before' column in {args.out}. Run with --phase capture BEFORE the "
            "source change to record the FG-tree baseline.",
            file=sys.stderr,
        )
        return 2

    rows = build_delta(before, after)
    artefact = {
        "schema": "fu4_assembly_delta/v2",
        "assembly": {
            "before": "FG-tree lumped insertion_loss_db + filtering_penalty_db on signal and noise",
            "after": "union: port loss + booster ASE + noise-only filtering_penalty_db",
        },
        "package_dir": package_dir,
        "before": before,
        "after": after,
        "delta": rows,
        "summary": summarise(rows),
    }

    if args.check:
        # The committed `after` column must reproduce exactly; that is what makes
        # the doc's table re-renderable rather than hand-copied.
        committed = json.loads(args.out.read_text())
        committed_after = {
            row["scenario"]["name"]: row["snr_db"] for row in committed.get("after", [])
        }
        mismatches: list[str] = []
        for live in after:
            name = live["scenario"]["name"]
            if name not in committed_after:
                mismatches.append(f"{name}: absent from committed 'after' column")
                continue
            if committed_after[name] != live["snr_db"]:
                mismatches.append(
                    f"{name}: live {live['snr_db']!r} != committed {committed_after[name]!r}"
                )

        print(render_delta_table(rows))
        print()
        print(json.dumps(artefact["summary"], indent=2))
        print()
        print(f"check: {len(after)} scenarios re-rendered, {len(mismatches)} mismatched vs artefact")
        for problem in mismatches:
            print(f"  MISMATCH {problem}")
        return 1 if mismatches else 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artefact, indent=2, ensure_ascii=False) + "\n")
    print(render_delta_table(rows))
    print()
    print(json.dumps(artefact["summary"], indent=2))
    print()
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
