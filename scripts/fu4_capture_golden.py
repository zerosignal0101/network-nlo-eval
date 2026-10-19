"""FU-4 fidelity-tier golden: capture/verify the union SNR assembly.

Why a *new* golden, and why the legacy one cannot be reused
----------------------------------------------------------
``results/fg4a_golden.json`` (the 7-scenario x 31-output legacy golden) is
**not** affected by FU-4, and that is a substantive fact rather than a
convenience. It is captured through ``MultiBandISRSGN.evaluate_path``, which
carries its own ROADM assembly inside ``models/isrs_gn.py``. FU-4 changes
``rwa/qot_checker.py``, which that path never calls. Hence:

* the legacy golden stays 31/31 bit-identical (verified: full suite still
  248 passed / 1 skipped after the change);
* but that also means **no pre-existing bit-exact gate covers
  ``rwa/qot_checker.py`` at all**. Before FU-4 the RWA validator's numbers
  were unpinned, which is precisely the gap that let FG-4g find a 0.2878 dB
  disagreement with the RL side without any test going red.

So FU-4 introduces the fidelity-tier golden that the RWA path always needed.
It is a **new** gate over **new** expectations. Per the brief and the G2
discipline, no existing ``tests/*`` expectation is modified to accommodate it.

Usage
-----
    # capture (already committed; regenerate only deliberately)
    PYTHONPATH=$PWD/src python scripts/fu4_capture_golden.py --write

    # verify the committed artefact reproduces, without writing
    PYTHONPATH=$PWD/src python scripts/fu4_capture_golden.py --check
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
# Scenario definitions live in tests/, same convention as tests/fg1_scenarios.py.
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fu4_scenarios import SCENARIOS, _numeric, run_scenario  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "fu4_fidelity_golden.json"


def capture() -> dict[str, Any]:
    """Run every scenario and record the union-assembly SNR."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        result = run_scenario(scenario)
        rows.append(
            {
                "name": scenario.name,
                "kind": scenario.kind,
                "accepted": result["accepted"],
                "snr_db": result["snr_db"],
            }
        )
    return {
        "schema": "fu4_fidelity_golden/v1",
        "description": (
            "Fidelity-tier golden for QoTValidator SNR assembly after FU-4 "
            "(union: port loss + booster ASE + noise-only filtering penalty)."
        ),
        "scenarios": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Capture or verify the fidelity golden and return an exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true", help="capture and write the golden")
    group.add_argument("--check", action="store_true", help="verify the committed golden")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    import network_nlo_eval

    package_dir = str(Path(network_nlo_eval.__file__).resolve().parent)
    if not package_dir.startswith(str(REPO_ROOT)):
        print(
            f"IMPORT_FAIL: imported {package_dir}, not under {REPO_ROOT}. "
            "Set PYTHONPATH=$PWD/src explicitly.",
            file=sys.stderr,
        )
        return 3

    live = capture()
    print(f"import-ok: {package_dir}")
    print(f"load: {Path('/proc/loadavg').read_text().strip()}")

    if args.write:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(live, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")
        return 0

    if not args.out.exists():
        print(f"error: {args.out} missing; run with --write", file=sys.stderr)
        return 2

    committed = json.loads(args.out.read_text())
    expected = {row["name"]: row for row in committed["scenarios"]}
    problems: list[str] = []
    for row in live["scenarios"]:
        prior = expected.get(row["name"])
        if prior is None:
            problems.append(f"{row['name']}: missing from committed golden")
            continue
        if prior["accepted"] != row["accepted"]:
            problems.append(
                f"{row['name']}: acceptance {row['accepted']} != committed {prior['accepted']}"
            )
        if _numeric(prior["snr_db"]) != _numeric(row["snr_db"]):
            problems.append(f"{row['name']}: {_numeric(row['snr_db'])} != committed {_numeric(prior['snr_db'])}")

    print(f"{len(live['scenarios'])} scenarios checked against {args.out}")
    for row in live["scenarios"]:
        value = _numeric(row["snr_db"])
        shown = "rejected" if value is None else f"{value:.6f}"
        print(f"  {row['name']:32s} {shown}")
    print(f"load: {Path('/proc/loadavg').read_text().strip()}")
    if problems:
        print(f"FAILED: {len(problems)} mismatch(es)", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("OK: fidelity golden reproduces bit-exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
