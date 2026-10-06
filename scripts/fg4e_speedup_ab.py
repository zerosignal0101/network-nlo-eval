r"""FG-4e CPU batch A/B: serial path vs batched path, interleaved in one process.

What this measures
------------------
The **production default** single-span kernel (``ode_accel="matvec"``) run two
ways over the same candidates:

* **serial** — ``B`` independent ``_calc_span_noise_and_power`` calls, i.e. what
  the library does today;
* **batch** — one ``calc_span_noise_and_power_batch`` call on a ``(B, N)`` batch.

Both arms are the *library* paths; there is no script-level prototype on either
side of the comparison. What is measured is the complete kernel (RK4/SRS ODE +
tensorised NLI + ASE), not a GEMM in isolation, because a GEMM-only number would
overstate the win: the per-candidate NLI and ASE work does not shrink with B.

Why same-process interleaving
-----------------------------
This machine is shared and its load average swings (often 5-45). Two runs taken
minutes apart differ by host load, not by the thing under test, so the arms are
interleaved sample-by-sample with the order flipped each round
(``A,B,B,A,A,B,...``) and the reported figure is the ratio of **min** samples.
``loadavg`` is recorded before, during and after every batch size so a contended
reading is visible rather than silently averaged in.

Workload shape
--------------
Per FG-4e brief §1.3 the batch is shaped like the real thing FG-4d measured in
production rollouts: **one base occupancy, one extra channel raised per
candidate**, spread across the band, with an eighth of the channels empty so the
zero-power guards are on the measured path.

Usage::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \
        $HOME/miniconda3/envs/rl/bin/python scripts/fg4e_speedup_ab.py \
        --out results/fg4e_speedup_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import network_nlo_eval  # noqa: E402
from network_nlo_eval.models.batch_kernels import (  # noqa: E402
    MAX_REL_DEVIATION,
    worst_relative_deviation,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN  # noqa: E402

from fg4a_scenarios import (  # noqa: E402
    CENTER_FREQUENCY_HZ,
    CHANNEL_SPACING_HZ,
    NUM_CHANNELS,
    PHI_KURTOSIS_GAUSSIAN,
    SYMBOL_RATE_HZ,
)

SPAN_LENGTH_M = 100e3

#: FG-4e brief §1.3: the production working point from FG-4d, bracketed by the
#: FG-4c knee at B>=4 and by a wide batch.
BATCH_SIZES: tuple[int, ...] = (1, 4, 16, 68, 256)

#: FG-4d's decision inputs, quoted here **only** so the measured numbers can be
#: placed against them. These are NOT measured by this script.
FG4D_BATCH_MEAN = 68.47
FG4D_BATCH_MAX = 142
FG4D_FRAC_GE_4 = 0.633
FG4D_EXTRAPOLATED_LOWER_BOUND = 5.85
FG4D_EXTRAPOLATED_UPPER = 7.20
FG4D_REALISED_PRIOR_KNIFE = 4.678


def read_loadavg() -> list[float]:
    try:
        return [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):  # pragma: no cover
        return [float("nan")] * 3


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def _evaluator() -> MultiBandISRSGN:
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.network.elements import FiberSpanConfig

    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS, channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    return MultiBandISRSGN(
        grid=grid, ref_fiber_config=fiber, symbol_rate_hz=SYMBOL_RATE_HZ,
        phi_kurtosis=PHI_KURTOSIS_GAUSSIAN, ode_accel="matvec",
    )


def candidate_batch(batch: int, n_ch: int = NUM_CHANNELS) -> np.ndarray:
    """``(B, N)`` candidates: one base occupancy, one channel raised per candidate.

    Channel indices are spread across the band (not adjacent) so the batch is not
    accidentally degenerate, and every 7th channel is blank plus both band edges,
    matching the occupancy profile of the FG-4a/FG-4b profiled workload.
    """
    base = np.full((batch, n_ch), 1e-3, dtype=np.float64)
    base[:, ::7] = 0.0
    base[:, 0] = 0.0
    base[:, -1] = 0.0
    step = max(1, n_ch // max(batch, 1))
    for b in range(batch):
        base[b, (b * step) % n_ch] = 1e-3
    return np.ascontiguousarray(base)


def serial_arm(ev: MultiBandISRSGN, profiles: np.ndarray) -> Any:
    """Today's production path: B independent per-candidate calls."""
    for b in range(profiles.shape[0]):
        ev._calc_span_noise_and_power(np.ascontiguousarray(profiles[b]), SPAN_LENGTH_M, None, None, 1)


def batch_arm(ev: MultiBandISRSGN, profiles: np.ndarray) -> Any:
    """The new batched entry point, one call for the whole batch."""
    ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M, None, None, 1)


def interleaved_ab(
    fa: Callable[[], Any], fb: Callable[[], Any], rounds: int, *, warmup: int = 1
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Alternate the two arms round-robin, flipping the order each round."""
    for _ in range(warmup):
        fa()
        fb()
    sa: list[float] = []
    sb: list[float] = []
    for r in range(rounds):
        order = ((sa, fa), (sb, fb)) if r % 2 == 0 else ((sb, fb), (sa, fa))
        for acc, fn in order:
            start = time.perf_counter()
            fn()
            acc.append(time.perf_counter() - start)

    def stat(xs: list[float]) -> dict[str, Any]:
        srt = sorted(xs)
        return {
            "min_s": srt[0],
            "median_s": srt[len(srt) // 2],
            "mean_s": sum(srt) / len(srt),
            "spread_ratio": (srt[-1] / srt[0]) if srt[0] > 0 else float("nan"),
            "rounds": len(srt),
        }

    return stat(sa), stat(sb)


def numeric_check(ev: MultiBandISRSGN, profiles: np.ndarray) -> dict[str, Any]:
    """Per-candidate deviation of the batch arm against the serial arm."""
    ref = np.stack([
        np.stack(ev._calc_span_noise_and_power(np.ascontiguousarray(profiles[b]), SPAN_LENGTH_M, None, None, 1))
        for b in range(profiles.shape[0])
    ])
    bat = np.moveaxis(
        np.stack(ev.calc_span_noise_and_power_batch(profiles, SPAN_LENGTH_M, None, None, 1)), 0, 1
    )
    per_channel = {
        key: worst_relative_deviation(ref[:, k], bat[:, k])
        for k, key in enumerate(("p_out", "spm", "xpm", "ase"))
    }
    worst = max(per_channel.values())
    return {
        "batch": int(profiles.shape[0]),
        "worst_rel": worst,
        "per_channel_rel": per_channel,
        "within_budget": worst <= MAX_REL_DEVIATION,
        "bit_identical": worst == 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "fg4e_speedup_ab.json")
    parser.add_argument("--rounds", type=int, default=12, help="Round cap per arm.")
    parser.add_argument(
        "--target-seconds", type=float, default=2.0,
        help="Per-arm sampling budget; the round count adapts down to respect it.",
    )
    parser.add_argument("--batches", type=int, nargs="*", default=list(BATCH_SIZES))
    args = parser.parse_args()

    print(f"package: {Path(network_nlo_eval.__file__).resolve()}")
    load_before = read_loadavg()
    ev = _evaluator()

    # Scenario-level warmup: JIT cache load and BLAS init are one-offs and run an
    # order of magnitude above the steady-state cost. They must not land in a sample.
    warm = candidate_batch(1)
    serial_arm(ev, warm)
    batch_arm(ev, warm)

    rows: list[dict[str, Any]] = []
    numeric: list[dict[str, Any]] = []

    for b in args.batches:
        profiles = candidate_batch(b)
        # Numeric check first, on the same scenario, so the timed curve cannot be
        # quoting numbers for a different computation.
        check = numeric_check(ev, profiles)
        numeric.append(check)
        print(
            f"  numeric B={b:4d}: worst per-candidate rel {check['worst_rel']:.3e} "
            f"({'bit-identical' if check['bit_identical'] else 'within' if check['within_budget'] else 'OVER BUDGET'})",
            flush=True,
        )

        t0 = time.perf_counter(); serial_arm(ev, profiles); serial_unit = time.perf_counter() - t0
        t0 = time.perf_counter(); batch_arm(ev, profiles); batch_unit = time.perf_counter() - t0
        rounds = max(1, min(args.rounds, int(args.target_seconds / max(serial_unit, 1e-9))))

        load_during_before = read_loadavg()
        serial_row, batch_row = interleaved_ab(
            lambda: serial_arm(ev, profiles), lambda: batch_arm(ev, profiles), rounds=rounds, warmup=0
        )
        load_during = read_loadavg()

        speedup = serial_row["min_s"] / batch_row["min_s"]
        row = {
            "batch": b,
            "serial": serial_row,
            "batched": batch_row,
            "speedup_min_sample": speedup,
            "speedup_median_sample": serial_row["median_s"] / batch_row["median_s"],
            "serial_per_candidate_s": serial_row["min_s"] / b,
            "batched_per_candidate_s": batch_row["min_s"] / b,
            "per_candidate_speedup": serial_row["min_s"] / batch_row["min_s"],
            "rounds_run": rounds,
            "loadavg_during": load_during,
            "loadavg_delta_1min": load_during[0] - load_during_before[0],
        }
        rows.append(row)
        print(
            f"  B={b:4d}: serial {serial_row['min_s'] * 1e3:9.3f} ms "
            f"({row['serial_per_candidate_s'] * 1e6:8.2f} us/cand) | "
            f"batch {batch_row['min_s'] * 1e3:8.3f} ms "
            f"({row['batched_per_candidate_s'] * 1e6:8.2f} us/cand) | "
            f"speedup {speedup:6.3f}x | rounds {rounds} | loadavg1 {load_during[0]:.2f}",
            flush=True,
        )

    load_after = read_loadavg()
    by_batch = {r["batch"]: r for r in rows}
    worst_numeric = max((c["worst_rel"] for c in numeric), default=float("nan"))
    b68 = by_batch.get(68)

    payload = {
        "schema_version": 1,
        "method": (
            "same-process interleaved serial-vs-batch (order flipped each round), min-sample "
            "ratio; complete single-span kernel (RK4/SRS ODE + tensorised NLI + ASE) on both "
            "arms; both arms are library paths"
        ),
        "provenance": {
            "git_rev": _git_rev(),
            "package_path": str(Path(network_nlo_eval.__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
        "config": {
            "n_ch": NUM_CHANNELS,
            "channel_spacing_hz": CHANNEL_SPACING_HZ,
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "span_length_m": SPAN_LENGTH_M,
            "phi_kurtosis": PHI_KURTOSIS_GAUSSIAN,
            "ode_accel": "matvec",
            "batches": args.batches,
            "rounds_cap": args.rounds,
            "target_seconds": args.target_seconds,
            "max_rel_deviation_budget": MAX_REL_DEVIATION,
        },
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "loadavg_delta_1min": load_after[0] - load_before[0],
        "numeric_check": numeric,
        "worst_rel_deviation_overall": worst_numeric,
        "rows": rows,
        "fg4d_reference": {
            "note": "FG-4d decision inputs, quoted for comparison only; NOT measured here.",
            "real_batch_mean": FG4D_BATCH_MEAN,
            "real_batch_max": FG4D_BATCH_MAX,
            "frac_ge_4": FG4D_FRAC_GE_4,
            "extrapolated_speedup_lower_bound": FG4D_EXTRAPOLATED_LOWER_BOUND,
            "extrapolated_speedup_upper": FG4D_EXTRAPOLATED_UPPER,
            "realised_prior_knife": FG4D_REALISED_PRIOR_KNIFE,
        },
        "b68_measured_speedup": (b68["speedup_min_sample"] if b68 else None),
        "b68_vs_fg4d_lower_bound_ratio": (
            (b68["speedup_min_sample"] / FG4D_EXTRAPOLATED_LOWER_BOUND) if b68 else None
        ),
        "b1_measured_speedup": (by_batch[1]["speedup_min_sample"] if 1 in by_batch else None),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out}")
    print(f"loadavg: {load_before} -> {load_after}")
    if b68 is not None:
        measured = b68["speedup_min_sample"]
        print(
            f"B=68 measured {measured:.3f}x vs FG-4d extrapolated lower bound "
            f"{FG4D_EXTRAPOLATED_LOWER_BOUND:.2f}x "
            f"({measured / FG4D_EXTRAPOLATED_LOWER_BOUND:.3f}x of the bound)"
        )
    if 1 in by_batch:
        print(f"B=1  measured {by_batch[1]['speedup_min_sample']:.3f}x (below 1.0 = batch is slower)")
    print(f"worst per-candidate relative deviation: {worst_numeric:.3e} (budget {MAX_REL_DEVIATION:.0e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
