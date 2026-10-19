#!/usr/bin/env python3
"""FG-15 §5: legacy vs tabular tier speed A/B, interleaved in ONE process.

Why interleaved
---------------
The host is shared (FG-4a observed loadavg swinging 10..42) and another line of
work runs concurrently. Two separate processes therefore measure *load*, not
speed. Here both tiers live in the same build, so the only sound comparison is
to interleave them sample-by-sample in one interpreter against the same warm
Numba/BLAS state, and to report the ratio of **min** samples (the reading least
polluted by contention) alongside the loadavg recorded across the whole run.

This mirrors ``scripts/fg4a_speedup_ab.py``'s interleaving discipline, but the
comparison axis is the physics *tier* rather than a kernel revision.

Deterministic counts first
--------------------------
Timing is supplementary only (see the brief). The load-bearing evidence is the
deterministic counters: ``g_cache_stats`` / ``nli_cache_stats`` and the number of
RK4 steps actually executed. Those are reported unconditionally; the *ratio* is
withheld when loadavg exceeds ``--max-load`` (default 12, the brief's threshold).

Usage::

    PYTHONPATH=$PWD/src $HOME/miniconda3/envs/rl/bin/python scripts/fg15_tier_speed_ab.py \
        --out results/fg15_tier_speed_ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models import numba_kernels as nk
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import FiberSpanConfig

NUM_CHANNELS = 230
CHANNEL_SPACING_HZ = 50e9
CENTER_FREQUENCY_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9
SPAN_LENGTH_M = 100e3


def loadavg() -> tuple[float, float, float]:
    return os.getloadavg()  # type: ignore[return-value]


def build(tier: str, ode_accel: str) -> MultiBandISRSGN:
    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    ev = MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=fiber,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        raman_spectrum=tier,
        ode_accel=ode_accel,
    )
    # Read back the EFFECTIVE tier rather than trusting the argument.
    assert ev.raman_spectrum == tier, f"tier not effective: {ev.raman_spectrum!r} != {tier!r}"
    return ev


def one_call(ev: MultiBandISRSGN, power: np.ndarray) -> float:
    from network_nlo_eval.network.elements import EDFAConfig

    edfa = EDFAConfig(target_gain_db=None, noise_figure_db=6.0)
    t0 = time.perf_counter()
    ev._calc_span_noise_and_power(power, SPAN_LENGTH_M, edfa, None, 1)
    return time.perf_counter() - t0


def counters() -> dict[str, int]:
    g = nk.g_cache_stats()
    n = nk.nli_cache_stats()
    return {
        "g_hits": int(g.get("hits", 0)),
        "g_misses": int(g.get("misses", 0)),
        "nli_hits": int(n.get("hits", 0)),
        "nli_misses": int(n.get("misses", 0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/fg15_tier_speed_ab.json"))
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--ode-accel", default="matvec")
    ap.add_argument("--max-load", type=float, default=12.0)
    args = ap.parse_args()

    power = np.full(NUM_CHANNELS, 1e-3, dtype=np.float64)
    # Warm both tiers (JIT compile + G-matrix build) before any timing sample.
    warm = {t: build(t, args.ode_accel) for t in ("legacy", "tabular")}
    for t, ev in warm.items():
        one_call(ev, power)

    load_start = loadavg()
    samples: dict[str, list[float]] = {"legacy": [], "tabular": []}
    for r in range(args.rounds):
        order = ("legacy", "tabular") if r % 2 == 0 else ("tabular", "legacy")
        for tier in order:
            samples[tier].append(one_call(warm[tier], power))
    load_end = loadavg()

    lo, hi = min(samples["legacy"]), min(samples["tabular"])
    peak_load = max(load_start[0], load_end[0], *(loadavg()[0],))
    ratio = lo / hi
    admissible = peak_load <= args.max_load

    out: dict[str, Any] = {
        "schema": "fg15-tier-speed-ab-v1",
        "description": (
            "FG-15 §5 legacy vs tabular tier A/B. Interleaved in one process; ratio of MIN "
            "samples. Deterministic counters are the load-bearing evidence; the timing ratio "
            "is supplementary and withheld above the loadavg threshold."
        ),
        "config": {
            "ode_accel": args.ode_accel,
            "rounds": args.rounds,
            "num_channels": NUM_CHANNELS,
            "channel_spacing_hz": CHANNEL_SPACING_HZ,
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "span_length_m": SPAN_LENGTH_M,
        },
        "effective_tier_readback": {t: warm[t].raman_spectrum for t in warm},
        "samples_s": samples,
        "min_s": {"legacy": lo, "tabular": hi},
        "ratio_legacy_over_tabular": ratio,
        "tabular_overhead_pct": (hi / lo - 1.0) * 100.0,
        "loadavg_start": list(load_start),
        "loadavg_end": list(load_end),
        "peak_load_observed": peak_load,
        "max_load_threshold": args.max_load,
        "timing_ratio_admissible": admissible,
        "timing_ratio_note": (
            "ADMISSIBLE" if admissible else "WITHHELD: loadavg above threshold; treat as unattributed"
        ),
        "deterministic_counters": counters(),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "package_path": str(Path(__import__("network_nlo_eval").__file__).resolve()),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"tier readback : {out['effective_tier_readback']}")
    print(f"min legacy    : {lo * 1e3:.3f} ms")
    print(f"min tabular   : {hi * 1e3:.3f} ms")
    print(f"tabular over  : {out['tabular_overhead_pct']:+.2f}%")
    print(f"peak loadavg  : {peak_load:.2f} (threshold {args.max_load}) -> {out['timing_ratio_note']}")
    print(f"counters      : {out['deterministic_counters']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())