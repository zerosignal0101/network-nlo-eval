r"""FG-4a speedup A/B: baseline vs optimised, interleaved in one process.

Why this script exists
----------------------
The orchestrator's machine is shared and its load average was observed swinging
between roughly 10 and 42 during FG-4a. A before/after speedup measured in two
separate processes is therefore **not a measurement**: the apparent difference
is dominated by how busy the host happened to be. ``scripts/fg4a_bench.py``
records ``loadavg`` precisely so that this trap is visible, and it refuses to
quote a ratio when the two runs' loads differ by more than its tolerance.

This script removes the load difference instead of reporting around it:

1. It materialises the **baseline** ``numba_kernels.py`` from the baseline commit
   into a temporary file and imports it under a second module name, so the old
   and new kernels are both callable in the *same* interpreter, against the
   *same* already-warm Numba/BLAS state, on the *same* CPU at the *same* moment.
2. It **interleaves** the two targets sample-by-sample (``A,B,B,A,A,B,...`` with
   the order flipped each round) so that any drift in machine load hits both
   targets equally instead of biasing one.
3. It reports the ratio of **min** samples — the reading least polluted by
   contention — together with the load average recorded across the whole run.

Interleaving is the property that makes the number trustworthy: the ratio
compares two measurements that were taken microseconds apart from each other,
not two runs that were minutes apart.

Usage::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \\
        $HOME/miniconda3/envs/rl/bin/python scripts/fg4a_speedup_ab.py \\
        --out results/fg4a_speedup_ab.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

from fg4a_scenarios import (  # noqa: E402
    CENTER_FREQUENCY_HZ,
    CHANNEL_SPACING_HZ,
    NUM_CHANNELS,
    SYMBOL_RATE_HZ,
)

import network_nlo_eval  # noqa: E402
from network_nlo_eval.core.spectrum import SpectrumGrid  # noqa: E402
from network_nlo_eval.models import numba_kernels as nk  # noqa: E402
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN  # noqa: E402
from network_nlo_eval.network.elements import FiberSpanConfig  # noqa: E402

BASELINE_REV = "ea7ecb0"
SPAN_LENGTH_M = 80e3
N_STEPS = 20
ROUNDS = 40
SAMPLES_PER_ROUND = 3


def read_loadavg() -> list[float]:
    """Read the 1/5/15-minute load averages from procfs."""
    try:
        return [float(value) for value in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):  # pragma: no cover
        return [float("nan")] * 3


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


def load_baseline_kernels() -> Any:
    """Import the baseline ``numba_kernels`` under a distinct module name.

    The baseline module imports only ``core.constants``/``core.types``, so it
    loads cleanly alongside the current one. Both definitions of
    ``_calc_span_noise_and_power_jit`` then coexist in one process.
    """
    source = subprocess.run(
        ["git", "show", f"{BASELINE_REV}:src/network_nlo_eval/models/numba_kernels.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fg4a_baseline_numba_kernels.py"
        path.write_text(source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("fg4a_baseline_kernels", path)
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RuntimeError("could not build an import spec for the baseline kernels")
        module = importlib.util.module_from_spec(spec)
        # The baseline source carries its own `import numpy as np` and
        # `from numba import njit`; registering under a private name keeps the two
        # copies from colliding in sys.modules.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        # Copy the module's globals so it survives the TemporaryDirectory being
        # removed (its code object and caches live on, but keep the dict alive).
        globals()["_baseline_kernels"] = module
    return module


def interleaved_ab(
    target_a: Callable[[], Any],
    target_b: Callable[[], Any],
    *,
    rounds: int = ROUNDS,
    per_round: int = SAMPLES_PER_ROUND,
) -> dict[str, Any]:
    """Time two callables alternately, flipping the order each round.

    Returns min/median per side plus the min-based speedup of A over B. Flipping
    the order each round cancels any systematic advantage from whichever target
    is measured first within a round.
    """
    target_a()  # warm: JIT compile, Numba cache load, BLAS init
    target_b()

    samples_a: list[float] = []
    samples_b: list[float] = []
    for round_index in range(rounds):
        order = (("a", target_a), ("b", target_b))
        if round_index % 2:
            order = (("b", target_b), ("a", target_a))
        for label, fn in order:
            for _ in range(per_round):
                start = time.perf_counter()
                fn()
                elapsed = time.perf_counter() - start
                (samples_a if label == "a" else samples_b).append(elapsed)

    a_sorted = sorted(samples_a)
    b_sorted = sorted(samples_b)
    a_min, b_min = a_sorted[0], b_sorted[0]
    return {
        "samples": len(samples_a),
        "a_min_s": a_min,
        "a_median_s": a_sorted[len(a_sorted) // 2],
        "a_spread_ratio": a_sorted[-1] / a_min if a_min > 0 else float("nan"),
        "b_min_s": b_min,
        "b_median_s": b_sorted[len(b_sorted) // 2],
        "b_spread_ratio": b_sorted[-1] / b_min if b_min > 0 else float("nan"),
        "speedup_a_over_b": a_min / b_min if b_min > 0 else float("nan"),
        "median_speedup_a_over_b": (
            a_sorted[len(a_sorted) // 2] / b_sorted[len(b_sorted) // 2]
            if b_sorted[len(b_sorted) // 2] > 0
            else float("nan")
        ),
    }


def main() -> int:
    """Run the interleaved A/B and write the evidence JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "fg4a_speedup_ab.json")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    baseline = load_baseline_kernels()

    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    ev = MultiBandISRSGN(grid=grid, ref_fiber_config=fiber, symbol_rate_hz=SYMBOL_RATE_HZ, phi_kurtosis=0.0)

    power = np.full(NUM_CHANNELS, 1e-4, dtype=np.float64)
    power[: max(1, NUM_CHANNELS // 8)] = 0.0
    nf = ev.nf_lin_channels
    cr = ev.cr_w_m_hz
    span = SPAN_LENGTH_M

    g_full = nk.raman_g_matrix(ev.f_abs_hz, cr)
    g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, cr)

    common = (
        ev.f_rel_hz,
        ev.f_abs_hz,
        ev.lambdas_m,
        ev.alpha_power_npm_channels,
        ev.a_eff_m2_channels,
        ev.beta2_s2_m,
        ev.beta3_s3_m,
        ev.n2_m2_w,
        nf,
        ev.symbol_rate_hz_channels,
        1,
        cr,
        ev.delta_f_co_hz,
        ev.phi_kurtosis,
    )

    def baseline_span() -> Any:
        return baseline._calc_span_noise_and_power_jit(power, span, *common)

    def cached_g_span() -> Any:
        return nk._calc_span_noise_and_power_jit(power, span, *common, g_full=g_full, g_gain=None, g_loss=None)

    def matvec_span() -> Any:
        return nk._calc_span_noise_and_power_jit(power, span, *common, g_full=None, g_gain=g_gain, g_loss=g_loss)

    load_before = read_loadavg()

    # A = baseline, B = accelerated, so speedup_a_over_b > 1 means the
    # accelerated path is faster.
    g_cache_only = interleaved_ab(baseline_span, cached_g_span, rounds=args.rounds)
    g_cache_and_matvec = interleaved_ab(baseline_span, matvec_span, rounds=args.rounds)

    load_after = read_loadavg()

    # Confirm the two accelerated paths agree numerically on this workload too,
    # so the timing table is not quoting numbers for the wrong computation.
    base_out = baseline_span()
    cached_out = cached_g_span()
    matvec_out = matvec_span()
    cached_identical = all(
        np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True) for a, b in zip(base_out, cached_out, strict=True)
    )
    worst_rel = 0.0
    for a, b in zip(base_out, matvec_out, strict=True):
        exp_arr = np.asarray(a, dtype=np.float64)
        act_arr = np.asarray(b, dtype=np.float64)
        both = np.isfinite(exp_arr) & np.isfinite(act_arr)
        if both.any():
            diff = np.abs(exp_arr[both] - act_arr[both])
            denom = np.abs(exp_arr[both])
            rel = np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)
            worst_rel = max(worst_rel, float(rel.max()))

    payload = {
        "schema_version": 1,
        "method": (
            "baseline kernels imported from the baseline commit into the same process; "
            "targets interleaved sample-by-sample with the order flipped each round; "
            "speedup is the ratio of min samples"
        ),
        "provenance": {
            "git_rev": _git_rev(),
            "baseline_rev": BASELINE_REV,
            "package_path": str(Path(network_nlo_eval.__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "workload": {
            "num_channels": NUM_CHANNELS,
            "span_length_m": span,
            "n_steps": N_STEPS,
            "n_loaded_channels": int(np.count_nonzero(power)),
        },
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "loadavg_delta_1min": load_after[0] - load_before[0],
        "numeric_check": {
            "cached_g_bit_identical_to_baseline": cached_identical,
            "matvec_worst_relative_deviation": worst_rel,
            "matvec_within_1e-12": worst_rel <= 1e-12,
        },
        "span_kernel_speedups": {
            "g_cache_only": g_cache_only,
            "g_cache_plus_branchless_matvec": g_cache_and_matvec,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"baseline rev : {BASELINE_REV} (imported into this same process)")
    print(f"package      : {payload['provenance']['package_path']}")
    print(f"loadavg      : {load_before} -> {load_after} (1min delta {payload['loadavg_delta_1min']:+.2f})")
    print(
        f"numeric check: cached_g bit-identical={cached_identical}, "
        f"matvec worst rel dev={worst_rel:.3e} (within 1e-12: {worst_rel <= 1e-12})"
    )
    print()
    for name, row in payload["span_kernel_speedups"].items():
        print(f"{name}:")
        print(
            f"  baseline  min {row['a_min_s'] * 1e3:8.3f} ms  median {row['a_median_s'] * 1e3:8.3f} ms"
            f"  spread {row['a_spread_ratio']:.2f}"
        )
        print(
            f"  optimised min {row['b_min_s'] * 1e3:8.3f} ms  median {row['b_median_s'] * 1e3:8.3f} ms"
            f"  spread {row['b_spread_ratio']:.2f}"
        )
        print(
            f"  speedup (min) {row['speedup_a_over_b']:.3f}x   speedup (median) {row['median_speedup_a_over_b']:.3f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
