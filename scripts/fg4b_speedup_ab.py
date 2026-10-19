r"""FG-4b speedup A/B: NLI vectorisation, isolated in one process.

Methodology (inherited from FG-4a, and non-negotiable here)
----------------------------------------------------------
The orchestrator's machine is shared and its load average swings roughly 4.6-42
during a work package. A before/after speedup measured in two separate processes
is therefore **not a measurement** — the apparent difference is dominated by how
busy the host happened to be. This script removes the load difference instead of
reporting around it:

1. Both targets are callable in the **same** interpreter, against the same warm
   Numba/BLAS state, on the same CPU at the same moment.
2. The two targets are **interleaved sample-by-sample** (``A,B,B,A,A,B...`` with
   the order flipped each round) so drift in machine load hits both equally.
3. The reported speedup is the ratio of **min** samples, with ``loadavg``
   recorded before and after the whole run.

Interleaving is what makes the number trustworthy: it compares two measurements
taken microseconds apart, not two runs minutes apart.

What is compared
----------------
The interesting quantity is *this knife alone*, so the ODE is held **bit-exact**
across both arms. ``_calc_span_noise_and_power_jit`` is called with the same
``g_full`` (the FG-4a bit-exact G cache) in both arms, so the RK4/SRS solve
produces identical bits and the only difference measured is the NLI evaluation:

===============  =============================================================
``frozen_nli``   ``nli_*`` all ``None`` -> FG-2 frozen per-pair double loop
``tensor_nli``   the three (N,N) build-time tensors -> pure tensor hot path
===============  =============================================================

A third pair measures the **cumulative** effect against the FG-4a baseline
kernels (materialised from the baseline commit, imported under a second module
name), so the headline number for the work package is also reproducible.

Usage::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \\
        $HOME/miniconda3/envs/rl/bin/python scripts/fg4b_speedup_ab.py \\
        --out results/fg4b_speedup_ab.json
"""

from __future__ import annotations

import argparse
import json
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

#: The pre-FG-4a baseline the cumulative comparison is made against.
BASELINE_REV = "ea7ecb0"

SPAN_LENGTH_M = 100e3
N_STEPS = 20
ROUNDS = 40
SAMPLES_PER_ROUND = 3

#: The brief's theoretical ceiling for this knife, i.e. what a perfect NLI
#: elimination would buy *relative to the FG-4a state*. Reported against the
#: measurement so the shortfall is visible rather than implied.
THEORETICAL_CEILING = 5.2


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
    """Import the baseline ``numba_kernels`` under a distinct module name."""
    from fg4a_speedup_ab import load_baseline_kernels as _loader

    return _loader()


def interleaved_ab(
    target_a: Callable[[], Any],
    target_b: Callable[[], Any],
    *,
    rounds: int = ROUNDS,
    per_round: int = SAMPLES_PER_ROUND,
) -> dict[str, Any]:
    """Time two callables alternately, flipping the order each round.

    ``target_a`` is the *slower* (pre-optimisation) side, so
    ``speedup_a_over_b > 1`` means the optimisation helped.
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
    a_med, b_med = a_sorted[len(a_sorted) // 2], b_sorted[len(b_sorted) // 2]
    return {
        "samples": len(samples_a),
        "a_min_s": a_min,
        "a_median_s": a_med,
        "a_spread_ratio": a_sorted[-1] / a_min if a_min > 0 else float("nan"),
        "b_min_s": b_min,
        "b_median_s": b_med,
        "b_spread_ratio": b_sorted[-1] / b_min if b_min > 0 else float("nan"),
        "speedup_a_over_b": a_min / b_min if b_min > 0 else float("nan"),
        "median_speedup_a_over_b": a_med / b_med if b_med > 0 else float("nan"),
    }


def worst_relative(expected: Any, actual: Any) -> float:
    """Worst relative deviation over elements finite on both sides."""
    exp_arr = np.asarray(expected, dtype=np.float64)
    act_arr = np.asarray(actual, dtype=np.float64)
    both = np.isfinite(exp_arr) & np.isfinite(act_arr)
    if not both.any():
        return 0.0
    diff = np.abs(exp_arr[both] - act_arr[both])
    denom = np.abs(exp_arr[both])
    return float(np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0).max())


def main() -> int:
    """Run the interleaved A/B and write the evidence JSON."""
    from fg4a_speedup_ab import load_baseline_kernels as load_baseline

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "fg4b_speedup_ab.json")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    baseline_kernels = load_baseline()

    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    ev = MultiBandISRSGN(grid=grid, ref_fiber_config=fiber, symbol_rate_hz=SYMBOL_RATE_HZ, phi_kurtosis=0.0)

    # The profiled workload: fully loaded 230x50 GHz, but with a slice of slots
    # unoccupied so the zero-power guards are on the measured path too.
    power = np.full(NUM_CHANNELS, 1e-3, dtype=np.float64)
    power[: max(1, NUM_CHANNELS // 8)] = 0.0

    span = SPAN_LENGTH_M
    n_span = 1
    # The ODE is held at the **matvec** path, i.e. the FG-4a state this knife
    # starts from. Holding it at ``g_full`` instead would be the wrong control:
    # ``g_full`` runs the original per-pair ``_calc_dp_dz``, which is itself a
    # pre-FG-4a cost, and would inflate the "before" arm with work this knife
    # did not remove.
    g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)
    tensors = nk.nli_tensors(
        ev.n_nodes,
        n_span,
        ev.symbol_rate_hz_channels,
        ev.f_rel_hz,
        ev.lambdas_m,
        ev.a_eff_m2_channels,
        ev.alpha_power_npm_channels,
        span,
        ev.beta2_s2_m,
        ev.beta3_s3_m,
        ev.n2_m2_w,
        ev.phi_kurtosis,
    )

    common = dict(
        f_rel_hz=ev.f_rel_hz,
        f_abs_hz=ev.f_abs_hz,
        lambdas_m=ev.lambdas_m,
        alpha_power_npm=ev.alpha_power_npm_channels,
        a_eff_m2=ev.a_eff_m2_channels,
        beta2_s2_m=ev.beta2_s2_m,
        beta3_s3_m=ev.beta3_s3_m,
        n2_m2_w=ev.n2_m2_w,
        nf_lin_channels=ev.nf_lin_channels,
        symbol_rate_hz=ev.symbol_rate_hz_channels,
        n_effective_spans=n_span,
        cr_w_m_hz=ev.cr_w_m_hz,
        delta_f_co_hz=ev.delta_f_co_hz,
        phi_kurtosis=ev.phi_kurtosis,
    )
    tensor_kwargs = dict(
        nli_spm0=tensors[0],
        nli_spm1=tensors[1],
        nli_spm2=tensors[2],
        nli_gamma_ii_sq=tensors[3],
        nli_gamma_sq=tensors[4],
        nli_w0=tensors[5],
        nli_w1=tensors[6],
        nli_w2=tensors[7],
    )

    # --- arm A: FG-4a state (matvec ODE) + frozen per-pair NLI ---------------
    def frozen_nli() -> Any:
        return nk._calc_span_noise_and_power_jit(power, span, g_gain=g_gain, g_loss=g_loss, **common)

    # --- arm B: identical ODE, tensorised NLI ---------------------------------
    def tensor_nli() -> Any:
        return nk._calc_span_noise_and_power_jit(
            power, span, g_gain=g_gain, g_loss=g_loss, **common, **tensor_kwargs
        )

    # --- component floor: the ODE alone, which no amount of NLI work can remove
    def ode_only() -> Any:
        return nk._calc_exact_power_profile_ode_matvec(
            power, g_gain, g_loss, ev.alpha_power_npm_channels, span, n_steps=N_STEPS
        )

    # --- cumulative: the pre-FG-4a baseline kernels, whole span kernel ----------
    def baseline_span() -> Any:
        return baseline_kernels._calc_span_noise_and_power_jit(power, span, *common.values())

    load_before = read_loadavg()
    nli_only = interleaved_ab(frozen_nli, tensor_nli, rounds=args.rounds)
    cumulative = interleaved_ab(baseline_span, tensor_nli, rounds=args.rounds)
    # The ODE floor: timed on its own so the residual after the knife is
    # attributable rather than merely asserted. ``b`` (self) is a constant-cost
    # no-op target; only ``a_min_s`` is read.
    ode_floor = interleaved_ab(ode_only, lambda: None, rounds=args.rounds)
    load_after = read_loadavg()

    # Numerical self-check on the very workload that was timed, so the timing
    # table cannot be quoting numbers for a different computation.
    a_out = frozen_nli()
    b_out = tensor_nli()
    nli_dev = max(worst_relative(a, b) for a, b in zip(a_out, b_out, strict=True))
    # The ODE half must be bit-identical: that is what isolates this knife.
    ode_identical = bool(np.array_equal(np.asarray(a_out[0]), np.asarray(b_out[0]), equal_nan=True))

    speedup = nli_only["speedup_a_over_b"]
    payload = {
        "schema_version": 1,
        "method": (
            "same-process interleaved A/B (order flipped each round), min-sample ratio; "
            "both arms share the bit-exact FG-4a G cache so only the NLI evaluation differs"
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
            "n_effective_spans": n_span,
            "n_loaded_channels": int(np.count_nonzero(power)),
        },
        "loadavg_before": load_before,
        "loadavg_after": load_after,
        "loadavg_delta_1min": load_after[0] - load_before[0],
        "numeric_check": {
            "ode_output_bit_identical_between_arms": ode_identical,
            "nli_worst_relative_deviation": nli_dev,
            "nli_within_1e-12": nli_dev <= 1e-12,
        },
        "theoretical_ceiling": THEORETICAL_CEILING,
        "speedups": {
            "nli_vectorisation_only": nli_only,
            "cumulative_vs_pre_fg4a_baseline": cumulative,
        },
    }
    payload["nli_speedup_vs_ceiling"] = {
        "measured": speedup,
        "nominal_ceiling": THEORETICAL_CEILING,
        "fraction_of_nominal_ceiling": speedup / THEORETICAL_CEILING if THEORETICAL_CEILING else float("nan"),
        # The brief's nominal ceiling assumes the NLI cost goes to *zero*, leaving
        # only the ODE. It does not: the ODE floor is a hard floor for this knife,
        # so the ratio a *perfect* NLI elimination would achieve is bounded by
        # before/ode_floor. Where that attainable ceiling sits below the nominal
        # one, the nominal number is unreachable by construction and the shortfall
        # must be read against the attainable one instead.
        "ode_floor_ms": ode_floor["a_min_s"] * 1e3,
        "attainable_ceiling": (
            nli_only["a_min_s"] / ode_floor["a_min_s"] if ode_floor["a_min_s"] > 0 else float("nan")
        ),
        "fraction_of_attainable_ceiling": (
            speedup * ode_floor["a_min_s"] / nli_only["a_min_s"]
            if nli_only["a_min_s"] > 0
            else float("nan")
        ),
        "nli_cost_before_ms": (nli_only["a_min_s"] - ode_floor["a_min_s"]) * 1e3,
        "nli_cost_after_ms": (nli_only["b_min_s"] - ode_floor["a_min_s"]) * 1e3,
        "residual_nli_fraction_of_before": (
            (nli_only["b_min_s"] - ode_floor["a_min_s"]) / nli_only["a_min_s"]
            if nli_only["a_min_s"] > 0
            else float("nan")
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"package : {payload['provenance']['package_path']}")
    print(f"loadavg : {load_before} -> {load_after} (1min delta {payload['loadavg_delta_1min']:+.2f})")
    print(
        f"numeric : ODE bit-identical across arms={ode_identical}, "
        f"NLI worst rel dev={nli_dev:.3e} (within 1e-12: {nli_dev <= 1e-12})"
    )
    print()
    for name, row in payload["speedups"].items():
        print(f"{name}:")
        print(
            f"  before  min {row['a_min_s'] * 1e3:8.3f} ms  median {row['a_median_s'] * 1e3:8.3f} ms"
            f"  spread {row['a_spread_ratio']:.2f}"
        )
        print(
            f"  after   min {row['b_min_s'] * 1e3:8.3f} ms  median {row['b_median_s'] * 1e3:8.3f} ms"
            f"  spread {row['b_spread_ratio']:.2f}"
        )
        print(
            f"  speedup min {row['speedup_a_over_b']:.3f}x   median {row['median_speedup_a_over_b']:.3f}x"
        )
    info = payload["nli_speedup_vs_ceiling"]
    print()
    print(
        f"NLI knife: {info['measured']:.3f}x  "
        f"({info['fraction_of_nominal_ceiling'] * 100:.1f}% of the nominal {info['nominal_ceiling']}x ceiling, "
        f"{info['fraction_of_attainable_ceiling'] * 100:.1f}% of the attainable {info['attainable_ceiling']:.2f}x)"
    )
    print(f"  ODE floor            {info['ode_floor_ms']:7.3f} ms   (unreachable by NLI work)")
    print(f"  NLI cost before     {info['nli_cost_before_ms']:7.3f} ms")
    print(
        f"  NLI cost after      {info['nli_cost_after_ms']:7.3f} ms"
        f"  ({info['residual_nli_fraction_of_before'] * 100:.1f}% of before remains)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
