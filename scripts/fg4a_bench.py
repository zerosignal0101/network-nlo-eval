r"""FG-4a timing harness: per-stage cost with machine load recorded per reading.

This is an **evidence generator**, not a test. It times the pieces the FG-4a
brief identified inside a single span's physics kernel and writes them to JSON
together with the machine's ``loadavg`` at the moment of measurement.

Why the load average is recorded per reading
---------------------------------------------
This machine is shared, and the orchestrator observed ``loadavg`` around 41
while the FG-4a baseline was taken. A speedup ratio computed from two readings
taken under different load is not a measurement. Recording ``/proc/loadavg``
next to every number makes an unfair comparison *visible* rather than something
a reader has to take on trust, and lets :func:`compare` refuse to quote a ratio
when the loads are not comparable.

What is timed
-------------
``_calc_dp_dz``            one call (the RK4 loop calls it 4x20 = 80 times)
``G`` construction         the double Python loop over (i, k) — the FG-4a target
``_calc_exact_power_profile_ode``   the ODE (G build + 80 dp_dz calls)
``_compute_nli_variances`` the NLI block
``_calculate_ase_noise_variance``   the ASE block
``_calc_span_noise_and_power_jit``  the whole span kernel
``evaluate_path``          one 2-span path, for end-to-end context

Methodology
-----------
* One untimed warm-up call per target so Numba JIT compilation (and the
  ``NUMBA_CACHE_DIR`` load) is excluded from every reading.
* ``reps`` timed repetitions, reported as **min** plus the median and full
  sample. Min is the headline number because it is the reading least polluted by
  contention from the other workload on this host; the spread across samples is
  reported so a noisy run cannot masquerade as a clean one.
* ``loadavg`` is read before and after each target's repetition block.

Usage::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \\
        $HOME/miniconda3/envs/rl/bin/python scripts/fg4a_bench.py \\
        --out results/fg4a_bench_baseline.json --label baseline
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

from fg4a_scenarios import (  # noqa: E402
    CENTER_FREQUENCY_HZ,
    CHANNEL_SPACING_HZ,
    EDFA_NF_DB,
    NUM_CHANNELS,
    SYMBOL_RATE_HZ,
)

import network_nlo_eval  # noqa: E402
from network_nlo_eval.core.spectrum import SpectrumGrid  # noqa: E402
from network_nlo_eval.models import numba_kernels as nk  # noqa: E402
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath  # noqa: E402
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig  # noqa: E402

SPAN_LENGTH_M = 80e3
N_STEPS = 20
REPS = 15

#: Above this load-average difference between two runs, :func:`compare` refuses
#: to quote a speedup and says so instead.
LOAD_COMPARABILITY_TOLERANCE = 5.0


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


def time_target(fn: Callable[[], Any], reps: int = REPS) -> dict[str, Any]:
    """Time ``fn`` ``reps`` times, returning min/median/spread plus load.

    One untimed warm-up call first, so JIT compilation is never counted.
    """
    fn()
    load_before = read_loadavg()
    samples: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    load_after = read_loadavg()
    ordered = sorted(samples)
    return {
        "reps": reps,
        "min_s": ordered[0],
        "median_s": ordered[len(ordered) // 2],
        "max_s": ordered[-1],
        "spread_ratio": ordered[-1] / ordered[0] if ordered[0] > 0 else float("nan"),
        "loadavg_before": load_before,
        "loadavg_after": load_after,
    }


def build_evaluator() -> MultiBandISRSGN:
    """The 230x50 GHz C+L evaluator, matching the profiled workload."""
    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    return MultiBandISRSGN(grid=grid, ref_fiber_config=fiber, symbol_rate_hz=SYMBOL_RATE_HZ, phi_kurtosis=0.0)


def build_g_reference(f_hz: np.ndarray, cr_w_m_hz: float) -> np.ndarray:
    """Python mirror of the in-kernel ``G`` construction (for the _calc_dp_dz timing).

    Reproduces the kernel's loop exactly, including operation order, so the
    timed ``_calc_dp_dz`` is doing real work on a real matrix.
    """
    n = len(f_hz)
    g = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(n):
            df_thz = abs(f_hz[i] - f_hz[k]) / 1e12
            if df_thz <= 13.2:
                value = cr_w_m_hz * (df_thz * 1e12)
            elif df_thz < 15.6:
                peak_g = cr_w_m_hz * 13.2e12
                value = peak_g * (15.6 - df_thz) / (15.6 - 13.2)
            else:
                value = 0.0
            g[i, k] = value
    return g


def collect(label: str, reps: int = REPS) -> dict[str, Any]:
    """Time every FG-4a target and return the full payload."""
    ev = build_evaluator()
    n = NUM_CHANNELS
    power = np.full(n, 1e-4, dtype=np.float64)
    power[: max(1, n // 8)] = 0.0  # some empty slots, like the profiled run
    nf = ev.nf_lin_channels
    cr = ev.cr_w_m_hz
    span = SPAN_LENGTH_M
    g_ref = build_g_reference(ev.f_abs_hz, cr)
    p_out = nk._calc_exact_power_profile_ode(power, ev.f_abs_hz, ev.alpha_power_npm_channels, span, cr, N_STEPS)

    def span_kernel(g_full=None, g_gain=None, g_loss=None) -> Any:
        return nk._calc_span_noise_and_power_jit(
            power,
            span,
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
            g_full=g_full,
            g_gain=g_gain,
            g_loss=g_loss,
        )

    # Which optional accelerations exist in this tree? Recorded so a baseline
    # JSON and an optimised JSON can be compared field-by-field honestly.
    features = {
        "has_g_cache_api": hasattr(nk, "_build_g_pair_matrices"),
        "has_branchless_dp_dz": hasattr(nk, "_calc_dp_dz_matvec"),
        "has_memoised_g": hasattr(nk, "raman_g_pair_matrices"),
    }

    targets: dict[str, Callable[[], Any]] = {
        # NOTE: this is the *pure-Python* mirror, NOT the njit G build that runs
        # inside the kernel. It exists to document how slow the interpreted
        # double loop is, and to keep the reference matrix for _calc_dp_dz. Do
        # not read its cost as the cost of the kernel's own G construction --
        # see "derived" below, which isolates the njit one by subtraction.
        "g_matrix_python_loop": lambda: build_g_reference(ev.f_abs_hz, cr),
        "calc_dp_dz_single": lambda: nk._calc_dp_dz(power, g_ref, ev.f_abs_hz, ev.alpha_power_npm_channels),
        "calc_exact_power_profile_ode": lambda: nk._calc_exact_power_profile_ode(
            power, ev.f_abs_hz, ev.alpha_power_npm_channels, span, cr, N_STEPS
        ),
        # n_steps=1 still builds G but does only 4 dp_dz calls; the difference
        # against n_steps=20 isolates the 76 extra dp_dz calls, and the
        # remainder isolates the njit G build.
        "calc_exact_power_profile_ode_nsteps1": lambda: nk._calc_exact_power_profile_ode(
            power, ev.f_abs_hz, ev.alpha_power_npm_channels, span, cr, 1
        ),
        "compute_nli_variances": lambda: nk._compute_nli_variances(
            n,
            1,
            ev.symbol_rate_hz_channels,
            power,
            p_out,
            ev.f_rel_hz,
            ev.lambdas_m,
            ev.a_eff_m2_channels,
            ev.alpha_power_npm_channels,
            span,
            ev.beta2_s2_m,
            ev.beta3_s3_m,
            ev.n2_m2_w,
            ev.phi_kurtosis,
        ),
        "calculate_ase_noise_variance": lambda: nk._calculate_ase_noise_variance(
            power, p_out, ev.f_abs_hz, nf, ev.symbol_rate_hz_channels, 1
        ),
        "calc_span_noise_and_power_jit": span_kernel,
    }
    if features["has_branchless_dp_dz"]:
        g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, cr)
        targets["calc_dp_dz_matvec"] = lambda: nk._calc_dp_dz_matvec(power, g_gain, g_loss, ev.alpha_power_npm_channels)

    # Both accelerated span-kernel variants, timed in the *same* process and the
    # same load window as the baseline call above. That matters: comparing a
    # cached-G reading taken under load L1 against a baseline reading taken under
    # load L2 would not be a measurement, so all three share one window.
    if features["has_g_cache_api"] and features["has_memoised_g"]:
        g_full = nk.raman_g_matrix(ev.f_abs_hz, cr)
        targets["calc_span_noise_and_power_jit_cached_g"] = lambda: span_kernel(g_full=g_full)
    if features["has_branchless_dp_dz"] and features["has_memoised_g"]:
        targets["calc_span_noise_and_power_jit_matvec"] = lambda: span_kernel(g_gain=g_gain, g_loss=g_loss)

    def path_evaluate() -> Any:
        path = MultiSpanOpticalPath(
            fiber_configs=[FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)] * 2,
            edfa_configs=[EDFAConfig(target_gain_db=None, noise_figure_db=EDFA_NF_DB)] * 2,
            roadm_configs=[None, None],
            link_keys=[(0, 1), (1, 2)],
            node_path=[0, 1, 2],
        )
        return ev.evaluate_path(path, power)

    targets["evaluate_path_2_span"] = path_evaluate

    measurements = {name: time_target(fn, reps) for name, fn in targets.items()}

    span_min = measurements["calc_span_noise_and_power_jit"]["min_s"]

    # Isolate the njit G build by subtraction. The ODE is
    #     T(n_steps) = T_g_build + 4 * n_steps * T_dp_dz
    # so T(1) - T(20) is the cost of 76 extra dp_dz calls, and
    #     T_g_build = T(1) - 4 * T_dp_dz_single
    # Using min-of-reps for all three keeps the subtraction on one statistic.
    # This is an estimate from noisy measurements, so it is reported as such and
    # may go slightly negative when the G build is below the timing noise floor.
    ode_20 = measurements["calc_exact_power_profile_ode"]["min_s"]
    ode_1 = measurements["calc_exact_power_profile_ode_nsteps1"]["min_s"]
    dp_single = measurements["calc_dp_dz_single"]["min_s"]
    derived = {
        "dp_dz_calls_per_span_nsteps20": 4 * N_STEPS,
        "dp_dz_total_share_of_span_kernel": (4 * N_STEPS * dp_single / span_min) if span_min else float("nan"),
        "g_build_njit_est_s": ode_1 - 4 * dp_single,
        "g_build_njit_est_share_of_span_kernel": ((ode_1 - 4 * dp_single) / span_min if span_min else float("nan")),
        "ode_slope_s_per_step": (ode_20 - ode_1) / (N_STEPS - 1),
        "note": (
            "g_build_njit_est_s is derived by subtraction from min-of-reps timings, not "
            "measured directly; it can be negative when the true G-build cost sits below "
            "this machine's timing noise floor."
        ),
    }

    return {
        "schema_version": 1,
        "label": label,
        "provenance": {
            "git_rev": _git_rev(),
            "package_path": str(Path(network_nlo_eval.__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR", "<unset>"),
            "cpu_model": _cpu_model(),
        },
        "workload": {
            "num_channels": n,
            "channel_spacing_hz": CHANNEL_SPACING_HZ,
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "span_length_m": span,
            "n_steps": N_STEPS,
            "n_loaded_channels": int(np.count_nonzero(power)),
            "launch_power_w": 1e-4,
            "cr_w_m_hz": cr,
        },
        "features": features,
        "loadavg_at_start": read_loadavg(),
        "measurements": measurements,
        "derived": derived,
        "share_of_span_kernel": {
            name: data["min_s"] / span_min if span_min > 0 else float("nan") for name, data in measurements.items()
        },
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:  # pragma: no cover
        pass
    return "unknown"


def compare(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    """Compare two benchmark payloads, refusing to quote ratios under differing load."""
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))

    base_load = baseline.get("loadavg_at_start", [float("nan")])[0]
    cand_load = candidate.get("loadavg_at_start", [float("nan")])[0]
    comparable = bool(
        np.isfinite(base_load) and np.isfinite(cand_load) and abs(base_load - cand_load) <= LOAD_COMPARABILITY_TOLERANCE
    )

    rows: dict[str, Any] = {}
    for name, cand_data in candidate.get("measurements", {}).items():
        base_data = baseline.get("measurements", {}).get(name)
        if base_data is None:
            rows[name] = {"status": "absent_in_baseline"}
            continue
        base_min = base_data["min_s"]
        cand_min = cand_data["min_s"]
        rows[name] = {
            "baseline_min_s": base_min,
            "candidate_min_s": cand_min,
            "speedup": base_min / cand_min if cand_min > 0 else float("nan"),
            "baseline_loadavg": base_data["loadavg_before"],
            "candidate_loadavg": cand_data["loadavg_before"],
            "comparable": comparable,
        }

    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_label": baseline.get("label"),
        "candidate_label": candidate.get("label"),
        "loadavg_1min": {"baseline": base_load, "candidate": cand_load},
        "load_comparability_tolerance": LOAD_COMPARABILITY_TOLERANCE,
        "loads_comparable": comparable,
        "verdict": (
            "load averages are comparable within tolerance; speedups are quotable"
            if comparable
            else "load averages DIFFER by more than the tolerance; speedups are NOT quotable "
            "as a clean speedup on this machine — re-measure when the host is quiet"
        ),
        "measurements": rows,
    }


def main() -> int:
    """Run the benchmark and optionally compare against a baseline file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output JSON path")
    parser.add_argument("--label", default="run", help="label recorded in the payload")
    parser.add_argument("--reps", type=int, default=REPS, help="timed repetitions per target")
    parser.add_argument("--compare-to", type=Path, default=None, help="baseline JSON to compare against")
    args = parser.parse_args()

    payload = collect(args.label, args.reps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"label     : {args.label}")
    print(f"git rev   : {payload['provenance']['git_rev']}")
    print(f"package   : {payload['provenance']['package_path']}")
    print(f"features  : {payload['features']}")
    print(f"loadavg   : {payload['loadavg_at_start']}")
    print()
    print(f"{'target':<32}{'min ms':>10}{'median ms':>11}{'spread':>9}{'share':>9}")
    for name, data in payload["measurements"].items():
        share = payload["share_of_span_kernel"][name]
        print(
            f"{name:<32}{data['min_s'] * 1e3:>10.3f}{data['median_s'] * 1e3:>11.3f}"
            f"{data['spread_ratio']:>9.2f}{share * 100:>8.1f}%"
        )

    if args.compare_to:
        result = compare(args.compare_to, args.out)
        print()
        print(f"vs {result['baseline']} ({result['baseline_label']}):")
        print(
            f"  loadavg 1min: baseline={result['loadavg_1min']['baseline']} "
            f"candidate={result['loadavg_1min']['candidate']}"
        )
        print(f"  {result['verdict']}")
        for name, row in result["measurements"].items():
            if "speedup" in row:
                print(
                    f"  {name:<32}{row['speedup']:>8.3f}x  ({row['baseline_min_s'] * 1e3:.3f} -> "
                    f"{row['candidate_min_s'] * 1e3:.3f} ms)"
                )
        compare_path = args.out.with_name(args.out.stem + "_vs_baseline.json")
        compare_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
