"""Capture the F2-3 golden arrays for a fully loaded C+L span.

This script is an **evidence generator**, not a test. It records the four
returns of ``_calc_span_noise_and_power_jit`` for the fully loaded 230x50 GHz
C+L scenario **as produced by the pre-FG-2 scalar-``symbol_rate_hz``
implementation**, together with every input that produced them.

``tests/test_fg2_flexible_grid.py`` then replays exactly these inputs through
the post-FG-2 per-connection-array kernel and asserts ``np.array_equal``
(F2-3). Because the inputs are stored verbatim, the comparison is bit-exact
rather than tolerance-based: any change in how ``symbol_rate_hz`` is applied
per connection shows up immediately.

Regenerate with (from the FG-1 baseline commit, i.e. *before* the kernel
change)::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-nb-impl \\
        $HOME/miniconda3/envs/rl/bin/python scripts/fg2_capture_golden.py \\
        --out results/fg2_golden_cplusl_230x50.npz

The recorded ``git rev-parse HEAD`` is stored alongside the arrays so the
provenance of the golden data is auditable.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.numba_kernels import _calc_span_noise_and_power_jit
from network_nlo_eval.network.elements import FiberSpanConfig

NUM_CHANNELS = 230
CHANNEL_SPACING_HZ = 50e9
CENTER_FREQUENCY_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9
PHI_KURTOSIS = 0.0  # Gaussian; -0.6190 would be 64QAM
SPAN_LENGTH_M = 100e3
LAUNCH_POWER_W = 1e-3
N_EFFECTIVE_SPANS = 1


def build_inputs() -> dict[str, np.ndarray | float]:
    """Build the fully loaded 230x50 GHz C+L span inputs (pre-FG-2 form)."""
    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    ev = MultiBandISRSGN(
        grid=grid, ref_fiber_config=fiber, symbol_rate_hz=SYMBOL_RATE_HZ, phi_kurtosis=PHI_KURTOSIS
    )
    return {
        "power_in_w": np.full(NUM_CHANNELS, LAUNCH_POWER_W),
        "span_length_m": SPAN_LENGTH_M,
        "f_rel_hz": ev.f_rel_hz,
        "f_abs_hz": ev.f_abs_hz,
        "lambdas_m": ev.lambdas_m,
        "alpha_power_npm": ev.alpha_power_npm_channels,
        "a_eff_m2": ev.a_eff_m2_channels,
        "beta2_s2_m": ev.beta2_s2_m,
        "beta3_s3_m": ev.beta3_s3_m,
        "n2_m2_w": ev.n2_m2_w,
        "nf_lin_channels": ev.nf_lin_channels,
        "symbol_rate_hz": SYMBOL_RATE_HZ,
        "n_effective_spans": N_EFFECTIVE_SPANS,
        "cr_w_m_hz": ev.cr_w_m_hz,
        "delta_f_co_hz": ev.delta_f_co_hz,
    }


def main() -> int:
    """Generate the golden .npz and print a short digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output .npz path")
    args = parser.parse_args()

    kwargs = build_inputs()
    # The pre-FG-2 kernel takes a scalar symbol_rate_hz; the njit function
    # signature is unchanged by storing the scalar as a 0-d array here.
    p_out, s2_spm, s2_xpm, s2_ase = _calc_span_noise_and_power_jit(**kwargs)

    payload: dict[str, np.ndarray] = {
        "p_out_w": np.asarray(p_out),
        "sigma2_spm_w": np.asarray(s2_spm),
        "sigma2_xpm_w": np.asarray(s2_xpm),
        "sigma2_ase_w": np.asarray(s2_ase),
    }
    for name, value in kwargs.items():
        payload[f"in__{name}"] = np.asarray(value)

    payload["meta__num_channels"] = np.asarray(NUM_CHANNELS)
    payload["meta__channel_spacing_hz"] = np.asarray(CHANNEL_SPACING_HZ)
    payload["meta__center_frequency_hz"] = np.asarray(CENTER_FREQUENCY_HZ)
    payload["meta__symbol_rate_hz"] = np.asarray(SYMBOL_RATE_HZ)
    payload["meta__span_length_m"] = np.asarray(SPAN_LENGTH_M)
    payload["meta__launch_power_w"] = np.asarray(LAUNCH_POWER_W)
    payload["meta__n_effective_spans"] = np.asarray(N_EFFECTIVE_SPANS)
    # 0.0 = Gaussian; MultiBandISRSGN's default and the kernel's default.
    payload["meta__phi_kurtosis"] = np.asarray(PHI_KURTOSIS)
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        rev = "unknown"
    payload["meta__git_rev"] = np.asarray(rev)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)

    print(f"wrote {args.out}")
    print(f"git rev      : {rev}")
    print(f"n_connections: {NUM_CHANNELS}")
    for name in ("p_out_w", "sigma2_spm_w", "sigma2_xpm_w", "sigma2_ase_w"):
        arr = payload[name]
        print(f"  {name:16s} shape={arr.shape} min={arr.min():.12e} max={arr.max():.12e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
