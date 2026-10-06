r"""FG-4c batch-knee measurement harness (script-level prototype, NOT a production path).

What this measures
------------------
The **complete single-span kernel** (RK4/SRS ODE + tensorised NLI + ASE), not
just the GEMV, swept over candidate batch size ``B`` and channel count ``N`` on
CPU and GPU. Per-candidate time = total / B. The knee is where the GPU's
per-candidate time drops below the CPU's.

Why the batched form is three GEMMs, not a (B,N,N) tensor
--------------------------------------------------------
A naive batching of the FG-4b XPM term materialises a ``(B,N,N)`` tensor. At
N=2100, B=1024 that is 36 GB -- impossible, and it is exactly why a naive
"just add a batch axis" GPU port looks attractive on paper and dies in
practice. The sum over the perturbation axis ``l`` is however *linear* once the
data-dependent ratio is pulled out:

    sigma_xpm[b,i] = (32/27) * big_t[b,i]^2 * p_in[b,i] *
        sum_l ( gamma_sq[i,l] * p_sq[b,l] / rs[l] ) *
               ( w0[i,l] + 2*ratio[b,l]*w1[i,l] + ratio_sq[b,l]*w2[i,l] )

Expanding the bracket in powers of ``ratio[b,l]`` and collecting the ``l`` index
into the matrix side gives three matrix products of shape ``(N,N) @ (N,B)``:

    u0[b,l] = p_sq[b,l] / rs[l]
    u1[b,l] = u0[b,l] * ratio[b,l]
    u2[b,l] = u0[b,l] * ratio_sq[b,l]
    M0 = gamma_sq * w0   (diagonal zeroed)
    M1 = gamma_sq * w1   (diagonal zeroed)
    M2 = gamma_sq * w2   (diagonal zeroed)
    sigma_xpm[b,i] = (32/27)*big_t[b,i]^2*p_in[b,i] * ( (M0@u0.T)[i,b]
                                                      + 2*(M1@u1.T)[i,b]
                                                      +   (M2@u2.T)[i,b] )

So working memory is O(B*N) for the batch and O(N^2) for three matrices that
are **shared across the whole batch** -- they are built once. The RK4 part is
80 such products ((N,N)@(N,B), two per substep). This is the shape the whole
acceleration programme was aiming at, and it is why the batch dimension is
worth measuring at all.

Masking equivalence
-------------------
The per-pair mask ``p_in[b,i]>0 & p_in[b,l]>0 & i!=l`` is reproduced exactly:

* ``i == l`` -- zeroed on the diagonal of M0/M1/M2 at build time (the baseline
  ``continue``-skips self pairs);
* ``p_in[b,i] == 0`` -- the ``p_in[b,i]`` factor makes the whole row zero, and
  the result is masked to zero;
* ``p_in[b,l] == 0`` -- ``p_sq[b,l] == 0`` makes ``u*`` zero for that ``l``.

No ``nan``/``inf`` can be introduced: every build-time tensor is finite
(verified in ``fg4c_probe.py``), and ``ratio[b,l]`` is 0 whenever
``p_in[b,l] == 0`` (t_tilde is only fitted for positive power), so the
``xpm_sum`` bracket stays finite.

The one thing this reordering does change is float summation order, so the
1e-12 relative-deviation budget is checked numerically, not assumed -- see
``numeric_check`` in the emitted JSON.

Usage
-----
    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \
        $HOME/miniconda3/envs/rl/bin/python scripts/fg4c_batch_knee.py \
        --out results/fg4c_batch_knee.json

All timing is same-process interleaved CPU/GPU (the FG-4a/FG-4b discipline), and
``loadavg`` plus GPU utilisation/VRAM are sampled around every run.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import network_nlo_eval  # noqa: E402
from network_nlo_eval.core.spectrum import SpectrumGrid  # noqa: E402
from network_nlo_eval.models import numba_kernels as nk  # noqa: E402
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN  # noqa: E402
from network_nlo_eval.network.elements import FiberSpanConfig  # noqa: E402

H_PLANCK = 6.62607015e-34
SPAN_LENGTH_M = 100e3
N_STEPS = 20
CENTER_FREQUENCY_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9

#: (N, channel spacing). N=230 is the current C+L fixed grid. 900 and 2100 are
#: the S+C+L+U slot-level scales named in the brief (~45 and ~52 THz of
#: bandwidth respectively), so the Raman band edges still see real neighbours.
GRID_SPECS: tuple[tuple[int, float], ...] = ((230, 50e9), (900, 50e9), (2100, 25e9))
BATCH_SIZES: tuple[int, ...] = (1, 4, 16, 64, 256, 1024)

#: FG-4a/FG-4b budget, carried forward unchanged.
MAX_REL_DEVIATION = 1e-12


# -----------------------------------------------------------------------------
# telemetry
# -----------------------------------------------------------------------------
def read_loadavg() -> list[float]:
    try:
        return [float(v) for v in Path("/proc/loadavg").read_text().split()[:3]]
    except (OSError, ValueError):  # pragma: no cover
        return [float("nan")] * 3


def read_gpu() -> dict[str, Any]:
    """GPU utilisation / VRAM in use, plus any other compute process on the card.

    Other tenants matter: a contended reading is not a measurement, so the
    contention is recorded rather than silently averaged into the curve.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        util, used, total = (part.strip() for part in out.splitlines()[0].split(","))
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        others = [ln for ln in procs.splitlines() if ln.strip()] if procs else []
        return {
            "utilization_pct": float(util),
            "memory_used_mib": float(used),
            "memory_total_mib": float(total),
            "compute_apps": others,
            "foreign_process_count": len(others),
        }
    except (subprocess.SubprocessError, ValueError, IndexError) as exc:  # pragma: no cover
        return {"error": str(exc)}


def _git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        return "unknown"


# -----------------------------------------------------------------------------
# scenario construction
# -----------------------------------------------------------------------------
class Scenario:
    """All the build-time constants for one N, shared by every B and device.

    The NLI/G matrices depend only on the frequency grid and the span physics,
    never on the candidate powers, so one build serves the whole sweep. That is
    also what the real batched path would do: pay the O(N^2) build once, then
    amortise it over B.
    """

    def __init__(self, n_ch: int, spacing_hz: float, span_length_m: float = SPAN_LENGTH_M) -> None:
        self.n_ch = n_ch
        self.span_length_m = span_length_m
        grid = SpectrumGrid(
            num_channels=n_ch, channel_spacing_hz=spacing_hz, center_frequency_hz=CENTER_FREQUENCY_HZ
        )
        fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
        ev = MultiBandISRSGN(
            grid=grid, ref_fiber_config=fiber, symbol_rate_hz=SYMBOL_RATE_HZ, phi_kurtosis=0.0
        )
        self.evaluator = ev
        g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)
        spm0, spm1, spm2, gamma_ii_sq, gamma_sq, w0, w1, w2 = nk.nli_tensors(
            n_ch, 1, ev.symbol_rate_hz_channels, ev.f_rel_hz, ev.lambdas_m,
            ev.a_eff_m2_channels, ev.alpha_power_npm_channels, span_length_m,
            ev.beta2_s2_m, ev.beta3_s3_m, ev.n2_m2_w, ev.phi_kurtosis,
        )
        self.kernel_common = dict(
            f_rel_hz=ev.f_rel_hz, f_abs_hz=ev.f_abs_hz, lambdas_m=ev.lambdas_m,
            alpha_power_npm=ev.alpha_power_npm_channels, a_eff_m2=ev.a_eff_m2_channels,
            beta2_s2_m=ev.beta2_s2_m, beta3_s3_m=ev.beta3_s3_m, n2_m2_w=ev.n2_m2_w,
            nf_lin_channels=ev.nf_lin_channels, symbol_rate_hz=ev.symbol_rate_hz_channels,
            n_effective_spans=1, cr_w_m_hz=ev.cr_w_m_hz, delta_f_co_hz=ev.delta_f_co_hz,
            phi_kurtosis=ev.phi_kurtosis,
        )
        self.tensor_kwargs = dict(
            nli_spm0=spm0, nli_spm1=spm1, nli_spm2=spm2, nli_gamma_ii_sq=gamma_ii_sq,
            nli_gamma_sq=gamma_sq, nli_w0=w0, nli_w1=w1, nli_w2=w2,
        )
        # --- batched-path constants (numpy) ---
        self.g_gain = np.ascontiguousarray(g_gain, dtype=np.float64)
        self.g_loss = np.ascontiguousarray(g_loss, dtype=np.float64)
        self.alpha = np.ascontiguousarray(ev.alpha_power_npm_channels, dtype=np.float64)
        self.spm0 = np.ascontiguousarray(spm0, dtype=np.float64)
        self.spm1 = np.ascontiguousarray(spm1, dtype=np.float64)
        self.spm2 = np.ascontiguousarray(spm2, dtype=np.float64)
        self.gamma_ii_sq = np.ascontiguousarray(gamma_ii_sq, dtype=np.float64)
        # Diagonal zeroed: the per-pair mask excludes self pairs (i == l).
        #
        # Stored **transposed**, i.e. m0[l,i] = gamma_sq[i,l]*w0[i,l]. The batch
        # contraction wants S[b,i] = sum_l u0[b,l] * m0[i,l], which is
        # ``u0 @ m0.T``. Keeping the transposed form here means the hot loop is a
        # plain (B,N) @ (N,N) product with no per-call transpose.
        eye = np.eye(n_ch, dtype=np.float64)
        self.m0 = np.ascontiguousarray((gamma_sq * w0 * (1.0 - eye)).T)
        self.m1 = np.ascontiguousarray((gamma_sq * w1 * (1.0 - eye)).T)
        self.m2 = np.ascontiguousarray((gamma_sq * w2 * (1.0 - eye)).T)
        self.rs = np.ascontiguousarray(ev.symbol_rate_hz_channels, dtype=np.float64)
        self.f_abs = np.ascontiguousarray(ev.f_abs_hz, dtype=np.float64)
        self.nf_lin = np.ascontiguousarray(ev.nf_lin_channels, dtype=np.float64)
        self._lin_decay = np.exp(-self.alpha * span_length_m)
        self._one_minus_decay = 1.0 - self._lin_decay

    def candidates(self, batch: int) -> np.ndarray:
        """B distinct candidate power profiles, shaped (B, N).

        Modelled on what ``scan.py::verify_for_scan`` really does: each candidate
        is the *same* link occupancy with one extra channel raised to launch
        power, so the batch is a set of profiles that differ in a single
        channel each. Channel indices are spread across the whole band (not
        adjacent) so the batch is not accidentally degenerate.
        """
        base = np.full((batch, self.n_ch), 1e-3, dtype=np.float64)
        # A 1/8 empty slice, exactly as in the FG-4a/FG-4b profiled workload, so
        # the zero-power guards are on the measured path.
        base[:, ::7] = 0.0
        base[:, 0] = 0.0
        base[:, -1] = 0.0
        step = max(1, self.n_ch // max(batch, 1))
        for b in range(batch):
            base[b, (b * step) % self.n_ch] = 1e-3
        return np.ascontiguousarray(base)


# -----------------------------------------------------------------------------
# CPU production path: the real per-candidate kernel, B times
# -----------------------------------------------------------------------------
def cpu_serial(scen: Scenario, profiles: np.ndarray) -> np.ndarray:
    """Today's path: B independent calls to the production single-span kernel."""
    out = np.empty((profiles.shape[0], 4, profiles.shape[1]), dtype=np.float64)
    for b in range(profiles.shape[0]):
        p_out, s2_spm, s2_xpm, s2_ase = nk._calc_span_noise_and_power_jit(
            np.ascontiguousarray(profiles[b]), scen.span_length_m,
            g_gain=scen.g_gain, g_loss=scen.g_loss, **scen.kernel_common, **scen.tensor_kwargs,
        )
        out[b] = np.stack([p_out, s2_spm, s2_xpm, s2_ase])
    return out


# -----------------------------------------------------------------------------
# Batched paths (numpy on CPU, torch on GPU) - same math, (B,N) layout
# -----------------------------------------------------------------------------
def _batched_numpy(scen: Scenario, P: np.ndarray) -> np.ndarray:
    """Vectorised single-span kernel over a (B,N) power batch (numpy/CPU)."""
    p = P.copy()
    dz = scen.span_length_m / N_STEPS
    half = 0.5 * dz
    six = dz / 6.0
    gT_gain = scen.g_gain.T  # (N,N) -> want g @ p[b]
    gT_loss = scen.g_loss.T
    for _ in range(N_STEPS):
        def dp(x: np.ndarray) -> np.ndarray:
            return (x @ gT_gain - x @ gT_loss - scen.alpha) * x
        k1 = dp(p)
        k2 = dp(p + half * k1)
        k3 = dp(p + half * k2)
        k4 = dp(p + dz * k3)
        p += six * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        np.maximum(p, 0.0, out=p)  # baseline clamps negatives to 0
    p_out = p

    # --- NLI: t_tilde is the only data-dependent step ---
    p_lin_out = P * scen._lin_decay
    t_tilde = np.zeros_like(P)
    np.divide(p_out, p_lin_out, out=t_tilde, where=p_lin_out > 0)
    t_tilde = np.where(p_lin_out > 0, (t_tilde - 1.0) / scen._one_minus_decay, 0.0)
    big_t = 1.0 + t_tilde
    ratio = -t_tilde / big_t
    ratio_sq = ratio * ratio

    p_sq = P * P
    p_cube = p_sq * P
    spm_sum = scen.spm0 + 2.0 * ratio * scen.spm1 + ratio_sq * scen.spm2
    spm_pre = ((16.0 / 27.0) * (scen.gamma_ii_sq * p_cube / (scen.rs * scen.rs))) * (big_t * big_t)
    sigma_spm = np.where(P > 0, np.maximum(spm_pre * spm_sum, 0.0), 0.0)

    # The ``big_t**2`` factor sits **inside** the l-sum, not outside it: in the
    # baseline ``pre = ((32/27) * pre) * (big_t * big_t)`` multiplies an (N,N)
    # array by an (N,) vector, which broadcasts along the *perturbation* axis l.
    # Hoisting it out of the sum (the obvious-looking reading) is a silent 2x
    # error -- the numerics below are what caught it.
    u0 = p_sq * (big_t * big_t) / scen.rs
    u1 = u0 * ratio
    u2 = u0 * ratio_sq
    acc = u0 @ scen.m0 + 2.0 * (u1 @ scen.m1) + u2 @ scen.m2  # (B,N)
    sigma_xpm = np.where(
        P > 0, np.maximum((32.0 / 27.0) * P * acc, 0.0), 0.0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        gain = np.where(p_out > 0, P / np.where(p_out > 0, p_out, 1.0), 0.0)
    sigma_ase = np.where(
        (P > 0) & (p_out > 0),
        1.0 * scen.nf_lin * H_PLANCK * scen.f_abs * np.maximum(gain - 1.0, 0.0) * scen.rs,
        0.0,
    )
    return np.stack([p_out, sigma_spm, sigma_xpm, sigma_ase], axis=1)


def _batched_torch(scen: Scenario, T: Any) -> Any:
    """The same batched math on torch tensors (device/dtype agnostic)."""
    import torch

    p = T["P"].clone()
    dz = scen.span_length_m / N_STEPS
    half = T["scalar"](0.5 * dz)
    six = T["scalar"](dz / 6.0)
    alpha = T["alpha"]
    gT_gain = T["gT_gain"]
    gT_loss = T["gT_loss"]

    def dp(x: Any) -> Any:
        return (x @ gT_gain - x @ gT_loss - alpha) * x

    for _ in range(N_STEPS):
        k1 = dp(p)
        k2 = dp(p + half * k1)
        k3 = dp(p + half * k2)
        k4 = dp(p + dz * k3)
        p = p + six * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        p = p.clamp(min=0.0)
    p_out = p

    P = T["P"]
    lin_decay = T["lin_decay"]
    one_minus = T["one_minus_decay"]
    p_lin_out = P * lin_decay
    ratio_r = torch.where(p_lin_out > 0, p_out / torch.where(p_lin_out > 0, p_lin_out, one_minus), one_minus)
    t_tilde = torch.where(p_lin_out > 0, (ratio_r - 1.0) / one_minus, T["zero"])
    big_t = 1.0 + t_tilde
    ratio = -t_tilde / big_t
    ratio_sq = ratio * ratio

    p_sq = P * P
    p_cube = p_sq * P
    spm_sum = T["spm0"] + 2.0 * ratio * T["spm1"] + ratio_sq * T["spm2"]
    spm_pre = ((16.0 / 27.0) * (T["gamma_ii_sq"] * p_cube / (T["rs"] * T["rs"]))) * (big_t * big_t)
    sigma_spm = torch.where(P > 0, (spm_pre * spm_sum).clamp(min=0.0), T["zero"])

    # ``big_t**2`` belongs inside the l-sum (it broadcasts along the
    # perturbation axis in the baseline), so it rides on u0.
    u0 = p_sq * (big_t * big_t) / T["rs"]
    u1 = u0 * ratio
    u2 = u0 * ratio_sq
    acc = u0 @ T["m0"] + 2.0 * (u1 @ T["m1"]) + u2 @ T["m2"]
    sigma_xpm = torch.where(
        P > 0, (((32.0 / 27.0) * P * acc).clamp(min=0.0)), T["zero"]
    )

    gain = torch.where(p_out > 0, P / torch.where(p_out > 0, p_out, T["one"]), T["zero"])
    sigma_ase = torch.where(
        (P > 0) & (p_out > 0),
        T["scalar"](1.0) * T["nf_lin"] * T["h"] * T["f_abs"] * (gain - 1.0).clamp(min=0.0) * T["rs"],
        T["zero"],
    )
    return torch.stack([p_out, sigma_spm, sigma_xpm, sigma_ase], dim=1)


def make_torch_pack(scen: Scenario, profiles: np.ndarray, device: str, dtype: Any) -> dict[str, Any]:
    import torch

    def t(arr: np.ndarray) -> Any:
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)

    zero = torch.zeros((), dtype=dtype, device=device)
    one = torch.ones((), dtype=dtype, device=device)
    return {
        "P": t(profiles),
        "gT_gain": t(scen.g_gain.T),
        "gT_loss": t(scen.g_loss.T),
        "alpha": t(scen.alpha),
        "spm0": t(scen.spm0), "spm1": t(scen.spm1), "spm2": t(scen.spm2),
        "gamma_ii_sq": t(scen.gamma_ii_sq),
        "m0": t(scen.m0), "m1": t(scen.m1), "m2": t(scen.m2),
        "rs": t(scen.rs), "f_abs": t(scen.f_abs), "nf_lin": t(scen.nf_lin),
        "lin_decay": t(scen._lin_decay), "one_minus_decay": t(scen._one_minus_decay),
        "h": torch.as_tensor(H_PLANCK, dtype=dtype, device=device),
        "zero": zero, "one": one,
        "scalar": lambda v: torch.as_tensor(v, dtype=dtype, device=device),
    }


# -----------------------------------------------------------------------------
# timing
# -----------------------------------------------------------------------------
def time_call(fn: Any, rounds: int, warmup: int = 1) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(rounds):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    srt = sorted(samples)
    return {
        "min_s": srt[0],
        "median_s": srt[len(srt) // 2],
        "mean_s": sum(samples) / len(samples),
        "spread_ratio": srt[-1] / srt[0] if srt[0] > 0 else float("nan"),
        "samples": len(samples),
    }


def interleaved(
    fa: Any, fb: Any, rounds: int, *, warmup: int = 1
) -> tuple[dict[str, float], dict[str, float]]:
    """Alternate the two arms round-robin, flipping the order each round.

    Same-process interleaving is the authoritative discipline from FG-4a/FG-4b:
    a CPU reading taken minutes apart from a GPU reading is dominated by host
    load drift, not by the thing being measured.

    ``warmup`` is deliberately separate from ``rounds``: a cold call carries
    Numba cache-load and BLAS-init cost that is an order of magnitude larger
    than the steady-state number (a cold single-candidate call at N=230 measures
    13.0 ms against FG-4b's 1.096 ms warm figure), so warmup samples are
    discarded rather than averaged in.
    """
    for _ in range(warmup):
        fa(); fb()
    sa: list[float] = []
    sb: list[float] = []
    for r in range(rounds):
        order = ((sa, fa), (sb, fb)) if r % 2 == 0 else ((sb, fb), (sa, fa))
        for acc, fn in order:
            start = time.perf_counter()
            fn()
            acc.append(time.perf_counter() - start)
    def stat(xs: list[float]) -> dict[str, float]:
        srt = sorted(xs)
        return {
            "min_s": srt[0], "median_s": srt[len(srt) // 2],
            "spread_ratio": srt[-1] / srt[0] if srt[0] > 0 else float("nan"),
            "samples": len(srt),
        }
    return stat(sa), stat(sb)


def worst_relative(expected: np.ndarray, actual: np.ndarray) -> float:
    exp_arr = np.asarray(expected, dtype=np.float64).ravel()
    act_arr = np.asarray(actual, dtype=np.float64).ravel()
    both = np.isfinite(exp_arr) & np.isfinite(act_arr)
    if not both.any():
        return 0.0
    diff = np.abs(exp_arr[both] - act_arr[both])
    denom = np.abs(exp_arr[both])
    return float(np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0).max())


# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "fg4c_batch_knee.json")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--target-seconds", type=float, default=2.0,
        help="Per-arm sampling budget; the round count adapts down to respect it.",
    )
    parser.add_argument("--numeric-batches", type=int, nargs="*", default=[1, 4, 16, 64])
    parser.add_argument("--max-batch", type=int, default=1024)
    parser.add_argument(
        "--mem-fraction", type=float, default=0.5,
        help="Cap this process at this fraction of VRAM; the card is shared.",
    )
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("CUDA unavailable; cannot run the GPU arm.", file=sys.stderr)
        return 2
    torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 honest
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction, 0)
    device = "cuda:0"

    batches = [b for b in BATCH_SIZES if b <= args.max_batch]
    rows: list[dict[str, Any]] = []
    numeric: list[dict[str, Any]] = []
    env_start = {"loadavg": read_loadavg(), "gpu": read_gpu()}

    for n_ch, spacing in GRID_SPECS:
        print(f"\n=== N={n_ch} (spacing {spacing/1e9:.0f} GHz) ===", flush=True)
        load_before = read_loadavg()
        scen = Scenario(n_ch, spacing)
        gpu_before = read_gpu()

        # Scenario-level warmup. JIT cache load, BLAS init and cuBLAS handle
        # creation are one-offs per (N, device) and run 10x+ the steady-state
        # cost; they must not land in any sample.
        _p1 = scen.candidates(1)
        cpu_serial(scen, _p1)
        _batched_numpy(scen, _p1)
        _pk = make_torch_pack(scen, _p1, device, torch.float64)
        _batched_torch(scen, _pk)
        torch.cuda.synchronize()
        del _pk
        torch.cuda.empty_cache()

        def pick_rounds(cost_per_sample: float, cap: int) -> int:
            """Round count that keeps one arm's sampling near --target-seconds."""
            if not cost_per_sample > 0:
                return cap
            return max(1, min(cap, int(args.target_seconds / cost_per_sample)))

        # Numeric check first, on the same scenario, so the timed curve cannot
        # be quoting numbers for a different computation.
        for b in args.numeric_batches:
            prof = scen.candidates(b)
            ref = cpu_serial(scen, prof)
            cpu_b = _batched_numpy(scen, prof)
            pack64 = make_torch_pack(scen, prof, device, torch.float64)
            gpu64 = _batched_torch(scen, pack64).double().cpu().numpy()
            pack32 = make_torch_pack(scen, prof, device, torch.float32)
            gpu32 = _batched_torch(scen, pack32).double().cpu().numpy()
            dev_cpu = worst_relative(ref, cpu_b)
            dev_gpu = worst_relative(ref, gpu64)
            dev_32 = worst_relative(ref, gpu32)
            numeric.append({
                "n_ch": n_ch, "batch": b,
                "cpu_batched_worst_rel": dev_cpu,
                "gpu_f64_worst_rel": dev_gpu,
                "gpu_f32_worst_rel": dev_32,
                "gpu_f64_within_1e-12": dev_gpu <= MAX_REL_DEVIATION,
                "gpu_f32_within_1e-12": dev_32 <= MAX_REL_DEVIATION,
            })
            print(
                f"  numeric B={b:5d}: cpu_batch {dev_cpu:.3e}  gpu_f64 {dev_gpu:.3e}  "
                f"gpu_f32 {dev_32:.3e}", flush=True,
            )
            del pack64, pack32

        for b in batches:
            prof = scen.candidates(b)
            pack = make_torch_pack(scen, prof, device, torch.float64)
            pack32 = make_torch_pack_cached(scen, prof, device, torch.float32)

            # Pilots (already warm) size the round count per arm, so a cheap
            # arm keeps its statistics while an expensive one degrades to fewer
            # samples instead of blowing the time budget. The sample count that
            # actually ran is recorded in the row.
            t0 = time.perf_counter(); cpu_serial(scen, _p1); cpu_unit = (time.perf_counter() - t0)
            t0 = time.perf_counter(); _batched_numpy(scen, _p1); cpu_b_unit = (time.perf_counter() - t0)
            t0 = time.perf_counter(); _batched_torch(scen, pack); torch.cuda.synchronize()
            gpu_unit = (time.perf_counter() - t0) / max(b, 1)

            r_serial = pick_rounds(cpu_unit * b, args.rounds)
            r_batch = pick_rounds(cpu_b_unit * b, args.rounds)
            r_gpu = pick_rounds(gpu_unit * b, args.rounds)
            gpu_t0 = read_gpu()

            def cpu_arm() -> None:
                cpu_serial(scen, prof)

            def gpu_arm() -> None:
                _batched_torch(scen, pack)
            torch.cuda.synchronize()
            # The interleaved pair runs ``r_serial`` rounds, not
            # ``max(r_serial, r_gpu)``: the GPU arm is two to three orders of
            # magnitude cheaper per sample, so letting it set the round count
            # makes the *CPU* arm run its expensive B-loop 7x over for nothing
            # (at N=2100, B=1024 that is ~2 h instead of ~8 min). The GPU gets
            # its own wide timing below instead, and both counts are recorded.
            cpu_row, gpu_row = interleaved(cpu_arm, gpu_arm, r_serial, warmup=0)
            torch.cuda.synchronize()
            gpu_t1 = read_gpu()
            gpu_wide = time_call(gpu_arm, r_gpu, warmup=0)
            torch.cuda.synchronize()

            row = {
                "n_ch": n_ch,
                "batch": b,
                "cpu_serial": cpu_row,
                "cpu_batched": time_call(lambda: _batched_numpy(scen, prof), r_batch, warmup=0),
                "gpu_f64": gpu_row,
                "gpu_f64_wide": gpu_wide,
                "gpu_f64_per_candidate_s": gpu_row["min_s"] / b,
                "gpu_f32": time_call(lambda: _batched_torch(scen, pack32), r_gpu, warmup=0),
            }
            row["cpu_serial_per_candidate_s"] = cpu_row["min_s"] / b
            row["cpu_batched_per_candidate_s"] = row["cpu_batched"]["min_s"] / b
            row["gpu_f64_speedup_vs_cpu_serial"] = (
                row["cpu_serial"]["min_s"] / gpu_row["min_s"]
            )
            row["cpu_batched_speedup_vs_cpu_serial"] = (
                cpu_row["min_s"] / row["cpu_batched"]["min_s"]
            )
            row["gpu_util_pct"] = max(gpu_t0.get("utilization_pct", 0.0), gpu_t1.get("utilization_pct", 0.0))
            row["vram_peak_mib"] = gpu_t1.get("memory_used_mib", float("nan"))
            row["loadavg_during"] = read_loadavg()
            row["rounds_cpu_serial"] = r_serial
            row["rounds_cpu_batched"] = r_batch
            row["rounds_gpu"] = r_gpu
            rows.append(row)
            print(
                f"  B={b:5d}: cpu_serial {row['cpu_serial']['min_s']*1e3:9.3f} ms "
                f"({row['cpu_serial_per_candidate_s']*1e6:9.2f} us/cand) | "
                f"cpu_batch {row['cpu_batched']['min_s']*1e3:9.3f} ms | "
                f"gpu {gpu_row['min_s']*1e3:9.3f} ms ({row['gpu_f64_per_candidate_s']*1e6:9.2f} us/cand) "
                f"| gpu/cpu {row['gpu_f64_speedup_vs_cpu_serial']:7.3f}x | "
                f"vram {row['vram_peak_mib']:.0f} MiB", flush=True,
            )
            del pack
            torch.cuda.empty_cache()

        rows_for_n = [r for r in rows if r["n_ch"] == n_ch]
        knee = find_knee(rows_for_n)
        print(f"  knee: {knee}", flush=True)
        load_after = read_loadavg()
        print(f"  loadavg {load_before[0]:.2f} -> {load_after[0]:.2f}", flush=True)

        for r in rows_for_n:
            r["knee_for_this_n"] = knee
        scen_rows = [r for r in rows if r["n_ch"] == n_ch]
        scen_rows[0].setdefault("n_loadavg_before", load_before[0])
        scen_rows[0].setdefault("n_loadavg_after", load_after[0])
        scen_rows[0].setdefault("n_gpu_before", gpu_before)
        scen_rows[0]["knee"] = knee

        # Checkpoint after every N. A full sweep is tens of minutes on a shared
        # host; without this a kill near the end loses everything measured so far.
        _partial_knees = {
            str(n): find_knee([r for r in rows if r["n_ch"] == n]) for n, _ in GRID_SPECS
            if any(r["n_ch"] == n for r in rows)
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "schema_version": 1,
            "status": "partial" if len(_partial_knees) < len(GRID_SPECS) else "complete",
            "method": "same-process interleaved CPU/GPU, min-sample; complete single-span kernel",
            "config": {"grid_specs": [{"n_ch": n, "spacing_hz": s} for n, s in GRID_SPECS],
                       "batch_sizes": batches, "span_length_m": SPAN_LENGTH_M,
                       "n_steps": N_STEPS, "rounds": args.rounds,
                       "max_rel_deviation_budget": MAX_REL_DEVIATION,
                       "gpu_mem_fraction_cap": args.mem_fraction},
            "numeric_check": numeric,
            "knee_by_n": _partial_knees,
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"  checkpointed {len(rows)} rows -> {args.out}", flush=True)

    payload = {
        "schema_version": 1,
        "method": (
            "same-process interleaved CPU/GPU, order flipped each round, min-sample ratio; "
            "complete single-span kernel (RK4/SRS ODE + tensorised NLI + ASE), not a GEMV alone"
        ),
        "provenance": {
            "git_rev": _git_rev(),
            "package_path": str(Path(network_nlo_eval.__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "cpu_count": os.cpu_count(),
        },
        "config": {
            "grid_specs": [{"n_ch": n, "spacing_hz": s} for n, s in GRID_SPECS],
            "batch_sizes": batches,
            "span_length_m": SPAN_LENGTH_M,
            "n_steps": N_STEPS,
            "rounds": args.rounds,
            "max_rel_deviation_budget": MAX_REL_DEVIATION,
            "gpu_mem_fraction_cap": args.mem_fraction,
        },
        "env_start": env_start,
        "env_end": {"loadavg": read_loadavg(), "gpu": read_gpu()},
        "numeric_check": numeric,
        "rows": rows,
    }
    knees = {}
    for n_ch, _ in GRID_SPECS:
        n_rows = [r for r in rows if r["n_ch"] == n_ch]
        knees[str(n_ch)] = find_knee(n_rows)
    payload["knee_by_n"] = knees
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"\nwrote {args.out}")
    print("knees:", json.dumps(knees, indent=2))
    return 0


def make_torch_pack_cached(scen: Scenario, profiles: np.ndarray, device: str, dtype: Any) -> Any:
    """fp32 packs live on the scenario, keyed by batch size.

    Keyed off the scenario object rather than ``id(scen)``: CPython reuses
    ``id()`` after collection, so an id-keyed cache can hand a *stale* pack
    belonging to a previous, already-freed N -- which would silently time the
    wrong shapes.
    """
    cache = getattr(scen, "_fp32_packs", None)
    if cache is None:
        cache = {}
        scen._fp32_packs = cache  # type: ignore[attr-defined]
    key = profiles.shape[0]
    pack = cache.get(key)
    if pack is None:
        import torch

        pack = make_torch_pack(scen, profiles, device, torch.float32)
        cache[key] = pack
    return pack


def find_knee(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Smallest B at which the GPU's per-candidate time is below the CPU's.

    "CPU" is the *production* per-candidate path (``cpu_serial``), because that
    is the thing a GPU path would have to beat in the real RL loop. The
    batched-numpy arm is reported alongside, since if plain CPU batching
    captures the same win the GPU is redundant.
    """
    ordered = sorted(rows, key=lambda r: r["batch"])
    out: dict[str, Any] = {
        "gpu_beats_cpu_serial_from_batch": None,
        "cpu_batched_beats_cpu_serial_from_batch": None,
        "first_gpu_win_ratio": None,
        "best_gpu_speedup_vs_cpu_serial": None,
        "best_gpu_speedup_vs_cpu_batched": None,
        "per_batch": [],
    }
    for r in ordered:
        g = r["gpu_f64_per_candidate_s"]
        c = r["cpu_serial_per_candidate_s"]
        cb = r["cpu_batched_per_candidate_s"]
        wins = g < c
        if wins and out["gpu_beats_cpu_serial_from_batch"] is None:
            out["gpu_beats_cpu_serial_from_batch"] = r["batch"]
            out["first_gpu_win_ratio"] = c / g
        if cb < c and out["cpu_batched_beats_cpu_serial_from_batch"] is None:
            out["cpu_batched_beats_cpu_serial_from_batch"] = r["batch"]
        if out["best_gpu_speedup_vs_cpu_serial"] is None or c / g > out["best_gpu_speedup_vs_cpu_serial"]:
            out["best_gpu_speedup_vs_cpu_serial"] = c / g
        if out["best_gpu_speedup_vs_cpu_batched"] is None or cb / g > out["best_gpu_speedup_vs_cpu_batched"]:
            out["best_gpu_speedup_vs_cpu_batched"] = cb / g
        out["per_batch"].append({
            "batch": r["batch"],
            "gpu_us_per_candidate": g * 1e6,
            "cpu_serial_us_per_candidate": c * 1e6,
            "cpu_batched_us_per_candidate": cb * 1e6,
            "gpu_over_cpu_serial": g / c,
            "gpu_over_cpu_batched": g / cb,
        })
    return out


if __name__ == "__main__":
    raise SystemExit(main())
