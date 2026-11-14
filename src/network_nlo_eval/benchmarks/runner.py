"""Benchmark runner: accuracy against a reference model plus latency and throughput.

Every model is constructed and driven through the same helpers, so a difference in
the numbers can only come from the model itself:

- same grid (:func:`cases.grid_for`), same reference fiber, same EDFA, same path;
- same launch profiles (:func:`cases.launch_profile`);
- the SNR/speed measurement calls only the public ``evaluate_path``.

Cold latency is the first call (includes Numba JIT and any lazily built (N, N)
matrices); warm latency is the median of repeated calls on the same inputs, which is
the figure that matters for high-frequency QoT access.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.benchmarks.cases import (
    POWER_SWEEP_DBM,
    BenchCase,
    build_case,
    build_path,
    grid_for,
    launch_profile,
)
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.power_conventions import resolve_launch_power_convention

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayFloat

__all__ = [
    "ModelRun",
    "accuracy_vs_reference",
    "make_model",
    "run_case",
    "sweep_snr",
    "time_calls",
]


def make_model(name: str, case: BenchCase, **options: Any) -> Any:
    """Construct a registered model for the benchmark case.

    Parameters
    ----------
    name : str
        Model registry name.
    case : BenchCase
        Case supplying grid and reference fiber.
    **options
        Model-specific constructor options.

    Returns
    -------
    object
        Constructed model.
    """
    fiber = build_path(case).fiber_configs[0]
    return create_model(
        name, grid_for(case), fiber, symbol_rate_hz=case.symbol_rate_hz, **options
    )


@dataclass(frozen=True)
class ModelRun:
    """Measurements from one model and case.

    Attributes
    ----------
    model, case : str
        Model and case names.
    snr_db : numpy.ndarray
        Per-channel SNR for each power point.
    powers_dbm : tuple of float
        Per-channel physical total power sweep in dBm.
    cold_ms, warm_median_ms, warm_p95_ms : float
        Cold, warm median and warm 95th-percentile latency in milliseconds.
    qps : float
        Warm throughput in calls per second.
    peak_kib : float
        Peak traced memory in KiB during a warm call.
    span_count, reps_used : int
        Physical span count and actual number of timed warm repetitions.
    """

    model: str
    case: str
    snr_db: NDArrayFloat
    powers_dbm: tuple[float, ...]
    cold_ms: float
    warm_median_ms: float
    warm_p95_ms: float
    qps: float
    peak_kib: float
    span_count: int
    reps_used: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible measurements, replacing non-finite SNR with None."""
        return {
            "model": self.model,
            "case": self.case,
            "powers_dbm": list(self.powers_dbm),
            "snr_db": [
                [float(value) if np.isfinite(value) else None for value in row]
                for row in self.snr_db
            ],
            "cold_ms": self.cold_ms,
            "warm_median_ms": self.warm_median_ms,
            "warm_p95_ms": self.warm_p95_ms,
            "qps": self.qps,
            "peak_kib": self.peak_kib,
            "span_count": self.span_count,
            "reps_used": self.reps_used,
        }


def time_calls(
    model: Any, path: Any, power: NDArrayFloat, *, reps: int = 25, budget_s: float = 3.0
) -> tuple[float, float, float, float, int]:
    """Measure cold and warm latency and peak traced memory separately.

    Separating memory tracing from timing avoids inflating latency measurements.
    The warm repetition count is adapted to the time budget and returned.

    Parameters
    ----------
    model : object
        QoT model.
    path : object
        Optical path.
    power : numpy.ndarray
        Per-channel input powers.
    reps : int, optional
        Maximum requested warm repetitions, subject to the minimum of three.
    budget_s : float, optional
        Approximate warm-call time budget in seconds.

    Returns
    -------
    tuple
        Cold latency, warm median and p95 latency in milliseconds, peak KiB, and
        the number of warm repetitions.
    """
    started = time.perf_counter()
    model.evaluate_path(path, power)
    cold_ms = (time.perf_counter() - started) * 1e3

    samples: list[float] = []
    started = time.perf_counter()
    model.evaluate_path(path, power)
    first_ms = (time.perf_counter() - started) * 1e3
    reps_used = max(3, min(reps, int(budget_s * 1e3 / max(first_ms, 1e-3))))
    for _ in range(reps_used):
        started = time.perf_counter()
        model.evaluate_path(path, power)
        samples.append((time.perf_counter() - started) * 1e3)

    tracemalloc.start()
    model.evaluate_path(path, power)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    timings = np.asarray(samples, dtype=np.float64)
    return (
        cold_ms,
        float(np.median(timings)),
        float(np.percentile(timings, 95)),
        peak / 1024.0,
        reps_used,
    )


def _require_case_convention(model: Any, case: BenchCase) -> str:
    """Return the case convention and reject mismatched model conventions.

    The same power array must have the same interpretation in model and case.
    """
    convention = resolve_launch_power_convention(case.launch_power_convention)
    model_convention = resolve_launch_power_convention(
        str(getattr(model, "launch_power_convention", "total"))
    )
    if model_convention != convention:
        msg = (
            f"model declares launch_power_convention={model_convention!r} but benchmark case "
            f"{case.name!r} drives it with {convention!r}; pass the matching model option "
            "(e.g. launch_power_convention='per_pol' with signal_polarizations=2) or change "
            "the case convention."
        )
        raise ValueError(msg)
    return convention


def sweep_snr(
    model: Any, case: BenchCase, powers_dbm: tuple[float, ...] = POWER_SWEEP_DBM
) -> NDArrayFloat:
    """Evaluate per-channel SNR at each launch power.

    Parameters
    ----------
    model : object
        Model whose launch convention matches the case.
    case : BenchCase
        Benchmark case.
    powers_dbm : tuple of float, optional
        Per-channel physical total launch powers in dBm.

    Returns
    -------
    numpy.ndarray
        SNR in dB, with shape ``(number of power points, number of channels)``.

    Raises
    ------
    ValueError
        If the model and case power conventions differ.
    """
    convention = _require_case_convention(model, case)
    path = build_path(case)
    rows = [
        model.evaluate_path(
            path, launch_profile(case, power, launch_power_convention=convention)
        ).snr_db
        for power in powers_dbm
    ]
    return np.vstack(rows)


def run_case(
    model_name: str,
    case_name: str,
    *,
    powers_dbm: tuple[float, ...] = POWER_SWEEP_DBM,
    reps: int = 25,
    options: dict[str, Any] | None = None,
) -> ModelRun:
    """Construct a model, measure latency and sweep launch powers.

    Parameters
    ----------
    model_name, case_name : str
        Registry model and benchmark case names.
    powers_dbm : tuple of float, optional
        Per-channel physical total power sweep in dBm.
    reps : int, optional
        Requested warm timing repetitions.
    options : dict or None, optional
        Model-specific constructor options.

    Returns
    -------
    ModelRun
        SNR, latency and memory measurements.

    Raises
    ------
    ValueError
        If the model and case power conventions differ.
    """
    case = build_case(case_name)
    model = make_model(model_name, case, **(options or {}))
    convention = _require_case_convention(model, case)
    # Measure before the sweep so the cold call includes lazy initialization.
    # Use a middle power point to avoid extremely low-power SNR branches.
    probe = powers_dbm[len(powers_dbm) // 2]
    cold_ms, median_ms, p95_ms, peak_kib, reps_used = time_calls(
        model,
        build_path(case),
        launch_profile(case, probe, launch_power_convention=convention),
        reps=reps,
    )
    snr = sweep_snr(model, case, powers_dbm)
    return ModelRun(
        model=model_name,
        case=case_name,
        snr_db=snr,
        powers_dbm=tuple(powers_dbm),
        cold_ms=cold_ms,
        warm_median_ms=median_ms,
        warm_p95_ms=p95_ms,
        qps=1000.0 / median_ms if median_ms > 0 else float("inf"),
        peak_kib=peak_kib,
        span_count=len(build_path(case).fiber_configs),
        reps_used=reps_used,
    )


def accuracy_vs_reference(
    run: ModelRun,
    reference: ModelRun,
    *,
    power_index: int | None = None,
) -> dict[str, float]:
    """Compare SNR measurements against a reference model.

    Parameters
    ----------
    run, reference : ModelRun
        Runs evaluated at identical power points.
    power_index : int or None, optional
        Restrict the comparison to one power point; None compares all points.

    Returns
    -------
    dict of str to float
        Mean, RMS, maximum and 95th-percentile absolute error in dB, together
        with sample and dropped non-finite counts.

    Raises
    ------
    ValueError
        If the power sweeps differ.
    """
    if run.powers_dbm != reference.powers_dbm:
        msg = "runs must share the same power sweep to be compared."
        raise ValueError(msg)
    delta = run.snr_db - reference.snr_db
    if power_index is not None:
        delta = delta[power_index : power_index + 1]
    finite = np.isfinite(delta)
    values = delta[finite]
    dropped = int(delta.size - values.size)
    if values.size == 0:
        empty = float("nan")
        return {
            "samples": 0.0,
            "dropped": float(dropped),
            "mean_db": empty,
            "rms_db": empty,
            "max_abs_db": empty,
            "p95_abs_db": empty,
        }
    return {
        "samples": float(values.size),
        "dropped": float(dropped),
        "mean_db": float(np.mean(values)),
        "rms_db": float(np.sqrt(np.mean(values**2))),
        "max_abs_db": float(np.max(np.abs(values))),
        "p95_abs_db": float(np.percentile(np.abs(values), 95)),
    }
