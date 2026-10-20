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
from typing import Any

import numpy as np

from network_nlo_eval.benchmarks.cases import (
    POWER_SWEEP_DBM,
    BenchCase,
    build_case,
    build_path,
    grid_for,
    launch_profile,
)
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import create_model

__all__ = [
    "ModelRun",
    "accuracy_vs_reference",
    "make_model",
    "run_case",
    "sweep_snr",
    "time_calls",
]


def make_model(name: str, case: BenchCase, **options: Any) -> Any:
    """按名称构造算例对应的模型实例。

    Args:
        name: 注册表键。
        case: 基准算例，提供网格与参考光纤。
        **options: 传给该模型的专属选项。

    Returns
    -------
        模型实例。
    """
    fiber = build_path(case).fiber_configs[0]
    return create_model(name, grid_for(case), fiber, symbol_rate_hz=case.symbol_rate_hz, **options)


@dataclass(frozen=True)
class ModelRun:
    """一次算例运行的完整读数。

    Attributes
    ----------
        model: 模型名。
        case: 算例名。
        snr_db: ``(len(powers), N)`` 逐功率点的逐信道 SNR。
        powers_dbm: 功率扫描点。
        cold_ms: 首次调用耗时 (ms)，含 JIT 编译与惰性常量构建。
        warm_median_ms: 热调用耗时中位数 (ms)。
        warm_p95_ms: 热调用耗时 p95 (ms)。
        qps: 热调用吞吐 (次/秒)。
        peak_kib: 热调用期间的 ``tracemalloc`` 峰值 (KiB)。
        span_count: 物理跨段数。
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

    def to_dict(self) -> dict[str, Any]:
        """JSON 兼容表示；``snr_db`` 保留完整精度。"""
        return {
            "model": self.model,
            "case": self.case,
            "powers_dbm": list(self.powers_dbm),
            "snr_db": [[float(value) if np.isfinite(value) else None for value in row] for row in self.snr_db],
            "cold_ms": self.cold_ms,
            "warm_median_ms": self.warm_median_ms,
            "warm_p95_ms": self.warm_p95_ms,
            "qps": self.qps,
            "peak_kib": self.peak_kib,
            "span_count": self.span_count,
        }


def time_calls(model: Any, path: Any, power: NDArrayFloat, *, reps: int = 25) -> tuple[float, float, float, float]:
    """测量冷/热调用耗时与内存峰值。

    计时与内存分两趟测：``tracemalloc`` 自身有可观开销，混在计时段里会把延迟
    抬高数倍，得到一个不可比的"速度"数字。

    Args:
        model: 模型实例。
        path: 光路。
        power: 入纤功率谱 ``(N,)``。
        reps: 热调用次数。

    Returns
    -------
        ``(cold_ms, warm_median_ms, warm_p95_ms, peak_kib)``。
    """
    started = time.perf_counter()
    model.evaluate_path(path, power)
    cold_ms = (time.perf_counter() - started) * 1e3

    samples: list[float] = []
    for _ in range(reps):
        started = time.perf_counter()
        model.evaluate_path(path, power)
        samples.append((time.perf_counter() - started) * 1e3)

    tracemalloc.start()
    model.evaluate_path(path, power)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    timings = np.asarray(samples, dtype=np.float64)
    return cold_ms, float(np.median(timings)), float(np.percentile(timings, 95)), peak / 1024.0


def sweep_snr(model: Any, case: BenchCase, powers_dbm: tuple[float, ...] = POWER_SWEEP_DBM) -> NDArrayFloat:
    """在功率扫描点上逐信道算 SNR。

    Args:
        model: 模型实例。
        case: 算例。
        powers_dbm: 功率点。

    Returns
    -------
        ``(len(powers_dbm), N)`` SNR (dB)。
    """
    path = build_path(case)
    rows = [model.evaluate_path(path, launch_profile(case, power)).snr_db for power in powers_dbm]
    return np.vstack(rows)


def run_case(
    model_name: str,
    case_name: str,
    *,
    powers_dbm: tuple[float, ...] = POWER_SWEEP_DBM,
    reps: int = 25,
    options: dict[str, Any] | None = None,
) -> ModelRun:
    """构造模型、扫功率、测速，一次返回。

    Args:
        model_name: 注册表键。
        case_name: 算例名。
        powers_dbm: 功率扫描点。
        reps: 热调用重复次数。
        options: 模型专属选项。

    Returns
    -------
        读数组。
    """
    case = build_case(case_name)
    model = make_model(model_name, case, **(options or {}))
    # 先测速再扫功率：cold 必须是该实例的第一次调用，才包含 JIT 编译与惰性
    # (N, N) 常量构建。测速用中间功率点，避开极低功率下 SNR 为 +inf 的分支差异。
    probe = powers_dbm[len(powers_dbm) // 2]
    cold_ms, median_ms, p95_ms, peak_kib = time_calls(model, build_path(case), launch_profile(case, probe), reps=reps)
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
    )


def accuracy_vs_reference(
    run: ModelRun,
    reference: ModelRun,
    *,
    power_index: int | None = None,
) -> dict[str, float]:
    """比较两个模型在相同功率点上的逐信道 SNR。

    Args:
        run: 待评估模型。
        reference: 参考模型（通常为 ``numerical``）。
        power_index: 只比较该功率点；``None`` 时比较全部共同功率点。

    Returns
    -------
        ``mean_db`` / ``rms_db`` / ``max_abs_db`` / ``p95_abs_db`` 与样本数。
        非有限值（被阻断或噪声为零）在统计前剔除，并单独报 ``dropped`` 计数。
    """
    if run.powers_dbm != reference.powers_dbm:
        raise ValueError("runs must share the same power sweep to be compared.")
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
