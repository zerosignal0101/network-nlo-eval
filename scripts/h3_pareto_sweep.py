r"""H3 / ``pcfm_h3_raman`` 精度-效率 Pareto 扫描（只读脚本）。

扫描四个分辨率自由度——``spp_steps``、``raman_steps``、
``XCI_EDGE_QUADRATURE_POINTS``、``SCI_X_SERIES_SWITCH``——在同一算例上的
SNR/NLI 误差与 ``evaluate_path`` 耗时，用来回答两个问题：

1. 论文口径的 H3 默认（``poly_degree=3``、``spp_steps="auto"``、
   ``raman_steps="auto"``、quad=24、SCI 切换 2.0）离"收敛"有多远；
2. 是否存在一个可证明安全的、显式开启的低分辨率点。

参考解是同一算例的最高分辨率设置（spp=96、raman=96、quad=64、SCI=2.0）。
另用 ``numerical_dra`` 在少量单跨点上校验高分辨率 H3 本身没有漂移。

计时口径
--------
``raman_steps`` 会改变粗 ODE 的刚度：步数过低时反向泵浦不动点不收敛，
求解器会耗尽 24 次迭代并得到 NaN，既不准也更慢。为了让"耗时"反映真实
计算量而不是跨段缓存命中率，主计时用一个**新建模型**的首次 ``evaluate_path``
（Numba 已在进程内预热，跨段缓存的命中与否则是真实链路的成本）；另外记录
同一模型第二次调用的缓存命中耗时作参考。

运行::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        PYTHONPATH=src python scripts/h3_pareto_sweep.py

输出 ``results/h3_pareto/sweep.json``。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings

# 单线程计时：必须在 numpy 导入前设置，否则 BLAS 线程池会污染小矩阵的耗时。
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

import network_nlo_eval.models.pcfm as pcfm_module  # noqa: E402
from network_nlo_eval.benchmarks.cases import (  # noqa: E402
    build_case,
    build_path,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model  # noqa: E402
from network_nlo_eval.models.base import MultiSpanOpticalPath  # noqa: E402
from network_nlo_eval.network.elements import EDFAConfig, RamanPumpConfig  # noqa: E402

#: C+L 六泵：与 ``tests/test_pcfm_h3_raman.py::SIX_PUMPS`` 同一配置。
SIX_PUMPS = (
    (1340.0, 0.63),
    (1360.0, 0.63),
    (1427.0, 0.165),
    (1445.0, 0.131),
    (1476.0, 0.125),
    (1505.0, 0.075),
)

#: 扫描算例：波段、信道数与跨数。
CASES = (
    ("c_57ch", "C", 57, 10),
    ("cl_152ch", "C+L", 152, 10),
    ("scl_277ch", "S+C+L", 277, 10),
    ("escl_479ch_1span", "E+S+C+L", 479, 1),
)

#: 泵浦配置。
PUMPS = ("none", "bwd", "fwd")
#: 每信道物理总功率 (dBm)。
POWERS = (0.0, 3.0, 6.0)

#: 单自由度扫描。参考点取每个列表的"最细"值。
KNOBS = {
    "spp_steps": (12, 24, 32, 48, 96),
    "raman_steps": (8, 12, 24, 48, 96),
    "xci_quad": (8, 12, 24, 48, 64),
    "sci_switch": (0.5, 1.0, 2.0, 4.0),
}

#: 参考（收敛）设置。
REFERENCE = {"spp_steps": 96, "raman_steps": 96, "xci_quad": 64, "sci_switch": 2.0}

#: 组合候选点：(标签, spp, raman, quad, sci)。显式整数 / "auto" 都可。
CANDIDATES = (
    ("high", 96, 96, 64, 2.0),
    ("default", "auto", "auto", 24, 2.0),
    ("spp24_raman24", 24, 24, 24, 2.0),
    ("spp32_raman24", 32, 24, 24, 2.0),
    ("spp12_raman24", 12, 24, 24, 2.0),
    ("spp24_raman48", 24, 48, 24, 2.0),
    ("spp32_raman48", 32, 48, 24, 2.0),
)

#: 跨数扫描用的算例。
SPAN_CASE = "cl_152ch"
SPAN_COUNTS = (1, 5, 10)

REPS = 3


def _pumps(kind: str) -> list[RamanPumpConfig]:
    """按方向构造 C+L 六泵；``"none"`` 返回空列表。"""
    if kind == "none":
        return []
    return [
        RamanPumpConfig(
            wavelength_nm=wavelength,
            power_w=power,
            direction="forward" if kind == "fwd" else "backward",
            attenuation_db_per_km=0.25,
        )
        for wavelength, power in SIX_PUMPS
    ]


def _path(case, span_count: int) -> MultiSpanOpticalPath:
    """按算例光纤构造 ``span_count`` 个相同跨的路径。"""
    fiber = (
        build_path(case)
        .fiber_configs[0]
        .model_copy(update={"length_km": case.span_length_km, "max_span_length_km": case.span_length_km})
    )
    edfa = EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)
    return MultiSpanOpticalPath([fiber] * span_count, [edfa] * span_count)


def _build(case, pumps: list[RamanPumpConfig], spp_steps, raman_steps: int):
    """构造一个 H3 模型；``spp_steps`` 允许 ``"auto"``。"""
    options: dict[str, object] = {
        "poly_degree": 3,
        "egn": "auto",
        "edge_fix": True,
        "spp_steps": spp_steps,
        "raman_steps": raman_steps,
    }
    if pumps:
        options["raman_pumps"] = pumps
    return make_model("pcfm_h3_raman", case, **options)


def _with_quad(points: int):
    """临时把 XCI 边缘积分的默认节点数换成 ``points``。

    ``_xci_edge_weight_matrix`` 的 ``n_quad`` 是函数默认值，模块常量改不动它；
    这里替换 ``__defaults__``，用完即恢复。内核按跨长缓存边缘权重，所以必须
    在**新建模型之后**再调用，缓存才不会串。
    """

    class _Patch:
        def __enter__(self):
            self._old = pcfm_module._xci_edge_weight_matrix.__defaults__
            validity, materiality, _ = self._old
            pcfm_module._xci_edge_weight_matrix.__defaults__ = (validity, materiality, int(points))
            return self

        def __exit__(self, *exc):
            pcfm_module._xci_edge_weight_matrix.__defaults__ = self._old
            return False

    return _Patch()


def _with_sci(switch: float):
    """临时替换 ``SCI_X_SERIES_SWITCH``；内核构造时读取该模块全局量。"""

    class _Patch:
        def __enter__(self):
            self._old = pcfm_module.SCI_X_SERIES_SWITCH
            pcfm_module.SCI_X_SERIES_SWITCH = float(switch)
            return self

        def __exit__(self, *exc):
            pcfm_module.SCI_X_SERIES_SWITCH = self._old
            return False

    return _Patch()


def _timed(case, pumps, spp_steps, raman_steps, quad, sci, path, power):
    """新建模型跑一次 ``evaluate_path``，再在同一模型上量缓存命中耗时。

    Returns
    -------
        ``(result, cold_s, cached_median_s)``。
    """
    with _with_quad(quad), _with_sci(sci):
        model = _build(case, pumps, spp_steps, raman_steps)
        started = time.perf_counter()
        result = model.evaluate_path(path, power)
        cold = time.perf_counter() - started
        samples = []
        for _ in range(REPS):
            started = time.perf_counter()
            model.evaluate_path(path, power)
            samples.append(time.perf_counter() - started)
    return result, cold, float(np.median(samples))


def _aggregate_noise(result) -> tuple[float, float, float, float]:
    """返回 ``(spm, xpm, ase, signal)`` 的总功率（W，便于看量级）。"""
    return (
        float(np.sum(np.asarray(result.spm_noise_w))),
        float(np.sum(np.asarray(result.xpm_noise_w))),
        float(np.sum(np.asarray(result.ase_noise_w))),
        float(np.sum(np.asarray(result.signal_power_w))),
    )


def _relative(reference: float, value: float) -> float:
    """相对差；参考为零时用绝对差，避免除零。"""
    if reference == 0.0:
        return abs(value)
    return abs(value - reference) / abs(reference)


def _metrics(reference, value) -> dict[str, float | bool | None]:
    """逐信道误差指标。

    ``snr_max_db`` 是逐信道 SNR 差的最大值（最坏信道），``snr_mean_db`` 是
    平均绝对差；SPM/XPM/ASE 用**总功率**相对差，另外给出逐信道最大相对差
    （只统计参考值非零的信道）。
    """
    snr_ref = np.asarray(reference.snr_db, dtype=np.float64)
    snr_val = np.asarray(value.snr_db, dtype=np.float64)
    finite = np.isfinite(snr_ref) & np.isfinite(snr_val)
    out: dict[str, float | bool | None] = {
        "non_finite": bool(not finite.all()),
        "channels_non_finite": int(np.size(finite) - np.count_nonzero(finite)),
    }
    if np.any(finite):
        diff = np.abs(snr_ref[finite] - snr_val[finite])
        out["snr_max_db"] = float(np.max(diff))
        out["snr_mean_db"] = float(np.mean(diff))
    else:
        out["snr_max_db"] = None
        out["snr_mean_db"] = None
    ref_spm, ref_xpm, ref_ase, _ = _aggregate_noise(reference)
    val_spm, val_xpm, val_ase, _ = _aggregate_noise(value)
    out["spm_rel"] = _relative(ref_spm, val_spm)
    out["xpm_rel"] = _relative(ref_xpm, val_xpm)
    out["ase_rel"] = _relative(ref_ase, val_ase)
    out["nli_rel"] = _relative(ref_spm + ref_xpm, val_spm + val_xpm)
    for name in ("spm_noise_w", "xpm_noise_w", "ase_noise_w"):
        ref_values = np.asarray(getattr(reference, name), dtype=np.float64)
        val_values = np.asarray(getattr(value, name), dtype=np.float64)
        mask = np.isfinite(ref_values) & np.isfinite(val_values) & (ref_values > 0.0)
        if np.any(mask):
            out[f"{name}_max_rel"] = float(np.max(np.abs(val_values[mask] - ref_values[mask]) / ref_values[mask]))
        else:
            out[f"{name}_max_rel"] = None
    return out


def _reference_point(case, pumps, power, span_count):
    """同一算例的最高分辨率参考解。"""
    result, cold, cached = _timed(
        case,
        pumps,
        REFERENCE["spp_steps"],
        REFERENCE["raman_steps"],
        REFERENCE["xci_quad"],
        REFERENCE["sci_switch"],
        _path(case, span_count),
        power,
    )
    return result, {"cold_s": cold, "cached_s": cached}


def _scan_knobs(out: dict) -> list[dict]:
    """单自由度扫描：每次只动一个旋钮，其余固定在收敛值。"""
    records: list[dict] = []
    for name, bands, channels, spans in CASES:
        case = build_case(name)
        path = _path(case, spans)
        for pump_kind in PUMPS:
            pumps = _pumps(pump_kind)
            for power_dbm in POWERS:
                power = launch_profile(case, power_dbm)
                reference, ref_time = _reference_point(case, pumps, power, spans)
                base = {
                    "case": name,
                    "bands": bands,
                    "channels": channels,
                    "spans": spans,
                    "pump": pump_kind,
                    "power_dbm": power_dbm,
                    "reference_cold_s": ref_time["cold_s"],
                }
                for knob, values in KNOBS.items():
                    for value in values:
                        settings = dict(REFERENCE)
                        settings[knob] = value
                        result, cold, cached = _timed(
                            case,
                            pumps,
                            settings["spp_steps"],
                            settings["raman_steps"],
                            settings["xci_quad"],
                            settings["sci_switch"],
                            path,
                            power,
                        )
                        record = dict(base)
                        record.update(
                            {
                                "point": "knob",
                                "knob": knob,
                                "value": value,
                                "cold_s": cold,
                                "cached_s": cached,
                                "speedup": ref_time["cold_s"] / cold if cold > 0 else None,
                            }
                        )
                        record.update(_metrics(reference, result))
                        records.append(record)
                # 默认 auto 规则的位置。
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result, cold, cached = _timed(case, pumps, "auto", "auto", 24, 2.0, path, power)
                record = dict(base)
                record.update(
                    {
                        "point": "default",
                        "knob": "auto",
                        "value": "auto",
                        "cold_s": cold,
                        "cached_s": cached,
                        "speedup": ref_time["cold_s"] / cold if cold > 0 else None,
                    }
                )
                record.update(_metrics(reference, result))
                records.append(record)
                print(
                    f"[knob] {name:18s} {pump_kind:4s} {power_dbm:4.1f} dBm ref={ref_time['cold_s'] * 1e3:8.2f} ms",
                    flush=True,
                )
    out["knob_records"] = records
    return records


def _scan_candidates(out: dict) -> list[dict]:
    """组合候选点：同时改多个旋钮，用来看真实的 Pareto 前沿。"""
    records: list[dict] = []
    for name, bands, channels, spans in CASES:
        case = build_case(name)
        path = _path(case, spans)
        for pump_kind in PUMPS:
            pumps = _pumps(pump_kind)
            for power_dbm in POWERS:
                power = launch_profile(case, power_dbm)
                reference, ref_time = _reference_point(case, pumps, power, spans)
                for label, spp, raman, quad, sci in CANDIDATES:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result, cold, cached = _timed(case, pumps, spp, raman, quad, sci, path, power)
                    record = {
                        "case": name,
                        "bands": bands,
                        "channels": channels,
                        "spans": spans,
                        "pump": pump_kind,
                        "power_dbm": power_dbm,
                        "point": label,
                        "spp_steps": spp,
                        "raman_steps": raman,
                        "xci_quad": quad,
                        "sci_switch": sci,
                        "cold_s": cold,
                        "cached_s": cached,
                        "reference_cold_s": ref_time["cold_s"],
                        "speedup": ref_time["cold_s"] / cold if cold > 0 else None,
                    }
                    record.update(_metrics(reference, result))
                    records.append(record)
                print(f"[cand] {name:18s} {pump_kind:4s} {power_dbm:4.1f} dBm done", flush=True)
    out["candidate_records"] = records
    return records


def _scan_spans(out: dict) -> list[dict]:
    """跨数扫描：同一 cl_152ch 算例的 1/5/10 跨。"""
    records: list[dict] = []
    case = build_case(SPAN_CASE)
    for spans in SPAN_COUNTS:
        path = _path(case, spans)
        for pump_kind in PUMPS:
            pumps = _pumps(pump_kind)
            for power_dbm in POWERS:
                power = launch_profile(case, power_dbm)
                reference, ref_time = _reference_point(case, pumps, power, spans)
                for label, spp, raman, quad, sci in (
                    ("default", "auto", "auto", 24, 2.0),
                    ("spp24_raman24", 24, 24, 24, 2.0),
                    ("spp12_raman24", 12, 24, 24, 2.0),
                ):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result, cold, cached = _timed(case, pumps, spp, raman, quad, sci, path, power)
                    record = {
                        "case": SPAN_CASE,
                        "spans": spans,
                        "pump": pump_kind,
                        "power_dbm": power_dbm,
                        "point": label,
                        "cold_s": cold,
                        "cached_s": cached,
                        "reference_cold_s": ref_time["cold_s"],
                        "speedup": ref_time["cold_s"] / cold if cold > 0 else None,
                    }
                    record.update(_metrics(reference, result))
                    records.append(record)
                print(f"[span] {SPAN_CASE} {spans:2d} spans {pump_kind:4s} {power_dbm:4.1f} dBm done", flush=True)
    out["span_records"] = records
    return records


def _check_reference_convergence(out: dict) -> None:
    """校验 96 步参考本身是否收敛。

    两个自由度分开加倍：只把 ``raman_steps`` 加到 192、以及只把
    ``spp_steps`` 加到 192。合在一起加倍会把两个误差源混成一个数，无法
    判断参考自身的误差地板来自哪一个。
    """
    rows = []
    probes = (
        ("cl_152ch", 10, "bwd", 3.0),
        ("escl_479ch_1span", 1, "bwd", 3.0),
        ("cl_152ch", 10, "none", 6.0),
        ("escl_479ch_1span", 1, "none", 6.0),
    )
    for name, spans, pump_kind, power_dbm in probes:
        case = build_case(name)
        path = _path(case, spans)
        pumps = _pumps(pump_kind)
        power = launch_profile(case, power_dbm)
        base, _, _ = _timed(case, pumps, 96, 96, 64, 2.0, path, power)
        more_raman, _, _ = _timed(case, pumps, 96, 192, 64, 2.0, path, power)
        more_spp, _, _ = _timed(case, pumps, 192, 96, 64, 2.0, path, power)
        for knob, finer in (("raman_steps", more_raman), ("spp_steps", more_spp)):
            row = {
                "case": name,
                "spans": spans,
                "pump": pump_kind,
                "power_dbm": power_dbm,
                "doubled": knob,
            }
            row.update(_metrics(base, finer))
            rows.append(row)
            print(
                f"[ref] {name:18s} {pump_kind:4s} {power_dbm:4.1f} dBm {knob:11s} snr_max={row['snr_max_db']}",
                flush=True,
            )
    out["reference_convergence"] = rows


def _numerical_anchor(out: dict) -> None:
    """用 numerical_dra 校验高分辨率 H3 的绝对口径（少量单跨点）。"""
    rows = []
    probes = (
        ("scaling_40ch", "none", 3.0),
        ("scaling_40ch", "bwd", 3.0),
    )
    for name, pump_kind, power_dbm in probes:
        case = build_case(name)
        path = _path(case, 1)
        pumps = _pumps(pump_kind)
        power = launch_profile(case, power_dbm)
        h3_options: dict[str, object] = {
            "poly_degree": 3,
            "egn": "auto",
            "spp_steps": 96,
            "raman_steps": 96,
        }
        if pumps:
            h3_options["raman_pumps"] = pumps
        h3 = make_model("pcfm_h3_raman", case, **h3_options).evaluate_path(path, power)
        started = time.perf_counter()
        reference = make_model("numerical_dra", case).evaluate_path(path, power)
        elapsed = time.perf_counter() - started
        row = {
            "case": name,
            "pump": pump_kind,
            "power_dbm": power_dbm,
            "numerical_dra_s": elapsed,
        }
        row.update(_metrics(reference, h3))
        rows.append(row)
        print(f"[num] {name} {pump_kind}: snr_max={row['snr_max_db']}", flush=True)
    out["numerical_anchor"] = rows


def main(argv: list[str] | None = None) -> int:
    """跑选定的阶段并把结果合并进输出 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/h3_pareto/sweep.json")
    parser.add_argument(
        "--stages",
        default="knobs,candidates,spans,reference,anchor",
        help="逗号分隔的阶段：knobs,candidates,spans,reference,anchor",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略已有的输出文件；默认合并，便于分阶段跑完再汇总。",
    )
    args = parser.parse_args(argv)
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]

    out: dict = {}
    if os.path.exists(args.out) and not args.fresh:
        with open(args.out, encoding="utf-8") as handle:
            out = json.load(handle)
    out["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "numba": __import__("numba").__version__,
        "processor": platform.processor(),
        "threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "reps": REPS,
    }
    out["matrix"] = {
        "cases": [{"name": n, "bands": b, "channels": c, "spans": s} for n, b, c, s in CASES],
        "pumps": list(PUMPS),
        "powers_dbm": list(POWERS),
        "knobs": {key: list(values) for key, values in KNOBS.items()},
        "reference": REFERENCE,
        "candidates": [list(item) for item in CANDIDATES],
    }

    if "knobs" in stages:
        _scan_knobs(out)
    if "candidates" in stages:
        _scan_candidates(out)
    if "spans" in stages:
        _scan_spans(out)
    if "reference" in stages:
        _check_reference_convergence(out)
    if "anchor" in stages:
        _numerical_anchor(out)

    destination = os.path.join(os.path.dirname(args.out) or ".", os.path.basename(args.out))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    print(f"written {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
