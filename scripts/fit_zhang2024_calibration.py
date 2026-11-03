"""Fit the Zhang 2024 Sec. II-C per-term calibration.

The paper writes the calibration as

    P.P.^EGN / P.P.^CFM = f(Params)
        = prod_i 1 / (a_i * exp(b_i * Param_i^c_i + d_i))

with ``a_i, b_i, c_i, d_i`` fitted by least squares over a simulation data set.
The paper does **not** publish the coefficients, the parameter list or the data
set, so this script rebuilds the data set inside the paper's Table I ranges and
fits the coefficients in this repository.  The output is
``src/network_nlo_eval/models/data/zhang2024_calibration.json`` and is labelled
"fitted here" everywhere it is reported.

The fit uses ``c_i = 1`` (the paper's form reduces to a linear model in
``ln(ratio)`` for that choice), two parameters — per-channel launch power and
span count — and absorbs the constant term into ``d`` of the span factor.

Run:

    PYTHONPATH=src python scripts/fit_zhang2024_calibration.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.zhang2024_coherent import cfm_and_egn_terms
from network_nlo_eval.network.elements import RamanPumpConfig

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "src" / "network_nlo_eval" / "models" / "data" / "zhang2024_calibration.json"

#: 论文 Sec. III 的六个后向泵（1340/1360/1427/1445/1476/1505 nm）。
PAPER_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]

#: 数据集的 (跨数, 每信道功率 dBm) 组合，落在 Table I 的范围内。
GRID = [(spans, power) for spans in (8, 12, 20) for power in (-1.0, 1.0, 3.0)]


def _ratio(case, power_dbm: float) -> dict[str, float]:
    """一条配置的 ``EGN/CFM`` 中位比值（SPM 与 XPM 分开）。"""
    model = make_model("zhang2024_dra_egn", case, raman_pumps=PAPER_PUMPS, use_calibration=False)
    launch = launch_profile(case, power_dbm=power_dbm)
    model.evaluate_path(build_path(case), launch)
    records = model._span_records  # noqa: SLF001
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    from network_nlo_eval.models.zhang2024_coherent import ZhangCoherentSpan

    spans = []
    start = 0.0
    for profile, span_launch, length, _transfer in records:
        spans.append(ZhangCoherentSpan(profile, np.asarray(span_launch, dtype=np.float64)[kernel.order], start))
        start += length
    cfm_spm, cfm_xpm, egn_spm, egn_xpm = cfm_and_egn_terms(
        spans,
        spans[0].launch_power_w,
        kernel.b_hz,
        kernel.gamma_per_w_m,
        kernel.beta2,
        kernel.beta3,
        kernel.reference_frequency_hz,
        n_frequency=48,
        n_spm_frequency=32,
    )
    result = {}
    for name, cfm, egn in (("spm", cfm_spm, egn_spm), ("xpm", cfm_xpm, egn_xpm)):
        mask = (cfm > 0.0) & (egn > 0.0)
        result[name] = float(np.median(egn[mask] / cfm[mask]))
        result[f"{name}_spread_db"] = float(
            10.0 * np.log10(np.max(egn[mask] / cfm[mask]) / np.min(egn[mask] / cfm[mask]))
        )
    return result


def _fit(samples: list[tuple[float, float, float]]) -> list[dict[str, object]]:
    """线性最小二乘 ``ln(ratio) = -k0 - k1*P - k2*N``，返回论文形式的系数。"""
    design = np.array([[1.0, p, float(n)] for p, n, _ in samples], dtype=np.float64)
    target = np.log(np.array([ratio for _, _, ratio in samples], dtype=np.float64))
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    k0, k1, k2 = (float(value) for value in coefficients)
    residual = design @ coefficients - target
    # f(Params) = prod 1/(a exp(b Param^c + d)) = exp(-(k0 + k1 P + k2 N))，
    # 而我们要的是 EGN/CFM = ratio = exp(k0 + k1 P + k2 N)，故 b = -k、d = -k0。
    return [
        {"name": "launch_power_dbm", "parameters": [1.0, -k1, 1.0, 0.0]},
        {"name": "spans", "parameters": [1.0, -k2, 1.0, -k0]},
    ], float(np.sqrt(np.mean(residual**2)))


def main() -> None:
    """生成数据集、拟合系数并写 JSON。"""
    case0 = build_case("c_57ch")
    rows: dict[str, list[tuple[float, float, float]]] = {"spm": [], "xpm": []}
    for spans, power_dbm in GRID:
        case = replace(
            case0,
            name=f"cal_{spans}span_{power_dbm:+.0f}dBm",
            bands="C+L",
            num_channels=24,
            center_frequency_hz=190.0e12,
            span_count=spans,
        )
        ratios = _ratio(case, power_dbm)
        for name in ("spm", "xpm"):
            rows[name].append((power_dbm, float(spans), ratios[name]))
        print(
            f"{case.name}: spm {ratios['spm']:.4f} ({ratios['spm_spread_db']:.2f} dB) "
            f"xpm {ratios['xpm']:.4f} ({ratios['xpm_spread_db']:.2f} dB)"
        )

    terms = {}
    quality = {}
    for name, samples in rows.items():
        entries, rms = _fit(samples)
        terms[name] = entries
        quality[name] = {"log_ratio_rms": rms}
    payload = {
        "generated_by": "scripts/fit_zhang2024_calibration.py",
        "provenance": (
            "Coefficients fitted IN THIS REPOSITORY. Zhang 2024 Sec. II-B states that the "
            "closed-form terms are not printed and Sec. II-C gives only the functional form, "
            "so no paper coefficient exists to copy."
        ),
        "ranges": {"spans": [8, 20], "launch_power_dbm": [-1.0, 3.0], "channels": 24, "bands": "C+L"},
        "fit": quality,
        "terms": terms,
    }
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {TARGET}")
    for name, entries in terms.items():
        print(name, entries, f"log-rms {quality[name]['log_ratio_rms']:.5f}")


if __name__ == "__main__":
    main()
