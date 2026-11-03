"""Hybrid Raman 对照基准：PCFM、lumped、论文 Eq. (6)-(9) 与旧 PCFM 口径。

输出 ``results/raman/paper_comparison.json``。所有解析模型走同一份
``build_path`` / ``launch_profile``，只比较单跨物理；物理 RamanPump 只用于
增益形状对照（它本身也是未耗尽集总模型，不是论文 coupled ODE）。

``hybrid_raman_paper`` 跑两次：``nli_form="paper"``（Eq. (6)-(9) 闭式，默认）
与 ``nli_form="pcfm"``（升级前的 PCFM 核近似，回归对照）。

运行：

    PYTHONPATH=src python scripts/raman_paper_comparison.py
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from network_nlo_eval.benchmarks.cases import BenchCase, build_case, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.hybrid_raman import RamanPaperSpanSolver
from network_nlo_eval.models.numba_kernels import _calc_aeff_dynamic, _calc_fiber_attenuation_npm
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.network.elements import RamanPumpConfig

PUMP = RamanPumpConfig(
    wavelength_nm=1450.0,
    power_w=0.2,
    direction="backward",
    attenuation_db_per_km=0.25,
)

CASES: tuple[BenchCase, ...] = (
    build_case("c_57ch"),
    replace(build_case("c_57ch"), name="c_57ch_1span", span_count=1),
    build_case("scaling_40ch"),
    replace(build_case("scaling_40ch"), name="scaling_40ch_10span", span_count=10),
)


def _run(case: BenchCase, model_name: str, **options: Any) -> dict[str, Any]:
    """构造模型、计时并返回逐信道统计。"""
    model = make_model(model_name, case, **options)
    path = build_path(case)
    power = launch_profile(case)

    started = time.perf_counter()
    result = model.evaluate_path(path, power)
    cold_ms = (time.perf_counter() - started) * 1e3

    samples = []
    for _ in range(3):
        started = time.perf_counter()
        result = model.evaluate_path(path, power)
        samples.append((time.perf_counter() - started) * 1e3)

    nli = np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)
    serializable_options = {
        key: (
            [pump.model_dump() for pump in value]
            if key == "raman_pumps" and value is not None
            else value
        )
        for key, value in options.items()
    }
    return {
        "model": model_name,
        "options": serializable_options,
        "case": case.name,
        "span_count": case.span_count,
        "mean_snr_db": float(np.mean(result.snr_db)),
        "min_snr_db": float(np.min(result.snr_db)),
        "max_snr_db": float(np.max(result.snr_db)),
        "mean_ase_w": float(np.mean(result.ase_noise_w)),
        "mean_nli_w": float(np.mean(nli)),
        "mean_signal_power_w": float(np.mean(result.signal_power_w)),
        "cold_ms": float(cold_ms),
        "warm_median_ms": float(np.median(samples)),
        "snr_db": [float(value) for value in result.snr_db],
    }


def _physical_gain_comparison(case: BenchCase) -> dict[str, Any]:
    """用物理 ``RamanPump`` 的 Gaussian 增益形状对照解析模型。"""
    grid = grid_for(case)
    frequencies = np.asarray(grid.frequencies, dtype=np.float64)
    lumped = RamanGainModel()
    lumped_gain = lumped.on_off_gain(frequencies, [PUMP], case.span_length_km * 1e3, a_eff_m2=80e-12)

    try:
        from physics_nlo_eval.channel.network import PumpSpec, RamanPump
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}

    physical = RamanPump(
        PumpSpec(**PUMP.model_dump()),
        length_km=case.span_length_km,
        effective_area_um2=80.0,
        add_noise=False,
    )
    physical_gain = physical._gain(frequencies)

    # 论文 ODE 在案例入纤功率下的 on-off 增益（含泵浦耗尽）。
    fiber = build_path(case).fiber_configs[0]
    lambdas_m = C_LIGHT / frequencies
    a_eff_m2 = _calc_aeff_dynamic(
        lambdas_m,
        fiber.effective_area_um2_ref,
        fiber.aeff_slope_um2_nm,
        fiber.reference_wavelength_nm,
    )
    if fiber.use_attenuation_polynomial:
        alpha_npm = _calc_fiber_attenuation_npm(
            lambdas_m,
            fiber.reference_wavelength_nm,
            fiber.attenuation_alpha2,
            fiber.attenuation_alpha1,
            fiber.attenuation_alpha0,
        )
    else:
        alpha_npm = np.full(
            frequencies.size,
            fiber.attenuation_db_km_ref * np.log(10.0) / 10.0 / 1e3,
            dtype=np.float64,
        )
    pump_a_eff = _calc_aeff_dynamic(
        np.array([PUMP.wavelength_nm * 1e-9]),
        fiber.effective_area_um2_ref,
        fiber.aeff_slope_um2_nm,
        fiber.reference_wavelength_nm,
    )
    order = np.argsort(-frequencies, kind="stable")
    solver = RamanPaperSpanSolver(
        frequencies[order],
        np.full(frequencies.size, case.symbol_rate_hz),
        [PUMP],
        case.span_length_km * 1e3,
        alpha_npm[order],
        a_eff_m2[order],
        np.array([PUMP.attenuation_db_per_km * np.log(10.0) / 10.0 / 1e3]),
        pump_a_eff,
        RamanGainModel(),
        n_steps=32,
        n_iterations=24,
    )
    launch = launch_profile(case)
    ode = solver.solve(launch[order])
    passive = launch[order] * np.exp(-alpha_npm[order] * case.span_length_km * 1e3)
    ode_gain = np.empty_like(ode.power_out_w)
    ode_gain[order] = ode.power_out_w / passive

    return {
        "available": True,
        "pump": PUMP.model_dump(),
        "lumped_gain_db": [float(value) for value in 10.0 * np.log10(lumped_gain)],
        "physical_gain_db": [float(value) for value in 10.0 * np.log10(physical_gain)],
        "ode_gain_db": [float(value) for value in 10.0 * np.log10(ode_gain)],
        "ode_pump_power_out_w": float(ode.pump_power_out_w[0]),
        "note": (
            "physical RamanPump is itself an undepleted lumped-equivalent block; "
            "it validates the gain shape, not the coupled-ODE depletion."
        ),
    }


def main() -> None:
    """运行全部对照并写 JSON。"""
    rows: list[dict[str, Any]] = []
    for case in CASES:
        rows.append(_run(case, "pcfm_original"))
        rows.append(_run(case, "pcfm"))
        rows.append(_run(case, "pcfm", raman_model="lumped", raman_pumps=[PUMP]))
        rows.append(_run(case, "hybrid_raman_paper", raman_pumps=[PUMP], nli_form="paper"))
        rows.append(_run(case, "hybrid_raman_paper", raman_pumps=[PUMP], nli_form="pcfm"))

    payload = {
        "generated_by": "scripts/raman_paper_comparison.py",
        "pump": PUMP.model_dump(),
        "cases": [case.name for case in CASES],
        "models": {
            "pcfm_original": "unmodified PCFM paper configuration",
            "pcfm": "H3 PCFM (edge fix + span cache)",
            "raman_lumped": "PCFM + lumped-equivalent Raman (span-end gain, NLI unchanged)",
            "hybrid_raman_paper": (
                "Buglia 2024 hybrid: Eq. (2)/(4) coupled ODE, Eq. (10) five-coefficient "
                "profile, Eq. (8)/(9) SPM/XPM closed forms, Eq. (6) n^epsilon "
            ),
        },
        "rows": rows,
        "physical_raman_pump": _physical_gain_comparison(build_case("c_57ch")),
        "limitations": [
            "hybrid_raman_paper implements Eq. (6)-(9); the Raman gain spectrum is the "
            "repository Gaussian/CZ approximation, not the measured silica spectrum.",
            "nli_form='pcfm' keeps the previous PCFM-kernel approximation only for regression.",
            "Raman gain spectrum is the repository Gaussian/CZ approximation, not measured silica.",
            "Pump ASE is not solved separately; signal ASE coupling is included.",
            "The physical RamanPump is an undepleted lumped block, used only for gain-shape comparison.",
        ],
    }
    output = Path("results/raman/paper_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
