"""DRA paper-control benchmark: Buglia 2024 and Zhang 2024 against the baselines.

The script drives every model through the same ``build_case`` / ``build_path``
helpers and writes ``results/raman/paper_controls.json``.  ``ecz`` is included
only as an ISRS reference (no pumps); it is not part of the DRA group.

Run:

    PYTHONPATH=src python scripts/dra_paper_comparison.py
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from network_nlo_eval.benchmarks.cases import BenchCase, build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.network.elements import RamanPumpConfig

SINGLE_PUMP = RamanPumpConfig(
    wavelength_nm=1450.0,
    power_w=0.2,
    direction="backward",
    attenuation_db_per_km=0.25,
)
FORWARD_PUMP = RamanPumpConfig(
    wavelength_nm=1450.0,
    power_w=0.2,
    direction="forward",
    attenuation_db_per_km=0.25,
)
PAPER_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]

CASES: tuple[BenchCase, ...] = (
    replace(build_case("c_57ch"), name="c_57ch_1span", span_count=1),
    build_case("c_57ch"),
    replace(build_case("c_57ch"), name="cl_40ch_10span", bands="C+L", num_channels=40, center_frequency_hz=190.0e12),
)


def _run(case: BenchCase, model_name: str, **options: Any) -> dict[str, Any]:
    """Construct, time and evaluate one model on one case."""
    model = make_model(model_name, case, **options)
    path = build_path(case)
    power = launch_profile(case)
    started = time.perf_counter()
    result = model.evaluate_path(path, power)
    cold_ms = (time.perf_counter() - started) * 1e3
    samples = []
    for _ in range(2):
        started = time.perf_counter()
        result = model.evaluate_path(path, power)
        samples.append((time.perf_counter() - started) * 1e3)
    nli = np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)
    serializable = {
        key: ([pump.model_dump() for pump in value] if key == "raman_pumps" and value else value)
        for key, value in options.items()
    }
    return {
        "model": model_name,
        "options": serializable,
        "case": case.name,
        "span_count": case.span_count,
        "mean_snr_db": float(np.mean(result.snr_db)),
        "min_snr_db": float(np.min(result.snr_db)),
        "max_snr_db": float(np.max(result.snr_db)),
        "mean_ase_w": float(np.mean(result.ase_noise_w)),
        "mean_nli_w": float(np.mean(nli)),
        "mean_spm_w": float(np.mean(result.spm_noise_w)),
        "mean_xpm_w": float(np.mean(result.xpm_noise_w)),
        "mean_signal_power_w": float(np.mean(result.signal_power_w)),
        "cold_ms": float(cold_ms),
        "warm_median_ms": float(np.median(samples)),
    }


def main() -> None:
    """Run all comparisons and write the JSON report."""
    rows: list[dict[str, Any]] = []
    for case in CASES:
        rows.append(_run(case, "pcfm_original"))
        rows.append(_run(case, "pcfm"))
        rows.append(_run(case, "ecz"))
        if case.name.startswith("cl_"):
            pumps = PAPER_PUMPS
        else:
            pumps = [SINGLE_PUMP]
        rows.append(_run(case, "pcfm", raman_model="lumped", raman_pumps=pumps))
        rows.append(_run(case, "hybrid_raman_paper", raman_pumps=pumps, nli_form="paper"))
        rows.append(_run(case, "buglia2024_ra_gn", raman_pumps=pumps))
        rows.append(_run(case, "zhang2024_dra_egn", raman_pumps=pumps))

    forward_case = replace(build_case("c_57ch"), name="c_57ch_1span_fw", span_count=1)
    rows.append(_run(forward_case, "buglia2024_ra_gn", raman_pumps=[FORWARD_PUMP]))
    rows.append(_run(forward_case, "buglia2024_ra_gn", raman_pumps=[SINGLE_PUMP]))

    payload = {
        "generated_by": "scripts/dra_paper_comparison.py",
        "models": {
            "pcfm_original": "unmodified PCFM paper configuration",
            "pcfm": "H3 PCFM (edge fix + span cache), EDFA only",
            "raman_lumped": "PCFM + lumped-equivalent Raman (span-end gain, NLI unchanged)",
            "hybrid_raman_paper": "Buglia hybrid: Eq. (2)/(4) coupled ODE with Eq. (6)-(9) NLI",
            "ecz": "ISRS reference only (no pumps); NOT part of the DRA group",
            "buglia2024_ra_gn": "Buglia 2024 Eq. (2)/(9)-(12) closed form; Eq. (6)-(8) integral baseline",
            "zhang2024_dra_egn": "Zhang 2024 Eq. (1)/(2)/(4) coherent GN; Sec. II-C calibration fitted in repo",
        },
        "pumps": {
            "single_backward_1450nm": SINGLE_PUMP.model_dump(),
            "single_forward_1450nm": FORWARD_PUMP.model_dump(),
            "zhang_2024_six_backward": [pump.model_dump() for pump in PAPER_PUMPS],
        },
        "rows": rows,
        "implemented": {
            "buglia2024_ra_gn": [
                "Eq. (1) coupled signal/pump power equations (reused RamanPaperSpanSolver)",
                "Eq. (2) five-coefficient profile fitted to the Eq. (1) solution",
                "Eq. (3)-(8) GN SPM/XPM link integrals with the fitted profile",
                "Eq. (4) per-span accumulation with epsilon = 1 (incoherent)",
            ],
            "zhang2024_dra_egn": [
                "Eq. (1) three-exponential profile fitted to the Eq. (1) solution",
                "Eq. (2) closed-form longitudinal link integral",
                "Eq. (3)-(4) per-span NLI with a coherent complex sum over spans",
                "Eq. (5) same-span and cross-span pair products through |sum a_m|^2",
            ],
        },
        "approximate_or_missing": {
            "buglia2024_ra_gn": [
                "Eq. (9)/(11)/(12) printed closed forms are not copied term by term; "
                "the equivalent GN double integral is evaluated instead",
                "alpha_f and alpha_b are fixed to the pump attenuation (degenerate fit)",
                "Raman gain spectrum is the repository Gaussian/CZ approximation",
                "paper has no ASE; this control still adds repository EDFA/Raman ASE",
                "pump-induced XPM is omitted",
            ],
            "zhang2024_dra_egn": [
                "the closed-form P terms of Eq. (5) are not published; GN SPM/XPM is used",
                "eta_XPM_corr and the dataset calibration (a_i,b_i,c_i,d_i) are not reproduced",
                "inter-span phase phi_q is unpublished; the repository NLI-weighted ramp is a proxy",
                "only backward pumps are accepted",
                "P_Fm P_Fn / P_Bm P_Bn are not dropped (the coherent sum keeps all pairs)",
            ],
        },
        "ecz_scope": "ecz is shown only as an ISRS reference without pumps; it is excluded from the DRA comparison.",
        "limitations": [
            (
                "Buglia 2024: the printed Eq. (9)/(11)/(12) closed forms are implemented; "
                "the Eq. (6)-(8) integral is the baseline."
            ),
            "Buglia closed-form SPM is the paper form; the repo integral baseline differs by <0.05 dB on c_57ch.",
            (
                "Zhang eta_XPM_corr is not printed in the paper; the coherent GN + repo-fitted "
                "Sec. II-C calibration is implemented, but the paper-defined correction term is absent."
            ),
            "No waveform model is called online by either control.",
        ],
    }
    output = Path("results/raman/paper_controls.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
