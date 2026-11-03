"""三篇 DRA 论文 + ECZ/PCFM 的总表：``results/raman/full_paper_comparison.json``。

三篇 DRA：

* Buglia 2024 JLT（RA-GN 闭式，``buglia2024_ra_gn``）；
* Buglia 2024 hybrid（本仓库的 ``hybrid_raman_paper``，Eq. (2)/(4) ODE +
  Eq. (6)-(9) 闭式）；
* Zhang 2024（相干 EGN，``zhang2024_dra_egn``）。

``ecz`` 是 ISRS 对照（无泵），``pcfm`` 是 EDFA-only 基线，``raman_lumped``
是集总等效近似。所有模型走同一份 ``build_case`` / ``build_path`` /
``launch_profile``，只比较单跨物理；跨数由算例决定。

运行：

    PYTHONPATH=src python scripts/full_paper_comparison.py
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
#: Zhang 2024 / 论文 hybrid 的多泵 E 波段配置（与 ``dra_paper_comparison`` 一致）。
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
    replace(build_case("scaling_40ch"), name="scaling_40ch_10span", span_count=10),
    replace(build_case("c_57ch"), name="cl_40ch_10span", bands="C+L", num_channels=40, center_frequency_hz=190.0e12),
)


def _pumps_for(case: BenchCase) -> list[RamanPumpConfig]:
    """Use the multi-pump E-band set only for the C+L case."""
    return PAPER_PUMPS if case.name.startswith("cl_") else [SINGLE_PUMP]


def _run(case: BenchCase, model_name: str, **options: Any) -> dict[str, Any]:
    """构造、计时并返回逐信道统计。"""
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

    spm = np.asarray(result.spm_noise_w)
    xpm = np.asarray(result.xpm_noise_w)
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
        "mean_spm_w": float(np.mean(spm)),
        "mean_xpm_w": float(np.mean(xpm)),
        "mean_nli_w": float(np.mean(spm + xpm)),
        "mean_signal_power_w": float(np.mean(result.signal_power_w)),
        "cold_ms": float(cold_ms),
        "warm_median_ms": float(np.median(samples)),
    }


def main() -> None:
    """运行全部对照并写 JSON。"""
    rows: list[dict[str, Any]] = []
    for case in CASES:
        pumps = _pumps_for(case)
        rows.append(_run(case, "pcfm_original"))
        rows.append(_run(case, "pcfm"))
        rows.append(_run(case, "ecz"))
        rows.append(_run(case, "pcfm", raman_model="lumped", raman_pumps=pumps))
        rows.append(_run(case, "hybrid_raman_paper", raman_pumps=pumps, nli_form="paper"))
        rows.append(_run(case, "hybrid_raman_paper", raman_pumps=pumps, nli_form="pcfm"))
        rows.append(_run(case, "buglia2024_ra_gn", raman_pumps=pumps, nli_form="closed"))
        rows.append(_run(case, "zhang2024_dra_egn", raman_pumps=pumps))
        # 无泵退化：hybrid 的 paper 口径在单跨上应回到无 Raman 的闭式基线。
        if case.span_count == 1:
            rows.append(_run(case, "hybrid_raman_paper", nli_form="paper"))

    payload = {
        "generated_by": "scripts/full_paper_comparison.py",
        "papers": {
            "buglia2024_ra_gn": {
                "citation": "H. Buglia et al., JLT 42(2):636-648, 2024, doi:10.1109/jlt.2023.3315127",
                "role": "distributed Raman amplification, closed-form RA-GN",
            },
            "hybrid_raman_paper": {
                "citation": (
                    "H. Buglia, R. I. Killey, P. Bayvel, 'Ultra-Wideband Modeling of Optical Fibre "
                    "Nonlinearity in Hybrid-Amplified Links', JLT 2024, doi:10.1109/JLT.2024.3439999"
                ),
                "role": "hybrid Raman + lumped amplification, UWB NLI/ASE",
            },
            "zhang2024_dra_egn": {
                "citation": "Zhang et al., OECC 2024, doi:10.1109/OECC54135.2024.10975528",
                "role": "distributed Raman amplification, coherent EGN closed form",
            },
        },
        "models": {
            "pcfm_original": "unmodified PCFM paper configuration (EDFA only, no Raman)",
            "pcfm": "H3 PCFM (edge fix + span cache), EDFA only",
            "ecz": "ISRS reference only (no pumps); NOT part of the DRA group",
            "raman_lumped": "PCFM + lumped-equivalent Raman (span-end gain, NLI unchanged)",
            "hybrid_raman_paper": (
                "Eq. (2)/(4) coupled ODE + Eq. (10) five-coefficient fit + Eq. (8)/(9) "
                "SPM/XPM closed forms + Eq. (6) n^epsilon (nli_form='paper')"
            ),
            "buglia2024_ra_gn": "Eq. (2)/(9)-(12) closed form of the RA-GN paper; Eq. (6)-(8) integral baseline",
            "zhang2024_dra_egn": "Eq. (1)/(2)/(4) coherent GN; Sec. II-C calibration fitted in repo",
        },
        "pumps": {
            "single_backward_1450nm": SINGLE_PUMP.model_dump(),
            "six_backward_e_band": [pump.model_dump() for pump in PAPER_PUMPS],
            "note": "single pump for C-band cases, six-pump E-band set for the C+L case",
        },
        "formula_map_hybrid": {
            "Eq. (2)": "RamanPaperSpanSolver coupled signal/pump power equations",
            "Eq. (3)": "RamanPaperSpanSolver with pumps=[] (LA limit, ASE neglected)",
            "Eq. (4)": "RamanPaperSpanSolver signal ASE equations",
            "Eq. (6)": "nli_closed_form span accumulation with n**epsilon",
            "Eq. (7)": "sum over interferers inside nli_closed_form",
            "Eq. (8)": "buglia2024_closed.eta_xpm_closed",
            "Eq. (9)": "buglia2024_closed.eta_spm_closed",
            "Eq. (10)": "buglia2024_closed.fit_buglia_profile_5param",
            "Eq. (6)-(8) baseline": "buglia2024_closed.nli_integral (nli_form='integral')",
        },
        "implemented": {
            "hybrid_raman_paper": [
                "Eq. (2)/(4) coupled signal/pump/ASE ODE, forward and backward pumps",
                "Eq. (10) five-coefficient profile fitted per channel to the ODE solution",
                "Eq. (8) XPM with Upsilon, alpha_l, kappa_f/kappa_b, l1+l2 sum",
                "Eq. (9) SPM with asinh/log terms of Appendix D",
                "Eq. (6)/(7) n^epsilon span accumulation and interferer sum",
                "Eq. (6)-(8) numerical integral baseline available as nli_form='integral'",
            ],
            "buglia2024_ra_gn": [
                "Eq. (1) coupled signal/pump power equations (same RamanPaperSpanSolver)",
                "Eq. (2) five-coefficient profile fitted to the Eq. (1) solution",
                "Eq. (9)-(12) printed SPM/XPM closed forms",
                "Eq. (4) per-span accumulation with epsilon = 0 (incoherent)",
            ],
            "zhang2024_dra_egn": [
                "Eq. (1) three-exponential profile fitted to the ODE solution",
                "Eq. (2) closed-form longitudinal link integral",
                "Eq. (3)-(5) coherent complex sum of the NLI amplitudes over spans",
            ],
        },
        "approximate_or_missing": {
            "hybrid_raman_paper": [
                "Raman gain spectrum is the repository Gaussian/CZ approximation, not the measured "
                "silica spectrum of the paper's Fig. 4",
                "pump ASE is not solved separately; the signal ASE coupling of Eq. (4) is complete",
                "pump-signal XPM omitted, as stated after Eq. (7)",
                "epsilon is not printed in the paper; default 0 (incoherent span addition)",
                "Eq. (8)/(9) printed signs follow the derivation of Appendix C/D (see buglia2024_closed)",
            ],
            "buglia2024_ra_gn": [
                "Raman gain spectrum is the repository Gaussian/CZ approximation",
                "paper has no ASE; this control still adds repository EDFA/Raman ASE",
                "pump-induced XPM is omitted",
            ],
            "zhang2024_dra_egn": [
                "eta_XPM_corr and the dataset calibration are not reproduced (not printed)",
                "inter-span phase phi_q is unpublished; the repository NLI-weighted ramp is a proxy",
                "only backward pumps are accepted",
            ],
        },
        "ecz_scope": "ecz is shown only as an ISRS reference without pumps; it is excluded from the DRA comparison.",
        "rows": rows,
    }
    output = Path("results/raman/full_paper_comparison.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
