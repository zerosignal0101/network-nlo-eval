"""FG-1 physics parity regression test.

Guards the reconciliation result: ``wp/fg-reconcile`` must reproduce
``origin/main``'s ODE physics *exactly* while carrying the path-level
pipeline from ``ecz-dev``.

The tolerance is zero. That is deliberate: ``models/numba_kernels.py`` is
byte-identical to ``origin/main`` and the ported pipeline only orchestrates
the same low-level ``_calc_span_noise_and_power`` calls, so any deviation is
a regression rather than floating-point noise.

Two properties are asserted per scenario:

1. ``run_reference`` on this tree equals the recorded golden values from
   ``origin/main`` -- the kernel was not altered.
2. ``MultiBandISRSGN.evaluate_path`` equals the same reference -- the new
   pipeline is physics-neutral and does not double-count or reorder spans.

Regenerate the golden file from a pristine main checkout with::

    PYTHONPATH=<main-tree>/src python scripts/fg1_parity.py \\
        --emit-golden results/fg1_golden_main.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fg1_scenarios import SCENARIOS, run_reference
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "results" / "fg1_golden_main.json"
PARITY_PATH = Path(__file__).resolve().parent.parent / "results" / "fg1_physics_parity.json"
SCENARIO_IDS = [scenario.name for scenario in SCENARIOS]


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"missing golden file {GOLDEN_PATH}; regenerate it from a pristine "
            "origin/main checkout with scripts/fg1_parity.py --emit-golden"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _pipeline_result(scenario):
    """Run the scenario through the ported path-level pipeline."""
    grid = scenario.grid()
    evaluator = MultiBandISRSGN(
        grid,
        scenario.fiber(scenario.hops[0]),
        symbol_rate_hz=scenario.symbol_rate_hz,
        phi_kurtosis=scenario.phi_kurtosis,
        # FG-4b §1.2: this test's proposition is "the ported pipeline is
        # physics-neutral" — a statement about port fidelity, pinned to the frozen
        # ``off`` tier. Production default is ``matvec`` and is gated separately at
        # <=1e-12; the bit-equality assertion and its expectation are unchanged.
        ode_accel="off",
    )
    fibers = [scenario.fiber(hop).model_copy(update={"max_span_length_km": hop.span_km}) for hop in scenario.hops]
    edfas = [scenario.edfa(hop) for hop in scenario.hops]
    roadms = []
    for hop in scenario.hops:
        roadm = scenario.roadm(hop)
        if roadm is not None:
            roadm = roadm.model_copy(update={"blocked_channels": list(hop.blocked_channels)})
        roadms.append(roadm)
    path = MultiSpanOpticalPath(fiber_configs=fibers, edfa_configs=edfas, roadm_configs=roadms)
    return evaluator.evaluate_path(path, scenario.launch_power())


def test_golden_file_declares_origin_main(golden: dict) -> None:
    """The golden data is traceable to origin/main."""
    assert golden["source_branch"] == "origin/main"
    assert set(golden["scenarios"]) == set(SCENARIO_IDS)


def test_kernel_uses_ode_and_not_the_analytic_tilt_variant() -> None:
    """The ODE kernel is present and the ecz-dev analytic replacement is not.

    The reconciliation direction is one-way: the pipeline must call main's
    RK4 power evolution, never the analytic ``r_f``/``L_eff`` model.
    """
    from network_nlo_eval.models import numba_kernels as kernels

    assert hasattr(kernels, "_calc_exact_power_profile_ode")
    assert hasattr(kernels, "_calc_dp_dz")
    assert not hasattr(kernels, "_calc_span_out_power_jit")
    assert not hasattr(kernels, "_calc_raman_profile_jit")


@pytest.mark.parametrize("scenario_index", range(len(SCENARIOS)))
def test_lowlevel_matches_origin_main(golden: dict, scenario_index: int) -> None:
    """Low-level span calls reproduce origin/main bit for bit."""
    scenario = SCENARIOS[scenario_index]
    expected = golden["scenarios"][scenario.name]
    actual = run_reference(scenario)

    for key in ("signal_power_w", "spm_w", "xpm_w", "ase_w", "snr_db"):
        assert np.array_equal(
            np.asarray(expected[key], dtype=float),
            np.asarray(actual[key], dtype=float),
        ), f"{scenario.name}.{key} deviates from origin/main"

    assert len(actual["span_curve"]) == len(expected["span_curve"])
    for exp_span, act_span in zip(expected["span_curve"], actual["span_curve"], strict=True):
        for key in ("power_out_w", "spm_w", "xpm_w", "ase_w"):
            assert np.array_equal(
                np.asarray(exp_span[key], dtype=float),
                np.asarray(act_span[key], dtype=float),
            ), f"{scenario.name} span {exp_span['hop']}/{exp_span['span']} {key} deviates"


@pytest.mark.parametrize("scenario_index", range(len(SCENARIOS)))
def test_pipeline_matches_origin_main(golden: dict, scenario_index: int) -> None:
    """evaluate_path reproduces origin/main bit for bit.

    This is the load-bearing assertion: it shows the newly ported pipeline is
    physics-neutral rather than a re-implementation.
    """
    scenario = SCENARIOS[scenario_index]
    expected = golden["scenarios"][scenario.name]
    result = _pipeline_result(scenario)

    for key, actual in (
        ("signal_power_w", result.signal_power_w),
        ("spm_w", result.spm_noise_w),
        ("xpm_w", result.xpm_noise_w),
        ("ase_w", result.ase_noise_w),
        ("snr_db", result.snr_db),
    ):
        assert np.array_equal(
            np.asarray(expected[key], dtype=float),
            np.asarray(actual, dtype=float),
        ), f"{scenario.name}.evaluate_path().{key} deviates from origin/main"

    assert result.span_count == len(expected["span_curve"])


def test_scenarios_are_non_degenerate() -> None:
    """Guard against a vacuous parity check over all-zero physics."""
    for scenario in SCENARIOS:
        result = run_reference(scenario)
        span0 = np.asarray(result["span_curve"][0]["power_out_w"], dtype=float)
        assert np.all(span0 > 0), f"{scenario.name}: non-positive span power"
        assert np.all(np.asarray(result["spm_w"], dtype=float) > 0), f"{scenario.name}: no SPM"
        assert np.all(np.asarray(result["ase_w"], dtype=float) > 0), f"{scenario.name}: no ASE"
        assert np.all(np.isfinite(np.asarray(result["snr_db"], dtype=float))), f"{scenario.name}: non-finite SNR"

    # The wide multi-band scenario must show real inter-channel Raman tilt,
    # otherwise the ODE's channel coupling is not being exercised.
    wide = next(s for s in SCENARIOS if s.name == "multiband_wide")
    span0 = np.asarray(run_reference(wide)["span_curve"][0]["power_out_w"], dtype=float)
    assert span0.max() / span0.min() > 1.05, "multiband_wide shows no SRS tilt across the band"


def test_kurtosis_scenario_is_sensitive_to_phi_kurtosis(golden: dict) -> None:
    """The 64QAM scenario actually depends on phi_kurtosis.

    Ensures the parity set would notice if the kurtosis-aware XPM code were
    swapped for a Gaussian-only variant.
    """
    from fg1_scenarios import Scenario

    base = next(s for s in SCENARIOS if s.name == "kurtosis_64qam")
    gaussian = Scenario(**{**base.to_dict(), "phi_kurtosis": 0.0, "hops": base.hops})
    with_kurtosis = np.asarray(run_reference(base)["snr_db"], dtype=float)
    without = np.asarray(run_reference(gaussian)["snr_db"], dtype=float)
    assert not np.array_equal(with_kurtosis, without), (
        "kurtosis_64qam is insensitive to phi_kurtosis; the scenario does not "
        "cover main's kurtosis correction"
    )


def test_parity_artifact_is_present_and_ok() -> None:
    """The committed parity report exists and records a clean result."""
    if not PARITY_PATH.exists():
        pytest.fail(f"missing {PARITY_PATH}; run scripts/fg1_parity.py --check")
    report = json.loads(PARITY_PATH.read_text(encoding="utf-8"))
    assert report["parity_ok"] is True
    assert report["failures"] == []
    assert report["tolerance"] == 0.0
    assert set(report["scenarios"]) == set(SCENARIO_IDS)
