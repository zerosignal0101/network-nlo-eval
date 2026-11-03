"""Tests for the two DRA paper-control models.

Coverage required by the task:

* registration and honest summaries;
* path shapes;
* no-pump / zero-pump degradation;
* pump power and direction;
* bounded difference against the existing models;
* at least one SNR case;
* the Zhang coherent cross-span sum and the Buglia/Zhang independence.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath, get_model_info
from network_nlo_eval.models.buglia2024_ra import BUGLIA2024_RA_MODEL_INFO
from network_nlo_eval.models.dra_common import average_link_squared, fit_buglia_profile
from network_nlo_eval.models.zhang2024_dra import ZHANG2024_DRA_MODEL_INFO
from network_nlo_eval.network.elements import RamanPumpConfig

CASE = build_case("scaling_40ch")
CASE_2SPAN = replace(CASE, name="scaling_40ch_2span", span_count=2)
CL_CASE = replace(build_case("c_57ch"), name="cl_40ch", bands="C+L", num_channels=40, center_frequency_hz=190.0e12)

PAPER_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]


def _pump(power_w: float = 0.2, direction: str = "backward") -> RamanPumpConfig:
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _mean_nli(result) -> float:
    return float(np.mean(np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_dra_paper_models_are_independently_registered() -> None:
    """Both papers must have their own registry entry and honest summary."""
    buglia = get_model_info("buglia2024_ra_gn")
    zhang = get_model_info("zhang2024_dra_egn")
    assert buglia is BUGLIA2024_RA_MODEL_INFO
    assert zhang is ZHANG2024_DRA_MODEL_INFO
    assert "buglia" in buglia.reference.lower()
    assert "zhang" in zhang.reference.lower()
    assert "subset" in buglia.summary.lower()
    assert "subset" in zhang.summary.lower()
    assert "not reproduced" in zhang.summary.lower()
    assert "buglia2024_ra_gn" != "zhang2024_dra_egn"


# ---------------------------------------------------------------------------
# Shapes and construction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["buglia2024_ra_gn", "zhang2024_dra_egn"])
def test_models_return_path_shapes(name: str) -> None:
    """Both controls return the path-level arrays with finite SNR."""
    power = launch_profile(CASE)
    path = build_path(CASE)
    result = make_model(name, CASE, raman_pumps=[_pump()]).evaluate_path(path, power)
    for attribute in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.asarray(getattr(result, attribute)).shape == (CASE.num_channels,)
    assert np.all(np.isfinite(result.snr_db))


def test_invalid_options_are_rejected() -> None:
    """Invalid iteration counts and quadrature sizes fail loudly."""
    with pytest.raises(ValueError, match="fixed_point_iterations"):
        make_model("buglia2024_ra_gn", CASE, fixed_point_iterations=0)
    with pytest.raises(ValueError, match="n_quadrature"):
        make_model("zhang2024_dra_egn", CASE, n_quadrature=4)


# ---------------------------------------------------------------------------
# No-pump / zero-pump degradation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["buglia2024_ra_gn", "zhang2024_dra_egn"])
def test_zero_power_pump_is_the_no_pump_case(name: str) -> None:
    """A zero-power pump must be bit-for-bit the no-pump case."""
    power = launch_profile(CASE)
    path = build_path(CASE)
    no_pump = make_model(name, CASE).evaluate_path(path, power)
    zero_pump = make_model(name, CASE, raman_pumps=[_pump(power_w=0.0)]).evaluate_path(path, power)
    for attribute in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        np.testing.assert_array_equal(
            np.asarray(getattr(no_pump, attribute)),
            np.asarray(getattr(zero_pump, attribute)),
        )


@pytest.mark.parametrize("name", ["buglia2024_ra_gn", "zhang2024_dra_egn"])
def test_no_pump_is_close_to_the_pcfm_baseline(name: str) -> None:
    """Without pumps the controls degrade to an ISRS-GN-like result."""
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    control = make_model(name, CASE).evaluate_path(path, power)
    assert abs(float(np.mean(control.snr_db)) - float(np.mean(base.snr_db))) < 1.2
    ratio = _mean_nli(control) / _mean_nli(base)
    assert 0.25 < ratio < 4.0


# ---------------------------------------------------------------------------
# Pump power and direction
# ---------------------------------------------------------------------------


def test_buglia_pump_power_increases_gain_and_ase() -> None:
    """More pump power must raise both signal output and Raman ASE."""
    power = launch_profile(CASE)
    fiber = build_path(CASE).fiber_configs[0]
    outputs = []
    ases = []
    for pump_w in (0.05, 0.1, 0.2):
        model = make_model("buglia2024_ra_gn", CASE, raman_pumps=[_pump(power_w=pump_w)])
        kernel = model.kernel_for(fiber)
        fiber_output = kernel.evaluate(power, CASE.span_length_km * 1e3)
        outputs.append(float(np.mean(fiber_output.power_out_w)))
        ases.append(float(np.mean(kernel.last_raman_ase_w)))
    assert np.all(np.diff(outputs) > 0.0)
    assert np.all(np.diff(ases) > 0.0)


def test_buglia_forward_and_backward_pumps_differ() -> None:
    """FW and BW pumps place the gain differently and change NLI."""
    power = launch_profile(CASE)
    path = build_path(CASE)
    fiber = path.fiber_configs[0]
    backward_model = make_model("buglia2024_ra_gn", CASE, raman_pumps=[_pump(direction="backward")])
    forward_model = make_model("buglia2024_ra_gn", CASE, raman_pumps=[_pump(direction="forward")])
    backward_fiber = backward_model.kernel_for(fiber).evaluate(power, CASE.span_length_km * 1e3)
    forward_fiber = forward_model.kernel_for(fiber).evaluate(power, CASE.span_length_km * 1e3)
    assert not np.allclose(backward_fiber.power_out_w, forward_fiber.power_out_w)
    backward = backward_model.evaluate_path(path, power)
    forward = forward_model.evaluate_path(path, power)
    assert _mean_nli(forward) > _mean_nli(backward)
    assert float(np.mean(forward.snr_db)) < float(np.mean(backward.snr_db))


def test_zhang_rejects_forward_pumps() -> None:
    """Zhang 2024 is a backward-DRA model; a forward pump is rejected."""
    with pytest.raises(ValueError, match="backward Raman pumps only"):
        make_model("zhang2024_dra_egn", CASE, raman_pumps=[_pump(direction="forward")])


# ---------------------------------------------------------------------------
# Bounded difference against existing models
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["buglia2024_ra_gn", "zhang2024_dra_egn"])
def test_dra_controls_are_bounded_against_the_existing_models(name: str) -> None:
    """The controls stay within a documented band of the existing models."""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pump = _pump()
    control = make_model(name, CASE, raman_pumps=[pump]).evaluate_path(path, power)
    hybrid = make_model("hybrid_raman_paper", CASE, raman_pumps=[pump]).evaluate_path(path, power)
    lumped = make_model("pcfm", CASE, raman_model="lumped", raman_pumps=[pump]).evaluate_path(path, power)
    pcfm = make_model("pcfm", CASE).evaluate_path(path, power)

    for reference in (hybrid, lumped, pcfm):
        assert abs(float(np.mean(control.snr_db)) - float(np.mean(reference.snr_db))) < 3.0
        ratio = _mean_nli(control) / _mean_nli(reference)
        assert 0.1 < ratio < 10.0


def test_the_two_dra_controls_are_not_identical() -> None:
    """The two papers are independent implementations, not aliases."""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pump = _pump()
    buglia = make_model("buglia2024_ra_gn", CASE, raman_pumps=[pump]).evaluate_path(path, power)
    zhang = make_model("zhang2024_dra_egn", CASE, raman_pumps=[pump]).evaluate_path(path, power)
    assert _mean_nli(buglia) != pytest.approx(_mean_nli(zhang), rel=1e-3)


# ---------------------------------------------------------------------------
# Zhang coherent cross-span sum
# ---------------------------------------------------------------------------


def test_zhang_uses_complex_amplitudes_across_spans() -> None:
    """Multi-span Zhang returns NLI amplitudes consistent with the powers."""
    power = launch_profile(CASE)
    fiber = build_path(CASE).fiber_configs[0]
    model = make_model("zhang2024_dra_egn", CASE, raman_pumps=[_pump()])
    kernel = model.kernel_for(fiber)
    from network_nlo_eval.models.base import SpanContext

    output = kernel.evaluate(
        power,
        CASE.span_length_km * 1e3,
        1,
        span_context=SpanContext(span_index=0, total_spans=2, cumulative_length_m=0.0, span_length_m=100e3),
    )
    assert output.nli_amplitude_spm is not None
    assert output.nli_amplitude_xpm is not None
    np.testing.assert_allclose(np.abs(output.nli_amplitude_spm) ** 2, output.spm_noise_w, rtol=1e-12)
    np.testing.assert_allclose(np.abs(output.nli_amplitude_xpm) ** 2, output.xpm_noise_w, rtol=1e-12)


def test_zhang_two_span_nli_is_coherent_not_exactly_double() -> None:
    """The two-span sum is coherent, so it is not exactly twice one span."""
    power = launch_profile(CASE)
    one_span_path = build_path(CASE)
    fiber = one_span_path.fiber_configs[0]
    edfa = one_span_path.edfa_configs[0]
    two_span_path = MultiSpanOpticalPath([fiber, fiber], [edfa, edfa])
    pump = _pump()
    one = make_model("zhang2024_dra_egn", CASE, raman_pumps=[pump]).evaluate_path(one_span_path, power)
    two = make_model("zhang2024_dra_egn", CASE, raman_pumps=[pump]).evaluate_path(two_span_path, power)
    ratio = _mean_nli(two) / _mean_nli(one)
    assert 1.0 < ratio < 4.0
    assert ratio != pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Profile fit and quadrature sanity
# ---------------------------------------------------------------------------


def test_average_link_squared_handles_zero_phase() -> None:
    """A zero phase slope must reduce to the squared zero-phase link."""
    fit = fit_buglia_profile(
        np.tile(np.linspace(1.0, 0.2, 8)[:, None], (1, 2)),
        100e3,
        np.array([195.0e12, 194.0e12]),
        np.full(2, 0.2 * np.log(10.0) / 10.0 / 1e3),
        200.0e12,
        0.25 * np.log(10.0) / 10.0 / 1e3,
        0.25 * np.log(10.0) / 10.0 / 1e3,
        n_steps=4,
    )
    nodes, weights = np.polynomial.legendre.leggauss(16)
    value = average_link_squared(fit, 195.0e12, 0.0, 32e9, 4.6e-5, nodes, weights)
    link_zero = fit.link(195.0e12, np.zeros(1))[0]
    assert value == pytest.approx(float(abs(link_zero) ** 2), rel=1e-12)


# ---------------------------------------------------------------------------
# SNR case
# ---------------------------------------------------------------------------


def test_snr_case_with_dra_beats_edfa_only() -> None:
    """At least one pump-on SNR case is finite and improves on EDFA-only."""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pcfm = make_model("pcfm", CASE).evaluate_path(path, power)
    buglia = make_model("buglia2024_ra_gn", CASE, raman_pumps=[PAPER_PUMPS[-1]]).evaluate_path(path, power)
    zhang = make_model("zhang2024_dra_egn", CL_CASE, raman_pumps=PAPER_PUMPS).evaluate_path(
        build_path(CL_CASE), launch_profile(CL_CASE)
    )
    assert float(np.mean(buglia.snr_db)) > float(np.mean(pcfm.snr_db))
    assert np.all(np.isfinite(zhang.snr_db))
    assert float(np.mean(zhang.snr_db)) > 0.0
