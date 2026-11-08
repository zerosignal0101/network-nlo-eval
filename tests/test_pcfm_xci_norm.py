"""Regression tests of the XCI edge-integral normalisation.

The exact island integral of :func:`network_nlo_eval.models.pcfm._xci_edge_weight_matrix`
is normalised by ``F_inf = int_0^inf W dt = (pi / (2 alpha)) (1 - e**(-2 alpha L))``.
The pre-fix code used ``(pi / (2 alpha)) (1 - e**(-alpha L))**2``, which inflated
the corrected XCI kernel by ``(1 + e**(-alpha L)) / (1 - e**(-alpha L))``.
On 100 km that is a 4.85 % bias, on 40 km 57.6 %, and it made ``edge_fix``
re-gate every C-band row below 100 km.  These tests pin the analytic limits
and the short-span behaviour so the bug cannot come back silently.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm import (
    XCI_EDGE_MATERIALITY,
    XCI_EDGE_VALIDITY_SWITCH,
    _xci_edge_weight_matrix,
    _xci_validity_s,
)

#: Attenuation of the benchmark SMF (about 0.163 dB/km), Np/m.
_ALPHA_NPM_M = 3.7437e-5
#: Span length of the analytic limit tests (m).
_SPAN_LENGTH_M = 100e3
#: Two-channel synthetic comb (Hz).
_F_ABS_HZ = np.array([193.0e12, 194.0e12])
_B_HZ = np.array([64.0e9, 64.0e9])


def _synthetic_pair(
    beta2_s2_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the ``(N=2)`` synthetic inputs of the exact-integral tests.

    ``gamma_xci`` is one, so the returned weight matrix *is* the kernel, and
    ``log_xci`` reproduces Eq. (24) from the band edges:
    ``log = |log((y2)/(y1))|``.  With two non-overlapping channels ``y1`` and
    ``y2`` are the same for both directions.
    """
    delta = _F_ABS_HZ[None, :] - _F_ABS_HZ[:, None]
    half_b = 0.5 * _B_HZ[None, :]
    ratio = (delta + half_b) / (delta - half_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_xci = np.abs(np.log(np.abs(ratio)))
    overlap = np.abs(delta) <= half_b
    log_xci = np.where(overlap, 0.0, log_xci)
    np.fill_diagonal(log_xci, 0.0)
    gamma_xci = np.ones_like(log_xci)
    beta2_eff_xci = np.full_like(log_xci, beta2_s2_m)
    return log_xci, gamma_xci, beta2_eff_xci, delta, half_b


def _edge_matrix(
    beta2_s2_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the exact island integral on the synthetic pair with the gate open."""
    log_xci, gamma_xci, beta2_eff_xci, _, _ = _synthetic_pair(beta2_s2_m)
    with np.errstate(divide="ignore", invalid="ignore"):
        k_xci = log_xci / (2.0 * np.pi * np.abs(beta2_eff_xci))
    k_xci = np.where(np.isfinite(k_xci), k_xci, 0.0)
    np.fill_diagonal(k_xci, 0.0)
    w_xci = gamma_xci**2 * k_xci
    # materiality=-1 forces every valid row through the correction, so the
    # returned matrix exposes the corrected kernel itself.
    w_edge, gated, _ = _xci_edge_weight_matrix(
        w_xci,
        k_xci,
        log_xci,
        gamma_xci,
        beta2_eff_xci,
        _F_ABS_HZ,
        _B_HZ,
        _ALPHA_NPM_M,
        _SPAN_LENGTH_M,
        validity_switch=1.0e30,
        materiality=-1.0,
    )
    return w_edge, k_xci, gated, beta2_eff_xci, log_xci


def test_xci_edge_kernel_converges_to_closed_form_at_large_s() -> None:
    """With both island edges at large ``s`` the exact kernel equals Eq. (24).

    At ``s -> infinity`` the island integral must reduce to the printed
    logarithmic kernel.  A constant normalisation error (the pre-fix
    ``(1 - e**(-ab))**2``) survives at every ``s``, so this ratio is a direct
    discriminator: the shipped bug gave 1.0485 on 100 km instead of 1.
    """
    # beta2=3e-26 gives s ~ 400 at both island edges; the gate now uses the
    # same s as the integral (2 pi**2 beta2 y B / alpha).
    w_edge, k_closed, gated, beta2_eff_xci, _ = _edge_matrix(3.0e-26)
    y1 = np.abs(_F_ABS_HZ[None, :] - _F_ABS_HZ[:, None]) - 0.5 * _B_HZ[None, :]
    s_pair = _xci_validity_s(beta2_eff_xci, y1 + _B_HZ[None, :], 0.5 * _B_HZ, _ALPHA_NPM_M)
    assert np.all(gated), "the synthetic pair must be corrected"
    for cut, nch in ((0, 1), (1, 0)):
        assert s_pair[cut, nch] > 250.0
        ratio = w_edge[cut, nch] / k_closed[cut, nch]
        assert abs(float(ratio) - 1.0) < 1.0e-3, (
            f"pair ({cut}, {nch}): exact/closed = {float(ratio):.6f} at s={float(s_pair[cut, nch]):.1f}"
        )


def test_xci_zero_dispersion_branch_matches_tanh_limit() -> None:
    """At ``beta2_eff = 0`` the corrected kernel must be ``2 B_cut B_nch/alpha tanh(ab/2)``.

    Expanding ``W`` around ``t = 0`` makes the island integral
    ``(1 - e**(-ab))**2 B_cut B_nch / alpha**2``; dividing by the correct
    ``F_inf`` gives the ``tanh`` limit.  The old ``F_inf`` and the old
    (unused) ``w0`` produced ``2 B_cut B_nch / alpha``, i.e. the
    ``tanh(ab/2) = 0.634`` shortfall at 40 km.
    """
    w_edge, _, gated, _, _ = _edge_matrix(0.0)
    ab = _ALPHA_NPM_M * _SPAN_LENGTH_M
    expected = 2.0 * _B_HZ[0] * _B_HZ[1] / _ALPHA_NPM_M * np.tanh(0.5 * ab)
    assert np.all(gated)
    for cut, nch in ((0, 1), (1, 0)):
        assert w_edge[cut, nch] == pytest.approx(float(expected), rel=1.0e-12)


@pytest.mark.parametrize("span_length_km", [40.0, 60.0, 80.0])
def test_short_span_c_band_edge_fix_is_immaterial(span_length_km: float) -> None:
    """Below 100 km the exact correction of the validated C band is below the gate.

    Pre-fix, the inflated normalisation gated 57/57 rows at 40 km with a
    maximum row change of 33 %, so ``edge_fix=True`` changed the already
    validated C-band result.  The true normalisation leaves the correction at
    about 0.7 %, below :data:`XCI_EDGE_MATERIALITY`, and the two paths must be
    bit identical.
    """
    case = replace(build_case("c_57ch"), span_length_km=span_length_km, span_count=1)
    path = build_path(case)
    power = launch_profile(case)
    kernel = make_model("pcfm", case, edge_fix=True).kernel_for(path.fiber_configs[0])
    gated, fraction = kernel.xci_edge_audit(span_length_km * 1e3)
    assert not np.any(gated), f"{span_length_km} km: the validated C band must not be gated"
    assert float(np.max(fraction)) < XCI_EDGE_MATERIALITY

    fixed = make_model("pcfm", case, edge_fix=True).evaluate_path(path, power)
    old = make_model("pcfm", case, edge_fix=False).evaluate_path(path, power)
    for field in ("snr_db", "xpm_noise_w", "spm_noise_w"):
        assert np.array_equal(np.asarray(getattr(fixed, field)), np.asarray(getattr(old, field))), (
            f"{span_length_km} km: edge_fix changed {field} on the validated C band"
        )


@pytest.mark.parametrize("case_name", ["c_57ch", "cl_152ch", "scl_277ch"])
def test_validated_100km_bands_keep_the_closed_form(case_name: str) -> None:
    """At 100 km the C / C+L / S+C+L rows must stay below the materiality gate.

    This is the kernel-level counterpart of
    ``test_pcfm.py::test_edge_fix_keeps_the_validated_bands_bit_identical``:
    the corrected XCI matrix is only *returned* on gated rows, so an empty
    gate is exactly why the validated numbers do not move.
    """
    case = build_case(case_name)
    path = build_path(case)
    kernel = make_model("pcfm", case, edge_fix=True).kernel_for(path.fiber_configs[0])
    gated, fraction = kernel.xci_edge_audit(case.span_length_km * 1e3)
    assert not np.any(gated), case_name
    assert float(np.max(fraction)) < XCI_EDGE_MATERIALITY, case_name


def _half_s(
    beta2_eff_xci: np.ndarray,
    y2_hz: np.ndarray,
    half_b_cut_hz: np.ndarray,
    alpha_npm_m: float,
) -> np.ndarray:
    """Return the pre-fix gate definition ``pi**2 |beta2| y B / alpha`` (half the integral)."""
    return 2.0 * np.pi**2 * np.abs(beta2_eff_xci) * y2_hz * half_b_cut_hz[:, None] / alpha_npm_m


def _integral_s(
    kernel,
    y_hz: np.ndarray,
    half_b_cut_hz: np.ndarray,
) -> np.ndarray:
    """Return the ``s`` the exact island integral evaluates internally."""
    c_mix = 4.0 * np.pi**2 * np.abs(kernel.beta2_eff_xci)
    return c_mix * y_hz * half_b_cut_hz[:, None] / kernel.alpha_visibility_npm


def test_xci_validity_s_is_the_integral_s() -> None:
    """The gate's ``s`` must be bit-for-bit the integral's ``s``, not half of it.

    The integral uses ``s = 4 pi**2 |beta2| y (B/2)/alpha``; the old helper
    returned ``2 pi**2 |beta2| y (B/2)/alpha``.  Anything but exact equality
    means the gate and the diagnostic have drifted apart again.
    """
    case = build_case("scl_277ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    delta = np.abs(kernel.f_abs_hz[:, None] - kernel.f_abs_hz[None, :])
    half_b = 0.5 * kernel.b_hz
    y2 = delta + half_b[None, :]

    s_gate = _xci_validity_s(kernel.beta2_eff_xci, y2, half_b, kernel.alpha_visibility_npm)
    s_integral = _integral_s(kernel, y2, half_b)
    mask = (kernel.log_xci != 0.0) & (delta - half_b[None, :] > 0.0)

    assert np.array_equal(s_gate[mask], s_integral[mask])
    # 旧口径恰好是一半；这个断言同时钉住“不要回退到 2 pi**2”。
    half = _half_s(kernel.beta2_eff_xci, y2, half_b, kernel.alpha_visibility_npm)
    assert np.array_equal(half[mask], 0.5 * s_integral[mask])


@pytest.mark.parametrize("case_name", ["c_57ch", "cl_152ch", "scl_277ch"])
@pytest.mark.parametrize("span_km", [40.0, 60.0, 80.0, 100.0])
def test_gate_mask_is_unchanged_by_the_semantics_fix(
    case_name: str,
    span_km: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old (half-s, switch 100) and new (true-s, switch 200) must gate identically.

    This is the regression that justifies the constant change: the whole
    validated band, at every audited span length, keeps the same rows and the
    same corrected matrix bit for bit.
    """
    import network_nlo_eval.models.pcfm as pcfm_module

    case = build_case(case_name)
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    length_m = span_km * 1e3

    def _gate():
        return _xci_edge_weight_matrix(
            kernel.w_xci,
            kernel.k_xci_matrix,
            kernel.log_xci,
            kernel.gamma_xci,
            kernel.beta2_eff_xci,
            kernel.f_abs_hz,
            kernel.b_hz,
            kernel.alpha_visibility_npm,
            length_m,
            validity_switch=switch,
        )

    monkeypatch.setattr(pcfm_module, "_xci_validity_s", _half_s)
    switch = 100.0
    w_old, gated_old, fraction_old = _gate()

    monkeypatch.setattr(pcfm_module, "_xci_validity_s", _xci_validity_s)
    switch = XCI_EDGE_VALIDITY_SWITCH
    w_new, gated_new, fraction_new = _gate()

    assert np.array_equal(gated_old, gated_new), f"{case_name} {span_km} km: gate mask moved"
    assert np.array_equal(w_old, w_new), f"{case_name} {span_km} km: corrected matrix moved"
    assert np.array_equal(fraction_old, fraction_new)


def test_xci_s_min_is_reported_in_integral_units() -> None:
    """``xci_s_min`` must equal the integral's ``s`` and use the 200 switch."""
    case = build_case("scl_277ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    power = launch_profile(case, None)
    length_m = case.span_length_km * 1e3
    report = kernel.validity_report(length_m, power)

    delta = np.abs(kernel.f_abs_hz[:, None] - kernel.f_abs_hz[None, :])
    half_b = 0.5 * kernel.b_hz
    y2 = delta + half_b[None, :]
    pairs = (kernel.log_xci != 0.0) & (delta - half_b[None, :] > 0.0)
    s_pair = _integral_s(kernel, y2, half_b)
    expected = np.where(pairs, s_pair, np.inf).min(axis=1)

    assert np.array_equal(np.asarray(report.xci_s_min), expected[kernel.inverse_order])
    assert np.array_equal(
        np.asarray(report.xci_s_below_switch),
        np.asarray(report.xci_s_min) < XCI_EDGE_VALIDITY_SWITCH,
    )

