"""Polynomial closed-form model (PCFM) acceptance tests.

Model under test: L. Poggiolini, M. Jiang, G. Gao, F. Forghieri, "Polynomial
Closed Form Model for Ultra-Wideband Transmission Systems", J. Lightwave
Technol., 2026, doi:10.1109/jlt.2026.3678322.

The file covers

* the bias of the PCFM against the numerical GN reference of this repository,
  which is the accuracy statement of Section IV of the paper;
* the stability of the SPP polynomial degree (Section IV-C);
* the linear accumulation of the NLI over transparent spans, Eq. (8);
* the lumped-loss capability of Section IV-D, exercised at kernel level;
* the equivalence of the closed-form kernels with the formulas printed in the
  paper (Appendix C for SCI, Eq. (24) for XCI);
* the auxiliary function ``J(X)`` of Eq. (30);
* the EGN correction of Zefreh et al. 2020, Eq. (14).

Every number quoted in the test output is produced by the test itself.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import sici

from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    build_case,
    build_path,
    grid_for,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.pcfm import (
    EGN_A_COEFFICIENTS,
    EGN_PHI,
    PCFM_MODEL_INFO,
    PCFMModel,
    PCFMSpanKernel,
    egn_rho_cut,
    egn_rho_nch,
)
from network_nlo_eval.models.pcfm_kernels import _j_batch_jit, _profile_energy_jit
from network_nlo_eval.network.elements import EDFAConfig

#: The numerical GN reference of the repository, used as the benchmark.
#: The test is skipped when the module is not importable; no stand-in
#: benchmark is ever built here.
_NUMERICAL_AVAILABLE = importlib.util.find_spec("network_nlo_eval.models.numerical") is not None

#: Per-channel launch power of the bias test (dBm).
BIAS_POWER_DBM = -1.0
#: Channel bandwidth of the kernel-level equivalence tests (Hz).
TEST_BANDWIDTH_HZ = 64e9
#: Span length of the kernel-level equivalence tests (m).
TEST_SPAN_LENGTH_M = 10e3


def _single_span_path(case) -> MultiSpanOpticalPath:
    """Path with a single span of the benchmark case."""
    fiber = build_path(case).fiber_configs[0]
    return MultiSpanOpticalPath(
        [fiber],
        [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)],
    )


def model_frequencies(case) -> np.ndarray:
    """Absolute centre frequencies of the case (Hz)."""
    return np.asarray(grid_for(case).frequencies, dtype=np.float64)


def model_bandwidths(case) -> np.ndarray:
    """Channel bandwidths of the case (Hz)."""
    return np.full(case.num_channels, float(case.symbol_rate_hz), dtype=np.float64)


def _nli_snr_db(result) -> np.ndarray:
    """GSNR_NLI of Eq. (31): the signal-to-NLI ratio in dB."""
    power = np.asarray(result.signal_power_w, dtype=np.float64)
    nli = np.asarray(result.spm_noise_w, dtype=np.float64) + np.asarray(
        result.xpm_noise_w, dtype=np.float64
    )
    return 10.0 * np.log10(power / nli)


def _j_by_quadrature(x: float) -> float:
    """``J(X) = int_0^X Si(t)/t dt`` by direct quadrature (independent of the model)."""
    return quad(lambda t: sici(t)[0] / t, 0.0, x, limit=400, epsabs=1e-12, epsrel=1e-12)[0]


def _appendix_c_np0(p0: float, b2: float, bandwidth_hz: float, length_m: float) -> float:
    """SCI kernel of Eq. (30) at ``N_p = 0``, as printed in Appendix C.

    ``K = 2 B^2 L^2 p0^2 H + 2 p0^2 (1-C)/(pi^4 b2^2 B^2)
    - 2 L p0^2 SI/(pi^2 b2)`` with ``H = J(x)/x``, ``x = pi^2 b2 B^2 L``.
    """
    x = np.pi**2 * b2 * bandwidth_hz**2 * length_m
    hyper = _j_by_quadrature(x) / x
    return (
        2.0 * bandwidth_hz**2 * length_m**2 * p0**2 * hyper
        + 2.0 * p0**2 * (1.0 - np.cos(x)) / (np.pi**4 * b2**2 * bandwidth_hz**2)
        - 2.0 * length_m * p0**2 * sici(x)[0] / (np.pi**2 * b2)
    )


def _appendix_c_np1(
    p0: float,
    p1: float,
    b2: float,
    bandwidth_hz: float,
    length_m: float,
) -> float:
    """SCI kernel of Eq. (30) at ``N_p = 1``, as printed in Appendix C."""
    x = np.pi**2 * b2 * bandwidth_hz**2 * length_m
    hyper = _j_by_quadrature(x) / x
    sine = np.sin(x)
    cosine = np.cos(x)
    sine_integral = sici(x)[0]
    b2b4 = np.pi**4 * b2**2 * bandwidth_hz**4
    bracket = (
        2.0 * p1**2
        + 9.0 * b2b4 * (2.0 * p0**2 + 2.0 * length_m * p0 * p1 + length_m**2 * p1**2)
        - 2.0
        * (p1**2 + b2b4 * (9.0 * p0**2 + 9.0 * length_m * p0 * p1 + 4.0 * length_m**2 * p1**2))
        * cosine
        + 6.0
        * np.pi**8
        * b2**4
        * bandwidth_hz**8
        * length_m**2
        * (3.0 * p0**2 + 3.0 * length_m * p0 * p1 + length_m**2 * p1**2)
        * hyper
        - 2.0 * np.pi**2 * b2 * bandwidth_hz**2 * length_m * p1**2 * sine
        - 2.0
        * np.pi**6
        * b2**3
        * bandwidth_hz**6
        * length_m
        * (9.0 * p0**2 + 9.0 * length_m * p0 * p1 + 4.0 * length_m**2 * p1**2)
        * sine_integral
    )
    return bracket / (9.0 * np.pi**8 * b2**4 * bandwidth_hz**6)


def _test_kernel(poly_degree: int, **kwargs) -> PCFMSpanKernel:
    """Single-channel kernel used by the closed-form equivalence tests."""
    centre_hz = 193.4e12
    return PCFMSpanKernel(
        np.array([centre_hz], dtype=np.float64),
        np.array([TEST_BANDWIDTH_HZ], dtype=np.float64),
        SMF_100KM,
        poly_degree=poly_degree,
        symbol_rate_hz=TEST_BANDWIDTH_HZ,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# accuracy against the numerical GN reference
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _NUMERICAL_AVAILABLE, reason="the numerical GN reference is not available")
def test_bias_vs_numerical_gn() -> None:
    """On cl_152ch, one span, -1 dBm, the mean GSNR_NLI bias stays in [-1.2, +0.2] dB.

    The paper reports a systematic bias of about -0.5 dB and a standard
    deviation of 0.1 dB for the PCFM against the GN benchmark (Section IV-C).
    """
    case = build_case("cl_152ch")
    path = _single_span_path(case)
    power = launch_profile(case, BIAS_POWER_DBM)

    pcfm = make_model("pcfm", case).evaluate_path(path, power)
    reference = make_model("numerical", case).evaluate_path(path, power)

    delta = _nli_snr_db(pcfm) - _nli_snr_db(reference)
    mean_delta = float(np.mean(delta))
    std_delta = float(np.std(delta))
    print(
        f"cl_152ch 1 span -1 dBm: mean Delta GSNR_NLI {mean_delta:+.4f} dB, "
        f"std {std_delta:.4f} dB, min {float(np.min(delta)):+.4f} dB, max {float(np.max(delta)):+.4f} dB"
    )
    assert -1.2 <= mean_delta <= 0.2, f"mean Delta GSNR_NLI {mean_delta:+.4f} dB is out of [-1.2, +0.2] dB"


@pytest.mark.skipif(not _NUMERICAL_AVAILABLE, reason="the numerical GN reference is not available")
def test_sci_and_xci_match_the_numerical_reference() -> None:
    """The SCI and the XCI part must each track the numerical reference."""
    case = build_case("cl_152ch")
    path = _single_span_path(case)
    power = launch_profile(case, BIAS_POWER_DBM)

    pcfm = make_model("pcfm", case).evaluate_path(path, power)
    reference = make_model("numerical", case).evaluate_path(path, power)

    for name, mine, theirs in (
        ("SCI", np.asarray(pcfm.spm_noise_w), np.asarray(reference.spm_noise_w)),
        ("XCI", np.asarray(pcfm.xpm_noise_w), np.asarray(reference.xpm_noise_w)),
    ):
        delta = 10.0 * np.log10(mine / theirs)
        print(f"{name}: mean {float(np.mean(delta)):+.4f} dB, std {float(np.std(delta)):.4f} dB")
        assert abs(float(np.mean(delta))) < 0.3, f"{name} bias {float(np.mean(delta)):+.4f} dB exceeds 0.3 dB"
        assert float(np.std(delta)) < 0.2, f"{name} spread {float(np.std(delta)):.4f} dB exceeds 0.2 dB"


def test_poly_degree_stability() -> None:
    """``poly_degree=5`` and ``poly_degree=9`` must agree within 0.1 dB per channel."""
    case = build_case("cl_152ch")
    path = _single_span_path(case)
    power = launch_profile(case, BIAS_POWER_DBM)

    coarse = make_model("pcfm", case, poly_degree=5).evaluate_path(path, power)
    fine = make_model("pcfm", case, poly_degree=9).evaluate_path(path, power)

    delta = np.abs(np.asarray(coarse.snr_db) - np.asarray(fine.snr_db))
    max_delta = float(np.max(delta))
    print(f"poly_degree 5 vs 9: max per-channel SNR difference {max_delta:.5f} dB")
    assert max_delta < 0.1, f"SNR difference between N_p=5 and N_p=9 is {max_delta:.4f} dB"


def test_transparent_spans_accumulate_linearly() -> None:
    """Over transparent spans the NLI adds up, Eq. (8).

    The EDFA is left without a target gain, so the route layer compensates the
    span loss exactly and the signal power at the input of every span is the
    launch power.  The transfer factor of Eq. (7) is then one and the NLI of
    ``N`` spans must be ``N`` times the NLI of a single span.
    """
    case = build_case("scaling_40ch")
    fiber = build_path(case).fiber_configs[0]
    power = launch_profile(case, None)

    def _run(span_count: int):
        path = MultiSpanOpticalPath(
            [fiber] * span_count,
            [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)] * span_count,
        )
        return make_model("pcfm", case).evaluate_path(path, power)

    single = _run(1)
    five = _run(5)

    single_nli = np.asarray(single.spm_noise_w) + np.asarray(single.xpm_noise_w)
    five_nli = np.asarray(five.spm_noise_w) + np.asarray(five.xpm_noise_w)
    relative_error = np.abs(five_nli / (5.0 * single_nli) - 1.0)
    print(
        "transparent spans: max relative NLI error over 5 spans "
        f"{float(np.max(relative_error)):.3e}, max deviation {float(np.max(np.abs(five_nli / single_nli - 5.0))):.3e}"
    )
    assert np.max(relative_error) < 0.02, f"NLI does not accumulate linearly: {relative_error}"


def test_lumped_loss_supported() -> None:
    """A lumped loss inside the span must give a finite and sensible result.

    The framework has no field for lumped loss, so the feature is exposed at
    kernel level through the ``lumped_loss`` argument: a list of
    ``(position_m, loss_db)`` elements.
    """
    case = build_case("scaling_40ch")
    fiber = build_path(case).fiber_configs[0]
    length_m = case.span_length_km * 1e3
    power = launch_profile(case, None)

    plain = PCFMSpanKernel(
        model_frequencies(case),
        model_bandwidths(case),
        fiber,
        poly_degree=5,
        symbol_rate_hz=case.symbol_rate_hz,
    )
    lossy = PCFMSpanKernel(
        model_frequencies(case),
        model_bandwidths(case),
        fiber,
        poly_degree=5,
        symbol_rate_hz=case.symbol_rate_hz,
        lumped_loss=[(50e3, 3.0)],
    )

    out_plain = plain.evaluate(power, length_m, 1)
    out_lossy = lossy.evaluate(power, length_m, 1)

    for name, values in (
        ("SCI", out_lossy.spm_noise_w),
        ("XCI", out_lossy.xpm_noise_w),
        ("power", out_lossy.power_out_w),
    ):
        assert np.all(np.isfinite(np.asarray(values))), f"{name} is not finite with a lumped loss"
        assert np.all(np.asarray(values) > 0.0), f"{name} is not positive with a lumped loss"

    # The 3 dB element removes power, so the output power must drop by about
    # 3 dB on top of the span loss, and the NLI must stay in the same ballpark
    # as the lossless case (paper, Section IV-D).
    drop_db = 10.0 * np.log10(
        np.asarray(out_plain.power_out_w) / np.asarray(out_lossy.power_out_w)
    )
    assert np.all(drop_db > 2.9) and np.all(drop_db < 3.6), f"unexpected power drop {drop_db}"

    nli_plain = np.asarray(out_plain.spm_noise_w) + np.asarray(out_plain.xpm_noise_w)
    nli_lossy = np.asarray(out_lossy.spm_noise_w) + np.asarray(out_lossy.xpm_noise_w)
    ratio_db = 10.0 * np.log10(nli_lossy / nli_plain)
    print(
        f"lumped loss 3 dB at 50 km: power drop {float(np.mean(drop_db)):.3f} dB, "
        f"NLI change {float(np.mean(ratio_db)):+.3f} dB "
        f"(min {float(np.min(ratio_db)):+.3f}, max {float(np.max(ratio_db)):+.3f})"
    )
    assert np.all(ratio_db < 0.0), "a lumped loss must not increase the NLI"
    assert np.all(ratio_db > -3.0), f"NLI change {ratio_db} dB is out of the expected range"

    with pytest.raises(ValueError, match="outside the span"):
        PCFMSpanKernel(
            model_frequencies(case),
            model_bandwidths(case),
            fiber,
            poly_degree=5,
            symbol_rate_hz=case.symbol_rate_hz,
            lumped_loss=[(2.0 * length_m, 1.0)],
        ).evaluate(power, length_m, 1)


# ---------------------------------------------------------------------------
# closed-form kernels against the printed formulas
# ---------------------------------------------------------------------------


def test_sci_kernel_matches_paper_appendix_c_np0() -> None:
    """At ``N_p = 0`` the kernel must reproduce the closed form of Appendix C."""
    kernel = _test_kernel(poly_degree=0)
    q_scaled = np.array([[0.87]], dtype=np.float64)

    modelled = float(kernel.sci_kernel(TEST_SPAN_LENGTH_M, q_scaled)[0])
    reference = _appendix_c_np0(
        p0=0.87,
        b2=float(kernel.beta2_eff_sci[0]),
        bandwidth_hz=float(kernel.b_hz[0]),
        length_m=TEST_SPAN_LENGTH_M,
    )
    relative_error = abs(modelled / reference - 1.0)
    print(f"N_p=0 SCI kernel: model {modelled:.9e}, Appendix C {reference:.9e}, rel err {relative_error:.3e}")
    assert relative_error < 1e-9, f"N_p=0 SCI kernel differs from Appendix C by {relative_error:.3e}"


def test_sci_kernel_matches_paper_appendix_c_np1() -> None:
    """At ``N_p = 1`` the kernel must reproduce the closed form of Appendix C."""
    kernel = _test_kernel(poly_degree=1)
    p0, p1 = 0.83, 1.7e-6
    # The kernel works in the scaled basis q_n = p_n L^n.
    q_scaled = np.array([[p0], [p1 * TEST_SPAN_LENGTH_M]], dtype=np.float64)

    modelled = float(kernel.sci_kernel(TEST_SPAN_LENGTH_M, q_scaled)[0])
    reference = _appendix_c_np1(
        p0=p0,
        p1=p1,
        b2=float(kernel.beta2_eff_sci[0]),
        bandwidth_hz=float(kernel.b_hz[0]),
        length_m=TEST_SPAN_LENGTH_M,
    )
    relative_error = abs(modelled / reference - 1.0)
    print(f"N_p=1 SCI kernel: model {modelled:.9e}, Appendix C {reference:.9e}, rel err {relative_error:.3e}")
    assert relative_error < 1e-9, f"N_p=1 SCI kernel differs from Appendix C by {relative_error:.3e}"


def test_xci_kernel_matches_paper_eq24() -> None:
    """The XCI kernel must match Eq. (24) and its logarithmic frequency factor.

    Eq. (24) reads ``K_XCI = L/(2 pi |beta2_eff|) |log(...)| sum_{n,k} (2-delta)
    p_n p_k L^(n+k)/(n+k+1)``, which is the profile energy
    ``int_0^L p^2 dz`` times the logarithmic factor of the stretched island.
    """
    case = build_case("scaling_40ch")
    fiber = build_path(case).fiber_configs[0]
    kernel = PCFMSpanKernel(
        model_frequencies(case),
        model_bandwidths(case),
        fiber,
        poly_degree=5,
        symbol_rate_hz=case.symbol_rate_hz,
    )
    power = launch_profile(case, None)
    length_m = case.span_length_km * 1e3
    q_scaled = kernel.profile_polynomial(length_m, power)
    k_xci = kernel.xci_kernel(length_m, q_scaled)

    # 1. The logarithmic factor is the stretched-island frequency integral.
    for cut in (0, 10, 20):
        for nch in (5, 15, 25):
            delta_f = float(kernel.f_abs_hz[nch] - kernel.f_abs_hz[cut])
            half_b = 0.5 * float(kernel.b_hz[nch])
            direct = abs(np.log(abs((delta_f + half_b) / (delta_f - half_b))))
            assert abs(float(kernel.log_xci[cut, nch]) - direct) < 1e-12

    # 2. The kernel equals the printed formula of Eq. (24) for every pair.
    #    Eq. (24): K_XCI = L/(2 pi |beta2_eff|) |log(...)| sum_{n,k} (2-delta)
    #    p_n p_k L^(n+k)/(n+k+1), with the z-basis coefficients p_n.
    energy = _profile_energy_jit(np.ascontiguousarray(q_scaled)) * length_m
    z_basis = q_scaled / length_m ** np.arange(q_scaled.shape[0])[:, None]
    for cut in (0, 10, 20):
        for nch in (5, 15, 25):
            beta2 = float(kernel.beta2_eff_xci[cut, nch])
            symmetric = 0.0
            for n in range(z_basis.shape[0]):
                for k in range(n, z_basis.shape[0]):
                    weight = 1.0 if n == k else 2.0
                    symmetric += (
                        weight
                        * float(z_basis[n, nch])
                        * float(z_basis[k, nch])
                        * length_m ** (n + k)
                        / (n + k + 1)
                    )
            printed = (
                length_m
                / (2.0 * np.pi * abs(beta2))
                * abs(float(kernel.log_xci[cut, nch]))
                * symmetric
            )
            modelled = float(kernel.xci_kernel(length_m, q_scaled)[cut, nch])
            assert abs(modelled / printed - 1.0) < 1e-9
            # The profile energy of the printed formula is int_0^L p^2 dz.
            assert abs(float(energy[nch]) / (length_m * symmetric) - 1.0) < 1e-9

    # 3. The profile energy is the integral of the squared polynomial.
    for nch in (0, 17, 33):
        polynomial = np.poly1d(list(reversed(q_scaled[:, nch].tolist())))
        u = np.linspace(0.0, 1.0, 20001)
        reference = length_m * np.trapz(polynomial(u) ** 2, u)
        assert abs(float(energy[nch]) / reference - 1.0) < 1e-6


def test_j_function_matches_quadrature() -> None:
    """``J(X) = int_0^X Si(t)/t dt`` must match a direct quadrature."""
    from network_nlo_eval.models.pcfm import _j_constant

    values = np.array([0.5, 2.0, 8.0, 20.0, -3.0], dtype=np.float64)
    modelled = _j_batch_jit(values, _j_constant())
    for value, model_value in zip(values, modelled, strict=True):
        reference = _j_by_quadrature(float(value))
        assert abs(float(model_value) / reference - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# SPP polynomial fit
# ---------------------------------------------------------------------------


def test_spp_polynomial_fit_quality() -> None:
    """The SPP polynomial fit must improve with the degree and converge at 9.

    The reference is the same ISRS integration carried out with four times
    as many RK4 steps (the 32-step integration itself is converged to 0.18 %
    in the profile, so the residual measured here is the fitting error).  The
    paper observes the same behaviour in Fig. 11: at ``N_p = 5`` the fit of
    the most demanding channel is still visibly off, while the repeated
    integration inside the kernels averages the fitting error out (see
    :func:`test_poly_degree_stability`).
    """
    case = build_case("cl_152ch")
    fiber = build_path(case).fiber_configs[0]
    power = launch_profile(case, BIAS_POWER_DBM)
    length_m = case.span_length_km * 1e3

    reference = PCFMSpanKernel(
        model_frequencies(case),
        model_bandwidths(case),
        fiber,
        poly_degree=9,
        symbol_rate_hz=case.symbol_rate_hz,
        spp_steps=128,
    )
    reference_positions, reference_profile = reference.spp_samples(length_m, power)

    errors: list[tuple[int, float, float]] = []
    for degree in (3, 5, 7, 9):
        kernel = PCFMSpanKernel(
            model_frequencies(case),
            model_bandwidths(case),
            fiber,
            poly_degree=degree,
            symbol_rate_hz=case.symbol_rate_hz,
        )
        q_scaled = kernel.profile_polynomial(length_m, power)
        fitted = np.vander(reference_positions, degree + 1, increasing=True) @ q_scaled
        relative = np.abs(fitted - reference_profile) / np.maximum(np.abs(reference_profile), 1e-12)
        max_error = float(np.max(relative))
        rms_error = float(np.sqrt(np.mean(relative**2)))
        print(
            f"N_p={degree}: max SPP fit error {max_error * 100:.4f} %, RMS {rms_error * 100:.4f} %"
        )
        errors.append((degree, max_error, rms_error))

    by_degree = {degree: (max_error, rms_error) for degree, max_error, rms_error in errors}
    for degree in (3, 5, 7):
        assert by_degree[degree][0] > by_degree[degree + 2][0], (
            f"the SPP fit error must decrease from N_p={degree} to N_p={degree + 2}"
        )
    assert by_degree[5][1] < 0.02, f"N_p=5 RMS SPP fit error {by_degree[5][1]:.4e} exceeds 0.02"
    assert by_degree[9][0] < 0.005, f"N_p=9 max SPP fit error {by_degree[9][0]:.4e} exceeds 0.005"
    assert by_degree[9][1] < 0.0005, f"N_p=9 RMS SPP fit error {by_degree[9][1]:.4e} exceeds 0.0005"


# ---------------------------------------------------------------------------
# EGN correction
# ---------------------------------------------------------------------------


def test_egn_rho_matches_paper_equation() -> None:
    """The machine-learning factors must reproduce Eq. (14) of Zefreh et al. 2020."""
    beta2_acc = np.array([0.0, 1.0e4, -5.0e4], dtype=np.float64)
    rate_hz = 64e9
    roll_off = 0.0

    a = EGN_A_COEFFICIENTS
    phi = EGN_PHI
    acc = np.abs(beta2_acc)
    expected_nch = (1.0 + a["a19"] * roll_off ** a["a20"] + a["a21"] * roll_off ** a["a22"]) * (
        a["a1"]
        + a["a2"] * phi ** a["a3"]
        + a["a4"] * phi ** a["a5"] * (1.0 + a["a6"] * (acc + a["a7"]) ** a["a8"])
    )
    expected_cut = (1.0 + a["a23"] * roll_off ** a["a24"]) * (
        a["a9"]
        + a["a10"] * phi ** a["a11"]
        + a["a12"]
        * phi ** a["a13"]
        * (
            1.0
            + a["a14"] * (rate_hz / 1e12) ** a["a15"]
            + a["a16"] * (acc + a["a17"]) ** a["a18"]
        )
    )

    np.testing.assert_allclose(egn_rho_nch(beta2_acc), expected_nch, rtol=1e-12)
    np.testing.assert_allclose(egn_rho_cut(beta2_acc, rate_hz), expected_cut, rtol=1e-12)


def test_egn_correction_scales_sci_and_xci() -> None:
    """With ``egn=True`` the SCI is scaled by ``rho_CUT`` and the XCI by ``rho_nch``.

    The format constant of the Gaussian channel is ``Phi = 0`` and the roll-off
    is zero, so both factors reduce to constants: ``rho_CUT = a9`` and
    ``rho_nch = a1`` of Table IV of Zefreh et al. 2020.
    """
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)

    gn = make_model("pcfm", case).evaluate_path(path, power)
    egn = make_model("pcfm", case, egn=True).evaluate_path(path, power)

    sci_ratio = np.asarray(egn.spm_noise_w) / np.asarray(gn.spm_noise_w)
    xci_ratio = np.asarray(egn.xpm_noise_w) / np.asarray(gn.xpm_noise_w)
    print(
        f"EGN correction: SCI factor {sci_ratio.mean():.6f} (rho_CUT = {EGN_A_COEFFICIENTS['a9']:.6f}), "
        f"XCI factor {xci_ratio.mean():.6f} (rho_nch = {EGN_A_COEFFICIENTS['a1']:.6f})"
    )
    np.testing.assert_allclose(sci_ratio, EGN_A_COEFFICIENTS["a9"], rtol=1e-10)
    np.testing.assert_allclose(xci_ratio, EGN_A_COEFFICIENTS["a1"], rtol=1e-10)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_model_is_registered_with_the_expected_metadata() -> None:
    """The model must be registered as ``pcfm`` with the documented options."""
    from network_nlo_eval.models.base import available_models
    from network_nlo_eval.models.base import create_model as _create

    info = available_models()["pcfm"]
    assert info.name == "pcfm"
    assert info.family == "gn"
    assert "10.1109/jlt.2026.3678322" in info.reference
    assert [option[0] for option in info.options] == ["poly_degree", "egn"]
    assert PCFM_MODEL_INFO == info

    case = build_case("scaling_40ch")
    model = _create("pcfm", grid_for(case), build_path(case).fiber_configs[0], symbol_rate_hz=64e9)
    assert isinstance(model, PCFMModel)
    with pytest.raises(ValueError, match="poly_degree"):
        _create(
            "pcfm",
            grid_for(case),
            build_path(case).fiber_configs[0],
            symbol_rate_hz=64e9,
            poly_degree=-1,
        )
