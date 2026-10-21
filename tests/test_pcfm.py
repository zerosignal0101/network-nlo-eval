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
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath, SpanContext
from network_nlo_eval.models.pcfm import (
    EGN_A_COEFFICIENTS,
    EGN_PHI,
    PCFM_MODEL_INFO,
    PCFMModel,
    PCFMSpanKernel,
    _coherent_factor,
    _equivalent_phase_ramp,
    _gn_weighted_xpm_visibility,
    _self_beating_visibility,
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

#: Pre-change (pure power convention) bit-exact baseline of the single-span
#: link, for ``(case, egn) -> (SNR_dB, SPM_W, XPM_W)`` at the mid channel.
#:
#: The inter-span coherence degenerates to the single span when the link has
#: one span: the route layer keeps the pure power branch, so every bit of the
#: noise and of the SNR must be the one of the implementation without
#: coherence.  The numbers are taken from the implementation at the commit
#: before the coherence was added.
PRE_CHANGE_SINGLE_SPAN = {
    ("scaling_40ch", False): (27.774925839000055, 1.5657871571852133e-07, 3.0948944394444263e-07),
    ("scaling_40ch", True): (27.803266272644372, 1.32279264826164e-07, 3.2298318370042034e-07),
    ("scaling_160ch", False): (27.589510907224284, 1.264131064364181e-07, 3.352181851990189e-07),
    ("scaling_160ch", True): (27.602119772219286, 1.0679505644855038e-07, 3.498336980736962e-07),
    ("c_57ch", False): (26.520897523858203, 8.556243108642051e-07, 1.855548265913478e-06),
    ("c_57ch", True): (26.57892953449453, 7.228399740611891e-07, 1.936450170307306e-06),
    ("cl_152ch", False): (27.425820112126246, 7.489518949850022e-08, 1.951868545624842e-07),
    ("cl_152ch", True): (27.43485281929605, 6.327220504022797e-08, 2.0369700142140856e-07),
    ("escl_479ch", False): (22.636569650334728, 5.072100113450804e-08, 2.194110831711496e-07),
    ("escl_479ch", True): (22.63452402288508, 4.2849608968443736e-08, 2.2897740639741178e-07),
}

#: ``c_57ch``, 10 spans, SMF: the SPM inter-span buildup.
#:
#: The brief asks for ``1.40 +- 0.03``; the reference values are the
#: independent quadrature of the work package (1.402) and the numerical GN
#: model of the repository (1.4079).  The visibility caliber of this model is
#: the one of the numerical model, so the value must land on 1.4079.
SPM_BUILDUP_TARGET = 1.4079
SPM_BUILDUP_TOLERANCE = 0.03

#: Number of spans of the buildup tests.
COHERENCE_SPANS = 10

#: Span length of the coherence tests (m).
COHERENCE_SPAN_LENGTH_M = 100e3


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


def test_transparent_spans_accumulate_with_the_coherent_factor() -> None:
    """Over transparent spans the NLI follows the coherent factor, not Eq. (8).

    The EDFA is left without a target gain, so the route layer compensates the
    span loss exactly and the signal power at the input of every span is the
    launch power.  The transfer factor of Eq. (7) is then one and the NLI of
    ``N`` spans is the single-span NLI times the coherent factor
    ``R(N) = N + 2 sum_k (N-k) V(k L)``.

    Eq. (8) of the paper, the linear accumulation, is the *incoherent* limit
    of that relation (``V = 0`` gives ``R = N``).  The inter-span coherence
    replaces it: on SMF the factor is above ``N``, because the GN kernel
    concentrates the NLI power on the phase-matched frequency pairs, and those
    pairs stay coherent across the spans.  The test therefore asserts the
    analytic factor (an exact statement) instead of the old "five times the
    single span" (which only held in the incoherent limit).
    """
    case = build_case("scaling_40ch")
    fiber = build_path(case).fiber_configs[0]
    power = launch_profile(case, None)
    model = make_model("pcfm", case)
    kernel = model.kernel_for(fiber)

    def _run(span_count: int):
        path = MultiSpanOpticalPath(
            [fiber] * span_count,
            [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)] * span_count,
        )
        return model.evaluate_path(path, power)

    single = _run(1)
    five = _run(5)
    factor_spm, factor_xpm = kernel.inter_span_coherent_factor(5, 100e3)

    spm_ratio = np.asarray(five.spm_noise_w) / np.asarray(single.spm_noise_w)
    xpm_ratio = np.asarray(five.xpm_noise_w) / np.asarray(single.xpm_noise_w)
    assert np.allclose(spm_ratio, factor_spm, rtol=1e-9, atol=0.0)
    assert np.allclose(xpm_ratio, factor_xpm, rtol=1e-9, atol=0.0)

    # The coherent gain must be real: the old linear law (a factor of five)
    # no longer holds, otherwise the assertion above would be vacuous.
    assert np.all(factor_spm > 5.0)
    assert np.max(np.abs(spm_ratio / 5.0 - 1.0)) > 0.02
    print(
        "transparent spans: coherent factor "
        f"{float(factor_spm[0]):.4f} (linear law: 5.0), max deviation "
        f"{float(np.max(np.abs(spm_ratio / 5.0 - 1.0))):.3e}"
    )


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


# ---------------------------------------------------------------------------
# inter-span coherence (this round of work)
# ---------------------------------------------------------------------------
#
# The PCFM paper accumulates the NLI linearly over transparent spans, Eq. (8).
# That is the incoherent limit; the repository now uses the convention in
# which the inter-span visibility is weighted by the NLI noise power and the
# route layer sums |sum_k a_k|**2.  The caliber and the derivation are
# documented in :mod:`network_nlo_eval.models.pcfm`.


def _coherence_path(case, n_spans: int) -> MultiSpanOpticalPath:
    """Path of ``n_spans`` identical spans of the benchmark fibre."""
    fiber = build_path(case).fiber_configs[0]
    return MultiSpanOpticalPath(
        [fiber] * n_spans,
        [EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)] * n_spans,
    )


def _graded_grid(low: float, high: float, n_points: int, power: float = 2.0) -> NDArrayFloat:
    """Grid on ``[low, high]`` clustered towards the band centre.

    The GN weight ``1/(abar**2 + theta**2)`` peaks where the mismatch
    vanishes, i.e. at the corner of the admissible region that contains the
    origin.  A uniform grid under-resolves that peak, so the independent
    reference below uses a clustered grid.
    """
    s = np.linspace(0.0, 1.0, n_points)
    s = np.sign(s) * np.abs(s) ** power
    return low + (high - low) * s


def _independent_visibility(
    beta2_abs_s2_m: float,
    alpha_npm_m: float,
    bandwidth_hz: float,
    span_length_m: float,
    distances_m: NDArrayFloat,
    n_frequency: int = 21,
    n_offset: int = 801,
) -> NDArrayFloat:
    """Direct two-dimensional quadrature of the NLI-weighted visibility.

    This is an independent reference: it integrates the GN weight over the
    whole ``(observed frequency, first offset, second offset)`` domain
    without the one-dimensional reduction and without the tabulated
    antiderivatives of the model.

    Args:
        beta2_abs_s2_m: ``|beta_2|`` (s^2/m).
        alpha_npm_m: Power attenuation of the fibre (Np/m).
        bandwidth_hz: Channel bandwidth (Hz).
        span_length_m: Length of one span (m).
        distances_m: ``(K,)`` distances between the two copies (m).
        n_frequency: Number of observed frequencies.
        n_offset: Number of nodes per offset direction.

    Returns
    -------
        The ``(K,)`` visibility.
    """
    half_band = 0.5 * bandwidth_hz
    abar = alpha_npm_m * span_length_m
    exp_abar = float(np.exp(-abar))
    rate = 4.0 * np.pi**2 * beta2_abs_s2_m * span_length_m
    index = np.arange(n_frequency, dtype=np.float64)
    frequencies = -half_band * np.cos(np.pi * index / (n_frequency - 1))
    frequency_weights = np.empty(n_frequency, dtype=np.float64)
    frequency_weights[0] = 0.5 * (frequencies[1] - frequencies[0])
    frequency_weights[-1] = 0.5 * (frequencies[-1] - frequencies[-2])
    frequency_weights[1:-1] = 0.5 * (frequencies[2:] - frequencies[:-2])
    numerator = np.zeros(distances_m.size, dtype=np.float64)
    denominator = 0.0
    for frequency, frequency_weight in zip(frequencies, frequency_weights, strict=True):
        low = -half_band - float(frequency)
        high = half_band - float(frequency)
        axis = _graded_grid(low, high, n_offset)
        weights = np.empty(n_offset, dtype=np.float64)
        weights[0] = 0.5 * (axis[1] - axis[0])
        weights[-1] = 0.5 * (axis[-1] - axis[-2])
        weights[1:-1] = 0.5 * (axis[2:] - axis[:-2])
        first, second = np.meshgrid(axis, axis, indexing="ij")
        admissible = (first + second >= low) & (first + second <= high)
        theta = np.abs(rate * first * second)
        weight = np.where(
            admissible,
            (1.0 - 2.0 * exp_abar * np.cos(theta) + exp_abar**2) / (abar**2 + theta**2),
            0.0,
        )
        measure = weights[:, None] * weights[None, :]
        cosine = np.cos(theta[None, :, :] * (distances_m[:, None, None] / span_length_m))
        numerator += frequency_weight * np.sum(
            np.where(admissible[None, :, :], weight[None, :, :] * cosine, 0.0) * measure[None, :, :],
            axis=(1, 2),
        )
        denominator += frequency_weight * np.sum(weight * measure)
    return numerator / denominator


def test_model_declares_span_context_support() -> None:
    """The model must declare the inter-span coherence support."""
    assert PCFMModel.supports_span_context is True
    case = build_case("scaling_40ch")
    model = make_model("pcfm", case)
    assert getattr(model, "supports_span_context", False) is True


def test_single_span_is_bit_identical_to_the_pre_change_model() -> None:
    """A single-span link must be bit-for-bit identical to the pre-change model.

    The inter-span coherence degenerates to the single span when the link has
    one span: the route layer keeps the pure power branch, so every bit of the
    noise and of the SNR must be the one of the implementation without
    coherence.
    """
    for (case_name, egn), (snr_db, spm_w, xpm_w) in PRE_CHANGE_SINGLE_SPAN.items():
        case = build_case(case_name)
        power = launch_profile(case, None)
        model = make_model("pcfm", case, egn=egn)
        result = model.evaluate_path(_coherence_path(case, 1), power)
        index = case.num_channels // 2
        assert result.snr_db[index] == snr_db, f"{case_name} egn={egn}: the SNR changed"
        assert result.spm_noise_w[index] == spm_w, f"{case_name} egn={egn}: the SPM changed"
        assert result.xpm_noise_w[index] == xpm_w, f"{case_name} egn={egn}: the XPM changed"


def test_single_span_uses_the_power_only_branch() -> None:
    """With one span the kernel must not hand out complex amplitudes."""
    case = build_case("scaling_40ch")
    fiber = build_path(case).fiber_configs[0]
    power = launch_profile(case, None)
    model = make_model("pcfm", case)
    step = model.span_step(
        fiber,
        EDFAConfig(noise_figure_db=case.edfa_noise_figure_db),
        power,
        100e3,
        span_context=SpanContext(
            span_index=0, total_spans=1, cumulative_length_m=0.0, span_length_m=100e3
        ),
    )
    assert step.nli_amplitude_spm is None
    assert step.nli_amplitude_xpm is None


def test_amplitude_magnitude_equals_the_noise_power() -> None:
    """``|a|**2`` must equal the noise power of the same name (``rtol=1e-12``)."""
    case = build_case("scaling_40ch")
    fiber = build_path(case).fiber_configs[0]
    power = launch_profile(case, None)
    model = make_model("pcfm", case)
    step = model.span_step(
        fiber,
        EDFAConfig(noise_figure_db=case.edfa_noise_figure_db),
        power,
        100e3,
        span_context=SpanContext(
            span_index=2, total_spans=10, cumulative_length_m=200e3, span_length_m=100e3
        ),
    )
    assert step.nli_amplitude_spm is not None
    assert step.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(step.nli_amplitude_spm) ** 2, step.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(step.nli_amplitude_xpm) ** 2, step.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_c_57ch_ten_span_buildup_matches_the_brief() -> None:
    """``c_57ch``, 10 spans: the SPM buildup must land on 1.40 +- 0.03.

    Caliber: the visibility weighted by the NLI power.  The reference values
    are 1.402 (independent quadrature of the work package) and 1.4079 (the
    numerical GN model of the repository, whose caliber this model follows).
    """
    case = build_case("c_57ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    factor_spm, _ = kernel.inter_span_coherent_factor(COHERENCE_SPANS, COHERENCE_SPAN_LENGTH_M)
    buildup = factor_spm / COHERENCE_SPANS
    assert np.all(
        np.abs(buildup - SPM_BUILDUP_TARGET) <= SPM_BUILDUP_TOLERANCE
    ), f"SPM buildup {buildup.min():.4f}..{buildup.max():.4f} outside the brief"
    assert np.allclose(buildup, SPM_BUILDUP_TARGET, rtol=1e-4)


def test_buildup_table_is_monotone_in_the_span_count() -> None:
    """The buildup grows with the span count and starts at one."""
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = make_model("pcfm", case)
    single = model.evaluate_path(_coherence_path(case, 1), power)
    values = []
    for n_spans in (1, 2, 5, 10):
        result = model.evaluate_path(_coherence_path(case, n_spans), power)
        values.append(
            float(
                np.mean(
                    np.asarray(result.spm_noise_w) / np.asarray(single.spm_noise_w) / n_spans
                )
            )
        )
    assert values[0] == pytest.approx(1.0, rel=1e-12)
    assert values == sorted(values)
    assert values[-1] == pytest.approx(SPM_BUILDUP_TARGET, abs=SPM_BUILDUP_TOLERANCE)
    print(f"SPM buildup over the span count: {values}")


def test_zero_dispersion_gives_full_coherence() -> None:
    """Without dispersion the inter-span accumulation is fully coherent.

    With ``D -> 0`` the visibility goes to one, so the coherent factor goes
    to ``N**2`` and the buildup to ``N``.  The *noise* of the PCFM itself is
    singular at exactly zero dispersion (the kernels carry ``1/beta2``), so
    the test asserts on the coherent factor, which is the quantity the
    inter-span coherence is about.
    """
    case = build_case("c_57ch")
    fiber = build_path(case).fiber_configs[0]
    near_zero = fiber.model_copy(
        update={"dispersion_parameter_d": 1e-9, "dispersion_slope_s": 1e-9}
    )
    model = make_model("pcfm", case)
    kernel = model.kernel_for(near_zero)
    factor_spm, factor_xpm = kernel.inter_span_coherent_factor(
        COHERENCE_SPANS, COHERENCE_SPAN_LENGTH_M
    )
    assert np.allclose(factor_spm, 100.0, rtol=1e-4)
    assert np.allclose(factor_xpm, 100.0, rtol=1e-6)
    print(
        f"zero dispersion: R_spm({COHERENCE_SPANS}) = {factor_spm[0]:.6f}, "
        f"buildup = {factor_spm[0] / COHERENCE_SPANS:.6f}"
    )


def test_strong_dispersion_stays_below_full_coherence() -> None:
    """A strong dispersion must not reach the fully coherent limit."""
    case = build_case("c_57ch")
    fiber = build_path(case).fiber_configs[0]
    strong = fiber.model_copy(update={"dispersion_parameter_d": 100.0, "dispersion_slope_s": 0.0})
    model = make_model("pcfm", case)
    kernel = model.kernel_for(strong)
    factor_spm, factor_xpm = kernel.inter_span_coherent_factor(
        COHERENCE_SPANS, COHERENCE_SPAN_LENGTH_M
    )
    # The coherent factor is above N (the GN kernel keeps the phase-matched
    # pairs coherent) but far below N**2.
    assert np.all(factor_spm > 10.0)
    assert np.all(factor_spm < 20.0)
    # The XPM part is essentially incoherent at any dispersion.
    assert np.all(np.abs(factor_xpm - 10.0) < 0.2)


def test_visibility_matches_the_numerical_reference() -> None:
    """The visibility must be the one of the numerical GN model, bit for bit.

    The caliber of the work package is the one of the numerical model: the
    visibility is weighted by the NLI noise power.  The two implementations
    are independent (no import between the models), so the equality is a
    genuine cross-check of the algorithm.
    """
    if not _NUMERICAL_AVAILABLE:
        pytest.skip("the numerical GN reference is not available")
    from network_nlo_eval.models.numerical import (
        _self_beating_visibility as numerical_self_beating_visibility,
    )

    case = build_case("c_57ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    distances = np.arange(1, COHERENCE_SPANS, dtype=np.float64) * COHERENCE_SPAN_LENGTH_M
    mine = _self_beating_visibility(
        kernel.beta2_abs_s2_m,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        COHERENCE_SPAN_LENGTH_M,
        distances,
    )
    reference = numerical_self_beating_visibility(
        kernel.beta2_abs_s2_m,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        COHERENCE_SPAN_LENGTH_M,
        distances,
    )
    assert np.array_equal(mine, reference)
    assert float(mine[0, 0]) == pytest.approx(0.113769, rel=1e-4)


def test_visibility_matches_an_independent_quadrature() -> None:
    """The reduced quadrature must agree with a direct two-dimensional one."""
    case = build_case("c_57ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    distances = np.array([COHERENCE_SPAN_LENGTH_M], dtype=np.float64)
    mine = _self_beating_visibility(
        kernel.beta2_abs_s2_m,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        COHERENCE_SPAN_LENGTH_M,
        distances,
    )[0, 0]
    reference = _independent_visibility(
        kernel.beta2_abs_s2_m,
        kernel.alpha_visibility_npm,
        float(kernel.b_hz[0]),
        COHERENCE_SPAN_LENGTH_M,
        distances,
    )[0]
    assert float(mine) == pytest.approx(float(reference), rel=1e-3)


def test_visibility_is_converged_in_the_quadrature() -> None:
    """Doubling the quadrature points must not move the visibility much."""
    case = build_case("c_57ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    distances = np.arange(1, COHERENCE_SPANS, dtype=np.float64) * COHERENCE_SPAN_LENGTH_M
    coarse = _self_beating_visibility(
        kernel.beta2_abs_s2_m,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        COHERENCE_SPAN_LENGTH_M,
        distances,
        n_frequency=21,
        n_quadrature=100,
    )
    fine = _self_beating_visibility(
        kernel.beta2_abs_s2_m,
        kernel.alpha_visibility_npm,
        kernel.b_hz,
        COHERENCE_SPAN_LENGTH_M,
        distances,
        n_frequency=41,
        n_quadrature=200,
    )
    assert np.max(np.abs(fine - coarse)) < 5.0e-4


def test_phase_comes_only_from_the_span_context() -> None:
    """The phase of the amplitudes must come from the span context only.

    The model keeps no cross-span counter of its own: the same context gives
    the same amplitude bit for bit, and a context with another span index but
    the same accumulated length gives the same amplitude, because the phase
    follows the accumulated length.
    """
    case = build_case("scaling_40ch")
    fiber = build_path(case).fiber_configs[0]
    power = launch_profile(case, None)
    model = make_model("pcfm", case)
    edfa = EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)
    context = SpanContext(
        span_index=3, total_spans=10, cumulative_length_m=300e3, span_length_m=100e3
    )
    first = model.span_step(fiber, edfa, power, 100e3, span_context=context)
    second = model.span_step(fiber, edfa, power, 100e3, span_context=context)
    assert np.array_equal(first.nli_amplitude_spm, second.nli_amplitude_spm)
    assert np.array_equal(first.nli_amplitude_xpm, second.nli_amplitude_xpm)

    # Same accumulated length, different span index: the phase is unchanged.
    other = model.span_step(
        fiber,
        edfa,
        power,
        100e3,
        span_context=SpanContext(
            span_index=7, total_spans=10, cumulative_length_m=300e3, span_length_m=100e3
        ),
    )
    assert np.array_equal(first.nli_amplitude_spm, other.nli_amplitude_spm)


def test_result_is_deterministic() -> None:
    """The same path must give bit-identical results."""
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = make_model("pcfm", case)
    first = model.evaluate_path(_coherence_path(case, 10), power)
    second = model.evaluate_path(_coherence_path(case, 10), power)
    for name in ("spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(getattr(first, name), getattr(second, name)), name


def test_route_accumulation_matches_the_analytic_coherent_factor() -> None:
    """The route layer must reproduce the analytic coherent factor.

    The route layer sums the complex amplitudes span by span, so the NLI of
    the link must be the single-span NLI times the coherent factor
    ``R(N) = N + 2 sum_k (N-k) V(k L)``.
    """
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = make_model("pcfm", case)
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    single = model.evaluate_path(_coherence_path(case, 1), power)
    many = model.evaluate_path(_coherence_path(case, 10), power)
    factor_spm, factor_xpm = kernel.inter_span_coherent_factor(
        10, COHERENCE_SPAN_LENGTH_M
    )
    assert np.allclose(
        np.asarray(many.spm_noise_w),
        np.asarray(single.spm_noise_w) * factor_spm,
        rtol=1e-9,
        atol=0.0,
    )
    assert np.allclose(
        np.asarray(many.xpm_noise_w),
        np.asarray(single.xpm_noise_w) * factor_xpm,
        rtol=1e-9,
        atol=0.0,
    )


def test_xpm_inter_span_coherence_is_negligible() -> None:
    """The XPM inter-span coherence is negligible, in both calibers.

    Audit (``c_57ch``, 75 GHz spacing, 100 km SMF, mid channel):

    * band average (the convention kept): ``V_xpm(1 span) = -0.0046`` and a
      coherent factor of ``R_xpm(10) = 9.943`` (buildup 0.994);
    * NLI-weighted: ``V_xpm(1 span) = 3.08e-4`` and ``R_xpm(10) = 10.008``
      (buildup 1.001).

    The weight moves the XPM noise by 0.028 dB, i.e. below the 0.05 dB
    threshold of the work package, so the band average is kept.
    """
    case = build_case("c_57ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    _, factor_xpm = kernel.inter_span_coherent_factor(10, COHERENCE_SPAN_LENGTH_M)
    assert np.all(factor_xpm < 10.05)
    assert np.all(factor_xpm > 9.90)

    # The NLI-weighted caliber moves the XPM noise by less than 0.05 dB.
    bandwidth = float(kernel.b_hz[0])
    offset = float(kernel.partner_offset_hz[0])
    distances = np.arange(1, 10, dtype=np.float64) * COHERENCE_SPAN_LENGTH_M
    weighted = np.array(
        [
            _gn_weighted_xpm_visibility(
                kernel.beta2_abs_s2_m,
                kernel.alpha_visibility_npm,
                bandwidth,
                COHERENCE_SPAN_LENGTH_M,
                distance,
                offset,
            )
            for distance in distances
        ]
    )
    argument = (
        2.0 * np.pi**2 * kernel.beta2_abs_s2_m * distances * offset * bandwidth
    )
    band_average = np.sin(argument) / argument
    factor_weighted = _coherent_factor(10, weighted[:, None])[0]
    factor_band = _coherent_factor(10, band_average[:, None])[0]
    effect_db = 10.0 * np.log10(factor_weighted / factor_band)
    assert abs(effect_db) < 0.05, f"the XPM weight moves the noise by {effect_db:.4f} dB"
    print(
        f"XPM weight effect: {effect_db:+.4f} dB "
        f"(R band {factor_band:.5f}, R weighted {factor_weighted:.5f})"
    )


def test_equivalent_phase_ramp_inverts_the_coherent_factor() -> None:
    """The phase ramp must reproduce the coherent factor it was built from."""
    case = build_case("c_57ch")
    kernel = make_model("pcfm", case).kernel_for(build_path(case).fiber_configs[0])
    factor_spm, _ = kernel.inter_span_coherent_factor(10, COHERENCE_SPAN_LENGTH_M)
    ramp = _equivalent_phase_ramp(10, factor_spm)
    rebuilt = (np.sin(0.5 * 10 * ramp) / np.sin(0.5 * ramp)) ** 2
    assert np.allclose(rebuilt, factor_spm, rtol=1e-9)


def test_coherent_accumulation_does_not_regress_the_latency() -> None:
    """The coherence must not slow the hot path down.

    The inter-span quantities are cached per (span count, span length) and
    per (span count, span length, accumulated length), so the hot path only
    pays an ``O(N)`` complex multiplication per span.
    """
    import time

    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    model = make_model("pcfm", case)

    def median_ms(n_spans: int) -> float:
        path = _coherence_path(case, n_spans)
        model.evaluate_path(path, power)
        samples = []
        for _ in range(15):
            started = time.perf_counter()
            model.evaluate_path(path, power)
            samples.append((time.perf_counter() - started) * 1e3)
        return float(np.median(samples))

    single = median_ms(1)
    multi = median_ms(10)
    print(f"PCFM hot path: single span {single:.3f} ms, 10 spans {multi:.3f} ms")
    assert single < 2.34, f"the single-span hot path takes {single:.3f} ms"
    assert multi / 10.0 < 2.0 * single, (
        f"the per-span cost {multi / 10.0:.3f} ms is much higher than the single-span {single:.3f} ms"
    )
