r"""FG-4b gate: the NLI tensorisation must not move a bit, and the freeze point must hold.

Every assertion here is executable and recomputable inside this worktree::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-fast-nb \\
        $HOME/miniconda3/envs/rl/bin/python -m pytest tests/test_fg4b_nli_vectorization.py -v

Mapping to the FG-4b acceptance table
=====================================

======  ==============================================================
 F4b-1  the **default** tier (``matvec``) is within 1e-12 of the golden
 F4b-2  ``off`` is still selectable and still **bit-identical** to
        origin/main — the freeze point has not been quietly moved
 F4b-3  the tensorised NLI alone stays within 1e-12 of the frozen
        per-pair NLI, with the ODE held bit-identical across the arms
 F4b-4  the per-connection bandwidth semantics (``rs_l`` vs ``rs_i``)
        and the zero-power mask both survive vectorisation
 F4b-5  the build-time tensors are provably reused, and provably keyed
        on everything they actually depend on
======  ==============================================================

Why the isolation in F4b-3 matters
----------------------------------
The tempting mistake is to compare the tensor path against the *baseline
commit* and call the difference "the NLI knife". That number is really
"matvec ODE + tensor NLI" versus "per-pair ODE + per-pair NLI" — two changes at
once, so a failure could not be attributed. Here both arms pass the same
``g_gain``/``g_loss``, so the RK4/SRS solve produces **identical bits** and the
measured difference is attributable to the NLI evaluation alone. The test
asserts that bit-identity explicitly rather than assuming it.

A second trap this file is written to close
-------------------------------------------
A 1-D array broadcasts against the **last** axis. In the XPM pre-factor the
inner channel power ``p_l`` may be broadcast implicitly, but the outer channel
power ``p_i`` may not — writing ``* p_in_w`` instead of ``* p_in_w[:, None]``
silently computes ``p_l`` and is **invisible on a fully loaded grid**, where
``p_i == p_l`` for every surviving pair. It only shows up once Raman tilt has
made the per-channel powers differ, i.e. in ``path_cplusl_roadm``. That exact bug
shipped and passed five of the seven scenarios before being caught; the
non-uniform and end-to-end scenarios below are what make it impossible to
reintroduce silently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fg1_scenarios import SCENARIOS as FG1_SCENARIOS
from fg1_scenarios import run_reference as run_reference_off
from fg4a_scenarios import (
    _flexgrid_evaluator,
    _full_load_power,
    _legacy_evaluator,
    _partial_load_power,
)

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models import numba_kernels_p2 as nk
from network_nlo_eval.models.isrs_gn import ODE_ACCEL_DEFAULT, ODE_ACCEL_MODES, MultiBandISRSGN, MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
FG1_GOLDEN_PATH = REPO_ROOT / "results" / "fg1_golden_main.json"

#: The brief's 1e-12 budget. A module constant so a failure is fixed in the
#: implementation, never by loosening the number.
MAX_REL_DEVIATION_ALLOWED = 1e-12

#: The three per-connection bandwidth semantics must stay distinct: ``rs_i`` is
#: the interfering channel's rate (term1's arctan) and ``rs_l`` is the channel
#: of interest's own rate (term2's logarithm and denominator).
HETEROGENEOUS_RS_HZ = (50e9, 100e9 / 6, 100e9)


def _deviation(expected: np.ndarray, actual: np.ndarray) -> tuple[int, int, float, float]:
    """(n_differing, n_finiteness_mismatches, max_abs, max_rel), same rule as FG-4a."""
    n_differing = int(np.count_nonzero(expected != actual))
    n_nonfinite = int(np.count_nonzero(np.isfinite(expected) != np.isfinite(actual)))
    both = np.isfinite(expected) & np.isfinite(actual)
    if not both.any():
        return n_differing, n_nonfinite, 0.0, 0.0
    exp_f, act_f = expected[both], actual[both]
    diff = np.abs(exp_f - act_f)
    denom = np.abs(exp_f)
    rel = np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 0)
    return n_differing, n_nonfinite, float(diff.max()), float(rel.max())


def _span_call(
    evaluator: MultiBandISRSGN,
    power: np.ndarray,
    span_length_m: float,
    n_effective_spans: int = 1,
    edfa: EDFAConfig | None = None,
) -> tuple[Any, ...]:
    """One ``_calc_span_noise_and_power`` call through the stable evaluator API."""
    return evaluator._calc_span_noise_and_power(
        power, span_length_m, edfa or EDFAConfig(target_gain_db=None, noise_figure_db=5.5), None, n_effective_spans
    )


def _tensors_for(evaluator: MultiBandISRSGN, n_span: int, span_length_m: float) -> dict[str, Any]:
    """Build the eight ``nli_*`` kernel kwargs for an evaluator's grid."""
    built = nk.nli_tensors(
        evaluator.n_nodes,
        n_span,
        evaluator.symbol_rate_hz_channels,
        evaluator.f_rel_hz,
        evaluator.lambdas_m,
        evaluator.a_eff_m2_channels,
        evaluator.alpha_power_npm_channels,
        span_length_m,
        evaluator.beta2_s2_m,
        evaluator.beta3_s3_m,
        evaluator.n2_m2_w,
        evaluator.phi_kurtosis,
    )
    names = ("spm0", "spm1", "spm2", "gamma_ii_sq", "gamma_sq", "w0", "w1", "w2")
    return {f"nli_{name}": value for name, value in zip(names, built, strict=True)}


def _kernel_common(evaluator: MultiBandISRSGN, n_span: int) -> dict[str, Any]:
    return dict(
        f_rel_hz=evaluator.f_rel_hz,
        f_abs_hz=evaluator.f_abs_hz,
        lambdas_m=evaluator.lambdas_m,
        alpha_power_npm=evaluator.alpha_power_npm_channels,
        a_eff_m2=evaluator.a_eff_m2_channels,
        beta2_s2_m=evaluator.beta2_s2_m,
        beta3_s3_m=evaluator.beta3_s3_m,
        n2_m2_w=evaluator.n2_m2_w,
        nf_lin_channels=evaluator.nf_lin_channels,
        symbol_rate_hz=evaluator.symbol_rate_hz_channels,
        n_effective_spans=n_span,
        cr_w_m_hz=evaluator.cr_w_m_hz,
        delta_f_co_hz=evaluator.delta_f_co_hz,
        phi_kurtosis=evaluator.phi_kurtosis,
    )


# ======================================================================
# F4b-1  the production default is within budget
# ======================================================================
def test_f4b_1_default_tier_is_matvec() -> None:
    """The adjudication in §1 must actually be in force, not merely documented."""
    assert ODE_ACCEL_DEFAULT == "matvec", (
        f"production default is {ODE_ACCEL_DEFAULT!r}; the FG-4b §1 adjudication set it to 'matvec'"
    )
    assert "off" in ODE_ACCEL_MODES, "the freeze point must remain selectable"


@pytest.mark.parametrize(
    "phi_kurtosis,span_m,n_span,power_factory",
    [
        pytest.param(0.0, 100e3, 1, _full_load_power, id="full_100km"),
        pytest.param(0.0, 100e3, 1, _partial_load_power, id="partial_100km"),
        pytest.param(-0.6190, 80e3, 1, _full_load_power, id="64qam_80km"),
        pytest.param(0.0, 1e3, 1, _full_load_power, id="short_1km_taylor_branch"),
        pytest.param(0.0, 100e3, 10, _partial_load_power, id="nspan10"),
    ],
)
def test_f4b_1_default_tier_within_budget(
    phi_kurtosis: float, span_m: float, n_span: int, power_factory: Any
) -> None:
    """``matvec`` (ODE + tensorised NLI) must stay within 1e-12 of ``off``."""
    power = power_factory()
    off_ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=phi_kurtosis, ode_accel="off"
    )
    default_ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=phi_kurtosis, ode_accel=ODE_ACCEL_DEFAULT
    )
    expected = _span_call(off_ev, power, span_m, n_span)
    actual = _span_call(default_ev, power, span_m, n_span)

    for name, exp, act in zip(("p_out", "spm", "xpm", "ase"), expected, actual, strict=True):
        exp_arr = np.asarray(exp, dtype=np.float64)
        act_arr = np.asarray(act, dtype=np.float64)
        _n_diff, n_nonfinite, max_abs, max_rel = _deviation(exp_arr, act_arr)
        assert n_nonfinite == 0, f"{name}: {n_nonfinite} elements changed between finite and infinite"
        assert max_rel <= MAX_REL_DEVIATION_ALLOWED, (
            f"{name}: default tier deviates {max_rel:.3e} from the frozen reference "
            f"(max abs {max_abs:.3e}), budget is {MAX_REL_DEVIATION_ALLOWED:.0e}"
        )


def test_f4b_1_cached_g_remains_the_bit_exact_tier() -> None:
    """``cached_g`` must stay bit-exact: one tier introduces one kind of reordering.

    This is the guard on the tiering itself. If the tensorised NLI were allowed
    into ``cached_g``, FG-4a's own F4a-3 assertion (cached_g is bit-identical to
    the golden) would silently stop describing reality.
    """
    ev = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="cached_g")
    assert all(value is None for value in ev._nli_kernel_kwargs(1, 100e3).values()), (
        "cached_g must not receive NLI tensors; it is the bit-exact tier"
    )
    matvec_ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="matvec"
    )
    assert all(value is not None for value in matvec_ev._nli_kernel_kwargs(1, 100e3).values()), (
        "matvec must receive the tensorised NLI"
    )


# ======================================================================
# F4b-2  the freeze point has not moved
# ======================================================================
def test_f4b_2_off_is_bit_identical_to_origin_main() -> None:
    """``off`` still reproduces origin/main bit for bit, across every FG-1 scenario."""
    if not FG1_GOLDEN_PATH.exists():  # pragma: no cover
        pytest.fail(f"missing {FG1_GOLDEN_PATH}")
    golden = json.loads(FG1_GOLDEN_PATH.read_text(encoding="utf-8"))
    assert golden["source_branch"] == "origin/main"

    for scenario in FG1_SCENARIOS:
        expected = golden["scenarios"][scenario.name]
        actual = run_reference_off(scenario)  # run_reference is pinned to ode_accel="off"
        for key in ("signal_power_w", "spm_w", "xpm_w", "ase_w", "snr_db"):
            assert np.array_equal(
                np.asarray(expected[key], dtype=float),
                np.asarray(actual[key], dtype=float),
            ), f"{scenario.name}.{key}: the 'off' freeze point moved away from origin/main"


def test_f4b_2_off_tier_passes_no_tensors_to_the_kernel() -> None:
    """``off`` must route to the frozen per-pair NLI, not merely look similar to it.

    Bit-equality with origin/main would also be satisfied by a coincidentally
    close implementation. What makes the freeze point real is that the frozen
    code is still the code being executed.
    """
    ev = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="off")
    kwargs = ev._nli_kernel_kwargs(1, 100e3)
    assert set(kwargs) == {
        "nli_spm0",
        "nli_spm1",
        "nli_spm2",
        "nli_gamma_ii_sq",
        "nli_gamma_sq",
        "nli_w0",
        "nli_w1",
        "nli_w2",
    }
    assert all(value is None for value in kwargs.values()), "the 'off' tier must not be given NLI tensors"


def test_f4b_2_frozen_nli_source_is_intact() -> None:
    """The FG-2 frozen per-pair NLI must still be a per-pair double loop.

    Same intent as FG-4a's frozen ``_calc_dp_dz`` test: the branchy reference is
    the bit-exact oracle the tensor path is judged against, so it must not be
    quietly rewritten in place.
    """
    source = Path(nk.__file__).read_text(encoding="utf-8")
    start = source.index("def _compute_nli_variances(")
    body = source[start : source.index("\n\n\n", start)]
    assert "for l_idx in range(n_ch):" in body, (
        "_compute_nli_variances lost its per-pair loop; the tensor variant must be a "
        "separate function so the frozen reference stays available"
    )
    assert "_calc_xpm_variance_term_jit(" in body, "the frozen per-pair XPM term call was removed"
    assert "def _compute_nli_variances_t(" in source, "the FG-4b tensor variant is missing"


# ======================================================================
# F4b-3  the knife, in isolation, stays within budget
# ======================================================================
@pytest.mark.parametrize(
    "power_factory,span_m,n_span,phi_kurtosis,label",
    [
        pytest.param(_full_load_power, 100e3, 1, 0.0, "full_100km", id="full_100km"),
        pytest.param(_partial_load_power, 100e3, 1, 0.0, "partial_100km", id="partial_100km"),
        pytest.param(_full_load_power, 80e3, 1, -0.6190, "64qam_80km", id="64qam_80km"),
        pytest.param(_full_load_power, 1e3, 1, 0.0, "short_1km", id="short_1km"),
        pytest.param(_partial_load_power, 100e3, 10, 0.0, "nspan10", id="nspan10"),
    ],
)
def test_f4b_3_tensor_nli_matches_frozen_nli_with_budget(
    power_factory: Any, span_m: float, n_span: int, phi_kurtosis: float, label: str
) -> None:
    """Both arms share the matvec ODE, so only the NLI evaluation differs.

    ``ODE_output_bit_identical`` is asserted first: without it this comparison
    would silently also be measuring the ODE reordering, and the attribution
    would be wrong rather than merely imprecise.
    """
    power = power_factory()
    ev = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=phi_kurtosis, ode_accel="matvec"
    )
    g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)
    common = _kernel_common(ev, n_span)

    frozen = nk._calc_span_noise_and_power_jit(power, span_m, g_gain=g_gain, g_loss=g_loss, **common)
    tensor = nk._calc_span_noise_and_power_jit(
        power, span_m, g_gain=g_gain, g_loss=g_loss, **common, **_tensors_for(ev, n_span, span_m)
    )

    assert np.array_equal(np.asarray(frozen[0]), np.asarray(tensor[0]), equal_nan=True), (
        f"{label}: the ODE half must be bit-identical across arms, or this comparison "
        "is not isolating the NLI knife"
    )
    for name, exp, act in zip(("spm", "xpm"), frozen[1:3], tensor[1:3], strict=True):
        _n_diff, n_nonfinite, max_abs, max_rel = _deviation(
            np.asarray(exp, dtype=np.float64), np.asarray(act, dtype=np.float64)
        )
        assert n_nonfinite == 0, f"{label}/{name}: finiteness changed"
        assert max_rel <= MAX_REL_DEVIATION_ALLOWED, (
            f"{label}/{name}: tensorised NLI deviates {max_rel:.3e} "
            f"(max abs {max_abs:.3e}), budget {MAX_REL_DEVIATION_ALLOWED:.0e}"
        )


def test_f4b_3_tensor_nli_survives_non_uniform_per_channel_power() -> None:
    """The regression guard for the ``p_i``/``p_l`` broadcast-axis bug.

    A fully loaded grid has ``p_i == p_l`` for every surviving pair, so
    broadcasting the wrong one is invisible. Raman tilt makes them differ. This
    test therefore drives the *kernel* directly with a deliberately
    non-uniform power profile, which is the condition under which a transposed
    pre-factor shows up — and it is why the end-to-end ``path_cplusl_roadm``
    scenario is part of the golden set.
    """
    ev = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="matvec")
    g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)
    n = ev.n_nodes
    ramp = np.linspace(1e-4, 4e-3, n)  # strictly increasing: no two channels share a power
    ramp[::11] = 0.0  # and some unoccupied, so the zero mask is on the path too
    common = _kernel_common(ev, 1)

    frozen = nk._calc_span_noise_and_power_jit(ramp, 100e3, g_gain=g_gain, g_loss=g_loss, **common)
    tensor = nk._calc_span_noise_and_power_jit(
        ramp, 100e3, g_gain=g_gain, g_loss=g_loss, **common, **_tensors_for(ev, 1, 100e3)
    )
    assert np.array_equal(np.asarray(frozen[0]), np.asarray(tensor[0]), equal_nan=True)
    _nd, nf, max_abs, max_rel = _deviation(
        np.asarray(frozen[2], dtype=np.float64), np.asarray(tensor[2], dtype=np.float64)
    )
    assert nf == 0
    assert max_rel <= MAX_REL_DEVIATION_ALLOWED, (
        f"XPM deviates {max_rel:.3e} under a non-uniform power profile (max abs {max_abs:.3e}); "
        "a p_i/p_l broadcast-axis error shows up exactly here"
    )
    # A transposed pre-factor would be catastrophically wrong, not marginally off.
    assert max_rel < 1e-9, f"deviation {max_rel:.3e} is far too large for a summation-order effect"


# ======================================================================
# F4b-4  the irregular parts of the original survive vectorisation
# ======================================================================
def test_f4b_4_heterogeneous_symbol_rates_are_preserved() -> None:
    """``rs_l`` and ``rs_i`` must stay distinct per-connection quantities.

    The flexgrid scenario mixes three symbol rates, so a regression that
    collapsed the XPM kernel's separate ``rs_l_hz`` / ``rs_i_hz`` semantics into
    one scalar would be caught here.
    """
    ev = _flexgrid_evaluator("matvec")
    rates = np.unique(ev.symbol_rate_hz_channels)
    assert len(rates) > 1, "this scenario is supposed to exercise heterogeneous per-connection rates"
    assert set(np.round(rates, 3)) == {round(rs, 3) for rs in HETEROGENEOUS_RS_HZ}

    power = np.where(ev.symbol_rate_hz_channels > 0, 1e-3, 0.0)
    power[np.arange(ev.n_nodes) % 5 == 4] = 0.0
    g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)
    common = _kernel_common(ev, 1)

    frozen = nk._calc_span_noise_and_power_jit(power, 100e3, g_gain=g_gain, g_loss=g_loss, **common)
    tensor = nk._calc_span_noise_and_power_jit(
        power, 100e3, g_gain=g_gain, g_loss=g_loss, **common, **_tensors_for(ev, 1, 100e3)
    )
    assert np.array_equal(np.asarray(frozen[0]), np.asarray(tensor[0]), equal_nan=True)
    for name, exp, act in zip(("spm", "xpm"), frozen[1:3], tensor[1:3], strict=True):
        _nd, nf, _ma, max_rel = _deviation(np.asarray(exp, dtype=np.float64), np.asarray(act, dtype=np.float64))
        assert nf == 0, f"{name}: finiteness changed on the heterogeneous-rate grid"
        assert max_rel <= MAX_REL_DEVIATION_ALLOWED, f"{name}: {max_rel:.3e} on heterogeneous rates"


def test_f4b_4_zero_power_channels_are_excluded() -> None:
    """A channel with no input power contributes no NLI, and says so loudly."""
    ev = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="matvec")
    power = _full_load_power()
    silent = power.copy()
    silent[[3, 40, 199]] = 0.0
    tensors = _tensors_for(ev, 1, 100e3)
    g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)
    common = _kernel_common(ev, 1)

    _, spm, xpm, _ = nk._calc_span_noise_and_power_jit(
        silent, 100e3, g_gain=g_gain, g_loss=g_loss, **common, **tensors
    )
    for index in (3, 40, 199):
        assert spm[index] == 0.0, f"channel {index} has no input power but reported SPM"
        assert xpm[index] == 0.0, f"channel {index} has no input power but reported XPM"
    assert np.all(np.isfinite(np.asarray(xpm))), "the zero mask failed to keep nan/inf out of the XPM sum"
    # And an unoccupied channel must not perturb its neighbours either.
    loaded = nk._calc_span_noise_and_power_jit(
        power, 100e3, g_gain=g_gain, g_loss=g_loss, **common, **tensors
    )
    for index in (5, 41, 200):
        assert loaded[2][index] > 0.0


def test_f4b_4_t_tilde_fit_and_kurtosis_branches_are_covered() -> None:
    """The two branches the brief flags as the source of irregularity.

    ``t_tilde`` is the only data-dependent input, so it must actually change the
    answer (otherwise the tensor path could be ignoring it); and the
    ``phi_kurtosis`` split (0.0 Gaussian / -0.619 64QAM) must still be honoured.
    """
    ev_gauss = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="matvec")
    ev_64qam = _legacy_evaluator(
        length_km=100.0, max_span_length_km=100.0, phi_kurtosis=-0.6190, ode_accel="matvec"
    )
    power = _full_load_power()
    g_gain, g_loss = nk.raman_g_pair_matrices(ev_gauss.f_abs_hz, ev_gauss.cr_w_m_hz)

    def xpm(ev: MultiBandISRSGN, tensorise: bool) -> np.ndarray:
        extra = _tensors_for(ev, 1, 100e3) if tensorise else {}
        return np.asarray(
            nk._calc_span_noise_and_power_jit(
                power, 100e3, g_gain=g_gain, g_loss=g_loss, **_kernel_common(ev, 1), **extra
            )[2],
            dtype=np.float64,
        )

    # phi_kurtosis must still matter, on the tensor path as much as the frozen one.
    g_frozen, q_frozen = xpm(ev_gauss, False), xpm(ev_64qam, False)
    g_tensor, q_tensor = xpm(ev_gauss, True), xpm(ev_64qam, True)
    assert not np.array_equal(g_frozen, q_frozen), "phi_kurtosis has no effect on the frozen path"
    assert not np.array_equal(g_tensor, q_tensor), "the tensor path dropped the phi_kurtosis branch"
    for label, expected, actual in (("gaussian", g_frozen, g_tensor), ("64qam", q_frozen, q_tensor)):
        _nd, nf, _ma, max_rel = _deviation(expected, actual)
        assert nf == 0, f"{label}: finiteness changed"
        assert max_rel <= MAX_REL_DEVIATION_ALLOWED, f"{label}: {max_rel:.3e}"

    # t_tilde must actually reach the answer: drive the ODE output by using two
    # different span lengths, which changes the Raman tilt that t_tilde fits.
    short = np.asarray(
        nk._calc_span_noise_and_power_jit(
            power, 1e3, g_gain=g_gain, g_loss=g_loss, **_kernel_common(ev_gauss, 1), **_tensors_for(ev_gauss, 1, 1e3)
        )[2],
        dtype=np.float64,
    )
    long = xpm(ev_gauss, True)
    assert not np.allclose(short, long, rtol=1e-6), (
        "XPM is insensitive to span length, which would mean t_tilde is not reaching the tensor path"
    )


# ======================================================================
# F4b-5  the build-time tensors are cached, and keyed correctly
# ======================================================================
def test_f4b_5_tensors_are_reused_across_calls() -> None:
    """The (N,N) tensors must be built once and reused, not rebuilt per span."""
    ev = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="matvec")
    kwargs = ev._nli_kernel_kwargs(1, 100e3)
    first = kwargs["nli_w0"]
    for _ in range(5):
        again = ev._nli_kernel_kwargs(1, 100e3)["nli_w0"]
        assert again is first, "the NLI tensors were rebuilt instead of reused across calls"
    assert nk.nli_cache_stats()["hits"] > 0, "the tensor cache never registered a hit"


def test_f4b_5_tensor_cache_is_keyed_on_what_it_depends_on() -> None:
    """Changing ``n_span`` or the span length must not return a stale tensor.

    ``n_span`` is folded into the XPM weights (``n_span + (5/6)*phi_kurtosis``
    and ``n_tilde``) and the span length into ``a_tilde``/``kappa``, so a cache
    that omitted either would silently return another regime's numbers.
    """
    ev = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="matvec")
    g_gain, g_loss = nk.raman_g_pair_matrices(ev.f_abs_hz, ev.cr_w_m_hz)
    power = _full_load_power()
    common = _kernel_common(ev, 1)

    def xpm(n_span: int, span_m: float) -> np.ndarray:
        extra = _tensors_for(ev, n_span, span_m)
        return np.asarray(
            nk._calc_span_noise_and_power_jit(
                power, span_m, g_gain=g_gain, g_loss=g_loss, **{**common, "n_effective_spans": n_span}, **extra
            )[2],
            dtype=np.float64,
        )

    # Same key -> same object; different key -> different object.
    assert _tensors_for(ev, 1, 100e3)["nli_w0"] is _tensors_for(ev, 1, 100e3)["nli_w0"]
    assert _tensors_for(ev, 1, 100e3)["nli_w0"] is not _tensors_for(ev, 2, 100e3)["nli_w0"]
    assert _tensors_for(ev, 1, 100e3)["nli_w0"] is not _tensors_for(ev, 1, 80e3)["nli_w0"]

    # ...and each must still match the frozen NLI in its own regime.
    for n_span, span_m in ((1, 100e3), (2, 100e3), (5, 80e3)):
        frozen = np.asarray(
            nk._calc_span_noise_and_power_jit(
                power, span_m, g_gain=g_gain, g_loss=g_loss, **{**common, "n_effective_spans": n_span}
            )[2],
            dtype=np.float64,
        )
        _nd, nf, _ma, max_rel = _deviation(frozen, xpm(n_span, span_m))
        assert nf == 0, f"n_span={n_span}, L={span_m}: finiteness changed"
        assert max_rel <= MAX_REL_DEVIATION_ALLOWED, (
            f"n_span={n_span}, L={span_m}: {max_rel:.3e} — a stale cache entry is the likely cause"
        )


def test_f4b_5_tensors_are_read_only() -> None:
    """Cached tensors are shared; in-place mutation would corrupt other callers."""
    ev = _legacy_evaluator(length_km=100.0, max_span_length_km=100.0, phi_kurtosis=0.0, ode_accel="matvec")
    tensors = _tensors_for(ev, 1, 100e3)
    for name, value in tensors.items():
        assert not value.flags.writeable, f"{name} is cached and shared but left writeable"


# ======================================================================
# F4b-6  the end-to-end path scenario (the one that caught the axis bug)
# ======================================================================
def test_f4b_6_end_to_end_path_with_roadm_and_tilt() -> None:
    """``evaluate_path`` end-to-end, across hops, span splits and a ROADM.

    Raman tilt plus insertion loss make the per-channel powers differ, which is
    exactly the condition a ``p_i``/``p_l`` broadcast error needs. The frozen
    path is used as the oracle, so the comparison is independent of the golden.
    """
    from fg4a_scenarios import CENTER_FREQUENCY_HZ, CHANNEL_SPACING_HZ, NUM_CHANNELS

    from network_nlo_eval.core.spectrum import SpectrumGrid as _SpectrumGrid  # noqa: F401

    def run(ode_accel: str) -> dict[str, Any]:
        grid = SpectrumGrid(
            num_channels=NUM_CHANNELS,
            channel_spacing_hz=CHANNEL_SPACING_HZ,
            center_frequency_hz=CENTER_FREQUENCY_HZ,
        )
        ev = MultiBandISRSGN(
            grid=grid,
            ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
            symbol_rate_hz=50e9,
            phi_kurtosis=0.0,
            ode_accel=ode_accel,
            # FG-15: pin the tier to ``legacy``. ``ode_accel="off"`` is the frozen
            # legacy-only reference path, and this test compares that reference
            # against the accelerated path on the same spectrum.
            raman_spectrum="legacy",
            nli_model="p2_gn",
        )
        optical_path = MultiSpanOpticalPath(
            fiber_configs=[
                FiberSpanConfig(length_km=160.0, max_span_length_km=80.0),
                FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
            ],
            edfa_configs=[
                EDFAConfig(target_gain_db=None, noise_figure_db=5.5),
                EDFAConfig(target_gain_db=None, noise_figure_db=6.0),
            ],
            roadm_configs=[
                ROADMConfig(insertion_loss_db=5.0, filtering_penalty_db=0.2, blocked_slots=[40, 41, 42, 43]),
                None,
            ],
            link_keys=[(0, 1), (1, 2)],
            node_path=[0, 1, 2],
        )
        return ev.evaluate_path(optical_path, _partial_load_power())

    expected = run("off")
    actual = run(ODE_ACCEL_DEFAULT)
    for name, exp, act in (
        ("signal_power_w", expected.signal_power_w, actual.signal_power_w),
        ("spm_noise_w", expected.spm_noise_w, actual.spm_noise_w),
        ("xpm_noise_w", expected.xpm_noise_w, actual.xpm_noise_w),
        ("ase_noise_w", expected.ase_noise_w, actual.ase_noise_w),
        ("snr_db", expected.snr_db, actual.snr_db),
    ):
        exp_arr = np.asarray(exp, dtype=np.float64)
        act_arr = np.asarray(act, dtype=np.float64)
        _nd, nf, max_abs, max_rel = _deviation(exp_arr, act_arr)
        assert nf == 0, f"{name}: {nf} elements changed between finite and infinite"
        assert max_rel <= MAX_REL_DEVIATION_ALLOWED, (
            f"path/{name}: deviates {max_rel:.3e} (max abs {max_abs:.3e}), "
            f"budget {MAX_REL_DEVIATION_ALLOWED:.0e}"
        )
    assert int(actual.span_count) == int(expected.span_count)
    assert int(actual.roadm_count) == int(expected.roadm_count)
