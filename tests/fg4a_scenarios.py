"""FG-4a fixed scenarios: the input side of the numeric-identity gate.

Why this file is written the way it is
--------------------------------------
The FG-4a acceleration work (G-matrix cache, branchless ``_calc_dp_dz``) is
allowed to change the **kernel's** internals and even its signature, but it must
not change a single bit of the numbers the model produces. The gate for that is
:mod:`tests.test_fg4a_numeric_identity`, which compares against
``results/fg4a_golden.json``.

That golden file has to stay replayable *across* those kernel changes, so the
scenarios here are defined **only against the stable evaluator API**
(``MultiBandISRSGN._calc_span_noise_and_power`` and ``MultiBandISRSGN.evaluate_path``)
— never against ``_calc_span_noise_and_power_jit``'s argument list. If the
golden were captured at the kernel level, every legitimate kernel refactor would
force a golden regeneration, which would destroy the gate's whole purpose.

This module therefore *also* runs unchanged on the pre-FG-4a baseline commit,
so the golden can be captured before any acceleration is written.

Scenario coverage is chosen to pin the branches that the two optimisations
touch, plus the branches they must **not** disturb:

======================  ====================================================
scenario                branch it pins
======================  ====================================================
``cl_full_100km``       the profiled 230x50 GHz C+L case, fully loaded
``cl_partial_100km``    unoccupied slots (``power_in_w == 0``) interleaved
``cl_full_64qam_80km``  non-Gaussian ``phi_kurtosis`` + a different span length
``cl_short_1km``        the ``den < 1e-12`` Taylor branch of ``a_tilde``
``cl_nspan10``          multi-span NLI accumulation (``n_span > 1``)
``flexgrid_mixed``      non-uniform centres + per-connection symbol rates
``path_cplusl_roadm``   ``evaluate_path`` end-to-end incl. ROADM and ``snr_db``
======================  ====================================================

All inputs are literals or deterministic closed-form expressions: no RNG, no
clock, no environment lookups, no filesystem state. Re-running this module on any
machine must produce bit-identical inputs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import (
    ODE_ACCEL_DEFAULT,
    ODE_ACCEL_MODES,
    MultiBandISRSGN,
    MultiSpanOpticalPath,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

# --- Frozen C+L fixed-grid constants (identical to the FG-2 golden) --------
NUM_CHANNELS = 230
CHANNEL_SPACING_HZ = 50e9
CENTER_FREQUENCY_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9
SLOT_WIDTH_HZ = 12.5e9
SLOTS_PER_CHANNEL = 4
NUM_SLOTS = NUM_CHANNELS * SLOTS_PER_CHANNEL  # 920

# The two kurtosis values the evaluator documents: 0.0 Gaussian, -0.6190 64QAM.
PHI_KURTOSIS_GAUSSIAN = 0.0
PHI_KURTOSIS_64QAM = -0.6190

# nf applied at the call site; the evaluator default is 5.0 dB.
EDFA_NF_DB = 5.0

#: Keys that every single-span scenario contributes, in canonical order.
SPAN_KEYS = ("p_out_w", "sigma2_spm_w", "sigma2_xpm_w", "sigma2_ase_w")

#: Keys that the end-to-end path scenario contributes, in canonical order.
PATH_KEYS = (
    "signal_power_w",
    "spm_noise_w",
    "xpm_noise_w",
    "ase_noise_w",
    "snr_db",
    "span_count",
    "roadm_count",
)


def _to_list(value: Any) -> Any:
    """Convert a scenario output into something ``json`` stores bit-exactly.

    ``json`` serialises Python floats with ``float.__repr__`` (the shortest
    string that round-trips), so a ``float`` survives ``json.load`` unchanged.
    Converting the arrays to nested lists keeps that guarantee: the golden is
    full ``repr`` precision, not a rounded printout.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


@dataclass(frozen=True)
class Scenario:
    """One frozen input set and the outputs it must keep producing bit-exactly.

    Attributes
    ----------
        name: Stable identifier; becomes the key in the golden JSON.
        kind: ``"span"`` or ``"path"``; decides which output keys are expected.
        description: Human-readable purpose, copied into the golden for audit.
        run: Callable returning a ``{key: value}`` dict of results.
    """

    name: str
    kind: str
    description: str
    run: Callable[[], dict[str, Any]]

    def outputs(self) -> dict[str, Any]:
        """Run the scenario and json-ready its outputs."""
        return {key: _to_list(value) for key, value in self.run().items()}


# ======================================================================
# Single-span scenarios (driven through the stable evaluator API)
# ======================================================================
def _legacy_evaluator(
    *,
    length_km: float,
    max_span_length_km: float,
    phi_kurtosis: float,
    ode_accel: str,
) -> MultiBandISRSGN:
    """Build the legacy 230x50 GHz fixed-grid evaluator."""
    grid = SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )
    fiber = FiberSpanConfig(length_km=length_km, max_span_length_km=max_span_length_km)
    return MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=fiber,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        phi_kurtosis=phi_kurtosis,
        ode_accel=ode_accel,
    )


def _span_scenario(
    name: str,
    description: str,
    *,
    length_km: float,
    phi_kurtosis: float,
    span_length_m: float,
    power_in_w: np.ndarray,
    n_effective_spans: int = 1,
    nf_db: float = EDFA_NF_DB,
    ode_accel: str,
) -> Scenario:
    """Wrap one ``_calc_span_noise_and_power`` call as a scenario."""
    edfa = EDFAConfig(target_gain_db=None, noise_figure_db=nf_db)

    def run() -> dict[str, Any]:
        ev = _legacy_evaluator(
            length_km=length_km,
            max_span_length_km=100.0,
            phi_kurtosis=phi_kurtosis,
            ode_accel=ode_accel,
        )
        p_out, s2_spm, s2_xpm, s2_ase = ev._calc_span_noise_and_power(
            power_in_w, span_length_m, edfa, None, n_effective_spans
        )
        return {
            "p_out_w": p_out,
            "sigma2_spm_w": s2_spm,
            "sigma2_xpm_w": s2_xpm,
            "sigma2_ase_w": s2_ase,
        }

    return Scenario(name=name, kind="span", description=description, run=run)


def _full_load_power() -> np.ndarray:
    """Uniform 1 mW launch on every channel (~0 dBm)."""
    return np.full(NUM_CHANNELS, 1e-3, dtype=np.float64)


def _partial_load_power() -> np.ndarray:
    """Interleaved empty slots: every 7th channel carries no power.

    The 0-power slots are what pin the ``if p_in_w[i] <= 0: continue`` guards in
    both ``_compute_nli_variances`` and ``_calculate_ase_noise_variance``; they
    must survive the G-matrix cache and the branchless ``_calc_dp_dz`` unchanged.
    """
    power = _full_load_power()
    power[::7] = 0.0
    # Also blank both band edges, so the trimming guard has edge material.
    power[0] = 0.0
    power[-1] = 0.0
    return power


def _flexgrid_evaluator(ode_accel: str) -> MultiBandISRSGN:
    """Build a non-uniform flexible-grid evaluator with per-connection rates.

    Deliberately mixes connection widths and symbol rates so the node
    frequencies are not uniformly spaced: an evenly spaced grid would let a
    frequency-sorted code path pass while the real workload breaks.
    """
    grid = FlexibleGrid(slot_width_hz=SLOT_WIDTH_HZ, num_slots=NUM_SLOTS, band_low_edge_hz=186.1e12)
    connections: list[ConnectionSpec] = []
    connection_id = 0
    first_slot = 0
    # Three widths cycling: 4-slot QPSK @ 100 Gbps, 2-slot 64QAM @ 100 Gbps,
    # 8-slot QPSK @ 200 Gbps. Leaves 2 idle slots between the 8-slot ones.
    plan = ((4, 100e9, "QPSK", 50e9), (2, 100e9, "64QAM", 100e9 / 6), (8, 200e9, "QPSK", 100e9))
    index = 0
    while first_slot + 2 <= NUM_SLOTS:
        n_slots, bit_rate, modulation, rs = plan[index % len(plan)]
        if first_slot + n_slots > NUM_SLOTS:
            break
        connections.append(
            ConnectionSpec(
                grid=grid,
                connection_id=connection_id,
                first_slot=first_slot,
                n_slots=n_slots,
                bit_rate_bps=bit_rate,
                modulation=modulation,
                symbol_rate_hz=rs,
                roll_off=0.0,
                # Every 5th connection is unoccupied but keeps its array slot.
                launch_power_w=0.0 if connection_id % 5 == 4 else 1e-3,
            )
        )
        connection_id += 1
        first_slot += n_slots + (2 if n_slots == 8 else 1)
        index += 1

    fiber = FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)
    return MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=fiber,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        phi_kurtosis=PHI_KURTOSIS_GAUSSIAN,
        connections=connections,
        slots_per_channel=SLOTS_PER_CHANNEL,
        ode_accel=ode_accel,
    )


# ======================================================================
# Scenario table
# ======================================================================
def build_scenarios(ode_accel: str = ODE_ACCEL_DEFAULT) -> list[Scenario]:
    """Return every FG-4a scenario, in a stable order.

    Args:
        ode_accel: Which FG-4a acceleration path the evaluators should use.
            ``"off"`` and ``"cached_g"`` must reproduce the golden bit-for-bit;
            ``"matvec"`` is allowed a bounded float-reordering shift.
    """
    if ode_accel not in ODE_ACCEL_MODES:
        raise ValueError(f"ode_accel must be one of {ODE_ACCEL_MODES}, got {ode_accel!r}.")

    def flex_run() -> dict[str, Any]:
        ev = _flexgrid_evaluator(ode_accel)
        flex_power = np.array([conn.launch_power_w for conn in ev.connections], dtype=np.float64)
        edfa = EDFAConfig(target_gain_db=None, noise_figure_db=EDFA_NF_DB)
        p_out, s2_spm, s2_xpm, s2_ase = ev._calc_span_noise_and_power(flex_power, 100e3, edfa, None, 1)
        return {
            "p_out_w": p_out,
            "sigma2_spm_w": s2_spm,
            "sigma2_xpm_w": s2_xpm,
            "sigma2_ase_w": s2_ase,
        }

    def path_run() -> dict[str, Any]:
        ev = _legacy_evaluator(
            length_km=100.0,
            max_span_length_km=80.0,
            phi_kurtosis=PHI_KURTOSIS_GAUSSIAN,
            ode_accel=ode_accel,
        )
        # 3 hops: two 80 km spans (one per fibre) plus a ROADM between them.
        # max_span_length_km=80 means the 160 km fibre is split into 2 spans,
        # exercising the multi-span accumulation branch of evaluate_path.
        optical_path = MultiSpanOpticalPath(
            fiber_configs=[
                FiberSpanConfig(length_km=160.0, max_span_length_km=80.0),
                FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
            ],
            edfa_configs=[
                EDFAConfig(target_gain_db=None, noise_figure_db=EDFA_NF_DB),
                EDFAConfig(target_gain_db=None, noise_figure_db=6.0),
            ],
            roadm_configs=[
                ROADMConfig(
                    insertion_loss_db=5.0,
                    filtering_penalty_db=0.2,
                    blocked_slots=[40, 41, 42, 43],
                ),
                None,
            ],
            link_keys=[(0, 1), (1, 2)],
            node_path=[0, 1, 2],
        )
        result = ev.evaluate_path(optical_path, _partial_load_power())
        return {
            "signal_power_w": result.signal_power_w,
            "spm_noise_w": result.spm_noise_w,
            "xpm_noise_w": result.xpm_noise_w,
            "ase_noise_w": result.ase_noise_w,
            "snr_db": result.snr_db,
            "span_count": result.span_count,
            "roadm_count": result.roadm_count,
        }

    return [
        _span_scenario(
            "cl_full_100km",
            "The profiled case: 230x50 GHz C+L, every channel loaded, 100 km, Gaussian.",
            length_km=100.0,
            phi_kurtosis=PHI_KURTOSIS_GAUSSIAN,
            span_length_m=100e3,
            power_in_w=_full_load_power(),
            ode_accel=ode_accel,
        ),
        _span_scenario(
            "cl_partial_100km",
            "Interleaved unoccupied slots plus blanked band edges (0-power guards).",
            length_km=100.0,
            phi_kurtosis=PHI_KURTOSIS_GAUSSIAN,
            span_length_m=100e3,
            power_in_w=_partial_load_power(),
            ode_accel=ode_accel,
        ),
        _span_scenario(
            "cl_full_64qam_80km",
            "64QAM kurtosis (-0.619) over an 80 km span; 6 dB EDFA.",
            length_km=100.0,
            phi_kurtosis=PHI_KURTOSIS_64QAM,
            span_length_m=80e3,
            power_in_w=_full_load_power(),
            nf_db=6.0,
            ode_accel=ode_accel,
        ),
        _span_scenario(
            "cl_short_1km",
            "1 km span: pins the a_tilde/den<1e-12 Taylor branch and a different R_k.",
            length_km=100.0,
            phi_kurtosis=PHI_KURTOSIS_GAUSSIAN,
            span_length_m=1e3,
            power_in_w=_full_load_power(),
            ode_accel=ode_accel,
        ),
        _span_scenario(
            "cl_nspan10",
            "n_effective_spans=10: multi-span NLI accumulation (n_span>1, n_tilde=n_span).",
            length_km=100.0,
            phi_kurtosis=PHI_KURTOSIS_GAUSSIAN,
            span_length_m=100e3,
            power_in_w=_partial_load_power(),
            n_effective_spans=10,
            ode_accel=ode_accel,
        ),
        Scenario(
            name="flexgrid_mixed",
            kind="span",
            description="Non-uniform flexible grid: mixed 2/4/8-slot connections, per-connection Rs.",
            run=flex_run,
        ),
        Scenario(
            name="path_cplusl_roadm",
            kind="path",
            description=("evaluate_path: 160 km fibre split into 2 spans, ROADM with blocked slots, then 100 km."),
            run=path_run,
        ),
    ]


def scenario_by_name(name: str, ode_accel: str = ODE_ACCEL_DEFAULT) -> Scenario:
    """Look one scenario up by name."""
    for scenario in build_scenarios(ode_accel):
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown FG-4a scenario: {name!r}")
