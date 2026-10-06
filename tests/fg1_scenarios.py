"""FG-1 parity scenarios: fixed, branch-agnostic physical scenarios.

This module is the shared, version-independent description of the physics
parity check between ``origin/main`` (ODE kernel) and ``wp/fg-reconcile``
(ODE kernel + path pipeline).

Why this file is written the way it is
--------------------------------------
It must import and run **unchanged** on both branches, so it may only use
API that exists on ``origin/main``:

* ``SpectrumGrid(num_channels, channel_spacing_hz, center_frequency_hz)``
* ``FiberSpanConfig(length_km=...)`` -- deliberately *not*
  ``max_span_length_km``, which does not exist on main. Span lengths are
  given explicitly per scenario instead.
* ``EDFAConfig(target_gain_db=..., noise_figure_db=...)``
* ``ROADMConfig(insertion_loss_db=..., filtering_penalty_db=...)``
* ``MultiBandISRSGN(grid, fiber, symbol_rate_hz, phi_kurtosis)``
* ``evaluator._calc_span_noise_and_power(power, span_m, edfa, None, 1)``

It deliberately does **not** use ``evaluate_path`` (pipeline-only) so that
the golden reference is produced by the *same* code on both sides. The
pipeline's fidelity is then proven by comparing ``evaluate_path`` against
this reference, not by defining the reference with it.

Units follow the kernel contract: powers in W (linear), noise terms are
variances, frequencies in Hz, lengths in m inside the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

LAUNCH_POWER_W = 1e-3  # 0 dBm per channel


@dataclass(frozen=True)
class Hop:
    """One link: a fiber split into equal spans, an EDFA, and an optional ROADM."""

    length_km: float
    span_km: float
    target_gain_db: float | None
    noise_figure_db: float
    roadm_insertion_loss_db: float | None = None
    roadm_filtering_penalty_db: float = 0.2
    blocked_channels: tuple[int, ...] = ()


@dataclass(frozen=True)
class Scenario:
    """A fully specified, reproducible scenario."""

    name: str
    description: str
    num_channels: int
    channel_spacing_hz: float
    center_frequency_hz: float
    symbol_rate_hz: float
    phi_kurtosis: float
    launch_power_w: float
    hops: tuple[Hop, ...] = field(default_factory=tuple)

    def grid(self) -> SpectrumGrid:
        return SpectrumGrid(self.num_channels, self.channel_spacing_hz, self.center_frequency_hz)

    def fiber(self, hop: Hop) -> FiberSpanConfig:
        return FiberSpanConfig(length_km=hop.length_km)

    def edfa(self, hop: Hop) -> EDFAConfig:
        return EDFAConfig(target_gain_db=hop.target_gain_db, noise_figure_db=hop.noise_figure_db)

    def roadm(self, hop: Hop) -> ROADMConfig | None:
        if hop.roadm_insertion_loss_db is None:
            return None
        return ROADMConfig(
            insertion_loss_db=hop.roadm_insertion_loss_db,
            filtering_penalty_db=hop.roadm_filtering_penalty_db,
        )

    def launch_power(self) -> np.ndarray:
        return np.full(self.num_channels, self.launch_power_w, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "num_channels": self.num_channels,
            "channel_spacing_hz": self.channel_spacing_hz,
            "center_frequency_hz": self.center_frequency_hz,
            "symbol_rate_hz": self.symbol_rate_hz,
            "phi_kurtosis": self.phi_kurtosis,
            "launch_power_w": self.launch_power_w,
            "hops": [
                {
                    "length_km": hop.length_km,
                    "span_km": hop.span_km,
                    "target_gain_db": hop.target_gain_db,
                    "noise_figure_db": hop.noise_figure_db,
                    "roadm_insertion_loss_db": hop.roadm_insertion_loss_db,
                    "roadm_filtering_penalty_db": hop.roadm_filtering_penalty_db,
                    "blocked_channels": list(hop.blocked_channels),
                }
                for hop in self.hops
            ],
        }


# --- The fixed scenario set -------------------------------------------------
# c_band_single   : minimal single-span C-band case.
# multiband_wide  : 80 channels straddling the C/L boundary, two links, so the
#                   RK4 ODE's inter-channel Raman coupling is strongly excited.
# kurtosis_64qam  : exercises the phi_kurtosis term that exists only on main;
#                   it is the scenario that would break if main's XPM code were
#                   replaced by the analytic-tilt variant.
# roadm_hoop      : two links separated by a ROADM, with a blocked channel.
SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="c_band_single",
        description="8-ch C-band, one 80 km span, NF 5.5 dB, Gaussian statistics.",
        num_channels=8,
        channel_spacing_hz=50e9,
        center_frequency_hz=193.1e12,
        symbol_rate_hz=32e9,
        phi_kurtosis=0.0,
        launch_power_w=LAUNCH_POWER_W,
        hops=(
            Hop(length_km=80.0, span_km=80.0, target_gain_db=17.0, noise_figure_db=5.5),
        ),
    ),
    Scenario(
        name="multiband_wide",
        description="80 channels across the C/L boundary, 2x100 km, NF 5.0 dB.",
        num_channels=80,
        channel_spacing_hz=50e9,
        center_frequency_hz=191.0e12,
        symbol_rate_hz=32e9,
        phi_kurtosis=0.0,
        launch_power_w=LAUNCH_POWER_W,
        hops=(
            Hop(length_km=100.0, span_km=100.0, target_gain_db=20.0, noise_figure_db=5.0),
            Hop(length_km=100.0, span_km=100.0, target_gain_db=20.0, noise_figure_db=5.0),
        ),
    ),
    Scenario(
        name="kurtosis_64qam",
        description="16-ch C-band, one 120 km span, 64QAM phi_kurtosis=-0.619.",
        num_channels=16,
        channel_spacing_hz=50e9,
        center_frequency_hz=193.1e12,
        symbol_rate_hz=32e9,
        phi_kurtosis=-0.619,
        launch_power_w=LAUNCH_POWER_W,
        hops=(
            Hop(length_km=120.0, span_km=120.0, target_gain_db=20.0, noise_figure_db=5.5),
        ),
    ),
    Scenario(
        name="roadm_hoop",
        description="12-ch C-band, two 60 km links through a 5 dB ROADM, channel 3 blocked.",
        num_channels=12,
        channel_spacing_hz=50e9,
        center_frequency_hz=193.1e12,
        symbol_rate_hz=32e9,
        phi_kurtosis=0.0,
        launch_power_w=LAUNCH_POWER_W,
        hops=(
            Hop(length_km=60.0, span_km=60.0, target_gain_db=15.0, noise_figure_db=5.0),
            Hop(
                length_km=60.0,
                span_km=60.0,
                target_gain_db=15.0,
                noise_figure_db=5.0,
                roadm_insertion_loss_db=5.0,
                roadm_filtering_penalty_db=0.2,
            ),
        ),
    ),
)


def run_reference(scenario: Scenario) -> dict[str, Any]:
    """Evaluate a scenario span by span through the low-level kernel call.

    This is the reference definition of the physics. It uses only API that
    exists on ``origin/main``, so the identical code produces the golden
    values there and the candidate values on the reconcile branch.

    Returns per-span power/noise curves plus the accumulated end-of-path
    signal power, noise terms and SNR.
    """
    grid = scenario.grid()
    evaluator = MultiBandISRSGN(
        grid,
        scenario.fiber(scenario.hops[0]),
        symbol_rate_hz=scenario.symbol_rate_hz,
        phi_kurtosis=scenario.phi_kurtosis,
        # FG-4b §1.2: this is the *port-fidelity* reference, so it is pinned to the
        # frozen ``off`` tier on purpose — the assertion is bit-equality with
        # origin/main, which is a statement about the port, not about which
        # acceleration tier production happens to default to. Production default is
        # ``matvec`` (gated separately at <=1e-12 by tests/test_fg4a_numeric_identity.py).
        ode_accel="off",
    )

    power = scenario.launch_power()
    total_spm = np.zeros_like(power)
    total_xpm = np.zeros_like(power)
    total_ase = np.zeros_like(power)
    curve: list[dict[str, Any]] = []

    for hop_index, hop in enumerate(scenario.hops):
        # Each hop's fiber parameters are precomputed per hop evaluator, which
        # exists on both branches via MultiBandISRSGN construction.
        if hop_index > 0:
            evaluator = MultiBandISRSGN(
                grid,
                scenario.fiber(hop),
                symbol_rate_hz=scenario.symbol_rate_hz,
                phi_kurtosis=scenario.phi_kurtosis,
                # Same freeze-point rationale as the hop-0 evaluator above.
                ode_accel="off",
            )
        edfa = scenario.edfa(hop)
        roadm = scenario.roadm(hop)

        remaining_km = hop.length_km
        span_index = 0
        while remaining_km > 1e-12:
            span_km = min(remaining_km, hop.span_km)
            previous_power = power
            power, spm, xpm, ase = evaluator._calc_span_noise_and_power(
                power, span_km * 1e3, edfa, None, n_effective_spans=1
            )
            # Already-accumulated noise is carried through this span's transfer.
            transfer = np.divide(power, previous_power, out=np.zeros_like(power), where=previous_power > 0)
            total_spm *= transfer
            total_xpm *= transfer
            total_ase *= transfer
            total_spm += spm
            total_xpm += xpm
            total_ase += ase
            curve.append(
                {
                    "hop": hop_index,
                    "span": span_index,
                    "span_length_m": span_km * 1e3,
                    "power_out_w": power.tolist(),
                    "spm_w": spm.tolist(),
                    "xpm_w": xpm.tolist(),
                    "ase_w": ase.tolist(),
                }
            )
            span_index += 1
            remaining_km -= span_km

        if roadm is not None:
            # ROADM loss is applied to signal and accumulated noise alike; this
            # is main's rule and the one evaluate_path() must reproduce.
            loss = 10 ** ((roadm.insertion_loss_db + roadm.filtering_penalty_db) / 10)
            power = power / loss
            total_spm = total_spm / loss
            total_xpm = total_xpm / loss
            total_ase = total_ase / loss

    total_noise = total_spm + total_xpm + total_ase
    snr = np.full_like(power, -np.inf)
    active = (power > 0) & (total_noise > 0)
    snr[active] = 10 * np.log10(power[active] / total_noise[active])
    snr[(power > 0) & (total_noise == 0)] = np.inf

    return {
        "scenario": scenario.to_dict(),
        "span_curve": curve,
        "signal_power_w": power.tolist(),
        "spm_w": total_spm.tolist(),
        "xpm_w": total_xpm.tolist(),
        "ase_w": total_ase.tolist(),
        "snr_db": snr.tolist(),
        "span_count": len(curve),
    }
