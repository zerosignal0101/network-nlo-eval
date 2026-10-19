"""Diagnostic: unpack the SNR terms for the equalize_output_power=False case.

Purpose: confirm the union assembly does what the doc will claim, by printing
the actual signal/SPM/XPM/ASE terms rather than reasoning about them.

Two hand-derived expectations are checked:

* FG-tree assembly divides signal AND every noise term by the same lumped loss,
  so the loss cancels in the SNR ratio. FG-tree SNR should therefore be
  *invariant* to ``insertion_loss_db``.
* Union assembly with ``equalize_output_power=False`` keeps the port loss
  cancelling but leaves the filtering penalty acting on noise only, so FG-tree
  SNR minus union SNR should equal ``n_hops * filtering_penalty_db``.

Run:  PYTHONPATH=$PWD/src python scripts/fu4_probe_terms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from fu4_scenarios import (  # noqa: E402
    LAUNCH_POWER_W,
    SCENARIOS,
    SYMBOL_RATE_HZ,
    Scenario,
    run_scenario,
)

TARGET = "legacy_two_hop_equalize_off"


def main() -> int:
    """Print the sensitivity probes and the FG-tree/union decomposition."""
    print(f"load: {Path('/proc/loadavg').read_text().strip()}")

    # 1. Is the union SNR invariant to insertion_loss_db when equalize is off?
    print("\n== union SNR vs insertion_loss_db (equalize_output_power=False) ==")
    base = next(s for s in SCENARIOS if s.name == TARGET)
    for loss_db in (0.0, 5.0, 10.0, 20.0):
        variant = Scenario(
            name=f"{base.name}_il{loss_db}",
            description=base.description,
            kind=base.kind,
            path=base.path,
            roadm_overrides={
                node: {**override, "insertion_loss_db": loss_db}
                for node, override in base.roadm_overrides.items()
            },
            channel_idx=base.channel_idx,
            slots_per_channel=base.slots_per_channel,
        )
        print(f"  insertion_loss_db={loss_db:5.1f} -> {run_scenario(variant)['snr_db']}")

    # 2. Is the union SNR sensitive to filtering_penalty_db when equalize is off?
    print("\n== union SNR vs filtering_penalty_db (equalize_output_power=False) ==")
    for penalty_db in (0.0, 0.2, 1.0):
        variant = Scenario(
            name=f"{base.name}_fp{penalty_db}",
            description=base.description,
            kind=base.kind,
            path=base.path,
            roadm_overrides={
                node: {**override, "filtering_penalty_db": penalty_db}
                for node, override in base.roadm_overrides.items()
            },
            channel_idx=base.channel_idx,
            slots_per_channel=base.slots_per_channel,
        )
        print(f"  filtering_penalty_db={penalty_db:4.1f} -> {run_scenario(variant)['snr_db']}")

    # 3. drop vs express asymmetry, isolated on a single ROADM.
    print("\n== union SNR vs express attenuation, equalize ON (1 hop -> drop only) ==")
    one_hop = Scenario(
        name="probe_one_hop",
        description="single hop, probes the drop port only",
        kind="legacy",
        path=(0, 1),
    )
    for attenuation_db in (0.0, 3.0):
        variant = Scenario(
            name=f"probe_one_hop_{attenuation_db}",
            description=one_hop.description,
            kind=one_hop.kind,
            path=one_hop.path,
            roadm_overrides={1: {"express_attenuation_db": attenuation_db}},
            channel_idx=one_hop.channel_idx,
            slots_per_channel=one_hop.slots_per_channel,
        )
        print(f"  express_attenuation_db={attenuation_db:4.1f} -> {run_scenario(variant)['snr_db']}")

    decompose()
    print(f"\nload: {Path('/proc/loadavg').read_text().strip()}")
    return 0



def decompose() -> int:
    """Compare the union assembly against a faithful FG-tree reimplementation.

    Both sides are evaluated from the *same* noise caches, so the difference is
    attributable to the assembly rules alone rather than to the ODE kernel.
    """
    import math

    from fu4_scenarios import _grid_and_connections, _line_topology


    scenario = next(s for s in SCENARIOS if s.name == TARGET)
    topology = _line_topology(max(scenario.path) + 1, scenario)
    grid, connections, state_grid, num_channels = _grid_and_connections(scenario)
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.state import NetworkState
    from network_nlo_eval.rwa.qot_checker import QoTValidator

    evaluator = MultiBandISRSGN(
        grid, topology.get_default_fiber_config(), symbol_rate_hz=SYMBOL_RATE_HZ,
        connections=connections,
        slots_per_channel=scenario.slots_per_channel if connections is not None else None,
    )
    state = NetworkState(topology, state_grid)
    for ls in state.get_all_link_states().values():
        ls.launch_power_profile_w[:] = LAUNCH_POWER_W
    validator = QoTValidator(evaluator, topology, state_grid)

    # Build the noise caches exactly as verify_allocation would.
    path = list(scenario.path)
    profiles = {}
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        prof = state.get_link_state(u, v).launch_power_profile_w.copy()
        prof[scenario.channel_idx] = LAUNCH_POWER_W
        profiles[tuple(sorted((u, v)))] = prof
    caches = {k: validator._calc_link_noise(k, v, None) for k, v in profiles.items()}

    def accumulate(use_fg_tree: bool) -> tuple[float, float]:
        spm = xpm = ase = 0.0
        signal = LAUNCH_POWER_W
        spans = 0
        for i in range(len(path) - 1):
            key = tuple(sorted((path[i], path[i + 1])))
            c = caches[key]
            spm += float(c.spm[scenario.channel_idx])
            xpm += float(c.xpm[scenario.channel_idx])
            ase += float(c.ase[scenario.channel_idx])
            spans += c.span_count
            roadm = topology.get_roadm_config(path[i + 1])
            if use_fg_tree:
                # origin/main / FG-tree rule: lumped loss on signal AND noise.
                loss = 10 ** ((roadm.insertion_loss_db + roadm.filtering_penalty_db) / 10)
                signal /= loss
                spm /= loss
                xpm /= loss
                ase /= loss
            else:
                loss, fpen, nf = validator._roadm_terms(roadm, i == len(path) - 2)
                if nf is not None:
                    f_hz, rs_hz = validator._node_frequency_and_rate(scenario.channel_idx)
                    ase += nf * 6.62607015e-34 * f_hz * rs_hz * (loss - 1.0)
                else:
                    signal /= loss
                    spm /= loss
                    xpm /= loss
                    ase /= loss
                spm *= fpen
                xpm *= fpen
                ase *= fpen
        pen = spans ** 0.05 if spans > 0 else 1.0
        return signal, spm * pen + xpm + ase

    fg_sig, fg_noise = accumulate(use_fg_tree=True)
    un_sig, un_noise = accumulate(use_fg_tree=False)
    fg_snr = 10 * math.log10(fg_sig / fg_noise)
    un_snr = 10 * math.log10(un_sig / un_noise)
    print("\n== FG-tree rule vs union rule on identical noise caches ==")
    print(f"  FG-tree : signal={fg_sig:.6e} W noise={fg_noise:.6e} W SNR={fg_snr:.6f} dB")
    print(f"  union   : signal={un_sig:.6e} W noise={un_noise:.6e} W SNR={un_snr:.6f} dB")
    print(f"  delta   : {un_snr - fg_snr:+.6f} dB   (signal ratio {un_sig / fg_sig:.6f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
