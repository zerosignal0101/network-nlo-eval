"""FU-4 fixed SNR-probe scenarios, shared by the tests and the evidence scripts.

Source of truth for the fidelity-tier gate introduced by FU-4, following the
same convention as ``tests/fg1_scenarios.py``: the scenario description lives in
``tests/``, and ``scripts/`` import it (never the other way round) so a test and
an evidence script can never drift apart.

Every field is a literal -- no RNG, no clock, no environment lookup -- so a
re-run on any machine must reproduce bit-identical numbers.

See ``docs/fu4_qot_assembly_adjudication.md`` for what the delta between the
FG-tree and union assemblies means, and why these scenarios are shaped this way.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import ServiceRequest

LAUNCH_POWER_W = 1e-3
SYMBOL_RATE_HZ = 32e9
LINK_LENGTH_KM = 80.0
MAX_SPAN_KM = 80.0
EDFA_NF_DB = 5.5
EDFA_GAIN_DB = 17.0
#: Probe channel: a mid-band index so ``drop``/``express`` asymmetry and any
#: per-connection frequency lookup are both exercised away from the edges.
PROBE_CHANNEL = 3

#: ROADM baseline shared by every scenario. ``insertion_loss_db=5``,
#: ``add_drop_loss_db=1`` -> drop = 6 dB, express = 5 dB. So the drop/express
#: asymmetry is non-degenerate in every single scenario: the two assemblies
#: cannot accidentally agree because the port losses happened to be equal.
ROADM_BASE: dict[str, Any] = {
    "insertion_loss_db": 5.0,
    "express_attenuation_db": 0.0,
    "add_drop_loss_db": 1.0,
    "booster_noise_figure_db": 5.0,
    "equalize_output_power": True,
    "filtering_penalty_db": 0.2,
}

@dataclass(frozen=True)
class Scenario:
    """One fully-specified, deterministic SNR probe.

    Every field is a literal: no RNG, no clock, no environment lookup.
    """

    name: str
    description: str
    kind: str  # "legacy" | "flex"
    #: internal node ids along the line, e.g. ``[0, 1, 2]`` = two hops
    path: tuple[int, ...]
    #: per-node ROADM keyword overrides, keyed by original node id
    roadm_overrides: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: channel / connection index handed to ``verify_allocation``
    channel_idx: int = PROBE_CHANNEL
    #: flexgrid only: slots occupied per connection
    slots_per_channel: int = 4

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of this scenario."""
        return {
            "name": self.name,
            "description": self.description,
            "kind": self.kind,
            "path": list(self.path),
            "roadm_overrides": {str(k): v for k, v in sorted(self.roadm_overrides.items())},
            "channel_idx": self.channel_idx,
            "slots_per_channel": self.slots_per_channel,
        }


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="legacy_two_hop_equalize_on",
        description="Fixed grid, 2 hops, booster on (equalize_output_power=True).",
        kind="legacy",
        path=(0, 1, 2),
    ),
    Scenario(
        name="legacy_three_hop_equalize_on",
        description="Fixed grid, 3 hops, booster on. Accumulates booster ASE 3x.",
        kind="legacy",
        path=(0, 1, 2, 3),
    ),
    Scenario(
        name="legacy_two_hop_equalize_off",
        description=(
            "Fixed grid, 2 hops, equalize_output_power=False. Answers the brief's required "
            "measurement: does the union collapse to the FG-tree result here?"
        ),
        kind="legacy",
        path=(0, 1, 2),
        roadm_overrides={1: {"equalize_output_power": False}, 2: {"equalize_output_power": False}},
    ),
    Scenario(
        name="legacy_express_attenuation",
        description="Fixed grid, 2 hops, express port attenuated 2 dB so express != drop.",
        kind="legacy",
        path=(0, 1, 2),
        roadm_overrides={1: {"express_attenuation_db": 2.0}},
    ),
    Scenario(
        name="legacy_high_filtering",
        description="Fixed grid, 2 hops, filtering_penalty_db=2.0 (heavily cascading WSS).",
        kind="legacy",
        path=(0, 1, 2),
        roadm_overrides={1: {"filtering_penalty_db": 2.0}},
    ),
    Scenario(
        name="legacy_blocked_channel",
        description="Fixed grid, 2 hops, probe channel blocked at node 1 -> SNR must be -inf.",
        kind="legacy",
        path=(0, 1, 2),
        roadm_overrides={1: {"blocked_channels": [PROBE_CHANNEL]}},
    ),
    Scenario(
        name="flex_two_hop_equalize_on",
        description="Flexible grid, 2 hops, booster on. Guards the FG-2 capability survives.",
        kind="flex",
        path=(0, 1, 2),
    ),
    Scenario(
        name="flex_three_hop_equalize_on",
        description="Flexible grid, 3 hops, booster on.",
        kind="flex",
        path=(0, 1, 2, 3),
    ),
    Scenario(
        name="flex_two_hop_equalize_off",
        description="Flexible grid, 2 hops, equalize_output_power=False.",
        kind="flex",
        path=(0, 1, 2),
        roadm_overrides={1: {"equalize_output_power": False}, 2: {"equalize_output_power": False}},
    ),
    Scenario(
        name="flex_blocked_slots",
        description=(
            "Flexible grid, 2 hops, probe connection's slot interval overlaps a blocked slot "
            "at node 1 -> SNR must be -inf. This is the FG-2 slicing semantics."
        ),
        kind="flex",
        path=(0, 1, 2),
        roadm_overrides={1: {"blocked_slots": [PROBE_CHANNEL * 4 + 1]}},
    ),
)


def _line_topology(num_nodes: int, scenario: Scenario) -> NetworkTopology:
    """Build a line graph carrying this scenario's per-node ROADM overrides."""
    graph = nx.Graph()
    for node in range(num_nodes):
        graph.add_node(node)
    for node in range(num_nodes - 1):
        graph.add_edge(
            node,
            node + 1,
            weight=LINK_LENGTH_KM,
            fiber={"length_km": LINK_LENGTH_KM, "max_span_length_km": MAX_SPAN_KM},
            edfa={"target_gain_db": EDFA_GAIN_DB, "noise_figure_db": EDFA_NF_DB},
        )
    for node, overrides in scenario.roadm_overrides.items():
        graph.nodes[node]["roadm"] = dict(overrides)

    return NetworkTopology(
        graph,
        default_fiber_config=FiberSpanConfig(length_km=LINK_LENGTH_KM, max_span_length_km=MAX_SPAN_KM),
        default_edfa_config=EDFAConfig(target_gain_db=EDFA_GAIN_DB, noise_figure_db=EDFA_NF_DB),
        default_roadm_config=ROADMConfig(**ROADM_BASE),
    )


def _grid_and_connections(
    scenario: Scenario,
) -> tuple[Any, list[ConnectionSpec] | None, SpectrumGrid, int]:
    """Build the evaluator grid, the connection list, the state grid, node count.

    Legacy scenarios use a fixed ``SpectrumGrid`` throughout.

    Flex scenarios use a ``FlexibleGrid`` for the evaluator (one ODE node per
    connection) but a legacy ``SpectrumGrid`` for ``NetworkState``, because
    ``NetworkState`` is indexed by ``spectrum_grid.num_channels`` and has no
    slot awareness -- occupancy bookkeeping lives in the connection list. The
    two grids have bit-identical centres (FG-2 §F2-1 froze exactly that), so
    the flex scenarios differ from the legacy ones in indexing, not physics.

    Returns
    -------
        (evaluator_grid, connections, state_grid, node_count)
    """
    num_channels = 8
    spacing_hz = 50e9
    centre_hz = 193.1e12
    legacy_grid = SpectrumGrid(num_channels, spacing_hz, centre_hz)

    if scenario.kind == "legacy":
        return legacy_grid, None, legacy_grid, num_channels

    flex = FlexibleGrid.from_legacy_grid(
        num_channels, spacing_hz, centre_hz, slots_per_channel=scenario.slots_per_channel
    )
    connections = [
        ConnectionSpec(
            grid=flex,
            connection_id=index,
            first_slot=index * scenario.slots_per_channel,
            n_slots=scenario.slots_per_channel,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=SYMBOL_RATE_HZ,
            roll_off=0.0,
            launch_power_w=LAUNCH_POWER_W,
        )
        for index in range(num_channels)
    ]
    return flex, connections, legacy_grid, num_channels


def run_scenario(scenario: Scenario) -> dict[str, Any]:
    """Run one scenario through the real ``verify_allocation`` and report SNR."""
    topology = _line_topology(max(scenario.path) + 1, scenario)
    grid, connections, state_grid, num_channels = _grid_and_connections(scenario)
    if scenario.channel_idx >= num_channels:
        raise ValueError(f"{scenario.name}: channel {scenario.channel_idx} >= {num_channels}")

    evaluator = MultiBandISRSGN(
        grid,
        topology.get_default_fiber_config(),
        symbol_rate_hz=SYMBOL_RATE_HZ,
        connections=connections,
        slots_per_channel=scenario.slots_per_channel if connections is not None else None,
    )
    state = NetworkState(topology, state_grid)
    validator = QoTValidator(evaluator, topology, grid)

    # Give every link a non-zero launch profile so XPM is actually excited:
    # an all-zero profile would make the XPM term vanish and hide assembly
    # differences that only show up in the nonlinear term.
    for link_state in state.get_all_link_states().values():
        link_state.launch_power_profile_w[:] = LAUNCH_POWER_W

    request = ServiceRequest(
        service_id=0,
        source_id=scenario.path[0],
        destination_id=scenario.path[-1],
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=-1e9,  # never gate on the requirement; we want the raw SNR
        launch_power_w=LAUNCH_POWER_W,
    )

    accepted, snrs_db = validator.verify_allocation(
        request, list(scenario.path), scenario.channel_idx, state
    )

    # A blocked channel makes the new service's SNR -inf, which fails the
    # requirement gate, so `verify_allocation` correctly returns
    # `(False, None)` with no array. That is a *rejection*, not a missing
    # measurement, and it is an invariant worth recording: both assemblies
    # must keep rejecting blocked channels.
    if snrs_db is None:
        return {
            "scenario": scenario.to_dict(),
            "node_count": int(validator.num_nodes),
            "accepted": False,
            "snr_db": None,
        }

    value = float(np.asarray(snrs_db, dtype=np.float64)[scenario.channel_idx])
    # Round-trips through JSON: -inf is not valid JSON, so encode it as a string
    # marker rather than letting the artefact become unparseable.
    serialisable: float | str
    if math.isinf(value) and value < 0:
        serialisable = "-inf"
    elif math.isinf(value):
        serialisable = "inf"
    else:
        serialisable = value

    return {
        "scenario": scenario.to_dict(),
        "node_count": int(validator.num_nodes),
        "accepted": bool(accepted),
        "snr_db": serialisable,
    }


def _numeric(snr: float | str | None) -> float | None:
    """Decode the JSON encoding back to a float (``-inf``/``inf`` included).

    ``None`` means the scenario was *rejected* by ``verify_allocation`` (no SNR
    array returned), which is a legitimate outcome and is preserved as-is
    rather than being coerced into a number.
    """
    if snr is None:
        return None
    if isinstance(snr, str):
        return float("-inf") if snr == "-inf" else float("inf")
    return float(snr)


def _cell(value: float | None) -> str:
    """Render one SNR cell for the markdown table."""
    if value is None:
        return "rejected"
    return f"{value:.6f}"


# ======================================================================
# Delta table
# ======================================================================
