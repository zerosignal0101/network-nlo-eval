"""NetworkX topology management, node mapping, and link configuration."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    OpticalSpanConfig,
    ROADMConfig,
)
from network_nlo_eval.network.equipment import EquipmentProfiles

if TYPE_CHECKING:
    import networkx as nx

    from network_nlo_eval.core.types import LinkKey, NodeID, OriginalNodeID
    from network_nlo_eval.models.base import MultiSpanOpticalPath


class NetworkTopology:
    """Manage node mappings and physical network configurations.

    The internal representation assigns consecutive integer node IDs for
    array operations; the underlying graph and exports use original IDs.

    Parameters
    ----------
    network_raw : networkx.Graph
        Original graph with node and edge attributes. Directed graphs use
        independent physical configurations for each direction.
    default_fiber_config : FiberSpanConfig
        Default fiber configuration, overridden by edge attributes.
    default_edfa_config : EDFAConfig
        Default amplifier configuration, overridden by node or edge attributes.
    default_roadm_config : ROADMConfig
        Default ROADM configuration, overridden by node attributes.
    equipment_profiles : EquipmentProfiles or mapping, optional
        Named physical settings. If omitted, use the graph's
        ``equipment_profiles`` attribute.
    """

    def __init__(
        self,
        network_raw: nx.Graph[OriginalNodeID] | nx.DiGraph[OriginalNodeID],
        default_fiber_config: FiberSpanConfig,
        default_edfa_config: EDFAConfig,
        default_roadm_config: ROADMConfig,
        *,
        equipment_profiles: EquipmentProfiles | Mapping[str, Any] | None = None,
    ):
        """Initialize the node mapping and element configurations."""
        if network_raw.is_multigraph():
            msg = "Parallel graph edges require distinct endpoint pairs."
            raise ValueError(msg)
        self._network_raw = network_raw
        self._directed = network_raw.is_directed()
        self._node_id_to_idx: dict[OriginalNodeID, NodeID] = {}
        self._idx_to_node_id: dict[NodeID, OriginalNodeID] = {}
        self._fiber_configs: dict[LinkKey, FiberSpanConfig] = {}
        self._link_edfa_configs: dict[LinkKey, EDFAConfig] = {}
        self._link_spans: dict[LinkKey, tuple[OpticalSpanConfig, ...]] = {}
        self._link_lengths_km: dict[LinkKey, float] = {}
        self._edfa_configs: dict[NodeID, EDFAConfig] = {}
        self._roadm_configs: dict[NodeID, ROADMConfig] = {}
        self._roadm_degree_configs: dict[LinkKey, ROADMConfig] = {}
        self._roadm_path_configs: dict[
            tuple[NodeID | None, NodeID, NodeID | None], ROADMConfig
        ] = {}
        self._adj_list: dict[NodeID, list[NodeID]] = {}

        self._default_fiber_config = default_fiber_config
        self._default_edfa_config = default_edfa_config
        self._default_roadm_config = default_roadm_config

        profiles = EquipmentProfiles.model_validate(
            network_raw.graph.get("equipment_profiles", {})
            if equipment_profiles is None
            else equipment_profiles
        )
        self._setup_node_mapping()
        self._parse_links_and_nodes_configs(profiles)

    def _setup_node_mapping(self) -> None:
        """Map original node IDs to consecutive internal integers."""
        for idx, original_id in enumerate(self._network_raw.nodes()):
            self._node_id_to_idx[original_id] = idx
            self._idx_to_node_id[idx] = original_id
            self._adj_list[idx] = []  # Initialize the adjacency list.

    def _parse_links_and_nodes_configs(self, profiles: EquipmentProfiles) -> None:
        """Parse edge and node attributes into physical configurations."""
        default_fiber = self._default_fiber_config.model_dump()
        default_edfa = self._default_edfa_config.model_dump()
        default_roadm = self._default_roadm_config.model_dump()
        for u_orig, v_orig, data in self._network_raw.edges(data=True):
            u_idx = self._node_id_to_idx[u_orig]
            v_idx = self._node_id_to_idx[v_orig]
            link_key = self.link_key(u_idx, v_idx)
            self._adj_list[u_idx].append(v_idx)
            if not self._directed:
                self._adj_list[v_idx].append(u_idx)

            if "spans" in data:
                raw_spans = data["spans"]
                if (
                    not self._directed
                    or not isinstance(raw_spans, (list, tuple))
                    or not raw_spans
                ):
                    msg = (
                        f"Link {u_orig}->{v_orig} requires a directed edge and at "
                        "least one span when specifying installed equipment."
                    )
                    raise ValueError(msg)
                spans = tuple(self._resolve_span(profiles, span) for span in raw_spans)
                self._link_spans[link_key] = spans
                self._link_lengths_km[link_key] = sum(
                    span.fiber.length_km for span in spans
                )
                continue

            # Support nested physical settings and legacy edge weights as lengths.
            placement = {}
            if "length_km" in data:
                placement["length_km"] = data["length_km"]
            elif "weight" in data:
                placement["length_km"] = data["weight"]
            fiber_overrides = profiles.resolve(
                "fiber",
                data.get("fiber", data.get("fiber_config", {})),
                placement=placement,
            )
            fiber_overrides.setdefault("length_km", 100.0)
            fiber_config = FiberSpanConfig.model_validate(
                default_fiber | fiber_overrides
            )
            edfa_overrides = profiles.resolve(
                "edfa", data.get("edfa", data.get("edfa_config", {}))
            )
            link_edfa = EDFAConfig.model_validate(default_edfa | edfa_overrides)

            self._fiber_configs[link_key] = fiber_config
            self._link_edfa_configs[link_key] = link_edfa
            self._link_lengths_km[link_key] = fiber_config.length_km

        for original_id, data in self._network_raw.nodes(data=True):
            node_idx = self._node_id_to_idx[original_id]
            edfa_overrides = profiles.resolve(
                "edfa", data.get("edfa", data.get("edfa_config", {}))
            )
            roadm_overrides = profiles.resolve(
                "roadm", data.get("roadm", data.get("roadm_config", {}))
            )
            self._edfa_configs[node_idx] = EDFAConfig.model_validate(
                default_edfa | edfa_overrides
            )
            roadm = ROADMConfig.model_validate(default_roadm | roadm_overrides)
            self._roadm_configs[node_idx] = roadm
            if roadm.degree_targets or roadm.path_losses:
                self._resolve_roadm_overrides(original_id, node_idx, roadm)

    @staticmethod
    def _resolve_span(profiles: EquipmentProfiles, span: Any) -> OpticalSpanConfig:
        if isinstance(span, Mapping):
            fields = dict(span)
            for kind in ("fiber", "edfa"):
                raw_element = fields.get(kind)
                if isinstance(raw_element, (str, Mapping)):
                    fields[kind] = profiles.resolve(kind, raw_element)
            return OpticalSpanConfig.model_validate(fields)
        return OpticalSpanConfig.model_validate(span)

    def _resolve_roadm_overrides(
        self, original_id: OriginalNodeID, node_idx: NodeID, roadm: ROADMConfig
    ) -> None:
        """Validate port directions and compile the configured switching overrides."""
        for egress_degree, target in roadm.degree_targets.items():
            if not self._network_raw.has_edge(original_id, egress_degree):
                msg = f"ROADM {original_id} has no outgoing degree to {egress_degree}."
                raise ValueError(msg)
            out_idx = self._node_id_to_idx[egress_degree]
            self._roadm_degree_configs[node_idx, out_idx] = roadm.model_copy(
                update={"output_target": target}
            )
        for override in roadm.path_losses:
            in_idx = None
            if (ingress := override.from_degree) is not None:
                if not self._network_raw.has_edge(ingress, original_id):
                    msg = f"ROADM {original_id} has no incoming degree from {ingress}."
                    raise ValueError(msg)
                in_idx = self._node_id_to_idx[ingress]
            out_idx_or_drop = None
            template = roadm
            if (to_degree := override.to_degree) is not None:
                if not self._network_raw.has_edge(original_id, to_degree):
                    msg = f"ROADM {original_id} has no outgoing degree to {to_degree}."
                    raise ValueError(msg)
                out_idx_or_drop = self._node_id_to_idx[to_degree]
                template = self._roadm_degree_configs.get(
                    (node_idx, out_idx_or_drop), roadm
                )
            path_config = template.model_copy(
                update={
                    "insertion_loss_db": override.loss_db,
                    "express_attenuation_db": 0.0,
                    "add_drop_loss_db": 0.0,
                }
            )
            self._roadm_path_configs[in_idx, node_idx, out_idx_or_drop] = path_config

    def get_internal_node_id(self, original_id: OriginalNodeID) -> NodeID:
        """Return the internal ID of an original node."""
        return self._node_id_to_idx[original_id]

    def get_original_node_id(self, internal_id: NodeID) -> OriginalNodeID:
        """Return the original ID of an internal node."""
        return self._idx_to_node_id[internal_id]

    def get_internal_node_count(self) -> int:
        """Return the number of internal nodes."""
        return len(self._node_id_to_idx)

    def link_key(self, u_idx: NodeID, v_idx: NodeID) -> LinkKey:
        """Keep directed endpoint order or canonicalize an undirected pair."""
        if self._directed or u_idx <= v_idx:
            return u_idx, v_idx
        return v_idx, u_idx

    def get_fiber_config(self, u_idx: NodeID, v_idx: NodeID) -> FiberSpanConfig:
        """Return the fiber configuration of an implicitly split link."""
        return self._fiber_configs[self.link_key(u_idx, v_idx)]

    def get_edfa_config(self, node_idx: NodeID) -> EDFAConfig:
        """Return the EDFA configuration for a node."""
        return self._edfa_configs[node_idx]

    def get_link_edfa_config(self, u_idx: NodeID, v_idx: NodeID) -> EDFAConfig:
        """Return the amplifier configuration of an implicitly split link."""
        return self._link_edfa_configs[self.link_key(u_idx, v_idx)]

    def get_link_length_km(self, u_idx: NodeID, v_idx: NodeID) -> float:
        """Return total fiber length for a link."""
        return self._link_lengths_km[self.link_key(u_idx, v_idx)]

    def get_link_spans(
        self, u_idx: NodeID, v_idx: NodeID
    ) -> tuple[OpticalSpanConfig, ...] | None:
        """Return installed spans, or None for a link with implicit splitting."""
        return self._link_spans.get(self.link_key(u_idx, v_idx))

    def get_roadm_config(self, node_idx: NodeID) -> ROADMConfig:
        """Return the ROADM configuration for a node."""
        return self._roadm_configs[node_idx]

    def get_roadm_path_config(self, path: list[NodeID], hop_index: int) -> ROADMConfig:
        """Return switching settings at the receiving node of a route hop."""
        node = path[hop_index + 1]
        roadm = self._roadm_configs[node]
        if not (roadm.degree_targets or roadm.path_losses):
            return roadm
        egress = path[hop_index + 2] if hop_index + 2 < len(path) else None
        override = self._roadm_path_configs.get((path[hop_index], node, egress))
        if override is not None:
            return override
        if egress is not None:
            return self._roadm_degree_configs.get((node, egress), roadm)
        return roadm

    def get_roadm_add_config(self, node: NodeID, egress: NodeID) -> ROADMConfig:
        """Resolve local add-port loss and the outgoing degree target."""
        override = self._roadm_path_configs.get((None, node, egress))
        if override is not None:
            return override
        return self._roadm_degree_configs.get((node, egress), self._roadm_configs[node])

    def get_adjacency_list(self) -> dict[NodeID, list[NodeID]]:
        """Return the adjacency list indexed by internal node IDs."""
        return self._adj_list

    def get_all_internal_nodes(self) -> list[NodeID]:
        """Return all internal node IDs."""
        return list(self._idx_to_node_id.keys())

    def get_all_internal_links(self) -> list[LinkKey]:
        """Return the keys of all internal links."""
        return list(self._link_lengths_km)

    def get_network_graph(
        self,
    ) -> nx.Graph[OriginalNodeID] | nx.DiGraph[OriginalNodeID]:
        """Return the underlying NetworkX graph for read-only path computations."""
        return self._network_raw

    def get_default_fiber_config(self) -> FiberSpanConfig:
        """Return the default fiber configuration."""
        return self._default_fiber_config

    def get_default_edfa_config(self) -> EDFAConfig:
        """Return the default EDFA configuration."""
        return self._default_edfa_config

    def get_default_roadm_config(self) -> ROADMConfig:
        """Return the default ROADM configuration."""
        return self._default_roadm_config

    def build_optical_path(
        self, path: list[NodeID], *, fixed_span_length_km: float | None = None
    ) -> MultiSpanOpticalPath:
        """Convert internal node IDs to an ordered physical optical path."""
        if len(path) < 2:
            msg = "An optical path must contain at least two nodes."
            raise ValueError(msg)
        if fixed_span_length_km is not None and (
            not math.isfinite(fixed_span_length_km) or fixed_span_length_km <= 0
        ):
            msg = "The fixed span length must be finite and positive."
            raise ValueError(msg)
        from network_nlo_eval.models import base  # noqa: PLC0415

        fiber_configs: list[FiberSpanConfig] = []
        edfa_configs: list[EDFAConfig | None] = []
        roadm_configs: list[ROADMConfig | None] = []
        link_keys: list[LinkKey] = []
        # The shifted sequence intentionally has one fewer element.
        for hop_index, (u_idx, v_idx) in enumerate(itertools.pairwise(path)):
            link_key = self.link_key(u_idx, v_idx)
            if link_key not in self._link_lengths_km:
                msg = f"Path contains non-adjacent nodes: {u_idx}, {v_idx}."
                raise ValueError(msg)
            spans = self._link_spans.get(link_key)
            roadm = self.get_roadm_path_config(path, hop_index)
            if spans is None:
                fiber = self._fiber_configs[link_key]
                if fixed_span_length_km is not None:
                    fiber = fiber.model_copy(
                        update={"max_span_length_km": fixed_span_length_km}
                    )
                fiber_configs.append(fiber)
                edfa_configs.append(self._link_edfa_configs[link_key])
                roadm_configs.append(roadm)
                link_keys.append(link_key)
                continue
            for span in spans:
                fiber_configs.append(span.fiber)
                edfa_configs.append(span.edfa)
                roadm_configs.append(None)
                link_keys.append(link_key)
            roadm_configs[-1] = roadm
        return base.MultiSpanOpticalPath(
            fiber_configs=fiber_configs,
            edfa_configs=edfa_configs,
            roadm_configs=roadm_configs,
            link_keys=link_keys,
            node_path=path.copy(),
        )

    def export_to_dict(self) -> dict[str, Any]:
        """Export the topology as a dictionary for visualization or storage."""
        exported_nodes = []
        for original_node_id, data in self._network_raw.nodes(data=True):
            pos = data.get("pos")
            x, y = pos or (data.get("x", 0.0), data.get("y", 0.0))
            node_idx = self._node_id_to_idx[original_node_id]
            exported_nodes.append(
                {
                    "id": int(original_node_id),
                    "name": str(original_node_id),
                    "position": {"x": float(x), "y": float(y)},
                    "edfa": self.get_edfa_config(node_idx).model_dump(mode="json"),
                    "roadm": self.get_roadm_config(node_idx).model_dump(mode="json"),
                }
            )

        exported_connections = []
        connection_id_counter = itertools.count()
        for u, v, _ in self._network_raw.edges(data=True):
            u_idx = self._node_id_to_idx[u]
            v_idx = self._node_id_to_idx[v]
            key = self.link_key(u_idx, v_idx)
            connection: dict[str, Any] = {
                "id": next(connection_id_counter),
                "from_node": int(u),
                "to_node": int(v),
                "length_km": self._link_lengths_km[key],
            }
            if (spans := self._link_spans.get(key)) is None:
                connection["fiber"] = self._fiber_configs[key].model_dump(mode="json")
                connection["edfa"] = self._link_edfa_configs[key].model_dump(
                    mode="json"
                )
            else:
                connection["spans"] = [span.model_dump(mode="json") for span in spans]
            exported_connections.append(connection)

        return {
            "directed": self._directed,
            "nodes": {node["id"]: node for node in exported_nodes},
            "connections": {conn["id"]: conn for conn in exported_connections},
        }
