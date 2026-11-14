"""Tests for topology mappings and physical configuration overrides."""

import networkx as nx
import pytest

from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology


def make_topology() -> NetworkTopology:
    graph = nx.Graph()
    graph.add_node(101, pos=(1, 2), roadm={"insertion_loss_db": 3})
    graph.add_node(203, edfa={"noise_figure_db": 6})
    graph.add_node(301)
    graph.add_edge(
        101,
        203,
        weight=80,
        fiber={"max_span_length_km": 40},
        edfa={"noise_figure_db": 4},
    )
    graph.add_edge(203, 301, length_km=100)
    return NetworkTopology(
        graph, FiberSpanConfig(length_km=50), EDFAConfig(), ROADMConfig()
    )


def test_mapping_and_edge_node_overrides() -> None:
    topology = make_topology()
    u, v, w = (topology.get_internal_node_id(node) for node in (101, 203, 301))
    assert topology.get_original_node_id(v) == 203
    assert topology.get_internal_node_count() == 3
    assert topology.get_adjacency_list()[v] == [u, w]
    assert topology.get_fiber_config(v, u).length_km == 80
    assert topology.get_fiber_config(u, v).max_span_length_km == 40
    assert topology.get_fiber_config(v, w).length_km == 100
    assert topology.get_link_edfa_config(u, v).noise_figure_db == 4
    assert topology.get_edfa_config(v).noise_figure_db == 6
    assert topology.get_roadm_config(u).insertion_loss_db == 3


def test_export_preserves_original_ids_and_physical_configuration() -> None:
    exported = make_topology().export_to_dict()
    assert exported["nodes"][101]["position"] == {"x": 1.0, "y": 2.0}
    assert exported["nodes"][101]["roadm"]["insertion_loss_db"] == 3
    first = exported["connections"][0]
    assert (first["from_node"], first["to_node"], first["length_km"]) == (101, 203, 80)
    assert first["edfa"]["noise_figure_db"] == 4


def test_optical_path_requires_two_nodes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        make_topology().build_optical_path([0])


def test_build_optical_path_preserves_order_and_link_configuration() -> None:
    topology = make_topology()
    path = topology.build_optical_path([0, 1, 2])
    assert path.node_path == [0, 1, 2]
    assert path.link_keys == [(0, 1), (1, 2)]
    assert [fiber.length_km for fiber in path.fiber_configs] == [80, 100]
    assert [edfa.noise_figure_db for edfa in path.edfa_configs] == [4, 5]
    with pytest.raises(ValueError, match="non-adjacent"):
        topology.build_optical_path([0, 2])
