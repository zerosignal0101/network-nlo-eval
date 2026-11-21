"""Tests for topology mappings and physical configuration overrides."""

import networkx as nx
import pytest

from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology


def make_topology() -> NetworkTopology:
    graph: nx.Graph[int] = nx.Graph()
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
    assert [
        edfa.noise_figure_db if edfa is not None else None for edfa in path.edfa_configs
    ] == [4, 5]
    with pytest.raises(ValueError, match="non-adjacent"):
        topology.build_optical_path([0, 2])


def test_roadm_path_loss_and_target_follow_different_degree_keys() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph(
        [(101, 203), (203, 301), (301, 203), (203, 101), (203, 400), (500, 203)]
    )
    graph.nodes[203]["roadm"] = {
        "degree_targets": {"301": {"power_w": 0.4e-3}},
        "path_losses": [
            {"from_degree": 101, "to_degree": 301, "loss_db": 3},
            {"from_degree": 301, "to_degree": 101, "loss_db": 9},
            {"from_degree": 101, "loss_db": 4},
        ],
    }
    topology = NetworkTopology(
        graph, FiberSpanConfig(length_km=80), EDFAConfig(), ROADMConfig()
    )

    def switching_config(*nodes: int) -> ROADMConfig:
        path = [topology.get_internal_node_id(node) for node in nodes]
        configs = topology.build_optical_path(path).roadm_configs
        assert configs is not None
        assert configs[0] is not None
        return configs[0]

    forward = switching_config(101, 203, 301)
    assert forward.path_loss_db("express") == 3
    assert forward.output_target is not None
    assert forward.output_target.power_w == 0.4e-3
    reverse = switching_config(301, 203, 101)
    assert reverse.path_loss_db("express") == 9
    assert reverse.output_target is None
    assert switching_config(101, 203, 400).path_loss_db("express") == 5
    assert switching_config(500, 203, 301).output_target == forward.output_target
    drop = switching_config(101, 203)
    assert drop.path_loss_db("drop") == 4
    assert drop.output_target is None
    exported = topology.export_to_dict()["nodes"][203]["roadm"]
    assert exported["degree_targets"]["301"]["power_w"] == 0.4e-3


def test_roadm_rejects_a_path_loss_on_the_wrong_degree_direction() -> None:
    graph: nx.DiGraph[int] = nx.DiGraph([(101, 203), (203, 301)])
    graph.nodes[203]["roadm"] = {"path_losses": [{"from_degree": 301, "loss_db": 2}]}
    with pytest.raises(ValueError, match="203 has no incoming degree from 301"):
        NetworkTopology(
            graph, FiberSpanConfig(length_km=80), EDFAConfig(), ROADMConfig()
        )
