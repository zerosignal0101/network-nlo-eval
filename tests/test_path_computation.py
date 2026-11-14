"""Tests for weighted shortest path caching."""

import networkx as nx

from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.path_computation import PathCache


def test_weighted_paths_are_ordered_and_reversible() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from([(10, 20, 1), (20, 30, 1), (10, 30, 5)])
    graph.add_node(40)
    topology = NetworkTopology(
        graph, FiberSpanConfig(length_km=10), EDFAConfig(), ROADMConfig()
    )
    cache = PathCache(topology, max_paths_per_pair=2)
    source, middle, target, isolated = (
        topology.get_internal_node_id(i) for i in (10, 20, 30, 40)
    )
    assert cache.get_paths(source, target) == [
        [source, middle, target],
        [source, target],
    ]
    assert cache.get_paths(target, source) == [
        [target, middle, source],
        [target, source],
    ]
    assert cache.get_paths(source, isolated) == []
    assert (source, isolated) not in cache.get_all_ksp_pairs()
