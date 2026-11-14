"""K-shortest-path routing and caching."""

import warnings
from itertools import islice
from typing import TYPE_CHECKING

import networkx as nx

from network_nlo_eval.core.types import LinkKey, NodeID, OriginalNodeID
from network_nlo_eval.network.topology import NetworkTopology

if TYPE_CHECKING:
    from collections.abc import Iterator


class PathCache:
    """Precompute K shortest simple paths for each connected node pair.

    Parameters
    ----------
    network_topology : NetworkTopology
        Topology providing the graph and internal node mapping.
    max_paths_per_pair : int, optional
        Maximum number of paths to cache for each node pair.
    """

    def __init__(self, network_topology: NetworkTopology, max_paths_per_pair: int = 5):
        """Initialize and precompute the path cache."""
        self.network_topology = network_topology
        self.max_paths_per_pair = max_paths_per_pair
        self._ksp_cache: dict[tuple[NodeID, NodeID], list[list[NodeID]]] = {}
        self._precompute_all_ksp_paths()

    def _precompute_all_ksp_paths(self) -> None:
        """Precompute paths for all node pairs using internal node IDs."""
        internal_nodes = self.network_topology.get_all_internal_nodes()

        # NetworkX uses the original graph and its optional weight attributes.
        nx_graph = self.network_topology.get_network_graph()

        # Visit each source-destination pair.
        for i in range(len(internal_nodes)):
            for j in range(
                i + 1, len(internal_nodes)
            ):  # The graph is undirected; compute each pair once.
                source_idx = internal_nodes[i]
                dest_idx = internal_nodes[j]

                # NetworkX operates on original node IDs.
                original_source_id = self.network_topology.get_original_node_id(
                    source_idx
                )
                original_dest_id = self.network_topology.get_original_node_id(dest_idx)

                # Compute shortest simple paths using edge weights.

                try:
                    paths_iterator: Iterator[list[OriginalNodeID]] = (
                        nx.shortest_simple_paths(
                            nx_graph,
                            original_source_id,
                            original_dest_id,
                            weight="weight",
                        )
                    )

                    # Limit the number of paths.
                    sliced_paths = list(islice(paths_iterator, self.max_paths_per_pair))

                    # Convert path nodes back to internal IDs.
                    internal_paths: list[list[NodeID]] = [
                        [
                            self.network_topology.get_internal_node_id(n_orig)
                            for n_orig in path
                        ]
                        for path in sliced_paths
                    ]

                    # Cache paths.
                    # Normalize keys as (min, max).
                    key = LinkKey(
                        (source_idx, dest_idx)
                        if source_idx < dest_idx
                        else (dest_idx, source_idx)
                    )
                    self._ksp_cache[key] = internal_paths
                except nx.NetworkXNoPath:
                    # Leave disconnected pairs out of the cache.
                    pass
                except (KeyError, TypeError, ValueError) as exc:
                    warnings.warn(
                        f"Error precomputing KSP for ({source_idx}, {dest_idx}): {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )

    def get_paths(self, source_idx: NodeID, dest_idx: NodeID) -> list[list[NodeID]]:
        """Return cached paths between internal node IDs.

        Parameters
        ----------
        source_idx : int
            Internal source node ID.
        dest_idx : int
            Internal destination node ID.

        Returns
        -------
        list of list of int
            Paths ordered from source to destination, or an empty list if
            the pair is absent or disconnected.
        """
        key = LinkKey(
            (source_idx, dest_idx) if source_idx < dest_idx else (dest_idx, source_idx)
        )
        paths = self._ksp_cache.get(key, [])

        # Reverse the cached paths when traversing the pair in reverse.
        if source_idx != key[0]:
            return [list(reversed(p)) for p in paths]
        return paths

    def get_all_ksp_pairs(self) -> list[tuple[NodeID, NodeID]]:
        """Return all cached source-destination pairs as normalized keys."""
        return list(self._ksp_cache.keys())
