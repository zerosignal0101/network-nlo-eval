"""路由计算 (K-Shortest Path) 模块。"""

from collections.abc import Iterator
from itertools import islice

import networkx as nx

from network_nlo_eval.core.types import LinkKey, NodeID, OriginalNodeID
from network_nlo_eval.network.topology import NetworkTopology


class PathCache:
    """K-Shortest Path (KSP) 缓存机制。

    预先计算网络中所有节点对之间的 K 条最短路径，以加速 RWA 过程。
    """

    def __init__(self, network_topology: NetworkTopology, max_paths_per_pair: int = 5):
        """初始化路径缓存。

        Args:
            network_topology: 网络拓扑管理实例。
            max_paths_per_pair: 为每对源-目的节点计算的最大路径数量。
        """
        self.network_topology = network_topology
        self.max_paths_per_pair = max_paths_per_pair
        self._ksp_cache: dict[tuple[NodeID, NodeID], list[list[NodeID]]] = {}
        self._precompute_all_ksp_paths()

    def _precompute_all_ksp_paths(self) -> None:
        """为网络中所有可能的节点对预计算 KSP 路径并缓存。

        路径以内部节点ID表示。
        """
        internal_nodes = self.network_topology.get_all_internal_nodes()

        # NetworkX 的 shortest_simple_paths 需要一个图对象。
        # 我们从 NetworkTopology 内部的 NetworkX 图获取。
        # 确保使用 "weight" 属性，如果它存在于原始图中
        nx_graph = self.network_topology._network_raw

        # 遍历所有可能的源-目的节点对
        for i in range(len(internal_nodes)):
            for j in range(i + 1, len(internal_nodes)):  # 无向图，只需计算一次
                source_idx = internal_nodes[i]
                dest_idx = internal_nodes[j]

                # NetworkX shortest_simple_paths 需要原始节点ID
                original_source_id = self.network_topology.get_original_node_id(source_idx)
                original_dest_id = self.network_topology.get_original_node_id(dest_idx)

                # 使用 NetworkX 计算 KSP
                # 假设链路长度存储在 'weight' 属性中
                # 确保 graph 是 NetworkX.Graph 类型，shortest_simple_paths 对 DiGraph 和 Graph 都适用

                try:
                    paths_iterator: Iterator[list[OriginalNodeID]] = nx.shortest_simple_paths(
                        nx_graph,
                        original_source_id,
                        original_dest_id,
                        weight="weight",
                    )

                    # 限制路径数量
                    sliced_paths = list(islice(paths_iterator, self.max_paths_per_pair))

                    # 将路径中的原始节点ID转换为内部节点ID
                    internal_paths: list[list[NodeID]] = [
                        [self.network_topology.get_internal_node_id(n_orig) for n_orig in path] for path in sliced_paths
                    ]

                    # 缓存路径
                    # 确保键是规范化的 (min, max)
                    key = LinkKey(sorted((source_idx, dest_idx)))
                    self._ksp_cache[key] = internal_paths
                except nx.NetworkXNoPath:
                    # 如果没有路径，则该对之间没有可用的 KSP
                    pass
                except Exception as e:
                    print(f"Error precomputing KSP for ({source_idx}, {dest_idx}): {e}")

    def get_paths(self, source_idx: NodeID, dest_idx: NodeID) -> list[list[NodeID]]:
        """从缓存中获取指定源-目的节点对的 KSP 路径。

        Args:
            source_idx: 源节点内部ID。
            dest_idx: 目的节点内部ID。

        Returns
        -------
            List[List[NodeID]]: 包含 KSP 路径的列表，每条路径是内部节点ID的列表。
                                如果没有路径或路径不存在于缓存中，返回空列表。
        """
        key = LinkKey(sorted((source_idx, dest_idx)))
        paths = self._ksp_cache.get(key, [])

        # 如果源节点不是路径的起始点，则反转路径
        if source_idx != key[0]:
            return [list(reversed(p)) for p in paths]
        return paths

    def get_all_ksp_pairs(self) -> list[tuple[NodeID, NodeID]]:
        """获取所有已缓存 KSP 路径的源-目的节点对 (规范化键)。"""
        return list(self._ksp_cache.keys())
