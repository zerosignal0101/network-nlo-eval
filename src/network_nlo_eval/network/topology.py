"""基于 NetworkX 的网络图管理，处理节点ID映射和链路配置."""

import itertools
from typing import Any

import networkx as nx

from network_nlo_eval.core.types import LinkKey, NodeID, OriginalNodeID
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig


class NetworkTopology:
    """
    管理网络拓扑结构，包括节点ID映射、链路距离、以及链路和节点上的物理元件配置。

    内部使用连续整数ID (0, 1, ...) 来表示节点，以便于数组操作和Numba JIT。
    外部API和导出将使用原始拓扑中的节点ID。
    """

    def __init__(
        self,
        network_raw: nx.Graph,
        default_fiber_config: FiberSpanConfig,
        default_edfa_config: EDFAConfig,
        default_roadm_config: ROADMConfig,
    ):
        """
        初始化网络拓扑。

        Args:
            network_raw: 原始 NetworkX 图，节点ID可以是任意可哈希类型。
            default_fiber_config: 默认的光纤跨段配置。
            default_edfa_config: 默认的 EDFA 配置。
            default_roadm_config: 默认的 ROADM 配置。
        """
        self._network_raw = network_raw
        self._node_id_to_idx: dict[OriginalNodeID, NodeID] = {}
        self._idx_to_node_id: dict[NodeID, OriginalNodeID] = {}
        self._fiber_configs: dict[LinkKey, FiberSpanConfig] = {}
        self._edfa_configs: dict[NodeID, EDFAConfig] = {}
        self._roadm_configs: dict[NodeID, ROADMConfig] = {}
        self._adj_list: dict[NodeID, list[NodeID]] = {}

        self._default_fiber_config = default_fiber_config
        self._default_edfa_config = default_edfa_config
        self._default_roadm_config = default_roadm_config

        self._setup_node_mapping()
        self._parse_links_and_nodes_configs()

    def _setup_node_mapping(self) -> None:
        """为原始节点ID创建内部连续整数ID映射."""
        for idx, original_id in enumerate(self._network_raw.nodes()):
            self._node_id_to_idx[original_id] = idx
            self._idx_to_node_id[idx] = original_id
            self._adj_list[idx] = []  # 初始化邻接列表

    def _parse_links_and_nodes_configs(self) -> None:
        """解析链路配置 (如长度) 和节点配置 (如 EDFA/ROADM)。

        为每个链路和节点分配配置对象。
        """
        for u_orig, v_orig, data in self._network_raw.edges(data=True):
            u_idx = self._node_id_to_idx[u_orig]
            v_idx = self._node_id_to_idx[v_orig]
            link_key = LinkKey((u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx))  # 规范化链路键

            # 复制默认配置，并用链路特有数据覆盖
            fiber_config = self._default_fiber_config.model_copy(deep=True)
            fiber_config.length_km = data.get("weight", 100.0)  # 假设 'weight' 是长度 (km)

            self._fiber_configs[link_key] = fiber_config
            self._adj_list[u_idx].append(v_idx)
            self._adj_list[v_idx].append(u_idx)  # 无向图

        for original_id, _data in self._network_raw.nodes(data=True):
            node_idx = self._node_id_to_idx[original_id]
            # 默认所有节点都有 EDFA 和 ROADM，可根据需要调整
            self._edfa_configs[node_idx] = self._default_edfa_config.model_copy(deep=True)
            self._roadm_configs[node_idx] = self._default_roadm_config.model_copy(deep=True)

    def get_internal_node_id(self, original_id: OriginalNodeID) -> NodeID:
        """根据原始节点ID获取内部节点ID."""
        return self._node_id_to_idx[original_id]

    def get_original_node_id(self, internal_id: NodeID) -> OriginalNodeID:
        """根据内部节点ID获取原始节点ID."""
        return self._idx_to_node_id[internal_id]

    def get_internal_node_count(self) -> int:
        """获取内部节点的总数量."""
        return len(self._node_id_to_idx)

    def get_fiber_config(self, u_idx: NodeID, v_idx: NodeID) -> FiberSpanConfig:
        """获取指定链路的光纤配置."""
        link_key = LinkKey((u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx))
        return self._fiber_configs[link_key]

    def get_edfa_config(self, node_idx: NodeID) -> EDFAConfig:
        """获取指定节点的 EDFA 配置."""
        return self._edfa_configs[node_idx]

    def get_roadm_config(self, node_idx: NodeID) -> ROADMConfig:
        """获取指定节点的 ROADM 配置."""
        return self._roadm_configs[node_idx]

    def get_adjacency_list(self) -> dict[NodeID, list[NodeID]]:
        """获取内部节点ID的邻接列表."""
        return self._adj_list

    def get_all_internal_nodes(self) -> list[NodeID]:
        """获取所有内部节点ID的列表."""
        return list(self._idx_to_node_id.keys())

    def get_all_internal_links(self) -> list[LinkKey]:
        """获取所有内部链路的规范化键列表."""
        return list(self._fiber_configs.keys())

    def get_default_fiber_config(self) -> FiberSpanConfig:
        """获取默认光纤配置."""
        return self._default_fiber_config

    def get_default_edfa_config(self) -> EDFAConfig:
        """获取默认EDFA配置."""
        return self._default_edfa_config

    def get_default_roadm_config(self) -> ROADMConfig:
        """获取默认ROADM配置."""
        return self._default_roadm_config

    def export_to_dict(self) -> dict[str, Any]:
        """将拓扑数据导出为字典，主要用于可视化或存储."""
        exported_nodes = []
        for original_node_id, data in self._network_raw.nodes(data=True):
            pos = data.get("pos")
            x, y = pos if pos else (data.get("x", 0.0), data.get("y", 0.0))
            exported_nodes.append(
                {
                    "id": int(original_node_id),
                    "name": str(original_node_id),
                    "position": {"x": float(x), "y": float(y)},
                }
            )

        exported_connections = []
        connection_id_counter = itertools.count()
        for u, v, _ in self._network_raw.edges(data=True):
            exported_connections.append(
                {
                    "id": next(connection_id_counter),
                    "from_node": int(u),
                    "to_node": int(v),
                }
            )

        return {
            "nodes": {node["id"]: node for node in exported_nodes},
            "connections": {conn["id"]: conn for conn in exported_connections},
        }
