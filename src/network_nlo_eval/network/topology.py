"""基于 NetworkX 的网络图管理，处理节点ID映射和链路配置."""

import itertools
from typing import TYPE_CHECKING, Any

import networkx as nx

from network_nlo_eval.core.types import LinkKey, NodeID, OriginalNodeID
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

if TYPE_CHECKING:
    from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath


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
        self._link_edfa_configs: dict[LinkKey, EDFAConfig] = {}
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

            # 支持 topology JSON 中的嵌套物理参数，同时兼容 weight=长度的旧格式。
            fiber_overrides = dict(data.get("fiber", data.get("fiber_config", {})))
            fiber_overrides.setdefault("length_km", data.get("length_km", data.get("weight", 100.0)))
            fiber_config = FiberSpanConfig.model_validate(
                {**self._default_fiber_config.model_dump(), **fiber_overrides}
            )
            edfa_overrides = dict(data.get("edfa", data.get("edfa_config", {})))
            link_edfa = EDFAConfig.model_validate({**self._default_edfa_config.model_dump(), **edfa_overrides})

            self._fiber_configs[link_key] = fiber_config
            self._link_edfa_configs[link_key] = link_edfa
            self._adj_list[u_idx].append(v_idx)
            self._adj_list[v_idx].append(u_idx)  # 无向图

        for original_id, data in self._network_raw.nodes(data=True):
            node_idx = self._node_id_to_idx[original_id]
            edfa_overrides = dict(data.get("edfa", data.get("edfa_config", {})))
            roadm_overrides = dict(data.get("roadm", data.get("roadm_config", {})))
            self._edfa_configs[node_idx] = EDFAConfig.model_validate(
                {**self._default_edfa_config.model_dump(), **edfa_overrides}
            )
            self._roadm_configs[node_idx] = ROADMConfig.model_validate(
                {**self._default_roadm_config.model_dump(), **roadm_overrides}
            )

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

    def get_link_edfa_config(self, u_idx: NodeID, v_idx: NodeID) -> EDFAConfig:
        """获取链路跨段放大器配置."""
        link_key = LinkKey((u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx))
        return self._link_edfa_configs[link_key]

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

    def get_network_graph(self) -> nx.Graph:
        """返回只读约定的底层 NetworkX 图供路径算法使用."""
        return self._network_raw

    def get_default_fiber_config(self) -> FiberSpanConfig:
        """获取默认光纤配置."""
        return self._default_fiber_config

    def get_default_edfa_config(self) -> EDFAConfig:
        """获取默认EDFA配置."""
        return self._default_edfa_config

    def get_default_roadm_config(self) -> ROADMConfig:
        """获取默认ROADM配置."""
        return self._default_roadm_config

    def build_optical_path(self, path: list[NodeID]) -> "MultiSpanOpticalPath":
        """将节点路径转换为统计与波形引擎共享的有序物理路径."""
        if len(path) < 2:
            raise ValueError("An optical path must contain at least two nodes.")
        from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath

        fiber_configs: list[FiberSpanConfig] = []
        edfa_configs: list[EDFAConfig] = []
        roadm_configs: list[ROADMConfig] = []
        link_keys: list[LinkKey] = []
        # 相邻节点对：path[1:] 比 path 少一个元素，长度不相等是预期的。
        for u_idx, v_idx in zip(path, path[1:], strict=False):
            link_key = LinkKey((u_idx, v_idx) if u_idx < v_idx else (v_idx, u_idx))
            if link_key not in self._fiber_configs:
                raise ValueError(f"Path contains non-adjacent nodes: {u_idx}, {v_idx}.")
            link_keys.append(link_key)
            fiber_configs.append(self.get_fiber_config(u_idx, v_idx))
            edfa_configs.append(self.get_link_edfa_config(u_idx, v_idx))
            roadm_configs.append(self.get_roadm_config(v_idx))
        return MultiSpanOpticalPath(
            fiber_configs=fiber_configs,
            edfa_configs=edfa_configs,
            roadm_configs=roadm_configs,
            link_keys=link_keys,
            node_path=path.copy(),
        )

    def export_to_dict(self) -> dict[str, Any]:
        """将拓扑数据导出为字典，主要用于可视化或存储."""
        exported_nodes = []
        for original_node_id, data in self._network_raw.nodes(data=True):
            pos = data.get("pos")
            x, y = pos if pos else (data.get("x", 0.0), data.get("y", 0.0))
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
            fiber = self.get_fiber_config(u_idx, v_idx)
            exported_connections.append(
                {
                    "id": next(connection_id_counter),
                    "from_node": int(u),
                    "to_node": int(v),
                    "length_km": fiber.length_km,
                    "fiber": fiber.model_dump(mode="json"),
                    "edfa": self.get_link_edfa_config(u_idx, v_idx).model_dump(mode="json"),
                }
            )

        return {
            "nodes": {node["id"]: node for node in exported_nodes},
            "connections": {conn["id"]: conn for conn in exported_connections},
        }
