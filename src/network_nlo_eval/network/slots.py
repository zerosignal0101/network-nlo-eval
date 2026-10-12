"""灵活栅格的槽位化占用与资产 schema 迁移（设计 §2.6）。

新语义：`blocked_slots` 存**槽索引**（12.5 GHz 粒度），`blocked_channels`
降级为**只读兼容输入**。固定 50 GHz 栅格下 `slots_per_channel = 4`，故
`len(blocked_slots) == 4 * len(blocked_channels)`。

语义差异（必须写进迁移说明）：旧语义"阻断信道 i"= 该 50 GHz 不可用；新语义
"阻断槽 s"= 该 12.5 GHz 不可用。**灵活栅格下一个宽连接若与任一被阻断槽重叠即
不可行**——阻断粒度变细，可能让原本可行的解变得不可行。这是**语义改**。

兼容策略（冻结）：

1. 解析顺序：若存在 `blocked_slots` → 用它；**否则**若存在 `blocked_channels`
   → 按 `first_slot = index * slots_per_channel` 展开为区间。
2. `slots_per_channel = round(channel_spacing_hz / slot_width_hz)`；若
   `channel_spacing_hz` 不是槽宽整数倍 → `ValueError`（不做取整猜测）。
3. 两者**同时**出现且展开后不一致 → `ValueError`（不做静默合并）。
4. 越界：任一槽 `0 <= s < num_slots`，否则 `ValueError`（构建期拦截，不静默丢弃）。
5. 迁移**只写新字段，保留旧字段原样**，保证迁移可重入、diff 可读。
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "slots_per_channel",
    "expand_blocked_channels",
    "resolve_blocked_slots",
    "migrate_roadm_blocked_slots",
    "connection_overlaps_slots",
]


def slots_per_channel(channel_spacing_hz: float, slot_width_hz: float) -> int:
    """遗留信道间隔折算的每信道槽数。

    Args:
        channel_spacing_hz: 遗留信道间隔 (Hz)。
        slot_width_hz: 槽宽 (Hz)。

    Returns
    -------
        每信道槽数 (int)，>= 1。

    Raises
    ------
        ValueError: slot_width_hz <= 0；channel_spacing_hz 不是槽宽的整数倍
            （不做取整猜测）。
    """
    if slot_width_hz <= 0:
        raise ValueError(f"slot_width_hz must be positive, got {slot_width_hz}.")
    if channel_spacing_hz <= 0:
        raise ValueError(f"channel_spacing_hz must be positive, got {channel_spacing_hz}.")
    exact = channel_spacing_hz / slot_width_hz
    nearest = round(exact)
    if nearest < 1 or not math.isclose(exact, nearest, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"channel_spacing_hz ({channel_spacing_hz}) must be an integer multiple of "
            f"slot_width_hz ({slot_width_hz}); got ratio {exact}. Refusing to round."
        )
    return int(nearest)


def expand_blocked_channels(blocked_channels: list[int], slots_per_channel_: int) -> list[int]:
    """把遗留信道索引展开为槽索引区间。

    信道 i -> 槽区间 ``[i*slots_per_channel_, (i+1)*slots_per_channel_)``。

    Raises
    ------
        ValueError: 存在负信道索引。
    """
    if slots_per_channel_ < 1:
        raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel_}.")
    slots: list[int] = []
    for index in blocked_channels:
        if index < 0:
            raise ValueError(f"ROADM blocked channel indices must be non-negative, got {index}.")
        slots.extend(range(index * slots_per_channel_, (index + 1) * slots_per_channel_))
    return slots


def resolve_blocked_slots(
    roadm: dict[str, Any],
    slot_width_hz: float,
    channel_spacing_hz: float,
    num_slots: int,
) -> list[int]:
    """按冻结策略解析一个 ``roadm`` 字典的权威阻断槽列表。

    Args:
        roadm: 节点的 ``roadm`` 子字典。
        slot_width_hz: 槽宽 (Hz)。
        channel_spacing_hz: 遗留信道间隔 (Hz)。
        num_slots: 栅格槽总数，用于越界校验。

    Returns
    -------
        升序去重后的槽索引列表。

    Raises
    ------
        ValueError: 槽宽/间隔非法、间隔非槽宽整数倍、新旧字段冲突、槽索引越界。
    """
    per_channel = slots_per_channel(channel_spacing_hz, slot_width_hz)
    has_slots = "blocked_slots" in roadm
    has_channels = "blocked_channels" in roadm
    if not has_slots and not has_channels:
        return []

    if has_slots:
        authoritative = [int(s) for s in roadm["blocked_slots"]]
        if has_channels and roadm["blocked_channels"]:
            legacy = expand_blocked_channels(list(roadm["blocked_channels"]), per_channel)
            if sorted(set(authoritative)) != sorted(set(legacy)):
                raise ValueError(
                    f"roadm has both blocked_slots {sorted(set(authoritative))} and blocked_channels "
                    f"{list(roadm['blocked_channels'])} which expand to {sorted(set(legacy))}; "
                    "refusing to merge silently."
                )
    else:
        authoritative = expand_blocked_channels(list(roadm["blocked_channels"]), per_channel)

    for slot in authoritative:
        if not (0 <= slot < num_slots):
            raise ValueError(f"ROADM blocked slot {slot} out of bounds for {num_slots} slots.")
    return sorted(set(authoritative))


def connection_overlaps_slots(first_slot: int, n_slots: int, blocked: set[int]) -> bool:
    """连接槽区间是否与任一被阻断槽重叠（切片语义，设计 §2.5 末段）。"""
    if not blocked:
        return False
    window = range(first_slot, first_slot + n_slots)
    return any(slot in blocked for slot in window)


def _declared_num_slots(graph_json: dict[str, Any], nodes: list[Any]) -> int:
    """从图 JSON 中解析声明的槽总数，供迁移时的越界校验使用。

    冻结签名 `migrate_roadm_blocked_slots` **不含** `num_slots` 参数，因此容量
    只能从图里读。查找顺序（均为可选声明）：

    1. ``graph_json["num_slots"]``
    2. ``graph_json["graph"]["num_slots"]``
    3. ``graph_json["graph"]["num_channels"] * slots_per_channel``（遗留图）
    4. 任一 ``node["roadm"]["num_slots"]``

    全部缺失时返回 0，调用方**跳过**越界校验（负索引仍照常抛错）。这一点如实
    标注：现网资产（`assets/*.json`）既无 `blocked_slots` 也未声明槽数，越界拦截
    在迁移路径上需由调用方先写入容量声明才会生效；`resolve_blocked_slots(...)`
    显式接收 `num_slots`，始终执行越界校验（F3-4 的执行点）。
    """
    candidates: list[Any] = [graph_json.get("num_slots")]
    graph_block = graph_json.get("graph")
    if isinstance(graph_block, dict):
        candidates.append(graph_block.get("num_slots"))
        candidates.append(graph_block.get("num_channels"))
    for node in nodes:
        roadm = node.get("roadm") if isinstance(node, dict) else None
        if isinstance(roadm, dict):
            candidates.append(roadm.get("num_slots"))
    for value in candidates:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value > 0:
            return int(value)
    return 0


def migrate_roadm_blocked_slots(
    graph_json: dict[str, Any],
    slot_width_hz: float,
    channel_spacing_hz: float,
) -> tuple[dict[str, Any], dict[str, int]]:
    """把 node["roadm"]["blocked_channels"] 迁移为 node["roadm"]["blocked_slots"]。

    幂等：已存在且一致的 `blocked_slots` 的节点记为 `nodes_skipped`，不改写。
    只写新字段、保留旧字段原样，因此重跑第二次 `nodes_migrated == 0`。

    Args:
        graph_json: load_graph() 的返回值（原样传入，不修改）。
        slot_width_hz: 槽宽 (Hz)，须能整除 channel_spacing_hz。
        channel_spacing_hz: 遗留信道间隔 (Hz)。

    Returns
    -------
        (新 graph_json, 统计)。统计键固定为：
          "nodes_migrated"  实际发生字段新增的节点数
          "nodes_skipped"   已有 blocked_slots 而未改动的节点数
          "slots_written"   写入 blocked_slots 的槽总数
          "legacy_channels_read" 从旧字段读到的信道索引总数

    Raises
    ------
        ValueError: channel_spacing_hz 不是 slot_width_hz 的整数倍；
                    新旧字段同时存在且展开结果不一致；
                    展开后出现越界槽索引。
    """
    per_channel = slots_per_channel(channel_spacing_hz, slot_width_hz)
    nodes = graph_json.get("nodes", []) or []
    num_slots = _declared_num_slots(graph_json, nodes)

    stats = {
        "nodes_migrated": 0,
        "nodes_skipped": 0,
        "slots_written": 0,
        "legacy_channels_read": 0,
    }

    migrated: dict[str, Any] = {
        **graph_json,
        "nodes": [dict(node) for node in nodes],
    }

    for index, node in enumerate(migrated["nodes"]):
        roadm = node.get("roadm")
        if not isinstance(roadm, dict):
            continue
        has_slots = "blocked_slots" in roadm
        legacy = list(roadm.get("blocked_channels", []) or [])
        if has_slots and not legacy:
            stats["nodes_skipped"] += 1
            continue
        if not legacy and not has_slots:
            continue

        expanded = expand_blocked_channels(legacy, per_channel)
        if has_slots:
            existing = [int(s) for s in roadm["blocked_slots"]]
            if sorted(set(existing)) != sorted(set(expanded)):
                raise ValueError(
                    f"node {index}: blocked_slots {sorted(set(existing))} conflicts with "
                    f"blocked_channels {legacy} which expand to {sorted(set(expanded))}."
                )
            stats["nodes_skipped"] += 1
            continue

        for slot in expanded:
            if num_slots and not (0 <= slot < num_slots):
                raise ValueError(
                    f"node {index}: legacy blocked_channel {legacy} expands to out-of-bounds slot {slot} "
                    f"for {num_slots} slots."
                )

        node["roadm"] = {**roadm, "blocked_slots": sorted(set(expanded))}
        stats["nodes_migrated"] += 1
        stats["slots_written"] += len(set(expanded))
        stats["legacy_channels_read"] += len(legacy)

    return migrated, stats
