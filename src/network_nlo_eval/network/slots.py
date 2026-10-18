"""灵活栅格的槽级占用辅助函数。"""

from __future__ import annotations

__all__ = ["connection_overlaps_slots"]


def connection_overlaps_slots(first_slot: int, n_slots: int, blocked: set[int]) -> bool:
    """连接槽区间是否与任一被阻断槽重叠。

    切片语义：连接 [first_slot, first_slot + n_slots) 只要命中一个阻断槽即不可用。

    Args:
        first_slot: 起始槽 (0-based, 闭)。
        n_slots: 槽数，>= 1。
        blocked: 被阻断的槽集合。

    Returns
    -------
        有重叠为 True。
    """
    if not blocked:
        return False
    window = range(first_slot, first_slot + n_slots)
    return any(slot in blocked for slot in window)
