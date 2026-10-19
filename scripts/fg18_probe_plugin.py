"""FG-18 可达性探针的 pytest 插件（被 ``fg18_reachability_probe.py`` 以 ``-p`` 加载）。

只包一层转发、不改被测行为；计数落到 ``/tmp/fg18_roadm_probe_counts.json``。
"""

from __future__ import annotations

import json
import pathlib

COUNTER_PATH = pathlib.Path("/tmp/fg18_roadm_probe_counts.json")

_METHODS = ("_calc_span_noise_and_power", "calc_span_noise_and_power_batch")


def pytest_configure(config) -> None:  # noqa: ANN001, ANN201
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN

    counts = {m: {"calls": 0, "roadm_not_none": 0} for m in _METHODS}

    def make(name: str, orig):  # noqa: ANN001, ANN202
        def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN202
            # 两个方法的第 4 个位置形参都是 roadm_config。
            roadm = args[3] if len(args) > 3 else kwargs.get("roadm_config")
            counts[name]["calls"] += 1
            if roadm is not None:
                counts[name]["roadm_not_none"] += 1
            # 每次调用都落盘：早先只在非 None 时写盘，导致"全部为 None"的套件
            # 读出 0/0，看起来像"方法从没被调用"——那是探针的假阴性，不是结论。
            COUNTER_PATH.write_text(json.dumps(counts, indent=2))
            return orig(self, *args, **kwargs)

        wrapper.__name__ = name
        wrapper.__doc__ = orig.__doc__
        return wrapper

    for m in _METHODS:
        setattr(MultiBandISRSGN, m, make(m, getattr(MultiBandISRSGN, m)))
    COUNTER_PATH.write_text(json.dumps(counts, indent=2))
