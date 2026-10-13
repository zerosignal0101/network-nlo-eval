"""两棵树共用的 legacy 场景跑法。

与上游 `fu4_scenarios.run_scenario` 的差别只有一处（实测得到的签名差异）：

| 参数 | wp/fg-uwb-fid | ecz-dev |
|---|---|---|
| `symbol_rate_hz` | 有 | 有 |
| `connections` | 有 | **无** |
| `slots_per_channel` | 有 | **无** |

本模块用 `inspect` 探测后按需传参，保证物理输入（grid / fiber / symbol_rate）
在两棵树上尽量一致，差异只来自被测的 SNR 装配逻辑本身。

上游 `fu4_scenarios.py` 的 flexgrid import 已剥离（ecz-dev 无 core.flexgrid），
且只跑 `kind == "legacy"` 的场景。
"""
from __future__ import annotations

import inspect
import math

import numpy as np

import fu4_scenarios_legacy as fs
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.simulation.traffic import ServiceRequest
from network_nlo_eval.rwa.qot_checker import QoTValidator

SCENARIOS = [s for s in fs.SCENARIOS if s.kind == "legacy"]
LAUNCH_POWER_W = fs.LAUNCH_POWER_W
SYMBOL_RATE_HZ = fs.SYMBOL_RATE_HZ


def run_scenario(scenario):
    topology = fs._line_topology(max(scenario.path) + 1, scenario)
    grid, connections, state_grid, num_channels = fs._grid_and_connections(scenario)
    if scenario.channel_idx >= num_channels:
        raise ValueError(
            "%s: channel %d >= %d" % (scenario.name, scenario.channel_idx, num_channels)
        )

    params = set(inspect.signature(MultiBandISRSGN.__init__).parameters)
    kwargs = {}
    if "symbol_rate_hz" in params:
        kwargs["symbol_rate_hz"] = SYMBOL_RATE_HZ
    if "phi_kurtosis" in params:
        # fu4 树用逐连接 Rs，phi_kurtosis 不在 Scenario 上；ecz-dev 才有该字段。
        pk = getattr(scenario, "phi_kurtosis", None)
        if pk is not None:
            kwargs["phi_kurtosis"] = pk
    if "connections" in params and connections is not None:
        kwargs["connections"] = connections
        if "slots_per_channel" in params:
            kwargs["slots_per_channel"] = scenario.slots_per_channel

    evaluator = MultiBandISRSGN(grid, topology.get_default_fiber_config(), **kwargs)
    state = NetworkState(topology, state_grid)
    validator = QoTValidator(evaluator, topology, grid)

    # 全零发射谱会让 XPM 项消失，掩盖只出现在非线性项里的装配差异。
    for link_state in state.get_all_link_states().values():
        link_state.launch_power_profile_w[:] = LAUNCH_POWER_W

    request = ServiceRequest(
        service_id=0,
        source_id=scenario.path[0],
        destination_id=scenario.path[-1],
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=-1e9,
        launch_power_w=LAUNCH_POWER_W,
    )

    accepted, snrs_db = validator.verify_allocation(
        request, list(scenario.path), scenario.channel_idx, state
    )
    if snrs_db is None:
        return {
            "scenario": scenario.to_dict(),
            "accepted": False,
            "snr_db": None,
            "init_params": sorted(params),
        }

    value = float(np.asarray(snrs_db, dtype=np.float64)[scenario.channel_idx])
    if math.isinf(value) and value < 0:
        value = "-inf"
    elif math.isinf(value):
        value = "inf"
    return {
        "scenario": scenario.to_dict(),
        "accepted": bool(accepted),
        "snr_db": value,
        "init_params": sorted(params),
    }