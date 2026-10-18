"""业务模型与槽数推导的行为测试。"""

from __future__ import annotations

from dataclasses import asdict

import networkx as nx
import pytest

from network_nlo_eval.core.flexgrid import slots_for_rate
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.simulation.traffic import (
    AllocatedService,
    ServiceRequest,
    generate_services,
    slots_for_bit_rate,
)

SLOT_WIDTH_HZ = 12.5e9
SLOTS_PER_CHANNEL = 4


def _request(**overrides: object) -> ServiceRequest:
    fields: dict[str, object] = {
        "service_id": 0,
        "source_id": 0,
        "destination_id": 3,
        "arrival_time": 0.0,
        "departure_time": 1.0,
        "bit_rate_gbps": 100.0,
        "snr_requirement_db": 10.0,
        "launch_power_w": 1e-3,
    }
    fields.update(overrides)
    return ServiceRequest(**fields)  # type: ignore[arg-type]


def _path_cache() -> PathCache:
    graph = nx.Graph()
    for node in range(4):
        graph.add_node(node)
    for u in range(3):
        graph.add_edge(u, u + 1, weight=80.0)
    topology = NetworkTopology(graph, FiberSpanConfig(length_km=80.0), EDFAConfig(), ROADMConfig())
    return PathCache(topology)


def test_service_request_defaults_to_one_slot() -> None:
    """槽需求默认为 1，即遗留单信道语义。"""
    assert _request().n_slots == 1


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_service_request_rejects_invalid_n_slots(bad: object) -> None:
    """槽需求必须是正整数且非 bool。"""
    with pytest.raises(ValueError, match="n_slots"):
        _request(n_slots=bad)


def test_slots_for_bit_rate_matches_flexgrid_formula() -> None:
    """业务层推导与 flexgrid 的槽数公式同口径。"""
    for bit_rate, modulation, bits in ((100.0, "QPSK", 2), (400.0, "8QAM", 3), (600.0, "64QAM", 6)):
        assert slots_for_bit_rate(bit_rate, slot_width_hz=SLOT_WIDTH_HZ, modulation=modulation) == slots_for_rate(
            bit_rate * 1e9, bits, 0.0, SLOT_WIDTH_HZ
        )
    # 200 Gbps / QPSK / 12.5 GHz 槽 = 200/2/12.5 = 8 槽
    assert slots_for_bit_rate(200.0, slot_width_hz=SLOT_WIDTH_HZ, modulation="QPSK") == 8


def test_allocated_service_enforces_wavelength_consistency() -> None:
    """Wavelength 必须等于 first_slot // slots_per_channel。"""
    fields = asdict(_request(n_slots=2))  # slots=True 的 dataclass 没有 __dict__

    with pytest.raises(ValueError, match="inconsistent"):
        AllocatedService(**fields, path=[0, 1, 3], wavelength=2, first_slot=5, slots_per_channel=SLOTS_PER_CHANNEL)

    allocated = AllocatedService(
        **fields, path=[0, 1, 3], wavelength=1, first_slot=5, slots_per_channel=SLOTS_PER_CHANNEL
    )
    assert allocated.slot_interval == (5, 7)


def test_allocated_service_without_slots_per_channel_skips_consistency_check() -> None:
    """slots_per_channel=None 表示遗留信道粒度，不做槽区间一致性校验。"""
    fields = asdict(_request())
    legacy = AllocatedService(**fields, path=[0, 1, 3], wavelength=7, first_slot=28)
    assert legacy.wavelength == 7
    assert legacy.slots_per_channel is None


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_allocated_service_rejects_invalid_slots_per_channel(bad: object) -> None:
    """槽/信道折算率必须是正整数。"""
    with pytest.raises(ValueError, match="slots_per_channel"):
        AllocatedService(**asdict(_request()), path=[0, 1], slots_per_channel=bad)


def test_generate_services_derives_slots_without_disturbing_random_stream() -> None:
    """给槽宽时按比特率推导 n_slots，且不改变到达时间/比特率序列。"""
    path_cache = _path_cache()
    legacy = generate_services(path_cache, 20, 1.0, 10.0, seed=42)
    flexible = generate_services(path_cache, 20, 1.0, 10.0, seed=42, slot_width_hz=SLOT_WIDTH_HZ)

    assert all(service.n_slots == 1 for service in legacy)
    for old, new in zip(legacy, flexible, strict=True):
        assert new.n_slots == slots_for_bit_rate(new.bit_rate_gbps, slot_width_hz=SLOT_WIDTH_HZ)
        assert new.arrival_time == old.arrival_time
        assert new.bit_rate_gbps == old.bit_rate_gbps
    assert any(service.n_slots > 1 for service in flexible)
