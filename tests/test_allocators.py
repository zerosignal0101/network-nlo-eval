"""First-fit routing and physical QoT integration."""

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.allocators import (
    KSPFirstFitAllocator,
    KSPFirstFitFlexAllocator,
    first_fit_contiguous_windows,
)
from network_nlo_eval.rwa.path_computation import PathCache
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import ServiceRequest


def test_window_candidates_are_free_and_deduplicated_by_anchor() -> None:
    assert first_fit_contiguous_windows(
        np.array([1, 1, 1, 1, 0, 1, 1, 1], dtype=bool), 2, 2
    ) == [0, 2, 5, 6]
    assert first_fit_contiguous_windows(np.ones(2, dtype=bool), 3, 2) == []
    with pytest.raises(ValueError, match="n_slots"):
        first_fit_contiguous_windows(np.ones(2, dtype=bool), 0, 2)


@pytest.mark.parametrize("flex", [False, True])
def test_allocator_uses_real_qot_and_releases_resources(flex: bool) -> None:
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(nx.Graph([(0, 1)]), fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    state = NetworkState(topology, grid, slots_per_channel=4 if flex else 1)
    validator = QoTValidator(
        MultiBandISRSGN(grid, fiber, 32e9, ode_accel="off"), topology, grid
    )
    paths = PathCache(topology)
    allocator = (
        KSPFirstFitFlexAllocator(paths, validator, grid, topology, 4)
        if flex
        else KSPFirstFitAllocator(paths, validator, grid)
    )
    rejected = ServiceRequest(0, 0, 1, 0, 1, 100, 1000, 0.001)
    assert allocator.allocate(rejected, state) == (False, None)
    assert state.get_link_state(0, 1).noise_cache is None
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 0.001)
    success, service = allocator.allocate(request, state)
    assert success
    assert service is not None
    assert service.wavelength == 0
    assert state.get_link_state(0, 1).noise_cache is not None
    state.allocate_service(service)
    assert state.get_link_state(0, 1).occupied_channels[0]
    allocator.release(service, state)
    assert not state.get_link_state(0, 1).occupied_channels.any()
