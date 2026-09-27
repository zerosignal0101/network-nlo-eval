"""Topology-to-physical-layer integration regression tests."""

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics.comparison import TopologyQoTComparator


def _topology(*, roadm_loss_db: float = 5.0) -> NetworkTopology:
    graph = nx.Graph()
    graph.add_node(100)
    graph.add_node(
        200,
        roadm={
            "insertion_loss_db": roadm_loss_db,
            "passband_hz": 50e9,
            "filter_order": 6,
            "booster_noise_figure_db": 5.0,
        },
    )
    graph.add_edge(
        100,
        200,
        weight=160.0,
        fiber={"max_span_length_km": 80.0, "attenuation_db_km_ref": 0.2},
        edfa={"target_gain_db": None, "noise_figure_db": 5.0},
    )
    return NetworkTopology(
        graph,
        FiberSpanConfig(length_km=100.0),
        EDFAConfig(),
        ROADMConfig(),
    )


def test_topology_overrides_build_ordered_physical_path() -> None:
    """Link and node overrides survive into the ordered physical path."""
    topology = _topology()
    path = topology.build_optical_path([topology.get_internal_node_id(100), topology.get_internal_node_id(200)])
    assert path.fiber_configs[0].length_km == 160
    assert path.fiber_configs[0].max_span_length_km == 80
    assert path.roadm_configs is not None
    assert path.roadm_configs[0].filter_order == 6


def test_exact_multiple_link_applies_spans_and_roadm() -> None:
    """A 160 km link split into 80 km spans yields two spans plus one ROADM."""
    topology = _topology()
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    evaluator = MultiBandISRSGN(grid, topology.get_default_fiber_config(), symbol_rate_hz=32e9)
    path = topology.build_optical_path([0, 1])
    result = evaluator.evaluate_path(path, np.full(3, 1e-3))
    assert result.span_count == 2
    assert result.roadm_count == 1
    assert np.all(result.ase_noise_w > 0)
    assert np.all(np.isfinite(result.snr_db))


def test_comparator_statistical_result_is_json_compatible() -> None:
    """Statistical-only comparison serializes without waveform metrics."""
    topology = _topology()
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    result = TopologyQoTComparator(topology, grid).evaluate([0, 1], np.full(3, 1e-3))
    serialized = result.to_dict()
    assert serialized["waveform"] is None
    assert serialized["statistical"]["roadm_count"] == 1


def test_spectrum_grid_rejects_invalid_physical_frequency() -> None:
    """A grid whose lowest channel reaches DC is rejected at construction."""
    with pytest.raises(ValueError):
        SpectrumGrid(10, 50e9, 1.0)


def test_waveform_route_smoke() -> None:
    """The waveform engine propagates a route and drops blocked channels."""
    pytest.importorskip("physics_nlo_eval")
    from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
    from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator

    grid = SpectrumGrid(3, 2e9, 193.1e12)
    fiber = FiberSpanConfig(
        length_km=0.01,
        max_span_length_km=0.01,
        dispersion_parameter_d=0,
        dispersion_slope_s=0,
        nonlinear_index_n2=0,
    )
    path = MultiSpanOpticalPath(
        [fiber],
        [EDFAConfig()],
        [ROADMConfig(insertion_loss_db=1, add_drop_loss_db=0, passband_hz=2e9)],
    )
    evaluator = PyNLOQoTEvaluator(grid, symbol_rate_hz=1e9, sps=2, seed=3)
    signal, _ = evaluator.create_wdm_signal(np.array([0, 1e-3, 0]), num_symbols=64)
    output = evaluator.propagate_path(signal, path)
    assert output.data.shape == signal.data.shape
    assert output.ctx.meta["roadm_pass_channels"] == [1]
