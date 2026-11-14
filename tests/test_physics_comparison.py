"""Tests for topology-based statistical and waveform QoT comparison."""

import networkx as nx
import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics.comparison import TopologyQoTComparator


class WaveformStub:
    """Record input power while providing deterministic waveform SNR."""

    launch_power_convention = "total"

    def evaluate_path(self, path, power, *, num_symbols):
        self.power = power.copy()
        self.num_symbols = num_symbols
        self.path = path
        return None, {"snr": np.array([10.0, np.inf])}


def test_comparator_converts_each_model_input_and_serializes_nonfinite_values() -> None:
    graph = nx.Graph([(10, 20)])
    fiber = FiberSpanConfig(length_km=80)
    topology = NetworkTopology(graph, fiber, EDFAConfig(), ROADMConfig())
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    statistical = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention="per_pol")
    waveform = WaveformStub()
    comparator = TopologyQoTComparator(
        topology, grid, statistical, waveform, launch_power_convention="total"
    )
    total_power = np.array([1e-3, 0])
    result = comparator.evaluate(
        [0, 1], total_power, run_waveform=True, num_symbols=128
    )
    independent = statistical.evaluate_path(
        topology.build_optical_path([0, 1]), total_power * 0.5
    )
    np.testing.assert_array_equal(result.statistical.snr_db, independent.snr_db)
    np.testing.assert_array_equal(waveform.power, total_power)
    assert waveform.num_symbols == 128
    assert waveform.path.node_path == [0, 1]
    assert result.snr_delta_db[0] == 10 - independent.snr_db[0]
    exported = result.to_dict()
    assert exported["statistical"]["snr_db"][1] is None
    assert exported["waveform"]["snr"][1] is None
