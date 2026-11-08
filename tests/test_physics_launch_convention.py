"""波形评估器与比较器的入纤功率口径（任务 O2）。

波形引擎内部坐标永远是双偏振总功率。``PyNLOQoTEvaluator`` 只声明输入口径，
``create_wdm_signal`` 先归一化到总功率再做原有的 ``/n_pol``；``per_pol`` 输入
不会被二次除 2。``TopologyQoTComparator`` 把同一入参分别换算到统计侧与波形侧
各自的模型口径，默认 ``total`` 时两次换算都返回原对象。
"""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics import qot_checker as physics_qot
from network_nlo_eval.physics.comparison import TopologyQoTComparator


def _topology() -> NetworkTopology:
    """两节点单链路拓扑。"""
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, weight=80.0)
    return NetworkTopology(graph, FiberSpanConfig(length_km=80.0), EDFAConfig(), ROADMConfig())


class _StatisticalRecorder:
    """记录收到的功率数组的统计侧替身。"""

    def __init__(self, convention: str = "total", snr_db: float = 20.0) -> None:
        self.launch_power_convention = convention
        self.symbol_rate_hz = 32e9
        self.snr_db = snr_db
        self.calls: list[np.ndarray] = []

    def evaluate_path(self, optical_path, power_in_w):
        self.calls.append(power_in_w)
        size = np.asarray(power_in_w).shape[0]
        return SimpleNamespace(snr_db=np.full(size, self.snr_db, dtype=np.float64))


class _WaveformRecorder:
    """记录收到的功率数组的波形侧替身。"""

    def __init__(self, convention: str = "total", snr_db: float = 18.0) -> None:
        self.launch_power_convention = convention
        self.snr_db = snr_db
        self.calls: list[np.ndarray] = []

    def evaluate_path(self, optical_path, power_in_w, num_symbols: int = 1024):
        self.calls.append(power_in_w)
        size = np.asarray(power_in_w).shape[0]
        return None, {"snr": np.full(size, self.snr_db, dtype=np.float64)}


# ---------------------------------------------------------------------------
# 波形评估器
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL", "auto"])
def test_waveform_evaluator_rejects_invalid_convention(convention: str) -> None:
    """口径校验发生在加载物理层之前。"""
    grid = SpectrumGrid(2, 50e9, 193.1e12)
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        physics_qot.PyNLOQoTEvaluator(grid, launch_power_convention=convention)


def test_waveform_per_pol_input_is_not_halved_twice() -> None:
    """``per_pol`` 的 ``P/2`` 与 ``total`` 的 ``P`` 生成逐位相同的波形。"""
    pytest.importorskip("physics_nlo_eval")
    grid = SpectrumGrid(3, 2e9, 193.1e12)
    total = physics_qot.PyNLOQoTEvaluator(grid, symbol_rate_hz=1e9, sps=2, seed=3)
    per_pol = physics_qot.PyNLOQoTEvaluator(
        grid, symbol_rate_hz=1e9, sps=2, seed=3, launch_power_convention="per_pol"
    )
    profile = np.array([0.0, 1e-3, 0.0])
    total_signal, total_bits = total.create_wdm_signal(profile, num_symbols=32)
    per_pol_signal, per_pol_bits = per_pol.create_wdm_signal(profile * 0.5, num_symbols=32)
    assert np.array_equal(np.asarray(total_signal.data), np.asarray(per_pol_signal.data))
    assert all(np.array_equal(a, b) for a, b in zip(total_bits, per_pol_bits, strict=True))
    assert total.launch_power_convention == "total"
    assert per_pol.launch_power_convention == "per_pol"


def test_waveform_default_total_keeps_the_explicit_total_waveform() -> None:
    """默认构造与显式 ``total`` 的波形逐位相同。"""
    pytest.importorskip("physics_nlo_eval")
    grid = SpectrumGrid(3, 2e9, 193.1e12)
    default = physics_qot.PyNLOQoTEvaluator(grid, symbol_rate_hz=1e9, sps=2, seed=3)
    explicit = physics_qot.PyNLOQoTEvaluator(
        grid, symbol_rate_hz=1e9, sps=2, seed=3, launch_power_convention="total"
    )
    profile = np.array([0.0, 1e-3, 0.0])
    a, _ = default.create_wdm_signal(profile, num_symbols=32)
    b, _ = explicit.create_wdm_signal(profile, num_symbols=32)
    assert np.array_equal(np.asarray(a.data), np.asarray(b.data))


# ---------------------------------------------------------------------------
# 比较器：统计侧与波形侧各自坐标
# ---------------------------------------------------------------------------


def test_comparator_default_total_passes_the_same_array_object() -> None:
    """默认 total 三边一致：没有换算、没有复制。"""
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    statistical = _StatisticalRecorder()
    waveform = _WaveformRecorder()
    comparator = TopologyQoTComparator(_topology(), grid, statistical, waveform)
    profile = np.full(3, 1e-3)
    comparator.evaluate([0, 1], profile, run_waveform=True)
    assert comparator.launch_power_convention == "total"
    assert statistical.calls[0] is profile
    assert waveform.calls[0] is profile


def test_comparator_converts_each_side_to_its_own_convention() -> None:
    """统计侧 per_pol、波形侧 total：同一入参分别换算。"""
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    statistical = _StatisticalRecorder(convention="per_pol")
    waveform = _WaveformRecorder(convention="total")
    comparator = TopologyQoTComparator(_topology(), grid, statistical, waveform)
    profile = np.full(3, 1e-3)
    comparator.evaluate([0, 1], profile, run_waveform=True)
    assert np.array_equal(statistical.calls[0], profile * 0.5)
    assert waveform.calls[0] is profile


def test_comparator_per_pol_input_expands_for_total_engines() -> None:
    """入参声明 per_pol 时，两个 total 引擎各自收到物理总功率。"""
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    statistical = _StatisticalRecorder(convention="total")
    waveform = _WaveformRecorder(convention="total")
    comparator = TopologyQoTComparator(
        _topology(), grid, statistical, waveform, launch_power_convention="per_pol"
    )
    profile = np.full(3, 0.5e-3)
    comparator.evaluate([0, 1], profile, run_waveform=True)
    assert np.array_equal(statistical.calls[0], np.full(3, 1e-3))
    assert np.array_equal(waveform.calls[0], np.full(3, 1e-3))


def test_comparator_lazily_creates_a_total_waveform_evaluator(monkeypatch) -> None:
    """惰性创建的波形评估器显式声明 total，不跟随比较器入参。"""
    created: list[dict] = []

    class _FakeEvaluator:
        def __init__(self, *args, **kwargs) -> None:
            created.append(kwargs)
            self.launch_power_convention = kwargs.get("launch_power_convention", "total")
            self.calls: list[np.ndarray] = []

        def evaluate_path(self, optical_path, power_in_w, num_symbols: int = 1024):
            self.calls.append(power_in_w)
            size = np.asarray(power_in_w).shape[0]
            return None, {"snr": np.full(size, 18.0)}

    monkeypatch.setattr(physics_qot, "PyNLOQoTEvaluator", _FakeEvaluator)
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    statistical = _StatisticalRecorder(convention="total")
    comparator = TopologyQoTComparator(
        _topology(), grid, statistical, launch_power_convention="per_pol"
    )
    profile = np.full(3, 0.5e-3)
    comparator.evaluate([0, 1], profile, run_waveform=True)
    assert created == [{"symbol_rate_hz": 32e9, "launch_power_convention": "total"}]
    assert comparator.waveform_evaluator is not None
    assert np.array_equal(comparator.waveform_evaluator.calls[0], np.full(3, 1e-3))


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL", "auto"])
def test_comparator_rejects_invalid_convention(convention: str) -> None:
    """比较器构造期拒绝非法口径。"""
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        TopologyQoTComparator(
            _topology(),
            SpectrumGrid(3, 50e9, 193.1e12),
            _StatisticalRecorder(),
            launch_power_convention=convention,
        )
