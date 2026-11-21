"""Receiver budgets for opt-in, identity-aware PCFM2 span pairs."""

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.transceiver import TransceiverMode
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.pcfm2 import PCFM2Model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest


def _models(
    *, attenuation: float = 0.2
) -> tuple[FiberSpanConfig, SpectrumGrid, PCFM2Model]:
    fiber = FiberSpanConfig(
        length_km=40,
        max_span_length_km=40,
        dispersion_parameter_d=0,
        dispersion_slope_s=0,
        attenuation_db_km_ref=attenuation,
    )
    grid = SpectrumGrid(2, 32e9, 193.1e12)
    model = PCFM2Model(
        grid, fiber, 32e9, include_isrs=False, island_cells=1, band_points=3
    )
    return fiber, grid, model


@pytest.mark.parametrize("interferer", ["continuing", "replacement", "departing"])
@pytest.mark.parametrize("flexible", [False, True])
def test_continuing_interactions_use_service_identity_and_actual_transfer(
    interferer: str, flexible: bool
) -> None:
    fiber, grid, model = _models()
    first_amp = EDFAConfig(max_output_signal_power_w=1.2e-3)
    second_amp = EDFAConfig(target_gain_db=7)
    roadm = ROADMConfig(
        insertion_loss_db=2,
        add_drop_loss_db=0,
        filtering_penalty_db=1,
        equalize_output_power=False,
    )
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=80, edfa=first_amp.model_dump())
    graph.add_edge(1, 2, length_km=40, edfa=second_amp.model_dump())
    for endpoint in (0, 2):
        graph.nodes[endpoint]["roadm"] = {"insertion_loss_db": 0}
    topology = NetworkTopology(graph, fiber, first_amp, roadm)
    state = NetworkState(topology, grid, slots_per_channel=4)
    through = interferer == "continuing"
    end = 2 if through else 1
    state.allocate_service(
        AllocatedService(
            0, 0, end, 0, 1, 100, 0, 0.6e-3, path=list(range(end + 1)), wavelength=0
        )
    )
    if interferer == "replacement":
        state.allocate_service(
            AllocatedService(1, 1, 2, 0, 1, 100, 0, 0.6e-3, path=[1, 2], wavelength=0)
        )
    path = [0, 1, 2]
    request = ServiceRequest(2, 0, 2, 0, 1, 100, 0, 1e-3, n_slots=4)
    switch_transfer = 10 ** (-roadm.path_loss_db("express") / 10)
    penalty = 10 ** (roadm.filtering_penalty_db / 10)
    first = model.evaluate_path(
        MultiSpanOpticalPath([fiber] * 2, [first_amp] * 2), np.array([0.6e-3, 1e-3])
    )
    next_interferer = {
        "continuing": first.signal_power_w[0] * switch_transfer,
        "replacement": 0.6e-3 * 10 ** (-roadm.path_loss_db("add") / 10),
        "departing": 0,
    }[interferer]
    second = model.evaluate_path(
        MultiSpanOpticalPath([fiber], [second_amp]),
        np.array([next_interferer, first.signal_power_w[1] * switch_transfer]),
    )
    signal = second.signal_power_w[1]
    first_scale = signal / first.signal_power_w[1] * penalty**2
    ase = first.ase_noise_w[1] * first_scale + second.ase_noise_w[1] * penalty
    first_sci = first.spm_noise_w[1] * first_scale
    second_sci = second.spm_noise_w[1] * penalty
    first_xci = first.xpm_noise_w[1] * first_scale
    second_xci = second.xpm_noise_w[1] * penalty
    incoherent = first_sci + second_sci + first_xci + second_xci + ase
    coherent = incoherent + 2 * np.sqrt(first_sci * second_sci)
    if through:
        coherent += 2 * np.sqrt(first_xci * second_xci)
        direct = model.evaluate_path(
            topology.build_optical_path(path), np.array([0.6e-3, 1e-3])
        )
        assert coherent == pytest.approx(
            direct.spm_noise_w[1] + direct.xpm_noise_w[1] + direct.ase_noise_w[1],
            rel=1e-12,
        )
    for enabled in (False, True):
        noise = coherent if enabled or through else incoherent
        validator = QoTValidator(model, topology, grid, mixed_load_coherence=enabled)
        accepted, snrs = validator.verify_allocation(
            request, path, 1, state, first_slot=4 if flexible else None
        )
        assert accepted
        assert snrs is not None
        assert snrs[1] == pytest.approx(10 * np.log10(signal / noise), abs=1e-10)


def test_mixed_load_coherence_rejects_incumbent_atomically() -> None:
    fiber, grid, model = _models(attenuation=1e-15)
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)], length_km=40)
    topology = NetworkTopology(
        graph, fiber, EDFAConfig(), ROADMConfig(insertion_loss_db=0, add_drop_loss_db=0)
    )
    state = NetworkState(topology, grid)
    one = model.evaluate_path(
        topology.build_optical_path([0, 1]), np.array([1e-3, 1e-3])
    )
    two = model.evaluate_path(topology.build_optical_path([1, 2]), np.array([1e-3, 0]))
    noise = sum(
        budget.spm_noise_w[0] + budget.xpm_noise_w[0] + budget.ase_noise_w[0]
        for budget in (one, two)
    )
    threshold = 10 * np.log10(1e-3 / noise) - 0.5
    mode = TransceiverMode(name="32 GBd", symbol_rate_hz=32e9, min_gsnr_db=threshold)
    state.allocate_service(
        AllocatedService(
            0,
            0,
            2,
            0,
            1,
            100,
            0,
            1e-3,
            path=[0, 1, 2],
            wavelength=0,
            transceiver_mode=mode,
        )
    )
    request = ServiceRequest(1, 0, 1, 0, 1, 100, 0, 1e-3)
    disabled = QoTValidator(model, topology, grid)
    enabled = QoTValidator(model, topology, grid, mixed_load_coherence=True)
    assert disabled.verify_allocation(request, [0, 1], 1, state)[0]
    assert enabled.verify_allocation(request, [0, 1], 1, state) == (False, None)
    assert not enabled.last_receiver_feasibility[0].feasible
    assert state.get_allocated_service(1) is None
    assert all(
        not link.occupied_channels[1] and link.noise_cache is None
        for link in state.get_all_link_states().values()
    )


def test_joining_carrier_keeps_existing_dispersive_span_pairs() -> None:
    fiber, _, _ = _models()
    forward = fiber.model_copy(update={"dispersion_parameter_d": 4.0})
    reverse = forward.model_copy(update={"dispersion_parameter_d": -2.0})
    grid = SpectrumGrid(3, 32e9, 193.1e12)
    model = PCFM2Model(
        grid, forward, 32e9, include_isrs=False, island_cells=1, band_points=3
    )
    graph: nx.DiGraph[int] = nx.DiGraph()
    graph.add_edge(0, 1, length_km=40)
    graph.add_edge(1, 2, length_km=40, fiber=reverse.model_dump())
    topology = NetworkTopology(
        graph,
        forward,
        EDFAConfig(),
        ROADMConfig(insertion_loss_db=0, add_drop_loss_db=0),
    )
    state = NetworkState(topology, grid)
    state.allocate_service(
        AllocatedService(0, 0, 2, 0, 1, 100, 0, 1e-3, path=[0, 1, 2], wavelength=0)
    )
    state.allocate_service(
        AllocatedService(1, 1, 2, 0, 1, 100, 0, 1e-3, path=[1, 2], wavelength=1)
    )

    path = [0, 1, 2]
    shared = np.array([1e-3, 0, 1e-3])
    full_route = model.evaluate_path(topology.build_optical_path(path), shared)
    last_link = topology.build_optical_path([1, 2])
    with_join = model.evaluate_path(last_link, np.full(3, 1e-3))
    without_join = model.evaluate_path(last_link, shared)
    new_islands = with_join.xpm_noise_w[2] - without_join.xpm_noise_w[2]
    assert new_islands > 0
    noise = (
        full_route.spm_noise_w[2]
        + full_route.xpm_noise_w[2]
        + full_route.ase_noise_w[2]
        + new_islands
    )
    validator = QoTValidator(model, topology, grid, mixed_load_coherence=True)
    request = ServiceRequest(2, 0, 2, 0, 1, 100, 0, 1e-3)
    accepted, snrs = validator.verify_allocation(request, path, 2, state, first_slot=2)
    assert accepted
    assert snrs is not None
    assert snrs[2] == pytest.approx(10 * np.log10(full_route.signal_power_w[2] / noise))


def test_mixed_load_coherence_requires_coherent_pcfm2() -> None:
    fiber, grid, _ = _models()
    topology = NetworkTopology(nx.path_graph(2), fiber, EDFAConfig(), ROADMConfig())
    incoherent = PCFM2Model(grid, fiber, 32e9, span_coherence="incoherent")
    ecz = create_model("ecz", grid, fiber, 32e9)
    for model in (incoherent, ecz):
        with pytest.raises(ValueError, match="requires the 'pcfm2' QoT model"):
            QoTValidator(model, topology, grid, mixed_load_coherence=True)
