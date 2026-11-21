"""QoT admission using exact carrier geometry and the load on each link."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import pairwise
from typing import TYPE_CHECKING, NamedTuple, cast

import networkx as nx
import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import CarrierGeometry, FlexibleGrid
from network_nlo_eval.core.transceiver import evaluate_receiver
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.power_conventions import launch_power_scale
from network_nlo_eval.models.route import H_PLANCK
from network_nlo_eval.rwa.validity import ChannelValidity, OpticalPathValidityProvider

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Literal

    from network_nlo_eval.core.types import LinkKey, NDArrayFloat
    from network_nlo_eval.models.base import QoTModel
    from network_nlo_eval.network.elements import EDFAConfig, ROADMConfig
    from network_nlo_eval.network.state import LinkState, NetworkState
    from network_nlo_eval.network.topology import NetworkTopology
    from network_nlo_eval.rwa.qot_checker import QoTValidator
    from network_nlo_eval.rwa.validity import LinkValidity, LinkValidityProvider
    from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

    CarrierModelFactory = Callable[[tuple[CarrierGeometry, ...], int], QoTModel]
    CarrierValidityFactory = Callable[[QoTModel], OpticalPathValidityProvider]

_DISPERSION_REFERENCE_NM = 1550.0


class _Carrier(NamedTuple):
    geometry: CarrierGeometry
    launch_power_w: float
    service_id: int | None


class _Switching(NamedTuple):
    transfer: float
    limit_w: float
    booster_ase_w: float
    noise_penalty: float


_PASS_THROUGH = _Switching(1.0, math.inf, 0.0, 1.0)


def _switching(
    roadm: ROADMConfig,
    operation: Literal["add", "express", "drop"],
    polarization_scale: float,
    frequency: float,
    bandwidth: float,
    *,
    blocked: bool,
) -> _Switching:
    loss = 10 ** (roadm.path_loss_db(operation) / 10)
    penalty = 10 ** (roadm.filtering_penalty_db / 10)
    if blocked:
        return _Switching(0.0, math.inf, 0.0, penalty)
    target = roadm.output_target
    if target is not None:
        limit = target.power_w
        if limit is None:
            assert target.psd_w_per_hz is not None
            limit = target.psd_w_per_hz * bandwidth
        return _Switching(1.0 / loss, limit * polarization_scale, 0.0, penalty)
    if roadm.equalize_output_power:
        nf = 10 ** (roadm.booster_noise_figure_db / 10)
        ase = nf * H_PLANCK * frequency * bandwidth * (loss - 1)
        return _Switching(1.0, math.inf, ase * polarization_scale, penalty)
    return _Switching(1.0 / loss, math.inf, 0.0, penalty)


@dataclass(slots=True)
class _LinkLoad:
    model: QoTModel
    optical_path: MultiSpanOpticalPath
    rows: dict[int, int]
    source_w: NDArrayFloat
    source_ase_w: NDArrayFloat
    width_scale: NDArrayFloat
    upstream: list[tuple[int, int, LinkKey]]
    roadm_transfer: NDArrayFloat
    roadm_limits_w: NDArrayFloat
    roadm_ase_w: NDArrayFloat
    roadm_penalty: NDArrayFloat
    provider: LinkValidityProvider | None
    bandwidths_hz: list[float]


@dataclass(slots=True)
class _LinkBudget:
    signal_w: NDArrayFloat
    noise_w: NDArrayFloat
    noise_transfer: NDArrayFloat
    validity: LinkValidity | None
    ase_w: NDArrayFloat | None


class _PathImpairments(NamedTuple):
    cd_at_reference_ps_nm: float
    cd_slope_ps_nm2: float
    pmd_ps: float
    pdl_db: float


def preserves_through_power(topology: NetworkTopology) -> bool:
    """Return whether network equipment preserves through-carrier power."""
    for node in topology.get_all_internal_nodes():
        roadm = topology.get_roadm_config(node)
        if (
            not roadm.equalize_output_power
            or roadm.output_target is not None
            or roadm.degree_targets
            or roadm.blocked_channels
            or roadm.blocked_slots
        ):
            return False
    for key in topology.get_all_internal_links():
        spans = topology.get_link_spans(*key)
        amplifiers = (
            (topology.get_link_edfa_config(*key),)
            if spans is None
            else (span.edfa for span in spans)
        )
        if any(edfa is None or not edfa.is_loss_compensating for edfa in amplifiers):
            return False
    return True


class CarrierAdmission:
    """Evaluate the component of the network coupled to a candidate's load.

    Spectra are compact and ordered by frequency on each link. Carrier IDs
    map through-traffic from one spectrum to the next, independently of slot
    reuse. Power dependencies are solved before receiver noise is assembled.
    """

    def __init__(
        self,
        validator: QoTValidator,
        model_factory: CarrierModelFactory | None,
        validity_factory: CarrierValidityFactory | None,
        *,
        independent_power: bool,
    ) -> None:
        self.validator = validator
        self.topology = validator.network_topology
        self.grid = validator.spectrum_grid
        self.nominal_frequencies = self.grid.frequencies
        self.independent_power = independent_power
        self.power_scale = launch_power_scale(
            validator.eval_power_convention
        ) / launch_power_scale(validator.launch_power_convention)

        def bind(
            geometry: tuple[CarrierGeometry, ...], slots_per_channel: int
        ) -> tuple[QoTModel, OpticalPathValidityProvider | None]:
            assert model_factory is not None
            model = model_factory(geometry, slots_per_channel)
            if (
                model.n_nodes != len(geometry)
                or not np.array_equal(
                    model.f_abs_hz, [carrier.centre_hz for carrier in geometry]
                )
                or getattr(model, "launch_power_convention", "total")
                != validator.eval_power_convention
            ):
                msg = "The carrier model must match the supplied geometry and power convention."
                raise ValueError(msg)
            provider = validity_factory(model) if validity_factory is not None else None
            if (
                getattr(
                    provider, "input_power_convention", validator.eval_power_convention
                )
                != validator.eval_power_convention
            ):
                msg = "The carrier validity provider must use the evaluator's input power convention."
                raise ValueError(msg)
            return model, provider

        self._models = lru_cache(maxsize=32)(bind)
        self._link_paths = lru_cache(maxsize=128)(self._build_link_path)

    def verify(
        self,
        request: ServiceRequest,
        path: list[int],
        channel_idx: int,
        first_slot: int | None,
        state: NetworkState,
        span_km: float | None,
        *,
        nominal_grid: bool = False,
    ) -> tuple[bool, NDArrayFloat | None]:
        per_channel = state.slots_per_channel
        slot = channel_idx * per_channel if first_slot is None else first_slot
        width = per_channel if first_slot is None else request.n_slots
        flex_grid = FlexibleGrid.from_legacy_grid(
            self.grid.num_channels,
            self.grid.channel_spacing_hz,
            self.grid.center_frequency_hz,
            per_channel,
        )
        if (
            slot < 0
            or slot + width > flex_grid.num_slots
            or slot // per_channel != channel_idx
            or request.required_bandwidth_hz > width * flex_grid.slot_width_hz
            or state.get_allocated_service(request.service_id) is not None
        ):
            return False, None

        route = tuple(self.topology.link_key(u, v) for u, v in pairwise(path))
        candidate_links = set(route)
        if len(route) != len(candidate_links) or any(
            state.get_link_state(*key).slot_occupancy[slot : slot + width].any()
            for key in route
        ):
            return False, None

        candidate = _Carrier(
            self._geometry(flex_grid, slot, width, request),
            request.launch_power_w,
            request.service_id,
        )
        paths = {request.service_id: (path, route)}
        incumbents: dict[int, AllocatedService] = {}
        with_ase = request.transceiver_mode is not None
        graph: nx.DiGraph[LinkKey] = nx.DiGraph()
        graph.add_nodes_from(route)
        if not self.independent_power:
            graph.add_edges_from(pairwise(route))
        loads: dict[LinkKey, _LinkLoad] = {}
        pending = deque(route)
        while pending:
            key = pending.popleft()
            carriers = self._link_carriers(
                state, state.get_link_state(*key), flex_grid, nominal_grid=nominal_grid
            )
            for carrier in carriers:
                service_id = carrier.service_id
                if service_id is None:
                    continue
                service = state.get_allocated_service(service_id)
                assert service is not None
                if service_id not in paths:
                    service_route = tuple(
                        self.topology.link_key(u, v) for u, v in pairwise(service.path)
                    )
                    paths[service_id] = (service.path, service_route)
                if service_id in incumbents or (
                    self.independent_power and key not in candidate_links
                ):
                    continue
                incumbents[service_id] = service
                with_ase = with_ase or service.transceiver_mode is not None
                _, service_route = paths[service_id]
                pending.extend(hop for hop in service_route if hop not in graph)
                graph.add_nodes_from(service_route)
                if not self.independent_power:
                    graph.add_edges_from(pairwise(service_route))
            if key in candidate_links:
                carriers.append(candidate)
            loads[key] = self._prepare_link(
                key, carriers, paths, span_km, per_channel, nominal_grid=nominal_grid
            )

        if (
            not nominal_grid
            and not self.validator.validity_policy.enabled
            and len(loads) == len(route)
            and all(service.path == path for service in incumbents.values())
            and all(
                load.model.n_nodes == len(incumbents) + 1 for load in loads.values()
            )
        ):
            first = loads[route[0]]
            snrs, signal, noise, ase = self._uniform_budget(
                path, route, loads, with_ase=with_ase
            )
            impairments = (
                self._path_impairments(path, route, loads) if with_ase else None
            )
            projected = np.full(self.grid.num_channels, -np.inf)
            for checked_service in (request, *incumbents.values()):
                row = first.rows[checked_service.service_id]
                snr = float(snrs[row])
                if checked_service.transceiver_mode is not None:
                    assert ase is not None
                    assert impairments is not None
                    if not self._receiver_feasible(
                        checked_service,
                        first,
                        impairments,
                        snr,
                        float(signal[row]),
                        float(noise[row]),
                        float(ase[row]),
                    ):
                        return False, None
                if not (snr > -math.inf and snr >= checked_service.snr_requirement_db):
                    return False, None
                if checked_service is not request:
                    anchor = incumbents[checked_service.service_id].wavelength
                    self._record_anchor(projected, anchor, snr)
            projected[channel_idx] = snrs[first.rows[request.service_id]]
            return True, projected

        budgets: dict[LinkKey, _LinkBudget] = {}
        try:
            ordering = list(nx.topological_sort(graph))
        except nx.NetworkXUnfeasible:
            condensation = nx.condensation(graph)
            for component in nx.topological_sort(condensation):
                members = sorted(condensation.nodes[component]["members"])
                if len(members) > 1 and not self._settle_component(
                    members, loads, budgets
                ):
                    return False, None
                for key in members:
                    budgets[key] = self._evaluate_link(
                        key, loads[key], loads, budgets, with_ase=with_ase
                    )
        else:
            for key in ordering:
                budgets[key] = self._evaluate_link(
                    key, loads[key], loads, budgets, with_ase=with_ase
                )

        candidate_snr = self._receiver_snr(request, path, route, loads, budgets)
        if not (
            candidate_snr > -math.inf and candidate_snr >= request.snr_requirement_db
        ):
            return False, None
        affected = set(route)
        if not self.independent_power and len(loads) > len(route):
            pending.extend(route)
            while pending:
                for successor in graph.successors(pending.popleft()):
                    if successor not in affected:
                        affected.add(successor)
                        pending.append(successor)

        snrs = np.full(self.grid.num_channels, -np.inf, dtype=np.float64)
        for service_id, service in incumbents.items():
            old_path, old_route = paths[service_id]
            if affected.isdisjoint(old_route):
                continue
            snr = self._receiver_snr(service, old_path, old_route, loads, budgets)
            if not (snr > -math.inf and snr >= service.snr_requirement_db):
                return False, None
            if old_route == route and service.path == path:
                self._record_anchor(snrs, service.wavelength, snr)
        snrs[channel_idx] = candidate_snr
        return True, snrs

    @staticmethod
    def _record_anchor(snrs: NDArrayFloat, anchor: int, snr: float) -> None:
        if snrs[anchor] == -math.inf or snr < snrs[anchor]:
            snrs[anchor] = snr

    def _uniform_budget(
        self,
        path: list[int],
        route: tuple[LinkKey, ...],
        loads: dict[LinkKey, _LinkLoad],
        *,
        with_ase: bool,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat | None]:
        """Use the model's full-route evaluation for identical loads and ports."""
        first = loads[route[0]]
        fibers = []
        amplifiers: list[EDFAConfig | None] = []
        roadms: list[ROADMConfig | None] = []
        penalty = 1.0
        for index, key in enumerate(route):
            link = loads[key].optical_path
            fibers.extend(link.fiber_configs)
            amplifiers.extend(link.edfa_configs)
            roadm = self.topology.get_roadm_path_config(path, index)
            roadms.extend([None] * (len(link.fiber_configs) - 1))
            roadms.append(roadm)
            penalty *= 10 ** (roadm.filtering_penalty_db / 10)
        # A uniform route can use the model's native inter-span coherence.
        result = first.model.evaluate_path(
            MultiSpanOpticalPath(
                fibers,
                amplifiers,
                roadms,
                [key for key in route for _ in loads[key].optical_path.fiber_configs],
                path,
            ),
            first.source_w / first.width_scale,
        )
        signal = result.signal_power_w
        noise = result.spm_noise_w + result.xpm_noise_w + result.ase_noise_w
        add_noise = np.divide(
            first.source_ase_w * signal * penalty,
            first.source_w,
            out=np.zeros_like(signal),
            where=first.source_w > 0,
        )
        noise += add_noise
        snrs = np.full_like(signal, -np.inf)
        noisy = (signal > 0) & (noise > 0)
        snrs[noisy] = 10 * np.log10(signal[noisy] / noise[noisy])
        snrs[(signal > 0) & (noise == 0)] = np.inf
        return snrs, signal, noise, result.ase_noise_w + add_noise if with_ase else None

    def _geometry(
        self,
        grid: FlexibleGrid,
        slot: int,
        width: int,
        request: ServiceRequest | None,
    ) -> CarrierGeometry:
        explicit = None if request is None else request.symbol_rate_hz
        if request is not None and request.transceiver_mode is not None:
            explicit = request.transceiver_mode.symbol_rate_hz
        baud = (
            min(self.validator.gn_evaluator.symbol_rate_hz, width * grid.slot_width_hz)
            if explicit is None
            else explicit
        )
        return CarrierGeometry(grid, slot, width, baud)

    def _link_carriers(
        self,
        state: NetworkState,
        link: LinkState,
        grid: FlexibleGrid,
        *,
        nominal_grid: bool,
    ) -> list[_Carrier]:
        carriers = []
        slot_anchors = set()
        width_per_channel = link.slots_per_channel
        for slot, (width, power, service_id) in (
            () if nominal_grid else link.slot_allocations()
        ):
            slot_anchors.add(slot // width_per_channel)
            service = state.get_allocated_service(service_id)
            carriers.append(
                _Carrier(
                    self._geometry(grid, slot, width, service),
                    power,
                    service_id if service is not None else None,
                )
            )
        active = link.occupied_channels & (
            (link.allocated_service_ids != -1) | (link.launch_power_profile_w > 0)
        )
        for row in np.flatnonzero(active).tolist():
            if row in slot_anchors:
                continue
            service_id = int(link.allocated_service_ids[row])
            service = (
                state.get_allocated_service(service_id) if service_id != -1 else None
            )
            if service is not None and service.slots_per_channel is not None:
                continue
            carriers.append(
                _Carrier(
                    self._geometry(
                        grid, row * width_per_channel, width_per_channel, service
                    ),
                    float(link.launch_power_profile_w[row]),
                    service_id if service is not None else None,
                )
            )
        return carriers

    def _prepare_link(
        self,
        key: LinkKey,
        carriers: list[_Carrier],
        paths: dict[int, tuple[list[int], tuple[LinkKey, ...]]],
        span_km: float | None,
        per_channel: int,
        *,
        nominal_grid: bool,
    ) -> _LinkLoad:
        provider: LinkValidityProvider | None
        optical_path = self._link_paths(key, span_km)
        if nominal_grid:
            model = self.validator.gn_evaluator
            provider = self.validator.validity_provider
        else:
            carriers.sort(key=lambda carrier: carrier.geometry.first_slot)
            geometry = tuple(carrier.geometry for carrier in carriers)
            model, provider = self._models(geometry, per_channel)
            optical_path = self._sample_amplifiers(optical_path, model)
        width_scale = model.power_width_scale()
        polar_scale = self.validator.ase_polarization_scale
        frequencies = model.f_abs_hz.tolist()
        bandwidths = model.channel_bandwidth_hz_channels.tolist()
        blocked_rows: dict[int, set[int]] = {}

        def switching_for(
            config: ROADMConfig, operation: Literal["add", "express", "drop"], row: int
        ) -> _Switching:
            blocked = False
            if config.blocked_slots or config.blocked_channels:
                config_key = id(config)
                if config_key not in blocked_rows:
                    blocked_rows[config_key] = set(model.blocked_nodes(config))
                blocked = row in blocked_rows[config_key]
            return _switching(
                config,
                operation,
                polar_scale,
                frequencies[row],
                bandwidths[row],
                blocked=blocked,
            )

        upstream = []
        sources = [0.0] * model.n_nodes
        source_noise = [0.0] * model.n_nodes
        switches = [_PASS_THROUGH] * model.n_nodes
        rows = {}
        for index, carrier in enumerate(carriers):
            row = carrier.geometry.first_slot // per_channel if nominal_grid else index
            power = carrier.launch_power_w * self.power_scale * width_scale[row]
            add_ase = 0.0
            switching = _PASS_THROUGH
            if (service_id := carrier.service_id) is not None:
                path, route = paths[service_id]
                hop = route.index(key)
                if hop == 0:
                    add = switching_for(
                        self.topology.get_roadm_add_config(path[0], path[1]),
                        "add",
                        row,
                    )
                    power = min(power * add.transfer, add.limit_w)
                    add_ase = (
                        add.booster_ase_w * add.noise_penalty if power > 0 else 0.0
                    )
                elif not self.independent_power:
                    upstream.append((row, service_id, route[hop - 1]))
                    power = 0.0
                switching = switching_for(
                    self.topology.get_roadm_path_config(path, hop),
                    "drop" if hop == len(route) - 1 else "express",
                    row,
                )
                rows[service_id] = row
            sources[row] = power
            source_noise[row] = add_ase
            switches[row] = switching
        transfer, limits, ase, penalty = zip(*switches, strict=True)
        return _LinkLoad(
            model,
            optical_path,
            rows,
            np.array(sources),
            np.array(source_noise),
            width_scale,
            upstream,
            np.array(transfer),
            np.array(limits),
            np.array(ase),
            np.array(penalty),
            provider,
            bandwidths,
        )

    def _build_link_path(
        self, key: LinkKey, span_km: float | None
    ) -> MultiSpanOpticalPath:
        optical_path = self.topology.build_optical_path(
            list(key), fixed_span_length_km=span_km
        )
        optical_path.roadm_configs = None
        return optical_path

    def _sample_amplifiers(
        self, path: MultiSpanOpticalPath, model: QoTModel
    ) -> MultiSpanOpticalPath:
        amplifiers = []
        for edfa in path.edfa_configs:
            updates = {}
            if edfa is not None:
                for name in ("noise_figure_db", "gain_ripple_db"):
                    values = getattr(edfa, name)
                    if isinstance(values, np.ndarray) and values.ndim > 0:
                        if values.shape != self.nominal_frequencies.shape:
                            msg = (
                                f"EDFA {name} must have one value per nominal channel."
                            )
                            raise ValueError(msg)
                        updates[name] = np.interp(
                            model.f_abs_hz, self.nominal_frequencies, values
                        )
            amplifiers.append(
                edfa.model_copy(update=updates)
                if edfa is not None and updates
                else edfa
            )
        if any(
            current is not original
            for current, original in zip(amplifiers, path.edfa_configs, strict=True)
        ):
            return replace(path, edfa_configs=amplifiers)
        return path

    def _evaluate_link(
        self,
        key: LinkKey,
        load: _LinkLoad,
        loads: dict[LinkKey, _LinkLoad],
        budgets: dict[LinkKey, _LinkBudget],
        *,
        with_validity: bool = True,
        with_ase: bool = False,
    ) -> _LinkBudget:
        power = load.source_w.copy()
        for row, service_id, upstream in load.upstream:
            if upstream in budgets:
                power[row] = budgets[upstream].signal_w[
                    loads[upstream].rows[service_id]
                ]
        normalized = power / load.width_scale
        result = load.model.evaluate_path(load.optical_path, normalized)
        signal = np.minimum(
            result.signal_power_w * load.roadm_transfer, load.roadm_limits_w
        )
        switch_transfer = np.divide(
            signal,
            result.signal_power_w,
            out=np.zeros_like(signal),
            where=result.signal_power_w > 0,
        )
        noise_transfer = np.divide(
            signal,
            power,
            out=np.zeros_like(signal),
            where=power > 0,
        )
        noise_transfer *= load.roadm_penalty
        noise = result.spm_noise_w + result.xpm_noise_w + result.ase_noise_w
        noise *= switch_transfer
        roadm_ase = np.where(signal > 0, load.roadm_ase_w, 0.0)
        noise += roadm_ase
        noise *= load.roadm_penalty
        ase = None
        if with_ase:
            ase = (
                result.ase_noise_w * switch_transfer + roadm_ase
            ) * load.roadm_penalty
        report = None
        if with_validity and (provider := load.provider) is not None:
            if isinstance(provider, OpticalPathValidityProvider):
                report = provider.evaluate_path(
                    key,
                    load.optical_path,
                    normalized,
                    result=result if provider.model is load.model else None,
                )
            else:
                if self.topology.get_link_spans(*key) is not None:
                    msg = "Installed spans require an optical-path validity provider."
                    raise TypeError(msg)
                fiber = load.optical_path.fiber_configs[0]
                report = provider(
                    key,
                    fiber,
                    load.optical_path.edfa_configs[0],
                    normalized,
                    fiber.max_span_length_km * 1000,
                )
            self.validator.last_link_validity[key] = report
        return _LinkBudget(signal, noise, noise_transfer, report, ase)

    def _settle_component(
        self,
        members: list[LinkKey],
        loads: dict[LinkKey, _LinkLoad],
        budgets: dict[LinkKey, _LinkBudget],
    ) -> bool:
        for _ in range(64):
            stable = True
            for key in members:
                update = self._evaluate_link(
                    key, loads[key], loads, budgets, with_validity=False
                )
                previous = budgets.get(key)
                stable = (
                    stable
                    and previous is not None
                    and np.allclose(
                        update.signal_w, previous.signal_w, rtol=1e-9, atol=1e-15
                    )
                )
                budgets[key] = update
            if stable:
                return True
        return False

    def _receiver_snr(
        self,
        service: ServiceRequest,
        path: list[int],
        route: tuple[LinkKey, ...],
        loads: dict[LinkKey, _LinkLoad],
        budgets: dict[LinkKey, _LinkBudget],
    ) -> float:
        service_id = service.service_id
        with_ase = service.transceiver_mode is not None
        first = loads[route[0]]
        noise = float(first.source_ase_w[first.rows[service_id]])
        ase = noise
        reports = []
        signal = 0.0
        for key in route:
            row = loads[key].rows[service_id]
            budget = budgets[key]
            signal = float(budget.signal_w[row])
            transfer = float(budget.noise_transfer[row])
            noise = noise * transfer + float(budget.noise_w[row])
            if with_ase:
                assert budget.ase_w is not None
                ase = ase * transfer + float(budget.ase_w[row])
            if (report := budget.validity) is not None:
                reports.append((report, row))
        snr = -math.inf
        if signal > 0 and noise >= 0:
            snr = 10 * math.log10(signal / noise) if noise > 0 else math.inf
        adjustment = 0.0
        if reports and (snr > -math.inf or with_ase):
            verdict = self._path_validity(reports)
            policy = self.validator.validity_policy
            if with_ase:
                adjusted = verdict.adjusted_snr_db(np.array([0.0]), policy)
                adjustment = float(adjusted[0])
            snr = float(verdict.adjusted_snr_db(np.array([snr]), policy)[0])
        if with_ase and not self._receiver_feasible(
            service,
            first,
            self._path_impairments(path, route, loads),
            snr,
            signal,
            noise,
            ase,
            adjustment,
        ):
            return -math.inf
        return snr

    @staticmethod
    def _path_validity(reports: list[tuple[LinkValidity, int]]) -> ChannelValidity:
        valid = all(report.valid[row] for report, row in reports)
        return ChannelValidity(
            valid=np.array([valid]),
            margin_db=np.array(
                [
                    sum(
                        report.margin_db[row]
                        for report, row in reports
                        if report.valid[row]
                    )
                ]
            ),
            soft_margin_db=np.array(
                [
                    sum(
                        cast("NDArrayFloat", report.soft_margin_db)[row]
                        for report, row in reports
                        if report.valid[row]
                    )
                ]
            ),
            dense_grid_overlap=any(report.dense_grid_overlap for report, _ in reports),
            raman_auto_under_resolved=any(
                report.raman_auto_under_resolved for report, _ in reports
            ),
            fwm_required=any(report.fwm_required for report, _ in reports),
        )

    def _path_impairments(
        self,
        path: list[int],
        route: tuple[LinkKey, ...],
        loads: dict[LinkKey, _LinkLoad],
    ) -> _PathImpairments:
        add = self.topology.get_roadm_add_config(path[0], path[1])
        cd_at_reference = 0.0
        cd_slope = 0.0
        pmd_squared = add.pmd_ps**2
        pdl_squared = add.pdl_db**2
        for index, key in enumerate(route):
            optical = loads[key].optical_path
            for fiber, amp in zip(
                optical.fiber_configs, optical.edfa_configs, strict=True
            ):
                cd_at_reference += fiber.length_km * (
                    fiber.dispersion_parameter_d
                    + fiber.dispersion_slope_s
                    * (_DISPERSION_REFERENCE_NM - fiber.reference_wavelength_nm)
                )
                cd_slope += fiber.length_km * fiber.dispersion_slope_s
                pmd_squared += fiber.length_km * fiber.pmd_coefficient_ps_sqrt_km**2
                if amp is not None and (amp.pmd_ps or amp.pdl_db):
                    count = math.ceil(fiber.length_km / fiber.max_span_length_km)
                    pmd_squared += count * amp.pmd_ps**2
                    pdl_squared += count * amp.pdl_db**2
            roadm = self.topology.get_roadm_path_config(path, index)
            pmd_squared += roadm.pmd_ps**2
            pdl_squared += roadm.pdl_db**2
        return _PathImpairments(
            cd_at_reference, cd_slope, math.sqrt(pmd_squared), math.sqrt(pdl_squared)
        )

    def _receiver_feasible(
        self,
        service: ServiceRequest,
        first: _LinkLoad,
        impairments: _PathImpairments,
        line_snr_db: float,
        signal_w: float,
        noise_w: float,
        ase_w: float,
        validity_adjustment_db: float = 0.0,
    ) -> bool:
        mode = service.transceiver_mode
        assert mode is not None
        row = first.rows[service.service_id]
        wavelength_nm = C_LIGHT * 1e9 / first.model.f_abs_hz[row]
        cd = impairments.cd_at_reference_ps_nm + impairments.cd_slope_ps_nm2 * (
            wavelength_nm - _DISPERSION_REFERENCE_NM
        )
        power_scale = self.validator.ase_polarization_scale
        report = evaluate_receiver(
            mode,
            signal_power_w=signal_w / power_scale,
            ase_noise_w=ase_w / power_scale,
            noise_power_w=noise_w / power_scale,
            noise_bandwidth_hz=first.bandwidths_hz[row],
            line_snr_db=line_snr_db,
            snr_requirement_db=service.snr_requirement_db,
            cd_ps_nm=cd,
            pmd_ps=impairments.pmd_ps,
            pdl_db=impairments.pdl_db,
            validity_adjustment_db=validity_adjustment_db,
        )
        self.validator.last_receiver_feasibility[service.service_id] = report
        return report.feasible
