"""Quality-of-transmission validation for routing and wavelength assignment.

The validator evaluates candidate and affected services without changing the
network occupancy. It uses temporary spectra and noise caches, committing the
candidate-link caches only after all affected services meet their thresholds.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey, NDArrayFloat, NodeID
from network_nlo_eval.models.base import QoTModel
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.power_conventions import (
    ase_polarization_scale as convention_ase_polarization_scale,
)
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)
from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, LinkState, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.link_noise import (
    assemble_link_noise,
    assemble_model_link_noise,
)
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    LinkValidity,
    LinkValidityPolicy,
    LinkValidityProvider,
)
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

#: Planck constant (J s), used in booster ASE.
_H_PLANCK = 6.62607015e-34


def _normalize(u: NodeID, v: NodeID) -> LinkKey:
    """Return a canonical undirected link key.

    Parameters
    ----------
    u : NodeID
        First endpoint.
    v : NodeID
        Second endpoint.

    Returns
    -------
    LinkKey
        Canonical link key.
    """
    return (u, v) if u < v else (v, u)


@dataclass(frozen=True)
class RoadmTerms:
    """ROADM constants for one node and port operation.

    Attributes
    ----------
    config : ROADMConfig
        Original configuration for block checks.
    loss, loss_db : float
        Linear and directly computed dB path loss.
    filtering_penalty : float
        Linear noise-only filtering penalty.
    equalize : bool
        Whether the booster restores launch power.
    booster_nf : float or None
        Linear booster noise figure, or None without equalization.
    booster_base : float
        Booster noise figure times Planck's constant.
    booster_nf_db : float
        Booster noise figure in dB.
    """

    config: ROADMConfig = field(compare=False, repr=False)
    loss: float
    loss_db: float
    filtering_penalty: float
    equalize: bool
    booster_nf: float | None
    booster_base: float
    booster_nf_db: float


def fold_roadm_terms(config: ROADMConfig, *, is_last_hop: bool) -> RoadmTerms:
    """Fold a ROADM configuration into shared constants.

    Drop and express losses are distinct; the filtering penalty affects noise
    only. Equalization restores signal power and adds booster ASE.

    Parameters
    ----------
    config : ROADMConfig
        ROADM configuration.
    is_last_hop : bool
        Use the drop port for the last hop, otherwise the express port.

    Returns
    -------
    RoadmTerms
        Folded ROADM constants.
    """
    operation: Literal["drop", "express"] = "drop" if is_last_hop else "express"
    nf_lin = 10 ** (config.booster_noise_figure_db / 10)
    loss_db = config.path_loss_db(operation)
    return RoadmTerms(
        config=config,
        loss=10 ** (loss_db / 10),
        loss_db=loss_db,
        filtering_penalty=10 ** (config.filtering_penalty_db / 10),
        equalize=config.equalize_output_power,
        booster_nf=nf_lin if config.equalize_output_power else None,
        booster_base=nf_lin * _H_PLANCK,
        booster_nf_db=config.booster_noise_figure_db,
    )


#: One hop consists of ROADM constants and a link noise cache.
HopPair = tuple[RoadmTerms, LinkNoiseCache]

#: Callback returning an optional hop validity report.
ValidityOfHop = Callable[[RoadmTerms, LinkNoiseCache], "LinkValidity | None"]


@dataclass(frozen=True)
class ServiceQoT:
    """Physical SNR and validity aggregated over one service path.

    Attributes
    ----------
    snr_db : float
        Physical SNR in dB; negative infinity when blocked, positive infinity for
        zero noise.
    margin_db, soft_margin_db : float
        Sum of valid-link recommended and tilt-derived margins in dB.
    valid : bool
        Whether all reported links consider the channel valid.
    invalid_links : tuple of LinkKey
        Deduplicated invalid links in path order.
    dense_grid_overlap, raman_auto_under_resolved, fwm_required : bool
        Applicability flags aggregated across reported links.
    """

    snr_db: float
    margin_db: float = 0.0
    soft_margin_db: float = 0.0
    valid: bool = True
    invalid_links: tuple[LinkKey, ...] = ()
    dense_grid_overlap: bool = False
    raman_auto_under_resolved: bool = False
    fwm_required: bool = False

    def effective_snr_db(self, policy: LinkValidityPolicy | None = None) -> float:
        """Apply a validity policy to this service SNR.

        Parameters
        ----------
        policy : LinkValidityPolicy | None
            Validity policy; None disables adjustment.

        Returns
        -------
        float
            Policy-adjusted SNR (dB).
        """
        return float(
            (policy or LinkValidityPolicy()).apply(
                np.array([self.snr_db], dtype=np.float64),
                np.array([self.valid], dtype=bool),
                np.array([self.margin_db], dtype=np.float64),
                np.array([self.soft_margin_db], dtype=np.float64),
                dense_grid_overlap=self.dense_grid_overlap,
                raman_auto_under_resolved=self.raman_auto_under_resolved,
                fwm_required=self.fwm_required,
            )[0]
        )


def accumulate_service_snr(
    hops: Iterable[HopPair],
    *,
    channel_idx: int,
    launch_power_w: float,
    frequency_hz: float,
    channel_bandwidth_hz: float,
    is_blocked: Callable[[RoadmTerms], bool],
    validity_of: ValidityOfHop | None = None,
    validity_policy: LinkValidityPolicy | None = None,
    ase_polarization_scale: float = 1.0,
) -> float:
    """Accumulate path noise and return policy-adjusted single-channel SNR.

    Hops may be lazy: a blocked hop stops iteration immediately. The booster ASE
    scale is one for total power and one-half for per-polarization power.

    Parameters
    ----------
    hops : Iterable[HopPair]
        Lazy or eager sequence of ROADM terms and link noise caches.
    channel_idx : int
        Channel or evaluator-node index.
    launch_power_w : float
        Service launch power (W).
    frequency_hz : float
        Node center frequency (Hz).
    channel_bandwidth_hz : float
        Node bandwidth for booster ASE (Hz).
    is_blocked : Callable[[RoadmTerms], bool]
        Callback indicating whether the ROADM blocks this node.
    validity_of : ValidityOfHop | None
        Optional callback returning link validity.
    validity_policy : LinkValidityPolicy | None
        Optional policy for link validity.
    ase_polarization_scale : float
        Scale for newly generated booster ASE.

    Returns
    -------
    float
        SNR in dB; negative infinity when blocked, positive infinity for zero noise.
    """
    return accumulate_service_snr_detail(
        hops,
        channel_idx=channel_idx,
        launch_power_w=launch_power_w,
        frequency_hz=frequency_hz,
        channel_bandwidth_hz=channel_bandwidth_hz,
        is_blocked=is_blocked,
        validity_of=validity_of,
        ase_polarization_scale=ase_polarization_scale,
    ).effective_snr_db(validity_policy)


def accumulate_service_snr_detail(
    hops: Iterable[HopPair],
    *,
    channel_idx: int,
    launch_power_w: float,
    frequency_hz: float,
    channel_bandwidth_hz: float,
    is_blocked: Callable[[RoadmTerms], bool],
    validity_of: ValidityOfHop | None = None,
    ase_polarization_scale: float = 1.0,
) -> ServiceQoT:
    """Accumulate physical SNR and validity over the service path.

    A blocked channel returns negative infinity; zero noise yields positive
    infinity. Recommended and soft margins sum only over valid reported links.
    The booster ASE scale applies only to newly generated ROADM noise.

    Parameters
    ----------
    hops : Iterable[HopPair]
        Lazy or eager sequence of ROADM terms and link noise caches.
    channel_idx : int
        Channel or evaluator-node index.
    launch_power_w : float
        Service launch power (W).
    frequency_hz : float
        Node center frequency (Hz).
    channel_bandwidth_hz : float
        Node bandwidth for booster ASE (Hz).
    is_blocked : Callable[[RoadmTerms], bool]
        Callback indicating whether the ROADM blocks this node.
    validity_of : ValidityOfHop | None
        Optional callback returning link validity.
    ase_polarization_scale : float
        Scale for newly generated booster ASE.

    Returns
    -------
    ServiceQoT
        Physical SNR and aggregated link validity.
    """
    total_spm = 0.0
    total_xpm = 0.0
    total_ase = 0.0
    signal_power = launch_power_w
    total_spans = 0
    total_margin_db = 0.0
    total_soft_margin_db = 0.0
    valid = True
    invalid_links: list[LinkKey] = []
    dense_grid_overlap = False
    raman_auto_under_resolved = False
    fwm_required = False

    for roadm, cache in hops:
        total_spm += cache.spm.item(channel_idx)
        total_xpm += cache.xpm.item(channel_idx)
        total_ase += cache.ase.item(channel_idx)
        total_spans += cache.span_count

        if is_blocked(roadm):
            return ServiceQoT(snr_db=float("-inf"), valid=False)

        # Read validity only after the hop passes the block check.
        if validity_of is not None:
            link_validity = validity_of(roadm, cache)
            if link_validity is not None:
                # Applicability flags propagate from any hop to the path.
                dense_grid_overlap = dense_grid_overlap or bool(
                    link_validity.dense_grid_overlap
                )
                raman_auto_under_resolved = raman_auto_under_resolved or bool(
                    link_validity.raman_auto_under_resolved
                )
                fwm_required = fwm_required or bool(link_validity.fwm_required)
                if link_validity.channel_valid(channel_idx):
                    total_margin_db += link_validity.margin_at(channel_idx)
                    total_soft_margin_db += link_validity.soft_margin_at(channel_idx)
                else:
                    valid = False
                    if (
                        link_validity.link_key is not None
                        and link_validity.link_key not in invalid_links
                    ):
                        invalid_links.append(link_validity.link_key)

        if roadm.equalize:
            booster_ase = (
                roadm.booster_base
                * frequency_hz
                * channel_bandwidth_hz
                * (roadm.loss - 1.0)
            )
            # Avoid multiplication on the default total-power path.
            if ase_polarization_scale != 1.0:
                booster_ase *= ase_polarization_scale
            total_ase += booster_ase
        else:
            signal_power /= roadm.loss
            total_spm /= roadm.loss
            total_xpm /= roadm.loss
            total_ase /= roadm.loss
        total_spm *= roadm.filtering_penalty
        total_xpm *= roadm.filtering_penalty
        total_ase *= roadm.filtering_penalty

    # SPM coherence penalty across spans.
    penalty = total_spans**0.05 if total_spans > 0 else 1.0
    noise = total_spm * penalty + total_xpm + total_ase
    snr_db = float("inf") if noise <= 0 else 10.0 * math.log10(signal_power / noise)
    return ServiceQoT(
        snr_db=snr_db,
        margin_db=total_margin_db,
        soft_margin_db=total_soft_margin_db,
        valid=valid,
        invalid_links=tuple(invalid_links),
        dense_grid_overlap=dense_grid_overlap,
        raman_auto_under_resolved=raman_auto_under_resolved,
        fwm_required=fwm_required,
    )


class QoTValidator:
    """Validate candidate allocations against QoT requirements.

    Candidate link spectra and noise are evaluated in local caches. The candidate
    and affected existing services are checked on their own paths; candidate-link
    caches are committed only after all checks pass. Optional link validity requires
    an enabled policy and a provider; off is the default.

    Network state, requests, and provider input use the declared launch convention.
    Power is converted once to the evaluator convention before noise calculation.
    The evaluator must provide the corresponding ASE polarization scale, including
    for ROADM booster ASE. Known cached conventions that do not match are recomputed;
    legacy caches without a convention retain their historical reuse behavior.
    Coherent link assembly is opt-in and depends on evaluator support; it does not
    change validity policy or the service-level SPM span penalty.
    """

    def __init__(
        self,
        gn_evaluator: QoTModel,
        network_topology: NetworkTopology,
        spectrum_grid: SpectrumGrid,
        fixed_span_length_km: float | None = None,
        *,
        launch_power_convention: str = "total",
        validity_policy: LinkValidityPolicy | None = None,
        validity_provider: LinkValidityProvider | None = None,
        coherent_assembly: bool = False,
    ):
        """Initialize the QoT validator and validate policy and power conventions.

        An enabled validity policy requires a provider; a provider with off mode is
        rejected. Its input convention must match the evaluator convention.

        Parameters
        ----------
        gn_evaluator : QoTModel
            Link noise evaluator.
        network_topology : NetworkTopology
            Network topology and element configurations.
        spectrum_grid : SpectrumGrid
            Spectrum grid.
        fixed_span_length_km : float | None
            Optional fixed span length (km).
        launch_power_convention : str
            Network and request power convention.
        validity_policy : LinkValidityPolicy | None
            Optional policy for link validity.
        validity_provider : LinkValidityProvider | None
            Optional link validity provider.
        coherent_assembly : bool
            Request coherent accumulation when the evaluator supports it.

        Raises
        ------
        ValueError
            If provider, policy or power-convention settings are incompatible.
        """
        policy = validity_policy or LinkValidityPolicy()
        if policy.enabled and validity_provider is None:
            msg = (
                f"validity_policy mode={policy.mode} requires a validity_provider; "
                "pass an H3LinkValidityProvider or use mode='off'."
            )
            raise ValueError(msg)
        if not policy.enabled and validity_provider is not None:
            msg = (
                "validity_provider was given with validity_policy mode='off'; "
                "choose an explicit mode ('flag', 'reject', 'degrade' or 'soft') to enable it."
            )
            raise ValueError(msg)
        self.gn_evaluator = gn_evaluator
        self.network_topology = network_topology
        self.spectrum_grid = spectrum_grid
        self._fixed_span_length_km = fixed_span_length_km
        self.launch_power_convention = resolve_launch_power_convention(
            launch_power_convention
        )
        #: Evaluator and noise-cache power convention; converted once at entry.
        self.eval_power_convention = resolve_launch_power_convention(
            str(getattr(gn_evaluator, "launch_power_convention", "total"))
        )
        #: Booster ASE scale must match the evaluator power convention.
        scale = float(getattr(gn_evaluator, "ase_polarization_scale", 1.0))
        if not np.isfinite(scale) or scale <= 0.0:
            msg = f"gn_evaluator.ase_polarization_scale must be finite and positive, got {scale!r}."
            raise ValueError(msg)
        expected_scale = convention_ase_polarization_scale(self.eval_power_convention)
        if scale != expected_scale:
            msg = (
                "gn_evaluator.ase_polarization_scale must be "
                f"{expected_scale:g} for launch_power_convention="
                f"{self.eval_power_convention!r}, got {scale:g}. The shared ROADM booster ASE "
                "would otherwise be accumulated in a different coordinate than the link-noise "
                "cache."
            )
            raise ValueError(msg)
        self.ase_polarization_scale = scale
        provider_convention = getattr(validity_provider, "input_power_convention", None)
        if provider_convention is not None:
            provider_convention = resolve_launch_power_convention(
                str(provider_convention)
            )
            if provider_convention != self.eval_power_convention:
                msg = (
                    "validity_provider input_power_convention="
                    f"{provider_convention!r} must equal the evaluator convention "
                    f"{self.eval_power_convention!r}; QoTValidator passes evaluator-coordinate power "
                    "to the provider, so a mismatch would convert the spectrum twice."
                )
                raise ValueError(msg)
        self.validity_policy = policy
        self.validity_provider = validity_provider
        #: Coherent assembly is opt-in and requires evaluator support.
        self.coherent_assembly = bool(coherent_assembly)
        #: Reports for links affected by the most recent verification.
        self.last_link_validity: dict[LinkKey, LinkValidity] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        """Return the number of evaluator nodes (channels or connections).

        Returns
        -------
        int
            Evaluator node count.
        """
        return self.gn_evaluator.n_nodes

    def _is_blocked(self, roadm: ROADMConfig, node_index: int) -> bool:
        """Return whether the ROADM blocks this evaluator node.

        Parameters
        ----------
        roadm : ROADMConfig
            ROADM configuration.
        node_index : int
            Evaluator node index.

        Returns
        -------
        bool
            Whether the node is blocked.
        """
        return node_index in self.gn_evaluator.blocked_nodes(roadm)

    def _node_frequency_and_bandwidth(self, channel_idx: int) -> tuple[float, float]:
        """Return node center frequency and bandwidth (Hz).

        The evaluator arrays use node order, which may differ from legacy channels
        for a flexible grid.

        Parameters
        ----------
        channel_idx : int
            Channel or evaluator-node index.

        Returns
        -------
        tuple[float, float]
            Center frequency and bandwidth in Hz.
        """
        evaluator = self.gn_evaluator
        return (
            float(evaluator.f_abs_hz[channel_idx]),
            float(evaluator.channel_bandwidth_hz_channels[channel_idx]),
        )

    def verify_allocation(
        self,
        service_request: ServiceRequest,
        path: list[NodeID],
        channel_idx: int,
        current_network_state: NetworkState,
        fixed_span_length_km: float | None = None,
    ) -> tuple[bool, NDArrayFloat | None]:
        """Verify the candidate and all affected existing services.

        Candidate spectra are temporary. Affected links are computed once; on success
        their caches are stored and the path SNR array is returned. A rejected
        candidate does not change link occupancy.

        Parameters
        ----------
        service_request : ServiceRequest
            Candidate service request.
        path : list[NodeID]
            Ordered path nodes.
        channel_idx : int
            Channel or evaluator-node index.
        current_network_state : NetworkState
            Current network state.
        fixed_span_length_km : float | None
            Optional fixed span length (km).

        Returns
        -------
        tuple[bool, ndarray | None]
            Success flag and path-node SNR array, or None on rejection.
        """
        span_km = fixed_span_length_km or self._fixed_span_length_km
        self.last_link_validity.clear()

        # Copy candidate spectra and check availability.
        temp_power_profiles: dict[LinkKey, NDArrayFloat] = {}
        affected_links: set[LinkKey] = set()

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = current_network_state.get_link_state(u, v)

            if link_state.occupied_channels[channel_idx]:
                return False, None  # Channel already occupied.

            link_key = _normalize(u, v)
            affected_links.add(link_key)
            temp_profile = link_state.launch_power_profile_w.copy()
            temp_profile[channel_idx] = service_request.launch_power_w
            temp_power_profiles[link_key] = temp_profile

        # Compute each affected link once in the temporary cache.
        temp_noise_cache: dict[LinkKey, LinkNoiseCache] = {}
        for link_key in affected_links:
            temp_noise_cache[link_key] = self._calc_link_noise(
                link_key,
                temp_power_profiles[link_key],
                span_km,
            )

        # Evaluate the candidate service.
        new_snr_db = self._eval_service_snr(
            path,
            channel_idx,
            service_request.launch_power_w,
            temp_noise_cache,
            current_network_state,
            span_km,
        )
        if new_snr_db < service_request.snr_requirement_db:
            return False, None

        # Collect affected existing services.
        affected_old_services = self._collect_affected_services(
            path,
            channel_idx,
            current_network_state,
        )

        # Evaluate each existing service on its own path and launch power.
        for old_service in affected_old_services:
            old_snr_db = self._eval_service_snr(
                old_service.path,
                old_service.wavelength,
                old_service.launch_power_w,
                temp_noise_cache,
                current_network_state,
                span_km,
            )
            if old_snr_db < old_service.snr_requirement_db:
                return False, None

        # Commit candidate caches after all checks pass.
        for link_key, cache in temp_noise_cache.items():
            u, v = link_key
            link_state = current_network_state.get_link_state(u, v)
            link_state.noise_cache = cache

        # Build the path SNR array for the returned result.
        final_snrs_db = self._build_path_snr_array(
            path,
            temp_power_profiles,
            temp_noise_cache,
            current_network_state,
            span_km,
        )

        return True, final_snrs_db

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _calc_link_noise(
        self,
        link_key: LinkKey,
        power_profile: NDArrayFloat,
        fixed_span_length_km: float | None,
    ) -> LinkNoiseCache:
        """Build a link noise cache for the supplied spectrum.

        Power is converted to evaluator convention once. An enabled provider adds a
        validity report without changing the noise calculation.

        Parameters
        ----------
        link_key : LinkKey
            Canonical link endpoints.
        power_profile : ndarray
            Candidate network-coordinate launch powers (W).
        fixed_span_length_km : float | None
            Optional fixed span length (km).

        Returns
        -------
        LinkNoiseCache
            Link noise cache with optional validity report.
        """
        u, v = link_key
        fiber_config = self.network_topology.get_fiber_config(u, v)
        edfa_config = self.network_topology.get_link_edfa_config(u, v)
        power_eval = to_convention(
            power_profile, self.launch_power_convention, self.eval_power_convention
        )

        link_total_length_m = fiber_config.length_km * 1000
        span_length_m = (fixed_span_length_km or fiber_config.max_span_length_km) * 1000
        num_full_spans = int(link_total_length_m // span_length_m)
        remainder_length_m = link_total_length_m % span_length_m

        validity: LinkValidity | None = None
        if self.validity_policy.enabled:
            provider = self.validity_provider
            if (
                provider is None
            ):  # pragma: no cover - constructor rejects this combination
                msg = "validity_policy is enabled but no provider is configured."
                raise RuntimeError(msg)
            validity = provider(
                link_key=link_key,
                fiber_config=fiber_config,
                edfa_config=edfa_config,
                power_profile_w=power_eval,
                span_length_m=span_length_m,
            )
            self.last_link_validity[link_key] = validity

        if isinstance(self.gn_evaluator, MultiBandISRSGN):
            return assemble_link_noise(
                evaluator=self.gn_evaluator.for_fiber(fiber_config),
                edfa_config=edfa_config,
                power_profile=power_eval,
                span_length_m=span_length_m,
                num_full_spans=num_full_spans,
                remainder_length_m=remainder_length_m,
                num_channels=self.num_nodes,
                validity=validity,
                launch_power_convention=self.eval_power_convention,
                coherent=self.coherent_assembly,
            )
        return assemble_model_link_noise(
            evaluator=self.gn_evaluator,
            fiber_config=fiber_config,
            edfa_config=edfa_config,
            power_profile=power_eval,
            span_length_m=span_length_m,
            num_full_spans=num_full_spans,
            remainder_length_m=remainder_length_m,
            validity=validity,
            launch_power_convention=self.eval_power_convention,
            coherent=self.coherent_assembly,
        )

    def _get_or_calc_noise_cache(
        self,
        link_key: LinkKey,
        link_state: LinkState,
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        fixed_span_length_km: float | None,
    ) -> LinkNoiseCache:
        """Get temporary or link-state noise, computing it lazily if needed.

        A cache with a known, different power convention is recomputed. Legacy
        unlabelled caches retain their existing reuse behavior.

        Parameters
        ----------
        link_key : LinkKey
            Canonical link endpoints.
        link_state : object
            Link state.
        temp_noise_cache : dict[LinkKey, LinkNoiseCache]
            Candidate-link noise caches.
        fixed_span_length_km : float | None
            Optional fixed span length (km).

        Returns
        -------
        LinkNoiseCache
            Reusable or newly calculated link noise cache.
        """
        if link_key in temp_noise_cache:
            return temp_noise_cache[link_key]

        cached = link_state.noise_cache
        if cached is not None:
            convention = cached.launch_power_convention
            if convention is None or convention == self.eval_power_convention:
                return cached

        cache = self._calc_link_noise(
            link_key,
            link_state.launch_power_profile_w,
            fixed_span_length_km,
        )
        link_state.noise_cache = cache
        return cache

    def _eval_service_snr(
        self,
        path: list[NodeID],
        channel_idx: int,
        launch_power_w: float,
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        network_state: NetworkState,
        fixed_span_length_km: float | None,
    ) -> float:
        """Calculate policy-adjusted SNR along the service path.

        Affected links use temporary caches; other links are fetched lazily. A blocked
        hop stops iteration before later link caches are fetched or written. Launch
        power is converted to the evaluator convention before accumulation.

        Parameters
        ----------
        path : list[NodeID]
            Ordered path nodes.
        channel_idx : int
            Channel or evaluator-node index.
        launch_power_w : float
            Service launch power (W).
        temp_noise_cache : dict[LinkKey, LinkNoiseCache]
            Candidate-link noise caches.
        network_state : NetworkState
            Current network state.
        fixed_span_length_km : float | None
            Optional fixed span length (km).

        Returns
        -------
        float
            Policy-adjusted service SNR (dB).
        """
        frequency_hz, channel_bandwidth_hz = self._node_frequency_and_bandwidth(
            channel_idx
        )
        launch_power_eval = to_convention(
            launch_power_w, self.launch_power_convention, self.eval_power_convention
        )
        last_hop = len(path) - 2

        def _hops() -> Iterable[HopPair]:
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                link_key = _normalize(u, v)
                link_state = network_state.get_link_state(u, v)
                cache = self._get_or_calc_noise_cache(
                    link_key, link_state, temp_noise_cache, fixed_span_length_km
                )
                roadm = self.network_topology.get_roadm_config(v)
                yield fold_roadm_terms(roadm, is_last_hop=i == last_hop), cache

        qot = accumulate_service_snr_detail(
            _hops(),
            channel_idx=channel_idx,
            launch_power_w=launch_power_eval,
            frequency_hz=frequency_hz,
            channel_bandwidth_hz=channel_bandwidth_hz,
            is_blocked=lambda terms: self._is_blocked(terms.config, channel_idx),
            validity_of=self._validity_of if self.validity_policy.enabled else None,
            ase_polarization_scale=self.ase_polarization_scale,
        )
        return qot.effective_snr_db(self.validity_policy)

    @staticmethod
    def _validity_of(_roadm: RoadmTerms, cache: LinkNoiseCache) -> LinkValidity | None:
        """Return the validity report attached to a hop noise cache.

        Parameters
        ----------
        _roadm : RoadmTerms
            ROADM terms for the hop.
        cache : LinkNoiseCache
            Noise cache for the hop.

        Returns
        -------
        LinkValidity | None
            Attached link report, if present.
        """
        return cache.validity

    def _collect_affected_services(
        self,
        path: list[NodeID],
        channel_idx: int,
        network_state: NetworkState,
    ) -> list[AllocatedService]:
        """Collect existing services affected by candidate-path links.

        Parameters
        ----------
        path : list[NodeID]
            Ordered path nodes.
        channel_idx : int
            Channel or evaluator-node index.
        network_state : NetworkState
            Current network state.

        Returns
        -------
        list[AllocatedService]
            Affected existing services, deduplicated by identifier.
        """
        seen_ids: set[int] = set()
        affected: list[AllocatedService] = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_state = network_state.get_link_state(u, v)

            for ch_idx in range(self.num_nodes):
                if ch_idx == channel_idx:
                    continue
                if not link_state.occupied_channels[ch_idx]:
                    continue
                service_id = link_state.allocated_service_ids[ch_idx]
                if service_id == -1 or service_id in seen_ids:
                    continue
                seen_ids.add(service_id)
                svc = network_state.get_allocated_service(service_id)
                if svc is not None:
                    affected.append(svc)

        return affected

    def _build_path_snr_array(
        self,
        path: list[NodeID],
        temp_power_profiles: dict[LinkKey, NDArrayFloat],
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        network_state: NetworkState,
        fixed_span_length_km: float | None,
    ) -> NDArrayFloat:
        """Build path SNR by node, applying the configured link validity policy.

        Parameters
        ----------
        path : list[NodeID]
            Ordered path nodes.
        temp_power_profiles : dict[LinkKey, ndarray]
            Candidate link power profiles.
        temp_noise_cache : dict[LinkKey, LinkNoiseCache]
            Candidate-link noise caches.
        network_state : NetworkState
            Current network state.
        fixed_span_length_km : float | None
            Optional fixed span length (km).

        Returns
        -------
        ndarray
            Policy-adjusted SNR per evaluator node (dB).
        """
        total_spm = np.zeros(self.num_nodes, dtype=np.float64)
        total_xpm = np.zeros(self.num_nodes, dtype=np.float64)
        total_ase = np.zeros(self.num_nodes, dtype=np.float64)
        total_spans = 0
        link_validities: list[LinkValidity] = []
        u0, v0 = path[0], path[1]
        link_key0 = _normalize(u0, v0)
        if link_key0 in temp_power_profiles:
            signal_power = temp_power_profiles[link_key0].copy()
        else:
            signal_power = network_state.get_link_state(
                u0, v0
            ).launch_power_profile_w.copy()
        signal_power = to_convention(
            signal_power, self.launch_power_convention, self.eval_power_convention
        )

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link_key = _normalize(u, v)
            link_state = network_state.get_link_state(u, v)

            cache = self._get_or_calc_noise_cache(
                link_key,
                link_state,
                temp_noise_cache,
                fixed_span_length_km,
            )

            total_spm += cache.spm
            total_xpm += cache.xpm
            total_ase += cache.ase
            total_spans += cache.span_count
            if self.validity_policy.enabled and cache.validity is not None:
                link_validities.append(cache.validity)

            roadm = self.network_topology.get_roadm_config(v)
            # Use the same ROADM terms as scalar evaluation, with node arrays.
            terms = fold_roadm_terms(roadm, is_last_hop=i == len(path) - 2)
            if terms.booster_nf is not None:
                booster_ase = (
                    terms.booster_base
                    * self.gn_evaluator.f_abs_hz
                    * self.gn_evaluator.channel_bandwidth_hz_channels
                    * (terms.loss - 1.0)
                )
                # Zero-signal nodes have negative-infinite SNR.
                booster_ase[signal_power <= 0] = 0
                # Preserve the total-power path when the scale is one.
                if self.ase_polarization_scale != 1.0:
                    booster_ase = booster_ase * self.ase_polarization_scale
                total_ase += booster_ase
            else:
                signal_power /= terms.loss
                total_spm /= terms.loss
                total_xpm /= terms.loss
                total_ase /= terms.loss
            total_spm *= terms.filtering_penalty
            total_xpm *= terms.filtering_penalty
            total_ase *= terms.filtering_penalty
            blocked = [
                index
                for index in range(self.num_nodes)
                if self._is_blocked(roadm, index)
            ]
            if blocked:
                signal_power[blocked] = 0

        penalty = total_spans**0.05 if total_spans > 0 else 1.0
        total_noise = total_spm * penalty + total_xpm + total_ase

        snrs_db = np.full(self.num_nodes, -np.inf, dtype=np.float64)
        for ch_idx in range(self.num_nodes):
            if total_noise[ch_idx] <= 0:
                continue
            if signal_power[ch_idx] > 0:
                snrs_db[ch_idx] = 10.0 * math.log10(
                    signal_power[ch_idx] / total_noise[ch_idx]
                )

        if not link_validities:
            return snrs_db
        verdict = ChannelValidity.combine(link_validities, self.num_nodes)
        return np.asarray(
            verdict.adjusted_snr_db(snrs_db, self.validity_policy), dtype=np.float64
        )


# --- Base allocator interface ---
class BaseRWAAllocator:
    """Base interface for routing and wavelength allocators."""

    def allocate(
        self, service_request: ServiceRequest, network_state: NetworkState
    ) -> tuple[bool, AllocatedService | None]:
        """Try to allocate a request and return success and the allocated service.

        Parameters
        ----------
        service_request : ServiceRequest
            Candidate service request.
        network_state : NetworkState
            Current network state.

        Returns
        -------
        tuple[bool, AllocatedService | None]
            Success flag and allocated service, or None on failure.
        """
        raise NotImplementedError

    def release(
        self, allocated_service: AllocatedService, network_state: NetworkState
    ) -> None:
        """Release the allocated service resources.

        Parameters
        ----------
        allocated_service : AllocatedService
            Service to release.
        network_state : NetworkState
            Current network state.
        """
        raise NotImplementedError
