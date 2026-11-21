"""Quality-of-transmission validation for routing and wavelength assignment.

The validator evaluates candidate and affected services without changing the
network occupancy. It uses temporary spectra and noise caches, committing the
candidate-link caches only after all affected services meet their thresholds.
"""

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey, NDArrayFloat, NodeID
from network_nlo_eval.models.base import MultiSpanOpticalPath, QoTModel
from network_nlo_eval.models.power_conventions import (
    ase_polarization_scale as convention_ase_polarization_scale,
)
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)
from network_nlo_eval.models.route import roadm_target_transfer
from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache, LinkState, NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.carrier_loading import CarrierAdmission
from network_nlo_eval.rwa.link_noise import (
    FiberBoundQoTModel,
    assemble_link_noise,
    assemble_model_link_noise,
)
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    LinkValidity,
    LinkValidityPolicy,
    LinkValidityProvider,
    OpticalPathValidityProvider,
)
from network_nlo_eval.simulation.traffic import AllocatedService, ServiceRequest

if TYPE_CHECKING:
    from network_nlo_eval.core.transceiver import ReceiverFeasibility
    from network_nlo_eval.models.base import PathQoTResult
    from network_nlo_eval.rwa.carrier_loading import (
        CarrierModelFactory,
        CarrierValidityFactory,
    )

#: Planck constant (J s), used in booster ASE.
_H_PLANCK = 6.62607015e-34


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
    only. A configured target takes precedence over the legacy booster.

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
    equalize = config.equalize_output_power and config.output_target is None
    return RoadmTerms(
        config=config,
        loss=10 ** (loss_db / 10),
        loss_db=loss_db,
        filtering_penalty=10 ** (config.filtering_penalty_db / 10),
        equalize=equalize,
        booster_nf=nf_lin if equalize else None,
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
    total_margin_db = 0.0
    total_soft_margin_db = 0.0
    valid = True
    invalid_links: list[LinkKey] = []
    dense_grid_overlap = False
    raman_auto_under_resolved = False
    fwm_required = False

    for roadm, cache in hops:
        if cache.power_transfer is not None:
            transfer = cache.power_transfer.item(channel_idx)
            signal_power *= transfer
            total_spm *= transfer
            total_xpm *= transfer
            total_ase *= transfer
        total_spm += cache.spm.item(channel_idx)
        total_xpm += cache.xpm.item(channel_idx)
        total_ase += cache.ase.item(channel_idx)

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
            transfer = 1.0 / roadm.loss
            if (target := roadm.config.output_target) is not None and signal_power > 0:
                limit = target.power_w
                if limit is None:
                    assert target.psd_w_per_hz is not None
                    limit = target.psd_w_per_hz * channel_bandwidth_hz
                transfer = min(
                    transfer, (limit * ase_polarization_scale) / signal_power
                )
            signal_power *= transfer
            total_spm *= transfer
            total_xpm *= transfer
            total_ase *= transfer
        total_spm *= roadm.filtering_penalty
        total_xpm *= roadm.filtering_penalty
        total_ase *= roadm.filtering_penalty

    noise = total_spm + total_xpm + total_ase
    if signal_power <= 0:
        snr_db = -math.inf
    elif noise <= 0:
        snr_db = math.inf
    else:
        snr_db = 10.0 * math.log10(signal_power / noise)
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
    Coherent assembly is opt-in and depends on evaluator support.
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
        carrier_model_factory: "CarrierModelFactory | None" = None,
        carrier_validity_factory: "CarrierValidityFactory | None" = None,
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
            Request full-route evaluation for a uniform load and coherent
            link assembly for varying loads when the evaluator supports it.
        carrier_model_factory : callable or None
            Build a model from the current carrier geometries and slots per
            channel. Defaults to the evaluator's ``for_connections`` method.
        carrier_validity_factory : callable or None
            Build an optical-path validity provider for a geometry-bound model.
            Required for carrier loading when validity policy is enabled.

        Raises
        ------
        ValueError
            If provider, policy or power-convention settings are incompatible.
        """
        policy = validity_policy or LinkValidityPolicy()
        if (
            policy.enabled
            and validity_provider is None
            and carrier_validity_factory is None
        ):
            msg = (
                f"validity_policy mode={policy.mode} requires a validity_provider; "
                "pass an H3LinkValidityProvider or use mode='off'."
            )
            raise ValueError(msg)
        if not policy.enabled and (
            validity_provider is not None or carrier_validity_factory is not None
        ):
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
        self._carrier_model_factory = carrier_model_factory or getattr(
            gn_evaluator, "for_connections", None
        )
        self._carrier_validity_factory = carrier_validity_factory
        self._carrier_admission: CarrierAdmission | None = None
        #: Reports for links affected by the most recent verification.
        self.last_link_validity: dict[LinkKey, LinkValidity] = {}
        self.last_receiver_feasibility: dict[int, ReceiverFeasibility] = {}

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

    def verify_allocation(
        self,
        service_request: ServiceRequest,
        path: list[NodeID],
        channel_idx: int,
        current_network_state: NetworkState,
        fixed_span_length_km: float | None = None,
        *,
        first_slot: int | None = None,
    ) -> tuple[bool, NDArrayFloat | None]:
        """Verify the candidate and all affected existing services.

        Candidate spectra are temporary. Routes and affected link budgets
        are evaluated once per attempt. On success, store any computed
        link caches and return the same SNR array used for admission.
        Receiver tests reached in the most recent attempt are available in
        ``last_receiver_feasibility``, keyed by service ID, including failures.

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
        first_slot : int or None
            First reserved slot for flexible-grid admission. The returned array
            uses anchor indices. The candidate takes precedence at its anchor.

        Returns
        -------
        tuple[bool, ndarray | None]
            Success flag and path-node SNR array, or None on rejection.
        """
        span_km = (
            self._fixed_span_length_km
            if fixed_span_length_km is None
            else fixed_span_length_km
        )
        self.last_link_validity.clear()
        self.last_receiver_feasibility.clear()

        launch_power = service_request.launch_power_w
        if len(path) < 2 or not math.isfinite(launch_power) or launch_power <= 0:
            return False, None
        if span_km is not None and (not math.isfinite(span_km) or span_km <= 0):
            msg = "The fixed span length must be finite and positive."
            raise ValueError(msg)

        if (
            first_slot is not None
            or service_request.symbol_rate_hz is not None
            or service_request.transceiver_mode is not None
            or current_network_state.has_per_carrier_services
        ):
            if self._carrier_admission is None:
                if self._carrier_model_factory is None:
                    msg = "Per-carrier admission requires a carrier_model_factory or an evaluator with for_connections."
                    raise TypeError(msg)
                if (
                    self.validity_policy.enabled
                    and self._carrier_validity_factory is None
                ):
                    msg = "Per-carrier admission with validity enabled requires a carrier_validity_factory."
                    raise TypeError(msg)
                self._carrier_admission = CarrierAdmission(
                    self, self._carrier_model_factory, self._carrier_validity_factory
                )
            return self._carrier_admission.verify(
                service_request,
                path,
                channel_idx,
                first_slot,
                current_network_state,
                span_km,
            )
        if self.validity_policy.enabled and self.validity_provider is None:
            msg = "Fixed-grid admission with validity enabled requires a validity_provider."
            raise TypeError(msg)

        temp_power_profiles: dict[LinkKey, NDArrayFloat] = {}

        for u, v in pairwise(path):
            link_state = current_network_state.get_link_state(u, v)

            if link_state.occupied_channels[channel_idx]:
                return False, None  # Channel already occupied.

            link_key = link_state.link_key
            temp_profile = link_state.launch_power_profile_w.copy()
            temp_profile[channel_idx] = launch_power
            temp_power_profiles[link_key] = temp_profile

        temp_noise_cache: dict[LinkKey, LinkNoiseCache] = {}
        final_snrs_db = self._build_path_snr_array(
            path,
            temp_power_profiles,
            temp_noise_cache,
            current_network_state,
            span_km,
        )
        candidate_snr = final_snrs_db[channel_idx]
        if not (
            candidate_snr > -math.inf
            and candidate_snr >= service_request.snr_requirement_db
        ):
            return False, None

        # Collect affected existing services.
        affected_old_services = self._collect_affected_services(
            path,
            channel_idx,
            current_network_state,
        )

        path_snrs = {tuple(path): final_snrs_db}
        for old_service in affected_old_services:
            path_key = tuple(old_service.path)
            if (snrs := path_snrs.get(path_key)) is None:
                snrs = self._build_path_snr_array(
                    old_service.path,
                    temp_power_profiles,
                    temp_noise_cache,
                    current_network_state,
                    span_km,
                )
                path_snrs[path_key] = snrs
            incumbent_snr = snrs[old_service.wavelength]
            if not (
                incumbent_snr > -math.inf
                and incumbent_snr >= old_service.snr_requirement_db
            ):
                return False, None

        # Commit candidate caches after all checks pass.
        for link_key, cache in temp_noise_cache.items():
            u, v = link_key
            link_state = current_network_state.get_link_state(u, v)
            link_state.noise_cache = cache

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
        power_eval = to_convention(
            power_profile, self.launch_power_convention, self.eval_power_convention
        )
        if (spans := self.network_topology.get_link_spans(u, v)) is not None:
            path = MultiSpanOpticalPath(
                [span.fiber for span in spans], [span.edfa for span in spans]
            )
            provider = self.validity_provider
            if provider is not None and not isinstance(
                provider, OpticalPathValidityProvider
            ):
                msg = "Installed spans require an optical-path validity provider."
                raise TypeError(msg)
            result = self.gn_evaluator.evaluate_path(path, power_eval)
            installed_validity = None
            if provider is not None:
                installed_validity = provider.evaluate_path(
                    link_key,
                    path,
                    power_eval,
                    result=result if provider.model is self.gn_evaluator else None,
                )
                self.last_link_validity[link_key] = installed_validity
            return self._cache_link_result(
                result, power_eval, fixed_span_length_km, installed_validity
            )

        fiber_config = self.network_topology.get_fiber_config(u, v)
        edfa_config = self.network_topology.get_link_edfa_config(u, v)
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

        if not edfa_config.is_loss_compensating:
            if fixed_span_length_km is not None:
                fiber_config = fiber_config.model_copy(
                    update={"max_span_length_km": fixed_span_length_km}
                )
            result = self.gn_evaluator.evaluate_path(
                MultiSpanOpticalPath([fiber_config], [edfa_config]), power_eval
            )
            return self._cache_link_result(
                result,
                power_eval,
                fixed_span_length_km,
                validity,
            )

        if isinstance(self.gn_evaluator, FiberBoundQoTModel):
            cache = assemble_link_noise(
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
        else:
            cache = assemble_model_link_noise(
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
        cache.fixed_span_length_km = fixed_span_length_km
        return cache

    def _cache_link_result(
        self,
        result: "PathQoTResult",
        power_eval: NDArrayFloat,
        fixed_span_length_km: float | None,
        validity: LinkValidity | None,
    ) -> LinkNoiseCache:
        """Store the output budget of an installed or non-compensating link."""
        return LinkNoiseCache(
            spm=result.spm_noise_w,
            xpm=result.xpm_noise_w,
            ase=result.ase_noise_w,
            span_count=result.span_count,
            validity=validity,
            launch_power_convention=self.eval_power_convention,
            power_transfer=np.divide(
                result.signal_power_w,
                power_eval * self.gn_evaluator.power_width_scale(),
                out=np.ones_like(power_eval),
                where=power_eval > 0,
            ),
            fixed_span_length_km=fixed_span_length_km,
        )

    def _get_or_calc_noise_cache(
        self,
        link_state: LinkState,
        temp_power_profiles: dict[LinkKey, NDArrayFloat],
        temp_noise_cache: dict[LinkKey, LinkNoiseCache],
        fixed_span_length_km: float | None,
    ) -> LinkNoiseCache:
        """Get temporary or link-state noise, computing it lazily if needed.

        A cache with a known, different power convention is recomputed. Legacy
        unlabelled caches retain their existing reuse behavior.

        Parameters
        ----------
        link_state : LinkState
            Link state.
        temp_power_profiles : dict[LinkKey, ndarray]
            Candidate link power profiles.
        temp_noise_cache : dict[LinkKey, LinkNoiseCache]
            Candidate-link noise caches.
        fixed_span_length_km : float | None
            Optional fixed span length (km).

        Returns
        -------
        LinkNoiseCache
            Reusable or newly calculated link noise cache.
        """
        link_key = link_state.link_key
        if (profile := temp_power_profiles.get(link_key)) is not None:
            if (cache := temp_noise_cache.get(link_key)) is None:
                cache = self._calc_link_noise(link_key, profile, fixed_span_length_km)
                temp_noise_cache[link_key] = cache
            return cache

        cached = link_state.noise_cache
        if cached is not None:
            convention = cached.launch_power_convention
            if (
                convention is None or convention == self.eval_power_convention
            ) and cached.fixed_span_length_km == fixed_span_length_km:
                return cached

        cache = self._calc_link_noise(
            link_key,
            link_state.launch_power_profile_w,
            fixed_span_length_km,
        )
        link_state.noise_cache = cache
        return cache

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

        for u, v in pairwise(path):
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
        link_validities: list[LinkValidity] = []
        u0, v0 = path[0], path[1]
        first_link_state = network_state.get_link_state(u0, v0)
        first_profile = temp_power_profiles.get(
            first_link_state.link_key, first_link_state.launch_power_profile_w
        )
        needs_path_propagation = len(path) > 2 and (
            self.coherent_assembly or self._path_changes_signal_power(path)
        )
        use_direct_path = needs_path_propagation and self._has_uniform_loading(
            path, first_link_state, first_profile, temp_power_profiles, network_state
        )
        if use_direct_path and not self.validity_policy.enabled:
            return self._direct_path_snr(path, first_profile, fixed_span_length_km)

        signal_power = to_convention(
            first_profile.copy(),
            self.launch_power_convention,
            self.eval_power_convention,
        )
        signal_power *= self.gn_evaluator.power_width_scale()

        for index, (u, v) in enumerate(pairwise(path)):
            link_state = network_state.get_link_state(u, v)

            cache = self._get_or_calc_noise_cache(
                link_state,
                temp_power_profiles,
                temp_noise_cache,
                fixed_span_length_km,
            )

            if cache.power_transfer is not None:
                signal_power *= cache.power_transfer
                total_spm *= cache.power_transfer
                total_xpm *= cache.power_transfer
                total_ase *= cache.power_transfer
            total_spm += cache.spm
            total_xpm += cache.xpm
            total_ase += cache.ase
            if self.validity_policy.enabled and cache.validity is not None:
                link_validities.append(cache.validity)

            roadm = self.network_topology.get_roadm_path_config(path, index)
            # Use the same ROADM terms as scalar evaluation, with node arrays.
            terms = fold_roadm_terms(roadm, is_last_hop=index == len(path) - 2)
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
                transfer: NDArrayFloat | float = 1.0 / terms.loss
                if roadm.output_target is not None:
                    transfer = roadm_target_transfer(
                        roadm.output_target,
                        signal_power,
                        self.gn_evaluator.channel_bandwidth_hz_channels,
                        terms.loss,
                        self.ase_polarization_scale,
                    )
                signal_power *= transfer
                total_spm *= transfer
                total_xpm *= transfer
                total_ase *= transfer
            total_spm *= terms.filtering_penalty
            total_xpm *= terms.filtering_penalty
            total_ase *= terms.filtering_penalty
            blocked = self.gn_evaluator.blocked_nodes(roadm)
            if blocked:
                signal_power[blocked] = 0

        if use_direct_path:
            snrs_db = self._direct_path_snr(path, first_profile, fixed_span_length_km)
        else:
            total_noise = total_spm + total_xpm + total_ase
            snrs_db = np.full(self.num_nodes, -np.inf, dtype=np.float64)
            active = signal_power > 0
            noisy = active & (total_noise > 0)
            snrs_db[noisy] = 10 * np.log10(signal_power[noisy] / total_noise[noisy])
            snrs_db[active & (total_noise == 0)] = np.inf

        if not link_validities:
            return snrs_db
        verdict = ChannelValidity.combine(link_validities, self.num_nodes)
        return np.asarray(
            verdict.adjusted_snr_db(snrs_db, self.validity_policy), dtype=np.float64
        )

    def _path_changes_signal_power(self, path: list[NodeID]) -> bool:
        """Identify equipment for which a repeated nominal spectrum is insufficient."""
        for index, (u, v) in enumerate(pairwise(path)):
            roadm = self.network_topology.get_roadm_path_config(path, index)
            if (
                not roadm.equalize_output_power
                or roadm.output_target is not None
                or roadm.blocked_channels
                or roadm.blocked_slots
            ):
                return True
            if self.network_topology.get_link_spans(u, v) is not None:
                return True
            edfa = self.network_topology.get_link_edfa_config(u, v)
            if not edfa.is_loss_compensating:
                return True
        return False

    def _direct_path_snr(
        self,
        path: list[NodeID],
        profile: NDArrayFloat,
        fixed_span_length_km: float | None,
    ) -> NDArrayFloat:
        """Evaluate the model's route-wide propagation for a uniform load."""
        physical_path = self.network_topology.build_optical_path(
            path, fixed_span_length_km=fixed_span_length_km
        )
        power_eval = to_convention(
            profile, self.launch_power_convention, self.eval_power_convention
        )
        return self.gn_evaluator.evaluate_path(physical_path, power_eval).snr_db

    @staticmethod
    def _has_uniform_loading(
        path: list[NodeID],
        first_link_state: LinkState,
        first_profile: NDArrayFloat,
        temp_power_profiles: dict[LinkKey, NDArrayFloat],
        network_state: NetworkState,
    ) -> bool:
        """Check that the same carriers and nominal powers occupy all links."""
        for index in range(1, len(path) - 1):
            state = network_state.get_link_state(path[index], path[index + 1])
            profile = temp_power_profiles.get(
                state.link_key, state.launch_power_profile_w
            )
            if not np.array_equal(first_profile, profile) or not np.array_equal(
                first_link_state.allocated_service_ids, state.allocated_service_ids
            ):
                return False
        return True


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
