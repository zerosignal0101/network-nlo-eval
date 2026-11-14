"""Topology-aware waveform QoT evaluation using ``physics-nlo-eval``."""

# ruff: noqa: PLC0415  # Load waveform dependencies only when requested.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
from network_nlo_eval.models.numba_kernels import (
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _w_to_dbm,
)
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)
from network_nlo_eval.network.slots import connection_overlaps_slots
from network_nlo_eval.physics.metrics import compute_snr_from_evm
from network_nlo_eval.rwa.qot_checker import fold_roadm_terms
from network_nlo_eval.rwa.validity import LinkValidity, LinkValidityPolicy

__all__ = [
    "PyNLOQoTEvaluator",
    "WaveformQoTEvaluator",
    "apply_link_validity",
]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.flexgrid import ConnectionSpec
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat, NodeID
    from network_nlo_eval.network.elements import ROADMConfig
    from network_nlo_eval.network.topology import NetworkTopology


def apply_link_validity(
    metrics: dict[str, NDArrayFloat],
    validity: LinkValidity,
    policy: LinkValidityPolicy,
) -> dict[str, NDArrayFloat]:
    """Apply an explicit link validity policy to waveform metrics.

    Parameters
    ----------
    metrics : dict
        Waveform metrics containing an SNR array.
    validity : LinkValidity
        Validity report for the evaluated link and spectrum.
    policy : LinkValidityPolicy
        Explicit policy governing SNR and validity adjustments.

    Returns
    -------
    dict
        A new metrics dictionary. Enabled policies add validity and margin arrays;
        off mode retains the original arrays without modification.

    Raises
    ------
    KeyError
        If SNR is missing.
    ValueError
        If the SNR shape does not match the report.
    """
    snr = np.asarray(metrics["snr"], dtype=np.float64)
    if snr.shape != validity.valid.shape:
        msg = f"snr shape {snr.shape} does not match the report {validity.valid.shape}."
        raise ValueError(msg)
    if not policy.enabled:
        return dict(metrics)
    adjusted = dict(metrics)
    adjusted["snr"] = validity.adjusted_snr_db(snr, policy)
    adjusted["valid"] = np.array(
        validity.effective_valid(policy), dtype=bool, copy=True
    )
    adjusted["margin_db"] = np.array(validity.margin_db, dtype=np.float64, copy=True)
    adjusted["soft_margin_db"] = np.array(
        policy.scale_soft_margin(validity.soft_margin_db), dtype=np.float64, copy=True
    )
    return adjusted


def _load_physical_layer() -> dict[str, Any]:
    """Import the optional numerical simulator only when waveform QoT is requested."""
    try:
        from physics_nlo_eval.channel.fiber import EDFA, FiberChannel, IdealAmplifier
        from physics_nlo_eval.channel.network import ROADM, PumpSpec, RamanPump
        from physics_nlo_eval.channel.wdm import (
            ChannelExtractor,
            ChannelSpec,
            WDMDemux,
            WDMMux,
            WDMPlan,
        )
        from physics_nlo_eval.dsp.carrier import PhaseNoiseCompensation
        from physics_nlo_eval.dsp.time_domain import (
            ChromaticDispersionComp,
            TimingRecovery,
        )
        from physics_nlo_eval.metrics.ber_evm import BERCalculator, Demapper
        from physics_nlo_eval.tx.mapper import Mapper
        from physics_nlo_eval.tx.optics import IQModulator
        from physics_nlo_eval.tx.rrc import RRCFilter
    except ImportError as exc:
        msg = "Waveform comparison requires physics-nlo-eval installed in the same Python environment."
        raise ImportError(msg) from exc
    # Explicit registry defines the contract with the waveform engine.
    return {
        "BERCalculator": BERCalculator,
        "ChannelExtractor": ChannelExtractor,
        "ChannelSpec": ChannelSpec,
        "ChromaticDispersionComp": ChromaticDispersionComp,
        "Demapper": Demapper,
        "EDFA": EDFA,
        "FiberChannel": FiberChannel,
        "IQModulator": IQModulator,
        "IdealAmplifier": IdealAmplifier,
        "Mapper": Mapper,
        "PhaseNoiseCompensation": PhaseNoiseCompensation,
        "PumpSpec": PumpSpec,
        "ROADM": ROADM,
        "RRCFilter": RRCFilter,
        "RamanPump": RamanPump,
        "TimingRecovery": TimingRecovery,
        "WDMDemux": WDMDemux,
        "WDMMux": WDMMux,
        "WDMPlan": WDMPlan,
    }


class PyNLOQoTEvaluator:
    """Numerically evaluate the same ordered route used by the statistical model.

    The historical name is retained for API compatibility. The physical
    package runs the Manakov solver through PyNLO, which requires a CuPy
    compute context for dual-polarization WDM propagation.

    ``launch_power_convention`` declares how ``launch_power_profile_w`` is
    interpreted (``"total"``, the default network reference plane, or
    ``"per_pol"``).  The waveform engine itself always works in total dual-
    polarization power, so :meth:`create_wdm_signal` normalises the input to
    that coordinate exactly once before the existing ``/n_pol`` split for the
    IQ modulator; a per-polarization input is never halved twice.
    """

    def __init__(
        self,
        spectrum_grid: SpectrumGrid,
        modulation: str = "DP-QPSK",
        symbol_rate_hz: float = 50e9,
        sps: int = 4,
        rrc_roll_off: float = 0.1,
        seed: int = 42,
        compute_ctx_name: str = "cupy",
        pilot_info: dict[str, Any] | None = None,
        connections: Sequence[ConnectionSpec] | None = None,
        slots_per_channel: int | None = None,
        *,
        launch_power_convention: str = "total",
    ) -> None:
        """Initialize the topology-aware waveform evaluator.

        The default CuPy context is required by the dual-polarization Manakov solver.
        Input launch power follows the declared total or per-polarization convention.

        Raises
        ------
        ValueError
            If symbol rate, samples per symbol, or power convention is invalid.
        """
        if sps <= 0 or symbol_rate_hz <= 0:
            msg = "Symbol rate and samples per symbol must be positive."
            raise ValueError(msg)
        self.launch_power_convention = resolve_launch_power_convention(
            launch_power_convention
        )
        self.physical = _load_physical_layer()
        self.spectrum_grid = spectrum_grid
        self.connections = list(connections) if connections is not None else None
        self.slots_per_channel = slots_per_channel
        self.modulation = modulation
        self.symbol_rate_hz = symbol_rate_hz
        self.sps = sps
        self.rrc_roll_off = rrc_roll_off
        self.seed = seed
        self.compute_ctx_name = compute_ctx_name
        self.pilot_info = pilot_info

        self.mapper = self.physical["Mapper"](
            modulation=modulation,
            power_norm=True,
            pilot_info=pilot_info,
            sym_rate=symbol_rate_hz,
        )
        self.tx_rrc = self.physical["RRCFilter"](
            sps=sps, roll_off=rrc_roll_off, is_matched_filter=False
        )
        self.rx_rrc = self.physical["RRCFilter"](
            sps=sps, roll_off=rrc_roll_off, is_matched_filter=True
        )
        self.demapper = self.physical["Demapper"](
            modulation=modulation, power_norm=True
        )
        if self.connections is not None:
            # Flexible connections use their individual center frequencies.
            self.frequencies_hz = np.array(
                [conn.centre_hz for conn in self.connections], dtype=float
            )
            self.center_freq_hz = (
                self.connections[0].grid.band_center_hz
                if self.connections
                else spectrum_grid.center_frequency_hz
            )
        else:
            self.frequencies_hz = spectrum_grid.frequencies
            self.center_freq_hz = spectrum_grid.center_frequency_hz
        self.channels_spec = [
            self.physical["ChannelSpec"](index, frequency - self.center_freq_hz)
            for index, frequency in enumerate(self.frequencies_hz)
        ]
        self.wdm_plan = self.physical["WDMPlan"](
            self.center_freq_hz, self.channels_spec
        )
        # Use actual connection edges to determine flexible-grid occupied width.
        if self.connections is not None:
            low = min(
                c.centre_hz - c.occupied_bandwidth_hz / 2 for c in self.connections
            )
            high = max(
                c.centre_hz + c.occupied_bandwidth_hz / 2 for c in self.connections
            )
            occupied_width = high - low
            # The default WSS passband must accommodate the widest connection.
            self._default_passband_hz = max(
                c.occupied_bandwidth_hz for c in self.connections
            )
        else:
            occupied_width = (
                spectrum_grid.num_channels - 1
            ) * spectrum_grid.channel_spacing_hz
            self._default_passband_hz = spectrum_grid.channel_spacing_hz
        self._mux_target_sr = max(
            symbol_rate_hz * sps,
            1.25 * (occupied_width + symbol_rate_hz * (1 + rrc_roll_off)),
        )

    def _is_blocked(self, roadm: ROADMConfig, node_index: int) -> bool:
        """Check channel blocking, using slot overlap for flexible connections.

        Blocked channels are expanded into slots when necessary; this requires
        slots_per_channel to be supplied.
        """
        if self.connections is not None:
            if node_index >= len(self.connections):
                return False
            blocked_slots = set(roadm.blocked_slots)
            if roadm.blocked_channels:
                if not self.slots_per_channel:
                    msg = (
                        "Per-connection mode needs slots_per_channel to expand ROADM "
                        "blocked_channels into slots; or use blocked_slots."
                    )
                    raise ValueError(msg)
                per_channel = self.slots_per_channel
                for channel_index in roadm.blocked_channels:
                    blocked_slots.update(
                        range(
                            channel_index * per_channel,
                            (channel_index + 1) * per_channel,
                        )
                    )
            conn = self.connections[node_index]
            return connection_overlaps_slots(
                conn.first_slot, conn.n_slots, blocked_slots
            )
        return node_index in roadm.blocked_channels

    def create_wdm_signal(
        self, launch_power_profile_w: NDArrayFloat, num_symbols: int = 1024
    ) -> tuple[Any, list[NDArrayBool]]:
        """Create a dual-polarization WDM signal using the configured power convention.

        Power is converted to total power before division between polarizations.
        """
        powers = np.asarray(launch_power_profile_w, dtype=float)
        if powers.shape != (self.spectrum_grid.num_channels,) or np.any(powers < 0):
            msg = "Launch power must contain one non-negative value per channel."
            raise ValueError(msg)
        powers = to_convention(powers, self.launch_power_convention, "total")
        rng = np.random.default_rng(self.seed)
        channel_signals: list[Any] = []
        all_bits: list[NDArrayBool] = []
        for channel_idx in range(self.spectrum_grid.num_channels):
            total_bits = num_symbols * self.mapper.bits_per_symbol * self.mapper.n_pol
            bits = rng.integers(0, 2, size=total_bits, dtype=np.int8).astype(bool)
            all_bits.append(bits)
            electrical = self.tx_rrc.forward(self.mapper.forward(bits))
            # Network launch power is total DP power; IQModulator expects per-pol power.
            per_pol_power = powers[channel_idx] / self.mapper.n_pol
            optical = self.physical["IQModulator"](
                launch_power_dbm_per_pol=_w_to_dbm(per_pol_power),
                wavelength_nm=C_LIGHT / self.frequencies_hz[channel_idx] * 1e9,
            ).forward(electrical)
            channel_signals.append(optical)
        mux = self.physical["WDMMux"](
            self.wdm_plan, target_sample_rate_hz=self._mux_target_sr
        )
        return mux.combine(channel_signals), all_bits

    @staticmethod
    def _fiber_dispersion(
        fiber: Any, center_frequency_hz: float
    ) -> tuple[float, float]:
        """Convert network dispersion parameters to solver beta2 and beta3 units."""
        wavelength_m = C_LIGHT / center_frequency_hz
        beta2 = _calc_beta2_from_d(fiber.dispersion_parameter_d, wavelength_m) * 1e27
        beta3 = (
            _calc_beta3_from_s_d(
                fiber.dispersion_slope_s, fiber.dispersion_parameter_d, wavelength_m
            )
            * 1e39
        )
        return float(beta2), float(beta3)

    @staticmethod
    def _edfa_values(edfa: Any) -> tuple[float | None, float]:
        """Return scalar gain and noise figure supported by the physical EDFA."""
        noise_figure = np.asarray(edfa.noise_figure_db)
        if noise_figure.ndim != 0 or edfa.gain_ripple_db is not None:
            msg = "Waveform comparison currently requires scalar EDFA noise figure and no gain-ripple array."
            raise ValueError(msg)
        return edfa.target_gain_db, float(noise_figure)

    def propagate_path(
        self, wdm_signal: Any, optical_path: MultiSpanOpticalPath
    ) -> Any:
        """Propagate the signal through fibers, amplifiers, Raman pumps and ROADMs."""
        current = wdm_signal
        roadms = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
        channel_info = current.ctx.meta["wdm_channels_info"]
        span_sequence = 0
        active_channels = np.flatnonzero(
            [
                entry.get("original_ctx").meta.get("tx_power_per_pol_w", 0) > 0
                for entry in channel_info
            ]
        ).tolist()
        for hop_index, (fiber, edfa, roadm) in enumerate(
            zip(
                optical_path.fiber_configs,
                optical_path.edfa_configs,
                roadms,
                strict=True,
            )
        ):
            if fiber.use_attenuation_polynomial:
                msg = "Waveform comparison currently requires constant attenuation_db_km_ref per fiber link."
                raise ValueError(msg)
            beta2, beta3 = self._fiber_dispersion(fiber, current.ctx.center_freq)
            target_gain_db, noise_figure_db = self._edfa_values(edfa)
            remaining_km = fiber.length_km
            while remaining_km > 1e-12:
                span_km = min(remaining_km, fiber.max_span_length_km)
                channel = self.physical["FiberChannel"](
                    length_m=span_km * 1e3,
                    alpha_db_per_km=fiber.attenuation_db_km_ref,
                    beta2_ps2_per_km=beta2,
                    beta3_ps3_per_km=beta3,
                    coarse_step_size_m=min(2000.0, span_km * 1e3),
                    n2_si=fiber.nonlinear_index_n2,
                    a_eff_um2=fiber.effective_area_um2_ref,
                    slope_a_eff_um2_per_nm=fiber.aeff_slope_um2_nm,
                    ref_lambda_nm=fiber.reference_wavelength_nm,
                    compute_ctx_name=self.compute_ctx_name,
                )
                current = channel.forward(current)
                if fiber.raman_pumps:
                    pumps = [
                        self.physical["PumpSpec"](**pump.model_dump())
                        for pump in fiber.raman_pumps
                    ]
                    current = self.physical["RamanPump"](
                        pumps, span_km, seed=self.seed + span_sequence
                    ).forward(current)
                if target_gain_db is None:
                    current = self.physical["IdealAmplifier"](
                        nf_db=noise_figure_db, default_power_dbm=0.0
                    ).forward(current)
                else:
                    current = self.physical["EDFA"](
                        gain_db=target_gain_db,
                        nf_db=noise_figure_db,
                        seed=self.seed + span_sequence,
                    ).forward(current)
                remaining_km -= span_km
                span_sequence += 1

            if roadm is None:
                continue
            allowed = [
                index for index in active_channels if not self._is_blocked(roadm, index)
            ]
            # Share ROADM port and loss semantics with the statistical evaluator.
            terms = fold_roadm_terms(
                roadm, is_last_hop=hop_index == len(optical_path.fiber_configs) - 1
            )
            current = self.physical["ROADM"](
                pass_channels=allowed,
                passband_hz=roadm.passband_hz or self._default_passband_hz,
                # Use the dB loss computed while folding ROADM terms.
                insertion_loss_db=terms.loss_db,
                filter_order=roadm.filter_order,
            ).forward(current)
            if terms.equalize:
                current = self.physical["EDFA"](
                    gain_db=terms.loss_db,
                    nf_db=terms.booster_nf_db,
                    seed=self.seed + hop_index + 1,
                ).forward(current)
        return current

    def _receiver_metrics(
        self,
        rx_wdm: Any,
        tx_bits: list[NDArrayBool],
        optical_path: MultiSpanOpticalPath,
        active_channels: NDArrayBool,
    ) -> dict[str, NDArrayFloat]:
        """Compensate path dispersion and calculate BER, EVM and SNR per active channel."""
        compensated = rx_wdm
        for fiber in optical_path.fiber_configs:
            beta2, beta3 = self._fiber_dispersion(fiber, compensated.ctx.center_freq)
            if beta2 == 0 and beta3 == 0:
                continue
            compensated = self.physical["ChromaticDispersionComp"](
                fiber_length_km=fiber.length_km,
                beta2_ps2_per_km=beta2,
                beta3_ps3_per_km=beta3,
            ).forward(compensated)
        demuxed = self.physical["WDMDemux"](filter_type="ideal_rectangular").forward(
            compensated
        )
        ber = np.full(self.spectrum_grid.num_channels, np.nan)
        evm = np.full(self.spectrum_grid.num_channels, np.nan)
        snr = np.full(self.spectrum_grid.num_channels, np.nan)
        calculator = self.physical["BERCalculator"]()
        for index in range(self.spectrum_grid.num_channels):
            if not active_channels[index]:
                continue
            channel = self.physical["ChannelExtractor"](channel_idx=index).forward(
                demuxed
            )
            channel = self.rx_rrc.forward(channel)
            channel = self.physical["TimingRecovery"](sps_in=channel.ctx.sps).forward(
                channel
            )
            method = "pilot_assisted" if self.pilot_info is not None else "BPS"
            kwargs = {"method": method, "constellation": self.demapper.constellation}
            if method == "BPS":
                kwargs.update({"n_test_phases": 256, "block_size": 256})
            channel = self.physical["PhaseNoiseCompensation"](**kwargs).forward(channel)
            received_bits = self.demapper.forward(channel)
            evm[index] = (
                calculator.calculate_evm(
                    channel, self.demapper.constellation, tx_bits[index]
                )
                * 0.01
            )
            ber[index] = calculator.calculate_ber(received_bits, tx_bits[index])
            snr[index] = compute_snr_from_evm(evm[index])
        return {"ber": ber, "evm": evm, "snr": snr}

    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        launch_power_profile_w: NDArrayFloat,
        num_symbols: int = 1024,
        *,
        validity: LinkValidity | None = None,
        validity_policy: LinkValidityPolicy | None = None,
    ) -> tuple[Any, dict[str, NDArrayFloat]]:
        """Evaluate waveform QoT on an ordered optical path.

        Parameters
        ----------
        optical_path : MultiSpanOpticalPath
            Ordered physical path.
        launch_power_profile_w : numpy.ndarray
            Input launch powers in the configured power convention.
        num_symbols : int, optional
            Symbols per channel.
        validity : LinkValidity or None, optional
            Optional link validity report; requires an explicit policy.
        validity_policy : LinkValidityPolicy or None, optional
            Policy applied to waveform metrics.

        Returns
        -------
        tuple
            Received signal and metrics dictionary.

        Raises
        ------
        ValueError
            If a validity report is provided without a policy.
        """
        if validity is not None and validity_policy is None:
            msg = "validity requires an explicit validity_policy; nothing is applied implicitly."
            raise ValueError(msg)
        signal, bits = self.create_wdm_signal(
            launch_power_profile_w, num_symbols=num_symbols
        )
        received = self.propagate_path(signal, optical_path)
        active = np.asarray(launch_power_profile_w) > 0
        for roadm in optical_path.roadm_configs or []:
            active[
                [index for index in roadm.blocked_channels if index < len(active)]
            ] = False
        metrics = self._receiver_metrics(received, bits, optical_path, active)
        if validity is not None:
            metrics = apply_link_validity(
                metrics, validity, validity_policy or LinkValidityPolicy()
            )
        return received, metrics

    def evaluate_topology_path(
        self,
        topology: NetworkTopology,
        path: list[NodeID],
        launch_power_profile_w: NDArrayFloat,
        num_symbols: int = 1024,
        *,
        validity: LinkValidity | None = None,
        validity_policy: LinkValidityPolicy | None = None,
    ) -> tuple[Any, dict[str, NDArrayFloat]]:
        """Build a physical path from topology nodes and evaluate waveform QoT.

        Parameters
        ----------
        topology : NetworkTopology
            Network topology.
        path : list of int
            Ordered node path.
        launch_power_profile_w : numpy.ndarray
            Input powers in the configured convention.
        num_symbols : int, optional
            Symbols per channel.
        validity : LinkValidity or None, optional
            Optional link validity report.
        validity_policy : LinkValidityPolicy or None, optional
            Explicit validity policy.
        """
        return self.evaluate_path(
            topology.build_optical_path(path),
            launch_power_profile_w,
            num_symbols,
            validity=validity,
            validity_policy=validity_policy,
        )

    def evaluate_channel_evm_and_ber(
        self,
        launch_power_profile_w: NDArrayFloat,
        fiber_length_km: float = 100.0,
        num_spans: int = 1,
        amplifier_nf_db: float = 5.5,
    ) -> tuple[Any, dict[str, NDArrayFloat]]:
        """Evaluate a uniform single link using the legacy API."""
        from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

        path = MultiSpanOpticalPath(
            fiber_configs=[
                FiberSpanConfig(length_km=fiber_length_km) for _ in range(num_spans)
            ],
            edfa_configs=[
                EDFAConfig(noise_figure_db=amplifier_nf_db) for _ in range(num_spans)
            ],
        )
        return self.evaluate_path(path, launch_power_profile_w)


WaveformQoTEvaluator = PyNLOQoTEvaluator
