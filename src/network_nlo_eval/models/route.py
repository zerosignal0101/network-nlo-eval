"""Shared optical-route accumulation and element post-processing.

Models differ in their single-span fiber physics. Amplifier gain and ASE,
ROADM losses and equalization, blocking, filtering, and SNR assembly are
handled here. The order of floating-point operations is part of the
numerical behavior: transfer accumulated noise first, then add new noise.
"""

# ruff: noqa: PLC0415

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, cast

import numpy as np

from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanContext,
    SpanFiberOutput,
    SpanKernel,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.network.elements import (
        EDFAConfig,
        FiberSpanConfig,
        ROADMConfig,
    )

__all__ = [
    "H_PLANCK",
    "accumulate_route",
    "apply_amplifier_and_roadm",
    "resolve_noise_figure",
]


class _ContextualSpanModel(Protocol):
    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput: ...


class _RouteSpanModel(Protocol):
    """The channel and span operations required by route accumulation."""

    @property
    def n_nodes(self) -> int: ...

    @property
    def f_abs_hz(self) -> NDArrayFloat: ...

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat: ...

    def blocked_nodes(self, roadm_config: ROADMConfig, /) -> list[int]: ...

    def power_width_scale(self) -> NDArrayFloat: ...

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        /,
    ) -> ElementOutput: ...


#: Planck constant in J s, used in the ASE expression.
H_PLANCK = 6.62607015e-34


def resolve_noise_figure(
    edfa_config: EDFAConfig | None,
    default_nf_lin: NDArrayFloat,
    n_nodes: int,
) -> NDArrayFloat:
    """Convert an EDFA configuration to per-channel linear noise figures.

    Parameters
    ----------
    edfa_config : EDFAConfig or None
        Span amplifier configuration; None means no amplifier.
    default_nf_lin : numpy.ndarray
        Default linear noise figures, used for the shape when no EDFA exists.
    n_nodes : int
        Number of channels or connections.

    Returns
    -------
    numpy.ndarray
        Linear noise figures for all nodes; ones when no EDFA is supplied.

    Raises
    ------
    ValueError
        If an array-valued noise figure does not have one entry per node.
    """
    if edfa_config is None:
        return np.ones_like(default_nf_lin)

    noise_figure_db = np.asarray(edfa_config.noise_figure_db, dtype=float)
    if noise_figure_db.ndim == 0:
        return np.full(n_nodes, 10 ** (float(noise_figure_db) / 10))
    if noise_figure_db.shape != (n_nodes,):
        msg = "EDFA noise-figure array must contain one value per channel."
        raise ValueError(msg)
    return 10 ** (noise_figure_db / 10)


def _validate_ase_polarization_scale(ase_polarization_scale: float) -> float:
    """Validate the scaling applied to newly generated ASE noise.

    Parameters
    ----------
    ase_polarization_scale : float
        Scaling relative to total dual-polarization ASE.

    Returns
    -------
    float
        Finite, positive scaling factor.

    Raises
    ------
    ValueError
        If the factor is non-finite or non-positive.
    """
    scale = float(ase_polarization_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        msg = f"ase_polarization_scale must be finite and positive, got {ase_polarization_scale!r}."
        raise ValueError(msg)
    return scale


def apply_amplifier_and_roadm(
    *,
    power_in_w: NDArrayFloat,
    fiber_output: SpanFiberOutput,
    edfa_config: EDFAConfig | None,
    roadm_config: ROADMConfig | None,
    nf_lin: NDArrayFloat,
    f_abs_hz: NDArrayFloat,
    bandwidth_hz: NDArrayFloat,
    input_ase_w: NDArrayFloat | None = None,
    ase_polarization_scale: float = 1.0,
    fwm_is_additive: bool = False,
) -> ElementOutput:
    """Apply amplifier and optional ROADM effects to one span's fiber output.

    Parameters
    ----------
    power_in_w : numpy.ndarray
        Per-channel span input power in watts.
    fiber_output : SpanFiberOutput
        Fiber-output signal and compensated NLI powers and amplitudes.
    edfa_config : EDFAConfig or None
        Span amplifier; None disables gain and newly generated ASE.
    roadm_config : ROADMConfig or None
        Optional ROADM at the span end.
    nf_lin, f_abs_hz, bandwidth_hz : numpy.ndarray
        Per-channel linear noise figure, absolute frequency, and bandwidth.
    input_ase_w : numpy.ndarray or None, optional
        Existing ASE entering the amplifier in the target power convention.
        It is amplified but never rescaled by ``ase_polarization_scale``.
    ase_polarization_scale : float, optional
        Scale for newly generated EDFA ASE: 1 for total dual-polarization
        power and 0.5 for per-polarization power.
    fwm_is_additive : bool, optional
        Propagate FWM separately only when the model has excluded it from XPM.

    Returns
    -------
    ElementOutput
        Span output power and generated or propagated noise.

    Raises
    ------
    ValueError
        If the ASE scaling is invalid or the gain-ripple array has wrong shape.

    Notes
    -----
    Kernels return NLI normalized to exact per-channel span-loss compensation.
    The output transfer is therefore ``gain / compensation_gain`` for powers
    and its square root for amplitudes. This is the ``Gamma * p_CUT(L)``
    factor in the PCFM paper. Without an amplifier it reduces to the fiber
    signal transfer. ROADM loss then attenuates signal and noise together.
    """
    ase_polarization_scale = _validate_ase_polarization_scale(ase_polarization_scale)
    power_out_fiber = fiber_output.power_out_w
    compensation_gain = np.divide(
        power_in_w,
        power_out_fiber,
        out=np.ones_like(power_in_w),
        where=power_out_fiber > 0,
    )
    spm = fiber_output.spm_noise_w
    xpm = fiber_output.xpm_noise_w
    amp_spm = getattr(fiber_output, "nli_amplitude_spm", None)
    amp_xpm = getattr(fiber_output, "nli_amplitude_xpm", None)
    # Propagate FWM only when explicitly separated from XPM, preventing
    # diagnostic subsets from being counted twice.
    fwm_extra = None
    if fwm_is_additive:
        raw_fwm = getattr(fiber_output, "fwm_noise_w", None)
        if raw_fwm is not None:
            fwm_extra = np.array(raw_fwm, dtype=np.float64, copy=True)

    if edfa_config is None:
        power_out = power_out_fiber
        nli_transfer = np.divide(
            power_out_fiber,
            power_in_w,
            out=np.zeros_like(power_in_w),
            where=power_in_w > 0,
        )
        ase = (
            np.zeros_like(power_in_w)
            if input_ase_w is None
            else np.array(input_ase_w, dtype=np.float64, copy=True)
        )
        ase[power_in_w <= 0] = 0
    else:
        if edfa_config.target_gain_db is None:
            gain = compensation_gain.copy()
        else:
            gain = np.full_like(power_in_w, 10 ** (edfa_config.target_gain_db / 10))
        if edfa_config.gain_ripple_db is not None:
            ripple = np.asarray(edfa_config.gain_ripple_db, dtype=float)
            if ripple.shape != bandwidth_hz.shape:
                msg = "EDFA gain-ripple array must contain one value per channel."
                raise ValueError(msg)
            gain *= 10 ** (ripple / 10)
        power_out = power_out_fiber * gain
        nli_transfer = gain / compensation_gain
        ase = nf_lin * H_PLANCK * f_abs_hz * bandwidth_hz * np.maximum(gain - 1, 0)
        if ase_polarization_scale != 1.0:
            # Avoid multiplication in the default total-power convention.
            ase = ase * ase_polarization_scale
        if input_ase_w is not None:
            # Input ASE already uses the caller's convention; do not rescale it.
            ase = ase + np.asarray(input_ase_w, dtype=np.float64) * gain
        ase[power_in_w <= 0] = 0

    if not np.all(nli_transfer == 1.0):
        spm = spm * nli_transfer
        xpm = xpm * nli_transfer
        if fwm_extra is not None:
            fwm_extra *= nli_transfer
        if amp_spm is not None:
            amplitude_transfer = np.sqrt(nli_transfer)
            amp_spm = amp_spm * amplitude_transfer
            amp_xpm = amp_xpm * amplitude_transfer

    if roadm_config is not None:
        loss_lin = 10 ** (roadm_config.path_loss_db() / 10)
        power_out /= loss_lin
        spm /= loss_lin
        xpm /= loss_lin
        ase /= loss_lin
        if fwm_extra is not None:
            fwm_extra /= loss_lin
        if amp_spm is not None:
            amp_root = np.sqrt(1.0 / loss_lin)
            amp_spm = amp_spm * amp_root
            amp_xpm = amp_xpm * amp_root

    if amp_spm is None:
        return ElementOutput(power_out, spm, xpm, ase, fwm_noise_w=fwm_extra)
    return ElementOutput(power_out, spm, xpm, ase, amp_spm, amp_xpm, fwm_extra)


def accumulate_route(
    model: _RouteSpanModel,
    optical_path: MultiSpanOpticalPath,
    power_in_w: NDArrayFloat,
    *,
    ase_polarization_scale: float = 1.0,
) -> PathQoTResult:
    """Accumulate span and ROADM effects along an optical path.

    Parameters
    ----------
    model : object
        Model supplying channel geometry and span evaluation.
    optical_path : MultiSpanOpticalPath
        Ordered path of physical link configurations.
    power_in_w : numpy.ndarray
        Non-negative input power per channel in watts.
    ase_polarization_scale : float, optional
        Polarization scale applied only to newly generated ROADM booster ASE.

    Returns
    -------
    PathQoTResult
        Per-channel signal, SPM, XPM, ASE, SNR, and component counts.

    Raises
    ------
    ValueError
        If input power dimensions or values are invalid, or ASE scaling is
        non-positive or non-finite.
    TypeError
        If the model mixes coherent amplitudes and power-only span results.

    Notes
    -----
    At each span, accumulated noise is first multiplied by power transfer,
    then newly generated noise is added. Coherent NLI amplitudes instead add
    as complex amplitudes, with transfer applied by its square root. ROADM
    filtering affects noise, and equalizing boosters add ASE.
    """
    ase_polarization_scale = _validate_ase_polarization_scale(ase_polarization_scale)
    power = np.asarray(power_in_w, dtype=np.float64).copy()
    if power.shape != (model.n_nodes,) or np.any(power < 0):
        msg = "power_in_w must be a non-negative vector with one value per channel."
        raise ValueError(msg)
    power = power * model.power_width_scale()

    total_spm = np.zeros_like(power)
    total_xpm = np.zeros_like(power)
    total_ase = np.zeros_like(power)
    # Optional additive FWM is separated from XPM by the model and
    # accumulated by power before being added back to total XPM.
    # The default ignores FWM diagnostics, avoiding double counting
    # when they are already included in XPM.
    fwm_is_additive = bool(getattr(model, "fwm_noise_is_additive", False))
    total_fwm = np.zeros_like(power)
    fwm_present = False
    # Sum supplied complex amplitudes coherently; otherwise sum powers.
    # None means no span has established the convention yet.
    coherent: bool | None = None
    amp_spm = np.zeros_like(power, dtype=np.complex128)
    amp_xpm = np.zeros_like(power, dtype=np.complex128)
    span_count = 0
    # Count spans for models that need context for inter-span phase.
    total_spans = sum(
        max(1, int(np.ceil(fiber.length_km / fiber.max_span_length_km)))
        for fiber in optical_path.fiber_configs
    )
    cumulative_m = 0.0
    wants_context = bool(getattr(model, "supports_span_context", False))
    roadm_configs: Sequence[ROADMConfig | None] = optical_path.roadm_configs or [
        None
    ] * len(optical_path.fiber_configs)
    hop_count = len(optical_path.fiber_configs)

    for hop_index, (fiber, edfa, roadm) in enumerate(
        zip(
            optical_path.fiber_configs,
            optical_path.edfa_configs,
            roadm_configs,
            strict=True,
        )
    ):
        remaining_km = fiber.length_km
        while remaining_km > 1e-12:
            span_km = min(remaining_km, fiber.max_span_length_km)
            previous_power = power
            if wants_context:
                step = cast("_ContextualSpanModel", model).span_step(
                    fiber,
                    edfa,
                    power,
                    span_km * 1e3,
                    span_context=SpanContext(
                        span_index=span_count,
                        total_spans=total_spans,
                        cumulative_length_m=cumulative_m,
                        span_length_m=span_km * 1e3,
                    ),
                )
            else:
                step = model.span_step(fiber, edfa, power, span_km * 1e3)
            power = step.power_out_w
            transfer = np.divide(
                power,
                previous_power,
                out=np.zeros_like(power),
                where=previous_power > 0,
            )
            step_amp = getattr(step, "nli_amplitude_spm", None)
            if step_amp is None:
                if coherent:
                    msg = (
                        "model returned a coherent NLI amplitude for an earlier span "
                        "and a power-only result for a later one; a model must pick one convention."
                    )
                    raise TypeError(msg)
                coherent = False
                total_spm *= transfer
                total_xpm *= transfer
                total_ase *= transfer
                total_spm += step.spm_noise_w
                total_xpm += step.xpm_noise_w
                total_ase += step.ase_noise_w
                if fwm_is_additive and fwm_present:
                    total_fwm *= transfer
                step_fwm = (
                    getattr(step, "fwm_noise_w", None) if fwm_is_additive else None
                )
                if step_fwm is not None:
                    total_fwm += step_fwm
                    fwm_present = True
            else:
                if coherent is False:
                    msg = (
                        "model returned a power-only NLI result for an earlier span "
                        "and a coherent amplitude for a later one; a model must pick one convention."
                    )
                    raise TypeError(msg)
                coherent = True
                amp_root = np.sqrt(transfer)
                amp_spm = amp_spm * amp_root + step_amp
                amp_xpm = amp_xpm * amp_root + step.nli_amplitude_xpm
                total_ase *= transfer
                total_ase += step.ase_noise_w
                if fwm_is_additive:
                    if fwm_present:
                        total_fwm *= transfer
                    step_fwm = getattr(step, "fwm_noise_w", None)
                    if step_fwm is not None:
                        total_fwm += step_fwm
                        fwm_present = True
            span_count += 1
            cumulative_m += span_km * 1e3
            remaining_km -= span_km

        if roadm is None:
            continue
        blocked = model.blocked_nodes(roadm)
        operation: Literal["drop", "express"] = (
            "drop" if hop_index == hop_count - 1 else "express"
        )
        loss_db = roadm.path_loss_db(operation)
        loss = 10 ** (loss_db / 10)
        power /= loss
        if coherent:
            amp_root = np.sqrt(1.0 / loss)
            amp_spm *= amp_root
            amp_xpm *= amp_root
        else:
            total_spm /= loss
            total_xpm /= loss
        total_ase /= loss
        if fwm_present:
            total_fwm /= loss
        if roadm.equalize_output_power:
            gain = loss
            power *= gain
            if coherent:
                amp_root = np.sqrt(gain)
                amp_spm *= amp_root
                amp_xpm *= amp_root
            else:
                total_spm *= gain
                total_xpm *= gain
            total_ase *= gain
            if fwm_present:
                total_fwm *= gain
            nf = 10 ** (roadm.booster_noise_figure_db / 10)
            booster_ase = (
                nf
                * H_PLANCK
                * model.f_abs_hz
                * model.channel_bandwidth_hz_channels
                * (gain - 1)
            )
            if ase_polarization_scale != 1.0:
                # Scale only newly generated booster ASE; accumulated ASE
                # has already followed the booster gain above.
                booster_ase = booster_ase * ase_polarization_scale
            booster_ase[power <= 0] = 0
            total_ase += booster_ase
        penalty = 10 ** (roadm.filtering_penalty_db / 10)
        if coherent:
            amp_root = np.sqrt(penalty)
            amp_spm *= amp_root
            amp_xpm *= amp_root
        else:
            total_spm *= penalty
            total_xpm *= penalty
        total_ase *= penalty
        if fwm_present:
            total_fwm *= penalty
        if blocked:
            power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[
                blocked
            ] = 0
            if fwm_present:
                total_fwm[blocked] = 0
            if coherent:
                amp_spm[blocked] = 0
                amp_xpm[blocked] = 0

    if coherent:
        total_spm = cast("NDArrayFloat", np.abs(amp_spm) ** 2)
        total_xpm = cast("NDArrayFloat", np.abs(amp_xpm) ** 2)
    if fwm_present:
        # Accumulate separated FWM by power, then include it in path XPM.
        # Path-level XPM therefore retains its established FWM-inclusive meaning.
        total_xpm = total_xpm + total_fwm

    total_noise = total_spm + total_xpm + total_ase
    snr = np.full_like(power, -np.inf)
    active = (power > 0) & (total_noise > 0)
    snr[active] = 10 * np.log10(power[active] / total_noise[active])
    snr[(power > 0) & (total_noise == 0)] = np.inf
    return PathQoTResult(
        power,
        total_spm,
        total_xpm,
        total_ase,
        snr,
        span_count,
        sum(roadm is not None for roadm in roadm_configs),
    )


class RouteModel:
    """Base class for common route processing and cached span kernels.

    Subclasses implement ``_build_kernel`` and define model information.
    Kernels are cached by the fiber configuration's JSON representation.
    """

    info: ModelInfo  # Subclasses supply model metadata.

    supports_distributed_raman = False
    supports_lumped_loss = False

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        slots_per_channel: int | None = None,
        channel_bandwidth_source: str = "symbol_rate",
        power_width_mode: str = "fixed_power",
    ) -> None:
        """Initialize channel or connection geometry for route evaluation.

        Parameters
        ----------
        grid : SpectrumGrid or FlexibleGrid
            Fixed spectrum grid, or flexible grid when connections are supplied.
        ref_fiber_config : FiberSpanConfig
            Reference fiber configuration.
        symbol_rate_hz : float, optional
            Symbol rate in Hz.
        connections : sequence of ConnectionSpec or None, optional
            Connections for flexible-grid mode.
        slots_per_channel : int or None, optional
            Slot count for expanding legacy ROADM channel blocks.
        channel_bandwidth_source : {"symbol_rate", "occupied"}, optional
            Source of per-connection bandwidths.
        power_width_mode : {"fixed_power", "fixed_psd"}, optional
            Input-power scaling convention.
        """
        from network_nlo_eval.core.flexgrid import (
            FlexibleGrid,
        )
        from network_nlo_eval.core.spectrum import SpectrumGrid

        if channel_bandwidth_source not in ("symbol_rate", "occupied"):
            msg = f"channel_bandwidth_source must be 'symbol_rate' or 'occupied', got {channel_bandwidth_source!r}."
            raise ValueError(msg)
        if power_width_mode not in ("fixed_power", "fixed_psd"):
            msg = f"power_width_mode must be 'fixed_power' or 'fixed_psd', got {power_width_mode!r}."
            raise ValueError(msg)
        if slots_per_channel is not None and slots_per_channel < 1:
            msg = f"slots_per_channel must be >= 1, got {slots_per_channel}."
            raise ValueError(msg)

        self._validate_fiber_scenario(ref_fiber_config)

        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.connections = list(connections) if connections is not None else None
        self.slots_per_channel = slots_per_channel
        self.channel_bandwidth_source = channel_bandwidth_source
        self.power_width_mode = power_width_mode

        if self.connections is not None:
            if not isinstance(grid, FlexibleGrid):
                msg = "connections= requires a FlexibleGrid."
                raise TypeError(msg)
            self._f_abs_hz = np.array(
                [conn.centre_hz for conn in self.connections], dtype=np.float64
            )
            self._lambdas_m = 2.99792458e8 / self._f_abs_hz
        elif isinstance(grid, SpectrumGrid):
            self._lambdas_m = grid.wavelengths
            self._f_abs_hz = grid.frequencies
        else:
            msg = "Legacy mode requires a SpectrumGrid; pass connections= for a FlexibleGrid."
            raise TypeError(msg)

        self._kernel_cache: dict[str, SpanKernel] = {}

    # --- Geometry ----------------------------------------------------------
    @property
    def ase_polarization_scale(self) -> float:
        """Return the scale applied to newly generated ASE.

        The default is 1 for total dual-polarization power. Models using
        per-polarization power may override it with 0.5.
        """
        return 1.0

    @property
    def n_nodes(self) -> int:
        """Return the number of channels or connections."""
        return int(self._f_abs_hz.shape[0])

    @property
    def f_abs_hz(self) -> NDArrayFloat:
        """Return the per-node absolute center frequencies in Hz."""
        return self._f_abs_hz

    @property
    def lambdas_m(self) -> NDArrayFloat:
        """Return the per-node center wavelengths in meters."""
        return self._lambdas_m

    @property
    def band_center_hz(self) -> float:
        """Return the frequency origin for relative frequencies in Hz."""
        from network_nlo_eval.core.flexgrid import (
            FlexibleGrid,
        )

        if isinstance(self.grid, FlexibleGrid):
            return self.grid.band_center_hz
        return self.grid.center_frequency_hz

    @property
    def f_rel_hz(self) -> NDArrayFloat:
        """Return the per-node relative center frequencies in Hz."""
        return self._f_abs_hz - self.band_center_hz

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """Return per-node channel bandwidths in Hz.

        Raises
        ------
        ValueError
            If a selected bandwidth is non-finite or non-positive.
        """
        if self.connections is None:
            return np.full(self.n_nodes, float(self.symbol_rate_hz), dtype=np.float64)
        if self.channel_bandwidth_source == "occupied":
            values = np.array(
                [float(conn.occupied_bandwidth_hz) for conn in self.connections],
                dtype=np.float64,
            )
        else:
            values = np.array(
                [float(conn.symbol_rate_hz) for conn in self.connections],
                dtype=np.float64,
            )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            msg = (
                f"per-channel bandwidth B_i (source={self.channel_bandwidth_source!r}) must be "
                f"finite and positive; got {values.tolist()}."
            )
            raise ValueError(msg)
        return values

    def power_width_scale(self) -> NDArrayFloat:
        """Return per-node power scaling for fixed power or fixed PSD."""
        if self.power_width_mode == "fixed_power":
            return np.ones(self.n_nodes, dtype=np.float64)
        return self.channel_bandwidth_hz_channels / float(self.symbol_rate_hz)

    # --- Blocking ----------------------------------------------------------
    def blocked_nodes(self, roadm_config: ROADMConfig) -> list[int]:
        """Return indices blocked by a ROADM.

        In connection mode, check slot overlap and expand blocked legacy channels
        using ``slots_per_channel``. Both blocked-slot and channel lists contribute.
        A channel list requires a configured slot count.
        """
        from network_nlo_eval.network.slots import (
            connection_overlaps_slots,
        )

        if self.connections is None:
            return [
                index for index in roadm_config.blocked_channels if index < self.n_nodes
            ]

        blocked_slots = set(roadm_config.blocked_slots)
        if roadm_config.blocked_channels:
            if not self.slots_per_channel:
                msg = (
                    "Per-connection mode needs slots_per_channel "
                    "(channel_spacing_hz / slot_width_hz) to expand ROADM "
                    "blocked_channels into slots. Construct the evaluator with "
                    "slots_per_channel=..., or use blocked_slots."
                )
                raise ValueError(msg)
            for channel_index in roadm_config.blocked_channels:
                blocked_slots.update(
                    range(
                        channel_index * self.slots_per_channel,
                        (channel_index + 1) * self.slots_per_channel,
                    )
                )
        return [
            index
            for index, conn in enumerate(self.connections)
            if connection_overlaps_slots(conn.first_slot, conn.n_slots, blocked_slots)
        ]

    def roadm_blocks_node(self, roadm_config: ROADMConfig, node_index: int) -> bool:
        """Check whether one channel or connection is blocked by a ROADM.

        Parameters
        ----------
        roadm_config : ROADMConfig
            ROADM blocking configuration.
        node_index : int
            Channel or connection index.

        Returns
        -------
        bool
            Whether the specified channel or connection is blocked.

        Raises
        ------
        ValueError
            If legacy blocked channels need expansion but slots_per_channel is absent.
        """
        from network_nlo_eval.network.slots import (
            connection_overlaps_slots,
        )

        if self.connections is None:
            return node_index in roadm_config.blocked_channels
        if node_index >= len(self.connections):
            return False

        blocked_slots = set(roadm_config.blocked_slots)
        if roadm_config.blocked_channels:
            if not self.slots_per_channel:
                msg = (
                    "Per-connection mode needs slots_per_channel "
                    "(channel_spacing_hz / slot_width_hz) to expand ROADM "
                    "blocked_channels into slots. Construct the evaluator with "
                    "slots_per_channel=..., or use blocked_slots."
                )
                raise ValueError(msg)
            for channel_index in roadm_config.blocked_channels:
                blocked_slots.update(
                    range(
                        channel_index * self.slots_per_channel,
                        (channel_index + 1) * self.slots_per_channel,
                    )
                )
        conn = self.connections[node_index]
        return connection_overlaps_slots(conn.first_slot, conn.n_slots, blocked_slots)

    # --- Kernels -----------------------------------------------------------
    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Build a single-span kernel for a fiber; subclasses must implement this."""
        msg = "RouteModel subclasses must implement _build_kernel()."
        raise NotImplementedError(msg)

    def _validate_fiber_scenario(self, fiber_config: FiberSpanConfig) -> None:
        """Validate the model's pump and in-span loss capabilities."""
        if not self.supports_distributed_raman and any(
            pump.power_w > 0 for pump in fiber_config.raman_pumps
        ):
            msg = "This model does not support active Raman pumps; select a distributed-Raman model."
            raise ValueError(msg)
        if fiber_config.lumped_loss and not self.supports_lumped_loss:
            msg = "This model does not support in-span lumped losses; select pcfm or pcfm_h3_raman."
            raise ValueError(msg)

    def kernel_for(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Return a cached kernel for the supplied fiber configuration."""
        self._validate_fiber_scenario(fiber_config)
        key = fiber_config.model_dump_json()
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self._build_kernel(fiber_config)
            self._kernel_cache[key] = kernel
        return kernel

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """Evaluate fiber, amplifier, and ASE for a span without a ROADM."""
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate(power_in_w, span_length_m, 1)
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        return apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            ase_polarization_scale=self.ase_polarization_scale,
        )

    @property
    def default_nf_lin(self) -> NDArrayFloat:
        """Return default linear noise figures; subclasses may override them."""
        return np.ones(self.n_nodes, dtype=np.float64)

    # --- Paths -------------------------------------------------------------
    def evaluate_path(
        self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat
    ) -> PathQoTResult:
        """Evaluate the optical path and return per-channel results."""
        return accumulate_route(
            self,
            optical_path,
            power_in_w,
            ase_polarization_scale=self.ase_polarization_scale,
        )
