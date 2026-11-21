"""Multiband inter-channel stimulated Raman scattering GN evaluator.

The ``ecz`` registry entry implements the ECZ single-span closed-form
model and accumulates noise powers incoherently across spans. Its
multispan results should not be interpreted as including inter-span
coherence. The separate ``ecz_coherent`` model adds complex-amplitude
accumulation and is the appropriate entry for coherent multispan use.

The power profile and single-span SPM/XPM terms follow Lasagni 2023
Eqs. (18), (20), and (29)-(32). The shaping profile assumes uniform launched
PSD; using it for sparse or unequal-power loading is an approximation.
The paper's dimensionless ``k_bar`` is launch pre-emphasis, not Raman pump
power. These ECZ models use zero pre-emphasis and reject active Raman pumps.
"""

from collections.abc import Sequence

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import (
    CarrierGeometry,
    ConnectionGeometry,
    FlexibleGrid,
)
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanContext,
    SpanFiberOutput,
    register_model,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_fiber_attenuation_npm,
    _calc_nf_lin_jit,
    _calc_span_noise_and_power_jit,
    _db_to_lin,
    _dispersion_at_frequency,
)
from network_nlo_eval.models.numba_kernels_p1_accel import (
    P1XpmMatrices,
    build_p1_xpm_pair_matrices,
    calc_span_noise_and_power_p1_matvec,
)
from network_nlo_eval.models.power_conventions import (
    ase_polarization_scale as _ase_scale_for_convention,
)
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
)
from network_nlo_eval.models.route import (
    accumulate_route,
    apply_amplifier_and_roadm,
    apply_edfa,
    resolve_noise_figure,
    roadm_target_transfer,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.slots import connection_overlaps_slots

_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)]

#: Supported per-channel bandwidth conventions.
#:
#: ``"symbol_rate"``
#:     Use ``ConnectionSpec.symbol_rate_hz`` (default, legacy behavior).
#: ``"occupied"``
#:     Use ``ConnectionSpec.occupied_bandwidth_hz`` (slot count * slot width).
#:     This option includes the actual occupied width in QoT.
#:
#: The source paper does not distinguish these bandwidth definitions.
CHANNEL_BANDWIDTH_SOURCES = ("symbol_rate", "occupied")
CHANNEL_BANDWIDTH_DEFAULT = "symbol_rate"

#: Power and bandwidth normalization conventions.
#:
#: ``"fixed_power"``
#:     Each connection uses its launch power independently of bandwidth.
#:     A wider connection spreads that power over a wider spectrum,
#:     reducing its power spectral density.
#: ``"fixed_psd"``
#:     Constant PSD: ``P_i_eff = P_i * B_i / symbol_rate_hz``, using
#:     the configured symbol rate as the reference bandwidth.
POWER_WIDTH_MODES = ("fixed_power", "fixed_psd")
POWER_WIDTH_DEFAULT = "fixed_power"

#: Single-span acceleration modes preserve the same physical equations.
#:
#: ``"off"``
#:     Use the reference JIT pairwise kernel on every call.
#: ``"cached"``
#:     Cache span results by launch power, length, and effective span count;
#:     results are identical to ``"off"``.
#: ``"matvec"``
#:     Replace XPM pairwise sums with pair matrices and matrix products
#:     (see ``network_nlo_eval.models.numba_kernels_p1_accel``).
#:     Only summation order changes, with a 1e-12 relative error budget.
#:
#: Select ``"off"`` explicitly for bitwise pairwise reference values.
ODE_ACCEL_MODES = ("off", "cached", "matvec")
ODE_ACCEL_DEFAULT = "matvec"

#: Minimum active-channel count for matrix-vector evaluation; below it,
#: use pairwise summation. The threshold is an absolute count because
#: the observed crossover is about 34 channels; 40 provides a margin
#: for the dense matrix work.
MATVEC_MIN_ACTIVE_CHANNELS = 40


#: Registry metadata for the ECZ-SRS triangular approximation and closed-form GN.
ECZ_MODEL_INFO = ModelInfo(
    name="ecz",
    summary="ECZ-SRS triangular SRS approximation with closed-form GN (SPM/XPM).",
    family="gn",
    reference="Lasagni et al., 'A Generalized Raman Scattering Model for Real-Time SNR "
    "Estimation of Multi-Band Systems', JLT 2023, doi:10.1109/jlt.2023.3250751",
    options=(
        ("ode_accel", "'off' | 'cached' | 'matvec'; single-span acceleration tier."),
        (
            "channel_bandwidth_source",
            "'symbol_rate' | 'occupied'; per-node B_i definition.",
        ),
        (
            "power_width_mode",
            "'fixed_power' | 'fixed_psd'; launch-power normalisation.",
        ),
        (
            "launch_power_convention",
            "'total' | 'per_pol'; coordinate of power_in_w and outputs.",
        ),
    ),
)


class MultiBandISRSGN:
    """Evaluate QoT using a multiband ISRS-GN single-span approximation.

    The model supports fixed channel grids and flexible connection grids.
    Total dual-polarization power is the default convention. In the explicit
    per-polarization convention the fiber kernel is driven with twice the
    input power, then signal and nonlinear noise are halved on output; newly
    generated amplifier and booster ASE is also halved. Signal and noise
    therefore use the same convention and produce the same SNR.
    """

    info = ECZ_MODEL_INFO

    #: Network-visible power convention; instances can override the total default.
    launch_power_convention = "total"

    #: Scale newly generated ASE by 1.0 in total-power coordinates,
    #: or 0.5 in per-polarization coordinates.
    ase_polarization_scale = 1.0

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionGeometry] | None = None,
        ode_accel: str = ODE_ACCEL_DEFAULT,
        slots_per_channel: int | None = None,
        channel_bandwidth_source: str = CHANNEL_BANDWIDTH_DEFAULT,
        power_width_mode: str = POWER_WIDTH_DEFAULT,
        launch_power_convention: str | None = None,
    ):
        """Initialize the ISRS-GN evaluator.

        Parameters
        ----------
        grid : SpectrumGrid or FlexibleGrid
            Frequency grid; a flexible grid requires ``connections``.
        ref_fiber_config : FiberSpanConfig
            Reference fiber properties used to compute wavelength-dependent values.
        symbol_rate_hz : float, optional
            System symbol rate in Hz.
        connections : sequence of ConnectionSpec or CarrierGeometry or None, optional
            Connections for per-connection mode; each uses its center frequency
            and occupied band. None selects the fixed-grid legacy behavior.
        ode_accel : {"off", "cached", "matvec"}, optional
            Single-span calculation strategy.
        slots_per_channel : int or None, optional
            Slot count used to expand blocked legacy ROADM channel indices.
        channel_bandwidth_source : {"symbol_rate", "occupied"}, optional
            Source for each connection's bandwidth.
        power_width_mode : {"fixed_power", "fixed_psd"}, optional
            Launch-power normalization convention.
        launch_power_convention : {"total", "per_pol"} or None, optional
            Signal and noise power convention. None inherits the class convention;
            an explicit convention also enables this class's kernel conversion.

        Raises
        ------
        ValueError
            If an acceleration, slot, bandwidth, or power convention is invalid.
        """
        if ode_accel not in ODE_ACCEL_MODES:
            msg = f"ode_accel must be one of {ODE_ACCEL_MODES}, got {ode_accel!r}."
            raise ValueError(msg)
        if slots_per_channel is not None and slots_per_channel < 1:
            msg = f"slots_per_channel must be >= 1, got {slots_per_channel}."
            raise ValueError(msg)
        if channel_bandwidth_source not in CHANNEL_BANDWIDTH_SOURCES:
            msg = (
                f"channel_bandwidth_source must be one of {CHANNEL_BANDWIDTH_SOURCES}, "
                f"got {channel_bandwidth_source!r}."
            )
            raise ValueError(msg)
        if power_width_mode not in POWER_WIDTH_MODES:
            msg = f"power_width_mode must be one of {POWER_WIDTH_MODES}, got {power_width_mode!r}."
            raise ValueError(msg)
        if any(pump.power_w > 0.0 for pump in ref_fiber_config.raman_pumps):
            msg = (
                "ECZ models do not support active Raman pumps. The paper's k_bar "
                "is dimensionless launch pre-emphasis, not pump power. Select a "
                "distributed-Raman model for pumped fibers."
            )
            raise ValueError(msg)
        if ref_fiber_config.lumped_loss:
            msg = "ECZ models do not support in-span lumped losses; select pcfm or pcfm_h3_raman."
            raise ValueError(msg)
        if launch_power_convention is None:
            # Preserve the inherited convention and ASE scale of subclasses
            # when no convention is explicitly supplied.
            resolved_convention = resolve_launch_power_convention(
                type(self).launch_power_convention
            )
            resolved_ase_scale = float(type(self).ase_polarization_scale)
            # Subclasses declaring a convention handle their own kernel conversion;
            # do not apply it twice here.
            native_conversion = False
        else:
            resolved_convention = resolve_launch_power_convention(
                launch_power_convention
            )
            resolved_ase_scale = _ase_scale_for_convention(resolved_convention)
            native_conversion = True

        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.cr_w_m_hz = _SRS_CR  # Raman gain coefficient.
        self.k_bar = 0.0  # Lasagni 2023 Eq. (22): launch pre-emphasis is disabled.
        self.delta_f_co_hz = (
            15e12  # SRS gain-spectrum cutoff frequency for silica fiber.
        )
        self.n2_m2_w = ref_fiber_config.nonlinear_index_n2
        self.ode_accel = ode_accel
        self.slots_per_channel = slots_per_channel
        self.channel_bandwidth_source = channel_bandwidth_source
        self.power_width_mode = power_width_mode
        self.launch_power_convention = resolved_convention
        #: Scale newly generated ASE for the selected power convention.
        self.ase_polarization_scale = resolved_ase_scale
        #: Whether this class performs the per-polarization kernel conversion.
        #: Explicit conventions enable it; inherited class conventions leave
        #: conversion to the subclass.
        self._native_power_conversion = native_conversion

        self.connections = list(connections) if connections is not None else None

        self._span_cache: dict[
            tuple[bytes, float, int], tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]
        ] = {}
        self._p1_xpm_matrices: P1XpmMatrices | None = None
        self._precompute_channel_dependent_properties()
        self._fiber_evaluator_cache: dict[str, MultiBandISRSGN] = {}

    @property
    def physical_power_scale(self) -> float:
        """Return the multiplier from input power to physical total power.

        An explicitly configured per-polarization evaluator returns 2.0; total
        power and subclasses handling their own conversion return 1.0.
        """
        if self._native_power_conversion and self.launch_power_convention == "per_pol":
            return 2.0
        return 1.0

    @property
    def n_nodes(self) -> int:
        """Return the number of channels or connections."""
        return int(self.f_abs_hz.shape[0])

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """Return the bandwidth in Hz of each channel or connection.

        Fixed-grid mode uses the symbol rate. Connection mode uses the selected
        symbol-rate or occupied-bandwidth convention.

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
        """Return per-node launch-power scaling for the configured width mode.

        Fixed PSD scales power by channel bandwidth divided by symbol rate;
        fixed power returns one for every node.
        """
        if self.power_width_mode == "fixed_power":
            return np.ones(self.n_nodes, dtype=np.float64)
        return self.channel_bandwidth_hz_channels / float(self.symbol_rate_hz)

    # Preserve the legacy method name.
    _power_width_scale = power_width_scale

    def roadm_blocks_node(self, roadm: ROADMConfig, node_index: int) -> bool:
        """Check if a ROADM blocks a channel or connection.

        Connection mode checks overlap against blocked slots and expands blocked
        channel indices using ``slots_per_channel``. Channel mode checks channel
        indices directly. Expansion requires an explicit slot count.
        """
        if self.connections is not None:
            if node_index >= len(self.connections):
                return False
            blocked_slots = set(roadm.blocked_slots)
            if roadm.blocked_channels:
                if not self.slots_per_channel:
                    msg = (
                        "Per-connection mode needs slots_per_channel "
                        "(channel_spacing_hz / slot_width_hz) to expand ROADM "
                        "blocked_channels into slots. Construct the evaluator with "
                        "slots_per_channel=..., or use blocked_slots."
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

    def blocked_nodes(self, roadm: ROADMConfig) -> list[int]:
        """Return indices of all nodes blocked by the ROADM."""
        return [
            index
            for index in range(self.n_nodes)
            if self.roadm_blocks_node(roadm, index)
        ]

    # Preserve the legacy method name.
    _blocked_nodes = blocked_nodes

    @property
    def band_center_hz(self) -> float:
        """Return the origin of relative frequencies in Hz."""
        if isinstance(self.grid, FlexibleGrid):
            return self.grid.band_center_hz
        return self.grid.center_frequency_hz

    def _precompute_channel_dependent_properties(self) -> None:
        """Precompute wavelength-dependent fiber and spectrum properties.

        These include channel frequencies, attenuation, effective area, noise
        figure, dispersion, and frequency band edges.
        """
        if self.connections is not None:
            # Use one node at the center frequency of each connection.
            self.f_abs_hz = np.array(
                [conn.centre_hz for conn in self.connections], dtype=np.float64
            )
            self.lambdas_m = C_LIGHT / self.f_abs_hz  # (m)
        elif isinstance(self.grid, SpectrumGrid):
            self.lambdas_m = self.grid.wavelengths  # (m)
            self.f_abs_hz = self.grid.frequencies  # (Hz)
        else:
            msg = "Legacy mode requires a SpectrumGrid; pass connections= for a FlexibleGrid."
            raise TypeError(msg)

        # Power attenuation (Np/m).
        if self.ref_fiber_config.use_attenuation_polynomial:
            self.alpha_power_npm_channels = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                self.ref_fiber_config.reference_wavelength_nm,
                self.ref_fiber_config.attenuation_alpha2,
                self.ref_fiber_config.attenuation_alpha1,
                self.ref_fiber_config.attenuation_alpha0,
            )
        else:
            alpha = self.ref_fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            self.alpha_power_npm_channels = np.full(self.n_nodes, alpha)

        # Effective mode area (m²).
        self.a_eff_m2_channels = _calc_aeff_dynamic(
            self.lambdas_m,
            self.ref_fiber_config.effective_area_um2_ref,
            self.ref_fiber_config.aeff_slope_um2_nm,
            self.ref_fiber_config.reference_wavelength_nm,
        )

        # Linear noise figure.
        self.nf_lin_channels = _calc_nf_lin_jit(self.lambdas_m)

        self.beta2_s2_m, self.beta3_s3_m = _dispersion_at_frequency(
            self.ref_fiber_config.dispersion_parameter_d,
            self.ref_fiber_config.dispersion_slope_s,
            self.ref_fiber_config.reference_wavelength_nm,
            self.band_center_hz,
        )

        # Relative frequencies in Hz for NLI and SRS calculations.
        self.f_rel_hz = self.f_abs_hz - self.band_center_hz
        if self.connections is not None:
            # For variable-width connections, calculate actual occupied band edges.
            low_edges = np.array(
                [
                    conn.centre_hz - conn.n_slots * conn.grid.slot_width_hz / 2
                    for conn in self.connections
                ],
                dtype=np.float64,
            )
            high_edges = np.array(
                [
                    conn.centre_hz + conn.n_slots * conn.grid.slot_width_hz / 2
                    for conn in self.connections
                ],
                dtype=np.float64,
            )
            self.f_m_hz = float(low_edges.min() - self.band_center_hz)
            self.f_M_hz = float(high_edges.max() - self.band_center_hz)
        else:
            self.f_m_hz = self.f_rel_hz[0] - self.symbol_rate_hz / 2
            self.f_M_hz = self.f_rel_hz[-1] + self.symbol_rate_hz / 2

    def _p1_xpm_matrices_for(self) -> P1XpmMatrices:
        """Return cached channel-pair XPM matrices, building them on first use.

        The matrices depend on channel geometry and fiber parameters, not on
        candidate launch power or span length.
        """
        if self._p1_xpm_matrices is not None:
            return self._p1_xpm_matrices
        g1, g2, phi_raw, gamma = build_p1_xpm_pair_matrices(
            self.n_nodes,
            self.channel_bandwidth_hz_channels,
            self.f_rel_hz,
            self.lambdas_m,
            self.a_eff_m2_channels,
            self.alpha_power_npm_channels,
            self.beta2_s2_m,
            self.beta3_s3_m,
            self.n2_m2_w,
        )
        self._p1_xpm_matrices = P1XpmMatrices(
            g1=g1,
            g2=g2,
            g1_t=np.ascontiguousarray(g1.T),
            g2_t=np.ascontiguousarray(g2.T),
            phi_raw=phi_raw,
            gamma=gamma,
        )
        return self._p1_xpm_matrices

    def _p1_span_matvec(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int,
        nf_lin_channels: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Evaluate the P1 matrix-vector kernel for one launch-power profile.

        Output power and SPM match the pairwise kernel; XPM differs only in the
        order of floating-point summation.
        """
        mats = self._p1_xpm_matrices_for()
        p_out, s2_spm, s2_xpm, s2_ase = calc_span_noise_and_power_p1_matvec(
            power_in_w=np.asarray(power_in_w, dtype=np.float64)[None, :],
            span_length_m=float(span_length_m),
            f_rel_hz=self.f_rel_hz,
            f_abs_hz=self.f_abs_hz,
            lambdas_m=self.lambdas_m,
            alpha_power_npm=self.alpha_power_npm_channels,
            a_eff_m2=self.a_eff_m2_channels,
            beta_2_s2_m=self.beta2_s2_m,
            beta_3_s3_m=self.beta3_s3_m,
            n_2_m2_w=self.n2_m2_w,
            nf_lin_channels=nf_lin_channels,
            symbol_rate_hz=self.symbol_rate_hz,
            b_hz_channels=self.channel_bandwidth_hz_channels,
            n_effective_spans=int(n_effective_spans),
            cr_w_m_hz=self.cr_w_m_hz,
            k_bar=self.k_bar,
            delta_f_co_hz=self.delta_f_co_hz,
            mats=mats,
        )
        # Convert (1, N) to the baseline single-candidate (N,) shape.
        return p_out[0], s2_spm[0], s2_xpm[0], s2_ase[0]

    def _calc_span_noise_and_power(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Calculate fiber, amplifier, and optional ROADM output for one span.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Per-channel input power in the evaluator's power convention.
        span_length_m : float
            Span length in meters.
        edfa_config : EDFAConfig or None
            Span amplifier configuration.
        roadm_config : ROADMConfig or None, optional
            ROADM configuration at the span end.
        n_effective_spans : int, optional
            Effective span count used for NLI accumulation.

        Returns
        -------
        tuple of numpy.ndarray
            Output signal, SPM, XPM, and ASE powers in watts, in the evaluator's
            power convention. Newly generated EDFA ASE uses its polarization scale.
        """
        # 1. Evaluate fiber physics before amplification and ROADM processing.
        fiber_output = self.span_fiber_output(
            power_in_w, span_length_m, n_effective_spans
        )

        # 2. Apply amplifier ASE and ROADM loss using the shared implementation.
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=roadm_config,
            nf_lin=self._resolve_nf(edfa_config),
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            ase_polarization_scale=self.ase_polarization_scale,
        )
        return (
            element.power_out_w,
            element.spm_noise_w,
            element.xpm_noise_w,
            element.ase_noise_w,
        )

    def _resolve_nf(self, edfa_config: EDFAConfig | None) -> NDArrayFloat:
        """Return per-channel linear amplifier noise figures."""
        return resolve_noise_figure(edfa_config, self.nf_lin_channels, self.n_nodes)

    def span_fiber_output(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """Evaluate fiber output signal and nonlinear noise for one span.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Per-channel input powers in the evaluator's power convention.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Effective span count for NLI accumulation.

        Returns
        -------
        SpanFiberOutput
            Fresh signal, SPM, and XPM arrays in the evaluator's convention.

        Notes
        -----
        Cached results are stored in physical total-power coordinates and copied
        before return. Matrix-vector evaluation is selected for sufficiently many
        active channels. ASE is generated later by the amplifier layer.
        """
        power_scale = self.physical_power_scale
        physical_power_in_w = (
            power_in_w if power_scale == 1.0 else power_in_w * power_scale
        )

        if self.ode_accel == "cached":
            cache_key = (
                physical_power_in_w.tobytes(),
                float(span_length_m),
                int(n_effective_spans),
            )
            cached = self._span_cache.get(cache_key)
            if cached is not None:
                # Return copies because callers may modify output arrays in place.
                p_out_fiber, s2_spm, s2_xpm = (array.copy() for array in cached)
                return self._span_output_in_convention(p_out_fiber, s2_spm, s2_xpm)
        else:
            cache_key = None

        if (
            self.ode_accel == "matvec"
            and np.count_nonzero(physical_power_in_w) >= MATVEC_MIN_ACTIVE_CHANNELS
        ):
            # Use matrix-vector evaluation only above the active-channel threshold;
            # fewer channels use the numerically equivalent pairwise kernel.
            p_out_fiber, s2_spm, s2_xpm, _ = self._p1_span_matvec(
                physical_power_in_w,
                span_length_m,
                n_effective_spans,
                self.nf_lin_channels,
            )
        else:
            p_out_fiber, s2_spm, s2_xpm, _ = _calc_span_noise_and_power_jit(
                power_in_w=physical_power_in_w,
                span_length_m=span_length_m,
                f_rel_hz=self.f_rel_hz,
                f_abs_hz=self.f_abs_hz,
                lambdas_m=self.lambdas_m,
                alpha_power_npm=self.alpha_power_npm_channels,
                a_eff_m2=self.a_eff_m2_channels,
                beta2_s2_m=self.beta2_s2_m,
                beta3_s3_m=self.beta3_s3_m,
                n2_m2_w=self.n2_m2_w,
                nf_lin_channels=self.nf_lin_channels,
                symbol_rate_hz=self.symbol_rate_hz,
                b_hz_channels=self.channel_bandwidth_hz_channels,
                n_effective_spans=n_effective_spans,
                cr_w_m_hz=self.cr_w_m_hz,
                k_bar=self.k_bar,
                delta_f_co_hz=self.delta_f_co_hz,
            )
        if cache_key is not None:
            self._span_cache[cache_key] = (
                p_out_fiber.copy(),
                s2_spm.copy(),
                s2_xpm.copy(),
            )
        return self._span_output_in_convention(p_out_fiber, s2_spm, s2_xpm)

    def _span_output_in_convention(
        self,
        p_out_fiber: NDArrayFloat,
        s2_spm: NDArrayFloat,
        s2_xpm: NDArrayFloat,
    ) -> SpanFiberOutput:
        """Convert kernel output from physical total power to model coordinates.

        Explicit per-polarization mode halves signal and NLI once; total mode and
        subclasses that handle their own conversions return the arrays directly.
        """
        if self.physical_power_scale != 1.0:
            return SpanFiberOutput(p_out_fiber * 0.5, s2_spm * 0.5, s2_xpm * 0.5)
        return SpanFiberOutput(p_out_fiber, s2_spm, s2_xpm)

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """Evaluate fiber and amplifier effects for a span; the route handles ROADMs."""
        evaluator = self.for_fiber(fiber_config)
        p_out, spm, xpm, ase = evaluator._calc_span_noise_and_power(
            power_in_w, span_length_m, edfa_config, None, n_effective_spans=1
        )
        return ElementOutput(p_out, spm, xpm, ase)

    def _calc_span_output_with_context(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        span_context: SpanContext | None,
    ) -> ElementOutput:
        """Evaluate one bound-fiber span for network-level assembly.

        The base model ignores span context and returns power-only NLI. Coherent
        subclasses can override this method to provide complex amplitudes.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Per-channel input power in the evaluator's power convention.
        span_length_m : float
            Span length in meters.
        edfa_config : EDFAConfig or None
            Span amplifier configuration.
        span_context : SpanContext or None
            Span position, unused by the base model.

        Returns
        -------
        ElementOutput
            Output signal and noise for this span.
        """
        del span_context  # The base model has no inter-span phase.
        p_out, spm, xpm, ase = self._calc_span_noise_and_power(
            power_in_w, span_length_m, edfa_config, None, n_effective_spans=1
        )
        return ElementOutput(p_out, spm, xpm, ase)

    def calc_span_noise_and_power_batch(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None = None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Evaluate a batch of single-span launch-power candidates.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Candidate launch powers with shape ``(B, N)``.
        span_length_m : float
            Span length in meters.
        edfa_config : EDFAConfig or None, optional
            Span amplifier configuration.
        roadm_config : ROADMConfig or None, optional
            ROADM configuration at the span end.
        n_effective_spans : int, optional
            Effective span count used for NLI accumulation.

        Returns
        -------
        tuple of numpy.ndarray
            Signal, SPM, XPM, and ASE arrays, each of shape ``(B, N)``.

        Raises
        ------
        ValueError
            If acceleration is not ``"matvec"`` or an input shape is invalid.

        Notes
        -----
        Results agree with the serial path within the matrix-vector relative
        error budget. Explicit per-polarization mode applies the same input,
        output, and ASE scaling as serial evaluation.
        """
        if self.ode_accel != "matvec":
            msg = (
                "calc_span_noise_and_power_batch requires ode_accel='matvec': the batched kernel "
                "is the batch form of the P1 closed-form NLI (XPM channel-pair matrices + matvec). "
                f"Got ode_accel={self.ode_accel!r}."
            )
            raise ValueError(msg)

        P = np.asarray(power_in_w, dtype=np.float64)
        n_conn = self.n_nodes
        if P.ndim != 2 or P.shape[1] != n_conn:
            msg = (
                f"power_in_w must have shape (B, {n_conn}), got {np.shape(power_in_w)}."
            )
            raise ValueError(msg)

        # The kernel uses physical total power; convert per-polarization input
        # before evaluation and halve the output. Total power is unchanged.
        power_scale = self.physical_power_scale
        kernel_power = P if power_scale == 1.0 else P * power_scale

        current_nf_lin = self._resolve_nf(edfa_config)

        p_out_fiber, s2_spm, s2_xpm, _ = calc_span_noise_and_power_p1_matvec(
            power_in_w=kernel_power,
            span_length_m=float(span_length_m),
            f_rel_hz=self.f_rel_hz,
            f_abs_hz=self.f_abs_hz,
            lambdas_m=self.lambdas_m,
            alpha_power_npm=self.alpha_power_npm_channels,
            a_eff_m2=self.a_eff_m2_channels,
            beta_2_s2_m=self.beta2_s2_m,
            beta_3_s3_m=self.beta3_s3_m,
            n_2_m2_w=self.n2_m2_w,
            nf_lin_channels=current_nf_lin,
            symbol_rate_hz=self.symbol_rate_hz,
            b_hz_channels=self.channel_bandwidth_hz_channels,
            n_effective_spans=int(n_effective_spans),
            cr_w_m_hz=self.cr_w_m_hz,
            k_bar=self.k_bar,
            delta_f_co_hz=self.delta_f_co_hz,
            mats=self._p1_xpm_matrices_for(),
        )
        if power_scale != 1.0:
            p_out_fiber = p_out_fiber * 0.5
            s2_spm = s2_spm * 0.5
            s2_xpm = s2_xpm * 0.5

        if edfa_config is None:
            p_out = p_out_fiber
            nli_transfer = np.divide(p_out_fiber, P, out=np.zeros_like(P), where=P > 0)
            s2_ase = np.zeros_like(P)
        else:
            p_out, nli_transfer, s2_ase = apply_edfa(
                edfa_config,
                P,
                p_out_fiber,
                current_nf_lin,
                self.f_abs_hz,
                self.channel_bandwidth_hz_channels,
                ase_polarization_scale=self.ase_polarization_scale,
            )
        s2_spm = s2_spm * nli_transfer
        s2_xpm = s2_xpm * nli_transfer

        if roadm_config:
            loss_lin = _db_to_lin(roadm_config.path_loss_db())
            roadm_transfer: NDArrayFloat | float = 1.0 / loss_lin
            if roadm_config.output_target is not None:
                roadm_transfer = roadm_target_transfer(
                    roadm_config.output_target,
                    p_out,
                    self.channel_bandwidth_hz_channels,
                    loss_lin,
                    self.ase_polarization_scale,
                )
            p_out *= roadm_transfer
            s2_spm *= roadm_transfer
            s2_xpm *= roadm_transfer
            s2_ase *= roadm_transfer

        return p_out, s2_spm, s2_xpm, s2_ase

    def for_fiber(self, fiber_config: FiberSpanConfig) -> "MultiBandISRSGN":
        """Return a cached evaluator configured for a particular fiber.

        Connection geometry and power convention are passed to the child
        unchanged so its kernel uses the same physical launch-power coordinates.
        """
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = MultiBandISRSGN(
                self.grid,
                fiber_config,
                self.symbol_rate_hz,
                connections=self.connections,
                ode_accel=self.ode_accel,
                slots_per_channel=self.slots_per_channel,
                channel_bandwidth_source=self.channel_bandwidth_source,
                power_width_mode=self.power_width_mode,
                launch_power_convention=self.launch_power_convention,
            )
        return self._fiber_evaluator_cache[key]

    def for_connections(
        self, connections: tuple[CarrierGeometry, ...], slots_per_channel: int
    ) -> "MultiBandISRSGN":
        """Bind the ECZ model settings to the active carrier geometry."""
        return type(self)(
            connections[0].grid,
            self.ref_fiber_config,
            self.symbol_rate_hz,
            connections=connections,
            ode_accel=self.ode_accel,
            slots_per_channel=slots_per_channel,
            channel_bandwidth_source=self.channel_bandwidth_source,
            power_width_mode=self.power_width_mode,
            launch_power_convention=self.launch_power_convention,
        )

    def evaluate_path(
        self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat
    ) -> PathQoTResult:
        """Evaluate signal and noise along a physical optical path.

        Parameters
        ----------
        optical_path : MultiSpanOpticalPath
            Ordered fiber, amplifier, and optional ROADM configurations.
        power_in_w : numpy.ndarray
            Per-node launch powers in the evaluator's power convention.

        Returns
        -------
        PathQoTResult
            Per-channel signal, noise, SNR, and path component counts.

        Notes
        -----
        The shared route accumulator applies span splitting, amplification,
        ROADM processing, and SNR assembly.
        """
        return accumulate_route(
            self,
            optical_path,
            power_in_w,
            ase_polarization_scale=self.ase_polarization_scale,
        )


# Register the evaluator for construction through the QoT registry.
register_model(ECZ_MODEL_INFO, MultiBandISRSGN)
