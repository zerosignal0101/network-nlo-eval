"""Hybrid Raman model with coupled ODE propagation and closed-form NLI.

Signal, pump and ASE equations use the shared Raman solver. The default paper
path fits a five-parameter profile and applies the Buglia closed-form SPM and
XPM expressions; ``integral`` uses numerical quadrature and ``pcfm`` retains
the older PCFM approximation for comparison. Spans accumulate as powers, with
SPM scaled by ``n ** epsilon`` and epsilon defaulting to zero.

The gain spectrum is an approximation rather than the measured silica spectrum;
pump ASE and pump-signal XPM are omitted. The closed-form implementation uses
the sign corrections described in :mod:`network_nlo_eval.models.buglia2024_closed`.

References
----------
H. Buglia, R. I. Killey and P. Bayvel, "Ultra-Wideband Modeling of Optical
Fibre Nonlinearity in Hybrid-Amplified Links", J. Lightwave Technol., 2024,
doi:10.1109/JLT.2024.3439999.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanContext,
    register_model,
)
from network_nlo_eval.models.buglia2024_closed import (
    BugliaClosedProfile,
    fit_buglia_profile_5param,
    nli_closed_form,
    nli_integral,
)
from network_nlo_eval.models.dra_common import DRASpanKernelBase
from network_nlo_eval.models.numba_kernels import _calc_aeff_dynamic
from network_nlo_eval.models.pcfm import PCFMModel, PCFMSpanKernel
from network_nlo_eval.models.raman import (
    C_LIGHT,
    RamanGainModel,
    resolve_raman_gain_model,
)
from network_nlo_eval.models.raman_ode import (
    DEFAULT_FIXED_POINT_ITERATIONS,
    DEFAULT_PAPER_SPP_STEPS,
    PaperRamanSpanResult,
    RamanPaperSpanSolver,
    resolve_launch_power_convention,
)
from network_nlo_eval.models.route import (
    apply_amplifier_and_roadm,
    resolve_noise_figure,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.network.elements import (
        EDFAConfig,
        FiberSpanConfig,
        RamanPumpConfig,
    )

__all__ = [
    "DEFAULT_FIXED_POINT_ITERATIONS",
    "DEFAULT_PAPER_SPP_STEPS",
    "HYBRID_RAMAN_PAPER_MODEL_INFO",
    "NLI_FORMS",
    "HybridRamanPaperModel",
    "HybridRamanPaperSpanKernel",
    "HybridRamanPcfmSpanKernel",
    "PaperRamanSpanResult",
    "RamanPaperSpanSolver",
]

#: NLI forms: paper closed form, numerical integral baseline, or legacy PCFM.
NLI_FORMS: tuple[str, ...] = ("paper", "integral", "pcfm")


class HybridRamanPaperSpanKernel(DRASpanKernelBase[BugliaClosedProfile]):
    """Hybrid Raman span kernel using the paper's closed-form SPM and XPM.

    The coupled ODE profile is fitted to Eq. (10), then the SPM and XPM closed
    forms are evaluated and accumulated as powers with Eq. (6)'s coherence factor.

    Parameters
    ----------
    f_abs_hz, b_hz : numpy.ndarray
        Channel center frequencies and bandwidths in Hz.
    fiber_config : FiberSpanConfig
        Span fiber parameters.
    pumps : sequence of RamanPumpConfig
        Forward and backward pumps; an empty sequence disables Raman pumping.
    gain_model : RamanGainModel or str
        Raman gain spectrum or model name.
    raman_ase : bool, optional
        Expose newly generated Raman ASE to the route layer.
    n_iterations : int, optional
        Maximum backward-pump fixed-point iterations.
    include_isrs : bool, optional
        Include signal-signal ISRS.
    pump_depletion : bool, optional
        Include pump depletion; disabling it is a comparison convention.
    signal_polarizations : int, optional
        Signal polarization multiplier for the ODE coupling drive.
    pump_pump_raman : bool, optional
        Include Raman coupling between pumps.
    launch_power_convention : {"total", "per_pol"}, optional
        Input and output power coordinate. Per-polarization input requires two
        signal polarizations; NLI is calculated using total power and scaled back.
    n_steps : int, optional
        RK4 steps and profile fit samples.
    nli_form : {"paper", "integral"}, optional
        Paper closed form or numerical integration baseline.
    spm_coherence_exponent : float, optional
        Eq. (6) SPM coherence exponent.
    n_quadrature : int, optional
        Frequency quadrature nodes for the integral baseline.

    Raises
    ------
    ValueError
        If the NLI form, ODE parameters or power conventions are incompatible.
    """

    #: Profile kind used in diagnostics.
    profile_kind = "buglia_hybrid_eq2_5param"

    #: Eq. (6) accumulates powers; n^epsilon is applied without amplitudes.
    emits_nli_amplitudes = False

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        *,
        pumps: Sequence[RamanPumpConfig],
        gain_model: RamanGainModel | str,
        raman_ase: bool = True,
        n_steps: int = DEFAULT_PAPER_SPP_STEPS,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        nli_form: str = "paper",
        spm_coherence_exponent: float = 0.0,
        n_quadrature: int = 24,
        pump_depletion: bool = True,
        signal_polarizations: int = 1,
        pump_pump_raman: bool = True,
        launch_power_convention: str = "total",
    ) -> None:
        """Validate the NLI form and initialize the span kernel."""
        if nli_form not in ("paper", "integral"):
            msg = "nli_form must be 'paper' or 'integral' for the paper kernel."
            raise ValueError(msg)
        resolved_convention = resolve_launch_power_convention(
            launch_power_convention, signal_polarizations
        )
        super().__init__(
            f_abs_hz,
            b_hz,
            fiber_config,
            pumps=pumps,
            gain_model=resolve_raman_gain_model(gain_model),
            raman_ase=raman_ase,
            n_steps=n_steps,
            n_iterations=n_iterations,
            include_isrs=include_isrs,
            n_quadrature=n_quadrature,
            pump_depletion=pump_depletion,
            signal_polarizations=signal_polarizations,
            pump_pump_raman=pump_pump_raman,
        )
        self.launch_power_convention = resolved_convention
        # GN weights use total power; convert per-polarization inputs and outputs.
        self._nli_drive_scale = 2.0 if resolved_convention == "per_pol" else 1.0
        self._nli_form = str(nli_form)
        self._spm_exponent = float(spm_coherence_exponent)
        # Eq. (8)/(9) uses the per-channel nonlinear coefficient.
        self.gamma_per_w_m = (2.0 * np.pi * self.f_abs_hz / C_LIGHT) * (
            self.n2_m2_w / self.a_eff_m2
        )
        self._f_hat_hz = self._pump_mean_frequency()

    def _pump_mean_frequency(self) -> float:
        """Return the power-weighted mean pump frequency, or the channel mean without pumps."""
        pumps = list(self._pumps)
        if not pumps:
            return float(np.mean(self.f_abs_hz))
        powers = np.asarray([float(pump.power_w) for pump in pumps], dtype=np.float64)
        frequencies = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in pumps],
            dtype=np.float64,
        )
        if float(np.sum(powers)) <= 0.0:
            return float(np.mean(self.f_abs_hz))
        return float(np.average(frequencies, weights=powers))

    def _fit_profile(
        self, profile_norm: NDArrayFloat, span_length_m: float
    ) -> BugliaClosedProfile:
        """Fit the ODE profile to the five-parameter Eq. (10) form."""
        return fit_buglia_profile_5param(
            profile_norm,
            span_length_m,
            self.f_abs_hz,
            self.b_hz,
            self.alpha_npm,
            self._f_hat_hz,
            self._n_steps,
        )

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: BugliaClosedProfile,
        span_context: SpanContext | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Evaluate per-span SPM and XPM using Eq. (6)-(9).

        Per-polarization input is converted to physical total power for the GN kernel,
        then noise is converted back to the signal and ASE coordinate.
        """
        powers = (
            power_internal
            if self._nli_drive_scale == 1.0
            else power_internal * self._nli_drive_scale
        )
        if self._nli_form == "integral":
            spm, xpm = nli_integral(
                profile,
                self.b_hz,
                powers,
                self.gamma_per_w_m,
                self.beta2,
                self.beta3,
                self.reference_frequency_hz,
                n_frequency=max(16, min(64, int(self._n_quadrature) * 2)),
            )
        else:
            n_spans = (
                1 if span_context is None else max(1, int(span_context.total_spans))
            )
            spm, xpm = nli_closed_form(
                profile,
                self.b_hz,
                powers,
                self.gamma_per_w_m,
                self.beta2,
                self.beta3,
                self.reference_frequency_hz,
                spm_coherence_exponent=self._spm_exponent,
                n_spans=n_spans,
            )
        if self._nli_drive_scale != 1.0:
            # Division by two is exact in binary floating point.
            spm = spm / self._nli_drive_scale
            xpm = xpm / self._nli_drive_scale
        return spm, xpm


class HybridRamanPcfmSpanKernel(PCFMSpanKernel):
    """Legacy PCFM SCI/XCI kernel driven by the coupled Raman ODE profile.

    Without active pumps the parent PCFM integration path is used. This kernel is
    provided for comparison and does not implement the paper's Eq. (6)-(9).

    Parameters
    ----------
    f_abs_hz, b_hz : numpy.ndarray
        Channel center frequencies and bandwidths in Hz.
    fiber_config : FiberSpanConfig
        Span fiber parameters.
    pumps : sequence of RamanPumpConfig
        Forward and backward pumps; an empty sequence disables Raman pumping.
    gain_model : RamanGainModel or str
        Raman gain spectrum or model name.
    raman_ase : bool, optional
        Expose newly generated Raman ASE to the route layer.
    n_iterations : int, optional
        Maximum backward-pump fixed-point iterations.
    include_isrs : bool, optional
        Include signal-signal ISRS.
    pump_depletion : bool, optional
        Include pump depletion; disabling it is a comparison convention.
    signal_polarizations : int, optional
        Signal polarization multiplier for the ODE coupling drive.
    pump_pump_raman : bool, optional
        Include Raman coupling between pumps.
    launch_power_convention : {"total", "per_pol"}, optional
        Input and output power coordinate. Per-polarization input requires two
        signal polarizations; NLI is calculated using total power and scaled back.
    **kwargs
        Additional PCFM span kernel options.
    """

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        *,
        pumps: Sequence[RamanPumpConfig],
        gain_model: RamanGainModel | str,
        raman_ase: bool = True,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        pump_depletion: bool = True,
        signal_polarizations: int = 1,
        pump_pump_raman: bool = True,
        launch_power_convention: str = "total",
        **kwargs: Any,
    ) -> None:
        """Initialize the PCFM comparison kernel and cache pump parameters."""
        if not isinstance(cast("object", pump_pump_raman), (bool, np.bool_)):
            msg = f"pump_pump_raman must be a bool, got {pump_pump_raman!r}."
            raise ValueError(msg)  # noqa: TRY004 - Preserve the public exception type.
        resolved_convention = resolve_launch_power_convention(
            launch_power_convention, signal_polarizations
        )
        kwargs.setdefault("cache_spans", False)
        super().__init__(
            f_abs_hz,
            b_hz,
            fiber_config,
            nli_power_convention=resolved_convention,
            **kwargs,
        )
        self.launch_power_convention = resolved_convention
        self._pumps = list(pumps)
        self._gain_model = resolve_raman_gain_model(gain_model)
        self._raman_ase_enabled = bool(raman_ase)
        self._n_iterations = int(n_iterations)
        self._include_isrs = bool(include_isrs)
        self._pump_depletion = bool(pump_depletion)
        self._signal_polarizations = int(signal_polarizations)
        self._pump_pump_raman = bool(pump_pump_raman)
        self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._solver_cache: dict[int, RamanPaperSpanSolver] = {}
        self._pump_alpha_npm = np.asarray(
            [
                float(pump.attenuation_db_per_km) * np.log(10.0) / 10.0 / 1e3
                for pump in self._pumps
            ],
            dtype=np.float64,
        )
        pump_f = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in self._pumps],
            dtype=np.float64,
        )
        self._pump_a_eff_m2 = _calc_aeff_dynamic(
            C_LIGHT / pump_f,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        self._active = any(float(pump.power_w) > 0.0 for pump in self._pumps)

    @property
    def active(self) -> bool:
        """Return whether at least one pump has nonzero power."""
        return self._active

    def set_input_ase_w(self, ase_w: NDArrayFloat | None) -> None:
        """Set the input signal ASE in model channel order.

        Parameters
        ----------
        ase_w : numpy.ndarray or None
            Per-channel ASE in watts; None means zero.

        Raises
        ------
        ValueError
            If the input shape does not match the channel count.
        """
        if ase_w is None:
            self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
            return
        values = np.asarray(ase_w, dtype=np.float64)
        if values.shape != (self.n_nodes,):
            msg = "ase_w must contain one value per channel."
            raise ValueError(msg)
        self._input_ase_internal = np.ascontiguousarray(values[self.order])

    @property
    def last_raman_ase_w(self) -> NDArrayFloat:
        """Return the last span's new Raman ASE in model order."""
        return np.ascontiguousarray(self._last_raman_ase_internal[self.inverse_order])

    @property
    def last_pump_power_out_w(self) -> NDArrayFloat | None:
        """Return the last span's pump output powers, or None if unavailable."""
        return getattr(self, "_last_pump_power_out_w", None)

    def _solver_for(self, span_length_m: float) -> RamanPaperSpanSolver:
        """Return the Raman ODE solver cached by span length."""
        key = round(span_length_m * 1e6)
        solver = self._solver_cache.get(key)
        if solver is None:
            solver = RamanPaperSpanSolver(
                self.f_abs_hz,
                self.b_hz,
                self._pumps,
                span_length_m,
                self.alpha_npm,
                self.a_eff_m2,
                self._pump_alpha_npm,
                self._pump_a_eff_m2,
                self._gain_model,
                n_steps=self.spp_steps,
                n_iterations=self._n_iterations,
                include_isrs=self._include_isrs,
                include_ase=self._raman_ase_enabled,
                pump_depletion=self._pump_depletion,
                signal_polarizations=self._signal_polarizations,
                pump_pump_raman=self._pump_pump_raman,
                launch_power_convention=self.launch_power_convention,
            )
            self._solver_cache[key] = solver
        return solver

    def _integrate_spp(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute the normalized power profile with the coupled Raman ODE.

        Parameters
        ----------
        power : numpy.ndarray
            Input signal powers in internal descending-frequency order.
        span_length_m : float
            Span length in meters.

        Returns
        -------
        tuple of numpy.ndarray
            Normalized profile and output powers.
        """
        if not self._active:
            self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
            self._last_pump_power_out_w = None
            return super()._integrate_spp(power, span_length_m)

        solver = self._solver_for(span_length_m)
        result = solver.solve(power, self._input_ase_internal)
        if self._raman_ase_enabled:
            self._last_raman_ase_internal = np.ascontiguousarray(result.raman_ase_w)
        else:
            self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_pump_power_out_w = np.ascontiguousarray(result.pump_power_out_w)
        return result.profile_norm, result.power_out_w


HYBRID_RAMAN_PAPER_MODEL_INFO = ModelInfo(
    name="hybrid_raman_paper",
    summary=(
        "Buglia 2024 hybrid Raman: Eq. (2)/(4) coupled signal/pump/ASE ODE, Eq. (10) "
        "five-coefficient profile fit, Eq. (8)/(9) SPM/XPM closed forms and Eq. (6) "
        "n^epsilon span accumulation (default nli_form='paper'). nli_form='pcfm' keeps "
        "the legacy PCFM-kernel approximation for regression."
    ),
    family="gn+raman",
    reference=(
        "Buglia, Killey, Bayvel, 'Ultra-Wideband Modeling of Optical Fibre Nonlinearity "
        "in Hybrid-Amplified Links', JLT 2024, doi:10.1109/JLT.2024.3439999"
    ),
    options=(
        (
            "raman_pumps",
            "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps.",
        ),
        (
            "raman_gain_model",
            (
                "'gaussian' (default), 'linear' (CZ), 'table' (normalized spectrum given by "
                "raman_gain_table), a .json/.csv Raman gain table file path, or a RamanGainModel instance "
                "(use RamanGainModel.from_file(path)). Loaded once at construction, never in the hot path."
            ),
        ),
        (
            "raman_gain_table",
            (
                "(offsets_hz, gains) normalized gain table used when raman_gain_model='table'. Linear "
                "interpolation inside the table, zero outside. It replaces the Gaussian/CZ shape for every "
                "pump-signal and pump-pump coefficient of the coupled ODE. Mutually exclusive with a "
                "RamanGainModel instance or a table file path."
            ),
        ),
        (
            "raman_ase",
            "Expose the newly generated Raman ASE before the EDFA. Default True.",
        ),
        (
            "raman_temperature_k",
            (
                "Fibre temperature used by the spontaneous Raman factor. Default 300 K (None). "
                "An explicit value overrides a gain table file and warns."
            ),
        ),
        (
            "raman_peak_gain_coefficient_m_per_w",
            (
                "Peak local Raman gain rate before A_eff division. Default 6e-14 (None). "
                "An explicit value overrides a gain table file and warns."
            ),
        ),
        (
            "include_isrs",
            "Include inter-channel ISRS in the coupled ODE. Default True.",
        ),
        (
            "fixed_point_iterations",
            "Maximum BW-pump fixed-point iterations. Default 24.",
        ),
        (
            "nli_form",
            "'paper' (Eq. 6-9 closed form, default), 'integral' (Eq. 6-8 baseline) or 'pcfm' (legacy).",
        ),
        (
            "spm_coherence_exponent",
            "Eq. (6) epsilon; 0 (default) sums spans incoherently.",
        ),
        (
            "poly_degree",
            "PCFM SPP polynomial degree; only used by nli_form='pcfm'. Default 5.",
        ),
        ("egn", "EGN correction; only used by nli_form='pcfm'. Default False."),
        (
            "edge_fix",
            "PCFM zero-dispersion edge fix; only used by nli_form='pcfm'. Default True.",
        ),
        (
            "spp_steps",
            "RK4 steps of the coupled ODE and of the Eq. (10) fit. Default 32.",
        ),
        (
            "pump_depletion",
            (
                "Deplete the pumps with the signals (default True). False freezes the pumps to their own "
                "attenuation, the undepleted-pump control convention of the waveform reference."
            ),
        ),
        (
            "signal_polarizations",
            (
                "Polarizations represented by each ODE signal state. Default 1 (historical per-polarization "
                "drive, bit-for-bit unchanged). 2 drives the coupling with the dual-polarization total signal "
                "power, matching the physical waveform pump-depletion drive; together with "
                "launch_power_convention='per_pol' it is the end-to-end per-polarization coordinate."
            ),
        ),
        (
            "launch_power_convention",
            (
                "'total' (default) means power_in_w is the per-channel dual-polarization total power; 'per_pol' "
                "means per-polarization power (physical total = 2x) and requires signal_polarizations=2. Under "
                "'per_pol' the coupled ODE state, the NLI and the newly generated EDFA/ROADM booster ASE all stay "
                "per-polarization: the Raman/NLI drive takes the physical total power, the NLI is halved back and "
                "ase_polarization_scale=0.5 is passed to the route layer. input_ase_w (the Raman ASE, already in "
                "model coordinates) is never rescaled. 'total' with signal_polarizations=2 keeps the legacy "
                "unit-less 2x multiplier as a mixed-coordinate comparison control and warns at construction."
            ),
        ),
        (
            "pump_pump_raman",
            (
                "Include pump-pump Raman coupling in the coupled ODE. Default True (historical behaviour). "
                "False zeroes only the pump-pump block: pumps are still depleted by the signals and still "
                "follow their own attenuation; pump-signal and signal-signal coupling are unchanged. A C+L "
                "residual-audit control convention, not a more physical default."
            ),
        ),
    ),
)


class HybridRamanPaperModel(PCFMModel):
    """Hybrid Raman model with coupled ODE propagation and closed-form NLI.

    Signals and pumps are solved with the coupled ODE of Eq. (2)/(4).  The
    default ``nli_form="paper"`` evaluates the paper's Eq. (6)-(9): the ODE
    profile is fitted to the five-coefficient form of Eq. (10), then Eq. (8)
    gives the per-interferer XPM and Eq. (9) the SPM; the spans accumulate as
    in Eq. (6) with ``n**epsilon``.  ``nli_form="pcfm"`` keeps the previous
    approximation (PCFM ``K_SCI``/``K_XCI`` on the ODE profile) for regression
    only, and ``nli_form="integral"`` evaluates Eq. (6)-(8) numerically.

    The model keeps its own cross-span ASE state and hands only the newly
    generated Raman ASE to the span EDFA, while the shared route layer
    transports the accumulated ASE with the signal transfer factor.

    ``launch_power_convention`` declares the coordinate of ``power_in_w``.
    ``"total"`` (default) is the historical per-channel dual-polarization total
    power and is bit-for-bit unchanged.  ``"per_pol"`` declares a
    per-polarization input with ``signal_polarizations=2``; the coupled ODE
    state, the NLI and the newly generated EDFA/booster ASE then all stay
    per-polarization (``ase_polarization_scale=0.5``), so ``snr_db`` equals the
    ``"total"`` result of the same physical link.  The Raman ASE is already
    produced in the same coordinate by
    :class:`~network_nlo_eval.models.raman_ode.RamanPaperSpanSolver`, so it is
    never rescaled a second time.  ``"total"`` with ``signal_polarizations=2``
    keeps the legacy unit-less multiplier as a mixed-coordinate control and
    warns at construction.

    Parameters
    ----------
    grid : object
        Spectrum grid.
    ref_fiber_config : FiberSpanConfig
        Reference fibre.
    symbol_rate_hz : float
        Symbol rate (Hz).
    connections : object
        Optional per-connection grid.
    raman_pumps : Sequence[RamanPumpConfig] | None
        Pumps applied to every span.  When omitted, the pumps of
        each :class:`FiberSpanConfig` are used.
    raman_gain_model : str | RamanGainModel
        ``"gaussian"`` (default), ``"linear"`` (CZ),
        ``"table"``, a ``.json``/``.csv`` gain table path, or an explicit
        :class:`~network_nlo_eval.models.raman.RamanGainModel` instance
        (``RamanGainModel.from_file(path)``). Loaded once at construction.
    raman_gain_table : tuple[Sequence[float], Sequence[float]] | None
        ``(offsets_hz, gains)`` normalized gain table used
        when ``raman_gain_model="table"``.  Linear interpolation inside
        the table, zero outside.  It replaces the Gaussian/CZ shape in
        every pump-signal and pump-pump coefficient of Eq. (2) and in the
        Raman ASE of Eq. (4).  Mutually exclusive with an instance or a
        file path.
    raman_ase : bool
        Include the Raman ASE before the EDFA.
    raman_temperature_k : float | None
        Fibre temperature (K).  ``None`` (default) means
        300 K; an explicit value is a calibration override and warns for
        file-backed gain tables, even when it equals 300.0.
    raman_peak_gain_coefficient_m_per_w : float | None
        Peak local Raman gain rate.
        ``None`` (default) means ``6e-14``; an explicit value is a
        calibration override and warns for file-backed gain tables, even
        when it equals 6e-14.
    include_isrs : bool
        Include inter-channel ISRS in the ODE.
    fixed_point_iterations : int
        Maximum BW-pump fixed-point iterations.
    nli_form : str
        ``"paper"`` (default), ``"integral"`` or ``"pcfm"``.
    spm_coherence_exponent : float
        Eq. (6) ``epsilon``.
    poly_degree : int
        PCFM polynomial degree (legacy path only).
    egn : bool
        EGN correction (legacy path only).
    edge_fix : bool
        PCFM edge fix (legacy path only).
    spp_steps : int
        RK4 steps of the ODE and of the Eq. (10) fit.
    pump_depletion : bool
        Let the pumps be depleted by the signals.  ``True``
        (default) solves the full coupled Eq. (2); ``False`` freezes the
        pump evolution to its own attenuation, matching the undepleted-pump
        waveform reference.  A control convention, not a more physical
        default.
    signal_polarizations : int
        Polarizations represented by each ODE signal
        state.  ``1`` (default) is the historical per-polarization drive
        and keeps the old results bit-for-bit; ``2`` uses the
        dual-polarization total signal power as the coupling drive, which
        is the physical waveform convention.  Combine with
        ``launch_power_convention="per_pol"`` for an end-to-end
        per-polarization link.
    pump_pump_raman : bool
        Include pump-pump Raman coupling in the coupled ODE.
        ``True`` (default) keeps the historical behaviour; ``False`` is the
        pump-pump-off control convention of the C+L residual audit.
    launch_power_convention : str
        Declared power convention of ``power_in_w``.
        ``"total"`` (default) keeps the historical per-channel total power
        and is bit-for-bit unchanged.  ``"per_pol"`` declares that the
        input is per-polarization power (physical total = 2x) and requires
        ``signal_polarizations=2``; the ODE state, the NLI and the newly
        generated EDFA/ROADM booster ASE all stay per-polarization, and the
        Raman ASE of the solver is already in that coordinate.
        ``"total"`` with ``signal_polarizations=2`` is retained as the
        legacy unit-less multiplier and emits a :class:`UserWarning` at
        construction.
    **route_options : dict
        Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.

    Raises
    ------
    ValueError
        Invalid NLI form, ODE or PCFM options, or an inconsistent
        ``launch_power_convention``/``signal_polarizations`` pair.
    """

    info = HYBRID_RAMAN_PAPER_MODEL_INFO

    supports_distributed_raman = True
    supports_lumped_loss = False

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str | RamanGainModel = "gaussian",
        raman_gain_table: tuple[Sequence[float], Sequence[float]] | None = None,
        raman_ase: bool = True,
        raman_temperature_k: float | None = None,
        raman_peak_gain_coefficient_m_per_w: float | None = None,
        include_isrs: bool = True,
        fixed_point_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        nli_form: str = "paper",
        spm_coherence_exponent: float = 0.0,
        poly_degree: int = 5,
        egn: bool = False,
        edge_fix: bool = True,
        spp_steps: int = DEFAULT_PAPER_SPP_STEPS,
        pump_depletion: bool = True,
        signal_polarizations: int = 1,
        pump_pump_raman: bool = True,
        launch_power_convention: str = "total",
        **route_options: Any,
    ) -> None:
        """Build the paper model."""
        if fixed_point_iterations < 1:
            msg = "fixed_point_iterations must be >= 1."
            raise ValueError(msg)
        if nli_form not in NLI_FORMS:
            msg = f"nli_form must be one of {NLI_FORMS}, got {nli_form!r}."
            raise ValueError(msg)
        if int(signal_polarizations) < 1:
            msg = "signal_polarizations must be >= 1."
            raise ValueError(msg)
        resolved_convention = resolve_launch_power_convention(
            launch_power_convention, signal_polarizations
        )
        if resolved_convention == "total" and int(signal_polarizations) != 1:
            # Legacy total power with a factor of two double-counts the drive.
            warnings.warn(
                "hybrid_raman_paper: launch_power_convention='total' with "
                f"signal_polarizations={int(signal_polarizations)} multiplies the Raman coupling drive "
                "by an undeclared unit factor: the input power is already the per-channel total power, "
                "so the signal photons are counted twice in the pump depletion and ISRS/NLI drive. "
                "This is a mixed-coordinate comparison control, not a per-pol result. Use "
                "launch_power_convention='per_pol' when power_in_w really is per-polarization power "
                "(the physical total is then 2x), or keep signal_polarizations=1.",
                UserWarning,
                stacklevel=2,
            )
        if not isinstance(cast("object", pump_pump_raman), (bool, np.bool_)):
            msg = f"pump_pump_raman must be a bool, got {pump_pump_raman!r}."
            raise ValueError(msg)  # noqa: TRY004 - Preserve the public exception type.
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            poly_degree=poly_degree,
            egn=egn,
            edge_fix=edge_fix,
            cache_spans=False,
            spp_steps=spp_steps,
            raman_model=None,
            raman_pumps=None,
            raman_gain_model=raman_gain_model,
            raman_gain_table=raman_gain_table,
            raman_ase=raman_ase,
            raman_temperature_k=raman_temperature_k,
            raman_peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            **route_options,
        )
        self._paper_pumps = None if raman_pumps is None else list(raman_pumps)
        self._include_isrs = bool(include_isrs)
        self._fixed_point_iterations = int(fixed_point_iterations)
        self._nli_form = str(nli_form)
        self._spm_exponent = float(spm_coherence_exponent)
        self._pump_depletion = bool(pump_depletion)
        self._signal_polarizations = int(signal_polarizations)
        self._pump_pump_raman = bool(pump_pump_raman)
        self._launch_power_convention = resolved_convention
        # Keep NLI in the signal coordinate after computing it from total power.
        self.nli_power_convention = resolved_convention
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)

    @property
    def launch_power_convention(self) -> str:
        """Declared power convention of ``power_in_w`` (``"total"``/``"per_pol"``)."""
        return self._launch_power_convention

    @property
    def ase_polarization_scale(self) -> float:
        """Scale of newly generated EDFA/ROADM ASE for this model's coordinate.

        An end-to-end per-polarization model returns ``0.5``: its signal, NLI
        and Raman ASE are per-polarization, so the dual-polarization ASE of the
        shared route layer must be halved once.  The default ``"total"``
        coordinate returns ``1.0`` and multiplies nothing.  The route layer
        never scales ``input_ase_w`` (the Raman ASE is already in model
        coordinates).
        """
        if self._launch_power_convention == "per_pol":
            return 0.5
        return 1.0

    def _build_kernel(
        self, fiber_config: FiberSpanConfig
    ) -> HybridRamanPcfmSpanKernel | HybridRamanPaperSpanKernel:
        """Build the span kernel of this fibre for the selected NLI form."""
        pumps = (
            self._paper_pumps
            if self._paper_pumps is not None
            else fiber_config.raman_pumps
        )
        if self._nli_form == "pcfm":
            return HybridRamanPcfmSpanKernel(
                self.f_abs_hz,
                self.channel_bandwidth_hz_channels,
                fiber_config,
                pumps=pumps,
                gain_model=self._raman_gain_model,
                raman_ase=self._raman_ase,
                n_iterations=self._fixed_point_iterations,
                include_isrs=self._include_isrs,
                poly_degree=self.poly_degree,
                symbol_rate_hz=self.symbol_rate_hz,
                egn=self.egn,
                edge_fix=self.edge_fix,
                cache_spans=False,
                spp_steps=self._active_spp_steps,
                pump_depletion=self._pump_depletion,
                signal_polarizations=self._signal_polarizations,
                pump_pump_raman=self._pump_pump_raman,
                launch_power_convention=self._launch_power_convention,
            )
        return HybridRamanPaperSpanKernel(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            pumps=pumps,
            gain_model=self._raman_gain_model,
            raman_ase=self._raman_ase,
            n_steps=self._active_spp_steps,
            n_iterations=self._fixed_point_iterations,
            include_isrs=self._include_isrs,
            nli_form=self._nli_form,
            spm_coherence_exponent=self._spm_exponent,
            pump_depletion=self._pump_depletion,
            signal_polarizations=self._signal_polarizations,
            pump_pump_raman=self._pump_pump_raman,
            launch_power_convention=self._launch_power_convention,
        )

    def kernel_for(
        self, fiber_config: FiberSpanConfig
    ) -> HybridRamanPcfmSpanKernel | HybridRamanPaperSpanKernel:
        """Return the configured hybrid kernel for a fiber."""
        return cast(
            "HybridRamanPcfmSpanKernel | HybridRamanPaperSpanKernel",
            super().kernel_for(fiber_config),
        )

    def evaluate_path(
        self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat
    ) -> PathQoTResult:
        """Evaluate a path, resetting the cross-span ASE and kernel state."""
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)
        for kernel in self._kernel_cache.values():
            reset = getattr(kernel, "reset_state", None)
            if reset is not None:
                reset()
        return super().evaluate_path(optical_path, power_in_w)

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """Single span: ODE kernel, then the shared amplifier handling.

        Parameters
        ----------
        fiber_config : FiberSpanConfig
            Fiber parameters of the span.
        edfa_config : EDFAConfig or None
            Amplifier configuration of the span.
        power_in_w : numpy.ndarray
            ``(N,)`` launch power per channel (W).
        span_length_m : float
            Span length (m).
        span_context : SpanContext | None
            Position of the span in the link.

        Returns
        -------
        ElementOutput
            Output power and noise for the span.
        """
        kernel = self.kernel_for(fiber_config)
        kernel.set_input_ase_w(self._ase_in_w)
        if self._nli_form == "pcfm":
            pcfm_kernel = cast("HybridRamanPcfmSpanKernel", kernel)
            fiber_output = pcfm_kernel.evaluate(
                power_in_w,
                span_length_m,
                1,
                span_index=(
                    self._span_index
                    if span_context is None
                    else int(span_context.span_index)
                ),
                beta2_accum_ps2=self._beta2_accum_ps2,
                span_context=span_context,
            )
        else:
            fiber_output = kernel.evaluate(
                power_in_w,
                span_length_m,
                1,
                span_context=span_context,
            )
        raman_ase_w = kernel.last_raman_ase_w if self._raman_ase else None
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            input_ase_w=raman_ase_w,
            ase_polarization_scale=self.ase_polarization_scale,
        )
        self._ase_in_w = np.array(element.ase_noise_w, dtype=np.float64, copy=True)
        self._span_index += 1
        if self._nli_form == "pcfm":
            self._beta2_accum_ps2 = self._beta2_accum_ps2 + cast(
                "HybridRamanPcfmSpanKernel", kernel
            ).effective_dispersion_ps2(span_length_m)
        return element


register_model(HYBRID_RAMAN_PAPER_MODEL_INFO, HybridRamanPaperModel)
