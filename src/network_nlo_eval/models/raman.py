"""Lumped-equivalent distributed Raman amplification.

This module implements the lumped subset of U. C. de Moura et al.,
"Ultra-Wideband Modeling of Optical Fibre Nonlinearity in Hybrid-Amplified
Links", sections III-B/C. The coupled ODE and NLI models are implemented
separately. For undepleted pumps, on-off gain integrates the exponential pump
attenuation into an effective length and places the gain at the span end.
Forward and backward pumps have equal effective lengths but different spatial
profiles. Raman ASE is integrated along the span with pump and signal loss,
then passed to the span EDFA.

The model neglects pump depletion, pump-pump interactions, second-order pumps,
pump-signal XPM, dynamic control and droop. Lumped gain does not change the
in-span signal profile or NLI. Gaussian and CZ linear gain spectra are
approximations; a measured normalized table can be supplied with zero outside
its range. Frequencies are optical frequencies in Hz, powers in W, and lengths
in m. Defaults allow comparison with a Gaussian reference implementation and
are not a calibration for any particular fiber.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.models.raman_table import (
        RamanGainTable,
        load_raman_gain_table,
        save_raman_gain_table,
    )
    from network_nlo_eval.network.elements import RamanPumpConfig

__all__ = [
    "C_LIGHT",
    "H_PLANCK",
    "K_BOLTZMANN",
    "RamanAmplifier",
    "RamanGainModel",
    "RamanGainTable",
    "load_raman_gain_table",
    "resolve_raman_gain_model",
    "save_raman_gain_table",
]


def __getattr__(name: str) -> Any:
    """Lazily export table helpers to avoid a circular import.

    Parameters
    ----------
    name : str
        Attribute name.

    Returns
    -------
    object
        Requested table helper.
    """
    if name in {"RamanGainTable", "load_raman_gain_table", "save_raman_gain_table"}:
        from network_nlo_eval.models import raman_table  # noqa: PLC0415

        return getattr(raman_table, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


#: Speed of light in vacuum (m/s).
C_LIGHT = 299_792_458.0

#: Planck constant (J s).
H_PLANCK = 6.62607015e-34

#: Boltzmann constant (J/K).
K_BOLTZMANN = 1.380649e-23

#: Polarization-averaged local gain coefficient (m/W); divide by A_eff for 1/(W m).
#:
#: The default supports comparison with a waveform reference; real fibers require calibration.
DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W = 6e-14

#: Frequency shift at the silica Raman gain peak (Hz).
DEFAULT_DELTA_F_PEAK_HZ = 13.2e12

#: Gaussian approximation width (Hz).
DEFAULT_DELTA_F_SIGMA_HZ = 2.12e12

#: CZ linear approximation cutoff (Hz); gain is zero above it.
DEFAULT_DELTA_F_CUTOFF_HZ = 15.0e12


def _looks_like_gain_table_path(value: str) -> bool:
    """Recognize JSON/CSV filenames or strings containing path separators.

    Parameters
    ----------
    value : str
        Potential path or spectrum name.

    Returns
    -------
    bool
        Whether the value resembles a table path.
    """
    lowered = value.lower()
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    return lowered.endswith((".json", ".csv")) or any(
        separator in value for separator in separators
    )


class RamanGainModel:
    """Lumped pump-to-signal Raman gain spectrum and ASE model.

    Parameters
    ----------
    gain_model : {"gaussian", "linear", "table"}, optional
        Gaussian peak, truncated CZ linear approximation, or interpolated table.
    table : tuple of array_like or None, optional
        Frequency offsets (Hz) and normalized gains. Offsets must be nonnegative
        and strictly increasing; gains must be finite and nonnegative. Both arrays
        need at least two points. The spectrum is zero outside the table.
    peak_gain_coefficient_m_per_w : float, optional
        Peak gain coefficient (m/W), divided by effective area in the calculation.
    delta_f_peak_hz, delta_f_sigma_hz, delta_f_cutoff_hz : float, optional
        Peak shift, Gaussian width, and linear cutoff (Hz).
    temperature_k : float, optional
        Fiber temperature (K), used for spontaneous Raman scattering.
    include_photon_energy_factor : bool, optional
        Include the pump-to-signal frequency ratio in the gain coefficient.
    ase_steps : int, optional
        Equally spaced ASE integration nodes; at least eight.
    reference_a_eff_m2 : float, optional
        Effective area (m²) used when one is not supplied.

    Attributes
    ----------
    source_table : RamanGainTable or None
        Source table, if constructed from a file.
    table_metadata : dict
        Provenance used for reporting, not computation.
    reference_wavelength_nm : float or None
        Calibration wavelength for provenance; it does not scale the gain.
    fiber, source, note : str or None
        Fiber and source provenance.
    calibration_overrides : dict
        Explicitly overridden calibration fields.

    Raises
    ------
    ValueError
        If a parameter, spectrum, or table is invalid.
    TypeError
        If an existing model is supplied as a gain model name.
    """

    def __init__(
        self,
        *,
        gain_model: str = "gaussian",
        table: tuple[Sequence[float], Sequence[float]] | None = None,
        peak_gain_coefficient_m_per_w: float = DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
        delta_f_peak_hz: float = DEFAULT_DELTA_F_PEAK_HZ,
        delta_f_sigma_hz: float = DEFAULT_DELTA_F_SIGMA_HZ,
        delta_f_cutoff_hz: float = DEFAULT_DELTA_F_CUTOFF_HZ,
        temperature_k: float = 300.0,
        include_photon_energy_factor: bool = True,
        ase_steps: int = 64,
        reference_a_eff_m2: float = 80e-12,
    ) -> None:
        """Validate and store the gain-model parameters."""
        if isinstance(cast("object", gain_model), RamanGainModel):
            msg = (
                "gain_model must be a string name, not an already constructed RamanGainModel. "
                "File-backed tables are only supported by pcfm (raman_model='lumped'), "
                "pcfm_h3_raman, hybrid_raman_paper and RamanPaperSpanSolver; pass the instance only "
                "to those models' raman_gain_model option."
            )
            raise TypeError(msg)
        if isinstance(cast("object", gain_model), os.PathLike) or (
            isinstance(cast("object", gain_model), str)
            and _looks_like_gain_table_path(gain_model)
        ):
            msg = (
                f"gain_model does not accept the table file {gain_model!r}; construct it with "
                "RamanGainModel.from_file(path) and pass the instance to raman_gain_model. File-backed "
                "tables are supported by pcfm (raman_model='lumped'), pcfm_h3_raman, hybrid_raman_paper "
                "and RamanPaperSpanSolver; buglia2024_ra_gn, zhang2024_dra_egn, numerical_dra, "
                "semrau2019 and ecz do not expose this entry."
            )
            raise ValueError(msg)
        if gain_model not in ("gaussian", "linear", "table"):
            msg = f"gain_model must be 'gaussian', 'linear' or 'table', got {gain_model!r}."
            raise ValueError(msg)
        if gain_model == "table" and table is None:
            msg = "gain_model='table' requires table=(offsets_hz, gains)."
            raise ValueError(msg)
        if gain_model != "table" and table is not None:
            msg = "table is only used when gain_model='table'."
            raise ValueError(msg)
        if peak_gain_coefficient_m_per_w < 0:
            msg = "peak_gain_coefficient_m_per_w must be non-negative."
            raise ValueError(msg)
        if delta_f_peak_hz <= 0 or delta_f_sigma_hz <= 0 or delta_f_cutoff_hz <= 0:
            msg = "Raman frequency scales must be positive."
            raise ValueError(msg)
        if delta_f_cutoff_hz < delta_f_peak_hz:
            msg = "delta_f_cutoff_hz must be >= delta_f_peak_hz."
            raise ValueError(msg)
        if temperature_k <= 0:
            msg = "temperature_k must be positive."
            raise ValueError(msg)
        if ase_steps < 8:
            msg = "ase_steps must be >= 8."
            raise ValueError(msg)
        if reference_a_eff_m2 <= 0:
            msg = "reference_a_eff_m2 must be positive."
            raise ValueError(msg)

        self.gain_model = gain_model
        self.table: tuple[NDArrayFloat, NDArrayFloat] | None = (
            None if table is None else self._validate_gain_table(table)
        )
        self.peak_gain_coefficient_m_per_w = float(peak_gain_coefficient_m_per_w)
        self.delta_f_peak_hz = float(delta_f_peak_hz)
        self.delta_f_sigma_hz = float(delta_f_sigma_hz)
        self.delta_f_cutoff_hz = float(delta_f_cutoff_hz)
        self.temperature_k = float(temperature_k)
        self.include_photon_energy_factor = bool(include_photon_energy_factor)
        self.ase_steps = int(ase_steps)
        self.reference_a_eff_m2 = float(reference_a_eff_m2)

        # Provenance does not affect gain or ASE; reference wavelength does not scale gain.
        self.source_table: RamanGainTable | None = None
        self.table_metadata: dict[str, object] = {}
        self.reference_wavelength_nm: float | None = None
        self.fiber: str | None = None
        self.source: str | None = None
        self.note: str | None = None
        self.calibration_overrides: dict[str, float | None] = {}

    @classmethod
    def from_table(
        cls,
        offsets_hz: Sequence[float],
        gains: Sequence[float],
        *,
        peak_gain_coefficient_m_per_w: float = DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
        temperature_k: float = 300.0,
        include_photon_energy_factor: bool = True,
        ase_steps: int = 64,
        reference_a_eff_m2: float = 80e-12,
    ) -> RamanGainModel:
        """Construct a gain model from a normalized spectrum.

        Offsets must be strictly increasing and nonnegative; gains are normalized
        to the configured peak coefficient. The table is copied and made read-only.

        Parameters
        ----------
        offsets_hz : Sequence[float]
            Nonnegative, strictly increasing pump-signal offsets (Hz).
        gains : Sequence[float]
            Normalized gain values.
        peak_gain_coefficient_m_per_w : float
            Peak gain coefficient (m/W).
        temperature_k : float
            Fiber temperature (K).
        include_photon_energy_factor : bool
            Include the pump-to-signal photon energy ratio.
        ase_steps : int
            Number of ASE integration nodes.
        reference_a_eff_m2 : float
            Reference effective area (m²).

        Returns
        -------
        RamanGainModel
            Model using the validated table.

        Raises
        ------
        ValueError
            If the configuration or gain table is invalid.
        """
        return cls(
            gain_model="table",
            table=(offsets_hz, gains),
            peak_gain_coefficient_m_per_w=peak_gain_coefficient_m_per_w,
            temperature_k=temperature_k,
            include_photon_energy_factor=include_photon_energy_factor,
            ase_steps=ase_steps,
            reference_a_eff_m2=reference_a_eff_m2,
        )

    @classmethod
    def from_file(
        cls,
        path: str | os.PathLike[str],
        *,
        strict_calibration_override: bool = False,
        **overrides: Any,
    ) -> RamanGainModel:
        """Load a measured Raman spectrum from a JSON or CSV table.

        Calibration-field overrides issue a warning and are recorded; strict mode
        rejects them. File provenance is retained and its reference wavelength is
        metadata only. Unknown overrides and changes to the spectrum are rejected.

        Parameters
        ----------
        path : str | Path
            JSON or CSV gain-table path.
        strict_calibration_override : bool
            Reject calibration overrides rather than warn.
        **overrides : object
            Model parameter overrides.

        Returns
        -------
        RamanGainModel
            Model constructed from the file.

        Raises
        ------
        FileNotFoundError
            If the table file does not exist.
        ValueError
            If the file or overrides are invalid.
        """
        # Import lazily to avoid the table module's dependency on this class.
        from network_nlo_eval.models.raman_table import load_raman_gain_table  # noqa: I001, PLC0415

        return load_raman_gain_table(os.fspath(path)).to_model(
            strict_calibration_override=strict_calibration_override,
            **overrides,
        )

    @staticmethod
    def _validate_gain_table(
        table: tuple[Sequence[float], Sequence[float]],
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Validate and freeze a normalized spectrum.

        Absolute gain is set by the peak coefficient and effective area. The table
        need not peak at one; offsets must increase and values must be nonnegative.

        Parameters
        ----------
        table : tuple[Sequence[float], Sequence[float]]
            Pair of offset and normalized-gain arrays.

        Returns
        -------
        tuple of ndarray
            Read-only float64 copies of offsets and gains.

        Raises
        ------
        ValueError
            If the configuration or gain table is invalid.
        """
        try:
            offsets_raw, gains_raw = table
        except (TypeError, ValueError):
            msg = "table must be a pair (offsets_hz, gains)."
            raise ValueError(msg) from None
        offsets = np.array(offsets_raw, dtype=np.float64, copy=True)
        gains = np.array(gains_raw, dtype=np.float64, copy=True)
        if offsets.ndim != 1 or gains.ndim != 1:
            msg = "gain table offsets and gains must be one-dimensional."
            raise ValueError(msg)
        if offsets.size != gains.size:
            msg = "gain table offsets and gains must have the same length."
            raise ValueError(msg)
        if offsets.size < 2:
            msg = "gain table must contain at least two points."
            raise ValueError(msg)
        if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(gains)):
            msg = "gain table values must be finite."
            raise ValueError(msg)
        if np.any(offsets < 0.0):
            msg = "gain table offsets must be non-negative."
            raise ValueError(msg)
        if np.any(np.diff(offsets) <= 0.0):
            msg = "gain table offsets must be strictly increasing."
            raise ValueError(msg)
        if np.any(gains < 0.0):
            msg = "gain table gains must be non-negative."
            raise ValueError(msg)
        offsets.setflags(write=False)
        gains.setflags(write=False)
        return offsets, gains

    def gain_shape(self, shift_hz: NDArrayFloat | float) -> NDArrayFloat:
        """Return the normalized Raman gain shape.

        Nonpositive shifts and table shifts outside the tabulated range give zero.

        Parameters
        ----------
        shift_hz : ndarray | float
            Pump-signal frequency shift (Hz), scalar or array.

        Returns
        -------
        ndarray
            Dimensionless gain shape.
        """
        shift = np.asarray(shift_hz, dtype=np.float64)
        if self.gain_model == "gaussian":
            gaussian = np.exp(
                -0.5 * ((shift - self.delta_f_peak_hz) / self.delta_f_sigma_hz) ** 2
            )
            return np.where(shift > 0, gaussian, 0.0)
        if self.gain_model == "table":
            assert self.table is not None
            offsets, gains = self.table
            # Avoid extrapolating measurements beyond the available spectrum.
            interpolated = np.interp(shift, offsets, gains, left=0.0, right=0.0)
            return np.where(shift > 0, interpolated, 0.0)
        linear = shift / self.delta_f_peak_hz
        return np.where((shift > 0) & (shift <= self.delta_f_cutoff_hz), linear, 0.0)

    @staticmethod
    def _pump_alpha_npm(pump: RamanPumpConfig) -> float:
        """Return the pump power attenuation coefficient (1/m).

        Parameters
        ----------
        pump : RamanPumpConfig
            Pump configuration.

        Returns
        -------
        float
            Pump attenuation (1/m).
        """
        return float(float(pump.attenuation_db_per_km) * np.log(10.0) / 10.0 / 1e3)

    @classmethod
    def _pump_effective_length_m(cls, pump: RamanPumpConfig, length_m: float) -> float:
        """Return the undepleted pump effective length (m).

        Forward and backward pumps have equal end-of-span on-off gain, although
        their spatial profiles differ.

        Parameters
        ----------
        pump : RamanPumpConfig
            Pump configuration.
        length_m : float
            Span length (m).

        Returns
        -------
        float
            Effective length (m).
        """
        alpha = cls._pump_alpha_npm(pump)
        if alpha <= 0:
            return float(length_m)
        return float(-np.expm1(-alpha * length_m) / alpha)

    def _pump_rate_per_w(
        self,
        pump: RamanPumpConfig,
        frequencies_hz: NDArrayFloat,
        a_eff_m2: NDArrayFloat,
    ) -> NDArrayFloat:
        """Return the signed local gain rate per pump power (1/(W m)).

        A higher-frequency pump amplifies the signal; a lower-frequency pump
        depletes it. Effective area is evaluated per channel.

        Parameters
        ----------
        pump : RamanPumpConfig
            Pump configuration.
        frequencies_hz : ndarray
            Signal center frequencies (Hz).
        a_eff_m2 : ndarray
            Scalar or per-channel effective area (m²); None uses the reference.

        Returns
        -------
        ndarray
            Signed local gain rates (1/(W m)).
        """
        pump_frequency_hz = C_LIGHT / (float(pump.wavelength_nm) * 1e-9)
        shift = pump_frequency_hz - frequencies_hz
        shape = self.gain_shape(np.abs(shift))
        coefficient = self.peak_gain_coefficient_m_per_w / a_eff_m2
        if self.include_photon_energy_factor:
            coefficient = coefficient * (pump_frequency_hz / frequencies_hz)
        return cast("NDArrayFloat", np.sign(shift) * coefficient * shape)

    def _pump_power_profile(
        self, pump: RamanPumpConfig, z_m: NDArrayFloat, length_m: float
    ) -> NDArrayFloat:
        """Return pump power along the span (W).

        Parameters
        ----------
        pump : RamanPumpConfig
            Pump configuration.
        z_m : ndarray
            Positions along the span (m).
        length_m : float
            Span length (m).

        Returns
        -------
        ndarray
            Pump power at each position (W).
        """
        alpha = self._pump_alpha_npm(pump)
        if str(pump.direction) == "forward":
            return cast("NDArrayFloat", float(pump.power_w) * np.exp(-alpha * z_m))
        return cast(
            "NDArrayFloat", float(pump.power_w) * np.exp(-alpha * (length_m - z_m))
        )

    def _pump_nsp(
        self, pump: RamanPumpConfig, frequencies_hz: NDArrayFloat
    ) -> NDArrayFloat:
        """Return the spontaneous scattering factor for one pump.

        ``n_sp = 1 / (1 - exp(-h*abs(delta_f)/(k_B*T)))``. At zero shift the
        function returns one, where the gain shape is also zero.

        Parameters
        ----------
        pump : RamanPumpConfig
            Pump configuration.
        frequencies_hz : ndarray
            Signal center frequencies (Hz).

        Returns
        -------
        ndarray
            Spontaneous scattering factors.
        """
        shift = np.abs(C_LIGHT / (float(pump.wavelength_nm) * 1e-9) - frequencies_hz)
        x = H_PLANCK * shift / (K_BOLTZMANN * self.temperature_k)
        safe = x > 1e-12
        nsp = np.ones_like(x)
        nsp[safe] = 1.0 / (-np.expm1(-x[safe]))
        return cast("NDArrayFloat", nsp)

    def on_off_gain(
        self,
        frequencies_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        a_eff_m2: NDArrayFloat | float | None = None,
    ) -> NDArrayFloat:
        """Calculate linear Raman on-off gain at the end of the span.

        Parameters
        ----------
        frequencies_hz : ndarray
            Signal center frequencies (Hz).
        pumps : Sequence[RamanPumpConfig]
            Forward and backward pump configurations.
        length_m : float
            Span length (m).
        a_eff_m2 : ndarray | float | None
            Scalar or per-channel effective area (m²); None uses the reference.

        Returns
        -------
        ndarray
            Linear on-off gain per channel; unity without pump power.
        """
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        a_eff = self._resolve_a_eff(a_eff_m2, frequencies.shape)
        exponent = np.zeros_like(frequencies)
        for pump in pumps:
            if float(pump.power_w) <= 0:
                continue
            rate = self._pump_rate_per_w(pump, frequencies, a_eff)
            exponent += (
                rate
                * float(pump.power_w)
                * self._pump_effective_length_m(pump, length_m)
            )
        return np.exp(exponent)

    def on_off_gain_db(
        self,
        frequencies_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        a_eff_m2: NDArrayFloat | float | None = None,
    ) -> NDArrayFloat:
        """Calculate Raman on-off gain at the end of the span (dB).

        Parameters
        ----------
        frequencies_hz : ndarray
            Signal center frequencies (Hz).
        pumps : Sequence[RamanPumpConfig]
            Forward and backward pump configurations.
        length_m : float
            Span length (m).
        a_eff_m2 : ndarray | float | None
            Scalar or per-channel effective area (m²); None uses the reference.

        Returns
        -------
        ndarray
            On-off gain per channel (dB).
        """
        gain = self.on_off_gain(frequencies_hz, pumps, length_m, a_eff_m2=a_eff_m2)
        return 10.0 * np.log10(gain)

    def ase_w(
        self,
        frequencies_hz: NDArrayFloat,
        bandwidths_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        a_eff_m2: NDArrayFloat | float | None = None,
        signal_attenuation_npm: NDArrayFloat | float | None = None,
    ) -> NDArrayFloat:
        """Calculate Raman ASE power at the span end, before the EDFA.

        The span integral uses the local gain rate ``g`` and its positive part
        ``g_pos``::

            P_ASE = 2*n_sp*h*f*B * integral_0^L g_pos(z)
                    * exp(-alpha_s*(L-z)) * exp(integral_z^L g(z') dz') dz

        Only amplification contributes to the ASE source; signal and pump losses
        are retained. This distributed estimate is more conservative than the simple
        ``2*n_sp*h*f*B*(G_R-1)`` approximation.

        Parameters
        ----------
        frequencies_hz : ndarray
            Signal center frequencies (Hz).
        bandwidths_hz : ndarray
            Signal bandwidth per channel (Hz).
        pumps : Sequence[RamanPumpConfig]
            Forward and backward pump configurations.
        length_m : float
            Span length (m).
        a_eff_m2 : ndarray | float | None
            Scalar or per-channel effective area (m²); None uses the reference.
        signal_attenuation_npm : ndarray | float | None
            Scalar or per-channel signal attenuation (1/m); None uses 0.2 dB/km.

        Returns
        -------
        ndarray
            Raman ASE per channel (W).
        """
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
        if bandwidths.shape != frequencies.shape:
            msg = "bandwidths_hz must contain one value per signal frequency."
            raise ValueError(msg)
        if length_m <= 0:
            msg = "length_m must be positive."
            raise ValueError(msg)
        a_eff = self._resolve_a_eff(a_eff_m2, frequencies.shape)
        alpha_s = self._resolve_signal_attenuation(
            signal_attenuation_npm, frequencies.shape
        )
        if not any(float(pump.power_w) > 0 for pump in pumps):
            return np.zeros_like(frequencies)

        z_m = np.linspace(0.0, float(length_m), self.ase_steps)
        rate_total = np.zeros((frequencies.size, z_m.size), dtype=np.float64)
        rate_source = np.zeros_like(rate_total)
        nsp_weighted = np.zeros_like(rate_total)
        for pump in pumps:
            if float(pump.power_w) <= 0:
                continue
            rate = (
                self._pump_rate_per_w(pump, frequencies, a_eff)[:, None]
                * self._pump_power_profile(pump, z_m, length_m)[None, :]
            )
            positive = np.maximum(rate, 0.0)
            rate_total += rate
            rate_source += positive
            nsp_weighted += positive * self._pump_nsp(pump, frequencies)[:, None]
        weight = trapezoid(rate_source, z_m, axis=1)
        nsp = np.divide(
            trapezoid(nsp_weighted, z_m, axis=1),
            weight,
            out=np.ones_like(weight),
            where=weight > 0,
        )
        cumulative = cumulative_trapezoid(rate_total, z_m, axis=1, initial=0)
        tail = cumulative[:, -1:] - cumulative
        integrand = (
            rate_source
            * np.exp(-alpha_s[:, None] * (float(length_m) - z_m)[None, :])
            * np.exp(tail)
        )
        ase = (
            2.0
            * nsp
            * H_PLANCK
            * frequencies
            * bandwidths
            * trapezoid(integrand, z_m, axis=1)
        )
        return cast("NDArrayFloat", np.maximum(ase, 0.0))

    def _resolve_a_eff(
        self,
        a_eff_m2: NDArrayFloat | float | None,
        shape: tuple[int, ...],
    ) -> NDArrayFloat:
        """Broadcast effective area to the signal shape.

        Parameters
        ----------
        a_eff_m2 : ndarray | float | None
            Scalar or per-channel effective area (m²); None uses the reference.
        shape : tuple[int, ...]
            Target signal array shape.

        Returns
        -------
        ndarray
            Effective area per channel (m²).
        """
        if a_eff_m2 is None:
            return np.full(shape, self.reference_a_eff_m2, dtype=np.float64)
        values = np.asarray(a_eff_m2, dtype=np.float64)
        if values.ndim == 0:
            return np.full(shape, float(values), dtype=np.float64)
        if values.shape != shape:
            msg = "a_eff_m2 must be a scalar or contain one value per signal frequency."
            raise ValueError(msg)
        return values

    @staticmethod
    def _resolve_signal_attenuation(
        signal_attenuation_npm: NDArrayFloat | float | None,
        shape: tuple[int, ...],
    ) -> NDArrayFloat:
        """Broadcast signal attenuation to the signal shape.

        Parameters
        ----------
        signal_attenuation_npm : ndarray | float | None
            Scalar or per-channel signal attenuation (1/m); None uses 0.2 dB/km.
        shape : tuple[int, ...]
            Target signal array shape.

        Returns
        -------
        ndarray
            Signal attenuation per channel (1/m).
        """
        if signal_attenuation_npm is None:
            return np.full(shape, 0.2 * np.log(10.0) / 10.0 / 1e3, dtype=np.float64)
        values = np.asarray(signal_attenuation_npm, dtype=np.float64)
        if values.ndim == 0:
            return np.full(shape, float(values), dtype=np.float64)
        if values.shape != shape:
            msg = "signal_attenuation_npm must be a scalar or contain one value per signal."
            raise ValueError(msg)
        return values


def resolve_raman_gain_model(
    gain_model: str | os.PathLike[str] | RamanGainModel,
    *,
    table: tuple[Sequence[float], Sequence[float]] | None = None,
    peak_gain_coefficient_m_per_w: float | None = None,
    temperature_k: float | None = None,
) -> RamanGainModel:
    """Resolve a gain-model instance, named spectrum, or table file.

    An existing model is returned unchanged. For file paths, explicitly supplied
    calibration fields override the file and trigger its override policy, even
    when equal to defaults. Only ``None`` means that a value was not supplied.

    Parameters
    ----------
    gain_model : str | os.PathLike[str] | RamanGainModel
        Named spectrum, file path, or existing gain model.
    table : tuple[Sequence[float], Sequence[float]] | None
        Pair of offset and normalized-gain arrays.
    peak_gain_coefficient_m_per_w : float | None
        Peak gain coefficient (m/W).
    temperature_k : float | None
        Fiber temperature (K).

    Returns
    -------
    RamanGainModel
        Resolved gain model.

    Raises
    ------
    ValueError
        If the configuration or gain table is invalid.
    """
    if isinstance(gain_model, RamanGainModel):
        if table is not None:
            msg = "table must not be given when raman_gain_model is already a RamanGainModel."
            raise ValueError(msg)
        return gain_model
    is_path = isinstance(gain_model, os.PathLike) or (
        isinstance(cast("object", gain_model), str)
        and _looks_like_gain_table_path(gain_model)
    )
    if is_path:
        if table is not None:
            msg = (
                "raman_gain_table must not be given together with a Raman gain table file path; "
                "the table is read from the file."
            )
            raise ValueError(msg)
        # None alone means omitted; explicit defaults still override file calibration.
        overrides: dict[str, Any] = {}
        if peak_gain_coefficient_m_per_w is not None:
            overrides["peak_gain_coefficient_m_per_w"] = float(
                peak_gain_coefficient_m_per_w
            )
        if temperature_k is not None:
            overrides["temperature_k"] = float(temperature_k)
        return RamanGainModel.from_file(gain_model, **overrides)
    return RamanGainModel(
        gain_model=cast("str", gain_model),
        table=table,
        # Only omitted values use the defaults.
        peak_gain_coefficient_m_per_w=(
            DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W
            if peak_gain_coefficient_m_per_w is None
            else peak_gain_coefficient_m_per_w
        ),
        temperature_k=300.0 if temperature_k is None else temperature_k,
    )


class RamanAmplifier:
    """Lumped Raman amplifier for a pump set and span.

    The span-end gain scales the signal; the calculated Raman ASE enters the EDFA.

    Parameters
    ----------
    pumps : sequence of RamanPumpConfig
        Forward and backward pumps.
    length_m : float
        Span length (m).
    model : RamanGainModel or None, optional
        Gain model; ``None`` selects the default Gaussian model.
    a_eff_m2 : float or ndarray or None, optional
        Scalar or per-channel effective area (m²).
    signal_attenuation_npm : float or ndarray or None, optional
        Scalar or per-channel signal attenuation (1/m).

    Raises
    ------
    ValueError
        If the span length is not positive.
    """

    def __init__(
        self,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        model: RamanGainModel | None = None,
        a_eff_m2: NDArrayFloat | float | None = None,
        signal_attenuation_npm: NDArrayFloat | float | None = None,
    ) -> None:
        """Store the pumps and span geometry."""
        if length_m <= 0:
            msg = "length_m must be positive."
            raise ValueError(msg)
        self.pumps = list(pumps)
        self.length_m = float(length_m)
        self.model = model if model is not None else RamanGainModel()
        self.a_eff_m2 = a_eff_m2
        self.signal_attenuation_npm = signal_attenuation_npm

    @property
    def active(self) -> bool:
        """Return whether at least one pump has nonzero power.

        Returns
        -------
        bool
            Whether a pump is active.
        """
        return any(float(pump.power_w) > 0 for pump in self.pumps)

    def gain(self, frequencies_hz: NDArrayFloat) -> NDArrayFloat:
        """Return linear on-off gain at the span end.

        Parameters
        ----------
        frequencies_hz : ndarray
            Signal center frequencies (Hz).

        Returns
        -------
        ndarray
            Linear gain per channel.
        """
        return self.model.on_off_gain(
            frequencies_hz, self.pumps, self.length_m, a_eff_m2=self.a_eff_m2
        )

    def gain_db(self, frequencies_hz: NDArrayFloat) -> NDArrayFloat:
        """Return on-off gain at the span end (dB).

        Parameters
        ----------
        frequencies_hz : ndarray
            Signal center frequencies (Hz).

        Returns
        -------
        ndarray
            Gain per channel (dB).
        """
        return self.model.on_off_gain_db(
            frequencies_hz, self.pumps, self.length_m, a_eff_m2=self.a_eff_m2
        )

    def ase_w(
        self, frequencies_hz: NDArrayFloat, bandwidths_hz: NDArrayFloat
    ) -> NDArrayFloat:
        """Return Raman ASE power at the span end (W).

        Parameters
        ----------
        frequencies_hz : ndarray
            Signal center frequencies (Hz).
        bandwidths_hz : ndarray
            Signal bandwidth per channel (Hz).

        Returns
        -------
        ndarray
            Raman ASE per channel (W).
        """
        return self.model.ase_w(
            frequencies_hz,
            bandwidths_hz,
            self.pumps,
            self.length_m,
            a_eff_m2=self.a_eff_m2,
            signal_attenuation_npm=self.signal_attenuation_npm,
        )

    def apply(
        self,
        power_w: NDArrayFloat,
        frequencies_hz: NDArrayFloat,
        bandwidths_hz: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Apply span-end gain to the signal and return the Raman ASE.

        Parameters
        ----------
        power_w : ndarray
            Span-end signal powers (W).
        frequencies_hz : ndarray
            Signal center frequencies (Hz).
        bandwidths_hz : ndarray
            Signal bandwidth per channel (Hz).

        Returns
        -------
        tuple of ndarray
            New signal-power and ASE arrays (W).
        """
        gain = self.gain(frequencies_hz)
        return np.asarray(power_w, dtype=np.float64) * gain, self.ase_w(
            frequencies_hz, bandwidths_hz
        )
