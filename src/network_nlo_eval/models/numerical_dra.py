"""Independent numerical distributed-Raman reference model.

Signal and pump powers are solved with adaptive DOP853 integration; signal ASE
is integrated separately after power convergence. NLI is calculated directly
from the raw ODE profile using numerical frequency and longitudinal integrals,
without the fitted profiles or closed forms of the models under comparison.
Backward pumps use a relaxed fixed-point iteration. Frequency interpolation is
linear between channel centers, and the XPM quadrature resolves the coherence
peak with a tangent transform and truncates distant oscillatory tails.

The negative Raman coupling uses receiver/source frequency to conserve photon
number for frequency-independent effective area; the printed paper uses its
reciprocal. Inter-span NLI is added as power, pump ASE is omitted, and the gain
spectrum uses the repository's Gaussian or linear approximation. Signal
polarizations scale the coupling drive, while the receiver state remains in
its input power coordinate.

References
----------
H. Buglia et al., "A Closed-Form Expression for the Gaussian Noise Model in the
Presence of Raman Amplification", JLT 42(2), 2024, Eqs. (1)-(8).
C. Semrau et al., "The Gaussian Noise Model in the Presence of Inter-Channel
Stimulated Raman Scattering", JLT, 2018.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from scipy.integrate import solve_ivp

from network_nlo_eval.models.base import ModelInfo, SpanFiberOutput, register_model
from network_nlo_eval.models.dra_common import (
    DEFAULT_DRA_NLI_QUADRATURE,
    DEFAULT_DRA_STEPS,
    DRARamanModelBase,
    DRASpanKernelBase,
)
from network_nlo_eval.models.raman import C_LIGHT, H_PLANCK, K_BOLTZMANN, RamanGainModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "NUMERICAL_DRA_MODEL_INFO",
    "NumericalDRAKernel",
    "NumericalDRAModel",
    "NumericalDRASpanResult",
    "NumericalDRASpanSolver",
    "ODESpanProfile",
    "nli_power_from_ode_profile",
]

#: Use an odd number of longitudinal samples for composite Simpson integration.
DEFAULT_NUMERICAL_DRA_PROFILE_POINTS: int = 257

#: DOP853 tolerances for the numerical reference.
DEFAULT_NUMERICAL_DRA_RTOL: float = 1.0e-10
DEFAULT_NUMERICAL_DRA_ATOL: float = 1.0e-15

#: XPM nodes: resolve the coherence peak along x and integrate smoothly along y.
DEFAULT_XPM_QUADRATURE_X: int = 16
DEFAULT_XPM_QUADRATURE_Y: int = 6

#: Truncate XPM phase to bound oscillations resolvable by the longitudinal grid.
#: At a limit of 200 the omitted Lorentzian tail is approximately one percent.
DEFAULT_XPM_PHASE_LIMIT: float = 300.0

#: SPM has a logarithmic peak at the origin and needs a denser two-dimensional grid.
DEFAULT_SPM_QUADRATURE_X: int = 48
DEFAULT_SPM_QUADRATURE_Y: int = 24

#: Backward-pump relaxation and convergence threshold relative to peak pump power.
_BW_RELAXATION: float = 0.5
_BW_TOLERANCE: float = 1.0e-7

#: Power floor to avoid division by zero.
_POWER_FLOOR: float = 1.0e-30


class _OdeSolution(Protocol):
    """Dense output returned by a successful SciPy integration."""

    def sol(self, t: float | NDArrayFloat) -> NDArrayFloat:
        """Evaluate the integrated state at one or more positions."""
        ...


@dataclass(frozen=True)
class NumericalDRASpanResult:
    """Numerical ODE solution for a single span.

    Attributes
    ----------
    profile_norm : numpy.ndarray
        Normalized signal power profile at the longitudinal sample points.
    z_m : numpy.ndarray
        Longitudinal positions in meters.
    power_out_w, raman_ase_w : numpy.ndarray
        Output signal power and newly generated Raman ASE in watts.
    pump_power_out_w : numpy.ndarray
        Pump powers at their respective exit ends in watts.
    iterations : int
        Number of backward-pump fixed-point iterations.
    converged : bool
        Whether the iteration converged.
    """

    profile_norm: NDArrayFloat
    z_m: NDArrayFloat
    power_out_w: NDArrayFloat
    raman_ase_w: NDArrayFloat
    pump_power_out_w: NDArrayFloat
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ODESpanProfile:
    """Unfitted ODE power profile used for the NLI integral.

    Attributes
    ----------
    profile_norm : numpy.ndarray
        Normalized profile in descending channel frequency order.
    z_m : numpy.ndarray
        Longitudinal positions in meters.
    length_m : float
        Span length in meters.
    frequencies_hz : numpy.ndarray
        Channel frequencies in the profile order.
    """

    profile_norm: NDArrayFloat
    z_m: NDArrayFloat
    length_m: float
    frequencies_hz: NDArrayFloat


class NumericalDRASpanSolver:
    """Independently solve a span's signal, pump and ASE equations with DOP853.

    Power and ASE are integrated separately, unlike the fixed-step RK4 solver;
    profiles are sampled from the adaptive integrator's dense output.

    Parameters
    ----------
    signal_frequencies_hz, signal_bandwidths_hz : numpy.ndarray
        Signal center frequencies and bandwidths in Hz.
    pumps : sequence of RamanPumpConfig
        Positive-power forward or backward pumps.
    length_m : float
        Span length in meters.
    signal_alpha_npm, pump_alpha_npm : numpy.ndarray
        Signal and pump attenuations in 1/m.
    signal_a_eff_m2, pump_a_eff_m2 : numpy.ndarray
        Signal and pump effective areas in square meters.
    gain_model : RamanGainModel
        Raman gain spectrum.
    n_profile : int, optional
        Odd number of profile sample points.
    rtol, atol : float, optional
        Relative and absolute DOP853 tolerances.
    n_iterations : int, optional
        Maximum backward-pump fixed-point iterations.
    include_isrs, include_ase : bool, optional
        Enable signal-signal ISRS and signal ASE.
    signal_polarizations : int, optional
        Multiplier applied to the signal coupling drive, while the receiver state
        remains in its original coordinate.

    Raises
    ------
    ValueError
        If input shapes, span length or profile point count are invalid.
    """

    def __init__(
        self,
        signal_frequencies_hz: NDArrayFloat,
        signal_bandwidths_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        signal_alpha_npm: NDArrayFloat,
        signal_a_eff_m2: NDArrayFloat,
        pump_alpha_npm: NDArrayFloat,
        pump_a_eff_m2: NDArrayFloat,
        gain_model: RamanGainModel,
        *,
        n_profile: int = DEFAULT_NUMERICAL_DRA_PROFILE_POINTS,
        rtol: float = DEFAULT_NUMERICAL_DRA_RTOL,
        atol: float = DEFAULT_NUMERICAL_DRA_ATOL,
        n_iterations: int = 24,
        include_isrs: bool = True,
        include_ase: bool = True,
        signal_polarizations: int = 1,
    ) -> None:
        if length_m <= 0:
            msg = "length_m must be positive."
            raise ValueError(msg)
        if n_profile < 5 or n_profile % 2 == 0:
            msg = "n_profile must be an odd integer >= 5."
            raise ValueError(msg)
        if n_iterations < 1:
            msg = "n_iterations must be >= 1."
            raise ValueError(msg)
        if rtol <= 0 or atol <= 0:
            msg = "rtol and atol must be positive."
            raise ValueError(msg)
        if int(signal_polarizations) < 1:
            msg = "signal_polarizations must be >= 1."
            raise ValueError(msg)

        self.signal_frequencies_hz = np.asarray(signal_frequencies_hz, dtype=np.float64)
        self.signal_bandwidths_hz = np.asarray(signal_bandwidths_hz, dtype=np.float64)
        self.signal_alpha_npm = np.asarray(signal_alpha_npm, dtype=np.float64)
        self.signal_a_eff_m2 = np.asarray(signal_a_eff_m2, dtype=np.float64)
        self.pumps = list(pumps)
        self.length_m = float(length_m)
        self.gain_model = gain_model
        self.n_profile = int(n_profile)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.n_iterations = int(n_iterations)
        self.include_isrs = bool(include_isrs)
        self.include_ase = bool(include_ase)
        self.signal_polarizations = int(signal_polarizations)

        n_signal = self.signal_frequencies_hz.size
        if self.signal_bandwidths_hz.shape != (n_signal,):
            msg = "signal_bandwidths_hz must contain one value per signal."
            raise ValueError(msg)
        if self.signal_alpha_npm.shape != (n_signal,):
            msg = "signal_alpha_npm must contain one value per signal."
            raise ValueError(msg)
        if self.signal_a_eff_m2.shape != (n_signal,):
            msg = "signal_a_eff_m2 must contain one value per signal."
            raise ValueError(msg)

        pump_f = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in self.pumps],
            dtype=np.float64,
        )
        pump_alpha = np.asarray(pump_alpha_npm, dtype=np.float64)
        pump_aeff = np.asarray(pump_a_eff_m2, dtype=np.float64)
        if pump_alpha.shape != (len(self.pumps),):
            msg = "pump_alpha_npm must contain one value per pump."
            raise ValueError(msg)
        if pump_aeff.shape != (len(self.pumps),):
            msg = "pump_a_eff_m2 must contain one value per pump."
            raise ValueError(msg)

        self._n_signal = n_signal
        self._n_pump = len(self.pumps)
        self._pump_power_w = np.asarray(
            [float(pump.power_w) for pump in self.pumps], dtype=np.float64
        )
        self._pump_forward = np.asarray(
            [str(pump.direction) == "forward" for pump in self.pumps], dtype=bool
        )

        # Combine waves as signals followed by pumps, consistently with the coupling matrix.
        self._f_hz = np.concatenate([self.signal_frequencies_hz, pump_f])
        self._alpha_npm = np.concatenate([self.signal_alpha_npm, pump_alpha])
        self._a_eff_m2 = np.concatenate([self.signal_a_eff_m2, pump_aeff])
        self._bandwidth_hz = np.concatenate(
            [self.signal_bandwidths_hz, np.zeros(self._n_pump, dtype=np.float64)]
        )
        self._is_pump = np.concatenate(
            [np.zeros(n_signal, dtype=bool), np.ones(self._n_pump, dtype=bool)]
        )
        self._direction = np.concatenate(
            [
                np.ones(n_signal, dtype=np.float64),
                np.where(self._pump_forward, 1.0, -1.0),
            ]
        )
        self._n_waves = self._f_hz.size

        # The coupling drive scales signal state by its polarization count; pumps are
        # single-polarization. A factor of one preserves the historical numerical path.
        self._drive_scale = np.where(
            self._is_pump,
            1.0,
            float(self.signal_polarizations),
        ).astype(np.float64)

        self._signal_slots = np.arange(n_signal, dtype=np.int64)
        self._fwd_slots = np.flatnonzero(self._direction > 0).astype(np.int64)
        self._bw_slots = np.flatnonzero(self._direction < 0).astype(np.int64)
        self._fwd_is_signal = ~self._is_pump[self._fwd_slots]
        self._fwd_signal_local = np.flatnonzero(self._fwd_is_signal).astype(np.int64)
        self._fwd_signal_original = self._fwd_slots[self._fwd_signal_local]
        self._fwd_pump_original = self._fwd_slots[~self._fwd_is_signal] - n_signal
        self._bw_pump_original = self._bw_slots - n_signal

        self._alpha_fwd = np.ascontiguousarray(self._alpha_npm[self._fwd_slots])
        self._alpha_bw = np.ascontiguousarray(self._alpha_npm[self._bw_slots])
        self._alpha_signal = np.ascontiguousarray(self._alpha_npm[self._signal_slots])
        self._f_signal = np.ascontiguousarray(self._f_hz[self._signal_slots])
        self._b_signal = np.ascontiguousarray(self._bandwidth_hz[self._signal_slots])

        self._coupling = self._build_coupling_matrix()
        self._source = self._build_source_matrix()
        self._coupling_fwd = np.ascontiguousarray(self._coupling[self._fwd_slots, :])
        self._coupling_bw = np.ascontiguousarray(self._coupling[self._bw_slots, :])
        self._coupling_signal = np.ascontiguousarray(
            self._coupling[self._signal_slots, :]
        )
        self._source_signal = np.ascontiguousarray(self._source[self._signal_slots, :])
        self._z_profile = np.linspace(
            0.0, self.length_m, self.n_profile, dtype=np.float64
        )

    @property
    def has_backward_pumps(self) -> bool:
        """Return whether the span has backward pumps."""
        return bool(self._bw_slots.size)

    def _build_coupling_matrix(self) -> NDArrayFloat:
        """Build the Raman coupling matrix in 1/(W m).

        Negative coupling uses the receiver/source frequency ratio to conserve photon
        number when effective area is frequency-independent.
        """
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        shape = self.gain_model.gain_shape(delta)
        gain = (
            self.gain_model.peak_gain_coefficient_m_per_w
            / self._a_eff_m2[:, None]
            * shape
        )
        # Higher-frequency sources supply power; the receiver/source frequency ratio
        # for negative coupling conserves photon number.
        source_higher = self._f_hz[None, :] > self._f_hz[:, None]
        coupling = np.where(
            source_higher,
            gain,
            -(self._f_hz[:, None] / self._f_hz[None, :]) * gain,
        )
        np.fill_diagonal(coupling, 0.0)
        if not self.include_isrs:
            slots = self._signal_slots
            coupling[np.ix_(slots, slots)] = 0.0
        return np.ascontiguousarray(coupling)

    def _build_source_matrix(self) -> NDArrayFloat:
        """Build the spontaneous Raman source matrix from positive coupling."""
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        safe_delta = np.where(delta > 0.0, delta, 1.0)
        x = H_PLANCK * safe_delta / (K_BOLTZMANN * self.gain_model.temperature_k)
        kappa = np.ones_like(x)
        mask = x > 1.0e-12
        kappa[mask] = 1.0 / (-np.expm1(-x[mask]))
        return np.ascontiguousarray(np.maximum(self._coupling, 0.0) * kappa)

    def _initial_forward_state(self, signal_power_in_w: NDArrayFloat) -> NDArrayFloat:
        """Construct the initial forward signal and pump state."""
        state = np.zeros(self._fwd_slots.size, dtype=np.float64)
        state[self._fwd_signal_local] = signal_power_in_w[self._fwd_signal_original]
        pump_mask = ~self._fwd_is_signal
        if np.any(pump_mask):
            state[pump_mask] = self._pump_power_w[self._fwd_pump_original]
        return state

    def _initial_backward_profile(self) -> NDArrayFloat:
        """Construct the passive backward-pump profile from z=L to z=0."""
        n_bw = self._bw_slots.size
        profile = np.empty((self.n_profile, n_bw), dtype=np.float64)
        for column, pump_index in enumerate(self._bw_pump_original):
            alpha = self._alpha_npm[self._bw_slots[column]]
            profile[:, column] = self._pump_power_w[pump_index] * np.exp(
                -alpha * (self.length_m - self._z_profile)
            )
        return profile

    def _interpolate_backward(self, z_m: float, profile: NDArrayFloat) -> NDArrayFloat:
        """Interpolate the backward-pump profile at the given position."""
        n_bw = self._bw_slots.size
        if n_bw == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(
            [
                np.interp(z_m, self._z_profile, profile[:, column])
                for column in range(n_bw)
            ],
            dtype=np.float64,
        )

    def _forward_rhs(self, state: NDArrayFloat, backward: NDArrayFloat) -> NDArrayFloat:
        """Evaluate forward signal and pump power derivatives."""
        total = np.empty(self._n_waves, dtype=np.float64)
        total[self._fwd_slots] = state
        total[self._bw_slots] = backward
        # Use total source drive for coupling but apply receiver state only once.
        drive = total * self._drive_scale
        return cast(
            "NDArrayFloat",
            state * (self._coupling_fwd @ drive) - self._alpha_fwd * state,
        )

    def _backward_rhs(
        self,
        state: NDArrayFloat,
        forward_state: NDArrayFloat,
    ) -> NDArrayFloat:
        """Evaluate backward-pump derivatives using ``s = L - z``."""
        total = np.empty(self._n_waves, dtype=np.float64)
        total[self._fwd_slots] = forward_state
        total[self._bw_slots] = state
        drive = total * self._drive_scale
        return cast(
            "NDArrayFloat", state * (self._coupling_bw @ drive) - self._alpha_bw * state
        )

    def _solve_forward(
        self, signal_power_in_w: NDArrayFloat, backward_profile: NDArrayFloat
    ) -> _OdeSolution:
        """Integrate forward waves for a given backward-pump profile."""
        initial = self._initial_forward_state(signal_power_in_w)

        def rhs(z_m: float, state: NDArrayFloat) -> NDArrayFloat:
            return self._forward_rhs(
                state, self._interpolate_backward(z_m, backward_profile)
            )

        solution = solve_ivp(
            rhs,
            (0.0, self.length_m),
            initial,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
        )
        if not solution.success:
            msg = f"solve_ivp forward failed: {solution.message}"
            raise RuntimeError(msg)
        return cast("_OdeSolution", solution)

    def _solve_backward(self, forward_solution: _OdeSolution) -> _OdeSolution:
        """Integrate backward pumps for a given forward-wave solution."""
        initial = self._pump_power_w[self._bw_pump_original].copy()

        def rhs(s_m: float, state: NDArrayFloat) -> NDArrayFloat:
            return self._backward_rhs(
                state, np.asarray(forward_solution.sol(self.length_m - s_m))
            )

        solution = solve_ivp(
            rhs,
            (0.0, self.length_m),
            initial,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
        )
        if not solution.success:
            msg = f"solve_ivp backward failed: {solution.message}"
            raise RuntimeError(msg)
        return cast("_OdeSolution", solution)

    def _solve_ase(
        self,
        signal_power_in_w: NDArrayFloat,  # noqa: ARG002 - Retain the solver interface.
        ase_in_w: NDArrayFloat,
        forward_solution: _OdeSolution,
        backward_solution: _OdeSolution | None,
    ) -> NDArrayFloat:
        """Integrate the linear signal ASE equation for the converged power profile."""
        if not self.include_ase:
            return np.zeros(self._n_signal, dtype=np.float64)

        def rhs(z_m: float, ase: NDArrayFloat) -> NDArrayFloat:
            forward_state = np.asarray(forward_solution.sol(z_m))
            backward_state = (
                np.asarray(backward_solution.sol(self.length_m - z_m))
                if backward_solution is not None
                else np.zeros(0, dtype=np.float64)
            )
            total = np.empty(self._n_waves, dtype=np.float64)
            total[self._fwd_slots] = forward_state
            total[self._bw_slots] = backward_state
            # Signal ASE contributes to stimulated gain and spontaneous emission; pump ASE is zero.
            total[self._signal_slots] += ase
            # Stimulated gain uses the physical drive; the spontaneous source is not
            # multiplied by the polarization count again.
            gain_rate = self._coupling_signal @ (total * self._drive_scale)
            source_rate = (
                2.0
                * H_PLANCK
                * self._f_signal
                * self._b_signal
                * (self._source_signal @ total)
            )
            return cast(
                "NDArrayFloat", gain_rate * ase - self._alpha_signal * ase + source_rate
            )

        solution = solve_ivp(
            rhs,
            (0.0, self.length_m),
            ase_in_w,
            method="DOP853",
            rtol=max(self.rtol, 1.0e-9),
            atol=max(self.atol, 1.0e-18),
        )
        if not solution.success:
            msg = f"solve_ivp ASE failed: {solution.message}"
            raise RuntimeError(msg)
        return np.asarray(solution.y[:, -1], dtype=np.float64)

    def solve(
        self,
        signal_power_in_w: NDArrayFloat,
        signal_ase_in_w: NDArrayFloat | None = None,
    ) -> NumericalDRASpanResult:
        """Solve the numerical Raman equations for one span.

        Parameters
        ----------
        signal_power_in_w : numpy.ndarray
            Input signal powers in watts.
        signal_ase_in_w : numpy.ndarray or None, optional
            Input signal ASE; None means zero.

        Returns
        -------
        NumericalDRASpanResult
            Signal profile, output powers, newly generated ASE and pump outputs.

        Raises
        ------
        ValueError
            If input shapes do not match the signal count.
        """
        power_in = np.asarray(signal_power_in_w, dtype=np.float64)
        if power_in.shape != (self._n_signal,):
            msg = "signal_power_in_w must contain one value per signal."
            raise ValueError(msg)
        ase_in = np.zeros(self._n_signal, dtype=np.float64)
        if signal_ase_in_w is not None:
            ase_in = np.asarray(signal_ase_in_w, dtype=np.float64)
            if ase_in.shape != (self._n_signal,):
                msg = "signal_ase_in_w must contain one value per signal."
                raise ValueError(msg)

        iterations = 0
        converged = not self.has_backward_pumps
        backward_solution = None
        if self.has_backward_pumps:
            backward_profile = self._initial_backward_profile()
            peak_pump = max(
                float(np.max(self._pump_power_w, initial=0.0)), _POWER_FLOOR
            )
            for iteration in range(self.n_iterations):
                iterations = iteration + 1
                forward_solution = self._solve_forward(power_in, backward_profile)
                backward_solution = self._solve_backward(forward_solution)
                updated = np.asarray(
                    backward_solution.sol(self.length_m - self._z_profile)
                ).T
                change = float(np.max(np.abs(updated - backward_profile))) / peak_pump
                backward_profile = (
                    _BW_RELAXATION * updated + (1.0 - _BW_RELAXATION) * backward_profile
                )
                if change < _BW_TOLERANCE:
                    converged = True
                    break
            # Reintegrate forward with the final backward profile for self-consistency.
            forward_solution = self._solve_forward(power_in, backward_profile)
            backward_solution = self._solve_backward(forward_solution)
        else:
            forward_solution = self._solve_forward(
                power_in, np.zeros((self.n_profile, 0))
            )

        profile_forward = np.asarray(forward_solution.sol(self._z_profile))
        profile_signal = profile_forward[self._fwd_signal_local, :].T
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_norm = np.where(
                power_in[None, :] > 0.0,
                profile_signal
                / np.where(power_in[None, :] > 0.0, power_in[None, :], 1.0),
                1.0,
            )

        power_end = np.asarray(forward_solution.sol(self.length_m))
        power_out = power_end[self._fwd_signal_local].copy()
        pump_out = np.zeros(self._n_pump, dtype=np.float64)
        if self._fwd_pump_original.size:
            pump_out[self._fwd_pump_original] = power_end[~self._fwd_is_signal]
        if self._bw_pump_original.size:
            assert backward_solution is not None
            pump_out[self._bw_pump_original] = np.asarray(
                backward_solution.sol(self.length_m)
            )

        ase_out = self._solve_ase(power_in, ase_in, forward_solution, backward_solution)
        fiber_gain = np.divide(
            power_out,
            power_in,
            out=np.ones_like(power_out),
            where=power_in > 0.0,
        )
        raman_ase = np.maximum(ase_out - ase_in * fiber_gain, 0.0)

        return NumericalDRASpanResult(
            profile_norm=profile_norm,
            z_m=self._z_profile.copy(),
            power_out_w=power_out,
            raman_ase_w=raman_ase,
            pump_power_out_w=pump_out,
            iterations=iterations,
            converged=converged,
        )


def _tan_nodes(
    lower: float,
    upper: float,
    scale: float,
    nodes: int,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Generate quadrature nodes and weights using a tangent substitution."""
    if upper <= lower or nodes < 1:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    scale = max(float(scale), 1.0)
    u_lower = float(np.arctan(lower / scale))
    u_upper = float(np.arctan(upper / scale))
    base_nodes, base_weights = np.polynomial.legendre.leggauss(int(nodes))  # type: ignore[no-untyped-call]
    u = 0.5 * (u_upper - u_lower) * base_nodes + 0.5 * (u_upper + u_lower)
    weights = 0.5 * (u_upper - u_lower) * base_weights
    values = scale * np.tan(u)
    jacobian = scale / np.cos(u) ** 2
    return values, weights * jacobian


def _simpson_weights(z_m: NDArrayFloat) -> NDArrayFloat:
    """Return composite Simpson weights for equally spaced nodes."""
    n = z_m.size
    if n < 3 or n % 2 == 0:
        msg = "the longitudinal grid must have an odd number of nodes >= 3."
        raise ValueError(msg)
    step = float(z_m[1] - z_m[0])
    weights = np.ones(n, dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * step / 3.0


def _interpolate_profile(
    frequencies_hz: NDArrayFloat,
    profile_norm: NDArrayFloat,
    query_hz: NDArrayFloat,
) -> NDArrayFloat:
    """Interpolate the longitudinal profile along channel frequency."""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    order = np.argsort(frequencies)
    f_sorted = frequencies[order]
    rho_sorted = profile_norm[:, order]
    query = np.asarray(query_hz, dtype=np.float64)
    shape = query.shape
    flat = query.ravel()
    index = np.searchsorted(f_sorted, flat, side="right") - 1
    index = np.clip(index, 0, f_sorted.size - 2)
    span = f_sorted[index + 1] - f_sorted[index]
    weight = np.divide(
        flat - f_sorted[index],
        span,
        out=np.zeros_like(flat),
        where=span > 0.0,
    )
    values = (1.0 - weight)[:, None] * rho_sorted[:, index].T + weight[
        :, None
    ] * rho_sorted[:, index + 1].T
    return cast("NDArrayFloat", values.reshape((*shape, profile_norm.shape[0])))


def _xpm_x_nodes(
    alpha_npm: float,
    separation_hz: float,
    dispersion: float,
    half_band_hz: float,
    length_m: float,
    nodes: int,
    phase_limit: float,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Generate XPM channel-of-interest quadrature nodes with bounded phase.

    The tangent transform resolves the narrow coherence peak. When the maximum
    phase exceeds ``phase_limit``, the oscillatory tails are truncated so that the
    longitudinal Simpson grid can resolve the remaining phase. Otherwise the full
    channel is integrated.

    Parameters
    ----------
    alpha_npm : float
        Channel attenuation in 1/m.
    separation_hz : float
        Interferer frequency minus channel-of-interest frequency.
    dispersion : float
        Effective dispersion in s²/m.
    half_band_hz : float
        Channel-of-interest half-bandwidth in Hz.
    length_m : float
        Span length in meters.
    nodes : int
        Quadrature node count.
    phase_limit : float
        Maximum absolute longitudinal phase.

    Returns
    -------
    tuple of numpy.ndarray
        Frequency offsets and quadrature weights in Hz.
    """
    coefficient = 4.0 * np.pi**2 * float(separation_hz) * float(dispersion)
    max_phase = abs(coefficient) * length_m * half_band_hz
    alpha_length = max(float(alpha_npm) * length_m, 1.0e-12)
    if max_phase <= phase_limit or coefficient == 0.0:
        # When the full band fits the phase limit, retain the coherence-peak scale
        # so that tangent quadrature does not miss the narrow peak.
        scale = (
            half_band_hz
            if coefficient == 0.0
            else half_band_hz * alpha_length / max_phase
        )
        return _tan_nodes(-half_band_hz, half_band_hz, scale, nodes)
    t_values, t_weights = _tan_nodes(-phase_limit, phase_limit, alpha_length, nodes)
    denominator = coefficient * length_m
    return t_values / denominator, t_weights / abs(denominator)


def _phase_xpm(
    beta2: float,
    beta3: float,
    coi_hz: float,
    interferer_hz: float,
    x_hz: float,
    y_hz: NDArrayFloat,
    reference_hz: float,
) -> NDArrayFloat:
    """Return the XPM phase mismatch relative to the dispersion reference."""
    dispersion = beta2 + np.pi * beta3 * (
        coi_hz + interferer_hz + x_hz + y_hz - 2.0 * reference_hz
    )
    return cast(
        "NDArrayFloat",
        -4.0 * np.pi**2 * x_hz * (interferer_hz + y_hz - coi_hz) * dispersion,
    )


def _phase_spm(
    beta2: float,
    beta3: float,
    center_hz: float,
    x_hz: NDArrayFloat,
    y_hz: NDArrayFloat,
    reference_hz: float,
) -> NDArrayFloat:
    """Return the SPM phase mismatch relative to the dispersion reference."""
    dispersion = beta2 + np.pi * beta3 * (
        2.0 * center_hz + x_hz + y_hz - 2.0 * reference_hz
    )
    return -4.0 * np.pi**2 * x_hz * y_hz * dispersion


def _link_squared(
    amplitude: NDArrayFloat,
    phase: NDArrayFloat,
    z_m: NDArrayFloat,
    weights: NDArrayFloat,
) -> NDArrayFloat:
    """Return the squared magnitude of the longitudinal complex link integral."""
    integrand = (
        amplitude * weights[None, :] * np.exp(1j * phase[:, None] * z_m[None, :])
    )
    return cast("NDArrayFloat", np.abs(np.sum(integrand, axis=1)) ** 2)


def nli_power_from_ode_profile(
    profile: ODESpanProfile,
    frequencies_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n2_m2_w: float,
    *,
    n_xpm_x: int = DEFAULT_XPM_QUADRATURE_X,
    n_xpm_y: int = DEFAULT_XPM_QUADRATURE_Y,
    n_spm_x: int = DEFAULT_SPM_QUADRATURE_X,
    n_spm_y: int = DEFAULT_SPM_QUADRATURE_Y,
    xpm_phase_limit: float = DEFAULT_XPM_PHASE_LIMIT,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """Numerically integrate SPM and XPM noise from the raw ODE profile.

    No semi-analytic profile fit or closed-form NLI expression is used.

    Parameters
    ----------
    profile : ODESpanProfile
        Raw longitudinal ODE profile.
    frequencies_hz, powers_w, bandwidths_hz : numpy.ndarray
        Per-channel frequencies (Hz), launch powers (W), and bandwidths (Hz).
    alpha_npm, a_eff_m2 : numpy.ndarray
        Per-channel attenuation (1/m) and effective area (m²).
    beta2, beta3 : float
        Dispersion and dispersion slope at the reference frequency.
    reference_frequency_hz : float
        Dispersion reference frequency in Hz.
    n2_m2_w : float
        Nonlinear refractive index in m²/W.
    n_xpm_x, n_xpm_y, n_spm_x, n_spm_y : int, optional
        Frequency quadrature node counts for the XPM and SPM dimensions.
    xpm_phase_limit : float, optional
        Maximum absolute XPM longitudinal phase.

    Returns
    -------
    tuple of numpy.ndarray
        Per-channel SPM and XPM noise powers in watts.
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    powers = np.asarray(powers_w, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    alpha = np.asarray(alpha_npm, dtype=np.float64)
    areas = np.asarray(a_eff_m2, dtype=np.float64)
    n_channels = frequencies.size
    if powers.shape != (n_channels,) or bandwidths.shape != (n_channels,):
        msg = "powers_w and bandwidths_hz must contain one value per channel."
        raise ValueError(msg)

    z_m = np.asarray(profile.z_m, dtype=np.float64)
    weights = _simpson_weights(z_m)
    rho = np.asarray(profile.profile_norm, dtype=np.float64)
    gamma = (2.0 * np.pi * frequencies / C_LIGHT) * (n2_m2_w / areas)

    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)

    for index in range(n_channels):
        launch = float(powers[index])
        if launch <= 0.0 or bandwidths[index] <= 0.0:
            continue
        half_band = 0.5 * float(bandwidths[index])
        rho_coi = _interpolate_profile(
            frequencies, rho, np.array([frequencies[index]])
        )[0]
        dispersion_coi = beta2 + np.pi * beta3 * (
            2.0 * frequencies[index] - 2.0 * reference_frequency_hz
        )
        spm_scale = _coherence_scale(alpha[index], dispersion_coi, half_band)
        x_grid, x_weights = _tan_nodes(-half_band, half_band, spm_scale, n_spm_x)
        rho_x = _interpolate_profile(frequencies, rho, frequencies[index] + x_grid)

        spm_integral = 0.0
        for x_index, x_value in enumerate(x_grid):
            y_lower = max(-half_band, -half_band - x_value)
            y_upper = min(half_band, half_band - x_value)
            if y_upper <= y_lower:
                continue
            y_grid, y_weights = _tan_nodes(y_lower, y_upper, spm_scale, n_spm_y)
            rho_y = _interpolate_profile(frequencies, rho, frequencies[index] + y_grid)
            rho_idler = _interpolate_profile(
                frequencies, rho, frequencies[index] + x_value + y_grid
            )
            amplitude = np.sqrt(
                np.maximum(
                    rho_x[x_index][None, :] * rho_y * rho_idler / rho_coi[None, :], 0.0
                )
            )
            phase = _phase_spm(
                beta2,
                beta3,
                float(frequencies[index]),
                np.full_like(y_grid, x_value),
                y_grid,
                reference_frequency_hz,
            )
            spm_integral += x_weights[x_index] * float(
                np.sum(y_weights * _link_squared(amplitude, phase, z_m, weights))
            )
        spm[index] = (
            (16.0 / 27.0)
            * gamma[index] ** 2
            * spm_integral
            / bandwidths[index] ** 2
            * launch**3
        )

        xpm_total = 0.0
        for interferer in range(n_channels):
            if interferer == index:
                continue
            interferer_power = float(powers[interferer])
            if interferer_power <= 0.0 or bandwidths[interferer] <= 0.0:
                continue
            separation = float(frequencies[interferer] - frequencies[index])
            dispersion_interferer = beta2 + np.pi * beta3 * (
                frequencies[index]
                + frequencies[interferer]
                - 2.0 * reference_frequency_hz
            )
            x_grid, x_weights = _xpm_x_nodes(
                alpha[index],
                separation,
                dispersion_interferer,
                half_band,
                float(profile.length_m),
                n_xpm_x,
                xpm_phase_limit,
            )
            rho_x = _interpolate_profile(frequencies, rho, frequencies[index] + x_grid)
            half_band_interferer = 0.5 * float(bandwidths[interferer])
            pair_integral = 0.0
            for x_index, x_value in enumerate(x_grid):
                y_lower = max(-half_band_interferer, -half_band_interferer - x_value)
                y_upper = min(half_band_interferer, half_band_interferer - x_value)
                if y_upper <= y_lower:
                    continue
                # The coherence peak is along x; the smooth y integral uses interferer half-bandwidth.
                y_grid, y_weights = _tan_nodes(
                    y_lower, y_upper, half_band_interferer, n_xpm_y
                )
                rho_y = _interpolate_profile(
                    frequencies, rho, frequencies[interferer] + y_grid
                )
                rho_idler = _interpolate_profile(
                    frequencies,
                    rho,
                    frequencies[interferer] + x_value + y_grid,
                )
                amplitude = np.sqrt(
                    np.maximum(
                        rho_x[x_index][None, :] * rho_y * rho_idler / rho_coi[None, :],
                        0.0,
                    )
                )
                phase = _phase_xpm(
                    beta2,
                    beta3,
                    float(frequencies[index]),
                    float(frequencies[interferer]),
                    x_value,
                    y_grid,
                    reference_frequency_hz,
                )
                pair_integral += x_weights[x_index] * float(
                    np.sum(y_weights * _link_squared(amplitude, phase, z_m, weights))
                )
            xpm_total += (
                (interferer_power / launch) ** 2
                * pair_integral
                / bandwidths[interferer] ** 2
            )
        xpm[index] = (32.0 / 27.0) * gamma[index] ** 2 * xpm_total * launch**3

    return spm, xpm


def _coherence_scale(alpha_npm: float, dispersion: float, width_hz: float) -> float:
    """Return the characteristic XPM coherence-peak width in Hz."""
    if width_hz <= 0.0 or dispersion == 0.0:
        return max(float(width_hz), 1.0)
    scale = float(alpha_npm) / (
        4.0 * np.pi**2 * abs(float(dispersion)) * float(width_hz)
    )
    return float(np.clip(scale, 1.0, max(float(width_hz), 1.0)))


class NumericalDRAKernel(DRASpanKernelBase[ODESpanProfile]):
    """Numerical Raman span kernel using DOP853 and double NLI integration.

    Parameters
    ----------
    *args
        Positional arguments forwarded to DRASpanKernelBase.
    rtol, atol : float, optional
        DOP853 relative and absolute tolerances.
    n_profile : int, optional
        Odd number of longitudinal profile samples.
    n_xpm_x, n_xpm_y, n_spm_x, n_spm_y : int, optional
        Quadrature node counts in each frequency dimension.
    xpm_phase_limit : float, optional
        Maximum XPM longitudinal phase.
    **kwargs
        Options forwarded to DRASpanKernelBase.
    """

    profile_kind = "numerical_ode"
    emits_nli_amplitudes = False

    def __init__(
        self,
        *args: object,
        rtol: float = DEFAULT_NUMERICAL_DRA_RTOL,
        atol: float = DEFAULT_NUMERICAL_DRA_ATOL,
        n_profile: int = DEFAULT_NUMERICAL_DRA_PROFILE_POINTS,
        n_xpm_x: int = DEFAULT_XPM_QUADRATURE_X,
        n_xpm_y: int = DEFAULT_XPM_QUADRATURE_Y,
        n_spm_x: int = DEFAULT_SPM_QUADRATURE_X,
        n_spm_y: int = DEFAULT_SPM_QUADRATURE_Y,
        xpm_phase_limit: float = DEFAULT_XPM_PHASE_LIMIT,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if n_profile < 5 or n_profile % 2 == 0:
            msg = "n_profile must be an odd integer >= 5."
            raise ValueError(msg)
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._n_profile = int(n_profile)
        self._n_xpm_x = int(n_xpm_x)
        self._n_xpm_y = int(n_xpm_y)
        self._n_spm_x = int(n_spm_x)
        self._n_spm_y = int(n_spm_y)
        self._xpm_phase_limit = float(xpm_phase_limit)
        self._result_cache: dict[
            tuple[int, bytes], tuple[NumericalDRASpanResult, NDArrayFloat, NDArrayFloat]
        ] = {}
        self._nli_cache: dict[tuple[int, bytes], tuple[NDArrayFloat, NDArrayFloat]] = {}

    def _solver_for(self, span_length_m: float) -> NumericalDRASpanSolver:
        """Return a DOP853 solver cached by span length."""
        key = round(span_length_m * 1e6)
        solver = cast("NumericalDRASpanSolver | None", self._solver_cache.get(key))
        if solver is None:
            solver = NumericalDRASpanSolver(
                self.f_abs_hz,
                self.b_hz,
                self._pumps,
                span_length_m,
                self.alpha_npm,
                self.a_eff_m2,
                self._pump_alpha_npm,
                self._pump_a_eff_m2,
                self._gain_model,
                n_profile=self._n_profile,
                rtol=self._rtol,
                atol=self._atol,
                n_iterations=self._n_iterations,
                include_isrs=self._include_isrs,
                include_ase=self._raman_ase_enabled,
                signal_polarizations=self._signal_polarizations,
            )
            self._solver_cache[key] = solver
        return solver

    def _fit_profile(
        self, profile_norm: NDArrayFloat, span_length_m: float
    ) -> ODESpanProfile:
        """Wrap the raw ODE profile for NLI integration without fitting it."""
        return ODESpanProfile(
            profile_norm=np.asarray(profile_norm, dtype=np.float64),
            z_m=np.linspace(
                0.0, span_length_m, profile_norm.shape[0], dtype=np.float64
            ),
            length_m=float(span_length_m),
            frequencies_hz=self.f_abs_hz,
        )

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: ODESpanProfile,
        span_context: object | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Compute per-span SPM and XPM using numerical double integration."""
        del span_context
        key = (
            id(profile),
            np.round(
                np.asarray(power_internal)
                / max(float(np.max(power_internal, initial=0.0)), _POWER_FLOOR)
                * 1.0e8
            )
            .astype(np.int64)
            .tobytes(),
        )
        cached = self._nli_cache.get(key)
        if cached is not None:
            return cached
        result = nli_power_from_ode_profile(
            profile,
            self.f_abs_hz,
            power_internal,
            self.b_hz,
            self.alpha_npm,
            self.a_eff_m2,
            self.beta2,
            self.beta3,
            self.reference_frequency_hz,
            self.n2_m2_w,
            n_xpm_x=self._n_xpm_x,
            n_xpm_y=self._n_xpm_y,
            n_spm_x=self._n_spm_x,
            n_spm_y=self._n_spm_y,
            xpm_phase_limit=self._xpm_phase_limit,
        )
        if len(self._nli_cache) >= 8:
            self._nli_cache.clear()
        self._nli_cache[key] = result
        return result

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        *,
        span_context: object | None = None,
    ) -> SpanFiberOutput:
        """Evaluate a span and cache its ODE and NLI by launch spectrum and length.

        New Raman ASE is treated separately from input ASE, which the route layer
        transports. Span count and context are accepted for protocol compatibility;
        this reference accumulates span NLI as power.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Input powers in model channel order.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Unused protocol argument.
        span_context : object or None, optional
            Unused protocol argument.

        Returns
        -------
        SpanFiberOutput
            Output powers and SPM/XPM noise in model channel order.
        """
        del n_effective_spans, span_context
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != (self.n_nodes,):
            msg = "power_in_w must contain one value per channel."
            raise ValueError(msg)
        power_internal = np.ascontiguousarray(power[self.order])
        scale = max(float(np.max(power_internal, initial=0.0)), _POWER_FLOOR)
        key = (
            round(span_length_m * 1e6),
            np.round(power_internal / scale * 1.0e8).astype(np.int64).tobytes(),
        )
        cached = self._result_cache.get(key)
        if cached is None:
            solver = self._solver_for(span_length_m)
            result = solver.solve(power_internal)
            profile = self._profile_for(
                power_internal, span_length_m, result.profile_norm
            )
            spm, xpm = self._nli_span(power_internal, profile, None)
            if len(self._result_cache) >= 8:
                self._result_cache.clear()
            cached = (result, spm, xpm)
            self._result_cache[key] = cached
        result, spm, xpm = cached
        self._last_raman_ase_internal = (
            np.ascontiguousarray(result.raman_ase_w)
            if self._raman_ase_enabled
            else np.zeros(self.n_nodes, dtype=np.float64)
        )
        self._last_pump_power_out_w = np.ascontiguousarray(result.pump_power_out_w)
        return SpanFiberOutput(
            power_out_w=np.ascontiguousarray(result.power_out_w[self.inverse_order]),
            spm_noise_w=np.ascontiguousarray(spm[self.inverse_order]),
            xpm_noise_w=np.ascontiguousarray(xpm[self.inverse_order]),
        )

    def reset_state(self) -> None:
        """Clear per-route caches."""
        super().reset_state()
        self._result_cache.clear()
        self._nli_cache.clear()


NUMERICAL_DRA_MODEL_INFO = ModelInfo(
    name="numerical_dra",
    summary=(
        "Independent numerical DRA reference: scipy solve_ivp coupled signal/pump/ASE ODE "
        "(DOP853, tight tolerances) plus the Eq. (6)-(8) GN double integral on the raw ODE "
        "profile. Arbitrary forward/backward pumps; Raman ASE is separated from the EDFA ASE."
    ),
    family="numerical+raman",
    reference=(
        "Buglia et al., JLT 42(2):636-648, 2024, Eq. (1)-(8) for the Raman ODE and GN kernel; "
        "Semrau et al., JLT 2018 for the ISRS-GN numerical baseline"
    ),
    options=(
        (
            "raman_pumps",
            "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps.",
        ),
        ("raman_gain_model", "'gaussian' (default) or 'linear' (CZ)."),
        (
            "raman_ase",
            "Expose the newly generated Raman ASE before the EDFA. Default True.",
        ),
        (
            "raman_temperature_k",
            "Fibre temperature of the spontaneous Raman factor. Default 300 K.",
        ),
        (
            "raman_peak_gain_coefficient_m_per_w",
            "Peak local Raman gain rate. Default 6e-14.",
        ),
        (
            "include_isrs",
            "Include channel-channel ISRS in the coupled ODE. Default True.",
        ),
        (
            "signal_polarizations",
            (
                "Polarizations represented by each per-polarization ODE signal state. Default 1 "
                "is the historical drive factor; 2 drives the coupling with the dual-polarization "
                "total signal power while the state itself is not scaled. The f_receiver/f_source "
                "fix changes forward-pump results independently of this switch."
            ),
        ),
        (
            "fixed_point_iterations",
            "Maximum backward-pump fixed-point iterations. Default 24.",
        ),
        ("rtol", "DOP853 relative tolerance. Default 1e-10."),
        ("atol", "DOP853 absolute tolerance. Default 1e-15."),
        ("n_profile", "Longitudinal ODE profile nodes (odd). Default 257."),
        ("n_xpm_x", "XPM COI-direction quadrature nodes. Default 16."),
        ("n_xpm_y", "XPM interferer-direction quadrature nodes. Default 6."),
        ("n_spm_x", "SPM first-direction quadrature nodes. Default 48."),
        ("n_spm_y", "SPM second-direction quadrature nodes. Default 24."),
        (
            "xpm_phase_limit",
            "XPM coherence-peak phase truncation |phi L|. Default 300.",
        ),
    ),
)


class NumericalDRAModel(DRARamanModelBase):
    """Independent numerical reference for distributed Raman amplification.

    The ODE uses DOP853, NLI uses direct double integration and span NLI powers are
    summed without inter-span coherence.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Spectrum grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber parameters.
    symbol_rate_hz : float, optional
        Symbol rate in Hz.
    connections : sequence or None, optional
        Flexible-grid connections.
    raman_pumps : sequence of RamanPumpConfig or None, optional
        Per-span pumps; None uses fiber configuration.
    raman_gain_model : str, optional
        Raman gain spectrum model.
    raman_ase : bool, optional
        Include newly generated Raman ASE before the EDFA.
    raman_temperature_k : float, optional
        Fiber temperature in kelvin.
    raman_peak_gain_coefficient_m_per_w : float, optional
        Peak Raman gain coefficient in m/W.
    include_isrs : bool, optional
        Include signal-signal ISRS.
    fixed_point_iterations : int, optional
        Maximum backward-pump iterations.
    rtol, atol : float, optional
        DOP853 tolerances.
    n_profile : int, optional
        Odd number of longitudinal profile samples.
    n_xpm_x, n_xpm_y, n_spm_x, n_spm_y : int, optional
        Quadrature node counts in each frequency dimension.
    xpm_phase_limit : float, optional
        Maximum XPM longitudinal phase.
    signal_polarizations : int, optional
        Signal polarization multiplier for the coupling drive.
    **route_options
        Options forwarded to the shared route model.
    """

    info = NUMERICAL_DRA_MODEL_INFO
    _kernel_class = NumericalDRAKernel
    supports_span_context = False

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str = "gaussian",
        raman_ase: bool = True,
        raman_temperature_k: float = 300.0,
        raman_peak_gain_coefficient_m_per_w: float = 6e-14,
        include_isrs: bool = True,
        fixed_point_iterations: int = 24,
        rtol: float = DEFAULT_NUMERICAL_DRA_RTOL,
        atol: float = DEFAULT_NUMERICAL_DRA_ATOL,
        n_profile: int = DEFAULT_NUMERICAL_DRA_PROFILE_POINTS,
        n_xpm_x: int = DEFAULT_XPM_QUADRATURE_X,
        n_xpm_y: int = DEFAULT_XPM_QUADRATURE_Y,
        n_spm_x: int = DEFAULT_SPM_QUADRATURE_X,
        n_spm_y: int = DEFAULT_SPM_QUADRATURE_Y,
        xpm_phase_limit: float = DEFAULT_XPM_PHASE_LIMIT,
        signal_polarizations: int = 1,
        **route_options: Any,
    ) -> None:
        if n_profile < 5 or n_profile % 2 == 0:
            msg = "n_profile must be an odd integer >= 5."
            raise ValueError(msg)
        if int(signal_polarizations) < 1:
            msg = "signal_polarizations must be >= 1."
            raise ValueError(msg)
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._n_profile = int(n_profile)
        self._n_xpm_x = int(n_xpm_x)
        self._n_xpm_y = int(n_xpm_y)
        self._n_spm_x = int(n_spm_x)
        self._n_spm_y = int(n_spm_y)
        self._xpm_phase_limit = float(xpm_phase_limit)
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            raman_pumps=raman_pumps,
            raman_gain_model=raman_gain_model,
            raman_ase=raman_ase,
            raman_temperature_k=raman_temperature_k,
            raman_peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            include_isrs=include_isrs,
            fixed_point_iterations=fixed_point_iterations,
            n_steps=DEFAULT_DRA_STEPS,
            n_quadrature=DEFAULT_DRA_NLI_QUADRATURE,
            signal_polarizations=signal_polarizations,
            **route_options,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> NumericalDRAKernel:
        """Build a numerical span kernel."""
        return self._kernel_class(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            pumps=self._resolved_pumps(fiber_config),
            gain_model=self._gain_model,
            raman_ase=self._raman_ase,
            n_steps=DEFAULT_DRA_STEPS,
            n_iterations=self._fixed_point_iterations,
            include_isrs=self._include_isrs,
            n_quadrature=DEFAULT_DRA_NLI_QUADRATURE,
            signal_polarizations=self._signal_polarizations,
            rtol=self._rtol,
            atol=self._atol,
            n_profile=self._n_profile,
            n_xpm_x=self._n_xpm_x,
            n_xpm_y=self._n_xpm_y,
            n_spm_x=self._n_spm_x,
            n_spm_y=self._n_spm_y,
            xpm_phase_limit=self._xpm_phase_limit,
        )


register_model(NUMERICAL_DRA_MODEL_INFO, NumericalDRAModel)
