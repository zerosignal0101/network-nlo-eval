"""Coupled Raman ODEs for the signal, pump and ASE evolution over one span.

This module contains the shared physical equations without an NLI kernel or
model-specific dependencies, allowing hybrid and distributed Raman models to
reuse the same solver without circular imports.

References
----------
H. Buglia, R. I. Killey and P. Bayvel, "Ultra-Wideband Modeling of Optical
Fibre Nonlinearity in Hybrid-Amplified Links", J. Lightwave Technol., 2024,
Eqs. (2) and (4).
"""

# ruff: noqa: PLC0415  # Load the optional Numba backend lazily.

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from network_nlo_eval.models.pcfm import DEFAULT_SPP_STEPS
from network_nlo_eval.models.power_conventions import (
    LAUNCH_POWER_CONVENTIONS,
    resolve_launch_power_convention,
)
from network_nlo_eval.models.raman import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    RamanGainModel,
    resolve_raman_gain_model,
)
from network_nlo_eval.models.raman_iteration import (
    BackwardPumpIteration,
    validate_raman_iteration_method,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from network_nlo_eval.core.types import NDArrayFloat, NDArrayInt
    from network_nlo_eval.network.elements import RamanPumpConfig

__all__ = [
    "AUTO_NUMBA_MIN_STEPS",
    "DEFAULT_FIXED_POINT_ITERATIONS",
    "DEFAULT_PAPER_SPP_STEPS",
    "LAUNCH_POWER_CONVENTIONS",
    "RAMAN_BACKENDS",
    "RAMAN_MATRIX_CACHE_MAX_ENTRIES",
    "PaperRamanSpanResult",
    "RamanPaperSpanSolver",
    "clear_raman_matrix_cache",
    "numba_available",
    "raman_matrix_cache_info",
    "resolve_launch_power_convention",
    "resolve_raman_backend",
    "validate_raman_backend",
]

#: Accepted values of the opt-in ``backend`` switch.  ``"auto"`` is never a
#: default: the model and the solver both default to ``"python"``.
RAMAN_BACKENDS: tuple[str, ...] = ("python", "numba", "auto")

#: ``"auto"`` selects the JIT backend only from this step count upwards.  Below
#: it the fixed-point overhead dominates and the Python path is kept.
AUTO_NUMBA_MIN_STEPS: int = 24

#: Default RK4 steps; sample the midpoint and endpoint of each PCFM SPP step.
DEFAULT_PAPER_SPP_STEPS: int = DEFAULT_SPP_STEPS

#: Matrix cache capacity. Each entry contains two (N + N_p) squared float64
#: matrices; eight entries use approximately 18 MB for 361 channels and six pumps.
RAMAN_MATRIX_CACHE_MAX_ENTRIES: int = 8

#: Increment the cache-key version whenever matrix construction changes.
_MATRIX_CACHE_VERSION: bytes = b"raman-paper-span-matrices-v1"


@dataclass(frozen=True)
class _CachedMatrices:
    """Read-only coupling and source matrices shared by solver instances."""

    coupling: NDArrayFloat
    source: NDArrayFloat


#: The matrix cache is not thread-safe. Concurrent construction may duplicate
#: work or interleave eviction, but solving does not access the cache.
_MATRIX_CACHE: OrderedDict[bytes, _CachedMatrices] = OrderedDict()
_MATRIX_CACHE_STATS: dict[str, int] = {"hits": 0, "misses": 0}


def raman_matrix_cache_info() -> dict[str, int]:
    """Return process-wide Raman matrix cache statistics.

    Returns
    -------
    dict of str to int
        Cache size, capacity, hits and misses. Counts refer to solver construction;
        solving does not query the cache.
    """
    return {
        "size": len(_MATRIX_CACHE),
        "max_entries": RAMAN_MATRIX_CACHE_MAX_ENTRIES,
        "hits": int(_MATRIX_CACHE_STATS["hits"]),
        "misses": int(_MATRIX_CACHE_STATS["misses"]),
    }


def clear_raman_matrix_cache() -> None:
    """Clear the process-wide matrix cache and reset hit and miss counters."""
    _MATRIX_CACHE.clear()
    _MATRIX_CACHE_STATS["hits"] = 0
    _MATRIX_CACHE_STATS["misses"] = 0


def validate_raman_backend(backend: object) -> str:
    """Validate and return the requested Raman integration backend.

    Parameters
    ----------
    backend : str
        Requested backend: ``python``, ``numba`` or ``auto``.

    Returns
    -------
    str
        The validated backend name.

    Raises
    ------
    ValueError
        If the backend is unknown.
    """
    if not isinstance(backend, str) or backend not in RAMAN_BACKENDS:
        msg = f"raman_backend must be one of {RAMAN_BACKENDS}, got {backend!r}."
        raise ValueError(msg)
    return backend


def numba_available() -> bool:
    """Return whether the optional Numba Raman kernels are available."""
    try:
        from network_nlo_eval.models import raman_numba
    except ImportError:
        return False
    return bool(raman_numba.NUMBA_AVAILABLE)


def resolve_raman_backend(
    backend: str,
    *,
    n_steps: int,
    n_lumped_loss: int = 0,
    available: bool | None = None,
) -> str:
    """Resolve a requested backend to Python or Numba.

    Auto selects Numba only when available, sufficiently many steps are requested,
    and there are no lumped losses. An explicit Numba request never falls back.

    Parameters
    ----------
    backend : str
        Validated backend name.
    n_steps : int
        RK4 step count.
    n_lumped_loss : int
        Number of lumped-loss elements.
    available : bool or None, optional
        Override availability detection; None probes the current process.

    Returns
    -------
    str
        Resolved backend, ``python`` or ``numba``.

    Raises
    ------
    RuntimeError
        If Numba was explicitly requested but is unavailable.
    ValueError
        If Numba was explicitly requested with lumped losses.
    """
    validate_raman_backend(backend)
    if backend == "python":
        return "python"
    usable = numba_available() if available is None else bool(available)
    if backend == "numba":
        if not usable:
            msg = (
                "raman_backend='numba' requires numba, which is not importable in this "
                "environment. Use raman_backend='python' (the default), or 'auto' to fall "
                "back to Python when numba is unavailable."
            )
            raise RuntimeError(msg)
        if n_lumped_loss:
            msg = (
                "raman_backend='numba' does not support lumped_loss: the JIT kernels "
                "integrate the uniform no-lumped-loss grid only. Remove the lumped loss "
                "or use raman_backend='python' (the default)."
            )
            raise ValueError(msg)
        return "numba"
    # "auto": numba only for the long uniform-grid solves it actually accelerates.
    if usable and n_lumped_loss == 0 and int(n_steps) >= AUTO_NUMBA_MIN_STEPS:
        return "numba"
    return "python"


def _numba_kernels() -> Any:
    """Return the lazily loaded Numba kernels or raise RuntimeError."""
    try:
        from network_nlo_eval.models import raman_numba
    except ImportError as exc:  # pragma: no cover - numba is a hard dependency.
        msg = (
            "raman_backend='numba' requires the optional numba kernels, which could not "
            "be imported."
        )
        raise RuntimeError(msg) from exc
    if not raman_numba.NUMBA_AVAILABLE:  # pragma: no cover - defensive.
        msg = "raman_backend='numba' requires numba, which is not installed."
        raise RuntimeError(msg)
    return raman_numba


def _digest_array(digest: Any, tag: bytes, array: NDArrayFloat) -> None:
    """Append an array tag, dtype, shape and raw bytes to the digest."""
    data = np.ascontiguousarray(array)
    digest.update(tag)
    digest.update(str(data.dtype).encode())
    digest.update(str(data.shape).encode())
    digest.update(data.tobytes())


def _digest_scalar(digest: Any, tag: bytes, value: float | bool | str) -> None:
    """Append a scalar to the digest, encoding floats as float64 bytes."""
    digest.update(tag)
    if isinstance(value, str):
        digest.update(value.encode())
    elif isinstance(value, (bool, np.bool_)):
        digest.update(b"\x01" if bool(value) else b"\x00")
    elif isinstance(value, int):
        digest.update(int(value).to_bytes(8, "little", signed=True))
    else:
        digest.update(np.float64(value).tobytes())


#: Maximum fixed-point iterations for backward pumps.
DEFAULT_FIXED_POINT_ITERATIONS: int = 24

#: Convergence threshold relative to peak pump power.
_FIXED_POINT_TOLERANCE: float = 1.0e-7


@dataclass(frozen=True)
class PaperRamanSpanResult:
    """Solution of the signal, pump and ASE equations for one span.

    Attributes
    ----------
    profile_norm : numpy.ndarray
        Normalized signal profile, sampled at every RK4 midpoint and endpoint.
    power_out_w : numpy.ndarray
        Output signal powers in watts.
    raman_ase_w : numpy.ndarray
        Newly generated Raman ASE in the solver's launch-power convention.
    pump_power_out_w : numpy.ndarray
        Pump output powers at the appropriate end for each direction.
    iterations : int
        Number of fixed-point iterations performed.
    converged : bool
        Whether the fixed-point iteration converged.
    sample_positions_u : numpy.ndarray or None
        Normalized positions corresponding to the profile rows. None means the
        uniform grid ``k / (2 * n_steps)``; segmented grids can be nonuniform.
    """

    profile_norm: NDArrayFloat
    power_out_w: NDArrayFloat
    raman_ase_w: NDArrayFloat
    pump_power_out_w: NDArrayFloat
    iterations: int
    converged: bool
    sample_positions_u: NDArrayFloat | None = None


def _resolve_lumped_loss(
    lumped_loss: Sequence[tuple[float, float]] | None,
    length_m: float,
) -> tuple[list[tuple[float, float]], NDArrayFloat, NDArrayFloat]:
    """Validate, sort and merge lumped losses inside a span.

    Repeated positions are merged by adding their dB losses. Positions must lie
    strictly inside the span and losses must be nonnegative and finite.

    Parameters
    ----------
    lumped_loss : sequence of tuple or None
        Pairs of position in meters and loss in dB.
    length_m : float
        Span length in meters.

    Returns
    -------
    tuple
        Merged pairs, position array, and linear power transmission factors.

    Raises
    ------
    ValueError
        If an entry, position or loss is invalid.
    """
    if lumped_loss is None or len(lumped_loss) == 0:
        empty = np.zeros(0, dtype=np.float64)
        return [], empty, empty
    entries: list[tuple[float, float]] = []
    for item in lumped_loss:
        try:
            position_m, loss_db = item
        except (TypeError, ValueError) as exc:
            msg = f"lumped_loss entries must be (position_m, loss_db) pairs, got {item!r}."
            raise ValueError(msg) from exc
        position = float(position_m)
        loss = float(loss_db)
        if not np.isfinite(position) or not np.isfinite(loss):
            msg = f"lumped loss must be finite, got ({position_m!r}, {loss_db!r})."
            raise ValueError(msg)
        if position <= 0.0:
            msg = f"lumped loss position must be positive, got {position}."
            raise ValueError(msg)
        if position >= length_m:
            msg = f"lumped loss at {position} m lies outside the span of {length_m} m."
            raise ValueError(msg)
        if loss < 0.0:
            msg = f"lumped loss must not be a gain, got {loss} dB."
            raise ValueError(msg)
        entries.append((position, loss))
    entries.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = []
    for position, loss in entries:
        if merged and position == merged[-1][0]:
            merged[-1] = (position, merged[-1][1] + loss)
        else:
            merged.append((position, loss))
    positions = np.asarray([item[0] for item in merged], dtype=np.float64)
    factors = np.asarray(
        [10.0 ** (-item[1] / 10.0) for item in merged], dtype=np.float64
    )
    return merged, positions, factors


def _interp_rows(
    grid: NDArrayFloat, values: NDArrayFloat, query: float
) -> NDArrayFloat:
    """Linearly interpolate rows on an increasing, possibly nonuniform grid.

    Parameters
    ----------
    grid : numpy.ndarray
        Increasing coordinates of shape ``(S,)``.
    values : numpy.ndarray
        Values of shape ``(S, M)``.
    query : float
        Coordinate to interpolate.

    Returns
    -------
    numpy.ndarray
        Interpolated row, clamped to the endpoints outside the grid.
    """
    if values.shape[0] == 1:
        return cast("NDArrayFloat", values[0].copy())
    index = int(np.clip(np.searchsorted(grid, query) - 1, 0, values.shape[0] - 2))
    span = float(grid[index + 1] - grid[index])
    weight = 0.0 if span <= 0.0 else float((query - grid[index]) / span)
    return cast(
        "NDArrayFloat", (1.0 - weight) * values[index] + weight * values[index + 1]
    )


def _digest_gain_model(digest: Any, gain_model: RamanGainModel) -> None:
    """Append only the gain-model fields used in calculations to the digest."""
    _digest_scalar(digest, b"gain.kind", str(gain_model.gain_model))
    table = gain_model.table
    if table is None:
        digest.update(b"gain.table=none")
    else:
        _digest_array(digest, b"gain.table.offsets", table[0])
        _digest_array(digest, b"gain.table.gains", table[1])
    _digest_scalar(digest, b"gain.peak", gain_model.peak_gain_coefficient_m_per_w)
    _digest_scalar(digest, b"gain.df_peak", gain_model.delta_f_peak_hz)
    _digest_scalar(digest, b"gain.df_sigma", gain_model.delta_f_sigma_hz)
    _digest_scalar(digest, b"gain.df_cutoff", gain_model.delta_f_cutoff_hz)
    _digest_scalar(digest, b"gain.temperature", gain_model.temperature_k)


def _cached_raman_matrices(
    key: bytes,
    builder: Callable[[], tuple[NDArrayFloat, NDArrayFloat]],
) -> _CachedMatrices:
    """Return cached read-only matrices, building and storing them on a miss.

    The least recently used entry is evicted when the cache reaches capacity.
    """
    entry = _MATRIX_CACHE.get(key)
    if entry is not None:
        _MATRIX_CACHE.move_to_end(key)
        _MATRIX_CACHE_STATS["hits"] += 1
        return entry
    _MATRIX_CACHE_STATS["misses"] += 1
    coupling, source = builder()
    coupling.setflags(write=False)
    source.setflags(write=False)
    entry = _CachedMatrices(coupling=coupling, source=source)
    _MATRIX_CACHE[key] = entry
    while len(_MATRIX_CACHE) > RAMAN_MATRIX_CACHE_MAX_ENTRIES:
        _MATRIX_CACHE.popitem(last=False)
    return entry


class RamanPaperSpanSolver:
    """Solve the coupled signal, pump and ASE equations over one fiber span.

    For receiver j and source k, the coupling coefficient is positive when the
    source frequency is higher and negative otherwise. The negative coefficient
    uses ``f_j / f_k``: for frequency-independent effective area this preserves
    photon number, ``C[j,k]/f_j + C[k,j]/f_k = 0``. The printed equation in the
    cited paper uses the reciprocal factor; this implementation uses the
    photon-conserving receiver/source ratio. The same gain matrix covers
    signal-signal, pump-signal and pump-pump interactions.

    Parameters
    ----------
    signal_frequencies_hz, signal_bandwidths_hz : numpy.ndarray
        Signal center frequencies and bandwidths in Hz, in output order.
    pumps : sequence of RamanPumpConfig
        Forward and backward pumps.
    length_m : float
        Fiber span length in meters.
    signal_alpha_npm, pump_alpha_npm : numpy.ndarray
        Signal and pump power attenuation coefficients in inverse meters.
    signal_a_eff_m2, pump_a_eff_m2 : numpy.ndarray
        Signal and pump effective areas in square meters.
    gain_model : RamanGainModel or str
        Gain model, built-in model name, or path to a gain table.
    n_steps : int, optional
        RK4 steps; profiles contain a midpoint and endpoint per step.
    n_iterations : int, optional
        Maximum backward-pump fixed-point iterations.
    include_isrs, include_ase : bool, optional
        Enable signal-signal ISRS and the ASE equations, respectively.
    pump_depletion : bool, optional
        Include pump depletion; disabling it is an undepleted-pump comparison.
    signal_polarizations : int, optional
        Multiplier applied to signal power when driving Raman coupling. Per-
        polarization coordinates require two; the default one preserves the
        historical total-power behavior.
    pump_pump_raman : bool, optional
        Enable coupling between pumps without changing other interactions.
    launch_power_convention : {"total", "per_pol"}, optional
        Coordinate convention for input and output signal power and ASE. The
        coupling drive uses physical total power in per-polarization coordinates.
    lumped_loss : sequence of tuple or None, optional
        Pairs of interior position in meters and nonnegative loss in dB. The grid
        is segmented at each position, and signal, ASE and pumps are attenuated
        in their respective propagation directions.
    backend : {"python", "numba", "auto"}, optional
        Integration backend. Python preserves the reference numerical path;
        Numba supports uniform spans without lumped losses. Auto chooses Numba
        only when available and the grid has enough steps.
    iteration_method : {"relaxed", "picard", "anderson"}, optional
        Backward-pump boundary update. The default uses half damping.
        Experimental Picard and safeguarded Anderson updates use the same
        physical equations and unmixed residual tolerance.

    Raises
    ------
    ValueError
        If inputs, power conventions, loss elements or backend are incompatible.
    RuntimeError
        If the explicitly requested Numba backend is unavailable.
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
        gain_model: RamanGainModel | str,
        *,
        n_steps: int = DEFAULT_PAPER_SPP_STEPS,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        include_ase: bool = True,
        pump_depletion: bool = True,
        signal_polarizations: int = 1,
        pump_pump_raman: object = True,
        launch_power_convention: str = "total",
        lumped_loss: Sequence[tuple[float, float]] | None = None,
        backend: str = "python",
        iteration_method: str = "relaxed",
    ) -> None:
        """Validate inputs and build the position-independent Raman matrices."""
        if length_m <= 0:
            msg = "length_m must be positive."
            raise ValueError(msg)
        if n_steps < 1:
            msg = "n_steps must be >= 1."
            raise ValueError(msg)
        if n_iterations < 1:
            msg = "n_iterations must be >= 1."
            raise ValueError(msg)
        if int(signal_polarizations) < 1:
            msg = "signal_polarizations must be >= 1."
            raise ValueError(msg)
        if not isinstance(pump_pump_raman, (bool, np.bool_)):
            msg = f"pump_pump_raman must be a bool, got {pump_pump_raman!r}."
            raise ValueError(msg)  # noqa: TRY004 - Preserve the public exception type.
        # Reject invalid backends before constructing matrices.
        requested_backend = validate_raman_backend(backend)
        self.iteration_method = validate_raman_iteration_method(iteration_method)
        resolved_convention = resolve_launch_power_convention(
            launch_power_convention, signal_polarizations
        )

        self.signal_frequencies_hz = np.asarray(signal_frequencies_hz, dtype=np.float64)
        self.signal_bandwidths_hz = np.asarray(signal_bandwidths_hz, dtype=np.float64)
        self.signal_alpha_npm = np.asarray(signal_alpha_npm, dtype=np.float64)
        self.signal_a_eff_m2 = np.asarray(signal_a_eff_m2, dtype=np.float64)
        self.pumps = list(pumps)
        self.length_m = float(length_m)
        self.gain_model = resolve_raman_gain_model(gain_model)
        self.n_steps = int(n_steps)
        self.n_iterations = int(n_iterations)
        self.include_isrs = bool(include_isrs)
        self.include_ase = bool(include_ase)
        self.pump_depletion = bool(pump_depletion)
        self.signal_polarizations = int(signal_polarizations)
        self.pump_pump_raman = bool(pump_pump_raman)
        self.launch_power_convention = resolved_convention

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
        pump_a_eff = np.asarray(pump_a_eff_m2, dtype=np.float64)
        if pump_alpha.shape != (len(self.pumps),):
            msg = "pump_alpha_npm must contain one value per pump."
            raise ValueError(msg)
        if pump_a_eff.shape != (len(self.pumps),):
            msg = "pump_a_eff_m2 must contain one value per pump."
            raise ValueError(msg)

        self._pump_f_hz = pump_f
        self._pump_alpha_npm = pump_alpha
        self._pump_a_eff_m2 = pump_a_eff
        self._pump_power_w = np.asarray(
            [float(pump.power_w) for pump in self.pumps], dtype=np.float64
        )
        self._pump_forward = np.asarray(
            [str(pump.direction) == "forward" for pump in self.pumps], dtype=bool
        )

        # Combine signal and pump waves, sorted by descending frequency.
        self._n_signal = n_signal
        self._n_pump = len(self.pumps)
        f_concat = np.concatenate([self.signal_frequencies_hz, pump_f])
        alpha_concat = np.concatenate([self.signal_alpha_npm, pump_alpha])
        a_eff_concat = np.concatenate([self.signal_a_eff_m2, pump_a_eff])
        b_concat = np.concatenate(
            [self.signal_bandwidths_hz, np.zeros(self._n_pump, dtype=np.float64)]
        )
        is_pump_concat = np.concatenate(
            [np.zeros(n_signal, dtype=bool), np.ones(self._n_pump, dtype=bool)]
        )
        direction_concat = np.concatenate(
            [
                np.ones(n_signal, dtype=np.float64),
                np.where(self._pump_forward, 1.0, -1.0),
            ]
        )
        p0_concat = np.concatenate(
            [np.zeros(n_signal, dtype=np.float64), self._pump_power_w]
        )

        order = np.argsort(-f_concat, kind="stable")
        self._order = order
        self._f_hz = np.ascontiguousarray(f_concat[order])
        self._alpha_npm = np.ascontiguousarray(alpha_concat[order])
        self._a_eff_m2 = np.ascontiguousarray(a_eff_concat[order])
        self._bandwidth_hz = np.ascontiguousarray(b_concat[order])
        self._is_pump = np.ascontiguousarray(is_pump_concat[order])
        self._direction = np.ascontiguousarray(direction_concat[order])
        self._p0_w = np.ascontiguousarray(p0_concat[order])
        self._n_waves = self._f_hz.size

        # Coupling drive equals state power times polarization count. In per-pol
        # coordinates this yields physical total signal power while keeping state and
        # output per polarization. Pump state is single-polarization power.
        self._drive_scale = np.where(
            self._is_pump,
            1.0,
            float(self.signal_polarizations),
        ).astype(np.float64)
        # RHS temporary buffers are overwritten on every call and can be reused.
        self._rhs_total = np.empty(self._n_waves, dtype=np.float64)
        self._rhs_drive = np.empty(self._n_waves, dtype=np.float64)
        # When all drive factors are exactly one, skip the identity multiplication.
        self._drive_uniform = bool(np.all(self._drive_scale == 1.0))

        # Total-power ASE uses a two-polarization source; per-pol ASE uses one.
        # The source is driven by physical total power in both conventions, so the
        # per-pol output and input ASE remain in the same coordinate.
        self._ase_source_scale = 1.0 if resolved_convention == "per_pol" else 2.0

        self._signal_slots = np.flatnonzero(~self._is_pump)
        self._fwd_slots = np.flatnonzero(self._direction > 0)
        self._bw_slots = np.flatnonzero(self._direction < 0)
        self._fwd_is_signal = ~self._is_pump[self._fwd_slots]
        self._fwd_original = order[self._fwd_slots]
        self._fwd_signal_original = np.where(
            self._fwd_is_signal, self._fwd_original, -1
        )
        self._fwd_pump_original = np.where(
            ~self._fwd_is_signal, self._fwd_original - n_signal, -1
        )
        self._bw_original = order[self._bw_slots] - n_signal

        self._alpha_fwd = np.ascontiguousarray(self._alpha_npm[self._fwd_slots])
        self._alpha_bw = np.ascontiguousarray(self._alpha_npm[self._bw_slots])
        self._b_fwd = np.ascontiguousarray(self._bandwidth_hz[self._fwd_slots])
        self._f_fwd = np.ascontiguousarray(self._f_hz[self._fwd_slots])
        # Precompute the scalar source coefficient in the original operation order.
        self._ase_source_coeff = (
            self._ase_source_scale * H_PLANCK * self._b_fwd * self._f_fwd
        )

        matrices = _cached_raman_matrices(
            self._matrix_cache_key(), self._build_matrices
        )
        self._coupling = matrices.coupling
        self._source = matrices.source
        self._source_fwd = np.ascontiguousarray(self._source[self._fwd_slots, :])
        self._z_m = np.linspace(0.0, self.length_m, self.n_steps + 1, dtype=np.float64)
        # Validate lumped losses once; no losses retains the original uniform path.
        self._lumped_loss, self._lumped_loss_positions_m, self._lumped_loss_factors = (
            _resolve_lumped_loss(lumped_loss, self.length_m)
        )
        self.n_lumped_loss = len(self._lumped_loss)
        self._sample_positions_u: NDArrayFloat | None
        if self.n_lumped_loss:
            edges = np.concatenate(
                ([0.0], self._lumped_loss_positions_m, [self.length_m])
            )
            lengths = np.diff(edges)
            # Allocate steps by segment length, with at least one step per segment.
            steps = np.maximum(
                1,
                np.round(self.n_steps * lengths / self.length_m).astype(np.int64),
            )
            self._z_segments = [
                np.linspace(
                    float(edges[i]),
                    float(edges[i + 1]),
                    int(steps[i]) + 1,
                    dtype=np.float64,
                )
                for i in range(steps.size)
            ]
            self._n_total_steps = int(steps.sum())
            self._sample_positions_u = self._build_sample_positions()
        else:
            self._z_segments = [self._z_m]
            self._n_total_steps = self.n_steps
            # None preserves uniform-grid sampling without numerical resampling.
            self._sample_positions_u = None
        # Backward pumps enter at z=L and attenuate toward z=0.
        self._bw_initial_total = np.zeros(
            (self.n_steps + 1, self._bw_slots.size), dtype=np.float64
        )
        for column, pump_index in enumerate(self._bw_original):
            if pump_index < 0:
                continue
            alpha = self._pump_alpha_npm[pump_index]
            self._bw_initial_total[:, column] = self._pump_power_w[pump_index] * np.exp(
                -alpha * (self.length_m - self._z_m)
            )
        # Each backward segment includes all losses already crossed from the right;
        # the last segment has not crossed any loss.
        if self.n_lumped_loss:
            self._bw_initial_segments = [
                self._passive_backward_segment(z_nodes, index)
                for index, z_nodes in enumerate(self._z_segments)
            ]
        else:
            self._bw_initial_segments = [self._bw_initial_total]

        # Precompute backward-stage interpolation indices and weights for the uniform
        # grid using the same expressions as _interp_rows.
        self._bw_interp_index, self._bw_interp_weight = (
            self._build_backward_interpolation()
        )

        # The default Python backend avoids importing Numba; explicit Numba requests
        # reject lumped losses, which have already been resolved.
        self.backend = requested_backend
        self._resolved_backend = resolve_raman_backend(
            requested_backend,
            n_steps=self.n_steps,
            n_lumped_loss=self.n_lumped_loss,
        )
        # The backward kernel needs only pump rows; create that copy lazily.
        self._coupling_bw: NDArrayFloat | None = None

    def _matrix_cache_key(self) -> bytes:
        """Return a stable digest of span-length-independent matrix inputs."""
        digest = hashlib.blake2b(digest_size=16)
        digest.update(_MATRIX_CACHE_VERSION)
        _digest_array(digest, b"f_hz", self._f_hz)
        _digest_array(digest, b"b_hz", self._bandwidth_hz)
        _digest_array(digest, b"alpha_npm", self._alpha_npm)
        _digest_array(digest, b"a_eff_m2", self._a_eff_m2)
        _digest_array(digest, b"is_pump", self._is_pump.astype(np.uint8))
        _digest_scalar(digest, b"n_signal", int(self._n_signal))
        _digest_scalar(digest, b"n_pump", int(self._n_pump))
        _digest_scalar(digest, b"include_isrs", self.include_isrs)
        _digest_scalar(digest, b"pump_pump_raman", self.pump_pump_raman)
        _digest_scalar(digest, b"pump_depletion", self.pump_depletion)
        _digest_scalar(digest, b"signal_polarizations", int(self.signal_polarizations))
        _digest_scalar(
            digest, b"launch_power_convention", str(self.launch_power_convention)
        )
        _digest_gain_model(digest, self.gain_model)
        return digest.digest()

    def _build_matrices(self) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Build the position-independent coupling and source matrices."""
        coupling = self._build_coupling_matrix()
        self._coupling = coupling
        return coupling, self._build_source_matrix()

    def _build_backward_interpolation(self) -> tuple[NDArrayInt, NDArrayFloat]:
        """Precompute interpolation indices and weights for backward RK4 stages."""
        n_steps = self.n_steps
        index = np.zeros((n_steps, 3), dtype=np.int64)
        weight = np.zeros((n_steps, 3), dtype=np.float64)
        if self._bw_slots.size == 0 or n_steps == 0:
            return index, weight
        grid = self._z_m
        last = grid.shape[0] - 2
        z0 = self.length_m - grid[:n_steps]
        z1 = self.length_m - grid[1 : n_steps + 1]
        queries = np.column_stack((z0, 0.5 * (z0 + z1), z1))
        positions = np.searchsorted(grid, queries.ravel()) - 1
        positions = np.clip(positions, 0, last).reshape(queries.shape)
        span = grid[positions + 1] - grid[positions]
        shift = queries - grid[positions]
        weights = np.zeros(queries.shape, dtype=np.float64)
        np.divide(shift, span, out=weights, where=span > 0.0)
        return positions.astype(np.int64), weights

    def _build_sample_positions(self) -> NDArrayFloat:
        """Return normalized, potentially nonuniform profile sample positions."""
        blocks = []
        for z_nodes in self._z_segments:
            n_step = z_nodes.size - 1
            block = np.empty(2 * n_step, dtype=np.float64)
            block[0::2] = 0.5 * (z_nodes[:-1] + z_nodes[1:])
            block[1::2] = z_nodes[1:]
            blocks.append(block / self.length_m)
        return np.ascontiguousarray(np.concatenate(blocks))

    def _passive_backward_segment(
        self, z_nodes: NDArrayFloat, segment_index: int
    ) -> NDArrayFloat:
        """Return the undepleted backward-pump profile including preceding losses."""
        factors = self._lumped_loss_factors[segment_index:]
        factor = float(np.prod(factors)) if factors.size else 1.0
        values = np.zeros((z_nodes.size, self._bw_slots.size), dtype=np.float64)
        for column, pump_index in enumerate(self._bw_original):
            if pump_index < 0:
                continue
            alpha = self._pump_alpha_npm[pump_index]
            values[:, column] = (
                self._pump_power_w[pump_index]
                * np.exp(-alpha * (self.length_m - z_nodes))
                * factor
            )
        return values

    def _build_coupling_matrix(self) -> NDArrayFloat:
        """Build the Raman coupling matrix in inverse watt-meters."""
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        shape = self.gain_model.gain_shape(delta)
        coefficient = (
            self.gain_model.peak_gain_coefficient_m_per_w / self._a_eff_m2[:, None]
        )
        gain = coefficient * shape
        # Higher-frequency sources supply power; lower-frequency sources receive it.
        # The receiver/source frequency factor conserves photons.
        source_higher = self._f_hz[None, :] > self._f_hz[:, None]
        coupling = np.where(
            source_higher, gain, -(self._f_hz[:, None] / self._f_hz[None, :]) * gain
        )
        np.fill_diagonal(coupling, 0.0)
        if not self.include_isrs:
            signal_slots = self._signal_slots
            coupling[np.ix_(signal_slots, signal_slots)] = 0.0
        if not self.pump_pump_raman:
            pump_slots = np.flatnonzero(self._is_pump)
            coupling[np.ix_(pump_slots, pump_slots)] = 0.0
        if not self.pump_depletion:
            pump_slots = np.flatnonzero(self._is_pump)
            coupling[pump_slots, :] = 0.0
        return coupling

    def _build_source_matrix(self) -> NDArrayFloat:
        """Build the spontaneous Raman scattering source matrix."""
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        safe_delta = np.where(delta > 0.0, delta, 1.0)
        x = H_PLANCK * safe_delta / (K_BOLTZMANN * self.gain_model.temperature_k)
        kappa = np.ones_like(x)
        mask = x > 1.0e-12
        kappa[mask] = 1.0 / (-np.expm1(-x[mask]))
        return cast("NDArrayFloat", np.maximum(self._coupling, 0.0) * kappa)

    def _scaled_drive(self, total: NDArrayFloat) -> NDArrayFloat:
        """Scale total powers into the reusable drive buffer."""
        np.multiply(total, self._drive_scale, out=self._rhs_drive)
        return self._rhs_drive

    def _rhs_forward(
        self,
        power: NDArrayFloat,
        ase: NDArrayFloat,
        backward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Evaluate the forward-wave signal and pump derivatives."""
        # Reuse temporary buffers without changing the elementwise operation order.
        total = self._rhs_total
        fwd_slots = self._fwd_slots
        total[fwd_slots] = power + ase
        total[self._bw_slots] = backward_total
        # The coupling drive uses total source power; receiver power is applied once.
        # An all-one drive multiplier is an exact identity and can be skipped.
        drive = total if self._drive_uniform else self._scaled_drive(total)
        gain_fwd = (self._coupling @ drive)[fwd_slots]
        alpha_fwd = self._alpha_fwd
        d_power = power * gain_fwd - alpha_fwd * power
        if not self.include_ase:
            return d_power, np.zeros_like(ase)
        # Per-pol ASE uses one receiver polarization but physical total drive power;
        # otherwise signal-signal spontaneous emission would be underestimated.
        ase_drive = total if self._ase_source_scale == 2.0 else drive
        source = self._ase_source_coeff * (self._source_fwd @ ase_drive)
        d_ase = gain_fwd * ase + source - alpha_fwd * ase
        return d_power, d_ase

    def _rhs_backward(
        self,
        power: NDArrayFloat,
        ase: NDArrayFloat,
        forward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Evaluate backward-pump derivatives using ``s = L - z``."""
        total = self._rhs_total
        np.copyto(total, forward_total)
        total[self._bw_slots] = power + ase
        drive = total if self._drive_uniform else self._scaled_drive(total)
        gain_bw = (self._coupling @ drive)[self._bw_slots]
        alpha_bw = self._alpha_bw
        d_power = power * gain_bw - alpha_bw * power
        if not self.include_ase:
            return d_power, np.zeros_like(ase)
        # Backward pumps have zero bandwidth and no spontaneous ASE source.
        # Forward integration accounts for signal ASE.
        d_ase = gain_bw * ase - alpha_bw * ase
        return d_power, d_ase

    def _integrate_forward(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        backward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Integrate forward waves with fixed-step RK4, sampling midpoints and endpoints."""
        n_fwd = self._fwd_slots.size
        power = power_in.copy()
        ase = ase_in.copy()
        power_grid = np.empty((self.n_steps + 1, n_fwd), dtype=np.float64)
        ase_grid = np.empty_like(power_grid)
        samples = np.empty((2 * self.n_steps, n_fwd), dtype=np.float64)
        power_grid[0] = power
        ase_grid[0] = ase
        for step in range(self.n_steps):
            h = self._z_m[step + 1] - self._z_m[step]
            bw0 = backward_total[step]
            bw1 = backward_total[step + 1]
            bw_mid = 0.5 * (bw0 + bw1)

            dp1, da1 = self._rhs_forward(power, ase, bw0)
            dp2, da2 = self._rhs_forward(
                power + 0.5 * h * dp1, ase + 0.5 * h * da1, bw_mid
            )
            dp3, da3 = self._rhs_forward(
                power + 0.5 * h * dp2, ase + 0.5 * h * da2, bw_mid
            )
            dp4, da4 = self._rhs_forward(power + h * dp3, ase + h * da3, bw1)

            samples[2 * step] = power + 0.5 * h * dp1 + 0.25 * h * (dp2 - dp1)
            power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
            samples[2 * step + 1] = power
            power_grid[step + 1] = power
            ase_grid[step + 1] = ase
        return power, ase, power_grid, ase_grid, samples

    def _integrate_forward_segment(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        backward_segment: NDArrayFloat,
        z_nodes: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Integrate one loss-free segment with aligned backward-pump grid values."""
        n_step = z_nodes.size - 1
        n_fwd = self._fwd_slots.size
        power = power_in.copy()
        ase = ase_in.copy()
        power_grid = np.empty((n_step + 1, n_fwd), dtype=np.float64)
        ase_grid = np.empty_like(power_grid)
        samples = np.empty((2 * n_step, n_fwd), dtype=np.float64)
        power_grid[0] = power
        ase_grid[0] = ase
        for step in range(n_step):
            h = z_nodes[step + 1] - z_nodes[step]
            bw0 = backward_segment[step]
            bw1 = backward_segment[step + 1]
            bw_mid = 0.5 * (bw0 + bw1)

            dp1, da1 = self._rhs_forward(power, ase, bw0)
            dp2, da2 = self._rhs_forward(
                power + 0.5 * h * dp1, ase + 0.5 * h * da1, bw_mid
            )
            dp3, da3 = self._rhs_forward(
                power + 0.5 * h * dp2, ase + 0.5 * h * da2, bw_mid
            )
            dp4, da4 = self._rhs_forward(power + h * dp3, ase + h * da3, bw1)

            samples[2 * step] = power + 0.5 * h * dp1 + 0.25 * h * (dp2 - dp1)
            power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
            samples[2 * step + 1] = power
            power_grid[step + 1] = power
            ase_grid[step + 1] = ase
        return power, ase, power_grid, ase_grid, samples

    def _integrate_forward_segments(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        backward_segments: list[NDArrayFloat],
    ) -> tuple[
        NDArrayFloat,
        NDArrayFloat,
        list[NDArrayFloat],
        list[NDArrayFloat],
        NDArrayFloat,
    ]:
        """Integrate forward by segments, applying loss to signal and ASE at boundaries."""
        power = power_in
        ase = ase_in
        power_segments: list[NDArrayFloat] = []
        ase_segments: list[NDArrayFloat] = []
        sample_blocks: list[NDArrayFloat] = []
        for index, z_nodes in enumerate(self._z_segments):
            if index > 0:
                factor = float(self._lumped_loss_factors[index - 1])
                power = power * factor
                ase = ase * factor
            power, ase, power_grid, ase_grid, samples = self._integrate_forward_segment(
                power,
                ase,
                backward_segments[index],
                z_nodes,
            )
            power_segments.append(power_grid)
            ase_segments.append(ase_grid)
            sample_blocks.append(samples)
        return power, ase, power_segments, ase_segments, np.vstack(sample_blocks)

    def _forward_total_segments(
        self,
        power_segments: list[NDArrayFloat],
        ase_segments: list[NDArrayFloat],
        backward_segments: list[NDArrayFloat],
    ) -> list[NDArrayFloat]:
        """Combine forward power, ASE and backward pumps into segment state arrays."""
        totals: list[NDArrayFloat] = []
        for index, z_nodes in enumerate(self._z_segments):
            total = np.empty((z_nodes.size, self._n_waves), dtype=np.float64)
            total[:, self._fwd_slots] = power_segments[index] + ase_segments[index]
            total[:, self._bw_slots] = backward_segments[index]
            totals.append(total)
        return totals

    def _integrate_backward(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        forward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Integrate backward pumps with RK4 in increasing ``s = L - z``."""
        n_bw = self._bw_slots.size
        power = power_in.copy()
        ase = ase_in.copy()
        power_grid = np.empty((self.n_steps + 1, n_bw), dtype=np.float64)
        ase_grid = np.empty_like(power_grid)
        power_grid[0] = power
        ase_grid[0] = ase
        for step in range(self.n_steps):
            h = self._z_m[step + 1] - self._z_m[step]
            index0, index_mid, index1 = self._bw_interp_index[step]
            weight0, weight_mid, weight1 = self._bw_interp_weight[step]
            # Use the same interpolation expression with precomputed indices and weights.
            total0 = (1.0 - weight0) * forward_total[index0] + weight0 * forward_total[
                index0 + 1
            ]
            total_mid = (1.0 - weight_mid) * forward_total[
                index_mid
            ] + weight_mid * forward_total[index_mid + 1]
            total1 = (1.0 - weight1) * forward_total[index1] + weight1 * forward_total[
                index1 + 1
            ]

            dp1, da1 = self._rhs_backward(power, ase, total0)
            dp2, da2 = self._rhs_backward(
                power + 0.5 * h * dp1, ase + 0.5 * h * da1, total_mid
            )
            dp3, da3 = self._rhs_backward(
                power + 0.5 * h * dp2, ase + 0.5 * h * da2, total_mid
            )
            dp4, da4 = self._rhs_backward(power + h * dp3, ase + h * da3, total1)

            power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
            power_grid[step + 1] = power
            ase_grid[step + 1] = ase
        return power, ase, power_grid, ase_grid

    # -- opt-in numba backend ------------------------------------------------
    # These wrappers retain the Python integration result layouts while replacing
    # the loops with lazily loaded compiled kernels.

    def _integrate_forward_numba(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        backward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Run compiled forward RK4 with the same layout as the Python implementation."""
        kernels = _numba_kernels()
        return cast(
            "tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]",
            kernels.rk4_forward(
                self._coupling,
                self._fwd_slots,
                self._bw_slots,
                self._alpha_fwd,
                self._source_fwd,
                self._ase_source_coeff,
                self._drive_scale,
                self._z_m,
                np.ascontiguousarray(power_in),
                np.ascontiguousarray(ase_in),
                np.ascontiguousarray(backward_total),
                self._drive_uniform,
                float(self._ase_source_scale),
                self.include_ase,
            ),
        )

    def _integrate_backward_numba(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        forward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Run compiled backward RK4 with the same layout as the Python implementation."""
        kernels = _numba_kernels()
        if self._coupling_bw is None:
            self._coupling_bw = np.ascontiguousarray(self._coupling[self._bw_slots, :])
        return cast(
            "tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]",
            kernels.rk4_backward(
                self._coupling_bw,
                self._bw_slots,
                self._alpha_bw,
                self._drive_scale,
                self._z_m,
                self._bw_interp_index,
                self._bw_interp_weight,
                np.ascontiguousarray(power_in),
                np.ascontiguousarray(ase_in),
                np.ascontiguousarray(forward_total),
                self._drive_uniform,
                self.include_ase,
            ),
        )

    def _integrate_backward_segment(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        forward_segment: NDArrayFloat,
        z_nodes: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """Integrate one segment backward, returning grids in increasing z order."""
        n_step = z_nodes.size - 1
        n_bw = self._bw_slots.size
        power = power_in.copy()
        ase = ase_in.copy()
        power_grid = np.empty((n_step + 1, n_bw), dtype=np.float64)
        ase_grid = np.empty_like(power_grid)
        power_grid[n_step] = power
        ase_grid[n_step] = ase
        for step in range(n_step, 0, -1):
            z0 = z_nodes[step]
            z1 = z_nodes[step - 1]
            h = z0 - z1
            z_mid = 0.5 * (z0 + z1)
            total0 = _interp_rows(z_nodes, forward_segment, z0)
            total_mid = _interp_rows(z_nodes, forward_segment, z_mid)
            total1 = _interp_rows(z_nodes, forward_segment, z1)

            dp1, da1 = self._rhs_backward(power, ase, total0)
            dp2, da2 = self._rhs_backward(
                power + 0.5 * h * dp1, ase + 0.5 * h * da1, total_mid
            )
            dp3, da3 = self._rhs_backward(
                power + 0.5 * h * dp2, ase + 0.5 * h * da2, total_mid
            )
            dp4, da4 = self._rhs_backward(power + h * dp3, ase + h * da3, total1)

            power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
            power_grid[step - 1] = power
            ase_grid[step - 1] = ase
        return power, ase, power_grid, ase_grid

    def _integrate_backward_segments(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        forward_segments: list[NDArrayFloat],
    ) -> tuple[
        NDArrayFloat,
        NDArrayFloat,
        list[NDArrayFloat],
        list[NDArrayFloat],
    ]:
        """Integrate pumps backward by segments and apply loss at each boundary."""
        power = power_in
        ase = ase_in
        n_segments = len(self._z_segments)
        power_segments: list[NDArrayFloat] = [
            np.empty((0, 0), dtype=np.float64) for _ in range(n_segments)
        ]
        ase_segments: list[NDArrayFloat] = [
            np.empty((0, 0), dtype=np.float64) for _ in range(n_segments)
        ]
        for index in range(n_segments - 1, -1, -1):
            power, ase, power_grid, ase_grid = self._integrate_backward_segment(
                power,
                ase,
                forward_segments[index],
                self._z_segments[index],
            )
            power_segments[index] = power_grid
            ase_segments[index] = ase_grid
            if index > 0:
                factor = float(self._lumped_loss_factors[index - 1])
                power = power * factor
                ase = ase * factor
        return power, ase, power_segments, ase_segments

    def _initial_forward_state(
        self,
        signal_power_in_w: NDArrayFloat,
        signal_ase_in_w: NDArrayFloat | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """Construct the initial forward state from signal and pump inputs."""
        power = np.zeros(self._fwd_slots.size, dtype=np.float64)
        ase = np.zeros_like(power)
        signal_mask = self._fwd_is_signal
        signal_original = self._fwd_signal_original[signal_mask]
        power[signal_mask] = signal_power_in_w[signal_original]
        if signal_ase_in_w is not None:
            ase[signal_mask] = signal_ase_in_w[signal_original]
        pump_mask = ~signal_mask
        pump_original = self._fwd_pump_original[pump_mask]
        if pump_original.size:
            power[pump_mask] = self._pump_power_w[pump_original]
        return power, ase

    def _solve_uniform(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
    ) -> PaperRamanSpanResult:
        """Use the original uniform-grid solve path when no lumped losses are present."""
        power_fw, ase_fw = self._initial_forward_state(power_in, ase_in)
        power_bw = self._p0_w[self._bw_slots].copy()
        ase_bw = np.zeros_like(power_bw)
        backward_total = self._bw_initial_total.copy()
        iterations = 0
        converged = self._bw_slots.size == 0 or not self.pump_depletion

        if self._bw_slots.size and self.pump_depletion:
            updater = BackwardPumpIteration(self.iteration_method)
            peak_pump = max(float(np.max(self._pump_power_w, initial=0.0)), 1.0e-30)
            for iteration in range(self.n_iterations):
                iterations = iteration + 1
                power_end, ase_end, power_grid, ase_grid, samples = (
                    self._integrate_forward(power_fw, ase_fw, backward_total)
                )
                forward_total = np.empty(
                    (self.n_steps + 1, self._n_waves), dtype=np.float64
                )
                forward_total[:, self._fwd_slots] = power_grid + ase_grid
                forward_total[:, self._bw_slots] = backward_total

                _, _, power_bw_grid_s, ase_bw_grid_s = self._integrate_backward(
                    power_bw, ase_bw, forward_total
                )
                new_backward = (power_bw_grid_s + ase_bw_grid_s)[::-1]
                change = (
                    float(np.max(np.abs(new_backward - backward_total))) / peak_pump
                )
                if (
                    change < _FIXED_POINT_TOLERANCE
                    and self.iteration_method != "relaxed"
                ):
                    converged = True
                    break
                backward_total = updater.update(backward_total, new_backward)
                if change < _FIXED_POINT_TOLERANCE:
                    converged = True
                    break

        power_end, ase_end, _power_grid, _ase_grid, samples = self._integrate_forward(
            power_fw, ase_fw, backward_total
        )

        return self._assemble_uniform_result(
            power_end=power_end,
            ase_end=ase_end,
            samples=samples,
            power_in=power_in,
            ase_in=ase_in,
            backward_total=backward_total,
            iterations=iterations,
            converged=converged,
        )

    def _solve_uniform_numba(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
    ) -> PaperRamanSpanResult:
        """Solve on the uniform grid using compiled RK4 and the same iteration logic."""
        power_fw, ase_fw = self._initial_forward_state(power_in, ase_in)
        power_bw = self._p0_w[self._bw_slots].copy()
        ase_bw = np.zeros_like(power_bw)
        backward_total = self._bw_initial_total.copy()
        iterations = 0
        converged = self._bw_slots.size == 0 or not self.pump_depletion

        if self._bw_slots.size and self.pump_depletion:
            updater = BackwardPumpIteration(self.iteration_method)
            peak_pump = max(float(np.max(self._pump_power_w, initial=0.0)), 1.0e-30)
            for iteration in range(self.n_iterations):
                iterations = iteration + 1
                power_end, ase_end, power_grid, ase_grid, samples = (
                    self._integrate_forward_numba(power_fw, ase_fw, backward_total)
                )
                forward_total = np.empty(
                    (self.n_steps + 1, self._n_waves), dtype=np.float64
                )
                forward_total[:, self._fwd_slots] = power_grid + ase_grid
                forward_total[:, self._bw_slots] = backward_total

                _, _, power_bw_grid_s, ase_bw_grid_s = self._integrate_backward_numba(
                    power_bw, ase_bw, forward_total
                )
                new_backward = (power_bw_grid_s + ase_bw_grid_s)[::-1]
                change = (
                    float(np.max(np.abs(new_backward - backward_total))) / peak_pump
                )
                if (
                    change < _FIXED_POINT_TOLERANCE
                    and self.iteration_method != "relaxed"
                ):
                    converged = True
                    break
                backward_total = updater.update(backward_total, new_backward)
                if change < _FIXED_POINT_TOLERANCE:
                    converged = True
                    break

        power_end, ase_end, _power_grid, _ase_grid, samples = (
            self._integrate_forward_numba(power_fw, ase_fw, backward_total)
        )

        return self._assemble_uniform_result(
            power_end=power_end,
            ase_end=ase_end,
            samples=samples,
            power_in=power_in,
            ase_in=ase_in,
            backward_total=backward_total,
            iterations=iterations,
            converged=converged,
        )

    def _assemble_uniform_result(
        self,
        *,
        power_end: NDArrayFloat,
        ase_end: NDArrayFloat,
        samples: NDArrayFloat,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        backward_total: NDArrayFloat,
        iterations: int,
        converged: bool,
    ) -> PaperRamanSpanResult:
        """Assemble a span result from the terminal uniform-grid states."""
        signal_mask = self._fwd_is_signal
        signal_columns = np.flatnonzero(signal_mask)
        signal_original = self._fwd_signal_original[signal_columns]

        power_out = np.zeros(self._n_signal, dtype=np.float64)
        power_out[signal_original] = power_end[signal_columns]
        ase_out = np.zeros(self._n_signal, dtype=np.float64)
        ase_out[signal_original] = ase_end[signal_columns]

        profile = np.zeros((2 * self.n_steps, self._n_signal), dtype=np.float64)
        profile[:, signal_original] = samples[:, signal_columns]
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_norm = np.where(
                power_in[None, :] > 0.0,
                profile / np.where(power_in[None, :] > 0.0, power_in[None, :], 1.0),
                1.0,
            )

        fiber_gain = np.divide(
            power_out,
            power_in,
            out=np.ones_like(power_out),
            where=power_in > 0.0,
        )
        raman_ase = np.maximum(ase_out - ase_in * fiber_gain, 0.0)

        pump_out = np.zeros(self._n_pump, dtype=np.float64)
        pump_columns = np.flatnonzero(~signal_mask)
        forward_pump_original = self._fwd_pump_original[pump_columns]
        if forward_pump_original.size:
            pump_out[forward_pump_original] = power_end[pump_columns]
        if self._bw_slots.size:
            pump_out[self._bw_original] = backward_total[0]

        return PaperRamanSpanResult(
            profile_norm=profile_norm,
            power_out_w=power_out,
            raman_ase_w=raman_ase,
            pump_power_out_w=pump_out,
            iterations=iterations,
            converged=converged,
        )

    def solve(
        self,
        signal_power_in_w: NDArrayFloat,
        signal_ase_in_w: NDArrayFloat | None = None,
    ) -> PaperRamanSpanResult:
        """Solve the coupled signal, pump and ASE equations for one span.

        Parameters
        ----------
        signal_power_in_w : numpy.ndarray
            Input signal powers in watts, in the declared launch-power convention.
        signal_ase_in_w : numpy.ndarray or None, optional
            Input ASE in the same convention; None means zero.

        Returns
        -------
        PaperRamanSpanResult
            Signal profile, output powers, generated ASE and pump outputs.

        Raises
        ------
        ValueError
            If input shapes do not match the configured signals.
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

        # Use the original uniform solve path without lumped losses; Numba implements
        # the same control flow with compiled RK4 kernels.
        if self.n_lumped_loss == 0:
            if self._resolved_backend == "numba":
                return self._solve_uniform_numba(power_in, ase_in)
            return self._solve_uniform(power_in, ase_in)

        power_fw, ase_fw = self._initial_forward_state(power_in, ase_in)
        power_bw = self._p0_w[self._bw_slots].copy()
        ase_bw = np.zeros_like(power_bw)
        backward_segments = [segment.copy() for segment in self._bw_initial_segments]
        iterations = 0
        # Undepleted backward pumps follow the passive profile without fixed-point
        # iteration; that profile already includes each crossed lumped loss.
        converged = self._bw_slots.size == 0 or not self.pump_depletion

        if self._bw_slots.size and self.pump_depletion:
            updater = BackwardPumpIteration(self.iteration_method)
            peak_pump = max(float(np.max(self._pump_power_w, initial=0.0)), 1.0e-30)
            for iteration in range(self.n_iterations):
                iterations = iteration + 1
                _, _, power_segments, ase_segments, _ = (
                    self._integrate_forward_segments(
                        power_fw, ase_fw, backward_segments
                    )
                )
                forward_segments = self._forward_total_segments(
                    power_segments, ase_segments, backward_segments
                )

                _, _, power_bw_segments, ase_bw_segments = (
                    self._integrate_backward_segments(
                        power_bw, ase_bw, forward_segments
                    )
                )
                new_backward = [
                    new_power + new_ase
                    for new_power, new_ase in zip(
                        power_bw_segments, ase_bw_segments, strict=True
                    )
                ]
                change = (
                    max(
                        float(np.max(np.abs(new - old)))
                        for new, old in zip(
                            new_backward, backward_segments, strict=True
                        )
                    )
                    / peak_pump
                )
                if (
                    change < _FIXED_POINT_TOLERANCE
                    and self.iteration_method != "relaxed"
                ):
                    converged = True
                    break
                if self.iteration_method == "relaxed":
                    backward_segments = [
                        updater.update(old, new)
                        for new, old in zip(
                            new_backward, backward_segments, strict=True
                        )
                    ]
                else:
                    # Keep both sides of every loss discontinuity in the state.
                    # Mixing the flattened blocks preserves each boundary and
                    # avoids interpolation across the lumped-loss jump.
                    shapes = [segment.shape for segment in backward_segments]
                    sizes = [segment.size for segment in backward_segments]
                    updated = updater.update(
                        np.concatenate([s.ravel() for s in backward_segments]),
                        np.concatenate([s.ravel() for s in new_backward]),
                    )
                    blocks = np.split(updated, np.cumsum(sizes)[:-1])
                    backward_segments = [
                        block.reshape(shape)
                        for block, shape in zip(blocks, shapes, strict=True)
                    ]
                if change < _FIXED_POINT_TOLERANCE:
                    converged = True
                    break

        # Integrate forward once more using the final self-consistent pump profile.
        power_end, ase_end, _power_segments, _ase_segments, samples = (
            self._integrate_forward_segments(power_fw, ase_fw, backward_segments)
        )

        signal_mask = self._fwd_is_signal
        signal_columns = np.flatnonzero(signal_mask)
        signal_original = self._fwd_signal_original[signal_columns]

        power_out = np.zeros(self._n_signal, dtype=np.float64)
        power_out[signal_original] = power_end[signal_columns]
        ase_out = np.zeros(self._n_signal, dtype=np.float64)
        ase_out[signal_original] = ase_end[signal_columns]

        profile = np.zeros((2 * self._n_total_steps, self._n_signal), dtype=np.float64)
        profile[:, signal_original] = samples[:, signal_columns]
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_norm = np.where(
                power_in[None, :] > 0.0,
                profile / np.where(power_in[None, :] > 0.0, power_in[None, :], 1.0),
                1.0,
            )

        fiber_gain = np.divide(
            power_out,
            power_in,
            out=np.ones_like(power_out),
            where=power_in > 0.0,
        )
        raman_ase = np.maximum(ase_out - ase_in * fiber_gain, 0.0)

        pump_out = np.zeros(self._n_pump, dtype=np.float64)
        pump_columns = np.flatnonzero(~signal_mask)
        forward_pump_original = self._fwd_pump_original[pump_columns]
        if forward_pump_original.size:
            pump_out[forward_pump_original] = power_end[pump_columns]
        if self._bw_slots.size:
            # Backward pump ASE is zero, so the first segment starts with the pump
            # power at z=0.
            pump_out[self._bw_original] = backward_segments[0][0]

        return PaperRamanSpanResult(
            profile_norm=profile_norm,
            power_out_w=power_out,
            raman_ase_w=raman_ase,
            pump_power_out_w=pump_out,
            iterations=iterations,
            converged=converged,
            sample_positions_u=self._sample_positions_u,
        )
