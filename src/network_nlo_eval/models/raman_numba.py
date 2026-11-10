"""Numba kernels of the paper Raman coupled ODE (opt-in ``raman_backend="numba"``).

The module is imported **lazily** by
:class:`~network_nlo_eval.models.raman_ode.RamanPaperSpanSolver`: the default
``backend="python"`` never imports it, so the historical path is untouched and
no JIT compilation is triggered.  Only two kernels live here, the forward and
the backward fixed-step RK4 loop of one span; the fixed-point iteration of the
backward pumps stays in Python and merely calls these functions once per
iteration.

Design notes
------------
* All arguments are contiguous ``float64`` arrays or scalars: no Python object
  enters the JIT boundary, so one compilation serves every span of a link.
* The matrix-vector products use ``np.dot``.  Numba lowers it to the same BLAS
  ``dgemv`` as NumPy, so the result is bit-for-bit the Python one (verified in
  ``tests/test_h3_raman_numba.py``); the backward kernel is handed only its
  ``n_bw`` rows, which leaves the six-row pump block of a 361-channel solve at
  a few hundred nanoseconds instead of a full 367x367 product.
* ``cache=True`` writes the compiled artifacts into the system temporary
  directory (``NUMBA_CACHE_DIR`` when set, else
  ``<tmp>/network-nlo-eval-numba-cache``), never into the repository tree.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat

try:  # pragma: no cover - the branch without numba is covered by monkeypatch.
    import numba.core.config as _numba_config
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _numba_config = None
    njit = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False

__all__ = [
    "NUMBA_AVAILABLE",
    "cpu_count_hint",
    "rk4_backward",
    "rk4_forward",
]

#: Sub-directory of the system temporary directory used when the caller did not
#: configure ``NUMBA_CACHE_DIR``.
_CACHE_DIR_NAME = "network-nlo-eval-numba-cache"


def _redirect_cache() -> None:
    """Point Numba's on-disk cache at the system temporary directory.

    Numba's default locator writes ``.nbc``/``.nbi`` files into the
    ``__pycache__`` directory next to the source file, i.e. into the checkout.
    The locator is chosen when the decorator runs, so assigning
    ``config.CACHE_DIR`` here (before the kernels are defined) is enough.  A
    user-provided ``NUMBA_CACHE_DIR`` always wins.
    """
    if not NUMBA_AVAILABLE or _numba_config is None:
        return
    if _numba_config.CACHE_DIR:
        return
    cache_dir = os.environ.get("NUMBA_CACHE_DIR") or os.path.join(
        tempfile.gettempdir(), _CACHE_DIR_NAME
    )
    _numba_config.CACHE_DIR = cache_dir
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError:  # pragma: no cover - a read-only tmp still compiles in memory.
        pass


_redirect_cache()


def _jit(function):
    """Decorate with ``njit(cache=True, nogil=True)``, or leave plain Python."""
    if not NUMBA_AVAILABLE:
        return function
    return njit(cache=True, nogil=True)(function)


def cpu_count_hint() -> int:
    """Return the CPU count used by the benchmark script (kept here for the report)."""
    return os.cpu_count() or 1


@_jit
def _rhs_forward(
    coupling: NDArrayFloat,
    fwd_slots: np.ndarray,
    bw_slots: np.ndarray,
    alpha_fwd: NDArrayFloat,
    source_fwd: NDArrayFloat,
    ase_source_coeff: NDArrayFloat,
    drive_scale: NDArrayFloat,
    drive_uniform: bool,
    ase_source_scale: float,
    include_ase: bool,
    power: NDArrayFloat,
    ase: NDArrayFloat,
    backward_total: NDArrayFloat,
    total: NDArrayFloat,
    drive: NDArrayFloat,
    d_power: NDArrayFloat,
    d_ase: NDArrayFloat,
) -> None:
    """Forward right-hand side, writing ``d_power``/``d_ase`` in place.

    Expression order is the one of ``RamanPaperSpanSolver._rhs_forward``:
    ``total`` is assembled first, then the coupling drive and the two
    derivatives.
    """
    total[fwd_slots] = power + ase
    total[bw_slots] = backward_total
    if drive_uniform:
        drive = total
    else:
        drive[:] = total * drive_scale
    gain_rate = np.dot(coupling, drive)
    gain_fwd = gain_rate[fwd_slots]
    d_power[:] = power * gain_fwd - alpha_fwd * power
    if not include_ase:
        d_ase[:] = 0.0
        return
    if ase_source_scale == 2.0:
        ase_drive = total
    else:
        ase_drive = drive
    source = ase_source_coeff * np.dot(source_fwd, ase_drive)
    d_ase[:] = gain_fwd * ase + source - alpha_fwd * ase


@_jit
def _interp_row_into(
    forward_total: NDArrayFloat,
    index: int,
    weight: float,
    total: NDArrayFloat,
) -> None:
    """``total = (1 - w) * row[index] + w * row[index + 1]`` (same as Python)."""
    row0 = forward_total[index]
    row1 = forward_total[index + 1]
    one_minus = 1.0 - weight
    for column in range(total.size):
        total[column] = one_minus * row0[column] + weight * row1[column]


@_jit
def _rhs_backward(
    coupling_bw: NDArrayFloat,
    bw_slots: np.ndarray,
    alpha_bw: NDArrayFloat,
    drive_scale: NDArrayFloat,
    drive_uniform: bool,
    include_ase: bool,
    power: NDArrayFloat,
    ase: NDArrayFloat,
    total: NDArrayFloat,
    drive: NDArrayFloat,
    d_power: NDArrayFloat,
    d_ase: NDArrayFloat,
) -> None:
    """Backward right-hand side on an already interpolated ``total`` row."""
    total[bw_slots] = power + ase
    if drive_uniform:
        drive = total
    else:
        drive[:] = total * drive_scale
    gain_rate = np.dot(coupling_bw, drive)
    gain_bw = gain_rate
    d_power[:] = power * gain_bw - alpha_bw * power
    if not include_ase:
        d_ase[:] = 0.0
    else:
        d_ase[:] = gain_bw * ase - alpha_bw * ase


@_jit
def rk4_forward(
    coupling: NDArrayFloat,
    fwd_slots: np.ndarray,
    bw_slots: np.ndarray,
    alpha_fwd: NDArrayFloat,
    source_fwd: NDArrayFloat,
    ase_source_coeff: NDArrayFloat,
    drive_scale: NDArrayFloat,
    z_m: NDArrayFloat,
    power_in: NDArrayFloat,
    ase_in: NDArrayFloat,
    backward_total: NDArrayFloat,
    drive_uniform: bool,
    ase_source_scale: float,
    include_ase: bool,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Fixed-step RK4 integration of the forward waves.

    Args:
        coupling: ``(M, M)`` Raman coupling matrix (read-only).
        fwd_slots: ``(n_fwd,)`` indices of the forward waves in the state.
        bw_slots: ``(n_bw,)`` indices of the backward pumps in the state.
        alpha_fwd: ``(n_fwd,)`` attenuation coefficients (1/m).
        source_fwd: ``(n_fwd, M)`` spontaneous-Raman source rows.
        ase_source_coeff: ``(n_fwd,)`` pre-multiplied ``2 h nu B``.
        drive_scale: ``(M,)`` polarization drive factor per wave.
        z_m: ``(n_steps + 1,)`` integration nodes (m).
        power_in: ``(n_fwd,)`` span-input signal/FW-pump power (W).
        ase_in: ``(n_fwd,)`` span-input ASE (W).
        backward_total: ``(n_steps + 1, n_bw)`` backward pump ``P + A`` (W).
        drive_uniform: ``True`` when every ``drive_scale`` is exactly ``1.0``.
        ase_source_scale: ``2.0`` (total) or ``1.0`` (per-polarization).
        include_ase: Whether the ASE state is integrated.

    Returns
    -------
        ``(power_end, ase_end, power_grid, ase_grid, samples)``; identical
        layout to :meth:`RamanPaperSpanSolver._integrate_forward`.
    """
    n_steps = z_m.size - 1
    n_fwd = power_in.size
    n_waves = drive_scale.size

    power = power_in.copy()
    ase = ase_in.copy()
    power_grid = np.empty((n_steps + 1, n_fwd), dtype=np.float64)
    ase_grid = np.empty((n_steps + 1, n_fwd), dtype=np.float64)
    samples = np.empty((2 * n_steps, n_fwd), dtype=np.float64)
    power_grid[0] = power
    ase_grid[0] = ase

    total = np.empty(n_waves, dtype=np.float64)
    drive = np.empty(n_waves, dtype=np.float64)
    dp1 = np.empty(n_fwd, dtype=np.float64)
    da1 = np.empty(n_fwd, dtype=np.float64)
    dp2 = np.empty(n_fwd, dtype=np.float64)
    da2 = np.empty(n_fwd, dtype=np.float64)
    dp3 = np.empty(n_fwd, dtype=np.float64)
    da3 = np.empty(n_fwd, dtype=np.float64)
    dp4 = np.empty(n_fwd, dtype=np.float64)
    da4 = np.empty(n_fwd, dtype=np.float64)

    for step in range(n_steps):
        h = z_m[step + 1] - z_m[step]
        bw0 = backward_total[step]
        bw1 = backward_total[step + 1]
        bw_mid = 0.5 * (bw0 + bw1)

        _rhs_forward(
            coupling, fwd_slots, bw_slots, alpha_fwd, source_fwd,
            ase_source_coeff, drive_scale, drive_uniform, ase_source_scale,
            include_ase, power, ase, bw0, total, drive, dp1, da1,
        )
        _rhs_forward(
            coupling, fwd_slots, bw_slots, alpha_fwd, source_fwd,
            ase_source_coeff, drive_scale, drive_uniform, ase_source_scale,
            include_ase, power + 0.5 * h * dp1, ase + 0.5 * h * da1, bw_mid,
            total, drive, dp2, da2,
        )
        _rhs_forward(
            coupling, fwd_slots, bw_slots, alpha_fwd, source_fwd,
            ase_source_coeff, drive_scale, drive_uniform, ase_source_scale,
            include_ase, power + 0.5 * h * dp2, ase + 0.5 * h * da2, bw_mid,
            total, drive, dp3, da3,
        )
        _rhs_forward(
            coupling, fwd_slots, bw_slots, alpha_fwd, source_fwd,
            ase_source_coeff, drive_scale, drive_uniform, ase_source_scale,
            include_ase, power + h * dp3, ase + h * da3, bw1,
            total, drive, dp4, da4,
        )

        samples[2 * step] = power + 0.5 * h * dp1 + 0.25 * h * (dp2 - dp1)
        power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
        ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
        samples[2 * step + 1] = power
        power_grid[step + 1] = power
        ase_grid[step + 1] = ase

    return power, ase, power_grid, ase_grid, samples


@_jit
def rk4_backward(
    coupling_bw: NDArrayFloat,
    bw_slots: np.ndarray,
    alpha_bw: NDArrayFloat,
    drive_scale: NDArrayFloat,
    z_m: NDArrayFloat,
    interp_index: np.ndarray,
    interp_weight: NDArrayFloat,
    power_in: NDArrayFloat,
    ase_in: NDArrayFloat,
    forward_total: NDArrayFloat,
    drive_uniform: bool,
    include_ase: bool,
) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """Fixed-step RK4 integration of the backward pumps in ``s = L - z``.

    Args:
        coupling_bw: ``(n_bw, M)`` rows of the coupling matrix of the pumps.
        bw_slots: ``(n_bw,)`` indices of the backward pumps in the state.
        alpha_bw: ``(n_bw,)`` pump attenuation coefficients (1/m).
        drive_scale: ``(M,)`` polarization drive factor per wave.
        z_m: ``(n_steps + 1,)`` integration nodes (m).
        interp_index: ``(n_steps, 3)`` pre-computed interpolation indices.
        interp_weight: ``(n_steps, 3)`` pre-computed interpolation weights.
        power_in: ``(n_bw,)`` backward pump power at ``z = L`` (W).
        ase_in: ``(n_bw,)`` backward ASE at ``z = L`` (W).
        forward_total: ``(n_steps + 1, M)`` forward ``P + A`` profile.
        drive_uniform: ``True`` when every ``drive_scale`` is exactly ``1.0``.
        include_ase: Whether the ASE state is integrated.

    Returns
    -------
        ``(power_end, ase_end, power_grid_s, ase_grid_s)``; same layout as
        :meth:`RamanPaperSpanSolver._integrate_backward`.
    """
    n_steps = z_m.size - 1
    n_bw = power_in.size
    n_waves = drive_scale.size

    power = power_in.copy()
    ase = ase_in.copy()
    power_grid = np.empty((n_steps + 1, n_bw), dtype=np.float64)
    ase_grid = np.empty((n_steps + 1, n_bw), dtype=np.float64)
    power_grid[0] = power
    ase_grid[0] = ase

    total = np.empty(n_waves, dtype=np.float64)
    drive = np.empty(n_waves, dtype=np.float64)
    dp1 = np.empty(n_bw, dtype=np.float64)
    da1 = np.empty(n_bw, dtype=np.float64)
    dp2 = np.empty(n_bw, dtype=np.float64)
    da2 = np.empty(n_bw, dtype=np.float64)
    dp3 = np.empty(n_bw, dtype=np.float64)
    da3 = np.empty(n_bw, dtype=np.float64)
    dp4 = np.empty(n_bw, dtype=np.float64)
    da4 = np.empty(n_bw, dtype=np.float64)

    for step in range(n_steps):
        h = z_m[step + 1] - z_m[step]
        index0 = interp_index[step, 0]
        index_mid = interp_index[step, 1]
        index1 = interp_index[step, 2]
        weight0 = interp_weight[step, 0]
        weight_mid = interp_weight[step, 1]
        weight1 = interp_weight[step, 2]

        _interp_row_into(forward_total, index0, weight0, total)
        _rhs_backward(
            coupling_bw, bw_slots, alpha_bw, drive_scale, drive_uniform,
            include_ase, power, ase, total, drive, dp1, da1,
        )

        _interp_row_into(forward_total, index_mid, weight_mid, total)
        _rhs_backward(
            coupling_bw, bw_slots, alpha_bw, drive_scale, drive_uniform,
            include_ase, power + 0.5 * h * dp1, ase + 0.5 * h * da1, total,
            drive, dp2, da2,
        )
        _rhs_backward(
            coupling_bw, bw_slots, alpha_bw, drive_scale, drive_uniform,
            include_ase, power + 0.5 * h * dp2, ase + 0.5 * h * da2, total,
            drive, dp3, da3,
        )

        _interp_row_into(forward_total, index1, weight1, total)
        _rhs_backward(
            coupling_bw, bw_slots, alpha_bw, drive_scale, drive_uniform,
            include_ase, power + h * dp3, ase + h * da3, total, drive, dp4, da4,
        )

        power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
        ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
        power_grid[step + 1] = power
        ase_grid[step + 1] = ase

    return power, ase, power_grid, ase_grid
