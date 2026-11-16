"""Experimental updates for the backward-pump boundary fixed point.

Anderson type II uses residual differences to solve a small least-squares
problem; see Walker and Ni (2011), doi:10.1137/10078356X. The positivity,
step-length and residual-growth safeguards below are project-specific.
The stopping test belongs to the Raman solver and always uses the actual,
unmixed forward/backward residual.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayFloat

RAMAN_ITERATION_METHODS = ("relaxed", "picard", "anderson")


def validate_raman_iteration_method(method: str) -> str:
    """Validate the backward-pump iteration method before any solve."""
    if method not in RAMAN_ITERATION_METHODS:
        msg = (
            f"raman_iteration_method must be one of {RAMAN_ITERATION_METHODS}, "
            f"got {method!r}."
        )
        raise ValueError(msg)
    return method


class BackwardPumpIteration:
    """Keep at most four map/residual pairs for one boundary-value solve.

    ``relaxed`` reproduces the original half-damped arithmetic. ``picard``
    takes the unmixed map directly. ``anderson`` minimizes the residual over
    three difference columns with an SVD rank cutoff, without normal equations.
    History is local to one solve and never reused for another launch state.
    """

    def __init__(self, method: str) -> None:
        self.method = validate_raman_iteration_method(method)
        self._maps: list[NDArrayFloat] = []
        self._residuals: list[NDArrayFloat] = []
        self._previous_norm: float | None = None
        self._fallback: NDArrayFloat | None = None

    def update(self, current: NDArrayFloat, mapped: NDArrayFloat) -> NDArrayFloat:
        """Return a proposed iterate; its residual is checked on the next sweep."""
        if self.method == "picard":
            return mapped
        relaxed = 0.5 * mapped + (1.0 - 0.5) * current
        if self.method == "relaxed":
            return relaxed

        residual = (mapped - current).ravel()
        norm = float(np.max(np.abs(residual)))
        # Reject a harmful extrapolation using a true map evaluation, and
        # resume at the half-damped proposal from the preceding iterate.
        if self._fallback is not None and (
            not np.isfinite(norm)
            or (self._previous_norm is not None and norm > 2.0 * self._previous_norm)
        ):
            fallback = self._fallback
            self._maps.clear()
            self._residuals.clear()
            self._previous_norm = None
            self._fallback = None
            return fallback

        self._previous_norm = norm
        self._fallback = relaxed
        self._maps.append(mapped.ravel().copy())
        self._residuals.append(residual.copy())
        if len(self._maps) > 4:
            self._maps.pop(0)
            self._residuals.pop(0)
        if len(self._maps) < 2 or norm == 0.0 or not np.isfinite(norm):
            return relaxed

        delta_r = np.column_stack(
            [
                new - old
                for old, new in zip(
                    self._residuals[:-1], self._residuals[1:], strict=True
                )
            ]
        )
        delta_g = np.column_stack(
            [
                new - old
                for old, new in zip(self._maps[:-1], self._maps[1:], strict=True)
            ]
        )
        scale = max(float(np.max(np.abs(delta_r))), norm)
        try:
            coefficients = np.linalg.lstsq(
                delta_r / scale, residual / scale, rcond=1e-10
            )[0]
        except np.linalg.LinAlgError:
            self._maps.clear()
            self._residuals.clear()
            return relaxed
        proposal = np.asarray(
            (mapped.ravel() - delta_g @ coefficients).reshape(current.shape),
            dtype=np.float64,
        )
        if (
            np.all(np.isfinite(proposal))
            and np.all(proposal >= 0.0)
            and float(np.max(np.abs(proposal - current))) <= 4.0 * norm
        ):
            return proposal
        self._maps.clear()
        self._residuals.clear()
        return relaxed
