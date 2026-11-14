"""Two-tier QoT evaluation with optional numerical fallback.

The fallback evaluates the full comb so that all interfering channels contribute
to XPM, then replaces only flagged channels. The default produces a plan without
importing or running the numerical model. The fallback budget rejects excessive
work rather than silently truncating the candidate set."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from network_nlo_eval.models.base import MultiSpanOpticalPath, PathQoTResult, QoTModel
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat
    from network_nlo_eval.models.uncertainty import H3ErrorBudget, H3UncertaintyConfig
    from network_nlo_eval.network.elements import FiberSpanConfig

__all__ = [
    "H3UncertaintyModel",
    "TwoTierConfig",
    "TwoTierEvaluator",
    "TwoTierResult",
    "numerical_dra_factory",
]


class H3UncertaintyModel(QoTModel, Protocol):
    """Protocol for an H3 point result and its uncertainty budget."""

    def evaluate_path_with_uncertainty(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
        *,
        config: H3UncertaintyConfig | None = None,
    ) -> tuple[PathQoTResult, H3ErrorBudget]:
        """Return the default point result together with its error budget."""
        ...


def numerical_dra_factory(
    grid: Any,
    ref_fiber_config: FiberSpanConfig,
    symbol_rate_hz: float = 50e9,
    **options: Any,
) -> Callable[[], QoTModel]:
    """Build a lazy factory for the independent numerical DRA reference.

    Parameters
    ----------
    grid : object
        Spectrum grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber.
    symbol_rate_hz : float, optional
        Symbol rate in Hz.
    **options
        Options forwarded to the numerical model.

    Returns
    -------
    callable
        Factory that creates a new model when called."""

    def factory() -> QoTModel:
        """Construct one ``NumericalDRAModel`` instance."""
        from network_nlo_eval.models.numerical_dra import (  # noqa: PLC0415
            NumericalDRAModel,
        )

        return NumericalDRAModel(
            grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **options
        )

    return factory


@dataclass(frozen=True)
class TwoTierConfig:
    """Control numerical fallback and its channel budget.

    Parameters
    ----------
    enabled : bool, optional
        Run numerical fallback; by default only produce a plan.
    max_fallback_channels : int, optional
        Maximum flagged channels per evaluation."""

    enabled: bool = False
    max_fallback_channels: int = 32

    def __post_init__(self) -> None:
        """Reject a non-positive fallback budget."""
        if self.max_fallback_channels < 1:
            msg = (
                f"max_fallback_channels must be >= 1, got {self.max_fallback_channels}."
            )
            raise ValueError(msg)


@dataclass(frozen=True)
class TwoTierResult:
    """Complete result of a two-tier evaluation.

    Attributes
    ----------
    field : PathQoTResult
        H3 full-field point result.
    budget : H3ErrorBudget
        Associated uncertainty budget.
    fallback_channels : numpy.ndarray
        Mask of flagged active channels.
    fallback_snr_db : numpy.ndarray
        Numerical SNR, with NaN outside the fallback mask.
    effective_snr_db : numpy.ndarray
        Final SNR after uncertainty adjustment and fallback.
    fallback_ran : bool
        Whether the numerical model ran.
    notes : tuple of str
        Human-readable evaluation notes."""

    field: PathQoTResult
    budget: H3ErrorBudget
    fallback_channels: NDArrayBool
    fallback_snr_db: NDArrayFloat
    effective_snr_db: NDArrayFloat
    fallback_ran: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Copy the result arrays and validate their shapes."""
        shape = np.asarray(self.field.snr_db).shape
        for name in ("fallback_channels", "fallback_snr_db", "effective_snr_db"):
            value = np.array(getattr(self, name), copy=True)
            if value.shape != shape:
                msg = f"{name} shape {value.shape} does not match the field result {shape}."
                raise ValueError(msg)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (non-finite values as ``None``)."""
        return {
            "fallback_ran": bool(self.fallback_ran),
            "fallback_channels": [
                int(index) for index in np.flatnonzero(self.fallback_channels)
            ],
            "fallback_snr_db": [
                float(value) if np.isfinite(value) else None
                for value in self.fallback_snr_db
            ],
            "effective_snr_db": [
                float(value) if np.isfinite(value) else None
                for value in self.effective_snr_db
            ],
            "budget": self.budget.to_dict(),
            "notes": list(self.notes),
        }


class TwoTierEvaluator:
    """Evaluate the H3 field and optionally fall back for flagged channels.

    Parameters
    ----------
    h3_model : H3UncertaintyModel
        Model returning a point result and uncertainty budget.
    fallback_factory : callable or None, optional
        Lazy factory for a numerical model of the same path.
    config : TwoTierConfig or None, optional
        Fallback switch and channel budget.
    launch_power_convention : str, optional
        Input power convention, either total or per polarization. Each model
        receives power converted to its own convention."""

    def __init__(
        self,
        h3_model: H3UncertaintyModel,
        fallback_factory: Callable[[], QoTModel] | None = None,
        *,
        config: TwoTierConfig | None = None,
        launch_power_convention: str = "total",
    ) -> None:
        self.h3_model = h3_model
        self.fallback_factory = fallback_factory
        self.config = config or TwoTierConfig()
        self.launch_power_convention = resolve_launch_power_convention(
            launch_power_convention
        )

    def plan(self, budget: H3ErrorBudget) -> NDArrayBool:
        """Return the flagged and active channels selected for fallback.

        Parameters
        ----------
        budget : H3ErrorBudget
            Uncertainty budget from the same H3 evaluation.

        Returns
        -------
        numpy.ndarray
            Boolean fallback mask."""
        validity = budget.validity
        flagged: NDArrayBool = np.asarray(validity.flagged, dtype=bool)
        active: NDArrayBool = np.asarray(validity.active, dtype=bool)
        return np.asarray(flagged & active, dtype=bool)

    def evaluate(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> TwoTierResult:
        """Evaluate the H3 field and optionally run numerical fallback.

        Parameters
        ----------
        optical_path : MultiSpanOpticalPath
            Ordered optical path.
        power_in_w : numpy.ndarray
            Input launch powers in the configured convention.

        Returns
        -------
        TwoTierResult
            H3 result, budget, fallback plan and effective SNR.

        Raises
        ------
        ValueError
            If fallback lacks a factory, exceeds its channel budget, or returns a
            mismatched shape."""
        h3_convention = str(getattr(self.h3_model, "launch_power_convention", "total"))
        h3_power = to_convention(
            power_in_w, self.launch_power_convention, h3_convention
        )
        result, budget = self.h3_model.evaluate_path_with_uncertainty(
            optical_path, h3_power
        )
        flagged = self.plan(budget)
        effective = np.array(
            budget.adjusted_snr_db(result.snr_db), dtype=np.float64, copy=True
        )
        fallback_snr = np.full(effective.shape, np.nan, dtype=np.float64)
        count = int(np.count_nonzero(flagged))

        if not self.config.enabled:
            return TwoTierResult(
                field=result,
                budget=budget,
                fallback_channels=flagged,
                fallback_snr_db=fallback_snr,
                effective_snr_db=effective,
                fallback_ran=False,
                notes=(f"fallback disabled; {count} channel(s) would be re-evaluated",),
            )

        if count == 0:
            return TwoTierResult(
                field=result,
                budget=budget,
                fallback_channels=flagged,
                fallback_snr_db=fallback_snr,
                effective_snr_db=effective,
                fallback_ran=False,
                notes=("fallback enabled but no channel was flagged",),
            )

        if count > self.config.max_fallback_channels:
            msg = (
                f"{count} flagged channels exceed max_fallback_channels="
                f"{self.config.max_fallback_channels}; raise the budget explicitly or fix the operating point."
            )
            raise ValueError(msg)
        if self.fallback_factory is None:
            msg = "fallback is enabled but no fallback_factory was supplied."
            raise ValueError(msg)

        fallback = self.fallback_factory()
        fallback_convention = str(getattr(fallback, "launch_power_convention", "total"))
        fallback_power = to_convention(
            power_in_w, self.launch_power_convention, fallback_convention
        )
        fallback_result = fallback.evaluate_path(optical_path, fallback_power)
        values = np.asarray(fallback_result.snr_db, dtype=np.float64)
        if values.shape != effective.shape:
            msg = f"fallback SNR shape {values.shape} does not match the H3 result {effective.shape}."
            raise ValueError(msg)

        fallback_snr[flagged] = values[flagged]
        # NaN or negative-infinite fallback SNR must not pass QoT.
        usable = flagged & ~np.isnan(values) & ~np.isneginf(values)
        effective[flagged] = -np.inf
        effective[usable] = values[usable]
        return TwoTierResult(
            field=result,
            budget=budget,
            fallback_channels=flagged,
            fallback_snr_db=fallback_snr,
            effective_snr_db=effective,
            fallback_ran=True,
            notes=(
                f"numerical fallback ran for {count} flagged channel(s) of {effective.size}",
            ),
        )
