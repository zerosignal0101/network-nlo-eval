"""Opt-in, budget-based PCFM2/H3 dispatch with an inspectable selection reason."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.models.base import ModelInfo, create_model, register_model
from network_nlo_eval.models.numba_kernels import (
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
)
from network_nlo_eval.models.route import RouteModel

if TYPE_CHECKING:
    from network_nlo_eval.core.flexgrid import FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.core.types import NDArrayFloat
    from network_nlo_eval.models.base import (
        ElementOutput,
        MultiSpanOpticalPath,
        PathQoTResult,
        QoTModel,
        SpanKernel,
    )
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = ["PCFM_AUTO_MODEL_INFO", "PCFMAutoModel", "PCFMSelection"]

PCFM_AUTO_MODEL_INFO = ModelInfo(
    name="pcfm_auto",
    summary="Opt-in PCFM2/H3 dispatch by supported physics and estimated work.",
    family="gn",
    reference="Repository selection policy; see docs/pcfm2.md.",
    options=(
        (
            "max_pcfm2_work",
            "PCFM2 work budget; active_channels**4 * spans**2 * band * cells (20000).",
        ),
        ("pcfm2_options", "Dictionary of explicit PCFM2 resolution/profile options."),
        (
            "h3_options",
            "Dictionary of explicit H3 options; defaults to GN with 32 SPP steps.",
        ),
    ),
)


@dataclass(frozen=True)
class PCFMSelection:
    """Record the selected implementation, policy reason, and estimated work."""

    model: str
    reason: str
    estimated_work: int


class PCFMAutoModel(RouteModel):
    """Choose PCFM2 for bounded workloads and H3 for larger routine workloads.

    This is an explicit computational policy, not an accuracy guarantee.
    Low dispersion never silently falls back to H3 when the budget is exceeded.
    Existing application defaults remain unchanged.

    Parameters
    ----------
    grid : SpectrumGrid or FlexibleGrid
        Shared channel grid.
    ref_fiber_config : FiberSpanConfig
        Reference fiber configuration.
    symbol_rate_hz : float, optional
        Reference rectangular bandwidth.
    max_pcfm2_work : int, optional
        Work budget before H3 dispatch.
    pcfm2_options, h3_options : dict or None, optional
        Options forwarded to the corresponding implementation.
    **route_options
        Shared route and flexible-grid settings.
    """

    info = PCFM_AUTO_MODEL_INFO
    supports_distributed_raman = True
    supports_lumped_loss = True

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        max_pcfm2_work: int = 20000,
        pcfm2_options: dict[str, Any] | None = None,
        h3_options: dict[str, Any] | None = None,
        **route_options: Any,
    ) -> None:
        """Initialize the opt-in policy and lazy model delegates."""
        if (
            isinstance(max_pcfm2_work, bool)
            or int(max_pcfm2_work) != max_pcfm2_work
            or max_pcfm2_work < 1
        ):
            msg = "max_pcfm2_work must be a positive integer."
            raise ValueError(msg)
        super().__init__(grid, ref_fiber_config, symbol_rate_hz, **route_options)
        self.max_pcfm2_work = int(max_pcfm2_work)
        self.pcfm2_options = dict(pcfm2_options or {})
        self.h3_options = {"egn": False, "spp_steps": 32, **(h3_options or {})}
        self._route_options = route_options.copy()
        self._delegates: dict[str, QoTModel] = {}
        self.last_selection: PCFMSelection | None = None

    def _select(
        self,
        fibers: list[FiberSpanConfig],
        power: NDArrayFloat,
    ) -> QoTModel:
        """Select from current route physics and active-channel work."""
        self.last_selection = None
        active = int(np.count_nonzero(power > 0))
        spans = sum(int(np.ceil(f.length_km / f.max_span_length_km)) for f in fibers)
        work = (
            active**4
            * spans**2
            * int(self.pcfm2_options.get("band_points", 5))
            * int(self.pcfm2_options.get("island_cells", 8))
        )
        lower = self.f_abs_hz - self.channel_bandwidth_hz_channels / 2
        upper = self.f_abs_hz + self.channel_bandwidth_hz_channels / 2
        low_dispersion = False
        for fiber in fibers:
            reference = fiber.reference_wavelength_nm * 1e-9
            beta2 = _calc_beta2_from_d(fiber.dispersion_parameter_d, reference)
            beta3 = _calc_beta3_from_s_d(
                fiber.dispersion_slope_s, fiber.dispersion_parameter_d, reference
            )
            limits = beta2 + 2 * np.pi * beta3 * (
                np.array([np.min(lower), np.max(upper)]) - C_LIGHT / reference
            )
            low_dispersion |= bool(
                limits[0] * limits[1] <= 0 or np.min(np.abs(limits)) < 5e-27
            )
            for pump in fiber.raman_pumps:
                frequency = C_LIGHT / (pump.wavelength_nm * 1e-9)
                if pump.power_w > 0 and np.any(
                    (lower <= frequency) & (frequency <= upper)
                ):
                    msg = "Automatic PCFM selection rejects a Raman pump overlapping a signal channel."
                    raise ValueError(msg)
        if any(f.lumped_loss for f in fibers):
            if low_dispersion:
                msg = "Automatic PCFM selection cannot cover low dispersion with in-span losses."
                raise ValueError(msg)
            name, reason = (
                "pcfm_h3_raman",
                "In-span losses require H3's segmented profile support.",
            )
        elif work <= self.max_pcfm2_work:
            name, reason = (
                "pcfm2",
                "Full-island coherent PCFM2 fits the configured work budget.",
            )
        elif low_dispersion:
            msg = (
                f"Low-dispersion PCFM2 work estimate {work} exceeds budget "
                f"{self.max_pcfm2_work}; increase max_pcfm2_work or explicitly select pcfm2."
            )
            raise ValueError(msg)
        else:
            name, reason = (
                "pcfm_h3_raman",
                "PCFM2 exceeds the work budget; use H3 for this routine workload.",
            )
        self.last_selection = PCFMSelection(name, reason, work)
        if name not in self._delegates:
            self._delegates[name] = create_model(
                name,
                self.grid,
                self.ref_fiber_config,
                self.symbol_rate_hz,
                **self._route_options,
                **(self.pcfm2_options if name == "pcfm2" else self.h3_options),
            )
        return self._delegates[name]

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Require explicit implementation choice for direct kernel inspection."""
        msg = "pcfm_auto dispatches per path/span; inspect the selected delegate's kernel."
        raise NotImplementedError(msg)

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """Dispatch a standalone span using the same support and budget policy."""
        model = self._select([fiber_config], power_in_w)
        return model.span_step(fiber_config, edfa_config, power_in_w, span_length_m)

    def evaluate_path(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> PathQoTResult:
        """Choose once for the entire path and record the decision."""
        power = np.asarray(power_in_w, dtype=float)
        if (
            power.shape != (self.n_nodes,)
            or not np.all(np.isfinite(power))
            or np.any(power < 0)
        ):
            msg = "power_in_w must be a finite nonnegative channel vector."
            raise ValueError(msg)
        return self._select(optical_path.fiber_configs, power).evaluate_path(
            optical_path, power
        )


register_model(PCFM_AUTO_MODEL_INFO, PCFMAutoModel)
