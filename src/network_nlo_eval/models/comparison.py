"""Reproducible comparison configurations for PCFM and lumped Raman.

The historical PCFM configuration disables the later edge fix and span cache.
It still inherits the repository's power convention and inter-span coherence;
it is not an unmodified reproduction of the published model or author software.
The ``raman_lumped`` key is retained as an unpumped compatibility preset.
Active pumps require a model with coupled signal, pump, and ASE equations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from network_nlo_eval.models.base import ModelInfo, register_model
from network_nlo_eval.models.pcfm import DEFAULT_SPP_STEPS, PCFMModel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "PCFM_ORIGINAL_MODEL_INFO",
    "RAMAN_LUMPED_MODEL_INFO",
    "PCFMOriginalModel",
    "RamanLumpedModel",
]


PCFM_ORIGINAL_MODEL_INFO = ModelInfo(
    name="pcfm_original",
    summary="Historical PCFM preset (degree 5, GN, no edge fix/cache); retains repository coherence.",
    family="gn",
    reference="Poggiolini et al., JLT 2026, doi:10.1109/jlt.2026.3678322",
)


class PCFMOriginalModel(PCFMModel):
    """Historical PCFM preset retaining the parent model's physics extensions."""

    info = PCFM_ORIGINAL_MODEL_INFO

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        **route_options: Any,
    ) -> None:
        """Disable the later edge correction and span cache in the shared model."""
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            poly_degree=5,
            egn=False,
            edge_fix=False,
            cache_spans=False,
            spp_steps=DEFAULT_SPP_STEPS,
            **route_options,
        )


RAMAN_LUMPED_MODEL_INFO = ModelInfo(
    name="raman_lumped",
    summary="Unpumped PCFM compatibility preset; active pumps require a distributed-Raman model.",
    family="gn+raman",
    reference="Poggiolini et al., JLT 2026, doi:10.1109/jlt.2026.3678322",
    options=(
        (
            "raman_pumps",
            "Compatibility setting: active pumps require pcfm_h3_raman or hybrid_raman_paper.",
        ),
        ("raman_gain_model", "'gaussian' (default) or 'linear'."),
        ("raman_ase", "Retained compatibility option; active pumps are unsupported."),
        ("poly_degree", "SPP polynomial degree. Default 5."),
        ("egn", "Apply the Zefreh 2020 EGN correction. Default False."),
        ("edge_fix", "Apply the H3 zero-dispersion edge fix. Default True."),
        ("cache_spans", "Memoise identical transparent spans. Default True."),
        ("spp_steps", "SPP RK4 steps or 'auto'. Default 32."),
    ),
)


class RamanLumpedModel(PCFMModel):
    """Unpumped PCFM compatibility preset; use a DRA model for active pumps."""

    info = RAMAN_LUMPED_MODEL_INFO

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str = "gaussian",
        raman_ase: bool = True,
        poly_degree: int = 5,
        egn: bool = False,
        edge_fix: bool = True,
        cache_spans: bool = True,
        spp_steps: int | str = DEFAULT_SPP_STEPS,
        **route_options: Any,
    ) -> None:
        """Build the opt-in lumped Raman comparison model."""
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            raman_model="lumped",
            raman_pumps=raman_pumps,
            raman_gain_model=raman_gain_model,
            raman_ase=raman_ase,
            poly_degree=poly_degree,
            egn=egn,
            edge_fix=edge_fix,
            cache_spans=cache_spans,
            spp_steps=spp_steps,
            **route_options,
        )


register_model(PCFM_ORIGINAL_MODEL_INFO, PCFMOriginalModel)
register_model(RAMAN_LUMPED_MODEL_INFO, RamanLumpedModel)
