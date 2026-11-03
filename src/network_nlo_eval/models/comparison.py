"""对照模型注册：未改 PCFM、集总拉曼模型与论文 hybrid 子集。

前两个类不是新的物理模型，而是把已有实现固定成**可复现的对照入口**；第三个
类在 ``models.hybrid_raman`` 中独立注册，提供论文 coupled-ODE 的可验证子集。

``pcfm_original``
    论文 PCFM 的原始配置：``poly_degree=5``、``egn=False``、
    ``edge_fix=False``、``cache_spans=False``、``spp_steps=32``。
    它对应 H3 之前的 PCFM，不含零色散边缘修正，也不含跨段缓存。

``raman_lumped``
    PCFM + 本仓的集总等效分布式拉曼放大（``models.raman``）。
    它仍**不是** hybrid Raman 论文的 coupled-ODE 模型；论文模型的完整复现
    仍需把泵浦项加入 SPP/NLI 积分，见 ``docs/raman-amplification.md``。

``hybrid_raman_paper``
    独立的论文对照模型（``models.hybrid_raman``）：Eq. (2)/(4) coupled
    signal/pump/ASE ODE 的可验证子集，NLI 用 PCFM 核 + ODE 剖面近似，
    **不是** Eq. (6)-(9) 全式。它不替代上面的 lumped 模型，而是用来量化
    lumped 近似的偏差。
"""

from __future__ import annotations

from collections.abc import Sequence

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import ModelInfo, register_model
from network_nlo_eval.models.pcfm import DEFAULT_SPP_STEPS, PCFMModel
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "PCFMOriginalModel",
    "RamanLumpedModel",
    "PCFM_ORIGINAL_MODEL_INFO",
    "RAMAN_LUMPED_MODEL_INFO",
]


PCFM_ORIGINAL_MODEL_INFO = ModelInfo(
    name="pcfm_original",
    summary="Unmodified PCFM paper model (degree 5, GN, no edge fix, no span cache).",
    family="gn",
    reference="Poggiolini et al., JLT 2026, doi:10.1109/jlt.2026.3678322",
)


class PCFMOriginalModel(PCFMModel):
    """Original PCFM configuration, before the H3 edge/cache changes."""

    info = PCFM_ORIGINAL_MODEL_INFO

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        **route_options: object,
    ) -> None:
        """Build the paper configuration with its original numerical path."""
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
    summary="PCFM with the lumped-equivalent distributed Raman amplifier (not the paper coupled-ODE model).",
    family="gn+raman",
    reference="Poggiolini et al., JLT 2026, doi:10.1109/jlt.2026.3678322; "
    "de Moura et al., 'Ultra-Wideband Modeling of Optical Fibre Nonlinearity in Hybrid-Amplified Links'; "
    "lumped equivalent implemented in network_nlo_eval.models.raman",
    options=(
        ("raman_pumps", "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear'."),
        ("raman_ase", "Include distributed Raman ASE before the EDFA. Default True."),
        ("poly_degree", "SPP polynomial degree. Default 5."),
        ("egn", "Apply the Zefreh 2020 EGN correction. Default False."),
        ("edge_fix", "Apply the H3 zero-dispersion edge fix. Default True."),
        ("cache_spans", "Memoise identical transparent spans. Default True."),
        ("spp_steps", "SPP RK4 steps or 'auto'. Default 32."),
    ),
)


class RamanLumpedModel(PCFMModel):
    """PCFM plus the repository's lumped-equivalent Raman amplifier."""

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
        **route_options: object,
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
