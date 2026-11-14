"""Contracts and registry for pluggable quality-of-transmission models.

Span kernels compute fiber propagation and nonlinear noise. The shared
route layer handles span splitting, amplification, ROADMs, and noise
accumulation, keeping post-processing consistent across model choices.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat
    from network_nlo_eval.network.elements import (
        EDFAConfig,
        FiberSpanConfig,
        ROADMConfig,
    )

__all__ = [
    "MODEL_REGISTRY",
    "ModelInfo",
    "MultiSpanOpticalPath",
    "PathQoTResult",
    "QoTModel",
    "SpanContext",
    "SpanFiberOutput",
    "SpanKernel",
    "available_models",
    "create_model",
    "get_model_info",
    "register_model",
]


@dataclass
class MultiSpanOpticalPath:
    """Describe the ordered physical components of an optical path.

    Parameters
    ----------
    fiber_configs : list of FiberSpanConfig
        Fiber configuration for each route link.
    edfa_configs : list of EDFAConfig
        Amplifier configuration for each route link.
    roadm_configs : list of ROADMConfig or None, optional
        Optional ROADM configuration for each route link.
    link_keys : list of tuple of int, optional
        Normalized link keys associated with the path.
    node_path : list of int, optional
        Ordered internal node IDs.
    """

    fiber_configs: list[FiberSpanConfig]
    edfa_configs: list[EDFAConfig]
    roadm_configs: list[ROADMConfig] | None = None
    # Link keys associate path segments with network state.
    link_keys: list[tuple[int, int]] = field(default_factory=list)
    node_path: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate that the per-link configuration counts match."""
        count = len(self.fiber_configs)
        if count == 0 or len(self.edfa_configs) != count:
            msg = (
                "Optical path requires matching non-empty fiber and EDFA config lists."
            )
            raise ValueError(msg)
        if self.roadm_configs is not None and len(self.roadm_configs) != count:
            msg = "ROADM config count must match the number of route links."
            raise ValueError(msg)
        if self.link_keys and len(self.link_keys) != count:
            msg = "Link key count must match the number of route links."
            raise ValueError(msg)


@dataclass(frozen=True)
class PathQoTResult:
    """Per-channel statistical physical-layer results along a path.

    Attributes
    ----------
    signal_power_w, spm_noise_w, xpm_noise_w, ase_noise_w : numpy.ndarray
        Signal and noise powers per channel in watts.
    snr_db : numpy.ndarray
        Signal-to-noise ratio per channel in dB.
    span_count, roadm_count : int
        Number of traversed spans and ROADMs.
    """

    signal_power_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    ase_noise_w: NDArrayFloat
    snr_db: NDArrayFloat
    span_count: int
    roadm_count: int


@dataclass(frozen=True)
class SpanFiberOutput:
    """Per-channel output of one fiber span before amplification and ROADMs.

    Attributes
    ----------
    power_out_w : numpy.ndarray
        Output signal power in watts, including fiber loss and Raman transfer.
    spm_noise_w, xpm_noise_w : numpy.ndarray
        SPM/SCI and XPM/XCI noise powers generated in the span, in watts.
    nli_amplitude_spm, nli_amplitude_xpm : numpy.ndarray or None
        Optional complex amplitudes at the fiber output. Their squared
        magnitudes must equal the corresponding noise powers.
    fwm_noise_w : numpy.ndarray or None
        Optional FWM/MCI noise in watts. Its meaning is model-dependent: it
        may be a diagnostic subset already included in SPM/XPM, or an
        explicitly separated additive contribution to be accumulated by power.

    Notes
    -----
    The route layer propagates amplitudes to the receiver using the square
    root of later power-transfer factors. Models must encode inter-span phase
    in the amplitudes. Supplied amplitudes are summed coherently; otherwise
    noise powers are summed incoherently.
    """

    power_out_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    nli_amplitude_spm: NDArrayComplex | None = None
    nli_amplitude_xpm: NDArrayComplex | None = None
    fwm_noise_w: NDArrayFloat | None = None


@dataclass(frozen=True)
class SpanContext:
    """Describe a span's position in the complete optical path.

    Attributes
    ----------
    span_index, total_spans : int
        Index of the span and total number of spans.
    cumulative_length_m, span_length_m : float
        Cumulative path length and current span length in meters.

    Notes
    -----
    The route layer supplies this context only to models declaring support
    for span context, allowing them to encode inter-span phase.
    """

    span_index: int
    total_spans: int
    cumulative_length_m: float
    span_length_m: float


@dataclass(frozen=True)
class ElementOutput:
    """Output of one span after amplification and optional ROADM processing.

    Attributes
    ----------
    power_out_w, spm_noise_w, xpm_noise_w, ase_noise_w : numpy.ndarray
        Output signal and generated noise powers per channel in watts.
    nli_amplitude_spm, nli_amplitude_xpm : numpy.ndarray or None
        Optional complex SPM and XPM amplitudes after gain and loss.
    fwm_noise_w : numpy.ndarray or None
        Optional additive FWM power explicitly separated from XPM. The route
        layer accumulates it incoherently and adds it back to path-level XPM.
        Diagnostic FWM already included in XPM must not be added again.
    """

    power_out_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    ase_noise_w: NDArrayFloat
    nli_amplitude_spm: NDArrayComplex | None = None
    nli_amplitude_xpm: NDArrayComplex | None = None
    fwm_noise_w: NDArrayFloat | None = None


@runtime_checkable
class SpanKernel(Protocol):
    """Protocol for a single-span fiber physics kernel.

    Kernels depend on fiber and spectrum parameters; amplification, ROADMs,
    and accumulated path noise are handled separately.
    """

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """Evaluate fiber output power and nonlinear noise for one span.

        Parameters
        ----------
        power_in_w : numpy.ndarray
            Per-channel span input powers in watts, with zero for idle channels.
        span_length_m : float
            Span length in meters.
        n_effective_spans : int, optional
            Effective span count for NLI accumulation correction.

        Returns
        -------
        SpanFiberOutput
            New output and noise arrays that do not share memory with the input.
        """
        ...


@runtime_checkable
class BatchSpanKernel(SpanKernel, Protocol):
    """Protocol for kernels accepting a batch of candidate power profiles.

    Input and output arrays have shape ``(candidates, channels)``. Results
    must agree with separate evaluations of each candidate.
    """

    def evaluate_batch(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """Evaluate a batch of power profiles with the same semantics as ``evaluate``."""
        ...


@dataclass(frozen=True)
class ModelInfo:
    """Static description of a registered QoT model.

    Attributes
    ----------
    name : str
        Registry key and CLI model name.
    summary : str
        Short model description.
    family : str
        Model family, such as ``"gn"``, ``"egn"``, or ``"numerical"``.
    reference : str
        Reference to the model publication.
    options : tuple of tuple of str
        Pairs of model-specific option names and descriptions.
    """

    name: str
    summary: str
    family: str
    reference: str
    options: tuple[tuple[str, str], ...] = ()


@runtime_checkable
class QoTModel(Protocol):
    """Protocol for a quality-of-transmission model that evaluates a path.

    Implementations provide span kernels, per-node arrays, and ROADM blocking
    information consumed by the common route layer.
    """

    info: ModelInfo
    symbol_rate_hz: float

    @property
    def n_nodes(self) -> int:
        """Return the number of channels or connections."""
        ...

    @property
    def f_abs_hz(self) -> NDArrayFloat:
        """Return the absolute center frequency in Hz for each node."""
        ...

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """Return the channel bandwidth in Hz for each node."""
        ...

    def kernel_for(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """Return a possibly cached single-span kernel for a fiber configuration."""
        ...

    def blocked_nodes(self, roadm_config: ROADMConfig) -> list[int]:
        """Return indices blocked by a ROADM configuration."""
        ...

    def power_width_scale(self) -> NDArrayFloat:
        """Return the input-power width normalization factor for each node."""
        ...

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """Evaluate fiber and amplifier effects for a span without a ROADM."""
        ...

    def evaluate_path(
        self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat
    ) -> PathQoTResult:
        """Evaluate a path and return per-channel results."""
        ...


ModelFactory = Callable[..., QoTModel]

#: Whether model implementation modules have been discovered.
_DISCOVERED = False


def _discover_models() -> None:
    """Import model modules once to trigger their registration.

    Discovery scans one level of the models package. Each implementation
    registers itself as an import side effect.
    """
    global _DISCOVERED  # noqa: PLW0603 - cache one-time discovery
    if _DISCOVERED:
        return
    _DISCOVERED = True
    package = importlib.import_module("network_nlo_eval.models")
    for module in pkgutil.iter_modules(package.__path__):
        if module.name in {"base", "route"}:
            continue
        importlib.import_module(f"{package.__name__}.{module.name}")


#: Model name to constructor.
MODEL_REGISTRY: dict[str, ModelFactory] = {}

#: Model name to static description; maintained alongside the registry.
MODEL_INFO: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo, factory: ModelFactory) -> None:
    """Register a model implementation.

    Parameters
    ----------
    info : ModelInfo
        Model description; its name is used as the registry key.
    factory : callable
        Model constructor.

    Raises
    ------
    ValueError
        If a model with the same name is already registered.
    """
    if info.name in MODEL_REGISTRY:
        msg = f"QoT model {info.name!r} is already registered."
        raise ValueError(msg)
    MODEL_REGISTRY[info.name] = factory
    MODEL_INFO[info.name] = info


def available_models() -> dict[str, ModelInfo]:
    """Return registered model descriptions sorted by name."""
    _discover_models()
    return dict(sorted(MODEL_INFO.items()))


def get_model_info(name: str) -> ModelInfo:
    """Return the description of a registered model.

    Parameters
    ----------
    name : str
        Model registry key.

    Returns
    -------
    ModelInfo
        Registered model description.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    _discover_models()
    try:
        return MODEL_INFO[name]
    except KeyError:
        msg = f"Unknown QoT model {name!r}; available: {sorted(MODEL_REGISTRY)}."
        raise KeyError(msg) from None


def create_model(name: str, *args, **kwargs) -> QoTModel:
    """Construct a registered model, forwarding all arguments to its factory.

    Parameters
    ----------
    name : str
        Model registry key.
    *args, **kwargs
        Positional and keyword arguments passed to the model factory.

    Returns
    -------
    QoTModel
        Constructed model.

    Raises
    ------
    KeyError
        If the name is not registered.
    """
    _discover_models()
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError:
        msg = f"Unknown QoT model {name!r}; available: {sorted(MODEL_REGISTRY)}."
        raise KeyError(msg) from None
    return factory(*args, **kwargs)


def describe_models() -> list[dict[str, object]]:
    """Return model descriptions in a JSON-serializable list."""
    return [
        {
            "name": info.name,
            "summary": info.summary,
            "family": info.family,
            "reference": info.reference,
            "options": [list(option) for option in info.options],
        }
        for info in available_models().values()
    ]
