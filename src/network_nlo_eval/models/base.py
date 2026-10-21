"""QoT 模型的可插拔契约：单跨物理核、模型信息、注册表。

三层分工（见 `PLAN.md`）：

``base``
    协议与注册表。不含物理。
``route``
    路由累积：跨段切分、EDFA 增益与 ASE、ROADM 损耗/均衡/滤波惩罚/阻断、SNR。
    所有模型共用这一份，模型之间只许在"单跨物理"上不同。
``models/<name>``
    只实现单跨核：给定入纤功率谱，返回出纤功率与 NLI 噪声（SPM/SCI、XPM/XCI）。

协议刻意保持窄：一个核只需要 ``evaluate``（可再加 ``evaluate_batch``）。放大器、
ROADM 与噪声累积都不属于核，因此换模型不会改变后处理语义——这是同条件对比的
前提，也是新增模型不必重写一遍 SNR 逻辑的原因。
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat

if TYPE_CHECKING:
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

__all__ = [
    "MODEL_REGISTRY",
    "ModelInfo",
    "MultiSpanOpticalPath",
    "SpanContext",
    "PathQoTResult",
    "QoTModel",
    "SpanFiberOutput",
    "SpanKernel",
    "available_models",
    "create_model",
    "get_model_info",
    "register_model",
]


@dataclass
class MultiSpanOpticalPath:
    """一条光路：逐跳的光纤、跨段 EDFA 与（可选）ROADM 配置。

    这里的类型只用于注解，运行时不构造网络元件，因此 ``base`` 不 import
    ``network.elements``。挪到本模块是为了让 ``models.route`` 与各模型实现都能
    引用同一个路径类型，而不形成 ``isrs_gn`` <-> ``route`` 的循环导入。
    """

    fiber_configs: list[FiberSpanConfig]
    edfa_configs: list[EDFAConfig]
    roadm_configs: list[ROADMConfig] | None = None
    # 实际路径中的每个 link_key, 方便关联到 network_state
    link_keys: list[tuple[int, int]] = field(default_factory=list)
    node_path: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证所有逐跳配置长度一致."""
        count = len(self.fiber_configs)
        if count == 0 or len(self.edfa_configs) != count:
            raise ValueError("Optical path requires matching non-empty fiber and EDFA config lists.")
        if self.roadm_configs is not None and len(self.roadm_configs) != count:
            raise ValueError("ROADM config count must match the number of route links.")
        if self.link_keys and len(self.link_keys) != count:
            raise ValueError("Link key count must match the number of route links.")


@dataclass(frozen=True)
class PathQoTResult:
    """统计物理层沿路径计算的逐信道结果."""

    signal_power_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    ase_noise_w: NDArrayFloat
    snr_db: NDArrayFloat
    span_count: int
    roadm_count: int


@dataclass(frozen=True)
class SpanFiberOutput:
    """单跨光纤内的结果，尚未经过放大器与 ROADM。

    Attributes
    ----------
        power_out_w: 出纤功率 (W)，含光纤损耗与 SRS/ISRS 功率转移，不含放大。
        spm_noise_w: 本跨产生的 SPM（或论文口径的 SCI）噪声功率 (W)。
        xpm_noise_w: 本跨产生的 XPM（或论文口径的 XCI）噪声功率 (W)。
        nli_amplitude_spm: 可选的 SPM 复振幅，``|amp|**2`` 必须等于 ``spm_noise_w``。
        nli_amplitude_xpm: 可选的 XPM 复振幅，``|amp|**2`` 必须等于 ``xpm_noise_w``。

    振幅的参考面是**本跨出纤处**：路由层负责把它按后续跨的功率传输因子（取
    ``sqrt``）送到接收端。跨间相位必须由模型自己写进振幅，否则相干累加无从谈起。
    给出振幅时，跨与跨之间按复数相加（``|Σ a_k|²``）；不给时按功率相加（``Σ|a_k|²``），
    后者是既有七个模型的行为，逐位不变。
    """

    power_out_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    nli_amplitude_spm: NDArrayComplex | None = None
    nli_amplitude_xpm: NDArrayComplex | None = None


@dataclass(frozen=True)
class SpanContext:
    """一跨在整条链路中的位置。

    跨间相干累加需要知道"这是第几跨"以及到接收端还有多长，否则模型无法给出各跨
    NLI 振幅之间的相对相位。路由层按路径顺序填充这个结构，只传给声明了
    ``supports_span_context = True`` 的模型；其余模型继续走纯功率口径，行为不变。
    """

    span_index: int
    total_spans: int
    cumulative_length_m: float
    span_length_m: float


@dataclass(frozen=True)
class ElementOutput:
    """经过放大器与（可选）ROADM 之后的一跨结果。

    Attributes
    ----------
        power_out_w: 放大器后、ROADM 后的出跨功率 (W)。
        spm_noise_w: 本跨 SPM/SCI 噪声 (W)，随信号一起被放大与损耗。
        xpm_noise_w: 本跨 XPM/XCI 噪声 (W)。
        ase_noise_w: 本跨放大器 ASE (W)。
        nli_amplitude_spm: 可选的 SPM 复振幅（已含放大器增益与 ROADM 损耗的幅度因子）。
        nli_amplitude_xpm: 可选的 XPM 复振幅。二者与 ``*_noise_w`` 的模方一一对应。
    """

    power_out_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    ase_noise_w: NDArrayFloat
    nli_amplitude_spm: NDArrayComplex | None = None
    nli_amplitude_xpm: NDArrayComplex | None = None


@runtime_checkable
class SpanKernel(Protocol):
    """单跨物理核。

    实现者只依赖光纤与频谱参数，不读 EDFA/ROADM 配置，也不做噪声累积。
    """

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """计算一个跨段的出纤功率与 NLI 噪声。

        Args:
            power_in_w: ``(N,)`` 跨段入口每信道功率 (W)。0 表示该信道未占用。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 用于 NLI 累积修正的有效跨段数，单跨取 1。

        Returns
        -------
            出纤功率与 SPM/XPM 噪声，均为 ``(N,)``，且必须是与输入不共享内存的新数组。
        """
        ...


@runtime_checkable
class BatchSpanKernel(SpanKernel, Protocol):
    """支持批量单跨计算的核。

    ``(B, N)`` 候选功率进，``(B, N)`` 结果出。批量形态的存在意义是把按候选重复的
    (N, N) 常量矩阵复用成一次 GEMM；结果与逐候选调用的一致性由各自测试保证。
    """

    def evaluate_batch(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """批量版本，参数语义同 :meth:`SpanKernel.evaluate`。"""
        ...


@dataclass(frozen=True)
class ModelInfo:
    """模型的静态描述，供注册表与文档使用。

    Attributes
    ----------
        name: 注册表键，CLI 的 ``--qot-model`` 取值。
        summary: 一句话说明，进入 ``network-nlo-eval models`` 输出。
        family: ``"gn"`` / ``"egn"`` / ``"numerical"`` / 组合，如 ``"gn+egn"``。
        reference: 论文引用（作者、期刊、年份、DOI 或文内编号）。
        options: ``(名称, 说明)`` 序列，列出构造函数接受的模型专属选项。
    """

    name: str
    summary: str
    family: str
    reference: str
    options: tuple[tuple[str, str], ...] = ()


@runtime_checkable
class QoTModel(Protocol):
    """一个可评估路径的 QoT 模型。

    实现者除了 :meth:`kernel_for`，还需要暴露路由层读的逐节点数组与阻断判定。
    通常直接继承 :class:`network_nlo_eval.models.route.RouteModel`，只实现
    ``kernel_for`` 与 ``_build_kernel``。
    """

    info: ModelInfo
    symbol_rate_hz: float

    @property
    def n_nodes(self) -> int:
        """节点数：遗留模式 = 信道数；逐连接模式 = 连接数。"""
        ...

    @property
    def f_abs_hz(self) -> NDArrayFloat:
        """``(N,)`` 节点中心绝对频率 (Hz)。"""
        ...

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """``(N,)`` 逐节点信道带宽 ``B_i`` (Hz)。"""
        ...

    def kernel_for(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """返回给定光纤参数对应的（可缓存的）单跨核。"""
        ...

    def blocked_nodes(self, roadm_config: ROADMConfig) -> list[int]:
        """该 ROADM 阻断的节点索引。"""
        ...

    def power_width_scale(self) -> NDArrayFloat:
        """``(N,)`` 入纤功率的宽度归一化因子；``fixed_power`` 恒为 1。"""
        ...

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """单跨：光纤物理 + 放大器，不含 ROADM。"""
        ...

    def evaluate_path(self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat) -> PathQoTResult:
        """沿路径评估，返回逐信道结果。"""
        ...


ModelFactory = Callable[..., QoTModel]

#: 是否已扫描过 ``network_nlo_eval.models`` 下的实现模块。
_DISCOVERED = False


def _discover_models() -> None:
    """导入包内所有模型模块，让它们各自完成注册。

    注册是模块导入的副作用。逐个 import 而不是维护一份手写清单，是为了让新增
    模型只需落一个文件、不必回来改这个中心模块——并行开发时中心文件的每一处
    改动都是冲突点。只扫描一层，不递归进子包。
    """
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    package = importlib.import_module("network_nlo_eval.models")
    for module in pkgutil.iter_modules(package.__path__):
        if module.name in {"base", "route"}:
            continue
        importlib.import_module(f"{package.__name__}.{module.name}")

#: 名称 -> 构造函数。构造函数签名与 :class:`QoTModel` 实现类的 ``__init__`` 一致。
MODEL_REGISTRY: dict[str, ModelFactory] = {}

#: 名称 -> 静态描述。与 ``MODEL_REGISTRY`` 同步维护，便于不构造实例就列出模型。
MODEL_INFO: dict[str, ModelInfo] = {}


def register_model(info: ModelInfo, factory: ModelFactory) -> None:
    """注册一个模型实现。

    Args:
        info: 静态描述，``info.name`` 即注册表键。
        factory: 构造函数。

    Raises
    ------
        ValueError: 名称重复。重名会让 ``--qot-model`` 的选择变得不确定。
    """
    if info.name in MODEL_REGISTRY:
        raise ValueError(f"QoT model {info.name!r} is already registered.")
    MODEL_REGISTRY[info.name] = factory
    MODEL_INFO[info.name] = info


def available_models() -> dict[str, ModelInfo]:
    """返回已注册模型的静态描述，按名称排序。"""
    _discover_models()
    return dict(sorted(MODEL_INFO.items()))


def get_model_info(name: str) -> ModelInfo:
    """返回单个模型描述。

    Raises
    ------
        KeyError: 名称未注册。
    """
    _discover_models()
    try:
        return MODEL_INFO[name]
    except KeyError:
        raise KeyError(f"Unknown QoT model {name!r}; available: {sorted(MODEL_REGISTRY)}.") from None


def create_model(name: str, *args, **kwargs) -> QoTModel:
    """按名称构造模型，参数原样传给该模型的构造函数。

    Raises
    ------
        KeyError: 名称未注册。
    """
    _discover_models()
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown QoT model {name!r}; available: {sorted(MODEL_REGISTRY)}.") from None
    return factory(*args, **kwargs)


def describe_models() -> list[dict[str, object]]:
    """把注册表拍平成可 JSON 序列化的列表，供 CLI 与基准脚本输出。"""
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
