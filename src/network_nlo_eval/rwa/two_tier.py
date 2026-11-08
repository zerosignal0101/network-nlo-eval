"""两级 QoT 原型：H3 全场 + 对 flagged 信道回退到 ``numerical_dra``（任务 H）。

默认 ``TwoTierConfig(enabled=False)``：只跑 H3，返回预算修正后的 SNR，以及
"如果启用会回退哪些信道"的计划。此时 ``numerical_dra`` 不会被导入，更不会被调用。

**为什么回退要对整条梳求解一次**

数值参考的 XPM/XCI 来自所有干扰信道。只把 flagged 子集当作发射谱会让其余信道
消失，解出来的不是同一个物理问题（NLI 会明显偏小）。因此回退路径对**整条梳**求解
一次 ``numerical_dra``，只把 flagged 信道的 SNR 取回来覆盖 H3 值。代价与 flagged
数量无关，与被评估链路的总信道数有关；``max_fallback_channels`` 是防跑飞的显式
上限：超过就报错，而不是静默截断或挑一部分算。

**不变量**

* 默认不导入、不调用数值模型；
* 回退只改变 ``effective_snr_db`` 中被 flag 的信道，其余信道仍是 H3 + 预算；
* 数值模型自己的 SNR 非有限时按 ``-inf`` 处理，不会把丢解当成合格。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from network_nlo_eval.core.types import NDArrayBool, NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath, PathQoTResult, QoTModel
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)
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
    """H3 模型额外实现的"点结果 + 误差预算"接口。"""

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
    """Build a lazy factory of the independent numerical DRA reference.

    The import happens inside the returned callable, so constructing (or merely
    holding) the factory never imports :mod:`network_nlo_eval.models.numerical_dra`.

    Args:
        grid: 频谱栅格。
        ref_fiber_config: 参考光纤。
        symbol_rate_hz: 符号率 (Hz)。
        **options: 透传给 ``NumericalDRAModel`` 的选项。

    Returns
    -------
        无参工厂；每次调用构造一个新模型实例。
    """

    def factory() -> QoTModel:
        """Construct one ``NumericalDRAModel`` instance."""
        from network_nlo_eval.models.numerical_dra import NumericalDRAModel

        return NumericalDRAModel(grid, ref_fiber_config, symbol_rate_hz=symbol_rate_hz, **options)

    return factory


@dataclass(frozen=True)
class TwoTierConfig:
    """两级评估的显式开关与预算。

    Args:
        enabled: 是否真的运行数值回退。默认 ``False``（只出计划）。
        max_fallback_channels: 单次评估允许回退的信道数上限；超过直接报错。
    """

    enabled: bool = False
    max_fallback_channels: int = 32

    def __post_init__(self) -> None:
        """Reject a non-positive fallback budget.

        Raises
        ------
            ValueError: ``max_fallback_channels < 1``。
        """
        if self.max_fallback_channels < 1:
            raise ValueError(f"max_fallback_channels must be >= 1, got {self.max_fallback_channels}.")


@dataclass(frozen=True)
class TwoTierResult:
    """一次两级评估的完整结果。

    Attributes
    ----------
        field: H3 全场点结果（与默认 ``evaluate_path`` 逐位一致）。
        budget: 同一次评估的误差预算。
        fallback_channels: 计划或实际回退的信道（H3 flagged 且占用）。
        fallback_snr_db: ``(N,)`` 回退 SNR；未回退处为 ``NaN``。
        effective_snr_db: ``(N,)`` 最终用于判定的 SNR：H3+预算，回退信道取数值值。
        fallback_ran: 数值回退是否真的运行过。
        notes: 人类可读的说明。
    """

    field: PathQoTResult
    budget: H3ErrorBudget
    fallback_channels: NDArrayBool
    fallback_snr_db: NDArrayFloat
    effective_snr_db: NDArrayFloat
    fallback_ran: bool
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Copy the arrays and validate their common shape.

        Raises
        ------
            ValueError: 各数组形状不一致或不是一维。
        """
        shape = np.asarray(self.field.snr_db).shape
        for name in ("fallback_channels", "fallback_snr_db", "effective_snr_db"):
            value = np.array(getattr(self, name), copy=True)
            if value.shape != shape:
                raise ValueError(f"{name} shape {value.shape} does not match the field result {shape}.")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (non-finite values as ``None``)."""
        return {
            "fallback_ran": bool(self.fallback_ran),
            "fallback_channels": [int(index) for index in np.flatnonzero(self.fallback_channels)],
            "fallback_snr_db": [
                float(value) if np.isfinite(value) else None for value in self.fallback_snr_db
            ],
            "effective_snr_db": [float(value) if np.isfinite(value) else None for value in self.effective_snr_db],
            "budget": self.budget.to_dict(),
            "notes": list(self.notes),
        }


class TwoTierEvaluator:
    """H3 full field plus an opt-in numerical fallback for flagged channels.

    Args:
        h3_model: H3 模型，必须实现 ``evaluate_path_with_uncertainty``。
        fallback_factory: 无参工厂，返回可用于同一光路的模型；只有
            ``config.enabled`` 为真时才会被调用。
        config: 显式开关与回退预算。
        launch_power_convention: ``power_in_w`` 的口径，``"total"``（默认）或
            ``"per_pol"``。H3 与回退模型各自换算到自己的
            ``launch_power_convention``；两端相同时不做任何算术。

    Raises
    ------
        ValueError: ``launch_power_convention`` 非法。
    """

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
        self.launch_power_convention = resolve_launch_power_convention(launch_power_convention)

    def plan(self, budget: H3ErrorBudget) -> NDArrayBool:
        """Return the channels that a fallback would be run for.

        The trigger is deliberately narrow: only channels that H3 flagged *and*
        actually launched.  A flagged channel is where the coefficient budget is
        stretched; an unlaunched channel has nothing to recompute.

        Args:
            budget: 同一次 H3 评估的误差预算。

        Returns
        -------
            ``(N,)`` 回退掩码。
        """
        validity = budget.validity
        flagged: NDArrayBool = np.asarray(validity.flagged, dtype=bool)
        active: NDArrayBool = np.asarray(validity.active, dtype=bool)
        return np.asarray(flagged & active, dtype=bool)

    def evaluate(
        self,
        optical_path: MultiSpanOpticalPath,
        power_in_w: NDArrayFloat,
    ) -> TwoTierResult:
        """Evaluate the H3 field and, if enabled, the flagged channels numerically.

        Args:
            optical_path: 有序光路。
            power_in_w: ``(N,)`` 入纤功率 (W)，口径为构造时的
                ``launch_power_convention``。

        Returns
        -------
            H3 结果、预算、回退计划与最终有效 SNR。

        Raises
        ------
            ValueError: 启用回退但没有工厂、或 flagged 信道数超过
                ``max_fallback_channels``、或数值模型返回的长度不一致。
        """
        h3_convention = str(getattr(self.h3_model, "launch_power_convention", "total"))
        h3_power = to_convention(power_in_w, self.launch_power_convention, h3_convention)
        result, budget = self.h3_model.evaluate_path_with_uncertainty(optical_path, h3_power)
        flagged = self.plan(budget)
        effective = np.array(budget.adjusted_snr_db(result.snr_db), dtype=np.float64, copy=True)
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
            raise ValueError(
                f"{count} flagged channels exceed max_fallback_channels="
                f"{self.config.max_fallback_channels}; raise the budget explicitly or fix the operating point."
            )
        if self.fallback_factory is None:
            raise ValueError("fallback is enabled but no fallback_factory was supplied.")

        fallback = self.fallback_factory()
        fallback_convention = str(getattr(fallback, "launch_power_convention", "total"))
        fallback_power = to_convention(power_in_w, self.launch_power_convention, fallback_convention)
        fallback_result = fallback.evaluate_path(optical_path, fallback_power)
        values = np.asarray(fallback_result.snr_db, dtype=np.float64)
        if values.shape != effective.shape:
            raise ValueError(
                f"fallback SNR shape {values.shape} does not match the H3 result {effective.shape}."
            )

        fallback_snr[flagged] = values[flagged]
        # 数值解也可能丢解：非有限的回退值按 -inf 处理，不能当合格。
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
            notes=(f"numerical fallback ran for {count} flagged channel(s) of {effective.size}",),
        )
