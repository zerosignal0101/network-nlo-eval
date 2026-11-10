"""网络级 QoT 有效性层：把 H3 的 validity flags 与 margin 接进 RWA/QoT（任务 H）。

设计原则
--------
* **默认关闭**：:class:`LinkValidityPolicy` 的 ``mode`` 默认是 ``"off"``。此时代码
  不计算报告、不读 flags、不改 SNR，默认路径与接入前逐位一致。
* **策略显式**：只有 ``"flag"`` / ``"reject"`` / ``"degrade"`` / ``"soft"``
  四种模式会启用接入，且必须由调用方显式给出；未知取值直接 ``ValueError``。
  没有"猜一个默认策略"的入口。
* **软 margin 不是 hard valid**：``"soft"`` 只按倾斜信息扣一个可加 margin，
  不改 ``valid`` / ``flagged``，也不把 informational 分数当成可用性门槛。
  它只扣 :attr:`LinkValidity.soft_margin_db`，不碰 ``margin_db``；``degrade``
  的既有语义因此保持不变。
* **密集栅格默认只 flag**：``dense_grid_action`` 默认 ``"flag"``，与接入前
  逐位一致；显式 ``"reject"`` 才把 ``dense_grid_overlap`` 的信道判为无效。
  它只改可用性，不改点值，也与 ``dense_grid_penalty_db``（降级 margin）正交。
* **Auto-Raman 分辨率默认只 flag**：``raman_resolution_action`` 默认
  ``"flag"``；``raman_auto_under_resolved`` 是 error-budget 警告，只有显式
  ``"reject"`` 才把它升级成可用性判定。该动作与 ``dense_grid_action`` 正交，
  两者可以同时打开。
* **FWM 未覆盖默认只 flag**：``fwm_action`` 默认 ``"flag"``；``fwm_required``
  是"模型未覆盖 FWM"的诚实标志，不是失败。只有显式 ``"reject"`` 才把命中
  低色散/零色散/参考点越界的信道判为不可用。它同样只改可用性，不改点值，
  并与 ``dense_grid_action`` / ``raman_resolution_action`` 正交。
* **非有限 SNR 永不算合格**：任何启用的模式都会把 ``NaN`` / ``-inf`` 的 SNR 映射为
  ``-inf``。这是 ``NaN < requirement`` 为 ``False`` 那个静默通过缺陷的唯一防线。
* **状态隔离**：:class:`LinkValidity` 是冻结值对象，内部数组在构造时拷贝并置为
  只读；聚合按链路、按信道独立进行，一条链路的结果不会写进另一条链路。

缓存分三层，粒度不同（详见 ``docs/h3-network-tier-design.md``）：

1. H3 单跨核（模型内部）：键为光纤参数 + 跨长，与占用谱无关；
2. :class:`LinkValidity`：键为 ``(link_key, 占用谱指纹, 跨长)``，占用谱任何一个
   信道变化都会换键，见 :class:`LinkValidityCache`；
3. ``LinkState.noise_cache``：RWA 分配/释放时失效，与第 2 层同步。

第 2 层是"链路 × 占用谱"的缓存，不是"链路"的缓存：D 的 margin 依赖泵浦模式、
带宽类别与功率倾斜，而这三者都会随占用谱变化。用链路键单独做缓存会在业务上下线后
把旧 margin 套到新频谱上，这是本层刻意避免的错误。
"""

from __future__ import annotations

import hashlib
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import numpy as np

from network_nlo_eval.core.types import LinkKey, NDArrayBool, NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath, QoTModel
from network_nlo_eval.models.power_conventions import (
    resolve_launch_power_convention,
    to_convention,
)
from network_nlo_eval.models.uncertainty import (
    H3ErrorBudget,
    H3UncertaintyConfig,
    estimate_h3_error_budget,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "ChannelValidity",
    "DenseGridAction",
    "FwmAction",
    "H3LinkValidityProvider",
    "InvalidChannelPolicy",
    "LinkValidity",
    "LinkValidityCache",
    "LinkValidityPolicy",
    "LinkValidityProvider",
    "RamanResolutionAction",
    "TILT_SOFT_MARGIN_FULL_DB",
    "TILT_SOFT_MARGIN_FULL_DB_RECOMMENDED",
    "link_optical_path",
    "occupancy_fingerprint",
    "tilt_soft_margin_db",
]


#: Full-scale additivity of the informational tilt soft margin (dB).
#:
#: The informational score decays from 1 at
#: :data:`~network_nlo_eval.models.pcfm.TILT_INFORMATIONAL_KNEE_DB` (10 dB) to 0
#: at :data:`~network_nlo_eval.models.pcfm.TILT_INFORMATIONAL_ZERO_DB` (30 dB).
#: A full scale of 1 dB puts the hard-flag boundary (15 dB) at
#: ``(1 - 0.84375) * 1 = 0.15625 dB``, which is the order of the ~0.15 dB
#: representation residual measured at the C+L six-pump benchmark.  The scale
#: is deliberately small: it is a soft correction and must never compete with
#: the hard flag, which is what actually gates validity.
TILT_SOFT_MARGIN_FULL_DB: float = 1.0

#: Recommended full scale of the informational tilt soft margin (dB).
#:
#: The fifth-round accuracy matrix measured the H3 residual at the hard-flag
#: boundary: 0.244 dB at 10 dB tilt, 0.302 dB at 15 dB and 0.302 dB at 20 dB.
#: At 15 dB the informational curve has lost ``1 - 0.84375 = 0.15625`` of its
#: confidence, so the default scale produces only 0.156 dB there -- about half
#: the measured residual.  Covering it requires
#: ``0.302 / 0.15625 = 1.9328`` dB; 1.933 dB is the first three-decimal value
#: that is not below the measurement.  The value is a documented
#: recommendation, not the default: :class:`LinkValidityPolicy` still starts
#: at :data:`TILT_SOFT_MARGIN_FULL_DB`, so no existing number moves.
TILT_SOFT_MARGIN_FULL_DB_RECOMMENDED: float = 1.933


def tilt_soft_margin_db(
    confidence: NDArrayFloat | float,
    *,
    full_scale_db: float = TILT_SOFT_MARGIN_FULL_DB,
) -> NDArrayFloat:
    r"""Map informational tilt confidence to an additive soft margin.

    The formula is linear in the *lost* confidence:

    .. math::

        m_{soft} = m_{full} \cdot \left(1 - \mathrm{clip}(c, 0, 1)\right).

    It is zero while the tilt is at or below the informational knee (``c = 1``),
    increases monotonically as the tilt grows, and reaches ``full_scale_db`` at
    the zero of the informational curve.  A non-finite confidence is treated as
    zero confidence (the worst case), so a missing or broken score can only
    increase the margin, never silently remove it.

    The mapping is intentionally not a validity rule: it only produces a
    number.  A caller opts in to subtracting it with
    :class:`LinkValidityPolicy` ``mode="soft"``.

    Args:
        confidence: Informational tilt confidence, scalar or ``(N,)`` array.
        full_scale_db: Margin at ``confidence == 0`` (dB); default
            :data:`TILT_SOFT_MARGIN_FULL_DB`.

    Returns
    -------
        Non-negative soft margin (dB), same shape as ``confidence``; a scalar
        input returns a 0-d array.

    Raises
    ------
        ValueError: ``full_scale_db`` is negative or not finite.
    """
    scale = float(full_scale_db)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError(f"full_scale_db must be finite and >= 0, got {full_scale_db!r}.")
    values = np.asarray(confidence, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    return scale * (1.0 - np.clip(values, 0.0, 1.0))


class InvalidChannelPolicy(StrEnum):
    """无效信道/被 flag 信道的处理策略。

    Attributes
    ----------
        OFF: 不接入有效性层，判定与接入前逐位一致（默认）。
        FLAG: 只审计。有限 SNR 的判定保持不变，无效信道记录在
            :attr:`LinkValidity.valid` 中；非有限 SNR 仍被拒绝。
        REJECT: 无效信道的有效 SNR 置为 ``-inf``；有效信道保持原 SNR。
        DEGRADE: 有效信道扣掉 :attr:`LinkValidity.margin_db`，无效信道置为
            ``-inf``。
        SOFT: 有效信道只扣掉倾斜软 margin :attr:`LinkValidity.soft_margin_db`，
            无效信道置为 ``-inf``。它不扣 ``margin_db``，也不改变任何
            ``valid`` / ``flagged`` 判定，因此可以单独观察"只按倾斜收缩余量"
            的影响。
    """

    OFF = "off"
    FLAG = "flag"
    REJECT = "reject"
    DEGRADE = "degrade"
    SOFT = "soft"


class DenseGridAction(StrEnum):
    """密集栅格重叠（``dense_grid_overlap``）的处理动作。

    Attributes
    ----------
        FLAG: 只记录密集栅格标志，不改变任何可用性判定（默认，保持历史行为）。
        REJECT: 把密集栅格覆盖的信道判为无效。它只做可用性判定，不改变点值
            SNR；后续按 :class:`InvalidChannelPolicy` 的既有模式落地：``flag``
            仍然只审计（SNR 不变，但报告的有效性为 ``False``），``reject`` /
            ``degrade`` / ``soft`` 把该信道压成 ``-inf``。
    """

    FLAG = "flag"
    REJECT = "reject"


class RamanResolutionAction(StrEnum):
    """Auto-Raman 分辨率不足（``raman_auto_under_resolved``）的处理动作。

    Attributes
    ----------
        FLAG: 只记录标志，不改变任何可用性判定（默认，保持历史行为）。
        REJECT: 把 ``raman_auto_under_resolved`` 的信道判为无效。它只做可用性
            判定，不改变点值 SNR；后续按 :class:`InvalidChannelPolicy` 的既有
            模式落地：``flag`` 仍然只审计（SNR 不变，但报告的有效性为
            ``False``），``reject`` / ``degrade`` / ``soft`` 把该信道压成
            ``-inf``。
    """

    FLAG = "flag"
    REJECT = "reject"


class FwmAction(StrEnum):
    """FWM 未覆盖（``fwm_required``）的处理动作。

    ``fwm_required`` 表示当前占用谱落在闭式 kernel 不覆盖 FWM 的区制（低色散、
    带内零色散或参考波长越界）。它是诚实的能力边界标志，不是失败。

    Attributes
    ----------
        FLAG: 只记录标志，不改变任何可用性判定（默认，保持历史行为）。
        REJECT: 把 ``fwm_required`` 的信道判为无效。它只做可用性判定，不改变
            点值 SNR；后续按 :class:`InvalidChannelPolicy` 的既有模式落地：
            ``flag`` 仍然只审计（SNR 不变，但报告的有效性为 ``False``），
            ``reject`` / ``degrade`` / ``soft`` 把该信道压成 ``-inf``。
    """

    FLAG = "flag"
    REJECT = "reject"


#: Accepted values of :attr:`LinkValidity.fwm_validity_mode`.
_FWM_VALIDITY_MODES: tuple[str, ...] = tuple(action.value for action in FwmAction)


@dataclass(frozen=True)
class LinkValidityPolicy:
    """网络级有效性接入的显式开关。

    Args:
        mode: ``"off"`` / ``"flag"`` / ``"reject"`` / ``"degrade"`` / ``"soft"``。
            默认 ``"off"``，即完全不接入；其余四种必须由调用方显式写出。
        soft_margin_full_scale_db: 倾斜软 margin 的满量程 (dB)，默认
            :data:`TILT_SOFT_MARGIN_FULL_DB`（1.0 dB，连同公式一起保持不变）。
            ``LinkValidity.soft_margin_db`` 始终按默认满量程存储；本参数按
            ``soft_margin_full_scale_db / TILT_SOFT_MARGIN_FULL_DB`` 的比例
            缩放它在 ``mode="soft"`` 中实际扣除的量，所以只有显式 opt-in 才会
            改变数值。推荐值见
            :data:`TILT_SOFT_MARGIN_FULL_DB_RECOMMENDED`（1.933 dB）。
        dense_grid_action: ``"flag"`` / ``"reject"``。默认 ``"flag"``：密集栅格
            只是标志，可用性与历史逐位一致。``"reject"`` 在策略已启用
            （``mode != "off"``）时把 ``dense_grid_overlap`` 的信道标为无效。
            它与 :attr:`~network_nlo_eval.models.uncertainty.H3UncertaintyConfig.dense_grid_penalty_db`
            正交：penalty 是降级 margin，本动作是可用性判定。
        raman_resolution_action: ``"flag"`` / ``"reject"``。默认 ``"flag"``：
            ``raman_auto_under_resolved`` 只是 error-budget 警告。``"reject"``
            在策略已启用时把这些信道标为无效。它与 ``dense_grid_action``
            正交：两者同时为 ``"reject"`` 时按逻辑或合并，互不覆盖。
        fwm_action: ``"flag"`` / ``"reject"``。默认 ``"flag"``：
            ``fwm_required`` 只是"模型未覆盖 FWM"的诚实标志。``"reject"`` 在
            策略已启用时把这些信道标为无效。它与 ``dense_grid_action`` /
            ``raman_resolution_action`` 正交，三者同时打开时按逻辑或合并。

    Raises
    ------
        ValueError: ``mode`` 不是上述五个取值之一，``dense_grid_action`` /
            ``raman_resolution_action`` / ``fwm_action`` 不是 ``"flag"`` /
            ``"reject"``，或满量程不是有限非负数。
    """

    mode: InvalidChannelPolicy | str = InvalidChannelPolicy.OFF
    soft_margin_full_scale_db: float = TILT_SOFT_MARGIN_FULL_DB
    dense_grid_action: DenseGridAction | str = DenseGridAction.FLAG
    raman_resolution_action: RamanResolutionAction | str = RamanResolutionAction.FLAG
    fwm_action: FwmAction | str = FwmAction.FLAG

    def __post_init__(self) -> None:
        """把字符串归一化成枚举，并拒绝未知策略、未知动作或非法满量程。"""
        try:
            normalized = InvalidChannelPolicy(self.mode)
        except ValueError:
            allowed = [policy.value for policy in InvalidChannelPolicy]
            raise ValueError(
                f"unknown validity policy {self.mode!r}; expected one of {allowed}."
            ) from None
        object.__setattr__(self, "mode", normalized)
        try:
            dense_action = DenseGridAction(self.dense_grid_action)
        except ValueError:
            allowed = [action.value for action in DenseGridAction]
            raise ValueError(
                f"unknown dense grid action {self.dense_grid_action!r}; expected one of {allowed}."
            ) from None
        object.__setattr__(self, "dense_grid_action", dense_action)
        try:
            raman_action = RamanResolutionAction(self.raman_resolution_action)
        except ValueError:
            allowed = [action.value for action in RamanResolutionAction]
            raise ValueError(
                f"unknown Raman resolution action {self.raman_resolution_action!r}; "
                f"expected one of {allowed}."
            ) from None
        object.__setattr__(self, "raman_resolution_action", raman_action)
        try:
            fwm_action = FwmAction(self.fwm_action)
        except ValueError:
            allowed = [action.value for action in FwmAction]
            raise ValueError(
                f"unknown FWM action {self.fwm_action!r}; expected one of {allowed}."
            ) from None
        object.__setattr__(self, "fwm_action", fwm_action)
        scale = float(self.soft_margin_full_scale_db)
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError(
                f"soft_margin_full_scale_db must be finite and >= 0, got {self.soft_margin_full_scale_db!r}."
            )
        object.__setattr__(self, "soft_margin_full_scale_db", scale)

    @property
    def enabled(self) -> bool:
        """是否启用了有效性接入（``mode != "off"``）。"""
        return self.mode is not InvalidChannelPolicy.OFF

    @property
    def rejects_dense_grid(self) -> bool:
        """``dense_grid_action="reject"`` 且策略已启用。"""
        return self.enabled and self.dense_grid_action is DenseGridAction.REJECT

    @property
    def rejects_raman_resolution(self) -> bool:
        """``raman_resolution_action="reject"`` 且策略已启用。"""
        return self.enabled and self.raman_resolution_action is RamanResolutionAction.REJECT

    @property
    def rejects_fwm(self) -> bool:
        """``fwm_action="reject"`` 且策略已启用。"""
        return self.enabled and self.fwm_action is FwmAction.REJECT

    @staticmethod
    def _flag_mask(flag: bool | NDArrayBool, shape: tuple[int, ...], name: str) -> NDArrayBool:
        """把标量/逐信道标志归一成可广播的布尔掩码。

        Args:
            flag: 标量或 ``(N,)`` 数组。
            shape: ``valid`` 的形状。
            name: 出错信息里的参数名。

        Returns
        -------
            0-d 或 ``(N,)`` 的布尔数组。

        Raises
        ------
            ValueError: 既不是标量，也不是与 ``valid`` 同形状的一维数组。
        """
        mask = np.asarray(flag, dtype=bool)
        if mask.ndim > 1 or mask.shape not in ((), shape):
            raise ValueError(f"{name} must be a scalar or have shape {shape}, got {mask.shape}.")
        return mask

    def effective_valid(
        self,
        valid: NDArrayBool,
        dense_grid_overlap: bool | NDArrayBool = False,
        *,
        raman_auto_under_resolved: bool | NDArrayBool = False,
        fwm_required: bool | NDArrayBool = False,
    ) -> NDArrayBool:
        """Return the availability mask this policy actually decides on.

        ``mode="off"``、默认的 ``dense_grid_action="flag"``、默认的
        ``raman_resolution_action="flag"`` 与默认的 ``fwm_action="flag"`` 都
        原样返回 ``valid``（同一个数组），所以默认路径不做任何算术。只有显式
        ``"reject"`` 动作才会把对应信道的可用性扣成 ``False``，且只影响可用性，
        不改变 SNR 点值。三个动作正交：同时打开时按逻辑或合并。

        Args:
            valid: ``(N,)`` 物理/预算判定给出的有效性。
            dense_grid_overlap: 该占用谱是否为密集栅格；标量或 ``(N,)`` 数组。
            raman_auto_under_resolved: 该占用谱是否命中 auto-Raman 分辨率不足；
                标量或 ``(N,)`` 数组，默认 ``False`` 时完全不参与运算。
            fwm_required: 该占用谱是否命中"模型未覆盖 FWM"；标量或 ``(N,)``
                数组，默认 ``False`` 时完全不参与运算。

        Returns
        -------
            ``(N,)`` 用于策略判定的有效性。

        Raises
        ------
            ValueError: 某个标志既不是标量也不是与 ``valid`` 同形状的一维
                数组。
        """
        values = np.asarray(valid, dtype=bool)
        if not (
            self.rejects_dense_grid or self.rejects_raman_resolution or self.rejects_fwm
        ):
            return values
        effective = values
        if self.rejects_dense_grid:
            dense = self._flag_mask(dense_grid_overlap, values.shape, "dense_grid_overlap")
            if bool(np.any(dense)):
                effective = effective & ~dense
        if self.rejects_raman_resolution:
            under = self._flag_mask(
                raman_auto_under_resolved, values.shape, "raman_auto_under_resolved"
            )
            if bool(np.any(under)):
                effective = effective & ~under
        if self.rejects_fwm:
            fwm = self._flag_mask(fwm_required, values.shape, "fwm_required")
            if bool(np.any(fwm)):
                effective = effective & ~fwm
        return effective

    def scale_soft_margin(self, soft_margin_db: NDArrayFloat) -> NDArrayFloat:
        """Rescale stored default-scale soft margins to this policy's full scale.

        Args:
            soft_margin_db: ``(N,)`` 软 margin (dB)，按
                :data:`TILT_SOFT_MARGIN_FULL_DB` 存储。

        Returns
        -------
            ``(N,)`` 按比例缩放后的软 margin；满量程为默认值时原样返回输入
            数组，保证默认路径逐位不变。
        """
        values = np.asarray(soft_margin_db, dtype=np.float64)
        factor = float(self.soft_margin_full_scale_db) / TILT_SOFT_MARGIN_FULL_DB
        if factor == 1.0:
            return values
        return values * factor

    def apply(
        self,
        snr_db: NDArrayFloat,
        valid: NDArrayBool,
        margin_db: NDArrayFloat,
        soft_margin_db: NDArrayFloat | None = None,
        *,
        dense_grid_overlap: bool | NDArrayBool = False,
        raman_auto_under_resolved: bool | NDArrayBool = False,
        fwm_required: bool | NDArrayBool = False,
    ) -> NDArrayFloat:
        """Apply the policy to an SNR vector.

        这是策略语义的唯一定义，链路报告、路径聚合与标量业务 SNR 都走这里。

        Args:
            snr_db: ``(N,)`` 物理 SNR (dB)。
            valid: ``(N,)`` 有效性标志。
            margin_db: ``(N,)`` 建议 margin (dB)。
            soft_margin_db: ``(N,)`` 倾斜软 margin (dB)；``None`` 视为全 0。
                只有 ``mode="soft"`` 会读这个参数，且会先按
                :attr:`soft_margin_full_scale_db` 缩放。
            dense_grid_overlap: 该占用谱是否为密集栅格；标量或 ``(N,)``。
                只有 ``dense_grid_action="reject"`` 会读它，把对应信道的
                有效性扣成 ``False`` 后交给 ``mode`` 的既有语义；默认
                ``"flag"`` 时完全不参与。
            raman_auto_under_resolved: 该占用谱是否命中 auto-Raman 分辨率不足；
                标量或 ``(N,)``。只有 ``raman_resolution_action="reject"`` 会读
                它，默认 ``"flag"`` 时完全不参与。
            fwm_required: 该占用谱是否命中"模型未覆盖 FWM"；标量或 ``(N,)``。
                只有 ``fwm_action="reject"`` 会读它，默认 ``"flag"`` 时完全不
                参与。

        Returns
        -------
            ``(N,)`` 用于阈值比较的有效 SNR。``off`` 原样返回输入数组。
        """
        values = np.asarray(snr_db, dtype=np.float64)
        mode = InvalidChannelPolicy(self.mode)
        if mode is InvalidChannelPolicy.OFF:
            return values
        return _effective_snr(
            values,
            self.effective_valid(
                valid,
                dense_grid_overlap,
                raman_auto_under_resolved=raman_auto_under_resolved,
                fwm_required=fwm_required,
            ),
            np.asarray(margin_db, dtype=np.float64),
            mode,
            None if soft_margin_db is None else self.scale_soft_margin(soft_margin_db),
        )


def occupancy_fingerprint(power_profile_w: NDArrayFloat, span_length_m: float) -> str:
    """Return a bit-exact key of one link's occupied spectrum.

    指纹覆盖整条功率谱的 IEEE-754 位模式与跨长：任一信道的占用/功率变化都会
    换键，因此 ``LinkValidity`` 不会跨占用谱复用。``-0.0`` 与 ``0.0`` 的位模式
    不同，会得到不同的键；这是有意的严格比较，代价只是多算一次。

    Args:
        power_profile_w: ``(N,)`` 逐信道入纤功率 (W)，0 表示未占用。
        span_length_m: 跨长 (m)。

    Returns
    -------
        16 字节 BLAKE2b 摘要的十六进制串。

    Raises
    ------
        ValueError: 功率谱不是一维数组。
    """
    values = np.asarray(power_profile_w, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"power_profile_w must be one-dimensional, got shape {values.shape}.")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    digest.update(np.float64(span_length_m).tobytes())
    return digest.hexdigest()


def _effective_snr(
    snr_db: NDArrayFloat,
    valid: NDArrayBool,
    margin_db: NDArrayFloat,
    mode: InvalidChannelPolicy,
    soft_margin_db: NDArrayFloat | None = None,
) -> NDArrayFloat:
    """Apply one policy to an SNR array.

    ``off`` 返回输入数组本身（不做任何算术），其余模式按
    :class:`InvalidChannelPolicy` 的定义返回新数组。``NaN`` 在任何启用模式下
    都变成 ``-inf``。

    Args:
        snr_db: ``(N,)`` 物理 SNR (dB)。
        valid: ``(N,)`` 有效性标志。
        margin_db: ``(N,)`` 建议 margin (dB)，无效信道为 ``inf``。
        mode: 已归一化的策略。
        soft_margin_db: ``(N,)`` 倾斜软 margin (dB)；``None`` 视为全 0。只有
            ``soft`` 会读它；``degrade`` 仍只扣 ``margin_db``，语义不变。

    Returns
    -------
        ``(N,)`` 用于阈值比较的有效 SNR。
    """
    values = np.asarray(snr_db, dtype=np.float64)
    if mode is InvalidChannelPolicy.OFF:
        return values
    # ``+inf`` 是零噪声的理想结果，H3 也把它当作可用；只排除 NaN 与 -inf。
    usable = ~np.isnan(values) & ~np.isneginf(values)
    if mode is InvalidChannelPolicy.FLAG:
        # 审计模式：有限 SNR 原样返回，只把 NaN/-inf 变成 -inf 防止静默通过。
        return np.where(usable, values, -np.inf)
    accept = np.asarray(valid, dtype=bool) & usable
    if mode is InvalidChannelPolicy.REJECT:
        return np.where(accept, values, -np.inf)
    if mode is InvalidChannelPolicy.SOFT:
        soft = np.zeros(values.shape, dtype=np.float64) if soft_margin_db is None else soft_margin_db
        # 非有限或负的软 margin 没有安全含义，按 0 处理；它是可加的保守量，
        # 不能因为坏输入让 SNR 变大。
        soft = np.where(np.isfinite(soft) & (soft >= 0.0), soft, 0.0)
        return np.where(accept, values - soft, -np.inf)
    penalty = np.asarray(margin_db, dtype=np.float64)
    penalty = np.where(accept & np.isfinite(penalty), penalty, 0.0)
    return np.where(accept, values - penalty, -np.inf)


@dataclass(frozen=True)
class LinkValidity:
    """一条链路、一份占用谱上的逐信道有效性。

    数组都按模型（ODE 节点）顺序排列，长度一致。构造时拷贝并置为只读，
    因此可以安全地在缓存里共享，也不会被调用方就地改写。

    Attributes
    ----------
        valid: 可用于硬 QoT 判定。未占用、非有限 SNR、泵不收敛或 D 判定无效
            的信道都是 ``False``。
        active: 该链路上有正入纤功率的信道。
        flagged: D 给出的"预算被拉伸"标志（不等于不可用）。
        confidence: ``[0, 1]`` 证据强度，驱动 margin 中的有效性罚项。
        margin_db: 建议加性 margin (dB)；无效信道为 ``inf``，未占用为 ``0``。
        tilt_confidence_informational: ``[0, 1]`` 倾斜信息分数，逐信道常数。
            它不参与 ``valid`` / ``flagged`` / ``confidence``，只用于映射软
            margin；``None`` 等价于全 1（无软 margin）。
        soft_margin_db: 倾斜软 margin (dB)，由
            :func:`tilt_soft_margin_db` 从 ``tilt_confidence_informational``
            按 :data:`TILT_SOFT_MARGIN_FULL_DB` 的默认满量程映射；未占用信道
            为 ``0``。它从不自动生效，只有 ``LinkValidityPolicy(mode="soft")``
            才会从 SNR 扣除；策略的 ``soft_margin_full_scale_db`` 会按比例缩放
            这个存储值（默认 1.0，即不缩放）。
        model: 产生该报告的模型名。
        span_count: 该链路的物理跨数。
        fingerprint: 占用谱指纹，见 :func:`occupancy_fingerprint`。
        link_key: 所属链路键；手工构造的测试对象可以为 ``None``。
        dense_grid_overlap: 占用谱是否为密集栅格（信道间隔小于带宽）。它只是
            域审计标志，不改变本报告存储的 ``valid``；``reject`` 要在策略层
            显式打开（见 :attr:`LinkValidityPolicy.dense_grid_action`）。需要
            额外余量时显式使用
            :attr:`~network_nlo_eval.models.uncertainty.H3UncertaintyConfig.dense_grid_penalty_db`。
        raman_auto_under_resolved: auto-Raman 分辨率审计是否命中（任一跨的
            可选检查发现步数不足）。同样是 error-budget 警告，不改变本报告的
            ``valid`` / margin；``reject`` 要在策略层显式打开（见
            :attr:`LinkValidityPolicy.raman_resolution_action`）。
        fwm_required: 预算是否报告"模型未覆盖 FWM"（低色散、带内零色散或参考
            波长越界）。同样只是诚实标志，不改变本报告的 ``valid`` / margin；
            ``reject`` 要在策略层显式打开（见
            :attr:`LinkValidityPolicy.fwm_action`）。
        fwm_reason: ``""`` 或预算给出的 ``+`` 连接原因串。
        fwm_validity_mode: 预算评估时的 FWM 模式，``"flag"`` 或 ``"reject"``。
    """

    valid: NDArrayBool
    active: NDArrayBool
    flagged: NDArrayBool
    confidence: NDArrayFloat
    margin_db: NDArrayFloat
    model: str
    span_count: int = 0
    fingerprint: str = ""
    link_key: LinkKey | None = None
    tilt_confidence_informational: NDArrayFloat | float | None = None
    soft_margin_db: NDArrayFloat | float | None = None
    dense_grid_overlap: bool = False
    raman_auto_under_resolved: bool = False
    fwm_required: bool = False
    fwm_reason: str = ""
    fwm_validity_mode: str = "flag"

    def __post_init__(self) -> None:
        """Copy, freeze and validate the per-channel arrays.

        ``tilt_confidence_informational`` 与 ``soft_margin_db`` 是可选的：
        ``None`` 分别退化为全 1 与全 0，因此旧调用方构造的报告逐位不变。
        标量输入会广播成逐信道数组。

        Raises
        ------
            ValueError: 数组形状不一致、不是一维、margin 为负或为 ``NaN``、
                confidence 越界、可选字段越界或非有限、``fwm_validity_mode``
                不是 ``"flag"`` / ``"reject"``。
        """
        arrays: dict[str, Any] = {}
        for name, dtype in (
            ("valid", np.bool_),
            ("active", np.bool_),
            ("flagged", np.bool_),
            ("confidence", np.float64),
            ("margin_db", np.float64),
        ):
            value = np.array(getattr(self, name), dtype=dtype, copy=True)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional, got shape {value.shape}.")
            value.setflags(write=False)
            arrays[name] = value
            object.__setattr__(self, name, value)

        shapes = {name: value.shape for name, value in arrays.items()}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"validity arrays must share one shape, got {shapes}.")
        # 可选字段在必需字段校验通过后按同一长度归一化。默认值保证旧调用方
        # 得到的软 margin 恒为 0，从而默认路径逐位不变。
        num_channels = int(arrays["valid"].shape[0])
        for name, default in (
            ("tilt_confidence_informational", 1.0),
            ("soft_margin_db", 0.0),
        ):
            raw = getattr(self, name)
            if raw is None:
                value = np.full(num_channels, default, dtype=np.float64)
            else:
                value = np.asarray(raw, dtype=np.float64)
                if value.ndim == 0:
                    value = np.full(num_channels, float(value), dtype=np.float64)
                elif value.shape != (num_channels,):
                    raise ValueError(f"{name} must have shape ({num_channels},), got {value.shape}.")
                value = np.array(value, dtype=np.float64, copy=True)
            value.setflags(write=False)
            arrays[name] = value
            object.__setattr__(self, name, value)

        if np.any(np.isnan(arrays["margin_db"])) or np.any(arrays["margin_db"] < 0.0):
            raise ValueError("margin_db must be non-negative and never NaN.")
        confidence = arrays["confidence"]
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("confidence must lie in [0, 1].")
        informational = arrays["tilt_confidence_informational"]
        if np.any(~np.isfinite(informational)) or np.any((informational < 0.0) | (informational > 1.0)):
            raise ValueError("tilt_confidence_informational must be finite and lie in [0, 1].")
        soft = arrays["soft_margin_db"]
        if np.any(~np.isfinite(soft)) or np.any(soft < 0.0):
            raise ValueError("soft_margin_db must be finite and non-negative.")
        if int(self.span_count) < 0:
            raise ValueError(f"span_count must be >= 0, got {self.span_count}.")
        object.__setattr__(
            self, "raman_auto_under_resolved", bool(self.raman_auto_under_resolved)
        )
        object.__setattr__(self, "fwm_required", bool(self.fwm_required))
        object.__setattr__(self, "fwm_reason", str(self.fwm_reason))
        if self.fwm_validity_mode not in _FWM_VALIDITY_MODES:
            raise ValueError(
                f"fwm_validity_mode must be one of {_FWM_VALIDITY_MODES}, got {self.fwm_validity_mode!r}."
            )
        object.__setattr__(self, "fwm_validity_mode", str(self.fwm_validity_mode))

    @property
    def num_channels(self) -> int:
        """信道数（数组长度）。"""
        return int(self.valid.shape[0])

    def channel_valid(self, channel_idx: int) -> bool:
        """该信道能否用于硬判定。"""
        return bool(self.valid[channel_idx])

    def margin_at(self, channel_idx: int) -> float:
        """该信道的建议 margin (dB)。"""
        return float(self.margin_db[channel_idx])

    def soft_margin_at(self, channel_idx: int) -> float:
        """该信道的倾斜软 margin (dB)。"""
        return float(self.soft_margin_db[channel_idx])

    def effective_valid(self, policy: LinkValidityPolicy | None = None) -> NDArrayBool:
        """该报告在给定策略下的可用性。

        与 :attr:`valid` 的区别只在显式 ``"reject"`` 动作：
        ``dense_grid_action="reject"`` 时 ``dense_grid_overlap`` 为真的信道返回
        ``False``，``raman_resolution_action="reject"`` 时
        :attr:`raman_auto_under_resolved` 为真的信道返回 ``False``，
        ``fwm_action="reject"`` 时 :attr:`fwm_required` 为真的信道返回 ``False``。
        默认策略（``off`` 或 ``flag`` 动作）原样返回 :attr:`valid`。

        Args:
            policy: 策略；``None`` 等价于默认 ``LinkValidityPolicy()``。

        Returns
        -------
            ``(N,)`` 用于可用性判定的有效标志。
        """
        return (policy or LinkValidityPolicy()).effective_valid(
            self.valid,
            self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )

    def adjusted_snr_db(
        self,
        snr_db: NDArrayFloat,
        policy: LinkValidityPolicy | None = None,
    ) -> NDArrayFloat:
        """Apply the policy to this link's SNR vector.

        Args:
            snr_db: ``(N,)`` 物理 SNR (dB)。
            policy: 策略；``None`` 等价于 ``"off"``（返回输入，不改数值）。

        Returns
        -------
            ``(N,)`` 用于阈值比较的 SNR。

        Raises
        ------
            ValueError: ``snr_db`` 形状与报告不一致。
        """
        values = np.asarray(snr_db, dtype=np.float64)
        if values.shape != self.valid.shape:
            raise ValueError(f"snr_db shape {values.shape} does not match the report {self.valid.shape}.")
        return (policy or LinkValidityPolicy()).apply(
            values,
            self.valid,
            self.margin_db,
            self.soft_margin_db,
            dense_grid_overlap=self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )

    @classmethod
    def from_budget(
        cls,
        budget: H3ErrorBudget,
        *,
        link_key: LinkKey | None = None,
        fingerprint: str = "",
        span_count: int | None = None,
    ) -> LinkValidity:
        """Derive a link report from an already computed H3 error budget.

        The budget must have been computed for exactly this link and launch
        profile; the caller owns that correspondence (the provider stamps the
        fingerprint).  ``valid`` additionally requires a finite margin, so a
        budget that could not be evaluated can never look valid here.

        Args:
            budget: :func:`~network_nlo_eval.models.uncertainty.estimate_h3_error_budget`
                的结果。
            link_key: 所属链路键。
            fingerprint: 占用谱指纹；网络桥使用**模型坐标**的功率谱，
                这样同一物理链路从不同输入口径进来会得到同一指纹。
            span_count: 跨数；``None`` 时用预算里的 ``validity.span_count``。

        Returns
        -------
            与预算同序的链路有效性。
        """
        report = budget.validity
        margin = np.asarray(budget.margin_db, dtype=np.float64)
        valid = np.asarray(report.valid, dtype=bool) & np.asarray(report.active, dtype=bool) & np.isfinite(margin)
        # 倾斜是整条占用谱的标量分数，广播成逐信道数组；软 margin 只在占用信道上
        # 有意义，未占用信道置 0（它们的 SNR 本来就不参与判定）。
        informational = float(report.tilt_confidence_informational)
        active = np.asarray(report.active, dtype=bool)
        soft = np.where(active, tilt_soft_margin_db(informational), 0.0)
        return cls(
            valid=valid,
            active=active,
            flagged=np.asarray(report.flagged, dtype=bool),
            confidence=np.asarray(report.confidence, dtype=np.float64),
            margin_db=margin,
            tilt_confidence_informational=np.full(active.shape, informational, dtype=np.float64),
            soft_margin_db=soft,
            model=str(budget.model),
            span_count=int(report.span_count if span_count is None else span_count),
            fingerprint=fingerprint,
            link_key=link_key,
            dense_grid_overlap=bool(report.dense_grid_overlap),
            raman_auto_under_resolved=bool(report.raman_auto_under_resolved),
            fwm_required=bool(report.fwm_required),
            fwm_reason=str(report.fwm_reason),
            fwm_validity_mode=str(report.fwm_validity_mode),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (non-finite margins as ``None``)."""
        return {
            "model": self.model,
            "link_key": list(self.link_key) if self.link_key is not None else None,
            "fingerprint": self.fingerprint,
            "span_count": int(self.span_count),
            "valid": [bool(value) for value in self.valid],
            "active": [bool(value) for value in self.active],
            "flagged": [bool(value) for value in self.flagged],
            "confidence": [float(value) for value in self.confidence],
            "margin_db": [float(value) if np.isfinite(value) else None for value in self.margin_db],
            "tilt_confidence_informational": [float(value) for value in self.tilt_confidence_informational],
            "soft_margin_db": [float(value) for value in self.soft_margin_db],
            "dense_grid_overlap": bool(self.dense_grid_overlap),
            "raman_auto_under_resolved": bool(self.raman_auto_under_resolved),
            "fwm_required": bool(self.fwm_required),
            "fwm_reason": str(self.fwm_reason),
            "fwm_validity_mode": str(self.fwm_validity_mode),
        }


@dataclass(frozen=True)
class ChannelValidity:
    """一条业务路径上逐信道的有效性聚合。

    聚合规则是"最坏链路 + margin 相加"：任一链路判无效则信道无效；margin 是
    各链路建议值之和（dB 域的保守加性近似，不与噪声做功率域换算）。
    ``soft_margin_db`` 同样按有效链路相加，只服务 ``mode="soft"``。

    Attributes
    ----------
        valid: ``(N,)`` 该信道在整条路径上是否可用。
        margin_db: ``(N,)`` 路径级加性 margin (dB)。
        soft_margin_db: ``(N,)`` 路径级倾斜软 margin (dB)；无效链路不贡献。
        dense_grid_overlap: 路径上任一链路是否为密集栅格。``reject`` 作用在
            整条路径的可用性上，与逐链路报告的关系同 :class:`LinkValidity`。
        raman_auto_under_resolved: 路径上任一链路是否命中 auto-Raman 分辨率
            不足（逻辑或）。同样只有 ``raman_resolution_action="reject"`` 才会
            影响可用性。
        fwm_required: 路径上任一链路是否报告"模型未覆盖 FWM"（逻辑或）。
            同样只有 ``fwm_action="reject"`` 才会影响可用性。
        fwm_reason: 各链路原因的 ``+`` 连接合并（按出现顺序去重，空串忽略）。
        fwm_validity_mode: 各链路模式的保守合并：任一链路 ``"reject"`` 即
            ``"reject"``，否则 ``"flag"``。
    """

    valid: NDArrayBool
    margin_db: NDArrayFloat
    soft_margin_db: NDArrayFloat | None = None
    dense_grid_overlap: bool = False
    raman_auto_under_resolved: bool = False
    fwm_required: bool = False
    fwm_reason: str = ""
    fwm_validity_mode: str = "flag"

    def __post_init__(self) -> None:
        """Copy and freeze the arrays.

        Raises
        ------
            ValueError: 两个必需数组形状不同或不是一维，可选软 margin 形状
                不匹配或非有限/为负，``fwm_validity_mode`` 不是
                ``"flag"`` / ``"reject"``。
        """
        valid = np.array(self.valid, dtype=bool, copy=True)
        margin = np.array(self.margin_db, dtype=np.float64, copy=True)
        if valid.ndim != 1 or valid.shape != margin.shape:
            raise ValueError(
                f"validity arrays must be one-dimensional and equal, got {valid.shape} and {margin.shape}."
            )
        if np.any(np.isnan(margin)) or np.any(margin < 0.0):
            raise ValueError("margin_db must be non-negative and never NaN.")
        num_channels = int(valid.shape[0])
        raw_soft = self.soft_margin_db
        if raw_soft is None:
            soft = np.zeros(num_channels, dtype=np.float64)
        else:
            soft = np.asarray(raw_soft, dtype=np.float64)
            if soft.ndim == 0:
                soft = np.full(num_channels, float(soft), dtype=np.float64)
            elif soft.shape != (num_channels,):
                raise ValueError(f"soft_margin_db must have shape ({num_channels},), got {soft.shape}.")
            soft = np.array(soft, dtype=np.float64, copy=True)
        if np.any(~np.isfinite(soft)) or np.any(soft < 0.0):
            raise ValueError("soft_margin_db must be finite and non-negative.")
        valid.setflags(write=False)
        margin.setflags(write=False)
        soft.setflags(write=False)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "margin_db", margin)
        object.__setattr__(self, "soft_margin_db", soft)
        object.__setattr__(self, "dense_grid_overlap", bool(self.dense_grid_overlap))
        object.__setattr__(
            self, "raman_auto_under_resolved", bool(self.raman_auto_under_resolved)
        )
        object.__setattr__(self, "fwm_required", bool(self.fwm_required))
        object.__setattr__(self, "fwm_reason", str(self.fwm_reason))
        if self.fwm_validity_mode not in _FWM_VALIDITY_MODES:
            raise ValueError(
                f"fwm_validity_mode must be one of {_FWM_VALIDITY_MODES}, got {self.fwm_validity_mode!r}."
            )
        object.__setattr__(self, "fwm_validity_mode", str(self.fwm_validity_mode))

    @classmethod
    def combine(cls, links: Sequence[LinkValidity], num_channels: int) -> ChannelValidity:
        """Fold per-link reports into one per-channel verdict.

        Args:
            links: 按路径顺序排列的链路报告；空序列表示没有接入有效性。
            num_channels: 信道数。

        Returns
        -------
            聚合后的逐信道有效性。``dense_grid_overlap``、
            ``raman_auto_under_resolved`` 与 ``fwm_required`` 都取各链路的
            逻辑或；``fwm_reason`` 按出现顺序去重合并，``fwm_validity_mode``
            取最保守值。

        Raises
        ------
            ValueError: 某条链路的长度与 ``num_channels`` 不一致。
        """
        valid = np.ones(num_channels, dtype=bool)
        margin = np.zeros(num_channels, dtype=np.float64)
        soft = np.zeros(num_channels, dtype=np.float64)
        dense = False
        raman_under = False
        fwm_required = False
        fwm_reasons: list[str] = []
        fwm_reject = False
        for link in links:
            if link.num_channels != num_channels:
                raise ValueError(
                    f"link {link.link_key} has {link.num_channels} channels, expected {num_channels}."
                )
            valid &= link.valid
            margin += np.where(link.valid, np.where(np.isfinite(link.margin_db), link.margin_db, 0.0), 0.0)
            soft += np.where(link.valid, link.soft_margin_db, 0.0)
            dense = dense or bool(link.dense_grid_overlap)
            raman_under = raman_under or bool(link.raman_auto_under_resolved)
            fwm_required = fwm_required or bool(link.fwm_required)
            reason = str(link.fwm_reason or "")
            for token in reason.split("+") if reason else ():
                if token and token not in fwm_reasons:
                    fwm_reasons.append(token)
            fwm_reject = fwm_reject or link.fwm_validity_mode == FwmAction.REJECT
        return cls(
            valid=valid,
            margin_db=margin,
            soft_margin_db=soft,
            dense_grid_overlap=dense,
            raman_auto_under_resolved=raman_under,
            fwm_required=fwm_required,
            fwm_reason="+".join(fwm_reasons),
            fwm_validity_mode=FwmAction.REJECT if fwm_reject else FwmAction.FLAG,
        )

    def adjusted_snr_db(
        self,
        snr_db: NDArrayFloat,
        policy: LinkValidityPolicy | None = None,
    ) -> NDArrayFloat:
        """Apply the policy to a per-channel SNR vector.

        Args:
            snr_db: ``(N,)`` 物理 SNR (dB)。
            policy: 策略；``None`` 等价于 ``"off"``。

        Returns
        -------
            ``(N,)`` 用于阈值比较的 SNR。

        Raises
        ------
            ValueError: ``snr_db`` 形状与聚合结果不一致。
        """
        values = np.asarray(snr_db, dtype=np.float64)
        if values.shape != self.valid.shape:
            raise ValueError(f"snr_db shape {values.shape} does not match the verdict {self.valid.shape}.")
        return (policy or LinkValidityPolicy()).apply(
            values,
            self.valid,
            self.margin_db,
            self.soft_margin_db,
            dense_grid_overlap=self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )

    def effective_valid(self, policy: LinkValidityPolicy | None = None) -> NDArrayBool:
        """该路径在给定策略下的可用性（``"reject"`` 动作扣对应标志）。"""
        return (policy or LinkValidityPolicy()).effective_valid(
            self.valid,
            self.dense_grid_overlap,
            raman_auto_under_resolved=self.raman_auto_under_resolved,
            fwm_required=self.fwm_required,
        )


class LinkValidityProvider(Protocol):
    """A callable that produces the validity report of one link.

    The signature mirrors the information :class:`QoTValidator` already has in
    ``_calc_link_noise``; a provider must be a pure function of its arguments
    plus its own caches.
    """

    def __call__(
        self,
        link_key: LinkKey,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_profile_w: NDArrayFloat,
        span_length_m: float,
    ) -> LinkValidity:
        """Return the validity report of one link at one occupied spectrum."""
        ...


@dataclass
class LinkValidityCache:
    """Bounded LRU cache keyed by ``(link_key, spectrum fingerprint, span length)``.

    The cache never keys on ``link_key`` alone: the recommendation margin depends
    on the occupied spectrum (pump mode, band class, tilt), so an entry is only
    valid for the exact power profile that produced it.

    Attributes
    ----------
        max_entries: 条目上限；超出时淘汰最久未用的一条。
        hits: 命中次数。
        misses: 未命中次数。
    """

    max_entries: int = 64
    hits: int = 0
    misses: int = 0
    _entries: OrderedDict[tuple[LinkKey, str, int], LinkValidity] = field(
        default_factory=OrderedDict, repr=False
    )

    def __post_init__(self) -> None:
        """Reject a non-positive capacity.

        Raises
        ------
            ValueError: ``max_entries < 1``。
        """
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}.")

    def __len__(self) -> int:
        """Return the number of cached links."""
        return len(self._entries)

    @staticmethod
    def key(link_key: LinkKey, power_profile_w: NDArrayFloat, span_length_m: float) -> tuple[LinkKey, str, int]:
        """Build the cache key of one link state.

        Args:
            link_key: 规范化链路键。
            power_profile_w: ``(N,)`` 逐信道入纤功率 (W)。
            span_length_m: 跨长 (m)。

        Returns
        -------
            ``(link_key, 指纹, 跨长 µm)``。
        """
        return (link_key, occupancy_fingerprint(power_profile_w, span_length_m), int(round(span_length_m * 1e6)))

    def get(self, link_key: LinkKey, power_profile_w: NDArrayFloat, span_length_m: float) -> LinkValidity | None:
        """Look up a link state, updating the hit/miss counters.

        Args:
            link_key: 规范化链路键。
            power_profile_w: ``(N,)`` 逐信道入纤功率 (W)。
            span_length_m: 跨长 (m)。

        Returns
        -------
            命中的报告；未命中为 ``None``。
        """
        key = self.key(link_key, power_profile_w, span_length_m)
        cached = self._entries.get(key)
        if cached is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return cached

    def put(
        self,
        link_key: LinkKey,
        power_profile_w: NDArrayFloat,
        span_length_m: float,
        validity: LinkValidity,
    ) -> None:
        """Store one report, evicting the least recently used entry when full.

        Args:
            link_key: 规范化链路键。
            power_profile_w: ``(N,)`` 逐信道入纤功率 (W)。
            span_length_m: 跨长 (m)。
            validity: 该状态的报告。
        """
        key = self.key(link_key, power_profile_w, span_length_m)
        self._entries[key] = validity
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def invalidate(self, link_key: LinkKey) -> int:
        """Drop every entry of one link, whatever its spectrum.

        Args:
            link_key: 规范化链路键。

        Returns
        -------
            被删除的条目数。
        """
        keys = [key for key in self._entries if key[0] == link_key]
        for key in keys:
            del self._entries[key]
        return len(keys)

    def clear(self) -> None:
        """Drop all entries and reset the counters."""
        self._entries.clear()
        self.hits = 0
        self.misses = 0


def link_optical_path(
    fiber_config: FiberSpanConfig,
    edfa_config: EDFAConfig | None,
    span_length_m: float,
) -> MultiSpanOpticalPath:
    """Build the one-link optical path used to evaluate a link's H3 field.

    Args:
        fiber_config: 链路光纤参数（``length_km`` 是整条链路长度）。
        edfa_config: 跨段 EDFA 配置；``None`` 时用默认值。
        span_length_m: 跨长 (m)。

    Returns
    -------
        单跳光路；路由层按 ``span_length_m`` 自行切分整跨与余段。

    Raises
    ------
        ValueError: 跨长或链路长度非正。
    """
    span_km = float(span_length_m) / 1e3
    if span_km <= 0.0:
        raise ValueError(f"span_length_m must be positive, got {span_length_m}.")
    if float(fiber_config.length_km) <= 0.0:
        raise ValueError(f"fiber length must be positive, got {fiber_config.length_km}.")
    fiber = fiber_config.model_copy(update={"max_span_length_km": span_km})
    default_edfa = EDFAConfig(target_gain_db=None, noise_figure_db=5.0, gain_ripple_db=None)
    return MultiSpanOpticalPath([fiber], [edfa_config or default_edfa])


class H3LinkValidityProvider:
    """Evaluate the H3 field of one link and turn it into a link report.

    This is the network-side bridge: it runs the existing ``evaluate_path`` plus
    :func:`estimate_h3_error_budget` for exactly one link and caches the result
    at ``(link, occupied spectrum, span length)`` granularity.  It never calls
    ``numerical_dra`` and never changes the point SNR.

    ``power_profile_w`` is declared by ``input_power_convention`` (default
    ``"total"``, the network reference plane).  It is converted **once** to the
    model's ``launch_power_convention`` before the H3 evaluation, and the cache
    key / fingerprint use that model-coordinate power, so the same physical
    link hits the same entry regardless of the caller's coordinate.

    Args:
        model: H3 模型（``pcfm`` / ``pcfm_h3_raman``），必须暴露 ``evaluate_path``、
            ``kernel_for`` 与 ``n_nodes``。
        input_power_convention: ``power_profile_w`` 的功率口径，``"total"``
            （默认）或 ``"per_pol"``。
        config: 可选误差预算配置。
        cache: 可选缓存；默认新建 :class:`LinkValidityCache`。

    Raises
    ------
        ValueError: ``input_power_convention`` 非法。
    """

    def __init__(
        self,
        model: QoTModel,
        *,
        input_power_convention: str = "total",
        config: H3UncertaintyConfig | None = None,
        cache: LinkValidityCache | None = None,
    ) -> None:
        self.model = model
        self.input_power_convention = resolve_launch_power_convention(input_power_convention)
        self.config = config or H3UncertaintyConfig()
        self.cache = cache if cache is not None else LinkValidityCache()
        self._warned_convention_mismatch = False

    @property
    def model_power_convention(self) -> str:
        """模型入口口径；不声明该属性的旧模型按 ``"total"`` 处理。"""
        return str(getattr(self.model, "launch_power_convention", "total"))

    def _warn_once_if_mismatched(self) -> None:
        """口径不同时每个 provider 实例只提示一次。"""
        if self._warned_convention_mismatch:
            return
        source = self.input_power_convention
        target = self.model_power_convention
        if source == target:
            return
        factor = 0.5 if target == "per_pol" else 2.0
        direction = "total -> per_pol" if target == "per_pol" else "per_pol -> total"
        warnings.warn(
            f"h3 validity provider: caller input_power_convention={source!r} differs from the model "
            f"launch_power_convention={target!r}; the launch power is scaled by {factor:g} "
            f"({direction}) before the H3 evaluation and used as the cache key.",
            UserWarning,
            stacklevel=2,
        )
        self._warned_convention_mismatch = True

    def __call__(
        self,
        link_key: LinkKey,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_profile_w: NDArrayFloat,
        span_length_m: float,
    ) -> LinkValidity:
        """Return the validity report of one link at one occupied spectrum.

        Args:
            link_key: 规范化链路键。
            fiber_config: 链路光纤参数。
            edfa_config: 跨段 EDFA 配置。
            power_profile_w: ``(N,)`` 逐信道入纤功率 (W)，口径为构造时的
                ``input_power_convention``。
            span_length_m: 跨长 (m)。

        Returns
        -------
            该链路在该占用谱上的逐信道有效性。

        Raises
        ------
            ValueError: 功率谱长度与模型节点数不一致。
        """
        power = np.asarray(power_profile_w, dtype=np.float64)
        if power.shape != (self.model.n_nodes,):
            raise ValueError(
                f"power_profile_w must have shape ({self.model.n_nodes},), got {power.shape}."
            )
        self._warn_once_if_mismatched()
        power_model = to_convention(power, self.input_power_convention, self.model_power_convention)
        cached = self.cache.get(link_key, power_model, span_length_m)
        if cached is not None:
            return cached

        path = link_optical_path(fiber_config, edfa_config, span_length_m)
        result = self.model.evaluate_path(path, power_model)
        budget = estimate_h3_error_budget(self.model, path, power_model, result, config=self.config)
        validity = LinkValidity.from_budget(
            budget,
            link_key=link_key,
            fingerprint=occupancy_fingerprint(power_model, span_length_m),
            span_count=result.span_count,
        )
        self.cache.put(link_key, power_model, span_length_m, validity)
        return validity
