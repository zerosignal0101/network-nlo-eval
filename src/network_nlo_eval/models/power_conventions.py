"""入纤功率口径的共享定义与换算。

网络/RWA 的参考面永远是**每信道双偏振总功率** ``P_total`` (W)：
``NetworkState.launch_power_profile_w``、``ServiceRequest.launch_power_w`` 与
``LinkNoiseCache`` 都是这个坐标。模型可以声明自己的入口口径：

* ``"total"``：``power_in_w`` 就是 ``P_total``；
* ``"per_pol"``：``power_in_w`` 是每偏振功率，物理总功率是它的 2 倍。

同一物理链路在两个坐标下只差一个因子 2，SNR = S/(NLI+ASE) 与坐标无关。
换算必须集中在模型边界做**唯一一次**，绝不能在网络层与模型层各减半一次。

本模块是叶子模块（只依赖标准库），避免 ``pcfm`` 与 ``raman_ode`` 之间的循环
导入；:mod:`network_nlo_eval.models.raman_ode` 从本模块导入并重导出
``LAUNCH_POWER_CONVENTIONS`` / ``resolve_launch_power_convention``，保持既有
公开 import 路径不变。
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "LAUNCH_POWER_CONVENTIONS",
    "ase_polarization_scale",
    "launch_power_scale",
    "resolve_launch_power_convention",
    "to_convention",
]

#: 支持的入纤功率口径。
#:
#: ``"total"``：``power_in_w`` 是每信道（双偏振）总功率，历史默认；
#: ``"per_pol"``：``power_in_w`` 是每偏振功率，物理总功率是它的 2 倍。
LAUNCH_POWER_CONVENTIONS: tuple[str, ...] = ("total", "per_pol")


def resolve_launch_power_convention(
    launch_power_convention: str,
    signal_polarizations: int | None = None,
) -> str:
    """校验入纤功率口径（以及与状态偏振数的组合）。

    ``signal_polarizations`` 只放大耦合项的**驱动功率**，不放大状态本身。因此
    只有两种自洽的声明：

    * ``"total"`` + ``signal_polarizations=1``：状态就是每信道总功率，驱动也
      用总功率，光子预算比值 1。这是历史默认。
    * ``"per_pol"`` + ``signal_polarizations=2``：状态是每偏振功率，驱动取
      ``2 * P`` 的双偏振总功率，物理总功率 = 2 × 输入。

    ``"total"`` + ``signal_polarizations=2`` 是把总功率再乘 2 驱动，物理上
    重复计数；该组合在求解器层保留（历史对照与倍率探针），但调用方必须显式
    warning，见 :class:`~network_nlo_eval.models.pcfm_h3_raman.PCFH3RamanModel`。
    ``"per_pol"`` + ``signal_polarizations=1`` 自相矛盾（声明每偏振却没有第二
    个偏振），直接报错。

    网络/QoT 层只需要校验口径名，不需要知道模型的偏振状态数，因此
    ``signal_polarizations`` 默认 ``None``（跳过组合校验）。

    Args:
        launch_power_convention: ``"total"`` 或 ``"per_pol"``。
        signal_polarizations: ODE 信号状态代表的偏振数；``None`` 表示不做
            组合校验（仅网络层使用）。

    Returns
    -------
        规范化后的口径名。

    Raises
    ------
        ValueError: 口径非法，或 ``"per_pol"`` 与 ``signal_polarizations`` 不匹配。
    """
    convention = str(launch_power_convention)
    if convention not in LAUNCH_POWER_CONVENTIONS:
        raise ValueError(
            f"launch_power_convention must be one of {LAUNCH_POWER_CONVENTIONS}, got {launch_power_convention!r}."
        )
    if (
        convention == "per_pol"
        and signal_polarizations is not None
        and int(signal_polarizations) != 2
    ):
        raise ValueError(
            "launch_power_convention='per_pol' declares a per-polarization input, so the ODE state "
            "must be per-polarization: signal_polarizations must be 2, got "
            f"{int(signal_polarizations)}. Use launch_power_convention='total' for a "
            "per-channel total-power input."
        )
    return convention


def launch_power_scale(convention: str) -> float:
    """该口径下模型功率相对物理总功率的倍数 ``P_model / P_total``。

    Args:
        convention: ``"total"`` 或 ``"per_pol"``。

    Returns
    -------
        ``"total"`` 为 ``1.0``，``"per_pol"`` 为 ``0.5``。

    Raises
    ------
        ValueError: 口径非法。
    """
    return 0.5 if resolve_launch_power_convention(convention) == "per_pol" else 1.0


def ase_polarization_scale(convention: str) -> float:
    """本跨**新产生** ASE 相对双偏振总量的缩放。

    ``"per_pol"`` 模型输出每偏振信号与 NLI，因此共享 route 层产生的双偏振
    EDFA/ROADM booster ASE 必须恰好减半一次；``"total"`` 保持 ``1.0``。
    已经处于模型坐标的 ``input_ase_w``（例如 Raman ASE）**不**再缩放。

    Args:
        convention: ``"total"`` 或 ``"per_pol"``。

    Returns
    -------
        ``"total"`` 为 ``1.0``，``"per_pol"`` 为 ``0.5``。

    Raises
    ------
        ValueError: 口径非法。
    """
    return launch_power_scale(convention)


def to_convention(power_w: Any, from_convention: str, to_convention: str) -> Any:
    """把功率从一种口径换算到另一种口径。

    相同口径时**直接返回原对象**，不做复制、不做乘法，保证默认 ``total``
    路径逐位不变；不同口径时只做一次 ``0.5`` / ``2.0`` 乘法。

    Args:
        power_w: 功率数组（``numpy.ndarray``）或标量。
        from_convention: 输入的功率口径。
        to_convention: 目标功率口径。

    Returns
    -------
        换算后的功率；``from_convention == to_convention`` 时是 ``power_w``
        本身（同一对象）。

    Raises
    ------
        ValueError: 任一端口径非法。
    """
    source = resolve_launch_power_convention(from_convention)
    target = resolve_launch_power_convention(to_convention)
    if source == target:
        return power_w
    # 目标坐标相对源坐标的比值：total -> per_pol 是 0.5，反向是 2.0。
    return power_w * (launch_power_scale(target) / launch_power_scale(source))
