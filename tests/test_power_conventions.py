"""共享入纤功率口径模块的单元测试（任务 O1）。

守住四条契约：

1. ``"total"`` / ``"per_pol"`` 的线性换算（0.5 / 2.0）；
2. 相同口径**直接返回原对象**，不做复制、不做乘法（默认路径逐位不变的根基）；
3. 非法口径在两端都明确报错，不静默降级；
4. ASE 偏振缩放与功率换算是同一个 0.5，且 ``raman_ode`` 的公开 import 路径仍可用。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.models.power_conventions import (
    LAUNCH_POWER_CONVENTIONS,
    ase_polarization_scale,
    launch_power_scale,
    resolve_launch_power_convention,
    to_convention,
)


def test_supported_conventions_are_total_and_per_pol() -> None:
    """口径只有 ``total`` 与 ``per_pol``。"""
    assert LAUNCH_POWER_CONVENTIONS == ("total", "per_pol")


@pytest.mark.parametrize("convention", ["total", "per_pol"])
def test_resolve_accepts_both_conventions(convention: str) -> None:
    """两种合法口径都按原名返回。"""
    assert resolve_launch_power_convention(convention) == convention


def test_resolve_skips_the_polarization_check_when_it_is_not_given() -> None:
    """不传 ``signal_polarizations`` 时只校验口径名。"""
    """网络层只知道口径名，不需要模型的偏振状态数。"""
    assert resolve_launch_power_convention("per_pol") == "per_pol"
    assert resolve_launch_power_convention("per_pol", 2) == "per_pol"


def test_resolve_rejects_per_pol_with_a_single_polarization_state() -> None:
    """``per_pol`` 声明每偏振输入，单偏振状态必须报错。"""
    with pytest.raises(ValueError, match="signal_polarizations must be 2"):
        resolve_launch_power_convention("per_pol", 1)


def test_resolve_keeps_the_legacy_total_with_two_polarizations_combination() -> None:
    """``total``+2 是历史对照组合，求解器层继续接受。"""
    """``total``+2 是历史对照口径，求解器层仍接受（模型层负责 warning）。"""
    assert resolve_launch_power_convention("total", 2) == "total"


@pytest.mark.parametrize("convention", ["per-pal", "perpol", "", "TOTAL", "Total"])
def test_resolve_rejects_unknown_conventions(convention: str) -> None:
    """拼写错误不得静默降级为某个默认口径。"""
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        resolve_launch_power_convention(convention)


def test_launch_power_scale_is_one_and_a_half() -> None:
    """``P_model/P_total`` 对两种口径分别是 1.0 / 0.5。"""
    assert launch_power_scale("total") == 1.0
    assert launch_power_scale("per_pol") == 0.5


def test_ase_polarization_scale_matches_the_power_scale() -> None:
    """新产生 ASE 的缩放与功率坐标缩放一致。"""
    assert ase_polarization_scale("total") == 1.0
    assert ase_polarization_scale("per_pol") == 0.5


def test_to_convention_scales_total_down_to_per_pol() -> None:
    """``total`` 输入换算到 ``per_pol`` 恰好乘 0.5。"""
    power = np.array([1e-3, 2e-3, 0.0])
    converted = to_convention(power, "total", "per_pol")
    assert np.array_equal(converted, power * 0.5)
    assert converted is not power


def test_to_convention_scales_per_pol_up_to_total() -> None:
    """``per_pol`` 输入换算到 ``total`` 恰好乘 2。"""
    power = np.array([0.5e-3, 1e-3, 0.0])
    converted = to_convention(power, "per_pol", "total")
    assert np.array_equal(converted, power * 2.0)
    assert converted is not power


def test_to_convention_round_trips() -> None:
    """一次下换算加一次上换算回到原值。"""
    power = np.array([1.25e-3, 3.5e-3])
    restored = to_convention(to_convention(power, "total", "per_pol"), "per_pol", "total")
    assert np.array_equal(restored, power)


@pytest.mark.parametrize("convention", ["total", "per_pol"])
def test_to_convention_returns_the_same_object_when_conventions_match(convention: str) -> None:
    """相同口径直接返回原数组对象，不做复制。"""
    power = np.array([1e-3, 2e-3])
    assert to_convention(power, convention, convention) is power


def test_to_convention_returns_the_same_scalar_object_when_conventions_match() -> None:
    """标量输入在相同口径下也返回原对象。"""
    value = 1e-3
    assert to_convention(value, "total", "total") is value


def test_to_convention_validates_both_endpoints() -> None:
    """源口径与目标口径都做校验。"""
    power = np.array([1e-3])
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        to_convention(power, "per-pal", "total")
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        to_convention(power, "total", "per-pal")


def test_raman_ode_reexports_the_shared_definitions() -> None:
    """旧的 ``raman_ode`` import 路径与对象身份保持不变。"""
    from network_nlo_eval.models import raman_ode

    assert raman_ode.LAUNCH_POWER_CONVENTIONS is LAUNCH_POWER_CONVENTIONS
    assert raman_ode.resolve_launch_power_convention is resolve_launch_power_convention


def test_raman_ode_keeps_the_legacy_error_messages() -> None:
    """重导出后旧错误消息与校验组合保持不变。"""
    from network_nlo_eval.models.raman_ode import resolve_launch_power_convention as legacy

    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        legacy("total ", 1)
    with pytest.raises(ValueError, match="signal_polarizations must be 2"):
        legacy("per_pol", 1)
