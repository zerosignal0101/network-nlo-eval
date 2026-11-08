"""H3 网络桥的入纤功率口径测试（任务 O1）。

网络/RWA 参考面是 ``P_total``；H3 模型可以声明 ``per_pol``。桥接层负责在
模型边界做**唯一一次**换算：

* 默认 ``input_power_convention="total"`` + 默认 ``total`` 模型：与旧调用逐位一致；
* ``per_pol`` 模型 + ``total`` 输入：内部按 ``P/2`` 交给模型，缓存/指纹也用模型坐标；
* ``per_pol`` 模型 + ``per_pol`` 输入：不再换算；
* 口径不匹配时每个 provider 实例只发一次 ``UserWarning``；
* 同一物理链路从两种输入口径进来命中同一个缓存键。
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.types import LinkKey
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
from network_nlo_eval.rwa.validity import (
    H3LinkValidityProvider,
    LinkValidity,
    LinkValidityCache,
    link_optical_path,
    occupancy_fingerprint,
)

CASE = build_case("scaling_40ch")
LINK_KEY: LinkKey = (0, 1)
SPAN_LENGTH_M = 100e3

_REPORT_FIELDS = (
    "valid",
    "active",
    "flagged",
    "confidence",
    "margin_db",
    "tilt_confidence_informational",
    "soft_margin_db",
)


def _per_pol_model():
    """自洽的端到端 per_pol H3 模型（无泵，不会触发倍率 warning）。"""
    return make_model(
        "pcfm_h3_raman",
        CASE,
        signal_polarizations=2,
        launch_power_convention="per_pol",
    )


def _legacy_link_validity(model, fiber, power: np.ndarray) -> LinkValidity:
    """改动前的 provider 路径：直接把输入功率交给模型。"""
    path = link_optical_path(fiber, None, SPAN_LENGTH_M)
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)
    return LinkValidity.from_budget(
        budget,
        link_key=LINK_KEY,
        fingerprint=occupancy_fingerprint(power, SPAN_LENGTH_M),
        span_count=result.span_count,
    )


def _assert_reports_equal(actual: LinkValidity, expected: LinkValidity) -> None:
    for name in _REPORT_FIELDS:
        assert np.array_equal(np.asarray(getattr(actual, name)), np.asarray(getattr(expected, name))), name
    assert actual.fingerprint == expected.fingerprint
    assert actual.span_count == expected.span_count


# ---------------------------------------------------------------------------
# 默认不变
# ---------------------------------------------------------------------------


def test_provider_default_total_is_bit_identical_to_the_legacy_call() -> None:
    """默认构造的参数与旧调用路径逐位一致（默认路径硬门槛）。"""
    model = make_model("pcfm_h3_raman", CASE)
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE)
    provider = H3LinkValidityProvider(model)

    report = provider(LINK_KEY, fiber, None, power, SPAN_LENGTH_M)
    expected = _legacy_link_validity(model, fiber, power)
    _assert_reports_equal(report, expected)
    assert provider.model_power_convention == "total"
    assert provider.cache.misses == 1


def test_explicit_total_input_is_identical_to_the_default() -> None:
    """显式 ``total`` 输入与默认构造逐位一致。"""
    model = make_model("pcfm_h3_raman", CASE)
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE)
    default = H3LinkValidityProvider(model)(LINK_KEY, fiber, None, power, SPAN_LENGTH_M)
    explicit = H3LinkValidityProvider(model, input_power_convention="total")(
        LINK_KEY, fiber, None, power, SPAN_LENGTH_M
    )
    _assert_reports_equal(default, explicit)


# ---------------------------------------------------------------------------
# 换算
# ---------------------------------------------------------------------------


def test_per_pol_model_with_total_input_converts_to_the_model_power() -> None:
    """``total`` 输入交给 per_pol 模型前恰好乘 0.5，缓存/指纹用模型坐标。"""
    model = _per_pol_model()
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE)
    provider = H3LinkValidityProvider(model)

    with pytest.warns(UserWarning, match="total -> per_pol"):
        report = provider(LINK_KEY, fiber, None, power, SPAN_LENGTH_M)

    half = power * 0.5
    expected = _legacy_link_validity(model, fiber, half)
    _assert_reports_equal(report, expected)
    assert report.fingerprint == occupancy_fingerprint(half, SPAN_LENGTH_M)
    assert report.fingerprint != occupancy_fingerprint(power, SPAN_LENGTH_M)
    # 真的换算过：未换算时报告会对应 total 功率。
    unconverted = _legacy_link_validity(model, fiber, power)
    assert report.fingerprint != unconverted.fingerprint


def test_per_pol_model_with_per_pol_input_does_not_convert() -> None:
    """输入口径与模型一致时不产生任何算术，也不 warning。"""
    model = _per_pol_model()
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE) * 0.5
    provider = H3LinkValidityProvider(model, input_power_convention="per_pol")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = provider(LINK_KEY, fiber, None, power, SPAN_LENGTH_M)

    assert [item for item in caught if issubclass(item.category, UserWarning)] == []
    expected = _legacy_link_validity(model, fiber, power)
    _assert_reports_equal(report, expected)
    assert report.fingerprint == occupancy_fingerprint(power, SPAN_LENGTH_M)


def test_both_input_conventions_hit_the_same_model_coordinate_cache() -> None:
    """同一物理链路：total 输入与 per_pol 输入共享模型坐标缓存键。"""
    model = _per_pol_model()
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE)
    cache = LinkValidityCache()
    from_total = H3LinkValidityProvider(model, cache=cache)
    from_per_pol = H3LinkValidityProvider(model, input_power_convention="per_pol", cache=cache)

    with pytest.warns(UserWarning, match="total -> per_pol"):
        first = from_total(LINK_KEY, fiber, None, power, SPAN_LENGTH_M)
    second = from_per_pol(LINK_KEY, fiber, None, power * 0.5, SPAN_LENGTH_M)

    assert cache.misses == 1
    assert cache.hits == 1
    assert len(cache) == 1
    assert second is first
    _assert_reports_equal(second, first)


# ---------------------------------------------------------------------------
# warning / 非法口径
# ---------------------------------------------------------------------------


def test_convention_mismatch_warns_only_once_per_provider() -> None:
    """缓存命中与重复调用都不能重复刷 warning。"""
    model = _per_pol_model()
    fiber = build_path(CASE).fiber_configs[0]
    power = launch_profile(CASE)
    provider = H3LinkValidityProvider(model)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        provider(LINK_KEY, fiber, None, power, SPAN_LENGTH_M)
        provider(LINK_KEY, fiber, None, power, SPAN_LENGTH_M)
        provider((1, 2), fiber, None, power, SPAN_LENGTH_M)

    mismatched = [item for item in caught if issubclass(item.category, UserWarning)]
    assert len(mismatched) == 1


def test_matching_conventions_never_warn() -> None:
    """口径一致时不应产生任何 warning。"""
    model = make_model("pcfm_h3_raman", CASE)
    fiber = build_path(CASE).fiber_configs[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H3LinkValidityProvider(model)(LINK_KEY, fiber, None, launch_profile(CASE), SPAN_LENGTH_M)
    assert [item for item in caught if issubclass(item.category, UserWarning)] == []


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL"])
def test_invalid_input_power_convention_is_rejected(convention: str) -> None:
    """构造期就拒绝非法输入口径。"""
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        H3LinkValidityProvider(make_model("pcfm_h3_raman", CASE), input_power_convention=convention)
