"""两级 QoT 的入纤功率口径测试（任务 O1）。

``TwoTierEvaluator`` 自己声明 ``launch_power_convention``（默认 ``total``），
H3 与数值回退各自换算到自己的模型口径：

* 默认 total + total H3 + total fallback：不产生任何算术，逐位不变；
* per_pol H3 + total fallback：H3 收 ``P/2``，fallback 原样收 ``P``；
* per_pol 输入 + per_pol H3：不再换算；
* 回退模型自己的口径与输入不同时，只对回退侧换算。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, grid_for, launch_profile
from network_nlo_eval.models.base import MultiSpanOpticalPath, PathQoTResult, create_model
from network_nlo_eval.network.elements import EDFAConfig
from network_nlo_eval.rwa.two_tier import TwoTierConfig, TwoTierEvaluator

CASE = build_case("scaling_40ch")
_FIELD_ARRAYS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


class _RecordingFallback:
    """记录收到的功率并返回固定 SNR 的假数值模型。"""

    def __init__(self, convention: str = "total", snr_db: float = 42.0) -> None:
        self.launch_power_convention = convention
        self.received: np.ndarray | None = None
        self.calls = 0
        self._snr_db = float(snr_db)

    def evaluate_path(self, optical_path, power_in_w) -> PathQoTResult:
        self.calls += 1
        self.received = np.array(power_in_w, dtype=np.float64, copy=True)
        n = self.received.shape[0]
        zeros = np.zeros(n, dtype=np.float64)
        return PathQoTResult(
            signal_power_w=zeros,
            spm_noise_w=zeros,
            xpm_noise_w=zeros,
            ase_noise_w=zeros,
            snr_db=np.full(n, self._snr_db, dtype=np.float64),
            span_count=1,
            roadm_count=0,
        )


def _edge_case_model(**options):
    """低色散算例：整条梳都会被 H3 flag，确保回退真的运行。"""
    fiber = (
        build_path(CASE)
        .fiber_configs[0]
        .model_copy(update={"dispersion_parameter_d": 1.0, "use_attenuation_polynomial": False})
    )
    model = create_model("pcfm_h3_raman", grid_for(CASE), fiber, symbol_rate_hz=CASE.symbol_rate_hz, **options)
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=CASE.edfa_noise_figure_db)])
    return model, path


def _per_pol_options() -> dict[str, object]:
    return {"signal_polarizations": 2, "launch_power_convention": "per_pol"}


def _enabled(fallback) -> tuple:
    return (
        lambda: fallback,
        TwoTierConfig(enabled=True, max_fallback_channels=CASE.num_channels),
    )


# ---------------------------------------------------------------------------
# 默认不变
# ---------------------------------------------------------------------------


def test_default_total_is_bit_identical_to_the_legacy_two_tier() -> None:
    """默认构造 + 显式 total 与直接调用 H3 逐位一致（不产生算术）。"""
    model, path = _edge_case_model()
    power = launch_profile(CASE)
    default = TwoTierEvaluator(model).evaluate(path, power)
    explicit = TwoTierEvaluator(model, launch_power_convention="total").evaluate(path, power)

    for name in _FIELD_ARRAYS:
        assert np.array_equal(
            np.asarray(getattr(default.field, name)), np.asarray(getattr(explicit.field, name))
        ), name
    assert np.array_equal(default.effective_snr_db, explicit.effective_snr_db)
    assert np.array_equal(default.fallback_snr_db, explicit.fallback_snr_db, equal_nan=True)

    direct = model.evaluate_path(path, power)
    for name in _FIELD_ARRAYS:
        assert np.array_equal(
            np.asarray(getattr(default.field, name)), np.asarray(getattr(direct, name))
        ), name
    assert np.array_equal(default.effective_snr_db, default.budget.adjusted_snr_db(direct.snr_db))


def test_default_total_never_scales_the_fallback() -> None:
    """默认路径把原数组直接交给 fallback，不做乘法。"""
    model, path = _edge_case_model()
    power = launch_profile(CASE)
    fallback = _RecordingFallback("total")
    factory, config = _enabled(fallback)
    outcome = TwoTierEvaluator(model, factory, config=config).evaluate(path, power)
    assert outcome.fallback_ran is True
    assert np.array_equal(fallback.received, power)


# ---------------------------------------------------------------------------
# 两侧各自换算
# ---------------------------------------------------------------------------


def test_per_pol_h3_and_total_fallback_receive_their_own_power() -> None:
    """Total 输入：H3 收 ``P/2``，total fallback 原样收 ``P``。"""
    model, path = _edge_case_model(**_per_pol_options())
    power = launch_profile(CASE)
    fallback = _RecordingFallback("total")
    factory, config = _enabled(fallback)

    outcome = TwoTierEvaluator(model, factory, config=config).evaluate(path, power)

    assert outcome.fallback_ran is True
    expected_h3 = model.evaluate_path(path, power * 0.5)
    for name in _FIELD_ARRAYS:
        assert np.array_equal(
            np.asarray(getattr(outcome.field, name)), np.asarray(getattr(expected_h3, name))
        ), name
    assert np.array_equal(fallback.received, power)


def test_per_pol_input_and_per_pol_h3_do_not_convert() -> None:
    """输入口径与 H3 一致时不再换算；fallback 也按自己的 per_pol 口径收同一数组。"""
    model, path = _edge_case_model(**_per_pol_options())
    power = launch_profile(CASE) * 0.5
    fallback = _RecordingFallback("per_pol")
    factory, config = _enabled(fallback)

    outcome = TwoTierEvaluator(
        model, factory, config=config, launch_power_convention="per_pol"
    ).evaluate(path, power)

    assert outcome.fallback_ran is True
    expected_h3 = model.evaluate_path(path, power)
    for name in _FIELD_ARRAYS:
        assert np.array_equal(
            np.asarray(getattr(outcome.field, name)), np.asarray(getattr(expected_h3, name))
        ), name
    assert np.array_equal(fallback.received, power)


def test_fallback_receives_its_own_convention_under_a_total_input() -> None:
    """Total 输入 + total H3 + per_pol fallback：只有回退侧乘 0.5。"""
    model, path = _edge_case_model()
    power = launch_profile(CASE)
    fallback = _RecordingFallback("per_pol")
    factory, config = _enabled(fallback)

    outcome = TwoTierEvaluator(model, factory, config=config).evaluate(path, power)

    assert outcome.fallback_ran is True
    expected_h3 = model.evaluate_path(path, power)
    for name in _FIELD_ARRAYS:
        assert np.array_equal(
            np.asarray(getattr(outcome.field, name)), np.asarray(getattr(expected_h3, name))
        ), name
    assert np.array_equal(fallback.received, power * 0.5)


def test_effective_snr_uses_the_converted_h3_field() -> None:
    """换算只作用于模型边界，判定仍用同一物理链路的 SNR。"""
    model, path = _edge_case_model(**_per_pol_options())
    power = launch_profile(CASE)
    fallback = _RecordingFallback("total", snr_db=10.0)
    factory, config = _enabled(fallback)

    outcome = TwoTierEvaluator(model, factory, config=config).evaluate(path, power)
    assert np.all(outcome.effective_snr_db[outcome.fallback_channels] == 10.0)


# ---------------------------------------------------------------------------
# 非法口径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL"])
def test_invalid_launch_power_convention_is_rejected(convention: str) -> None:
    """构造期就拒绝非法输入口径。"""
    model, _ = _edge_case_model()
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        TwoTierEvaluator(model, launch_power_convention=convention)
