"""任务 AU：C 密集栅格 v4 复核与按间距 dense-grid penalty schedule。

本文件钉住四件事：

* 默认 ``dense_grid_penalty_schedule=None`` 时，预算与引入 schedule 之前
  逐位相同（标量路径完全不变）；
* schedule 按 ``validity.spacing_to_bandwidth`` 查表：50 / 37.5 / 25 GHz
  分别取 0.6 / 2.0 / 3.6 dB，超出最后一级时取最后一级并写 ``clamped``；
* 非法 schedule（空、upper 非递增、负 penalty、NaN/inf）显式报错；
* schedule 优先级高于标量 penalty，且预算 term/note 明确写
  "calibrated dense-grid penalty schedule"，不伪装成模型修正。

另外复核 v4 代表点：``dense_grid_fix=True`` 的 max|Δ| 必须优于默认，
且标定 penalty 覆盖修复后的实测残差；100 GHz 非密集栅格完全不受影响。

参考一律是 ``numerical``（ISRS-GN），不读 ``results/``，也不跑 661 点矩阵。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.uncertainty import (
    DENSE_GRID_PENALTY_SCHEDULE_V4,
    H3UncertaintyConfig,
    dense_grid_schedule_penalty,
    estimate_h3_error_budget,
)

_TERM_FIELDS = ("name", "kind", "source", "calibrated")


def _case(spacing_hz: float, span_count: int = 1) -> BenchCase:
    """构造与精度矩阵同源的 C 波段算例（64 GBaud、100 km/跨、NF 5.5 dB）。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"au_dense_{spacing_hz / 1e9:g}_{span_count}",
        bands="C",
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=span_count,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


def _evaluate(spacing_hz: float, *, dense_grid_fix: bool = False, span_count: int = 1, power_dbm: float = 6.0):
    """返回 ``(model, path, power, result)``。"""
    case = _case(spacing_hz, span_count)
    path = build_path(case)
    power = launch_profile(case, power_dbm)
    model = make_model("pcfm_h3_raman", case, dense_grid_fix=dense_grid_fix)
    return model, path, power, model.evaluate_path(path, power)


def _budget(spacing_hz: float, *, fix: bool = False, **config: object):
    """构造一个点的预算与点值结果。"""
    model, path, power, result = _evaluate(spacing_hz, dense_grid_fix=fix)
    return model, result, estimate_h3_error_budget(model, path, power, result, config=H3UncertaintyConfig(**config))


def _max_abs_error(result, reference) -> float:
    """逐信道最大 ``|H3 - numerical|`` SNR 误差 (dB)。"""
    return float(np.max(np.abs(np.asarray(result.snr_db) - np.asarray(reference.snr_db))))


# ---------------------------------------------------------------------------
# 默认路径逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", [50e9, 37.5e9, 25e9])
def test_default_schedule_is_bit_identical_to_the_scalar_zero_path(spacing_hz: float) -> None:
    """``schedule=None`` 完全走标量路径：margin/terms/notes 与显式 0 标量一致。"""
    _model, _result, baseline = _budget(spacing_hz)
    _model, _result, explicit_none = _budget(spacing_hz, dense_grid_penalty_schedule=None)
    _model, _result, scalar_zero = _budget(spacing_hz, dense_grid_penalty_db=0.0)

    for other in (explicit_none, scalar_zero):
        assert np.array_equal(baseline.margin_db, other.margin_db)
        assert baseline.notes == other.notes
        assert len(baseline.terms) == len(other.terms) == 4
        for left, right in zip(baseline.terms, other.terms, strict=True):
            assert tuple(getattr(left, field) for field in _TERM_FIELDS) == tuple(
                getattr(right, field) for field in _TERM_FIELDS
            )
            assert np.array_equal(left.value_db, right.value_db)

    # 50 GHz 窄带默认 margin 仍是 1.414 dB：schedule 默认不改变任何预算。
    # 37.5 / 25 GHz 的高功率点会触发强倾斜行（8.79 dB），那不是密集栅格余量。
    assert baseline.validity.dense_grid_overlap is True
    if spacing_hz == 50e9:
        assert float(np.median(baseline.margin_db)) == pytest.approx(np.sqrt(2.0), abs=1e-12)


# ---------------------------------------------------------------------------
# 查表与 clamped
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spacing_hz", "expected_penalty"),
    [(50e9, 0.6), (37.5e9, 2.0), (25e9, 3.6)],
)
def test_schedule_lookup_uses_the_v4_tiers(spacing_hz: float, expected_penalty: float) -> None:
    """50 / 37.5 / 25 GHz 分别落在 v4 的 0.6 / 2.0 / 3.6 dB 档。"""
    _model, _result, baseline = _budget(spacing_hz)
    _model, _result, tuned = _budget(
        spacing_hz,
        dense_grid_penalty_schedule=DENSE_GRID_PENALTY_SCHEDULE_V4,
    )

    ratio = baseline.validity.spacing_to_bandwidth
    assert ratio == pytest.approx(spacing_hz / 64e9)
    increment = np.asarray(tuned.margin_db) - np.asarray(baseline.margin_db)
    assert np.allclose(increment, expected_penalty, rtol=0.0, atol=1e-12)

    dense_term = tuned.terms[-1]
    assert dense_term.name == "dense_grid_penalty"
    assert dense_term.kind == "penalty"
    assert np.allclose(np.asarray(dense_term.value_db)[tuned.validity.active], expected_penalty, atol=1e-12)


def test_schedule_clamps_above_the_last_tier_and_says_so() -> None:
    """Ratio 高于最后一级时复用最后一级，并在 note 明确写 clamped。"""
    # 60 GHz / 64 GHz = 0.9375，仍满足 dense（ratio < 1）但超出 v4 标定范围。
    _model, _result, baseline = _budget(60e9)
    _model, _result, tuned = _budget(
        60e9,
        dense_grid_penalty_schedule=DENSE_GRID_PENALTY_SCHEDULE_V4,
    )

    assert baseline.validity.dense_grid_overlap is True
    assert baseline.validity.spacing_to_bandwidth == pytest.approx(0.9375)
    assert np.allclose(np.asarray(tuned.margin_db) - np.asarray(baseline.margin_db), 0.6)

    note = next(note for note in tuned.notes if "calibrated dense-grid penalty schedule" in note)
    assert "clamped" in note
    assert "above the last calibrated upper bound" in note
    assert "outside the calibrated range" in note


def test_schedule_lookup_helper_boundaries() -> None:
    """查表边界：首个 ``ratio <= upper`` 生效，只有超出最后一级才 clamped。"""
    schedule = DENSE_GRID_PENALTY_SCHEDULE_V4
    assert dense_grid_schedule_penalty(schedule, 0.390625) == (3.6, False)
    assert dense_grid_schedule_penalty(schedule, 0.5) == (2.0, False)
    assert dense_grid_schedule_penalty(schedule, 0.78125) == (0.6, False)
    assert dense_grid_schedule_penalty(schedule, 0.78126) == (0.6, True)

    with pytest.raises(ValueError, match="non-empty schedule"):
        dense_grid_schedule_penalty((), 0.5)


# ---------------------------------------------------------------------------
# 非法输入
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "schedule",
    [
        (),
        ((0.5, 1.0), (0.5, 2.0)),
        ((0.5, 1.0), (0.4, 2.0)),
        ((-0.1, 1.0),),
        ((0.0, 1.0),),
        ((0.5, -1.0),),
        ((float("nan"), 1.0),),
        ((0.5, float("nan")),),
        ((float("inf"), 1.0),),
        ((0.5, float("inf")),),
    ],
)
def test_invalid_schedules_are_rejected(schedule) -> None:
    """空、非递增、非正/非有限 upper、负/非有限 penalty 都必须显式报错。"""
    with pytest.raises(ValueError):
        H3UncertaintyConfig(dense_grid_penalty_schedule=schedule)


# ---------------------------------------------------------------------------
# 与标量 penalty 的优先级
# ---------------------------------------------------------------------------


def test_schedule_overrides_the_scalar_penalty() -> None:
    """Schedule 非空时覆盖 ``dense_grid_penalty_db``，不叠加。"""
    _model, _result, baseline = _budget(50e9)
    _model, _result, scalar = _budget(50e9, dense_grid_penalty_db=5.0)
    _model, _result, both = _budget(
        50e9,
        dense_grid_penalty_db=5.0,
        dense_grid_penalty_schedule=DENSE_GRID_PENALTY_SCHEDULE_V4,
    )
    _model, _result, schedule_only = _budget(
        50e9,
        dense_grid_penalty_schedule=DENSE_GRID_PENALTY_SCHEDULE_V4,
    )

    assert np.allclose(np.asarray(scalar.margin_db) - np.asarray(baseline.margin_db), 5.0)
    assert np.allclose(np.asarray(both.margin_db) - np.asarray(baseline.margin_db), 0.6)
    assert np.array_equal(both.margin_db, schedule_only.margin_db)
    assert "calibrated dense-grid penalty schedule" in both.terms[-1].source
    assert "opt-in dense-grid penalty" not in both.terms[-1].source


# ---------------------------------------------------------------------------
# term / note 语义
# ---------------------------------------------------------------------------


def test_schedule_term_and_note_are_calibrated_margin_not_a_model_fix() -> None:
    """预算必须把 schedule 写成标定余量，不能写成模型修正。"""
    _model, _result, tuned = _budget(
        37.5e9,
        dense_grid_penalty_schedule=DENSE_GRID_PENALTY_SCHEDULE_V4,
    )

    dense_term = tuned.terms[-1]
    assert dense_term.name == "dense_grid_penalty"
    assert dense_term.calibrated is False
    assert "calibrated dense-grid penalty schedule" in dense_term.source
    assert "not a model correction" in dense_term.source

    note = next(note for note in tuned.notes if "calibrated dense-grid penalty schedule" in note)
    assert "safety margin" in note
    assert "not a model correction" in note
    assert "2.00 dB" in note

    payload = tuned.to_dict()
    assert len(payload["terms"]) == 5
    assert payload["terms"][-1]["name"] == "dense_grid_penalty"
    assert "calibrated dense-grid penalty schedule" in payload["terms"][-1]["source"]


# ---------------------------------------------------------------------------
# v4 代表点：修复仍优于默认，标定 penalty 覆盖修复残差
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spacing_hz", "upper_fix_error"),
    [(50e9, 0.6), (37.5e9, 2.1), (25e9, 3.8)],
)
def test_dense_grid_fix_still_beats_the_default_and_is_covered_by_the_schedule(
    spacing_hz: float, upper_fix_error: float
) -> None:
    """v4 代表点（1 跨、6 dBm）：修复后 max|Δ| 更小，且被对应档 penalty 覆盖。"""
    case = _case(spacing_hz)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    default = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    fixed = make_model("pcfm_h3_raman", case, dense_grid_fix=True).evaluate_path(path, power)

    default_error = _max_abs_error(default, reference)
    fixed_error = _max_abs_error(fixed, reference)
    assert fixed_error < default_error - 0.3, (default_error, fixed_error)
    assert fixed_error < upper_fix_error, fixed_error

    penalty, clamped = dense_grid_schedule_penalty(DENSE_GRID_PENALTY_SCHEDULE_V4, spacing_hz / 64e9)
    assert not clamped
    assert penalty >= fixed_error, (penalty, fixed_error)


# ---------------------------------------------------------------------------
# 100 GHz 非密集栅格不受影响
# ---------------------------------------------------------------------------


def test_non_dense_100ghz_is_untouched_by_the_schedule() -> None:
    """100 GHz 不触发 dense_grid_overlap，schedule 不产生 term、不改 margin。"""
    _model, _result, baseline = _budget(100e9)
    _model, _result, tuned = _budget(
        100e9,
        dense_grid_penalty_schedule=DENSE_GRID_PENALTY_SCHEDULE_V4,
    )

    assert baseline.validity.dense_grid_overlap is False
    assert baseline.validity.spacing_to_bandwidth > 1.0
    assert np.array_equal(tuned.margin_db, baseline.margin_db)
    assert len(tuned.terms) == 4
    assert not any("dense-grid penalty" in note for note in tuned.notes)
