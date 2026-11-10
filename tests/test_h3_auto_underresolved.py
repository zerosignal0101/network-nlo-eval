"""任务 AL：auto Raman 分辨率告警的验收测试。

覆盖：

* 默认 ``raman_resolution_check=False`` 逐位不变，且不构造额外 solver；
* C+L 六泵 0/6 dBm：24 步相对 192 步的变化超过阈值，flag=True，点值不变；
* C 57ch 单泵：12 步检查不误报；
* E+S+C+L 六泵：96 步检查不误报；
* 报告字段默认兼容、错误预算聚合与 note；
* 非法 tol/布尔值拒绝。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm import SpanValidityReport
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
from network_nlo_eval.network.elements import RamanPumpConfig

SIX = [
    RamanPumpConfig(wavelength_nm=w, power_w=p, direction="backward", attenuation_db_per_km=0.25)
    for w, p in [
        (1340.0, 0.63),
        (1360.0, 0.63),
        (1427.0, 0.165),
        (1445.0, 0.131),
        (1476.0, 0.125),
        (1505.0, 0.075),
    ]
]
_ONE_BW = RamanPumpConfig(
    wavelength_nm=1450.0,
    power_w=0.2,
    direction="backward",
    attenuation_db_per_km=0.25,
)


def _run(case_name: str, pumps, dbm: float, **options):
    """返回 ``(model, path, power, result)``。"""
    case = build_case(case_name)
    path = build_path(case)
    power = launch_profile(case, dbm)
    model = make_model("pcfm_h3_raman", case, raman_pumps=pumps, **options)
    result = model.evaluate_path(path, power)
    return model, path, power, result


def test_default_check_is_bit_identical_and_adds_no_solve() -> None:
    """默认关闭：SNR 逐位相同，perf_ode_solves 不增加。"""
    base_model, _path, _power, base = _run("cl_152ch", SIX, 0.0)
    check_model, _path, _power, check = _run(
        "cl_152ch", SIX, 0.0, raman_resolution_check=False
    )
    assert np.array_equal(np.asarray(base.snr_db), np.asarray(check.snr_db))
    assert base_model.perf_ode_solves == 1
    assert check_model.perf_ode_solves == 1
    report = check_model.last_span_validity[0]
    assert report.raman_resolution_checked is False
    assert report.raman_auto_under_resolved is False
    assert report.raman_resolution_check_steps == 0
    assert report.raman_resolution_change == 1.0


@pytest.mark.parametrize("dbm", [0.0, 6.0])
def test_c_l_six_pump_flags_under_resolution(dbm: float) -> None:
    """C+L 六泵：24 步相对更细档变化 >10%，flag=True，点值不变。"""
    _base_model, _path, _power, base = _run("cl_152ch", SIX, dbm)
    model, _path, _power, checked = _run(
        "cl_152ch", SIX, dbm, raman_resolution_check=True
    )
    report = model.last_span_validity[0]

    assert report.raman_steps == 24
    assert report.raman_resolution_checked is True
    assert report.raman_auto_under_resolved is True
    assert report.raman_resolution_check_steps >= 48
    assert report.raman_resolution_change > 0.10
    assert np.array_equal(np.asarray(base.snr_db), np.asarray(checked.snr_db))
    assert model.perf_ode_solves > 1


def test_c_single_pump_does_not_flag() -> None:
    """C 57ch 单泵：12 步检查变化约 4%，低于 10% 阈值。"""
    model, _path, _power, _result = _run(
        "c_57ch", [_ONE_BW], 0.0, raman_resolution_check=True
    )
    report = model.last_span_validity[0]
    assert report.raman_steps == 12
    assert report.raman_resolution_checked is True
    assert report.raman_auto_under_resolved is False
    assert report.raman_resolution_change < 0.10


def test_e_s_c_l_six_pump_does_not_flag() -> None:
    """E+S+C+L 六泵：96 步相对 192 步变化约 1.4%，不误报。"""
    model, _path, _power, _result = _run(
        "escl_479ch", SIX, 6.0, raman_resolution_check=True
    )
    report = model.last_span_validity[0]
    assert report.raman_steps == 96
    assert report.raman_resolution_checked is True
    assert report.raman_auto_under_resolved is False
    assert report.raman_resolution_change < 0.10


def test_error_budget_exposes_the_flag_without_changing_margin() -> None:
    """错误预算传播 flag/字段并写 note，margin 仍只由原有项决定。"""
    model, path, power, result = _run(
        "cl_152ch", SIX, 0.0, raman_resolution_check=True
    )
    budget = estimate_h3_error_budget(model, path, power, result)
    assert budget.validity.raman_resolution_checked is True
    assert budget.validity.raman_auto_under_resolved is True
    assert budget.validity.raman_resolution_check_steps >= 48
    assert budget.validity.raman_resolution_change > 0.10
    assert any("auto resolution not certified" in note for note in budget.notes)
    # 点值/valid 不被该告警改写；flag 会进入 flagged。
    assert bool(np.all(budget.validity.valid))
    assert bool(np.any(budget.validity.flagged))


def test_new_report_fields_have_compatible_defaults() -> None:
    """旧报告构造不需要新字段；默认值保持中性。"""
    names = {field.name: field for field in dataclasses.fields(SpanValidityReport)}
    assert names["raman_auto_under_resolved"].default is False
    assert names["raman_resolution_check_steps"].default == 0
    assert names["raman_resolution_change"].default == 1.0
    assert names["raman_resolution_checked"].default is False


@pytest.mark.parametrize("tol", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_tolerance_is_rejected(tol: float) -> None:
    """Tol 必须是有限正数。"""
    with pytest.raises(ValueError, match="raman_resolution_tol"):
        build = make_model(
            "pcfm_h3_raman",
            build_case("c_57ch"),
            raman_resolution_check=True,
            raman_resolution_tol=tol,
        )
        del build


def test_non_bool_check_is_rejected() -> None:
    """Check 必须是 bool。"""
    with pytest.raises(ValueError, match="raman_resolution_check"):
        make_model(
            "pcfm_h3_raman",
            build_case("c_57ch"),
            raman_resolution_check=1,
        )
