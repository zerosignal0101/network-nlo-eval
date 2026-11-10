"""任务 AI：泵浦跨内集中损耗的验收测试。

覆盖：

* 参数校验：位置必须在 ``(0, L)``、损耗非负、重复位置按 dB 合并；
* 默认 ``lumped_loss=None`` 与不传选项逐位一致；
* 无泵路径透传给父类 PCFM，并给出 ``lumped_loss_in_ode=False``；
* 有泵前向/后向路径进入 ODE，报告段数与位置；
* 前向泵中点 3 dB 损耗与“两段串联 + 衰减器”参考的功率/Raman ASE 一致；
* 跨末附近 2 dB 损耗使前向泵输出恰好再低 2 dB。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.numba_kernels import _calc_aeff_dynamic
from network_nlo_eval.models.raman import C_LIGHT
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver, _resolve_lumped_loss
from network_nlo_eval.network.elements import RamanPumpConfig

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")
_FIBER_LENGTH_M = 100e3


def _pump(direction: str = "forward", power_w: float = 0.2) -> RamanPumpConfig:
    """1450 nm 单泵。"""
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _evaluate(pumps, lumped_loss, **options):
    """在 ``c_57ch`` 上跑一次路径，返回 ``(model, result)``。"""
    case = build_case("c_57ch")
    path = build_path(case)
    power = launch_profile(case)
    model = make_model("pcfm_h3_raman", case, raman_pumps=pumps, lumped_loss=lumped_loss, **options)
    result = model.evaluate_path(path, power)
    return model, result


def _kernel(model, path):
    """第一跨的 kernel。"""
    return model.kernel_for(path.fiber_configs[0])


def _solver(kernel, pumps, length_m: float, n_steps: int, lumped_loss=None):
    """按 kernel 的常数构造一个独立的 ``RamanPaperSpanSolver``。"""
    pump_lambdas = np.asarray([C_LIGHT / (p.wavelength_nm * 1e-9) for p in pumps])
    pump_a_eff = _calc_aeff_dynamic(
        pump_lambdas,
        kernel._fiber_config.effective_area_um2_ref,
        kernel._fiber_config.aeff_slope_um2_nm,
        kernel._fiber_config.reference_wavelength_nm,
    )
    pump_alpha = np.asarray(
        [p.attenuation_db_per_km * np.log(10.0) / 10.0 / 1e3 for p in pumps]
    )
    return RamanPaperSpanSolver(
        kernel.f_abs_hz,
        kernel.b_hz,
        pumps,
        length_m,
        kernel.alpha_npm,
        kernel.a_eff_m2,
        pump_alpha,
        pump_a_eff,
        kernel._raman_gain_model,
        n_steps=n_steps,
        n_iterations=64,
        lumped_loss=lumped_loss,
    )


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        [(0.0, 1.0)],
        [(100e3, 1.0)],
        [(-1.0, 1.0)],
        [(50e3, -1.0)],
        [(50e3, float("nan"))],
        [(float("nan"), 1.0)],
    ],
)
def test_invalid_lumped_loss_is_rejected(entry) -> None:
    """位置必须在 (0,L)、损耗非负且有限。"""
    with pytest.raises(ValueError):
        _resolve_lumped_loss(entry, _FIBER_LENGTH_M)


def test_duplicate_positions_merge_by_db() -> None:
    """同一位置的连接器按 dB 相加合并。"""
    merged, positions, factors = _resolve_lumped_loss(
        [(50e3, 1.0), (20e3, 0.5), (50e3, 2.0)], _FIBER_LENGTH_M
    )
    assert [item[0] for item in merged] == [20e3, 50e3]
    assert [item[1] for item in merged] == [0.5, 3.0]
    assert np.allclose(factors, 10.0 ** (-np.asarray([0.5, 3.0]) / 10.0))


# ---------------------------------------------------------------------------
# 默认路径
# ---------------------------------------------------------------------------


def test_default_none_is_bit_identical() -> None:
    """``None`` 与不传选项逐位一致。"""
    implicit_model, implicit = _evaluate([_pump("backward")], None)
    explicit_model, explicit = _evaluate([_pump("backward")], None)
    for name in _ARRAY_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(implicit, name)), np.asarray(getattr(explicit, name))
        ), name
    report = implicit_model.last_span_validity[0]
    assert report.lumped_loss_in_ode is False
    assert report.lumped_loss_segments == 1
    assert report.lumped_loss_positions_m.size == 0


# ---------------------------------------------------------------------------
# 无泵与有泵报告
# ---------------------------------------------------------------------------


def test_pump_free_path_forwards_to_parent_and_reports_static_fields() -> None:
    """无泵：损耗交给父类 SPP，报告 in_ode=False、段数=元素数+1。"""
    model, result = _evaluate([], [(50e3, 3.0)])
    report = model.last_span_validity[0]
    assert report.lumped_loss_in_ode is False
    assert report.lumped_loss_segments == 2
    assert np.allclose(report.lumped_loss_positions_m, [50e3])
    assert np.allclose(report.lumped_loss_db, [3.0])
    assert np.all(np.isfinite(result.snr_db))


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_pumped_path_applies_lumped_loss_inside_the_ode(direction: str) -> None:
    """有泵：报告 in_ode=True，前向/后向都有限且固定点收敛。"""
    model, result = _evaluate([_pump(direction, 0.2)], [(50e3, 3.0)])
    report = model.last_span_validity[0]
    assert report.lumped_loss_in_ode is True
    assert report.lumped_loss_segments == 2
    assert bool(report.raman_converged)
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(np.asarray(result.signal_power_w) > 0.0)


# ---------------------------------------------------------------------------
# 数值参考
# ---------------------------------------------------------------------------


def test_forward_mid_span_loss_matches_two_segment_reference() -> None:
    """前向泵：单跨分段与“两段 solver + 中间衰减器”逐信道一致。"""
    case = build_case("c_57ch")
    path = build_path(case)
    power = launch_profile(case)
    pump = _pump("forward", 0.2)
    model = make_model("pcfm_h3_raman", case, raman_pumps=[pump])
    kernel = _kernel(model, path)

    factor = 10.0 ** (-3.0 / 10.0)
    single = _solver(kernel, [pump], _FIBER_LENGTH_M, 12, [(50e3, 3.0)])
    single_result = single.solve(power)

    first = _solver(kernel, [pump], 50e3, 6)
    first_result = first.solve(power)
    pump_mid = RamanPumpConfig(
        wavelength_nm=pump.wavelength_nm,
        power_w=float(first_result.pump_power_out_w[0]) * factor,
        direction="forward",
        attenuation_db_per_km=pump.attenuation_db_per_km,
    )
    second = _solver(kernel, [pump_mid], 50e3, 6)
    second_result = second.solve(first_result.power_out_w * factor, first_result.raman_ase_w * factor)

    np.testing.assert_allclose(
        second_result.power_out_w, single_result.power_out_w, rtol=1e-5, atol=0.0
    )
    total_ase = second_result.raman_ase_w + first_result.raman_ase_w * factor * (
        second_result.power_out_w / (first_result.power_out_w * factor)
    )
    np.testing.assert_allclose(total_ase, single_result.raman_ase_w, rtol=1e-5, atol=0.0)


def test_forward_near_end_loss_adds_exactly_its_db() -> None:
    """前向泵：L-1 m 处 2 dB 损耗使光纤出射信号恰好再低 2 dB。

    路由层的 EDFA 会把 ``signal_power_w`` 补偿回目标功率，所以这里比较
    kernel 的跨末 ``power_out_w``，而不是路径级信号功率。
    """
    case = build_case("c_57ch")
    path = build_path(case)
    power = launch_profile(case)
    baseline = make_model("pcfm_h3_raman", case, raman_pumps=[_pump("forward", 0.2)])
    lossy = make_model(
        "pcfm_h3_raman",
        case,
        raman_pumps=[_pump("forward", 0.2)],
        lumped_loss=[(_FIBER_LENGTH_M - 1.0, 2.0)],
    )
    baseline_out = baseline.kernel_for(path.fiber_configs[0]).evaluate(power, _FIBER_LENGTH_M)
    lossy_out = lossy.kernel_for(path.fiber_configs[0]).evaluate(power, _FIBER_LENGTH_M)
    ratio = np.asarray(lossy_out.power_out_w) / np.asarray(baseline_out.power_out_w)
    np.testing.assert_allclose(ratio, 10.0 ** (-2.0 / 10.0), rtol=1e-5)
