"""H3 有效性诊断层（``validity_report`` / ``validity_mode``）的验收测试。

覆盖任务 B 的要求：

* E 带 479 信道梳的零色散边缘必须被 flag，且定位到正确的信道；
* 已验证的 C / C+L / S+C+L 波段不得被 flag；
* 强功率倾斜与强泵浦增益倾斜必须被 flag；
* 有泵 / 无泵时默认（``warn``）数值逐位不变；
* ``strict`` 只把无效信道的 SNR 置为 ``NaN``，其余数组不动；
* 泵方向与泵耗尽度按“相对无信号泵浦出射功率”报告。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm import POWER_TILT_VALIDITY_DB, XCI_EDGE_MATERIALITY
from network_nlo_eval.network.elements import RamanPumpConfig


def _kernel(case, model_name: str = "pcfm", **options):
    """构造模型并返回（单跨核，跨长 m，入纤功率）。"""
    model = make_model(model_name, case, **options)
    fiber = build_path(case).fiber_configs[0]
    return model.kernel_for(fiber), case.span_length_km * 1e3, launch_profile(case)


def _pump(power_w: float = 0.2, direction: str = "backward") -> RamanPumpConfig:
    """1450 nm 泵浦。"""
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


#: 论文 C+L 六泵配置。
SIX_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]

_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w")


def _assert_arrays_equal(left, right) -> None:
    """逐位比较除 SNR 外的所有输出数组。"""
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


# ---------------------------------------------------------------------------
# 报告结构
# ---------------------------------------------------------------------------


def test_report_has_one_entry_per_channel_and_does_not_change_the_span() -> None:
    """报告形状正确，重复调用幂等，且不会改变同一次评估的数值。"""
    case = build_case("c_57ch")
    model = make_model("pcfm", case)
    fiber = build_path(case).fiber_configs[0]
    kernel = model.kernel_for(fiber)
    length_m = case.span_length_km * 1e3
    power = launch_profile(case)

    first = kernel.validity_report(length_m, power)
    second = kernel.validity_report(length_m, power)
    for name in (
        "xci_gated",
        "xci_row_correction",
        "xci_s_min",
        "sci_x",
        "valid",
        "confidence",
    ):
        left = np.asarray(getattr(first, name))
        right = np.asarray(getattr(second, name))
        assert left.shape == (case.num_channels,), name
        assert np.array_equal(left, right), name

    before = kernel.evaluate(power, length_m, 1)
    kernel.validity_report(length_m, power)
    after = kernel.evaluate(power, length_m, 1)
    for name in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        assert np.array_equal(np.asarray(getattr(before, name)), np.asarray(getattr(after, name))), name


# ---------------------------------------------------------------------------
# E 带边缘：必须 flag
# ---------------------------------------------------------------------------


def test_e_band_zero_dispersion_edge_is_flagged() -> None:
    """479 信道 E+S+C+L 梳在零色散交叉处必须被识别，且该行确实被修正。"""
    case = build_case("escl_479ch_1span")
    kernel, length_m, power = _kernel(case)
    report = kernel.validity_report(length_m, power)

    # 零色散交叉点由有效色散的最小绝对值定位，与报告无关。
    crossing = int(kernel.inverse_order[np.argmin(np.abs(kernel.beta2_eff_sci))])
    assert report.flagged[crossing]
    assert report.xci_gated[crossing]
    assert report.sci_taylor[crossing]
    assert report.xci_row_correction[crossing] > 0.5
    assert report.sci_x_abs_min < 0.5

    # 边缘不是单个信道：整段 E 带行都被门控修正。
    assert int(np.sum(report.xci_gated)) >= 50
    assert int(np.sum(report.flagged)) >= 50
    # 精确积分兜底后，这些信道仍然可用（strict 不会把它们变成 NaN）。
    assert bool(np.all(report.valid))


def test_edge_without_fix_is_reported_invalid() -> None:
    """关掉 edge_fix 时，同一批信道必须从“已修正”变成“无效”。"""
    case = build_case("escl_479ch_1span")
    kernel, length_m, power = _kernel(case, edge_fix=False)
    report = kernel.validity_report(length_m, power)

    crossing = int(kernel.inverse_order[np.argmin(np.abs(kernel.beta2_eff_sci))])
    assert report.flagged[crossing]
    assert not report.xci_gated[crossing]
    assert not report.valid[crossing]
    assert report.xci_row_correction[crossing] > XCI_EDGE_MATERIALITY
    # 未修正的 SCI 小 X 分支同样必须判为无效。
    assert not bool(np.all(report.valid))


# ---------------------------------------------------------------------------
# 已验证波段：不得 flag
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", ["c_57ch", "cl_152ch", "scl_277ch"])
def test_validated_bands_are_not_flagged(case_name: str) -> None:
    """C / C+L / S+C+L 的闭式核已在 0.1 dB 内验证，不得产生误报。"""
    case = build_case(case_name)
    kernel, length_m, power = _kernel(case)
    report = kernel.validity_report(length_m, power)

    assert not bool(np.any(report.flagged)), case_name
    assert bool(np.all(report.valid)), case_name
    assert not bool(np.any(report.xci_gated)), case_name
    assert not bool(np.any(report.sci_taylor)), case_name
    # 原始 s 可以低于 switch（最近邻对），但行修正未超过 3 % 的材质门槛。
    assert float(np.max(report.xci_row_correction)) < XCI_EDGE_MATERIALITY
    assert report.power_tilt_db <= POWER_TILT_VALIDITY_DB


# ---------------------------------------------------------------------------
# 强倾斜：必须 flag
# ---------------------------------------------------------------------------


def test_strong_launch_power_tilt_is_flagged() -> None:
    """20 dB 的入纤功率倾斜超出已验证范围，必须 flag 并判为无效。"""
    case = build_case("c_57ch")
    kernel, length_m, power = _kernel(case)
    tilted = np.array(power, dtype=np.float64, copy=True)
    tilted[: case.num_channels // 2] *= 100.0  # +20 dB

    report = kernel.validity_report(length_m, tilted)
    assert report.power_tilt_db > POWER_TILT_VALIDITY_DB
    assert bool(np.any(report.flagged))
    assert bool(np.any(~report.valid))


def test_pumped_gain_tilt_is_flagged() -> None:
    """强泵浦增益倾斜（>15 dB）必须 flag；常规 C+L 六泵不得 flag。"""
    case = build_case("cl_152ch")
    kernel, length_m, power = _kernel(case, "pcfm_h3_raman", raman_pumps=SIX_PUMPS)
    normal = kernel.validity_report(length_m, power)
    assert normal.gain_tilt_db <= POWER_TILT_VALIDITY_DB
    assert not bool(np.any(normal.flagged))

    # 未耗尽对照避免强泵下的 ODE 溢出，同时隔离增益倾斜这个判据。
    strong = [
        RamanPumpConfig(
            wavelength_nm=pump.wavelength_nm,
            power_w=2.0 * pump.power_w,
            direction="backward",
            attenuation_db_per_km=pump.attenuation_db_per_km,
        )
        for pump in SIX_PUMPS
    ]
    strong_kernel, _, _ = _kernel(
        case,
        "pcfm_h3_raman",
        raman_pumps=strong,
        pump_depletion=False,
    )
    report = strong_kernel.validity_report(length_m, power)
    assert report.gain_tilt_db > POWER_TILT_VALIDITY_DB
    assert bool(np.any(report.flagged))
    assert bool(np.any(~report.valid))


# ---------------------------------------------------------------------------
# 泵字段
# ---------------------------------------------------------------------------


def test_pump_direction_and_depletion_are_reported() -> None:
    """前向/后向泵的方向与耗尽度来自同一次耦合求解。"""
    case = build_case("c_57ch")
    pumps = [_pump(0.2, "forward"), _pump(0.15, "backward")]
    kernel, length_m, power = _kernel(case, "pcfm_h3_raman", raman_pumps=pumps)
    report = kernel.validity_report(length_m, power)

    assert report.pump_active
    assert report.pump_directions == ("forward", "backward")
    assert report.pump_depletion.shape == (2,)
    assert bool(np.all(report.pump_depletion >= 0.0))
    assert bool(np.all(report.pump_depletion <= 1.0))
    assert report.pump_depletion_max > 0.5


def test_no_pump_report_is_pump_free() -> None:
    """无泵核的泵字段为空，与基类报告一致。"""
    case = build_case("c_57ch")
    base_kernel, length_m, power = _kernel(case, "pcfm")
    h3_kernel, _, _ = _kernel(case, "pcfm_h3_raman")

    base = base_kernel.validity_report(length_m, power)
    h3 = h3_kernel.validity_report(length_m, power)
    assert not base.pump_active and not h3.pump_active
    assert base.pump_directions == () and h3.pump_directions == ()
    assert base.pump_depletion_max == 0.0 and h3.pump_depletion_max == 0.0
    for name in ("valid", "flagged", "confidence", "xci_gated", "sci_taylor"):
        assert np.array_equal(np.asarray(getattr(base, name)), np.asarray(getattr(h3, name))), name


# ---------------------------------------------------------------------------
# warn / strict
# ---------------------------------------------------------------------------


def test_invalid_validity_mode_is_rejected() -> None:
    """只接受 ``warn`` 与 ``strict``。"""
    with pytest.raises(ValueError, match="validity_mode"):
        make_model("pcfm", build_case("c_57ch"), validity_mode="off")


@pytest.mark.parametrize(
    ("model_name", "options"),
    [
        ("pcfm", {}),
        ("pcfm_h3_raman", {"raman_pumps": [_pump()]}),
        ("pcfm_h3_raman", {"raman_pumps": [_pump(0.2, "forward")]}),
    ],
    ids=["nopump", "backward", "forward"],
)
def test_warn_does_not_change_any_number(model_name: str, options) -> None:
    """默认 warn 与 strict 在无无效信道时逐位一致（包括有泵）。"""
    case = build_case("c_57ch")
    path = build_path(case)
    power = launch_profile(case)
    warn = make_model(model_name, case, validity_mode="warn", **options).evaluate_path(path, power)
    strict = make_model(model_name, case, validity_mode="strict", **options).evaluate_path(path, power)

    _assert_arrays_equal(warn, strict)
    assert np.array_equal(np.asarray(warn.snr_db), np.asarray(strict.snr_db))


def test_warn_records_the_link_report_without_masking() -> None:
    """warn 记录逐跨与链路级报告，但不改动 SNR。"""
    case = build_case("c_57ch")
    model = make_model("pcfm", case, validity_mode="warn")
    result = model.evaluate_path(build_path(case), launch_profile(case))

    assert len(model.last_span_validity) == 10
    assert model.last_validity_valid is not None
    assert model.last_validity_flagged is not None
    assert model.last_validity_confidence is not None
    assert bool(np.all(model.last_validity_valid))
    assert not bool(np.any(model.last_validity_flagged))
    assert bool(np.all(np.isfinite(result.snr_db)))


def test_strict_masks_only_the_invalid_channels() -> None:
    """strict 只把无效信道的 SNR 变成 NaN，其余信道与功率数组逐位不变。"""
    case = build_case("escl_479ch_1span")
    path = build_path(case)
    power = launch_profile(case)
    warn_model = make_model("pcfm", case, edge_fix=False, validity_mode="warn")
    strict_model = make_model("pcfm", case, edge_fix=False, validity_mode="strict")

    warn = warn_model.evaluate_path(path, power)
    strict = strict_model.evaluate_path(path, power)
    invalid = ~warn_model.last_validity_valid
    assert int(np.sum(invalid)) > 0

    _assert_arrays_equal(warn, strict)
    strict_snr = np.asarray(strict.snr_db)
    assert np.all(np.isnan(strict_snr[invalid]))
    assert np.array_equal(np.asarray(warn.snr_db)[~invalid], strict_snr[~invalid])
    assert int(np.sum(np.isnan(strict_snr))) == int(np.sum(invalid))


def test_strict_keeps_edge_fixed_channels_finite() -> None:
    """默认 edge_fix 已兜底 E 带边缘，strict 不应把修正过的信道丢掉。"""
    case = build_case("escl_479ch_1span")
    model = make_model("pcfm", case, validity_mode="strict")
    result = model.evaluate_path(build_path(case), launch_profile(case))

    assert bool(np.all(model.last_validity_valid))
    assert int(np.sum(model.last_validity_flagged)) >= 50
    assert bool(np.all(np.isfinite(result.snr_db)))
