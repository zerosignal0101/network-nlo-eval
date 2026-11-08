"""集总等效拉曼放大组件与 PCFM 集成测试。

覆盖任务要求的可验证点：

* 无泵时 PCFM 结果逐位不变；
* 单泵 on-off 增益为正且随泵浦功率单调；
* 增益频率响应在 13.2 THz 频移附近达到峰值；
* 拉曼 ASE 随泵浦功率与跨长单调；
* 与 ``physics_nlo_eval.RamanPump`` 的 Gaussian 增益形状逐项一致；
* 一个 hybrid（Raman + EDFA）链路的 SNR 提升例子。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, grid_for, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.raman import (
    C_LIGHT,
    RamanAmplifier,
    RamanGainModel,
)
from network_nlo_eval.network.elements import EDFAConfig, RamanPumpConfig

#: C 波段的 57 信道基准；10 跨，每跨 100 km。
CASE = build_case("c_57ch")

#: 典型 C 波段信号频率 (Hz)。
C_FREQUENCIES_HZ: NDArrayFloat = C_LIGHT / np.linspace(1530e-9, 1565e-9, 57)


def _pump(
    wavelength_nm: float = 1450.0,
    power_w: float = 0.2,
    direction: str = "backward",
) -> RamanPumpConfig:
    """构造一个泵浦配置。"""
    return RamanPumpConfig(
        wavelength_nm=wavelength_nm,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _snr_linear(result) -> NDArrayFloat:
    """由路径结果计算逐信道线性 SNR。"""
    noise = np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w) + np.asarray(result.ase_noise_w)
    return np.asarray(result.signal_power_w) / noise


# ---------------------------------------------------------------------------
# 默认关闭 / 无泵
# ---------------------------------------------------------------------------


def test_lumped_raman_with_empty_pumps_is_bit_for_bit_unchanged() -> None:
    """开启拉曼但泵浦为空时，所有输出必须与关闭时完全一致。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    empty = make_model("pcfm", CASE, raman_model="lumped", raman_pumps=[]).evaluate_path(path, power)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(np.asarray(getattr(base, name)), np.asarray(getattr(empty, name)))


def test_fiber_pumps_are_ignored_unless_raman_is_enabled() -> None:
    """``FiberSpanConfig.raman_pumps`` 不应在未 opt-in 时改变 PCFM 结果。"""
    power = launch_profile(CASE)
    plain = build_path(CASE)
    with_pumps = MultiSpanOpticalPath(
        [fiber.model_copy(update={"raman_pumps": [_pump()]}) for fiber in plain.fiber_configs],
        plain.edfa_configs,
    )
    base = make_model("pcfm", CASE).evaluate_path(plain, power)
    ignored = make_model("pcfm", CASE).evaluate_path(with_pumps, power)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(np.asarray(getattr(base, name)), np.asarray(getattr(ignored, name)))


def test_zero_power_pump_is_identity() -> None:
    """功率为零的泵浦必须给出逐位为 1 的增益与零 ASE。"""
    model = RamanGainModel()
    pump = _pump(power_w=0.0)
    gain = model.on_off_gain(C_FREQUENCIES_HZ, [pump], 100e3)
    ase = model.ase_w(C_FREQUENCIES_HZ, np.full(C_FREQUENCIES_HZ.size, 64e9), [pump], 100e3)
    assert np.array_equal(gain, np.ones_like(C_FREQUENCIES_HZ))
    assert np.array_equal(ase, np.zeros_like(C_FREQUENCIES_HZ))


# ---------------------------------------------------------------------------
# 增益
# ---------------------------------------------------------------------------


def test_single_pump_positive_gain_and_power_monotonic() -> None:
    """单泵 on-off 增益为正，且随泵浦功率严格单调增加。"""
    model = RamanGainModel()
    powers = np.array([0.02, 0.05, 0.1, 0.2, 0.3])
    gains = np.array(
        [model.on_off_gain_db(C_FREQUENCIES_HZ, [_pump(power_w=value)], 100e3) for value in powers]
    )
    assert np.all(gains > 0)
    assert np.all(np.diff(gains, axis=0) > 0)


def test_frequency_response_peaks_at_the_raman_shift() -> None:
    """给定泵浦，增益应在泵浦-信号频移 13.2 THz 附近最大。"""
    model = RamanGainModel()
    pump = _pump(wavelength_nm=1450.0, power_w=0.15)
    pump_frequency_hz = C_LIGHT / (pump.wavelength_nm * 1e-9)
    # 覆盖泵浦下方 5-22 THz 的连续信号频率，使峰值落在网格内部。
    frequencies = pump_frequency_hz - np.linspace(5e12, 22e12, 341)
    gain_db = model.on_off_gain_db(frequencies, [pump], 80e3)
    peak_shift_hz = pump_frequency_hz - frequencies[int(np.argmax(gain_db))]
    assert abs(peak_shift_hz - 13.2e12) < 0.1e12
    # 增益剖面应是单峰的：峰值两侧都更低。
    peak_index = int(np.argmax(gain_db))
    assert np.all(np.diff(gain_db[: peak_index + 1]) > 0)
    assert np.all(np.diff(gain_db[peak_index:]) < 0)


def test_linear_gain_model_rises_to_cutoff() -> None:
    """CZ 线性近似在截止频移前单调上升，截止后为零。"""
    model = RamanGainModel(gain_model="linear", delta_f_cutoff_hz=15e12)
    shifts = np.array([0.0, 5e12, 10e12, 13.2e12, 14.5e12, 16e12])
    shape = model.gain_shape(shifts)
    assert shape[0] == 0.0
    assert np.all(np.diff(shape[1:5]) > 0)
    assert shape[5] == 0.0


# ---------------------------------------------------------------------------
# 表格增益谱
# ---------------------------------------------------------------------------


def _gaussian_table(points: int = 1201, span_hz: float = 30e12):
    """把解析 Gaussian 采样成表，用于验证表入口。"""
    model = RamanGainModel()
    offsets = np.linspace(0.0, span_hz, points)
    return model, offsets, model.gain_shape(offsets)


def test_table_matches_sampled_gaussian() -> None:
    """Gaussian 采样表与解析 Gaussian 的差应在插值精度内。"""
    analytic, offsets, gains = _gaussian_table()
    table = RamanGainModel.from_table(offsets, gains)
    shifts = np.linspace(1e9, 30e12, 500)
    # 25 GHz 网格上的线性插值误差约 (h/sigma)^2/8 ~ 2e-5。
    assert np.allclose(table.gain_shape(shifts), analytic.gain_shape(shifts), rtol=0.0, atol=5e-5)
    # 表节点上必须精确复现采样值。
    assert np.allclose(table.gain_shape(offsets), gains, rtol=0.0, atol=1e-12)

    pump = _pump(wavelength_nm=1450.0, power_w=0.15)
    table_gain = table.on_off_gain_db(C_FREQUENCIES_HZ, [pump], 100e3)
    analytic_gain = analytic.on_off_gain_db(C_FREQUENCIES_HZ, [pump], 100e3)
    assert np.allclose(table_gain, analytic_gain, rtol=0.0, atol=5e-4)


def test_table_gain_is_zero_outside_the_table() -> None:
    """表外频移与非正频移的增益必须为零，不做端点外推。"""
    _, offsets, gains = _gaussian_table()
    table = RamanGainModel.from_table(offsets, gains)
    outside = table.gain_shape(np.array([0.0, -1e12, 40e12, 1e15]))
    assert np.array_equal(outside, np.zeros(4))


def test_table_entry_and_named_constructor_agree_bit_for_bit() -> None:
    """``from_table`` 与 ``gain_model='table'`` 必须构造出同一张表。"""
    _, offsets, gains = _gaussian_table()
    named = RamanGainModel(gain_model="table", table=(offsets, gains))
    factory = RamanGainModel.from_table(offsets, gains)
    probe = np.linspace(0.0, 32e12, 257)
    assert np.array_equal(named.gain_shape(probe), factory.gain_shape(probe))
    assert np.array_equal(named.table[0], factory.table[0])
    assert np.array_equal(named.table[1], factory.table[1])


def test_default_gaussian_shape_is_bit_for_bit_unchanged() -> None:
    """默认构造仍走解析 Gaussian 分支，数值与显式 ``"gaussian"`` 逐位一致。"""
    implicit = RamanGainModel()
    explicit = RamanGainModel(gain_model="gaussian")
    shifts = np.linspace(0.0, 30e12, 2001)
    assert np.array_equal(implicit.gain_shape(shifts), explicit.gain_shape(shifts))
    expected = np.where(
        shifts > 0,
        np.exp(-0.5 * ((shifts - 13.2e12) / 2.12e12) ** 2),
        0.0,
    )
    assert np.array_equal(implicit.gain_shape(shifts), expected)
    pump = _pump(power_w=0.15)
    bandwidths = np.full(C_FREQUENCIES_HZ.size, 64e9)
    assert np.array_equal(
        implicit.on_off_gain(C_FREQUENCIES_HZ, [pump], 100e3),
        explicit.on_off_gain(C_FREQUENCIES_HZ, [pump], 100e3),
    )
    assert np.array_equal(
        implicit.ase_w(C_FREQUENCIES_HZ, bandwidths, [pump], 100e3),
        explicit.ase_w(C_FREQUENCIES_HZ, bandwidths, [pump], 100e3),
    )


def test_invalid_gain_tables_are_rejected() -> None:
    """长度、单调性、符号与有限性错误都必须显式报错。"""
    offsets = np.linspace(0.0, 20e12, 9)
    gains = np.linspace(0.0, 1.0, 9)

    with pytest.raises(ValueError, match="same length"):
        RamanGainModel.from_table(offsets, gains[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        RamanGainModel.from_table(offsets[::-1], gains)
    with pytest.raises(ValueError, match="strictly increasing"):
        RamanGainModel.from_table(np.array([0.0, 1e12, 1e12]), np.array([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError, match="non-negative"):
        RamanGainModel.from_table(offsets, -gains)
    with pytest.raises(ValueError, match="finite"):
        RamanGainModel.from_table(offsets, np.where(np.arange(9) == 4, np.nan, gains))
    with pytest.raises(ValueError, match="finite"):
        RamanGainModel.from_table(offsets, np.where(np.arange(9) == 4, np.inf, gains))
    with pytest.raises(ValueError, match="non-negative"):
        RamanGainModel.from_table(np.linspace(-20e12, -1e12, 9), gains)
    with pytest.raises(ValueError, match="at least two"):
        RamanGainModel.from_table(np.array([1e12]), np.array([1.0]))
    with pytest.raises(ValueError, match="one-dimensional"):
        RamanGainModel.from_table(offsets[:, None], gains)
    with pytest.raises(ValueError, match="requires table"):
        RamanGainModel(gain_model="table")
    with pytest.raises(ValueError, match="only used when"):
        RamanGainModel(gain_model="gaussian", table=(offsets, gains))


def test_lumped_amplifier_accepts_a_table_model() -> None:
    """表格模型必须能直接替换集总放大器的解析谱。"""
    analytic, offsets, gains = _gaussian_table()
    table = RamanGainModel.from_table(offsets, gains)
    pump = _pump(power_w=0.15)
    bandwidths = np.full(C_FREQUENCIES_HZ.size, 64e9)
    expected_gain = RamanAmplifier([pump], 100e3, model=analytic).gain(C_FREQUENCIES_HZ)
    table_gain = RamanAmplifier([pump], 100e3, model=table).gain(C_FREQUENCIES_HZ)
    assert np.allclose(table_gain, expected_gain, rtol=1e-4, atol=0.0)
    expected_ase = RamanAmplifier([pump], 100e3, model=analytic).ase_w(C_FREQUENCIES_HZ, bandwidths)
    table_ase = RamanAmplifier([pump], 100e3, model=table).ase_w(C_FREQUENCIES_HZ, bandwidths)
    assert np.allclose(table_ase, expected_ase, rtol=1e-3, atol=0.0)


def test_forward_and_backward_pumps_have_same_on_off_gain() -> None:
    """未耗尽集总等效下，FW/BW 的末端 on-off 增益幅度相同。"""
    model = RamanGainModel()
    forward = model.on_off_gain_db(C_FREQUENCIES_HZ, [_pump(direction="forward")], 100e3)
    backward = model.on_off_gain_db(C_FREQUENCIES_HZ, [_pump(direction="backward")], 100e3)
    assert np.allclose(forward, backward, rtol=0.0, atol=1e-12)


def test_multiple_pumps_add_in_the_exponent() -> None:
    """多个泵浦的增益在指数域相加，而不是简单取最大。"""
    model = RamanGainModel()
    first = _pump(wavelength_nm=1430.0, power_w=0.1)
    second = _pump(wavelength_nm=1460.0, power_w=0.1)
    combined = model.on_off_gain(C_FREQUENCIES_HZ, [first, second], 100e3)
    separate = model.on_off_gain(C_FREQUENCIES_HZ, [first], 100e3) * model.on_off_gain(
        C_FREQUENCIES_HZ, [second], 100e3
    )
    assert np.allclose(combined, separate, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# ASE
# ---------------------------------------------------------------------------


def test_ase_monotonic_in_pump_power_and_span_length() -> None:
    """拉曼 ASE 随泵浦功率与跨长单调增加。"""
    model = RamanGainModel()
    bandwidths = np.full(C_FREQUENCIES_HZ.size, 64e9)
    powers = [0.05, 0.1, 0.2, 0.3]
    by_power = [model.ase_w(C_FREQUENCIES_HZ, bandwidths, [_pump(power_w=value)], 100e3) for value in powers]
    assert np.all(np.diff(np.array(by_power), axis=0) > 0)

    lengths = [20e3, 50e3, 100e3]
    by_length = [model.ase_w(C_FREQUENCIES_HZ, bandwidths, [_pump()], value) for value in lengths]
    assert np.all(np.diff(np.array(by_length), axis=0) > 0)


def test_backward_pump_generates_more_ase_than_forward() -> None:
    """相同 on-off 增益下，反向泵浦的 ASE 更高（增益集中在跨段末端）。"""
    model = RamanGainModel()
    bandwidths = np.full(C_FREQUENCIES_HZ.size, 64e9)
    forward = model.ase_w(C_FREQUENCIES_HZ, bandwidths, [_pump(direction="forward")], 100e3)
    backward = model.ase_w(C_FREQUENCIES_HZ, bandwidths, [_pump(direction="backward")], 100e3)
    assert np.all(backward > forward)
    assert np.all(forward > 0)


# ---------------------------------------------------------------------------
# 与波形级 RamanPump 的增益形状对照
# ---------------------------------------------------------------------------


def test_gain_shape_matches_physics_nlo_raman_pump() -> None:
    """关闭光子能量因子后，与 ``physics_nlo_eval.RamanPump`` 的增益逐项一致。"""
    network = pytest.importorskip("physics_nlo_eval.channel.network")
    model = RamanGainModel(include_photon_energy_factor=False)
    pump_config = _pump(wavelength_nm=1450.0, power_w=0.15)
    gain = model.on_off_gain(C_FREQUENCIES_HZ, [pump_config], 100e3, a_eff_m2=80e-12)
    waveform_pump = network.RamanPump(
        network.PumpSpec(
            wavelength_nm=pump_config.wavelength_nm,
            power_w=pump_config.power_w,
            direction=pump_config.direction,
            attenuation_db_per_km=pump_config.attenuation_db_per_km,
        ),
        length_km=100.0,
        effective_area_um2=80.0,
        add_noise=False,
    )
    reference = waveform_pump._gain(C_FREQUENCIES_HZ)
    assert np.allclose(gain, reference, rtol=1e-12, atol=0.0)
    assert np.corrcoef(gain, reference)[0, 1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PCFM / 路由集成
# ---------------------------------------------------------------------------


def test_auto_gain_edfa_recovers_power_after_raman() -> None:
    """自动增益 EDFA 在拉曼增益后只补偿剩余损耗，输出功率不变。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    hybrid = make_model("pcfm", CASE, raman_model="lumped", raman_pumps=[_pump()]).evaluate_path(path, power)
    assert np.allclose(hybrid.signal_power_w, base.signal_power_w, rtol=1e-12, atol=0.0)
    # 集总等效不改变跨内 NLI，只有浮点级的传输因子差异。
    assert np.allclose(hybrid.spm_noise_w, base.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(hybrid.xpm_noise_w, base.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_fixed_gain_edfa_adds_the_raman_gain() -> None:
    """固定增益 EDFA 下，拉曼增益直接体现在跨段输出功率上。"""
    case = build_case("scaling_40ch")
    power = launch_profile(case)
    fiber = build_path(case).fiber_configs[0]
    path = MultiSpanOpticalPath([fiber], [EDFAConfig(target_gain_db=20.0, noise_figure_db=5.5)])
    base = make_model("pcfm", case).evaluate_path(path, power)
    pump = _pump(wavelength_nm=1450.0, power_w=0.1)
    hybrid = make_model("pcfm", case, raman_model="lumped", raman_pumps=[pump]).evaluate_path(path, power)
    wavelengths_m = np.asarray(grid_for(case).wavelengths)
    a_eff_m2 = (80.0 + 0.05 * (wavelengths_m * 1e9 - 1550.0)) * 1e-12
    amplifier = RamanAmplifier([pump], fiber.length_km * 1e3, a_eff_m2=a_eff_m2)
    expected_gain = amplifier.gain(np.asarray(grid_for(case).frequencies))
    assert np.allclose(hybrid.signal_power_w / base.signal_power_w, expected_gain, rtol=1e-12, atol=0.0)


def test_hybrid_link_snr_improves_over_edfa_only() -> None:
    """一个 hybrid（Raman + EDFA）链路的平均 SNR 应优于纯 EDFA。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    hybrid = make_model("pcfm", CASE, raman_model="lumped", raman_pumps=[_pump()]).evaluate_path(path, power)
    base_snr = 10.0 * np.log10(_snr_linear(base))
    hybrid_snr = 10.0 * np.log10(_snr_linear(hybrid))
    assert np.mean(hybrid_snr) > np.mean(base_snr) + 0.3
    assert np.all(hybrid_snr > base_snr)
    assert np.mean(hybrid.ase_noise_w) < np.mean(base.ase_noise_w)


def test_raman_ase_can_be_switched_off() -> None:
    """``raman_ase=False`` 只保留拉曼增益，总 ASE 应等于纯 EDFA 的 ASE。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    no_raman_ase = make_model(
        "pcfm",
        CASE,
        raman_model="lumped",
        raman_pumps=[_pump()],
        raman_ase=False,
    ).evaluate_path(path, power)
    with_raman_ase = make_model("pcfm", CASE, raman_model="lumped", raman_pumps=[_pump()]).evaluate_path(
        path, power
    )
    # 增益相同，EDFA 增益被压低；关掉拉曼 ASE 后总 ASE 应低于开启时，且低于纯 EDFA。
    assert np.all(no_raman_ase.ase_noise_w < with_raman_ase.ase_noise_w)
    assert np.mean(no_raman_ase.ase_noise_w) < np.mean(base.ase_noise_w)


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------


def test_invalid_raman_options_are_rejected() -> None:
    """非法拉曼选项必须直接报错，不静默降级。"""
    case = build_case("scaling_40ch")
    with pytest.raises(ValueError, match="raman_model"):
        make_model("pcfm", case, raman_model="coupled")
    with pytest.raises(ValueError, match="gain_model"):
        make_model("pcfm", case, raman_model="lumped", raman_gain_model="measured")
    with pytest.raises(ValueError, match="delta_f_cutoff_hz"):
        RamanGainModel(gain_model="linear", delta_f_peak_hz=13.2e12, delta_f_cutoff_hz=10e12)
    with pytest.raises(ValueError, match="ase_steps"):
        RamanGainModel(ase_steps=4)
    with pytest.raises(ValueError, match="length_m"):
        RamanAmplifier([_pump()], 0.0)


def test_fiber_pumps_are_used_when_raman_is_enabled() -> None:
    """With ``raman_model='lumped'`` the per-fibre pump list is authoritative."""
    power = launch_profile(CASE)
    plain = build_path(CASE)
    pump = _pump()
    with_pumps = MultiSpanOpticalPath(
        [fiber.model_copy(update={"raman_pumps": [pump]}) for fiber in plain.fiber_configs],
        plain.edfa_configs,
    )
    from_fibre = make_model("pcfm", CASE, raman_model="lumped").evaluate_path(with_pumps, power)
    explicit = make_model("pcfm", CASE, raman_model="lumped", raman_pumps=[pump]).evaluate_path(with_pumps, power)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        np.testing.assert_array_equal(
            np.asarray(getattr(from_fibre, name)), np.asarray(getattr(explicit, name))
        )
