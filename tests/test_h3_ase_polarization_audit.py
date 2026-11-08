"""ASE 每偏振化与 ``per_pol`` 输出语义的回归测试（任务 L）。

口径约定（与任务 J/设计 O 一致）：

* 网络参考面永远是每信道**双偏振总功率** ``P_total``；
* ``total`` 模型拿 ``P_total``，信号/NLI/ASE 都是总量；
* ``per_pol`` 模型拿 ``P_total/2``，信号/NLI/EDFA ASE/ROADM booster ASE/
  Raman ASE 都必须恰好减半一次；``snr_db`` 与坐标无关。

本文件钉住五件事：

1. ``total``+1 的默认路径（有泵/无泵）逐位不变；
2. 同一物理总功率下 ``per_pol``+2 的 SNR 与 ``total``+1 一致（C 波段
   平均差 <0.01 dB）；
3. ASE 数组在 ``per_pol`` 下严格减半，且 ``input_ase_w`` 不被 route 二次缩放；
4. C+L 残差按场景量化：有泵时 ODE 与 ASE 都同坐标，无泵时任务 L2 已把
   PCFM 的 ISRS 剖面改为按物理总功率驱动，两种口径同样逐信道一致；
5. 非法缩放因子与矛盾的 launch/NLI 口径必须报错。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import ElementOutput, MultiSpanOpticalPath, SpanFiberOutput
from network_nlo_eval.models.raman import H_PLANCK, RamanGainModel
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.models.route import accumulate_route, apply_amplifier_and_roadm
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
    ROADMConfig,
)

C_BAND = build_case("scaling_40ch")
C_PLUS_L = build_case("scaling_160ch")
#: 10 跨 C+L：透明跨记忆化与累积 ASE 都在这里被覆盖。
C_PLUS_L_10_SPAN = build_case("cl_152ch")

_BACKWARD = RamanPumpConfig(
    wavelength_nm=1450.0,
    power_w=0.25,
    direction="backward",
    attenuation_db_per_km=0.25,
)
_FORWARD = RamanPumpConfig(
    wavelength_nm=1450.0,
    power_w=0.25,
    direction="forward",
    attenuation_db_per_km=0.25,
)

_OUTPUT_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _arrays(result) -> dict[str, np.ndarray]:
    """Return the five path-level arrays as ``float64``."""
    return {name: np.asarray(getattr(result, name), dtype=np.float64) for name in _OUTPUT_FIELDS}


def _evaluate(case, *, pumps=None, **options):
    """Evaluate one case; ``pumps`` is only passed when non-empty."""
    opts = {"egn": "auto", **options}
    if pumps:
        opts["raman_pumps"] = list(pumps)
    return make_model("pcfm_h3_raman", case, **opts).evaluate_path(
        build_path(case), launch_profile(case)
    )


def _evaluate_at(case, power_scale: float, *, pumps=None, **options):
    """Evaluate one case with the launch power multiplied by ``power_scale``."""
    opts = {"egn": "auto", **options}
    if pumps:
        opts["raman_pumps"] = list(pumps)
    return make_model("pcfm_h3_raman", case, **opts).evaluate_path(
        build_path(case), launch_profile(case) * power_scale
    )


def _nli(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """SPM+XPM total NLI."""
    return arrays["spm_noise_w"] + arrays["xpm_noise_w"]


# ---------------------------------------------------------------------------
# 默认 total 路径逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pumps", [None, [_BACKWARD], [_FORWARD]], ids=["unpumped", "backward", "forward"])
@pytest.mark.parametrize("case", [C_BAND, C_PLUS_L], ids=["c_band", "c_plus_l"])
def test_default_total_plus_one_is_bit_identical(case, pumps) -> None:
    """不传新参数与显式 ``total``+1 的五个输出数组逐位一致（硬门槛）。"""
    baseline = _arrays(_evaluate(case, pumps=pumps))
    explicit = _arrays(
        _evaluate(
            case,
            pumps=pumps,
            signal_polarizations=1,
            launch_power_convention="total",
            nli_power_convention="total",
        )
    )
    for name in _OUTPUT_FIELDS:
        assert np.array_equal(baseline[name], explicit[name]), name


@pytest.mark.parametrize("pumps", [None, [_BACKWARD], [_FORWARD]], ids=["unpumped", "backward", "forward"])
def test_default_total_plus_one_is_bit_identical_on_a_ten_span_link(pumps) -> None:
    """10 跨 C+L：任务 L2 的 ISRS 驱动改动不得触碰默认 ``total`` 路径。"""
    baseline = _arrays(_evaluate(C_PLUS_L_10_SPAN, pumps=pumps))
    explicit = _arrays(
        _evaluate(
            C_PLUS_L_10_SPAN,
            pumps=pumps,
            signal_polarizations=1,
            launch_power_convention="total",
            nli_power_convention="total",
        )
    )
    for name in _OUTPUT_FIELDS:
        assert np.array_equal(baseline[name], explicit[name]), name


def test_route_default_ase_scale_is_bit_identical() -> None:
    """``apply_amplifier_and_roadm`` 不传缩放与传 ``1.0`` 逐位一致。"""
    power = np.array([0.5e-3, 0.2e-3])
    fiber = SpanFiberOutput(
        power_out_w=np.array([0.25e-3, 0.1e-3]),
        spm_noise_w=np.array([1e-9, 2e-9]),
        xpm_noise_w=np.array([3e-9, 4e-9]),
    )
    kwargs = dict(
        power_in_w=power,
        fiber_output=fiber,
        edfa_config=EDFAConfig(target_gain_db=10.0, noise_figure_db=5.0),
        roadm_config=None,
        nf_lin=np.full(2, 10 ** 0.5),
        f_abs_hz=np.array([193.4e12, 193.5e12]),
        bandwidth_hz=np.full(2, 50e9),
        input_ase_w=np.array([1e-10, 2e-10]),
    )
    implicit = apply_amplifier_and_roadm(**kwargs)
    explicit = apply_amplifier_and_roadm(**kwargs, ase_polarization_scale=1.0)
    for name in ("power_out_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"):
        assert np.array_equal(np.asarray(getattr(implicit, name)), np.asarray(getattr(explicit, name))), name


# ---------------------------------------------------------------------------
# 每偏振 ASE 与 input_ase 不二次缩放
# ---------------------------------------------------------------------------


def test_route_scales_only_the_newly_generated_ase() -> None:
    """``ase_polarization_scale`` 只缩本跨新 ASE，``input_ase_w`` 原样放大。"""
    power = np.array([0.5e-3])
    fiber = SpanFiberOutput(
        power_out_w=np.array([0.25e-3]),
        spm_noise_w=np.zeros(1),
        xpm_noise_w=np.zeros(1),
    )
    common = dict(
        power_in_w=power,
        fiber_output=fiber,
        edfa_config=EDFAConfig(target_gain_db=10.0, noise_figure_db=5.0),
        roadm_config=None,
        nf_lin=np.array([10 ** 0.5]),
        f_abs_hz=np.array([193.4e12]),
        bandwidth_hz=np.array([50e9]),
    )
    input_ase = np.array([2e-10])
    native = apply_amplifier_and_roadm(**common, input_ase_w=None)
    with_input = apply_amplifier_and_roadm(**common, input_ase_w=input_ase)
    scaled = apply_amplifier_and_roadm(**common, input_ase_w=input_ase, ase_polarization_scale=0.5)
    expected = 0.5 * np.asarray(native.ase_noise_w) + (
        np.asarray(with_input.ase_noise_w) - np.asarray(native.ase_noise_w)
    )
    assert np.allclose(np.asarray(scaled.ase_noise_w), expected, rtol=1e-15)


def test_no_edfa_passes_input_ase_through_unscaled() -> None:
    """无 EDFA 时没有“新产生”的 ASE，缩放因子不得改写输入 ASE。"""
    input_ase = np.array([1e-10, 2e-10])
    fiber = SpanFiberOutput(
        power_out_w=np.array([0.5e-3, 1e-3]),
        spm_noise_w=np.zeros(2),
        xpm_noise_w=np.zeros(2),
    )
    out = apply_amplifier_and_roadm(
        power_in_w=np.array([1e-3, 2e-3]),
        fiber_output=fiber,
        edfa_config=None,
        roadm_config=None,
        nf_lin=np.ones(2),
        f_abs_hz=np.array([193.4e12, 193.5e12]),
        bandwidth_hz=np.full(2, 50e9),
        input_ase_w=input_ase,
        ase_polarization_scale=0.5,
    )
    assert np.array_equal(np.asarray(out.ase_noise_w), input_ase)


class _TransparentModel:
    """Toy route model: transparent span, no NLI, no EDFA."""

    f_abs_hz = np.array([193.4e12])
    channel_bandwidth_hz_channels = np.array([50e9])
    n_nodes = 1

    def power_width_scale(self) -> np.ndarray:
        return np.ones(1)

    def blocked_nodes(self, roadm_config) -> list[int]:
        return []

    def span_step(self, fiber_config, edfa_config, power_in_w, span_length_m) -> ElementOutput:
        zeros = np.zeros_like(power_in_w)
        return ElementOutput(power_in_w, zeros, zeros, zeros)


def _booster_path() -> MultiSpanOpticalPath:
    """One transparent span followed by an equalizing ROADM booster."""
    return MultiSpanOpticalPath(
        fiber_configs=[FiberSpanConfig(length_km=1.0, max_span_length_km=1.0)],
        edfa_configs=[None],
        roadm_configs=[
            ROADMConfig(
                insertion_loss_db=5.0,
                express_attenuation_db=0.0,
                add_drop_loss_db=0.0,
                filtering_penalty_db=0.0,
                booster_noise_figure_db=5.0,
                equalize_output_power=True,
            )
        ],
    )


def test_accumulate_route_scales_the_booster_ase_once() -> None:
    """ROADM booster ASE 只按显式缩放一次，默认 1.0 逐位不变。"""
    model = _TransparentModel()
    path = _booster_path()
    power = np.array([1e-3])
    default = accumulate_route(model, path, power)
    explicit = accumulate_route(model, path, power, ase_polarization_scale=1.0)
    scaled = accumulate_route(model, path, power, ase_polarization_scale=0.5)
    assert np.array_equal(np.asarray(default.ase_noise_w), np.asarray(explicit.ase_noise_w))

    loss = 10 ** (5.0 / 10)
    expected = 10 ** (5.0 / 10) * H_PLANCK * 193.4e12 * 50e9 * (loss - 1)
    assert np.allclose(np.asarray(default.ase_noise_w), expected, rtol=1e-12)
    assert np.allclose(np.asarray(scaled.ase_noise_w), 0.5 * expected, rtol=1e-12)
    # 信号不随 ASE 缩放。
    assert np.array_equal(np.asarray(default.signal_power_w), np.asarray(scaled.signal_power_w))


# ---------------------------------------------------------------------------
# 相同物理总功率：SNR 与 ASE 坐标
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pumps", [[_BACKWARD], [_FORWARD]], ids=["backward", "forward"])
def test_per_pol_outputs_are_per_polarization_quantities(pumps) -> None:
    """``per_pol``+2 在 ``P/2`` 下四个数组都是 ``total``+1 的一半。

    有泵时 Raman OE 与 EDFA 都在同一坐标；无泵时任务 L2 把 PCFM 的 ISRS
    剖面也改为按物理总功率驱动，两种口径因此都是严格的 2 倍关系。
    """
    total = _arrays(_evaluate(C_BAND, pumps=pumps, signal_polarizations=1, launch_power_convention="total"))
    per_pol = _arrays(
        _evaluate_at(
            C_BAND,
            0.5,
            pumps=pumps,
            signal_polarizations=2,
            launch_power_convention="per_pol",
        )
    )
    assert np.allclose(2.0 * per_pol["signal_power_w"], total["signal_power_w"], rtol=1e-12)
    assert np.allclose(2.0 * _nli(per_pol), _nli(total), rtol=1e-4)
    assert np.allclose(2.0 * per_pol["ase_noise_w"], total["ase_noise_w"], rtol=1e-12)
    assert np.max(np.abs(per_pol["snr_db"] - total["snr_db"])) < 1e-9


def test_c_band_snr_gap_is_zero_after_the_isrs_fix() -> None:
    """C 波段无泵：ISRS 按物理总功率驱动后，SNR 逐信道一致。

    任务 L 记录过平均 +0.001 dB、逐信道 max 0.064 dB 的残差，来源是 PCFM
    的 ISRS 剖面按半功率积分；任务 L2 修掉后残差为最后一位舍入。
    """
    total = _arrays(_evaluate(C_BAND, pumps=None))
    per_pol = _arrays(
        _evaluate_at(C_BAND, 0.5, pumps=None, signal_polarizations=2, launch_power_convention="per_pol")
    )
    gap = per_pol["snr_db"] - total["snr_db"]
    assert abs(float(np.mean(gap))) < 1e-9
    assert float(np.max(np.abs(gap))) < 1e-9


def test_pumped_c_plus_l_snr_gap_is_zero() -> None:
    """C+L 有泵：Raman OE 与 EDFA 都同坐标，SNR 逐信道完全一致。"""
    for pumps in ([_BACKWARD], [_FORWARD]):
        total = _arrays(_evaluate(C_PLUS_L, pumps=pumps, signal_polarizations=1, launch_power_convention="total"))
        per_pol = _arrays(
            _evaluate_at(
                C_PLUS_L,
                0.5,
                pumps=pumps,
                signal_polarizations=2,
                launch_power_convention="per_pol",
            )
        )
        assert np.allclose(2.0 * per_pol["ase_noise_w"], total["ase_noise_w"], rtol=1e-9)
        assert np.max(np.abs(per_pol["snr_db"] - total["snr_db"])) < 1e-9


def test_c_plus_l_unpumped_residual_is_closed() -> None:
    """C+L 无泵残差（任务 L：0.193 dB 平均 / 1.303 dB 最坏）已由 L2 消除。

    无泵时 H3 不进入 Raman ODE（``last_raman_ase_w`` 为零），残差全部来自
    PCFM 的 ISRS 积分按每偏振半功率计算剖面：半功率让 ISRS 增益倾斜与
    EDFA 自动增益补偿都偏掉，ASE/SNR 随之出现 0.2 dB 量级偏差。任务 L2 让
    该积分按物理总功率驱动并把出射功率减半，两种口径逐位对齐。
    """
    total = _arrays(_evaluate(C_PLUS_L, pumps=None))
    per_pol = _arrays(
        _evaluate_at(C_PLUS_L, 0.5, pumps=None, signal_polarizations=2, launch_power_convention="per_pol")
    )
    gap = per_pol["snr_db"] - total["snr_db"]
    assert abs(float(np.mean(gap))) < 0.01
    assert float(np.max(np.abs(gap))) < 0.01
    # 信号/ASE 逐位成 2 倍关系，NLI 只差幅度平方根往返的舍入。
    assert np.array_equal(2.0 * per_pol["signal_power_w"], total["signal_power_w"])
    assert np.array_equal(2.0 * per_pol["ase_noise_w"], total["ase_noise_w"])
    assert np.allclose(2.0 * _nli(per_pol), _nli(total), rtol=1e-12)


def test_ten_span_unpumped_link_has_no_residual() -> None:
    """10 跨 C+L 无泵：累积 ASE 也逐信道对齐，残差 <0.01 dB。

    多跨会经过 10 次 EDFA 自动增益补偿，任何 ISRS 剖面偏差都会被放大，
    因此这是任务 L2 修复的最强回归。
    """
    total = _arrays(_evaluate(C_PLUS_L_10_SPAN, pumps=None))
    per_pol = _arrays(
        _evaluate_at(
            C_PLUS_L_10_SPAN,
            0.5,
            pumps=None,
            signal_polarizations=2,
            launch_power_convention="per_pol",
        )
    )
    gap = per_pol["snr_db"] - total["snr_db"]
    assert abs(float(np.mean(gap))) < 0.01
    assert float(np.max(np.abs(gap))) < 0.01
    assert np.array_equal(2.0 * per_pol["signal_power_w"], total["signal_power_w"])
    assert np.array_equal(2.0 * per_pol["ase_noise_w"], total["ase_noise_w"])


def test_raman_solver_returns_per_polarization_ase() -> None:
    """求解器层：``per_pol`` 的 ``raman_ase_w`` 恰为 ``total`` 的一半。"""
    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="forward",
        attenuation_db_per_km=0.0,
    )

    def solver(signal_polarizations: int, convention: str) -> RamanPaperSpanSolver:
        return RamanPaperSpanSolver(
            np.array([193.75e12]),
            np.array([75e9]),
            [pump],
            100e3,
            np.zeros(1),
            np.array([80e-12]),
            np.zeros(1),
            np.array([80e-12]),
            RamanGainModel(),
            n_steps=64,
            signal_polarizations=signal_polarizations,
            launch_power_convention=convention,
        )

    total = solver(1, "total").solve(np.array([30e-3]))
    per_pol = solver(2, "per_pol").solve(np.array([15e-3]))
    assert np.array_equal(per_pol.profile_norm, total.profile_norm)
    assert np.allclose(per_pol.power_out_w, 0.5 * total.power_out_w, rtol=1e-12)
    assert np.allclose(per_pol.raman_ase_w, 0.5 * total.raman_ase_w, rtol=1e-12)


# ---------------------------------------------------------------------------
# 口径选择与非法组合
# ---------------------------------------------------------------------------


def test_h3_ase_scale_follows_the_end_to_end_per_pol_convention() -> None:
    """只有 launch 与 NLI 都是 ``per_pol`` 时才把 0.5 传给 route。"""
    assert make_model("pcfm_h3_raman", C_BAND).ase_polarization_scale == 1.0
    end_to_end = make_model(
        "pcfm_h3_raman",
        C_BAND,
        signal_polarizations=2,
        launch_power_convention="per_pol",
    )
    assert end_to_end.launch_power_convention == "per_pol"
    assert end_to_end.nli_power_convention == "per_pol"
    assert end_to_end.ase_polarization_scale == 0.5
    mixed = make_model(
        "pcfm_h3_raman",
        C_BAND,
        signal_polarizations=2,
        launch_power_convention="per_pol",
        nli_power_convention="total",
    )
    assert mixed.ase_polarization_scale == 1.0
    assert make_model("pcfm", C_BAND).ase_polarization_scale == 1.0


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_ase_polarization_scale_is_rejected(scale: float) -> None:
    """非正/非有限缩放必须在 route 入口报错。"""
    power = np.array([1e-3])
    fiber = SpanFiberOutput(power_out_w=power, spm_noise_w=np.zeros(1), xpm_noise_w=np.zeros(1))
    with pytest.raises(ValueError, match="ase_polarization_scale"):
        apply_amplifier_and_roadm(
            power_in_w=power,
            fiber_output=fiber,
            edfa_config=None,
            roadm_config=None,
            nf_lin=np.ones(1),
            f_abs_hz=np.array([193.4e12]),
            bandwidth_hz=np.array([50e9]),
            ase_polarization_scale=scale,
        )
    with pytest.raises(ValueError, match="ase_polarization_scale"):
        accumulate_route(_TransparentModel(), _booster_path(), power, ase_polarization_scale=scale)


def test_inconsistent_launch_and_nli_conventions_are_rejected() -> None:
    """``per_pol`` NLI 配 ``total`` launch、``per_pol`` launch 配单偏振都报错。"""
    with pytest.raises(ValueError, match="nli_power_convention='per_pol'"):
        make_model("pcfm_h3_raman", C_BAND, nli_power_convention="per_pol")
    with pytest.raises(ValueError, match="signal_polarizations must be 2"):
        make_model("pcfm_h3_raman", C_BAND, launch_power_convention="per_pol")
