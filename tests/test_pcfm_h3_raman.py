"""``pcfm_h3_raman`` 的回归与验收测试。

覆盖任务要求：
* 注册与非法选项；
* 无泵时与 H3 PCFM **逐位一致**；
* EGN 自动口径：前向泵关闭，无泵/后向保持开启且逐位不变；
* 泵功率单调（跨内出纤功率与 Raman ASE 随泵功率增加）；
* FW/BW/混合泵都可用；
* C 波段 1/10 跨相对 ``hybrid_raman_paper`` 的误差；
* 10 跨速度门槛；
* C+L 六泵场景相对 paper/lumped 的误差。
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model, get_model_info
from network_nlo_eval.models.pcfm_h3_raman import H3_RAMAN_MODEL_INFO
from network_nlo_eval.models.raman import RamanGainModel
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, RamanPumpConfig

#: H3 最优配置；无泵逐位一致与所有精度对照都用它。
H3_OPTIONS = {
    "poly_degree": 3,
    "egn": True,
    "edge_fix": True,
    "cache_spans": True,
    "spp_steps": "auto",
}

#: 10 跨 C 波段算例。
CASE_10 = build_case("c_57ch")
#: 单跨 C 波段算例。
CASE_1 = build_case("scaling_40ch")

#: 论文 hybrid 的六泵 E 波段配置（与 full_paper_comparison 一致）。
SIX_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]


def _pump(power_w: float = 0.2, direction: str = "backward") -> RamanPumpConfig:
    """构造一个 1450 nm 泵浦。"""
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _path_result(case, model_name: str, **options):
    """构造模型并跑一次路径。"""
    return make_model(model_name, case, **options).evaluate_path(build_path(case), launch_profile(case))


# ---------------------------------------------------------------------------
# 注册与选项
# ---------------------------------------------------------------------------


def test_model_is_registered_as_an_independent_model() -> None:
    """新模型必须独立注册，并声明是 PCFM 核 + 粗 ODE 剖面。"""
    info = get_model_info("pcfm_h3_raman")
    assert info is H3_RAMAN_MODEL_INFO
    assert info.family == "gn+raman"
    assert "profile-aware" in info.summary.lower()
    assert "pcfm" in info.summary.lower()
    assert "buglia" in info.reference.lower()


def test_invalid_raman_steps_and_coherence_are_rejected() -> None:
    """非法的粗 ODE 步数与跨间口径必须直接报错。"""
    with pytest.raises(ValueError, match="raman_steps"):
        make_model("pcfm_h3_raman", CASE_1, raman_steps=2)
    with pytest.raises(ValueError, match="raman_iterations"):
        make_model("pcfm_h3_raman", CASE_1, raman_iterations=0)
    with pytest.raises(ValueError, match="span_coherence"):
        make_model("pcfm_h3_raman", CASE_1, span_coherence="coherent")


def test_invalid_egn_is_rejected() -> None:
    """``egn`` 只接受 ``True``/``False``/``"auto"``。"""
    with pytest.raises(ValueError, match="egn"):
        make_model("pcfm_h3_raman", CASE_1, egn="on")
    with pytest.raises(ValueError, match="egn"):
        make_model("pcfm_h3_raman", CASE_1, egn=1)


# ---------------------------------------------------------------------------
# 泵浦-泵浦拉曼开关
# ---------------------------------------------------------------------------


def test_pump_pump_raman_option_is_documented_as_default_on() -> None:
    """模型元数据必须声明该开关，且默认值保持历史行为。"""
    options = dict(H3_RAMAN_MODEL_INFO.options)
    assert "pump_pump_raman" in options
    assert "Default True" in options["pump_pump_raman"]
    assert make_model("pcfm_h3_raman", CASE_1)._h3_pump_pump_raman is True


def test_invalid_pump_pump_raman_is_rejected() -> None:
    """非布尔的开关必须直接报错，不能被静默 ``bool()`` 掉。"""
    for value in (1, 0, "on", None):
        with pytest.raises(ValueError, match="pump_pump_raman"):
            make_model("pcfm_h3_raman", CASE_1, pump_pump_raman=value)


def test_no_pump_pump_pump_off_is_bit_identical_to_h3_pcfm() -> None:
    """无泵时关掉泵浦-泵浦耦合不得改变 H3 PCFM 路径。"""
    baseline = _path_result(CASE_1, "pcfm", **H3_OPTIONS)
    new = _path_result(CASE_1, "pcfm_h3_raman", pump_pump_raman=False)
    _assert_bit_identical(baseline, new)


def test_single_pump_pump_pump_off_is_bit_identical() -> None:
    """单泵没有泵浦-泵浦对，开关前后必须逐位一致。"""
    on = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=[_pump()])
    off = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=[_pump()], pump_pump_raman=False)
    _assert_bit_identical(on, off)


def test_c_l_six_pumps_pump_pump_off_changes_pump_output() -> None:
    """C+L 六泵之间会互相转移功率；关掉后泵浦输出必须改变。"""
    case = replace(
        build_case("c_57ch"),
        name="cl_40ch_1span",
        bands="C+L",
        num_channels=40,
        center_frequency_hz=190.0e12,
        span_count=1,
    )
    path = build_path(case)
    power = launch_profile(case)
    fiber = path.fiber_configs[0]
    on_model = make_model("pcfm_h3_raman", case, raman_pumps=SIX_PUMPS)
    off_model = make_model("pcfm_h3_raman", case, raman_pumps=SIX_PUMPS, pump_pump_raman=False)
    on_model.evaluate_path(path, power)
    off_model.evaluate_path(path, power)
    pump_on = np.asarray(on_model.kernel_for(fiber).last_pump_power_out_w, dtype=np.float64)
    pump_off = np.asarray(off_model.kernel_for(fiber).last_pump_power_out_w, dtype=np.float64)
    assert pump_on.shape == (len(SIX_PUMPS),)
    assert not np.allclose(pump_on, pump_off, rtol=1e-6)
    # 泵浦仍被信号耗尽、仍按自身衰减：开关只移除泵浦之间的耦合。
    assert np.all(pump_on > 0.0) and np.all(pump_off > 0.0)


# ---------------------------------------------------------------------------
# 无泵退化：逐位一致
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [CASE_1, CASE_10], ids=["1span", "10span"])
def test_no_pump_is_bit_identical_to_h3_pcfm(case) -> None:
    """无泵时新模型的每个输出数组都必须与 H3 PCFM 完全相同。"""
    baseline = _path_result(case, "pcfm", **H3_OPTIONS)
    new = _path_result(case, "pcfm_h3_raman")
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(np.asarray(getattr(baseline, name)), np.asarray(getattr(new, name))), name


def test_no_pump_with_undepleted_option_is_bit_identical_to_h3_pcfm() -> None:
    """``pump_depletion=False`` 不得改变无泵路径。"""
    baseline = _path_result(CASE_1, "pcfm", **H3_OPTIONS)
    new = _path_result(CASE_1, "pcfm_h3_raman", pump_depletion=False)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(np.asarray(getattr(baseline, name)), np.asarray(getattr(new, name))), name


def test_forward_pump_undepleted_has_more_nli_and_lower_snr() -> None:
    """前向泵未耗尽口径保留更多泵浦功率，NLI 更大、SNR 更低。"""
    case = CASE_1
    path = build_path(case)
    power = launch_profile(case)
    pump = [_pump(0.2, "forward")]
    depleted = make_model("pcfm_h3_raman", case, raman_pumps=pump, pump_depletion=True).evaluate_path(
        path, power
    )
    undepleted = make_model("pcfm_h3_raman", case, raman_pumps=pump, pump_depletion=False).evaluate_path(
        path, power
    )
    nli_depleted = np.mean(np.asarray(depleted.spm_noise_w) + np.asarray(depleted.xpm_noise_w))
    nli_undepleted = np.mean(np.asarray(undepleted.spm_noise_w) + np.asarray(undepleted.xpm_noise_w))
    assert nli_undepleted > nli_depleted
    assert float(np.mean(undepleted.snr_db)) < float(np.mean(depleted.snr_db)) - 0.3


def test_signal_polarizations_no_pump_is_bit_identical_to_h3_pcfm() -> None:
    """无泵时 ``signal_polarizations=2`` 不得改变 H3 路径（仍逐位等于 PCFM）。"""
    baseline = _path_result(CASE_1, "pcfm", **H3_OPTIONS)
    # ``=2`` 在 ``launch_power_convention="total"`` 下是未声明单位的倍率，必须 warning。
    with pytest.warns(UserWarning, match="launch_power_convention"):
        new = _path_result(CASE_1, "pcfm_h3_raman", signal_polarizations=2)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(np.asarray(getattr(baseline, name)), np.asarray(getattr(new, name))), name


def test_signal_polarizations_option_propagates_to_the_coupled_ode() -> None:
    """有泵时因子 2 必须改变 H3 的剖面/NLI（不是被静默忽略）。"""
    case = CASE_1
    path = build_path(case)
    power = launch_profile(case)
    pump = [_pump(0.2, "forward")]
    one = make_model(
        "pcfm_h3_raman", case, raman_pumps=pump, signal_polarizations=1
    ).evaluate_path(path, power)
    with pytest.warns(UserWarning, match="launch_power_convention"):
        two = make_model(
            "pcfm_h3_raman", case, raman_pumps=pump, signal_polarizations=2
        ).evaluate_path(path, power)
    nli_one = np.asarray(one.spm_noise_w) + np.asarray(one.xpm_noise_w)
    nli_two = np.asarray(two.spm_noise_w) + np.asarray(two.xpm_noise_w)
    assert not np.allclose(nli_one, nli_two, rtol=1e-6)
    # 更强的泵浦耗尽降低 NLI，SNR 因而更高，与前向 hybrid 的预期一致。
    assert float(np.mean(two.snr_db)) > float(np.mean(one.snr_db))


# ---------------------------------------------------------------------------
# EGN 自动口径：前向泵关闭，其余保持开启
# ---------------------------------------------------------------------------

#: 逐位一致比较覆盖的输出数组。
_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _assert_bit_identical(left, right) -> None:
    """两个路径结果的所有输出数组必须逐位相同。"""
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


def test_egn_auto_resolution_follows_pump_direction() -> None:
    """``"auto"`` 只在存在正向活动泵时关闭 EGN；零功率泵不算。"""
    assert make_model("pcfm_h3_raman", CASE_1).egn is True
    assert make_model("pcfm_h3_raman", CASE_1, raman_pumps=[_pump()]).egn is True
    assert make_model("pcfm_h3_raman", CASE_1, raman_pumps=[_pump(0.0, "forward")]).egn is True
    assert make_model("pcfm_h3_raman", CASE_1, raman_pumps=[_pump(0.2, "forward")]).egn is False
    assert make_model(
        "pcfm_h3_raman", CASE_1, raman_pumps=[_pump(0.2, "forward"), _pump()]
    ).egn is False


@pytest.mark.parametrize(
    "pumps",
    [None, [_pump()], [_pump(0.2, "backward"), _pump(0.1, "backward")]],
    ids=["nopump", "backward", "two_backward"],
)
def test_egn_auto_keeps_egn_without_forward_pump(pumps) -> None:
    """无泵与纯后向泵：默认 ``"auto"`` 与显式 ``egn=True`` 逐位一致。"""
    options = {} if pumps is None else {"raman_pumps": pumps}
    auto = _path_result(CASE_1, "pcfm_h3_raman", **options)
    legacy = _path_result(CASE_1, "pcfm_h3_raman", **options, egn=True)
    _assert_bit_identical(auto, legacy)


@pytest.mark.parametrize(
    "pumps",
    [[_pump(0.2, "forward")], [_pump(0.2, "forward"), _pump(0.15, "backward")]],
    ids=["forward", "mixed"],
)
def test_egn_auto_disables_egn_with_forward_pump(pumps) -> None:
    """纯前向或混合泵：默认 ``"auto"`` 与显式 ``egn=False`` 逐位一致。"""
    auto = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=pumps)
    legacy_off = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=pumps, egn=False)
    _assert_bit_identical(auto, legacy_off)
    # 显式 True 仍保留旧的 EGN 行为，说明 auto 确实改变了结果而非空操作。
    forced_on = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=pumps, egn=True)
    assert not np.array_equal(np.asarray(auto.spm_noise_w), np.asarray(forced_on.spm_noise_w))


# ---------------------------------------------------------------------------
# 泵功率单调与 FW/BW
# ---------------------------------------------------------------------------


def test_fiber_output_power_and_raman_ase_increase_with_pump_power() -> None:
    """跨内出纤功率与新产生 Raman ASE 随泵功率单调增加。"""
    case = CASE_1
    power = launch_profile(case)
    fiber = build_path(case).fiber_configs[0]
    length_m = case.span_length_km * 1e3
    powers = []
    ases = []
    for pump_power in (0.05, 0.1, 0.2, 0.3):
        model = make_model("pcfm_h3_raman", case, raman_pumps=[_pump(pump_power)])
        kernel = model.kernel_for(fiber)
        output = kernel.evaluate(
            power,
            length_m,
            1,
            span_index=0,
            beta2_accum_ps2=np.zeros(case.num_channels, dtype=np.float64),
            span_context=None,
        )
        powers.append(float(np.mean(output.power_out_w)))
        ases.append(float(np.mean(kernel.last_raman_ase_w)))
    assert np.all(np.diff(powers) > 0.0), powers
    assert np.all(np.diff(ases) > 0.0), ases


def test_forward_backward_and_mixed_pumps_run() -> None:
    """FW、BW 与混合泵都能给出有限、正的 NLI 与 ASE。"""
    case = CASE_1
    path = build_path(case)
    power = launch_profile(case)
    results = {}
    for name, pumps in (
        ("fw", [_pump(0.2, "forward")]),
        ("bw", [_pump(0.2, "backward")]),
        ("mixed", [_pump(0.2, "forward"), _pump(0.15, "backward")]),
    ):
        result = make_model("pcfm_h3_raman", case, raman_pumps=pumps).evaluate_path(path, power)
        assert np.all(np.isfinite(result.snr_db))
        assert np.all(np.asarray(result.spm_noise_w) > 0.0)
        assert np.all(np.asarray(result.xpm_noise_w) > 0.0)
        assert np.all(np.asarray(result.ase_noise_w) > 0.0)
        results[name] = result
    # FW 与 BW 的增益位置不同，NLI 不应逐位相同。
    assert not np.array_equal(np.asarray(results["fw"].spm_noise_w), np.asarray(results["bw"].spm_noise_w))


# ---------------------------------------------------------------------------
# 精度对照
# ---------------------------------------------------------------------------


def _pump_effect(result, base) -> float:
    """相对同一模型无泵基线的平均 SNR 变化 (dB)。"""
    return float(np.mean(result.snr_db) - np.mean(base.snr_db))


def test_c_band_one_span_matches_paper_within_015_db() -> None:
    """C 波段 1 跨（后向泵）：新模型与 paper 的平均 SNR 差 <0.15 dB。"""
    paper = _path_result(CASE_1, "hybrid_raman_paper", raman_pumps=[_pump()], nli_form="paper")
    new = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=[_pump()])
    delta = float(np.mean(new.snr_db) - np.mean(paper.snr_db))
    assert abs(delta) < 0.15, f"1-span SNR error {delta:+.3f} dB"


def test_forward_pump_one_span_matches_paper_within_020_db() -> None:
    """前向泵默认关闭 EGN 后，H3 与 paper 的残差只是核公式差，<0.2 dB。"""
    paper = _path_result(
        CASE_1,
        "hybrid_raman_paper",
        raman_pumps=[_pump(direction="forward")],
        nli_form="paper",
    )
    new = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=[_pump(direction="forward")])
    delta = float(np.mean(new.snr_db) - np.mean(paper.snr_db))
    assert abs(delta) < 0.2, f"1-span forward-pump SNR error {delta:+.3f} dB"


def test_c_band_ten_span_raman_effect_matches_paper_within_015_db() -> None:
    """C 波段 10 跨：相对各自无泵基线的 Raman 效应差 <0.15 dB。

    绝对 SNR 差还包含 PCFM 与 Buglia 闭式核的跨间口径差（paper 默认
    ``epsilon=0`` 功率相加，PCFM 是可见度加权相干和），因此这里把 Raman
    剖面/ASE 的贡献单独隔离出来。
    """
    base = _path_result(CASE_10, "pcfm", **H3_OPTIONS)
    paper_base = _path_result(CASE_10, "hybrid_raman_paper", nli_form="paper")
    paper = _path_result(CASE_10, "hybrid_raman_paper", raman_pumps=[_pump()], nli_form="paper")
    new = _path_result(CASE_10, "pcfm_h3_raman", raman_pumps=[_pump()], span_coherence="pcfm")
    effect_error = _pump_effect(new, base) - _pump_effect(paper, paper_base)
    assert abs(effect_error) < 0.15, f"Raman-effect error {effect_error:+.3f} dB"


def test_c_band_ten_span_paper_matched_mode_is_within_015_db() -> None:
    """开 ``span_coherence="incoherent"`` 时，10 跨绝对 SNR 也进入 0.15 dB。"""
    paper = _path_result(CASE_10, "hybrid_raman_paper", raman_pumps=[_pump()], nli_form="paper")
    new = _path_result(
        CASE_10,
        "pcfm_h3_raman",
        raman_pumps=[_pump()],
        span_coherence="incoherent",
    )
    delta = float(np.mean(new.snr_db) - np.mean(paper.snr_db))
    assert abs(delta) < 0.15, f"paper-matched 10-span SNR error {delta:+.3f} dB"


def test_c_l_six_pumps_is_not_worse_than_lumped() -> None:
    """C+L 六泵：新模型相对 paper 的误差必须小于 lumped 的误差。"""
    case = replace(
        build_case("c_57ch"),
        name="cl_40ch_10span",
        bands="C+L",
        num_channels=40,
        center_frequency_hz=190.0e12,
    )
    paper = _path_result(case, "hybrid_raman_paper", raman_pumps=SIX_PUMPS, nli_form="paper")
    lumped = _path_result(
        case,
        "pcfm",
        **H3_OPTIONS,
        raman_model="lumped",
        raman_pumps=SIX_PUMPS,
    )
    new = _path_result(case, "pcfm_h3_raman", raman_pumps=SIX_PUMPS)
    new_error = abs(float(np.mean(new.snr_db) - np.mean(paper.snr_db)))
    lumped_error = abs(float(np.mean(lumped.snr_db) - np.mean(paper.snr_db)))
    assert new_error < lumped_error, f"new {new_error:.3f} dB vs lumped {lumped_error:.3f} dB"
    assert new_error < 0.15, f"C+L SNR error {new_error:.3f} dB"


# ---------------------------------------------------------------------------
# 表格增益谱：表入口、C+L 可用性与默认逐位不变
# ---------------------------------------------------------------------------

#: 默认（Gaussian）C+L 六泵单跨指纹，抓取自基线 ``830c062``。
#:
#: 数值以 ``float.hex()`` 保存并按 IEEE-754 位比较：表入口只应新增分支，
#: 默认路径的任何末位变化都必须让这个测试变红。
GOLDEN_C_L_DEFAULT = {
    "mean_snr_db": "0x1.9559b63ce1ebep+4",
    "mean_nli_w": "0x1.d752067939da0p-18",
    "mean_ase_w": "0x1.4434a591829c4p-22",
    "mean_signal_w": "0x1.47ae147ae147ap-9",
}


def _c_l_single_span_case():
    """C+L 40 信道单跨算例，六泵场景与 ``test_c_l_six_pumps_*`` 共用。"""
    return replace(
        build_case("c_57ch"),
        name="cl_40ch_1span",
        bands="C+L",
        num_channels=40,
        center_frequency_hz=190.0e12,
        span_count=1,
    )


def _sampled_gaussian_gain_table(points: int = 1201, span_hz: float = 30e12):
    """把解析 Gaussian 采样成 ``(offsets, gains)``。"""
    analytic = RamanGainModel()
    offsets = np.linspace(0.0, span_hz, points)
    return offsets, analytic.gain_shape(offsets)


def _ripple_gain_table(amplitude: float = 0.25, period_hz: float = 1.5e12):
    """在 Gaussian 表上叠加相对纹波，模拟实测谱的起伏。"""
    offsets, gains = _sampled_gaussian_gain_table()
    rippled = gains * (1.0 + amplitude * np.sin(2.0 * np.pi * offsets / period_hz))
    return offsets, np.maximum(rippled, 0.0)


def _c_l_result(**options):
    """跑一次 C+L 六泵单跨并返回均值读数与泵浦输出。"""
    case = _c_l_single_span_case()
    path = build_path(case)
    model = make_model("pcfm_h3_raman", case, raman_pumps=SIX_PUMPS, **options)
    result = model.evaluate_path(path, launch_profile(case))
    nli = np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)
    pump_out = np.asarray(model.kernel_for(path.fiber_configs[0]).last_pump_power_out_w, dtype=np.float64)
    return {
        "mean_snr_db": float(np.mean(result.snr_db)),
        "mean_nli_w": float(np.mean(nli)),
        "mean_ase_w": float(np.mean(result.ase_noise_w)),
        "mean_signal_w": float(np.mean(result.signal_power_w)),
        "pump_out_w": pump_out,
    }


def test_default_c_l_six_pump_result_is_bit_for_bit_unchanged() -> None:
    """表入口不得改变默认 Gaussian 路径的任何一位。"""
    baseline = _c_l_result()
    for key, expected_hex in GOLDEN_C_L_DEFAULT.items():
        assert baseline[key].hex() == expected_hex, f"{key} drifted from the 830c062 baseline"


def test_gain_table_resampling_gaussian_matches_analytic_c_l() -> None:
    """Gaussian 采样表在 C+L 六泵下应复现解析谱的结果。"""
    offsets, gains = _sampled_gaussian_gain_table()
    analytic = _c_l_result()
    resampled = _c_l_result(raman_gain_model="table", raman_gain_table=(offsets, gains))
    assert abs(resampled["mean_snr_db"] - analytic["mean_snr_db"]) < 0.01
    assert abs(resampled["mean_nli_w"] / analytic["mean_nli_w"] - 1.0) < 1e-3
    assert np.allclose(resampled["pump_out_w"], analytic["pump_out_w"], rtol=1e-3)


def test_gain_table_ripple_changes_c_l_six_pump_result() -> None:
    """谱形纹波必须透传到泵浦输出、NLI 与 SNR，而不是被静默忽略。"""
    offsets, gains = _ripple_gain_table()
    analytic = _c_l_result()
    rippled = _c_l_result(raman_gain_model="table", raman_gain_table=(offsets, gains))
    # 纹波改变泵浦-泵浦能量转移（六泵输出可差到 O(100%)）与泵浦-信号谱形。
    assert np.max(np.abs(rippled["pump_out_w"] / analytic["pump_out_w"] - 1.0)) > 0.01
    assert rippled["mean_nli_w"] / analytic["mean_nli_w"] - 1.0 > 0.01
    assert rippled["mean_snr_db"] < analytic["mean_snr_db"] - 0.05


def test_gain_table_option_reaches_the_coupled_solver() -> None:
    """模型层构造的表格模型必须原样进入核与 ODE 求解器。"""
    case = _c_l_single_span_case()
    offsets, gains = _sampled_gaussian_gain_table()
    model = make_model(
        "pcfm_h3_raman",
        case,
        raman_pumps=SIX_PUMPS,
        raman_gain_model="table",
        raman_gain_table=(offsets, gains),
    )
    assert model._raman_gain_model.gain_model == "table"
    kernel = model.kernel_for(build_path(case).fiber_configs[0])
    solver = kernel._solver_for(100e3)
    assert solver.gain_model.gain_model == "table"
    assert np.array_equal(solver.gain_model.table[0], offsets)
    assert np.array_equal(solver.gain_model.table[1], gains)


def test_gain_table_options_are_documented() -> None:
    """``ModelInfo.options`` 必须列出表入口，CLI/文档才能发现它。"""
    options = dict(H3_RAMAN_MODEL_INFO.options)
    assert "raman_gain_table" in options
    assert "table" in options["raman_gain_model"]


# ---------------------------------------------------------------------------
# 前向泵核审计（scripts/h3_forward_audit.py 的固定场景）
# ---------------------------------------------------------------------------

#: 审计脚本的光纤：100 km、0.2 dB/km、D=15.68、A_eff=80。
AUDIT_FIBER = FiberSpanConfig(
    length_km=100.0,
    attenuation_db_km_ref=0.2,
    dispersion_parameter_d=15.68076226687911,
    dispersion_slope_s=0.04123833468291227,
    effective_area_um2_ref=80.0,
    nonlinear_index_n2=2.6e-20,
)
AUDIT_SYMBOL_RATE_HZ = 50e9
AUDIT_LAUNCH_POWER_W = 1e-3


def _audit_model(model_name: str, n_channels: int, config: str, **options):
    """按审计脚本的栅格构造单跨模型。"""
    grid = FlexibleGrid(slot_width_hz=12.5e9, num_slots=8 * n_channels + 8, band_low_edge_hz=192.95e12)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=index,
            first_slot=2 + 8 * index,
            n_slots=4,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=AUDIT_SYMBOL_RATE_HZ,
            roll_off=0.0,
            launch_power_w=AUDIT_LAUNCH_POWER_W,
        )
        for index in range(n_channels)
    ]
    if config != "nopump":
        options["raman_pumps"] = [
            RamanPumpConfig(
                wavelength_nm=1450.0,
                power_w=0.2,
                direction="forward",
                attenuation_db_per_km=0.25,
            )
        ]
    return create_model(
        model_name,
        grid,
        AUDIT_FIBER,
        symbol_rate_hz=AUDIT_SYMBOL_RATE_HZ,
        connections=connections,
        slots_per_channel=8,
        channel_bandwidth_source="occupied",
        **options,
    )


def _audit_result(model_name: str, n_channels: int, config: str, **options):
    """按审计脚本的栅格跑一次单跨路径。"""
    model = _audit_model(model_name, n_channels, config, **options)
    path = MultiSpanOpticalPath([AUDIT_FIBER], [EDFAConfig(noise_figure_db=5.5)])
    return model.evaluate_path(path, np.full(n_channels, AUDIT_LAUNCH_POWER_W))


def _audit_snr(model_name: str, n_channels: int, config: str, **options) -> float:
    """平均 NLI-limited SNR (dB)，与审计脚本同口径。"""
    result = _audit_result(model_name, n_channels, config, **options)
    signal = np.asarray(result.signal_power_w, dtype=np.float64)
    nli = np.asarray(result.spm_noise_w, dtype=np.float64) + np.asarray(result.xpm_noise_w, dtype=np.float64)
    return float(np.mean(10.0 * np.log10(signal / nli)))


def _audit_center_snr(model_name: str, n_channels: int, config: str, **options) -> float:
    """中心信道 NLI-limited SNR (dB)，与 ``dra_waveform_compare.py`` 同口径。"""
    result = _audit_result(model_name, n_channels, config, **options)
    center = n_channels // 2
    signal = float(np.asarray(result.signal_power_w)[center])
    nli = float(np.asarray(result.spm_noise_w)[center] + np.asarray(result.xpm_noise_w)[center])
    return float(10.0 * np.log10(signal / nli))


#: 62.5 m 收敛耗尽参考；波形的中心信道值由测试读取，不硬编码。
CONVERGED_DEPLETED_JSON = (
    Path(__file__).resolve().parents[1] / "data/dra_waveform_converged/dra_reference_depleted.json"
)


def _waveform_center_snr(dataset: str, config: str) -> float:
    """从收敛耗尽参考读中心信道的波形 SNR_NLI (dB)。"""
    reference = json.loads(CONVERGED_DEPLETED_JSON.read_text(encoding="utf-8"))
    entry = reference["datasets"][dataset]
    center = int(entry["center_channel_index"])
    return float(entry["configs"][config]["per_channel"][center]["snr_nli_db"])


def test_forward_three_channel_legacy_egn_gap_is_pinned() -> None:
    """显式 ``egn=True`` 保留旧的 0.34 dB 固定值（历史/兼容性）。

    该差在无泵时同样存在（见审计文档），因此它钉住的是 H3 核 + EGN 口径，
    而不是拉曼剖面的拟合路径；拟合路径的任何漂移都会直接改变该差。
    默认 ``egn="auto"`` 在前向泵下关闭 EGN，见下一个测试。
    """
    h3 = _audit_snr("pcfm_h3_raman", 3, "fwd_0p2w", egn=True)
    paper = _audit_snr("hybrid_raman_paper", 3, "fwd_0p2w", nli_form="paper")
    delta = h3 - paper
    assert abs(delta - 0.34) < 0.05, f"forward 3ch H3-paper gap {delta:+.3f} dB"


def test_forward_egn_auto_is_not_optimistic_against_hybrid_or_waveform() -> None:
    """默认 ``egn="auto"`` 下，前向 3ch 的 H3 不再比 hybrid 乐观。

    关闭前向 EGN 后，H3 只比 hybrid 悲观约 0.19 dB（PCFM 核 vs Buglia 核的
    已知差），且相对收敛耗尽波形的残差比 hybrid 更小。
    """
    wave = _waveform_center_snr("3ch", "fwd_0p20w")
    h3 = _audit_center_snr("pcfm_h3_raman", 3, "fwd_0p2w")
    hybrid = _audit_center_snr("hybrid_raman_paper", 3, "fwd_0p2w", nli_form="paper")
    delta = h3 - hybrid
    assert -0.35 < delta < 0.0, f"forward 3ch H3-hybrid gap {delta:+.3f} dB"
    assert h3 - wave <= hybrid - wave + 0.1, (
        f"H3 residual {h3 - wave:+.3f} dB vs hybrid {hybrid - wave:+.3f} dB"
    )


def test_forward_gap_is_dominated_by_egn_not_by_the_polynomial_fit() -> None:
    """拟合收敛只移动 <0.05 dB；EGN 贡献 >0.3 dB；前向不额外放大差距。

    这里全部用显式 ``egn=True``：它钉住的是旧 EGN 口径下的归因预算，与默认
    ``"auto"`` 的行为无关。
    """
    default = _audit_snr("pcfm_h3_raman", 3, "fwd_0p2w", egn=True)
    converged = _audit_snr("pcfm_h3_raman", 3, "fwd_0p2w", egn=True, poly_degree=9, spp_steps=96)
    assert abs(converged - default) < 0.05, f"fit-convergence shift {converged - default:+.3f} dB"

    no_egn = _audit_snr("pcfm_h3_raman", 3, "fwd_0p2w", egn=False)
    assert default - no_egn > 0.3, f"EGN contribution {default - no_egn:+.3f} dB"

    # 同核对照：hybrid 的旧 PCFM 路径与 H3 共用 SCI/XCI 代码和 EGN，
    # 只差 ODE 步数（24 vs 12），应在 0.05 dB 内。
    control = _audit_snr(
        "hybrid_raman_paper",
        3,
        "fwd_0p2w",
        nli_form="pcfm",
        poly_degree=3,
        spp_steps=12,
        egn=True,
    )
    assert abs(default - control) < 0.06, f"same-kernel control shift {default - control:+.3f} dB"

    paper = _audit_snr("hybrid_raman_paper", 3, "fwd_0p2w", nli_form="paper")
    h3_nopump = _audit_snr("pcfm_h3_raman", 3, "nopump")
    paper_nopump = _audit_snr("hybrid_raman_paper", 3, "nopump", nli_form="paper")
    pump_specific = (default - paper) - (h3_nopump - paper_nopump)
    assert abs(pump_specific) < 0.1, f"pump-specific excess {pump_specific:+.3f} dB"


# ---------------------------------------------------------------------------
# 速度
# ---------------------------------------------------------------------------


def test_ten_span_warm_latency_is_under_the_failure_threshold() -> None:
    """10 跨热调用中位耗时应明显低于 100 ms 的失败门槛。"""
    model = make_model("pcfm_h3_raman", CASE_10, raman_pumps=[_pump()])
    path = build_path(CASE_10)
    power = launch_profile(CASE_10)
    model.evaluate_path(path, power)  # warm-up：JIT 与求解器构造
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        model.evaluate_path(path, power)
        samples.append((time.perf_counter() - started) * 1e3)
    median_ms = float(np.median(samples))
    assert median_ms < 100.0, f"10-span warm latency {median_ms:.1f} ms"
