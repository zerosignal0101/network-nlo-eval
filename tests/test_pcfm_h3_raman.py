"""``pcfm_h3_raman`` 的回归与验收测试。

覆盖任务要求：
* 注册与非法选项；
* 无泵时与 H3 PCFM **逐位一致**；
* 泵功率单调（跨内出纤功率与 Raman ASE 随泵功率增加）；
* FW/BW/混合泵都可用；
* C 波段 1/10 跨相对 ``hybrid_raman_paper`` 的误差；
* 10 跨速度门槛；
* C+L 六泵场景相对 paper/lumped 的误差。
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import get_model_info
from network_nlo_eval.models.pcfm_h3_raman import H3_RAMAN_MODEL_INFO
from network_nlo_eval.network.elements import RamanPumpConfig

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


def test_forward_pump_one_span_matches_paper_within_015_db() -> None:
    """前向泵的跨内剖面效应更强，1 跨仍应与 paper 差 <0.15 dB。"""
    paper = _path_result(
        CASE_1,
        "hybrid_raman_paper",
        raman_pumps=[_pump(direction="forward")],
        nli_form="paper",
    )
    new = _path_result(CASE_1, "pcfm_h3_raman", raman_pumps=[_pump(direction="forward")])
    delta = float(np.mean(new.snr_db) - np.mean(paper.snr_db))
    assert abs(delta) < 0.15, f"1-span forward-pump SNR error {delta:+.3f} dB"


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
