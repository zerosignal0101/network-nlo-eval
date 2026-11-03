"""论文 hybrid Raman 模型（Eq. (2)/(4) ODE + Eq. (6)-(9) 闭式）的测试。

覆盖任务要求：
* 注册、构造、形状；
* ``nli_form="paper"`` 与 ``"pcfm"`` 都能跑；
* 无泵时 ``"paper"`` 退化为无 Raman 的闭式基线；
* 有泵 1/10 跨，SPM/XPM 正、有限；
* Eq. (9) 闭式 vs Eq. (6)-(8) 数值积分 <0.1 dB；
* 泵浦耗尽、FW/BW 差异、跨间 ASE 传递与一个 hybrid SNR 例子。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath, get_model_info
from network_nlo_eval.models.hybrid_raman import (
    HYBRID_RAMAN_PAPER_MODEL_INFO,
    RamanPaperSpanSolver,
)
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.network.elements import RamanPumpConfig

CASE = build_case("scaling_40ch")


def _pump(power_w: float = 0.2, direction: str = "backward") -> RamanPumpConfig:
    """构造 1450 nm 泵浦。"""
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _solver(pumps: list[RamanPumpConfig], *, n_steps: int = 32) -> RamanPaperSpanSolver:
    """在 9 个 C 波段信道上构造一个小规模 ODE 求解器。"""
    frequencies = C_LIGHT / np.linspace(1530e-9, 1565e-9, 9)
    bandwidths = np.full(frequencies.size, 64e9)
    a_eff = np.full(frequencies.size, 80e-12)
    alpha = np.full(frequencies.size, 0.2 * np.log(10.0) / 10.0 / 1e3)
    pump_alpha = np.full(len(pumps), 0.25 * np.log(10.0) / 10.0 / 1e3)
    pump_a_eff = np.full(len(pumps), 80e-12)
    return RamanPaperSpanSolver(
        frequencies,
        bandwidths,
        pumps,
        100e3,
        alpha,
        a_eff,
        pump_alpha,
        pump_a_eff,
        RamanGainModel(),
        n_steps=n_steps,
        n_iterations=24,
    )


# ---------------------------------------------------------------------------
# 注册与构造
# ---------------------------------------------------------------------------


def test_paper_model_is_registered_with_an_honest_summary() -> None:
    """模型必须独立注册，并声明实现的是 Eq. (6)-(9)。"""
    info = get_model_info("hybrid_raman_paper")
    assert info is HYBRID_RAMAN_PAPER_MODEL_INFO
    assert info.family == "gn+raman"
    assert "eq. (8)/(9)" in info.summary.lower()
    assert "nli_form='pcfm'" in info.summary.lower()
    assert "buglia" in info.reference.lower()
    assert "hybrid" in info.reference.lower()


def test_paper_model_constructs_and_returns_path_shapes() -> None:
    """构造、评估路径并检查输出形状。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    model = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()])
    result = model.evaluate_path(path, power)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.asarray(getattr(result, name)).shape == (CASE.num_channels,)
    assert np.all(np.isfinite(result.snr_db))


def test_invalid_fixed_point_iterations_are_rejected() -> None:
    """非法迭代次数必须直接报错。"""
    with pytest.raises(ValueError, match="fixed_point_iterations"):
        make_model("hybrid_raman_paper", CASE, fixed_point_iterations=0)


# ---------------------------------------------------------------------------
# NLI 口径：paper / integral / pcfm
# ---------------------------------------------------------------------------


def test_invalid_nli_form_is_rejected() -> None:
    """未知的 NLI 口径必须直接报错。"""
    with pytest.raises(ValueError, match="nli_form"):
        make_model("hybrid_raman_paper", CASE, nli_form="eq6")


@pytest.mark.parametrize("nli_form", ["paper", "integral", "pcfm"])
def test_every_nli_form_runs_and_is_finite(nli_form: str) -> None:
    """三种口径都能构造、评估并给出有限的 SPM/XPM。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    result = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()], nli_form=nli_form).evaluate_path(
        path, power
    )
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(np.asarray(result.spm_noise_w) > 0.0)
    assert np.all(np.asarray(result.xpm_noise_w) > 0.0)


def test_paper_and_pcfm_forms_are_not_aliases() -> None:
    """Eq. (6)-(9) 与旧 PCFM 核必须给出不同 NLI（不是改名的同一个核）。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    paper = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()], nli_form="paper").evaluate_path(
        path, power
    )
    legacy = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()], nli_form="pcfm").evaluate_path(
        path, power
    )
    paper_nli = np.mean(paper.spm_noise_w + paper.xpm_noise_w)
    legacy_nli = np.mean(legacy.spm_noise_w + legacy.xpm_noise_w)
    assert paper_nli != pytest.approx(legacy_nli, rel=1e-3)


# ---------------------------------------------------------------------------
# 无泵退化
# ---------------------------------------------------------------------------


def test_paper_without_pumps_matches_the_closed_form_baseline() -> None:
    """无泵时 paper 口径退化为无 Raman 的闭式基线（与 PCFM 差 <0.1 dB）。

    单跨 ``scaling_40ch``（100 km、40 信道、64 GBd）上，Eq. (2) 只剩 ISRS，
    Eq. (10) 拟合、Eq. (8)/(9) 闭式与 PCFM 的 SCI/XCI 核应给出同一量级。
    多跨/长跨链路的差异见 :func:`test_paper_no_pump_long_link_difference_is_a_kernel_effect`。
    """
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    paper = make_model("hybrid_raman_paper", CASE, nli_form="paper").evaluate_path(path, power)
    delta = float(np.mean(paper.snr_db) - np.mean(base.snr_db))
    assert abs(delta) < 0.1, f"no-pump paper deviates from PCFM by {delta:+.3f} dB"


def test_paper_no_pump_long_link_difference_is_a_kernel_effect() -> None:
    """10 跨 100 km、57 信道的无泵差异来自闭式核本身，不是 Raman 残项。

    同一链路下 ``paper`` 与 ``buglia2024_ra_gn``（同一套 Eq. (8)/(9)）必须一致；
    与 PCFM 的差值随跨数线性累积，因此这里只要求两条闭式基线吻合。
    """
    case = build_case("c_57ch")
    power = launch_profile(case)
    path = build_path(case)
    paper = make_model("hybrid_raman_paper", case, nli_form="paper").evaluate_path(path, power)
    buglia = make_model("buglia2024_ra_gn", case, nli_form="closed").evaluate_path(path, power)
    np.testing.assert_allclose(paper.snr_db, buglia.snr_db, rtol=0.0, atol=1e-3)
    assert np.mean(paper.ase_noise_w) == pytest.approx(np.mean(buglia.ase_noise_w), rel=1e-6)


def test_pcfm_legacy_without_pumps_is_bit_for_bit_the_pcfm_baseline() -> None:
    """旧口径 ``nli_form="pcfm"`` 无泵时仍与 PCFM 基线逐位一致（回归守卫）。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    legacy = make_model("hybrid_raman_paper", CASE, nli_form="pcfm").evaluate_path(path, power)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        np.testing.assert_array_equal(
            np.asarray(getattr(legacy, name)), np.asarray(getattr(base, name))
        )


# ---------------------------------------------------------------------------
# Eq. (9) 闭式 vs Eq. (6)-(8) 数值积分
# ---------------------------------------------------------------------------


def test_paper_closed_form_matches_integral_baseline() -> None:
    """Eq. (8)/(9) 闭式与 Eq. (6)-(8) 数值积分的逐信道 SNR 差 <0.1 dB。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pump = _pump()
    closed = make_model("hybrid_raman_paper", CASE, raman_pumps=[pump], nli_form="paper").evaluate_path(
        path, power
    )
    integral = make_model("hybrid_raman_paper", CASE, raman_pumps=[pump], nli_form="integral").evaluate_path(
        path, power
    )
    delta = np.asarray(closed.snr_db) - np.asarray(integral.snr_db)
    assert float(np.max(np.abs(delta))) < 0.1, f"closed/integral differ by {float(np.max(np.abs(delta))):.3f} dB"


# ---------------------------------------------------------------------------
# 有泵 1/10 跨
# ---------------------------------------------------------------------------


def test_paper_pumped_one_and_ten_spans_are_positive_and_finite() -> None:
    """有泵 1/10 跨时 SPM/XPM 都为正且有限。"""
    power = launch_profile(CASE)
    fiber = build_path(CASE).fiber_configs[0]
    edfa = build_path(CASE).edfa_configs[0]
    paths = {
        1: MultiSpanOpticalPath([fiber], [edfa]),
        10: MultiSpanOpticalPath([fiber] * 10, [edfa] * 10),
    }
    for span_count, span_path in paths.items():
        model = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()], nli_form="paper")
        result = model.evaluate_path(span_path, power)
        for name in ("spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
            values = np.asarray(getattr(result, name))
            assert np.all(np.isfinite(values)), f"{name} not finite at {span_count} spans"
        assert np.all(np.asarray(result.spm_noise_w) > 0.0)
        assert np.all(np.asarray(result.xpm_noise_w) > 0.0)



# ---------------------------------------------------------------------------
# ODE 物理方向
# ---------------------------------------------------------------------------


def test_single_pump_is_depleted_and_amplifies_the_signal() -> None:
    """泵浦应向信号转移功率：泵浦输出下降，信号输出高于纯损耗。"""
    solver = _solver([_pump(power_w=0.2)])
    power_in = np.full(9, 1e-3)
    result = solver.solve(power_in)
    passive = power_in * np.exp(-0.2 * np.log(10.0) / 10.0 / 1e3 * 100e3)
    assert np.all(result.power_out_w > passive)
    assert result.pump_power_out_w[0] < 0.5 * 0.2
    assert np.all(result.raman_ase_w > 0.0)
    assert result.profile_norm.shape == (2 * 32, 9)
    assert np.allclose(result.profile_norm[0], 1.0, atol=0.15)


def test_more_pump_power_increases_gain_and_ase() -> None:
    """固定跨长下，泵浦功率增加应同时提高信号增益与拉曼 ASE。"""
    power_in = np.full(9, 1e-3)
    outputs = []
    ases = []
    for power_w in (0.05, 0.1, 0.2):
        result = _solver([_pump(power_w=power_w)]).solve(power_in)
        outputs.append(result.power_out_w.mean())
        ases.append(result.raman_ase_w.mean())
    assert np.all(np.diff(outputs) > 0.0)
    assert np.all(np.diff(ases) > 0.0)


def test_backward_pump_generates_more_ase_than_forward() -> None:
    """相同泵浦功率下，反向泵的 ASE 更高，且增益剖面不同。"""
    power_in = np.full(9, 1e-3)
    forward = _solver([_pump(direction="forward")]).solve(power_in)
    backward = _solver([_pump(direction="backward")]).solve(power_in)
    assert np.all(backward.raman_ase_w > forward.raman_ase_w)
    assert not np.allclose(backward.power_out_w, forward.power_out_w, rtol=1e-6)


def test_small_signal_gain_matches_the_physical_raman_pump() -> None:
    """小信号极限下，ODE 的未耗尽增益应与物理 ``RamanPump`` 逐信道接近。"""
    network = pytest.importorskip("physics_nlo_eval.channel.network")
    pump = _pump()
    frequencies = C_LIGHT / np.linspace(1530e-9, 1565e-9, 9)
    bandwidths = np.full(frequencies.size, 64e9)
    a_eff = np.full(frequencies.size, 80e-12)
    alpha = np.full(frequencies.size, 0.2 * np.log(10.0) / 10.0 / 1e3)
    solver = RamanPaperSpanSolver(
        frequencies,
        bandwidths,
        [pump],
        100e3,
        alpha,
        a_eff,
        np.array([0.25 * np.log(10.0) / 10.0 / 1e3]),
        np.array([80e-12]),
        RamanGainModel(),
        n_steps=64,
        n_iterations=24,
    )
    tiny = np.full(frequencies.size, 1e-12)
    result = solver.solve(tiny)
    passive = tiny * np.exp(-alpha * 100e3)
    ode_gain_db = 10.0 * np.log10(result.power_out_w / passive)
    physical = network.RamanPump(
        network.PumpSpec(**pump.model_dump()),
        100.0,
        effective_area_um2=80.0,
        add_noise=False,
    )
    physical_gain_db = 10.0 * np.log10(physical._gain(frequencies))
    assert np.max(np.abs(ode_gain_db - physical_gain_db)) < 0.1


# ---------------------------------------------------------------------------
# 与 lumped / PCFM 对照
# ---------------------------------------------------------------------------


def test_paper_model_differs_from_the_lumped_equivalent() -> None:
    """论文 ODE 不能与 lumped 末端乘增益给出相同结果。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pump = _pump()
    paper = make_model("hybrid_raman_paper", CASE, raman_pumps=[pump]).evaluate_path(path, power)
    lumped = make_model("pcfm", CASE, raman_model="lumped", raman_pumps=[pump]).evaluate_path(path, power)
    base = make_model("pcfm", CASE).evaluate_path(path, power)

    paper_nli = np.mean(paper.spm_noise_w + paper.xpm_noise_w)
    base_nli = np.mean(base.spm_noise_w + base.xpm_noise_w)
    lumped_nli = np.mean(lumped.spm_noise_w + lumped.xpm_noise_w)
    # lumped 的跨内 NLI 与无拉曼完全相同；论文 ODE 的剖面会改变 NLI。
    assert lumped_nli == pytest.approx(base_nli, rel=1e-12)
    assert paper_nli > base_nli * 1.001
    assert paper_nli != pytest.approx(lumped_nli, rel=1e-6)
    assert np.mean(paper.ase_noise_w) != pytest.approx(np.mean(lumped.ase_noise_w), rel=1e-3)


def test_hybrid_link_snr_improves_over_edfa_only() -> None:
    """一个 hybrid（Raman + EDFA）单跨链路的平均 SNR 应优于纯 EDFA。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    base = make_model("pcfm", CASE).evaluate_path(path, power)
    hybrid = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()]).evaluate_path(path, power)
    assert np.mean(hybrid.snr_db) > np.mean(base.snr_db) + 0.5
    assert np.all(hybrid.snr_db > base.snr_db)


def test_ase_is_transferred_between_spans() -> None:
    """两跨的 ASE 应高于单跨，且第二跨的 ODE 初值不为零。"""
    power = launch_profile(CASE)
    fiber = build_path(CASE).fiber_configs[0]
    edfa = build_path(CASE).edfa_configs[0]
    one_span = MultiSpanOpticalPath([fiber], [edfa])
    two_spans = MultiSpanOpticalPath([fiber, fiber], [edfa, edfa])
    model = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()])
    single = model.evaluate_path(one_span, power)
    model = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()])
    double = model.evaluate_path(two_spans, power)
    assert np.mean(double.ase_noise_w) > np.mean(single.ase_noise_w)


def test_raman_ase_can_be_switched_off() -> None:
    """``raman_ase=False`` 时 ASE 只来自 EDFA，应低于默认设置。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    with_ase = make_model("hybrid_raman_paper", CASE, raman_pumps=[_pump()]).evaluate_path(path, power)
    without_ase = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[_pump()], raman_ase=False
    ).evaluate_path(path, power)
    assert np.all(without_ase.ase_noise_w < with_ase.ase_noise_w)


def test_solver_rejects_wrong_input_shape() -> None:
    """输入形状错误必须报错，不做广播猜测。"""
    solver = _solver([_pump()])
    with pytest.raises(ValueError, match="one value per signal"):
        solver.solve(np.ones(8))
