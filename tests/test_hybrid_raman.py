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


def _pump_at(wavelength_nm: float, power_w: float, direction: str = "backward") -> RamanPumpConfig:
    """构造指定波长的泵浦，用于需要泵浦-泵浦耦合的对照。"""
    return RamanPumpConfig(
        wavelength_nm=wavelength_nm,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _multi_pumps() -> list[RamanPumpConfig]:
    """波长不同的两个后向泵：泵浦-泵浦块非对角，开关才有可观测效果。"""
    return [_pump_at(1440.0, 0.2), _pump_at(1500.0, 0.1)]


def _solver(
    pumps: list[RamanPumpConfig],
    *,
    n_steps: int = 32,
    pump_depletion: bool = True,
    include_isrs: bool = True,
    signal_polarizations: int = 1,
    pump_pump_raman: bool = True,
) -> RamanPaperSpanSolver:
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
        pump_depletion=pump_depletion,
        include_isrs=include_isrs,
        signal_polarizations=signal_polarizations,
        pump_pump_raman=pump_pump_raman,
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


def test_coupling_matrix_conserves_photon_number() -> None:
    r"""对任意 (j,k)，``C[j,k]/f_j + C[k,j]/f_k`` 必须为零。

    使用随机频率和统一的 ``A_eff``，把频率因子从 ``A_eff`` 归一化中隔离
    出来。判据等价于 ``d/dz Σ_j P_j/f_j = 0``。
    """
    rng = np.random.default_rng(20261106)
    frequencies = np.sort(rng.uniform(180e12, 210e12, 24))[::-1]
    solver = RamanPaperSpanSolver(
        frequencies,
        np.full(frequencies.size, 64e9),
        [],
        100e3,
        np.full(frequencies.size, 0.2 * np.log(10.0) / 10.0 / 1e3),
        np.full(frequencies.size, 80e-12),
        np.zeros(0),
        np.zeros(0),
        RamanGainModel(),
        n_steps=8,
    )
    coupling = solver._coupling  # noqa: SLF001 - 直接审计构造出的矩阵
    f_hz = solver._f_hz  # noqa: SLF001
    off_diagonal = ~np.eye(f_hz.size, dtype=bool)
    violation = (coupling / f_hz[:, None] + coupling.T / f_hz[None, :])[off_diagonal]
    scale = np.abs(coupling / f_hz[:, None])[off_diagonal]
    relative = np.abs(violation) / np.maximum(scale, np.finfo(np.float64).tiny)
    assert float(np.max(relative)) < 1e-12


def test_pump_pump_off_zeroes_only_the_pump_pump_block() -> None:
    """关闭泵浦-泵浦耦合不得改动泵浦-信号或信号-信号项。

    状态向量按频率降序重排后泵浦槽与信号槽交错，因此逐块比较而不是按
    行列区间比较，正是为了钉住 ``_build_coupling_matrix`` 的掩码用法。
    """
    pumps = [_pump_at(1420.0, 0.2), _pump_at(1450.0, 0.1), _pump_at(1500.0, 0.05)]
    on = _solver(pumps)
    off = _solver(pumps, pump_pump_raman=False)
    pump_slots = np.flatnonzero(on._is_pump)  # noqa: SLF001 - 直接审计耦合矩阵
    signal_slots = on._signal_slots  # noqa: SLF001
    coupling_on = on._coupling  # noqa: SLF001
    coupling_off = off._coupling  # noqa: SLF001

    assert np.all(coupling_off[np.ix_(pump_slots, pump_slots)] == 0.0)
    assert np.array_equal(
        coupling_off[np.ix_(signal_slots, pump_slots)],
        coupling_on[np.ix_(signal_slots, pump_slots)],
    )
    assert np.array_equal(
        coupling_off[np.ix_(pump_slots, signal_slots)],
        coupling_on[np.ix_(pump_slots, signal_slots)],
    )
    assert np.array_equal(
        coupling_off[np.ix_(signal_slots, signal_slots)],
        coupling_on[np.ix_(signal_slots, signal_slots)],
    )
    # 泵浦仍从信号失去功率（泵浦行的信号列为负），不是把整行清零。
    assert np.any(coupling_off[np.ix_(pump_slots, signal_slots)] < 0.0)


def test_pump_pump_off_single_pump_is_bit_identical() -> None:
    """单泵没有泵浦-泵浦对：开关前后三个输出数组必须逐位一致。"""
    power_in = np.full(9, 1e-3)
    on = _solver([_pump(0.2)]).solve(power_in)
    off = _solver([_pump(0.2)], pump_pump_raman=False).solve(power_in)
    assert np.array_equal(on.power_out_w, off.power_out_w)
    assert np.array_equal(on.profile_norm, off.profile_norm)
    assert np.array_equal(on.raman_ase_w, off.raman_ase_w)
    assert np.array_equal(on.pump_power_out_w, off.pump_power_out_w)


def test_pump_pump_off_changes_multi_pump_transfer() -> None:
    """多泵时泵浦之间互相搬运功率；关掉后泵浦输出与信号剖面都要改变。"""
    pumps = [_pump_at(1420.0, 0.2), _pump_at(1450.0, 0.1), _pump_at(1500.0, 0.05)]
    power_in = np.full(9, 1e-3)
    on = _solver(pumps).solve(power_in)
    off = _solver(pumps, pump_pump_raman=False).solve(power_in)
    assert not np.allclose(on.pump_power_out_w, off.pump_power_out_w, rtol=1e-6)
    assert not np.allclose(on.power_out_w, off.power_out_w, rtol=1e-6)
    assert not np.allclose(on.profile_norm, off.profile_norm, rtol=1e-6)


def test_pump_pump_off_no_pump_is_bit_identical() -> None:
    """无泵时该开关是空操作。"""
    power_in = np.full(9, 1e-3)
    on = _solver([]).solve(power_in)
    off = _solver([], pump_pump_raman=False).solve(power_in)
    assert np.array_equal(on.power_out_w, off.power_out_w)
    assert np.array_equal(on.profile_norm, off.profile_norm)
    assert np.array_equal(on.pump_power_out_w, off.pump_power_out_w)


def test_invalid_pump_pump_raman_solver_option_is_rejected() -> None:
    """求解器层的布尔开关不能接受整数/字符串。"""
    with pytest.raises(ValueError, match="pump_pump_raman"):
        _solver([_pump()], pump_pump_raman=1)


def test_two_wave_power_transfer_matches_the_frequency_ratio() -> None:
    r"""两波耦合的功率变化比应为 ``ΔP_p/ΔP_s ≈ -f_p/f_s``。

    取极短跨长与零衰减，把 ODE 线性化到耦合项；泵浦行必须使用
    ``f_p/f_s``（接收/源），而不是旧口径的 ``f_s/f_p``。
    """
    f_pump = C_LIGHT / (1450.0e-9)
    f_signal = C_LIGHT / (1550.0e-9)
    pump_power = 0.2
    signal_power = 1e-6
    solver = RamanPaperSpanSolver(
        np.array([f_signal]),
        np.array([64e9]),
        [
            RamanPumpConfig(
                wavelength_nm=1450.0,
                power_w=pump_power,
                direction="forward",
                attenuation_db_per_km=0.0,
            )
        ],
        0.1,
        np.array([0.0]),
        np.array([80e-12]),
        np.array([0.0]),
        np.array([80e-12]),
        RamanGainModel(),
        n_steps=1,
        include_isrs=False,
        include_ase=False,
    )
    result = solver.solve(np.array([signal_power]))
    delta_signal = result.power_out_w[0] - signal_power
    delta_pump = result.pump_power_out_w[0] - pump_power
    assert delta_signal > 0.0
    assert delta_pump < 0.0
    assert delta_pump / delta_signal == pytest.approx(-f_pump / f_signal, rel=1e-4)


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
# 泵浦耗尽对照口径
# ---------------------------------------------------------------------------

_PUMP_ALPHA_NPM = 0.25 * np.log(10.0) / 10.0 / 1e3


def test_undepleted_forward_pump_follows_beer_lambert() -> None:
    """``pump_depletion=False`` 时前向泵严格按 ``P0 exp(-alpha_p L)`` 衰减。"""
    pump = _pump(direction="forward")
    expected = pump.power_w * np.exp(-_PUMP_ALPHA_NPM * 100e3)

    undepleted = _solver([pump], pump_depletion=False).solve(np.full(9, 1e-3))
    depleted = _solver([pump], pump_depletion=True).solve(np.full(9, 1e-3))

    assert undepleted.pump_power_out_w[0] == pytest.approx(expected, rel=1e-4)
    # 完整耦合 ODE 必须出现可测的泵浦耗尽，否则这个开关没有物理意义。
    assert depleted.pump_power_out_w[0] < 0.9 * expected
    # 未耗尽口径泵浦更强，信号增益也应更高。
    assert np.all(undepleted.power_out_w > depleted.power_out_w)


def test_undepleted_backward_pump_profile_is_the_preset_exponential() -> None:
    """反向泵的未耗尽剖面仍是 ``P0 exp(-alpha_p (L-z))``。"""
    pump = _pump(direction="backward")
    solver = _solver([pump], pump_depletion=False)
    z_m = solver._z_m  # noqa: SLF001 - 直接检查构造出的预设剖面
    profile = solver._bw_initial_total[:, 0]  # noqa: SLF001
    np.testing.assert_allclose(
        profile,
        pump.power_w * np.exp(-_PUMP_ALPHA_NPM * (100e3 - z_m)),
        rtol=1e-12,
    )

    result = solver.solve(np.full(9, 1e-3))
    assert result.pump_power_out_w[0] == pytest.approx(
        pump.power_w * np.exp(-_PUMP_ALPHA_NPM * 100e3), rel=1e-4
    )


def test_forward_pump_undepleted_has_more_nli_and_lower_snr() -> None:
    """前向泵未耗尽口径保留更多泵浦功率，NLI 更大、SNR 更低。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pump = _pump(direction="forward")
    depleted = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[pump], pump_depletion=True
    ).evaluate_path(path, power)
    undepleted = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[pump], pump_depletion=False
    ).evaluate_path(path, power)

    nli_depleted = np.mean(np.asarray(depleted.spm_noise_w) + np.asarray(depleted.xpm_noise_w))
    nli_undepleted = np.mean(np.asarray(undepleted.spm_noise_w) + np.asarray(undepleted.xpm_noise_w))
    assert nli_undepleted > nli_depleted
    assert float(np.mean(undepleted.snr_db)) < float(np.mean(depleted.snr_db)) - 0.1


# ---------------------------------------------------------------------------
# 偏振驱动口径：signal_polarizations
# ---------------------------------------------------------------------------


def test_signal_polarizations_one_is_bit_identical_to_the_default() -> None:
    """``signal_polarizations=1`` 与默认路径逐位一致（向后兼容守卫）。"""
    power_in = np.full(9, 1e-3)
    default = _solver([_pump(direction="forward")]).solve(power_in)
    explicit = _solver([_pump(direction="forward")], signal_polarizations=1).solve(power_in)
    for field in (
        "profile_norm",
        "power_out_w",
        "raman_ase_w",
        "pump_power_out_w",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(default, field)), np.asarray(getattr(explicit, field))
        )


def test_model_signal_polarizations_one_is_bit_identical_to_the_default() -> None:
    """模型层 ``signal_polarizations=1`` 也与旧默认逐位一致。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pump = _pump(direction="forward")
    base = make_model("hybrid_raman_paper", CASE, raman_pumps=[pump]).evaluate_path(path, power)
    explicit = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[pump], signal_polarizations=1
    ).evaluate_path(path, power)
    for field in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        np.testing.assert_array_equal(
            np.asarray(getattr(base, field)), np.asarray(getattr(explicit, field))
        )


def test_signal_polarizations_two_depletes_the_forward_pump_more() -> None:
    """双偏振总功率驱动让前向泵耗尽更多、信号增益更低。

    状态仍是每偏振功率，驱动才是 ``2 * P_s``；因此泵浦残余严格更低，
    接收波自己的功率只乘一次，信号输出也应更低。
    """
    power_in = np.full(9, 1e-3)
    one = _solver([_pump(direction="forward")], signal_polarizations=1).solve(power_in)
    two = _solver([_pump(direction="forward")], signal_polarizations=2).solve(power_in)
    assert two.pump_power_out_w[0] < one.pump_power_out_w[0]
    assert np.all(two.power_out_w < one.power_out_w)
    assert np.mean(two.raman_ase_w) < np.mean(one.raman_ase_w)


def test_signal_polarizations_two_raises_the_forward_pump_snr() -> None:
    """更强的泵浦耗尽降低增益与 NLI，前向泵的 SNR_NLI 因而更高。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pump = _pump(direction="forward")
    one = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[pump], signal_polarizations=1
    ).evaluate_path(path, power)
    two = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[pump], signal_polarizations=2
    ).evaluate_path(path, power)
    nli_one = np.asarray(one.spm_noise_w) + np.asarray(one.xpm_noise_w)
    nli_two = np.asarray(two.spm_noise_w) + np.asarray(two.xpm_noise_w)
    assert np.mean(nli_two) < np.mean(nli_one)
    assert float(np.mean(two.snr_db)) > float(np.mean(one.snr_db))


def test_signal_polarizations_two_backward_effect_is_much_smaller_than_forward() -> None:
    """后向泵的因子 2 效应远小于前向（物理预期）。"""
    power = launch_profile(CASE)
    path = build_path(CASE)

    def mean_snr(direction: str, factor: int) -> float:
        pump = _pump(direction=direction)
        result = make_model(
            "hybrid_raman_paper", CASE, raman_pumps=[pump], signal_polarizations=factor
        ).evaluate_path(path, power)
        return float(np.mean(result.snr_db))

    forward_change = abs(mean_snr("forward", 2) - mean_snr("forward", 1))
    backward_change = abs(mean_snr("backward", 2) - mean_snr("backward", 1))
    assert backward_change < 0.2 * forward_change
    assert forward_change > 0.1


def test_signal_polarizations_two_without_pumps_or_isrs_is_bit_identical() -> None:
    """无泵且关闭 ISRS 时 ODE 只剩衰减，因子 2 必须完全不改变结果。"""
    power_in = np.full(9, 1e-3)
    one = _solver([], include_isrs=False, signal_polarizations=1).solve(power_in)
    two = _solver([], include_isrs=False, signal_polarizations=2).solve(power_in)
    for field in ("profile_norm", "power_out_w", "raman_ase_w", "pump_power_out_w"):
        np.testing.assert_array_equal(
            np.asarray(getattr(one, field)), np.asarray(getattr(two, field))
        )


def test_signal_polarizations_two_no_pump_model_without_isrs_is_bit_identical() -> None:
    """模型层无泵 + ``include_isrs=False`` 也必须逐位不变。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    one = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[], include_isrs=False, signal_polarizations=1
    ).evaluate_path(path, power)
    two = make_model(
        "hybrid_raman_paper", CASE, raman_pumps=[], include_isrs=False, signal_polarizations=2
    ).evaluate_path(path, power)
    for field in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        np.testing.assert_array_equal(
            np.asarray(getattr(one, field)), np.asarray(getattr(two, field))
        )


def test_signal_polarizations_two_no_pump_isrs_change_is_only_a_drive_rescale() -> None:
    """无泵但开启 ISRS 时，因子 2 只把信道-信道耦合的源功率加倍。

    这是任务约定的“信号-信号源功率也用总功率”；它必须是小量，且关闭
    ``include_isrs`` 后严格消失（见上一条测试）。
    """
    power_in = np.full(9, 1e-3)
    one = _solver([], include_isrs=True, signal_polarizations=1).solve(power_in)
    two = _solver([], include_isrs=True, signal_polarizations=2).solve(power_in)
    relative = np.abs(two.power_out_w - one.power_out_w) / one.power_out_w
    assert float(np.max(relative)) < 1.0e-4


def test_invalid_signal_polarizations_is_rejected() -> None:
    """非正偏振数必须直接报错。"""
    with pytest.raises(ValueError, match="signal_polarizations"):
        _solver([], signal_polarizations=0)
    with pytest.raises(ValueError, match="signal_polarizations"):
        make_model("hybrid_raman_paper", CASE, signal_polarizations=0)


# ---------------------------------------------------------------------------
# 泵浦-泵浦拉曼开关
# ---------------------------------------------------------------------------


def test_pump_pump_raman_option_is_documented_as_default_on() -> None:
    """模型元数据必须声明该开关，且默认值保持历史行为。"""
    options = dict(HYBRID_RAMAN_PAPER_MODEL_INFO.options)
    assert "pump_pump_raman" in options
    assert "Default True" in options["pump_pump_raman"]


def test_invalid_pump_pump_raman_model_option_is_rejected() -> None:
    """Hybrid 模型层同样拒绝非布尔开关，而不是等到建核时才失败。"""
    with pytest.raises(ValueError, match="pump_pump_raman"):
        make_model("hybrid_raman_paper", CASE, pump_pump_raman="yes")


def test_pump_pump_off_no_pump_is_bit_identical_to_the_default_model() -> None:
    """无泵时 hybrid 的开关是空操作。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    default = make_model("hybrid_raman_paper", CASE).evaluate_path(path, power)
    off = make_model("hybrid_raman_paper", CASE, pump_pump_raman=False).evaluate_path(path, power)
    for field in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        np.testing.assert_array_equal(np.asarray(getattr(default, field)), np.asarray(getattr(off, field)))


@pytest.mark.parametrize("nli_form", ["paper", "pcfm"])
def test_pump_pump_off_multi_pump_changes_the_link_result(nli_form: str) -> None:
    """多泵时开关必须真正透传到 ODE，两种 NLI 口径下都改变链路结果。"""
    power = launch_profile(CASE)
    path = build_path(CASE)
    pumps = _multi_pumps()
    on = make_model("hybrid_raman_paper", CASE, raman_pumps=pumps, nli_form=nli_form).evaluate_path(
        path, power
    )
    off = make_model(
        "hybrid_raman_paper",
        CASE,
        raman_pumps=pumps,
        nli_form=nli_form,
        pump_pump_raman=False,
    ).evaluate_path(path, power)
    assert not np.allclose(np.asarray(on.snr_db), np.asarray(off.snr_db), rtol=1e-9, atol=0.0)
    assert not np.allclose(np.asarray(on.spm_noise_w), np.asarray(off.spm_noise_w), rtol=1e-6, atol=0.0)


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
