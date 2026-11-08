"""独立 DRA 数值参考的最小验收测试。

覆盖范围：

* 注册与非法选项；
* 无泵时与论文闭式基线一致（参考自身的回归）；
* 后向/前向泵的 NLI 行为；
* 耦合矩阵的光子数守恒（``C[j,k]/f_j + C[k,j]/f_k = 0``）；
* ``signal_polarizations=2`` 的驱动语义（状态每偏振功率、耦合驱动双偏振总功率）；
* 无 NLI 时与 :class:`RamanPaperSpanSolver` 的 ODE 剖面/泵浦输出一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath, get_model_info
from network_nlo_eval.models.numerical_dra import NumericalDRASpanSolver
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.network.elements import RamanPumpConfig

CASE = build_case("c_57ch")


def _pump(direction: str = "backward", power_w: float = 0.2) -> RamanPumpConfig:
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _one_span(case):
    path = build_path(case)
    return MultiSpanOpticalPath(path.fiber_configs[:1], path.edfa_configs[:1])


# ---------------------------------------------------------------------------
# 耦合矩阵：光子数守恒
# ---------------------------------------------------------------------------


def _two_wave_solver(polarizations: int = 1) -> NumericalDRASpanSolver:
    """无损耗单泵-单信号两波求解器，``A_eff`` 与频率无关。

    守恒判据 ``C[j,k]/f_j + C[k,j]/f_k = 0`` 只在 ``A_eff`` 与频率无关时
    严格成立（矩阵按接收波归一 ``A_eff``），所以这里把两波的 ``A_eff`` 取成
    同一个值。
    """
    return NumericalDRASpanSolver(
        np.array([193.4e12]),
        np.array([64e9]),
        [_pump("forward")],
        10e3,
        np.zeros(1),
        np.full(1, 80e-12),
        np.zeros(1),
        np.full(1, 80e-12),
        RamanGainModel(),
        n_profile=33,
        include_ase=False,
        signal_polarizations=polarizations,
    )


def test_coupling_matrix_conserves_photon_number() -> None:
    """``C[j,k]/f_j + C[k,j]/f_k`` 必须在 1e-12 相对量级上为零。"""
    solver = _two_wave_solver()
    coupling = solver._coupling
    ratio = coupling / solver._f_hz[:, None]
    residual = ratio + ratio.T
    scale = float(np.max(np.abs(ratio)))
    assert scale > 0.0
    assert float(np.max(np.abs(residual))) <= 1.0e-12 * scale


@pytest.mark.parametrize(("polarizations", "expected"), [(1, 1.0), (2, 2.0)])
def test_photon_budget_matches_the_declared_state_semantics(
    polarizations: int, expected: float
) -> None:
    """无损耗两波系统：泵浦光子损失 / 信号光子增益 = 偏振数。

    ``signal_polarizations=1`` 时状态是每信道总功率，比值 1；``=2`` 时状态
    是每偏振功率，物理总信号功率是 2 倍状态，泵浦耗尽翻倍而每偏振增益
    不变，因此比值 2。两条都是守恒的，只是状态口径不同。
    """
    solver = _two_wave_solver(polarizations)
    power_in = np.array([30e-3])
    result = solver.solve(power_in)

    signal_frequency = float(solver._f_signal[0])
    pump_frequency = float(C_LIGHT / (1450.0e-9))
    signal_gain_photons = (float(result.power_out_w[0]) - float(power_in[0])) / signal_frequency
    pump_loss_photons = (0.2 - float(result.pump_power_out_w[0])) / pump_frequency
    assert signal_gain_photons > 0.0
    assert pump_loss_photons / signal_gain_photons == pytest.approx(expected, rel=1.0e-9)


# ---------------------------------------------------------------------------
# 与 RamanPaperSpanSolver 的 ODE 一致性（无 NLI）
# ---------------------------------------------------------------------------


def _solver_fixture(polarizations: int):
    """同一条 80 km 跨，前向 + 后向泵，供两台 ODE 求解器对照。"""
    frequencies = np.array([193.6e12, 193.45e12, 193.3e12])
    bandwidths = np.full(3, 64e9)
    alpha = np.full(3, 0.2 * np.log(10.0) / 10.0 / 1e3)
    a_eff = np.array([80e-12, 82e-12, 84e-12])
    pumps = [_pump("backward"), _pump("forward", power_w=0.1)]
    pump_alpha = np.full(2, 0.25 * np.log(10.0) / 10.0 / 1e3)
    pump_a_eff = np.full(2, 75e-12)
    power_in = np.array([1.5e-3, 1.3e-3, 1.1e-3])
    numerical = NumericalDRASpanSolver(
        frequencies,
        bandwidths,
        pumps,
        80e3,
        alpha,
        a_eff,
        pump_alpha,
        pump_a_eff,
        RamanGainModel(),
        n_profile=129,
        signal_polarizations=polarizations,
    )
    paper = RamanPaperSpanSolver(
        frequencies,
        bandwidths,
        pumps,
        80e3,
        alpha,
        a_eff,
        pump_alpha,
        pump_a_eff,
        RamanGainModel(),
        n_steps=64,
        signal_polarizations=polarizations,
    )
    return numerical, paper, power_in


@pytest.mark.parametrize("polarizations", [1, 2])
def test_ode_profile_and_pump_output_match_the_paper_solver(polarizations: int) -> None:
    """无 NLI 的 ODE 解必须与 ``RamanPaperSpanSolver`` 在 RK4 误差内一致。

    * 输出功率/泵浦输出：DOP853 紧容差 vs 64 步 RK4，相对差 < 1e-3；
    * 归一化剖面：``RamanPaperSpanSolver`` 的采样点从 ``h/2`` 开始，因此
      比较区间取 ``z >= 1 km``，相对差 < 2e-3。
    两台求解器现在都用 ``f_receiver/f_source`` 的光子数守恒负耦合。
    """
    numerical, paper, power_in = _solver_fixture(polarizations)
    numerical_result = numerical.solve(power_in)
    paper_result = paper.solve(power_in)

    assert np.allclose(
        numerical_result.power_out_w, paper_result.power_out_w, rtol=1.0e-3, atol=0.0
    )
    assert np.allclose(
        numerical_result.pump_power_out_w,
        paper_result.pump_power_out_w,
        rtol=1.0e-3,
        atol=0.0,
    )
    # 剖面：paper 的第 i 行在 z = h*(i//2 + 0.5 或 1) 上。
    n_steps = 64
    step = 80e3 / n_steps
    paper_z = np.empty(2 * n_steps, dtype=np.float64)
    paper_z[0::2] = step * (np.arange(n_steps) + 0.5)
    paper_z[1::2] = step * (np.arange(n_steps) + 1.0)

    z_query = np.linspace(1.0e3, 80e3, 40)
    for channel in range(3):
        numerical_profile = np.interp(
            z_query, numerical_result.z_m, numerical_result.profile_norm[:, channel]
        )
        paper_profile = np.interp(
            z_query, paper_z, paper_result.profile_norm[:, channel]
        )
        relative = np.abs(numerical_profile / paper_profile - 1.0)
        assert float(np.max(relative)) < 2.0e-3


# ---------------------------------------------------------------------------
# 模型级行为
# ---------------------------------------------------------------------------


def test_model_is_registered_as_a_numerical_reference() -> None:
    """The reference must be registered as a numerical model."""
    info = get_model_info("numerical_dra")
    assert info.name == "numerical_dra"
    assert "numerical" in info.summary.lower()
    assert "closed" not in info.summary.lower()
    assert any(name == "signal_polarizations" for name, _ in info.options)


def test_no_pump_matches_the_paper_baseline() -> None:
    """无泵时独立参考必须与论文闭式基线一致，否则参考本身有问题。"""
    path = build_path(CASE)
    power = launch_profile(CASE)
    reference = make_model("numerical_dra", CASE).evaluate_path(path, power)
    paper = make_model("hybrid_raman_paper", CASE, nli_form="paper").evaluate_path(path, power)
    delta = np.asarray(reference.snr_db) - np.asarray(paper.snr_db)
    assert float(np.max(np.abs(delta))) < 0.05, f"max |delta|={float(np.max(np.abs(delta))):.3f} dB"


def test_backward_pump_matches_the_full_paper_model() -> None:
    """A backward pump must match the full paper model within the reference error."""
    case = build_case("c_57ch")
    path = _one_span(case)
    power = launch_profile(case)
    reference = make_model("numerical_dra", case, raman_pumps=[_pump()]).evaluate_path(path, power)
    paper = make_model("hybrid_raman_paper", case, raman_pumps=[_pump()], nli_form="paper").evaluate_path(path, power)
    delta = np.asarray(reference.snr_db) - np.asarray(paper.snr_db)
    assert float(np.max(np.abs(delta))) < 0.1, f"max |delta|={float(np.max(np.abs(delta))):.3f} dB"


def test_forward_pump_changes_the_nli() -> None:
    """前向泵把增益放在跨首，NLI 必须明显高于无泵；这是 lumped 抓不到的。"""
    case = build_case("scaling_40ch")
    path = _one_span(case)
    power = launch_profile(case)
    reference = make_model("numerical_dra", case, raman_pumps=[_pump("forward")]).evaluate_path(path, power)
    base = make_model("numerical_dra", case).evaluate_path(path, power)
    nli = np.mean(np.asarray(reference.spm_noise_w) + np.asarray(reference.xpm_noise_w))
    base_nli = np.mean(np.asarray(base.spm_noise_w) + np.asarray(base.xpm_noise_w))
    assert nli > 1.5 * base_nli


def test_signal_polarizations_two_changes_the_pump_depletion() -> None:
    """``signal_polarizations=2`` 必须可用、有限，并加深泵浦耗尽。

    状态仍是每偏振功率，所以输出本身不能被 2 倍放大；变化的只是耦合驱动
    （泵浦看到双偏振总信号功率）。前向 0.2 W 泵跨 100 km 的耗尽足够大，
    可以直接从泵浦出射功率上分辨。
    """
    case = build_case("scaling_40ch")
    path = _one_span(case)
    power = launch_profile(case)
    pump = _pump("forward")

    results = {}
    pump_out = {}
    fiber_out = {}
    for polarizations in (1, 2):
        model = make_model(
            "numerical_dra", case, raman_pumps=[pump], signal_polarizations=polarizations
        )
        result = model.evaluate_path(path, power)
        results[polarizations] = result
        kernel = model.kernel_for(path.fiber_configs[0])
        # 取 EDFA 之前的单跨输出：路由层的信号功率被放大器归一，看不到单跨状态。
        # ``evaluate_path`` 已把该跨算过并缓存，这次调用不会重算。
        fiber_out[polarizations] = np.asarray(
            kernel.evaluate(power, path.fiber_configs[0].length_km * 1e3).power_out_w,
            dtype=np.float64,
        )
        pump_out[polarizations] = np.asarray(kernel.last_pump_power_out_w, dtype=np.float64)
        for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
            assert np.all(np.isfinite(np.asarray(getattr(result, name)))), name

    # 双偏振驱动让泵浦耗尽更快，出射泵浦功率必须明显更低。
    assert float(pump_out[2][0]) < 0.5 * float(pump_out[1][0])
    # 状态本身没有被 2 倍放大：出纤信号功率比在 (0.3, 1) 内，变化只来自
    # 更快的泵浦耗尽（若状态被放大，这里的比值会 > 1）。
    signal_ratio = fiber_out[2] / fiber_out[1]
    assert float(np.min(signal_ratio)) > 0.3
    assert float(np.max(signal_ratio)) < 1.0
    # 跨内增益更低 -> NLI 更低。
    nli_1 = np.asarray(results[1].spm_noise_w) + np.asarray(results[1].xpm_noise_w)
    nli_2 = np.asarray(results[2].spm_noise_w) + np.asarray(results[2].xpm_noise_w)
    assert float(np.mean(nli_2) / np.mean(nli_1)) < 0.9


def test_invalid_profile_is_rejected() -> None:
    """An even/too-small longitudinal grid must be rejected."""
    with pytest.raises(ValueError, match="n_profile"):
        make_model("numerical_dra", CASE, n_profile=4)


def test_invalid_signal_polarizations_is_rejected() -> None:
    """``signal_polarizations`` 必须 >= 1。"""
    with pytest.raises(ValueError, match="signal_polarizations"):
        make_model("numerical_dra", CASE, signal_polarizations=0)
