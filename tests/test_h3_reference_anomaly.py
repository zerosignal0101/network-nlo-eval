"""无泵 S/L 边缘“Raman ASE 异常”的回归测试（任务 T）。

第五轮精度矩阵记录：S+C+L 100 GHz、6 dBm/信道、1 跨、203 THz 处
``numerical_dra`` 的总 ASE 比 ``pcfm_h3_raman``/``numerical`` 高约 19 dB
（3.86e-3 W vs 5.43e-5 W），并据此建议无泵点改用 ``numerical`` 作主参考。
本文件把该异常的**真实根因**钉住，防止未来再次把它读成“拉曼 ASE 源项 bug”。

根因链（逐项由下面的测试断言）：

1. 无泵时 ``hybrid_raman_paper``/``numerical_dra`` 的 ODE 仍产生自发拉曼源项，
   但量级只有 ``~1e-18 W``（203 THz）/ ``~5e-8 W``（全带），**不是** 19 dB 的来源；
   ``pcfm_h3_raman``/``numerical`` 在无泵时根本不进入拉曼 ODE，报零。
2. 真正相差 18.5 dB 的是**跨末信号功率**：两个 ODE 模型在 203 THz 的跨损
   约 51 dB，而 ``pcfm_h3_raman``/``numerical`` 约 32.5 dB。这是两种拉曼增益谱
   （``RamanGainModel`` Gaussian，峰值 6e-14 m/W；Semrau Table I(b) 代理谱，
   斜率 0.028 1/(W km THz)）在 21 THz 带宽上的 ISRS 倾斜之差。
3. 路由层的 EDFA ``target_gain_db=None`` 按 ``P_in/P_out_fiber`` 自动增益，
   于是 18.5 dB 的跨损差被原样转成 18.5 dB 的 EDFA ASE 差——矩阵里记录的
   ``ase_noise_w`` 是**总 ASE**，不是新产生的拉曼 ASE。
4. 把同一 ODE 的拉曼谱峰值调到 Semrau 量级后，跨损与 ASE 都回到
   ``pcfm_h3_raman`` 附近，证明这是谱标定/参考面差异，不是 ASE 记账错误。

参考选择规则（由 ``test_unpumped_reference_rule_*`` 守卫）：无泵点的主参考
用 ``numerical``（或同族的 ``pcfm_h3_raman``），**不要**用
``numerical_dra``/``hybrid_raman_paper``；后两者的无泵 S/L 边缘绝对倾斜带着
Gaussian 谱标定，只适合做“同一谱模型内部”的对照。

为了在几秒内跑完，本文件只解**功率/ASE ODE**，不计算 SPM/XPM 的
NLI 求积——NLI 与 ASE 记账相互独立，异常完全落在 ASE 一侧。
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.benchmarks.cases import BenchCase, band_edges_hz, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import SpanFiberOutput
from network_nlo_eval.models.numba_kernels import _calc_aeff_dynamic, _calc_fiber_attenuation_npm
from network_nlo_eval.models.numerical_dra import NumericalDRASpanSolver
from network_nlo_eval.models.numerical_kernels import _solve_isrs_profile_jit
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.models.route import apply_amplifier_and_roadm

#: 第五轮矩阵里的异常点：S+C+L、100 GHz、6 dBm/信道、1 跨 100 km。
SPACING_HZ = 100e9
TOTAL_CHANNELS = 211
POWER_DBM = 6.0
SPAN_LENGTH_M = 100e3
S_EDGE_HZ = 203.0e12
#: 第五轮记录的异常量级（比较基准，不是“正确值”）。
ANOMALOUS_ASE_W = 3.86e-3
REFERENCE_ASE_W = 5.43e-5


def _repro_case() -> BenchCase:
    """重建矩阵点，不依赖 ``data/``/``results/``。"""
    low, high = band_edges_hz("S+C+L")
    return BenchCase(
        name="h3_reference_anomaly",
        bands="S+C+L",
        num_channels=TOTAL_CHANNELS,
        channel_spacing_hz=SPACING_HZ,
        center_frequency_hz=(low + high) / 2.0,
        symbol_rate_hz=64e9,
        span_length_km=SPAN_LENGTH_M / 1e3,
        span_count=1,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


CASE = _repro_case()
PATH = build_path(CASE)
FIBER = PATH.fiber_configs[0]
EDFA = PATH.edfa_configs[0]
POWER_W = launch_profile(CASE, POWER_DBM)
FREQUENCIES_HZ = CASE.center_frequency_hz + (
    np.arange(CASE.num_channels) - (CASE.num_channels - 1) / 2.0
) * SPACING_HZ
#: 模型顺序（升频）下 203 THz 的下标。
EDGE = int(np.argmin(np.abs(FREQUENCIES_HZ - S_EDGE_HZ)))
#: 核内部按降序保存频率，203 THz 的镜像下标。
EDGE_DESC = CASE.num_channels - 1 - EDGE


def _fiber_alpha() -> np.ndarray:
    """与核一致的逐信道衰减 (1/m)，降序。"""
    lambdas_m = C_LIGHT / FREQUENCIES_HZ[::-1]
    return _calc_fiber_attenuation_npm(
        lambdas_m,
        FIBER.reference_wavelength_nm,
        FIBER.attenuation_alpha2,
        FIBER.attenuation_alpha1,
        FIBER.attenuation_alpha0,
    )


def _fiber_a_eff() -> np.ndarray:
    """与核一致的逐信道有效模面积 (m²)，降序。"""
    return _calc_aeff_dynamic(
        C_LIGHT / FREQUENCIES_HZ[::-1],
        FIBER.effective_area_um2_ref,
        FIBER.aeff_slope_um2_nm,
        FIBER.reference_wavelength_nm,
    )


def _hybrid_span(*, peak_m_per_w: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``hybrid_raman_paper`` 的跨末功率与新产生 ASE，模型顺序。"""
    gain_model = RamanGainModel(
        peak_gain_coefficient_m_per_w=(
            RamanGainModel().peak_gain_coefficient_m_per_w if peak_m_per_w is None else peak_m_per_w
        )
    )
    solver = RamanPaperSpanSolver(
        np.ascontiguousarray(FREQUENCIES_HZ[::-1]),
        np.full(CASE.num_channels, CASE.symbol_rate_hz),
        [],
        SPAN_LENGTH_M,
        _fiber_alpha(),
        _fiber_a_eff(),
        np.zeros(0),
        np.zeros(0),
        gain_model,
    )
    result = solver.solve(np.ascontiguousarray(POWER_W[::-1]))
    return result.power_out_w[::-1], result.raman_ase_w[::-1]


def _h3_span() -> np.ndarray:
    """``pcfm_h3_raman`` 无泵时的跨末功率（只走 ISRS 剖面），模型顺序。"""
    model = make_model("pcfm_h3_raman", CASE, egn="auto")
    kernel = model.kernel_for(FIBER)
    _, power_out = kernel._integrate_spp(
        np.ascontiguousarray(POWER_W[kernel.order]), SPAN_LENGTH_M
    )
    return np.asarray(power_out)[kernel.inverse_order]


def _numerical_span() -> np.ndarray:
    """``numerical`` 的跨末功率（Semrau ISRS 代理谱），模型顺序。"""
    model = make_model("numerical", CASE)
    kernel = model.kernel_for(FIBER)
    _, power_out = _solve_isrs_profile_jit(
        kernel.f_abs_hz,
        np.ascontiguousarray(POWER_W[kernel.order]),
        kernel.alpha_npm,
        SPAN_LENGTH_M,
        kernel.ode_steps,
    )
    return np.asarray(power_out)[np.argsort(kernel.order)]


def _numerical_dra_span() -> tuple[np.ndarray, np.ndarray]:
    """``numerical_dra`` 的单跨 ODE（跳过 NLI），模型顺序。"""
    solver = NumericalDRASpanSolver(
        np.ascontiguousarray(FREQUENCIES_HZ[::-1]),
        np.full(CASE.num_channels, CASE.symbol_rate_hz),
        [],
        SPAN_LENGTH_M,
        _fiber_alpha(),
        _fiber_a_eff(),
        np.zeros(0),
        np.zeros(0),
        RamanGainModel(),
    )
    result = solver.solve(np.ascontiguousarray(POWER_W[::-1]))
    return result.power_out_w[::-1], result.raman_ase_w[::-1]


def _route_ase(power_out_w: np.ndarray, raman_ase_w: np.ndarray | None) -> np.ndarray:
    """把跨末功率与拉曼 ASE 交给共享 route 层，返回 EDFA 后的总 ASE (W)。"""
    element = apply_amplifier_and_roadm(
        power_in_w=POWER_W,
        fiber_output=SpanFiberOutput(
            power_out_w=power_out_w,
            spm_noise_w=np.zeros(CASE.num_channels),
            xpm_noise_w=np.zeros(CASE.num_channels),
        ),
        edfa_config=EDFA,
        roadm_config=None,
        nf_lin=np.full(CASE.num_channels, 10.0 ** (CASE.edfa_noise_figure_db / 10.0)),
        f_abs_hz=FREQUENCIES_HZ,
        bandwidth_hz=np.full(CASE.num_channels, CASE.symbol_rate_hz),
        input_ase_w=raman_ase_w,
    )
    return np.asarray(element.ase_noise_w)


def _loss_db(power_out_w: np.ndarray) -> float:
    """203 THz 处跨末（EDFA 前）的功率损耗 (dB)。"""
    return float(10.0 * np.log10(POWER_W[EDGE] / power_out_w[EDGE]))


def test_unpumped_span_raman_ase_is_negligible_at_the_s_edge() -> None:
    """无泵 ODE 确有自发源项，但 203 THz 只有 ~1e-18 W，不是 19 dB 异常来源。"""
    _, hybrid_raman = _hybrid_span()
    assert hybrid_raman[EDGE] < 1.0e-12
    assert hybrid_raman.sum() < 1.0e-6

    _, numdra_raman = _numerical_dra_span()
    assert numdra_raman[EDGE] < 1.0e-12
    assert numdra_raman.sum() < 1.0e-6
    # 两个 ODE 的自发源项一致（同一谱、同一温度）。
    assert abs(hybrid_raman.sum() / numdra_raman.sum() - 1.0) < 0.05

    # 同族模型无泵时不进入拉曼 ODE，直接报零；这是记录口径，不是物理差异。
    model = make_model("pcfm_h3_raman", CASE, egn="auto")
    kernel = model.kernel_for(FIBER)
    kernel.evaluate(POWER_W, SPAN_LENGTH_M)
    assert np.array_equal(np.asarray(kernel.last_raman_ase_w), np.zeros(CASE.num_channels))


def test_s_edge_gap_is_isrs_tilt_not_raman_ase() -> None:
    """203 THz 的 18.5 dB 差在跨末信号功率上，与拉曼 ASE 源项无关。"""
    hybrid_out, _ = _hybrid_span()
    numdra_out, _ = _numerical_dra_span()
    h3_out = _h3_span()
    numerical_out = _numerical_span()

    # 两个 ODE 模型（同一谱、不同积分器）一致；闭式族彼此一致。
    assert abs(_loss_db(hybrid_out) - _loss_db(numdra_out)) < 0.2
    assert abs(_loss_db(h3_out) - _loss_db(numerical_out)) < 0.2
    # 异常量级：18–19 dB，且是 ODE 族更深。
    gap = _loss_db(hybrid_out) - _loss_db(h3_out)
    assert 15.0 < gap < 22.0
    assert _loss_db(hybrid_out) > 45.0
    assert _loss_db(h3_out) < 38.0


def test_edfa_auto_gain_turns_the_tilt_gap_into_an_ase_gap() -> None:
    """``gain = P_in/P_out_fiber`` 把跨损差原样搬到 EDFA ASE 上。"""
    hybrid_out, hybrid_raman = _hybrid_span()
    h3_out = _h3_span()
    hybrid_ase = _route_ase(hybrid_out, hybrid_raman)
    h3_ase = _route_ase(h3_out, None)

    loss_gap_db = _loss_db(hybrid_out) - _loss_db(h3_out)
    ase_gap_db = float(10.0 * np.log10(hybrid_ase[EDGE] / h3_ase[EDGE]))
    assert abs(ase_gap_db - loss_gap_db) < 0.2
    # 把第五轮的数字钉住：参考 ~5.4e-5 W，异常 ~3.9e-3 W。
    assert 0.5 * REFERENCE_ASE_W < h3_ase[EDGE] < 1.6 * REFERENCE_ASE_W
    assert 0.6 * ANOMALOUS_ASE_W < hybrid_ase[EDGE] < 1.5 * ANOMALOUS_ASE_W


def test_aligned_raman_spectrum_removes_the_gap() -> None:
    """把同一 ODE 的谱峰值换到 Semrau 量级，跨损与 ASE 都回到参考附近。"""
    semrau_peak = 2.33e-14  # 2.8e-17 * 12.34 THz * exp(-1/8) * A_eff(76.3 um^2)
    hybrid_out, _ = _hybrid_span()
    aligned_out, aligned_raman = _hybrid_span(peak_m_per_w=semrau_peak)
    h3_out = _h3_span()

    assert _loss_db(aligned_out) - _loss_db(h3_out) < 5.0
    assert _loss_db(hybrid_out) - _loss_db(aligned_out) > 10.0

    aligned_ase = _route_ase(aligned_out, aligned_raman)
    h3_ase = _route_ase(h3_out, None)
    assert abs(float(10.0 * np.log10(aligned_ase[EDGE] / h3_ase[EDGE]))) < 5.0


def test_unpumped_reference_rule_prefers_numerical_over_the_dra_ode() -> None:
    """无泵主参考规则：``numerical`` 与 H3 同量级，DRA ODE 族偏离 >10 dB。"""
    numerical_ase = _route_ase(_numerical_span(), None)
    h3_ase = _route_ase(_h3_span(), None)
    hybrid_out, hybrid_raman = _hybrid_span()
    hybrid_ase = _route_ase(hybrid_out, hybrid_raman)

    assert abs(float(10.0 * np.log10(h3_ase[EDGE] / numerical_ase[EDGE]))) < 0.5
    assert float(10.0 * np.log10(hybrid_ase[EDGE] / numerical_ase[EDGE])) > 10.0


def test_solver_reports_only_the_newly_generated_raman_ase() -> None:
    """输入 ASE 不得被当成“新产生”的量：``raman_ase_w`` 与输入无关。"""
    frequencies = np.array([196.0e12, 195.0e12, 194.0e12, 193.0e12])
    bandwidths = np.full(4, 64e9)
    alpha = np.full(4, 0.2 * np.log(10.0) / 10.0 / 1e3)
    a_eff = np.full(4, 80e-12)
    power = np.full(4, 2e-3)
    ase_in = np.array([1e-8, 2e-8, 3e-8, 4e-8])

    solver = RamanPaperSpanSolver(
        frequencies,
        bandwidths,
        [],
        SPAN_LENGTH_M,
        alpha,
        a_eff,
        np.zeros(0),
        np.zeros(0),
        RamanGainModel(),
        n_steps=64,
    )
    without = solver.solve(power)
    with_input = solver.solve(power, ase_in)
    # 新产生量只差 ASE 参与受激拉曼的二阶项；若输入被重复计入，
    # 差值会是 ~1e-8 W 量级。
    assert np.max(np.abs(with_input.raman_ase_w - without.raman_ase_w)) < 1.0e-11
