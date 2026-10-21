r"""cGN 的 MCI 型图元（两束相互作用波都在被测信道带外）测试。

**背景**

GN 参考式（论文 Eq. (1)）的双重积分域是"三个场各自落在某个信道带内"的
全部三元组 :math:`(c_1, c_2, c_3)`。cGN 的既有实现把第一波钉在被测信道
带内，因此只覆盖 :math:`c_1 = \mathrm{CUT}` 与 :math:`c_2 = \mathrm{CUT}`
两族；**MCI 型图元**（:math:`c_1 \ne \mathrm{CUT}` 且
:math:`c_2 \ne \mathrm{CUT}`，被测信道由第三波决定）需要单独枚举。

该族的份额随色散减小而增大：官方 MEX 的三信道算例上，β2 = −1 ps²/km 时
占官方总 NLI 的 7.1%（0.30 dB），β2 = −10 ps²/km 时占 0.53%，
β2 = −21.68 ps²/km 时占 0.20%。

**本文件覆盖**

* **官方对照**：三信道带内积分值与官方 cGN MEX 的残差（四个低色散点
  < 0.05 dB，高色散档不被改坏）；
* **独立暴力求积**：把 GN 参考式按岛元做二维稠密求积（不复用本仓的相关域
  变换），验证工程实现没有走样；
* **图元枚举**：槽位表枚举出的三元组与解析预期一致；
* **数值口径**：结果对相关域截断因子不敏感（同一个常数、不是拟合出来的）；
* **确定性**：重复调用逐位相同。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.models import cgn as cgn_module
from network_nlo_eval.models.cgn import CGNSpanKernel, _mci_slot_table
from network_nlo_eval.network.elements import FiberSpanConfig

F_REF = 193.4145e12
SPACING = 75e9
CH_BR = 64e9
L_M = 100e3
P_CH = 1e-3

#: 官方 cGN MEX：三信道（中间信道）总 NLI，带内积分 (W)。
#:
#: 本机 MATLAB R2026a（``matlab -batch``）跑官方 MEX 实测：
#:
#: .. code-block:: bash
#:
#:     matlab -batch "run('<workspace>/evidence/matlab-crosscheck/mci_official_cgn.m')"
#:
#: 脚本用官方包 ``cGN/cGN/functions/NLI_calculator.mexa64``，三信道、
#: 75 GHz 间隔、64 GHz 带宽、100 km SMF、每信道 1 mW、单跨，观测频率在信道
#: 带内取 501 点并对 NLI PSD 积分。β2 = −1/−10/−21.68/−25 的读数与
#: ``tests/test_cgn_mex_crosscheck.py`` 里已有的常量逐位一致；β2 = −2/−3/−5
#: 是本轮新测的。
OFFICIAL_BAND_3CH: dict[float, float] = {
    -1.0: 1.752335e-06,
    -2.0: 1.169646e-06,
    -3.0: 9.047120e-07,
    -5.0: 6.434779e-07,
    -10.0: 3.942328e-07,
    -21.682619: 2.205036e-07,
    -25.0: 1.974589e-07,
}


def _fiber() -> FiberSpanConfig:
    """与 MATLAB 算例同参的 SMF（功率口径衰减 0.162 dB/km，不用衰减多项式）。"""
    return FiberSpanConfig(
        length_km=100.0,
        max_span_length_km=100.0,
        attenuation_db_km_ref=0.162,
        use_attenuation_polynomial=False,
        reference_wavelength_nm=1550.0,
        dispersion_parameter_d=17.0,
        dispersion_slope_s=0.06,
        nonlinear_index_n2=2.6e-20,
        effective_area_um2_ref=80.0,
        aeff_slope_um2_nm=0.05,
    )


def _kernel(n_channels: int, beta2_ps2_km: float, **options) -> CGNSpanKernel:
    """构造指定信道数与 β2 的单跨核（不含 ISRS，与 MATLAB 算例一致）。"""
    centers = F_REF + (np.arange(n_channels) - (n_channels - 1) / 2) * SPACING
    kernel = CGNSpanKernel(
        fiber_config=_fiber(),
        f_abs_hz=centers,
        b_hz=np.full(n_channels, CH_BR),
        include_isrs=False,
        **options,
    )
    kernel.beta2_s2_m = beta2_ps2_km * 1e-27
    return kernel


def _nli(kernel: CGNSpanKernel, channel: int) -> float:
    """返回逐信道的总 NLI 功率 (W)。"""
    out = kernel.evaluate(np.full(kernel.n_nodes, P_CH), L_M)
    return float(out.spm_noise_w[channel] + out.xpm_noise_w[channel])


def _db(value: float, reference: float) -> float:
    return 10.0 * np.log10(value / reference)


# ---------------------------------------------------------------------------
# 独立参考实现：GN 参考式的二维稠密求积
# ---------------------------------------------------------------------------


def _reference_nli(
    n_channels: int,
    cut: int,
    beta2_ps2_km: float,
    *,
    n_band: int = 9,
    n_grid: int = 201,
) -> float:
    r"""GN 参考式的直接二维求积（全部图元），作为独立对照。

    这里**不复用**本仓的相关域变换：直接在 :math:`(x, y) = (f_1 - f, f_2 - f)`
    平面上对每个图元做中点法求积，掩掉第三波落在信道带外的点。

    约定与 ``models/cgn.py`` 一致：

    * 相位失配 :math:`\\Omega = 4\\pi^2\\beta_{2,\\mathrm{eff}} x y`，
      :math:`\\beta_{2,\\mathrm{eff}} = \\beta_2 + \\pi\\beta_3 (x + y)`；
    * NSP 能量谱 :math:`S_p(\\Omega) = \\gamma^2 (1 - 2e^{-\\alpha_x L}
      \\cos(\\Omega L) + e^{-2\\alpha_x L})/(\\alpha_x^2 + \\Omega^2)`，
      :math:`\\alpha_x = (\\alpha_{c_1}+\\alpha_{c_2}+\\alpha_{c_3}-\\alpha_c)/2`；
    * PSD 因子 :math:`G_{c_1}G_{c_2}G_{c_3}`（矩形谱，信道带内为常数）。
    """
    fiber = _fiber()
    alpha = fiber.attenuation_db_km_ref * np.log(10.0) / 10.0 / 1e3
    gamma = 2.0 * np.pi * fiber.nonlinear_index_n2 / (
        fiber.reference_wavelength_nm * 1e-9 * fiber.effective_area_um2_ref * 1e-12
    )
    beta2 = beta2_ps2_km * 1e-27
    beta3 = cgn_module._calc_beta3_from_s_d(
        fiber.dispersion_slope_s,
        fiber.dispersion_parameter_d,
        fiber.reference_wavelength_nm * 1e-9,
    )
    centers = F_REF + (np.arange(n_channels) - (n_channels - 1) / 2) * SPACING
    f_cut = centers[cut]
    half = 0.5 * CH_BR

    nodes, weights = np.polynomial.legendre.leggauss(n_band)
    total = 0.0
    for node, weight in zip(nodes, weights, strict=True):
        delta = 0.5 * CH_BR * node
        obs = f_cut + delta
        grid = np.arange(n_grid) + 0.5
        step = CH_BR / n_grid
        for c1 in range(n_channels):
            x = (centers[c1] - half - obs) + step * grid
            for c2 in range(n_channels):
                y = (centers[c2] - half - obs) + step * grid
                xx = x[:, None]
                yy = y[None, :]
                for c3 in range(n_channels):
                    third = xx + yy + obs
                    mask = (third >= centers[c3] - half) & (third <= centers[c3] + half)
                    if not mask.any():
                        continue
                    alpha_x = 0.5 * (alpha + alpha + alpha - alpha)
                    omega = 4.0 * np.pi**2 * (beta2 + np.pi * beta3 * (xx + yy)) * (xx * yy)
                    decay = np.exp(-alpha_x * L_M)
                    numerator = 1.0 - 2.0 * decay * np.cos(omega * L_M) + decay * decay
                    spectrum = gamma**2 * numerator / (alpha_x**2 + omega * omega)
                    total += (
                        weight * 0.5 * CH_BR * float(spectrum[mask].sum() * step * step)
                    )
    psd = P_CH / CH_BR
    return (16.0 / 27.0) * psd**3 * total


# ---------------------------------------------------------------------------
# 官方对照
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta2", [-1.0, -2.0, -3.0, -5.0])
def test_low_dispersion_residual_is_below_five_hundredths_of_a_db(beta2: float) -> None:
    """低色散点的残差压到 0.05 dB 以内（MCI 图元补齐之后）。

    四个点都用官方 MEX 的实测值作对照（见
    :data:`OFFICIAL_BAND_3CH` 的来源说明）。
    """
    value = _nli(_kernel(3, beta2, band_points=9), 1)
    assert _db(value, OFFICIAL_BAND_3CH[beta2]) == pytest.approx(0.0, abs=0.05)


def test_high_dispersion_residual_is_not_degraded() -> None:
    """高色散档（β2 = −21.68 ps²/km）不得被改坏。"""
    value = _nli(_kernel(3, -21.682619, band_points=9), 1)
    assert _db(value, OFFICIAL_BAND_3CH[-21.682619]) == pytest.approx(0.0, abs=0.02)


def test_two_channel_residual_is_within_a_tenth_of_a_db() -> None:
    """两信道算例同样受益（MCI 图元在两信道下也存在）。"""
    official = 1.067365e-06
    value = _nli(_kernel(2, -1.0, band_points=9), 0)
    assert _db(value, official) == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# 与独立二维求积对照
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta2", [-1.0, -21.682619])
def test_engineered_result_matches_the_dense_quadrature(beta2: float) -> None:
    """工程实现（含 MCI 图元）与 GN 参考式的二维稠密求积一致。

    稠密求积枚举**全部**图元，故它同时校验了带内图元与 MCI 图元：两者
    合计应与官方 MEX 一致，也应与本实现一致。
    """
    reference = _reference_nli(3, 1, beta2)
    value = _nli(_kernel(3, beta2, band_points=9), 1)
    assert _db(value, reference) == pytest.approx(0.0, abs=0.02)


def test_reference_quadrature_reproduces_the_official_value() -> None:
    """独立参考实现本身复现官方 MEX（三信道，β2 = −1 ps²/km）。"""
    reference = _reference_nli(3, 1, -1.0)
    assert _db(reference, OFFICIAL_BAND_3CH[-1.0]) == pytest.approx(0.0, abs=0.02)


# ---------------------------------------------------------------------------
# 图元枚举
# ---------------------------------------------------------------------------


def test_slot_table_enumerates_the_expected_islands() -> None:
    """三信道下的 MCI 槽位表就是解析预期的五个三元组。

    被测信道取中间信道 1，第一波与第二波都在带外：

    * ``(2, 0, k)``（权重 2，覆盖镜像 ``(0, 2, k)``）：k 取第三波可达的
      三个信道 0/1/2；
    * ``(0, 0, 0)`` 与 ``(2, 2, 2)``（权重 1）。
    """
    kernel = _kernel(3, -1.0, band_points=9)
    index, weight = _mci_slot_table(
        kernel.f_abs_hz, kernel.b_hz, kernel.alpha_npm, kernel.beta2_s2_m
    )
    cut = 1
    rows = sorted(
        (int(index[cut, slot, 0]), int(index[cut, slot, 1]), int(index[cut, slot, 2]),
         float(weight[cut, slot]))
        for slot in range(index.shape[1])
    )
    assert rows == [
        (0, 0, 0, 1.0),
        (2, 0, 0, 2.0),
        (2, 0, 1, 2.0),
        (2, 0, 2, 2.0),
        (2, 2, 2, 1.0),
    ]


def test_single_channel_has_no_mci_islands() -> None:
    """单信道没有"两束波都在带外"的图元，故 MCI 不改结果。"""
    kernel = _kernel(1, -1.0, band_points=9)
    index, weight = _mci_slot_table(
        kernel.f_abs_hz, kernel.b_hz, kernel.alpha_npm, kernel.beta2_s2_m
    )
    assert index.shape[1] == 0
    assert weight.size == 0


# ---------------------------------------------------------------------------
# 数值口径与确定性
# ---------------------------------------------------------------------------


def test_result_is_insensitive_to_the_correlation_truncation_factor(monkeypatch) -> None:
    r"""结果对相关域截断因子不敏感（阈值是推导出来的，不是拟合出来的）。

    截断因子只决定"枚举到多远的图元"，而 MCI 图元的贡献按
    :math:`1/(\\Delta f_1 \\Delta f_2)^2` 衰减，故把因子在 10…160 之间
    扫描，低色散点的残差基本不动。
    """
    reference = _nli(_kernel(3, -1.0, band_points=9), 1)
    for factor in (10.0, 160.0):
        monkeypatch.setattr(cgn_module, "TRUNCATION_FACTOR", factor)
        value = _nli(_kernel(3, -1.0, band_points=9), 1)
        assert _db(value, reference) == pytest.approx(0.0, abs=0.01)


def test_mci_result_is_deterministic() -> None:
    """重复调用逐位相同（MCI 核心矩阵是缓存键的纯函数）。"""
    kernel = _kernel(3, -1.0, band_points=9)
    first = _nli(kernel, 1)
    for _ in range(3):
        assert _nli(kernel, 1) == first
