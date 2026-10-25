"""MCI 路径的守卫：积分域的符号与带内积分的口径。

这两条各对应一个曾经的实现错误，修复前都能被论文 Fig. 6(a) 的三信道 XMCI 抓出来
（修复前偏差 +0.42…+0.49 dB，修复后 −0.01…+0.05 dB）：

1. 论文 Eq. (24)-(26) 的 ``f2`` 取 CUT **下侧** INT 带，相对 CUT 的频偏为负。原实现
   传正号，下侧带被放到 CUT 正侧，``kappa_m0`` 的积分域恒空（精确为 0）。
2. 论文 Eq. (27) 是带内积分，原实现用带内求和并当作 W 级功率，与 XCI 的比值口径
   直接相加，漏掉 ``Δf``。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.network.elements import FiberSpanConfig

SPACING_HZ = 33.6e9
SYMBOL_RATE_HZ = 32e9
SPAN_LENGTH_M = 100e3


class _Case:
    name = "mci_guard"
    bands = "C"
    num_channels = 3
    channel_spacing_hz = SPACING_HZ
    center_frequency_hz = 193.1e12
    symbol_rate_hz = SYMBOL_RATE_HZ
    span_length_km = 100.0
    span_count = 1
    edfa_noise_figure_db = 5.0
    total_power_dbm = 0.0
    per_channel_power_dbm = 0.0


def _fiber() -> FiberSpanConfig:
    """论文 Sect. 3 的 SMF：D = 16.7, gamma = 1.3 /(W km), alpha = 0.22 dB/km。"""
    return FiberSpanConfig(
        length_km=100.0,
        max_span_length_km=100.0,
        attenuation_db_km_ref=0.22,
        use_attenuation_polynomial=False,
        attenuation_alpha2=0.0,
        attenuation_alpha1=0.0,
        attenuation_alpha0=0.22,
        reference_wavelength_nm=1550.0,
        dispersion_parameter_d=16.7,
        dispersion_slope_s=0.06,
        nonlinear_index_n2=2.6e-20,
        effective_area_um2_ref=80.0,
        aeff_slope_um2_nm=0.0,
    )


@pytest.fixture(scope="module")
def kernel():
    """三信道算例的 span kernel（跨段级，含 MCI kappa 缓存）。"""
    model = make_model("carena2014", _Case())
    return model.kernel_for(_fiber())


def test_lower_interferer_band_sits_below_the_cut(kernel) -> None:
    """下侧 INT 的频偏必须是负的；正号会让 M0 域恒空。"""
    table = kernel.mci_kappas(1, SPAN_LENGTH_M)
    assert table, "三信道中心信道必须有一组 (上侧, 下侧) 组合"
    for upper, lower in table:
        assert upper > 0.0
        assert lower < 0.0, f"下侧频偏为 {lower}，应为负"
        assert abs(lower) == pytest.approx(upper)


def test_kappa_m0_is_not_identically_zero(kernel) -> None:
    """``kappa_m0`` 是 Eq. (24) 的 M0 域；域放错侧时它精确为 0，这是可判定的错。"""
    table = kernel.mci_kappas(10, SPAN_LENGTH_M)
    peak = max(float(np.max(np.abs(item.kappa_m0))) for item in table.values())
    assert peak > 0.0, "kappa_m0 全为 0：M0 积分域是空的"


def test_channel_kappas_are_band_integrals_not_sums(kernel) -> None:
    """逐信道 kappa 必须是带内积分（Eq. (27)），不是带内求和。"""
    grid = kernel._frequency_grid()
    dz = float(grid[1] - grid[0])
    table = kernel.mci_kappas(10, SPAN_LENGTH_M)
    km0, km11, km12 = kernel._mci_channel_kappas(10, SPAN_LENGTH_M)
    index = 1
    pair = (
        round(float(kernel.f_rel_hz[kernel.mci_upper[index]] - kernel.f_rel_hz[index]), 6),
        round(float(kernel.f_rel_hz[kernel.mci_lower[index]] - kernel.f_rel_hz[index]), 6),
    )
    item = table[pair]
    assert km0[index] == pytest.approx(
        float(np.trapz(item.kappa_m0, grid)), rel=1e-12
    )
    assert km11[index] == pytest.approx(
        float(np.trapz(item.kappa_m1_1, grid)), rel=1e-12
    )
    assert km12[index] == pytest.approx(
        float(np.trapz(item.kappa_m1_2, grid)), rel=1e-12
    )
    # 求和与积分的比值是 1/Δf 的量级（kappa 在 41 点栅格上比 Δf 更窄，梯形法不精确
    # 等于 sum*Δf，所以只判量级）；两者相等说明又退回了求和口径。
    raw = float(np.sum(item.kappa_m0))
    assert not np.isclose(raw, km0[index], rtol=1e-6)
    assert raw / km0[index] == pytest.approx(1.0 / dz, rel=0.05)


def test_mci_term_is_positive_and_scales_as_the_cube_of_power(kernel) -> None:
    """MCI 是三光子项：装配后的功率项必须为正，且随 p 三次方缩放（不是恒零）。

    修复前 ``kappa_m0`` 恒为 0、``kappa_m1`` 落镜像域，装配项比 XCI 小 8 个量级；
    这条只判"没有退回那个状态"，绝对定标另由论文 Fig. 6(a) 的对照负责。
    """
    for power in (1e-3, 2e-3):
        gn, corr = kernel._mci_terms(10, SPAN_LENGTH_M, np.full(3, power))
        assert gn[1] > 0.0, f"p={power} 时 MCI 的 GN 项为 {gn[1]}"
        assert np.isfinite(corr[1])
    gn1, _ = kernel._mci_terms(10, SPAN_LENGTH_M, np.full(3, 1e-3))
    gn2, _ = kernel._mci_terms(10, SPAN_LENGTH_M, np.full(3, 2e-3))
    assert gn2[1] / gn1[1] == pytest.approx(8.0, rel=1e-9)
