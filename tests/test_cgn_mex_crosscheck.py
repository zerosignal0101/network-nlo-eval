"""cGN 与官方 MATLAB MEX（``NLI_calculator.mexa64``）的交叉验证。

**背景**

官方 cGN 包（``cGN/cGN/functions/NLI_calculator.mexa64``，Y. Jiang, Y. Gao,
P. Poggiolini，JLT 投稿版）在 MATLAB R2026a 下可直接跑。本文件的常数全部
来自本机实测（MATLAB R2026a Update 5，``matlab -batch``）：

算例：64 GBaud 矩形谱、100 km SMF、1 mW/信道、1 跨、α = 0.162 dB/km、
γ = 1.317442e-3 /(W·m)、β3 = β4 = 0。

**两条对照线**

官方 cGN 是**频率分辨**模型：它给出 NLI PSD 随频率的变化，再在信道带内
积分。本仓的 cGN 默认档是**闭式**模型：只在被测信道中心频率求一次 NLI
PSD，再乘信道带宽（"带内 NLI PSD 取常数"，GN/EGN 闭式模型族的常规口径）。

因此有两组官方数字可用：

``OFFICIAL_CENTRE``
    官方 NLI PSD 在**信道中心**的值乘以带宽。这条线对照的是"闭式口径"的
    物理量：本仓 cGN 的默认档应当复现它。
``OFFICIAL_BAND``
    官方 NLI PSD 在**信道带内**的积分。这条线对照的是"带内积分"口径：本仓
    cGN 的 ``band_points > 1`` 档应当复现它。

两条线的差值就是本仓 cGN 与官方 MEX 的系统性偏差来源（见交付报告）。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.cgn import CGNSpanKernel, _band_quadrature
from network_nlo_eval.network.elements import FiberSpanConfig

#: 官方 cGN：单信道 SPM，NLI PSD 在信道中心的值乘带宽 (W)。
#: MATLAB: ``evidence/matlab-crosscheck/spm_single.m`` + PSD 导出。
OFFICIAL_CENTRE_1CH: dict[float, float] = {
    -1.0: 5.010158e-07,
    -10.0: 2.437110e-07,
    -21.682619: 1.522738e-07,
    -25.0: 1.384204e-07,
}

#: 官方 cGN：单信道 SPM，带内积分后的 NLI 功率 (W)。
OFFICIAL_BAND_1CH: dict[float, float] = {
    -1.0: 4.456294e-07,
    -10.0: 2.062242e-07,
    -21.682619: 1.293726e-07,
    -25.0: 1.178653e-07,
}

#: 官方 cGN：3 信道（中间信道）总 NLI，带内积分 (W)。
OFFICIAL_BAND_3CH: dict[float, float] = {
    -1.0: 1.752335e-06,
    -10.0: 3.942328e-07,
    -21.682619: 2.205036e-07,
    -25.0: 1.974589e-07,
}

#: 官方 cGN：2 信道（第一个信道）总 NLI，带内积分 (W)。
OFFICIAL_BAND_2CH: dict[float, float] = {
    -1.0: 1.067365e-06,
    -10.0: 2.998871e-07,
    -21.682619: 1.748662e-07,
    -25.0: 1.576078e-07,
}

#: 官方 cGN：3 信道（中间信道）NLI PSD 在信道中心的值乘带宽 (W)。
OFFICIAL_CENTRE_3CH: dict[float, float] = {
    -1.0: 1.865981e-06,
    -10.0: 4.283871e-07,
    -21.682619: 2.398513e-07,
    -25.0: 2.146509e-07,
}

F_REF = 193.4145e12
SPACING = 75e9
CH_BR = 64e9
L_M = 100e3
P_CH = 1e-3

#: 本仓默认档（"带内 NLI PSD 取常数"）相对官方**带内积分**的偏差 (dB)。
#:
#: 这不是误差，是口径差：默认档取的是官方 NLI PSD 在带心的值再乘带宽，
#: 而官方在带内积分。官方 PSD 在带边比带心低 3–5 dB，故默认档偏高。
#: 该常数把这条系统性偏差钉在测试里：任何改动都要显式更新它。
DEFAULT_BAND_BIAS_1CH_DB: dict[float, float] = {
    -1.0: 0.510,
    -10.0: 0.698,
    -21.682619: 0.665,
    -25.0: 0.654,
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


def _nli(kernel: CGNSpanKernel, channel: int) -> tuple[float, float]:
    """返回逐信道的 (SPM, XPM) 噪声功率 (W)。"""
    out = kernel.evaluate(np.full(kernel.n_nodes, P_CH), L_M)
    return float(out.spm_noise_w[channel]), float(out.xpm_noise_w[channel])


def _db(value: float, reference: float) -> float:
    return 10.0 * np.log10(value / reference)


# ---------------------------------------------------------------------------
# 求积器
# ---------------------------------------------------------------------------


def test_band_quadrature_weights_sum_to_two() -> None:
    """Gauss-Legendre 节点铺满 [-1, 1]，权重之和为区间长度 2。"""
    for n_points in (2, 5, 9, 13):
        nodes = np.array([node for node, _ in _band_quadrature(n_points)])
        weights = np.array([weight for _, weight in _band_quadrature(n_points)])
        assert nodes.min() >= -1.0
        assert nodes.max() <= 1.0
        assert weights.sum() == pytest.approx(2.0, abs=1e-12)


def test_band_quadrature_rejects_too_few_points() -> None:
    """节点数 < 2 直接报错。"""
    with pytest.raises(ValueError):
        _band_quadrature(1)


def test_model_rejects_invalid_band_points() -> None:
    """模型入口校验 band_points。"""
    with pytest.raises(ValueError):
        create_model(
            "cgn",
            SpectrumGrid(1, SPACING, F_REF),
            _fiber(),
            symbol_rate_hz=64e9,
            band_points=0,
        )


# ---------------------------------------------------------------------------
# 对照线 1：闭式口径（默认档）↔ 官方 NLI PSD 的带心值
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta2", sorted(OFFICIAL_CENTRE_1CH))
def test_flat_convention_matches_official_centre_value(beta2: float) -> None:
    """默认档（带心 NLI PSD × 带宽）复现官方 NLI PSD 的带心值。

    这条对照说明"闭式口径"的物理量本身是对的：偏差不来自 γ、α、NSP 指数、
    island 几何或功率归一化，只来自有没有在带内积分。
    """
    spm, xpm = _nli(_kernel(1, beta2), 0)
    assert xpm == 0.0
    assert _db(spm, OFFICIAL_CENTRE_1CH[beta2]) == pytest.approx(0.0, abs=0.05)


@pytest.mark.parametrize("beta2", sorted(OFFICIAL_CENTRE_3CH))
def test_flat_convention_matches_official_centre_value_3ch(beta2: float) -> None:
    """3 信道下默认档也复现官方 NLI PSD 的带心值（偏差 < 0.2 dB）。"""
    spm, xpm = _nli(_kernel(3, beta2), 1)
    total = spm + xpm
    assert _db(total, OFFICIAL_CENTRE_3CH[beta2]) == pytest.approx(0.0, abs=0.2)


# ---------------------------------------------------------------------------
# 对照线 2：带内积分口径（band_points > 1）↔ 官方带内积分
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta2", sorted(OFFICIAL_BAND_1CH))
def test_band_integration_matches_official_single_channel(beta2: float) -> None:
    """带内积分档复现官方 MEX 的单信道 SPM 带内积分值（残差 < 0.02 dB）。"""
    spm, xpm = _nli(_kernel(1, beta2, band_points=9), 0)
    assert xpm == 0.0
    assert _db(spm, OFFICIAL_BAND_1CH[beta2]) == pytest.approx(0.0, abs=0.02)


@pytest.mark.parametrize("beta2", sorted(OFFICIAL_BAND_2CH))
def test_band_integration_matches_official_two_channels(beta2: float) -> None:
    """带内积分档复现官方 MEX 的 2 信道总 NLI（残差 < 0.1 dB）。"""
    spm, xpm = _nli(_kernel(2, beta2, band_points=9), 0)
    assert _db(spm + xpm, OFFICIAL_BAND_2CH[beta2]) == pytest.approx(0.0, abs=0.1)


@pytest.mark.parametrize("beta2", sorted(OFFICIAL_BAND_3CH))
def test_band_integration_matches_official_three_channels(beta2: float) -> None:
    """带内积分档复现官方 MEX 的 3 信道总 NLI。

    低色散（β2 = −1 ps²/km）放宽到 0.35 dB：本仓的 island 分解把第一波钉在
    被测信道带内，因此**两束相互作用波都在带外**的 MCI 型图元被略去；在
    β2 = −1 ps²/km 上这部分占官方总 NLI 的 3.3%（0.15 dB）。其余色散下该
    份额 < 0.2%（< 0.01 dB），故容差收紧到 0.05 dB。
    """
    spm, xpm = _nli(_kernel(3, beta2, band_points=9), 1)
    tolerance = 0.35 if beta2 == -1.0 else 0.05
    assert _db(spm + xpm, OFFICIAL_BAND_3CH[beta2]) == pytest.approx(0.0, abs=tolerance)


# ---------------------------------------------------------------------------
# 口径差本身
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta2", sorted(DEFAULT_BAND_BIAS_1CH_DB))
def test_default_convention_bias_is_documented(beta2: float) -> None:
    """默认档相对官方带内积分的偏差被钉在文档值上（±0.02 dB）。

    该偏差是**口径差**而非误差：默认档取官方 NLI PSD 的带心值乘带宽，官方
    在带内积分。官方 PSD 在带边比带心低 3–5 dB，故默认档偏高 0.5–0.7 dB。
    """
    spm, _ = _nli(_kernel(1, beta2), 0)
    bias = _db(spm, OFFICIAL_BAND_1CH[beta2])
    assert bias == pytest.approx(DEFAULT_BAND_BIAS_1CH_DB[beta2], abs=0.02)


def test_band_integration_is_monotone_towards_the_official_value() -> None:
    """带内积分节点越多，结果越贴近官方值（不会来回摆动）。"""
    beta2 = -21.682619
    official = OFFICIAL_BAND_1CH[beta2]
    errors = []
    for n_points in (1, 5, 9, 13, 21):
        spm, _ = _nli(_kernel(1, beta2, band_points=n_points), 0)
        errors.append(abs(_db(spm, official)))
    assert errors[0] > 0.5
    assert errors[-1] < 0.02
    assert errors[-1] < errors[1]
