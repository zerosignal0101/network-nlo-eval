"""Canonical benchmark cases for cross-model QoT comparison.

The band sizes follow the reference system of Lasagni et al. 2023 (the ECZ baseline
paper): 64 GBaud channels on a 75 GHz grid, no guard bands, covering C+L (11.4 THz),
S+C+L (20.8 THz) or E+S+C+L (35.9 THz) over a 10x100 km SMF link. Keeping the case
definition in one module is what makes "same conditions" checkable: every model is
driven through :func:`build_path` and :func:`launch_profile`, never by its own
hand-rolled path.

Absolute band edges used here (frequency, THz): L 184.4-191.6, C 191.6-195.9,
S 195.9-205.4, E 205.4-220.4. The engine models a contiguous uniform grid, so a
multi-band case is the contiguous span covering the same total width; the per-band
gaps of a real comb are not represented. That limitation is deliberate and is stated
in the comparison report rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.power_conventions import resolve_launch_power_convention
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

__all__ = [
    "BENCH_CASES",
    "POWER_SWEEP_DBM",
    "SMF_100KM",
    "TILT_REFERENCES",
    "BenchCase",
    "band_edges_hz",
    "build_case",
    "build_path",
    "grid_for",
    "launch_profile",
]

#: 倾斜发射谱的功率锚定口径。
#:
#: ``"total"``：保持**总入纤功率**等于同功率平坦谱的总功率，于是不同倾斜度
#: 之间比较的是“同一台放大器输出、只改谱形”。这是倾斜审计的默认口径。
#: ``"center"``：保持倾斜直线在**网格中心频率**处穿过标称每信道功率（奇数信道
#: 时严格等于中间信道功率），总功率随倾斜度增大；15 dB 时总功率比 ``"total"``
#: 高约 2 dB（离散求和 ``10 log10(mean(10**(x/10)))``，``x`` 在 ±7.5 dB 均匀分布）。
TILT_REFERENCES: tuple[str, ...] = ("total", "center")

#: 每信道入纤功率扫描区间 (dBm)。覆盖线性区到明显非线性区。
POWER_SWEEP_DBM: tuple[float, ...] = (-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

#: 参考光纤：SMF，衰减用二次多项式（与 ECZ 基线一致的系数）。
SMF_100KM = FiberSpanConfig(
    length_km=100.0,
    max_span_length_km=100.0,
    attenuation_db_km_ref=0.2,
    use_attenuation_polynomial=True,
    attenuation_alpha2=3.7685e-6,
    attenuation_alpha1=-7.3764e-5,
    attenuation_alpha0=0.162,
    reference_wavelength_nm=1550.0,
    dispersion_parameter_d=17.0,
    dispersion_slope_s=0.06,
    nonlinear_index_n2=2.6e-20,
    effective_area_um2_ref=80.0,
    aeff_slope_um2_nm=0.05,
)

#: 名义波段边界 (Hz)，用于报告与算例构造。
_BANDS_HZ = {
    "L": (184.4e12, 191.6e12),
    "C": (191.6e12, 195.9e12),
    "S": (195.9e12, 205.4e12),
    "E": (205.4e12, 220.4e12),
}


def band_edges_hz(bands: str) -> tuple[float, float]:
    """返回给定波段集合的连续频率范围 ``(low, high)``。

    Args:
        bands: 形如 ``"C+L"`` 或 ``"E+S+C+L"`` 的组合串，字母取自 L/C/S/E。

    Returns
    -------
        最低与最高频率 (Hz)。

    Raises
    ------
        ValueError: 波段字母非法或为空。
    """
    letters = [part for part in bands.replace("+", " ").split() if part]
    if not letters:
        raise ValueError("bands must name at least one of L, C, S, E.")
    unknown = [letter for letter in letters if letter not in _BANDS_HZ]
    if unknown:
        raise ValueError(f"unknown bands {unknown}; expected letters from {sorted(_BANDS_HZ)}.")
    lows = [_BANDS_HZ[letter][0] for letter in letters]
    highs = [_BANDS_HZ[letter][1] for letter in letters]
    return min(lows), max(highs)


@dataclass(frozen=True)
class BenchCase:
    """一个可复现的基准算例。

    Attributes
    ----------
        name: 算例键。
        bands: 波段组合串，仅用于报告。
        num_channels: 信道数。
        channel_spacing_hz: 信道间隔 (Hz)。
        center_frequency_hz: 网格中心频率 (Hz)。
        symbol_rate_hz: 符号率 (Hz)。
        span_length_km: 单跨长度 (km)。
        span_count: 跨段数。
        edfa_noise_figure_db: 每跨 EDFA 噪声系数 (dB)。
        total_power_dbm: 标称总入纤功率 (dBm)，用于把"每信道功率"解释成论文口径。
            它始终是**物理总功率**，与 ``launch_power_convention`` 无关。
        launch_power_convention: 算例驱动的功率口径，``"total"``（默认）或
            ``"per_pol"``。``per_channel_power_dbm`` 与 ``launch_profile`` 的
            ``power_dbm`` 仍表示每信道物理总功率，只有 ``launch_profile`` 的
            输出坐标随本字段变化。
    """

    name: str
    bands: str
    num_channels: int
    channel_spacing_hz: float
    center_frequency_hz: float
    symbol_rate_hz: float
    span_length_km: float
    span_count: int
    edfa_noise_figure_db: float
    total_power_dbm: float
    launch_power_convention: str = "total"

    @property
    def per_channel_power_dbm(self) -> float:
        """标称总功率摊到每信道的 dBm 值（每信道物理总功率）。"""
        return self.total_power_dbm - 10.0 * np.log10(self.num_channels)

    @property
    def per_channel_power_dbm_per_pol(self) -> float:
        """同一每信道功率的每偏振 dBm 值（比总功率低 ``10 log10 2``）。"""
        return self.per_channel_power_dbm - 10.0 * np.log10(2.0)


def _case(
    name: str,
    bands: str,
    num_channels: int,
    spacing_hz: float,
    span_count: int,
    total_power_dbm: float,
    launch_power_convention: str = "total",
) -> BenchCase:
    """由波段范围与信道数反推网格中心，避免手填中心频率写错半个带宽。"""
    low, high = band_edges_hz(bands)
    center = (low + high) / 2.0
    return BenchCase(
        name=name,
        bands=bands,
        num_channels=num_channels,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=center,
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=span_count,
        edfa_noise_figure_db=5.5,
        total_power_dbm=total_power_dbm,
        launch_power_convention=resolve_launch_power_convention(launch_power_convention),
    )


#: 算例表。C+L/S+C+L/E+S+C+L 的宽度与 Lasagni 2023 的 11.4/20.8/35.9 THz 一致。
BENCH_CASES: dict[str, BenchCase] = {
    "cl_152ch": _case("cl_152ch", "C+L", 152, 75e9, 10, 21.0),
    "scl_277ch": _case("scl_277ch", "S+C+L", 277, 75e9, 10, 23.16),
    "escl_479ch": _case("escl_479ch", "E+S+C+L", 479, 75e9, 10, 25.0),
    "escl_479ch_1span": _case("escl_479ch_1span", "E+S+C+L", 479, 75e9, 1, 25.0),
    # 亚奈奎斯特间隔：64 GBaud 信号塞进 50 GHz 栅格，相邻信道谱重叠 22%。
    # 所有闭式模型都假设信道谱不重叠，本算例因此**不在**它们的适用域内，
    # 只作为失效边界用例，不参与精度统计（见 run_all.py 的 ACCURACY_CASES）。
    "c_80ch_subnyquist": _case("c_80ch_subnyquist", "C", 80, 50e9, 10, 20.0),
    "c_57ch": _case("c_57ch", "C", 57, 75e9, 10, 20.0),
    "scaling_40ch": _case("scaling_40ch", "C", 40, 75e9, 1, 16.0),
    "scaling_80ch": _case("scaling_80ch", "C", 80, 75e9, 1, 19.0),
    "scaling_160ch": _case("scaling_160ch", "C+L", 160, 75e9, 1, 22.0),
    "scaling_320ch": _case("scaling_320ch", "S+C+L", 320, 75e9, 1, 25.0),
    "scaling_479ch": _case("scaling_479ch", "E+S+C+L", 479, 75e9, 1, 27.0),
}


def build_case(name: str) -> BenchCase:
    """按名称取算例。

    Raises
    ------
        KeyError: 名称不存在。
    """
    try:
        return BENCH_CASES[name]
    except KeyError:
        raise KeyError(f"unknown benchmark case {name!r}; available: {sorted(BENCH_CASES)}.") from None


def grid_for(case: BenchCase) -> SpectrumGrid:
    """该算例的频谱网格。"""
    return SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz)


def build_path(case: BenchCase) -> MultiSpanOpticalPath:
    """该算例的光路：均匀跨段 + 等 NF 的 EDFA，无 ROADM。

    无 ROADM 是有意的：基准要隔离"模型本身的单跨物理"，不掺入逐跳损耗/
    均衡/滤波惩罚的差异。路径级语义另由 ``tests/test_route_model.py`` 覆盖。
    """
    fiber = SMF_100KM.model_copy(update={"length_km": case.span_length_km, "max_span_length_km": case.span_length_km})
    edfa = EDFAConfig(noise_figure_db=case.edfa_noise_figure_db)
    return MultiSpanOpticalPath([fiber] * case.span_count, [edfa] * case.span_count)


def launch_profile(
    case: BenchCase,
    power_dbm: float | None = None,
    *,
    tilt_db: float = 0.0,
    tilt_reference: str = "total",
    launch_power_convention: str = "auto",
) -> NDArrayFloat:
    r"""入纤功率谱 ``(N,)`` (W)，可选线性（dB 域）功率倾斜。

    倾斜在 dB 域按信道序号线性分布，且**功率随频率升高而增大**：

    .. math::

        P_i = P_0 10^{x_i/10}, \\qquad
        x_i = \\left(\\frac{i}{N-1} - \\frac{1}{2}\\right) \\mathrm{tilt_{dB}},

    因此 ``10 log10(max P / min P)`` 恰为 ``tilt_db``，与
    :attr:`~network_nlo_eval.models.pcfm.SpanValidityReport.power_tilt_db`
    的度量一致。频偏方向固定（红端低、蓝端高），调用方无需再猜符号。

    ``power_dbm`` 始终是每信道**物理总功率**，与输出口径无关。函数先生成
    总功率谱（含 ``tilt_reference="total"`` 的总功率归一化），最后按
    ``launch_power_convention`` 决定是否乘 ``0.5`` 输出每偏振功率。

    Args:
        case: 算例。
        power_dbm: 目标每信道物理总功率 (dBm)；``None`` 时取算例的标称每信道
            功率。它同时是 ``tilt_reference="center"`` 的中心锚定值。
        tilt_db: 最大/最小功率比 (dB)，必须非负有限。``0`` 时走与旧实现
            完全相同的 ``np.full`` 分支（逐位不变）。
        tilt_reference: ``"total"``（默认）把总功率归一到
            ``N * 10**((power_dbm - 10 log10 N)/10)``；``"center"`` 不归一，
            让倾斜直线在网格中心频率处等于标称每信道功率。
        launch_power_convention: 输出坐标，``"auto"``（默认）取
            ``case.launch_power_convention``；``"per_pol"`` 在总功率结果上乘
            ``0.5``（每偏振功率），``"total"`` 原样返回。

    Returns
    -------
        ``(N,)`` 功率 (W)，口径见 ``launch_power_convention``。

    Raises
    ------
        ValueError: ``tilt_db`` 为负/非有限、``tilt_reference`` 非法、口径非法，
            或在单信道算例上要求非零倾斜。
    """
    convention = (
        case.launch_power_convention
        if launch_power_convention == "auto"
        else resolve_launch_power_convention(launch_power_convention)
    )
    convention = resolve_launch_power_convention(convention)
    level = case.per_channel_power_dbm if power_dbm is None else power_dbm
    base_w = 1e-3 * 10 ** (level / 10.0)
    if tilt_reference not in TILT_REFERENCES:
        raise ValueError(f"tilt_reference must be one of {TILT_REFERENCES}, got {tilt_reference!r}.")
    if not np.isfinite(tilt_db) or tilt_db < 0.0:
        raise ValueError(f"tilt_db must be a finite non-negative value, got {tilt_db!r}.")
    if tilt_db == 0.0:
        profile = np.full(case.num_channels, base_w, dtype=np.float64)
        return profile if convention == "total" else profile * 0.5
    if case.num_channels < 2:
        raise ValueError("a non-zero tilt needs at least two channels.")

    # 均匀 dB 步进：端点恰为 ∓tilt/2，中间信道（奇数 N）正好落在 0 dB。
    offsets_db = np.linspace(-0.5, 0.5, case.num_channels, dtype=np.float64) * float(tilt_db)
    power = base_w * 10.0 ** (offsets_db / 10.0)
    if tilt_reference == "total":
        # 归一化到平坦谱的总功率：倾斜只改谱形，不改链路总功率。
        power *= (case.num_channels * base_w) / float(np.sum(power))
    if convention == "per_pol":
        # 总功率归一化之后再折算，per_pol 输出的物理总功率仍是同一链路功率。
        power = power * 0.5
    return np.ascontiguousarray(power)
