"""集总等效的分布式拉曼放大（DRA）组件。

本模块只实现文献中的**集总等效**子集；论文耦合 ODE 与 Eq. (6)-(9) 的 NLI 见 ``models.hybrid_raman``：

    U. C. de Moura et al., "Ultra-Wideband Modeling of Optical Fibre
    Nonlinearity in Hybrid-Amplified Links" (hybrid Raman + lumped model),
    Section III-B/C.

论文沿光纤积分耦合功率方程 (2) 与 ASE 方程 (4)，并用拟合的归一化功率
剖面 (10) 计算 NLI。本模块只保留其中可独立验证的两件事：

1. **泵浦到信号的净增益**：在泵浦未耗尽的假设下，把泵浦功率沿跨段的
   指数衰减积分成有效长度，得到 on-off 增益
   ``G_R(f) = exp(sum_p sign_p * g_p(f) * P_p * L_eff_p)``。正向（FW）与
   反向（BW）泵浦的 ``L_eff`` 大小相同，差别只体现在增益沿跨段的位置；
   集总等效把这段增益统一挪到跨段末端。
2. **拉曼 ASE**：用局部拉曼增益率沿跨段做一维积分，保留信号衰减与泵浦
   衰减，得到跨段末端的自发拉曼散射功率。它随后由跨段 EDFA 放大，与
   EDFA 自身 ASE 在路由层相加。

明确的近似（详见 ``docs/raman-amplification.md``）：

* 泵浦未耗尽：忽略信号对泵浦的消耗、泵浦之间的相互作用与二阶泵浦；
* 增益集总在跨段末端：跨内信号功率剖面与 NLI 不随拉曼增益改变；
* 拉曼增益谱默认用 Gaussian 或论文 CZ 线性近似，而非实测二氧化硅谱；
  需要标定时可由调用方通过 ``gain_model="table"`` 给出表格谱，表项线性
  插值、表外置零，未给定表格时仍走解析分支；
* 泵浦与信号之间的 XPM 不计入 NLI；
* 只对稳态、单跨静态配置建模，不包含动态增益控制或 droop 效应。

所有频率均为光频 (Hz)，功率为 W，长度为 m。默认参数与
``physics_nlo_eval.channel.network.RamanPump`` 的 Gaussian 近似对齐，便于
做增益形状对照；这不代表对任何具体光纤的标定。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.network.elements import RamanPumpConfig

__all__ = [
    "C_LIGHT",
    "H_PLANCK",
    "K_BOLTZMANN",
    "RamanAmplifier",
    "RamanGainModel",
    "resolve_raman_gain_model",
]

#: 真空光速 (m/s)。
C_LIGHT = 299_792_458.0

#: 普朗克常数 (J·s)。
H_PLANCK = 6.62607015e-34

#: 玻尔兹曼常数 (J/K)。
K_BOLTZMANN = 1.380649e-23

#: 单个泵浦的局部增益率系数 (m/W)，已含偏振平均，除以 A_eff 后得到 1/(W·m)。
#:
#: 该默认值与 ``physics_nlo_eval`` 的波形级 ``RamanPump`` 一致，目的是让
#: 增益形状对照有可比较的绝对尺度；真实光纤的峰值系数随 A_eff 与拉曼谱
#: 标定而变，使用前应替换为实测值。
DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W = 6e-14

#: 二氧化硅拉曼增益峰值的频移 (Hz)。
DEFAULT_DELTA_F_PEAK_HZ = 13.2e12

#: Gaussian 近似的高斯宽度 (Hz)。
DEFAULT_DELTA_F_SIGMA_HZ = 2.12e12

#: CZ 线性近似的截断频移 (Hz)，超过该频移增益置零。
DEFAULT_DELTA_F_CUTOFF_HZ = 15.0e12


class RamanGainModel:
    """泵浦到信号的集总拉曼增益谱与 ASE 模型。

    Parameters
    ----------
        gain_model: ``"gaussian"`` 使用以 13.2 THz 为峰的高斯谱；
            ``"linear"`` 使用 CZ 的线性近似，并在 ``delta_f_cutoff_hz`` 处截断；
            ``"table"`` 使用 ``table`` 给出的归一化表格谱（线性插值，表外
            置零）。默认 ``"gaussian"`` 逐位保持历史数值。
        table: ``(offsets_hz, gains)``，仅 ``gain_model="table"`` 时使用。
            ``offsets_hz`` 是泵浦-信号频移 (Hz)，必须非负且严格递增；
            ``gains`` 是对应的**归一化**增益形（峰值 1 对应
            ``peak_gain_coefficient_m_per_w``），必须有限且非负。两者长度
            相同且至少两个点。表外频移的增益为 0，因此插值不会外推出
            物理上不存在的增益尾巴。
        peak_gain_coefficient_m_per_w: 峰值局部增益率系数 (m/W)，使用前除以
            信号波长的有效模面积 ``A_eff``。
        delta_f_peak_hz: 拉曼增益峰值对应的泵浦-信号频移 (Hz)。
        delta_f_sigma_hz: Gaussian 近似的标准差 (Hz)。
        delta_f_cutoff_hz: 线性近似的截断频移 (Hz)。
        temperature_k: 光纤温度 (K)，只用于自发拉曼散射的声子占据因子。
        include_photon_energy_factor: 是否保留论文式 (2) 的 ``f_p / f_s``
            光子能量因子。关闭后与 ``physics_nlo_eval.RamanPump`` 的
            Gaussian 公式逐项一致，便于对照。
        ase_steps: ASE 沿跨段积分的等距节点数，至少 8。
        reference_a_eff_m2: 未显式给出 ``A_eff`` 时使用的有效模面积 (m²)。

    Raises
    ------
        ValueError: 任一参数越界、``gain_model`` 不受支持、``table`` 与
            ``gain_model`` 不匹配，或表格本身不合法。
    """

    def __init__(
        self,
        *,
        gain_model: Literal["gaussian", "linear", "table"] = "gaussian",
        table: tuple[Sequence[float], Sequence[float]] | None = None,
        peak_gain_coefficient_m_per_w: float = DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
        delta_f_peak_hz: float = DEFAULT_DELTA_F_PEAK_HZ,
        delta_f_sigma_hz: float = DEFAULT_DELTA_F_SIGMA_HZ,
        delta_f_cutoff_hz: float = DEFAULT_DELTA_F_CUTOFF_HZ,
        temperature_k: float = 300.0,
        include_photon_energy_factor: bool = True,
        ase_steps: int = 64,
        reference_a_eff_m2: float = 80e-12,
    ) -> None:
        """校验并保存模型参数。"""
        if gain_model not in ("gaussian", "linear", "table"):
            raise ValueError(f"gain_model must be 'gaussian', 'linear' or 'table', got {gain_model!r}.")
        if gain_model == "table" and table is None:
            raise ValueError("gain_model='table' requires table=(offsets_hz, gains).")
        if gain_model != "table" and table is not None:
            raise ValueError("table is only used when gain_model='table'.")
        if peak_gain_coefficient_m_per_w < 0:
            raise ValueError("peak_gain_coefficient_m_per_w must be non-negative.")
        if delta_f_peak_hz <= 0 or delta_f_sigma_hz <= 0 or delta_f_cutoff_hz <= 0:
            raise ValueError("Raman frequency scales must be positive.")
        if delta_f_cutoff_hz < delta_f_peak_hz:
            raise ValueError("delta_f_cutoff_hz must be >= delta_f_peak_hz.")
        if temperature_k <= 0:
            raise ValueError("temperature_k must be positive.")
        if ase_steps < 8:
            raise ValueError("ase_steps must be >= 8.")
        if reference_a_eff_m2 <= 0:
            raise ValueError("reference_a_eff_m2 must be positive.")

        self.gain_model = gain_model
        self.table: tuple[NDArrayFloat, NDArrayFloat] | None = (
            None if table is None else self._validate_gain_table(table)
        )
        self.peak_gain_coefficient_m_per_w = float(peak_gain_coefficient_m_per_w)
        self.delta_f_peak_hz = float(delta_f_peak_hz)
        self.delta_f_sigma_hz = float(delta_f_sigma_hz)
        self.delta_f_cutoff_hz = float(delta_f_cutoff_hz)
        self.temperature_k = float(temperature_k)
        self.include_photon_energy_factor = bool(include_photon_energy_factor)
        self.ase_steps = int(ase_steps)
        self.reference_a_eff_m2 = float(reference_a_eff_m2)

    @classmethod
    def from_table(
        cls,
        offsets_hz: Sequence[float],
        gains: Sequence[float],
        *,
        peak_gain_coefficient_m_per_w: float = DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
        temperature_k: float = 300.0,
        include_photon_energy_factor: bool = True,
        ase_steps: int = 64,
        reference_a_eff_m2: float = 80e-12,
    ) -> RamanGainModel:
        """用归一化增益表构造模型。

        Args:
            offsets_hz: ``(K,)`` 泵浦-信号频移 (Hz)，非负、严格递增。
            gains: ``(K,)`` 归一化增益形；``1.0`` 对应峰值系数。
            peak_gain_coefficient_m_per_w: 峰值局部增益率系数 (m/W)。
            temperature_k: 光纤温度 (K)。
            include_photon_energy_factor: 是否保留 ``f_p / f_s`` 因子。
            ase_steps: ASE 积分节点数。
            reference_a_eff_m2: 参考有效模面积 (m²)。

        Returns
        -------
            以 ``gain_model="table"`` 构造的模型。

        Raises
        ------
            ValueError: 表格不合法，或任一标量参数越界。
        """
        return cls(
            gain_model="table",
            table=(offsets_hz, gains),
            peak_gain_coefficient_m_per_w=peak_gain_coefficient_m_per_w,
            temperature_k=temperature_k,
            include_photon_energy_factor=include_photon_energy_factor,
            ase_steps=ase_steps,
            reference_a_eff_m2=reference_a_eff_m2,
        )

    @staticmethod
    def _validate_gain_table(
        table: tuple[Sequence[float], Sequence[float]],
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """校验并冻结增益表。

        表是**归一化谱形**，不是绝对增益：绝对值由
        ``peak_gain_coefficient_m_per_w / A_eff`` 给出。因此这里只要求
        频移非负、严格递增、增益有限非负，不要求峰值为 1（允许用户保留
        实测谱的相对起伏）。

        Args:
            table: ``(offsets_hz, gains)``。

        Returns
        -------
            ``(offsets, gains)`` 的 ``float64`` 只读副本。

        Raises
        ------
            ValueError: 形状、单调性、有限性或符号不合法。
        """
        try:
            offsets_raw, gains_raw = table
        except (TypeError, ValueError):
            raise ValueError("table must be a pair (offsets_hz, gains).") from None
        offsets = np.array(offsets_raw, dtype=np.float64, copy=True)
        gains = np.array(gains_raw, dtype=np.float64, copy=True)
        if offsets.ndim != 1 or gains.ndim != 1:
            raise ValueError("gain table offsets and gains must be one-dimensional.")
        if offsets.size != gains.size:
            raise ValueError("gain table offsets and gains must have the same length.")
        if offsets.size < 2:
            raise ValueError("gain table must contain at least two points.")
        if not np.all(np.isfinite(offsets)) or not np.all(np.isfinite(gains)):
            raise ValueError("gain table values must be finite.")
        if np.any(offsets < 0.0):
            raise ValueError("gain table offsets must be non-negative.")
        if np.any(np.diff(offsets) <= 0.0):
            raise ValueError("gain table offsets must be strictly increasing.")
        if np.any(gains < 0.0):
            raise ValueError("gain table gains must be non-negative.")
        offsets.setflags(write=False)
        gains.setflags(write=False)
        return offsets, gains

    def gain_shape(self, shift_hz: NDArrayFloat | float) -> NDArrayFloat:
        """返回归一化拉曼增益谱形状 ``g(|Δf|) / g_peak``。

        Args:
            shift_hz: 泵浦与信号的绝对频移 ``|f_p - f_s|`` (Hz)，可为标量或数组。

        Returns
        -------
            无量纲形状，峰值 1（表格谱由表定义）；负频移与非正频移返回 0，
            表格谱在 ``offsets_hz`` 范围外返回 0。
        """
        shift = np.asarray(shift_hz, dtype=np.float64)
        if self.gain_model == "gaussian":
            gaussian = np.exp(-0.5 * ((shift - self.delta_f_peak_hz) / self.delta_f_sigma_hz) ** 2)
            return np.where(shift > 0, gaussian, 0.0)
        if self.gain_model == "table":
            offsets, gains = self.table
            # 表外置零而不是按端点值外推：实测谱在未覆盖的频移上没有可
            # 信数据，外推会把截断误差放大到增益指数里。
            interpolated = np.interp(shift, offsets, gains, left=0.0, right=0.0)
            return np.where(shift > 0, interpolated, 0.0)
        linear = shift / self.delta_f_peak_hz
        return np.where((shift > 0) & (shift <= self.delta_f_cutoff_hz), linear, 0.0)

    @staticmethod
    def _pump_alpha_npm(pump: RamanPumpConfig) -> float:
        """泵浦波长的线性功率衰减系数 (1/m)。"""
        return float(pump.attenuation_db_per_km) * np.log(10.0) / 10.0 / 1e3

    @classmethod
    def _pump_effective_length_m(cls, pump: RamanPumpConfig, length_m: float) -> float:
        """泵浦的未耗尽有效长度 (m)。

        正向与反向泵浦在跨段末端的 on-off 增益幅度相同：正向为
        ``(1 - exp(-alpha L)) / alpha``，反向积分 ``exp(-alpha (L-z))``
        从 0 到 L 也得到同一表达式。二者沿跨段的分布不同，但集总等效
        只保留末端幅度。
        """
        alpha = cls._pump_alpha_npm(pump)
        if alpha <= 0:
            return float(length_m)
        return float(-np.expm1(-alpha * length_m) / alpha)

    def _pump_rate_per_w(
        self,
        pump: RamanPumpConfig,
        frequencies_hz: NDArrayFloat,
        a_eff_m2: NDArrayFloat,
    ) -> NDArrayFloat:
        """单位泵浦功率、单位长度上的有符号局部增益率 (1/(W·m))。

        正号表示泵浦频率高于信号（信号被放大），负号表示泵浦频率低于信号
        （信号被消耗）。``A_eff`` 逐信道取值，与 PCFM 的波长相关口径一致。
        """
        pump_frequency_hz = C_LIGHT / (float(pump.wavelength_nm) * 1e-9)
        shift = pump_frequency_hz - frequencies_hz
        shape = self.gain_shape(np.abs(shift))
        coefficient = self.peak_gain_coefficient_m_per_w / a_eff_m2
        if self.include_photon_energy_factor:
            coefficient = coefficient * (pump_frequency_hz / frequencies_hz)
        return np.sign(shift) * coefficient * shape

    def _pump_power_profile(self, pump: RamanPumpConfig, z_m: NDArrayFloat, length_m: float) -> NDArrayFloat:
        """泵浦功率沿跨段的分布 ``P_p(z)`` (W)。"""
        alpha = self._pump_alpha_npm(pump)
        if str(pump.direction) == "forward":
            return float(pump.power_w) * np.exp(-alpha * z_m)
        return float(pump.power_w) * np.exp(-alpha * (length_m - z_m))

    def _pump_nsp(self, pump: RamanPumpConfig, frequencies_hz: NDArrayFloat) -> NDArrayFloat:
        """单个泵浦的自发拉曼散射因子 ``n_sp``。

        ``n_sp = 1 / (1 - exp(-h |Δf| / k_B T))``；频移趋于 0 时返回 1，
        因为此时增益形状也为 0，权重自然为零。
        """
        shift = np.abs(C_LIGHT / (float(pump.wavelength_nm) * 1e-9) - frequencies_hz)
        x = H_PLANCK * shift / (K_BOLTZMANN * self.temperature_k)
        safe = x > 1e-12
        nsp = np.ones_like(x)
        nsp[safe] = 1.0 / (-np.expm1(-x[safe]))
        return nsp

    def on_off_gain(
        self,
        frequencies_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        a_eff_m2: NDArrayFloat | float | None = None,
    ) -> NDArrayFloat:
        """跨段末端的拉曼 on-off 增益（线性，非 dB）。

        Args:
            frequencies_hz: ``(N,)`` 信号中心频率 (Hz)。
            pumps: 任意数量的正向/反向泵浦。
            length_m: 跨段长度 (m)。
            a_eff_m2: 逐信道有效模面积 (m²)；``None`` 时用参考值。

        Returns
        -------
            ``(N,)`` 线性 on-off 增益 ``G_R``，泵浦功率为 0 时逐位等于 1。
        """
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        a_eff = self._resolve_a_eff(a_eff_m2, frequencies.shape)
        exponent = np.zeros_like(frequencies)
        for pump in pumps:
            if float(pump.power_w) <= 0:
                continue
            rate = self._pump_rate_per_w(pump, frequencies, a_eff)
            exponent += rate * float(pump.power_w) * self._pump_effective_length_m(pump, length_m)
        return np.exp(exponent)

    def on_off_gain_db(
        self,
        frequencies_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        a_eff_m2: NDArrayFloat | float | None = None,
    ) -> NDArrayFloat:
        """跨段末端的拉曼 on-off 增益 (dB)。"""
        gain = self.on_off_gain(frequencies_hz, pumps, length_m, a_eff_m2=a_eff_m2)
        return 10.0 * np.log10(gain)

    def ase_w(
        self,
        frequencies_hz: NDArrayFloat,
        bandwidths_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        a_eff_m2: NDArrayFloat | float | None = None,
        signal_attenuation_npm: NDArrayFloat | float | None = None,
    ) -> NDArrayFloat:
        """跨段末端的拉曼 ASE 功率 (W)。

        采用沿跨段的一维积分

        .. code-block:: text

            P_ASE(f) = 2 n_sp h f B * int_0^L g_pos(z,f)
                       exp(-alpha_s (L-z)) exp(int_z^L g(z',f) dz') dz

        其中 ``g`` 是局部拉曼增益率 (1/m)，``g_pos`` 只保留放大项（泵浦频率
        高于信号），因此消耗项不会产生负的 ASE。该式保留了信号衰减与泵浦
        衰减，是比 ``2 n_sp h f B (G_R - 1)`` 更保守的分布估计。

        Args:
            frequencies_hz: ``(N,)`` 信号中心频率 (Hz)。
            bandwidths_hz: ``(N,)`` 信号信道带宽 (Hz)。
            pumps: 任意数量的正向/反向泵浦。
            length_m: 跨段长度 (m)。
            a_eff_m2: 逐信道有效模面积 (m²)；``None`` 时用参考值。
            signal_attenuation_npm: 逐信道信号线性衰减系数 (1/m)；
                ``None`` 时用 ``0.2 dB/km``。

        Returns
        -------
            ``(N,)`` 跨段末端、进入 EDFA 之前的拉曼 ASE 功率 (W)。
        """
        frequencies = np.asarray(frequencies_hz, dtype=np.float64)
        bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
        if bandwidths.shape != frequencies.shape:
            raise ValueError("bandwidths_hz must contain one value per signal frequency.")
        if length_m <= 0:
            raise ValueError("length_m must be positive.")
        a_eff = self._resolve_a_eff(a_eff_m2, frequencies.shape)
        alpha_s = self._resolve_signal_attenuation(signal_attenuation_npm, frequencies.shape)
        if not any(float(pump.power_w) > 0 for pump in pumps):
            return np.zeros_like(frequencies)

        z_m = np.linspace(0.0, float(length_m), self.ase_steps)
        rate_total = np.zeros((frequencies.size, z_m.size), dtype=np.float64)
        rate_source = np.zeros_like(rate_total)
        nsp_weighted = np.zeros_like(rate_total)
        for pump in pumps:
            if float(pump.power_w) <= 0:
                continue
            rate = self._pump_rate_per_w(pump, frequencies, a_eff)[:, None] * self._pump_power_profile(
                pump, z_m, length_m
            )[None, :]
            positive = np.maximum(rate, 0.0)
            rate_total += rate
            rate_source += positive
            nsp_weighted += positive * self._pump_nsp(pump, frequencies)[:, None]
        weight = trapezoid(rate_source, z_m, axis=1)
        nsp = np.divide(
            trapezoid(nsp_weighted, z_m, axis=1),
            weight,
            out=np.ones_like(weight),
            where=weight > 0,
        )
        cumulative = cumulative_trapezoid(rate_total, z_m, axis=1, initial=0.0)
        tail = cumulative[:, -1:] - cumulative
        integrand = rate_source * np.exp(-alpha_s[:, None] * (float(length_m) - z_m)[None, :]) * np.exp(tail)
        ase = 2.0 * nsp * H_PLANCK * frequencies * bandwidths * trapezoid(integrand, z_m, axis=1)
        return np.maximum(ase, 0.0)

    def _resolve_a_eff(
        self,
        a_eff_m2: NDArrayFloat | float | None,
        shape: tuple[int, ...],
    ) -> NDArrayFloat:
        """把标量/数组有效模面积广播到信号形状。"""
        if a_eff_m2 is None:
            return np.full(shape, self.reference_a_eff_m2, dtype=np.float64)
        values = np.asarray(a_eff_m2, dtype=np.float64)
        if values.ndim == 0:
            return np.full(shape, float(values), dtype=np.float64)
        if values.shape != shape:
            raise ValueError("a_eff_m2 must be a scalar or contain one value per signal frequency.")
        return values

    @staticmethod
    def _resolve_signal_attenuation(
        signal_attenuation_npm: NDArrayFloat | float | None,
        shape: tuple[int, ...],
    ) -> NDArrayFloat:
        """把标量/数组信号衰减系数广播到信号形状。"""
        if signal_attenuation_npm is None:
            return np.full(shape, 0.2 * np.log(10.0) / 10.0 / 1e3, dtype=np.float64)
        values = np.asarray(signal_attenuation_npm, dtype=np.float64)
        if values.ndim == 0:
            return np.full(shape, float(values), dtype=np.float64)
        if values.shape != shape:
            raise ValueError("signal_attenuation_npm must be a scalar or contain one value per signal.")
        return values


def resolve_raman_gain_model(
    gain_model: str | RamanGainModel,
    *,
    table: tuple[Sequence[float], Sequence[float]] | None = None,
    peak_gain_coefficient_m_per_w: float = DEFAULT_PEAK_GAIN_COEFFICIENT_M_PER_W,
    temperature_k: float = 300.0,
) -> RamanGainModel:
    """把模型层的 ``raman_gain_model`` 选项解析成 :class:`RamanGainModel`。

    模型层历史上只接受 ``"gaussian"``/``"linear"`` 字符串，并在内部把峰值
    系数和温度写进 :class:`RamanGainModel`。表格谱需要一个额外的 ``table``
    参数，因此统一由本函数构造，避免每个模型各写一份分支。已经构造好的
    :class:`RamanGainModel` 原样返回：调用方显式给了模型时，峰值系数与温度
    以该实例为准，不再被这里的默认值覆盖。

    Args:
        gain_model: ``"gaussian"``、``"linear"``、``"table"`` 或现成实例。
        table: ``(offsets_hz, gains)``，仅 ``"table"`` 使用。
        peak_gain_coefficient_m_per_w: 峰值局部增益率系数 (m/W)。
        temperature_k: 光纤温度 (K)。

    Returns
    -------
        可直接交给 ODE 求解器的增益模型。

    Raises
    ------
        ValueError: 字符串不受支持，或 ``table`` 与 ``gain_model`` 不匹配。
    """
    if isinstance(gain_model, RamanGainModel):
        if table is not None:
            raise ValueError("table must not be given when raman_gain_model is already a RamanGainModel.")
        return gain_model
    return RamanGainModel(
        gain_model=gain_model,
        table=table,
        peak_gain_coefficient_m_per_w=peak_gain_coefficient_m_per_w,
        temperature_k=temperature_k,
    )


class RamanAmplifier:
    """一段光纤上的集总等效拉曼放大器。

    该类把 :class:`RamanGainModel` 与一组泵浦、跨长绑定，提供 ``gain``、
    ``gain_db``、``ase_w`` 与 ``apply``。PCFM 在每个跨段构造一次，把跨段
    末端的信号功率乘以 ``gain``，并把 ``ase_w`` 作为进入 EDFA 的 ASE。

    Parameters
    ----------
        pumps: 任意数量的 :class:`~network_nlo_eval.network.elements.RamanPumpConfig`。
        length_m: 跨段长度 (m)。
        model: 复用已有增益模型；``None`` 时使用默认 Gaussian 模型。
        a_eff_m2: 逐信道有效模面积 (m²) 或标量。
        signal_attenuation_npm: 逐信道信号线性衰减 (1/m) 或标量。

    Raises
    ------
        ValueError: 跨长非正。
    """

    def __init__(
        self,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        *,
        model: RamanGainModel | None = None,
        a_eff_m2: NDArrayFloat | float | None = None,
        signal_attenuation_npm: NDArrayFloat | float | None = None,
    ) -> None:
        """保存泵浦与跨段几何。"""
        if length_m <= 0:
            raise ValueError("length_m must be positive.")
        self.pumps = list(pumps)
        self.length_m = float(length_m)
        self.model = model if model is not None else RamanGainModel()
        self.a_eff_m2 = a_eff_m2
        self.signal_attenuation_npm = signal_attenuation_npm

    @property
    def active(self) -> bool:
        """是否至少有一个非零泵浦。"""
        return any(float(pump.power_w) > 0 for pump in self.pumps)

    def gain(self, frequencies_hz: NDArrayFloat) -> NDArrayFloat:
        """跨段末端的线性 on-off 增益。"""
        return self.model.on_off_gain(frequencies_hz, self.pumps, self.length_m, a_eff_m2=self.a_eff_m2)

    def gain_db(self, frequencies_hz: NDArrayFloat) -> NDArrayFloat:
        """跨段末端的 on-off 增益 (dB)。"""
        return self.model.on_off_gain_db(frequencies_hz, self.pumps, self.length_m, a_eff_m2=self.a_eff_m2)

    def ase_w(self, frequencies_hz: NDArrayFloat, bandwidths_hz: NDArrayFloat) -> NDArrayFloat:
        """跨段末端的拉曼 ASE 功率 (W)。"""
        return self.model.ase_w(
            frequencies_hz,
            bandwidths_hz,
            self.pumps,
            self.length_m,
            a_eff_m2=self.a_eff_m2,
            signal_attenuation_npm=self.signal_attenuation_npm,
        )

    def apply(
        self,
        power_w: NDArrayFloat,
        frequencies_hz: NDArrayFloat,
        bandwidths_hz: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """对跨段末端信号施加拉曼增益，并返回拉曼 ASE。

        Args:
            power_w: ``(N,)`` 跨段末端信号功率 (W)。
            frequencies_hz: ``(N,)`` 信号中心频率 (Hz)。
            bandwidths_hz: ``(N,)`` 信号带宽 (Hz)。

        Returns
        -------
            ``(power_out_w, ase_w)``，均为新数组。
        """
        gain = self.gain(frequencies_hz)
        return np.asarray(power_w, dtype=np.float64) * gain, self.ase_w(frequencies_hz, bandwidths_hz)
