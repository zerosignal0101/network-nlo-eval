r"""独立数值 DRA 参考：``scipy.integrate.solve_ivp`` + Eq. (6)-(8) 双重积分.

本模块是 DRA benchmark 的**独立数值参考**，刻意不复用仓库里两台被测机器的核心：

* 信号/泵浦/ASE 的耦合 ODE 用 :func:`scipy.integrate.solve_ivp`（DOP853，紧容差）
  求解，不使用 :class:`~network_nlo_eval.models.raman_ode.RamanPaperSpanSolver`
  的固定步长 RK4 结果；
* NLI 直接用 Buglia 2024 Eq. (6)-(8) 的数值双重积分，剖面取 ODE 的原始
  ``rho(z, f)``，不拟合 Eq. (2)/(1)，也不调用 Eq. (9)/(11)/(12) 的任何闭式。

物理范围与论文一致：任意数量、任意方向（FW/BW）的泵浦，泵浦耗尽、泵浦-泵浦、
信道-信道 ISRS、波长相关的衰减/有效面积/色散都保留。信号 ASE 单独求解，跨段
EDFA 的 ASE 仍由路由层负责，因此“拉曼 ASE”与“EDFA ASE”在结果里是分开的。

数值口径
--------
1. **ODE**：状态按 ``[信号, FW 泵]`` 前向积分，``[BW 泵]`` 以 ``s = L - z`` 反向
   积分；BW 泵用不动点迭代与松弛因子收敛。ASE 是线性方程，在前向功率剖面收敛后
   单独积分，不参与功率 ODE 的迭代。
2. **NLI**：对每个信道对 ``(i, k)`` 计算

   .. math::

       I_{ik} = \int_{-B_i/2}^{B_i/2} df_1 \int_{-B_k/2}^{B_k/2} df_2\,
                \Pi\left(\frac{f_1+f_2}{B_k}\right)
                \left|\int_0^L d\zeta\,
                \sqrt{\frac{\rho(\zeta,f_1)\rho(\zeta,f_2)\rho(\zeta,f_1+f_2-f_i)}
                {\rho(\zeta,f_i)}} e^{j\phi\zeta}\right|^2 ,

   其中 :math:`\phi` 用与仓库闭式模型相同的参考频率约定
   ``-4 pi^2 (f1-fi)(f2-fi)[beta2 + pi beta3(f1+f2-2 f_ref)]``。频率方向用
   ``tan`` 变换分辨 XPM 的窄相干峰（峰宽约 ``alpha / (4 pi^2 |Delta f| |beta2|)``），
   纵向用复合 Simpson 求积。SPM 由 Eq. (7) 取 ``1/2`` 的自干扰项。
3. **剖面插值**：ODE 在信道中心求解，``rho(z, f)`` 在频率方向按相邻信道线性插值。
   这是本参考唯一的频率近似；对 75 GHz 栅格、64 GBd 信道，信道内剖面变化远小于
   信道间变化，实测对 NLI 的影响小于 0.1 dB（见 ``docs/dra-benchmark.md``）。

已知限制
--------
* 跨间 NLI 按功率相加（与 Buglia/hybrid 的 ``epsilon=0`` 口径一致），不建模
  跨间相干；Zhang 2024 的相干累加不在本参考的范围内。
* 拉曼增益谱沿用仓库的 Gaussian/CZ 近似，不是实测二氧化硅谱。
* 泵浦 ASE 未单独求解（泵浦带宽为零，自发源项对泵浦本身为零）；信号 ASE 完整。
* 零色散点附近 ``tan`` 变换的尺度退化，代码把尺度截断到信道半带宽。

References
----------
* H. Buglia et al., "A Closed-Form Expression for the Gaussian Noise Model in the
  Presence of Raman Amplification", JLT 42(2), 2024, Eq. (1)-(8).
* C. Semrau et al., "The Gaussian Noise Model in the Presence of Inter-Channel
  Stimulated Raman Scattering", JLT 2018.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, SpanFiberOutput, register_model
from network_nlo_eval.models.dra_common import (
    DEFAULT_DRA_NLI_QUADRATURE,
    DEFAULT_DRA_STEPS,
    DRARamanModelBase,
    DRASpanKernelBase,
)
from network_nlo_eval.models.raman import C_LIGHT, H_PLANCK, K_BOLTZMANN, RamanGainModel
from network_nlo_eval.models.raman_ode import PaperRamanSpanResult
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "NUMERICAL_DRA_MODEL_INFO",
    "NumericalDRAKernel",
    "NumericalDRAModel",
    "NumericalDRASpanResult",
    "NumericalDRASpanSolver",
    "ODESpanProfile",
    "nli_power_from_ode_profile",
]

#: 默认纵向剖面节点数。必须为奇数，复合 Simpson 才完整。
DEFAULT_NUMERICAL_DRA_PROFILE_POINTS: int = 257

#: DOP853 的相对/绝对容差。紧容差是“参考”一词的最低要求。
DEFAULT_NUMERICAL_DRA_RTOL: float = 1.0e-10
DEFAULT_NUMERICAL_DRA_ATOL: float = 1.0e-15

#: XPM 双重积分的节点数：x 方向分辨相干峰，y 方向只做平滑积分。
DEFAULT_XPM_QUADRATURE_X: int = 16
DEFAULT_XPM_QUADRATURE_Y: int = 6

#: XPM 相干峰的相位截断 ``|phi L| <= limit``。在该范围外 Lorentzian 尾部
#: 的积分占比约 ``1/(pi * limit / (alpha L))``；200 对应约 1%，已远低于
#: 闭式模型本身的误差。截断同时保证纵向 Simpson 网格能分辨相位振荡。
DEFAULT_XPM_PHASE_LIMIT: float = 300.0

#: SPM 双重积分的节点数。SPM 在原点有对数型峰，需要更密的二维网格。
DEFAULT_SPM_QUADRATURE_X: int = 48
DEFAULT_SPM_QUADRATURE_Y: int = 24

#: BW 泵不动点迭代的松弛因子与收敛阈值（相对峰值泵浦功率）。
_BW_RELAXATION: float = 0.5
_BW_TOLERANCE: float = 1.0e-7

#: 避免除零的功率下限。
_POWER_FLOOR: float = 1.0e-30


@dataclass(frozen=True)
class NumericalDRASpanResult:
    """一跨数值 ODE 的解。

    Attributes
    ----------
        profile_norm: ``(n_profile, N)`` 归一化信号剖面 ``P_i(z)/P_i(0)``。
        z_m: ``(n_profile,)`` 剖面节点 (m)。
        power_out_w: ``(N,)`` 跨末信号功率 (W)。
        raman_ase_w: ``(N,)`` 本跨新产生的拉曼 ASE (W)，已扣除输入 ASE 的净增益。
        pump_power_out_w: ``(N_p,)`` 泵浦在各自出射端的功率 (W)。
        iterations: BW 泵不动点迭代次数。
        converged: 不动点是否在阈值内收敛。
    """

    profile_norm: NDArrayFloat
    z_m: NDArrayFloat
    power_out_w: NDArrayFloat
    raman_ase_w: NDArrayFloat
    pump_power_out_w: NDArrayFloat
    iterations: int
    converged: bool


@dataclass(frozen=True)
class ODESpanProfile:
    """给 NLI 双重积分用的原始 ODE 剖面（不做任何拟合）。

    Attributes
    ----------
        profile_norm: ``(n_profile, N)`` 归一化信号剖面，降序频率。
        z_m: ``(n_profile,)`` 纵向节点 (m)。
        length_m: 跨长 (m)。
        frequencies_hz: ``(N,)`` 信道中心频率 (Hz)，与剖面同序。
    """

    profile_norm: NDArrayFloat
    z_m: NDArrayFloat
    length_m: float
    frequencies_hz: NDArrayFloat


class NumericalDRASpanSolver:
    r"""用 ``solve_ivp`` 独立求解一跨的信号/泵浦/ASE 耦合 ODE。

    与 :class:`~network_nlo_eval.models.raman_ode.RamanPaperSpanSolver` 的区别：

    * 积分器是 DOP853 自适应步长，不是固定步长 RK4；
    * 功率 ODE 与 ASE 方程分两步求解，ASE 不参与 BW 泵的不动点迭代；
    * 不提供 ``n_steps`` 的 RK4 采样网格，剖面直接取 ``solve_ivp`` 的 dense output。

    Args:
        signal_frequencies_hz: ``(N,)`` 信号频率 (Hz)，建议降序。
        signal_bandwidths_hz: ``(N,)`` 信号带宽 (Hz)。
        pumps: 任意数量、任意方向的泵浦；功率为 0 的泵浦应事先剔除。
        length_m: 跨长 (m)。
        signal_alpha_npm: ``(N,)`` 信号衰减 (1/m)。
        signal_a_eff_m2: ``(N,)`` 信号有效面积 (m²)。
        pump_alpha_npm: ``(N_p,)`` 泵浦衰减 (1/m)。
        pump_a_eff_m2: ``(N_p,)`` 泵浦有效面积 (m²)。
        gain_model: 拉曼增益谱模型。
        n_profile: 剖面节点数（奇数）。
        rtol: DOP853 相对容差。
        atol: DOP853 绝对容差。
        n_iterations: BW 泵不动点迭代上限。
        include_isrs: 是否计入信道-信道 ISRS。
        include_ase: 是否求解信号 ASE。

    Raises
    ------
        ValueError: 形状不匹配、跨长/节点数非法。
    """

    def __init__(
        self,
        signal_frequencies_hz: NDArrayFloat,
        signal_bandwidths_hz: NDArrayFloat,
        pumps: Sequence[RamanPumpConfig],
        length_m: float,
        signal_alpha_npm: NDArrayFloat,
        signal_a_eff_m2: NDArrayFloat,
        pump_alpha_npm: NDArrayFloat,
        pump_a_eff_m2: NDArrayFloat,
        gain_model: RamanGainModel,
        *,
        n_profile: int = DEFAULT_NUMERICAL_DRA_PROFILE_POINTS,
        rtol: float = DEFAULT_NUMERICAL_DRA_RTOL,
        atol: float = DEFAULT_NUMERICAL_DRA_ATOL,
        n_iterations: int = 24,
        include_isrs: bool = True,
        include_ase: bool = True,
    ) -> None:
        if length_m <= 0:
            raise ValueError("length_m must be positive.")
        if n_profile < 5 or n_profile % 2 == 0:
            raise ValueError("n_profile must be an odd integer >= 5.")
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")
        if rtol <= 0 or atol <= 0:
            raise ValueError("rtol and atol must be positive.")

        self.signal_frequencies_hz = np.asarray(signal_frequencies_hz, dtype=np.float64)
        self.signal_bandwidths_hz = np.asarray(signal_bandwidths_hz, dtype=np.float64)
        self.signal_alpha_npm = np.asarray(signal_alpha_npm, dtype=np.float64)
        self.signal_a_eff_m2 = np.asarray(signal_a_eff_m2, dtype=np.float64)
        self.pumps = list(pumps)
        self.length_m = float(length_m)
        self.gain_model = gain_model
        self.n_profile = int(n_profile)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.n_iterations = int(n_iterations)
        self.include_isrs = bool(include_isrs)
        self.include_ase = bool(include_ase)

        n_signal = self.signal_frequencies_hz.size
        if self.signal_bandwidths_hz.shape != (n_signal,):
            raise ValueError("signal_bandwidths_hz must contain one value per signal.")
        if self.signal_alpha_npm.shape != (n_signal,):
            raise ValueError("signal_alpha_npm must contain one value per signal.")
        if self.signal_a_eff_m2.shape != (n_signal,):
            raise ValueError("signal_a_eff_m2 must contain one value per signal.")

        pump_f = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in self.pumps],
            dtype=np.float64,
        )
        pump_alpha = np.asarray(pump_alpha_npm, dtype=np.float64)
        pump_aeff = np.asarray(pump_a_eff_m2, dtype=np.float64)
        if pump_alpha.shape != (len(self.pumps),):
            raise ValueError("pump_alpha_npm must contain one value per pump.")
        if pump_aeff.shape != (len(self.pumps),):
            raise ValueError("pump_a_eff_m2 must contain one value per pump.")

        self._n_signal = n_signal
        self._n_pump = len(self.pumps)
        self._pump_power_w = np.asarray([float(pump.power_w) for pump in self.pumps], dtype=np.float64)
        self._pump_forward = np.asarray([str(pump.direction) == "forward" for pump in self.pumps], dtype=bool)

        # 所有波按 [信号, 泵浦] 拼接；顺序不必排序，耦合矩阵按同一顺序构造。
        self._f_hz = np.concatenate([self.signal_frequencies_hz, pump_f])
        self._alpha_npm = np.concatenate([self.signal_alpha_npm, pump_alpha])
        self._a_eff_m2 = np.concatenate([self.signal_a_eff_m2, pump_aeff])
        self._bandwidth_hz = np.concatenate(
            [self.signal_bandwidths_hz, np.zeros(self._n_pump, dtype=np.float64)]
        )
        self._is_pump = np.concatenate(
            [np.zeros(n_signal, dtype=bool), np.ones(self._n_pump, dtype=bool)]
        )
        self._direction = np.concatenate(
            [np.ones(n_signal, dtype=np.float64), np.where(self._pump_forward, 1.0, -1.0)]
        )
        self._n_waves = self._f_hz.size

        self._signal_slots = np.arange(n_signal, dtype=np.int64)
        self._fwd_slots = np.flatnonzero(self._direction > 0).astype(np.int64)
        self._bw_slots = np.flatnonzero(self._direction < 0).astype(np.int64)
        self._fwd_is_signal = ~self._is_pump[self._fwd_slots]
        self._fwd_signal_local = np.flatnonzero(self._fwd_is_signal).astype(np.int64)
        self._fwd_signal_original = self._fwd_slots[self._fwd_signal_local]
        self._fwd_pump_original = self._fwd_slots[~self._fwd_is_signal] - n_signal
        self._bw_pump_original = self._bw_slots - n_signal

        self._alpha_fwd = np.ascontiguousarray(self._alpha_npm[self._fwd_slots])
        self._alpha_bw = np.ascontiguousarray(self._alpha_npm[self._bw_slots])
        self._alpha_signal = np.ascontiguousarray(self._alpha_npm[self._signal_slots])
        self._f_signal = np.ascontiguousarray(self._f_hz[self._signal_slots])
        self._b_signal = np.ascontiguousarray(self._bandwidth_hz[self._signal_slots])

        self._coupling = self._build_coupling_matrix()
        self._source = self._build_source_matrix()
        self._coupling_fwd = np.ascontiguousarray(self._coupling[self._fwd_slots, :])
        self._coupling_bw = np.ascontiguousarray(self._coupling[self._bw_slots, :])
        self._coupling_signal = np.ascontiguousarray(self._coupling[self._signal_slots, :])
        self._source_signal = np.ascontiguousarray(self._source[self._signal_slots, :])
        self._z_profile = np.linspace(0.0, self.length_m, self.n_profile, dtype=np.float64)

    @property
    def has_backward_pumps(self) -> bool:
        """是否存在后向泵。"""
        return bool(self._bw_slots.size)

    def _build_coupling_matrix(self) -> NDArrayFloat:
        """按 Eq. (1) 构造 ``(M, M)`` 拉曼耦合矩阵 (1/(W·m))。"""
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        shape = self.gain_model.gain_shape(delta)
        gain = self.gain_model.peak_gain_coefficient_m_per_w / self._a_eff_m2[:, None] * shape
        source_higher = self._f_hz[None, :] > self._f_hz[:, None]
        coupling = np.where(
            source_higher,
            gain,
            -(self._f_hz[None, :] / self._f_hz[:, None]) * gain,
        )
        np.fill_diagonal(coupling, 0.0)
        if not self.include_isrs:
            slots = self._signal_slots
            coupling[np.ix_(slots, slots)] = 0.0
        return np.ascontiguousarray(coupling)

    def _build_source_matrix(self) -> NDArrayFloat:
        """自发拉曼源矩阵 ``max(C, 0) * kappa``，``kappa = 1/(1-exp(-h|df|/kT))``。"""
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        safe_delta = np.where(delta > 0.0, delta, 1.0)
        x = H_PLANCK * safe_delta / (K_BOLTZMANN * self.gain_model.temperature_k)
        kappa = np.ones_like(x)
        mask = x > 1.0e-12
        kappa[mask] = 1.0 / (-np.expm1(-x[mask]))
        return np.ascontiguousarray(np.maximum(self._coupling, 0.0) * kappa)

    def _initial_forward_state(self, signal_power_in_w: NDArrayFloat) -> NDArrayFloat:
        """把信号与 FW 泵的初值装进正向状态向量。"""
        state = np.zeros(self._fwd_slots.size, dtype=np.float64)
        state[self._fwd_signal_local] = signal_power_in_w[self._fwd_signal_original]
        pump_mask = ~self._fwd_is_signal
        if np.any(pump_mask):
            state[pump_mask] = self._pump_power_w[self._fwd_pump_original]
        return state

    def _initial_backward_profile(self) -> NDArrayFloat:
        """BW 泵的初值：在 z=L 注入、向 z=0 指数衰减。"""
        n_bw = self._bw_slots.size
        profile = np.empty((self.n_profile, n_bw), dtype=np.float64)
        for column, pump_index in enumerate(self._bw_pump_original):
            alpha = self._alpha_npm[self._bw_slots[column]]
            profile[:, column] = self._pump_power_w[pump_index] * np.exp(
                -alpha * (self.length_m - self._z_profile)
            )
        return profile

    def _interpolate_backward(self, z_m: float, profile: NDArrayFloat) -> NDArrayFloat:
        """把 BW 泵剖面插值到 ``z_m``。"""
        n_bw = self._bw_slots.size
        if n_bw == 0:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(
            [np.interp(z_m, self._z_profile, profile[:, column]) for column in range(n_bw)],
            dtype=np.float64,
        )

    def _forward_rhs(self, state: NDArrayFloat, backward: NDArrayFloat) -> NDArrayFloat:
        """正向波（信号 + FW 泵）右端项。"""
        total = np.empty(self._n_waves, dtype=np.float64)
        total[self._fwd_slots] = state
        total[self._bw_slots] = backward
        return state * (self._coupling_fwd @ total) - self._alpha_fwd * state

    def _backward_rhs(
        self,
        state: NDArrayFloat,
        forward_state: NDArrayFloat,
    ) -> NDArrayFloat:
        """反向泵浦右端项（积分变量 ``s = L - z``）。"""
        total = np.empty(self._n_waves, dtype=np.float64)
        total[self._fwd_slots] = forward_state
        total[self._bw_slots] = state
        return state * (self._coupling_bw @ total) - self._alpha_bw * state

    def _solve_forward(self, signal_power_in_w: NDArrayFloat, backward_profile: NDArrayFloat):
        """给定 BW 泵剖面，前向积分信号 + FW 泵。"""
        initial = self._initial_forward_state(signal_power_in_w)

        def rhs(z_m: float, state: NDArrayFloat) -> NDArrayFloat:
            return self._forward_rhs(state, self._interpolate_backward(z_m, backward_profile))

        solution = solve_ivp(
            rhs,
            (0.0, self.length_m),
            initial,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(f"solve_ivp forward failed: {solution.message}")
        return solution

    def _solve_backward(self, forward_solution):
        """给定前向功率剖面，反向积分 BW 泵。"""
        initial = self._pump_power_w[self._bw_pump_original].copy()

        def rhs(s_m: float, state: NDArrayFloat) -> NDArrayFloat:
            return self._backward_rhs(state, np.asarray(forward_solution.sol(self.length_m - s_m)))

        solution = solve_ivp(
            rhs,
            (0.0, self.length_m),
            initial,
            method="DOP853",
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
        )
        if not solution.success:
            raise RuntimeError(f"solve_ivp backward failed: {solution.message}")
        return solution

    def _solve_ase(
        self,
        signal_power_in_w: NDArrayFloat,
        ase_in_w: NDArrayFloat,
        forward_solution,
        backward_solution,
    ) -> NDArrayFloat:
        """线性 ASE 方程：给定功率剖面后单独前向积分。"""
        if not self.include_ase:
            return np.zeros(self._n_signal, dtype=np.float64)

        def rhs(z_m: float, ase: NDArrayFloat) -> NDArrayFloat:
            forward_state = np.asarray(forward_solution.sol(z_m))
            backward_state = (
                np.asarray(backward_solution.sol(self.length_m - z_m))
                if backward_solution is not None
                else np.zeros(0, dtype=np.float64)
            )
            total = np.empty(self._n_waves, dtype=np.float64)
            total[self._fwd_slots] = forward_state
            total[self._bw_slots] = backward_state
            # 信号 ASE 进入总功率，参与受激放大与自发源；泵浦 ASE 视为零。
            total[self._signal_slots] += ase
            gain_rate = self._coupling_signal @ total
            source_rate = 2.0 * H_PLANCK * self._f_signal * self._b_signal * (
                self._source_signal @ total
            )
            return gain_rate * ase - self._alpha_signal * ase + source_rate

        solution = solve_ivp(
            rhs,
            (0.0, self.length_m),
            ase_in_w,
            method="DOP853",
            rtol=max(self.rtol, 1.0e-9),
            atol=max(self.atol, 1.0e-18),
        )
        if not solution.success:
            raise RuntimeError(f"solve_ivp ASE failed: {solution.message}")
        return np.asarray(solution.y[:, -1], dtype=np.float64)

    def solve(
        self,
        signal_power_in_w: NDArrayFloat,
        signal_ase_in_w: NDArrayFloat | None = None,
    ) -> NumericalDRASpanResult:
        """求解一跨 ODE。

        Args:
            signal_power_in_w: ``(N,)`` 跨首信号功率 (W)。
            signal_ase_in_w: ``(N,)`` 进入本跨的信号 ASE (W)；``None`` 表示零。

        Returns
        -------
            剖面、输出功率、新产生 ASE 与泵浦输出。

        Raises
        ------
            ValueError: 输入形状不匹配。
        """
        power_in = np.asarray(signal_power_in_w, dtype=np.float64)
        if power_in.shape != (self._n_signal,):
            raise ValueError("signal_power_in_w must contain one value per signal.")
        ase_in = np.zeros(self._n_signal, dtype=np.float64)
        if signal_ase_in_w is not None:
            ase_in = np.asarray(signal_ase_in_w, dtype=np.float64)
            if ase_in.shape != (self._n_signal,):
                raise ValueError("signal_ase_in_w must contain one value per signal.")

        iterations = 0
        converged = not self.has_backward_pumps
        backward_solution = None
        if self.has_backward_pumps:
            backward_profile = self._initial_backward_profile()
            peak_pump = max(float(np.max(self._pump_power_w, initial=0.0)), _POWER_FLOOR)
            for iteration in range(self.n_iterations):
                iterations = iteration + 1
                forward_solution = self._solve_forward(power_in, backward_profile)
                backward_solution = self._solve_backward(forward_solution)
                updated = np.asarray(backward_solution.sol(self.length_m - self._z_profile)).T
                change = float(np.max(np.abs(updated - backward_profile))) / peak_pump
                backward_profile = (
                    _BW_RELAXATION * updated + (1.0 - _BW_RELAXATION) * backward_profile
                )
                if change < _BW_TOLERANCE:
                    converged = True
                    break
            # 用最终 BW 剖面再做一次前向积分，使信号剖面与泵浦自洽。
            forward_solution = self._solve_forward(power_in, backward_profile)
            backward_solution = self._solve_backward(forward_solution)
        else:
            forward_solution = self._solve_forward(power_in, np.zeros((self.n_profile, 0)))

        profile_forward = np.asarray(forward_solution.sol(self._z_profile))
        profile_signal = profile_forward[self._fwd_signal_local, :].T
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_norm = np.where(
                power_in[None, :] > 0.0,
                profile_signal / np.where(power_in[None, :] > 0.0, power_in[None, :], 1.0),
                1.0,
            )

        power_end = np.asarray(forward_solution.sol(self.length_m))
        power_out = power_end[self._fwd_signal_local].copy()
        pump_out = np.zeros(self._n_pump, dtype=np.float64)
        if self._fwd_pump_original.size:
            pump_out[self._fwd_pump_original] = power_end[~self._fwd_is_signal]
        if self._bw_pump_original.size:
            pump_out[self._bw_pump_original] = np.asarray(
                backward_solution.sol(self.length_m)
            )

        ase_out = self._solve_ase(power_in, ase_in, forward_solution, backward_solution)
        fiber_gain = np.divide(
            power_out,
            power_in,
            out=np.ones_like(power_out),
            where=power_in > 0.0,
        )
        raman_ase = np.maximum(ase_out - ase_in * fiber_gain, 0.0)

        return NumericalDRASpanResult(
            profile_norm=profile_norm,
            z_m=self._z_profile.copy(),
            power_out_w=power_out,
            raman_ase_w=raman_ase,
            pump_power_out_w=pump_out,
            iterations=iterations,
            converged=converged,
        )


def _tan_nodes(
    lower: float,
    upper: float,
    scale: float,
    nodes: int,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    """在 ``[lower, upper]`` 上用 ``x = scale * tan(u)`` 生成求积节点与权重。"""
    if upper <= lower or nodes < 1:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    scale = max(float(scale), 1.0)
    u_lower = float(np.arctan(lower / scale))
    u_upper = float(np.arctan(upper / scale))
    base_nodes, base_weights = np.polynomial.legendre.leggauss(int(nodes))
    u = 0.5 * (u_upper - u_lower) * base_nodes + 0.5 * (u_upper + u_lower)
    weights = 0.5 * (u_upper - u_lower) * base_weights
    values = scale * np.tan(u)
    jacobian = scale / np.cos(u) ** 2
    return values, weights * jacobian


def _simpson_weights(z_m: NDArrayFloat) -> NDArrayFloat:
    """等距节点的复合 Simpson 权重。"""
    n = z_m.size
    if n < 3 or n % 2 == 0:
        raise ValueError("the longitudinal grid must have an odd number of nodes >= 3.")
    step = float(z_m[1] - z_m[0])
    weights = np.ones(n, dtype=np.float64)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * step / 3.0


def _interpolate_profile(
    frequencies_hz: NDArrayFloat,
    profile_norm: NDArrayFloat,
    query_hz: NDArrayFloat,
) -> NDArrayFloat:
    """沿频率方向线性插值 ``rho(z, f)``，返回 ``(..., n_z)``。"""
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    order = np.argsort(frequencies)
    f_sorted = frequencies[order]
    rho_sorted = profile_norm[:, order]
    query = np.asarray(query_hz, dtype=np.float64)
    shape = query.shape
    flat = query.ravel()
    index = np.searchsorted(f_sorted, flat, side="right") - 1
    index = np.clip(index, 0, f_sorted.size - 2)
    span = f_sorted[index + 1] - f_sorted[index]
    weight = np.divide(
        flat - f_sorted[index],
        span,
        out=np.zeros_like(flat),
        where=span > 0.0,
    )
    values = (1.0 - weight)[:, None] * rho_sorted[:, index].T + weight[:, None] * rho_sorted[
        :, index + 1
    ].T
    return values.reshape(shape + (profile_norm.shape[0],))


def _xpm_x_nodes(
    alpha_npm: float,
    separation_hz: float,
    dispersion: float,
    half_band_hz: float,
    length_m: float,
    nodes: int,
    phase_limit: float,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    r"""生成 XPM 的 COI 方向节点，使 ``|phi L|`` 有界。

    XPM 的相位近似为 ``phi = 4 pi^2 Delta f D x``，相干峰宽
    ``x0 = alpha / (4 pi^2 |Delta f D|)``。直接在整条信道上做 ``tan`` 变换会把
    绝大多数节点放到 ``|phi L|`` 很大的尾部，而纵向 Simpson 网格无法分辨那里的
    振荡，产生混叠。这里改为在 ``t = phi L`` 上做 ``tan`` 变换：

    .. math::

        t = (\alpha L) \tan(u), \qquad x = \frac{t}{4\pi^2 \Delta f D L},

    于是所有节点都满足 ``|t| <= phase_limit``，纵向网格始终能分辨；代价是丢掉
    Lorentzian 尾部。``phase_limit = 200`` 时尾部约 1%，见模块文档。

    Args:
        alpha_npm: COI 的功率衰减 (1/m)。
        separation_hz: ``f_k - f_i`` (Hz)。
        dispersion: ``beta2 + pi beta3 (f_i+f_k-2 f_ref)`` (s²/m)。
        half_band_hz: COI 半带宽 (Hz)。
        length_m: 跨长 (m)。
        nodes: 节点数。
        phase_limit: ``|phi L|`` 的最大值。

    Returns
    -------
        ``(x_hz, weights_hz)``。若整条信道的最大相位已小于 ``phase_limit``，
        退化为整条信道上的 ``tan`` 变换（无截断）。
    """
    coefficient = 4.0 * np.pi**2 * float(separation_hz) * float(dispersion)
    max_phase = abs(coefficient) * length_m * half_band_hz
    alpha_length = max(float(alpha_npm) * length_m, 1.0e-12)
    if max_phase <= phase_limit or coefficient == 0.0:
        # 整条信道的相位都在网格能分辨的范围内：仍用相干峰宽度做 tan 尺度，
        # 不能直接用半带宽，否则窄峰会被完全跳过。
        scale = half_band_hz if coefficient == 0.0 else half_band_hz * alpha_length / max_phase
        return _tan_nodes(-half_band_hz, half_band_hz, scale, nodes)
    t_values, t_weights = _tan_nodes(-phase_limit, phase_limit, alpha_length, nodes)
    denominator = coefficient * length_m
    return t_values / denominator, t_weights / abs(denominator)


def _phase_xpm(
    beta2: float,
    beta3: float,
    coi_hz: float,
    interferer_hz: float,
    x_hz: float,
    y_hz: float,
    reference_hz: float,
) -> NDArrayFloat:
    """Eq. (8) 的相位失配，按参考频率展开。"""
    dispersion = beta2 + np.pi * beta3 * (coi_hz + interferer_hz + x_hz + y_hz - 2.0 * reference_hz)
    return -4.0 * np.pi**2 * x_hz * (interferer_hz + y_hz - coi_hz) * dispersion


def _phase_spm(
    beta2: float,
    beta3: float,
    center_hz: float,
    x_hz: NDArrayFloat,
    y_hz: NDArrayFloat,
    reference_hz: float,
) -> NDArrayFloat:
    """SPM 的相位失配，按参考频率展开。"""
    dispersion = beta2 + np.pi * beta3 * (2.0 * center_hz + x_hz + y_hz - 2.0 * reference_hz)
    return -4.0 * np.pi**2 * x_hz * y_hz * dispersion


def _link_squared(
    amplitude: NDArrayFloat,
    phase: NDArrayFloat,
    z_m: NDArrayFloat,
    weights: NDArrayFloat,
) -> NDArrayFloat:
    """``|int_0^L amplitude(z) exp(j phase z) dz|**2``，返回 ``(n_freq,)``。"""
    integrand = amplitude * weights[None, :] * np.exp(1j * phase[:, None] * z_m[None, :])
    return np.abs(np.sum(integrand, axis=1)) ** 2


def nli_power_from_ode_profile(
    profile: ODESpanProfile,
    frequencies_hz: NDArrayFloat,
    powers_w: NDArrayFloat,
    bandwidths_hz: NDArrayFloat,
    alpha_npm: NDArrayFloat,
    a_eff_m2: NDArrayFloat,
    beta2: float,
    beta3: float,
    reference_frequency_hz: float,
    n2_m2_w: float,
    *,
    n_xpm_x: int = DEFAULT_XPM_QUADRATURE_X,
    n_xpm_y: int = DEFAULT_XPM_QUADRATURE_Y,
    n_spm_x: int = DEFAULT_SPM_QUADRATURE_X,
    n_spm_y: int = DEFAULT_SPM_QUADRATURE_Y,
    xpm_phase_limit: float = DEFAULT_XPM_PHASE_LIMIT,
) -> tuple[NDArrayFloat, NDArrayFloat]:
    r"""用原始 ODE 剖面数值计算 Eq. (6)-(8) 的 SPM/XPM 噪声功率。

    这是本模块的“数值参考”核心：剖面是 ``P(z,f)/P(0,f)`` 的 ODE 解，不拟合任何
    半解析形式；频率双重积分与纵向积分全部数值求积，不调用 Eq. (9)/(11)/(12)。

    Args:
        profile: ODE 剖面（见 :class:`ODESpanProfile`）。
        frequencies_hz: ``(N,)`` 信道中心频率 (Hz)，降序。
        powers_w: ``(N,)`` 本跨入纤功率 (W)。
        bandwidths_hz: ``(N,)`` 信道带宽 (Hz)。
        alpha_npm: ``(N,)`` 信道衰减 (1/m)。
        a_eff_m2: ``(N,)`` 信道有效面积 (m²)。
        beta2: 参考频率处的 GVD (s²/m)。
        beta3: 色散斜率 (s³/m)。
        reference_frequency_hz: ``beta2``/``beta3`` 的参考频率 (Hz)。
        n2_m2_w: 非线性折射率 (m²/W)。
        n_xpm_x: XPM 频率积分中 COI 方向的节点数。
        n_xpm_y: XPM 频率积分中干扰信道方向的节点数。
        n_spm_x: SPM 频率积分中第一个方向的节点数。
        n_spm_y: SPM 频率积分中第二个方向的节点数。
        xpm_phase_limit: XPM 相干峰的相位截断 ``|phi L|``。

    Returns
    -------
        ``(spm_w, xpm_w)``，均为 ``(N,)``，与输入同序。
    """
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    powers = np.asarray(powers_w, dtype=np.float64)
    bandwidths = np.asarray(bandwidths_hz, dtype=np.float64)
    alpha = np.asarray(alpha_npm, dtype=np.float64)
    areas = np.asarray(a_eff_m2, dtype=np.float64)
    n_channels = frequencies.size
    if powers.shape != (n_channels,) or bandwidths.shape != (n_channels,):
        raise ValueError("powers_w and bandwidths_hz must contain one value per channel.")

    z_m = np.asarray(profile.z_m, dtype=np.float64)
    weights = _simpson_weights(z_m)
    rho = np.asarray(profile.profile_norm, dtype=np.float64)
    gamma = (2.0 * np.pi * frequencies / C_LIGHT) * (n2_m2_w / areas)

    spm = np.zeros(n_channels, dtype=np.float64)
    xpm = np.zeros(n_channels, dtype=np.float64)

    for index in range(n_channels):
        launch = float(powers[index])
        if launch <= 0.0 or bandwidths[index] <= 0.0:
            continue
        half_band = 0.5 * float(bandwidths[index])
        rho_coi = _interpolate_profile(frequencies, rho, np.array([frequencies[index]]))[0]
        dispersion_coi = beta2 + np.pi * beta3 * (2.0 * frequencies[index] - 2.0 * reference_frequency_hz)
        spm_scale = _coherence_scale(alpha[index], dispersion_coi, half_band)
        x_grid, x_weights = _tan_nodes(-half_band, half_band, spm_scale, n_spm_x)
        rho_x = _interpolate_profile(frequencies, rho, frequencies[index] + x_grid)

        spm_integral = 0.0
        for x_index, x_value in enumerate(x_grid):
            y_lower = max(-half_band, -half_band - x_value)
            y_upper = min(half_band, half_band - x_value)
            if y_upper <= y_lower:
                continue
            y_grid, y_weights = _tan_nodes(y_lower, y_upper, spm_scale, n_spm_y)
            rho_y = _interpolate_profile(frequencies, rho, frequencies[index] + y_grid)
            rho_idler = _interpolate_profile(frequencies, rho, frequencies[index] + x_value + y_grid)
            amplitude = np.sqrt(
                np.maximum(rho_x[x_index][None, :] * rho_y * rho_idler / rho_coi[None, :], 0.0)
            )
            phase = _phase_spm(
                beta2,
                beta3,
                float(frequencies[index]),
                np.full_like(y_grid, x_value),
                y_grid,
                reference_frequency_hz,
            )
            spm_integral += x_weights[x_index] * float(
                np.sum(y_weights * _link_squared(amplitude, phase, z_m, weights))
            )
        spm[index] = (
            (16.0 / 27.0)
            * gamma[index] ** 2
            * spm_integral
            / bandwidths[index] ** 2
            * launch**3
        )

        xpm_total = 0.0
        for interferer in range(n_channels):
            if interferer == index:
                continue
            interferer_power = float(powers[interferer])
            if interferer_power <= 0.0 or bandwidths[interferer] <= 0.0:
                continue
            separation = float(frequencies[interferer] - frequencies[index])
            dispersion_interferer = beta2 + np.pi * beta3 * (
                frequencies[index] + frequencies[interferer] - 2.0 * reference_frequency_hz
            )
            x_grid, x_weights = _xpm_x_nodes(
                alpha[index],
                separation,
                dispersion_interferer,
                half_band,
                float(profile.length_m),
                n_xpm_x,
                xpm_phase_limit,
            )
            rho_x = _interpolate_profile(frequencies, rho, frequencies[index] + x_grid)
            half_band_interferer = 0.5 * float(bandwidths[interferer])
            pair_integral = 0.0
            for x_index, x_value in enumerate(x_grid):
                y_lower = max(-half_band_interferer, -half_band_interferer - x_value)
                y_upper = min(half_band_interferer, half_band_interferer - x_value)
                if y_upper <= y_lower:
                    continue
                # XPM 的相干峰由 x 决定，y 方向是平滑积分，尺度取干扰信道半带宽。
                y_grid, y_weights = _tan_nodes(y_lower, y_upper, half_band_interferer, n_xpm_y)
                rho_y = _interpolate_profile(frequencies, rho, frequencies[interferer] + y_grid)
                rho_idler = _interpolate_profile(
                    frequencies,
                    rho,
                    frequencies[interferer] + x_value + y_grid,
                )
                amplitude = np.sqrt(
                    np.maximum(rho_x[x_index][None, :] * rho_y * rho_idler / rho_coi[None, :], 0.0)
                )
                phase = _phase_xpm(
                    beta2,
                    beta3,
                    float(frequencies[index]),
                    float(frequencies[interferer]),
                    x_value,
                    y_grid,
                    reference_frequency_hz,
                )
                pair_integral += x_weights[x_index] * float(
                    np.sum(y_weights * _link_squared(amplitude, phase, z_m, weights))
                )
            xpm_total += (
                (interferer_power / launch) ** 2
                * pair_integral
                / bandwidths[interferer] ** 2
            )
        xpm[index] = (32.0 / 27.0) * gamma[index] ** 2 * xpm_total * launch**3

    return spm, xpm


def _coherence_scale(alpha_npm: float, dispersion: float, width_hz: float) -> float:
    """XPM 相干峰的特征宽度 ``alpha / (4 pi^2 |dispersion| * width)`` (Hz)。"""
    if width_hz <= 0.0 or dispersion == 0.0:
        return max(float(width_hz), 1.0)
    scale = float(alpha_npm) / (4.0 * np.pi**2 * abs(float(dispersion)) * float(width_hz))
    return float(np.clip(scale, 1.0, max(float(width_hz), 1.0)))


class NumericalDRAKernel(DRASpanKernelBase):
    """``numerical_dra`` 的单跨核：``solve_ivp`` ODE + Eq. (6)-(8) 双重积分。

    与 :class:`~network_nlo_eval.models.dra_common.DRASpanKernelBase` 的差别只有
    求解器与 NLI 口径。跨段功率/ASE 的处理、路由层语义完全复用基类。

    Args:
        *args: 透传给 :class:`DRASpanKernelBase`。
        rtol: DOP853 相对容差。
        atol: DOP853 绝对容差。
        n_profile: 纵向剖面节点数（奇数）。
        n_xpm_x / n_xpm_y: XPM 双重积分节点数。
        n_spm_x / n_spm_y: SPM 双重积分节点数。
        **kwargs: 透传给 :class:`DRASpanKernelBase`。
    """

    profile_kind = "numerical_ode"
    emits_nli_amplitudes = False

    def __init__(
        self,
        *args: object,
        rtol: float = DEFAULT_NUMERICAL_DRA_RTOL,
        atol: float = DEFAULT_NUMERICAL_DRA_ATOL,
        n_profile: int = DEFAULT_NUMERICAL_DRA_PROFILE_POINTS,
        n_xpm_x: int = DEFAULT_XPM_QUADRATURE_X,
        n_xpm_y: int = DEFAULT_XPM_QUADRATURE_Y,
        n_spm_x: int = DEFAULT_SPM_QUADRATURE_X,
        n_spm_y: int = DEFAULT_SPM_QUADRATURE_Y,
        xpm_phase_limit: float = DEFAULT_XPM_PHASE_LIMIT,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        if n_profile < 5 or n_profile % 2 == 0:
            raise ValueError("n_profile must be an odd integer >= 5.")
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._n_profile = int(n_profile)
        self._n_xpm_x = int(n_xpm_x)
        self._n_xpm_y = int(n_xpm_y)
        self._n_spm_x = int(n_spm_x)
        self._n_spm_y = int(n_spm_y)
        self._xpm_phase_limit = float(xpm_phase_limit)
        self._result_cache: dict[tuple[int, bytes], tuple[PaperRamanSpanResult, NDArrayFloat, NDArrayFloat]] = {}
        self._nli_cache: dict[tuple[int, bytes], tuple[NDArrayFloat, NDArrayFloat]] = {}

    def _solver_for(self, span_length_m: float) -> NumericalDRASpanSolver:
        """按跨长缓存 ``solve_ivp`` 求解器。"""
        key = int(round(span_length_m * 1e6))
        solver = self._solver_cache.get(key)
        if solver is None:
            solver = NumericalDRASpanSolver(
                self.f_abs_hz,
                self.b_hz,
                self._pumps,
                span_length_m,
                self.alpha_npm,
                self.a_eff_m2,
                self._pump_alpha_npm,
                self._pump_a_eff_m2,
                self._gain_model,
                n_profile=self._n_profile,
                rtol=self._rtol,
                atol=self._atol,
                n_iterations=self._n_iterations,
                include_isrs=self._include_isrs,
                include_ase=self._raman_ase_enabled,
            )
            self._solver_cache[key] = solver
        return solver

    def _fit_profile(self, profile_norm: NDArrayFloat, span_length_m: float) -> ODESpanProfile:
        """把原始 ODE 剖面包装成 NLI 积分需要的对象（不做拟合）。"""
        return ODESpanProfile(
            profile_norm=np.asarray(profile_norm, dtype=np.float64),
            z_m=np.linspace(0.0, span_length_m, profile_norm.shape[0], dtype=np.float64),
            length_m=float(span_length_m),
            frequencies_hz=self.f_abs_hz,
        )

    def _nli_span(
        self,
        power_internal: NDArrayFloat,
        profile: ODESpanProfile,  # type: ignore[override]
        span_context: object | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """本跨 SPM/XPM 噪声功率 (W)，用 Eq. (6)-(8) 数值双重积分。"""
        del span_context
        key = (
            id(profile),
            np.round(
                np.asarray(power_internal)
                / max(float(np.max(power_internal, initial=0.0)), _POWER_FLOOR)
                * 1.0e8
            )
            .astype(np.int64)
            .tobytes(),
        )
        cached = self._nli_cache.get(key)
        if cached is not None:
            return cached
        result = nli_power_from_ode_profile(
            profile,
            self.f_abs_hz,
            power_internal,
            self.b_hz,
            self.alpha_npm,
            self.a_eff_m2,
            self.beta2,
            self.beta3,
            self.reference_frequency_hz,
            self.n2_m2_w,
            n_xpm_x=self._n_xpm_x,
            n_xpm_y=self._n_xpm_y,
            n_spm_x=self._n_spm_x,
            n_spm_y=self._n_spm_y,
            xpm_phase_limit=self._xpm_phase_limit,
        )
        if len(self._nli_cache) >= 8:
            self._nli_cache.clear()
        self._nli_cache[key] = result
        return result

    def evaluate(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
        *,
        span_context: object | None = None,
    ) -> SpanFiberOutput:
        """评估一跨，缓存相同入纤功率的 ODE/NLI 结果。

        拉曼 ASE 是“新产生”的量，与输入 ASE 无关（ASE 方程是线性的）；因此缓存以
        入纤信号功率和跨长为键，输入 ASE 仍由路由层搬运。这一点与
        :class:`~network_nlo_eval.models.dra_common.DRASpanKernelBase` 的逐次求解
        结果一致，只是省掉了相同跨的重复 ODE 积分。

        Args:
            power_in_w: ``(N,)`` 入纤功率 (W)，模型顺序。
            span_length_m: 跨长 (m)。
            n_effective_spans: 协议兼容参数，忽略。
            span_context: 协议兼容参数，本参考按功率累加，忽略。

        Returns
        -------
            出纤功率与 SPM/XPM 噪声，模型顺序。
        """
        del n_effective_spans, span_context
        power = np.asarray(power_in_w, dtype=np.float64)
        if power.shape != (self.n_nodes,):
            raise ValueError("power_in_w must contain one value per channel.")
        power_internal = np.ascontiguousarray(power[self.order])
        scale = max(float(np.max(power_internal, initial=0.0)), _POWER_FLOOR)
        key = (
            int(round(span_length_m * 1e6)),
            np.round(power_internal / scale * 1.0e8).astype(np.int64).tobytes(),
        )
        cached = self._result_cache.get(key)
        if cached is None:
            solver = self._solver_for(span_length_m)
            result = solver.solve(power_internal)
            profile = self._profile_for(power_internal, span_length_m, result.profile_norm)
            spm, xpm = self._nli_span(power_internal, profile, None)
            if len(self._result_cache) >= 8:
                self._result_cache.clear()
            cached = (result, spm, xpm)
            self._result_cache[key] = cached
        result, spm, xpm = cached
        self._last_raman_ase_internal = (
            np.ascontiguousarray(result.raman_ase_w)
            if self._raman_ase_enabled
            else np.zeros(self.n_nodes, dtype=np.float64)
        )
        self._last_pump_power_out_w = np.ascontiguousarray(result.pump_power_out_w)
        return SpanFiberOutput(
            power_out_w=np.ascontiguousarray(result.power_out_w[self.inverse_order]),
            spm_noise_w=np.ascontiguousarray(spm[self.inverse_order]),
            xpm_noise_w=np.ascontiguousarray(xpm[self.inverse_order]),
        )

    def reset_state(self) -> None:
        """清空逐路径缓存。"""
        super().reset_state()
        self._result_cache.clear()
        self._nli_cache.clear()


NUMERICAL_DRA_MODEL_INFO = ModelInfo(
    name="numerical_dra",
    summary=(
        "Independent numerical DRA reference: scipy solve_ivp coupled signal/pump/ASE ODE "
        "(DOP853, tight tolerances) plus the Eq. (6)-(8) GN double integral on the raw ODE "
        "profile. Arbitrary forward/backward pumps; Raman ASE is separated from the EDFA ASE."
    ),
    family="numerical+raman",
    reference=(
        "Buglia et al., JLT 42(2):636-648, 2024, Eq. (1)-(8) for the Raman ODE and GN kernel; "
        "Semrau et al., JLT 2018 for the ISRS-GN numerical baseline"
    ),
    options=(
        ("raman_pumps", "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear' (CZ)."),
        ("raman_ase", "Expose the newly generated Raman ASE before the EDFA. Default True."),
        ("raman_temperature_k", "Fibre temperature of the spontaneous Raman factor. Default 300 K."),
        ("raman_peak_gain_coefficient_m_per_w", "Peak local Raman gain rate. Default 6e-14."),
        ("include_isrs", "Include channel-channel ISRS in the coupled ODE. Default True."),
        ("fixed_point_iterations", "Maximum backward-pump fixed-point iterations. Default 24."),
        ("rtol", "DOP853 relative tolerance. Default 1e-10."),
        ("atol", "DOP853 absolute tolerance. Default 1e-15."),
        ("n_profile", "Longitudinal ODE profile nodes (odd). Default 257."),
        ("n_xpm_x", "XPM COI-direction quadrature nodes. Default 16."),
        ("n_xpm_y", "XPM interferer-direction quadrature nodes. Default 6."),
        ("n_spm_x", "SPM first-direction quadrature nodes. Default 48."),
        ("n_spm_y", "SPM second-direction quadrature nodes. Default 24."),
        ("xpm_phase_limit", "XPM coherence-peak phase truncation |phi L|. Default 300."),
    ),
)


class NumericalDRAModel(DRARamanModelBase):
    """独立数值 DRA 参考模型。

    信号/泵浦/ASE 用 :class:`NumericalDRASpanSolver` 的 ``solve_ivp`` 解；NLI 用
    Eq. (6)-(8) 的数值双重积分。跨间按功率相加，不建模相干相位，因此与
    Buglia/hybrid 的 ``epsilon=0`` 口径同口径，而与 Zhang 的相干累加不同。

    Args:
        grid: 频谱栅格。
        ref_fiber_config: 参考光纤。
        symbol_rate_hz: 符号率 (Hz)。
        connections: 可选逐连接栅格。
        raman_pumps: 每跨的泵浦；``None`` 用光纤配置里的泵浦。
        raman_gain_model: ``"gaussian"`` 或 ``"linear"``。
        raman_ase: 是否在 EDFA 前计入拉曼 ASE。
        raman_temperature_k: 光纤温度 (K)。
        raman_peak_gain_coefficient_m_per_w: 峰值拉曼增益率。
        include_isrs: 是否计入信道-信道 ISRS。
        fixed_point_iterations: 后向泵不定点迭代上限。
        rtol: DOP853 相对容差。
        atol: DOP853 绝对容差。
        n_profile: 纵向剖面节点数（奇数）。
        n_xpm_x / n_xpm_y: XPM 双重积分节点数。
        n_spm_x / n_spm_y: SPM 双重积分节点数。
        xpm_phase_limit: XPM 相干峰的相位截断 ``|phi L|``。
        **route_options: 透传给 :class:`~network_nlo_eval.models.route.RouteModel`。
    """

    info = NUMERICAL_DRA_MODEL_INFO
    _kernel_class = NumericalDRAKernel
    supports_span_context = False

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections=None,
        raman_pumps: Sequence[RamanPumpConfig] | None = None,
        raman_gain_model: str = "gaussian",
        raman_ase: bool = True,
        raman_temperature_k: float = 300.0,
        raman_peak_gain_coefficient_m_per_w: float = 6e-14,
        include_isrs: bool = True,
        fixed_point_iterations: int = 24,
        rtol: float = DEFAULT_NUMERICAL_DRA_RTOL,
        atol: float = DEFAULT_NUMERICAL_DRA_ATOL,
        n_profile: int = DEFAULT_NUMERICAL_DRA_PROFILE_POINTS,
        n_xpm_x: int = DEFAULT_XPM_QUADRATURE_X,
        n_xpm_y: int = DEFAULT_XPM_QUADRATURE_Y,
        n_spm_x: int = DEFAULT_SPM_QUADRATURE_X,
        n_spm_y: int = DEFAULT_SPM_QUADRATURE_Y,
        xpm_phase_limit: float = DEFAULT_XPM_PHASE_LIMIT,
        **route_options: object,
    ) -> None:
        if n_profile < 5 or n_profile % 2 == 0:
            raise ValueError("n_profile must be an odd integer >= 5.")
        self._rtol = float(rtol)
        self._atol = float(atol)
        self._n_profile = int(n_profile)
        self._n_xpm_x = int(n_xpm_x)
        self._n_xpm_y = int(n_xpm_y)
        self._n_spm_x = int(n_spm_x)
        self._n_spm_y = int(n_spm_y)
        self._xpm_phase_limit = float(xpm_phase_limit)
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            raman_pumps=raman_pumps,
            raman_gain_model=raman_gain_model,
            raman_ase=raman_ase,
            raman_temperature_k=raman_temperature_k,
            raman_peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            include_isrs=include_isrs,
            fixed_point_iterations=fixed_point_iterations,
            n_steps=DEFAULT_DRA_STEPS,
            n_quadrature=DEFAULT_DRA_NLI_QUADRATURE,
            **route_options,
        )

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> NumericalDRAKernel:
        """构造单跨数值核。"""
        return self._kernel_class(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            pumps=self._resolved_pumps(fiber_config),
            gain_model=self._gain_model,
            raman_ase=self._raman_ase,
            n_steps=DEFAULT_DRA_STEPS,
            n_iterations=self._fixed_point_iterations,
            include_isrs=self._include_isrs,
            n_quadrature=DEFAULT_DRA_NLI_QUADRATURE,
            rtol=self._rtol,
            atol=self._atol,
            n_profile=self._n_profile,
            n_xpm_x=self._n_xpm_x,
            n_xpm_y=self._n_xpm_y,
            n_spm_x=self._n_spm_x,
            n_spm_y=self._n_spm_y,
            xpm_phase_limit=self._xpm_phase_limit,
        )


register_model(NUMERICAL_DRA_MODEL_INFO, NumericalDRAModel)
