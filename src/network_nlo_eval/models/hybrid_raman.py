"""论文 hybrid Raman 模型的**可验证子集**（对照模型）。

本模块实现 de Moura 等人的 hybrid Raman + lumped 模型中的核心部分，但**不是**
论文 Eq. (6)-(9) 的完整复现。明确边界如下：

复现（数值可验证）
------------------
* 论文 Eq. (2)：信号、泵浦的耦合功率方程。任意数量 FW/BW 泵，泵浦耗尽、
  泵浦-泵浦、泵浦-信号与信道间 ISRS 都进入同一个耦合矩阵。
* 论文 Eq. (4)：信号 ASE 的耦合方程，包含自发拉曼散射源项
  ``2 h kappa B_i f_i``，并把上一跨的 ASE 作为本跨初值。
* 每跨独立的初值；跨间 ASE 由路由层按信号传输因子搬运，模型只把本跨
  **新产生**的拉曼 ASE 交给跨末 EDFA，避免与路由层重复计数。
* BW 泵用“正向信号 + 反向泵浦”的不动点迭代求解；FW 泵与信号同向积分。
  没有任何 BW 泵时只需一次前向积分。

近似（不要当成论文全式）
------------------------
* NLI **不是** Eq. (6)-(9) 的 SPM/XPM 闭式。这里复用 PCFM 的
  SCI/XCI 闭式核（``PCFMSpanKernel``），只把核里的归一化功率剖面
  ``p_i(z)`` 换成 Eq. (2)/(4) ODE 解出的剖面。因此 Raman 对 NLI 的作用
  通过剖面进入了核，但没有论文的拟合系数 ``alpha_i, C_f,i, ...`` 与
  Eq. (8)/(9) 的解析形式。
* 拉曼增益谱仍用 :class:`~network_nlo_eval.models.raman.RamanGainModel`
  的 Gaussian/CZ 近似，不是论文 Fig. 4 的实测二氧化硅谱。
* 泵浦 ASE 方程没有单独求解（泵浦 ASE 对信号 ASE 的贡献远小于泵浦-信号
  自发散射）；信号 ASE 的耦合是完整的。
* 不包含泵浦-信号 XPM、droop 的显式控制、动态增益控制、非均匀跨长拟合。
* 无泵时直接走 PCFM 的 ISRS-only 积分路径，保证与基线逐位一致；这对应论文
  Eq. (3) 的 LA 情形，而不是本模块的 Raman ODE。

因此 ``ModelInfo.summary`` 写的是 “paper-inspired / reproduced subset”，
不能据此宣称复现了论文的完整 hybrid 模型。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import ModelInfo, SpanContext, register_model
from network_nlo_eval.models.numba_kernels import _calc_aeff_dynamic
from network_nlo_eval.models.pcfm import DEFAULT_SPP_STEPS, PCFMModel, PCFMSpanKernel
from network_nlo_eval.models.raman import C_LIGHT, H_PLANCK, K_BOLTZMANN, RamanGainModel
from network_nlo_eval.models.route import apply_amplifier_and_roadm, resolve_noise_figure
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig

__all__ = [
    "DEFAULT_FIXED_POINT_ITERATIONS",
    "DEFAULT_PAPER_SPP_STEPS",
    "HYBRID_RAMAN_PAPER_MODEL_INFO",
    "HybridRamanPaperModel",
    "HybridRamanPaperSpanKernel",
    "PaperRamanSpanResult",
    "RamanPaperSpanSolver",
]

#: ODE 默认的 RK4 步数。与 PCFM 的 SPP 采样网格一致（每步取中点与终点），
#: 因此无泵时能与 PCFM 基线逐位对齐；有泵时可显式调大以提高 ODE 精度。
DEFAULT_PAPER_SPP_STEPS: int = DEFAULT_SPP_STEPS

#: BW 泵不动点迭代的最大次数。0.2 W 泵、40 km 有效长度的基准在 17 次内收敛。
DEFAULT_FIXED_POINT_ITERATIONS: int = 24

#: 不动点迭代的松弛因子。BW 泵与信号互相耦合，直接替换会振荡。
_FIXED_POINT_RELAXATION: float = 0.5

#: 不动点收敛阈值（相对泵浦峰值功率）。
_FIXED_POINT_TOLERANCE: float = 1.0e-7


@dataclass(frozen=True)
class PaperRamanSpanResult:
    """一跨 ODE 的解。

    Attributes
    ----------
        profile_norm: ``(2*n_steps, N)`` 归一化信号功率剖面
            ``p_i(z) = P_i(z)/P_i(0)``，顺序与输入信号一致。
        power_out_w: ``(N,)`` 跨末信号功率 (W)，已含拉曼增益/泵浦耗尽。
        raman_ase_w: ``(N,)`` 本跨**新产生**的拉曼 ASE (W)，已扣除输入 ASE
            按信号净增益搬运的部分。
        pump_power_out_w: ``(N_p,)`` 泵浦在各自出射端的功率 (W)。FW 泵取
            ``z=L``，BW 泵取 ``z=0``。
        iterations: 实际做的不动点迭代次数。
        converged: 不动点是否在阈值内收敛。
    """

    profile_norm: NDArrayFloat
    power_out_w: NDArrayFloat
    raman_ase_w: NDArrayFloat
    pump_power_out_w: NDArrayFloat
    iterations: int
    converged: bool


def _interp_rows(grid: NDArrayFloat, values: NDArrayFloat, query: float) -> NDArrayFloat:
    """在等距/非等距网格上按行线性插值。

    Args:
        grid: ``(S,)`` 单调递增的坐标。
        values: ``(S, M)`` 每个坐标上的值。
        query: 标量查询坐标。

    Returns
    -------
        ``(M,)`` 插值结果；查询超出网格时取端点值。
    """
    if values.shape[0] == 1:
        return values[0].copy()
    index = int(np.clip(np.searchsorted(grid, query) - 1, 0, values.shape[0] - 2))
    span = float(grid[index + 1] - grid[index])
    weight = 0.0 if span <= 0.0 else float((query - grid[index]) / span)
    return (1.0 - weight) * values[index] + weight * values[index + 1]


class RamanPaperSpanSolver:
    r"""单跨信号/泵浦/ASE coupled ODE 求解器。

    状态向量按“信号在前、泵浦在后”拼接后按频率降序排列，耦合矩阵
    ``C`` 的第 ``(j, k)`` 项遵循论文 Eq. (2)：

    * ``f_k > f_j``（源频率更高）：``C[j,k] = +g(|f_j-f_k|)``，j 获得功率；
    * ``f_k < f_j``（源频率更低）：``C[j,k] = -(f_k/f_j) g(|f_j-f_k|)``，j 失去功率。

    ``g`` 用 :class:`RamanGainModel` 的峰值系数与归一化谱形，并按接收波的
    ``A_eff`` 归一。信道-信道、泵浦-信道、泵浦-泵浦共用这一矩阵。

    Args:
        signal_frequencies_hz: ``(N,)`` 信号频率 (Hz)，建议降序；顺序会原样
            出现在结果里。
        signal_bandwidths_hz: ``(N,)`` 信号带宽 (Hz)。
        pumps: 任意数量的正向/反向泵。
        length_m: 跨长 (m)。
        signal_alpha_npm: ``(N,)`` 信号功率衰减系数 (1/m)。
        signal_a_eff_m2: ``(N,)`` 信号有效模面积 (m²)。
        pump_alpha_npm: ``(N_p,)`` 泵浦功率衰减系数 (1/m)。
        pump_a_eff_m2: ``(N_p,)`` 泵浦有效模面积 (m²)。
        gain_model: 拉曼增益谱模型。
        n_steps: RK4 步数；剖面按每步中点/终点采样，共 ``2*n_steps`` 行。
        n_iterations: BW 泵不动点迭代上限。
        include_isrs: 是否计入信道-信道 ISRS；关闭后只保留泵浦相关耦合。
        include_ase: 是否求解 Eq. (4) 的 ASE 状态。关闭时 ASE 恒为零。

    Raises
    ------
        ValueError: 任一数组形状不匹配或跨长/步数非正。
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
        n_steps: int = DEFAULT_PAPER_SPP_STEPS,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        include_ase: bool = True,
    ) -> None:
        """校验参数并构建与 z 无关的耦合矩阵。"""
        if length_m <= 0:
            raise ValueError("length_m must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")

        self.signal_frequencies_hz = np.asarray(signal_frequencies_hz, dtype=np.float64)
        self.signal_bandwidths_hz = np.asarray(signal_bandwidths_hz, dtype=np.float64)
        self.signal_alpha_npm = np.asarray(signal_alpha_npm, dtype=np.float64)
        self.signal_a_eff_m2 = np.asarray(signal_a_eff_m2, dtype=np.float64)
        self.pumps = list(pumps)
        self.length_m = float(length_m)
        self.gain_model = gain_model
        self.n_steps = int(n_steps)
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
        pump_a_eff = np.asarray(pump_a_eff_m2, dtype=np.float64)
        if pump_alpha.shape != (len(self.pumps),):
            raise ValueError("pump_alpha_npm must contain one value per pump.")
        if pump_a_eff.shape != (len(self.pumps),):
            raise ValueError("pump_a_eff_m2 must contain one value per pump.")

        self._pump_f_hz = pump_f
        self._pump_alpha_npm = pump_alpha
        self._pump_a_eff_m2 = pump_a_eff
        self._pump_power_w = np.asarray([float(pump.power_w) for pump in self.pumps], dtype=np.float64)
        self._pump_forward = np.asarray([str(pump.direction) == "forward" for pump in self.pumps], dtype=bool)

        # 把信号与泵浦拼成一个波集合，再按频率降序排列。
        self._n_signal = n_signal
        self._n_pump = len(self.pumps)
        f_concat = np.concatenate([self.signal_frequencies_hz, pump_f])
        alpha_concat = np.concatenate([self.signal_alpha_npm, pump_alpha])
        a_eff_concat = np.concatenate([self.signal_a_eff_m2, pump_a_eff])
        b_concat = np.concatenate([self.signal_bandwidths_hz, np.zeros(self._n_pump, dtype=np.float64)])
        is_pump_concat = np.concatenate(
            [np.zeros(n_signal, dtype=bool), np.ones(self._n_pump, dtype=bool)]
        )
        direction_concat = np.concatenate(
            [np.ones(n_signal, dtype=np.float64), np.where(self._pump_forward, 1.0, -1.0)]
        )
        p0_concat = np.concatenate([np.zeros(n_signal, dtype=np.float64), self._pump_power_w])

        order = np.argsort(-f_concat, kind="stable")
        self._order = order
        self._f_hz = np.ascontiguousarray(f_concat[order])
        self._alpha_npm = np.ascontiguousarray(alpha_concat[order])
        self._a_eff_m2 = np.ascontiguousarray(a_eff_concat[order])
        self._bandwidth_hz = np.ascontiguousarray(b_concat[order])
        self._is_pump = np.ascontiguousarray(is_pump_concat[order])
        self._direction = np.ascontiguousarray(direction_concat[order])
        self._p0_w = np.ascontiguousarray(p0_concat[order])
        self._n_waves = self._f_hz.size

        self._signal_slots = np.flatnonzero(~self._is_pump)
        self._fwd_slots = np.flatnonzero(self._direction > 0)
        self._bw_slots = np.flatnonzero(self._direction < 0)
        self._fwd_is_signal = ~self._is_pump[self._fwd_slots]
        self._fwd_original = order[self._fwd_slots]
        self._fwd_signal_original = np.where(self._fwd_is_signal, self._fwd_original, -1)
        self._fwd_pump_original = np.where(~self._fwd_is_signal, self._fwd_original - n_signal, -1)
        self._bw_original = order[self._bw_slots] - n_signal

        self._alpha_fwd = np.ascontiguousarray(self._alpha_npm[self._fwd_slots])
        self._alpha_bw = np.ascontiguousarray(self._alpha_npm[self._bw_slots])
        self._b_fwd = np.ascontiguousarray(self._bandwidth_hz[self._fwd_slots])
        self._f_fwd = np.ascontiguousarray(self._f_hz[self._fwd_slots])

        self._coupling = self._build_coupling_matrix()
        self._source_fwd = np.ascontiguousarray(self._build_source_matrix()[self._fwd_slots, :])
        self._z_m = np.linspace(0.0, self.length_m, self.n_steps + 1, dtype=np.float64)
        # 反向泵浦的初值：在 z=L 注入，向 z=0 指数衰减。
        self._bw_initial_total = np.zeros((self.n_steps + 1, self._bw_slots.size), dtype=np.float64)
        for column, pump_index in enumerate(self._bw_original):
            if pump_index < 0:
                continue
            alpha = self._pump_alpha_npm[pump_index]
            self._bw_initial_total[:, column] = self._pump_power_w[pump_index] * np.exp(
                -alpha * (self.length_m - self._z_m)
            )

    def _build_coupling_matrix(self) -> NDArrayFloat:
        """按论文 Eq. (2) 构建 ``(M, M)`` 耦合矩阵 (1/(W·m))。"""
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        shape = self.gain_model.gain_shape(delta)
        coefficient = self.gain_model.peak_gain_coefficient_m_per_w / self._a_eff_m2[:, None]
        gain = coefficient * shape
        # 源频率更高 -> 接收波获得功率；源频率更低 -> 接收波把功率交给源，
        # 并按 Eq. (2) 乘 f_source/f_receiver。
        source_higher = self._f_hz[None, :] > self._f_hz[:, None]
        coupling = np.where(source_higher, gain, -(self._f_hz[None, :] / self._f_hz[:, None]) * gain)
        np.fill_diagonal(coupling, 0.0)
        if not self.include_isrs:
            signal_slots = self._signal_slots
            coupling[np.ix_(signal_slots, signal_slots)] = 0.0
        return coupling

    def _build_source_matrix(self) -> NDArrayFloat:
        r"""自发拉曼散射源矩阵 ``C^+_{jk} kappa_{jk}``。

        ``kappa = 1/(1-exp(-h|Δf|/k_B T))``；只有正耦合（泵浦/信道把功率交给
        接收波）产生自发辐射，因此负耦合被截断为零。
        """
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        safe_delta = np.where(delta > 0.0, delta, 1.0)
        x = H_PLANCK * safe_delta / (K_BOLTZMANN * self.gain_model.temperature_k)
        kappa = np.ones_like(x)
        mask = x > 1.0e-12
        kappa[mask] = 1.0 / (-np.expm1(-x[mask]))
        return np.maximum(self._coupling, 0.0) * kappa

    def _rhs_forward(
        self,
        power: NDArrayFloat,
        ase: NDArrayFloat,
        backward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """正向波（信号与 FW 泵）的右端项。"""
        total = np.empty(self._n_waves, dtype=np.float64)
        total[self._fwd_slots] = power + ase
        total[self._bw_slots] = backward_total
        gain_rate = self._coupling @ total
        gain_fwd = gain_rate[self._fwd_slots]
        d_power = power * gain_fwd - self._alpha_fwd * power
        if not self.include_ase:
            return d_power, np.zeros_like(ase)
        source = 2.0 * H_PLANCK * self._b_fwd * self._f_fwd * (self._source_fwd @ total)
        d_ase = gain_fwd * ase + source - self._alpha_fwd * ase
        return d_power, d_ase

    def _rhs_backward(
        self,
        power: NDArrayFloat,
        ase: NDArrayFloat,
        forward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """反向泵浦的右端项（以 ``s = L - z`` 为积分变量）。"""
        total = forward_total.copy()
        total[self._bw_slots] = power + ase
        gain_rate = self._coupling @ total
        gain_bw = gain_rate[self._bw_slots]
        d_power = power * gain_bw - self._alpha_bw * power
        if not self.include_ase:
            return d_power, np.zeros_like(ase)
        # 反向泵浦的带宽为 0，自发拉曼源项对泵浦 ASE 为零；信号 ASE 的空闲
        # 项由正向积分负责。
        d_ase = gain_bw * ase - self._alpha_bw * ase
        return d_power, d_ase

    def _integrate_forward(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        backward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """固定步长 RK4 前向积分。

        Returns
        -------
            ``(power_end, ase_end, power_grid, ase_grid, power_samples)``。
            ``power_grid`` 为 ``(n_steps+1, n_fwd)``，``power_samples`` 为
            ``(2*n_steps, n_fwd)``（每步中点与终点）。
        """
        n_fwd = self._fwd_slots.size
        power = power_in.copy()
        ase = ase_in.copy()
        power_grid = np.empty((self.n_steps + 1, n_fwd), dtype=np.float64)
        ase_grid = np.empty_like(power_grid)
        samples = np.empty((2 * self.n_steps, n_fwd), dtype=np.float64)
        power_grid[0] = power
        ase_grid[0] = ase
        for step in range(self.n_steps):
            h = self._z_m[step + 1] - self._z_m[step]
            bw0 = backward_total[step]
            bw1 = backward_total[step + 1]
            bw_mid = 0.5 * (bw0 + bw1)

            dp1, da1 = self._rhs_forward(power, ase, bw0)
            dp2, da2 = self._rhs_forward(power + 0.5 * h * dp1, ase + 0.5 * h * da1, bw_mid)
            dp3, da3 = self._rhs_forward(power + 0.5 * h * dp2, ase + 0.5 * h * da2, bw_mid)
            dp4, da4 = self._rhs_forward(power + h * dp3, ase + h * da3, bw1)

            samples[2 * step] = power + 0.5 * h * dp1 + 0.25 * h * (dp2 - dp1)
            power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
            samples[2 * step + 1] = power
            power_grid[step + 1] = power
            ase_grid[step + 1] = ase
        return power, ase, power_grid, ase_grid, samples

    def _integrate_backward(
        self,
        power_in: NDArrayFloat,
        ase_in: NDArrayFloat,
        forward_total: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """固定步长 RK4 反向泵浦积分（积分变量 ``s = L - z``）。

        Returns
        -------
            ``(power_end, ase_end, power_grid_s, ase_grid_s)``。网格按 ``s``
            升序，因此 ``power_grid_s[0]`` 对应 ``z=L``，``[-1]`` 对应 ``z=0``。
        """
        n_bw = self._bw_slots.size
        power = power_in.copy()
        ase = ase_in.copy()
        power_grid = np.empty((self.n_steps + 1, n_bw), dtype=np.float64)
        ase_grid = np.empty_like(power_grid)
        power_grid[0] = power
        ase_grid[0] = ase
        for step in range(self.n_steps):
            h = self._z_m[step + 1] - self._z_m[step]
            z0 = self.length_m - self._z_m[step]
            z1 = self.length_m - self._z_m[step + 1]
            z_mid = 0.5 * (z0 + z1)
            total0 = _interp_rows(self._z_m, forward_total, z0)
            total_mid = _interp_rows(self._z_m, forward_total, z_mid)
            total1 = _interp_rows(self._z_m, forward_total, z1)

            dp1, da1 = self._rhs_backward(power, ase, total0)
            dp2, da2 = self._rhs_backward(power + 0.5 * h * dp1, ase + 0.5 * h * da1, total_mid)
            dp3, da3 = self._rhs_backward(power + 0.5 * h * dp2, ase + 0.5 * h * da2, total_mid)
            dp4, da4 = self._rhs_backward(power + h * dp3, ase + h * da3, total1)

            power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
            power_grid[step + 1] = power
            ase_grid[step + 1] = ase
        return power, ase, power_grid, ase_grid

    def _initial_forward_state(
        self,
        signal_power_in_w: NDArrayFloat,
        signal_ase_in_w: NDArrayFloat | None,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """把信号初值与泵浦初值装进正向状态向量。"""
        power = np.zeros(self._fwd_slots.size, dtype=np.float64)
        ase = np.zeros_like(power)
        signal_mask = self._fwd_is_signal
        signal_original = self._fwd_signal_original[signal_mask]
        power[signal_mask] = signal_power_in_w[signal_original]
        if signal_ase_in_w is not None:
            ase[signal_mask] = signal_ase_in_w[signal_original]
        pump_mask = ~signal_mask
        pump_original = self._fwd_pump_original[pump_mask]
        if pump_original.size:
            power[pump_mask] = self._pump_power_w[pump_original]
        return power, ase

    def solve(
        self,
        signal_power_in_w: NDArrayFloat,
        signal_ase_in_w: NDArrayFloat | None = None,
    ) -> PaperRamanSpanResult:
        """求解一跨的信号/泵浦/ASE ODE。

        Args:
            signal_power_in_w: ``(N,)`` 跨首信号功率 (W)。
            signal_ase_in_w: ``(N,)`` 进入本跨的信号 ASE (W)，``None`` 表示 0。

        Returns
        -------
            一跨的剖面、输出功率、新产生 ASE 与泵浦输出。

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

        power_fw, ase_fw = self._initial_forward_state(power_in, ase_in)
        power_bw = self._p0_w[self._bw_slots].copy()
        ase_bw = np.zeros_like(power_bw)
        backward_total = self._bw_initial_total.copy()
        iterations = 0
        converged = self._bw_slots.size == 0

        if self._bw_slots.size:
            peak_pump = max(float(np.max(self._pump_power_w, initial=0.0)), 1.0e-30)
            for iteration in range(self.n_iterations):
                iterations = iteration + 1
                power_end, ase_end, power_grid, ase_grid, samples = self._integrate_forward(
                    power_fw, ase_fw, backward_total
                )
                forward_total = np.empty((self.n_steps + 1, self._n_waves), dtype=np.float64)
                forward_total[:, self._fwd_slots] = power_grid + ase_grid
                forward_total[:, self._bw_slots] = backward_total

                _, _, power_bw_grid_s, ase_bw_grid_s = self._integrate_backward(
                    power_bw, ase_bw, forward_total
                )
                new_backward = (power_bw_grid_s + ase_bw_grid_s)[::-1]
                change = float(np.max(np.abs(new_backward - backward_total))) / peak_pump
                backward_total = (
                    _FIXED_POINT_RELAXATION * new_backward
                    + (1.0 - _FIXED_POINT_RELAXATION) * backward_total
                )
                if change < _FIXED_POINT_TOLERANCE:
                    converged = True
                    break

        # 用最终的反向泵浦剖面再做一次前向积分，使信号剖面与泵浦剖面自洽。
        power_end, ase_end, power_grid, _ase_grid, samples = self._integrate_forward(
            power_fw, ase_fw, backward_total
        )

        signal_mask = self._fwd_is_signal
        signal_columns = np.flatnonzero(signal_mask)
        signal_original = self._fwd_signal_original[signal_columns]

        power_out = np.zeros(self._n_signal, dtype=np.float64)
        power_out[signal_original] = power_end[signal_columns]
        ase_out = np.zeros(self._n_signal, dtype=np.float64)
        ase_out[signal_original] = ase_end[signal_columns]

        profile = np.zeros((2 * self.n_steps, self._n_signal), dtype=np.float64)
        profile[:, signal_original] = samples[:, signal_columns]
        with np.errstate(divide="ignore", invalid="ignore"):
            profile_norm = np.where(
                power_in[None, :] > 0.0,
                profile / np.where(power_in[None, :] > 0.0, power_in[None, :], 1.0),
                1.0,
            )

        fiber_gain = np.divide(
            power_out,
            power_in,
            out=np.ones_like(power_out),
            where=power_in > 0.0,
        )
        raman_ase = np.maximum(ase_out - ase_in * fiber_gain, 0.0)

        pump_out = np.zeros(self._n_pump, dtype=np.float64)
        pump_columns = np.flatnonzero(~signal_mask)
        forward_pump_original = self._fwd_pump_original[pump_columns]
        if forward_pump_original.size:
            pump_out[forward_pump_original] = power_end[pump_columns]
        if self._bw_slots.size:
            # backward_total 是 P+A；泵浦 ASE 为零，所以反向泵浦功率就是索引 0。
            pump_out[self._bw_original] = backward_total[0]

        return PaperRamanSpanResult(
            profile_norm=profile_norm,
            power_out_w=power_out,
            raman_ase_w=raman_ase,
            pump_power_out_w=pump_out,
            iterations=iterations,
            converged=converged,
        )


class HybridRamanPaperSpanKernel(PCFMSpanKernel):
    """单跨核：用论文 ODE 的剖面替换 PCFM 的 ISRS-only 剖面。

    继承 PCFM 的 SCI/XCI 闭式核与多项式拟合，只覆写
    :meth:`_integrate_spp`。无有效泵浦时回落到父类实现，因此与 PCFM 基线
    逐位一致。

    Args:
        f_abs_hz: ``(N,)`` 信号中心频率 (Hz)。
        b_hz: ``(N,)`` 信号带宽 (Hz)。
        fiber_config: 跨段光纤参数。
        pumps: 本跨使用的泵浦；空列表表示无拉曼。
        gain_model: 拉曼增益谱模型。
        raman_ase: 是否把新产生的拉曼 ASE 暴露给模型层。
        n_iterations: BW 泵不动点迭代上限。
        include_isrs: 是否计入信道-信道 ISRS。
        **kwargs: 透传给 :class:`PCFMSpanKernel`。
    """

    def __init__(
        self,
        f_abs_hz: NDArrayFloat,
        b_hz: NDArrayFloat,
        fiber_config: FiberSpanConfig,
        *,
        pumps: Sequence[RamanPumpConfig],
        gain_model: RamanGainModel,
        raman_ase: bool = True,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        **kwargs: object,
    ) -> None:
        """构造核并缓存泵浦参数。"""
        kwargs.setdefault("cache_spans", False)
        super().__init__(f_abs_hz, b_hz, fiber_config, **kwargs)  # type: ignore[arg-type]
        self._pumps = list(pumps)
        self._gain_model = gain_model
        self._raman_ase_enabled = bool(raman_ase)
        self._n_iterations = int(n_iterations)
        self._include_isrs = bool(include_isrs)
        self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._solver_cache: dict[int, RamanPaperSpanSolver] = {}
        self._pump_alpha_npm = np.asarray(
            [
                float(pump.attenuation_db_per_km) * np.log(10.0) / 10.0 / 1e3
                for pump in self._pumps
            ],
            dtype=np.float64,
        )
        pump_f = np.asarray(
            [C_LIGHT / (float(pump.wavelength_nm) * 1e-9) for pump in self._pumps],
            dtype=np.float64,
        )
        self._pump_a_eff_m2 = _calc_aeff_dynamic(
            C_LIGHT / pump_f,
            fiber_config.effective_area_um2_ref,
            fiber_config.aeff_slope_um2_nm,
            fiber_config.reference_wavelength_nm,
        )
        self._active = any(float(pump.power_w) > 0.0 for pump in self._pumps)

    @property
    def active(self) -> bool:
        """是否至少有一个非零泵浦。"""
        return self._active

    def set_input_ase_w(self, ase_w: NDArrayFloat | None) -> None:
        """设置进入本跨光纤的信号 ASE（模型顺序）。

        Args:
            ase_w: ``(N,)`` ASE 功率 (W)；``None`` 表示零。

        Raises
        ------
            ValueError: 形状不匹配。
        """
        if ase_w is None:
            self._input_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
            return
        values = np.asarray(ase_w, dtype=np.float64)
        if values.shape != (self.n_nodes,):
            raise ValueError("ase_w must contain one value per channel.")
        self._input_ase_internal = np.ascontiguousarray(values[self.order])

    @property
    def last_raman_ase_w(self) -> NDArrayFloat:
        """上一跨新产生的拉曼 ASE (W)，模型顺序。"""
        return np.ascontiguousarray(self._last_raman_ase_internal[self.inverse_order])

    @property
    def last_pump_power_out_w(self) -> NDArrayFloat | None:
        """上一跨的泵浦输出功率 (W)；无泵或尚未求解时返回 ``None``。"""
        return getattr(self, "_last_pump_power_out_w", None)

    def _solver_for(self, span_length_m: float) -> RamanPaperSpanSolver:
        """按跨长缓存 ODE 求解器。"""
        key = int(round(span_length_m * 1e6))
        solver = self._solver_cache.get(key)
        if solver is None:
            solver = RamanPaperSpanSolver(
                self.f_abs_hz,
                self.b_hz,
                self._pumps,
                span_length_m,
                self.alpha_npm,
                self.a_eff_m2,
                self._pump_alpha_npm,
                self._pump_a_eff_m2,
                self._gain_model,
                n_steps=self.spp_steps,
                n_iterations=self._n_iterations,
                include_isrs=self._include_isrs,
                include_ase=self._raman_ase_enabled,
            )
            self._solver_cache[key] = solver
        return solver

    def _integrate_spp(
        self,
        power: NDArrayFloat,
        span_length_m: float,
    ) -> tuple[NDArrayFloat, NDArrayFloat]:
        """用耦合 ODE 的剖面替代 ISRS-only 剖面。

        Args:
            power: ``(N,)`` 跨首信号功率 (W)，内部降序。
            span_length_m: 跨长 (m)。

        Returns
        -------
            ``(profile_norm, power_out)``，含义同父类。
        """
        if not self._active:
            self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
            self._last_pump_power_out_w = None
            return super()._integrate_spp(power, span_length_m)

        solver = self._solver_for(span_length_m)
        result = solver.solve(power, self._input_ase_internal)
        if self._raman_ase_enabled:
            self._last_raman_ase_internal = np.ascontiguousarray(result.raman_ase_w)
        else:
            self._last_raman_ase_internal = np.zeros(self.n_nodes, dtype=np.float64)
        self._last_pump_power_out_w = np.ascontiguousarray(result.pump_power_out_w)
        return result.profile_norm, result.power_out_w


HYBRID_RAMAN_PAPER_MODEL_INFO = ModelInfo(
    name="hybrid_raman_paper",
    summary=(
        "Paper-inspired hybrid Raman subset: coupled signal/pump/ASE ODE (Eq. 2/4) "
        "with PCFM/ISRS-GN closed-form NLI on the ODE profile (NLI is approximate, "
        "not Eq. 6-9)."
    ),
    family="gn+raman",
    reference=(
        "de Moura et al., 'Ultra-Wideband Modeling of Optical Fibre Nonlinearity in "
        "Hybrid-Amplified Links'; Poggiolini et al., JLT 2026, doi:10.1109/jlt.2026.3678322"
    ),
    options=(
        ("raman_pumps", "RamanPumpConfig sequence; omitted means use FiberSpanConfig.raman_pumps."),
        ("raman_gain_model", "'gaussian' (default) or 'linear'."),
        ("raman_ase", "Expose the newly generated Raman ASE before the EDFA. Default True."),
        ("raman_temperature_k", "Fibre temperature used by the spontaneous Raman factor. Default 300 K."),
        ("raman_peak_gain_coefficient_m_per_w", "Peak local Raman gain rate before A_eff division. Default 6e-14."),
        ("include_isrs", "Include inter-channel ISRS in the coupled ODE. Default True."),
        ("fixed_point_iterations", "Maximum BW-pump fixed-point iterations. Default 24."),
        ("poly_degree", "PCFM SPP polynomial degree used by the NLI kernel. Default 5."),
        ("egn", "Apply the Zefreh 2020 EGN correction to the NLI kernel. Default False."),
        ("edge_fix", "Apply the PCFM zero-dispersion edge fix. Default True."),
        ("spp_steps", "RK4 steps of the coupled ODE and of the NLI profile fit. Default 64."),
    ),
)


class HybridRamanPaperModel(PCFMModel):
    """论文 hybrid Raman 的对照模型（可验证子集）。

    Signals and pumps are solved with the coupled ODE of Eq. (2)/(4); the NLI is
    the PCFM closed-form kernel evaluated on that ODE profile.  The model keeps
    its own cross-span ASE state and hands only the newly generated Raman ASE
    to the span EDFA, while the shared route layer transports the accumulated
    ASE with the signal transfer factor.

    Args:
        grid: Spectrum grid.
        ref_fiber_config: Reference fibre.
        symbol_rate_hz: Symbol rate (Hz).
        connections: Optional per-connection grid.
        raman_pumps: Pumps applied to every span.  When omitted, the pumps of
            each :class:`FiberSpanConfig` are used.
        raman_gain_model: ``"gaussian"`` or ``"linear"``.
        raman_ase: Include the Raman ASE before the EDFA.
        raman_temperature_k: Fibre temperature (K).
        raman_peak_gain_coefficient_m_per_w: Peak local Raman gain rate.
        include_isrs: Include inter-channel ISRS in the ODE.
        fixed_point_iterations: Maximum BW-pump fixed-point iterations.
        poly_degree: PCFM polynomial degree of the NLI kernel.
        egn: Apply the EGN correction to the NLI kernel.
        edge_fix: Apply the PCFM edge fix.
        spp_steps: RK4 steps of the ODE and of the NLI profile fit.
        **route_options: Forwarded to :class:`~network_nlo_eval.models.route.RouteModel`.
    """

    info = HYBRID_RAMAN_PAPER_MODEL_INFO

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
        fixed_point_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        poly_degree: int = 5,
        egn: bool = False,
        edge_fix: bool = True,
        spp_steps: int = DEFAULT_PAPER_SPP_STEPS,
        **route_options: object,
    ) -> None:
        """Build the paper comparison model.

        Raises
        ------
            ValueError: Invalid ODE or PCFM options.
        """
        if fixed_point_iterations < 1:
            raise ValueError("fixed_point_iterations must be >= 1.")
        super().__init__(
            grid,
            ref_fiber_config,
            symbol_rate_hz=symbol_rate_hz,
            connections=connections,
            poly_degree=poly_degree,
            egn=egn,
            edge_fix=edge_fix,
            cache_spans=False,
            spp_steps=spp_steps,
            raman_model=None,
            raman_pumps=None,
            raman_gain_model=raman_gain_model,
            raman_ase=raman_ase,
            raman_temperature_k=raman_temperature_k,
            raman_peak_gain_coefficient_m_per_w=raman_peak_gain_coefficient_m_per_w,
            **route_options,
        )
        self._paper_pumps = None if raman_pumps is None else list(raman_pumps)
        self._include_isrs = bool(include_isrs)
        self._fixed_point_iterations = int(fixed_point_iterations)
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)

    def _build_kernel(self, fiber_config: FiberSpanConfig) -> HybridRamanPaperSpanKernel:
        """Build the paper span kernel of this fibre."""
        pumps = self._paper_pumps if self._paper_pumps is not None else fiber_config.raman_pumps
        return HybridRamanPaperSpanKernel(
            self.f_abs_hz,
            self.channel_bandwidth_hz_channels,
            fiber_config,
            pumps=pumps,
            gain_model=self._raman_gain_model,
            raman_ase=self._raman_ase,
            n_iterations=self._fixed_point_iterations,
            include_isrs=self._include_isrs,
            poly_degree=self.poly_degree,
            symbol_rate_hz=self.symbol_rate_hz,
            egn=self.egn,
            edge_fix=self.edge_fix,
            cache_spans=False,
            spp_steps=self._active_spp_steps,
        )

    def evaluate_path(self, optical_path, power_in_w):
        """Evaluate a path, resetting the cross-span ASE state."""
        self._ase_in_w = np.zeros(self.n_nodes, dtype=np.float64)
        return super().evaluate_path(optical_path, power_in_w)

    def span_step(
        self,
        fiber_config,
        edfa_config,
        power_in_w,
        span_length_m,
        span_context: SpanContext | None = None,
    ):
        """Single span: ODE kernel, then the shared amplifier handling.

        Args:
            fiber_config: Fibre parameters of the span.
            edfa_config: Amplifier configuration of the span.
            power_in_w: ``(N,)`` launch power per channel (W).
            span_length_m: Span length (m).
            span_context: Position of the span in the link.

        Returns
        -------
            The :class:`~network_nlo_eval.models.base.ElementOutput` of the span.
        """
        kernel = self.kernel_for(fiber_config)
        kernel.set_input_ase_w(self._ase_in_w)
        fiber_output = kernel.evaluate(
            power_in_w,
            span_length_m,
            1,
            span_index=(self._span_index if span_context is None else int(span_context.span_index)),
            beta2_accum_ps2=self._beta2_accum_ps2,
            span_context=span_context,
        )
        raman_ase_w = kernel.last_raman_ase_w if self._raman_ase else None
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
            input_ase_w=raman_ase_w,
        )
        self._ase_in_w = np.array(element.ase_noise_w, dtype=np.float64, copy=True)
        self._span_index += 1
        self._beta2_accum_ps2 = self._beta2_accum_ps2 + kernel.effective_dispersion_ps2(span_length_m)
        return element


register_model(HYBRID_RAMAN_PAPER_MODEL_INFO, HybridRamanPaperModel)
