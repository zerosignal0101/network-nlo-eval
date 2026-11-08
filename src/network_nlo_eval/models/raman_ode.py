"""论文拉曼耦合 ODE：信号、泵浦与 ASE 的沿跨演化（Eq. (2)/(4)）。

本模块只放一跨的物理方程，不含任何 NLI 核，也不依赖
:mod:`~network_nlo_eval.models.dra_common` 或具体的论文模型。这样
``hybrid_raman``（hybrid 论文模型）与 ``dra_common``（Buglia/Zhang DRA 对照）
都可以引用同一份 ODE，而不会形成循环导入。

被 ``hybrid_raman`` 重新导出，旧的
``from network_nlo_eval.models.hybrid_raman import RamanPaperSpanSolver``
仍可用。

References
----------
H. Buglia, R. I. Killey, P. Bayvel, "Ultra-Wideband Modeling of Optical Fibre
Nonlinearity in Hybrid-Amplified Links", J. Lightwave Technol., 2024.
Eq. (2) 信号/泵浦耦合功率方程，Eq. (4) 信号 ASE 耦合方程。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.pcfm import DEFAULT_SPP_STEPS
from network_nlo_eval.models.raman import (
    C_LIGHT,
    H_PLANCK,
    K_BOLTZMANN,
    RamanGainModel,
    resolve_raman_gain_model,
)
from network_nlo_eval.network.elements import RamanPumpConfig

__all__ = [
    "DEFAULT_FIXED_POINT_ITERATIONS",
    "DEFAULT_PAPER_SPP_STEPS",
    "LAUNCH_POWER_CONVENTIONS",
    "PaperRamanSpanResult",
    "RamanPaperSpanSolver",
    "resolve_launch_power_convention",
]

#: 支持的入纤功率口径。
#:
#: ``"total"``：``power_in_w`` 是每信道（双偏振）总功率，历史默认；
#: ``"per_pol"``：``power_in_w`` 是每偏振功率，物理总功率是它的 2 倍。
LAUNCH_POWER_CONVENTIONS: tuple[str, ...] = ("total", "per_pol")

#: ODE 默认的 RK4 步数。与 PCFM 的 SPP 采样网格一致（每步取中点与终点）。
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
            按信号净增益搬运的部分。坐标与求解器的
            ``launch_power_convention`` 一致：``total`` 是双偏振总量，
            ``per_pol`` 是每偏振量。
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


def resolve_launch_power_convention(
    launch_power_convention: str,
    signal_polarizations: int,
) -> str:
    """校验入纤功率口径与状态偏振数的组合。

    ``signal_polarizations`` 只放大耦合项的**驱动功率**，不放大状态本身。因此
    只有两种自洽的声明：

    * ``"total"`` + ``signal_polarizations=1``：状态就是每信道总功率，驱动也
      用总功率，光子预算比值 1。这是历史默认。
    * ``"per_pol"`` + ``signal_polarizations=2``：状态是每偏振功率，驱动取
      ``2 * P`` 的双偏振总功率，物理总功率 = 2 × 输入。光子预算（按物理总
      功率计）比值 1。

    ``"total"`` + ``signal_polarizations=2`` 是把总功率再乘 2 驱动，物理上
    重复计数；该组合在求解器层保留（历史对照与倍率探针），但调用方必须显式
    warning，见 :class:`~network_nlo_eval.models.pcfm_h3_raman.PCFH3RamanModel`。
    ``"per_pol"`` + ``signal_polarizations=1`` 自相矛盾（声明每偏振却没有第二
    个偏振），直接报错。

    Args:
        launch_power_convention: ``"total"`` 或 ``"per_pol"``。
        signal_polarizations: ODE 信号状态代表的偏振数。

    Returns
    -------
        规范化后的口径名。

    Raises
    ------
        ValueError: 口径非法，或 ``"per_pol"`` 与 ``signal_polarizations`` 不匹配。
    """
    convention = str(launch_power_convention)
    if convention not in LAUNCH_POWER_CONVENTIONS:
        raise ValueError(
            f"launch_power_convention must be one of {LAUNCH_POWER_CONVENTIONS}, got {launch_power_convention!r}."
        )
    if convention == "per_pol" and int(signal_polarizations) != 2:
        raise ValueError(
            "launch_power_convention='per_pol' declares a per-polarization input, so the ODE state "
            "must be per-polarization: signal_polarizations must be 2, got "
            f"{int(signal_polarizations)}. Use launch_power_convention='total' for a "
            "per-channel total-power input."
        )
    return convention


class RamanPaperSpanSolver:
    r"""单跨信号/泵浦/ASE coupled ODE 求解器。

    状态向量按“信号在前、泵浦在后”拼接后按频率降序排列，耦合矩阵
    ``C`` 的第 ``(j, k)`` 项（j 为接收波、k 为源波）满足光子数守恒：

    * ``f_k > f_j``（源频率更高）：``C[j,k] = +g(|f_j-f_k|)``，j 获得功率；
    * ``f_k < f_j``（源频率更低）：``C[j,k] = -(f_j/f_k) g(|f_j-f_k|)``，j 失去功率。

    负耦合的频率因子取**接收波/源波**（``f_j/f_k``），因此对任意一对
    ``(j,k)`` 有 ``C[j,k]/f_j + C[k,j]/f_k = 0``（在 ``A_eff`` 与频率无关
    时严格成立），即 ``d/dz Σ_j P_j/f_j = 0``。

    .. note::
       论文 Eq. (2)/(2.39) 印刷的负耦合因子是 ``f_source/f_receiver``
       （``f_k/f_j``），与这里互为倒数。按印刷式实现的旧代码不满足上述
       光子数守恒判据：泵浦-信号对会少耗尽 ``(f_s/f_p)^2`` 的泵浦功率。
       本实现采用标准的接收/源频率因子；旧口径（论文印刷式）的数值可在
       ``e1ee4af`` 及其之前的结果中复现。详见 ``docs/dra-open-issues.md``。

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
        gain_model: 拉曼增益谱模型，或 ``"gaussian"``/``"linear"`` 字符串；
            表格谱用 :meth:`RamanGainModel.from_table` 构造后传入实例。
        n_steps: RK4 步数；剖面按每步中点/终点采样，共 ``2*n_steps`` 行。
        n_iterations: BW 泵不动点迭代上限。
        include_isrs: 是否计入信道-信道 ISRS；关闭后只保留泵浦相关耦合。
        include_ase: 是否求解 Eq. (4) 的 ASE 状态。关闭时 ASE 恒为零。
        signal_polarizations: 信号状态 ``P_s,i`` 代表的偏振数，只用于耦合项的
            **驱动功率** ``signal_polarizations * P_s,i``。默认 ``1`` 是历史的
            每偏振口径，逐位保持旧行为；``2`` 表示状态仍是每偏振功率，但
            信号-信号、泵浦-信号、泵浦-泵浦耦合的源功率取双偏振总功率，与
            物理波形 ``DistributedRamanFiber`` 的泵浦耗尽驱动一致。接收波
            自己的功率仍然只乘一次，因此不会像 ``solve(2 * power)`` 那样把
            信号状态本身也放大。
        launch_power_convention: ``power_in_w`` 的功率口径。``"total"``（默认）
            表示每信道双偏振总功率，``signal_polarizations=1`` 时逐位保持旧
            行为；``"per_pol"`` 表示每偏振功率，物理总功率是它的 2 倍，此时
            状态与输出都保持每偏振口径，只有耦合驱动取双偏振总功率，因此
            ``signal_polarizations`` 必须是 ``2``。两种口径是同一物理的两种
            坐标：``per_pol`` 用 ``P/2`` 得到的驱动与 ``total`` 用 ``P`` 完全
            相同，信号/泵浦剖面按每偏振功率等比缩放。
            ``total`` + ``signal_polarizations=2`` 是把总功率再乘 2 的
            **未声明单位倍率**（物理上重复计数），本层保留它只为历史对照，
            模型层会就此发出 warning。
        pump_depletion: 泵浦是否被信号/其他泵浦耗尽。``True``（默认）解完整
            耦合 Eq. (2)；``False`` 把泵浦行从耦合矩阵中清零，泵浦只按自身
            衰减 ``dP_p/dz = -alpha_p P_p`` 演化；信号仍从泵浦获得增益，泵浦-泵浦
            耦合被关闭。这是一个**对照口径**，用于和采用未耗尽泵浦的波形/物理
            参考比较，不是更物理的默认（真实 DRA 在高信号功率下会发生泵浦耗尽）。
        pump_pump_raman: 是否计入泵浦-泵浦拉曼耦合。``True``（默认）保持历史
            行为，六泵 C+L 的泵浦之间会互相转移功率；``False`` 只把耦合矩阵
            的泵浦×泵浦块清零，泵浦仍被信号耗尽、仍按自身 ``alpha_p`` 衰减，
            泵浦-信号与信号-信号耦合逐项不变。这是一个**对照口径**：它把
            “泵浦之间互相搬运功率”从 C+L 残差里隔离出来，不是更物理的默认。

    Raises
    ------
        ValueError: 任一数组形状不匹配、跨长/步数非正、布尔开关不是 ``bool``；
            口径非法或 ``"per_pol"`` 与 ``signal_polarizations`` 不匹配。
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
        gain_model: RamanGainModel | str,
        *,
        n_steps: int = DEFAULT_PAPER_SPP_STEPS,
        n_iterations: int = DEFAULT_FIXED_POINT_ITERATIONS,
        include_isrs: bool = True,
        include_ase: bool = True,
        pump_depletion: bool = True,
        signal_polarizations: int = 1,
        pump_pump_raman: bool = True,
        launch_power_convention: str = "total",
    ) -> None:
        """校验参数并构建与 z 无关的耦合矩阵。"""
        if length_m <= 0:
            raise ValueError("length_m must be positive.")
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        if n_iterations < 1:
            raise ValueError("n_iterations must be >= 1.")
        if int(signal_polarizations) < 1:
            raise ValueError("signal_polarizations must be >= 1.")
        if not isinstance(pump_pump_raman, (bool, np.bool_)):
            raise ValueError(f"pump_pump_raman must be a bool, got {pump_pump_raman!r}.")
        resolved_convention = resolve_launch_power_convention(launch_power_convention, signal_polarizations)

        self.signal_frequencies_hz = np.asarray(signal_frequencies_hz, dtype=np.float64)
        self.signal_bandwidths_hz = np.asarray(signal_bandwidths_hz, dtype=np.float64)
        self.signal_alpha_npm = np.asarray(signal_alpha_npm, dtype=np.float64)
        self.signal_a_eff_m2 = np.asarray(signal_a_eff_m2, dtype=np.float64)
        self.pumps = list(pumps)
        self.length_m = float(length_m)
        self.gain_model = resolve_raman_gain_model(gain_model)
        self.n_steps = int(n_steps)
        self.n_iterations = int(n_iterations)
        self.include_isrs = bool(include_isrs)
        self.include_ase = bool(include_ase)
        self.pump_depletion = bool(pump_depletion)
        self.signal_polarizations = int(signal_polarizations)
        self.pump_pump_raman = bool(pump_pump_raman)
        self.launch_power_convention = resolved_convention

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

        # 驱动功率 = 状态功率 * 该波的偏振数。信号状态是每偏振功率，泵浦是
        # 单偏振泵浦功率；``signal_polarizations=1`` 时该因子恒为 1，逐位不变。
        # ``launch_power_convention="per_pol"`` 要求 ``signal_polarizations=2``，
        # 于是驱动恰为双偏振总功率，而状态/输出仍是每偏振功率。
        self._drive_scale = np.where(
            self._is_pump,
            1.0,
            float(self.signal_polarizations),
        ).astype(np.float64)

        # ASE 坐标由 ``launch_power_convention`` 决定：``total``（默认，含历史
        # 的 ``signal_polarizations=2`` 倍率对照）保留 ``2*h*nu*B`` 的双偏振
        # 源项；``per_pol`` 的 ASE 是每偏振量，源项降为 ``1*h*nu*B``，但源
        # 功率仍用物理总功率（``drive``，信号槽乘偏振数）驱动。这样
        # ``per_pol`` 的 raman_ase_w 恰为 ``total`` 的一半，输入 ASE 与输出
        # ASE 在同一坐标，route 层不需要再缩放。
        self._ase_source_scale = 1.0 if resolved_convention == "per_pol" else 2.0

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
        """构建 ``(M, M)`` 拉曼耦合矩阵 (1/(W·m))。

        负耦合（接收波频率高于源波）按光子数守恒取 ``-(f_receiver/f_source)``，
        使 ``C[j,k]/f_j + C[k,j]/f_k = 0``。``A_eff`` 仍按接收波归一，因此
        逐波 ``A_eff`` 不同时会残留一个与频率因子无关的 ``O(ΔA_eff/A_eff)``
        项；本函数只修正 Eq. (2) 的频率因子，不改 ``A_eff`` 口径。

        ``pump_depletion=False`` 时把泵浦行（含泵浦-泵浦项）清零：泵浦不再
        从信号或其他泵浦获得/失去功率，只剩自身 ``-alpha_p P_p`` 衰减，而
        信号/泵浦仍通过信号行从泵浦获得增益。这是未耗尽泵浦的对照口径。

        ``pump_pump_raman=False`` 只清零 ``(泵浦, 泵浦)`` 子块。状态向量的
        顺序是“信号在前、泵浦在后”再按频率降序重排，泵浦槽由
        ``self._is_pump`` 标出，因此这里用显式掩码而不是行列区间，避免把
        泵浦-信号项（泵浦行/信号列与信号行/泵浦列）一起误伤。
        """
        delta = np.abs(self._f_hz[:, None] - self._f_hz[None, :])
        shape = self.gain_model.gain_shape(delta)
        coefficient = self.gain_model.peak_gain_coefficient_m_per_w / self._a_eff_m2[:, None]
        gain = coefficient * shape
        # 源频率更高 -> 接收波获得功率；源频率更低 -> 接收波把功率交给源，
        # 并按光子数守恒乘 f_receiver/f_source。
        source_higher = self._f_hz[None, :] > self._f_hz[:, None]
        coupling = np.where(source_higher, gain, -(self._f_hz[:, None] / self._f_hz[None, :]) * gain)
        np.fill_diagonal(coupling, 0.0)
        if not self.include_isrs:
            signal_slots = self._signal_slots
            coupling[np.ix_(signal_slots, signal_slots)] = 0.0
        if not self.pump_pump_raman:
            pump_slots = np.flatnonzero(self._is_pump)
            coupling[np.ix_(pump_slots, pump_slots)] = 0.0
        if not self.pump_depletion:
            pump_slots = np.flatnonzero(self._is_pump)
            coupling[pump_slots, :] = 0.0
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
        # 耦合的源功率用总功率（信号槽乘偏振数），接收功率 ``power`` 仍只乘一次。
        drive = total * self._drive_scale
        gain_rate = self._coupling @ drive
        gain_fwd = gain_rate[self._fwd_slots]
        d_power = power * gain_fwd - self._alpha_fwd * power
        if not self.include_ase:
            return d_power, np.zeros_like(ase)
        # 自发源项：``total`` 坐标用双偏振状态与 2 个接收偏振；``per_pol``
        # 坐标的 ASE 只有每偏振一份，必须用物理总功率（``drive``）驱动，否则
        # 信号-信号项会因状态减半而低 2 倍。源项系数只在 ``per_pol`` 时降为 1。
        if self._ase_source_scale == 2.0:
            ase_drive = total
        else:
            ase_drive = drive
        source = self._ase_source_scale * H_PLANCK * self._b_fwd * self._f_fwd * (self._source_fwd @ ase_drive)
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
        drive = total * self._drive_scale
        gain_rate = self._coupling @ drive
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
            其坐标必须与 ``launch_power_convention`` 一致：``total`` 时是
            双偏振总 ASE，``per_pol`` 时是每偏振 ASE。返回值
            :attr:`PaperRamanSpanResult.raman_ase_w` 使用同一坐标。

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
        # 未耗尽口径下反向泵不依赖信号，预设指数剖面就是解，无需不动点迭代。
        converged = self._bw_slots.size == 0 or not self.pump_depletion

        if self._bw_slots.size and self.pump_depletion:
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
