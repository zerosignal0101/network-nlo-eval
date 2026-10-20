"""路由累积与元件后处理：所有 QoT 模型共用的一份实现。

模型之间只许在"单跨物理"上不同。放大器增益/ASE、ROADM 损耗与均衡、滤波惩罚、
阻断、逐跨功率传递与 SNR 合成都放在这里，换模型不会改变这些语义。

浮点求和顺序是契约的一部分：`transfer -> 旧噪声乘以 transfer -> 再加本跨噪声`
的顺序一旦改动就可能动末位，从而静默改变 SNR 判定。`tests/test_route_model.py`
的黄金向量守卫会抓住这种漂移。
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    MultiSpanOpticalPath,
    PathQoTResult,
    QoTModel,
    SpanFiberOutput,
    SpanKernel,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

__all__ = [
    "H_PLANCK",
    "accumulate_route",
    "apply_amplifier_and_roadm",
    "resolve_noise_figure",
]

#: 普朗克常数 (J·s)。ASE 项 ``NF * h * nu * B_i * (G-1)`` 的系数。
H_PLANCK = 6.62607015e-34


def resolve_noise_figure(
    edfa_config: EDFAConfig | None,
    default_nf_lin: NDArrayFloat,
    n_nodes: int,
) -> NDArrayFloat:
    """把 EDFA 配置折成线性噪声系数数组。

    未给 EDFA 时返回全 1（不产生 ASE 的语义由调用方保证）。标量 NF 广播到全部
    节点；数组 NF 必须逐节点对齐，长度不符直接报错，不做补齐猜测。

    Args:
        edfa_config: 跨段放大器配置；``None`` 表示该跨无放大。
        default_nf_lin: ``(N,)`` 波长相关的默认线性 NF，仅在无 EDFA 时取用。
        n_nodes: 节点数。

    Returns
    -------
        ``(N,)`` 线性噪声系数。

    Raises
    ------
        ValueError: 数组型 NF 的长度与节点数不符。
    """
    if edfa_config is None:
        return np.ones_like(default_nf_lin)

    noise_figure_db = np.asarray(edfa_config.noise_figure_db, dtype=float)
    if noise_figure_db.ndim == 0:
        return np.full(n_nodes, 10 ** (float(noise_figure_db) / 10))
    if noise_figure_db.shape != (n_nodes,):
        raise ValueError("EDFA noise-figure array must contain one value per channel.")
    return 10 ** (noise_figure_db / 10)


def apply_amplifier_and_roadm(
    *,
    power_in_w: NDArrayFloat,
    fiber_output: SpanFiberOutput,
    edfa_config: EDFAConfig | None,
    roadm_config: ROADMConfig | None,
    nf_lin: NDArrayFloat,
    f_abs_hz: NDArrayFloat,
    bandwidth_hz: NDArrayFloat,
) -> ElementOutput:
    """把单跨光纤结果接到放大器（与可选 ROADM），得到出跨功率与 ASE。

    ``edfa_config`` 为 ``None`` 时不放大、不产生 ASE；``target_gain_db`` 为
    ``None`` 时按"补偿本跨损耗"取增益 ``P_in / P_out_fiber``。ROADM 的确定性损耗
    同等衰减信号与已生成噪声，但**不**衰减 booster 新产生的 ASE（与旧实现一致：
    booster 在同一节点上先补偿损耗再计入 ASE）。

    Args:
        power_in_w: ``(N,)`` 本跨入口功率 (W)，用于增益与"未占用"判定。
        fiber_output: 本跨光纤结果。
        edfa_config: 跨段放大器配置。
        roadm_config: 本跨末端的 ROADM 配置；``None`` 表示无。
        nf_lin: ``(N,)`` 线性噪声系数。
        f_abs_hz: ``(N,)`` 中心绝对频率 (Hz)。
        bandwidth_hz: ``(N,)`` 每信道带宽 ``B_i`` (Hz)。

    Returns
    -------
        出跨功率与三项噪声。
    """
    power_out_fiber = fiber_output.power_out_w
    spm = fiber_output.spm_noise_w
    xpm = fiber_output.xpm_noise_w
    amp_spm = getattr(fiber_output, "nli_amplitude_spm", None)
    amp_xpm = getattr(fiber_output, "nli_amplitude_xpm", None)

    if edfa_config is None:
        power_out = power_out_fiber
        ase = np.zeros_like(power_in_w)
    else:
        if edfa_config.target_gain_db is None:
            gain = np.divide(
                power_in_w,
                power_out_fiber,
                out=np.ones_like(power_in_w),
                where=power_out_fiber > 0,
            )
        else:
            gain = np.full_like(power_in_w, 10 ** (edfa_config.target_gain_db / 10))
        if edfa_config.gain_ripple_db is not None:
            ripple = np.asarray(edfa_config.gain_ripple_db, dtype=float)
            if ripple.shape != bandwidth_hz.shape:
                raise ValueError("EDFA gain-ripple array must contain one value per channel.")
            gain *= 10 ** (ripple / 10)
        power_out = power_out_fiber * gain
        ase = nf_lin * H_PLANCK * f_abs_hz * bandwidth_hz * np.maximum(gain - 1, 0)
        ase[power_in_w <= 0] = 0
        if amp_spm is not None:
            # 噪声功率被放大 gain 倍，振幅只放大 sqrt(gain)。
            amp_root = np.sqrt(gain)
            amp_spm = amp_spm * amp_root
            amp_xpm = amp_xpm * amp_root

    if roadm_config is not None:
        loss_lin = 10 ** (roadm_config.path_loss_db() / 10)
        power_out /= loss_lin
        spm /= loss_lin
        xpm /= loss_lin
        ase /= loss_lin
        if amp_spm is not None:
            amp_root = np.sqrt(1.0 / loss_lin)
            amp_spm = amp_spm * amp_root
            amp_xpm = amp_xpm * amp_root

    if amp_spm is None:
        return ElementOutput(power_out, spm, xpm, ase)
    return ElementOutput(power_out, spm, xpm, ase, amp_spm, amp_xpm)


def accumulate_route(
    model: QoTModel,
    optical_path: MultiSpanOpticalPath,
    power_in_w: NDArrayFloat,
) -> PathQoTResult:
    """按拓扑顺序把逐跨结果累加成路径级 QoT。

    逐跨递推：``totals *= transfer``（已累积噪声随功率同比例变化），再加本跨
    新产生的噪声。ROADM 处按 ``equalize_output_power`` 二选一：均衡时信号不被
    ``loss`` 衰减，代价是 booster ASE ``NF*h*f*B_i*(loss-1)``；不均衡时信号与
    噪声一起除以 ``loss``。滤波惩罚只作用于噪声。

    Args:
        model: 提供逐节点数组与单跨步进的模型。
        optical_path: 有序逐跳路径。
        power_in_w: ``(N,)`` 入口功率 (W)，非负。

    Returns
    -------
        逐信道信号功率、SPM/XPM/ASE 噪声与 SNR。

    Raises
    ------
        ValueError: 功率向量形状不符或含负值。
    """
    power = np.asarray(power_in_w, dtype=np.float64).copy()
    if power.shape != (model.n_nodes,) or np.any(power < 0):
        raise ValueError("power_in_w must be a non-negative vector with one value per channel.")
    power = power * model.power_width_scale()

    total_spm = np.zeros_like(power)
    total_xpm = np.zeros_like(power)
    total_ase = np.zeros_like(power)
    # 相干档：模型给了复振幅就按 |Σ a_k|² 累加，否则走原来的功率相加。
    # None 表示尚未判定（路径上还没有跨段）。
    coherent: bool | None = None
    amp_spm = np.zeros_like(power, dtype=np.complex128)
    amp_xpm = np.zeros_like(power, dtype=np.complex128)
    span_count = 0
    roadm_configs = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
    hop_count = len(optical_path.fiber_configs)

    for hop_index, (fiber, edfa, roadm) in enumerate(
        zip(optical_path.fiber_configs, optical_path.edfa_configs, roadm_configs, strict=True)
    ):
        remaining_km = fiber.length_km
        while remaining_km > 1e-12:
            span_km = min(remaining_km, fiber.max_span_length_km)
            previous_power = power
            step = model.span_step(fiber, edfa, power, span_km * 1e3)
            power = step.power_out_w
            transfer = np.divide(power, previous_power, out=np.zeros_like(power), where=previous_power > 0)
            step_amp = getattr(step, "nli_amplitude_spm", None)
            if step_amp is None:
                if coherent:
                    raise TypeError(
                        "model returned a coherent NLI amplitude for an earlier span "
                        "and a power-only result for a later one; a model must pick one convention."
                    )
                coherent = False
                total_spm *= transfer
                total_xpm *= transfer
                total_ase *= transfer
                total_spm += step.spm_noise_w
                total_xpm += step.xpm_noise_w
                total_ase += step.ase_noise_w
            else:
                if coherent is False:
                    raise TypeError(
                        "model returned a power-only NLI result for an earlier span "
                        "and a coherent amplitude for a later one; a model must pick one convention."
                    )
                coherent = True
                amp_root = np.sqrt(transfer)
                amp_spm = amp_spm * amp_root + step_amp
                amp_xpm = amp_xpm * amp_root + step.nli_amplitude_xpm
                total_ase *= transfer
                total_ase += step.ase_noise_w
            span_count += 1
            remaining_km -= span_km

        if roadm is None:
            continue
        blocked = model.blocked_nodes(roadm)
        operation = "drop" if hop_index == hop_count - 1 else "express"
        loss_db = roadm.path_loss_db(operation)
        loss = 10 ** (loss_db / 10)
        power /= loss
        if coherent:
            amp_root = np.sqrt(1.0 / loss)
            amp_spm *= amp_root
            amp_xpm *= amp_root
        else:
            total_spm /= loss
            total_xpm /= loss
        total_ase /= loss
        if roadm.equalize_output_power:
            gain = loss
            power *= gain
            if coherent:
                amp_root = np.sqrt(gain)
                amp_spm *= amp_root
                amp_xpm *= amp_root
            else:
                total_spm *= gain
                total_xpm *= gain
            total_ase *= gain
            nf = 10 ** (roadm.booster_noise_figure_db / 10)
            booster_ase = (
                nf * H_PLANCK * model.f_abs_hz * model.channel_bandwidth_hz_channels * (gain - 1)
            )
            booster_ase[power <= 0] = 0
            total_ase += booster_ase
        penalty = 10 ** (roadm.filtering_penalty_db / 10)
        if coherent:
            amp_root = np.sqrt(penalty)
            amp_spm *= amp_root
            amp_xpm *= amp_root
        else:
            total_spm *= penalty
            total_xpm *= penalty
        total_ase *= penalty
        if blocked:
            power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0
            if coherent:
                amp_spm[blocked] = 0
                amp_xpm[blocked] = 0

    if coherent:
        total_spm = np.abs(amp_spm) ** 2
        total_xpm = np.abs(amp_xpm) ** 2

    total_noise = total_spm + total_xpm + total_ase
    snr = np.full_like(power, -np.inf)
    active = (power > 0) & (total_noise > 0)
    snr[active] = 10 * np.log10(power[active] / total_noise[active])
    snr[(power > 0) & (total_noise == 0)] = np.inf
    return PathQoTResult(
        power,
        total_spm,
        total_xpm,
        total_ase,
        snr,
        span_count,
        sum(roadm is not None for roadm in roadm_configs),
    )


class RouteModel:
    """新模型的基类：逐节点数组 + 单跨核缓存 + 共用路由累积。

    子类只需要实现 :meth:`_build_kernel`（返回该光纤参数下的 :class:`SpanKernel`），
    并设置类属性 :attr:`info`。核的构造按 ``FiberSpanConfig`` 的 JSON 记忆化，
    因为逐信道常量（波长、A_eff、色散、成对矩阵）在整个仿真中不变。
    """

    info: object  # ModelInfo，子类必须给出

    def __init__(
        self,
        grid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections=None,
        slots_per_channel: int | None = None,
        channel_bandwidth_source: str = "symbol_rate",
        power_width_mode: str = "fixed_power",
    ) -> None:
        """初始化节点几何；物理量由子类内核在构造期自行准备。

        Args:
            grid: ``SpectrumGrid``（遗留固定栅格）或 ``FlexibleGrid``（需给 connections）。
            ref_fiber_config: 参考光纤配置，用于波长相关量。
            symbol_rate_hz: 符号率 (Hz)。
            connections: 逐连接模式下的连接列表。
            slots_per_channel: 每遗留信道占用的槽数，用于展开 ROADM 阻断信道。
            channel_bandwidth_source: ``B_i`` 口径，``"symbol_rate"`` 或 ``"occupied"``。
            power_width_mode: 功率归一化口径，``"fixed_power"`` 或 ``"fixed_psd"``。
        """
        from network_nlo_eval.core.flexgrid import FlexibleGrid
        from network_nlo_eval.core.spectrum import SpectrumGrid

        if channel_bandwidth_source not in ("symbol_rate", "occupied"):
            raise ValueError(
                f"channel_bandwidth_source must be 'symbol_rate' or 'occupied', got {channel_bandwidth_source!r}."
            )
        if power_width_mode not in ("fixed_power", "fixed_psd"):
            raise ValueError(f"power_width_mode must be 'fixed_power' or 'fixed_psd', got {power_width_mode!r}.")
        if slots_per_channel is not None and slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")

        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.connections = list(connections) if connections is not None else None
        self.slots_per_channel = slots_per_channel
        self.channel_bandwidth_source = channel_bandwidth_source
        self.power_width_mode = power_width_mode

        if self.connections is not None:
            if not isinstance(grid, FlexibleGrid):
                raise TypeError("connections= requires a FlexibleGrid.")
            self._f_abs_hz = np.array([conn.centre_hz for conn in self.connections], dtype=np.float64)
            self._lambdas_m = 2.99792458e8 / self._f_abs_hz
        elif isinstance(grid, SpectrumGrid):
            self._lambdas_m = grid.wavelengths
            self._f_abs_hz = grid.frequencies
        else:
            raise TypeError("Legacy mode requires a SpectrumGrid; pass connections= for a FlexibleGrid.")

        self._kernel_cache: dict[str, SpanKernel] = {}

    # --- 几何 -------------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        """节点数：遗留模式 = 信道数；逐连接模式 = 连接数。"""
        return int(self._f_abs_hz.shape[0])

    @property
    def f_abs_hz(self) -> NDArrayFloat:
        """``(N,)`` 节点中心绝对频率 (Hz)。"""
        return self._f_abs_hz

    @property
    def lambdas_m(self) -> NDArrayFloat:
        """``(N,)`` 节点中心波长 (m)。"""
        return self._lambdas_m

    @property
    def band_center_hz(self) -> float:
        """``f_rel_hz`` 的参考原点：遗留网格取中心频率，灵活栅格取频段中心。"""
        from network_nlo_eval.core.flexgrid import FlexibleGrid

        if isinstance(self.grid, FlexibleGrid):
            return self.grid.band_center_hz
        return self.grid.center_frequency_hz

    @property
    def f_rel_hz(self) -> NDArrayFloat:
        """``(N,)`` 相对频率 (Hz)。"""
        return self._f_abs_hz - self.band_center_hz

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """``(N,)`` 逐节点信道带宽 ``B_i`` (Hz)。

        Raises
        ------
            ValueError: 取到的带宽非正或非有限。
        """
        if self.connections is None:
            return np.full(self.n_nodes, float(self.symbol_rate_hz), dtype=np.float64)
        if self.channel_bandwidth_source == "occupied":
            values = np.array([float(conn.occupied_bandwidth_hz) for conn in self.connections], dtype=np.float64)
        else:
            values = np.array([float(conn.symbol_rate_hz) for conn in self.connections], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(
                f"per-channel bandwidth B_i (source={self.channel_bandwidth_source!r}) must be "
                f"finite and positive; got {values.tolist()}."
            )
        return values

    def power_width_scale(self) -> NDArrayFloat:
        """``fixed_psd`` 口径下的逐节点功率缩放；``fixed_power`` 恒为 1。"""
        if self.power_width_mode == "fixed_power":
            return np.ones(self.n_nodes, dtype=np.float64)
        return self.channel_bandwidth_hz_channels / float(self.symbol_rate_hz)

    # --- 阻断 -------------------------------------------------------------
    def blocked_nodes(self, roadm_config: ROADMConfig) -> list[int]:
        """该 ROADM 阻断的全部节点索引。

        逐连接模式按槽区间重叠判定，``blocked_slots`` 优先，否则把
        ``blocked_channels`` 按 ``slots_per_channel`` 展开；缺 ``slots_per_channel``
        时直接报错，不做取整猜测。
        """
        from network_nlo_eval.network.slots import connection_overlaps_slots

        if self.connections is None:
            return [index for index in roadm_config.blocked_channels if index < self.n_nodes]

        blocked_slots = set(roadm_config.blocked_slots)
        if roadm_config.blocked_channels:
            if not self.slots_per_channel:
                raise ValueError(
                    "Per-connection mode needs slots_per_channel "
                    "(channel_spacing_hz / slot_width_hz) to expand ROADM "
                    "blocked_channels into slots. Construct the evaluator with "
                    "slots_per_channel=..., or use blocked_slots."
                )
            for channel_index in roadm_config.blocked_channels:
                blocked_slots.update(
                    range(channel_index * self.slots_per_channel, (channel_index + 1) * self.slots_per_channel)
                )
        return [
            index
            for index, conn in enumerate(self.connections)
            if connection_overlaps_slots(conn.first_slot, conn.n_slots, blocked_slots)
        ]

    # --- 核 ---------------------------------------------------------------
    def _build_kernel(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """构造该光纤参数下的单跨核。子类必须实现。"""
        raise NotImplementedError("RouteModel subclasses must implement _build_kernel().")

    def kernel_for(self, fiber_config: FiberSpanConfig) -> SpanKernel:
        """返回匹配光纤参数的缓存核。"""
        key = fiber_config.model_dump_json()
        kernel = self._kernel_cache.get(key)
        if kernel is None:
            kernel = self._build_kernel(fiber_config)
            self._kernel_cache[key] = kernel
        return kernel

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """单跨：调用核，再统一走放大器/ASE。"""
        kernel = self.kernel_for(fiber_config)
        fiber_output = kernel.evaluate(power_in_w, span_length_m, 1)
        nf_lin = resolve_noise_figure(edfa_config, self.default_nf_lin, self.n_nodes)
        return apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=None,
            nf_lin=nf_lin,
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )

    @property
    def default_nf_lin(self) -> NDArrayFloat:
        """无 EDFA 时使用的默认线性 NF。默认全 1，子类可按波长覆盖。"""
        return np.ones(self.n_nodes, dtype=np.float64)

    # --- 路径 -------------------------------------------------------------
    def evaluate_path(self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat) -> PathQoTResult:
        """沿路径评估，返回逐信道结果。"""
        return accumulate_route(self, optical_path, power_in_w)
