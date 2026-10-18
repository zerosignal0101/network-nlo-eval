"""多波段 ISRS-GN 评估器。

ISRS-GN = Inter-channel Stimulated Raman Scattering - Generalized Noise。
"""

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_nf_lin_jit,
    _calc_span_noise_and_power_jit,
    _db_to_lin,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.slots import connection_overlaps_slots

_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)]

#: 逐信道带宽 ``B_i`` 的取值口径。
#:
#: ``"symbol_rate"``
#:     ``B_i`` = :attr:`ConnectionSpec.symbol_rate_hz`。默认档，与旧行为逐位一致。
#: ``"occupied"``
#:     ``B_i`` = :attr:`ConnectionSpec.occupied_bandwidth_hz`（n_slots × 槽宽）。
#:     只有这一档让多槽宽度真正进入 QoT。
#:
#: 论文只说 "channel bandwidth"，未区分符号率与占用带，故本仓显式暴露该选择。
CHANNEL_BANDWIDTH_SOURCES = ("symbol_rate", "occupied")
CHANNEL_BANDWIDTH_DEFAULT = "symbol_rate"

#: ``P_i`` 与 ``B_i`` 的归一化口径。
#:
#: ``"fixed_power"``
#:     每条连接的 ``P_i`` 即其入纤总功率，与 ``B_i`` 无关。默认档，与
#:     :attr:`ConnectionSpec.launch_power_w` 语义一致：宽连接 = 同样功率摊到
#:     更宽的谱上，PSD 更低。
#: ``"fixed_psd"``
#:     恒定功率谱密度：``P_i_eff = P_i × B_i / symbol_rate_hz``，以构造参数
#:     ``symbol_rate_hz`` 为锚点。``B_i`` 均匀时缩放因子恒为 1.0。
POWER_WIDTH_MODES = ("fixed_power", "fixed_psd")
POWER_WIDTH_DEFAULT = "fixed_power"

#: 单跨段计算的加速档位，只影响数值路径，不改物理。
#:
#: ``"off"``
#:     每次调用都进 njit 核。默认档。
#: ``"cached"``
#:     按 (入纤功率, 跨段长度, 跨段数, NF) 记忆化单跨段结果，结果与 ``"off"``
#:     逐位相同。
ODE_ACCEL_MODES = ("off", "cached")
ODE_ACCEL_DEFAULT = "off"


@dataclass
class MultiSpanOpticalPath:
    """定义一条光路，包含多个跨段及其对应的配置。"""

    fiber_configs: list[FiberSpanConfig]
    edfa_configs: list[EDFAConfig]
    roadm_configs: list[ROADMConfig] | None = None
    # 实际路径中的每个 link_key, 方便关联到 network_state
    link_keys: list[tuple[int, int]] = field(default_factory=list)
    node_path: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """验证所有逐跳配置长度一致."""
        count = len(self.fiber_configs)
        if count == 0 or len(self.edfa_configs) != count:
            raise ValueError("Optical path requires matching non-empty fiber and EDFA config lists.")
        if self.roadm_configs is not None and len(self.roadm_configs) != count:
            raise ValueError("ROADM config count must match the number of route links.")
        if self.link_keys and len(self.link_keys) != count:
            raise ValueError("Link key count must match the number of route links.")


@dataclass(frozen=True)
class PathQoTResult:
    """统计物理层沿路径计算的逐信道结果."""

    signal_power_w: NDArrayFloat
    spm_noise_w: NDArrayFloat
    xpm_noise_w: NDArrayFloat
    ase_noise_w: NDArrayFloat
    snr_db: NDArrayFloat
    span_count: int
    roadm_count: int


class MultiBandISRSGN:
    """多波段 ISRS-GN 模型评估器。

    用于快速计算多波段光网络中单跨段或多跨段的传输质量 (QoT)。
    """

    def __init__(
        self,
        grid: SpectrumGrid | FlexibleGrid,
        ref_fiber_config: FiberSpanConfig,
        symbol_rate_hz: float = 50e9,
        connections: Sequence[ConnectionSpec] | None = None,
        ode_accel: str = ODE_ACCEL_DEFAULT,
        slots_per_channel: int | None = None,
        channel_bandwidth_source: str = CHANNEL_BANDWIDTH_DEFAULT,
        power_width_mode: str = POWER_WIDTH_DEFAULT,
    ):
        """初始化 ISRS-GN 评估器。

        Args:
            grid: 频谱网格配置（遗留 ``SpectrumGrid`` 或 ``FlexibleGrid``）。
            ref_fiber_config: 参考光纤配置，用于计算各种波长相关参数。
            symbol_rate_hz: 系统的符号速率 (Hz)。
            connections: 灵活栅格连接列表。给定后进入逐连接模式：每条连接一个
                节点，频率取连接中心，带边缘按实际占用带计算。为 ``None`` 时
                保持遗留行为。
            ode_accel: 加速档位，见 :data:`ODE_ACCEL_MODES`。
            slots_per_channel: 每遗留信道占用的槽数，仅用于把 ROADM 的
                ``blocked_channels`` 展开为槽（见 :meth:`roadm_blocks_node`）。
                逐连接模式下需要该展开时必须显式传入，不做取整猜测。
            channel_bandwidth_source: 逐信道 ``B_i`` 口径，见
                :data:`CHANNEL_BANDWIDTH_SOURCES`。
            power_width_mode: 功率归一化口径，见 :data:`POWER_WIDTH_MODES`。

        Raises
        ------
            ValueError: 口径取值非法。
        """
        if ode_accel not in ODE_ACCEL_MODES:
            raise ValueError(f"ode_accel must be one of {ODE_ACCEL_MODES}, got {ode_accel!r}.")
        if slots_per_channel is not None and slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")
        if channel_bandwidth_source not in CHANNEL_BANDWIDTH_SOURCES:
            raise ValueError(
                f"channel_bandwidth_source must be one of {CHANNEL_BANDWIDTH_SOURCES}, "
                f"got {channel_bandwidth_source!r}."
            )
        if power_width_mode not in POWER_WIDTH_MODES:
            raise ValueError(f"power_width_mode must be one of {POWER_WIDTH_MODES}, got {power_width_mode!r}.")

        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.cr_w_m_hz = _SRS_CR  # 拉曼增益系数
        self.k_bar = min(sum(pump.power_w for pump in ref_fiber_config.raman_pumps), 1.0)
        self.delta_f_co_hz = 15e12  # 石英光纤受激拉曼散射（SRS）增益谱的截断频率
        self.n2_m2_w = ref_fiber_config.nonlinear_index_n2
        self.ode_accel = ode_accel
        self.slots_per_channel = slots_per_channel
        self.channel_bandwidth_source = channel_bandwidth_source
        self.power_width_mode = power_width_mode

        self.connections = list(connections) if connections is not None else None

        self._span_cache: dict[tuple, tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = {}
        self._precompute_channel_dependent_properties()
        self._fiber_evaluator_cache: dict[str, MultiBandISRSGN] = {}

    @property
    def n_nodes(self) -> int:
        """节点数：遗留模式 = 信道数；逐连接模式 = 活跃连接数。"""
        return int(self.f_abs_hz.shape[0])

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """逐节点信道带宽 ``B_i`` 数组 (Hz)，长度 = :attr:`n_nodes`。

        SPM/XPM/ASE 三处公式都读它。遗留模式下恒为 ``np.full(n_nodes,
        symbol_rate_hz)``，与旧的全局标量是同一个 float64，数值不变；逐连接模式
        按 :attr:`channel_bandwidth_source` 从 :class:`ConnectionSpec` 取，支持
        逐连接异构带宽。

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

    def _power_width_scale(self) -> NDArrayFloat:
        """``fixed_psd`` 口径下的逐节点功率缩放因子。

        恒定功率谱密度时 ``P_i_eff = P_i × B_i / symbol_rate_hz``；``B_i`` 均匀时
        因子恒为 1.0。``fixed_power`` 口径恒返回全 1。
        """
        if self.power_width_mode == "fixed_power":
            return np.ones(self.n_nodes, dtype=np.float64)
        return self.channel_bandwidth_hz_channels / float(self.symbol_rate_hz)

    def roadm_blocks_node(self, roadm: ROADMConfig, node_index: int) -> bool:
        """判断某节点（连接）是否被该 ROADM 阻断。

        逐连接模式按槽区间重叠判定：连接与任一阻断槽重叠即不可用。遗留模式按
        信道索引判定。解析顺序为 ``blocked_slots`` 优先，否则把
        ``blocked_channels`` 按 ``slots_per_channel`` 展开；缺少
        ``slots_per_channel`` 时直接报错，不做取整猜测。
        """
        if self.connections is not None:
            if node_index >= len(self.connections):
                return False
            blocked_slots = set(roadm.blocked_slots)
            if roadm.blocked_channels:
                if not self.slots_per_channel:
                    raise ValueError(
                        "Per-connection mode needs slots_per_channel "
                        "(channel_spacing_hz / slot_width_hz) to expand ROADM "
                        "blocked_channels into slots. Construct the evaluator with "
                        "slots_per_channel=..., or use blocked_slots."
                    )
                per_channel = self.slots_per_channel
                for channel_index in roadm.blocked_channels:
                    blocked_slots.update(range(channel_index * per_channel, (channel_index + 1) * per_channel))
            conn = self.connections[node_index]
            return connection_overlaps_slots(conn.first_slot, conn.n_slots, blocked_slots)
        return node_index in roadm.blocked_channels

    @property
    def band_center_hz(self) -> float:
        """``f_rel_hz`` 的参考原点。遗留网格用中心频率，灵活栅格用频段中心。"""
        if isinstance(self.grid, FlexibleGrid):
            return self.grid.band_center_hz
        return self.grid.center_frequency_hz

    def _precompute_channel_dependent_properties(self) -> None:
        """预计算所有信道通用的波长依赖物理属性，如波长、有效模面积等。

        这些属性在整个仿真中通常不变。
        """
        if self.connections is not None:
            # 每连接一个节点，节点频率 = 连接中心频率。
            self.f_abs_hz = np.array([conn.centre_hz for conn in self.connections], dtype=np.float64)
            self.lambdas_m = C_LIGHT / self.f_abs_hz  # (m)
        else:
            self.lambdas_m = self.grid.wavelengths  # (m)
            self.f_abs_hz = self.grid.frequencies  # (Hz)

        # 衰减系数 (Np/m)
        if self.ref_fiber_config.use_attenuation_polynomial:
            self.alpha_power_npm_channels = _calc_fiber_attenuation_npm(
                self.lambdas_m,
                self.ref_fiber_config.reference_wavelength_nm,
                self.ref_fiber_config.attenuation_alpha2,
                self.ref_fiber_config.attenuation_alpha1,
                self.ref_fiber_config.attenuation_alpha0,
            )
        else:
            alpha = self.ref_fiber_config.attenuation_db_km_ref * np.log(10) / 10 / 1e3
            self.alpha_power_npm_channels = np.full(self.n_nodes, alpha)

        # 有效模面积 (m^2)
        self.a_eff_m2_channels = _calc_aeff_dynamic(
            self.lambdas_m,
            self.ref_fiber_config.effective_area_um2_ref,
            self.ref_fiber_config.aeff_slope_um2_nm,
            self.ref_fiber_config.reference_wavelength_nm,
        )

        # 噪声系数 (线性)
        self.nf_lin_channels = _calc_nf_lin_jit(self.lambdas_m)

        # beta2 和 beta3 (s^2/m, s^3/m)
        # 简化：在每个跨段的中心波长处计算 beta2 和 beta3，或使用参考波长
        # 这里使用参考波长处的 D 和 S 来计算
        ref_lambda_m = C_LIGHT / self.band_center_hz
        self.beta2_s2_m = _calc_beta2_from_d(self.ref_fiber_config.dispersion_parameter_d, ref_lambda_m)
        self.beta3_s3_m = _calc_beta3_from_s_d(
            self.ref_fiber_config.dispersion_slope_s,
            self.ref_fiber_config.dispersion_parameter_d,
            ref_lambda_m,
        )

        # 相对频率 (Hz)，用于 NLI/SRS 计算
        self.f_rel_hz = self.f_abs_hz - self.band_center_hz
        if self.connections is not None:
            # 变宽连接下单条 Rs/2 不再是带边，必须按实际占用带计算。
            low_edges = np.array(
                [conn.centre_hz - conn.n_slots * conn.grid.slot_width_hz / 2 for conn in self.connections],
                dtype=np.float64,
            )
            high_edges = np.array(
                [conn.centre_hz + conn.n_slots * conn.grid.slot_width_hz / 2 for conn in self.connections],
                dtype=np.float64,
            )
            self.f_m_hz = float(low_edges.min() - self.band_center_hz)
            self.f_M_hz = float(high_edges.max() - self.band_center_hz)
        else:
            self.f_m_hz = self.f_rel_hz[0] - self.symbol_rate_hz / 2
            self.f_M_hz = self.f_rel_hz[-1] + self.symbol_rate_hz / 2

    def _calc_span_noise_and_power(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """计算单跨段的功率演化、NLI 噪声和 ASE 噪声。

        Args:
            power_in_w: 各信道入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            edfa_config: 跨段后的 EDFA 配置。
            roadm_config: (可选) 跨段后的 ROADM 配置。
            n_effective_spans: 有效跨段数，用于 NLI 累积效应，对于单跨段通常为 1。

        Returns
        -------
            Tuple 包含:
            - power_out_w: 经历损耗、SRS 和放大后的出纤功率。
            - sigma2_spm_w: 产生的 SPM 噪声方差 (W)。
            - sigma2_xpm_w: 产生的 XPM 噪声方差 (W)。
            - sigma2_ase_w: 产生的 ASE 噪声方差 (W)。
        """
        # 1. 处理放大器噪声系数 (NF)
        # 如果 edfa_config 提供了全频段的 NF 数组则使用之，否则使用预计算的默认值
        if edfa_config is not None:
            if edfa_config.noise_figure_db is not None:
                noise_figure_db = np.asarray(edfa_config.noise_figure_db, dtype=float)
                if noise_figure_db.ndim == 0:
                    current_nf_lin = np.full_like(self.nf_lin_channels, 10 ** (float(noise_figure_db) / 10))
                else:
                    if noise_figure_db.shape != (self.n_nodes,):
                        raise ValueError("EDFA noise-figure array must contain one value per channel.")
                    current_nf_lin = 10 ** (noise_figure_db / 10)
        else:
            current_nf_lin = np.ones_like(self.nf_lin_channels)

        # 2. 调用 JIT 内核 (热路径)；cached 档按输入记忆化单跨段结果。
        cached = None
        if self.ode_accel == "cached":
            cache_key = (
                power_in_w.tobytes(),
                float(span_length_m),
                int(n_effective_spans),
                current_nf_lin.tobytes(),
            )
            cached = self._span_cache.get(cache_key)

        if cached is not None:
            # 必须返回副本：调用方会原地改动出纤功率数组，交出去会污染缓存。
            p_out_fiber, s2_spm, s2_xpm = (array.copy() for array in cached)
        else:
            p_out_fiber, s2_spm, s2_xpm, _ = _calc_span_noise_and_power_jit(
                power_in_w=power_in_w,
                span_length_m=span_length_m,
                f_rel_hz=self.f_rel_hz,
                f_abs_hz=self.f_abs_hz,
                lambdas_m=self.lambdas_m,
                alpha_power_npm=self.alpha_power_npm_channels,
                a_eff_m2=self.a_eff_m2_channels,
                beta2_s2_m=self.beta2_s2_m,
                beta3_s3_m=self.beta3_s3_m,
                n2_m2_w=self.n2_m2_w,
                nf_lin_channels=current_nf_lin,
                symbol_rate_hz=self.symbol_rate_hz,
                b_hz_channels=self.channel_bandwidth_hz_channels,
                n_effective_spans=n_effective_spans,
                cr_w_m_hz=self.cr_w_m_hz,
                k_bar=self.k_bar,
                delta_f_co_hz=self.delta_f_co_hz,
            )
            if self.ode_accel == "cached":
                self._span_cache[cache_key] = (p_out_fiber.copy(), s2_spm.copy(), s2_xpm.copy())

        # 3. 显式应用 EDFA；edfa_config 为 None 时不产生 ASE。
        if edfa_config is None:
            p_out = p_out_fiber
            s2_ase = np.zeros_like(power_in_w)
        else:
            if edfa_config.target_gain_db is None:
                gain = np.divide(power_in_w, p_out_fiber, out=np.ones_like(power_in_w), where=p_out_fiber > 0)
            else:
                gain = np.full_like(power_in_w, 10 ** (edfa_config.target_gain_db / 10))
            if edfa_config.gain_ripple_db is not None:
                ripple = np.asarray(edfa_config.gain_ripple_db, dtype=float)
                if ripple.shape != (self.n_nodes,):
                    raise ValueError("EDFA gain-ripple array must contain one value per channel.")
                gain *= 10 ** (ripple / 10)
            p_out = p_out_fiber * gain
            # ASE 的 B_i 取逐信道数组；遗留模式下它恒等于标量 symbol_rate_hz。
            s2_ase = (
                current_nf_lin
                * 6.62607015e-34
                * self.f_abs_hz
                * self.channel_bandwidth_hz_channels
                * np.maximum(gain - 1, 0)
            )
            s2_ase[power_in_w <= 0] = 0

        # 4. ROADM 损耗同等衰减信号和已生成噪声。
        if roadm_config:
            loss_lin = _db_to_lin(roadm_config.path_loss_db())
            p_out /= loss_lin
            s2_spm /= loss_lin
            s2_xpm /= loss_lin
            s2_ase /= loss_lin

        return p_out, s2_spm, s2_xpm, s2_ase

    def for_fiber(self, fiber_config: FiberSpanConfig) -> "MultiBandISRSGN":
        """返回匹配链路光纤参数的缓存评估器。

        逐连接模式下必须把 ``connections`` 一并传给子评估器，否则它会退回遗留
        分支并读 ``FlexibleGrid.wavelengths``（该属性不存在）。
        """
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = MultiBandISRSGN(
                self.grid,
                fiber_config,
                self.symbol_rate_hz,
                connections=self.connections,
                ode_accel=self.ode_accel,
                slots_per_channel=self.slots_per_channel,
                channel_bandwidth_source=self.channel_bandwidth_source,
                power_width_mode=self.power_width_mode,
            )
        return self._fiber_evaluator_cache[key]

    def evaluate_path(self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat) -> PathQoTResult:
        """按拓扑顺序评估光纤、跨段 EDFA、ROADM 和 booster."""
        power = np.asarray(power_in_w, dtype=np.float64).copy()
        if power.shape != (self.n_nodes,) or np.any(power < 0):
            raise ValueError("power_in_w must be a non-negative vector with one value per channel.")
        # 功率归一化口径在此显式生效；fixed_power 为逐位 no-op。
        power = power * self._power_width_scale()
        total_spm = np.zeros_like(power)
        total_xpm = np.zeros_like(power)
        total_ase = np.zeros_like(power)
        span_count = 0
        roadm_configs = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)

        for hop_index, (fiber, edfa, roadm) in enumerate(
            zip(optical_path.fiber_configs, optical_path.edfa_configs, roadm_configs, strict=True)
        ):
            evaluator = self.for_fiber(fiber)
            remaining_km = fiber.length_km
            while remaining_km > 1e-12:
                span_km = min(remaining_km, fiber.max_span_length_km)
                previous_power = power
                power, spm, xpm, ase = evaluator._calc_span_noise_and_power(
                    power, span_km * 1e3, edfa, n_effective_spans=1
                )
                transfer = np.divide(power, previous_power, out=np.zeros_like(power), where=previous_power > 0)
                total_spm *= transfer
                total_xpm *= transfer
                total_ase *= transfer
                total_spm += spm
                total_xpm += xpm
                total_ase += ase
                span_count += 1
                remaining_km -= span_km

            if roadm is None:
                continue
            blocked = [index for index in range(self.n_nodes) if self.roadm_blocks_node(roadm, index)]
            operation = "drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"
            loss_db = roadm.path_loss_db(operation)
            loss = 10 ** (loss_db / 10)
            power /= loss
            total_spm /= loss
            total_xpm /= loss
            total_ase /= loss
            if roadm.equalize_output_power:
                gain = loss
                power *= gain
                total_spm *= gain
                total_xpm *= gain
                total_ase *= gain
                nf = 10 ** (roadm.booster_noise_figure_db / 10)
                booster_ase = nf * 6.62607015e-34 * self.f_abs_hz * self.channel_bandwidth_hz_channels * (gain - 1)
                booster_ase[power <= 0] = 0
                total_ase += booster_ase
            penalty = 10 ** (roadm.filtering_penalty_db / 10)
            total_spm *= penalty
            total_xpm *= penalty
            total_ase *= penalty
            if blocked:
                power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0

        total_noise = total_spm + total_xpm + total_ase
        snr = np.full_like(power, -np.inf)
        active = (power > 0) & (total_noise > 0)
        snr[active] = 10 * np.log10(power[active] / total_noise[active])
        snr[(power > 0) & (total_noise == 0)] = np.inf
        return PathQoTResult(
            power, total_spm, total_xpm, total_ase, snr, span_count, sum(roadm is not None for roadm in roadm_configs)
        )
