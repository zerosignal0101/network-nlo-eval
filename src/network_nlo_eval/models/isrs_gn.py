"""多波段广义高斯噪声 (Generalized Gaussian Noise, GGN) 模型评估器。

ISRS-GN (Inter-channel Stimulated Raman Scattering - Generalized Noise)
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
        """验证所有逐跳配置长度一致。"""
        count = len(self.fiber_configs)
        if count == 0 or len(self.edfa_configs) != count:
            raise ValueError("Optical path requires matching non-empty fiber and EDFA config lists.")
        if self.roadm_configs is not None and len(self.roadm_configs) != count:
            raise ValueError("ROADM config count must match the number of route links.")
        if self.link_keys and len(self.link_keys) != count:
            raise ValueError("Link key count must match the number of route links.")


@dataclass(frozen=True)
class PathQoTResult:
    """统计物理层沿路径计算的逐信道结果。"""

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
        phi_kurtosis: float = 0.0,  # 0.0为高斯(默认), -0.6190为64QAM
        connections: Sequence[ConnectionSpec] | None = None,
        slots_per_channel: int | None = None,
    ):
        """初始化 ISRS-GN 评估器。

        Args:
            grid: 频谱网格配置（遗留 `SpectrumGrid` 或 `FlexibleGrid`）。
            ref_fiber_config: 参考光纤配置，用于计算各种波长相关参数。
            symbol_rate_hz: 系统符号速率 (Hz)。仅在 `connections is None`
                （遗留模式）时作为标量使用；给定 `connections` 时每连接的
                `ConnectionSpec.symbol_rate_hz` 优先。
            phi_kurtosis: 峰度修正，0.0 为高斯，-0.6190 为 64QAM。
            connections: (可选) 灵活栅格连接列表。给定后进入**逐连接模式**：
                全部逐连接数组按连接中心频率计算，`symbol_rate_hz` 变为逐连接
                数组（设计 §2.5）。为 None 时保持遗留行为，数值逐位不变。
        """
        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.phi_kurtosis = phi_kurtosis
        self.cr_w_m_hz = _SRS_CR  # 拉曼增益系数
        self.delta_f_co_hz = 15e12  # 石英光纤受激拉曼散射（SRS）增益谱的截断频率
        self.n2_m2_w = ref_fiber_config.nonlinear_index_n2
        self.connections = list(connections) if connections is not None else None
        if slots_per_channel is not None and slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")
        self.slots_per_channel = slots_per_channel

        self._precompute_channel_dependent_properties()
        self._fiber_evaluator_cache: dict[str, MultiBandISRSGN] = {}

    @property
    def n_nodes(self) -> int:
        """ODE 节点数 = 活跃连接数（设计 §2.4：一条连接 = 一个节点）。"""
        return int(self.f_abs_hz.shape[0])

    @property
    def band_center_hz(self) -> float:
        """`f_rel_hz` 的参考原点。遗留网格用中心频率，灵活栅格用频段中心。"""
        if isinstance(self.grid, FlexibleGrid):
            return self.grid.band_center_hz
        return self.grid.center_frequency_hz

    def _precompute_channel_dependent_properties(self) -> None:
        """预计算逐连接通用的波长依赖物理属性，如波长、有效模面积等。

        灵活栅格下这些量按**连接中心波长**计算（设计 §2.5 机械改），遗留模式下
        按栅格频率计算——两者在固定栅格特例下逐位相同。
        """
        if self.connections is not None:
            # 每连接一个节点，节点频率 = 连接中心频率（§3.4）。
            self.f_abs_hz = np.array([conn.centre_hz for conn in self.connections], dtype=np.float64)
            self.symbol_rate_hz_channels = np.array(
                [conn.symbol_rate_hz for conn in self.connections], dtype=np.float64
            )
        else:
            self.f_abs_hz = self.grid.frequencies  # (Hz)
            self.symbol_rate_hz_channels = np.full(self.f_abs_hz.shape[0], float(self.symbol_rate_hz))

        self.lambdas_m = C_LIGHT / self.f_abs_hz  # (m)

        # 衰减系数 (Np/m)
        self.alpha_power_npm_channels = _calc_fiber_attenuation_npm(
            self.lambdas_m,
            self.ref_fiber_config.reference_wavelength_nm,
            self.ref_fiber_config.attenuation_alpha2,
            self.ref_fiber_config.attenuation_alpha1,
            self.ref_fiber_config.attenuation_alpha0,
        )

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
        # 频带边缘 (Hz)：变宽连接下单条 Rs/2 不再是带边，必须按实际占用带计算
        # （设计 §2.5）。遗留模式下占用带 = channel_spacing_hz，与旧的 Rs/2 口径
        # 在 Rs == channel_spacing_hz 时一致。
        if self.connections is not None:
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
        if edfa_config is not None and edfa_config.noise_figure_db is not None:
            if isinstance(edfa_config.noise_figure_db, float):
                current_nf_lin = np.full_like(self.nf_lin_channels, 10 ** (edfa_config.noise_figure_db / 10))
            else:
                # noise_figure_db 是数组，需要转换为线性值
                current_nf_lin = 10 ** (np.asarray(edfa_config.noise_figure_db, dtype=float) / 10)
        else:
            current_nf_lin = self.nf_lin_channels

        # 1.5 形状契约校验（设计 §3.4）。逐连接数组必须长度恰为 N 且同序对齐；
        # 消息含期望形状 (N,) 与实际形状。njit 内核只有静态消息，故完整诊断在此。
        n_conn = self.n_nodes
        if np.shape(power_in_w) != (n_conn,):
            raise ValueError(f"power_in_w must have shape ({n_conn},), got {np.shape(power_in_w)}.")
        current_nf_lin = np.asarray(current_nf_lin, dtype=np.float64)
        if current_nf_lin.shape != (n_conn,):
            raise ValueError(f"nf_lin_channels must have shape ({n_conn},), got {current_nf_lin.shape}.")

        # 2. 调用 JIT 内核 (热路径)
        p_out_fiber, s2_spm, s2_xpm, s2_ase = _calc_span_noise_and_power_jit(
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
            symbol_rate_hz=self.symbol_rate_hz_channels,
            n_effective_spans=n_effective_spans,
            cr_w_m_hz=self.cr_w_m_hz,
            delta_f_co_hz=self.delta_f_co_hz,
            phi_kurtosis=self.phi_kurtosis,
        )

        # 3. 处理 ROADM 插入损耗 (可选，在 Python 层处理非线性外围逻辑)
        if roadm_config:
            # 若有 ROADM，出纤功率需要扣除插入损耗
            # 损耗通常不增加线性噪声方差，但会降低后续跨段的入纤功率
            loss_lin = _db_to_lin(roadm_config.insertion_loss_db + roadm_config.filtering_penalty_db)
            p_out_fiber = p_out_fiber / loss_lin

        return p_out_fiber, s2_spm, s2_xpm, s2_ase

    def for_fiber(self, fiber_config: FiberSpanConfig) -> "MultiBandISRSGN":
        """返回匹配链路光纤参数的缓存评估器."""
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = MultiBandISRSGN(
                self.grid,
                fiber_config,
                self.symbol_rate_hz,
                self.phi_kurtosis,
                self.connections,
                self.slots_per_channel,
            )
        return self._fiber_evaluator_cache[key]

    # Backward-compatible internal alias for pre-integration callers.
    _for_fiber = for_fiber

    def _resolve_blocked_node_indices(self, roadm: "ROADMConfig") -> list[int]:
        """把 ROADM 的阻断声明解析为**节点（连接）索引**列表。

        遗留模式（`connections is None`）：`blocked_channels` 直接就是节点索引，
        越界者按既有行为丢弃（保持遗留数值不变）。

        逐连接模式：`blocked_slots` 权威——连接只要与任一被阻断槽**重叠**即被
        阻断（切片语义，设计 §2.5 末段 / §2.6）。宽连接因此会因细粒度阻断而
        整体不可行，这是**语义改**。`blocked_channels` 仍作为只读兼容输入，按
        `self.slots_per_channel` 展开后合并。

        Raises:
            ValueError: 逐连接模式下只给了 `blocked_channels` 而未提供
                `slots_per_channel`（不做取整猜测，避免静默误判）。
        """
        n_nodes = self.n_nodes
        blocked: set[int] = set()

        if self.connections is not None:
            blocked_slots: set[int] = set(roadm.blocked_slots)
            if roadm.blocked_channels:
                if self.slots_per_channel is None:
                    raise ValueError(
                        "blocked_channels is a legacy per-channel input; expanding it into slots "
                        "requires slots_per_channel (channel_spacing_hz / slot_width_hz). "
                        "Construct the evaluator with slots_per_channel=..., or use blocked_slots."
                    )
                for channel_index in roadm.blocked_channels:
                    if channel_index < 0:
                        continue
                    blocked_slots.update(
                        range(channel_index * self.slots_per_channel, (channel_index + 1) * self.slots_per_channel)
                    )
            for index, conn in enumerate(self.connections):
                if connection_overlaps_slots(conn.first_slot, conn.n_slots, blocked_slots):
                    blocked.add(index)
            return sorted(blocked)

        blocked.update(index for index in roadm.blocked_channels if index < n_nodes)
        blocked.update(index for index in roadm.blocked_slots if index < n_nodes)
        return sorted(blocked)

    def evaluate_path(self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat) -> PathQoTResult:
        """按拓扑顺序评估光纤、跨段 EDFA 和 ROADM。

        本方法只做**编排**：每个跨段都调用与 `origin/main` 完全相同的
        `_calc_span_noise_and_power`（内部走 main 的 RK4 ODE 内核），并在本层
        累积噪声。ROADM 衰减在此处按 main 的约定统一施加于信号与已累积噪声，
        与 `rwa/qot_checker` 的链路切分规则一致。
        """
        power = np.asarray(power_in_w, dtype=np.float64).copy()
        if power.shape != (self.n_nodes,) or np.any(power < 0):
            raise ValueError("power_in_w must be a non-negative vector with one value per connection.")
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
            # ROADM 损耗同时作用于信号与此前累积的噪声（纯线性，不引入新噪声）。
            loss = 10 ** ((roadm.insertion_loss_db + roadm.filtering_penalty_db) / 10)
            power /= loss
            total_spm /= loss
            total_xpm /= loss
            total_ase /= loss
            blocked = self._resolve_blocked_node_indices(roadm)
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
