r"""多波段 ISRS-GN 评估器。

ISRS-GN = Inter-channel Stimulated Raman Scattering - Generalized Noise。

注册表里的两个入口
------------------
* ``ecz``：本模块的 :class:`MultiBandISRSGN` 直接注册。它是 ECZ 论文的单跨闭式
  结果，逐跨只给功率，路由层按 :math:`\sum|a_k|^2` 相干性无关地累加，因此
  **不含跨间相干**，在多跨上比参考低 0.2…0.3 dB。保留它是为了：冻结的 14 个
  黄金向量钉的就是这个类；它也让"补上相干项"的收益可被测量（C 波段 +0.28 →
  +0.02 dB）。**不要拿 ``ecz`` 做多跨的绝对精度结论。**
* ``ecz_coherent``：:class:`~network_nlo_eval.models.ecz.ECZCoherentModel`，同一套
  单跨物理，逐跨给复振幅、由路由层按 :math:`|\sum a_k|^2` 累加。这是该论文口径下
  多跨应使用的入口。
"""

from collections.abc import Sequence

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import (
    ElementOutput,
    ModelInfo,
    MultiSpanOpticalPath,
    PathQoTResult,
    SpanFiberOutput,
    register_model,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_nf_lin_jit,
    _calc_span_noise_and_power_jit,
    _db_to_lin,
)
from network_nlo_eval.models.numba_kernels_p1_accel import (
    P1XpmMatrices,
    build_p1_xpm_pair_matrices,
    calc_span_noise_and_power_p1_matvec,
)
from network_nlo_eval.models.route import accumulate_route, apply_amplifier_and_roadm, resolve_noise_figure
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
#:     每次调用都进 njit 核。冻结参考路径，与逐对求和的闭式实现逐位相同。
#: ``"cached"``
#:     按 (入纤功率, 跨段长度, 跨段数, NF) 记忆化单跨段结果，结果与 ``"off"``
#:     逐位相同。
#: ``"matvec"``
#:     把 XPM 的逐对求和换信道对矩阵 + 矩阵-向量乘（见
#:     :mod:`network_nlo_eval.models.numba_kernels_p1_accel`）。算式不变，只有
#:     ``l`` 方向的浮点求和次序变，相对偏差预算 1e-12。默认档。
#:
#: 想拿与 ``ode_accel="off"`` 逐位相同的读数，显式指定它。
ODE_ACCEL_MODES = ("off", "cached", "matvec")
ODE_ACCEL_DEFAULT = "matvec"

#: ``matvec`` 档的自适应门：活跃信道数达到该阈值才走矩阵-向量乘，否则落回逐对
#: 求和。``off`` 的单跨成本正比于活跃信道数，``matvec`` 近似恒定（满网格矩阵的
#: 工作量），实测交叉点在 34 条附近，取 40 保证至少 1.2 倍收益。交叉点是绝对
#: 条数而非比例，故不按 ``k/N`` 设门。
MATVEC_MIN_ACTIVE_CHANNELS = 40


#: 本模型在注册表里的静态描述。``family`` 标 ``"gn"``：ECZ-SRS 三角近似 + 闭式 GN。
ECZ_MODEL_INFO = ModelInfo(
    name="ecz",
    summary="ECZ-SRS triangular SRS approximation with closed-form GN (SPM/XPM).",
    family="gn",
    reference="Lasagni et al., 'A Generalized Raman Scattering Model for Real-Time SNR "
    "Estimation of Multi-Band Systems', JLT 2023, doi:10.1109/jlt.2023.3250751",
    options=(
        ("ode_accel", "'off' | 'cached' | 'matvec'; single-span acceleration tier."),
        ("channel_bandwidth_source", "'symbol_rate' | 'occupied'; per-node B_i definition."),
        ("power_width_mode", "'fixed_power' | 'fixed_psd'; launch-power normalisation."),
    ),
)

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

        self._span_cache: dict[tuple[bytes, float, int], tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat]] = {}
        self._p1_xpm_matrices: P1XpmMatrices | None = None
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

    def power_width_scale(self) -> NDArrayFloat:
        """``fixed_psd`` 口径下的逐节点功率缩放因子。

        恒定功率谱密度时 ``P_i_eff = P_i × B_i / symbol_rate_hz``；``B_i`` 均匀时
        因子恒为 1.0。``fixed_power`` 口径恒返回全 1。
        """
        if self.power_width_mode == "fixed_power":
            return np.ones(self.n_nodes, dtype=np.float64)
        return self.channel_bandwidth_hz_channels / float(self.symbol_rate_hz)

    # 兼容旧调用名。
    _power_width_scale = power_width_scale

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

    def blocked_nodes(self, roadm: ROADMConfig) -> list[int]:
        """该 ROADM 阻断的全部节点索引。"""
        return [index for index in range(self.n_nodes) if self.roadm_blocks_node(roadm, index)]

    # 兼容旧调用名。
    _blocked_nodes = blocked_nodes

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
        elif isinstance(self.grid, SpectrumGrid):
            self.lambdas_m = self.grid.wavelengths  # (m)
            self.f_abs_hz = self.grid.frequencies  # (Hz)
        else:
            raise TypeError("Legacy mode requires a SpectrumGrid; pass connections= for a FlexibleGrid.")

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

    def _p1_xpm_matrices_for(self) -> P1XpmMatrices:
        """P1 XPM 的 (N,N) 信道对矩阵，实例内记忆化。

        只依赖 evaluator 级量（波长、相对频率、``B_i``、衰减、色散、n2），不随
        跨段、不随候选变，故建一次就够。延迟到首次使用，构造期不做无谓的 O(N^2)。
        """
        if self._p1_xpm_matrices is not None:
            return self._p1_xpm_matrices
        g1, g2, phi_raw, gamma = build_p1_xpm_pair_matrices(
            self.n_nodes,
            self.channel_bandwidth_hz_channels,
            self.f_rel_hz,
            self.lambdas_m,
            self.a_eff_m2_channels,
            self.alpha_power_npm_channels,
            self.beta2_s2_m,
            self.beta3_s3_m,
            self.n2_m2_w,
        )
        self._p1_xpm_matrices = P1XpmMatrices(
            g1=g1,
            g2=g2,
            g1_t=np.ascontiguousarray(g1.T),
            g2_t=np.ascontiguousarray(g2.T),
            phi_raw=phi_raw,
            gamma=gamma,
        )
        return self._p1_xpm_matrices

    def _p1_span_matvec(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int,
        nf_lin_channels: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """P1 闭式核的 ``matvec`` 形态（单候选）。

        出纤功率与 SPM 走冻结核的同一个函数，逐位相同；只有 XPM 的求和次序变了。
        """
        mats = self._p1_xpm_matrices_for()
        p_out, s2_spm, s2_xpm, s2_ase = calc_span_noise_and_power_p1_matvec(
            power_in_w=np.asarray(power_in_w, dtype=np.float64)[None, :],
            span_length_m=float(span_length_m),
            f_rel_hz=self.f_rel_hz,
            f_abs_hz=self.f_abs_hz,
            lambdas_m=self.lambdas_m,
            alpha_power_npm=self.alpha_power_npm_channels,
            a_eff_m2=self.a_eff_m2_channels,
            beta_2_s2_m=self.beta2_s2_m,
            beta_3_s3_m=self.beta3_s3_m,
            n_2_m2_w=self.n2_m2_w,
            nf_lin_channels=nf_lin_channels,
            symbol_rate_hz=self.symbol_rate_hz,
            b_hz_channels=self.channel_bandwidth_hz_channels,
            n_effective_spans=int(n_effective_spans),
            cr_w_m_hz=self.cr_w_m_hz,
            k_bar=self.k_bar,
            delta_f_co_hz=self.delta_f_co_hz,
            mats=mats,
        )
        # (1, N) → (N,)：单候选入口的形状契约与冻结内核一致。
        return p_out[0], s2_spm[0], s2_xpm[0], s2_ase[0]

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
        # 1. 光纤物理核（放大器与 ROADM 在下一步统一施加）。
        fiber_output = self.span_fiber_output(power_in_w, span_length_m, n_effective_spans)

        # 2. 放大器 ASE 与 ROADM 损耗：与所有模型共用同一份实现。
        element = apply_amplifier_and_roadm(
            power_in_w=power_in_w,
            fiber_output=fiber_output,
            edfa_config=edfa_config,
            roadm_config=roadm_config,
            nf_lin=self._resolve_nf(edfa_config),
            f_abs_hz=self.f_abs_hz,
            bandwidth_hz=self.channel_bandwidth_hz_channels,
        )
        return element.power_out_w, element.spm_noise_w, element.xpm_noise_w, element.ase_noise_w

    def _resolve_nf(self, edfa_config: EDFAConfig | None) -> NDArrayFloat:
        """跨段放大器的线性噪声系数，语义见 :func:`resolve_noise_figure`。"""
        return resolve_noise_figure(edfa_config, self.nf_lin_channels, self.n_nodes)

    def span_fiber_output(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        n_effective_spans: int = 1,
    ) -> SpanFiberOutput:
        """单跨光纤物理核：出纤功率与 SPM/XPM，不含放大器。

        ``cached`` 档按 ``(入纤功率, 跨段长度, 跨段数)`` 记忆化；``matvec`` 档在活跃
        信道数达到 :data:`MATVEC_MIN_ACTIVE_CHANNELS` 时走矩阵-向量乘，否则落回逐对
        求和。噪声系数不进入本核，也不进入缓存键：ASE 只在放大器一层产生。

        Args:
            power_in_w: 各信道入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            n_effective_spans: 有效跨段数，用于 NLI 累积修正。

        Returns
        -------
            出纤功率与 SPM/XPM 噪声，均为与输入不共享内存的新数组。
        """
        if self.ode_accel == "cached":
            cache_key = (
                power_in_w.tobytes(),
                float(span_length_m),
                int(n_effective_spans),
            )
            cached = self._span_cache.get(cache_key)
            if cached is not None:
                # 必须返回副本：调用方会原地改动出纤功率数组，交出去会污染缓存。
                p_out_fiber, s2_spm, s2_xpm = (array.copy() for array in cached)
                return SpanFiberOutput(p_out_fiber, s2_spm, s2_xpm)
        else:
            cache_key = None

        if self.ode_accel == "matvec" and np.count_nonzero(power_in_w) >= MATVEC_MIN_ACTIVE_CHANNELS:
            # matvec 只在活跃信道够多时才有收益，少信道落回逐对求和（二者数值差
            # 在 1e-16 量级，门只影响速度）。
            p_out_fiber, s2_spm, s2_xpm, _ = self._p1_span_matvec(
                power_in_w, span_length_m, n_effective_spans, self.nf_lin_channels
            )
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
                nf_lin_channels=self.nf_lin_channels,
                symbol_rate_hz=self.symbol_rate_hz,
                b_hz_channels=self.channel_bandwidth_hz_channels,
                n_effective_spans=n_effective_spans,
                cr_w_m_hz=self.cr_w_m_hz,
                k_bar=self.k_bar,
                delta_f_co_hz=self.delta_f_co_hz,
            )
        if cache_key is not None:
            self._span_cache[cache_key] = (p_out_fiber.copy(), s2_spm.copy(), s2_xpm.copy())
        return SpanFiberOutput(p_out_fiber, s2_spm, s2_xpm)

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: NDArrayFloat,
        span_length_m: float,
    ) -> ElementOutput:
        """路由层单跨步进：光纤 + 放大器（ROADM 由路由层统一处理）。"""
        evaluator = self.for_fiber(fiber_config)
        p_out, spm, xpm, ase = evaluator._calc_span_noise_and_power(
            power_in_w, span_length_m, edfa_config, None, n_effective_spans=1
        )
        return ElementOutput(p_out, spm, xpm, ase)

    def calc_span_noise_and_power_batch(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None = None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """单跨段内核的批量入口：``(B, N)`` 进，四个 ``(B, N)`` 出。

        与 :meth:`_calc_span_noise_and_power` 算同一件事，区别只在布局。热路径是
        ``(B,N) @ (N,N)`` 的 GEMM，无数据相关控制流，(N,N) 常量不按 batch 复制。
        逐候选结果与串行路径的相对偏差预算 1e-12。

        批量形态只在 ``ode_accel="matvec"`` 下存在；档位不匹配时抛
        ``ValueError``，不静默回落。

        Args:
            power_in_w: ``(B, N)`` 候选入纤功率 (W)。零功率合法：惰性连接仍占槽位。
            span_length_m: 跨段长度 (m)。
            edfa_config: (可选) 跨段后 EDFA 配置，语义与串行一致。
            roadm_config: (可选) 跨段后 ROADM 配置，插入损耗处理与串行一致。
            n_effective_spans: 有效跨段数。

        Returns
        -------
            ``(power_out_w, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w)``，均为 ``(B, N)``。

        Raises
        ------
            ValueError: ``ode_accel`` 不是 ``"matvec"``；``power_in_w`` 形状不对；
                NF 或增益纹波数组长度不对。
        """
        if self.ode_accel != "matvec":
            raise ValueError(
                "calc_span_noise_and_power_batch requires ode_accel='matvec': the batched kernel "
                "is the batch form of the P1 closed-form NLI (XPM channel-pair matrices + matvec). "
                f"Got ode_accel={self.ode_accel!r}."
            )

        P = np.asarray(power_in_w, dtype=np.float64)
        n_conn = self.n_nodes
        if P.ndim != 2 or P.shape[1] != n_conn:
            raise ValueError(f"power_in_w must have shape (B, {n_conn}), got {np.shape(power_in_w)}.")

        if edfa_config is not None and edfa_config.noise_figure_db is not None:
            noise_figure_db = np.asarray(edfa_config.noise_figure_db, dtype=float)
            if noise_figure_db.ndim == 0:
                current_nf_lin = np.full(n_conn, 10 ** (float(noise_figure_db) / 10))
            else:
                if noise_figure_db.shape != (n_conn,):
                    raise ValueError("EDFA noise-figure array must contain one value per channel.")
                current_nf_lin = 10 ** (noise_figure_db / 10)
        else:
            current_nf_lin = np.ones_like(self.nf_lin_channels)
        current_nf_lin = np.asarray(current_nf_lin, dtype=np.float64)

        p_out_fiber, s2_spm, s2_xpm, _ = calc_span_noise_and_power_p1_matvec(
            power_in_w=P,
            span_length_m=float(span_length_m),
            f_rel_hz=self.f_rel_hz,
            f_abs_hz=self.f_abs_hz,
            lambdas_m=self.lambdas_m,
            alpha_power_npm=self.alpha_power_npm_channels,
            a_eff_m2=self.a_eff_m2_channels,
            beta_2_s2_m=self.beta2_s2_m,
            beta_3_s3_m=self.beta3_s3_m,
            n_2_m2_w=self.n2_m2_w,
            nf_lin_channels=current_nf_lin,
            symbol_rate_hz=self.symbol_rate_hz,
            b_hz_channels=self.channel_bandwidth_hz_channels,
            n_effective_spans=int(n_effective_spans),
            cr_w_m_hz=self.cr_w_m_hz,
            k_bar=self.k_bar,
            delta_f_co_hz=self.delta_f_co_hz,
            mats=self._p1_xpm_matrices_for(),
        )

        if edfa_config is None:
            p_out = p_out_fiber
            s2_ase = np.zeros_like(P)
        else:
            if edfa_config.target_gain_db is None:
                gain = np.divide(P, p_out_fiber, out=np.ones_like(P), where=p_out_fiber > 0)
            else:
                gain = np.full_like(P, 10 ** (edfa_config.target_gain_db / 10))
            if edfa_config.gain_ripple_db is not None:
                ripple = np.asarray(edfa_config.gain_ripple_db, dtype=float)
                if ripple.shape != (n_conn,):
                    raise ValueError("EDFA gain-ripple array must contain one value per channel.")
                gain *= 10 ** (ripple / 10)
            p_out = p_out_fiber * gain
            s2_ase = (
                current_nf_lin
                * 6.62607015e-34
                * self.f_abs_hz
                * self.channel_bandwidth_hz_channels
                * np.maximum(gain - 1, 0)
            )
            s2_ase[P <= 0] = 0

        if roadm_config:
            loss_lin = _db_to_lin(roadm_config.path_loss_db())
            p_out = p_out / loss_lin
            s2_spm = s2_spm / loss_lin
            s2_xpm = s2_xpm / loss_lin
            s2_ase = s2_ase / loss_lin

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
        """按拓扑顺序评估光纤、跨段 EDFA、ROADM 和 booster。

        路由累积（跨段切分、放大器、ROADM、SNR）由
        :func:`network_nlo_eval.models.route.accumulate_route` 统一实现，本模型只提供
        单跨步进 :meth:`span_step`。换模型不会改变后处理语义。
        """
        return accumulate_route(self, optical_path, power_in_w)


# 注册为可选的 QoT 模型实现；构造函数签名即 ``create_model`` 的透传参数。
register_model(ECZ_MODEL_INFO, MultiBandISRSGN)
