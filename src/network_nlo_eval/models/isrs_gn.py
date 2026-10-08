"""多波段广义高斯噪声 (Generalized Gaussian Noise, GGN) 模型评估器。

ISRS-GN (Inter-channel Stimulated Raman Scattering - Generalized Noise)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.batch_kernels import (
    BatchSpanConstants,
    batched_span_noise_and_power,
    build_batch_constants,
)
from network_nlo_eval.models.numba_kernels import (
    _calc_aeff_dynamic,
    _calc_beta2_from_d,
    _calc_beta3_from_s_d,
    _calc_fiber_attenuation_npm,
    _calc_nf_lin_jit,
    _calc_span_noise_and_power_jit,
    _db_to_lin,
    g_cache_stats,
    nli_cache_stats,
    nli_tensors,
    raman_g_matrix,
    raman_g_matrix_tabular,
    raman_g_pair_matrices,
    raman_g_pair_matrices_tabular,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.slots import connection_overlaps_slots

_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)]

#: FG-4a ODE 加速档位。
#:
#: ``"off"``
#:     每次调用都在核内重建 G —— FG-2 冻结的参考路径。
#: ``"cached_g"``
#:     G 按 (频率栅格, Cr, 谱形常数) 预计算并缓存，内部仍用未改动的
#:     ``_calc_dp_dz``。输出与基线**逐位相等**。
#: ``"matvec"``
#:     G 缓存**并且**把逐对分支换成两次 matvec。数学式不变，但浮点求和次序
#:     改变，实测最大相对偏差 ~1e-14（F4a-4 预算是 1e-12）。
#:
#: **默认是 ``"matvec"``**（FG-4b 编排方裁定）——生产路径要快。
#:
#: FG-4a 最初把默认档定为 ``"cached_g"``，理由是 F1-1 / F1-2 / F2-3 三个既有测试
#: 用 ``np.array_equal`` 把公共 API 输出钉死在 origin/main 的金标准上，而 matvec
#: 的末位重排会让它们失败。裁定**不通过放宽那三条断言来解除**这条锁，而是把三条
#: 测试**显式构造为 ``ode_accel="off"``**：
#:
#: * 这三个测试的命题是"reconcile 没有改变 origin/main 的物理"——这是关于**移植
#:   保真度**的陈述，与"生产默认档选哪个"无关。用 off 档跑，命题原样成立、强度
#:   原样保留（仍是逐位相等，期望值一字未改）。
#: * 生产默认档（matvec）由 FG-4a 的金标准以 <=1e-12 相对偏差把关
#:   （``tests/test_fg4a_numeric_identity.py``，预算常量不得放宽）。
#:
#: 于是证据链被拆成两段，每段各自成立、各自有测试，且都比"把三条断言改成容差"
#: 更强::
#:
#:     origin/main --(off 档, 逐位)--> FG-2 树 --(默认 matvec, <=1e-12)--> 生产
#:
#: 这是一次**有意记录在案的契约放宽**（见 docs/fg4b_nli_vectorization.md §1），
#: 不是顺手绕过：整个加速程序（matvec、NLI 张量化、GPU/批量）都会重排浮点求和，
#: 若默认档被逐位相等钉死，后面每一刀都会被同一条断言挡住。
ODE_ACCEL_MODES = ("off", "cached_g", "matvec")
ODE_ACCEL_DEFAULT = "matvec"

#: FG-U1 拉曼增益谱档位。
#:
#: ``"legacy"``
#:     合成折线（13.2 THz 线性爬升、15.6 THz 跌到零、之后恒零）。这是 FG-4a 的
#:     数值金标准与下游全部既有期望值所依赖的那条曲线，**默认档，逐位不变**。
#: ``"tabular"``
#:     有出处的宽波段熔融石英谱（Lin & Agrawal 2006 的响应函数经傅里叶变换得到的
#:     闭式增益谱，见 `physics.raman_spectrum`）。它会**改变结果**——这是物理改进，
#:     不是数值重排，因此走**自己的验收闸门**，不与速度线的 <=1e-12 同结果预算混用。
#:
#: 两档都用 ``G = cr_w_m_hz * 13.2e12 * S(|df|)`` 的同一语义：``cr_w_m_hz`` 管幅度、
#: ``S`` 管谱形，且在 13.2 THz 处**逐位相等**（legacy 的 ``S`` 就是那条折线）。
RAMAN_SPECTRUM_MODES = ("legacy", "tabular")
RAMAN_SPECTRUM_DEFAULT = "legacy"

#: FG-4e 批量常量包的命中/未命中计数——证明 (N,N) 常量确实跨候选复用。
_BATCH_CACHE_STATS = {"hits": 0, "misses": 0}


def batch_cache_stats() -> dict[str, int]:
    """返回批量常量包缓存的命中/未命中计数。

    与 :func:`~network_nlo_eval.models.numba_kernels.g_cache_stats` 同构：证据不是
    "我说它复用了"，而是命中计数确实随批量调用增长。
    """
    return dict(_BATCH_CACHE_STATS)


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
        ode_accel: str = ODE_ACCEL_DEFAULT,
        raman_spectrum: str = RAMAN_SPECTRUM_DEFAULT,
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
            ode_accel: FG-4a ODE 加速档位，见 :data:`ODE_ACCEL_MODES`。
                ``"off"`` 保持 FG-2 冻结路径；``"cached_g"`` 与基线逐位相等；
                ``"matvec"``（默认）用去分支两次 matvec，相对偏差 <= 1e-12。
            raman_spectrum: FG-U1 拉曼增益谱档位，见 :data:`RAMAN_SPECTRUM_MODES`。
                ``"legacy"``（默认）= 合成折线，逐位不变；``"tabular"`` = 有出处的
                宽波段熔融石英谱，**会改变结果**，需走自己的验收闸门。
                ``"tabular"`` 与 ``ode_accel="off"`` 不兼容（见下）。
        Raises:
            ValueError: 档位名非法，或 ``tabular`` 与 ``ode_accel="off"`` 的组合。
        """
        if ode_accel not in ODE_ACCEL_MODES:
            raise ValueError(f"ode_accel must be one of {ODE_ACCEL_MODES}, got {ode_accel!r}.")
        if raman_spectrum not in RAMAN_SPECTRUM_MODES:
            raise ValueError(f"raman_spectrum must be one of {RAMAN_SPECTRUM_MODES}, got {raman_spectrum!r}.")
        if raman_spectrum == "tabular" and ode_accel == "off":
            # `off` 是 FG-2 冻结的参考路径，它在核内逐对重建 legacy 折线，
            # 结构上无法换谱。这里显式拒绝，而不是悄悄退回 legacy。
            raise ValueError(
                'raman_spectrum="tabular" requires ode_accel="cached_g" or "matvec"; '
                'ode_accel="off" is the frozen legacy-only reference path.'
            )
        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.phi_kurtosis = phi_kurtosis
        self.cr_w_m_hz = _SRS_CR  # 拉曼增益系数
        self.delta_f_co_hz = 15e12  # 石英光纤受激拉曼散射（SRS）增益谱的截断频率
        self.ode_accel = ode_accel
        self.raman_spectrum = raman_spectrum
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

        # FG-4a：拉曼矩阵只依赖频率栅格与 Cr，与功率无关，因此在这里（每个
        # evaluator 一次）预计算一次，跨所有跨段复用。缓存键是频率栅格的**值**，
        # 所以同一栅格的不同 evaluator 会命中同一条缓存条目。
        # `g_cache_stats()` 暴露命中/未命中计数以证明复用确实发生。
        self._g_full: NDArrayFloat | None = None
        self._g_gain: NDArrayFloat | None = None
        self._g_loss: NDArrayFloat | None = None
        if self.ode_accel == "cached_g":
            if self.raman_spectrum == "tabular":
                self._g_full = raman_g_matrix_tabular(self.f_abs_hz, self.cr_w_m_hz)
            else:
                self._g_full = raman_g_matrix(self.f_abs_hz, self.cr_w_m_hz)
        elif self.ode_accel == "matvec":
            if self.raman_spectrum == "tabular":
                self._g_gain, self._g_loss = raman_g_pair_matrices_tabular(self.f_abs_hz, self.cr_w_m_hz)
            else:
                self._g_gain, self._g_loss = raman_g_pair_matrices(self.f_abs_hz, self.cr_w_m_hz)

        # FG-4b：NLI 张量是 (N,N) 量级、且含全部超越函数，只在构建期算一次。
        # 与功率无关（唯一的数据相关量 t_tilde 留在热路径），因此同样每个
        # evaluator 预计算一次、跨所有跨段复用。`n_span` 进缓存键，故这里延迟到
        # 首次调用再建，见 _nli_tensors_for_span。
        self._nli_cache_key: tuple[Any, ...] | None = None
        self._nli_cache_value: tuple[NDArrayFloat, ...] | None = None

        # FG-4e：批量入口的构建期常量包。与 _nli_cache 同构（同样按
        # (跨段数, 跨段长度) 记忆化），但它把拉曼矩阵**转置**并把
        # gamma_sq*w_k 折成转置且对角置零的三张 (N,N)，因此**必须**
        # 在实例内单独缓存，不能与 _nli_cache 共用同一份。
        self._batch_constants_key: tuple[Any, ...] | None = None
        self._batch_constants_value: Any | None = None

    def _batch_constants_for_span(
        self, n_effective_spans: int, span_length_m: float
    ) -> "BatchSpanConstants":
        """取该 (跨段数, 跨段长度) 对应的批量常量包（实例内记忆化）。

        缓存键与 :meth:`_nli_tensors_for_span` 同构：``span_length_m`` 必须进键
        （``lin_decay`` 依赖 ``alpha*L``），``n_effective_spans`` 也进键（决定
        SPM 跨段指数因子，且已经折进 NLI 张量）。

        与串行路径的差别只有一处、且是刻意的：串行每个 evaluator 都持有
        ``(N,N)`` 矩阵复用，但批量路径把它们**跨候选**复用，所以一个 batch
        调用只付一次构建成本。
        """
        key = (int(n_effective_spans), float(span_length_m))
        if self._batch_constants_key == key and self._batch_constants_value is not None:
            # 直接改模块级 dict：``batch_cache_stats()`` 返回的是**副本**，
            # 对返回值自增会写进临时对象，计数永远停在 0（证据就废了）。
            _BATCH_CACHE_STATS["hits"] += 1
            return self._batch_constants_value
        if self.ode_accel != "matvec":
            raise ValueError(
                "calc_span_noise_and_power_batch requires ode_accel='matvec': the batched "
                "kernel is the batch form of the default tier's maths (matvec ODE + tensorised "
                f"NLI). Got ode_accel={self.ode_accel!r}."
            )
        g_gain, g_loss = self._g_gain, self._g_loss
        if g_gain is None or g_loss is None:
            raise ValueError("batched kernel requires the cached Raman pair matrices.")
        tensors = self._nli_tensors_for_span(n_effective_spans, span_length_m)
        spm0, spm1, spm2, gamma_ii_sq, gamma_sq, w0, w1, w2 = tensors
        value = build_batch_constants(
            g_gain=g_gain,
            g_loss=g_loss,
            alpha=self.alpha_power_npm_channels,
            spm0=spm0,
            spm1=spm1,
            spm2=spm2,
            gamma_ii_sq=gamma_ii_sq,
            gamma_sq=gamma_sq,
            w0=w0,
            w1=w1,
            w2=w2,
            rs=self.symbol_rate_hz_channels,
            f_abs=self.f_abs_hz,
            nf_lin=self.nf_lin_channels,
            span_length_m=span_length_m,
        )
        _BATCH_CACHE_STATS["misses"] += 1
        self._batch_constants_key = key
        self._batch_constants_value = value
        return value

    def _nli_kernel_kwargs(self, n_effective_spans: int, span_length_m: float) -> dict[str, Any]:
        """给内核的八个 ``nli_*`` 参数。

        档位分工是刻意保持的：**只有 ``matvec`` 使用张量化 NLI**。

        * ``off`` —— 冻结点，八个参数全 None，逐字节走 FG-2 冻结的逐对基线。
          这正是 F4b-2「off 档仍与 origin/main 逐位相等」得以成立的前提。
        * ``cached_g`` —— FG-4a 的**逐位相等**档：只做 G 缓存，不重排任何求和。
          因此它**必须**继续走冻结的逐对 NLI；否则 F4a-3（cached_g 与金标准逐位
          相等）会被本刀悄悄破坏。一个档位只引入一类浮点重排，是这张表能读懂、
          能逐档取证的原因。
        * ``matvec`` —— 生产默认：去分支 matvec + 张量化 NLI，合计允许 <=1e-12。
        """
        if self.ode_accel != "matvec":
            return {
                "nli_spm0": None,
                "nli_spm1": None,
                "nli_spm2": None,
                "nli_gamma_ii_sq": None,
                "nli_gamma_sq": None,
                "nli_w0": None,
                "nli_w1": None,
                "nli_w2": None,
            }
        spm0, spm1, spm2, gamma_ii_sq, gamma_sq, w0, w1, w2 = self._nli_tensors_for_span(
            n_effective_spans, span_length_m
        )
        return {
            "nli_spm0": spm0,
            "nli_spm1": spm1,
            "nli_spm2": spm2,
            "nli_gamma_ii_sq": gamma_ii_sq,
            "nli_gamma_sq": gamma_sq,
            "nli_w0": w0,
            "nli_w1": w1,
            "nli_w2": w2,
        }

    def _nli_tensors_for_span(self, n_effective_spans: int, span_length_m: float) -> tuple[NDArrayFloat, ...]:
        """取该 (跨段数, 跨段长度) 对应的 NLI 构建期张量（实例内记忆化）。

        ``span_length_m`` 必须进键：``a_tilde``/``kappa`` 依赖 ``alpha * L``，
        折进 W0/W1/W2 与 spm* 后就不可分离了。跨段数在 ``evaluate_path`` 里逐段
        推进，因此同一组合重复出现时直接复用，换组合才重建。
        """
        key = (int(n_effective_spans), float(span_length_m))
        if self._nli_cache_key == key and self._nli_cache_value is not None:
            nli_cache_stats()["hits"] += 1
            return self._nli_cache_value
        tensors = nli_tensors(
            self.n_nodes,
            int(n_effective_spans),
            self.symbol_rate_hz_channels,
            self.f_rel_hz,
            self.lambdas_m,
            self.a_eff_m2_channels,
            self.alpha_power_npm_channels,
            float(span_length_m),
            self.beta2_s2_m,
            self.beta3_s3_m,
            self.n2_m2_w,
            self.phi_kurtosis,
        )
        nli_cache_stats()["misses"] += 1
        self._nli_cache_key = key
        self._nli_cache_value = tensors
        return tensors

    @property
    def g_cache_hits(self) -> int:
        """FG-4a 证据：G 缓存命中次数（跨调用复用的直接计数）。"""
        return g_cache_stats()["hits"]

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
            # FG-4a：全部为 None 时内核走 FG-2 冻结的参考路径。
            g_full=self._g_full,
            g_gain=self._g_gain,
            g_loss=self._g_loss,
            # FG-4b：默认档传构建期张量；``ode_accel="off"`` 传全 None，
            # 逐字节走冻结的逐对 NLI 基线。
            **self._nli_kernel_kwargs(n_effective_spans, span_length_m),
        )

        # 3. 处理 ROADM 插入损耗 (可选，在 Python 层处理非线性外围逻辑)
        if roadm_config:
            # 若有 ROADM，出纤功率需要扣除插入损耗
            # 损耗通常不增加线性噪声方差，但会降低后续跨段的入纤功率
            loss_lin = _db_to_lin(roadm_config.insertion_loss_db + roadm_config.filtering_penalty_db)
            p_out_fiber = p_out_fiber / loss_lin

        return p_out_fiber, s2_spm, s2_xpm, s2_ase

    def calc_span_noise_and_power_batch(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None = None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """FG-4e：单跨段内核的**批量入口**（新增，默认路径不受影响）。

        与 :meth:`_calc_span_noise_and_power` 算同一件事，区别只在布局：这里一次
        评估 B 个候选，形状是 ``(B, N)`` 进、四个 ``(B, N)`` 出。热路径是
        ``(B,N) @ (N,N)`` 的 GEMM（ODE 每子步两次 + XPM 三次），**无数据相关
        控制流**；(N,N) 的构建期常量不按 batch 复制。

        本方法是**新增入口，不是替换**：``ode_accel`` 的 ``"off"`` / ``"cached_g"``
        / ``"matvec"``（默认）三个档位与 ``evaluate_path`` 的编排逻辑一字未改。
        批量是默认档数学式的批量形态，所以要求 ``ode_accel == "matvec"``。

        逐候选与串行路径的相对偏差预算为 1e-12（沿用 FG-4a/FG-4b 常量）。
        唯一差别是 l 方向求和由成对求和变成 GEMM 累加。

        Args:
            power_in_w: ``(B, N)`` 候选入纤功率 (W)。**零功率是合法的**：惰性连接
                仍占一个槽位，掩码语义与串行完全一致。
            span_length_m: 跨段长度 (m)。
            edfa_config: (可选) 跨段后 EDFA 配置，仅用其全频段 NF 覆盖，语义与
                串行一致。
            roadm_config: (可选) 跨段后 ROADM 配置，插入损耗处理与串行一致。
            n_effective_spans: 有效跨段数。

        Returns:
            Tuple 包含 ``(power_out_w, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w)``，
            四个都是 ``(B, N)``，逐候选等于串行路径对同一候选的结果。

        Raises:
            ValueError: ``ode_accel`` 不是 ``"matvec"``；``power_in_w`` 不是
                ``(B, N)`` 或 N 与本 evaluator 不符；NF 覆盖数组长度不对。
        """
        P = np.asarray(power_in_w, dtype=np.float64)
        n_conn = self.n_nodes
        if P.ndim != 2 or P.shape[1] != n_conn:
            raise ValueError(
                f"power_in_w must have shape (B, {n_conn}), got {np.shape(power_in_w)}."
            )

        # NF 覆盖：与串行同一套解析（全频段 float 或逐通道数组），只在形状
        # (N,) 时才是"逐连接覆盖"，因此不会被 batch 轴误伤。
        if edfa_config is not None and edfa_config.noise_figure_db is not None:
            nf_db = edfa_config.noise_figure_db
            if isinstance(nf_db, float):
                current_nf_lin = np.full(n_conn, 10 ** (nf_db / 10), dtype=np.float64)
            else:
                current_nf_lin = 10 ** (np.asarray(nf_db, dtype=float) / 10)
        else:
            current_nf_lin = None

        constants = self._batch_constants_for_span(n_effective_spans, span_length_m)
        p_out, s2_spm, s2_xpm, s2_ase = batched_span_noise_and_power(
            constants, P, span_length_m, n_effective_spans, current_nf_lin
        )

        if roadm_config is not None:
            loss_lin = _db_to_lin(roadm_config.insertion_loss_db + roadm_config.filtering_penalty_db)
            p_out = p_out / loss_lin

        return p_out, s2_spm, s2_xpm, s2_ase

    def for_fiber(self, fiber_config: FiberSpanConfig) -> "MultiBandISRSGN":
        """返回匹配链路光纤参数的缓存评估器.

        FU-2：这里过去**漏传** ``raman_spectrum``，子评估器因此总是退回默认档
        ``legacy``。后果是 ``evaluate_path``（唯一真正的端到端路径）在结构上
        **永远拿不到 tabular 谱**——FG-U1 的证据脚本因为直接调用
        ``_calc_span_noise_and_power`` 绕开了 ``for_fiber``，所以一直没暴露。
        超宽带评估必须走 ``evaluate_path``，故本刀补上这一处透传。

        ``legacy`` 不受影响：它本就是默认值，传不传都是同一个值，逐位不变。
        """
        key = fiber_config.model_dump_json()
        if key not in self._fiber_evaluator_cache:
            self._fiber_evaluator_cache[key] = MultiBandISRSGN(
                self.grid,
                fiber_config,
                self.symbol_rate_hz,
                self.phi_kurtosis,
                self.connections,
                self.slots_per_channel,
                self.ode_accel,
                self.raman_spectrum,
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
        累积噪声。**内核未被改动**，命题 (A) 的逐位保证不受本刀影响。

        ROADM 装配遵循**仓内权威定义** `ROADMConfig.path_loss_db()`
        （``network/elements.py``），由 ``rwa.qot_checker.fold_roadm_terms``
        折叠出常量：按端口取损耗（末跳 ``drop``、其余 ``express``）、
        ``equalize_output_power`` 为真时改以 booster ASE
        ``NF·h·ν·Rs·(loss−1)`` 补偿、``filtering_penalty_db`` 作为**只乘噪声**
        的独立乘子。这与同仓另外两处施加点（``physics/qot_checker`` 的波形级、
        ``rwa/qot_checker`` 的解析级）同口径。

        FG-14 记录：此处与 ``origin/main`` 的分歧是**被授权的出处**差异，
        不是物理回归。幅度与机制见 ``results/fg14_assembly_divergence.json``，
        裁决见 ``docs/fg14_assembly_provenance.md``。
        """
        power = np.asarray(power_in_w, dtype=np.float64).copy()
        if power.shape != (self.n_nodes,) or np.any(power < 0):
            raise ValueError("power_in_w must be a non-negative vector with one value per connection.")
        total_spm = np.zeros_like(power)
        total_xpm = np.zeros_like(power)
        total_ase = np.zeros_like(power)
        span_count = 0
        roadm_configs = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
        n_hops = len(optical_path.fiber_configs)

        # 惰性导入：``rwa.qot_checker`` 从本模块取 ``MultiBandISRSGN``，
        # 模块级互相导入会成环。折叠逻辑仍然**只有这一处**（FG-11 的性质）。
        from network_nlo_eval.rwa.qot_checker import fold_roadm_terms

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
            # ROADM 装配：权威口径（FG-14）。按端口取损耗，booster 恢复入纤功率时
            # 改注入 ASE 而不衰减信号，滤波惩罚只作用在噪声上。
            # 与 rwa.qot_checker.accumulate_service_snr / physics.qot_checker 同口径。
            terms = fold_roadm_terms(roadm, is_last_hop=hop_index == n_hops - 1)
            if terms.equalize:
                total_ase += (
                    terms.booster_base
                    * np.asarray(self.f_abs_hz, dtype=np.float64)
                    * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64)
                    * (terms.loss - 1.0)
                )
            else:
                power /= terms.loss
                total_spm /= terms.loss
                total_xpm /= terms.loss
                total_ase /= terms.loss
            total_spm *= terms.filtering_penalty
            total_xpm *= terms.filtering_penalty
            total_ase *= terms.filtering_penalty
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
