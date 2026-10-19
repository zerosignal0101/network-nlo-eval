"""多波段广义高斯噪声 (Generalized Gaussian Noise, GGN) 模型评估器。

ISRS-GN (Inter-channel Stimulated Raman Scattering - Generalized Noise)
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import NDArrayFloat
# ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：第二套 NLI 物理（见下方「NLI 物理」章节）
# 已从本树**整体移除**。
#
# 本模块现在**只有一套物理**：`numba_kernels` —— P1 (23)/(29)/(30)/(34) 闭式、
# 逐信道 `B_i`，字节 = ``flexgrid-dev @ cb24704`` 的冻结内核（``6611c416…``，
# 逐字节未变）。加速建在**同一套** P1 算式上
# （`numba_kernels_p1_accel.py`），**不引入第二条物理**。
#
# 被移除的那一档是 `wp/fg-integrate` 的**另一条 NLI 模型线**：
# P2 Eq. 21 张量化、符号率口径（本仓历史文档中称「P2 档」）。
# 本产品由 `ecz-dev` 衍生、只需要闭式 ECZ，故该档在本次交付中**整体移除**
# ——是移除，不是搬家，也没有留开关。
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
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.slots import connection_overlaps_slots

_SRS_CR = 0.028 / 1e3 / 1e12  # [1/(W·m·Hz)]

#: ``ode_accel`` 档位（WP-MERGE-FG-FLEX **第 5 轮**：只服务闭式 ECZ）。
#:
#: ⚠️ 第 5 轮移除第二套物理后，本常量随之收窄为 **ECZ 实际支持的三档**：
#:
#: * ``"cached_g"`` —— **随该物理一起移除**。它是**拉曼 G 矩阵预计算缓存**，
#:   只对拉曼 ODE 有意义；闭式 ECZ 没有拉曼 ODE（走解析出纤功率），给它配
#:   ``"cached_g"`` 会静默跑成别的档——那是最坏的一种错。移除前它本来也只对
#:   那套物理可见（见历史按档分派表），故对 ECZ 是**零行为变化**。
#:
#: ``"off"``
#:     每次调用都在核内现算。**冻结参考路径**，与 ``flexgrid-dev @ cb24704``
#:     逐字节相同（判据 V1''' 钉在这一档）。
#: ``"cached"``
#:     跨段记忆化（按 (入纤功率, 跨段长度, 跨段数, NF) 缓存整跨段输出，命中时返回
#:     同一个 ndarray 对象 ⇒ 与 ``"off"`` 逐位相等）。
#: ``"matvec"``
#:     把 P1 XPM 的逐对分支换成**信道对矩阵 + 矩阵-向量乘**。**算式不变**，
#:     但浮点求和次序改变，实测最大相对偏差 **1.269e-15**（预算 1e-12，**不放宽**）。
#:
#: **默认是 ``"matvec"``**（FG-4b 编排方裁定，WP-MERGE-FG-FLEX §3 禁止改动该默认档）。
#: 注意这与合并前 flexgrid-dev 的 ``"off"`` 不同——那是 flexgrid 侧单边默认值，
#: 本次合并按任务书统一到 FG 的生产档；flexgrid 的 ``B_i``/``P_i`` 参数化在**任一档下**
#: 都照常生效，不因默认档切换而失效。
#:
#: 想拿**与 ``cb24704`` 逐位相同**的读数，显式配 ``ode_accel="off"``。
ODE_ACCEL_MODES = ("off", "cached", "matvec")
ODE_ACCEL_DEFAULT = "matvec"

#: 逐信道带宽 ``B_i`` 的取值口径（P1 (23)/(29)/(30)/(34) 里的 ``B_i``，flexgrid 侧 WP-Q1）。
#:
#: ``"symbol_rate"``
#:     ``B_i`` = 连接自己声明的 :attr:`ConnectionSpec.symbol_rate_hz`。这是**默认档**：
#:     它与 WP-Q1 之前内核拿到的那个全局标量在均匀速率下是同一个 float64，故 legacy
#:     **逐位不变**，且既有逐连接场景（统一 50 GHz 声明速率、n_slots 各异）也逐位不变。
#:
#: ``"occupied"``
#:     ``B_i`` = :attr:`ConnectionSpec.occupied_bandwidth_hz`（= ``n_slots`` × 槽宽）。
#:     这一档才让**多槽宽度真正进入 QoT**（本 WP 的目标）。P1 (23) 称 ``B_i`` 为
#:     "the channel bandwidth"，P2 称 ``B_k`` 为 "bandwidth of the channel k"——
#:     两者都没说清是符号率还是占用带，故本仓**显式暴露**该选择，不替读者隐含。
#:
#: ⚠️ 两档都只是"用哪个既有字段当 ``B_i``"，**不引入任何新物理**：取值分别来自
#: ``ConnectionSpec`` 已有的两个字段，未改动任何公式系数。
CHANNEL_BANDWIDTH_SOURCES = ("symbol_rate", "occupied")
CHANNEL_BANDWIDTH_DEFAULT = "symbol_rate"

#: ``P_i`` 与 ``B_i`` 的归一化口径（WP-Q1 §3.4：必须显式，不许隐含）。
#:
#: ``"fixed_power"``
#:     每条连接的 ``P_i`` 就是它自己的入纤总功率，**与 ``B_i`` 无关**。这是**默认档**，
#:     也是本仓 ``ConnectionSpec.launch_power_w`` 既有语义（"单连接入纤总功率"）。
#:     ⇒ 宽连接 = 同样功率摊到更宽的谱上 ⇒ PSD 更低、噪声更大。
#:
#: ``"fixed_psd"``
#:     恒定功率谱密度：``P_i`` 按 ``B_i`` 成比例放大，锚点是构造参数 ``symbol_rate_hz``，
#:     即 ``P_i_eff = P_i × B_i / symbol_rate_hz``。⇒ 宽连接功率同比增长，PSD 不变。
#:
#: ⚠️ ``B_i`` 均匀时该缩放因子恒为 1.0，与 ``1.0`` 相乘逐位不变 ⇒ legacy 安全。
POWER_WIDTH_MODES = ("fixed_power", "fixed_psd")
POWER_WIDTH_DEFAULT = "fixed_power"

#: FG-U1 拉曼增益谱档位。
#:
#: ``"legacy"``
#:     合成折线（13.2 THz 线性爬升、15.6 THz 跌到零、之后恒零）。这是 FG-4a 的
#:     数值金标准与下游全部既有期望值所依赖的那条曲线。**FG-15 起不再是默认档，
#:     但档必须留着**：旧结果的可复现性、以及 FG-1 的 (A) 出处断言，全都依赖它
#:     仍可显式取到且结果逐位不变（见 `tests/test_fg15_legacy_bitwise.py`）。
#: ``"tabular"``
#:     有出处的宽波段熔融石英谱（Lin & Agrawal 2006 的响应函数经傅里叶变换得到的
#: ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：拉曼谱档位随第二套物理一起收窄为 **``("legacy",)``**。
#:
#: ``"legacy"``
#:     合成折线（13.2 THz 线性爬升、15.6 THz 跌到零、之后恒零）。闭式 ECZ 的冻结核
#:     ``numba_kernels.py`` 内部走 ``_calc_raman_profile_jit``，**只认这条折线**。
#:     这是它与 ``flexgrid-dev @ cb24704`` 逐字节相同的前提之一。
#:
#: ``"tabular"``（**随第二套物理移除**）
#:     有出处的宽波段熔融石英谱（Lin & Agrawal 2006 的响应函数经傅里叶变换得到的
#:     闭式增益谱，见 `physics.raman_spectrum`）。它**只经由拉曼 ODE** 进入
#:     计算——闭式 ECZ 没有拉曼 ODE，结构上无处可用，故随该档一并移除。
#:
#:     ⚠️ `physics/raman_spectrum.py`（谱表 + 闭式 + 互校）**本身保留**：它是超宽带
#:     物理资产，有自己的出处与互校测试，不在本轮的移除清单内。移除的只是
#:     **评估器不再暴露该档**，不是删掉那份物理谱。
RAMAN_SPECTRUM_MODES = ("legacy",)
RAMAN_SPECTRUM_DEFAULT = "legacy"

#: ══════════════════════════════════════════════════════════════════════
#: NLI 物理：**唯一一套**（WP-MERGE-FG-FLEX **第 5 轮**移除第二套物理）
#: ══════════════════════════════════════════════════════════════════════
#:
#: 第 3 轮曾把 NLI 物理拆成两套（闭式 / Eq. 21 张量化）并用一个物理档位形参
#: 区分。第 5 轮**移除了该档位机制本身**：
#:
#: * 本产品（``flexgrid-dev``）由 ``ecz-dev`` 衍生，**只需要闭式 ECZ**；
#: * 被移除的那套是 ``wp/fg-integrate`` 的**另一条模型线**（P2 Eq. 21 张量化、
#:   符号率口径），不属于本产品交付物；
#: * 加速已在 ECZ 侧建好（``numba_kernels_p1_accel.py``，第 4 轮），不依赖它。
#:
#: 故该物理档位形参、默认档常量与三张按档分派表**一并移除**——只剩一套物理时，
#: 按档查表没有消费者，留着是死代码。
#:
#: **唯一物理**：P1 (23)/(29)/(30)/(34) 闭式。带宽量是**逐信道** ``B_i`` / ``B_ℓ``
#: （P1 (29) 的 ``asinh(φ_i B_i²/πα)``、(30) 的 ``atan(φ_{i,ℓ} B_i/α_ℓ)``、
#: (34) 的 ``B_i``）。核在 :mod:`network_nlo_eval.models.numba_kernels`，
#: 字节 = ``flexgrid-dev @ cb24704`` 的冻结内核（``6611c416…``，逐字节未变）。
#:
#: ⚠️ 本档承诺与 ``cb24704`` **逐位**相同，因此 V1''' 判据钉在 ``ode_accel="off"``。
#: 默认档 ``"matvec"`` 与 ``off`` 的差是 max_rel **1.269e-15**（仅 XPM 求和次序，
#: 预算 1e-12 **不放宽**），那是第 4 轮**按实测**做的裁决，本轮不动。

#: ⚠️ 历史留档（**第 4 轮**的默认档裁决，本轮**不动**该结论）：
#:
#: ECZ 的三档是 ``"off"``（冻结参考）、``"cached"``（跨段记忆化）、``"matvec"``。
#: 第 4 轮把默认档由 ``"off"`` 改成 ``"matvec"``，依据是**实测**：
#:
#: * **延迟**（N=68、混合 n_slots、loadavg 1.5–3.1、20 核、MKL 12 线程）：
#:   串行 2.71–2.94x、批量 3.66–6.91x、端到端 10.22–19.86x（四轮实测的**区间**）。
#: * **精度**：只有 ``sigma2_xpm_w`` 漂移（max_rel 1.269e-15，max 11 ULP，
#:   相对 1e-12 预算有 ~790 倍余量）；出纤功率 / SPM / ASE 与 ``off`` 档
#:   **逐位相同**。
#: * **代价**（如实记）：默认档的读数**不再与 ``cb24704`` 逐位相同**——
#:   逐位相同的是**显式指定**的 ``"off"`` 档，故 V1''' 钉在 ``ode_accel="off"``。
#: * **冷启动**：(N,N) 对矩阵的一次性构建 0.074 ms（evaluator 级，跨跨段/跨候选
#:   复用）。**第一次**调用比 ``off`` 略慢 1.3–1.7x（多付一次构建），第二次起反超。
#:
#: 移除前的按档分派表形状（**已删**，此处留档以便追溯）：ECZ 三档
#: ``("off", "cached", "matvec")``；被移除的那套四档 ``("off", "cached",
#: "cached_g", "matvec")``。
#:
#: ⚠️ 其中 ``"cached_g"``（拉曼 G 矩阵缓存）**只属于**那套物理：G 矩阵缓存是
#: 拉曼 ODE 的结构，闭式 ECZ **没有**拉曼 ODE，故它一起移除。

#: ⚠️ 历史留档（**已删**的两张按档表，留档以便追溯）：ECZ 允许 ``("legacy",)``、
#: 默认 ``"legacy"``；被移除的那套允许 ``("legacy", "tabular")``、默认
#: ``"tabular"``。
#:
#: ECZ 的拉曼谱**只有** ``"legacy"`` 一档：闭式冻结核内部走
#: ``_calc_raman_profile_jit`` 那条合成折线。``"tabular"`` 只经由拉曼 ODE
#: 进入计算，故随该物理移除（见 :data:`RAMAN_SPECTRUM_MODES`）。


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
        ode_accel: str | None = None,
        channel_bandwidth_source: str = CHANNEL_BANDWIDTH_DEFAULT,
        power_width_mode: str = POWER_WIDTH_DEFAULT,
        raman_spectrum: str | None = None,
    ):
        """初始化 ISRS-GN 评估器。

        Args:
            grid: 频谱网格配置（遗留 ``SpectrumGrid`` 或 ``FlexibleGrid``）。
            ref_fiber_config: 参考光纤配置，用于计算各种波长相关参数。
            symbol_rate_hz: 系统符号速率 (Hz)。仅在 `connections is None`
                （遗留模式）时作为标量使用；给定 `connections` 时每连接的
                `ConnectionSpec.symbol_rate_hz` 优先。
            phi_kurtosis: 峰度修正，0.0 为高斯，-0.6190 为 64QAM。
            connections: (可选) 灵活栅格连接列表。给定后进入**逐连接模式**：
                全部逐连接数组按连接中心频率计算。为 None 时保持遗留行为，数值逐位不变。
            ode_accel: 加速档位，见 :data:`ODE_ACCEL_MODES`。**给 ``None`` 时取**
                :data:`ODE_ACCEL_DEFAULT`（= ``"matvec"``，第 4 轮按实测定档，**不动**）。
                三档：``"off"`` 冻结参考（与 ``cb24704`` 逐位相同）、``"cached"``
                跨段记忆化（与 ``"off"`` 逐位相等）、``"matvec"`` 去分支 matvec
                （XPM 信道对矩阵 + 矩阵-向量乘，相对偏差 <= 1e-12）。
                ⚠️ 想拿**与 ``cb24704`` 逐位相同**的读数，显式配 ``ode_accel="off"``。
            slots_per_channel: (可选) 每**遗留信道**占用的槽数，仅用于把 ROADM 的
                ``blocked_channels`` 展开成 ``blocked_slots``。遗留模式下不影响任何数值。
            raman_spectrum: 拉曼增益谱档位，见 :data:`RAMAN_SPECTRUM_MODES`。
                **给 ``None`` 时取** :data:`RAMAN_SPECTRUM_DEFAULT`（= ``"legacy"``）。
                ⚠️ 第 5 轮**只保留** ``"legacy"``：闭式冻结核内部走
                ``_calc_raman_profile_jit`` 那条合成折线；``"tabular"`` 只经由 P2 的
                拉曼 ODE 进入计算，已随该档移除（谱本身仍留在
                ``physics/raman_spectrum.py``）。
            channel_bandwidth_source: 逐信道 ``B_i`` 的取值口径，见
                :data:`CHANNEL_BANDWIDTH_SOURCES`。默认 ``"symbol_rate"``（legacy 逐位不变）。
            power_width_mode: ``P_i`` / ``B_i`` 归一化口径，见 :data:`POWER_WIDTH_MODES`。
                默认 ``"fixed_power"``。

        Raises:
            ValueError: 档位名非法；``channel_bandwidth_source`` / ``power_width_mode``
                取值非法；逐连接带宽 ``B_i`` 非正 / 非有限；``slots_per_channel < 1``。
        """
        if ode_accel is None:
            ode_accel = ODE_ACCEL_DEFAULT
        if raman_spectrum is None:
            raman_spectrum = RAMAN_SPECTRUM_DEFAULT
        if ode_accel not in ODE_ACCEL_MODES:
            raise ValueError(f"ode_accel must be one of {ODE_ACCEL_MODES}, got {ode_accel!r}.")
        if raman_spectrum not in RAMAN_SPECTRUM_MODES:
            raise ValueError(f"raman_spectrum must be one of {RAMAN_SPECTRUM_MODES}, got {raman_spectrum!r}.")
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
        self.phi_kurtosis = phi_kurtosis
        self.cr_w_m_hz = _SRS_CR  # 拉曼增益系数
        self.k_bar = min(sum(pump.power_w for pump in ref_fiber_config.raman_pumps), 1.0)
        self.delta_f_co_hz = 15e12  # 石英光纤受激拉曼散射（SRS）增益谱的截断频率
        self.ode_accel = ode_accel
        self.raman_spectrum = raman_spectrum
        self.n2_m2_w = ref_fiber_config.nonlinear_index_n2
        self.connections = list(connections) if connections is not None else None
        if slots_per_channel is not None and slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")
        self.slots_per_channel = slots_per_channel
        self.channel_bandwidth_source = channel_bandwidth_source
        self.power_width_mode = power_width_mode

        self._span_cache: dict[tuple, tuple] = {}
        self._precompute_channel_dependent_properties()
        self._fiber_evaluator_cache: dict[str, MultiBandISRSGN] = {}

    @property
    def n_nodes(self) -> int:
        """节点数：遗留模式 = 信道数；逐连接模式 = 活跃连接数。"""
        return int(self.f_abs_hz.shape[0])

    @property
    def symbol_rate_hz_channels(self) -> NDArrayFloat:
        """逐节点**声明符号率**数组 (Hz)，长度 = :attr:`n_nodes`。

        这是 `wp/fg-uwb-fid` 的 `isrs_gn.py:215-216` 同一份索引空间适配：
        遗留模式下 ``f_abs_hz is spectrum_grid.frequencies``，且本属性恒等于
        ``np.full(n_nodes, symbol_rate_hz)``，故走它取值的代码在 legacy 下
        **逐位等价**于走标量 ``symbol_rate_hz``。

        ⚠️ WP-Q1 之后**不要**再用它当 P1 的 ``B_i``：本属性是"声明的符号率"，
        而 P1 (29)/(30)/(34) 里的 ``B_i`` 是"信道带宽"，两者在
        ``channel_bandwidth_source="occupied"`` 下并不相等。取 ``B_i`` 请走
        :attr:`channel_bandwidth_hz_channels`。
        """
        if self.connections is not None:
            return np.array([float(conn.symbol_rate_hz) for conn in self.connections], dtype=np.float64)
        return np.full(self.n_nodes, float(self.symbol_rate_hz), dtype=np.float64)

    @property
    def channel_bandwidth_hz_channels(self) -> NDArrayFloat:
        """逐节点**信道带宽** ``B_i`` 数组 (Hz)，长度 = :attr:`n_nodes`。

        这是 P1 §III (23)/(29)/(30)/(34) 四个公式里那个 ``B_i`` / ``B_ℓ`` 的
        **唯一**取值源：SPM 的 ``P_i³/B_i²`` 与 ``asinh(φ_i B_i²/πα)``、XPM 的
        ``P_ℓ²/B_ℓ`` 与 ``atan(φ_{i,ℓ} B_i/α_ℓ)``、ASE 的 ``B_i`` 全部读它。

        遗留模式（``connections is None``）：恒为 ``np.full(n_nodes, symbol_rate_hz)``，
        与 WP-Q1 之前内核拿到的那个全局标量是**同一个 float64** ⇒ legacy 逐位不变
        （Q-J1 的"均匀 50 GHz ⇒ ``B_i`` 退化标量"就是这个恒等式）。

        逐连接模式：按 :attr:`channel_bandwidth_source` 从 ``ConnectionSpec`` 已有的
        字段取（``symbol_rate_hz`` 或 ``occupied_bandwidth_hz``），逐连接异构带宽因此
        受支持——WP-Q1 §3.3 解除了原先那条 ``ValueError``。

        Raises:
            ValueError: 取到的带宽非正或非有限（构造期由 :func:`__init__` 校验）。
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

    @property
    def ase_bandwidth_hz_channels(self) -> NDArrayFloat:
        """**ASE 用的**逐节点带宽 (Hz)。

        P1 (34) 的 ASE 是 ``NF·h·ν·B_i·(G−1)`` ⇒ 取
        :attr:`channel_bandwidth_hz_channels`（信道带宽 ``B_i``）。

        ⚠️ 第 5 轮移除第二套物理后，本属性**恒等于**
        :attr:`channel_bandwidth_hz_channels`；保留这个属性是为了让 SNR 装配层
        （``rwa/qot_checker.py``）**不用**知道 ASE 带宽取自哪个物理量。装配层只读
        这一个已解析好的逐节点数组 —— 「换加速档不应改变 SNR」的结构性保证因此
        继续成立（`tests/test_fg4g_matvec_attribution.py::
        test_fg4g_qot_gate_is_blind_to_ode_accel` 守的就是这一条）。

        遗留模式（``channel_bandwidth_source="symbol_rate"``）下
        :attr:`symbol_rate_hz_channels` 与本属性**逐位相同**。
        """
        return self.channel_bandwidth_hz_channels

    def _power_width_scale(self) -> NDArrayFloat:
        """``fixed_psd`` 口径下的逐节点功率缩放因子 (无量纲)。

        恒定功率谱密度 ⇒ ``P_i_eff = P_i × B_i / symbol_rate_hz``，锚点是构造参数
        ``symbol_rate_hz``（名义速率）。``B_i`` 均匀时因子恒为 1.0，与 1.0 相乘逐位
        不变 ⇒ legacy 安全。``fixed_power`` 口径恒返回全 1（逐位 no-op）。
        """
        if self.power_width_mode == "fixed_power":
            return np.ones(self.n_nodes, dtype=np.float64)
        return self.channel_bandwidth_hz_channels / float(self.symbol_rate_hz)

    def roadm_blocks_node(self, roadm: ROADMConfig, node_index: int) -> bool:
        """判断某节点（连接）是否被该 ROADM 阻断。

        灵活栅格下按**槽区间重叠**判定（切片语义）：连接与任一 ``blocked_slots``
        重叠即不可用。遗留模式保持按信道索引判定的既有行为（逐位不变）。

        冻结解析顺序：有 ``blocked_slots`` 用之；否则把 ``blocked_channels`` 按
        ``slots_per_channel`` 展开。逐连接模式下若给了 ``blocked_channels`` 但没给
        ``slots_per_channel``，则不做取整猜测——直接抛错，避免静默误判。

        ⚠️ WP-MERGE-FG-FLEX：本方法**委派**给 :meth:`_resolve_blocked_node_indices`，
        不再自带一份展开/重叠判定。合并前 FG 侧（`_resolve_blocked_node_indices`）与
        flexgrid 侧（本方法）各写了一遍同样的槽重叠逻辑；保留两份就等于两处可能漂移。
        现在阻断判定是**单一真源**，本方法只作为 flexgrid 侧的公开逐节点问法保留。
        """
        if self.connections is not None and node_index >= len(self.connections):
            return False
        return node_index in self._resolve_blocked_node_indices(roadm)

    @property
    def band_center_hz(self) -> float:
        """``f_rel_hz`` 的参考原点。遗留网格用中心频率，灵活栅格用频段中心。"""
        if isinstance(self.grid, FlexibleGrid):
            return self.grid.band_center_hz
        return self.grid.center_frequency_hz

    def _precompute_channel_dependent_properties(self) -> None:
        """预计算逐连接通用的波长依赖物理属性，如波长、有效模面积等。

        灵活栅格下这些量按**连接中心波长**计算（设计 §2.5 机械改），遗留模式下
        按栅格频率计算——两者在固定栅格特例下逐位相同。
        """
        if self.connections is not None:
            # 每连接一个节点，节点频率 = 连接中心频率。
            self.f_abs_hz = np.array([conn.centre_hz for conn in self.connections], dtype=np.float64)
            self.lambdas_m = C_LIGHT / self.f_abs_hz  # (m)
        else:
            self.lambdas_m = self.grid.wavelengths  # (m)
            self.f_abs_hz = self.grid.frequencies  # (Hz)

        # 衰减系数 (Np/m)
        #
        # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：这里**曾经**按 NLI 档位分派
        # （ECZ 走 flexgrid-dev 口径、被移除的那套走 FG 的 alpha0/1/2 波长
        # 多项式口径）。移除后只剩 flexgrid-dev 那一条。
        #
        # 保留的是 **flexgrid-dev 口径**：读
        # ``FiberSpanConfig.use_attenuation_polynomial``。为假（默认）时用
        # **常数**衰减 ``attenuation_db_km_ref·ln(10)/10/1e3``。
        # 这是本模块与 ``cb24704`` 逐位相同的前提：alpha 不同 ⇒
        # ``p_out_fiber`` 不同 ⇒ 自动增益 ``gain`` 不同 ⇒ P1 (34) 的 ASE
        # （``NF·h·ν·B_i·(gain−1)``）不同 ⇒ 场景读数不同。
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

        # ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：这里**曾经**有 P2 的构建期常量——
        # 拉曼 G 矩阵、张量化 NLI 张量与批量常量包。三者都**只**服务那套物理的
        # 拉曼 ODE，故一并移除。
        #
        # 顺带的性能事实（对 ECZ 是**净收益**）：``matvec`` 档此前每次构造都要
        # 多付一次 O(N²) 的 Raman G 对矩阵构建，而闭式 ECZ 的热路径**从不读**
        # Raman G 对矩阵。移除后该构建开销**归零**，ECZ 的数值**一位不变**。
        #
        # ECZ 侧现存的唯一构建期常量是 P1 XPM 的 (N,N) 信道对矩阵（见下）。
        #
        # ⚠️ WP-MERGE-FG-FLEX **第 4 轮**：P1 XPM 的 (N,N) 信道对矩阵。
        #
        # 它只依赖**信道几何**（波长、频率、带宽、逐信道衰减、色散、n2）——
        # 这些全是 **evaluator 级**量，不随跨段、不随候选变 ⇒ 一次构建、整批复用。
        # 延迟到首次调用再建（构造期不做无谓的 O(N^2) 工作）。
        self._p1_xpm_matrices: P1XpmMatrices | None = None

    def _p1_xpm_matrices_for(self) -> P1XpmMatrices:
        """P1 XPM 的 (N,N) 信道对矩阵（实例内记忆化，跨跨段/跨候选复用）。

        ⚠️ WP-MERGE-FG-FLEX **第 4 轮**。**构建期算一次的热路径常量**。

        它只依赖 evaluator 级量（``lambdas_m`` / ``f_rel_hz`` / ``B_i`` /
        ``alpha`` / ``beta2`` / ``beta3`` / ``n2``），**不随跨段、不随候选变**，
        所以缓存键就是 evaluator 本身——建一次就够。

        与 ``off`` 档的关系：``off`` 档的每一对都在**热路径**里现算 ``gamma_il`` /
        ``phi_il`` / 两个 ``arctan``；本档把它们**提前**到构建期，热路径只剩
        矩阵-向量乘。**P1 的算式一个字没改**。
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
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """P1 闭式核的 ``matvec`` 形态（单候选）。

        ⚠️ WP-MERGE-FG-FLEX **第 4 轮**：这是 ``off`` 档那个冻结内核的**加速替身**，
        不是第二条物理。出纤功率与 SPM **逐位相同**（同一个冻结核函数），
        只有 XPM 的求和次序变了。
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
            nf_lin_channels=self.nf_lin_channels,
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

        ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：这曾经是一个按物理档位分派到两套实现
        的**派发口**。移除第二套物理后只剩一套，派发**没有分支可分**，故本方法
        直接委派 ECZ 实现。
        """
        return self._calc_span_noise_and_power_p1(
            power_in_w, span_length_m, edfa_config, roadm_config, n_effective_spans
        )

    def _calc_span_noise_and_power_p1(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """P1 (23)/(29)/(30)/(34) 闭式路径（唯一物理）。

        本方法的算式**逐字取自** ``flexgrid-dev @ cb24704`` 的
        ``models/isrs_gn.py``（同一份 P1 物理），目的是让 ``p1_closed`` 档与
        flexgrid-dev **逐位相同**。三处与 P2 路径不同、且都是**物理**差异：

        1. **NF 默认值**：edfa 为 None（或其 NF 为 None）时取
           ``np.ones_like(self.nf_lin_channels)``（即 NF = 0 dB 的"未指定"口径），
           **不是** P2 路径的 ``self.nf_lin_channels``。
        2. **跨段后放大在 Python 层显式施加**（P1 (34)）：增益取自 EDFA 配置，
           ASE 按 ``NF·h·ν·B_i·(gain−1)`` 逐信道算。P1 核里那个 ASE 分支是
           **死代码**（核返回的 ASE 被丢弃，见 ``p1_eq34_ase_dead_kernel`` 锚点）。
        3. **ROADM 口径**：按 ``ROADMConfig.path_loss_db(operation)`` 取损耗，
           等化档把损耗补回来并加 booster ASE，滤波惩罚只乘噪声。

        Args:
            power_in_w: 各信道入纤功率 (W)。
            span_length_m: 跨段长度 (m)。
            edfa_config: 跨段后的 EDFA 配置。
            roadm_config: (可选) 跨段后的 ROADM 配置。
            n_effective_spans: 有效跨段数，用于 NLI 累积效应。

        Returns:
            Tuple 包含 ``(power_out_w, sigma2_spm_w, sigma2_xpm_w, sigma2_ase_w)``。
        """
        # 1. 处理放大器噪声系数 (NF)。
        # ⚠️ 与 P2 路径的差异见方法 docstring 第 1 条：P1 取 flexgrid-dev 口径
        #    （未指定 NF 时恒为 1.0），这是 p1_closed 与 cb24704 逐位相同的前提。
        if edfa_config is not None and edfa_config.noise_figure_db is not None:
            noise_figure_db = np.asarray(edfa_config.noise_figure_db, dtype=float)
            if noise_figure_db.ndim == 0:
                current_nf_lin = np.full_like(self.nf_lin_channels, 10 ** (float(noise_figure_db) / 10))
            else:
                if noise_figure_db.shape != (self.n_nodes,):
                    raise ValueError(
                        "EDFA noise-figure array must contain one value per channel "
                        f"({self.n_nodes},); got {noise_figure_db.shape}."
                    )
                current_nf_lin = 10 ** (noise_figure_db / 10)
        else:
            current_nf_lin = np.ones_like(self.nf_lin_channels)

        # 1.5 形状契约校验（设计 §3.4）。逐连接数组必须长度恰为 N 且同序对齐。
        n_conn = self.n_nodes
        if np.shape(power_in_w) != (n_conn,):
            raise ValueError(f"power_in_w must have shape ({n_conn},), got {np.shape(power_in_w)}.")
        current_nf_lin = np.asarray(current_nf_lin, dtype=np.float64)
        if current_nf_lin.shape != (n_conn,):
            raise ValueError(f"nf_lin_channels must have shape ({n_conn},), got {current_nf_lin.shape}.")

        # 2. 调用 P1 冻结内核 (热路径)。跨段记忆化是纯 memoization，命中时返回的
        #    就是上一次算出来的那些数组 ⇒ 与 "off" 逐位相等。
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
            # 纯 memoization：返回的就是上一次算出来的那些数组，逐位相等。
            p_out_fiber, s2_spm, s2_xpm = cached
        elif self.ode_accel == "matvec":
            # ⚠️ WP-MERGE-FG-FLEX **第 4 轮**：``matvec`` 档走 **P1 自己的**加速实现
            #    （``models/numba_kernels_p1_accel.py``）。
            #
            #    与 ``off`` 档的关系：
            #    * **出纤功率与 SPM 逐位相同**（走冻结核的同一个函数）；
            #    * **只有 XPM 的求和次序**变了（成对求和 → 矩阵-向量乘），
            #      实测 max_rel ~1e-15（预算 1e-12，**不放宽**）。
            #    * P1 的**算式一个字没改**——``numba_kernels.py`` 仍是 ``6611c416…``。
            p_out_fiber, s2_spm, s2_xpm, _ = self._p1_span_matvec(
                power_in_w, span_length_m, n_effective_spans
            )
        else:
            # ⚠️ 核的第四个返回值（ASE）是**死代码**：P1 (34) 的 ASE 在下面按
            #    "显式施加 EDFA" 那一步算，故这里丢弃。锚点 `p1_eq34_ase_dead_kernel`
            #    就是这条"死代码"事实的可执行形式。
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
                # 标量符号率：只用于 SRS 频带边界 f_m/f_M，**不是** B_i。
                symbol_rate_hz=self.symbol_rate_hz,
                # ⚠️ 逐信道带宽 B_i —— P1 (23)/(29)/(30)/(34) 的那个 B_i / B_ℓ。
                #    B_i 不许退回标量（任务书 §3）。
                b_hz_channels=self.channel_bandwidth_hz_channels,
                n_effective_spans=n_effective_spans,
                cr_w_m_hz=self.cr_w_m_hz,
                k_bar=self.k_bar,
                delta_f_co_hz=self.delta_f_co_hz,
            )
            if self.ode_accel == "cached":
                self._span_cache[cache_key] = (p_out_fiber, s2_spm, s2_xpm)

        # 3. 显式应用 EDFA。旧实现返回未放大的出纤功率，且 edfa=None 仍产生 ASE。
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
            # P1 (34) 的 B_i 走逐信道数组（WP-Q1）。legacy 下该数组恒等于标量
            # symbol_rate_hz ⇒ 与 WP-Q1 之前的算式逐位相同。
            s2_ase = current_nf_lin * 6.62607015e-34 * self.f_abs_hz * self.channel_bandwidth_hz_channels * np.maximum(
                gain - 1, 0
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

    def calc_span_noise_and_power_batch(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None = None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """单跨段内核的**批量入口**（FG-4e）。

        与 :meth:`_calc_span_noise_and_power` 算同一件事，区别只在布局：这里一次
        评估 B 个候选，形状是 ``(B, N)`` 进、四个 ``(B, N)`` 出。热路径是
        ``(B,N) @ (N,N)`` 的 GEMM（XPM 三次），**无数据相关控制流**；(N,N) 的
        构建期常量不按 batch 复制。

        ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：这曾经是按物理档位分派的（ECZ 走
        :meth:`_calc_span_noise_and_power_batch_p1`，被移除的那套走它的张量化
        批量核）。移除后只剩 ECZ 一套物理，本方法**直接委派** ECZ 批量实现。

        批量是默认档数学式的批量形态，所以要求 ``ode_accel == "matvec"``；
        **不匹配时抛** ``ValueError``，绝不静默回落到串行。

        逐候选与串行路径的相对偏差预算为 1e-12（**不放宽**）。
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
        return self._calc_span_noise_and_power_batch_p1(
            power_in_w, span_length_m, edfa_config, roadm_config, n_effective_spans
        )

    def _calc_span_noise_and_power_batch_p1(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """闭式 ECZ 的**批量**入口（第 4 轮新增）。

        与 :meth:`_calc_span_noise_and_power_p1` 算同一件事，区别只在布局：
        这里一次评估 B 个候选，形状是 ``(B, N)`` 进、四个 ``(B, N)`` 出。

        ⚠️ **不是静默回落**：档位不匹配时**抛** ``ValueError``。

        「显式施加 EDFA」与「ROADM 损耗」两步都按**逐候选**做，语义与串行入口
        :meth:`_calc_span_noise_and_power_p1` **逐项一致**（含 ECZ 档特有的 NF
        默认值口径：未指定 NF 时取全 1）。

        Args:
            power_in_w: ``(B, N)`` 候选入纤功率 (W)。零功率合法。
            span_length_m: 跨段长度 (m)。
            edfa_config: (可选) 跨段后 EDFA 配置。
            roadm_config: (可选) 跨段后 ROADM 配置。
            n_effective_spans: 有效跨段数。

        Returns:
            四个 ``(B, N)`` 数组。

        Raises:
            ValueError: ``ode_accel`` 不是 ``"matvec"``；``power_in_w`` 形状不对；
                NF 覆盖数组长度不对。
        """
        if self.ode_accel != "matvec":
            raise ValueError(
                "calc_span_noise_and_power_batch requires ode_accel='matvec': the batched "
                "kernel is the batch form of the P1 closed-form NLI (XPM channel-pair "
                f"matrices + matvec). Got ode_accel={self.ode_accel!r}."
            )

        P = np.asarray(power_in_w, dtype=np.float64)
        n_conn = self.n_nodes
        if P.ndim != 2 or P.shape[1] != n_conn:
            raise ValueError(
                f"power_in_w must have shape (B, {n_conn}), got {np.shape(power_in_w)}."
            )

        # --- 1. NF：P1 档口径（未指定时取全 1 = NF 0 dB）---
        if edfa_config is not None and edfa_config.noise_figure_db is not None:
            noise_figure_db = np.asarray(edfa_config.noise_figure_db, dtype=float)
            if noise_figure_db.ndim == 0:
                current_nf_lin = np.full(n_conn, 10 ** (float(noise_figure_db) / 10))
            else:
                if noise_figure_db.shape != (n_conn,):
                    raise ValueError(
                        "EDFA noise-figure array must contain one value per channel "
                        f"({n_conn},); got {noise_figure_db.shape}."
                    )
                current_nf_lin = 10 ** (noise_figure_db / 10)
        else:
            current_nf_lin = np.ones_like(self.nf_lin_channels)
        current_nf_lin = np.asarray(current_nf_lin, dtype=np.float64)

        # --- 2. P1 闭式核的批量形态 ---
        mats = self._p1_xpm_matrices_for()
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
            mats=mats,
        )

        # --- 3. 显式应用 EDFA（与串行入口同式，逐候选）---
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

        # --- 4. ROADM 损耗（P1 档口径：除以 path_loss_db）---
        if roadm_config:
            loss_lin = _db_to_lin(roadm_config.path_loss_db())
            p_out = p_out / loss_lin
            s2_spm = s2_spm / loss_lin
            s2_xpm = s2_xpm / loss_lin
            s2_ase = s2_ase / loss_lin

        return p_out, s2_spm, s2_xpm, s2_ase

    def for_fiber(self, fiber_config: FiberSpanConfig) -> "MultiBandISRSGN":
        """返回匹配链路光纤参数的缓存评估器.

        FU-2：这里过去**漏传** ``raman_spectrum``，子评估器因此总是退回默认档。
        该透传**保留**（第 5 轮后 ``raman_spectrum`` 只剩 ``"legacy"`` 一档，
        传不传是同一个值，**逐位不变**）——它是正确的接线形状。

        WP-O3 接线修正（flexgrid 侧，随本次合并并入）：逐连接模式下必须把
        ``connections``（以及 ``slots_per_channel`` / ``channel_bandwidth_source`` /
        ``power_width_mode``）一并传下去。否则子评估器会退回遗留分支并读
        ``FlexibleGrid.wavelengths``（不存在），直接 AttributeError。
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
                self.channel_bandwidth_source,
                self.power_width_mode,
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

    def _apply_roadm_p1(
        self,
        roadm: "ROADMConfig",
        is_last_hop: bool,
        power: NDArrayFloat,
        total_spm: NDArrayFloat,
        total_xpm: NDArrayFloat,
        total_ase: NDArrayFloat,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """ECZ 的 ROADM 装配（**逐字取自 flexgrid-dev**）。

        ⚠️ 为什么 P1 档**不**走 P2 档的 ``fold_roadm_terms`` 权威折叠：两者在
        **浮点最后几位**上不同——P1 的等化档是「先除以 loss、再乘回 loss」
        （``x / loss * loss``），而 IEEE-754 下 ``x / loss * loss != x``。这一位差别
        会传到 SNR 上，``p1_closed`` 档承诺与 ``flexgrid-dev @ cb24704`` **逐位**相同，
        所以必须走这一支。

        ⚠️ 与 FG-11「折叠逻辑只允许有一处」的关系：那条性质说的是 **P2 那套物理**
        的装配只允许有一处实现。本方法是**另一套物理**（P1）的装配，按编排方第 3 轮
        裁决「两套物理都留」而**必须**独立存在——把它并进 ``fold_roadm_terms`` 就等于
        强行把两套物理统一成一套，正是本轮要消除的东西。
        ``tests/test_fg13_evaluate_path_roadm.py`` 那条结构守卫扫的是
        ``evaluate_path`` 的**函数体**：P1 档的装配被放进本方法（而不是塞在
        ``evaluate_path`` 里），所以那条守卫**一个字都没改**、照样通过。

        Returns:
            装配后的 ``(power, total_spm, total_xpm, total_ase)``。
        """
        operation = "drop" if is_last_hop else "express"
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
        blocked = self._blocked_nodes_p1(roadm)
        if blocked:
            power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0
        return power, total_spm, total_xpm, total_ase

    def _blocked_nodes_p1(self, roadm: "ROADMConfig") -> list[int]:
        """ECZ 用的阻断节点解析（**逐字取自 flexgrid-dev**）。

        ⚠️ 与 :meth:`_resolve_blocked_node_indices`（合并后的单一真源）的**口径
        差异**只在遗留模式：flexgrid-dev 的遗留模式只认 ``blocked_channels``，
        **不**把 ``blocked_slots`` 当节点索引用；合并后的真源两者都认。

        这是**有意的**：`p1_closed` 档要跟 `flexgrid-dev @ cb24704` **逐位相同**，
        阻断集合不同 ⇒ 被阻断节点的 SNR（= −inf）就不同 ⇒ 场景读数就不同。
        两套物理各用各的阻断口径，不在这里"统一"。

        逐连接模式（``connections is not None``）：按槽区间重叠判定，与
        flexgrid-dev 的 ``roadm_blocks_node`` 逐字相同。
        """
        if self.connections is not None:
            blocked: list[int] = []
            for index in range(self.n_nodes):
                if self._roadm_blocks_node_p1(roadm, index):
                    blocked.append(index)
            return blocked
        return [index for index in range(self.n_nodes) if index in roadm.blocked_channels]

    def _roadm_blocks_node_p1(self, roadm: "ROADMConfig", node_index: int) -> bool:
        """逐节点问法（P1 档），逐字取自 ``flexgrid-dev`` 的 ``roadm_blocks_node``。"""
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

    def evaluate_path(self, optical_path: MultiSpanOpticalPath, power_in_w: NDArrayFloat) -> PathQoTResult:
        """按拓扑顺序评估光纤、跨段 EDFA 和 ROADM。

        本方法只做**编排**：每个跨段都调用与 `origin/main` 完全相同的
        `_calc_span_noise_and_power`（内部走 main 的 RK4 ODE 内核），并在本层
        累积噪声。**内核未被改动**，命题 (A) 的逐位保证不受本刀影响。

        ROADM 装配由 :meth:`_apply_roadm_p1` 完成，**逐字取自**
        ``flexgrid-dev @ cb24704``：按端口取损耗（末跳 ``drop``、其余 ``express``）、
        ``equalize_output_power`` 为真时把损耗补回来并注入 booster ASE
        ``NF·h·ν·B_i·(gain−1)``、``filtering_penalty_db`` 作为**只乘噪声**的
        独立乘子。这是本模块与 ``cb24704`` 逐位相同的一部分。

        ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：FG-14 记录的「装配口径分歧」（本路径 vs
        ``rwa.qot_checker.fold_roadm_terms``）**随 P2 移除而消失**：那条分歧是
        **两套物理之间**的分歧，现在只剩一套。``fold_roadm_terms`` 函数本身
        **保留**，仍被 ``rwa/qot_checker`` / ``physics/qot_checker`` 使用。
        """
        power = np.asarray(power_in_w, dtype=np.float64).copy()
        if power.shape != (self.n_nodes,) or np.any(power < 0):
            raise ValueError("power_in_w must be a non-negative vector with one value per connection.")
        # WP-Q1 §3.4：功率-宽度归一化口径**显式**生效处。fixed_power（全 1，逐位
        # no-op）与均匀 B_i 下的 fixed_psd（因子恒 1.0）都不改动 legacy 数字。
        power = power * self._power_width_scale()
        total_spm = np.zeros_like(power)
        total_xpm = np.zeros_like(power)
        total_ase = np.zeros_like(power)
        span_count = 0
        roadm_configs = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
        n_hops = len(optical_path.fiber_configs)

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
            # ROADM 装配**逐字取自** `flexgrid-dev @ cb24704`（见 :meth:`_apply_roadm_p1`）。
            #
            # ⚠️ WP-MERGE-FG-FLEX 第 5 轮：这曾经按 NLI 档位分派——ECZ 走本方法，
            # P2 走 FG-14 的 `fold_roadm_terms` 口径。两者差别是**浮点级**的：
            # `x / loss * loss` 与 `x` 在 IEEE-754 下**不**逐位相等，两套口径的 SNR
            # 会差最后几位。移除 P2 后只剩本方法，``fold_roadm_terms`` 不再被本模块
            # 的 ``evaluate_path`` 调用。
            #
            # ⚠️ 注意：``fold_roadm_terms`` **本身没有删除**，它仍是 ``rwa/qot_checker``
            # 与 ``physics/qot_checker`` 的权威折叠函数；这里只是不再由本路径调用。
            power, total_spm, total_xpm, total_ase = self._apply_roadm_p1(
                roadm, hop_index == n_hops - 1, power, total_spm, total_xpm, total_ase
            )

        total_noise = total_spm + total_xpm + total_ase
        snr = np.full_like(power, -np.inf)
        active = (power > 0) & (total_noise > 0)
        snr[active] = 10 * np.log10(power[active] / total_noise[active])
        snr[(power > 0) & (total_noise == 0)] = np.inf
        return PathQoTResult(
            power, total_spm, total_xpm, total_ase, snr, span_count, sum(roadm is not None for roadm in roadm_configs)
        )
