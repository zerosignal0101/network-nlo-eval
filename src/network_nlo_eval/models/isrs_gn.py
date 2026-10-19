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
from network_nlo_eval.models.batch_kernels import (
    BatchSpanConstants,
    batched_span_noise_and_power,
    build_batch_constants,
)
# ⚠️ WP-MERGE-FG-FLEX 第 3 轮：两套 NLI 物理**都留**，用 `nli_model` 档位区分。
#
# * `numba_kernels`     —— P1 (23)/(29)/(30)/(34) 闭式，逐信道 `B_i`。
#                          字节 = flexgrid-dev @ cb24704 的冻结内核。
# * `numba_kernels_p2`  —— P2 Eq. 21 张量化，符号率口径。
#                          字节 = wp/fg-integrate @ 57c7221 的冻结内核。
#
# 两份核**不共用**任何一个物理函数；只有纯数学工具（`_db_to_lin` 等）在两边
# 逐位相同——那份等同性由 `tests/test_merge_nli_models.py` 守着。
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
from network_nlo_eval.models.numba_kernels_p2 import (
    _calc_span_noise_and_power_jit as _p2_calc_span_noise_and_power_jit,
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

#: ``ode_accel`` 档位（WP-MERGE-FG-FLEX 合并两侧档位）。
#:
#: ⚠️ 本常量在本次合并里取**两侧并集**，不是二选一：
#: flexgrid-dev 提供了 ``"off"`` / ``"cached"``；wp/fg-integrate 提供了
#: ``"off"`` / ``"cached_g"`` / ``"matvec"``。两套机制**互补且彼此独立**：
#: ``"cached"`` 是**跨段结果记忆化**（按 (入纤功率, 跨段长度, 跨段数, NF) 缓存整跨段
#: 输出，命中时返回同一个 ndarray 对象 ⇒ 与 ``"off"`` 逐位相等）；
#: ``"cached_g"`` 是**拉曼 G 矩阵预计算缓存**（内部仍用未改动的 ``_calc_dp_dz``）。
#: 合并后四档都在，``"off"`` 仍是二者的公共冻结参考路径。
#:
#: ``"off"``
#:     每次调用都在核内重建 G —— FG-2 与 WP-O2 共同的冻结参考路径。
#: ``"cached"``
#:     跨段记忆化（flexgrid 侧 WP-O2 移植，纯 memoization，与 ``"off"`` 逐位相等）。
#: ``"cached_g"``
#:     G 按 (频率栅格, Cr, 谱形常数) 预计算并缓存，内部仍用未改动的
#:     ``_calc_dp_dz``。输出与基线**逐位相等**。
#: ``"matvec"``
#:     G 缓存**并且**把逐对分支换成两次 matvec。数学式不变，但浮点求和次序
#:     改变，实测最大相对偏差 ~1e-14（F4a-4 预算是 1e-12）。
#:
#: **默认是 ``"matvec"``**（FG-4b 编排方裁定，WP-MERGE-FG-FLEX §3 禁止改动该默认档）。
#: 注意这与合并前 flexgrid-dev 的 ``"off"`` 不同——那是 flexgrid 侧单边默认值，
#: 本次合并按任务书统一到 FG 的生产档；flexgrid 的 ``B_i``/``P_i`` 参数化在**任一档下**
#: 都照常生效，不因默认档切换而失效。
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
ODE_ACCEL_MODES = ("off", "cached", "cached_g", "matvec")
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
#:     闭式增益谱，见 `physics.raman_spectrum`）。它会**改变结果**——这是物理改进，
#:     不是数值重排，因此走**自己的验收闸门**，不与速度线的 <=1e-12 同结果预算混用。
#:
#: **FG-15 起 ``"tabular"`` 是全局默认。** 理由是物理，不是偏好：legacy 的两个
#: 截断（13.2 / 15.6 THz）使得**相隔 >15.6 THz 的耦合被迫为零**，而 S+C+L+U 跨
#: 约 27 THz，FG-U1 实测保真误差也集中在 S/C 段（最差 −8.260 dB）。tabular 是这个
#: 运行区间的正确模型。
#:
#: 两档都用 ``G = cr_w_m_hz * 13.2e12 * S(|df|)`` 的同一语义：``cr_w_m_hz`` 管幅度、
#: ``S`` 管谱形，且在 13.2 THz 处**逐位相等**（legacy 的 ``S`` 就是那条折线）。
RAMAN_SPECTRUM_MODES = ("legacy", "tabular")
RAMAN_SPECTRUM_DEFAULT = "tabular"

#: ══════════════════════════════════════════════════════════════════════
#: NLI 物理档位（WP-MERGE-FG-FLEX **第 3 轮**裁决）
#: ══════════════════════════════════════════════════════════════════════
#:
#: ⚠️ **为什么有这一档**：第 2 轮把 :mod:`network_nlo_eval.models.numba_kernels`
#: 整份换成了 FG 的字节，同时把 P1 的 ``B_i`` **改名成** ``symbol_rate_hz`` 喂进
#: P2 的公式。那个组合是「P2 的 GN 模型 + 占用带宽当符号率」，**两边都没验证过**。
#: 第 3 轮改判：**两套物理都留**，用显式档位区分，谁都不许被另一套替换。
#:
#: ``"p1_closed"``（**默认**）
#:     P1 (23)/(29)/(30)/(34) 闭式。带宽量是**逐信道** ``B_i`` / ``B_ℓ``
#:     （P1 (29) 的 ``asinh(φ_i B_i²/πα)``、(30) 的 ``atan(φ_{i,ℓ} B_i/α_ℓ)``、
#:     (34) 的 ``B_i``）。核在 :mod:`network_nlo_eval.models.numba_kernels`，
#:     字节 = ``flexgrid-dev @ cb24704`` 的冻结内核。
#:
#: ``"p2_gn"``
#:     P2 Eq. 21 张量化。带宽量是**符号率** ``rs_hz``（核内的 ``symbol_rate_hz``），
#:     指纹是 ``(16/27)·γ²P³/rs²·T²`` 与 ``asinh(3φ·rs²/(8π·a_t))``。核在
#:     :mod:`network_nlo_eval.models.numba_kernels_p2`，字节 =
#:     ``wp/fg-integrate @ 57c7221`` 的冻结内核。
#:
#: ⚠️ **这两档不能互相比较**：它们是两套物理，数字本来就该不同。拿 ``p1_closed``
#: 的读数去对 FG 的金标准（或反过来）并期望逐位相同，是把两套物理当成一套——
#: 这正是第 2 轮 V1 判据本身的错误。各自的源头才是各自的判据。
#:
#: 默认取 ``"p1_closed"``：本产品是**逐连接带宽分辨率**的 QoT，P1 的 ``B_i``
#: 才是它的物理；FG 的加速是**实现结构**，不该反过来决定物理。
NLI_MODELS = ("p1_closed", "p2_gn")
NLI_MODEL_DEFAULT = "p1_closed"

#: 各 NLI 档位**允许**的 ``ode_accel`` 取值。
#:
#: ``"p1_closed"`` 三档：``"off"``（冻结参考）、``"cached"``（跨段记忆化）、
#: ``"matvec"``（**第 4 轮新增**：P1 XPM 的信道对矩阵 + 矩阵-向量乘）。
#: ⚠️ 第 4 轮之前这里是**两档**，注释写着「P1 核里没有拉曼 G 矩阵、没有去分支
#: matvec、没有张量化 NLI——那三样都是 FG 的**实现结构**，不是 P1 的物理。
#: 本轮**不做** P1 核的加速（另立一轮）」。**第 4 轮就是那一轮**：
#: P1 侧的 ``matvec`` 是**新实现**（``models/numba_kernels_p1_accel.py``），
#: 它**不动 P1 的算式**——P1 (23)/(29)/(30)/(34) 的常数与 ``asinh``/``atan``
#: 参数形式逐字不变，``numba_kernels.py`` 仍是 ``6611c416…`` 的冻结内核，
#: ``"off"`` 档继续以它为基准。
#:
#: ⚠️ ``"cached_g"`` 仍然**只属于** ``"p2_gn"``：G 矩阵缓存是 FG 侧拉曼 ODE 的
#: 结构，P1 核里**没有**拉曼 ODE（``p1_closed`` 走的是解析出纤功率），
#: 给它配 ``"cached_g"`` 会静默跑成别的档——那是最坏的一种错。
#:
#: ``"p2_gn"`` 四档全在：``"off"``（冻结参考）、``"cached"``（跨段记忆化）、
#: ``"cached_g"``（G 缓存）、``"matvec"``（默认）。
ODE_ACCEL_MODES_BY_NLI_MODEL: dict[str, tuple[str, ...]] = {
    "p1_closed": ("off", "cached", "matvec"),
    "p2_gn": ("off", "cached", "cached_g", "matvec"),
}

#: 各 NLI 档位**默认**的 ``ode_accel``（构造参数给 ``None`` 时取这个）。
#:
#: ⚠️ ``ODE_ACCEL_DEFAULT``（= ``"matvec"``）**保持不变**——它仍是 ``p2_gn`` 档的
#: 生产默认，``tests/test_fg4b_nli_vectorization.py`` 断言它等于 ``"matvec"``，
#: 本轮不动那个常量。
#:
#: ⚠️ WP-MERGE-FG-FLEX **第 4 轮**：``p1_closed`` 档的默认档由 ``"off"`` 改成
#: ``"matvec"``。这是**按实测**（延迟 vs 精度）做的裁决，读数依据见
#: ``docs/wp_merge_fg_into_flex.md`` 的「第 4 轮」章节：
#:
#: * **延迟**（N=68、混合 n_slots、loadavg 0.87/1.08/0.99、MKL 12 线程）：
#:   串行 2.70x、批量 6.66x (B=68) / 7.32x (B=142)、端到端 18.0x / 19.9x。
#: * **精度**：只有 ``sigma2_xpm_w`` 漂移（max_rel 1.269e-15，max 11 ULP，
#:   相对 1e-12 预算有 ~790 倍余量）；出纤功率 / SPM / ASE 与 ``off`` 档
#:   **逐位相同**。
#: * **代价**（如实记）：默认档的读数**不再与 ``cb24704`` 逐位相同**——
#:   逐位相同的是**显式指定**的 ``"off"`` 档。V1' 的 5752 项对照因此**钉在
#:   ``ode_accel="off"``**（任务书 V1'' 的原文也正是「``p1_closed`` 的 **off 档**」）。
#: * **先例**：FG-4a 已经把生产默认档定成加速档 ``"matvec"``（同一 1e-12 预算），
#:   并把逐位对照的测试**钉在** ``"off"``；FG-15 对拉曼档做的是同一件事。
#:   第 3 轮给 ``p1_closed`` 定 ``"off"`` 的**原因**是「本轮不做 P1 核的加速
#:   （另立一轮）」——那个原因在本轮已经消失。
#: * **冷启动**：(N,N) 对矩阵的一次性构建 0.074 ms（evaluator 级，跨跨段/跨候选
#:   复用）。**第一次**调用比 ``off`` 略慢（多付一次构建），第二次起就反超。
ODE_ACCEL_DEFAULT_BY_NLI_MODEL: dict[str, str] = {
    "p1_closed": "matvec",
    "p2_gn": ODE_ACCEL_DEFAULT,
}

#: 各 NLI 档位**允许**的 ``raman_spectrum`` 取值。
#:
#: ``"p1_closed"`` 只有 ``("legacy",)``：P1 核用的是 legacy 那条合成折线
#: （``_calc_raman_profile_jit``），没有 tabular 闭式增益谱那条路。
RAMAN_SPECTRUM_MODES_BY_NLI_MODEL: dict[str, tuple[str, ...]] = {
    "p1_closed": ("legacy",),
    "p2_gn": ("legacy", "tabular"),
}

#: 各 NLI 档位**默认**的 ``raman_spectrum``（构造参数给 ``None`` 时取这个）。
#: ``RAMAN_SPECTRUM_DEFAULT``（= ``"tabular"``）同样保持不变，仍是 ``p2_gn`` 档
#: 的生产默认；``p1_closed`` 档唯一可选的是 ``"legacy"``。
RAMAN_SPECTRUM_DEFAULT_BY_NLI_MODEL: dict[str, str] = {
    "p1_closed": "legacy",
    "p2_gn": RAMAN_SPECTRUM_DEFAULT,
}

#: FG-4e 批量常量包的命中/未命中计数——证明 (N,N) 常量确实跨候选复用。
_BATCH_CACHE_STATS = {"hits": 0, "misses": 0}


def batch_cache_stats() -> dict[str, int]:
    """返回批量常量包缓存的命中/未命中计数。

    与 :func:`~network_nlo_eval.models.numba_kernels_p2.g_cache_stats` 同构：证据不是
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
        ode_accel: str | None = None,
        channel_bandwidth_source: str = CHANNEL_BANDWIDTH_DEFAULT,
        power_width_mode: str = POWER_WIDTH_DEFAULT,
        raman_spectrum: str | None = None,
        nli_model: str = NLI_MODEL_DEFAULT,
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
            ode_accel: 加速档位，见 :data:`ODE_ACCEL_MODES`。**给 ``None`` 时由
                :attr:`nli_model` 解析**：``"p1_closed"`` → ``"matvec"``（**第 4 轮**改判；
                P1 闭式核自己的加速档：XPM 信道对矩阵 + 矩阵-向量乘）；
                ``"p2_gn"`` → ``"matvec"``。
                ⚠️ 合并后的并集档位见 :data:`ODE_ACCEL_MODES`：``"off"`` 是 FG-2 与
                WP-O2 共同的冻结参考路径；``"cached"``（flexgrid 侧跨段记忆化）；
                ``"cached_g"`` 与基线逐位相等；``"matvec"``（两档的默认）用
                去分支 matvec，相对偏差 <= 1e-12。
                ⚠️ **档位与物理必须匹配**：给 ``"p1_closed"`` 配 ``"cached_g"`` 会抛
                ``ValueError``——P1 核走的是**解析**出纤功率，没有拉曼 G 矩阵可缓存，
                硬跑等于**静默**把加速请求丢掉。
                ⚠️ **第 4 轮**：``"p1_closed"`` **现在有** ``"matvec"`` 档了
                （``models/numba_kernels_p1_accel.py``，P1 自己的实现，不动 P1 算式）。
                想拿**与 ``cb24704`` 逐位相同**的读数，显式配 ``ode_accel="off"``。
            slots_per_channel: (可选) 每**遗留信道**占用的槽数，仅用于把 ROADM 的
                ``blocked_channels`` 展开成 ``blocked_slots``。遗留模式下不影响任何数值。
            raman_spectrum: FG-U1 拉曼增益谱档位，见 :data:`RAMAN_SPECTRUM_MODES`。
                **给 ``None`` 时由 :attr:`nli_model` 解析**：``"p1_closed"`` →
                ``"legacy"``（P1 核只有 legacy 折线那条路）；``"p2_gn"`` →
                ``"tabular"``。
                ⚠️ **合并说明**：``"off"`` 与 ``"cached"`` 是 **legacy-only** 的冻结路径
                （前者逐对重建 legacy 折线，后者只把它记忆化），结构上无法换谱。
                因此这两档必须**显式**配 ``raman_spectrum="legacy"``，否则构造期抛
                ``ValueError``——沿用 FG-15 既有的"显式拒绝而非静默回退"口径，
                不为 flexgrid 的旧调用点开静默回退的后门。
                ⚠️ ``"p1_closed"`` 档只允许 ``"legacy"``（P1 核没有 tabular 谱）。
            channel_bandwidth_source: 逐信道 ``B_i`` 的取值口径，见
                :data:`CHANNEL_BANDWIDTH_SOURCES`。默认 ``"symbol_rate"``（legacy 逐位不变）。
            power_width_mode: ``P_i`` / ``B_i`` 归一化口径，见 :data:`POWER_WIDTH_MODES`。
                默认 ``"fixed_power"``。
            nli_model: NLI 物理档位，见 :data:`NLI_MODELS`。默认 ``"p1_closed"``
                （P1 (23)/(29)/(30)/(34) 闭式，逐信道 ``B_i``）。
                ``"p2_gn"`` 走 P2 Eq. 21 张量化（符号率口径）。
                ⚠️ 两档是**两套物理**，数字本来就该不同，不得互相比较。

        Raises:
            ValueError: 档位名非法；``tabular`` 与 legacy-only 档位组合；
                ``ode_accel`` / ``raman_spectrum`` 与 ``nli_model`` 不匹配；
                ``channel_bandwidth_source`` / ``power_width_mode`` 取值非法；
                逐连接带宽 ``B_i`` 非正 / 非有限；``slots_per_channel < 1``。
        """
        if nli_model not in NLI_MODELS:
            raise ValueError(f"nli_model must be one of {NLI_MODELS}, got {nli_model!r}.")
        allowed_ode_accel = ODE_ACCEL_MODES_BY_NLI_MODEL[nli_model]
        allowed_raman_spectrum = RAMAN_SPECTRUM_MODES_BY_NLI_MODEL[nli_model]
        if ode_accel is None:
            ode_accel = ODE_ACCEL_DEFAULT_BY_NLI_MODEL[nli_model]
        if raman_spectrum is None:
            raman_spectrum = RAMAN_SPECTRUM_DEFAULT_BY_NLI_MODEL[nli_model]
        if ode_accel not in allowed_ode_accel:
            raise ValueError(
                f"ode_accel must be one of {allowed_ode_accel} for nli_model={nli_model!r}, "
                f"got {ode_accel!r}. The other modes belong to the other NLI physics: "
                f"see ODE_ACCEL_MODES_BY_NLI_MODEL."
            )
        if raman_spectrum not in allowed_raman_spectrum:
            raise ValueError(
                f"raman_spectrum must be one of {allowed_raman_spectrum} for "
                f"nli_model={nli_model!r}, got {raman_spectrum!r}. The other tiers belong "
                f"to the other NLI physics: see RAMAN_SPECTRUM_MODES_BY_NLI_MODEL."
            )
        if slots_per_channel is not None and slots_per_channel < 1:
            raise ValueError(f"slots_per_channel must be >= 1, got {slots_per_channel}.")
        if channel_bandwidth_source not in CHANNEL_BANDWIDTH_SOURCES:
            raise ValueError(
                f"channel_bandwidth_source must be one of {CHANNEL_BANDWIDTH_SOURCES}, "
                f"got {channel_bandwidth_source!r}."
            )
        if power_width_mode not in POWER_WIDTH_MODES:
            raise ValueError(f"power_width_mode must be one of {POWER_WIDTH_MODES}, got {power_width_mode!r}.")
        if raman_spectrum not in RAMAN_SPECTRUM_MODES:
            raise ValueError(f"raman_spectrum must be one of {RAMAN_SPECTRUM_MODES}, got {raman_spectrum!r}.")
        if raman_spectrum == "tabular" and ode_accel in ("off", "cached"):
            # `off` / `cached` 是 legacy-only 的冻结路径（核内逐对重建 legacy 折线，
            # `cached` 只把它记忆化），结构上无法换谱。这里显式拒绝，而不是悄悄退回。
            raise ValueError(
                'raman_spectrum="tabular" requires ode_accel="cached_g" or "matvec"; '
                f'ode_accel={ode_accel!r} is a frozen legacy-only reference path.'
            )
        self.grid = grid
        self.ref_fiber_config = ref_fiber_config
        self.symbol_rate_hz = symbol_rate_hz
        self.phi_kurtosis = phi_kurtosis
        self.cr_w_m_hz = _SRS_CR  # 拉曼增益系数
        self.k_bar = min(sum(pump.power_w for pump in ref_fiber_config.raman_pumps), 1.0)
        self.delta_f_co_hz = 15e12  # 石英光纤受激拉曼散射（SRS）增益谱的截断频率
        self.ode_accel = ode_accel
        self.raman_spectrum = raman_spectrum
        self.nli_model = nli_model
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
        """本档位下 **ASE 用的**逐节点带宽 (Hz)。

        两套物理的 ASE 带宽量**本来就不同**：

        * ``"p1_closed"``：P1 (34) 的 ASE 是 ``NF·h·ν·B_i·(G−1)`` ⇒ 取
          :attr:`channel_bandwidth_hz_channels`（信道带宽 ``B_i``）。
        * ``"p2_gn"``：P2 Eq. 21 的 ASE 用**符号率** ⇒ 取
          :attr:`symbol_rate_hz_channels`。

        ⚠️ **档位只活在本模块**（`models/isrs_gn.py`），**不进 SNR 装配层**：
        装配层（``rwa/qot_checker.py``）只读**这个已解析好的**逐节点数组，
        它自己**不知道**也不需要知道当前是哪套物理。这样「换档位不应改变 SNR」
        的结构性保证仍然成立（`tests/test_fg4g_matvec_attribution.py::
        test_fg4g_qot_gate_is_blind_to_ode_accel` 守的就是这一条）。

        遗留模式（``channel_bandwidth_source="symbol_rate"``）下两个来源**逐位
        相同**，故本属性在 legacy 下与直接读任一个都一样。
        """
        if self.nli_model == "p1_closed":
            return self.channel_bandwidth_hz_channels
        return self.symbol_rate_hz_channels

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
        # ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：这里按 NLI 档位分派，两套物理各用各的
        # 口径，**不**在这里"统一"。
        #
        # * ``"p1_closed"`` → **flexgrid-dev 口径**：读
        #   ``FiberSpanConfig.use_attenuation_polynomial``。为假（默认）时用
        #   **常数**衰减 ``attenuation_db_km_ref·ln(10)/10/1e3``。
        #   这是 p1_closed 档与 ``cb24704`` 逐位相同的前提：alpha 不同 ⇒
        #   ``p_out_fiber`` 不同 ⇒ 自动增益 ``gain`` 不同 ⇒ P1 (34) 的 ASE
        #   （``NF·h·ν·B_i·(gain−1)``）不同 ⇒ 场景读数不同。
        # * ``"p2_gn"`` → **FG 口径**：本路径**恒用** alpha0/1/2 波长多项式，
        #   不看 ``use_attenuation_polynomial``。这是 FG-14 记录并裁决过的
        #   「装配口径分歧」（`docs/fg14_assembly_provenance.md` /
        #   `results/fg14_assembly_divergence.json`），本轮**原样保留**。
        if self.nli_model == "p1_closed":
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
        else:
            # FG 侧口径：恒用 alpha0/1/2 波长多项式。
            #
            # 为什么 P2 档不读该开关：它在 **FG 树里本来就只对波形级
            # `physics/qot_checker` 生效**，`models/isrs_gn.py` 从来不读它——
            # 这正是 FG-14 记录并裁决过的分歧。合并若让 P2 档也开始读它，
            # 等于单方面改掉一条已裁决的分歧，并让默认档（``False`` ⇒ 常数衰减）
            # 与 `wp/fg-integrate` 的冻结结果全场景不等。
            # 保留分歧本身：想用常数衰减的调用方应走 `physics/qot_checker`。
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

        # ⚠️ WP-MERGE-FG-FLEX **第 4 轮**：P1 XPM 的 (N,N) 信道对矩阵。
        #
        # 它只依赖**信道几何**（波长、频率、带宽、逐信道衰减、色散、n2）——
        # 这些全是 **evaluator 级**量，不随跨段、不随候选变 ⇒ 一次构建、整批复用。
        # 与 FG 侧 ``_g_gain`` / ``_nli_cache`` 同一套「构建期常量」结构。
        # 延迟到首次调用再建（构造期不做无谓的 O(N^2) 工作）。
        self._p1_xpm_matrices: P1XpmMatrices | None = None

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
            # ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：张量化 NLI 的 ``rs`` 是 **P2 的符号率**，
            # 不是 P1 的逐信道 ``B_i``。上一轮这里塞的是 ``channel_bandwidth_hz_channels``
            # （把 ``B_i`` 改名成 ``symbol_rate_hz`` 喂进 P2 公式）——那正是本轮判废的
            # 「第三种东西」。改成符号率后，P2 档的 ``off`` 与 ``matvec`` 重新
            # 一致（实测 max_rel 1.8e-16，与 ``wp/fg-integrate`` 同）。
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

    def _p1_xpm_matrices_for(self) -> P1XpmMatrices:
        """P1 XPM 的 (N,N) 信道对矩阵（实例内记忆化，跨跨段/跨候选复用）。

        ⚠️ WP-MERGE-FG-FLEX **第 4 轮**。与 FG 侧的 ``_g_gain`` / ``_nli_cache``
        同一套结构：**构建期算一次的热路径常量**。

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

        ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：这是一个**派发口**，按 :attr:`nli_model`
        分派到两套**互不共用**的物理实现：

        * ``"p1_closed"`` → :meth:`_calc_span_noise_and_power_p1`
          （P1 (23)/(29)/(30)/(34) 闭式 + 逐信道 ``B_i``）
        * ``"p2_gn"`` → :meth:`_calc_span_noise_and_power_p2`
          （P2 Eq. 21 张量化 + 符号率）

        两套的数值**不互相比较**：它们是两套物理。
        """
        if self.nli_model == "p1_closed":
            return self._calc_span_noise_and_power_p1(
                power_in_w, span_length_m, edfa_config, roadm_config, n_effective_spans
            )
        return self._calc_span_noise_and_power_p2(
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
        """``nli_model="p1_closed"``：P1 (23)/(29)/(30)/(34) 闭式路径。

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

    def _calc_span_noise_and_power_p2(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """``nli_model="p2_gn"``：P2 Eq. 21 张量化路径。

        本方法的算式逐字取自 ``wp/fg-integrate @ 57c7221`` 的 ``models/isrs_gn.py``，
        目的是让 ``p2_gn`` 档与 FG **逐位相同**。

        计算单跨段的功率演化、NLI 噪声和 ASE 噪声。

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
        #
        # ⚠️ WP-MERGE-FG-FLEX：NF 形状解析取 **flexgrid 侧**（`ndim == 0` 判标量 +
        # 逐通道数组必须恰为 (n_nodes,) 否则报错，比 FG 的 `isinstance(float)` 更严），
        # 但**默认值取 FG 侧**：edfa 为 None（或其 NF 为 None）时用
        # ``self.nf_lin_channels``，**不是**全 1。
        #
        # 为什么必须这样：flexgrid-dev 在这个 else 分支上写的是
        # ``np.ones_like(self.nf_lin_channels)``，那等价于 **NF = 0 dB**。实测该差异
        # 使串行路径的 ``sigma2_ase_w`` 比 FG 小 3.981 倍（`_calc_nf_lin_jit` 在这些
        # 波长上约 3.98），批量路径不受影响 ⇒ `test_fg4e_cpu_batching` 的
        # batch≡serial 直接差 2.981（相对预算 1e-12）。
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
            p_out_fiber, s2_spm, s2_xpm, s2_ase = cached
        else:
            p_out_fiber, s2_spm, s2_xpm, s2_ase = _p2_calc_span_noise_and_power_jit(
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
                # ⚠️ WP-MERGE-FG-FLEX **第 3 轮改判**：P2 的带宽量是**符号率**，
                # 不是 P1 的信道带宽。上一轮把 P1 的 `b_hz_channels` **改名**成
                # `symbol_rate_hz` 喂进 P2 公式（还把占用带宽当符号率塞进去），
                # 得到的「P2 的 GN 模型 + 占用带宽当符号率」两边都没验证过。
                # 现在 P2 档**回到 FG 的口径**：传**声明的符号率**数组。
                # 逐信道的 `B_i` 属于 P1 档，去 `_calc_span_noise_and_power_p1`。
                # ⚠️ 默认档（`channel_bandwidth_source="symbol_rate"`）下两者逐位
                # 相同（同一恒等式），故 `p2_gn` 档的既有读数不受影响。
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
            if self.ode_accel == "cached":
                self._span_cache[cache_key] = (p_out_fiber, s2_spm, s2_xpm, s2_ase)

        # 3. 处理 ROADM 装配 (可选，在 Python 层处理非线性外围逻辑)
        if roadm_config:
            # FG-18：走 rwa.qot_checker.fold_roadm_terms 这**同一份**折叠函数对象，
            # 不再在本处合并 ``insertion_loss_db + filtering_penalty_db``。
            # 折叠的懒导入与 evaluate_path 同一个理由：rwa.qot_checker 在模块级
            # 反过来取本模块的 MultiBandISRSGN，模块级互相导入会成环。
            #
            # 口径差异（相对收敛前的合并式写法）：
            #   * 损耗改为 ``path_loss_db("express")`` = ``insertion_loss_db +
            #     express_attenuation_db``——**含**直通端口附加衰减；收敛前的
            #     合并式只用 ``insertion_loss_db``，不含端口项。
            #   * ``filtering_penalty_db`` **不再**并进光功率：它是噪声域乘子
            #     （fold_roadm_terms 文档第 3 条），折进信号功率是 FG 树旧口径。
            #   * ``equalize_output_power`` 为真时信号**不再**被 loss 衰减，改注入
            #     booster ASE ``NF·h·ν·Rs·(loss−1)``——与 evaluate_path 同式。
            #
            # ``is_last_hop`` 取 False：逐跨段 API **不知道**自己是否末端跳（端口
            # 选择依赖 hop 序号，见 evaluate_path 的
            # ``is_last_hop=hop_index == n_hops - 1``），此处按 express 端口求值。
            from network_nlo_eval.rwa.qot_checker import fold_roadm_terms

            terms = fold_roadm_terms(roadm_config, is_last_hop=False)
            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
            else:
                p_out_fiber = p_out_fiber / terms.loss
                s2_spm = s2_spm / terms.loss
                s2_xpm = s2_xpm / terms.loss
                s2_ase = s2_ase / terms.loss

        # 3. ⚠️ WP-MERGE-FG-FLEX：此处**不**施加 EDFA，返回的就是 njit 内核算出的
        #    光纤出纤功率与光纤噪声，与 `wp/fg-integrate` 逐位一致。
        #
        #    flexgrid-dev 曾在这里加过一段"显式应用 EDFA"（`p_out = p_out_fiber * gain`
        #    并用增益与 NF 重算 ASE）。合并取回 FG 口径，理由三条：
        #      * 判据 V1（首要判据）要求与 `wp/fg-integrate` 逐位相同。实测保留该块时
        #        `p_out_w` 由 3.775e-05（光纤输出）变成 1.000e-03（EDFA 后），
        #        `sigma2_ase_w` 同步被重写 ⇒ 7 个场景 29 项全部不一致。
        #      * 该块会与调用方的 EDFA 噪声**重复计入**：RWA 层
        #        (`rwa/qot_checker.py` / `rwa/link_noise.py`) 自己按 `NF·h·ν·B_i·(gain−1)`
        #        算放大器 ASE。
        #      * FG 的场景/脚本 (`fg1_scenarios` / `fg4a_scenarios` / `fg14_divergence_probe`)
        #        都把本函数的返回值直接喂回下一跨段，语义就是"跨段输出"。
        #    跨段后放大由调用方负责，本函数只负责跨段内物理。
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
                ``(B, N)`` 或 N 与本 evaluator 不符；NF 覆盖数组长度不对；
                ``nli_model`` 不在 ``("p2_gn", "p1_closed")`` 里。
        """
        # ⚠️ WP-MERGE-FG-FLEX **第 4 轮**：P1 档**现在有**批量路径了。
        #
        # 第 3 轮这里**显式拒绝** ``p1_closed``（而不是悄悄退回串行）——那是对的，
        # 因为当时批量核确实只是 P2 的实现结构。第 4 轮给 P1 补上了**自己的**
        # 批量实现（``models/numba_kernels_p1_accel.py``），所以现在按
        # :attr:`nli_model` **分派**，两套物理各走各的批量核。
        #
        # ⚠️ 仍然**没有**静默回落：档位不匹配时照样抛 ``ValueError``。
        if self.nli_model == "p1_closed":
            return self._calc_span_noise_and_power_batch_p1(
                power_in_w, span_length_m, edfa_config, roadm_config, n_effective_spans
            )
        if self.nli_model != "p2_gn":
            raise ValueError(
                "calc_span_noise_and_power_batch is only available for nli_model='p2_gn' "
                "or 'p1_closed': the batched kernel is the batch form of the tier's own "
                f"NLI physics. Got nli_model={self.nli_model!r}."
            )
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
            # FG-18：与串行入口 :524 走**同一份** ``fold_roadm_terms``，
            # 不再在本处合并 ``insertion_loss_db + filtering_penalty_db``。
            # 口径与理由见串行入口的注释；``is_last_hop`` 同样取 False（逐跨段
            # API 无从知道 hop 序号）。(B, N) 布局下 booster 项按 (N,) 广播。
            from network_nlo_eval.rwa.qot_checker import fold_roadm_terms

            terms = fold_roadm_terms(roadm_config, is_last_hop=False)
            if terms.equalize:
                s2_ase = s2_ase + terms.booster_base * np.asarray(
                    self.f_abs_hz, dtype=np.float64
                ) * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64) * (terms.loss - 1.0)
            else:
                p_out = p_out / terms.loss
                s2_spm = s2_spm / terms.loss
                s2_xpm = s2_xpm / terms.loss
                s2_ase = s2_ase / terms.loss

        return p_out, s2_spm, s2_xpm, s2_ase

    def _calc_span_noise_and_power_batch_p1(
        self,
        power_in_w: NDArrayFloat,
        span_length_m: float,
        edfa_config: EDFAConfig | None,
        roadm_config: ROADMConfig | None = None,
        n_effective_spans: int = 1,
    ) -> tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """``nli_model="p1_closed"`` 的**批量**入口（第 4 轮新增）。

        与 :meth:`_calc_span_noise_and_power_p1` 算同一件事，区别只在布局：
        这里一次评估 B 个候选，形状是 ``(B, N)`` 进、四个 ``(B, N)`` 出。

        与 P2 的批量入口**不共用**任何物理函数：P1 的 NLI 是闭式，P2 的是张量化。

        ⚠️ **不是静默回落**：档位不匹配时**抛** ``ValueError``。

        P1 侧的「显式施加 EDFA」与「ROADM 折叠」两步都按**逐候选**做，
        语义与串行入口 :meth:`_calc_span_noise_and_power_p1` **逐项一致**
        （含 P1 档特有的 NF 默认值口径：未指定 NF 时取全 1，不是 P2 档的
        ``self.nf_lin_channels``）。

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

        FU-2：这里过去**漏传** ``raman_spectrum``，子评估器因此总是退回默认档
        ``legacy``。后果是 ``evaluate_path``（唯一真正的端到端路径）在结构上
        **永远拿不到 tabular 谱**——FG-U1 的证据脚本因为直接调用
        ``_calc_span_noise_and_power`` 绕开了 ``for_fiber``，所以一直没暴露。
        超宽带评估必须走 ``evaluate_path``，故本刀补上这一处透传。

        ``legacy`` 不受影响：它本就是默认值，传不传都是同一个值，逐位不变。

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
                self.nli_model,
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
        """``nli_model="p1_closed"`` 的 ROADM 装配（**逐字取自 flexgrid-dev**）。

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
        """``nli_model="p1_closed"`` 用的阻断节点解析（**逐字取自 flexgrid-dev**）。

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
        # WP-Q1 §3.4：功率-宽度归一化口径**显式**生效处。fixed_power（全 1，逐位
        # no-op）与均匀 B_i 下的 fixed_psd（因子恒 1.0）都不改动 legacy 数字。
        power = power * self._power_width_scale()
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
            if self.nli_model == "p1_closed":
                # ⚠️ WP-MERGE-FG-FLEX 第 3 轮：P1 档的 ROADM 装配**逐字取自**
                #    `flexgrid-dev @ cb24704`，与 P2 档（FG-14 的 `fold_roadm_terms`
                #    口径）的差别是**浮点级**的：`x / loss * loss` 与 `x` 在 IEEE-754
                #    下**不**逐位相等，所以两套口径的 SNR 会差最后几位。
                #    `p1_closed` 档既然承诺与 cb24704 逐位相同，就必须走这一支。
                power, total_spm, total_xpm, total_ase = self._apply_roadm_p1(
                    roadm, hop_index == n_hops - 1, power, total_spm, total_xpm, total_ase
                )
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
