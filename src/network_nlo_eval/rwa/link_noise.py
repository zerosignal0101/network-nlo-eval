"""链路噪声装配的唯一真源（FG-5）。

**为什么要有这个模块。** `QoTValidator._calc_link_noise` 与 RL 侧
`ScanValidator._calc_link_noise` 曾经各写一份逐字相同的累加逻辑。两份副本在
保真线改物理时**悄悄分叉过**：FU-4 实测两边 SNR 相差 0.2878 dB，而全仓没有一条
测试变红——因为金标准照不到 RL 抄过去的那条路径。重复实现本身就是分叉的成因，
所以本模块把这段累加提成**一个**公开函数，两侧都只做"取值 + 调用"。

两侧保留的差异因此只剩**取值来源**，不含任何算术：

* 物理侧 `QoTValidator._calc_link_noise`：每次从拓扑取光纤参数、EDFA 配置与
  跨段切分，再调用本函数。
* RL 侧 `ScanValidator._calc_link_noise`：用自己缓存的 `_LinkPlan`（避开
  `for_fiber` 的 pydantic 序列化），再调用本函数。

缓存写时序（谁在什么时刻把 `LinkNoiseCache` 写进 `LinkState.noise_cache`）是 RL
侧**保留**的行为，它属于调用方策略而不是噪声装配语义，仍留在 RL 侧，见
``docs/fg5_integration_note.md`` §2 的逐行差异清单。

**浮点求和顺序是本模块的契约。** 这里的写法逐字照抄上游
`QoTValidator._calc_link_noise`（从三个全零数组开始累加 `((0 + s*n) + r)`），
不是"等价写法"。任何改动都可能改变末位，从而静默改掉 SNR 判定。
"""

from __future__ import annotations

import numpy as np
from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig
from network_nlo_eval.network.state import LinkNoiseCache

#: 余段短于这个长度（m）时不单独成段——与上游 `remainder_length_m > 1.0` 同源同值。
_REMAINDER_EPSILON_M = 1.0


def assemble_link_noise(
    *,
    evaluator: MultiBandISRSGN,
    edfa_config: EDFAConfig | None,
    power_profile: NDArrayFloat,
    span_length_m: float,
    num_full_spans: int,
    remainder_length_m: float,
    num_channels: int,
) -> LinkNoiseCache:
    """把一条链路的整跨/余段噪声累加成单个 `LinkNoiseCache`。

    这是物理仓与 RL 仓**共用**的噪声装配函数对象。任何一侧都不得再复制这段算术。

    Args:
        evaluator: 该链路光纤参数对应的 `MultiBandISRSGN`（通常是 `for_fiber` 的
            缓存结果）。本函数**不**缓存它，由调用方决定。
        edfa_config: 跨段 EDFA 配置，语义与串行逐跨路径一致。
        power_profile: ``(N,)`` 入纤功率谱 (W)。
        span_length_m: 整跨长度 (m)。
        num_full_spans: 整跨个数；``0`` 表示该链路连一个整跨都放不下。
        remainder_length_m: 余段长度 (m)；``<= 1.0`` 时不成段（见
            :data:`_REMAINDER_EPSILON_M`）。
        num_channels: 信道数，仅在"零整跨且无成段余段"时用来开三个全零数组。

    Returns:
        LinkNoiseCache: 三个 ``(N,)`` 噪声分量（SPM / XPM / ASE）与物理跨段数。
        零跨段情形返回三个独立的全零数组（各自一份副本，不共享内存）。
    """
    total_spm = np.zeros(num_channels, dtype=np.float64)
    total_xpm = np.zeros(num_channels, dtype=np.float64)
    total_ase = np.zeros(num_channels, dtype=np.float64)
    total_spans = 0

    if num_full_spans > 0:
        _, s_spm, s_xpm, s_ase = evaluator._calc_span_noise_and_power(
            power_profile,
            span_length_m,
            edfa_config,
            None,
            n_effective_spans=1,
        )
        total_spm += s_spm * num_full_spans
        total_xpm += s_xpm * num_full_spans
        total_ase += s_ase * num_full_spans
        total_spans += num_full_spans

    if remainder_length_m > _REMAINDER_EPSILON_M:
        _, r_spm, r_xpm, r_ase = evaluator._calc_span_noise_and_power(
            power_profile,
            remainder_length_m,
            edfa_config,
            n_effective_spans=1,
        )
        total_spm += r_spm
        total_xpm += r_xpm
        total_ase += r_ase
        total_spans += 1

    return LinkNoiseCache(spm=total_spm, xpm=total_xpm, ase=total_ase, span_count=total_spans)