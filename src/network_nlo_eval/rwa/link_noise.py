"""链路噪声装配：把一条链路的整跨与余段噪声累加成单个缓存。

这段算术原先只存在于 :meth:`QoTValidator._calc_link_noise` 内。抽出来是为了让
"整跨 × N + 余段" 的累加只有一份定义，调用方只负责取光纤参数、EDFA 与跨段切分。

浮点求和顺序是契约的一部分：从三个全零数组开始累加 ``((0 + s*n) + r)``，改动
写法可能动末位，从而静默改掉 SNR 判定。
"""

from __future__ import annotations

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig
from network_nlo_eval.network.state import LinkNoiseCache

#: 余段短于这个长度（m）时不单独成段。
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
    """把一条链路的整跨/余段噪声累加成单个 :class:`LinkNoiseCache`。

    Args:
        evaluator: 该链路光纤参数对应的评估器（通常是 ``for_fiber`` 的缓存结果）。
        edfa_config: 跨段 EDFA 配置，语义与串行逐跨路径一致。
        power_profile: ``(N,)`` 入纤功率谱 (W)。
        span_length_m: 整跨长度 (m)。
        num_full_spans: 整跨个数；``0`` 表示该链路连一个整跨都放不下。
        remainder_length_m: 余段长度 (m)；``<= 1.0`` 时不成段。
        num_channels: 信道数，仅在零跨段时用来开三个全零数组。

    Returns
    -------
        三个 ``(N,)`` 噪声分量（SPM / XPM / ASE）与物理跨段数。零跨段时三个数组
        各自独立，不共享内存。
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
