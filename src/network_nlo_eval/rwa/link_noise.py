"""链路噪声装配：把一条链路的整跨与余段噪声累加成单个缓存。

这段算术原先只存在于 :meth:`QoTValidator._calc_link_noise` 内。抽出来是为了让
"整跨 × N + 余段" 的累加只有一份定义，调用方只负责取光纤参数、EDFA 与跨段切分。

浮点求和顺序是契约的一部分：从三个全零数组开始累加 ``((0 + s*n) + r)``，改动
写法可能动末位，从而静默改掉 SNR 判定。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from network_nlo_eval.core.types import NDArrayFloat
from network_nlo_eval.models.base import SpanContext
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.network.elements import EDFAConfig
from network_nlo_eval.network.state import LinkNoiseCache

if TYPE_CHECKING:
    from network_nlo_eval.rwa.validity import LinkValidity

#: 余段短于这个长度（m）时不单独成段。
_REMAINDER_EPSILON_M = 1.0


def _emits_nli_amplitudes(evaluator: object) -> bool:
    """评估器是否给出跨间相干的 NLI 复振幅。

    ``supports_span_context`` 是模型侧已有的能力声明：只有声明了的模型才会在
    ``span_step`` 里接收 :class:`~network_nlo_eval.models.base.SpanContext`，并把
    ``|a_k|^2`` 等于 ``*_noise_w`` 的复振幅放进 ``ElementOutput``。没有该属性
    的评估器（如冻结基线 ``MultiBandISRSGN``）保持纯功率口径。
    """
    return bool(getattr(evaluator, "supports_span_context", False))


def _assemble_coherent_link_noise(
    *,
    evaluator: MultiBandISRSGN,
    edfa_config: EDFAConfig | None,
    power_profile: NDArrayFloat,
    span_length_m: float,
    num_full_spans: int,
    remainder_length_m: float,
    num_channels: int,
    convention: str,
    validity: LinkValidity | None,
) -> LinkNoiseCache:
    """按跨展开链路，用 ``|sum a_k|^2`` 做相干装配。

    递推顺序与 :func:`network_nlo_eval.models.route.accumulate_route` 的相干
    档逐行对应：先 ``amp *= sqrt(transfer)`` 把已累积振幅送到本跨入口，再加本
    跨振幅；ASE 始终按功率递推。拆出的 additive FWM 项按功率累加，最后并回
    ``xpm``，与路由层语义一致。

    只有 ``supports_span_context=True`` 的评估器会走到这里；第一跨若仍返回纯
    功率（例如模型只在 ``total_spans > 1`` 时才给振幅），本函数与路由层一样
    退回功率累加，不会把两种口径混在一条链路上。

    Raises
    ------
        TypeError: 同一链路上前半段给复振幅、后半段只给功率（或相反）。与
            ``accumulate_route`` 相同：口径必须二选一。
    """
    total_ase = np.zeros(num_channels, dtype=np.float64)
    total_spm_power = np.zeros(num_channels, dtype=np.float64)
    total_xpm_power = np.zeros(num_channels, dtype=np.float64)
    total_fwm = np.zeros(num_channels, dtype=np.float64)
    amp_spm = np.zeros(num_channels, dtype=np.complex128)
    amp_xpm = np.zeros(num_channels, dtype=np.complex128)
    #: ``None`` 表示链路还没有跨；之后锁定为 True（复振幅）或 False（纯功率）。
    coherent: bool | None = None
    fwm_present = False
    fwm_is_additive = bool(getattr(evaluator, "fwm_noise_is_additive", False))

    span_lengths = [span_length_m] * num_full_spans
    if remainder_length_m > _REMAINDER_EPSILON_M:
        span_lengths.append(remainder_length_m)
    total_spans = len(span_lengths)

    power = np.array(power_profile, dtype=np.float64, copy=True)
    cumulative_m = 0.0
    for span_index, length_m in enumerate(span_lengths):
        previous_power = power
        step = evaluator._calc_span_output_with_context(
            power,
            length_m,
            edfa_config,
            SpanContext(
                span_index=span_index,
                total_spans=total_spans,
                cumulative_length_m=cumulative_m,
                span_length_m=length_m,
            ),
        )
        power = step.power_out_w
        transfer = np.divide(power, previous_power, out=np.zeros_like(power), where=previous_power > 0)

        step_amp = step.nli_amplitude_spm
        if step_amp is None:
            if coherent is True:
                raise TypeError(
                    "evaluator returned a coherent NLI amplitude for an earlier span and a "
                    "power-only result for a later one; a model must pick one convention."
                )
            coherent = False
            total_spm_power *= transfer
            total_xpm_power *= transfer
            total_spm_power += step.spm_noise_w
            total_xpm_power += step.xpm_noise_w
            total_ase *= transfer
            total_ase += step.ase_noise_w
            if fwm_is_additive and fwm_present:
                total_fwm *= transfer
            step_fwm = getattr(step, "fwm_noise_w", None) if fwm_is_additive else None
            if step_fwm is not None:
                total_fwm += step_fwm
                fwm_present = True
        else:
            if coherent is False:
                raise TypeError(
                    "evaluator returned a power-only NLI result for an earlier span and a "
                    "coherent amplitude for a later one; a model must pick one convention."
                )
            coherent = True
            amp_root = np.sqrt(transfer)
            amp_spm = amp_spm * amp_root + step_amp
            amp_xpm = amp_xpm * amp_root + step.nli_amplitude_xpm
            total_ase *= transfer
            total_ase += step.ase_noise_w
            if fwm_is_additive:
                if fwm_present:
                    total_fwm *= transfer
                step_fwm = getattr(step, "fwm_noise_w", None)
                if step_fwm is not None:
                    total_fwm += step_fwm
                    fwm_present = True

        cumulative_m += length_m

    if coherent:
        total_spm = np.abs(amp_spm) ** 2
        total_xpm = np.abs(amp_xpm) ** 2
    else:
        total_spm = total_spm_power
        total_xpm = total_xpm_power
    if fwm_present:
        total_xpm = total_xpm + total_fwm

    return LinkNoiseCache(
        spm=total_spm,
        xpm=total_xpm,
        ase=total_ase,
        span_count=total_spans,
        validity=validity,
        launch_power_convention=convention,
    )



def assemble_link_noise(
    *,
    evaluator: MultiBandISRSGN,
    edfa_config: EDFAConfig | None,
    power_profile: NDArrayFloat,
    span_length_m: float,
    num_full_spans: int,
    remainder_length_m: float,
    num_channels: int,
    validity: LinkValidity | None = None,
    launch_power_convention: str | None = None,
    coherent: bool = False,
) -> LinkNoiseCache:
    """把一条链路的整跨/余段噪声累加成单个 :class:`LinkNoiseCache`。

    Args:
        evaluator: 该链路光纤参数对应的评估器（通常是 ``for_fiber`` 的缓存结果）。
        edfa_config: 跨段 EDFA 配置，语义与串行逐跨路径一致。
        power_profile: ``(N,)`` 入纤功率谱 (W)，口径为 ``evaluator`` 的
            ``launch_power_convention``。本函数只做累加，不跨口径换算。
        span_length_m: 整跨长度 (m)。
        num_full_spans: 整跨个数；``0`` 表示该链路连一个整跨都放不下。
        remainder_length_m: 余段长度 (m)；``<= 1.0`` 时不成段。
        num_channels: 信道数，仅在零跨段时用来开三个全零数组。
        validity: 可选的有效性报告。只做**附加**，不参与噪声算术：三个噪声数组
            的求和顺序与结果与未接入时逐位相同。
        launch_power_convention: 写入缓存的功率口径标签。``None``（默认）时取
            评估器的 ``launch_power_convention``；网络层调用方应显式传入验证器
            的评估器口径，避免光纤子评估器丢失声明。该标签只写入元数据，
            不改变任何数值。
        coherent: 跨间相干装配开关，默认 ``False``（逐位保持历史行为）。
            ``True`` 时把链路按跨展开，逐跨调用评估器的
            ``_calc_span_output_with_context`` 并把 NLI 复振幅按
            ``|sum a_k|^2`` 相加（ASE 仍按功率递推），与
            :meth:`~network_nlo_eval.models.route.accumulate_route` 的相干档
            同源。只有声明了 ``supports_span_context`` 的评估器会生效；其余
            评估器（包括有活跃泵、``span_coherence="incoherent"`` 的
            ``pcfm_h3_raman`` 与冻结基线 ``ecz``）保持原来的"整跨 × N"功率
            路径，结果与 ``coherent=False`` 逐位相同。

    Returns
    -------
        三个 ``(N,)`` 噪声分量（SPM / XPM / ASE）与物理跨段数。三个数组都处于
        评估器口径（``total`` 或 ``per_pol``），默认 ``total`` 时是物理总功率。
        ``launch_power_convention`` 记录该口径，供读取方判断能否复用。零跨段时
        三个数组各自独立，不共享内存。
    """
    convention = launch_power_convention
    if convention is None:
        convention = str(getattr(evaluator, "launch_power_convention", "total"))

    if coherent and _emits_nli_amplitudes(evaluator):
        return _assemble_coherent_link_noise(
            evaluator=evaluator,
            edfa_config=edfa_config,
            power_profile=power_profile,
            span_length_m=span_length_m,
            num_full_spans=num_full_spans,
            remainder_length_m=remainder_length_m,
            num_channels=num_channels,
            convention=convention,
            validity=validity,
        )

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

    return LinkNoiseCache(
        spm=total_spm,
        xpm=total_xpm,
        ase=total_ase,
        span_count=total_spans,
        validity=validity,
        launch_power_convention=convention,
    )
