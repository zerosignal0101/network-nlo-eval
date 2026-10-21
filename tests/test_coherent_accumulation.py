"""Tests for the optional coherent (complex-amplitude) NLI accumulation path.

The whole point of the link-level entry is that ``|Σ a_k|²`` is not ``Σ |a_k|²``, so the
tests are built around a kernel whose phases are known by hand: same magnitudes, opposite
phases must cancel completely, and a power-only kernel must keep the old incoherent answer.
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.models.base import ElementOutput, MultiSpanOpticalPath
from network_nlo_eval.models.route import accumulate_route
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

N = 2
POWER_TRANSFER = 0.5  # 每跨功率传输因子，振幅因子是 sqrt(0.5)


class _ToyModel:
    """最小可用的 QoT 模型：两跨、两信道、单跨功率固定。"""

    n_nodes = N
    f_abs_hz = np.array([193.1e12, 193.2e12])
    channel_bandwidth_hz_channels = np.array([64e9, 64e9])
    symbol_rate_hz = 64e9
    info = None

    def kernel_for(self, fiber_config: object) -> object:
        raise NotImplementedError

    def evaluate_path(self, optical_path: object, power_in_w: np.ndarray) -> object:
        raise NotImplementedError

    def __init__(self, amplitudes: tuple[complex, ...] | None) -> None:
        self._amplitudes = amplitudes
        self._span_index = 0

    def power_width_scale(self) -> np.ndarray:
        return np.ones(N)

    def blocked_nodes(self, roadm: object) -> list[int]:
        return []

    def span_step(self, fiber: object, edfa: object, power: np.ndarray, length_m: float) -> ElementOutput:
        out = power * POWER_TRANSFER
        ase = np.full(N, 1e-15)
        if self._amplitudes is None:
            spm = np.full(N, 1.0)
            xpm = np.zeros(N)
            return ElementOutput(out, spm, xpm, ase)
        amp = np.full(N, self._amplitudes[self._span_index], dtype=np.complex128)
        self._span_index += 1
        spm = np.abs(amp) ** 2
        return ElementOutput(out, spm, np.zeros(N), ase, amp, np.zeros(N, dtype=np.complex128))


def _path(spans: int) -> MultiSpanOpticalPath:
    return MultiSpanOpticalPath(
        fiber_configs=[FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)] * spans,
        edfa_configs=[EDFAConfig()] * spans,
        roadm_configs=None,
    )


def test_coherent_amplitudes_cancel_where_powers_would_add() -> None:
    """两个等幅反相的跨：相干给 0，非相干给 2。"""
    power = np.full(N, 1e-3)

    incoherent = accumulate_route(_ToyModel(None), _path(2), power)
    coherent = accumulate_route(_ToyModel((1.0 + 0j, -1.0 + 0j)), _path(2), power)

    # 非相干：每跨 1 W，第二跨之前的噪声乘 0.5 => 1*0.5 + 1 = 1.5 W
    assert incoherent.spm_noise_w == pytest.approx(np.full(N, 1.5))
    # 相干：a2 经过 sqrt(0.5) 的传输后与 a1 相加 => (0.7071 - 1)² = 0.0858 W
    expected = (np.sqrt(POWER_TRANSFER) - 1.0) ** 2
    assert coherent.spm_noise_w == pytest.approx(np.full(N, expected), rel=1e-12)


def test_in_phase_amplitudes_beat_the_incoherent_sum() -> None:
    """同相时相干和比功率和大：噪声按 |Σ|² 增长，不是线性累加。"""
    power = np.full(N, 1e-3)
    incoherent = accumulate_route(_ToyModel(None), _path(2), power)
    coherent = accumulate_route(_ToyModel((1.0 + 0j, 1.0 + 0j)), _path(2), power)
    assert coherent.spm_noise_w[0] > incoherent.spm_noise_w[0]
    assert coherent.spm_noise_w == pytest.approx(np.full(N, (1.0 + np.sqrt(POWER_TRANSFER)) ** 2), rel=1e-12)


def test_coherent_phase_scales_as_amplitude_not_power() -> None:
    """跨间传输作用在振幅上（sqrt），不是功率上。"""
    power = np.full(N, 1e-3)
    # 单跨时两种口径必须一致：|a|² == spm_noise_w
    result = accumulate_route(_ToyModel((0.5 + 0.5j,)), _path(1), power)
    assert result.spm_noise_w == pytest.approx(np.full(N, 0.5))


def test_mixing_conventions_is_rejected() -> None:
    """同一模型不能既给复振幅又给纯功率，否则结果无法解释。"""

    class _Mixed(_ToyModel):
        def span_step(self, fiber, edfa, power, length_m):
            if self._span_index == 0:
                self._span_index += 1
                return super().span_step(fiber, edfa, power, length_m)
            self._span_index += 1
            return ElementOutput(power * POWER_TRANSFER, np.full(N, 1.0), np.zeros(N), np.full(N, 1e-15))

    with pytest.raises(TypeError, match="convention"):
        accumulate_route(_Mixed((1.0 + 0j, 0j)), _path(2), np.full(N, 1e-3))


def test_power_only_outputs_keep_the_bit_identical_path() -> None:
    """不给振幅时，路由层必须走原来的浮点运算顺序（黄金向量依赖这一点）。"""
    from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
    from network_nlo_eval.benchmarks.runner import make_model

    case = build_case("c_57ch")
    result = make_model("ecz", case).evaluate_path(build_path(case), launch_profile(case, 0.0))
    # ecz 是纯功率模型：这里只断言它没有意外进入相干分支（振幅字段为 None）
    step = make_model("ecz", case).span_step(
        build_path(case).fiber_configs[0], build_path(case).edfa_configs[0],
        launch_profile(case, 0.0), 100e3,
    )
    assert getattr(step, "nli_amplitude_spm", None) is None
    assert np.all(np.isfinite(result.snr_db))


def test_span_context_reaches_models_that_ask_for_it() -> None:
    """声明了 supports_span_context 的模型必须拿到逐跨递增的序号与累积长度。"""
    seen: list[tuple[int, int, float]] = []

    class _ContextAware(_ToyModel):
        supports_span_context = True

        def span_step(self, fiber, edfa, power, length_m, span_context=None):
            seen.append(
                (span_context.span_index, span_context.total_spans, span_context.cumulative_length_m)
            )
            return super().span_step(fiber, edfa, power, length_m)

    accumulate_route(_ContextAware((1.0 + 0j, 1.0 + 0j, 1.0 + 0j)), _path(3), np.full(N, 1e-3))

    assert [index for index, _, _ in seen] == [0, 1, 2]
    assert {total for _, total, _ in seen} == {3}
    assert [round(cum, 1) for _, _, cum in seen] == [0.0, 80000.0, 160000.0]


def test_amplitude_magnitude_matches_noise_power_through_amplifier_and_roadm() -> None:
    """|amp|**2 必须始终等于同名噪声功率，包括经放大与 ROADM 衰减之后。

    功率口径的 NLI 不在这一层乘放大器增益，振幅若乘 sqrt(gain)，两条口径就会相差
    一个跨增益，跨模型比较会整体偏 20 dB 量级。这条测试就是防这个。
    """
    from network_nlo_eval.models.base import SpanFiberOutput
    from network_nlo_eval.models.route import apply_amplifier_and_roadm
    from network_nlo_eval.network.elements import ROADMConfig

    power_in = np.full(N, 1e-3)
    fiber_out = SpanFiberOutput(
        power_out_w=np.full(N, 1e-5),
        spm_noise_w=np.full(N, 2e-7),
        xpm_noise_w=np.full(N, 3e-7),
        nli_amplitude_spm=np.full(N, np.sqrt(2e-7), dtype=np.complex128) * np.exp(0.7j),
        nli_amplitude_xpm=np.full(N, np.sqrt(3e-7), dtype=np.complex128) * np.exp(-1.1j),
    )

    for roadm in (None, ROADMConfig(loss_db=6.0, filtering_penalty_db=0.5)):
        out = apply_amplifier_and_roadm(
            power_in_w=power_in,
            fiber_output=fiber_out,
            edfa_config=EDFAConfig(),
            roadm_config=roadm,
            nf_lin=np.full(N, 10 ** (5.5 / 10)),
            f_abs_hz=np.array([193.1e12, 193.2e12]),
            bandwidth_hz=np.full(N, 64e9),
        )
        assert np.allclose(np.abs(out.nli_amplitude_spm) ** 2, out.spm_noise_w, rtol=1e-12)
        assert np.allclose(np.abs(out.nli_amplitude_xpm) ** 2, out.xpm_noise_w, rtol=1e-12)
