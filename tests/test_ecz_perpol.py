"""``ECZCoherentModel`` 的原生 ``per_pol`` 口径（第十轮任务 AE）。

背景
----
第九轮（任务 AB）把功率口径换算收进了 :class:`MultiBandISRSGN`，但
:class:`~network_nlo_eval.models.ecz.ECZCoherentModel` 的构造签名与
:meth:`ECZCoherentModel.for_fiber` 没有透传 ``launch_power_convention``，ECZ
因此只能走 ``total``（或旧类属性替身）。本任务把口径接到 ECZ 上。

坐标边界
--------
* ``total``（默认）：入纤功率、三项噪声与 NLI 复振幅都是每信道双偏振总功率。
  历史路径不做任何乘法，逐位不变。
* ``per_pol``：入纤功率、三项噪声与复振幅都是每偏振功率。基类单跨核在边界做
  ``×2`` 驱动 / ``×0.5`` 回每偏振；新产生的 EDFA/booster ASE 乘
  ``ase_polarization_scale=0.5``。

同一物理链路下，``total`` 与 ``per_pol`` 的信号/NLI/ASE 应严格差 2 倍、
SNR 相同。单跨与网络 QoT（逐跨**非相干**求和）可以逐位钉住；多跨
``evaluate_path`` 走相干累加，复振幅取 ``sqrt(spm_noise_w)``，
``sqrt(x / 2)`` 与 ``sqrt(x) / sqrt(2)`` 并不逐位相同（差约 1 ulp），
所以那里用 1e-14 的相对容差，SNR 用 1e-12 dB。

跨间相干量（可见度、相干因子、相位斜坡）只依赖色散/衰减/跨长/带宽，与功率
坐标无关，两种坐标共用同一份缓存——这也是坐标不会串线的根据之一。
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, SpanContext
from network_nlo_eval.models.ecz import ECZCoherentModel
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import ServiceRequest

CHANNEL_SPACING_HZ = 50e9
TOTAL_POWER_W = 1e-3
PER_POL_POWER_W = TOTAL_POWER_W / 2
SPAN_LENGTH_M = 80e3
EDFA = EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)
#: 多跨相干 NLI 的容差：复振幅开方后两种坐标只差 1 ulp 量级。
COHERENT_RTOL = 1e-14
#: 多跨相干 SNR 的容差 (dB)。
SNR_ATOL_DB = 1e-12


def _grid(num_channels: int = 8) -> SpectrumGrid:
    return SpectrumGrid(num_channels, CHANNEL_SPACING_HZ, 193.1e12)


def _fiber(length_km: float = 80.0) -> FiberSpanConfig:
    return FiberSpanConfig(length_km=length_km, max_span_length_km=length_km)


def _model(convention: str | None = None, *, num_channels: int = 8, ode_accel: str = "off") -> ECZCoherentModel:
    kwargs = {} if convention is None else {"launch_power_convention": convention}
    return ECZCoherentModel(_grid(num_channels), _fiber(), 32e9, ode_accel=ode_accel, **kwargs)


def _path(spans: int = 1, *, roadm: ROADMConfig | None = None) -> MultiSpanOpticalPath:
    roadms = [roadm] * spans if roadm is not None else None
    return MultiSpanOpticalPath([_fiber()] * spans, [EDFA] * spans, roadms)


def _bits(array: np.ndarray) -> np.ndarray:
    return np.asarray(array, dtype=np.float64).view(np.int64)


def _assert_bitwise_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    assert np.array_equal(_bits(left), _bits(right)), f"{label} 不逐位相同"


def _assert_ratio_two(actual: np.ndarray, half: np.ndarray, label: str, rtol: float = 0.0) -> None:
    """断言 ``actual`` 是 ``half`` 的 2 倍；``rtol=0`` 时逐位。"""
    doubled = 2.0 * half
    if rtol == 0.0:
        _assert_bitwise_equal(actual, doubled, label)
        return
    scale = np.maximum(np.abs(doubled), 1e-30)
    assert np.allclose(actual, doubled, rtol=rtol, atol=0.0), (
        f"{label} 不是严格 2 倍：最大相对偏差 {float(np.max(np.abs(actual - doubled) / scale)):.3e}"
    )


# ---------------------------------------------------------------------------
# total 默认逐位不变
# ---------------------------------------------------------------------------


def test_default_constructor_declares_total_coordinates() -> None:
    """不传参数时保持历史口径：total、ASE scale 1、核不做 ×2。"""
    model = _model()
    assert model.launch_power_convention == "total"
    assert model.ase_polarization_scale == 1.0
    assert model.physical_power_scale == 1.0


def test_default_and_explicit_total_are_bit_identical() -> None:
    """默认构造与显式 ``total`` 在核、单跨与多跨路径上逐位相同。"""
    power = np.full(8, TOTAL_POWER_W)
    roadm = ROADMConfig(equalize_output_power=True)
    default = _model()
    explicit = _model("total")

    default_fiber = default.span_fiber_output(power, SPAN_LENGTH_M, 1)
    explicit_fiber = explicit.span_fiber_output(power, SPAN_LENGTH_M, 1)
    for name in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        _assert_bitwise_equal(getattr(default_fiber, name), getattr(explicit_fiber, name), f"fiber.{name}")

    default_span = default._calc_span_noise_and_power(power, SPAN_LENGTH_M, EDFA, roadm)
    explicit_span = explicit._calc_span_noise_and_power(power, SPAN_LENGTH_M, EDFA, roadm)
    for index, (a, b) in enumerate(zip(default_span, explicit_span, strict=True)):
        _assert_bitwise_equal(a, b, f"span[{index}]")

    for spans in (1, 4):
        path = _path(spans, roadm=roadm)
        default_result = default.evaluate_path(path, power)
        explicit_result = explicit.evaluate_path(path, power)
        for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
            _assert_bitwise_equal(
                getattr(default_result, name),
                getattr(explicit_result, name),
                f"evaluate_path[{spans}].{name}",
            )


# ---------------------------------------------------------------------------
# 原生 per_pol：单跨与核边界严格 2 倍
# ---------------------------------------------------------------------------


def test_native_per_pol_attributes() -> None:
    """显式 per_pol 由模型自己换算：ASE 减半、物理驱动功率 ×2。"""
    model = _model("per_pol")
    assert model.launch_power_convention == "per_pol"
    assert model.ase_polarization_scale == 0.5
    assert model.physical_power_scale == 2.0


def test_single_span_per_pol_is_exactly_two_times_total() -> None:
    """单跨：核与 ``evaluate_path`` 的信号/NLI/ASE 严格 2 倍，SNR 逐位相同。

    覆盖无 EDFA、固定增益、补偿增益与 ROADM（含 booster ASE）四种边界，确认
    ``span_step`` 里的 EDFA ASE 与 ``_calc_span_noise_and_power`` 同坐标。
    """
    roadm = ROADMConfig(insertion_loss_db=12.0, add_drop_loss_db=1.0, equalize_output_power=True)
    cases = (
        (None, None),
        (EDFAConfig(noise_figure_db=5.5), None),
        (EDFAConfig(noise_figure_db=5.5), roadm),
    )
    total = _model("total")
    per_pol = _model("per_pol")
    total_power = np.full(8, TOTAL_POWER_W)
    per_pol_power = np.full(8, PER_POL_POWER_W)

    for edfa, roadm_config in cases:
        label = f"edfa={edfa!r},roadm={roadm_config!r}"
        total_span = total._calc_span_noise_and_power(total_power, SPAN_LENGTH_M, edfa, roadm_config)
        per_span = per_pol._calc_span_noise_and_power(per_pol_power, SPAN_LENGTH_M, edfa, roadm_config)
        for index, (t_value, p_value) in enumerate(zip(total_span, per_span, strict=True)):
            _assert_ratio_two(t_value, p_value, f"kernel[{index}] ({label})")

        path = MultiSpanOpticalPath([_fiber()], [edfa] if edfa is not None else [None], [roadm_config])
        total_result = total.evaluate_path(path, total_power)
        per_result = per_pol.evaluate_path(path, per_pol_power)
        for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"):
            field_label = f"evaluate_path.{name} ({label})"
            _assert_ratio_two(getattr(total_result, name), getattr(per_result, name), field_label)
        _assert_bitwise_equal(total_result.snr_db, per_result.snr_db, f"evaluate_path.snr_db ({label})")


# ---------------------------------------------------------------------------
# 多跨相干路径：复振幅必须与噪声同坐标
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spans", [2, 4, 10])
def test_multi_span_coherent_nli_scales_by_two(spans: int) -> None:
    """多跨相干累加：信号/ASE 逐位 2 倍，NLI 与 SNR 在 ulp 级等价。

    复振幅取 ``sqrt(spm_noise_w)``；``sqrt(x / 2)`` 与 ``sqrt(x) / sqrt(2)``
    在二进制浮点下只差 1 ulp，所以相干和只能给到 1e-14 的相对容差。若坐标
    串线（例如少减半一次），比值会偏离 2 达 4 倍或 2 倍，远超该容差。
    """
    total = _model("total")
    per_pol = _model("per_pol")
    path = _path(spans)
    total_result = total.evaluate_path(path, np.full(8, TOTAL_POWER_W))
    per_result = per_pol.evaluate_path(path, np.full(8, PER_POL_POWER_W))

    _assert_ratio_two(total_result.signal_power_w, per_result.signal_power_w, "signal")
    _assert_ratio_two(total_result.ase_noise_w, per_result.ase_noise_w, "ase")
    _assert_ratio_two(total_result.spm_noise_w, per_result.spm_noise_w, "spm", rtol=COHERENT_RTOL)
    _assert_ratio_two(total_result.xpm_noise_w, per_result.xpm_noise_w, "xpm", rtol=COHERENT_RTOL)
    delta = float(np.max(np.abs(total_result.snr_db - per_result.snr_db)))
    assert delta <= SNR_ATOL_DB, f"多跨相干 SNR 偏差 {delta:.3e} dB 超过 {SNR_ATOL_DB:g}"


def test_multi_span_amplitude_magnitude_matches_same_coordinate_noise() -> None:
    """``per_pol`` 的复振幅模方必须等于**同一坐标**下的噪声功率。

    同时与 total 的振幅比较：``|a_per_pol|**2`` 是 ``|a_total|**2`` 的一半。
    若振幅仍按总功率坐标构造，``|sum a_k|^2`` 会整体多一倍。
    """
    total = _model("total").for_fiber(_fiber())
    per_pol = _model("per_pol").for_fiber(_fiber())
    context = SpanContext(span_index=1, total_spans=4, cumulative_length_m=SPAN_LENGTH_M, span_length_m=SPAN_LENGTH_M)

    total_output = total.span_fiber_output(np.full(8, TOTAL_POWER_W), SPAN_LENGTH_M, 1, context)
    per_output = per_pol.span_fiber_output(np.full(8, PER_POL_POWER_W), SPAN_LENGTH_M, 1, context)
    assert per_output.nli_amplitude_spm is not None and per_output.nli_amplitude_xpm is not None

    for amplitude, noise, label in (
        (per_output.nli_amplitude_spm, per_output.spm_noise_w, "spm"),
        (per_output.nli_amplitude_xpm, per_output.xpm_noise_w, "xpm"),
    ):
        assert np.allclose(np.abs(amplitude) ** 2, noise, rtol=1e-12, atol=0.0), f"{label}: |a|^2 与噪声不同坐标"

    total_power_amplitude = np.abs(total_output.nli_amplitude_spm) ** 2
    per_power_amplitude = np.abs(per_output.nli_amplitude_spm) ** 2
    _assert_ratio_two(total_power_amplitude, per_power_amplitude, "spm amplitude", rtol=COHERENT_RTOL)


# ---------------------------------------------------------------------------
# for_fiber 与网络 QoT 的口径传播
# ---------------------------------------------------------------------------


def test_for_fiber_keeps_the_power_convention() -> None:
    """光纤子评估器必须继承口径与 ASE 缩放，否则核会把每偏振功率当总功率。"""
    per_pol = _model("per_pol")
    child = per_pol.for_fiber(_fiber(100.0))
    assert child is per_pol.for_fiber(_fiber(100.0)), "子评估器应按光纤参数缓存"
    assert child.launch_power_convention == "per_pol"
    assert child.ase_polarization_scale == 0.5
    assert child.physical_power_scale == 2.0

    default = _model()
    default_child = default.for_fiber(_fiber(100.0))
    assert default_child.launch_power_convention == "total"
    assert default_child.ase_polarization_scale == 1.0
    assert default_child.physical_power_scale == 1.0


def _network(
    convention: str, *, equalize: bool
) -> tuple[NetworkTopology, SpectrumGrid, ECZCoherentModel, NetworkState]:
    """两节点单链路的最小 RWA 栈，与 ``test_isrs_gn_perpol`` 同构。"""
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, weight=80.0)
    fiber = _fiber()
    topology = NetworkTopology(
        graph, fiber, EDFAConfig(noise_figure_db=5.0), ROADMConfig(equalize_output_power=equalize)
    )
    grid = _grid(4)
    evaluator = ECZCoherentModel(grid, fiber, 32e9, launch_power_convention=convention)
    return topology, grid, evaluator, NetworkState(topology, grid)


def _request(launch_power_w: float) -> ServiceRequest:
    """单信道业务请求；0 dB 阈值让基线必然通过。"""
    return ServiceRequest(
        service_id=1,
        source_id=0,
        destination_id=1,
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=0.0,
        launch_power_w=launch_power_w,
    )


@pytest.mark.parametrize("equalize", [False, True])
def test_network_qot_per_pol_matches_total(equalize: bool) -> None:
    """网络 QoT（逐跨非相干求和）在两种坐标下 SNR 逐位相同、噪声缓存严格 2 倍。"""
    total_topo, total_grid, total_evaluator, total_state = _network("total", equalize=equalize)
    per_topo, per_grid, per_evaluator, per_state = _network("per_pol", equalize=equalize)
    total_validator = QoTValidator(total_evaluator, total_topo, total_grid, fixed_span_length_km=80.0)
    per_validator = QoTValidator(
        per_evaluator,
        per_topo,
        per_grid,
        fixed_span_length_km=80.0,
        launch_power_convention="per_pol",
    )
    assert per_validator.eval_power_convention == "per_pol"
    assert per_validator.ase_polarization_scale == 0.5

    ok_total, snr_total = total_validator.verify_allocation(_request(TOTAL_POWER_W), [0, 1], 0, total_state)
    ok_per, snr_per = per_validator.verify_allocation(_request(PER_POL_POWER_W), [0, 1], 0, per_state)
    assert ok_total is ok_per is True
    assert snr_total is not None and snr_per is not None
    _assert_bitwise_equal(snr_total, snr_per, f"network.snr (equalize={equalize})")

    total_cache = total_state.get_link_state(0, 1).noise_cache
    per_cache = per_state.get_link_state(0, 1).noise_cache
    assert total_cache is not None and per_cache is not None
    assert total_cache.launch_power_convention == "total"
    assert per_cache.launch_power_convention == "per_pol"
    for name in ("spm", "xpm", "ase"):
        _assert_ratio_two(getattr(total_cache, name), getattr(per_cache, name), f"cache.{name}")


def test_multi_span_path_does_not_mix_single_span_or_network_coordinates() -> None:
    """单跨、多跨相干与网络非相干三条路径各自守在自己的坐标里。

    以 ``per_pol`` 为准：

    * 单跨 ``evaluate_path`` 的 SPM 与 ``span_fiber_output`` 逐位相同（同一坐标）；
    * 多跨相干与单跨的 SPM 比值在两种坐标下一致（相干因子与功率坐标无关）；
    * 多跨相干 SPM 与各自 total 对应量严格差 2 倍——若复振幅误用 total 坐标，
      这里会变成 4 倍；
    * 网络 QoT（``assemble_link_noise``，逐跨非相干求和）等于跨数 × 单跨，
      明显低于相干值，说明网络路径没有借用相干振幅。
    """
    fiber = _fiber()
    per_pol = _model("per_pol")
    total = _model("total")
    power_per = np.full(8, PER_POL_POWER_W)
    power_total = np.full(8, TOTAL_POWER_W)

    single_per = per_pol.evaluate_path(_path(1), power_per)
    single_total = total.evaluate_path(_path(1), power_total)
    fiber_per = per_pol.for_fiber(fiber).span_fiber_output(power_per, SPAN_LENGTH_M, 1)
    _assert_bitwise_equal(single_per.spm_noise_w, fiber_per.spm_noise_w, "单跨与 span_fiber_output")

    spans = 4
    multi_per = per_pol.evaluate_path(_path(spans), power_per)
    multi_total = total.evaluate_path(_path(spans), power_total)
    ratio_per = np.asarray(multi_per.spm_noise_w) / np.asarray(single_per.spm_noise_w)
    ratio_total = np.asarray(multi_total.spm_noise_w) / np.asarray(single_total.spm_noise_w)
    assert np.allclose(ratio_per, ratio_total, rtol=1e-12, atol=0.0), "两种坐标的相干因子不一致"
    _assert_ratio_two(multi_total.spm_noise_w, multi_per.spm_noise_w, "多跨 total/per_pol SPM", rtol=COHERENT_RTOL)
    _assert_ratio_two(multi_total.xpm_noise_w, multi_per.xpm_noise_w, "多跨 total/per_pol XPM", rtol=COHERENT_RTOL)

    # 网络 QoT 走逐跨非相干求和：缓存 = 跨数 × 单跨，且低于相干累加。
    per_cache = assemble_link_noise(
        evaluator=per_pol.for_fiber(fiber),
        edfa_config=EDFA,
        power_profile=power_per,
        span_length_m=SPAN_LENGTH_M,
        num_full_spans=spans,
        remainder_length_m=0.0,
        num_channels=8,
    )
    total_cache = assemble_link_noise(
        evaluator=total.for_fiber(fiber),
        edfa_config=EDFA,
        power_profile=power_total,
        span_length_m=SPAN_LENGTH_M,
        num_full_spans=spans,
        remainder_length_m=0.0,
        num_channels=8,
    )
    _assert_bitwise_equal(per_cache.spm, np.asarray(fiber_per.spm_noise_w) * spans, "网络缓存 = 跨数 × 单跨")
    _assert_ratio_two(total_cache.spm, per_cache.spm, "网络缓存 total/per_pol SPM")
    assert np.all(np.asarray(multi_per.spm_noise_w) > np.asarray(per_cache.spm)), "相干路径借用了非相干缓存"


# ---------------------------------------------------------------------------
# 非法口径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL", "auto", "per-pol"])
def test_invalid_convention_is_rejected(convention: str) -> None:
    """口径只接受 ``total`` / ``per_pol``，非法值在构造期直接报错。"""
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        ECZCoherentModel(_grid(), _fiber(), 50e9, launch_power_convention=convention)
