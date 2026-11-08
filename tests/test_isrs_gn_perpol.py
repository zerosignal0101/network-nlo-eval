"""``MultiBandISRSGN`` 的原生 ``per_pol`` 口径（第九轮任务 AB）。

背景
----
第七轮（任务 X）打通了网络级 per_pol evaluator，但裸 ``MultiBandISRSGN`` 仍把
``power_in_w`` 当作双偏振总功率。per_pol 网络路径必须靠外部替身
（``tests/test_qot_network_perpol.py`` 的 ``_PerPolISRSGN``）在核边界手工做
``×2 → 核 → ×0.5``。本任务把该换算收进模型本身。

约定
----
* ``launch_power_convention="total"``（基类默认）：``power_in_w`` 与全部输出都是
  每信道双偏振总功率，历史路径不做任何乘法，逐位不变。
* ``launch_power_convention="per_pol"``：``power_in_w`` 是每偏振功率。SRS/NLI
  核按物理总功率 ``2 * P`` 驱动，出纤信号与 SPM/XPM 乘 ``0.5`` 回到每偏振
  坐标；EDFA/ROADM 增益用同一坐标的入纤功率与 fiber_output 相除（比值与 total
  坐标相同），只有新产生的 EDFA/booster ASE 按 ``ase_polarization_scale=0.5``
  减半。

同一物理链路下，``total`` 的每个数组都严格等于 ``per_pol`` 的 2 倍，SNR
逐位相同。2.0/0.5 都是二进制精确因子，因此测试用 ``np.array_equal`` 钉住
**逐位**关系，而不是近似。
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import ServiceRequest

CHANNEL_SPACING_HZ = 50e9
TOTAL_POWER_W = 1e-3
PER_POL_POWER_W = TOTAL_POWER_W / 2
EDFA = EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)


def _grid(num_channels: int = 8) -> SpectrumGrid:
    return SpectrumGrid(num_channels, CHANNEL_SPACING_HZ, 193.1e12)


def _fiber(length_km: float = 80.0) -> FiberSpanConfig:
    return FiberSpanConfig(length_km=length_km, max_span_length_km=80.0)


def _evaluator(
    convention: str | None = None,
    *,
    num_channels: int = 8,
    ode_accel: str = "off",
    fiber: FiberSpanConfig | None = None,
) -> MultiBandISRSGN:
    kwargs = {} if convention is None else {"launch_power_convention": convention}
    return MultiBandISRSGN(
        _grid(num_channels),
        fiber or _fiber(),
        32e9,
        ode_accel=ode_accel,
        **kwargs,
    )


def _result_arrays(result) -> dict[str, np.ndarray]:
    return {
        "signal_power_w": result.signal_power_w,
        "spm_noise_w": result.spm_noise_w,
        "xpm_noise_w": result.xpm_noise_w,
        "ase_noise_w": result.ase_noise_w,
        "snr_db": result.snr_db,
    }


def _assert_bitwise_equal(left: np.ndarray, right: np.ndarray, label: str) -> None:
    assert np.array_equal(left.view(np.int64), right.view(np.int64)), f"{label} 不逐位相同"


# ---------------------------------------------------------------------------
# total 默认逐位不变
# ---------------------------------------------------------------------------


def test_default_constructor_declares_total_coordinates() -> None:
    """不传参数时与历史一致：total、ASE scale 1、核不做 ×2。"""
    evaluator = _evaluator()
    assert evaluator.launch_power_convention == "total"
    assert evaluator.ase_polarization_scale == 1.0
    assert evaluator.physical_power_scale == 1.0


def test_default_and_explicit_total_are_bit_identical_pairwise() -> None:
    """逐对求和路径：默认构造与显式 ``total`` 的核、光纤与路径结果逐位相同。"""
    power = np.full(8, TOTAL_POWER_W)
    roadm = ROADMConfig(equalize_output_power=True)
    path = MultiSpanOpticalPath([_fiber()], [EDFA], roadm_configs=[roadm])
    default = _evaluator()
    explicit = _evaluator("total")

    default_fiber = default.span_fiber_output(power, 80e3, 1)
    explicit_fiber = explicit.span_fiber_output(power, 80e3, 1)
    for name in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        _assert_bitwise_equal(
            getattr(default_fiber, name),
            getattr(explicit_fiber, name),
            f"span_fiber_output.{name}",
        )

    default_span = default._calc_span_noise_and_power(power, 80e3, EDFA, roadm)
    explicit_span = explicit._calc_span_noise_and_power(power, 80e3, EDFA, roadm)
    for index, (a, b) in enumerate(zip(default_span, explicit_span, strict=True)):
        _assert_bitwise_equal(a, b, f"_calc_span_noise_and_power[{index}]")

    default_result = _result_arrays(default.evaluate_path(path, power))
    explicit_result = _result_arrays(explicit.evaluate_path(path, power))
    for name, a in default_result.items():
        _assert_bitwise_equal(a, explicit_result[name], f"evaluate_path.{name}")


def test_default_and_explicit_total_are_bit_identical_matvec() -> None:
    """Matvec 路径（≥40 活跃信道）同样逐位不变，批量入口也一致。"""
    power = np.full(48, TOTAL_POWER_W)
    roadm = ROADMConfig(equalize_output_power=True)
    path = MultiSpanOpticalPath([_fiber()], [EDFA], roadm_configs=[roadm])
    default = _evaluator("total", num_channels=48, ode_accel="matvec")
    default_class = _evaluator(None, num_channels=48, ode_accel="matvec")
    batch_input = np.vstack([power, power * 2.0, np.zeros(48)])

    default_span = default._calc_span_noise_and_power(power, 80e3, EDFA, roadm)
    baseline_span = default_class._calc_span_noise_and_power(power, 80e3, EDFA, roadm)
    assert default._p1_xpm_matrices is not None
    for index, (a, b) in enumerate(zip(default_span, baseline_span, strict=True)):
        _assert_bitwise_equal(a, b, f"matvec._calc_span_noise_and_power[{index}]")

    default_batch = default.calc_span_noise_and_power_batch(batch_input, 80e3, EDFA, roadm)
    baseline_batch = default_class.calc_span_noise_and_power_batch(batch_input, 80e3, EDFA, roadm)
    for index, (a, b) in enumerate(zip(default_batch, baseline_batch, strict=True)):
        _assert_bitwise_equal(a, b, f"batch[{index}]")

    default_result = _result_arrays(default.evaluate_path(path, power))
    baseline_result = _result_arrays(default_class.evaluate_path(path, power))
    for name, a in default_result.items():
        _assert_bitwise_equal(a, baseline_result[name], f"matvec.evaluate_path.{name}")


# ---------------------------------------------------------------------------
# 原生 per_pol：与 total 严格 2 倍
# ---------------------------------------------------------------------------


def test_native_per_pol_attributes() -> None:
    """显式 per_pol 由模型自己换算：ASE 减半、物理驱动功率 ×2。"""
    evaluator = _evaluator("per_pol")
    assert evaluator.launch_power_convention == "per_pol"
    assert evaluator.ase_polarization_scale == 0.5
    assert evaluator.physical_power_scale == 2.0


def test_native_per_pol_matches_total_pairwise_kernel() -> None:
    """逐对求和路径：total 的每个分量严格等于 per_pol 的 2 倍。

    覆盖无 EDFA、固定增益、补偿损耗（``target_gain_db=None``）与 ROADM 损耗，
    确保增益用的是同一坐标的功率比、ASE 只减半一次。
    """
    roadm = ROADMConfig(insertion_loss_db=12.0, add_drop_loss_db=1.0)
    cases = (
        (None, None),
        (EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5), None),
        (EDFAConfig(noise_figure_db=5.5), None),
        (EDFAConfig(noise_figure_db=5.5), roadm),
    )
    total = _evaluator("total")
    per_pol = _evaluator("per_pol")
    total_power = np.full(8, TOTAL_POWER_W)
    per_pol_power = np.full(8, PER_POL_POWER_W)

    for edfa, roadm_config in cases:
        label = f"edfa={edfa!r},roadm={roadm_config!r}"
        total_fiber = total.span_fiber_output(total_power, 80e3, 1)
        per_fiber = per_pol.span_fiber_output(per_pol_power, 80e3, 1)
        for name in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
            _assert_bitwise_equal(2.0 * getattr(per_fiber, name), getattr(total_fiber, name), f"fiber.{name} ({label})")

        total_span = total._calc_span_noise_and_power(total_power, 80e3, edfa, roadm_config)
        per_span = per_pol._calc_span_noise_and_power(per_pol_power, 80e3, edfa, roadm_config)
        for index, (t_value, p_value) in enumerate(zip(total_span, per_span, strict=True)):
            _assert_bitwise_equal(t_value, 2.0 * p_value, f"span[{index}] ({label})")


def test_native_per_pol_matches_total_matvec_kernel() -> None:
    """Matvec 路径：``(N,N)`` 成对矩阵分支同样保持严格 2 倍关系。"""
    total = _evaluator("total", num_channels=48, ode_accel="matvec")
    per_pol = _evaluator("per_pol", num_channels=48, ode_accel="matvec")
    total_power = np.full(48, TOTAL_POWER_W)
    per_pol_power = np.full(48, PER_POL_POWER_W)

    total_span = total._calc_span_noise_and_power(total_power, 80e3, EDFA)
    per_span = per_pol._calc_span_noise_and_power(per_pol_power, 80e3, EDFA)
    # 确认真的走了 matvec 分支，而不是因活跃信道不足落回逐对求和。
    assert total._p1_xpm_matrices is not None
    assert per_pol._p1_xpm_matrices is not None

    for index, (t_value, p_value) in enumerate(zip(total_span, per_span, strict=True)):
        _assert_bitwise_equal(t_value, 2.0 * p_value, f"matvec.span[{index}]")


def test_native_per_pol_batch_matches_total_and_serial() -> None:
    """批量 matvec 入口同样遵守 2 倍关系，且逐候选与串行路径一致。"""
    total = _evaluator("total", num_channels=48, ode_accel="matvec")
    per_pol = _evaluator("per_pol", num_channels=48, ode_accel="matvec")
    total_input = np.vstack([np.full(48, TOTAL_POWER_W), np.full(48, 3e-3), np.zeros(48)])
    per_pol_input = total_input / 2.0

    total_batch = total.calc_span_noise_and_power_batch(total_input, 80e3, EDFA)
    per_batch = per_pol.calc_span_noise_and_power_batch(per_pol_input, 80e3, EDFA)
    for index in range(total_input.shape[0]):
        for component in (0, 1, 3):
            _assert_bitwise_equal(
                total_batch[component][index],
                2.0 * per_batch[component][index],
                f"batch[{component}][{index}]",
            )
        relative = np.abs(2.0 * per_batch[2][index] - total_batch[2][index]) / np.maximum(
            np.abs(total_batch[2][index]), 1e-30
        )
        assert float(relative.max()) <= 1e-12

        serial = per_pol._calc_span_noise_and_power(per_pol_input[index], 80e3, EDFA)
        for component in (0, 1, 3):
            _assert_bitwise_equal(per_batch[component][index], serial[component], f"serial[{component}][{index}]")
        relative = np.abs(per_batch[2][index] - serial[2]) / np.maximum(np.abs(serial[2]), 1e-30)
        assert float(relative.max()) <= 1e-12


# ---------------------------------------------------------------------------
# for_fiber 与 evaluate_path 的口径传播
# ---------------------------------------------------------------------------


def test_for_fiber_keeps_per_pol_convention() -> None:
    """光纤子评估器必须继承口径与 ASE 缩放，否则核会把每偏振功率当总功率。"""
    per_pol = _evaluator("per_pol")
    child = per_pol.for_fiber(_fiber(100.0))
    assert child is per_pol.for_fiber(_fiber(100.0)), "子评估器应按光纤参数缓存"
    assert child.launch_power_convention == "per_pol"
    assert child.ase_polarization_scale == 0.5
    assert child.physical_power_scale == 2.0

    total = _evaluator("total")
    total_child = total.for_fiber(_fiber(100.0))
    assert total_child.launch_power_convention == "total"
    assert total_child.ase_polarization_scale == 1.0
    assert total_child.physical_power_scale == 1.0


@pytest.mark.parametrize("equalize", [False, True])
def test_native_per_pol_evaluate_path_is_total_scaled_by_two(equalize: bool) -> None:
    """``evaluate_path`` 的 per_pol 结果与 total 严格 2 倍，SNR 逐位相同。

    ``equalize=True`` 会打开 ROADM booster ASE，验证 ``accumulate_route`` 确实
    收到 ``ase_polarization_scale=0.5``，新 ASE 不会漏减半。
    """
    roadm = ROADMConfig(equalize_output_power=equalize)
    path = MultiSpanOpticalPath([_fiber()], [EDFA], roadm_configs=[roadm])
    total = _evaluator("total").evaluate_path(path, np.full(8, TOTAL_POWER_W))
    per_pol = _evaluator("per_pol").evaluate_path(path, np.full(8, PER_POL_POWER_W))

    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w"):
        _assert_bitwise_equal(
            getattr(total, name),
            2.0 * getattr(per_pol, name),
            f"evaluate_path.{name} (equalize={equalize})",
        )
    _assert_bitwise_equal(total.snr_db, per_pol.snr_db, f"evaluate_path.snr_db (equalize={equalize})")


@pytest.mark.parametrize("equalize", [False, True])
def test_native_per_pol_network_qot_matches_total(equalize: bool) -> None:
    """网络 QoT 路径（``QoTValidator``）与路径级 ``evaluate_path`` 同口径。

    两节点单链路：total 评估器 + 总功率输入，对 per_pol 评估器 + 每偏振功率
    输入，判定与 SNR 数组必须逐位相同；链路噪声缓存严格 2 倍。
    """
    total_topo, total_grid, total_evaluator, total_state = _network(equalize=equalize, convention="total")
    per_topo, per_grid, per_evaluator, per_state = _network(equalize=equalize, convention="per_pol")

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
        _assert_bitwise_equal(
            getattr(total_cache, name),
            2.0 * getattr(per_cache, name),
            f"network.cache.{name} (equalize={equalize})",
        )
    assert total_cache.span_count == per_cache.span_count


# ---------------------------------------------------------------------------
# 最小网络栈
# ---------------------------------------------------------------------------


def _network(
    *,
    equalize: bool,
    convention: str,
) -> tuple[NetworkTopology, SpectrumGrid, MultiBandISRSGN, NetworkState]:
    """两节点单链路的最小 RWA 栈，与 ``test_qot_network_perpol`` 同构。"""
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, weight=80.0)
    fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)
    topology = NetworkTopology(
        graph,
        fiber,
        EDFAConfig(noise_figure_db=5.0),
        ROADMConfig(equalize_output_power=equalize),
    )
    grid = _grid(4)
    evaluator = MultiBandISRSGN(grid, fiber, 32e9, launch_power_convention=convention)
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


# ---------------------------------------------------------------------------
# 非法口径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("convention", ["per-pal", "", "TOTAL", "auto", "per-pol"])
def test_invalid_convention_is_rejected(convention: str) -> None:
    """口径只接受 ``total`` / ``per_pol``，非法值在构造期直接报错。"""
    with pytest.raises(ValueError, match="launch_power_convention must be one of"):
        MultiBandISRSGN(_grid(), _fiber(), 50e9, launch_power_convention=convention)
