r"""FG-20 §1：``roadm_configs`` 里的 ``None`` 成员按"无 ROADM"裁决的**行为**守卫。

裁决（编排方已裁，见 ``docs/fg20_*.md`` §1）
--------------------------------------------
列表中的 ``None`` 成员**视为无 ROADM 并跳过**，与字段级 ``roadm_configs or []``
的兜底**同义**：``[cfg, None]`` 与 ``[cfg]`` 给出逐位相同的掩码，``[None, None]``
与 ``[]`` 逐位相同。

本文件守的是**行为**（掩码逐位相同），不是源码形状。FG-18 取证记录的原始崩溃
（``docs/fg18_assembly_convergence.md`` §6）来自
``physics/qot_checker.py::evaluate_path`` 里直接解引用 ``roadm.blocked_channels``：
``[ROADMConfig(), None]`` 抛 ``AttributeError``。

⚠️ **残余风险，本文件不掩盖它**
-------------------------------
"跳过"会把调用方的**畸形输入**静默折叠成"确实没有 ROADM"：``[None]`` 本该是
"这跳漏填了还是确实没有？"的信号，现在无声无息。少一段 WSS 滤波与插入损耗而
SNR 仍然算得出来——**依据被丢掉了**。这是本裁决明确接受的代价；生产路径
``NetworkTopology.build_optical_path`` 结构上只产 ``ROADMConfig``，故该折叠
只可能由仓外畸形构造触发。本文件用 ``test_none_member_is_masked_not_crashed``
把该行为**钉成显式契约**，并在下面一条用例里留出"这是有意的"注释锚点。
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics.qot_checker import _active_channel_mask


def _path(*, roadm_configs: list[ROADMConfig | None] | None, num_spans: int = 2) -> MultiSpanOpticalPath:
    """构造一条仅用于掩码语义的多跨段路径（不传播、不解算 ODE）。"""
    return MultiSpanOpticalPath(
        fiber_configs=[FiberSpanConfig(length_km=100.0) for _ in range(num_spans)],
        edfa_configs=[EDFAConfig() for _ in range(num_spans)],
        roadm_configs=roadm_configs,
    )


def _topology() -> NetworkTopology:
    graph = nx.Graph()
    graph.add_node(100)
    graph.add_node(200, roadm={"insertion_loss_db": 5.0})
    graph.add_edge(100, 200, weight=100.0, fiber={"max_span_length_km": 100.0}, edfa={"noise_figure_db": 5.0})
    return NetworkTopology(graph, FiberSpanConfig(length_km=100.0), EDFAConfig(), ROADMConfig())


#: 发射功率剖面：第 2 个信道零功率，故"被发射阻断"与"被 ROADM 阻断"可区分。
_LAUNCH = np.array([1e-3, 0.0, 2e-3])


def _mask_bytes(roadm_configs: list[ROADMConfig | None] | None) -> bytes:
    """掩码的**逐位**表示。

    用 ``.tobytes()`` 而不是 ``np.array_equal``：后者对 ``bool`` 数组本身已经够用，
    但 ``tobytes()`` 额外证明**没有任何未被断言的表示差异**（dtype、shape、
    字节序），即"逐位相同"这句话本身也被验证了。
    """
    return _active_channel_mask(_LAUNCH, roadm_configs).tobytes()


# ---------------------------------------------------------------------------
# 裁决本体：None 成员 ≡ 不存在
# ---------------------------------------------------------------------------


def test_trailing_none_is_bitwise_identical_to_absent_member() -> None:
    """``[cfg, None]`` 与 ``[cfg]`` **逐位**相同 —— 这是裁决的主命题。"""
    cfg = ROADMConfig(blocked_channels=[0])
    assert _mask_bytes([cfg, None]) == _mask_bytes([cfg])


def test_leading_none_is_bitwise_identical_to_absent_member() -> None:
    """``[None, cfg]`` 与 ``[cfg]`` **逐位**相同（``None`` 不必在尾部）。"""
    cfg = ROADMConfig(blocked_channels=[2])
    assert _mask_bytes([None, cfg]) == _mask_bytes([cfg])


def test_all_none_is_bitwise_identical_to_empty_list() -> None:
    """``[None, None]`` 与 ``[]`` **逐位**相同。"""
    assert _mask_bytes([None, None]) == _mask_bytes([])


def test_all_none_is_bitwise_identical_to_field_level_none() -> None:
    """``[None, None]`` 与字段级 ``None``（``or []`` 兜底）**逐位**相同。

    这是裁决措辞"与字段级 ``or []`` 同义"的直接读数：三种"没有 ROADM"的写法
    收敛到同一个掩码。
    """
    assert _mask_bytes([None, None]) == _mask_bytes(None)


def test_none_member_is_masked_not_crashed() -> None:
    """含 ``None`` 成员**不抛**，且掩码只由非 ``None`` 成员决定。

    ⚠️ 本用例即残余风险的锚点：``[None]`` 被**静默**当成"没有 ROADM"。
    若将来编排方改裁为"报错"，这条会红 —— 那是**有意的**红，不是回归。
    """
    cfg = ROADMConfig(blocked_channels=[0])
    # 只有 cfg 生效：第 0 信道被阻断，第 2 信道保留，第 1 因零发射本就未激活。
    assert _active_channel_mask(_LAUNCH, [cfg, None]).tolist() == [False, False, True]
    # [None] 静默等价于 [] —— 覆盖发生，但没有任何告警。
    assert _mask_bytes([None]) == _mask_bytes([])


def test_empty_list_is_unaffected_by_the_adjudication() -> None:
    """``[]`` 仍正常（FG-18 取证里它本来就不崩，裁决不得改变它）。"""
    assert _active_channel_mask(_LAUNCH, []).tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# 非 None 语义不得因裁决而移动
# ---------------------------------------------------------------------------


def test_non_none_members_still_apply_their_blocking() -> None:
    """裁决只动 ``None``，非 ``None`` 成员的阻断语义逐位不变。"""
    mask = _active_channel_mask(_LAUNCH, [ROADMConfig(blocked_channels=[0, 2])])
    assert mask.tolist() == [False, False, False]


def test_out_of_range_blocked_index_is_dropped_as_before() -> None:
    """越界阻断下标仍按既有行为丢弃（``index < len(active)``），不新抛。"""
    mask = _active_channel_mask(_LAUNCH, [ROADMConfig(blocked_channels=[99])])
    assert mask.tolist() == [True, False, True]


def test_mask_starts_from_launch_power_not_from_all_true() -> None:
    """掩码起点是 ``launch > 0``，不是全 True。"""
    mask = _active_channel_mask(_LAUNCH, [None])
    assert mask.tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# 生产路径结构上不可能产出 None 成员（本裁决的实际触发面）
# ---------------------------------------------------------------------------


def test_build_optical_path_never_emits_a_none_member() -> None:
    """``build_optical_path`` 只 append ``ROADMConfig``，故生产路径无需该防御。

    这条同时是**残余风险的边界声明**：仓内活调用方拿不到含 ``None`` 的列表，
    所以"静默掩盖畸形输入"在本仓**不会**发生；它只在仓外构造时才有后果。
    """
    topology = _topology()
    path = [topology.get_internal_node_id(100), topology.get_internal_node_id(200)]
    built = topology.build_optical_path(path)
    assert built.roadm_configs is not None
    assert built.roadm_configs, "生产路径应至少含一个 ROADM 配置"
    assert all(isinstance(cfg, ROADMConfig) for cfg in built.roadm_configs)


def test_mask_from_production_path_is_reachable_without_none() -> None:
    """生产路径的掩码本身可用（守卫函数不是只在畸形输入下才被调用）。"""
    topology = _topology()
    built = topology.build_optical_path(
        [topology.get_internal_node_id(100), topology.get_internal_node_id(200)]
    )
    assert _active_channel_mask(_LAUNCH, built.roadm_configs).tolist() == [True, False, True]


# ---------------------------------------------------------------------------
# 端到端：`evaluate_path` 不再因 None 成员崩溃（需要波形引擎）
# ---------------------------------------------------------------------------


def test_evaluate_path_tolerates_none_member_end_to_end() -> None:
    """波形级 ``evaluate_path`` 在 ``roadm_configs`` 含 ``None`` 时**不再抛**。

    这是 FG-18 取证记录的崩溃点的直接回归。波形引擎只在 ``nlo`` 环境可用，故本用例
    在物理仓全量门（``rl``）下被 ``importorskip`` 跳过；``nlo`` 下的读数见
    ``docs/fg20_*.md`` §5 的"波形侧读数"小节（须显式指定解释器复跑）。
    """
    pytest.importorskip("physics_nlo_eval")
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator

    grid_cfg = SpectrumGrid(3, 2e9, 193.1e12)
    fiber = FiberSpanConfig(
        length_km=0.01,
        max_span_length_km=0.01,
        dispersion_parameter_d=0,
        dispersion_slope_s=0,
        nonlinear_index_n2=0,
    )
    roadm = ROADMConfig(insertion_loss_db=1, add_drop_loss_db=0, passband_hz=2e9)
    evaluator = PyNLOQoTEvaluator(grid_cfg, symbol_rate_hz=1e9, sps=2, seed=3)
    launch = np.array([0.0, 1e-3, 0.0])

    # 判决用例：末跳是 None 成员。原实现在此抛 AttributeError。
    with_none = MultiSpanOpticalPath([fiber], [EDFAConfig()], [None])
    signal, _ = evaluator.create_wdm_signal(launch, num_symbols=64)
    out_none = evaluator.propagate_path(signal, with_none)
    mask_none = _active_channel_mask(launch, with_none.roadm_configs)
    assert mask_none.tolist() == [False, True, False]

    # 逐位对照：只有 cfg 时应给出相同掩码（信号本身含随机种子，只比掩码）。
    with_cfg = MultiSpanOpticalPath([fiber], [EDFAConfig()], [roadm])
    mask_cfg = _active_channel_mask(launch, with_cfg.roadm_configs)
    assert mask_none.tobytes() == mask_cfg.tobytes()
    assert np.all(np.isfinite(out_none.data))
