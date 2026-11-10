"""任务 AS：网络级第十二轮开关（光纤级集中损耗 + auto-Raman 分辨率动作）。

覆盖：

* ``FiberSpanConfig.lumped_loss`` 的 Pydantic 校验、空默认与序列化稳定性；
* 无泵 ``pcfm`` 与有泵 ``pcfm_h3_raman`` 都把光纤级集中损耗真正送进 kernel，
  默认空列表逐位不变；
* ``PCFH3RamanModel`` 的“模型级优先于光纤级”规则；
* ``LinkValidity.raman_auto_under_resolved`` 从预算透传、``to_dict`` 与
  ``ChannelValidity.combine`` 的逻辑或；
* ``LinkValidityPolicy.raman_resolution_action`` 的 ``flag`` / ``reject`` 语义，
  与 ``dense_grid_action`` 正交；
* 非法动作与形状错误在校验期报错。

所有算例都由 ``benchmarks.cases`` 与本地构造的 fiber/validity 生成，不读
``data/`` 或 ``results/``。数值参考沿用 ``tests/test_h3_lumped_loss.py`` 的
分段语义：那里已经用“两段串联 + 中间衰减器”钉住中间损耗，本文件只补光纤级
透传与端点自洽，不再重复那段参考。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
from network_nlo_eval.network.elements import FiberSpanConfig, RamanPumpConfig, ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache
from network_nlo_eval.rwa.qot_checker import accumulate_service_snr_detail, fold_roadm_terms
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    DenseGridAction,
    LinkValidity,
    LinkValidityPolicy,
    RamanResolutionAction,
)

_RESULT_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _pump(direction: str = "forward", power_w: float = 0.2) -> RamanPumpConfig:
    """1450 nm 单泵。"""
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=power_w,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _with_loss(fiber: FiberSpanConfig, lumped_loss: list[tuple[float, float]]) -> FiberSpanConfig:
    """复制光纤配置并替换集中损耗（与网络桥的 ``model_copy`` 同路径）。"""
    return fiber.model_copy(update={"lumped_loss": lumped_loss})


def _path_with(path, fiber: FiberSpanConfig) -> MultiSpanOpticalPath:
    """把路径的每一跨都替换成给定光纤配置。"""
    return dataclasses.replace(path, fiber_configs=[fiber] * len(path.fiber_configs))


# ---------------------------------------------------------------------------
# FiberSpanConfig 校验
# ---------------------------------------------------------------------------


def test_default_lumped_loss_is_empty_and_round_trips() -> None:
    """默认空列表；dump/validate 与 JSON 往返稳定。"""
    fiber = FiberSpanConfig(length_km=100.0)
    assert fiber.lumped_loss == []
    assert fiber.model_dump()["lumped_loss"] == []
    assert FiberSpanConfig.model_validate(fiber.model_dump()) == fiber
    assert FiberSpanConfig.model_validate_json(fiber.model_dump_json()) == fiber

    loaded = FiberSpanConfig(length_km=100.0, lumped_loss=[(50e3, 3.0), (80e3, 0.5)])
    assert loaded.lumped_loss == [(50e3, 3.0), (80e3, 0.5)]
    assert FiberSpanConfig.model_validate_json(loaded.model_dump_json()) == loaded


@pytest.mark.parametrize(
    "entry",
    [
        (0.0, 1.0),
        (100e3, 1.0),
        (100.001e3, 1.0),
        (-1.0, 1.0),
        (float("nan"), 1.0),
        (float("inf"), 1.0),
        (50e3, -0.5),
        (50e3, float("nan")),
        (50e3, float("inf")),
    ],
)
def test_invalid_lumped_loss_is_rejected_by_pydantic(entry: tuple[float, float]) -> None:
    """位置必须严格在 (0, L)、损耗有限非负；错误在构造期就抛出。"""
    with pytest.raises(ValueError):
        FiberSpanConfig(length_km=100.0, lumped_loss=[entry])


def test_duplicate_positions_are_allowed_by_the_config() -> None:
    """同一位置的多个连接器允许出现，由 kernel 按 dB 合并。"""
    fiber = FiberSpanConfig(length_km=100.0, lumped_loss=[(50e3, 1.0), (50e3, 2.0)])
    assert fiber.lumped_loss == [(50e3, 1.0), (50e3, 2.0)]


# ---------------------------------------------------------------------------
# 无泵 PCFM：光纤级集中损耗
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pcfm() -> dict[str, object]:
    """``scaling_40ch`` 的 PCFM 模型、参考光纤、路径与功率。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    return {
        "case": case,
        "path": path,
        "power": launch_profile(case, None),
        "model": make_model("pcfm", case),
        "fiber": path.fiber_configs[0],
        "length_m": case.span_length_km * 1e3,
    }


def test_pcfm_default_matches_empty_list_bitwise(pcfm) -> None:
    """默认 ``[]`` 与显式空列表逐位一致：新字段不进入任何分支。"""
    model = pcfm["model"]
    path = pcfm["path"]
    power = pcfm["power"]
    fiber = pcfm["fiber"]

    implicit = model.evaluate_path(path, power)
    explicit = model.evaluate_path(_path_with(path, _with_loss(fiber, [])), power)
    for name in _RESULT_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(implicit, name)), np.asarray(getattr(explicit, name))
        ), name
    assert model.kernel_for(fiber).lumped_loss == []
    assert model.kernel_for(_with_loss(fiber, [])).lumped_loss == []


def test_pcfm_fiber_lumped_loss_reaches_the_kernel(pcfm) -> None:
    """光纤级损耗进入 SPP 分段：中间 3 dB 使跨末功率约降 3 dB，端点精确。"""
    model = pcfm["model"]
    power = pcfm["power"]
    fiber = pcfm["fiber"]
    length_m = pcfm["length_m"]

    plain = model.kernel_for(fiber).evaluate(power, length_m, 1)
    midpoint = _with_loss(fiber, [(50e3, 3.0)])
    lossy_kernel = model.kernel_for(midpoint)
    assert lossy_kernel.lumped_loss == [(50e3, 3.0)]
    lossy = lossy_kernel.evaluate(power, length_m, 1)
    np.testing.assert_allclose(
        np.asarray(lossy.power_out_w) / np.asarray(plain.power_out_w),
        10.0 ** (-3.0 / 10.0),
        rtol=1e-2,
    )
    nli_plain = np.asarray(plain.spm_noise_w) + np.asarray(plain.xpm_noise_w)
    nli_lossy = np.asarray(lossy.spm_noise_w) + np.asarray(lossy.xpm_noise_w)
    assert np.all(nli_lossy > 0.0) and np.all(nli_lossy < nli_plain)

    # 端点：L-1 m 处的 2 dB 不参与分段积分，跨末功率恰好再低 2 dB。
    near_end = model.kernel_for(_with_loss(fiber, [(length_m - 1.0, 2.0)])).evaluate(
        power, length_m, 1
    )
    np.testing.assert_allclose(
        np.asarray(near_end.power_out_w) / np.asarray(plain.power_out_w),
        10.0 ** (-2.0 / 10.0),
        rtol=1e-6,
    )


def test_pcfm_kernel_merges_duplicate_positions(pcfm) -> None:
    """无泵 kernel 与有泵一致：同一位置按 dB 相加合并。"""
    kernel = pcfm["model"].kernel_for(_with_loss(pcfm["fiber"], [(50e3, 1.0), (50e3, 2.0)]))
    assert kernel.lumped_loss == [(50e3, 3.0)]


# ---------------------------------------------------------------------------
# 有泵 H3：光纤级集中损耗与模型级优先级
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h3() -> dict[str, object]:
    """``c_57ch`` 的单前向泵 H3 模型、参考光纤、路径与功率。"""
    case = build_case("c_57ch")
    path = build_path(case)
    return {
        "case": case,
        "path": path,
        "power": launch_profile(case),
        "model": make_model("pcfm_h3_raman", case, raman_pumps=[_pump("forward")]),
        "fiber": path.fiber_configs[0],
        "length_m": case.span_length_km * 1e3,
    }


def test_h3_default_is_bit_identical(h3) -> None:
    """有泵路径的默认空列表逐位不变。"""
    model = h3["model"]
    path = h3["path"]
    power = h3["power"]
    implicit = model.evaluate_path(path, power)
    explicit = model.evaluate_path(_path_with(path, _with_loss(h3["fiber"], [])), power)
    for name in _RESULT_FIELDS:
        assert np.array_equal(
            np.asarray(getattr(implicit, name)), np.asarray(getattr(explicit, name))
        ), name
    assert model.kernel_for(h3["fiber"]).lumped_loss == []


def test_h3_fiber_lumped_loss_enters_the_pumped_ode(h3) -> None:
    """光纤级损耗在有泵路径进入耦合 ODE，并真实改变功率剖面。"""
    model = h3["model"]
    power = h3["power"]
    length_m = h3["length_m"]
    plain = model.kernel_for(h3["fiber"]).evaluate(power, length_m, 1)
    kernel = model.kernel_for(_with_loss(h3["fiber"], [(50e3, 3.0)]))
    assert kernel.lumped_loss == [(50e3, 3.0)]
    assert kernel.lumped_loss_in_ode is True
    assert kernel.lumped_loss_segments == 2
    lossy = kernel.evaluate(power, length_m, 1)
    ratio = np.asarray(lossy.power_out_w) / np.asarray(plain.power_out_w)
    assert np.all(ratio < 0.6)


def test_h3_fiber_and_model_level_lists_produce_the_same_kernel(h3) -> None:
    """同一组损耗从光纤级或模型级进来，kernel 输出逐位相同。"""
    case = h3["case"]
    power = h3["power"]
    length_m = h3["length_m"]
    fiber_level = h3["model"].kernel_for(_with_loss(h3["fiber"], [(50e3, 3.0)]))
    model_level = make_model(
        "pcfm_h3_raman", case, raman_pumps=[_pump("forward")], lumped_loss=[(50e3, 3.0)]
    ).kernel_for(h3["fiber"])
    left = fiber_level.evaluate(power, length_m, 1)
    right = model_level.evaluate(power, length_m, 1)
    for name in ("power_out_w", "spm_noise_w", "xpm_noise_w"):
        assert np.array_equal(
            np.asarray(getattr(left, name)), np.asarray(getattr(right, name))
        ), name


def test_h3_model_level_takes_precedence_over_the_fiber(h3) -> None:
    """模型级显式给出时优先；未给出时用光纤级；两者都空时为空。"""
    case = h3["case"]
    fiber = h3["fiber"]
    fiber_specific = _with_loss(fiber, [(20e3, 5.0)])

    override = make_model(
        "pcfm_h3_raman", case, raman_pumps=[_pump("forward")], lumped_loss=[(60e3, 1.0)]
    )
    assert override.kernel_for(fiber_specific).lumped_loss == [(60e3, 1.0)]

    inherited = make_model("pcfm_h3_raman", case, raman_pumps=[_pump("forward")])
    assert inherited.kernel_for(fiber_specific).lumped_loss == [(20e3, 5.0)]
    assert inherited.kernel_for(fiber).lumped_loss == []


def test_h3_without_pumps_forwards_the_fiber_list_to_the_parent(h3) -> None:
    """无泵路径把光纤级损耗交给父类 SPP，报告 ``in_ode=False``。"""
    case = h3["case"]
    model = make_model("pcfm_h3_raman", case, raman_pumps=[])
    kernel = model.kernel_for(_with_loss(h3["fiber"], [(50e3, 3.0)]))
    assert kernel.lumped_loss == [(50e3, 3.0)]
    assert kernel.lumped_loss_in_ode is False
    assert kernel.lumped_loss_segments == 2

    result = model.evaluate_path(
        _path_with(h3["path"], _with_loss(h3["fiber"], [(50e3, 3.0)])), h3["power"]
    )
    assert np.all(np.isfinite(np.asarray(result.snr_db)))


# ---------------------------------------------------------------------------
# LinkValidity / ChannelValidity 透传
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def budget_pair() -> dict[str, object]:
    """同一份真实预算，flag 由本地 ``replace`` 打开/关闭（不伪造物理量）。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    model = make_model("pcfm", case)
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)
    flagged = dataclasses.replace(
        budget, validity=dataclasses.replace(budget.validity, raman_auto_under_resolved=True)
    )
    return {"plain": budget, "flagged": flagged, "snr": np.asarray(result.snr_db)}


def test_link_validity_reads_the_budget_flag(budget_pair) -> None:
    """``from_budget`` 读该字段，``to_dict`` 输出该键，默认 False。"""
    plain = LinkValidity.from_budget(budget_pair["plain"])
    flagged = LinkValidity.from_budget(budget_pair["flagged"], link_key=(0, 1))
    assert plain.raman_auto_under_resolved is False
    assert flagged.raman_auto_under_resolved is True
    assert plain.to_dict()["raman_auto_under_resolved"] is False
    assert flagged.to_dict()["raman_auto_under_resolved"] is True
    # 旧调用方不传：字段默认 False，报告数值不变。
    manual = LinkValidity(
        valid=plain.valid,
        active=plain.active,
        flagged=plain.flagged,
        confidence=plain.confidence,
        margin_db=plain.margin_db,
        model=plain.model,
    )
    assert manual.raman_auto_under_resolved is False
    assert np.array_equal(manual.margin_db, plain.margin_db)


def test_channel_validity_combines_the_flag_with_or(budget_pair) -> None:
    """路径上任一链路命中即为 True；都不命中为 False。"""
    plain = LinkValidity.from_budget(budget_pair["plain"])
    flagged = LinkValidity.from_budget(budget_pair["flagged"])
    n = plain.num_channels
    assert ChannelValidity.combine([plain, plain], n).raman_auto_under_resolved is False
    assert ChannelValidity.combine([plain, flagged], n).raman_auto_under_resolved is True
    assert ChannelValidity.combine([flagged, plain], n).raman_auto_under_resolved is True


def test_link_validity_passes_the_flag_into_the_policy(budget_pair) -> None:
    """链路报告的 flag 走到策略：flag 不动 SNR，reject 只扣可用性。"""
    flagged = LinkValidity.from_budget(budget_pair["flagged"])
    snr = budget_pair["snr"]
    audit = LinkValidityPolicy(mode="flag", raman_resolution_action="reject")
    rejected = LinkValidityPolicy(mode="reject", raman_resolution_action="reject")

    assert not bool(np.any(flagged.effective_valid(audit)))
    assert np.array_equal(flagged.adjusted_snr_db(snr, audit), snr)
    assert np.all(np.isneginf(flagged.adjusted_snr_db(snr, rejected)))
    # 报告存储的点值/有效性没有被策略改写。
    assert np.array_equal(flagged.to_dict()["valid"], [bool(v) for v in flagged.valid])
    assert bool(np.all(flagged.valid))


# ---------------------------------------------------------------------------
# 策略：flag / reject / 非法动作 / 与密集栅格正交
# ---------------------------------------------------------------------------


def _stub(n: int = 4, *, dense: bool = False, under_resolved: bool = False) -> LinkValidity:
    """全有效的可控报告。"""
    return LinkValidity(
        valid=np.ones(n, dtype=bool),
        active=np.ones(n, dtype=bool),
        flagged=np.zeros(n, dtype=bool),
        confidence=np.ones(n, dtype=np.float64),
        margin_db=np.ones(n, dtype=np.float64),
        model="stub",
        span_count=1,
        dense_grid_overlap=dense,
        raman_auto_under_resolved=under_resolved,
    )


def test_policy_defaults_are_flag_only() -> None:
    """默认动作是 FLAG，属性与数值都不参与运算。"""
    policy = LinkValidityPolicy()
    assert policy.raman_resolution_action is RamanResolutionAction.FLAG
    assert policy.rejects_raman_resolution is False
    # 显式写出默认值也一样。
    explicit = LinkValidityPolicy(raman_resolution_action="flag")
    assert explicit.raman_resolution_action is RamanResolutionAction.FLAG

    valid = np.array([True, False, True, True])
    snr = np.array([20.0, 20.0, 20.0, 20.0])
    # 默认路径原样返回同一个数组：不做任何算术。
    assert policy.effective_valid(valid) is valid
    assert policy.effective_valid(valid, raman_auto_under_resolved=True) is valid
    assert np.array_equal(
        policy.apply(snr, valid, np.ones(4), raman_auto_under_resolved=True), snr
    )
    # mode="off" 时 reject 动作也不生效。
    off = LinkValidityPolicy(mode="off", raman_resolution_action="reject")
    assert off.rejects_raman_resolution is False
    assert np.array_equal(off.effective_valid(valid, raman_auto_under_resolved=True), valid)


def test_reject_action_only_changes_availability() -> None:
    """``reject`` 动作只把命中信道判为不可用，SNR 点值只在落地的 mode 中变 -inf。"""
    valid = np.ones(4, dtype=bool)
    under = np.array([True, False, True, False])
    snr = np.array([20.0, 21.0, 22.0, 23.0])
    margin = np.ones(4)

    reject = LinkValidityPolicy(mode="reject", raman_resolution_action="reject")
    assert np.array_equal(
        reject.effective_valid(valid, raman_auto_under_resolved=under),
        np.array([False, True, False, True]),
    )
    adjusted = reject.apply(snr, valid, margin, raman_auto_under_resolved=under)
    assert np.isneginf(adjusted[0]) and np.isneginf(adjusted[2])
    assert adjusted[1] == np.float64(21.0) and adjusted[3] == np.float64(23.0)

    audit = LinkValidityPolicy(mode="flag", raman_resolution_action="reject")
    assert np.array_equal(audit.apply(snr, valid, margin, raman_auto_under_resolved=under), snr)


def test_unknown_raman_resolution_action_is_rejected() -> None:
    """动作必须显式且合法，不允许静默退回默认。"""
    with pytest.raises(ValueError, match="unknown Raman resolution action"):
        LinkValidityPolicy(raman_resolution_action="maybe")


def test_raman_flag_shape_is_validated() -> None:
    """形状校验与 ``dense_grid_overlap`` 同风格。"""
    policy = LinkValidityPolicy(mode="reject", raman_resolution_action="reject")
    valid = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="raman_auto_under_resolved must be a scalar"):
        policy.effective_valid(valid, raman_auto_under_resolved=np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="raman_auto_under_resolved must be a scalar"):
        policy.effective_valid(valid, raman_auto_under_resolved=np.ones(3, dtype=bool))


def test_dense_and_raman_actions_are_orthogonal() -> None:
    """两个动作各自只扣自己的标志，同时打开时按逻辑或合并。"""
    valid = np.ones(4, dtype=bool)
    dense = np.array([True, True, False, False])
    under = np.array([True, False, True, False])
    dense_only = LinkValidityPolicy(mode="reject", dense_grid_action="reject")
    raman_only = LinkValidityPolicy(mode="reject", raman_resolution_action="reject")
    both = LinkValidityPolicy(
        mode="reject", dense_grid_action="reject", raman_resolution_action="reject"
    )

    np.testing.assert_array_equal(
        dense_only.effective_valid(valid, dense, raman_auto_under_resolved=under),
        np.array([False, False, True, True]),
    )
    np.testing.assert_array_equal(
        raman_only.effective_valid(valid, dense, raman_auto_under_resolved=under),
        np.array([False, True, False, True]),
    )
    np.testing.assert_array_equal(
        both.effective_valid(valid, dense, raman_auto_under_resolved=under),
        np.array([False, False, False, True]),
    )
    # 关闭一个动作不会抹掉另一个的效果。
    np.testing.assert_array_equal(
        both.effective_valid(valid, dense, raman_auto_under_resolved=False),
        np.array([False, False, True, True]),
    )
    np.testing.assert_array_equal(
        both.effective_valid(valid, False, raman_auto_under_resolved=under),
        np.array([False, True, False, True]),
    )


def test_channel_validity_carries_both_flags() -> None:
    """路径聚合同时保留两个标志，并交给同一份策略。"""
    verdict = ChannelValidity.combine([_stub(dense=True), _stub(under_resolved=True)], 4)
    assert verdict.dense_grid_overlap is True
    assert verdict.raman_auto_under_resolved is True
    policy = LinkValidityPolicy(
        mode="reject", dense_grid_action=DenseGridAction.REJECT, raman_resolution_action="reject"
    )
    assert not bool(np.any(verdict.effective_valid(policy)))
    assert np.all(np.isneginf(verdict.adjusted_snr_db(np.full(4, 20.0), policy)))

    raman_only = ChannelValidity.combine([_stub(), _stub(under_resolved=True)], 4)
    assert raman_only.dense_grid_overlap is False
    assert raman_only.raman_auto_under_resolved is True
    assert not bool(np.any(raman_only.effective_valid(policy)))
    assert np.array_equal(
        raman_only.adjusted_snr_db(np.full(4, 20.0), LinkValidityPolicy(mode="reject")),
        np.full(4, 20.0),
    )


def test_scalar_service_aggregation_carries_the_flag(budget_pair) -> None:
    """标量业务 SNR 与逐信道路径走同一份策略语义。"""
    terms = fold_roadm_terms(ROADMConfig(insertion_loss_db=0.0, add_drop_loss_db=0.0), is_last_hop=True)
    zero = LinkNoiseCache(
        spm=np.zeros(2),
        xpm=np.zeros(2),
        ase=np.full(2, 1e-6),
        span_count=1,
    )

    def detail(link: LinkValidity):
        cache = dataclasses.replace(zero, validity=link)
        return accumulate_service_snr_detail(
            [(terms, cache)],
            channel_idx=0,
            launch_power_w=1e-3,
            frequency_hz=193.1e12,
            channel_bandwidth_hz=50e9,
            is_blocked=lambda _terms: False,
            validity_of=lambda _terms, item: item.validity,
        )

    flagged = detail(LinkValidity.from_budget(budget_pair["flagged"]))
    plain = detail(LinkValidity.from_budget(budget_pair["plain"]))
    assert flagged.raman_auto_under_resolved is True
    assert plain.raman_auto_under_resolved is False

    reject = LinkValidityPolicy(mode="reject", raman_resolution_action="reject")
    assert np.isneginf(flagged.effective_snr_db(reject))
    # 默认动作与 flag 模式都不改标量：可用性只看物理有效性。
    assert flagged.effective_snr_db(LinkValidityPolicy(mode="reject")) == flagged.snr_db
    assert plain.effective_snr_db(reject) == plain.snr_db
