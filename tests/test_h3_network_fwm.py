"""任务 AV：网络级 FWM 有效性动作（``fwm_required`` 透传 + ``fwm_action``）。

覆盖：

* ``LinkValidity`` 从预算透传 ``fwm_required`` / ``fwm_reason`` /
  ``fwm_validity_mode``，``to_dict`` 输出三个键，旧调用方默认逐位不变；
* ``ChannelValidity.combine`` 的逻辑或、原因去重合并与最保守模式；
* ``ServiceQoT`` / ``accumulate_service_snr_detail`` 的逻辑或与标量策略；
* ``LinkValidityPolicy.fwm_action`` 的 ``flag`` / ``reject`` / 非法动作语义，
  与 ``dense_grid_action`` / ``raman_resolution_action`` 正交；
* 形状错误在校验期报错。

所有算例都由 ``benchmarks.cases`` 与本地构造的 fiber/validity/budget 生成，
不读 ``data/`` 或 ``results/``。
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.uncertainty import estimate_h3_error_budget
from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache
from network_nlo_eval.rwa.qot_checker import accumulate_service_snr_detail, fold_roadm_terms
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    FwmAction,
    LinkValidity,
    LinkValidityPolicy,
)

_OLD_TO_DICT_KEYS = (
    "model",
    "link_key",
    "fingerprint",
    "span_count",
    "valid",
    "active",
    "flagged",
    "confidence",
    "margin_db",
    "tilt_confidence_informational",
    "soft_margin_db",
    "dense_grid_overlap",
    "raman_auto_under_resolved",
)


@pytest.fixture(scope="module")
def budget_pair() -> dict[str, object]:
    """同一份真实预算，FWM 字段由本地 ``replace`` 打开（不伪造物理量）。"""
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case, None)
    model = make_model("pcfm", case)
    result = model.evaluate_path(path, power)
    budget = estimate_h3_error_budget(model, path, power, result)
    flagged = dataclasses.replace(
        budget,
        validity=dataclasses.replace(
            budget.validity,
            fwm_required=True,
            fwm_reason="low_dispersion",
            fwm_validity_mode="reject",
        ),
    )
    return {"plain": budget, "flagged": flagged, "snr": np.asarray(result.snr_db)}


def _stub(
    n: int = 4,
    *,
    dense: bool = False,
    under_resolved: bool = False,
    fwm: bool = False,
    fwm_reason: str = "",
    fwm_mode: str = "flag",
) -> LinkValidity:
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
        fwm_required=fwm,
        fwm_reason=fwm_reason,
        fwm_validity_mode=fwm_mode,
    )


# ---------------------------------------------------------------------------
# 默认不变
# ---------------------------------------------------------------------------


def test_default_link_validity_is_bitwise_unchanged(budget_pair) -> None:
    """默认字段是 ``False`` / ``""`` / ``"flag"``；旧键数值逐位不变。"""
    plain = LinkValidity.from_budget(budget_pair["plain"])
    assert plain.fwm_required is False
    assert plain.fwm_reason == ""
    assert plain.fwm_validity_mode == "flag"

    data = plain.to_dict()
    assert data["fwm_required"] is False
    assert data["fwm_reason"] == ""
    assert data["fwm_validity_mode"] == "flag"

    manual = LinkValidity(
        valid=plain.valid,
        active=plain.active,
        flagged=plain.flagged,
        confidence=plain.confidence,
        margin_db=plain.margin_db,
        model=plain.model,
        span_count=plain.span_count,
        fingerprint=plain.fingerprint,
        link_key=plain.link_key,
    )
    assert manual.fwm_required is False
    assert manual.fwm_reason == ""
    assert manual.fwm_validity_mode == "flag"
    assert np.array_equal(manual.valid, plain.valid)
    assert np.array_equal(manual.margin_db, plain.margin_db)
    # 除新增键外，输出与旧版本逐位一致。
    assert {key: data[key] for key in _OLD_TO_DICT_KEYS} == {
        key: manual.to_dict()[key] for key in _OLD_TO_DICT_KEYS
    }


def test_default_policy_returns_the_same_array() -> None:
    """关闭三个动作时 ``effective_valid`` 返回同一个数组，不做算术。"""
    policy = LinkValidityPolicy()
    valid = np.array([True, False, True, True])
    assert policy.fwm_action is FwmAction.FLAG
    assert policy.rejects_fwm is False
    assert policy.effective_valid(valid) is valid
    assert policy.effective_valid(valid, fwm_required=True) is valid
    assert LinkValidityPolicy(fwm_action="flag").fwm_action is FwmAction.FLAG
    off = LinkValidityPolicy(mode="off", fwm_action="reject")
    assert off.rejects_fwm is False
    assert off.effective_valid(valid, fwm_required=True) is valid


# ---------------------------------------------------------------------------
# 预算透传 / to_dict / 路径 combine
# ---------------------------------------------------------------------------


def test_link_validity_reads_the_fwm_fields(budget_pair) -> None:
    """``from_budget`` 读三个字段，``to_dict`` 原样输出。"""
    plain = LinkValidity.from_budget(budget_pair["plain"])
    flagged = LinkValidity.from_budget(budget_pair["flagged"])
    assert plain.fwm_required is False and flagged.fwm_required is True
    assert flagged.fwm_reason == "low_dispersion"
    assert flagged.fwm_validity_mode == "reject"
    data = flagged.to_dict()
    assert data["fwm_required"] is True
    assert data["fwm_reason"] == "low_dispersion"
    assert data["fwm_validity_mode"] == "reject"
    # 报告存储的可用性/点值没有被新字段改写。
    assert np.array_equal(flagged.valid, plain.valid)
    assert np.array_equal(flagged.margin_db, plain.margin_db)


def test_channel_validity_combines_fwm_flags() -> None:
    """``fwm_required`` 逻辑或；原因按顺序去重；模式取最保守。"""
    n = 4
    plain = _stub(fwm=False)
    low = _stub(fwm=True, fwm_reason="low_dispersion")
    zero = _stub(fwm=True, fwm_reason="zero_dispersion_in_band", fwm_mode="reject")
    mixed = _stub(fwm=True, fwm_reason="low_dispersion+dispersion_reference_out_of_band")

    assert ChannelValidity.combine([plain, plain], n).fwm_required is False
    verdict = ChannelValidity.combine([mixed, low, zero], n)
    assert verdict.fwm_required is True
    assert verdict.fwm_reason == (
        "low_dispersion+dispersion_reference_out_of_band+zero_dispersion_in_band"
    )
    assert verdict.fwm_validity_mode == "reject"
    # 没有 reject 时保持 flag。
    assert ChannelValidity.combine([low, mixed], n).fwm_validity_mode == "flag"
    # 空原因被忽略。
    assert ChannelValidity.combine([plain, low], n).fwm_reason == "low_dispersion"
    # combine 仍按原规则聚合 valid / margin。
    assert np.array_equal(verdict.valid, np.ones(n, dtype=bool))
    assert np.array_equal(verdict.margin_db, np.full(n, 3.0))


def test_link_validity_passes_fwm_into_the_policy(budget_pair) -> None:
    """链路报告把 ``fwm_required`` 交给策略；reject 只扣可用性。"""
    flagged_budget = budget_pair["flagged"]
    flagged = LinkValidity.from_budget(
        dataclasses.replace(
            flagged_budget,
            validity=dataclasses.replace(flagged_budget.validity, fwm_validity_mode="flag"),
        )
    )
    snr = budget_pair["snr"]
    audit = LinkValidityPolicy(mode="flag", fwm_action="reject")
    rejected = LinkValidityPolicy(mode="reject", fwm_action="reject")

    assert not bool(np.any(flagged.effective_valid(audit)))
    assert np.array_equal(flagged.adjusted_snr_db(snr, audit), snr)
    assert np.all(np.isneginf(flagged.adjusted_snr_db(snr, rejected)))


# ---------------------------------------------------------------------------
# 策略：flag / reject / 非法动作 / 正交 / 形状
# ---------------------------------------------------------------------------


def test_reject_action_only_changes_availability() -> None:
    """``reject`` 把命中信道的可用性扣成 ``False``，模式落地后才变 ``-inf``。"""
    valid = np.ones(4, dtype=bool)
    fwm = np.array([True, False, True, False])
    snr = np.array([20.0, 21.0, 22.0, 23.0])
    margin = np.ones(4)

    reject = LinkValidityPolicy(mode="reject", fwm_action="reject")
    assert np.array_equal(
        reject.effective_valid(valid, fwm_required=fwm),
        np.array([False, True, False, True]),
    )
    adjusted = reject.apply(snr, valid, margin, fwm_required=fwm)
    assert np.isneginf(adjusted[0]) and np.isneginf(adjusted[2])
    assert adjusted[1] == np.float64(21.0) and adjusted[3] == np.float64(23.0)

    audit = LinkValidityPolicy(mode="flag", fwm_action="reject")
    assert np.array_equal(audit.apply(snr, valid, margin, fwm_required=fwm), snr)


def test_unknown_fwm_action_is_rejected() -> None:
    """动作必须显式且合法，不允许静默退回默认。"""
    with pytest.raises(ValueError, match="unknown FWM action"):
        LinkValidityPolicy(fwm_action="maybe")


def test_invalid_fwm_validity_mode_is_rejected() -> None:
    """``LinkValidity`` / ``ChannelValidity`` 只接受 ``flag`` / ``reject``。"""
    with pytest.raises(ValueError, match="fwm_validity_mode"):
        _stub(fwm_mode="soft")
    with pytest.raises(ValueError, match="fwm_validity_mode"):
        ChannelValidity(
            valid=np.ones(2, dtype=bool),
            margin_db=np.zeros(2),
            fwm_validity_mode="off",
        )


def test_fwm_flag_shape_is_validated() -> None:
    """形状校验与 ``dense_grid_overlap`` 同风格。"""
    policy = LinkValidityPolicy(mode="reject", fwm_action="reject")
    valid = np.ones(4, dtype=bool)
    with pytest.raises(ValueError, match="fwm_required must be a scalar"):
        policy.effective_valid(valid, fwm_required=np.zeros((2, 2), dtype=bool))
    with pytest.raises(ValueError, match="fwm_required must be a scalar"):
        policy.effective_valid(valid, fwm_required=np.ones(3, dtype=bool))


def test_fwm_action_is_orthogonal_to_dense_and_raman() -> None:
    """三个动作各自只扣自己的标志，同时打开时按逻辑或合并。"""
    valid = np.ones(4, dtype=bool)
    dense = np.array([True, True, False, False])
    under = np.array([True, False, True, False])
    fwm = np.array([True, False, False, False])
    dense_only = LinkValidityPolicy(mode="reject", dense_grid_action="reject")
    raman_only = LinkValidityPolicy(mode="reject", raman_resolution_action="reject")
    fwm_only = LinkValidityPolicy(mode="reject", fwm_action="reject")
    all_three = LinkValidityPolicy(
        mode="reject",
        dense_grid_action="reject",
        raman_resolution_action="reject",
        fwm_action="reject",
    )

    np.testing.assert_array_equal(
        dense_only.effective_valid(
            valid, dense, raman_auto_under_resolved=under, fwm_required=fwm
        ),
        np.array([False, False, True, True]),
    )
    np.testing.assert_array_equal(
        raman_only.effective_valid(
            valid, dense, raman_auto_under_resolved=under, fwm_required=fwm
        ),
        np.array([False, True, False, True]),
    )
    np.testing.assert_array_equal(
        fwm_only.effective_valid(
            valid, dense, raman_auto_under_resolved=under, fwm_required=fwm
        ),
        np.array([False, True, True, True]),
    )
    np.testing.assert_array_equal(
        all_three.effective_valid(
            valid, dense, raman_auto_under_resolved=under, fwm_required=fwm
        ),
        np.array([False, False, False, True]),
    )
    # 关闭一个动作不会抹掉其他动作的效果。
    np.testing.assert_array_equal(
        all_three.effective_valid(
            valid, dense, raman_auto_under_resolved=under, fwm_required=False
        ),
        np.array([False, False, False, True]),
    )
    np.testing.assert_array_equal(
        all_three.effective_valid(
            valid, False, raman_auto_under_resolved=False, fwm_required=fwm
        ),
        np.array([False, True, True, True]),
    )


def test_channel_validity_carries_all_three_flags() -> None:
    """路径聚合同时保留三个标志，并交给同一份策略。"""
    verdict = ChannelValidity.combine(
        [_stub(dense=True), _stub(under_resolved=True), _stub(fwm=True)], 4
    )
    assert verdict.dense_grid_overlap is True
    assert verdict.raman_auto_under_resolved is True
    assert verdict.fwm_required is True
    policy = LinkValidityPolicy(
        mode="reject",
        dense_grid_action="reject",
        raman_resolution_action="reject",
        fwm_action="reject",
    )
    assert not bool(np.any(verdict.effective_valid(policy)))
    assert np.all(np.isneginf(verdict.adjusted_snr_db(np.full(4, 20.0), policy)))

    # 只打开 FWM 动作时，其他标志不参与。
    fwm_only = ChannelValidity.combine([_stub(), _stub(fwm=True)], 4)
    fwm_policy = LinkValidityPolicy(mode="reject", fwm_action="reject")
    assert not bool(np.any(fwm_only.effective_valid(fwm_policy)))
    assert np.array_equal(
        fwm_only.adjusted_snr_db(np.full(4, 20.0), LinkValidityPolicy(mode="reject")),
        np.full(4, 20.0),
    )


# ---------------------------------------------------------------------------
# 标量业务聚合
# ---------------------------------------------------------------------------


def test_scalar_service_aggregation_carries_fwm(budget_pair) -> None:
    """``ServiceQoT`` 从链路报告逻辑或，标量与逐信道走同一份策略语义。"""
    terms = fold_roadm_terms(
        ROADMConfig(insertion_loss_db=0.0, add_drop_loss_db=0.0), is_last_hop=True
    )
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

    plain = detail(LinkValidity.from_budget(budget_pair["plain"]))
    flagged = detail(LinkValidity.from_budget(budget_pair["flagged"]))
    assert plain.fwm_required is False
    assert flagged.fwm_required is True
    # 新增标志不改变物理 SNR 点值。
    assert flagged.snr_db == plain.snr_db

    reject = LinkValidityPolicy(mode="reject", fwm_action="reject")
    assert np.isneginf(flagged.effective_snr_db(reject))
    assert flagged.effective_snr_db(LinkValidityPolicy(mode="reject")) == flagged.snr_db
    assert plain.effective_snr_db(reject) == plain.snr_db
    assert flagged.effective_snr_db() == flagged.snr_db
