"""入纤功率口径 ``launch_power_convention`` 的回归测试（任务 F）。

口径定义：

* ``"total"``（默认）= ``power_in_w`` 是每信道双偏振总功率，历史行为；
* ``"per_pol"`` = ``power_in_w`` 是每偏振功率，物理总功率是它的 2 倍，ODE
  状态与输出仍是每偏振功率，只有耦合驱动取双偏振总功率。

本文件的守卫分四类：

1. ``"total"`` + ``signal_polarizations=1`` 与历史逐位一致（默认不变）；
2. ``"per_pol"`` + ``signal_polarizations=2`` 的**物理**光子预算比值为 1
   （泵浦光子损失 / 双偏振信号光子增益），而 ``"total"`` + ``2`` 仍为 2；
3. ``"total"`` + ``2`` 是“未声明单位的倍率”，模型层必须 warning；
   ``"per_pol"`` + ``1`` 自相矛盾，必须报错；
4. 无泵时 ODE 没有可作用的耦合项；NLI 口径独立，``per_pol`` 默认按物理
   总功率驱动 GN 权重（同一数组下 NLI 4 倍），显式 ``"total"`` 才逐位一致。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm_h3_raman import H3_RAMAN_MODEL_INFO
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.network.elements import RamanPumpConfig

CASE = build_case("scaling_40ch")

#: 单泵 1450 nm / 0.2 W，无衰减，用于隔离拉曼光子转移。
_PUMP = RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="forward", attenuation_db_per_km=0.0)

#: 有泵路径选项（耗尽打开，ISRS 默认打开）。
PUMPED = {
    "raman_pumps": [_PUMP],
    "pump_depletion": True,
}

_OUTPUT_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _two_wave_solver(
    *,
    signal_polarizations: int = 1,
    launch_power_convention: str = "total",
) -> RamanPaperSpanSolver:
    """无损耗的单泵-单信号两波求解器。"""
    return RamanPaperSpanSolver(
        np.array([193.75e12]),
        np.array([75e9]),
        [_PUMP],
        100e3,
        np.zeros(1),
        np.array([80e-12]),
        np.zeros(1),
        np.array([80e-12]),
        RamanGainModel(),
        n_steps=64,
        signal_polarizations=signal_polarizations,
        launch_power_convention=launch_power_convention,
    )


def _photon_budget(solver: RamanPaperSpanSolver, signal_in_w: float) -> float:
    """泵浦光子损失 / (双偏振) 信号光子增益。

    ``launch_power_convention="per_pol"`` 时状态是每偏振功率，物理信号光子
    增益要乘 2；``"total"`` 时状态已经是总功率，不再乘。
    """
    result = solver.solve(np.array([signal_in_w]))
    signal_photons = (float(result.power_out_w[0]) - signal_in_w) / 193.75e12
    pump_photons = (0.2 - float(result.pump_power_out_w[0])) / (C_LIGHT / 1450e-9)
    if solver.launch_power_convention == "per_pol":
        signal_photons *= 2.0
    return pump_photons / signal_photons


# ---------------------------------------------------------------------------
# 默认不变：total + 单偏振
# ---------------------------------------------------------------------------


def test_total_convention_with_one_polarization_is_bit_identical_to_the_legacy_solver() -> None:
    """显式 ``"total"`` + ``1`` 与不带新参数的求解器逐位一致。"""
    legacy = _two_wave_solver()
    explicit = _two_wave_solver(launch_power_convention="total")
    power_in = np.array([30e-3])
    a = legacy.solve(power_in)
    b = explicit.solve(power_in)
    for name in ("power_out_w", "raman_ase_w", "pump_power_out_w"):
        assert np.array_equal(np.asarray(getattr(a, name)), np.asarray(getattr(b, name))), name
    assert np.array_equal(a.profile_norm, b.profile_norm)
    assert explicit.launch_power_convention == "total"
    assert explicit.signal_polarizations == 1


def test_model_total_convention_with_one_polarization_is_bit_identical_to_the_legacy_default() -> None:
    """模型层显式 ``"total"`` + ``1`` 与不传新参数的路径逐位一致（硬门槛）。"""
    path = build_path(CASE)
    power = launch_profile(CASE)
    baseline = make_model("pcfm_h3_raman", CASE, **PUMPED).evaluate_path(path, power)
    explicit = make_model(
        "pcfm_h3_raman", CASE, launch_power_convention="total", signal_polarizations=1, **PUMPED
    ).evaluate_path(path, power)
    for name in _OUTPUT_FIELDS:
        assert np.array_equal(np.asarray(getattr(baseline, name)), np.asarray(getattr(explicit, name))), name


# ---------------------------------------------------------------------------
# 光子预算
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("convention", "signal_polarizations", "expected_ratio"),
    [
        ("total", 1, 1.0),
        ("total", 2, 2.0),
        ("per_pol", 2, 1.0),
    ],
    ids=["total_pol1", "total_pol2_undeclared_multiplier", "per_pol_pol2"],
)
def test_photon_budget_ratio_follows_the_declared_convention(
    convention: str, signal_polarizations: int, expected_ratio: float
) -> None:
    """物理光子预算：只有 ``total``+1 与 ``per_pol``+2 守恒。

    ``per_pol`` 把状态保持为每偏振功率、驱动乘 2，所以物理信号光子增益要乘
    2 才与泵浦光子损失相等；``total``+2 则是把总功率再乘 2 驱动，比值 2 是
    重复计数，不是物理。
    """
    solver = _two_wave_solver(signal_polarizations=signal_polarizations, launch_power_convention=convention)
    ratio = _photon_budget(solver, 30e-3)
    assert ratio == pytest.approx(expected_ratio, rel=1e-5), f"photon budget ratio {ratio:.6f}"


def test_per_pol_drive_equals_the_total_power_of_the_same_physical_link() -> None:
    """per_pol 用 ``P/2`` 的驱动与 total 用 ``P`` 一致（只差单位坐标）。

    信号/泵浦剖面按每偏振功率等比缩放，泵浦输出在 1e-4 相对误差内相等；
    每偏振输出功率为总功率的一半。残差来自自发拉曼源项：泵浦驱动的 ASE
    不随信号偏振数拆分，所以两种坐标不是严格逐位，量级 <1e-5。
    这条钉住“只放大耦合驱动、不放大状态”的实现。
    """
    total = _two_wave_solver(launch_power_convention="total").solve(np.array([30e-3]))
    per_pol = _two_wave_solver(signal_polarizations=2, launch_power_convention="per_pol").solve(np.array([15e-3]))
    assert total.pump_power_out_w == pytest.approx(per_pol.pump_power_out_w, rel=1e-4)
    assert total.power_out_w == pytest.approx(2.0 * per_pol.power_out_w, rel=1e-6)
    assert total.profile_norm == pytest.approx(per_pol.profile_norm, rel=1e-6)


# ---------------------------------------------------------------------------
# warning / 非法组合
# ---------------------------------------------------------------------------


def test_total_with_two_polarizations_warns_about_the_undeclared_multiplier() -> None:
    """``"total"`` + ``pol2`` 不能静默：模型层必须明确 warning。"""
    with pytest.warns(UserWarning, match="launch_power_convention='per_pol'"):
        make_model("pcfm_h3_raman", CASE, signal_polarizations=2, **PUMPED)


def test_per_pol_with_two_polarizations_does_not_warn() -> None:
    """声明的 ``"per_pol"`` + ``pol2`` 是自洽组合，不应 warning。"""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = make_model("pcfm_h3_raman", CASE, signal_polarizations=2, launch_power_convention="per_pol", **PUMPED)
    assert [item for item in caught if issubclass(item.category, UserWarning)] == []
    assert model._launch_power_convention == "per_pol"


@pytest.mark.parametrize("convention", ["per-pal", "perpol", "", "TOTAL"])
def test_unknown_convention_is_rejected(convention: str) -> None:
    """口径只接受 ``"total"`` / ``"per_pol"``，拼写错误不得静默降级。"""
    with pytest.raises(ValueError, match="launch_power_convention"):
        make_model("pcfm_h3_raman", CASE, launch_power_convention=convention, **PUMPED)
    with pytest.raises(ValueError, match="launch_power_convention"):
        _two_wave_solver(launch_power_convention=convention)


def test_per_pol_requires_two_polarizations() -> None:
    """``per_pol`` 声明每偏振输入，状态却是单偏振，必须报错。"""
    with pytest.raises(ValueError, match="signal_polarizations must be 2"):
        make_model("pcfm_h3_raman", CASE, launch_power_convention="per_pol", **PUMPED)
    with pytest.raises(ValueError, match="signal_polarizations must be 2"):
        _two_wave_solver(signal_polarizations=1, launch_power_convention="per_pol")


# ---------------------------------------------------------------------------
# 无泵退化
# ---------------------------------------------------------------------------


def test_no_pump_path_is_bit_identical_across_conventions() -> None:
    """无泵时只有显式的旧 NLI 口径保持同一数组逐位一致。

    拉曼分支不进入，所以 ODE 层面两种口径没有可作用的耦合项；NLI 口径则
    独立：``nli_power_convention="auto"`` 让 ``per_pol`` 用物理总功率驱动
    GN 权重（``2P`` 驱动、输出减半），同一数组下 NLI 变成 4 倍。显式
    ``"total"`` 复现改动前的逐位一致，是对照口径而非默认行为。
    """
    path = build_path(CASE)
    power = launch_profile(CASE)
    baseline = make_model("pcfm_h3_raman", CASE).evaluate_path(path, power)
    legacy_nli = make_model(
        "pcfm_h3_raman",
        CASE,
        signal_polarizations=2,
        launch_power_convention="per_pol",
        nli_power_convention="total",
    ).evaluate_path(path, power)
    per_pol = make_model(
        "pcfm_h3_raman", CASE, signal_polarizations=2, launch_power_convention="per_pol"
    ).evaluate_path(path, power)
    with pytest.warns(UserWarning):
        legacy_multiplier = make_model("pcfm_h3_raman", CASE, signal_polarizations=2).evaluate_path(path, power)
    for name in _OUTPUT_FIELDS:
        assert np.array_equal(np.asarray(getattr(baseline, name)), np.asarray(getattr(legacy_nli, name))), name
        assert np.array_equal(np.asarray(getattr(baseline, name)), np.asarray(getattr(legacy_multiplier, name))), name
    # auto 口径：信号与 ASE 仍是同一数组，只有 NLI 按 2**3/2 = 4 倍变化。
    assert np.array_equal(np.asarray(baseline.signal_power_w), np.asarray(per_pol.signal_power_w))
    assert np.array_equal(np.asarray(baseline.ase_noise_w), np.asarray(per_pol.ase_noise_w))
    for name in ("spm_noise_w", "xpm_noise_w"):
        assert np.allclose(
            np.asarray(getattr(per_pol, name)), 4.0 * np.asarray(getattr(baseline, name)), rtol=1e-12
        ), name


def test_no_pump_solver_without_isrs_is_bit_identical_across_conventions() -> None:
    """无泵 + 关 ISRS 时求解器没有任何耦合项，两种口径逐位一致。"""
    args = (
        np.array([193.75e12]),
        np.array([75e9]),
        [],
        100e3,
        np.zeros(1),
        np.array([80e-12]),
        np.zeros(0),
        np.zeros(0),
        RamanGainModel(),
    )
    total = RamanPaperSpanSolver(*args, n_steps=8, include_isrs=False)
    per_pol = RamanPaperSpanSolver(
        *args, n_steps=8, include_isrs=False, signal_polarizations=2, launch_power_convention="per_pol"
    )
    power_in = np.array([1e-3])
    a = total.solve(power_in)
    b = per_pol.solve(power_in)
    assert np.array_equal(a.power_out_w, b.power_out_w)
    assert np.array_equal(a.profile_norm, b.profile_norm)


# ---------------------------------------------------------------------------
# 数值等价与元数据
# ---------------------------------------------------------------------------


def test_per_pol_and_total_with_two_polarizations_share_the_ode_state() -> None:
    """同一数组下 ``per_pol``+2 与 ``total``+2 的 ODE 完全相同。

    口径只改变输入的**声明含义**（因此 ``total``+2 被 warning 标记为重复
    计数），不改变同一数组算出来的信号与 ASE。NLI 口径独立：``per_pol``
    默认按物理总功率驱动 GN 权重，所以 NLI 是 ``4 = 2**3 / 2`` 倍。
    """
    path = build_path(CASE)
    power = launch_profile(CASE)
    per_pol = make_model(
        "pcfm_h3_raman", CASE, signal_polarizations=2, launch_power_convention="per_pol", **PUMPED
    ).evaluate_path(path, power)
    with pytest.warns(UserWarning):
        total = make_model("pcfm_h3_raman", CASE, signal_polarizations=2, **PUMPED).evaluate_path(path, power)
    for name in ("signal_power_w", "ase_noise_w"):
        assert np.array_equal(np.asarray(getattr(per_pol, name)), np.asarray(getattr(total, name))), name
    for name in ("spm_noise_w", "xpm_noise_w"):
        assert np.allclose(
            np.asarray(getattr(per_pol, name)), 4.0 * np.asarray(getattr(total, name)), rtol=1e-12
        ), name


def test_explicit_legacy_nli_convention_restores_identical_arrays() -> None:
    """显式 ``nli_power_convention="total"`` 复现改动前的同数组逐位一致。

    这条把旧对照口径保留下来：NLI 口径是独立声明，不与 launch 口径强行
    绑定；改动只发生在默认 ``"auto"`` 选择的那一侧。
    """
    path = build_path(CASE)
    power = launch_profile(CASE)
    per_pol = make_model(
        "pcfm_h3_raman",
        CASE,
        signal_polarizations=2,
        launch_power_convention="per_pol",
        nli_power_convention="total",
        **PUMPED,
    ).evaluate_path(path, power)
    with pytest.warns(UserWarning):
        total = make_model("pcfm_h3_raman", CASE, signal_polarizations=2, **PUMPED).evaluate_path(path, power)
    for name in _OUTPUT_FIELDS:
        assert np.array_equal(np.asarray(getattr(per_pol, name)), np.asarray(getattr(total, name))), name


def test_option_metadata_documents_the_convention() -> None:
    """模型 options 元数据必须同步暴露新口径。"""
    options = dict(H3_RAMAN_MODEL_INFO.options)
    assert "launch_power_convention" in options
    text = options["launch_power_convention"]
    assert "'total'" in text and "'per_pol'" in text
    assert "signal_polarizations=2" in text
    assert "nli_power_convention" in options
    nli_text = options["nli_power_convention"]
    assert "'auto'" in nli_text and "'total'" in nli_text and "'per_pol'" in nli_text
