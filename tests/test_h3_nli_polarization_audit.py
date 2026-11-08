"""NLI 功率口径审计的回归测试（任务 J）。

审计结论：PCFM 的 ``16/27``（SCI）与 ``32/27``（XCI）是**双偏振总功率**
口径的权重（``16/27 = (8/9)^2 * 6/2^3``），核的输入必须是物理总功率。
``per_pol`` 声明下的每偏振功率必须先乘 2 再进核，输出再减半回到每偏振
坐标，否则 NLI 会低 ``2**3 = 8`` 倍、而信号只低 2 倍。

本文件钉住四件事：

1. 默认 ``total`` + ``signal_polarizations=1`` 与改动前逐位一致（硬门槛，
   有泵/无泵都查）；
2. 相同物理总功率下 ``total``+1 与 ``per_pol``+2 给出同一 NLI（相差一个
   ``1/2`` 的每偏振输出因子）与同一信号，EGN 开关不改变该关系；
3. ``total``+2 对照与 ``per_pol``+2 在同一数组下只差 NLI 的 4 倍；
4. 剩余 SNR 残差来自 EDFA ASE 仍是双偏振总量：把 ASE 也减半后，两种坐标
   的 SNR 在 0.1 dB 内一致——这正是 ``signal_polarizations=2`` 改默认前
   NLI 侧之外还缺的那一项。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.pcfm import (
    NLI_POWER_CONVENTIONS,
    PCFMSpanKernel,
    resolve_nli_power_convention,
)
from network_nlo_eval.models.raman import RamanGainModel
from network_nlo_eval.network.elements import RamanPumpConfig

CASE = build_case("scaling_40ch")

#: 单泵 1450 nm / 0.25 W，无衰减、后向：EGN 的 auto 规则保持开启。
_PUMP = RamanPumpConfig(wavelength_nm=1450.0, power_w=0.25, direction="backward", attenuation_db_per_km=0.25)
_PUMPS = [_PUMP]

_OUTPUT_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


def _arrays(result) -> dict[str, np.ndarray]:
    """Return the five path-level arrays as ``float64``."""
    return {name: np.asarray(getattr(result, name), dtype=np.float64) for name in _OUTPUT_FIELDS}


def _nli(arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Total NLI (SPM+XPM) of one result."""
    return arrays["spm_noise_w"] + arrays["xpm_noise_w"]


def _evaluate(*, pumps=_PUMPS, egn="auto", **options):
    """Evaluate the single-span C-band benchmark with the given options."""
    opts = {"egn": egn}
    if pumps is not None:
        opts["raman_pumps"] = list(pumps)
    return make_model("pcfm_h3_raman", CASE, **opts, **options).evaluate_path(
        build_path(CASE), launch_profile(CASE)
    )


def _evaluate_at(power_scale: float, *, pumps=_PUMPS, egn="auto", **options):
    """Evaluate the benchmark with the launch power scaled by ``power_scale``."""
    opts = {"egn": egn}
    if pumps is not None:
        opts["raman_pumps"] = list(pumps)
    return make_model("pcfm_h3_raman", CASE, **opts, **options).evaluate_path(
        build_path(CASE), launch_profile(CASE) * power_scale
    )


# ---------------------------------------------------------------------------
# 默认逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pumps", [None, [_PUMP]], ids=["unpumped", "pumped"])
def test_default_total_plus_one_is_bit_identical_to_the_pre_audit_model(pumps) -> None:
    """不传新选项与显式 ``nli_power_convention="total"`` 逐位一致（硬门槛）。"""
    baseline = _arrays(_evaluate(pumps=pumps))
    explicit = _arrays(_evaluate(pumps=pumps, nli_power_convention="total"))
    for name in _OUTPUT_FIELDS:
        assert np.array_equal(baseline[name], explicit[name]), name


def test_plain_pcfm_default_is_bit_identical() -> None:
    """``pcfm`` 的默认路径也不得因为新增选项而改变。"""
    path = build_path(CASE)
    power = launch_profile(CASE)
    baseline = _arrays(make_model("pcfm", CASE).evaluate_path(path, power))
    explicit = _arrays(make_model("pcfm", CASE, nli_power_convention="total").evaluate_path(path, power))
    for name in _OUTPUT_FIELDS:
        assert np.array_equal(baseline[name], explicit[name]), name


# ---------------------------------------------------------------------------
# 相同物理总功率
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pumps", [None, [_PUMP]], ids=["unpumped", "pumped"])
def test_same_physical_total_power_gives_the_same_nli(pumps) -> None:
    """``total``+1 在 ``P`` 与 ``per_pol``+2 在 ``P/2`` 是同一物理链路。

    每偏振坐标下信号与 NLI 都是总量的一半；残差只来自粗 ODE 的自发源项
    不随偏振数拆分（<1e-4 相对）。
    """
    total = _arrays(_evaluate(pumps=pumps, signal_polarizations=1, launch_power_convention="total"))
    per_pol = _arrays(
        _evaluate_at(0.5, pumps=pumps, signal_polarizations=2, launch_power_convention="per_pol")
    )
    assert np.allclose(total["signal_power_w"], 2.0 * per_pol["signal_power_w"], rtol=1e-9)
    assert np.allclose(_nli(total), 2.0 * _nli(per_pol), rtol=1e-4)
    assert np.allclose(total["spm_noise_w"], 2.0 * per_pol["spm_noise_w"], rtol=1e-4)
    assert np.allclose(total["xpm_noise_w"], 2.0 * per_pol["xpm_noise_w"], rtol=1e-4)


def test_egn_does_not_change_the_two_convention_ratio() -> None:
    """EGN 的 rho 只依赖色散与符号率，不依赖功率，不能解释口径差。"""
    ratios = []
    for egn in (False, True):
        total = _arrays(_evaluate(egn=egn, signal_polarizations=1, launch_power_convention="total"))
        per_pol = _arrays(
            _evaluate_at(0.5, egn=egn, signal_polarizations=2, launch_power_convention="per_pol")
        )
        ratios.append(float(np.mean(_nli(total) / _nli(per_pol))))
    assert ratios[0] == pytest.approx(2.0, rel=1e-4)
    assert ratios[1] == pytest.approx(2.0, rel=1e-4)
    assert ratios[0] == pytest.approx(ratios[1], rel=1e-3)


def test_control_total_plus_two_is_four_times_below_the_per_pol_kernel() -> None:
    """同一数组下 ``total``+2 与 ``per_pol``+2 只差 NLI 口径的 4 倍。

    ODE 驱动相同（``2 * P/2 = P``），所以信号与 ASE 逐位一致；NLI 在
    ``per_pol`` 下用 ``2P`` 驱动并减半输出，是旧口径的 ``2**3 / 2 = 4`` 倍。
    """
    per_pol = _arrays(_evaluate_at(0.5, signal_polarizations=2, launch_power_convention="per_pol"))
    with pytest.warns(UserWarning, match="launch_power_convention"):
        legacy = _arrays(_evaluate_at(0.5, signal_polarizations=2, launch_power_convention="total"))
    for name in ("signal_power_w", "ase_noise_w"):
        assert np.array_equal(per_pol[name], legacy[name]), name
    for name in ("spm_noise_w", "xpm_noise_w"):
        assert np.allclose(per_pol[name], 4.0 * legacy[name], rtol=1e-4), name


# ---------------------------------------------------------------------------
# 剩余残差：EDFA ASE 口径
# ---------------------------------------------------------------------------


def test_the_remaining_snr_gap_is_the_edfa_ase_bookkeeping() -> None:
    """NLI 修好后，SNR 残差只剩 EDFA ASE 未按每偏振减半。

    ``route.apply_amplifier_and_roadm`` 的 ASE 是双偏振总量
    ``NF h nu B (G-1)``，而 ``per_pol`` 的信号与 NLI 都是每偏振量。把 ASE
    也减半后，两种坐标的逐信道 SNR 在 0.1 dB 内一致；不修 ASE 时残差约
    -2.4 dB（本算例 ASE 约为 NLI 的 2.7 倍）。
    """
    total = _arrays(_evaluate(pumps=None))
    per_pol = _arrays(_evaluate_at(0.5, pumps=None, signal_polarizations=2, launch_power_convention="per_pol"))
    snr_fixed = 10.0 * np.log10(per_pol["signal_power_w"] / (_nli(per_pol) + 0.5 * per_pol["ase_noise_w"]))
    assert np.max(np.abs(snr_fixed - total["snr_db"])) < 0.1
    naive_gap = float(np.mean(per_pol["snr_db"] - total["snr_db"]))
    assert naive_gap < -1.0


# ---------------------------------------------------------------------------
# 口径解析与非法组合
# ---------------------------------------------------------------------------


def test_auto_nli_convention_follows_the_launch_convention() -> None:
    """``auto`` 让 NLI 口径跟随已声明的 launch 口径。"""
    total = make_model("pcfm_h3_raman", CASE)
    assert total._launch_power_convention == "total"
    assert total._nli_power_convention == "total"
    per_pol = make_model("pcfm_h3_raman", CASE, signal_polarizations=2, launch_power_convention="per_pol")
    assert per_pol._launch_power_convention == "per_pol"
    assert per_pol._nli_power_convention == "per_pol"


def test_nli_per_pol_under_a_total_launch_is_rejected() -> None:
    """``per_pol`` NLI 配 ``total`` launch 会把总功率再乘 2，必须报错。"""
    with pytest.raises(ValueError, match="nli_power_convention='per_pol'"):
        make_model("pcfm_h3_raman", CASE, nli_power_convention="per_pol")
    from network_nlo_eval.models.pcfm_h3_raman import PCFH3RamanSpanKernel

    with pytest.raises(ValueError, match="nli_power_convention='per_pol'"):
        PCFH3RamanSpanKernel(
            np.array([193.75e12]),
            np.array([75e9]),
            build_path(CASE).fiber_configs[0],
            gain_model=RamanGainModel(),
            nli_power_convention="per_pol",
        )


@pytest.mark.parametrize("convention", ["per-pal", "perpol", "", "TOTAL", "auto"])
def test_unknown_nli_convention_is_rejected(convention: str) -> None:
    """口径只接受 ``total`` / ``per_pol``；``auto`` 只在模型/内核解析层有效。"""
    with pytest.raises(ValueError, match="nli_power_convention"):
        resolve_nli_power_convention(convention)
    with pytest.raises(ValueError, match="nli_power_convention"):
        make_model("pcfm", CASE, nli_power_convention=convention)


def test_plain_kernel_per_pol_scales_the_nli_by_four() -> None:
    """裸核：``per_pol`` 的 2P 驱动与输出减半合起来是同一数组的 4 倍。"""
    from network_nlo_eval.benchmarks.cases import grid_for

    grid = grid_for(CASE)
    fiber = build_path(CASE).fiber_configs[0]
    bandwidths = np.full(CASE.num_channels, CASE.symbol_rate_hz)
    power = launch_profile(CASE)
    total = PCFMSpanKernel(grid.frequencies, bandwidths, fiber).evaluate(power, 100e3)
    per_pol = PCFMSpanKernel(
        grid.frequencies, bandwidths, fiber, nli_power_convention="per_pol"
    ).evaluate(power, 100e3)
    assert np.array_equal(total.power_out_w, per_pol.power_out_w)
    assert np.allclose(per_pol.spm_noise_w, 4.0 * total.spm_noise_w, rtol=1e-12)
    assert np.allclose(per_pol.xpm_noise_w, 4.0 * total.xpm_noise_w, rtol=1e-12)


def test_nli_power_conventions_are_documented() -> None:
    """公开常量与模型元数据必须同步暴露新口径。"""
    assert NLI_POWER_CONVENTIONS == ("total", "per_pol")
    from network_nlo_eval.models.pcfm import PCFM_MODEL_INFO

    assert "nli_power_convention" in dict(PCFM_MODEL_INFO.options)
