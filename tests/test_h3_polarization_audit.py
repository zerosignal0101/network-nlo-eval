"""偏振驱动倍率的口径守卫（任务 C 审计的回归测试）。

这些测试不改变任何默认数值，只把 ``signal_polarizations`` 的真实作用钉死：

* 它只放大**耦合的驱动功率**（泵浦耗尽与信道-信道 ISRS），不改信号从泵浦
  获得的增益，也不改 NLI 的偏振因子；
* 在无损耗单泵-单信号两波系统里，``=1`` 时泵浦光子损失等于信号光子增益
  （比值 1），``=2`` 时为 2 倍——后者只有在“输入数组是每偏振功率”时守恒。

因此 ``=2`` 不是一个可以独立于入纤功率口径切换的“更物理”开关。默认保持
``=1``；任何改默认的提案必须先显式声明基准算例的发射功率是总功率还是
每偏振功率。

任务 F 已把该声明落地为模型选项 ``launch_power_convention``：``"total"``
（默认）与 ``"per_pol"``。本文件的 ``=2`` 用例都在 ``"total"`` 下运行，
即历史倍率，因此会收到 warning，并用 :func:`pytest.warns` 固定下来。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.raman import C_LIGHT, RamanGainModel
from network_nlo_eval.models.raman_ode import RamanPaperSpanSolver
from network_nlo_eval.network.elements import RamanPumpConfig

#: 单泵 1450 nm / 0.2 W，无衰减，用于隔离拉曼光子转移。
_PUMP = RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="forward", attenuation_db_per_km=0.0)


def _two_wave_solver(signal_polarizations: int) -> RamanPaperSpanSolver:
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
    )


@pytest.mark.parametrize(
    ("signal_polarizations", "expected_ratio"),
    [(1, 1.0), (2, 2.0)],
    ids=["total_power_convention", "per_polarization_convention"],
)
def test_drive_polarizations_set_the_pump_photon_budget(signal_polarizations: int, expected_ratio: float) -> None:
    """泵浦光子损失 / 信号光子增益 = ``signal_polarizations``。

    ``=1`` 是“状态即每信道总功率”的守恒口径；``=2`` 把泵浦驱动翻倍，
    只有在“状态是每偏振功率、总功率 = 2×状态”时才与光子数守恒一致。
    该比值是判断能否改默认的核心预算，任何一侧漂移都应失败。
    """
    solver = _two_wave_solver(signal_polarizations)
    signal_in_w = 30e-3
    result = solver.solve(np.array([signal_in_w]))
    signal_photons = (float(result.power_out_w[0]) - signal_in_w) / 193.75e12
    pump_photons = (0.2 - float(result.pump_power_out_w[0])) / (C_LIGHT / 1450e-9)
    ratio = pump_photons / signal_photons
    assert ratio == pytest.approx(expected_ratio, rel=1e-5), f"photon budget ratio {ratio:.6f}"


def test_undepleted_pumps_without_isrs_ignore_signal_polarizations() -> None:
    """未耗尽 + 无 ISRS 时，偏振倍率没有可作用的耦合项，结果必须逐位一致。

    这条守住 ``signal_polarizations`` 的作用域：它只进入泵浦行（耗尽）和
    信号-信号块（ISRS）；不进入泵浦-信号增益、ASE 源项或 NLI 核。
    """
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case)
    pump = [RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="forward", attenuation_db_per_km=0.25)]
    options = {"raman_pumps": pump, "pump_depletion": False, "include_isrs": False}
    one = make_model("pcfm_h3_raman", case, signal_polarizations=1, **options).evaluate_path(path, power)
    with pytest.warns(UserWarning, match="launch_power_convention"):
        two = make_model("pcfm_h3_raman", case, signal_polarizations=2, **options).evaluate_path(path, power)
    for name in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        assert np.array_equal(np.asarray(getattr(one, name)), np.asarray(getattr(two, name))), name


def test_undepleted_pumps_with_isrs_change_only_the_isrs_term() -> None:
    """未耗尽时打开 ISRS，偏振倍率只移动 ISRS 项，信号功率仍逐位一致。

    平坦发射下 ISRS 净转移接近零，因此 NLI 的相对变化也应远小于 1e-3。
    这条同时说明：当前基准矩阵的 ISRS 影响小，是因为所有信道等功率，
    而不是因为 ISRS 开关无效。
    """
    case = build_case("scaling_40ch")
    path = build_path(case)
    power = launch_profile(case)
    pump = [RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="forward", attenuation_db_per_km=0.25)]
    options = {"raman_pumps": pump, "pump_depletion": False, "include_isrs": True}
    one = make_model("pcfm_h3_raman", case, signal_polarizations=1, **options).evaluate_path(path, power)
    with pytest.warns(UserWarning, match="launch_power_convention"):
        two = make_model("pcfm_h3_raman", case, signal_polarizations=2, **options).evaluate_path(path, power)
    assert np.array_equal(np.asarray(one.signal_power_w), np.asarray(two.signal_power_w))
    nli_one = np.asarray(one.spm_noise_w) + np.asarray(one.xpm_noise_w)
    nli_two = np.asarray(two.spm_noise_w) + np.asarray(two.xpm_noise_w)
    assert np.max(np.abs(nli_two - nli_one) / nli_one) < 1e-3
