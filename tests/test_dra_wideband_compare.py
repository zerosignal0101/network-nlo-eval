"""``scripts/dra_wideband_compare.py`` 的 plan/fiber 解析测试。

不跑 57/152ch：用 3ch 收敛参考与合成 plan 检查解析口径。
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONVERGED = ROOT / "data" / "dra_waveform_converged" / "dra_reference.json"


def _load(name: str) -> ModuleType:
    """按脚本目录加载模块（脚本不是包）。"""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def wideband() -> ModuleType:
    """被测模块。"""
    return _load("dra_wideband_compare")


def test_plan_from_wideband_block(wideband: ModuleType) -> None:
    """解析时优先使用 plan 段，且保留 64 GBd / 75 GHz。"""
    reference = {
        "plan": {
            "name": "c57",
            "n_channels": 57,
            "channel_spacing_hz": 75e9,
            "center_frequency_hz": 193.75e12,
            "symbol_rate_hz": 64e9,
            "launch_power_dbm_per_pol": 1.0,
        },
        "datasets": {"c57": {"n_channels": 57, "configs": {}}},
    }
    plan = wideband.plan_from_reference(reference)
    assert plan["n_channels"] == 57
    assert plan["channel_spacing_hz"] == pytest.approx(75e9)
    assert plan["center_frequency_hz"] == pytest.approx(193.75e12)
    assert plan["symbol_rate_hz"] == pytest.approx(64e9)
    assert plan["launch_power_dbm_per_pol"] == pytest.approx(1.0)


def test_datasets_from_top_level_configs(wideband: ModuleType) -> None:
    """WP7 全 WDM schema：几何在 plan、配置在顶层 configs。"""
    reference = {
        "plan": {
            "name": "c57",
            "n_channels": 57,
            "channel_spacing_hz": 75e9,
            "center_frequency_hz": 193.75e12,
            "symbol_rate_hz": 64e9,
            "launch_power_dbm_per_pol": 0.0,
        },
        "configs": {"nopump": {"pumps": [], "per_channel": []}},
    }
    datasets = wideband._datasets_from_reference(reference)
    assert list(datasets) == ["c57"]
    assert datasets["c57"]["n_channels"] == 57
    assert datasets["c57"]["center_channel_index"] == 28
    assert datasets["c57"]["configs"]["nopump"]["pumps"] == []


def test_plan_from_legacy_3ch_reference(wideband: ModuleType) -> None:
    """3ch 收敛参考没有 plan：从 dataset/signal/频偏回退。"""
    reference = json.loads(CONVERGED.read_text(encoding="utf-8"))
    plan = wideband.plan_from_reference(reference)
    assert plan["n_channels"] == 3
    assert plan["channel_spacing_hz"] == pytest.approx(100e9)
    assert plan["center_frequency_hz"] == pytest.approx(193.1e12)
    assert plan["symbol_rate_hz"] == pytest.approx(50e9)
    assert plan["launch_power_dbm_per_pol"] == pytest.approx(0.0)


def test_plan_requires_channel_count(wideband: ModuleType) -> None:
    """缺 n_channels 时给出明确退出，而不是猜。"""
    with pytest.raises(SystemExit, match="n_channels"):
        wideband.plan_from_reference({"datasets": {"x": {"configs": {}}}})


def test_beta2_to_dispersion_matches_reference(wideband: ModuleType) -> None:
    """β2=-20 ps²/km 在 1550 nm 对应 D≈15.68。"""
    dispersion = wideband._beta2_to_dispersion(-20.0, 1550.0)
    assert dispersion == pytest.approx(15.6808, abs=1e-3)


def test_beta3_to_slope_round_trips_network_kernel(wideband: ModuleType) -> None:
    """β3↔S 换算与 network-nlo-eval 的 ``_calc_beta3_from_s_d`` 互逆。"""
    from network_nlo_eval.models.numba_kernels import _calc_beta3_from_s_d

    dispersion = 17.0
    beta3 = _calc_beta3_from_s_d(0.06, dispersion, 1550e-9)
    slope = wideband._beta3_to_slope(beta3 * 1e39, dispersion, 1550.0)
    assert slope == pytest.approx(0.06, abs=1e-9)


def test_fiber_derives_slope_from_beta3(wideband: ModuleType) -> None:
    """波形只给 β3=1e-4 ps³/km 时，解析侧斜率必须匹配它而不是默认 0.06。"""
    reference = {
        "fiber": {
            "length_km": 100.0,
            "alpha_db_per_km": 0.162,
            "beta2_ps2_per_km": -21.7,
            "beta3_ps3_per_km": 1e-4,
            "n2_si": 2.6e-20,
            "a_eff_um2": 80.0,
        }
    }
    fiber, metadata = wideband.fiber_from_reference(reference)
    assert metadata["dispersion_slope_source"] == "beta3_ps3_per_km"
    assert fiber.dispersion_slope_s == pytest.approx(-0.021892, abs=1e-5)


def test_fiber_defaults_are_flat_wideband(wideband: ModuleType) -> None:
    """没有 fiber 段时用 D=17、S=0.06、α=0.162 的平坦近似。"""
    fiber, metadata = wideband.fiber_from_reference({"plan": {"n_channels": 57}})
    assert fiber.use_attenuation_polynomial is False
    assert fiber.attenuation_db_km_ref == pytest.approx(0.162)
    assert fiber.dispersion_parameter_d == pytest.approx(17.0)
    assert fiber.dispersion_slope_s == pytest.approx(0.06)
    assert fiber.effective_area_um2_ref == pytest.approx(80.0)
    assert fiber.nonlinear_index_n2 == pytest.approx(2.6e-20)
    assert metadata["attenuation_model"] == "flat"
    assert metadata["dispersion_source"] == "default"


def test_fiber_derives_dispersion_from_beta2(wideband: ModuleType) -> None:
    """只给 β2 时按 D = -(2πc/λ²)β2 换算，并标出来源。"""
    reference = {
        "fiber": {
            "length_km": 100.0,
            "alpha_db_per_km": 0.162,
            "beta2_ps2_per_km": -21.7,
            "n2_si": 2.6e-20,
            "a_eff_um2": 80.0,
        }
    }
    fiber, metadata = wideband.fiber_from_reference(reference)
    assert fiber.dispersion_parameter_d == pytest.approx(17.0136, abs=1e-3)
    assert metadata["dispersion_source"] == "beta2_ps2_per_km"


def test_pumps_from_six_pump_config(wideband: ModuleType) -> None:
    """多泵后向配置逐项读取，不依赖配置名。"""
    config = {
        "pumps": [
            {"wavelength_nm": 1340.0, "power_w": 0.63, "direction": "backward", "attenuation_db_per_km": 0.25},
            {"wavelength_nm": 1505.0, "power_w": 0.075, "direction": "backward", "attenuation_db_per_km": 0.25},
        ]
    }
    pumps = wideband.pumps_from_config("bwd_sixpumps", config, {})
    assert [pump.wavelength_nm for pump in pumps] == [1340.0, 1505.0]
    assert [pump.power_w for pump in pumps] == [0.63, 0.075]


def test_pumps_legacy_name_fallback(wideband: ModuleType) -> None:
    """没有 pumps 字段时回退到 fwd_0p20w 的名字解析。"""
    pumps = wideband.pumps_from_config("fwd_0p20w", {}, {})
    assert len(pumps) == 1
    assert pumps[0].power_w == pytest.approx(0.2)
    assert pumps[0].direction == "forward"
    assert wideband.pumps_from_config("nopump", {}, {}) == []
