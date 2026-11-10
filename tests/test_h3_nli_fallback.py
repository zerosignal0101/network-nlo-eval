"""任务 AW：``nli_fallback="numerical"`` 的参考回退验收测试。

覆盖契约：

* 默认 ``nli_fallback=None``：C / C+L / S+C+L / O 的点值逐位不变，且不构造
  数值核（惰性 import 也不发生）；
* 只有 flagged span 触发：O 波段低色散、C 37.5/25 GHz 密集栅格；C 100 GHz 与
  C+L 75 GHz 不触发且逐位不变；
* 触发后 O 波段 / 密集 C 的 NLI 与 SNR 误差都下降（参考一律是 ``numerical``）；
* 有泵 H3 在构造期拒绝 ``nli_fallback``；
* ``per_pol`` 口径下 NLI 与 ``0.5 * numerical(2P)`` 一致；
* ``(span_length, power)`` 缓存与诊断字段（应用标志、NLI 快照、次数、耗时）。

参考只跑 ``network_nlo_eval.models.numerical``，不读 ``data/`` 或 ``results/``。
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    BenchCase,
    band_edges_hz,
    build_case,
    build_path,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.pcfm import PCFMSpanKernel
from network_nlo_eval.models.pcfm_h3_raman import PCFH3RamanSpanKernel
from network_nlo_eval.network.elements import EDFAConfig, RamanPumpConfig

C_LIGHT = 299_792_458.0
O_SYMBOL_RATE_HZ = 96e9
_RESULT_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


# ---------------------------------------------------------------------------
# 算例构造
# ---------------------------------------------------------------------------


def _o_fiber() -> object:
    """O 波段合成光纤：1310 nm 锚点、D=2.6 ps/nm/km（与 AP 同一根光纤）。"""
    return SMF_100KM.model_copy(
        update={
            "reference_wavelength_nm": 1310.0,
            "dispersion_parameter_d": 2.6,
            "dispersion_slope_s": 0.06,
            "use_attenuation_polynomial": False,
            "attenuation_db_km_ref": 0.35,
            "effective_area_um2_ref": 80.0,
            "aeff_slope_um2_nm": 0.05,
        }
    )


def _o_grid(num_channels: int = 41) -> SpectrumGrid:
    """16.1 THz 占用带、中心 229.15 THz 的 O 波段栅格（161 ch → 100.6 GHz）。"""
    return SpectrumGrid(num_channels, 16.1e12 / (num_channels - 1), 229.15e12)


def _dense_case(spacing_hz: float, span_count: int) -> BenchCase:
    """C 波段密集栅格算例：64 GBaud、100 km/跨、NF 5.5 dB。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"aw_{spacing_hz / 1e9:g}_{span_count}",
        bands="C",
        num_channels=int(round((high - low) / spacing_hz)) + 1,
        channel_spacing_hz=spacing_hz,
        center_frequency_hz=0.5 * (low + high),
        symbol_rate_hz=64e9,
        span_length_km=100.0,
        span_count=span_count,
        edfa_noise_figure_db=5.5,
        total_power_dbm=0.0,
    )


def _o_model(grid: SpectrumGrid, fiber, **options):
    return create_model(
        "pcfm_h3_raman",
        grid,
        fiber,
        symbol_rate_hz=O_SYMBOL_RATE_HZ,
        poly_degree=3,
        egn=True,
        **options,
    )


def _o_path(fiber, span_count: int = 1) -> MultiSpanOpticalPath:
    return MultiSpanOpticalPath(
        [fiber] * span_count,
        [EDFAConfig(noise_figure_db=5.5)] * span_count,
    )


def _nli(result) -> np.ndarray:
    """路由层累积的 NLI（spm+xpm），两个模型都按这个口径比较。"""
    return np.asarray(result.spm_noise_w) + np.asarray(result.xpm_noise_w)


def _max_delta_db(values, reference) -> float:
    return float(np.max(np.abs(10.0 * np.log10(np.asarray(values) / np.asarray(reference)))))


def _assert_bitwise_equal(left, right) -> None:
    for name in _RESULT_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


# ---------------------------------------------------------------------------
# 默认路径：逐位不变、不构造数值核
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", ["c_57ch", "cl_152ch", "scl_277ch", "escl_479ch_1span"])
def test_default_fallback_is_bit_identical(case_name: str) -> None:
    """``nli_fallback=None`` 与显式 ``None`` 在 C/C+L/S+C+L 上逐位相同。"""
    case = build_case(case_name)
    path = build_path(case)
    power = launch_profile(case)
    plain = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    explicit = make_model("pcfm_h3_raman", case, nli_fallback=None)
    result = explicit.evaluate_path(path, power)

    _assert_bitwise_equal(plain, result)
    assert explicit.last_nli_fallback_applied is False
    assert explicit.last_fallback_nli_w is None
    assert explicit.fallback_evaluations == 0
    assert explicit.fallback_seconds == 0.0
    for kernel in explicit._kernel_cache.values():
        assert kernel._fallback_kernel is None


def test_default_path_does_not_import_the_numerical_kernel() -> None:
    """只 import ``pcfm`` 并构造 kernel 时，``numerical`` 模块不得进入 ``sys.modules``。"""
    code = (
        "import sys, numpy as np\n"
        "from network_nlo_eval.models.pcfm import PCFMSpanKernel\n"
        "from network_nlo_eval.benchmarks.cases import SMF_100KM\n"
        "f = np.linspace(195.9e12, 191.6e12, 8)\n"
        "b = np.full(8, 50e9)\n"
        "k = PCFMSpanKernel(f, b, SMF_100KM)\n"
        "k.evaluate(np.full(8, 1e-3), 80e3)\n"
        "assert 'network_nlo_eval.models.numerical' not in sys.modules\n"
        "assert k._fallback_kernel is None\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script, no user input.
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


# ---------------------------------------------------------------------------
# 触发条件
# ---------------------------------------------------------------------------


def test_o_band_is_flagged_and_fallback_fires() -> None:
    """O 波段（零色散 + 低色散）必须触发，并把 NLI 换成 numerical 的 spm+xpm。"""
    fiber = _o_fiber()
    grid = _o_grid(41)
    power = np.full(41, 1e-3)
    path = _o_path(fiber)
    model = _o_model(grid, fiber, nli_fallback="numerical")
    result = model.evaluate_path(path, power)

    assert model.last_nli_fallback_applied is True
    assert model.fallback_evaluations == 1
    assert model.last_fallback_nli_w is not None
    assert np.all(np.asarray(model.last_fallback_nli_w) > 0.0)
    assert np.all(np.isfinite(result.snr_db))


@pytest.mark.parametrize("case_name", ["c_57ch", "cl_152ch"])
def test_normal_grids_do_not_trigger_and_stay_bit_identical(case_name: str) -> None:
    """C 100 GHz 与 C+L 75 GHz 不是 flagged span：不触发、数值逐位不变。"""
    case = build_case(case_name)
    path = build_path(case)
    power = launch_profile(case)
    plain = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
    fallback = make_model("pcfm_h3_raman", case, nli_fallback="numerical")

    result = fallback.evaluate_path(path, power)
    _assert_bitwise_equal(plain, result)
    assert fallback.last_nli_fallback_applied is False
    assert fallback.fallback_evaluations == 0
    for kernel in fallback._kernel_cache.values():
        assert kernel._fallback_kernel is None


# ---------------------------------------------------------------------------
# 精度：O 波段与密集 C
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def o_band_6dbm() -> dict[str, object]:
    """O 波段 161 ch、80 km、6 dBm：闭式 / 回退 / numerical 三点。"""
    fiber = _o_fiber()
    grid = _o_grid(161)
    power = np.full(161, 1e-3 * 10 ** (6.0 / 10.0))
    path = _o_path(fiber)
    reference = create_model("numerical", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ).evaluate_path(path, power)
    base = _o_model(grid, fiber).evaluate_path(path, power)
    model = _o_model(grid, fiber, nli_fallback="numerical")
    fallback = model.evaluate_path(path, power)
    return {"reference": reference, "base": base, "fallback": fallback, "model": model}


def test_o_band_error_drops_below_budget(o_band_6dbm: dict[str, object]) -> None:
    """O 波段 6 dBm：NLI 误差从闭式的 ~0.8 dB 降到 0，SNR 也在 0.1 dB 内。"""
    reference = o_band_6dbm["reference"]
    base = o_band_6dbm["base"]
    fallback = o_band_6dbm["fallback"]

    reference_nli = _nli(reference)
    assert _max_delta_db(_nli(base), reference_nli) > 0.5
    assert _max_delta_db(_nli(fallback), reference_nli) < 1e-9
    assert float(np.max(np.abs(np.asarray(fallback.snr_db) - np.asarray(reference.snr_db)))) < 0.1


def test_o_band_zero_dbm_residual_is_the_auto_spp_not_the_fallback() -> None:
    """0 dBm 的 SNR 残差来自 auto 12 步 SPP（剖面），不是数值核：显式 32 步即消失。"""
    fiber = _o_fiber()
    grid = _o_grid(81)
    power = np.full(81, 1e-3)
    path = _o_path(fiber)
    reference = create_model("numerical", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ).evaluate_path(path, power)

    auto = _o_model(grid, fiber, nli_fallback="numerical").evaluate_path(path, power)
    fine = _o_model(grid, fiber, nli_fallback="numerical", spp_steps=32).evaluate_path(path, power)

    assert _max_delta_db(_nli(auto), _nli(reference)) < 1e-9
    auto_delta = float(np.max(np.abs(np.asarray(auto.snr_db) - np.asarray(reference.snr_db))))
    fine_delta = float(np.max(np.abs(np.asarray(fine.snr_db) - np.asarray(reference.snr_db))))
    # auto 在 <=0 dBm 用 12 步：跨末剖面差 ~0.08 dB，经 EDFA 增益进入 ASE。
    assert auto_delta < 0.2
    assert fine_delta < 0.02
    assert fine_delta < auto_delta


@pytest.fixture(scope="module")
def dense_c() -> dict[str, dict[str, object]]:
    """密集 C：37.5 / 25 GHz、1/5/10 跨、6 dBm。"""
    out: dict[str, dict[str, object]] = {}
    for spacing in (37.5e9, 25e9):
        for spans in (1, 5, 10):
            case = _dense_case(spacing, spans)
            path = build_path(case)
            power = launch_profile(case, 6.0)
            reference = make_model("numerical", case).evaluate_path(path, power)
            base = make_model("pcfm_h3_raman", case).evaluate_path(path, power)
            model = make_model("pcfm_h3_raman", case, nli_fallback="numerical")
            out[f"{spacing / 1e9:g}_{spans}"] = {
                "reference": reference,
                "base": base,
                "fallback": model.evaluate_path(path, power),
                "model": model,
            }
    return out


@pytest.mark.parametrize("key", ["37.5_1", "37.5_5", "37.5_10", "25_1", "25_5", "25_10"])
def test_dense_c_error_drops_below_budget(dense_c: dict[str, dict[str, object]], key: str) -> None:
    """密集 C 的 NLI/SNR 误差都降到 0.1 dB 内（闭式是 2.5-6.6 dB）。"""
    entry = dense_c[key]
    reference_nli = _nli(entry["reference"])
    assert _max_delta_db(_nli(entry["base"]), reference_nli) > 1.0
    assert _max_delta_db(_nli(entry["fallback"]), reference_nli) < 1e-9
    assert float(np.max(np.abs(np.asarray(entry["fallback"].snr_db) - np.asarray(entry["reference"].snr_db)))) < 0.1
    assert entry["model"].last_nli_fallback_applied is True


# ---------------------------------------------------------------------------
# 有泵 H3 拒绝
# ---------------------------------------------------------------------------


def _pump() -> RamanPumpConfig:
    return RamanPumpConfig(wavelength_nm=1450.0, power_w=0.3, direction="backward")


def test_pumped_h3_rejects_the_fallback_at_construction() -> None:
    """有泵链路必须在模型与 kernel 两层构造期 ValueError。"""
    case = build_case("c_57ch")
    with pytest.raises(ValueError, match="nli_fallback"):
        make_model("pcfm_h3_raman", case, raman_pumps=[_pump()], nli_fallback="numerical")

    # 光纤自带泵浦（未显式给 raman_pumps）同样必须在构造期拒绝。
    pumped_fiber = SMF_100KM.model_copy(update={"raman_pumps": [_pump()]})
    with pytest.raises(ValueError, match="nli_fallback"):
        create_model(
            "pcfm_h3_raman",
            SpectrumGrid(8, 50e9, 193.1e12),
            pumped_fiber,
            nli_fallback="numerical",
        )

    with pytest.raises(ValueError, match="nli_fallback"):
        PCFH3RamanSpanKernel(
            np.linspace(195.9e12, 191.6e12, 8),
            np.full(8, 50e9),
            pumped_fiber,
            pumps=[_pump()],
            gain_model="gaussian",
            nli_fallback="numerical",
        )


def test_unpumped_h3_accepts_the_fallback() -> None:
    """无泵 H3 走父类路径：可以构造并触发。"""
    fiber = _o_fiber()
    model = _o_model(_o_grid(41), fiber, nli_fallback="numerical")
    model.evaluate_path(_o_path(fiber), np.full(41, 1e-3))
    assert model.last_nli_fallback_applied is True


# ---------------------------------------------------------------------------
# per_pol 与缓存、诊断
# ---------------------------------------------------------------------------


def test_per_pol_matches_half_of_the_total_power_reference() -> None:
    """per_pol：回退 NLI = 0.5 * numerical(2P)，诊断仍是 total 口径。"""
    fiber = _o_fiber()
    grid = _o_grid(41)
    per_pol_power = np.full(41, 0.5e-3)
    path = _o_path(fiber)
    model = _o_model(
        grid,
        fiber,
        launch_power_convention="per_pol",
        signal_polarizations=2,
        nli_fallback="numerical",
    )
    result = model.evaluate_path(path, per_pol_power)
    reference = create_model("numerical", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ).evaluate_path(
        path, 2.0 * per_pol_power
    )

    np.testing.assert_allclose(
        _nli(result),
        0.5 * _nli(reference),
        rtol=0.0,
        atol=1e-30,
    )
    np.testing.assert_allclose(
        np.asarray(model.last_fallback_nli_w),
        _nli(reference),
        rtol=0.0,
        atol=1e-30,
    )
    assert np.all(np.isfinite(result.snr_db))


def test_fallback_is_cached_per_span_state_and_reports_cost() -> None:
    """透明链路 5 跨只付一次双积分；诊断字段与缓存命中一致。"""
    case = _dense_case(37.5e9, 5)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    model = make_model("pcfm_h3_raman", case, nli_fallback="numerical")

    first = model.evaluate_path(path, power)
    assert model.fallback_evaluations == 1
    assert model.fallback_seconds > 0.0
    assert model.last_nli_fallback_applied is True
    assert np.asarray(model.last_fallback_nli_w).shape == (case.num_channels,)

    second = model.evaluate_path(path, power)
    # 缓存是 (跨长, 功率) 的纯函数，第二条路径仍只付一次。
    assert model.fallback_evaluations == 1
    _assert_bitwise_equal(first, second)


def test_invalid_fallback_value_is_rejected() -> None:
    """非法值在 kernel 与 model 两层都被拒绝。"""
    fiber = SMF_100KM
    with pytest.raises(ValueError, match="nli_fallback"):
        PCFMSpanKernel(
            np.linspace(195.9e12, 191.6e12, 8),
            np.full(8, 50e9),
            fiber,
            nli_fallback="fwm",
        )
    with pytest.raises(ValueError, match="nli_fallback"):
        create_model("pcfm", SpectrumGrid(8, 50e9, 193.1e12), fiber, nli_fallback="numerical ")
    with pytest.raises(ValueError, match="nli_fallback"):
        create_model(
            "pcfm_h3_raman",
            SpectrumGrid(8, 50e9, 193.1e12),
            fiber,
            nli_fallback="on",
        )
