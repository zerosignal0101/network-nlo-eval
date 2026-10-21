"""Tests for the coherent ECZ model (``network_nlo_eval.models.ecz``).

The coherent model lives in its own module so that the frozen ECZ baseline
(``models/isrs_gn.py``) keeps its bit-exact golden guard.  This module therefore
carries three kinds of evidence:

* the **single-span identity** -- one span must be bit-identical to the frozen
  baseline (hard-coded numbers plus a direct class-to-class comparison);
* the **new golden baseline** -- ``fixtures/ecz_golden_mm_coh_ecz.json`` is the
  re-captured baseline of the coherent model, while the single-span cases of the
  *old* fixture must still match it bit-for-bit;
* the **coherence physics** -- buildup inside the band required by the work
  package, the zero-dispersion limit, the amplitude/noise contract and a
  synthetic-kernel route test that pins the ``|sum a_k|**2`` accumulation of the
  route layer without depending on any physical model.

Run ``python -m tests.test_ecz --capture`` to re-capture the golden fixture.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import SMF_100KM, build_case, launch_profile
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid, bits_per_symbol
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    ElementOutput,
    MultiSpanOpticalPath,
    SpanContext,
)
from network_nlo_eval.models.ecz import (
    ECZ_COHERENT_MODEL_INFO,
    ECZCoherentModel,
    _coherent_factor,
    _cross_beating_visibility,
    _equivalent_phase_ramp,
    _self_beating_visibility,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
from network_nlo_eval.models.route import accumulate_route
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig

FIXTURE = Path(__file__).parent / "fixtures" / "ecz_golden_mm_coh_ecz.json"
LEGACY_FIXTURE = Path(__file__).parent / "fixtures" / "ecz_golden_bbb9ed4.json"

#: Work-package band for the ``c_57ch`` ten-span SPM buildup.
BUILDUP_LOW = 1.37
BUILDUP_HIGH = 1.43


def _bits(array: Any) -> np.ndarray:
    """View a float array as raw IEEE-754 bits."""
    return np.asarray(array, dtype=np.float64).view(np.int64)


def _assert_bit_equal(expected: Any, actual: Any, label: str) -> None:
    """Assert two float arrays are bit-for-bit identical.

    Args:
        expected: The expected values.
        actual: The computed values.
        label: Label used in the failure message.
    """
    expected_array = np.asarray(expected, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=np.float64)
    assert expected_array.shape == actual_array.shape, f"{label}: shape {expected_array.shape} != {actual_array.shape}"
    mismatch = _bits(expected_array) != _bits(actual_array)
    assert not mismatch.any(), (
        f"{label}: {int(mismatch.sum())}/{expected_array.size} elements differ; "
        f"first at flat index {int(np.flatnonzero(mismatch)[0])}"
    )


def _fiber() -> FiberSpanConfig:
    """Build the 80 km span used by the legacy golden vectors."""
    return FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)


def _smf(
    dispersion_d: float | None = None,
    dispersion_s: float | None = None,
    length_km: float = 100.0,
) -> FiberSpanConfig:
    """Build the 100 km SMF span of the ``c_57ch`` benchmark (same coefficients).

    Args:
        dispersion_d: Override of the dispersion parameter ``D`` (ps/(nm*km)).
        dispersion_s: Override of the dispersion slope ``S`` (ps/(nm^2*km)).
        length_km: Span length (km).

    Returns
    -------
        The SMF span configuration.
    """
    update: dict[str, Any] = {"length_km": length_km, "max_span_length_km": length_km}
    if dispersion_d is not None:
        update["dispersion_parameter_d"] = dispersion_d
    if dispersion_s is not None:
        update["dispersion_slope_s"] = dispersion_s
    return SMF_100KM.model_copy(update=update)


def _path(n_spans: int, noise_figure_db: float = 5.5) -> MultiSpanOpticalPath:
    """Build a path of ``n_spans`` identical SMF spans."""
    fiber = _smf()
    return MultiSpanOpticalPath(
        [fiber] * n_spans,
        [EDFAConfig(noise_figure_db=noise_figure_db)] * n_spans,
    )


# ---------------------------------------------------------------------------
# 注册与契约
# ---------------------------------------------------------------------------


def test_model_is_registered_and_constructible() -> None:
    """``ecz_coherent`` 必须出现在注册表里，且能按注册表的参数构造。"""
    from network_nlo_eval.models.base import available_models, get_model_info

    assert ECZ_COHERENT_MODEL_INFO.name in available_models()
    info = get_model_info("ecz_coherent")
    assert info.family == "gn"
    assert info.reference


def test_model_declares_span_context_support() -> None:
    """模型必须声明 ``supports_span_context``，且 ``span_step`` 接受 ``span_context``。"""
    import inspect

    assert ECZCoherentModel.supports_span_context is True
    parameters = inspect.signature(ECZCoherentModel.span_step).parameters
    assert "span_context" in parameters
    assert parameters["span_context"].default is None


def test_single_span_uses_the_power_only_branch() -> None:
    """单跨时核不给出复振幅，路由层因此走纯功率分支（与改造前同一条路）。"""
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    step = model.span_step(
        _smf(),
        EDFAConfig(noise_figure_db=5.5),
        power,
        100e3,
        span_context=SpanContext(span_index=0, total_spans=1, cumulative_length_m=0.0, span_length_m=100e3),
    )
    assert step.nli_amplitude_spm is None
    assert step.nli_amplitude_xpm is None


# ---------------------------------------------------------------------------
# 单跨逐位一致
# ---------------------------------------------------------------------------


def test_single_span_is_bit_identical_to_the_frozen_baseline() -> None:
    """单跨必须与冻结基线逐位一致（硬编码数值断言）。

    基准值取自冻结基线 ``models/isrs_gn.py``（``c_57ch``、1 跨、中间信道）。
    跨间相干只在 ``total_spans > 1`` 时生效，单跨必须逐位不变。
    """
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    grid = SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz)
    path = MultiSpanOpticalPath([_smf()], [EDFAConfig(noise_figure_db=5.5)])
    index = case.num_channels // 2

    model = ECZCoherentModel(grid, _smf(), symbol_rate_hz=case.symbol_rate_hz)
    result = model.evaluate_path(path, power)
    assert result.snr_db[index] == 27.74347690137821
    assert result.spm_noise_w[index] == 1.5494554507921815e-07
    assert result.xpm_noise_w[index] == 3.323160873560721e-07


def test_single_span_is_bit_identical_to_the_frozen_model_class() -> None:
    """单跨链路：与冻结基线**类**的输出逐位一致（整条链路，不只是中信道）。"""
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    grid = SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz)
    path = MultiSpanOpticalPath([_smf()], [EDFAConfig(noise_figure_db=5.5)])
    frozen = MultiBandISRSGN(grid, _smf(), symbol_rate_hz=case.symbol_rate_hz)
    coherent = ECZCoherentModel(grid, _smf(), symbol_rate_hz=case.symbol_rate_hz)
    expected = frozen.evaluate_path(path, power)
    actual = coherent.evaluate_path(path, power)
    for field in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        _assert_bit_equal(getattr(expected, field), getattr(actual, field), field)


# ---------------------------------------------------------------------------
# 黄金向量（新基线 + 旧基线的单跨一致性）
# ---------------------------------------------------------------------------


def _capture_payload() -> dict[str, dict]:
    """Rebuild every captured case with the coherent model.

    Returns
    -------
        The payload of the golden fixture.
    """
    cases: dict[str, dict] = {}

    grid = SpectrumGrid(80, 50e9, 193.1e12)
    model = ECZCoherentModel(grid, _fiber(), symbol_rate_hz=50e9, ode_accel="off")
    power = np.full(80, 1e-3)
    result = model.evaluate_path(MultiSpanOpticalPath([_fiber()], [EDFAConfig(noise_figure_db=5.5)]), power)
    cases["legacy_off_80ch_1span"] = {
        "power_in_w": power.tolist(),
        "signal_power_w": result.signal_power_w.tolist(),
        "spm_noise_w": result.spm_noise_w.tolist(),
        "xpm_noise_w": result.xpm_noise_w.tolist(),
        "ase_noise_w": result.ase_noise_w.tolist(),
        "snr_db": result.snr_db.tolist(),
        "span_count": result.span_count,
        "roadm_count": result.roadm_count,
        "n_ch": 80,
        "spans": 1,
    }

    grid = SpectrumGrid(320, 50e9, 193.1e12)
    model = ECZCoherentModel(grid, _fiber(), symbol_rate_hz=50e9, ode_accel="matvec")
    power = np.full(320, 1e-3)
    result = model.evaluate_path(MultiSpanOpticalPath([_fiber()], [EDFAConfig(noise_figure_db=5.5)]), power)
    cases["legacy_matvec_320ch_1span"] = {
        "power_in_w": power.tolist(),
        "signal_power_w": result.signal_power_w.tolist(),
        "spm_noise_w": result.spm_noise_w.tolist(),
        "xpm_noise_w": result.xpm_noise_w.tolist(),
        "ase_noise_w": result.ase_noise_w.tolist(),
        "snr_db": result.snr_db.tolist(),
        "span_count": result.span_count,
        "roadm_count": result.roadm_count,
        "n_ch": 320,
        "spans": 1,
    }

    grid = SpectrumGrid(150, 50e9, 193.1e12)
    model = ECZCoherentModel(grid, _fiber(), symbol_rate_hz=50e9)
    power = np.full(150, 1e-3) * 10 ** (np.linspace(-3.0, 3.0, 150) / 10.0)
    power[::3] = 0.0
    result = model.evaluate_path(MultiSpanOpticalPath([_fiber()] * 4, [EDFAConfig(noise_figure_db=5.5)] * 4), power)
    cases["legacy_preemphasis_150ch_4span"] = {
        "power_in_w": power.tolist(),
        "signal_power_w": result.signal_power_w.tolist(),
        "spm_noise_w": result.spm_noise_w.tolist(),
        "xpm_noise_w": result.xpm_noise_w.tolist(),
        "ase_noise_w": result.ase_noise_w.tolist(),
        "snr_db": result.snr_db.tolist(),
        "span_count": result.span_count,
        "roadm_count": result.roadm_count,
        "n_ch": 150,
        "spans": 4,
    }

    fiber = FiberSpanConfig(
        length_km=100.0,
        max_span_length_km=100.0,
        use_attenuation_polynomial=True,
        effective_area_um2_ref=83.0,
        aeff_slope_um2_nm=0.06,
    )
    edfa = EDFAConfig(noise_figure_db=np.linspace(5.0, 6.5, 64))
    roadm = ROADMConfig(
        insertion_loss_db=6.0,
        express_attenuation_db=0.4,
        add_drop_loss_db=1.2,
        booster_noise_figure_db=5.2,
        filtering_penalty_db=0.3,
        equalize_output_power=True,
        blocked_channels=[0, 7, 63],
    )
    grid = SpectrumGrid(64, 50e9, 193.1e12)
    model = ECZCoherentModel(grid, fiber, symbol_rate_hz=32e9)
    power = np.full(64, 2e-3)
    result = model.evaluate_path(MultiSpanOpticalPath([fiber, fiber], [edfa, edfa], [roadm, roadm]), power)
    cases["elements_poly_nf_array_roadm_64ch"] = {
        "power_in_w": power.tolist(),
        "signal_power_w": result.signal_power_w.tolist(),
        "spm_noise_w": result.spm_noise_w.tolist(),
        "xpm_noise_w": result.xpm_noise_w.tolist(),
        "ase_noise_w": result.ase_noise_w.tolist(),
        "snr_db": result.snr_db.tolist(),
        "span_count": result.span_count,
        "roadm_count": result.roadm_count,
        "n_ch": 64,
    }

    grid = FlexibleGrid.from_legacy_grid(4, 50e9, 193.1e12, slots_per_channel=4)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=index,
            first_slot=first_slot,
            n_slots=n_slots,
            bit_rate_bps=bit_rate,
            modulation=modulation,
            symbol_rate_hz=bit_rate / bits_per_symbol(modulation),
            roll_off=0.0,
            launch_power_w=1e-3,
        )
        for index, (first_slot, n_slots, bit_rate, modulation) in enumerate(
            ((0, 4, 100e9, "QPSK"), (4, 4, 100e9, "QPSK"), (8, 8, 400e9, "16QAM"))
        )
    ]
    fiber = FiberSpanConfig(length_km=60.0, max_span_length_km=60.0)
    model = ECZCoherentModel(
        grid,
        fiber,
        symbol_rate_hz=50e9,
        connections=connections,
        slots_per_channel=4,
        channel_bandwidth_source="occupied",
        ode_accel="off",
    )
    power = np.full(3, 1e-3)
    result = model.evaluate_path(MultiSpanOpticalPath([fiber] * 2, [EDFAConfig(noise_figure_db=5.5)] * 2), power)
    cases["flex_hetero_4conn_2span"] = {
        "power_in_w": power.tolist(),
        "signal_power_w": result.signal_power_w.tolist(),
        "spm_noise_w": result.spm_noise_w.tolist(),
        "xpm_noise_w": result.xpm_noise_w.tolist(),
        "ase_noise_w": result.ase_noise_w.tolist(),
        "snr_db": result.snr_db.tolist(),
        "span_count": result.span_count,
        "roadm_count": result.roadm_count,
        "n_ch": 3,
    }

    grid = SpectrumGrid(120, 50e9, 193.1e12)
    model = ECZCoherentModel(grid, _fiber(), symbol_rate_hz=50e9, ode_accel="matvec")
    matrix = np.array([[5e-4] * 120, [1e-3] * 120, [2e-3] * 120])
    outputs = model.calc_span_noise_and_power_batch(matrix, 80e3, EDFAConfig(noise_figure_db=5.5))
    cases["kernel_batch_3x120ch"] = {
        "power_in_w": matrix.ravel().tolist(),
        "power_out_w": np.asarray(outputs[0]).ravel().tolist(),
        "spm_noise_w": np.asarray(outputs[1]).ravel().tolist(),
        "xpm_noise_w": np.asarray(outputs[2]).ravel().tolist(),
        "ase_noise_w": np.asarray(outputs[3]).ravel().tolist(),
    }
    return cases


def _write_fixture() -> None:
    """Write the golden fixture of the coherent model."""
    import platform

    payload = {
        "cases": _capture_payload(),
        "meta": {
            "note": (
                "Re-captured baseline of the coherent ECZ model (network_nlo_eval.models.ecz). "
                "The single-span cases are bit-identical to the frozen baseline "
                "ecz_golden_bbb9ed4.json; the multi-span cases differ because the model now "
                "adds inter-span coherence (NLI-weighted visibility)."
            ),
            "numpy": np.__version__,
            "python": platform.python_version(),
        },
    }
    FIXTURE.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


@pytest.fixture(scope="module")
def golden() -> dict[str, dict]:
    """Load the captured cases of the coherent model once per module."""
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def legacy_golden() -> dict[str, dict]:
    """Load the frozen baseline cases once per module."""
    return json.loads(LEGACY_FIXTURE.read_text(encoding="utf-8"))["cases"]


def test_golden_single_span_still_matches_the_frozen_baseline(
    golden: dict[str, dict], legacy_golden: dict[str, dict]
) -> None:
    """单跨算例必须与**旧**基线逐位一致（重捕获的原因只是跨间相干）。

    这是本工作包最关键的一条断言：新基线里单跨的每一个比特都必须等于旧基线，
    多跨的差异只能来自跨间相干。
    """
    for name in ("legacy_off_80ch_1span", "legacy_matvec_320ch_1span"):
        case = golden[name]
        expected = legacy_golden[name]
        for field in (
            "power_in_w",
            "signal_power_w",
            "spm_noise_w",
            "xpm_noise_w",
            "ase_noise_w",
            "snr_db",
        ):
            _assert_bit_equal(expected[field], case[field], f"{name}.{field}")


def test_golden_multi_span_differs_only_in_the_nli(golden: dict[str, dict], legacy_golden: dict[str, dict]) -> None:
    """多跨算例：信号功率与 ASE 不变，只有 NLI 变（相干累加的直接影响）。

    SPM 的相干因子 ``R(N) > N``（可见度为正），因此相干累加后的 SPM 噪声必然
    **高于**旧的纯功率累加；XPM 的相干因子贴着 ``N``，变化在 1% 量级。
    """
    for name in (
        "legacy_preemphasis_150ch_4span",
        "elements_poly_nf_array_roadm_64ch",
        "flex_hetero_4conn_2span",
    ):
        case = golden[name]
        expected = legacy_golden[name]
        _assert_bit_equal(expected["signal_power_w"], case["signal_power_w"], f"{name}.signal")
        _assert_bit_equal(expected["ase_noise_w"], case["ase_noise_w"], f"{name}.ase")
        old_spm = np.asarray(expected["spm_noise_w"], dtype=np.float64)
        new_spm = np.asarray(case["spm_noise_w"], dtype=np.float64)
        active = old_spm > 0.0
        assert active.any()
        # 阻断信道仍然是 0。
        _assert_bit_equal(old_spm[~active], new_spm[~active], f"{name}.spm (blocked)")
        # 相干增益：SPM 必然变大。
        assert np.all(new_spm[active] > old_spm[active]), f"{name}: SPM 没有变大"
        ratio = new_spm[active] / old_spm[active]
        assert ratio.max() > 1.05, f"{name}: SPM 变化只有 {ratio.max():.4f}"


def test_golden_cases_reproduce_bit_for_bit(golden: dict[str, dict]) -> None:
    """新基线本身必须可复现（逐位）。"""
    payload = _capture_payload()
    for name, case in payload.items():
        for field, values in case.items():
            if field in ("span_count", "roadm_count", "n_ch", "spans"):
                assert values == golden[name][field], f"{name}.{field}"
                continue
            _assert_bit_equal(golden[name][field], values, f"{name}.{field}")


def test_golden_single_span_case_matches_the_model(golden: dict[str, dict]) -> None:
    """单跨算例：用模型重算必须与新基线逐位一致（冻结基线的逐位守卫仍有效）。"""
    case = golden["legacy_off_80ch_1span"]
    model = ECZCoherentModel(SpectrumGrid(80, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9, ode_accel="off")
    result = model.evaluate_path(
        MultiSpanOpticalPath([_fiber()], [EDFAConfig(noise_figure_db=5.5)]),
        np.asarray(case["power_in_w"]),
    )
    for field in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        _assert_bit_equal(case[field], getattr(result, field), field)


# ---------------------------------------------------------------------------
# 跨间相干物理
# ---------------------------------------------------------------------------


def test_c_57ch_ten_span_spm_buildup_is_inside_the_work_package_band() -> None:
    """``c_57ch``、10 跨：SPM buildup 必须落在 1.40 ± 0.03。

    参考值：独立求积 1.402、``numerical`` 1.408、``semrau2019`` 1.392。本实现
    用的是同一套 NLI 加权可见度口径（与数值参考逐位一致），落在 1.408 附近。
    """
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    ten = model.evaluate_path(_path(10, case.edfa_noise_figure_db), power)
    buildup = np.asarray(ten.spm_noise_w) / np.asarray(one.spm_noise_w) / 10.0
    assert np.all(buildup > BUILDUP_LOW), f"SPM buildup {buildup.min():.4f} 低于 {BUILDUP_LOW}"
    assert np.all(buildup < BUILDUP_HIGH), f"SPM buildup {buildup.max():.4f} 高于 {BUILDUP_HIGH}"


def test_nli_ratio_equals_the_visibility_buildup() -> None:
    """多跨的 NLI 比值必须等于该口径下的 buildup（相干因子只作用一次）。"""
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    evaluator = model.for_fiber(_smf())
    _, _, factor_spm, factor_xpm = evaluator._coherence(10, 100e3)
    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    ten = model.evaluate_path(_path(10, case.edfa_noise_figure_db), power)
    ratio_spm = np.asarray(ten.spm_noise_w) / np.asarray(one.spm_noise_w)
    ratio_xpm = np.asarray(ten.xpm_noise_w) / np.asarray(one.xpm_noise_w)
    assert np.allclose(ratio_spm, factor_spm, rtol=1e-12)
    assert np.allclose(ratio_xpm, factor_xpm, rtol=1e-12)


def test_buildup_table_matches_the_work_package() -> None:
    """逐跨 buildup 表（``c_57ch``）与工作包的参考量级一致。

    参考：独立求积 1.402、``numerical`` 1.408、``semrau2019`` 1.392。
    """
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    expected = {1: 1.0, 2: 1.114, 5: 1.278, 10: 1.408}
    for n_spans, reference in expected.items():
        result = model.evaluate_path(_path(n_spans, case.edfa_noise_figure_db), power)
        buildup = float(np.sum(np.asarray(result.spm_noise_w)) / (n_spans * np.sum(np.asarray(one.spm_noise_w))))
        assert buildup == pytest.approx(reference, rel=2e-3), f"N={n_spans}: buildup {buildup:.4f}"


def test_zero_dispersion_is_fully_coherent() -> None:
    """色散趋零时跨间完全相干：buildup -> N。

    可见度只依赖 ``beta_2``，而 ``beta_2`` 来自色散参数 ``D``（斜率 ``S`` 只进入
    ``beta_3``），所以零色散是把 ``D`` 调到 0，而不是把 ``S`` 调到 0。
    """
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    fiber = _smf(dispersion_d=1e-9, dispersion_s=1e-9)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        fiber,
        symbol_rate_hz=case.symbol_rate_hz,
    )
    path = MultiSpanOpticalPath([fiber] * 10, [EDFAConfig(noise_figure_db=5.5)] * 10)
    one = MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=5.5)])
    result = model.evaluate_path(path, power)
    single = model.evaluate_path(one, power)
    buildup = float(np.sum(np.asarray(result.spm_noise_w)) / (10.0 * np.sum(np.asarray(single.spm_noise_w))))
    assert buildup == pytest.approx(10.0, rel=1e-3), f"零色散 buildup {buildup:.6f}"


def test_strong_dispersion_stays_below_full_coherence() -> None:
    """强色散下远低于完全相干，但相干增益仍为正。"""
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    fiber = _smf(dispersion_d=100.0, dispersion_s=1e-9)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        fiber,
        symbol_rate_hz=case.symbol_rate_hz,
    )
    result = model.evaluate_path(MultiSpanOpticalPath([fiber] * 10, [EDFAConfig(noise_figure_db=5.5)] * 10), power)
    single = model.evaluate_path(MultiSpanOpticalPath([fiber], [EDFAConfig(noise_figure_db=5.5)]), power)
    ratio = float(np.sum(np.asarray(result.spm_noise_w)) / np.sum(np.asarray(single.spm_noise_w)))
    assert ratio > 10.0, f"强色散下 10 跨/1 跨比值 {ratio:.4f} 未超过 10"
    assert ratio < 25.0, f"强色散下应远低于完全相干，实测 {ratio:.4f}"


def test_xpm_inter_span_coherence_is_negligible() -> None:
    """XPM 的跨间相干几乎为零（互拍频的失配是信道间量级）。

    权重效应对 XPM 的影响因此小于 0.05 dB，可见度函数保持数值参考的带内均匀
    平均口径。
    """
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    evaluator = model.for_fiber(_smf())
    _, _, _, factor_xpm = evaluator._coherence(10, 100e3)
    assert np.all(factor_xpm < 10.05)
    assert np.all(factor_xpm > 9.90)
    one = model.evaluate_path(_path(1, case.edfa_noise_figure_db), power)
    ten = model.evaluate_path(_path(10, case.edfa_noise_figure_db), power)
    ratio = float(np.sum(np.asarray(ten.xpm_noise_w)) / np.sum(np.asarray(one.xpm_noise_w)))
    gain_db = 10.0 * np.log10(ratio / 10.0)
    assert abs(gain_db) < 0.05, f"XPM 的相干增益 {gain_db:.4f} dB 不可忽略"


def test_visibility_matches_the_numerical_reference_caliber() -> None:
    """本实现的可见度必须与数值参考**逐位一致**（同一套 NLI 加权口径）。

    两个模型各自实现（不互相 import 生产代码），逐位一致说明口径没有跑偏。
    """
    from network_nlo_eval.models import numerical as numerical_model

    beta2 = 2.1608e-26
    alpha = 3.7437e-5
    bandwidth = 64e9
    length = 100e3
    distances = np.arange(1, 10, dtype=np.float64) * length
    reference = numerical_model._self_beating_visibility(beta2, alpha, np.array([bandwidth]), length, distances)
    mine = _self_beating_visibility(beta2, alpha, np.array([bandwidth]), length, distances)
    _assert_bit_equal(reference, mine, "visibility")


def test_coherent_factor_and_ramp_are_consistent() -> None:
    """相干因子与等效斜坡必须自洽（斜坡回代即得相干因子）。"""
    visibility = np.array([[0.0983], [0.0446], [0.0294]])
    factor = _coherent_factor(4, visibility)
    assert factor[0] == pytest.approx(4.0 + 2.0 * (3 * 0.0983 + 2 * 0.0446 + 1 * 0.0294))
    ramp = _equivalent_phase_ramp(4, factor)
    recovered = (np.sin(0.5 * 4 * ramp) / np.sin(0.5 * ramp)) ** 2
    assert recovered[0] == pytest.approx(factor[0], rel=1e-9)


def test_cross_beating_visibility_is_negligible_for_neighbouring_channels() -> None:
    """互拍频：相邻信道的可见度必须接近 0（失配是信道间量级）。"""
    visibility = _cross_beating_visibility(
        2.1608e-26,
        np.array([64e9]),
        100e3,
        np.array([75e9]),
    )
    assert abs(float(visibility[0])) < 0.05


# ---------------------------------------------------------------------------
# 振幅契约与确定性
# ---------------------------------------------------------------------------


def test_amplitude_magnitude_equals_the_noise_power() -> None:
    """``|a|**2`` 必须逐位等于同名噪声功率（``rtol=1e-12``、``atol=0``）。"""
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    step = model.span_step(
        _smf(),
        EDFAConfig(noise_figure_db=5.5),
        power,
        100e3,
        span_context=SpanContext(span_index=2, total_spans=10, cumulative_length_m=200e3, span_length_m=100e3),
    )
    assert step.nli_amplitude_spm is not None
    assert step.nli_amplitude_xpm is not None
    assert np.allclose(np.abs(step.nli_amplitude_spm) ** 2, step.spm_noise_w, rtol=1e-12, atol=0.0)
    assert np.allclose(np.abs(step.nli_amplitude_xpm) ** 2, step.xpm_noise_w, rtol=1e-12, atol=0.0)


def test_phase_comes_only_from_the_span_context() -> None:
    """振幅的相位只由 ``SpanContext`` 决定，模型内部没有跨号计数器。

    同一个上下文重复调用必须给出逐位相同的振幅；换一个上下文（跨号不同、
    累计长度相同）必须给出同一个振幅——相位只跟累计长度走。
    """
    case = build_case("c_57ch")
    power = launch_profile(case, 0.0)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    evaluator = model.for_fiber(_smf())
    context = SpanContext(span_index=3, total_spans=10, cumulative_length_m=300e3, span_length_m=100e3)
    first = evaluator.span_fiber_output(power, 100e3, 1, context)
    second = evaluator.span_fiber_output(power, 100e3, 1, context)
    assert np.array_equal(first.nli_amplitude_spm, second.nli_amplitude_spm)
    assert np.array_equal(first.nli_amplitude_xpm, second.nli_amplitude_xpm)
    other = evaluator.span_fiber_output(
        power,
        100e3,
        1,
        SpanContext(span_index=7, total_spans=10, cumulative_length_m=300e3, span_length_m=100e3),
    )
    assert np.allclose(first.nli_amplitude_spm, other.nli_amplitude_spm, rtol=1e-15, atol=0.0)


def test_repeated_path_evaluations_are_bit_identical() -> None:
    """同一实例、同一条链路、同一功率连调三次必须逐位相同。"""
    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    path = _path(case.span_count, case.edfa_noise_figure_db)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )
    first = model.evaluate_path(path, power)
    second = model.evaluate_path(path, power)
    third = model.evaluate_path(path, power)
    for field in ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db"):
        _assert_bit_equal(getattr(first, field), getattr(second, field), f"{field} (2nd)")
        _assert_bit_equal(getattr(first, field), getattr(third, field), f"{field} (3rd)")


def test_coherent_accumulation_does_not_regress_the_latency() -> None:
    """相干累加不得把热路径拖慢一个量级。

    基准（本机实测）：相干模型的单跨热调用与冻结基线同量级；这里用绝对上限加
    一条与机器无关的相对守卫：多跨的每跨开销不得显著高于单跨。
    """
    import time

    case = build_case("cl_152ch")
    power = launch_profile(case, None)
    model = ECZCoherentModel(
        SpectrumGrid(case.num_channels, case.channel_spacing_hz, case.center_frequency_hz),
        _smf(),
        symbol_rate_hz=case.symbol_rate_hz,
    )

    def median_ms(path: MultiSpanOpticalPath) -> float:
        model.evaluate_path(path, power)
        samples = []
        for _ in range(15):
            started = time.perf_counter()
            model.evaluate_path(path, power)
            samples.append((time.perf_counter() - started) * 1e3)
        return float(np.median(samples))

    single = median_ms(_path(1, case.edfa_noise_figure_db))
    multi = median_ms(_path(10, case.edfa_noise_figure_db))
    assert single < 2.0, f"单跨热调用 {single:.3f} ms 超标"
    assert multi < 4.0, f"10 跨热调用 {multi:.3f} ms 超标"
    assert multi / 10.0 < 2.0 * single, f"每跨开销 {multi / 10.0:.3f} ms 明显高于单跨 {single:.3f} ms"


# ---------------------------------------------------------------------------
# 合成核路由测试（不依赖任何真实模型）
# ---------------------------------------------------------------------------

#: 玩具模型的信道数。
TOY_NODES = 4

#: 玩具模型的每跨功率传输因子（振幅因子是它的平方根）。
TOY_TRANSFER = 0.5


class _ToyModel:
    """只实现路由层需要的最小接口的玩具模型。

    每次跨段给出固定的噪声功率 1 W 与可选的复振幅；出纤功率是入纤功率的
    :data:`TOY_TRANSFER` 倍，因此路由层的传输因子是已知的常数。

    Attributes
    ----------
        amplitudes: 逐跨的 SPM 复振幅；``None`` 表示走纯功率口径。
        noise_w: 逐跨的 SPM 噪声功率 (W)。
    """

    supports_span_context = True
    n_nodes = TOY_NODES
    f_abs_hz = np.array([193.1e12, 193.2e12, 193.3e12, 193.4e12])
    channel_bandwidth_hz_channels = np.full(TOY_NODES, 64e9)

    def __init__(self, amplitudes: tuple[complex, ...] | None, noise_w: float = 1.0) -> None:
        """记住逐跨的振幅（``None`` 表示纯功率口径）与噪声功率。"""
        self.amplitudes = amplitudes
        self.noise_w = noise_w
        self.span_index = 0

    def power_width_scale(self) -> np.ndarray:
        """功率归一化因子。"""
        return np.ones(TOY_NODES)

    def blocked_nodes(self, roadm: object) -> list[int]:
        """被阻断的信道。"""
        return []

    def span_step(
        self,
        fiber_config: FiberSpanConfig,
        edfa_config: EDFAConfig | None,
        power_in_w: np.ndarray,
        span_length_m: float,
        span_context: SpanContext | None = None,
    ) -> ElementOutput:
        """返回固定功率与（可选的）固定复振幅。"""
        power = np.asarray(power_in_w, dtype=np.float64)
        out = power * TOY_TRANSFER
        spm = np.full(TOY_NODES, self.noise_w)
        ase = np.full(TOY_NODES, 1e-15)
        if self.amplitudes is None:
            self.span_index += 1
            return ElementOutput(out, spm, np.zeros(TOY_NODES), ase)
        amplitude = self.amplitudes[self.span_index]
        self.span_index += 1
        complex_amplitude = np.full(TOY_NODES, amplitude, dtype=np.complex128)
        return ElementOutput(
            out,
            spm,
            np.zeros(TOY_NODES),
            ase,
            complex_amplitude,
            np.zeros(TOY_NODES, dtype=np.complex128),
        )


def _toy_path(spans: int) -> MultiSpanOpticalPath:
    """玩具路径。"""
    return MultiSpanOpticalPath(
        fiber_configs=[_fiber()] * spans,
        edfa_configs=[EDFAConfig()] * spans,
        roadm_configs=None,
    )


def test_route_layer_accumulates_the_complex_amplitudes() -> None:
    """路由层必须按 ``|sum a_k|^2`` 累加玩具核的复振幅（不依赖任何真实模型）。

    两跨给出等幅同相的振幅 ``(1, 1)`` 与等幅反相的 ``(1, -1)``：前者相长得
    ``(1 + sqrt(0.5))^2``，后者相消得 ``(1 - sqrt(0.5))^2``。纯功率口径对两种
    情况都给 1.5 W，因此这条断言只对相干累加成立。
    """
    power = np.full(TOY_NODES, 1e-3)

    incoherent = accumulate_route(_ToyModel(None), _toy_path(2), power)
    assert incoherent.spm_noise_w == pytest.approx(np.full(TOY_NODES, 1.5))

    aligned = accumulate_route(_ToyModel((1.0 + 0j, 1.0 + 0j)), _toy_path(2), power)
    expected_aligned = (1.0 + np.sqrt(TOY_TRANSFER)) ** 2
    assert aligned.spm_noise_w == pytest.approx(np.full(TOY_NODES, expected_aligned), rel=1e-12)

    cancelling = accumulate_route(_ToyModel((1.0 + 0j, -1.0 + 0j)), _toy_path(2), power)
    expected_cancelling = (np.sqrt(TOY_TRANSFER) - 1.0) ** 2
    assert cancelling.spm_noise_w == pytest.approx(np.full(TOY_NODES, expected_cancelling), rel=1e-12)
    # 纯功率口径给 1.5 W：相干相消必须明显低于它。
    assert cancelling.spm_noise_w[0] < incoherent.spm_noise_w[0]


def test_route_layer_keeps_the_power_only_branch_without_amplitudes() -> None:
    """没有复振幅时路由层必须走纯功率累加（逐跨乘传输因子再相加）。"""
    power = np.full(TOY_NODES, 1e-3)
    result = accumulate_route(_ToyModel(None), _toy_path(2), power)
    assert result.spm_noise_w == pytest.approx(np.full(TOY_NODES, 1.5))
    assert result.xpm_noise_w == pytest.approx(np.zeros(TOY_NODES))


def test_single_span_amplitude_magnitude_equals_its_noise_power() -> None:
    """单跨时玩具核的 ``|a|^2`` 必须等于它给出的噪声功率。"""
    power = np.full(TOY_NODES, 1e-3)
    result = accumulate_route(_ToyModel((0.5 + 0.5j,)), _toy_path(1), power)
    assert result.spm_noise_w == pytest.approx(np.full(TOY_NODES, 0.5))


if __name__ == "__main__":  # pragma: no cover - 手工重捕获入口
    if "--capture" not in sys.argv:
        raise SystemExit("用法: python -m tests.test_ecz --capture")
    _write_fixture()
    print(f"已写入 {FIXTURE}")
