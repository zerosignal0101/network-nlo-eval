"""任务 BD：``nli_fallback_trigger`` 的触发集合与组合验收测试。

覆盖契约：

* 默认 ``nli_fallback_trigger="audit"``：与不传该选项逐位相同，且
  ``nli_fallback=None`` 时该选项完全不改变结果、不构造数值核；
* ``"fwm_only"`` 在 dense C（50/37.5/25 GHz）上不触发，
  ``fallback_evaluations == 0``，且加上 ``dense_grid_fwm_fix=True`` 后与
  “只用 FWM 修正”逐位相同；
* ``"fwm_only"`` 在 O 波段仍触发，NLI 等于 ``numerical`` 参考；
* ``"fwm_only" + dense_grid_fwm_fix``：dense 单跨 ≤0.5 dB，O 波段 6 dBm
  SNR ≤0.1 dB；
* ``"audit" + dense_grid_fwm_fix`` 在 kernel 与 model 两层构造期 ValueError；
* 非法 trigger 在两层构造期 ValueError；
* CLI：默认不传、``fwm_only`` 才传、``audit`` 不传、ECZ 明确报错；
* 有泵 H3 + fallback（含 ``fwm_only``）仍在构造期拒绝。

参考只跑仓库内的 ``network_nlo_eval.models.numerical``，不读 ``data/`` 或
``results/``。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from network_nlo_eval import __main__
from network_nlo_eval.benchmarks.cases import (
    SMF_100KM,
    BenchCase,
    band_edges_hz,
    build_path,
    launch_profile,
)
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.pcfm import PCFMSpanKernel
from network_nlo_eval.models.pcfm_h3_raman import PCFH3RamanSpanKernel
from network_nlo_eval.network.elements import EDFAConfig, RamanPumpConfig

O_SYMBOL_RATE_HZ = 96e9
_ARRAY_FIELDS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")


# ---------------------------------------------------------------------------
# 算例构造
# ---------------------------------------------------------------------------


def _dense_case(spacing_hz: float, span_count: int = 1) -> BenchCase:
    """C 波段密集栅格算例：64 GBaud、100 km/跨、NF 5.5 dB。"""
    low, high = band_edges_hz("C")
    return BenchCase(
        name=f"bd_dense_{spacing_hz / 1e9:g}_{span_count}span",
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


def _o_fiber():
    """O 波段合成光纤：1310 nm 锚点、D=2.6 ps/nm/km（与 AW 同一根光纤）。"""
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
    """16.1 THz 占用带、中心 229.15 THz 的 O 波段栅格。"""
    return SpectrumGrid(num_channels, 16.1e12 / (num_channels - 1), 229.15e12)


def _o_model(grid: SpectrumGrid, fiber, **options: Any):
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


def _dense_frequencies() -> np.ndarray:
    """57 个 50 GHz 间隔的 C 波段中心频率（64 GBaud 下为 dense 计划）。"""
    return np.linspace(195.9e12, 195.9e12 - 50e9 * 56, 57)


def _nli(result) -> np.ndarray:
    return np.asarray(result.spm_noise_w, dtype=np.float64) + np.asarray(
        result.xpm_noise_w, dtype=np.float64
    )


def _max_nli_error(result, reference) -> float:
    ratio = np.maximum(_nli(result), 1e-300) / np.maximum(_nli(reference), 1e-300)
    return float(np.max(np.abs(10.0 * np.log10(ratio))))


def _max_snr_error(result, reference) -> float:
    return float(
        np.max(np.abs(np.asarray(result.snr_db, dtype=np.float64) - np.asarray(reference.snr_db)))
    )


def _assert_bitwise_equal(left, right) -> None:
    for name in _ARRAY_FIELDS:
        assert np.array_equal(np.asarray(getattr(left, name)), np.asarray(getattr(right, name))), name


# ---------------------------------------------------------------------------
# 默认 audit：逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", [50e9, 37.5e9, 25e9])
def test_default_trigger_is_bit_identical_to_explicit_audit(spacing_hz: float) -> None:
    """不传 trigger 与显式 ``audit`` 在 dense C 上逐位相同。"""
    case = _dense_case(spacing_hz, 1)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    implicit = make_model("pcfm_h3_raman", case, nli_fallback="numerical")
    explicit = make_model(
        "pcfm_h3_raman", case, nli_fallback="numerical", nli_fallback_trigger="audit"
    )

    implicit_result = implicit.evaluate_path(path, power)
    explicit_result = explicit.evaluate_path(path, power)

    _assert_bitwise_equal(implicit_result, explicit_result)
    assert implicit.fallback_evaluations == explicit.fallback_evaluations == 1
    assert implicit.last_nli_fallback_applied is True
    assert explicit.last_nli_fallback_applied is True


def test_trigger_has_no_effect_when_fallback_is_off() -> None:
    """``nli_fallback=None``：两个 trigger 值都不改变数值，也不建数值核。"""
    case = _dense_case(37.5e9, 1)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    plain = make_model("pcfm_h3_raman", case).evaluate_path(path, power)

    for trigger in ("audit", "fwm_only"):
        model = make_model("pcfm_h3_raman", case, nli_fallback_trigger=trigger)
        result = model.evaluate_path(path, power)
        _assert_bitwise_equal(plain, result)
        assert model.fallback_evaluations == 0
        assert model.last_nli_fallback_applied is False
        for kernel in model._kernel_cache.values():
            assert kernel._fallback_kernel is None


# ---------------------------------------------------------------------------
# 触发集合：dense vs O 波段
# ---------------------------------------------------------------------------


def test_kernel_trigger_sets_are_disjoint_on_dense_and_o_band() -> None:
    """Dense 计划：audit 触发、fwm_only 不触发；O 波段：两者都触发。"""
    f_dense = _dense_frequencies()
    b_dense = np.full(57, 64e9)
    active = np.ones(57, dtype=bool)
    audit = PCFMSpanKernel(f_dense, b_dense, SMF_100KM, nli_fallback="numerical")
    fwm_only = PCFMSpanKernel(
        f_dense,
        b_dense,
        SMF_100KM,
        nli_fallback="numerical",
        nli_fallback_trigger="fwm_only",
    )

    assert audit.dense_grid_plan is True
    assert fwm_only.dense_grid_plan is True
    assert audit._nli_fallback_trigger(active) is True
    assert fwm_only._nli_fallback_trigger(active) is False
    # audit 在 dense 上短路，不计算也不缓存色散审计；fwm_only 必须缓存。
    assert audit._fwm_required_cache == {}
    assert fwm_only._fwm_required_cache != {}

    fiber = _o_fiber()
    f_o = np.linspace(229.15e12 + 8.05e12, 229.15e12 - 8.05e12, 41)
    b_o = np.full(41, O_SYMBOL_RATE_HZ)
    o_fwm_only = PCFMSpanKernel(
        f_o,
        b_o,
        fiber,
        nli_fallback="numerical",
        nli_fallback_trigger="fwm_only",
    )
    o_active = np.ones(41, dtype=bool)
    assert o_fwm_only.dense_grid_plan is False
    assert o_fwm_only._nli_fallback_trigger(o_active) is True


def test_fwm_only_dense_c_never_evaluates_the_numerical_kernel() -> None:
    """Dense C 上 ``fwm_only`` 的 fallback 次数与耗时都是零。"""
    for spacing_hz in (50e9, 37.5e9, 25e9):
        case = _dense_case(spacing_hz, 1)
        path = build_path(case)
        power = launch_profile(case, 6.0)
        model = make_model(
            "pcfm_h3_raman",
            case,
            nli_fallback="numerical",
            nli_fallback_trigger="fwm_only",
        )
        model.evaluate_path(path, power)

        assert model.fallback_evaluations == 0
        assert model.fallback_seconds == 0.0
        assert model.last_nli_fallback_applied is False
        for kernel in model._kernel_cache.values():
            assert kernel._fallback_kernel is None


def test_fwm_only_plus_fix_matches_fix_only_bitwise() -> None:
    """Dense 点回退不触发：组合结果与只用 ``dense_grid_fwm_fix`` 逐位相同。"""
    for spacing_hz in (50e9, 37.5e9, 25e9):
        case = _dense_case(spacing_hz, 1)
        path = build_path(case)
        power = launch_profile(case, 6.0)
        combined_model = make_model(
            "pcfm_h3_raman",
            case,
            nli_fallback="numerical",
            nli_fallback_trigger="fwm_only",
            dense_grid_fwm_fix=True,
        )
        fix_model = make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True)

        combined = combined_model.evaluate_path(path, power)
        fix_only = fix_model.evaluate_path(path, power)

        _assert_bitwise_equal(combined, fix_only)
        assert combined_model.fallback_evaluations == 0
        assert combined_model.last_dense_fwm_fix_applied is True
        assert fix_model.last_dense_fwm_fix_applied is True


def test_fwm_only_o_band_still_triggers_and_matches_the_reference() -> None:
    """O 波段 ``fwm_only`` 仍触发，NLI 与 numerical 完全一致。"""
    fiber = _o_fiber()
    grid = _o_grid(41)
    power = np.full(41, 1e-3)
    path = _o_path(fiber)
    model = _o_model(grid, fiber, nli_fallback="numerical", nli_fallback_trigger="fwm_only")
    result = model.evaluate_path(path, power)
    reference = create_model("numerical", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ).evaluate_path(
        path, power
    )

    assert model.last_nli_fallback_applied is True
    assert model.fallback_evaluations == 1
    assert _max_nli_error(result, reference) < 1e-9


# ---------------------------------------------------------------------------
# 精度：dense 点 ≤0.5 dB，O 波段 6 dBm ≤0.1 dB
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", [50e9, 37.5e9, 25e9])
def test_fwm_only_plus_fix_single_span_dense_is_within_half_db(spacing_hz: float) -> None:
    """Dense 单跨 6 dBm：组合配置的 NLI/SNR 误差都在 0.5 dB 内。"""
    case = _dense_case(spacing_hz, 1)
    path = build_path(case)
    power = launch_profile(case, 6.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    model = make_model(
        "pcfm_h3_raman",
        case,
        nli_fallback="numerical",
        nli_fallback_trigger="fwm_only",
        dense_grid_fwm_fix=True,
    )
    result = model.evaluate_path(path, power)

    assert _max_nli_error(result, reference) <= 0.5
    assert _max_snr_error(result, reference) <= 0.5
    assert model.fallback_evaluations == 0


def test_fwm_only_plus_fix_multi_span_dense_is_within_half_db() -> None:
    """25 GHz × 10 跨 0 dBm：组合配置的 NLI 误差 ≤0.5 dB。"""
    case = _dense_case(25e9, 10)
    path = build_path(case)
    power = launch_profile(case, 0.0)
    reference = make_model("numerical", case).evaluate_path(path, power)
    model = make_model(
        "pcfm_h3_raman",
        case,
        nli_fallback="numerical",
        nli_fallback_trigger="fwm_only",
        dense_grid_fwm_fix=True,
    )
    result = model.evaluate_path(path, power)

    assert _max_nli_error(result, reference) <= 0.5
    assert _max_snr_error(result, reference) <= 0.5
    assert np.all(np.isfinite(np.asarray(result.snr_db)))


def test_fwm_only_o_band_snr_is_within_a_tenth_of_a_db_at_6dbm() -> None:
    """O 波段 6 dBm：``fwm_only`` 的 SNR 误差 ≤0.1 dB。"""
    fiber = _o_fiber()
    grid = _o_grid(41)
    power = np.full(41, 1e-3 * 10.0 ** (6.0 / 10.0))
    path = _o_path(fiber)
    model = _o_model(grid, fiber, nli_fallback="numerical", nli_fallback_trigger="fwm_only")
    result = model.evaluate_path(path, power)
    reference = create_model("numerical", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ).evaluate_path(
        path, power
    )

    assert _max_nli_error(result, reference) < 1e-9
    assert _max_snr_error(result, reference) <= 0.1
    assert model.fallback_evaluations == 1


# ---------------------------------------------------------------------------
# 组合合法性：audit 拒绝、fwm_only 允许
# ---------------------------------------------------------------------------


def test_audit_plus_dense_fwm_fix_is_rejected_at_construction() -> None:
    """``audit`` 下 dense 点会同时命中回退与 FWM 修正：构造期 ValueError。"""
    case = _dense_case(50e9, 1)
    with pytest.raises(ValueError, match="nli_fallback_trigger"):
        make_model("pcfm_h3_raman", case, dense_grid_fwm_fix=True, nli_fallback="numerical")
    with pytest.raises(ValueError, match="nli_fallback_trigger"):
        make_model(
            "pcfm_h3_raman",
            case,
            dense_grid_fwm_fix=True,
            nli_fallback="numerical",
            nli_fallback_trigger="audit",
        )
    with pytest.raises(ValueError, match="nli_fallback_trigger"):
        PCFMSpanKernel(
            _dense_frequencies(),
            np.full(57, 64e9),
            SMF_100KM,
            dense_grid_fwm_fix=True,
            nli_fallback="numerical",
        )


def test_fwm_only_plus_dense_fwm_fix_is_accepted_at_construction() -> None:
    """``fwm_only`` 使触发集合不重叠：模型与 kernel 都能构造。"""
    case = _dense_case(50e9, 1)
    model = make_model(
        "pcfm_h3_raman",
        case,
        dense_grid_fwm_fix=True,
        nli_fallback="numerical",
        nli_fallback_trigger="fwm_only",
    )
    assert model.nli_fallback_trigger == "fwm_only"

    kernel = PCFMSpanKernel(
        _dense_frequencies(),
        np.full(57, 64e9),
        SMF_100KM,
        dense_grid_fwm_fix=True,
        nli_fallback="numerical",
        nli_fallback_trigger="fwm_only",
    )
    assert kernel.nli_fallback_trigger == "fwm_only"


def test_invalid_trigger_value_is_rejected() -> None:
    """非法 trigger 在 kernel、pcfm 与 H3 三层构造期都 ValueError。"""
    with pytest.raises(ValueError, match="nli_fallback_trigger"):
        PCFMSpanKernel(
            np.linspace(195.9e12, 191.6e12, 8),
            np.full(8, 50e9),
            SMF_100KM,
            nli_fallback_trigger="dense",
        )
    with pytest.raises(ValueError, match="nli_fallback_trigger"):
        create_model(
            "pcfm",
            SpectrumGrid(8, 50e9, 193.1e12),
            SMF_100KM,
            nli_fallback_trigger="dense",
        )
    with pytest.raises(ValueError, match="nli_fallback_trigger"):
        create_model(
            "pcfm_h3_raman",
            SpectrumGrid(8, 50e9, 193.1e12),
            SMF_100KM,
            nli_fallback_trigger="dense",
        )


# ---------------------------------------------------------------------------
# 有泵 H3 拒绝
# ---------------------------------------------------------------------------


def _pump() -> RamanPumpConfig:
    return RamanPumpConfig(wavelength_nm=1450.0, power_w=0.3, direction="backward")


def test_pumped_h3_still_rejects_the_fallback_with_fwm_only() -> None:
    """有泵 H3 + fallback 仍拒绝，trigger 不绕过该检查。"""
    case = _dense_case(50e9, 1)
    with pytest.raises(ValueError, match="nli_fallback"):
        make_model(
            "pcfm_h3_raman",
            case,
            raman_pumps=[_pump()],
            nli_fallback="numerical",
            nli_fallback_trigger="fwm_only",
        )
    with pytest.raises(ValueError, match="nli_fallback"):
        make_model(
            "pcfm_h3_raman",
            case,
            raman_pumps=[_pump()],
            nli_fallback="numerical",
            nli_fallback_trigger="fwm_only",
            dense_grid_fwm_fix=True,
        )
    with pytest.raises(ValueError, match="nli_fallback"):
        PCFH3RamanSpanKernel(
            np.linspace(195.9e12, 191.6e12, 8),
            np.full(8, 50e9),
            SMF_100KM,
            pumps=[_pump()],
            gain_model="gaussian",
            nli_fallback="numerical",
            nli_fallback_trigger="fwm_only",
        )


# ---------------------------------------------------------------------------
# CLI：--nli-fallback-trigger 的转发与拒绝
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """CLI 测试用的 runner。"""
    return CliRunner()


def _write_topology(directory: Path) -> Path:
    """写一个两节点单链路拓扑，供 CLI 冒烟测试使用。"""
    payload = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": 1}, {"id": 2}],
        "edges": [{"source": 1, "target": 2, "weight": 100.0}],
    }
    topology_file = directory / "topology.json"
    topology_file.write_text(json.dumps(payload), encoding="utf-8")
    return topology_file


def _spy_on_create_model(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    real_create_model = __main__.create_model

    def spy(name: str, *args: object, **kwargs: object):
        calls.append({"name": name, "kwargs": dict(kwargs)})
        return real_create_model(name, *args, **kwargs)

    monkeypatch.setattr(__main__, "create_model", spy)
    return calls


def _simulate_args(topology_file: Path, output_dir: Path, *extra: str) -> list[str]:
    return [
        "simulate",
        "-t",
        str(topology_file),
        "-s",
        "2",
        "--num-channels",
        "4",
        "-o",
        str(output_dir),
        *extra,
    ]


def test_cli_default_does_not_pass_the_trigger(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认 ``--nli-fallback-trigger audit`` 不传给模型。"""
    topology_file = _write_topology(tmp_path)
    calls = _spy_on_create_model(monkeypatch)

    result = runner.invoke(__main__.main, _simulate_args(topology_file, tmp_path / "out"))

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert "nli_fallback" not in calls[0]["kwargs"]
    assert "nli_fallback_trigger" not in calls[0]["kwargs"]


def test_cli_fwm_only_is_forwarded_with_numerical(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """只有 ``numerical`` + ``fwm_only`` 才把 trigger 传给模型。"""
    topology_file = _write_topology(tmp_path)
    calls = _spy_on_create_model(monkeypatch)

    result = runner.invoke(
        __main__.main,
        _simulate_args(
            topology_file,
            tmp_path / "out",
            "--qot-model",
            "pcfm",
            "--nli-fallback",
            "numerical",
            "--nli-fallback-trigger",
            "fwm_only",
        ),
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["kwargs"]["nli_fallback"] == "numerical"
    assert calls[0]["kwargs"]["nli_fallback_trigger"] == "fwm_only"


def test_cli_explicit_audit_is_not_forwarded(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式 ``audit`` 仍是模型默认值，不传，保持旧模型兼容。"""
    topology_file = _write_topology(tmp_path)
    calls = _spy_on_create_model(monkeypatch)

    result = runner.invoke(
        __main__.main,
        _simulate_args(
            topology_file,
            tmp_path / "out",
            "--qot-model",
            "pcfm",
            "--nli-fallback",
            "numerical",
            "--nli-fallback-trigger",
            "audit",
        ),
    )

    assert result.exit_code == 0, result.output
    assert calls[0]["kwargs"]["nli_fallback"] == "numerical"
    assert "nli_fallback_trigger" not in calls[0]["kwargs"]


def test_cli_fwm_only_without_numerical_is_not_forwarded(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trigger 只修饰 numerical；没有 fallback 时静默不传。"""
    topology_file = _write_topology(tmp_path)
    calls = _spy_on_create_model(monkeypatch)

    result = runner.invoke(
        __main__.main,
        _simulate_args(topology_file, tmp_path / "out", "--nli-fallback-trigger", "fwm_only"),
    )

    assert result.exit_code == 0, result.output
    assert "nli_fallback" not in calls[0]["kwargs"]
    assert "nli_fallback_trigger" not in calls[0]["kwargs"]


def test_cli_rejects_a_model_without_the_trigger_option(runner: CliRunner, tmp_path: Path) -> None:
    """ECZ 不接受该选项：必须给出清晰的 ClickException，而不是静默吞掉。"""
    topology_file = _write_topology(tmp_path)

    result = runner.invoke(
        __main__.main,
        _simulate_args(
            topology_file,
            tmp_path / "out",
            "--qot-model",
            "ecz",
            "--nli-fallback",
            "numerical",
            "--nli-fallback-trigger",
            "fwm_only",
        ),
    )

    assert result.exit_code != 0
    assert "does not accept --nli-fallback-trigger" in result.output
    assert "ecz" in result.output
