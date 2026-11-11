"""任务 BI：网络相干装配接入 ``QoTValidator`` / CLI 的验收测试。

覆盖契约：

* ``QoTValidator(coherent_assembly=False)``（默认）与"整跨 × N + 余段"的
  历史功率装配逐位相同，且构造参数不改变其他语义；
* ``coherent_assembly=True`` 时 ``LinkNoiseCache`` 的 SPM/XPM/ASE 与同模型
  ``evaluate_path`` 的对应数组一致（末位容差），缓存 SNR 差 ≤0.01 dB；
  覆盖 1/5/10 跨与 C 100 GHz / C 37.5 GHz；
* 没有跨间振幅能力的模型（``ecz``、有活跃单泵的 ``pcfm_h3_raman``、
  C+L 六泵 ``pcfm_h3_raman``）在 ``True`` 下数组逐位不变——开关只是请求，
  模型能力决定是否生效；
* CLI ``simulate`` 默认不传 ``coherent_assembly``，显式
  ``--coherent-assembly`` 才转发给 ``QoTValidator``，且对 ``ecz`` 不报错。

参考只用仓库内的 ``evaluate_path`` 与同一评估器的单跨入口，不读 ``data/``
或 ``results/``。
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner

from network_nlo_eval import __main__
from network_nlo_eval.benchmarks.cases import SMF_100KM
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
    ROADMConfig,
)
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.qot_checker import QoTValidator

_SPAN_KM = 100.0
_SYMBOL_RATE_HZ = 64e9
_EDFA = EDFAConfig(noise_figure_db=5.5)
_POWER_W = 1e-3 * 10 ** (6.0 / 10.0)  # 6 dBm/信道：非线性项主导的工作点

#: C+L 六泵（与 ``test_pcfm_h3_raman`` 同源）。
_SIX_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]


def _fiber(length_km: float) -> FiberSpanConfig:
    """C 波段 SMF，跨长固定 100 km；参考光纤与基准算例同源。"""
    return SMF_100KM.model_copy(update={"length_km": length_km, "max_span_length_km": _SPAN_KM})


def _model(name: str, grid: SpectrumGrid, **options: object):
    """构造注册表模型。"""
    return create_model(name, grid, _fiber(_SPAN_KM), symbol_rate_hz=_SYMBOL_RATE_HZ, **options)


def _link_topology(spans: int) -> NetworkTopology:
    """两节点单链路拓扑，链路长度 = ``spans * 100 km``。"""
    length_km = spans * _SPAN_KM
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=length_km)
    return NetworkTopology(
        network_raw=graph,
        default_fiber_config=_fiber(length_km),
        default_edfa_config=_EDFA,
        default_roadm_config=ROADMConfig(),
    )


def _validator(model, grid: SpectrumGrid, spans: int, *, coherent: bool) -> QoTValidator:
    """按给定开关构造验证器；``coherent`` 总是显式给出，便于对准两种口径。"""
    return QoTValidator(
        gn_evaluator=model,
        network_topology=_link_topology(spans),
        spectrum_grid=grid,
        coherent_assembly=coherent,
    )


def _cache(validator: QoTValidator, power: np.ndarray):
    """单链路噪声缓存；拓扑只有一条 (0, 1) 链路。"""
    return validator._calc_link_noise((0, 1), power, None)


def _snr_db(power: np.ndarray, spm: np.ndarray, xpm: np.ndarray, ase: np.ndarray) -> np.ndarray:
    """与 ``evaluate_path`` 相同的单链路 SNR 定义（无 SPM 跨数惩罚）。"""
    return 10.0 * np.log10(power / (spm + xpm + ase))


def _path(spans: int) -> MultiSpanOpticalPath:
    """与链路等长的均匀光路。"""
    return MultiSpanOpticalPath([_fiber(_SPAN_KM)] * spans, [_EDFA] * spans)


def _legacy_power_assembly(model, power: np.ndarray, spans: int) -> tuple[np.ndarray, ...]:
    """历史"整跨 × N + 余段"的功率装配，按文档顺序从全零数组累加。"""
    evaluator = model.for_fiber(_fiber(spans * _SPAN_KM))
    _, s_spm, s_xpm, s_ase = evaluator._calc_span_noise_and_power(
        power, _SPAN_KM * 1e3, _EDFA, None, n_effective_spans=1
    )
    totals = []
    for single in (s_spm, s_xpm, s_ase):
        total = np.zeros(power.size, dtype=np.float64)
        total += single * spans
        totals.append(total)
    return tuple(totals)


# ---------------------------------------------------------------------------
# 默认路径：逐位不变
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", [100e9, 37.5e9], ids=["100GHz", "37.5GHz"])
@pytest.mark.parametrize("spans", [1, 5, 10])
def test_default_assembly_is_bit_identical_to_legacy(spans: int, spacing_hz: float) -> None:
    """默认 ``coherent_assembly=False``：缓存与"整跨 × N"的历史算术逐位相同。"""
    num_channels = 8
    grid = SpectrumGrid(num_channels, spacing_hz, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(num_channels, _POWER_W)

    validator = _validator(model, grid, spans, coherent=False)
    assert validator.coherent_assembly is False
    cache = _cache(validator, power)

    expected_spm, expected_xpm, expected_ase = _legacy_power_assembly(model, power, spans)
    assert cache.span_count == spans
    assert cache.launch_power_convention == "total"
    assert np.array_equal(cache.spm, expected_spm)
    assert np.array_equal(cache.xpm, expected_xpm)
    assert np.array_equal(cache.ase, expected_ase)


def test_default_construction_does_not_set_the_flag_by_default() -> None:
    """不传参数时属性为 ``False``，且默认缓存的物理量与原路径一致。"""
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(6, _POWER_W)

    implicit = QoTValidator(
        gn_evaluator=model,
        network_topology=_link_topology(5),
        spectrum_grid=grid,
    )
    explicit = _validator(model, grid, 5, coherent=False)
    assert implicit.coherent_assembly is False

    implicit_cache = _cache(implicit, power)
    explicit_cache = _cache(explicit, power)
    for name in ("spm", "xpm", "ase"):
        assert np.array_equal(getattr(implicit_cache, name), getattr(explicit_cache, name))
    assert implicit_cache.span_count == explicit_cache.span_count


# ---------------------------------------------------------------------------
# 相干装配：与 evaluate_path 对齐
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spacing_hz", [100e9, 37.5e9], ids=["100GHz", "37.5GHz"])
@pytest.mark.parametrize("spans", [1, 5, 10])
def test_coherent_assembly_matches_evaluate_path(spans: int, spacing_hz: float) -> None:
    """``True`` 时缓存数组复现 ``evaluate_path``，缓存 SNR 差 ≤0.01 dB。

    单跨时模型只给功率，相干分支自动退回功率累加，因此 1 跨也在这条断言里。
    """
    num_channels = 8
    grid = SpectrumGrid(num_channels, spacing_hz, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(num_channels, _POWER_W)

    validator = _validator(model, grid, spans, coherent=True)
    assert validator.coherent_assembly is True
    cache = _cache(validator, power)
    reference = model.evaluate_path(_path(spans), power)

    assert cache.span_count == spans
    assert np.allclose(cache.spm, reference.spm_noise_w, rtol=1e-10, atol=0.0)
    assert np.allclose(cache.xpm, reference.xpm_noise_w, rtol=1e-10, atol=0.0)
    assert np.allclose(cache.ase, reference.ase_noise_w, rtol=1e-10, atol=0.0)

    snr_cache = _snr_db(power, cache.spm, cache.xpm, cache.ase)
    snr_path = _snr_db(power, reference.spm_noise_w, reference.xpm_noise_w, reference.ase_noise_w)
    assert np.all(np.isfinite(snr_cache))
    assert float(np.max(np.abs(snr_cache - snr_path))) <= 0.01


def test_coherent_assembly_matches_evaluate_path_with_additive_fwm() -> None:
    """密集栅格的 additive FWM 拆分：相干装配把 FWM 按功率并回 XPM。"""
    grid = SpectrumGrid(10, 37.5e9, 193.1e12)
    model = _model("pcfm_h3_raman", grid, dense_grid_fwm_fix=True)
    power = np.full(10, _POWER_W)

    validator = _validator(model, grid, 5, coherent=True)
    cache = _cache(validator, power)
    reference = model.evaluate_path(_path(5), power)

    assert np.allclose(cache.xpm, reference.xpm_noise_w, rtol=1e-10, atol=0.0)
    snr_cache = _snr_db(power, cache.spm, cache.xpm, cache.ase)
    snr_path = _snr_db(power, reference.spm_noise_w, reference.xpm_noise_w, reference.ase_noise_w)
    assert float(np.max(np.abs(snr_cache - snr_path))) <= 0.01


def test_coherent_assembly_differs_when_coherence_matters() -> None:
    """10 跨 6 dBm：默认非相干口径与相干/路径级结果不可忽略地偏离。

    守住"开关不是无操作"：默认路径与 ``evaluate_path`` 的差大于 0.05 dB，
    打开后回到 ≤0.01 dB。
    """
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    model = _model("pcfm_h3_raman", grid)
    power = np.full(8, _POWER_W)

    default_cache = _cache(_validator(model, grid, 10, coherent=False), power)
    coherent_cache = _cache(_validator(model, grid, 10, coherent=True), power)
    reference = model.evaluate_path(_path(10), power)

    snr_path = _snr_db(power, reference.spm_noise_w, reference.xpm_noise_w, reference.ase_noise_w)
    snr_default = _snr_db(power, default_cache.spm, default_cache.xpm, default_cache.ase)
    snr_coherent = _snr_db(power, coherent_cache.spm, coherent_cache.xpm, coherent_cache.ase)
    assert float(np.max(np.abs(snr_default - snr_path))) > 0.05
    assert float(np.max(np.abs(snr_coherent - snr_path))) <= 0.01


# ---------------------------------------------------------------------------
# 模型能力决定开关是否生效
# ---------------------------------------------------------------------------


def _assert_power_path_unchanged(model, grid: SpectrumGrid, spans: int, power: np.ndarray) -> None:
    """断言开关打开与关闭时三个缓存数组逐位相同。"""
    default = _cache(_validator(model, grid, spans, coherent=False), power)
    requested = _cache(_validator(model, grid, spans, coherent=True), power)
    assert requested.span_count == default.span_count
    assert np.array_equal(requested.spm, default.spm)
    assert np.array_equal(requested.xpm, default.xpm)
    assert np.array_equal(requested.ase, default.ase)


@pytest.mark.parametrize("spans", [2, 5])
def test_frozen_baseline_ecz_ignores_the_request(spans: int) -> None:
    """冻结基线 ``ecz`` 没有跨间相干状态：请求不改变任何数组。"""
    grid = SpectrumGrid(6, 100e9, 193.1e12)
    model = _model("ecz", grid)
    assert not getattr(model.for_fiber(_fiber(spans * _SPAN_KM)), "supports_span_context", False)
    _assert_power_path_unchanged(model, grid, spans, np.full(6, _POWER_W))


def test_pumped_h3_ignores_the_request() -> None:
    """有活跃泵的 ``pcfm_h3_raman`` 解析为 incoherent：请求不改变任何数组。"""
    grid = SpectrumGrid(4, 100e9, 193.1e12)
    pump = RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction="backward",
        attenuation_db_per_km=0.25,
    )
    model = _model("pcfm_h3_raman", grid, raman_pumps=[pump])
    assert model.for_fiber(_fiber(200.0)).supports_span_context is False
    _assert_power_path_unchanged(model, grid, 2, np.full(4, _POWER_W))


def test_c_l_six_pumps_ignore_the_request() -> None:
    """C+L 六泵：有泵 H3 仍是功率路径，``True`` 下数组逐位不变。"""
    grid = SpectrumGrid(40, 100e9, 190.0e12)
    model = _model("pcfm_h3_raman", grid, raman_pumps=_SIX_PUMPS)
    assert model.for_fiber(_fiber(200.0)).supports_span_context is False
    _assert_power_path_unchanged(model, grid, 2, np.full(40, _POWER_W))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    """Click 命令行测试的运行器。"""
    return CliRunner()


def _write_topology(path: Path) -> Path:
    """三节点线形拓扑，供 CLI 冒烟测试使用。"""
    payload = {
        "directed": False,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": 1}, {"id": 2}, {"id": 3}],
        "edges": [
            {"source": 1, "target": 2, "weight": 80.0},
            {"source": 2, "target": 3, "weight": 80.0},
        ],
    }
    topology_file = path / "topology.json"
    topology_file.write_text(json.dumps(payload), encoding="utf-8")
    return topology_file


def _record_validator(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """包装 ``__main__.QoTValidator``，记录构造时收到的关键字。"""
    recorded: list[dict[str, object]] = []
    real = __main__.QoTValidator

    class _RecordingValidator(real):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            recorded.append(dict(kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(__main__, "QoTValidator", _RecordingValidator)
    return recorded


def _simulate_args(topology_file: Path, output_dir: Path, *extra: str) -> list[str]:
    return [
        "simulate",
        "-t",
        str(topology_file),
        "-s",
        "3",
        "--num-channels",
        "8",
        "-o",
        str(output_dir),
        *extra,
    ]


def test_cli_default_does_not_pass_coherent_assembly(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认不传 ``coherent_assembly``：构造调用与接入前保持一致。"""
    recorded = _record_validator(monkeypatch)
    topology_file = _write_topology(tmp_path)
    result = runner.invoke(__main__.main, _simulate_args(topology_file, tmp_path / "out"))
    assert result.exit_code == 0, result.output
    assert len(recorded) == 1
    assert "coherent_assembly" not in recorded[0]


def test_cli_explicit_flag_is_forwarded(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式 ``--coherent-assembly`` 才把 ``coherent_assembly=True`` 传给验证器。"""
    recorded = _record_validator(monkeypatch)
    topology_file = _write_topology(tmp_path)
    result = runner.invoke(
        __main__.main,
        _simulate_args(topology_file, tmp_path / "out", "--coherent-assembly"),
    )
    assert result.exit_code == 0, result.output
    assert len(recorded) == 1
    assert recorded[0]["coherent_assembly"] is True


def test_cli_coherent_flag_does_not_break_power_only_model(
    runner: CliRunner, tmp_path: Path
) -> None:
    """默认 ``ecz`` 收到请求后仍走功率路径，仿真正常完成且落盘。"""
    topology_file = _write_topology(tmp_path)
    output_dir = tmp_path / "out"
    result = runner.invoke(
        __main__.main,
        _simulate_args(topology_file, output_dir, "--coherent-assembly", "--qot-model", "ecz"),
    )
    assert result.exit_code == 0, result.output
    report = json.loads((output_dir / "simulation_results.json").read_text(encoding="utf-8"))
    assert "simulation_metrics" in report


def test_cli_help_states_the_request_semantics(runner: CliRunner) -> None:
    """帮助文本写清"开关是请求，模型能力决定是否生效"。"""
    result = runner.invoke(__main__.main, ["simulate", "--help"])
    assert result.exit_code == 0, result.output
    assert "--coherent-assembly" in result.output
    flat = " ".join(result.output.split())
    assert "request" in flat.lower()
    assert "model" in flat.lower()
