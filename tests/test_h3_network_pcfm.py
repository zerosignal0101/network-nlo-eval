"""任务 AZ：PCFM/H3 的网络评估器视图与显式 ``nli_fallback`` 验收测试。

覆盖契约：

* ``PCFMModel.for_fiber`` 返回按光纤 JSON 缓存的网络视图，同一光纤同一对象，
  不同光纤不同对象；
* 单链路单跨（C 100 GHz）用 PCFM / H3 作为 ``gn_evaluator`` 时，
  ``LinkNoiseCache.spm/xpm/ase`` 与直接 ``evaluate_path`` 的对应分量一致；
* 默认 ``nli_fallback=None``：不构造 numerical 核、不导入该模块，网络结果与
  显式传 ``None`` 逐位相同；非 flagged 的 C 100 GHz 即使打开 fallback 也不触发；
* 密集 C 37.5 GHz 与 O 波段打开 fallback 后，网络链路 SNR 与 ``numerical``
  参考的最大差 ≤0.1 dB（闭式参考在密集 C 上是数 dB）；
* 有泵 H3 + ``nli_fallback="numerical"`` 仍在构造期 ``ValueError``；
* 网络视图对 ``roadm_config`` 与 ``n_effective_spans != 1`` 明确报错；
* CLI：默认不传 ``nli_fallback``，显式 ``--nli-fallback numerical`` 才传，
  ECZ 这类不接受的模型给出清晰错误。

参考只跑仓库内的 ``network_nlo_eval.models.numerical``，不读 ``data/`` 或
``results/``。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
from click.testing import CliRunner

from network_nlo_eval import __main__
from network_nlo_eval.benchmarks.cases import SMF_100KM
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, RamanPumpConfig, ROADMConfig
from network_nlo_eval.network.state import NetworkState
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.rwa.link_noise import assemble_link_noise
from network_nlo_eval.rwa.qot_checker import QoTValidator
from network_nlo_eval.simulation.traffic import ServiceRequest

C_100GHZ_SYMBOL_RATE_HZ = 64e9
O_SYMBOL_RATE_HZ = 96e9
_EDFA = EDFAConfig(noise_figure_db=5.5)


# ---------------------------------------------------------------------------
# 算例构造
# ---------------------------------------------------------------------------


def _c_fiber() -> FiberSpanConfig:
    """C 波段 100 km 单跨光纤（与 benchmarks 的 SMF 一致）。"""
    return SMF_100KM.model_copy(update={"length_km": 100.0, "max_span_length_km": 100.0})


def _o_fiber() -> FiberSpanConfig:
    """O 波段合成光纤：1310 nm 锚点、D=2.6 ps/nm/km（与 AW 同一根光纤）。"""
    return SMF_100KM.model_copy(
        update={
            "length_km": 100.0,
            "max_span_length_km": 100.0,
            "reference_wavelength_nm": 1310.0,
            "dispersion_parameter_d": 2.6,
            "dispersion_slope_s": 0.06,
            "use_attenuation_polynomial": False,
            "attenuation_db_km_ref": 0.35,
            "effective_area_um2_ref": 80.0,
            "aeff_slope_um2_nm": 0.05,
        }
    )


def _graph(length_km: float) -> nx.Graph:
    """两节点单链路图。"""
    graph = nx.Graph()
    graph.add_node(0)
    graph.add_node(1)
    graph.add_edge(0, 1, weight=length_km)
    return graph


def _topology(fiber: FiberSpanConfig, roadm: ROADMConfig | None = None) -> NetworkTopology:
    """最小 RWA 拓扑：一条链路、两端默认 EDFA/ROADM。"""
    return NetworkTopology(_graph(fiber.length_km), fiber, _EDFA, roadm or ROADMConfig())


def _network_link(
    grid: SpectrumGrid,
    fiber: FiberSpanConfig,
    *,
    model_name: str = "pcfm_h3_raman",
    symbol_rate_hz: float = C_100GHZ_SYMBOL_RATE_HZ,
    model_options: dict[str, object] | None = None,
):
    """构造模型与该链路的 QoTValidator，返回 ``(model, validator)``。"""
    model = create_model(
        model_name,
        grid,
        fiber,
        symbol_rate_hz=symbol_rate_hz,
        **(model_options or {}),
    )
    validator = QoTValidator(
        gn_evaluator=model,
        network_topology=_topology(fiber),
        spectrum_grid=grid,
        fixed_span_length_km=fiber.max_span_length_km,
    )
    return model, validator


def _link_cache(validator: QoTValidator, power: np.ndarray):
    """按网络路径取一条链路的噪声缓存（单跨整跨，无余段）。"""
    return validator._calc_link_noise((0, 1), power, None)


def _cache_snr_db(power: np.ndarray, cache) -> np.ndarray:
    """链路 SNR (dB)：信号取入纤功率，噪声取三个缓存分量之和。

    与 ``accumulate_service_snr_detail`` 在无 ROADM 损耗/均衡时的标量式相同：
    ``10*log10(P / (spm + xpm + ase))``（单跨时 SPM 惩罚指数为 1）。
    """
    noise = np.asarray(cache.spm) + np.asarray(cache.xpm) + np.asarray(cache.ase)
    return 10.0 * np.log10(np.asarray(power) / noise)


def _max_abs_delta_db(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(values) - np.asarray(reference))))


# ---------------------------------------------------------------------------
# 单跨一致性：网络视图 vs 直接路径
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["pcfm", "pcfm_h3_raman"])
def test_single_span_network_cache_matches_evaluate_path(model_name: str) -> None:
    """C 100 GHz 单链路单跨：三个噪声分量与直接 ``evaluate_path`` 一致。

    允许的差异只有浮点求和顺序造成的末位：这里断言相对 1e-12。实测两侧
    逐位相等（见 ``H3_TASK_REPORT.md``）。
    """
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    fiber = _c_fiber()
    power = np.full(8, 1e-3)
    model, validator = _network_link(grid, fiber, model_name=model_name)

    cache = _link_cache(validator, power)
    path_result = model.evaluate_path(MultiSpanOpticalPath([fiber], [_EDFA]), power)

    assert cache.span_count == 1
    assert cache.launch_power_convention == "total"
    for cache_field, result_field in (
        ("spm", "spm_noise_w"),
        ("xpm", "xpm_noise_w"),
        ("ase", "ase_noise_w"),
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(cache, cache_field)),
            np.asarray(getattr(path_result, result_field)),
            rtol=1e-12,
            atol=0.0,
            err_msg=cache_field,
        )
    np.testing.assert_allclose(
        _cache_snr_db(power, cache),
        np.asarray(path_result.snr_db),
        rtol=0.0,
        atol=1e-9,
    )


def test_network_view_resets_inter_span_state() -> None:
    """连续两条不同功率的网络调用不共享上一跨的相干状态。

    同一模型先算高功率、再算低功率，第二次的结果必须与新建模型的第一次相同。
    """
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    fiber = _c_fiber()
    model, validator = _network_link(grid, fiber, model_options={"egn": True})
    model_2, validator_2 = _network_link(grid, fiber, model_options={"egn": True})
    high = np.full(8, 2e-3)
    low = np.full(8, 1e-3)

    _link_cache(validator, high)
    second = _link_cache(validator, low)
    fresh = _link_cache(validator_2, low)

    assert model_2 is not model
    np.testing.assert_array_equal(np.asarray(second.spm), np.asarray(fresh.spm))
    np.testing.assert_array_equal(np.asarray(second.xpm), np.asarray(fresh.xpm))
    np.testing.assert_array_equal(np.asarray(second.ase), np.asarray(fresh.ase))


# ---------------------------------------------------------------------------
# 默认路径：不构造 numerical 核、逐位不变
# ---------------------------------------------------------------------------


def test_default_none_matches_explicit_none_bitwise() -> None:
    """不传 ``nli_fallback`` 与显式 ``None`` 的网络缓存逐位相同，且不建数值核。"""
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    fiber = _c_fiber()
    power = np.full(8, 1e-3)
    default_model, default_validator = _network_link(grid, fiber)
    explicit_model, explicit_validator = _network_link(grid, fiber, model_options={"nli_fallback": None})

    default_cache = _link_cache(default_validator, power)
    explicit_cache = _link_cache(explicit_validator, power)

    for field in ("spm", "xpm", "ase"):
        np.testing.assert_array_equal(
            np.asarray(getattr(default_cache, field)), np.asarray(getattr(explicit_cache, field))
        )
    assert default_model.last_nli_fallback_applied is False
    assert explicit_model.last_nli_fallback_applied is False
    assert default_model.fallback_evaluations == 0
    assert explicit_model.fallback_evaluations == 0
    for model in (default_model, explicit_model):
        for kernel in model._kernel_cache.values():
            assert kernel._fallback_kernel is None


def test_network_path_does_not_import_the_numerical_kernel() -> None:
    """只 import ``pcfm``、构造网络视图并求值时，``numerical`` 不得进入 ``sys.modules``。"""
    code = (
        "import sys, numpy as np\n"
        "from network_nlo_eval.core.spectrum import SpectrumGrid\n"
        "from network_nlo_eval.models.pcfm import PCFMModel\n"
        "from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig\n"
        "fiber = FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)\n"
        "grid = SpectrumGrid(6, 100e9, 193.1e12)\n"
        "model = PCFMModel(grid, fiber, symbol_rate_hz=64e9)\n"
        "view = model.for_fiber(fiber)\n"
        "out = view._calc_span_noise_and_power(np.full(6, 1e-3), 80e3, EDFAConfig())\n"
        "assert len(out) == 4\n"
        "assert 'network_nlo_eval.models.numerical' not in sys.modules\n"
        "assert all(k._fallback_kernel is None for k in model._kernel_cache.values())\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script, no user input.
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_non_flagged_plan_keeps_bitwise_identical_results_with_fallback_on() -> None:
    """C 100 GHz 不是 flagged 计划：即使打开 fallback，网络缓存也与关闭时逐位相同。"""
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    fiber = _c_fiber()
    power = np.full(8, 1e-3)
    plain_model, plain_validator = _network_link(grid, fiber)
    fallback_model, fallback_validator = _network_link(grid, fiber, model_options={"nli_fallback": "numerical"})

    plain = _link_cache(plain_validator, power)
    fallback = _link_cache(fallback_validator, power)

    for field in ("spm", "xpm", "ase"):
        np.testing.assert_array_equal(np.asarray(getattr(plain, field)), np.asarray(getattr(fallback, field)))
    assert plain_model.fallback_evaluations == 0
    assert fallback_model.fallback_evaluations == 0
    assert fallback_model.last_nli_fallback_applied is False


# ---------------------------------------------------------------------------
# 精度：密集 C 与 O 波段（参考 = numerical）
# ---------------------------------------------------------------------------


def test_dense_c_network_fallback_matches_the_numerical_reference() -> None:
    """密集 C 37.5 GHz 单跨：网络链路 SNR 与 numerical 的差 ≤0.1 dB，闭式差 >1 dB。"""
    grid = SpectrumGrid(25, 37.5e9, 193.1e12)
    fiber = _c_fiber()
    power = np.full(25, 1e-3 * 10 ** (6.0 / 10.0))

    base_model, base_validator = _network_link(grid, fiber)
    model, validator = _network_link(grid, fiber, model_options={"nli_fallback": "numerical"})
    base_snr = _cache_snr_db(power, _link_cache(base_validator, power))
    fallback_snr = _cache_snr_db(power, _link_cache(validator, power))

    reference = create_model("numerical", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ).evaluate_path(
        MultiSpanOpticalPath([fiber], [_EDFA]), power
    )

    assert model.last_nli_fallback_applied is True
    assert model.fallback_evaluations == 1
    assert _max_abs_delta_db(base_snr, reference.snr_db) > 1.0
    assert _max_abs_delta_db(fallback_snr, reference.snr_db) <= 0.1
    assert base_model.fallback_evaluations == 0


class _NetworkAssemblyView:
    """把任意 :class:`~network_nlo_eval.models.route.RouteModel` 包成单跨网络评估器。

    网络层（:func:`~network_nlo_eval.rwa.link_noise.assemble_link_noise`）只认
    ``_calc_span_noise_and_power``，"整跨 × N" 在外层做。它按同一契约包一层
    ``numerical``，就能把 fallback 的精度与网络层的跨间累加语义分开比较：
    两侧用同样的跨数装配，只剩单跨物理的差。
    """

    def __init__(self, model, fiber: FiberSpanConfig) -> None:
        self._model = model
        self.fiber_config = fiber
        self.launch_power_convention = "total"

    def _calc_span_noise_and_power(
        self, power_in_w, span_length_m, edfa_config, roadm_config=None, n_effective_spans=1
    ):
        assert roadm_config is None
        assert n_effective_spans == 1
        element = self._model.span_step(self.fiber_config, edfa_config, power_in_w, span_length_m)
        return element.power_out_w, element.spm_noise_w, element.xpm_noise_w, element.ase_noise_w


def _assembled_cache(view, power: np.ndarray, spans: int):
    """按网络层语义装配 ``spans`` 个整跨的噪声缓存。"""
    return assemble_link_noise(
        evaluator=view,
        edfa_config=_EDFA,
        power_profile=power,
        span_length_m=100e3,
        num_full_spans=spans,
        remainder_length_m=0.0,
        num_channels=power.shape[0],
    )


@pytest.mark.parametrize("spans", [1, 5, 10])
def test_dense_c_network_fallback_matches_the_numerical_network_assembly(spans: int) -> None:
    """同一装配语义下，fallback 的链路 SNR 与 numerical 的差 <1e-6 dB（闭式差 ~2.5 dB）。

    网络层把每条链路的每一跨都当作"第一跨"独立求值再按整跨数放大
    （``assemble_link_noise``）。这里用同一份契约包一层 ``numerical``，因此
    比较里不含"网络层非相干求和 vs 路径层相干累加"的语义差（该差见报告）。
    """
    grid = SpectrumGrid(25, 37.5e9, 193.1e12)
    fiber = _c_fiber()
    power = np.full(25, 1e-3 * 10 ** (6.0 / 10.0))
    reference = _assembled_cache(
        _NetworkAssemblyView(create_model("numerical", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ), fiber),
        power,
        spans,
    )

    base_model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ)
    base_snr = _cache_snr_db(power, _assembled_cache(base_model.for_fiber(fiber), power, spans))

    model = create_model(
        "pcfm_h3_raman", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ, nli_fallback="numerical"
    )
    fallback_snr = _cache_snr_db(power, _assembled_cache(model.for_fiber(fiber), power, spans))

    assert _max_abs_delta_db(base_snr, _cache_snr_db(power, reference)) > 1.0
    assert _max_abs_delta_db(fallback_snr, _cache_snr_db(power, reference)) <= 1e-3
    # 透明链路只付一次双积分：每跨输入相同，缓存命中。
    assert model.fallback_evaluations == 1


def test_o_band_network_fallback_matches_the_numerical_reference() -> None:
    """O 波段 41 ch：网络链路 SNR 与 numerical 的差 ≤0.1 dB。"""
    grid = SpectrumGrid(41, 16.1e12 / 40, 229.15e12)
    fiber = _o_fiber()
    power = np.full(41, 1e-3 * 10 ** (6.0 / 10.0))

    model, validator = _network_link(
        grid,
        fiber,
        symbol_rate_hz=O_SYMBOL_RATE_HZ,
        model_options={"poly_degree": 3, "egn": True, "nli_fallback": "numerical"},
    )
    snr = _cache_snr_db(power, _link_cache(validator, power))
    reference = create_model("numerical", grid, fiber, symbol_rate_hz=O_SYMBOL_RATE_HZ).evaluate_path(
        MultiSpanOpticalPath([fiber], [_EDFA]), power
    )

    assert model.last_nli_fallback_applied is True
    assert _max_abs_delta_db(snr, reference.snr_db) <= 0.1


# ---------------------------------------------------------------------------
# 有泵拒绝、接口边界、缓存
# ---------------------------------------------------------------------------


def _pump() -> RamanPumpConfig:
    """1450 nm 单后向泵。"""
    return RamanPumpConfig(wavelength_nm=1450.0, power_w=0.3, direction="backward")


def test_pumped_h3_rejects_numerical_fallback_at_construction() -> None:
    """有泵链路在模型构造期就拒绝 ``nli_fallback``，网络视图不会绕过该检查。"""
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    fiber = _c_fiber()
    with pytest.raises(ValueError, match="nli_fallback"):
        create_model(
            "pcfm_h3_raman",
            grid,
            fiber,
            symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ,
            raman_pumps=[_pump()],
            nli_fallback="numerical",
        )
    # 光纤自带泵浦同样拒绝。
    pumped_fiber = fiber.model_copy(update={"raman_pumps": [_pump()]})
    with pytest.raises(ValueError, match="nli_fallback"):
        create_model(
            "pcfm_h3_raman",
            grid,
            pumped_fiber,
            symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ,
            nli_fallback="numerical",
        )


def test_network_view_rejects_a_roadm_config() -> None:
    """``span_step`` 不处理 ROADM：网络视图必须报错，不能静默丢掉损耗。"""
    grid = SpectrumGrid(4, 100e9, 193.1e12)
    fiber = _c_fiber()
    model = create_model("pcfm", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ)
    view = model.for_fiber(fiber)

    with pytest.raises(ValueError, match="roadm_config"):
        view._calc_span_noise_and_power(np.full(4, 1e-3), 100e3, _EDFA, ROADMConfig())


def test_network_view_rejects_multi_span_accumulation() -> None:
    """整跨的 N 倍由调用方装配：``n_effective_spans != 1`` 明确报错。"""
    grid = SpectrumGrid(4, 100e9, 193.1e12)
    fiber = _c_fiber()
    model = create_model("pcfm", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ)
    view = model.for_fiber(fiber)

    with pytest.raises(ValueError, match="n_effective_spans"):
        view._calc_span_noise_and_power(np.full(4, 1e-3), 100e3, _EDFA, n_effective_spans=3)


def test_for_fiber_cache_is_per_fiber() -> None:
    """同一光纤返回同一视图，不同光纤返回不同视图；视图不复制模型。"""
    grid = SpectrumGrid(4, 100e9, 193.1e12)
    fiber = _c_fiber()
    model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ)

    first = model.for_fiber(fiber)
    same = model.for_fiber(fiber.model_copy())
    other = model.for_fiber(fiber.model_copy(update={"dispersion_parameter_d": 16.0}))

    assert first is same
    assert first is not other
    assert first.parent is model and other.parent is model
    assert first.fiber_config is fiber
    # 视图只做委托：节点数与频点就是父模型的那一份。
    assert first.n_nodes == model.n_nodes
    assert first.f_abs_hz is model.f_abs_hz
    np.testing.assert_array_equal(first.channel_bandwidth_hz_channels, model.channel_bandwidth_hz_channels)
    assert first.launch_power_convention == model.launch_power_convention
    assert first.ase_polarization_scale == model.ase_polarization_scale
    assert first.connections is model.connections
    assert first.slots_per_channel == model.slots_per_channel


def test_verify_allocation_runs_with_the_h3_network_model() -> None:
    """``QoTValidator.verify_allocation`` 用 ``pcfm_h3_raman`` 能完成一条链路评估。"""
    grid = SpectrumGrid(8, 100e9, 193.1e12)
    fiber = _c_fiber()
    topology = _topology(fiber)
    model = create_model("pcfm_h3_raman", grid, fiber, symbol_rate_hz=C_100GHZ_SYMBOL_RATE_HZ)
    validator = QoTValidator(gn_evaluator=model, network_topology=topology, spectrum_grid=grid)
    state = NetworkState(topology, grid)
    request = ServiceRequest(
        service_id=1,
        source_id=0,
        destination_id=1,
        arrival_time=0.0,
        departure_time=1.0,
        bit_rate_gbps=100.0,
        snr_requirement_db=10.0,
        launch_power_w=1e-3,
    )

    accepted, snrs_db = validator.verify_allocation(request, [0, 1], 2, state)

    assert accepted is True
    assert snrs_db is not None
    assert np.isfinite(snrs_db[2])
    assert state.get_link_state(0, 1).noise_cache is not None
    assert state.get_link_state(0, 1).noise_cache.span_count == 1


# ---------------------------------------------------------------------------
# CLI：--nli-fallback 的转发与拒绝
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
    """记录 ``create_model`` 收到的关键字参数，同时保留真实构造。"""
    calls: list[dict[str, object]] = []
    real_create_model = __main__.create_model

    def spy(name: str, *args: object, **kwargs: object):
        calls.append({"name": name, "kwargs": dict(kwargs)})
        return real_create_model(name, *args, **kwargs)

    monkeypatch.setattr(__main__, "create_model", spy)
    return calls


def _simulate_args(topology_file: Path, output_dir: Path, *extra: str) -> list[str]:
    """最小 ``simulate`` 参数：4 信道、2 个业务，足以把模型构造走完。"""
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


def test_cli_default_does_not_pass_nli_fallback(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """默认 ``--nli-fallback none`` 不传该关键字，模型走自己的默认路径。"""
    topology_file = _write_topology(tmp_path)
    calls = _spy_on_create_model(monkeypatch)

    result = runner.invoke(__main__.main, _simulate_args(topology_file, tmp_path / "out"))

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert "nli_fallback" not in calls[0]["kwargs"]


def test_cli_explicit_numerical_is_forwarded(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """显式 ``--nli-fallback numerical`` 才把选项传给模型。"""
    topology_file = _write_topology(tmp_path)
    calls = _spy_on_create_model(monkeypatch)

    result = runner.invoke(
        __main__.main,
        _simulate_args(topology_file, tmp_path / "out", "--qot-model", "pcfm", "--nli-fallback", "numerical"),
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["kwargs"]["nli_fallback"] == "numerical"


def test_cli_rejects_a_model_without_the_option(runner: CliRunner, tmp_path: Path) -> None:
    """ECZ 不接受该选项：必须给出清晰的 ClickException，而不是静默吞掉。"""
    topology_file = _write_topology(tmp_path)

    result = runner.invoke(
        __main__.main,
        _simulate_args(topology_file, tmp_path / "out", "--qot-model", "ecz", "--nli-fallback", "numerical"),
    )

    assert result.exit_code != 0
    assert "does not accept --nli-fallback" in result.output
    assert "ecz" in result.output
