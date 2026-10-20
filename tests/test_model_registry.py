"""Contract tests for the pluggable QoT model registry and the shared route layer."""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import (
    MODEL_REGISTRY,
    SpanKernel,
    available_models,
    create_model,
    describe_models,
    get_model_info,
    register_model,
)
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
from network_nlo_eval.models.route import RouteModel, accumulate_route, apply_amplifier_and_roadm
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig


def _fiber() -> FiberSpanConfig:
    return FiberSpanConfig(length_km=80.0, max_span_length_km=80.0)


def test_ecz_is_registered_and_constructible() -> None:
    """The built-in model is reachable by name, not only by class import."""
    assert "ecz" in MODEL_REGISTRY
    info = get_model_info("ecz")
    assert info.family == "gn"
    assert "Lasagni" in info.reference

    evaluator = create_model("ecz", SpectrumGrid(8, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9)
    assert isinstance(evaluator, MultiBandISRSGN)


def test_registered_model_satisfies_route_protocol() -> None:
    """Whatever the registry returns must expose the five members the route layer reads."""
    evaluator = create_model("ecz", SpectrumGrid(8, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9)
    assert evaluator.n_nodes == 8
    assert np.allclose(evaluator.power_width_scale(), 1.0)
    assert evaluator.blocked_nodes(ROADMConfig(blocked_channels=[1, 3])) == [1, 3]
    step = evaluator.span_step(_fiber(), EDFAConfig(), np.full(8, 1e-3), 80e3)
    assert step.power_out_w.shape == (8,)
    assert np.all(step.ase_noise_w > 0)


def test_unknown_model_names_are_rejected() -> None:
    """A typo in --qot-model must fail loudly and list the alternatives."""
    with pytest.raises(KeyError, match="ecz"):
        create_model("no-such-model", SpectrumGrid(4, 50e9, 193.1e12), _fiber())
    with pytest.raises(KeyError):
        get_model_info("no-such-model")


def test_duplicate_registration_is_rejected() -> None:
    """Registering the same name twice would make model selection ambiguous."""
    info = get_model_info("ecz")
    with pytest.raises(ValueError, match="already registered"):
        register_model(info, MultiBandISRSGN)


def test_describe_models_is_json_ready() -> None:
    """The CLI prints this payload, so it must not carry numpy or classes."""
    entries = describe_models()
    assert entries
    entry = next(item for item in entries if item["name"] == "ecz")
    assert isinstance(entry["options"], list)
    assert all(isinstance(option, list) and len(option) == 2 for option in entry["options"])
    assert set(available_models()) == {item["name"] for item in entries}


def test_route_layer_owns_amplifier_physics() -> None:
    """ASE appears once per span and only when an amplifier is configured."""
    kernel_like = type(
        "StubKernel",
        (),
        {
            "evaluate": lambda self, power, span_length_m, n=1: __import__(
                "network_nlo_eval.models.base", fromlist=["SpanFiberOutput"]
            ).SpanFiberOutput(power * 0.5, np.zeros_like(power), np.zeros_like(power))
        },
    )()

    fiber = _fiber()
    evaluator = MultiBandISRSGN(SpectrumGrid(4, 50e9, 193.1e12), fiber, symbol_rate_hz=50e9)
    power = np.full(4, 1e-3)
    fiber_output = kernel_like.evaluate(power, 80e3)

    without = apply_amplifier_and_roadm(
        power_in_w=power,
        fiber_output=fiber_output,
        edfa_config=None,
        roadm_config=None,
        nf_lin=np.ones(4),
        f_abs_hz=evaluator.f_abs_hz,
        bandwidth_hz=evaluator.channel_bandwidth_hz_channels,
    )
    assert np.all(without.ase_noise_w == 0.0)
    assert np.allclose(without.power_out_w, power * 0.5)

    with_edfa = apply_amplifier_and_roadm(
        power_in_w=power,
        fiber_output=fiber_output,
        edfa_config=EDFAConfig(noise_figure_db=5.0),
        roadm_config=None,
        nf_lin=np.full(4, 10**0.5),
        f_abs_hz=evaluator.f_abs_hz,
        bandwidth_hz=evaluator.channel_bandwidth_hz_channels,
    )
    # Gain-None means "restore span loss" -> power back to the launch value.
    assert np.allclose(with_edfa.power_out_w, power)
    assert np.all(with_edfa.ase_noise_w > 0.0)


def test_span_kernel_protocol_is_structural() -> None:
    """Kernels are duck-typed; the protocol must accept any object with evaluate()."""
    kernel = type("K", (), {"evaluate": lambda self, power, span_length_m, n=1: None})()
    assert isinstance(kernel, SpanKernel)


def test_accumulate_route_rejects_bad_power() -> None:
    """Negative power and wrong length are caller errors, not silent clamps."""
    evaluator = MultiBandISRSGN(SpectrumGrid(4, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9)
    path = MultiSpanOpticalPath([_fiber()], [EDFAConfig()])
    with pytest.raises(ValueError, match="non-negative"):
        accumulate_route(evaluator, path, np.array([1e-3, -1e-3, 1e-3, 1e-3]))
    with pytest.raises(ValueError, match="non-negative"):
        accumulate_route(evaluator, path, np.full(3, 1e-3))


def test_route_model_requires_a_kernel() -> None:
    """The shared base must refuse to evaluate without a physics kernel."""
    model = RouteModel(SpectrumGrid(4, 50e9, 193.1e12), _fiber(), symbol_rate_hz=50e9)
    with pytest.raises(NotImplementedError):
        model.kernel_for(_fiber())


@pytest.mark.parametrize("model_name", sorted(available_models()))
def test_model_output_does_not_depend_on_call_history(model_name: str) -> None:
    """同一 (path, power) 必须给出逐位相同的 SNR，不管之前调过什么。

    QoT 引擎在一次路由搜索里会对同一模型反复评估不同路径和功率。任何跨调用累积的
    状态都会让"同一个问题得到不同答案"，而且这种 bug 在单点测试里看不出来：
    它只在扫码或复用实例时出现。真实出现过一次（cGN 功率键缓存），这里把它钉死。
    """
    from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
    from network_nlo_eval.benchmarks.runner import make_model

    case = build_case("scaling_40ch")
    path = build_path(case)

    def snr(model: object, power_dbm: float) -> np.ndarray:
        result = model.evaluate_path(path, launch_profile(case, power_dbm))
        return np.asarray(result.snr_db, dtype=float)

    fresh = snr(make_model(model_name, case), -4.0)

    reused = make_model(model_name, case)
    snr(reused, 0.0)
    snr(reused, 4.0)
    after = snr(reused, -4.0)
    again = snr(reused, -4.0)

    assert np.array_equal(fresh, after), f"{model_name}: 先调 0/+4 dBm 改变了 -4 dBm 的结果"
    assert np.array_equal(after, again), f"{model_name}: 同一输入连调两次结果不同"
