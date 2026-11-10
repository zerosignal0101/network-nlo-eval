"""Opt-in ``raman_backend`` of the paper Raman ODE（第十三轮 AO）。

覆盖任务验收点：

* 默认 ``raman_backend="python"`` 不导入 numba、不编译，数值逐位不变；
* ``"numba"`` 与 ``"python"`` 的前向泵、后向泵、混合泵与无泵结果一致到
  ``1e-12`` 相对差（实测逐位相同，见 ``H3_TASK_REPORT.md``）；
* ``"auto"`` 的选择逻辑（numba 缺失 / 步数不足时退回 python）；
* 非法后端在 solver / kernel / model 三层都被拒绝；
* ``lumped_loss`` + ``"numba"`` 明确 ``ValueError``，``"auto"`` 退回 python；
* 既有 opt-in（``raman_resolution_check``、``modulation_format_correction``）
  的语义不因后端而改变。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models import raman_ode
from network_nlo_eval.models.pcfm_h3_raman import PCFH3RamanSpanKernel
from network_nlo_eval.models.raman_ode import (
    AUTO_NUMBA_MIN_STEPS,
    RamanPaperSpanSolver,
    numba_available,
    resolve_raman_backend,
    validate_raman_backend,
)
from network_nlo_eval.network.elements import RamanPumpConfig

#: 论文 hybrid 的六泵组（与 ``tests/test_pcfm_h3_raman.py`` 一致）。
SIX_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]

#: 单后向泵。
ONE_PUMP = [
    RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction="backward", attenuation_db_per_km=0.25)
]

#: FW/BW 混合，两条积分路径都会走。
MIXED_PUMPS = [
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="forward", attenuation_db_per_km=0.25),
]

#: solver 级一致性用例：``(id, pumps)``。
PUMP_CASES = [
    pytest.param([], id="nopump"),
    pytest.param(ONE_PUMP, id="backward"),
    pytest.param(MIXED_PUMPS, id="mixed"),
    pytest.param(SIX_PUMPS, id="six_backward"),
]

_RESULT_FIELDS = ("profile_norm", "power_out_w", "raman_ase_w", "pump_power_out_w")
_PATH_FIELDS = ("signal_power_w", "ase_noise_w", "spm_noise_w", "xpm_noise_w", "snr_db")

RAMAN_NUMBA_MODULE = "network_nlo_eval.models.raman_numba"


def _grid(n: int = 24, low_hz: float = 191.0e12, high_hz: float = 196.0e12) -> np.ndarray:
    """降序等距栅格。"""
    return np.linspace(high_hz, low_hz, n)


def _make_solver(*, pumps=SIX_PUMPS, n_steps: int = 32, backend: str = "python", **kwargs):
    """构造一个 C 波段求解器。"""
    frequencies = _grid()
    n = frequencies.size
    return RamanPaperSpanSolver(
        frequencies,
        np.full(n, 75e9),
        list(pumps),
        kwargs.pop("length_m", 80e3),
        np.full(n, 0.2 * np.log(10.0) / 10.0 / 1e3),
        np.full(n, 80e-12),
        np.full(len(list(pumps)), 0.25 * np.log(10.0) / 10.0 / 1e3),
        np.full(len(list(pumps)), 70e-12),
        "gaussian",
        n_steps=n_steps,
        backend=backend,
        **kwargs,
    )


def _inputs(n: int = 24):
    """固定的信号功率与 ASE 输入。"""
    rng = np.random.default_rng(17)
    return rng.random(n) * 1e-3, rng.random(n) * 1e-10


def _assert_bit_identical(left, right, fields=_RESULT_FIELDS) -> None:
    """逐位比较两组结果。"""
    for field in fields:
        got = np.asarray(getattr(left, field))
        want = np.asarray(getattr(right, field))
        assert got.shape == want.shape, field
        assert got.tobytes() == want.tobytes(), f"{field} drifted"


def _assert_close(left, right, fields=_RESULT_FIELDS, rtol: float = 1e-12) -> None:
    """按相对差比较两组结果（零值用绝对差；空数组直接跳过）。"""
    for field in fields:
        got = np.asarray(getattr(left, field), dtype=np.float64)
        want = np.asarray(getattr(right, field), dtype=np.float64)
        assert got.shape == want.shape, field
        if got.size == 0:
            continue
        scale = np.maximum(np.abs(want), np.finfo(np.float64).tiny)
        relative = np.abs(got - want) / scale
        assert float(np.max(relative)) <= rtol, f"{field} relative diff {np.max(relative):.3e}"


# ---------------------------------------------------------------------------
# 默认路径：不导入 numba、逐位不变
# ---------------------------------------------------------------------------


def test_default_backend_is_python_and_compiles_nothing(monkeypatch) -> None:
    """默认 solver 解析到 python；求值后 JIT 内核没有任何编译签名。"""
    monkeypatch.delitem(sys.modules, RAMAN_NUMBA_MODULE, raising=False)
    solver = _make_solver()
    assert solver.backend == "python"
    assert solver._resolved_backend == "python"

    power, ase = _inputs()
    solver.solve(power, ase)
    # 模型发现会导入 raman_numba 模块本身；这里断言的是“没有编译”。
    kernels = sys.modules.get(RAMAN_NUMBA_MODULE)
    if kernels is not None:
        assert kernels.rk4_forward.signatures == []
        assert kernels.rk4_backward.signatures == []


def test_explicit_python_backend_matches_the_default_bit_for_bit() -> None:
    """显式 ``backend="python"`` 与默认值逐位相同（默认就是 python）。"""
    power, ase = _inputs()
    default = _make_solver().solve(power, ase)
    explicit = _make_solver(backend="python").solve(power, ase)
    _assert_bit_identical(default, explicit)


def test_default_model_construction_does_not_compile_jit_kernels() -> None:
    """全新进程里默认模型求值后，两个 JIT 内核仍然没有编译签名。"""
    script = (
        "import numpy as np\n"
        "from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile\n"
        "from network_nlo_eval.benchmarks.runner import make_model\n"
        "from network_nlo_eval.models import raman_numba\n"
        "from network_nlo_eval.network.elements import RamanPumpConfig\n"
        "case = build_case('c_57ch')\n"
        "pump = RamanPumpConfig(wavelength_nm=1450.0, power_w=0.2, direction='backward',"
        " attenuation_db_per_km=0.25)\n"
        "model = make_model('pcfm_h3_raman', case, raman_pumps=[pump], raman_steps=24)\n"
        "assert model.raman_backend == 'python'\n"
        "model.evaluate_path(build_path(case), launch_profile(case))\n"
        "print(len(raman_numba.rk4_forward.signatures), len(raman_numba.rk4_backward.signatures))\n"
    )
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and script, no user input.
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert completed.stdout.strip().splitlines()[-1] == "0 0"


# ---------------------------------------------------------------------------
# numba / python 一致性
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not numba_available(), reason="numba is not importable")
@pytest.mark.parametrize("pumps", PUMP_CASES)
def test_numba_matches_python_to_one_e_minus_twelve(pumps) -> None:
    """前向泵、后向泵、混合与无泵都与 python 后端一致到 ``1e-12``。"""
    power, ase = _inputs()
    reference = _make_solver(pumps=pumps, backend="python").solve(power, ase)
    jitted = _make_solver(pumps=pumps, backend="numba").solve(power, ase)
    _assert_close(jitted, reference)
    assert jitted.iterations == reference.iterations
    assert jitted.converged == reference.converged


@pytest.mark.skipif(not numba_available(), reason="numba is not importable")
def test_numba_records_a_jit_signature() -> None:
    """``"numba"`` 后端确实调用了编译内核（而不是静默走 python）。"""
    solver = _make_solver(backend="numba")
    assert solver._resolved_backend == "numba"
    power, ase = _inputs()
    solver.solve(power, ase)

    kernels = sys.modules[RAMAN_NUMBA_MODULE]
    assert kernels.rk4_forward.signatures
    assert kernels.rk4_backward.signatures


@pytest.mark.skipif(not numba_available(), reason="numba is not importable")
def test_numba_model_outputs_match_python_for_c_l_six_pumps() -> None:
    """模型端到端：C+L 六泵的 NLI/ASE/SNR 与 python 后端一致。"""
    case = build_case("c_57ch")
    path = build_path(case)
    power = launch_profile(case)
    python = make_model("pcfm_h3_raman", case, raman_pumps=SIX_PUMPS, raman_steps=24)
    jitted = make_model(
        "pcfm_h3_raman", case, raman_pumps=SIX_PUMPS, raman_steps=24, raman_backend="numba"
    )
    reference = python.evaluate_path(path, power)
    got = jitted.evaluate_path(path, power)
    _assert_close(got, reference, fields=_PATH_FIELDS)

    python_kernel = python.kernel_for(path.fiber_configs[0])
    jit_kernel = jitted.kernel_for(path.fiber_configs[0])
    for name in ("last_pump_power_out_w", "last_raman_ase_w"):
        assert np.asarray(getattr(jit_kernel, name)).shape == np.asarray(
            getattr(python_kernel, name)
        ).shape
        np.testing.assert_allclose(
            np.asarray(getattr(jit_kernel, name)),
            np.asarray(getattr(python_kernel, name)),
            rtol=1e-12,
            atol=0.0,
        )


# ---------------------------------------------------------------------------
# auto 选择逻辑与 numba 缺失
# ---------------------------------------------------------------------------


def test_validate_raman_backend_accepts_only_documented_values() -> None:
    """三个合法值原样返回，其他值 ``ValueError``。"""
    for name in ("python", "numba", "auto"):
        assert validate_raman_backend(name) == name
    for bad in ("jit", "Python", "", None, 1):
        with pytest.raises(ValueError, match="raman_backend"):
            validate_raman_backend(bad)  # type: ignore[arg-type]


def test_resolve_backend_auto_threshold_and_fallback() -> None:
    """``auto`` 只在大步数 + numba 可用时选 numba。"""
    assert resolve_raman_backend("auto", n_steps=AUTO_NUMBA_MIN_STEPS, available=True) == "numba"
    assert resolve_raman_backend("auto", n_steps=AUTO_NUMBA_MIN_STEPS - 1, available=True) == "python"
    assert resolve_raman_backend("auto", n_steps=96, available=False) == "python"
    assert resolve_raman_backend("auto", n_steps=96, n_lumped_loss=1, available=True) == "python"
    assert resolve_raman_backend("python", n_steps=96, available=True) == "python"


def test_explicit_numba_without_numba_raises_runtime_error(monkeypatch) -> None:
    """``"numba"`` 不可用时给出清晰 ``RuntimeError``，不退化成 python。"""
    monkeypatch.setattr(raman_ode, "numba_available", lambda: False)
    with pytest.raises(RuntimeError, match="numba"):
        resolve_raman_backend("numba", n_steps=96)
    with pytest.raises(RuntimeError, match="numba"):
        _make_solver(backend="numba")


def test_auto_falls_back_to_python_when_numba_is_missing(monkeypatch) -> None:
    """``auto`` + numba 缺失：solver 解析到 python 且结果与 python 逐位相同。"""
    monkeypatch.setattr(raman_ode, "numba_available", lambda: False)
    solver = _make_solver(backend="auto", n_steps=96)
    assert solver.backend == "auto"
    assert solver._resolved_backend == "python"

    power, ase = _inputs()
    auto = solver.solve(power, ase)
    python = _make_solver(backend="python", n_steps=96).solve(power, ase)
    _assert_bit_identical(auto, python)


def test_auto_uses_python_below_the_step_threshold(monkeypatch) -> None:
    """``auto`` 在步数不足时选 python，即使 numba 可用。"""
    monkeypatch.setattr(raman_ode, "numba_available", lambda: True)
    solver = _make_solver(backend="auto", n_steps=AUTO_NUMBA_MIN_STEPS - 1)
    assert solver._resolved_backend == "python"


def test_auto_selects_numba_at_or_above_the_threshold(monkeypatch) -> None:
    """``auto`` 在步数达标且 numba 可用时选 numba。"""
    monkeypatch.setattr(raman_ode, "numba_available", lambda: True)
    solver = _make_solver(backend="auto", n_steps=AUTO_NUMBA_MIN_STEPS)
    assert solver._resolved_backend == "numba"


# ---------------------------------------------------------------------------
# 非法值 / lumped_loss 边界
# ---------------------------------------------------------------------------


def test_invalid_backend_is_rejected_by_solver_kernel_and_model() -> None:
    """三层参数校验都只接受三个合法值。"""
    with pytest.raises(ValueError, match="raman_backend"):
        _make_solver(backend="jit")
    case = build_case("c_57ch")
    fiber = build_path(case).fiber_configs[0]
    with pytest.raises(ValueError, match="raman_backend"):
        PCFH3RamanSpanKernel(
            np.asarray(_grid(8)),
            np.full(8, 75e9),
            fiber,
            pumps=ONE_PUMP,
            gain_model="gaussian",
            raman_backend="jit",
        )
    with pytest.raises(ValueError, match="raman_backend"):
        make_model("pcfm_h3_raman", case, raman_backend="jit")


def test_numba_with_lumped_loss_raises_value_error() -> None:
    """``"numba"`` + 集中损耗：显式拒绝，不静默换路径。"""
    with pytest.raises(ValueError, match="lumped"):
        _make_solver(backend="numba", lumped_loss=[(40e3, 0.5)])
    case = build_case("c_57ch")
    fiber = build_path(case).fiber_configs[0]
    with pytest.raises(ValueError, match="lumped"):
        PCFH3RamanSpanKernel(
            np.asarray(_grid(8)),
            np.full(8, 75e9),
            fiber,
            pumps=ONE_PUMP,
            gain_model="gaussian",
            lumped_loss=[(40e3, 0.5)],
            raman_backend="numba",
        )
    with pytest.raises(ValueError, match="lumped"):
        make_model(
            "pcfm_h3_raman",
            case,
            raman_pumps=ONE_PUMP,
            lumped_loss=[(40e3, 0.5)],
            raman_backend="numba",
        )


def test_auto_with_lumped_loss_uses_the_python_segmented_path() -> None:
    """``auto`` + 集中损耗按定义退回 python，且结果与 python 后端逐位相同。"""
    power, ase = _inputs()
    auto = _make_solver(backend="auto", lumped_loss=[(30e3, 0.3), (55e3, 0.2)]).solve(power, ase)
    python = _make_solver(backend="python", lumped_loss=[(30e3, 0.3), (55e3, 0.2)]).solve(power, ase)
    _assert_bit_identical(auto, python)
    assert auto.sample_positions_u is not None


# ---------------------------------------------------------------------------
# 既有 opt-in 语义不变
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not numba_available(), reason="numba is not importable")
def test_resolution_audit_semantics_do_not_depend_on_the_backend() -> None:
    """``raman_resolution_check`` 的审计在两种后端下给出同一结论。"""
    case = replace(build_case("c_57ch"), name="c_57ch_1span", span_count=1)
    path = build_path(case)
    power = launch_profile(case)
    python = make_model(
        "pcfm_h3_raman",
        case,
        raman_pumps=ONE_PUMP,
        raman_steps=12,
        raman_resolution_check=True,
    )
    jitted = make_model(
        "pcfm_h3_raman",
        case,
        raman_pumps=ONE_PUMP,
        raman_steps=12,
        raman_resolution_check=True,
        raman_backend="numba",
    )
    reference = python.evaluate_path(path, power)
    got = jitted.evaluate_path(path, power)
    _assert_close(got, reference, fields=_PATH_FIELDS)

    python_report = python.kernel_for(path.fiber_configs[0]).validity_report(
        100e3, np.asarray(power), power_out_w=np.asarray(reference.signal_power_w)
    )
    jit_report = jitted.kernel_for(path.fiber_configs[0]).validity_report(
        100e3, np.asarray(power), power_out_w=np.asarray(got.signal_power_w)
    )
    assert python_report.raman_resolution_checked is True
    assert jit_report.raman_resolution_checked is True
    assert python_report.raman_auto_under_resolved == jit_report.raman_auto_under_resolved
    assert python_report.raman_resolution_check_steps == jit_report.raman_resolution_check_steps


@pytest.mark.skipif(not numba_available(), reason="numba is not importable")
def test_modulation_format_correction_semantics_do_not_depend_on_the_backend() -> None:
    """2019b 调制格式修正仍只改 XCI，且与后端无关。"""
    case = replace(build_case("c_57ch"), name="c_57ch_1span", span_count=1)
    path = build_path(case)
    power = launch_profile(case)
    options = dict(
        egn=False,
        modulation_format_correction=True,
        excess_kurtosis=0.3,
        raman_pumps=ONE_PUMP,
        raman_steps=12,
    )
    reference = make_model("pcfm_h3_raman", case, **options).evaluate_path(path, power)
    got = make_model("pcfm_h3_raman", case, raman_backend="numba", **options).evaluate_path(path, power)
    _assert_close(got, reference, fields=_PATH_FIELDS)


def test_python_backend_keeps_the_uniform_sampling_contract() -> None:
    """Python 后端的等距采样契约（``sample_positions_u is None``）不变。"""
    power, ase = _inputs()
    solver = _make_solver(backend="python")
    result = solver.solve(power, ase)
    assert result.sample_positions_u is None
    assert result.profile_norm.shape == (2 * solver.n_steps, _grid().size)
