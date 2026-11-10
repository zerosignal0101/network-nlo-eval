"""``raman_ode`` 缓存与热路径优化的回归测试（第十二轮 AM）。

覆盖任务的验收点：

* 优化后的 RHS（反向子矩阵乘法、预计算插值、预计算 ASE 源系数）与优化前
  写法**逐位一致**；
* 模块级矩阵缓存命中后 ``_coupling`` 是同一只读对象，跨跨长复用；
* 不同泵浦/栅格不会错误命中；
* 缓存有界（:data:`RAMAN_MATRIX_CACHE_MAX_ENTRIES`）；
* 模型端到端结果与“旧 RHS”逐位一致（C+L 六泵基准）。

默认数值不变的另一半证据在 ``tests/test_pcfm_h3_raman.py`` 的
``GOLDEN_C_L_DEFAULT``：那里钉的是 830c062 基线的 ``float.hex()``。
"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models import pcfm_h3_raman
from network_nlo_eval.models.raman import H_PLANCK
from network_nlo_eval.models.raman_ode import (
    RAMAN_MATRIX_CACHE_MAX_ENTRIES,
    RamanPaperSpanSolver,
    _interp_rows,
    clear_raman_matrix_cache,
    raman_matrix_cache_info,
)
from network_nlo_eval.network.elements import RamanPumpConfig

#: 论文 hybrid 的六泵组（与 ``tests/test_pcfm_h3_raman.py::SIX_PUMPS`` 一致）。
SIX_PUMPS = [
    RamanPumpConfig(wavelength_nm=1340.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1360.0, power_w=0.63, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1427.0, power_w=0.165, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1505.0, power_w=0.075, direction="backward", attenuation_db_per_km=0.25),
]

#: 混合方向的两泵组：FW/BW 两条积分路径都会走。
MIXED_PUMPS = [
    RamanPumpConfig(wavelength_nm=1445.0, power_w=0.131, direction="backward", attenuation_db_per_km=0.25),
    RamanPumpConfig(wavelength_nm=1476.0, power_w=0.125, direction="forward", attenuation_db_per_km=0.25),
]


def _grid(n: int = 12, low_hz: float = 191.0e12, high_hz: float = 196.0e12) -> np.ndarray:
    """一个降序的等距栅格（与调用方约定一致）。"""
    return np.linspace(high_hz, low_hz, n)


def _make_solver(
    *,
    cls: type[RamanPaperSpanSolver] = RamanPaperSpanSolver,
    frequencies: np.ndarray | None = None,
    pumps=SIX_PUMPS,
    length_m: float = 80e3,
    n_steps: int = 24,
    **kwargs,
) -> RamanPaperSpanSolver:
    """用同一套 C 波段光纤参数构造求解器。"""
    f = _grid() if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    n = f.size
    return cls(
        f,
        np.full(n, 75e9),
        list(pumps),
        length_m,
        np.full(n, 0.2 * np.log(10.0) / 10.0 / 1e3),
        np.full(n, 80e-12),
        np.full(len(list(pumps)), 0.25 * np.log(10.0) / 10.0 / 1e3),
        np.full(len(list(pumps)), 70e-12),
        "gaussian",
        n_steps=n_steps,
        **kwargs,
    )


class _LegacyRamanPaperSpanSolver(RamanPaperSpanSolver):
    """优化前的热路径：整矩阵乘法 + 每次 ``searchsorted`` 插值。

    只用于逐位对照，生产代码不再走这些分支。
    """

    def _rhs_forward(self, power, ase, backward_total):
        total = np.empty(self._n_waves, dtype=np.float64)
        total[self._fwd_slots] = power + ase
        total[self._bw_slots] = backward_total
        drive = total * self._drive_scale
        gain_rate = self._coupling @ drive
        gain_fwd = gain_rate[self._fwd_slots]
        d_power = power * gain_fwd - self._alpha_fwd * power
        if not self.include_ase:
            return d_power, np.zeros_like(ase)
        if self._ase_source_scale == 2.0:
            ase_drive = total
        else:
            ase_drive = drive
        source = (
            self._ase_source_scale
            * H_PLANCK
            * self._b_fwd
            * self._f_fwd
            * (self._source_fwd @ ase_drive)
        )
        d_ase = gain_fwd * ase + source - self._alpha_fwd * ase
        return d_power, d_ase

    def _rhs_backward(self, power, ase, forward_total):
        total = forward_total.copy()
        total[self._bw_slots] = power + ase
        drive = total * self._drive_scale
        gain_rate = self._coupling @ drive
        gain_bw = gain_rate[self._bw_slots]
        d_power = power * gain_bw - self._alpha_bw * power
        if not self.include_ase:
            return d_power, np.zeros_like(ase)
        d_ase = gain_bw * ase - self._alpha_bw * ase
        return d_power, d_ase

    def _integrate_backward(self, power_in, ase_in, forward_total):
        n_bw = self._bw_slots.size
        power = power_in.copy()
        ase = ase_in.copy()
        power_grid = np.empty((self.n_steps + 1, n_bw), dtype=np.float64)
        ase_grid = np.empty_like(power_grid)
        power_grid[0] = power
        ase_grid[0] = ase
        for step in range(self.n_steps):
            h = self._z_m[step + 1] - self._z_m[step]
            z0 = self.length_m - self._z_m[step]
            z1 = self.length_m - self._z_m[step + 1]
            z_mid = 0.5 * (z0 + z1)
            total0 = _interp_rows(self._z_m, forward_total, z0)
            total_mid = _interp_rows(self._z_m, forward_total, z_mid)
            total1 = _interp_rows(self._z_m, forward_total, z1)

            dp1, da1 = self._rhs_backward(power, ase, total0)
            dp2, da2 = self._rhs_backward(power + 0.5 * h * dp1, ase + 0.5 * h * da1, total_mid)
            dp3, da3 = self._rhs_backward(power + 0.5 * h * dp2, ase + 0.5 * h * da2, total_mid)
            dp4, da4 = self._rhs_backward(power + h * dp3, ase + h * da3, total1)

            power = power + (h / 6.0) * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4)
            ase = ase + (h / 6.0) * (da1 + 2.0 * da2 + 2.0 * da3 + da4)
            power_grid[step + 1] = power
            ase_grid[step + 1] = ase
        return power, ase, power_grid, ase_grid


def _assert_same_result(actual, expected) -> None:
    """逐位比较两个 :class:`PaperRamanSpanResult`。"""
    for field in ("profile_norm", "power_out_w", "raman_ase_w", "pump_power_out_w"):
        left = np.asarray(getattr(actual, field))
        right = np.asarray(getattr(expected, field))
        assert left.shape == right.shape
        assert left.tobytes() == right.tobytes(), f"{field} drifted"
    assert actual.iterations == expected.iterations
    assert actual.converged == expected.converged


# ---------------------------------------------------------------------------
# 优化前后逐位一致
# ---------------------------------------------------------------------------


def test_rhs_is_bit_identical_to_the_full_matvec_reference() -> None:
    """复用的缓冲与预计算系数必须与逐次分配的旧写法逐位相同。"""
    solver = _make_solver()
    legacy = _make_solver(cls=_LegacyRamanPaperSpanSolver)
    rng = np.random.default_rng(7)
    n_fwd = solver._fwd_slots.size
    n_bw = solver._bw_slots.size
    power = rng.random(n_fwd) * 1e-3
    ase = rng.random(n_fwd) * 1e-9
    backward_total = rng.random(n_bw) * 1e-1
    backward_power = rng.random(n_bw) * 1e-1
    backward_ase = rng.random(n_bw) * 1e-10
    forward_row = rng.random(solver._n_waves) * 1e-3
    forward_row[solver._bw_slots] = rng.random(n_bw)

    fwd_got = solver._rhs_forward(power, ase, backward_total)
    fwd_want = legacy._rhs_forward(power, ase, backward_total)
    assert fwd_got[0].tobytes() == fwd_want[0].tobytes()
    assert fwd_got[1].tobytes() == fwd_want[1].tobytes()

    bwd_got = solver._rhs_backward(backward_power, backward_ase, forward_row)
    bwd_want = legacy._rhs_backward(backward_power, backward_ase, forward_row)
    assert bwd_got[0].tobytes() == bwd_want[0].tobytes()
    assert bwd_got[1].tobytes() == bwd_want[1].tobytes()

    # 复用缓冲不得让结果依赖前一次调用的残留内容：交错调用后仍然逐位相同。
    solver._rhs_backward(backward_power, backward_ase, forward_row)
    again = solver._rhs_forward(power, ase, backward_total)
    assert again[0].tobytes() == fwd_want[0].tobytes()
    assert again[1].tobytes() == fwd_want[1].tobytes()


def test_precomputed_backward_interpolation_matches_interp_rows() -> None:
    """构造期算好的插值索引/权重与 ``_interp_rows`` 逐位等价。"""
    solver = _make_solver(n_steps=9)
    rng = np.random.default_rng(11)
    state = rng.random((solver.n_steps + 1, solver._n_waves))
    grid = solver._z_m
    for step in range(solver.n_steps):
        z0 = solver.length_m - grid[step]
        z1 = solver.length_m - grid[step + 1]
        for column, query in enumerate((z0, 0.5 * (z0 + z1), z1)):
            index = solver._bw_interp_index[step, column]
            weight = solver._bw_interp_weight[step, column]
            got = (1.0 - weight) * state[index] + weight * state[index + 1]
            expected = _interp_rows(grid, state, query)
            assert got.tobytes() == expected.tobytes(), (step, column)


@pytest.mark.parametrize("pumps", [SIX_PUMPS, MIXED_PUMPS])
def test_solver_result_is_bit_identical_to_the_legacy_rhs(pumps) -> None:
    """整跨求解结果与旧 RHS 逐位一致（BW 六泵与 FW/BW 混合）。"""
    rng = np.random.default_rng(3)
    power_in = rng.random(_grid().size) * 1e-3
    ase_in = rng.random(_grid().size) * 1e-10

    new = _make_solver(pumps=pumps).solve(power_in, ase_in)
    legacy = _make_solver(cls=_LegacyRamanPaperSpanSolver, pumps=pumps).solve(power_in, ase_in)
    _assert_same_result(new, legacy)


def test_c_l_six_pump_model_output_is_bit_identical_with_the_legacy_rhs(monkeypatch) -> None:
    """模型端到端：C+L 六泵的 NLI/ASE/SNR 与旧 RHS 逐位一致。"""
    case = build_case("c_57ch")
    path = build_path(case)
    power = launch_profile(case)

    def run(solver_class):
        monkeypatch.setattr(pcfm_h3_raman, "RamanPaperSpanSolver", solver_class)
        clear_raman_matrix_cache()
        model = make_model("pcfm_h3_raman", case, raman_pumps=MIXED_PUMPS, raman_steps=24)
        result = model.evaluate_path(path, power)
        kernel = model.kernel_for(path.fiber_configs[0])
        return result, kernel

    new_result, new_kernel = run(RamanPaperSpanSolver)
    old_result, old_kernel = run(_LegacyRamanPaperSpanSolver)

    for field in ("signal_power_w", "ase_noise_w", "spm_noise_w", "xpm_noise_w", "snr_db"):
        left = np.asarray(getattr(new_result, field))
        right = np.asarray(getattr(old_result, field))
        assert left.tobytes() == right.tobytes(), field
    assert np.asarray(new_kernel.last_pump_power_out_w).tobytes() == (
        np.asarray(old_kernel.last_pump_power_out_w).tobytes()
    )
    assert np.asarray(new_kernel.last_raman_ase_w).tobytes() == (
        np.asarray(old_kernel.last_raman_ase_w).tobytes()
    )
    assert new_kernel.last_raman_iterations == old_kernel.last_raman_iterations


# ---------------------------------------------------------------------------
# 缓存命中、误命中与上限
# ---------------------------------------------------------------------------


def test_matrix_cache_is_shared_across_span_lengths() -> None:
    """同一光纤/泵浦/栅格在不同跨长上复用同一对只读矩阵。"""
    clear_raman_matrix_cache()
    short = _make_solver(length_m=60e3)
    long = _make_solver(length_m=100e3)

    assert short._coupling is long._coupling
    assert short._source is long._source
    assert not short._coupling.flags.writeable
    assert not short._source.flags.writeable

    info = raman_matrix_cache_info()
    assert (info["hits"], info["misses"], info["size"]) == (1, 1, 1)

    rng = np.random.default_rng(5)
    power = rng.random(_grid().size) * 1e-3
    result_short = short.solve(power, None)
    result_long = long.solve(power, None)
    # 跨长不同 -> 剖面权重不同，但矩阵共享不改变任何一位数值。
    assert not np.array_equal(result_short.profile_norm, result_long.profile_norm)


def test_matrix_cache_distinguishes_pumps_and_grids() -> None:
    """不同泵浦或不同栅格必须未命中，且矩阵数值确实不同。"""
    clear_raman_matrix_cache()
    base = _make_solver()
    other_pumps = _make_solver(
        pumps=[RamanPumpConfig(wavelength_nm=1420.0, power_w=0.1, direction="backward", attenuation_db_per_km=0.25)]
    )
    other_grid = _make_solver(frequencies=_grid(n=12, low_hz=191.5e12, high_hz=196.5e12))

    assert base._coupling is not other_pumps._coupling
    assert base._coupling is not other_grid._coupling
    assert not np.array_equal(base._coupling, other_grid._coupling)
    info = raman_matrix_cache_info()
    assert (info["hits"], info["misses"], info["size"]) == (0, 3, 3)


@pytest.mark.parametrize("flag,value", [("include_isrs", False), ("pump_pump_raman", False)])
def test_matrix_cache_distinguishes_switches(flag, value) -> None:
    """会改动矩阵的开关必须进键。"""
    clear_raman_matrix_cache()
    base = _make_solver()
    switched = _make_solver(**{flag: value})
    assert base._coupling is not switched._coupling
    assert not np.array_equal(base._coupling, switched._coupling)


def test_matrix_cache_is_bounded() -> None:
    """构造多于一页的互不相同矩阵时，缓存规模停在上限。"""
    clear_raman_matrix_cache()
    for offset in range(RAMAN_MATRIX_CACHE_MAX_ENTRIES + 4):
        _make_solver(frequencies=_grid(low_hz=190.0e12 + offset * 1e9, high_hz=196.0e12 + offset * 1e9))
    info = raman_matrix_cache_info()
    assert info["size"] == RAMAN_MATRIX_CACHE_MAX_ENTRIES
    assert info["misses"] == RAMAN_MATRIX_CACHE_MAX_ENTRIES + 4
    assert info["max_entries"] == RAMAN_MATRIX_CACHE_MAX_ENTRIES


def test_cache_clear_resets_counters() -> None:
    """``clear_raman_matrix_cache`` 清空条目并复位命中/未命中计数。"""
    _make_solver()
    clear_raman_matrix_cache()
    assert raman_matrix_cache_info() == {
        "size": 0,
        "max_entries": RAMAN_MATRIX_CACHE_MAX_ENTRIES,
        "hits": 0,
        "misses": 0,
    }
