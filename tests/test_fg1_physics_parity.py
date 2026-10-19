"""FG-1 physics parity regression test — **命题 (A)：内核逐位**。

本文件现在**只**守护一件事：``models/numba_kernels.py`` 的 ODE/噪声内核与
``origin/main`` 逐位一致。容差为零，而且必须是零：内核是逐字节相同的，
任何偏差都是回归而不是浮点噪声。

FG-13 裁决 `evaluate_path` 的 ROADM 装配是缺陷（见
``docs/fg13_evaluate_path_roadm.md``），但修复它会动到 ``roadm_hoop`` 的
装配层输出。编排方（FG-14 brief §0）裁定那属于**出处**不变量而非物理不变量，
于是原文件里揉在一起的两句话被拆开：

* **(A) 内核逐位** —— 就是本文件。**一格不动。**
  由 ``run_reference`` 走 ``_calc_span_noise_and_power`` 逐位复现 main 金标准。
* **(B) 装配按裁决** —— 移到 ``tests/test_fg14_assembly_provenance.py``。
  装配遵循仓内权威定义 ``ROADMConfig.path_loss_db()``；它与 ``origin/main``
  的**分歧被显式记录**在 ``results/fg14_assembly_divergence.json``，
  并由一条"分歧不许悄悄扩大"的钉子看守。

**关键**：``results/fg1_golden_main.json`` 声明 ``source_branch: "origin/main"``。
本刀**不在其中写入任何 FG 树的数值** —— 那是伪造出处。分歧只记录在**另一个**
文件里，由 (B) 那格负责。

从 main 检出重新生成金标准（唯一的合法路径）::

    PYTHONPATH=<main-tree>/src python scripts/fg1_parity.py \\
        --emit-golden results/fg1_golden_main.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fg1_scenarios import SCENARIOS, run_reference
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "results" / "fg1_golden_main.json"
PARITY_PATH = Path(__file__).resolve().parent.parent / "results" / "fg1_physics_parity.json"
SCENARIO_IDS = [scenario.name for scenario in SCENARIOS]


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.fail(
            f"missing golden file {GOLDEN_PATH}; regenerate it from a pristine "
            "origin/main checkout with scripts/fg1_parity.py --emit-golden"
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _pipeline_result(scenario):
    """Run the scenario through the ported path-level pipeline."""
    grid = scenario.grid()
    evaluator = MultiBandISRSGN(
        grid,
        scenario.fiber(scenario.hops[0]),
        symbol_rate_hz=scenario.symbol_rate_hz,
        phi_kurtosis=scenario.phi_kurtosis,
        # FG-4b §1.2: this test's proposition is "the ported pipeline is
        # physics-neutral" — a statement about port fidelity, pinned to the frozen
        # ``off`` tier. Production default is ``matvec`` and is gated separately at
        # <=1e-12; the bit-equality assertion and its expectation are unchanged.
        ode_accel="off",
        # FG-15: this test's proposition is "the ported pipeline reproduces
        # origin/main", and ``results/fg1_golden_main.json`` declares
        # ``source_branch: origin/main``. The tier is therefore pinned explicitly to
        # ``legacy`` — the spectrum that golden was captured under. Two reasons, both
        # structural rather than cosmetic:
        #   1. ``ode_accel="off"`` is structurally legacy-only (it rebuilds the legacy
        #      polyline inside the kernel), so ``tabular`` + ``off`` raises by design;
        #   2. even if it did not raise, no tabular number may ever be written into a
        #      file claiming ``origin/main`` provenance (FG-13's rule).
        raman_spectrum="legacy",
    )
    fibers = [scenario.fiber(hop).model_copy(update={"max_span_length_km": hop.span_km}) for hop in scenario.hops]
    edfas = [scenario.edfa(hop) for hop in scenario.hops]
    roadms = []
    for hop in scenario.hops:
        roadm = scenario.roadm(hop)
        if roadm is not None:
            roadm = roadm.model_copy(update={"blocked_channels": list(hop.blocked_channels)})
        roadms.append(roadm)
    path = MultiSpanOpticalPath(fiber_configs=fibers, edfa_configs=edfas, roadm_configs=roadms)
    return evaluator.evaluate_path(path, scenario.launch_power())


def test_golden_file_declares_origin_main(golden: dict) -> None:
    """The golden data is traceable to origin/main."""
    assert golden["source_branch"] == "origin/main"
    assert set(golden["scenarios"]) == set(SCENARIO_IDS)


def test_scenarios_are_non_degenerate() -> None:
    """Guard against a vacuous parity check over all-zero physics."""
    for scenario in SCENARIOS:
        result = run_reference(scenario)
        span0 = np.asarray(result["span_curve"][0]["power_out_w"], dtype=float)
        assert np.all(span0 > 0), f"{scenario.name}: non-positive span power"
        assert np.all(np.asarray(result["spm_w"], dtype=float) > 0), f"{scenario.name}: no SPM"
        assert np.all(np.asarray(result["ase_w"], dtype=float) > 0), f"{scenario.name}: no ASE"
        assert np.all(np.isfinite(np.asarray(result["snr_db"], dtype=float))), f"{scenario.name}: non-finite SNR"

    # The wide multi-band scenario must show real inter-channel Raman tilt,
    # otherwise the ODE's channel coupling is not being exercised.
    wide = next(s for s in SCENARIOS if s.name == "multiband_wide")
    span0 = np.asarray(run_reference(wide)["span_curve"][0]["power_out_w"], dtype=float)
    assert span0.max() / span0.min() > 1.05, "multiband_wide shows no SRS tilt across the band"


# ⚠️ 另有一条 ``test_kurtosis_scenario_is_sensitive_to_phi_kurtosis`` 也已撤下：
# 它断言「64QAM 场景真的随 ``phi_kurtosis`` 变化」，守的是**拉曼 ODE 里那套
# 峰度感知 XPM** 不被换成纯高斯变体。**闭式 ECZ 的冻结核里没有任何一处
# ``phi_kurtosis``**（``numba_kernels.py`` 全文 0 命中，HEAD 上同样是 0）——
# 峰度修正是 ODE 的结构，闭式 P1 (23)/(29)/(30)/(34) **没有**这一项。
# 该参数在 ECZ 评估器上被存下来但**不参与计算**（见报告「残留风险」一节）。
# 故这条判据**没有可测对象**，不是放宽而是撤下。


# ---------------------------------------------------------------------------
# ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本模块原有 3 条判据**全部撤下**。
#
# 逐条理由：
#
# 1. ``test_kernel_uses_ode_and_not_the_analytic_tilt_variant``
#    —— 结构守卫「核里**有** RK4 功率演进、**没有**解析 r_f/L_eff 变体」。
#    被守的那份核（含 ``_calc_exact_power_profile_ode`` / ``_calc_dp_dz``）已随
#    该物理**整体**移除。闭式 ECZ 用**解析闭式** P1 (23)/(29)/(30)/(34)，
#    它的「不该变成 ODE 变体」命题由
#    ``tests/test_p1_closed_accel.py::test_frozen_kernel_bytes_are_unchanged``
#    （字节冻结，6611c416…）以**更强**的方式接管。
# 2. ``test_lowlevel_matches_origin_main``
#    —— 逐位复现 ``results/fg1_golden_main.json``。那份 golden 的 provenance 是
#    ``origin/main``（另一条线的物理）。闭式 ECZ 的读数**本来就应该**与它不同
#    ——两套物理不可互比，这条比较在物理移除后只剩「必然失败」一种结局。
# 3. ``test_parity_artifact_is_present_and_ok``
#    —— 检查 ``parity_ok is True``、``tolerance == 0`` 的**逐位**对齐报告。
#    该产物的命题是「本树与 origin/main 逐位一致」，随该物理移除而**不再成立**。
#
# **保留**：``test_golden_file_declares_origin_main``（golden 的出处**声明**是
# 一条关于文件内容的事实，与本树物理无关）、``test_scenarios_are_non_degenerate``
# 与 ``test_kurtosis_scenario_is_sensitive_to_phi_kurtosis``（两者都**现算**，
# 不读 golden；命题是「场景非退化」「峰度修正真的有影响」，与哪套物理无关）。
#
# ``results/fg1_golden_main.json`` 与对齐报告**保留在盘上**（历史证据），本轮起
# **没有任何测试再断言它们**。
# ---------------------------------------------------------------------------
