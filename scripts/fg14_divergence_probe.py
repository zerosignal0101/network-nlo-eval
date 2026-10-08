#!/usr/bin/env python
"""FG-14：出处分歧框架——把 FG-1 金标准里的两个命题拆开，并各自钉住。

`results/fg1_golden_main.json` 声明 ``source_branch: "origin/main"``，
`tests/test_fg1_physics_parity.py` 曾把**两件不同的事**揉进同一句话：

* **(A) 内核逐位** —— ODE / 噪声内核与 ``origin/main`` 逐位一致。
  这是真正的重构保证，由 FG-11 复核过（``off ≡ matvec ≤ 1.8e-15``，G1 31/31 佐证）。
* **(B) 装配按裁决** —— 设备装配（ROADM / booster / 端口）遵循**仓内权威定义**
  ``ROADMConfig.path_loss_db()``（``network/elements.py:113-116``）。
  它与 ``origin/main`` 的**分歧是被授权的**，幅度必须被显式记录并钉住。

编排方裁定：两条**都算**不变量，但 (B) 是**出处**不变量而不是**物理**不变量。
⇒ 本脚本**不改** ``fg1_golden_main.json`` 的任何一个数字（那是伪造出处）；
它把 (B) 的分歧量到一个**新文件**里，配上机制与授权来源，并留一条
"分歧不许悄悄扩大"的钉子。

三个档位（brief §0 要求"仅端口化"那一档也留在记录里，让选择可复核）：

``committed``
    本仓当前代码的实测分歧。FG-14 阶段 0（零物理变化）应为 **0.0 逐位**；
    阶段 1 合入权威装配后即为那组被授权的分歧。
``prospective_full_union``
    进程内替换成权威口径（含 booster 注入），**不写仓库任何文件**。
``prospective_ports_only``
    只把 ``insertion_loss_db`` 换成 ``path_loss_db(port)``，其余行为一律不动
    —— 这是**单变量对照**，用来把"端口语义"与"booster/滤波重排"两笔账分开。

复算：
    PYTHONPATH=$PWD/src python scripts/fg14_divergence_probe.py --check
    PYTHONPATH=$PWD/src python scripts/fg14_divergence_probe.py --mutation 1e-6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from network_nlo_eval.models.isrs_gn import (  # noqa: E402
    MultiBandISRSGN,
    MultiSpanOpticalPath,
    PathQoTResult,
)
from network_nlo_eval.rwa.qot_checker import fold_roadm_terms  # noqa: E402

GOLDEN_PATH = REPO_ROOT / "results" / "fg1_golden_main.json"
RECORD_PATH = REPO_ROOT / "results" / "fg14_assembly_divergence.json"
BASELINE_PATH = REPO_ROOT / "results" / "fg14_pre_fix_baseline.json"

COMMITTED_EVALUATE_PATH = MultiBandISRSGN.evaluate_path

#: ``evaluate_path`` 的属性名 → main 金标准里的键名。
OUTPUT_KEYS = ("signal_power_w", "spm_noise_w", "xpm_noise_w", "ase_noise_w", "snr_db")
GOLDEN_KEYS = {
    "signal_power_w": "signal_power_w",
    "spm_noise_w": "spm_w",
    "xpm_noise_w": "xpm_w",
    "ase_noise_w": "ase_w",
    "snr_db": "snr_db",
}

#: (B) 的钉子精度。与仓内既有 golden 纪律一致：确定性重算要求**逐位**，
#: 所以"不许悄悄扩大"的下界也必须是精确比较，而不是一个宽松的浮点容差。
#: 变异测试用它证明这格不是真空断言。
TOLERANCE = 0.0


# ======================================================================
# 候选装配：两个**进程内**替身，都不写仓库文件
# ======================================================================


def evaluate_path_full_union(
    self: MultiBandISRSGN,
    optical_path: MultiSpanOpticalPath,
    power_in_w: np.ndarray,
) -> PathQoTResult:
    """**完整并集**：按端口取损耗 + booster ASE 注入 + 滤波作独立噪声乘子。

    逐字对齐 ``rwa/qot_checker.accumulate_service_snr`` 的 ROADM 段，并与
    波形级装配 ``physics/qot_checker.py:248-261`` 同口径（编排方 §0 采纳档）。

    刻意**不**引入 ``accumulate_service_snr`` 末尾的 ``total_spans ** 0.05``：
    那属于服务 SNR 公式、不是 ROADM 装配，混进来会把两个变量搅在一起。
    """
    power = np.asarray(power_in_w, dtype=np.float64).copy()
    if power.shape != (self.n_nodes,) or np.any(power < 0):
        raise ValueError("power_in_w must be a non-negative vector with one value per connection.")
    total_spm = np.zeros_like(power)
    total_xpm = np.zeros_like(power)
    total_ase = np.zeros_like(power)
    span_count = 0
    roadm_configs = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
    n_hops = len(optical_path.fiber_configs)

    for hop_index, (fiber, edfa, roadm) in enumerate(
        zip(optical_path.fiber_configs, optical_path.edfa_configs, roadm_configs, strict=True)
    ):
        evaluator = self.for_fiber(fiber)
        remaining_km = fiber.length_km
        while remaining_km > 1e-12:
            span_km = min(remaining_km, fiber.max_span_length_km)
            previous_power = power
            power, spm, xpm, ase = evaluator._calc_span_noise_and_power(
                power, span_km * 1e3, edfa, n_effective_spans=1
            )
            transfer = np.divide(power, previous_power, out=np.zeros_like(power), where=previous_power > 0)
            total_spm *= transfer
            total_xpm *= transfer
            total_ase *= transfer
            total_spm += spm
            total_xpm += xpm
            total_ase += ase
            span_count += 1
            remaining_km -= span_km

        if roadm is None:
            continue
        terms = fold_roadm_terms(roadm, is_last_hop=hop_index == n_hops - 1)
        if terms.equalize:
            total_ase += (
                terms.booster_base
                * np.asarray(self.f_abs_hz, dtype=np.float64)
                * np.asarray(self.symbol_rate_hz_channels, dtype=np.float64)
                * (terms.loss - 1.0)
            )
        else:
            power /= terms.loss
            total_spm /= terms.loss
            total_xpm /= terms.loss
            total_ase /= terms.loss
        total_spm *= terms.filtering_penalty
        total_xpm *= terms.filtering_penalty
        total_ase *= terms.filtering_penalty
        blocked = self._resolve_blocked_node_indices(roadm)
        if blocked:
            power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0

    total_noise = total_spm + total_xpm + total_ase
    snr = np.full_like(power, -np.inf)
    active = (power > 0) & (total_noise > 0)
    snr[active] = 10 * np.log10(power[active] / total_noise[active])
    snr[(power > 0) & (total_noise == 0)] = np.inf
    return PathQoTResult(
        power, total_spm, total_xpm, total_ase, snr, span_count, sum(r is not None for r in roadm_configs)
    )


def evaluate_path_ports_only(
    self: MultiBandISRSGN,
    optical_path: MultiSpanOpticalPath,
    power_in_w: np.ndarray,
) -> PathQoTResult:
    """**仅端口化**：单变量对照。

    与仓库现状相比**只有一处**不同：``insertion_loss_db`` → ``path_loss_db(port)``。
    滤波仍并入损耗、booster 仍不注入、损耗仍同时除信号与全部累积噪声。

    这一档**不是**候选答案——它只修一半（``equalize_output_power`` 与
    ``booster_noise_figure_db`` 仍是惰性字段，FG-13 §2 实测）。它的作用是让
    "采纳完整并集"这个选择**可复核**：读者能看清端口语义自己值多少 dB、
    booster 重排又添了多少 dB。
    """
    power = np.asarray(power_in_w, dtype=np.float64).copy()
    if power.shape != (self.n_nodes,) or np.any(power < 0):
        raise ValueError("power_in_w must be a non-negative vector with one value per connection.")
    total_spm = np.zeros_like(power)
    total_xpm = np.zeros_like(power)
    total_ase = np.zeros_like(power)
    span_count = 0
    roadm_configs = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
    n_hops = len(optical_path.fiber_configs)

    for hop_index, (fiber, edfa, roadm) in enumerate(
        zip(optical_path.fiber_configs, optical_path.edfa_configs, roadm_configs, strict=True)
    ):
        evaluator = self.for_fiber(fiber)
        remaining_km = fiber.length_km
        while remaining_km > 1e-12:
            span_km = min(remaining_km, fiber.max_span_length_km)
            previous_power = power
            power, spm, xpm, ase = evaluator._calc_span_noise_and_power(
                power, span_km * 1e3, edfa, n_effective_spans=1
            )
            transfer = np.divide(power, previous_power, out=np.zeros_like(power), where=previous_power > 0)
            total_spm *= transfer
            total_xpm *= transfer
            total_ase *= transfer
            total_spm += spm
            total_xpm += xpm
            total_ase += ase
            span_count += 1
            remaining_km -= span_km

        if roadm is None:
            continue
        operation = "drop" if hop_index == n_hops - 1 else "express"
        loss = 10 ** ((roadm.path_loss_db(operation) + roadm.filtering_penalty_db) / 10)
        power /= loss
        total_spm /= loss
        total_xpm /= loss
        total_ase /= loss
        blocked = self._resolve_blocked_node_indices(roadm)
        if blocked:
            power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0

    total_noise = total_spm + total_xpm + total_ase
    snr = np.full_like(power, -np.inf)
    active = (power > 0) & (total_noise > 0)
    snr[active] = 10 * np.log10(power[active] / total_noise[active])
    snr[(power > 0) & (total_noise == 0)] = np.inf
    return PathQoTResult(
        power, total_spm, total_xpm, total_ase, snr, span_count, sum(r is not None for r in roadm_configs)
    )


TIERS: dict[str, Any] = {
    "committed": COMMITTED_EVALUATE_PATH,
    "prospective_full_union": evaluate_path_full_union,
    "prospective_ports_only": evaluate_path_ports_only,
}


# ======================================================================
# 逐位比较
# ======================================================================


def compare(golden: Any, candidate: Any, *, inflate: float = 0.0) -> dict[str, Any]:
    """逐位比较 main 金标准与本仓读数；非有限值同侧视为相等。"""
    g = np.asarray(golden, dtype=np.float64)
    c = np.asarray(candidate, dtype=np.float64) + inflate
    if g.shape != c.shape:
        return {"comparable": False, "reason": f"shape mismatch {g.shape} vs {c.shape}"}
    g_finite, c_finite = np.isfinite(g), np.isfinite(c)
    nf_mismatch = int(np.count_nonzero(g_finite != c_finite))
    gf, cf = g[g_finite], c[c_finite]
    if gf.size == 0:
        return {
            "comparable": True,
            "n_points": 0,
            "max_abs_dev": 0.0,
            "max_rel_dev": 0.0,
            "non_finite_mismatch": nf_mismatch,
            "bit_exact": nf_mismatch == 0,
        }
    abs_dev = np.abs(gf - cf)
    rel_dev = abs_dev / np.maximum(np.abs(gf), 1e-300)
    return {
        "comparable": True,
        "n_points": int(gf.size),
        "max_abs_dev": float(np.max(abs_dev)),
        "max_rel_dev": float(np.max(rel_dev)),
        "non_finite_mismatch": nf_mismatch,
        "bit_exact": bool(np.max(abs_dev) == 0.0 and nf_mismatch == 0),
    }


def _scenarios() -> list[Any]:
    """复用 FG-1 测试自己的构造路径，保证口径与 (A) 那格完全一致。"""
    from fg1_scenarios import SCENARIOS

    return list(SCENARIOS)


def _pipeline_result(scenario: Any) -> PathQoTResult:
    """取 FG-1 测试的 ``evaluate_path`` 驱动路径（口径与 (A) 那格同源）。"""
    from test_fg1_physics_parity import _pipeline_result as run

    return run(scenario)


def _jsonable(value: Any) -> Any:
    """整数量保持 int，数组转 list（json 用 ``float.__repr`` 往返，无损）。"""
    if isinstance(value, (int, np.integer)):
        return int(value)
    return np.asarray(value, dtype=np.float64).tolist()


def _path_scenarios() -> list[tuple[str, Any, tuple[str, ...]]]:
    """两套既有金标准里**经** ``evaluate_path`` 的全部场景。

    FG-1 的四个场景走 ``tests/fg1_scenarios.py``，FG-4a 那个 ``kind == "path"``
    的场景走 ``tests/fg4a_scenarios.py``。两者的 ROADM 位置**互补**：

    * ``roadm_hoop`` 的 ROADM 在**末跳** ⇒ 端口语义在 SNR 比值里代数抵消；
    * ``path_cplusl_roadm`` 的 ROADM 在**非末跳**（3 跳中的第 1 跳）⇒ 可观测。

    只测其中一个就会把"端口语义值多少 dB"这个问题答错，所以两套都要在表里。
    """
    from fg4a_scenarios import PATH_KEYS, build_scenarios

    rows: list[tuple[str, Any, tuple[str, ...]]] = [
        (f"fg1/{s.name}", (lambda s=s: _pipeline_result(s)), OUTPUT_KEYS) for s in _scenarios()
    ]
    for scenario in build_scenarios("off"):
        if scenario.kind == "path":
            rows.append((f"fg4a/{scenario.name}", scenario.run, PATH_KEYS))
    return rows


def measure_raw(tier: str) -> dict[str, Any]:
    """某一档下全部路径场景的**原始**读数（用来冻结基线 / 算影响量）。"""
    MultiBandISRSGN.evaluate_path = TIERS[tier]
    try:
        out: dict[str, Any] = {}
        for key, thunk, keys in _path_scenarios():
            got = thunk()
            # FG-1 的驱动返回 ``PathQoTResult``（属性访问），FG-4a 的 ``scenario.run``
            # 返回 dict。归一成 dict 之后两条路径就能被同一段代码读。
            getter = got.__getitem__ if isinstance(got, dict) else lambda name, g=got: getattr(g, name)
            out[key] = {name: _jsonable(getter(name)) for name in keys}
        return out
    finally:
        MultiBandISRSGN.evaluate_path = COMMITTED_EVALUATE_PATH


def impact(baseline: dict[str, Any], tier: str) -> dict[str, Any]:
    """某一档相对**冻结的修复前基线**的移动量（门 2 的 ΔSNR 表数据源）。"""
    measured = measure_raw(tier)
    out: dict[str, Any] = {}
    for key, outputs in measured.items():
        base = baseline[key]
        out[key] = {name: compare(base[name], outputs[name]) for name in outputs}
    return out


def measure(tier: str, golden: dict[str, Any], *, inflate: float = 0.0) -> dict[str, Any]:
    """实测某一档下每个场景的 ``evaluate_path`` 与 main 金标准的分歧。

    ``inflate`` 给变异测试用：把候选读数整体抬高一个量，用来证明钉子不是真空断言。
    """
    result: dict[str, Any] = {}
    for scenario in _scenarios():
        MultiBandISRSGN.evaluate_path = TIERS[tier]
        try:
            got = _pipeline_result(scenario)
        finally:
            MultiBandISRSGN.evaluate_path = COMMITTED_EVALUATE_PATH
        result[scenario.name] = {
            "outputs": {
                key: compare(
                    golden["scenarios"][scenario.name][GOLDEN_KEYS[key]], getattr(got, key), inflate=inflate
                )
                for key in OUTPUT_KEYS
            }
        }
    return result


def build_record() -> dict[str, Any]:
    """组装 ``results/fg14_assembly_divergence.json`` 的内容。"""
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    record: dict[str, Any] = {
        "schema": "fg14-assembly-divergence-v1",
        "propositions": {
            "A_kernel": {
                "claim": "ODE/噪声内核与 origin/main 逐位一致。",
                "governed_by": "tests/test_fg1_physics_parity.py::test_lowlevel_matches_origin_main",
                "golden": "results/fg1_golden_main.json",
                "note": "FG-14 未触碰 models/numba_kernels.py 或 _calc_exact_power_profile_ode。",
            },
            "B_assembly": {
                "claim": "设备装配遵循仓内权威定义 ROADMConfig.path_loss_db()，其与 origin/main 的分歧被授权并记录。",
                "authoritative_definition": (
                    "ROADMConfig.path_loss_db() @ "
                    "src/network_nlo_eval/network/elements.py:113-116"
                ),
                "adjudicated_by": "FG-14 brief §0（编排方裁定）",
                "golden_provenance_rule": (
                    "results/fg1_golden_main.json 声明 source_branch=origin/main，"
                    "本刀不在其中写入任何 FG 树数值；分歧只记录在本文件。"
                ),
            },
        },
        "adopted_tier": "prospective_full_union",
        "adoption_rationale": (
            "同仓两处施加点（physics/qot_checker.py:248-261 波形级、rwa/qot_checker.py:118-128 解析级）"
            "均按端口 + booster；字段文档 network/elements.py:76 亦支持。"
            "prospective_ports_only 作为单变量对照保留，供复核该选择。"
        ),
        "tolerance": TOLERANCE,
        "committed": measure("committed", golden),
        "prospective_full_union": measure("prospective_full_union", golden),
        "prospective_ports_only": measure("prospective_ports_only", golden),
    }

    # (B) 的钉子：授权幅度 = 实测幅度。"不许悄悄扩大"于是变成一条精确比较。
    for entry in record["committed"].values():
        for cmp in entry["outputs"].values():
            cmp["authorized_max_abs_dev"] = cmp.get("max_abs_dev", 0.0)
            cmp["authorized_max_rel_dev"] = cmp.get("max_rel_dev", 0.0)

    # 门 2 的 ΔSNR 表：相对**冻结的修复前基线**。基线在阶段 0 采集并提交，
    # 于是阶段 1 的每一个数字都能回溯到"改动之前那个树"的具体读数。
    if BASELINE_PATH.exists():
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["scenarios"]
        record["impact_vs_pre_fix_baseline"] = {
            tier: impact(baseline, tier)
            for tier in ("committed", "prospective_ports_only", "prospective_full_union")
        }
    else:
        record["impact_vs_pre_fix_baseline"] = None
    return record


def violations(record: dict[str, Any], measured: dict[str, Any]) -> list[str]:
    """(B) 的钉子：实测分歧不得**超过**记录里授权的幅度。

    这条不是"等于逐位不变"——阶段 1 之后授权幅度非零，它检查的是**上界**。
    任何超出授权的移动（哪怕只有 1e-6）都必须先改授权、再改代码、再在文档归因。
    """
    problems: list[str] = []
    for name, entry in measured.items():
        authorized_entry = record["committed"].get(name)
        if authorized_entry is None:
            problems.append(f"{name}: 记录里没有这个场景的授权幅度")
            continue
        for key, cmp in entry["outputs"].items():
            auth = authorized_entry["outputs"].get(key)
            if auth is None:
                problems.append(f"{name}.{key}: 记录里没有这个输出的授权幅度")
                continue
            for metric in ("max_abs_dev", "max_rel_dev"):
                if auth.get("comparable") is False or cmp.get("comparable") is False:
                    if not auth.get("comparable") == cmp.get("comparable"):
                        problems.append(f"{name}.{key}.{metric}: 不可比状态与记录不符")
                    continue
                actual = float(cmp.get(metric) or 0.0)
                allowed = float(auth.get(f"authorized_{metric}") or 0.0)
                if actual > allowed + TOLERANCE:
                    problems.append(
                        f"{name}.{key}.{metric}: 实测 {actual!r} 超过授权幅度 {allowed!r}"
                        f"（超出 {actual - allowed:.6e}）——分歧悄悄扩大了"
                    )
    return problems


def main() -> int:
    """打印三档分歧表；按 ``--emit`` / ``--check`` / ``--mutation`` 固化或自证。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, default=None, help="写出记录 JSON 的路径")
    parser.add_argument("--check", type=Path, default=None, help="重算并与既有记录逐条比对")
    parser.add_argument(
        "--emit-baseline",
        type=Path,
        default=None,
        help="冻结当前（修复前）读数到 JSON —— 阶段 0 采集一次，阶段 1 据此算 ΔSNR",
    )
    parser.add_argument(
        "--mutation",
        type=float,
        default=None,
        metavar="EPS",
        help="把分歧人为放大 EPS（默认 1e-6）后跑钉子，证明它不是真空断言",
    )
    args = parser.parse_args()

    import network_nlo_eval

    print(f"# imported: {network_nlo_eval.__file__}")

    if args.emit_baseline is not None:
        payload = json.dumps(
            {
                "schema": "fg14-pre-fix-baseline-v1",
                "tier": "committed",
                "note": (
                    "FG-14 阶段 0（零物理变化）采集。阶段 1 的 ΔSNR 表以本文件为"
                    "'改前'一侧，所以每个数字都能回溯到改动之前那个树的具体读数。"
                ),
                "scenarios": measure_raw("committed"),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        args.emit_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.emit_baseline.write_text(payload + "\n", encoding="utf-8")
        print(f"wrote {args.emit_baseline}")
        return 0

    if args.mutation is not None:
        golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        record = json.loads(RECORD_PATH.read_text(encoding="utf-8")) if RECORD_PATH.exists() else build_record()
        inflated = measure("committed", golden, inflate=args.mutation)
        problems = violations(record, inflated)
        print(f"\n== 变异测试：把分歧人为放大 {args.mutation:g} ==")
        if problems:
            print(f"MUTATION_CAUGHT: 钉子变红，共 {len(problems)} 条：")
            for item in problems[:10]:
                print(f"  - {item}")
            return 0
        print("MUTATION_MISSED: 钉子仍然是绿的 —— (B) 是真空断言，纪律失效。")
        return 1

    record = build_record()
    print("\n== 分歧幅度（与 origin/main 金标准逐位比较）==")
    print(f"{'tier'.ljust(26)}{'scenario':<20}{'max|Δ| SNR| (dB)':>18}{'rel':>14}")
    print("-" * 80)
    for tier in ("committed", "prospective_ports_only", "prospective_full_union"):
        for name, entry in record[tier].items():
            cmp = entry["outputs"]["snr_db"]
            if cmp.get("bit_exact"):
                print(f"{tier:<26}{name:<20}{'0.0 (逐位)':>18}{'—':>14}")
            else:
                print(
                    f"{tier:<26}{name:<20}{cmp['max_abs_dev']:>18.6f}{cmp['max_rel_dev']:>14.3e}"
                )

    payload = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False)
    impact_table = record.get("impact_vs_pre_fix_baseline")
    if impact_table is not None:
        print("\n== 门 2 ΔSNR 表：相对**冻结的修复前基线** ==")
        print(f"{'tier'.ljust(26)}{'scenario':<30}{'ΔSNR max|Δ| (dB)':>19}{'rel':>13}")
        print("-" * 88)
        for tier in ("committed", "prospective_ports_only", "prospective_full_union"):
            for key, outputs in impact_table[tier].items():
                cmp = outputs.get("snr_db")
                if cmp is None:
                    continue
                if cmp.get("bit_exact"):
                    print(f"{tier:<26}{key:<30}{'0.0 (逐位)':>19}{'—':>13}")
                else:
                    print(
                        f"{tier:<26}{key:<30}{cmp['max_abs_dev']:>19.6f}"
                        f"{cmp['max_rel_dev']:>13.3e}"
                    )
    if args.emit is not None:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(payload + "\n", encoding="utf-8")
        print(f"\nwrote {args.emit}")
    if args.check is not None:
        if not args.check.exists():
            print(f"CHECK_FAIL: 缺少 {args.check}")
            return 1
        if json.loads(args.check.read_text(encoding="utf-8")) != record:
            print("CHECK_FAIL: 记录与既有 JSON 不一致")
            return 1
        problems = violations(record, record["committed"])
        if problems:
            print(f"CHECK_FAIL: 钉子不满: {problems[:5]}")
            return 1
        print(f"CHECK_OK: 与 {args.check} 逐条一致，且钉子未被触发")
    return 0


if __name__ == "__main__":
    sys.exit(main())
