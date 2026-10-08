#!/usr/bin/env python
"""FG-13 影响量化：**若**把 `evaluate_path` 的 ROADM 装配改成权威口径，会动到什么。

本脚本**不修改仓库任何文件**。它用 monkeypatch 在进程内把
``MultiBandISRSGN.evaluate_path`` 换成一个"按 ``rwa.qot_checker.fold_roadm_terms``
口径"的版本，跑同一批既有场景，对比 before/after。

这样做的理由：FG-13 的裁决问题不是"能不能改"，而是"改了会撞到哪些**既有期望值**"。
先把碰撞面量出来，才谈得上逐条归因；而且**量出来的数字不依赖我是否真的提交那次改动**。

被替换的版本逐字对齐 ``rwa.qot_checker.accumulate_service_snr`` 的 ROADM 语义：

1. ``path_loss_db("drop" if is_last_hop else "express")`` —— 按端口取损耗；
2. ``equalize_output_power`` 为真时信号**不**被损耗衰减，改为加 booster ASE
   ``NF·h·ν·Rs·(loss−1)``；为假时才除 ``loss``；
3. ``filtering_penalty_db`` 作为**只乘噪声**的独立乘子。

刻意**不**引入的：`accumulate_service_snr` 末尾那个 ``total_spans ** 0.05`` 惩罚项。
它属于服务 SNR 公式、不是 ROADM 装配，混进来会把两个变量搅在一起。

复算：
    PYTHONPATH=$PWD/src python scripts/fg13_impact_probe.py
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

ORIGINAL_EVALUATE_PATH = MultiBandISRSGN.evaluate_path


def evaluate_path_authoritative(
    self: MultiBandISRSGN,
    optical_path: MultiSpanOpticalPath,
    power_in_w: np.ndarray,
) -> PathQoTResult:
    """``evaluate_path`` 的"按权威口径"版本（仅用于量化影响，不提交）。

    除 ROADM 那一段以外，逐行照抄 ``models/isrs_gn.py`` 的 ``evaluate_path``，
    以便 before/after 的差异**只可能**来自 ROADM 装配。
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
        # ---- 唯一的差异段：权威 ROADM 装配 ----
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
        # ---- 差异段结束 ----
        blocked = self._resolve_blocked_node_indices(roadm)
        if blocked:
            power[blocked] = total_spm[blocked] = total_xpm[blocked] = total_ase[blocked] = 0

    total_noise = total_spm + total_xpm + total_ase
    snr = np.full_like(power, -np.inf)
    active = (power > 0) & (total_noise > 0)
    snr[active] = 10 * np.log10(power[active] / total_noise[active])
    snr[(power > 0) & (total_noise == 0)] = np.inf
    return PathQoTResult(
        power, total_spm, total_xpm, total_ase, snr, span_count, sum(roadm is not None for roadm in roadm_configs)
    )


def _diff(before: Any, after: Any) -> dict[str, Any]:
    """逐位比较两个输出，给出 bitwise / max|Δ| / 相对量级。"""
    if isinstance(before, (int, np.integer)):
        return {
            "bitwise_identical": bool(before == after),
            "before": int(before),
            "after": int(after),
            "max_abs_delta": None,
        }
    b = np.asarray(before, dtype=np.float64)
    a = np.asarray(after, dtype=np.float64)
    finite = np.isfinite(b) & np.isfinite(a)
    delta = np.abs(a[finite] - b[finite])
    scale = np.maximum(np.abs(b[finite]), 1e-300)
    return {
        "bitwise_identical": bool(np.array_equal(b, a)),
        "max_abs_delta": float(delta.max()) if delta.size else None,
        "max_rel_delta": float((delta / scale).max()) if delta.size else None,
    }


def run_fg1_scenarios() -> list[dict[str, Any]]:
    """FG-1 的 4 个场景（走 ``evaluate_path``，含 ``roadm_hoop``）。"""
    from fg1_scenarios import SCENARIOS
    from test_fg1_physics_parity import _pipeline_result

    rows = []
    for scenario in SCENARIOS:
        MultiBandISRSGN.evaluate_path = ORIGINAL_EVALUATE_PATH
        before = _pipeline_result(scenario)
        MultiBandISRSGN.evaluate_path = evaluate_path_authoritative
        after = _pipeline_result(scenario)
        MultiBandISRSGN.evaluate_path = ORIGINAL_EVALUATE_PATH

        rows.append(
            {
                "scenario_set": "tests/fg1_scenarios.py",
                "scenario": scenario.name,
                "outputs": {
                    key: _diff(getattr(before, attr), getattr(after, attr))
                    for key, attr in (
                        ("signal_power_w", "signal_power_w"),
                        ("spm_noise_w", "spm_noise_w"),
                        ("xpm_noise_w", "xpm_noise_w"),
                        ("ase_noise_w", "ase_noise_w"),
                        ("snr_db", "snr_db"),
                        ("span_count", "span_count"),
                    )
                },
            }
        )
    return rows


def run_fg4a_scenarios() -> list[dict[str, Any]]:
    """FG-4a 的场景集（只有 ``path_cplusl_roadm`` 经过 ``evaluate_path``）。"""
    from fg4a_scenarios import PATH_KEYS, build_scenarios

    rows = []
    for scenario in build_scenarios("off"):
        if scenario.kind != "path":
            continue
        MultiBandISRSGN.evaluate_path = ORIGINAL_EVALUATE_PATH
        before = scenario.run()
        MultiBandISRSGN.evaluate_path = evaluate_path_authoritative
        after = scenario.run()
        MultiBandISRSGN.evaluate_path = ORIGINAL_EVALUATE_PATH

        rows.append(
            {
                "scenario_set": "tests/fg4a_scenarios.py",
                "scenario": scenario.name,
                "outputs": {key: _diff(before[key], after[key]) for key in PATH_KEYS},
            }
        )
    return rows


def main() -> int:
    """采集并打印 before/after 对比表；``--emit`` 时把读数固化成 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, default=None)
    args = parser.parse_args()

    import network_nlo_eval

    print(f"# imported: {network_nlo_eval.__file__}")

    report: dict[str, Any] = {}
    for name, fn in (("fg1", run_fg1_scenarios), ("fg4a", run_fg4a_scenarios)):
        report[name] = fn()

    moved: list[str] = []
    print()
    for setname, rows in report.items():
        for row in rows:
            for key, stat in row["outputs"].items():
                if not stat["bitwise_identical"]:
                    moved.append(f"{setname}/{row['scenario']}.{key}")

    print("== 若按权威口径修复，下列既有输出会移动 ==")
    for item in moved:
        print(f"  {item}")
    if not moved:
        print("  (none)")

    print("\n== 逐场景 ΔSNR (dB) ==")
    for setname, rows in report.items():
        for row in rows:
            snr = row["outputs"].get("snr_db")
            if snr is None:
                continue
            if snr["bitwise_identical"]:
                print(f"  {setname}/{row['scenario']:<24} 0.000000 (逐位不变)")
            else:
                print(f"  {setname}/{row['scenario']:<24} {snr['max_abs_delta']:.6f}  (rel {snr['max_rel_delta']:.3e})")

    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.emit is not None:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(payload + "\n", encoding="utf-8")
        print(f"\nwrote {args.emit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
