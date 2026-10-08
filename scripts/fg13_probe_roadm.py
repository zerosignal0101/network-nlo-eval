#!/usr/bin/env python
"""FG-13 机理探针：`evaluate_path` 上 ROADM 各字段的**存活情况**实测。

本脚本回答 FG-13 brief 的核心取证要求 §1.5：
只改 `express_attenuation_db`，走 `MultiBandISRSGN.evaluate_path`，SNR 是否逐位不变。

设计要点（每条都是为了让读数可解释，不为了让结论好看）：

1. **两个 ROADM 分处两跳**：hop0 的 ROADM 是**非末跳**（express 端口，物理上可观测），
   hop1 的是**末跳**（drop 端口）。只用一个 ROADM 会踩到一个结构性陷阱：
   `evaluate_path` 把损耗同时除信号与**全部已累积噪声**，末跳的损耗在
   `power/(spm+xpm+ase)` 的比值里**恒等抵消**——那会让"末跳 drop 损耗"看起来也是死的，
   但那是**代数恒等**、不是字段没被消费。本脚本把两者分开测，结论才站得住。
2. **逐字段扫描**，不止 express：`add_drop_loss_db` / `equalize_output_power` /
   `booster_noise_figure_db` 一并测，避免只证明一个字段而漏掉同一机理下的其它惰性字段。
3. **对照 RWA 路径**：同一批字段在 `rwa/qot_checker.fold_roadm_terms` 上测一遍，
   于是"这条路径死 / 那条路活"是同一把尺子量出来的，不是两套口径各说各话。
4. 报 **逐位相等**（bit-identical）而不只是报差值——brief 要的是"逐位不变"。

复算：
    PYTHONPATH=$PWD/src python scripts/fg13_probe_roadm.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.rwa.qot_checker import fold_roadm_terms

NUM_CHANNELS = 12
CHANNEL_SPACING_HZ = 50e9
CENTER_FREQUENCY_HZ = 193.1e12
SYMBOL_RATE_HZ = 32e9
LAUNCH_POWER_W = 1e-3
HOP_LENGTH_KM = 60.0
HOP_GAIN_DB = 15.0
HOP_NF_DB = 5.0

#: 各字段的"基线值 → 扰动值"。基线取 ``ROADMConfig`` 的默认值，
#: 于是 ``express_attenuation_db`` 0.0 / ``add_drop_loss_db`` 1.0 / ``equalize`` True。
FIELD_SWEEPS: tuple[tuple[str, Any, Any], ...] = (
    ("insertion_loss_db", 5.0, 8.0),
    ("filtering_penalty_db", 0.2, 2.0),
    ("express_attenuation_db", 0.0, 2.0),
    ("add_drop_loss_db", 1.0, 3.0),
    ("booster_noise_figure_db", 5.0, 8.0),
    ("equalize_output_power", True, False),
)


def _base_roadm() -> ROADMConfig:
    """两跳共用的 ROADM 基线配置（全部走 ``ROADMConfig`` 默认值）。"""
    return ROADMConfig(insertion_loss_db=5.0, filtering_penalty_db=0.2)


def build_path(roadm_per_hop: tuple[ROADMConfig | None, ROADMConfig | None]) -> MultiSpanOpticalPath:
    """构造两跳路径，两跳各挂一个 ROADM。

    hop0 的 ROADM 处于**非末跳**位置 ⇒ 按端口语义它是 express；
    hop1 的 ROADM 处于**末跳** ⇒ 按端口语义它是 drop。
    """
    return MultiSpanOpticalPath(
        fiber_configs=[
            FiberSpanConfig(length_km=HOP_LENGTH_KM, max_span_length_km=HOP_LENGTH_KM)
            for _ in range(2)
        ],
        edfa_configs=[
            EDFAConfig(target_gain_db=HOP_GAIN_DB, noise_figure_db=HOP_NF_DB) for _ in range(2)
        ],
        roadm_configs=list(roadm_per_hop),
    )


def evaluate_two_hop(roadm_per_hop: tuple[ROADMConfig | None, ROADMConfig | None]) -> np.ndarray:
    """走 ``evaluate_path`` 返回逐信道 SNR（dB）。

    固定 ``ode_accel="off"``：本探针比较的是**装配规则**，不是加速档位；
    固定到 ``off`` 才能让"逐位不变"这个断言有意义。
    """
    grid = None
    evaluator = MultiBandISRSGN(
        _grid(),
        FiberSpanConfig(length_km=HOP_LENGTH_KM),
        symbol_rate_hz=SYMBOL_RATE_HZ,
        ode_accel="off",
    )
    del grid
    result = evaluator.evaluate_path(build_path(roadm_per_hop), np.full(NUM_CHANNELS, LAUNCH_POWER_W))
    return np.asarray(result.snr_db, dtype=np.float64)


_GRID = None


def _grid():
    """惰性构造 SpectrumGrid（避免模块级副作用）。"""
    global _GRID
    if _GRID is None:
        from network_nlo_eval.core.spectrum import SpectrumGrid

        _GRID = SpectrumGrid(NUM_CHANNELS, CHANNEL_SPACING_HZ, CENTER_FREQUENCY_HZ)
    return _GRID


def fold_loss_on(roadm: ROADMConfig, *, is_last_hop: bool) -> float:
    """RWA 路径（并集实现）折叠出的**信号侧**损耗因子。"""
    return float(fold_roadm_terms(roadm, is_last_hop=is_last_hop).loss)


def evaluate_last_hop_only(roadm: ROADMConfig) -> np.ndarray:
    """单 ROADM 且挂在**末跳**——用来演示末跳损耗在 SNR 比值里的代数抵消。"""
    grid_evaluator = MultiBandISRSGN(
        _grid(),
        FiberSpanConfig(length_km=HOP_LENGTH_KM),
        symbol_rate_hz=SYMBOL_RATE_HZ,
        ode_accel="off",
    )
    path = build_path((None, roadm))
    return np.asarray(grid_evaluator.evaluate_path(path, np.full(NUM_CHANNELS, LAUNCH_POWER_W)).snr_db)


def evaluate_express_only(roadm: ROADMConfig) -> np.ndarray:
    """单 ROADM 且挂在**非末跳**（express 端口）——express 损耗应当可观测。"""
    grid_evaluator = MultiBandISRSGN(
        _grid(),
        FiberSpanConfig(length_km=HOP_LENGTH_KM),
        symbol_rate_hz=SYMBOL_RATE_HZ,
        ode_accel="off",
    )
    path = build_path((roadm, None))
    return np.asarray(grid_evaluator.evaluate_path(path, np.full(NUM_CHANNELS, LAUNCH_POWER_W)).snr_db)


def probe() -> dict[str, Any]:
    """跑完整套机理读数，返回可 JSON 化的报告。"""
    base_pair = (_base_roadm(), _base_roadm())
    baseline = evaluate_two_hop(base_pair)

    field_rows: list[dict[str, Any]] = []
    for field, base_value, bumped in FIELD_SWEEPS:
        pair = tuple(roadm.model_copy(update={field: bumped}) for roadm in base_pair)  # type: ignore[assignment]
        snr = evaluate_two_hop(pair)  # type: ignore[arg-type]
        bitwise = bool(np.array_equal(baseline, snr))
        delta = snr - baseline
        finite = delta[np.isfinite(delta)]
        field_rows.append(
            {
                "field": field,
                "baseline_value": base_value,
                "perturbed_value": bumped,
                "evaluate_path_snr_bitwise_identical": bitwise,
                "evaluate_path_max_abs_delta_db": float(np.max(np.abs(finite))) if finite.size else None,
                "rwa_fold_express_loss_delta": None,
                "rwa_fold_drop_loss_delta": None,
            }
        )
        # RWA 对照：同一字段在并集折叠出的两个端口损耗上的效果。
        base = _base_roadm()
        bumped_cfg = base.model_copy(update={field: bumped})
        row = field_rows[-1]
        row["rwa_fold_express_loss_delta"] = fold_loss_on(bumped_cfg, is_last_hop=False) - fold_loss_on(
            base, is_last_hop=False
        )
        row["rwa_fold_drop_loss_delta"] = fold_loss_on(bumped_cfg, is_last_hop=True) - fold_loss_on(
            base, is_last_hop=True
        )

    # 结构性演示：末跳 vs 非末跳，同一个**活字段**在两处的可观测性不同。
    #
    # 这里必须用 ``insertion_loss_db``（``evaluate_path`` 确实读的字段），**不能**用
    # ``express_attenuation_db``：后者在这条路径上根本没被读，无论挂在哪一跳都是
    # 逐位不变，用它做对照会把"代数抵消"和"字段是死的"两种解释混为一谈——
    # 那正是本刀最容易踩的假阳性。
    #
    # ``evaluate_path`` 把损耗同时除信号与**全部已累积噪声**。末跳的 ROADM 之后不再有
    # 新噪声产生，于是 L 在 ``power/(spm+xpm+ase)`` 里约掉；非末跳的 ROADM 之后还要
    # 累加新噪声，L 只作用于此前的噪声项，故**应当**可观测。
    base = _base_roadm()
    bumped = base.model_copy(update={"insertion_loss_db": 8.0})
    last_hop_snr = evaluate_last_hop_only(base)
    last_hop_snr_bumped = evaluate_last_hop_only(bumped)
    express_snr = evaluate_express_only(base)
    express_snr_bumped = evaluate_express_only(bumped)

    def _delta(a: np.ndarray, b: np.ndarray) -> float | None:
        finite = (b - a)[np.isfinite(b - a)]
        return float(np.max(np.abs(finite))) if finite.size else None

    structural = {
        "probe_field": "insertion_loss_db (5.0 -> 8.0 dB，一个 evaluate_path 确实读取的字段)",
        "on_LAST_hop_only": {
            "bitwise_identical": bool(np.array_equal(last_hop_snr, last_hop_snr_bumped)),
            "max_abs_delta_db": _delta(last_hop_snr, last_hop_snr_bumped),
            "note": "末跳损耗同时除信号与全部累积噪声，在 SNR 比值里代数抵消 ⇒ 预期逐位不变",
        },
        "on_NON_LAST_hop_only": {
            "bitwise_identical": bool(np.array_equal(express_snr, express_snr_bumped)),
            "max_abs_delta_db": _delta(express_snr, express_snr_bumped),
            "note": "非末跳损耗只作用于此前累积噪声，其后还要累加新噪声 ⇒ 预期可观测",
        },
    }

    dead = [r["field"] for r in field_rows if r["evaluate_path_snr_bitwise_identical"]]
    alive = [r["field"] for r in field_rows if not r["evaluate_path_snr_bitwise_identical"]]

    return {
        "scenario": {
            "num_channels": NUM_CHANNELS,
            "channel_spacing_hz": CHANNEL_SPACING_HZ,
            "center_frequency_hz": CENTER_FREQUENCY_HZ,
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "launch_power_w": LAUNCH_POWER_W,
            "hop_length_km": HOP_LENGTH_KM,
            "hops": 2,
            "roadm_layout": "hop0=非末跳(express 语义), hop1=末跳(drop 语义)",
            "ode_accel": "off",
        },
        "baseline_snr_db": baseline.tolist(),
        "field_sweep": field_rows,
        "structural_demo": structural,
        "summary": {
            "inert_on_evaluate_path": dead,
            "live_on_evaluate_path": alive,
            "rwa_live_fields": [
                r["field"]
                for r in field_rows
                if abs(r["rwa_fold_express_loss_delta"] or 0.0) > 0
                or abs(r["rwa_fold_drop_loss_delta"] or 0.0) > 0
            ],
        },
    }


def main() -> int:
    """采集并打印字段存活普查；``--emit`` 固化 JSON，``--check`` 与既有报告逐条比对。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit", type=Path, default=None, help="写出 JSON 报告的路径")
    parser.add_argument("--check", type=Path, default=None, help="与既有报告逐位比对")
    args = parser.parse_args()

    import network_nlo_eval

    print(f"# imported: {network_nlo_eval.__file__}")
    report = probe()

    rows = report["field_sweep"]
    width = max(len(r["field"]) for r in rows)
    print(f"\n{'field'.ljust(width)}  {'evaluate_path':<26} {'max|ΔSNR| dB':>14}   RWA express / drop 损耗变化")
    print("-" * 100)
    for r in rows:
        verdict = "逐位不变（死字段）" if r["evaluate_path_snr_bitwise_identical"] else "有响应（活字段）"
        delta = r["evaluate_path_max_abs_delta_db"]
        delta_s = "n/a" if delta is None else f"{delta:.6f}"
        print(
            f"{r['field'].ljust(width)}  {verdict:<26} {delta_s:>14}   "
            f"{r['rwa_fold_express_loss_delta']:+.9f} / {r['rwa_fold_drop_loss_delta']:+.9f}"
        )

    print(f"\n# 结构性演示（{report['structural_demo']['probe_field']}）")
    for key in ("on_LAST_hop_only", "on_NON_LAST_hop_only"):
        val = report["structural_demo"][key]
        delta = val["max_abs_delta_db"]
        delta_s = "n/a" if delta is None else f"{delta:.6f}"
        print(
            f"  {key}: 逐位不变={val['bitwise_identical']}  max|ΔSNR|={delta_s} dB"
            f"  ({val['note']})"
        )

    print("\n# 汇总")
    print(f"  evaluate_path 上的死字段: {report['summary']['inert_on_evaluate_path']}")
    print(f"  evaluate_path 上的活字段: {report['summary']['live_on_evaluate_path']}")
    print(f"  RWA 并集折叠会响应的字段: {report['summary']['rwa_live_fields']}")

    payload = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    if args.emit is not None:
        args.emit.parent.mkdir(parents=True, exist_ok=True)
        args.emit.write_text(payload + "\n", encoding="utf-8")
        print(f"\nwrote {args.emit}")
    if args.check is not None:
        if not args.check.exists():
            print(f"CHECK_FAIL: 缺少 {args.check}")
            return 1
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if expected != report:
            print("CHECK_FAIL: 报告与既有 JSON 不一致")
            return 1
        print(f"CHECK_OK: 与 {args.check} 逐条一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
