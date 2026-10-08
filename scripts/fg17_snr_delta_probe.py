"""FG-17 §3：收 ``snr_delta_db`` 尾巴（FG-13/FG-14 未闭合项）。

``physics/comparison.py:80`` 是 ``models/isrs_gn.py::evaluate_path`` 的**唯一统计侧调用方**，
FG-14 端口化修好后它没被重跑。本脚本在收敛后的树上把它跑起来，报 ``snr_delta_db`` 的
改前 / 改后读数与逐场景对照。

## 先决问题：这条尾巴能不能当"逐位门"用？

不能。**实测**：把**完全相同**的代码经 ``TopologyQoTComparator`` 连跑 N 次，
``snr_delta_db`` 本身就不重现（本仓实测噪声底 ~0.2-0.6 dB，见 ``null_control``）。
而本刀的装配改动量级是 **精确 0**（同 evaluator 实例下逐位相同）。若直接比"改前/改后"，
任何非零差都落在噪声底里，**无法归因**。

因此本脚本分两层报：

1. **零层（能逐位的那层）**：``PyNLOQoTEvaluator.evaluate_path`` 直接跑、**不先跑统计侧**。
   实测该路径同 evaluator 实例**逐位可重现**（这是 FG-17 的波形级 A/B，见
   ``fg17_waveform_ab.py``）。本层用来给"装配改动 = 0"下逐位结论。
2. **尾巴层（本 brief §3 点名的那层）**：``TopologyQoTComparator`` 端到端跑，
   报 ``snr_delta_db`` 的改前 / 改后读数，并且**先测同代码的 null 噪声底**，
   再看改前/改后差是否落在噪声底内。落在底内 = 与噪声不可区分（不是"证明相等"）。

## 用法

    CUDA_PATH=<cuda-root> PYTHONPATH=<PH>/src ~/miniconda3/envs/nlo/bin/python \\
        scripts/fg17_snr_delta_probe.py -o results/fg17_snr_delta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

import network_nlo_eval
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.network.topology import NetworkTopology
from network_nlo_eval.physics.comparison import TopologyQoTComparator

REPO_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_FILE = REPO_ROOT / "assets" / "example_physical_topology.json"


def _topology() -> NetworkTopology:
    with open(TOPOLOGY_FILE, encoding="utf-8") as fh:
        obj = json.load(fh)
    return NetworkTopology(
        network_raw=nx.node_link_graph(obj, edges="edges"),
        default_fiber_config=FiberSpanConfig(length_km=100.0),
        default_edfa_config=EDFAConfig(),
        default_roadm_config=ROADMConfig(),
    )


def _orig_roadm(self, current, optical_path, roadm, hop_index, active_channels):
    """``25ed30d`` 的原始 ROADM 装配（收敛前），逐字复刻。"""
    allowed = [i for i in active_channels if not self._is_blocked(roadm, i)]
    current = self.physical["ROADM"](
        pass_channels=allowed,
        passband_hz=roadm.passband_hz or self._default_passband_hz,
        insertion_loss_db=roadm.path_loss_db(
            "drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"
        ),
        filter_order=roadm.filter_order,
    ).forward(current)
    if roadm.equalize_output_power:
        current = self.physical["EDFA"](
            gain_db=roadm.path_loss_db(
                "drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"
            ),
            nf_db=roadm.booster_noise_figure_db,
            seed=self.seed + hop_index + 1,
        ).forward(current)
    return current


def _make_patched(self, wdm_signal, optical_path):
    """收敛前的 ``propagate_path``：除 ROADM 那一段外与现在逐行相同。"""
    current = wdm_signal
    roadms = optical_path.roadm_configs or [None] * len(optical_path.fiber_configs)
    channel_info = current.ctx.meta["wdm_channels_info"]
    span_sequence = 0
    active_channels = np.flatnonzero(
        [e.get("original_ctx").meta.get("tx_power_per_pol_w", 0) > 0 for e in channel_info]
    ).tolist()
    for hop_index, (fiber, edfa, roadm) in enumerate(
        zip(optical_path.fiber_configs, optical_path.edfa_configs, roadms, strict=True)
    ):
        if fiber.use_attenuation_polynomial:
            raise ValueError("unsupported")
        beta2, beta3 = self._fiber_dispersion(fiber, current.ctx.center_freq)
        target_gain_db, nf_db = self._edfa_values(edfa)
        remaining_km = fiber.length_km
        while remaining_km > 1e-12:
            span_km = min(remaining_km, fiber.max_span_length_km)
            channel = self.physical["FiberChannel"](
                length_m=span_km * 1e3,
                alpha_db_per_km=fiber.attenuation_db_km_ref,
                beta2_ps2_per_km=beta2,
                beta3_ps3_per_km=beta3,
                coarse_step_size_m=min(2000.0, span_km * 1e3),
                n2_si=fiber.nonlinear_index_n2,
                a_eff_um2=fiber.effective_area_um2_ref,
                slope_a_eff_um2_per_nm=fiber.aeff_slope_um2_nm,
                ref_lambda_nm=fiber.reference_wavelength_nm,
                compute_ctx_name=self.compute_ctx_name,
            )
            current = channel.forward(current)
            if fiber.raman_pumps:
                pumps = [self.physical["PumpSpec"](**p.model_dump()) for p in fiber.raman_pumps]
                current = self.physical["RamanPump"](pumps, span_km, seed=self.seed + span_sequence).forward(
                    current
                )
            if target_gain_db is None:
                current = self.physical["IdealAmplifier"](nf_db=nf_db, default_power_dbm=0.0).forward(current)
            else:
                current = self.physical["EDFA"](
                    gain_db=target_gain_db, nf_db=nf_db, seed=self.seed + span_sequence
                ).forward(current)
            remaining_km -= span_km
            span_sequence += 1
        if roadm is None:
            continue
        current = _orig_roadm(self, current, optical_path, roadm, hop_index, active_channels)
    return current


def _pinned_gain_path(topo: NetworkTopology, path: list[int]) -> MultiSpanOpticalPath:
    """拓扑路径，但把 EDFA 的 ``target_gain_db`` 钉成显式增益。

    **为什么要钉**：默认 ``EDFAConfig()`` 的 ``target_gain_db=None`` 会走
    ``IdealAmplifier``。FG-17 实测（``scripts/fg17_snr_delta_probe.py`` 同款实验）：
    ``IdealAmplifier`` 这条路径**逐次不可重现**——同一 evaluator、同一输入、同一 seed，
    连跑 4 次有 3 次两两不同（max|ΔSNR| ~0.33 dB）；换成显式 ``target_gain_db``
    （真 ``EDFA``）后 4 次**逐位相同**。所以零层要能下"逐位"结论，必须避开
    ``IdealAmplifier``。第 1 层（真实尾巴）则保留默认配置，并显式记录这条噪声底。
    """
    import copy

    p = copy.deepcopy(topo.build_optical_path(path))
    p.edfa_configs = [
        EDFAConfig(target_gain_db=20.0, noise_figure_db=float(np.asarray(e.noise_figure_db).max()))
        if not isinstance(e.noise_figure_db, (list, tuple)) and np.asarray(e.noise_figure_db).ndim == 0
        else e
        for e in p.edfa_configs
    ]
    return p


def _arr(values: list[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def _spread(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "results" / "fg17_snr_delta.json")
    parser.add_argument("--num-channels", type=int, default=20)
    parser.add_argument("--num-symbols", type=int, default=512)
    parser.add_argument("--null-repeats", type=int, default=6, help="同代码 null 噪声底的重复次数")
    args = parser.parse_args()

    print(f"package under test: {network_nlo_eval.__file__}")
    assert str(REPO_ROOT / "src") in network_nlo_eval.__file__, "import 落到了 worktree 之外"

    import network_nlo_eval.physics.qot_checker as pq

    converged = pq.PyNLOQoTEvaluator.propagate_path
    topo = _topology()
    grid = SpectrumGrid(num_channels=args.num_channels, channel_spacing_hz=50e9, center_frequency_hz=193.1e12)
    path = [topo.get_internal_node_id(n) for n in sorted(topo._network_raw.nodes())]

    configs: list[tuple[str, np.ndarray]] = []
    for active in [(3, 4, 5), (0, 9), (2,)]:
        for dbm in (0.0, -3.0):
            prof = np.zeros(args.num_channels, dtype=np.float64)
            prof[list(active)] = 10 ** (dbm / 10) * 1e-3
            configs.append((f"active={list(active)},launch={dbm:+.1f}dBm", prof))

    # ---------- 第 0 层：零层（同 evaluator 实例、直接跑波形）----------
    #
    # **不要在这里构造 ``MultiBandISRSGN``。** 实测（FG-17）：一旦同进程内构造/运行过
    # 统计模型，波形侧就不再 run-to-run 可重现（连"完全相同的两次调用"都差 ~0.03-0.26 dB，
    # 比本刀改动量级大三个数量级）。故零层用固定符号率、不碰统计模型，
    # 这样前后两次调用才**逐位可重现**，零层读出的 0 才有意义。
    print("\n=== layer 0: waveform-only, same evaluator instance (the bit-exact layer) ===")
    symbol_rate_hz = 50e9
    zero_rows = []
    for label, prof in configs:
        physical_path = _pinned_gain_path(topo, path)
        from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator

        def _spin(use_orig: bool) -> np.ndarray:
            # **必须复用同一个 evaluator 实例。** 实测（FG-17）：新建实例会改变 GPU 内存
            # 布局，波形读数随之变化（同代码、不同实例，差 ~0.02-0.24 dB）；复用同一实例
            # 时逐位可重现。所以下面三个读数共用 `_shared_ev`，一个实例都不新建。
            _shared_ev.evaluate_path(physical_path, prof, num_symbols=args.num_symbols)  # 空转，建状态
            pq.PyNLOQoTEvaluator.propagate_path = _make_patched if use_orig else converged
            try:
                return _arr(_shared_ev.evaluate_path(physical_path, prof, num_symbols=args.num_symbols)[1]["snr"])
            finally:
                pq.PyNLOQoTEvaluator.propagate_path = converged

        _shared_ev = PyNLOQoTEvaluator(grid, symbol_rate_hz=symbol_rate_hz, sps=4, seed=42)
        after = _spin(False)
        before = _spin(True)
        null = _spin(False)  # 零层自身的 null：同代码两次
        exact = bool(np.array_equal(before, after, equal_nan=True))
        zero_rows.append(
            {
                "config": label,
                "bit_exact": exact,
                "max_abs_delta": _spread(before, after),
                "null_max_abs_delta_same_code": _spread(after, null),
                "waveform_snr_before": [None if not np.isfinite(v) else float(v) for v in before],
                "waveform_snr_after": [None if not np.isfinite(v) else float(v) for v in after],
            }
        )
        print(
            f"  {'OK ' if exact else 'DIFF'} {label:<40} max|d|={_spread(before, after):.3e} "
            f"(same-code null={_spread(after, null):.3e})"
        )
    print(
        f"  layer0 all bit-exact: {all(r['bit_exact'] for r in zero_rows)} "
        f"({len(zero_rows)}/{len(zero_rows)} configs)"
    )

    # ---------- 第 1 层：尾巴层（TopologyQoTComparator 端到端）----------
    print("\n=== layer 1: snr_delta_db via TopologyQoTComparator (comparison.py:80) ===")
    tail_rows = []
    for label, prof in configs:
        # (a) null 噪声底：完全相同的代码跑 N 次（取 max 与均值都要，单次样本不能当噪声底）
        nulls = []
        for _ in range(args.null_repeats):
            gn = MultiBandISRSGN(grid, topo.get_default_fiber_config())
            r = TopologyQoTComparator(topo, grid, gn_evaluator=gn).evaluate(
                path, prof, run_waveform=True, num_symbols=args.num_symbols
            )
            nulls.append(_arr(r.to_dict()["snr_delta_db"]))
        null_pairs = [_spread(nulls[0], n) for n in nulls[1:]]
        null_spread = max(null_pairs) if null_pairs else 0.0
        null_mean = float(np.mean(null_pairs)) if null_pairs else 0.0

        # (b) 改后
        gn = MultiBandISRSGN(grid, topo.get_default_fiber_config())
        after = _arr(
            TopologyQoTComparator(topo, grid, gn_evaluator=gn)
            .evaluate(path, prof, run_waveform=True, num_symbols=args.num_symbols)
            .to_dict()["snr_delta_db"]
        )
        # (c) 改前
        pq.PyNLOQoTEvaluator.propagate_path = _make_patched
        try:
            gn = MultiBandISRSGN(grid, topo.get_default_fiber_config())
            before = _arr(
                TopologyQoTComparator(topo, grid, gn_evaluator=gn)
                .evaluate(path, prof, run_waveform=True, num_symbols=args.num_symbols)
                .to_dict()["snr_delta_db"]
            )
        finally:
            pq.PyNLOQoTEvaluator.propagate_path = converged

        delta_spread = _spread(before, after)
        tail_rows.append(
            {
                "config": label,
                "null_control_max_spread_db": null_spread,
                "null_control_mean_spread_db": null_mean,
                "null_control_pairs_db": null_pairs,
                "before_after_max_delta_db": delta_spread,
                "delta_within_null_floor": bool(delta_spread <= null_spread),
                "bit_exact": bool(np.array_equal(before, after, equal_nan=True)),
                "snr_delta_db_before": [None if not np.isfinite(v) else float(v) for v in before],
                "snr_delta_db_after": [None if not np.isfinite(v) else float(v) for v in after],
                "n_channels_finite": int(np.isfinite(after).sum()),
            }
        )
        print(
            f"  {label:<40} null_floor={null_spread:.3e}  before/after|d|={delta_spread:.3e}  "
            f"within_floor={tail_rows[-1]['delta_within_null_floor']}"
        )

    payload = {
        "package": network_nlo_eval.__file__,
        "topology": str(TOPOLOGY_FILE),
        "grid": {
            "num_channels": grid.num_channels,
            "spacing_hz": grid.channel_spacing_hz,
            "center_hz": grid.center_frequency_hz,
        },
        "num_symbols": args.num_symbols,
        "null_repeats": args.null_repeats,
        "interpretation": {
            "layer0": "waveform-only, same evaluator instance: bit-exact => assembly change is exactly 0",
            "layer1": (
                "TopologyQoTComparator snr_delta_db is NOT run-to-run reproducible even with identical "
                "code; compare before/after spread against the null floor, never against 0"
            ),
            "statistical_side": (
                "MultiBandISRSGN.evaluate_path via comparison.py:80 is deterministic (verified bit-stable "
                "across repeats); the nondeterminism is on the GPU waveform side"
            ),
        },
        "summary": {
            "layer0_all_bit_exact": all(r["bit_exact"] for r in zero_rows),
            "layer0_max_abs_delta": max(r["max_abs_delta"] for r in zero_rows),
            "layer1_max_null_floor_db": max(r["null_control_max_spread_db"] for r in tail_rows),
            "layer1_mean_null_floor_db": float(np.mean([r["null_control_mean_spread_db"] for r in tail_rows])),
            "layer1_max_before_after_delta_db": max(r["before_after_max_delta_db"] for r in tail_rows),
            "layer1_all_within_null_floor": all(r["delta_within_null_floor"] for r in tail_rows),
        },
        "layer0_waveform_only": zero_rows,
        "layer1_snr_delta": tail_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== FG-17 §3 summary ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
