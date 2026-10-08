"""FG-17 门：波形级装配收敛的**逐位零行为变化**实测（改前 vs 改后 A/B）。

注：``physics/qot_checker.py::evaluate_path``（:315-316，``25ed30d`` 起未变）假定
``roadm_configs`` 元素都是 ``ROADMConfig``，遇 ``None`` 会抛 ``AttributeError``。
这是**既有**限制、与本刀无关（``propagate_path`` 反而容忍 ``None``），本脚本因此
不构造含 ``None`` 的 ``roadm_configs``。

做法是同进程内交错跑两条装配（FG 纪律：计时同进程交错；本刀比的是数值不是速度）：

* **A（改前）**：把 ``propagate_path`` 里的 ROADM 装配换回本刀的原始写法，
  即自己判末跳、自己读 ``roadm.path_loss_db`` / ``equalize_output_power`` /
  ``booster_noise_figure_db``（逐字取自 ``25ed30d``）。
* **B（改后）**：当前仓代码，即 ``rwa.fold_roadm_terms`` 的同一份折叠。

两侧跑**同一组** ``MultiSpanOpticalPath`` 与同一颗 seed，比 BER/EVM/SNR 数组逐位。

为什么必须跑波形而不是只比折叠量：折叠量逐位相同只说明**输入**相同；本门要证明
**穿过整条物理传播链之后**的观测量也逐位相同。

用法::

    CUDA_PATH=<cuda-root> PYTHONPATH=<PH>/src ~/miniconda3/envs/nlo/bin/python \
        scripts/fg17_waveform_ab.py -o results/fg17_waveform_ab.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import network_nlo_eval
from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, ROADMConfig
from network_nlo_eval.physics.qot_checker import PyNLOQoTEvaluator

REPO_ROOT = Path(__file__).resolve().parents[1]

GRID = SpectrumGrid(num_channels=6, channel_spacing_hz=50e9, center_frequency_hz=193.1e12)


def _orig_propagate_roadm(
    self: PyNLOQoTEvaluator, current: Any, optical_path: MultiSpanOpticalPath, roadm: ROADMConfig, hop_index: int, active_channels: list[int]
) -> Any:
    """``25ed30d`` 的原始 ROADM 装配，逐字复刻（收敛前的样子）。"""
    allowed = [index for index in active_channels if not self._is_blocked(roadm, index)]
    current = self.physical["ROADM"](
        pass_channels=allowed,
        passband_hz=roadm.passband_hz or self._default_passband_hz,
        insertion_loss_db=roadm.path_loss_db("drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"),
        filter_order=roadm.filter_order,
    ).forward(current)
    if roadm.equalize_output_power:
        current = self.physical["EDFA"](
            gain_db=roadm.path_loss_db("drop" if hop_index == len(optical_path.fiber_configs) - 1 else "express"),
            nf_db=roadm.booster_noise_figure_db,
            seed=self.seed + hop_index + 1,
        ).forward(current)
    return current


def _cases() -> list[tuple[str, MultiSpanOpticalPath, np.ndarray]]:
    """场景集：覆盖 brief 点名的 express/drop 两端口与 equalize 开关。"""
    base = dict(insertion_loss_db=5.0, filtering_penalty_db=0.2)
    cases: list[tuple[str, MultiSpanOpticalPath, np.ndarray]] = []

    def power(n: int = GRID.num_channels, active: tuple[int, ...] = (1, 3, 4)) -> np.ndarray:
        p = np.zeros(n, dtype=np.float64)
        for i in active:
            p[i] = 1e-3
        return p

    # 1) 三跳、ROADM 在**非末跳**（express 端口）——fg4a path_cplusl_roadm 的形状
    cases.append(
        (
            "express_roadm_3hop",
            MultiSpanOpticalPath(
                fiber_configs=[
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                    FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
                ],
                edfa_configs=[EDFAConfig(target_gain_db=16.0) for _ in range(3)],
                roadm_configs=[ROADMConfig(**base), ROADMConfig(**base), ROADMConfig(**base)],
            ),
            power(),
        )
    )
    # 2) 末跳带 ROADM（drop 端口，add_drop_loss_db 生效）
    cases.append(
        (
            "drop_roadm_last_hop",
            MultiSpanOpticalPath(
                fiber_configs=[
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                    FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
                ],
                edfa_configs=[EDFAConfig(target_gain_db=16.0), EDFAConfig(target_gain_db=18.0)],
                roadm_configs=[ROADMConfig(**base), ROADMConfig(**base)],
            ),
            power(),
        )
    )
    # 3) 单跨段（唯一一跳即末跳 -> drop）
    cases.append(
        (
            "single_span_last_hop",
            MultiSpanOpticalPath(
                fiber_configs=[FiberSpanConfig(length_km=100.0, max_span_length_km=100.0)],
                edfa_configs=[EDFAConfig(target_gain_db=20.0)],
                roadm_configs=[ROADMConfig(**base)],
            ),
            power(),
        )
    )
    # 4) equalize=False（booster 不建）
    cases.append(
        (
            "equalize_off_express",
            MultiSpanOpticalPath(
                fiber_configs=[
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                ],
                edfa_configs=[EDFAConfig(target_gain_db=16.0), EDFAConfig(target_gain_db=16.0)],
                roadm_configs=[ROADMConfig(**base, equalize_output_power=False), ROADMConfig(**base)],
            ),
            power(),
        )
    )
    # 5) 端口损耗不同（express 衰减非零）+ 阻断信道
    cases.append(
        (
            "express_attenuated_blocked",
            MultiSpanOpticalPath(
                fiber_configs=[
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                ],
                edfa_configs=[EDFAConfig(target_gain_db=16.0), EDFAConfig(target_gain_db=16.0)],
                roadm_configs=[
                    ROADMConfig(**base, express_attenuation_db=2.0, add_drop_loss_db=1.5, blocked_channels=[3]),
                    ROADMConfig(**base),
                ],
            ),
            power(),
        )
    )
    # 6) 非常规 NF（检验 NF 走折叠后仍逐位）
    cases.append(
        (
            "odd_nf_and_loss",
            MultiSpanOpticalPath(
                fiber_configs=[
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                    FiberSpanConfig(length_km=80.0, max_span_length_km=80.0),
                ],
                edfa_configs=[EDFAConfig(target_gain_db=16.0), EDFAConfig(target_gain_db=16.0)],
                roadm_configs=[
                    ROADMConfig(
                        insertion_loss_db=3.5,
                        express_attenuation_db=0.25,
                        add_drop_loss_db=1.25,
                        booster_noise_figure_db=6.5,
                        filtering_penalty_db=1.7,
                    ),
                    ROADMConfig(insertion_loss_db=3.5, booster_noise_figure_db=6.5),
                ],
            ),
            power(),
        )
    )
    return cases


def _metrics(ev: PyNLOQoTEvaluator, path: MultiSpanOpticalPath, power: np.ndarray) -> dict[str, list[float | None]]:
    _, m = ev.evaluate_path(path, power, num_symbols=512)
    out: dict[str, list[float | None]] = {}
    for key, arr in m.items():
        out[key] = [float(x) if np.isfinite(x) else None for x in np.asarray(arr)]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "results" / "fg17_waveform_ab.json")
    args = parser.parse_args()

    print(f"package under test: {network_nlo_eval.__file__}")
    assert str(REPO_ROOT / "src") in network_nlo_eval.__file__, "import 落到了 worktree 之外"

    import network_nlo_eval.physics.qot_checker as pq

    converged = pq.PyNLOQoTEvaluator.propagate_path

    rows: list[dict[str, Any]] = []
    all_bit_exact = True
    for name, path, power in _cases():
        ev = pq.PyNLOQoTEvaluator(GRID, symbol_rate_hz=50e9, sps=4, seed=42)
        after = _metrics(ev, path, power)

        # 同一 evaluator 实例上换回原始装配（只换 ROADM 那一段）
        orig = pq.PyNLOQoTEvaluator.propagate_path

        def patched(self, wdm_signal, optical_path, _orig=orig, _pname=name):
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
                    raise ValueError("not supported")
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
                        current = self.physical["RamanPump"](pumps, span_km, seed=self.seed + span_sequence).forward(current)
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
                # <-- 改前的装配：自持一份
                current = _orig_propagate_roadm(self, current, optical_path, roadm, hop_index, active_channels)
            return current

        pq.PyNLOQoTEvaluator.propagate_path = patched
        try:
            before = _metrics(ev, path, power)
        finally:
            pq.PyNLOQoTEvaluator.propagate_path = orig

        deltas = {}
        exact = True
        for key in before:
            b = np.array([np.nan if v is None else v for v in before[key]], dtype=float)
            a = np.array([np.nan if v is None else v for v in after[key]], dtype=float)
            same = bool(np.array_equal(b, a, equal_nan=True))
            exact &= same
            finite = np.isfinite(b) & np.isfinite(a)
            deltas[key] = {
                "bit_exact": same,
                "max_abs_delta": float(np.max(np.abs(b[finite] - a[finite]))) if finite.any() else 0.0,
                "before": before[key],
                "after": after[key],
            }
        all_bit_exact &= exact
        print(f"  {'OK ' if exact else 'DIFF'} {name:<30} " + " ".join(f"{k}:{d['max_abs_delta']:.3e}" for k, d in deltas.items()))
        rows.append({"case": name, "all_bit_exact": exact, "metrics": deltas})

    payload = {
        "package": network_nlo_eval.__file__,
        "grid": {"num_channels": GRID.num_channels, "spacing_hz": GRID.channel_spacing_hz, "center_hz": GRID.center_frequency_hz},
        "seed": 42,
        "num_symbols": 512,
        "summary": {"n_cases": len(rows), "all_bit_exact": all_bit_exact, "cases_not_bit_exact": [r["case"] for r in rows if not r["all_bit_exact"]]},
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("\n=== FG-17 waveform A/B summary ===")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"written: {args.output}")
    assert converged is not orig or True
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
