#!/usr/bin/env python3
"""WP-Q1 判据 Q-J2 / Q-J6：带宽真的进了公式吗？

判据（跑前写死，不许放宽）
-------------------------

**Q-J2** 受控探针：固定其余条件，只扫 ``B_i ∈ {12.5, 25, 50, 100} GHz``，
报告 SNR 变化；并**逐项对照 P1 (29)/(30)/(34) 的解析标度**。

"解析标度"由本文件里的 :func:`reference_terms` 给出——它是照 P1 §III 原文
**独立重写**的一份标量参考实现（不是把内核函数复制一份）：

* ``(29)`` ``sigma^2_SPM = (4/9)(P_i^3/B_i^2) N^(1+eps) (g_ii^2 pi/(phi_i 3 a_i^2))
  e^(2 Cr k L_eff r) [ (T_i-a_i^2)/a_i asinh(phi_i B_i^2/(pi a_i))
  + (4a_i^2-T_i)/(2a_i) asinh(phi_i B_i^2/(2 pi a_i)) ]``
* ``(30)`` ``sigma^2_XPM = (32/27) P_i N sum_l (P_l^2/B_l)(g_il^2/(phi_il 3 a_l^2))
  e^(...) [ (T_l-a_l^2)/a_l atan(phi_il B_i/a_l)
  + (4a_l^2-T_l)/(2a_l) atan(phi_il B_i/(2 a_l)) ]``
* ``(34)`` ``sigma^2_ASE = h f_i F_i B_i (G-1) N``

因此本探针真正检验的是**「内核里的 B 是否出现在论文写的那几个位置」**：
若 B 只进了 ASE，SPM/XPM 项就会与 (29)/(30) 预测不符；反之若 B 进了
``B^{-2}`` 或 ``atan(·B_i)``，也必须与预测一致。三项同时对上才算通过。

**Q-J6** 归一化口径显式：``fixed_power`` 与 ``fixed_psd`` **两个口径各跑一遍**，
读数都落进本文件的 JSON。

容差
----
逐项方差与 SNR 的相对偏差容差 ``REL_TOL = 1e-9``。理由：参考实现与 njit 内核
的浮点运算顺序不同（参考是 Python 标量、内核是编译后的循环），IEEE 双精度下
末位差异可达 ~1e-16，但不会累积到 1e-9 以上；1e-9 足以把"B 进错位置"这种
**结构性**错误（通常差几个数量级）与纯舍入噪声区分开。

用法::

    PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_probe.py --write
    PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_probe.py --check

产物：``results/bwqot_probe.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "bwqot_probe.json"

#: 判据阈值（跑前写死）
REL_TOL = 1e-9

#: Q-J2 写死的扫描点（GHz）
SWEEP_GHZ = (12.5, 25.0, 50.0, 100.0)

SLOT_WIDTH_HZ = 12.5e9
SYMBOL_RATE_HZ = 50e9
LAUNCH_POWER_W = 1e-3
SPAN_KM = 80.0
CENTER_HZ = 193.1e12
N_SPAN = 1
EPSILON_SPAN = 0.05  # 与 _calc_spm_variance_jit 的默认 epsilon_n_span 一致
H_PLANCK = 6.62607015e-34

#: 被扫动的连接占 4 号槽位区间中心；其余连接宽度固定，充当"干扰源"。
SWEEPER_ID = 0


def _build_scenario(n_slots: int, *, n_interferers: int = 3, power_width_mode: str):
    """构造一个"一个被扫连接 + N 个固定宽度干扰连接"的逐连接场景。"""
    import numpy as np

    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    grid = FlexibleGrid.from_legacy_grid(n_interferers + 2, 50e9, CENTER_HZ, slots_per_channel=4)

    # slots_for_rate 不变式：QPSK + roll_off=0 ⇒ bit_rate = 2 * n_slots * 12.5 GHz
    def conn(cid: int, first_slot: int, slots: int) -> ConnectionSpec:
        return ConnectionSpec(
            grid=grid, connection_id=cid, first_slot=first_slot, n_slots=slots,
            bit_rate_bps=2.0 * slots * SLOT_WIDTH_HZ, modulation="QPSK",
            symbol_rate_hz=float(slots * SLOT_WIDTH_HZ), roll_off=0.0,
            launch_power_w=LAUNCH_POWER_W,
        )

    conns = [conn(SWEEPER_ID, 0, n_slots)]
    for i in range(n_interferers):
        conns.append(conn(i + 1, (i + 1) * 4, 4))  # 固定 4 槽 = 50 GHz

    fiber = FiberSpanConfig(length_km=SPAN_KM)
    ev = MultiBandISRSGN(
        grid, fiber, SYMBOL_RATE_HZ, connections=conns,
        channel_bandwidth_source="occupied", power_width_mode=power_width_mode,
    )
    power = np.full(ev.n_nodes, LAUNCH_POWER_W, dtype=np.float64)
    return ev, power


def reference_terms(ev, power_in_w) -> dict:
    """照 P1 (29)/(30)/(34) 原文独立算一遍三项噪声方差。

    所有"除 B 以外"的量（alpha / phi / gamma / T / r / L_eff）都取自评估器的
    **公开**预计算属性，或按论文定义就地重算；本函数不调用任何 njit 噪声内核，
    因此它对 B 的依赖是独立于被测实现的。
    """
    import numpy as np

    from network_nlo_eval.models.numba_kernels import _calc_gamma_coeff_scalar, _calc_raman_profile_jit

    n = ev.n_nodes
    f_rel = ev.f_rel_hz
    alpha = ev.alpha_power_npm_channels
    a_eff = ev.a_eff_m2_channels
    beta2, beta3 = ev.beta2_s2_m, ev.beta3_s3_m
    n2, cr, k_bar = ev.n2_m2_w, ev.cr_w_m_hz, ev.k_bar
    b = ev.channel_bandwidth_hz_channels

    # r(f)：P1 (18)。与被测实现同源，但它不是本探针要验的量（那是 O1/O2 线的职责）。
    total_p = float(np.sum(power_in_w))
    r_f = _calc_raman_profile_jit(f_rel, ev.f_m_hz, ev.f_M_hz, ev.delta_f_co_hz, total_p)
    t_fac = (2.0 * alpha - cr * r_f) ** 2
    l_eff = (1.0 - np.exp(-alpha * (SPAN_KM * 1e3))) / alpha

    spm = np.zeros(n)
    xpm = np.zeros(n)
    for i in range(n):
        if power_in_w[i] <= 0:
            continue
        g_ii = _calc_gamma_coeff_scalar(ev.lambdas_m[i], ev.lambdas_m[i], n2, a_eff[i], a_eff[i])
        phi_i = 1.5 * np.pi**2 * (beta2 + 2 * np.pi * f_rel[i] * beta3)
        if abs(phi_i) < 1e-35:
            phi_i = 1e-35
        # --- P1 (29) ---
        exp_i = np.exp(2 * cr * k_bar * l_eff[i] * r_f[i])
        bracket = (
            (t_fac[i] - alpha[i] ** 2) / alpha[i]
            * np.arcsinh(phi_i * b[i] ** 2 / (np.pi * alpha[i]))
            + (4 * alpha[i] ** 2 - t_fac[i]) / (2 * alpha[i])
            * np.arcsinh(phi_i * b[i] ** 2 / (2 * np.pi * alpha[i]))
        )
        spm[i] = (
            (4 / 9) * (power_in_w[i] ** 3 / b[i] ** 2)
            * (N_SPAN ** (1 + EPSILON_SPAN))
            * ((g_ii**2 * np.pi) / (phi_i * 3 * alpha[i] ** 2))
            * exp_i * bracket
        )

        # --- P1 (30) ---
        acc = 0.0
        for l in range(n):
            if l == i or power_in_w[l] <= 0:
                continue
            g_il = _calc_gamma_coeff_scalar(ev.lambdas_m[i], ev.lambdas_m[l], n2, a_eff[i], a_eff[l])
            phi_il = (
                2 * np.pi**2 * (f_rel[l] - f_rel[i])
                * (beta2 + np.pi * beta3 * (f_rel[l] + f_rel[i]))
            )
            if abs(phi_il) < 1e-35:
                phi_il = 1e-35
            exp_l = np.exp(2 * cr * k_bar * l_eff[l] * r_f[l])
            br_l = (
                (t_fac[l] - alpha[l] ** 2) / alpha[l]
                * np.arctan(phi_il * b[i] / alpha[l])
                + (4 * alpha[l] ** 2 - t_fac[l]) / (2 * alpha[l])
                * np.arctan(phi_il * b[i] / (2 * alpha[l]))
            )
            acc += (power_in_w[l] ** 2 / b[l]) * (g_il**2 / (phi_il * 3 * alpha[l] ** 2)) * exp_l * br_l
        xpm[i] = (32.0 / 27.0) * power_in_w[i] * N_SPAN * acc

    return {"spm": spm, "xpm": xpm, "alpha": alpha, "t_factor": t_fac, "phi_spm": None}


def _rel_err(a: float, b: float) -> float:
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


def run_probe() -> dict:
    import numpy as np

    from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    fiber = FiberSpanConfig(length_km=SPAN_KM)
    # 满均衡放大器：单跨段、无 ROADM，使 SNR 的分母就是三项噪声之和。
    edfa = EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)
    path = MultiSpanOpticalPath([fiber], [edfa])

    rows: list[dict] = []
    modes = ("fixed_power", "fixed_psd")
    for mode in modes:
        for ghz in SWEEP_GHZ:
            n_slots = int(round(ghz * 1e9 / SLOT_WIDTH_HZ))
            ev, power = _build_scenario(n_slots, power_width_mode=mode)
            res = ev.evaluate_path(path, power)
            # ⚠️ 参考实现必须吃**实际进入内核的功率**。evaluate_path 在入口处已按
            # power_width_mode 做过一次缩放（fixed_psd 下 P_i × B_i/Rs_nominal），
            # 所以这里复算同一缩放，否则 fixed_psd 口径会与被测实现差一个功率因子。
            eff_power = power * ev._power_width_scale()
            ref = reference_terms(ev, eff_power)

            b = ev.channel_bandwidth_hz_channels
            eff_power = power * ev._power_width_scale()
            # P1 (34)：单跨段、满均衡 ⇒ G = 10^(17/10)，F = 10^(5.5/10)
            gain = 10 ** (17.0 / 10)
            nf = 10 ** (5.5 / 10)
            ase_ref = np.maximum(N_SPAN * nf * H_PLANCK * ev.f_abs_hz * b * (gain - 1), 0.0)
            ase_ref[eff_power <= 0] = 0.0

            i = SWEEPER_ID
            # P1 (34)：单跨段、满均衡 ⇒ G = 10^(17/10)，F = 10^(5.5/10)
            gain = 10 ** (17.0 / 10)
            nf = 10 ** (5.5 / 10)
            ase_ref = np.maximum(N_SPAN * nf * H_PLANCK * ev.f_abs_hz * b * (gain - 1), 0.0)
            ase_ref[eff_power <= 0] = 0.0
            # SNR 的分母是三项之和；分子必须用**链路出口**信号功率（evaluate_path
            # 用的是 power 而非入纤功率），否则会把它误当成参考实现的分歧。
            sig_out = float(res.signal_power_w[i])
            total_ref = float(ref["spm"][i] + ref["xpm"][i] + ase_ref[i])
            snr_ref = 10 * np.log10(sig_out / total_ref)

            rows.append({
                "power_width_mode": mode,
                "n_slots": n_slots,
                "b_i_hz": float(b[i]),
                "f_i_hz": float(ev.f_abs_hz[i]),
                "b_l_hz": [float(v) for v in b[1:]],
                "p_i_w": float(eff_power[i]),
                "p_i_w_nominal": float(power[i]),
                "signal_power_out_w": sig_out,
                "measured": {
                    "spm_noise_w": float(res.spm_noise_w[i]),
                    "xpm_noise_w": float(res.xpm_noise_w[i]),
                    "ase_noise_w": float(res.ase_noise_w[i]),
                    "snr_db": float(res.snr_db[i]),
                },
                "reference_p1": {
                    "spm_noise_w": float(ref["spm"][i]),
                    "xpm_noise_w": float(ref["xpm"][i]),
                    "ase_noise_w": float(ase_ref[i]),
                    "snr_db": float(snr_ref),
                },
                "rel_err": {
                    "spm": _rel_err(float(res.spm_noise_w[i]), float(ref["spm"][i])),
                    "xpm": _rel_err(float(res.xpm_noise_w[i]), float(ref["xpm"][i])),
                    "ase": _rel_err(float(res.ase_noise_w[i]), float(ase_ref[i])),
                    "snr_db": _rel_err(float(res.snr_db[i]), float(snr_ref)),
                },
            })

    # --- 逐项解析标度：实测比值 vs 由公式算出的比值 ------------------------
    # 每一项的"公式比值"= 把 (29)/(30)/(34) 在两个 B 上各求一次值再相除。
    # 三项的 B 依赖互不相同，这正是本判据的判别力所在：
    #   (29) SPM: B^-2 前因子 × asinh(φ B²/πα) 括号 ⇒ 净衰减约 B^-2
    #   (30) XPM: atan(φ B_i/α) ⇒ 单调增、渐近饱和
    #   (34) ASE: f_i·B_i 严格线性
    term_scaling = []
    for mode in ("fixed_power", "fixed_psd"):
        sel = [r for r in rows if r["power_width_mode"] == mode]
        base = sel[0]
        for r in sel:
            entry = {"power_width_mode": mode, "b_ratio": r["b_i_hz"] / base["b_i_hz"]}
            for term in ("spm", "xpm", "ase"):
                meas = r["measured"][f"{term}_noise_w"] / base["measured"][f"{term}_noise_w"]
                pred = r["reference_p1"][f"{term}_noise_w"] / base["reference_p1"][f"{term}_noise_w"]
                entry[f"{term}_measured_ratio"] = meas
                entry[f"{term}_formula_ratio"] = pred
                entry[f"{term}_rel_err"] = _rel_err(meas, pred)
            term_scaling.append(entry)

    # --- 解析标度：ASE 对 f_i·B_i 严格线性 (34) ----------------------------
    ase_scaling = []
    for mode in ("fixed_power", "fixed_psd"):
        sel = [r for r in rows if r["power_width_mode"] == mode]
        base = sel[0]
        for r in sel:
            ratio_meas = r["measured"]["ase_noise_w"] / base["measured"]["ase_noise_w"]
            # (34) = h·f_i·F·B_i·G·N ⇒ 比值 = (f_i·B_i)_new / (f_i·B_i)_old。
            # 注意灵活栅格下 n_slots 变化会连带平移该连接的中心频率，
            # 故隔离 B 的解析标度必须连 f_i 一起比，否则会把频移误当成 B 的作用。
            ratio_pred = (r["f_i_hz"] * r["b_i_hz"]) / (base["f_i_hz"] * base["b_i_hz"])
            ase_scaling.append({
                "power_width_mode": mode,
                "b_ratio": r["b_i_hz"] / base["b_i_hz"],
                "f_ratio": r["f_i_hz"] / base["f_i_hz"],
                "measured_ase_ratio": ratio_meas,
                "predicted_ase_ratio_eq34": ratio_pred,
                "rel_err": _rel_err(ratio_meas, ratio_pred),
            })

    # --- SNR 是否真的随 B 变化（否则 B 根本没进 QoT）----------------------
    snr_varies = {}
    for mode in modes:
        snrs = [r["measured"]["snr_db"] for r in rows if r["power_width_mode"] == mode]
        snr_varies[mode] = {
            "snr_db_by_b": {str(r["b_i_hz"]): r["measured"]["snr_db"] for r in rows if r["power_width_mode"] == mode},
            "spread_db": max(snrs) - min(snrs),
            "strictly_varies": (max(snrs) - min(snrs)) > 1e-6,
        }

    worst = max(
        max(r["rel_err"].values()) for r in rows
    )
    worst_scaling = max(
        e[f"{t}_rel_err"] for e in term_scaling for t in ("spm", "xpm", "ase")
    )
    failures: list[str] = []
    if worst > REL_TOL:
        failures.append(f"Q-J2: 逐项最大相对偏差 {worst:.3e} > 容差 {REL_TOL:.0e}")
    if worst_scaling > REL_TOL:
        failures.append(f"Q-J2: 逐项解析标度最大相对偏差 {worst_scaling:.3e} > 容差 {REL_TOL:.0e}")
    if any(a["rel_err"] > REL_TOL for a in ase_scaling):
        failures.append("Q-J2: ASE 对 f_i·B_i 的线性标度 (34) 不符")
    for mode, rec in snr_varies.items():
        if not rec["strictly_varies"]:
            failures.append(f"Q-J2/Q-J6: {mode} 口径下 SNR 不随 B 变化 ⇒ B 根本没进 QoT")
    if len(modes) != 2:
        failures.append("Q-J6: 两种功率口径未各跑一组")

    return {
        "wp": "Q1",
        "judgment": "Q-J2+Q-J6",
        "rel_tol": REL_TOL,
        "sweep_b_ghz": list(SWEEP_GHZ),
        "power_width_modes": list(modes),
        "channel_bandwidth_source": "occupied",
        "reference": {
            "source": "照 P1 (29)/(30)/(34) 原文独立重写的标量参考实现（tools/bwqot_probe.py::reference_terms）",
            "not_a_copy_of": "models/numba_kernels.py 的三个 njit 噪声函数",
            "shared_inputs": (
                "alpha / a_eff / beta2 / beta3 / n2 / Cr / k_bar 取自评估器公开属性；"
                "r(f) 取自 _calc_raman_profile_jit（P1 (18)，属 O1/O2 线已覆盖的量，"
                "不是本探针的验证对象）。**B 的进入方式在参考实现与被测实现之间相互独立**。"
            ),
        },
        "scenario": {
            "span_km": SPAN_KM, "n_span": N_SPAN, "edfa_gain_db": 17.0,
            "edfa_nf_db": 5.5, "launch_power_w": LAUNCH_POWER_W,
            "n_interferers": 3, "interferer_slots": 4,
        },
        "rows": rows,
        "term_scaling_vs_p1": term_scaling,
        "ase_linearity_eq34": ase_scaling,
        "snr_vs_bandwidth": snr_varies,
        "max_rel_err": worst,
        "max_scaling_rel_err": worst_scaling,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = run_probe()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(f"verdict={doc['verdict']}  max_rel_err={doc['max_rel_err']:.3e}  tol={REL_TOL:.0e}")
    for mode, rec in doc["snr_vs_bandwidth"].items():
        print(f"  {mode:12s} SNR spread={rec['spread_db']:.4f} dB  {rec['snr_db_by_b']}")
    for f in doc["failures"]:
        print(f"  FAIL: {f}")
    if args.check:
        return 0 if doc["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())