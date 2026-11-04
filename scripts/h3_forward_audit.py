"""``pcfm_h3_raman`` 前向泵核/剖面拟合审计。

该脚本回答一个具体问题：H3 相对 ``hybrid_raman_paper`` 前向泵下多乐观的
0.1–0.3 dB，是否来自 SPP 的三次多项式拟合？

做法（全部为仓库内已有模型的只读调用）：

1. 用与 :class:`~network_nlo_eval.models.raman_ode.RamanPaperSpanSolver`
   完全相同的参数重建原始 ``profile_norm``，检查
   ``PCFH3RamanSpanKernel._resample_profile`` 是否只是把粗网格线性插值到
   拟合网格（应当逐位等于同一插值）。
2. 用 ``spp_samples`` / ``profile_polynomial`` / ``sci_kernel`` /
   ``xci_kernel`` 比较多项式表示与原始剖面：逐点残差、``u=0`` 外推残差、
   ``integral p**2`` 相对误差，以及 SCI 核相对**直接二维积分**
   ``int int p(z) p(z') Q(z-z') dz dz'`` 的相对误差。
3. 扫 ``poly_degree`` 3/5/7/9 × ``spp_steps`` 12/24/48/96，记录最终
   SPM/XPM/NLI-SNR 与相对 ``poly_degree=9, spp_steps=96`` 的差。
4. 扫 ``raman_steps`` 12/24/48/96，隔离粗 ODE 剖面本身的误差。
5. 逐项比较 H3 与 ``hybrid_raman_paper`` 的 ``nli_form="paper"`` /
   ``"integral"``：逐信道 SPM/XPM/SNR，以及其中 EGN 修正的贡献。

输出 ``results/raman/h3_forward_audit.json``。所有场景为单跨 100 km、
50 GBd、1 mW/pol、1450 nm 0.2 W 泵浦（或无泵/后向对照）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import sici

from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig, RamanPumpConfig

#: 与 ``scripts/dra_waveform_compare.py`` 完全一致的光纤参数。
FIBER = FiberSpanConfig(
    length_km=100.0,
    attenuation_db_km_ref=0.2,
    dispersion_parameter_d=15.68076226687911,
    dispersion_slope_s=0.04123833468291227,
    effective_area_um2_ref=80.0,
    nonlinear_index_n2=2.6e-20,
)

SYMBOL_RATE_HZ = 50e9
LAUNCH_POWER_W = 1e-3
SPAN_LENGTH_M = FIBER.length_km * 1e3
EDFA = EDFAConfig(noise_figure_db=5.5)

#: 扫的拟合配置。
POLY_DEGREES = (3, 5, 7, 9)
SPP_STEPS = (12, 24, 48, 96)
#: 粗 ODE 步数扫描。
RAMAN_STEPS = (12, 24, 48, 96)
#: 拟合参考点。
REFERENCE_POLY_DEGREE = 9
REFERENCE_SPP_STEPS = 96

#: 场景：信道数 × 泵浦配置。
CHANNEL_COUNTS = (1, 3, 5)
CONFIGS = ("nopump", "fwd_0p2w", "bwd_0p2w")


def _grid(n_channels: int) -> tuple[FlexibleGrid, list[ConnectionSpec]]:
    """构造与波形对照脚本相同的 100 GHz 栅格。"""
    grid = FlexibleGrid(slot_width_hz=12.5e9, num_slots=8 * n_channels + 8, band_low_edge_hz=192.95e12)
    connections = [
        ConnectionSpec(
            grid=grid,
            connection_id=index,
            first_slot=2 + 8 * index,
            n_slots=4,
            bit_rate_bps=100e9,
            modulation="QPSK",
            symbol_rate_hz=SYMBOL_RATE_HZ,
            roll_off=0.0,
            launch_power_w=LAUNCH_POWER_W,
        )
        for index in range(n_channels)
    ]
    return grid, connections


def _pumps(config: str) -> list[RamanPumpConfig]:
    """把配置名映射为泵浦列表。"""
    if config == "nopump":
        return []
    direction = "forward" if config.startswith("fwd") else "backward"
    power = float(config.split("_")[1].replace("p", ".").removesuffix("w"))
    return [
        RamanPumpConfig(
            wavelength_nm=1450.0,
            power_w=power,
            direction=direction,  # type: ignore[arg-type]
            attenuation_db_per_km=0.25,
        )
    ]


def _model(model_name: str, n_channels: int, config: str, **options: Any):
    """构造一个单跨模型。"""
    grid, connections = _grid(n_channels)
    pumps = _pumps(config)
    if pumps:
        options.setdefault("raman_pumps", pumps)
    if model_name == "pcfm_h3_raman":
        # 本文的表与 JSON 是旧 EGN 口径下的审计结果。WP6 起模型默认
        # egn="auto"（前向泵关闭 EGN），这里显式钉住 egn=True，保证 §3/§5
        # 与 results/raman/h3_forward_audit.json 可复现；auto 的三档对照见
        # scripts/h3_egn_auto.py。
        options.setdefault("egn", True)
    return create_model(
        model_name,
        grid,
        FIBER,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        connections=connections,
        slots_per_channel=8,
        channel_bandwidth_source="occupied",
        **options,
    )


def _run(model_name: str, n_channels: int, config: str, **options: Any) -> tuple[dict[str, Any], Any]:
    """跑一次路径，返回逐信道 SPM/XPM/NLI-SNR 与均值，以及已评估的模型。"""
    model = _model(model_name, n_channels, config, **options)
    path = MultiSpanOpticalPath([FIBER], [EDFA])
    power = np.full(n_channels, LAUNCH_POWER_W, dtype=np.float64)
    result = model.evaluate_path(path, power)
    signal = np.asarray(result.signal_power_w, dtype=np.float64)
    spm = np.asarray(result.spm_noise_w, dtype=np.float64)
    xpm = np.asarray(result.xpm_noise_w, dtype=np.float64)
    nli = spm + xpm
    snr = 10.0 * np.log10(signal / nli)
    payload = {
        "snr_nli_db_mean": float(np.mean(snr)),
        "snr_nli_db_per_channel": snr,
        "spm_noise_w_per_channel": spm,
        "xpm_noise_w_per_channel": xpm,
        "nli_noise_w_per_channel": nli,
        "spm_db_mean": _mean_db(spm),
        "xpm_db_mean": _mean_db(xpm),
        "snr_db_mean": float(np.mean(np.asarray(result.snr_db, dtype=np.float64))),
    }
    return payload, model


def _mean_db(values: np.ndarray) -> float | None:
    """功率均值换成 dBm；全零（如 1 信道无 XPM）返回 ``None``。"""
    mean = float(np.mean(values))
    return float(10.0 * np.log10(mean)) if mean > 0.0 else None


def _profile_positions(profile: np.ndarray) -> np.ndarray:
    """粗/细剖面的归一化位置 ``z/L``（不含 ``u=0``）。"""
    rows = profile.shape[0]
    return np.arange(1, rows + 1, dtype=np.float64) / rows


def _raw_profile(kernel, power_internal: np.ndarray) -> np.ndarray:
    """取核实际使用的原始归一化剖面（泵浦时是耦合 ODE 解）。"""
    if getattr(kernel, "active", False):
        return np.ascontiguousarray(kernel._raman_result(power_internal, SPAN_LENGTH_M).profile_norm)
    profile, _ = kernel._integrate_spp(power_internal, SPAN_LENGTH_M)
    return np.ascontiguousarray(profile)


def _poly_energy(q_scaled: np.ndarray) -> np.ndarray:
    """``int_0^1 p(u)**2 du``，由缩放基系数解析得到。"""
    n_poly = q_scaled.shape[0]
    ramp = np.arange(n_poly, dtype=np.float64)
    denom = ramp[:, None] + ramp[None, :] + 1.0
    return np.einsum("nc,kc,nk->c", q_scaled, q_scaled, 1.0 / denom)


def _direct_kernels(kernel, profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用原始剖面直接积分 SCI 核与剖面能量（无多项式）。

    ``K_SCI = int_0^L int_0^L p(z) p(z') Q(z-z') dz dz'``，
    ``Q(d) = 4 Si(lam d) / (c d)``，``lam = pi**2 beta2_eff B**2``、
    ``c = 4 pi**2 beta2_eff``（见 ``pcfm_kernels`` 的推导）。二维梯形积分
    与多项式核的差别就是拟合误差（外加梯形离散误差，相同剖面下 <1e-3）。
    """
    n_ch = profile.shape[1]
    span = profile.shape[0]
    u = np.concatenate(([0.0], _profile_positions(profile)))
    z = u * SPAN_LENGTH_M
    p = np.vstack([np.ones((1, n_ch), dtype=np.float64), profile])
    k_sci = np.empty(n_ch, dtype=np.float64)
    energy = np.empty(n_ch, dtype=np.float64)
    for ch in range(n_ch):
        beta2 = float(kernel.beta2_eff_sci[ch])
        lam = np.pi**2 * beta2 * float(kernel.b_hz[ch]) ** 2
        c = 4.0 * np.pi**2 * beta2
        distance = np.abs(z[None, :] - z[:, None])
        with np.errstate(divide="ignore", invalid="ignore"):
            q_mat = 4.0 * sici(lam * distance)[0] / (c * distance)
        q_mat[distance == 0.0] = 4.0 * lam / c
        pair = p[:, ch][:, None] * p[:, ch][None, :] * q_mat
        k_sci[ch] = np.trapz(np.trapz(pair, z, axis=1), z)
        energy[ch] = np.trapz(p[:, ch] ** 2, z)
    return k_sci, energy


def _fit_metrics(kernel, power_model_order: np.ndarray) -> dict[str, Any]:
    """多项式拟合相对原始剖面的逐点/积分残差（模型顺序）。"""
    power_internal = np.ascontiguousarray(power_model_order[kernel.order])
    profile = _raw_profile(kernel, power_internal)
    positions = _profile_positions(profile)
    extended_positions = np.concatenate(([0.0], positions))
    extended_profile = np.vstack([np.ones((1, profile.shape[1])), profile])

    q_scaled = kernel.profile_polynomial(SPAN_LENGTH_M, power_model_order)
    design = np.vander(extended_positions, kernel.n_poly, increasing=True)
    fitted = design @ q_scaled
    relative = np.abs(fitted - extended_profile) / np.maximum(extended_profile, 1e-300)

    poly_energy = _poly_energy(q_scaled) * SPAN_LENGTH_M
    direct_sci, direct_energy = _direct_kernels(kernel, profile)
    poly_sci = kernel.sci_kernel(SPAN_LENGTH_M, q_scaled)
    # XCI 核 = k_xci_matrix * int p**2 dz；用它交叉验证能量误差。
    poly_xci = kernel.xci_kernel(SPAN_LENGTH_M, q_scaled)
    direct_xci = kernel.k_xci_matrix * direct_energy[None, :]
    xci_mask = kernel.k_xci_matrix != 0.0
    xci_rel = (
        float(np.max(np.abs((poly_xci[xci_mask] - direct_xci[xci_mask]) / direct_xci[xci_mask])))
        if np.any(xci_mask)
        else 0.0
    )

    # 公开的 spp_samples 返回核实际喂给拟合的剖面（内部顺序）。
    _, sampled_profile = kernel.spp_samples(SPAN_LENGTH_M, power_model_order)
    target = kernel._sample_positions(SPAN_LENGTH_M)
    # 同一线性插值：重采样应当逐位等于它，用来验证映射本身没有额外误差。
    manual = np.stack(
        [np.interp(target, extended_positions, extended_profile[:, ch]) for ch in range(profile.shape[1])],
        axis=1,
    )

    inverse = kernel.inverse_order
    return {
        "poly_degree": int(kernel.poly_degree),
        "spp_steps": int(kernel.spp_steps),
        "raman_steps": int(getattr(kernel, "raman_steps", 0)),
        "profile_rows": int(profile.shape[0]),
        "fit_rows": int(q_scaled.shape[0]),
        "pointwise_max_rel": np.asarray(relative.max(axis=0))[inverse],
        "pointwise_median_rel": np.asarray(np.median(relative[1:], axis=0))[inverse],
        "u0_extrapolation_rel": np.asarray(relative[0])[inverse],
        "energy_rel_error": (poly_energy - direct_energy) / direct_energy,
        "sci_kernel_rel_error": (poly_sci - direct_sci) / direct_sci,
        "xci_kernel_max_rel_error": xci_rel,
        "profile_peak": np.asarray(profile.max(axis=0))[inverse],
        "profile_peak_position": np.asarray(positions[np.argmax(profile, axis=0)])[inverse],
        "profile_end": np.asarray(profile[-1])[inverse],
        "resample_rows": int(sampled_profile.shape[0]),
        "resample_vs_linear_interp_max_abs": float(np.max(np.abs(sampled_profile - manual))),
        "resample_is_direct": bool(sampled_profile.shape[0] == profile.shape[0]),
    }


def _compact_fit(fit: dict[str, Any]) -> dict[str, Any]:
    """扫表里只留聚合后的拟合误差，避免 JSON 膨胀。"""
    return {
        "sci_rel_max_abs": float(np.max(np.abs(fit["sci_kernel_rel_error"]))),
        "energy_rel_max_abs": float(np.max(np.abs(fit["energy_rel_error"]))),
        "xci_rel_max_abs": fit["xci_kernel_max_rel_error"],
        "pointwise_max_rel": float(np.max(fit["pointwise_max_rel"])),
        "pointwise_median_rel": float(np.max(fit["pointwise_median_rel"])),
        "u0_extrapolation_rel_max_abs": float(np.max(np.abs(fit["u0_extrapolation_rel"]))),
    }


def _scenario(n_channels: int, config: str) -> dict[str, Any]:
    """单个场景的全部审计量。"""
    power = np.full(n_channels, LAUNCH_POWER_W, dtype=np.float64)

    h3, h3_model = _run("pcfm_h3_raman", n_channels, config)
    h3_noegn, _ = _run("pcfm_h3_raman", n_channels, config, egn=False)
    paper, _ = _run("hybrid_raman_paper", n_channels, config, nli_form="paper")
    integral, _ = _run("hybrid_raman_paper", n_channels, config, nli_form="integral")
    # 对照：hybrid 的旧 PCFM 核路径，与 H3 共用同一套 SCI/XCI 代码和 EGN，
    # 只有剖面提供者（ODE 步数/拟合网格）不同。它把 H3 的剖面路径从
    # “核公式 + EGN”里分离出来。
    hybrid_pcfm, _ = _run(
        "hybrid_raman_paper",
        n_channels,
        config,
        nli_form="pcfm",
        poly_degree=3,
        spp_steps=12,
        egn=True,
    )

    # 用 evaluate_path 实际使用的模型（auto spp_steps 已在其中解析）。
    kernel = h3_model.kernel_for(FIBER)
    fit = _fit_metrics(kernel, power)

    sweep = []
    default_snr = None
    default_fit = None
    for poly_degree in POLY_DEGREES:
        for spp_steps in SPP_STEPS:
            result, sweep_model = _run(
                "pcfm_h3_raman",
                n_channels,
                config,
                poly_degree=poly_degree,
                spp_steps=spp_steps,
            )
            metrics = _fit_metrics(sweep_model.kernel_for(FIBER), power)
            if poly_degree == 3 and spp_steps == 12:
                default_snr = result["snr_nli_db_mean"]
                default_fit = metrics
            sweep.append(
                {
                    "poly_degree": poly_degree,
                    "spp_steps": spp_steps,
                    "snr_nli_db_mean": result["snr_nli_db_mean"],
                    "spm_db_mean": result["spm_db_mean"],
                    "xpm_db_mean": result["xpm_db_mean"],
                    "fit": _compact_fit(metrics),
                }
            )
    if default_fit is None or default_snr is None:  # POLY_DEGREES/SPP_STEPS 的首项
        raise RuntimeError("default fit configuration (3, 12) is not in the sweep")
    reference_snr = next(
        row["snr_nli_db_mean"]
        for row in sweep
        if row["poly_degree"] == REFERENCE_POLY_DEGREE and row["spp_steps"] == REFERENCE_SPP_STEPS
    )
    for row in sweep:
        row["delta_snr_vs_reference_db"] = row["snr_nli_db_mean"] - reference_snr
        row["delta_snr_vs_default_db"] = row["snr_nli_db_mean"] - default_snr

    ode_steps = []
    if config != "nopump":
        for raman_steps in RAMAN_STEPS:
            result, _ = _run(
                "pcfm_h3_raman",
                n_channels,
                config,
                poly_degree=REFERENCE_POLY_DEGREE,
                spp_steps=REFERENCE_SPP_STEPS,
                raman_steps=raman_steps,
            )
            ode_steps.append(
                {
                    "raman_steps": raman_steps,
                    "snr_nli_db_mean": result["snr_nli_db_mean"],
                    "spm_db_mean": result["spm_db_mean"],
                    "xpm_db_mean": result["xpm_db_mean"],
                }
            )
        ode_reference = ode_steps[-1]["snr_nli_db_mean"]
        for row in ode_steps:
            row["delta_snr_vs_finest_db"] = row["snr_nli_db_mean"] - ode_reference

    delta_paper = h3["snr_nli_db_mean"] - paper["snr_nli_db_mean"]
    delta_integral = h3["snr_nli_db_mean"] - integral["snr_nli_db_mean"]
    return {
        "name": f"{n_channels}ch/{config}",
        "n_channels": n_channels,
        "config": config,
        "pump": [pump.__dict__ for pump in _pumps(config)],
        "h3": _serialise(h3),
        "h3_no_egn": _serialise(h3_noegn),
        "hybrid_paper": _serialise(paper),
        "hybrid_integral": _serialise(integral),
        "hybrid_pcfm_same_kernel": _serialise(hybrid_pcfm),
        "delta_h3_minus_hybrid_pcfm_db": {
            "mean": h3["snr_nli_db_mean"] - hybrid_pcfm["snr_nli_db_mean"],
            "per_channel": np.asarray(h3["snr_nli_db_per_channel"]) - np.asarray(hybrid_pcfm["snr_nli_db_per_channel"]),
        },
        "delta_hybrid_pcfm_minus_paper_db_mean": hybrid_pcfm["snr_nli_db_mean"] - paper["snr_nli_db_mean"],
        "delta_h3_minus_paper_db": {
            "mean": delta_paper,
            "per_channel": np.asarray(h3["snr_nli_db_per_channel"]) - np.asarray(paper["snr_nli_db_per_channel"]),
        },
        "delta_h3_minus_integral_db": {
            "mean": delta_integral,
            "per_channel": np.asarray(h3["snr_nli_db_per_channel"]) - np.asarray(integral["snr_nli_db_per_channel"]),
        },
        "delta_paper_minus_integral_db_mean": paper["snr_nli_db_mean"] - integral["snr_nli_db_mean"],
        "egn_contribution_db": h3["snr_nli_db_mean"] - h3_noegn["snr_nli_db_mean"],
        "egn_free_delta_h3_minus_paper_db": h3_noegn["snr_nli_db_mean"] - paper["snr_nli_db_mean"],
        "fit": _serialise(fit),
        "fit_sweep": _serialise(sweep),
        "fit_default_snr_shift_db": default_snr - reference_snr,
        "fit_default_kernel_error": {
            "sci_rel_max_abs": float(np.max(np.abs(default_fit["sci_kernel_rel_error"]))),
            "energy_rel_max_abs": float(np.max(np.abs(default_fit["energy_rel_error"]))),
        },
        "ode_steps": _serialise(ode_steps),
        "resolved_span_coherence": h3_model.span_coherence,
    }


def _serialise(value: Any) -> Any:
    """把 numpy 标量/数组与 ``inf``/``nan`` 转成 JSON 可写形式。"""
    if isinstance(value, dict):
        return {key: _serialise(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialise(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_serialise(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return _serialise(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def _summary(scenarios: list[dict[str, Any]]) -> dict[str, Any]:
    """按信道数汇总：拟合误差界与泵浦相关超额。"""
    rows = []
    for n_channels in CHANNEL_COUNTS:
        by_config = {row["config"]: row for row in scenarios if row["n_channels"] == n_channels}
        rows.append(
            {
                "n_channels": n_channels,
                "delta_h3_minus_paper_db": {
                    config: by_config[config]["delta_h3_minus_paper_db"]["mean"] for config in CONFIGS
                },
                "pump_specific_excess_db": {
                    config: by_config[config]["delta_h3_minus_paper_db"]["mean"]
                    - by_config["nopump"]["delta_h3_minus_paper_db"]["mean"]
                    for config in CONFIGS
                },
                "h3_profile_path_effect_db": {
                    config: by_config[config]["delta_h3_minus_hybrid_pcfm_db"]["mean"] for config in CONFIGS
                },
                "same_kernel_control_gap_db": {
                    config: by_config[config]["delta_hybrid_pcfm_minus_paper_db_mean"] for config in CONFIGS
                },
                "egn_contribution_db": {config: by_config[config]["egn_contribution_db"] for config in CONFIGS},
                "fit_default_snr_shift_db": {
                    config: by_config[config]["fit_default_snr_shift_db"] for config in CONFIGS
                },
                "fit_default_kernel_error": {
                    config: by_config[config]["fit_default_kernel_error"] for config in CONFIGS
                },
                "ode_step_shift_db": {
                    config: (
                        max(abs(row["delta_snr_vs_finest_db"]) for row in by_config[config]["ode_steps"])
                        if by_config[config]["ode_steps"]
                        else None
                    )
                    for config in CONFIGS
                },
            }
        )
    fit_bound = max(abs(row["fit_default_snr_shift_db"]) for row in scenarios)
    return {
        "rows": rows,
        "fit_bound_db": fit_bound,
        "note": (
            "delta_h3_minus_paper_db 是 H3 默认配置（egn=True）减 hybrid_raman_paper；"
            "pump_specific_excess_db 是同配置减去无泵配置，用来判断前向泵是否额外放大差距；"
            "fit_default_snr_shift_db 是默认 (3,12) 相对 (9,96) 的 SNR 变化；"
            "ode_step_shift_db 是 raman_steps 12..96 下 (9,96) 结果的最大摆幅。"
        ),
    }


def main() -> None:
    """运行全部场景并写 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/raman/h3_forward_audit.json"),
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    scenarios = [_scenario(n_channels, config) for n_channels in CHANNEL_COUNTS for config in CONFIGS]
    output = {
        "generated_by": "scripts/h3_forward_audit.py",
        "config": {
            "fiber": {
                "length_km": FIBER.length_km,
                "attenuation_db_km_ref": FIBER.attenuation_db_km_ref,
                "dispersion_parameter_d": FIBER.dispersion_parameter_d,
                "dispersion_slope_s": FIBER.dispersion_slope_s,
                "effective_area_um2_ref": FIBER.effective_area_um2_ref,
                "nonlinear_index_n2": FIBER.nonlinear_index_n2,
            },
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "launch_power_w_per_pol": LAUNCH_POWER_W,
            "grid": "100 GHz spacing, 50 GHz occupied, FlexibleGrid 12.5 GHz",
            "pump": {"wavelength_nm": 1450.0, "power_w": 0.2, "attenuation_db_per_km": 0.25},
            "edfa_noise_figure_db": EDFA.noise_figure_db,
            "scenario": "single span, model order = ascending frequency",
            "h3_defaults": {"poly_degree": 3, "egn": True, "edge_fix": True, "spp_steps": "auto"},
        },
        "summary": _summary(scenarios),
        "scenarios": _serialise(scenarios),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    header = (
        f"{'scenario':<14}{'H3':>9}{'paper':>9}{'integ':>9}"
        f"{'d(paper)':>10}{'d(integ)':>10}{'EGN':>8}{'fit':>8}{'ODE':>8}{'d(ctrl)':>9}"
    )
    print(header)
    for row in scenarios:
        fit_shift = row["fit_default_snr_shift_db"]
        ode_shift = max(abs(item["delta_snr_vs_finest_db"]) for item in row["ode_steps"]) if row["ode_steps"] else 0.0
        print(
            f"{row['name']:<14}"
            f"{row['h3']['snr_nli_db_mean']:>9.4f}"
            f"{row['hybrid_paper']['snr_nli_db_mean']:>9.4f}"
            f"{row['hybrid_integral']['snr_nli_db_mean']:>9.4f}"
            f"{row['delta_h3_minus_paper_db']['mean']:>+10.3f}"
            f"{row['delta_h3_minus_integral_db']['mean']:>+10.3f}"
            f"{row['egn_contribution_db']:>+8.3f}"
            f"{fit_shift:>+8.3f}"
            f"{ode_shift:>8.3f}"
            f"{row['delta_h3_minus_hybrid_pcfm_db']['mean']:>+9.3f}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
