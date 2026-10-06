r"""FG-U1 证据生成器：`legacy` vs `tabular` 拉曼谱的宽波段影响量化。

这是一个**证据生成器**，不是测试。它把本刀要报的每一个数字都落成 JSON：

* **谱形对比**：两档在若干代表性频差上的形状比，以及 `>15.6 THz` 处 legacy
  被人为置零的量级。
* **插值误差**：提交在 ``references/`` 里的谱表相对**闭式**谱的最大/均方误差
  （自证容差）。
* **影响量化**：在 **C+L（11.45 THz）** 与 **S+C+L+U（~27 THz）** 两档上，
  单跨段输出功率**拉曼倾斜 (dB)** 与**逐信道 SNR (dB)** 的 legacy/tabular 之差。
* **不对称近似量级**：玻色因子 `(n_B + 1)/(n_B + 1/2)` 在各频差上的取值，
  即"沿用对称近似"在本谱上值多少。
* **运行上下文**：实际 import 到的包路径、机器 load average。

用法::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-uwb-nb \\
        $HOME/miniconda3/envs/rl/bin/python \\
        scripts/fg_u1_raman_spectrum_evidence.py --out results/fg_u1_raman.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from network_nlo_eval.core.spectrum import SpectrumGrid  # noqa: E402
from network_nlo_eval.models.isrs_gn import MultiBandISRSGN  # noqa: E402
from network_nlo_eval.models.numba_kernels import (  # noqa: E402
    _build_g_matrix,
    _build_g_matrix_tabular,
    raman_g_pair_matrices,
    raman_g_pair_matrices_tabular,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig  # noqa: E402
from network_nlo_eval.physics import raman_spectrum as rs  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "fg_u1_raman.json"

#: SRS 增益系数，与 `isrs_gn._SRS_CR` 同值（两档共用同一幅度语义）。
CR_W_M_HZ = 0.028 / 1e3 / 1e12

#: C+L 冻结常数（与 `tests/fg4a_scenarios.py` 一致，保证可比）。
CL_NUM_CHANNELS = 230
CL_SPACING_HZ = 50e9
CL_CENTER_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9
SPAN_LENGTH_KM = 100.0
LAUNCH_POWER_W = 1e-3

#: S+C+L+U：27 THz 跨度、50 GHz 信道间隔的**合成**等间隔栅格。
#: 这不是某个 ITU SCLU 频段计划，而是一个跨度等效的均匀栅格——本刀要的是
#: "跨距 ~27 THz 时两档差多少"，频段内的疏密分布会改变绝对数值。
SCLU_NUM_CHANNELS = 541
SCLU_SPACING_HZ = 50e9
SCLU_CENTER_HZ = 190.2127e12

#: 声明并自证的插值容差（相对峰值的绝对偏差）。线性插值在 0.05 THz 步长、
#: 谱最陡处（13 THz 附近）的二阶误差量级约为 step^2 * |S''| / 8。
INTERP_TOLERANCE = 1e-3


def _package_path() -> str:
    """记录真正提供数字的那棵 ``network_nlo_eval`` 树。"""
    import network_nlo_eval

    return str(Path(network_nlo_eval.__file__).resolve())


def _loadavg() -> str:
    """记录运行时刻的 load average（跨 WP 纪律要求报计时时必须同时报负载）。"""
    return Path("/proc/loadavg").read_text(encoding="utf-8").strip()


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 1e-300))


def spectrum_evidence() -> dict[str, Any]:
    """谱形对比 + 插值误差 + 不对称近似量级。"""
    probe_thz = np.array([0.0, 1.0, 5.0, 10.0, 11.45, 13.2, 15.6, 18.0, 20.0, 25.0, 27.0, 30.0, 40.0])

    shape_table = np.asarray(rs.raman_gain_shape(probe_thz), dtype=np.float64)
    # legacy 形状：G = cr*df，折线在 13.2 达峰、15.6 归零；除以锚点即形状。
    legacy_shape = np.where(
        probe_thz <= 13.2,
        probe_thz / 13.2,
        np.where(probe_thz < 15.6, (15.6 - probe_thz) / 2.4, 0.0),
    )

    # 插值误差：表（线性插值） vs 闭式，在表区间内密集取样。
    df_grid, shape_grid = rs.load_raman_gain_table()
    dense = np.linspace(0.0, rs.TABLE_MAX_THZ, 200_001)
    interp_vals = np.array(
        [rs.interp_raman_gain_shape(v, df_grid, shape_grid) for v in dense[::100]], dtype=np.float64
    )
    dense_sparse = dense[::100]
    exact_vals = np.asarray(rs.raman_gain_shape(dense_sparse), dtype=np.float64)
    abs_err = np.abs(interp_vals - exact_vals)
    peak_value = float(rs.raman_gain_shape(rs.PEAK_ANCHOR_THZ))

    asym = {
        f"{d:g}": {
            "n_bose_300K": float(rs.bose_occupation(d, 300.0)),
            "stokes_over_symmetric": float(rs.asymmetry_factor(d, "stokes", 300.0)),
            "anti_stokes_over_symmetric": float(rs.asymmetry_factor(d, "anti_stokes", 300.0)),
        }
        for d in (1.0, 5.0, 11.45, 13.2, 20.0, 27.0, 30.0)
    }

    return {
        "metadata": dict(rs.SPECTRUM_METADATA),
        "table_csv": str(rs.TABLE_CSV_PATH),
        "table_rows": int(len(df_grid)),
        "table_range_thz": [float(df_grid[0]), float(df_grid[-1])],
        "table_step_thz": rs.TABLE_STEP_THZ,
        "closed_form_peak_thz": float(
            np.linspace(0.0, 40.0, 400_001)[np.argmax(rs.raman_gain_shape(np.linspace(0.0, 40.0, 400_001)))]
        ),
        "probe": {
            "df_thz": probe_thz.tolist(),
            "shape_tabular": shape_table.tolist(),
            "shape_legacy": legacy_shape.tolist(),
            "tabular_over_legacy": [
                (float(tab) / float(leg)) if leg > 1e-15 else None
                for tab, leg in zip(shape_table, legacy_shape, strict=True)
            ],
        },
        "interpolation": {
            "declared_tolerance_abs": INTERP_TOLERANCE,
            "n_probe": int(len(dense_sparse)),
            "max_abs_error_vs_closed_form": float(abs_err.max()),
            "mean_abs_error_vs_closed_form": float(abs_err.mean()),
            "within_tolerance": bool(abs_err.max() <= INTERP_TOLERANCE),
            "normalised_max_error": float(abs_err.max() / peak_value),
        },
        "asymmetry_symmetric_approximation_300K": asym,
        "tail_truncated_above_thz": rs.TABLE_MAX_THZ,
        "tail_value_at_truncation_rel_peak": float(rs.raman_gain_shape(rs.TABLE_MAX_THZ) / peak_value),
    }


def anchor_equivalence() -> dict[str, Any]:
    """证明两档在 13.2 THz 锚点**逐位相等**（这是"只差谱形"的根据）。"""
    probes = []
    for df_hz in (13.2e12, 2.0e12, 0.5e12, 27e12):
        f_hz = np.ascontiguousarray(np.array([190e12, 190e12 + df_hz]), dtype=np.float64)
        legacy = _build_g_matrix(f_hz, CR_W_M_HZ)
        tabular = _build_g_matrix_tabular(
            f_hz, CR_W_M_HZ, *rs.load_raman_gain_table(), rs.TABLE_STEP_THZ
        )
        probes.append(
            {
                "df_thz": df_hz / 1e12,
                "legacy_g": float(legacy[0, 1]),
                "tabular_g": float(tabular[0, 1]),
                "bit_identical": bool(legacy[0, 1] == tabular[0, 1]),
            }
        )
    return {"anchor_df_thz": rs.PEAK_ANCHOR_THZ, "probes": probes}


def _span_powers(
    *, num_channels: int, spacing_hz: float, center_hz: float, raman_spectrum: str, cr: float
) -> dict[str, Any]:
    """跑一个单跨段，返回出纤功率与逐信道 SNR；`cr=0` 即纯线性（无拉曼）参考。"""
    grid = SpectrumGrid(num_channels=num_channels, channel_spacing_hz=spacing_hz, center_frequency_hz=center_hz)
    fiber = FiberSpanConfig(length_km=SPAN_LENGTH_KM, max_span_length_km=SPAN_LENGTH_KM)
    ev = MultiBandISRSGN(
        grid=grid,
        ref_fiber_config=fiber,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        phi_kurtosis=0.0,
        raman_spectrum=raman_spectrum,
    )
    # 用公开的 G 构建 API 换掉 Cr，而不是去改模块级常量：这样"线性参考"走的是
    # 与被测档位**完全相同**的代码路径（含谱档），唯一的差别就是 Cr=0。
    # 注意必须按 ev.raman_spectrum 分派——否则会拿 legacy 矩阵覆盖掉 tabular 档。
    ev.cr_w_m_hz = cr
    builder = raman_g_pair_matrices_tabular if ev.raman_spectrum == "tabular" else raman_g_pair_matrices
    ev._g_gain, ev._g_loss = builder(ev.f_abs_hz, cr)

    p_in = np.full(num_channels, LAUNCH_POWER_W, dtype=np.float64)
    edfa = EDFAConfig(target_gain_db=None, noise_figure_db=5.0)
    p_out, s2_spm, s2_xpm, s2_ase = ev._calc_span_noise_and_power(p_in, SPAN_LENGTH_KM * 1e3, edfa, None, 1)
    total_noise = s2_spm + s2_xpm + s2_ase
    out_db = _db(p_out)
    snr_db = _db(p_out) - _db(total_noise)
    return {
        "raman_spectrum": raman_spectrum,
        "cr_w_m_hz": cr,
        "span_thz": (num_channels - 1) * spacing_hz / 1e12,
        "p_out_w": p_out,
        "tilt_total_profile_db": float(out_db.max() - out_db.min()),
        "snr_db": snr_db,
        "snr_min_db": float(snr_db.min()),
        "snr_max_db": float(snr_db.max()),
    }


def _band_tilt_and_snr(
    *, num_channels: int, spacing_hz: float, center_hz: float, raman_spectrum: str
) -> dict[str, Any]:
    """一个档位在给定频带上的结果，含相对 Cr=0 参考的**纯拉曼**倾斜。"""
    run = _span_powers(
        num_channels=num_channels,
        spacing_hz=spacing_hz,
        center_hz=center_hz,
        raman_spectrum=raman_spectrum,
        cr=CR_W_M_HZ,
    )
    linear = _span_powers(
        num_channels=num_channels,
        spacing_hz=spacing_hz,
        center_hz=center_hz,
        raman_spectrum=raman_spectrum,
        cr=0.0,
    )
    raman_only_db = _db(run["p_out_w"]) - _db(linear["p_out_w"])
    run["tilt_raman_only_db"] = float(raman_only_db.max() - raman_only_db.min())
    run["tilt_linear_only_db"] = linear["tilt_total_profile_db"]
    run["raman_only_profile_db"] = raman_only_db.tolist()
    run["power_out_db"] = _db(run["p_out_w"]).tolist()
    run["snr_db"] = run["snr_db"].tolist()
    run.pop("p_out_w")
    return run


def band_evidence() -> dict[str, Any]:
    """C+L 与 S+C+L+U 两档上的 legacy / tabular 对比。"""
    bands = {
        "c_plus_l": {"num_channels": CL_NUM_CHANNELS, "spacing_hz": CL_SPACING_HZ, "center_hz": CL_CENTER_HZ},
        "s_c_l_u": {
            "num_channels": SCLU_NUM_CHANNELS,
            "spacing_hz": SCLU_SPACING_HZ,
            "center_hz": SCLU_CENTER_HZ,
        },
    }
    out: dict[str, Any] = {}
    for name, cfg in bands.items():
        runs = {tier: _band_tilt_and_snr(raman_spectrum=tier, **cfg) for tier in ("legacy", "tabular")}
        snr_legacy = np.asarray(runs["legacy"]["snr_db"], dtype=np.float64)
        snr_tab = np.asarray(runs["tabular"]["snr_db"], dtype=np.float64)
        delta = snr_tab - snr_legacy
        worst = int(np.argmax(delta))
        best = int(np.argmin(delta))
        legacy, tabular = runs["legacy"], runs["tabular"]
        out[name] = {
            "config": cfg,
            "runs": runs,
            "delta": {
                "tilt_total_profile_db_tabular_minus_legacy": (
                    tabular["tilt_total_profile_db"] - legacy["tilt_total_profile_db"]
                ),
                "tilt_raman_only_db_tabular_minus_legacy": (
                    tabular["tilt_raman_only_db"] - legacy["tilt_raman_only_db"]
                ),
                "snr_delta_db_max": float(delta.max()),
                "snr_delta_db_min": float(delta.min()),
                "snr_delta_db_mean": float(delta.mean()),
                "snr_delta_db_rms": float(np.sqrt(np.mean(delta**2))),
                "worst_channel_index": worst,
                "worst_channel_snr_delta_db": float(delta[worst]),
                "best_channel_index": best,
                "best_channel_snr_delta_db": float(delta[best]),
            },
        }
    return out


def main() -> int:
    """生成证据 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 JSON 路径")
    args = parser.parse_args()

    payload = {
        "work_package": "FG-U1",
        "package_path": _package_path(),
        "platform": platform.platform(),
        "loadavg_at_run": _loadavg(),
        "spectrum": spectrum_evidence(),
        "anchor_equivalence": anchor_equivalence(),
        "bands": band_evidence(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    interp = payload["spectrum"]["interpolation"]
    print(f"package: {payload['package_path']}")
    print(f"loadavg : {payload['loadavg_at_run']}")
    print(
        "interp vs closed form: max={:.3e} (tolerance {:.0e}) -> {}".format(
            interp["max_abs_error_vs_closed_form"],
            interp["declared_tolerance_abs"],
            "OK" if interp["within_tolerance"] else "FAIL",
        )
    )
    for band, data in payload["bands"].items():
        legacy, tabular, delta = data["runs"]["legacy"], data["runs"]["tabular"], data["delta"]
        print(
            "{:>9} span={:.2f} THz | tilt(total) legacy={:.3f} tabular={:.3f} | "
            "tilt(Raman-only) legacy={:.3f} tabular={:.3f} | "
            "SNR delta [{:+.3f}, {:+.3f}] dB (rms {:.3f})".format(
                band,
                legacy["span_thz"],
                legacy["tilt_total_profile_db"],
                tabular["tilt_total_profile_db"],
                legacy["tilt_raman_only_db"],
                tabular["tilt_raman_only_db"],
                delta["snr_delta_db_min"],
                delta["snr_delta_db_max"],
                delta["snr_delta_db_rms"],
            )
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
