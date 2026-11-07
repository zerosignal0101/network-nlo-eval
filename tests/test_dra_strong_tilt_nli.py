"""强增益倾斜 NLI 复核脚本的轻量单测。"""

from __future__ import annotations

import numpy as np
from dra_strong_tilt_nli import (
    _fmt_signed,
    _json_float,
    _nan_stat,
    build_analytic_inputs,
    build_arg_parser,
    evaluate_analytic,
    representative_indices,
    residual_stats,
)


def test_json_helpers_convert_non_finite_to_none() -> None:
    """NaN/Inf 写 JSON 时必须是 None。"""
    assert _json_float(np.nan) is None
    assert _json_float(np.inf) is None
    assert _json_float(1.25) == 1.25
    assert _nan_stat(np.array([1.0, np.nan, 3.0]), np.mean) == 2.0
    assert _nan_stat(np.array([np.nan]), np.mean) is None


def test_residual_stats_reports_center_edges_and_extremes() -> None:
    """残差统计要给出中心/两端/最大最小位置。"""
    analytic = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    waveform = np.array([9.0, 10.0, 10.0, 9.0, 8.0])
    stats = residual_stats(analytic, waveform, center_index=2, net_gain_db=np.arange(5.0))
    assert stats["count"] == 5
    assert stats["center_db"] == 2.0
    assert stats["edge_low_db"] == 1.0
    assert stats["edge_high_db"] == 6.0
    assert stats["max_db"] == 6.0
    assert stats["max_channel"] == 4
    assert stats["min_db"] == 1.0
    assert stats["min_channel"] == 0
    assert abs(stats["mean_db"] - 2.8) < 1e-12
    assert stats["rms_db"] is not None


def test_residual_stats_ignores_non_finite_channels() -> None:
    """单点 NaN 不应污染统计。"""
    analytic = np.array([10.0, np.nan, 12.0])
    waveform = np.array([9.0, 9.0, 10.0])
    stats = residual_stats(analytic, waveform, center_index=0)
    assert stats["count"] == 2
    assert stats["mean_db"] == 1.5


def test_representative_indices_include_center_and_edges() -> None:
    """代表信道必须覆盖两端与中心邻域。"""
    indices = representative_indices(152, 76)
    assert indices[0] == 0
    assert indices[-1] == 151
    assert {75, 76, 77}.issubset(indices)
    assert indices == sorted(set(indices))


def test_fmt_signed_handles_none() -> None:
    """空值在表格里显示为短横线。"""
    assert _fmt_signed(None) == "-"
    assert _fmt_signed(1.23456) == "+1.235"
    assert _fmt_signed(-1.23456) == "-1.235"


def _small_metadata() -> dict:
    """4ch 解析自检用的最小参考元数据。"""
    return {
        "plan": {
            "name": "tiny",
            "n_channels": 4,
            "channel_spacing_hz": 75e9,
            "center_frequency_hz": 190.15e12,
            "symbol_rate_hz": 64e9,
            "launch_power_dbm_per_pol": 0.0,
        },
        "fiber": {
            "length_km": 10.0,
            "alpha_db_per_km": 0.162,
            "beta2_ps2_per_km": -21.7,
            "beta3_ps3_per_km": 1e-4,
            "n2_si": 2.6e-20,
            "a_eff_um2": 80.0,
        },
        "pump": {"configs": {"nopump": []}},
    }


def test_build_analytic_inputs_and_evaluate_hybrid() -> None:
    """4ch 无泵的解析路径应返回有限逐信道 SNR。"""
    metadata = _small_metadata()
    grid, fiber_config, pumps = build_analytic_inputs(metadata, "nopump")
    assert grid.num_channels == 4
    assert pumps == []
    result = evaluate_analytic(
        "hybrid_raman_paper",
        grid,
        fiber_config,
        pumps,
        symbol_rate_hz=64e9,
        power_w=np.full(4, 1e-3),
        signal_polarizations=1,
    )
    assert result["snr_nli_db"].shape == (4,)
    assert np.all(np.isfinite(result["snr_nli_db"]))
    assert np.all(result["nli_power_w"] > 0.0)


def test_parser_defaults_and_probe_options() -> None:
    """CLI 默认值应与文档/脚本一致。"""
    parser = build_arg_parser()
    args = parser.parse_args([])
    assert args.plan == "cl152"
    assert args.config == "bwd_sixpumps"
    assert args.step_m == 0.48828125
    assert args.pump_pump is True
    assert args.probe_symbols is None
    probe = parser.parse_args(["--probe-symbols", "48", "--no-pump-pump"])
    assert probe.probe_symbols == 48
    assert probe.pump_pump is False
