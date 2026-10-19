r"""FU-2 证据生成器：S+C+L+U 端到端评估 + 逐波段发射功率 + legacy/tabular 之差。

这是一个**证据生成器**，不是测试。它把本刀要报的每一个数字都落成 JSON，
且每个数字都能由 ``--check`` 重跑复算：

* **波段定义**：S/C/L/U 的波长与频率两套边界、guard 间隔、跨距口径，
  以及 U 波段逐条假设（:data:`network_nlo_eval.core.bands.U_BAND_ASSUMPTIONS`）。
* **逐波段功率**：两种功率配置的逐信道 profile，以及它们如何进入
  ``launch_power_profile_w``（逐信道）与 ``ConnectionSpec.launch_power_w``（逐连接）。
* **端到端评估**：经 ``evaluate_path`` 的完整多跨段链路结果，含
  N（ODE 节点数）、总带宽、占用、逐波段/逐信道 SNR、**拉曼倾斜**。
* **legacy vs tabular**：同一场景下两档的差，含逐波段最差 SNR 差。
* **运行上下文**：实际 import 到的包路径、机器 load average。

用法::

    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-uwb-nb \\
        $HOME/miniconda3/envs/rl/bin/python \\
        scripts/fu2_scu_evidence.py --out results/fg_u2_scu.json

    # 复算：重跑并与已提交 JSON 比对
    PYTHONPATH=$PWD/src NUMBA_CACHE_DIR=/tmp/fg-uwb-nb \\
        $HOME/miniconda3/envs/rl/bin/python \\
        scripts/fu2_scu_evidence.py --check --out results/fg_u2_scu.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from network_nlo_eval.core.bands import (  # noqa: E402
    SCLU_BAND_SPECS,
    U_BAND_ASSUMPTIONS,
    BandPlan,
    BandSpec,
    sclu_span_report,
)
from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid  # noqa: E402
from network_nlo_eval.models.isrs_gn import (  # noqa: E402
    MultiBandISRSGN,
    MultiSpanOpticalPath,
)
from network_nlo_eval.models.numba_kernels import (  # noqa: E402
    raman_g_pair_matrices,
    raman_g_pair_matrices_tabular,
)
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "fg_u2_scu.json"

#: SRS 增益系数，与 ``isrs_gn._SRS_CR`` 同值（两档共用同一幅度语义）。
CR_W_M_HZ = 0.028 / 1e3 / 1e12

#: 信道间隔 (Hz)。50 GHz 与既有 C+L / S+C+L+U 场景一致，便于与 FG-U1 对比。
CHANNEL_SPACING_HZ = 50e9
SYMBOL_RATE_HZ = 50e9

#: 单连接参数：300 Gbps 64QAM、roll_off=0 -> 50 Gbaud -> 恰好吃满一个 50 GHz 槽。
#: `slots_for_rate(300e9, 6, 0.0, 50e9) == 1`，与 ConnectionSpec 的强制不变式自洽。
CONN_BIT_RATE_BPS = 300e9
CONN_MODULATION = "64QAM"
CONN_ROLL_OFF = 0.0

#: 场景链路：两跨段 x 100 km，**不放 ROADM**。
#:
#: 曾试过在末端挂一个 15 dB ROADM，实测 SNR 与不加时**逐位相同**：``evaluate_path``
#: 对 ROADM 是把信号与已累积噪声**同除**同一个损耗因子，纯线性、不引入新噪声，
#: 故 SNR 不变（实测 roadm_count=1 vs 0，snr_min 均 -46.780097）。
#: 既然它对本场景的 SNR 毫无影响，就不放，让场景只含"跨段 + 拉曼"两件事。
SPAN_LENGTH_KM = 100.0
SPAN_COUNT = 2
EDFA_NOISE_FIGURE_DB = 5.0

#: 峰度修正：0.0 为高斯，-0.6190 为 64QAM（本场景信道就是 64QAM）。
PHI_KURTOSIS = -0.6190

#: 逐波段功率配置 (W)。两套：
#:  * ``flat``    —— 四段同功率 1e-3 W。可与 FG-U1 的合成均匀栅格直接对照
#:    （注意总功率 28.1 dBm 偏高，因为 642 道全开；此处只为可比性而保留）。
#:  * ``raman_precomp`` —— **拉曼倾斜预补偿**：按 flat 单跨段实测的逐频段
#:    Raman-only 平均偏移（U +3.55 / L -0.41 / C -1.35 / S -4.80 dB）反向加权，
#:    使短波长(S)多发、长波长(U)少发，把 8.35 dB 的频段间倾斜压平。
#:    该组数值由 legacy 单跨段实测反推，用于**演示逐波段功率确实改变结果**；
#:    它不是收敛后的优化解，也没有用 tabular 谱重新拟合。
POWER_PROFILES: dict[str, dict[str, float]] = {
    "flat": {"U": 1e-3, "L": 1e-3, "C": 1e-3, "S": 1e-3},
    "raman_precomp": {"U": 1.5e-4, "L": 3.6e-4, "C": 4.5e-4, "S": 1e-3},
}

#: 拓扑：跨段数 -> 名称。单跨段与 FG-U1 的 100 km 单跨段可比；双跨段演示
#: ``evaluate_path`` 的跨段噪声累积（倾斜会显著累积，SNR 因此低得多）。
TOPOLOGIES: dict[str, int] = {"single_span": 1, "two_span": 2}

#: --check 比对容差（相对）。numba 内核逐位确定，1e-9 纯属防御。
CHECK_RTOL = 1e-9


def _package_path() -> str:
    """记录真正提供数字的那棵 ``network_nlo_eval`` 树。"""
    import network_nlo_eval

    return str(Path(network_nlo_eval.__file__).resolve())


def _loadavg() -> str:
    """记录运行时刻的 load average（跨 WP 纪律要求报计时时必须同时报负载）。"""
    return Path("/proc/loadavg").read_text(encoding="utf-8").strip()


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(values, 1e-300))


def build_specs(profile_name: str) -> tuple[BandSpec, ...]:
    """按功率配置给每个频段填上 ``launch_power_w``，其余字段不动。"""
    powers = POWER_PROFILES[profile_name]
    return tuple(replace(spec, launch_power_w=powers[spec.name]) for spec in SCLU_BAND_SPECS)


def build_sclu_connections(specs: tuple[BandSpec, ...]) -> tuple[FlexibleGrid, list[ConnectionSpec], BandPlan]:
    """构造 S+C+L+U 连接集合。

    栅格横跨**包络**（U 下沿 -> S 上沿，35.963 THz），但**只在各频段内部**放信道：
    U 与 L 之间的 3.890 THz 空隙里一个信道也没有。空隙靠"不放连接"实现，而不是
    靠把它的信道功率设为 0——后者仍会让它在 ODE 里占一个节点，是个假信道。

    Args:
        specs: 已填好逐波段功率的频段定义。

    Returns
    -------
        ``(grid, connections, plan)``。

    Raises
    ------
        ValueError: 频段规格为空。
    """
    if not specs:
        raise ValueError("build_sclu_connections requires at least one band specification.")

    lowest_hz = min(spec.low_hz for spec in specs)
    highest_hz = max(spec.high_hz for spec in specs)
    # band_low_edge 下移半个信道，使第 k 个单槽连接的中心恰好落在
    # lowest_hz + k*CHANNEL_SPACING_HZ 上（避免 25 GHz 的半槽偏移）。
    n_slots = int(np.ceil((highest_hz - lowest_hz) / CHANNEL_SPACING_HZ)) + 1
    grid = FlexibleGrid(
        slot_width_hz=CHANNEL_SPACING_HZ,
        num_slots=n_slots,
        band_low_edge_hz=lowest_hz - CHANNEL_SPACING_HZ / 2,
    )

    connections: list[ConnectionSpec] = []
    for spec in specs:
        first = int(np.floor((spec.low_hz - grid.band_low_edge_hz) / CHANNEL_SPACING_HZ))
        last = int(np.ceil((spec.high_hz - grid.band_low_edge_hz) / CHANNEL_SPACING_HZ))
        for slot in range(first, last):
            centre = grid.centre_hz(slot, 1)
            if not spec.contains_hz(centre):
                continue
            connections.append(
                ConnectionSpec(
                    grid=grid,
                    connection_id=len(connections),
                    first_slot=slot,
                    n_slots=1,
                    bit_rate_bps=CONN_BIT_RATE_BPS,
                    modulation=CONN_MODULATION,
                    symbol_rate_hz=SYMBOL_RATE_HZ,
                    roll_off=CONN_ROLL_OFF,
                    launch_power_w=float(spec.launch_power_w),
                )
            )
    plan = BandPlan.from_connections(connections, specs)
    return grid, connections, plan


def build_path(n_spans: int) -> MultiSpanOpticalPath:
    """``n_spans`` 个 100 km 跨段的光路（无 ROADM，见 :data:`SPAN_LENGTH_KM` 处说明）。"""
    fiber = FiberSpanConfig(length_km=SPAN_LENGTH_KM, max_span_length_km=SPAN_LENGTH_KM)
    edfa = EDFAConfig(target_gain_db=None, noise_figure_db=EDFA_NOISE_FIGURE_DB)
    return MultiSpanOpticalPath(
        fiber_configs=[fiber] * n_spans,
        edfa_configs=[edfa] * n_spans,
        roadm_configs=None,
    )


def _evaluate(
    connections: list[ConnectionSpec],
    plan: BandPlan,
    path: MultiSpanOpticalPath,
    *,
    raman_spectrum: str,
    cr: float,
    phi_kurtosis: float = PHI_KURTOSIS,
) -> dict[str, Any]:
    """跑一次端到端 ``evaluate_path``，返回逐波段与整体的功率/SNR 摘要。

    ``cr=0`` 即"纯线性参考"：它走的是与被测档位**完全相同**的代码路径（含谱档），
    唯一差别是拉曼增益系数为 0，因此用它把纯拉曼倾斜从总轮廓里分离出来。
    """
    fiber = path.fiber_configs[0]
    evaluator = MultiBandISRSGN(
        grid=connections[0].grid,
        ref_fiber_config=fiber,
        symbol_rate_hz=SYMBOL_RATE_HZ,
        phi_kurtosis=phi_kurtosis,
        connections=connections,
        raman_spectrum=raman_spectrum,
    )
    # `evaluate_path` 逐跨段调用 `self.for_fiber(fiber)` 取**子**评估器，真正的
    # 计算发生在子对象上。因此 Cr 与 G 矩阵必须在父子两边都覆盖——只改父对象
    # 会被缓存的子对象丢弃（与 raman_spectrum 漏传同一类坑）。
    targets = [evaluator] + [evaluator.for_fiber(f) for f in path.fiber_configs]
    for target in targets:
        target.cr_w_m_hz = cr
        builder = (
            raman_g_pair_matrices_tabular if target.raman_spectrum == "tabular" else raman_g_pair_matrices
        )
        target._g_gain, target._g_loss = builder(target.f_abs_hz, cr)

    power_profile = plan.launch_power_profile_w()
    result = evaluator.evaluate_path(path, power_profile)

    total_noise = result.spm_noise_w + result.xpm_noise_w + result.ase_noise_w
    power_db = _db(result.signal_power_w)
    snr_db = result.snr_db
    snr_finite = np.where(np.isfinite(snr_db), snr_db, np.nan)

    per_band: dict[str, Any] = {}
    for band_index, spec in enumerate(plan.specs):
        idx = plan.band_element_indices(band_index)
        band_snr = snr_finite[idx]
        band_power = power_db[idx]
        per_band[spec.name] = {
            "n_connections": int(idx.size),
            "launch_power_w": spec.launch_power_w,
            "noise_figure_db": spec.noise_figure_db,
            "snr_db_min": float(np.nanmin(band_snr)),
            "snr_db_max": float(np.nanmax(band_snr)),
            "snr_db_mean": float(np.nanmean(band_snr)),
            "snr_db_p05": float(np.nanpercentile(band_snr, 5)),
            "power_out_db_min": float(band_power.min()),
            "power_out_db_max": float(band_power.max()),
            "power_out_db_span": float(band_power.max() - band_power.min()),
        }

    return {
        "raman_spectrum": raman_spectrum,
        "cr_w_m_hz": cr,
        "n_nodes": int(evaluator.n_nodes),
        "span_count": result.span_count,
        "roadm_count": result.roadm_count,
        "snr_db_min": float(np.nanmin(snr_finite)),
        "snr_db_max": float(np.nanmax(snr_finite)),
        "snr_db_mean": float(np.nanmean(snr_finite)),
        "tilt_total_profile_db": float(power_db.max() - power_db.min()),
        "snr_db_per_connection": [float(v) for v in snr_finite],
        "power_out_db_per_connection": [float(v) for v in power_db],
        "per_band": per_band,
    }


def _scenario_report(plan: BandPlan, connections: list[ConnectionSpec], specs: tuple[BandSpec, ...]) -> dict[str, Any]:
    """场景的 N / 总带宽 / 占用 / 逐波段功率。"""
    span = sclu_span_report(specs)
    centres = np.array([conn.centre_hz for conn in connections], dtype=np.float64)
    occupied_hz = float(centres.size * CHANNEL_SPACING_HZ)
    envelope_hz = float(span["envelope_span_thz"]) * 1e12
    profile = plan.launch_power_profile_w()
    total_launch_w = float(profile.sum())
    return {
        "n_ode_nodes": len(connections),
        "n_slots_in_grid": int(connections[0].grid.num_slots),
        "slot_width_hz": CHANNEL_SPACING_HZ,
        "channel_spacing_hz": CHANNEL_SPACING_HZ,
        "per_connection": {
            "bit_rate_bps": CONN_BIT_RATE_BPS,
            "modulation": CONN_MODULATION,
            "symbol_rate_hz": SYMBOL_RATE_HZ,
            "roll_off": CONN_ROLL_OFF,
        },
        "occupied_bandwidth_thz": occupied_hz / 1e12,
        "envelope_span_thz": envelope_hz / 1e12,
        "occupancy_vs_envelope": occupied_hz / envelope_hz,
        "occupancy_vs_band_material": occupied_hz / (float(span["sum_of_band_spans_thz"]) * 1e12),
        "connections_per_band": {spec.name: plan.element_count(i) for i, spec in enumerate(specs)},
        "per_band_launch_power_w": plan.per_band_launch_power_w(),
        "per_band_launch_power_dbm": {
            spec.name: (None if spec.launch_power_w <= 0 else 10.0 * np.log10(spec.launch_power_w / 1e-3))
            for spec in specs
        },
        "total_launch_power_w": total_launch_w,
        "total_launch_power_dbm": 10.0 * np.log10(total_launch_w / 1e-3),
        "launch_power_profile_w_distinct_values": sorted(float(v) for v in np.unique(profile)),
        "centre_frequency_thz_min": float(centres.min() / 1e12),
        "centre_frequency_thz_max": float(centres.max() / 1e12),
        "span_report": span,
    }


def _gap_audit(specs: tuple[BandSpec, ...], connections: list[ConnectionSpec]) -> dict[str, Any]:
    """审计：U/L 空隙里确实一个信道都没有。"""
    centres = np.array([conn.centre_hz for conn in connections], dtype=np.float64)
    u_band, l_band = specs[0], specs[1]
    in_gap = centres[(centres > u_band.high_hz) & (centres < l_band.low_hz)]
    return {
        "u_high_hz": u_band.high_hz,
        "l_low_hz": l_band.low_hz,
        "gap_thz": (l_band.low_hz - u_band.high_hz) / 1e12,
        "n_connections_in_gap": int(in_gap.size),
        "clean": bool(in_gap.size == 0),
    }


class _NonBandPlan:
    """给"消融布局"用的极简 plan，**只**提供 :func:`_evaluate` 用到的接口。

    为什么不用 :class:`BandPlan`：`BandPlan` 会拒绝"频段覆盖不全"的栅格，而消融
    布局的定义**就是**把 U/L 之间的空隙也铺满信道——它按构造不是合法的频段布局。
    所以这里绕开 ``BandPlan`` 的覆盖校验，按"最近频段"把每个信道归段。
    这是**消融专用**的旁路，不改变 ``BandPlan`` 的任何校验强度。
    """

    def __init__(self, labels: np.ndarray, specs: tuple[BandSpec, ...], power_w: float) -> None:
        self.specs = specs
        self.labels = labels
        self._power_w = power_w
        self._elements = tuple(
            np.flatnonzero(labels == index).astype(np.int_) for index in range(len(specs))
        )
        self.frequencies_hz = np.zeros(labels.size)

    def band_element_indices(self, band: int) -> np.ndarray:
        return self._elements[band]

    def launch_power_profile_w(self) -> np.ndarray:
        return np.full(self.labels.size, self._power_w)


def gap_ablation() -> dict[str, Any]:
    """消融：把 U/L 空隙**填满**信道，观察 legacy/tabular 之差怎么变。

    这是 §4.2 那个结论的证据来源：空隙切掉了 U 段的大 Δf 耦合伙伴，
    填满后误差应当回到 FG-U1 的量级。

    Returns:
        含两种布局下逐频段 delta 与全体最差的字典。
    """
    specs = tuple(replace(spec, launch_power_w=1e-3) for spec in SCLU_BAND_SPECS)
    lowest_hz = min(spec.low_hz for spec in specs)
    highest_hz = max(spec.high_hz for spec in specs)
    n_slots = int(np.ceil((highest_hz - lowest_hz) / CHANNEL_SPACING_HZ)) + 1
    grid = FlexibleGrid(
        slot_width_hz=CHANNEL_SPACING_HZ,
        num_slots=n_slots,
        band_low_edge_hz=lowest_hz - CHANNEL_SPACING_HZ / 2,
    )
    connections: list[ConnectionSpec] = []
    for slot in range(n_slots):
        if grid.centre_hz(slot, 1) >= highest_hz:
            continue
        connections.append(
            ConnectionSpec(
                grid=grid,
                connection_id=len(connections),
                first_slot=slot,
                n_slots=1,
                bit_rate_bps=CONN_BIT_RATE_BPS,
                modulation=CONN_MODULATION,
                symbol_rate_hz=SYMBOL_RATE_HZ,
                roll_off=CONN_ROLL_OFF,
                launch_power_w=1e-3,
            )
        )
    centres = np.array([conn.centre_hz for conn in connections], dtype=np.float64)
    # 最近频段归段（消融专用；见 _NonBandPlan 的说明）
    labels = np.array(
        [
            int(np.argmin([abs(f - (s.low_hz + s.high_hz) / 2) for s in specs]))
            for f in centres
        ]
    )
    plan = _NonBandPlan(labels, specs, power_w=1e-3)
    path = build_path(1)

    runs = {
        mode: _evaluate(connections, plan, path, raman_spectrum=mode, cr=CR_W_M_HZ)
        for mode in ("legacy", "tabular")
    }
    delta = np.asarray(runs["tabular"]["snr_db_per_connection"], dtype=np.float64) - np.asarray(
        runs["legacy"]["snr_db_per_connection"], dtype=np.float64
    )
    per_band = {}
    for index, spec in enumerate(specs):
        idx = plan.band_element_indices(index)
        seg = delta[idx]
        per_band[spec.name] = {
            "n_connections": int(idx.size),
            "snr_delta_db_min": float(seg.min()),
            "snr_delta_db_max": float(seg.max()),
            "snr_delta_db_rms": float(np.sqrt(np.mean(seg**2))),
        }
    worst = int(np.argmin(delta))
    return {
        "note": "gap filled; channel->band assignment is nearest-centre (NOT a legal band layout)",
        "n_connections": len(connections),
        "snr_delta_db_min": float(delta.min()),
        "snr_delta_db_max": float(delta.max()),
        "snr_delta_db_rms": float(np.sqrt(np.mean(delta**2))),
        "worst_connection_centre_thz": float(centres[worst] / 1e12),
        "per_band": per_band,
    }


def build_payload() -> dict[str, Any]:
    """跑全部配置（功率档 x 拓扑），组装证据 JSON。"""
    runs: dict[str, Any] = {}
    scenarios: dict[str, Any] = {}
    plans: dict[str, BandPlan] = {}
    spec_sets: dict[str, tuple[BandSpec, ...]] = {}

    for profile_name in POWER_PROFILES:
        specs = build_specs(profile_name)
        spec_sets[profile_name] = specs
        _grid, connections, plan = build_sclu_connections(specs)
        plans[profile_name] = plan
        for topology_name, n_spans in TOPOLOGIES.items():
            key = f"{profile_name}/{topology_name}"
            path = build_path(n_spans)
            scenarios[key] = {
                **_scenario_report(plan, connections, specs),
                "topology": topology_name,
                "span_count": n_spans,
                "gap_audit": _gap_audit(specs, connections),
                "band_plan": plan.as_dict(),
            }
            profile_runs: dict[str, Any] = {}
            for mode in ("legacy", "tabular"):
                with_raman = _evaluate(connections, plan, path, raman_spectrum=mode, cr=CR_W_M_HZ)
                linear = _evaluate(connections, plan, path, raman_spectrum=mode, cr=0.0)
                raman_only_db = np.asarray(
                    with_raman["power_out_db_per_connection"], dtype=np.float64
                ) - np.asarray(linear["power_out_db_per_connection"], dtype=np.float64)
                with_raman["tilt_raman_only_db"] = float(max(raman_only_db) - min(raman_only_db))
                with_raman["raman_only_profile_db"] = [float(v) for v in raman_only_db]
                # 逐频段 Raman-only 平均偏移：用来看"频段间"倾斜，与总轮廓倾斜区分开。
                with_raman["raman_only_band_mean_db"] = {
                    spec.name: float(raman_only_db[plan.band_element_indices(i)].mean())
                    for i, spec in enumerate(specs)
                }
                with_raman["raman_only_band_mean_spread_db"] = float(
                    max(with_raman["raman_only_band_mean_db"].values())
                    - min(with_raman["raman_only_band_mean_db"].values())
                )
                del linear["power_out_db_per_connection"]
                del linear["snr_db_per_connection"]
                with_raman["linear_reference"] = {
                    "tilt_total_profile_db": linear["tilt_total_profile_db"],
                    "per_band": linear["per_band"],
                }
                profile_runs[mode] = with_raman
            runs[key] = profile_runs

    # legacy vs tabular 之差
    comparison: dict[str, Any] = {}
    for key, profile_runs in runs.items():
        legacy_snr = np.array(profile_runs["legacy"]["snr_db_per_connection"], dtype=np.float64)
        tabular_snr = np.array(profile_runs["tabular"]["snr_db_per_connection"], dtype=np.float64)
        delta = tabular_snr - legacy_snr
        per_band_delta: dict[str, Any] = {}
        profile_name = key.split("/")[0]
        plan = plans[profile_name]
        specs = spec_sets[profile_name]
        for band_index, spec in enumerate(specs):
            idx = plan.band_element_indices(band_index)
            band_delta = delta[idx]
            worst = int(idx[int(np.argmin(band_delta))])
            per_band_delta[spec.name] = {
                "n_connections": int(idx.size),
                "snr_delta_db_min": float(band_delta.min()),
                "snr_delta_db_max": float(band_delta.max()),
                "snr_delta_db_mean": float(band_delta.mean()),
                "snr_delta_db_rms": float(np.sqrt(np.mean(band_delta**2))),
                "worst_connection_id": worst,
                "worst_connection_centre_thz": float(plan.frequencies_hz[worst] / 1e12),
                "worst_snr_delta_db": float(band_delta.min()),
            }
        comparison[key] = {
            "snr_delta_db_min": float(delta.min()),
            "snr_delta_db_max": float(delta.max()),
            "snr_delta_db_mean": float(delta.mean()),
            "snr_delta_db_rms": float(np.sqrt(np.mean(delta**2))),
            "tilt_total_delta_db": (
                profile_runs["tabular"]["tilt_total_profile_db"]
                - profile_runs["legacy"]["tilt_total_profile_db"]
            ),
            "tilt_raman_only_delta_db": (
                profile_runs["tabular"]["tilt_raman_only_db"]
                - profile_runs["legacy"]["tilt_raman_only_db"]
            ),
            "raman_only_band_mean_spread_delta_db": (
                profile_runs["tabular"]["raman_only_band_mean_spread_db"]
                - profile_runs["legacy"]["raman_only_band_mean_spread_db"]
            ),
            "per_band": per_band_delta,
        }

    return {
        "work_package": "FU-2",
        "package_path": _package_path(),
        "platform": platform.platform(),
        "loadavg_at_run": _loadavg(),
        "link": {
            "span_length_km": SPAN_LENGTH_KM,
            "span_counts": dict(TOPOLOGIES),
            "roadm_count": 0,
            "roadm_note": "no ROADM; evaluate_path divides signal and accumulated noise by the "
            "same linear ROADM loss, so a terminal ROADM is SNR-neutral in this model",
            "edfa_noise_figure_db": EDFA_NOISE_FIGURE_DB,
            "phi_kurtosis": PHI_KURTOSIS,
        },
        "u_band_assumptions": list(U_BAND_ASSUMPTIONS),
        "power_profiles_w": POWER_PROFILES,
        "scenarios": scenarios,
        "gap_ablation": gap_ablation(),
        "runs": runs,
        "legacy_vs_tabular": comparison,
    }


def _flatten(payload: dict[str, Any]) -> dict[str, float]:
    """抽出所有标量数值，供 --check 逐项比对（数组不入表，避免体积与脆性）。"""
    flat: dict[str, float] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}.{key}" if prefix else str(key))
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            flat[prefix] = float(node)

    for key in ("link", "scenarios", "legacy_vs_tabular", "gap_ablation"):
        walk(payload[key], key)
    for run_key, profile_runs in payload["runs"].items():
        for mode, run in profile_runs.items():
            walk(run["per_band"], f"runs.{run_key}.{mode}.per_band")
            walk(run["raman_only_band_mean_db"], f"runs.{run_key}.{mode}.raman_only_band_mean_db")
            walk(
                {
                    "n_nodes": run["n_nodes"],
                    "span_count": run["span_count"],
                    "tilt_total_profile_db": run["tilt_total_profile_db"],
                    "tilt_raman_only_db": run["tilt_raman_only_db"],
                    "raman_only_band_mean_spread_db": run["raman_only_band_mean_spread_db"],
                    "snr_db_min": run["snr_db_min"],
                    "snr_db_max": run["snr_db_max"],
                    "snr_db_mean": run["snr_db_mean"],
                },
                f"runs.{run_key}.{mode}",
            )
    return flat


def check(payload: dict[str, Any], committed: dict[str, Any]) -> int:
    """复算比对：逐项标量相对误差必须 <= CHECK_RTOL。"""
    new_flat = _flatten(payload)
    old_flat = _flatten(committed)
    only_new = sorted(set(new_flat) - set(old_flat))
    only_old = sorted(set(old_flat) - set(new_flat))
    if only_new or only_old:
        print(f"FAIL: key set differs (+{len(only_new)} new, -{len(only_old)} missing)")
        for key in (only_new + only_old)[:20]:
            print(f"  {key}")
        return 1

    worst_key, worst_rel = "", 0.0
    bad: list[tuple[str, float, float, float]] = []
    for key, old_value in old_flat.items():
        new_value = new_flat[key]
        denom = max(abs(old_value), 1e-300)
        rel = abs(new_value - old_value) / denom
        if rel > worst_rel:
            worst_key, worst_rel = key, rel
        if rel > CHECK_RTOL:
            bad.append((key, old_value, new_value, rel))
    if bad:
        print(f"FAIL: {len(bad)}/{len(old_flat)} scalars exceed rtol {CHECK_RTOL:g}")
        for key, old_value, new_value, rel in bad[:20]:
            print(f"  {key}: committed={old_value!r} recomputed={new_value!r} rel={rel:.3e}")
        return 1
    print(f"OK: {len(old_flat)} scalars reproduce within rtol {CHECK_RTOL:g} (worst {worst_rel:.3e} at {worst_key})")
    return 0


def main() -> int:
    """生成（或复算比对）证据 JSON。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="输出 JSON 路径")
    parser.add_argument("--check", action="store_true", help="重跑并与已提交的 --out JSON 比对")
    args = parser.parse_args()

    payload = build_payload()

    print(f"package: {payload['package_path']}")
    print(f"loadavg : {payload['loadavg_at_run']}")
    for scenario_key, scenario in payload["scenarios"].items():
        print(
            "{:>24}: N={} nodes | occupied {:.4f} THz of {:.4f} THz envelope "
            "({:.2%}) | total launch {:.2f} dBm | per-band {}".format(
                scenario_key,
                scenario["n_ode_nodes"],
                scenario["occupied_bandwidth_thz"],
                scenario["envelope_span_thz"],
                scenario["occupancy_vs_envelope"],
                scenario["total_launch_power_dbm"],
                scenario["connections_per_band"],
            )
        )
        if not scenario["gap_audit"]["clean"]:
            raise AssertionError(f"{scenario_key}: U/L gap must contain no connections")
        for mode in ("legacy", "tabular"):
            run = payload["runs"][scenario_key][mode]
            print(
                "                         {:>7}: SNR [{:.3f}, {:.3f}] dB | tilt(total) {:.3f} dB | "
                "tilt(Raman-only) {:.3f} dB | band-mean Raman spread {:.3f} dB".format(
                    mode,
                    run["snr_db_min"],
                    run["snr_db_max"],
                    run["tilt_total_profile_db"],
                    run["tilt_raman_only_db"],
                    run["raman_only_band_mean_spread_db"],
                )
            )
        cmp_data = payload["legacy_vs_tabular"][scenario_key]
        worst_band = min(
            cmp_data["per_band"],
            key=lambda name: cmp_data["per_band"][name]["snr_delta_db_min"],
        )
        print(
            "                         tabular-legacy SNR delta [{:+.3f}, {:+.3f}] dB (rms {:.3f}) | "
            "worst band {} at {:.3f} THz ({:+.3f} dB)".format(
                cmp_data["snr_delta_db_min"],
                cmp_data["snr_delta_db_max"],
                cmp_data["snr_delta_db_rms"],
                worst_band,
                cmp_data["per_band"][worst_band]["worst_connection_centre_thz"],
                cmp_data["per_band"][worst_band]["worst_snr_delta_db"],
            )
        )

    if args.check:
        if not args.out.exists():
            print(f"FAIL: committed evidence {args.out} not found")
            return 1
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        return check(payload, committed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
