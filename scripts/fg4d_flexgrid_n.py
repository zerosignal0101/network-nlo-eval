"""FG-4d：S+C+L / S+C+L+U 灵活栅格下的真实 ODE 节点数 N。

FG-4c 的 N=900/2100 来自"50/25 GHz 固定栅格铺满超宽带"的**假设**。本脚本
把它换成**灵活栅格下的真实 N**，并且区分两种模式——这是本 WP 最重要的结构性发现：

模式 A（当前生产实现，RL 仓实际走的路）
    `launch_power_profile_w` 长度 = `spectrum_grid.num_channels`，且
    `_calc_span_noise_and_power` 的形状契约要求 `power_in_w.shape == (n_nodes,)`
    （`isrs_gn.py:368`）。未占用信道以 **0 功率**参与 ODE，仍然是**一个节点**。
    ⇒ N = 栅格宽度，**与负载无关**。C+L 恒为 230。

模式 B（FG-D 设计的逐连接模式，`connections=` 已给出）
    `f_abs_hz` 由 `ConnectionSpec.centre_hz` 逐条构造 ⇒ N = **活跃连接数**。
    一条占用 8 槽的宽连接只占 **1 个**节点。
    ⇒ N 随**连接数**长，随**带宽**不长。

本脚本对两档更宽的频段（S+C+L、S+C+L+U）在模式 B 下按现实负载算出真实 N，
并用真实内核调用计时，给出 N 落在哪一档。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

C_LIGHT = 299792458.0

#: 频段边缘（THz）。L/C/S 沿用 RL 仓 `config.py` 的值（由 nm 边界精确换算）；
#: U 段本仓无定义，此处按 ITU-T U 波段 1660–1770 nm 换算并在产物中标注来源。
BAND_EDGES_THZ: dict[str, tuple[float, float]] = {
    "L": (184.4877, 191.5607),
    "C": (191.5607, 195.9428),
    "S": (195.9428, 205.3373),
    "U": (C_LIGHT / 1770e-9 / 1e12, C_LIGHT / 1660e-9 / 1e12),  # 169.3733 – 180.5983
}

#: 灵活栅格槽宽。FG-2 的冻结测试用 12.5 GHz。
SLOT_WIDTH_HZ = 12.5e9

#: RL 仓 `request.py::DEFAULT_BITRATE_SNR` 的现实业务阶梯 (Gbps, SNR dB)。
#: 调制按每 GHz 谱效率的常见搭配给出；`slots_for_rate` 会用 ceil 强制槽数。
REALISTIC_DEMANDS: tuple[tuple[float, str], ...] = (
    (100.0, "QPSK"),
    (200.0, "QPSK"),
    (300.0, "16QAM"),
    (400.0, "64QAM"),
)

#: 各频段组合的负载档（占满度）。1.0 = 该栅格被连接**完全铺满**。
LOAD_FRACTIONS: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)


def tier_spans() -> dict[str, list[str]]:
    """三档频段组合。"""
    return {
        "C+L": ["L", "C"],
        "S+C+L": ["L", "C", "S"],
        "S+C+L+U": ["L", "C", "S", "U"],
    }


def build_grid(band_names: list[str], slot_width_hz: float):
    """按频段组合构造 FlexibleGrid（低频在下沿、高频在上沿）。"""
    from network_nlo_eval.core.flexgrid import FlexibleGrid

    low_edge_thz = min(BAND_EDGES_THZ[b][0] for b in band_names)
    high_edge_thz = max(BAND_EDGES_THZ[b][1] for b in band_names)
    # 留半个槽，使连接中心严格落在频段内部（与 FG-2 冻结配置的取法一致）。
    band_low_edge_hz = low_edge_thz * 1e12 + slot_width_hz / 2
    num_slots = int((high_edge_thz - low_edge_thz) * 1e12 // slot_width_hz)
    return FlexibleGrid(
        slot_width_hz=slot_width_hz, num_slots=num_slots, band_low_edge_hz=band_low_edge_hz
    )


def demand_slots(grid, demands=REALISTIC_DEMANDS) -> list[int]:
    """每条现实业务占用的槽数（`ConnectionSpec` 的不变式 3 会强制它）。"""
    from network_nlo_eval.core.flexgrid import bits_per_symbol, slots_for_rate

    return [
        slots_for_rate(
            bit_rate_bps=rate * 1e9,
            bits_per_symbol=bits_per_symbol(mod),
            roll_off=0.0,
            slot_width_hz=grid.slot_width_hz,
        )
        for rate, mod in demands
    ]


def pack_connections(grid, n_slots_each: list[int], load_fraction: float):
    """按需求槽宽**混合**贪心装箱，返回 (连接数, 占用槽数)。

    `n_slots_each` 是**每条业务的槽宽**（需求分布的一个样本），不是要铺满栅格的
    连接清单。现实负载下这四类业务按 `request.py::generate_requests` 的权重
    （∝ 1/rate）反复到达，所以这里把这条需求**循环**铺进栅格直到装不下——
    这才是"填满该频段需要多少条连接"的正确读法。

    装箱是**一维首次适配**，不做最优打包。它对连接数的偏差在 ±1 条量级
    （混排顺序固定、每条宽度在 4–8 槽之间），对 N 的量级结论无影响。
    """
    cycle = sorted(n_slots_each)
    free_slots = int(grid.num_slots * load_fraction)
    count = 0
    used = 0
    index = 0
    while True:
        width = cycle[index % len(cycle)]
        if used + width > free_slots:
            # 剩余空间放不下这条最窄的需求才停；否则试下一条（更宽的也放不下）。
            if width == cycle[0]:
                break
            index += 1
            continue
        used += width
        count += 1
        index += 1
    return count, used


def time_kernel_n(nodes_n: int, seconds: float = 0.6) -> dict:
    """在真实内核上测 N 节点的一次跨段调用耗时（mode B 形态）。"""
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN
    from network_nlo_eval.network.elements import FiberSpanConfig

    fiber = FiberSpanConfig(
        length_km=100.0,
        attenuation_alpha0=0.2 / 4.343,
        attenuation_alpha1=0.0,
        attenuation_alpha2=0.0,
        reference_wavelength_nm=1550.0,
        dispersion_parameter_d=16.0,
        dispersion_slope_s=0.058,
        effective_area_um2_ref=80.0,
        aeff_slope_um2_nm=0.0,
        nonlinear_index_n2=2.6e-20,
    )
    # 逐连接模式：N 个节点 = N 条连接，等间距铺在 C+L 上。
    centres = np.linspace(190.0e12, 195.0e12, nodes_n)
    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid

    grid = FlexibleGrid(slot_width_hz=12.5e9, num_slots=4 * max(nodes_n, 1), band_low_edge_hz=189.0e12)
    conns = []
    for i, c in enumerate(centres):
        # 4 槽 = 50 GHz，与 100 Gbps QPSK 对齐（slots_for_rate: 50/12.5 = 4）
        start = int(i * 4)
        if start + 4 > grid.num_slots:
            break
        conns.append(
            ConnectionSpec(
                grid=grid,
                connection_id=i,
                first_slot=start,
                n_slots=4,
                bit_rate_bps=100e9,
                modulation="QPSK",
                symbol_rate_hz=50e9,
                roll_off=0.0,
                launch_power_w=1e-3,
            )
        )
    evaluator = MultiBandISRSGN(grid, fiber, connections=conns)
    power = np.full(len(conns), 1e-3, dtype=np.float64)
    edfa = None

    evaluator._calc_span_noise_and_power(power, 100e3, edfa, n_effective_spans=1)  # 预热
    samples = []
    started = time.perf_counter()
    while time.perf_counter() - started < seconds:
        t0 = time.perf_counter()
        evaluator._calc_span_noise_and_power(power, 100e3, edfa, n_effective_spans=1)
        samples.append(time.perf_counter() - t0)
    return {
        "N": int(evaluator.n_nodes),
        "min_seconds": float(np.min(samples)),
        "median_seconds": float(np.median(samples)),
        "samples": len(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/fg4d_flexgrid_n.json")
    parser.add_argument("--time-nodes", type=int, nargs="*", default=[230, 417, 719, 900, 1668, 2877])
    args = parser.parse_args()

    import os

    rows = []
    for tier, bands in tier_spans().items():
        grid = build_grid(bands, SLOT_WIDTH_HZ)
        widths = demand_slots(grid)
        for load in LOAD_FRACTIONS:
            count, used = pack_connections(grid, widths, load)
            rows.append(
                {
                    "tier": tier,
                    "bands": bands,
                    "load_fraction": load,
                    "slot_width_hz": SLOT_WIDTH_HZ,
                    "num_slots": grid.num_slots,
                    "bandwidth_thz": grid.total_bandwidth_hz / 1e12,
                    "channels_at_50ghz": int(grid.total_bandwidth_hz // 50e9),
                    "demand_widths_slots": widths,
                    "connections_modeB_N": count,
                    "slots_used": used,
                    "mean_slots_per_connection": used / count if count else None,
                }
            )
            print(
                f"{tier:<9} load={load:>4.0%} slots={grid.num_slots:>5} "
                f"50G-ch={int(grid.total_bandwidth_hz // 50e9):>4} "
                f"N(modeB)={count:>4} mean_slots/conn="
                f"{used / count if count else 0:.1f}",
                flush=True,
            )

    timings = []
    for n in args.time_nodes:
        t = time_kernel_n(n)
        timings.append(t)
        print(
            f"kernel N={t['N']:>5} median={t['median_seconds'] * 1e3:>9.3f} ms "
            f"(loadavg {os.getloadavg()[0]:.1f})",
            flush=True,
        )

    payload = {
        "slot_width_hz": SLOT_WIDTH_HZ,
        "band_edges_thz": {k: list(v) for k, v in BAND_EDGES_THZ.items()},
        "u_band_source": "ITU-T U 波段 1660–1770 nm（按 c 精确换算）；本仓与 RL 仓均无 U 段定义",
        "realistic_demands_gbps_modulation": [list(d) for d in REALISTIC_DEMANDS],
        "rows": rows,
        "kernel_timings_modeB": timings,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
