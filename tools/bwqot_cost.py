#!/usr/bin/env python3
"""WP-Q1 判据 Q-J3：不引入 ODE。

判据（跑前写死，不许放宽）::

    1) 单次 QoT 验证的**中位耗时 < 10 ms**（落盘实测读数）
    2) ``models/numba_kernels.py`` 的标识符里 grep 无 ``ode`` / ``RK4`` /
       ``ode_accel`` / ``phi_kurtosis`` 符号

第 2 条沿用 WP-M1 的做法：对内核做 ``tokenize`` 级扫描（先剥掉注释与字符串
字面量），再逐标识符比对禁用符号子串。裸 ``grep`` 会在论文标题的英文单词
"Model"/"Modeling" 上误命中，故裸计数只作透明记录、不参与判定。

第 1 条**完全复用 WP-M1 的计时口径**（``ms_alloc_cost.py`` 同一套
``build_stack`` + ``verify_allocation`` 预热与信道挑选），这样 Q-J3 与 M-J11
是同一把尺子，不因换测量实现而失去可比性。在此之上额外补一组
**带宽逐连接化之后**的计时（``channel_bandwidth_source="occupied"``，
逐连接异构 B_i），确认把 B 改成数组没有让热路径掉出预算。

用法::

    PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_cost.py --write
    PYTHONPATH=src ~/miniconda3/envs/rl/bin/python tools/bwqot_cost.py --check

产物：``results/bwqot_cost.json``
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import subprocess
import sys
import time
import tokenize
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "bwqot_cost.json"

#: 判据阈值（跑前写死）
MEDIAN_LATENCY_BUDGET_MS = 10.0
N_TIMED = 200
WARMUP = 5

FROZEN_KERNEL = "src/network_nlo_eval/models/numba_kernels.py"

#: _PREAMBLE_QOT.md §3 禁止搬入的 ODE 相关符号（大小写不敏感）
FORBIDDEN_SYMBOLS = ("ode", "rk4", "ode_accel", "phi_kurtosis")


def scan_forbidden_symbols() -> dict:
    """对冻结内核做标识符级扫描，返回命中明细。"""
    path = REPO / FROZEN_KERNEL
    source = path.read_text(encoding="utf-8")
    hits = []
    identifiers = 0
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type != tokenize.NAME:
            continue  # STRING 与 COMMENT 已排除
        identifiers += 1
        low = tok.string.lower()
        for symbol in FORBIDDEN_SYMBOLS:
            if symbol in low:
                hits.append({"token": tok.string, "line": tok.start[0], "symbol": symbol})
    grep_raw = {}
    for symbol in ("ode", "RK4", "ode_accel", "phi_kurtosis"):
        proc = subprocess.run(
            ["grep", "-c", symbol, FROZEN_KERNEL], cwd=str(REPO),
            capture_output=True, text=True,
        )
        grep_raw[symbol] = {"exit_code": proc.returncode, "count": proc.stdout.strip() or "0"}
    return {
        "file": FROZEN_KERNEL,
        "method": "tokenize 剥离注释与字符串字面量后，逐标识符比对禁用符号表",
        "n_identifiers_scanned": identifiers,
        "hits": hits,
        "grep_raw": grep_raw,
        "grep_raw_note": (
            "裸 grep -c 会在论文标题的英文单词 'Model'/'Modeling' 上命中 'ode'"
            "（属既有内容）。这些是字符串字面量里的参考文献名，不是代码符号，"
            "故不计入 hits，计数在此透明保留。"
        ),
    }


def _time_verify(validator, probe, path, channel, state) -> dict:
    """稳态单次 QoT 验证计时（预热后统计）。"""
    for _ in range(WARMUP):
        validator.verify_allocation(probe, path, channel, state)
    samples_ms = []
    for _ in range(N_TIMED):
        start = time.perf_counter()
        validator.verify_allocation(probe, path, channel, state)
        samples_ms.append((time.perf_counter() - start) * 1e3)
    s = sorted(samples_ms)
    return {
        "n_timed": N_TIMED,
        "warmup_calls": WARMUP,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "p95_ms": s[int(0.95 * (len(s) - 1))],
        "samples_ms": samples_ms,
    }


def time_single_verification() -> dict:
    """实测单次 QoT 验证耗时（与 WP-M1 的 M-J11 完全同一口径）。"""
    import numpy as np

    import ms_alloc_scenarios as sc
    from network_nlo_eval.network.state import NetworkState  # noqa: F401
    from network_nlo_eval.rwa.allocators import KSPFirstFitFlexAllocator

    topology, grid, gn, validator, path_cache, state = sc.build_stack()

    warm = []
    for i, n_slots in enumerate((1, 2, 3, 4, 6, 8, 2, 5)):
        warm.append(sc.make_request(sc.ServicePlan(
            service_id=i, source_id=0, destination_id=3,
            arrival_time=0.0, departure_time=100.0,
            bit_rate_gbps=400.0, n_slots=n_slots,
        )))
    allocator = KSPFirstFitFlexAllocator(
        path_cache=path_cache, qot_validator=validator, spectrum_grid=grid,
        network_topology=topology, slots_per_channel=sc.SLOTS_PER_CHANNEL,
    )
    for svc_req in warm:
        ok, svc = allocator.allocate(svc_req, state)
        if ok:
            state.allocate_service(svc)

    probe = sc.make_request(sc.ServicePlan(
        service_id=999, source_id=0, destination_id=3,
        arrival_time=0.0, departure_time=100.0, bit_rate_gbps=100.0, n_slots=1,
    ))
    path = list(path_cache.get_paths(0, 3)[0])

    # 必须挑真正空闲的信道，否则量到的是 verify_allocation 开头的早退分支。
    link_state = state.get_link_state(0, 1)
    free = [int(c) for c in np.flatnonzero(~link_state.occupied_channels)]
    if not free:
        raise RuntimeError("no free channel left to time; reduce warm-up services")
    channel = free[0]

    accepted, snrs = validator.verify_allocation(probe, path, channel, state)
    if not accepted:
        raise RuntimeError("probe was rejected; timing would measure a rejection path")

    timing = _time_verify(validator, probe, path, channel, state)
    timing.update({
        "channel_used": channel,
        "channel_was_free": True,
        "verify_accepted_on_probe_channel": bool(accepted),
        "probe_snr_db": float(snrs[channel]),
        "warm_services_occupancy": [int(c) for c in np.flatnonzero(link_state.occupied_channels)],
        "note": (
            "与 WP-M1 M-J11 同一口径：预热让 numba JIT 完成编译且不计入统计；"
            "计时信道确认为空闲且预验证返回 True，故覆盖完整 SNR 计算路径。"
        ),
    })
    return timing


def time_bandwidth_resolved() -> dict:
    """额外一组：逐连接**异构** B_i 下的单跨段 QoT 热路径计时。

    这不是判据要求（判据口径就是上面的 :func:`time_single_verification`），
    只是用来确认「把 B 从标量改成逐信道数组」没有让 njit 热路径掉出预算。
    """
    import numpy as np

    from network_nlo_eval.core.flexgrid import ConnectionSpec, FlexibleGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN, MultiSpanOpticalPath
    from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

    grid = FlexibleGrid.from_legacy_grid(8, 50e9, 193.1e12, slots_per_channel=4)
    conns = [
        ConnectionSpec(
            grid=grid, connection_id=i, first_slot=i * 4, n_slots=n_slots,
            bit_rate_bps=2.0 * n_slots * 12.5e9, modulation="QPSK",
            symbol_rate_hz=float(n_slots * 12.5e9), roll_off=0.0, launch_power_w=1e-3,
        )
        for i, n_slots in enumerate((1, 2, 3, 4, 6, 8))
    ]
    ev = MultiBandISRSGN(
        grid, FiberSpanConfig(length_km=80.0), 50e9, connections=conns,
        channel_bandwidth_source="occupied",
    )
    path = MultiSpanOpticalPath(
        [FiberSpanConfig(length_km=80.0)], [EDFAConfig(target_gain_db=17.0, noise_figure_db=5.5)]
    )
    power = np.full(ev.n_nodes, 1e-3, dtype=np.float64)
    res = ev.evaluate_path(path, power)

    samples_ms = []
    for _ in range(N_TIMED):
        start = time.perf_counter()
        ev.evaluate_path(path, power)
        samples_ms.append((time.perf_counter() - start) * 1e3)
    s = sorted(samples_ms)
    return {
        "n_timed": N_TIMED,
        "n_connections": ev.n_nodes,
        "b_i_hz": [float(b) for b in ev.channel_bandwidth_hz_channels],
        "heterogeneous": bool(np.ptp(ev.channel_bandwidth_hz_channels) > 0),
        "snr_db_min": float(np.min(res.snr_db)),
        "snr_db_max": float(np.max(res.snr_db)),
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "max_ms": max(samples_ms),
        "p95_ms": s[int(0.95 * (len(s) - 1))],
        "note": "单跨段 evaluate_path 的端到端耗时（含 Python 侧装配，非纯 njit）。",
    }


def run_cost() -> dict:
    scan = scan_forbidden_symbols()
    timing = time_single_verification()
    bw_timing = time_bandwidth_resolved()

    median = timing["median_ms"]
    latency_pass = median < MEDIAN_LATENCY_BUDGET_MS
    no_ode_pass = not scan["hits"]
    failures = []
    if not latency_pass:
        failures.append(f"Q-J3: 中位耗时 {median:.4f} ms >= 预算 {MEDIAN_LATENCY_BUDGET_MS} ms")
    if not no_ode_pass:
        failures.append(f"Q-J3: 内核出现 ODE 相关标识符 {scan['hits']}")
    if not bw_timing["heterogeneous"]:
        failures.append("Q-J3: 带宽探针没有真正构造出异构 B_i，计时无意义")

    return {
        "wp": "Q1",
        "judgment": "Q-J3",
        "budget_ms": MEDIAN_LATENCY_BUDGET_MS,
        "forbidden_symbols": list(FORBIDDEN_SYMBOLS),
        "latency": timing,
        "median_ms": median,
        "latency_budget_ok": latency_pass,
        "bandwidth_resolved_timing": bw_timing,
        "no_ode_scan": scan,
        "no_ode_ok": no_ode_pass,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
        "notes": [
            "ode_accel 是本仓既有加速档位名（isrs_gn.py），取值 'off'/'cached' 均为 "
            "memoization，不是 ODE 求解器；本 WP 未新增、未搬入任何 ODE 路径。",
            "计时口径与 WP-M1 的 M-J11 完全一致，便于两 WP 直接比较。",
            "WP-Q1 只把 B_i 由标量改成逐信道数组：内核的循环结构、求和次序、"
            "浮点运算序列均未改变，故 B 均匀时逐位相同、耗时同量级。",
        ],
        "self_check": {
            "median_under_budget": latency_pass,
            "kernel_has_no_ode_symbols": no_ode_pass,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    doc = run_cost()
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
    print(f"verdict={doc['verdict']}  median={doc['median_ms']:.4f} ms  budget={MEDIAN_LATENCY_BUDGET_MS} ms")
    print(f"  带宽探针 median={doc['bandwidth_resolved_timing']['median_ms']:.4f} ms  "
          f"heterogeneous={doc['bandwidth_resolved_timing']['heterogeneous']}")
    print(f"  ODE 符号命中: {doc['no_ode_scan']['hits'] or '无'}")
    for f in doc["failures"]:
        print(f"  FAIL: {f}")
    if args.check:
        return 0 if doc["verdict"] == "PASS" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())