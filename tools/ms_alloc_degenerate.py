#!/usr/bin/env python3
"""WP-M1 判据 M-J10：退化等价。

条件（写死，不许放宽）::

    slots_per_channel = 4   且   所有请求 n_slots = 1   且   同一 seed

此时多槽分配器必须退化成遗留单信道分配器：**逐请求的 (path, wavelength)
与 ``KSPFirstFitAllocator`` 逐位一致**。

为什么这在设计上成立（不是碰巧）：

1. 槽占用按**信道对齐**展开（``LinkState.allocate_slot_range``），故
   ``n_slots=1`` 的窗口占满它所在的整个 4 槽信道；
2. 于是「槽全空闲」集合 == 「信道全空闲」集合；
3. 候选窗口起点升序 ⇒ 锚信道 ``first_slot // 4`` 升序 == 遗留信道升序；
4. QoT 只吃锚信道，两套分配器在该信道上的 ``verify_allocation`` 输入完全相同；
5. 同信道多起点被去重，故尝试序列也一致。

本工具**不**断言上面任何一条推理，而是真跑两套分配器并逐行对比结果。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ms_alloc_degenerate.py --write
    ~/miniconda3/envs/rl/bin/python tools/ms_alloc_degenerate.py --check

产物：``results/ms_alloc_degenerate.json``
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

OUT = REPO / "results" / "ms_alloc_degenerate.json"


def _run_one(allocator_kind: str):
    """跑一遍完整事件循环，返回逐请求 ``(path, wavelength)`` 记录。

    两套分配器各自持有**独立但同构**的 NetworkState，因此占用演化互不干扰，
    比较的是纯粹的算法选择差异，而非状态串扰。
    """
    import ms_alloc_scenarios as sc
    from network_nlo_eval.rwa.allocators import KSPFirstFitAllocator, KSPFirstFitFlexAllocator
    from network_nlo_eval.simulation.engine import SimulatorEngine
    from network_nlo_eval.simulation.reporter import MetricsCollector

    topology, grid, gn, validator, path_cache, state = sc.build_stack()

    if allocator_kind == "legacy":
        allocator = KSPFirstFitAllocator(
            path_cache=path_cache,
            qot_validator=validator,
            spectrum_grid=grid,
        )
    elif allocator_kind == "flex":
        allocator = KSPFirstFitFlexAllocator(
            path_cache=path_cache,
            qot_validator=validator,
            spectrum_grid=grid,
            network_topology=topology,
            slots_per_channel=sc.SLOTS_PER_CHANNEL,
        )
    else:  # pragma: no cover
        raise ValueError(allocator_kind)

    plans = sc.degenerate_service_plans()
    services = [sc.make_request(plan) for plan in plans]

    recorded: dict[int, dict] = {}

    class _Capture:
        def __init__(self, inner):
            self.inner = inner

        def allocate(self, service_request, network_state):
            ok, svc = self.inner.allocate(service_request, network_state)
            if ok and svc is not None:
                recorded[service_request.service_id] = {
                    "path": list(svc.path),
                    "wavelength": int(svc.wavelength),
                    "first_slot": int(getattr(svc, "first_slot", 0)),
                    "n_slots": int(getattr(svc, "n_slots", 1)),
                }
            return ok, svc

        def release(self, allocated_service, network_state):
            return self.inner.release(allocated_service, network_state)

        def __getattr__(self, item):
            return getattr(self.inner, item)

    collector = MetricsCollector(
        network_topology=topology,
        spectrum_grid=grid,
        total_incoming_services=len(services),
    )
    engine = SimulatorEngine(
        allocator=_Capture(allocator),
        network_state=state,
        metrics_collector=collector,
        incoming_services=services,
    )
    engine.run()

    return [
        {
            "service_id": plan.service_id,
            "n_slots": plan.n_slots,
            "allocated": plan.service_id in recorded,
            "legacy_path": recorded.get(plan.service_id, {}).get("path"),
            "legacy_wavelength": recorded.get(plan.service_id, {}).get("wavelength"),
            "flex_path": None,
            "flex_wavelength": None,
        }
        for plan in plans
    ], recorded


def run_degenerate() -> dict:
    import ms_alloc_scenarios as sc

    legacy_rows, legacy_rec = _run_one("legacy")
    flex_rows, flex_rec = _run_one("flex")

    comparison = []
    n_compared = 0
    n_identical = 0
    for lrow, frow in zip(legacy_rows, flex_rows, strict=True):
        assert lrow["service_id"] == frow["service_id"]
        sid = lrow["service_id"]
        lrec = legacy_rec.get(sid)
        frec = flex_rec.get(sid)
        row = {
            "service_id": sid,
            "n_slots": 1,
            "legacy_allocated": lrec is not None,
            "flex_allocated": frec is not None,
            "legacy_path": lrec["path"] if lrec else None,
            "legacy_wavelength": lrec["wavelength"] if lrec else None,
            "flex_path": frec["path"] if frec else None,
            "flex_wavelength": frec["wavelength"] if frec else None,
            "flex_first_slot": frec["first_slot"] if frec else None,
            "identical": False,
        }
        if lrec is None and frec is None:
            row["identical"] = True  # 两端都拒：也算一致
            n_compared += 1
            n_identical += 1
        elif lrec is not None and frec is not None:
            row["identical"] = (
                lrec["path"] == frec["path"] and lrec["wavelength"] == frec["wavelength"]
            )
            n_compared += 1
            n_identical += int(row["identical"])
        comparison.append(row)

    return {
        "wp": "M1",
        "judgment": "M-J10",
        "scenario": "degenerate_slots_per_channel_4_all_single_slot",
        "conditions": {
            "slots_per_channel": sc.SLOTS_PER_CHANNEL,
            "all_requests_n_slots": 1,
            "seed": "N/A — 业务序列由 ServicePlan 显式给出，不依赖随机流（确定性更强）",
            "n_requests": len(legacy_rows),
        },
        "protocol": (
            "两套分配器各自跑一遍同构的独立 NetworkState，逐请求比较 (path, wavelength)；"
            "不由本工具推理正确性，直接比结果。"
        ),
        "comparison": comparison,
        "summary": {
            "n_compared": n_compared,
            "n_identical": n_identical,
            "n_allocated_legacy": sum(1 for r in comparison if r["legacy_allocated"]),
            "n_allocated_flex": sum(1 for r in comparison if r["flex_allocated"]),
            "all_identical": n_identical == n_compared and n_compared > 0,
        },
        "self_check": {
            "degenerate_conditions_hold": sc.SLOTS_PER_CHANNEL == 4
            and all(r["n_slots"] == 1 for r in comparison),
            "flex_first_slot_is_channel_aligned": all(
                r["flex_first_slot"] % sc.SLOTS_PER_CHANNEL == 0
                for r in comparison
                if r["flex_allocated"]
            ),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    doc = run_degenerate()

    failures = []
    if not doc["summary"]["all_identical"]:
        failures.append("M-J10: 存在逐请求 (path, wavelength) 不一致的请求")
    if not doc["self_check"]["degenerate_conditions_hold"]:
        failures.append("M-J10: 退化条件未成立")
    if not doc["self_check"]["flex_first_slot_is_channel_aligned"]:
        failures.append("M-J10: flex 分配的 first_slot 未按信道对齐")

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {OUT.relative_to(REPO)}")
        print(f"  逐请求一致 {doc['summary']['n_identical']}/{doc['summary']['n_compared']}")
        for row in doc["comparison"]:
            print(
                f"    svc{row['service_id']:<3} legacy(w={row['legacy_wavelength']}) "
                f"flex(w={row['flex_wavelength']}, slot={row['flex_first_slot']}) "
                f"identical={row['identical']}"
            )

    if args.check:
        if not OUT.is_file():
            print(f"FAIL 缺少 {OUT.relative_to(REPO)}")
            return 1
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        for key in ("comparison", "summary", "conditions"):
            if stored.get(key) != doc.get(key):
                print(f"FAIL 复算不一致: {key}")
                return 1
        if failures:
            for f in failures:
                print(f"FAIL {f}")
            return 1
        print("OK  M-J10 复算通过：退化条件下逐请求 (path, wavelength) 逐位一致")
        return 0

    if failures:
        for f in failures:
            print(f"FAIL {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())