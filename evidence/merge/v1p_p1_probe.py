"""WP-MERGE-FG-FLEX **第 3 轮**判据 V1' 的 ``p1_closed`` 档探针。

为什么要有这条探针（第 3 轮改判的直接后果）：

第 2 轮的 V1 判据是「合并树的 4 个判决摘要必须与 ``wp/fg-integrate`` 逐位相同」。
那条判据**只**在 ``numba_kernels.py`` 被换成 FG 的字节之后才成立，而那份字节是
**P2 Eq. 21 的物理**（符号率口径、`(16/27)` 张量化）。编排方第 3 轮查明：这与
flexgrid-dev 的 **P1 (29)/(30)/(34) 闭式**（逐信道 ``B_i``）是**两套不同物理**，
于是把 V1 作废，改判为 V1'：

* ``nli_model="p2_gn"`` → 与 ``wp/fg-integrate @ 57c7221`` 逐位相同（FG 侧读数，
  复用第 2 轮已拿到的 7 场景 29 项，只需把档位写明）。
* ``nli_model="p1_closed"`` → 与 ``flexgrid-dev @ cb24704`` 逐位相同。

本探针就是后者的**读数工具**。它刻意**只用两棵树都有的符号**
（``SpectrumGrid`` / ``MultiBandISRSGN`` / ``FiberSpanConfig`` / ``EDFAConfig`` /
``ROADMConfig`` / ``MultiSpanOpticalPath``），因此**同一个脚本**能分别量
``flexgrid-dev`` 与合并树，两边的输出 JSON 可以逐字段比较。

用法（物理树由 ``PYTHONPATH`` 切换）::

    cd <tree root>
    PYTHONPATH=$PWD/src ~/miniconda3/envs/rl/bin/python \
        <MERGE_WT>/evidence/merge/v1p_p1_probe.py --out <json> [--compare <other.json>]

⚠️ **档位必须显式给**：合并树支持 ``nli_model`` 参数，本探针会检查签名里有没有它；
有就必须给 ``--nli-model``，不给直接报错——绝不静默测到另一个档位去。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: 与 FG 的 ``fg4a_scenarios`` 同口径的 C+L 频段规模（230 × 50 GHz @ 190.2127 THz）。
#: 逐位比较要看**每一个**元素，所以规模必须够大才谈得上"逐位"。
NUM_CHANNELS = 230
CHANNEL_SPACING_HZ = 50e9
CENTER_FREQUENCY_HZ = 190.2127e12
SYMBOL_RATE_HZ = 50e9

PHI_KURTOSIS_GAUSSIAN = 0.0
PHI_KURTOSIS_64QAM = -0.6190

#: 跨段 EDFA 的 NF（dB）。
EDFA_NF_DB = 5.0

SPAN_KEYS = ("p_out_w", "sigma2_spm_w", "sigma2_xpm_w", "sigma2_ase_w")
PATH_KEYS = (
    "signal_power_w",
    "spm_noise_w",
    "xpm_noise_w",
    "ase_noise_w",
    "snr_db",
    "span_count",
    "roadm_count",
)

def _provenance() -> dict[str, Any]:
    import network_nlo_eval

    root = Path(network_nlo_eval.__file__).resolve().parents[2]
    commit = (
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        if (root / ".git").exists()
        else f"unknown (no .git at {root})"
    )
    return {
        "phys_commit": commit,
        "phys_tree": str(Path(network_nlo_eval.__file__).resolve()),
        "python": sys.version.split()[0],
    }


def _to_list(value: Any) -> Any:
    """把读数转成 json 能**逐位**存下的形式。

    ``json`` 用 ``float.__repr__`` 序列化 Python float（能往返的最短串），
    所以 float 原样往返不变；数组转成嵌套 list 同理。
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _grid():
    from network_nlo_eval.core.spectrum import SpectrumGrid

    return SpectrumGrid(
        num_channels=NUM_CHANNELS,
        channel_spacing_hz=CHANNEL_SPACING_HZ,
        center_frequency_hz=CENTER_FREQUENCY_HZ,
    )


def _fiber(length_km: float, max_span_length_km: float = 100.0):
    from network_nlo_eval.network.elements import FiberSpanConfig

    return FiberSpanConfig(length_km=length_km, max_span_length_km=max_span_length_km)


def _edfa(nf_db: float = EDFA_NF_DB):
    from network_nlo_eval.network.elements import EDFAConfig

    return EDFAConfig(target_gain_db=None, noise_figure_db=nf_db)


def _full_load_power() -> np.ndarray:
    """全信道等功率的确定性入纤功率剖面（不用随机数，两棵树必须拿到同一份）。"""
    return np.full(NUM_CHANNELS, 1.0e-3, dtype=np.float64)


def _partial_load_power() -> np.ndarray:
    """隔一个空一个 + 频带边缘留 0 功率保护带的确定性剖面。"""
    power = np.zeros(NUM_CHANNELS, dtype=np.float64)
    power[1:-1:2] = 1.0e-3
    return power


def _build_evaluator(*, nli_model: str | None, length_km: float, ode_accel: str | None):
    """造一个 evaluator；``nli_model`` 只在支持它的树上给。

    ⚠️ **不给** ``nli_model`` 的树就是 ``flexgrid-dev``——它**只有** P1 那一套物理，
    所以"不给"不是"测错档"，而是"那棵树没有档位概念"。这一点写进 provenance，
    避免读者以为两边读数出自同一档。

    ⚠️ 全部用**关键字**传参：两棵树的 ``__init__`` 位置参数顺序不同
    （合并树把 ``phi_kurtosis`` 插在 ``connections`` 之前），按位置传会串位。
    ⚠️ **不传** ``phi_kurtosis``：P1 的 (29)/(30) 用的是非线性相位 ``φ_i``，
    **没有**峰度旋钮（那是 P2 Eq. 21 的 ``phi_kurtosis``）。P1 侧不存在这个参数。
    """
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN

    kwargs: dict[str, Any] = {}
    if nli_model is not None:
        kwargs["nli_model"] = nli_model
    if ode_accel is not None:
        kwargs["ode_accel"] = ode_accel
    return MultiBandISRSGN(
        _grid(),
        _fiber(length_km),
        symbol_rate_hz=SYMBOL_RATE_HZ,
        **kwargs,
    )


def _span_case(name: str, nli_model: str | None, *, power, span_length_m, nf_db, n_effective_spans):
    ev = _build_evaluator(nli_model=nli_model, length_km=100.0, ode_accel=None)
    p_out, s2_spm, s2_xpm, s2_ase = ev._calc_span_noise_and_power(
        power, span_length_m, _edfa(nf_db), None, n_effective_spans
    )
    return {
        "name": name,
        "kind": "span",
        "outputs": {
            "p_out_w": _to_list(p_out),
            "sigma2_spm_w": _to_list(s2_spm),
            "sigma2_xpm_w": _to_list(s2_xpm),
            "sigma2_ase_w": _to_list(s2_ase),
        },
    }


def _path_case(name: str, nli_model: str | None):
    """端到端路径：160 km 光纤拆 2 跨段 + 中间一个 ROADM（阻断 4 个槽）+ 100 km。"""
    from network_nlo_eval.models.isrs_gn import MultiSpanOpticalPath
    from network_nlo_eval.network.elements import ROADMConfig

    ev = _build_evaluator(nli_model=nli_model, length_km=100.0, ode_accel=None)
    optical_path = MultiSpanOpticalPath(
        fiber_configs=[
            _fiber(160.0, max_span_length_km=80.0),
            _fiber(100.0, max_span_length_km=100.0),
        ],
        edfa_configs=[_edfa(EDFA_NF_DB), _edfa(6.0)],
        roadm_configs=[
            ROADMConfig(
                insertion_loss_db=5.0,
                filtering_penalty_db=0.2,
                blocked_slots=[40, 41, 42, 43],
            ),
            None,
        ],
        link_keys=[(0, 1), (1, 2)],
        node_path=[0, 1, 2],
    )
    result = ev.evaluate_path(optical_path, _partial_load_power())
    return {
        "name": name,
        "kind": "path",
        "outputs": {
            "signal_power_w": _to_list(result.signal_power_w),
            "spm_noise_w": _to_list(result.spm_noise_w),
            "xpm_noise_w": _to_list(result.xpm_noise_w),
            "ase_noise_w": _to_list(result.ase_noise_w),
            "snr_db": _to_list(result.snr_db),
            "span_count": int(result.span_count),
            "roadm_count": int(result.roadm_count),
        },
    }


def capture(nli_model: str | None) -> dict[str, Any]:
    cases = [
        _span_case(
            "cl_full_100km",
            nli_model,
            power=_full_load_power(),
            span_length_m=100e3,
            nf_db=EDFA_NF_DB,
            n_effective_spans=1,
        ),
        _span_case(
            "cl_partial_100km",
            nli_model,
            power=_partial_load_power(),
            span_length_m=100e3,
            nf_db=EDFA_NF_DB,
            n_effective_spans=1,
        ),
        _span_case(
            "cl_full_nf6_80km",
            nli_model,
            power=_full_load_power(),
            span_length_m=80e3,
            nf_db=6.0,
            n_effective_spans=1,
        ),
        _span_case(
            "cl_short_1km",
            nli_model,
            power=_full_load_power(),
            span_length_m=1e3,
            nf_db=EDFA_NF_DB,
            n_effective_spans=1,
        ),
        _span_case(
            "cl_nspan10",
            nli_model,
            power=_partial_load_power(),
            span_length_m=100e3,
            nf_db=EDFA_NF_DB,
            n_effective_spans=10,
        ),
        _path_case("path_cplusl_roadm", nli_model),
    ]
    payload = {
        "probe": "v1p_p1_probe",
        "wp": "WP-MERGE-FG-FLEX",
        "round": 3,
        "criterion": "V1' p1_closed vs flexgrid-dev @ cb24704 (bit-exact)",
        "nli_model_requested": nli_model,
        "provenance": _provenance(),
        "cases": cases,
    }
    payload["readings_sha256"] = hashlib.sha256(
        json.dumps(
            [(c["name"], c["outputs"]) for c in cases], sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _flatten(payload: dict[str, Any]) -> dict[str, Any]:
    """把 cases 压平成 ``{case: {key: value}}``，供逐项比较。"""
    return {c["name"]: c["outputs"] for c in payload["cases"]}


def compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """逐项逐位比较两份读数。

    数组按**元素**比（``repr`` 精确），标量按 ``==`` 比。
    """
    a, b = _flatten(left), _flatten(right)
    mismatches: list[dict[str, Any]] = []
    total_values = 0
    for case in sorted(set(a) | set(b)):
        if case not in a or case not in b:
            mismatches.append({"case": case, "reason": "case missing on one side"})
            continue
        for key in sorted(set(a[case]) | set(b[case])):
            if key not in a[case] or key not in b[case]:
                mismatches.append({"case": case, "key": key, "reason": "key missing on one side"})
                continue
            va, vb = a[case][key], b[case][key]
            if isinstance(va, list) and isinstance(vb, list):
                if len(va) != len(vb):
                    mismatches.append(
                        {"case": case, "key": key, "reason": f"length {len(va)} != {len(vb)}"}
                    )
                    continue
                for i, (xa, xb) in enumerate(zip(va, vb, strict=True)):
                    total_values += 1
                    if repr(xa) != repr(xb):
                        mismatches.append(
                            {
                                "case": case,
                                "key": key,
                                "index": i,
                                "left": repr(xa),
                                "right": repr(xb),
                            }
                        )
            else:
                total_values += 1
                if repr(va) != repr(vb):
                    mismatches.append({"case": case, "key": key, "left": repr(va), "right": repr(vb)})
    return {
        "cases": len(set(a) | set(b)),
        "values_compared": total_values,
        "mismatches": len(mismatches),
        "mismatch_detail": mismatches[:20],
        "left_sha256": left.get("readings_sha256"),
        "right_sha256": right.get("readings_sha256"),
        "sha256_equal": left.get("readings_sha256") == right.get("readings_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--compare", default=None)
    parser.add_argument(
        "--nli-model",
        default=None,
        choices=("p1_closed", "p2_gn"),
        help="NLI 物理档位。合并树**必须**显式给；flexgrid-dev 没有这个参数。",
    )
    args = parser.parse_args()

    import inspect

    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN

    supports_gear = "nli_model" in inspect.signature(MultiBandISRSGN.__init__).parameters
    if supports_gear and args.nli_model is None:
        raise SystemExit(
            "this tree supports the nli_model gear; pass --nli-model explicitly "
            "(refusing to measure an implicit gear)."
        )
    if not supports_gear and args.nli_model is not None:
        raise SystemExit("this tree has no nli_model gear; drop --nli-model.")

    payload = capture(args.nli_model)
    payload["tree_supports_nli_gear"] = supports_gear
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"OK: cases={len(payload['cases'])} sha256={payload['readings_sha256']}")
    print(f"    tree={payload['provenance']['phys_tree']} commit={payload['provenance']['phys_commit']}")
    print(f"    nli_model_requested={payload['nli_model_requested']} gear_supported={supports_gear}")

    if args.compare:
        other = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        result = compare(payload, other)
        print(
            f"COMPARE: cases={result['cases']} values={result['values_compared']} "
            f"mismatched={result['mismatches']} sha256_equal={result['sha256_equal']}"
        )
        for detail in result["mismatch_detail"]:
            print(f"  MISMATCH {detail}")
        return 0 if result["mismatches"] == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
