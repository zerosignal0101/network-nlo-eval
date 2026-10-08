"""FG-18 §5 取证：``physics/qot_checker.py::evaluate_path`` 遇 ``roadm_configs``
含 ``None`` 时抛 ``AttributeError`` —— **只取证，不修**。

复现不依赖 cupy/物理库：那一行（:319-320）在 ``propagate_path`` 与
``_receiver_metrics`` **之后**才执行，所以本探针用一个只填 ``active`` 逻辑的
替身来直接驱动那一段循环，避免为了取证去装 ``physics_nlo_eval``。

判据（三问，逐一给出读数）：

1. **会不会抛？** 含 ``None`` 的 ``roadm_configs`` 是否确实触发 ``AttributeError``。
2. **有没有活调用方？** 谁会构造出含 ``None`` 的 ``roadm_configs``。
3. **生产路径上会不会发生？** ``NetworkTopology.build_optical_path`` 是否可能
   产出 ``None``。
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import textwrap

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SRC = REPO / "src/network_nlo_eval/physics/qot_checker.py"


def _simulate_blocking(roadm_configs, n_channels: int = 4) -> list[bool]:
    """逐字复刻 ``evaluate_path`` :318-320 的三行（只看数据流，不跑物理）。

    ``active`` 必须是 **numpy** 布尔数组：源码里是 ``np.asarray(...) > 0``，
    而 ``active[[...]] = False`` 的花式索引**只有 numpy 才支持**。早先用 Python
    list 复刻会先撞上 ``TypeError: list indices must be integers``，把真正的
    ``AttributeError`` 掩盖成探针假象——与本刀在守卫上踩的坑同类。
    """
    import numpy as np

    active = np.ones(n_channels, dtype=bool)
    for roadm in roadm_configs or []:
        active[[index for index in roadm.blocked_channels if index < len(active)]] = False
    return active.tolist()


def main() -> int:
    from network_nlo_eval.network.elements import ROADMConfig
    from network_nlo_eval.network.topology import NetworkTopology

    findings: dict = {}

    # --- Q1: 会不会抛 ---
    cases = {
        "all_real": [ROADMConfig(), ROADMConfig()],
        "contains_None": [ROADMConfig(), None],
        "all_None": [None, None],
        "roadm_configs_None": None,
        "empty_list": [],
    }
    q1 = {}
    for name, cfg in cases.items():
        try:
            q1[name] = {"raised": None, "active": _simulate_blocking(cfg)}
        except Exception as exc:  # noqa: BLE001 - 取证需要记录异常类型
            q1[name] = {"raised": f"{type(exc).__name__}: {exc}"}
    findings["q1_does_it_raise"] = q1

    # --- Q3: 生产路径 build_optical_path 是否可能产出 None ---
    import inspect

    build_src = textwrap.dedent(inspect.getsource(NetworkTopology.build_optical_path))
    tree = ast.parse(build_src)
    appends = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "append"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "roadm_configs"
    ]
    findings["q3_build_optical_path"] = {
        "roadm_configs_append_count": len(appends),
        "appended_exprs": [ast.unparse(n.args[0]) for n in appends],
        # get_roadm_config 的返回类型注解（pydantic 模型类，转成字符串才能 JSON 化）
        "get_roadm_config_returns": str(
            inspect.signature(NetworkTopology.get_roadm_config).return_annotation
        ),
    }

    # --- Q2: 活调用方 ---
    findings["q2_callers"] = {
        "note": "见文档 §5；grep 证据在 results/fg18_crash_probe.json 同目录的报告中",
        "physics_evaluate_path_callers": [
            "src/network_nlo_eval/physics/comparison.py:90 (QoTComparer.evaluate, run_waveform=True)",
            "src/network_nlo_eval/physics/qot_checker.py:331 (evaluate_topology_path)",
            "src/network_nlo_eval/physics/qot_checker.py:347 (evaluate_channel_evm_and_ber, 自建 path)",
        ],
        "build_optical_path_produces_None": False,
        "live_callers_that_pass_None": [],
    }

    payload = {
        "question": "physics/qot_checker.py::evaluate_path 在 roadm_configs 含 None 时是否抛错",
        "file": str(SRC),
        "line": 319,
        "findings": findings,
        "verdict": {
            "raises_on_None": q1["contains_None"]["raised"] is not None,
            "production_path_safe": True,
            "fix_recommended": False,
            "fix_decision": "交回编排方裁（本刀不改：改它会动 evaluation 契约）",
        },
    }
    out = REPO / "results/fg18_crash_probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())