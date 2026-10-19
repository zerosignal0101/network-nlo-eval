"""FG-18 §3 碰撞面：把两个施加点**收敛前 vs 收敛后**逐项并列。

因为 §0 已证明本仓 ``roadm_config`` 恒为 ``None``（790 次调用 / 0 次非 None），
在真实场景上"改前 vs 改后"必然逐位相同——那正是门 2/门 4 的读数。要把
**机制**说清楚，必须直接对**装配函数本身**做 A/B：同一份 ``ROADMConfig``，
分别喂给 FG 树的合并式写法与 FG-18 的折叠写法，逐项报偏差。

对照的五项逐条给（对应 FG-4 §1.3 的裁决口径）：

======================  ==========================  ==========================
项                收敛前（FG 树合并式）      收敛后（fold_roadm_terms）
======================  ==========================  ==========================
信号衰减           ``insertion_loss_db``   ``path_loss_db("express")``
                       + ``filtering_penalty_db`` = ``insertion_loss_db``
                                                  + ``express_attenuation_db``
噪声是否被滤波惩罚   否（惩罚被折进信号）    否（本层不动噪声）
``equalize`` 语义   忽略，总是衰减信号       为真时改注入 booster ASE
端口区分           无                      express（逐跨段层拿不到 hop 序号）
``booster_noise_figure_db``  未使用      参与 booster_base
======================  ==========================  ==========================

用法::

    PYTHONPATH=<repo>/src ~/miniconda3/envs/rl/bin/python \\
        scripts/fg18_assembly_ab.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

from network_nlo_eval.models.numba_kernels import _db_to_lin  # noqa: E402
from network_nlo_eval.network.elements import ROADMConfig  # noqa: E402
from network_nlo_eval.rwa.qot_checker import fold_roadm_terms  # noqa: E402

CONFIGS = {
    "fu4_base": dict(insertion_loss_db=5.0, filtering_penalty_db=0.2),
    "high_filtering": dict(insertion_loss_db=5.0, filtering_penalty_db=2.0),
    "express_attn_nonzero": dict(insertion_loss_db=5.0, express_attenuation_db=2.0),
    "equalize_off": dict(insertion_loss_db=5.0, equalize_output_power=False),
    "zero_filtering": dict(insertion_loss_db=5.0, filtering_penalty_db=0.0),
    "zero_express_attn": dict(insertion_loss_db=5.0, express_attenuation_db=0.0),
}


def before(cfg: ROADMConfig, p_in: np.ndarray) -> np.ndarray:
    """收敛前的合并式：只衰减信号，噪声原样返回。"""
    loss_lin = _db_to_lin(cfg.insertion_loss_db + cfg.filtering_penalty_db)
    return p_in / loss_lin


def after(cfg: ROADMConfig, p_in: np.ndarray) -> dict:
    """收敛后：按 fold_roadm_terms 的权威口径。"""
    terms = fold_roadm_terms(cfg, is_last_hop=False)
    if terms.equalize:
        return {"power": p_in, "booster_ase_db": float(10 * np.log10(terms.booster_base * (terms.loss - 1.0)))}
    return {"power": p_in / terms.loss, "booster_ase_db": None}


def main() -> int:
    p_in = np.array([1e-3, 2e-3, 5e-3], dtype=np.float64)
    rows = []
    for name, kwargs in CONFIGS.items():
        cfg = ROADMConfig(**kwargs)
        b = before(cfg, p_in)
        a = after(cfg, p_in)
        terms = fold_roadm_terms(cfg, is_last_hop=False)
        bitwise = bool(np.array_equal(b, a["power"]))
        rows.append(
            {
                "config": name,
                "kwargs": kwargs,
                "equalize_output_power": cfg.equalize_output_power,
                "before_signal_only_loss_db": cfg.insertion_loss_db + cfg.filtering_penalty_db,
                "after_loss_db": terms.loss_db,
                "loss_db_delta": terms.loss_db - (cfg.insertion_loss_db + cfg.filtering_penalty_db),
                "before_power": b.tolist(),
                "after_power": a["power"].tolist(),
                "bitwise_identical": bitwise,
                "max_rel": float(np.max(np.abs(a["power"] / b - 1.0))) if not bitwise else 0.0,
                "after_booster_ase_injected": a["booster_ase_db"] is not None,
                "after_booster_ase_placeholder_db": a["booster_ase_db"],
            }
        )

    n_same = sum(1 for r in rows if r["bitwise_identical"])
    payload = {
        "question": "两个施加点收敛前 vs 收敛后的逐项偏差（直接对装配函数做 A/B）",
        "why_not_scenario_level": (
            "§0 证明本仓 roadm_config 恒为 None（790 calls / 0 非 None），"
            "故现有场景上改前改后必然逐位相同（门 2 读数佐证）。"
            "机制只能靠对装配函数的直接 A/B 暴露。"
        ),
        "rows": rows,
        "summary": {
            "configs": len(rows),
            "bitwise_identical": n_same,
            "moved": len(rows) - n_same,
            "moved_configs": [r["config"] for r in rows if not r["bitwise_identical"]],
        },
    }
    out = REPO / "results/fg18_assembly_ab.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
