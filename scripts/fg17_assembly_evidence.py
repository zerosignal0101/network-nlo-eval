"""FG-17 §0 取证：``physics/qot_checker.py`` 与 ``rwa/qot_checker.py`` 两份波形级/统计级
ROADM 装配的**公式级对照**与**逐位实测**。

本脚本**只读**：它不 monkeypatch 任何实现，只把两份装配各自会施加的物理量算出来，
逐项对照并报最大偏差。为 FG-17 的收敛决定提供证据（见 ``docs/fg17_waveform_assembly_convergence.md``）。

用法（物理仓 worktree 内，**必须**带 ``PYTHONPATH``）::

    PYTHONPATH=<PH>/src ~/miniconda3/envs/rl/bin/python scripts/fg17_assembly_evidence.py \
        -o results/fg17_assembly_evidence.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))

import network_nlo_eval  # noqa: E402
from network_nlo_eval.network.elements import ROADMConfig  # noqa: E402
from network_nlo_eval.rwa.qot_checker import fold_roadm_terms  # noqa: E402

#: 统计侧（``rwa/qot_checker.py``）折出的是**线性** ``loss``，波形侧
#: （``physics/qot_checker.py``）喂给物理元件的是 **dB**。这个单位差是本刀的核心。
_H_PLANCK = 6.62607015e-34


def _physics_side_roadm(config: ROADMConfig, *, is_last_hop: bool, n_fibers: int, hop_index: int) -> dict[str, Any]:
    """复刻 ``physics/qot_checker.py`` ``propagate_path`` 里 ROADM 那一段的**实际取值**。

    逐字照抄 ``:247-262`` 的表达式，包括 ``hop_index == len(fiber_configs) - 1``
    这个"末跳"判据本身——它与 ``rwa`` 侧的 ``is_last_hop`` 是两个独立写法的同一语义。
    """
    insertion_loss_db = config.path_loss_db("drop" if hop_index == n_fibers - 1 else "express")
    out: dict[str, Any] = {
        "insertion_loss_db": insertion_loss_db,
        "passband_hz": config.passband_hz,
        "filter_order": config.filter_order,
        "equalize": config.equalize_output_power,
    }
    if config.equalize_output_power:
        # 波形侧的 booster 是一台**真实 EDFA**：增益恰好补回 ROADM 损耗。
        out["booster_gain_db"] = config.path_loss_db("drop" if hop_index == n_fibers - 1 else "express")
        out["booster_nf_db"] = config.booster_noise_figure_db
    else:
        out["booster_gain_db"] = None
        out["booster_nf_db"] = None
    return out


def _rwa_side_roadm(config: ROADMConfig, *, is_last_hop: bool) -> dict[str, Any]:
    """复刻 ``rwa/qot_checker.py::fold_roadm_terms`` 的实际取值（线性域）。"""
    t = fold_roadm_terms(config, is_last_hop=is_last_hop)
    return {
        "loss_lin": t.loss,
        "filtering_penalty": t.filtering_penalty,
        "equalize": t.equalize,
        "booster_nf_lin": t.booster_nf,
        "booster_base": t.booster_base,
        "blocked": sorted(t.blocked),
        "loss_db_reconstructed": 10.0 * math.log10(t.loss),
        "booster_nf_db_reconstructed": None if t.booster_nf is None else 10.0 * math.log10(t.booster_nf),
    }


def _scenario_configs() -> list[tuple[str, ROADMConfig, bool, int, int]]:
    """覆盖 brief 点名的场景：``path_cplusl_roadm`` + ``fu4`` 全部用例 + 两个单跨段。

    每项是 ``(场景名, ROADM 配置, is_last_hop, n_fibers, hop_index)``。
    """
    from fg4a_scenarios import build_scenarios as fg4a_build
    from fu4_scenarios import ROADM_BASE, SCENARIOS as FU4_SCENARIOS

    rows: list[tuple[str, ROADMConfig, bool, int, int]] = []

    # (1) fg4a ``path_cplusl_roadm``：2 根光纤、3 跳编排，第一跳带 ROADM（**非末跳**）。
    #     该 ROADM 用的是 ROADMConfig 的**默认** booster/add-drop 值。
    rows.append(
        (
            "fg4a:path_cplusl_roadm/hop0(express)",
            ROADMConfig(insertion_loss_db=5.0, filtering_penalty_db=0.2, blocked_slots=[40, 41, 42, 43]),
            False,
            2,
            0,
        )
    )

    # (2) fu4 全部 9 个用例的 ROADM_BASE 变体，跑 express/drop 两个端口。
    for scenario in FU4_SCENARIOS:
        # ``roadm_overrides`` 是"逐节点覆盖"，合并到 ROADM_BASE 上才是该用例真正用的配置。
        roadm_kwargs = dict(ROADM_BASE)
        for override in scenario.roadm_overrides.values():
            roadm_kwargs.update(override)
        for last in (False, True):
            rows.append(
                (
                    f"fu4:{scenario.name}/{'drop' if last else 'express'}",
                    ROADMConfig(**roadm_kwargs),
                    last,
                    2,
                    1 if last else 0,
                )
            )

    # (3) 两个单跨段场景：n_fibers=1 时唯一一跳**就是**末跳（drop 端口），
    #     与 n_fibers>=2 时"非末跳走 express"的分支互补。
    rows.append(("single_span:hop0_is_last(drop)", ROADMConfig(**ROADM_BASE), True, 1, 0))
    rows.append(("single_span_express_lookalike", ROADMConfig(**ROADM_BASE), False, 1, 0))

    # (4) flexgrid_mixed 的 ROADM（若该场景带自定义配置）
    for scenario in fg4a_build():
        if scenario.name == "flexgrid_mixed":
            for last in (False, True):
                rows.append(
                    (f"fg4a:flexgrid_mixed/{'drop' if last else 'express'}", ROADMConfig(**ROADM_BASE), last, 2, 1 if last else 0)
                )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "results" / "fg17_assembly_evidence.json")
    args = parser.parse_args()

    print(f"package under test: {network_nlo_eval.__file__}")
    assert str(REPO_ROOT / "src") in network_nlo_eval.__file__, "import 落到了 worktree 之外，本轮读数作废"

    rows: list[dict[str, Any]] = []
    for name, config, is_last_hop, n_fibers, hop_index in _scenario_configs():
        phys = _physics_side_roadm(config, is_last_hop=is_last_hop, n_fibers=n_fibers, hop_index=hop_index)
        rwa = _rwa_side_roadm(config, is_last_hop=is_last_hop)

        # ---- 逐位对照 ----
        equalize_match = phys["equalize"] == rwa["equalize"]
        # 统计侧把 booster NF 门控成 None；波形侧把它门控成 "不建这台 EDFA"。
        booster_presence_match = (phys["booster_nf_db"] is None) == (rwa["booster_nf_lin"] is None)
        # 损耗：两侧共用**同一个** ROADMConfig.path_loss_db，只是一个在线性域一个在 dB 域。
        db_direct = config.path_loss_db("drop" if is_last_hop else "express")
        loss_roundtrip_exact = rwa["loss_db_reconstructed"] == db_direct
        nf_roundtrip_exact = (
            rwa["booster_nf_db_reconstructed"] == config.booster_noise_figure_db
            if rwa["booster_nf_lin"] is not None
            else None
        )

        rows.append(
            {
                "scenario": name,
                "roadm_config": config.model_dump(),
                "is_last_hop": is_last_hop,
                "n_fibers": n_fibers,
                "hop_index": hop_index,
                "physics_side": phys,
                "rwa_side": rwa,
                "verdict": {
                    "equalize_match": equalize_match,
                    "booster_presence_match": booster_presence_match,
                    "loss_db_direct": db_direct,
                    "loss_db_reconstructed_from_linear": rwa["loss_db_reconstructed"],
                    "loss_roundtrip_bit_exact": loss_roundtrip_exact,
                    "loss_roundtrip_abs_err_db": abs(rwa["loss_db_reconstructed"] - db_direct),
                    "nf_roundtrip_bit_exact": nf_roundtrip_exact,
                    "nf_roundtrip_abs_err_db": (
                        None
                        if rwa["booster_nf_db_reconstructed"] is None
                        else abs(rwa["booster_nf_db_reconstructed"] - config.booster_noise_figure_db)
                    ),
                },
            }
        )

        flag = "OK " if loss_roundtrip_exact else "ULP"
        print(
            f"  {flag} {name:<44} equalize={phys['equalize']} "
            f"loss_db={db_direct!r} rebuilt={rwa['loss_db_reconstructed']!r}"
        )

    roundtrip_fail = [r for r in rows if not r["verdict"]["loss_roundtrip_bit_exact"]]
    eq_fail = [r for r in rows if not r["verdict"]["equalize_match"]]
    pres_fail = [r for r in rows if not r["verdict"]["booster_presence_match"]]

    summary = {
        "n_scenarios": len(rows),
        "equalize_mismatches": len(eq_fail),
        "booster_presence_mismatches": len(pres_fail),
        "loss_roundtrip_non_exact": len(roundtrip_fail),
        "loss_roundtrip_non_exact_scenarios": [r["scenario"] for r in roundtrip_fail],
        "max_loss_roundtrip_abs_err_db": max((r["verdict"]["loss_roundtrip_abs_err_db"] for r in rows), default=0.0),
        "max_nf_roundtrip_abs_err_db": max(
            (r["verdict"]["nf_roundtrip_abs_err_db"] for r in rows if r["verdict"]["nf_roundtrip_abs_err_db"] is not None),
            default=0.0,
        ),
    }

    # ---- 公式级对照（写进产物，供文档直接引用）----
    formulas = {
        "rwa/qot_checker.py::fold_roadm_terms": {
            "loss": "10 ** (config.path_loss_db(operation) / 10)   [LINEAR]",
            "operation": '"drop" if is_last_hop else "express"',
            "filtering_penalty": "10 ** (config.filtering_penalty_db / 10)   [LINEAR, noise-only]",
            "equalize": "config.equalize_output_power",
            "booster_nf": "10 ** (config.booster_noise_figure_db / 10) if equalize else None",
            "booster_base": "10 ** (config.booster_noise_figure_db / 10) * h",
            "booster_ase_term": "booster_base * f * Rs * (loss - 1.0)   [statistical ASE, W]",
            "signal_when_equalize": "NOT attenuated by loss",
            "signal_when_not_equalize": "signal /= loss",
        },
        "physics/qot_checker.py::propagate_path": {
            "insertion_loss_db": 'roadm.path_loss_db("drop" if hop_index == len(fiber_configs)-1 else "express")  [dB]',
            "equalize": "roadm.equalize_output_power",
            "booster": "physical EDFA(gain_db=same loss dB, nf_db=roadm.booster_noise_figure_db)",
            "signal_when_equalize": "ROADM attenuates by loss, then a real EDFA adds gain_db=loss back",
            "signal_when_not_equalize": "ROADM attenuates by loss, no EDFA",
            "filtering_penalty_db": "NOT READ — filtering is done by the physical WSS (filter_order/passband_hz)",
            "add_drop_loss_db / express_attenuation_db": "NOT READ directly — only via path_loss_db(operation)",
        },
    }

    payload = {
        "package": network_nlo_eval.__file__,
        "h_planck": _H_PLANCK,
        "formulas": formulas,
        "summary": summary,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== FG-17 §0 summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwritten: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
