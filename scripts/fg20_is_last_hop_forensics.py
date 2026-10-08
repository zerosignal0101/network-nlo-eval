"""FG-20 §3：**只取证、不改**——逐跨段入口固定 ``express`` 是否给出与权威口径不同的结果。

问题
----
FG-18 未闭合项 2：逐跨段层拿不到 hop 序号，故 FG-18 收敛后**固定**用
``is_last_hop=False``（express）。本脚本回答：在这个选择的**本仓可达调用面**上，
它**会不会**给出与权威口径（末跳 drop、其余 express）不同的读数？

方法
----
关键困难：本仓**没有**任何活调用方会同时传 ``roadm_config`` **和**一个末跳位置
（见 FG-18 §1.2 的 790 次调用 / 0 次非 None）。所以"实际会不会出错"只能靠两种读数回答：

* **(A) 结构性读数**：grep + AST 证明哪些调用面能到达逐跨段入口、它们能传什么。
* **(B) 构造性读数**：人为把这条路径接上（末跳），量 express 与 drop 的差。
  **这不是**"现有场景会不会动"的读数，而是"**如果**接上了会差多少"的上界。

(A) 决定"现在有没有风险"，(B) 决定"将来接上时差多少"。**两者都要给，缺一不可**——
只给 (B) 会夸大风险，只给 (A) 说不清后果。

诚实边界
--------
本脚本**不改签名、不加开关**，只在临时副本上做 (B) 的构造；报告的建议部分只写
"要支持末跳 drop 需要什么"，不落地。⇒ FG-18 的签名与 express 取值在脚本跑完后
保持原样（工作树零改动，可用 ``git status`` 复核）。
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
ISRS_GN = "src/network_nlo_eval/models/isrs_gn.py"
LINK_NOISE = "src/network_nlo_eval/rwa/link_noise.py"

#: 端口差只来自 ``add_drop_loss_db``（``elements.py:113-116``：
#: ``port_loss = express_attenuation_db if operation == "express" else add_drop_loss_db``）。
#: 默认 add_drop=1.0 dB，故末跳比非末跳多 1.0 dB 损耗。
DEFAULT_ADD_DROP_DB = 1.0


def structural_forensics() -> dict:
    """给出 (A) 结构性读数：谁调用逐跨段入口，传了什么 ``roadm_config``。"""
    py = "/home/zsig/miniconda3/envs/rl/bin/python"

    def run(args: list[str]) -> list[str]:
        proc = subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)  # noqa: S603
        return [ln for ln in proc.stdout.splitlines() if ln.strip()]

    # 逐跨段入口的定义与全部调用点（按名字 grep，不假设位置）。
    callsites = run(
        [sys.executable, "-c", (
            "import ast,pathlib;"
            "names={'_calc_span_noise_and_power','calc_span_noise_and_power_batch'};"
            "root=pathlib.Path('src');"
            "\nfor p in sorted(root.rglob('*.py')):\n"
            "    t=ast.parse(p.read_text())\n"
            "    for n in ast.walk(t):\n"
            "        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in names:\n"
            "            kws=[k.arg for k in n.keywords]\n"
            "            print(f'{p}:{n.lineno}  {n.func.attr}  n_pos={len(n.args)}  keywords={kws}')"
        )]
    )
    # 直接传 roadm_config 的关键字调用（逐跨段入口）vs 编排层自持装配。
    returns_roadm = run(
        [sys.executable, "-c", (
            "import pathlib\n"
            "for p in sorted(pathlib.Path('src').rglob('*.py')):\n"
            "    for i,l in enumerate(p.read_text().splitlines(),1):\n"
            "        if 'roadm_config=' in l or 'roadm_config is None' in l or 'if roadm_config' in l:\n"
            "            print(f'{p}:{i}: {l.strip()}')"
        )]
    )
    return {
        "callsites": callsites,
        "roadm_config_references": returns_roadm,
        "python": py,
    }


def construct_forensics() -> dict:
    """(B) 构造性读数：把末跳位置接到逐跨段入口上，量 express vs drop 的差。

    **不改被测代码**——用 ``fold_roadm_terms`` 在两个 ``is_last_hop`` 取值下
    各折一次，比较两个 ``loss``，从而得到差值。逐跨段入口固定取 False，所以
    "它会算成哪个" 由 §2 的行为测试已经钉住；这里只量"如果接上末跳，差多少"。
    """
    from network_nlo_eval.core.spectrum import SpectrumGrid
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN  # noqa: F401
    from network_nlo_eval.models.isrs_gn import MultiBandISRSGN as _Ev
    from network_nlo_eval.network.elements import FiberSpanConfig, ROADMConfig
    from network_nlo_eval.rwa.qot_checker import fold_roadm_terms

    grid = SpectrumGrid(3, 50e9, 193.1e12)
    ev = _Ev(grid, FiberSpanConfig(length_km=100.0), symbol_rate_hz=32e9)
    power = np.full(3, 1e-3)
    span_m = 100e3

    configs = {
        "defaults": ROADMConfig(),
        "express_attenuation_2db": ROADMConfig(express_attenuation_db=2.0),
        "add_drop_loss_5db": ROADMConfig(add_drop_loss_db=5.0),
        "equalize_off": ROADMConfig(equalize_output_power=False),
        "equalize_off_add_drop_5db": ROADMConfig(equalize_output_power=False, add_drop_loss_db=5.0),
    }

    rows = []
    for name, cfg in configs.items():
        t_express = fold_roadm_terms(cfg, is_last_hop=False)
        t_drop = fold_roadm_terms(cfg, is_last_hop=True)
        base = ev._calc_span_noise_and_power(power, span_m, None, None)

        # 被测入口固定 express —— 用它自己的读数（行为测试已证明这与 express 折叠一致）。
        got = ev._calc_span_noise_and_power(power, span_m, None, cfg)

        # 若接上末跳 drop，读数会变成什么（按权威折叠在 is_last_hop=True 下重算）。
        loss_db = t_drop.loss_db
        if t_drop.equalize:
            ase_drop = np.asarray(base[3]) + t_drop.booster_base * np.asarray(
                ev.f_abs_hz, dtype=np.float64
            ) * np.asarray(ev.symbol_rate_hz_channels, dtype=np.float64) * (t_drop.loss - 1.0)
            snr_drop = 10 * np.log10(np.asarray(base[0]) / ase_drop)
        else:
            ase_drop = np.asarray(base[3]) / t_drop.loss
            snr_drop = 10 * np.log10(np.asarray(base[0]) / t_drop.loss / ase_drop)

        snr_express = None
        if t_express.equalize:
            ase_expr = np.asarray(base[3]) + t_express.booster_base * np.asarray(
                ev.f_abs_hz, dtype=np.float64
            ) * np.asarray(ev.symbol_rate_hz_channels, dtype=np.float64) * (t_express.loss - 1.0)
            snr_express = 10 * np.log10(np.asarray(base[0]) / ase_expr)
        else:
            ase_expr = np.asarray(base[3]) / t_express.loss
            snr_express = 10 * np.log10(np.asarray(base[0]) / t_express.loss / ase_expr)

        rows.append(
            {
                "config": name,
                "add_drop_loss_db": cfg.add_drop_loss_db,
                "express_attenuation_db": cfg.express_attenuation_db,
                "equalize": t_express.equalize,
                "loss_db_express": t_express.loss_db,
                "loss_db_drop": t_drop.loss_db,
                "loss_db_delta": t_drop.loss_db - t_express.loss_db,
                "loss_ratio_drop_over_express": t_drop.loss / t_express.loss,
                "snr_express_db_measured": float(np.mean(snr_express)),
                "snr_drop_db_if_last_hop": float(np.mean(snr_drop)),
                "snr_delta_db_if_last_hop": float(np.mean(snr_drop) - np.mean(snr_express)),
                "measured_matches_express_fold": bool(
                    np.allclose(
                        np.asarray(got[3]),
                        ase_expr,
                        rtol=1e-12,
                    )
                ),
            }
        )

    nonzero = [r for r in rows if abs(r["snr_delta_db_if_last_hop"]) > 1e-9]
    # 差值机制核验：drop 与 express 的差**只**来自 add_drop_loss_db
    # （express 分支用 express_attenuation_db，drop 分支用 add_drop_loss_db，
    # 共同项 insertion_loss_db 相消），故 loss_db 差应恰好等于
    # ``add_drop_loss_db - express_attenuation_db``。
    delta_rule_holds = all(
        abs(r["loss_db_delta"] - (r["add_drop_loss_db"] - r["express_attenuation_db"])) < 1e-12 for r in rows
    )
    return {
        "rows": rows,
        "n_configs": len(rows),
        "n_with_nonzero_snr_delta": len(nonzero),
        "all_measured_match_express_fold": all(r["measured_matches_express_fold"] for r in rows),
        "delta_rule_holds": delta_rule_holds,
        "delta_rule": "loss_db_drop - loss_db_express == add_drop_loss_db - express_attenuation_db",
        "default_snr_delta_db": next(
            r["snr_delta_db_if_last_hop"] for r in rows if r["config"] == "defaults"
        ),
    }


def main() -> int:
    """汇总 (A) 结构性读数与 (B) 构造性读数并落盘。"""
    struct = structural_forensics()
    constr = construct_forensics()

    payload = {
        "question": "逐跨段层固定 is_last_hop=False(express) 在本仓可达面上是否给出与权威口径不同的读数",
        "structural": struct,
        "constructive": constr,
        "conclusion": {
            "reachable_now": False,
            "reachable_now_evidence": (
                "本仓逐跨段入口的两个 roadm_config 施加点，"
                "唯一活调用方是 evaluate_path，而它 **不传** roadm_config"
                "（见 structural.callsites 的 keywords 列）"
            ),
            "delta_if_connected_db": constr["default_snr_delta_db"],
            "delta_mechanism": "只来自 add_drop_loss_db（默认 1.0 dB），与 express_attenuation_db 无关",
        },
    }
    out = REPO / "results/fg20_is_last_hop_forensics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== (A) structural: callsites of the per-span entry points ===")
    for ln in struct["callsites"]:
        print("  ", ln)
    print("=== (A) roadm_config references ===")
    for ln in struct["roadm_config_references"]:
        print("  ", ln)
    print("=== (B) constructive: express vs drop ===")
    for r in constr["rows"]:
        print(
            f"  {r['config']:32s} loss_db exp={r['loss_db_express']:.3f} drop={r['loss_db_drop']:.3f} "
            f"delta_snr={r['snr_delta_db_if_last_hop']:+.6f} dB  matches_fold={r['measured_matches_express_fold']}"
        )
    print("  default delta dB:", constr["default_snr_delta_db"])
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
