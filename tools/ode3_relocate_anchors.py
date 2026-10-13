#!/usr/bin/env python3
"""WP-O3：把 O2 `tools/ode_formula_map.py` 里的**代码锚点行号**重定位到当前源码。

为什么需要这个工具
------------------
`tools/ode_formula_map.py` 把每条物理/工程公式的代码位置记成
``{"file": ..., "line": N, "contains": "..."}``，而 ``--check`` 会断言
**第 N 行确实含有 ``contains`` 那一串**——这正是 J1「物理一字不改」的可执行守卫。

WP-O3 在 ``models/isrs_gn.py`` 里插入了 flexgrid 接线（``symbol_rate_hz_channels``、
``slots_per_channel``、``roadm_blocks_node``、``for_fiber`` 透传），把后续物理行的
行号整体下移，于是这 17 条守卫**因为行号漂移**而失败——它们指向的**物理代码内容
一个字都没变**。

本工具**只重排行号，不碰任何 ``contains`` 字符串**，因此：

* 若某条物理公式真的被改写了，``contains`` 串会匹配不上，本工具**报错退出**
  （而不是悄悄挪到别的行去掩盖改动）；
* 只有"内容仍在、只是移位"的情况才会被修正。

⚠️ ``models/numba_kernels.py``（J5 禁改）的锚点若发生漂移，本工具同样会拒绝
静默修正——那应当按 J5 上报，而不是改行号了事。

用法::

    ~/miniconda3/envs/rl/bin/python tools/ode3_relocate_anchors.py --write
    ~/miniconda3/envs/rl/bin/python tools/ode3_relocate_anchors.py --check

``--check`` 不写文件，只报告"还有多少条锚点对不上"。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TARGET = REPO / "tools" / "ode_formula_map.py"
OUT = REPO / "results" / "ode_dev_anchor_relocation.json"

#: 匹配源码里的一条锚点三元组。``contains`` 里允许出现转义引号。
TRIPLE = re.compile(
    r'(\{"file": (?P<ident>[A-Z_][A-Z_0-9]*), "line": )(?P<line>\d+)(, "contains": )(?P<contains>"(?:[^"\\]|\\.)*")\}'
)


def load_map_module():
    """按路径加载 ``tools/ode_formula_map.py``，取其路径常量与全部条目。"""
    spec = importlib.util.spec_from_file_location("ode_formula_map", TARGET)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _resolve(module, ident: str, fallback: str | None) -> str:
    value = getattr(module, ident, fallback)
    if not isinstance(value, str):
        raise SystemExit(f"无法把标识符 {ident!r} 解析成路径（值={value!r}）")
    return value


def stale_anchors(module) -> list[dict]:
    """返回所有**当前行号已对不上**的锚点及其修正建议。

    对每条失配锚点，在目标文件里找 ``contains`` 串**唯一**出现的行。
    找不到或找到多处 ⇒ 视为"物理真的被改了"，不给出建议（交由人判定）。
    """
    out: list[dict] = []
    for entry in list(module.FORMULAS) + list(module.ENGINEERING):
        for anchor in entry.get("code", []):
            path = REPO / anchor["file"]
            if not path.is_file():
                out.append({"id": entry["id"], "file": anchor["file"], "error": "file missing"})
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            old = int(anchor["line"])
            if 1 <= old <= len(lines) and anchor["contains"] in lines[old - 1]:
                continue  # 仍然对得上，不动
            hits = [i + 1 for i, line in enumerate(lines) if anchor["contains"] in line]
            item = {
                "id": entry["id"],
                "class": entry.get("class", entry.get("kind")),
                "file": anchor["file"],
                "old_line": old,
                "contains": anchor["contains"],
                "occurrences": len(hits),
            }
            if len(hits) == 1:
                item["new_line"] = hits[0]
                item["delta"] = hits[0] - old
            else:
                item["new_line"] = None
                item["error"] = "contains 在目标文件里出现 %d 次，无法唯一确定行号" % len(hits)
            out.append(item)
    return out


def apply_relocations(text: str, items: list[dict]) -> tuple[str, list[dict]]:
    """把 ``items`` 里可修正的行号写回源码文本（只改 ``"line": N`` 的数字）。"""
    by_key: dict[tuple[str, str], int] = {}
    for item in items:
        if item.get("new_line") is not None:
            by_key[(item["file"], item["contains"])] = item["new_line"]

    changes: list[dict] = []
    out_lines: list[str] = []
    for line in text.splitlines():
        match = TRIPLE.search(line)
        if match:
            ident = match.group("ident")
            contains_literal = match.group("contains")
            try:
                contains = json.loads(contains_literal)
            except json.JSONDecodeError:
                out_lines.append(line)
                continue
            rel = _resolve(_CURRENT_MODULE[0], ident, None)
            target = by_key.get((rel, contains))
            if target is not None and int(match.group("line")) != target:
                old = int(match.group("line"))
                new_line = (
                    line[: match.start("line")] + str(target) + line[match.end("line"):]
                )
                changes.append(
                    {
                        "file": rel,
                        "contains": contains,
                        "old_line": old,
                        "new_line": target,
                        "delta": target - old,
                    }
                )
                out_lines.append(new_line)
                continue
        out_lines.append(line)
    return "\n".join(out_lines) + "\n", changes


_CURRENT_MODULE: list = [None]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="改写 tools/ode_formula_map.py 的行号并落盘证据")
    ap.add_argument("--check", action="store_true", help="只报告是否还有失配锚点")
    args = ap.parse_args()

    module = load_map_module()
    _CURRENT_MODULE[0] = module
    items = stale_anchors(module)
    unfixable = [i for i in items if i.get("new_line") is None]

    if args.write:
        text = TARGET.read_text(encoding="utf-8")
        new_text, changes = apply_relocations(text, items)
        TARGET.write_text(new_text, encoding="utf-8")
        OUT.parent.mkdir(parents=True, exist_ok=True)

        # ⚠️ 证据必须**累计**而不是被最后一次运行覆盖。本工具是幂等的：第二次
        # ``--write`` 会重定位 0 条。若每次都覆盖落盘，守卫 8（"应当有锚点被重定位"）
        # 就会因为一次无关的重复运行而假失败。WP-Q1 因此改为：保留历史条目，
        # 并按当前源码把每条的 ``new_line`` 刷新成真值。
        history: dict[tuple[str, str], dict] = {}
        if OUT.is_file():
            try:
                prior = json.loads(OUT.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prior = {}
            for item in prior.get("relocations", []):
                history[(item["file"], item["contains"])] = item
        for item in changes:
            history[(item["file"], item["contains"])] = item

        # 刷新每条历史锚点当前真实所在的行号（防止源码再次漂移后证据说谎）。
        # 若 ``contains`` 已不在源码里（= 有人**刻意改写**了那一行，而不是挪了它），
        # 它就不再是"同一条锚点的重定位"，移入 superseded 透明留档，而不是留一条
        # new_line=None 的坏记录把守卫 8 打挂。
        merged: list[dict] = []
        superseded: list[dict] = []
        for (file_rel, contains), item in history.items():
            path = REPO / file_rel
            if not path.is_file():
                superseded.append({**item, "reason": "target file missing"})
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
            hits = [i + 1 for i, line in enumerate(lines) if contains in line]
            if len(hits) != 1:
                superseded.append({
                    **item,
                    "reason": (
                        "contains 串已不在源码中（锚点内容被刻意改写）"
                        if not hits else f"contains 在目标文件里出现 {len(hits)} 次，不再唯一"
                    ),
                })
                continue
            merged.append({**item, "new_line": hits[0]})

        OUT.write_text(
            json.dumps(
                {
                    "wp": "O3+Q1",
                    "purpose": "接线插入导致物理行行号下移；只重排行号，不改 contains。"
                    "证据为累计口径（幂等重跑不会把它抹成 0）。",
                    "target": str(TARGET.relative_to(REPO)),
                    "n_relocated": len(merged),
                    "n_relocated_this_run": len(changes),
                    "n_unfixable": len(unfixable),
                    "relocations": merged,
                    "superseded": superseded,
                    "unfixable": unfixable,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"重定位 {len(changes)} 条锚点 -> {TARGET.relative_to(REPO)}")
        print(f"证据 -> {OUT.relative_to(REPO)}")
        for c in changes:
            print(f"  {c['file']}:{c['old_line']} -> {c['new_line']}  ({c['delta']:+d})")
        for u in unfixable:
            print(f"  [需人工判定] {u['id']}: {u.get('error')}")
        # 重载并复算
        module2 = load_map_module()
        _CURRENT_MODULE[0] = module2
        remaining = stale_anchors(module2)
        if remaining:
            print(f"FAIL: 仍有 {len(remaining)} 条失配")
            return 1
        print("OK: 全部锚点重新对齐")
        return 0

    if args.check:
        if items:
            print(f"FAIL: {len(items)} 条锚点失配（{len(unfixable)} 条无法自动修正）")
            for i in items:
                print(f"  {i['id']}: {i['file']}:{i['old_line']} occurrences={i['occurrences']}")
            return 1
        print("OK: 无失配锚点")
        return 0

    print(json.dumps({"n_stale": len(items), "n_unfixable": len(unfixable)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
