r"""FG-18 §2：把 FU4-6 的源码级守卫从**单文件**扩到**整个 ``src/`` 树**。

背景
----
FU-4 的 ``test_fu4_6_lumped_assembly_is_gone_from_the_source`` 只读
``rwa/qot_checker.py`` **一个文件**。FG-17 因此发现盲区：``models/isrs_gn.py``
里的两个施加点仍在做 FG 树的合并式装配
（``insertion_loss_db + filtering_penalty_db``），而那条守卫扫不到它。

本文件补上这一层。**不放宽**既有的单文件断言：``test_fu4_assembly_union.py``
原样保留，本文件是**新增**的全树覆盖。

为什么用 AST 而不是正则
--------------------
正则 ``insertion_loss_db\\s*\\+\\s*\\w+\\.filtering_penalty_db`` 在扩到全树后会
**误伤合法写法**——FG-18 自己在 ``isrs_gn.py`` 的两处注释里就写了这段字面量
（说明"不再在本处合并…"），正则会把注释和文档字符串当成回归。踩到自己的注释
正是把断言改宽/改脆的起点。

因此这里按 **AST** 判定：只匹配真正参与加法的 ``ast.BinOp(op=Add)``，且两侧
分别是 ``.insertion_loss_db`` 与 ``.filtering_penalty_db`` 的属性访问。注释、
文档字符串、纯字符串常量在 AST 里根本不出现，天然不误伤；真正把合并式装配
塞回代码的变异则必然被抓住。

反向：不靠"白名单"排除任何 ``src/`` 文件。``rwa/qot_checker.py`` 同样被扫到，
FG-2 的端口级实现 ``path_loss_db`` 走的是 ``config.path_loss_db(...)`` 调用而非
属性相加，故不误报。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"

#: 源码根不存在时直接失败，而不是静默扫过 0 个文件变成假绿。
assert SRC_ROOT.is_dir(), f"missing source root: {SRC_ROOT}"


def _python_sources() -> list[Path]:
    return sorted(p for p in SRC_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _attr_name(node: ast.expr) -> str | None:
    """``a.b.c`` -> ``"c"``；其它形状返回 None。"""
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _lumped_assembly_nodes(tree: ast.AST) -> list[ast.BinOp]:
    """返回形如 ``X.insertion_loss_db + Y.filtering_penalty_db`` 的加法节点。

    两个操作数可以互换顺序（``filtering_penalty_db`` 在左也算），因为这是同一个
    物理误用的两种写法。只认直接属性，避免把 ``sum([...])`` 之类无关结构拖进来。
    """
    found: list[ast.BinOp] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
            continue
        left_name = _attr_name(node.left)
        right_name = _attr_name(node.right)
        if {left_name, right_name} == {"insertion_loss_db", "filtering_penalty_db"}:
            found.append(node)
    return found


def test_src_tree_has_lumped_roadm_assembly() -> None:
    """全 ``src/`` 树不得再出现合并式 ROADM 装配。

    FG-18 之前，这条守卫只覆盖 ``rwa/qot_checker.py``；``models/isrs_gn.py`` 的两处
    （逐跨段入口与 batch 入口）因此长期逃过源码级检查。
    """
    offenders: list[str] = []
    sources = _python_sources()
    assert sources, f"no python sources found under {SRC_ROOT}"

    for path in sources:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in _lumped_assembly_nodes(tree):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")

    assert not offenders, (
        "FG-tree lumped ROADM assembly (``insertion_loss_db + "
        "filtering_penalty_db``) reintroduced; use "
        "rwa.qot_checker.fold_roadm_terms instead:\n  " + "\n  ".join(sorted(offenders))
    )


REGRESSION_SHAPES = [
    # 真实回归的形状（FG-17 在 isrs_gn.py 观察到的两处）。
    "loss_lin = _db_to_lin(cfg.insertion_loss_db + cfg.filtering_penalty_db)\n",
    # 换接收者、换顺序、换空白，都必须仍然被抓。
    "x = other.insertion_loss_db + another.filtering_penalty_db\n",
    "x = cfg.filtering_penalty_db + cfg.insertion_loss_db\n",
    "x = cfg.insertion_loss_db+cfg.filtering_penalty_db\n",
    "x = (cfg.insertion_loss_db\n     + cfg.filtering_penalty_db)\n",
    # 嵌在更深表达式里。
    "p = p / (10 ** ((cfg.insertion_loss_db + cfg.filtering_penalty_db) / 10))\n",
    # 批量入口的形状（FG-17 的 batch 施加点）。
    "p_out = p_out / _db_to_lin(rc.insertion_loss_db + rc.filtering_penalty_db)\n",
]

LEGAL_SHAPES = [
    # 注释里的字面量：FG-18 自己在 isrs_gn.py 写了两条这种注释，必须不误伤。
    "# no longer lump insertion_loss_db + filtering_penalty_db here\n",
    '"""docstring mentioning insertion_loss_db + filtering_penalty_db"""\n',
    # 权威口径：端口级损耗走方法调用，不是属性相加。
    "loss = 10 ** (config.path_loss_db(operation) / 10)\n",
    "penalty = 10 ** (config.filtering_penalty_db / 10)\n",
    # 折叠内部按字段分别取用，不相加。
    "a = config.insertion_loss_db\nb = config.filtering_penalty_db\n",
    # 减法不是本守卫要拦的合并式。
    "x = cfg.insertion_loss_db - cfg.filtering_penalty_db\n",
    # path_loss_db 自身确实是加法，但那是**合法**的口径实现：
    # 它加的是 insertion_loss + port_loss，不是 filtering_penalty。
    "return self.insertion_loss_db + port_loss\n",
]


@pytest.mark.parametrize("snippet", REGRESSION_SHAPES)
def test_detector_catches_regression_shapes(snippet: str) -> None:
    """守卫自身的识别能力：真实回归形状必须被判红。

    这是"断言改宽到抓不住真回归"的反面自检——若检测器对上面任一形状失灵，
    全树守卫就只是形式上存在。
    """
    assert _lumped_assembly_nodes(ast.parse(snippet)), (
        f"detector failed to flag a real lumped-assembly shape:\n{snippet}"
    )


@pytest.mark.parametrize("snippet", LEGAL_SHAPES)
def test_detector_does_not_fire_on_legal_forms(snippet: str) -> None:
    """合法写法不得被判红——否则守卫会被"改宽"到失去意义。"""
    assert not _lumped_assembly_nodes(ast.parse(snippet)), (
        f"detector fired on a legal form (over-broad):\n{snippet}"
    )


def test_guard_covers_the_fg17_blind_spot_file() -> None:
    """显式钉住 FG-17 的盲区文件本身。

    全树扫描若因路径写错而退化成"扫不到东西"，上面的参数化测试仍会绿（它们喂的是
    字符串，不是文件）。这里读**真实文件**，保证 ``models/isrs_gn.py`` 确实在扫描
    范围内，且它已不再含合并式装配。
    """
    target = SRC_ROOT / "network_nlo_eval" / "models" / "isrs_gn.py"
    assert target.is_file(), f"missing: {target}"
    assert target in _python_sources(), "isrs_gn.py is not covered by the tree-wide guard"

    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    assert not _lumped_assembly_nodes(tree), "isrs_gn.py still has lumped ROADM assembly"


def test_original_single_file_guard_still_applies() -> None:
    """既有的单文件守卫未被放宽：``rwa/qot_checker.py`` 仍必须单独干净。

    ``test_fu4_assembly_union.py::test_fu4_6_*`` 一字未改；这里只是把"它还在生效"
    也钉进新文件，避免将来有人删掉旧断言而不被察觉。
    """
    target = SRC_ROOT / "network_nlo_eval" / "rwa" / "qot_checker.py"
    assert target.is_file(), f"missing: {target}"
    tree = ast.parse(target.read_text(encoding="utf-8"), filename=str(target))
    assert not _lumped_assembly_nodes(tree), "rwa/qot_checker.py regressed to lumped assembly"
