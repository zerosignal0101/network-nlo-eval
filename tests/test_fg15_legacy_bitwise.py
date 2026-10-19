r"""FG-15 §4 的**遗留部分**：``legacy`` 拉曼谱档必须始终显式可选。

⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本模块是「拉曼档默认翻转」那条裁决的守卫。
那条裁决本身（把全局默认从 ``legacy`` 翻到 ``"tabular"``）所服务的物理——
宽波段谱驱动拉曼 ODE 计算——**只存在于被移除的那套物理里**。闭式 ECZ 的冻结核
``numba_kernels.py`` 内部走的是 legacy 那条合成折线（``_calc_raman_profile_jit``），
**结构上没有第二条路**。因此：

* 评估器现在只暴露 ``RAMAN_SPECTRUM_MODES = ("legacy",)`` 一档，
  ``RAMAN_SPECTRUM_DEFAULT == "legacy"``；
* 谱**本身**（``physics/raman_spectrum.py``：Lin & Agrawal 2006 响应函数的傅里叶
  变换闭式增益谱 + 出处 CSV + 互校）**保留在树里**，由它自己的测试守着——
  移除的只是**评估器不再暴露该档**，不是删掉那份物理谱。

本文件保留的那一条判据是本模块**唯一**与被移除物理无关的命题：
「``legacy`` 档不许被删、且必须可显式选中（读回生效值而不是相信入参）」。

原模块的其余 4 条判据已撤下，**逐条理由见文件末的迁移说明**。

Every assertion is recomputable inside this worktree::

    PYTHONPATH=$PWD/src $HOME/miniconda3/envs/rl/bin/python \
        -m pytest tests/test_fg15_legacy_bitwise.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fg4a_scenarios import PATH_KEYS, SPAN_KEYS, build_scenarios

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.isrs_gn import (
    RAMAN_SPECTRUM_DEFAULT,
    RAMAN_SPECTRUM_MODES,
    MultiBandISRSGN,
)
from network_nlo_eval.network.elements import FiberSpanConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "results" / "fg4a_golden.json"

#: ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：原为 ``("off", "cached_g")``。``cached_g``
#: （拉曼 G 矩阵缓存）随被移除的物理**整体**离场；本模块下面依赖 FG 金标准的两条
#: 判据**同时撤下**（理由见文件末的迁移说明），故此处不再有批量模式可枚举。
BIT_EXACT_MODES = ("off",)


def _load_golden() -> dict[str, Any]:
    if not GOLDEN_PATH.exists():  # pragma: no cover - fails loudly, by design
        pytest.fail(
            f"{GOLDEN_PATH} is missing; FG-15 §4 is pinned against the pre-FG-4a golden "
            "and cannot fall back to freshly computed expectations."
        )
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


GOLDEN = _load_golden()


def _expected_keys(scenario: dict[str, Any]) -> tuple[str, ...]:
    return SPAN_KEYS if scenario["kind"] == "span" else PATH_KEYS


def _golden_cases() -> list[tuple[str, str]]:
    return [(name, key) for name, entry in GOLDEN.get("scenarios", {}).items() for key in _expected_keys(entry)]


def _legacy_array(scenario_name: str, key: str, ode_accel: str) -> np.ndarray:
    scenario = next(s for s in build_scenarios(ode_accel, raman_spectrum="legacy") if s.name == scenario_name)
    return np.asarray(scenario.run()[key], dtype=np.float64)


def _golden_array(scenario_name: str, key: str) -> np.ndarray:
    return np.asarray(GOLDEN["scenarios"][scenario_name]["outputs"][key], dtype=np.float64)


# ======================================================================
# The tier is still reachable, and the default really moved
# ======================================================================
def test_legacy_tier_still_exists_and_is_explicitly_selectable() -> None:
    """``legacy`` must remain in the mode tuple and construct on demand."""
    # ⚠️ 第 5 轮：原此处还断言 ``"tabular" in RAMAN_SPECTRUM_MODES``。``"tabular"``
    # 只经由被移除那套物理的拉曼 ODE 进入计算，故随该物理**整体**撤下评估器暴露面
    # （谱本身仍在 ``physics/raman_spectrum.py``，见该模块自己的出处与互校测试）。
    # 「legacy 档不许被删」这条命题**原样保留**。
    assert "legacy" in RAMAN_SPECTRUM_MODES, "FG-15 §4: the legacy tier must not be deleted"

    ev = MultiBandISRSGN(
        grid=SpectrumGrid(num_channels=40, channel_spacing_hz=50e9, center_frequency_hz=190e12),
        ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        symbol_rate_hz=50e9,
        raman_spectrum="legacy",
    )
    # Read back the *effective* tier rather than trusting the argument.
    assert ev.raman_spectrum == "legacy"


# ---------------------------------------------------------------------------
# ⚠️ WP-MERGE-FG-FLEX **第 5 轮**：本模块原有的其余 4 条判据**全部撤下**。
#
# 逐条理由（文件 + 测试名 + 一行理由）：
#
# 1. ``test_default_tier_is_now_tabular``
#    —— 断言「拉曼谱默认档是 ``"tabular"``」并读回生效值。``"tabular"`` 档只经由
#    被移除那套物理的拉曼 ODE 进入计算，闭式 ECZ 的冻结核内部只走 legacy 折线，
#    结构上没有第二条路 ⇒ **被测档位不再可达**。
# 2. ``test_tabular_and_legacy_are_distinguishable``
#    —— 断言「两档必须给出不同结果」（翻默认档必须「有牙齿」）。同上，``"tabular"``
#    在评估器上已不存在，**被测对象消失**。
# 3. ``test_legacy_still_reproduces_pre_flip_golden_bitwise``
#    —— 逐字节复现 ``results/fg4a_golden.json``。该金标准是**在 FG 树里、于本物理
#    被换底之前**捕获的，其数值属于被移除那条线；把它当作闭式 ECZ 的期望值，等于
#    拿两套物理的读数互相比较——**这正是第 3 轮判废的那类错误比较**。
# 4. ``test_legacy_golden_pin_survives_for_every_declared_scenario``
#    —— 断言「golden 覆盖的场景集合 == 显式 legacy 钉子声明的场景集合」。依赖同
#    一份金标准，**同样失去意义**。
#
# **没有削弱的地方**：闭式 ECZ 的「默认加速档 vs 冻结参考档」逐位/预算等同性，
# 由 ``tests/test_p1_closed_accel.py`` 以**本轮实测**的判据接管
# （``off``/``cached`` 逐位、``matvec`` max_rel 1.269e-15 ≤ 1e-12），
# 比这四条**更强**：它们比的是本仓物理自己的参考档，不是另一条线的历史快照。
#
# ``results/fg4a_golden.json`` **保留在盘上**（历史证据，不删），但本轮起
# **没有任何测试再断言它**。
# ---------------------------------------------------------------------------
