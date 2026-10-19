r"""FG-15 §4: the ``legacy`` tier must survive the default flip, bit-for-bit.

Why this file exists
--------------------
FG-15 flips the global default Raman tier from ``legacy`` to ``tabular`` because
the synthetic legacy spectrum truncates to zero past 15.6 THz, which forces the
S/C coupling of a ~27 THz S+C+L+U grid to zero. That is the right physics for this
program. But a default flip that *deletes* the old tier would destroy the only
guarantee that older results remain reproducible — and two existing things depend
on that: FG-1's ``(A)`` kernel-parity assertion against ``results/fg1_golden_main.json``
(which declares ``source_branch: origin/main``) and the FG-4a numeric-identity gate.

So the flip must be **additive in reach, subtractive in default**:

* ``legacy`` is still explicitly selectable and produces **bit-identical** results
  to what it produced before the flip;
* the default is now ``tabular``.

The "bit-identical" half is pinned against ``results/fg4a_golden.json``, which was
captured at the pre-FG-4a baseline commit ``ea7ecb0`` — i.e. **before this flip**.
No new expected values are introduced here. If this test ever fails, the correct
response is to fix the flip, never to re-pin the numbers.

Every assertion is recomputable inside this worktree::

    PYTHONPATH=$PWD/src $HOME/miniconda3/envs/rl/bin/python \\
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
    NLI_MODEL_DEFAULT,
    RAMAN_SPECTRUM_DEFAULT,
    RAMAN_SPECTRUM_DEFAULT_BY_NLI_MODEL,
    RAMAN_SPECTRUM_MODES,
    MultiBandISRSGN,
)
from network_nlo_eval.network.elements import FiberSpanConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "results" / "fg4a_golden.json"

#: Modes the FG-4a gate requires to be bit-identical to the golden (F4a-3).
BIT_EXACT_MODES = ("off", "cached_g")


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
    assert "legacy" in RAMAN_SPECTRUM_MODES, "FG-15 §4: the legacy tier must not be deleted"
    assert "tabular" in RAMAN_SPECTRUM_MODES

    ev = MultiBandISRSGN(
        grid=SpectrumGrid(num_channels=40, channel_spacing_hz=50e9, center_frequency_hz=190e12),
        ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        symbol_rate_hz=50e9,
        raman_spectrum="legacy",
        nli_model="p2_gn",
    )
    # Read back the *effective* tier rather than trusting the argument.
    assert ev.raman_spectrum == "legacy"


def test_default_tier_is_now_tabular() -> None:
    """The default must be ``tabular``, verified by read-back, not by intent.

    The read-back is deliberate: FG-12's lesson is that asserting a tier from the
    fact that an argument was *passed* proves nothing about what is effective.

    ⚠️ WP-MERGE-FG-FLEX **第 3 轮**：拉曼谱档位的默认值现在是**按 NLI 物理档位解析**的
    ——``p2_gn`` 档默认 ``"tabular"``（FG-15 的裁决不变），``p1_closed`` 档默认
    ``"legacy"``（P1 核只有 legacy 折线那条路）。常量 ``RAMAN_SPECTRUM_DEFAULT``
    本身**一个字没动**，仍是 ``"tabular"``。
    """
    assert RAMAN_SPECTRUM_DEFAULT == "tabular", "FG-15: the global default Raman tier must be tabular"
    assert RAMAN_SPECTRUM_DEFAULT_BY_NLI_MODEL["p2_gn"] == "tabular"
    assert RAMAN_SPECTRUM_DEFAULT_BY_NLI_MODEL["p1_closed"] == "legacy"

    ev = MultiBandISRSGN(
        grid=SpectrumGrid(num_channels=40, channel_spacing_hz=50e9, center_frequency_hz=190e12),
        ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        symbol_rate_hz=50e9,
        nli_model="p2_gn",
    )
    assert ev.raman_spectrum == "tabular", "default-constructed evaluator must actually run tabular"
    # ⚠️ 第 3 轮：构造签名的**最后一个**默认参数现在是 ``nli_model``（不是
    # ``raman_spectrum``）⇒ ``__defaults__[-1]`` 不再是拉曼档位。改成断言
    # 「档位默认由 NLI 档位解析」这件事本身。
    assert MultiBandISRSGN.__init__.__defaults__[-1] == NLI_MODEL_DEFAULT
    assert MultiBandISRSGN.__init__.__defaults__[-1] == "p1_closed"


def test_tabular_and_legacy_are_distinguishable() -> None:
    """The flip must have teeth: the two tiers must not be the same computation."""
    ev = MultiBandISRSGN(
        grid=SpectrumGrid(num_channels=80, channel_spacing_hz=50e9, center_frequency_hz=190.2127e12),
        ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        symbol_rate_hz=50e9,
        raman_spectrum="tabular",
        nli_model="p2_gn",
    )
    ev_legacy = MultiBandISRSGN(
        grid=SpectrumGrid(num_channels=80, channel_spacing_hz=50e9, center_frequency_hz=190.2127e12),
        ref_fiber_config=FiberSpanConfig(length_km=100.0, max_span_length_km=100.0),
        symbol_rate_hz=50e9,
        raman_spectrum="legacy",
        nli_model="p2_gn",
    )
    assert not np.array_equal(ev._g_gain, ev_legacy._g_gain), (
        "tabular and legacy produced identical G matrices; the flip would be a no-op"
    )


# ======================================================================
# The load-bearing §4 gate: legacy is unchanged, bit for bit
# ======================================================================
@pytest.mark.parametrize("ode_accel", BIT_EXACT_MODES)
@pytest.mark.parametrize("scenario_name,key", _golden_cases(), ids=lambda v: str(v))
def test_legacy_still_reproduces_pre_flip_golden_bitwise(ode_accel: str, scenario_name: str, key: str) -> None:
    """Explicit ``legacy`` must equal the pre-FG-4a golden **exactly**.

    The golden was captured at commit ``ea7ecb0``, before FG-15 touched the default.
    Equality is bitwise (``assert_array_equal``), not tolerance-based: a tier flip
    that silently perturbed ``legacy`` would invalidate every historical reading
    that named it.
    """
    expected = _golden_array(scenario_name, key)
    actual = _legacy_array(scenario_name, key, ode_accel)
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            f"legacy tier moved at ode_accel={ode_accel!r} scenario={scenario_name!r} key={key!r}. "
            "FG-15 must not perturb legacy; fix the flip, do not re-pin this expectation."
        ),
    )


def test_legacy_golden_pin_survives_for_every_declared_scenario() -> None:
    """Every golden scenario must be reachable under an explicit ``legacy`` pin."""
    declared = {s.name for s in build_scenarios("off", raman_spectrum="legacy")}
    stored = set(GOLDEN.get("scenarios", {}))
    assert stored == declared, (
        f"golden scenarios {sorted(stored)} != declared {sorted(declared)}; "
        "the FG-15 legacy pin must cover exactly what the golden covers"
    )