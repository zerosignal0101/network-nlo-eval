"""收敛 DRA 波形参考的元数据检查。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "dra_waveform_converged"
REFERENCES = ("dra_reference.json", "dra_reference_depleted.json")
MAX_STEP_M = 62.5


@pytest.mark.parametrize("name", REFERENCES)
def test_converged_reference_metadata(name: str) -> None:
    """每个数据集必须步长收敛、步长 <= 62.5 m，并声明泵浦口径。"""
    reference = json.loads((DATA_DIR / name).read_text(encoding="utf-8"))
    assert reference["pump_model"] in {"undepleted", "depleted"}
    assert reference["coarse_step_m"] <= MAX_STEP_M
    for tag, dataset in reference["datasets"].items():
        assert dataset["pump_model"] == reference["pump_model"], tag
        assert dataset["step_converged"] is True, tag
        assert dataset["coarse_step_m"] <= MAX_STEP_M, tag


def test_reference_pump_models_are_distinct() -> None:
    """两个收敛文件必须覆盖未耗尽与耗尽两种口径。"""
    models = {
        json.loads((DATA_DIR / name).read_text(encoding="utf-8"))["pump_model"]
        for name in REFERENCES
    }
    assert models == {"undepleted", "depleted"}
