"""独立 DRA 数值参考的最小验收测试。"""

from __future__ import annotations

import numpy as np
import pytest

from network_nlo_eval.benchmarks.cases import build_case, build_path, launch_profile
from network_nlo_eval.benchmarks.runner import make_model
from network_nlo_eval.models.base import MultiSpanOpticalPath, get_model_info
from network_nlo_eval.network.elements import RamanPumpConfig

CASE = build_case("c_57ch")


def _pump(direction: str = "backward") -> RamanPumpConfig:
    return RamanPumpConfig(
        wavelength_nm=1450.0,
        power_w=0.2,
        direction=direction,  # type: ignore[arg-type]
        attenuation_db_per_km=0.25,
    )


def _one_span(case):
    path = build_path(case)
    return MultiSpanOpticalPath(path.fiber_configs[:1], path.edfa_configs[:1])


def test_model_is_registered_as_a_numerical_reference() -> None:
    """The reference must be registered as a numerical model."""
    info = get_model_info("numerical_dra")
    assert info.name == "numerical_dra"
    assert "numerical" in info.summary.lower()
    assert "closed" not in info.summary.lower()


def test_no_pump_matches_the_paper_baseline() -> None:
    """无泵时独立参考必须与论文闭式基线一致，否则参考本身有问题。"""
    path = build_path(CASE)
    power = launch_profile(CASE)
    reference = make_model("numerical_dra", CASE).evaluate_path(path, power)
    paper = make_model("hybrid_raman_paper", CASE, nli_form="paper").evaluate_path(path, power)
    delta = np.asarray(reference.snr_db) - np.asarray(paper.snr_db)
    assert float(np.max(np.abs(delta))) < 0.05, f"max |delta|={float(np.max(np.abs(delta))):.3f} dB"


def test_backward_pump_matches_the_full_paper_model() -> None:
    """A backward pump must match the full paper model within the reference error."""
    case = build_case("c_57ch")
    path = _one_span(case)
    power = launch_profile(case)
    reference = make_model("numerical_dra", case, raman_pumps=[_pump()]).evaluate_path(path, power)
    paper = make_model("hybrid_raman_paper", case, raman_pumps=[_pump()], nli_form="paper").evaluate_path(path, power)
    delta = np.asarray(reference.snr_db) - np.asarray(paper.snr_db)
    assert float(np.max(np.abs(delta))) < 0.1, f"max |delta|={float(np.max(np.abs(delta))):.3f} dB"


def test_forward_pump_changes_the_nli() -> None:
    """前向泵把增益放在跨首，NLI 必须明显高于无泵；这是 lumped 抓不到的。"""
    case = build_case("scaling_40ch")
    path = _one_span(case)
    power = launch_profile(case)
    reference = make_model("numerical_dra", case, raman_pumps=[_pump("forward")]).evaluate_path(path, power)
    base = make_model("numerical_dra", case).evaluate_path(path, power)
    nli = np.mean(np.asarray(reference.spm_noise_w) + np.asarray(reference.xpm_noise_w))
    base_nli = np.mean(np.asarray(base.spm_noise_w) + np.asarray(base.xpm_noise_w))
    assert nli > 1.5 * base_nli


def test_invalid_profile_is_rejected() -> None:
    """An even/too-small longitudinal grid must be rejected."""
    with pytest.raises(ValueError, match="n_profile"):
        make_model("numerical_dra", CASE, n_profile=4)
