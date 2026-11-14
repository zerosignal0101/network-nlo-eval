"""Waveform validity policy integration without an optional solver."""

import numpy as np
import pytest

from network_nlo_eval.physics.qot_checker import apply_link_validity
from network_nlo_eval.rwa.validity import LinkValidity, LinkValidityPolicy


def test_validity_overlay_preserves_input_and_applies_policy() -> None:
    report = LinkValidity(
        valid=np.array([True, False]),
        active=np.array([True, True]),
        flagged=np.array([False, True]),
        confidence=np.array([1.0, 0.5]),
        margin_db=np.array([2.0, np.inf]),
        soft_margin_db=np.array([0.2, 0.4]),
        model="test",
    )
    snr = np.array([20.0, 15.0])
    metrics = {"snr": snr}
    off = apply_link_validity(metrics, report, LinkValidityPolicy())
    assert off["snr"] is snr
    adjusted = apply_link_validity(metrics, report, LinkValidityPolicy(mode="degrade"))
    np.testing.assert_array_equal(adjusted["snr"], [18.0, -np.inf])
    np.testing.assert_array_equal(snr, [20.0, 15.0])
    np.testing.assert_array_equal(adjusted["valid"], [True, False])
    with pytest.raises(ValueError, match="shape"):
        apply_link_validity({"snr": np.ones(3)}, report, LinkValidityPolicy())
