"""Two-tier fallback planning and full-spectrum evaluation."""

from types import SimpleNamespace

import numpy as np
import pytest

from network_nlo_eval.rwa.two_tier import TwoTierConfig, TwoTierEvaluator


class Budget:
    validity = SimpleNamespace(
        flagged=np.array([True, False, True]), active=np.array([True, True, False])
    )

    def adjusted_snr_db(self, snr):
        return snr - 2


class H3:
    def evaluate_path_with_uncertainty(self, _path, _power):
        return SimpleNamespace(snr_db=np.array([10.0, 20.0, 30.0])), Budget()


def test_fallback_is_lazy_and_replaces_only_flagged_active_channels() -> None:
    power = np.array([1.0, 2.0, 0.0])
    calls = []

    class Fallback:
        def evaluate_path(self, _path, actual_power):
            calls.append(actual_power.copy())
            return SimpleNamespace(snr_db=np.array([15.0, 50.0, 60.0]))

    evaluator = TwoTierEvaluator(H3(), Fallback)
    planned = evaluator.evaluate(None, power)
    assert not planned.fallback_ran
    assert not calls
    np.testing.assert_array_equal(planned.effective_snr_db, [8, 18, 28])
    evaluator = TwoTierEvaluator(H3(), Fallback, config=TwoTierConfig(enabled=True))
    result = evaluator.evaluate(None, power)
    assert result.fallback_ran
    np.testing.assert_array_equal(calls[0], power)
    np.testing.assert_array_equal(result.effective_snr_db, [15, 18, 28])
    assert not result.effective_snr_db.flags.writeable


def test_enabled_fallback_requires_factory() -> None:
    evaluator = TwoTierEvaluator(H3(), config=TwoTierConfig(enabled=True))
    with pytest.raises(ValueError, match="fallback_factory"):
        evaluator.evaluate(None, np.array([1.0, 2.0, 0.0]))
