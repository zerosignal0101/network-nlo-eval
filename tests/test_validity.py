"""Network validity policies, aggregation, and spectrum-aware caching."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.core.types import LinkKey
from network_nlo_eval.models.base import MultiSpanOpticalPath
from network_nlo_eval.models.pcfm import PCFMModel
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig
from network_nlo_eval.rwa.validity import (
    ChannelValidity,
    H3LinkValidityProvider,
    LinkValidity,
    LinkValidityCache,
    LinkValidityPolicy,
    occupancy_fingerprint,
)


def _report(key: LinkKey = (0, 1), *, dense: bool = False) -> LinkValidity:
    return LinkValidity(
        valid=np.array([True, False, True]),
        active=np.ones(3, dtype=bool),
        flagged=np.array([False, True, False]),
        confidence=np.array([1.0, 0.5, 1.0]),
        margin_db=np.array([2.0, np.inf, 1.0]),
        soft_margin_db=np.array([0.2, 0.4, 0.1]),
        model="test",
        link_key=key,
        dense_grid_overlap=dense,
    )


def test_policies_preserve_off_and_distinguish_flag_reject_and_margins() -> None:
    report = _report()
    snr = np.array([20.0, 15.0, np.nan])
    assert report.adjusted_snr_db(snr) is snr
    np.testing.assert_array_equal(
        report.adjusted_snr_db(snr, LinkValidityPolicy(mode="flag")), [20, 15, -np.inf]
    )
    np.testing.assert_array_equal(
        report.adjusted_snr_db(snr, LinkValidityPolicy(mode="reject")),
        [20, -np.inf, -np.inf],
    )
    np.testing.assert_array_equal(
        report.adjusted_snr_db(snr, LinkValidityPolicy(mode="degrade")),
        [18, -np.inf, -np.inf],
    )
    np.testing.assert_array_equal(
        report.adjusted_snr_db(snr, LinkValidityPolicy(mode="soft")),
        [19.8, -np.inf, -np.inf],
    )
    dense = _report(dense=True)
    assert not np.any(
        dense.effective_valid(
            LinkValidityPolicy(mode="reject", dense_grid_action="reject")
        )
    )
    np.testing.assert_array_equal(
        dense.effective_valid(LinkValidityPolicy()), dense.valid
    )
    assert not report.valid.flags.writeable


def test_path_aggregation_and_cache_isolate_spectrum() -> None:
    first, second = _report((0, 1)), _report((1, 2), dense=True)
    path = ChannelValidity.combine([first, second], 3)
    np.testing.assert_array_equal(path.valid, [True, False, True])
    np.testing.assert_allclose(path.margin_db, [4, 0, 2])
    assert path.soft_margin_db is not None
    np.testing.assert_allclose(path.soft_margin_db, [0.4, 0, 0.2])
    assert path.dense_grid_overlap
    cache = LinkValidityCache(max_entries=1)
    power = np.array([1e-3, 0.0, 1e-3])
    cache.put((0, 1), power, 80e3, first)
    assert cache.get((0, 1), power, 80e3) is first
    assert cache.get((0, 1), power * 2.0, 80e3) is None
    assert occupancy_fingerprint(power, 80e3) != occupancy_fingerprint(power, 81e3)
    assert cache.invalidate((0, 1)) == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_pcfm_provider_evaluates_and_reuses_only_matching_spectrum() -> None:
    fiber = FiberSpanConfig(length_km=80)
    model = PCFMModel(SpectrumGrid(3, 50e9, 193.1e12), fiber, 32e9)
    provider = H3LinkValidityProvider(model)
    power = np.array([1e-3, 2e-3, 0.0])
    first = provider((0, 1), fiber, None, power, 80e3)
    assert first is provider((0, 1), fiber, None, power.copy(), 80e3)
    assert provider.cache.hits == 1
    changed = provider((0, 1), fiber, None, power * 1.1, 80e3)
    assert changed.fingerprint != first.fingerprint
    assert provider.cache.misses == 2
    with pytest.raises(ValueError, match="shape"):
        provider((0, 1), fiber, None, np.ones(2), 80e3)


def test_validity_cache_distinguishes_ordered_equipment_for_the_same_link() -> None:
    short_fiber = FiberSpanConfig(length_km=30)
    long_fiber = FiberSpanConfig(length_km=50)
    model = PCFMModel(SpectrumGrid(2, 50e9, 193.1e12), long_fiber, 32e9)
    provider = H3LinkValidityProvider(model)
    power = np.array([1e-3, 2e-3])
    edfa = EDFAConfig(noise_figure_db=np.array([5.0, 6.0]))
    path = MultiSpanOpticalPath([short_fiber, long_fiber], [None, edfa])
    reversed_path = MultiSpanOpticalPath([long_fiber, short_fiber], [edfa, None])

    first = provider.evaluate_path((0, 1), path, power)
    assert first.span_count == 2
    assert provider.evaluate_path((0, 1), path, power.copy()) is first
    reverse_order = provider.evaluate_path((0, 1), reversed_path, power)
    assert reverse_order.fingerprint != first.fingerprint
    assert provider.cache.hits == 1
    assert provider.cache.misses == 2
    assert provider.cache.invalidate((0, 1)) == 2
