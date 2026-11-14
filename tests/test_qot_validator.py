"""Service-level QoT accumulation and blocking."""

import numpy as np
import pytest

from network_nlo_eval.network.elements import ROADMConfig
from network_nlo_eval.network.state import LinkNoiseCache
from network_nlo_eval.rwa.qot_checker import accumulate_service_snr, fold_roadm_terms


def test_accumulates_link_noise_and_roadm_filtering() -> None:
    roadm = fold_roadm_terms(
        ROADMConfig(insertion_loss_db=0, add_drop_loss_db=0, filtering_penalty_db=3),
        is_last_hop=True,
    )
    cache = LinkNoiseCache(np.array([1e-6]), np.array([2e-6]), np.array([3e-6]), 1)
    snr = accumulate_service_snr(
        [(roadm, cache)],
        channel_idx=0,
        launch_power_w=1e-3,
        frequency_hz=193.1e12,
        channel_bandwidth_hz=32e9,
        is_blocked=lambda _: False,
    )
    assert snr == pytest.approx(10 * np.log10(1e-3 / (6e-6 * 10**0.3)))


def test_blocked_hop_stops_lazy_iteration() -> None:
    roadm = fold_roadm_terms(ROADMConfig(), is_last_hop=False)
    cache = LinkNoiseCache(np.zeros(1), np.zeros(1), np.zeros(1), 1)

    def hops():
        yield roadm, cache
        msg = "blocked paths must not evaluate later hops"
        raise AssertionError(msg)

    assert (
        accumulate_service_snr(
            hops(),
            channel_idx=0,
            launch_power_w=1e-3,
            frequency_hz=193.1e12,
            channel_bandwidth_hz=32e9,
            is_blocked=lambda _: True,
        )
        == -np.inf
    )
