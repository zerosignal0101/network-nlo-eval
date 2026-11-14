"""Tests for network component validation and slot occupancy."""

import numpy as np
import pytest
from pydantic import ValidationError

from network_nlo_eval.network.elements import (
    EDFAConfig,
    FiberSpanConfig,
    RamanPumpConfig,
    ROADMConfig,
)
from network_nlo_eval.network.slots import connection_overlaps_slots


def test_fiber_lumped_losses_and_pumps() -> None:
    pump = RamanPumpConfig(wavelength_nm=1450, power_w=0.2, direction="forward")
    fiber = FiberSpanConfig(
        length_km=80, raman_pumps=[pump], lumped_loss=[(1000, 1.5), (1000, 0.5)]
    )
    assert fiber.raman_pumps == [pump]
    assert fiber.lumped_loss == [(1000, 1.5), (1000, 0.5)]


@pytest.mark.parametrize(
    "event", [(0, 1), (80_000, 1), (float("nan"), 1), (1000, -1), (1000, float("inf"))]
)
def test_fiber_rejects_invalid_lumped_losses(event: tuple[float, float]) -> None:
    with pytest.raises(ValidationError, match="lumped loss"):
        FiberSpanConfig(length_km=80, lumped_loss=[event])


def test_amplifier_accepts_per_channel_noise_figures_and_rejects_nonfinite_values() -> (
    None
):
    amplifier = EDFAConfig(
        noise_figure_db=np.array([4.0, 5.0]), gain_ripple_db=np.array([0.1, -0.1])
    )
    np.testing.assert_array_equal(amplifier.noise_figure_db, [4, 5])
    with pytest.raises(ValidationError, match="noise figure"):
        EDFAConfig(noise_figure_db=np.array([4.0, np.nan]))
    with pytest.raises(ValidationError, match="gain ripple"):
        EDFAConfig(gain_ripple_db=np.array([np.inf]))


def test_roadm_loss_and_blocked_indices() -> None:
    roadm = ROADMConfig(
        insertion_loss_db=5,
        express_attenuation_db=2,
        add_drop_loss_db=1,
        blocked_slots=[1, 4],
    )
    assert roadm.path_loss_db("express") == 7
    assert roadm.path_loss_db("add") == 6
    assert roadm.path_loss_db("drop") == 6
    with pytest.raises(ValidationError, match="unique"):
        ROADMConfig(blocked_slots=[1, 1])
    with pytest.raises(ValidationError, match="non-negative"):
        ROADMConfig(blocked_channels=[-1])


def test_connection_slot_overlap_uses_half_open_interval() -> None:
    assert connection_overlaps_slots(3, 2, {4})
    assert not connection_overlaps_slots(3, 2, {2, 5})
    assert not connection_overlaps_slots(3, 2, set())
