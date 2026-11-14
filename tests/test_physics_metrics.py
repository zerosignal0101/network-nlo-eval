"""Tests for BER and error-vector-magnitude signal quality metrics."""

import math

import pytest
from scipy.special import erfc

from network_nlo_eval.physics.metrics import compute_ber_from_snr, compute_snr_from_evm


def test_qpsk_and_16qam_ber() -> None:
    assert compute_ber_from_snr(0, "QPSK") == pytest.approx(0.5 * erfc(1))
    qam_erfc = erfc(math.sqrt(0.1))
    assert compute_ber_from_snr(0, "16QAM") == pytest.approx(
        3 / 8 * qam_erfc * (1 - 0.5 * qam_erfc)
    )
    assert compute_ber_from_snr(100, "QPSK") == 1e-15
    with pytest.raises(ValueError, match="Unsupported modulation"):
        compute_ber_from_snr(0, "BPSK")


def test_evm_to_snr_and_bounds() -> None:
    assert compute_snr_from_evm(0.1) == pytest.approx(20)
    assert compute_snr_from_evm(100) == -20
    assert math.isinf(compute_snr_from_evm(0))
