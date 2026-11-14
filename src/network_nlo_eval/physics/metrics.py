"""Physical-layer metrics for signal quality and performance."""

import numpy as np
from scipy.special import erfc

from network_nlo_eval.models.numba_kernels import (
    _db_to_lin,
    _lin_to_db,
)


def compute_ber_from_snr(snr_db: float, modulation: str = "QPSK") -> float:
    """Estimate bit error rate from SNR using a Q-function approximation.

    Parameters
    ----------
    snr_db : float
        Signal-to-noise ratio in dB.
    modulation : str, optional
        Modulation format: ``"QPSK"`` or ``"16QAM"``.

    Returns
    -------
    float
        Estimated BER, bounded below by 1e-15.

    Raises
    ------
    ValueError
        If the modulation format is unsupported.
    """
    snr_linear = _db_to_lin(snr_db)

    # QPSK: BER = Q(sqrt(2*SNR))
    # 16QAM: BER = (3/8) * Q(sqrt(SNR/5))
    # etc.
    if modulation == "QPSK":
        # Q(x) = 0.5 * erfc(x / sqrt(2))
        q = np.sqrt(2 * snr_linear)
        ber = 0.5 * erfc(q / np.sqrt(2))
    elif modulation == "16QAM":
        q = np.sqrt(snr_linear / 5)
        ber = (3 / 8) * erfc(q / np.sqrt(2)) * (1 - 0.5 * erfc(q / np.sqrt(2)))
    else:
        msg = "Unsupported modulation format."
        raise ValueError(msg)

    return float(max(ber, 1e-15))  # Floor to avoid log(0)


def compute_snr_from_evm(evm_linear: float, modulation: str = "QPSK") -> float:  # noqa: ARG001
    """Estimate SNR from error vector magnitude using SNR = 1 / EVM².

    Parameters
    ----------
    evm_linear : float
        Linear error vector magnitude.
    modulation : str, optional
        Modulation format; retained for interface compatibility and unused.

    Returns
    -------
    float
        Estimated SNR in dB, bounded below by -20 dB. Non-positive EVM
        returns positive infinity.
    """
    if evm_linear > 0:
        evm_sq = evm_linear * evm_linear
        # EVM² = P_error / P_signal ≈ 1/SNR (Standard EVM-to-SNR relation)
        snr_linear = 1.0 / evm_sq if evm_sq > 0 else 1e20
        return float(max(_lin_to_db(snr_linear), -20.0))

    return float("inf")
