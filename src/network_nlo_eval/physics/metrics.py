"""用于评估信号质量和性能的物理层指标。"""

import numpy as np

from network_nlo_eval.models.numba_kernels import (
    _db_to_lin,
    _lin_to_db,
)


def compute_ber_from_snr(snr_db: float, modulation: str = "QPSK") -> float:
    """从 SNR 估计 BER (使用 Q 函数近似)。

    Parameters
    ----------
    snr_db : float
        SNR (dB)。
    modulation : str
        调制格式。

    Returns
    -------
    float
        估计的 BER。
    """
    snr_linear = _db_to_lin(snr_db)

    # QPSK: BER = Q(sqrt(2*SNR))
    # 16QAM: BER = (3/8) * Q(sqrt(SNR/5))
    # etc.
    from scipy.special import erfc

    if modulation == "QPSK":
        # Q(x) = 0.5 * erfc(x / sqrt(2))
        q = np.sqrt(2 * snr_linear)
        ber = 0.5 * erfc(q / np.sqrt(2))
    elif modulation == "16QAM":
        q = np.sqrt(snr_linear / 5)
        ber = (3 / 8) * erfc(q / np.sqrt(2)) * (1 - 0.5 * erfc(q / np.sqrt(2)))
    else:
        raise ValueError("Unsupported modulation format.")

    return max(ber, 1e-15)  # Floor to avoid log(0)


def compute_snr_from_evm(evm_linear: float, modulation: str = "QPSK") -> float:
    """从 EVM 估计 SNR。

    Parameters
    ----------
    evm_linear : float
        EVM (线性值)。
    modulation : str
        调制格式。

    Returns
    -------
    float
        估计的 SNR (dB)。
    """
    # EVM^2 = 1/(1+SNR) for QPSK => SNR = 1/EVM^2 - 1
    if evm_linear > 0:
        snr_linear = 1 / (evm_linear**2) - 1
        if snr_linear > 0:
            return _lin_to_db(snr_linear)

    return 40.0  # Very high SNR
