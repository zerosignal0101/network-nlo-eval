"""用于评估信号质量和性能的物理层指标。"""

import numpy as np


def snr_to_q_factor(snr_linear: float) -> float:
    """将线性信噪比 (SNR) 转换为 Q 因子。

    Q 因子定义为 sqrt(SNR)。
    Args:
        snr_linear: 线性形式的 SNR 值 (不是 dB)。
    Returns:
        float: Q 因子。
    """
    if snr_linear < 0:
        raise ValueError("SNR must be non-negative.")
    return np.sqrt(snr_linear)


def q_factor_to_snr(q_factor: float) -> float:
    """将 Q 因子转换为线性信噪比 (SNR)。

    Args:
        q_factor: Q 因子。
    Returns:
        float: 线性形式的 SNR 值。
    """
    return q_factor**2


def snr_to_ber(snr_linear: float, modulation_order: int = 4) -> float:
    """根据线性信噪比 (SNR) 估算误码率 (BER)。

    这里使用一个简化模型，假设 QPSK (modulation_order=4, M=4)。
    对于 QPSK，BER ≈ 0.5 * erfc(sqrt(SNR/2))。
    Args:
        snr_linear: 线性形式的 SNR 值。
        modulation_order: 调制阶数 (例如，BPSK 为 2, QPSK 为 4, 16QAM 为 16)。
    Returns:
        float: 估算的误码率。
    """
    if snr_linear < 0:
        return 0.5  # 极低 SNR 接近 0.5

    # 简化：假设 QPSK (M=4)，对于相干接收，可以近似为：
    # BER = 0.5 * erfc(sqrt(SNR_linear / (2 * np.log2(M))))
    # 对于 BPSK (M=2) BER = 0.5 * erfc(sqrt(SNR_linear))
    # 对于 QPSK (M=4) BER = 0.5 * erfc(sqrt(SNR_linear / 2))

    # 这里使用一个更通用的高斯近似
    q_factor = snr_to_q_factor(snr_linear)
    if modulation_order == 2:  # BPSK
        return 0.5 * np.erfc(q_factor / np.sqrt(2))
    elif modulation_order == 4:  # QPSK
        return 0.5 * np.erfc(q_factor / 2.0)  # Simplified, more complex for QPSK
    else:  # Fallback to a general Gaussian approximation with a scaling factor
        return 0.5 * np.erfc(q_factor / (np.sqrt(2) * np.log2(modulation_order) / 2))  # Very simplified


def evm_to_snr_db(evm_percentage: float) -> float:
    """将误差矢量幅度 (EVM) 百分比转换为信噪比 (SNR) (dB)。

    SNR_dB = 10 * log10(1 / (EVM_linear^2))
    EVM_linear = EVM_percentage / 100
    Args:
        evm_percentage: EVM 值 (百分比，例如 3.5 代表 3.5%)。
    Returns:
        float: SNR 值 (dB)。
    """
    if evm_percentage <= 0:
        return np.inf  # 无 EVM 代表无限 SNR
    evm_linear = evm_percentage / 100.0
    return 10 * np.log10(1.0 / (evm_linear**2))


def snr_db_to_evm(snr_db: float) -> float:
    """将信噪比 (SNR) (dB) 转换为误差矢量幅度 (EVM) (百分比)。

    EVM_linear = sqrt(1 / SNR_linear)
    Args:
        snr_db: SNR 值 (dB)。
    Returns:
        float: EVM 值 (百分比)。
    """
    snr_linear = 10 ** (snr_db / 10.0)
    if snr_linear <= 0:
        return 1000.0  # 极低 SNR (或负数) 对应极高 EVM
    return np.sqrt(1.0 / snr_linear) * 100.0
