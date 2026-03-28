"""数字信号处理 (DSP) 模块，用于处理仿真结果。"""

import numpy as np

from network_nlo_eval.core.types import NDArrayComplex, NDArrayFloat


class DSPChain:
    """接收端数字信号处理链。

    用于恢复信号并评估性能。
    """

    @staticmethod
    def chromatic_dispersion_compensation(
        signal_freq_domain: NDArrayComplex,
        length_m: float,
        beta2: float,
        omega: NDArrayFloat,
    ) -> NDArrayComplex:
        """频域静态色散补偿 (CDC)。

        Args:
            signal_freq_domain: 频域信号。
            length_m: 链路总长度 (m)。
            beta2: 色散参数 beta2 (s^2/m)。
            omega: 角频率轴 (rad/s)。
        Returns:
            NDArrayComplex: 补偿后的频域信号。
        """
        # 注意补偿滤波器是传播算子的共轭 (反向传播)
        cdc_filter = np.exp(1j * (beta2 / 2) * (omega**2) * length_m)
        return signal_freq_domain * cdc_filter

    @staticmethod
    def matched_filter(
        signal_time_domain: NDArrayComplex,
        symbol_rate_hz: float,
        samples_per_symbol: int,
        roll_off_factor: float = 0.1,
    ) -> NDArrayComplex:
        """根升余弦 (Root Raised Cosine, RRC) 匹配滤波。

        这里仅提供一个概念性的接口，实际实现需要一个 RRC 滤波器设计。
        Args:
            signal_time_domain: 时域信号。
            symbol_rate_hz: 符号速率 (Hz)。
            samples_per_symbol: 每个符号的采样点数。
            roll_off_factor: RRC 滤波器的滚降系数。
        Returns:
            NDArrayComplex: 滤波后的信号。
        """
        # 这是一个简化实现，实际 RRC 滤波器需要更多参数和设计
        # 实际实现需要生成 RRC 脉冲，然后与信号进行卷积
        # 暂时返回原信号，表示一个占位符
        print("Warning: Matched filter is a placeholder, returning original signal.")
        return signal_time_domain

    # 更多 DSP 功能可在此添加，如：
    # @staticmethod
    # def cpr(signal: NDArrayComplex) -> NDArrayComplex:
    #     """载波相位恢复 (CPE)。"""
    #     pass
