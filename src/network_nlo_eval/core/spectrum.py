"""定义多波段系统的频谱网格 (C+L+S 波段)."""

from dataclasses import dataclass

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.types import NDArrayFloat


@dataclass(frozen=True)
class SpectrumGrid:
    """定义多波段系统的频谱网格配置。

    所有频率均以 Hz 为单位。
    """

    num_channels: int  # 信道总数
    channel_spacing_hz: float  # 信道间隔 (Hz)
    center_frequency_hz: float  # 中心频率 (Hz)

    @property
    def frequencies(self) -> NDArrayFloat:
        """生成绝对频率网格 (Hz).

        例如，如果 num_channels=3，center_frequency_hz=193.1e12，spacing=50e9，
        则频率为 [193.05e12, 193.1e12, 193.15e12]。
        """
        # 计算相对频率，并使其关于 0 对称
        f_rel = (np.arange(self.num_channels) - (self.num_channels - 1) / 2) * self.channel_spacing_hz
        return self.center_frequency_hz + f_rel

    @property
    def wavelengths(self) -> NDArrayFloat:
        """生成绝对波长网格 (m)."""
        return C_LIGHT / self.frequencies

    def get_channel_info(self, channel_idx: int) -> tuple[float, float]:
        """获取特定信道的频率和波长."""
        if not (0 <= channel_idx < self.num_channels):
            raise IndexError(f"Channel index {channel_idx} out of bounds for {self.num_channels} channels.")
        freq = self.frequencies[channel_idx]
        wavelength = self.wavelengths[channel_idx]
        return freq, wavelength
