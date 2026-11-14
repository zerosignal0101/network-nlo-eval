"""Frequency grids for multiband optical systems."""

from dataclasses import dataclass
from typing import cast

import numpy as np

from network_nlo_eval.core.constants import C_LIGHT
from network_nlo_eval.core.types import NDArrayFloat


@dataclass(frozen=True)
class SpectrumGrid:
    """Describe an evenly spaced optical frequency grid.

    Parameters
    ----------
    num_channels : int
        Number of channels.
    channel_spacing_hz : float
        Frequency spacing between adjacent channels, in Hz.
    center_frequency_hz : float
        Center frequency of the grid, in Hz.
    """

    num_channels: int
    channel_spacing_hz: float
    center_frequency_hz: float

    def __post_init__(self) -> None:
        """Validate the grid dimensions and absolute frequencies."""
        if (
            isinstance(cast("object", self.num_channels), bool)
            or not isinstance(cast("object", self.num_channels), int)
            or self.num_channels <= 0
        ):
            msg = "num_channels must be a positive integer."
            raise ValueError(msg)
        if self.channel_spacing_hz <= 0 or self.center_frequency_hz <= 0:
            msg = "Channel spacing and center frequency must be positive."
            raise ValueError(msg)
        lowest = (
            self.center_frequency_hz
            - (self.num_channels - 1) * self.channel_spacing_hz / 2
        )
        if lowest <= 0:
            msg = "Spectrum grid contains a non-positive optical frequency."
            raise ValueError(msg)

    @property
    def frequencies(self) -> NDArrayFloat:
        """Return the absolute channel frequencies in Hz."""
        f_rel = (
            np.arange(self.num_channels) - (self.num_channels - 1) / 2
        ) * self.channel_spacing_hz
        return self.center_frequency_hz + f_rel

    @property
    def wavelengths(self) -> NDArrayFloat:
        """Return the absolute channel wavelengths in meters."""
        return C_LIGHT / self.frequencies

    def get_channel_info(self, channel_idx: int) -> tuple[float, float]:
        """Return a channel's frequency in Hz and wavelength in meters.

        Parameters
        ----------
        channel_idx : int
            Zero-based channel index.

        Returns
        -------
        tuple of float
            Frequency and wavelength, respectively.

        Raises
        ------
        IndexError
            If the channel index is outside the grid.
        """
        if not (0 <= channel_idx < self.num_channels):
            msg = f"Channel index {channel_idx} out of bounds for {self.num_channels} channels."
            raise IndexError(msg)
        freq = self.frequencies[channel_idx]
        wavelength = self.wavelengths[channel_idx]
        return freq, wavelength
