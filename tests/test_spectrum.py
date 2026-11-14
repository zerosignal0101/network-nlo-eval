"""Key checks for the optical spectrum grid."""

import numpy as np
import pytest

from network_nlo_eval.core.constants import C_LIGHT, DB_TO_NP, NP_TO_DB
from network_nlo_eval.core.spectrum import SpectrumGrid


def test_grid_frequencies_and_wavelengths() -> None:
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    np.testing.assert_allclose(grid.frequencies, [193.05e12, 193.1e12, 193.15e12])
    np.testing.assert_allclose(grid.wavelengths * grid.frequencies, C_LIGHT)
    assert grid.get_channel_info(1) == (grid.frequencies[1], grid.wavelengths[1])
    assert DB_TO_NP * NP_TO_DB == pytest.approx(1)


@pytest.mark.parametrize("args", [(0, 1, 1), (True, 1, 1), (3, 0, 1), (3, 1, 0), (3, 2, 1)])
def test_invalid_grids(args: tuple[int, float, float]) -> None:
    with pytest.raises(ValueError):
        SpectrumGrid(*args)


def test_channel_index_out_of_bounds() -> None:
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    with pytest.raises(IndexError):
        grid.get_channel_info(-1)
    with pytest.raises(IndexError):
        grid.get_channel_info(3)
