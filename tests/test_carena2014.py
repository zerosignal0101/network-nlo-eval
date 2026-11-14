"""Carena SCI integration domains, MCI scaling, and route coherence."""

import numpy as np
import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.models.carena2014 import EgnIntegrator, LinkFunction
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


@pytest.mark.parametrize(
    ("n_spans", "expected_l1", "expected_l2"),
    [
        (1, 2.8383902996779713e34, 1.836751441606607e34),
        (10, 1.368149429674118e35, 3.311288792547734e34),
    ],
)
def test_distinct_fourth_order_sci_integrals(n_spans, expected_l1, expected_l2):
    # Independent direct two-dimensional Gauss-Legendre quadrature, 32 nodes
    # per oscillation period, without the model's antiderivative table.
    link = LinkFunction(
        gamma=1.3219450475712609e-3,
        alpha_power_npm=3.735371857445309e-5,
        beta2_s2_m=-2.1607590148802473e-26,
        span_length_m=100e3,
        n_spans=n_spans,
    )
    integrator = EgnIntegrator(link=link, symbol_rate_hz=64e9, points=6)
    band = (-32e9, 32e9)
    l1 = integrator._core(0.0, band, band, band, "fourth")[1]
    l2 = integrator._core(0.0, band, band, band, "l2")[1]
    assert l1 == pytest.approx(expected_l1, rel=1e-3)
    assert l2 == pytest.approx(expected_l2, rel=1e-3)
    assert l1 != pytest.approx(l2, rel=1e-3)


def test_mci_and_multispan_route():
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=20)
    model = create_model(
        "carena2014", grid, fiber, 32e9, quadrature_points=4, frequency_points=5
    )
    kernel = model.kernel_for(fiber)
    power = np.full(3, 1e-3)
    table = kernel.mci_kappas(1, 20e3)
    assert all(upper > 0 > lower for upper, lower in table)
    assert max(np.max(item.kappa_m0) for item in table.values()) > 0
    gn, _ = kernel._mci_terms(1, 20e3, power)
    doubled, _ = kernel._mci_terms(1, 20e3, 2 * power)
    assert gn[1] > 0
    assert doubled[1] == pytest.approx(8 * gn[1])
    path = MultiSpanOpticalPath([fiber] * 2, [EDFAConfig()] * 2)
    result = model.evaluate_path(path, power)
    assert np.all(np.isfinite(result.snr_db))
    assert np.all(result.spm_noise_w > 0)
    assert np.all(result.xpm_noise_w > 0)
