"""Comparison model configurations exposed through the registry."""

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import create_model
from network_nlo_eval.models.comparison import PCFMOriginalModel
from network_nlo_eval.network.elements import FiberSpanConfig


def test_original_pcfm_comparison_configuration() -> None:
    model = create_model(
        "pcfm_original", SpectrumGrid(2, 50e9, 193.1e12), FiberSpanConfig(length_km=80)
    )
    assert isinstance(model, PCFMOriginalModel)
    assert model.info.name == "pcfm_original"
    assert model.poly_degree == 5
    assert not model.egn
    assert not model.edge_fix
    assert not model.cache_spans
