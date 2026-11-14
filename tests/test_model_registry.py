"""Tests for path configuration and the QoT model registry."""

from typing import cast

import pytest

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models import base
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig


def test_optical_path_configuration_lengths() -> None:
    fiber = FiberSpanConfig(length_km=80)
    edfa = EDFAConfig()
    path = base.MultiSpanOpticalPath([fiber], [edfa], link_keys=[(0, 1)])
    assert path.fiber_configs[0].length_km == 80
    with pytest.raises(ValueError, match="matching non-empty"):
        base.MultiSpanOpticalPath([fiber], [])
    with pytest.raises(ValueError, match="ROADM config count"):
        base.MultiSpanOpticalPath([fiber], [edfa], roadm_configs=[])
    with pytest.raises(ValueError, match="Link key count"):
        base.MultiSpanOpticalPath([fiber], [edfa], link_keys=[(0, 1), (1, 2)])


def test_registry_registration_discovery_and_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(base, "_DISCOVERED", True)
    monkeypatch.setattr(base, "MODEL_REGISTRY", {})
    monkeypatch.setattr(base, "MODEL_INFO", {})
    info = base.ModelInfo("sample", "Test model", "gn", "Test reference")
    model = cast("base.QoTModel", object())
    base.register_model(info, lambda *_args, **_kwargs: model)
    assert base.available_models() == {"sample": info}
    assert base.get_model_info("sample") is info
    assert base.create_model("sample", 1, value=2) is model
    assert base.describe_models()[0]["name"] == "sample"
    with pytest.raises(ValueError, match="already registered"):
        base.register_model(info, lambda: model)
    with pytest.raises(KeyError, match="Unknown QoT model"):
        base.create_model("missing")


def test_registered_models_expose_their_supported_protocols() -> None:
    """Path evaluation does not require every model to expose a span kernel."""
    grid = SpectrumGrid(3, 50e9, 193.1e12)
    fiber = FiberSpanConfig(length_km=80)
    legacy_model = base.create_model("ecz", grid, fiber, 32e9)
    kernel_model = base.create_model("carena2014", grid, fiber, 32e9)

    assert isinstance(legacy_model, base.QoTModel)
    assert legacy_model.info.name == "ecz"
    assert not isinstance(legacy_model, base.KernelQoTModel)
    assert isinstance(kernel_model, base.KernelQoTModel)
