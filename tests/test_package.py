from __future__ import annotations

import importlib.metadata

import network_nlo_eval as m


def test_version() -> None:
    assert importlib.metadata.version("network_nlo_eval") == m.__version__
