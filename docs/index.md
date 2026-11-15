# network-nlo-eval

`network-nlo-eval` evaluates quality of transmission (QoT) in multi-band WDM
optical networks. It combines physical-layer models with topology and spectrum
state, QoT-aware routing and spectrum assignment, and discrete-event simulation.

## Installation

Python 3.11 or 3.12 is required. Install from a checkout with:

```bash
python -m pip install .
```

Waveform-based comparisons additionally require `physics-nlo-eval` in the same
Python environment. Statistical QoT models work without it.

## Path evaluation

```python
import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

grid = SpectrumGrid(
    num_channels=3, channel_spacing_hz=50e9, center_frequency_hz=193.1e12
)
fiber = FiberSpanConfig(length_km=80)
path = MultiSpanOpticalPath(fiber_configs=[fiber], edfa_configs=[EDFAConfig()])
model = create_model("ecz", grid, fiber, symbol_rate_hz=32e9)
result = model.evaluate_path(path, np.full(3, 1e-3))
print(result.snr_db)
```

The example launches 1 mW per channel. Channel frequencies use hertz, fiber
lengths use kilometers, and result powers use watts. Models differ in their
assumptions and power conventions; check each model's documentation before using
its results.

## Command line

Use `network-nlo-eval models` to list registered models and their options.
`network-nlo-eval simulate --service-num 10` runs the bundled example topology;
`network-nlo-eval compare --help` and `network-nlo-eval bench --help` describe
path comparison and benchmarking. All commands provide `--help`.

See the [Python API](api.md) for the model interface and network components.

See the [model literature audit](model-literature-audit.md) for all registered
models, their paper equations, supported scenarios, and implementation variants.
