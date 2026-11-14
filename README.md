# network-nlo-eval

A Python toolkit for evaluating physical-layer quality of transmission (QoT)
in wavelength-division multiplexed optical networks. It provides nonlinear
interference models, network topology and spectrum state, QoT-aware routing
and spectrum assignment, and a discrete-event simulator.

## Install

Python 3.11 or 3.12 is required. From a checkout, install the package and its
dependencies with:

```bash
python -m pip install .
```

Waveform-based comparisons additionally require `physics-nlo-eval` in the same
Python environment. Statistical models do not require it.

## Evaluate a path

```python
import numpy as np

from network_nlo_eval.core.spectrum import SpectrumGrid
from network_nlo_eval.models.base import MultiSpanOpticalPath, create_model
from network_nlo_eval.network.elements import EDFAConfig, FiberSpanConfig

grid = SpectrumGrid(num_channels=3, channel_spacing_hz=50e9, center_frequency_hz=193.1e12)
fiber = FiberSpanConfig(length_km=80)
path = MultiSpanOpticalPath(fiber_configs=[fiber], edfa_configs=[EDFAConfig()])
model = create_model("ecz", grid, fiber, symbol_rate_hz=32e9)
result = model.evaluate_path(path, np.full(3, 1e-3))
print(result.snr_db)
```

Launch powers and returned signal and noise powers use watts; the example
uses 1 mW per channel. Frequencies are in hertz and fiber lengths in
`FiberSpanConfig` are in kilometers. Models have different approximations,
power conventions, and applicability limits: consult their documentation and
options when selecting a model.

## Command line

```bash
network-nlo-eval models                 # List models and options as JSON
network-nlo-eval simulate --service-num 10 --output-dir results
network-nlo-eval compare --help         # Compare a topology path
network-nlo-eval bench --help           # Run a model benchmark
```

The simulation command includes an example topology. Provide `--topology-file`
to use a NetworkX node-link JSON topology of your own. Use `--help` on each
command for its options, including flexible-grid and launch-power settings.

## Development

Run the focused test suite with `python -m pytest`. The project uses NumPy-style
Python documentation and Ruff for linting and formatting.

## License

Apache License 2.0; see [LICENSE](LICENSE).
