# network-nlo-eval

A Python toolkit for evaluating physical-layer quality of transmission (QoT) in
wavelength-division multiplexed optical networks. It provides nonlinear
interference models, network topology and spectrum state, QoT-aware routing and
spectrum assignment, and a discrete-event simulator.

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

grid = SpectrumGrid(
    num_channels=3, channel_spacing_hz=50e9, center_frequency_hz=193.1e12
)
fiber = FiberSpanConfig(length_km=80)
path = MultiSpanOpticalPath(fiber_configs=[fiber], edfa_configs=[EDFAConfig()])
model = create_model("ecz", grid, fiber, symbol_rate_hz=32e9)
result = model.evaluate_path(path, np.full(3, 1e-3))
print(result.snr_db)
```

Launch powers and returned signal and noise powers use watts; the example uses 1
mW per channel. Frequencies are in hertz and fiber lengths in `FiberSpanConfig`
are in kilometers. Models have different approximations, power conventions, and
applicability limits: consult their documentation and options when selecting a
model.

The [model literature audit](docs/model-literature-audit.md) maps all 18 model
entries to their sources, equations, and supported scenarios.

The [coherent PCFM2 guide](docs/pcfm2.md) describes full SCI/XCI/MCI island
integration, receiver-band PSD integration, explicit span-pair coherence, and
the opt-in `pcfm_auto` policy for choosing PCFM2 or `pcfm_h3_raman`.

Some model documentation refers to historical audit reports and calibration
scripts from the development repository. Those research artifacts are not
included here. In particular, the Zhang calibration data records the path of its
original fitting script as provenance; that script is not bundled.

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

Use a NetworkX `DiGraph` to model independent fiber, amplifier, and spectrum
state in each direction. An undirected `Graph` shares these resources between
directions. Use one edge per ordered endpoint pair.

For an engineered link, set the directed edge's `spans` attribute to an ordered
list of `{"fiber": {"length_km": ...}, "edfa": ...}` configurations. Each entry
represents one installed span. Omit `edfa` for a passive span. The destination
ROADM is applied after the last span. This form uses the fiber lengths in the
list. `H3LinkValidityProvider` evaluates the installed span chain when link
validity is enabled.

Set an EDFA's `max_output_signal_power_w` to model gain compression under WDM
loading. The limit applies to aggregate signal power across both polarizations.
The optional `compression_curve` adds gain tilt and a noise-figure penalty at
the compressed operating point. Its entries specify `backoff_db`,
`tilt_db_per_thz`, and `noise_figure_penalty_db`. Start at zero backoff and zero
corrections, then list points in increasing backoff order. The evaluator
interpolates corrections and holds the last point beyond the curve. It
normalizes the tilted gain to preserve total signal output and calculates ASE
from the resulting per-channel gain and noise figure.

An EDFA can restrict propagation with `passbands_hz`, a list of
`[lower_hz, upper_hz]` intervals. A channel passes when its full modeled
bandwidth fits within one interval. All passing channels share the amplifier's
gain compression and power limit. `input_voa_db` reduces the load before
amplification. `output_voa_db` attenuates the signal and noise after
amplification, so the power cap applies upstream of that loss. Automatic gain
compensates the fiber span loss and leaves both VOA losses in the link budget.

Under a node's `roadm` settings, `path_losses` can specify total switching loss
by ingress and egress. For example,
`{"from_degree": 101, "to_degree": 301, "loss_db": 7}` selects the path from
neighbor 101 to neighbor 301. Set `to_degree` to `null` to specify loss for a
local drop. Degrees use original topology IDs. `degree_targets` maps an outgoing
degree to a target, such as `{"301": {"power_w": 0.0005}}`. `output_target`
supplies the node default. A target specifies either `power_w` per channel or
`psd_w_per_hz` over the modeled channel bandwidth. It caps the available power
after switching loss by passive attenuation. The statistical evaluators apply
the target to signal and noise together. The existing booster applies on paths
without a target when `equalize_output_power` is enabled.

With `simulate --slot-width-ghz 12.5`, each service reserves an exact slot
interval. Its midpoint sets the carrier frequency. For Python traffic inputs,
set `ServiceRequest.symbol_rate_hz` to the carrier's baud rate. The model's
`channel_bandwidth_source` selects baud rate or reserved slot width for its
rectangular bandwidth. Distinct carriers can share the same nominal channel
index when their slot intervals are disjoint.

Per-carrier admission uses the actual load on each fiber. Through traffic
carries forward its signal and noise, and each added carrier starts at its
transmitter power. Set `from_degree` to `null` and specify `to_degree` to
configure a local add-port loss. The egress target applies to both add and
express traffic. QoT checks include carriers affected by downstream load
changes.

Uniform per-carrier routes use the model's full-route calculation. With mixed
loads, admission uses the model within each physical link and accumulates noise
powers across links. Python callers using ECZ can construct `QoTValidator`
directly. Other models can supply `carrier_model_factory`, which receives a
frequency-ordered tuple of `CarrierGeometry` objects and the slots per nominal
channel.

## Development

Run the focused test suite with `python -m pytest`. The project uses NumPy-style
Python documentation and Ruff for linting and formatting.

## License

Apache License 2.0; see [LICENSE](LICENSE).
