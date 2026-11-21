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

## Receiver feasibility

Attach a `TransceiverMode` to a `ServiceRequest` to evaluate a specific optical
carrier using the route's actual load. Both first-fit allocators preserve the
mode on the allocation and recheck it when later admissions affect the route.

```python
from network_nlo_eval.core.transceiver import ReceiverPenaltyCurve, TransceiverMode
from network_nlo_eval.simulation.traffic import ServiceRequest

mode = TransceiverMode(
    name="100G",
    symbol_rate_hz=32e9,
    roll_off=0.15,
    min_slot_width_hz=37.5e9,
    min_gsnr_db=12,
    min_osnr_db=16,
    tx_osnr_db=40,
    rx_power_min_dbm=-12,
    rx_power_max_dbm=0,
    pmd_penalty=ReceiverPenaltyCurve(points=((0, 0), (10, 1.5))),
)
request = ServiceRequest(1, 0, 1, 0, 1, 100, 10, 1e-3, transceiver_mode=mode)
```

Flex first-fit reserves at least the mode width rounded up to a whole number of
slots and honors any larger `n_slots` requested. Fixed first-fit requires a mode
that fits in one channel. For direct flex calls to `verify_allocation`, pass the
actual reservation as `n_slots` and `first_slot`.

The path must meet the request's line SNR threshold and every configured mode
limit. Receiver GSNR combines modeled ASE, nonlinear noise, and transmitter
noise in the evaluator's bandwidth. OSNR combines ASE and transmitter noise at
`osnr_reference_bandwidth_hz`, which defaults to the 12.5 GHz planning
convention for 0.1 nm. ASE and transmitter noise are treated as spectrally flat
when changing bandwidth. The line noise budget includes any configured ROADM
filtering penalty. Receiver and model-validity penalties apply to the GSNR
margin. Power limits apply to total dual-polarization signal power after the
destination drop port.

Signed fiber dispersion is summed at the carrier wavelength. Fiber PMD and
configured amplifier and ROADM PMD and PDL are combined in quadrature. The
mode's CD, PMD, and PDL curves give additive GSNR penalties, with values beyond
the last calibrated point treated as infeasible.

`verify_allocation` returns the usual acceptance flag and line SNR array.
`QoTValidator.last_receiver_feasibility` contains the mode-aware receiver
reports reached during the most recent verification, including failed checks.
The reports are keyed by service ID and include the limiting margin in dB.

### Fixed-grid loading

On mixed routes with power-changing equipment, fixed-grid admission applies add,
express, and drop settings by service and uses the propagated ingress power to
compute each downstream link's load. The solver retains the model's nominal
frequency grid and checks downstream incumbents affected by the new load.
Uniform fixed-grid routes and loss-compensated fixed-grid networks can use the
existing link-budget path.

## Equipment profiles

Use `EquipmentProfiles` to share physical settings across links, installed
spans, and nodes. Pass a catalog as `equipment_profiles=` to `NetworkTopology`
in Python. In a NetworkX node-link JSON topology, put it under
`graph.equipment_profiles`.

```json
{
  "fiber": { "smf": { "attenuation_db_km_ref": 0.19 } },
  "edfa": { "line": { "noise_figure_db": 4.5 } },
  "roadm": { "switch": { "insertion_loss_db": 3 } }
}
```

An edge can reference these entries with
`"fiber": {"profile": "smf", "length_km": 80}` and `"edfa": "line"`. A node can
use `"roadm": "switch"`. Use the same reference syntax inside `spans` on
directed links. An omitted or null span `edfa` keeps that span passive.

For links and nodes, profile settings override the topology defaults and inline
settings take precedence. Edge `length_km` or `weight` takes precedence over a
profile's length, and an inline fiber `length_km` takes precedence over either.
Profiles resolve at topology construction and exports contain the resolved
equipment settings.

## Command line

Use `network-nlo-eval models` to list registered models and their options.
`network-nlo-eval simulate --service-num 10` runs the bundled example topology;
`network-nlo-eval compare --help` and `network-nlo-eval bench --help` describe
path comparison and benchmarking. All commands provide `--help`.

See the [Python API](api.md) for the model interface and network components.

See the [model literature audit](model-literature-audit.md) for all registered
models, their paper equations, supported scenarios, and implementation variants.
