# Network NLO Eval

Physical-layer transmission quality (QoT) evaluation for multi-band WDM optical networks, with QoT-aware routing and wavelength assignment (RWA) and discrete-event simulation.

## Features

- **Multi-band spectrum** — configurable WDM grid (C+L+S bands) with arbitrary channel count, spacing, and center frequency.
- **ISRS-GN model** — Inter-channel Stimulated Raman Scattering Generalized Noise (ISRS-GN) evaluator, computing per-channel SPM, XPM, and ASE noise variances, accelerated with Numba JIT kernels.
- **Network element modeling** — Validated per-link fiber/EDFA overrides and detailed per-node ROADMs: super-Gaussian passband/order, express/add/drop loss, blocked channels, booster NF/equalization, and Raman-pump definitions.
- **Topology management** — NetworkX-based topology loader supporting arbitrary fiber network topologies with per-link length and per-node element configs.
- **QoT-aware RWA** — K-Shortest Paths (KSP) with First-Fit wavelength allocation, validated against physical-layer SNR degradation on both new and existing services.
- **Discrete-event simulation** — Poisson arrival traffic with exponential holding times, event-driven engine with heapq scheduling, progress reporting, and configurable load.
- **Metrics collection** — blocking rate, wavelength utilization, average hop count, throughput, and fragmentation index.
- **Topology-aware waveform integration** — the analytical ISRS-GN model and `physics-nlo-eval` Manakov/SSFM engine consume the same ordered fiber → amplifier → ROADM route contract.
- **Cross-model comparison** — `TopologyQoTComparator` and the `compare` CLI report statistical SPM/XPM/ASE/SNR beside waveform BER/EVM/SNR and their per-channel SNR delta.
- **CLI** — Click-based `simulate` command wiring topology loading, spectrum definition, RWA, simulation engine, and metrics export into a single invocation.

## Requirements

- Python >= 3.11, <= 3.13
- Dependencies: click, numpy, scipy, numba, networkx, pydantic (see [pyproject.toml](pyproject.toml) for versions)
- Optional numerical comparison: install the sibling `physics-nlo-eval` package in the same environment.

## Installation

```console
$ pip install network-nlo-eval
```

Or with Poetry:

```console
$ poetry install
```

## Usage

### CLI

```console
$ network-nlo-eval simulate --topology-file assets/example_pan_europe_network.json \
    --service-num 500 --avg-arrival-interval 10 --avg-holding-time 400 \
    --num-channels 80 --channel-spacing-ghz 50 --center-freq-thz 193.1 \
    --max-ksp-paths 5 --output-dir results
```

Compare an analytical route result, optionally adding the numerical waveform run:

```console
$ network-nlo-eval compare --topology-file assets/example_physical_topology.json \
    --path 101,102,103 \
    --channel 39 --channel 40 --launch-power-dbm 0 --statistical-only

$ network-nlo-eval compare ... --waveform --num-symbols 2048 --output-file results/comparison.json
```

Topology nodes may contain `roadm` and `edfa` objects; edges may contain `fiber`
and `edfa` objects. Legacy `weight` remains the fiber length in km. For example:

```json
{
  "id": 101,
  "roadm": {
    "insertion_loss_db": 5.0,
    "passband_hz": 50000000000.0,
    "filter_order": 4,
    "express_attenuation_db": 0.5,
    "add_drop_loss_db": 1.0,
    "filtering_penalty_db": 0.2,
    "booster_noise_figure_db": 5.0,
    "equalize_output_power": true,
    "blocked_channels": []
  }
}
```

Reference planes are explicit: channel launch power is total dual-polarization
power at each span input. A span EDFA restores span loss when `target_gain_db`
is null. At a ROADM, signal and accumulated noise see the same deterministic
loss; an optional booster restores power and contributes ASE. The statistical
filtering penalty represents cascade narrowing, while the waveform engine uses
the configured super-Gaussian transfer function directly.

## Project structure

```
src/network_nlo_eval/
├── core/          # Physical constants, type aliases, SpectrumGrid
├── models/        # ISRS-GN evaluator and Numba JIT kernels
├── network/       # Fiber/EDFA/ROADM configs, NetworkTopology, NetworkState
├── physics/       # SNR/BER/EVM utilities, PyNLO-based QoT evaluator
├── rwa/           # PathCache, QoTValidator, KSPFirstFitAllocator
└── simulation/    # Event engine, traffic generation, metrics collection
```

## License

GPL 3.0. See [LICENSE](LICENSE).
