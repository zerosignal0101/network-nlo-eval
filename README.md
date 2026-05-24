# Network NLO Eval

Physical-layer transmission quality (QoT) evaluation for multi-band WDM optical networks, with QoT-aware routing and wavelength assignment (RWA) and discrete-event simulation.

## Features

- **Multi-band spectrum** — configurable WDM grid (C+L+S bands) with arbitrary channel count, spacing, and center frequency.
- **ISRS-GN model** — Inter-channel Stimulated Raman Scattering Generalized Noise (ISRS-GN) evaluator, computing per-channel SPM, XPM, and ASE noise variances, accelerated with Numba JIT kernels.
- **Network element modeling** — Pydantic-based configuration for fiber spans (attenuation, dispersion, nonlinear coefficient), EDFAs (gain, noise figure), and ROADMs (insertion loss, filtering penalty).
- **Topology management** — NetworkX-based topology loader supporting arbitrary fiber network topologies with per-link length and per-node element configs.
- **QoT-aware RWA** — K-Shortest Paths (KSP) with First-Fit wavelength allocation, validated against physical-layer SNR degradation on both new and existing services.
- **Discrete-event simulation** — Poisson arrival traffic with exponential holding times, event-driven engine with heapq scheduling, progress reporting, and configurable load.
- **Metrics collection** — blocking rate, wavelength utilization, average hop count, throughput, and fragmentation index.
- **PyNLO integration** — physical-layer evaluation via Manakov solver in `network_nlo_eval.physics.qot_checker`.
- **CLI** — Click-based `simulate` command wiring topology loading, spectrum definition, RWA, simulation engine, and metrics export into a single invocation.

## Requirements

- Python >= 3.11, <= 3.13
- Dependencies: click, numpy, scipy, numba, networkx, pydantic (see [pyproject.toml](pyproject.toml) for versions)

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
