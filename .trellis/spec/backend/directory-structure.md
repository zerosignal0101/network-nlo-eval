# Directory Structure

> How backend code is organized in this project.

---

## Overview

This is a **scientific computing / simulation project** for optical network evaluation. The codebase is organized around domain concepts rather than technical layers.

---

## Directory Layout

```
src/network_nlo_eval/
├── __init__.py           # Package init (can contain __version__)
├── __main__.py           # CLI entry point using click
├── core/                 # Core types, constants, spectrum grid
│   ├── types.py          # Type aliases (NDArrayFloat, NodeID, LinkKey, etc.)
│   ├── constants.py      # Physical constants
│   └── spectrum.py       # SpectrumGrid dataclass
├── network/              # Network domain
│   ├── elements.py       # Pydantic configs (FiberSpanConfig, EDFAConfig, ROADMConfig)
│   ├── topology.py       # NetworkTopology class
│   └── state.py          # NetworkState (dynamic simulation state)
├── physics/              # Physical layer models
│   ├── metrics.py        # GSNR/ASE/NLI calculations
│   └── dsp.py            # Digital signal processing
├── models/               # Mathematical models
│   ├── isrs_gn.py        # MultiBandISRSGN model
│   └── numba_kernels.py  # Numba JIT-compiled kernels
├── rwa/                  # Routing and Wavelength Assignment
│   ├── path_computation.py  # KSP path caching
│   ├── allocators.py        # RWA allocation strategies
│   └── qot_checker.py       # QoT validation
└── simulation/           # Discrete Event Simulation
    ├── engine.py         # SimulatorEngine (event-driven simulation)
    ├── traffic.py        # ServiceRequest, AllocatedService dataclasses
    └── reporter.py       # MetricsCollector, ServiceEvent
```

---

## Module Organization

### Core Principles

1. **Domain-driven organization** - Modules are grouped by domain concept (network, physics, rwa, simulation)
2. **No traditional "layers"** - This is not an API service; it's a simulation library
3. **Clear entry points** - `__main__.py` for CLI, `__init__.py` for package API

### Key Patterns

- **Type aliases** - `core/types.py` defines `NDArrayFloat`, `NodeID`, `LinkKey`, etc.
- **Config models** - `network/elements.py` has Pydantic models (`FiberSpanConfig`, `EDFAConfig`)
- **Data models** - `simulation/traffic.py` has dataclasses (`ServiceRequest`, `AllocatedService`)
- **Calculator/Validator** - `physics/`, `rwa/` contain `MultiBandISRSGN`, `QoTValidator`

---

## Naming Conventions

- Python files: `snake_case.py` (e.g., `path_computation.py`, `numba_kernels.py`)
- Classes: `PascalCase` (e.g., `SimulatorEngine`, `NetworkTopology`, `FiberSpanConfig`)
- Functions/methods: `snake_case` (e.g., `allocate()`, `generate_services()`, `verify_allocation()`)
- Type aliases: `PascalCase` suffix (e.g., `NDArrayFloat`, `NodeID`, `LinkKey`)
- Constants: `UPPER_SNAKE_CASE` (in `core/constants.py`)
- Private members: `_leading_underscore` (e.g., `_handle_arrival()`, `_get_available_channels_on_path()`)

---

## Examples

### Well-Organized Modules

- `simulation/engine.py` - Clear separation of concerns, good docstrings
- `network/elements.py` - Pydantic models with clear field descriptions
- `rwa/allocators.py` - Strategy pattern for RWA algorithms

### Adding a New Module

1. Place it in the appropriate domain directory
2. Use Pydantic `BaseModel` for configuration/data validation
3. Use `@dataclass` for plain data carriers
4. Use `numpy.typing.NDArray` type aliases from `core/types.py`
5. Add Chinese comments for domain-specific explanations

---

## Anti-Patterns

- **DO NOT** create "utils" or "helpers" directories - prefer organizing code in domain modules
- **DO NOT** use classes where simple functions suffice (e.g., pure calculation functions)
- **DO NOT** import `numpy` as `np` in type alias files - keep type aliases clean for static analysis
