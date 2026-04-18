# Error Handling

> How errors are handled in this project.

---

## Overview

This project is a **simulation library** that runs batch computations. Error handling focuses on:

1. **Input validation** via Pydantic models
2. **Runtime assertions** via Python exceptions
3. **Graceful CLI failures** with user-friendly error messages

---

## Error Types

### Pydantic Validation Errors

Used for invalid configuration or input data. These are caught at the boundary and reported clearly.

**Example** from `network/elements.py`:

```python
class FiberSpanConfig(BaseModel):
    """单跨段光纤的静态配置参数。"""

    length_km: float = Field(..., description="光纤跨段长度 (km)")
    attenuation_db_km_ref: float = Field(0.2, description="衰减系数 (dB/km)")
```

### Dataclass Validation

Used for internal data structures. The project uses assertions rather than exception types.

**Example** from `simulation/engine.py`:

```python
is_success, allocated_service = self.allocator.allocate(service_request, self.network_state)

if is_success:
    if allocated_service is None:  # 理论上不应该发生
        raise ValueError("Allocator returned success but allocated_service is None.")
```

---

## Error Handling Patterns

### 1. Input Validation at Boundaries

Validate all input at the entry point using Pydantic:

```python
# From __main__.py
@click.option("-t", "--topology-file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def simulate(topology_file: Path, ...):
    try:
        with open(topology_file, encoding="utf-8") as f:
            network_obj = json.load(f)
    except Exception as e:
        click.secho(f"Error loading topology file: {e}", fg="red")
        return
```

### 2. Assertion for Internal Invariants

Use `raise ValueError` for internal state that should never occur:

```python
# From simulation/engine.py
if allocated_service is None:
    raise ValueError("Allocator returned success but allocated_service is None.")
```

### 3. Graceful CLI Degradation

For non-critical errors in CLI, print error and continue rather than crashing:

```python
click.secho(f"Error loading topology file: {e}", fg="red")
return
```

---

## API Error Responses

This is a **batch simulation tool**, not an API server. There are no HTTP error responses.

Instead:

- Simulation failures are logged to output files
- CLI commands return non-zero exit codes on critical failures
- Metrics collector records `AllocationFailure` events with reason strings

---

## Common Mistakes

1. **Don't swallow exceptions silently** - Always either re-raise or log the error
2. **Don't use bare `except:`** - Catch specific exceptions (`json.JSONDecodeError`, `FileNotFoundError`, etc.)
3. **Don't create custom exception classes** unless truly needed - prefer standard `ValueError`, `TypeError`
4. **Don't forget `encoding="utf-8"`** when opening files - hardcoded errors on Windows

---

## Testing Error Handling

- Test that invalid config raises `ValidationError`
- Test that missing files raise `FileNotFoundError`
- Test that corrupted JSON raises `json.JSONDecodeError`
