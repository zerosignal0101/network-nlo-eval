# Quality Guidelines

> Code quality standards for backend development.

---

## Overview

This is a **scientific computing project** using Python with NumPy, Numba, and Pydantic. Quality standards follow:

- `ruff` for linting and formatting
- `mypy` for type checking
- `pytest` for testing
- `pre-commit` hooks for automated checks

---

## Code Standards

### Linting (ruff)

Configuration from `pyproject.toml`:

```toml
[tool.ruff]
line-length = 120
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "B", "I", "UP", "D", "S"]
ignore = ["D100", "D104", "D106", "D400", "F841"]
```

### Type Checking (mypy)

Configuration from `pyproject.toml`:

```toml
[tool.mypy]
strict = true
warn_unreachable = true
```

### Docstrings

Use **numpy-style** docstrings as configured:

```toml
[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

Example from `simulation/engine.py`:

```python
class SimulatorEngine:
    """离散事件仿真主引擎。

    按时间顺序处理业务到达和离开事件，调用 RWA 算法进行资源分配和释放。
    """

    def run(self) -> None:
        """运行仿真，直到事件队列为空。"""
        ...
```

---

## Forbidden Patterns

1. **Don't use `from numpy import *`** - Always `import numpy as np`
2. **Don't leave unused variables** - Causes `F841` lint error
3. **Don't use `print()`** - Use `click.echo()` for CLI output
4. **Don't create circular imports** - Keep modules independent
5. **Don't use mutable default arguments** - Use `None` and initialize in body
6. **Don't skip type hints** - Required for `mypy strict` mode

---

## Required Patterns

1. **Use Pydantic for configuration** - `BaseModel` with `Field()` descriptions
2. **Use dataclasses for data carriers** - With `from dataclasses import dataclass`
3. **Use type aliases** - Define in `core/types.py` (e.g., `NDArrayFloat`)
4. **Use numpy.typing.NDArray** - For array type hints
5. **Use Chinese comments** - For domain-specific explanations (as seen in codebase)
6. **Use numpy constants** - `np.pi`, `np.e` instead of hardcoded values

---

## Testing Requirements

- **Coverage target**: 100% (see `pyproject.toml`)
- **Test framework**: `pytest`
- **CLI testing**: Use `click.testing.CliRunner`

Example from `tests/test_main.py`:

```python
from click.testing import CliRunner

def test_main_succeeds(runner: CliRunner) -> None:
    """It exits with a status code of zero."""
    result = runner.invoke(__main__.main, "--help")
    assert result.exit_code == 0
```

---

## Pre-Commit Hooks

The project uses `pre-commit` with these checks:

```yaml
repos:
  - repo: local
    hooks:
      - id: check-added-large-files
      - id: check-toml
      - id: check-yaml
      - id: end-of-file-fixer
      - id: trailing-whitespace
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

---

## Code Review Checklist

When reviewing code, check:

- [ ] All new functions have docstrings (numpy style)
- [ ] Type hints are present and correct
- [ ] No `print()` statements (use `click.echo()`)
- [ ] No unused variables or imports
- [ ] Tests pass with `pytest`
- [ ] Type checking passes with `mypy`
- [ ] Linting passes with `ruff`
- [ ] New domain modules follow existing patterns
