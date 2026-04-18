# Logging Guidelines

> How logging is done in this project.

---

## Overview

This project is a **scientific simulation library** that outputs results to files. Logging is minimal:

- No logger configuration - use `click.echo()` for CLI output
- Progress tracking via `tqdm` progress bars
- Results written as JSON files

---

## Log Levels

Since this is a batch simulation tool (not a long-running service), logging conventions are minimal:

- `click.echo()` - CLI output, progress updates
- `click.secho()` - Warnings/errors with color
- `tqdm` - Progress bars for long operations
- `logging` module - Not used in this project

---

## Structured Output

CLI output uses `click.echo()` for plain messages:

```python
# From __main__.py
click.echo(f"Starting simulation with topology: {topology_file}")
click.echo(f"Generated {len(services)} services.")
click.echo("Running simulation...")
```

Progress bars use `tqdm`:

```python
# From simulation/engine.py
with tqdm(total=len(self.event_queue), desc="Running Simulation") as pbar:
    while self.event_queue:
        # ... process events
        pbar.update(1)
```

---

## What to Log

- Simulation start/stop with key parameters
- Number of services generated
- Final metrics (blocking rate, utilization, etc.)
- Errors with context (filename, type of error)

---

## What NOT to Log

- **Debug information** - Use breakpoints or IDE debuggers instead
- **Individual event processing** - Too verbose, use progress bar instead
- **PII or secrets** - This project doesn't handle user data
- **Intermediate numerical results** - Output to JSON files instead

---

## Anti-Patterns

- **Don't use `logging.getLogger()`** - This project doesn't use the logging module
- **Don't print() directly** - Use `click.echo()` for CLI-safe output
- **Don't log in tight loops** - Use tqdm progress bar instead
- **Don't log numerical arrays** - Write to files if needed for debugging
