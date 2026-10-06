"""Shared pytest configuration for the test suite.

Ensures sibling helper modules (for example ``fg1_scenarios``, which holds the
branch-agnostic parity scenarios) are importable regardless of the active
pytest import mode or the working directory.
"""

import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))
