"""
Root conftest.py — ensures pyserep is importable from the project root
regardless of whether it is installed.

Also registers project-wide pytest marks.
"""

import sys
import os

# Ensure the project root is on sys.path so `import pyserep` works
# even without `pip install -e .`
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with -m 'not slow')"
    )
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests"
    )
