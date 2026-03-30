"""tests/conftest.py — Session-scoped fixtures shared across all validation tests."""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from tests.validation.shared import RealWorldData


@pytest.fixture(scope="session")
def rw_data() -> RealWorldData:
    """Load OSHA bulk data and build pre/post-cutoff splits once per test session."""
    return RealWorldData.get()
