"""
SAGE test fixtures — shared across all test modules.

Uses SAGE_MOCK_MODE=1 environment and temporary artifact roots
so tests never pollute production data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

# Ensure the SAGE runtime root is on sys.path
_SAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_SAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAGE_ROOT))

# Set mock mode at collection time so module-level imports run fast
os.environ["SAGE_MOCK_MODE"] = "1"


@pytest.fixture(autouse=True)
def _mock_mode_env(monkeypatch):
    """Ensure all tests run with SAGE_MOCK_MODE=1."""
    monkeypatch.setenv("SAGE_MOCK_MODE", "1")


@pytest.fixture()
def tmp_artifact_root(tmp_path):
    """Provide a temporary artifact root for tests.

    Avoids polluting the real artifacts/ directory.
    Cleaned up automatically by pytest's tmp_path.
    """
    art_root = tmp_path / "artifacts"
    art_root.mkdir()
    return art_root


@pytest.fixture()
def tmp_chroma_root(tmp_path):
    """Provide a temporary chroma root for tests."""
    chroma_root = tmp_path / "chroma_db"
    chroma_root.mkdir()
    return chroma_root
