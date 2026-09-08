"""
sage_memory/config.py
Global configuration for the SAGE hot/cold memory layer.

Follows the same env-overridable constant style as sage_document_db/config.py.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directory — anchored to the unified SAGE runtime root
# (parent of sage_memory/)
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = _PACKAGE_DIR.parent  # SAGE/ root

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------
MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED", "1") == "1"

# ---------------------------------------------------------------------------
# Canonical store paths — JSON-on-disk is source of truth
# ---------------------------------------------------------------------------
MEMORY_STORE_PATH = Path(
    os.environ.get("MEMORY_STORE_PATH", str(BASE_DIR / "data" / "memory"))
)
MEMORY_HOT_DIR = MEMORY_STORE_PATH / "hot"
MEMORY_COLD_DIR = MEMORY_STORE_PATH / "cold"

# ---------------------------------------------------------------------------
# Chroma — shared root with document DB; separate collections
# ---------------------------------------------------------------------------
CHROMA_ROOT = Path(os.environ.get("SAGE_CHROMA_ROOT", str(BASE_DIR / "chroma_db")))
MEMORY_HOT_COLLECTION = os.environ.get("MEMORY_HOT_COLLECTION", "sage_memory_hot")
MEMORY_COLD_COLLECTION = os.environ.get("MEMORY_COLD_COLLECTION", "sage_memory_cold")

# ---------------------------------------------------------------------------
# Embedding — reuse the document DB model; do not instantiate separately
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.environ.get(
    "SAGE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ---------------------------------------------------------------------------
# Retrieval / summarization defaults
# ---------------------------------------------------------------------------
MEMORY_DEFAULT_TOP_K = int(os.environ.get("MEMORY_DEFAULT_TOP_K", "5"))
MEMORY_HOT_SUMMARY_THRESHOLD = int(
    os.environ.get("MEMORY_HOT_SUMMARY_THRESHOLD", "20")
)
