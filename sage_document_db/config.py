"""
sage_document_db/config.py
Global configuration for the SAGE document database layer.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base directory — anchored to the unified SAGE runtime root
# (parent of sage_document_db/)
# ---------------------------------------------------------------------------
_PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = _PACKAGE_DIR.parent  # SAGE/ root

# ---------------------------------------------------------------------------
# Paths — absolute, not CWD-dependent
# ---------------------------------------------------------------------------
ARTIFACTS_ROOT = Path(os.environ.get("SAGE_ARTIFACTS_ROOT", str(BASE_DIR / "artifacts")))
CHROMA_ROOT = Path(os.environ.get("SAGE_CHROMA_ROOT", str(BASE_DIR / "chroma_db")))

# ---------------------------------------------------------------------------
# Chroma collections
# ---------------------------------------------------------------------------
SOURCE_COLLECTION = "sage_source"
DERIVED_COLLECTION = "sage_derived"

# ---------------------------------------------------------------------------
# Embedding model — frozen for MVP
#
# Large LLMs/VLMs:
#     llama.cpp + CUDA (managed by model_manager)
#
# MiniLM embeddings:
#     SentenceTransformers/PyTorch + CUDA when available, CPU fallback
#     Do NOT route MiniLM through llama.cpp.
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "auto")
EMBED_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_TARGET_TOKENS = 180
CHUNK_OVERLAP_TOKENS = 40

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_RAG_TOP_K = 5
