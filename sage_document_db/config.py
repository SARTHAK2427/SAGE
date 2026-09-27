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
DATA_ROOT = Path(os.environ.get("SAGE_DATA_ROOT", str(BASE_DIR / "data"))).expanduser().resolve()
DOCUMENTS_ROOT = DATA_ROOT / "documents"
ARTIFACTS_ROOT = Path(os.environ.get("SAGE_ARTIFACTS_ROOT", str(DOCUMENTS_ROOT / "artifacts"))).expanduser().resolve()
# BGE-M3 vectors are 1024-dimensional and cannot share a collection with the
# legacy MiniLM (384-dimensional) vectors.  A new default keeps the old index
# intact. SAGE_CHROMA_ROOT is kept as a compatibility fallback; new installs
# should use SAGE_DOCUMENT_CHROMA_ROOT to avoid sharing storage with memory.
CHROMA_ROOT = Path(os.environ.get("SAGE_DOCUMENT_CHROMA_ROOT", os.environ.get("SAGE_CHROMA_ROOT", str(DOCUMENTS_ROOT / "chroma_bge_m3")))).expanduser().resolve()

# ---------------------------------------------------------------------------
# Chroma collections
# ---------------------------------------------------------------------------
SOURCE_COLLECTION = "sage_source"
DERIVED_COLLECTION = "sage_derived"

# ---------------------------------------------------------------------------
# Embedding model — BGE-M3 is the canonical document RAG encoder.
#
# Large LLMs/VLMs:
#     llama.cpp + CUDA (managed by model_manager)
#
# Document embeddings:
#     CPU by default. Flash reserves the 8 GB GPU for the resident
#     Gemma/Qwen pair; opting into CUDA here is explicit through .env.
#     Do NOT route embeddings through llama.cpp.
# ---------------------------------------------------------------------------
# CPU is intentional: Flash owns the 8 GB GPU.  CUDA is opt-in for machines
# with VRAM left after Gemma/Qwen have been loaded.
EMBEDDING_MODEL = os.environ.get("SAGE_RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cpu")
EMBED_BATCH_SIZE = int(os.environ.get("SAGE_RAG_EMBED_BATCH_SIZE", "16"))
EMBEDDING_DIMENSION = 1024

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_TARGET_TOKENS = int(os.environ.get("SAGE_RAG_CHUNK_TARGET_TOKENS", "600"))
CHUNK_OVERLAP_TOKENS = int(os.environ.get("SAGE_RAG_CHUNK_OVERLAP_TOKENS", "90"))

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
DEFAULT_RAG_TOP_K = 5
RAG_CONFIDENCE_THRESHOLD = float(os.environ.get("SAGE_RAG_THRESHOLD", "0.60"))

# Cross-encoder reranking is deliberately opt-in and CPU-first.  It is loaded
# only after SAGE_RERANKER_ENABLED is enabled, so the default costs no RAM.
RERANKER_MODEL = os.environ.get("SAGE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RERANKER_DEVICE = os.environ.get("RERANKER_DEVICE", "cpu")
RERANKER_ENABLED = os.environ.get("SAGE_RERANKER_ENABLED", "false").lower() in {"1", "true", "yes"}
RERANKER_CANDIDATE_K = int(os.environ.get("SAGE_RERANKER_CANDIDATE_K", "20"))
RERANK_BATCH_SIZE = int(os.environ.get("SAGE_RERANK_BATCH_SIZE", "8"))
