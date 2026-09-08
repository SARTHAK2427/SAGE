"""
SAGE/db_service.py
Process-level SageDocumentDB singleton.

Import this module from any backend component to obtain
the shared document-database instance:

    from db_service import document_db

The instance is created once at import time and reused for
the entire process lifetime.  Embedding model, Chroma client,
and artifact store are not repeatedly reloaded.

Startup diagnostics are printed/logged on first construction.
"""

from __future__ import annotations
import logging
from pathlib import Path

from sage_document_db import SageDocumentDB
from sage_document_db.config import ARTIFACTS_ROOT, CHROMA_ROOT, EMBEDDING_MODEL, EMBEDDING_DEVICE

logger = logging.getLogger(__name__)


def _create_db() -> SageDocumentDB:
    """Construct the singleton SageDocumentDB and log diagnostics."""
    logger.info("=" * 60)
    logger.info("SAGE Document Database — initializing")
    logger.info("  Artifact root : %s", ARTIFACTS_ROOT.resolve())
    logger.info("  Chroma root   : %s", CHROMA_ROOT.resolve())
    logger.info("  Embedding model: %s", EMBEDDING_MODEL)
    logger.info("  Embedding device config: %s", EMBEDDING_DEVICE)
    logger.info("=" * 60)

    db = SageDocumentDB(
        artifacts_root=ARTIFACTS_ROOT,
        chroma_root=CHROMA_ROOT,
    )

    # Log resolved embedding device after construction
    logger.info("  Embedding device resolved: %s", db._emb.device)
    logger.info("SAGE Document Database — ready")
    return db


_instance: SageDocumentDB | None = None


def get_db() -> SageDocumentDB:
    """Return the singleton SageDocumentDB, initializing it on first call."""
    global _instance
    if _instance is None:
        _instance = _create_db()
    return _instance


class _LazyDocumentDB:
    """Transparent lazy proxy to SageDocumentDB singleton."""
    def __getattr__(self, name: str):
        return getattr(get_db(), name)

    def __repr__(self) -> str:
        return f"<LazyDocumentDB instance={_instance is not None}>"


# ── Singleton instances ───────────────────────────────────────────────────
document_db: SageDocumentDB = _LazyDocumentDB()  # type: ignore[assignment]


# ── MemoryManager Singleton ─────────────────────────────────────────────────

_mem_instance: Any = None


def get_memory_manager(db: SageDocumentDB | None = None) -> Any:
    """Return the singleton MemoryManager, initializing it on first call.

    Reuses the existing embedding model and Chroma persistent directory.
    """
    global _mem_instance
    if _mem_instance is None:
        from sage_memory.config import (
            CHROMA_ROOT as MEMORY_CHROMA_ROOT,
            MEMORY_COLD_COLLECTION,
            MEMORY_HOT_COLLECTION,
            MEMORY_STORE_PATH,
        )
        from sage_memory.memory_index import MemoryChromaStore
        from sage_memory.memory_manager import MemoryManager
        from sage_memory.memory_store import MemoryStore
        from sage_document_db.embeddings import EmbeddingService

        if db is None:
            db = get_db()

        emb = getattr(db, "_emb", None) or EmbeddingService()
        store = MemoryStore(MEMORY_STORE_PATH)
        index = MemoryChromaStore(
            chroma_root=MEMORY_CHROMA_ROOT,
            embedding_service=emb,
            hot_collection=MEMORY_HOT_COLLECTION,
            cold_collection=MEMORY_COLD_COLLECTION,
        )
        _mem_instance = MemoryManager(store=store, index=index)
        try:
            _mem_instance.sync_indexes()
        except Exception as exc:
            logger.warning("Memory startup sync check failed: %s", exc)
    return _mem_instance


class _LazyMemoryManager:
    """Transparent lazy proxy to MemoryManager singleton."""
    def __getattr__(self, name: str):
        return getattr(get_memory_manager(), name)

    def __repr__(self) -> str:
        return f"<LazyMemoryManager instance={_mem_instance is not None}>"


memory_manager: Any = _LazyMemoryManager()

