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


# ── Singleton instance ─────────────────────────────────────────────────────
document_db: SageDocumentDB = _LazyDocumentDB()  # type: ignore[assignment]

