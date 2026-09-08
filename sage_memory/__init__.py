"""
sage_memory/__init__.py
SAGE Hot/Cold Memory — package entry point.

Canonical JSON store is source of truth; Chroma is a rebuildable search index.
Gemma is the sole semantic controller; Python executes deterministically.
"""

from .config import (
    CHROMA_ROOT,
    MEMORY_COLD_COLLECTION,
    MEMORY_COLD_DIR,
    MEMORY_DEFAULT_TOP_K,
    MEMORY_ENABLED,
    MEMORY_HOT_COLLECTION,
    MEMORY_HOT_DIR,
    MEMORY_HOT_SUMMARY_THRESHOLD,
    MEMORY_STORE_PATH,
)
from .memory_index import MemoryChromaStore, MemoryIndexError
from .memory_manager import MemoryManager, MemoryManagerError
from .memory_models import (
    KNOWN_CATEGORIES,
    MemoryRecord,
    MemorySearchResult,
    MemoryType,
    MemoryValidationError,
    make_memory_id,
    validate_memory_record,
)
from .memory_store import MemoryStore, MemoryStoreError
from .memory_summarizer import MemorySummarizer

__all__ = [
    "CHROMA_ROOT",
    "KNOWN_CATEGORIES",
    "MEMORY_COLD_COLLECTION",
    "MEMORY_COLD_DIR",
    "MEMORY_DEFAULT_TOP_K",
    "MEMORY_ENABLED",
    "MEMORY_HOT_COLLECTION",
    "MEMORY_HOT_DIR",
    "MEMORY_HOT_SUMMARY_THRESHOLD",
    "MEMORY_STORE_PATH",
    "MemoryChromaStore",
    "MemoryIndexError",
    "MemoryManager",
    "MemoryManagerError",
    "MemoryRecord",
    "MemorySearchResult",
    "MemoryStore",
    "MemoryStoreError",
    "MemorySummarizer",
    "MemoryType",
    "MemoryValidationError",
    "make_memory_id",
    "validate_memory_record",
]
