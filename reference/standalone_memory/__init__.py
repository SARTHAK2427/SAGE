"""
standalone_memory: The Multi-Tier Semantic Memory Engine for AI Agents.
Provides Hot, Cold, Global, and Recent Chat Memory with ChromaDB and SQLite/PostgreSQL.
"""

from .config import *
from .memory_engine import (
    MemoryEngine,
    memory_engine,
    ALLOWED_CATEGORIES,
    ALLOWED_STATUSES,
    ALLOWED_TIERS,
    validate_category,
)
from .vector_index import (
    MemoryVectorIndex,
    memory_vector_index,
    EmbeddingService,
)
from .tools import (
    memory_store_hot,
    memory_search_hot,
    memory_store_cold,
    memory_search_cold,
    memory_promote_to_cold,
    memory_get_context,
    memory_decay_check,
    memory_delete,
    TOOL_DEFINITIONS,
)

__all__ = [
    "MemoryEngine",
    "memory_engine",
    "MemoryVectorIndex",
    "memory_vector_index",
    "EmbeddingService",
    "memory_store_hot",
    "memory_search_hot",
    "memory_store_cold",
    "memory_search_cold",
    "memory_promote_to_cold",
    "memory_get_context",
    "memory_decay_check",
    "memory_delete",
    "TOOL_DEFINITIONS",
    "ALLOWED_CATEGORIES",
]

__version__ = "1.0.0"
