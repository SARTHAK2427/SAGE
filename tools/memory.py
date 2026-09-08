"""
SAGE/tools/memory.py
Tool adapters wrapping the SAGE Hot/Cold Memory subsystem.

These wrappers expose memory capabilities through the generic dispatcher
without leaking internal objects, Chroma internals, or filesystem paths to the agent.

Architectural rules:
    - Accept plain serializable arguments
    - Call MemoryManager
    - Convert dataclasses/records into JSON-safe dicts
    - Return bounded output shaped {"status": "success"|"error", ...}
    - Never leak filesystem paths or internal memory objects
    - Never autonomously call another registered Sage tool
    - Gemma 4B is the sole semantic controller; this code executes deterministically
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from sage_document_db.utils import iso_now
from sage_memory.config import MEMORY_DEFAULT_TOP_K
from sage_memory.memory_models import MemoryRecord, make_memory_id

logger = logging.getLogger(__name__)


def _dataclass_to_dict(obj: Any) -> dict:
    """Convert a dataclass or object to a JSON-safe dict, stripping private attributes."""
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = dataclasses.asdict(obj)
        return {k: v for k, v in d.items() if v is not None and not k.startswith("_")}
    if isinstance(obj, dict):
        return {k: v for k, v in obj.items() if not k.startswith("_")}
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if v is not None and not k.startswith("_")}
    return {"value": str(obj)}


# ─── 1. Hot Memory Search ──────────────────────────────────────────────────────

def tool_memory_search_hot(
    manager: Any,
    query: str = "",
    session_id: str = "",
    top_k: int = MEMORY_DEFAULT_TOP_K,
    category: str | None = None,
    is_summary: bool | None = None,
    text: str | None = None,
) -> dict:
    """Selective session-scoped semantic search over hot memory.

    Args:
        manager:    MemoryManager instance (injected at registration)
        query:      Search query string
        session_id: Target session identifier (required for hot memory)
        top_k:      Maximum results to return (defaults to MEMORY_DEFAULT_TOP_K)
        category:   Optional category filter
        is_summary: Optional summary flag filter

    Returns:
        {"status": "success", "session_id": str, "results": [...], "count": int}
    """
    if not query and text:
        query = text
    if not session_id or not str(session_id).strip():
        return {
            "status": "error",
            "error": "session_id is required for hot memory search",
            "session_id": session_id or "",
            "results": [],
            "count": 0,
        }

    if top_k is None:
        top_k = MEMORY_DEFAULT_TOP_K

    try:
        results = manager.search_hot_memory(
            query=query,
            session_id=session_id,
            top_k=top_k,
            category=category,
            is_summary=is_summary,
        )
        serialized = [_dataclass_to_dict(r) for r in results]
        return {
            "status": "success",
            "session_id": session_id,
            "results": serialized,
            "count": len(serialized),
        }
    except Exception as exc:
        logger.warning("tool_memory_search_hot failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "session_id": session_id,
            "results": [],
            "count": 0,
        }


# ─── 2. Cold Memory Search ─────────────────────────────────────────────────────

def tool_memory_search_cold(
    manager: Any,
    query: str = "",
    top_k: int = MEMORY_DEFAULT_TOP_K,
    category: str | None = None,
    is_summary: bool | None = None,
    text: str | None = None,
) -> dict:
    """Selective semantic search over durable cross-session cold memory.

    Args:
        manager:    MemoryManager instance (injected at registration)
        query:      Search query string
        top_k:      Maximum results to return (defaults to MEMORY_DEFAULT_TOP_K)
        category:   Optional category filter
        is_summary: Optional summary flag filter

    Returns:
        {"status": "success", "results": [...], "count": int}
    """
    if not query and text:
        query = text
    if not query or not str(query).strip():
        return {
            "status": "error",
            "error": "query is required for cold memory search",
            "results": [],
            "count": 0,
        }

    if top_k is None:
        top_k = MEMORY_DEFAULT_TOP_K

    try:
        results = manager.search_cold_memory(
            query=query,
            top_k=top_k,
            category=category,
            is_summary=is_summary,
        )
        serialized = [_dataclass_to_dict(r) for r in results]
        return {
            "status": "success",
            "results": serialized,
            "count": len(serialized),
        }
    except Exception as exc:
        logger.warning("tool_memory_search_cold failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "results": [],
            "count": 0,
        }


# ─── 3. Hot Memory Store ───────────────────────────────────────────────────────

def tool_memory_store_hot(
    manager: Any,
    content: str = "",
    session_id: str = "",
    category: str = "task",
    importance: float = 0.5,
    confidence: float = 0.5,
    source: str = "agent_call",
    text: str | None = None,
) -> dict:
    """Store a session-scoped hot memory record.

    Args:
        manager:    MemoryManager instance
        content:    Memory content text
        session_id: Required session identifier
        category:   Category label (e.g. task, decision, instruction)
        importance: Numeric importance in [0.0, 1.0]
        confidence: Numeric confidence in [0.0, 1.0]
        source:     Originating source tag

    Returns:
        {"status": "success", "memory": dict, "memory_id": str}
    """
    if not content and text:
        content = text
    if not content or not str(content).strip():
        return {
            "status": "error",
            "error": "content is required for hot memory store",
            "memory_id": None,
        }
    if not session_id or not str(session_id).strip():
        return {
            "status": "error",
            "error": "session_id is required for hot memory store",
            "memory_id": None,
        }

    try:
        imp = float(importance)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error": f"importance must be numeric, got {importance!r}",
            "memory_id": None,
        }
    if imp < 0.0 or imp > 1.0:
        return {
            "status": "error",
            "error": f"importance must be in [0.0, 1.0], got {importance}",
            "memory_id": None,
        }

    try:
        conf = float(confidence)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error": f"confidence must be numeric, got {confidence!r}",
            "memory_id": None,
        }
    if conf < 0.0 or conf > 1.0:
        return {
            "status": "error",
            "error": f"confidence must be in [0.0, 1.0], got {confidence}",
            "memory_id": None,
        }

    try:
        now = iso_now()
        record = MemoryRecord(
            memory_id=make_memory_id(),
            content=content.strip(),
            memory_type="hot",
            category=category,
            source=source,
            session_id=session_id.strip(),
            created_at=now,
            updated_at=now,
            importance=imp,
            confidence=conf,
            is_summary=False,
            parent_memory_ids=[],
        )
        persisted = manager.add_hot_memory(record)
        return {
            "status": "success",
            "memory": persisted.to_dict(),
            "memory_id": persisted.memory_id,
        }
    except Exception as exc:
        logger.warning("tool_memory_store_hot failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "memory_id": None,
        }


# ─── 4. Cold Memory Store ──────────────────────────────────────────────────────

def tool_memory_store_cold(
    manager: Any,
    content: str = "",
    category: str = "preference",
    session_id: str | None = None,
    importance: float = 0.5,
    confidence: float = 0.5,
    source: str = "agent_call",
    text: str | None = None,
) -> dict:
    """Store a durable cross-session cold memory record.

    Args:
        manager:    MemoryManager instance
        content:    Memory content text
        category:   Category label (e.g. personal, preference, project)
        session_id: Optional session ID indicating origin
        importance: Numeric importance in [0.0, 1.0]
        confidence: Numeric confidence in [0.0, 1.0]
        source:     Originating source tag

    Returns:
        {"status": "success", "memory": dict, "memory_id": str}
    """
    if not content and text:
        content = text
    if not content or not str(content).strip():
        return {
            "status": "error",
            "error": "content is required for cold memory store",
            "memory_id": None,
        }

    try:
        imp = float(importance)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error": f"importance must be numeric, got {importance!r}",
            "memory_id": None,
        }
    if imp < 0.0 or imp > 1.0:
        return {
            "status": "error",
            "error": f"importance must be in [0.0, 1.0], got {importance}",
            "memory_id": None,
        }

    try:
        conf = float(confidence)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error": f"confidence must be numeric, got {confidence!r}",
            "memory_id": None,
        }
    if conf < 0.0 or conf > 1.0:
        return {
            "status": "error",
            "error": f"confidence must be in [0.0, 1.0], got {confidence}",
            "memory_id": None,
        }

    try:
        now = iso_now()
        record = MemoryRecord(
            memory_id=make_memory_id(),
            content=content.strip(),
            memory_type="cold",
            category=category,
            source=source,
            session_id=session_id.strip() if session_id and session_id.strip() else None,
            created_at=now,
            updated_at=now,
            importance=imp,
            confidence=conf,
            is_summary=False,
            parent_memory_ids=[],
        )
        persisted = manager.add_cold_memory(record)
        return {
            "status": "success",
            "memory": persisted.to_dict(),
            "memory_id": persisted.memory_id,
        }
    except Exception as exc:
        logger.warning("tool_memory_store_cold failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "memory_id": None,
        }


# ─── 5. Memory Update ─────────────────────────────────────────────────────────

def tool_memory_update(
    manager: Any,
    memory_id: str,
    memory_type: str,
    content: str | None = None,
    category: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    text: str | None = None,
) -> dict:
    """Update an existing memory record in canonical store and index.

    Args:
        manager:     MemoryManager instance
        memory_id:   Memory ID to update
        memory_type: "hot" or "cold"
        content:     Optional updated text
        category:    Optional updated category
        importance:  Optional updated importance in [0.0, 1.0]
        confidence:  Optional updated confidence in [0.0, 1.0]

    Returns:
        {"status": "success", "memory": dict, "memory_id": str}
        or {"status": "error", "error": str, "memory_id": str}
    """
    if content is None and text is not None:
        content = text
    if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
        return {
            "status": "error",
            "error": "memory_id is required",
            "memory_id": memory_id,
        }
    clean_id = memory_id.strip()
    if not clean_id.startswith("mem_"):
        return {
            "status": "error",
            "error": f"memory_id must start with 'mem_', got {memory_id!r}",
            "memory_id": clean_id,
        }

    if memory_type not in ("hot", "cold"):
        return {
            "status": "error",
            "error": f"memory_type must be 'hot' or 'cold', got {memory_type!r}",
            "memory_id": clean_id,
        }

    if content is None and category is None and importance is None and confidence is None:
        return {
            "status": "error",
            "error": "At least one field to update must be provided (content, category, importance, confidence)",
            "memory_id": clean_id,
        }

    if content is not None:
        if not isinstance(content, str) or not content.strip():
            return {
                "status": "error",
                "error": "content cannot be empty",
                "memory_id": clean_id,
            }

    if category is not None:
        if not isinstance(category, str) or not category.strip():
            return {
                "status": "error",
                "error": "category cannot be empty",
                "memory_id": clean_id,
            }

    if importance is not None:
        try:
            imp = float(importance)
        except (ValueError, TypeError):
            return {
                "status": "error",
                "error": f"importance must be numeric, got {importance!r}",
                "memory_id": clean_id,
            }
        if imp < 0.0 or imp > 1.0:
            return {
                "status": "error",
                "error": f"importance must be in [0.0, 1.0], got {importance}",
                "memory_id": clean_id,
            }
    else:
        imp = None

    if confidence is not None:
        try:
            conf = float(confidence)
        except (ValueError, TypeError):
            return {
                "status": "error",
                "error": f"confidence must be numeric, got {confidence!r}",
                "memory_id": clean_id,
            }
        if conf < 0.0 or conf > 1.0:
            return {
                "status": "error",
                "error": f"confidence must be in [0.0, 1.0], got {confidence}",
                "memory_id": clean_id,
            }
    else:
        conf = None

    try:
        existing = manager.get_memory(clean_id, memory_type)
        if existing is None:
            return {
                "status": "error",
                "error": f"Memory {clean_id!r} ({memory_type}) not found",
                "memory_id": clean_id,
            }

        now = iso_now()
        updated_record = MemoryRecord(
            memory_id=existing.memory_id,
            content=content.strip() if content is not None else existing.content,
            memory_type=existing.memory_type,
            category=category.strip() if category is not None else existing.category,
            source=existing.source,
            session_id=existing.session_id,
            created_at=existing.created_at,
            updated_at=now,
            importance=imp if imp is not None else existing.importance,
            confidence=conf if conf is not None else existing.confidence,
            access_count=existing.access_count,
            last_accessed=existing.last_accessed,
            is_summary=existing.is_summary,
            parent_memory_ids=list(existing.parent_memory_ids),
        )
        persisted = manager.update_memory(updated_record)
        return {
            "status": "success",
            "memory": persisted.to_dict(),
            "memory_id": persisted.memory_id,
        }
    except Exception as exc:
        logger.warning("tool_memory_update failed for %s: %s", clean_id, exc)
        return {
            "status": "error",
            "error": str(exc),
            "memory_id": clean_id,
        }


# ─── 6. Memory Deletion ───────────────────────────────────────────────────────

def tool_memory_delete(
    manager: Any,
    memory_id: str,
    memory_type: str,
) -> dict:
    """Delete a memory record from both canonical store and Chroma.

    Args:
        manager:     MemoryManager instance
        memory_id:   Memory ID to delete
        memory_type: "hot" or "cold"

    Returns:
        {"status": "success", "deleted": bool, "memory_id": str, "memory_type": str}
        or {"status": "error", "error": str, "deleted": False, "memory_id": str, "memory_type": str}
    """
    if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
        return {
            "status": "error",
            "error": "memory_id is required",
            "deleted": False,
            "memory_id": memory_id,
            "memory_type": memory_type,
        }
    clean_id = memory_id.strip()
    if not clean_id.startswith("mem_"):
        return {
            "status": "error",
            "error": f"memory_id must start with 'mem_', got {memory_id!r}",
            "deleted": False,
            "memory_id": clean_id,
            "memory_type": memory_type,
        }

    if memory_type not in ("hot", "cold"):
        return {
            "status": "error",
            "error": f"memory_type must be 'hot' or 'cold', got {memory_type!r}",
            "deleted": False,
            "memory_id": clean_id,
            "memory_type": memory_type,
        }

    try:
        deleted = manager.delete_memory(clean_id, memory_type)
        return {
            "status": "success",
            "deleted": deleted,
            "memory_id": clean_id,
            "memory_type": memory_type,
        }
    except Exception as exc:
        logger.warning("tool_memory_delete failed for %s: %s", clean_id, exc)
        return {
            "status": "error",
            "error": str(exc),
            "deleted": False,
            "memory_id": clean_id,
            "memory_type": memory_type,
        }


# ─── 7. Memory Summarization ──────────────────────────────────────────────────

def tool_memory_summarize(
    manager: Any,
    session_id: str,
    summary_content: str,
    parent_memory_ids: list[str],
    category: str = "summary",
    importance: float = 0.5,
    confidence: float = 0.5,
) -> dict:
    """Store a summary hot memory linked to parent IDs. Retains parent memories.

    Args:
        manager:           MemoryManager instance
        session_id:        Session identifier
        summary_content:   Consolidated summary text
        parent_memory_ids: List of parent memory IDs being summarized
        category:          Category label (defaults to "summary")
        importance:        Numeric importance
        confidence:        Numeric confidence

    Returns:
        {"status": "success", "memory": dict, "memory_id": str}
    """
    if not session_id or not str(session_id).strip():
        return {
            "status": "error",
            "error": "session_id is required for memory summarization",
            "memory_id": None,
        }

    if not summary_content or not str(summary_content).strip():
        return {
            "status": "error",
            "error": "summary_content is required for memory summarization",
            "memory_id": None,
        }

    if not parent_memory_ids or not isinstance(parent_memory_ids, list):
        return {
            "status": "error",
            "error": "parent_memory_ids must be a non-empty list of memory IDs",
            "memory_id": None,
        }

    try:
        imp = float(importance)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error": f"importance must be numeric, got {importance!r}",
            "memory_id": None,
        }
    if imp < 0.0 or imp > 1.0:
        return {
            "status": "error",
            "error": f"importance must be in [0.0, 1.0], got {importance}",
            "memory_id": None,
        }

    try:
        conf = float(confidence)
    except (ValueError, TypeError):
        return {
            "status": "error",
            "error": f"confidence must be numeric, got {confidence!r}",
            "memory_id": None,
        }
    if conf < 0.0 or conf > 1.0:
        return {
            "status": "error",
            "error": f"confidence must be in [0.0, 1.0], got {confidence}",
            "memory_id": None,
        }

    try:
        persisted = manager.summarize_hot_memory(
            session_id=session_id.strip(),
            summary_content=summary_content.strip(),
            parent_memory_ids=parent_memory_ids,
            category=category,
            importance=imp,
            confidence=conf,
        )
        return {
            "status": "success",
            "memory": persisted.to_dict(),
            "memory_id": persisted.memory_id,
        }
    except Exception as exc:
        logger.warning("tool_memory_summarize failed: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "memory_id": None,
        }


# ─── 8. Hot -> Cold Promotion ─────────────────────────────────────────────────

def tool_memory_promote(
    manager: Any,
    memory_id: str,
) -> dict:
    """Promote a hot memory to durable cold storage (Gemma-driven decision).

    Args:
        manager:   MemoryManager instance
        memory_id: Hot memory ID to promote

    Returns:
        {"status": "success", "memory": dict, "memory_id": str, "promoted": True}
    """
    if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
        return {
            "status": "error",
            "error": "memory_id is required for memory promotion",
            "memory_id": memory_id or "",
            "promoted": False,
        }
    if not memory_id.strip().startswith("mem_"):
        return {
            "status": "error",
            "error": f"memory_id must start with 'mem_', got {memory_id!r}",
            "memory_id": memory_id,
            "promoted": False,
        }

    try:
        promoted = manager.promote_to_cold(memory_id.strip())
        return {
            "status": "success",
            "memory": promoted.to_dict(),
            "memory_id": promoted.memory_id,
            "promoted": True,
        }
    except Exception as exc:
        logger.warning("tool_memory_promote failed for %s: %s", memory_id, exc)
        return {
            "status": "error",
            "error": str(exc),
            "memory_id": memory_id,
            "promoted": False,
        }


# ─── Registry Wiring Helper ───────────────────────────────────────────────────

def register_memory_tools(registry: Any, memory_manager: Any) -> None:
    """Register all memory tools with a ToolRegistry.

    Canonical registrations (matching tools.json / abilities.json):
        memory / memory_search_hot
        memory / memory_search_cold
        memory / memory_store_hot
        memory / memory_store_cold
        memory / memory_update
        memory / memory_delete
        memory / memory_summarize
        memory / memory_promote

    Convenience short-name aliases:
        memory / search_hot
        memory / search_cold
        memory / store_hot
        memory / store_cold
        memory / update
        memory / delete
        memory / summarize
        memory / promote
    """
    from functools import partial

    fns = {
        # Canonical names
        "memory_search_hot": partial(tool_memory_search_hot, memory_manager),
        "memory_search_cold": partial(tool_memory_search_cold, memory_manager),
        "memory_store_hot": partial(tool_memory_store_hot, memory_manager),
        "memory_store_cold": partial(tool_memory_store_cold, memory_manager),
        "memory_update": partial(tool_memory_update, memory_manager),
        "memory_delete": partial(tool_memory_delete, memory_manager),
        "memory_summarize": partial(tool_memory_summarize, memory_manager),
        "memory_promote": partial(tool_memory_promote, memory_manager),
        # Short aliases
        "search_hot": partial(tool_memory_search_hot, memory_manager),
        "search_cold": partial(tool_memory_search_cold, memory_manager),
        "store_hot": partial(tool_memory_store_hot, memory_manager),
        "store_cold": partial(tool_memory_store_cold, memory_manager),
        "update": partial(tool_memory_update, memory_manager),
        "delete": partial(tool_memory_delete, memory_manager),
        "summarize": partial(tool_memory_summarize, memory_manager),
        "promote": partial(tool_memory_promote, memory_manager),
    }

    for fn_name, fn in fns.items():
        registry.register("memory", fn_name, fn)
