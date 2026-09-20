"""
SAGE/tools/durable_memory.py
Phase 3: Semantic Memory Decision & Retrieval Tool Adapter
Phase 5: Gemma-Controlled Memory Extraction & Writing

Provides deterministic model-facing memory operations:
- memory_search
- memory_get
- memory_store (Phase 5)

Strict validation rules:
1. Model read operations: 'search', 'get'. Model write operation: 'store'.
   'update', 'delete', 'supersede' remain protected internal operations.
2. Categories validated against allowed Phase 2 set.
3. Limits clamped to [1, 20] (default 5).
4. User ID authorization bound from RunState for strict user isolation.
   Gemma-supplied user_id is NEVER trusted; trusted identity comes from runtime.
5. Returned data sanitized to expose only safe, model-usable properties.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import config
from sage_memory import sage_memory, validate_category, ALLOWED_CATEGORIES

logger = logging.getLogger(__name__)

# 'store' is now model-facing in Phase 5; 'update', 'delete', 'supersede' remain protected.
REJECTED_MODEL_OPERATIONS = {"update", "delete", "supersede"}


def _sanitize_memory_record(mem: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize internal memory dictionary for model consumption."""
    tier_val = str(mem.get("memory_tier") or mem.get("tier") or "cold")
    res = {
        "memory_id": str(mem.get("memory_id", "")),
        "content": str(mem.get("content", "")),
        "category": str(mem.get("category", "")),
        "importance": float(mem.get("importance", 0.5)),
        "confidence": float(mem.get("confidence", 1.0)),
        "status": str(mem.get("status", "active")),
        "memory_tier": tier_val,
        "tier": tier_val,
        "source_chat_id": mem.get("source_chat_id"),
        "created_at": str(mem.get("created_at", "")),
        "updated_at": str(mem.get("updated_at", "")),
    }
    if mem.get("supersedes_memory_id"):
        res["supersedes_memory_id"] = str(mem["supersedes_memory_id"])
    return res


def memory_search(
    user_id: Optional[str] = None,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 5,
    operation: Optional[str] = None,
    memory_needed: Optional[bool] = None,
    tier: str = "cold",
    chat_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute controlled durable memory search (backward compatible, defaults to cold)."""
    # 0. Validate operation — reject protected or unknown operations (Phase 3 contract)
    if operation is not None and str(operation).strip():
        op_val = str(operation).strip().lower()
        if op_val in REJECTED_MODEL_OPERATIONS:
            return {
                "status": "error",
                "error": {
                    "code": "INVALID_MEMORY_OPERATION",
                    "message": (
                        f"Operation '{op_val}' is not permitted from the model-facing interface. "
                        "Use the deterministic backend for update/delete/supersede operations."
                    ),
                    "retryable": False,
                },
            }
        # 'store' is handled by memory_store, not memory_search
        KNOWN_SEARCH_OPERATIONS = {
            "search", "get", "memory_search_hot", "memory_search_cold",
            "memory_store_hot", "store_hot", "memory_store_cold", "store_cold",
            "memory_promote", "promote", "memory_summarize", "summarize",
        }
        if op_val not in KNOWN_SEARCH_OPERATIONS:
            return {
                "status": "error",
                "error": {
                    "code": "UNKNOWN_MEMORY_OPERATION",
                    "message": (
                        f"Unknown operation '{op_val}'. "
                        "Use memory_store for storing memories."
                    ),
                    "retryable": False,
                },
            }

    # 1. Check if action or operation routes to Phase 7 operations
    action = (kwargs.get("action") or operation or "").lower().strip()
    if action == "memory_search_hot":
        return memory_search_hot(user_id=user_id, chat_id=chat_id, query=query, category=category, limit=limit, **kwargs)
    elif action == "memory_search_cold":
        return memory_search_cold(user_id=user_id, query=query, category=category, limit=limit, **kwargs)
    elif action in ("memory_store_hot", "store_hot"):
        return memory_store_hot(user_id=user_id, chat_id=chat_id, query=query, category=category, **kwargs)
    elif action in ("memory_store_cold", "store_cold"):
        return memory_store_cold(user_id=user_id, chat_id=chat_id, query=query, category=category, **kwargs)
    elif action in ("memory_promote", "promote"):
        return memory_promote(user_id=user_id, **kwargs)
    elif action in ("memory_summarize", "summarize"):
        return memory_summarize(user_id=user_id, chat_id=chat_id, **kwargs)

    # If tier explicitly specified as hot, route to memory_search_hot
    if (tier or "").lower().strip() == "hot":
        return memory_search_hot(user_id=user_id, chat_id=chat_id, query=query, category=category, limit=limit, **kwargs)

    # Otherwise default to cold search
    return memory_search_cold(user_id=user_id, query=query, category=category, limit=limit, memory_needed=memory_needed, **kwargs)


def memory_search_hot(
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute controlled hot memory search scoped to trusted user_id and chat_id."""
    # 1. Category validation
    validated_cat = None
    if category and str(category).strip():
        try:
            validated_cat = validate_category(str(category))
        except ValueError as exc:
            return {
                "status": "error",
                "error": {
                    "code": "INVALID_MEMORY_CATEGORY",
                    "message": str(exc),
                    "retryable": False,
                },
            }

    # 2. Bound limit: default = 5, maximum = 20
    try:
        raw_limit = int(limit) if limit is not None else 5
    except (ValueError, TypeError):
        raw_limit = 5
    clamped_limit = max(1, min(raw_limit, 20))

    # 3. Determine active user_id and chat_id (never trust model-provided identity)
    effective_user_id = user_id or config.DEFAULT_USER_ID
    effective_chat_id = chat_id

    # 4. Semantic retrieval via Chroma vector search with canonical PostgreSQL/SQLite verification
    has_query = query is not None and isinstance(query, str) and bool(query.strip())
    raw_memories: List[Dict[str, Any]] = []

    if has_query:
        from sage_memory_index import memory_vector_index
        vector_results = memory_vector_index.search_memory_vectors(
            user_id=effective_user_id,
            query=str(query).strip(),
            tier="hot",
            chat_id=effective_chat_id,
            category=validated_cat,
            limit=clamped_limit,
        )
        seen_ids = set()
        for v_res in vector_results:
            m_id = v_res.get("memory_id")
            if m_id and m_id not in seen_ids:
                canonical_mem = sage_memory.get_memory(m_id, touch_access=True)
                if (
                    canonical_mem
                    and canonical_mem.get("user_id") == effective_user_id
                    and canonical_mem.get("status") == "active"
                    and canonical_mem.get("memory_tier") == "hot"
                    and (not effective_chat_id or canonical_mem.get("source_chat_id") == effective_chat_id)
                ):
                    if not validated_cat or canonical_mem.get("category") == validated_cat:
                        raw_memories.append(canonical_mem)
                        seen_ids.add(m_id)
        # Chroma is a derived index.  If it is unavailable, stale, or has no
        # eligible vectors, preserve session continuity with the canonical
        # database rather than treating durable memory as empty.
        if not raw_memories:
            raw_memories = sage_memory.list_memories(
                user_id=effective_user_id,
                category=validated_cat,
                memory_tier="hot",
                chat_id=effective_chat_id,
                limit=clamped_limit,
            )
            for m in raw_memories:
                sage_memory.touch_memory_access(m["memory_id"])
    else:
        # Fallback to listing active hot memories for this session
        raw_memories = sage_memory.list_memories(
            user_id=effective_user_id,
            category=validated_cat,
            memory_tier="hot",
            chat_id=effective_chat_id,
            limit=clamped_limit,
        )
        for m in raw_memories:
            sage_memory.touch_memory_access(m["memory_id"])

    sanitized = [_sanitize_memory_record(m) for m in raw_memories]

    return {
        "status": "success",
        "memories": sanitized,
        "returned": len(sanitized),
    }


def memory_search_cold(
    user_id: Optional[str] = None,
    query: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 5,
    memory_needed: Optional[bool] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute controlled cold memory search scoped to trusted user_id across chats."""
    # 1. No-memory decision check
    if memory_needed is False:
        return {
            "status": "success",
            "memories": [],
            "returned": 0,
        }

    # 2. Category validation
    validated_cat = None
    if category and str(category).strip():
        try:
            validated_cat = validate_category(str(category))
        except ValueError as exc:
            return {
                "status": "error",
                "error": {
                    "code": "INVALID_MEMORY_CATEGORY",
                    "message": str(exc),
                    "retryable": False,
                },
            }

    # 3. Bound limit: default = 5, maximum = 20
    try:
        raw_limit = int(limit) if limit is not None else 5
    except (ValueError, TypeError):
        raw_limit = 5
    clamped_limit = max(1, min(raw_limit, 20))

    # 4. Determine active user_id
    effective_user_id = user_id or config.DEFAULT_USER_ID

    # 5. Semantic retrieval via Chroma cold vector search with canonical PostgreSQL/SQLite verification
    has_query = query is not None and isinstance(query, str) and bool(query.strip())
    raw_memories: List[Dict[str, Any]] = []

    if has_query:
        from sage_memory_index import memory_vector_index
        vector_results = memory_vector_index.search_memory_vectors(
            user_id=effective_user_id,
            query=str(query).strip(),
            tier="cold",
            category=validated_cat,
            limit=clamped_limit,
        )
        seen_ids = set()
        for v_res in vector_results:
            m_id = v_res.get("memory_id")
            if m_id and m_id not in seen_ids:
                canonical_mem = sage_memory.get_memory(m_id, touch_access=True)
                if (
                    canonical_mem
                    and canonical_mem.get("user_id") == effective_user_id
                    and canonical_mem.get("status") == "active"
                    and canonical_mem.get("memory_tier") == "cold"
                ):
                    if not validated_cat or canonical_mem.get("category") == validated_cat:
                        raw_memories.append(canonical_mem)
                        seen_ids.add(m_id)
        # The canonical database remains the source of truth.  Fall back to
        # its deterministic ordering when semantic-index retrieval is empty.
        if not raw_memories:
            raw_memories = sage_memory.list_memories(
                user_id=effective_user_id,
                category=validated_cat,
                memory_tier="cold",
                limit=clamped_limit,
            )
            for m in raw_memories:
                sage_memory.touch_memory_access(m["memory_id"])
    else:
        # Fallback to listing active cold memories
        raw_memories = sage_memory.list_memories(
            user_id=effective_user_id,
            category=validated_cat,
            memory_tier="cold",
            limit=clamped_limit,
        )
        for m in raw_memories:
            sage_memory.touch_memory_access(m["memory_id"])

    sanitized = [_sanitize_memory_record(m) for m in raw_memories]

    return {
        "status": "success",
        "memories": sanitized,
        "returned": len(sanitized),
    }


def memory_get(
    user_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute deterministic lookup by memory_id with user isolation enforcement."""
    if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_ID",
                "message": "memory_id must be a non-empty string.",
                "retryable": False,
            },
        }

    effective_user_id = user_id or config.DEFAULT_USER_ID
    fetched = sage_memory.get_memory(memory_id.strip(), touch_access=True)

    # User isolation & active status enforcement
    if not fetched or fetched.get("user_id") != effective_user_id or fetched.get("status") != "active":
        return {
            "status": "success",
            "memory": None,
        }

    return {
        "status": "success",
        "memory": _sanitize_memory_record(fetched),
    }


def memory_store(
    user_id: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    chat_id: Optional[str] = None,
    source_msg_id: Optional[str] = None,
    supersedes_memory_id: Optional[str] = None,
    memory_tier: Optional[str] = None,
    tier: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute controlled durable memory store (backward compatible dispatcher)."""
    selected_tier = (tier or memory_tier or "cold").lower().strip()
    if selected_tier == "hot":
        return memory_store_hot(
            user_id=user_id,
            chat_id=chat_id,
            content=content,
            category=category,
            importance=importance,
            confidence=confidence,
            source_msg_id=source_msg_id,
            **kwargs,
        )
    return memory_store_cold(
        user_id=user_id,
        chat_id=chat_id,
        content=content,
        category=category,
        importance=importance,
        confidence=confidence,
        source_msg_id=source_msg_id,
        supersedes_memory_id=supersedes_memory_id,
        **kwargs,
    )


def memory_store_hot(
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    source_msg_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute controlled hot memory store bound to trusted user_id and chat_id."""
    # 1. Content validation
    if not content or not isinstance(content, str) or not content.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CONTENT",
                "message": "content must be a non-empty string.",
                "retryable": False,
            },
        }

    # 2. Category validation
    if not category or not isinstance(category, str) or not category.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CATEGORY",
                "message": f"category is required. Allowed: {sorted(list(ALLOWED_CATEGORIES))}",
                "retryable": False,
            },
        }
    try:
        validated_cat = validate_category(str(category))
    except ValueError as exc:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CATEGORY",
                "message": str(exc),
                "retryable": False,
            },
        }

    # 3. Importance: clamp to [0, 1], default 0.5
    try:
        raw_importance = float(importance) if importance is not None else 0.5
    except (ValueError, TypeError):
        raw_importance = 0.5
    clamped_importance = max(0.0, min(raw_importance, 1.0))

    # 4. Confidence: clamp to [0, 1], default 1.0
    try:
        raw_confidence = float(confidence) if confidence is not None else 1.0
    except (ValueError, TypeError):
        raw_confidence = 1.0
    clamped_confidence = max(0.0, min(raw_confidence, 1.0))

    # 5. Hot memories are always session-scoped.  The orchestrator supplies
    # this from trusted RunState; reject direct calls that omit it rather than
    # creating an unscoped hot record.
    if not chat_id or not isinstance(chat_id, str) or not chat_id.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CHAT",
                "message": "chat_id is required for hot memory storage.",
                "retryable": False,
            },
        }

    # 6. Trusted runtime identity (user_id and chat_id from RunState)
    effective_user_id = user_id or config.DEFAULT_USER_ID
    effective_chat_id = chat_id.strip()

    # 7. Store in canonical DB with memory_tier="hot"
    try:
        stored = sage_memory.store_memory(
            user_id=effective_user_id,
            content=content.strip(),
            category=validated_cat,
            importance=clamped_importance,
            confidence=clamped_confidence,
            source_chat_id=effective_chat_id,
            source_msg_id=source_msg_id or None,
            memory_tier="hot",
        )
    except Exception as exc:
        logger.error("memory_store_hot: canonical DB write failed: %s", exc)
        return {
            "status": "error",
            "error": {
                "code": "MEMORY_STORE_FAILED",
                "message": "Failed to persist hot memory to canonical database.",
                "retryable": True,
            },
        }

    return {
        "status": "success",
        "memory": {
            "memory_id": str(stored["memory_id"]),
            "content": str(stored["content"]),
            "category": str(stored["category"]),
            "importance": float(stored["importance"]),
            "confidence": float(stored["confidence"]),
            "status": "active",
            "memory_tier": "hot",
            "tier": "hot",
            "source_chat_id": stored.get("source_chat_id"),
        },
    }


def memory_store_cold(
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    source_msg_id: Optional[str] = None,
    supersedes_memory_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute controlled cold durable memory store or supersession."""
    # 1. Content validation
    if not content or not isinstance(content, str) or not content.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CONTENT",
                "message": "content must be a non-empty string.",
                "retryable": False,
            },
        }

    # 2. Category validation
    if not category or not isinstance(category, str) or not category.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CATEGORY",
                "message": f"category is required. Allowed: {sorted(list(ALLOWED_CATEGORIES))}",
                "retryable": False,
            },
        }
    try:
        validated_cat = validate_category(str(category))
    except ValueError as exc:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CATEGORY",
                "message": str(exc),
                "retryable": False,
            },
        }

    # 3. Importance clamp
    try:
        raw_importance = float(importance) if importance is not None else 0.5
    except (ValueError, TypeError):
        raw_importance = 0.5
    clamped_importance = max(0.0, min(raw_importance, 1.0))

    # 4. Confidence clamp
    try:
        raw_confidence = float(confidence) if confidence is not None else 1.0
    except (ValueError, TypeError):
        raw_confidence = 1.0
    clamped_confidence = max(0.0, min(raw_confidence, 1.0))

    # 5. Trusted runtime user_id
    effective_user_id = user_id or config.DEFAULT_USER_ID

    # 6. Target memory validation if supersedes_memory_id is provided
    target_old_id = str(supersedes_memory_id).strip() if supersedes_memory_id and str(supersedes_memory_id).strip() else None

    if target_old_id:
        target_mem = sage_memory.get_memory(target_old_id)
        if not target_mem:
            return {
                "status": "error",
                "error": {
                    "code": "TARGET_MEMORY_NOT_FOUND",
                    "message": f"Target memory '{target_old_id}' does not exist.",
                    "retryable": False,
                },
            }

        # User isolation: target must belong to effective_user_id
        if target_mem.get("user_id") != effective_user_id:
            return {
                "status": "error",
                "error": {
                    "code": "UNAUTHORIZED_MEMORY_ACCESS",
                    "message": f"Target memory '{target_old_id}' does not belong to the active user.",
                    "retryable": False,
                },
            }

        # Active status check
        target_status = target_mem.get("status")
        if target_status != "active":
            return {
                "status": "error",
                "error": {
                    "code": "TARGET_MEMORY_INACTIVE",
                    "message": f"Target memory '{target_old_id}' is not active (current status: '{target_status}'). Cannot supersede inactive memory.",
                    "retryable": False,
                },
            }

        # Category compatibility check
        if target_mem.get("category") != validated_cat:
            return {
                "status": "error",
                "error": {
                    "code": "CATEGORY_MISMATCH",
                    "message": f"Target memory category '{target_mem.get('category')}' does not match new memory category '{validated_cat}'. Cross-category supersession is not permitted.",
                    "retryable": False,
                },
            }

    # 7. Persist in canonical database (PostgreSQL / SQLite)
    try:
        if target_old_id:
            stored = sage_memory.supersede_memory(
                old_memory_id=target_old_id,
                new_content=content.strip(),
                user_id=effective_user_id,
                category=validated_cat,
                importance=clamped_importance,
                confidence=clamped_confidence,
                source_chat_id=chat_id or None,
                source_msg_id=source_msg_id or None,
                memory_tier="cold",
            )
        else:
            stored = sage_memory.store_memory(
                user_id=effective_user_id,
                content=content.strip(),
                category=validated_cat,
                importance=clamped_importance,
                confidence=clamped_confidence,
                source_chat_id=chat_id or None,
                source_msg_id=source_msg_id or None,
                memory_tier="cold",
            )
    except Exception as exc:
        logger.error("memory_store_cold: canonical DB write failed: %s", exc)
        return {
            "status": "error",
            "error": {
                "code": "MEMORY_STORE_FAILED",
                "message": "Failed to persist cold memory to canonical database.",
                "retryable": True,
            },
        }

    safe_memory = {
        "memory_id": str(stored["memory_id"]),
        "content": str(stored["content"]),
        "category": str(stored["category"]),
        "importance": float(stored["importance"]),
        "confidence": float(stored["confidence"]),
        "status": "active",
        "memory_tier": "cold",
        "tier": "cold",
    }
    if stored.get("supersedes_memory_id"):
        safe_memory["supersedes_memory_id"] = str(stored["supersedes_memory_id"])
    if stored.get("source_chat_id"):
        safe_memory["source_chat_id"] = str(stored["source_chat_id"])

    return {
        "status": "success",
        "memory": safe_memory,
    }


def memory_promote(
    user_id: Optional[str] = None,
    memory_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute memory promotion from HOT tier to COLD tier."""
    if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_ID",
                "message": "memory_id must be a non-empty string.",
                "retryable": False,
            },
        }

    effective_user_id = user_id or config.DEFAULT_USER_ID
    target_id = memory_id.strip()

    target_mem = sage_memory.get_memory(target_id)
    if not target_mem:
        return {
            "status": "error",
            "error": {
                "code": "TARGET_MEMORY_NOT_FOUND",
                "message": f"Target memory '{target_id}' does not exist.",
                "retryable": False,
            },
        }

    if target_mem.get("user_id") != effective_user_id:
        return {
            "status": "error",
            "error": {
                "code": "UNAUTHORIZED_MEMORY_ACCESS",
                "message": f"Target memory '{target_id}' does not belong to active user.",
                "retryable": False,
            },
        }

    if target_mem.get("status") != "active":
        return {
            "status": "error",
            "error": {
                "code": "TARGET_MEMORY_INACTIVE",
                "message": f"Target memory '{target_id}' is not active (status: '{target_mem.get('status')}').",
                "retryable": False,
            },
        }

    if target_mem.get("memory_tier") != "hot":
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_TIER",
                "message": f"Target memory '{target_id}' is tier '{target_mem.get('memory_tier')}'. Only hot memories can be promoted to cold.",
                "retryable": False,
            },
        }

    try:
        promoted = sage_memory.promote_memory(memory_id=target_id, user_id=effective_user_id)
    except Exception as exc:
        logger.error("memory_promote failed: %s", exc)
        return {
            "status": "error",
            "error": {
                "code": "MEMORY_PROMOTE_FAILED",
                "message": str(exc),
                "retryable": False,
            },
        }

    safe_memory = {
        "memory_id": str(promoted["memory_id"]),
        "content": str(promoted["content"]),
        "category": str(promoted["category"]),
        "importance": float(promoted["importance"]),
        "confidence": float(promoted["confidence"]),
        "status": "active",
        "memory_tier": "cold",
        "tier": "cold",
    }
    if promoted.get("supersedes_memory_id"):
        safe_memory["supersedes_memory_id"] = str(promoted["supersedes_memory_id"])
    if promoted.get("source_chat_id"):
        safe_memory["source_chat_id"] = str(promoted["source_chat_id"])

    return {
        "status": "success",
        "memory": safe_memory,
    }


def memory_summarize(
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    memory_ids: Optional[List[str]] = None,
    summary_content: Optional[str] = None,
    content: Optional[str] = None,
    category: str = "fact",
    importance: float = 0.5,
    confidence: float = 1.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compact multiple related hot memories into a single concise hot summary memory."""
    # 1. Validate summary content
    resolved_content = (summary_content or content or "").strip()
    if not resolved_content:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CONTENT",
                "message": "summary_content must be a non-empty string.",
                "retryable": False,
            },
        }

    # 2. Validate memory_ids
    if not memory_ids or not isinstance(memory_ids, (list, tuple)) or len(memory_ids) == 0:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_IDS",
                "message": "memory_ids must be a non-empty list of memory IDs.",
                "retryable": False,
            },
        }

    # 3. Validate category
    try:
        validated_cat = validate_category(str(category))
    except ValueError as exc:
        return {
            "status": "error",
            "error": {
                "code": "INVALID_MEMORY_CATEGORY",
                "message": str(exc),
                "retryable": False,
            },
        }

    effective_user_id = user_id or config.DEFAULT_USER_ID
    unique_ids = list(dict.fromkeys(str(mid).strip() for mid in memory_ids if str(mid).strip()))

    # 4. Check each memory ID for existence, user isolation, status, tier, and session isolation
    first_chat_id = chat_id
    for mid in unique_ids:
        mem = sage_memory.get_memory(mid)
        if not mem:
            return {
                "status": "error",
                "error": {
                    "code": "MEMORY_NOT_FOUND",
                    "message": f"Source memory '{mid}' not found.",
                    "retryable": False,
                },
            }
        if mem.get("user_id") != effective_user_id:
            return {
                "status": "error",
                "error": {
                    "code": "UNAUTHORIZED_MEMORY_ACCESS",
                    "message": f"Source memory '{mid}' does not belong to active user.",
                    "retryable": False,
                },
            }
        if mem.get("status") != "active":
            return {
                "status": "error",
                "error": {
                    "code": "TARGET_MEMORY_INACTIVE",
                    "message": f"Source memory '{mid}' is not active (status: '{mem.get('status')}').",
                    "retryable": False,
                },
            }
        if mem.get("memory_tier") != "hot":
            return {
                "status": "error",
                "error": {
                    "code": "INVALID_MEMORY_TIER",
                    "message": f"Source memory '{mid}' is tier '{mem.get('memory_tier')}'. Only hot memories can be compacted.",
                    "retryable": False,
                },
            }
        # Session check
        mem_chat = mem.get("source_chat_id")
        if first_chat_id is None:
            first_chat_id = mem_chat
        elif mem_chat != first_chat_id:
            return {
                "status": "error",
                "error": {
                    "code": "INVALID_SESSION",
                    "message": f"Source memory '{mid}' chat '{mem_chat}' does not match expected chat session '{first_chat_id}'.",
                    "retryable": False,
                },
            }

    # 5. Execute compaction in canonical DB
    try:
        summary_mem = sage_memory.compact_memories(
            memory_ids=unique_ids,
            summary_content=resolved_content,
            user_id=effective_user_id,
            chat_id=first_chat_id,
            category=validated_cat,
            importance=float(importance) if importance is not None else 0.5,
            confidence=float(confidence) if confidence is not None else 1.0,
        )
    except Exception as exc:
        logger.error("memory_summarize: compaction failed: %s", exc)
        return {
            "status": "error",
            "error": {
                "code": "MEMORY_COMPACTION_FAILED",
                "message": str(exc),
                "retryable": False,
            },
        }

    return {
        "status": "success",
        "memory": {
            "memory_id": str(summary_mem["memory_id"]),
            "content": str(summary_mem["content"]),
            "category": str(summary_mem["category"]),
            "importance": float(summary_mem["importance"]),
            "confidence": float(summary_mem["confidence"]),
            "status": "active",
            "memory_tier": "hot",
            "source_chat_id": summary_mem.get("source_chat_id"),
        },
    }


def register_durable_memory_tools(registry: Any) -> None:
    """Register durable memory tools in ToolRegistry."""
    # Register under primary group 'durable_memory'
    registry.register("durable_memory", "memory_search", memory_search)
    registry.register("durable_memory", "memory_get", memory_get)
    registry.register("durable_memory", "memory_store", memory_store)
    registry.register("durable_memory", "memory_search_hot", memory_search_hot)
    registry.register("durable_memory", "memory_search_cold", memory_search_cold)
    registry.register("durable_memory", "memory_store_hot", memory_store_hot)
    registry.register("durable_memory", "memory_store_cold", memory_store_cold)
    registry.register("durable_memory", "memory_promote", memory_promote)
    registry.register("durable_memory", "memory_summarize", memory_summarize)

    # Register aliases for flexible invocation
    for grp in ("memory", "memory_search", "memory_get", "memory_store"):
        registry.register(grp, "memory_search", memory_search)
        registry.register(grp, "memory_get", memory_get)
        registry.register(grp, "memory_store", memory_store)
        registry.register(grp, "memory_search_hot", memory_search_hot)
        registry.register(grp, "memory_search_cold", memory_search_cold)
        registry.register(grp, "memory_store_hot", memory_store_hot)
        registry.register(grp, "memory_store_cold", memory_store_cold)
        registry.register(grp, "memory_promote", memory_promote)
        registry.register(grp, "memory_summarize", memory_summarize)
