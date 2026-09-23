"""
standalone_memory/tools.py
The 8 Model-Facing Memory Tools.
Callable directly by LLMs (OpenAI, Anthropic, Ollama, LangChain, etc.) or Python agents.
Includes standardized JSON function calling schemas.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from . import config
from .memory_engine import memory_engine, validate_category, ALLOWED_CATEGORIES
from .vector_index import memory_vector_index

logger = logging.getLogger(__name__)


# ── Tool 1: Store Hot Memory ──────────────────────────────────────────────────

def memory_store_hot(
    user_id: Optional[str] = None,
    chat_id: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Store working session memory bound to chat_id and user_id."""
    if not content or not isinstance(content, str) or not content.strip():
        return {"status": "error", "error": {"code": "INVALID_MEMORY_CONTENT", "message": "content is required."}}

    if not category or not isinstance(category, str) or not category.strip():
        return {"status": "error", "error": {"code": "INVALID_MEMORY_CATEGORY", "message": f"category is required. Allowed: {sorted(list(ALLOWED_CATEGORIES))}"}}

    try:
        val_cat = validate_category(str(category))
    except ValueError as e:
        return {"status": "error", "error": {"code": "INVALID_MEMORY_CATEGORY", "message": str(e)}}

    if not chat_id or not isinstance(chat_id, str) or not chat_id.strip():
        return {"status": "error", "error": {"code": "INVALID_MEMORY_CHAT", "message": "chat_id is required for hot memory."}}

    uid = user_id or config.DEFAULT_USER_ID
    imp = max(0.0, min(float(importance if importance is not None else 0.5), 1.0))
    conf = max(0.0, min(float(confidence if confidence is not None else 1.0), 1.0))

    record = memory_engine.store_memory(
        user_id=uid,
        content=content.strip(),
        tier="hot",
        category=val_cat,
        importance=imp,
        confidence=conf,
        source_chat_id=chat_id.strip(),
    )
    return {"status": "success", "memory": record}


# ── Tool 2: Search Hot Memory ─────────────────────────────────────────────────

def memory_search_hot(
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Search working session (Hot) memory semantically."""
    if not query or not isinstance(query, str) or not query.strip():
        return {"status": "error", "error": {"code": "INVALID_QUERY", "message": "query is required."}}

    uid = user_id or config.DEFAULT_USER_ID
    clamped_limit = max(1, min(int(limit or 5), 20))
    hits = memory_vector_index.search_memories(
        query=query.strip(),
        user_id=uid,
        tier="hot",
        category=category.strip() if category else None,
        limit=clamped_limit,
    )
    return {"status": "success", "count": len(hits), "memories": hits}


# ── Tool 3: Store Cold Memory ─────────────────────────────────────────────────

def memory_store_cold(
    user_id: Optional[str] = None,
    content: Optional[str] = None,
    category: Optional[str] = None,
    importance: Optional[float] = None,
    confidence: Optional[float] = None,
    is_global: Optional[bool] = False,
    chat_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Store long-term permanent knowledge."""
    if not content or not isinstance(content, str) or not content.strip():
        return {"status": "error", "error": {"code": "INVALID_MEMORY_CONTENT", "message": "content is required."}}

    if not category or not isinstance(category, str) or not category.strip():
        return {"status": "error", "error": {"code": "INVALID_MEMORY_CATEGORY", "message": f"category is required. Allowed: {sorted(list(ALLOWED_CATEGORIES))}"}}

    try:
        val_cat = validate_category(str(category))
    except ValueError as e:
        return {"status": "error", "error": {"code": "INVALID_MEMORY_CATEGORY", "message": str(e)}}

    uid = user_id or config.DEFAULT_USER_ID
    imp = max(0.0, min(float(importance if importance is not None else 0.5), 1.0))
    conf = max(0.0, min(float(confidence if confidence is not None else 1.0), 1.0))

    record = memory_engine.store_memory(
        user_id=uid,
        content=content.strip(),
        tier="cold",
        category=val_cat,
        importance=imp,
        confidence=conf,
        is_global=bool(is_global),
        source_chat_id=chat_id,
    )
    return {"status": "success", "memory": record}


# ── Tool 4: Search Cold Memory ────────────────────────────────────────────────

def memory_search_cold(
    query: Optional[str] = None,
    user_id: Optional[str] = None,
    category: Optional[str] = None,
    limit: Optional[int] = 5,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Search long-term archival (Cold) memory semantically."""
    if not query or not isinstance(query, str) or not query.strip():
        return {"status": "error", "error": {"code": "INVALID_QUERY", "message": "query is required."}}

    uid = user_id or config.DEFAULT_USER_ID
    clamped_limit = max(1, min(int(limit or 5), 20))
    hits = memory_vector_index.search_memories(
        query=query.strip(),
        user_id=uid,
        tier="cold",
        category=category.strip() if category else None,
        limit=clamped_limit,
    )
    return {"status": "success", "count": len(hits), "memories": hits}


# ── Tool 5: Promote Hot to Cold ───────────────────────────────────────────────

def memory_promote_to_cold(
    memory_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Promote a session memory to durable long-term storage."""
    if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
        return {"status": "error", "error": {"code": "INVALID_MEMORY_ID", "message": "memory_id is required."}}

    success = memory_engine.promote_memory(memory_id.strip())
    if not success:
        return {"status": "error", "error": {"code": "MEMORY_NOT_FOUND", "message": f"Memory '{memory_id}' not found."}}
    return {"status": "success", "promoted_id": memory_id.strip()}


# ── Tool 6: Get Context ───────────────────────────────────────────────────────

def memory_get_context(
    query: str,
    chat_id: Optional[str] = None,
    user_id: Optional[str] = None,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Assemble composite context (Recent Chat + Global Directives + Hot/Cold Recall)."""
    uid = user_id or config.DEFAULT_USER_ID
    budget = max_tokens or config.CONTEXT_MEMORY_BUDGET_TOKENS

    # 1. Global memories
    globals_list = memory_engine.list_memories(user_id=uid, is_global=True, status="active", limit=config.GLOBAL_MEMORY_MAX_ITEMS)
    global_lines = [f"• [GLOBAL RULE] {m['content']}" for m in globals_list]

    # 2. Recent chat messages
    recent_msgs = []
    if chat_id:
        msgs = memory_engine.get_messages(chat_id=chat_id, limit=config.RECENT_CHAT_MAX_MESSAGES)
        recent_msgs = [f"[{m['role'].upper()}]: {m['content']}" for m in msgs]

    # 3. Semantic search on Hot and Cold tiers
    hot_hits = memory_vector_index.search_memories(query=query, user_id=uid, tier="hot", limit=3)
    cold_hits = memory_vector_index.search_memories(query=query, user_id=uid, tier="cold", limit=3)

    relevant_lines = []
    for h in hot_hits:
        relevant_lines.append(f"• [HOT MEMORY - {h['category']}] {h['content']} (relevance: {h['similarity']})")
    for c in cold_hits:
        relevant_lines.append(f"• [COLD MEMORY - {c['category']}] {c['content']} (relevance: {c['similarity']})")

    # Combine blocks
    sections = []
    if global_lines:
        sections.append("=== GLOBAL DIRECTIVES ===\n" + "\n".join(global_lines))
    if recent_msgs:
        sections.append("=== RECENT CONVERSATION ===\n" + "\n".join(recent_msgs))
    if relevant_lines:
        sections.append("=== RELEVANT MEMORIES ===\n" + "\n".join(relevant_lines))

    assembled_text = "\n\n".join(sections)
    return {
        "status": "success",
        "context_text": assembled_text,
        "token_estimate": len(assembled_text.split()),
        "breakdown": {
            "globals_count": len(global_lines),
            "recent_messages_count": len(recent_msgs),
            "hot_hits_count": len(hot_hits),
            "cold_hits_count": len(cold_hits),
        },
    }


# ── Tool 7: Decay Check ───────────────────────────────────────────────────────

def memory_decay_check(
    user_id: Optional[str] = None,
    threshold: Optional[float] = 0.3,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Trigger decay evaluation for stale hot memories."""
    uid = user_id or config.DEFAULT_USER_ID
    thresh = float(threshold if threshold is not None else 0.3)
    decayed = memory_engine.apply_decay(user_id=uid, threshold=thresh)
    return {"status": "success", "decayed_count": decayed}


# ── Tool 8: Delete Memory ─────────────────────────────────────────────────────

def memory_delete(
    memory_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Permanently delete a memory record."""
    if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
        return {"status": "error", "error": {"code": "INVALID_MEMORY_ID", "message": "memory_id is required."}}

    success = memory_engine.delete_memory(memory_id.strip())
    return {"status": "success", "deleted_id": memory_id.strip(), "success": success}


# ── Standardized Function Calling Schemas ─────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "memory_store_hot",
            "description": "Store working session memory into Hot tier. Requires chat_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The fact or task detail to remember."},
                    "category": {"type": "string", "enum": list(ALLOWED_CATEGORIES)},
                    "chat_id": {"type": "string", "description": "Active session chat ID."},
                    "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["content", "category", "chat_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search_hot",
            "description": "Search current working session memories semantically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query."},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_store_cold",
            "description": "Store permanent, long-term archival knowledge into Cold tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Permanent fact, preference, or rule."},
                    "category": {"type": "string", "enum": list(ALLOWED_CATEGORIES)},
                    "is_global": {"type": "boolean", "description": "If true, injected on every conversation turn."},
                },
                "required": ["content", "category"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_search_cold",
            "description": "Search long-term archival memories semantically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "limit": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_promote_to_cold",
            "description": "Promote a session memory from Hot to Cold tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "ID of memory to promote."},
                },
                "required": ["memory_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_get_context",
            "description": "Assemble composite memory context (Recent Chat + Global Rules + Semantic Recall).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User prompt or topic."},
                    "chat_id": {"type": "string", "description": "Current conversation ID."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_decay_check",
            "description": "Evaluate and decay under-used session memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threshold": {"type": "number", "default": 0.3},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "memory_delete",
            "description": "Delete a memory by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string", "description": "Memory ID to delete."},
                },
                "required": ["memory_id"],
            },
        },
    },
]
