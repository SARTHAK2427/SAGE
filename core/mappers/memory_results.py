"""
SAGE/core/mappers/memory_results.py
Rich internal socket → Gemma-facing memory tool result projections.

Every mapper here is a pure projection:
    - Bounded output
    - Zero filesystem paths or internal storage details
    - Preserves semantic attributes needed by Gemma (memory_id, content, category,
      similarity, importance, created_at, is_summary)
"""

from __future__ import annotations

from typing import Any

from .gemma_results import strip_internal_fields


def map_memory_search_result(data: dict) -> dict:
    """Project memory search result -> Gemma-facing ranked matches.

    Input shape (from tool_memory_search_hot / tool_memory_search_cold):
        {"status": "success", "session_id": "...", "results": [...], "count": int}

    Gemma-facing shape:
        {
            "matches": [
                {
                    "memory_id": "mem_...",
                    "content": "...",
                    "memory_type": "hot"|"cold",
                    "category": "...",
                    "similarity": 0.85,
                    "importance": 0.7,
                    "created_at": "...",
                    "is_summary": false
                }
            ],
            "count": int,
            "session_id": "..." (if present)
        }
    """
    raw_results = data.get("results") or []
    matches = []
    for r in raw_results:
        if not isinstance(r, dict):
            continue
        sim = r.get("derived_similarity")
        if sim is None and "distance" in r:
            sim = round(max(0.0, min(1.0, 1.0 - float(r["distance"]))), 4)

        match_item = {
            "memory_id": r.get("memory_id", ""),
            "content": r.get("content", ""),
            "memory_type": r.get("memory_type", ""),
            "category": r.get("category", ""),
            "similarity": sim if sim is not None else 1.0,
            "importance": float(r.get("importance", 0.5)),
            "created_at": r.get("created_at"),
            "is_summary": bool(r.get("is_summary", False)),
        }
        matches.append(strip_internal_fields(match_item))

    res: dict[str, Any] = {
        "matches": matches,
        "count": len(matches),
    }
    if "session_id" in data and data["session_id"]:
        res["session_id"] = data["session_id"]
    return res


def map_memory_store_result(data: dict) -> dict:
    """Project memory store confirmation -> Gemma-facing response."""
    mem = data.get("memory") or {}
    return strip_internal_fields({
        "memory_id": data.get("memory_id") or mem.get("memory_id", ""),
        "status": "stored",
        "memory_type": mem.get("memory_type", ""),
        "category": mem.get("category", ""),
    })


def map_memory_update_result(data: dict) -> dict:
    """Project memory update confirmation -> Gemma-facing response."""
    mem = data.get("memory") or {}
    return strip_internal_fields({
        "memory_id": data.get("memory_id") or mem.get("memory_id", ""),
        "status": "updated",
        "memory_type": mem.get("memory_type", ""),
    })


def map_memory_delete_result(data: dict) -> dict:
    """Project memory delete confirmation -> Gemma-facing response."""
    return strip_internal_fields({
        "memory_id": data.get("memory_id", ""),
        "deleted": bool(data.get("deleted", True)),
    })


def map_memory_summarize_result(data: dict) -> dict:
    """Project memory summarization confirmation -> Gemma-facing response."""
    mem = data.get("memory") or {}
    return strip_internal_fields({
        "memory_id": data.get("memory_id") or mem.get("memory_id", ""),
        "status": "summarized",
        "is_summary": True,
        "parent_memory_ids": list(mem.get("parent_memory_ids") or []),
    })


def map_memory_promote_result(data: dict) -> dict:
    """Project memory promotion confirmation -> Gemma-facing response."""
    mem = data.get("memory") or {}
    return strip_internal_fields({
        "memory_id": data.get("memory_id") or mem.get("memory_id", ""),
        "promoted": bool(data.get("promoted", True)),
        "memory_type": "cold",
    })
