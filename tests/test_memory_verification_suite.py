"""Comprehensive deterministic verification suite for SAGE Memory System.

Tests run completely without models or GPU hardware.
Covers:
1. REST API wiring (/api/memory/status, /api/memory/recent-chat, /api/memory/memories,
   CRUD, /promote, /activity).
2. Tool dispatcher wiring for all 8 durable memory tools.
3. Deterministic memory lifecycle (hot vs cold, promotion, deletion, search, activity logging).
4. Memory budgeting and context bounds.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

import config
import sage_memory as sage_memory_module
import sage_memory_index
from app import app
from core.dispatcher import ToolRegistry
from sage_memory import (
    ALLOWED_CATEGORIES,
    SageMemory,
    sage_memory,
    set_sqlite_path,
)
from sage_memory_index import MemoryVectorIndex
from tools.durable_memory import (
    memory_delete,
    memory_promote,
    memory_search_cold,
    memory_search_hot,
    memory_store_cold,
    memory_store_hot,
    memory_summarize,
    memory_update,
)
from tools.registry import create_default_registry


@pytest.fixture(autouse=True)
def isolated_verification_environment(monkeypatch):
    """Provide isolated, clean SQLite and vector index instances for every test."""
    temp_dir = tempfile.TemporaryDirectory()
    root = Path(temp_dir.name)

    monkeypatch.setenv("SAGE_MEMORY_DB", "sqlite")
    test_db = str(root / "verification_memory.db")
    set_sqlite_path(test_db)
    sage_memory._initialized_backends.clear()

    # Isolated Chroma vector store
    index = MemoryVectorIndex(chroma_root=root / "verification_chroma")
    orig_index = sage_memory_index.memory_vector_index
    orig_store = sage_memory_module._sync_index_store
    orig_status = sage_memory_module._sync_index_status

    sage_memory_index.memory_vector_index = index
    sage_memory_module._sync_index_store = lambda m: index.index_memory(m)
    sage_memory_module._sync_index_status = lambda mid, st: index.update_memory_status(mid, st)

    yield

    index.close()
    sage_memory_index.memory_vector_index = orig_index
    sage_memory_module._sync_index_store = orig_store
    sage_memory_module._sync_index_status = orig_status
    set_sqlite_path(None)
    temp_dir.cleanup()


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════
# 1. REST API VERIFICATION
# ══════════════════════════════════════════════════════════════════════════

def test_api_memory_status(client):
    """GET /api/memory/status must return live system state with model status."""
    res = client.get("/api/memory/status")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "success"
    assert data["database"]["engine"] == "sqlite"
    assert data["database"]["status"] == "connected"
    assert "sqlite_path" in data["database"]

    assert data["vector_index"]["engine"] == "Chroma"
    assert "status" in data["vector_index"]
    assert "collections" in data["vector_index"]

    assert isinstance(data["model_runtime"]["available"], bool)
    assert data["model_runtime"]["label"] in {
        "2B CURATOR READY", "CURATOR NOT CONFIGURED", "CURATOR NEEDS ATTENTION", "CURATOR DISABLED",
    }
    assert data["model_runtime"]["provider"] in {"local_cpu", "local_gpu", "remote"}
    assert "detail" in data["model_runtime"]

    assert "counts" in data
    assert "hot_active" in data["counts"]
    assert "cold_active" in data["counts"]

    assert data["configuration"]["recent_chat_max_turns"] == 5
    assert data["configuration"]["global_budget_tokens"] == config.SAGE_GLOBAL_MEMORY_BUDGET_TOKENS


def test_api_memory_lifecycle_crud(client):
    """Verify full CRUD lifecycle via /api/memory endpoints."""
    # 1. Create hot memory with session chat_id
    create_hot_payload = {
        "content": "Working on authentication refactoring in auth.py",
        "category": "task",
        "tier": "hot",
        "chat_id": "test_session_123",
        "importance": 0.8,
    }
    res = client.post("/api/memory", json=create_hot_payload)
    assert res.status_code == 200, res.text
    hot_data = res.json()
    assert hot_data["status"] == "success"
    hot_mem = hot_data["memory"]
    hot_id = hot_mem["memory_id"]
    assert hot_mem["memory_tier"] == "hot"
    assert hot_mem["source_chat_id"] == "test_session_123"

    # 2. Create cold memory
    create_cold_payload = {
        "content": "Always prefer FastAPI over Flask for async endpoints",
        "category": "preference",
        "tier": "cold",
        "importance": 0.9,
    }
    res = client.post("/api/memory", json=create_cold_payload)
    assert res.status_code == 200
    cold_data = res.json()
    cold_id = cold_data["memory"]["memory_id"]
    assert cold_data["memory"]["memory_tier"] == "cold"

    # 3. List memories with tier filter
    res_list_hot = client.get("/api/memory/memories?tier=hot")
    assert res_list_hot.status_code == 200
    mems = res_list_hot.json()["memories"]
    assert any(m["memory_id"] == hot_id for m in mems)
    assert not any(m["memory_id"] == cold_id for m in mems)

    # 4. Search memories via text query
    res_search = client.get("/api/memory/memories?search=FastAPI")
    assert res_search.status_code == 200
    search_mems = res_search.json()["memories"]
    assert any(m["memory_id"] == cold_id for m in search_mems)

    # 5. In-place Update PUT
    update_payload = {
        "content": "Always prefer FastAPI over Flask (updated for async high performance)",
        "importance": 0.95,
    }
    res_put = client.put(f"/api/memory/{cold_id}", json=update_payload)
    assert res_put.status_code == 200
    updated_mem = res_put.json()["memory"]
    assert "updated for async high performance" in updated_mem["content"]

    # 6. Promote Hot to Cold POST
    res_promote = client.post(f"/api/memory/{hot_id}/promote", json={})
    assert res_promote.status_code == 200
    promoted_mem = res_promote.json()["memory"]
    assert promoted_mem["memory_tier"] == "cold"

    # 7. Delete Memory DELETE
    res_del = client.delete(f"/api/memory/{cold_id}")
    assert res_del.status_code == 200
    assert res_del.json()["deleted"] is True

    # Confirm it is no longer in active list
    res_after_del = client.get("/api/memory/memories?status=active")
    active_ids = [m["memory_id"] for m in res_after_del.json()["memories"]]
    assert cold_id not in active_ids


def test_api_recent_chat_memory(client):
    """GET /api/memory/recent-chat returns chronological completed turns."""
    chat_id = "test_chat_continuity_456"
    # Seed 7 messages in the ledger using the correct method: write_message
    for i in range(1, 8):
        role = "user" if i % 2 == 1 else "assistant"
        sage_memory.write_message(
            chat_id=chat_id,
            user_id=config.DEFAULT_USER_ID,
            role=role,
            content=f"Message turn #{i}",
        )

    res = client.get(f"/api/memory/recent-chat?chat_id={chat_id}&limit=5")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert data["chat_id"] == chat_id
    assert data["turn_count"] == 3
    assert data["count"] == 6
    messages = data["messages"]
    assert len(messages) == 6
    # Seven alternating messages form three completed pairs plus one open user
    # turn, which is retained in the ledger but excluded from Hot context.
    assert messages[0]["content"] == "Message turn #1"
    assert messages[-1]["content"] == "Message turn #6"
    assert "Recent conversation context" in data["label"]


def test_api_activity_audit_log(client):
    """GET /api/memory/activity must track memory operations.

    The DB field is 'event_type'. The API endpoint may expose it as either
    'action' or 'event_type' — accept both so this test is robust.
    """
    sage_memory.log_activity("STORE", config.DEFAULT_USER_ID, details="Stored hot memory mem_1")
    sage_memory.log_activity("PROMOTE", config.DEFAULT_USER_ID, details="Promoted mem_1 to cold")
    sage_memory.log_activity("SEARCH", config.DEFAULT_USER_ID, details="Query: 'database', matches: 2")

    res = client.get("/api/memory/activity?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"

    activities = data["activities"]
    assert len(activities) >= 3

    # Accept either 'action' or 'event_type' key from the API response
    def _get_event(act: dict) -> str:
        return act.get("action") or act.get("event_type") or ""

    events = [_get_event(a) for a in activities]
    assert "STORE" in events
    assert "PROMOTE" in events
    assert "SEARCH" in events


def test_api_bulk_hard_delete_global_memory_and_activity_log(client, monkeypatch):
    """Bulk actions permanently remove only selected Global records and audit rows."""
    monkeypatch.setattr(sage_memory_module, "_sync_index_delete", lambda *_args, **_kwargs: None)
    global_mem = sage_memory.store_memory(
        user_id=config.DEFAULT_USER_ID, content="Permanent global test record", category="fact",
        source_chat_id=None, memory_tier="cold",
    )
    chat_mem = sage_memory.store_memory(
        user_id=config.DEFAULT_USER_ID, content="Protected chat-scoped record", category="fact",
        source_chat_id="chat-protected", memory_tier="cold",
    )
    response = client.post("/api/memory/bulk-hard-delete", json={
        "memory_ids": [global_mem["memory_id"], chat_mem["memory_id"]],
    })
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["hard_deleted"] == 1
    assert global_mem["memory_id"] in payload["deleted_ids"]
    assert sage_memory.get_memory(global_mem["memory_id"]) is None
    assert sage_memory.get_memory(chat_mem["memory_id"]) is not None

    sage_memory.log_activity("SEARCH", config.DEFAULT_USER_ID, details="Hard-delete test activity")
    activities = sage_memory.get_recent_activity(config.DEFAULT_USER_ID, limit=10)
    target = next(activity for activity in activities if activity["details"] == "Hard-delete test activity")
    activity_response = client.post("/api/memory/activity/bulk-hard-delete", json={
        "activity_ids": [target["activity_id"]],
    })
    assert activity_response.status_code == 200, activity_response.text
    assert activity_response.json()["hard_deleted"] == 1
    remaining_ids = {activity["activity_id"] for activity in sage_memory.get_recent_activity(config.DEFAULT_USER_ID, limit=20)}
    assert target["activity_id"] not in remaining_ids


# ══════════════════════════════════════════════════════════════════════════
# 2. TOOL DISPATCHER WIRING FOR ALL 8 TOOLS
#
#   ToolRegistry uses a 2-part key: (tool_name, function_name).
#   All durable memory tools are registered under the "durable_memory" group.
#   Dispatch:  registry.dispatch("durable_memory", "memory_store_hot", **kwargs)
#   Returns:   ToolResult  with  .status / .result / .error attributes.
# ══════════════════════════════════════════════════════════════════════════

# The 8 required function names under the "durable_memory" group
_REQUIRED_MEMORY_FUNCTIONS = [
    "memory_store_hot",
    "memory_search_hot",
    "memory_store_cold",
    "memory_search_cold",
    "memory_promote",
    "memory_summarize",
    "memory_update",
    "memory_delete",
]


def test_tool_registry_has_all_eight_memory_tools():
    """Verify that create_default_registry registers all 8 durable memory functions
    under the 'durable_memory' tool group."""
    registry = create_default_registry()

    # 'durable_memory' must be a registered tool group
    tool_groups = registry.list_tools()          # returns list of group names
    assert "durable_memory" in tool_groups, (
        f"'durable_memory' group missing. Registered groups: {tool_groups}"
    )

    # Every required function must be reachable
    for fn_name in _REQUIRED_MEMORY_FUNCTIONS:
        assert registry.is_registered("durable_memory", fn_name), (
            f"Missing: durable_memory.{fn_name}"
        )
        fns = registry.list_functions("durable_memory")
        assert fn_name in fns, (
            f"durable_memory.{fn_name} missing from list_functions: {fns}"
        )


def test_tool_execution_via_registry():
    """Verify tool functions execute deterministically through registry dispatch.

    registry.dispatch(tool_name, function_name, **kwargs) -> ToolResult
    ToolResult.status: 'success' | 'error' | 'unknown_tool' | 'unknown_function'
    ToolResult.result: the dict returned by the tool function
    """
    registry = create_default_registry()
    chat_id = "dispatch_chat_001"

    # 1. memory_store_hot
    tr_hot = registry.dispatch(
        "durable_memory", "memory_store_hot",
        content="Current goal is to verify all memory tools.",
        category="task",
        chat_id=chat_id,
        importance=0.85,
    )
    assert tr_hot.status == "success", f"memory_store_hot dispatch failed: {tr_hot.error}"
    res_hot = tr_hot.result
    assert res_hot.get("status") == "success"
    mem_id = res_hot["memory"]["memory_id"]
    assert res_hot["memory"]["memory_tier"] == "hot"

    # 2. memory_search_hot
    tr_search = registry.dispatch(
        "durable_memory", "memory_search_hot",
        query="verify all memory tools",
        chat_id=chat_id,
        limit=5,
    )
    assert tr_search.status == "success", f"memory_search_hot dispatch failed: {tr_search.error}"
    res_search = tr_search.result
    assert res_search.get("status") == "success"
    assert len(res_search.get("memories", [])) >= 1

    # 3. memory_update
    tr_upd = registry.dispatch(
        "durable_memory", "memory_update",
        memory_id=mem_id,
        content="Updated: Current goal is to verify all memory tools thoroughly.",
        importance=0.95,
    )
    assert tr_upd.status == "success", f"memory_update dispatch failed: {tr_upd.error}"
    res_upd = tr_upd.result
    assert res_upd.get("status") == "success"
    assert "thoroughly" in res_upd["memory"]["content"]

    # 4. memory_promote
    tr_prom = registry.dispatch(
        "durable_memory", "memory_promote",
        memory_id=mem_id,
    )
    assert tr_prom.status == "success", f"memory_promote dispatch failed: {tr_prom.error}"
    res_prom = tr_prom.result
    assert res_prom.get("status") == "success"
    assert res_prom["memory"]["memory_tier"] == "cold"

    # 5. memory_delete
    tr_del = registry.dispatch(
        "durable_memory", "memory_delete",
        memory_id=mem_id,
    )
    assert tr_del.status == "success", f"memory_delete dispatch failed: {tr_del.error}"
    res_del = tr_del.result
    assert res_del.get("status") == "success"
    assert res_del.get("deleted") is True


# ══════════════════════════════════════════════════════════════════════════
# 3. DETERMINISTIC MEMORY CORE & BUDGETING
# ══════════════════════════════════════════════════════════════════════════

def test_hot_cold_tier_isolation():
    """Hot memory must require source_chat_id; cold memory is user-global."""
    # Attempting to store hot memory without chat_id must fail
    with pytest.raises(ValueError, match="source_chat_id is required"):
        sage_memory.store_memory(
            user_id="user_1",
            content="Invalid hot memory without chat",
            category="fact",
            memory_tier="hot",
            source_chat_id=None,
        )

    # Cold memory without chat_id succeeds
    cold = sage_memory.store_memory(
        user_id="user_1",
        content="Global user preference for dark theme",
        category="preference",
        memory_tier="cold",
    )
    assert cold["memory_tier"] == "cold"
    assert cold["source_chat_id"] is None


def test_budget_constraints_enforced():
    """Verify context and global memory budgets are strictly enforced.

    Exercises the exact orchestrator pipeline:
        1. list_memories(tier='cold', limit=SAGE_GLOBAL_MEMORY_MAX_ITEMS)
        2. _bounded_global_memory_context(memories, token_budget)

    This is the real path; get_global_memories_for_injection does NOT exist on SageMemory.
    """
    from orchestrator import _bounded_global_memory_context

    # Store more memories than the max_items cap allows
    for i in range(15):
        sage_memory.store_memory(
            user_id="user_budget",
            content=f"Durable fact number #{i}: Detailed architectural guideline for component {i}",
            category="project",
            memory_tier="cold",
            importance=0.5 + (i * 0.02),
        )

    # Step 1: canonical DB retrieval (enforces item cap at the query level)
    retrieved = sage_memory.list_memories(
        user_id="user_budget",
        memory_tier="cold",
        limit=config.SAGE_GLOBAL_MEMORY_MAX_ITEMS,
    )

    # Must be bounded by max_items at the DB layer
    assert len(retrieved) <= config.SAGE_GLOBAL_MEMORY_MAX_ITEMS

    # Step 2: token budgeting (same function the orchestrator calls)
    context_block, memory_ids, used_tokens = _bounded_global_memory_context(
        retrieved,
        config.SAGE_GLOBAL_MEMORY_BUDGET_TOKENS,
    )

    # Token budget must not be exceeded
    assert used_tokens <= config.SAGE_GLOBAL_MEMORY_BUDGET_TOKENS
    assert len(memory_ids) <= config.SAGE_GLOBAL_MEMORY_MAX_ITEMS
    # Context block must be a non-empty string when there are memories
    if memory_ids:
        assert isinstance(context_block, str) and len(context_block) > 0
