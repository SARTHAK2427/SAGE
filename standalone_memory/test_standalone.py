"""
standalone_memory/test_standalone.py
Deterministic pytest suite for the standalone memory package.
Can run anywhere without an LLM or GPU.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest

from standalone_memory import (
    MemoryEngine,
    MemoryVectorIndex,
    EmbeddingService,
    memory_store_hot,
    memory_search_hot,
    memory_store_cold,
    memory_search_cold,
    memory_promote_to_cold,
    memory_get_context,
    memory_delete,
    memory_engine,
    memory_vector_index,
)


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Isolate DB and Chroma files for each test."""
    db_path = str(tmp_path / "test_memory.db")
    chroma_path = str(tmp_path / "test_chroma")

    monkeypatch.setenv("STANDALONE_MEMORY_MOCK", "1")
    monkeypatch.setattr("standalone_memory.config.SQLITE_DB_PATH", db_path)
    monkeypatch.setattr("standalone_memory.config.CHROMA_PERSIST_DIR", chroma_path)

    memory_engine.sqlite_path = db_path
    memory_engine._initialized = False

    memory_vector_index.close()
    memory_vector_index._chroma_dir = Path(chroma_path)
    memory_vector_index._collections.clear()
    memory_vector_index._client = None
    memory_vector_index._emb = EmbeddingService()

    yield

    memory_vector_index.close()


def test_crud_and_status():
    user = "u1"
    # Store
    res = memory_store_cold(user_id=user, content="User prefers dark mode", category="preference")
    assert res["status"] == "success"
    mid = res["memory"]["memory_id"]

    # Retrieve
    mem = memory_engine.get_memory(mid)
    assert mem is not None
    assert mem["content"] == "User prefers dark mode"
    assert mem["status"] == "active"

    # List
    active_mems = memory_engine.list_memories(user_id=user, status="active")
    assert any(m["memory_id"] == mid for m in active_mems)

    # Delete
    del_res = memory_delete(memory_id=mid)
    assert del_res["status"] == "success"
    deleted_mem = memory_engine.get_memory(mid)
    assert deleted_mem["status"] == "deleted"


def test_tier_isolation_and_search():
    user = "u_iso"
    # Store Hot
    h_res = memory_store_hot(user_id=user, chat_id="c1", content="Working on Task Alpha 42", category="task")
    assert h_res["status"] == "success"
    h_id = h_res["memory"]["memory_id"]

    # Store Cold
    c_res = memory_store_cold(user_id=user, content="Company policy on code review", category="instruction")
    assert c_res["status"] == "success"
    c_id = c_res["memory"]["memory_id"]

    # Hot search should find hot, NOT cold
    hot_hits = memory_search_hot(query="Task Alpha", user_id=user)
    assert hot_hits["status"] == "success"
    assert any(m["memory_id"] == h_id for m in hot_hits["memories"])
    assert not any(m["memory_id"] == c_id for m in hot_hits["memories"])

    # Cold search should find cold, NOT hot
    cold_hits = memory_search_cold(query="code review policy", user_id=user)
    assert cold_hits["status"] == "success"
    assert any(m["memory_id"] == c_id for m in cold_hits["memories"])
    assert not any(m["memory_id"] == h_id for m in cold_hits["memories"])


def test_promotion_from_hot_to_cold():
    user = "u_prom"
    h_res = memory_store_hot(user_id=user, chat_id="c1", content="Discovered bug in authentication middleware", category="technical")
    mid = h_res["memory"]["memory_id"]

    prom = memory_promote_to_cold(memory_id=mid)
    assert prom["status"] == "success"

    # Verify DB tier is now cold
    mem = memory_engine.get_memory(mid)
    assert mem["memory_tier"] == "cold"

    # Search cold tier should now find it
    cold_search = memory_search_cold(query="authentication middleware bug", user_id=user)
    assert any(m["memory_id"] == mid for m in cold_search["memories"])


def test_compaction():
    user = "u_comp"
    m1 = memory_engine.store_memory(user_id=user, content="Server node 1 high CPU", tier="cold", category="technical")
    m2 = memory_engine.store_memory(user_id=user, content="Server node 2 high CPU", tier="cold", category="technical")

    compacted = memory_engine.compact_memories(
        user_id=user,
        memory_ids=[m1["memory_id"], m2["memory_id"]],
        summary_content="Cluster nodes 1 and 2 experienced high CPU load simultaneously.",
    )

    assert compacted["category"] == "summary"
    assert memory_engine.get_memory(m1["memory_id"])["status"] == "compacted"
    assert memory_engine.get_memory(m2["memory_id"])["status"] == "compacted"


def test_chat_ledger_and_context_assembly():
    user = "u_ctx"
    chat_id = "chat_456"

    # Add global directive
    memory_store_cold(user_id=user, content="Be polite and concise.", category="instruction", is_global=True)

    # Add messages
    memory_engine.add_message(chat_id=chat_id, role="user", content="Hello, who are you?", user_id=user)
    memory_engine.add_message(chat_id=chat_id, role="assistant", content="I am your assistant.", user_id=user)

    # Store a relevant memory
    memory_store_cold(user_id=user, content="Project Phoenix release date is November 1st.", category="project")

    # Get composite context
    ctx = memory_get_context(query="When is Project Phoenix scheduled?", chat_id=chat_id, user_id=user)
    assert ctx["status"] == "success"
    text = ctx["context_text"]

    assert "=== GLOBAL DIRECTIVES ===" in text
    assert "Be polite and concise." in text
    assert "=== RECENT CONVERSATION ===" in text
    assert "Project Phoenix release date is November 1st." in text


def test_validation_errors():
    # Empty content
    r1 = memory_store_hot(user_id="u", chat_id="c", content="", category="fact")
    assert r1["status"] == "error"
    assert r1["error"]["code"] == "INVALID_MEMORY_CONTENT"

    # Invalid category
    r2 = memory_store_hot(user_id="u", chat_id="c", content="Hello", category="invalid_cat_xyz")
    assert r2["status"] == "error"
    assert r2["error"]["code"] == "INVALID_MEMORY_CATEGORY"

    # Missing chat_id for hot tier
    r3 = memory_store_hot(user_id="u", chat_id="", content="Hello", category="fact")
    assert r3["status"] == "error"
    assert r3["error"]["code"] == "INVALID_MEMORY_CHAT"
