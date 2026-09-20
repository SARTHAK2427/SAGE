"""
tests/test_phase2_memory.py
Phase 2: Global Facts / Durable Memory Store Tests
"""

import os
import time
import pytest

import config
from sage_memory import sage_memory, set_sqlite_path, validate_category, ALLOWED_CATEGORIES


@pytest.fixture(autouse=True)
def use_sqlite_memory(tmp_path):
    """Force SQLite memory database backend for all Phase 2 tests."""
    orig_env = os.environ.get("SAGE_MEMORY_DB")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"

    db_path = str(tmp_path / "test_phase2_memory.db")
    set_sqlite_path(db_path)
    sage_memory._initialized_backends.clear()

    yield

    set_sqlite_path(None)
    if orig_env is not None:
        os.environ["SAGE_MEMORY_DB"] = orig_env
    else:
        os.environ.pop("SAGE_MEMORY_DB", None)


def test_store_memory_basic():
    """A. Store memory and verify created object fields."""
    mem = sage_memory.store_memory(
        user_id="user_123",
        content="User prefers Python over Java.",
        category="preference",
        importance=0.8,
        confidence=0.9,
        source_chat_id="chat_001",
    )
    assert mem["memory_id"] is not None
    assert mem["user_id"] == "user_123"
    assert mem["content"] == "User prefers Python over Java."
    assert mem["category"] == "preference"
    assert mem["importance"] == 0.8
    assert mem["confidence"] == 0.9
    assert mem["source_chat_id"] == "chat_001"
    assert mem["status"] == "active"
    assert mem["supersedes_memory_id"] is None

    # Verify retrieval by get_memory
    fetched = sage_memory.get_memory(mem["memory_id"])
    assert fetched == mem


def test_required_fields_validation():
    """B. Verify missing user_id or content raises ValueError."""
    with pytest.raises(ValueError, match="user_id is required"):
        sage_memory.store_memory(user_id="", content="Fact content")

    with pytest.raises(ValueError, match="content is required"):
        sage_memory.store_memory(user_id="user_1", content="  ")


def test_category_validation():
    """C. Verify category validation enforces controlled categories."""
    for cat in ALLOWED_CATEGORIES:
        validated = validate_category(cat)
        assert validated == cat
        mem = sage_memory.store_memory(user_id="u1", content=f"Sample {cat}", category=cat)
        assert mem["category"] == cat

    with pytest.raises(ValueError, match="Invalid category"):
        sage_memory.store_memory(user_id="u1", content="Invalid cat test", category="random_custom")


def test_importance_and_confidence_clamping():
    """D. Verify importance and confidence values are clamped to [0.0, 1.0]."""
    mem_low = sage_memory.store_memory(user_id="u1", content="Low values", importance=-0.5, confidence=-2.0)
    assert mem_low["importance"] == 0.0
    assert mem_low["confidence"] == 0.0

    mem_high = sage_memory.store_memory(user_id="u1", content="High values", importance=1.5, confidence=3.0)
    assert mem_high["importance"] == 1.0
    assert mem_high["confidence"] == 1.0

    mem_mid = sage_memory.store_memory(user_id="u1", content="Mid values", importance=0.5, confidence=0.7)
    assert mem_mid["importance"] == 0.5
    assert mem_mid["confidence"] == 0.7


def test_user_isolation():
    """E. Verify memories for user_A never leak to user_B."""
    mem_a = sage_memory.store_memory(user_id="user_A", content="Secret A")
    mem_b = sage_memory.store_memory(user_id="user_B", content="Secret B")

    mems_a = sage_memory.list_memories("user_A")
    mems_b = sage_memory.list_memories("user_B")

    assert len(mems_a) == 1
    assert mems_a[0]["memory_id"] == mem_a["memory_id"]

    assert len(mems_b) == 1
    assert mems_b[0]["memory_id"] == mem_b["memory_id"]


def test_active_status_filtering():
    """F. Verify list_memories returns only active memories."""
    user = "user_filter_test"
    m_active = sage_memory.store_memory(user_id=user, content="Active memory")
    m_superseded = sage_memory.store_memory(user_id=user, content="Old memory")
    m_deleted = sage_memory.store_memory(user_id=user, content="Deleted memory")

    sage_memory.supersede_memory(m_superseded["memory_id"], new_content="New memory replacement", user_id=user)
    sage_memory.delete_memory(m_deleted["memory_id"])

    active_list = sage_memory.list_memories(user)
    active_ids = {m["memory_id"] for m in active_list}

    assert m_active["memory_id"] in active_ids
    assert m_superseded["memory_id"] not in active_ids
    assert m_deleted["memory_id"] not in active_ids


def test_update_memory():
    """G. Verify updating memory fields updates content and updated_at."""
    mem = sage_memory.store_memory(user_id="u1", content="Original content", importance=0.4)
    time.sleep(0.01)

    updated = sage_memory.update_memory(
        mem["memory_id"],
        content="Updated content",
        importance=0.9,
    )
    assert updated is not None
    assert updated["content"] == "Updated content"
    assert updated["importance"] == 0.9
    assert updated["updated_at"] >= mem["updated_at"]


def test_logical_delete_memory():
    """H. Verify logical deletion sets status='deleted'."""
    mem = sage_memory.store_memory(user_id="u1", content="To be deleted")
    success = sage_memory.delete_memory(mem["memory_id"])
    assert success is True

    fetched = sage_memory.get_memory(mem["memory_id"])
    assert fetched["status"] == "deleted"

    active_memories = sage_memory.list_memories("u1")
    assert not any(m["memory_id"] == mem["memory_id"] for m in active_memories)


def test_supersede_memory():
    """I. Verify memory supersession relationship and status changes."""
    old_mem = sage_memory.store_memory(user_id="u1", content="User prefers Java.", category="preference")
    new_mem = sage_memory.supersede_memory(
        old_memory_id=old_mem["memory_id"],
        new_content="User prefers Python.",
        category="preference",
    )

    assert new_mem["supersedes_memory_id"] == old_mem["memory_id"]
    assert new_mem["status"] == "active"
    assert new_mem["content"] == "User prefers Python."

    fetched_old = sage_memory.get_memory(old_mem["memory_id"])
    assert fetched_old["status"] == "superseded"


def test_db_persistence_across_connections(tmp_path):
    """J. Verify memories persist across separate DB connections."""
    db_file = str(tmp_path / "persistent_mem.db")
    set_sqlite_path(db_file)
    sage_memory._initialized_backends.clear()

    mem = sage_memory.store_memory(user_id="u_persistent", content="Persistent fact")

    # Clear cached backends and reconnect
    sage_memory._initialized_backends.clear()
    fetched = sage_memory.get_memory(mem["memory_id"])
    assert fetched is not None
    assert fetched["content"] == "Persistent fact"
