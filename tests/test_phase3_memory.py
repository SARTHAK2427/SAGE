"""
tests/test_phase3_memory.py
Phase 3: Semantic Memory Decision & Retrieval Tests
"""

import os
import pytest

import config
from sage_memory import sage_memory, set_sqlite_path
from tools.durable_memory import memory_search, memory_get


@pytest.fixture(autouse=True)
def use_sqlite_memory(tmp_path):
    """Force SQLite memory database backend for all Phase 3 tests."""
    orig_env = os.environ.get("SAGE_MEMORY_DB")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"

    db_path = str(tmp_path / "test_phase3_memory.db")
    set_sqlite_path(db_path)
    sage_memory._initialized_backends.clear()

    yield

    set_sqlite_path(None)
    if orig_env is not None:
        os.environ["SAGE_MEMORY_DB"] = orig_env
    else:
        os.environ.pop("SAGE_MEMORY_DB", None)


def test_no_memory_decision():
    """A. Verify memory_needed=False decision produces empty results without querying."""
    res = memory_search(user_id="u1", memory_needed=False)
    assert res["status"] == "success"
    assert res["memories"] == []
    assert res["returned"] == 0


def test_search_decision_execution():
    """B. Verify memory_needed=True and operation='search' executes search."""
    sage_memory.store_memory(user_id="user_b", content="User prefers Python over Java.", category="preference")
    res = memory_search(user_id="user_b", memory_needed=True, operation="search", category="preference")

    assert res["status"] == "success"
    assert res["returned"] == 1
    assert res["memories"][0]["content"] == "User prefers Python over Java."


def test_category_filtering():
    """C. Verify only requested category is returned."""
    user = "user_cat"
    sage_memory.store_memory(user_id=user, content="Prefers dark mode", category="preference")
    sage_memory.store_memory(user_id=user, content="Working on SAGE project", category="project")
    sage_memory.store_memory(user_id=user, content="PostgreSQL decision", category="decision")

    res_pref = memory_search(user_id=user, category="preference")
    assert res_pref["returned"] == 1
    assert res_pref["memories"][0]["category"] == "preference"

    res_proj = memory_search(user_id=user, category="project")
    assert res_proj["returned"] == 1
    assert res_proj["memories"][0]["category"] == "project"


def test_user_isolation():
    """D. Verify user_A cannot retrieve user_B memory via search or get."""
    mem_a = sage_memory.store_memory(user_id="user_A", content="User A private fact")
    mem_b = sage_memory.store_memory(user_id="user_B", content="User B private fact")

    # Search isolation
    res_a = memory_search(user_id="user_A")
    res_a_ids = {m["memory_id"] for m in res_a["memories"]}
    assert mem_a["memory_id"] in res_a_ids
    assert mem_b["memory_id"] not in res_a_ids

    # Get isolation
    get_b_as_a = memory_get(user_id="user_A", memory_id=mem_b["memory_id"])
    assert get_b_as_a["status"] == "success"
    assert get_b_as_a["memory"] is None


def test_deleted_memories_excluded():
    """E. Verify deleted memories are excluded from memory_search and memory_get."""
    user = "user_del"
    mem = sage_memory.store_memory(user_id=user, content="Memory to delete")
    sage_memory.delete_memory(mem["memory_id"])

    res_search = memory_search(user_id=user)
    assert not any(m["memory_id"] == mem["memory_id"] for m in res_search["memories"])

    res_get = memory_get(user_id=user, memory_id=mem["memory_id"])
    assert res_get["memory"] is None


def test_superseded_memories_excluded():
    """F. Verify superseded memories are excluded from memory_search and memory_get."""
    user = "user_sup"
    old_mem = sage_memory.store_memory(user_id=user, content="Old preference")
    new_mem = sage_memory.supersede_memory(old_mem["memory_id"], new_content="New preference", user_id=user)

    res_search = memory_search(user_id=user)
    returned_ids = {m["memory_id"] for m in res_search["memories"]}
    assert old_mem["memory_id"] not in returned_ids
    assert new_mem["memory_id"] in returned_ids

    res_get = memory_get(user_id=user, memory_id=old_mem["memory_id"])
    assert res_get["memory"] is None


def test_limit_clamping():
    """G. Verify default limit=5 and maximum limit=20 clamping."""
    user = "user_limit"
    for i in range(25):
        sage_memory.store_memory(user_id=user, content=f"Fact {i}")

    # Default limit
    res_default = memory_search(user_id=user)
    assert res_default["returned"] == 5

    # Excessive limit -> clamped to 20
    res_excessive = memory_search(user_id=user, limit=100)
    assert res_excessive["returned"] == 20

    # Explicit small limit
    res_small = memory_search(user_id=user, limit=3)
    assert res_small["returned"] == 3


def test_invalid_operation_rejected():
    """H. Verify rejected model operations return error structure."""
    # 'store' is now a model-facing operation (Phase 5) via memory_store;
    # passing operation='store' to memory_search is an unknown operation
    for op in ("update", "delete", "supersede"):
        res = memory_search(user_id="u1", operation=op)
        assert res["status"] == "error"
        assert res["error"]["code"] == "INVALID_MEMORY_OPERATION"

    # 'store' is not a valid operation for memory_search (use memory_store instead)
    res_store = memory_search(user_id="u1", operation="store")
    assert res_store["status"] == "error"
    assert res_store["error"]["code"] == "UNKNOWN_MEMORY_OPERATION"


def test_invalid_category_rejected():
    """I. Verify unsupported categories return error structure."""
    res = memory_search(user_id="u1", category="arbitrary_invalid_cat")
    assert res["status"] == "error"
    assert res["error"]["code"] == "INVALID_MEMORY_CATEGORY"


def test_invalid_memory_id_rejected():
    """J. Verify malformed or empty memory_id returns error structure."""
    res_empty = memory_get(user_id="u1", memory_id="")
    assert res_empty["status"] == "error"
    assert res_empty["error"]["code"] == "INVALID_MEMORY_ID"

    res_none = memory_get(user_id="u1", memory_id=None)
    assert res_none["status"] == "error"
    assert res_none["error"]["code"] == "INVALID_MEMORY_ID"


def test_no_semantic_pretending_assertion():
    """K. Verify Phase 3 uses deterministic category/user filtering and NO fake keyword/vector matching."""
    user = "user_no_fake"
    sage_memory.store_memory(user_id=user, content="Alpha fact", category="fact")
    sage_memory.store_memory(user_id=user, content="Beta fact", category="fact")

    # Passing a query string does NOT filter out results via LIKE '%query%'
    res = memory_search(user_id=user, query="NonexistentSearchTermString", category="fact")
    assert res["status"] == "success"
    assert res["returned"] == 2  # Returns deterministic active memories without fake keyword filtering
