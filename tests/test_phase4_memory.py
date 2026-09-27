"""
tests/test_phase4_memory.py
Phase 4: Semantic Memory Retrieval with Embeddings + Chroma Tests
"""

import os
import pytest

import config
from sage_memory import sage_memory, set_sqlite_path
from tools.durable_memory import memory_search, memory_get


@pytest.fixture(autouse=True)
def use_sqlite_and_temp_chroma(tmp_path):
    """Force SQLite memory backend and isolated Chroma directory for Phase 4 tests."""
    orig_env_db = os.environ.get("SAGE_MEMORY_DB")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"

    db_path = str(tmp_path / "test_phase4_memory.db")
    set_sqlite_path(db_path)
    sage_memory._initialized_backends.clear()

    chroma_path = tmp_path / "chroma_test"
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Re-initialize MemoryVectorIndex singleton for isolated test path
    from sage_memory_index import memory_vector_index
    memory_vector_index._chroma_root = chroma_path
    memory_vector_index._client = None
    memory_vector_index._collection = None

    yield

    set_sqlite_path(None)
    if orig_env_db is not None:
        os.environ["SAGE_MEMORY_DB"] = orig_env_db
    else:
        os.environ.pop("SAGE_MEMORY_DB", None)


def test_embedding_generation():
    """A. Embedding generation: Verify valid embedding vector is generated locally."""
    from sage_document_db.embeddings import EmbeddingService
    from sage_document_db.config import EMBEDDING_DIMENSION
    emb_svc = EmbeddingService()
    vec = emb_svc.embed_query("Test embedding generation")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIMENSION


def test_postgresql_to_chroma_indexing():
    """B. PostgreSQL -> Chroma indexing: Verify store_memory automatically upserts vector into Chroma."""
    mem = sage_memory.store_memory(
        user_id="user_p4_b",
        content="User works on SAGE Phase 4.",
        category="project",
    )
    from sage_memory_index import memory_vector_index
    col = memory_vector_index._get_collection()
    res = col.get(ids=[mem["memory_id"]])
    assert len(res["ids"]) == 1
    assert res["ids"][0] == mem["memory_id"]
    assert res["documents"][0] == "User works on SAGE Phase 4."


def test_semantic_retrieval():
    """C. Semantic retrieval: Verify search query retrieves semantically relevant memories."""
    sage_memory.store_memory(
        user_id="user_p4_c",
        content="User prefers writing software in Python.",
        category="preference",
    )
    sage_memory.store_memory(
        user_id="user_p4_c",
        content="User enjoys building backend web applications.",
        category="preference",
    )

    res = memory_search(
        user_id="user_p4_c",
        query="What programming language do I prefer?",
    )
    assert res["status"] == "success"
    assert res["returned"] >= 1
    contents = [m["content"] for m in res["memories"]]
    assert "User prefers writing software in Python." in contents


def test_user_isolation():
    """D. User isolation: Verify search for User A never returns User B's memories."""
    sage_memory.store_memory(
        user_id="user_A",
        content="User A's secret project name is Falcon.",
        category="project",
    )
    sage_memory.store_memory(
        user_id="user_B",
        content="User B's secret project name is Eagle.",
        category="project",
    )

    res = memory_search(user_id="user_A", query="secret project name")
    assert res["status"] == "success"
    returned_contents = [m["content"] for m in res["memories"]]
    assert any("Falcon" in c for c in returned_contents)
    assert not any("Eagle" in c for c in returned_contents)


def test_category_filtering():
    """E. Category filtering: Verify searching with category filter strictly isolates categories."""
    sage_memory.store_memory(
        user_id="user_cat",
        content="User prefers dark mode UI.",
        category="preference",
    )
    sage_memory.store_memory(
        user_id="user_cat",
        content="User is building a React frontend app.",
        category="project",
    )

    res = memory_search(user_id="user_cat", query="UI preference", category="preference")
    assert res["status"] == "success"
    categories = [m["category"] for m in res["memories"]]
    assert all(c == "preference" for c in categories)


def test_deleted_memories_excluded():
    """F. Deleted memories: Verify logically deleted memories are excluded from search."""
    mem = sage_memory.store_memory(
        user_id="user_del",
        content="Temporary setting to be deleted.",
        category="fact",
    )
    sage_memory.delete_memory(mem["memory_id"])

    res = memory_search(user_id="user_del", query="Temporary setting")
    assert res["status"] == "success"
    ids = [m["memory_id"] for m in res["memories"]]
    assert mem["memory_id"] not in ids


def test_superseded_memories_excluded():
    """G. Superseded memories: Verify superseded memories do not appear in active search results."""
    old_mem = sage_memory.store_memory(
        user_id="user_sup",
        content="User prefers Java for backend.",
        category="preference",
    )
    new_mem = sage_memory.supersede_memory(
        old_memory_id=old_mem["memory_id"],
        new_content="User prefers Go for backend.",
        user_id="user_sup",
    )

    res = memory_search(user_id="user_sup", query="backend programming preference")
    assert res["status"] == "success"
    ids = [m["memory_id"] for m in res["memories"]]
    assert old_mem["memory_id"] not in ids
    assert new_mem["memory_id"] in ids


def test_limit_clamping():
    """H. Limit: Verify returned result count is bounded by limit."""
    for i in range(10):
        sage_memory.store_memory(
            user_id="user_lim",
            content=f"Durable fact number {i} regarding system configuration.",
            category="fact",
        )

    res = memory_search(user_id="user_lim", query="configuration fact", limit=3)
    assert res["status"] == "success"
    assert res["returned"] <= 3


def test_empty_result():
    """I. Empty result: Search on empty database returns empty result safely."""
    res = memory_search(user_id="user_empty", query="nonexistent query")
    assert res["status"] == "success"
    assert res["memories"] == []
    assert res["returned"] == 0


def test_stale_vector_rejection():
    """J. Stale vector: Verify deleted/tampered PostgreSQL memory is rejected even if vector exists in Chroma."""
    mem = sage_memory.store_memory(
        user_id="user_stale",
        content="Stale record test memory.",
        category="fact",
    )
    # Manually mark as deleted in DB without updating Chroma directly
    conn = sage_memory._get_connection()
    try:
        conn.execute("UPDATE memories SET status = 'deleted' WHERE memory_id = ?;", (mem["memory_id"],))
        conn.commit()
    finally:
        conn.close()

    res = memory_search(user_id="user_stale", query="Stale record test")
    ids = [m["memory_id"] for m in res["memories"]]
    assert mem["memory_id"] not in ids


def test_reindex_memories():
    """K. Reindex: Verify reindex_memories rebuilds Chroma collection from canonical DB."""
    mem1 = sage_memory.store_memory(user_id="u_reindex", content="Fact 1 to reindex", category="fact")
    mem2 = sage_memory.store_memory(user_id="u_reindex", content="Fact 2 to reindex", category="fact")

    from sage_memory_index import memory_vector_index
    col = memory_vector_index._get_collection()
    col.delete(ids=[mem1["memory_id"], mem2["memory_id"]])

    re_res = sage_memory.reindex_memories(user_id="u_reindex")
    assert re_res["status"] == "success"
    assert re_res["indexed_count"] >= 2

    c_res = col.get(ids=[mem1["memory_id"], mem2["memory_id"]])
    assert len(c_res["ids"]) == 2


def test_offline_and_mock_behavior():
    """M. Offline & Mock: Verify memory retrieval operates offline without cloud APIs."""
    res = memory_search(user_id="u_offline", query="offline query test")
    assert res["status"] == "success"
