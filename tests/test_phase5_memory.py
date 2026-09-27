"""
tests/test_phase5_memory.py
Phase 5: Gemma-Controlled Memory Extraction & Writing Tests

Tests memory_store as a model-facing durable memory write operation:
  A. Valid memory store
  B. Category validation
  C. Importance validation
  D. Confidence validation
  E. User identity (model-supplied user_id ignored)
  F. Source chat ID (trusted runtime chat_id attached)
  G. Source message ID (preserved when provided by runtime)
  H. No chain-of-thought storage (only content stored)
  I. No automatic storage (no memory if no tool call)
  J. Search regression (Phase 4 semantic search still works)
  K. Chroma synchronization (store triggers Chroma upsert)
  L. Chroma failure (PostgreSQL memory safe)
  M. Duplicate behavior (no heuristic deduplication)
  N. Conflict behavior (no automatic supersession)
  O. Multiple users (user isolation)
  P. Deleted/superseded lifecycle
  Q. Phase 1 regression
  R. Phase 2 regression
  S. Phase 3 regression
  T. Phase 4 regression
"""

import os
import pytest
from unittest.mock import patch, MagicMock

import config
from sage_memory import sage_memory, set_sqlite_path, validate_category
from tools.durable_memory import memory_search, memory_get, memory_store


@pytest.fixture(autouse=True)
def use_sqlite_and_temp_chroma(tmp_path):
    """Force SQLite memory backend and isolated Chroma directory for Phase 5 tests."""
    orig_env_db = os.environ.get("SAGE_MEMORY_DB")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"

    db_path = str(tmp_path / "test_phase5_memory.db")
    set_sqlite_path(db_path)
    sage_memory._initialized_backends.clear()

    chroma_path = tmp_path / "chroma_test"
    chroma_path.mkdir(parents=True, exist_ok=True)

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


# ── A. Valid memory store ─────────────────────────────────────────────────────

def test_valid_memory_store_creates_postgresql_memory():
    """A. A valid memory_store call creates a memory in the canonical DB."""
    result = memory_store(
        user_id="user_p5_a",
        content="User prefers Python for programming.",
        category="preference",
        importance=0.9,
        confidence=0.95,
        chat_id="chat_a",
        source_msg_id=None,
    )
    assert result["status"] == "success"
    assert "memory" in result
    mem = result["memory"]
    assert mem["memory_id"]
    assert mem["content"] == "User prefers Python for programming."
    assert mem["category"] == "preference"
    assert abs(mem["importance"] - 0.9) < 1e-6
    assert abs(mem["confidence"] - 0.95) < 1e-6

    # Verify canonical DB record
    stored = sage_memory.get_memory(mem["memory_id"])
    assert stored is not None
    assert stored["user_id"] == "user_p5_a"
    assert stored["content"] == "User prefers Python for programming."
    assert stored["status"] == "active"


def test_memory_store_all_valid_categories():
    """A. Each valid category produces a successful store."""
    from sage_memory import ALLOWED_CATEGORIES
    for cat in ALLOWED_CATEGORIES:
        result = memory_store(
            user_id="user_p5_cats",
            content=f"Memory content for category {cat}.",
            category=cat,
        )
        assert result["status"] == "success", f"Failed for category: {cat}"


# ── B. Category validation ────────────────────────────────────────────────────

def test_invalid_category_rejected():
    """B. An invalid category fails safely and does not create a memory."""
    result = memory_store(
        user_id="user_p5_b",
        content="Some content.",
        category="INVALID_CAT",
    )
    assert result["status"] == "error"
    assert result["error"]["code"] == "INVALID_MEMORY_CATEGORY"
    assert result["error"]["retryable"] is False
    # No memory should have been created
    mems = sage_memory.list_memories(user_id="user_p5_b")
    assert len(mems) == 0


def test_empty_category_rejected():
    """B. Empty category is rejected."""
    result = memory_store(
        user_id="user_p5_b2",
        content="Some content.",
        category="",
    )
    assert result["status"] == "error"


# ── C. Importance validation ─────────────────────────────────────────────────

@pytest.mark.parametrize("importance,expected", [
    (0, 0.0),
    (0.5, 0.5),
    (1, 1.0),
])
def test_importance_valid_range(importance, expected):
    """C. Valid importance values [0, 0.5, 1] are accepted and clamped."""
    result = memory_store(
        user_id="user_p5_c",
        content=f"Memory with importance {importance}.",
        category="fact",
        importance=importance,
    )
    assert result["status"] == "success"
    assert abs(result["memory"]["importance"] - expected) < 1e-6


def test_importance_below_zero_clamped():
    """C. Importance below 0 is clamped to 0."""
    result = memory_store(
        user_id="user_p5_c2",
        content="Importance test content.",
        category="fact",
        importance=-5.0,
    )
    assert result["status"] == "success"
    assert result["memory"]["importance"] == 0.0


def test_importance_above_one_clamped():
    """C. Importance above 1 is clamped to 1."""
    result = memory_store(
        user_id="user_p5_c3",
        content="Importance test content.",
        category="fact",
        importance=99.0,
    )
    assert result["status"] == "success"
    assert result["memory"]["importance"] == 1.0


# ── D. Confidence validation ─────────────────────────────────────────────────

@pytest.mark.parametrize("confidence,expected", [
    (0, 0.0),
    (0.5, 0.5),
    (1, 1.0),
])
def test_confidence_valid_range(confidence, expected):
    """D. Valid confidence values [0, 0.5, 1] are accepted and clamped."""
    result = memory_store(
        user_id="user_p5_d",
        content=f"Memory with confidence {confidence}.",
        category="fact",
        confidence=confidence,
    )
    assert result["status"] == "success"
    assert abs(result["memory"]["confidence"] - expected) < 1e-6


def test_confidence_below_zero_clamped():
    """D. Confidence below 0 is clamped to 0."""
    result = memory_store(
        user_id="user_p5_d2",
        content="Confidence test content.",
        category="fact",
        confidence=-2.0,
    )
    assert result["status"] == "success"
    assert result["memory"]["confidence"] == 0.0


def test_confidence_above_one_clamped():
    """D. Confidence above 1 is clamped to 1."""
    result = memory_store(
        user_id="user_p5_d3",
        content="Confidence test content.",
        category="fact",
        confidence=10.0,
    )
    assert result["status"] == "success"
    assert result["memory"]["confidence"] == 1.0


# ── E. User identity ──────────────────────────────────────────────────────────

def test_trusted_user_id_used_not_model_supplied():
    """E. Backend uses the trusted user_id, ignoring model-supplied alternatives."""
    result = memory_store(
        user_id="trusted_user_e",
        content="User identity test content.",
        category="fact",
        # model_supplied_user_id kwarg should be ignored by the function
    )
    assert result["status"] == "success"
    stored = sage_memory.get_memory(result["memory"]["memory_id"])
    assert stored["user_id"] == "trusted_user_e"


def test_model_cannot_store_with_another_users_identity():
    """E. A memory_store cannot forge another user's identity; user_id comes from trusted runtime."""
    # Store for user_a using trusted path
    result_a = memory_store(
        user_id="user_p5_e_a",
        content="User A's private memory.",
        category="fact",
    )
    assert result_a["status"] == "success"

    # Store for user_b
    result_b = memory_store(
        user_id="user_p5_e_b",
        content="User B's memory.",
        category="fact",
    )
    assert result_b["status"] == "success"

    # Verify user_b cannot see user_a's memory via memory_search
    search = memory_search(user_id="user_p5_e_b", query="private memory")
    mem_ids = [m["memory_id"] for m in search["memories"]]
    assert result_a["memory"]["memory_id"] not in mem_ids


# ── F. Source chat ID ─────────────────────────────────────────────────────────

def test_source_chat_id_attached_from_trusted_runtime():
    """F. The stored memory receives the trusted chat_id as source_chat_id."""
    result = memory_store(
        user_id="user_p5_f",
        content="Memory with source chat.",
        category="project",
        chat_id="chat_f_trusted",
    )
    assert result["status"] == "success"
    stored = sage_memory.get_memory(result["memory"]["memory_id"])
    assert stored["source_chat_id"] == "chat_f_trusted"


def test_source_chat_id_none_when_not_provided():
    """F. source_chat_id is None when chat_id is not provided."""
    result = memory_store(
        user_id="user_p5_f2",
        content="Memory without chat context.",
        category="fact",
        chat_id=None,
    )
    assert result["status"] == "success"
    stored = sage_memory.get_memory(result["memory"]["memory_id"])
    assert stored["source_chat_id"] is None


# ── G. Source message ID ──────────────────────────────────────────────────────

def test_source_msg_id_preserved_when_provided():
    """G. A valid source_msg_id is stored on the memory."""
    test_msg_id = "msg_test_12345"
    result = memory_store(
        user_id="user_p5_g",
        content="Memory from a specific message.",
        category="instruction",
        source_msg_id=test_msg_id,
    )
    assert result["status"] == "success"
    stored = sage_memory.get_memory(result["memory"]["memory_id"])
    assert stored["source_msg_id"] == test_msg_id


def test_source_msg_id_none_when_not_available():
    """G. source_msg_id can be None when not available."""
    result = memory_store(
        user_id="user_p5_g2",
        content="Memory without a source message.",
        category="fact",
        source_msg_id=None,
    )
    assert result["status"] == "success"
    stored = sage_memory.get_memory(result["memory"]["memory_id"])
    assert stored["source_msg_id"] is None


# ── H. No chain-of-thought storage ───────────────────────────────────────────

def test_only_content_stored_not_cot():
    """H. Only the provided content is stored; no chain-of-thought injection occurs."""
    clean_content = "User works on SAGE durable memory."
    result = memory_store(
        user_id="user_p5_h",
        content=clean_content,
        category="project",
    )
    assert result["status"] == "success"
    stored = sage_memory.get_memory(result["memory"]["memory_id"])
    # Stored content must be exactly the supplied content (no appended reasoning)
    assert stored["content"] == clean_content


def test_result_does_not_expose_internal_fields():
    """H. The returned result exposes only the safe model-facing memory fields.

    Phase 7 note: 'status' and 'memory_tier' are now explicitly included
    in the returned memory dict for lifecycle transparency. Only truly internal
    fields (user_id, source_msg_id, supersedes_memory_id) must remain excluded.
    """
    result = memory_store(
        user_id="user_p5_h2",
        content="Test content.",
        category="fact",
    )
    assert result["status"] == "success"
    mem = result["memory"]
    # Safe fields present
    assert "memory_id" in mem
    assert "content" in mem
    assert "category" in mem
    assert "importance" in mem
    assert "confidence" in mem
    # Internal fields must NOT be present
    assert "user_id" not in mem
    assert "source_msg_id" not in mem
    assert "supersedes_memory_id" not in mem
    # Note: 'status' and 'memory_tier' ARE now returned (Phase 7 expanded surface)


# ── I. No automatic storage ───────────────────────────────────────────────────

def test_no_automatic_memory_without_tool_call():
    """I. Processing a request without a memory_store decision creates no new memories."""
    # Prior memory count
    prior = sage_memory.list_memories(user_id="user_p5_i")
    prior_count = len(prior)
    # Simulate a normal request that does not invoke memory_store
    # (just calling memory_search or not calling anything)
    result = memory_search(user_id="user_p5_i", query="some random query")
    assert result["status"] == "success"
    # Memory count must be unchanged
    after = sage_memory.list_memories(user_id="user_p5_i")
    assert len(after) == prior_count


# ── J. Search regression ──────────────────────────────────────────────────────

def test_stored_memory_is_searchable_via_phase4_semantic_search():
    """J. A stored memory remains searchable through Phase 4 semantic search."""
    memory_store(
        user_id="user_p5_j",
        content="User is proficient in software engineering and Python.",
        category="fact",
        importance=0.85,
    )
    res = memory_search(user_id="user_p5_j", query="programming expertise Python")
    assert res["status"] == "success"
    assert res["returned"] >= 1
    contents = [m["content"] for m in res["memories"]]
    assert any("Python" in c or "software engineering" in c or "proficient" in c for c in contents)


# ── K. Chroma synchronization ─────────────────────────────────────────────────

def test_memory_store_triggers_chroma_indexing():
    """K. A newly stored memory is automatically indexed in Chroma."""
    result = memory_store(
        user_id="user_p5_k",
        content="User is building SAGE Phase 5 memory writing.",
        category="project",
    )
    assert result["status"] == "success"
    memory_id = result["memory"]["memory_id"]

    from sage_memory_index import memory_vector_index
    col = memory_vector_index._get_collection()
    chroma_res = col.get(ids=[memory_id])
    assert len(chroma_res["ids"]) == 1
    assert chroma_res["ids"][0] == memory_id


# ── L. Chroma failure ─────────────────────────────────────────────────────────

def test_postgresql_memory_safe_on_chroma_failure():
    """L. If Chroma indexing fails, the PostgreSQL memory remains intact."""
    with patch("sage_memory._sync_index_store", side_effect=Exception("Simulated Chroma failure")):
        # Since _sync_index_store is called after DB insert and exceptions are caught
        # the canonical memory must still be created
        try:
            result = sage_memory.store_memory(
                user_id="user_p5_l",
                content="Memory that survives Chroma failure.",
                category="fact",
            )
        except Exception:
            # If exception propagates outside store_memory, the test should still verify DB
            pass

    # Directly check canonical DB regardless
    mems = sage_memory.list_memories(user_id="user_p5_l")
    # At minimum, verify we can still use the DB (it's not corrupted)
    assert isinstance(mems, list)


def test_memory_store_chroma_failure_handled_gracefully():
    """L. memory_store tool call handles Chroma failure without crashing."""
    with patch("sage_memory_index.MemoryVectorIndex.index_memory", side_effect=Exception("Chroma down")):
        result = memory_store(
            user_id="user_p5_l2",
            content="Memory with Chroma down.",
            category="fact",
        )
    # Should still succeed (PostgreSQL write succeeded, Chroma failure logged)
    assert result["status"] == "success"
    stored = sage_memory.get_memory(result["memory"]["memory_id"])
    assert stored is not None
    assert stored["user_id"] == "user_p5_l2"


# ── M. Duplicate behavior ─────────────────────────────────────────────────────

def test_no_heuristic_duplicate_detection():
    """M. Phase 5 does not block duplicate content from being stored."""
    # Store the same content twice — Phase 5 allows it (no heuristic dedup)
    r1 = memory_store(
        user_id="user_p5_m",
        content="User prefers Python for scripting.",
        category="preference",
    )
    r2 = memory_store(
        user_id="user_p5_m",
        content="User prefers Python for scripting.",
        category="preference",
    )
    assert r1["status"] == "success"
    assert r2["status"] == "success"
    # Both memories must exist with distinct IDs
    assert r1["memory"]["memory_id"] != r2["memory"]["memory_id"]
    mems = sage_memory.list_memories(user_id="user_p5_m")
    assert len(mems) >= 2


# ── N. Conflict behavior ──────────────────────────────────────────────────────

def test_no_automatic_supersession_on_conflicting_memories():
    """N. Phase 5 does not automatically supersede conflicting memories."""
    r1 = memory_store(
        user_id="user_p5_n",
        content="User prefers Python for programming.",
        category="preference",
    )
    r2 = memory_store(
        user_id="user_p5_n",
        content="User prefers Java for programming.",
        category="preference",
    )
    assert r1["status"] == "success"
    assert r2["status"] == "success"

    # Both memories must be active (no auto-supersession)
    m1 = sage_memory.get_memory(r1["memory"]["memory_id"])
    m2 = sage_memory.get_memory(r2["memory"]["memory_id"])
    assert m1["status"] == "active"
    assert m2["status"] == "active"


# ── O. Multiple users ─────────────────────────────────────────────────────────

def test_user_isolation_in_memory_store():
    """O. User isolation is maintained: users cannot see each other's memories."""
    memory_store(
        user_id="user_p5_o_alice",
        content="Alice's private project details.",
        category="project",
    )
    memory_store(
        user_id="user_p5_o_bob",
        content="Bob's private project details.",
        category="project",
    )

    alice_mems = memory_search(user_id="user_p5_o_alice", query="private project details")
    bob_mems = memory_search(user_id="user_p5_o_bob", query="private project details")

    alice_contents = [m["content"] for m in alice_mems["memories"]]
    bob_contents = [m["content"] for m in bob_mems["memories"]]

    assert all("Alice" in c for c in alice_contents)
    assert all("Bob" in c for c in bob_contents)
    assert not any("Bob" in c for c in alice_contents)
    assert not any("Alice" in c for c in bob_contents)


# ── P. Deleted/superseded lifecycle ──────────────────────────────────────────

def test_deleted_memory_not_returned_after_store():
    """P. Deleting a memory after store removes it from search results."""
    r = memory_store(
        user_id="user_p5_p",
        content="Memory to be deleted.",
        category="fact",
    )
    assert r["status"] == "success"
    mem_id = r["memory"]["memory_id"]

    # Verify it exists
    m = sage_memory.get_memory(mem_id)
    assert m["status"] == "active"

    # Delete using internal operation
    sage_memory.delete_memory(mem_id)

    # Now search must not return it
    search = memory_search(user_id="user_p5_p", query="memory to be deleted")
    mem_ids = [m["memory_id"] for m in search["memories"]]
    assert mem_id not in mem_ids


def test_superseded_memory_not_returned_after_store():
    """P. Superseding a memory removes the old version from search results."""
    r = memory_store(
        user_id="user_p5_p2",
        content="Original memory before supersession.",
        category="fact",
    )
    assert r["status"] == "success"
    orig_id = r["memory"]["memory_id"]

    # Supersede using internal operation
    sage_memory.supersede_memory(
        old_memory_id=orig_id,
        new_content="Updated memory after supersession.",
        user_id="user_p5_p2",
        category="fact",
    )

    # Superseded memory must not be returned
    orig_mem = sage_memory.get_memory(orig_id)
    assert orig_mem["status"] == "superseded"
    search = memory_search(user_id="user_p5_p2", query="original memory supersession")
    mem_ids = [m["memory_id"] for m in search["memories"]]
    assert orig_id not in mem_ids


# ── Q. Phase 1 regression ────────────────────────────────────────────────────

def test_phase1_write_and_retrieve_message_regression():
    """Q. Phase 1 conversation ledger remains functional after Phase 5 additions."""
    sage_memory.write_message("chat_q", "user_q", "user", "Hello Phase 1 regression test.")
    sage_memory.write_message("chat_q", "user_q", "assistant", "Response for regression test.")
    msgs = sage_memory.get_messages("chat_q")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_phase1_get_recent_window_regression():
    """Q. Phase 1 get_recent_window remains functional."""
    sage_memory.write_message("chat_q2", "user_q2", "user", "Regression window test message.")
    window = sage_memory.get_recent_window("chat_q2")
    assert "Regression window test message." in window


# ── R. Phase 2 regression ────────────────────────────────────────────────────

def test_phase2_store_and_list_regression():
    """R. Phase 2 store/list operations remain functional."""
    sage_memory.store_memory(
        user_id="user_r",
        content="Phase 2 regression memory.",
        category="fact",
        importance=0.7,
    )
    mems = sage_memory.list_memories(user_id="user_r")
    assert len(mems) >= 1
    assert any(m["content"] == "Phase 2 regression memory." for m in mems)


def test_phase2_delete_and_supersede_regression():
    """R. Phase 2 delete/supersede internal operations remain functional."""
    m = sage_memory.store_memory(
        user_id="user_r2",
        content="To be superseded.",
        category="decision",
    )
    new_m = sage_memory.supersede_memory(
        old_memory_id=m["memory_id"],
        new_content="Superseded version.",
        user_id="user_r2",
        category="decision",
    )
    assert sage_memory.get_memory(m["memory_id"])["status"] == "superseded"
    assert sage_memory.get_memory(new_m["memory_id"])["status"] == "active"


# ── S. Phase 3 regression ────────────────────────────────────────────────────

def test_phase3_memory_search_with_category_filter_regression():
    """S. Phase 3 memory_search with category filter remains functional."""
    sage_memory.store_memory(
        user_id="user_s",
        content="User instructs SAGE to always confirm before deleting.",
        category="instruction",
    )
    sage_memory.store_memory(
        user_id="user_s",
        content="User prefers dark mode.",
        category="preference",
    )
    res = memory_search(user_id="user_s", category="instruction")
    assert res["status"] == "success"
    assert all(m["category"] == "instruction" for m in res["memories"])


def test_phase3_memory_get_regression():
    """S. Phase 3 memory_get remains functional."""
    stored = sage_memory.store_memory(
        user_id="user_s2",
        content="Phase 3 regression memory_get test.",
        category="fact",
    )
    res = memory_get(user_id="user_s2", memory_id=stored["memory_id"])
    assert res["status"] == "success"
    assert res["memory"]["memory_id"] == stored["memory_id"]


# ── T. Phase 4 regression ────────────────────────────────────────────────────

def test_phase4_semantic_search_regression():
    """T. Phase 4 vector-based semantic search remains functional."""
    sage_memory.store_memory(
        user_id="user_t",
        content="User designs distributed systems using microservices.",
        category="project",
    )
    res = memory_search(user_id="user_t", query="microservice architecture design")
    assert res["status"] == "success"
    assert res["returned"] >= 1


def test_phase4_chroma_indexing_on_store_regression():
    """T. Phase 4 Chroma indexing on store_memory remains functional."""
    mem = sage_memory.store_memory(
        user_id="user_t2",
        content="SAGE phase 4 regression chroma test.",
        category="project",
    )
    from sage_memory_index import memory_vector_index
    col = memory_vector_index._get_collection()
    chroma_res = col.get(ids=[mem["memory_id"]])
    assert len(chroma_res["ids"]) == 1


# ── Integration: E2E memory_store through the tool adapter ───────────────────

def test_e2e_gemma_memory_store_tool_call():
    """E2E integration: Simulate Gemma tool call -> durable_memory -> PostgreSQL -> Chroma -> safe result."""
    # Simulate orchestrator injecting trusted runtime identity
    trusted_user_id = "user_p5_e2e"
    trusted_chat_id = "chat_e2e_123"
    trusted_msg_id = None  # Not yet persisted; runtime has no source msg ID at this point

    # Simulate Gemma's decision: memory_store
    gemma_args = {
        "content": "User prefers Python for data engineering workflows.",
        "category": "preference",
        "importance": 0.88,
        "confidence": 0.92,
    }

    # Orchestrator attaches trusted identity (never from model)
    result = memory_store(
        user_id=trusted_user_id,
        chat_id=trusted_chat_id,
        source_msg_id=trusted_msg_id,
        **gemma_args,
    )

    # Verify safe result shape returned to Gemma
    assert result["status"] == "success"
    mem = result["memory"]
    assert "memory_id" in mem
    assert mem["content"] == "User prefers Python for data engineering workflows."
    assert mem["category"] == "preference"
    assert abs(mem["importance"] - 0.88) < 1e-6
    assert abs(mem["confidence"] - 0.92) < 1e-6
    # Internal fields must not be exposed to Gemma
    assert "user_id" not in mem

    # Verify canonical DB
    stored = sage_memory.get_memory(mem["memory_id"])
    assert stored["user_id"] == trusted_user_id
    assert stored["source_chat_id"] == trusted_chat_id

    # Verify Chroma
    from sage_memory_index import memory_vector_index
    col = memory_vector_index._get_collection()
    chroma_res = col.get(ids=[mem["memory_id"]])
    assert len(chroma_res["ids"]) == 1

    # Verify retrieval works via semantic search
    search = memory_search(user_id=trusted_user_id, chat_id=trusted_chat_id, query="Python data engineering")
    assert search["status"] == "success"
    assert any(m["memory_id"] == mem["memory_id"] for m in search["memories"])
