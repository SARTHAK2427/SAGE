"""
tests/test_phase6_memory.py
Phase 6 — Intelligent Memory Update / Conflict / Supersession Tests for SAGE.

Tests real execution and actual assertions for:
Test 1  — Valid supersession (old status == 'superseded', new status == 'active', supersedes_memory_id == old.id)
Test 2  — Retrieval excludes old memory (memory_search does not return superseded memory)
Test 3  — Chroma synchronization (old Chroma entry status='superseded', new Chroma entry status='active')
Test 4  — Foreign-user attack (user_B cannot supersede user_A's memory; rejected, user_A's memory remains active)
Test 5  — Nonexistent target (supersedes_memory_id = nonexistent ID safely rejected)
Test 6  — Already-superseded target (cannot supersede an already superseded memory)
Test 7  — Deleted target (cannot supersede a deleted memory)
Test 8  — Normal memory_store still works without supersedes_memory_id (backward compatibility with Phase 5)
Test 9  — User isolation (multiple users with supersessions see only their own memories)
Test 10 — Mapper sanitization (projections to Gemma expose safe fields, no leaked storage/database internals)
Test 11 — Duplicate decision boundary (backend does not heuristic-deduplicate; duplicate storage behaves deterministically)
Test 12 — Category behavior (cross-category supersession rejected deterministically)
"""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

import config
from sage_memory import sage_memory, set_sqlite_path
from sage_memory_index import MemoryVectorIndex
from tools.durable_memory import memory_search, memory_get, memory_store
from core.mappers.gemma_results import map_memory_store_result


@pytest.fixture(autouse=True)
def isolated_memory_environment():
    """Ensure every test runs with an isolated SQLite DB and Chroma vector index."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    db_file = str(temp_path / "test_p6.db")
    chroma_dir = temp_path / "test_p6_chroma"

    old_env = os.environ.get("SAGE_MEMORY_DB")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"
    set_sqlite_path(db_file)
    sage_memory._initialized_backends.clear()

    test_idx = MemoryVectorIndex(
        chroma_root=chroma_dir,
        collection_name=f"p6_test_{uuid.uuid4().hex[:8]}",
    )

    import sage_memory_index
    orig_idx = sage_memory_index.memory_vector_index
    sage_memory_index.memory_vector_index = test_idx

    import sage_memory as sm_mod
    orig_store = sm_mod._sync_index_store
    orig_status = sm_mod._sync_index_status

    sm_mod._sync_index_store = lambda m: test_idx.index_memory(m)
    sm_mod._sync_index_status = lambda m_id, stat: test_idx.update_memory_status(m_id, stat)

    yield {
        "vector_index": test_idx,
        "db_file": db_file,
    }

    sm_mod._sync_index_store = orig_store
    sm_mod._sync_index_status = orig_status
    sage_memory_index.memory_vector_index = orig_idx
    try:
        temp_dir.cleanup()
    except Exception:
        pass
    if old_env is not None:
        os.environ["SAGE_MEMORY_DB"] = old_env
    else:
        os.environ.pop("SAGE_MEMORY_DB", None)


# ── Test 1: Valid supersession ────────────────────────────────────────────────

def test_valid_supersession_flow():
    """Test 1 — Valid supersession: old status becomes 'superseded', new status 'active', reference linked."""
    user = "user_p6_test1"

    # Step 1: Create initial Python preference
    store_old = memory_store(
        user_id=user,
        content="I prefer Python for backend development.",
        category="preference",
        importance=0.8,
        confidence=1.0,
    )
    assert store_old["status"] == "success"
    old_id = store_old["memory"]["memory_id"]

    # Verify initial DB state
    old_mem_db = sage_memory.get_memory(old_id)
    assert old_mem_db is not None
    assert old_mem_db["status"] == "active"
    assert old_mem_db["supersedes_memory_id"] is None

    # Step 2: Supersede with C++ preference
    store_new = memory_store(
        user_id=user,
        content="I now use C++ for backend development.",
        category="preference",
        importance=0.9,
        confidence=1.0,
        supersedes_memory_id=old_id,
    )
    assert store_new["status"] == "success"
    new_id = store_new["memory"]["memory_id"]
    assert new_id != old_id
    assert store_new["memory"]["supersedes_memory_id"] == old_id

    # Verify canonical database statuses and links
    old_db = sage_memory.get_memory(old_id)
    new_db = sage_memory.get_memory(new_id)

    assert old_db["status"] == "superseded"
    assert new_db["status"] == "active"
    assert new_db["supersedes_memory_id"] == old_id


# ── Test 2: Retrieval excludes old memory ──────────────────────────────────────

def test_retrieval_excludes_superseded_memory():
    """Test 2 — Retrieval excludes old memory: memory_search returns new active memory and excludes superseded."""
    user = "user_p6_test2"

    store_old = memory_store(
        user_id=user,
        content="I prefer Python for backend development.",
        category="preference",
    )
    old_id = store_old["memory"]["memory_id"]

    store_new = memory_store(
        user_id=user,
        content="I now use C++ for backend development.",
        category="preference",
        supersedes_memory_id=old_id,
    )
    new_id = store_new["memory"]["memory_id"]

    # Search for backend programming language preference
    search_res = memory_search(user_id=user, query="backend development programming language")
    assert search_res["status"] == "success"
    retrieved_ids = [m["memory_id"] for m in search_res["memories"]]

    assert new_id in retrieved_ids, f"Expected active new memory {new_id} in search results."
    assert old_id not in retrieved_ids, f"Superseded memory {old_id} must NOT be returned in search results."


# ── Test 3: Chroma synchronization ────────────────────────────────────────────

def test_chroma_synchronization_on_supersession(isolated_memory_environment):
    """Test 3 — Chroma synchronization: old vector metadata status is 'superseded', new is 'active'."""
    user = "user_p6_test3"
    idx = isolated_memory_environment["vector_index"]
    col = idx._get_collection()

    store_old = memory_store(
        user_id=user,
        content="Frontend built with React.",
        category="project",
    )
    old_id = store_old["memory"]["memory_id"]

    # Verify old entry in Chroma initially
    chroma_initial = col.get(ids=[old_id], include=["metadatas"])
    assert len(chroma_initial["ids"]) == 1
    assert chroma_initial["metadatas"][0]["status"] == "active"

    # Supersede
    store_new = memory_store(
        user_id=user,
        content="Frontend migrated to Svelte.",
        category="project",
        supersedes_memory_id=old_id,
    )
    new_id = store_new["memory"]["memory_id"]

    # Verify old entry in Chroma updated to superseded
    chroma_old = col.get(ids=[old_id], include=["metadatas"])
    assert chroma_old["metadatas"][0]["status"] == "superseded"

    # Verify new entry in Chroma is active
    chroma_new = col.get(ids=[new_id], include=["metadatas"])
    assert chroma_new["metadatas"][0]["status"] == "active"


# ── Test 4: Foreign-user attack ───────────────────────────────────────────────

def test_foreign_user_supersession_rejected():
    """Test 4 — Foreign-user attack: user_B cannot supersede user_A's memory."""
    user_a = "user_p6_alice"
    user_b = "user_p6_attacker_bob"

    # Alice creates a private memory
    store_alice = memory_store(
        user_id=user_a,
        content="Alice's confidential deployment keys are in Vault.",
        category="fact",
    )
    alice_mem_id = store_alice["memory"]["memory_id"]

    # Bob attempts to supersede Alice's memory
    attack_attempt = memory_store(
        user_id=user_b,
        content="Bob replaces Alice's deployment keys.",
        category="fact",
        supersedes_memory_id=alice_mem_id,
    )

    # Must be deterministically rejected
    assert attack_attempt["status"] == "error"
    assert attack_attempt["error"]["code"] == "UNAUTHORIZED_MEMORY_ACCESS"

    # Alice's memory must remain completely active and untouched
    alice_mem = sage_memory.get_memory(alice_mem_id)
    assert alice_mem["status"] == "active"

    # No rogue memory should exist in Bob's space superseding Alice
    bob_mems = sage_memory.list_memories(user_id=user_b)
    assert not any(m.get("supersedes_memory_id") == alice_mem_id for m in bob_mems)


# ── Test 5: Nonexistent target ────────────────────────────────────────────────

def test_nonexistent_supersession_target_rejected():
    """Test 5 — Nonexistent target: supersedes_memory_id with nonexistent ID returns structured error."""
    user = "user_p6_test5"
    fake_id = str(uuid.uuid4())

    res = memory_store(
        user_id=user,
        content="Statement attempting to supersede a ghost memory.",
        category="fact",
        supersedes_memory_id=fake_id,
    )

    assert res["status"] == "error"
    assert res["error"]["code"] == "TARGET_MEMORY_NOT_FOUND"
    assert not res["error"]["retryable"]

    # No memory created
    user_mems = sage_memory.list_memories(user_id=user)
    assert len(user_mems) == 0


# ── Test 6: Already-superseded target ─────────────────────────────────────────

def test_already_superseded_target_rejected():
    """Test 6 — Already superseded target: branching supersession from an inactive memory is rejected."""
    user = "user_p6_test6"

    # Memory A
    m_a = memory_store(user_id=user, content="Version 1", category="project")
    id_a = m_a["memory"]["memory_id"]

    # Memory B supersedes A
    m_b = memory_store(user_id=user, content="Version 2", category="project", supersedes_memory_id=id_a)
    assert m_b["status"] == "success"

    # Memory C attempts to supersede A (which is already superseded)
    m_c = memory_store(user_id=user, content="Version 3 attempting to supersede Version 1", category="project", supersedes_memory_id=id_a)
    assert m_c["status"] == "error"
    assert m_c["error"]["code"] == "TARGET_MEMORY_INACTIVE"

    # Verify A remains superseded and B remains active
    assert sage_memory.get_memory(id_a)["status"] == "superseded"
    assert sage_memory.get_memory(m_b["memory"]["memory_id"])["status"] == "active"


# ── Test 7: Deleted target ────────────────────────────────────────────────────

def test_deleted_target_supersession_rejected():
    """Test 7 — Deleted target: attempting to supersede a logically deleted memory is rejected."""
    user = "user_p6_test7"

    # Store then logically delete
    mem = memory_store(user_id=user, content="Temporary fact", category="fact")
    mem_id = mem["memory"]["memory_id"]
    sage_memory.delete_memory(mem_id)
    assert sage_memory.get_memory(mem_id)["status"] == "deleted"

    # Attempt to supersede
    res = memory_store(
        user_id=user,
        content="Replacement for deleted fact",
        category="fact",
        supersedes_memory_id=mem_id,
    )
    assert res["status"] == "error"
    assert res["error"]["code"] == "TARGET_MEMORY_INACTIVE"


# ── Test 8: Normal memory_store still works ───────────────────────────────────

def test_normal_memory_store_without_supersedes_memory_id():
    """Test 8 — Normal memory_store still works: backward-compatible Phase 5 storage behavior preserved."""
    user = "user_p6_test8"

    res = memory_store(
        user_id=user,
        content="Standard independent fact.",
        category="fact",
        importance=0.7,
        confidence=0.9,
    )
    assert res["status"] == "success"
    assert "memory_id" in res["memory"]
    assert res["memory"]["content"] == "Standard independent fact."
    assert "supersedes_memory_id" not in res["memory"]

    stored_db = sage_memory.get_memory(res["memory"]["memory_id"])
    assert stored_db["status"] == "active"
    assert stored_db["supersedes_memory_id"] is None


# ── Test 9: User isolation ────────────────────────────────────────────────────

def test_user_isolation_with_supersessions():
    """Test 9 — User isolation: users perform supersessions independently without cross-leakage."""
    user_alice = "user_p6_alice_iso"
    user_bob = "user_p6_bob_iso"

    # Alice flow: Python -> Go
    a_old = memory_store(user_id=user_alice, content="Alice loves Python", category="preference")
    memory_store(user_id=user_alice, content="Alice loves Go", category="preference", supersedes_memory_id=a_old["memory"]["memory_id"])

    # Bob flow: Java -> Kotlin
    b_old = memory_store(user_id=user_bob, content="Bob loves Java", category="preference")
    memory_store(user_id=user_bob, content="Bob loves Kotlin", category="preference", supersedes_memory_id=b_old["memory"]["memory_id"])

    # Alice search
    a_search = memory_search(user_id=user_alice, query="programming language preference")
    a_contents = [m["content"] for m in a_search["memories"]]
    assert any("Go" in c for c in a_contents)
    assert not any("Kotlin" in c for c in a_contents)
    assert not any("Java" in c for c in a_contents)

    # Bob search
    b_search = memory_search(user_id=user_bob, query="programming language preference")
    b_contents = [m["content"] for m in b_search["memories"]]
    assert any("Kotlin" in c for c in b_contents)
    assert not any("Go" in c for c in b_contents)
    assert not any("Python" in c for c in b_contents)


# ── Test 10: Mapper sanitization ──────────────────────────────────────────────

def test_mapper_sanitization_for_supersession():
    """Test 10 — Mapper: supersession results project cleanly to Gemma without internal leakage."""
    socket_result = {
        "status": "success",
        "memory": {
            "memory_id": "mem_new_999",
            "content": "Updated rule",
            "category": "instruction",
            "importance": 0.8,
            "confidence": 1.0,
            "supersedes_memory_id": "mem_old_888",
            "user_id": "secret_user_id",
            "source_msg_id": "secret_msg_id",
            "local_path": "/var/data/secret.db",
        },
    }

    mapped = map_memory_store_result(socket_result)
    mem = mapped["memory"]
    assert mem["memory_id"] == "mem_new_999"
    assert mem["content"] == "Updated rule"
    assert mem["category"] == "instruction"
    assert mem["supersedes_memory_id"] == "mem_old_888"

    # Internal fields strictly stripped
    assert "local_path" not in mem
    assert "user_id" not in mem
    assert "source_msg_id" not in mem


# ── Test 11: Duplicate decision boundary ──────────────────────────────────────

def test_duplicate_decision_boundary_is_model_responsibility():
    """Test 11 — Duplicate decision boundary: backend does not perform keyword deduplication.

    The backend stores both memories deterministically; avoiding duplicates is Gemma's responsibility.
    """
    user = "user_p6_test11"
    content = "User prefers light theme in the daytime."

    r1 = memory_store(user_id=user, content=content, category="preference")
    r2 = memory_store(user_id=user, content=content, category="preference")

    assert r1["status"] == "success"
    assert r2["status"] == "success"
    assert r1["memory"]["memory_id"] != r2["memory"]["memory_id"]

    all_mems = sage_memory.list_memories(user_id=user)
    matches = [m for m in all_mems if m["content"] == content]
    assert len(matches) == 2, "Backend must not apply heuristic deduplication; both memories must persist."


# ── Test 12: Category behavior ────────────────────────────────────────────────

def test_category_mismatch_in_supersession_rejected():
    """Test 12 — Category behavior: cross-category supersession is rejected deterministically."""
    user = "user_p6_test12"

    # Create a preference memory
    pref_mem = memory_store(
        user_id=user,
        content="I prefer PostgreSQL over MySQL.",
        category="preference",
    )
    pref_id = pref_mem["memory"]["memory_id"]

    # Attempt to supersede the preference with a 'fact' category memory
    cross_cat_attempt = memory_store(
        user_id=user,
        content="The application database runs on PostgreSQL.",
        category="fact",
        supersedes_memory_id=pref_id,
    )

    assert cross_cat_attempt["status"] == "error"
    assert cross_cat_attempt["error"]["code"] == "CATEGORY_MISMATCH"

    # Preference memory remains active
    fetched = sage_memory.get_memory(pref_id)
    assert fetched["status"] == "active"
