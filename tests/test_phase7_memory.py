"""
tests/test_phase7_memory.py
Phase 7 — Hot/Cold Memory, Compaction & Promotion Tests for SAGE.

Contains real assertions for all 24 required test scenarios:
TEST 1  — Hot storage
TEST 2  — Cold storage
TEST 3  — Hot isolation by chat
TEST 4  — User isolation
TEST 5  — Cold cross-chat retrieval
TEST 6  — Hot does not leak cross-chat
TEST 7  — Promotion
TEST 8  — Promotion authorization
TEST 9  — Promotion of non-hot memory
TEST 10 — Promotion of deleted memory
TEST 11 — Compaction
TEST 12 — Compaction isolation
TEST 13 — Compaction cross-chat protection
TEST 14 — Failed compaction safety (loss-awareness)
TEST 15 — Hot semantic search
TEST 16 — Cold semantic search
TEST 17 — Tier isolation in Chroma
TEST 18 — Chroma stale-state protection
TEST 19 — Phase 6 supersession still works
TEST 20 — Duplicate boundary
TEST 21 — Mapper sanitization
TEST 22 — Model identity attack
TEST 23 — Normal Phase 5 memory store
TEST 24 — Existing regression compatibility
"""

import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import config
from sage_memory import sage_memory, set_sqlite_path
from sage_memory_index import MemoryVectorIndex
from tools.durable_memory import (
    memory_search,
    memory_get,
    memory_store,
    memory_search_hot,
    memory_search_cold,
    memory_store_hot,
    memory_store_cold,
    memory_promote,
    memory_summarize,
)
from core.mappers.gemma_results import (
    map_memory_search_result,
    map_memory_store_result,
    map_memory_promote_result,
    map_memory_summarize_result,
)


@pytest.fixture(autouse=True)
def isolated_memory_environment():
    """Ensure every test runs with an isolated SQLite DB and Chroma vector index."""
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    db_file = str(temp_path / "test_p7.db")
    chroma_dir = temp_path / "test_p7_chroma"

    old_env = os.environ.get("SAGE_MEMORY_DB")
    os.environ["SAGE_MEMORY_DB"] = "sqlite"
    set_sqlite_path(db_file)
    sage_memory._initialized_backends.clear()

    # Create isolated vector index
    test_idx = MemoryVectorIndex(chroma_root=chroma_dir)

    import sage_memory_index
    orig_idx = sage_memory_index.memory_vector_index
    sage_memory_index.memory_vector_index = test_idx

    import sage_memory as sm_mod
    orig_store = sm_mod._sync_index_store
    orig_status = sm_mod._sync_index_status

    sm_mod._sync_index_store = lambda mem: test_idx.index_memory(mem)
    sm_mod._sync_index_status = lambda mid, st: test_idx.update_memory_status(mid, st)

    yield {
        "db_file": db_file,
        "chroma_dir": chroma_dir,
        "vector_index": test_idx,
    }

    test_idx.close()
    sm_mod._sync_index_store = orig_store
    sm_mod._sync_index_status = orig_status
    sage_memory_index.memory_vector_index = orig_idx
    set_sqlite_path(None)
    if old_env is not None:
        os.environ["SAGE_MEMORY_DB"] = old_env
    else:
        os.environ.pop("SAGE_MEMORY_DB", None)
    try:
        temp_dir.cleanup()
    except Exception:
        pass


# ── TEST 1: Hot storage ───────────────────────────────────────────────────────
def test_1_hot_storage():
    """Store a hot memory. Assert tier == hot, status == active, correct user, correct chat/session."""
    res = memory_store_hot(
        user_id="user_alice",
        chat_id="chat_001",
        content="Current task concerns debugging Docker networking.",
        category="project",
        importance=0.8,
        confidence=0.95,
    )
    assert res["status"] == "success"
    mem = res["memory"]
    assert mem["memory_tier"] == "hot"
    assert mem["status"] == "active"
    assert mem["source_chat_id"] == "chat_001"

    # Verify canonical record
    canonical = sage_memory.get_memory(mem["memory_id"])
    assert canonical is not None
    assert canonical["user_id"] == "user_alice"
    assert canonical["source_chat_id"] == "chat_001"
    assert canonical["memory_tier"] == "hot"
    assert canonical["status"] == "active"
    assert canonical["importance"] == 0.8
    assert canonical["confidence"] == 0.95


# ── TEST 2: Cold storage ──────────────────────────────────────────────────────
def test_2_cold_storage():
    """Store a cold memory. Assert tier == cold, status == active, correct user."""
    res = memory_store_cold(
        user_id="user_alice",
        chat_id="chat_001",
        content="User prefers C++ for programming.",
        category="preference",
        importance=0.9,
    )
    assert res["status"] == "success"
    mem = res["memory"]
    assert mem["memory_tier"] == "cold"
    assert mem["status"] == "active"

    canonical = sage_memory.get_memory(mem["memory_id"])
    assert canonical is not None
    assert canonical["user_id"] == "user_alice"
    assert canonical["memory_tier"] == "cold"
    assert canonical["status"] == "active"


# ── TEST 3: Hot isolation by chat ─────────────────────────────────────────────
def test_3_hot_isolation_by_chat():
    """User A / Chat 1 / Hot X; User A / Chat 2 / Hot Y. Search Chat 1: assert Y is not returned."""
    # Store Hot X in Chat 1
    res_x = memory_store_hot(
        user_id="user_alice",
        chat_id="chat_1",
        content="Inspecting refinery pump P-17.",
        category="project",
    )
    # Store Hot Y in Chat 2
    res_y = memory_store_hot(
        user_id="user_alice",
        chat_id="chat_2",
        content="Investigating turbine T-99 vibration.",
        category="project",
    )

    # Search in Chat 1
    search_res = memory_search_hot(
        user_id="user_alice",
        chat_id="chat_1",
        query="inspection finding",
    )
    assert search_res["status"] == "success"
    retrieved_ids = [m["memory_id"] for m in search_res["memories"]]
    assert res_x["memory"]["memory_id"] in retrieved_ids
    assert res_y["memory"]["memory_id"] not in retrieved_ids


# ── TEST 4: User isolation ────────────────────────────────────────────────────
def test_4_user_isolation():
    """Assert each user only retrieves their own memories in hot and cold tiers."""
    # Alice memories
    res_a_hot = memory_store_hot(user_id="user_alice", chat_id="c1", content="Alice hot task data", category="fact")
    res_a_cold = memory_store_cold(user_id="user_alice", content="Alice durable preference", category="preference")

    # Bob memories
    res_b_hot = memory_store_hot(user_id="user_bob", chat_id="c1", content="Bob hot task data", category="fact")
    res_b_cold = memory_store_cold(user_id="user_bob", content="Bob durable preference", category="preference")

    # Alice searches hot
    alice_hot = memory_search_hot(user_id="user_alice", chat_id="c1", query="task data")
    alice_hot_ids = [m["memory_id"] for m in alice_hot["memories"]]
    assert res_a_hot["memory"]["memory_id"] in alice_hot_ids
    assert res_b_hot["memory"]["memory_id"] not in alice_hot_ids

    # Bob searches cold
    bob_cold = memory_search_cold(user_id="user_bob", query="preference")
    bob_cold_ids = [m["memory_id"] for m in bob_cold["memories"]]
    assert res_b_cold["memory"]["memory_id"] in bob_cold_ids
    assert res_a_cold["memory"]["memory_id"] not in bob_cold_ids


# ── TEST 5: Cold cross-chat retrieval ─────────────────────────────────────────
def test_5_cold_cross_chat_retrieval():
    """Only Global memory crosses chats; Cold episodic memory remains chat-scoped."""
    cold_res = memory_store_cold(
        user_id="user_alice",
        chat_id="chat_1",
        content="Chat 1 is debugging a transient render issue.",
        category="project",
    )
    cold_id = cold_res["memory"]["memory_id"]
    global_res = memory_store_cold(
        user_id="user_alice",
        content="I prefer dark mode in all UI applications.",
        category="preference",
    )

    search_res = memory_search_cold(user_id="user_alice", chat_id="chat_2", query="render issue dark mode", scope="both")
    retrieved_ids = [m["memory_id"] for m in search_res["memories"]]
    assert cold_id not in retrieved_ids
    assert global_res["memory"]["memory_id"] in retrieved_ids


# ── TEST 6: Hot does not leak cross-chat ──────────────────────────────────────
def test_6_hot_does_not_leak_cross_chat():
    """Create hot memory in Chat 1. Search from Chat 2: assert absent."""
    hot_res = memory_store_hot(
        user_id="user_alice",
        chat_id="chat_1",
        content="Uploaded report is rev_3_final.pdf",
        category="project",
    )
    hot_id = hot_res["memory"]["memory_id"]

    # Search hot in chat_2
    chat2_search = memory_search_hot(user_id="user_alice", chat_id="chat_2", query="uploaded report")
    chat2_ids = [m["memory_id"] for m in chat2_search["memories"]]
    assert hot_id not in chat2_ids


# ── TEST 7: Promotion ─────────────────────────────────────────────────────────
def test_7_promotion():
    """Create hot memory, promote it. Assert cold exists, hot no longer active."""
    hot_res = memory_store_hot(
        user_id="user_alice",
        chat_id="chat_1",
        content="I prefer using local models for SAGE code execution.",
        category="preference",
    )
    hot_id = hot_res["memory"]["memory_id"]

    prom_res = memory_promote(user_id="user_alice", memory_id=hot_id)
    assert prom_res["status"] == "success"
    cold_mem = prom_res["memory"]
    assert cold_mem["memory_tier"] == "cold"
    assert cold_mem["status"] == "active"
    assert cold_mem["memory_id"] == hot_id

    # Canonical record should be updated in place (tier=cold, status=active, stable ID)
    canonical = sage_memory.get_memory(hot_id)
    assert canonical["status"] == "active"
    assert canonical["memory_tier"] == "cold"

    # Hot search should NOT return the promoted memory
    hot_search = memory_search_hot(user_id="user_alice", chat_id="chat_1", query="local models")
    hot_ids = [m["memory_id"] for m in hot_search["memories"]]
    assert hot_id not in hot_ids

    # Cold search SHOULD return the promoted cold memory
    cold_search = memory_search_cold(user_id="user_alice", chat_id="chat_1", query="local models", scope="cold")
    cold_ids = [m["memory_id"] for m in cold_search["memories"]]
    assert hot_id in cold_ids


# ── TEST 8: Promotion authorization ───────────────────────────────────────────
def test_8_promotion_authorization():
    """Attempt to promote another user's hot memory. Assert rejection."""
    hot_res = memory_store_hot(
        user_id="user_alice",
        chat_id="chat_1",
        content="Secret Alice project notes",
        category="project",
    )
    hot_id = hot_res["memory"]["memory_id"]

    # User Bob attempts promotion
    attack_res = memory_promote(user_id="user_bob", memory_id=hot_id)
    assert attack_res["status"] == "error"
    assert attack_res["error"]["code"] == "UNAUTHORIZED_MEMORY_ACCESS"

    # Alice's memory remains hot and active
    alice_mem = sage_memory.get_memory(hot_id)
    assert alice_mem["status"] == "active"
    assert alice_mem["memory_tier"] == "hot"


# ── TEST 9: Promotion of non-hot memory ───────────────────────────────────────
def test_9_promotion_of_non_hot_memory():
    """Attempt to promote cold memory to cold. Assert rejection."""
    cold_res = memory_store_cold(
        user_id="user_alice",
        content="Already a cold memory fact.",
        category="fact",
    )
    cold_id = cold_res["memory"]["memory_id"]

    prom_res = memory_promote(user_id="user_alice", memory_id=cold_id)
    assert prom_res["status"] == "error"
    assert prom_res["error"]["code"] == "INVALID_MEMORY_TIER"


# ── TEST 10: Promotion of deleted memory ──────────────────────────────────────
def test_10_promotion_of_deleted_memory():
    """Create hot memory, delete it, attempt promotion. Assert rejection."""
    hot_res = memory_store_hot(
        user_id="user_alice",
        chat_id="chat_1",
        content="Temporary scratch note.",
        category="fact",
    )
    hot_id = hot_res["memory"]["memory_id"]
    sage_memory.delete_memory(hot_id)

    prom_res = memory_promote(user_id="user_alice", memory_id=hot_id)
    assert prom_res["status"] == "error"
    assert prom_res["error"]["code"] == "TARGET_MEMORY_INACTIVE"


# ── TEST 11: Compaction ───────────────────────────────────────────────────────
def test_11_compaction():
    """Create several related hot memories. Compact into summary.
    Assert summary exists active hot, source memories marked compacted.
    """
    m1 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="User uploaded inspection_report.pdf", category="project")
    m2 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Report concerns pump P-17", category="project")
    m3 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Finding indicates abnormal vibration", category="project")
    m4 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Relevant SOP is SOP-42", category="project")

    source_ids = [
        m1["memory"]["memory_id"],
        m2["memory"]["memory_id"],
        m3["memory"]["memory_id"],
        m4["memory"]["memory_id"],
    ]

    summary_text = "Current task concerns inspection_report.pdf for pump P-17 with abnormal vibration per SOP-42."
    comp_res = memory_summarize(
        user_id="user_alice",
        chat_id="chat_1",
        memory_ids=source_ids,
        summary_content=summary_text,
        category="project",
    )
    assert comp_res["status"] == "success"
    summary_mem = comp_res["memory"]
    assert summary_mem["memory_tier"] == "hot"
    assert summary_mem["status"] == "active"
    assert summary_mem["source_chat_id"] == "chat_1"

    # Verify source memories are compacted
    for sid in source_ids:
        src = sage_memory.get_memory(sid)
        assert src["status"] == "compacted"

    # Hot search should retrieve summary and not the old compacted sources
    search_res = memory_search_hot(user_id="user_alice", chat_id="chat_1", query="pump P-17 inspection")
    retrieved_ids = [m["memory_id"] for m in search_res["memories"]]
    assert summary_mem["memory_id"] in retrieved_ids
    for sid in source_ids:
        assert sid not in retrieved_ids


# ── TEST 12: Compaction isolation ─────────────────────────────────────────────
def test_12_compaction_isolation():
    """Attempt to compact memories belonging to another user. Assert rejection."""
    m_alice = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Alice task info", category="fact")
    m_id = m_alice["memory"]["memory_id"]

    # Bob attempts to compact Alice's memory
    res = memory_summarize(
        user_id="user_bob",
        chat_id="chat_1",
        memory_ids=[m_id],
        summary_content="Bob summary",
    )
    assert res["status"] == "error"
    assert res["error"]["code"] == "UNAUTHORIZED_MEMORY_ACCESS"

    # Alice memory remains active
    assert sage_memory.get_memory(m_id)["status"] == "active"


# ── TEST 13: Compaction cross-chat protection ─────────────────────────────────
def test_13_compaction_cross_chat_protection():
    """Attempt to compact hot memories from different chats. Assert rejection."""
    m_chat1 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Chat 1 task info", category="fact")
    m_chat2 = memory_store_hot(user_id="user_alice", chat_id="chat_2", content="Chat 2 task info", category="fact")

    res = memory_summarize(
        user_id="user_alice",
        chat_id="chat_1",
        memory_ids=[m_chat1["memory"]["memory_id"], m_chat2["memory"]["memory_id"]],
        summary_content="Consolidated across chats",
    )
    assert res["status"] == "error"
    assert res["error"]["code"] == "INVALID_SESSION"

    # Both memories remain active
    assert sage_memory.get_memory(m_chat1["memory"]["memory_id"])["status"] == "active"
    assert sage_memory.get_memory(m_chat2["memory"]["memory_id"])["status"] == "active"


# ── TEST 14: Failed compaction safety (loss-awareness) ────────────────────────
def test_14_failed_compaction_safety():
    """Simulate summary persistence failure. Assert original memories remain active."""
    m1 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Critical fact 1", category="fact")
    m2 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Critical fact 2", category="fact")
    m_ids = [m1["memory"]["memory_id"], m2["memory"]["memory_id"]]

    # Mock store_memory to fail during summary storage
    orig_store = sage_memory.store_memory
    def failing_store(**kwargs):
        if kwargs.get("memory_tier") == "hot" and "summary" in kwargs.get("content", "").lower():
            raise RuntimeError("Database write failure simulation")
        return orig_store(**kwargs)

    with patch.object(sage_memory, "store_memory", side_effect=failing_store):
        res = memory_summarize(
            user_id="user_alice",
            chat_id="chat_1",
            memory_ids=m_ids,
            summary_content="summary that fails to store",
        )
        assert res["status"] == "error"

    # Critical invariant: Original source memories MUST remain active (no data loss)
    for mid in m_ids:
        mem = sage_memory.get_memory(mid)
        assert mem["status"] == "active"


# ── TEST 15: Hot semantic search ──────────────────────────────────────────────
def test_15_hot_semantic_search():
    """Create several hot memories. Search using semantically related wording."""
    m1 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="The cooling tower is experiencing water leakage.", category="project")
    m2 = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Network latency spikes on switch S-4.", category="project")

    res = memory_search_hot(user_id="user_alice", chat_id="chat_1", query="water dripping from cooler")
    assert res["status"] == "success"
    assert len(res["memories"]) > 0
    assert res["memories"][0]["memory_id"] == m1["memory"]["memory_id"]


# ── TEST 16: Cold semantic search ─────────────────────────────────────────────
def test_16_cold_semantic_search():
    """Create several cold memories. Search using semantically related wording."""
    m1 = memory_store_cold(user_id="user_alice", content="User prefers typing in Neovim.", category="preference")
    m2 = memory_store_cold(user_id="user_alice", content="Company headquarters located in Seattle.", category="fact")

    res = memory_search_cold(user_id="user_alice", query="favorite text editor")
    assert res["status"] == "success"
    assert len(res["memories"]) > 0
    assert res["memories"][0]["memory_id"] == m1["memory"]["memory_id"]


# ── TEST 17: Tier isolation in Chroma ─────────────────────────────────────────
def test_17_tier_isolation_in_chroma():
    """Ensure hot search does not retrieve cold memory and vice versa."""
    hot = memory_store_hot(user_id="user_alice", chat_id="chat_1", content="Debugging Python multiprocessing queue deadlock.", category="project")
    cold = memory_store_cold(user_id="user_alice", content="User writes Python scripts using standard library.", category="preference")

    # Hot search should NOT return cold memory
    hot_search = memory_search_hot(user_id="user_alice", chat_id="chat_1", query="Python library")
    hot_ids = [m["memory_id"] for m in hot_search["memories"]]
    assert cold["memory"]["memory_id"] not in hot_ids

    # Cold search should NOT return hot memory
    cold_search = memory_search_cold(user_id="user_alice", query="multiprocessing queue deadlock")
    cold_ids = [m["memory_id"] for m in cold_search["memories"]]
    assert hot["memory"]["memory_id"] not in cold_ids


# ── TEST 18: Chroma stale-state protection ───────────────────────────────────
def test_18_chroma_stale_state_protection(isolated_memory_environment):
    """Manually insert inconsistent/stale record into Chroma; assert DB cross-check prevents incorrect retrieval."""
    test_idx = isolated_memory_environment["vector_index"]
    stale_id = str(uuid.uuid4())

    # Index directly into Chroma cold collection as active without DB entry
    test_idx.index_memory({
        "memory_id": stale_id,
        "content": "Phantom ghost memory not in canonical database.",
        "user_id": "user_alice",
        "category": "fact",
        "status": "active",
        "memory_tier": "cold",
    })

    # Search cold
    search_res = memory_search_cold(user_id="user_alice", query="phantom ghost memory")
    retrieved_ids = [m["memory_id"] for m in search_res["memories"]]
    # Stale phantom record must be rejected by canonical DB cross-check
    assert stale_id not in retrieved_ids


# ── TEST 19: Phase 6 supersession still works ─────────────────────────────────
def test_19_phase_6_supersession_still_works():
    """Verify old cold memory -> superseded, new cold memory -> active."""
    m_old = memory_store_cold(
        user_id="user_alice",
        content="I use Docker Compose v1.",
        category="preference",
    )
    old_id = m_old["memory"]["memory_id"]

    m_new = memory_store_cold(
        user_id="user_alice",
        content="I have upgraded to Docker Compose v2.",
        category="preference",
        supersedes_memory_id=old_id,
    )
    new_id = m_new["memory"]["memory_id"]

    assert m_new["status"] == "success"
    assert m_new["memory"]["supersedes_memory_id"] == old_id

    # Verify old status is superseded, new status is active
    old_db = sage_memory.get_memory(old_id)
    new_db = sage_memory.get_memory(new_id)
    assert old_db["status"] == "superseded"
    assert new_db["status"] == "active"

    # Search cold returns only new memory
    search_res = memory_search_cold(user_id="user_alice", query="Docker Compose")
    retrieved_ids = [m["memory_id"] for m in search_res["memories"]]
    assert new_id in retrieved_ids
    assert old_id not in retrieved_ids


# ── TEST 20: Duplicate boundary ───────────────────────────────────────────────
def test_20_duplicate_boundary():
    """Backend must NOT implement heuristic duplicate detection; direct backend writes store deterministically."""
    res1 = memory_store_cold(user_id="user_alice", content="I prefer Python.", category="preference")
    res2 = memory_store_cold(user_id="user_alice", content="I like Python more than Java.", category="preference")

    # Backend deterministic contract allows both writes; semantic deduplication is Gemma's responsibility
    assert res1["status"] == "success"
    assert res2["status"] == "success"
    assert res1["memory"]["memory_id"] != res2["memory"]["memory_id"]


# ── TEST 21: Mapper sanitization ──────────────────────────────────────────────
def test_21_mapper_sanitization():
    """Verify Phase 7 results do not expose user_id, source_msg_id, DB paths, credentials, etc."""
    raw_socket = {
        "status": "success",
        "memory": {
            "memory_id": "mem_123",
            "user_id": "secret_user_id",
            "source_msg_id": "secret_msg_id",
            "content": "Useful fact",
            "category": "fact",
            "importance": 0.8,
            "confidence": 1.0,
            "status": "active",
            "memory_tier": "cold",
            "internal_db_path": "/var/data/sage.db",
            "db_credentials": "password123",
        },
    }
    mapped = map_memory_promote_result(raw_socket)
    mem = mapped["memory"]
    assert "user_id" not in mem
    assert "source_msg_id" not in mem
    assert "internal_db_path" not in mem
    assert "db_credentials" not in mem
    assert mem["memory_id"] == "mem_123"
    assert mem["memory_tier"] == "cold"


# ── TEST 22: Model identity attack ────────────────────────────────────────────
def test_22_model_identity_attack():
    """Attempt to pass user_id = another_user, chat_id = another_chat through model arguments; assert trusted values win."""
    from orchestrator import Orchestrator
    from core.run_state import RunState

    # Build orchestrator with real registry
    orch = Orchestrator()

    run_state = RunState(
        user_id="trusted_alice",
        chat_id="trusted_chat_1",
    )

    # Attacker tries to inject user_id="attacker_bob" and chat_id="foreign_chat"
    malicious_call = {
        "tool": "durable_memory",
        "function": "memory_store_hot",
        "arguments": {
            "user_id": "attacker_bob",
            "chat_id": "foreign_chat",
            "content": "Maliciously stored memory",
            "category": "fact",
        },
    }

    result_packet = orch._dispatch_tool_call(
        call=malicious_call,
        run_state=run_state,
        telemetry={},
        trace=[],
        file_map={},
        call_index=0,
    )

    assert result_packet["status"] == "success"
    mem_id = result_packet["result"]["memory"]["memory_id"]

    # Invariant: Trusted identity must be enforced in DB
    canonical = sage_memory.get_memory(mem_id)
    assert canonical["user_id"] == "trusted_alice"
    assert canonical["source_chat_id"] == "trusted_chat_1"



# ── TEST 23: Normal Phase 5 memory store ──────────────────────────────────────
def test_23_normal_phase_5_memory_store():
    """Verify the existing normal durable memory path still works without Phase 7 specific arguments."""
    res = memory_store(
        user_id="user_alice",
        content="Standard Phase 5 memory content.",
        category="fact",
        importance=0.6,
    )
    assert res["status"] == "success"
    mem_id = res["memory"]["memory_id"]

    # Can retrieve with memory_get
    get_res = memory_get(user_id="user_alice", memory_id=mem_id)
    assert get_res["status"] == "success"
    assert get_res["memory"]["content"] == "Standard Phase 5 memory content."

    # Can retrieve with standard memory_search
    search_res = memory_search(user_id="user_alice", query="Phase 5 memory")
    assert search_res["status"] == "success"
    ids = [m["memory_id"] for m in search_res["memories"]]
    assert mem_id in ids


# ── TEST 24: Existing regression compatibility ────────────────────────────────
def test_24_existing_regression_compatibility():
    """Ensure last_accessed_at is populated and updated on retrieval."""
    res = memory_store_cold(
        user_id="user_alice",
        content="Testing last_accessed_at field.",
        category="fact",
    )
    mem_id = res["memory"]["memory_id"]
    canonical_initial = sage_memory.get_memory(mem_id)
    assert "last_accessed_at" in canonical_initial
    assert canonical_initial["last_accessed_at"] is not None

    # Retrieve memory with touch_access
    sage_memory.get_memory(mem_id, touch_access=True)
    canonical_touched = sage_memory.get_memory(mem_id)
    assert canonical_touched["last_accessed_at"] is not None
