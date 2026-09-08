"""
tests/test_memory_promotion.py
Phase 11 tests: Hot-to-cold memory promotion flow, atomic transition, and cross-session verification.

Validates:
1. Hot-to-cold promotion moves canonical JSON from data/memory/hot to data/memory/cold.
2. Atomic index transition: removed from Chroma hot collection, added to cold collection.
3. Metadata transition: memory_type becomes "cold", updated_at is refreshed, ID preserved.
4. Cross-session searchability: retrievable via search_cold_memory from any session.
5. Inaccessibility via search_hot_memory in original session after promotion.
6. Validation and error handling: non-existent ID, invalid ID, already-cold memory.
7. Tool adapter returns structured success and error envelopes.
8. Gemma projection via map_memory_promote_result (zero path leakage).
9. End-to-end dispatch through Orchestrator._dispatch_tool_call.
10. No auto-promotion policy: Python backend never autonomously promotes memories.
"""

from __future__ import annotations

import pytest

from core.dispatcher import ToolRegistry
from core.mappers.memory_results import map_memory_promote_result
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_document_db.utils import iso_now
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager, MemoryManagerError
from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore
from tools.memory import (
    register_memory_tools,
    tool_memory_promote,
    tool_memory_search_cold,
    tool_memory_search_hot,
    tool_memory_store_hot,
)


@pytest.fixture()
def memory_manager(tmp_memory_root, tmp_chroma_root):
    store = MemoryStore(tmp_memory_root)
    index = MemoryChromaStore(tmp_chroma_root, EmbeddingService())
    return MemoryManager(store=store, index=index)


@pytest.fixture()
def tool_registry(memory_manager):
    registry = ToolRegistry()
    register_memory_tools(registry, memory_manager)
    return registry


def _create_hot_record(session_id: str, content: str, category: str = "decision") -> MemoryRecord:
    now = iso_now()
    return MemoryRecord(
        memory_id=make_memory_id(),
        content=content,
        memory_type="hot",
        category=category,
        source="agent_call",
        session_id=session_id,
        created_at=now,
        updated_at=now,
        importance=0.9,
        confidence=0.95,
        is_summary=False,
        parent_memory_ids=[],
    )


# ─── 1. Atomic Hot-to-Cold Promotion Tests ────────────────────────────────────

class TestHotToColdPromotion:
    def test_promote_hot_memory_atomic_transition(self, memory_manager):
        """Promoting moves record from hot to cold in both canonical store and Chroma."""
        hot_rec = _create_hot_record(
            session_id="sess_promote_1",
            content="Architecture decision: use ChromaDB with cosine metric for vector search",
            category="decision",
        )
        memory_manager.add_hot_memory(hot_rec)

        # Initial checks
        assert memory_manager.index.count("hot") == 1
        assert memory_manager.index.count("cold") == 0
        assert memory_manager.store.get(hot_rec.memory_id, "hot") is not None
        assert memory_manager.store.get(hot_rec.memory_id, "cold") is None

        # Execute promotion
        promoted = memory_manager.promote_to_cold(hot_rec.memory_id)

        # 1. Returned record checks
        assert promoted.memory_id == hot_rec.memory_id
        assert promoted.memory_type == "cold"
        assert promoted.content == hot_rec.content
        assert promoted.category == hot_rec.category
        assert promoted.importance == hot_rec.importance
        assert promoted.confidence == hot_rec.confidence
        assert promoted.updated_at >= hot_rec.updated_at

        # 2. Canonical store checks
        assert memory_manager.store.get(hot_rec.memory_id, "hot") is None
        stored_cold = memory_manager.store.get(hot_rec.memory_id, "cold")
        assert stored_cold is not None
        assert stored_cold.memory_type == "cold"

        # 3. Chroma collection checks
        assert memory_manager.index.count("hot") == 0
        assert memory_manager.index.count("cold") == 1

        # 4. Retrieval checks: inaccessible in original session hot search
        hot_hits = memory_manager.search_hot_memory(
            query="ChromaDB cosine metric",
            session_id="sess_promote_1",
        )
        assert len(hot_hits) == 0

        # 5. Cross-session cold search retrieves the promoted record
        cold_hits = memory_manager.search_cold_memory(
            query="ChromaDB cosine metric",
        )
        assert len(cold_hits) == 1
        assert cold_hits[0].memory_id == hot_rec.memory_id
        assert cold_hits[0].memory_type == "cold"

    def test_promote_nonexistent_memory_fails(self, memory_manager):
        """Attempting to promote a non-existent memory raises MemoryManagerError."""
        with pytest.raises(MemoryManagerError, match="Hot memory not found for promotion"):
            memory_manager.promote_to_cold("mem_nonexistent_id")

    def test_promote_already_cold_memory_fails(self, memory_manager):
        """Attempting to promote an already-cold memory fails (not in hot store)."""
        hot_rec = _create_hot_record("sess_p2", "Durable fact")
        memory_manager.add_hot_memory(hot_rec)

        # First promotion succeeds
        memory_manager.promote_to_cold(hot_rec.memory_id)

        # Second promotion must fail
        with pytest.raises(MemoryManagerError, match="Hot memory not found for promotion"):
            memory_manager.promote_to_cold(hot_rec.memory_id)

    def test_promote_validation_memory_id(self, memory_manager):
        """Invalid memory_id formats fail deterministically."""
        with pytest.raises(MemoryManagerError, match="memory_id must be a non-empty string"):
            memory_manager.promote_to_cold("")

        with pytest.raises(MemoryManagerError, match="memory_id must be a non-empty string"):
            memory_manager.promote_to_cold("   ")

        with pytest.raises(MemoryManagerError, match="must start with 'mem_'"):
            memory_manager.promote_to_cold("invalid_prefix_123")


# ─── 2. Tool Adapter Promotion Tests ──────────────────────────────────────────

class TestToolAdapterPromotion:
    def test_tool_adapter_validation(self, memory_manager):
        """tool_memory_promote validates input and returns structured error envelopes."""
        # Missing memory_id
        res1 = tool_memory_promote(manager=memory_manager, memory_id="")
        assert res1["status"] == "error"
        assert "memory_id is required" in res1["error"]
        assert res1["promoted"] is False

        # Invalid prefix
        res2 = tool_memory_promote(manager=memory_manager, memory_id="not_a_mem_id")
        assert res2["status"] == "error"
        assert "must start with 'mem_'" in res2["error"]
        assert res2["promoted"] is False

        # Non-existent ID
        res3 = tool_memory_promote(manager=memory_manager, memory_id="mem_missing999")
        assert res3["status"] == "error"
        assert "not found for promotion" in res3["error"]
        assert res3["promoted"] is False

    def test_tool_adapter_success(self, memory_manager):
        """tool_memory_promote successfully promotes and returns structured success."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="User prefers asynchronous programming style",
            session_id="sess_adapter_promote",
            category="preference",
        )
        mid = stored["memory_id"]

        # Promote via tool adapter
        res = tool_memory_promote(manager=memory_manager, memory_id=mid)
        assert res["status"] == "success"
        assert res["promoted"] is True
        assert res["memory_id"] == mid
        assert res["memory"]["memory_type"] == "cold"

        # Verify via search adapters
        hot_res = tool_memory_search_hot(
            manager=memory_manager,
            query="asynchronous programming",
            session_id="sess_adapter_promote",
        )
        assert hot_res["count"] == 0

        cold_res = tool_memory_search_cold(
            manager=memory_manager,
            query="asynchronous programming",
        )
        assert cold_res["count"] == 1
        assert cold_res["results"][0]["memory_id"] == mid


# ─── 3. Gemma Projection & Orchestrator Dispatch Tests ────────────────────────

class TestPromoteGemmaIntegration:
    def test_map_memory_promote_result_projection(self):
        """map_memory_promote_result sanitizes internal fields and returns clean envelope."""
        adapter_output = {
            "status": "success",
            "memory_id": "mem_prom_123",
            "promoted": True,
            "memory": {
                "memory_id": "mem_prom_123",
                "content": "Promoted fact",
                "memory_type": "cold",
                "category": "decision",
                "local_path": "/data/memory/cold/mem_prom_123.json",  # stripped
                "chroma_collection": "sage_memory_cold",  # stripped
            },
        }

        projected = map_memory_promote_result(adapter_output)
        assert projected["memory_id"] == "mem_prom_123"
        assert projected["promoted"] is True
        assert projected["memory_type"] == "cold"
        assert "local_path" not in projected
        assert "chroma_collection" not in projected

    def test_orchestrator_dispatch_promote(self, memory_manager, tool_registry):
        """Orchestrator dispatches memory_promote tool call end-to-end."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Critical decision: adopt semantic versioning",
            session_id="sess_orch_prom",
            category="decision",
        )
        mid = stored["memory_id"]

        orch = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_orch_prom", user_text="Promote the versioning decision.")

        tool_res = orch._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_promote",
                "arguments": {
                    "memory_id": mid,
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )

        assert tool_res["status"] == "success"
        assert tool_res["result"]["promoted"] is True
        assert tool_res["result"]["memory_type"] == "cold"
        assert tool_res["result"]["memory_id"] == mid


# ─── 4. No Autonomous Promotion Policy Tests ──────────────────────────────────

class TestNoAutoPromotionPolicy:
    def test_hot_memories_are_never_auto_promoted(self, memory_manager):
        """High importance, frequent access, or aged hot memories are never auto-promoted."""
        session_id = "sess_no_auto_prom"

        # Create high importance hot memory
        rec = _create_hot_record(session_id, "Very important rule", category="decision")
        rec.importance = 1.0
        rec.confidence = 1.0
        rec.access_count = 100
        memory_manager.add_hot_memory(rec)

        # Verify it stays strictly in hot
        assert memory_manager.index.count("hot") == 1
        assert memory_manager.index.count("cold") == 0
        assert memory_manager.store.get(rec.memory_id, "hot") is not None
        assert memory_manager.store.get(rec.memory_id, "cold") is None

        # Even multiple searches / retrievals do NOT promote it
        for _ in range(5):
            hits = memory_manager.search_hot_memory("important rule", session_id=session_id)
            assert len(hits) == 1

        # Still 0 in cold
        assert memory_manager.index.count("cold") == 0
        assert memory_manager.store.get(rec.memory_id, "cold") is None

        # Only explicit promotion moves it
        memory_manager.promote_to_cold(rec.memory_id)
        assert memory_manager.index.count("cold") == 1
        assert memory_manager.index.count("hot") == 0


# ─── 5. Failure-Injection & Atomicity Tests ───────────────────────────────────

class TestPromotionFailureAtomicity:
    def test_cold_canonical_persistence_failure(self, memory_manager, monkeypatch):
        """A. Failure writing to cold store preserves hot record and leaves no cold orphan."""
        rec = _create_hot_record("sess_fail_a", "Hot item vulnerable to cold store disk failure")
        memory_manager.add_hot_memory(rec)

        assert memory_manager.store.get(rec.memory_id, "hot") is not None
        assert memory_manager.store.get(rec.memory_id, "cold") is None
        assert memory_manager.index.count("hot") == 1

        # Simulate cold store write failure
        original_write = memory_manager.store._write_record

        def failing_write(record: MemoryRecord) -> None:
            if record.memory_type == "cold":
                raise OSError("Simulated disk I/O error on cold canonical write")
            original_write(record)

        monkeypatch.setattr(memory_manager.store, "_write_record", failing_write)

        with pytest.raises(MemoryManagerError, match="Failed to promote canonical memory record"):
            memory_manager.promote_to_cold(rec.memory_id)

        # 1. Original hot canonical record remains available
        stored_hot = memory_manager.store.get(rec.memory_id, "hot")
        assert stored_hot is not None
        assert stored_hot.content == rec.content
        assert stored_hot.memory_type == "hot"

        # 2. Cold canonical record is not incorrectly left behind
        assert memory_manager.store.get(rec.memory_id, "cold") is None

        # 3. No memory loss: hot index remains intact
        assert memory_manager.index.count("hot") == 1
        assert memory_manager.index.count("cold") == 0

    def test_hot_chroma_deletion_failure(self, memory_manager, monkeypatch):
        """B. Failure removing from hot Chroma fails deterministically without canonical loss."""
        rec = _create_hot_record("sess_fail_b", "Hot item during Chroma hot deletion failure")
        memory_manager.add_hot_memory(rec)

        # Simulate Chroma hot deletion failure
        def failing_delete_memory(memory_id: str, memory_type: str) -> bool:
            if memory_type == "hot":
                raise RuntimeError("Simulated Chroma hot deletion network timeout")
            return True

        monkeypatch.setattr(memory_manager.index, "delete_memory", failing_delete_memory)

        with pytest.raises(MemoryManagerError, match="Chroma indexing failure while removing"):
            memory_manager.promote_to_cold(rec.memory_id)

        # 1. Canonical memory is not permanently lost (authoritative JSON on disk)
        stored_cold = memory_manager.store.get(rec.memory_id, "cold")
        assert stored_cold is not None
        assert stored_cold.content == rec.content

        # 2. Hot canonical record was removed once cold was written
        assert memory_manager.store.get(rec.memory_id, "hot") is None

    def test_cold_chroma_indexing_failure(self, memory_manager, monkeypatch):
        """C. Failure adding to cold Chroma reports error while canonical JSON remains recoverable."""
        rec = _create_hot_record("sess_fail_c", "Cold item during Chroma cold indexing failure")
        memory_manager.add_hot_memory(rec)

        # Simulate Chroma cold indexing failure
        def failing_add_memory(record: MemoryRecord) -> str:
            if record.memory_type == "cold":
                raise RuntimeError("Simulated Chroma cold indexing write failure")
            return record.memory_id

        monkeypatch.setattr(memory_manager.index, "add_memory", failing_add_memory)

        with pytest.raises(MemoryManagerError, match="Chroma indexing failure while adding"):
            memory_manager.promote_to_cold(rec.memory_id)

        # 1. Canonical memory remains recoverable on disk
        stored_cold = memory_manager.store.get(rec.memory_id, "cold")
        assert stored_cold is not None
        assert stored_cold.content == rec.content
        assert stored_cold.memory_type == "cold"

        # 2. Stale Chroma state is recoverable through index rebuild
        # Restore normal add_memory and rebuild index from canonical store
        monkeypatch.undo()
        memory_manager.index.add_memory(stored_cold)
        assert memory_manager.index.count("cold") == 1
        cold_hits = memory_manager.search_cold_memory("Chroma cold indexing failure")
        assert len(cold_hits) == 1
        assert cold_hits[0].memory_id == rec.memory_id

    def test_cold_destination_already_contains_same_id(self, memory_manager):
        """D. Destination already containing same memory_id fails closed without overwrite."""
        mid = make_memory_id()
        now = iso_now()

        # Create hot record
        hot_rec = MemoryRecord(
            memory_id=mid,
            content="Original hot content",
            memory_type="hot",
            category="decision",
            source="agent_call",
            session_id="sess_conflict",
            created_at=now,
            updated_at=now,
            importance=0.8,
            confidence=0.9,
            is_summary=False,
            parent_memory_ids=[],
        )
        memory_manager.add_hot_memory(hot_rec)

        # Pre-create conflicting cold record directly in canonical store
        cold_existing = MemoryRecord(
            memory_id=mid,
            content="Pre-existing cold content that must never be overwritten",
            memory_type="cold",
            category="preference",
            source="agent_call",
            session_id=None,
            created_at=now,
            updated_at=now,
            importance=0.95,
            confidence=0.99,
            is_summary=False,
            parent_memory_ids=[],
        )
        memory_manager.store._write_record(cold_existing)

        # Attempt promotion
        with pytest.raises(MemoryManagerError, match="Cold memory already exists with ID"):
            memory_manager.promote_to_cold(mid)

        # Verify:
        # 1. Pre-existing cold record was not overwritten
        preserved_cold = memory_manager.store.get(mid, "cold")
        assert preserved_cold is not None
        assert preserved_cold.content == "Pre-existing cold content that must never be overwritten"

        # 2. Original hot record remains safe
        preserved_hot = memory_manager.store.get(mid, "hot")
        assert preserved_hot is not None
        assert preserved_hot.content == "Original hot content"

    def test_repeated_promotion_after_failed_attempt(self, memory_manager, monkeypatch):
        """E. Failed promotion does not corrupt record; subsequent valid promotion succeeds."""
        rec = _create_hot_record("sess_fail_e", "Retryable promotion item")
        memory_manager.add_hot_memory(rec)

        # Inject temporary failure on cold write
        original_write = memory_manager.store._write_record
        fail_now = True

        def intermittent_write(record: MemoryRecord) -> None:
            if record.memory_type == "cold" and fail_now:
                raise OSError("Transient disk error")
            original_write(record)

        monkeypatch.setattr(memory_manager.store, "_write_record", intermittent_write)

        # Attempt 1: fails
        with pytest.raises(MemoryManagerError):
            memory_manager.promote_to_cold(rec.memory_id)

        # Hot record is uncorrupted
        stored_hot = memory_manager.store.get(rec.memory_id, "hot")
        assert stored_hot is not None
        assert stored_hot.content == rec.content
        assert stored_hot.memory_type == "hot"

        # Now clear the transient error
        fail_now = False

        # Attempt 2: succeeds completely
        promoted = memory_manager.promote_to_cold(rec.memory_id)
        assert promoted.memory_id == rec.memory_id
        assert promoted.memory_type == "cold"
        assert memory_manager.store.get(rec.memory_id, "hot") is None
        assert memory_manager.store.get(rec.memory_id, "cold") is not None
        assert memory_manager.index.count("hot") == 0
        assert memory_manager.index.count("cold") == 1

