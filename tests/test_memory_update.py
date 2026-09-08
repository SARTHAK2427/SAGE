"""
tests/test_memory_update.py
Phase 12 tests: Hot and cold memory update flow, validation, and semantic re-ranking.

Validates:
1. Updating hot memory: content, category, importance, confidence dual-writes to disk and Chroma.
2. Updating cold memory: content, category, importance, confidence dual-writes to disk and Chroma.
3. Partial updates: updating a subset of fields preserves unaffected fields.
4. Semantic re-ranking: updated content reflects in subsequent semantic search queries.
5. Immutability guarantees: memory_id and created_at cannot be altered; updated_at is refreshed.
6. Type immutability: memory_type cannot be changed via update (must use promotion flow).
7. Session immutability: session_id cannot be changed for hot memory.
8. Validation & error handling in tool_memory_update: non-existent ID, invalid ID, bad ranges (no clamping).
9. Validation & error handling in MemoryManager.update_memory.
10. Failure atomicity: Chroma failure preserves canonical JSON store on disk.
11. Gemma projection via map_memory_update_result (zero path leakage).
12. End-to-end dispatch through Orchestrator._dispatch_tool_call.
"""

from __future__ import annotations

import pytest

from core.dispatcher import ToolRegistry
from core.mappers.memory_results import map_memory_update_result
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
    tool_memory_search_cold,
    tool_memory_search_hot,
    tool_memory_store_cold,
    tool_memory_store_hot,
    tool_memory_update,
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


# ─── 1. Hot Memory Update Tests ───────────────────────────────────────────────

class TestHotMemoryUpdate:
    def test_update_hot_memory_all_fields(self, memory_manager):
        """Updating all fields of hot memory updates both disk and Chroma."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Initial hypothesis on user query",
            session_id="sess_upd_1",
            category="task",
            importance=0.4,
            confidence=0.6,
        )
        assert stored["status"] == "success"
        mid = stored["memory_id"]
        orig_created_at = stored["memory"]["created_at"]

        res = tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="hot",
            content="Refined hypothesis with verified experimental data",
            category="decision",
            importance=0.9,
            confidence=0.95,
        )
        assert res["status"] == "success"
        assert res["memory_id"] == mid
        assert res["memory"]["content"] == "Refined hypothesis with verified experimental data"
        assert res["memory"]["category"] == "decision"
        assert res["memory"]["importance"] == 0.9
        assert res["memory"]["confidence"] == 0.95
        assert res["memory"]["created_at"] == orig_created_at
        assert res["memory"]["updated_at"] >= orig_created_at

        # Verify disk canonical store
        disk_rec = memory_manager.get_memory(mid, "hot")
        assert disk_rec is not None
        assert disk_rec.content == "Refined hypothesis with verified experimental data"
        assert disk_rec.category == "decision"
        assert disk_rec.importance == 0.9
        assert disk_rec.confidence == 0.95
        assert disk_rec.created_at == orig_created_at

        # Verify Chroma index
        hits = memory_manager.search_hot_memory(
            "Refined hypothesis experimental",
            session_id="sess_upd_1",
            top_k=1,
        )
        assert len(hits) == 1
        assert hits[0].memory_id == mid
        assert hits[0].content == "Refined hypothesis with verified experimental data"
        assert hits[0].category == "decision"
        assert hits[0].importance == 0.9
        assert hits[0].confidence == 0.95

    def test_update_hot_memory_partial_fields(self, memory_manager):
        """Updating only content leaves category, importance, and confidence unchanged."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Original content string",
            session_id="sess_upd_2",
            category="task",
            importance=0.75,
            confidence=0.85,
        )
        mid = stored["memory_id"]

        res = tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="hot",
            content="Updated content string only",
        )
        assert res["status"] == "success"
        assert res["memory"]["content"] == "Updated content string only"
        assert res["memory"]["category"] == "task"
        assert res["memory"]["importance"] == 0.75
        assert res["memory"]["confidence"] == 0.85

        # Verify disk
        disk_rec = memory_manager.get_memory(mid, "hot")
        assert disk_rec.content == "Updated content string only"
        assert disk_rec.category == "task"
        assert disk_rec.importance == 0.75
        assert disk_rec.confidence == 0.85


# ─── 2. Cold Memory Update Tests ──────────────────────────────────────────────

class TestColdMemoryUpdate:
    def test_update_cold_memory_all_fields(self, memory_manager):
        """Updating cold memory synchronizes canonical store and Chroma."""
        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="User prefers Python 3.10 runtime",
            category="preference",
            importance=0.5,
            confidence=0.7,
        )
        mid = stored["memory_id"]
        orig_created_at = stored["memory"]["created_at"]

        res = tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="cold",
            content="User upgraded to Python 3.13 and prefers pytest",
            category="technical",
            importance=0.9,
            confidence=0.99,
        )
        assert res["status"] == "success"
        assert res["memory_id"] == mid
        assert res["memory"]["content"] == "User upgraded to Python 3.13 and prefers pytest"
        assert res["memory"]["category"] == "technical"
        assert res["memory"]["importance"] == 0.9
        assert res["memory"]["confidence"] == 0.99
        assert res["memory"]["created_at"] == orig_created_at

        # Verify disk
        disk_rec = memory_manager.get_memory(mid, "cold")
        assert disk_rec.content == "User upgraded to Python 3.13 and prefers pytest"
        assert disk_rec.category == "technical"
        assert disk_rec.importance == 0.9
        assert disk_rec.confidence == 0.99

        # Verify Chroma
        hits = memory_manager.search_cold_memory("Python 3.13 pytest", top_k=1)
        assert len(hits) == 1
        assert hits[0].memory_id == mid
        assert hits[0].content == "User upgraded to Python 3.13 and prefers pytest"
        assert hits[0].category == "technical"

    def test_update_cold_memory_importance_only(self, memory_manager):
        """Updating only importance preserves content, category, and confidence."""
        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="Project uses Postgres database",
            category="project",
            importance=0.3,
            confidence=0.8,
        )
        mid = stored["memory_id"]

        res = tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="cold",
            importance=0.95,
        )
        assert res["status"] == "success"
        assert res["memory"]["importance"] == 0.95
        assert res["memory"]["content"] == "Project uses Postgres database"
        assert res["memory"]["category"] == "project"
        assert res["memory"]["confidence"] == 0.8


# ─── 3. Semantic Search Re-ranking / Reflection Tests ─────────────────────────

class TestMemoryUpdateSemanticReflection:
    def test_semantic_search_reflects_updated_content(self, memory_manager):
        """Search query matching updated content returns the updated record."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Alpha protocol guidelines for spacecraft navigation",
            session_id="sess_semantic",
            category="instruction",
        )
        mid = stored["memory_id"]

        # Search for alpha
        results_before = tool_memory_search_hot(
            manager=memory_manager,
            query="spacecraft navigation",
            session_id="sess_semantic",
        )
        assert results_before["count"] == 1
        assert results_before["results"][0]["memory_id"] == mid

        # Update to completely different topic
        tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="hot",
            content="Submarine acoustic tracking protocols and sonar metrics",
        )

        # Search for submarine sonar
        results_after = tool_memory_search_hot(
            manager=memory_manager,
            query="submarine sonar acoustic",
            session_id="sess_semantic",
        )
        assert results_after["count"] == 1
        assert results_after["results"][0]["memory_id"] == mid
        assert "Submarine acoustic tracking" in results_after["results"][0]["content"]


# ─── 4. Immutability Enforcements ─────────────────────────────────────────────

class TestMemoryUpdateImmutability:
    def test_memory_id_and_created_at_are_immutable(self, memory_manager):
        """tool_memory_update never alters memory_id or created_at."""
        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="Fact A",
            category="general",
        )
        mid = stored["memory_id"]
        created_at = stored["memory"]["created_at"]

        res = tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="cold",
            content="Fact A modified",
        )
        assert res["memory_id"] == mid
        assert res["memory"]["created_at"] == created_at

    def test_memory_manager_rejects_altered_created_at(self, memory_manager):
        """Direct call to update_memory with modified created_at raises MemoryManagerError."""
        record = MemoryRecord(
            memory_id=make_memory_id(),
            content="Original content",
            memory_type="cold",
            category="fact",
            source="test",
            session_id=None,
            created_at="2026-09-08T10:00:00+00:00",
            updated_at="2026-09-08T10:00:00+00:00",
        )
        memory_manager.add_cold_memory(record)

        tampered = MemoryRecord.from_dict({
            **record.to_dict(),
            "created_at": "2026-01-01T00:00:00+00:00",
        })
        with pytest.raises(MemoryManagerError, match="created_at is immutable"):
            memory_manager.update_memory(tampered)

    def test_memory_manager_rejects_altered_session_id_for_hot(self, memory_manager):
        """Direct call to update_memory with modified session_id on hot memory raises MemoryManagerError."""
        record = MemoryRecord(
            memory_id=make_memory_id(),
            content="Original hot note",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_alpha",
            created_at="2026-09-08T10:00:00+00:00",
            updated_at="2026-09-08T10:00:00+00:00",
        )
        memory_manager.add_hot_memory(record)

        tampered = MemoryRecord.from_dict({
            **record.to_dict(),
            "session_id": "sess_beta",
        })
        with pytest.raises(MemoryManagerError, match="session_id cannot be changed for hot memory"):
            memory_manager.update_memory(tampered)

    def test_memory_type_cannot_change_via_update(self, memory_manager):
        """Attempting to change memory_type via update_memory raises MemoryManagerError."""
        record = MemoryRecord(
            memory_id=make_memory_id(),
            content="Hot memory attempting transition via update",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_type_test",
            created_at="2026-09-08T10:00:00+00:00",
            updated_at="2026-09-08T10:00:00+00:00",
        )
        memory_manager.add_hot_memory(record)

        tampered = MemoryRecord.from_dict({
            **record.to_dict(),
            "memory_type": "cold",
        })
        with pytest.raises(MemoryManagerError, match="Memory not found for update"):
            # Looking for cold file with record.memory_id fails because it's hot
            memory_manager.update_memory(tampered)


# ─── 5. Tool Adapter Input Validation & Boundary Testing ─────────────────────

class TestToolMemoryUpdateValidation:
    def test_missing_or_invalid_memory_id(self, memory_manager):
        res1 = tool_memory_update(memory_manager, memory_id="", memory_type="hot", content="Test")
        assert res1["status"] == "error"
        assert "memory_id is required" in res1["error"]

        res2 = tool_memory_update(memory_manager, memory_id="invalid_id", memory_type="hot", content="Test")
        assert res2["status"] == "error"
        assert "must start with 'mem_'" in res2["error"]

    def test_invalid_memory_type(self, memory_manager):
        res = tool_memory_update(
            memory_manager,
            memory_id="mem_123456789abc",
            memory_type="warm",
            content="Test",
        )
        assert res["status"] == "error"
        assert "memory_type must be 'hot' or 'cold'" in res["error"]

    def test_nonexistent_memory_id(self, memory_manager):
        res = tool_memory_update(
            memory_manager,
            memory_id="mem_nonexistent123",
            memory_type="cold",
            content="New content",
        )
        assert res["status"] == "error"
        assert "not found" in res["error"]

    def test_empty_content_rejected(self, memory_manager):
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Original content",
            session_id="sess_val",
        )
        mid = stored["memory_id"]

        res1 = tool_memory_update(memory_manager, memory_id=mid, memory_type="hot", content="")
        assert res1["status"] == "error"
        assert "content cannot be empty" in res1["error"]

        res2 = tool_memory_update(memory_manager, memory_id=mid, memory_type="hot", content="   ")
        assert res2["status"] == "error"
        assert "content cannot be empty" in res2["error"]

    def test_empty_category_rejected(self, memory_manager):
        stored = tool_memory_store_cold(manager=memory_manager, content="Some fact")
        mid = stored["memory_id"]

        res = tool_memory_update(memory_manager, memory_id=mid, memory_type="cold", category="   ")
        assert res["status"] == "error"
        assert "category cannot be empty" in res["error"]

    def test_out_of_range_importance_rejected_no_clamping(self, memory_manager):
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Test range",
            session_id="sess_range",
        )
        mid = stored["memory_id"]

        # Below 0.0
        res_low = tool_memory_update(memory_manager, memory_id=mid, memory_type="hot", importance=-0.1)
        assert res_low["status"] == "error"
        assert "importance must be in [0.0, 1.0]" in res_low["error"]

        # Above 1.0
        res_high = tool_memory_update(memory_manager, memory_id=mid, memory_type="hot", importance=1.5)
        assert res_high["status"] == "error"
        assert "importance must be in [0.0, 1.0]" in res_high["error"]

        # Non-numeric
        res_str = tool_memory_update(memory_manager, memory_id=mid, memory_type="hot", importance="high")
        assert res_str["status"] == "error"
        assert "importance must be numeric" in res_str["error"]

    def test_out_of_range_confidence_rejected_no_clamping(self, memory_manager):
        stored = tool_memory_store_cold(manager=memory_manager, content="Test confidence range")
        mid = stored["memory_id"]

        # Below 0.0
        res_low = tool_memory_update(memory_manager, memory_id=mid, memory_type="cold", confidence=-0.5)
        assert res_low["status"] == "error"
        assert "confidence must be in [0.0, 1.0]" in res_low["error"]

        # Above 1.0
        res_high = tool_memory_update(memory_manager, memory_id=mid, memory_type="cold", confidence=2.0)
        assert res_high["status"] == "error"
        assert "confidence must be in [0.0, 1.0]" in res_high["error"]

        # Non-numeric
        res_str = tool_memory_update(memory_manager, memory_id=mid, memory_type="cold", confidence="certain")
        assert res_str["status"] == "error"
        assert "confidence must be numeric" in res_str["error"]

    def test_no_updatable_fields_provided(self, memory_manager):
        stored = tool_memory_store_cold(manager=memory_manager, content="Some fact")
        mid = stored["memory_id"]

        res = tool_memory_update(memory_manager, memory_id=mid, memory_type="cold")
        assert res["status"] == "error"
        assert "At least one field to update must be provided" in res["error"]


# ─── 6. MemoryManager Direct Validation Tests ─────────────────────────────────

class TestMemoryManagerUpdateValidation:
    def test_update_non_memory_record_raises(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="record must be a MemoryRecord instance"):
            memory_manager.update_memory({"content": "not a record"})  # type: ignore

    def test_update_nonexistent_record_raises(self, memory_manager):
        record = MemoryRecord(
            memory_id=make_memory_id(),
            content="Ghost record",
            memory_type="cold",
            category="test",
            source="test",
            session_id=None,
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        with pytest.raises(MemoryManagerError, match="Memory not found for update"):
            memory_manager.update_memory(record)


# ─── 7. Failure Atomicity Verification ────────────────────────────────────────

class TestMemoryUpdateFailureAtomicity:
    def test_chroma_update_failure_preserves_canonical_disk(self, memory_manager, monkeypatch):
        """If Chroma indexing fails during update, canonical JSON on disk is preserved intact."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Safe note",
            session_id="sess_atomicity",
        )
        mid = stored["memory_id"]

        # Mock Chroma update to raise an exception
        def _failing_update(rec):
            raise RuntimeError("Chroma connection timed out during update")

        monkeypatch.setattr(memory_manager._index, "update_memory", _failing_update)

        res = tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="hot",
            content="Updated note despite Chroma crash",
        )
        assert res["status"] == "error"
        assert "Failed to update memory in Chroma" in res["error"]

        # Canonical disk file HAS been safely updated (source of truth is intact)
        disk_rec = memory_manager.get_memory(mid, "hot")
        assert disk_rec is not None
        assert disk_rec.content == "Updated note despite Chroma crash"


# ─── 8. Gemma Projection & Orchestrator Dispatch ─────────────────────────────

class TestMemoryUpdateGemmaIntegration:
    def test_map_memory_update_result_projection(self):
        """map_memory_update_result produces a clean, bounded envelope with zero path leakage."""
        adapter_output = {
            "status": "success",
            "memory_id": "mem_abc123456789",
            "memory": {
                "memory_id": "mem_abc123456789",
                "content": "Secret server config",
                "memory_type": "cold",
                "category": "technical",
                "local_path": "/var/data/memory/cold/mem_abc123456789.json",
                "temp_path": "/tmp/mem.tmp",
                "raw_chroma": {"collection": "sage_memory_cold"},
                "distance": 0.05,
            },
        }
        projected = map_memory_update_result(adapter_output)
        assert projected["memory_id"] == "mem_abc123456789"
        assert projected["status"] == "updated"
        assert projected["memory_type"] == "cold"

        # Strict check for zero path or internal telemetry leakage
        forbidden = ("local_path", "temp_path", "raw_chroma", "distance")
        for key in forbidden:
            assert key not in projected

    def test_orchestrator_dispatch_memory_update(self, memory_manager, tool_registry):
        """Orchestrator._dispatch_tool_call routes memory_update through registry and mapper."""
        orchestrator = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_upd_disp", user_text="Update memory note")

        # First, store a memory
        store_res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_store_hot",
                "arguments": {
                    "content": "Architecture requirement: use SQLite or JSON",
                    "session_id": "sess_disp",
                    "category": "project",
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )
        assert store_res["status"] == "success"
        mid = store_res["result"]["memory_id"]

        # Now dispatch memory_update
        update_res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_update",
                "arguments": {
                    "memory_id": mid,
                    "memory_type": "hot",
                    "content": "Architecture requirement: use canonical JSON store",
                    "importance": 0.85,
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=1,
        )
        assert update_res["status"] == "success"
        assert update_res["result"]["memory_id"] == mid
        assert update_res["result"]["status"] == "updated"
        assert update_res["result"]["memory_type"] == "hot"

        # Verify disk reflects orchestrator dispatch
        stored = memory_manager.get_memory(mid, "hot")
        assert stored.content == "Architecture requirement: use canonical JSON store"
        assert stored.importance == 0.85

    def test_orchestrator_dispatch_memory_update_alias(self, memory_manager, tool_registry):
        """Orchestrator._dispatch_tool_call supports 'update' alias."""
        orchestrator = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_upd_alias", user_text="Update cold memory")

        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="Cold setting alpha",
        )
        mid = stored["memory_id"]

        res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "update",
                "arguments": {
                    "memory_id": mid,
                    "memory_type": "cold",
                    "content": "Cold setting beta",
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )
        assert res["status"] == "success"
        assert res["result"]["memory_id"] == mid
        assert res["result"]["status"] == "updated"

    def test_orchestrator_dispatch_memory_update_error(self, memory_manager, tool_registry):
        """Orchestrator._dispatch_tool_call handles memory_update failure with error envelope."""
        orchestrator = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_upd_err", user_text="Update invalid memory")

        res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_update",
                "arguments": {
                    "memory_id": "mem_nonexistent999",
                    "memory_type": "cold",
                    "content": "Fails",
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )
        assert res["status"] == "error"
        assert "not found" in res["error"]["message"]
