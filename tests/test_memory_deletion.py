"""
tests/test_memory_deletion.py
Phase 13 tests: Explicit memory deletion, cross-tier isolation, and no-auto-prune verification.

Validates:
1. Deleting hot memory removes canonical JSON from disk and unindexes from Chroma hot collection.
2. Deleting cold memory removes canonical JSON from disk and unindexes from Chroma cold collection.
3. Deleting non-existent memory returns clean envelope with deleted=False.
4. Idempotent deletion: deleting twice succeeds gracefully with deleted=False on the second call.
5. Cross-tier deletion isolation: deleting hot with memory_type='cold' (or vice versa) does not delete the record.
6. Validation and error handling in tool_memory_delete (missing ID, bad format, invalid type).
7. Validation and error handling in MemoryManager.delete_memory.
8. No-auto-delete policy: Python backend never autonomously deletes or evicts memories.
9. Summarization retention: summarization preserves parent memories without auto-deletion.
10. Gemma projection via map_memory_delete_result (zero path leakage).
11. End-to-end dispatch through Orchestrator._dispatch_tool_call (canonical name and short alias).
"""

from __future__ import annotations

import pytest

from core.dispatcher import ToolRegistry
from core.mappers.memory_results import map_memory_delete_result
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager, MemoryManagerError
from sage_memory.memory_store import MemoryStore
from tools.memory import (
    register_memory_tools,
    tool_memory_delete,
    tool_memory_search_cold,
    tool_memory_search_hot,
    tool_memory_store_cold,
    tool_memory_store_hot,
    tool_memory_summarize,
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


# ─── 1. Hot Memory Deletion Tests ─────────────────────────────────────────────

class TestHotMemoryDeletion:
    def test_delete_hot_memory_success(self, memory_manager):
        """Deleting hot memory removes file from disk and deletes from Chroma hot collection."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Transient user thought for current session",
            session_id="sess_del_hot_1",
            category="task",
        )
        assert stored["status"] == "success"
        mid = stored["memory_id"]

        # Confirm existence
        assert memory_manager.get_memory(mid, "hot") is not None
        assert memory_manager.index.count("hot") == 1

        # Delete hot memory
        res = tool_memory_delete(
            manager=memory_manager,
            memory_id=mid,
            memory_type="hot",
        )
        assert res["status"] == "success"
        assert res["deleted"] is True
        assert res["memory_id"] == mid
        assert res["memory_type"] == "hot"

        # Verify disk: canonical JSON file unlinked
        assert memory_manager.get_memory(mid, "hot") is None
        hot_file = memory_manager.store.hot_dir / f"{mid}.json"
        assert not hot_file.exists()

        # Verify Chroma: count decremented and search returns empty
        assert memory_manager.index.count("hot") == 0
        search_res = tool_memory_search_hot(
            manager=memory_manager,
            query="Transient user thought",
            session_id="sess_del_hot_1",
        )
        assert search_res["count"] == 0

    def test_delete_hot_memory_isolated_to_session(self, memory_manager):
        """Deleting a hot memory in session A leaves session B memories intact."""
        rec_a = tool_memory_store_hot(
            manager=memory_manager,
            content="Session A note",
            session_id="sess_A",
        )
        rec_b = tool_memory_store_hot(
            manager=memory_manager,
            content="Session B note",
            session_id="sess_B",
        )
        assert memory_manager.index.count("hot") == 2

        # Delete session A's memory
        res = tool_memory_delete(memory_manager, rec_a["memory_id"], "hot")
        assert res["deleted"] is True

        # Session A search is empty
        res_a = tool_memory_search_hot(memory_manager, "Session A note", session_id="sess_A")
        assert res_a["count"] == 0

        # Session B search still finds session B note
        res_b = tool_memory_search_hot(memory_manager, "Session B note", session_id="sess_B")
        assert res_b["count"] == 1
        assert res_b["results"][0]["memory_id"] == rec_b["memory_id"]


# ─── 2. Cold Memory Deletion Tests ────────────────────────────────────────────

class TestColdMemoryDeletion:
    def test_delete_cold_memory_success(self, memory_manager):
        """Deleting cold memory removes file from disk and deletes from Chroma cold collection."""
        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="User prefers dark theme across all editors",
            category="preference",
        )
        assert stored["status"] == "success"
        mid = stored["memory_id"]

        # Confirm existence
        assert memory_manager.get_memory(mid, "cold") is not None
        assert memory_manager.index.count("cold") == 1

        # Delete cold memory
        res = tool_memory_delete(
            manager=memory_manager,
            memory_id=mid,
            memory_type="cold",
        )
        assert res["status"] == "success"
        assert res["deleted"] is True
        assert res["memory_id"] == mid
        assert res["memory_type"] == "cold"

        # Verify disk: canonical JSON file unlinked
        assert memory_manager.get_memory(mid, "cold") is None
        cold_file = memory_manager.store.cold_dir / f"{mid}.json"
        assert not cold_file.exists()

        # Verify Chroma: count decremented and search returns empty
        assert memory_manager.index.count("cold") == 0
        search_res = tool_memory_search_cold(
            manager=memory_manager,
            query="dark theme editor preference",
        )
        assert search_res["count"] == 0

    def test_delete_cold_memory_cross_session_removal(self, memory_manager):
        """Deleted cold memory is not accessible from any session."""
        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="Global project config: port 8080",
            category="project",
        )
        mid = stored["memory_id"]

        # Delete
        res = tool_memory_delete(memory_manager, mid, "cold")
        assert res["deleted"] is True

        # Query cold memory
        hits = memory_manager.search_cold_memory("port 8080")
        assert len(hits) == 0


# ─── 3. Non-Existent & Idempotent Deletion Tests ──────────────────────────────

class TestNonExistentAndIdempotentDeletion:
    def test_delete_nonexistent_hot_memory(self, memory_manager):
        """Deleting a non-existent hot memory returns deleted=False without crashing."""
        res = tool_memory_delete(
            manager=memory_manager,
            memory_id="mem_nonexistent_9999",
            memory_type="hot",
        )
        assert res["status"] == "success"
        assert res["deleted"] is False
        assert res["memory_id"] == "mem_nonexistent_9999"

    def test_delete_nonexistent_cold_memory(self, memory_manager):
        """Deleting a non-existent cold memory returns deleted=False without crashing."""
        res = tool_memory_delete(
            manager=memory_manager,
            memory_id="mem_nonexistent_8888",
            memory_type="cold",
        )
        assert res["status"] == "success"
        assert res["deleted"] is False
        assert res["memory_id"] == "mem_nonexistent_8888"

    def test_delete_idempotent_double_delete(self, memory_manager):
        """Deleting the same memory twice succeeds gracefully with deleted=False on second call."""
        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="Temporary fact to delete twice",
        )
        mid = stored["memory_id"]

        # First deletion
        del1 = tool_memory_delete(memory_manager, mid, "cold")
        assert del1["status"] == "success"
        assert del1["deleted"] is True

        # Second deletion
        del2 = tool_memory_delete(memory_manager, mid, "cold")
        assert del2["status"] == "success"
        assert del2["deleted"] is False
        assert del2["memory_id"] == mid


# ─── 4. Cross-Tier Deletion Isolation Tests ───────────────────────────────────

class TestCrossTierDeletionIsolation:
    def test_delete_hot_with_cold_type_leaves_hot_intact(self, memory_manager):
        """Calling delete with memory_type='cold' on a hot memory ID does not delete the hot memory."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Hot memory safe from wrong-tier deletion",
            session_id="sess_cross",
        )
        mid = stored["memory_id"]

        # Attempt to delete with cold tier
        res = tool_memory_delete(memory_manager, mid, "cold")
        assert res["status"] == "success"
        assert res["deleted"] is False

        # Hot memory is still intact on disk and in Chroma
        assert memory_manager.get_memory(mid, "hot") is not None
        assert memory_manager.index.count("hot") == 1

    def test_delete_cold_with_hot_type_leaves_cold_intact(self, memory_manager):
        """Calling delete with memory_type='hot' on a cold memory ID does not delete the cold memory."""
        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="Cold memory safe from wrong-tier deletion",
        )
        mid = stored["memory_id"]

        # Attempt to delete with hot tier
        res = tool_memory_delete(memory_manager, mid, "hot")
        assert res["status"] == "success"
        assert res["deleted"] is False

        # Cold memory is still intact on disk and in Chroma
        assert memory_manager.get_memory(mid, "cold") is not None
        assert memory_manager.index.count("cold") == 1


# ─── 5. Tool Adapter Input Validation Tests ───────────────────────────────────

class TestToolMemoryDeleteValidation:
    def test_missing_or_blank_memory_id(self, memory_manager):
        res1 = tool_memory_delete(memory_manager, memory_id="", memory_type="hot")
        assert res1["status"] == "error"
        assert "memory_id is required" in res1["error"]
        assert res1["deleted"] is False

        res2 = tool_memory_delete(memory_manager, memory_id="   ", memory_type="cold")
        assert res2["status"] == "error"
        assert "memory_id is required" in res2["error"]

    def test_invalid_memory_id_format(self, memory_manager):
        res = tool_memory_delete(memory_manager, memory_id="raw_id_1234", memory_type="hot")
        assert res["status"] == "error"
        assert "must start with 'mem_'" in res["error"]
        assert res["deleted"] is False

    def test_invalid_memory_type(self, memory_manager):
        res = tool_memory_delete(memory_manager, memory_id="mem_123456789abc", memory_type="archive")
        assert res["status"] == "error"
        assert "memory_type must be 'hot' or 'cold'" in res["error"]
        assert res["deleted"] is False


# ─── 6. MemoryManager Direct Validation Tests ─────────────────────────────────

class TestMemoryManagerDeleteValidation:
    def test_manager_delete_missing_id_raises(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="memory_id must be a non-empty string"):
            memory_manager.delete_memory("", "hot")

    def test_manager_delete_invalid_id_raises(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="must start with 'mem_'"):
            memory_manager.delete_memory("bad_prefix", "cold")

    def test_manager_delete_invalid_type_raises(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="memory_type must be 'hot' or 'cold'"):
            memory_manager.delete_memory("mem_123456789abc", "warm")


# ─── 7. No-Auto-Delete / Retention Policy Tests ───────────────────────────────

class TestNoAutoDeletePolicy:
    def test_creating_or_searching_never_deletes_records(self, memory_manager):
        """Storing new memories or running searches never deletes or evicts existing memories."""
        m1 = tool_memory_store_hot(memory_manager, "Fact 1", session_id="sess_retention")
        m2 = tool_memory_store_hot(memory_manager, "Fact 2", session_id="sess_retention")
        m3 = tool_memory_store_hot(memory_manager, "Fact 3", session_id="sess_retention")

        # Perform searches
        tool_memory_search_hot(memory_manager, "Fact 1", session_id="sess_retention")
        tool_memory_search_hot(memory_manager, "unknown", session_id="sess_retention")

        # All 3 records remain in store and index
        assert memory_manager.get_memory(m1["memory_id"], "hot") is not None
        assert memory_manager.get_memory(m2["memory_id"], "hot") is not None
        assert memory_manager.get_memory(m3["memory_id"], "hot") is not None
        assert memory_manager.index.count("hot") == 3

    def test_summarization_does_not_auto_delete_parents(self, memory_manager):
        """Summarization explicitly preserves parent memories (no autonomous deletion)."""
        p1 = tool_memory_store_hot(memory_manager, "Detailed note 1", session_id="sess_sum_del")
        p2 = tool_memory_store_hot(memory_manager, "Detailed note 2", session_id="sess_sum_del")

        sum_res = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_sum_del",
            summary_content="Consolidated summary",
            parent_memory_ids=[p1["memory_id"], p2["memory_id"]],
        )
        assert sum_res["status"] == "success"

        # Parents are still in canonical store and Chroma
        assert memory_manager.get_memory(p1["memory_id"], "hot") is not None
        assert memory_manager.get_memory(p2["memory_id"], "hot") is not None
        assert memory_manager.index.count("hot") == 3  # 2 parents + 1 summary


# ─── 8. Gemma Projection & Orchestrator Dispatch ─────────────────────────────

class TestMemoryDeleteGemmaIntegration:
    def test_map_memory_delete_result_projection(self):
        """map_memory_delete_result produces a clean envelope with zero path leakage."""
        adapter_output = {
            "status": "success",
            "memory_id": "mem_del_123456789",
            "deleted": True,
            "memory_type": "cold",
            "local_path": "/path/to/data/memory/cold/mem_del_123456789.json",
            "chroma_collection": "sage_memory_cold",
        }
        projected = map_memory_delete_result(adapter_output)
        assert projected["memory_id"] == "mem_del_123456789"
        assert projected["deleted"] is True

        # Strict check for zero path or internal telemetry leakage
        forbidden = ("local_path", "chroma_collection", "memory_type")
        for key in forbidden:
            assert key not in projected

    def test_orchestrator_dispatch_memory_delete(self, memory_manager, tool_registry):
        """Orchestrator._dispatch_tool_call routes memory_delete through registry and mapper."""
        orchestrator = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_del_disp", user_text="Delete obsolete note")

        # Store memory
        store_res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_store_hot",
                "arguments": {
                    "content": "Obsolete configuration note",
                    "session_id": "sess_del_orch",
                    "category": "task",
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

        # Dispatch memory_delete
        del_res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_delete",
                "arguments": {
                    "memory_id": mid,
                    "memory_type": "hot",
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=1,
        )
        assert del_res["status"] == "success"
        assert del_res["result"]["memory_id"] == mid
        assert del_res["result"]["deleted"] is True

        # Verify disk reflects deletion
        assert memory_manager.get_memory(mid, "hot") is None

    def test_orchestrator_dispatch_delete_alias(self, memory_manager, tool_registry):
        """Orchestrator._dispatch_tool_call supports 'delete' alias."""
        orchestrator = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_del_alias", user_text="Delete cold memory")

        stored = tool_memory_store_cold(
            manager=memory_manager,
            content="Cold note to delete via alias",
        )
        mid = stored["memory_id"]

        del_res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "delete",
                "arguments": {
                    "memory_id": mid,
                    "memory_type": "cold",
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )
        assert del_res["status"] == "success"
        assert del_res["result"]["memory_id"] == mid
        assert del_res["result"]["deleted"] is True

    def test_orchestrator_dispatch_delete_error(self, memory_manager, tool_registry):
        """Orchestrator._dispatch_tool_call handles memory_delete failure with error envelope."""
        orchestrator = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_del_err", user_text="Delete with invalid type")

        del_res = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_delete",
                "arguments": {
                    "memory_id": "mem_invalid_type_test",
                    "memory_type": "unknown_tier",
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )
        assert del_res["status"] == "error"
        assert "memory_type must be 'hot' or 'cold'" in del_res["error"]["message"]
