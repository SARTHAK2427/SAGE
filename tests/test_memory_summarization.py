"""
tests/test_memory_summarization.py
Phase 10 tests: Hot memory summarization, threshold evaluation, and parent retention.

Validates:
1. Summarize hot memory: dual-writes to canonical store + Chroma hot collection.
2. Summary record attributes: is_summary=True, category="summary", parent_memory_ids.
3. Retention of raw parent memories (never deleted or mutated during summarization).
4. Validation of session_id, summary_content, parent_memory_ids, and unit-interval scores.
5. Threshold evaluation via MemorySummarizer (MEMORY_HOT_SUMMARY_THRESHOLD).
6. Tool adapter validation and structured error envelope.
7. Pure Gemma projection via map_memory_summarize_result.
8. End-to-end dispatch through Orchestrator._dispatch_tool_call.
9. No-auto-summarize policy: Python backend never autonomously synthesizes or triggers summaries.
"""

from __future__ import annotations

import os
import pytest

from core.dispatcher import ToolRegistry
from core.mappers.memory_results import map_memory_summarize_result
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_document_db.utils import iso_now
from sage_memory.config import MEMORY_HOT_SUMMARY_THRESHOLD
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager, MemoryManagerError
from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore
from sage_memory.memory_summarizer import MemorySummarizer
from tools.memory import (
    register_memory_tools,
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


def _create_hot_record(session_id: str, content: str) -> MemoryRecord:
    now = iso_now()
    return MemoryRecord(
        memory_id=make_memory_id(),
        content=content,
        memory_type="hot",
        category="task",
        source="agent_call",
        session_id=session_id,
        created_at=now,
        updated_at=now,
        importance=0.6,
        confidence=0.9,
        is_summary=False,
        parent_memory_ids=[],
    )


# ─── 1. Hot Memory Summarization Core Tests ───────────────────────────────────

class TestHotMemorySummarization:
    def test_summarize_creates_summary_record_and_dual_writes(self, memory_manager):
        """Summarizing hot memories writes to canonical store and Chroma hot index."""
        p1 = memory_manager.add_hot_memory(_create_hot_record("sess_s1", "Step 1: parse input"))
        p2 = memory_manager.add_hot_memory(_create_hot_record("sess_s1", "Step 2: build AST"))
        p3 = memory_manager.add_hot_memory(_create_hot_record("sess_s1", "Step 3: typecheck AST"))

        assert memory_manager.index.count("hot") == 3
        assert memory_manager.index.count("cold") == 0

        summary = memory_manager.summarize_hot_memory(
            session_id="sess_s1",
            summary_content="Compilation pipeline steps 1 through 3 completed successfully",
            parent_memory_ids=[p1.memory_id, p2.memory_id, p3.memory_id],
            category="summary",
            importance=0.8,
            confidence=0.95,
        )

        # 1. Summary record properties
        assert summary.memory_id.startswith("mem_")
        assert summary.is_summary is True
        assert summary.category == "summary"
        assert summary.session_id == "sess_s1"
        assert summary.parent_memory_ids == [p1.memory_id, p2.memory_id, p3.memory_id]
        assert summary.importance == 0.8
        assert summary.confidence == 0.95

        # 2. Canonical store check
        stored = memory_manager.store.get(summary.memory_id, "hot")
        assert stored is not None
        assert stored.content == "Compilation pipeline steps 1 through 3 completed successfully"
        assert stored.is_summary is True

        # 3. Chroma hot index check (now 4 items)
        assert memory_manager.index.count("hot") == 4
        # 4. Cold collection untouched
        assert memory_manager.index.count("cold") == 0

        # 5. Search filtering by is_summary
        summaries = memory_manager.search_hot_memory(
            query="compilation pipeline",
            session_id="sess_s1",
            is_summary=True,
        )
        assert len(summaries) >= 1
        assert summaries[0].memory_id == summary.memory_id
        assert summaries[0].is_summary is True

        raw_records = memory_manager.search_hot_memory(
            query="pipeline",
            session_id="sess_s1",
            is_summary=False,
        )
        for r in raw_records:
            assert r.is_summary is False

    def test_parent_memories_retained_unmodified(self, memory_manager):
        """Parent raw memories are retained intact and never deleted upon summarization."""
        p1 = memory_manager.add_hot_memory(_create_hot_record("sess_s2", "Fact A"))
        p2 = memory_manager.add_hot_memory(_create_hot_record("sess_s2", "Fact B"))

        memory_manager.summarize_hot_memory(
            session_id="sess_s2",
            summary_content="Summary of A and B",
            parent_memory_ids=[p1.memory_id, p2.memory_id],
        )

        # Both parent records still exist in canonical store
        assert memory_manager.store.get(p1.memory_id, "hot") is not None
        assert memory_manager.store.get(p2.memory_id, "hot") is not None

        # Both parent records still exist in Chroma hot index
        raw_hits = memory_manager.search_hot_memory("Fact", session_id="sess_s2", is_summary=False)
        parent_ids_found = {r.memory_id for r in raw_hits}
        assert p1.memory_id in parent_ids_found
        assert p2.memory_id in parent_ids_found

    def test_summarize_validation_parent_ids(self, memory_manager):
        """Parent IDs must exist in hot store and belong to the same session."""
        p1 = memory_manager.add_hot_memory(_create_hot_record("sess_s3", "Session 3 fact"))
        p_other = memory_manager.add_hot_memory(_create_hot_record("sess_other", "Different session fact"))

        # Non-existent parent ID
        with pytest.raises(MemoryManagerError, match="not found in hot memory store"):
            memory_manager.summarize_hot_memory(
                session_id="sess_s3",
                summary_content="Summary",
                parent_memory_ids=["mem_nonexistent999"],
            )

        # Parent belonging to different session
        with pytest.raises(MemoryManagerError, match="belongs to session"):
            memory_manager.summarize_hot_memory(
                session_id="sess_s3",
                summary_content="Summary",
                parent_memory_ids=[p1.memory_id, p_other.memory_id],
            )

        # Empty parent_memory_ids
        with pytest.raises(MemoryManagerError, match="parent_memory_ids must be a non-empty list"):
            memory_manager.summarize_hot_memory(
                session_id="sess_s3",
                summary_content="Summary",
                parent_memory_ids=[],
            )

    def test_summarize_importance_confidence_validation(self, memory_manager):
        """Out-of-range importance and confidence must be rejected deterministically."""
        p1 = memory_manager.add_hot_memory(_create_hot_record("sess_s4", "Valid item"))

        with pytest.raises(MemoryManagerError, match="importance must be in \\[0.0, 1.0\\]"):
            memory_manager.summarize_hot_memory(
                session_id="sess_s4",
                summary_content="Summary",
                parent_memory_ids=[p1.memory_id],
                importance=1.5,
            )

        with pytest.raises(MemoryManagerError, match="confidence must be in \\[0.0, 1.0\\]"):
            memory_manager.summarize_hot_memory(
                session_id="sess_s4",
                summary_content="Summary",
                parent_memory_ids=[p1.memory_id],
                confidence=-0.3,
            )


# ─── 2. Threshold Evaluation Tests ───────────────────────────────────────────

class TestMemorySummarizerThreshold:
    def test_threshold_evaluation_values(self, memory_manager):
        """Threshold helper evaluates hot memory counts against configured threshold."""
        summarizer = memory_manager.summarizer
        assert summarizer.threshold == MEMORY_HOT_SUMMARY_THRESHOLD

        # Explicit count evaluations
        assert summarizer.should_summarize(MEMORY_HOT_SUMMARY_THRESHOLD - 1) is False
        assert summarizer.should_summarize(MEMORY_HOT_SUMMARY_THRESHOLD) is True
        assert summarizer.should_summarize(MEMORY_HOT_SUMMARY_THRESHOLD + 5) is True

    def test_session_threshold_detection(self, memory_manager):
        """should_summarize_session checks session count against threshold."""
        session_id = "sess_threshold_test"

        # Initially 0 memories
        assert memory_manager.should_summarize_session(session_id) is False

        # Add 19 memories
        for i in range(19):
            memory_manager.add_hot_memory(_create_hot_record(session_id, f"Hot memory {i}"))

        assert memory_manager.summarizer.count_hot_memories(session_id) == 19
        assert memory_manager.should_summarize_session(session_id) is False

        # Add 20th memory (reaches default threshold of 20)
        memory_manager.add_hot_memory(_create_hot_record(session_id, "Hot memory 19"))
        assert memory_manager.summarizer.count_hot_memories(session_id) == 20
        assert memory_manager.should_summarize_session(session_id) is True

    def test_summarizer_create_and_persist(self, memory_manager):
        """MemorySummarizer can create and persist validated summary records."""
        summarizer = memory_manager.summarizer
        p1 = memory_manager.add_hot_memory(_create_hot_record("sess_helper", "Detail A"))

        record = summarizer.create_summary_record(
            memory_id=make_memory_id(),
            content="Summary through summarizer helper",
            session_id="sess_helper",
            parent_memory_ids=[p1.memory_id],
            created_at=iso_now(),
            updated_at=iso_now(),
            importance=0.7,
            confidence=0.8,
        )
        assert record.is_summary is True

        persisted = summarizer.persist_summary(record)
        assert persisted.memory_id == record.memory_id
        assert memory_manager.store.get(record.memory_id, "hot") is not None
        assert memory_manager.index.count("hot") == 2

        # Calling persist_summary on non-summary record raises ValueError
        non_summary = _create_hot_record("sess_helper", "Not a summary")
        with pytest.raises(ValueError, match="is_summary must be True"):
            summarizer.persist_summary(non_summary)


# ─── 3. Tool Adapter Summarize Tests ───────────────────────────────────────────

class TestToolAdapterSummarize:
    def test_tool_memory_summarize_validation(self, memory_manager):
        """tool_memory_summarize validates input parameters and returns structured error."""
        # Missing session_id
        res1 = tool_memory_summarize(
            manager=memory_manager,
            session_id="",
            summary_content="Valid summary",
            parent_memory_ids=["mem_1"],
        )
        assert res1["status"] == "error"
        assert "session_id is required" in res1["error"]
        assert res1["memory_id"] is None

        # Missing summary_content
        res2 = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_1",
            summary_content="   ",
            parent_memory_ids=["mem_1"],
        )
        assert res2["status"] == "error"
        assert "summary_content is required" in res2["error"]

        # Invalid parent_memory_ids
        res3 = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_1",
            summary_content="Valid",
            parent_memory_ids=[],
        )
        assert res3["status"] == "error"
        assert "parent_memory_ids must be a non-empty list" in res3["error"]

        # Out-of-range importance
        res4 = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_1",
            summary_content="Valid",
            parent_memory_ids=["mem_1"],
            importance=1.6,
        )
        assert res4["status"] == "error"
        assert "importance must be in [0.0, 1.0]" in res4["error"]

        # Out-of-range confidence
        res5 = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_1",
            summary_content="Valid",
            parent_memory_ids=["mem_1"],
            confidence=-0.4,
        )
        assert res5["status"] == "error"
        assert "confidence must be in [0.0, 1.0]" in res5["error"]

    def test_tool_memory_summarize_success(self, memory_manager):
        """Successful tool_memory_summarize creates and returns summary record."""
        p1 = tool_memory_store_hot(memory_manager, "Hot task 1", session_id="sess_adapt")
        p2 = tool_memory_store_hot(memory_manager, "Hot task 2", session_id="sess_adapt")

        res = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_adapt",
            summary_content="Tasks 1 and 2 completed",
            parent_memory_ids=[p1["memory_id"], p2["memory_id"]],
            category="summary",
            importance=0.85,
            confidence=0.9,
        )

        assert res["status"] == "success"
        assert res["memory_id"].startswith("mem_")
        assert res["memory"]["is_summary"] is True
        assert res["memory"]["importance"] == 0.85
        assert res["memory"]["confidence"] == 0.9
        assert res["memory"]["parent_memory_ids"] == [p1["memory_id"], p2["memory_id"]]


# ─── 4. Gemma Result Projection & Orchestrator Dispatch Tests ─────────────────

class TestSummarizeGemmaIntegration:
    def test_map_memory_summarize_result_projection(self):
        """map_memory_summarize_result sanitizes internal fields and returns clean envelope."""
        adapter_output = {
            "status": "success",
            "memory_id": "mem_sum_123",
            "memory": {
                "memory_id": "mem_sum_123",
                "content": "Consolidated summary text",
                "memory_type": "hot",
                "category": "summary",
                "is_summary": True,
                "parent_memory_ids": ["mem_p1", "mem_p2"],
                "local_path": "/canonical/data/memory/hot/mem_sum_123.json",  # stripped
                "chroma_collection": "sage_memory_hot",  # stripped
            },
        }

        projected = map_memory_summarize_result(adapter_output)
        assert projected["memory_id"] == "mem_sum_123"
        assert projected["status"] == "summarized"
        assert projected["is_summary"] is True
        assert projected["parent_memory_ids"] == ["mem_p1", "mem_p2"]
        assert "local_path" not in projected
        assert "chroma_collection" not in projected

    def test_orchestrator_dispatch_summarize(self, memory_manager, tool_registry):
        """Orchestrator dispatches memory_summarize tool call end-to-end."""
        p1 = tool_memory_store_hot(memory_manager, "Work item 1", session_id="sess_orch")
        p2 = tool_memory_store_hot(memory_manager, "Work item 2", session_id="sess_orch")

        orch = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_orch_sum", user_text="Please summarize completed tasks.")

        tool_res = orch._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_summarize",
                "arguments": {
                    "session_id": "sess_orch",
                    "summary_content": "Work items 1 and 2 completed",
                    "parent_memory_ids": [p1["memory_id"], p2["memory_id"]],
                    "category": "summary",
                },
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )

        assert tool_res["status"] == "success"
        assert tool_res["result"]["status"] == "summarized"
        assert tool_res["result"]["is_summary"] is True
        assert tool_res["result"]["parent_memory_ids"] == [p1["memory_id"], p2["memory_id"]]


# ─── 5. No Autonomous Summarization Policy Tests ──────────────────────────────

class TestNoAutoSummarizePolicy:
    def test_backend_never_auto_summarizes(self, memory_manager):
        """Even if hot memory count exceeds threshold, backend does not automatically summarize."""
        session_id = "sess_no_auto"

        # Add 25 memories (exceeds threshold 20)
        for i in range(25):
            memory_manager.add_hot_memory(_create_hot_record(session_id, f"Raw memory {i}"))

        assert memory_manager.summarizer.count_hot_memories(session_id) == 25
        assert memory_manager.should_summarize_session(session_id) is True

        # Verify NO summary records exist yet (count of summaries in store and index is 0)
        hot_records = memory_manager.store.list_hot(session_id=session_id)
        summary_records = [r for r in hot_records if r.is_summary]
        assert len(summary_records) == 0

        # Only when Gemma explicitly issues a summarization call does a summary record get created
        summary = memory_manager.summarize_hot_memory(
            session_id=session_id,
            summary_content="Consolidated summary of 25 raw items",
            parent_memory_ids=[r.memory_id for r in hot_records[:5]],
        )
        assert summary.is_summary is True

        hot_records_after = memory_manager.store.list_hot(session_id=session_id)
        summaries_after = [r for r in hot_records_after if r.is_summary]
        assert len(summaries_after) == 1
