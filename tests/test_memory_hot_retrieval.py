"""
tests/test_memory_hot_retrieval.py
Phase 7 tests: Hot memory retrieval, session isolation, and metadata filtering.

Validates:
1. Strict session isolation (Session A memories never leak into Session B).
2. Config-driven default top_k (MEMORY_DEFAULT_TOP_K).
3. Query-time metadata filtering (category, is_summary, importance, timestamps).
4. Semantic ranking order.
5. Mandatory session_id validation at manager and adapter levels.
6. Clean Gemma projection via map_memory_search_result (zero path/distance leakage).
7. End-to-end dispatch through ToolRegistry and Orchestrator.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import pytest

from core.dispatcher import ToolRegistry
from core.mappers.memory_results import map_memory_search_result
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_document_db.utils import iso_now
from sage_memory.config import MEMORY_DEFAULT_TOP_K
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager, MemoryManagerError
from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore
from tools.memory import register_memory_tools, tool_memory_search_hot


def _create_hot_record(
    content: str,
    session_id: str,
    category: str = "task",
    importance: float = 0.5,
    is_summary: bool = False,
    created_at: str | None = None,
) -> MemoryRecord:
    now = created_at or iso_now()
    return MemoryRecord(
        memory_id=make_memory_id(),
        content=content,
        memory_type="hot",
        category=category,
        source="agent_call",
        session_id=session_id,
        created_at=now,
        updated_at=now,
        importance=importance,
        confidence=0.9,
        access_count=0,
        last_accessed=None,
        is_summary=is_summary,
        parent_memory_ids=[],
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


# ─── 1. Session Isolation Tests ───────────────────────────────────────────────

class TestHotMemorySessionIsolation:
    def test_hot_memory_session_isolation_strict(self, memory_manager):
        """Session A's hot memory must never leak into Session B or Session C."""
        # Session A items
        m_a1 = _create_hot_record("Discussing quarterly marketing budget for Q3", session_id="sess_alpha")
        m_a2 = _create_hot_record("Decided on allocating 40k to paid campaigns", session_id="sess_alpha")
        # Session B items
        m_b1 = _create_hot_record("Debugging Python asyncio deadlock in crawler", session_id="sess_beta")
        m_b2 = _create_hot_record("Decided to switch to threadpool executor", session_id="sess_beta")
        # Session C items
        m_c1 = _create_hot_record("Setting up Docker compose for Postgres 16", session_id="sess_gamma")

        for rec in [m_a1, m_a2, m_b1, m_b2, m_c1]:
            memory_manager.add_hot_memory(rec)

        # Search in Session A
        res_a = memory_manager.search_hot_memory(query="decided on allocation", session_id="sess_alpha")
        assert len(res_a) >= 1
        for hit in res_a:
            assert hit.session_id == "sess_alpha"
            assert "asyncio" not in hit.content
            assert "Postgres" not in hit.content

        # Search in Session B
        res_b = memory_manager.search_hot_memory(query="decided on executor", session_id="sess_beta")
        assert len(res_b) >= 1
        for hit in res_b:
            assert hit.session_id == "sess_beta"
            assert "budget" not in hit.content
            assert "Postgres" not in hit.content

        # Search in Session C
        res_c = memory_manager.search_hot_memory(query="Docker compose", session_id="sess_gamma")
        assert len(res_c) == 1
        assert res_c[0].session_id == "sess_gamma"
        assert res_c[0].memory_id == m_c1.memory_id

        # Search in non-existent Session D
        res_d = memory_manager.search_hot_memory(query="budget or executor", session_id="sess_unknown")
        assert len(res_d) == 0

    def test_search_hot_requires_session_id(self, memory_manager):
        """Omitting or providing empty session_id must fail deterministically."""
        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.search_hot_memory("test query", session_id="")

        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.search_hot_memory("test query", session_id="   ")

        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.search_hot_memory("test query", session_id=None)  # type: ignore[arg-type]


# ─── 2. Default top_k & Bounding Tests ────────────────────────────────────────

class TestHotMemoryTopK:
    def test_default_top_k_from_config(self, memory_manager):
        """Default top_k should match MEMORY_DEFAULT_TOP_K when not specified."""
        session_id = "sess_topk"
        for i in range(10):
            memory_manager.add_hot_memory(
                _create_hot_record(f"Item number {i} regarding deployment pipeline", session_id=session_id)
            )

        # Query with default top_k
        results = memory_manager.search_hot_memory(query="deployment pipeline", session_id=session_id)
        assert len(results) == MEMORY_DEFAULT_TOP_K

        # Explicit top_k=3 overrides default
        results_3 = memory_manager.search_hot_memory(query="deployment pipeline", session_id=session_id, top_k=3)
        assert len(results_3) == 3

        # top_k=None falls back to MEMORY_DEFAULT_TOP_K
        results_none = memory_manager.search_hot_memory(query="deployment pipeline", session_id=session_id, top_k=None)
        assert len(results_none) == MEMORY_DEFAULT_TOP_K

    def test_top_k_zero_or_negative_returns_empty(self, memory_manager):
        """top_k <= 0 should safely return empty list without Chroma errors."""
        session_id = "sess_zero"
        memory_manager.add_hot_memory(_create_hot_record("Sample context", session_id=session_id))

        assert memory_manager.search_hot_memory("Sample", session_id=session_id, top_k=0) == []
        assert memory_manager.search_hot_memory("Sample", session_id=session_id, top_k=-1) == []


# ─── 3. Query-Time Metadata Filtering Tests ───────────────────────────────────

class TestHotMemoryFiltering:
    def test_category_filtering(self, memory_manager):
        session_id = "sess_filter"
        m1 = _create_hot_record("Task: review PR #42", session_id=session_id, category="task")
        m2 = _create_hot_record("Decision: approved PR #42 without changes", session_id=session_id, category="decision")
        m3 = _create_hot_record("Instruction: do not deploy until tomorrow", session_id=session_id, category="instruction")

        for m in [m1, m2, m3]:
            memory_manager.add_hot_memory(m)

        # Filter by category="decision"
        hits = memory_manager.search_hot_memory("PR #42", session_id=session_id, category="decision")
        assert len(hits) == 1
        assert hits[0].memory_id == m2.memory_id
        assert hits[0].category == "decision"

        # Filter by category="task"
        task_hits = memory_manager.search_hot_memory("PR #42", session_id=session_id, category="task")
        assert len(task_hits) == 1
        assert task_hits[0].memory_id == m1.memory_id
        assert task_hits[0].category == "task"

        # Non-matching category
        empty_hits = memory_manager.search_hot_memory("PR #42", session_id=session_id, category="nonexistent")
        assert len(empty_hits) == 0

    def test_is_summary_filtering(self, memory_manager):
        session_id = "sess_summary"
        raw_m = _create_hot_record("Detailed point 1 on database migration", session_id=session_id, is_summary=False)
        sum_m = _create_hot_record("Summary of database migration discussion", session_id=session_id, is_summary=True)

        memory_manager.add_hot_memory(raw_m)
        memory_manager.add_hot_memory(sum_m)

        # Query only summaries
        sum_hits = memory_manager.search_hot_memory("database migration", session_id=session_id, is_summary=True)
        assert len(sum_hits) == 1
        assert sum_hits[0].memory_id == sum_m.memory_id
        assert sum_hits[0].is_summary is True

        # Query only raw (non-summaries)
        raw_hits = memory_manager.search_hot_memory("database migration", session_id=session_id, is_summary=False)
        assert len(raw_hits) == 1
        assert raw_hits[0].memory_id == raw_m.memory_id
        assert raw_hits[0].is_summary is False

    def test_importance_range_filtering(self, memory_manager):
        session_id = "sess_importance"
        low = _create_hot_record("Low importance trivia", session_id=session_id, importance=0.2)
        med = _create_hot_record("Medium importance task note", session_id=session_id, importance=0.5)
        high = _create_hot_record("Critical architecture decision", session_id=session_id, importance=0.95)

        for m in [low, med, high]:
            memory_manager.add_hot_memory(m)

        # Query with min_importance=0.7
        high_hits = memory_manager.search_hot_memory("note or decision", session_id=session_id, min_importance=0.7)
        assert len(high_hits) == 1
        assert high_hits[0].memory_id == high.memory_id

        # Query with max_importance=0.4
        low_hits = memory_manager.search_hot_memory("trivia", session_id=session_id, max_importance=0.4)
        assert len(low_hits) == 1
        assert low_hits[0].memory_id == low.memory_id


# ─── 4. Semantic Ranking Tests ────────────────────────────────────────────────

class DummySemanticEmbeddingService:
    """Embedding service generating distinct vectors for ranking verification."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * 384
            if "PyTorch" in t:
                vec[0] = 1.0
            elif "Italian" in t:
                vec[1] = 1.0
            out.append(vec)
        return out

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * 384
        if "PyTorch" in text or "framework" in text:
            vec[0] = 1.0
        return vec


class TestHotMemorySemanticRanking:
    def test_semantic_ranking_order(self, tmp_memory_root, tmp_chroma_root):
        """Ensure search returns closer semantic matches first with increasing distance."""
        store = MemoryStore(tmp_memory_root)
        index = MemoryChromaStore(tmp_chroma_root, DummySemanticEmbeddingService())
        mgr = MemoryManager(store=store, index=index)

        session_id = "sess_semantic"
        m_close = _create_hot_record("User requested using PyTorch for neural network training", session_id=session_id)
        m_far = _create_hot_record("User ordered lunch from the Italian restaurant down the street", session_id=session_id)

        mgr.add_hot_memory(m_close)
        mgr.add_hot_memory(m_far)

        hits = mgr.search_hot_memory("deep learning framework", session_id=session_id)
        assert len(hits) == 2
        assert hits[0].memory_id == m_close.memory_id
        assert hits[0].distance < hits[1].distance


# ─── 5. Tool Adapter & Gemma Projection Tests ─────────────────────────────────

class TestHotMemoryAdapterAndProjection:
    def test_tool_adapter_validation_and_projection(self, memory_manager):
        session_id = "sess_adapter"
        rec = _create_hot_record("Configured Redis cache with 60s TTL", session_id=session_id, category="task")
        memory_manager.add_hot_memory(rec)

        # Tool adapter call
        adapter_res = tool_memory_search_hot(
            manager=memory_manager,
            query="Redis cache TTL",
            session_id=session_id,
        )
        assert adapter_res["status"] == "success"
        assert adapter_res["session_id"] == session_id
        assert adapter_res["count"] == 1

        # Gemma projection
        projected = map_memory_search_result(adapter_res)
        assert "matches" in projected
        assert projected["count"] == 1
        assert projected["session_id"] == session_id

        hit = projected["matches"][0]
        assert hit["memory_id"] == rec.memory_id
        assert hit["content"] == rec.content
        assert hit["memory_type"] == "hot"
        assert hit["category"] == "task"
        assert 0.0 <= hit["similarity"] <= 1.0
        assert "distance" not in hit
        assert "local_path" not in hit

    def test_tool_adapter_missing_session_id(self, memory_manager):
        res = tool_memory_search_hot(manager=memory_manager, query="test", session_id="")
        assert res["status"] == "error"
        assert "session_id is required" in res["error"]
        assert res["count"] == 0

    def test_end_to_end_orchestrator_dispatch(self, memory_manager, tool_registry):
        from core.run_state import RunState

        session_id = "sess_orch"
        rec = _create_hot_record("User instruction: keep responses under 3 bullet points", session_id=session_id)
        memory_manager.add_hot_memory(rec)

        orch = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_orch", user_text="bullet points")
        telemetry: dict = {}
        trace: list = []

        gemma_plug = orch._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_search_hot",
                "arguments": {"query": "bullet points", "session_id": session_id},
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=0,
        )

        assert gemma_plug["call_index"] == 0
        assert gemma_plug["tool"] == "memory"
        assert gemma_plug["function"] == "memory_search_hot"
        assert gemma_plug["status"] == "success"
        assert gemma_plug["result"]["count"] == 1
        assert gemma_plug["result"]["matches"][0]["memory_id"] == rec.memory_id
        assert telemetry.get("memory_calls") == 1
