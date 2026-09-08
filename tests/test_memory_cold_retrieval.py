"""
tests/test_memory_cold_retrieval.py
Phase 8 tests: Cold memory retrieval, selective search, and cross-session accessibility.

Validates:
1. Selective search only (query is strictly required; no "retrieve all cold memories" path).
2. Cross-session persistence and retrieval across sessions.
3. Strict collection separation and zero cross-fallback between hot and cold collections.
4. Config-driven default top_k (MEMORY_DEFAULT_TOP_K) and bounds.
5. Query-time metadata filtering (category, is_summary, importance, timestamps).
6. Semantic ranking order.
7. Clean Gemma projection via map_memory_search_result (zero path/distance leakage).
8. End-to-end dispatch through ToolRegistry and Orchestrator.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import pytest

from core.dispatcher import ToolRegistry
from core.mappers.memory_results import map_memory_search_result
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_document_db.utils import iso_now
from sage_memory.config import MEMORY_DEFAULT_TOP_K
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager, MemoryManagerError
from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore
from tools.memory import register_memory_tools, tool_memory_search_cold, tool_memory_search_hot


def _create_cold_record(
    content: str,
    category: str = "preference",
    session_id: str | None = None,
    importance: float = 0.5,
    is_summary: bool = False,
    created_at: str | None = None,
) -> MemoryRecord:
    now = created_at or iso_now()
    return MemoryRecord(
        memory_id=make_memory_id(),
        content=content,
        memory_type="cold",
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


def _create_hot_record(
    content: str,
    session_id: str,
    category: str = "task",
) -> MemoryRecord:
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
        importance=0.5,
        confidence=0.9,
        access_count=0,
        last_accessed=None,
        is_summary=False,
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


# ─── 1. Selective Search & Validation Tests ───────────────────────────────────

class TestColdMemorySelectiveSearch:
    def test_query_is_mandatory_for_cold_search(self, memory_manager):
        """No retrieve-all path: query must be provided and non-empty."""
        with pytest.raises(MemoryManagerError, match="query is required"):
            memory_manager.search_cold_memory("")

        with pytest.raises(MemoryManagerError, match="query is required"):
            memory_manager.search_cold_memory("   ")

        with pytest.raises(MemoryManagerError, match="query is required"):
            memory_manager.search_cold_memory(None)  # type: ignore[arg-type]

    def test_tool_adapter_missing_query(self, memory_manager):
        """tool_memory_search_cold returns a structured error when query is empty."""
        res = tool_memory_search_cold(manager=memory_manager, query="")
        assert res["status"] == "error"
        assert "query is required" in res["error"]
        assert res["count"] == 0

        res_spaces = tool_memory_search_cold(manager=memory_manager, query="   ")
        assert res_spaces["status"] == "error"
        assert "query is required" in res_spaces["error"]

    def test_default_top_k_from_config(self, memory_manager):
        """Default top_k should match MEMORY_DEFAULT_TOP_K when not specified."""
        for i in range(12):
            memory_manager.add_cold_memory(
                _create_cold_record(f"Project guideline #{i}: maintain clear docstrings")
            )

        # Default top_k
        hits = memory_manager.search_cold_memory("guideline docstrings")
        assert len(hits) == MEMORY_DEFAULT_TOP_K

        # Explicit top_k=3
        hits_3 = memory_manager.search_cold_memory("guideline docstrings", top_k=3)
        assert len(hits_3) == 3

        # top_k=None falls back to MEMORY_DEFAULT_TOP_K
        hits_none = memory_manager.search_cold_memory("guideline docstrings", top_k=None)
        assert len(hits_none) == MEMORY_DEFAULT_TOP_K

    def test_top_k_zero_or_negative_returns_empty(self, memory_manager):
        """top_k <= 0 should safely return empty list without Chroma errors."""
        memory_manager.add_cold_memory(_create_cold_record("Sample cold memory"))
        assert memory_manager.search_cold_memory("Sample", top_k=0) == []
        assert memory_manager.search_cold_memory("Sample", top_k=-1) == []


# ─── 2. Cross-Session Persistence Tests ────────────────────────────────────────

class TestColdMemoryCrossSessionPersistence:
    def test_cold_memory_retrievable_across_sessions(self, memory_manager):
        """Cold memory stored in Session A is accessible across subsequent sessions."""
        # Stored during Session A
        c1 = _create_cold_record(
            "User prefers concise answers without apologies.",
            category="preference",
            session_id="session_alpha",
        )
        # Stored during Session B
        c2 = _create_cold_record(
            "Project architecture is built with FastAPI and PostgreSQL.",
            category="project",
            session_id="session_beta",
        )
        # Stored without any session
        c3 = _create_cold_record(
            "User name is Ojasvi.",
            category="personal",
            session_id=None,
        )

        for rec in [c1, c2, c3]:
            memory_manager.add_cold_memory(rec)

        # In a completely new Session Gamma, cold search finds all relevant facts
        hits_pref = memory_manager.search_cold_memory("preference for concise answers")
        assert any(h.memory_id == c1.memory_id for h in hits_pref)
        assert any(h.content == c1.content for h in hits_pref)

        hits_proj = memory_manager.search_cold_memory("FastAPI architecture")
        assert any(h.memory_id == c2.memory_id for h in hits_proj)

        hits_name = memory_manager.search_cold_memory("What is user name")
        assert any(h.memory_id == c3.memory_id for h in hits_name)


# ─── 3. Strict Collection Separation & Zero Fallback ──────────────────────────

class TestColdMemorySeparationAndNoFallback:
    def test_cold_search_never_returns_hot_memories(self, memory_manager):
        """Cold search strictly queries sage_memory_cold and never returns hot memories."""
        hot_m = _create_hot_record("Temporary scratch note for session X only", session_id="sess_x")
        memory_manager.add_hot_memory(hot_m)

        # Cold search for the exact hot text must return 0
        hits = memory_manager.search_cold_memory("Temporary scratch note")
        assert len(hits) == 0

    def test_hot_search_never_returns_cold_memories(self, memory_manager):
        """Hot search strictly queries sage_memory_hot and never returns cold memories."""
        cold_m = _create_cold_record("Permanent preference for dark theme UI", category="preference")
        memory_manager.add_cold_memory(cold_m)

        # Hot search in any session must return 0
        hits = memory_manager.search_hot_memory("dark theme UI", session_id="sess_x")
        assert len(hits) == 0

    def test_zero_automatic_fallback(self, memory_manager):
        """When cold search has no matches, it returns empty (no automatic fallback to hot)."""
        hot_m = _create_hot_record("Session-specific API key draft", session_id="sess_1")
        memory_manager.add_hot_memory(hot_m)

        # Cold search does not fall back to hot
        res = memory_manager.search_cold_memory("API key draft")
        assert res == []


# ─── 4. Query-Time Metadata Filtering Tests ───────────────────────────────────

class TestColdMemoryFiltering:
    def test_category_filtering(self, memory_manager):
        c_pref = _create_cold_record("Prefers pytest over unittest", category="preference")
        c_proj = _create_cold_record("Repo hosted on GitHub", category="project")
        c_pers = _create_cold_record("Speaks English and Hindi", category="personal")

        for m in [c_pref, c_proj, c_pers]:
            memory_manager.add_cold_memory(m)

        # Filter category="preference"
        pref_hits = memory_manager.search_cold_memory("testing framework", category="preference")
        assert len(pref_hits) == 1
        assert pref_hits[0].memory_id == c_pref.memory_id
        assert pref_hits[0].category == "preference"

        # Filter category="project"
        proj_hits = memory_manager.search_cold_memory("GitHub", category="project")
        assert len(proj_hits) == 1
        assert proj_hits[0].memory_id == c_proj.memory_id

        # Non-matching category
        empty_hits = memory_manager.search_cold_memory("pytest", category="decision")
        assert len(empty_hits) == 0

    def test_importance_filtering(self, memory_manager):
        low = _create_cold_record("Minor trivia about favorite editor font", importance=0.2)
        high = _create_cold_record("Core architectural design principle", importance=0.9)

        memory_manager.add_cold_memory(low)
        memory_manager.add_cold_memory(high)

        hits = memory_manager.search_cold_memory("design or font", min_importance=0.7)
        assert len(hits) == 1
        assert hits[0].memory_id == high.memory_id


# ─── 5. Semantic Ranking Tests ────────────────────────────────────────────────

class DummySemanticEmbeddingService:
    """Embedding service generating distinct vectors for ranking verification."""
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vec = [0.0] * 384
            if "PostgreSQL" in t:
                vec[0] = 1.0
            elif "gardening" in t:
                vec[1] = 1.0
            out.append(vec)
        return out

    def embed_query(self, text: str) -> list[float]:
        vec = [0.0] * 384
        if "relational database" in text or "PostgreSQL" in text:
            vec[0] = 1.0
        return vec


class TestColdMemorySemanticRanking:
    def test_semantic_ranking_order(self, tmp_memory_root, tmp_chroma_root):
        """Ensure search returns closer semantic matches first with increasing distance."""
        store = MemoryStore(tmp_memory_root)
        index = MemoryChromaStore(tmp_chroma_root, DummySemanticEmbeddingService())
        mgr = MemoryManager(store=store, index=index)

        m_close = _create_cold_record("User preference: always use PostgreSQL for relational data storage")
        m_far = _create_cold_record("User hobby: likes gardening on weekends")

        mgr.add_cold_memory(m_close)
        mgr.add_cold_memory(m_far)

        hits = mgr.search_cold_memory("relational database preference")
        assert len(hits) == 2
        assert hits[0].memory_id == m_close.memory_id
        assert hits[0].distance < hits[1].distance


# ─── 6. Tool Adapter & Gemma Projection Tests ─────────────────────────────────

class TestColdMemoryAdapterAndProjection:
    def test_tool_adapter_and_projection(self, memory_manager):
        rec = _create_cold_record("User prefers dark theme across all applications", category="preference")
        memory_manager.add_cold_memory(rec)

        # Tool adapter call
        adapter_res = tool_memory_search_cold(
            manager=memory_manager,
            query="dark theme preference",
        )
        assert adapter_res["status"] == "success"
        assert adapter_res["count"] == 1

        # Gemma projection
        projected = map_memory_search_result(adapter_res)
        assert "matches" in projected
        assert projected["count"] == 1

        hit = projected["matches"][0]
        assert hit["memory_id"] == rec.memory_id
        assert hit["content"] == rec.content
        assert hit["memory_type"] == "cold"
        assert hit["category"] == "preference"
        assert 0.0 <= hit["similarity"] <= 1.0
        assert "distance" not in hit
        assert "local_path" not in hit

    def test_end_to_end_orchestrator_dispatch(self, memory_manager, tool_registry):
        rec = _create_cold_record("User timezone is Asia/Kolkata", category="personal")
        memory_manager.add_cold_memory(rec)

        orch = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_orch_cold", user_text="what is my timezone")
        telemetry: dict = {}
        trace: list = []

        gemma_plug = orch._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_search_cold",
                "arguments": {"query": "user timezone"},
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=0,
        )

        assert gemma_plug["call_index"] == 0
        assert gemma_plug["tool"] == "memory"
        assert gemma_plug["function"] == "memory_search_cold"
        assert gemma_plug["status"] == "success"
        assert gemma_plug["result"]["count"] == 1
        assert gemma_plug["result"]["matches"][0]["memory_id"] == rec.memory_id
        assert gemma_plug["result"]["matches"][0]["memory_type"] == "cold"
        assert telemetry.get("memory_calls") == 1
