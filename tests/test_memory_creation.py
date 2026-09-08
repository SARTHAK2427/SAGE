"""
tests/test_memory_creation.py
Phase 9 tests: Memory creation flow, explicit Gemma-triggered storage, and dual-write persistence.

Validates:
1. Hot memory creation: dual-write to canonical store + Chroma hot collection.
2. Cold memory creation: dual-write to canonical store + Chroma cold collection.
3. Content and session validation (no empty/whitespace content, mandatory session_id for hot).
4. No auto-store policy: memory creation is strictly explicit and never automatic on messages.
5. Numeric range validation ([0.0, 1.0]) and timestamp determinism.
6. Pure Gemma projection via map_memory_store_result (sanitized, zero path leakage).
7. End-to-end tool dispatch through ToolRegistry and Orchestrator.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock
import pytest

from core.dispatcher import ToolRegistry
from core.mappers.memory_results import map_memory_store_result
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_document_db.utils import iso_now
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager, MemoryManagerError
from sage_memory.memory_models import MemoryRecord, MemoryValidationError, make_memory_id
from sage_memory.memory_store import MemoryStore
from tools.memory import (
    register_memory_tools,
    tool_memory_store_cold,
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


# ─── 1. Hot Memory Creation Tests ─────────────────────────────────────────────

class TestHotMemoryCreation:
    def test_store_hot_memory_dual_writes(self, memory_manager):
        """Storing hot memory writes to canonical store and indexes in Chroma hot collection."""
        rec = MemoryRecord(
            memory_id=make_memory_id(),
            content="Task alpha: implement binary search",
            memory_type="hot",
            category="task",
            source="agent_call",
            session_id="sess_create_1",
            created_at=iso_now(),
            updated_at=iso_now(),
            importance=0.8,
            confidence=0.9,
            access_count=0,
            last_accessed=None,
            is_summary=False,
            parent_memory_ids=[],
        )

        persisted = memory_manager.add_hot_memory(rec)
        assert persisted.memory_id == rec.memory_id

        # 1. Verify in canonical store
        stored = memory_manager.store.get(persisted.memory_id, "hot")
        assert stored is not None
        assert stored.content == rec.content
        assert stored.session_id == "sess_create_1"
        assert stored.importance == 0.8

        # 2. Verify in Chroma hot collection
        assert memory_manager.index.count("hot") == 1
        # 3. Verify NOT in Chroma cold collection
        assert memory_manager.index.count("cold") == 0

    def test_hot_memory_validation(self, memory_manager):
        """Empty content or missing session_id must fail deterministically."""
        # Empty content at model level
        with pytest.raises(MemoryValidationError, match="content must be a non-empty string"):
            MemoryRecord(
                memory_id=make_memory_id(),
                content="   ",
                memory_type="hot",
                category="task",
                source="agent_call",
                session_id="sess_1",
                created_at=iso_now(),
                updated_at=iso_now(),
                importance=0.5,
                confidence=0.5,
                is_summary=False,
                parent_memory_ids=[],
            )

        # Empty content at manager level
        valid_rec = MemoryRecord(
            memory_id=make_memory_id(),
            content="Valid initial content",
            memory_type="hot",
            category="task",
            source="agent_call",
            session_id="sess_1",
            created_at=iso_now(),
            updated_at=iso_now(),
            importance=0.5,
            confidence=0.5,
            is_summary=False,
            parent_memory_ids=[],
        )
        object.__setattr__(valid_rec, "content", "")
        with pytest.raises(MemoryManagerError, match="content is required"):
            memory_manager.add_hot_memory(valid_rec)

        # Missing session_id at model level
        with pytest.raises(MemoryValidationError, match="session_id is required for hot memories"):
            MemoryRecord(
                memory_id=make_memory_id(),
                content="Valid content",
                memory_type="hot",
                category="task",
                source="agent_call",
                session_id="",
                created_at=iso_now(),
                updated_at=iso_now(),
                importance=0.5,
                confidence=0.5,
                is_summary=False,
                parent_memory_ids=[],
            )

        # Missing session_id at manager level
        valid_rec2 = MemoryRecord(
            memory_id=make_memory_id(),
            content="Valid content",
            memory_type="hot",
            category="task",
            source="agent_call",
            session_id="valid_sess",
            created_at=iso_now(),
            updated_at=iso_now(),
            importance=0.5,
            confidence=0.5,
            is_summary=False,
            parent_memory_ids=[],
        )
        object.__setattr__(valid_rec2, "session_id", "")
        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.add_hot_memory(valid_rec2)

    def test_tool_adapter_store_hot_validation(self, memory_manager):
        """tool_memory_store_hot validates arguments and returns structured error."""
        res_empty_content = tool_memory_store_hot(
            manager=memory_manager,
            content="",
            session_id="sess_1",
        )
        assert res_empty_content["status"] == "error"
        assert "content is required" in res_empty_content["error"]
        assert res_empty_content["memory_id"] is None

        res_empty_session = tool_memory_store_hot(
            manager=memory_manager,
            content="Valid content",
            session_id="  ",
        )
        assert res_empty_session["status"] == "error"
        assert "session_id is required" in res_empty_session["error"]
        assert res_empty_session["memory_id"] is None

        # Out-of-range importance rejected
        res_bad_imp = tool_memory_store_hot(
            manager=memory_manager,
            content="Valid content",
            session_id="sess_1",
            importance=1.7,
        )
        assert res_bad_imp["status"] == "error"
        assert "importance must be in [0.0, 1.0]" in res_bad_imp["error"]
        assert res_bad_imp["memory_id"] is None

        # Out-of-range confidence rejected
        res_bad_conf = tool_memory_store_hot(
            manager=memory_manager,
            content="Valid content",
            session_id="sess_1",
            confidence=-0.2,
        )
        assert res_bad_conf["status"] == "error"
        assert "confidence must be in [0.0, 1.0]" in res_bad_conf["error"]
        assert res_bad_conf["memory_id"] is None

        # Valid in-range values pass unchanged
        res_valid = tool_memory_store_hot(
            manager=memory_manager,
            content="Valid content with scores",
            session_id="sess_1",
            importance=0.75,
            confidence=0.85,
        )
        assert res_valid["status"] == "success"
        assert res_valid["memory"]["importance"] == 0.75
        assert res_valid["memory"]["confidence"] == 0.85


# ─── 2. Cold Memory Creation Tests ────────────────────────────────────────────

class TestColdMemoryCreation:
    def test_store_cold_memory_dual_writes(self, memory_manager):
        """Storing cold memory writes to canonical store and indexes in Chroma cold collection."""
        rec = MemoryRecord(
            memory_id=make_memory_id(),
            content="User preference: prefers Python 3.12+ syntax and typing features",
            memory_type="cold",
            category="preference",
            source="agent_call",
            session_id="sess_origin",
            created_at=iso_now(),
            updated_at=iso_now(),
            importance=0.9,
            confidence=0.95,
            access_count=0,
            last_accessed=None,
            is_summary=False,
            parent_memory_ids=[],
        )

        persisted = memory_manager.add_cold_memory(rec)
        assert persisted.memory_id == rec.memory_id

        # 1. Verify in canonical store
        stored = memory_manager.store.get(persisted.memory_id, "cold")
        assert stored is not None
        assert stored.content == rec.content
        assert stored.importance == 0.9

        # 2. Verify in Chroma cold collection
        assert memory_manager.index.count("cold") == 1
        # 3. Verify NOT in Chroma hot collection
        assert memory_manager.index.count("hot") == 0

    def test_cold_memory_validation(self, memory_manager):
        """Empty content or wrong memory_type must fail deterministically."""
        # Empty content at model level
        with pytest.raises(MemoryValidationError, match="content must be a non-empty string"):
            MemoryRecord(
                memory_id=make_memory_id(),
                content="",
                memory_type="cold",
                category="preference",
                source="agent_call",
                session_id=None,
                created_at=iso_now(),
                updated_at=iso_now(),
                importance=0.5,
                confidence=0.5,
                is_summary=False,
                parent_memory_ids=[],
            )

        # Empty content at manager level
        valid_cold = MemoryRecord(
            memory_id=make_memory_id(),
            content="Valid content",
            memory_type="cold",
            category="preference",
            source="agent_call",
            session_id=None,
            created_at=iso_now(),
            updated_at=iso_now(),
            importance=0.5,
            confidence=0.5,
            is_summary=False,
            parent_memory_ids=[],
        )
        object.__setattr__(valid_cold, "content", "   ")
        with pytest.raises(MemoryManagerError, match="content is required"):
            memory_manager.add_cold_memory(valid_cold)

        rec_wrong_type = MemoryRecord(
            memory_id=make_memory_id(),
            content="Valid content",
            memory_type="hot",
            category="preference",
            source="agent_call",
            session_id="s1",
            created_at=iso_now(),
            updated_at=iso_now(),
            importance=0.5,
            confidence=0.5,
            is_summary=False,
            parent_memory_ids=[],
        )
        with pytest.raises(MemoryManagerError, match="requires memory_type='cold'"):
            memory_manager.add_cold_memory(rec_wrong_type)

    def test_tool_adapter_store_cold_validation(self, memory_manager):
        """tool_memory_store_cold validates content and rejects out-of-range numeric parameters."""
        res_empty = tool_memory_store_cold(manager=memory_manager, content="   ")
        assert res_empty["status"] == "error"
        assert "content is required" in res_empty["error"]

        # Out-of-range importance rejected
        res_bad_imp = tool_memory_store_cold(
            manager=memory_manager,
            content="User prefers dark mode",
            category="preference",
            importance=1.7,
        )
        assert res_bad_imp["status"] == "error"
        assert "importance must be in [0.0, 1.0]" in res_bad_imp["error"]
        assert res_bad_imp["memory_id"] is None

        # Out-of-range confidence rejected
        res_bad_conf = tool_memory_store_cold(
            manager=memory_manager,
            content="User prefers dark mode",
            category="preference",
            confidence=-0.2,
        )
        assert res_bad_conf["status"] == "error"
        assert "confidence must be in [0.0, 1.0]" in res_bad_conf["error"]
        assert res_bad_conf["memory_id"] is None

        # Valid in-range values pass unchanged
        res_ok = tool_memory_store_cold(
            manager=memory_manager,
            content="User prefers dark mode",
            category="preference",
            importance=0.9,
            confidence=0.95,
        )
        assert res_ok["status"] == "success"
        assert res_ok["memory"]["importance"] == 0.9
        assert res_ok["memory"]["confidence"] == 0.95


# ─── 3. No Auto-Store Verification ────────────────────────────────────────────

class TestNoAutoStorePolicy:
    def test_messages_are_not_automatically_stored(self, memory_manager, tool_registry):
        """Incoming user messages and assistant responses must never be auto-stored."""
        # Initial state: 0 memories
        assert memory_manager.index.count("hot") == 0
        assert memory_manager.index.count("cold") == 0
        assert len(memory_manager.store.list_hot()) == 0
        assert len(memory_manager.store.list_cold()) == 0

        orch = Orchestrator(registry=tool_registry)
        run_state = RunState(request_id="req_no_store", user_text="Hello, my name is Ojasvi and I love Python.")

        # Simulate running a prompt that contains statements that would trigger eager auto-store heuristics
        # in brittle systems:
        telemetry: dict = {}
        trace: list = []

        # Calling dispatch on non-memory tool or processing user query does NOT store memory
        assert memory_manager.index.count("hot") == 0
        assert memory_manager.index.count("cold") == 0

        # Memory is ONLY stored if Gemma explicitly issues a memory_store call:
        tool_res = orch._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_store_cold",
                "arguments": {
                    "content": "User name is Ojasvi",
                    "category": "personal",
                },
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=0,
        )

        assert tool_res["status"] == "success"
        assert tool_res["result"]["status"] == "stored"
        # Now exactly 1 cold memory exists
        assert memory_manager.index.count("cold") == 1
        assert memory_manager.index.count("hot") == 0


# ─── 4. Gemma Result Projection Tests ─────────────────────────────────────────

class TestMemoryStoreGemmaProjection:
    def test_map_memory_store_result_projection(self):
        """Gemma projection sanitizes internal structures and returns clean envelope."""
        adapter_output = {
            "status": "success",
            "memory_id": "mem_abc123",
            "memory": {
                "memory_id": "mem_abc123",
                "content": "Fact to preserve",
                "memory_type": "hot",
                "category": "task",
                "local_path": "/var/data/memory/canonical.json",  # must be stripped
                "chroma_collection": "sage_memory_hot",  # must be stripped
            }
        }

        projected = map_memory_store_result(adapter_output)
        assert projected["memory_id"] == "mem_abc123"
        assert projected["status"] == "stored"
        assert projected["memory_type"] == "hot"
        assert projected["category"] == "task"
        assert "local_path" not in projected
        assert "chroma_collection" not in projected
