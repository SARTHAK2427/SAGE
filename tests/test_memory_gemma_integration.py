"""
Tests for Phase 6: Gemma tool integration, memory result mappers, and orchestrator dispatch.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.dispatcher import ToolRegistry
from core.mappers import (
    map_memory_delete_result,
    map_memory_promote_result,
    map_memory_search_result,
    map_memory_store_result,
    map_memory_summarize_result,
    map_memory_update_result,
    map_tool_result_for_gemma,
)
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager
from sage_memory.memory_store import MemoryStore
from tools.memory import register_memory_tools
from tools.registry import create_default_registry


@pytest.fixture()
def memory_manager(tmp_memory_root, tmp_chroma_root):
    store = MemoryStore(tmp_memory_root)
    index = MemoryChromaStore(tmp_chroma_root, EmbeddingService())
    return MemoryManager(store=store, index=index)


@pytest.fixture()
def orchestrator(memory_manager):
    registry = create_default_registry(memory_manager=memory_manager)
    return Orchestrator(registry=registry)


# ─── 1. Protocol File Schema Validation ───────────────────────────────────────

class TestProtocolFiles:
    def test_tools_json_has_memory_tool(self):
        tools_path = Path("prompts/tools.json")
        with open(tools_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tool_names = [t["name"] for t in data["tools"]]
        assert "memory" in tool_names

        mem_tool = next(t for t in data["tools"] if t["name"] == "memory")
        fn_names = [fn["name"] for fn in mem_tool["functions"]]
        expected_fns = [
            "memory_search_hot",
            "memory_search_cold",
            "memory_store_hot",
            "memory_store_cold",
            "memory_update",
            "memory_delete",
            "memory_summarize",
            "memory_promote",
        ]
        for fn in expected_fns:
            assert fn in fn_names

    def test_abilities_json_has_memory_delegations(self):
        abilities_path = Path("prompts/abilities.json")
        with open(abilities_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        delegated = data.get("delegated", [])
        mem_delegations = [d for d in delegated if d.get("tool") == "memory"]
        assert len(mem_delegations) >= 8

        mem_fns = {d["function"] for d in mem_delegations}
        expected_fns = {
            "memory_search_hot",
            "memory_search_cold",
            "memory_store_hot",
            "memory_store_cold",
            "memory_update",
            "memory_delete",
            "memory_summarize",
            "memory_promote",
        }
        assert expected_fns.issubset(mem_fns)


# ─── 2. Memory Mapper Projections ─────────────────────────────────────────────

class TestMemoryMappers:
    def test_map_memory_search_result(self):
        data = {
            "status": "success",
            "session_id": "sess_abc",
            "count": 1,
            "results": [
                {
                    "memory_id": "mem_001",
                    "content": "User prefers pytest.",
                    "memory_type": "hot",
                    "category": "preference",
                    "distance": 0.15,
                    "derived_similarity": 0.85,
                    "importance": 0.9,
                    "created_at": "2026-09-08T10:00:00+00:00",
                    "is_summary": False,
                    "local_path": "/internal/data/memory/mem_001.json",  # should be stripped
                }
            ]
        }
        mapped = map_memory_search_result(data)
        assert mapped["count"] == 1
        assert mapped["session_id"] == "sess_abc"
        assert len(mapped["matches"]) == 1
        match = mapped["matches"][0]
        assert match["memory_id"] == "mem_001"
        assert match["similarity"] == 0.85
        assert match["importance"] == 0.9
        assert "local_path" not in match

    def test_map_memory_store_result(self):
        data = {
            "status": "success",
            "memory_id": "mem_123",
            "memory": {
                "memory_id": "mem_123",
                "memory_type": "cold",
                "category": "project",
                "content": "SAGE uses ChromaDB.",
            }
        }
        mapped = map_memory_store_result(data)
        assert mapped["memory_id"] == "mem_123"
        assert mapped["status"] == "stored"
        assert mapped["memory_type"] == "cold"
        assert mapped["category"] == "project"

    def test_map_memory_update_result(self):
        data = {
            "status": "success",
            "memory_id": "mem_123",
            "memory": {"memory_id": "mem_123", "memory_type": "hot"}
        }
        mapped = map_memory_update_result(data)
        assert mapped["memory_id"] == "mem_123"
        assert mapped["status"] == "updated"

    def test_map_memory_delete_result(self):
        data = {"status": "success", "memory_id": "mem_123", "deleted": True}
        mapped = map_memory_delete_result(data)
        assert mapped["memory_id"] == "mem_123"
        assert mapped["deleted"] is True

    def test_map_memory_summarize_result(self):
        data = {
            "status": "success",
            "memory_id": "mem_sum_1",
            "memory": {
                "memory_id": "mem_sum_1",
                "parent_memory_ids": ["mem_1", "mem_2"],
            }
        }
        mapped = map_memory_summarize_result(data)
        assert mapped["memory_id"] == "mem_sum_1"
        assert mapped["status"] == "summarized"
        assert mapped["is_summary"] is True
        assert mapped["parent_memory_ids"] == ["mem_1", "mem_2"]

    def test_map_memory_promote_result(self):
        data = {
            "status": "success",
            "memory_id": "mem_prom",
            "promoted": True,
        }
        mapped = map_memory_promote_result(data)
        assert mapped["memory_id"] == "mem_prom"
        assert mapped["promoted"] is True
        assert mapped["memory_type"] == "cold"

    def test_outer_envelope_wrapping(self):
        rich = {
            "status": "success",
            "memory_id": "mem_test",
            "memory": {"memory_id": "mem_test", "memory_type": "hot", "category": "task"}
        }
        envelope = map_tool_result_for_gemma(
            tool="memory",
            function="memory_store_hot",
            rich_socket=rich,
            call_index=0,
            mapper_fn=map_memory_store_result,
        )
        assert envelope["call_index"] == 0
        assert envelope["tool"] == "memory"
        assert envelope["function"] == "memory_store_hot"
        assert envelope["status"] == "success"
        assert envelope["result"]["memory_id"] == "mem_test"


# ─── 3. Orchestrator Dispatch Integration ──────────────────────────────────────

class TestOrchestratorMemoryDispatch:
    def test_dispatch_memory_store_hot(self, orchestrator):
        run_state = RunState(request_id="req_1", user_text="test")
        telemetry = {}
        trace = []

        call = {
            "tool": "memory",
            "function": "memory_store_hot",
            "arguments": {
                "content": "Keep this task context.",
                "session_id": "sess_orch_1",
                "category": "task",
            }
        }

        resp = orchestrator._dispatch_tool_call(
            call=call,
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=0,
        )

        assert resp["status"] == "success"
        assert resp["tool"] == "memory"
        assert resp["function"] == "memory_search_hot" or resp["function"] == "memory_store_hot"
        assert resp["result"]["status"] == "stored"
        assert telemetry.get("memory_calls") == 1
        assert len(trace) == 1
        assert trace[0]["actor"] == "memory"

    def test_dispatch_memory_search_hot(self, orchestrator):
        run_state = RunState(request_id="req_2", user_text="test")
        telemetry = {}
        trace = []

        # 1. Store
        orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_store_hot",
                "arguments": {
                    "content": "Task X requires algorithm Y.",
                    "session_id": "sess_orch_2",
                }
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=0,
        )

        # 2. Search
        search_call = {
            "tool": "memory",
            "function": "memory_search_hot",
            "arguments": {
                "query": "algorithm Y",
                "session_id": "sess_orch_2",
            }
        }
        resp = orchestrator._dispatch_tool_call(
            call=search_call,
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=1,
        )

        assert resp["status"] == "success"
        assert resp["tool"] == "memory"
        assert resp["result"]["count"] == 1
        assert "algorithm Y" in resp["result"]["matches"][0]["content"]

    def test_dispatch_promote_and_search_cold(self, orchestrator):
        run_state = RunState(request_id="req_3", user_text="test")
        telemetry = {}
        trace = []

        # Store hot
        store_resp = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_store_hot",
                "arguments": {
                    "content": "Permanent fact: architecture is microservices.",
                    "session_id": "sess_orch_3",
                    "category": "project",
                }
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=0,
        )
        mid = store_resp["result"]["memory_id"]

        # Promote
        prom_resp = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_promote",
                "arguments": {"memory_id": mid}
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=1,
        )
        assert prom_resp["status"] == "success"
        assert prom_resp["result"]["promoted"] is True
        assert prom_resp["result"]["memory_type"] == "cold"

        # Search cold
        search_resp = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_search_cold",
                "arguments": {"query": "microservices architecture"}
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=2,
        )
        assert search_resp["status"] == "success"
        assert search_resp["result"]["count"] == 1
        assert search_resp["result"]["matches"][0]["memory_id"] == mid

    def test_dispatch_unmapped_memory_function_fails_closed(self, orchestrator):
        run_state = RunState(request_id="req_4", user_text="test")
        resp = orchestrator._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "nonexistent_memory_fn",
                "arguments": {}
            },
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )
        assert resp["status"] == "error"
        assert resp["error"]["code"] == "UNMAPPED_TOOL_RESULT"
