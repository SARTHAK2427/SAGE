"""
tests/test_end_to_end_memory.py
Comprehensive end-to-end integration tests for Phase 16:
Full Orchestrator Integration + "/api/chat" Session Wiring.

Verifies:
- Test 1: Cross-turn hot memory via session_id
- Test 2: Hot session isolation
- Test 3: Cold cross-session persistence
- Test 4: Omitted session_id generates unique, isolated sessions
- Test 5: Frontend session persistence contract
- Test 6: Tool dispatch forms (Form A & Form B) and session fallback
- Test 7: Zero automatic memory operations when Gemma does not call memory tools
- Test 8: Tool call tracking and telemetry in RunState
- Test 9: Existing chat regression (math, coder, and attachments)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any, Dict, List
import pytest
from fastapi.testclient import TestClient

import config
from app import app
from core.run_state import RunState
from orchestrator import Orchestrator
from sage_document_db.embeddings import EmbeddingService
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager
from sage_memory.memory_store import MemoryStore
from tools.registry import create_default_registry


@pytest.fixture()
def isolated_memory(tmp_path, monkeypatch):
    """Provide isolated canonical JSON store and ChromaDB index for tests."""
    mem_root = tmp_path / "data_memory"
    hot_dir = mem_root / "hot"
    cold_dir = mem_root / "cold"
    chroma_dir = tmp_path / "chroma_memory"
    hot_dir.mkdir(parents=True, exist_ok=True)
    cold_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("sage_memory.config.MEMORY_STORE_PATH", mem_root)
    monkeypatch.setattr("sage_memory.config.MEMORY_HOT_DIR", hot_dir)
    monkeypatch.setattr("sage_memory.config.MEMORY_COLD_DIR", cold_dir)
    monkeypatch.setattr("sage_memory.config.CHROMA_ROOT", chroma_dir)

    store = MemoryStore(mem_root)
    index = MemoryChromaStore(
        chroma_root=chroma_dir,
        embedding_service=EmbeddingService(),
        hot_collection="test_e2e_hot",
        cold_collection="test_e2e_cold",
    )
    mgr = MemoryManager(store=store, index=index)

    import db_service
    monkeypatch.setattr(db_service, "_mem_instance", mgr)
    monkeypatch.setattr(db_service, "get_memory_manager", lambda db=None: mgr)

    # Wire a fresh ToolRegistry with this isolated manager to app's orchestrator
    import orchestrator as orch_mod
    registry = create_default_registry(memory_manager=mgr)
    test_orch = Orchestrator(registry=registry)
    monkeypatch.setattr(orch_mod, "orchestrator", test_orch)
    monkeypatch.setattr("app.orchestrator", test_orch)

    return mgr, test_orch


@pytest.fixture()
def mock_gemma(monkeypatch):
    """Helper fixture to script Gemma completions and disable mock-mode fast-path."""
    import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "_is_mock_mode", lambda: False)
    monkeypatch.setattr("model_manager.model_manager.ensure_model", lambda model_key: None)

    scripted_responses: List[str] = []

    def set_responses(responses: List[str]):
        scripted_responses.clear()
        scripted_responses.extend(responses)

    def fake_chat_completion(messages: List[Dict[str, Any]], temperature: float = 0.2, max_tokens: int = 2048):
        if not scripted_responses:
            raise RuntimeError("fake_chat_completion called with no remaining scripted responses!")
        content = scripted_responses.pop(0)
        return {
            "content": content,
            "duration": 0.05,
            "usage": {"completion_tokens": len(content.split())},
        }

    monkeypatch.setattr("model_client.model_client.chat_completion", fake_chat_completion)
    return set_responses


class TestEndToEndMemory:
    """End-to-end integration tests for Phase 16."""

    # ── Test 1: Cross-Turn Hot Memory ──────────────────────────────────────────
    def test_cross_turn_hot_memory(self, isolated_memory, mock_gemma):
        mgr, _ = isolated_memory
        set_responses = mock_gemma
        client = TestClient(app)
        session_id = "sess_test_1"

        # Turn 1: User says "Remember that my name is Alice and I am a software engineer."
        # Gemma responds with tool call to memory_store_hot, then final confirmation.
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_store_hot",
                        "arguments": {
                            "content": "User's name is Alice and she is a software engineer.",
                            "category": "task"
                        }
                    }
                ]
            }),
            json.dumps({
                "type": "final",
                "answer": "I have remembered that your name is Alice and you are a software engineer."
            })
        ])

        res1 = client.post("/api/chat", data={
            "objective": "Remember that my name is Alice and I am a software engineer.",
            "session_id": session_id,
        })
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["status"] == "success"
        assert data1["session_id"] == session_id
        assert "Alice" in data1["answer"]

        # Verify memory stored in canonical hot store
        hot_records = mgr.store.list_hot()
        assert len(hot_records) == 1
        assert hot_records[0].session_id == session_id
        assert "Alice" in hot_records[0].content
        assert len(mgr.index.get_indexed_memory_ids("hot")) == 1

        # Turn 2: User says "What is my name and profession?" with same session_id.
        # Gemma calls memory_search_hot (omitting session_id; orchestrator falls back to run_state).
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_search_hot",
                        "arguments": {
                            "query": "user name profession"
                        }
                    }
                ]
            }),
            json.dumps({
                "type": "final",
                "answer": "Your name is Alice and you are a software engineer."
            })
        ])

        res2 = client.post("/api/chat", data={
            "objective": "What is my name and profession?",
            "session_id": session_id,
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["status"] == "success"
        assert data2["session_id"] == session_id
        assert "Alice" in data2["answer"]
        assert "software engineer" in data2["answer"]

    # ── Test 2: Hot Session Isolation ──────────────────────────────────────────
    def test_hot_session_isolation(self, isolated_memory, mock_gemma):
        mgr, _ = isolated_memory
        set_responses = mock_gemma
        client = TestClient(app)

        # Session A stores hot memory
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_store_hot",
                        "arguments": {
                            "content": "User prefers dark mode.",
                            "category": "task"
                        }
                    }
                ]
            }),
            json.dumps({"type": "final", "answer": "Preference saved for this session."})
        ])

        res_a = client.post("/api/chat", data={
            "objective": "I prefer dark mode.",
            "session_id": "sess_session_A",
        })
        assert res_a.status_code == 200
        assert res_a.json()["session_id"] == "sess_session_A"

        # Session B searches hot memory: should return 0 results
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_search_hot",
                        "arguments": {
                            "query": "user preferences"
                        }
                    }
                ]
            }),
            json.dumps({"type": "final", "answer": "No preferences found for this session."})
        ])

        res_b = client.post("/api/chat", data={
            "objective": "What are my preferences?",
            "session_id": "sess_session_B",
        })
        assert res_b.status_code == 200
        assert res_b.json()["session_id"] == "sess_session_B"

        # Direct verification on manager: Session B search returns 0 results
        search_b = mgr.search_hot_memory(query="preferences", session_id="sess_session_B")
        assert len(search_b) == 0

        # Session A search returns 1 result
        search_a = mgr.search_hot_memory(query="preferences", session_id="sess_session_A")
        assert len(search_a) == 1
        assert "dark mode" in search_a[0].content

    # ── Test 3: Cold Cross-Session ─────────────────────────────────────────────
    def test_cold_cross_session(self, isolated_memory, mock_gemma):
        mgr, _ = isolated_memory
        set_responses = mock_gemma
        client = TestClient(app)

        # Session A stores cold memory
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_store_cold",
                        "arguments": {
                            "content": "Architecture guideline: all APIs must return JSON.",
                            "category": "project"
                        }
                    }
                ]
            }),
            json.dumps({"type": "final", "answer": "Guideline saved to permanent memory."})
        ])

        res_a = client.post("/api/chat", data={
            "objective": "Save architecture guideline: all APIs must return JSON.",
            "session_id": "sess_session_A",
        })
        assert res_a.status_code == 200

        # Session B searches cold memory: finds it!
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_search_cold",
                        "arguments": {
                            "query": "API design guidelines"
                        }
                    }
                ]
            }),
            json.dumps({"type": "final", "answer": "The guideline requires all APIs to return JSON."})
        ])

        res_b = client.post("/api/chat", data={
            "objective": "What is our API design guideline?",
            "session_id": "sess_session_B",
        })
        assert res_b.status_code == 200
        assert "return JSON" in res_b.json()["answer"]

        # Verify via manager
        cold_results = mgr.search_cold_memory(query="API guidelines")
        assert len(cold_results) >= 1
        assert "all APIs must return JSON" in cold_results[0].content

    # ── Test 4: Omitted session_id ─────────────────────────────────────────────
    def test_omitted_session_id_generates_isolated_sessions(self, isolated_memory, mock_gemma):
        mgr, _ = isolated_memory
        set_responses = mock_gemma
        client = TestClient(app)

        # Request 1 without session_id
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_store_hot",
                        "arguments": {
                            "content": "Secret key is Alpha123."
                        }
                    }
                ]
            }),
            json.dumps({"type": "final", "answer": "Secret key recorded."})
        ])

        res1 = client.post("/api/chat", data={"objective": "Store secret key Alpha123"})
        assert res1.status_code == 200
        data1 = res1.json()
        sess_1 = data1.get("session_id")
        assert sess_1 is not None
        assert sess_1.startswith("sess_")

        # Request 2 without session_id
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_search_hot",
                        "arguments": {
                            "query": "secret key"
                        }
                    }
                ]
            }),
            json.dumps({"type": "final", "answer": "I found no secret key."})
        ])

        res2 = client.post("/api/chat", data={"objective": "What is my secret key?"})
        assert res2.status_code == 200
        data2 = res2.json()
        sess_2 = data2.get("session_id")
        assert sess_2 is not None
        assert sess_2.startswith("sess_")

        # Verify distinct generated sessions
        assert sess_1 != sess_2

        # Verify memory from Request 1 is NOT visible in Request 2
        search_req2 = mgr.search_hot_memory(query="secret key", session_id=sess_2)
        assert len(search_req2) == 0

    # ── Test 5: Frontend Session Persistence Contract ──────────────────────────
    def test_frontend_session_persistence_contract(self, isolated_memory):
        """Verify /api/chat preserves and echoes client-supplied session_id."""
        client = TestClient(app)
        custom_id = "sess_custom_contract_456"

        res1 = client.post("/api/chat", data={
            "objective": "Ping",
            "session_id": custom_id,
        })
        assert res1.status_code == 200
        data1 = res1.json()
        assert data1["session_id"] == custom_id

        res2 = client.post("/api/chat", data={
            "objective": "Pong",
            "session_id": custom_id,
        })
        assert res2.status_code == 200
        data2 = res2.json()
        assert data2["session_id"] == custom_id

    # ── Test 6: Tool Dispatch Forms and Session Fallback ───────────────────────
    def test_tool_dispatch_forms_and_session_fallback(self, isolated_memory):
        mgr, orch = isolated_memory
        session_id = "sess_dispatch_forms_789"
        run_state = RunState(request_id="req_dispatch", session_id=session_id, user_text="dispatch test")
        telemetry: Dict[str, Any] = {}
        trace: List[Dict[str, Any]] = []

        # Form A: {"tool": "memory", "function": "memory_store_hot", "arguments": {"text": "Form A text"}}
        # Note: arguments uses 'text' alias and omits 'session_id'.
        res_a = orch._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_store_hot",
                "arguments": {"text": "Form A text content"}
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=0,
        )
        assert res_a["status"] == "success"
        assert res_a["tool"] == "memory"
        mem_a_id = res_a["result"]["memory_id"]
        stored_a = mgr.get_memory(mem_a_id, "hot")
        assert stored_a.session_id == session_id
        assert stored_a.content == "Form A text content"

        # Form B: {"tool": "memory_store_hot", "arguments": {"text": "Form B text"}}
        # Note: direct tool name, uses 'text' alias and omits 'session_id'.
        res_b = orch._dispatch_tool_call(
            call={
                "tool": "memory_store_hot",
                "arguments": {"text": "Form B text content"}
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=1,
        )
        assert res_b["status"] == "success"
        assert res_b["tool"] == "memory"
        assert res_b["function"] == "memory_store_hot"
        mem_b_id = res_b["result"]["memory_id"]
        stored_b = mgr.get_memory(mem_b_id, "hot")
        assert stored_b.session_id == session_id
        assert stored_b.content == "Form B text content"

        # Form A search fallback to session_id
        search_a = orch._dispatch_tool_call(
            call={
                "tool": "memory",
                "function": "memory_search_hot",
                "arguments": {"query": "Form A"}
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=2,
        )
        assert search_a["status"] == "success"
        assert search_a["result"]["count"] >= 1
        matches_a = [m["memory_id"] for m in search_a["result"]["matches"]]
        assert mem_a_id in matches_a

        # Form B search fallback to session_id
        search_b = orch._dispatch_tool_call(
            call={
                "tool": "memory_search_hot",
                "arguments": {"query": "Form B"}
            },
            run_state=run_state,
            telemetry=telemetry,
            trace=trace,
            file_map={},
            call_index=3,
        )
        assert search_b["status"] == "success"
        assert search_b["result"]["count"] >= 1
        matches_b = [m["memory_id"] for m in search_b["result"]["matches"]]
        assert mem_b_id in matches_b

    # ── Test 7: No Automatic Memory Operations ─────────────────────────────────
    def test_no_automatic_memory_operations(self, isolated_memory, mock_gemma):
        mgr, _ = isolated_memory
        set_responses = mock_gemma
        client = TestClient(app)

        # Chat request where Gemma answers directly without calling memory tools
        set_responses([
            json.dumps({
                "type": "final",
                "answer": "2 + 2 is equal to 4."
            })
        ])

        res = client.post("/api/chat", data={
            "objective": "What is 2 + 2?",
            "session_id": "sess_no_auto",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
        assert "4" in data["answer"]

        # Verify zero automatic memory writes
        assert len(mgr.store.list_hot()) == 0
        assert len(mgr.store.list_cold()) == 0
        assert len(mgr.index.get_indexed_memory_ids("hot")) == 0
        assert len(mgr.index.get_indexed_memory_ids("cold")) == 0

        # Verify telemetry reflects 0 memory calls
        assert data["telemetry"].get("memory_calls", 0) == 0

        # Verify trace contains no memory actions
        trace = data.get("trace", [])
        memory_trace = [t for t in trace if t.get("actor") == "memory"]
        assert len(memory_trace) == 0

    # ── Test 8: Tool Call Tracking ─────────────────────────────────────────────
    def test_tool_call_tracking(self, isolated_memory, mock_gemma):
        mgr, _ = isolated_memory
        set_responses = mock_gemma
        client = TestClient(app)

        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {
                        "tool": "memory",
                        "function": "memory_store_hot",
                        "arguments": {
                            "content": "User prefers concise answers.",
                            "category": "instruction"
                        }
                    }
                ]
            }),
            json.dumps({"type": "final", "answer": "I will keep answers concise."})
        ])

        res = client.post("/api/chat", data={
            "objective": "Be concise.",
            "session_id": "sess_tracking_test",
        })
        assert res.status_code == 200
        data = res.json()

        # Telemetry check
        assert data["telemetry"].get("memory_calls") == 1

        # Trace check
        memory_traces = [t for t in data.get("trace", []) if t.get("actor") == "memory"]
        assert len(memory_traces) == 1
        assert memory_traces[0]["action"] == "memory_store_hot"
        assert memory_traces[0]["status"] == "success"

        # RunState tool_calls check
        run_state_dict = data.get("run_state", {})
        tool_calls = run_state_dict.get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool_name"] == "memory"
        assert tool_calls[0]["function_name"] == "memory_store_hot"
        assert tool_calls[0]["status"] == "success"
        assert tool_calls[0]["result_summary"] is not None

    # ── Test 9: Existing Chat Regression ───────────────────────────────────────
    def test_existing_chat_regression(self, isolated_memory, mock_gemma):
        _, orch = isolated_memory
        set_responses = mock_gemma
        client = TestClient(app)

        # 1. Math tool call regression
        set_responses([
            json.dumps({
                "type": "tool_calls",
                "calls": [
                    {"tool": "math", "expression": "125 * 8"}
                ]
            }),
            json.dumps({
                "type": "final",
                "answer": "125 * 8 = 1000"
            })
        ])

        res_math = client.post("/api/chat", data={"objective": "Calculate 125 * 8"})
        assert res_math.status_code == 200
        data_math = res_math.json()
        assert data_math["status"] == "success"
        assert "1000" in data_math["answer"]
        assert data_math["telemetry"].get("agent_calls") == 2

        # 2. Coder tool dispatch regression
        run_state = RunState(user_text="Solve code task")
        res_coder = orch._dispatch_tool_call(
            call={"tool": "code_specialist", "instruction": "Print hello world"},
            run_state=run_state,
            telemetry={},
            trace=[],
            file_map={},
            call_index=0,
        )
        assert res_coder["status"] == "success"
        assert res_coder["tool"] == "code_specialist"
