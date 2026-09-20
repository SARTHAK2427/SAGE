"""Phase 8 integration tests for automatic, bounded cold-memory injection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import config
import orchestrator as orchestrator_module
import sage_memory as sage_memory_module
import sage_memory_index
from core.dispatcher import ToolRegistry
from core.run_state import RunState
from core.mappers.gemma_results import CANONICAL_GEMMA_MAPPERS
from orchestrator import Orchestrator
from sage_memory import sage_memory, set_sqlite_path
from sage_memory_index import MemoryVectorIndex
from tools.durable_memory import (
    memory_promote,
    memory_search_cold,
    memory_search_hot,
    memory_store_cold,
    memory_store_hot,
)
from tools.registry import create_default_registry


@pytest.fixture(autouse=True)
def isolated_phase8_memory(monkeypatch):
    """Use a canonical SQLite store plus a matching isolated vector index."""
    temp_dir = tempfile.TemporaryDirectory()
    root = Path(temp_dir.name)
    monkeypatch.setenv("SAGE_MEMORY_DB", "sqlite")
    set_sqlite_path(str(root / "phase8.db"))
    sage_memory._initialized_backends.clear()

    index = MemoryVectorIndex(chroma_root=root / "phase8_chroma")
    original_index = sage_memory_index.memory_vector_index
    original_store = sage_memory_module._sync_index_store
    original_status = sage_memory_module._sync_index_status
    sage_memory_index.memory_vector_index = index
    sage_memory_module._sync_index_store = lambda memory: index.index_memory(memory)
    sage_memory_module._sync_index_status = lambda memory_id, status: index.update_memory_status(memory_id, status)

    yield

    index.close()
    sage_memory_index.memory_vector_index = original_index
    sage_memory_module._sync_index_store = original_store
    sage_memory_module._sync_index_status = original_status
    set_sqlite_path(None)
    temp_dir.cleanup()


def _orchestrator_with_math_only() -> Orchestrator:
    registry = ToolRegistry()
    registry.register("math", "calculate", lambda expression, **_: {"status": "success", "value": 2})
    return Orchestrator(registry=registry)


def _run_and_capture_initial_gemma_payload(monkeypatch, run_state: RunState, objective: str) -> tuple[dict, str]:
    """Run a real non-mock orchestration path while capturing its first Gemma input."""
    monkeypatch.setenv("SAGE_MOCK_MODE", "0")
    orch = _orchestrator_with_math_only()
    agent_inputs = []

    def fake_completion(messages, model_key=None, **_kwargs):
        if model_key == "agent":
            agent_inputs.append(messages)
            if len(agent_inputs) == 1:
                return {
                    "content": json.dumps({
                        "type": "tool_calls",
                        "calls": [{
                            "tool": "math",
                            "function": "calculate",
                            "arguments": {"expression": "1 + 1"},
                        }],
                    }),
                    "duration": 0.01,
                    "usage": {},
                }
            return {"content": json.dumps({"type": "final", "answer": "done"}), "duration": 0.01, "usage": {}}
        return {"content": "done", "duration": 0.01, "usage": {}}

    monkeypatch.setattr(orchestrator_module.model_manager, "ensure_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(orchestrator_module.model_client, "chat_completion", fake_completion)
    result = orch.run(objective, [], {}, run_state=run_state)
    assert agent_inputs, "The test must observe a real payload sent to Gemma."
    return result, agent_inputs[0][1]["content"]


def test_global_memory_from_another_chat_reaches_gemma_automatically(monkeypatch):
    """A cold memory from Chat A is present in Chat B's first Gemma payload."""
    stored = memory_store_cold(
        user_id="phase8_alice", chat_id="chat_A",
        content="Alice is building SAGE as a local-first AI workbench.", category="project", importance=0.95,
    )
    assert stored["status"] == "success"

    result, payload = _run_and_capture_initial_gemma_payload(
        monkeypatch,
        RunState(user_id="phase8_alice", chat_id="chat_B", user_text="What should we build next?"),
        "What should we build next?",
    )

    assert "PERSISTENT MEMORY:" in payload
    assert "local-first AI workbench" in payload
    assert "memory_search" not in json.dumps(result["trace"])
    injected = next(item for item in result["trace"] if item["action"] == "global_memory_injected")
    assert stored["memory"]["memory_id"] in injected["global_memory_ids"]


def test_global_memory_is_strictly_user_isolated(monkeypatch):
    memory_store_cold(user_id="phase8_user_a", content="User A private durable preference.", category="preference")
    memory_store_cold(user_id="phase8_user_b", content="User B secret project name.", category="project")

    _, payload = _run_and_capture_initial_gemma_payload(
        monkeypatch,
        RunState(user_id="phase8_user_a", chat_id="chat_A", user_text="Continue."),
        "Continue.",
    )

    assert "User A private durable preference." in payload
    assert "User B secret project name." not in payload


def test_inactive_cold_memory_is_not_injected(monkeypatch):
    stored = memory_store_cold(
        user_id="phase8_inactive", content="This deleted fact must not reach Gemma.", category="fact"
    )
    sage_memory.delete_memory(stored["memory"]["memory_id"])

    _, payload = _run_and_capture_initial_gemma_payload(
        monkeypatch,
        RunState(user_id="phase8_inactive", chat_id="chat_new", user_text="Hello"), "Hello",
    )
    assert "This deleted fact must not reach Gemma." not in payload


def test_global_retrieval_honors_item_and_token_budgets(monkeypatch):
    monkeypatch.setattr(config, "SAGE_GLOBAL_MEMORY_MAX_ITEMS", 2)
    monkeypatch.setattr(config, "SAGE_GLOBAL_MEMORY_BUDGET_TOKENS", 40)
    monkeypatch.setattr(config, "SAGE_CONTEXT_MEMORY_BUDGET_TOKENS", 80)
    for number in range(5):
        memory_store_cold(
            user_id="phase8_bounded",
            content=f"Bounded durable memory number {number}.", category="fact", importance=1.0 - number / 10,
        )

    result, payload = _run_and_capture_initial_gemma_payload(
        monkeypatch,
        RunState(user_id="phase8_bounded", chat_id="chat_bound", user_text="Use my preferences."), "Use my preferences.",
    )
    injected = next(item for item in result["trace"] if item["action"] == "global_memory_injected")
    assert injected["global_memory_count"] == 2
    assert injected["global_memory_tokens"] <= 40
    assert payload.count("Bounded durable memory number") == 2


def test_complete_current_request_survives_memory_pressure(monkeypatch):
    monkeypatch.setattr(config, "SAGE_GLOBAL_MEMORY_BUDGET_TOKENS", 30)
    monkeypatch.setattr(config, "SAGE_CONTEXT_MEMORY_BUDGET_TOKENS", 30)
    memory_store_cold(
        user_id="phase8_request", content="A durable memory that fills the small memory budget.", category="fact"
    )
    objective = "CURRENT_REQUEST_MARKER " + "important " * 100
    _, payload = _run_and_capture_initial_gemma_payload(
        monkeypatch,
        RunState(user_id="phase8_request", chat_id="chat_request", user_text=objective), objective,
    )
    assert f"Objective:\n{objective}" in payload


def test_global_memory_failure_does_not_break_chat(monkeypatch):
    monkeypatch.setenv("SAGE_MOCK_MODE", "1")
    orch = _orchestrator_with_math_only()
    with patch.object(sage_memory, "list_memories", side_effect=RuntimeError("memory backend unavailable")):
        result = orch.run(
            "Continue without durable memory.", [], {},
            run_state=RunState(user_id="phase8_failure", chat_id="chat_failure"),
        )
    assert result["status"] == "success"
    assert any(item["action"] == "global_memory_retrieval_failed" for item in result["trace"])


def test_existing_memory_operations_and_gemma_wiring_remain_available():
    """Phase 8 adds injection without replacing Phase 1-7 operations or tools."""
    hot = memory_store_hot(user_id="phase8_tools", chat_id="chat_tools", content="Current task state.", category="project")
    cold = memory_store_cold(user_id="phase8_tools", content="Durable project preference.", category="preference")
    assert memory_search_hot(user_id="phase8_tools", chat_id="chat_tools", query="current task")["returned"] >= 1
    assert memory_search_cold(user_id="phase8_tools", query="durable preference")["returned"] >= 1
    assert memory_promote(user_id="phase8_tools", memory_id=hot["memory"]["memory_id"])["status"] == "success"

    # Mutable semantic-memory lifecycle remains canonical backend behavior.
    updated = sage_memory.update_memory(cold["memory"]["memory_id"], content="Updated durable preference.")
    assert updated and updated["content"] == "Updated durable preference."
    assert sage_memory.delete_memory(cold["memory"]["memory_id"])

    registry = create_default_registry()
    for function_name in (
        "memory_search_hot", "memory_search_cold", "memory_store_hot",
        "memory_store_cold", "memory_promote", "memory_summarize",
    ):
        assert registry.is_registered("durable_memory", function_name)
        assert ("durable_memory", function_name) in CANONICAL_GEMMA_MAPPERS

    tools_definition = (config.PROMPTS_DIR / "tools.json").read_text(encoding="utf-8")
    abilities_definition = (config.PROMPTS_DIR / "abilities.json").read_text(encoding="utf-8")
    for function_name in ("memory_search_hot", "memory_search_cold", "memory_store_hot", "memory_store_cold", "memory_promote"):
        assert function_name in tools_definition
        assert function_name in abilities_definition
