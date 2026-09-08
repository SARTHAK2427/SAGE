"""
Tests for tools/memory.py — Phase 5 Memory Tool Adapter.
"""

from __future__ import annotations

import pytest

from core.dispatcher import ToolRegistry, ToolResult
from sage_document_db.embeddings import EmbeddingService
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager
from sage_memory.memory_store import MemoryStore
from tools.memory import (
    register_memory_tools,
    tool_memory_delete,
    tool_memory_promote,
    tool_memory_search_cold,
    tool_memory_search_hot,
    tool_memory_store_cold,
    tool_memory_store_hot,
    tool_memory_summarize,
    tool_memory_update,
)
from tools.registry import create_default_registry


@pytest.fixture()
def memory_manager(tmp_memory_root, tmp_chroma_root):
    store = MemoryStore(tmp_memory_root)
    index = MemoryChromaStore(tmp_chroma_root, EmbeddingService())
    return MemoryManager(store=store, index=index)


@pytest.fixture()
def memory_registry(memory_manager):
    registry = ToolRegistry()
    register_memory_tools(registry, memory_manager)
    return registry


class TestMemoryToolDirectAdapters:
    def test_store_and_search_hot(self, memory_manager):
        # Store hot
        res = tool_memory_store_hot(
            manager=memory_manager,
            content="Task instruction alpha",
            session_id="sess_1",
            category="task",
            importance=0.8,
        )
        assert res["status"] == "success"
        assert res["memory_id"] is not None
        assert res["memory"]["content"] == "Task instruction alpha"
        assert res["memory"]["session_id"] == "sess_1"

        # Search hot
        search_res = tool_memory_search_hot(
            manager=memory_manager,
            query="instruction alpha",
            session_id="sess_1",
            top_k=5,
        )
        assert search_res["status"] == "success"
        assert search_res["count"] == 1
        assert search_res["results"][0]["content"] == "Task instruction alpha"

    def test_store_and_search_cold(self, memory_manager):
        # Store cold
        res = tool_memory_store_cold(
            manager=memory_manager,
            content="User prefers Python for data tasks",
            category="preference",
            importance=0.9,
        )
        assert res["status"] == "success"
        assert res["memory_id"] is not None
        assert res["memory"]["category"] == "preference"

        # Search cold
        search_res = tool_memory_search_cold(
            manager=memory_manager,
            query="Python preference",
            top_k=5,
        )
        assert search_res["status"] == "success"
        assert search_res["count"] == 1
        assert search_res["results"][0]["content"] == "User prefers Python for data tasks"

    def test_update_memory(self, memory_manager):
        store_res = tool_memory_store_hot(
            manager=memory_manager,
            content="Draft note",
            session_id="sess_1",
        )
        mid = store_res["memory_id"]

        update_res = tool_memory_update(
            manager=memory_manager,
            memory_id=mid,
            memory_type="hot",
            content="Final note approved",
            importance=0.7,
        )
        assert update_res["status"] == "success"
        assert update_res["memory"]["content"] == "Final note approved"
        assert update_res["memory"]["importance"] == 0.7

    def test_update_nonexistent_returns_error_dict(self, memory_manager):
        res = tool_memory_update(
            manager=memory_manager,
            memory_id="mem_nonexistent",
            memory_type="hot",
            content="Update",
        )
        assert res["status"] == "error"
        assert "not found" in res["error"]

    def test_delete_memory(self, memory_manager):
        store_res = tool_memory_store_cold(
            manager=memory_manager,
            content="To be deleted",
        )
        mid = store_res["memory_id"]

        del_res = tool_memory_delete(
            manager=memory_manager,
            memory_id=mid,
            memory_type="cold",
        )
        assert del_res["status"] == "success"
        assert del_res["deleted"] is True

        # Verify not found after delete
        search_res = tool_memory_search_cold(memory_manager, "To be deleted")
        assert search_res["count"] == 0

    def test_summarize_memory(self, memory_manager):
        r1 = tool_memory_store_hot(memory_manager, "Step 1 done", session_id="sess_1")
        r2 = tool_memory_store_hot(memory_manager, "Step 2 done", session_id="sess_1")

        sum_res = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_1",
            summary_content="Steps 1 and 2 completed",
            parent_memory_ids=[r1["memory_id"], r2["memory_id"]],
        )
        assert sum_res["status"] == "success"
        assert sum_res["memory"]["is_summary"] is True
        assert sum_res["memory"]["parent_memory_ids"] == [r1["memory_id"], r2["memory_id"]]

    def test_promote_memory(self, memory_manager):
        store_res = tool_memory_store_hot(
            manager=memory_manager,
            content="Important decision to keep forever",
            session_id="sess_1",
            category="decision",
        )
        mid = store_res["memory_id"]

        prom_res = tool_memory_promote(
            manager=memory_manager,
            memory_id=mid,
        )
        assert prom_res["status"] == "success"
        assert prom_res["promoted"] is True
        assert prom_res["memory"]["memory_type"] == "cold"

        # Not found in hot
        hot_search = tool_memory_search_hot(memory_manager, "Important decision", session_id="sess_1")
        assert hot_search["count"] == 0

        # Found in cold
        cold_search = tool_memory_search_cold(memory_manager, "Important decision")
        assert cold_search["count"] == 1

    def test_no_filesystem_paths_leak(self, memory_manager):
        res = tool_memory_store_hot(
            manager=memory_manager,
            content="Sanitization test",
            session_id="sess_1",
        )
        serialized_str = str(res)
        assert "\\" not in serialized_str or ":\\" not in serialized_str
        assert "data/memory" not in serialized_str
        assert "chroma_db" not in serialized_str

    def test_all_tool_outputs_are_json_serializable(self, memory_manager):
        import json
        r1 = tool_memory_store_hot(memory_manager, "Task", session_id="s1")
        assert json.loads(json.dumps(r1)) == r1

        r2 = tool_memory_store_cold(memory_manager, "Pref")
        assert json.loads(json.dumps(r2)) == r2

        r3 = tool_memory_search_hot(memory_manager, "Task", session_id="s1")
        assert json.loads(json.dumps(r3)) == r3

        r4 = tool_memory_search_cold(memory_manager, "Pref")
        assert json.loads(json.dumps(r4)) == r4

        r5 = tool_memory_update(memory_manager, r1["memory_id"], "hot", content="Task updated")
        assert json.loads(json.dumps(r5)) == r5

        r6 = tool_memory_summarize(memory_manager, "s1", "Summary", [r1["memory_id"]])
        assert json.loads(json.dumps(r6)) == r6

        r7 = tool_memory_promote(memory_manager, r1["memory_id"])
        assert json.loads(json.dumps(r7)) == r7

        r8 = tool_memory_delete(memory_manager, r2["memory_id"], "cold")
        assert json.loads(json.dumps(r8)) == r8


class TestMemoryToolDispatcherIntegration:
    def test_dispatch_canonical_names(self, memory_registry):
        # Store hot via registry
        res = memory_registry.dispatch(
            "memory",
            "memory_store_hot",
            content="Hot memory via dispatcher",
            session_id="sess_dispatch",
        )
        assert isinstance(res, ToolResult)
        assert res.status == "success"
        assert res.result["status"] == "success"
        mid = res.result["memory_id"]

        # Search hot via registry
        s_res = memory_registry.dispatch(
            "memory",
            "memory_search_hot",
            query="via dispatcher",
            session_id="sess_dispatch",
        )
        assert s_res.status == "success"
        assert s_res.result["count"] == 1
        assert s_res.result["results"][0]["memory_id"] == mid

    def test_dispatch_alias_names(self, memory_registry):
        # Store cold via alias "store_cold"
        res = memory_registry.dispatch(
            "memory",
            "store_cold",
            content="Cold memory via alias",
            category="project",
        )
        assert res.status == "success"
        assert res.result["status"] == "success"

        # Search cold via alias "search_cold"
        s_res = memory_registry.dispatch(
            "memory",
            "search_cold",
            query="alias",
        )
        assert s_res.status == "success"
        assert s_res.result["count"] == 1

    def test_dispatch_promote_and_delete(self, memory_registry):
        # Store hot
        res = memory_registry.dispatch(
            "memory",
            "store_hot",
            content="To promote and delete",
            session_id="sess_1",
        )
        mid = res.result["memory_id"]

        # Promote
        prom = memory_registry.dispatch(
            "memory",
            "promote",
            memory_id=mid,
        )
        assert prom.status == "success"
        assert prom.result["promoted"] is True

        # Delete
        dele = memory_registry.dispatch(
            "memory",
            "delete",
            memory_id=mid,
            memory_type="cold",
        )
        assert dele.status == "success"
        assert dele.result["deleted"] is True


class TestCreateDefaultRegistryWithMemory:
    def test_default_registry_includes_memory_tool(self, memory_manager):
        registry = create_default_registry(memory_manager=memory_manager)
        assert "memory" in registry.available_tools

        functions = registry.list_functions("memory")
        assert "memory_search_hot" in functions
        assert "memory_search_cold" in functions
        assert "memory_store_hot" in functions
        assert "memory_store_cold" in functions
        assert "memory_update" in functions
        assert "memory_delete" in functions
        assert "memory_summarize" in functions
        assert "memory_promote" in functions
