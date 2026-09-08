"""
tests/test_memory_run_state.py
Phase 15 tests: RunState integration of SAGE Hot/Cold Memory subsystem.

Validates:
1. RunState access to MemoryManager:
   - Default resolution lazily reuses db_service.get_memory_manager().
   - Explicit injection uses the provided manager instance.
   - Repeated access returns the identical manager instance (no competing instances).
2. Session ID context boundary:
   - session_id defaults to "" when unspecified.
   - Explicit session_id is preserved as contextual state.
   - session_id is serializable in to_dict() and to_json().
3. Serialization isolation:
   - MemoryManager instance is NEVER present in to_dict(), to_json(), or to_socket().
   - Dataclass asdict() does not crash or attempt deepcopy on Chroma bindings.
4. Existing RunState functionality preserved:
   - Document registration, tool call recording, completion, and socket generation.
5. Strict negative tests (NO automatic memory behavior):
   - Constructing RunState() does NOT create memory records on disk.
   - Constructing RunState() does NOT touch Chroma collections.
   - Registering a document does NOT auto-create memories.
   - Recording or completing a tool call does NOT auto-summarize or auto-promote.
   - Accessing run_state.memory_manager does NOT execute semantic memory operations.
6. Hot memory session requirements remain enforced:
   - Hot memory search/store through the manager still requires valid session_id.
7. Phase 14 index synchronization remains intact:
   - sync_indexes() and rebuild_memory_indexes() remain accessible and functional.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.run_state import RunState, RegisteredDocument, ToolCallRecord
from sage_document_db.embeddings import EmbeddingService
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager
from sage_memory.memory_models import MemoryRecord, MemoryValidationError
from sage_memory.memory_store import MemoryStore
from tools.memory import tool_memory_store_hot, tool_memory_search_hot


@pytest.fixture()
def custom_memory_manager(tmp_memory_root, tmp_chroma_root):
    store = MemoryStore(tmp_memory_root)
    index = MemoryChromaStore(tmp_chroma_root, EmbeddingService())
    return MemoryManager(store=store, index=index)


# ─── 1. RunState MemoryManager Access & Lifecycle ──────────────────────────────

class TestRunStateMemoryAccess:
    def test_run_state_lazy_default_memory_manager(self):
        """RunState without explicit manager lazily resolves the shared db_service manager."""
        state = RunState(request_id="req_test_lazy", user_text="Hello world")
        mgr = state.memory_manager
        assert mgr is not None
        assert isinstance(mgr, MemoryManager)

    def test_run_state_explicit_memory_manager_injection(self, custom_memory_manager):
        """RunState uses explicitly injected MemoryManager instance."""
        state = RunState(
            request_id="req_injected",
            session_id="sess_custom_inject",
            memory_manager=custom_memory_manager,
        )
        assert state.memory_manager is custom_memory_manager
        assert state.session_id == "sess_custom_inject"

    def test_repeated_access_returns_same_instance(self, custom_memory_manager):
        """Repeated property accesses return the exact same instance without re-instantiation."""
        state = RunState(memory_manager=custom_memory_manager)
        first = state.memory_manager
        second = state.memory_manager
        assert first is second
        assert first is custom_memory_manager

    def test_memory_manager_setter(self, custom_memory_manager):
        """RunState allows setting memory_manager explicitly."""
        state = RunState()
        state.memory_manager = custom_memory_manager
        assert state.memory_manager is custom_memory_manager


# ─── 2. Session ID Context Boundary ───────────────────────────────────────────

class TestRunStateSessionIdBoundary:
    def test_default_session_id_is_empty_string(self):
        """Default session_id is an empty string and does not invent sessions."""
        state = RunState()
        assert state.session_id == ""

    def test_explicit_session_id_context_preserved(self):
        """Session ID is stored as context only, without triggering session lifecycle."""
        state = RunState(session_id="sess_alpha_99")
        assert state.session_id == "sess_alpha_99"

    def test_session_id_in_serialization(self):
        """session_id is properly serialized in to_dict and to_json."""
        state = RunState(request_id="req_001", session_id="sess_beta_42", user_text="Check status")
        d = state.to_dict()
        assert d["session_id"] == "sess_beta_42"

        json_str = state.to_json()
        parsed = json.loads(json_str)
        assert parsed["session_id"] == "sess_beta_42"


# ─── 3. Serialization Isolation ───────────────────────────────────────────────

class TestRunStateSerializationIsolation:
    def test_memory_manager_never_in_to_dict(self, custom_memory_manager):
        """MemoryManager must NOT be included in to_dict() output."""
        state = RunState(
            request_id="req_serial",
            session_id="sess_serial",
            memory_manager=custom_memory_manager,
        )
        d = state.to_dict()
        assert "memory_manager" not in d
        assert "_memory_manager" not in d

    def test_memory_manager_never_in_to_json(self, custom_memory_manager):
        """MemoryManager must NOT be included in to_json() output."""
        state = RunState(
            request_id="req_serial",
            session_id="sess_serial",
            memory_manager=custom_memory_manager,
        )
        json_str = state.to_json()
        parsed = json.loads(json_str)
        assert "memory_manager" not in parsed
        assert "_memory_manager" not in parsed

    def test_to_socket_is_unaffected(self, custom_memory_manager):
        """to_socket() outputs standard Component O socket without leaking MemoryManager."""
        state = RunState(
            request_id="req_sock",
            user_text="Test prompt",
            memory_manager=custom_memory_manager,
        )
        sock = state.to_socket()
        assert sock["operation"] == "run_state_snapshot"
        assert "memory_manager" not in sock
        assert "memory_manager" not in sock["result"]
        assert "memory_manager" not in sock["context"]


# ─── 4. Preservation of Existing RunState Behavior ───────────────────────────

class TestExistingRunStateBehavior:
    def test_document_and_tool_call_lifecycle(self, custom_memory_manager):
        """All existing RunState methods work unchanged when memory_manager is present."""
        state = RunState(
            request_id="req_existing",
            user_text="Summarize docs",
            memory_manager=custom_memory_manager,
        )
        doc = state.register_document("doc_test_1", "spec.pdf", "pdf", "spec.pdf")
        assert doc.doc_id == "doc_test_1"
        assert len(state.registered_documents) == 1
        assert state.get_document("doc_test_1") is doc

        call = state.record_tool_call(
            call_id="call_01",
            tool_name="math",
            function_name="calculate",
            sanitized_args={"expression": "2+2"},
        )
        assert call.status == "pending"
        state.complete_tool_call("call_01", "success", result_summary="4")
        assert state.tool_calls[0].status == "success"
        assert state.tool_calls[0].result_summary == "4"


# ─── 5. Negative Tests: No Automatic Memory Operations ────────────────────────

class TestNoAutomaticMemoryOperations:
    def test_run_state_construction_does_not_touch_memory_store(self, custom_memory_manager):
        """Instantiating RunState must not create any files in hot or cold memory stores."""
        hot_before = list(custom_memory_manager.store.hot_dir.iterdir())
        cold_before = list(custom_memory_manager.store.cold_dir.iterdir())

        _ = RunState(
            request_id="req_negative",
            session_id="sess_neg",
            user_text="A very important secret that shouldn't be auto-saved",
            memory_manager=custom_memory_manager,
        )

        hot_after = list(custom_memory_manager.store.hot_dir.iterdir())
        cold_after = list(custom_memory_manager.store.cold_dir.iterdir())
        assert hot_before == hot_after
        assert cold_before == cold_after

    def test_run_state_construction_does_not_touch_chroma(self, custom_memory_manager):
        """Instantiating RunState must not alter Chroma counts."""
        hot_count_before = custom_memory_manager.index.count("hot")
        cold_count_before = custom_memory_manager.index.count("cold")

        _ = RunState(
            request_id="req_negative_chroma",
            session_id="sess_neg_chroma",
            user_text="Another prompt",
            memory_manager=custom_memory_manager,
        )

        assert custom_memory_manager.index.count("hot") == hot_count_before
        assert custom_memory_manager.index.count("cold") == cold_count_before

    def test_document_registration_does_not_create_memories(self, custom_memory_manager):
        """Registering a document in RunState does NOT auto-create memory records."""
        state = RunState(
            request_id="req_doc_mem",
            session_id="sess_doc_mem",
            memory_manager=custom_memory_manager,
        )
        state.register_document("doc_x", "doc_x.txt", "txt")

        assert len(custom_memory_manager.store.list_hot()) == 0
        assert len(custom_memory_manager.store.list_cold()) == 0

    def test_tool_call_recording_does_not_auto_summarize_or_promote(self, custom_memory_manager):
        """Recording tool calls in RunState does NOT trigger summarization or promotion."""
        state = RunState(
            request_id="req_tool_mem",
            session_id="sess_tool_mem",
            memory_manager=custom_memory_manager,
        )
        for i in range(15):
            call = state.record_tool_call(f"call_{i}", "test_tool", "do_something")
            state.complete_tool_call(f"call_{i}", "success", result_summary="done")

        assert len(custom_memory_manager.store.list_hot()) == 0
        assert len(custom_memory_manager.store.list_cold()) == 0

    def test_explicit_manager_access_does_not_execute_semantic_memory_operations(self, custom_memory_manager):
        """Accessing state.memory_manager does NOT execute semantic searches, creations, or deletions."""
        state = RunState(memory_manager=custom_memory_manager)
        mgr = state.memory_manager

        # Accessing mgr property must not have populated anything
        assert mgr.store.list_hot() == []
        assert mgr.store.list_cold() == []
        assert mgr.index.count("hot") == 0
        assert mgr.index.count("cold") == 0


# ─── 6. Hot Memory Session Requirement Enforcement ────────────────────────────

class TestHotMemorySessionRequirementEnforced:
    def test_hot_memory_tool_still_requires_explicit_session_id(self, custom_memory_manager):
        """tool_memory_store_hot still requires a non-empty session_id."""
        state = RunState(session_id="", memory_manager=custom_memory_manager)

        # Attempting to store hot memory with empty session_id fails
        res = tool_memory_store_hot(
            manager=state.memory_manager,
            content="Some hot memory",
            session_id=state.session_id,  # empty string
        )
        assert res["status"] == "error"
        assert "session_id is required" in res["error"]

    def test_hot_memory_search_requires_explicit_session_id(self, custom_memory_manager):
        """tool_memory_search_hot requires a valid session_id."""
        state = RunState(session_id="", memory_manager=custom_memory_manager)
        res = tool_memory_search_hot(
            manager=state.memory_manager,
            query="test",
            session_id=state.session_id,
        )
        assert res["status"] == "error"
        assert "session_id is required" in res["error"]


# ─── 7. Phase 14 Sync / Rebuild Integrity ─────────────────────────────────────

class TestPhase14SyncIntegrityViaRunState:
    def test_phase14_sync_accessible_via_run_state(self, custom_memory_manager):
        """Phase 14 startup sync operates correctly through the RunState memory_manager access."""
        state = RunState(memory_manager=custom_memory_manager)
        sync_res = state.memory_manager.sync_indexes()
        assert sync_res["status"] == "in_sync"
        assert sync_res["total"] == 0

    def test_phase14_rebuild_accessible_via_run_state(self, custom_memory_manager):
        """Phase 14 rebuild operates correctly through the RunState memory_manager access."""
        state = RunState(memory_manager=custom_memory_manager)
        rebuild_res = state.memory_manager.rebuild_memory_indexes()
        assert rebuild_res["status"] == "success"
        assert rebuild_res["total"] == 0
