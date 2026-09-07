"""
Tests for core/run_state.py — runtime execution state.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.run_state import RunState, RegisteredDocument, ToolCallRecord


class TestRunState:

    def test_create_default(self):
        state = RunState()
        assert state.run_id.startswith("run_")
        assert state.registered_documents == []
        assert state.tool_calls == []

    def test_register_document(self):
        state = RunState(request_id="req_001", user_text="analyze report")
        doc = state.register_document(
            doc_id="doc_abc123",
            display_name="report.pdf",
            file_type="pdf",
            source_name="report.pdf",
        )
        assert doc.doc_id == "doc_abc123"
        assert doc.display_name == "report.pdf"
        assert len(state.registered_documents) == 1

    def test_get_document(self):
        state = RunState()
        state.register_document("doc_aaa", "file_a.txt", "txt")
        state.register_document("doc_bbb", "file_b.pdf", "pdf")
        found = state.get_document("doc_bbb")
        assert found is not None
        assert found.display_name == "file_b.pdf"
        assert state.get_document("doc_nope") is None

    def test_list_document_ids(self):
        state = RunState()
        state.register_document("doc_1", "a.txt")
        state.register_document("doc_2", "b.pdf")
        assert state.list_document_ids() == ["doc_1", "doc_2"]

    def test_record_and_complete_tool_call(self):
        state = RunState()
        record = state.record_tool_call(
            call_id="call_001",
            tool_name="db",
            function_name="rag_search",
            sanitized_args={"query": "revenue"},
        )
        assert record.status == "pending"
        state.complete_tool_call(
            call_id="call_001",
            status="success",
            result_summary="Found 3 results",
        )
        completed = state.tool_calls[0]
        assert completed.status == "success"
        assert completed.duration_ms >= 0
        assert completed.result_summary == "Found 3 results"

    def test_serializable_to_json(self):
        state = RunState(request_id="req_test", user_text="hello")
        state.register_document("doc_x", "test.txt", "txt")
        state.record_tool_call("c1", "db", "rag_search")
        # Must not raise
        json_str = state.to_json()
        parsed = json.loads(json_str)
        assert parsed["request_id"] == "req_test"
        assert len(parsed["registered_documents"]) == 1
        assert len(parsed["tool_calls"]) == 1

    def test_to_dict(self):
        state = RunState()
        d = state.to_dict()
        assert isinstance(d, dict)
        assert "run_id" in d
        assert "registered_documents" in d
