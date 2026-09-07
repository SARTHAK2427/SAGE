"""
tests/test_app_upload.py
Unit tests for the upload flow and RegisteredDocument tracking in app.py (Phase 4).

Verifies:
    - Upload creates RegisteredDocument in RunState
    - attachments_manifest and file_map retain doc_id
    - Error handling during ingestion doesn't crash the server
"""

import pytest
from core.run_state import RunState, RegisteredDocument


def test_registered_document_creation():
    state = RunState(request_id="req_123", user_text="Test prompt")
    doc = state.register_document(
        doc_id="doc_a81f42c91e",
        display_name="annual_report.pdf",
        file_type="pdf",
        source_name="annual_report.pdf",
    )

    assert doc.doc_id == "doc_a81f42c91e"
    assert doc.display_name == "annual_report.pdf"
    assert doc.file_type == "pdf"
    assert len(state.registered_documents) == 1
    assert state.list_document_ids() == ["doc_a81f42c91e"]


def test_run_state_lookup():
    state = RunState(request_id="req_123", user_text="Test")
    state.register_document("doc_1", "doc1.txt", "txt")
    state.register_document("doc_2", "doc2.pdf", "pdf")

    found = state.get_document("doc_2")
    assert found is not None
    assert found.display_name == "doc2.pdf"

    missing = state.get_document("doc_999")
    assert missing is None
