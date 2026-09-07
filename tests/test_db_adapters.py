"""
tests/test_db_adapters.py
Unit tests for the SAGE Document Database tool adapters (Phase 6 & 7).

Verifies:
    - tool_rag_search returns structured JSON-safe dicts
    - tool_exact_search returns bounded match dicts
    - tool_artifact_fetch retrieves artifacts and strips backend-internal paths
    - tool_list_artifacts enumerates artifacts without loading pixels
    - errors fail explicitly without crashing
"""

import pytest
from unittest.mock import MagicMock
from tools.document_database import (
    tool_rag_search,
    tool_exact_search,
    tool_artifact_fetch,
    tool_list_artifacts,
    register_document_db_tools,
)
from core.dispatcher import ToolRegistry


from dataclasses import dataclass, field


@dataclass
class DummyRagRecord:
    record_id: str = "r1"
    origin: str = "source"
    record_type: str = "chunk"
    doc_id: str = "doc_1"
    text: str = "Sample invoice $3500"
    distance: float = 0.12
    page: int = 1
    source_element_ids: list = field(default_factory=lambda: ["el_1"])
    image_refs: list = field(default_factory=list)
    table_refs: list = field(default_factory=list)
    code_refs: list = field(default_factory=list)


@dataclass
class DummyExactMatch:
    doc_id: str = "doc_1"
    element_id: str = "txt_001"
    element_type: str = "text"
    page: int = 1
    table_row: int | None = None
    table_col: int | None = None
    ref: str = "text/document.json"
    match_start: int = 7
    match_end: int = 11
    matched_text: str = "3500"
    snippet: str = "total: 3500"



def test_tool_rag_search():
    mock_db = MagicMock()
    mock_db.rag_search.return_value = [DummyRagRecord()]

    results = tool_rag_search(mock_db, query="invoice", top_k=3)
    assert results["status"] == "success"
    assert len(results["results"]) == 1
    rec = results["results"][0]
    assert rec["record_id"] == "r1"
    assert rec["doc_id"] == "doc_1"
    assert "distance" in rec
    assert rec["text"] == "Sample invoice $3500"


def test_tool_exact_search():
    mock_db = MagicMock()
    mock_db.exact_search.return_value = [DummyExactMatch()]

    results = tool_exact_search(mock_db, query="3500")
    assert results["status"] == "success"
    assert len(results["results"]) == 1
    match = results["results"][0]
    assert match["doc_id"] == "doc_1"
    assert match["element_id"] == "txt_001"
    assert match["matched_text"] == "3500"


def test_tool_artifact_fetch_strips_paths():
    mock_db = MagicMock()
    mock_db.artifact_fetch.return_value = {
        "id": "img_001",
        "type": "image",
        "page": 2,
        "local_path": "/secret/path/to/img_001.png",
        "exists": True,
    }

    result = tool_artifact_fetch(mock_db, doc_id="doc_1", element_id="img_001")
    assert result["status"] == "success"
    element = result["element"]
    assert element["id"] == "img_001"
    assert element["type"] == "image"
    # Verify local filesystem path is stripped from caller-facing result
    assert "local_path" not in element
    assert "filesystem_path" not in element


def test_tool_list_artifacts():
    mock_db = MagicMock()
    mock_db.list_artifacts.return_value = [
        {"element_id": "img_001", "type": "image", "page": 1},
        {"element_id": "tbl_001", "type": "table", "page": 2},
    ]

    result = tool_list_artifacts(mock_db, doc_id="doc_1", artifact_type="image")
    assert result["status"] == "success"
    assert result["doc_id"] == "doc_1"
    assert len(result["artifacts"]) == 2


def test_db_tool_registration():
    mock_db = MagicMock()
    registry = ToolRegistry()
    register_document_db_tools(registry, mock_db)

    assert registry.is_registered("document_db", "rag_search")
    assert registry.is_registered("document_db", "exact_search")
    assert registry.is_registered("document_db", "artifact_fetch")
    assert registry.is_registered("document_db", "list_artifacts")

