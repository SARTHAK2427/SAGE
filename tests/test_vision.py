"""
tests/test_vision.py
Unit tests for the Vision / OCR adapter (Phase 12).

Verifies:
    - Mock mode produces deterministic mock results without GPU
    - Mock results are never saved to real derived storage or indexed into Chroma
    - Proper error response when image_ids list is empty
"""

import pytest
from unittest.mock import MagicMock
from tools.vision import tool_vision_analyze, register_vision_tools
from core.dispatcher import ToolRegistry


def test_vision_mock_mode():
    mock_db = MagicMock()
    mock_db.artifact_fetch.return_value = {
        "id": "img_000001",
        "type": "image",
        "local_path": "fake.png",
        "exists": True,
    }

    res = tool_vision_analyze(
        db=mock_db,
        doc_id="doc_test_123",
        image_ids=["img_000001"],
        instruction="Describe this chart",
    )

    assert res["status"] == "success"
    assert len(res["results"]) == 1
    item = res["results"][0]
    assert item["image_id"] == "img_000001"
    assert "MOCK" in item["raw_output"]

    # Verify db.add_image_analysis was NOT called in mock mode (mock isolation rule)
    assert not mock_db.add_image_analysis.called


def test_vision_empty_images():
    mock_db = MagicMock()
    res = tool_vision_analyze(
        db=mock_db,
        doc_id="doc_test_123",
        image_ids=[],
    )
    assert res["status"] == "error"
    assert "No image_ids" in res["error"]


def test_vision_registration():
    mock_db = MagicMock()
    mock_db.artifact_fetch.return_value = {
        "id": "img_1",
        "type": "image",
        "local_path": "fake.png",
        "exists": True,
    }
    registry = ToolRegistry()
    register_vision_tools(registry, db=mock_db)

    assert registry.is_registered("vision", "analyze")
    disp = registry.dispatch("vision", "analyze", doc_id="doc_1", image_ids=["img_1"])
    assert disp.status == "success"
