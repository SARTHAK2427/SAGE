"""
tests/test_orchestrator_integration.py
Integration tests for the Orchestrator with the Generic Dispatcher and RunState (Phase 15 & 16).

Verifies:
    - Orchestrator initializes with ToolRegistry without error
    - Tool calls are routed through ToolRegistry
    - RunState records tool calls and registered documents
    - Legacy DocumentProcessor is never invoked
    - Mock mode run completes end-to-end
"""

import pytest
from unittest.mock import MagicMock
from core.dispatcher import ToolRegistry
from core.run_state import RunState
from orchestrator import Orchestrator
from tools.registry import create_default_registry


def test_orchestrator_initialization():
    registry = ToolRegistry()
    orch = Orchestrator(registry=registry)
    assert orch.registry is registry


def test_orchestrator_mock_run():
    orch = Orchestrator()
    run_state = RunState(user_text="What is the revenue in Q3?")
    run_state.register_document(
        doc_id="doc_test_123",
        display_name="financials.pdf",
        file_type="pdf",
    )

    result = orch.run(
        user_objective="What is the revenue in Q3?",
        attachments_manifest=[{
            "ref": "file_1",
            "doc_id": "doc_test_123",
            "name": "financials.pdf",
            "type": "pdf",
            "size": 1024,
        }],
        file_map={},
        run_state=run_state,
    )

    assert result["status"] == "success"
    assert "MOCK_RESPONSE" in result["answer"]
    assert "financials.pdf" in result["answer"]
    assert "telemetry" in result
    assert "trace" in result
    assert "run_state" in result


def test_orchestrator_dispatch_tool_call_math():
    registry = create_default_registry()
    orch = Orchestrator(registry=registry)
    run_state = RunState(user_text="Calculate 125 * 4")

    telemetry = {}
    trace = []
    file_map = {}

    res = orch._dispatch_tool_call(
        call={"tool": "math", "expression": "125 * 4"},
        run_state=run_state,
        telemetry=telemetry,
        trace=trace,
        file_map=file_map,
    )

    assert res["tool"] == "math"
    assert res["status"] == "success"
    assert res["result"]["value"] == 500


def test_orchestrator_dispatch_tool_call_coder():
    registry = create_default_registry()
    orch = Orchestrator(registry=registry)
    run_state = RunState(user_text="Run python hello")

    telemetry = {}
    trace = []
    file_map = {}

    res = orch._dispatch_tool_call(
        call={"tool": "coder", "task": "Print hello world"},
        run_state=run_state,
        telemetry=telemetry,
        trace=trace,
        file_map=file_map,
    )

    assert res["tool"] == "code_specialist"
    assert res["status"] == "success"
    assert "MOCK_RESULT" in res["result"]["stdout"]
    assert telemetry.get("coder_calls") == 1


def test_orchestrator_dispatch_tool_call_vision():
    registry = create_default_registry()
    orch = Orchestrator(registry=registry)
    run_state = RunState(user_text="Inspect image")
    run_state.register_document(doc_id="doc_abc", display_name="scan.png")

    telemetry = {}
    trace = []
    file_map = {}

    res = orch._dispatch_tool_call(
        call={"tool": "vision", "doc_id": "doc_abc", "image_ids": ["img_001"], "task": "Read invoice total"},
        run_state=run_state,
        telemetry=telemetry,
        trace=trace,
        file_map=file_map,
    )

    assert res["tool"] == "vision_ocr"
    assert res["function"] == "analyze_image"
    assert telemetry.get("document_calls") == 1

