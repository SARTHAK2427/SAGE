"""
tests/test_coder.py
Unit tests for the Coder tool adapter (Phase 13).

Verifies:
    - Mock mode produces deterministic structured results without Docker/GPU
    - Both 'coder' and 'code_specialist' alias dispatch cleanly
    - Empty task validation
"""

import pytest
from unittest.mock import MagicMock
from tools.coder import tool_code_execute, register_coder_tools
from core.dispatcher import ToolRegistry


def test_coder_mock_mode():
    res = tool_code_execute(
        task="Calculate Fibonacci 10",
        context="Python 3.10",
    )
    assert res["status"] == "success"
    assert res["succeeded"] is True
    assert res["exit_code"] == 0
    assert "MOCK_RESULT" in res["stdout"]
    assert "Fibonacci 10" in res["stdout"]
    assert res["attempts"] == 1


def test_coder_empty_task():
    res = tool_code_execute(task="")
    assert res["status"] == "error"
    assert res["succeeded"] is False
    assert "No task" in res["error"]


def test_coder_registration():
    registry = ToolRegistry()
    register_coder_tools(registry)

    assert registry.is_registered("coder", "execute")
    assert registry.is_registered("code_specialist", "execute")

    disp1 = registry.dispatch("coder", "execute", task="test task")
    assert disp1.status == "success"

    disp2 = registry.dispatch("code_specialist", "execute", task="another task")
    assert disp2.status == "success"
