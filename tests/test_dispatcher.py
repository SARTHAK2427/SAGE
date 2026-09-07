"""
Tests for core/dispatcher.py — generic tool registry and dispatch.
"""

import sys
from pathlib import Path

# Ensure SAGE root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from core.dispatcher import ToolRegistry, ToolResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _echo_tool(message: str = "") -> dict:
    """Simple tool that echoes its argument."""
    return {"echo": message}


def _failing_tool() -> dict:
    """Tool that always raises."""
    raise ValueError("Something went wrong inside the tool")


def _mock_echo(message: str = "") -> dict:
    return {"echo": f"MOCK: {message}"}


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestToolRegistry:

    def test_register_and_dispatch_success(self):
        reg = ToolRegistry()
        reg.register("test_tool", "echo", _echo_tool)
        result = reg.dispatch("test_tool", "echo", message="hello")
        assert result.status == "success"
        assert result.result == {"echo": "hello"}
        assert result.call_id.startswith("call_")
        assert result.duration_ms >= 0

    def test_unknown_tool(self):
        reg = ToolRegistry()
        result = reg.dispatch("nonexistent", "func")
        assert result.status == "unknown_tool"
        assert "nonexistent" in result.error
        assert result.result is None

    def test_unknown_function(self):
        reg = ToolRegistry()
        reg.register("test_tool", "echo", _echo_tool)
        result = reg.dispatch("test_tool", "nonexistent_fn")
        assert result.status == "unknown_function"
        assert "nonexistent_fn" in result.error

    def test_tool_exception_captured(self):
        reg = ToolRegistry()
        reg.register("test_tool", "fail", _failing_tool)
        result = reg.dispatch("test_tool", "fail")
        assert result.status == "error"
        assert "Something went wrong" in result.error
        assert result.result is None

    def test_mock_replaces_real(self):
        reg = ToolRegistry()
        reg.register("test_tool", "echo", _echo_tool)
        # Verify real works
        r1 = reg.dispatch("test_tool", "echo", message="hi")
        assert r1.result == {"echo": "hi"}
        # Replace with mock
        reg.register_mock("test_tool", "echo", _mock_echo)
        r2 = reg.dispatch("test_tool", "echo", message="hi")
        assert r2.result == {"echo": "MOCK: hi"}

    def test_list_tools_and_functions(self):
        reg = ToolRegistry()
        reg.register("db", "rag_search", _echo_tool)
        reg.register("db", "exact_search", _echo_tool)
        reg.register("math", "calculate", _echo_tool)
        assert reg.list_tools() == ["db", "math"]
        assert reg.list_functions("db") == ["exact_search", "rag_search"]

    def test_is_registered(self):
        reg = ToolRegistry()
        reg.register("db", "rag_search", _echo_tool)
        assert reg.is_registered("db", "rag_search") is True
        assert reg.is_registered("db", "nope") is False

    def test_result_to_dict_success(self):
        reg = ToolRegistry()
        reg.register("t", "f", _echo_tool)
        result = reg.dispatch("t", "f", message="ok")
        d = result.to_dict()
        assert d["status"] == "success"
        assert d["result"] == {"echo": "ok"}
        assert "error" not in d

    def test_result_to_dict_error(self):
        reg = ToolRegistry()
        reg.register("t", "f", _failing_tool)
        result = reg.dispatch("t", "f")
        d = result.to_dict()
        assert d["status"] == "error"
        assert "error" in d
        assert "result" not in d
