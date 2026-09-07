"""
SAGE/core/dispatcher.py
Generic tool registry and dispatch layer.

Architectural constraint (enforced by design):
    Registered Sage tools may NOT autonomously invoke other registered
    Sage tools. Only Gemma may decide what tool should run next.

Allowed internal implementation operations (not agentic cross-tool calls):
    - argument validation
    - file/path resolution
    - manifest lookup
    - model loading
    - model health checks
    - logging, timing, cache reads/writes, persistence
    - JSON parsing, retry of deterministic I/O
"""

from __future__ import annotations
import logging
import time
import traceback
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─── Result Types ─────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Structured result of a tool dispatch.

    Always returned — never raises for tool-level errors.
    Infrastructure errors (registry lookup failures) also use this.
    """
    call_id: str
    tool_name: str
    function_name: str
    status: str               # "success" | "error" | "unknown_tool" | "unknown_function"
    result: Any = None        # Tool return value (success case)
    error: str | None = None  # Error message (failure case)
    duration_ms: float = 0.0
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-safe serialization."""
        d = {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "function_name": self.function_name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.status == "success":
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        return d

    def to_socket(self) -> dict:
        """Full rich Component N socket with nested tool_output preserved."""
        from core.sockets import build_dispatcher_socket, build_error_payload
        err_payload = None
        if self.error is not None:
            err_payload = build_error_payload(
                code=self.status.upper(),
                type_name="ToolDispatchError",
                message=self.error,
            )
        return build_dispatcher_socket(
            call_id=self.call_id,
            tool_name=self.tool_name,
            function_name=self.function_name,
            status=self.status,
            tool_output=self.result,
            sanitized_arguments=self.arguments,
            duration_ms=self.duration_ms,
            error=err_payload,
        )


# ─── Registry ─────────────────────────────────────────────────────────────────

class ToolRegistry:
    """Generic tool registry with dispatch, timing, and exception capture.

    Usage:
        registry = ToolRegistry()
        registry.register("document_db", "rag_search", tool_rag_search)
        result = registry.dispatch("document_db", "rag_search", query="hello")

    The registry must not itself decide what tool should run.
    It only executes the explicitly requested tool+function.
    """

    def __init__(self) -> None:
        self._tools: dict[tuple[str, str], Callable] = {}
        self._tool_names: set[str] = set()

    # ── Registration ──────────────────────────────────────────────────

    def register(
        self,
        tool_name: str,
        function_name: str,
        callable_fn: Callable,
    ) -> None:
        """Register a tool function.

        Args:
            tool_name:     Stable tool identifier (e.g. "document_db")
            function_name: Function within the tool (e.g. "rag_search")
            callable_fn:   The actual Python callable to invoke
        """
        key = (tool_name, function_name)
        if key in self._tools:
            logger.warning(
                "Overwriting existing registration: %s.%s", tool_name, function_name
            )
        self._tools[key] = callable_fn
        self._tool_names.add(tool_name)
        logger.info("Registered tool: %s.%s", tool_name, function_name)

    def register_mock(
        self,
        tool_name: str,
        function_name: str,
        mock_fn: Callable,
    ) -> None:
        """Replace a registered tool function with a mock implementation.

        Used for testing without GPU or external services.
        """
        key = (tool_name, function_name)
        self._tools[key] = mock_fn
        self._tool_names.add(tool_name)
        logger.info("Registered MOCK: %s.%s", tool_name, function_name)

    # ── Dispatch ──────────────────────────────────────────────────────

    def dispatch(
        self,
        tool_name: str,
        function_name: str,
        **kwargs: Any,
    ) -> ToolResult:
        """Dispatch a tool call by name.

        Returns a ToolResult in ALL cases — never raises for tool errors.
        Infrastructure errors (unknown tool/function) also return ToolResult.
        """
        call_id = f"call_{uuid.uuid4().hex[:12]}"

        # Validate tool exists
        if tool_name not in self._tool_names:
            available = sorted(self._tool_names) if self._tool_names else ["(none)"]
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                function_name=function_name,
                status="unknown_tool",
                arguments=dict(kwargs),
                error=f"Unknown tool '{tool_name}'. Available: {available}",
            )

        # Validate function exists
        key = (tool_name, function_name)
        if key not in self._tools:
            available_fns = sorted(
                fn for (tn, fn) in self._tools if tn == tool_name
            )
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                function_name=function_name,
                status="unknown_function",
                arguments=dict(kwargs),
                error=(
                    f"Unknown function '{function_name}' on tool '{tool_name}'. "
                    f"Available: {available_fns}"
                ),
            )

        # Execute with timing and exception capture
        callable_fn = self._tools[key]
        start = time.perf_counter()
        try:
            result = callable_fn(**kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Tool dispatch OK: %s.%s [%s] %.1fms",
                tool_name, function_name, call_id, duration_ms,
            )
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                function_name=function_name,
                status="success",
                result=result,
                duration_ms=duration_ms,
                arguments=dict(kwargs),
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            tb = traceback.format_exc()
            logger.error(
                "Tool dispatch FAILED: %s.%s [%s] %.1fms\n%s",
                tool_name, function_name, call_id, duration_ms, tb,
            )
            return ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                function_name=function_name,
                status="error",
                error=str(exc),
                duration_ms=duration_ms,
                arguments=dict(kwargs),
            )

    # ── Introspection ─────────────────────────────────────────────────

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tool_names)

    def list_functions(self, tool_name: str) -> list[str]:
        """Return sorted list of function names for a given tool."""
        return sorted(fn for (tn, fn) in self._tools if tn == tool_name)

    def is_registered(self, tool_name: str, function_name: str) -> bool:
        """Check whether a specific tool+function is registered."""
        return (tool_name, function_name) in self._tools

    @property
    def available_tools(self) -> dict[str, list[str]]:
        """Return a mapping of registered tool names to their functions."""
        res: dict[str, list[str]] = {}
        for (tool, func) in self._tools:
            res.setdefault(tool, []).append(func)
        return res

