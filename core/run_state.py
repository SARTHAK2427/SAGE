"""
SAGE/core/run_state.py
Backend-owned runtime execution state.

This is NOT Gemma's private reasoning — it is factual execution metadata
owned by Python. Gemma owns decisions; Python owns state.

Do not let Gemma invent:
    - actual file paths
    - actual doc IDs
    - call IDs
    - timestamps
    - execution status
    - model process IDs
"""

from __future__ import annotations
import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class RegisteredDocument:
    """Runtime metadata for an ingested document.

    This object is runtime metadata only — it does NOT duplicate
    the full canonical manifest from sage_document_db.

    The goal is to allow runtime state to say:
        available documents:
        - doc_a81f42c91e : report.pdf
        - doc_b2c3d4e5f6 : employees.docx
    without forcing the agent to know storage paths.
    """
    doc_id: str
    display_name: str
    file_type: str | None = None
    source_name: str | None = None


@dataclass
class ToolCallRecord:
    """One recorded tool dispatch."""
    call_id: str
    tool_name: str
    function_name: str
    sanitized_args: dict = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    status: str = "pending"
    result_summary: str | None = None
    error: str | None = None

    def to_socket(self) -> dict:
        return asdict(self)


@dataclass
class RunState:
    """Backend-owned state for one user request lifecycle.

    Serializable for debugging. Do not expose every internal field
    to Gemma by default.
    """
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    request_id: str = ""
    loop_index: int = 0
    user_text: str = ""

    registered_documents: list[RegisteredDocument] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    referenced_images: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    current_model: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.request_id:
            self.request_id = self.run_id

    # ── Document registration ─────────────────────────────────────────

    def register_document(
        self,
        doc_id: str,
        display_name: str,
        file_type: str | None = None,
        source_name: str | None = None,
    ) -> RegisteredDocument:
        """Register an ingested document in the run state."""
        doc = RegisteredDocument(
            doc_id=doc_id,
            display_name=display_name,
            file_type=file_type,
            source_name=source_name,
        )
        self.registered_documents.append(doc)
        self.updated_at = time.time()
        return doc

    def get_document(self, doc_id: str) -> RegisteredDocument | None:
        """Look up a registered document by doc_id."""
        for doc in self.registered_documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def list_document_ids(self) -> list[str]:
        """Return all registered doc_ids."""
        return [d.doc_id for d in self.registered_documents]

    # ── Tool call recording ───────────────────────────────────────────

    def record_tool_call(
        self,
        call_id: str,
        tool_name: str,
        function_name: str,
        sanitized_args: dict | None = None,
        start_time: float | None = None,
    ) -> ToolCallRecord:
        """Record the start of a tool call."""
        record = ToolCallRecord(
            call_id=call_id,
            tool_name=tool_name,
            function_name=function_name,
            sanitized_args=sanitized_args or {},
            start_time=start_time or time.time(),
        )
        self.tool_calls.append(record)
        self.updated_at = time.time()
        return record

    def complete_tool_call(
        self,
        call_id: str,
        status: str,
        result_summary: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record the completion of a tool call."""
        self.record_tool_result(
            call_id=call_id,
            status=status,
            result_summary=result_summary,
            error=error,
        )

    def record_tool_result(
        self,
        call_id: str,
        status: str,
        result_summary: str | None = None,
        error: str | None = None,
        end_time: float | None = None,
    ) -> None:
        """Record the completion and result of a tool call."""
        for record in self.tool_calls:
            if record.call_id == call_id:
                record.end_time = end_time or time.time()
                record.duration_ms = (record.end_time - record.start_time) * 1000
                record.status = status
                record.result_summary = result_summary
                record.error = error
                break
        self.updated_at = time.time()

    # ── Serialization ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-safe serialization for debugging."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON string serialization."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_socket(self) -> dict:
        """Full rich Component O socket."""
        from core.sockets import build_run_state_socket
        return build_run_state_socket(
            run_id=self.run_id,
            request_id=self.request_id,
            loop_index=self.loop_index,
            user_text=self.user_text,
            registered_documents=self.registered_documents,
            tool_calls=self.tool_calls,
            current_model=self.current_model,
            created_at=self.created_at,
            updated_at=self.updated_at,
            errors=self.errors,
        )
