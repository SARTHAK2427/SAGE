"""
sage_memory/memory_models.py
Validated data model for SAGE hot/cold memory.

These dataclasses form the shared contract between the canonical JSON store,
Chroma index, MemoryManager, and tool adapters. No external dependencies.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Literal

MemoryType = Literal["hot", "cold"]

# Suggested categories — extensible; any non-empty string is accepted.
KNOWN_CATEGORIES: frozenset[str] = frozenset({
    "personal",
    "preference",
    "project",
    "decision",
    "instruction",
    "conversation",
    "task",
    "technical",
    "summary",
})

_ISO8601_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
)


class MemoryValidationError(ValueError):
    """Raised when a memory record fails schema validation."""


def _validate_iso_timestamp(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{field_name} must be a non-empty ISO-8601 string")
    if not _ISO8601_RE.match(value.strip()):
        raise MemoryValidationError(
            f"{field_name} must be ISO-8601 formatted, got {value!r}"
        )


def _validate_unit_interval(value: float, field_name: str) -> None:
    if not isinstance(value, (int, float)):
        raise MemoryValidationError(f"{field_name} must be numeric")
    if value < 0.0 or value > 1.0:
        raise MemoryValidationError(f"{field_name} must be in [0.0, 1.0], got {value}")


def make_memory_id() -> str:
    """Generate a stable, unique memory identifier."""
    return f"mem_{uuid.uuid4().hex[:12]}"


@dataclass
class MemoryRecord:
    """One canonical memory unit stored on disk and indexed in Chroma.

    Hot memories are session-scoped; cold memories persist across sessions.
    Semantic importance is assigned by Gemma — this model only validates shape.
    """
    memory_id: str
    content: str
    memory_type: MemoryType
    category: str
    source: str
    session_id: str | None
    created_at: str
    updated_at: str
    importance: float = 0.5
    confidence: float = 0.5
    access_count: int = 0
    last_accessed: str | None = None
    is_summary: bool = False
    parent_memory_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        validate_memory_record(self)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe serialization for the canonical store."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "category": self.category,
            "source": self.source,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "access_count": int(self.access_count),
            "last_accessed": self.last_accessed,
            "is_summary": bool(self.is_summary),
            "parent_memory_ids": list(self.parent_memory_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        """Reconstruct a MemoryRecord from canonical JSON."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def to_json(self, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class MemorySearchResult:
    """Application-facing memory search hit — no raw Chroma blobs or embeddings."""
    memory_id: str
    content: str
    memory_type: MemoryType
    category: str
    distance: float
    session_id: str | None = None
    importance: float = 0.5
    confidence: float = 0.5
    is_summary: bool = False
    created_at: str | None = None
    source: str | None = None

    @property
    def derived_similarity(self) -> float:
        """Bounded similarity in [0, 1] from cosine distance."""
        return round(max(0.0, min(1.0, 1.0 - float(self.distance))), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "memory_type": self.memory_type,
            "category": self.category,
            "distance": float(self.distance),
            "derived_similarity": self.derived_similarity,
            "session_id": self.session_id,
            "importance": float(self.importance),
            "confidence": float(self.confidence),
            "is_summary": bool(self.is_summary),
            "created_at": self.created_at,
            "source": self.source,
        }


def validate_memory_record(record: MemoryRecord) -> None:
    """Validate a MemoryRecord in place. Raises MemoryValidationError on failure."""
    if not record.memory_id or not isinstance(record.memory_id, str):
        raise MemoryValidationError("memory_id must be a non-empty string")
    if not record.memory_id.startswith("mem_"):
        raise MemoryValidationError(
            f"memory_id must start with 'mem_', got {record.memory_id!r}"
        )

    if not isinstance(record.content, str) or not record.content.strip():
        raise MemoryValidationError("content must be a non-empty string")

    if record.memory_type not in ("hot", "cold"):
        raise MemoryValidationError(
            f"memory_type must be 'hot' or 'cold', got {record.memory_type!r}"
        )

    if not isinstance(record.category, str) or not record.category.strip():
        raise MemoryValidationError("category must be a non-empty string")

    if not isinstance(record.source, str) or not record.source.strip():
        raise MemoryValidationError("source must be a non-empty string")

    if record.memory_type == "hot":
        if not record.session_id or not str(record.session_id).strip():
            raise MemoryValidationError("session_id is required for hot memories")
    elif record.session_id is not None and not str(record.session_id).strip():
        raise MemoryValidationError("session_id must be a non-empty string or None")

    _validate_iso_timestamp(record.created_at, "created_at")
    _validate_iso_timestamp(record.updated_at, "updated_at")
    if record.last_accessed is not None:
        _validate_iso_timestamp(record.last_accessed, "last_accessed")

    _validate_unit_interval(record.importance, "importance")
    _validate_unit_interval(record.confidence, "confidence")

    if not isinstance(record.access_count, int) or record.access_count < 0:
        raise MemoryValidationError("access_count must be a non-negative integer")

    if not isinstance(record.is_summary, bool):
        raise MemoryValidationError("is_summary must be a boolean")

    if not isinstance(record.parent_memory_ids, list):
        raise MemoryValidationError("parent_memory_ids must be a list")
    for parent_id in record.parent_memory_ids:
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise MemoryValidationError(
                "parent_memory_ids must contain non-empty strings"
            )
