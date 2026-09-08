"""
sage_memory/memory_summarizer.py
Threshold-aware helper for hot memory summarization workflows.

Summarization content is produced by Gemma; this module only handles
deterministic bookkeeping (parent IDs, is_summary flag, threshold checks).
Implementation: Phase 10.
"""

from __future__ import annotations

from typing import Any

from .config import MEMORY_HOT_SUMMARY_THRESHOLD
from .memory_models import MemoryRecord


class MemorySummarizer:
    """Deterministic summarization bookkeeping — no LLM calls here."""

    def __init__(
        self,
        manager: Any,
        threshold: int = MEMORY_HOT_SUMMARY_THRESHOLD,
    ) -> None:
        self._manager = manager
        self._threshold = threshold

    @property
    def threshold(self) -> int:
        return self._threshold

    def should_summarize(self, hot_count: int) -> bool:
        """Return True when hot memory count meets or exceeds the threshold."""
        return hot_count >= self._threshold

    def create_summary_record(
        self,
        *,
        memory_id: str,
        content: str,
        session_id: str,
        parent_memory_ids: list[str],
        created_at: str,
        updated_at: str,
        source: str = "gemma_summarizer",
        importance: float = 0.5,
        confidence: float = 0.5,
    ) -> MemoryRecord:
        """Build a validated summary MemoryRecord (not yet persisted). Phase 10."""
        return MemoryRecord(
            memory_id=memory_id,
            content=content,
            memory_type="hot",
            category="summary",
            source=source,
            session_id=session_id,
            created_at=created_at,
            updated_at=updated_at,
            importance=importance,
            confidence=confidence,
            is_summary=True,
            parent_memory_ids=list(parent_memory_ids),
        )

    def count_hot_memories(self, session_id: str) -> int:
        """Return the count of hot memories for the given session."""
        if not session_id or not str(session_id).strip():
            return 0
        return len(self._manager.store.list_hot(session_id=session_id.strip()))

    def should_summarize_session(self, session_id: str) -> bool:
        """Return True when hot memory count for session meets or exceeds threshold."""
        return self.should_summarize(self.count_hot_memories(session_id))

    def persist_summary(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a summary via MemoryManager. Phase 10."""
        if not record.is_summary:
            raise ValueError("record.is_summary must be True for summary persistence")
        return self._manager.add_hot_memory(record)
