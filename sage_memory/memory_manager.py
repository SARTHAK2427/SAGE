"""
sage_memory/memory_manager.py
Coordinates canonical store + Chroma index.

Deterministic execution only — no semantic heuristics.
Gemma 4B decides what, when, and where to store, summarize, and promote.
MemoryManager strictly executes dual-writes, queries, updates, and deletes.
"""

from __future__ import annotations

import logging
from typing import Any

from sage_document_db.utils import iso_now

from .config import MEMORY_DEFAULT_TOP_K
from .memory_models import MemoryRecord, MemorySearchResult, make_memory_id

logger = logging.getLogger(__name__)


class MemoryManagerError(Exception):
    """Raised when a MemoryManager operation fails."""


class MemoryManager:
    """Facade coordinating canonical MemoryStore and MemoryChromaStore.

    Rules:
    - Canonical store (JSON on disk) is the source of truth.
    - Chroma is a disposable, rebuildable vector index.
    - Dual-write pattern: persist canonical store first, then index in Chroma.
    - Hot memory is strictly session-aware; cold memory persists cross-session.
    - Zero semantic heuristics: no auto-inference of importance, no keyword triggers.
    """

    def __init__(self, store: Any, index: Any) -> None:
        self._store = store
        self._index = index
        from .memory_summarizer import MemorySummarizer
        self._summarizer = MemorySummarizer(self)

    @property
    def store(self) -> Any:
        return self._store

    @property
    def index(self) -> Any:
        return self._index

    @property
    def summarizer(self) -> Any:
        return self._summarizer

    def should_summarize_session(self, session_id: str) -> bool:
        """Check if the session has reached or exceeded MEMORY_HOT_SUMMARY_THRESHOLD."""
        return self._summarizer.should_summarize_session(session_id)

    def add_hot_memory(self, record: MemoryRecord) -> MemoryRecord:
        """Persist and index a session-scoped hot memory record.

        Dual-writes to canonical store and Chroma hot collection.
        """
        if not record.content or not str(record.content).strip():
            raise MemoryManagerError("content is required for hot memory")
        if record.memory_type != "hot":
            raise MemoryManagerError(
                f"add_hot_memory requires memory_type='hot', got {record.memory_type!r}"
            )
        if not record.session_id or not str(record.session_id).strip():
            raise MemoryManagerError("session_id is required for hot memories")

        persisted = self._store.create(record)
        try:
            self._index.add_memory(persisted)
        except Exception as exc:
            logger.error("Chroma indexing failed for hot memory %s: %s", persisted.memory_id, exc)
            raise MemoryManagerError(f"Failed to index hot memory in Chroma: {exc}") from exc
        return persisted

    def add_cold_memory(self, record: MemoryRecord) -> MemoryRecord:
        """Persist and index a durable cross-session cold memory record.

        Dual-writes to canonical store and Chroma cold collection.
        """
        if not record.content or not str(record.content).strip():
            raise MemoryManagerError("content is required for cold memory")
        if record.memory_type != "cold":
            raise MemoryManagerError(
                f"add_cold_memory requires memory_type='cold', got {record.memory_type!r}"
            )

        persisted = self._store.create(record)
        try:
            self._index.add_memory(persisted)
        except Exception as exc:
            logger.error("Chroma indexing failed for cold memory %s: %s", persisted.memory_id, exc)
            raise MemoryManagerError(f"Failed to index cold memory in Chroma: {exc}") from exc
        return persisted

    def search_hot_memory(
        self,
        query: str,
        session_id: str,
        top_k: int = MEMORY_DEFAULT_TOP_K,
        *,
        category: str | None = None,
        is_summary: bool | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[MemorySearchResult]:
        """Session-scoped selective semantic search over hot memory."""
        if not session_id or not str(session_id).strip():
            raise MemoryManagerError("session_id is required for search_hot_memory")

        if top_k is None:
            top_k = MEMORY_DEFAULT_TOP_K

        return self._index.search_memory(
            query=query,
            memory_type="hot",
            top_k=top_k,
            session_id=session_id,
            category=category,
            is_summary=is_summary,
            min_importance=min_importance,
            max_importance=max_importance,
            created_after=created_after,
            created_before=created_before,
        )

    def search_cold_memory(
        self,
        query: str,
        top_k: int = MEMORY_DEFAULT_TOP_K,
        *,
        category: str | None = None,
        is_summary: bool | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[MemorySearchResult]:
        """Selective semantic search over cold memory across all sessions."""
        if not query or not str(query).strip():
            raise MemoryManagerError("query is required for search_cold_memory")

        if top_k is None:
            top_k = MEMORY_DEFAULT_TOP_K

        return self._index.search_memory(
            query=query,
            memory_type="cold",
            top_k=top_k,
            session_id=None,
            category=category,
            is_summary=is_summary,
            min_importance=min_importance,
            max_importance=max_importance,
            created_after=created_after,
            created_before=created_before,
        )

    def get_recent_hot_memory(
        self,
        session_id: str,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Return recent hot memories for a session from canonical store (newest first)."""
        if not session_id or not str(session_id).strip():
            raise MemoryManagerError("session_id is required for get_recent_hot_memory")

        records = self._store.list_hot(session_id=session_id)
        return records[:limit]

    def get_memory(self, memory_id: str, memory_type: str) -> MemoryRecord | None:
        """Fetch a single record from canonical store by ID and type."""
        return self._store.get(memory_id, memory_type)

    def update_memory(self, record: MemoryRecord) -> MemoryRecord:
        """Update an existing memory record in both canonical store and Chroma.

        Invariants enforced:
        - record must be a valid MemoryRecord instance.
        - memory_type must be 'hot' or 'cold'.
        - memory_id must start with 'mem_'.
        - content must be a non-empty string.
        - importance and confidence must be in [0.0, 1.0].
        - Record must already exist in the canonical store.
        - memory_type cannot change via update (must use promotion flow).
        - created_at is immutable and cannot be changed.
        - session_id is immutable for hot memories.
        - Canonical JSON on disk is updated first; then Chroma is updated.
        """
        if not isinstance(record, MemoryRecord):
            raise MemoryManagerError(
                f"record must be a MemoryRecord instance, got {type(record).__name__}"
            )
        if record.memory_type not in ("hot", "cold"):
            raise MemoryManagerError(
                f"memory_type must be 'hot' or 'cold', got {record.memory_type!r}"
            )
        clean_id = record.memory_id.strip() if (record.memory_id and isinstance(record.memory_id, str)) else ""
        if not clean_id or not clean_id.startswith("mem_"):
            raise MemoryManagerError(
                f"memory_id must start with 'mem_', got {record.memory_id!r}"
            )
        if not record.content or not str(record.content).strip():
            raise MemoryManagerError("content must be a non-empty string")
        if not isinstance(record.importance, (int, float)) or record.importance < 0.0 or record.importance > 1.0:
            raise MemoryManagerError(f"importance must be in [0.0, 1.0], got {record.importance}")
        if not isinstance(record.confidence, (int, float)) or record.confidence < 0.0 or record.confidence > 1.0:
            raise MemoryManagerError(f"confidence must be in [0.0, 1.0], got {record.confidence}")

        existing = self._store.get(clean_id, record.memory_type)
        if existing is None:
            raise MemoryManagerError(
                f"Memory not found for update: {clean_id!r} ({record.memory_type})"
            )
        if existing.memory_type != record.memory_type:
            raise MemoryManagerError(
                "memory_type cannot change via update; use promotion flow"
            )
        if record.created_at != existing.created_at:
            raise MemoryManagerError(
                f"created_at is immutable for memory {clean_id!r}"
            )
        if record.memory_type == "hot" and record.session_id != existing.session_id:
            raise MemoryManagerError(
                "session_id cannot be changed for hot memory"
            )

        try:
            updated = self._store.update(record)
        except Exception as exc:
            logger.error("Canonical store update failed for %s: %s", clean_id, exc)
            raise MemoryManagerError(f"Failed to update canonical memory record: {exc}") from exc

        try:
            self._index.update_memory(updated)
        except Exception as exc:
            logger.error("Chroma update failed for memory %s: %s", updated.memory_id, exc)
            raise MemoryManagerError(f"Failed to update memory in Chroma: {exc}") from exc
        return updated

    def delete_memory(self, memory_id: str, memory_type: str) -> bool:
        """Delete a memory record from both canonical store and Chroma.

        Invariants enforced:
        - memory_id must be a non-empty string starting with 'mem_'.
        - memory_type must be 'hot' or 'cold'.
        - Deletion removes from canonical JSON store on disk and Chroma vector index.
        - Returns True if record was found and deleted, False otherwise.
        """
        if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
            raise MemoryManagerError("memory_id must be a non-empty string for deletion")
        clean_id = memory_id.strip()
        if not clean_id.startswith("mem_"):
            raise MemoryManagerError(f"memory_id must start with 'mem_', got {memory_id!r}")
        if memory_type not in ("hot", "cold"):
            raise MemoryManagerError(
                f"memory_type must be 'hot' or 'cold', got {memory_type!r}"
            )

        try:
            store_deleted = self._store.delete(clean_id, memory_type)
        except Exception as exc:
            logger.error("Canonical store deletion failed for %s (%s): %s", clean_id, memory_type, exc)
            raise MemoryManagerError(f"Failed to delete canonical memory record: {exc}") from exc

        try:
            index_deleted = self._index.delete_memory(clean_id, memory_type)
        except Exception as exc:
            logger.warning("Chroma deletion failed for %s (%s): %s", clean_id, memory_type, exc)
            index_deleted = False

        return store_deleted or index_deleted

    def summarize_hot_memory(
        self,
        session_id: str,
        summary_content: str,
        parent_memory_ids: list[str],
        *,
        category: str = "summary",
        source: str = "memory_summarizer",
        importance: float = 0.5,
        confidence: float = 0.5,
    ) -> MemoryRecord:
        """Store a summary hot memory linked to parent IDs. Retains parent raw memories."""
        if not session_id or not str(session_id).strip():
            raise MemoryManagerError("session_id is required for summarize_hot_memory")
        if not summary_content or not str(summary_content).strip():
            raise MemoryManagerError("summary_content must be a non-empty string")
        if not parent_memory_ids or not isinstance(parent_memory_ids, list):
            raise MemoryManagerError("parent_memory_ids must be a non-empty list of IDs")

        for pid in parent_memory_ids:
            if not isinstance(pid, str) or not pid.strip():
                raise MemoryManagerError("parent_memory_ids must contain non-empty strings")
            parent = self._store.get(pid.strip(), "hot")
            if parent is None:
                raise MemoryManagerError(f"Parent memory {pid!r} not found in hot memory store")
            if parent.session_id != session_id.strip():
                raise MemoryManagerError(
                    f"Parent memory {pid!r} belongs to session {parent.session_id!r}, not {session_id!r}"
                )

        if not isinstance(importance, (int, float)) or importance < 0.0 or importance > 1.0:
            raise MemoryManagerError(f"importance must be in [0.0, 1.0], got {importance}")
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            raise MemoryManagerError(f"confidence must be in [0.0, 1.0], got {confidence}")

        now = iso_now()
        summary_record = MemoryRecord(
            memory_id=make_memory_id(),
            content=summary_content.strip(),
            memory_type="hot",
            category=category,
            source=source,
            session_id=session_id.strip(),
            created_at=now,
            updated_at=now,
            importance=float(importance),
            confidence=float(confidence),
            is_summary=True,
            parent_memory_ids=[p.strip() for p in parent_memory_ids],
        )
        return self.add_hot_memory(summary_record)

    def promote_to_cold(self, memory_id: str) -> MemoryRecord:
        """Promote a hot memory to durable cold storage (Gemma-driven decision).

        Atomic workflow:
        1. Validate ID format.
        2. Verify existence in hot canonical store and absence in cold store.
        3. Transition canonical JSON store from hot to cold via store.promote_record.
        4. Remove from hot Chroma collection (sage_memory_hot).
        5. Index into cold Chroma collection (sage_memory_cold).
        """
        if not memory_id or not isinstance(memory_id, str) or not memory_id.strip():
            raise MemoryManagerError("memory_id must be a non-empty string for promotion")
        if not memory_id.strip().startswith("mem_"):
            raise MemoryManagerError(f"memory_id must start with 'mem_', got {memory_id!r}")

        clean_id = memory_id.strip()

        existing = self._store.get(clean_id, "hot")
        if existing is None:
            raise MemoryManagerError(f"Hot memory not found for promotion: {clean_id!r}")

        if self._store.get(clean_id, "cold") is not None:
            raise MemoryManagerError(f"Cold memory already exists with ID: {clean_id!r}")

        now = iso_now()
        cold_record = MemoryRecord(
            memory_id=existing.memory_id,
            content=existing.content,
            memory_type="cold",
            category=existing.category,
            source=existing.source,
            session_id=existing.session_id,
            created_at=existing.created_at,
            updated_at=now,
            importance=existing.importance,
            confidence=existing.confidence,
            access_count=existing.access_count,
            last_accessed=existing.last_accessed,
            is_summary=existing.is_summary,
            parent_memory_ids=list(existing.parent_memory_ids),
        )

        # 1. Canonical store promotion (authoritative JSON transition)
        try:
            self._store.promote_record(cold_record)
        except Exception as exc:
            logger.error("Canonical store promotion failed for %s: %s", clean_id, exc)
            raise MemoryManagerError(f"Failed to promote canonical memory record: {exc}") from exc

        # 2. Chroma hot de-indexing
        try:
            self._index.delete_memory(clean_id, "hot")
        except Exception as exc:
            logger.error("Chroma hot de-indexing failed for %s: %s", clean_id, exc)
            raise MemoryManagerError(
                f"Chroma indexing failure while removing {clean_id} from hot index: {exc}"
            ) from exc

        # 3. Chroma cold indexing
        try:
            self._index.add_memory(cold_record)
        except Exception as exc:
            logger.error("Chroma cold indexing failed for %s: %s", clean_id, exc)
            raise MemoryManagerError(
                f"Chroma indexing failure while adding {clean_id} to cold index: {exc}"
            ) from exc

        return cold_record

    def rebuild_memory_indexes(self, *, ignore_corrupt: bool = True) -> dict[str, Any]:
        """Rebuild Chroma memory collections from canonical store. Phase 14."""
        if self._store is None or self._index is None:
            raise MemoryManagerError("MemoryManager requires configured store and index for rebuild")
        now = iso_now()
        counts = self._index.rebuild_from_store(self._store, ignore_corrupt=ignore_corrupt)
        return {
            "status": "success",
            "hot": counts["hot"],
            "cold": counts["cold"],
            "total": counts["total"],
            "rebuilt_at": now,
        }

    def sync_indexes(self, force: bool = False) -> dict[str, Any]:
        """Verify index consistency against canonical store using ID-set equality.

        Consistency mandate:
            canonical_hot_ids == chroma_hot_ids
            AND
            canonical_cold_ids == chroma_cold_ids

        Detects:
            - Missing Chroma records
            - Orphaned Chroma records
            - Wrong-tier records
            - Same-count/different-ID situations

        If desynchronized or force=True, rebuilds Chroma vector indexes.

        Returns:
            {"status": "in_sync" | "rebuilt", "hot": int, "cold": int, "total": int}
        """
        if self._store is None or self._index is None:
            raise MemoryManagerError("MemoryManager requires configured store and index for sync")

        canonical_hot_ids = {r.memory_id for r in self._store.list_hot(ignore_corrupt=True)}
        canonical_cold_ids = {r.memory_id for r in self._store.list_cold(ignore_corrupt=True)}

        chroma_hot_ids = set(self._index.get_indexed_memory_ids("hot"))
        chroma_cold_ids = set(self._index.get_indexed_memory_ids("cold"))

        hot_in_sync = (canonical_hot_ids == chroma_hot_ids)
        cold_in_sync = (canonical_cold_ids == chroma_cold_ids)

        if force or not hot_in_sync or not cold_in_sync:
            if not hot_in_sync:
                missing_hot = canonical_hot_ids - chroma_hot_ids
                orphaned_hot = chroma_hot_ids - canonical_hot_ids
                logger.info(
                    "Hot memory index desynchronization detected: %d in store, %d in Chroma (missing=%s, orphaned=%s). Rebuilding...",
                    len(canonical_hot_ids), len(chroma_hot_ids), missing_hot, orphaned_hot,
                )
            if not cold_in_sync:
                missing_cold = canonical_cold_ids - chroma_cold_ids
                orphaned_cold = chroma_cold_ids - canonical_cold_ids
                logger.info(
                    "Cold memory index desynchronization detected: %d in store, %d in Chroma (missing=%s, orphaned=%s). Rebuilding...",
                    len(canonical_cold_ids), len(chroma_cold_ids), missing_cold, orphaned_cold,
                )
            if force:
                logger.info("Forced memory index rebuild triggered.")

            rebuild_res = self.rebuild_memory_indexes()
            return {
                "status": "rebuilt",
                "hot": rebuild_res["hot"],
                "cold": rebuild_res["cold"],
                "total": rebuild_res["total"],
                "rebuilt_at": rebuild_res["rebuilt_at"],
            }

        return {
            "status": "in_sync",
            "hot": len(chroma_hot_ids),
            "cold": len(chroma_cold_ids),
            "total": len(chroma_hot_ids) + len(chroma_cold_ids),
        }
