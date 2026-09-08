"""
sage_memory/memory_index.py
Chroma-backed semantic index for hot and cold memory collections.

Uses sage_memory_hot / sage_memory_cold — never merged with document RAG.
Embeddings are computed externally via injected EmbeddingService.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import chromadb

from sage_document_db.utils import decode_list_field, encode_list_field

from .config import (
    CHROMA_ROOT,
    EMBEDDING_MODEL,
    MEMORY_COLD_COLLECTION,
    MEMORY_DEFAULT_TOP_K,
    MEMORY_HOT_COLLECTION,
)
from .memory_models import MemoryRecord, MemorySearchResult, MemoryType

logger = logging.getLogger(__name__)

MEMORY_INDEX_CONFIG_FILE = "memory_index_config.json"


class MemoryIndexError(Exception):
    """Raised when a memory Chroma index operation fails."""


def _check_memory_index_config(chroma_root: Path) -> None:
    """Ensure memory index embedding model matches current config."""
    config_path = chroma_root / MEMORY_INDEX_CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "embedding_model": EMBEDDING_MODEL,
        "hot_collection": MEMORY_HOT_COLLECTION,
        "cold_collection": MEMORY_COLD_COLLECTION,
    }

    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return

    with open(config_path, "r", encoding="utf-8") as f:
        stored = json.load(f)

    if stored.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Memory index was built with embedding model "
            f"'{stored.get('embedding_model')}' but current config uses "
            f"'{EMBEDDING_MODEL}'. Rebuild memory indexes."
        )


def _parse_iso_to_epoch(val: str | float | int) -> float:
    """Parse an ISO-8601 timestamp string or numeric timestamp to epoch seconds.

    ChromaDB comparison operators ($gt, $gte, $lt, $lte) require numeric operands.
    """
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except ValueError:
        pass
    s = val.strip()
    if s.endswith("Z") or s.endswith("z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _record_to_metadata(record: MemoryRecord) -> dict[str, Any]:
    """Convert MemoryRecord to Chroma-safe scalar metadata."""
    return {
        "memory_id": record.memory_id,
        "memory_type": record.memory_type,
        "category": record.category,
        "source": record.source,
        "session_id": record.session_id or "",
        "created_at": record.created_at,
        "created_at_ts": _parse_iso_to_epoch(record.created_at),
        "updated_at": record.updated_at,
        "importance": float(record.importance),
        "confidence": float(record.confidence),
        "access_count": int(record.access_count),
        "last_accessed": record.last_accessed or "",
        "is_summary": bool(record.is_summary),
        "parent_memory_ids_json": encode_list_field(record.parent_memory_ids),
    }


def _normalize_memory_results(
    raw: dict,
    *,
    memory_type: MemoryType,
) -> list[MemorySearchResult]:
    """Convert Chroma nested-array results to MemorySearchResult list."""
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results: list[MemorySearchResult] = []
    for memory_id, text, meta, dist in zip(ids, docs, metas, distances):
        meta = meta or {}
        session_raw = meta.get("session_id") or ""
        results.append(
            MemorySearchResult(
                memory_id=memory_id,
                content=text or "",
                memory_type=memory_type,
                category=meta.get("category", ""),
                distance=float(dist),
                session_id=session_raw or None,
                importance=float(meta.get("importance", 0.5)),
                confidence=float(meta.get("confidence", 0.5)),
                is_summary=bool(meta.get("is_summary", False)),
                created_at=meta.get("created_at"),
                source=meta.get("source"),
            )
        )
    return results


def _build_where_filter(
    *,
    session_id: str | None = None,
    category: str | None = None,
    is_summary: bool | None = None,
    min_importance: float | None = None,
    max_importance: float | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> dict[str, Any] | None:
    """Build a Chroma where clause from optional metadata filters."""
    clauses: list[dict[str, Any]] = []

    if session_id is not None:
        clauses.append({"session_id": {"$eq": session_id}})
    if category is not None:
        clauses.append({"category": {"$eq": category}})
    if is_summary is not None:
        clauses.append({"is_summary": {"$eq": is_summary}})
    if min_importance is not None:
        clauses.append({"importance": {"$gte": float(min_importance)}})
    if max_importance is not None:
        clauses.append({"importance": {"$lte": float(max_importance)}})
    if created_after is not None:
        clauses.append({"created_at_ts": {"$gte": _parse_iso_to_epoch(created_after)}})
    if created_before is not None:
        clauses.append({"created_at_ts": {"$lte": _parse_iso_to_epoch(created_before)}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class MemoryChromaStore:
    """Disposable Chroma index for memory — rebuildable from MemoryStore."""

    def __init__(
        self,
        chroma_root: str | Path,
        embedding_service: Any,
        hot_collection: str = MEMORY_HOT_COLLECTION,
        cold_collection: str = MEMORY_COLD_COLLECTION,
    ) -> None:
        self._chroma_root = Path(chroma_root)
        self._embedding_service = embedding_service
        self._hot_collection_name = hot_collection
        self._cold_collection_name = cold_collection

        _check_memory_index_config(self._chroma_root)

        self._client = chromadb.PersistentClient(path=str(self._chroma_root))
        self._hot = self._get_or_create_collection(hot_collection)
        self._cold = self._get_or_create_collection(cold_collection)

    def _get_or_create_collection(self, name: str):
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            return self._client.get_or_create_collection(name=name)

    def _collection_for_type(self, memory_type: str):
        if memory_type == "hot":
            return self._hot
        if memory_type == "cold":
            return self._cold
        raise MemoryIndexError(
            f"memory_type must be 'hot' or 'cold', got {memory_type!r}"
        )

    def _collection_name_for_type(self, memory_type: str) -> str:
        if memory_type == "hot":
            return self._hot_collection_name
        if memory_type == "cold":
            return self._cold_collection_name
        raise MemoryIndexError(
            f"memory_type must be 'hot' or 'cold', got {memory_type!r}"
        )

    # ------------------------------------------------------------------
    # Index operations
    # ------------------------------------------------------------------

    def add_memory(self, record: MemoryRecord) -> str:
        """Index one memory record. Returns memory_id."""
        if record.memory_type not in ("hot", "cold"):
            raise MemoryIndexError(
                f"Cannot index memory_type {record.memory_type!r}"
            )
        collection = self._collection_for_type(record.memory_type)
        vector = self._embedding_service.embed_documents([record.content])[0]
        metadata = _record_to_metadata(record)
        collection.upsert(
            ids=[record.memory_id],
            embeddings=[vector],
            documents=[record.content],
            metadatas=[metadata],
        )
        return record.memory_id

    def search_memory(
        self,
        query: str,
        memory_type: str,
        top_k: int = MEMORY_DEFAULT_TOP_K,
        *,
        session_id: str | None = None,
        category: str | None = None,
        is_summary: bool | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
    ) -> list[MemorySearchResult]:
        """Semantic search with metadata filters applied at query time."""
        if top_k is None:
            top_k = MEMORY_DEFAULT_TOP_K
        if top_k <= 0:
            return []

        collection = self._collection_for_type(memory_type)
        count = collection.count()
        if count == 0:
            return []

        query_vector = self._embedding_service.embed_query(query)
        where = _build_where_filter(
            session_id=session_id,
            category=category,
            is_summary=is_summary,
            min_importance=min_importance,
            max_importance=max_importance,
            created_after=created_after,
            created_before=created_before,
        )

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [query_vector],
            "n_results": min(top_k, count),
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        try:
            raw = collection.query(**query_kwargs)
        except Exception as exc:
            raise MemoryIndexError(f"Memory search failed: {exc}") from exc

        results = _normalize_memory_results(raw, memory_type=memory_type)  # type: ignore[arg-type]
        results.sort(key=lambda r: r.distance)
        return results[:top_k]

    def update_memory(self, record: MemoryRecord) -> None:
        """Re-index an updated memory record (upsert)."""
        self.add_memory(record)

    def delete_memory(self, memory_id: str, memory_type: str) -> bool:
        """Remove a memory from the Chroma index."""
        collection = self._collection_for_type(memory_type)
        try:
            existing = collection.get(ids=[memory_id])
            ids = existing.get("ids") or []
            if not ids:
                return False
            collection.delete(ids=[memory_id])
            return True
        except Exception as exc:
            logger.warning("delete_memory failed for %s: %s", memory_id, exc)
            return False

    def get_indexed_memory_ids(self, memory_type: str) -> list[str]:
        """Return all indexed memory IDs for a collection (testing/rebuild aid)."""
        collection = self._collection_for_type(memory_type)
        if collection.count() == 0:
            return []
        raw = collection.get(include=[])
        return list(raw.get("ids") or [])

    def count(self, memory_type: str) -> int:
        """Return indexed record count for hot or cold collection."""
        return self._collection_for_type(memory_type).count()

    def rebuild_from_store(self, store: Any, *, ignore_corrupt: bool = True) -> dict[str, int]:
        """Rebuild both memory collections from the canonical store.

        Workflow:
        1. Reset and wipe existing hot and cold Chroma collections.
        2. Read all hot records from canonical store (including summaries).
        3. Read all cold records from canonical store.
        4. Re-index each record with embeddings and metadata.
        5. Return index counts: {"hot": int, "cold": int, "total": int}.
        """
        # 1. Reset hot collection in-place (wiping stale records without triggering OS segment recreation race)
        try:
            hot_existing = self.get_indexed_memory_ids("hot")
            if hot_existing:
                self._hot.delete(ids=hot_existing)
        except Exception:
            try:
                self._client.delete_collection(name=self._hot_collection_name)
            except Exception:
                pass
            self._hot = self._get_or_create_collection(self._hot_collection_name)

        # 2. Reset cold collection in-place
        try:
            cold_existing = self.get_indexed_memory_ids("cold")
            if cold_existing:
                self._cold.delete(ids=cold_existing)
        except Exception:
            try:
                self._client.delete_collection(name=self._cold_collection_name)
            except Exception:
                pass
            self._cold = self._get_or_create_collection(self._cold_collection_name)

        # 3. Read records from canonical store
        try:
            hot_records = store.list_hot(ignore_corrupt=ignore_corrupt)
        except TypeError:
            hot_records = store.list_hot()

        try:
            cold_records = store.list_cold(ignore_corrupt=ignore_corrupt)
        except TypeError:
            cold_records = store.list_cold()

        # 4. Re-index hot records in batches
        hot_count = 0
        if hot_records:
            chunk_size = 100
            for i in range(0, len(hot_records), chunk_size):
                chunk = hot_records[i : i + chunk_size]
                texts = [r.content for r in chunk]
                vectors = self._embedding_service.embed_documents(texts)
                metadatas = [_record_to_metadata(r) for r in chunk]
                ids = [r.memory_id for r in chunk]
                self._hot.upsert(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=metadatas,
                )
                hot_count += len(ids)

        # 5. Re-index cold records in batches
        cold_count = 0
        if cold_records:
            chunk_size = 100
            for i in range(0, len(cold_records), chunk_size):
                chunk = cold_records[i : i + chunk_size]
                texts = [r.content for r in chunk]
                vectors = self._embedding_service.embed_documents(texts)
                metadatas = [_record_to_metadata(r) for r in chunk]
                ids = [r.memory_id for r in chunk]
                self._cold.upsert(
                    ids=ids,
                    embeddings=vectors,
                    documents=texts,
                    metadatas=metadatas,
                )
                cold_count += len(ids)

        logger.info(
            "Memory index rebuild complete: %d hot, %d cold (%d total)",
            hot_count, cold_count, hot_count + cold_count,
        )
        return {
            "hot": hot_count,
            "cold": cold_count,
            "total": hot_count + cold_count,
        }

    def to_dict(self) -> dict[str, Any]:
        """Debug serialization of index configuration."""
        return {
            "chroma_root": str(self._chroma_root),
            "hot_collection": self._hot_collection_name,
            "cold_collection": self._cold_collection_name,
            "hot_count": self.count("hot"),
            "cold_count": self.count("cold"),
        }
