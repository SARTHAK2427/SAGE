"""
sage_memory/memory_store.py
Canonical JSON-on-disk memory store — source of truth.

Layout (under store_root):
    hot/{memory_id}.json
    cold/{memory_id}.json

Chroma is a disposable search index; this store is rebuildable truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sage_document_db.utils import safe_mkdir

from .memory_models import MemoryRecord, MemoryValidationError, MemoryType

logger = logging.getLogger(__name__)


class MemoryStoreError(Exception):
    """Raised when a canonical store operation fails."""


class MemoryStore:
    """Canonical memory persistence under store_root/hot and store_root/cold."""

    def __init__(self, store_root: str | Path) -> None:
        self._store_root = Path(store_root)
        self._hot_dir = self._store_root / "hot"
        self._cold_dir = self._store_root / "cold"
        safe_mkdir(self._hot_dir)
        safe_mkdir(self._cold_dir)

    @property
    def store_root(self) -> Path:
        return self._store_root

    @property
    def hot_dir(self) -> Path:
        return self._hot_dir

    @property
    def cold_dir(self) -> Path:
        return self._cold_dir

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, record: MemoryRecord) -> MemoryRecord:
        """Persist a new memory record. Raises MemoryStoreError if ID exists."""
        self._ensure_type_dir(record.memory_type)
        if self._exists_anywhere(record.memory_id):
            raise MemoryStoreError(
                f"Memory already exists: {record.memory_id!r}"
            )
        self._write_record(record)
        return record

    def get(self, memory_id: str, memory_type: str) -> MemoryRecord | None:
        """Load one memory by ID and type. Returns None if not found."""
        path = self._record_path(memory_id, memory_type)
        if not path.exists():
            return None
        return self._read_record(path, expected_type=memory_type)

    def update(self, record: MemoryRecord) -> MemoryRecord:
        """Update an existing memory record in place (stable memory_id)."""
        path = self._record_path(record.memory_id, record.memory_type)
        if not path.exists():
            raise MemoryStoreError(
                f"Memory not found for update: {record.memory_id!r} ({record.memory_type})"
            )
        existing = self._read_record(path, expected_type=record.memory_type)
        if existing.memory_type != record.memory_type:
            raise MemoryStoreError(
                "memory_type cannot change via update; use promotion flow"
            )
        self._write_record(record)
        return record

    def delete(self, memory_id: str, memory_type: str) -> bool:
        """Delete a memory record from the canonical store."""
        path = self._record_path(memory_id, memory_type)
        if not path.exists():
            return False
        path.unlink()
        return True

    def promote_record(self, record: MemoryRecord) -> MemoryRecord:
        """Move a record from hot to cold canonical store safely.

        Guarantees:
        - Source hot record must exist.
        - Destination cold record must not exist.
        - Cold record is written before hot record is removed.
        - If writing cold record fails, hot record is preserved intact.
        - If removing hot record fails, cold file is cleaned up so hot remains authoritative.
        """
        if record.memory_type != "cold":
            raise MemoryStoreError(
                f"Promoted record must have memory_type='cold', got {record.memory_type!r}"
            )

        hot_path = self._record_path(record.memory_id, "hot")
        cold_path = self._record_path(record.memory_id, "cold")

        if not hot_path.exists():
            raise MemoryStoreError(
                f"Hot memory not found for promotion: {record.memory_id!r}"
            )
        if cold_path.exists():
            raise MemoryStoreError(
                f"Cold memory already exists for ID: {record.memory_id!r}"
            )

        self._ensure_type_dir("cold")
        # Write cold file first (atomic .tmp replace)
        try:
            self._write_record(record)
        except Exception as exc:
            if cold_path.exists():
                try:
                    cold_path.unlink()
                except Exception:
                    pass
            raise MemoryStoreError(
                f"Failed to write cold memory record for {record.memory_id}: {exc}"
            ) from exc

        # Once cold is verified written, remove hot file
        try:
            hot_path.unlink()
        except Exception as exc:
            logger.warning(
                "Hot memory removal failed after cold write for %s: %s",
                record.memory_id,
                exc,
            )
            # Clean up cold so hot remains primary and retryable
            if cold_path.exists():
                try:
                    cold_path.unlink()
                except Exception:
                    pass
            raise MemoryStoreError(
                f"Failed to remove hot record after cold write for {record.memory_id}: {exc}"
            ) from exc

        return record


    # ------------------------------------------------------------------
    # Listing
    # ------------------------------------------------------------------

    def list_hot(
        self, session_id: str | None = None, *, ignore_corrupt: bool = False
    ) -> list[MemoryRecord]:
        """List hot memories, optionally filtered by session_id."""
        records = self._load_all_in_dir(
            self._hot_dir, expected_type="hot", ignore_corrupt=ignore_corrupt
        )
        if session_id is not None:
            records = [r for r in records if r.session_id == session_id]
        return self._sort_records(records)

    def list_cold(self, *, ignore_corrupt: bool = False) -> list[MemoryRecord]:
        """List all cold memories."""
        records = self._load_all_in_dir(
            self._cold_dir, expected_type="cold", ignore_corrupt=ignore_corrupt
        )
        return self._sort_records(records)

    def list_all(self, memory_type: str) -> list[MemoryRecord]:
        """List all memories of a given type."""
        if memory_type == "hot":
            return self.list_hot()
        if memory_type == "cold":
            return self.list_cold()
        raise MemoryStoreError(
            f"memory_type must be 'hot' or 'cold', got {memory_type!r}"
        )

    def count_hot(self, session_id: str | None = None) -> int:
        """Count hot memories, optionally scoped to a session."""
        return len(self.list_hot(session_id=session_id))

    def exists(self, memory_id: str, memory_type: str) -> bool:
        """Return True if a memory file exists for the given id and type."""
        return self._record_path(memory_id, memory_type).exists()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dir_for_type(self, memory_type: str) -> Path:
        if memory_type == "hot":
            return self._hot_dir
        if memory_type == "cold":
            return self._cold_dir
        raise MemoryStoreError(
            f"memory_type must be 'hot' or 'cold', got {memory_type!r}"
        )

    def _ensure_type_dir(self, memory_type: str) -> Path:
        return safe_mkdir(self._dir_for_type(memory_type))

    def _record_path(self, memory_id: str, memory_type: str) -> Path:
        return self._dir_for_type(memory_type) / f"{memory_id}.json"

    def _exists_anywhere(self, memory_id: str) -> bool:
        return (
            (self._hot_dir / f"{memory_id}.json").exists()
            or (self._cold_dir / f"{memory_id}.json").exists()
        )

    def _write_record(self, record: MemoryRecord) -> None:
        path = self._record_path(record.memory_id, record.memory_type)
        tmp_path = path.with_suffix(".json.tmp")
        payload = record.to_dict()
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
        tmp_path.replace(path)

    def _read_record(
        self, path: Path, *, expected_type: MemoryType | None = None
    ) -> MemoryRecord:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            record = MemoryRecord.from_dict(data)
        except (json.JSONDecodeError, MemoryValidationError, TypeError) as exc:
            raise MemoryStoreError(
                f"Corrupt or invalid memory file: {path}"
            ) from exc

        if expected_type is not None and record.memory_type != expected_type:
            raise MemoryStoreError(
                f"Memory type mismatch in {path.name}: "
                f"expected {expected_type!r}, got {record.memory_type!r}"
            )
        return record

    def _load_all_in_dir(
        self, directory: Path, *, expected_type: MemoryType, ignore_corrupt: bool = False
    ) -> list[MemoryRecord]:
        if not directory.exists():
            return []
        records: list[MemoryRecord] = []
        for path in sorted(directory.glob("*.json")):
            if path.name.endswith(".tmp"):
                continue
            try:
                records.append(self._read_record(path, expected_type=expected_type))
            except Exception as exc:
                if ignore_corrupt:
                    logger.warning("Skipping corrupt or invalid memory file %s: %s", path, exc)
                    continue
                raise
        return records

    @staticmethod
    def _sort_records(records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Deterministic ordering: newest created_at first, then memory_id."""
        return sorted(
            records,
            key=lambda r: (r.created_at, r.memory_id),
            reverse=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Debug serialization of store configuration."""
        return {
            "store_root": str(self._store_root),
            "hot_dir": str(self._hot_dir),
            "cold_dir": str(self._cold_dir),
            "hot_count": len(list(self._hot_dir.glob("*.json"))),
            "cold_count": len(list(self._cold_dir.glob("*.json"))),
        }
