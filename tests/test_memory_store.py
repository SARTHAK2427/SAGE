"""
Tests for sage_memory/memory_store.py — Phase 2 canonical store CRUD.
"""

from __future__ import annotations

import json

import pytest

from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore, MemoryStoreError


def _hot(**overrides) -> MemoryRecord:
    base = {
        "memory_id": make_memory_id(),
        "content": "Use implementation X for this task.",
        "memory_type": "hot",
        "category": "task",
        "source": "user_message",
        "session_id": "sess_a",
        "created_at": "2026-09-08T12:00:00+00:00",
        "updated_at": "2026-09-08T12:00:00+00:00",
    }
    base.update(overrides)
    return MemoryRecord(**base)


def _cold(**overrides) -> MemoryRecord:
    base = {
        "memory_id": make_memory_id(),
        "content": "User prefers FastAPI for backend projects.",
        "memory_type": "cold",
        "category": "preference",
        "source": "user_message",
        "session_id": None,
        "created_at": "2026-09-08T12:00:00+00:00",
        "updated_at": "2026-09-08T12:00:00+00:00",
    }
    base.update(overrides)
    return MemoryRecord(**base)


class TestMemoryStoreCRUD:
    def test_create_and_get_hot(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        record = _hot()
        store.create(record)
        loaded = store.get(record.memory_id, "hot")
        assert loaded is not None
        assert loaded.content == record.content
        assert loaded.session_id == "sess_a"

    def test_create_and_get_cold(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        record = _cold()
        store.create(record)
        loaded = store.get(record.memory_id, "cold")
        assert loaded is not None
        assert loaded.memory_type == "cold"
        assert loaded.session_id is None

    def test_persistence_across_instances(self, tmp_memory_root):
        record = _cold()
        store1 = MemoryStore(tmp_memory_root)
        store1.create(record)

        store2 = MemoryStore(tmp_memory_root)
        loaded = store2.get(record.memory_id, "cold")
        assert loaded is not None
        assert loaded.content == record.content

    def test_json_file_on_disk(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        record = _hot(memory_id="mem_test000001")
        store.create(record)

        path = tmp_memory_root / "hot" / "mem_test000001.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["memory_id"] == "mem_test000001"
        assert data["memory_type"] == "hot"

    def test_duplicate_create_rejected(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        record = _hot()
        store.create(record)
        with pytest.raises(MemoryStoreError, match="already exists"):
            store.create(record)

    def test_duplicate_id_across_types_rejected(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        mid = make_memory_id()
        store.create(_hot(memory_id=mid))
        with pytest.raises(MemoryStoreError, match="already exists"):
            store.create(_cold(memory_id=mid))

    def test_get_missing_returns_none(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        assert store.get("mem_doesnotexist", "hot") is None

    def test_update_preserves_id(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        record = _hot()
        store.create(record)

        updated = MemoryRecord.from_dict({
            **record.to_dict(),
            "content": "Updated task instruction.",
            "updated_at": "2026-09-08T13:00:00+00:00",
        })
        store.update(updated)
        loaded = store.get(record.memory_id, "hot")
        assert loaded.content == "Updated task instruction."
        assert loaded.memory_id == record.memory_id
        assert loaded.created_at == record.created_at

    def test_update_missing_raises(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        with pytest.raises(MemoryStoreError, match="not found"):
            store.update(_hot())

    def test_delete_existing(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        record = _hot()
        store.create(record)
        assert store.delete(record.memory_id, "hot") is True
        assert store.get(record.memory_id, "hot") is None

    def test_delete_missing_returns_false(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        assert store.delete("mem_missing", "hot") is False

    def test_exists(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        record = _cold()
        assert store.exists(record.memory_id, "cold") is False
        store.create(record)
        assert store.exists(record.memory_id, "cold") is True


class TestMemoryStoreListing:
    def test_list_hot_all(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        store.create(_hot(session_id="sess_a", created_at="2026-09-08T10:00:00+00:00"))
        store.create(_hot(session_id="sess_b", created_at="2026-09-08T11:00:00+00:00"))
        all_hot = store.list_hot()
        assert len(all_hot) == 2

    def test_list_hot_filtered_by_session(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        r_a1 = _hot(session_id="sess_a", content="Task A1")
        r_a2 = _hot(session_id="sess_a", content="Task A2")
        r_b1 = _hot(session_id="sess_b", content="Task B1")
        store.create(r_a1)
        store.create(r_a2)
        store.create(r_b1)

        sess_a = store.list_hot(session_id="sess_a")
        sess_b = store.list_hot(session_id="sess_b")

        assert len(sess_a) == 2
        assert len(sess_b) == 1
        assert {r.content for r in sess_a} == {"Task A1", "Task A2"}
        assert sess_b[0].content == "Task B1"

    def test_session_isolation(self, tmp_memory_root):
        """Session A hot memory never appears in session B listing."""
        store = MemoryStore(tmp_memory_root)
        store.create(_hot(session_id="sess_a", content="Secret for A"))
        store.create(_hot(session_id="sess_b", content="Secret for B"))

        sess_b = store.list_hot(session_id="sess_b")
        assert all(r.session_id == "sess_b" for r in sess_b)
        assert not any(r.content == "Secret for A" for r in sess_b)

    def test_list_cold(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        store.create(_cold(content="Prefers FastAPI"))
        store.create(_cold(content="Name is Ojasvi"))
        cold = store.list_cold()
        assert len(cold) == 2
        assert {r.memory_type for r in cold} == {"cold"}

    def test_list_all_by_type(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        store.create(_hot())
        store.create(_cold())
        assert len(store.list_all("hot")) == 1
        assert len(store.list_all("cold")) == 1

    def test_list_all_invalid_type(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        with pytest.raises(MemoryStoreError, match="memory_type must be"):
            store.list_all("warm")

    def test_count_hot(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        store.create(_hot(session_id="sess_a"))
        store.create(_hot(session_id="sess_a"))
        store.create(_hot(session_id="sess_b"))
        assert store.count_hot() == 3
        assert store.count_hot(session_id="sess_a") == 2
        assert store.count_hot(session_id="sess_b") == 1

    def test_list_sorted_by_created_at_desc(self, tmp_memory_root):
        store = MemoryStore(tmp_memory_root)
        older = _hot(created_at="2026-09-08T10:00:00+00:00", updated_at="2026-09-08T10:00:00+00:00")
        newer = _hot(created_at="2026-09-08T12:00:00+00:00", updated_at="2026-09-08T12:00:00+00:00")
        store.create(older)
        store.create(newer)
        ordered = store.list_hot()
        assert ordered[0].memory_id == newer.memory_id
        assert ordered[1].memory_id == older.memory_id

    def test_creates_hot_and_cold_dirs(self, tmp_path):
        root = tmp_path / "new_memory_root"
        store = MemoryStore(root)
        assert store.hot_dir.is_dir()
        assert store.cold_dir.is_dir()
