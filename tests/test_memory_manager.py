"""
Tests for sage_memory/memory_manager.py — Phase 4 MemoryManager.
"""

from __future__ import annotations

import pytest

from sage_document_db.embeddings import EmbeddingService
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager, MemoryManagerError
from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore


def _hot(**overrides) -> MemoryRecord:
    base = {
        "memory_id": make_memory_id(),
        "content": "For this task, use implementation X.",
        "memory_type": "hot",
        "category": "task",
        "source": "user_message",
        "session_id": "sess_1",
        "created_at": "2026-09-08T10:00:00+00:00",
        "updated_at": "2026-09-08T10:00:00+00:00",
        "importance": 0.6,
        "confidence": 0.9,
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
        "created_at": "2026-09-08T10:00:00+00:00",
        "updated_at": "2026-09-08T10:00:00+00:00",
        "importance": 0.8,
        "confidence": 0.9,
    }
    base.update(overrides)
    return MemoryRecord(**base)


@pytest.fixture()
def embedding_service():
    return EmbeddingService()


@pytest.fixture()
def memory_manager(tmp_memory_root, tmp_chroma_root, embedding_service):
    store = MemoryStore(tmp_memory_root)
    index = MemoryChromaStore(tmp_chroma_root, embedding_service)
    return MemoryManager(store=store, index=index)


class TestMemoryManagerAdd:
    def test_add_hot_memory_dual_writes(self, memory_manager):
        record = _hot()
        result = memory_manager.add_hot_memory(record)
        assert result.memory_id == record.memory_id

        # Verify in store
        stored = memory_manager.store.get(record.memory_id, "hot")
        assert stored is not None
        assert stored.content == record.content

        # Verify in Chroma
        assert memory_manager.index.count("hot") == 1
        assert memory_manager.index.count("cold") == 0

    def test_add_cold_memory_dual_writes(self, memory_manager):
        record = _cold()
        result = memory_manager.add_cold_memory(record)
        assert result.memory_id == record.memory_id

        # Verify in store
        stored = memory_manager.store.get(record.memory_id, "cold")
        assert stored is not None
        assert stored.content == record.content

        # Verify in Chroma
        assert memory_manager.index.count("cold") == 1
        assert memory_manager.index.count("hot") == 0

    def test_add_hot_requires_hot_type(self, memory_manager):
        cold_record = _cold()
        with pytest.raises(MemoryManagerError, match="requires memory_type='hot'"):
            memory_manager.add_hot_memory(cold_record)

    def test_add_cold_requires_cold_type(self, memory_manager):
        hot_record = _hot()
        with pytest.raises(MemoryManagerError, match="requires memory_type='cold'"):
            memory_manager.add_cold_memory(hot_record)

    def test_add_hot_requires_session_id(self, memory_manager):
        record = _hot()
        object.__setattr__(record, "session_id", None)
        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.add_hot_memory(record)


class TestMemoryManagerSearch:
    def test_search_hot_memory_session_scoped(self, memory_manager):
        r1 = _hot(content="Session 1 instruction A", session_id="sess_1")
        r2 = _hot(content="Session 2 instruction B", session_id="sess_2")
        memory_manager.add_hot_memory(r1)
        memory_manager.add_hot_memory(r2)

        results = memory_manager.search_hot_memory(
            query="instruction",
            session_id="sess_1",
            top_k=5,
        )
        assert len(results) == 1
        assert results[0].memory_id == r1.memory_id
        assert results[0].session_id == "sess_1"

    def test_search_hot_requires_session_id(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.search_hot_memory("query", session_id="")

    def test_search_cold_memory_cross_session(self, memory_manager):
        r = _cold(content="User prefers PostgreSQL over MongoDB.")
        memory_manager.add_cold_memory(r)

        results = memory_manager.search_cold_memory("database preference", top_k=5)
        assert len(results) == 1
        assert results[0].memory_id == r.memory_id
        assert results[0].content == r.content

    def test_search_with_category_filter(self, memory_manager):
        memory_manager.add_cold_memory(_cold(content="User prefers tabs", category="preference"))
        memory_manager.add_cold_memory(_cold(content="Project uses Python 3.13", category="project"))

        results = memory_manager.search_cold_memory(
            query="preference or project",
            top_k=5,
            category="preference",
        )
        assert len(results) == 1
        assert results[0].category == "preference"


class TestMemoryManagerListingAndRecent:
    def test_get_recent_hot_memory(self, memory_manager):
        for i in range(5):
            memory_manager.add_hot_memory(_hot(
                content=f"Message {i}",
                session_id="sess_1",
                created_at=f"2026-09-08T10:0{i}:00+00:00",
                updated_at=f"2026-09-08T10:0{i}:00+00:00",
            ))

        recent = memory_manager.get_recent_hot_memory(session_id="sess_1", limit=3)
        assert len(recent) == 3
        # Newest first
        assert recent[0].content == "Message 4"
        assert recent[1].content == "Message 3"
        assert recent[2].content == "Message 2"

    def test_get_recent_hot_memory_session_isolated(self, memory_manager):
        memory_manager.add_hot_memory(_hot(content="Sess 1 msg", session_id="sess_1"))
        memory_manager.add_hot_memory(_hot(content="Sess 2 msg", session_id="sess_2"))

        recent_1 = memory_manager.get_recent_hot_memory(session_id="sess_1")
        assert len(recent_1) == 1
        assert recent_1[0].content == "Sess 1 msg"

        recent_2 = memory_manager.get_recent_hot_memory(session_id="sess_2")
        assert len(recent_2) == 1
        assert recent_2[0].content == "Sess 2 msg"

    def test_get_recent_hot_requires_session_id(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.get_recent_hot_memory(session_id="")


class TestMemoryManagerUpdateAndDelete:
    def test_update_memory_syncs_store_and_index(self, memory_manager):
        record = _hot(content="Original plan")
        memory_manager.add_hot_memory(record)

        updated = MemoryRecord.from_dict({
            **record.to_dict(),
            "content": "Revised plan after consultation",
            "updated_at": "2026-09-08T11:00:00+00:00",
        })
        memory_manager.update_memory(updated)

        # Check store
        stored = memory_manager.get_memory(record.memory_id, "hot")
        assert stored.content == "Revised plan after consultation"

        # Check index
        results = memory_manager.search_hot_memory("Revised plan", session_id="sess_1", top_k=1)
        assert len(results) == 1
        assert results[0].content == "Revised plan after consultation"

    def test_delete_memory_syncs_store_and_index(self, memory_manager):
        record = _cold(content="Temporary preference to delete")
        memory_manager.add_cold_memory(record)
        assert memory_manager.index.count("cold") == 1

        deleted = memory_manager.delete_memory(record.memory_id, "cold")
        assert deleted is True

        # Check store
        assert memory_manager.get_memory(record.memory_id, "cold") is None

        # Check index
        assert memory_manager.index.count("cold") == 0

    def test_delete_invalid_type_raises(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="memory_type must be 'hot' or 'cold'"):
            memory_manager.delete_memory("mem_123", "warm")


class TestMemoryManagerSummarization:
    def test_summarize_hot_memory(self, memory_manager):
        parent_1 = memory_manager.add_hot_memory(_hot(content="Detail step 1", session_id="sess_1"))
        parent_2 = memory_manager.add_hot_memory(_hot(content="Detail step 2", session_id="sess_1"))

        summary = memory_manager.summarize_hot_memory(
            session_id="sess_1",
            summary_content="Summary of steps 1 and 2",
            parent_memory_ids=[parent_1.memory_id, parent_2.memory_id],
        )

        assert summary.is_summary is True
        assert summary.category == "summary"
        assert summary.parent_memory_ids == [parent_1.memory_id, parent_2.memory_id]
        assert summary.session_id == "sess_1"

        # Parent memories are preserved (no deletion yet)
        assert memory_manager.get_memory(parent_1.memory_id, "hot") is not None
        assert memory_manager.get_memory(parent_2.memory_id, "hot") is not None

        # Summary is indexed and searchable
        results = memory_manager.search_hot_memory(
            "Summary of steps",
            session_id="sess_1",
            top_k=5,
            is_summary=True,
        )
        assert len(results) == 1
        assert results[0].memory_id == summary.memory_id

    def test_summarize_validation(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="session_id is required"):
            memory_manager.summarize_hot_memory(
                session_id="",
                summary_content="content",
                parent_memory_ids=["mem_1"],
            )

        with pytest.raises(MemoryManagerError, match="summary_content must be a non-empty string"):
            memory_manager.summarize_hot_memory(
                session_id="sess_1",
                summary_content="",
                parent_memory_ids=["mem_1"],
            )

        with pytest.raises(MemoryManagerError, match="parent_memory_ids must be a non-empty list"):
            memory_manager.summarize_hot_memory(
                session_id="sess_1",
                summary_content="content",
                parent_memory_ids=[],
            )


class TestMemoryManagerPromotion:
    def test_promote_to_cold_moves_record(self, memory_manager):
        hot_rec = _hot(content="User confirmed architecture preference for FastAPI.")
        memory_manager.add_hot_memory(hot_rec)

        assert memory_manager.index.count("hot") == 1
        assert memory_manager.index.count("cold") == 0
        assert memory_manager.get_memory(hot_rec.memory_id, "hot") is not None

        # Promote
        promoted = memory_manager.promote_to_cold(hot_rec.memory_id)

        assert promoted.memory_id == hot_rec.memory_id
        assert promoted.memory_type == "cold"
        assert promoted.content == hot_rec.content

        # Removed from hot store & index
        assert memory_manager.get_memory(hot_rec.memory_id, "hot") is None
        assert memory_manager.index.count("hot") == 0

        # Present in cold store & index
        assert memory_manager.get_memory(hot_rec.memory_id, "cold") is not None
        assert memory_manager.index.count("cold") == 1

        # Searchable in cold memory
        results = memory_manager.search_cold_memory("FastAPI architecture", top_k=1)
        assert len(results) == 1
        assert results[0].memory_id == hot_rec.memory_id
        assert results[0].memory_type == "cold"

    def test_promote_nonexistent_raises(self, memory_manager):
        with pytest.raises(MemoryManagerError, match="Hot memory not found for promotion"):
            memory_manager.promote_to_cold("mem_nonexistent")
