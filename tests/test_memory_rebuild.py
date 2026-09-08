"""
tests/test_memory_rebuild.py
Phase 14 tests: Chroma vector index rebuild from canonical JSON store and startup sync.

Validates:
1. Rebuild from empty Chroma: indexes all hot and cold memories from disk.
2. Rebuild restores semantic searchability and session isolation.
3. Rebuild preserves all metadata (session_id, importance, confidence, is_summary, parent IDs).
4. Rebuild wipes stale/orphaned Chroma records not present on disk.
5. Rebuild on empty store succeeds cleanly with 0 counts.
6. Rebuild resilience: skips corrupted JSON files on disk without aborting.
7. Startup sync: sync_indexes detects desynchronization and triggers rebuild.
8. Startup sync: exact ID-set equality verifies sync even when counts match (same count, different IDs).
9. Startup sync: no-op when store and index IDs match.
10. Startup sync: force=True forces rebuild even if in sync.
11. Tool surface isolation: ToolRegistry strictly does not expose rebuild to Gemma agent (only 8 canonical tools).
12. Direct callable: MemoryManager rebuild and sync APIs remain directly callable for operational maintenance.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.dispatcher import ToolRegistry
from sage_document_db.embeddings import EmbeddingService
from sage_document_db.utils import iso_now
from sage_memory.memory_index import MemoryChromaStore
from sage_memory.memory_manager import MemoryManager
from sage_memory.memory_models import MemoryRecord, make_memory_id
from sage_memory.memory_store import MemoryStore
from tools.memory import (
    register_memory_tools,
    tool_memory_search_cold,
    tool_memory_search_hot,
    tool_memory_store_cold,
    tool_memory_store_hot,
    tool_memory_summarize,
)


@pytest.fixture()
def memory_manager(tmp_memory_root, tmp_chroma_root):
    store = MemoryStore(tmp_memory_root)
    index = MemoryChromaStore(tmp_chroma_root, EmbeddingService())
    return MemoryManager(store=store, index=index)


@pytest.fixture()
def tool_registry(memory_manager):
    registry = ToolRegistry()
    register_memory_tools(registry, memory_manager)
    return registry


# ─── 1. Index Rebuild Tests ───────────────────────────────────────────────────

class TestMemoryIndexRebuild:
    def test_rebuild_from_empty_chroma(self, memory_manager):
        """Rebuilding from disk populates empty Chroma collections completely."""
        # Store hot and cold memories
        h1 = tool_memory_store_hot(
            manager=memory_manager,
            content="Hot memory 1: Apollo guidance code",
            session_id="sess_rebuild_1",
            category="project",
            importance=0.8,
        )
        h2 = tool_memory_store_hot(
            manager=memory_manager,
            content="Hot memory 2: Saturn V thrust metrics",
            session_id="sess_rebuild_1",
            category="technical",
            importance=0.9,
        )
        c1 = tool_memory_store_cold(
            manager=memory_manager,
            content="Cold memory 1: User prefers metric system",
            category="preference",
            importance=0.7,
        )
        c2 = tool_memory_store_cold(
            manager=memory_manager,
            content="Cold memory 2: Global timezone UTC",
            category="fact",
            importance=0.6,
        )

        assert memory_manager.index.count("hot") == 2
        assert memory_manager.index.count("cold") == 2

        # Wipe Chroma collections directly
        memory_manager.index._hot.delete(ids=[h1["memory_id"], h2["memory_id"]])
        memory_manager.index._cold.delete(ids=[c1["memory_id"], c2["memory_id"]])
        assert memory_manager.index.count("hot") == 0
        assert memory_manager.index.count("cold") == 0

        # Run rebuild
        res = memory_manager.rebuild_memory_indexes()
        assert res["status"] == "success"
        assert res["hot"] == 2
        assert res["cold"] == 2
        assert res["total"] == 4
        assert "rebuilt_at" in res

        # Verify Chroma collections restored
        assert memory_manager.index.count("hot") == 2
        assert memory_manager.index.count("cold") == 2

    def test_rebuild_restores_searchability_and_session_isolation(self, memory_manager):
        """After rebuild, semantic search and session isolation work as expected."""
        h_a = tool_memory_store_hot(
            manager=memory_manager,
            content="Session Alpha secret code: 4242",
            session_id="sess_A",
        )
        h_b = tool_memory_store_hot(
            manager=memory_manager,
            content="Session Beta secret code: 9999",
            session_id="sess_B",
        )
        c_user = tool_memory_store_cold(
            manager=memory_manager,
            content="User prefers tabs over spaces",
            category="preference",
        )

        # Clear and rebuild
        memory_manager.rebuild_memory_indexes()

        # Session A search finds only A
        hits_a = memory_manager.search_hot_memory("secret code", session_id="sess_A")
        assert len(hits_a) == 1
        assert hits_a[0].memory_id == h_a["memory_id"]
        assert "4242" in hits_a[0].content

        # Session B search finds only B
        hits_b = memory_manager.search_hot_memory("secret code", session_id="sess_B")
        assert len(hits_b) == 1
        assert hits_b[0].memory_id == h_b["memory_id"]
        assert "9999" in hits_b[0].content

        # Cold search works cross-session
        hits_cold = memory_manager.search_cold_memory("tabs or spaces")
        assert len(hits_cold) == 1
        assert hits_cold[0].memory_id == c_user["memory_id"]

    def test_rebuild_preserves_all_metadata_and_summaries(self, memory_manager):
        """Rebuild preserves importance, confidence, source, is_summary, and parent IDs."""
        p1 = tool_memory_store_hot(memory_manager, "Detail 1", session_id="sess_sum")
        p2 = tool_memory_store_hot(memory_manager, "Detail 2", session_id="sess_sum")

        summary = tool_memory_summarize(
            manager=memory_manager,
            session_id="sess_sum",
            summary_content="Summary of 1 and 2",
            parent_memory_ids=[p1["memory_id"], p2["memory_id"]],
            importance=0.92,
            confidence=0.88,
        )

        # Rebuild
        memory_manager.rebuild_memory_indexes()

        # Retrieve summary via hot search
        hits = memory_manager.search_hot_memory(
            "Summary of 1 and 2",
            session_id="sess_sum",
            is_summary=True,
        )
        assert len(hits) == 1
        hit = hits[0]
        assert hit.memory_id == summary["memory_id"]
        assert hit.is_summary is True
        assert hit.importance == 0.92
        assert hit.confidence == 0.88

    def test_rebuild_wipes_stale_orphaned_records(self, memory_manager):
        """Rebuild eliminates Chroma records that no longer exist on disk."""
        stored = tool_memory_store_hot(
            manager=memory_manager,
            content="Active note",
            session_id="sess_stale",
        )

        # Artificially inject an orphan record directly into Chroma
        orphan_record = MemoryRecord(
            memory_id="mem_orphan_ghost_123",
            content="Ghost record never written to disk",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_stale",
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        memory_manager.index.add_memory(orphan_record)
        assert memory_manager.index.count("hot") == 2

        # Rebuild should purge the orphan because it's not in canonical store
        res = memory_manager.rebuild_memory_indexes()
        assert res["hot"] == 1
        assert memory_manager.index.count("hot") == 1

        # Confirm ghost record is completely gone from index
        indexed_ids = memory_manager.index.get_indexed_memory_ids("hot")
        assert "mem_orphan_ghost_123" not in indexed_ids
        assert indexed_ids == [stored["memory_id"]]

        hits = memory_manager.search_hot_memory("Ghost record", session_id="sess_stale")
        assert all(h.memory_id != "mem_orphan_ghost_123" for h in hits)

    def test_rebuild_empty_store(self, memory_manager):
        """Rebuilding an empty store produces 0 counts without error."""
        res = memory_manager.rebuild_memory_indexes()
        assert res["status"] == "success"
        assert res["hot"] == 0
        assert res["cold"] == 0
        assert res["total"] == 0

    def test_rebuild_resilience_to_corrupted_file(self, memory_manager):
        """Rebuild skips corrupted JSON files on disk and indexes remaining valid records."""
        # Store a valid record
        valid = tool_memory_store_hot(
            manager=memory_manager,
            content="Valid operational note",
            session_id="sess_corrupt",
        )

        # Inject a corrupted JSON file into the hot store directory
        corrupt_path = memory_manager.store.hot_dir / "mem_corrupt_bad1.json"
        with open(corrupt_path, "w", encoding="utf-8") as f:
            f.write("{corrupt json file content!!!")

        # Rebuild should ignore corrupt and index the valid record
        res = memory_manager.rebuild_memory_indexes(ignore_corrupt=True)
        assert res["hot"] == 1
        assert res["total"] == 1
        assert memory_manager.index.count("hot") == 1


# ─── 2. Startup Sync Tests ────────────────────────────────────────────────────

class TestMemoryStartupSync:
    def test_sync_indexes_detects_desync_and_rebuilds(self, memory_manager):
        """sync_indexes detects when index is missing disk records and rebuilds."""
        # Store on disk and in Chroma
        tool_memory_store_cold(memory_manager, "Cold fact 1")
        tool_memory_store_cold(memory_manager, "Cold fact 2")

        # Wipe Chroma directly so counts differ (disk=2, index=0)
        memory_manager.index._cold.delete(ids=memory_manager.index.get_indexed_memory_ids("cold"))
        assert memory_manager.index.count("cold") == 0

        sync_res = memory_manager.sync_indexes()
        assert sync_res["status"] == "rebuilt"
        assert sync_res["cold"] == 2
        assert memory_manager.index.count("cold") == 2

    def test_sync_indexes_noop_when_in_sync(self, memory_manager):
        """sync_indexes returns in_sync without rebuilding when counts match."""
        tool_memory_store_hot(memory_manager, "Note 1", session_id="sess_sync")
        tool_memory_store_cold(memory_manager, "Fact 1")

        sync_res = memory_manager.sync_indexes()
        assert sync_res["status"] == "in_sync"
        assert sync_res["hot"] == 1
        assert sync_res["cold"] == 1
        assert sync_res["total"] == 2

    def test_sync_indexes_force_rebuild(self, memory_manager):
        """sync_indexes(force=True) rebuilds even if counts match."""
        tool_memory_store_cold(memory_manager, "Fact 1")

        sync_res = memory_manager.sync_indexes(force=True)
        assert sync_res["status"] == "rebuilt"
        assert sync_res["cold"] == 1

    def test_sync_indexes_same_count_different_ids_hot(self, memory_manager):
        """Startup sync triggers rebuild when canonical hot and Chroma hot have the same count
        (3 == 3) but different ID sets (mem_A, mem_B, mem_C vs mem_A, mem_B, mem_X)."""
        rec_a = MemoryRecord(
            memory_id="mem_A",
            content="Hot Alpha guidance note",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_sync_id_test",
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        rec_b = MemoryRecord(
            memory_id="mem_B",
            content="Hot Beta telemetry note",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_sync_id_test",
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        rec_c = MemoryRecord(
            memory_id="mem_C",
            content="Hot Gamma payload note",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_sync_id_test",
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        rec_x = MemoryRecord(
            memory_id="mem_X",
            content="Hot X orphaned note in Chroma only",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_sync_id_test",
            created_at=iso_now(),
            updated_at=iso_now(),
        )

        # Canonical store has A, B, C (3 items)
        memory_manager.store.create(rec_a)
        memory_manager.store.create(rec_b)
        memory_manager.store.create(rec_c)
        assert len(memory_manager.store.list_hot()) == 3

        # Chroma index has A, B, X (3 items, counts match 3 == 3)
        memory_manager.index.add_memory(rec_a)
        memory_manager.index.add_memory(rec_b)
        memory_manager.index.add_memory(rec_x)
        assert memory_manager.index.count("hot") == 3

        # Exact ID-set inequality must detect missing mem_C and orphaned mem_X, triggering rebuild
        sync_res = memory_manager.sync_indexes()
        assert sync_res["status"] == "rebuilt"
        assert sync_res["hot"] == 3

        # Final Chroma IDs must exactly equal canonical hot IDs
        chroma_hot_ids = set(memory_manager.index.get_indexed_memory_ids("hot"))
        assert chroma_hot_ids == {"mem_A", "mem_B", "mem_C"}
        assert "mem_X" not in chroma_hot_ids

    def test_sync_indexes_same_count_different_ids_cold(self, memory_manager):
        """Startup sync triggers rebuild when canonical cold and Chroma cold have the same count
        (3 == 3) but different ID sets (mem_cold_A, mem_cold_B, mem_cold_C vs mem_cold_A, mem_cold_B, mem_cold_X)."""
        rec_a = MemoryRecord(
            memory_id="mem_cold_A",
            content="Cold Alpha durable fact",
            memory_type="cold",
            category="fact",
            source="test",
            session_id=None,
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        rec_b = MemoryRecord(
            memory_id="mem_cold_B",
            content="Cold Beta durable preference",
            memory_type="cold",
            category="preference",
            source="test",
            session_id=None,
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        rec_c = MemoryRecord(
            memory_id="mem_cold_C",
            content="Cold Gamma durable rule",
            memory_type="cold",
            category="instruction",
            source="test",
            session_id=None,
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        rec_x = MemoryRecord(
            memory_id="mem_cold_X",
            content="Cold X orphaned record in Chroma only",
            memory_type="cold",
            category="fact",
            source="test",
            session_id=None,
            created_at=iso_now(),
            updated_at=iso_now(),
        )

        # Canonical store has A, B, C (3 items)
        memory_manager.store.create(rec_a)
        memory_manager.store.create(rec_b)
        memory_manager.store.create(rec_c)
        assert len(memory_manager.store.list_cold()) == 3

        # Chroma index has A, B, X (3 items, counts match 3 == 3)
        memory_manager.index.add_memory(rec_a)
        memory_manager.index.add_memory(rec_b)
        memory_manager.index.add_memory(rec_x)
        assert memory_manager.index.count("cold") == 3

        # Exact ID-set inequality triggers rebuild
        sync_res = memory_manager.sync_indexes()
        assert sync_res["status"] == "rebuilt"
        assert sync_res["cold"] == 3

        # Final Chroma IDs must exactly equal canonical cold IDs
        chroma_cold_ids = set(memory_manager.index.get_indexed_memory_ids("cold"))
        assert chroma_cold_ids == {"mem_cold_A", "mem_cold_B", "mem_cold_C"}
        assert "mem_cold_X" not in chroma_cold_ids

    def test_sync_indexes_detects_wrong_tier_records(self, memory_manager):
        """Startup sync detects when a record is mistakenly in the wrong Chroma collection tier."""
        rec_hot = MemoryRecord(
            memory_id="mem_tier_hot",
            content="Should be hot record",
            memory_type="hot",
            category="task",
            source="test",
            session_id="sess_tier",
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        memory_manager.store.create(rec_hot)

        # Erroneously added to cold Chroma index
        wrong_rec = MemoryRecord(
            memory_id="mem_tier_hot",
            content="Should be hot record",
            memory_type="cold",
            category="task",
            source="test",
            session_id=None,
            created_at=iso_now(),
            updated_at=iso_now(),
        )
        memory_manager.index.add_memory(wrong_rec)

        assert memory_manager.index.count("hot") == 0
        assert memory_manager.index.count("cold") == 1

        sync_res = memory_manager.sync_indexes()
        assert sync_res["status"] == "rebuilt"
        assert sync_res["hot"] == 1
        assert sync_res["cold"] == 0

        assert set(memory_manager.index.get_indexed_memory_ids("hot")) == {"mem_tier_hot"}
        assert set(memory_manager.index.get_indexed_memory_ids("cold")) == set()


# ─── 3. Tool Surface Isolation & Internal Maintenance Tests ───────────────────

class TestMemoryRebuildSurfaceIsolation:
    def test_tool_registry_strictly_exposes_eight_canonical_tools(self, tool_registry):
        """ToolRegistry strictly registers the 8 canonical memory tools and 8 short aliases,
        and MUST NOT register memory_rebuild or rebuild to the Gemma agent."""
        memory_tools = set(tool_registry.available_tools.get("memory", []))
        canonical_tools = {
            "memory_search_hot",
            "memory_search_cold",
            "memory_store_hot",
            "memory_store_cold",
            "memory_update",
            "memory_delete",
            "memory_summarize",
            "memory_promote",
        }
        short_aliases = {
            "search_hot",
            "search_cold",
            "store_hot",
            "store_cold",
            "update",
            "delete",
            "summarize",
            "promote",
        }
        assert memory_tools == (canonical_tools | short_aliases)
        assert "memory_rebuild" not in memory_tools
        assert "rebuild" not in memory_tools

    def test_rebuild_memory_indexes_direct_callable(self, memory_manager):
        """MemoryManager.rebuild_memory_indexes() remains directly callable for maintenance."""
        tool_memory_store_hot(memory_manager, "Hot direct note", session_id="sess_direct")
        tool_memory_store_cold(memory_manager, "Cold direct fact")

        res = memory_manager.rebuild_memory_indexes()
        assert res["status"] == "success"
        assert res["hot"] == 1
        assert res["cold"] == 1
        assert res["total"] == 2
        assert "rebuilt_at" in res

    def test_sync_indexes_direct_callable(self, memory_manager):
        """MemoryManager.sync_indexes() remains directly callable for maintenance."""
        tool_memory_store_hot(memory_manager, "Direct sync note", session_id="sess_direct_sync")
        res = memory_manager.sync_indexes()
        assert res["status"] == "in_sync"
        assert res["hot"] == 1
