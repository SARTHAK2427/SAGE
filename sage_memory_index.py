"""
SAGE/sage_memory_index.py
Phase 4: Semantic Memory Retrieval with Embeddings + Chroma

Manages the Chroma vector index ('sage_memory' collection) for durable memory.
PostgreSQL/SQLite remain the canonical source of truth; Chroma provides derived
vector retrieval.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import config
from sage_document_db.embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class MemoryVectorIndex:
    """Vector index manager for durable memory (separate hot and cold collections)."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        chroma_root: str | Path | None = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._emb = embedding_service
        self._chroma_root = Path(chroma_root or config.CHROMA_ROOT)
        self._custom_collection_name = collection_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collections: Dict[str, Any] = {}

    def _get_embedding_service(self) -> EmbeddingService:
        if self._emb is None:
            self._emb = EmbeddingService()
        return self._emb

    def _get_client(self) -> Any:
        if self._client is None:
            self._chroma_root.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._chroma_root))
        return self._client

    def _get_collection(self, tier: str = "cold") -> Any:
        tier_key = (tier or "cold").lower().strip()
        if self._custom_collection_name:
            col_name = self._custom_collection_name
        elif tier_key == "hot":
            col_name = getattr(config, "SAGE_MEMORY_HOT_COLLECTION", "sage_memory_hot")
        else:
            col_name = getattr(config, "SAGE_MEMORY_COLD_COLLECTION", "sage_memory_cold")

        if col_name not in self._collections:
            client = self._get_client()
            try:
                col = client.get_or_create_collection(
                    name=col_name,
                    metadata={"hnsw:space": "cosine"},
                )
            except Exception:
                col = client.get_or_create_collection(name=col_name)
            self._collections[col_name] = col
        return self._collections[col_name]

    @property
    def _collection(self) -> Any:
        """Backward compatibility property returning the cold/default collection."""
        return self._get_collection("cold")

    @_collection.setter
    def _collection(self, value: Any) -> None:
        """Backward compatibility setter. Assigning None resets all cached collections."""
        if value is None:
            self._collections = {}
            self._client = None
        # Non-None assignments are ignored; use _collections directly for internal management

    def close(self) -> None:
        """Release Chroma client, collections, and background file handles."""
        self._collections.clear()
        if self._client is not None:
            try:
                if hasattr(self._client, "clear_system_cache"):
                    self._client.clear_system_cache()
            except Exception:
                pass
            try:
                if hasattr(self._client, "_server") and hasattr(self._client._server, "stop"):
                    self._client._server.stop()
            except Exception:
                pass
            self._client = None
        import gc
        gc.collect()

    def __enter__(self) -> MemoryVectorIndex:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def index_memory(self, memory: Dict[str, Any]) -> bool:
        """Embed and upsert a canonical memory record into Chroma hot or cold collection."""
        memory_id = memory.get("memory_id")
        content = memory.get("content")
        user_id = memory.get("user_id")
        if not memory_id or not content or not user_id:
            logger.warning("Cannot index memory: missing required fields.")
            return False

        tier = (memory.get("memory_tier") or "cold").lower().strip()
        if tier not in ("hot", "cold"):
            tier = "cold"

        try:
            emb_svc = self._get_embedding_service()
            vector = emb_svc.embed_query(str(content))
            target_col = self._get_collection(tier)

            metadata = {
                "user_id": str(user_id),
                "chat_id": str(memory.get("source_chat_id") or ""),
                "category": str(memory.get("category", "")),
                "status": str(memory.get("status", "active")),
                "memory_tier": tier,
                "importance": float(memory.get("importance", 0.5)),
                "confidence": float(memory.get("confidence", 1.0)),
                "updated_at": str(memory.get("updated_at", "")),
            }

            target_col.upsert(
                ids=[str(memory_id)],
                embeddings=[vector],
                documents=[str(content)],
                metadatas=[metadata],
            )

            # If not a custom-named collection, ensure opposite tier collection does not retain stale record
            if not self._custom_collection_name:
                opp_tier = "cold" if tier == "hot" else "hot"
                try:
                    opp_col = self._get_collection(opp_tier)
                    opp_col.delete(ids=[str(memory_id)])
                except Exception:
                    pass

            return True
        # Chroma's Rust bindings can surface a PyO3 PanicException, which is
        # derived from BaseException rather than Exception.  The vector index
        # is derived data, so an index failure must not make canonical-memory
        # writes fail.
        except BaseException as exc:
            logger.error("Failed to index memory %s in Chroma (%s): %s", memory_id, tier, exc)
            return False

    def update_memory_status(self, memory_id: str, status: str, tier: Optional[str] = None) -> bool:
        """Update the status metadata for a vector in Chroma."""
        if not memory_id or not status:
            return False

        tiers_to_update = [tier.lower().strip()] if tier else (["hot", "cold"] if not self._custom_collection_name else ["cold"])
        updated_any = False

        for t in tiers_to_update:
            try:
                col = self._get_collection(t)
                existing = col.get(ids=[str(memory_id)], include=["metadatas"])
                if existing and existing.get("ids"):
                    metas = existing.get("metadatas") or [{}]
                    if metas and metas[0]:
                        updated_meta = dict(metas[0])
                        updated_meta["status"] = str(status)
                        col.update(ids=[str(memory_id)], metadatas=[updated_meta])
                        updated_any = True
            except Exception as exc:
                logger.error("Failed to update status for memory vector %s in %s: %s", memory_id, t, exc)

        return updated_any

    def delete_memory_vector(self, memory_id: str, tier: Optional[str] = None) -> bool:
        """Delete a vector from Chroma."""
        if not memory_id:
            return False

        tiers_to_delete = [tier.lower().strip()] if tier else (["hot", "cold"] if not self._custom_collection_name else ["cold"])
        deleted_any = False

        for t in tiers_to_delete:
            try:
                col = self._get_collection(t)
                col.delete(ids=[str(memory_id)])
                deleted_any = True
            except Exception as exc:
                logger.error("Failed to delete memory vector %s from %s: %s", memory_id, t, exc)

        return deleted_any

    def search_memory_vectors(
        self,
        user_id: str,
        query: str,
        tier: str = "cold",
        chat_id: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Perform user-isolated semantic search in Chroma for specified tier ('hot' or 'cold')."""
        if not user_id or not query or not query.strip():
            return []

        tier_key = (tier or "cold").lower().strip()
        try:
            emb_svc = self._get_embedding_service()
            query_vector = emb_svc.embed_query(query.strip())
            collection = self._get_collection(tier_key)

            count = collection.count()
            if count == 0:
                return []

            bounded_limit = max(1, min(limit, 20))
            n_results = min(bounded_limit, count)

            # Build user isolation + active status where clause
            where_conditions: List[Dict[str, Any]] = [
                {"user_id": {"$eq": str(user_id)}},
                {"status": {"$eq": "active"}},
                {"memory_tier": {"$eq": tier_key}},
            ]
            if tier_key == "hot" and chat_id:
                where_conditions.append({"chat_id": {"$eq": str(chat_id)}})
            if category:
                where_conditions.append({"category": {"$eq": str(category)}})

            if len(where_conditions) == 1:
                where_clause = where_conditions[0]
            else:
                where_clause = {"$and": where_conditions}

            raw = collection.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )

            ids = (raw.get("ids") or [[]])[0]
            docs = (raw.get("documents") or [[]])[0]
            metas = (raw.get("metadatas") or [[]])[0]
            distances = (raw.get("distances") or [[]])[0]

            results: List[Dict[str, Any]] = []
            for record_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
                results.append({
                    "memory_id": str(record_id),
                    "document": str(doc_text),
                    "metadata": meta or {},
                    "distance": float(dist),
                })
            return results
        # See index_memory: an unavailable/corrupt derived index must degrade
        # to the canonical store instead of aborting a memory lookup.
        except BaseException as exc:
            logger.error("Semantic memory search failed for user %s (tier=%s): %s", user_id, tier_key, exc)
            return []

    def reindex_memories(self, sage_memory_instance: Any, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Rebuild Chroma memory vector collections from canonical active memories in PostgreSQL/SQLite."""
        indexed_count = 0
        errors: List[str] = []

        try:
            backend_conn = sage_memory_instance._get_connection()
            try:
                b_type = config.SAGE_MEMORY_DB.lower().strip()
                if user_id:
                    memories = sage_memory_instance.list_memories(user_id=user_id, limit=5000)
                else:
                    query_sql = (
                        "SELECT memory_id, user_id, content, category, importance, confidence, "
                        "source_chat_id, status, memory_tier, updated_at "
                        "FROM memories WHERE status = 'active';"
                    )
                    if b_type == "sqlite":
                        cur = backend_conn.execute(query_sql)
                        rows = cur.fetchall()
                        memories = [
                            {
                                "memory_id": str(r["memory_id"]),
                                "user_id": str(r["user_id"]),
                                "content": str(r["content"]),
                                "category": str(r["category"]),
                                "importance": float(r["importance"]),
                                "confidence": float(r["confidence"]),
                                "source_chat_id": r["source_chat_id"],
                                "status": str(r["status"]),
                                "memory_tier": str(r["memory_tier"] or "cold") if "memory_tier" in r.keys() else "cold",
                                "updated_at": str(r["updated_at"]),
                            }
                            for r in rows
                        ]
                    else:
                        import psycopg2.extras
                        with backend_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                            cur.execute(query_sql)
                            rows = cur.fetchall()
                            memories = [
                                {
                                    "memory_id": str(r["memory_id"]),
                                    "user_id": str(r["user_id"]),
                                    "content": str(r["content"]),
                                    "category": str(r["category"]),
                                    "importance": float(r["importance"]),
                                    "confidence": float(r["confidence"]),
                                    "source_chat_id": r["source_chat_id"],
                                    "status": str(r["status"]),
                                    "memory_tier": str(r["memory_tier"] or "cold") if "memory_tier" in r.keys() else "cold",
                                    "updated_at": str(r["updated_at"]),
                                }
                                for r in rows
                            ]
            finally:
                if hasattr(backend_conn, "close"):
                    backend_conn.close()

            for mem in memories:
                if self.index_memory(mem):
                    indexed_count += 1
                else:
                    errors.append(f"Failed to index memory_id={mem.get('memory_id')}")

            return {
                "status": "success",
                "indexed_count": indexed_count,
                "errors": errors,
            }
        except Exception as exc:
            logger.error("Reindexing memories failed: %s", exc)
            return {
                "status": "error",
                "error": str(exc),
                "indexed_count": indexed_count,
            }


# Singleton instance
memory_vector_index = MemoryVectorIndex()
