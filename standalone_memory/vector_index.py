"""
standalone_memory/vector_index.py
Self-contained ChromaDB vector index manager with built-in EmbeddingService.
Maintains separate Hot and Cold collections.
"""

from __future__ import annotations
import gc
import hashlib
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Provides 384-dimensional embeddings for semantic search.
    
    Uses sentence-transformers if available.
    Falls back gracefully to deterministic normalized hash vectors when
    dependencies are not installed, allowing lightweight testing and CI execution.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        self.device = device or config.EMBEDDING_DEVICE
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        if os.environ.get("SAGE_MOCK_MODE") == "1" or os.environ.get("STANDALONE_MEMORY_MOCK") == "1":
            logger.info("EmbeddingService: Mock mode enabled, using deterministic fallback.")
            self._model = None
            return

        try:
            from sentence_transformers import SentenceTransformer
            dev = self.device
            if dev == "auto":
                try:
                    import torch
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                except Exception:
                    dev = "cpu"
            # Try loading from local cache first to avoid network hangs
            try:
                self._model = SentenceTransformer(self.model_name, device=dev, local_files_only=True)
                logger.info("SentenceTransformer '%s' loaded from cache on device '%s'", self.model_name, dev)
            except Exception as cache_err:
                logger.info("SentenceTransformer not cached locally (%s); using deterministic vector fallback.", cache_err)
                self._model = None
        except Exception as e:
            logger.info("Using deterministic fallback embeddings (sentence-transformers unavailable: %s)", e)
            self._model = None

    def embed_text(self, text: str) -> List[float]:
        """Generate a 384-dimensional normalized float embedding."""
        if self._model is not None:
            emb = self._model.encode(text, normalize_embeddings=True)
            return emb.tolist() if hasattr(emb, "tolist") else list(emb)
        return self._deterministic_fallback_vector(text)

    def _deterministic_fallback_vector(self, text: str, dim: int = 384) -> List[float]:
        """Compute a deterministic, unit-normalized vector for testing without Torch/models."""
        vec = [0.0] * dim
        normalized = text.lower().strip()
        words = normalized.split()
        for idx, word in enumerate(words):
            for i in range(4):
                digest = hashlib.sha256(f"{word}_{i}".encode("utf-8")).digest()
                pos = (int.from_bytes(digest[:4], "big") + idx) % dim
                weight = ((digest[4] / 255.0) - 0.5) * 2.0
                vec[pos] += weight

        # Normalize to unit length
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 1e-9:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0
        return vec


class MemoryVectorIndex:
    """Manages Chroma vector collections for Hot and Cold memory tiers."""

    def __init__(
        self,
        chroma_dir: Optional[str | Path] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ) -> None:
        self._chroma_dir = Path(chroma_dir or config.CHROMA_PERSIST_DIR)
        self._emb = embedding_service
        self._client: Any = None
        self._collections: Dict[str, Any] = {}

    def _get_embedding_service(self) -> EmbeddingService:
        if self._emb is None:
            self._emb = EmbeddingService()
        return self._emb

    def _get_client(self) -> Any:
        if self._client is None:
            import chromadb
            self._chroma_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        return self._client

    def _get_collection(self, tier: str = "cold") -> Any:
        tier_key = (tier or "cold").lower().strip()
        col_name = config.CHROMA_HOT_COLLECTION if tier_key == "hot" else config.CHROMA_COLD_COLLECTION

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

    def index_memory(self, memory_dict: Dict[str, Any]) -> bool:
        """Upsert a memory record into the appropriate Chroma vector tier."""
        mem_id = memory_dict.get("memory_id")
        content = memory_dict.get("content")
        tier = memory_dict.get("memory_tier", "cold")
        if not mem_id or not content:
            return False

        try:
            col = self._get_collection(tier)
            vec = self._get_embedding_service().embed_text(content)
            metadata = {
                "user_id": str(memory_dict.get("user_id", "")),
                "category": str(memory_dict.get("category", "")),
                "status": str(memory_dict.get("status", "active")),
                "is_global": int(memory_dict.get("is_global", 0)),
                "memory_tier": tier,
                "importance": float(memory_dict.get("importance", 0.5)),
            }
            col.upsert(
                ids=[mem_id],
                embeddings=[vec],
                documents=[content],
                metadatas=[metadata],
            )
            return True
        except Exception as exc:
            logger.error("Failed to index memory %s in Chroma: %s", mem_id, exc)
            return False

    def remove_memory(self, memory_id: str, tier: Optional[str] = None) -> bool:
        """Purge memory vector from one or both collections."""
        tiers = [tier] if tier else ["hot", "cold"]
        removed = False
        for t in tiers:
            try:
                col = self._get_collection(t)
                col.delete(ids=[memory_id])
                removed = True
            except Exception:
                pass
        return removed

    def update_memory_status(self, memory_id: str, status: str, tier: Optional[str] = None) -> bool:
        """Update status metadata in vector collections."""
        tiers = [tier] if tier else ["hot", "cold"]
        updated = False
        for t in tiers:
            try:
                col = self._get_collection(t)
                existing = col.get(ids=[memory_id], include=["metadatas", "documents"])
                if existing and existing.get("ids") and len(existing["ids"]) > 0:
                    meta = existing["metadatas"][0] if existing.get("metadatas") else {}
                    meta["status"] = status
                    col.update(ids=[memory_id], metadatas=[meta])
                    updated = True
            except Exception:
                pass
        return updated

    def search_memories(
        self,
        query: str,
        user_id: str,
        tier: str = "cold",
        category: Optional[str] = None,
        limit: int = 5,
        status: str = "active",
    ) -> List[Dict[str, Any]]:
        """Perform semantic similarity search on a specific tier."""
        if not query or not query.strip():
            return []

        try:
            col = self._get_collection(tier)
            query_vec = self._get_embedding_service().embed_text(query)

            # Build metadata filter
            where_clauses: List[Dict[str, Any]] = [
                {"user_id": {"$eq": str(user_id)}},
                {"status": {"$eq": str(status)}},
            ]
            if category:
                where_clauses.append({"category": {"$eq": str(category)}})

            where: Dict[str, Any]
            if len(where_clauses) == 1:
                where = where_clauses[0]
            else:
                where = {"$and": where_clauses}

            results = col.query(
                query_embeddings=[query_vec],
                n_results=min(limit, 20),
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            hits = []
            if results and results.get("ids") and results["ids"][0]:
                ids = results["ids"][0]
                docs = results["documents"][0] if results.get("documents") else []
                metas = results["metadatas"][0] if results.get("metadatas") else []
                dists = results["distances"][0] if results.get("distances") else []

                for i, mem_id in enumerate(ids):
                    doc = docs[i] if i < len(docs) else ""
                    meta = metas[i] if i < len(metas) else {}
                    dist = dists[i] if i < len(dists) else 1.0
                    similarity = max(0.0, 1.0 - float(dist)) if dist is not None else 0.5
                    hits.append({
                        "memory_id": mem_id,
                        "content": doc,
                        "similarity": round(similarity, 4),
                        "category": meta.get("category", ""),
                        "memory_tier": meta.get("memory_tier", tier),
                        "importance": meta.get("importance", 0.5),
                        "status": meta.get("status", status),
                    })
            return hits
        except Exception as exc:
            logger.error("Chroma search failed: %s", exc)
            return []

    def close(self) -> None:
        """Release Chroma client and file locks cleanly."""
        self._collections.clear()
        if self._client is not None:
            try:
                if hasattr(self._client, "clear_system_cache"):
                    self._client.clear_system_cache()
            except Exception:
                pass
            self._client = None
        gc.collect()


# Global default instance
memory_vector_index = MemoryVectorIndex()
