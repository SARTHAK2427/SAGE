"""
sage_document_db/chroma_store.py
Chroma vector store adapter.

Rules:
- PersistentClient at CHROMA_ROOT — never in-memory.
- Two collections: sage_source (deterministic) and sage_derived (model output).
- Embeddings are always computed externally and passed in — never let Chroma
  choose its own embedding function.
- Normalize raw Chroma nested-array results immediately; never leak them.
- No embeddings in application output.

Correction 2: doc_ids filtering uses Chroma where clause during query,
              not post-retrieval filtering.

Correction 3: reindex operations use ArtifactReader to reconstruct
              NormalizedDocument and feed it through the same Chunker.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import TYPE_CHECKING

import chromadb

from .config import (
    ARTIFACTS_ROOT,
    CHROMA_ROOT,
    CHUNK_TARGET_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    DEFAULT_RAG_TOP_K,
    EMBEDDING_MODEL,
    SOURCE_COLLECTION,
    DERIVED_COLLECTION,
)
from .models import IndexResult, RagResult, SearchRecord
from .utils import decode_list_field, iso_now

if TYPE_CHECKING:
    from .embeddings import EmbeddingService
    from .chunker import Chunker


# ---------------------------------------------------------------------------
# Index config guard
# ---------------------------------------------------------------------------

INDEX_CONFIG_FILE = Path(CHROMA_ROOT) / "index_config.json"

_INDEX_CONFIG = {
    "schema_version": "1.0",
    "embedding_model": EMBEDDING_MODEL,
    "chunk_target_tokens": CHUNK_TARGET_TOKENS,
    "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
    "source_collection": SOURCE_COLLECTION,
    "derived_collection": DERIVED_COLLECTION,
}


def _check_index_config() -> None:
    """Enforce that the stored index config matches current config.

    If absent: create it.
    If embedding model differs: raise to require rebuild.
    """
    config_path = INDEX_CONFIG_FILE
    config_path.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(_INDEX_CONFIG, f, indent=2)
        return

    with open(config_path, "r", encoding="utf-8") as f:
        stored = json.load(f)

    if stored.get("embedding_model") != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Chroma index was built with embedding model "
            f"'{stored.get('embedding_model')}' but current config uses "
            f"'{EMBEDDING_MODEL}'. Delete chroma_db/ and run rebuild-all."
        )

    if stored.get("chunk_target_tokens") != CHUNK_TARGET_TOKENS:
        print(
            f"[WARNING] Chunk target tokens changed "
            f"({stored.get('chunk_target_tokens')} → {CHUNK_TARGET_TOKENS}). "
            "Existing records may be from a different chunking schema. "
            "Consider running rebuild-all."
        )


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------

def _normalize_results(raw: dict, origin: str) -> list[RagResult]:
    """Convert Chroma's nested-array result format to list[RagResult].

    Chroma returns:
        {"ids": [[...]], "documents": [[...]], "metadatas": [[...]], "distances": [[...]]}
    """
    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    distances = (raw.get("distances") or [[]])[0]

    results = []
    for record_id, text, meta, dist in zip(ids, docs, metas, distances):
        meta = meta or {}
        results.append(RagResult(
            record_id=record_id,
            origin=origin,
            record_type=meta.get("record_type", "text_chunk"),
            text=text or "",
            distance=float(dist),
            doc_id=meta.get("doc_id", ""),
            page=meta.get("page"),
            source_element_ids=decode_list_field(meta.get("source_element_ids_json")),
            image_refs=decode_list_field(meta.get("image_refs_json")),
            table_refs=decode_list_field(meta.get("table_refs_json")),
            code_refs=decode_list_field(meta.get("code_refs_json")),
            derived_cache_ref=meta.get("derived_cache_ref"),
        ))
    return results


# ---------------------------------------------------------------------------
# ChromaStore
# ---------------------------------------------------------------------------

class ChromaStore:
    """Manages the sage_source and sage_derived Chroma collections."""

    def __init__(
        self,
        embedding_service: "EmbeddingService",
        chunker: "Chunker",
        chroma_root: str | Path = CHROMA_ROOT,
        artifacts_root: str | Path = ARTIFACTS_ROOT,
    ) -> None:
        self._emb = embedding_service
        self._chunker = chunker
        self._chroma_root = Path(chroma_root)
        self._artifacts_root = Path(artifacts_root)

        _check_index_config()

        self._client = chromadb.PersistentClient(path=str(self._chroma_root))
        self._source = self._get_or_create_collection(SOURCE_COLLECTION)
        self._derived = self._get_or_create_collection(DERIVED_COLLECTION)

    def _get_or_create_collection(self, name: str):
        """Get or create a collection with cosine distance metric."""
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception:
            # Fallback if metadata kwarg not supported in this Chroma version
            return self._client.get_or_create_collection(name=name)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_document(
        self,
        doc,  # NormalizedDocument
        replace_existing: bool = True,
    ) -> IndexResult:
        """Index a NormalizedDocument into sage_source.

        Steps:
        1. Generate SearchRecords via Chunker
        2. Validate non-empty text
        3. Delete old records if replacing
        4. Embed in batches
        5. Upsert
        """
        records = self._chunker.chunk_document(doc)
        records = [r for r in records if r.text.strip()]

        if replace_existing:
            self._delete_source_records(doc.doc_id)

        if not records:
            return IndexResult(
                doc_id=doc.doc_id,
                source_records=0,
                embedded_records=0,
            )

        self._upsert_records(self._source, records)

        return IndexResult(
            doc_id=doc.doc_id,
            source_records=len(records),
            embedded_records=len(records),
        )

    def upsert_derived_record(
        self,
        record_id: str,
        text: str,
        metadata: dict,
    ) -> None:
        """Upsert one record into sage_derived."""
        vector = self._emb.embed_documents([text])[0]
        self._derived.upsert(
            ids=[record_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata],
        )

    def _upsert_records(self, collection, records: list[SearchRecord]) -> None:
        """Embed and upsert a list of SearchRecords into a collection."""
        texts = [r.text for r in records]
        ids = [r.id for r in records]
        metadatas = [r.metadata for r in records]
        vectors = self._emb.embed_documents(texts)
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    def _delete_source_records(self, doc_id: str) -> None:
        """Delete all source records for a doc_id."""
        try:
            existing = self._source.get(where={"doc_id": {"$eq": doc_id}})
            ids = existing.get("ids", [])
            if ids:
                self._source.delete(ids=ids)
        except Exception as e:
            print(f"[WARNING] Could not delete existing source records for {doc_id}: {e}")

    def delete_derived_records(self, doc_id: str, image_id: str) -> None:
        """Delete a specific derived image record."""
        record_id = f"{doc_id}:derived:image:{image_id}"
        try:
            self._derived.delete(ids=[record_id])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Querying (Correction 2)
    # ------------------------------------------------------------------

    def rag_query(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_RAG_TOP_K,
        doc_ids: list[str] | None = None,
        include_source: bool = True,
        include_derived: bool = True,
    ) -> list[RagResult]:
        """Query Chroma collections and return normalized RagResults.

        Correction 2: doc_ids filtering uses Chroma where clause at query
        time — NOT post-retrieval filtering of global candidates.
        If doc_ids is None, no where clause is applied.
        """
        candidates_per_collection = top_k * 2
        results: list[RagResult] = []

        where = None
        if doc_ids is not None:
            if len(doc_ids) == 1:
                where = {"doc_id": {"$eq": doc_ids[0]}}
            else:
                where = {"doc_id": {"$in": doc_ids}}

        query_kwargs: dict = {
            "query_embeddings": [query_vector],
            "n_results": candidates_per_collection,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where

        if include_source:
            try:
                n_source = self._source.count()
                if n_source > 0:
                    # Chroma raises if n_results > collection size
                    query_kwargs["n_results"] = min(candidates_per_collection, n_source)
                    raw = self._source.query(**query_kwargs)
                    results.extend(_normalize_results(raw, origin="source"))
            except Exception as e:
                print(f"[WARNING] sage_source query failed: {e}")

        if include_derived:
            try:
                n_derived = self._derived.count()
                if n_derived > 0:
                    query_kwargs["n_results"] = min(candidates_per_collection, n_derived)
                    raw = self._derived.query(**query_kwargs)
                    results.extend(_normalize_results(raw, origin="derived"))
            except Exception as e:
                print(f"[WARNING] sage_derived query failed: {e}")

        # Deduplicate by record_id, sort by distance, take top_k
        seen: set[str] = set()
        unique: list[RagResult] = []
        for r in results:
            if r.record_id not in seen:
                seen.add(r.record_id)
                unique.append(r)

        unique.sort(key=lambda r: r.distance)
        return unique[:top_k]

    def rag_query_socket(
        self,
        query_vector: list[float],
        top_k: int = DEFAULT_RAG_TOP_K,
        doc_ids: list[str] | None = None,
        include_source: bool = True,
        include_derived: bool = True,
    ) -> dict:
        """Query Chroma and return the complete Component G socket."""
        import time
        from core.sockets import build_chroma_query_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        collections_queried = []
        if include_source:
            collections_queried.append(SOURCE_COLLECTION)
        if include_derived:
            collections_queried.append(DERIVED_COLLECTION)

        try:
            candidates_per_collection = top_k * 2
            where = None
            if doc_ids is not None:
                if len(doc_ids) == 1:
                    where = {"doc_id": {"$eq": doc_ids[0]}}
                else:
                    where = {"doc_id": {"$in": doc_ids}}

            query_kwargs: dict = {
                "query_embeddings": [query_vector],
                "n_results": candidates_per_collection,
                "include": ["documents", "metadatas", "distances"],
            }
            if where is not None:
                query_kwargs["where"] = where

            collection_responses: dict = {}
            total_returned = 0

            if include_source:
                try:
                    n_source = self._source.count()
                    if n_source > 0:
                        query_kwargs["n_results"] = min(candidates_per_collection, n_source)
                        raw_src = self._source.query(**query_kwargs)
                        collection_responses[SOURCE_COLLECTION] = raw_src
                        total_returned += len((raw_src.get("ids") or [[]])[0])
                except Exception as e:
                    collection_responses[SOURCE_COLLECTION] = {"error": str(e)}

            if include_derived:
                try:
                    n_derived = self._derived.count()
                    if n_derived > 0:
                        query_kwargs["n_results"] = min(candidates_per_collection, n_derived)
                        raw_der = self._derived.query(**query_kwargs)
                        collection_responses[DERIVED_COLLECTION] = raw_der
                        total_returned += len((raw_der.get("ids") or [[]])[0])
                except Exception as e:
                    collection_responses[DERIVED_COLLECTION] = {"error": str(e)}

            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)

            return build_chroma_query_socket(
                collections_queried=collections_queried,
                query_count=1,
                requested_top_k=top_k,
                returned_count=total_returned,
                collection_responses=collection_responses,
                raw_chroma_responses=dict(collection_responses),
                where_filter=where,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="CHROMA_QUERY_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_chroma_query_socket(
                collections_queried=collections_queried,
                query_count=1,
                requested_top_k=top_k,
                returned_count=0,
                collection_responses={},
                timing=timing,
                error=error_payload,
            )

    def index_document_socket(
        self,
        doc,
        replace_existing: bool = True,
    ) -> dict:
        """Index a document and return the complete Component H socket."""
        import time
        from core.sockets import build_chroma_upsert_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            res = self.index_document(doc, replace_existing=replace_existing)
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            return build_chroma_upsert_socket(
                collection_name=SOURCE_COLLECTION,
                record_count=res.source_records,
                embedded_count=res.embedded_records,
                requested_count=res.source_records,
                successful_count=res.source_records,
                failed_count=0,
                skipped_count=0,
                doc_ids_affected=[doc.doc_id],
                replace_existing=replace_existing,
                origin="source",
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="CHROMA_UPSERT_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_chroma_upsert_socket(
                collection_name=SOURCE_COLLECTION,
                record_count=0,
                embedded_count=0,
                requested_count=0,
                successful_count=0,
                failed_count=0,
                skipped_count=0,
                doc_ids_affected=[doc.doc_id] if doc else [],
                replace_existing=replace_existing,
                origin="source",
                timing=timing,
                error=error_payload,
            )

    # ------------------------------------------------------------------
    # Reindex (Correction 3)
    # ------------------------------------------------------------------

    def reindex_document_from_artifacts(self, doc_id: str) -> IndexResult:
        """Rebuild source index for one document from its canonical artifacts.

        Uses ArtifactReader to reconstruct NormalizedDocument, then
        feeds it through the SAME Chunker used during ingestion.
        """
        from .artifact_store import ArtifactReader
        reader = ArtifactReader(self._artifacts_root)
        doc = reader.load_normalized_document(doc_id)
        return self.index_document(doc, replace_existing=True)

    def rebuild_source_index_from_all_artifacts(self) -> dict:
        """Rebuild sage_source from all canonical artifacts.

        Does NOT delete artifacts. Only rebuilds the Chroma index.
        """
        from .artifact_store import ArtifactReader
        reader = ArtifactReader(self._artifacts_root)
        doc_ids = reader.list_available_doc_ids()

        total_records = 0
        errors: list[str] = []

        for doc_id in doc_ids:
            try:
                result = self.reindex_document_from_artifacts(doc_id)
                total_records += result.source_records
                print(f"  Reindexed {doc_id}: {result.source_records} source records")
            except Exception as e:
                errors.append(f"{doc_id}: {e}")
                print(f"  [ERROR] {doc_id}: {e}")

        return {
            "doc_ids": doc_ids,
            "total_source_records": total_records,
            "errors": errors,
        }

    def rebuild_derived_index_from_all_caches(self) -> dict:
        """Rebuild sage_derived by scanning artifacts/*/derived/vision/*.json."""
        import glob

        pattern = str(self._artifacts_root / "*" / "derived" / "vision" / "*.json")
        cache_files = glob.glob(pattern)

        total = 0
        errors: list[str] = []

        for cache_file in cache_files:
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                doc_id = data["doc_id"]
                image_id = data["image_id"]
                image_ref = data.get("image_ref", "")
                current = data.get("current", {})
                analyses = data.get("analyses", [])

                # Build searchable text
                text = _build_derived_searchable_text(image_id, current, analyses)
                if not text.strip():
                    continue

                latest_model = analyses[-1]["model"] if analyses else "unknown"
                record_id = f"{doc_id}:derived:image:{image_id}"
                metadata = {
                    "doc_id": doc_id,
                    "record_type": "derived_image",
                    "image_id": image_id,
                    "image_ref": image_ref,
                    "derived_cache_ref": f"derived/vision/{image_id}.json",
                    "source_model": latest_model,
                }

                self.upsert_derived_record(record_id, text, metadata)
                total += 1
                print(f"  Re-indexed derived: {doc_id}/{image_id}")
            except Exception as e:
                errors.append(f"{cache_file}: {e}")
                print(f"  [ERROR] derived cache {cache_file}: {e}")

        return {"total_derived_records": total, "errors": errors}


# ---------------------------------------------------------------------------
# Derived text builder (shared with derived.py)
# ---------------------------------------------------------------------------

def _build_derived_searchable_text(
    image_id: str,
    current: dict,
    analyses: list[dict],
) -> str:
    """Compose searchable text from a derived image cache."""
    parts = [f"Image {image_id}"]

    description = current.get("description")
    if description:
        parts.append(f"Description:\n{description}")

    ocr_text = current.get("ocr_text")
    if ocr_text:
        parts.append(f"OCR:\n{ocr_text}")

    observations = current.get("observations")
    if observations:
        obs_lines = "\n".join(f"{k}: {v}" for k, v in observations.items())
        parts.append(f"Observations:\n{obs_lines}")

    if not (description or ocr_text or observations) and analyses:
        latest_raw = analyses[-1].get("raw_output", "")
        if latest_raw:
            parts.append(f"Latest raw analysis:\n{latest_raw}")

    return "\n\n".join(parts)
