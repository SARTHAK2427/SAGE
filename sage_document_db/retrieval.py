"""
sage_document_db/retrieval.py
Clean retrieval facade.

The future Gemma4 agent sees ONLY these three functions:
    rag_search(...)      — semantic vector search
    exact_search(...)    — literal/regex search on canonical artifacts
    artifact_fetch(...)  — known-ID direct element retrieval

This module hides all Chroma internals, artifact directory traversal,
embedding model API, and Docling internals from the caller.

Mental contract frozen in code:
    CHROMA:    Where is relevant information likely to be?
    ARTIFACTS: What exactly was present in the source?
"""

from __future__ import annotations
from pathlib import Path

from .config import ARTIFACTS_ROOT, CHROMA_ROOT, DEFAULT_RAG_TOP_K
from .models import ExactSearchResult, RagResult


class RetrievalService:
    """Unified facade over RAG, exact search, and artifact fetch.

    Typically instantiated once via SageDocumentDB and reused.
    """

    def __init__(self, chroma_store, artifact_store) -> None:
        self._chroma = chroma_store
        self._artifacts = artifact_store

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def rag_search(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_TOP_K,
        doc_ids: list[str] | None = None,
        include_source: bool = True,
        include_derived: bool = True,
    ) -> list[RagResult]:
        """Semantic vector search over sage_source and/or sage_derived.

        Correction 2: doc_ids filtering uses Chroma where clause at
        query time — this is passed through to ChromaStore.rag_query().

        Algorithm:
        1. Validate query
        2. Embed query once
        3. Query sage_source (if enabled) with Chroma where filter
        4. Query sage_derived (if enabled) with same filter
        5. Normalize, merge by distance, deduplicate, return top_k

        No LLM is called. No distance threshold discards.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query must be a non-empty string.")

        query_vector = self._chroma._emb.embed_query(query)

        results = self._chroma.rag_query(
            query_vector=query_vector,
            top_k=top_k,
            doc_ids=doc_ids,
            include_source=include_source,
            include_derived=include_derived,
        )
        return results

    # ------------------------------------------------------------------
    # Exact / literal search
    # ------------------------------------------------------------------

    def exact_search(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 20,
    ) -> list[ExactSearchResult]:
        """Literal or regex search over canonical artifact files.

        Correction 6: derived/ is explicitly excluded inside exact_search.py.
        Only text/, tables/, code/ are searched.

        Raises ValueError for invalid regex patterns.
        """
        from .exact_search import exact_search as _exact_search
        return _exact_search(
            query=query,
            artifacts_root=self._artifacts.root,
            doc_ids=doc_ids,
            case_sensitive=case_sensitive,
            regex=regex,
            max_results=max_results,
        )

    # ------------------------------------------------------------------
    # Direct artifact fetch (Correction 1)
    # ------------------------------------------------------------------

    def artifact_fetch(
        self, doc_id: str, element_id: str
    ) -> dict:
        """Fetch the exact canonical element for a known element_id."""
        return self._artifacts.fetch_element(doc_id, element_id)

    # ------------------------------------------------------------------
    # Rich Output Sockets (Phases A, B, C, D)
    # ------------------------------------------------------------------

    def rag_search_socket(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_TOP_K,
        doc_ids: list[str] | None = None,
        include_source: bool = True,
        include_derived: bool = True,
    ) -> dict:
        """Execute RAG search and return the complete rich output socket."""
        import time
        from core.sockets import build_rag_search_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        stages: dict[str, float] = {}
        try:
            query_clean = query.strip()
            if not query_clean:
                raise ValueError("Query must be a non-empty string.")

            t_emb = time.perf_counter()
            query_vector = self._chroma._emb.embed_query(query_clean)
            stages["embedding_ms"] = (time.perf_counter() - t_emb) * 1000.0

            t_query = time.perf_counter()
            results = self._chroma.rag_query(
                query_vector=query_vector,
                top_k=top_k,
                doc_ids=doc_ids,
                include_source=include_source,
                include_derived=include_derived,
            )
            stages["chroma_query_ms"] = (time.perf_counter() - t_query) * 1000.0

            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms, stages=stages)
            return build_rag_search_socket(
                query=query_clean,
                records=[r.to_dict() for r in results],
                requested_top_k=top_k,
                doc_ids_filter=doc_ids,
                include_source=include_source,
                include_derived=include_derived,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms, stages=stages)
            error_payload = build_error_payload(
                code="RAG_SEARCH_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_rag_search_socket(
                query=query,
                records=[],
                requested_top_k=top_k,
                doc_ids_filter=doc_ids,
                include_source=include_source,
                include_derived=include_derived,
                timing=timing,
                error=error_payload,
            )

    def exact_search_socket(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 20,
    ) -> dict:
        """Execute exact search and return the complete rich output socket."""
        import time
        from core.sockets import build_exact_search_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            results = self.exact_search(
                query=query,
                doc_ids=doc_ids,
                case_sensitive=case_sensitive,
                regex=regex,
                max_results=max_results,
            )
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            return build_exact_search_socket(
                query=query,
                matches=[r.to_dict() for r in results],
                max_results=max_results,
                doc_ids_filter=doc_ids,
                case_sensitive=case_sensitive,
                regex=regex,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="EXACT_SEARCH_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_exact_search_socket(
                query=query,
                matches=[],
                max_results=max_results,
                doc_ids_filter=doc_ids,
                case_sensitive=case_sensitive,
                regex=regex,
                timing=timing,
                error=error_payload,
            )

    def artifact_fetch_socket(
        self, doc_id: str, element_id: str
    ) -> dict:
        """Fetch exact canonical element and return the complete rich output socket."""
        return self._artifacts.fetch_element_socket(doc_id, element_id)
