"""
sage_document_db/__init__.py
SAGE Document Database — top-level facade.

The SageDocumentDB class is the primary interface for the future Gemma4 agent.
It wires together EmbeddingService, Chunker, ArtifactStore, ChromaStore,
RetrievalService, and derived.py into a single cohesive object.

The agent should call:
    db = SageDocumentDB()
    db.ingest_document(path)
    db.rag_search(query)
    db.exact_search(query)
    db.artifact_fetch(doc_id, element_id)
    db.add_image_analysis(doc_id, image_id, ...)
    db.reindex_document(doc_id)
    db.rebuild_all_indexes()
"""

from __future__ import annotations
from pathlib import Path

from .config import ARTIFACTS_ROOT, CHROMA_ROOT, DEFAULT_RAG_TOP_K
from .models import DerivedInsertResult, ExactSearchResult, RagResult


class SageDocumentDB:
    """Main facade for the SAGE document database layer.

    Instantiate once and share across the application.
    All heavy objects (EmbeddingService, Chunker, ChromaStore) are
    created once and reused for the process lifetime.
    """

    def __init__(
        self,
        artifacts_root: str | Path = ARTIFACTS_ROOT,
        chroma_root: str | Path = CHROMA_ROOT,
    ) -> None:
        from .artifact_store import ArtifactStore
        from .chunker import Chunker
        from .chroma_store import ChromaStore
        from .embeddings import EmbeddingService
        from .retrieval import RetrievalService

        self._artifacts_root = Path(artifacts_root)
        self._chroma_root = Path(chroma_root)

        # One EmbeddingService per process — shared by all components
        self._emb = EmbeddingService()

        # Chunker receives EmbeddingService for token counting (Correction 5)
        self._chunker = Chunker(embedding_service=self._emb)

        # Artifact store — canonical source truth
        self._store = ArtifactStore(self._artifacts_root)

        # Chroma store — disposable/rebuildable vector index
        self._chroma = ChromaStore(
            embedding_service=self._emb,
            chunker=self._chunker,
            chroma_root=self._chroma_root,
            artifacts_root=self._artifacts_root,
        )

        # Retrieval facade
        self._retrieval = RetrievalService(self._chroma, self._store)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_document(
        self,
        source_path: str | Path,
        *,
        index_in_chroma: bool = True,
        overwrite_artifacts: bool = False,
        debug: bool = False,
    ) -> dict:
        """Ingest a document and return a result summary dict."""
        from .pipeline import ingest_document
        return ingest_document(
            source_path,
            artifacts_root=self._artifacts_root,
            chroma_root=self._chroma_root,
            embedding_service=self._emb,
            chunker=self._chunker,
            chroma_store=self._chroma,
            index_in_chroma=index_in_chroma,
            overwrite_artifacts=overwrite_artifacts,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def rag_search(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_TOP_K,
        doc_ids: list[str] | None = None,
        include_source: bool = True,
        include_derived: bool = True,
    ) -> list[RagResult]:
        """Semantic search over indexed documents."""
        return self._retrieval.rag_search(
            query,
            top_k=top_k,
            doc_ids=doc_ids,
            include_source=include_source,
            include_derived=include_derived,
        )

    def exact_search(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 20,
    ) -> list[ExactSearchResult]:
        """Literal or regex search over canonical artifact files."""
        return self._retrieval.exact_search(
            query,
            doc_ids=doc_ids,
            case_sensitive=case_sensitive,
            regex=regex,
            max_results=max_results,
        )

    def artifact_fetch(
        self, doc_id: str, element_id: str
    ) -> dict:
        """Fetch the exact canonical element by ID."""
        return self._retrieval.artifact_fetch(doc_id, element_id)

    def list_artifacts(
        self, doc_id: str, artifact_type: str | None = None
    ) -> list[dict]:
        """Enumerate artifacts in a document.

        Args:
            doc_id:        The document identifier.
            artifact_type: Optional filter — "text", "image", "table", "code".

        Returns metadata only — does not load image bytes or full text.
        """
        return self._store.list_elements(doc_id, artifact_type=artifact_type)

    # ------------------------------------------------------------------
    # Derived image analysis
    # ------------------------------------------------------------------

    def add_image_analysis(
        self,
        doc_id: str,
        image_id: str,
        model: str,
        task_type: str,
        raw_output: str,
        *,
        instruction: str | None = None,
        ocr_text: str | None = None,
        description: str | None = None,
        observations: dict | None = None,
        searchable_text: str | None = None,
        index_in_chroma: bool = True,
    ) -> DerivedInsertResult:
        """Store a vision model analysis result.

        Set index_in_chroma=False for pending/non-accepted attempts
        or mock-mode results that should not become searchable knowledge.
        Raw attempt is ALWAYS preserved in the JSON cache regardless.
        """
        from .derived import add_image_analysis
        return add_image_analysis(
            doc_id=doc_id,
            image_id=image_id,
            model=model,
            task_type=task_type,
            raw_output=raw_output,
            instruction=instruction,
            ocr_text=ocr_text,
            description=description,
            observations=observations,
            searchable_text=searchable_text,
            index_in_chroma=index_in_chroma,
            artifacts_root=self._artifacts_root,
            chroma_store=self._chroma,
        )

    # ------------------------------------------------------------------
    # Reindex
    # ------------------------------------------------------------------

    def reindex_document(self, doc_id: str) -> dict:
        """Rebuild the Chroma source index for one document from artifacts."""
        result = self._chroma.reindex_document_from_artifacts(doc_id)
        return {
            "doc_id": result.doc_id,
            "source_records": result.source_records,
        }

    def rebuild_all_indexes(self) -> dict:
        """Rebuild both sage_source and sage_derived from all artifacts.

        Does NOT delete artifacts. Safe to run after deleting chroma_db/.
        """
        source_result = self._chroma.rebuild_source_index_from_all_artifacts()
        derived_result = self._chroma.rebuild_derived_index_from_all_caches()
        return {
            "source": source_result,
            "derived": derived_result,
        }

    # ------------------------------------------------------------------
    # Rich Output Sockets (Phases B, C, D, E, J, K, L)
    # ------------------------------------------------------------------

    def rag_search_socket(
        self,
        query: str,
        top_k: int = DEFAULT_RAG_TOP_K,
        doc_ids: list[str] | None = None,
        include_source: bool = True,
        include_derived: bool = True,
    ) -> dict:
        """Execute RAG search and return the complete Component B socket."""
        return self._retrieval.rag_search_socket(
            query=query,
            top_k=top_k,
            doc_ids=doc_ids,
            include_source=include_source,
            include_derived=include_derived,
        )

    def exact_search_socket(
        self,
        query: str,
        doc_ids: list[str] | None = None,
        case_sensitive: bool = False,
        regex: bool = False,
        max_results: int = 20,
    ) -> dict:
        """Execute exact search and return the complete Component C socket."""
        return self._retrieval.exact_search_socket(
            query=query,
            doc_ids=doc_ids,
            case_sensitive=case_sensitive,
            regex=regex,
            max_results=max_results,
        )

    def artifact_fetch_socket(
        self, doc_id: str, element_id: str
    ) -> dict:
        """Fetch canonical element and return the complete Component D socket."""
        return self._retrieval.artifact_fetch_socket(doc_id, element_id)

    def list_artifacts_socket(
        self, doc_id: str, artifact_type: str | None = None
    ) -> dict:
        """Enumerate artifacts and return the complete Component E socket."""
        return self._store.list_elements_socket(doc_id, artifact_type=artifact_type)

    def add_image_analysis_socket(
        self,
        doc_id: str,
        image_id: str,
        model: str,
        task_type: str,
        raw_output: str,
        *,
        instruction: str | None = None,
        ocr_text: str | None = None,
        description: str | None = None,
        observations: dict | None = None,
        searchable_text: str | None = None,
        index_in_chroma: bool = True,
    ) -> dict:
        """Store vision model analysis and return the complete Component J socket."""
        from .derived import add_image_analysis_socket
        return add_image_analysis_socket(
            doc_id=doc_id,
            image_id=image_id,
            model=model,
            task_type=task_type,
            raw_output=raw_output,
            instruction=instruction,
            ocr_text=ocr_text,
            description=description,
            observations=observations,
            searchable_text=searchable_text,
            index_in_chroma=index_in_chroma,
            artifacts_root=self._artifacts_root,
            chroma_store=self._chroma,
        )

    def reindex_document_socket(self, doc_id: str) -> dict:
        """Rebuild Chroma index for one document and return Component K socket."""
        import time
        from core.sockets import build_reindex_document_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            result = self._chroma.reindex_document_from_artifacts(doc_id)
            manifest = self._artifacts.load_manifest(doc_id)
            total_elements = len(manifest.get("element_index", {}))
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            return build_reindex_document_socket(
                doc_id=doc_id,
                source_records_rebuilt=result.source_records,
                embedded_records=result.embedded_records,
                canonical_artifacts_discovered=total_elements,
                chunks_rebuilt=result.source_records,
                derived_analyses_discovered=0,
                derived_records_rebuilt=0,
                skipped_records=0,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload("REINDEX_ERROR", type(e).__name__, str(e), exc=e)
            return build_reindex_document_socket(
                doc_id=doc_id,
                source_records_rebuilt=0,
                embedded_records=0,
                canonical_artifacts_discovered=0,
                chunks_rebuilt=0,
                derived_analyses_discovered=0,
                derived_records_rebuilt=0,
                skipped_records=0,
                timing=timing,
                error=error_payload,
            )

    def rebuild_all_indexes_socket(self) -> dict:
        """Rebuild all Chroma indexes and return Component L socket."""
        import time
        from core.sockets import build_rebuild_indexes_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            source_result = self._chroma.rebuild_source_index_from_all_artifacts()
            derived_result = self._chroma.rebuild_derived_index_from_all_caches()
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)

            total_docs = source_result.get("documents", 0)
            total_recs = source_result.get("records", 0) + derived_result.get("records", 0)

            return build_rebuild_indexes_socket(
                source_summary=source_result,
                derived_summary=derived_result,
                total_documents=total_docs,
                total_records=total_recs,
                chroma_root=str(self._chroma_root),
                artifacts_root=str(self._artifacts_root),
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload("REBUILD_INDEXES_ERROR", type(e).__name__, str(e), exc=e)
            return build_rebuild_indexes_socket(
                source_summary={},
                derived_summary={},
                total_documents=0,
                total_records=0,
                chroma_root=str(self._chroma_root),
                artifacts_root=str(self._artifacts_root),
                timing=timing,
                error=error_payload,
            )
