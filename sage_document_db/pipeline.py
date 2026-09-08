"""
sage_document_db/pipeline.py
Main ingestion pipeline.

ingest_document() is the single entry point for adding documents.
It orchestrates: router → parser → ArtifactStore → ChromaStore.

Rules:
- Idempotent: same SHA-256 → same doc_id, no duplication.
- If artifacts write succeeds but Chroma fails: preserve artifacts,
  report error, allow later reindex_document_from_artifacts().
- Artifacts have priority over Chroma.
- debug=True prints element-by-element and chunk-by-chunk summaries.
"""

from __future__ import annotations
from pathlib import Path

from .config import ARTIFACTS_ROOT, CHROMA_ROOT
from .router import get_parser_tag


def ingest_document(
    source_path: str | Path,
    *,
    artifacts_root: str | Path = ARTIFACTS_ROOT,
    chroma_root: str | Path = CHROMA_ROOT,
    embedding_service=None,     # EmbeddingService instance (shared)
    chunker=None,               # Chunker instance (shared)
    chroma_store=None,          # ChromaStore instance (shared)
    index_in_chroma: bool = True,
    overwrite_artifacts: bool = False,
    debug: bool = False,
) -> dict:
    """Ingest a document into the SAGE document database.

    Pipeline:
        source_path
          → router (extension detection)
          → parser (Docling or simple)
          → NormalizedDocument
          → ArtifactStore.write_document()
          → ChromaStore.index_document()  [shared Chunker + EmbeddingService]
          → return result dict

    Idempotency:
        If the file was already ingested (same SHA-256 → same doc_id) and
        overwrite_artifacts=False: skip writing artifacts, optionally ensure
        Chroma index exists, return existing doc_id.

    Returns:
        dict with keys: doc_id, artifact_dir, manifest_path, parser,
                        counts, index, warnings
    """
    import time
    from core.sockets import iso_now, build_timing_payload, build_error_payload, build_socket_envelope

    t_start = time.perf_counter()
    started_at = iso_now()
    stage_timings: dict[str, float] = {}

    from .artifact_store import ArtifactStore
    from .utils import sha256_file, make_doc_id

    source_path = Path(source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found: '{source_path}'")

    file_size_bytes = source_path.stat().st_size
    artifacts_root = Path(artifacts_root)
    store = ArtifactStore(artifacts_root)

    # --- Identify document ---
    sha = sha256_file(source_path)
    doc_id = make_doc_id(sha)
    parser_tag = get_parser_tag(source_path)

    if debug:
        print(f"\n[INGEST] {source_path.name}")
        print(f"  Parser tag : {parser_tag}")
        print(f"  SHA-256    : {sha}")
        print(f"  doc_id     : {doc_id}")

    # --- Idempotency check ---
    artifact_exists = store.document_exists(doc_id)
    skip_write = False

    if artifact_exists and not overwrite_artifacts:
        # Verify SHA matches stored manifest
        try:
            manifest = store.load_manifest(doc_id)
            stored_sha = manifest.get("origin", {}).get("sha256", "")
            if stored_sha == sha:
                skip_write = True
                if debug:
                    print(f"  [SKIP] Artifact already exists and SHA matches.")
        except Exception:
            skip_write = False

    # --- Sanitize Office ZIP if needed (never edit original in-place) ---
    from .zip_sanitizer import sanitize_office_zip
    sanitation = sanitize_office_zip(source_path)
    working_path = sanitation.working_path

    # --- Parse ---
    doc = None
    t_parse_start = time.perf_counter()
    try:
        if not skip_write:
            doc = _parse(
                parser_tag,
                working_path,
                original_source_path=source_path,
                original_sha=sha,
                original_size=file_size_bytes,
                doc_id=doc_id,
            )
            if sanitation.was_sanitized and sanitation.sanitation_warning:
                doc.warnings.append(sanitation.sanitation_warning)
            if debug:
                _print_elements(doc)
        stage_timings["parse_ms"] = (time.perf_counter() - t_parse_start) * 1000.0

        # --- Write artifacts ---
        t_write_start = time.perf_counter()
        if not skip_write:
            doc_dir = store.write_document(doc)
        else:
            doc_dir = store.doc_dir(doc_id)
        stage_timings["artifact_write_ms"] = (time.perf_counter() - t_write_start) * 1000.0
    finally:
        # Clean up temporary sanitized copy if one was created
        if sanitation.was_sanitized and sanitation.working_path != source_path:
            try:
                if sanitation.working_path.exists():
                    sanitation.working_path.unlink()
            except Exception:
                pass

    # --- Load counts from manifest ---
    manifest = store.load_manifest(doc_id)
    index_data = manifest.get("element_index", {})
    counts = _count_elements(index_data)

    # --- Index in Chroma ---
    index_result_dict = {"source_records": 0}
    index_error: str | None = None
    indexed_records = 0

    t_chroma_start = time.perf_counter()
    if index_in_chroma:
        if chroma_store is None:
            if debug:
                print("  [WARN] No chroma_store provided; skipping Chroma indexing.")
        else:
            try:
                if skip_write and doc is None:
                    # Rebuild from artifacts using ArtifactReader
                    idx_result = chroma_store.reindex_document_from_artifacts(doc_id)
                else:
                    idx_result = chroma_store.index_document(doc, replace_existing=True)

                indexed_records = idx_result.source_records
                index_result_dict = {
                    "source_records": indexed_records,
                }
                if debug:
                    print(f"\n  Indexed {idx_result.source_records} source records into Chroma.")
            except Exception as e:
                index_error = str(e)
                print(
                    f"[ERROR] Chroma indexing failed for {doc_id}: {e}\n"
                    f"  Artifacts are preserved. Run reindex-doc {doc_id} to retry."
                )
    stage_timings["chroma_index_ms"] = (time.perf_counter() - t_chroma_start) * 1000.0

    duration_ms = (time.perf_counter() - t_start) * 1000.0
    finished_at = iso_now()

    # Determine overall status
    if index_error:
        status = "partial"  # Artifacts succeeded, vector index failed
        error_payload = build_error_payload(
            code="CHROMA_INDEXING_FAILED",
            type_name="ChromaIndexingError",
            message=index_error,
            recoverable=True,
            retryable=True,
            details={"doc_id": doc_id},
        )
    else:
        status = "success"
        error_payload = None

    timing_payload = build_timing_payload(
        started_at=started_at,
        finished_at=finished_at,
        duration_ms=duration_ms,
        stages=stage_timings,
    )

    manifest_origin = manifest.get("origin", {})
    mime_type = manifest_origin.get("mime_type")
    parser_name = doc.parser_name if doc else manifest.get("parser", {}).get("name", parser_tag)
    parser_version = doc.parser_version if doc else manifest.get("parser", {}).get("version")

    # Full rich socket
    result = {
        # --- Standard Socket Envelope ---
        "status": status,
        "operation": "document_ingestion",
        "result": {
            "doc_id": doc_id,
            "counts": counts,
            "indexed_source_records": indexed_records,
            "is_duplicate": skip_write,
        },
        "identity": {
            "doc_id": doc_id,
            "original_filename": source_path.name,
            "extension": source_path.suffix.lstrip(".").lower(),
            "mime_type": mime_type,
            "sha256": sha,
            "file_size_bytes": file_size_bytes,
        },
        "context": {
            "already_ingested": skip_write,
            "overwrite_artifacts": overwrite_artifacts,
            "index_in_chroma_requested": index_in_chroma,
            "parser": {
                "name": parser_name,
                "tag": parser_tag,
                "version": parser_version,
            },
        },
        "provenance": {
            "doc_id": doc_id,
            "source_path": str(source_path).replace("\\", "/"),
            "sha256": sha,
        },
        "storage": {
            "artifact_dir": str(doc_dir).replace("\\", "/"),
            "manifest_path": str(doc_dir / "manifest.json").replace("\\", "/"),
            "raw_source_retained": True,
        },
        "execution": {
            "chroma_indexing": {
                "performed": index_in_chroma and (chroma_store is not None) and (index_error is None),
                "source_records": indexed_records,
                "collection": "sage_source",
            },
        },
        "timing": timing_payload,
        "warnings": manifest.get("warnings", []),
        "error": error_payload,

        # --- Backward-compatible legacy flat keys ---
        "doc_id": doc_id,
        "artifact_dir": str(doc_dir),
        "manifest_path": str(doc_dir / "manifest.json"),
        "parser": parser_tag,
        "counts": counts,
        "index": index_result_dict,
    }
    if index_error:
        result["index_error"] = index_error

    return result


# ---------------------------------------------------------------------------
# Parser dispatch
# ---------------------------------------------------------------------------

def _parse(
    parser_tag: str,
    source_path: Path,
    original_source_path: Path | None = None,
    original_sha: str | None = None,
    original_size: int | None = None,
    doc_id: str | None = None,
):
    orig_path = original_source_path or source_path
    if parser_tag == "docling":
        from .docling_parser import parse_with_docling
        try:
            doc = parse_with_docling(source_path)
            # Ensure provenance reflects original uploaded file
            if original_source_path is not None:
                doc.source_path = str(orig_path)
                doc.original_name = orig_path.name
            if original_sha is not None:
                doc.sha256 = original_sha
            if original_size is not None:
                doc.file_size_bytes = original_size
            if doc_id is not None:
                doc.doc_id = doc_id
            return doc
        except Exception as docling_err:
            if orig_path.suffix.lower() == ".docx":
                from .docx_fallback import parse_docx_fallback
                from .utils import make_doc_id
                effective_sha = original_sha or sha256_file(orig_path)
                return parse_docx_fallback(
                    working_path=source_path,
                    original_source_path=orig_path,
                    doc_id=doc_id or make_doc_id(effective_sha),
                    original_sha=effective_sha,
                    original_size=original_size or orig_path.stat().st_size,
                    docling_error=str(docling_err),
                )
            raise
    elif parser_tag == "txt":
        from .simple_parsers import parse_txt
        return parse_txt(source_path)
    elif parser_tag == "json":
        from .simple_parsers import parse_json
        return parse_json(source_path)
    elif parser_tag == "csv":
        from .simple_parsers import parse_csv
        return parse_csv(source_path)
    elif parser_tag == "image":
        from .simple_parsers import parse_image
        return parse_image(source_path)
    else:
        raise ValueError(f"Unknown parser tag: '{parser_tag}'")


def _count_elements(index_data: dict) -> dict:
    counts = {"text": 0, "images": 0, "tables": 0, "code": 0, "links": 0}
    for entry in index_data.values():
        el_type = entry.get("type", "text")
        if el_type == "text":
            counts["text"] += 1
        elif el_type == "image":
            counts["images"] += 1
        elif el_type == "table":
            counts["tables"] += 1
        elif el_type == "code":
            counts["code"] += 1
    return counts


def _print_elements(doc) -> None:
    """Debug: print normalized reading order."""
    print(f"\n  Reading order ({len(doc.elements)} elements):")
    for el in doc.elements:
        loc = ""
        if el.page is not None:
            loc = f"page={el.page}"
        elif el.slide is not None:
            loc = f"slide={el.slide}"
        elif el.sheet is not None:
            loc = f"sheet={el.sheet}"
        sub = f" subtype={el.subtype}" if el.subtype else ""
        print(f"  {el.order+1:03d} {el.type.upper():8s} {loc:10s} id={el.id}{sub}")
    if doc.warnings:
        print(f"  Warnings: {len(doc.warnings)}")
        for w in doc.warnings[:5]:
            print(f"    - {w}")
