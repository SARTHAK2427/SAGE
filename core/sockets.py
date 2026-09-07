"""
SAGE/core/sockets.py
Standardized rich output socket infrastructure for deterministic SAGE components.

Core architectural principle:
    COMPONENT
       ↓
    FULL RICH OUTPUT SOCKET  (100% JSON-serializable, maximum useful data)
       ↓
    PYTHON MAPPER / CONNECTOR
       ↓
    SELECTED SUBSET / TRANSFORMED DATA
       ↓
    NEXT COMPONENT INPUT PLUG

Data Visibility Classifications:
    SAFE_FOR_AGENT   — May be presented directly to semantic LLM / Gemma.
    SAFE_FOR_UI      — May be displayed directly in user-facing UI / frontend.
    INTERNAL_ONLY    — Backend-internal plumbing (e.g. local host paths, PIDs).
    LOG_ONLY         — Diagnostics, stack traces, low-level debug dumps.
    PERSISTENCE_ONLY — Cold storage, manifest state, historical caches.
"""

from __future__ import annotations
import datetime
import hashlib
import json
import traceback
from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ─── Data Visibility Classification ──────────────────────────────────────────

class Visibility:
    SAFE_FOR_AGENT = "SAFE_FOR_AGENT"
    SAFE_FOR_UI = "SAFE_FOR_UI"
    INTERNAL_ONLY = "INTERNAL_ONLY"
    LOG_ONLY = "LOG_ONLY"
    PERSISTENCE_ONLY = "PERSISTENCE_ONLY"


# ─── JSON Serialization Helpers ──────────────────────────────────────────────

def to_json_serializable(obj: Any) -> Any:
    """Recursively convert Python objects, Paths, dataclasses, sets into JSON primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj).replace("\\", "/")
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted([to_json_serializable(x) for x in obj], key=lambda x: str(x))
    if isinstance(obj, dict):
        return {str(k): to_json_serializable(v) for k, v in obj.items()}
    if hasattr(obj, "to_socket"):
        return obj.to_socket()
    if is_dataclass(obj) and not isinstance(obj, type):
        return to_json_serializable(asdict(obj))
    if hasattr(obj, "to_dict"):
        return to_json_serializable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        clean = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return to_json_serializable(clean)
    return str(obj)


def iso_now() -> str:
    """Return UTC ISO-8601 formatted timestamp with 'Z' suffix."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


# ─── Standard Timing Builder ──────────────────────────────────────────────────

def build_timing_payload(
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    duration_ms: Optional[float] = None,
    stages: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Build a compliant timing dictionary."""
    t: Dict[str, Any] = {
        "started_at": started_at or iso_now(),
        "finished_at": finished_at or iso_now(),
        "duration_ms": round(float(duration_ms if duration_ms is not None else 0.0), 3),
    }
    if stages:
        t["stages"] = {k: round(float(v), 3) for k, v in stages.items()}
    return t


# ─── Standard Error Builder ───────────────────────────────────────────────────

def build_error_payload(
    code: str,
    type_name: str,
    message: str,
    recoverable: bool = False,
    retryable: bool = False,
    details: Optional[Dict[str, Any]] = None,
    exc: Optional[BaseException] = None,
    source_file: Optional[str] = None,
    source_function: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a compliant error dictionary."""
    err: Dict[str, Any] = {
        "code": code,
        "type": type_name,
        "message": message,
        "recoverable": recoverable,
        "retryable": retryable,
        "details": to_json_serializable(details or {}),
    }
    if exc is not None or source_file is not None or source_function is not None:
        debug: Dict[str, Any] = {}
        if exc is not None:
            debug["traceback"] = traceback.format_exc()
        if source_file is not None:
            debug["source_file"] = str(source_file).replace("\\", "/")
        if source_function is not None:
            debug["source_function"] = source_function
        err["debug"] = debug
    return err


# ─── Standard Top-Level Envelope ──────────────────────────────────────────────

def build_socket_envelope(
    status: str,
    operation: str,
    result: Optional[Any] = None,
    identity: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    provenance: Optional[Dict[str, Any]] = None,
    references: Optional[Dict[str, Any]] = None,
    execution: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, Any]] = None,
    storage: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
    raw: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build the common top-level socket dictionary."""
    socket: Dict[str, Any] = {
        "status": status,
        "operation": operation,
        "result": to_json_serializable(result) if result is not None else {},
    }
    if identity is not None:
        socket["identity"] = to_json_serializable(identity)
    if context is not None:
        socket["context"] = to_json_serializable(context)
    if provenance is not None:
        socket["provenance"] = to_json_serializable(provenance)
    if references is not None:
        socket["references"] = to_json_serializable(references)
    if execution is not None:
        socket["execution"] = to_json_serializable(execution)
    socket["timing"] = to_json_serializable(timing) if timing is not None else build_timing_payload()
    if storage is not None:
        socket["storage"] = to_json_serializable(storage)
    
    socket["warnings"] = [str(w) for w in (warnings or [])]
    socket["error"] = to_json_serializable(error) if error is not None else None

    if raw is not None:
        socket["raw"] = to_json_serializable(raw)

    return socket


# ==============================================================================
# INDIVIDUAL DETERMINISTIC SOCKET BUILDERS (A THROUGH W)
# ==============================================================================

# ── A. Document Ingestion Socket ─────────────────────────────────────────────

def build_ingestion_socket(
    status: str,
    doc_id: str,
    original_filename: str,
    sha256: str,
    file_size_bytes: int,
    extension: str,
    parser_name: str,
    parser_tag: str,
    counts: Dict[str, int],
    artifact_dir: str,
    manifest_path: str,
    indexed_source_records: int,
    already_ingested: bool = False,
    overwrite_artifacts: bool = False,
    mime_type: Optional[str] = None,
    parser_version: Optional[str] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_socket_envelope(
        status=status,
        operation="document_ingestion",
        result={
            "doc_id": doc_id,
            "counts": counts,
            "indexed_source_records": indexed_source_records,
            "is_duplicate": already_ingested,
        },
        identity={
            "doc_id": doc_id,
            "original_filename": original_filename,
            "extension": extension,
            "mime_type": mime_type,
            "sha256": sha256,
            "file_size_bytes": file_size_bytes,
        },
        context={
            "already_ingested": already_ingested,
            "overwrite_artifacts": overwrite_artifacts,
            "index_in_chroma_requested": indexed_source_records > 0 or error is not None,
            "parser": {
                "name": parser_name,
                "tag": parser_tag,
                "version": parser_version,
            },
        },
        provenance={
            "doc_id": doc_id,
            "source_path": original_filename,
            "sha256": sha256,
        },
        storage={
            "artifact_dir": str(artifact_dir).replace("\\", "/"),
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "raw_source_retained": True,
        },
        execution={
            "chroma_indexing": {
                "performed": indexed_source_records > 0 and error is None,
                "source_records": indexed_source_records,
                "collection": "sage_source",
            },
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── B. RAG Search Socket ─────────────────────────────────────────────────────

def build_rag_search_socket(
    query: str,
    records: List[Any],
    requested_top_k: int,
    doc_ids_filter: Optional[List[str]] = None,
    include_source: bool = True,
    include_derived: bool = True,
    collections_queried: Optional[List[str]] = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    serialized_records = []
    for r in records:
        rec = to_json_serializable(r)
        if isinstance(rec, dict):
            if "raw_distance" in rec and "derived_cosine_similarity" not in rec:
                rec["derived_cosine_similarity"] = round(1.0 - float(rec["raw_distance"]), 6)
            if "distance" in rec and "raw_distance" not in rec:
                rec["raw_distance"] = float(rec["distance"])
            if "raw_distance" in rec and "derived_similarity" not in rec:
                rec["derived_similarity"] = round(max(0.0, min(1.0, 1.0 - float(rec["raw_distance"]))), 6)
        serialized_records.append(rec)
    limit_hit = len(serialized_records) >= requested_top_k
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="rag_search",
        result={
            "query": query,
            "count": len(serialized_records),
            "limit_hit": limit_hit,
            "records": serialized_records,
        },
        context={
            "query": query,
            "requested_top_k": requested_top_k,
            "doc_ids_filter": doc_ids_filter,
            "include_source": include_source,
            "include_derived": include_derived,
            "collections_queried": collections_queried or (
                (["sage_source"] if include_source else []) + (["sage_derived"] if include_derived else [])
            ),
        },
        execution={
            "embedding_model": embedding_model,
            "distance_metric": "cosine",
            "similarity_formula": "derived_cosine_similarity = 1.0 - raw_distance",
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── C. Exact Search Socket ───────────────────────────────────────────────────

def build_exact_search_socket(
    query: str,
    matches: List[Any],
    max_results: int,
    doc_ids_filter: Optional[List[str]] = None,
    case_sensitive: bool = False,
    regex: bool = False,
    searched_subdirs: Optional[List[str]] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    serialized_matches = [to_json_serializable(m) for m in matches]
    status = "error" if error else "success"
    limit_hit = len(serialized_matches) >= max_results

    return build_socket_envelope(
        status=status,
        operation="exact_search",
        result={
            "query": query,
            "count": len(serialized_matches),
            "max_results_hit": limit_hit,
            "matches": serialized_matches,
        },
        context={
            "query": query,
            "case_sensitive": case_sensitive,
            "regex": regex,
            "doc_ids_filter": doc_ids_filter,
            "max_results": max_results,
            "searched_canonical_subdirs": searched_subdirs or ["text", "tables", "code"],
            "derived_excluded": True,
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── D. Artifact Fetch Socket ─────────────────────────────────────────────────

def build_artifact_fetch_socket(
    doc_id: str,
    element_id: str,
    element: Optional[Dict[str, Any]] = None,
    found: bool = True,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error or not found else "success"
    el_dict = to_json_serializable(element or {})
    element_type = el_dict.get("type", "unknown")

    return build_socket_envelope(
        status=status,
        operation="artifact_fetch",
        result={
            "found": found,
            "element": el_dict,
        },
        identity={
            "doc_id": doc_id,
            "element_id": element_id,
            "element_type": element_type,
            "subtype": el_dict.get("subtype"),
        },
        provenance={
            "doc_id": doc_id,
            "element_id": element_id,
            "order": el_dict.get("order"),
            "page": el_dict.get("page"),
            "slide": el_dict.get("slide"),
            "sheet": el_dict.get("sheet"),
            "bbox": el_dict.get("bbox"),
        },
        storage={
            "ref": el_dict.get("ref"),
            "local_path": el_dict.get("local_path"),
            "exists": el_dict.get("exists", True if found else False),
        },
        references={
            "links": el_dict.get("links", []),
            "previous_element": el_dict.get("previous_element"),
            "next_element": el_dict.get("next_element"),
        },
        execution={
            "visibility_mapping": {
                "local_path": Visibility.INTERNAL_ONLY,
                "text": Visibility.SAFE_FOR_AGENT,
                "table_data": Visibility.SAFE_FOR_AGENT,
            }
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── E. List Artifacts Socket ─────────────────────────────────────────────────

def build_list_artifacts_socket(
    doc_id: str,
    artifacts: List[Dict[str, Any]],
    artifact_type_filter: Optional[str] = None,
    total_elements_in_doc: Optional[int] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    serialized = [to_json_serializable(a) for a in artifacts]
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="list_artifacts",
        result={
            "doc_id": doc_id,
            "count": len(serialized),
            "total_in_document": total_elements_in_doc if total_elements_in_doc is not None else len(serialized),
            "artifacts": serialized,
        },
        identity={
            "doc_id": doc_id,
        },
        context={
            "doc_id": doc_id,
            "artifact_type_filter": artifact_type_filter,
            "deterministic_sort": "order_ascending",
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── F. Embedding Socket ──────────────────────────────────────────────────────

def build_embedding_socket(
    model_name: str,
    device: str,
    dimension: int,
    input_count: int,
    vectors: Optional[List[List[float]]] = None,
    batch_size: int = 32,
    dtype: str = "float32",
    is_mock: bool = False,
    approx_tokens: Optional[int] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"
    v_count = len(vectors) if vectors is not None else 0

    return build_socket_envelope(
        status=status,
        operation="embedding",
        result={
            "vector_count": v_count,
            "dimension": dimension,
            "output_shape": [v_count, dimension],
            "vectors": vectors if vectors is not None else [],
        },
        identity={
            "model_name": model_name,
            "device": device,
            "dimension": dimension,
            "dtype": dtype,
            "is_mock": is_mock,
        },
        context={
            "input_count": input_count,
            "approx_tokens": approx_tokens,
            "batch_size": batch_size,
            "normalize_embeddings": True,
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── G. Chroma Query Socket ───────────────────────────────────────────────────

def build_chroma_query_socket(
    collections_queried: List[str],
    query_count: int,
    requested_top_k: int,
    returned_count: int,
    collection_responses: Dict[str, Any],
    raw_chroma_responses: Optional[Dict[str, Any]] = None,
    where_filter: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="chroma_query",
        result={
            "returned_count": returned_count,
            "collection_responses": to_json_serializable(collection_responses),
        },
        context={
            "collections_queried": collections_queried,
            "query_count": query_count,
            "requested_top_k": requested_top_k,
            "where_filter": where_filter,
            "include_fields": ["documents", "metadatas", "distances"],
        },
        raw={"raw_chroma_responses": to_json_serializable(raw_chroma_responses)} if raw_chroma_responses else None,
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── H. Chroma Upsert / Indexing Socket ───────────────────────────────────────

def build_chroma_upsert_socket(
    collection_name: str,
    record_count: int,
    embedded_count: int,
    doc_ids_affected: List[str],
    ids_written: Optional[List[str]] = None,
    requested_count: Optional[int] = None,
    successful_count: Optional[int] = None,
    failed_count: int = 0,
    skipped_count: int = 0,
    replace_existing: bool = True,
    batches_count: int = 1,
    origin: str = "source",
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="chroma_upsert",
        result={
            "collection": collection_name,
            "record_count": record_count,
            "embedded_count": embedded_count,
            "requested_count": requested_count if requested_count is not None else record_count,
            "successful_count": successful_count if successful_count is not None else record_count,
            "failed_count": failed_count,
            "skipped_count": skipped_count,
            "ids_written": ids_written or [],
            "atomic_batch": True,
        },
        identity={
            "collection_name": collection_name,
            "origin": origin,
            "doc_ids_affected": doc_ids_affected,
        },
        context={
            "replace_existing": replace_existing,
            "batches_count": batches_count,
            "upsert_semantics": "atomic_batch_upsert",
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── I. Artifact Write Socket ─────────────────────────────────────────────────

def build_artifact_write_socket(
    doc_id: str,
    artifact_dir: str,
    manifest_path: str,
    files_written: List[Dict[str, Any]],
    element_counts: Dict[str, int],
    manifest_written: bool = True,
    artifact_dir_created: bool = True,
    overwrite: bool = False,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="artifact_write",
        result={
            "doc_id": doc_id,
            "files_written_count": len(files_written),
            "element_counts": element_counts,
        },
        identity={
            "doc_id": doc_id,
        },
        storage={
            "artifact_dir": str(artifact_dir).replace("\\", "/"),
            "manifest_path": str(manifest_path).replace("\\", "/"),
            "files_written": to_json_serializable(files_written),
            "manifest_written": manifest_written,
            "artifact_dir_created": artifact_dir_created,
            "overwrite": overwrite,
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── J. Derived Image Analysis Persistence Socket ─────────────────────────────

def build_derived_analysis_socket(
    doc_id: str,
    image_id: str,
    analysis_id: str,
    attempt_number: int,
    model: str,
    task_type: str,
    raw_output: str,
    derived_file_ref: str,
    indexed_in_chroma: bool,
    chroma_record_id: Optional[str] = None,
    ocr_text: Optional[str] = None,
    description: Optional[str] = None,
    observations: Optional[Dict[str, Any]] = None,
    became_current: bool = True,
    lifecycle_status: str = "accepted",
    previous_current_analysis_id: Optional[str] = None,
    instruction: Optional[str] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="derived_image_analysis_persistence",
        result={
            "doc_id": doc_id,
            "image_id": image_id,
            "analysis_id": analysis_id,
            "attempt_number": attempt_number,
            "became_current": became_current,
            "derived_file": derived_file_ref,
            "chroma_record_id": chroma_record_id if indexed_in_chroma else None,
        },
        identity={
            "doc_id": doc_id,
            "image_id": image_id,
            "analysis_id": analysis_id,
            "attempt_number": attempt_number,
            "model": model,
            "task_type": task_type,
        },
        context={
            "instruction": instruction,
            "status": lifecycle_status if not became_current else ("accepted" if lifecycle_status == "accepted" else lifecycle_status),
            "previous_current_analysis_id": previous_current_analysis_id,
        },
        storage={
            "derived_file_ref": derived_file_ref,
            "canonical_image_untouched": True,
        },
        execution={
            "indexed_in_chroma": indexed_in_chroma,
            "chroma_record_id": chroma_record_id if indexed_in_chroma else None,
            "collection": "sage_derived" if indexed_in_chroma else None,
        },
        raw={
            "raw_output": raw_output,
            "ocr_text": ocr_text,
            "description": description,
            "observations": to_json_serializable(observations or {}),
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── K. Reindex Document Socket ───────────────────────────────────────────────

def build_reindex_document_socket(
    doc_id: str,
    source_records_rebuilt: int,
    embedded_records: int,
    canonical_artifacts_discovered: Optional[int] = None,
    chunks_rebuilt: Optional[int] = None,
    derived_analyses_discovered: int = 0,
    derived_records_rebuilt: int = 0,
    skipped_records: int = 0,
    collection_name: str = "sage_source",
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="reindex_document",
        result={
            "doc_id": doc_id,
            "source_records_rebuilt": source_records_rebuilt,
            "embedded_records": embedded_records,
            "canonical_artifacts_discovered": canonical_artifacts_discovered if canonical_artifacts_discovered is not None else source_records_rebuilt,
            "chunks_rebuilt": chunks_rebuilt if chunks_rebuilt is not None else source_records_rebuilt,
            "derived_analyses_discovered": derived_analyses_discovered,
            "derived_records_rebuilt": derived_records_rebuilt,
            "skipped_records": skipped_records,
            "scope": "source_only",
            "collections_updated": [collection_name],
        },
        identity={
            "doc_id": doc_id,
        },
        execution={
            "rebuild_scope": "source_canonical_artifacts",
            "derived_reconstruction_supported": False,
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── L. Rebuild All Indexes Socket ────────────────────────────────────────────

def build_rebuild_indexes_socket(
    source_summary: Dict[str, Any],
    derived_summary: Dict[str, Any],
    total_documents: int,
    total_records: int,
    chroma_root: str,
    artifacts_root: str,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    has_errors = bool(source_summary.get("errors") or derived_summary.get("errors") or error)
    status = "partial" if has_errors and total_records > 0 else ("error" if has_errors else "success")

    return build_socket_envelope(
        status=status,
        operation="rebuild_all_indexes",
        result={
            "total_documents": total_documents,
            "total_records": total_records,
            "source": to_json_serializable(source_summary),
            "derived": to_json_serializable(derived_summary),
        },
        storage={
            "chroma_root": str(chroma_root).replace("\\", "/"),
            "artifacts_root": str(artifacts_root).replace("\\", "/"),
            "artifacts_preserved": True,
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── M. Safe Math Engine Socket ───────────────────────────────────────────────

def build_math_socket(
    expression: str,
    value: Optional[Union[int, float]] = None,
    operators_used: Optional[List[str]] = None,
    functions_used: Optional[List[str]] = None,
    constants_used: Optional[List[str]] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"
    value_type = type(value).__name__ if value is not None else None

    return build_socket_envelope(
        status=status,
        operation="calculate",
        result={
            "expression": expression,
            "value": value,
            "value_type": value_type,
        },
        execution={
            "engine": "safe_ast",
            "unrestricted_eval_used": False,
            "operators_used": operators_used or [],
            "functions_used": functions_used or [],
            "constants_used": constants_used or [],
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── N. Dispatcher Execution Socket ───────────────────────────────────────────

def build_dispatcher_socket(
    call_id: str,
    tool_name: str,
    function_name: str,
    status: str,
    tool_output: Any = None,
    sanitized_arguments: Optional[Dict[str, Any]] = None,
    is_mock: bool = False,
    duration_ms: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_socket_envelope(
        status=status,
        operation="tool_dispatch",
        result={
            "call_id": call_id,
            "tool_name": tool_name,
            "function_name": function_name,
            "status": status,
        },
        identity={
            "call_id": call_id,
            "tool_name": tool_name,
            "function_name": function_name,
        },
        context={
            "sanitized_arguments": to_json_serializable(sanitized_arguments or {}),
        },
        execution={
            "is_mock": is_mock,
            "registry_status": "found" if status in ("success", "error") else status,
        },
        references={
            "tool_output": to_json_serializable(tool_output),
        },
        timing=build_timing_payload(duration_ms=duration_ms),
        warnings=warnings,
        error=error,
    )


# ── O. Run State Socket ──────────────────────────────────────────────────────

def build_run_state_socket(
    run_id: str,
    request_id: str,
    loop_index: int,
    user_text: str,
    registered_documents: List[Any],
    tool_calls: List[Any],
    status: str = "active",
    current_model: Optional[str] = None,
    created_at: float = 0.0,
    updated_at: float = 0.0,
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    serialized_docs = [to_json_serializable(d) for d in registered_documents]
    serialized_calls = [to_json_serializable(c) for c in tool_calls]
    elapsed_ms = (updated_at - created_at) * 1000.0 if updated_at and created_at else 0.0

    return build_socket_envelope(
        status=status,
        operation="run_state_snapshot",
        result={
            "run_id": run_id,
            "request_id": request_id,
            "loop_index": loop_index,
            "document_count": len(serialized_docs),
            "tool_call_count": len(serialized_calls),
        },
        identity={
            "run_id": run_id,
            "request_id": request_id,
        },
        context={
            "user_text": user_text,
            "current_model": current_model,
            "loop_index": loop_index,
        },
        references={
            "registered_documents": serialized_docs,
            "tool_calls": serialized_calls,
        },
        timing=build_timing_payload(duration_ms=elapsed_ms),
        warnings=warnings,
        error=build_error_payload("RUN_ERRORS", "RunStateErrors", "; ".join(errors)) if errors else None,
    )


# ── P. Sandbox Execution Socket ──────────────────────────────────────────────

def build_sandbox_execution_socket(
    stdout: str,
    stderr: str,
    exit_code: int,
    wall_time_ms: float,
    memory_peak_mb: float,
    timed_out: bool,
    code_text: Optional[str] = None,
    container_image: str = "python:3.11-slim",
    memory_limit_mb: int = 512,
    cpu_cores: float = 1.0,
    timeout_seconds: float = 30.0,
    container_removed: bool = True,
    temp_cleaned: bool = True,
    execution_id: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if error:
        status = "infra_error"
    elif timed_out:
        status = "timeout"
    elif exit_code == 0:
        status = "success"
    else:
        status = "error"

    code_hash = hashlib.sha256(code_text.encode("utf-8")).hexdigest()[:16] if code_text else None

    return build_socket_envelope(
        status=status,
        operation="sandbox_execution",
        result={
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "succeeded": exit_code == 0 and not timed_out and error is None,
        },
        identity={
            "execution_id": execution_id or f"exec_{code_hash or 'unknown'}",
            "code_hash": code_hash,
        },
        execution={
            "container_image": container_image,
            "network": "none",
            "read_only_mount": True,
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges:true"],
            "pids_limit": 64,
            "memory_limit_mb": memory_limit_mb,
            "cpu_cores": cpu_cores,
            "timeout_seconds": timeout_seconds,
            "working_dir": "/sandbox/workspace",
            "exit_code": exit_code,
            "timed_out": timed_out,
        },
        storage={
            "container_removed": container_removed,
            "temp_cleaned": temp_cleaned,
            "host_filesystem_isolated": True,
        },
        timing=build_timing_payload(duration_ms=wall_time_ms),
        warnings=warnings,
        error=error,
    )


# ── Q. Code Extraction Socket ────────────────────────────────────────────────

def build_code_extraction_socket(
    code: str,
    language: str,
    had_fence: bool,
    block_count: int,
    raw_response: str,
    detected_blocks: Optional[List[Dict[str, Any]]] = None,
    selected_block_index: Optional[int] = None,
    extraction_strategy: Optional[str] = None,
    fallback_used: bool = False,
    duration_ms: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not code.strip():
        status = "empty"
        strategy = "empty"
    elif had_fence and language == "python":
        status = "success"
        strategy = "named_python_fence"
    elif had_fence:
        status = "success"
        strategy = "generic_fence"
    else:
        status = "fallback"
        strategy = "unfenced_fallback"

    return build_socket_envelope(
        status=status,
        operation="code_extraction",
        result={
            "code": code,
            "language": language,
            "had_fence": had_fence,
            "block_count": block_count,
            "strategy": extraction_strategy or strategy,
            "selected_block_index": selected_block_index,
            "extraction_strategy": extraction_strategy or strategy,
            "fallback_used": fallback_used,
            "detected_blocks": to_json_serializable(detected_blocks or []),
        },
        raw={
            "raw_response": raw_response,
            "raw_response_length": len(raw_response),
            "extracted_character_count": len(code),
            "extracted_line_count": len(code.splitlines()) if code else 0,
        },
        timing=build_timing_payload(duration_ms=duration_ms),
        warnings=warnings,
        error=error,
    )


# ── R. Code Repair Loop Socket ───────────────────────────────────────────────

def build_repair_loop_socket(
    status: str,
    final_code: str,
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    wall_time_ms: float,
    memory_peak_mb: float,
    timed_out: bool,
    attempts: int,
    task: str,
    fix_history: Optional[List[Any]] = None,
    infra_error: Optional[str] = None,
    extraction_had_fence: bool = False,
    extraction_language: str = "unknown",
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_socket_envelope(
        status=status,
        operation="code_repair_pipeline",
        result={
            "status": status,
            "succeeded": status == "success",
            "attempts": attempts,
            "final_code": final_code,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "memory_peak_mb": memory_peak_mb,
        },
        context={
            "task": task,
            "attempts_used": attempts,
            "extraction_had_fence": extraction_had_fence,
            "extraction_language": extraction_language,
        },
        references={
            "fix_history": to_json_serializable(fix_history or []),
        },
        timing=timing or build_timing_payload(duration_ms=wall_time_ms),
        warnings=warnings,
        error=error or (build_error_payload("INFRA_ERROR", "DockerInfraError", infra_error) if infra_error else None),
    )


# ── S. Model Manager Lifecycle Socket ────────────────────────────────────────

def build_model_manager_socket(
    action: str,
    model_key: str,
    model_name: str,
    status: str = "success",
    previous_model: Optional[str] = None,
    current_model: Optional[str] = None,
    switch_count: int = 0,
    port: int = 8080,
    host: str = "127.0.0.1",
    pid: Optional[int] = None,
    model_path: Optional[str] = None,
    mmproj_path: Optional[str] = None,
    context_size: int = 8192,
    gpu_layers: int = 999,
    log_file: Optional[str] = None,
    is_healthy: bool = True,
    duration_ms: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return build_socket_envelope(
        status=status,
        operation="model_lifecycle",
        result={
            "action": action,
            "model_key": model_key,
            "model_name": model_name,
            "is_healthy": is_healthy,
        },
        identity={
            "requested_model_key": model_key,
            "model_name": model_name,
        },
        context={
            "previous_model": previous_model,
            "current_model": current_model,
            "switch_count": switch_count,
        },
        execution={
            "port": port,
            "host": host,
            "context_size": context_size,
            "gpu_layers": gpu_layers,
            "pid": pid,
            "model_path": str(model_path).replace("\\", "/") if model_path else None,
            "mmproj_path": str(mmproj_path).replace("\\", "/") if mmproj_path else None,
            "log_file": str(log_file).replace("\\", "/") if log_file else None,
            "visibility_mapping": {
                "pid": Visibility.INTERNAL_ONLY,
                "model_path": Visibility.INTERNAL_ONLY,
                "mmproj_path": Visibility.INTERNAL_ONLY,
                "log_file": Visibility.INTERNAL_ONLY,
            },
        },
        timing=build_timing_payload(duration_ms=duration_ms),
        warnings=warnings,
        error=error,
    )


# ── T. Model Client HTTP / Transport Telemetry Socket ────────────────────────

def build_model_transport_socket(
    endpoint: str,
    url: str,
    duration_seconds: float,
    content: str,
    usage: Optional[Dict[str, Any]] = None,
    timings: Optional[Dict[str, Any]] = None,
    raw_response: Optional[Dict[str, Any]] = None,
    model_key: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
    http_status: int = 200,
    request_id: Optional[str] = None,
    finish_reason: Optional[str] = "stop",
    request_timestamp: Optional[str] = None,
    response_timestamp: Optional[str] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error or http_status >= 400 else "success"
    usage_dict = usage or {}
    timings_dict = timings or {}
    req_time = request_timestamp or iso_now()
    resp_time = response_timestamp or iso_now()

    return build_socket_envelope(
        status=status,
        operation="model_http_transport",
        result={
            "content": content,  # Raw LLM string — semantic parsing left to downstream
            "tokens": {
                "prompt_tokens": max(0, int(usage_dict.get("prompt_tokens", 0))),
                "completion_tokens": max(0, int(usage_dict.get("completion_tokens", 0))),
                "total_tokens": max(0, int(usage_dict.get("total_tokens", 0))),
            },
            "finish_reason": finish_reason,
        },
        context={
            "request_id": request_id,
            "endpoint": endpoint,
            "model_key": model_key,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        },
        execution={
            "http_status": http_status,
            "url": url,
            "request_timestamp": req_time,
            "response_timestamp": resp_time,
            "telemetry": {
                "prompt_ms": timings_dict.get("prompt_ms"),
                "completion_ms": timings_dict.get("predicted_ms"),
                "prompt_per_second": timings_dict.get("prompt_per_second"),
                "completion_per_second": timings_dict.get("predicted_per_second"),
            },
        },
        raw=raw_response,
        timing=build_timing_payload(started_at=req_time, finished_at=resp_time, duration_ms=duration_seconds * 1000.0),
        warnings=warnings,
        error=error,
    )


# ── U. Upload / Backend Ingestion Socket ──────────────────────────────────────

def build_upload_socket(
    request_id: str,
    original_filename: str,
    safe_filename: str,
    file_ref: str,
    file_type: str,
    byte_size: int,
    temp_path: str,
    doc_id: str,
    ingestion_socket: Optional[Dict[str, Any]] = None,
    timing: Optional[Dict[str, Any]] = None,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    status = "error" if error else "success"

    return build_socket_envelope(
        status=status,
        operation="backend_file_upload",
        result={
            "ref": file_ref,
            "doc_id": doc_id,
            "name": safe_filename,
            "type": file_type,
            "size": byte_size,
        },
        identity={
            "request_id": request_id,
            "doc_id": doc_id,
            "original_filename": original_filename,
            "safe_filename": safe_filename,
        },
        storage={
            "temp_path": str(temp_path).replace("\\", "/"),
            "file_ref": file_ref,
            "visibility_mapping": {
                "temp_path": Visibility.INTERNAL_ONLY,
            },
        },
        references={
            "ingestion_socket": to_json_serializable(ingestion_socket),
        },
        timing=timing,
        warnings=warnings,
        error=error,
    )


# ── V. JSON Parse / Repair Socket ────────────────────────────────────────────

def build_json_repair_socket(
    raw_input: str,
    parsed_json: Optional[Any],
    parsed_type: str,
    is_valid: bool,
    thought_tags_stripped: bool = False,
    markdown_fence_extracted: bool = False,
    outermost_braces_extracted: bool = False,
    repaired_string: Optional[str] = None,
    duration_ms: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if is_valid and not (thought_tags_stripped or markdown_fence_extracted or outermost_braces_extracted):
        status = "success"
    elif is_valid:
        status = "repaired"
    else:
        status = "error"

    return build_socket_envelope(
        status=status,
        operation="json_parse_repair",
        result={
            "is_valid": is_valid,
            "parsed_type": parsed_type,
            "parsed_json": to_json_serializable(parsed_json),
        },
        context={
            "repair_applied": thought_tags_stripped or markdown_fence_extracted or outermost_braces_extracted,
            "steps": {
                "thought_tags_stripped": thought_tags_stripped,
                "markdown_fence_extracted": markdown_fence_extracted,
                "outermost_braces_extracted": outermost_braces_extracted,
            },
        },
        raw={
            "raw_input": raw_input,
            "raw_length": len(raw_input),
            "repaired_string": repaired_string,
        },
        timing=build_timing_payload(duration_ms=duration_ms),
        warnings=warnings,
        error=error,
    )


# ── W. Derived Analysis Cache Lookup / Write Socket ───────────────────────────
# (Formerly generic cache; accurately scoped to derived image-analysis cache)

def build_derived_analysis_cache_socket(
    operation: str,  # "derived_analysis_cache_lookup" or "derived_analysis_cache_write"
    doc_id: str,
    image_id: str,
    cache_file_ref: str,
    hit: bool,
    cache_path: Optional[str] = None,
    analyses_count: int = 0,
    current_keys: Optional[List[str]] = None,
    byte_size: Optional[int] = None,
    duration_ms: float = 0.0,
    warnings: Optional[List[str]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    op_name = operation
    if op_name in ("cache_lookup", "derived_analysis_cache_lookup"):
        op_name = "derived_analysis_cache_lookup"
    elif op_name in ("cache_write", "derived_analysis_cache_write"):
        op_name = "derived_analysis_cache_write"

    status = "error" if error else ("hit" if hit else "miss" if "lookup" in op_name else "write_success")

    return build_socket_envelope(
        status=status,
        operation=op_name,
        result={
            "hit": hit,
            "analyses_count": analyses_count,
            "current_keys": current_keys or [],
        },
        identity={
            "doc_id": doc_id,
            "image_id": image_id,
        },
        storage={
            "cache_file_ref": cache_file_ref,
            "cache_path": str(cache_path).replace("\\", "/") if cache_path else None,
            "byte_size": byte_size,
            "visibility_mapping": {
                "cache_path": Visibility.INTERNAL_ONLY,
            },
        },
        timing=build_timing_payload(duration_ms=duration_ms),
        warnings=warnings,
        error=error,
    )


# Backward-compatible alias
build_cache_socket = build_derived_analysis_cache_socket
