"""
SAGE/core/mappers/gemma_results.py
Rich internal socket → Gemma-facing tool result projections.

Every mapper here is a PURE PROJECTION of already-computed data.
No tools are invoked. No models are called. No state is mutated.

Gemma-facing outer shell (per call):
    Success:
        {"call_index": i, "tool": "...", "function": "...",
         "status": "success", "result": {...}}
    Partial (usable results + warnings):
        {"call_index": i, "tool": "...", "function": "...",
         "status": "partial", "result": {...}, "warning": "..."}
    Error:
        {"call_index": i, "tool": "...", "function": "...",
         "status": "error",
         "error": {"code": "...", "message": "...", "retryable": bool}}

Strict strip rules (never included in Gemma-facing output):
    - local filesystem paths (local_path, temp paths, cache paths)
    - tracebacks, exception details, source_file, source_function
    - PIDs, ports, process identifiers
    - Docker/container IDs and security telemetry
    - raw Chroma payloads and collection internals
    - embedding vectors and embedding model telemetry
    - stage timings, throughput metrics
    - repair prompts and intermediate code versions
    - storage hashes/checksums
    - internal socket field names (sage.rag_search, sage.chroma_query, etc.)
"""

from __future__ import annotations

import json
from typing import Any

# ── Internal helpers ──────────────────────────────────────────────────────────

# Fields that are internal-only and must never reach Gemma.
_STRIP_ALWAYS = frozenset({
    "local_path", "temp_path", "cache_path", "derived_file",
    "traceback", "exc_info", "source_file", "source_function", "stack_trace",
    "pid", "port", "process_id", "container_id", "docker_id",
    "raw_chroma_responses", "chroma_ids", "embedding", "embeddings",
    "embedding_model", "embedding_device", "distance_metric", "similarity_formula",
    "duration_ms", "stage_timings", "throughput", "token_rate",
    "repair_prompt", "fix_history", "fix_attempts",
    "record_id", "raw_distance",           # keep derived_cosine_similarity as cosine_similarity
    "derived_similarity",                  # legacy clamped compat — prefer unclipped
    "limit_hit", "collections_queried", "include_source", "include_derived",
    "max_results_hit", "searched_canonical_subdirs", "derived_excluded",
    "total_in_document",                   # internal accounting
    "wall_time_ms", "memory_peak_mb",
    "infra_error",
    "extraction_had_fence", "extraction_language",
})


def strip_internal_fields(obj: Any) -> Any:
    """Recursively remove all internal/sensitive keys from an arbitrary dict/list."""
    if isinstance(obj, dict):
        return {
            k: strip_internal_fields(v)
            for k, v in obj.items()
            if k not in _STRIP_ALWAYS and not any(
                bad in k.lower()
                for bad in (
                    "traceback", "local_path", "temp_path", "cache_path",
                    "docker_id", "container_id", "raw_chroma",
                )
            )
        }
    if isinstance(obj, (list, tuple)):
        return [strip_internal_fields(v) for v in obj]
    return _safe_json(obj)


def _safe_json(obj: Any) -> Any:
    """Recursively ensure an object is JSON-serialisable."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_json(v) for v in obj]
    return str(obj)


# ── Common error mapper ───────────────────────────────────────────────────────

def map_error_for_gemma(err: Any) -> dict:
    """Project a rich error dict/object down to the Gemma-facing error shell.

    Gemma-facing shape:
        {"code": "...", "message": "...", "retryable": bool}

    Strips: tracebacks, exception objects, source lines, internal codes.
    """
    if isinstance(err, dict):
        code = err.get("code") or err.get("type") or "TOOL_ERROR"
        message = err.get("message") or err.get("error") or str(err)
        retryable = bool(err.get("retryable", False))
    elif isinstance(err, str):
        code = "TOOL_ERROR"
        message = err
        retryable = False
    else:
        code = "TOOL_ERROR"
        message = str(err)
        retryable = False

    return {"code": str(code), "message": str(message), "retryable": retryable}


# ── Outer envelope builder ────────────────────────────────────────────────────

def map_tool_result_for_gemma(
    tool: str,
    function: str,
    rich_socket: Any,
    call_index: int = 0,
    *,
    mapper_fn=None,
    **mapper_kwargs,
) -> dict:
    """Wrap a rich internal socket in the uniform Gemma-facing outer envelope.

    Outer status is one of: "success" | "partial" | "error".
    "partial" means usable results exist alongside a non-fatal warning/error.

    Fail-closed: if mapper_fn is None or not callable, returns UNMAPPED_TOOL_RESULT.

    Args:
        tool:         Canonical tool name (e.g. "document_database").
        function:     Canonical function name (e.g. "rag_search").
        rich_socket:  The raw dict/object returned from the internal producer.
        call_index:   0-based index of this call in the current batch.
        mapper_fn:    One of the map_*_result functions below.
        **mapper_kwargs: Extra keyword args forwarded to mapper_fn.
    """
    if mapper_fn is None or not callable(mapper_fn):
        return {
            "call_index": call_index,
            "tool": tool,
            "function": function,
            "status": "error",
            "error": {
                "code": "UNMAPPED_TOOL_RESULT",
                "message": "No Gemma-facing mapper exists for this tool result.",
                "retryable": False,
            },
        }

    data = rich_socket if isinstance(rich_socket, dict) else {}
    internal_status = data.get("status", "success")

    try:
        mapped = mapper_fn(data, **mapper_kwargs)
    except Exception as exc:  # never let a mapper bug crash the orchestrator
        return {
            "call_index": call_index,
            "tool": tool,
            "function": function,
            "status": "error",
            "error": {
                "code": "MAPPER_ERROR",
                "message": f"Internal mapper error: {exc}",
                "retryable": False,
            },
        }

    # Determine outer status --------------------------------------------------
    # A raw socket may report "partial_error" or carry a non-null error field.
    has_error_field = bool(data.get("error"))
    is_partial = (
        internal_status in ("partial", "partial_error")
        or (internal_status == "success" and has_error_field)
    )
    is_error = internal_status == "error"

    if is_error:
        err_payload = data.get("error") or data.get("error_message") or "Tool failed"
        return {
            "call_index": call_index,
            "tool": tool,
            "function": function,
            "status": "error",
            "error": map_error_for_gemma(err_payload),
        }
    if is_partial:
        warning_msg = (
            data.get("warning")
            or (str(data.get("error")) if data.get("error") else None)
            or "Partial results returned."
        )
        return {
            "call_index": call_index,
            "tool": tool,
            "function": function,
            "status": "partial",
            "result": strip_internal_fields(mapped),
            "warning": warning_msg,
        }
    return {
        "call_index": call_index,
        "tool": tool,
        "function": function,
        "status": "success",
        "result": strip_internal_fields(mapped),
    }


def build_tool_results_packet(mapped_results: list) -> dict:
    """Wrap all mapped call results into the tool_results packet fed back to Gemma.

    Shape:
        {
          "type": "tool_results",
          "results": [<mapped_result_0>, <mapped_result_1>, ...]
        }

    Order is always preserved (same as original Gemma call_index order).
    """
    return {"type": "tool_results", "results": _safe_json(mapped_results)}


# ── Per-function mappers ──────────────────────────────────────────────────────

def map_rag_result(data: dict) -> dict:
    """Project rag_search rich socket → Gemma-facing matches array.

    All ranked matches are forwarded (no top-1 truncation, no reranker).
    Unclipped cosine similarity (derived_cosine_similarity) is preferred
    and exposed as "cosine_similarity".

    Gemma-facing shape:
        {
          "matches": [
            {
              "text": "...",
              "doc_id": "...",
              "element_id": "...",    # from source_element_ids[0] if set
              "element_type": "...",
              "page": 7 | null,
              "slide": null,
              "sheet": null,
              "raw_distance": 0.21,
              "cosine_similarity": 0.79,
              "image_refs": ["img_12"],
              "source": "canonical" | "derived"
            },
            ...
          ],
          "returned": 5,
          "top_k": 5
        }

    Stripped: embedding vectors, Chroma internals, local paths, timings, hashes,
              record_id, raw Chroma distances duplicated by derived fields,
              limit_hit, collections_queried, derived_similarity (legacy clamped).
    """
    result_block = data.get("result") or {}
    context_block = data.get("context") or {}
    records = result_block.get("records") or data.get("results") or []

    matches = []
    for r in records:
        # Prefer unclipped cosine similarity; fall back to clamped for compat.
        cosine_sim = r.get("derived_cosine_similarity")
        if cosine_sim is None:
            cosine_sim = r.get("derived_similarity")
        if cosine_sim is None and r.get("distance") is not None:
            try:
                cosine_sim = max(0.0, 1.0 - float(r.get("distance")))
            except (ValueError, TypeError):
                cosine_sim = None

        # Source label: "canonical" for source records, "derived" for vision/OCR.
        origin = r.get("origin", "source")
        source_label = "canonical" if origin == "source" else "derived"

        # Element ID: RAG records reference chunks; preserve first source_element_id.
        element_ids = r.get("source_element_ids") or []
        element_id = element_ids[0] if element_ids else None

        match: dict[str, Any] = {
            "text": r.get("text", ""),
            "doc_id": r.get("doc_id", ""),
            "element_type": r.get("record_type", "text"),
            "raw_distance": r.get("distance") if r.get("distance") is not None else r.get("raw_distance"),
            "cosine_similarity": cosine_sim,
            "source": source_label,
        }
        if element_id:
            match["element_id"] = element_id

        # Positional metadata — only when actually available
        for pos_key in ("page", "slide", "sheet"):
            if r.get(pos_key) is not None:
                match[pos_key] = r[pos_key]

        # Artifact references Gemma may use in follow-up calls
        for ref_key in ("image_refs", "table_refs", "code_refs"):
            refs = r.get(ref_key)
            if refs:
                match[ref_key] = refs

        matches.append(match)

    return {
        "matches": matches,
        "returned": len(matches),
        "top_k": context_block.get("requested_top_k") or data.get("count") or len(matches),
    }


def map_exact_result(data: dict) -> dict:
    """Project exact_search rich socket → Gemma-facing matches array.

    Gemma-facing shape:
        {
          "matches": [
            {
              "doc_id": "...",
              "element_id": "...",
              "element_type": "...",
              "matched_text": "...",
              "context": "surrounding snippet",
              "page": 4 | null,
              "slide": null,
              "sheet": null,
              "start_offset": 132,
              "end_offset": 139,
              "table_row": null,
              "table_col": null
            }
          ]
        }

    Stripped: local paths, manifest internals, scan timings, debug tracebacks,
              order, ref (internal), max_results_hit, searched_canonical_subdirs.
    """
    result_block = data.get("result") or {}
    raw_matches = result_block.get("matches") or []

    matches = []
    for m in raw_matches:
        match: dict[str, Any] = {
            "doc_id": m.get("doc_id", ""),
            "element_id": m.get("element_id", ""),
            "element_type": m.get("element_type", "text"),
            "matched_text": m.get("matched_text", ""),
            "context": m.get("snippet", ""),
            "start_offset": m.get("match_start"),
            "end_offset": m.get("match_end"),
        }
        # Positional metadata — only when present
        for pos_key in ("page", "slide", "sheet"):
            if m.get(pos_key) is not None:
                match[pos_key] = m[pos_key]

        # Table coordinates — only when relevant
        if m.get("table_row") is not None:
            match["table_row"] = m["table_row"]
        if m.get("table_col") is not None:
            match["table_col"] = m["table_col"]

        matches.append(match)

    return {"matches": matches}


def map_artifact_fetch_result(data: dict, doc_id: str = "", element_id: str = "") -> dict:
    """Project artifact_fetch rich socket → Gemma-facing artifact object.

    Returns a single {"artifact": {...}} object typed by element kind.

    Text:
        {"artifact": {"doc_id": "...", "element_id": "...", "type": "text",
                      "text": "...", "page": 7}}
    Table:
        {"artifact": {"doc_id": "...", "element_id": "...", "type": "table",
                      "rows": [...], "page": 8}}
    Code:
        {"artifact": {"doc_id": "...", "element_id": "...", "type": "code",
                      "code": "...", "language": "python", "page": 5}}
    Image:
        {"artifact": {"doc_id": "...", "element_id": "...", "type": "image",
                      "image_id": "...", "image_ref": "...", "page": 6,
                      "caption": "..."}}

    Stripped: local_path, hashes, file timestamps, raw manifest structure,
              debug telemetry, bbox, exists flag, dimensions, subtype, provenance,
              links, previous_element, next_element, artifact_ref.
    """
    # The tool returns {"status": ..., "element": {...}}
    element = data.get("element") or {}
    if not element and data.get("result"):
        element = data["result"]

    elem_type = element.get("type", "text")
    elem_id = element.get("id") or element.get("element_id") or element_id
    # Use caller-provided doc_id when not embedded in element
    d_id = element.get("doc_id") or doc_id

    base: dict[str, Any] = {
        "doc_id": d_id,
        "element_id": elem_id,
        "type": elem_type,
    }

    # Optional positional metadata
    for pos_key in ("page", "slide", "sheet"):
        if element.get(pos_key) is not None:
            base[pos_key] = element[pos_key]

    if elem_type == "text":
        base["text"] = element.get("text", "")

    elif elem_type == "table":
        base["rows"] = element.get("rows") or []
        if element.get("headers"):
            base["headers"] = element["headers"]
        if element.get("markdown"):
            base["markdown"] = element["markdown"]

    elif elem_type == "code":
        base["code"] = element.get("code") or element.get("text", "")
        lang = element.get("language") or element.get("lang")
        if lang:
            base["language"] = lang

    elif elem_type == "image":
        # image_id as stable reference Gemma can reuse in vision calls
        base["image_id"] = elem_id
        # ref is the stable registered image reference (not a local path)
        ref = element.get("ref") or element.get("artifact_ref")
        if ref:
            base["image_ref"] = ref
        caption = element.get("caption")
        if caption:
            base["caption"] = caption
        derived_avail = element.get("derived_available")
        if derived_avail is not None:
            base["derived_available"] = derived_avail
        derived_analysis = element.get("derived_analysis")
        if derived_analysis:
            base["analysis"] = derived_analysis
        # NOTE: local_path is intentionally NOT included here.

    return {"artifact": base}


def map_list_artifacts_result(data: dict) -> dict:
    """Project list_artifacts rich socket → lightweight Gemma-facing inventory.

    Gemma-facing shape:
        {
          "artifacts": [
            {
              "element_id": "img_12",
              "type": "image",
              "order": 14,
              "page": 6,
              "caption": "optional",
              "heading": "optional",
              "derived_available": true
            }
          ]
        }

    Stripped: bbox, byte sizes, hashes, manifest paths, exists flag,
              storage implementation details, link_count, subtype, ref,
              slide/sheet if absent (only include when present).
    """
    result_block = data.get("result") or {}
    raw_artifacts = result_block.get("artifacts") or data.get("artifacts") or []
    default_doc_id = result_block.get("doc_id") or data.get("doc_id")

    artifacts = []
    for a in raw_artifacts:
        doc_id_val = a.get("doc_id") or default_doc_id
        entry: dict[str, Any] = {
            "element_id": a.get("element_id", ""),
            "type": a.get("type", ""),
            "order": a.get("order", 0),
        }
        if doc_id_val:
            entry["doc_id"] = doc_id_val
        # Positional metadata — only when present
        for pos_key in ("page", "slide", "sheet"):
            if a.get(pos_key) is not None:
                entry[pos_key] = a[pos_key]

        if a.get("caption"):
            entry["caption"] = a["caption"]
        if a.get("heading"):
            entry["heading"] = a["heading"]
        if a.get("derived_available") is not None:
            entry["derived_available"] = a["derived_available"]

        artifacts.append(entry)

    return {"artifacts": artifacts}


def map_vision_result(data: dict, doc_id: str = "", image_id: str = "") -> dict:
    """Project vision/OCR rich socket → Gemma-facing textual analysis.

    Gemma-facing shape:
        {
          "doc_id": "doc_x",
          "image_id": "img_12",
          "analysis": "Qwen-VL OCR/visual answer"
        }

    Optional (only when already available and useful for references):
        "derived_analysis_id": "analysis_x"

    Stripped: local image paths, cache paths, raw llama.cpp payloads,
              HTTP telemetry, token timings, PID/port, full persistence metadata,
              transport diagnostics, model name.
    """
    # The vision tool returns {"status": ..., "results": [{"image_id": ...,
    # "raw_output": ..., "model": ...}, ...]} for multi-image calls,
    # or single-image adapters may return {"image_id": ..., "analysis": ..., ...}
    results_list = data.get("results") or []
    if results_list:
        # Multi-image: find matching image_id or use first result
        target = next((r for r in results_list if r.get("image_id") == image_id), results_list[0])
    else:
        target = data

    analysis = (
        target.get("analysis")
        or target.get("raw_output")
        or target.get("result")
        or ""
    )

    img_id = target.get("image_id") or image_id
    d_id = target.get("doc_id") or doc_id

    mapped: dict[str, Any] = {
        "doc_id": d_id,
        "image_id": img_id,
        "analysis": str(analysis),
    }

    # Derived analysis ID only if genuinely useful for later references
    analysis_id = target.get("derived_analysis_id") or target.get("analysis_id")
    if analysis_id:
        mapped["derived_analysis_id"] = analysis_id

    return mapped


def map_coder_result(data: dict) -> dict:
    """Project coder rich socket → Gemma-facing execution summary.

    Execution states:
        "success"      — code ran, exited 0; stdout and final code included.
        "error"        — code exited non-zero after all fix attempts.
        "not_executed" — no sandbox run occurred (language unsupported,
                         no pipeline available, or extract_failed).

    Language is preserved from the actual extracted/requested language;
    never hardcoded to "python".

    Gemma-facing shapes:
        Success:
            {"code": "...", "language": "python", "execution_status": "success",
             "stdout": "...", "stderr": "", "exit_code": 0, "attempts_used": 1}
        Error:
            {"code": "...", "language": "python", "execution_status": "error",
             "stderr": "...", "exit_code": 1, "attempts_used": 3}
        Not executed:
            {"code": "...", "language": "python",
             "execution_status": "not_executed"}

    Stripped: Docker/container IDs, repair prompts, fix_history,
              intermediate code versions, sandbox security telemetry,
              wall_time_ms, memory_peak_mb, infra_error details, tracebacks.
    """
    # Detect not-executed cases: pipeline was not available or language unsupported
    execution_status_raw = data.get("execution_status")
    if execution_status_raw == "not_executed":
        code = data.get("code") or data.get("final_code") or ""
        lang = data.get("language") or data.get("extraction_language") or "unknown"
        return {
            "code": code,
            "language": lang,
            "execution_status": "not_executed",
        }

    # Check extract_failed / infra_error statuses from pipeline
    pipe_status = data.get("status", "")
    if pipe_status in ("extract_failed", "infra_error") or data.get("infra_error"):
        code = data.get("final_code") or data.get("code") or ""
        lang = data.get("extraction_language") or data.get("language") or "unknown"
        return {
            "code": code,
            "language": lang,
            "execution_status": "not_executed",
        }

    # Determine actual language (from extraction or caller-provided)
    lang = (
        data.get("language")
        or data.get("extraction_language")
        or "unknown"
    )

    final_code = data.get("final_code") or data.get("code") or ""

    # If the language is unsupported by the sandbox, no process was executed.
    # Return execution_status="not_executed" with no exit_code/stdout/stderr.
    from .coder_input import is_sandbox_supported
    if not is_sandbox_supported(lang):
        return {
            "code": final_code,
            "language": lang,
            "execution_status": "not_executed",
        }

    succeeded = data.get("succeeded", False) or pipe_status == "success"
    exit_code = data.get("exit_code")
    stdout = (data.get("stdout") or "")[:2000]
    stderr = (data.get("stderr") or "")[:1000]
    attempts = data.get("attempts") or data.get("attempts_used") or 1

    if succeeded:
        mapped: dict[str, Any] = {
            "code": final_code,
            "language": lang,
            "execution_status": "success",
            "stdout": stdout,
            "exit_code": exit_code if exit_code is not None else 0,
            "attempts_used": attempts,
        }
        if stderr:
            mapped["stderr"] = stderr
        return mapped

    # Error / timeout
    mapped = {
        "code": final_code,
        "language": lang,
        "execution_status": "error",
        "stderr": stderr,
        "exit_code": exit_code if exit_code is not None else -1,
        "attempts_used": attempts,
    }
    if stdout:
        mapped["stdout"] = stdout
    return mapped


def map_math_result(data: dict) -> dict:
    """Project math/calculate rich socket → tiny Gemma-facing result.

    Gemma-facing shape:
        {"expression": "17*23", "value": 391}

    Stripped: AST provenance, operator lists, function tables, evaluator
              internals, timing, operators_used, functions_used,
              constants_used.
    """
    expression = data.get("expression", "")
    # tool_calculate returns {"result": <value>}; socket version uses "value"
    value = data.get("result") if "result" in data else data.get("value")
    return {"expression": expression, "value": value}


def map_general_knowledge_result(data: dict) -> dict:
    """Project general_knowledge/answer_query socket → Gemma-facing clean answer.

    Gemma-facing shape:
        {"answer": "..."}
    """
    answer = data.get("answer") or data.get("content") or data.get("result") or ""
    return {"answer": str(answer)}


# ── Canonical Function Mappers Registry ───────────────────────────────────────

CANONICAL_GEMMA_MAPPERS: dict[tuple[str, str], Any] = {
    ("document_database", "rag_search"): map_rag_result,
    ("document_database", "exact_search"): map_exact_result,
    ("document_database", "artifact_fetch"): map_artifact_fetch_result,
    ("document_database", "list_artifacts"): map_list_artifacts_result,
    ("vision_ocr", "analyze_image"): map_vision_result,
    ("code_specialist", "solve_code_task"): map_coder_result,
    ("math", "calculate"): map_math_result,
    ("general_knowledge", "answer_query"): map_general_knowledge_result,
}
