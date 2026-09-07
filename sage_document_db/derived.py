"""
sage_document_db/derived.py
Derived image/OCR knowledge insertion path.

Implements add_image_analysis() — the future hook for Qwen3-VL output.
Qwen3-VL is NOT called here; this module only stores and indexes results.

Rules:
- Validate target image exists in manifest before saving.
- Preserve raw_output verbatim — never rewrite it.
- Append to analyses[] history on repeat calls; never overwrite prior entries.
- Update/merge current{} fields when new structured fields are supplied.
- Store derived artifacts under artifacts/doc_x/derived/vision/img_x.json
- Upsert searchable text into sage_derived Chroma collection.
- derived/ is created on first use only.
"""

from __future__ import annotations
import json
from pathlib import Path

from .config import ARTIFACTS_ROOT
from .models import DerivedInsertResult
from .utils import iso_now, safe_mkdir


def add_image_analysis(
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
    artifacts_root: str | Path = ARTIFACTS_ROOT,
    chroma_store=None,          # ChromaStore instance (injected)
) -> DerivedInsertResult:
    """Store image analysis output and optionally index it.

    Save-vs-index separation:
        - Raw analysis attempt is ALWAYS preserved in the JSON cache.
        - Chroma indexing only happens when index_in_chroma=True.
        - Use index_in_chroma=False for pending/non-accepted attempts
          or mock-mode results that should not become searchable.

    Steps:
    1. Validate document and image_id exist in manifest.
    2. Load or create derived/vision/img_<id>.json.
    3. Append new analysis to analyses[].
    4. Merge current{} fields (never delete prior ones).
    5. Compose searchable text (only if indexing).
    6. Upsert into sage_derived via chroma_store (only if indexing).
    """
    root = Path(artifacts_root)
    doc_dir = root / doc_id

    # --- Step 1: Validate ---
    manifest_path = doc_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Document '{doc_id}' not found in artifact store."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    index = manifest.get("element_index", {})
    if image_id not in index:
        raise KeyError(
            f"Image '{image_id}' not found in manifest for doc_id='{doc_id}'."
        )

    entry = index[image_id]
    if entry.get("type") != "image":
        raise ValueError(
            f"Element '{image_id}' is not an image element "
            f"(type='{entry.get('type')}')."
        )

    image_ref = entry.get("ref")  # May be None if image extraction failed

    # --- Step 2: Load or create derived cache file ---
    derived_dir = doc_dir / "derived" / "vision"
    safe_mkdir(derived_dir)
    cache_file = derived_dir / f"{image_id}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
    else:
        cache = {
            "schema_version": "1.0",
            "doc_id": doc_id,
            "image_id": image_id,
            "image_ref": image_ref,
            "analyses": [],
            "current": {},
        }

    # --- Step 3: Append new analysis ---
    analyses = cache.get("analyses", [])
    analysis_id = f"analysis_{len(analyses) + 1:06d}"
    new_analysis = {
        "analysis_id": analysis_id,
        "created_at": iso_now(),
        "model": model,
        "task_type": task_type,
        "instruction": instruction,
        "raw_output": raw_output,   # Stored verbatim — never rewritten
    }
    analyses.append(new_analysis)
    cache["analyses"] = analyses

    # --- Step 4: Merge current{} fields ---
    current = cache.get("current", {})
    if ocr_text is not None:
        current["ocr_text"] = ocr_text
    if description is not None:
        current["description"] = description
    if observations is not None:
        # Merge into existing observations dict
        existing_obs = current.get("observations", {})
        existing_obs.update(observations)
        current["observations"] = existing_obs
    cache["current"] = current

    # --- Save cache file ---
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    # --- Step 5 & 6: Compose searchable text + upsert (only if indexing) ---
    rel_cache_ref = f"derived/vision/{image_id}.json"
    record_id = None

    if index_in_chroma and chroma_store is not None:
        record_id = f"{doc_id}:derived:image:{image_id}"
        if searchable_text and searchable_text.strip():
            derived_text = searchable_text
        else:
            from .chroma_store import _build_derived_searchable_text
            derived_text = _build_derived_searchable_text(image_id, current, analyses)
            if not derived_text.strip():
                # Fallback to raw output so it's still searchable
                derived_text = f"Image {image_id}\n\nLatest raw analysis:\n{raw_output}"

        metadata = {
            "doc_id": doc_id,
            "record_type": "derived_image",
            "image_id": image_id,
            "image_ref": image_ref or "",
            "derived_cache_ref": rel_cache_ref,
            "source_model": model,
        }
        chroma_store.upsert_derived_record(record_id, derived_text, metadata)

    return DerivedInsertResult(
        doc_id=doc_id,
        image_id=image_id,
        chroma_record_id=record_id,
        analysis_id=analysis_id,
        derived_file=rel_cache_ref,
    )


# ---------------------------------------------------------------------------
# Rich Output Sockets (Phases J & W)
# ---------------------------------------------------------------------------

def add_image_analysis_socket(
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
    artifacts_root: str | Path = ARTIFACTS_ROOT,
    chroma_store=None,
) -> dict:
    """Store image analysis output and return complete Component J socket."""
    import time
    from core.sockets import build_derived_analysis_socket, build_timing_payload, build_error_payload

    t0 = time.perf_counter()
    try:
        res = add_image_analysis(
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
            artifacts_root=artifacts_root,
            chroma_store=chroma_store,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_payload(duration_ms=total_ms)

        cache_file = Path(artifacts_root) / doc_id / "derived" / "vision" / f"{image_id}.json"
        attempt_number = 1
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    c = json.load(f)
                attempt_number = len(c.get("analyses", []))
            except Exception:
                pass

        return build_derived_analysis_socket(
            doc_id=doc_id,
            image_id=image_id,
            analysis_id=res.analysis_id,
            attempt_number=attempt_number,
            model=model,
            task_type=task_type,
            raw_output=raw_output,
            derived_file_ref=res.derived_file,
            indexed_in_chroma=index_in_chroma,
            chroma_record_id=res.chroma_record_id,
            ocr_text=ocr_text,
            description=description,
            observations=observations,
            instruction=instruction,
            timing=timing,
        )
    except Exception as e:
        total_ms = (time.perf_counter() - t0) * 1000.0
        timing = build_timing_payload(duration_ms=total_ms)
        error_payload = build_error_payload(
            code="DERIVED_ANALYSIS_ERROR",
            type_name=type(e).__name__,
            message=str(e),
            exc=e,
        )
        return build_derived_analysis_socket(
            doc_id=doc_id,
            image_id=image_id,
            analysis_id="unknown",
            attempt_number=0,
            model=model,
            task_type=task_type,
            raw_output=raw_output,
            derived_file_ref=f"derived/vision/{image_id}.json",
            indexed_in_chroma=False,
            chroma_record_id=None,
            timing=timing,
            error=error_payload,
        )


def derived_cache_socket(
    doc_id: str,
    image_id: str,
    operation: str = "cache_lookup",
    artifacts_root: str | Path = ARTIFACTS_ROOT,
) -> dict:
    """Check or report on derived cache status for Component W."""
    import time
    from core.sockets import build_derived_analysis_cache_socket, build_timing_payload, build_error_payload

    t0 = time.perf_counter()
    cache_path = Path(artifacts_root) / doc_id / "derived" / "vision" / f"{image_id}.json"
    rel_ref = f"derived/vision/{image_id}.json"

    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            total_ms = (time.perf_counter() - t0) * 1000.0
            return build_derived_analysis_cache_socket(
                operation=operation,
                doc_id=doc_id,
                image_id=image_id,
                cache_file_ref=rel_ref,
                hit=True,
                cache_path=str(cache_path),
                analyses_count=len(cache.get("analyses", [])),
                current_keys=list(cache.get("current", {}).keys()),
                byte_size=cache_path.stat().st_size,
                duration_ms=total_ms,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            error_payload = build_error_payload("CACHE_READ_ERROR", type(e).__name__, str(e), exc=e)
            return build_derived_analysis_cache_socket(
                operation=operation,
                doc_id=doc_id,
                image_id=image_id,
                cache_file_ref=rel_ref,
                hit=False,
                cache_path=str(cache_path),
                duration_ms=total_ms,
                error=error_payload,
            )
    else:
        total_ms = (time.perf_counter() - t0) * 1000.0
        return build_derived_analysis_cache_socket(
            operation=operation,
            doc_id=doc_id,
            image_id=image_id,
            cache_file_ref=rel_ref,
            hit=False,
            cache_path=str(cache_path),
            duration_ms=total_ms,
        )
