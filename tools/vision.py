"""
SAGE/tools/vision.py
Vision/OCR adapter scaffold for Qwen3-VL.

Architectural rules:
    - Qwen3-VL is a targeted visual specialist
    - It must NOT reparse whole PDFs
    - It only inspects explicitly selected image artifacts
    - The agent does not need to know Qwen is a model
    - The vision tool does NOT autonomously invoke other registered tools

Runtime-owned reference resolution (allowed):
    doc_id + image_id → canonical image path
    This is deterministic plumbing, not an agentic decision.
    The path stays backend-internal — Gemma never constructs it.

Save-vs-index:
    Raw vision output is saved with index_in_chroma=False by default.
    The eventual acceptance policy will index accepted results.
    Mock outputs are NEVER indexed.
"""

from __future__ import annotations
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Mock mode detection ───────────────────────────────────────────────────────

def _is_mock_mode() -> bool:
    return os.environ.get("SAGE_MOCK_MODE", "0") == "1"


# ─── Vision Adapter ──────────────────────────────────────────────────────────

def tool_vision_analyze(
    db,
    model_manager=None,
    model_client=None,
    *,
    doc_id: str,
    image_ids: list[str],
    instruction: str = "Describe this image in detail.",
) -> dict:
    """Analyze one or more image artifacts using Qwen3-VL.

    Args:
        db:             SageDocumentDB instance (injected at registration)
        model_manager:  ModelManager instance (injected; None in mock mode)
        model_client:   ModelClient instance (injected; None in mock mode)
        doc_id:         Document identifier
        image_ids:      List of image element IDs (e.g. ["img_000001"])
        instruction:    Textual task/instruction for the vision model

    Returns:
        {"status": "success", "results": [...]}
        or
        {"status": "error", "error": "..."}

    In mock mode, returns deterministic mock output WITHOUT persisting
    to the real derived storage or indexing into Chroma.
    """
    if not image_ids:
        return {"status": "error", "error": "No image_ids provided"}

    results = []
    mock = _is_mock_mode()

    for image_id in image_ids:
        try:
            result = _analyze_single_image(
                db=db,
                model_manager=model_manager,
                model_client=model_client,
                doc_id=doc_id,
                image_id=image_id,
                instruction=instruction,
                mock=mock,
            )
            results.append(result)
        except Exception as exc:
            logger.error("Vision failed for %s/%s: %s", doc_id, image_id, exc)
            results.append({
                "image_id": image_id,
                "status": "error",
                "error": str(exc),
            })

    all_ok = all(r.get("status") == "success" for r in results)
    return {
        "status": "success" if all_ok else "partial_error",
        "results": results,
    }


def _analyze_single_image(
    db,
    model_manager,
    model_client,
    doc_id: str,
    image_id: str,
    instruction: str,
    mock: bool,
) -> dict:
    """Analyze a single image artifact.

    Runtime-owned reference resolution:
        doc_id + image_id → canonical image path (backend-internal)
    """
    # Step 1: Resolve canonical image path via artifact_fetch
    element = db.artifact_fetch(doc_id, image_id)
    if element.get("type") != "image":
        raise ValueError(f"Element '{image_id}' is not an image (type={element.get('type')})")

    local_path = element.get("local_path", "mock_image.png")
    if not mock and (not local_path or not element.get("exists", False)):
        raise FileNotFoundError(f"Image file not found for {image_id}")

    # Step 2: Run inference (or mock)
    if mock:
        raw_output = f"MOCK: Image {image_id} contains visual content. Instruction: {instruction}"
        model_name = "mock"
        logger.info("Vision MOCK for %s/%s", doc_id, image_id)
    else:
        raw_output = _run_qwen_vision(
            model_manager=model_manager,
            model_client=model_client,
            image_path=local_path,
            instruction=instruction,
        )
        model_name = "Qwen3-VL-4B"

    # Step 3: Persist raw result
    # Mock results: do NOT persist to real derived storage
    # Real results: save with index_in_chroma=False (pending acceptance)
    if not mock:
        try:
            db.add_image_analysis(
                doc_id=doc_id,
                image_id=image_id,
                model=model_name,
                task_type="vision_analyze",
                raw_output=raw_output,
                instruction=instruction,
                index_in_chroma=False,  # Pending — not accepted yet
            )
        except Exception as exc:
            logger.warning("Failed to persist vision result: %s", exc)

    return {
        "image_id": image_id,
        "status": "success",
        "raw_output": raw_output,
        "model": model_name,
    }


def _run_qwen_vision(
    model_manager,
    model_client,
    image_path: str,
    instruction: str,
) -> str:
    """Invoke Qwen3-VL through the existing model runtime.

    This is an internal implementation dependency, NOT a cross-tool call.
    """
    import base64
    from pathlib import Path

    # Load and swap to vision model
    model_manager.ensure_model("document_analyzer")

    # Read image and encode as base64 data URI
    img_path = Path(image_path)
    img_bytes = img_path.read_bytes()
    ext = img_path.suffix.lstrip(".").lower()
    mime = f"image/{ext}" if ext != "jpg" else "image/jpeg"
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_uri = f"data:{mime};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ],
        }
    ]

    res = model_client.chat_completion(
        messages=messages,
        temperature=0.05,
        max_tokens=2048,
    )

    return res["content"]


# ── Canonical interface: vision_ocr / analyze_image (single image) ────────────

def tool_vision_analyze_image(
    db,
    model_manager=None,
    model_client=None,
    *,
    instruction: str,
    doc_id: str | None = None,
    image_id: str = "",
    image_ref: str | None = None,
) -> dict:
    """Canonical endpoint matching tools.json: vision_ocr / analyze_image.

    Single-image adapter. Resolves image_id to canonical path internally.
    The resolved local path is backend-only and never returned to Gemma.

    Args:
        instruction: Textual task for Qwen-VL (TASK: <instruction>)
        doc_id:      Document identifier (may be None when image_ref resolves directly)
        image_id:    Element ID of the image artifact
        image_ref:   Optional stable registered image reference

    Returns:
        Rich dict consumed by map_vision_result() in core/mappers/gemma_results.
    """
    if not image_id:
        return {
            "status": "error",
            "error": "image_id is required for vision_ocr / analyze_image",
        }

    result = tool_vision_analyze(
        db,
        model_manager,
        model_client,
        doc_id=doc_id or "",
        image_ids=[image_id],
        instruction=instruction,
    )

    # Promote single-image result to top-level for the mapper
    results_list = result.get("results", [])
    single = results_list[0] if results_list else {}
    overall_status = single.get("status", result.get("status", "error"))

    return {
        "status": overall_status,
        "image_id": image_id,
        "doc_id": doc_id or "",
        "analysis": single.get("raw_output", ""),
        "model": single.get("model", ""),
        # raw_output is duplicated as analysis; local_path is intentionally absent
        "results": results_list,  # kept for internal mappers; stripped by map_vision_result
        "error": single.get("error") if overall_status == "error" else None,
    }


# ─── Registration Helper ─────────────────────────────────────────────────────

def register_vision_tools(registry, db, model_manager=None, model_client=None) -> None:
    """Register vision tools with a ToolRegistry.

    Canonical (Gemma-visible via tools.json):
        vision_ocr / analyze_image

    Internal backward-compat alias:
        vision / analyze  (multi-image; used by legacy orchestrator paths)
    """
    from functools import partial

    # Canonical single-image function (matches tools.json)
    registry.register(
        "vision_ocr", "analyze_image",
        partial(tool_vision_analyze_image, db, model_manager, model_client),
    )

    # Internal multi-image legacy alias — NOT exposed to Gemma
    registry.register(
        "vision", "analyze",
        partial(tool_vision_analyze, db, model_manager, model_client),
    )
