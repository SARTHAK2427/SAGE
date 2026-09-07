"""
SAGE/core/mappers/vision_input.py
Gemma vision arguments → Qwen-VL plain-text prompt + canonical image resolution.

Gemma emits:
    {
      "tool": "vision_ocr",
      "function": "analyze_image",
      "arguments": {
        "instruction": "...",
        "doc_id": "..." | null,
        "image_id": "...",
        "image_ref": "..." | null
      }
    }

Two responsibilities:
    1. build_vision_text_prompt(instruction)
       → Produces the plain TASK: text forwarded to Qwen-VL.
         Gemma's JSON wrapper NEVER reaches the model.
    2. resolve_vision_image(db, doc_id, image_id, image_ref)
       → Deterministically resolves image_id to a canonical local path
         via the artifact store. The local path is internal-only:
         it is used by the multimodal model transport but must NEVER
         appear in the Gemma-facing result contract.

Rules:
    - Do not send Gemma's JSON arguments to Qwen-VL.
    - Do not inject doc manifest, Chroma data, tool registry info,
      or SAGE architecture into the VL model input.
    - Qwen-VL returns plain task-focused text. Do NOT force JSON output.
    - The local image path stays internal (Python/transport only).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def build_vision_text_prompt(instruction: str) -> str:
    """Build the Qwen-VL user-turn text prompt from Gemma's instruction.

    Args:
        instruction: The task/instruction string from Gemma's tool arguments.

    Returns:
        Plain text following the vision input protocol:
            TASK:
            <instruction>
    """
    return f"TASK:\n{instruction.strip()}"


def resolve_vision_image(
    db: Any,
    doc_id: str | None,
    image_id: str,
    image_ref: str | None = None,
) -> str:
    """Deterministically resolve an image_id to its canonical local filesystem path.

    This is INTERNAL plumbing — not an agentic cross-tool call.
    The resolved path is used by the model transport layer only.
    It must NEVER appear in the Gemma-facing result contract.

    Resolution order:
        1. If image_ref resolves to an existing file path, use it directly.
        2. Fetch the canonical artifact element via db.artifact_fetch and
           read its local_path.

    Args:
        db:        SageDocumentDB instance (or mock compatible).
        doc_id:    Document identifier (may be None when image_ref is canonical).
        image_id:  Element ID of the image artifact (e.g. "img_000001").
        image_ref: Optional stable registered reference (may encode the path).

    Returns:
        Absolute local path string to the image file.

    Raises:
        FileNotFoundError: When no valid path can be resolved.
        ValueError:        When the referenced element is not an image type.
    """
    import os

    # Attempt resolution via image_ref first (may be a resolvable stable ref)
    if image_ref:
        # If the image_ref is already a valid existing path, use it directly
        if os.path.isabs(image_ref) and os.path.isfile(image_ref):
            return image_ref
        # Otherwise fall through to DB resolution

    if db is None:
        raise FileNotFoundError(f"No database available to resolve image '{image_id}'")

    try:
        element = db.artifact_fetch(doc_id or "", image_id)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve image '{image_id}' from doc '{doc_id}': {exc}"
        ) from exc

    elem_type = element.get("type") if isinstance(element, dict) else None
    if elem_type != "image":
        raise ValueError(
            f"Element '{image_id}' is not an image (type={elem_type!r})"
        )

    local_path = element.get("local_path") if isinstance(element, dict) else None
    if not local_path:
        raise FileNotFoundError(
            f"Image artifact '{image_id}' has no local_path in its element record"
        )

    if not os.path.isfile(local_path):
        raise FileNotFoundError(
            f"Resolved path for '{image_id}' does not exist: {local_path}"
        )

    return local_path
