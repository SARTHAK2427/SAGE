"""
sage_document_db/router.py
Deterministic file-type router.

Uses file extension only — no AI/MIME detection.
Raises ValueError for unsupported types with a helpful message.
"""

from __future__ import annotations
from pathlib import Path


# ---------------------------------------------------------------------------
# Supported extension → parser tag mapping
# ---------------------------------------------------------------------------

_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".md"}
_SIMPLE_TEXT_EXTENSIONS = {".txt"}
_SIMPLE_JSON_EXTENSIONS = {".json"}
_SIMPLE_CSV_EXTENSIONS = {".csv"}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

_ALL_SUPPORTED = (
    _DOCLING_EXTENSIONS
    | _SIMPLE_TEXT_EXTENSIONS
    | _SIMPLE_JSON_EXTENSIONS
    | _SIMPLE_CSV_EXTENSIONS
    | _IMAGE_EXTENSIONS
)


def get_parser_tag(source_path: str | Path) -> str:
    """Return the parser tag for the given source file path.

    Tags:
        "docling"       — PDF, DOCX, PPTX, XLSX, MD
        "txt"           — plain text
        "json"          — JSON
        "csv"           — CSV
        "image"         — standalone image (PNG/JPG/JPEG/WEBP)

    Raises ValueError for unsupported extensions.
    """
    ext = Path(source_path).suffix.lower()

    if ext in _DOCLING_EXTENSIONS:
        return "docling"
    if ext in _SIMPLE_TEXT_EXTENSIONS:
        return "txt"
    if ext in _SIMPLE_JSON_EXTENSIONS:
        return "json"
    if ext in _SIMPLE_CSV_EXTENSIONS:
        return "csv"
    if ext in _IMAGE_EXTENSIONS:
        return "image"

    raise ValueError(
        f"Unsupported file extension: '{ext}' (from '{Path(source_path).name}').\n"
        f"Supported extensions: {sorted(_ALL_SUPPORTED)}"
    )


def is_supported(source_path: str | Path) -> bool:
    """Return True if the extension is supported by the router."""
    ext = Path(source_path).suffix.lower()
    return ext in _ALL_SUPPORTED
