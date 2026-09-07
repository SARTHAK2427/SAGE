"""
sage_document_db/simple_parsers.py
Simple parsers for TXT, JSON, CSV, and standalone image files.

These parsers do not use Docling. They produce NormalizedDocument objects
with the same contract as the Docling parser.

Rules:
- Canonical text is never paraphrased.
- JSON: structured_data holds the real hierarchy; text holds deterministic
  serialization for search (Correction 4).
- No OCR of images at ingestion.
- Never invent page numbers or coordinates.
"""

from __future__ import annotations
import csv
import json
import shutil
from pathlib import Path

from .models import (
    LinkAnnotation,
    NormalizedDocument,
    NormalizedElement,
)
from .utils import iso_now, make_doc_id, safe_mkdir, sha256_file


# ---------------------------------------------------------------------------
# Internal ID counter helpers
# ---------------------------------------------------------------------------

def _txt_id(n: int) -> str:
    return f"txt_{n:06d}"

def _img_id(n: int) -> str:
    return f"img_{n:06d}"

def _table_id(n: int) -> str:
    return f"table_{n:06d}"


def _set_reading_order(elements: list[NormalizedElement]) -> None:
    """Set previous_element / next_element in a single pass."""
    for i, el in enumerate(elements):
        el.previous_element = elements[i - 1].id if i > 0 else None
        el.next_element = elements[i + 1].id if i + 1 < len(elements) else None


# ---------------------------------------------------------------------------
# TXT parser
# ---------------------------------------------------------------------------

def parse_txt(source_path: str | Path) -> NormalizedDocument:
    """Parse a plain text file into a NormalizedDocument.

    Reads UTF-8 strictly. If decoding fails, raises UnicodeDecodeError
    with a clear message — never silently corrupts text.
    """
    path = Path(source_path)
    sha = sha256_file(path)
    doc_id = make_doc_id(sha)

    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding, e.object, e.start, e.end,
            f"Failed to decode '{path.name}' as UTF-8. "
            "Ensure the file is UTF-8 encoded or convert it first."
        ) from e

    elements: list[NormalizedElement] = []
    if content.strip():
        elements.append(NormalizedElement(
            id=_txt_id(1),
            type="text",
            subtype="plain_text",
            order=0,
            text=content,
        ))

    _set_reading_order(elements)

    return NormalizedDocument(
        doc_id=doc_id,
        source_path=str(path.resolve()),
        original_name=path.name,
        extension=path.suffix.lower(),
        mime_type="text/plain",
        sha256=sha,
        file_size_bytes=path.stat().st_size,
        parser_name="simple_txt",
        parser_version=None,
        elements=elements,
        page_count=None,
    )


# ---------------------------------------------------------------------------
# JSON parser
# ---------------------------------------------------------------------------

def parse_json(source_path: str | Path) -> NormalizedDocument:
    """Parse a JSON file into a NormalizedDocument.

    Correction 4:
    - element.structured_data holds the full parsed Python object (canonical)
    - element.text holds json.dumps(obj, indent=2) for searchability only
    The structured_data field is the authoritative hierarchy.
    """
    path = Path(source_path)
    sha = sha256_file(path)
    doc_id = make_doc_id(sha)

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    serialized = json.dumps(obj, indent=2, ensure_ascii=False)

    elements: list[NormalizedElement] = [
        NormalizedElement(
            id=_txt_id(1),
            type="text",
            subtype="json_document",
            order=0,
            text=serialized,             # Deterministic search serialization
            structured_data=obj if isinstance(obj, dict) else {"root": obj},
        )
    ]

    _set_reading_order(elements)

    return NormalizedDocument(
        doc_id=doc_id,
        source_path=str(path.resolve()),
        original_name=path.name,
        extension=path.suffix.lower(),
        mime_type="application/json",
        sha256=sha,
        file_size_bytes=path.stat().st_size,
        parser_name="simple_json",
        parser_version=None,
        elements=elements,
        page_count=None,
    )


# ---------------------------------------------------------------------------
# CSV parser
# ---------------------------------------------------------------------------

def parse_csv(source_path: str | Path) -> NormalizedDocument:
    """Parse a CSV file into a NormalizedDocument.

    Each CSV file becomes one table element. Cell strings and row/column
    order are preserved exactly as read.
    """
    path = Path(source_path)
    sha = sha256_file(path)
    doc_id = make_doc_id(sha)

    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = [list(row) for row in reader]

    warnings: list[str] = []
    elements: list[NormalizedElement] = []

    if rows:
        elements.append(NormalizedElement(
            id=_table_id(1),
            type="table",
            subtype="csv_table",
            order=0,
            table_data={"rows": rows},
        ))
    else:
        warnings.append(f"CSV file '{path.name}' contained no rows.")

    _set_reading_order(elements)

    return NormalizedDocument(
        doc_id=doc_id,
        source_path=str(path.resolve()),
        original_name=path.name,
        extension=path.suffix.lower(),
        mime_type="text/csv",
        sha256=sha,
        file_size_bytes=path.stat().st_size,
        parser_name="simple_csv",
        parser_version=None,
        elements=elements,
        page_count=None,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Standalone image parser
# ---------------------------------------------------------------------------

def parse_image(source_path: str | Path) -> NormalizedDocument:
    """Parse a standalone image file into a NormalizedDocument.

    No OCR or image description is performed at ingestion time.
    The image file itself is recorded; actual copying to the artifact
    directory happens in ArtifactStore.write_document().
    """
    path = Path(source_path)
    sha = sha256_file(path)
    doc_id = make_doc_id(sha)

    ext_to_mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime = ext_to_mime.get(path.suffix.lower(), "image/unknown")

    elements: list[NormalizedElement] = [
        NormalizedElement(
            id=_img_id(1),
            type="image",
            subtype="standalone_image",
            order=0,
            # image bytes are not loaded into RAM here; the artifact store
            # will copy the file directly from source_path
        )
    ]

    _set_reading_order(elements)

    return NormalizedDocument(
        doc_id=doc_id,
        source_path=str(path.resolve()),
        original_name=path.name,
        extension=path.suffix.lower(),
        mime_type=mime,
        sha256=sha,
        file_size_bytes=path.stat().st_size,
        parser_name="simple_image",
        parser_version=None,
        elements=elements,
        page_count=None,
    )
