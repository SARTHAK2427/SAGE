"""Truthful PyMuPDF fallback used only when Docling cannot parse a PDF."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import NormalizedDocument, NormalizedElement


def parse_pdf_fallback(working_path: Path | str, original_source_path: Path | str,
                       doc_id: str, original_sha: str, original_size: int,
                       docling_error: Optional[str] = None) -> NormalizedDocument:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF fallback parsing. Install pymupdf.") from exc

    source = Path(original_source_path)
    elements: list[NormalizedElement] = []
    text_number = table_number = 0
    with fitz.open(str(working_path)) as pdf:
        for page_index, page in enumerate(pdf, start=1):
            table_rects = []
            try:
                tables = getattr(page.find_tables(), "tables", [])
            except Exception:
                tables = []
            for table in tables:
                rows = [[(cell or "").strip() for cell in row] for row in (table.extract() or [])]
                if not rows:
                    continue
                table_number += 1
                element_id = f"table_{table_number:06d}"
                markdown = "\n".join("| " + " | ".join(row) + " |" for row in rows)
                bbox = list(table.bbox) if getattr(table, "bbox", None) else None
                if bbox:
                    table_rects.append(fitz.Rect(bbox))
                elements.append(NormalizedElement(id=element_id, type="table", subtype="table", order=len(elements),
                    text=markdown, page=page_index, bbox=bbox, table_data={"rows": rows, "markdown": markdown},
                    artifact_ref=f"tables/{element_id}.json"))
            for block in page.get_text("blocks"):
                text = (block[4] or "").strip()
                bbox = list(block[:4])
                if not text or any(fitz.Rect(bbox).intersects(rect) for rect in table_rects):
                    continue
                text_number += 1
                element_id = f"txt_{text_number:06d}"
                elements.append(NormalizedElement(id=element_id, type="text", subtype="paragraph", order=len(elements),
                    text=text, page=page_index, bbox=bbox, artifact_ref=f"text/page_{page_index:04d}.json"))
        for index, element in enumerate(elements):
            element.order = index
            element.previous_element = elements[index - 1].id if index else None
            element.next_element = elements[index + 1].id if index + 1 < len(elements) else None
        return NormalizedDocument(doc_id=doc_id, source_path=str(source), original_name=source.name, extension=".pdf",
            mime_type="application/pdf", sha256=original_sha, file_size_bytes=original_size,
            parser_name="pdf_fallback", parser_version=getattr(fitz, "__version__", "unknown"), elements=elements,
            page_count=len(pdf), warnings=[f"Docling failed ({docling_error or 'conversion error'}); PyMuPDF fallback recovered text and native tables. Layout and images may be incomplete."])
