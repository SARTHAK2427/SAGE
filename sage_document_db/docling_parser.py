"""
sage_document_db/docling_parser.py
Docling-based parser for PDF, DOCX, PPTX, XLSX, and MD files.

Rules:
- OCR is disabled.
- Table structure extraction is enabled.
- Picture image generation is enabled; page image generation is disabled.
- Unknown item types with text → preserved as generic text + warning.
- Unknown item types without text → warning, skip.
- Hyperlinks are annotations of their source text element (no links/ dir).
- Canonical text is never paraphrased.
- Never invent page numbers or coordinates.
"""

from __future__ import annotations
import io
from pathlib import Path

from .models import (
    LinkAnnotation,
    NormalizedDocument,
    NormalizedElement,
)
from .utils import make_doc_id, sha256_file


# ---------------------------------------------------------------------------
# ID counter helpers
# ---------------------------------------------------------------------------

class _IDCounters:
    def __init__(self):
        self.txt = 0
        self.img = 0
        self.table = 0
        self.code = 0

    def next_txt(self) -> str:
        self.txt += 1
        return f"txt_{self.txt:06d}"

    def next_img(self) -> str:
        self.img += 1
        return f"img_{self.img:06d}"

    def next_table(self) -> str:
        self.table += 1
        return f"table_{self.table:06d}"

    def next_code(self) -> str:
        self.code += 1
        return f"code_{self.code:06d}"


# ---------------------------------------------------------------------------
# Docling version detection helper
# ---------------------------------------------------------------------------

def _get_docling_version() -> str | None:
    try:
        import importlib.metadata
        return importlib.metadata.version("docling")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF pipeline options builder
# ---------------------------------------------------------------------------

def _build_pdf_pipeline_options():
    """Build PdfPipelineOptions with required settings.

    Adapts to installed Docling API: if field names differ slightly,
    falls back gracefully and records a warning.
    """
    try:
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        opts = PdfPipelineOptions()
        opts.do_ocr = False
        opts.do_table_structure = True
        opts.generate_picture_images = True
        opts.generate_page_images = False

        # Disable picture description / VLM pipeline if available
        if hasattr(opts, "images_scale"):
            opts.images_scale = 1.0  # keep at 1x, not 0
        if hasattr(opts, "do_picture_description"):
            opts.do_picture_description = False

        return opts, []
    except Exception as e:
        return None, [f"Could not configure PdfPipelineOptions: {e}"]


# ---------------------------------------------------------------------------
# Hyperlink extraction helper
# ---------------------------------------------------------------------------

def _extract_links(item) -> list[LinkAnnotation]:
    """Extract hyperlinks from a Docling text item if available."""
    links: list[LinkAnnotation] = []
    try:
        # Docling may expose hyperlinks through different attributes
        # Try common patterns; never raise if unavailable
        if hasattr(item, "hyperlinks") and item.hyperlinks:
            for hl in item.hyperlinks:
                url = getattr(hl, "url", None) or getattr(hl, "uri", None)
                if url:
                    links.append(LinkAnnotation(
                        text=getattr(hl, "text", None),
                        url=str(url),
                        start_char=getattr(hl, "start_char", None),
                        end_char=getattr(hl, "end_char", None),
                    ))
        elif hasattr(item, "captions"):
            pass  # captions are not hyperlinks
    except Exception:
        pass
    return links


# ---------------------------------------------------------------------------
# Page/location extraction helpers
# ---------------------------------------------------------------------------

def _get_page(item) -> int | None:
    """Extract 1-based page number from a Docling item's provenance."""
    try:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            p = prov[0]
            page = getattr(p, "page_no", None)
            if page is not None:
                return int(page)
    except Exception:
        pass
    return None


def _get_bbox(item) -> list[float] | None:
    """Extract bounding box as [l, t, r, b] from a Docling item."""
    try:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            bbox = getattr(prov[0], "bbox", None)
            if bbox is not None:
                return [
                    float(getattr(bbox, "l", 0)),
                    float(getattr(bbox, "t", 0)),
                    float(getattr(bbox, "r", 0)),
                    float(getattr(bbox, "b", 0)),
                ]
    except Exception:
        pass
    return None


def _get_provenance(item) -> list[dict]:
    """Extract raw provenance dicts from a Docling item."""
    try:
        prov = getattr(item, "prov", None)
        if prov:
            result = []
            for p in prov:
                result.append({
                    "page_no": getattr(p, "page_no", None),
                    "bbox": str(getattr(p, "bbox", None)),
                })
            return result
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Image extraction helper
# ---------------------------------------------------------------------------

def _extract_image_bytes(item) -> bytes | None:
    """Try to get raw image bytes from a PictureItem."""
    try:
        img = getattr(item, "image", None)
        if img is None:
            return None
        # Docling may expose a PIL image or bytes
        if hasattr(img, "pil_image") and img.pil_image is not None:
            buf = io.BytesIO()
            img.pil_image.save(buf, format="PNG")
            return buf.getvalue()
        if hasattr(img, "uri") and img.uri:
            # Some versions expose a data URI or file URI
            uri = str(img.uri)
            if uri.startswith("data:image"):
                import base64
                _, data = uri.split(",", 1)
                return base64.b64decode(data)
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Table data extraction helper
# ---------------------------------------------------------------------------

def _extract_table_data(item) -> dict | None:
    """Extract table rows as list[list[str]] from a TableItem."""
    try:
        if hasattr(item, "data") and item.data is not None:
            data = item.data
            # Docling TableData has .grid: list[list[TableCell]]
            if hasattr(data, "grid"):
                rows = []
                for row in data.grid:
                    cells = []
                    for cell in row:
                        text = getattr(cell, "text", "")
                        cells.append(str(text) if text is not None else "")
                    rows.append(cells)
                return {"rows": rows}
            # Fallback: try export_to_dataframe or similar
        # Try export_to_markdown as last resort
        if hasattr(item, "export_to_markdown"):
            md = item.export_to_markdown()
            if md:
                return {"markdown": md, "rows": []}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Subtype detection: is this item a code block?
# ---------------------------------------------------------------------------

def _is_code_item(item, label: str) -> bool:
    """Return True only if Docling explicitly/reliably identifies code."""
    code_labels = {"code", "code_block", "code-block", "listing", "verbatim"}
    return label.lower() in code_labels


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_with_docling(source_path: str | Path) -> NormalizedDocument:
    """Parse a document with Docling and return a NormalizedDocument.

    Supported: PDF, DOCX, PPTX, XLSX, MD.
    """
    from docling.document_converter import DocumentConverter
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption
    except ImportError:
        InputFormat = None
        PdfFormatOption = None

    path = Path(source_path)
    sha = sha256_file(path)
    doc_id = make_doc_id(sha)
    warnings: list[str] = []
    version = _get_docling_version()

    # Build converter with PDF options
    pdf_opts, opt_warnings = _build_pdf_pipeline_options()
    warnings.extend(opt_warnings)

    try:
        if pdf_opts is not None and PdfFormatOption is not None and InputFormat is not None:
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
                }
            )
        else:
            # Fallback: plain converter, PDF options not applied
            converter = DocumentConverter()
            if path.suffix.lower() == ".pdf":
                warnings.append(
                    "PdfFormatOption not available; PDF pipeline options not applied. "
                    "OCR may or may not be active depending on installed Docling version."
                )
    except Exception as e:
        converter = DocumentConverter()
        warnings.append(f"Failed to configure DocumentConverter with PDF options: {e}")

    # Run conversion
    result = converter.convert(str(path))
    doc = result.document

    # Determine page count
    page_count: int | None = None
    try:
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)
    except Exception:
        pass

    # ---------------------------------------------------------------------------
    # Iterate document items in reading order
    # ---------------------------------------------------------------------------
    counters = _IDCounters()
    elements: list[NormalizedElement] = []

    try:
        items_iter = doc.iterate_items()
    except AttributeError:
        # Older Docling APIs
        try:
            items_iter = iter(doc.body.children) if hasattr(doc, "body") else iter([])
        except Exception:
            items_iter = iter([])
            warnings.append("Could not iterate document items; document may be empty.")

    for item, _ in items_iter:
        item_type = type(item).__name__
        label = ""
        try:
            label = str(getattr(item, "label", "")).strip()
        except Exception:
            pass

        # --- TextItem ---
        if "TextItem" in item_type or item_type == "TextItem":
            text = None
            try:
                text = item.text
            except Exception:
                pass
            if text is None:
                try:
                    text = str(item)
                except Exception:
                    pass

            # Determine if this is actually a code block
            if _is_code_item(item, label):
                el_id = counters.next_code()
                el_type = "code"
                el_subtype = label or "code"
            else:
                el_id = counters.next_txt()
                el_type = "text"
                el_subtype = label or "paragraph"

            elements.append(NormalizedElement(
                id=el_id,
                type=el_type,
                subtype=el_subtype,
                order=len(elements),
                text=text,
                page=_get_page(item),
                bbox=_get_bbox(item),
                provenance=_get_provenance(item),
                links=_extract_links(item),
            ))

        # --- PictureItem ---
        elif "PictureItem" in item_type or item_type == "PictureItem":
            el_id = counters.next_img()
            img_bytes = _extract_image_bytes(item)
            if img_bytes is None:
                warnings.append(
                    f"PictureItem exists but image bytes could not be extracted "
                    f"(will be recorded as image element with no file). "
                    f"Element: {el_id}"
                )
            elements.append(NormalizedElement(
                id=el_id,
                type="image",
                subtype=label or "picture",
                order=len(elements),
                image=img_bytes,
                page=_get_page(item),
                bbox=_get_bbox(item),
                provenance=_get_provenance(item),
            ))

        # --- TableItem ---
        elif "TableItem" in item_type or item_type == "TableItem":
            el_id = counters.next_table()
            table_data = _extract_table_data(item)
            if table_data is None:
                warnings.append(
                    f"TableItem serialization failed for element {el_id}; "
                    f"table will be recorded with empty data."
                )
                table_data = {"rows": []}
            elements.append(NormalizedElement(
                id=el_id,
                type="table",
                subtype=label or "table",
                order=len(elements),
                table_data=table_data,
                page=_get_page(item),
                bbox=_get_bbox(item),
                provenance=_get_provenance(item),
            ))

        # --- Unknown item type ---
        else:
            # Try to salvage text
            salvaged_text = None
            try:
                salvaged_text = getattr(item, "text", None)
                if salvaged_text is None:
                    exported = getattr(item, "export_to_markdown", None)
                    if exported:
                        salvaged_text = exported()
            except Exception:
                pass

            if salvaged_text and salvaged_text.strip():
                el_id = counters.next_txt()
                elements.append(NormalizedElement(
                    id=el_id,
                    type="text",
                    subtype=label or item_type,
                    order=len(elements),
                    text=salvaged_text,
                    page=_get_page(item),
                    bbox=_get_bbox(item),
                    provenance=_get_provenance(item),
                ))
                warnings.append(
                    f"Unknown Docling item type '{item_type}' (label='{label}') "
                    f"salvaged as text element {el_id}."
                )
            else:
                warnings.append(
                    f"Unknown Docling item type '{item_type}' (label='{label}') "
                    f"had no usable text; skipped."
                )

    # ---------------------------------------------------------------------------
    # Set global reading-order chain
    # ---------------------------------------------------------------------------
    for i, el in enumerate(elements):
        el.previous_element = elements[i - 1].id if i > 0 else None
        el.next_element = elements[i + 1].id if i + 1 < len(elements) else None

    # Determine MIME type
    ext_to_mime = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".md": "text/markdown",
    }

    return NormalizedDocument(
        doc_id=doc_id,
        source_path=str(path.resolve()),
        original_name=path.name,
        extension=path.suffix.lower(),
        mime_type=ext_to_mime.get(path.suffix.lower()),
        sha256=sha,
        file_size_bytes=path.stat().st_size,
        parser_name="docling",
        parser_version=version,
        elements=elements,
        page_count=page_count,
        warnings=warnings,
    )
