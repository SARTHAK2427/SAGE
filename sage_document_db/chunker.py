"""
sage_document_db/chunker.py
Structure-aware text chunker.

Converts a NormalizedDocument into SearchRecord objects for Chroma indexing.
Does NOT modify canonical artifacts.

Correction 5: Chunker receives an EmbeddingService instance and calls
embedding_service.count_tokens() for all token counting.
It does NOT import or instantiate SentenceTransformer itself.

Key rules:
- Never mix elements from different documents
- Preserve source element IDs in every record
- Keep chunks below CHUNK_TARGET_TOKENS
- Overlap CHUNK_OVERLAP_TOKENS between adjacent text chunks
- Attach nearby image refs based on reading-order adjacency
- Produce separate table and code records
- Globally unique Chroma record IDs: doc_<id>:chunk:NNNNNN etc.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field

from .config import CHUNK_TARGET_TOKENS, CHUNK_OVERLAP_TOKENS
from .models import NormalizedDocument, NormalizedElement, SearchRecord
from .utils import encode_list_field


# ---------------------------------------------------------------------------
# Chunk accumulator (internal)
# ---------------------------------------------------------------------------

@dataclass
class _ChunkBuffer:
    source_ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)
    table_refs: list[str] = field(default_factory=list)
    code_refs: list[str] = field(default_factory=list)
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    start_order: int = 0
    end_order: int = 0
    heading_context: str | None = None

    def combined_text(self) -> str:
        parts = []
        if self.heading_context:
            parts.append(self.heading_context)
        parts.extend(self.texts)
        return "\n\n".join(p for p in parts if p)

    def is_empty(self) -> bool:
        return not self.source_ids

    def clear(self) -> None:
        self.source_ids.clear()
        self.texts.clear()
        self.image_refs.clear()
        self.table_refs.clear()
        self.code_refs.clear()
        self.page = None
        self.slide = None
        self.sheet = None
        self.start_order = 0
        self.end_order = 0
        self.heading_context = None


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

_HEADING_SUBTYPES = {
    "title", "section_header", "section-header", "heading",
    "h1", "h2", "h3", "h4", "h5", "h6",
}

def _is_heading(el: NormalizedElement) -> bool:
    if el.subtype and el.subtype.lower() in _HEADING_SUBTYPES:
        return True
    return False


# ---------------------------------------------------------------------------
# Sentence splitting (simple, no NLTK dependency)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Main Chunker
# ---------------------------------------------------------------------------

class Chunker:
    """Converts NormalizedDocument → list[SearchRecord].

    Must be initialized with an EmbeddingService so token counting uses
    the same model tokenizer as the embeddings (Correction 5).
    """

    def __init__(
        self,
        embedding_service,               # EmbeddingService instance
        chunk_target_tokens: int = CHUNK_TARGET_TOKENS,
        chunk_overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
    ) -> None:
        self._emb = embedding_service
        self._target = chunk_target_tokens
        self._overlap = chunk_overlap_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: NormalizedDocument) -> list[SearchRecord]:
        """Return all SearchRecords for a NormalizedDocument."""
        records: list[SearchRecord] = []
        chunk_counter = [0]
        table_counter = [0]
        code_counter = [0]

        # Build a set of all image element IDs for adjacency lookup
        image_ids: set[str] = {el.id for el in doc.elements if el.type == "image"}

        def flush(buf: _ChunkBuffer, overlap_texts: list[str]) -> None:
            text = buf.combined_text()
            if not text.strip():
                return
            chunk_counter[0] += 1
            record_id = f"{doc.doc_id}:chunk:{chunk_counter[0]:06d}"

            meta = {
                "doc_id": doc.doc_id,
                "record_type": "text_chunk",
                "unit_type": _unit_type(buf),
                "page": buf.page,
                "slide": buf.slide,
                "sheet": buf.sheet,
                "start_order": buf.start_order,
                "end_order": buf.end_order,
                "source_element_ids_json": encode_list_field(buf.source_ids),
                "image_refs_json": encode_list_field(buf.image_refs),
                "table_refs_json": encode_list_field(buf.table_refs),
                "code_refs_json": encode_list_field(buf.code_refs),
            }
            # Remove None values so Chroma doesn't choke
            meta = {k: v for k, v in meta.items() if v is not None}

            records.append(SearchRecord(id=record_id, text=text, metadata=meta))

        buf = _ChunkBuffer()
        current_page_key = None

        for el in doc.elements:
            if el.type == "image":
                # Images are not directly chunked; their refs propagate into
                # adjacent text chunks via the image_refs field
                continue

            if el.type == "table":
                # Flush any pending text chunk first
                if not buf.is_empty():
                    flush(buf, [])
                    buf.clear()

                # Create separate table records
                tbl_records = self._chunk_table(el, doc.doc_id, table_counter)
                records.extend(tbl_records)
                continue

            if el.type == "code":
                # Flush any pending text chunk first
                if not buf.is_empty():
                    flush(buf, [])
                    buf.clear()

                code_records = self._chunk_code(el, doc.doc_id, code_counter, buf.heading_context)
                records.extend(code_records)
                continue

            # --- Text element ---
            if el.text is None or not el.text.strip():
                continue

            # Detect page boundary
            page_key = _page_key(el)
            if page_key != current_page_key and current_page_key is not None:
                if not buf.is_empty():
                    flush(buf, [])
                    buf.clear()
            current_page_key = page_key

            # Detect heading → flush and start new context
            if _is_heading(el):
                if not buf.is_empty():
                    flush(buf, [])
                    buf.clear()
                buf.heading_context = el.text

            # Collect adjacent image refs
            img_refs = _collect_image_refs(el, image_ids, doc.elements)

            # Check if this element alone exceeds target
            el_tokens = self._emb.count_tokens(el.text)

            if el_tokens >= self._target:
                # Flush pending buffer first
                if not buf.is_empty():
                    flush(buf, [])
                    buf.clear()
                # Split this oversized element
                split_records = self._split_large_element(
                    el, doc.doc_id, chunk_counter, img_refs, buf.heading_context
                )
                records.extend(split_records)
                continue

            # Check if adding this element would overflow the buffer
            combined = buf.combined_text() + "\n\n" + el.text
            if not buf.is_empty() and self._emb.count_tokens(combined) > self._target:
                flush(buf, [])
                # Keep last N tokens of previous chunk as overlap
                overlap_text = _build_overlap(buf.texts, self._overlap, self._emb)
                buf.clear()
                if overlap_text:
                    # Put overlap text at start (not as a source_id reference)
                    buf.texts.append(f"[...{overlap_text}]")

            # Add element to buffer
            if buf.is_empty():
                buf.start_order = el.order
                buf.page = el.page
                buf.slide = el.slide
                buf.sheet = el.sheet

            buf.source_ids.append(el.id)
            buf.texts.append(el.text)
            buf.end_order = el.order
            buf.image_refs.extend(r for r in img_refs if r not in buf.image_refs)

        # Flush final buffer
        if not buf.is_empty():
            flush(buf, [])

        return records

    # ------------------------------------------------------------------
    # Table chunking
    # ------------------------------------------------------------------

    def _chunk_table(
        self, el: NormalizedElement, doc_id: str, counter: list[int]
    ) -> list[SearchRecord]:
        table_data = el.table_data or {}
        rows = table_data.get("rows", [])
        markdown = table_data.get("markdown")

        # Convert rows to readable text
        if rows:
            lines = []
            for row in rows:
                lines.append(" | ".join(str(c) for c in row))
            text = "Table\n" + "\n".join(lines)
        elif markdown:
            text = f"Table\n{markdown}"
        else:
            return []

        # Split if too large
        if self._emb.count_tokens(text) <= self._target:
            counter[0] += 1
            record_id = f"{doc_id}:table:{el.id}:part:{counter[0]:03d}"
            meta = {
                "doc_id": doc_id,
                "record_type": "table",
                "unit_type": "table",
                "page": el.page,
                "start_order": el.order,
                "end_order": el.order,
                "source_element_ids_json": encode_list_field([]),
                "image_refs_json": encode_list_field([]),
                "table_refs_json": encode_list_field([el.id]),
                "code_refs_json": encode_list_field([]),
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            return [SearchRecord(id=record_id, text=text, metadata=meta)]
        else:
            # Split by rows
            return self._split_table_rows(el, rows, doc_id, counter)

    def _split_table_rows(
        self, el: NormalizedElement, rows: list, doc_id: str, counter: list[int]
    ) -> list[SearchRecord]:
        if not rows:
            return []
        records = []
        header = rows[0] if rows else []
        header_text = " | ".join(str(c) for c in header)
        batch: list[list] = [header] if header else []
        batch_text = f"Table\n{header_text}" if header else "Table"

        for row in rows[1:]:
            row_text = " | ".join(str(c) for c in row)
            candidate = batch_text + "\n" + row_text
            if self._emb.count_tokens(candidate) > self._target and len(batch) > 1:
                counter[0] += 1
                record_id = f"{doc_id}:table:{el.id}:part:{counter[0]:03d}"
                meta = {
                    "doc_id": doc_id,
                    "record_type": "table",
                    "unit_type": "table",
                    "page": el.page,
                    "start_order": el.order,
                    "end_order": el.order,
                    "source_element_ids_json": encode_list_field([]),
                    "image_refs_json": encode_list_field([]),
                    "table_refs_json": encode_list_field([el.id]),
                    "code_refs_json": encode_list_field([]),
                }
                meta = {k: v for k, v in meta.items() if v is not None}
                records.append(SearchRecord(id=record_id, text=batch_text, metadata=meta))
                batch = [header, row] if header else [row]
                batch_text = f"Table\n{header_text}\n{row_text}" if header else f"Table\n{row_text}"
            else:
                batch.append(row)
                batch_text = candidate

        if batch:
            counter[0] += 1
            record_id = f"{doc_id}:table:{el.id}:part:{counter[0]:03d}"
            meta = {
                "doc_id": doc_id,
                "record_type": "table",
                "unit_type": "table",
                "page": el.page,
                "start_order": el.order,
                "end_order": el.order,
                "source_element_ids_json": encode_list_field([]),
                "image_refs_json": encode_list_field([]),
                "table_refs_json": encode_list_field([el.id]),
                "code_refs_json": encode_list_field([]),
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            records.append(SearchRecord(id=record_id, text=batch_text, metadata=meta))

        return records

    # ------------------------------------------------------------------
    # Code chunking
    # ------------------------------------------------------------------

    def _chunk_code(
        self,
        el: NormalizedElement,
        doc_id: str,
        counter: list[int],
        heading_context: str | None,
    ) -> list[SearchRecord]:
        if not el.text or not el.text.strip():
            return []

        prefix = f"{heading_context}\n\n" if heading_context else ""
        text = prefix + el.text

        if self._emb.count_tokens(text) <= self._target:
            counter[0] += 1
            record_id = f"{doc_id}:code:{el.id}:part:{counter[0]:03d}"
            meta = {
                "doc_id": doc_id,
                "record_type": "code",
                "unit_type": "code",
                "page": el.page,
                "start_order": el.order,
                "end_order": el.order,
                "source_element_ids_json": encode_list_field([el.id]),
                "image_refs_json": encode_list_field([]),
                "table_refs_json": encode_list_field([]),
                "code_refs_json": encode_list_field([el.id]),
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            return [SearchRecord(id=record_id, text=text, metadata=meta)]
        else:
            lines = el.text.split("\n")
            return self._split_code_lines(el, lines, prefix, doc_id, counter)

    def _split_code_lines(
        self,
        el: NormalizedElement,
        lines: list[str],
        prefix: str,
        doc_id: str,
        counter: list[int],
    ) -> list[SearchRecord]:
        records = []
        batch_lines: list[str] = []
        batch_text = prefix

        for line in lines:
            candidate = batch_text + ("\n" if batch_text else "") + line
            if self._emb.count_tokens(candidate) > self._target and batch_lines:
                counter[0] += 1
                record_id = f"{doc_id}:code:{el.id}:part:{counter[0]:03d}"
                meta = {
                    "doc_id": doc_id,
                    "record_type": "code",
                    "unit_type": "code",
                    "page": el.page,
                    "start_order": el.order,
                    "end_order": el.order,
                    "source_element_ids_json": encode_list_field([el.id]),
                    "image_refs_json": encode_list_field([]),
                    "table_refs_json": encode_list_field([]),
                    "code_refs_json": encode_list_field([el.id]),
                }
                meta = {k: v for k, v in meta.items() if v is not None}
                records.append(SearchRecord(id=record_id, text=batch_text, metadata=meta))
                batch_lines = [line]
                batch_text = prefix + line
            else:
                batch_lines.append(line)
                batch_text = candidate

        if batch_lines:
            counter[0] += 1
            record_id = f"{doc_id}:code:{el.id}:part:{counter[0]:03d}"
            meta = {
                "doc_id": doc_id,
                "record_type": "code",
                "unit_type": "code",
                "page": el.page,
                "start_order": el.order,
                "end_order": el.order,
                "source_element_ids_json": encode_list_field([el.id]),
                "image_refs_json": encode_list_field([]),
                "table_refs_json": encode_list_field([]),
                "code_refs_json": encode_list_field([el.id]),
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            records.append(SearchRecord(id=record_id, text=batch_text, metadata=meta))

        return records

    # ------------------------------------------------------------------
    # Oversized element splitting
    # ------------------------------------------------------------------

    def _split_large_element(
        self,
        el: NormalizedElement,
        doc_id: str,
        counter: list[int],
        img_refs: list[str],
        heading_context: str | None,
    ) -> list[SearchRecord]:
        """Split a single oversized text element with sentence + token windows."""
        text = el.text or ""
        sentences = _split_sentences(text)

        if not sentences:
            return []

        records = []
        window: list[str] = []
        window_text = ""
        prefix = f"{heading_context}\n\n" if heading_context else ""

        def flush_window() -> None:
            nonlocal window, window_text
            full_text = prefix + window_text
            if not full_text.strip():
                return
            counter[0] += 1
            record_id = f"{doc_id}:chunk:{counter[0]:06d}"
            meta = {
                "doc_id": doc_id,
                "record_type": "text_chunk",
                "unit_type": "text",
                "page": el.page,
                "slide": el.slide,
                "sheet": el.sheet,
                "start_order": el.order,
                "end_order": el.order,
                "source_element_ids_json": encode_list_field([el.id]),
                "image_refs_json": encode_list_field(img_refs),
                "table_refs_json": encode_list_field([]),
                "code_refs_json": encode_list_field([]),
            }
            meta = {k: v for k, v in meta.items() if v is not None}
            records.append(SearchRecord(id=record_id, text=full_text, metadata=meta))

        for sent in sentences:
            candidate = (window_text + " " + sent).strip()
            if self._emb.count_tokens(prefix + candidate) > self._target and window:
                flush_window()
                # Overlap: keep last sentence
                overlap = window[-1] if window else ""
                window = [overlap, sent] if overlap else [sent]
                window_text = " ".join(window).strip()
            else:
                window.append(sent)
                window_text = " ".join(window).strip()

        if window:
            flush_window()

        return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page_key(el: NormalizedElement) -> str | None:
    """Return a string key representing the page/slide/sheet of this element."""
    if el.slide is not None:
        return f"slide_{el.slide}"
    if el.sheet is not None:
        return f"sheet_{el.sheet}"
    if el.page is not None:
        return f"page_{el.page}"
    return None


def _unit_type(buf: _ChunkBuffer) -> str:
    if buf.slide is not None:
        return "slide"
    if buf.sheet is not None:
        return "sheet"
    if buf.page is not None:
        return "page"
    return "document"


def _collect_image_refs(
    el: NormalizedElement,
    image_ids: set[str],
    all_elements: list[NormalizedElement],
) -> list[str]:
    """Return image IDs that are adjacent to this element in reading order.

    An image is associated with a chunk if:
        image.previous_element == el.id
        OR image.next_element == el.id
    """
    refs: list[str] = []
    for img_el in all_elements:
        if img_el.type != "image":
            continue
        if img_el.previous_element == el.id or img_el.next_element == el.id:
            refs.append(img_el.id)
    return refs


def _build_overlap(
    texts: list[str],
    overlap_tokens: int,
    embedding_service,
) -> str:
    """Build overlap text from the tail of a chunk's text list."""
    if not texts or overlap_tokens <= 0:
        return ""
    # Take sentences from the end until we have ~overlap_tokens worth
    overlap_parts: list[str] = []
    for text in reversed(texts):
        candidate = text + (" " + " ".join(reversed(overlap_parts)) if overlap_parts else "")
        if embedding_service.count_tokens(candidate) <= overlap_tokens:
            overlap_parts.insert(0, text)
        else:
            break
    return " ".join(overlap_parts)
