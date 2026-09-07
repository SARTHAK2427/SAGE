"""
sage_document_db/models.py
Normalized internal data model for the SAGE document database layer.

These dataclasses form the shared contract between parsers, the artifact
store, the chunker, Chroma, and exact search. No external dependencies.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Sub-element: hyperlink annotation
# ---------------------------------------------------------------------------

@dataclass
class LinkAnnotation:
    """A hyperlink attached to a source text element.

    Belongs to the visible text it annotates — there is no separate links
    directory. Character offsets are optional; never invent them.
    """
    text: str | None        # Anchor/visible text
    url: str                # Target URL
    start_char: int | None = None   # Byte offset in element.text (may be None)
    end_char: int | None = None     # Byte offset in element.text (may be None)


# ---------------------------------------------------------------------------
# Core document element
# ---------------------------------------------------------------------------

@dataclass
class NormalizedElement:
    """One atomic content unit extracted from a source document.

    IDs are deterministic and unique within a doc_id:
        txt_000001, img_000001, table_000001, code_000001

    Chroma record IDs are globally unique and include doc_id.
    Never use Chroma chunk IDs as canonical source IDs.
    """
    id: str                         # Canonical element ID, e.g. txt_000001
    type: str                       # text | image | table | code
    subtype: str | None             # Docling label or parser subtype
    order: int                      # Global reading-order position (0-based)

    # Textual content — exact as provided by parser, never paraphrased
    text: str | None = None

    # Location metadata (use only when reliably available — never invent)
    page: int | None = None         # 1-based page number (PDFs)
    slide: int | None = None        # 1-based slide number (PPTX)
    sheet: str | None = None        # Sheet name (XLSX)

    # Spatial metadata
    bbox: list[float] | None = None
    provenance: list[dict] = field(default_factory=list)

    # Hyperlinks — attached to this element's text, no separate dir
    links: list[LinkAnnotation] = field(default_factory=list)

    # Rich content
    image: Any | None = None        # Raw image bytes/object from parser
    table_data: Any | None = None   # Structured table representation

    # Correction 4: generic field for arbitrary JSON hierarchy
    # Used by the JSON parser to preserve the full parsed object canonically
    # without abusing text or table_data.
    # text is set to json.dumps(obj) for searchability; this holds the real tree.
    structured_data: dict | None = None

    # Reading-order chain (set after full element list is built)
    previous_element: str | None = None
    next_element: str | None = None

    # Back-reference to the artifact file that stores this element
    artifact_ref: str | None = None


# ---------------------------------------------------------------------------
# Full document
# ---------------------------------------------------------------------------

@dataclass
class NormalizedDocument:
    """Complete parsed representation of one source document.

    This is the contract between parsers and the artifact store + chunker.
    Docling internals must not leak past this boundary.
    """
    doc_id: str
    source_path: str
    original_name: str
    extension: str
    mime_type: str | None
    sha256: str
    file_size_bytes: int

    parser_name: str
    parser_version: str | None

    elements: list[NormalizedElement]
    page_count: int | None = None

    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Search / Chroma layer
# ---------------------------------------------------------------------------

@dataclass
class SearchRecord:
    """One record to be upserted into Chroma.

    id must be globally unique: doc_<id>:chunk:NNNNNN
    text is what gets embedded and semantically searched.
    metadata carries scalar provenance fields.
    """
    id: str
    text: str
    metadata: dict


@dataclass
class RagResult:
    """Application-facing RAG result — no raw Chroma blobs, no embeddings."""
    record_id: str
    origin: str                     # "source" or "derived"
    record_type: str                # text_chunk | table | code | derived_image

    text: str
    distance: float

    doc_id: str
    page: int | None = None

    source_element_ids: list[str] = field(default_factory=list)
    image_refs: list[str] = field(default_factory=list)
    table_refs: list[str] = field(default_factory=list)
    code_refs: list[str] = field(default_factory=list)

    derived_cache_ref: str | None = None

    @property
    def raw_distance(self) -> float:
        return float(self.distance)

    @property
    def derived_cosine_similarity(self) -> float:
        """Deterministic conversion: for cosine distance d in [0, 2], cosine similarity = 1 - d without clipping."""
        return round(1.0 - float(self.distance), 6)

    @property
    def derived_similarity(self) -> float:
        """Bounded similarity in [0, 1] preserved for backward compatibility."""
        return round(max(0.0, min(1.0, 1.0 - float(self.distance))), 4)

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "origin": self.origin,
            "record_type": self.record_type,
            "text": self.text,
            "distance": float(self.distance),
            "raw_distance": self.raw_distance,
            "derived_cosine_similarity": self.derived_cosine_similarity,
            "derived_similarity": self.derived_similarity,
            "doc_id": self.doc_id,
            "page": self.page,
            "source_element_ids": list(self.source_element_ids),
            "image_refs": list(self.image_refs),
            "table_refs": list(self.table_refs),
            "code_refs": list(self.code_refs),
            "derived_cache_ref": self.derived_cache_ref,
        }

    def to_socket(self) -> dict:
        return self.to_dict()


@dataclass
class ExactSearchResult:
    """One literal/regex match in canonical artifact files."""
    doc_id: str
    element_id: str
    element_type: str

    page: int | None
    order: int | None
    ref: str | None             # Relative path to artifact file

    match_start: int            # Char offset in matched text
    match_end: int
    snippet: str                # Context window around match
    matched_text: str           # The exact matched substring

    # Table-specific (populated when element_type == "table")
    table_row: int | None = None
    table_col: int | None = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "element_id": self.element_id,
            "element_type": self.element_type,
            "page": self.page,
            "order": self.order,
            "ref": self.ref,
            "match_start": self.match_start,
            "match_end": self.match_end,
            "snippet": self.snippet,
            "matched_text": self.matched_text,
            "table_row": self.table_row,
            "table_col": self.table_col,
        }

    def to_socket(self) -> dict:
        return self.to_dict()


@dataclass
class IndexResult:
    """Returned by ChromaStore.index_document()."""
    doc_id: str
    source_records: int
    embedded_records: int

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "source_records": self.source_records,
            "embedded_records": self.embedded_records,
        }

    def to_socket(self) -> dict:
        return {
            "status": "success",
            "operation": "chroma_upsert",
            "identity": {"doc_id": self.doc_id},
            "result": self.to_dict(),
        }


@dataclass
class DerivedInsertResult:
    """Returned by add_image_analysis()."""
    doc_id: str
    image_id: str
    analysis_id: str
    derived_file: str           # Relative path to derived/vision/img_x.json
    chroma_record_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "image_id": self.image_id,
            "chroma_record_id": self.chroma_record_id,
            "analysis_id": self.analysis_id,
            "derived_file": self.derived_file,
        }

    def to_socket(self) -> dict:
        return {
            "status": "success",
            "operation": "derived_image_analysis_persistence",
            "identity": {
                "doc_id": self.doc_id,
                "image_id": self.image_id,
                "analysis_id": self.analysis_id,
            },
            "storage": {
                "derived_file": self.derived_file,
            },
            "indexing": {
                "chroma_record_id": self.chroma_record_id,
            },
            "result": self.to_dict(),
        }
