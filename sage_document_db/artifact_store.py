"""
sage_document_db/artifact_store.py
Canonical artifact store — the source-of-truth layer.

ArtifactStore: writes and reads canonical artifacts.
ArtifactReader: reconstructs NormalizedDocument from disk (for reindex).

Rules:
- Artifacts are canonical source truth; Chroma is disposable.
- Never overwrite canonical source metadata with LLM/VLM conclusions.
- No empty folders.
- No attachments/ dir, no links/ dir, no page renders.
- Do not copy the original source file.

Correction 1: fetch_element() opens multi-element files and selects
              by element ID, never returning the whole file.

Correction 3: ArtifactReader.load_normalized_document() reconstructs
              a full NormalizedDocument from manifest + artifact files,
              using the same per-element selection logic as fetch_element.
"""

from __future__ import annotations
import dataclasses
import json
import re
import shutil
from pathlib import Path
from typing import Any, Generator

_SAFE_DOC_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_SAFE_ELEMENT_ID_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")

from .config import ARTIFACTS_ROOT
from .models import (
    LinkAnnotation,
    NormalizedDocument,
    NormalizedElement,
)
from .utils import iso_now, safe_mkdir


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _element_to_dict(el: NormalizedElement) -> dict:
    """Serialize a NormalizedElement to a JSON-safe dict."""
    d: dict = {
        "id": el.id,
        "type": el.type,
        "subtype": el.subtype,
        "order": el.order,
        "text": el.text,
        "page": el.page,
        "slide": el.slide,
        "sheet": el.sheet,
        "bbox": el.bbox,
        "provenance": el.provenance,
        "links": [dataclasses.asdict(lk) for lk in el.links],
        "previous_element": el.previous_element,
        "next_element": el.next_element,
        "artifact_ref": el.artifact_ref,
    }
    # structured_data is serialized only if present
    if el.structured_data is not None:
        d["structured_data"] = el.structured_data
    return d


def _dict_to_element(d: dict) -> NormalizedElement:
    """Reconstruct a NormalizedElement from a stored dict."""
    links = [
        LinkAnnotation(
            text=lk.get("text"),
            url=lk["url"],
            start_char=lk.get("start_char"),
            end_char=lk.get("end_char"),
        )
        for lk in d.get("links", [])
    ]
    return NormalizedElement(
        id=d["id"],
        type=d["type"],
        subtype=d.get("subtype"),
        order=d.get("order", 0),
        text=d.get("text"),
        page=d.get("page"),
        slide=d.get("slide"),
        sheet=d.get("sheet"),
        bbox=d.get("bbox"),
        provenance=d.get("provenance", []),
        links=links,
        structured_data=d.get("structured_data"),
        previous_element=d.get("previous_element"),
        next_element=d.get("next_element"),
        artifact_ref=d.get("artifact_ref"),
    )


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------

class ArtifactStore:
    """Writes and reads canonical document artifacts.

    The artifact directory is the permanent source of truth.
    Chroma is derived from this; it can be deleted and rebuilt.
    """

    def __init__(self, artifacts_root: str | Path = ARTIFACTS_ROOT) -> None:
        self.root = Path(artifacts_root)
        safe_mkdir(self.root)

    def doc_dir(self, doc_id: str) -> Path:
        if not doc_id or not isinstance(doc_id, str) or ".." in doc_id or not _SAFE_DOC_ID_RE.match(doc_id):
            raise ValueError(f"Invalid or unsafe doc_id: {doc_id!r}")
        path = (self.root / doc_id).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError(f"Path traversal detected in doc_id: {doc_id!r}")
        return path

    def document_exists(self, doc_id: str) -> bool:
        """Return True if the artifact directory and manifest exist."""
        return (self.doc_dir(doc_id) / "manifest.json").exists()

    def load_manifest(self, doc_id: str) -> dict:
        """Load and return the manifest.json for a document."""
        manifest_path = self.doc_dir(doc_id) / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for doc_id='{doc_id}'")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_doc_ids(self) -> list[str]:
        """List all valid document IDs that have an artifact directory and manifest."""
        if not self.root.exists():
            return []
        return sorted([
            p.name for p in self.root.iterdir()
            if p.is_dir() and (p / "manifest.json").exists()
        ])

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_document(self, doc: NormalizedDocument) -> Path:
        """Write all canonical artifacts for a NormalizedDocument.

        Returns the artifact directory Path.

        Layout:
            artifacts/doc_<id>/
            ├── manifest.json
            ├── text/          (page_NNNN.json or document.json)
            ├── images/        (img_NNNNNN.png / .jpg / etc.)
            ├── tables/        (table_NNNNNN.json)
            └── code/          (code_NNNNNN.txt)

        derived/ is NOT created here; it is created by derived.py.
        """
        doc_dir = self.doc_dir(doc.doc_id)
        safe_mkdir(doc_dir)

        # Group elements by type
        text_elements = [el for el in doc.elements if el.type in ("text", "code") or
                         (el.type not in ("image", "table"))]
        # Correct grouping
        txt_els = [el for el in doc.elements if el.type == "text"]
        img_els = [el for el in doc.elements if el.type == "image"]
        tbl_els = [el for el in doc.elements if el.type == "table"]
        code_els = [el for el in doc.elements if el.type == "code"]

        # --- Write text elements ---
        # PDF → group by page; others → document.json
        is_paginated = (
            doc.extension == ".pdf" or
            (doc.extension in (".pptx",) and any(el.slide is not None for el in txt_els)) or
            (doc.extension in (".xlsx",) and any(el.sheet is not None for el in txt_els))
        )

        text_dir = doc_dir / "text"
        element_index: dict[str, dict] = {}
        reading_order: list[str] = [el.id for el in doc.elements]

        if txt_els or code_els:
            # Include code in text dir for searchability
            searchable_text_els = txt_els + code_els
            searchable_text_els.sort(key=lambda e: e.order)

            if is_paginated:
                self._write_paginated_text(
                    searchable_text_els, text_dir, doc.extension, element_index
                )
            else:
                self._write_document_text(searchable_text_els, text_dir, element_index)

        # --- Write image elements ---
        if img_els:
            img_dir = doc_dir / "images"
            safe_mkdir(img_dir)
            for el in img_els:
                ext = doc.extension if doc.extension in (".png", ".jpg", ".jpeg", ".webp") else ".png"
                img_filename = f"{el.id}{ext}"
                img_path = img_dir / img_filename
                rel_ref = f"images/{img_filename}"

                if doc.extension in (".png", ".jpg", ".jpeg", ".webp"):
                    # Standalone image: copy source file
                    shutil.copy2(doc.source_path, img_path)
                elif el.image is not None:
                    # Docling-extracted image bytes
                    ext = ".png"
                    img_filename = f"{el.id}.png"
                    img_path = img_dir / img_filename
                    rel_ref = f"images/{img_filename}"
                    img_path.write_bytes(el.image)
                else:
                    rel_ref = None  # No image file; metadata only

                el.artifact_ref = rel_ref
                element_index[el.id] = {
                    "type": el.type,
                    "subtype": el.subtype,
                    "order": el.order,
                    "page": el.page,
                    "ref": rel_ref,
                    "bbox": el.bbox,
                    "link_count": len(el.links) if el.links else 0,
                    "caption": (el.structured_data or {}).get("caption") if el.structured_data else None,
                }

        # --- Write table elements ---
        if tbl_els:
            tbl_dir = doc_dir / "tables"
            safe_mkdir(tbl_dir)
            for el in tbl_els:
                tbl_filename = f"{el.id}.json"
                tbl_path = tbl_dir / tbl_filename
                rel_ref = f"tables/{tbl_filename}"
                rows = (el.table_data or {}).get("rows", [])
                num_rows = len(rows)
                num_cols = len(rows[0]) if rows and isinstance(rows[0], list) else 0
                tbl_data = {
                    "id": el.id,
                    "type": el.type,
                    "subtype": el.subtype,
                    "order": el.order,
                    "page": el.page,
                    "rows": rows,
                    "dimensions": [num_rows, num_cols],
                    "headers": [str(c) for c in rows[0]] if rows and isinstance(rows[0], list) else [],
                    "markdown": (el.table_data or {}).get("markdown"),
                }
                with open(tbl_path, "w", encoding="utf-8") as f:
                    json.dump(tbl_data, f, indent=2, ensure_ascii=False)
                el.artifact_ref = rel_ref
                element_index[el.id] = {
                    "type": el.type,
                    "subtype": el.subtype,
                    "order": el.order,
                    "page": el.page,
                    "ref": rel_ref,
                    "bbox": el.bbox,
                    "link_count": len(el.links) if el.links else 0,
                    "caption": (el.structured_data or {}).get("caption") if el.structured_data else None,
                }

        # --- Write manifest ---
        manifest = {
            "schema_version": "1.0",
            "doc_id": doc.doc_id,
            "origin": {
                "original_name": doc.original_name,
                "source_path": doc.source_path,
                "extension": doc.extension,
                "mime_type": doc.mime_type,
                "sha256": doc.sha256,
                "file_size_bytes": doc.file_size_bytes,
            },
            "parser": {
                "name": doc.parser_name,
                "version": doc.parser_version,
                "parsed_at": iso_now(),
            },
            "document": {
                "page_count": doc.page_count,
            },
            "reading_order": reading_order,
            "element_index": element_index,
            "warnings": doc.warnings,
        }
        with open(doc_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return doc_dir

    def _write_paginated_text(
        self,
        elements: list[NormalizedElement],
        text_dir: Path,
        extension: str,
        element_index: dict,
    ) -> None:
        """Write text elements grouped by page/slide/sheet."""
        safe_mkdir(text_dir)

        # Group by (page, slide, sheet)
        pages: dict[str, list[NormalizedElement]] = {}
        for el in elements:
            if extension == ".pptx":
                key = f"slide_{(el.slide or 1):04d}"
            elif extension == ".xlsx":
                key = f"sheet_{el.sheet or 'unknown'}"
            else:
                key = f"page_{(el.page or 1):04d}"
            pages.setdefault(key, []).append(el)

        for page_key, page_els in pages.items():
            page_els.sort(key=lambda e: e.order)
            file_path = text_dir / f"{page_key}.json"
            rel_ref = f"text/{page_key}.json"

            out_elements = []
            for el in page_els:
                d = _element_to_dict(el)
                el.artifact_ref = rel_ref
                d["artifact_ref"] = rel_ref
                out_elements.append(d)
                element_index[el.id] = {
                    "type": el.type,
                    "subtype": el.subtype,
                    "order": el.order,
                    "page": el.page,
                    "slide": el.slide,
                    "sheet": el.sheet,
                    "ref": rel_ref,
                }

            page_data = {
                "page": page_key,
                "elements": out_elements,
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(page_data, f, indent=2, ensure_ascii=False)

    def _write_document_text(
        self,
        elements: list[NormalizedElement],
        text_dir: Path,
        element_index: dict,
    ) -> None:
        """Write text elements to a single document.json."""
        safe_mkdir(text_dir)
        file_path = text_dir / "document.json"
        rel_ref = "text/document.json"

        out_elements = []
        for el in elements:
            d = _element_to_dict(el)
            el.artifact_ref = rel_ref
            d["artifact_ref"] = rel_ref
            out_elements.append(d)
            element_index[el.id] = {
                "type": el.type,
                "subtype": el.subtype,
                "order": el.order,
                "page": el.page,
                "ref": rel_ref,
            }

        doc_data = {"elements": out_elements}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(doc_data, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Fetch (Correction 1)
    # ------------------------------------------------------------------

    def fetch_element(self, doc_id: str, element_id: str) -> dict:
        """Fetch the exact canonical element dict for a given element_id.

        Correction 1:
        manifest.element_index[element_id].ref may point to a file that
        contains MULTIPLE elements (e.g. text/page_0001.json holds all
        text on that page). This method opens the file and selects ONLY
        the element whose id matches element_id.

        Never returns the whole file contents.

        For image elements: returns metadata + resolved local_path.
        For code elements: returns metadata + exact code text.
        For table elements: returns the full canonical table dict.
        """
        if not element_id or not isinstance(element_id, str) or ".." in element_id or not _SAFE_ELEMENT_ID_RE.match(element_id):
            raise ValueError(f"Invalid or unsafe element_id: {element_id!r}")

        manifest = self.load_manifest(doc_id)
        index = manifest.get("element_index", {})

        if element_id not in index:
            raise KeyError(
                f"Element '{element_id}' not found in manifest for doc_id='{doc_id}'"
            )

        entry = index[element_id]
        el_type = entry.get("type", "text")
        ref = entry.get("ref")

        if el_type == "image":
            return self._fetch_image_element(doc_id, element_id, entry, ref)

        if ref is None:
            raise FileNotFoundError(
                f"Element '{element_id}' has no artifact ref in doc_id='{doc_id}'"
            )

        doc_directory = self.doc_dir(doc_id)
        artifact_path = (doc_directory / ref).resolve()
        if not artifact_path.is_relative_to(doc_directory.resolve()):
            raise ValueError(f"Path traversal detected in artifact ref: {ref!r}")

        if el_type == "table":
            return self._fetch_table_element(doc_id, element_id, artifact_path)

        # text or code: open multi-element file and select by ID
        return self._fetch_text_or_code_element(doc_id, element_id, artifact_path)

    def _fetch_image_element(
        self, doc_id: str, element_id: str, entry: dict, ref: str | None
    ) -> dict:
        result: dict = {
            "id": element_id,
            "type": "image",
            "subtype": entry.get("subtype"),
            "order": entry.get("order"),
            "page": entry.get("page"),
            "ref": ref,
        }
        doc_directory = self.doc_dir(doc_id)
        if ref:
            local_path = (doc_directory / ref).resolve()
            if not local_path.is_relative_to(doc_directory.resolve()):
                raise ValueError(f"Path traversal detected in image ref: {ref!r}")
            result["local_path"] = str(local_path)
            result["exists"] = local_path.exists()
        else:
            result["local_path"] = None
            result["exists"] = False

        derived_cache = (doc_directory / "derived" / "vision" / f"{element_id}.json").resolve()
        if not derived_cache.is_relative_to(doc_directory.resolve()):
            raise ValueError(f"Path traversal detected in derived cache path for: {element_id!r}")
        if derived_cache.exists():
            result["derived_available"] = True
            try:
                with open(derived_cache, "r", encoding="utf-8") as df:
                    ddata = json.load(df)
                result["derived_analysis"] = ddata.get("current", {}).get("description")
            except Exception:
                pass
        return result

    def _fetch_table_element(
        self, doc_id: str, element_id: str, artifact_path: Path
    ) -> dict:
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Table artifact not found: '{artifact_path}'"
            )
        with open(artifact_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _fetch_text_or_code_element(
        self, doc_id: str, element_id: str, artifact_path: Path
    ) -> dict:
        """Open a (possibly multi-element) artifact file and return only
        the element dict whose 'id' matches element_id."""
        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Artifact file not found: '{artifact_path}'"
            )

        # Code elements are stored as plain text files
        if artifact_path.suffix == ".txt":
            code_text = artifact_path.read_text(encoding="utf-8")
            return {
                "id": element_id,
                "type": "code",
                "text": code_text,
                "ref": str(artifact_path),
            }

        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # data may have {"elements": [...]} or be a bare list
        candidates: list[dict] = []
        if isinstance(data, dict) and "elements" in data:
            candidates = data["elements"]
        elif isinstance(data, list):
            candidates = data
        else:
            # Single-element file (e.g. table, or bare element dict)
            if isinstance(data, dict) and data.get("id") == element_id:
                return data
            candidates = [data]

        for el_dict in candidates:
            if isinstance(el_dict, dict) and el_dict.get("id") == element_id:
                return el_dict

        raise KeyError(
            f"Element '{element_id}' not found in artifact file '{artifact_path}'"
        )

    # ------------------------------------------------------------------
    # Artifact listing / enumeration
    # ------------------------------------------------------------------

    def list_elements(
        self, doc_id: str, artifact_type: str | None = None
    ) -> list[dict]:
        """Enumerate artifacts in a document by reading the manifest.

        Args:
            doc_id:        The document identifier.
            artifact_type: Optional filter — "text", "image", "table", "code".
                           None returns all types.

        Returns:
            List of metadata dicts sorted by reading order. Each dict:
            {
                "element_id": str,
                "type": str,
                "subtype": str | None,
                "order": int,
                "page": int | None,
                "slide": int | None,
                "sheet": str | None,
            }

        Rules:
            - Derived from canonical artifact manifest — no inference.
            - Does not load image pixels or full document text.
            - Deterministic ordering by 'order' field.
            - Does not invoke vision or any other tool.
        """
        manifest = self.load_manifest(doc_id)
        index = manifest.get("element_index", {})

        elements = []
        for el_id, entry in index.items():
            if artifact_type and entry.get("type") != artifact_type:
                continue
            ref = entry.get("ref")
            local_exists = (self.doc_dir(doc_id) / ref).exists() if ref else False
            derived_available = False
            if entry.get("type") == "image":
                derived_cache = self.doc_dir(doc_id) / "derived" / "vision" / f"{el_id}.json"
                derived_available = derived_cache.exists()

            elements.append({
                "element_id": el_id,
                "type": entry.get("type"),
                "subtype": entry.get("subtype"),
                "order": entry.get("order", 0),
                "page": entry.get("page"),
                "slide": entry.get("slide"),
                "sheet": entry.get("sheet"),
                "bbox": entry.get("bbox"),
                "heading": entry.get("heading"),
                "caption": entry.get("caption"),
                "link_count": entry.get("link_count", 0),
                "ref": ref,
                "exists": local_exists,
                "derived_available": derived_available,
            })

        # Deterministic ordering by reading order
        elements.sort(key=lambda e: e.get("order", 0))
        return elements

    # ------------------------------------------------------------------
    # Rich Output Sockets (Phases D, E, I)
    # ------------------------------------------------------------------

    def fetch_element_socket(self, doc_id: str, element_id: str) -> dict:
        """Fetch canonical element and return complete Component D socket."""
        import time
        from core.sockets import build_artifact_fetch_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            element = self.fetch_element(doc_id, element_id)
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            return build_artifact_fetch_socket(
                doc_id=doc_id,
                element_id=element_id,
                element=element,
                found=True,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="ELEMENT_NOT_FOUND" if isinstance(e, (KeyError, FileNotFoundError)) else "FETCH_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_artifact_fetch_socket(
                doc_id=doc_id,
                element_id=element_id,
                element=None,
                found=False,
                timing=timing,
                error=error_payload,
            )

    def list_elements_socket(
        self, doc_id: str, artifact_type: str | None = None
    ) -> dict:
        """Enumerate artifacts and return complete Component E socket."""
        import time
        from core.sockets import build_list_artifacts_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            elements = self.list_elements(doc_id, artifact_type=artifact_type)
            manifest = self.load_manifest(doc_id)
            index = manifest.get("element_index", {})
            total_in_doc = len(index)

            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            return build_list_artifacts_socket(
                doc_id=doc_id,
                artifacts=elements,
                artifact_type_filter=artifact_type,
                total_elements_in_doc=total_in_doc,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="DOC_NOT_FOUND" if isinstance(e, FileNotFoundError) else "LIST_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_list_artifacts_socket(
                doc_id=doc_id,
                artifacts=[],
                artifact_type_filter=artifact_type,
                total_elements_in_doc=0,
                timing=timing,
                error=error_payload,
            )

    def write_document_socket(self, doc: NormalizedDocument) -> dict:
        """Write canonical artifacts and return complete Component I socket."""
        import time
        from core.sockets import build_artifact_write_socket, build_timing_payload, build_error_payload

        t0 = time.perf_counter()
        try:
            doc_dir = self.write_document(doc)
            files_written = []
            for p in doc_dir.rglob("*"):
                if p.is_file():
                    files_written.append({
                        "path": str(p.relative_to(self.root)).replace("\\", "/"),
                        "size_bytes": p.stat().st_size,
                        "suffix": p.suffix.lower(),
                        "created": True,
                        "overwritten": False,
                        "write_success": True,
                    })
            counts = {
                "text": sum(1 for el in doc.elements if el.type == "text"),
                "images": sum(1 for el in doc.elements if el.type == "image"),
                "tables": sum(1 for el in doc.elements if el.type == "table"),
                "code": sum(1 for el in doc.elements if el.type == "code"),
            }
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            return build_artifact_write_socket(
                doc_id=doc.doc_id,
                artifact_dir=str(doc_dir),
                manifest_path=str(doc_dir / "manifest.json"),
                files_written=files_written,
                element_counts=counts,
                timing=timing,
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t0) * 1000.0
            timing = build_timing_payload(duration_ms=total_ms)
            error_payload = build_error_payload(
                code="ARTIFACT_WRITE_ERROR",
                type_name=type(e).__name__,
                message=str(e),
                exc=e,
            )
            return build_artifact_write_socket(
                doc_id=doc.doc_id,
                artifact_dir=str(self.doc_dir(doc.doc_id)),
                manifest_path=str(self.doc_dir(doc.doc_id) / "manifest.json"),
                files_written=[],
                element_counts={},
                timing=timing,
                error=error_payload,
            )

    # ------------------------------------------------------------------
    # Iteration for exact search
    # ------------------------------------------------------------------

    def iter_searchable_elements(
        self, doc_ids: list[str] | None = None
    ) -> Generator[tuple[str, str, dict], None, None]:
        """Yield (doc_id, element_id, element_dict) for searchable elements.

        Correction 6 (enforced here as well): only canonical subdirs are
        iterated — text/, tables/, code/. The derived/ subtree is NEVER
        traversed here.

        Caller (exact_search.py) applies the same exclusion rule.
        """
        if doc_ids is not None:
            dirs = [self.doc_dir(d) for d in doc_ids if self.doc_dir(d).exists()]
        else:
            dirs = [d for d in self.root.iterdir() if d.is_dir()]

        for doc_dir in dirs:
            doc_id = doc_dir.name
            manifest_path = doc_dir / "manifest.json"
            if not manifest_path.exists():
                continue
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            index = manifest.get("element_index", {})
            for el_id, entry in index.items():
                ref = entry.get("ref")
                el_type = entry.get("type", "text")
                if ref is None or el_type == "image":
                    continue
                artifact_path = doc_dir / ref
                if not artifact_path.exists():
                    continue
                try:
                    el_dict = self._fetch_text_or_code_element(doc_id, el_id, artifact_path)
                    yield doc_id, el_id, el_dict
                except Exception:
                    continue


# ---------------------------------------------------------------------------
# ArtifactReader (Correction 3)
# ---------------------------------------------------------------------------

class ArtifactReader:
    """Reconstructs a complete NormalizedDocument from canonical artifacts.

    Used by all reindex operations so they go through the SAME chunker
    path as the original ingestion.
    """

    def __init__(self, artifacts_root: str | Path = ARTIFACTS_ROOT) -> None:
        self.store = ArtifactStore(artifacts_root)

    def load_normalized_document(self, doc_id: str) -> NormalizedDocument:
        """Reconstruct a NormalizedDocument from manifest + artifact files.

        Reconstruction steps:
        1. Read manifest.json
        2. For each element ID in reading_order, use element_index to find ref
        3. Open the artifact file and select element by ID (same logic as
           fetch_element — never load whole page files wholesale)
        4. Reconstruct NormalizedElement from stored dict
        5. Return NormalizedDocument in reading_order order
        """
        manifest = self.store.load_manifest(doc_id)
        doc_dir = self.store.doc_dir(doc_id)

        origin = manifest["origin"]
        parser = manifest["parser"]
        reading_order: list[str] = manifest.get("reading_order", [])
        index: dict = manifest.get("element_index", {})
        warnings: list[str] = manifest.get("warnings", [])

        elements: list[NormalizedElement] = []

        for el_id in reading_order:
            if el_id not in index:
                warnings.append(f"Element '{el_id}' in reading_order not found in element_index; skipped.")
                continue

            entry = index[el_id]
            el_type = entry.get("type", "text")
            ref = entry.get("ref")

            try:
                if el_type == "image":
                    el = self._reconstruct_image_element(doc_id, el_id, entry, ref, doc_dir)
                elif ref is None:
                    warnings.append(f"Element '{el_id}' has no ref; skipped.")
                    continue
                else:
                    artifact_path = doc_dir / ref
                    el_dict = self.store._fetch_text_or_code_element(doc_id, el_id, artifact_path)
                    el = _dict_to_element(el_dict)

                elements.append(el)
            except Exception as exc:
                warnings.append(f"Could not reconstruct element '{el_id}': {exc}")
                continue

        # Restore reading-order chain (in case it wasn't stored in artifact)
        for i, el in enumerate(elements):
            el.previous_element = elements[i - 1].id if i > 0 else None
            el.next_element = elements[i + 1].id if i + 1 < len(elements) else None

        return NormalizedDocument(
            doc_id=doc_id,
            source_path=origin.get("source_path", ""),
            original_name=origin.get("original_name", ""),
            extension=origin.get("extension", ""),
            mime_type=origin.get("mime_type"),
            sha256=origin.get("sha256", ""),
            file_size_bytes=origin.get("file_size_bytes", 0),
            parser_name=parser.get("name", "unknown"),
            parser_version=parser.get("version"),
            elements=elements,
            page_count=manifest.get("document", {}).get("page_count"),
            warnings=warnings,
        )

    def _reconstruct_image_element(
        self,
        doc_id: str,
        el_id: str,
        entry: dict,
        ref: str | None,
        doc_dir: Path,
    ) -> NormalizedElement:
        return NormalizedElement(
            id=el_id,
            type="image",
            subtype=entry.get("subtype"),
            order=entry.get("order", 0),
            page=entry.get("page"),
            artifact_ref=ref,
        )

    def list_available_doc_ids(self) -> list[str]:
        """Return all doc_ids that have a valid manifest."""
        if not self.store.root.exists():
            return []
        return [
            d.name
            for d in self.store.root.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()
        ]
