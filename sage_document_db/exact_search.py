"""
sage_document_db/exact_search.py
Exact / grep-like search over canonical artifact files.

Rules:
- Pure Python; no shell-out to grep; Windows-portable.
- Searches ONLY canonical source artifacts (text/, tables/, code/).
- Correction 6: derived/ and any future model-cache subdirectories are
  explicitly excluded — never searched.
- Regex mode uses Python re module.
- Invalid regex → clear ValueError.
- Case-insensitive by default.
"""

from __future__ import annotations
import json
import re
from pathlib import Path

from .config import ARTIFACTS_ROOT
from .models import ExactSearchResult
from .utils import make_snippet


# ---------------------------------------------------------------------------
# Canonical source subdirs to search (Correction 6: derived/ excluded)
# ---------------------------------------------------------------------------

_CANONICAL_TEXT_SUBDIRS = {"text", "tables", "code"}
# Note: images/ contains binary files; we search image metadata stored in
# manifest (for alt text / hyperlink URL) but not from files directly.


def exact_search(
    query: str,
    artifacts_root: str | Path = ARTIFACTS_ROOT,
    doc_ids: list[str] | None = None,
    case_sensitive: bool = False,
    regex: bool = False,
    max_results: int = 20,
) -> list[ExactSearchResult]:
    """Search canonical artifacts for literal or regex matches.

    Correction 6: Only text/, tables/, code/ directories are searched.
    The derived/ subtree is explicitly excluded.

    Args:
        query: Search string or regex pattern.
        artifacts_root: Root of the artifacts directory.
        doc_ids: Limit search to these doc_ids. None = search all.
        case_sensitive: If False, search is case-insensitive.
        regex: If True, treat query as a Python regex.
        max_results: Maximum number of results to return.

    Returns:
        List of ExactSearchResult sorted by doc_id then order.

    Raises:
        ValueError: If regex=True and query is an invalid pattern.
    """
    root = Path(artifacts_root)
    results: list[ExactSearchResult] = []

    # Compile pattern
    if regex:
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            pattern = re.compile(query, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{query}': {e}") from e
    else:
        if not case_sensitive:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        else:
            pattern = re.compile(re.escape(query))

    # Select doc dirs
    if doc_ids is not None:
        doc_dirs = [root / d for d in doc_ids if (root / d).is_dir()]
    else:
        if not root.exists():
            return []
        doc_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]

    for doc_dir in doc_dirs:
        if len(results) >= max_results:
            break

        doc_id = doc_dir.name
        manifest_path = doc_dir / "manifest.json"
        if not manifest_path.exists():
            continue

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        index = manifest.get("element_index", {})

        # Search text/ and code/ subdirs
        for subdir_name in ("text", "code"):
            subdir = doc_dir / subdir_name
            if not subdir.exists():
                continue
            for json_file in sorted(subdir.glob("*.json")):
                if len(results) >= max_results:
                    break
                _search_text_file(
                    json_file, doc_id, index, pattern, results, max_results
                )
            # code/ .txt files
            for txt_file in sorted(subdir.glob("*.txt")):
                if len(results) >= max_results:
                    break
                _search_code_txt_file(
                    txt_file, doc_id, index, pattern, results, max_results
                )

        # Search tables/ subdir
        tables_dir = doc_dir / "tables"
        if tables_dir.exists():
            for tbl_file in sorted(tables_dir.glob("*.json")):
                if len(results) >= max_results:
                    break
                _search_table_file(
                    tbl_file, doc_id, pattern, results, max_results
                )

        # Search hyperlinks (from manifest index annotations in text elements)
        # We re-open text files and look in their links fields
        if len(results) < max_results:
            _search_hyperlinks(doc_dir, doc_id, index, pattern, results, max_results)

    return results[:max_results]


# ---------------------------------------------------------------------------
# File-level search functions
# ---------------------------------------------------------------------------

def _search_text_file(
    file_path: Path,
    doc_id: str,
    index: dict,
    pattern: re.Pattern,
    results: list,
    max_results: int,
) -> None:
    """Search a text/page_NNNN.json or text/document.json for matches."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    elements: list[dict] = []
    if isinstance(data, dict) and "elements" in data:
        elements = data["elements"]
    elif isinstance(data, list):
        elements = data

    for el in elements:
        if len(results) >= max_results:
            return
        el_id = el.get("id", "")
        text = el.get("text") or ""
        if not text:
            continue

        idx_entry = index.get(el_id, {})
        ref = idx_entry.get("ref") or f"text/{file_path.name}"

        for match in pattern.finditer(text):
            if len(results) >= max_results:
                return
            results.append(ExactSearchResult(
                doc_id=doc_id,
                element_id=el_id,
                element_type=el.get("type", "text"),
                page=el.get("page"),
                order=el.get("order"),
                ref=ref,
                match_start=match.start(),
                match_end=match.end(),
                snippet=make_snippet(text, match.start(), match.end()),
                matched_text=match.group(0),
            ))


def _search_code_txt_file(
    file_path: Path,
    doc_id: str,
    index: dict,
    pattern: re.Pattern,
    results: list,
    max_results: int,
) -> None:
    """Search a code/*.txt file for matches."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except Exception:
        return

    # Derive element ID from filename (e.g. code_000001.txt → code_000001)
    el_id = file_path.stem
    idx_entry = index.get(el_id, {})
    ref = idx_entry.get("ref") or f"code/{file_path.name}"

    for match in pattern.finditer(text):
        if len(results) >= max_results:
            return
        results.append(ExactSearchResult(
            doc_id=doc_id,
            element_id=el_id,
            element_type="code",
            page=idx_entry.get("page"),
            order=idx_entry.get("order"),
            ref=ref,
            match_start=match.start(),
            match_end=match.end(),
            snippet=make_snippet(text, match.start(), match.end()),
            matched_text=match.group(0),
        ))


def _search_table_file(
    file_path: Path,
    doc_id: str,
    pattern: re.Pattern,
    results: list,
    max_results: int,
) -> None:
    """Search a tables/table_NNNNNN.json for matches in cell text."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    el_id = data.get("id", file_path.stem)
    rows = data.get("rows", [])
    el_order = data.get("order")
    el_page = data.get("page")
    ref = f"tables/{file_path.name}"

    for row_idx, row in enumerate(rows):
        if len(results) >= max_results:
            return
        for col_idx, cell in enumerate(row):
            cell_text = str(cell) if cell is not None else ""
            for match in pattern.finditer(cell_text):
                if len(results) >= max_results:
                    return
                results.append(ExactSearchResult(
                    doc_id=doc_id,
                    element_id=el_id,
                    element_type="table",
                    page=el_page,
                    order=el_order,
                    ref=ref,
                    match_start=match.start(),
                    match_end=match.end(),
                    snippet=make_snippet(cell_text, match.start(), match.end()),
                    matched_text=match.group(0),
                    table_row=row_idx,
                    table_col=col_idx,
                ))


def _search_hyperlinks(
    doc_dir: Path,
    doc_id: str,
    index: dict,
    pattern: re.Pattern,
    results: list,
    max_results: int,
) -> None:
    """Search hyperlink anchor text and URLs from text element files."""
    text_dir = doc_dir / "text"
    if not text_dir.exists():
        return

    for json_file in sorted(text_dir.glob("*.json")):
        if len(results) >= max_results:
            return
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        elements: list[dict] = []
        if isinstance(data, dict) and "elements" in data:
            elements = data["elements"]
        elif isinstance(data, list):
            elements = data

        for el in elements:
            if len(results) >= max_results:
                return
            el_id = el.get("id", "")
            links = el.get("links", [])
            for link in links:
                # Search anchor text
                anchor = link.get("text") or ""
                url = link.get("url") or ""

                for search_text in (anchor, url):
                    if not search_text:
                        continue
                    for match in pattern.finditer(search_text):
                        if len(results) >= max_results:
                            return
                        idx_entry = index.get(el_id, {})
                        results.append(ExactSearchResult(
                            doc_id=doc_id,
                            element_id=el_id,
                            element_type="hyperlink",
                            page=el.get("page"),
                            order=el.get("order"),
                            ref=idx_entry.get("ref") or f"text/{json_file.name}",
                            match_start=match.start(),
                            match_end=match.end(),
                            snippet=make_snippet(search_text, match.start(), match.end()),
                            matched_text=match.group(0),
                        ))
