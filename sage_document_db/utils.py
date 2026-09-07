"""
sage_document_db/utils.py
Shared utility helpers for the SAGE document database layer.
"""

from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Document identity
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path) -> str:
    """Return full lowercase SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def make_doc_id(sha256: str) -> str:
    """Return doc_<first 10 hex chars of sha256>.

    Same exact file always produces the same doc_id.
    """
    return f"doc_{sha256[:10]}"


# ---------------------------------------------------------------------------
# Chroma metadata helpers
# (Chroma metadata values must be scalar — lists are stored as JSON strings)
# ---------------------------------------------------------------------------

def encode_list_field(lst: list) -> str:
    """Encode a list to a JSON string for Chroma metadata storage."""
    return json.dumps(lst)


def decode_list_field(s: str | None) -> list:
    """Decode a JSON string back to a list from Chroma metadata."""
    if not s:
        return []
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []


# ---------------------------------------------------------------------------
# File system helpers
# ---------------------------------------------------------------------------

def safe_mkdir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

def iso_now() -> str:
    """Current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Snippet helpers
# ---------------------------------------------------------------------------

def make_snippet(text: str, start: int, end: int, context: int = 60) -> str:
    """Return a context window around [start, end) in text."""
    snip_start = max(0, start - context)
    snip_end = min(len(text), end + context)
    snippet = text[snip_start:snip_end]
    if snip_start > 0:
        snippet = "..." + snippet
    if snip_end < len(text):
        snippet = snippet + "..."
    return snippet
