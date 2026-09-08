"""
SAGE/tools/document_database.py
Tool adapters wrapping the SAGE Document Database.

These wrappers expose DB capabilities through the generic dispatcher
without leaking Chroma internals or filesystem paths to the agent.

Architectural rules:
    - Accept plain serializable arguments
    - Call SageDocumentDB
    - Convert dataclasses/objects into JSON-safe dicts
    - Return bounded output
    - Preserve: doc_id, element_id, page/slide/sheet, text/snippet,
      distance/score, image_refs
    - Never expose unsafe arbitrary filesystem operations
    - Never autonomously call another registered Sage tool

Note: internal path resolution (e.g. for vision adapter) IS allowed
because it is deterministic plumbing, not an agentic decision.
The path stays backend-internal — Gemma never constructs it.
"""

from __future__ import annotations
import dataclasses
from typing import Any


def _dataclass_to_dict(obj: Any) -> dict:
    """Convert a dataclass or object with __dict__ to a JSON-safe dict, stripping None values.
    
    If the object defines a custom to_dict() method (e.g. RagResult with computed
    similarity properties), calls it first to preserve derived/property fields.
    """
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        d = obj.to_dict()
        return {k: v for k, v in d.items() if v is not None}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = dataclasses.asdict(obj)
        return {k: v for k, v in d.items() if v is not None}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if v is not None and not k.startswith("_")}
    return {"value": str(obj)}



# ─── Semantic RAG Search ──────────────────────────────────────────────────────

def tool_rag_search(
    db,
    query: str,
    top_k: int = 5,
    doc_ids: list[str] | None = None,
) -> dict:
    """Semantic search over indexed documents.

    Args:
        db:      SageDocumentDB instance (injected at registration)
        query:   Natural language search query
        top_k:   Maximum results to return
        doc_ids: Optional list of doc_ids to restrict search

    Returns:
        {"status": "success", "results": [...], "count": int}
    """
    try:
        results = db.rag_search(
            query=query,
            top_k=top_k,
            doc_ids=doc_ids,
        )
        serialized = []
        for r in results:
            entry = _dataclass_to_dict(r)
            # Remove any accidental filesystem paths
            entry.pop("local_path", None)
            serialized.append(entry)

        return {
            "status": "success",
            "results": serialized,
            "count": len(serialized),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "results": [],
            "count": 0,
        }


# ─── Exact / Literal Search ──────────────────────────────────────────────────

def tool_exact_search(
    db,
    query: str,
    doc_ids: list[str] | None = None,
    case_sensitive: bool = False,
    regex: bool = False,
    max_results: int = 20,
) -> dict:
    """Literal or regex search over canonical artifact files.

    Args:
        db:             SageDocumentDB instance (injected)
        query:          Literal string or regex pattern
        doc_ids:        Optional doc_id filter
        case_sensitive: Whether matching is case-sensitive
        regex:          Whether query is a regex pattern
        max_results:    Maximum results to return

    Returns:
        {"status": "success", "results": [...], "count": int}
    """
    try:
        results = db.exact_search(
            query=query,
            doc_ids=doc_ids,
            case_sensitive=case_sensitive,
            regex=regex,
            max_results=max_results,
        )
        serialized = []
        for r in results:
            entry = _dataclass_to_dict(r)
            entry.pop("local_path", None)
            serialized.append(entry)

        return {
            "status": "success",
            "results": serialized,
            "count": len(serialized),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "results": [],
            "count": 0,
        }


# ─── Direct Artifact Fetch ───────────────────────────────────────────────────

def tool_artifact_fetch(
    db,
    element_id: str,
    doc_id: str | None = None,
    **kwargs,
) -> dict:
    """Fetch the exact canonical element by known ID.

    Args:
        db:         SageDocumentDB instance (injected)
        element_id: Element identifier (e.g. txt_000001, img_000003, or doc_id/img_000003)
        doc_id:     Optional document identifier (auto-discovered if omitted)

    Returns:
        {"status": "success", "element": {...}}
        or
        {"status": "error", "error": "..."}
    """
    try:
        # Handle composite "doc_id/element_id" or "doc_id:element_id"
        if element_id and "/" in str(element_id):
            doc_id, element_id = str(element_id).split("/", 1)
        elif element_id and ":" in str(element_id) and not str(element_id).startswith("http"):
            doc_id, element_id = str(element_id).split(":", 1)

        # Auto-discover doc_id from artifact store if missing
        if not doc_id:
            all_doc_ids = db.list_doc_ids() if hasattr(db, "list_doc_ids") else [p.name for p in db._store.root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
            for p_name in all_doc_ids:
                    try:
                        m = db._store.load_manifest(p_name)
                        if element_id in m.get("element_index", {}):
                            doc_id = p_name
                            break
                    except Exception:
                        pass

        if not doc_id:
            return {"status": "error", "error": f"Element '{element_id}' could not be located in any indexed document"}

        element = db.artifact_fetch(doc_id, element_id)
        # Strip internal filesystem paths before returning to agent
        if isinstance(element, dict):
            element.pop("local_path", None)
            element["doc_id"] = doc_id
        return {
            "status": "success",
            "element": element,
        }
    except (KeyError, FileNotFoundError) as exc:
        return {
            "status": "error",
            "error": str(exc),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Unexpected error: {exc}",
        }


# ─── Artifact Listing ────────────────────────────────────────────────────────

def tool_list_artifacts(
    db,
    doc_id: str | None = None,
    artifact_type: str | None = None,
    type: str | None = None,
    **kwargs,
) -> dict:
    """Enumerate artifacts in a document or across all documents.

    Args:
        db:            SageDocumentDB instance (injected)
        doc_id:        Optional document identifier (lists across all docs if omitted)
        artifact_type: Optional filter: "text", "image", "table", "code"
        type:          Alias for artifact_type matching tools.json schema

    Returns:
        {"status": "success", "doc_id": str, "artifacts": [...], "count": int}

    Rules:
        - Derives from canonical manifest — does not infer nonexistent artifacts
        - Does not invoke vision or load image pixels
        - Deterministic ordering by reading order
        - Does not dump entire document content
    """
    effective_type = artifact_type or type
    try:
        if not doc_id:
            all_docs = db.list_doc_ids() if hasattr(db, "list_doc_ids") else [p.name for p in db._store.root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
            if not all_docs:
                return {"status": "error", "error": "No documents found in store", "artifacts": [], "count": 0}
            all_artifacts = []
            for d in all_docs:
                d_arts = db.list_artifacts(d, artifact_type=effective_type)
                for a in d_arts:
                    a["doc_id"] = d
                all_artifacts.extend(d_arts)
            return {
                "status": "success",
                "doc_id": "all",
                "artifacts": all_artifacts,
                "count": len(all_artifacts),
            }

        artifacts = db.list_artifacts(doc_id, artifact_type=effective_type)
        for a in artifacts:
            a["doc_id"] = doc_id
        return {
            "status": "success",
            "doc_id": doc_id,
            "artifacts": artifacts,
            "count": len(artifacts),
        }
    except FileNotFoundError as exc:
        return {
            "status": "error",
            "error": str(exc),
            "doc_id": doc_id,
            "artifacts": [],
            "count": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Unexpected error: {exc}",
            "doc_id": doc_id,
            "artifacts": [],
            "count": 0,
        }


# ─── Registration Helper ─────────────────────────────────────────────────────

def register_document_db_tools(registry, db) -> None:
    """Register all document database tools with a ToolRegistry.

    Canonical registrations (match tools.json, Gemma-visible):
        document_database / rag_search
        document_database / exact_search
        document_database / artifact_fetch
        document_database / list_artifacts

    Internal backward-compat aliases (NOT exposed to Gemma):
        document_db / rag_search  (legacy orchestrator/dispatcher calls)
        document_db / exact_search
        document_db / artifact_fetch
        document_db / list_artifacts
    """
    from functools import partial

    fns = {
        "rag_search":    partial(tool_rag_search, db),
        "exact_search":  partial(tool_exact_search, db),
        "artifact_fetch": partial(tool_artifact_fetch, db),
        "list_artifacts": partial(tool_list_artifacts, db),
    }

    for fn_name, fn in fns.items():
        # Canonical name (Gemma-facing)
        registry.register("document_database", fn_name, fn)
        # Internal backward-compat alias
        registry.register("document_db", fn_name, fn)
