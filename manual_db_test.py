"""
manual_db_test.py
CLI test harness for the SAGE document database layer.

Subcommands:
  ingest            Ingest a document into the DB
  rag               Semantic RAG search
  exact             Literal / regex search on canonical artifacts
  fetch             Direct artifact element fetch by ID
  add-image-analysis  Insert a fake/real vision model result
  reindex-doc       Rebuild Chroma index for one document from artifacts
  rebuild-all       Rebuild entire Chroma DB from artifacts

Usage examples:
  python manual_db_test.py ingest "samples/test.pdf" --debug
  python manual_db_test.py rag "AI employee with four years experience" --top-k 5
  python manual_db_test.py exact "320000"
  python manual_db_test.py exact "AUTH_.*_V2" --regex
  python manual_db_test.py fetch doc_a81f42c91e txt_000004
  python manual_db_test.py add-image-analysis --doc-id doc_x --image-id img_000001 ...
  python manual_db_test.py add-image-analysis --json samples/derived_test.json
  python manual_db_test.py reindex-doc doc_a81f42c91e
  python manual_db_test.py rebuild-all
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Auto-switch to project venv if chromadb/docling are not installed in current interpreter
_venv_python = Path(__file__).resolve().parent / "sage_venv" / "Scripts" / "python.exe"
if _venv_python.exists() and sys.executable.lower() != str(_venv_python).lower():
    try:
        import chromadb  # type: ignore # noqa: F401
    except ImportError:
        import subprocess
        sys.exit(subprocess.call([str(_venv_python), str(Path(__file__).resolve())] + sys.argv[1:]))


def _get_db():
    """Lazy-import SageDocumentDB to avoid loading heavy deps on import."""
    from sage_document_db import SageDocumentDB
    return SageDocumentDB()


# ---------------------------------------------------------------------------
# Subcommand: ingest
# ---------------------------------------------------------------------------

def cmd_ingest(args) -> None:
    db = _get_db()
    result = db.ingest_document(
        args.source,
        index_in_chroma=not args.no_index,
        overwrite_artifacts=args.overwrite,
        debug=args.debug,
    )

    print(f"\n{'='*60}")
    print(f"  Source        : {args.source}")
    print(f"  Parser        : {result['parser']}")
    print(f"  doc_id        : {result['doc_id']}")
    print(f"  Artifact dir  : {result['artifact_dir']}")
    print(f"  Manifest      : {result['manifest_path']}")
    counts = result.get("counts", {})
    print(f"\n  Element counts:")
    print(f"    Text        : {counts.get('text', 0)}")
    print(f"    Images      : {counts.get('images', 0)}")
    print(f"    Tables      : {counts.get('tables', 0)}")
    print(f"    Code        : {counts.get('code', 0)}")
    print(f"    Links       : {counts.get('links', 0)}")
    idx = result.get("index", {})
    print(f"\n  Chroma source records: {idx.get('source_records', 0)}")
    if "index_error" in result:
        print(f"\n  [INDEX ERROR] {result['index_error']}")
    warnings = result.get("warnings", [])
    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Subcommand: rag
# ---------------------------------------------------------------------------

def cmd_rag(args) -> None:
    db = _get_db()
    doc_ids = args.doc_ids.split(",") if args.doc_ids else None
    results = db.rag_search(
        args.query,
        top_k=args.top_k,
        doc_ids=doc_ids,
        include_source=not args.derived_only,
        include_derived=not args.source_only,
    )

    print(f"\nRAG query: '{args.query}'")
    print(f"Results: {len(results)}\n")

    for rank, r in enumerate(results, 1):
        print(f"--- Rank {rank} ---")
        print(f"  record_id    : {r.record_id}")
        print(f"  origin       : {r.origin}")
        print(f"  record_type  : {r.record_type}")
        print(f"  doc_id       : {r.doc_id}")
        print(f"  distance     : {r.distance:.4f}")
        if r.page is not None:
            print(f"  page         : {r.page}")
        print(f"  source_ids   : {r.source_element_ids}")
        print(f"  image_refs   : {r.image_refs}")
        print(f"  table_refs   : {r.table_refs}")
        print(f"  code_refs    : {r.code_refs}")
        text_preview = r.text[:300].replace("\n", " ")
        print(f"  text         : {text_preview}{'...' if len(r.text) > 300 else ''}")
        print()


# ---------------------------------------------------------------------------
# Subcommand: exact
# ---------------------------------------------------------------------------

def cmd_exact(args) -> None:
    db = _get_db()
    doc_ids = args.doc_ids.split(",") if args.doc_ids else None
    try:
        results = db.exact_search(
            args.query,
            doc_ids=doc_ids,
            case_sensitive=args.case_sensitive,
            regex=args.regex,
            max_results=args.max_results,
        )
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\nExact search: '{args.query}'  "
          f"(regex={args.regex}, case_sensitive={args.case_sensitive})")
    print(f"Results: {len(results)}\n")

    for r in results:
        print(f"  doc_id     : {r.doc_id}")
        print(f"  element_id : {r.element_id}")
        print(f"  type       : {r.element_type}")
        if r.page is not None:
            print(f"  page       : {r.page}")
        if r.table_row is not None:
            print(f"  row/col    : {r.table_row}/{r.table_col}")
        print(f"  ref        : {r.ref}")
        print(f"  match      : [{r.match_start}:{r.match_end}] '{r.matched_text}'")
        print(f"  snippet    : {r.snippet}")
        print()


# ---------------------------------------------------------------------------
# Subcommand: fetch
# ---------------------------------------------------------------------------

def cmd_fetch(args) -> None:
    db = _get_db()
    try:
        result = db.artifact_fetch(args.doc_id, args.element_id)
    except (FileNotFoundError, KeyError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\nFetch: {args.doc_id} / {args.element_id}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print()

    # Convenience: if image, show resolved path
    if result.get("type") == "image":
        local = result.get("local_path")
        exists = result.get("exists", False)
        print(f"  Image path: {local}  (exists={exists})\n")


# ---------------------------------------------------------------------------
# Subcommand: add-image-analysis
# ---------------------------------------------------------------------------

def cmd_add_image_analysis(args) -> None:
    db = _get_db()

    if args.json:
        # Load from JSON file
        json_path = Path(args.json)
        if not json_path.exists():
            print(f"[ERROR] JSON file not found: {args.json}")
            sys.exit(1)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        # Build from CLI args
        if not args.doc_id or not args.image_id or not args.model or \
           not args.task_type or not args.raw_output:
            print("[ERROR] --doc-id, --image-id, --model, --task-type, --raw-output are required "
                  "when not using --json")
            sys.exit(1)
        data = {
            "doc_id": args.doc_id,
            "image_id": args.image_id,
            "model": args.model,
            "task_type": args.task_type,
            "raw_output": args.raw_output,
            "instruction": args.instruction,
            "ocr_text": args.ocr_text,
            "description": args.description,
        }
        if args.observations:
            try:
                data["observations"] = json.loads(args.observations)
            except json.JSONDecodeError:
                print(f"[ERROR] --observations must be valid JSON: {args.observations}")
                sys.exit(1)

    try:
        result = db.add_image_analysis(
            doc_id=data["doc_id"],
            image_id=data["image_id"],
            model=data["model"],
            task_type=data["task_type"],
            raw_output=data["raw_output"],
            instruction=data.get("instruction"),
            ocr_text=data.get("ocr_text"),
            description=data.get("description"),
            observations=data.get("observations"),
            searchable_text=data.get("searchable_text"),
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"\nImage analysis stored:")
    print(f"  doc_id          : {result.doc_id}")
    print(f"  image_id        : {result.image_id}")
    print(f"  analysis_id     : {result.analysis_id}")
    print(f"  chroma_record   : {result.chroma_record_id}")
    print(f"  derived_file    : {result.derived_file}")
    print()
    print("Now test derived retrieval:")
    print(f'  python manual_db_test.py rag "photo with blue background"\n')


# ---------------------------------------------------------------------------
# Subcommand: reindex-doc
# ---------------------------------------------------------------------------

def cmd_reindex_doc(args) -> None:
    db = _get_db()
    print(f"\nReindexing {args.doc_id} from artifacts...")
    result = db.reindex_document(args.doc_id)
    print(f"  Source records indexed: {result['source_records']}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: rebuild-all
# ---------------------------------------------------------------------------

def cmd_rebuild_all(args) -> None:
    db = _get_db()
    print("\nRebuilding all Chroma indexes from artifacts...")
    print("(Does NOT delete artifacts)\n")
    result = db.rebuild_all_indexes()

    src = result["source"]
    print(f"sage_source rebuilt:")
    print(f"  Documents     : {len(src.get('doc_ids', []))}")
    print(f"  Total records : {src.get('total_source_records', 0)}")
    if src.get("errors"):
        print(f"  Errors        : {len(src['errors'])}")
        for e in src["errors"]:
            print(f"    - {e}")

    drv = result["derived"]
    print(f"\nsage_derived rebuilt:")
    print(f"  Derived records: {drv.get('total_derived_records', 0)}")
    if drv.get("errors"):
        print(f"  Errors         : {len(drv['errors'])}")
        for e in drv["errors"]:
            print(f"    - {e}")
    print()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manual_db_test.py",
        description="SAGE Document Database — manual test CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a document")
    p_ingest.add_argument("source", help="Path to source file")
    p_ingest.add_argument("--debug", action="store_true", help="Print element/chunk details")
    p_ingest.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    p_ingest.add_argument("--no-index", action="store_true", help="Skip Chroma indexing")

    # rag
    p_rag = sub.add_parser("rag", help="Semantic RAG search")
    p_rag.add_argument("query", help="Search query")
    p_rag.add_argument("--top-k", type=int, default=5, help="Number of results")
    p_rag.add_argument("--doc-ids", default=None, help="Comma-separated doc_ids to filter")
    p_rag.add_argument("--source-only", action="store_true", help="Only search sage_source")
    p_rag.add_argument("--derived-only", action="store_true", help="Only search sage_derived")

    # exact
    p_exact = sub.add_parser("exact", help="Literal/regex search on canonical artifacts")
    p_exact.add_argument("query", help="Search string or regex pattern")
    p_exact.add_argument("--regex", action="store_true", help="Treat query as regex")
    p_exact.add_argument("--case-sensitive", action="store_true")
    p_exact.add_argument("--max-results", type=int, default=20)
    p_exact.add_argument("--doc-ids", default=None, help="Comma-separated doc_ids to filter")

    # fetch
    p_fetch = sub.add_parser("fetch", help="Direct artifact element fetch")
    p_fetch.add_argument("doc_id", help="Document ID")
    p_fetch.add_argument("element_id", help="Element ID (e.g. txt_000004, img_000001)")

    # add-image-analysis
    p_aia = sub.add_parser("add-image-analysis", help="Insert derived image analysis result")
    p_aia.add_argument("--json", default=None, metavar="FILE",
                       help="Load all fields from a JSON file")
    p_aia.add_argument("--doc-id", default=None)
    p_aia.add_argument("--image-id", default=None)
    p_aia.add_argument("--model", default=None)
    p_aia.add_argument("--task-type", default=None)
    p_aia.add_argument("--raw-output", default=None)
    p_aia.add_argument("--instruction", default=None)
    p_aia.add_argument("--ocr-text", default=None)
    p_aia.add_argument("--description", default=None)
    p_aia.add_argument("--observations", default=None, metavar="JSON",
                       help='JSON string e.g. \'{"background_color": "blue"}\'')

    # reindex-doc
    p_rid = sub.add_parser("reindex-doc", help="Rebuild Chroma index for one document")
    p_rid.add_argument("doc_id", help="Document ID to reindex")

    # rebuild-all
    sub.add_parser("rebuild-all", help="Rebuild entire Chroma DB from artifacts")

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "ingest": cmd_ingest,
        "rag": cmd_rag,
        "exact": cmd_exact,
        "fetch": cmd_fetch,
        "add-image-analysis": cmd_add_image_analysis,
        "reindex-doc": cmd_reindex_doc,
        "rebuild-all": cmd_rebuild_all,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
