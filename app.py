import os
import uuid
import shutil
import time
import logging
import threading
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager

import json
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

import config
from model_manager import model_manager
from core.run_state import RunState, RegisteredDocument

# Clean up older temp request folders on startup
def cleanup_temp_dirs():
    try:
        if config.TEMP_DIR.exists():
            for item in config.TEMP_DIR.iterdir():
                if item.is_dir() and item.name != "logs":
                    shutil.rmtree(item, ignore_errors=True)
    except Exception as e:
        print(f"Warning cleaning temp dirs: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cleanup_temp_dirs()
    print(f"SAGE Agent Orchestrator initialized. Static dir: {config.STATIC_DIR} | Gemma Context: {config.GEMMA_CONTEXT}")
    # The legacy Gemma warm-up is skipped when Flash is the default; otherwise
    # it would consume the VRAM reserved for the resident Gemma + Qwen pair.
    if not config.FLASH_DEFAULT_ENABLED:
        model_manager.rearm_agent_background()
    try:
        from flash.memory_worker import memory_worker
        threading.Thread(target=memory_worker.recover_pending, name="sage-memory-recovery", daemon=True).start()
        from memory_system.index_outbox import index_outbox
        threading.Thread(target=index_outbox.recover, name="sage-index-recovery", daemon=True).start()
    except Exception as exc:
        logger.warning("Memory curator recovery unavailable: %s", exc)
    yield
    # Shutdown
    print("Shutting down SAGE and stopping any running model server...")
    model_manager.stop_current()
    from flash.local_manager import local_flash_manager
    from flash.transport import flash_transport
    local_flash_manager.stop_all()
    flash_transport.close()

app = FastAPI(title="SAGE - Multi-Model Agent Orchestrator", lifespan=lifespan)

# Bump when shipping UI/static changes so HTML references cannot stick on old JS.
STATIC_ASSET_VERSION = "memory-observer-4"


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Prevent stale HTML/JS after UI changes; allow short caching for binary assets."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        path = request.url.path or ""
        if path in {"/", "/memory"} or path.endswith(".html"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        elif path.startswith("/static/"):
            if path.endswith((".js", ".css", ".html", ".map")):
                response.headers["Cache-Control"] = "no-cache, must-revalidate"
            else:
                response.headers["Cache-Control"] = "public, max-age=86400"
        return response


app.add_middleware(CacheControlMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def _save_uploaded_file(file_item: UploadFile, destination: Path, max_bytes: Optional[int] = None) -> int:
    """Streamingly save uploaded bytes to destination with bounded size enforcement."""
    limit = config.MAX_UPLOAD_SIZE_BYTES if max_bytes is None else max_bytes
    total_written = 0
    chunk_size = 64 * 1024
    exceeded = False
    try:
        with open(destination, "wb") as f:
            while True:
                chunk = await file_item.read(chunk_size)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > limit:
                    exceeded = True
                    break
                f.write(chunk)
        if exceeded:
            destination.unlink(missing_ok=True)
            raise HTTPException(
                status_code=413,
                detail=f"File '{file_item.filename}' exceeds maximum allowed upload size of {limit} bytes."
            )
        return total_written
    except HTTPException:
        destination.unlink(missing_ok=True)
        raise
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {exc}") from exc


@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "current_model": model_manager.current_model_key,
        "is_healthy": model_manager.is_healthy(),
        "llama_port": config.LLAMA_PORT,
        "gemma_context": config.GEMMA_CONTEXT,
        "models": {k: v["name"] for k, v in config.MODELS.items()},
        "model_configs": {
            k: {
                "name": v.get("name"),
                "context": v.get("context"),
                "max_tokens": v.get("max_tokens"),
                "temperature": v.get("temperature"),
            }
            for k, v in config.MODELS.items()
        }
    }


@app.get("/api/flash/catalog")
async def flash_catalog():
    """Return editable role/model metadata without exposing runtime secrets."""
    from flash.catalog import public_catalog
    return {"status": "success", "models": public_catalog()}


@app.get("/api/flash/status")
async def flash_status():
    from flash.catalog import public_catalog
    from flash.local_manager import local_flash_manager
    from flash.runtime_config import runtime_config
    mock_mode = os.environ.get("SAGE_MOCK_MODE", "0") == "1"
    llama_ok = bool(config.LLAMA_SERVER_PATH and Path(config.LLAMA_SERVER_PATH).is_file())
    model_dir = Path(config.MODEL_DIR) if config.MODEL_DIR else None
    catalog = public_catalog()
    missing_models = []
    if model_dir and model_dir.is_dir():
        for role, meta in catalog.items():
            file_name = meta.get("file")
            if file_name and not (model_dir / file_name).is_file():
                missing_models.append(file_name)
            mmproj = meta.get("mmproj")
            if mmproj and not (model_dir / mmproj).is_file():
                missing_models.append(mmproj)
    else:
        missing_models = [meta.get("file") for meta in catalog.values() if meta.get("file")]
    return {
        "status": "online",
        "mode": "flash",
        "runtime": runtime_config.public_snapshot(),
        "local_servers": local_flash_manager.status(),
        "models": catalog,
        "readiness": {
            "mock_mode": mock_mode,
            "llama_server_configured": llama_ok,
            "llama_server_path": config.LLAMA_SERVER_PATH or None,
            "model_dir": str(model_dir) if model_dir else None,
            "missing_model_files": missing_models,
            "local_inference_ready": mock_mode or (llama_ok and not missing_models),
        },
    }


def _valid_chat_id(chat_id: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_-]{1,128}", chat_id or ""))


@app.get("/api/chats")
async def list_chats(limit: int = 100):
    """List persisted conversations for the local user."""
    from sage_memory import sage_memory
    chats = sage_memory.list_chats(config.DEFAULT_USER_ID, limit=limit)
    return {"status": "success", "chats": chats}


@app.get("/api/chats/{chat_id}")
async def get_chat(chat_id: str):
    """Reload all messages for one conversation."""
    if not _valid_chat_id(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat id")
    from sage_memory import sage_memory
    messages = sage_memory.get_messages(chat_id)
    return {
        "status": "success",
        "chat_id": chat_id,
        "messages": messages,
    }


@app.delete("/api/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """Delete one conversation from the local SAGE memory ledger."""
    if not _valid_chat_id(chat_id):
        raise HTTPException(status_code=400, detail="Invalid chat id")
    from sage_memory import sage_memory
    from flash.memory_worker import memory_worker
    cleared = sage_memory.clear_chat(chat_id, user_id=config.DEFAULT_USER_ID)
    memory_worker.clear_session(chat_id)
    return {
        "status": "success",
        "chat_id": chat_id,
        **cleared,
    }


@app.post("/api/flash/runtime")
async def configure_flash_runtime(payload: Dict[str, Any] = Body(...)):
    """Apply an ephemeral local/remote role configuration for this server session."""
    from flash.runtime_config import runtime_config
    from flash.local_manager import local_flash_manager
    try:
        snapshot = runtime_config.configure(payload)
        local_flash_manager.reconcile(snapshot)
        from flash.transport import flash_transport
        threading.Thread(
            target=flash_transport.warm_configured_connections,
            name="sage-flash-warmup",
            daemon=True,
        ).start()
        return {"status": "success", "runtime": snapshot}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/flash/runtime")
async def reset_flash_runtime():
    from flash.local_manager import local_flash_manager
    from flash.runtime_config import runtime_config
    snapshot = runtime_config.reset()
    local_flash_manager.reconcile(snapshot)
    return {"status": "success", "runtime": snapshot}


@app.delete("/api/flash/session/{session_id}")
async def clear_flash_session(session_id: str):
    """Clear the local SQLite conversation and leftover Flash JSONL for one session."""
    from flash.memory_worker import memory_worker
    from sage_memory import sage_memory
    if not _valid_chat_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session id")
    cleared = sage_memory.clear_chat(session_id, user_id=config.DEFAULT_USER_ID)
    memory_worker.clear_session(session_id)
    return {
        "status": "success",
        "messages_removed": cleared.get("messages_removed", 0),
        "hot_memories_removed": cleared.get("memories_removed", 0),
    }


@app.post("/api/flash/runtime/test")
async def test_flash_runtime():
    """Test all three bindings after configuration; disabled memory is healthy."""
    from starlette.concurrency import run_in_threadpool
    from flash.transport import flash_transport
    results = []
    for role in ("gemma", "qwen", "memory"):
        results.append(await run_in_threadpool(flash_transport.test_role, role))
    return {"status": "success", "healthy": all(item.get("healthy") for item in results), "roles": results}


@app.post("/api/flash/deploy")
async def deploy_flash_model(payload: Dict[str, Any] = Body(...)):
    """Deploy one fixed catalog role through a compatible remote bridge."""
    from starlette.concurrency import run_in_threadpool
    from flash.transport import flash_transport
    role = str(payload.get("role") or "")
    if role not in {"gemma", "qwen", "memory"}:
        raise HTTPException(status_code=400, detail="role must be gemma, qwen, or memory")
    try:
        result = await run_in_threadpool(
            flash_transport.deploy,
            role,
            str(payload.get("connection_id") or ""),
            int(payload.get("gpu", 0)),
        )
        return {"status": "success", "result": result}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Remote deployment failed: {exc}") from exc


@app.get("/api/observer/runs")
async def observer_runs(limit: int = 30):
    from core.observer import observer
    return {"status": "success", "runs": observer.runs(limit)}


@app.get("/api/observer/runs/{run_id}")
async def observer_run(run_id: str, after: int = 0):
    from core.observer import observer
    return {"status": "success", "run_id": run_id, "events": observer.events(run_id, after)}


@app.get("/api/observer/stream/{run_id}")
async def observer_stream(run_id: str, request: Request, after: int = 0):
    """Live sanitized execution events for the compact and full Observer UI."""
    import asyncio
    from core.observer import observer

    async def generate():
        sequence = max(0, int(after))
        idle = 0
        while not await request.is_disconnected():
            events = observer.events(run_id, sequence)
            if events:
                idle = 0
                for event in events:
                    sequence = max(sequence, int(event["sequence"]))
                    yield f"id: {sequence}\nevent: observation\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            else:
                idle += 1
                if idle % 30 == 0:
                    yield f"event: heartbeat\ndata: {json.dumps({'run_id': run_id, 'sequence': sequence})}\n\n"
            await asyncio.sleep(0.25)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",
    })


@app.get("/api/memory/jobs")
async def memory_jobs(status: Optional[str] = None, limit: int = 100):
    from memory_system.job_store import memory_job_store
    return {"status": "success", "jobs": memory_job_store.list(status=status, limit=limit)}


@app.get("/api/memory/index-outbox")
async def memory_index_outbox(status: Optional[str] = None, limit: int = 100):
    from memory_system.index_outbox import index_outbox
    return {"status": "success", "events": index_outbox.list(status=status, limit=limit)}

@app.post("/api/stop")
async def stop_server():
    model_manager.stop_current()
    return {"status": "stopped"}

@app.get("/api/artifacts/graph")
async def get_artifacts_graph():
    """Return nodes and edges for Obsidian-inspired 3D artifact visualization."""
    try:
        from db_service import document_db
        store = getattr(document_db, "_store", None)
        if not store:
            return {"status": "success", "nodes": [], "edges": [], "summary": {"total_nodes": 0, "total_edges": 0}}

        doc_ids = store.list_doc_ids()
        nodes = []
        doc_meta_map = {}

        for doc_id in doc_ids:
            try:
                manifest = store.load_manifest(doc_id)
                origin = manifest.get("origin", {})
                doc_info = manifest.get("document", {})
                element_index = manifest.get("element_index", {})
                file_name = origin.get("original_name") or doc_id
                ext = (origin.get("extension") or Path(file_name).suffix.lstrip(".")).lower()
                size_bytes = origin.get("file_size_bytes") or 0
                page_count = doc_info.get("page_count") or 1
                total_elements = len(element_index)

                doc_meta_map[doc_id] = {
                    "id": doc_id,
                    "label": file_name,
                    "name": file_name,
                    "file_type": ext,
                    "size_bytes": size_bytes,
                    "metadata": {
                        "page_count": page_count,
                        "total_elements": total_elements,
                        "parser": manifest.get("parser", {}).get("name", "generic"),
                        "created_at": manifest.get("parser", {}).get("parsed_at"),
                    },
                    "degree": 0,
                }
            except Exception as e:
                logger.warning("Failed to read manifest for doc %s: %s", doc_id, e)

        # Build relational edges
        edges = []
        doc_id_list = list(doc_meta_map.keys())

        for i in range(len(doc_id_list)):
            for j in range(i + 1, len(doc_id_list)):
                id1, id2 = doc_id_list[i], doc_id_list[j]
                m1, m2 = doc_meta_map[id1], doc_meta_map[id2]
                if m1["file_type"] == m2["file_type"] or (len(doc_id_list) <= 10 and j == i + 1):
                    edges.append({
                        "source": id1,
                        "target": id2,
                        "relationship": "related_document"
                    })
                    m1["degree"] += 1
                    m2["degree"] += 1

        nodes = list(doc_meta_map.values())

        return {
            "status": "success",
            "nodes": nodes,
            "edges": edges,
            "summary": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "timestamp": time.time(),
            }
        }
    except Exception as exc:
        logger.error("Error generating artifact graph: %s", exc)
        return {
            "status": "error",
            "error": str(exc),
            "nodes": [],
            "edges": [],
            "summary": {"total_nodes": 0, "total_edges": 0}
        }

@app.post("/api/artifacts/upload")
async def upload_artifacts(files: List[UploadFile] = File(...)):
    """Direct artifact file ingestion from the 3D Artifacts view."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    from db_service import document_db
    store = getattr(document_db, "_store", None)
    uploaded = []

    intake_dir = config.TEMP_DIR / f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    intake_dir.mkdir(parents=True, exist_ok=True)

    for file_item in files:
        if not file_item.filename:
            continue
        safe_filename = Path(file_item.filename).name
        temp_dest = intake_dir / safe_filename

        try:
            file_size = await _save_uploaded_file(file_item, temp_dest)
            ingest_res = document_db.ingest_document(str(temp_dest))

            if isinstance(ingest_res, dict) and ingest_res.get("status") == "error":
                uploaded.append({
                    "name": safe_filename,
                    "status": "error",
                    "error": ingest_res.get("error", "Ingestion failed")
                })
                continue

            doc_id = ingest_res.get("doc_id") if isinstance(ingest_res, dict) else None
            if doc_id and store:
                doc_dir = store.doc_dir(doc_id)
                doc_orig_dir = doc_dir / "original"
                doc_orig_dir.mkdir(parents=True, exist_ok=True)
                orig_copy = doc_orig_dir / safe_filename
                if not orig_copy.exists() and temp_dest.exists():
                    shutil.copy2(temp_dest, orig_copy)

                uploaded.append({
                    "name": safe_filename,
                    "doc_id": doc_id,
                    "size": file_size,
                    "status": "success"
                })
            else:
                uploaded.append({
                    "name": safe_filename,
                    "status": "error",
                    "error": "No doc_id returned"
                })
        except Exception as exc:
            logger.error("Error uploading artifact '%s': %s", safe_filename, exc)
            uploaded.append({
                "name": safe_filename,
                "status": "error",
                "error": str(exc)
            })

    return {
        "status": "success",
        "uploaded": uploaded,
        "count": len([u for u in uploaded if u.get("status") == "success"])
    }

@app.get("/api/artifacts/{doc_id}/file")
async def get_artifact_file(doc_id: str):
    """Serve canonical raw file for in-browser modal preview."""
    if not doc_id or ".." in doc_id or not re.match(r"^[a-zA-Z0-9_-]+$", doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    try:
        from db_service import document_db
        store = getattr(document_db, "_store", None)
        if not store or not store.document_exists(doc_id):
            raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")

        doc_dir = store.doc_dir(doc_id)
        manifest = store.load_manifest(doc_id)
        origin = manifest.get("origin", {})
        original_name = origin.get("original_name") or f"{doc_id}.bin"

        candidates = []
        orig_dir = doc_dir / "original"
        if orig_dir.exists():
            candidates.extend(list(orig_dir.iterdir()))
        candidates.append(doc_dir / original_name)
        source_path = origin.get("source_path")
        if source_path:
            p = Path(source_path)
            if p.is_file():
                candidates.append(p)
        img_dir = doc_dir / "images"
        if img_dir.exists():
            candidates.extend(list(img_dir.iterdir()))

        found_path: Optional[Path] = None
        for cand in candidates:
            if cand and cand.is_file() and cand.stat().st_size > 0:
                found_path = cand
                break

        if not found_path:
            raise HTTPException(status_code=404, detail="File content not found on server")

        import mimetypes
        media_type, _ = mimetypes.guess_type(str(found_path))
        if not media_type:
            media_type = origin.get("mime_type") or "application/octet-stream"

        return FileResponse(
            path=found_path,
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{original_name}"'}
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error serving file for doc '%s': %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to serve artifact file: {exc}") from exc

@app.delete("/api/artifacts/{doc_id}")
async def delete_artifact(doc_id: str):
    """Delete document artifact from disk and vector database."""
    if not doc_id or ".." in doc_id or not re.match(r"^[a-zA-Z0-9_-]+$", doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id")

    try:
        from db_service import document_db
        store = getattr(document_db, "_store", None)
        chroma = getattr(document_db, "_chroma", None)

        if chroma:
            try:
                chroma._delete_source_records(doc_id)
            except Exception as ce:
                logger.warning("Chroma source deletion note for %s: %s", doc_id, ce)
            try:
                existing_derived = chroma._derived.get(where={"doc_id": {"$eq": doc_id}})
                d_ids = existing_derived.get("ids", [])
                if d_ids:
                    chroma._derived.delete(ids=d_ids)
            except Exception as de:
                logger.warning("Chroma derived deletion note for %s: %s", doc_id, de)

        if store:
            doc_dir = store.doc_dir(doc_id)
            if doc_dir.exists():
                shutil.rmtree(doc_dir, ignore_errors=True)

        return {"status": "success", "deleted": doc_id}
    except Exception as exc:
        logger.error("Error deleting artifact %s: %s", doc_id, exc)
        raise HTTPException(status_code=500, detail=f"Failed to delete artifact: {exc}") from exc

@app.get("/api/artifacts/{doc_id}/images/{image_id}")
async def get_artifact_image(doc_id: str, image_id: str):
    """Serve canonical image artifacts to the client for confident image display."""
    if not doc_id or ".." in doc_id or not re.match(r"^[a-zA-Z0-9_-]+$", doc_id):
        raise HTTPException(status_code=400, detail="Invalid doc_id")
    if not image_id or ".." in image_id or not re.match(r"^[a-zA-Z0-9_.-]+$", image_id):
        raise HTTPException(status_code=400, detail="Invalid image_id")
    try:
        from db_service import document_db
        element = document_db.artifact_fetch(doc_id, image_id)
        if element.get("type") == "image" and element.get("exists") and element.get("local_path"):
            path = Path(element["local_path"]).resolve()
            allowed_roots = [config.ARTIFACTS_ROOT.resolve()]
            store = getattr(document_db, "_store", None)
            if store and hasattr(store, "root"):
                allowed_roots.append(Path(store.root).resolve())
            is_confined = any(path.is_relative_to(r) for r in allowed_roots)
            if not is_confined:
                logger.warning("Path traversal attempt in artifact serving: %s", path)
                raise HTTPException(status_code=403, detail="Forbidden")
            if path.is_file():
                ext = path.suffix.lstrip(".").lower()
                media_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"
                return FileResponse(path, media_type=media_type)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to serve artifact image %s/%s: %s", doc_id, image_id, exc)
    raise HTTPException(status_code=404, detail="Image artifact not found")

async def _prepare_chat_request(
    objective: str,
    files: Optional[List[UploadFile]],
    direct_types: Optional[set[str]] = None,
    chat_id: Optional[str] = None,
) -> tuple[RunState, list[dict], dict]:
    """Prepare request files, optionally bypassing DB ingestion for simple Flash inputs."""
    request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    req_temp_dir = config.TEMP_DIR / request_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)

    final_chat_id = chat_id or f"chat_{uuid.uuid4().hex[:12]}"
    run_state = RunState(
        request_id=request_id,
        chat_id=final_chat_id,
        user_id=config.DEFAULT_USER_ID,
        user_text=objective.strip(),
    )
    attachments_manifest = []
    file_map = {}

    if files:
        for idx, file_item in enumerate(files, start=1):
            if not file_item.filename:
                continue

            ref_id = f"file_{idx}"
            safe_filename = Path(file_item.filename).name
            save_path = req_temp_dir / safe_filename

            # Save uploaded bytes streamingly with bounded size check
            file_size = await _save_uploaded_file(file_item, save_path)
            suffix = Path(safe_filename).suffix.lstrip(".").lower()
            direct_ready = suffix in (direct_types or set())

            # Ingest into SageDocumentDB
            doc_id = None
            ingest_error = None
            if not direct_ready:
                try:
                    from db_service import document_db
                    ingest_res = document_db.ingest_document(str(save_path))
                    if isinstance(ingest_res, dict) and ingest_res.get("status") == "error":
                        ingest_error = str(ingest_res.get("error", "Ingestion failed"))
                    else:
                        doc_id = ingest_res.get("doc_id")
                except Exception as exc:
                    print(f"Error: Document DB ingestion failed for '{safe_filename}': {exc}")
                    ingest_error = str(exc)

            if doc_id or direct_ready:
                # Permanently preserve source file in document artifact folder
                if doc_id:
                    try:
                        from db_service import document_db
                        store = getattr(document_db, "_store", None)
                        if store:
                            doc_dir = store.doc_dir(doc_id)
                            doc_orig_dir = doc_dir / "original"
                            doc_orig_dir.mkdir(parents=True, exist_ok=True)
                            orig_copy_path = doc_orig_dir / safe_filename
                            if not orig_copy_path.exists() and save_path.exists():
                                shutil.copy2(save_path, orig_copy_path)
                    except Exception as copy_err:
                        logger.warning("Could not preserve source copy for %s: %s", doc_id, copy_err)

                    run_state.register_document(
                        doc_id=doc_id,
                        display_name=safe_filename,
                        file_type=suffix,
                        source_name=safe_filename,
                    )
                img_id = None
                if doc_id and suffix in ("png", "jpg", "jpeg", "webp", "bmp", "gif"):
                    try:
                        from db_service import document_db
                        img_arts = document_db.list_artifacts(doc_id=doc_id, artifact_type="image")
                        if img_arts and isinstance(img_arts, list):
                            img_id = img_arts[0].get("element_id")
                    except Exception:
                        img_id = "img_000001"
                    if not img_id:
                        img_id = "img_000001"

                file_entry = {
                    "ref": ref_id,
                    "doc_id": doc_id,
                    "name": safe_filename,
                    "type": suffix,
                    "size": file_size,
                    "path": str(save_path),
                    "status": "ready" if direct_ready else "ingested",
                }
                att_manifest_entry = {
                    "ref": ref_id,
                    "doc_id": doc_id,
                    "name": safe_filename,
                    "type": suffix,
                    "size": file_size,
                    "status": "ready" if direct_ready else "ingested",
                }
                if img_id:
                    file_entry["image_id"] = img_id
                    att_manifest_entry["image_id"] = img_id

                attachments_manifest.append(att_manifest_entry)
            else:
                # Ingestion failed - do NOT fabricate or register a ghost doc_id!
                file_entry = {
                    "ref": ref_id,
                    "doc_id": None,
                    "name": safe_filename,
                    "type": suffix,
                    "size": file_size,
                    "path": str(save_path),
                    "status": "ingestion_failed",
                    "error": ingest_error or "Document DB ingestion failed",
                }
                attachments_manifest.append({
                    "ref": ref_id,
                    "doc_id": None,
                    "name": safe_filename,
                    "type": suffix,
                    "size": file_size,
                    "status": "ingestion_failed",
                    "error": ingest_error or "Document DB ingestion failed",
                })
            file_map[ref_id] = file_entry

    return run_state, attachments_manifest, file_map


@app.post("/api/chat")
async def chat_endpoint(
    objective: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    chat_id: Optional[str] = Form(None),
):
    if not objective or not objective.strip():
        raise HTTPException(status_code=400, detail="Objective prompt cannot be empty.")

    run_state, attachments_manifest, file_map = await _prepare_chat_request(
        objective, files, chat_id=chat_id
    )

    try:
        # Flash is the default runtime.  Keep the legacy orchestrator lazy so
        # a stale document-vector index cannot prevent Flash and durable
        # SQLite memory from starting.
        from orchestrator import orchestrator
        result = orchestrator.run(
            user_objective=objective.strip(),
            attachments_manifest=attachments_manifest,
            file_map=file_map,
            run_state=run_state
        )
        if isinstance(result, dict):
            result["chat_id"] = run_state.chat_id
            if run_state.registered_documents:
                # Backwards-compatible document reference metadata
                result["registered_documents"] = [
                    {"doc_id": d.doc_id, "name": d.display_name, "type": d.file_type}
                    for d in run_state.registered_documents
                ]
        return JSONResponse(content=result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )


@app.post("/api/flash")
async def flash_endpoint(
    objective: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    session_id: Optional[str] = Form(None),
    observer_run_id: Optional[str] = Form(None),
    temperature: Optional[float] = Form(None),
    save_history: Optional[str] = Form(None),
):
    """Run the host-agnostic Flash graph while leaving legacy chat untouched."""
    if not objective or not objective.strip():
        raise HTTPException(status_code=400, detail="Objective prompt cannot be empty.")
    request_started = time.perf_counter()
    direct_types = {"png", "jpg", "jpeg", "webp", "gif", "bmp", "txt", "md", "csv", "json", "log", "py", "js", "html", "xml", "yaml", "yml"}
    run_state, attachments_manifest, file_map = await _prepare_chat_request(
        objective, files, direct_types=direct_types
    )
    intake_seconds = time.perf_counter() - request_started
    persist = True
    if save_history is not None:
        persist = str(save_history).strip().lower() not in {"0", "false", "no", "off"}
    temp_value = None
    if temperature is not None:
        try:
            temp_value = max(0.0, min(2.0, float(temperature)))
        except (TypeError, ValueError):
            temp_value = None
    from starlette.concurrency import run_in_threadpool
    from flash.service import flash_service
    try:
        result = await run_in_threadpool(
            flash_service.run,
            objective=objective.strip(),
            attachments=attachments_manifest,
            file_map=file_map,
            session_id=session_id,
            observer_run_id=observer_run_id,
            user_id=config.DEFAULT_USER_ID,
            temperature=temp_value,
            save_history=persist,
        )
        if run_state.registered_documents:
            result["registered_documents"] = [
                {"doc_id": d.doc_id, "name": d.display_name, "type": d.file_type}
                for d in run_state.registered_documents
            ]
        result.setdefault("telemetry", {})["intake_seconds"] = round(intake_seconds, 4)
        result["telemetry"]["request_wall_time"] = round(time.perf_counter() - request_started, 4)
        return JSONResponse(content=result)
    except Exception as exc:
        logger.exception("Flash request failed")
        try:
            from core.observer import observer
            observer.emit(observer_run_id or session_id or "unknown", "sage", "request", "failed", "Flash request failed", {"error": str(exc)})
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"status": "error", "error": str(exc)})

@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    objective: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    chat_id: Optional[str] = Form(None),
):
    """Real-time SSE streaming endpoint for live multi-model telemetry and inspection."""
    if not objective or not objective.strip():
        raise HTTPException(status_code=400, detail="Objective prompt cannot be empty.")

    run_state, attachments_manifest, file_map = await _prepare_chat_request(
        objective, files, chat_id=chat_id
    )

    async def event_generator():
        import asyncio
        import threading
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def stream_callback(evt: dict):
            loop.call_soon_threadsafe(queue.put_nowait, evt)

        def worker():
            try:
                from orchestrator import orchestrator
                res = orchestrator.run(
                    user_objective=objective.strip(),
                    attachments_manifest=attachments_manifest,
                    file_map=file_map,
                    run_state=run_state,
                    event_callback=stream_callback,
                )
                if isinstance(res, dict):
                    res["chat_id"] = run_state.chat_id
                loop.call_soon_threadsafe(queue.put_nowait, {"event": "done", "result": res})
            except Exception as exc:
                import traceback
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"event": "error", "error": str(exc), "traceback": traceback.format_exc()}
                )

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        while True:
            item = await queue.get()
            event_name = item.get("event", "message")
            try:
                payload = json.dumps(item)
            except Exception as dump_err:
                payload = json.dumps({"event": event_name, "error": str(dump_err)})
            yield f"event: {event_name}\ndata: {payload}\n\n"
            if event_name in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# ── Memory API Endpoints ──────────────────────────────────────────────────

@app.get("/api/memory/status")
async def get_memory_status():
    """Return live status of backend database, vector index, embedding model, and model availability."""
    from sage_memory import sage_memory, get_backend_type, _get_sqlite_db_path
    
    # 1. Database status
    db_backend = get_backend_type()
    db_connected = False
    try:
        conn = sage_memory._get_connection()
        conn.close()
        db_connected = True
    except Exception as e:
        logger.warning("DB health check failed: %s", e)

    # 2. Chroma status
    chroma_status = "offline"
    hot_col_count = 0
    cold_col_count = 0
    try:
        from sage_memory_index import memory_vector_index
        hot_col = memory_vector_index._get_collection("hot")
        cold_col = memory_vector_index._get_collection("cold")
        hot_col_count = hot_col.count()
        cold_col_count = cold_col.count()
        chroma_status = "ready"
    except (Exception, BaseException) as e:
        chroma_status = f"unavailable ({e.__class__.__name__})"

    # 3. Curator runtime status (the model may be local CPU or a remote role).
    from flash.runtime_config import runtime_config
    from flash.local_manager import local_flash_manager
    from flash.catalog import load_catalog
    from memory_system.job_store import memory_job_store
    memory_role = runtime_config.role("memory")
    recent_jobs = memory_job_store.list(limit=100)
    provider = memory_role.get("provider")
    failed_jobs = [job for job in recent_jobs if job.get("status") == "failed"]
    latest_failed = bool(recent_jobs and recent_jobs[0].get("status") == "failed")
    configured = provider != "disabled"
    runtime_detail = ""
    available = configured
    if provider in {"local_cpu", "local_gpu"}:
        memory_runtime = local_flash_manager.status().get("memory", {})
        catalog_memory = load_catalog()["memory"]
        model_root = Path(config.MODEL_DIR or config.BASE_DIR / "models")
        prerequisites = bool(
            config.LLAMA_SERVER_PATH and Path(config.LLAMA_SERVER_PATH).is_file()
            and (model_root / catalog_memory["file"]).is_file()
        )
        available = bool(memory_runtime.get("healthy") or prerequisites)
        runtime_detail = "running and healthy" if memory_runtime.get("healthy") else (
            "configured; starts with the next memory task" if prerequisites
            else "not runnable: configure LLAMA_SERVER_PATH and install the catalog memory GGUF, or select a remote endpoint"
        )
    elif provider == "remote":
        available = bool(memory_role.get("connection") and memory_role.get("model_id"))
        runtime_detail = "remote endpoint configured" if available else "remote endpoint is incomplete"
    else:
        runtime_detail = "disabled in Flash runtime settings"
    if latest_failed:
        label = "CURATOR NEEDS ATTENTION"
    elif available:
        label = "2B CURATOR READY"
    elif configured:
        label = "CURATOR NOT CONFIGURED"
    else:
        label = "CURATOR DISABLED"
    model_status = {
        "available": available and not latest_failed,
        "configured": configured,
        "label": label,
        "detail": f"Provider: {provider}; model: {memory_role.get('model_id')}; {runtime_detail}. Global facts are extracted after every completed turn; Cold summaries are created when a turn leaves the five-turn Hot window.",
        "provider": provider,
        "model_id": memory_role.get("model_id"),
        "last_error": recent_jobs[0].get("error") if latest_failed else None,
        "jobs": {
            state: sum(1 for job in recent_jobs if job.get("status") == state)
            for state in ("queued", "scheduled", "running", "completed", "failed")
        },
    }

    # 4. Summary counts
    from memory_system.quality import is_valid_memory
    semantic = sage_memory.list_memories(user_id=config.DEFAULT_USER_ID, memory_tier="cold", status="active", limit=500)
    semantic = [memory for memory in semantic if is_valid_memory(
        str(memory.get("content") or ""), "global" if not memory.get("source_chat_id") else "cold"
    )]
    active_cold = sum(1 for memory in semantic if memory.get("source_chat_id"))
    active_global = sum(1 for memory in semantic if not memory.get("source_chat_id"))
    compacted_count = len(sage_memory.list_memories(user_id=config.DEFAULT_USER_ID, status="compacted", limit=500))

    return {
        "status": "success",
        "database": {
            "engine": db_backend,
            "status": "connected" if db_connected else "disconnected",
            "sqlite_path": str(_get_sqlite_db_path()) if db_backend == "sqlite" else None,
        },
        "vector_index": {
            "engine": "Chroma",
            "status": chroma_status,
            "hot_vectors": hot_col_count,
            "cold_vectors": cold_col_count,
            "collections": {
                "hot": getattr(config, "SAGE_MEMORY_HOT_COLLECTION", "sage_memory_hot"),
                "cold": getattr(config, "SAGE_MEMORY_COLD_COLLECTION", "sage_memory_cold"),
            },
        },
        "embedding": {
            "model": getattr(config, "EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"),
            "status": "ready" if chroma_status == "ready" else "fallback_deterministic",
        },
        "model_runtime": model_status,
        "counts": {
            # Legacy response keys are retained for the existing UI: its left
            # board displays chat-scoped Cold and its right board displays Global.
            "hot_active": active_cold,
            "cold_active": active_global,
            "chat_cold_active": active_cold,
            "global_active": active_global,
            "derived_compacted": compacted_count,
        },
        "configuration": {
            "recent_chat_max_turns": config.SAGE_RECENT_CHAT_MAX_TURNS,
            "recent_chat_budget_tokens": config.SAGE_RECENT_CHAT_BUDGET_TOKENS,
            "context_budget_tokens": config.SAGE_CONTEXT_MEMORY_BUDGET_TOKENS,
            "global_budget_tokens": config.SAGE_GLOBAL_MEMORY_BUDGET_TOKENS,
            "global_max_items": config.SAGE_GLOBAL_MEMORY_MAX_ITEMS,
        },
    }


@app.get("/api/memory/recent-chat")
async def get_recent_chat_memory(chat_id: Optional[str] = None, limit: int = 5):
    """Retrieve the bounded recent completed-turn window from the ledger."""
    from sage_memory import sage_memory
    from memory_system.coordinator import memory_coordinator

    resolved_chat_id = chat_id
    if not resolved_chat_id:
        # Default to most recent chat for the default user
        chats = sage_memory.list_chats(config.DEFAULT_USER_ID, limit=1)
        if chats:
            resolved_chat_id = chats[0]["chat_id"]

    recent = {"messages": [], "turn_count": 0, "token_estimate": 0}
    if resolved_chat_id:
        recent = memory_coordinator.recent_turns(
            resolved_chat_id,
            max_turns=max(1, min(limit or config.SAGE_RECENT_CHAT_MAX_TURNS, 20)),
            token_budget=config.SAGE_RECENT_CHAT_BUDGET_TOKENS,
        )

    return {
        "status": "success",
        "chat_id": resolved_chat_id,
        "configured_window_turns": config.SAGE_RECENT_CHAT_MAX_TURNS,
        "label": "Recent conversation context - chronological, not semantic memory.",
        "count": len(recent["messages"]),
        "turn_count": recent["turn_count"],
        "token_estimate": recent["token_estimate"],
        "messages": recent["messages"],
    }


@app.get("/api/memory/memories")
async def list_memories_api(
    tier: Optional[str] = None,
    chat_id: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = "active",
    search: Optional[str] = None,
    scope: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """List semantic memories with filtering and search."""
    from sage_memory import sage_memory, ALLOWED_CATEGORIES
    from memory_system.quality import is_valid_memory

    effective_user_id = user_id or config.DEFAULT_USER_ID
    tier_filter = tier.lower().strip() if tier and tier.lower().strip() in ("hot", "cold") else None
    cat_filter = category.lower().strip() if category and category.lower().strip() in ALLOWED_CATEGORIES else None

    # If search string provided, search via vector index or fallback to substring
    memories = []
    if search and search.strip():
        q = search.strip().lower()
        all_mems = sage_memory.list_memories(
            user_id=effective_user_id,
            category=cat_filter,
            memory_tier=tier_filter,
            chat_id=chat_id if tier_filter == "hot" else None,
            status=status or "active",
            limit=500,
        )
        memories = [m for m in all_mems if q in (m.get("content") or "").lower() or q in (m.get("category") or "").lower() or q in (m.get("memory_id") or "").lower()]
        sage_memory.log_activity("SEARCH", user_id=effective_user_id, chat_id=chat_id, details=f"UI Search query: '{search}', matches: {len(memories)}")
    else:
        memories = sage_memory.list_memories(
            user_id=effective_user_id,
            category=cat_filter,
            memory_tier=tier_filter,
            chat_id=chat_id if tier_filter == "hot" else None,
            status=status or "active",
            limit=500,
        )

    for memory in memories:
        memory["scope"] = "global" if not memory.get("source_chat_id") and memory.get("category") == "personal" else "cold"
    memories = [memory for memory in memories if is_valid_memory(str(memory.get("content") or ""), memory["scope"])]
    memories = [memory for memory in memories if memory["scope"] != "global" or memory.get("category") == "personal"]
    if chat_id and not tier_filter:
        memories = [memory for memory in memories if memory.get("source_chat_id") in {None, chat_id}]
    if scope in {"global", "cold"}:
        memories = [memory for memory in memories if memory["scope"] == scope]

    # Backward-compatible key. The UI now presents all chat-scoped Cold
    # records, not only records whose curator happened to choose "summary".
    derived_memories = [m for m in memories if m.get("scope") == "cold"]

    return {
        "status": "success",
        "total": len(memories),
        "memories": memories,
        "derived": derived_memories,
    }


@app.post("/api/memory")
async def create_memory_api(payload: Dict[str, Any] = Body(...)):
    """Create a new hot or cold memory in the canonical backend database."""
    from sage_memory import sage_memory, validate_category

    content = payload.get("content")
    if not content or not str(content).strip():
        raise HTTPException(status_code=400, detail="content must be a non-empty string.")

    cat_raw = payload.get("category", "fact")
    try:
        validated_cat = validate_category(cat_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    scope = str(payload.get("scope") or "").lower().strip()
    tier_raw = str(payload.get("tier") or payload.get("memory_tier") or "cold").lower().strip()
    if not scope and tier_raw != "hot":
        scope = "cold"
    if scope == "global" and validated_cat != "personal":
        raise HTTPException(
            status_code=400,
            detail="Global memory is reserved for personal information. Store preferences, projects, instructions, and other details in Chat Memory.",
        )
    tier = "hot" if tier_raw == "hot" and not scope else "cold"

    chat_id = payload.get("chat_id")
    if (tier == "hot" or scope == "cold") and not chat_id:
        chats = sage_memory.list_chats(config.DEFAULT_USER_ID, limit=1)
        if chats:
            chat_id = chats[0]["chat_id"]
        else:
            chat_id = "default_session"

    importance = max(0.0, min(1.0, float(payload.get("importance", 0.5))))
    confidence = max(0.0, min(1.0, float(payload.get("confidence", 1.0))))
    user_id = payload.get("user_id") or config.DEFAULT_USER_ID

    stored = sage_memory.store_memory(
        user_id=user_id,
        content=str(content).strip(),
        category=validated_cat,
        importance=importance,
        confidence=confidence,
        source_chat_id=None if scope == "global" else chat_id,
        memory_tier=tier,
    )
    stored["scope"] = "global" if not stored.get("source_chat_id") else "cold"

    return {"status": "success", "memory": stored}


@app.post("/api/memory/bulk-hard-delete")
async def bulk_hard_delete_global_memories(payload: Dict[str, Any] = Body(...)):
    """Permanently delete selected Global-memory records owned by the current user."""
    from sage_memory import sage_memory

    requested_ids = payload.get("memory_ids")
    if not isinstance(requested_ids, list) or not requested_ids:
        raise HTTPException(status_code=400, detail="memory_ids must be a non-empty list.")
    ids = list(dict.fromkeys(str(memory_id).strip() for memory_id in requested_ids if str(memory_id).strip()))[:500]
    user_id = str(payload.get("user_id") or config.DEFAULT_USER_ID)
    deleted: List[str] = []
    skipped: List[str] = []
    for memory_id in ids:
        existing = sage_memory.get_memory(memory_id)
        # This endpoint intentionally cannot remove chat-scoped records.
        if not existing or existing.get("user_id") != user_id or existing.get("source_chat_id"):
            skipped.append(memory_id)
            continue
        if sage_memory.hard_delete_memory(memory_id):
            deleted.append(memory_id)
        else:
            skipped.append(memory_id)
    return {"status": "success", "hard_deleted": len(deleted), "deleted_ids": deleted, "skipped_ids": skipped}


@app.put("/api/memory/{memory_id}")
async def update_memory_api(memory_id: str, payload: Dict[str, Any] = Body(...)):
    """In-place update of memory content, category, importance, or confidence."""
    from sage_memory import sage_memory, validate_category

    existing = sage_memory.get_memory(memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")

    category = payload.get("category")
    validated_cat = None
    if category is not None:
        try:
            validated_cat = validate_category(str(category))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    updated = sage_memory.update_memory(
        memory_id=memory_id,
        content=payload.get("content"),
        category=validated_cat,
        importance=payload.get("importance"),
        confidence=payload.get("confidence"),
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update memory in database.")

    return {"status": "success", "memory": updated}


@app.delete("/api/memory/{memory_id}")
async def delete_memory_api(memory_id: str):
    """Soft-delete memory record in backend database and sync vector index."""
    from sage_memory import sage_memory

    existing = sage_memory.get_memory(memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")

    success = sage_memory.delete_memory(memory_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete memory.")

    return {"status": "success", "deleted": True, "memory_id": memory_id}


@app.post("/api/memory/{memory_id}/promote")
async def promote_memory_api(memory_id: str, payload: Dict[str, Any] = Body(default={})):
    """Promote session hot memory to global cold memory."""
    from sage_memory import sage_memory

    existing = sage_memory.get_memory(memory_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found.")

    user_id = payload.get("user_id") or existing.get("user_id") or config.DEFAULT_USER_ID
    try:
        promoted = sage_memory.promote_memory(memory_id=memory_id, user_id=user_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {"status": "success", "memory": promoted}


@app.get("/api/memory/activity")
async def get_memory_activity_api(user_id: Optional[str] = None, limit: int = 25):
    """Retrieve operational memory audit log events (STORE, SEARCH, UPDATE, DELETE, PROMOTE, SUMMARIZE)."""
    from sage_memory import sage_memory

    events = sage_memory.get_recent_activity(user_id=user_id or config.DEFAULT_USER_ID, limit=limit)
    # Normalise: DB stores 'event_type' but the frontend reads 'action'.
    # Expose both so neither the DB schema nor the JS need to change.
    for ev in events:
        if "action" not in ev and "event_type" in ev:
            ev["action"] = ev["event_type"]
    return {"status": "success", "activities": events}


@app.post("/api/memory/activity/bulk-hard-delete")
async def bulk_hard_delete_memory_activity(payload: Dict[str, Any] = Body(...)):
    """Permanently delete selected audit-log rows belonging to the current user."""
    from sage_memory import sage_memory

    requested_ids = payload.get("activity_ids")
    if not isinstance(requested_ids, list) or not requested_ids:
        raise HTTPException(status_code=400, detail="activity_ids must be a non-empty list.")
    ids = list(dict.fromkeys(str(activity_id).strip() for activity_id in requested_ids if str(activity_id).strip()))[:500]
    user_id = str(payload.get("user_id") or config.DEFAULT_USER_ID)
    deleted = sage_memory.hard_delete_activity(ids, user_id)
    return {"status": "success", "hard_deleted": deleted, "requested": len(ids)}


# Mount static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")


def _html_with_asset_version(path: Path) -> Response:
    """Serve HTML with no-store caching and a stable asset cache-buster."""
    text = path.read_text(encoding="utf-8")
    # Keep any existing ?v=… markers in sync with the server version.
    text = re.sub(
        r'(/static/(?:app\.js|style\.css|artifact_graph\.js))(?:\?v=[^"\']*)?',
        rf'\1?v={STATIC_ASSET_VERSION}',
        text,
    )
    return Response(
        content=text,
        media_type="text/html; charset=utf-8",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/")
async def index():
    index_file = config.STATIC_DIR / "index.html"
    if index_file.exists():
        return _html_with_asset_version(index_file)
    return JSONResponse({"message": "SAGE Backend is running. Static files pending."})


@app.get("/memory")
async def memory_view():
    """Open the canonical in-app Memory workspace, not the retired mock page."""
    index_file = config.STATIC_DIR / "index.html"
    if index_file.exists():
        return _html_with_asset_version(index_file)
    return JSONResponse({"message": "Memory view pending."})

if __name__ == "__main__":
    print(f"\n========================================================")
    print(f"  SAGE - Minimal Multi-Model Agent Orchestrator")
    print(f"  Web UI: http://127.0.0.1:{config.APP_PORT}")
    print(f"========================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=config.APP_PORT, reload=False)

