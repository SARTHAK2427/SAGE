import os
import uuid
import shutil
import time
import logging
from typing import List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)
from contextlib import asynccontextmanager

import json
import re
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import config
from model_manager import model_manager
from orchestrator import orchestrator
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
    print(f"SAGE Agent Orchestrator initialized. Static dir: {config.STATIC_DIR}")
    # Pre-arm Gemma 4B in background so Turn 1 starts with 0s initiation delay
    model_manager.rearm_agent_background()
    yield
    # Shutdown
    print("Shutting down SAGE and stopping any running model server...")
    model_manager.stop_current()

app = FastAPI(title="SAGE - Multi-Model Agent Orchestrator", lifespan=lifespan)

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
        "models": {k: v["name"] for k, v in config.MODELS.items()}
    }

@app.post("/api/stop")
async def stop_server():
    model_manager.stop_current()
    return {"status": "stopped"}

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
) -> tuple[RunState, list[dict], dict]:
    """Prepare RunState, attachment manifest, and file map for incoming chat request."""
    request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    req_temp_dir = config.TEMP_DIR / request_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)

    run_state = RunState(request_id=request_id, user_text=objective.strip())
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

            # Ingest into SageDocumentDB
            doc_id = None
            ingest_error = None
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

            if doc_id:
                run_state.register_document(
                    doc_id=doc_id,
                    display_name=safe_filename,
                    file_type=suffix,
                    source_name=safe_filename,
                )
                img_id = None
                if suffix in ("png", "jpg", "jpeg", "webp", "bmp", "gif"):
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
                    "status": "ingested",
                }
                att_manifest_entry = {
                    "ref": ref_id,
                    "doc_id": doc_id,
                    "name": safe_filename,
                    "type": suffix,
                    "size": file_size,
                    "status": "ingested",
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
    files: Optional[List[UploadFile]] = File(None)
):
    if not objective or not objective.strip():
        raise HTTPException(status_code=400, detail="Objective prompt cannot be empty.")

    run_state, attachments_manifest, file_map = await _prepare_chat_request(objective, files)

    try:
        result = orchestrator.run(
            user_objective=objective.strip(),
            attachments_manifest=attachments_manifest,
            file_map=file_map,
            run_state=run_state
        )
        if isinstance(result, dict) and run_state.registered_documents:
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

@app.post("/api/chat/stream")
async def chat_stream_endpoint(
    objective: str = Form(...),
    files: Optional[List[UploadFile]] = File(None)
):
    """Real-time SSE streaming endpoint for live multi-model telemetry and inspection."""
    if not objective or not objective.strip():
        raise HTTPException(status_code=400, detail="Objective prompt cannot be empty.")

    run_state, attachments_manifest, file_map = await _prepare_chat_request(objective, files)

    async def event_generator():
        import asyncio
        import threading
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def stream_callback(evt: dict):
            loop.call_soon_threadsafe(queue.put_nowait, evt)

        def worker():
            try:
                res = orchestrator.run(
                    user_objective=objective.strip(),
                    attachments_manifest=attachments_manifest,
                    file_map=file_map,
                    run_state=run_state,
                    event_callback=stream_callback,
                )
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
            yield f"event: {event_name}\ndata: {json.dumps(item)}\n\n"
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

# Mount static files
app.mount("/static", StaticFiles(directory=str(config.STATIC_DIR)), name="static")

@app.get("/")
async def index():
    index_file = config.STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"message": "SAGE Backend is running. Static files pending."})

if __name__ == "__main__":
    print(f"\n========================================================")
    print(f"  SAGE — Minimal Multi-Model Agent Orchestrator")
    print(f"  Web UI: http://127.0.0.1:{config.APP_PORT}")
    print(f"========================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=config.APP_PORT, reload=False)

