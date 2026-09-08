import os
import uuid
import shutil
import time
from typing import List, Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
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
    yield
    # Shutdown
    print("Shutting down SAGE and stopping any running model server...")
    model_manager.stop_current()

app = FastAPI(title="SAGE — Multi-Model Agent Orchestrator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/api/chat")
async def chat_endpoint(
    objective: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    session_id: Optional[str] = Form(None),
):
    if not objective or not objective.strip():
        raise HTTPException(status_code=400, detail="Objective prompt cannot be empty.")

    request_id = f"req_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    req_temp_dir = config.TEMP_DIR / request_id
    req_temp_dir.mkdir(parents=True, exist_ok=True)

    effective_session_id = session_id.strip() if (session_id and session_id.strip()) else f"sess_{uuid.uuid4().hex[:12]}"

    run_state = RunState(
        request_id=request_id,
        session_id=effective_session_id,
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

            # Save uploaded bytes
            content = await file_item.read()
            with open(save_path, "wb") as f:
                f.write(content)

            suffix = Path(safe_filename).suffix.lstrip(".").lower()
            file_size = len(content)

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
                file_entry = {
                    "ref": ref_id,
                    "doc_id": doc_id,
                    "name": safe_filename,
                    "type": suffix,
                    "size": file_size,
                    "path": str(save_path),
                    "status": "ingested",
                }
                attachments_manifest.append({
                    "ref": ref_id,
                    "doc_id": doc_id,
                    "name": safe_filename,
                    "type": suffix,
                    "size": file_size,
                    "status": "ingested",
                })
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

    try:
        result = orchestrator.run(
            user_objective=objective.strip(),
            attachments_manifest=attachments_manifest,
            file_map=file_map,
            run_state=run_state
        )
        if isinstance(result, dict):
            result["session_id"] = effective_session_id
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
