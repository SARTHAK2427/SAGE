"""
core/model_runtime_cache.py

Dedicated runtime cache for model specialist outputs (code execution, OCR vision).
Maintains an in-memory layer for 0ms lookups during active orchestration,
while persisting artifacts under artifacts/model_runtime/<request_id>/ for
auditability, inspection, and lifecycle isolation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

import config

logger = logging.getLogger(__name__)

_SAFE_REQ_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
DEFAULT_MAX_CACHED_REQUESTS = int(os.environ.get("SAGE_MAX_RUNTIME_CACHE_REQUESTS", "100"))


class ModelRuntimeCache:
    """Zero-lag in-memory and disk caching layer for model runtime artifacts."""

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        max_requests: int = DEFAULT_MAX_CACHED_REQUESTS,
    ) -> None:
        self.root_dir = root_dir or config.MODEL_RUNTIME_ROOT
        self.max_requests = max(1, max_requests)
        self._memory_cache: OrderedDict[str, List[Dict[str, Any]]] = OrderedDict()

    def _validate_request_id(self, request_id: str) -> None:
        if not request_id or not isinstance(request_id, str) or ".." in request_id or not _SAFE_REQ_ID_RE.match(request_id):
            raise ValueError(f"Invalid or unsafe request_id: {request_id!r}")

    def _ensure_dir(self, request_id: str) -> Path:
        self._validate_request_id(request_id)
        req_dir = (self.root_dir / request_id).resolve()
        if not req_dir.is_relative_to(self.root_dir.resolve()):
            raise ValueError(f"Path traversal detected in request_id: {request_id!r}")
        req_dir.mkdir(parents=True, exist_ok=True)
        return req_dir

    def _append_to_memory(self, request_id: str, record: Dict[str, Any]) -> None:
        """Append record to memory cache with deterministic FIFO/LRU eviction."""
        if request_id not in self._memory_cache:
            if len(self._memory_cache) >= self.max_requests:
                self._memory_cache.popitem(last=False)
            self._memory_cache[request_id] = []
        else:
            self._memory_cache.move_to_end(request_id)
        self._memory_cache[request_id].append(record)

    def store_code_artifact(
        self,
        request_id: str,
        call_index: int,
        code: str,
        language: str = "python",
        execution_status: str = "success",
        stdout: str = "",
        stderr: str = "",
        exit_code: Optional[int] = 0,
        attempts_used: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store code artifact in memory (0ms) and persist to artifacts/model_runtime/<request_id>/."""
        self._validate_request_id(request_id)
        artifact_id = f"art_code_{call_index}_{uuid.uuid4().hex[:6]}"
        record: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "type": "code",
            "tool": "code_specialist",
            "call_index": call_index,
            "language": language or "python",
            "code": code or "",
            "execution_status": execution_status,
            "stdout": stdout or "",
            "stderr": stderr or "",
            "exit_code": exit_code,
            "attempts_used": attempts_used,
            "review_status": "review_required",
            "metadata": metadata or {},
        }

        # 1. Instant In-Memory Cache (0ms latency with deterministic bounded capacity)
        self._append_to_memory(request_id, record)

        # 2. Disk Persistence (isolated under artifacts/model_runtime)
        try:
            target_dir = self._ensure_dir(request_id)
            json_file = target_dir / f"{artifact_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)

            ext = "py" if (language or "").lower() in ("python", "py") else "txt"
            code_file = target_dir / f"{artifact_id}.{ext}"
            with open(code_file, "w", encoding="utf-8") as f:
                f.write(code or "")
        except Exception as exc:
            logger.warning("Failed to persist code artifact to disk: %s", exc)

        return record

    def store_vision_artifact(
        self,
        request_id: str,
        call_index: int,
        analysis: str,
        image_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store vision/OCR artifact in memory and persist to disk."""
        self._validate_request_id(request_id)
        artifact_id = f"art_vision_{call_index}_{uuid.uuid4().hex[:6]}"
        record: Dict[str, Any] = {
            "artifact_id": artifact_id,
            "type": "vision",
            "tool": "vision_ocr",
            "call_index": call_index,
            "analysis": analysis or "",
            "image_id": image_id,
            "review_status": "review_required",
            "metadata": metadata or {},
        }

        self._append_to_memory(request_id, record)

        try:
            target_dir = self._ensure_dir(request_id)
            json_file = target_dir / f"{artifact_id}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
        except Exception as exc:
            logger.warning("Failed to persist vision artifact to disk: %s", exc)

        return record

    def get_artifacts_for_request(self, request_id: str) -> List[Dict[str, Any]]:
        """Return all cached artifacts for a request (instant in-memory lookup)."""
        self._validate_request_id(request_id)
        return list(self._memory_cache.get(request_id, []))

    def clear_memory_cache(self, request_id: Optional[str] = None) -> None:
        """Clear cache for a specific request or all requests."""
        if request_id:
            self._validate_request_id(request_id)
            self._memory_cache.pop(request_id, None)
        else:
            self._memory_cache.clear()


# Module-level singleton
model_runtime_cache = ModelRuntimeCache()
