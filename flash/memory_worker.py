"""Non-blocking 2B memory compression worker with durable JSONL output."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import config
from flash.runtime_config import runtime_config
from flash.transport import flash_transport


class FlashMemoryWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sage-flash-memory")
        self._slots = threading.BoundedSemaphore(config.FLASH_MEMORY_QUEUE_SIZE)
        self._file_lock = threading.Lock()
        self._cleared_sessions: set[str] = set()
        self._store_path = config.MODEL_RUNTIME_ROOT / "flash_memory.jsonl"

    def schedule(
        self,
        session_id: str,
        user_message: str,
        answer: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        if runtime_config.role("memory")["provider"] == "disabled":
            return None
        with self._file_lock:
            if session_id in self._cleared_sessions:
                return None
        if not self._slots.acquire(blocking=False):
            return None
        job_id = f"mem_{int(time.time() * 1000)}"
        self._executor.submit(self._run, job_id, session_id, user_message, answer, user_id or config.DEFAULT_USER_ID)
        return job_id

    def _run(self, job_id: str, session_id: str, user_message: str, answer: str, user_id: str) -> None:
        try:
            prompt = (
                "Compress this completed conversation turn into durable memory. Preserve factual preferences, "
                "commitments, named entities, and unresolved tasks. Omit conversational filler. Return concise plain text.\n\n"
                f"USER:\n{user_message}\n\nASSISTANT:\n{answer}"
            )
            result = flash_transport.invoke("memory", [
                {"role": "system", "content": "You are SAGE's background memory compressor."},
                {"role": "user", "content": prompt},
            ])
            record = {
                "job_id": job_id, "session_id": session_id, "created_at": time.time(),
                "summary": result.get("content", ""), "source_user": user_message[:1000],
            }
            with self._file_lock:
                if session_id in self._cleared_sessions:
                    return
                self._store_path.parent.mkdir(parents=True, exist_ok=True)
                with self._store_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            # Keep the Flash compressor as a producer of session-scoped hot
            # memories, so Flash and the durable-memory system share the same
            # canonical store without promoting every turn to global memory.
            summary = str(record["summary"] or "").strip()
            if summary:
                try:
                    from sage_memory import sage_memory
                    sage_memory.store_memory(
                        user_id=user_id,
                        content=summary,
                        category="fact",
                        source_chat_id=session_id,
                        memory_tier="hot",
                    )
                except Exception:
                    pass
        finally:
            self._slots.release()

    def recent(self, session_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        if not self._store_path.is_file():
            return []
        rows: List[Dict[str, Any]] = []
        with self._file_lock:
            for line in self._store_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("session_id") == session_id:
                    rows.append(row)
        return rows[-limit:]

    def clear_session(self, session_id: str) -> int:
        """Remove durable summaries for one explicitly reset chat session."""
        if not self._store_path.is_file():
            with self._file_lock:
                self._cleared_sessions.add(session_id)
            return 0
        with self._file_lock:
            self._cleared_sessions.add(session_id)
            kept: List[str] = []
            removed = 0
            for line in self._store_path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if row.get("session_id") == session_id:
                    removed += 1
                else:
                    kept.append(line)
            content = "\n".join(kept)
            self._store_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
            return removed


memory_worker = FlashMemoryWorker()
