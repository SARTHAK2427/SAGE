"""Retired Flash 2B compressor.

Chat continuity and durable facts now live only in the local sage_memory
ledger. This module remains so leftover JSONL files can be cleared on chat
delete; it never invokes a memory model.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import config


class FlashMemoryWorker:
    def __init__(self) -> None:
        self._file_lock = threading.Lock()
        self._store_path = config.MODEL_RUNTIME_ROOT / "flash_memory.jsonl"

    def schedule(
        self,
        session_id: str,
        user_message: str,
        answer: str,
        *,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        return None

    def recent(self, session_id: str, limit: int = 4) -> List[Dict[str, Any]]:
        return []

    def clear_session(self, session_id: str) -> int:
        """Remove leftover Flash JSONL summaries if an old file still exists."""
        if not self._store_path.is_file():
            return 0
        with self._file_lock:
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
