"""Thread-safe, append-only runtime observation stream for SAGE."""

from __future__ import annotations

import copy
import re
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, List


_SECRET_KEYS = {"api_key", "authorization", "password", "secret", "access_token", "refresh_token", "hf_token"}
_DATA_URL = re.compile(r"data:[^;]+;base64,[A-Za-z0-9+/=]+")


def _sanitize(value: Any, depth: int = 0) -> Any:
    if depth > 7:
        return "[depth limited]"
    if isinstance(value, dict):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in _SECRET_KEYS or lowered.endswith(("_api_key", "_password", "_secret")):
                result[str(key)] = "[redacted]"
            else:
                result[str(key)] = _sanitize(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth + 1) for item in value[:200]]
    if isinstance(value, str):
        cleaned = _DATA_URL.sub("[base64 image redacted]", value)
        return cleaned if len(cleaned) <= 50000 else cleaned[:50000] + "… [truncated]"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return str(value)


class ObserverStore:
    def __init__(self, max_runs: int = 100, max_events_per_run: int = 1000) -> None:
        self._lock = threading.RLock()
        self._runs: Dict[str, Deque[Dict[str, Any]]] = {}
        self._order: Deque[str] = deque(maxlen=max_runs)
        self._max_events = max_events_per_run
        self._conditions: Dict[str, threading.Condition] = defaultdict(lambda: threading.Condition(self._lock))

    def emit(self, run_id: str, actor: str, phase: str, status: str, summary: str,
             payload: Any = None, duration_ms: float | None = None) -> Dict[str, Any]:
        rid = str(run_id or "system")
        with self._lock:
            if rid not in self._runs:
                if len(self._order) == self._order.maxlen and self._order:
                    expired = self._order.popleft()
                    self._runs.pop(expired, None)
                self._order.append(rid)
                self._runs[rid] = deque(maxlen=self._max_events)
            sequence = self._runs[rid][-1]["sequence"] + 1 if self._runs[rid] else 1
            event = {
                "run_id": rid, "sequence": sequence, "timestamp": time.time(),
                "actor": str(actor), "phase": str(phase), "status": str(status),
                "summary": str(summary), "payload": _sanitize(payload or {}),
            }
            if duration_ms is not None:
                event["duration_ms"] = round(float(duration_ms), 3)
            self._runs[rid].append(event)
            self._conditions[rid].notify_all()
            return copy.deepcopy(event)

    def events(self, run_id: str, after: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            return [copy.deepcopy(evt) for evt in self._runs.get(run_id, ()) if evt["sequence"] > after]

    def runs(self, limit: int = 30) -> List[Dict[str, Any]]:
        with self._lock:
            result = []
            for rid in list(self._order)[-max(1, min(limit, 100)):][::-1]:
                events = self._runs.get(rid) or []
                last = events[-1] if events else {}
                result.append({"run_id": rid, "event_count": len(events),
                               "last_timestamp": last.get("timestamp"),
                               "last_status": last.get("status"),
                               "last_summary": last.get("summary")})
            return result

    def clear(self, run_id: str) -> None:
        with self._lock:
            self._runs.pop(run_id, None)
            try:
                self._order.remove(run_id)
            except ValueError:
                pass


observer = ObserverStore()
