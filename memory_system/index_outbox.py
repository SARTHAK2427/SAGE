"""Durable relational outbox for rebuilding the derived Chroma index."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IndexOutbox:
    def ensure_schema(self) -> None:
        from sage_memory import get_backend_type, sage_memory
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS memory_index_outbox (
                            event_id TEXT PRIMARY KEY, memory_id TEXT NOT NULL,
                            operation TEXT NOT NULL, payload TEXT NOT NULL,
                            status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
                            error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                        )
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_index_outbox_status ON memory_index_outbox(status, created_at)")
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS memory_index_outbox (
                                event_id TEXT PRIMARY KEY, memory_id UUID NOT NULL,
                                operation TEXT NOT NULL, payload JSONB NOT NULL,
                                status TEXT NOT NULL DEFAULT 'queued', attempts INTEGER NOT NULL DEFAULT 0,
                                error TEXT, created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL
                            )
                        """)
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_index_outbox_status ON memory_index_outbox(status, created_at)")
        finally:
            conn.close()

    def enqueue(self, memory_id: str, operation: str, payload: Dict[str, Any]) -> str:
        from sage_memory import get_backend_type, sage_memory
        self.ensure_schema()
        event_id = f"idx_{uuid.uuid4().hex[:20]}"
        now = _now()
        encoded = json.dumps(payload, ensure_ascii=False, default=str)
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        "INSERT INTO memory_index_outbox(event_id,memory_id,operation,payload,created_at,updated_at) VALUES(?,?,?,?,?,?)",
                        (event_id, memory_id, operation, encoded, now, now),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "INSERT INTO memory_index_outbox(event_id,memory_id,operation,payload,created_at,updated_at) VALUES(%s,%s,%s,%s::jsonb,%s,%s)",
                            (event_id, memory_id, operation, encoded, now, now),
                        )
        finally:
            conn.close()
        self.process(event_id)
        return event_id

    def _set_status(self, event_id: str, status: str, error: str | None = None) -> None:
        from sage_memory import get_backend_type, sage_memory
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        "UPDATE memory_index_outbox SET status=?,attempts=attempts+1,error=?,updated_at=? WHERE event_id=?",
                        (status, error, _now(), event_id),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE memory_index_outbox SET status=%s,attempts=attempts+1,error=%s,updated_at=%s WHERE event_id=%s",
                            (status, error, _now(), event_id),
                        )
        finally:
            conn.close()

    def list(self, status: str | None = None, limit: int = 200) -> List[Dict[str, Any]]:
        from sage_memory import get_backend_type, sage_memory
        self.ensure_schema()
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        limit = max(1, min(int(limit), 1000))
        try:
            if backend == "sqlite":
                rows = (conn.execute("SELECT * FROM memory_index_outbox WHERE status=? ORDER BY created_at LIMIT ?", (status, limit)).fetchall()
                        if status else conn.execute("SELECT * FROM memory_index_outbox ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall())
            else:
                import psycopg2.extras
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if status:
                        cur.execute("SELECT * FROM memory_index_outbox WHERE status=%s ORDER BY created_at LIMIT %s", (status, limit))
                    else:
                        cur.execute("SELECT * FROM memory_index_outbox ORDER BY created_at DESC LIMIT %s", (limit,))
                    rows = cur.fetchall()
            result = []
            for row in rows:
                item = dict(row)
                if isinstance(item.get("payload"), str):
                    item["payload"] = json.loads(item["payload"])
                result.append(item)
            return result
        finally:
            conn.close()

    def process(self, event_id: str) -> bool:
        events = [event for event in self.list(limit=1000) if str(event.get("event_id")) == event_id]
        if not events:
            return False
        event = events[0]
        payload = dict(event.get("payload") or {})
        try:
            from sage_memory_index import memory_vector_index
            operation = str(event.get("operation"))
            if operation == "upsert":
                ok = memory_vector_index.index_memory(payload)
            elif operation == "status":
                ok = memory_vector_index.update_memory_status(str(event["memory_id"]), str(payload.get("status") or "active"))
            elif operation == "delete":
                ok = memory_vector_index.delete_memory_vector(str(event["memory_id"]), payload.get("tier"))
            else:
                raise ValueError(f"Unknown index operation: {operation}")
            if not ok:
                raise RuntimeError("Vector index operation returned false")
            self._set_status(event_id, "completed")
            return True
        except BaseException as exc:
            self._set_status(event_id, "queued", str(exc)[:2000])
            return False

    def recover(self, limit: int = 500) -> int:
        recovered = 0
        for event in self.list(status="queued", limit=limit):
            if self.process(str(event["event_id"])):
                recovered += 1
        return recovered


index_outbox = IndexOutbox()
