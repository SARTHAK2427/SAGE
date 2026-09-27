"""Durable curator-job ledger sharing SAGE's canonical database."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sage_memory import get_backend_type, sage_memory


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryJobStore:
    def __init__(self) -> None:
        self.ensure_schema()

    def ensure_schema(self) -> None:
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS memory_jobs (
                            job_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
                            user_id TEXT NOT NULL, chat_id TEXT NOT NULL, status TEXT NOT NULL,
                            payload TEXT NOT NULL, result TEXT, error TEXT,
                            attempts INTEGER NOT NULL DEFAULT 0,
                            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                        );
                    """)
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_jobs_status ON memory_jobs(status, created_at);")
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            CREATE TABLE IF NOT EXISTS memory_jobs (
                                job_id TEXT PRIMARY KEY, idempotency_key TEXT NOT NULL UNIQUE,
                                user_id TEXT NOT NULL, chat_id TEXT NOT NULL, status TEXT NOT NULL,
                                payload JSONB NOT NULL, result JSONB, error TEXT,
                                attempts INTEGER NOT NULL DEFAULT 0,
                                created_at TIMESTAMPTZ NOT NULL, updated_at TIMESTAMPTZ NOT NULL
                            );
                        """)
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_memory_jobs_status ON memory_jobs(status, created_at);")
        finally:
            conn.close()

    def create(self, *, key: str, user_id: str, chat_id: str, payload: Dict[str, Any]) -> Optional[str]:
        self.ensure_schema()
        job_id = f"memjob_{uuid.uuid4().hex[:16]}"
        now = _now()
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        try:
            try:
                if backend == "sqlite":
                    with conn:
                        conn.execute(
                            "INSERT INTO memory_jobs (job_id,idempotency_key,user_id,chat_id,status,payload,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?)",
                            (job_id, key, user_id, chat_id, "queued", json.dumps(payload, ensure_ascii=False), now, now),
                        )
                else:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "INSERT INTO memory_jobs (job_id,idempotency_key,user_id,chat_id,status,payload,created_at,updated_at) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s,%s)",
                                (job_id, key, user_id, chat_id, "queued", json.dumps(payload, ensure_ascii=False), now, now),
                            )
            except Exception as exc:
                if "unique" in str(exc).lower() or "duplicate" in str(exc).lower():
                    return None
                raise
        finally:
            conn.close()
        return job_id

    def update(self, job_id: str, status: str, *, result: Any = None, error: str | None = None) -> None:
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        now = _now()
        result_json = json.dumps(result, ensure_ascii=False) if result is not None else None
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute("UPDATE memory_jobs SET status=?,result=?,error=?,attempts=attempts+1,updated_at=? WHERE job_id=?",
                                 (status, result_json, error, now, job_id))
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE memory_jobs SET status=%s,result=%s::jsonb,error=%s,attempts=attempts+1,updated_at=%s WHERE job_id=%s",
                                    (status, result_json, error, now, job_id))
        finally:
            conn.close()

    def list(self, status: str | None = None, limit: int = 100) -> List[Dict[str, Any]]:
        self.ensure_schema()
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        limit = max(1, min(int(limit), 500))
        try:
            if backend == "sqlite":
                rows = (conn.execute("SELECT * FROM memory_jobs WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)).fetchall()
                        if status else conn.execute("SELECT * FROM memory_jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall())
            else:
                import psycopg2.extras
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if status:
                        cur.execute("SELECT * FROM memory_jobs WHERE status=%s ORDER BY created_at DESC LIMIT %s", (status, limit))
                    else:
                        cur.execute("SELECT * FROM memory_jobs ORDER BY created_at DESC LIMIT %s", (limit,))
                    rows = cur.fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for field in ("payload", "result"):
                    if isinstance(item.get(field), str):
                        try:
                            item[field] = json.loads(item[field])
                        except json.JSONDecodeError:
                            pass
                result.append(item)
            return result
        finally:
            conn.close()

    def cancel_chat(self, chat_id: str) -> int:
        """Cancel work that must not recreate memory after a chat reset."""
        self.ensure_schema()
        backend = get_backend_type()
        conn = sage_memory._get_connection()
        now = _now()
        try:
            if backend == "sqlite":
                with conn:
                    cur = conn.execute(
                        "UPDATE memory_jobs SET status='cancelled',updated_at=? WHERE chat_id=? AND status IN ('queued','running')",
                        (now, chat_id),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE memory_jobs SET status='cancelled',updated_at=%s WHERE chat_id=%s AND status IN ('queued','running')",
                            (now, chat_id),
                        )
            return int(cur.rowcount or 0)
        finally:
            conn.close()


memory_job_store = MemoryJobStore()
