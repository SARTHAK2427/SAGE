"""
SAGE/sage_memory.py
Phase 1: Chat Continuity + Conversation Ledger
Phase 2: Global Facts / Durable Memory

Handles:
1. Persistent conversation message storage in PostgreSQL/SQLite (messages table).
2. Recent conversation window retrieval for orchestrator context injection.
3. Persistent durable memory store for global user facts (memories table).
"""

from __future__ import annotations
import os
import sqlite3
import uuid
import logging
from typing import Any, List, Dict, Optional
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)

# Controlled categories for durable global facts (Phase 2 & Phase 7)
ALLOWED_CATEGORIES = {"fact", "preference", "project", "decision", "instruction", "task", "personal", "technical", "summary"}
ALLOWED_STATUSES = {"active", "superseded", "deleted", "promoted", "compacted"}
ALLOWED_TIERS = {"hot", "cold"}

# Module-level connection configuration override for testing
_sqlite_path_override: Optional[str] = None


def set_sqlite_path(path: str | None) -> None:
    """Override SQLite DB path (useful for unit tests)."""
    global _sqlite_path_override
    _sqlite_path_override = path


def get_backend_type() -> str:
    """Return configured backend type: 'postgres' or 'sqlite'."""
    db_env = os.environ.get("SAGE_MEMORY_DB", config.SAGE_MEMORY_DB).lower().strip()
    return "sqlite" if db_env == "sqlite" else "postgres"


def _get_sqlite_db_path() -> str:
    if _sqlite_path_override:
        return _sqlite_path_override
    env_path = os.environ.get("SAGE_SQLITE_PATH")
    if env_path:
        return env_path
    db_file = config.MEMORY_ROOT / "sage_memory.db"
    config.MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    return str(db_file)


def validate_category(category: str) -> str:
    """Validate and normalize memory category string."""
    cat = (category or "").lower().strip()
    if cat not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Allowed categories: {sorted(list(ALLOWED_CATEGORIES))}"
        )
    return cat


def _sync_index_store(mem: Dict[str, Any]) -> None:
    try:
        from memory_system.index_outbox import index_outbox
        index_outbox.enqueue(str(mem.get("memory_id") or ""), "upsert", mem)
    except Exception as exc:
        logger.error("Chroma memory index sync failed on store: %s", exc)


def _sync_index_status(memory_id: str, status: str) -> None:
    try:
        from memory_system.index_outbox import index_outbox
        index_outbox.enqueue(memory_id, "status", {"status": status})
    except Exception as exc:
        logger.error("Chroma memory status sync failed: %s", exc)


def _sync_index_delete(memory_id: str, tier: Optional[str] = None) -> None:
    try:
        from memory_system.index_outbox import index_outbox
        index_outbox.enqueue(memory_id, "delete", {"tier": tier})
    except Exception as exc:
        logger.error("Chroma memory delete sync failed: %s", exc)


class SageMemory:
    """Conversation ledger and durable memory manager for PostgreSQL and SQLite backends."""

    def __init__(self) -> None:
        self._initialized_backends: set[str] = set()
        self._initialized_sqlite_path: Optional[str] = None

    def _get_connection(self) -> Any:
        backend = get_backend_type()

        if backend == "sqlite":
            db_path = _get_sqlite_db_path()
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            if self._initialized_sqlite_path != db_path:
                self._init_sqlite_schema(conn)
                self._initialized_sqlite_path = db_path
                self._initialized_backends.add("sqlite")
            return conn

        elif backend == "postgres":
            db_url = (
                os.environ.get("SAGE_DATABASE_URL")
                or os.environ.get("DATABASE_URL")
                or config.SAGE_DATABASE_URL
            )
            if not db_url or not str(db_url).strip():
                raise RuntimeError(
                    "SAGE_MEMORY_DB is set to 'postgres', but no database URL is configured. "
                    "Set SAGE_DATABASE_URL or DATABASE_URL. For local SAGE, use SAGE_MEMORY_DB=sqlite."
                )
            try:
                import psycopg2
                import psycopg2.extras
            except ImportError as exc:
                raise RuntimeError(
                    "SAGE_MEMORY_DB is set to 'postgres', but psycopg2 is not installed. "
                    "For local SAGE memory set SAGE_MEMORY_DB=sqlite."
                ) from exc

            try:
                conn = psycopg2.connect(db_url)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to connect to PostgreSQL. "
                    "Verify PostgreSQL is running and SAGE_DATABASE_URL/DATABASE_URL is valid."
                ) from exc

            if "postgres" not in self._initialized_backends:
                self._init_postgres_schema(conn)
                self._initialized_backends.add("postgres")
            return conn

        else:
            raise ValueError(f"Unsupported SAGE_MEMORY_DB backend: '{backend}'")

    def _init_sqlite_schema(self, conn: sqlite3.Connection) -> None:
        with conn:
            # Phase 1: Conversation messages table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    msg_id TEXT PRIMARY KEY,
                    chat_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_chat
                ON messages(chat_id, created_at);
                """
            )

            # Phase 2 & Phase 7: Global facts / durable memories table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    category TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source_chat_id TEXT,
                    source_msg_id TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    supersedes_memory_id TEXT,
                    memory_tier TEXT NOT NULL DEFAULT 'cold',
                    last_accessed_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Additive migration for existing SQLite databases
            pragma_cur = conn.execute("PRAGMA table_info(memories);")
            existing_cols = {row[1] for row in pragma_cur.fetchall()}
            if "memory_tier" not in existing_cols:
                conn.execute("ALTER TABLE memories ADD COLUMN memory_tier TEXT NOT NULL DEFAULT 'cold';")
            if "last_accessed_at" not in existing_cols:
                conn.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;")

            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_status
                ON memories(user_id, status);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_category
                ON memories(user_id, category);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_user_tier_status
                ON memories(user_id, memory_tier, status);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_memories_chat_tier
                ON memories(source_chat_id, memory_tier);
                """
            )

            # Memory Activity ledger for operational tracking
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_activity (
                    activity_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    chat_id TEXT,
                    memory_id TEXT,
                    details TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_activity_user_created
                ON memory_activity(user_id, created_at DESC);
                """
            )

    def _init_postgres_schema(self, conn: Any) -> None:
        with conn:
            with conn.cursor() as cur:
                # Phase 1: Conversation messages table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS messages (
                        msg_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        chat_id TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_messages_chat
                    ON messages(chat_id, created_at);
                    """
                )

                # Phase 2 & Phase 7: Global facts / durable memories table
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        memory_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        user_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        category TEXT NOT NULL,
                        importance REAL NOT NULL DEFAULT 0.5,
                        confidence REAL NOT NULL DEFAULT 1.0,
                        source_chat_id TEXT,
                        source_msg_id UUID,
                        status TEXT NOT NULL DEFAULT 'active',
                        supersedes_memory_id UUID,
                        memory_tier TEXT NOT NULL DEFAULT 'cold',
                        last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                # Additive migration for existing Postgres databases
                cur.execute(
                    """
                    ALTER TABLE memories ADD COLUMN IF NOT EXISTS memory_tier TEXT NOT NULL DEFAULT 'cold';
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE memories ADD COLUMN IF NOT EXISTS last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT now();
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_user_status
                    ON memories(user_id, status);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_user_category
                    ON memories(user_id, category);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_user_tier_status
                    ON memories(user_id, memory_tier, status);
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_memories_chat_tier
                    ON memories(source_chat_id, memory_tier);
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memory_activity (
                        activity_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                        event_type TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        chat_id TEXT,
                        memory_id TEXT,
                        details TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_activity_user_created
                    ON memory_activity(user_id, created_at DESC);
                    """
                )

    # ── Phase 1: Conversation Ledger Operations ─────────────────────────────

    def write_message(
        self,
        chat_id: str,
        user_id: str,
        role: str,
        content: str,
    ) -> Optional[str]:
        """Persist a conversation message to the ledger."""
        if not chat_id or not content or not content.strip():
            return None

        backend = get_backend_type()
        msg_id = str(uuid.uuid4())
        created_at_iso = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        """
                        INSERT INTO messages (msg_id, chat_id, user_id, role, content, created_at)
                        VALUES (?, ?, ?, ?, ?, ?);
                        """,
                        (msg_id, chat_id, user_id, role, content.strip(), created_at_iso),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO messages (msg_id, chat_id, user_id, role, content, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s);
                            """,
                            (msg_id, chat_id, user_id, role, content.strip(), created_at_iso),
                        )
        finally:
            conn.close()
        return msg_id

    def get_messages(self, chat_id: str) -> List[Dict[str, Any]]:
        """Retrieve all messages for a chat_id in chronological order."""
        if not chat_id:
            return []

        backend = get_backend_type()
        conn = self._get_connection()
        results: List[Dict[str, Any]] = []

        try:
            if backend == "sqlite":
                cursor = conn.execute(
                    """
                    SELECT msg_id, chat_id, user_id, role, content, created_at
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY created_at ASC;
                    """,
                    (chat_id,),
                )
                rows = cursor.fetchall()
                for r in rows:
                    results.append({
                        "msg_id": str(r["msg_id"]),
                        "chat_id": str(r["chat_id"]),
                        "user_id": str(r["user_id"]),
                        "role": str(r["role"]),
                        "content": str(r["content"]),
                        "created_at": str(r["created_at"]),
                    })
            else:
                import psycopg2.extras
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT msg_id, chat_id, user_id, role, content, created_at
                        FROM messages
                        WHERE chat_id = %s
                        ORDER BY created_at ASC;
                        """,
                        (chat_id,),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        results.append({
                            "msg_id": str(r["msg_id"]),
                            "chat_id": str(r["chat_id"]),
                            "user_id": str(r["user_id"]),
                            "role": str(r["role"]),
                            "content": str(r["content"]),
                            "created_at": str(r["created_at"]),
                        })
        finally:
            conn.close()

        return results

    def get_recent_window(self, chat_id: str, n: int = 5) -> str:
        """Retrieve newest `n` messages for `chat_id` and format chronologically."""
        if not chat_id or n <= 0:
            return ""

        backend = get_backend_type()
        conn = self._get_connection()
        rows_newest_first: List[Dict[str, Any]] = []

        try:
            if backend == "sqlite":
                cursor = conn.execute(
                    """
                    SELECT role, content
                    FROM messages
                    WHERE chat_id = ?
                    ORDER BY created_at DESC, msg_id DESC
                    LIMIT ?;
                    """,
                    (chat_id, n),
                )
                rows = cursor.fetchall()
                for r in rows:
                    rows_newest_first.append({"role": r["role"], "content": r["content"]})
            else:
                import psycopg2.extras
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT role, content
                        FROM messages
                        WHERE chat_id = %s
                        ORDER BY created_at DESC, msg_id DESC
                        LIMIT %s;
                        """,
                        (chat_id, n),
                    )
                    rows = cur.fetchall()
                    for r in rows:
                        rows_newest_first.append({"role": r["role"], "content": r["content"]})
        finally:
            conn.close()

        if not rows_newest_first:
            return ""

        rows_chronological = list(reversed(rows_newest_first))

        lines = ["RECENT CONVERSATION:"]
        for msg in rows_chronological:
            lines.append(f"{msg['role']}: {msg['content']}")

        return "\n".join(lines)

    def list_chats(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """List conversations for a user, newest activity first."""
        if not user_id:
            return []
        limit = max(1, min(int(limit or 100), 500))
        backend = get_backend_type()
        conn = self._get_connection()
        results: List[Dict[str, Any]] = []
        try:
            if backend == "sqlite":
                cursor = conn.execute(
                    """
                    SELECT
                        m.chat_id AS chat_id,
                        m.user_id AS user_id,
                        MIN(m.created_at) AS created_at,
                        MAX(m.created_at) AS updated_at,
                        COUNT(*) AS message_count,
                        (
                            SELECT content FROM messages t
                            WHERE t.chat_id = m.chat_id AND t.role = 'user'
                            ORDER BY t.created_at ASC, t.msg_id ASC
                            LIMIT 1
                        ) AS title
                    FROM messages m
                    WHERE m.user_id = ?
                    GROUP BY m.chat_id, m.user_id
                    ORDER BY updated_at DESC
                    LIMIT ?;
                    """,
                    (user_id, limit),
                )
                for row in cursor.fetchall():
                    title = (row["title"] or "New conversation").strip()
                    if len(title) > 72:
                        title = title[:69].rstrip() + "…"
                    results.append({
                        "chat_id": str(row["chat_id"]),
                        "user_id": str(row["user_id"]),
                        "title": title or "New conversation",
                        "created_at": str(row["created_at"]),
                        "updated_at": str(row["updated_at"]),
                        "message_count": int(row["message_count"] or 0),
                    })
            else:
                import psycopg2.extras
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT
                            m.chat_id AS chat_id,
                            m.user_id AS user_id,
                            MIN(m.created_at) AS created_at,
                            MAX(m.created_at) AS updated_at,
                            COUNT(*) AS message_count,
                            (
                                SELECT content FROM messages t
                                WHERE t.chat_id = m.chat_id AND t.role = 'user'
                                ORDER BY t.created_at ASC, t.msg_id ASC
                                LIMIT 1
                            ) AS title
                        FROM messages m
                        WHERE m.user_id = %s
                        GROUP BY m.chat_id, m.user_id
                        ORDER BY updated_at DESC
                        LIMIT %s;
                        """,
                        (user_id, limit),
                    )
                    for row in cur.fetchall():
                        title = (row["title"] or "New conversation").strip()
                        if len(title) > 72:
                            title = title[:69].rstrip() + "…"
                        results.append({
                            "chat_id": str(row["chat_id"]),
                            "user_id": str(row["user_id"]),
                            "title": title or "New conversation",
                            "created_at": str(row["created_at"]),
                            "updated_at": str(row["updated_at"]),
                            "message_count": int(row["message_count"] or 0),
                        })
        finally:
            conn.close()
        return results

    def clear_chat(self, chat_id: str, user_id: Optional[str] = None) -> Dict[str, int]:
        """Delete canonical messages and every chat-scoped memory for one chat."""
        if not chat_id:
            return {"messages_removed": 0, "memories_removed": 0}
        # Capture derived-index IDs before the canonical rows disappear.
        indexed_memories = self.list_memories(
            user_id=user_id or config.DEFAULT_USER_ID,
            chat_id=chat_id,
            status="active",
            limit=500,
        )
        backend = get_backend_type()
        conn = self._get_connection()
        messages_removed = 0
        memories_removed = 0
        try:
            if backend == "sqlite":
                with conn:
                    if user_id:
                        cur = conn.execute(
                            "DELETE FROM messages WHERE chat_id = ? AND user_id = ?;",
                            (chat_id, user_id),
                        )
                    else:
                        cur = conn.execute(
                            "DELETE FROM messages WHERE chat_id = ?;",
                            (chat_id,),
                        )
                    messages_removed = int(cur.rowcount or 0)
                    if user_id:
                        cur = conn.execute(
                            """
                            DELETE FROM memories
                            WHERE source_chat_id = ? AND user_id = ?;
                            """,
                            (chat_id, user_id),
                        )
                    else:
                        cur = conn.execute(
                            """
                            DELETE FROM memories
                            WHERE source_chat_id = ?;
                            """,
                            (chat_id,),
                        )
                    memories_removed = int(cur.rowcount or 0)
            else:
                with conn:
                    with conn.cursor() as cur:
                        if user_id:
                            cur.execute(
                                "DELETE FROM messages WHERE chat_id = %s AND user_id = %s;",
                                (chat_id, user_id),
                            )
                        else:
                            cur.execute(
                                "DELETE FROM messages WHERE chat_id = %s;",
                                (chat_id,),
                            )
                        messages_removed = int(cur.rowcount or 0)
                        if user_id:
                            cur.execute(
                                """
                                DELETE FROM memories
                                WHERE source_chat_id = %s AND user_id = %s;
                                """,
                                (chat_id, user_id),
                            )
                        else:
                            cur.execute(
                                """
                                DELETE FROM memories
                                WHERE source_chat_id = %s;
                                """,
                                (chat_id,),
                            )
                        memories_removed = int(cur.rowcount or 0)
        finally:
            conn.close()
        for memory in indexed_memories:
            _sync_index_delete(
                str(memory.get("memory_id") or ""),
                str(memory.get("memory_tier") or "cold"),
            )
        return {
            "messages_removed": messages_removed,
            "memories_removed": memories_removed,
        }

    # ── Phase 2 & Phase 7: Global Facts / Durable Memory Operations ─────────

    def touch_memory_access(self, memory_id: str) -> bool:
        """Update last_accessed_at timestamp when a memory is retrieved."""
        if not memory_id:
            return False
        now_iso = datetime.now(timezone.utc).isoformat()
        backend = get_backend_type()
        conn = self._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        "UPDATE memories SET last_accessed_at = ? WHERE memory_id = ?;",
                        (now_iso, memory_id),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE memories SET last_accessed_at = %s WHERE memory_id = %s;",
                            (now_iso, memory_id),
                        )
            return True
        except Exception as exc:
            logger.error("Failed to touch memory access for %s: %s", memory_id, exc)
            return False
        finally:
            conn.close()

    def store_memory(
        self,
        user_id: str,
        content: str,
        category: str = "fact",
        importance: float = 0.5,
        confidence: float = 1.0,
        source_chat_id: Optional[str] = None,
        source_msg_id: Optional[str] = None,
        supersedes_memory_id: Optional[str] = None,
        memory_tier: str = "cold",
    ) -> Dict[str, Any]:
        """Store a new durable memory fact (hot or cold)."""
        if not user_id or not user_id.strip():
            raise ValueError("user_id is required for storing memory.")
        if not content or not content.strip():
            raise ValueError("content is required for storing memory.")

        cat = validate_category(category)
        imp = max(0.0, min(1.0, float(importance)))
        conf = max(0.0, min(1.0, float(confidence)))
        tier = (memory_tier or "cold").lower().strip()
        if tier not in ALLOWED_TIERS:
            raise ValueError(f"Invalid memory_tier '{memory_tier}'. Allowed tiers: {sorted(list(ALLOWED_TIERS))}")
        if tier == "hot" and not (source_chat_id and str(source_chat_id).strip()):
            raise ValueError("source_chat_id is required for session-scoped hot memory.")

        memory_id = str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()
        backend = get_backend_type()

        conn = self._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        """
                        INSERT INTO memories (
                            memory_id, user_id, content, category, importance, confidence,
                            source_chat_id, source_msg_id, status, supersedes_memory_id,
                            memory_tier, last_accessed_at, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            memory_id, user_id.strip(), content.strip(), cat, imp, conf,
                            source_chat_id, source_msg_id, "active", supersedes_memory_id,
                            tier, now_iso, now_iso, now_iso,
                        ),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO memories (
                                memory_id, user_id, content, category, importance, confidence,
                                source_chat_id, source_msg_id, status, supersedes_memory_id,
                                memory_tier, last_accessed_at, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                            """,
                            (
                                memory_id, user_id.strip(), content.strip(), cat, imp, conf,
                                source_chat_id, source_msg_id, "active", supersedes_memory_id,
                                tier, now_iso, now_iso, now_iso,
                            ),
                        )
        finally:
            conn.close()

        res = {
            "memory_id": memory_id,
            "user_id": user_id.strip(),
            "content": content.strip(),
            "category": cat,
            "importance": imp,
            "confidence": conf,
            "source_chat_id": source_chat_id,
            "source_msg_id": source_msg_id,
            "status": "active",
            "supersedes_memory_id": supersedes_memory_id,
            "memory_tier": tier,
            "last_accessed_at": now_iso,
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        _sync_index_store(res)
        self.log_activity("STORE", user_id=res["user_id"], memory_id=res["memory_id"], chat_id=res.get("source_chat_id"), details=f"Tier: {res['memory_tier']}, Category: {res['category']}")
        return res

    def get_memory(self, memory_id: str, touch_access: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch a single memory object by memory_id."""
        if not memory_id:
            return None

        backend = get_backend_type()
        conn = self._get_connection()

        try:
            if backend == "sqlite":
                cursor = conn.execute(
                    """
                    SELECT memory_id, user_id, content, category, importance, confidence,
                           source_chat_id, source_msg_id, status, supersedes_memory_id,
                           memory_tier, last_accessed_at, created_at, updated_at
                    FROM memories
                    WHERE memory_id = ?;
                    """,
                    (memory_id,),
                )
                r = cursor.fetchone()
                if not r:
                    return None
                col_keys = r.keys() if hasattr(r, "keys") else []
                res = {
                    "memory_id": str(r["memory_id"]),
                    "user_id": str(r["user_id"]),
                    "content": str(r["content"]),
                    "category": str(r["category"]),
                    "importance": float(r["importance"]),
                    "confidence": float(r["confidence"]),
                    "source_chat_id": r["source_chat_id"],
                    "source_msg_id": r["source_msg_id"],
                    "status": str(r["status"]),
                    "supersedes_memory_id": r["supersedes_memory_id"],
                    "memory_tier": str(r["memory_tier"] or "cold") if "memory_tier" in col_keys else "cold",
                    "last_accessed_at": str(r["last_accessed_at"] or r["created_at"]) if "last_accessed_at" in col_keys else str(r["created_at"]),
                    "created_at": str(r["created_at"]),
                    "updated_at": str(r["updated_at"]),
                }
            else:
                import psycopg2.extras
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(
                        """
                        SELECT memory_id, user_id, content, category, importance, confidence,
                               source_chat_id, source_msg_id, status, supersedes_memory_id,
                               memory_tier, last_accessed_at, created_at, updated_at
                        FROM memories
                        WHERE memory_id = %s;
                        """,
                        (memory_id,),
                    )
                    r = cur.fetchone()
                    if not r:
                        return None
                    col_keys = r.keys() if hasattr(r, "keys") else []
                    res = {
                        "memory_id": str(r["memory_id"]),
                        "user_id": str(r["user_id"]),
                        "content": str(r["content"]),
                        "category": str(r["category"]),
                        "importance": float(r["importance"]),
                        "confidence": float(r["confidence"]),
                        "source_chat_id": r["source_chat_id"],
                        "source_msg_id": r["source_msg_id"],
                        "status": str(r["status"]),
                        "supersedes_memory_id": r["supersedes_memory_id"],
                        "memory_tier": str(r["memory_tier"] or "cold") if "memory_tier" in col_keys else "cold",
                        "last_accessed_at": str(r["last_accessed_at"] or r["created_at"]) if "last_accessed_at" in col_keys else str(r["created_at"]),
                        "created_at": str(r["created_at"]),
                        "updated_at": str(r["updated_at"]),
                    }
        finally:
            conn.close()

        if touch_access and res:
            self.touch_memory_access(memory_id)
        return res

    def list_memories(
        self,
        user_id: str,
        category: Optional[str] = None,
        memory_tier: Optional[str] = None,
        chat_id: Optional[str] = None,
        status: Optional[str] = "active",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List memories for user_id with optional category, tier, status, and chat_id filters.

        Ordered deterministically by importance DESC, updated_at DESC, memory_id DESC.
        """
        if not user_id or not user_id.strip():
            return []

        cat_filter = validate_category(category) if category else None
        tier_filter = memory_tier.lower().strip() if memory_tier else None
        if tier_filter and tier_filter not in ALLOWED_TIERS:
            raise ValueError(f"Invalid memory_tier '{memory_tier}'. Allowed tiers: {sorted(list(ALLOWED_TIERS))}")
        bounded_limit = max(1, min(limit, 500))

        backend = get_backend_type()
        conn = self._get_connection()
        results: List[Dict[str, Any]] = []

        try:
            base_sql = """
                SELECT memory_id, user_id, content, category, importance, confidence,
                       source_chat_id, source_msg_id, status, supersedes_memory_id,
                       memory_tier, last_accessed_at, created_at, updated_at
                FROM memories
                WHERE user_id = {ph}
            """
            params: List[Any] = [user_id.strip()]

            if status:
                base_sql += " AND status = {ph}"
                params.append(status.strip().lower())

            if cat_filter:
                base_sql += " AND category = {ph}"
                params.append(cat_filter)
            if tier_filter:
                base_sql += " AND memory_tier = {ph}"
                params.append(tier_filter)
            if chat_id:
                base_sql += " AND source_chat_id = {ph}"
                params.append(chat_id)

            base_sql += " ORDER BY importance DESC, updated_at DESC, memory_id DESC LIMIT {ph};"
            params.append(bounded_limit)

            if backend == "sqlite":
                sql = base_sql.format(ph="?")
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
            else:
                import psycopg2.extras
                sql = base_sql.format(ph="%s")
                with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                    cur.execute(sql, params)
                    rows = cur.fetchall()

            for r in rows:
                col_keys = r.keys() if hasattr(r, "keys") else []
                results.append({
                    "memory_id": str(r["memory_id"]),
                    "user_id": str(r["user_id"]),
                    "content": str(r["content"]),
                    "category": str(r["category"]),
                    "importance": float(r["importance"]),
                    "confidence": float(r["confidence"]),
                    "source_chat_id": r["source_chat_id"],
                    "source_msg_id": r["source_msg_id"],
                    "status": str(r["status"]),
                    "supersedes_memory_id": r["supersedes_memory_id"],
                    "memory_tier": str(r["memory_tier"] or "cold") if "memory_tier" in col_keys else "cold",
                    "last_accessed_at": str(r["last_accessed_at"] or r["created_at"]) if "last_accessed_at" in col_keys else str(r["created_at"]),
                    "created_at": str(r["created_at"]),
                    "updated_at": str(r["updated_at"]),
                })
        finally:
            conn.close()

        return results

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[float] = None,
        confidence: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update fields of an existing memory and set updated_at timestamp."""
        existing = self.get_memory(memory_id)
        if not existing:
            return None

        new_content = content.strip() if content is not None and content.strip() else existing["content"]
        new_category = validate_category(category) if category is not None else existing["category"]
        new_importance = max(0.0, min(1.0, float(importance))) if importance is not None else existing["importance"]
        new_confidence = max(0.0, min(1.0, float(confidence))) if confidence is not None else existing["confidence"]
        now_iso = datetime.now(timezone.utc).isoformat()

        backend = get_backend_type()
        conn = self._get_connection()

        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        """
                        UPDATE memories
                        SET content = ?, category = ?, importance = ?, confidence = ?, updated_at = ?
                        WHERE memory_id = ?;
                        """,
                        (new_content, new_category, new_importance, new_confidence, now_iso, memory_id),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE memories
                            SET content = %s, category = %s, importance = %s, confidence = %s, updated_at = %s
                            WHERE memory_id = %s;
                            """,
                            (new_content, new_category, new_importance, new_confidence, now_iso, memory_id),
                        )
        finally:
            conn.close()

        updated = self.get_memory(memory_id)
        if updated:
            _sync_index_store(updated)
            self.log_activity("UPDATE", user_id=existing["user_id"], memory_id=memory_id, chat_id=existing.get("source_chat_id"), details=f"Category: {new_category}")
        return updated

    def delete_memory(self, memory_id: str) -> bool:
        """Logically delete a memory by setting status='deleted'."""
        existing = self.get_memory(memory_id)
        if not existing:
            return False

        now_iso = datetime.now(timezone.utc).isoformat()
        backend = get_backend_type()
        conn = self._get_connection()

        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        """
                        UPDATE memories
                        SET status = 'deleted', updated_at = ?
                        WHERE memory_id = ?;
                        """,
                        (now_iso, memory_id),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE memories
                            SET status = 'deleted', updated_at = %s
                            WHERE memory_id = %s;
                            """,
                            (now_iso, memory_id),
                        )
        finally:
            conn.close()

        _sync_index_status(memory_id, "deleted")
        self.log_activity("DELETE", user_id=existing["user_id"], memory_id=memory_id, chat_id=existing.get("source_chat_id"), details="Marked deleted")
        return True

    def hard_delete_memory(self, memory_id: str) -> bool:
        """Permanently remove one memory record and its vector representation."""
        existing = self.get_memory(memory_id)
        if not existing:
            return False
        backend = get_backend_type()
        conn = self._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute("DELETE FROM memories WHERE memory_id = ?;", (memory_id,))
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute("DELETE FROM memories WHERE memory_id = %s;", (memory_id,))
        finally:
            conn.close()
        _sync_index_delete(memory_id, existing.get("memory_tier"))
        return True

    def supersede_memory(
        self,
        old_memory_id: str,
        new_content: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        source_chat_id: Optional[str] = None,
        source_msg_id: Optional[str] = None,
        memory_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Supersede an existing memory with a new memory.

        Direction: The new memory's `supersedes_memory_id` points to `old_memory_id`.
        The old memory's status is set to `'superseded'`.
        """
        old_mem = self.get_memory(old_memory_id)
        if not old_mem:
            raise ValueError(f"Memory with ID '{old_memory_id}' not found for supersession.")

        target_user_id = user_id or old_mem["user_id"]
        target_category = category or old_mem["category"]
        tier = memory_tier or old_mem.get("memory_tier", "cold")

        # 1. Store new memory referencing old_memory_id
        new_mem = self.store_memory(
            user_id=target_user_id,
            content=new_content,
            category=target_category,
            importance=importance,
            confidence=confidence,
            source_chat_id=source_chat_id,
            source_msg_id=source_msg_id,
            supersedes_memory_id=old_memory_id,
            memory_tier=tier,
        )

        # 2. Mark old memory as superseded
        now_iso = datetime.now(timezone.utc).isoformat()
        backend = get_backend_type()
        conn = self._get_connection()

        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        """
                        UPDATE memories
                        SET status = 'superseded', updated_at = ?
                        WHERE memory_id = ?;
                        """,
                        (now_iso, old_memory_id),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE memories
                            SET status = 'superseded', updated_at = %s
                            WHERE memory_id = %s;
                            """,
                            (now_iso, old_memory_id),
                        )
        finally:
            conn.close()

        _sync_index_status(old_memory_id, "superseded")
        return new_mem

    def promote_memory(self, memory_id: str, user_id: str) -> Dict[str, Any]:
        """Promote a hot memory to cold memory in place.

        Lifecycle:
        1. Source memory must exist, belong to user_id, be active, and have tier=='hot'.
        2. Atomically update canonical record: memory_tier becomes 'cold' (status remains 'active').
        3. Sync Chroma (removed from hot collection, indexed into cold collection).
        """
        if not memory_id or not user_id:
            raise ValueError("memory_id and user_id are required for promotion.")

        source = self.get_memory(memory_id)
        if not source:
            raise ValueError(f"Memory with ID '{memory_id}' not found for promotion.")

        if source["user_id"] != user_id:
            raise PermissionError(f"Memory '{memory_id}' does not belong to active user.")

        if source["status"] != "active":
            raise ValueError(f"Memory '{memory_id}' is not active (current status: '{source['status']}'). Cannot promote inactive memory.")

        if source.get("memory_tier") != "hot":
            raise ValueError(f"Memory '{memory_id}' is tier '{source.get('memory_tier')}'. Only 'hot' memories can be promoted to cold.")

        # Update canonical database record: memory_tier becomes 'cold', status remains 'active'
        now_iso = datetime.now(timezone.utc).isoformat()
        backend = get_backend_type()
        conn = self._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    conn.execute(
                        "UPDATE memories SET memory_tier = 'cold', updated_at = ? WHERE memory_id = ?;",
                        (now_iso, memory_id),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE memories SET memory_tier = 'cold', updated_at = %s WHERE memory_id = %s;",
                            (now_iso, memory_id),
                        )
        finally:
            conn.close()

        updated = self.get_memory(memory_id)
        if not updated:
            raise RuntimeError(f"Failed to retrieve memory '{memory_id}' after promotion.")

        _sync_index_store(updated)
        self.log_activity("PROMOTE", user_id=user_id, memory_id=memory_id, chat_id=updated.get("source_chat_id"), details="Promoted hot to cold")
        return updated

    def compact_memories(
        self,
        memory_ids: List[str],
        summary_content: str,
        user_id: str,
        chat_id: Optional[str] = None,
        category: str = "fact",
        importance: float = 0.5,
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """Compact multiple related hot memories into a single hot summary memory.

        Loss-aware order:
        1. Validate all source IDs (exist, belong to user_id, status=='active', tier=='hot', match chat_id).
        2. Persist summary hot memory first.
        3. Only after summary persistence succeeds, update source memories to status='compacted'.
        4. Sync Chroma.
        """
        if not memory_ids or not isinstance(memory_ids, (list, tuple)):
            raise ValueError("memory_ids must be a non-empty list of IDs.")
        if not summary_content or not summary_content.strip():
            raise ValueError("summary_content must be a non-empty string.")
        if not user_id:
            raise ValueError("user_id is required.")

        unique_ids = list(dict.fromkeys(memory_ids))
        if len(unique_ids) < 1:
            raise ValueError("At least one memory ID is required for compaction.")

        # Step 1: Fetch and validate all source memories
        expected_chat_id = chat_id
        for mid in unique_ids:
            mem = self.get_memory(mid)
            if not mem:
                raise ValueError(f"Source memory '{mid}' not found for compaction.")
            if mem["user_id"] != user_id:
                raise PermissionError(f"Source memory '{mid}' does not belong to active user.")
            if mem["status"] != "active":
                raise ValueError(f"Source memory '{mid}' is not active (status: '{mem['status']}').")
            if mem.get("memory_tier") != "hot":
                raise ValueError(f"Source memory '{mid}' is tier '{mem.get('memory_tier')}'. Only hot memories can be compacted.")

            mem_chat = mem.get("source_chat_id")
            if expected_chat_id is None:
                expected_chat_id = mem_chat
            elif mem_chat != expected_chat_id:
                raise ValueError(
                    f"Source memory '{mid}' chat_id '{mem_chat}' does not match expected chat_id '{expected_chat_id}'. "
                    "Compaction across different sessions is not permitted."
                )

        # Step 2: Persist summary hot memory first (loss-aware: if this fails, sources remain active)
        summary_mem = self.store_memory(
            user_id=user_id,
            content=summary_content.strip(),
            category=category,
            importance=importance,
            confidence=confidence,
            source_chat_id=expected_chat_id,
            memory_tier="hot",
        )

        # Step 3: Deactivate source memories
        now_iso = datetime.now(timezone.utc).isoformat()
        backend = get_backend_type()
        conn = self._get_connection()
        try:
            if backend == "sqlite":
                with conn:
                    placeholders = ",".join("?" for _ in unique_ids)
                    conn.execute(
                        f"UPDATE memories SET status = 'compacted', updated_at = ? WHERE memory_id IN ({placeholders});",
                        [now_iso] + unique_ids,
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        placeholders = ",".join("%s" for _ in unique_ids)
                        cur.execute(
                            f"UPDATE memories SET status = 'compacted', updated_at = %s WHERE memory_id IN ({placeholders});",
                            [now_iso] + unique_ids,
                        )
        finally:
            conn.close()

        for mid in unique_ids:
            _sync_index_status(mid, "compacted")

        self.log_activity("SUMMARIZE", user_id=user_id, memory_id=summary_mem["memory_id"], chat_id=expected_chat_id, details=f"Compacted {len(unique_ids)} memories")

        return summary_mem

    def log_activity(
        self,
        event_type: str,
        user_id: str,
        memory_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        details: Optional[str] = None,
    ) -> None:
        """Log an operational memory event (STORE, SEARCH, UPDATE, DELETE, PROMOTE, SUMMARIZE)."""
        if not user_id:
            user_id = config.DEFAULT_USER_ID
        event_type = (event_type or "EVENT").upper().strip()
        backend = get_backend_type()
        conn = self._get_connection()
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            if backend == "sqlite":
                act_id = f"act_{uuid.uuid4().hex[:12]}"
                with conn:
                    conn.execute(
                        """
                        INSERT INTO memory_activity (activity_id, event_type, user_id, chat_id, memory_id, details, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        (act_id, event_type, user_id, chat_id, memory_id, details, now_iso),
                    )
            else:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO memory_activity (event_type, user_id, chat_id, memory_id, details, created_at)
                            VALUES (%s, %s, %s, %s, %s, %s);
                            """,
                            (event_type, user_id, chat_id, memory_id, details, now_iso),
                        )
        except Exception as exc:
            logger.warning("Failed to log memory activity: %s", exc)
        finally:
            conn.close()

    def get_recent_activity(
        self,
        user_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Fetch recent memory activity events."""
        limit = max(1, min(int(limit or 20), 100))
        backend = get_backend_type()
        conn = self._get_connection()
        events = []
        try:
            if backend == "sqlite":
                if user_id:
                    cursor = conn.execute(
                        """
                        SELECT activity_id, event_type, user_id, chat_id, memory_id, details, created_at
                        FROM memory_activity
                        WHERE user_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?;
                        """,
                        (user_id, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT activity_id, event_type, user_id, chat_id, memory_id, details, created_at
                        FROM memory_activity
                        ORDER BY created_at DESC
                        LIMIT ?;
                        """,
                        (limit,),
                    )
                for r in cursor.fetchall():
                    events.append({
                        "activity_id": str(r["activity_id"]),
                        "event_type": str(r["event_type"]),
                        "user_id": str(r["user_id"]),
                        "chat_id": r["chat_id"],
                        "memory_id": r["memory_id"],
                        "details": r["details"],
                        "created_at": str(r["created_at"]),
                    })
            else:
                with conn.cursor() as cur:
                    if user_id:
                        cur.execute(
                            """
                            SELECT activity_id, event_type, user_id, chat_id, memory_id, details, created_at
                            FROM memory_activity
                            WHERE user_id = %s
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            (user_id, limit),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT activity_id, event_type, user_id, chat_id, memory_id, details, created_at
                            FROM memory_activity
                            ORDER BY created_at DESC
                            LIMIT %s;
                            """,
                            (limit,),
                        )
                    for r in cur.fetchall():
                        events.append({
                            "activity_id": str(r[0]),
                            "event_type": str(r[1]),
                            "user_id": str(r[2]),
                            "chat_id": r[3],
                            "memory_id": r[4],
                            "details": r[5],
                            "created_at": str(r[6]),
                        })
        except Exception as exc:
            logger.warning("Failed to fetch memory activity: %s", exc)
        finally:
            conn.close()
        return events

    def hard_delete_activity(self, activity_ids: List[str], user_id: str) -> int:
        """Permanently remove selected audit rows belonging to one user."""
        ids = [str(activity_id).strip() for activity_id in activity_ids if str(activity_id).strip()]
        if not ids or not user_id:
            return 0
        backend = get_backend_type()
        conn = self._get_connection()
        try:
            if backend == "sqlite":
                placeholders = ", ".join("?" for _ in ids)
                with conn:
                    cursor = conn.execute(
                        f"DELETE FROM memory_activity WHERE user_id = ? AND activity_id IN ({placeholders});",
                        [user_id, *ids],
                    )
                    return max(0, int(cursor.rowcount))
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM memory_activity WHERE user_id = %s AND activity_id = ANY(%s);",
                        (user_id, ids),
                    )
                    return max(0, int(cur.rowcount))
        finally:
            conn.close()

    def reindex_memories(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Reindex active canonical memories from PostgreSQL/SQLite into Chroma vector store."""
        from sage_memory_index import memory_vector_index
        return memory_vector_index.reindex_memories(self, user_id=user_id)


# Singleton instance
sage_memory = SageMemory()
