"""
standalone_memory/memory_engine.py
Canonical Relational Ledger and Durable Memory Management Subsystem.
Supports SQLite (WAL mode) and PostgreSQL (psycopg3).
Synchronizes all active vector embeddings with vector_index.
"""

from __future__ import annotations
import logging
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config
from .vector_index import memory_vector_index

logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = {
    "fact", "preference", "project", "decision", "instruction",
    "task", "personal", "technical", "summary", "general"
}
ALLOWED_STATUSES = {"active", "superseded", "deleted", "promoted", "compacted", "decayed"}
ALLOWED_TIERS = {"hot", "cold"}


def validate_category(category: str) -> str:
    """Validate and normalize memory category string."""
    cat = (category or "").lower().strip()
    if cat not in ALLOWED_CATEGORIES:
        raise ValueError(
            f"Invalid category '{category}'. Allowed categories: {sorted(list(ALLOWED_CATEGORIES))}"
        )
    return cat


class MemoryEngine:
    """Primary persistence and memory state coordinator."""

    def __init__(
        self,
        db_backend: Optional[str] = None,
        sqlite_path: Optional[str] = None,
    ) -> None:
        self.backend = (db_backend or config.DB_BACKEND).lower().strip()
        self.sqlite_path = sqlite_path or config.SQLITE_DB_PATH
        self._initialized = False

    def _get_connection(self) -> Any:
        if self.backend == "sqlite":
            Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            if not self._initialized:
                self._init_sqlite_schema(conn)
                self._initialized = True
            return conn
        else:
            # PostgreSQL backend
            import psycopg
            conn = psycopg.connect(
                host=config.POSTGRES_HOST,
                port=config.POSTGRES_PORT,
                dbname=config.POSTGRES_DB,
                user=config.POSTGRES_USER,
                password=config.POSTGRES_PASSWORD,
            )
            if not self._initialized:
                self._init_postgres_schema(conn)
                self._initialized = True
            return conn

    def _init_sqlite_schema(self, conn: sqlite3.Connection) -> None:
        """Create required tables and WAL mode for SQLite."""
        with conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")

            conn.execute("""
            CREATE TABLE IF NOT EXISTS sage_chats (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS sage_messages (
                msg_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (chat_id) REFERENCES sage_chats(chat_id) ON DELETE CASCADE
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS sage_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_tier TEXT NOT NULL DEFAULT 'cold',
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 1.0,
                status TEXT NOT NULL DEFAULT 'active',
                is_global INTEGER DEFAULT 0,
                source_chat_id TEXT,
                source_msg_id TEXT,
                supersedes_memory_id TEXT,
                access_count INTEGER DEFAULT 0,
                decay_score REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_accessed_at TEXT
            );
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS sage_memory_activity (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT,
                action TEXT NOT NULL,
                tier TEXT,
                details TEXT,
                timestamp TEXT NOT NULL
            );
            """)

            # Useful indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_user_status ON sage_memories(user_id, status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_tier ON sage_memories(memory_tier);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_chat ON sage_messages(chat_id);")

    def _init_postgres_schema(self, conn: Any) -> None:
        """Create tables in PostgreSQL."""
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS sage_chats (
                chat_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sage_messages (
                msg_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL REFERENCES sage_chats(chat_id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sage_memories (
                memory_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                memory_tier TEXT NOT NULL DEFAULT 'cold',
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                confidence REAL DEFAULT 1.0,
                status TEXT NOT NULL DEFAULT 'active',
                is_global INTEGER DEFAULT 0,
                source_chat_id TEXT,
                source_msg_id TEXT,
                supersedes_memory_id TEXT,
                access_count INTEGER DEFAULT 0,
                decay_score REAL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                last_accessed_at TIMESTAMPTZ
            );
            CREATE TABLE IF NOT EXISTS sage_memory_activity (
                event_id TEXT PRIMARY KEY,
                memory_id TEXT,
                action TEXT NOT NULL,
                tier TEXT,
                details TEXT,
                timestamp TIMESTAMPTZ NOT NULL
            );
            """)
        conn.commit()

    # ── Core Memory Operations ───────────────────────────────────────────────

    def store_memory(
        self,
        user_id: str,
        content: str,
        tier: str = "cold",
        category: str = "general",
        importance: float = 0.5,
        confidence: float = 1.0,
        is_global: bool = False,
        source_chat_id: Optional[str] = None,
        source_msg_id: Optional[str] = None,
        supersedes_memory_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a new memory in canonical database and index in vector store."""
        validated_cat = validate_category(category)
        tier_clean = (tier or "cold").lower().strip()
        if tier_clean not in ALLOWED_TIERS:
            tier_clean = "cold"

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        record = {
            "memory_id": memory_id,
            "user_id": user_id,
            "memory_tier": tier_clean,
            "content": content.strip(),
            "category": validated_cat,
            "importance": max(0.0, min(float(importance), 1.0)),
            "confidence": max(0.0, min(float(confidence), 1.0)),
            "status": "active",
            "is_global": 1 if is_global else 0,
            "source_chat_id": source_chat_id,
            "source_msg_id": source_msg_id,
            "supersedes_memory_id": supersedes_memory_id,
            "access_count": 0,
            "decay_score": 1.0,
            "created_at": now,
            "updated_at": now,
            "last_accessed_at": now,
        }

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO sage_memories (
                        memory_id, user_id, memory_tier, content, category,
                        importance, confidence, status, is_global,
                        source_chat_id, source_msg_id, supersedes_memory_id,
                        access_count, decay_score, created_at, updated_at, last_accessed_at
                    ) VALUES (
                        :memory_id, :user_id, :memory_tier, :content, :category,
                        :importance, :confidence, :status, :is_global,
                        :source_chat_id, :source_msg_id, :supersedes_memory_id,
                        :access_count, :decay_score, :created_at, :updated_at, :last_accessed_at
                    )
                    """,
                    record,
                )
        finally:
            conn.close()

        # Sync with Chroma vector index
        memory_vector_index.index_memory(record)
        self.record_activity("store", memory_id, tier_clean, f"Stored new {tier_clean} memory: {content[:40]}...")

        return record

    def get_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sage_memories WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return dict(row)
        finally:
            conn.close()

    def list_memories(
        self,
        user_id: str,
        memory_tier: Optional[str] = None,
        status: Optional[str] = "active",
        is_global: Optional[bool] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List memories with flexible filtering."""
        query = "SELECT * FROM sage_memories WHERE user_id = ?"
        params: List[Any] = [user_id]

        if memory_tier:
            query += " AND memory_tier = ?"
            params.append(memory_tier)
        if status:
            query += " AND status = ?"
            params.append(status)
        if is_global is not None:
            query += " AND is_global = ?"
            params.append(1 if is_global else 0)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def update_memory_status(self, memory_id: str, new_status: str) -> bool:
        """Transition status of a memory."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "UPDATE sage_memories SET status = ?, updated_at = ? WHERE memory_id = ?",
                    (new_status, now, memory_id),
                )
        finally:
            conn.close()

        memory_vector_index.update_memory_status(memory_id, new_status)
        self.record_activity("status_change", memory_id, None, f"Changed status to {new_status}")
        return True

    def delete_memory(self, memory_id: str) -> bool:
        """Soft delete memory in DB and remove from vector index."""
        self.update_memory_status(memory_id, "deleted")
        memory_vector_index.remove_memory(memory_id)
        self.record_activity("delete", memory_id, None, "Deleted memory")
        return True

    def promote_memory(self, memory_id: str) -> bool:
        """Promote a hot memory to cold long-term storage."""
        mem = self.get_memory(memory_id)
        if not mem:
            return False
        if mem.get("memory_tier") == "cold":
            return True

        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "UPDATE sage_memories SET memory_tier = 'cold', updated_at = ? WHERE memory_id = ?",
                    (now, memory_id),
                )
        finally:
            conn.close()

        # Re-index in Chroma cold collection and remove from hot
        memory_vector_index.remove_memory(memory_id, tier="hot")
        mem["memory_tier"] = "cold"
        memory_vector_index.index_memory(mem)
        self.record_activity("promote", memory_id, "cold", "Promoted from hot to cold tier")
        return True

    def compact_memories(
        self,
        user_id: str,
        memory_ids: List[str],
        summary_content: str,
        tier: str = "cold",
    ) -> Dict[str, Any]:
        """Mark source memories as compacted and create a consolidated summary."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            with conn:
                for mid in memory_ids:
                    conn.execute(
                        "UPDATE sage_memories SET status = 'compacted', updated_at = ? WHERE memory_id = ?",
                        (now, mid),
                    )
                    memory_vector_index.update_memory_status(mid, "compacted")
        finally:
            conn.close()

        # Create consolidated memory
        compacted = self.store_memory(
            user_id=user_id,
            content=summary_content,
            tier=tier,
            category="summary",
            importance=0.8,
        )
        self.record_activity(
            "compact",
            compacted["memory_id"],
            tier,
            f"Compacted {len(memory_ids)} memories into new summary",
        )
        return compacted

    def apply_decay(self, user_id: str, decay_factor: float = 0.9, threshold: float = 0.3) -> int:
        """Simulate memory forgetting for stale, non-accessed hot memories."""
        conn = self._get_connection()
        decayed_count = 0
        try:
            with conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT memory_id, decay_score, access_count FROM sage_memories WHERE user_id = ? AND memory_tier = 'hot' AND status = 'active'",
                    (user_id,),
                )
                rows = cursor.fetchall()
                now = datetime.now(timezone.utc).isoformat()

                for r in rows:
                    mid = r["memory_id"]
                    curr_score = float(r["decay_score"] or 1.0)
                    new_score = curr_score * decay_factor
                    if new_score < threshold:
                        conn.execute(
                            "UPDATE sage_memories SET decay_score = ?, status = 'decayed', updated_at = ? WHERE memory_id = ?",
                            (new_score, now, mid),
                        )
                        memory_vector_index.update_memory_status(mid, "decayed")
                        decayed_count += 1
                    else:
                        conn.execute(
                            "UPDATE sage_memories SET decay_score = ?, updated_at = ? WHERE memory_id = ?",
                            (new_score, now, mid),
                        )
        finally:
            conn.close()

        if decayed_count > 0:
            self.record_activity("decay", None, "hot", f"Decayed {decayed_count} stale memories")
        return decayed_count

    # ── Audit Trail ──────────────────────────────────────────────────────────

    def record_activity(self, action: str, memory_id: Optional[str], tier: Optional[str], details: str) -> None:
        """Record an event in the memory activity audit trail."""
        event_id = f"ev_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO sage_memory_activity (event_id, memory_id, action, tier, details, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
                    (event_id, memory_id, action, tier, details, now),
                )
        except Exception as exc:
            logger.error("Failed to record memory activity: %s", exc)
        finally:
            conn.close()

    def get_activity_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent audit activity."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sage_memory_activity ORDER BY timestamp DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    # ── Chat Ledger Operations ───────────────────────────────────────────────

    def ensure_chat(self, chat_id: str, user_id: str, title: Optional[str] = None) -> None:
        """Ensure chat session exists in the ledger."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    """
                    INSERT INTO sage_chats (chat_id, user_id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(chat_id) DO UPDATE SET updated_at = excluded.updated_at
                    """,
                    (chat_id, user_id, title or "New Conversation", now, now),
                )
        finally:
            conn.close()

    def add_message(self, chat_id: str, role: str, content: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Record a conversation turn in the chat ledger."""
        uid = user_id or config.DEFAULT_USER_ID
        self.ensure_chat(chat_id, uid)
        msg_id = f"msg_{uuid.uuid4().hex[:10]}"
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO sage_messages (msg_id, chat_id, role, content, tokens, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (msg_id, chat_id, role, content, len(content.split()), now),
                )
        finally:
            conn.close()

        return {"msg_id": msg_id, "chat_id": chat_id, "role": role, "content": content, "created_at": now}

    def get_messages(self, chat_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve chronological messages for a conversation."""
        query = "SELECT * FROM sage_messages WHERE chat_id = ? ORDER BY created_at ASC"
        params: List[Any] = [chat_id]
        if limit:
            query = "SELECT * FROM (SELECT * FROM sage_messages WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?) ORDER BY created_at ASC"
            params = [chat_id, limit]

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()

    def list_chats(self, user_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        """List active chats for a user."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sage_chats WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?", (user_id, limit))
            return [dict(r) for r in cursor.fetchall()]
        finally:
            conn.close()


# Default singleton instance
memory_engine = MemoryEngine()
