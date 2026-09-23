"""
standalone_memory/config.py
Self-contained configuration for SAGE Standalone Memory System.
Reads from environment variables with safe defaults.
"""

from __future__ import annotations
import os
from pathlib import Path

# Base storage directory
DEFAULT_STORAGE_ROOT = Path(os.getenv("SAGE_MEMORY_DIR", "./sage_memory_data")).resolve()
DEFAULT_STORAGE_ROOT.mkdir(parents=True, exist_ok=True)

# Database Backend: 'sqlite' or 'postgres'
DB_BACKEND = os.getenv("SAGE_MEMORY_DB", "sqlite").lower().strip()
SQLITE_DB_PATH = os.getenv("SAGE_SQLITE_PATH", str(DEFAULT_STORAGE_ROOT / "sage_memory.db"))

# PostgreSQL settings (if DB_BACKEND == "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "sage")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

# ChromaDB persistence directory
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_STORAGE_ROOT / "chroma_db"))
CHROMA_HOT_COLLECTION = os.getenv("CHROMA_HOT_COLLECTION", "sage_memory_hot")
CHROMA_COLD_COLLECTION = os.getenv("CHROMA_COLD_COLLECTION", "sage_memory_cold")

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

# User defaults
DEFAULT_USER_ID = os.getenv("DEFAULT_USER_ID", "default_user")

# Memory Budgets & Limits
RECENT_CHAT_MAX_MESSAGES = int(os.getenv("SAGE_RECENT_CHAT_MAX_MESSAGES", "5"))
GLOBAL_MEMORY_BUDGET_TOKENS = int(os.getenv("SAGE_GLOBAL_MEMORY_BUDGET_TOKENS", "600"))
CONTEXT_MEMORY_BUDGET_TOKENS = int(os.getenv("SAGE_CONTEXT_MEMORY_BUDGET_TOKENS", "1200"))
GLOBAL_MEMORY_MAX_ITEMS = int(os.getenv("SAGE_GLOBAL_MEMORY_MAX_ITEMS", "8"))
