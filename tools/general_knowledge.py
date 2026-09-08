"""
SAGE/tools/general_knowledge.py
General Knowledge Specialist tool powered by Qwen3.5 2B.

Role:
    Handles general knowledge, factual, conceptual, educational, and
    conversational questions delegated by the Gemma semantic controller.
    Runs on the Qwen3.5 2B model using the prompts/qwen3.5/general_knowledge.txt persona.

Interface:
    general_knowledge / answer_query(query)
"""

from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import config
from core.dispatcher import ToolRegistry

logger = logging.getLogger(__name__)


def _is_mock_mode() -> bool:
    return os.environ.get("SAGE_MOCK_MODE", "0") == "1"


def _load_knowledge_prompt() -> str:
    path = config.PROMPTS_DIR / "qwen3.5" / "general_knowledge.txt"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return (
        "You are the general knowledge assistant for SAGE, powered by Qwen3.5. "
        "Answer the user's factual, conceptual, or conversational query directly and accurately."
    )


def tool_general_knowledge(
    *,
    query: str = "",
    model_manager=None,
    model_client=None,
) -> Dict[str, Any]:
    """Execute general knowledge query using Qwen3.5 2B with the general_knowledge persona.

    Args:
        query: Factual or conceptual query string.
        model_manager: ModelManager singleton (optional).
        model_client: ModelClient singleton (optional).

    Returns:
        Structured dict: {"status": "success"|"error", "answer": str, "query": str}
    """
    clean_query = (query or "").strip()
    if not clean_query:
        return {
            "status": "error",
            "answer": "",
            "query": "",
            "error": "Query cannot be empty for general_knowledge / answer_query.",
        }

    if _is_mock_mode() or model_client is None:
        logger.info("general_knowledge tool running in mock mode for query: %s", clean_query[:50])
        return {
            "status": "success",
            "answer": f"Mock general knowledge answer for query: '{clean_query}'.",
            "query": clean_query,
            "model": "qwen3.5-2b",
        }

    try:
        sys_prompt = _load_knowledge_prompt()
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": clean_query},
        ]

        if model_manager:
            model_manager.ensure_model("final_synthesizer")

        synth_cfg = config.MODELS.get("final_synthesizer", {})
        res = model_client.chat_completion(
            messages=messages,
            temperature=synth_cfg.get("temperature", 0.20),
            max_tokens=synth_cfg.get("max_tokens", 2048),
            model_key="final_synthesizer",
        )

        content = (res.get("content") or "").strip()
        duration = res.get("duration", 0.0)

        return {
            "status": "success" if content else "error",
            "answer": content or "No answer could be generated.",
            "query": clean_query,
            "duration": duration,
            "model": "qwen3.5-2b",
        }
    except Exception as exc:
        logger.exception("general_knowledge tool execution failed: %s", exc)
        return {
            "status": "error",
            "answer": "",
            "query": clean_query,
            "error": str(exc),
        }


def register_general_knowledge_tools(
    registry: ToolRegistry,
    model_manager=None,
    model_client=None,
) -> None:
    """Register general_knowledge tool endpoints into ToolRegistry."""
    def _handler(query: str = ""):
        return tool_general_knowledge(
            query=query,
            model_manager=model_manager,
            model_client=model_client,
        )

    registry.register("general_knowledge", "answer_query", _handler)
    # Register common backward-compatible aliases
    registry.register("knowledge_specialist", "answer_query", _handler)
    registry.register("general_chat", "answer_query", _handler)
