"""
SAGE/tools/coder.py
Coder adapter wrapping the CodeExecutionPipeline (Docker sandbox + LLM auto-repair).

Architectural rules:
    - Wraps the existing Docker sandbox and Qwen2.5-Coder pipeline
    - Exposes code execution through the generic ToolDispatcher
    - Supports mock mode for local CPU tests without Docker or GPU
    - Returns bounded, JSON-safe structured dicts
    - Does NOT autonomously invoke other registered Sage tools

Canonical interface (tools.json):
    code_specialist / solve_code_task(instruction, code=None, language=None)

Language-gating:
    The Docker sandbox (python:3.12-slim) only executes Python.
    If language is not in SANDBOX_SUPPORTED_LANGUAGES, the model generates
    code but execution_status is returned as "not_executed" rather than
    attempting to run another language through the Python executor.

Internal backward-compat aliases (NOT exposed to Gemma):
    "coder" / "execute"           → tool_code_execute (legacy)
    "code_specialist" / "execute" → tool_code_execute (legacy)
"""

from __future__ import annotations
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


def _is_mock_mode() -> bool:
    return os.environ.get("SAGE_MOCK_MODE", "0") == "1"


# ── Canonical interface: code_specialist / solve_code_task ─────────────────────

def tool_code_solve_task(
    pipeline=None,
    model_manager=None,
    model_client=None,
    *,
    instruction: str = "",
    code: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """Canonical endpoint matching tools.json: code_specialist / solve_code_task.

    Args:
        pipeline:      CodeExecutionPipeline instance (or None in mock mode)
        model_manager: ModelManager instance (or None in mock mode)
        model_client:  ModelClient instance (or None in mock mode)
        instruction:   Natural language task / instruction (from Gemma)
        code:          Optional existing code to modify/fix/translate
        language:      Target language (optional; sandbox-gated)

    Language-gating:
        If language is not Python/Python3, the coder model still generates code
        but the Docker sandbox is NOT invoked. execution_status = "not_executed".

    Returns:
        Rich dict consumed by map_coder_result() in core/mappers/gemma_results.py.
    """
    from core.mappers.coder_input import build_coder_input, is_sandbox_supported

    if not instruction and not code:
        return {
            "status": "error",
            "succeeded": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "No instruction or code provided to code_specialist",
            "attempts": 0,
            "final_code": "",
            "code": "",
            "language": language or "unknown",
            "wall_time_ms": 0.0,
            "memory_peak_mb": 0.0,
            "error": "No instruction or code provided to code_specialist",
        }

    sandbox_ok = is_sandbox_supported(language)
    effective_lang = language.strip() if language and language.strip() else "python"

    # ── Mock mode ─────────────────────────────────────────────────────────────
    if _is_mock_mode() or pipeline is None:
        logger.info("Coder tool running in MOCK mode for instruction: %s", instruction[:60])
        mock_code = code or f"# Mock code for: {instruction}\nprint('MOCK_RESULT: {instruction}')"
        mock_output = f"MOCK_RESULT: executed code for task '{instruction}' successfully."

        if not sandbox_ok:
            return {
                "status": "success",
                "succeeded": False,
                "execution_status": "not_executed",
                "code": mock_code,
                "final_code": mock_code,
                "language": effective_lang,
                "stdout": "",
                "stderr": "",
                "attempts": 1,
                "wall_time_ms": 0.0,
                "memory_peak_mb": 0.0,
                "error": None,
            }

        return {
            "status": "success",
            "succeeded": True,
            "exit_code": 0,
            "stdout": mock_output,
            "stderr": "",
            "attempts": 1,
            "final_code": mock_code,
            "code": mock_code,
            "language": effective_lang,
            "wall_time_ms": 15.0,
            "memory_peak_mb": 12.0,
            "error": None,
        }

    # ── Unsupported language: generate code but skip sandbox ──────────────────
    if not sandbox_ok:
        logger.info(
            "Language '%s' is not supported by sandbox; generating code without execution.",
            effective_lang,
        )
        generated = _generate_code_via_model(
            instruction=instruction, code=code, language=language,
            model_manager=model_manager, model_client=model_client,
        )
        return {
            "status": "success",
            "succeeded": False,
            "execution_status": "not_executed",
            "code": generated,
            "final_code": generated,
            "language": effective_lang,
            "stdout": "",
            "stderr": "",
            "attempts": 1,
            "wall_time_ms": 0.0,
            "memory_peak_mb": 0.0,
            "error": None,
        }

    # ── Live sandbox mode ──────────────────────────────────────────────────────
    try:
        if code:
            # Run the provided code directly in the sandbox
            raw_llm_response = f"```{effective_lang}\n{code}\n```"
        else:
            raw_llm_response = _get_llm_code_response(
                instruction=instruction, code=code, language=language,
                model_manager=model_manager, model_client=model_client,
            )

        pipe_result = pipeline.execute(task=instruction, llm_response=raw_llm_response)
        if model_manager and hasattr(model_manager, "rearm_agent_background"):
            model_manager.rearm_agent_background()
        extracted_lang = getattr(pipe_result, "extraction_language", None) or effective_lang

        return {
            "status": "success" if pipe_result.succeeded else "error",
            "succeeded": pipe_result.succeeded,
            "exit_code": pipe_result.exit_code,
            "stdout": pipe_result.stdout[:2000] if pipe_result.stdout else "",
            "stderr": pipe_result.stderr[:2000] if pipe_result.stderr else "",
            "attempts": pipe_result.attempts,
            "final_code": pipe_result.final_code or "",
            "code": pipe_result.final_code or "",
            "language": extracted_lang,
            "wall_time_ms": pipe_result.wall_time_ms,
            "memory_peak_mb": pipe_result.memory_peak_mb,
            "error": pipe_result.stderr if not pipe_result.succeeded else None,
        }

    except Exception as exc:
        logger.exception("Error during code_specialist solve_code_task")
        return {
            "status": "error",
            "succeeded": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": str(exc),
            "attempts": 0,
            "final_code": code or "",
            "code": code or "",
            "language": effective_lang,
            "wall_time_ms": 0.0,
            "memory_peak_mb": 0.0,
            "error": str(exc),
        }


# ── Legacy adapter: task/context → instruction ─────────────────────────────────

def tool_code_execute(
    pipeline=None,
    model_manager=None,
    model_client=None,
    *,
    task: str = "",
    context: str = "",
    code: Optional[str] = None,
    language: Optional[str] = None,
) -> dict:
    """Legacy adapter retaining backward-compat 'task'/'context' arg names.

    Retained for internal registry aliases only. Gemma uses solve_code_task.
    """
    if not task and not code:
        return {
            "status": "error",
            "succeeded": False,
            "exit_code": -1,
            "stdout": "",
            "stderr": "No task or code provided to coder tool",
            "attempts": 0,
            "final_code": "",
            "language": language or "unknown",
            "wall_time_ms": 0.0,
            "memory_peak_mb": 0.0,
            "error": "No task or code provided to coder tool",
        }
    instruction = task
    if context:
        instruction = f"{task}\n\nContext:\n{context}" if task else context
    return tool_code_solve_task(
        pipeline=pipeline,
        model_manager=model_manager,
        model_client=model_client,
        instruction=instruction,
        code=code,
        language=language,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _query_coder_llm(
    instruction: str,
    code: Optional[str],
    language: Optional[str],
    model_manager,
    model_client,
) -> str:
    """Invoke the coder specialist model with system prompt and structured input."""
    import config
    from core.mappers.coder_input import build_coder_input

    coder_prompt = ""
    prompt_file = getattr(config, "PROMPTS_DIR", None)
    if prompt_file and (prompt_file / "coder_system.txt").exists():
        with open(prompt_file / "coder_system.txt", "r", encoding="utf-8") as f:
            coder_prompt = f.read().strip()

    messages = []
    if coder_prompt:
        messages.append({"role": "system", "content": coder_prompt})
    messages.append({"role": "user", "content": build_coder_input(instruction, code, language)})

    model_manager.ensure_model("coder")
    coder_cfg = config.MODELS.get("coder", {})
    res = model_client.chat_completion(
        messages=messages,
        temperature=coder_cfg.get("temperature", 0.05),
        max_tokens=coder_cfg.get("max_tokens", 4096),
        model_key="coder",
    )
    return res.get("content", "") or ""


def _generate_code_via_model(
    instruction: str,
    code: Optional[str],
    language: Optional[str],
    model_manager,
    model_client,
) -> str:
    """Call Qwen-Coder to generate code (without sandbox execution)."""
    if model_manager is None or model_client is None:
        return code or f"# Code generation not available\n# Task: {instruction}"
    try:
        return _query_coder_llm(instruction, code, language, model_manager, model_client)
    except Exception as exc:
        logger.warning("Code generation via model failed: %s", exc)
        return code or ""


def _get_llm_code_response(
    instruction: str,
    code: Optional[str],
    language: Optional[str],
    model_manager,
    model_client,
) -> str:
    """Get raw LLM response (code in fenced block) for sandbox execution."""
    effective_lang = language.strip().lower() if language and language.strip() else "python"
    if model_manager is None or model_client is None:
        return f"```{effective_lang}\n{code or ''}\n```"
    try:
        return _query_coder_llm(instruction, code, language, model_manager, model_client)
    except Exception as exc:
        logger.exception("LLM code generation failed: %s", exc)
        return f"```{effective_lang}\n{code or ''}\n```"


# ─── Registration Helper ──────────────────────────────────────────────────────

def register_coder_tools(registry, pipeline=None, model_manager=None, model_client=None) -> None:
    """Register coder tools with a ToolRegistry.

    Canonical (Gemma-visible via tools.json):
        code_specialist / solve_code_task

    Internal backward-compat aliases (NOT exposed to Gemma):
        coder / execute
        code_specialist / execute
    """
    from functools import partial

    canonical_fn = partial(tool_code_solve_task, pipeline, model_manager, model_client)
    registry.register("code_specialist", "solve_code_task", canonical_fn)

    legacy_fn = partial(tool_code_execute, pipeline, model_manager, model_client)
    registry.register("coder", "execute", legacy_fn)
    registry.register("code_specialist", "execute", legacy_fn)
