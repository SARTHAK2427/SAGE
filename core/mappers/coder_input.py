"""
SAGE/core/mappers/coder_input.py
Gemma tool arguments → Qwen-Coder plain-text prompt builder.

Gemma emits:
    {
      "tool": "code_specialist",
      "function": "solve_code_task",
      "arguments": {
        "instruction": "...",
        "code": "..." | null,
        "language": "..." | null
      }
    }

This mapper produces the minimal plain-text prompt sent to Qwen-Coder.
The coder_system.txt already instructs Qwen to output raw code only.
The mapper's job is to format the user-turn prompt correctly.

Rules:
    - Preserve code text and indentation exactly (no re-formatting).
    - Do not inject JSON wrappers, tool names, or SAGE runtime metadata.
    - Do not add conversation history unless explicitly required.
    - TARGET LANGUAGE section is emitted only when language is provided.
    - CODE section is emitted only when existing code is provided.

Execution gating:
    The sandbox currently supports Python only.
    build_coder_input() returns the plain prompt for any language,
    but tool/coder adapters must check SANDBOX_SUPPORTED_LANGUAGES
    before routing to the Docker sandbox. For unsupported languages,
    execution_status should be "not_executed" rather than running
    a Python executor on C++ code.
"""

from __future__ import annotations

# Languages the Docker sandbox (python:3.12-slim) can actually execute.
# Expand this set only when the sandbox image is updated to support other runtimes.
SANDBOX_SUPPORTED_LANGUAGES: frozenset[str] = frozenset({"python", "python3"})


def build_coder_input(
    instruction: str,
    code: str | None = None,
    language: str | None = None,
) -> str:
    """Build the Qwen-Coder user-turn prompt from Gemma's tool arguments.

    Args:
        instruction: The task/instruction string from Gemma (required).
        code:        Existing code to modify/fix/translate (optional).
        language:    Target language (optional; passed through as-is).

    Returns:
        A plain-text string following the exact coder input protocol:

            TASK:
            <instruction>

            TARGET LANGUAGE:     (only when language is not None/empty)
            <language>

            CODE:                (only when code is not None/empty)
            <code exactly>
    """
    parts: list[str] = []

    parts.append(f"TASK:\n{instruction.strip()}")

    if language and language.strip():
        parts.append(f"TARGET LANGUAGE:\n{language.strip()}")

    if code and code.strip():
        # Preserve exact indentation — do NOT lstrip/rstrip individual lines.
        parts.append(f"CODE:\n{code}")

    return "\n\n".join(parts)


def is_sandbox_supported(language: str | None) -> bool:
    """Return True iff the requested language can be executed by the Docker sandbox.

    Args:
        language: Language string from Gemma (e.g. "python", "python3", "c++").
                  None or empty string is treated as "python" for backward compat.

    Returns:
        True when language normalises to a sandbox-supported runtime.
    """
    if not language:
        return True  # Default assumption is Python
    return language.strip().lower() in SANDBOX_SUPPORTED_LANGUAGES
