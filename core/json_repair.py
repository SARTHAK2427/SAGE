"""
SAGE/core/json_repair.py
Decoupled JSON extraction, cleanup, and repair logic for Gemma agent outputs.

Provides:
    - clean_json_string(text): extracts JSON from fences, thought tags, or brace boundaries
    - parse_agent_json(text): robustly parses agent structured JSON
    - build_repair_prompt(): standardized corrective prompt for 1-turn retry
"""

from __future__ import annotations
import json
import re
from typing import Any, Optional


def _extract_balanced_json(text: str) -> Optional[str]:
    """Find and extract a balanced { ... } JSON object, respecting quotes and escapes."""
    start = -1
    depth = 0
    in_str = False
    escape = False

    # Prefer the brace that starts the "type": "tool_calls" | "final" object if present
    match = re.search(r'\{\s*"type"\s*:\s*"(?:tool_calls|final)"', text)
    search_start = match.start() if match else 0

    for i in range(search_start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_str:
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue

        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start:i + 1].strip()
    return None


def clean_json_string(text: str) -> str:
    """Clean and isolate a JSON string from noisy LLM output.

    Steps:
        1. Strip thought/reasoning tags (<thought>...</thought>, <think>...</think>)
        2. Extract markdown code fences (```json ... ```)
        3. Extract balanced JSON object matching {"type": ...} without truncating nested dicts
        4. Fall back to outermost matching curly braces
    """
    text = text.strip()

    # 1. Strip reasoning / thought tags
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 2. Extract from markdown code fence anywhere in text
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        balanced = _extract_balanced_json(candidate)
        if balanced:
            return balanced
        if candidate.startswith("{") and candidate.endswith("}"):
            return candidate

    # 3. Extract balanced JSON object (respects strings, escapes, and nested braces)
    balanced = _extract_balanced_json(text)
    if balanced:
        return balanced

    # 4. Fallback to outermost braces
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx + 1].strip()

    return text


def _auto_close_brackets(cand: str) -> Optional[dict[str, Any]]:
    """Automatically balance unclosed strings, square brackets, and curly braces."""
    stack = []
    in_string = False
    escape = False
    for ch in cand:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ('{', '['):
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()

    closing = ""
    if in_string:
        closing += '"'
    while stack:
        b = stack.pop()
        closing += "}" if b == "{" else "]"

    for strict in (True, False):
        try:
            data = json.loads(cand + closing, strict=strict)
            if isinstance(data, dict) and "type" in data:
                return data
        except Exception:
            pass
    return None


def parse_agent_json(raw_text: str) -> Optional[dict[str, Any]]:
    """Parse raw agent output into a dictionary.

    Returns the parsed dict if valid and contains 'type', or None.
    Supports strict=False to handle literal newlines and control chars in LLM strings.
    """
    if not raw_text or not raw_text.strip():
        return None

    # Try direct parse first (strict and non-strict)
    for strict in (True, False):
        try:
            data = json.loads(raw_text.strip(), strict=strict)
            if isinstance(data, dict) and "type" in data:
                return data
        except Exception:
            pass

    # Clean and try again
    cleaned = clean_json_string(raw_text)
    for strict in (True, False):
        try:
            data = json.loads(cleaned, strict=strict)
            if isinstance(data, dict) and "type" in data:
                return data
        except Exception:
            pass

    # Recover when trailing closing brace(s), quote(s), or bracket(s) are omitted
    for candidate in (cleaned, raw_text.strip()):
        cand = candidate.strip()
        if not cand.startswith("{"):
            continue

        # Fast single trailing brace check
        if not cand.endswith("}"):
            for strict in (True, False):
                try:
                    data = json.loads(cand + "\n}", strict=strict)
                    if isinstance(data, dict) and "type" in data:
                        return data
                except Exception:
                    pass

        # Intelligent stack-based bracket auto-closure
        repaired = _auto_close_brackets(cand)
        if repaired is not None:
            return repaired

    # Robust regex fallback for final answer with unescaped internal quotes or newlines
    m = re.search(r'"type"\s*:\s*"final"\s*,\s*"answer"\s*:\s*"', raw_text, re.DOTALL)
    if not m:
        m = re.search(r'"answer"\s*:\s*"', raw_text, re.DOTALL)
    if m:
        start_pos = m.end()
        end_brace = raw_text.rfind('}')
        if end_brace > start_pos:
            end_quote = raw_text.rfind('"', start_pos, end_brace)
            if end_quote != -1:
                ans = raw_text[start_pos:end_quote]
                return {"type": "final", "answer": ans.strip()}

    return None


def parse_agent_json_socket(raw_text: str) -> dict:
    """Parse raw agent output and return the complete rich Component V socket."""
    import time
    from core.sockets import build_json_repair_socket, build_error_payload

    t0 = time.perf_counter()
    if not raw_text or not raw_text.strip():
        total_ms = (time.perf_counter() - t0) * 1000.0
        err = build_error_payload("EMPTY_INPUT", "ValueError", "Input string is empty", recoverable=False)
        return build_json_repair_socket(
            raw_input=raw_text or "",
            parsed_json=None,
            parsed_type="none",
            is_valid=False,
            duration_ms=total_ms,
            error=err,
        )

    # Check direct parse
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, dict):
            total_ms = (time.perf_counter() - t0) * 1000.0
            return build_json_repair_socket(
                raw_input=raw_text,
                parsed_json=data,
                parsed_type=data.get("type", "generic_object"),
                is_valid=True,
                duration_ms=total_ms,
            )
    except Exception:
        pass

    # Check cleaning steps
    thought_tags_stripped = bool(re.search(r"<thought>.*?</thought>", raw_text, flags=re.DOTALL))
    markdown_fence_extracted = bool(re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_text, flags=re.DOTALL))
    outermost_braces_extracted = False

    cleaned = clean_json_string(raw_text)
    if not markdown_fence_extracted and not thought_tags_stripped:
        outermost_braces_extracted = cleaned.startswith("{") and cleaned.endswith("}")

    try:
        data = json.loads(cleaned)
        total_ms = (time.perf_counter() - t0) * 1000.0
        return build_json_repair_socket(
            raw_input=raw_text,
            parsed_json=data,
            parsed_type=data.get("type", "generic_object") if isinstance(data, dict) else "generic_json",
            is_valid=True,
            thought_tags_stripped=thought_tags_stripped,
            markdown_fence_extracted=markdown_fence_extracted,
            outermost_braces_extracted=outermost_braces_extracted,
            repaired_string=cleaned,
            duration_ms=total_ms,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        err = build_error_payload("JSON_PARSE_ERROR", type(exc).__name__, str(exc), recoverable=True, retryable=True, exc=exc)
        return build_json_repair_socket(
            raw_input=raw_text,
            parsed_json=None,
            parsed_type="none",
            is_valid=False,
            thought_tags_stripped=thought_tags_stripped,
            markdown_fence_extracted=markdown_fence_extracted,
            outermost_braces_extracted=outermost_braces_extracted,
            repaired_string=cleaned,
            duration_ms=total_ms,
            error=err,
        )


def build_repair_prompt(is_initial_turn: bool = False) -> str:
    """Standardized 1-turn repair prompt when Gemma produces invalid JSON."""
    if is_initial_turn:
        return (
            "Your previous response was not valid JSON or attempted to answer directly.\n"
            "You are strictly the orchestrator and are FORBIDDEN from answering directly or writing code.\n"
            "Output ONLY a valid JSON tool call:\n"
            '{"type": "tool_calls", "calls": [{"tool": "...", "function": "...", "arguments": {...}}]}\n'
            "Delegate code to code_specialist, and general knowledge/questions to general_knowledge.\n"
            "No prose outside JSON."
        )
    return (
        "Your previous response was not valid JSON matching the required schema.\n"
        "Output ONLY a single valid JSON object:\n"
        "Either:\n"
        '{"type": "tool_calls", "calls": [...]}\n'
        "or (only after tool results are present):\n"
        '{"type": "final", "answer": "..."}\n'
        "No prose outside JSON."
    )

