"""
SAGE Code Executor — Code Extractor
=====================================
Extracts clean Python code from raw LLM responses that may contain:
  • ```python / ```python3 / ```py fenced blocks
  • Generic ``` blocks (no language tag)
  • Prose mixed with code
  • Multiple code blocks (largest wins)
  • Raw code with no fences at all (fallback)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Language tags that identify a Python block
_PYTHON_TAGS = frozenset({"python", "python3", "py"})
_PYTHON_PATTERN = r"```(?:python3?|py)\s*\n(.*?)```"
_GENERIC_PATTERN = r"```(?:\w+)?\s*\n(.*?)```"


@dataclass
class CodeExtractResult:
    code: str             # Clean, normalised code string
    language: str         # "python" | "unknown"
    had_fence: bool       # True if a ``` fence was found in the response
    block_count: int      # Number of fenced blocks found (0 if fallback)
    raw_response: str = ""
    detected_blocks: List[Dict[str, Any]] = field(default_factory=list)
    selected_block_index: Optional[int] = None
    extraction_strategy: str = "empty"
    fallback_used: bool = False
    warnings: List[str] = field(default_factory=list)

    def to_socket(self, raw_response: str = "") -> dict:
        """Full rich Component Q socket."""
        from core.sockets import build_code_extraction_socket
        return build_code_extraction_socket(
            code=self.code,
            language=self.language,
            had_fence=self.had_fence,
            block_count=self.block_count,
            raw_response=raw_response or self.raw_response,
            detected_blocks=self.detected_blocks,
            selected_block_index=self.selected_block_index,
            extraction_strategy=self.extraction_strategy,
            fallback_used=self.fallback_used,
            warnings=self.warnings,
        )


def extract_code(raw_response: str) -> CodeExtractResult:
    """
    Extract the best Python code block from an LLM response.

    Priority:
        1. Largest ```python / ```python3 / ```py fenced block
        2. Largest generic ``` fenced block (any or no language tag)
        3. Entire stripped response (fallback — treats whole reply as code)

    Returns:
        CodeExtractResult with normalised code and rich extraction metadata.
    """
    text = raw_response.strip()
    if not text:
        return CodeExtractResult(
            code="",
            language="unknown",
            had_fence=False,
            block_count=0,
            raw_response=raw_response,
            detected_blocks=[],
            selected_block_index=None,
            extraction_strategy="empty",
            fallback_used=False,
            warnings=["Raw response is empty"],
        )

    # Find all code blocks with their language tags
    fence_pattern = r"```([a-zA-Z0-9_-]*)\s*\n(.*?)```"
    matches = list(re.finditer(fence_pattern, text, re.DOTALL))

    detected_blocks: List[Dict[str, Any]] = []
    for idx, m in enumerate(matches):
        tag = m.group(1).strip().lower()
        block_body = _normalise(m.group(2))
        detected_blocks.append({
            "index": idx,
            "language": tag or "unknown",
            "code": block_body,
            "fenced": True,
            "char_count": len(block_body),
        })

    # ── Priority 1: named Python block ──────────────────────────────────
    py_candidates = [b for b in detected_blocks if b["language"] in _PYTHON_TAGS and b["code"]]
    if py_candidates:
        best = max(py_candidates, key=lambda b: len(b["code"]))
        return CodeExtractResult(
            code=best["code"],
            language="python",
            had_fence=True,
            block_count=len(detected_blocks),
            raw_response=raw_response,
            detected_blocks=detected_blocks,
            selected_block_index=best["index"],
            extraction_strategy="named_python_fence",
            fallback_used=False,
        )

    # ── Priority 2: any fenced block ────────────────────────────────────
    non_empty_blocks = [b for b in detected_blocks if b["code"]]
    if non_empty_blocks:
        best = max(non_empty_blocks, key=lambda b: len(b["code"]))
        return CodeExtractResult(
            code=best["code"],
            language=best["language"],
            had_fence=True,
            block_count=len(detected_blocks),
            raw_response=raw_response,
            detected_blocks=detected_blocks,
            selected_block_index=best["index"],
            extraction_strategy="generic_fence",
            fallback_used=False,
            warnings=["No explicit python language tag found on selected fence"],
        )

    # ── Priority 3: treat full response as code ──────────────────────────
    norm_text = _normalise(text)
    return CodeExtractResult(
        code=norm_text,
        language="unknown",
        had_fence=False,
        block_count=0,
        raw_response=raw_response,
        detected_blocks=[],
        selected_block_index=None,
        extraction_strategy="unfenced_fallback",
        fallback_used=True,
        warnings=["No code fences found in response; falling back to full text"],
    )


# ── Helpers ──────────────────────────────────────────────────────────────────

def _find_blocks(text: str, pattern: str) -> List[str]:
    """Return non-empty stripped code strings matching *pattern*."""
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    return [m.strip() for m in matches if m.strip()]


def _normalise(code: str) -> str:
    """Normalise line endings and strip surrounding whitespace."""
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()
