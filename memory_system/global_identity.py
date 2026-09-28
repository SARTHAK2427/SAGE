"""Deterministic identities for small, stable Global-memory facts.

The curator writes natural-language memories, so exact-string deduplication
cannot recognize that “User name is Rakshit Jain” and “The user identified
themselves as Rakshit Jain” are the same fact.  This module intentionally
handles only unambiguous profile slots.  It is a write-time guard, not a new
LLM or a replacement for the later memory redesign.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple


def normalized_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = re.sub(r"[^\w\s'-]", " ", text)
    return " ".join(text.split())


def _clean_value(value: str) -> str:
    return normalized_text(value).strip(" .,'\"")


def global_memory_identity(content: str) -> Optional[Tuple[str, str]]:
    """Return (stable slot, canonical value) for unambiguous Global facts."""
    raw = unicodedata.normalize("NFKC", str(content or ""))

    name_patterns = (
        r"\buser(?:'s)?\s+name\s+is\s+([^.,;\n]+)",
        r"\buser(?:'s)?\s+identifier\s+is\s+([^.,;\n]+)",
        r"\bthe\s+user\s+(?:is\s+named|identified\s+themselves\s+as|identified\s+themself\s+as)\s+([^.,;\n]+)",
        r"\buser\s+identity\s*:\s*([^,;\n]+)",
    )
    for pattern in name_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            value = _clean_value(match.group(1))
            # Names must contain letters and stay short. This avoids treating
            # arbitrary prose after “identity:” as a profile name.
            if value and re.fullmatch(r"[a-z][a-z' -]{0,78}", value):
                return "identity.name", value

    age_patterns = (
        r"\buser(?:'s)?\s+age\s+is\s+(\d{1,3})\b",
        r"\buser\s+is\s+(\d{1,3})\s+(?:years?\s+old|yo)\b",
    )
    for pattern in age_patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            age = int(match.group(1))
            if 0 < age < 130:
                return "identity.age", str(age)
    return None


def same_global_memory(left: str, right: str) -> bool:
    """Conservative generic fallback for repeated near-identical records."""
    left_normalized = normalized_text(left)
    right_normalized = normalized_text(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    left_words = set(left_normalized.split())
    right_words = set(right_normalized.split())
    union = left_words | right_words
    # Only collapse clearly near-identical longer statements. A loose matcher
    # would incorrectly merge separate preferences/projects.
    return len(union) >= 6 and len(left_words & right_words) / len(union) >= 0.9
