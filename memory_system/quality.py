"""Conservative quality checks shared by memory storage, recall, and UI APIs."""

from __future__ import annotations

import re
from typing import Optional


_GLOBAL_REJECTIONS = (
    (re.compile(r"\b(?:not yet|never|has not|hasn't)\s+(?:been\s+)?(?:shared|provided|disclosed|specified|stated|mentioned)\b", re.I), "missing-information statement"),
    (re.compile(r"\b(?:user(?:'s)?\s+[^.]{0,60}\s+is|identity is)\s+(?:unknown|not known|undisclosed|unavailable)\b", re.I), "unknown-user statement"),
    (re.compile(r"\b(?:i|assistant|model)\s+(?:do not|don't|does not|doesn't|cannot|can't)\s+know\b", re.I), "assistant uncertainty"),
    (re.compile(r"\bno (?:information|details?) (?:is|are|was|were) (?:available|provided|known)\b", re.I), "absence of information"),
    (re.compile(r"\buser (?:initiated|started) (?:a |the )?(?:greeting|conversation)\b", re.I), "greeting metadata"),
    (re.compile(r"\bno further action (?:is )?required\b", re.I), "workflow filler"),
    (re.compile(r"\bcurrently (?:asking|inquiring|requesting)\b", re.I), "transient request"),
    (re.compile(r"\b(?:implying|implied|suggesting|context suggests)\b", re.I), "inferred user trait"),
    (re.compile(r"\b(?:it is|it's|we|\w+ is) recommend(?:ed)?\b|\brecommendation:\s*", re.I), "assistant recommendation"),
    (re.compile(r"\bassistant(?:'s)? (?:knowledge|uncertainty|answer|response)\b", re.I), "assistant-state metadata"),
    (re.compile(r"\bwhich is considered\b|\bsuitability depends\b", re.I), "assistant evaluation"),
    (re.compile(r"\buser prefers to (?:start|begin) conversations? with (?:hi|hello|a greeting)\b", re.I), "greeting preference inference"),
)

_COMMON_REJECTIONS = (
    (re.compile(r"^\s*(?:hi|hello|hey|thanks|thank you)[.!\s]*$", re.I), "greeting or filler"),
)


def memory_rejection_reason(content: str, scope: str = "global") -> Optional[str]:
    """Return a reason when a proposed memory is unsafe or useless to retain."""
    # Curators often emit typographic apostrophes. Normalize them before the
    # rule checks so “User’s name is not disclosed” cannot bypass validation.
    text = " ".join(str(content or "").replace("\u2019", "'").replace("\u2018", "'").split())
    if not text:
        return "empty memory"
    for pattern, reason in _COMMON_REJECTIONS:
        if pattern.search(text):
            return reason
    if str(scope).lower() == "global":
        if re.search(r"\b(?:user(?:'s)?|the user|user name|name)\b[^.]{0,100}\b(?:do(?:es)? not|don't|doesn't|did not|didn't)\s+know\b", text, re.I):
            return "unknown-user statement"
        if re.search(r"\b(?:user(?:'s)?\s+)?name\s+(?:is|was)\s+(?:not\s+)?(?:disclosed|known|provided|available)\b", text, re.I):
            return "missing-information statement"
        if re.search(r"\b(?:assistant(?:'s)?|the assistant|model)\s+name\b", text, re.I):
            return "assistant-state metadata"
        for pattern, reason in _GLOBAL_REJECTIONS:
            if pattern.search(text):
                return reason
    return None


def is_valid_memory(content: str, scope: str = "global") -> bool:
    return memory_rejection_reason(content, scope) is None
