"""
Guard layer — runs on every agent response before it reaches the user.

Strips PII (CPF numbers, Brazilian phone numbers) from the response silently.
"""

import re

_PII_PATTERNS = [
    re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),  # CPF: 123.456.789-00
    re.compile(r"\(\d{2}\)\s*\d{4,5}-\d{4}"),  # Phone: (11) 91234-5678
]


def _strip_pii(text: str) -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


class GuardResult:
    def __init__(self, text: str):
        self.text = text
        self.hallucinated: list[str] = []

    @property
    def clean(self) -> bool:
        return True


def apply(response: str, tool_output: str = "") -> GuardResult:
    """Strip PII from response. tool_output is accepted for API compatibility."""
    return GuardResult(text=_strip_pii(response))
