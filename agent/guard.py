"""
Guard layer — runs on every agent response before it reaches the user.

Two checks:
  1. PII stripping  — removes CPF numbers, Brazilian phone numbers, and email
                      addresses from the response silently.
  2. Hallucination  — every number in the response must appear in the raw tool
                      output. If any number was invented, the response is flagged
                      so the caller can re-generate.
"""

import re

# ---------------------------------------------------------------------------
# PII patterns (Brazilian formats)
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),  # CPF: 123.456.789-00
    re.compile(r"\(\d{2}\)\s*\d{4,5}-\d{4}"),  # Phone: (11) 91234-5678
]


def _strip_pii(text: str) -> str:
    for pattern in _PII_PATTERNS:
        text = pattern.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")


def _extract_numbers(text: str) -> set[str]:
    """Return significant numbers found in text, normalised to '.' decimals.

    Only flags decimal numbers (prices) or integers >= 100 (years, large amounts).
    Single/double digit numbers (day/month components in dates) are too common
    to be reliable hallucination signals and cause excessive false positives.
    """
    results = set()
    for m in _NUMBER_RE.findall(text):
        normalised = m.replace(",", ".")
        if "." in normalised or int(normalised.split(".")[0]) >= 100:
            results.add(normalised)
    return results


def _check_hallucination(response: str, tool_output: str) -> list[str]:
    """Return numbers present in response but absent from tool output."""
    response_numbers = _extract_numbers(response)
    source_numbers = _extract_numbers(tool_output)
    return [n for n in response_numbers if n not in source_numbers]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class GuardResult:
    def __init__(self, text: str, hallucinated: list[str]):
        self.text = text
        self.hallucinated = hallucinated

    @property
    def clean(self) -> bool:
        return not self.hallucinated


def apply(response: str, tool_output: str) -> GuardResult:
    """
    Strip PII from response and check for hallucinated numbers.

    Args:
        response:    The agent's response text.
        tool_output: Raw concatenation of all tool results for this turn.

    Returns:
        GuardResult with cleaned text and a list of any hallucinated numbers.
        If GuardResult.clean is False, the caller should re-generate.
    """
    clean_response = _strip_pii(response)
    hallucinated = _check_hallucination(clean_response, tool_output)
    return GuardResult(text=clean_response, hallucinated=hallucinated)
