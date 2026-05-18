"""Custom LLM-as-judge for RAG evaluation — claim-level faithfulness.

One JSON-only judge call returns faithfulness + answer_relevancy on a 0-1 scale.
Faithfulness is graded at the CLAIM level (RAGAS-style): inferred conclusions count
as supported; only contradictions and absent facts are penalized.

Ported from karvanitis/vault-rag. Defaults:
  - model:    gpt-4o-mini
  - base URL: https://api.openai.com/v1
  - API key:  EVAL_JUDGE_API_KEY → OPENAI_API_KEY

Override via env vars:
  EVAL_JUDGE_MODEL     judge model
  EVAL_JUDGE_API_BASE  judge endpoint
  EVAL_JUDGE_API_KEY   judge key
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

import openai


def _judge_config() -> tuple[str, str, str]:
    model = os.getenv("EVAL_JUDGE_MODEL", "llama-3.3-70b-versatile")
    base = os.getenv("EVAL_JUDGE_API_BASE", "https://api.groq.com/openai/v1")
    key = os.getenv("EVAL_JUDGE_API_KEY") or os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("No judge API key: set EVAL_JUDGE_API_KEY or GROQ_API_KEY")
    return model, base, key


def _extract_json_scores(text: str) -> dict[str, Any] | None:
    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", text.strip(),
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _clamp_score(value: Any, fallback: float | None = None) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, score))


def _select_judge_context(
    contexts: list[str],
    question: str,
    answer: str,
    gold_answer: str,
    max_total: int = 12000,
) -> str:
    stop = {
        "the", "and", "for", "with", "from", "that", "what", "which",
        "according", "document", "documents", "report", "policy", "terms",
        "does", "about",
    }
    anchors = {
        t.lower()
        for t in re.findall(
            r"[A-Za-z0-9$£€.,¼½¾/-]{2,}", f"{question} {answer} {gold_answer}"
        )
        if t.lower() not in stop
    }
    scored: list[tuple[int, int, str]] = []
    for idx, context in enumerate(contexts):
        text = context.strip()
        lower = text.lower()
        score = sum(1 for a in anchors if a in lower)
        score += 3 * sum(
            1
            for tok in re.findall(
                r"[$£€]?\d[\d,./]*|[¼½¾]", f"{answer} {gold_answer}"
            )
            if tok in text
        )
        scored.append((score, -idx, text))

    selected = [t for s, _, t in sorted(scored, reverse=True) if t][:8]
    if not selected:
        selected = [c.strip() for c in contexts if c.strip()][:8]

    packed, used = [], 0
    for text in selected:
        remaining = max_total - used
        if remaining <= 0:
            break
        piece = text[:remaining]
        packed.append(piece)
        used += len(piece)
    return "\n\n---\n\n".join(packed) if packed else "No context retrieved."


_JUDGE_PROMPT = (  # noqa: E501
    "You are grading a RAG system. Return ONLY valid JSON with numeric scores from 0 to 1.\n"  # noqa: E501
    "Use this schema exactly: "
    '{{"faithfulness": number, "answer_relevancy": number, "reason": string}}\n\n'
    "Scoring rules:\n"
    "- faithfulness: judge at the CLAIM level (RAGAS-style) — a claim is supported if it can be "  # noqa: E501
    "inferred from the RETRIEVED CONTEXT, not only if it appears verbatim. A comparison, ranking, or "  # noqa: E501
    "conclusion that follows from facts that ARE present in the context (e.g. 'X allows a longer term "  # noqa: E501
    "than Y' when X's and Y's terms are both in the context) IS supported — score it faithful. "  # noqa: E501
    "Do not require exact wording; values under matching field labels count as support. "  # noqa: E501
    "Do not penalize missing citations. Penalize only claims that CONTRADICT the context, introduce "  # noqa: E501
    "facts ABSENT from the context, or mix values across the wrong sources.\n"
    "- answer_relevancy: score whether ACTUAL ANSWER directly addresses the QUESTION. "
    "Concise direct answers are relevant.\n\n"
    "QUESTION:\n{question}\n\n"
    "ACTUAL ANSWER:\n{answer}\n\n"
    "RETRIEVED CONTEXT:\n{context}\n"
)


def judge_answer(
    question: str,
    answer: str,
    gold_answer: str,
    contexts: list[str],
) -> dict[str, Any]:
    """Score one RAG answer. Returns {faithfulness, answer_relevancy, reason}."""
    model, base, key = _judge_config()
    context = _select_judge_context(contexts, question, answer, gold_answer)

    prompt = _JUDGE_PROMPT.format(question=question, answer=answer, context=context)

    client = openai.OpenAI(base_url=base, api_key=key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a strict but fair evaluation judge. Output valid JSON only.",  # noqa: E501
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    parsed = _extract_json_scores(raw)
    if not parsed:
        raise RuntimeError(f"judge returned invalid JSON: {raw[:200]!r}")

    return {
        "faithfulness": _clamp_score(parsed.get("faithfulness")),
        "answer_relevancy": _clamp_score(parsed.get("answer_relevancy")),
        "reason": parsed.get("reason", ""),
    }
