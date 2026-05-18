"""
Orion eval harness — fully local, no LangSmith dependency.

Evaluators
----------
- correctness     : LLM-as-judge (llama-3.3-70b-versatile) vs expected answer
- tool_selection  : exact match between tools called and expected_tool category
- faithfulness    : custom claim-level judge (gpt-4o-mini) — rag_only only
- answer_relevancy: custom judge — all RAG categories

Results are written to eval/<experiment>.json and eval/<experiment>-summary.json
after every example, so nothing is lost if the run is interrupted.

Usage
-----
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --experiment orion-v2
    uv run python eval/run_eval.py --skip-escalation
    uv run python eval/run_eval.py --limit 5
"""

import argparse
import json
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Disable LangSmith tracing before any LangChain import
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()
# Re-disable after load_dotenv in case .env sets it
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from agent.config import AGENT_MODEL  # noqa: E402
from eval.judge import judge_answer  # noqa: E402

orion_graph = None  # initialised in main()

DATASET_PATH = Path(__file__).parent / "dataset.json"

_counter_lock = threading.Lock()
_counter = 0
_total = 0

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOOL_CATEGORY_MAP = {
    "sql_only": {"query_database"},
    "rag_only": {"search_policies"},
    "both_tools": {"query_database", "search_policies"},
    "both": {"query_database", "search_policies"},
    "escalation": {"escalate"},
    "adversarial": set(),
}

FAITHFULNESS_CATEGORIES = {"rag_only"}
RAG_CATEGORIES = {"rag_only", "both_tools", "both"}
SCORED_TOOL_CATEGORIES = {
    "sql_only", "rag_only", "both_tools", "both", "escalation", "adversarial",
}

# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------

_judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

_CORRECTNESS_PROMPT = """\
You are evaluating a customer support AI agent for an e-commerce store.

Question: {question}
Expected answer: {expected}
Agent answer: {actual}

Score the agent answer from 0 to 1 using these criteria:

1.0 — The core answer is correct and complete for what was asked.
0.75 — The core answer is correct but omits supplementary context that was NOT
       directly asked for (e.g. the customer asked for order status and the agent
       gave the correct status but did not mention the purchase date or payment
       method). Also use 0.75 if one minor detail is missing from the conclusion.
0.5 — The agent has the right direction but the primary fact asked for is missing
      or wrong (e.g. gave a delivery date when freight cost was asked, or said
      "cannot find" when the order exists). For both-tool questions: use 0.5 if
      only one of the two required parts (order fact OR policy rule) is present.
0.25 — Key facts are wrong, hallucinated, or the conclusion contradicts the data.
0.0 — Wrong answer, refused to answer, or completely irrelevant.

Important: do NOT penalise the agent for omitting details the customer did not ask
for. Judge only whether the question was correctly answered.

For questions requiring BOTH a database lookup AND a policy rule, the answer must
include both the specific order fact AND the policy conclusion to score above 0.5.

Reply with ONLY a number: 0, 0.25, 0.5, 0.75, or 1.0. No explanation."""


def _run_one(example: dict, n: int) -> dict:
    """Run agent + all evaluators for a single example. Returns a result row."""
    question = example["question"]
    expected_answer = example["expected_answer"]
    expected_tool = example["expected_tool"]

    print(f"[{n}/{_total}] {question[:90]}", flush=True)
    t0 = time.monotonic()

    thread_id = str(uuid.uuid4())
    result = orion_graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result["messages"]
    answer = next(
        (m.content for m in reversed(messages)
         if isinstance(m, AIMessage) and m.content),
        "",
    )
    tools_called = {
        tc["name"]
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    }
    chunks = result.get("last_chunks", [])
    contexts = [c["content"] for c in chunks] if chunks else []

    elapsed = time.monotonic() - t0
    print(f"[{n}/{_total}] done in {elapsed:.1f}s — {answer[:80]!r}", flush=True)

    row: dict = {
        "question": question,
        "expected_answer": expected_answer,
        "expected_tool": expected_tool,
        "answer": answer,
        "tools_called": list(tools_called),
    }

    # correctness
    try:
        prompt = _CORRECTNESS_PROMPT.format(
            question=question, expected=expected_answer, actual=answer
        )
        resp = _judge.invoke(prompt)
        row["correctness"] = max(0.0, min(1.0, float(resp.content.strip())))
    except Exception as e:
        print(f"  [correctness ERROR] {e}", flush=True)
        row["correctness"] = None

    # tool_selection
    if expected_tool in SCORED_TOOL_CATEGORIES:
        expected_tools = TOOL_CATEGORY_MAP.get(expected_tool, set())
        row["tool_selection"] = 1.0 if tools_called == expected_tools else 0.0
    else:
        row["tool_selection"] = None

    # faithfulness + answer_relevancy (one judge call)
    want_faithfulness = expected_tool in FAITHFULNESS_CATEGORIES and bool(contexts)
    want_relevancy = expected_tool in RAG_CATEGORIES

    if want_faithfulness or want_relevancy:
        try:
            rag_scores = judge_answer(
                question=question,
                answer=answer,
                gold_answer=expected_answer,
                contexts=contexts,
            )
            row["faithfulness"] = (
                rag_scores.get("faithfulness") if want_faithfulness else None
            )
            row["answer_relevancy"] = (
                rag_scores.get("answer_relevancy") if want_relevancy else None
            )
        except Exception:
            print(f"  [rag_judge ERROR]\n{traceback.format_exc()}", flush=True)
            row["faithfulness"] = None
            row["answer_relevancy"] = None
    else:
        row["faithfulness"] = None
        row["answer_relevancy"] = None

    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Run Orion evaluation — fully local, results saved to JSON.",
    )
    parser.add_argument(
        "--experiment",
        metavar="NAME",
        default="orion-v1",
        help="Experiment name — used as output filename prefix.",
    )
    parser.add_argument(
        "--skip-escalation",
        action="store_true",
        help="Skip escalation test cases.",
    )
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Only run the first N examples (smoke test).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    global orion_graph
    from agent.graph import graph as _g

    orion_graph = _g
    print(f"Using Groq ({AGENT_MODEL}) for agent runs.")

    examples = json.loads(DATASET_PATH.read_text())
    if args.skip_escalation:
        examples = [e for e in examples if e.get("category") != "escalation"]
    if args.limit:
        examples = examples[: args.limit]

    global _total, _counter
    _total = len(examples)
    _counter = 0

    out_path = Path(__file__).parent / f"{args.experiment}.json"
    summary_path = Path(__file__).parent / f"{args.experiment}-summary.json"
    print(f"\nRunning '{args.experiment}' — {_total} examples → {out_path}\n")

    rows: list[dict] = []
    write_lock = threading.Lock()

    def _task(example: dict) -> dict:
        global _counter
        with _counter_lock:
            _counter += 1
            n = _counter
        row = _run_one(example, n)
        with write_lock:
            rows.append(row)
            out_path.write_text(json.dumps(rows, indent=2))
        return row

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_task, ex) for ex in examples]
        done = 0
        for _ in as_completed(futures):
            done += 1
            print(f"  completed {done}/{_total}", flush=True)

    metrics = ["correctness", "tool_selection", "faithfulness", "answer_relevancy"]
    summary: dict = {"experiment": args.experiment, "n": _total}
    for m in metrics:
        vals = [r[m] for r in rows if r.get(m) is not None]
        summary[m] = round(sum(vals) / len(vals), 4) if vals else None
        summary[f"{m}_n"] = len(vals)

    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n── Results ──────────────────────────────")
    for m in metrics:
        if summary.get(m) is not None:
            print(f"  {m:<20} {summary[m]:.2f}  ({summary[f'{m}_n']} examples)")
    print(f"\nSaved → {out_path}")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()
