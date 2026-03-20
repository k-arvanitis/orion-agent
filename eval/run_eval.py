"""
Orion eval harness — uploads the dataset to LangSmith and runs an experiment.

Evaluators
----------
- correctness   : LLM-as-judge (Groq) comparing agent answer vs expected answer
- tool_selection: exact match between tools called and expected_tool category

Usage
-----
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --experiment orion-v2
    uv run python eval/run_eval.py --skip-escalation   # skip escalation cases
"""

import argparse
import json
import sys
import uuid
from pathlib import Path

# Ensure project root is on the path when run from the eval/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langsmith import Client, evaluate
from langsmith.schemas import Example, Run

from agent.graph import graph as orion_graph

DATASET_NAME = "orion-eval"
DATASET_PATH = Path(__file__).parent / "dataset.json"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def upload_dataset(client: Client, skip_escalation: bool) -> None:
    """Create the LangSmith dataset if it doesn't exist yet."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists — skipping upload.")
        return

    examples = json.loads(DATASET_PATH.read_text())
    if skip_escalation:
        examples = [e for e in examples if e["category"] != "escalation"]

    client.create_dataset(DATASET_NAME, description="Orion agent evaluation set")
    client.create_examples(
        inputs=[{"question": e["question"]} for e in examples],
        outputs=[
            {
                "expected_answer": e["expected_answer"],
                "expected_tool": e["expected_tool"],
            }
            for e in examples
        ],
        dataset_name=DATASET_NAME,
    )
    print(f"Uploaded {len(examples)} examples to '{DATASET_NAME}'.")


# ---------------------------------------------------------------------------
# Target function
# ---------------------------------------------------------------------------


def run_agent(inputs: dict) -> dict:
    """Run a single question through the agent and return answer + tools used."""
    thread_id = str(uuid.uuid4())
    result = orion_graph.invoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result["messages"]
    answer = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        "",
    )

    tools_called = list({
        tc["name"]
        for m in messages
        if isinstance(m, AIMessage)
        for tc in (m.tool_calls or [])
    })

    return {"answer": answer, "tools_called": tools_called}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_judge = ChatOllama(model="qwen3:8b", temperature=0)

_CORRECTNESS_PROMPT = """\
You are evaluating a customer support AI agent.

Question: {question}
Expected answer: {expected}
Agent answer: {actual}

Score the agent answer from 0 to 1:
- 1.0  : correct and complete
- 0.75 : mostly correct, minor omission
- 0.5  : partially correct
- 0.25 : relevant but wrong details
- 0.0  : wrong or hallucinated

Reply with ONLY a number between 0 and 1. No explanation."""

TOOL_CATEGORY_MAP = {
    "sql_only":  {"query_database"},
    "rag_only":  {"search_policies"},
    "both":      {"query_database", "search_policies"},
    "escalation": {"escalate"},
}


def correctness_evaluator(run: Run, example: Example) -> dict:
    question = example.inputs["question"]
    expected = example.outputs["expected_answer"]
    actual = (run.outputs or {}).get("answer", "")

    prompt = _CORRECTNESS_PROMPT.format(
        question=question, expected=expected, actual=actual
    )
    response = _judge.invoke(prompt)
    try:
        score = float(response.content.strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.0

    return {"key": "correctness", "score": score}


def tool_selection_evaluator(run: Run, example: Example) -> dict:
    expected_tool = example.outputs.get("expected_tool", "")
    tools_called = set((run.outputs or {}).get("tools_called", []))
    expected_tools = TOOL_CATEGORY_MAP.get(expected_tool, set())

    score = 1.0 if tools_called == expected_tools else 0.0
    return {"key": "tool_selection", "score": score}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_eval",
        description="Run Orion evaluation experiment on LangSmith.",
    )
    parser.add_argument(
        "--experiment",
        metavar="NAME",
        default="orion-v1",
        help="Experiment prefix in LangSmith (default: orion-v1).",
    )
    parser.add_argument(
        "--skip-escalation",
        action="store_true",
        help="Skip escalation test cases (avoids sending real Slack/Gmail during eval).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = Client()

    upload_dataset(client, skip_escalation=args.skip_escalation)

    print(f"\nRunning experiment '{args.experiment}' ...")
    results = evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, tool_selection_evaluator],
        experiment_prefix=args.experiment,
        max_concurrency=1,
    )

    scores = {"correctness": [], "tool_selection": []}
    for r in results:
        for fb in (r.get("evaluation_results") or {}).get("results", []):
            if fb.key in scores:
                scores[fb.key].append(fb.score)

    print("\n── Results ──────────────────────────────")
    for metric, values in scores.items():
        if values:
            avg = sum(values) / len(values)
            print(f"  {metric:<20} {avg:.2f}  ({len(values)} examples)")
    print(f"\nFull results → LangSmith project '{client._settings.project}'")


if __name__ == "__main__":
    main()
