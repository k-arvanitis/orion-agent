"""
Orion eval harness — uploads the dataset to LangSmith and runs an experiment.

Evaluators
----------
- correctness   : LLM-as-judge (Llama 4 Scout via Groq) comparing agent answer vs expected answer
- tool_selection: exact match between tools called and expected_tool category
- faithfulness  : RAGAS — is the answer grounded in the retrieved chunks?
                  Only runs for rag_only and both_tools categories.

Usage
-----
    uv run python eval/run_eval.py
    uv run python eval/run_eval.py --experiment orion-v2
    uv run python eval/run_eval.py --skip-escalation   # skip escalation cases
    uv run python eval/run_eval.py --limit 5           # smoke test with 5 examples
"""

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

# Ensure project root is on the path when run from the eval/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langsmith import Client, evaluate
from langsmith.schemas import Example, Run
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall  # noqa: deprecated import, collections API requires OpenAI-only llm_factory
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from agent.config import AGENT_MODEL

orion_graph = None  # initialised in main()

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

    chunks = result.get("last_chunks", [])
    contexts = [c["content"] for c in chunks] if chunks else []

    return {"answer": answer, "tools_called": tools_called, "contexts": contexts}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_judge = ChatGroq(model=AGENT_MODEL, temperature=0)
_ragas_llm = LangchainLLMWrapper(ChatGroq(model=AGENT_MODEL, temperature=0))
_ragas_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
_faithfulness      = Faithfulness(llm=_ragas_llm)
_answer_relevancy  = AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_embeddings)
_context_precision = ContextPrecision(llm=_ragas_llm)
_context_recall    = ContextRecall(llm=_ragas_llm)

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
    "sql_only":   {"query_database"},
    "rag_only":   {"search_policies"},
    "both_tools": {"query_database", "search_policies"},
    "both":       {"query_database", "search_policies"},  # dataset alias for both_tools
    "escalation": {"escalate"},
    # adversarial: agent should call NO tools — correct answer is empty set
    "adversarial": set(),
}

RAG_CATEGORIES = {"rag_only", "both_tools", "both"}

# Categories where tool selection scoring is meaningful
SCORED_TOOL_CATEGORIES = {"sql_only", "rag_only", "both_tools", "both", "escalation", "adversarial"}


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


def _build_rag_sample(run: Run, example: Example) -> SingleTurnSample | None:
    if example.outputs.get("expected_tool") not in RAG_CATEGORIES:
        return None
    answer = (run.outputs or {}).get("answer", "")
    contexts = (run.outputs or {}).get("contexts", [])
    if not contexts:
        return None
    return SingleTurnSample(
        user_input=example.inputs["question"],
        response=answer,
        retrieved_contexts=contexts,
        reference=example.outputs.get("expected_answer", ""),
    )


def faithfulness_evaluator(run: Run, example: Example) -> dict:
    sample = _build_rag_sample(run, example)
    if not sample:
        return {"key": "faithfulness", "score": None}
    try:
        return {"key": "faithfulness", "score": float(_faithfulness.single_turn_score(sample))}
    except Exception:
        return {"key": "faithfulness", "score": None}


def answer_relevancy_evaluator(run: Run, example: Example) -> dict:
    sample = _build_rag_sample(run, example)
    if not sample:
        return {"key": "answer_relevancy", "score": None}
    try:
        return {"key": "answer_relevancy", "score": float(_answer_relevancy.single_turn_score(sample))}
    except Exception:
        return {"key": "answer_relevancy", "score": None}


def context_precision_evaluator(run: Run, example: Example) -> dict:
    sample = _build_rag_sample(run, example)
    if not sample:
        return {"key": "context_precision", "score": None}
    try:
        return {"key": "context_precision", "score": float(_context_precision.single_turn_score(sample))}
    except Exception:
        return {"key": "context_precision", "score": None}


def context_recall_evaluator(run: Run, example: Example) -> dict:
    sample = _build_rag_sample(run, example)
    if not sample:
        return {"key": "context_recall", "score": None}
    try:
        return {"key": "context_recall", "score": float(_context_recall.single_turn_score(sample))}
    except Exception:
        return {"key": "context_recall", "score": None}


def tool_selection_evaluator(run: Run, example: Example) -> dict:
    expected_tool = example.outputs.get("expected_tool", "")
    if expected_tool not in SCORED_TOOL_CATEGORIES:
        return {"key": "tool_selection", "score": None}

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
    parser.add_argument(
        "--limit",
        metavar="N",
        type=int,
        default=None,
        help="Only run the first N examples (useful for quick smoke tests).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    client = Client()

    global orion_graph
    from agent.graph import graph as _g
    orion_graph = _g
    print(f"Using Groq ({AGENT_MODEL}) for agent runs.")

    upload_dataset(client, skip_escalation=args.skip_escalation)

    data = DATASET_NAME
    if args.limit:
        examples = list(client.list_examples(dataset_name=DATASET_NAME, limit=args.limit))
        data = examples

    print(f"\nRunning experiment '{args.experiment}' ...")
    results = evaluate(
        run_agent,
        data=data,
        evaluators=[
            correctness_evaluator,
            tool_selection_evaluator,
            faithfulness_evaluator,
            answer_relevancy_evaluator,
            context_precision_evaluator,
            context_recall_evaluator,
        ],
        experiment_prefix=args.experiment,
        max_concurrency=4,
    )

    scores = {
        "correctness": [], "tool_selection": [],
        "faithfulness": [], "answer_relevancy": [],
        "context_precision": [], "context_recall": [],
    }
    for r in results:
        for fb in (r.get("evaluation_results") or {}).get("results", []):
            if fb.key in scores and fb.score is not None:
                scores[fb.key].append(fb.score)

    print("\n── Results ──────────────────────────────")
    for metric, values in scores.items():
        if values:
            avg = sum(values) / len(values)
            print(f"  {metric:<20} {avg:.2f}  ({len(values)} examples)")
    print("\nFull results → LangSmith dashboard")


if __name__ == "__main__":
    main()
