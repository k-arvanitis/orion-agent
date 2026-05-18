"""
Orion eval harness — uploads the dataset to LangSmith and runs an experiment.

Evaluators
----------
- correctness   : LLM-as-judge (Llama 4 Scout via Groq) comparing agent answer
                  vs expected answer
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
import concurrent.futures
import json
import sys
import threading
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from langsmith import Client, evaluate
from langsmith.schemas import Example, Run
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")

from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

# Ensure project root is on the path when run from the eval/ subdirectory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from agent.config import AGENT_MODEL, chat_groq_kwargs  # noqa: E402
from agent.embeddings import dense_embed  # noqa: E402


class _LocalEmbeddings(Embeddings):
    """Thin LangChain Embeddings adapter over agent.embeddings.dense_embed.

    RAGAS calls embed_query / embed_documents on this object; both delegate to
    the same fastembed encoder used by the RAG tool, so eval-time and
    runtime-time embeddings stay identical.
    """

    def embed_query(self, text: str) -> list[float]:
        return dense_embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [dense_embed(t) for t in texts]


orion_graph = None  # initialised in main()

DATASET_NAME = "orion-eval"
DATASET_PATH = Path(__file__).parent / "dataset.json"

_counter_lock = threading.Lock()
_counter = 0
_total = 0


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
    global _counter
    with _counter_lock:
        _counter += 1
        n = _counter
    question = inputs["question"]
    print(f"[{n}/{_total}] {question[:90]}", flush=True)
    t0 = time.monotonic()

    thread_id = str(uuid.uuid4())
    result = orion_graph.invoke(
        {"messages": [{"role": "user", "content": inputs["question"]}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    messages = result["messages"]
    answer = next(
        (
            m.content
            for m in reversed(messages)
            if isinstance(m, AIMessage) and m.content
        ),
        "",
    )

    tools_called = list(
        {
            tc["name"]
            for m in messages
            if isinstance(m, AIMessage)
            for tc in (m.tool_calls or [])
        }
    )

    chunks = result.get("last_chunks", [])
    contexts = [c["content"] for c in chunks] if chunks else []

    elapsed = time.monotonic() - t0
    print(f"[{n}/{_total}] done in {elapsed:.1f}s — {answer[:80]!r}", flush=True)
    return {"answer": answer, "tools_called": tools_called, "contexts": contexts}


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

_judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
_ragas_llm = LangchainLLMWrapper(
    ChatGroq(model=AGENT_MODEL, temperature=0, **chat_groq_kwargs())
)
_ragas_embeddings = LangchainEmbeddingsWrapper(_LocalEmbeddings())
_faithfulness = Faithfulness(llm=_ragas_llm)
_answer_relevancy = AnswerRelevancy(llm=_ragas_llm, embeddings=_ragas_embeddings)
_context_precision = ContextPrecision(llm=_ragas_llm)
_context_recall = ContextRecall(llm=_ragas_llm)

_CORRECTNESS_PROMPT = """\
You are evaluating a customer support AI agent for an e-commerce store.

Question: {question}
Expected answer: {expected}
Agent answer: {actual}

Score the agent answer from 0 to 1 using these criteria:

1.0 — All key facts are correct AND the conclusion/recommendation is correct.
0.75 — All key facts are correct but the conclusion is incomplete or one minor
       detail is missing (e.g. correct policy rule stated but no explicit
       eligibility verdict given).
0.5 — One of two required parts is correct (e.g. order fact retrieved correctly
      but policy rule missing, OR policy rule correct but wrong order facts).
      Also use 0.5 if the answer is directionally right but missing a specific
      date, amount, or threshold that the expected answer includes.
0.25 — The agent attempted to answer but key facts are wrong, hallucinated, or
       the conclusion contradicts the evidence.
0.0 — Wrong answer, refused to answer, or completely irrelevant.

For questions that require BOTH a database lookup AND a policy rule, the answer
must include both the specific order fact (date, amount, category) AND the policy
conclusion to score above 0.5.

Reply with ONLY a number: 0, 0.25, 0.5, 0.75, or 1.0. No explanation."""

TOOL_CATEGORY_MAP = {
    "sql_only": {"query_database"},
    "rag_only": {"search_policies"},
    "both_tools": {"query_database", "search_policies"},
    "both": {"query_database", "search_policies"},  # dataset alias for both_tools
    "escalation": {"escalate"},
    # adversarial: agent should call NO tools — correct answer is empty set
    "adversarial": set(),
}

RAG_CATEGORIES = {"rag_only", "both_tools", "both"}

# Categories where tool selection scoring is meaningful
SCORED_TOOL_CATEGORIES = {
    "sql_only",
    "rag_only",
    "both_tools",
    "both",
    "escalation",
    "adversarial",
}


_RAGAS_TIMEOUT = 60  # seconds per metric call before giving up


def _ragas_score(metric, sample) -> float | None:
    """Run a RAGAS metric with a timeout; return None if it hangs or errors."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(metric.single_turn_score, sample)
        try:
            return float(future.result(timeout=_RAGAS_TIMEOUT))
        except concurrent.futures.TimeoutError:
            print(f"  [timeout] {metric.__class__.__name__} exceeded {_RAGAS_TIMEOUT}s — skipping", flush=True)
            future.cancel()
            return None
        except Exception:
            return None


def correctness_evaluator(run: Run, example: Example) -> dict:
    question = example.inputs["question"]
    expected = example.outputs["expected_answer"]
    actual = (run.outputs or {}).get("answer", "")

    print(f"  scoring correctness: {question[:70]}", flush=True)
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
    print(f"  scoring faithfulness: {example.inputs['question'][:70]}", flush=True)
    return {"key": "faithfulness", "score": _ragas_score(_faithfulness, sample)}


def answer_relevancy_evaluator(run: Run, example: Example) -> dict:
    sample = _build_rag_sample(run, example)
    if not sample:
        return {"key": "answer_relevancy", "score": None}
    print(f"  scoring answer_relevancy: {example.inputs['question'][:70]}", flush=True)
    return {"key": "answer_relevancy", "score": _ragas_score(_answer_relevancy, sample)}


def context_precision_evaluator(run: Run, example: Example) -> dict:
    sample = _build_rag_sample(run, example)
    if not sample:
        return {"key": "context_precision", "score": None}
    print(f"  scoring context_precision: {example.inputs['question'][:70]}", flush=True)
    return {"key": "context_precision", "score": _ragas_score(_context_precision, sample)}


def context_recall_evaluator(run: Run, example: Example) -> dict:
    sample = _build_rag_sample(run, example)
    if not sample:
        return {"key": "context_recall", "score": None}
    print(f"  scoring context_recall: {example.inputs['question'][:70]}", flush=True)
    return {"key": "context_recall", "score": _ragas_score(_context_recall, sample)}


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
        help=(
            "Skip escalation test cases (avoids sending real Slack/Gmail during eval)."
        ),
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

    global _total, _counter
    _counter = 0

    data = DATASET_NAME
    if args.limit:
        examples = list(
            client.list_examples(dataset_name=DATASET_NAME, limit=args.limit)
        )
        data = examples
        _total = len(examples)
    else:
        _total = sum(1 for _ in client.list_examples(dataset_name=DATASET_NAME))

    print(f"\nRunning experiment '{args.experiment}' — {_total} examples ...")
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
        "correctness": [],
        "tool_selection": [],
        "faithfulness": [],
        "answer_relevancy": [],
        "context_precision": [],
        "context_recall": [],
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
