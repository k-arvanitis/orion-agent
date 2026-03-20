# Orion — AI Customer Support Agent

Orion is a production-minded AI agent for e-commerce customer support. It answers natural language questions by routing them to the right data source, handles multi-step queries that require both structured and unstructured data, and escalates unresolvable issues to a human operator via Slack and Gmail.

Built on a fictional Brazilian e-commerce store (ShopNova) using the real [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

---

## Architecture

```
User query
    │
    ▼
┌─────────────────────────────────────────────────┐
│               LangGraph ReAct Agent              │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌─────────────┐  │
│  │ RAG Tool │   │ SQL Tool │   │ Escalation  │  │
│  │          │   │          │   │    Tool     │  │
│  │  Qdrant  │   │ Supabase │   │Slack+Gmail  │  │
│  │ (hybrid) │   │(Text2SQL)│   │             │  │
│  └──────────┘   └──────────┘   └─────────────┘  │
│                                                  │
│  ┌──────────────────────────────────────────┐    │
│  │  Guard Layer (PII strip + hallucination) │    │
│  └──────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
    │
    ▼
Response + LangSmith trace
```

The agent runs a ReAct loop: it decides which tool(s) to call based on the question, executes them, and synthesizes a response. A post-processing guard strips PII and flags hallucinated numbers before the response reaches the user.

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph (stateful ReAct agent) |
| LLM | Groq — Llama 4 Scout |
| RAG | Qdrant Cloud (hybrid dense + sparse search) |
| Embeddings | nomic-embed-text + BM25 via Ollama / fastembed |
| Database | Supabase PostgreSQL (Olist dataset, 9 tables) |
| Text2SQL | SQLAlchemy + sqlparse validation |
| Escalation | Gmail API + Slack Incoming Webhooks |
| Observability | LangSmith |
| Evaluation | LangSmith + LLM-as-judge (qwen3:8b via Ollama) |

---

## Tools

### `search_policies` — RAG over policy documents
Hybrid search combining dense semantic vectors (nomic-embed-text, 768-dim) and sparse BM25 keyword vectors, fused with Reciprocal Rank Fusion (RRF). Covers payments, returns, shipping, and warranty policies.

### `query_database` — Text2SQL over order data
Translates natural language to SQL using Llama 4 Scout with full schema context injected. Validated by sqlparse (SELECT-only, table/column whitelist). One retry loop with error feedback. 100-row limit, 5s timeout.

### `escalate` — Human handoff
Triggered when the agent cannot resolve an issue or the customer requests a human. Fetches order details from Supabase, sends a confirmation email to the customer via Gmail, and posts an urgent alert to the operator Slack channel.

---

## Guard Layer

Every agent response passes through a two-step filter before reaching the user:

1. **PII stripping** — regex patterns remove Brazilian CPF numbers and phone numbers
2. **Hallucination check** — any number in the response must appear in the tool output; if not, the agent is re-prompted with a correction

---

## Setup

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with `nomic-embed-text` and `qwen3:8b`

### Install
```bash
git clone https://github.com/your-username/orion-agent
cd orion-agent
uv sync
```

### Environment variables
Copy `.env.example` to `.env` and fill in your keys:
```bash
cp .env.example .env
```

Required:
```
DATABASE_URL=postgresql://...
QDRANT_URL=https://...
QDRANT_API_KEY=...
GROQ_API_KEY=...
LANGCHAIN_API_KEY=...
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

For Gmail escalation, run the one-time OAuth flow:
```bash
uv run python scripts/auth_gmail.py
```

### Ingest policies into Qdrant
```bash
uv run chunker data/policies
uv run python ingestion/ingest.py
```

### Run the agent (CLI)
```bash
uv run python main.py
```

### Run the Streamlit UI
```bash
uv run streamlit run ui/app.py
```

---

## Example Questions

**Order lookup (SQL)**
```
What is the status of order 416e49799e9260d93c8f636ce6661a55?
How much did I pay for order 1e8c81805b92ff169971231458670460?
```

**Policy lookup (RAG)**
```
What payment methods does ShopNova accept?
How long do I have to return a product?
```

**Multi-tool (SQL + RAG)**
```
My order arrived late — am I eligible for a refund?
I want to return order e481f51cbdc54678b7cc49136f2d6af7. How much will I get back?
```

**Escalation**
```
I want to speak to a real person about my order.
This is unacceptable, I need a human to help me.
```

---

## Evaluation

The eval harness runs 30 labeled test cases through the agent and scores them with LLM-as-judge (qwen3:8b) for correctness and tool selection accuracy. Results are logged to LangSmith.

```bash
uv run python eval/run_eval.py --skip-escalation
```

---

## Project Structure

```
orion-agent/
├── agent/
│   ├── graph.py              # LangGraph ReAct agent
│   ├── guard.py              # PII filter + hallucination check
│   ├── prompts.py            # System prompt with DB schema
│   └── tools/
│       ├── rag_tool.py       # Hybrid Qdrant search
│       ├── sql_tool.py       # Text2SQL over Supabase
│       └── escalation_tool.py
├── ingestion/
│   ├── chunker.py            # Markdown → heading-based chunks
│   ├── ingest.py             # Embed + push to Qdrant
│   └── load_customer_data.py # CSV → Supabase with type inference
├── eval/
│   ├── run_eval.py           # LangSmith eval harness
│   └── dataset.json          # 30 labeled test cases
├── scripts/
│   └── auth_gmail.py         # One-time Gmail OAuth setup
├── data/
│   └── policies/             # Markdown policy documents
├── main.py                   # CLI entry point
└── .env.example
```
