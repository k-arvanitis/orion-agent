# orion-agent — Claude Code instructions

## What this is
AI customer-support agent for e-commerce: a LangGraph ReAct agent that deflects tier-1 tickets — order-status lookups (Text2SQL over a live DB), policy questions (hybrid RAG), and combined queries — with sources shown to the customer and a clean Slack + Gmail human-escalation path. Domain: e-commerce support over the Olist dataset (fictional store "ShopNova"). Real order data, synthetic policy docs.

## Conventions
- Package manager: **uv only** (`uv sync --frozen`), never pip. Python **>=3.11,<3.12**. Frontend is Next.js 14 — `npm` there only.
- Lint: **ruff** (`uv run ruff check .` must be clean, CI runs it before tests). Line length 88, `select = E,F,I`.
- Tests: **pytest, ALL external services mocked** — no live Groq/Qdrant/Supabase/Gmail/Slack/ElevenLabs/dense-encoder/network. `uv run pytest -q` (**40 tests**). `make test`.
- Config centralized in `agent/config.py` (all model names + defaults). Nothing hardcoded elsewhere.
- Errors caught per dependency — every external dep degrades independently; the system never returns a silent empty answer.
- ML models (fastembed dense encoder) get a warm-up call after load; first-use downloads ~133 MB BGE weights, then ~2 ms/embed.
- English everywhere. NEVER attribute commits/PRs/comments to Claude/AI/any assistant.

## Architecture (data flow)
```
Next.js 14 UI ──fetch+NDJSON stream──▶ FastAPI (:8088)  POST /api/chat, /api/transcribe, /api/tts
                                              ▼
                          LangGraph ReAct agent (OrionState per thread_id)
       ┌────────────────┬────────────────────┬──────────────────┐
   search_policies   query_database        escalate
   (Qdrant hybrid)   (Text2SQL→Supabase)   (Slack + Gmail, independent)
       └────────────────┴────────────────────┴──────────────────┘
                          Guard layer (PII strip) on every final response
```
ReAct loop decides which tool(s) to call. Tools return `{"answer", "chunks"|"sql"}`; the agent sees only `answer`, raw source data is stored in `OrionState` and forwarded to the UI in the final `trace` event.

| File (`agent/`) | Responsibility |
|---|---|
| `graph.py` | LangGraph ReAct agent + OrionState + `should_continue` routing |
| `embeddings.py` | fastembed BGE dense + fastembed BM25 sparse |
| `guard.py` | PII filter (Brazilian CPF + phone regex) |
| `prompts.py` | system prompt with tool-reasoning examples |
| `voice.py` | Groq Whisper + ElevenLabs (UI-only I/O wrapper) |
| `tools/rag_tool.py` / `sql_tool.py` / `escalation_tool.py` | the three tools |
| `api/main.py` / `api/schemas.py` | FastAPI app + Pydantic models |
| `ingestion/` / `eval/` | chunk+embed→Qdrant, CSV→Supabase / eval harness + judge + dataset |

## Key decisions & gotchas (read before editing)
- **Structured tool isolation** — the agent receives only the `answer` field; `chunks`/`sql` live in `OrionState` for the UI trace, never in the LLM's reasoning context.
- **PII guard runs AFTER the ReAct loop**, before the response reaches the user — does not affect tool execution.
- **Hybrid retrieval** = dense (fastembed `bge-small-en-v1.5`, 384-dim, local ONNX) + BM25, fused with **RRF in a single Qdrant prefetch**. Sparse is not optional — policy docs have exact terms ("Boleto", "CPF", "30-day return window").
- **SELECT-only SQL** validated by sqlparse (DML rejection, markdown-fence stripping); on failure the error is fed back to the LLM for one retry. NL SQL-injection is in the adversarial eval set.
- **Escalation: Slack and Gmail fire independently** — if Slack is down the email still sends, and vice versa. Never silently drop a ticket.
- **Partial-failure resilience** — Qdrant/embed-load failure → `search_policies` returns "temporarily unavailable", SQL still works; DB down → `query_database` retries once then fallback, RAG still works.
- LLM = Groq `qwen/qwen3-32b` (swappable to local vLLM via env). Voice is an I/O wrapper around the unchanged agent core — eval numbers carry over.
- **OrionState is persisted via SqliteSaver** (`data/checkpoints.db`) — conversations survive restarts. Override path with `CHECKPOINT_DB_PATH` env var.

## Running
```bash
uv sync --frozen
cd frontend && npm install && cd ..
cp .env.example .env   # DATABASE_URL, QDRANT_URL/API_KEY, GROQ_API_KEY, SLACK_WEBHOOK_URL; ELEVENLABS_API_KEY (voice only)
uv run --frozen python scripts/auth_gmail.py   # one-time Gmail OAuth
make ingest        # embed policy docs → Qdrant
make stack         # FastAPI :8088 + Next.js :3500
make api / make ui # individually (override API_PORT / WEB_PORT if in use)
make run           # CLI agent (no frontend)
make test / make eval
make docker-build && make docker-up
```
Remote dev: browser can't reach :8088/:3500 → almost always a missing SSH `-L` forward, not CORS.

## Eval
Local custom LLM-as-judge harness (`eval/run_eval.py`, results → `eval/orion-v9.json`). **120 labeled cases / 6 categories; 116 scored** (4 escalation cases excluded from quantitative metrics). Dataset generated then manually reviewed against the Olist data + ShopNova policies.
- Metrics: **Correctness** (LLM-judge vs expected, all), **Tool selection** (exact match, all), **Faithfulness** (claim-level, `rag_only` only — applying it to `both_tools` is structurally invalid since SQL facts aren't in RAG context), **Answer relevancy** (RAG categories).
- **orion-v9, 116 examples:** Correctness **0.87** (116), Tool selection **0.93** (111), Faithfulness **0.97** (44), Answer relevancy **0.94** (75).
- Custom judge over RAGAS because RAGAS faithfulness wrongly penalizes `both_tools` answers and LangSmith's free trace quota runs out mid-eval. `make eval`; `--limit 5` smoke test (Groq free-tier rate limits under full concurrent load).

## Scope discipline
- No numeric fact cross-checking — prices/dates in responses aren't verified against raw tool output (a production verification step is the documented next step).
- In-memory thread state (no persistence), demo-scoped Gmail OAuth (`token.json`), single-tenant eval dataset.
- Voice accuracy on accented/noisy audio is unmeasured.
