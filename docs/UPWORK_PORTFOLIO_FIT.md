# Orion portfolio fit for AI engineering work

The local Upwork market workbook dated August 2, 2026 groups 76 relevant
listings. Orion is recommended portfolio evidence for 49 of them. The project is
especially relevant to workflow automation, conversational/voice agents,
customer-facing assistants, structured-data workflows, and human-in-the-loop
systems.

## Recurring buyer requirements mapped to proof

| Requirement in the workbook | Frequency | Inspectable Orion proof |
|---|---:|---|
| API and deployment | 40 listings | FastAPI, OpenAPI docs, Docker Compose, uncommon configurable ports, one-command demo startup |
| RAG | 33 | Hybrid dense + sparse Qdrant policy retrieval with stored source chunks |
| Evaluation and QA | 32 | Automated unit/API tests plus the labeled agent evaluation harness |
| Text-to-SQL / structured data | 32 | Normalized support CRM, customer identity lookup, guarded SELECT-only SQL tool |
| Agents / LangGraph | 29 | Stateful ReAct graph, isolated structured tool outputs, per-session trace state |
| Private/internal knowledge | 18 | Policy corpus, database boundary, local embeddings, environment-driven services |
| Anti-hallucination / refusal | 18 | Response guard, retrieval grounding, deterministic no-key demo path, documented failure modes |
| Vector database / embeddings | 18 | Qdrant hybrid retrieval, local BGE embeddings, BM25 sparse embeddings, RRF fusion |
| Source attribution | 13 | Retrieved source/heading/content shown under Technical Details |
| Security / compliance | 11 | PII stripping, SELECT-only validation, secrets via environment, external-service isolation |
| Human review | 7 | Clear “Waiting for support” outcome, context-rich handoff, operator reply persisted to the customer thread |

## What a prospective client can verify quickly

1. `make demo` produces a usable product without hidden setup or provider spend.
2. A natural customer message identifies the correct database record without a
   customer dropdown.
3. Routine and approval-heavy conversations produce visibly different outcomes.
4. Both customer and operator views remain synchronized through the API and SQL
   store.
5. OpenAPI documentation, tests, architecture notes, and Technical Details make
   the implementation inspectable beyond screenshots.

## Positioning

Use Orion as proof for projects asking for an AI support agent, customer portal,
workflow automation, structured-data assistant, human escalation, voice/chat
interface, FastAPI/Next.js build, or production-oriented LangGraph system.

For pure PDF/OCR knowledge-base work, pair Orion with a document-focused project;
Orion’s strongest story is mixed structured and unstructured support automation,
not document ingestion by itself.
