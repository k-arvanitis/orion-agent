# Orion portfolio demo guide

This walkthrough demonstrates the complete customer-support loop in about 90
seconds. Every reply comes from the live LangGraph agent, so you need
`OPENROUTER_API_KEY` (or `GROQ_API_KEY`) in `.env` and a populated policy index.

## Start once

```bash
make ingest   # first time only: start local Qdrant + embed the policy docs
make demo
```

`make demo` starts the local Qdrant container, seeds the support CRM, starts
FastAPI on `http://localhost:8088`, and starts Next.js on `http://localhost:3500`.
The uncommon ports avoid the usual 3000/8000 development conflicts.

## The two pages

| URL | Viewpoint | What it proves |
|---|---|---|
| `http://localhost:3500/customer` | ShopNova customer | Identity comes from the conversation; customer and order records are retrieved from SQL. |
| `http://localhost:3500/support` | Support operator | Resolved and waiting conversations, customer CRM data, orders, handoff context, replies, and expandable technical details. |

FastAPI also publishes interactive endpoint documentation at
`http://localhost:8088/docs`.

## Flow 1: automatic resolution

1. Open `/customer` and `/support` in separate tabs.
2. In the customer page send: `Where is order 416e49799e9260d93c8f636ce6661a55?`
3. Orion identifies Maya Torres from the order ID and loads the matched product,
   parcel, and delivery status.
4. The conversation is marked **Resolved by Orion** because no human decision is
   required.
5. The database-persisted conversation appears automatically in `/support`.

## Flow 1b: policy retrieval

1. Click **New conversation** on `/customer`.
2. Send: `What is your return policy?`
3. The agent calls `search_policies` (hybrid dense + BM25 search over the
   Qdrant index) and answers without asking for personal information.
4. Expand **Technical details** to see the policy search entry and the
   retrieved document sections.

## Flow 2: human approval and reply

1. Click **New conversation** on `/customer`.
2. Send: `My email is maya.torres@example.com. I need a refund.`
3. Orion identifies the same customer. The agent decides a refund needs human
   approval and calls `escalate_to_human`, so the conversation moves to
   **Waiting for support**. If `SLACK_WEBHOOK_URL` is set, a Slack alert fires.
4. Open the new conversation in `/support`. The middle pane contains the full
   conversation and a concise handoff summary; the right pane contains the CRM
   and order record.
5. Reply from the support workspace. The reply is stored in SQL and appears on
   the customer page during its next sync.

The customer and support views expose the same database-persisted conversation.
Their technical-details sections show which lookup tools ran and which SQL
records were retrieved.

## What is real, and what is seeded

- Real: the LangGraph agent and its three tools, SQL tables and persistence,
  FastAPI request boundary, two synchronized interfaces, status routing,
  support replies, reset behavior, and automated tests.
- Seeded: the six demo customers, their orders, and the example support cases
  are fictional records created by `make seed-support`. Identity matching
  (email, customer ID, order ID, parcel ID) is a deterministic SQL lookup that
  runs before the agent so Orion never guesses who it is talking to.
- Every response, tool choice, and escalation decision comes from the agent.
  There is no scripted fallback: without an LLM key the demo does not answer.

## Reset only the generated demo conversations

```bash
curl -X POST http://localhost:8088/api/support/demo/reset
```

Removes only conversations created from the customer demo. The seeded CRM,
products, orders, and example support cases remain intact.
