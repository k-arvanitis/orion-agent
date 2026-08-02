# Orion portfolio demo guide

This walkthrough demonstrates the complete customer-support loop in about 90
seconds. It is intentionally simple enough for a client to run locally while
still exposing the engineering behind it.

## Start once

```bash
make demo
```

This seeds the support CRM, starts FastAPI on `http://localhost:8088`, and starts
Next.js on `http://localhost:3500`. The uncommon ports avoid the usual 3000/8000
development conflicts.

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
3. Orion searches the bundled policy chunks in a local Qdrant vector index and
   answers without asking for personal information.
4. Expand **Technical details** to see `policy_vector_search`, the retrieved
   document sections, and their similarity scores.

## Flow 2: human approval and reply

1. Click **New conversation** on `/customer`.
2. Send: `My email is maya.torres@example.com. I need a refund.`
3. Orion identifies the same customer, but marks the conversation **Waiting for
   support** because a refund requires approval.
4. Open the new conversation in `/support`. The middle pane contains the full
   conversation and a concise handoff summary; the right pane contains the CRM
   and order record.
5. Reply from the support workspace. The reply is stored in SQL and appears on
   the customer page during its next sync.

The customer and support views expose the same database-persisted conversation.
Their technical-details sections show which lookup tools ran and which SQL
records were retrieved.

## What is real, and what is simulated

- Real: SQL tables and persistence, customer/order/parcel matching, FastAPI
  request boundary, two synchronized interfaces, status routing, support replies,
  reset behavior, and automated tests.
- Reliable local simulation: the response policy used by `/customer` is
  deterministic, which keeps the portfolio demo usable without paid keys or
  third-party uptime.
- Provider-backed agent runtime: `/api/chat` uses LangGraph with policy RAG,
  Text-to-SQL, response guards, tracing, and escalation integrations when the
  documented environment variables are configured.

This separation is deliberate: the demonstration always works, while the
repository still exposes the production-oriented AI implementation for review.
