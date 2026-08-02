# Orion realistic demo workflow

This is the recommended portfolio walkthrough. The customer and support views
use the same FastAPI service and the same local SQLite demo database. Supabase
is not required for these scenarios.

## Start the demo

```bash
make demo
```

Open these side by side:

- Customer: `http://localhost:3500/customer`
- Support team: `http://localhost:3500/support`

`New conversation` starts a separate customer chat without deleting earlier
demo conversations, so the support queue builds up naturally during the demo.

## Main 90-second walkthrough

### 1. General policy question: vector retrieval

Click **New conversation**, then ask:

> What is your return policy?

Expected result:

- Orion answers without requesting personal information.
- The case is **Resolved by Orion**.
- In **Technical details**, the tool is `policy_vector_search`.
- The retrieved policy sections and similarity scores come from the local
  Qdrant vector index.
- The conversation appears under **Resolved** in the support view.

### 2. Identified customer: SQL order lookup

Click **New conversation**, then ask:

> Where is order 416e49799e9260d93c8f636ce6661a55?

Expected result:

- Orion identifies Maya Torres from the order number.
- Her customer and delayed order records are loaded from SQLite.
- Orion answers the delivery-status question and marks it **Resolved by Orion**.
- **Technical details** shows `customer_lookup`, `order_lookup`,
  `support_customers`, and `support_orders`.
- The new conversation appears live in the support view.

### 3. Automatic exchange: a second routine case resolved without a human

Click **New conversation**, then ask:

> My email is liam.chen@example.com. Can I exchange the jacket for a larger size?

Expected result:

- Liam is identified from his email and his order is looked up via SQL.
- The exchange is within policy, so Orion resolves it without escalating.
- Reinforces that identification + resolution is not a one-off — it works for
  a different customer, a different policy, and a different tool combination.

### 4. Human escalation: the case Orion won't resolve alone

Click **New conversation**, then send:

> My email is maya.torres@example.com. I need a refund because the order will arrive after my trip.

Expected result:

- Orion identifies Maya and her latest order.
- A refund requires human approval, so the case becomes **Waiting for support**.
- The conversation appears under **Needs support** in roughly 1–2 seconds.
- The support view shows the full conversation, Orion's handoff summary,
  customer record, matched orders, and SQL technical details.

From the support view, reply as Alex Kim:

> Hi Maya, I reviewed the delay and approved a refund for the shipping charge. You will receive a confirmation email shortly.

Expected result:

- The reply is persisted to the local SQL database.
- It appears automatically in the customer chat in roughly 1–2 seconds.
- Closing beat: Orion knows its own limits, and a human closes the loop in
  the same thread — the customer never leaves the conversation.

## What to explain while presenting

- Customer identity is inferred from the conversation; there is no customer
  picker.
- Customer, product, order, conversation, and message records are persisted in
  a real relational database.
- Routine factual questions are resolved automatically.
- Approval-heavy requests are handed to one simulated teammate, Alex Kim.
- Policy questions use vector retrieval; customer and order questions use SQL.
- Both interfaces poll the same API, which is why updates appear live.

## Reset only the generated demo conversations

Run this between presentations if you want a clean support queue:

```bash
curl -X POST http://localhost:8088/api/support/demo/reset
```

This removes only conversations created from the customer demo. The seeded CRM,
products, orders, and example support cases remain intact.
