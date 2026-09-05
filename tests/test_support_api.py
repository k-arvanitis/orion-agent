"""Tests for the persistent customer/support demo API.

The demo path (`/api/support/*`) now routes every customer message through
the real LangGraph agent (`orion_agent.agent.graph.run_turn`). Per CLAUDE.md,
all external services are mocked in tests — `run_turn` is faked here with a
small keyword router that mirrors what the real agent's tool-routing prompt
asks it to do, so these tests exercise support_store.py's translation of a
graph result into a conversation record, not the LLM itself.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _fake_run_turn(thread_id: str, message: str) -> dict:
    normalized = message.lower()

    if any(word in normalized for word in ("cancel", "refund")):
        return {
            "response": "I've passed this conversation to our support team.",
            "tools_called": [],
            "last_chunks": [],
            "last_sql": "",
            "escalation": {
                "subject": "Refund or cancellation request",
                "action_needed": "Approve the refund or cancellation request",
                "reason": "Customer requested a refund or cancellation.",
            },
        }

    if "return policy" in normalized or "policy" in normalized:
        return {
            "response": (
                "Most items can be returned within 30 calendar days of "
                "confirmed delivery if unused and in original packaging."
            ),
            "tools_called": ["search_policies"],
            "last_chunks": [
                {
                    "source": "return_policy.md",
                    "heading": "Returns",
                    "content": "30-day return window.",
                }
            ],
            "last_sql": "",
            "escalation": None,
        }

    return {
        "response": (
            "Your order is currently delayed: Expected Jul 30 · Standard "
            "shipping. Is there anything else I can help you with regarding "
            "your order?"
        ),
        "tools_called": ["query_database"],
        "last_chunks": [],
        "last_sql": "SELECT order_status FROM orders WHERE order_id = :id",
        "escalation": None,
    }


@pytest.fixture()
def support_client(tmp_path, monkeypatch):
    from api.main import app
    from api.support_store import configure_database

    monkeypatch.setattr("orion_agent.agent.graph.run_turn", _fake_run_turn)
    configure_database(f"sqlite:///{tmp_path / 'support.db'}")
    with TestClient(app) as client:
        yield client


def test_seeded_crm_data_is_returned(support_client):
    customers = support_client.get("/api/support/customers")
    products = support_client.get("/api/support/products")
    conversations = support_client.get("/api/support/conversations")

    assert customers.status_code == 200
    assert products.status_code == 200
    assert conversations.status_code == 200
    assert len(customers.json()) == 6
    assert len(products.json()) == 7
    assert len(conversations.json()) == 6
    assert customers.json()[0]["orders"][0]["item"]


def test_demo_overview_is_derived_from_seeded_database(support_client):
    response = support_client.get("/api/support/demo/overview")

    assert response.status_code == 200
    overview = response.json()
    assert overview["database"] == "SQLite"
    assert overview["counts"] == {
        "customers": 6,
        "products": 7,
        "orders": 7,
        "conversations": 6,
        "waiting": 3,
        "resolved": 3,
    }
    assert overview["sample"]["orderId"] in overview["examples"][0]["message"]
    assert overview["sample"]["email"] in overview["examples"][1]["message"]
    assert overview["sample"]["parcelId"] in overview["examples"][2]["message"]


def test_conversation_identifies_customer_and_persists_reply(support_client):
    unmatched = support_client.post(
        "/api/support/conversations/messages",
        json={"message": "Where is my order?", "conversation_id": None},
    )
    assert unmatched.status_code == 422

    matched = support_client.post(
        "/api/support/conversations/messages",
        json={
            "message": "Where is order 416e49799e9260d93c8f636ce6661a55?",
            "conversation_id": None,
        },
    )
    assert matched.status_code == 200
    ticket = matched.json()
    assert ticket["customer"]["name"] == "Maya Torres"
    assert ticket["status"] == "Resolved by Orion"
    assert ticket["subject"] == "Order status"
    order_reply = ticket["messages"][-1]["content"]
    assert "delayed" in order_reply
    assert ticket["technicalDetails"]["tools"][-1]["name"] == "order_database_query"

    escalated = support_client.post(
        "/api/support/conversations/messages",
        json={"message": "I need a refund.", "conversation_id": ticket["id"]},
    )
    assert escalated.status_code == 200
    escalated_ticket = escalated.json()
    assert escalated_ticket["status"] == "Waiting for support"
    assert escalated_ticket["subject"] == "Refund or cancellation request"
    assert (
        escalated_ticket["actionNeeded"]
        == "Approve the refund or cancellation request"
    )

    replied = support_client.post(
        f"/api/support/conversations/{ticket['id']}/reply",
        json={"message": "Your refund was approved."},
    )
    assert replied.status_code == 200
    assert replied.json()["messages"][-1]["content"] == "Your refund was approved."

    listed = support_client.get("/api/support/conversations").json()
    stored = next(
        conversation
        for conversation in listed
        if conversation["id"] == ticket["id"]
    )
    assert stored["messages"][-1]["content"] == "Your refund was approved."

    finished = support_client.post(
        f"/api/support/conversations/{ticket['id']}/finish"
    )
    assert finished.status_code == 200
    assert finished.json()["status"] == "Resolved by support"
    assert finished.json()["actionNeeded"] is None

    persisted = support_client.get("/api/support/conversations").json()
    finished_ticket = next(
        conversation
        for conversation in persisted
        if conversation["id"] == ticket["id"]
    )
    assert finished_ticket["status"] == "Resolved by support"


def test_policy_question_uses_vector_store_without_customer_identity(support_client):
    response = support_client.post(
        "/api/support/conversations/messages",
        json={"message": "What is your return policy?", "conversation_id": None},
    )

    assert response.status_code == 200
    ticket = response.json()
    assert ticket["customer"]["id"] == "CUS-VISITOR"
    assert ticket["status"] == "Resolved by Orion"
    assert ticket["technicalDetails"]["tools"] == [
        {
            "name": "policy_vector_search",
            "label": "Policy vector search",
            "result": "1 chunks",
        }
    ]
    assert ticket["technicalDetails"]["records"] == []
    assert ticket["technicalDetails"]["documents"][0]["source"] == "return_policy.md"
    assert "30 calendar days" in ticket["messages"][-1]["content"]


def test_policy_visitor_order_request_asks_for_identifier(support_client):
    policy = support_client.post(
        "/api/support/conversations/messages",
        json={"message": "What is your return policy?", "conversation_id": None},
    ).json()

    response = support_client.post(
        "/api/support/conversations/messages",
        json={"message": "I want a refund", "conversation_id": policy["id"]},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Please share your email, customer ID, order number, or parcel number "
        "so I can find the right order."
    )


def test_policy_visitor_can_be_identified_in_later_message(support_client):
    policy = support_client.post(
        "/api/support/conversations/messages",
        json={"message": "What is your return policy?", "conversation_id": None},
    ).json()

    response = support_client.post(
        "/api/support/conversations/messages",
        json={
            "message": "Where is order 416e49799e9260d93c8f636ce6661a55?",
            "conversation_id": policy["id"],
        },
    )

    assert response.status_code == 200
    assert response.json()["customer"]["name"] == "Maya Torres"
    assert response.json()["subject"] == "Order status"


def test_opening_conversation_clears_unread_count(support_client):
    tickets = support_client.get("/api/support/conversations").json()
    ticket = next(item for item in tickets if item["unread"] > 0)

    response = support_client.post(
        f"/api/support/conversations/{ticket['id']}/read"
    )

    assert response.status_code == 200
    assert response.json()["unread"] == 0
    stored = support_client.get("/api/support/conversations").json()
    assert next(item for item in stored if item["id"] == ticket["id"])["unread"] == 0


def test_support_can_delete_any_conversation(support_client):
    tickets = support_client.get("/api/support/conversations").json()
    ticket = next(item for item in tickets if item["source"] == "seed")

    response = support_client.delete(
        f"/api/support/conversations/{ticket['id']}"
    )

    assert response.status_code == 200
    assert response.json() == {"deleted": ticket["id"]}
    remaining = support_client.get("/api/support/conversations").json()
    assert all(item["id"] != ticket["id"] for item in remaining)
    assert support_client.delete(
        f"/api/support/conversations/{ticket['id']}"
    ).status_code == 404


def test_customer_matched_by_email_not_only_order_id(support_client):
    response = support_client.post(
        "/api/support/conversations/messages",
        json={
            "message": "My email is liam.chen@example.com, can I exchange my jacket?",
            "conversation_id": None,
        },
    )

    assert response.status_code == 200
    assert response.json()["customer"]["name"] == "Liam Chen"


def test_identity_match_is_indexed_not_a_full_table_scan(support_client, monkeypatch):
    """The lookup must use SQLAlchemy select(...).where(...in_(...)) — not
    load every customer/order row into Python and substring-scan them,
    which is what made the old implementation O(customers + orders)."""
    from api import support_store

    original_execute = support_store.Connection.execute
    calls = []

    def spying_execute(self, statement, *args, **kwargs):
        calls.append(str(statement))
        return original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(support_store.Connection, "execute", spying_execute)

    support_client.post(
        "/api/support/conversations/messages",
        json={
            "message": "Where is order 416e49799e9260d93c8f636ce6661a55?",
            "conversation_id": None,
        },
    )

    full_scans = [
        c for c in calls
        if ("FROM support_customers" in c or "FROM support_orders" in c)
        and "WHERE" not in c
        and "count" not in c.lower()
    ]
    assert not full_scans, f"identity match ran an unfiltered table scan: {full_scans}"
