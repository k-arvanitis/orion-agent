"""
Escalation tool — hands off unresolved issues to a human operator.

When the agent cannot resolve a customer issue, this tool:
  1. Fetches order details from Supabase if an order ID is available.
  2. Sends a confirmation email to the customer via Gmail.
  3. Posts an urgent alert to the operator Slack channel.

On every successful agent response, the graph calls notify_operator() directly
to post a brief resolution ping to Slack for monitoring.
"""

import base64
import logging
import os
from datetime import datetime
from email.mime.text import MIMEText

import requests

logger = logging.getLogger(__name__)
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_core.tools import tool
from sqlalchemy import create_engine, text

_engine = None
_gmail = None

OPERATOR_EMAIL = "support-team@shopnova.com.br"
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.send"]


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(os.environ["DATABASE_URL"])
    return _engine


def _get_gmail():
    global _gmail
    if _gmail is None:
        creds = Credentials.from_authorized_user_file("token.json", GMAIL_SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        _gmail = build("gmail", "v1", credentials=creds)
    return _gmail


# ---------------------------------------------------------------------------
# Order fetch
# ---------------------------------------------------------------------------


def _fetch_order(order_id: str) -> dict | None:
    query = text("""
        SELECT
            o.order_id,
            o.customer_id,
            o.order_status,
            o.order_purchase_timestamp,
            o.order_estimated_delivery_date,
            o.order_delivered_customer_date,
            SUM(oi.price + oi.freight_value) AS total_value,
            op.payment_type
        FROM orders o
        LEFT JOIN order_items oi ON o.order_id = oi.order_id
        LEFT JOIN order_payments op ON o.order_id = op.order_id
        WHERE o.order_id = :order_id
        GROUP BY o.order_id, o.customer_id, op.payment_type
        LIMIT 1
    """)
    with _get_engine().connect() as conn:
        row = conn.execute(query, {"order_id": order_id}).fetchone()
    if not row:
        return None
    keys = ["order_id", "customer_id", "order_status", "purchase_date",
            "estimated_delivery", "actual_delivery", "total_value", "payment_type"]
    return dict(zip(keys, row))


# ---------------------------------------------------------------------------
# Gmail
# ---------------------------------------------------------------------------


def _send_gmail(to: str, subject: str, body: str) -> None:
    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    _get_gmail().users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()


def _build_customer_email(
    customer_email: str,
    issue_summary: str,
    order: dict | None,
) -> tuple[str, str]:
    order_ref = f"#{order['order_id'][:8]}..." if order else ""
    subject = f"[ShopNova Support] {order_ref} — We've received your request"

    order_block = ""
    if order:
        total = f"R$ {order['total_value']:.2f}" if order["total_value"] else "N/A"
        order_block = f"""
Order details
-------------
Order ID:    {order["order_id"]}
Status:      {order["order_status"]}
Total:       {total}
"""

    body = f"""Hi,

We've received your support request and a member of our team will get back to you within 24 hours.

{order_block}
Thank you for your patience.

ShopNova Support Team
---
This message was sent automatically by Orion, ShopNova's support agent.
"""
    return subject, body


# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------


def _slack(text: str, urgent: bool = False) -> None:
    webhook = os.environ.get("SLACK_WEBHOOK_URL")
    if not webhook:
        logger.warning("SLACK_WEBHOOK_URL not set — Slack notification skipped")
        return
    prefix = ":rotating_light: *ESCALATION*" if urgent else ":white_check_mark: *Resolved*"
    try:
        requests.post(webhook, json={"text": f"{prefix}\n{text}"}, timeout=5)
    except Exception:
        logger.error("Failed to post Slack notification", exc_info=True)


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@tool
def escalate(
    customer_email: str,
    issue_summary: str,
    order_id: str,
) -> str:
    """
    Escalate an unresolved customer issue to a human support operator.

    Call this tool when:
      - You cannot resolve the issue with the available tools.
      - The customer explicitly asks to speak to a human.
      - The customer expresses high frustration or distress.

    Args:
        customer_email: The customer's email address (ask if not provided).
        issue_summary:  One or two sentences describing the unresolved issue.
        order_id:       The order ID if known, empty string otherwise.

    Returns:
        Confirmation that the escalation was sent.
    """
    if not customer_email or "@" not in customer_email:
        return "Ask the customer for their email address before escalating."

    order = None
    if order_id:
        try:
            order = _fetch_order(order_id)
        except Exception:
            logger.warning("Failed to fetch order details for %s", order_id, exc_info=True)

    # Gmail → customer confirmation
    subject, body = _build_customer_email(customer_email, issue_summary, order)
    try:
        _send_gmail(customer_email, subject, body)
    except Exception:
        logger.error("Failed to send Gmail confirmation to %s", customer_email, exc_info=True)

    # Slack → operator urgent alert
    order_line = f"Order: `{order_id}`\n" if order_id else ""
    customer_id_line = f"Customer ID: `{order['customer_id']}`\n" if order else ""
    slack_text = (
        f"Email: {customer_email}\n"
        f"{customer_id_line}"
        f"{order_line}"
        f"Issue: {issue_summary}"
    )
    _slack(slack_text, urgent=True)

    return (
        f"I've sent a confirmation to {customer_email} and alerted our support team. "
        "A human agent will get back to you within 24 hours."
    )
