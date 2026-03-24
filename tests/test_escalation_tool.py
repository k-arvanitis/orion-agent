"""
Unit tests for the escalation tool.

Gmail, Slack, and SQLAlchemy are fully mocked — no external services required.
Tests cover:
  - Missing or invalid email returns a prompt to provide one
  - Successful escalation calls Slack and Gmail
  - Gmail failure does not crash the tool (Slack still fires)
  - Slack failure does not crash the tool (Gmail still fires)
  - Order fetch failure is handled gracefully (escalation still proceeds)
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("QDRANT_API_KEY", "test-dummy-key")
os.environ.setdefault("DATABASE_URL", "postgresql://localhost/test")
os.environ.setdefault("SLACK_WEBHOOK_URL", "https://hooks.slack.com/test")
os.environ.setdefault("OPERATOR_EMAIL", "test-operator@example.com")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.tools.escalation_tool import escalate


# ---------------------------------------------------------------------------
# Email validation
# ---------------------------------------------------------------------------


def test_escalate_missing_email_returns_prompt():
    result = escalate.invoke({"customer_email": "", "issue_summary": "test", "order_id": ""})
    assert "email" in result.lower()


def test_escalate_invalid_email_returns_prompt():
    result = escalate.invoke({"customer_email": "notanemail", "issue_summary": "test", "order_id": ""})
    assert "email" in result.lower()


# ---------------------------------------------------------------------------
# Successful escalation
# ---------------------------------------------------------------------------


@patch("agent.tools.escalation_tool._get_gmail")
@patch("agent.tools.escalation_tool.requests.post")
def test_escalate_calls_slack_and_gmail(mock_post, mock_gmail):
    mock_post.return_value = MagicMock(status_code=200)
    mock_service = MagicMock()
    mock_gmail.return_value = mock_service

    result = escalate.invoke({
        "customer_email": "customer@example.com",
        "issue_summary": "Package not received after 30 days.",
        "order_id": "",
    })

    assert "customer@example.com" in result
    assert "24 hours" in result
    mock_post.assert_called_once()
    mock_service.users().messages().send().execute.assert_called_once()


@patch("agent.tools.escalation_tool._get_gmail")
@patch("agent.tools.escalation_tool.requests.post")
def test_escalation_confirmation_message_includes_email(mock_post, mock_gmail):
    mock_post.return_value = MagicMock(status_code=200)
    mock_gmail.return_value = MagicMock()

    result = escalate.invoke({
        "customer_email": "jane@example.com",
        "issue_summary": "Refund not processed.",
        "order_id": "",
    })

    assert "jane@example.com" in result


# ---------------------------------------------------------------------------
# Partial failures — tool must not crash
# ---------------------------------------------------------------------------


@patch("agent.tools.escalation_tool._get_gmail", side_effect=Exception("Gmail auth failed"))
@patch("agent.tools.escalation_tool.requests.post")
def test_gmail_failure_does_not_crash_escalation(mock_post, mock_gmail):
    mock_post.return_value = MagicMock(status_code=200)

    # Should not raise
    result = escalate.invoke({
        "customer_email": "user@example.com",
        "issue_summary": "Wrong item received.",
        "order_id": "",
    })

    assert isinstance(result, str)
    mock_post.assert_called_once()  # Slack still fires


@patch("agent.tools.escalation_tool._get_gmail")
@patch("agent.tools.escalation_tool.requests.post", side_effect=Exception("Slack webhook down"))
def test_slack_failure_does_not_crash_escalation(mock_post, mock_gmail):
    mock_gmail.return_value = MagicMock()

    result = escalate.invoke({
        "customer_email": "user@example.com",
        "issue_summary": "Damaged product.",
        "order_id": "",
    })

    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Order fetch failure
# ---------------------------------------------------------------------------


@patch("agent.tools.escalation_tool._fetch_order", side_effect=Exception("DB timeout"))
@patch("agent.tools.escalation_tool._get_gmail")
@patch("agent.tools.escalation_tool.requests.post")
def test_order_fetch_failure_still_escalates(mock_post, mock_gmail, mock_fetch):
    mock_post.return_value = MagicMock(status_code=200)
    mock_gmail.return_value = MagicMock()

    result = escalate.invoke({
        "customer_email": "user@example.com",
        "issue_summary": "Cannot track my order.",
        "order_id": "some-order-id",
    })

    assert isinstance(result, str)
    assert "user@example.com" in result
