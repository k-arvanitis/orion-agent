"""Unit tests for the escalate_to_human tool."""

import json
from unittest.mock import patch

from orion_agent.agent.tools import escalate_tool


def test_escalate_returns_structured_handoff_payload():
    with patch.object(escalate_tool, "SLACK_WEBHOOK_URL", ""):
        raw = escalate_tool.escalate_to_human.invoke(
            {
                "subject": "Refund request for order abc123",
                "action_needed": "Approve the refund",
                "reason": "Customer requested a refund for a damaged item.",
            }
        )

    data = json.loads(raw)
    assert data["escalate"] is True
    assert data["subject"] == "Refund request for order abc123"
    assert data["action_needed"] == "Approve the refund"
    assert "support team" in data["answer"]


def test_escalate_posts_to_slack_when_webhook_configured():
    with (
        patch.object(escalate_tool, "SLACK_WEBHOOK_URL", "https://hooks.slack.test/x"),
        patch.object(escalate_tool.requests, "post") as post,
    ):
        escalate_tool.escalate_to_human.invoke(
            {"subject": "s", "action_needed": "a", "reason": "r"}
        )

    post.assert_called_once()
    assert post.call_args.args[0] == "https://hooks.slack.test/x"
    assert "s" in post.call_args.kwargs["json"]["text"]


def test_escalate_skips_slack_when_webhook_not_configured():
    with (
        patch.object(escalate_tool, "SLACK_WEBHOOK_URL", ""),
        patch.object(escalate_tool.requests, "post") as post,
    ):
        escalate_tool.escalate_to_human.invoke(
            {"subject": "s", "action_needed": "a", "reason": "r"}
        )

    post.assert_not_called()


def test_escalate_swallows_slack_errors():
    with (
        patch.object(escalate_tool, "SLACK_WEBHOOK_URL", "https://hooks.slack.test/x"),
        patch.object(escalate_tool.requests, "post", side_effect=Exception("boom")),
    ):
        raw = escalate_tool.escalate_to_human.invoke(
            {"subject": "s", "action_needed": "a", "reason": "r"}
        )

    assert json.loads(raw)["escalate"] is True
