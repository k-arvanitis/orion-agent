"""
Escalation tool — hands a conversation off to a human support teammate.

The escalation signal itself has no external dependency to fail (it's a
structured payload the agent emits); the only external call is a best-effort
Slack notification so a teammate doesn't have to poll the queue to find out.

Return format:
  JSON string: {"answer": "<text for LLM>", "escalate": true,
                 "subject": "<ticket subject>", "action_needed": "<...>",
                 "reason": "<internal note>"}
  The graph's tools_node puts "answer" into ToolMessage.content (what the LLM
  sees) and stores "escalate"/"subject"/"action_needed"/"reason" in graph
  state (what the support dashboard shows).
"""

import json
import logging

import requests
from langchain_core.tools import tool

from orion_agent.agent.config import SLACK_WEBHOOK_URL

logger = logging.getLogger(__name__)


def _notify_slack(subject: str, action_needed: str, reason: str) -> None:
    if not SLACK_WEBHOOK_URL:
        logger.info("SLACK_WEBHOOK_URL not set — Slack notification skipped")
        return
    text = (
        f":rotating_light: *Escalation* — {subject}\n{reason}\n"
        f"*Action needed:* {action_needed}"
    )
    try:
        requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
    except Exception:
        logger.error("Failed to post Slack notification", exc_info=True)


@tool
def escalate_to_human(subject: str, action_needed: str, reason: str) -> str:
    """
    Hand this conversation off to a human support teammate. Call this when:
    - the customer needs a refund or cancellation approved
    - an item arrived damaged, wrong, or missing and needs review before a
      replacement is approved
    - the customer explicitly asks to speak with a person, or files a
      complaint

    Do not call this for questions already answerable from search_policies
    or query_database results.

    Args:
        subject: Short ticket subject line, e.g. "Refund request for order abc123".
        action_needed: What the teammate must do, e.g. "Approve the refund".
        reason: One-sentence internal note for the teammate explaining what
            happened and what's needed.

    Returns:
        JSON string confirming the handoff for the LLM to relay to the customer.
    """
    _notify_slack(subject, action_needed, reason)
    return json.dumps(
        {
            "answer": (
                "I've passed this conversation to our support team — a "
                "teammate will take it from here."
            ),
            "escalate": True,
            "subject": subject,
            "action_needed": action_needed,
            "reason": reason,
        }
    )
