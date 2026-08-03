"""
Escalation tool — hands a conversation off to a human support teammate.

Unlike search_policies/query_database, this tool has no external dependency
to fail; it's a structured signal the agent emits when it decides a request
needs human judgment or approval it cannot grant on its own.

Return format:
  JSON string: {"answer": "<text for LLM>", "escalate": true,
                 "subject": "<ticket subject>", "action_needed": "<...>",
                 "reason": "<internal note>"}
  The graph's tools_node puts "answer" into ToolMessage.content (what the LLM
  sees) and stores "escalate"/"subject"/"action_needed"/"reason" in graph
  state (what the support dashboard shows).
"""

import json

from langchain_core.tools import tool


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
