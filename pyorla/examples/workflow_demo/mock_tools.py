"""Mock tool implementations for the workflow demo."""

from __future__ import annotations

import logging
from typing import Literal

from pyorla.tool_decorators import orla_tool

log = logging.getLogger(__name__)

_POLICIES: dict[str, str] = {
    "billing": """\
policy:
  billing:
    duplicate_charges:
      action: refund
      conditions:
        - verified_duplicate: true
        - within_days: 30
      sla: "Refund processed within 3 business days"
    subscription_cancellation:
      action: cancel_and_prorate
      conditions:
        - active_subscription: true
      sla: "Effective end of current billing cycle"\
""",
    "technical": """\
policy:
  technical:
    service_degradation:
      action: investigate_and_credit
      conditions:
        - confirmed_outage: true
        - duration_minutes: ">15"
      sla: "Resolution within 4 hours, credit if SLA missed"
    bug_report:
      action: escalate_to_engineering
      conditions:
        - reproducible: true
      sla: "Acknowledgment within 24 hours"\
""",
    "account": """\
policy:
  account:
    access_issues:
      action: reset_and_verify
      sla: "Resolved within 1 hour"
    data_request:
      action: export_data
      conditions:
        - identity_verified: true
      sla: "Data export within 48 hours"\
""",
    "shipping": """\
policy:
  shipping:
    lost_package:
      action: reship_or_refund
      conditions:
        - tracking_shows_lost: true
      sla: "Replacement shipped within 2 business days"\
""",
}


@orla_tool
def read_policy_yaml(category: str) -> dict[str, str]:
    """Read the company support policy document for a given category.

    Returns the policy rules as structured text.
    """
    policy = _POLICIES.get(category, f"No specific policy found for category: {category}.")
    return {"policy_document": policy}


@orla_tool
def send_email(to: str, subject: str, body: str) -> dict[str, str]:
    """Send an email to a recipient with the given subject and body."""
    log.info("[send_email] To: %s | Subject: %s", to, subject)
    return {"status": "sent", "message_id": f"msg-{to}-001"}


@orla_tool
def read_team_descriptions() -> dict[str, list[dict[str, str]]]:
    """Read descriptions of internal support teams to determine the best routing destination."""
    return {
        "teams": [
            {
                "name": "billing_ops",
                "description": "Handles refunds, subscription changes, payment disputes, and invoice corrections.",
                "email": "billing-ops@company.com",
            },
            {
                "name": "technical_support",
                "description": "Handles service outages, performance issues, bug reports, and API problems.",
                "email": "tech-support@company.com",
            },
            {
                "name": "account_management",
                "description": "Handles account access, data requests, plan upgrades, and enterprise onboarding.",
                "email": "account-mgmt@company.com",
            },
            {
                "name": "escalation_team",
                "description": "Handles critical/VIP issues, multi-department problems, and unresolved complaints.",
                "email": "escalation@company.com",
            },
        ]
    }


@orla_tool
def send_ticket(
    team: str,
    priority: Literal["critical", "high", "medium", "low"],
    summary: str,
    customer_email: str | None = None,
) -> dict[str, str]:
    """Create and send an internal support ticket to the designated team."""
    log.info("[send_ticket] Team: %s | Priority: %s", team, priority)
    _ = customer_email
    return {"ticket_id": f"TKT-{team}-42", "status": "created"}
