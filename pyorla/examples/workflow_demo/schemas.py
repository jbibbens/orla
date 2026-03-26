"""Structured-output schemas for the workflow demo."""

CLASSIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["billing", "technical", "account", "shipping", "general"],
        },
        "product": {
            "type": "string",
            "description": "Product or service mentioned in the ticket",
        },
        "key_issue": {
            "type": "string",
            "description": "One-sentence summary of the core issue",
        },
        "customer_request": {
            "type": "string",
            "description": "What the customer is actually asking for",
        },
        "needs_escalation": {
            "type": "boolean",
            "description": (
                "Whether this ticket requires human team intervention (true) "
                "or can be fully resolved automatically (false)"
            ),
        },
        "escalation_reason": {
            "type": "string",
            "description": "If needs_escalation is true, explain why human intervention is needed",
        },
    },
    "required": ["category", "product", "key_issue", "customer_request", "needs_escalation"],
}

POLICY_DECISION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["accept", "deny"],
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of why the request is accepted or denied per company policy",
        },
        "applicable_policy": {
            "type": "string",
            "description": "The specific policy section that applies",
        },
    },
    "required": ["decision", "reasoning", "applicable_policy"],
}

REPLY_CONFIRMATION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "email_sent": {
            "type": "boolean",
            "description": "Whether the reply email was sent successfully",
        },
        "summary": {
            "type": "string",
            "description": "Brief summary of the reply that was sent",
        },
    },
    "required": ["email_sent", "summary"],
}

SAMPLE_TICKET = """\
Subject: Charged twice for my subscription - URGENT

Hi,

I just noticed that my credit card was charged $49.99 TWICE for my Pro
subscription this month (Oct 3 and Oct 5). I only have one account and
I definitely did not sign up for a second subscription.

I've been a customer for 2 years and this has never happened before.
I need a refund for the duplicate charge ASAP - I'm on a tight budget
this month and that extra $50 really hurts.

Also, while I have your attention - the dashboard has been loading
really slowly for the past week. Sometimes it takes 30+ seconds.
Is there something going on with the servers?

Thanks,
Leonhard Euler
Account: leonhard.euler@email.com
Plan: Pro ($49.99/month)\
"""
