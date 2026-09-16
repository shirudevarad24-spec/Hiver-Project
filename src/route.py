"""
Auto-handle vs. escalate-to-human decision, with a stated reason.

Deliberately a rules-first gate rather than a single LLM call: the rules
are the parts we want to be able to audit and guarantee (never auto-handle
a fraud/security claim), and the LLM only adjudicates the residual "is this
actually resolved by the drafted reply" question. This keeps the riskiest
decisions out of an LLM's hands and keeps the stated reason inspectable.
"""

import re

from llm import complete_json

# Intents that are never auto-handled regardless of classifier/retrieval confidence.
ALWAYS_ESCALATE_INTENTS = {
    "billing_or_payment",
    "account_access_or_security",
    "delivery_not_received_marked_delivered",  # fraud/loss-adjacent, needs investigation not a canned reply
}

# Crude but auditable signals for high-stakes content the rules should always catch.
SAFETY_PATTERNS = re.compile(
    r"\b(fraud|hack(ed)?|scam|lawyer|legal action|sue|police|unauthorized charge|stolen)\b",
    re.I,
)

MIN_CLASSIFIER_CONFIDENCE = 0.6
MIN_RETRIEVAL_SIMILARITY = 0.12

JUDGE_SYSTEM = """You review a drafted customer-support reply and decide if \
it fully resolves the customer's message well enough to send as-is, with no \
human review. Say no if it's generic, dodges the question, or the customer \
sounds highly frustrated/urgent. Respond as JSON: \
{"send_as_is": <bool>, "reason": <one short sentence>}."""


def decide(customer_text: str, intent: str, classifier_confidence: float, draft: dict) -> dict:
    if intent in ALWAYS_ESCALATE_INTENTS:
        return _escalate(f"intent '{intent}' is always escalated (billing/security risk)")

    if SAFETY_PATTERNS.search(customer_text):
        return _escalate("message contains a safety/fraud/legal keyword")

    if classifier_confidence < MIN_CLASSIFIER_CONFIDENCE:
        return _escalate(f"intent classification confidence {classifier_confidence:.2f} below threshold")

    if not draft["grounded"] or max((g["similarity"] for g in draft["grounding"]), default=0) < MIN_RETRIEVAL_SIMILARITY:
        return _escalate("no sufficiently similar historical resolution found to ground the reply")

    verdict = complete_json(
        JUDGE_SYSTEM,
        f"Customer message: {customer_text}\n\nDrafted reply: {draft['reply']}",
        max_tokens=150,
    )
    if not verdict.get("send_as_is", False):
        return _escalate(f"LLM reviewer judged reply insufficient: {verdict.get('reason', 'no reason given')}")

    return {"action": "auto_handle", "reason": "passed all rule checks and LLM review"}


def _escalate(reason: str) -> dict:
    return {"action": "escalate", "reason": reason}