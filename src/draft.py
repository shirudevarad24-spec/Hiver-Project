"""
Grounded reply drafting: retrieves similar historically-resolved exchanges
and asks the LLM to draft a reply for the new message using ONLY facts
present in those retrieved replies - the model is explicitly told not to
invent policy details (refund windows, phone numbers, etc.) it wasn't shown.
"""

from llm import complete
from retrieve import ExchangeRetriever

SYSTEM = """You draft a customer support reply for Amazon's Twitter support \
account, in Amazon's typical tone: brief, empathetic, and action-oriented \
(pointing the customer to a next step). You will be given the new customer \
message and 1-3 examples of how Amazon replied to similar past complaints.

Ground your reply in those examples: reuse the same kinds of next steps \
(e.g. "DM us", "contact Customer Service", specific links/phrasing patterns) \
that appear in them. Do NOT invent specific facts you were not given - no \
made-up refund amounts, delivery dates, order numbers, or policies. If the \
examples don't cover the situation well, write a generic but honest \
acknowledgement-and-redirect reply instead of guessing.

Respond with ONLY the reply text, no preamble."""


def draft_reply(customer_text: str, retriever: ExchangeRetriever, k: int = 3) -> dict:
    grounding = retriever.retrieve(customer_text, k=k)
    grounding_block = "\n\n".join(
        f"Past customer message: {g['customer_text']}\nAmazon's reply: {g['support_text']}"
        for g in grounding
    ) or "(no sufficiently similar past example found)"

    user = f"New customer message: {customer_text}\n\nSimilar past examples:\n{grounding_block}"
    reply = complete(SYSTEM, user, max_tokens=300)
    return {
        "reply": reply.strip(),
        "grounding": grounding,
        "grounded": len(grounding) > 0,
    }
