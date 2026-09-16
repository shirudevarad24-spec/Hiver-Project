"""
End-to-end CLI: classify -> retrieve+draft -> route, for a subsample of
AmazonHelp customer messages. This is the "runnable pipeline" deliverable.
Usage:
    python3 src/pipeline.py --n 20
    python3 src/pipeline.py --text "my package says delivered but never arrived"
"""
import argparse
import json
import random
from pathlib import Path

from classify import llm_classify
from draft import draft_reply
from retrieve import ExchangeRetriever
from route import decide


PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def run_one(customer_text: str, retriever: ExchangeRetriever) -> dict:
    classification = llm_classify(customer_text)
    draft = draft_reply(customer_text, retriever)
    routing = decide(customer_text, classification["intent"], classification["confidence"], draft)
    return {
        "customer_text": customer_text,
        "intent": classification["intent"],
        "intent_confidence": classification["confidence"],
        "intent_rationale": classification["rationale"],
        "reply": draft["reply"],
        "grounded_on": [g["exchange_id"] for g in draft["grounding"]],
        "action": routing["action"],
        "action_reason": routing["reason"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="number of random sampled exchanges to run")
    parser.add_argument("--text", type=str, default=None, help="run on a single ad-hoc message instead")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if args.text:
        retriever = ExchangeRetriever.from_jsonl()
        print(json.dumps(run_one(args.text, retriever), indent=2))
        return

    with (PROCESSED_DIR / "exchanges.jsonl").open() as f:
        exchanges = [json.loads(l) for l in f]
    if not exchanges:
        raise RuntimeError("No exchanges found. Run src/data_prep.py first.")
    random.seed(args.seed)
    sample = random.sample(exchanges, min(args.n, len(exchanges)))

    # Exclude the demo sample itself from the retrieval corpus so the demo
    # doesn't trivially "retrieve" a message's own paired reply as grounding.
    retriever = ExchangeRetriever.from_jsonl(exclude_ids={e["exchange_id"] for e in sample})
    for e in sample:
        result = run_one(e["customer_text"], retriever)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
