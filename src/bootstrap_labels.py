"""
Weak-supervision labeling pass: asks the LLM to assign one of the 9 taxonomy
intents to a batch of customer messages. Used ONLY to produce training data
for the "simple baseline" classifier (src/classify.py) - never used as the
evaluation ground truth. The golden eval set (eval/build_golden_set.py) is a
separate, smaller, more carefully reviewed sample.

This distinction matters for the "what's misleading about my headline
number" report section: the simple baseline's F1 against the golden set can
look artificially close to the LLM classifier's, because both the baseline's
training labels and the LLM classifier share the same underlying model's
blind spots.
"""

import json
import random
from pathlib import Path

from llm import complete
from taxonomy import INTENTS, INTENT_NAMES

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
BATCH_SIZE = 15

TAXONOMY_BLOCK = "\n".join(
    f"- {name}: {info['description']}" for name, info in INTENTS.items()
)

SYSTEM = f"""You are labeling customer support tweets sent to Amazon with an intent \
tag. Choose exactly one intent from this list for each message:

{TAXONOMY_BLOCK}

Return a JSON array, one object per input message, in the same order, each \
shaped as {{"id": <input id>, "intent": <one of the intent names above>}}. \
Use exactly the intent names given, no others."""


def label_batch(batch: list[dict]) -> dict[str, str]:
    user = json.dumps([{"id": e["exchange_id"], "text": e["customer_text"]} for e in batch])
    text = complete(SYSTEM, user, max_tokens=2000)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return {}
    try:
        items = json.loads(text[start : end + 1])
        return {item["id"]: item["intent"] for item in items if item.get("intent") in INTENT_NAMES}
    except Exception:
        return {}


def bootstrap(n: int, seed: int = 42, exclude_ids: set[str] | None = None) -> Path:
    exclude_ids = exclude_ids or set()
    with (PROCESSED_DIR / "exchanges.jsonl").open() as f:
        exchanges = [json.loads(l) for l in f]
    exchanges = [e for e in exchanges if e["exchange_id"] not in exclude_ids]

    random.seed(seed)
    sample = random.sample(exchanges, min(n, len(exchanges)))

    labeled = []
    for i in range(0, len(sample), BATCH_SIZE):
        batch = sample[i : i + BATCH_SIZE]
        try:
            id_to_intent = label_batch(batch)
        except (json.JSONDecodeError, ValueError):
            continue  # drop the batch rather than guess
        for e in batch:
            intent = id_to_intent.get(e["exchange_id"])
            if intent:
                labeled.append({**e, "intent": intent})
        print(f"labeled {len(labeled)}/{len(sample)}", end="\r")

    out_path = PROCESSED_DIR / "bootstrap_labels.jsonl"
    with out_path.open("w") as f:
        for row in labeled:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(labeled)} bootstrap-labeled examples -> {out_path}")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=1200)
    args = parser.parse_args()
    bootstrap(args.n)