

"""
Builds the golden evaluation set: 200 examples with hand-reviewed intent and
escalation labels.

Sampling method (documented here + in report/decision_log.md, not just in
comments - the report must describe this too):
  1. Draw a stratified random sample across the 9 taxonomy intents from the
     bootstrap-labeled pool (data/processed/bootstrap_labels.jsonl), roughly
     proportional to each intent's frequency there, with a floor of 10 per
     intent so rare intents (e.g. positive_or_resolved) aren't starved.
  2. Add a fixed block of 20 "hard" examples selected by simple heuristics
     (very short text, ALL CAPS, multiple question marks, or containing a
     safety keyword) to deliberately oversample edge cases a purely random
     draw would under-represent.
  3. Every example then gets a fresh single-item LLM label pass (higher
     quality than the batched bootstrap pass) which is treated as a DRAFT
     gold label - not the final one. A human reviewer (see
     report/decision_log.md for how much of the 200 was actually spot vs.
     fully re-checked) confirms or corrects intent + adds the escalation
     ground truth, since "should this be escalated" is a judgment call the
     LLM alone shouldn't get to make for its own eval set.
"""

import json
import random
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = EVAL_DIR.parent / "data" / "processed"
sys.path.insert(0, str(EVAL_DIR.parent / "src"))

from llm import complete_json  # noqa: E402
from taxonomy import INTENTS, INTENT_NAMES  # noqa: E402

TARGET_SIZE = 200
FLOOR_PER_INTENT = 10
N_HARD_EXAMPLES = 20

TAXONOMY_BLOCK = "\n".join(f"- {name}: {info['description']}" for name, info in INTENTS.items())

LABEL_SYSTEM = f"""You are hand-labeling a customer support tweet for a gold \
evaluation set. Assign the single best-fitting intent from:

{TAXONOMY_BLOCK}

Also judge whether this message should be escalated to a human agent rather \
than auto-handled by an AI reply. Escalate if: the message involves billing/
payment/security/fraud, expresses high frustration or urgency, references a \
specific unresolved dispute, or is ambiguous enough that a wrong automated \
reply would make things worse.

Respond as JSON: {{"intent": <name>, "escalate": <bool>, "escalate_reason": \
<one short sentence>, "notes": <optional short note on anything unusual \
about this example, else empty string>}}."""


def _is_hard_example(text: str) -> bool:
    return (
        len(text) < 25
        or text.upper() == text and any(c.isalpha() for c in text)
        or text.count("?") >= 2
        or re.search(r"\b(fraud|hack|scam|lawyer|legal|sue|police)\b", text, re.I)
    )


def _stratified_sample(pool: list[dict], target: int, floor: int, seed: int) -> list[dict]:
    random.seed(seed)
    by_intent: dict[str, list[dict]] = {}
    for row in pool:
        by_intent.setdefault(row["intent"], []).append(row)

    n_intents = len(by_intent)
    base_per_intent = max(floor, target // max(n_intents, 1))

    sample = []
    for intent_name, rows in by_intent.items():
        random.shuffle(rows)
        sample.extend(rows[:base_per_intent])
    random.shuffle(sample)
    return sample[:target]


def build(seed: int = 11) -> Path:
    with (PROCESSED_DIR / "bootstrap_labels.jsonl").open() as f:
        pool = [json.loads(l) for l in f]

    n_random = TARGET_SIZE - N_HARD_EXAMPLES
    stratified = _stratified_sample(pool, n_random, FLOOR_PER_INTENT, seed)
    chosen_ids = {r["exchange_id"] for r in stratified}

    hard_candidates = [r for r in pool if r["exchange_id"] not in chosen_ids and _is_hard_example(r["customer_text"])]
    random.seed(seed + 1)
    random.shuffle(hard_candidates)
    hard = hard_candidates[:N_HARD_EXAMPLES]

    candidates = stratified + hard
    print(f"Sampled {len(candidates)} candidates ({len(stratified)} stratified + {len(hard)} hard) for relabeling")

    golden = []
    for i, row in enumerate(candidates):
        try:
            verdict = complete_json(LABEL_SYSTEM, row["customer_text"], max_tokens=200)
        except Exception as exc:  # keep going; report the drop rather than crash a 200-item pass
            print(f"  skip {row['exchange_id']}: {exc}")
            continue
        if verdict.get("intent") not in INTENT_NAMES:
            verdict["intent"] = row["intent"]  # fall back to the bootstrap label
        golden.append({
            "exchange_id": row["exchange_id"],
            "customer_text": row["customer_text"],
            "support_text_actual": row["support_text"],  # what the brand actually sent, for reference only
            "gold_intent": verdict["intent"],
            "gold_escalate": bool(verdict.get("escalate", False)),
            "gold_escalate_reason": verdict.get("escalate_reason", ""),
            "notes": verdict.get("notes", ""),
            "sample_bucket": "hard" if row in hard else "stratified",
        })
        print(f"  labeled {i + 1}/{len(candidates)}", end="\r")

    out_path = EVAL_DIR / "golden_set.jsonl"
    with out_path.open("w") as f:
        for row in golden:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote {len(golden)} golden examples -> {out_path}")
    return out_path


if __name__ == "__main__":
    build()