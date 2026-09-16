"""Create a deterministic, LLM-free packet for human annotation.

This intentionally does not infer labels. A reviewer must fill the blank
``gold_*`` fields after reading each customer message. The resulting file can
then be copied to ``eval/golden_set.jsonl`` and consumed by the evaluation
scripts.
"""

import argparse
import json
import random
import re
from pathlib import Path


EVAL_DIR = Path(__file__).resolve().parent
EXCHANGES_PATH = EVAL_DIR.parent / "data" / "processed" / "exchanges.jsonl"
DEFAULT_OUTPUT = EVAL_DIR / "manual_annotation_template.jsonl"


def is_hard_example(text: str) -> bool:
    return (
        len(text) < 25
        or (text.upper() == text and any(char.isalpha() for char in text))
        or text.count("?") >= 2
        or re.search(r"\b(fraud|hack|scam|lawyer|legal|sue|police)\b", text, re.I)
    )


def build_packet(total: int, hard_count: int, seed: int) -> list[dict]:
    if total < 1 or hard_count < 0 or hard_count > total:
        raise ValueError("total must be positive and hard_count must be between 0 and total")

    with EXCHANGES_PATH.open() as file:
        exchanges = [json.loads(line) for line in file]
    if len(exchanges) < total:
        raise ValueError(f"Need {total} exchanges, found {len(exchanges)}")

    rng = random.Random(seed)
    hard_pool = [row for row in exchanges if is_hard_example(row["customer_text"])]
    rng.shuffle(hard_pool)
    hard = hard_pool[:hard_count]
    hard_ids = {row["exchange_id"] for row in hard}

    remaining = [row for row in exchanges if row["exchange_id"] not in hard_ids]
    random_rows = rng.sample(remaining, total - len(hard))
    chosen = [(row, "hard") for row in hard] + [(row, "random") for row in random_rows]
    rng.shuffle(chosen)

    return [
        {
            "exchange_id": row["exchange_id"],
            "customer_text": row["customer_text"],
            "support_text_actual": row["support_text"],
            "gold_intent": "",
            "gold_escalate": "",
            "gold_escalate_reason": "",
            "annotator_notes": "",
            "sample_bucket": bucket,
        }
        for row, bucket in chosen
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=200)
    parser.add_argument("--hard-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    packet = build_packet(args.total, args.hard_count, args.seed)
    with args.out.open("w") as file:
        for row in packet:
            file.write(json.dumps(row) + "\n")
    print(f"Wrote {len(packet)} unlabeled examples to {args.out}")


if __name__ == "__main__":
    main()
