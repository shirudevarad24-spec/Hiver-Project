"""Validate processed data and write factual dataset-summary metadata."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
THREADS_PATH = ROOT_DIR / "data" / "processed" / "threads.jsonl"
EXCHANGES_PATH = ROOT_DIR / "data" / "processed" / "exchanges.jsonl"
DEFAULT_OUTPUT = ROOT_DIR / "report" / "data_summary.json"


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as file:
        return [json.loads(line) for line in file]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    threads = load_jsonl(THREADS_PATH)
    exchanges = load_jsonl(EXCHANGES_PATH)
    required = {"exchange_id", "brand", "customer_text", "support_text"}
    malformed = [row["exchange_id"] for row in exchanges if not required <= row.keys()]
    empty_customer = sum(not row["customer_text"].strip() for row in exchanges)
    empty_support = sum(not row["support_text"].strip() for row in exchanges)
    unique_ids = len({row["exchange_id"] for row in exchanges})
    summary = {
        "threads": len(threads),
        "exchanges": len(exchanges),
        "brand_counts": dict(sorted(Counter(row["brand"] for row in exchanges).items())),
        "unique_exchange_ids": unique_ids,
        "empty_customer_texts": empty_customer,
        "empty_support_texts": empty_support,
        "malformed_exchange_rows": len(malformed),
    }
    if malformed or empty_customer or empty_support or unique_ids != len(exchanges):
        raise ValueError(f"Processed-data validation failed: {summary}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
