"""
Generates drafted replies for a sample of the golden set, scores them with
the LLM judge, and reports judge/human agreement on the subset covered by
eval/human_scores.jsonl.

Usage: python3 eval/run_judge_eval.py --n 60
"""

import argparse
import json
import random
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent / "src"))

from draft import draft_reply  # noqa: E402
from retrieve import ExchangeRetriever  # noqa: E402
from judge import human_agreement, score_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    with (EVAL_DIR / "golden_set.jsonl").open() as f:
        golden = [json.loads(l) for l in f]
    random.seed(args.seed)
    sample = random.sample(golden, min(args.n, len(golden)))

    # Exclude the full golden set (not just the drafted sample) from the
    # retrieval corpus to avoid self-retrieval leakage - see metrics.py.
    golden_ids = {g["exchange_id"] for g in golden}
    retriever = ExchangeRetriever.from_jsonl(exclude_ids=golden_ids)
    rows = []
    for i, g in enumerate(sample):
        draft = draft_reply(g["customer_text"], retriever)
        rows.append({"exchange_id": g["exchange_id"], "customer_text": g["customer_text"], "reply": draft["reply"]})
        print(f"  drafted {i + 1}/{len(sample)}", end="\r")
    print()

    replies_path = EVAL_DIR / "drafted_replies.jsonl"
    with replies_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    scored = score_dataset(rows)
    judged_path = EVAL_DIR / "judged_replies.jsonl"
    with judged_path.open("w") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(scored)} judged replies -> {judged_path}")

    human_path = EVAL_DIR / "human_scores.jsonl"
    if human_path.exists():
        agreement = human_agreement(judged_path, human_path)
        print("\nJudge/human agreement:")
        for dim, stats in agreement.items():
            print(f"  {dim:12s} pearson_r={stats['pearson_r']:.2f} exact_match={stats['exact_match_rate']:.2f} (n={stats['n']})")
        with (EVAL_DIR / "agreement_results.json").open("w") as f:
            json.dump(agreement, f, indent=2)
    else:
        print(f"\nNo {human_path} found yet - skipping agreement check.")


if __name__ == "__main__":
    main()